"""Deterministic continuous-time control problems on the torus.

    dX = f(X,a) dt,   V(x) = inf_a int_0^inf e^{-lam t} ell(X_t,a_t) dt
    HJB (viscosity):  lam V + H(x, DV) = 0,   H(x,p) = sup_a { -ell(x,a) - f(x,a).p }

All problems use a finite action set (K,d); H is the exact max over that set.
Index 0 of the action set is always the zero action (used for tie-breaking studies).
"""
import math
import torch
from .torus import wrap, periodic_interp


class Problem:
    dim: int = 1
    lam: float = 1.0
    name: str = "base"
    L_V: float = 1.0      # Lipschitz constant of V (torus, Euclidean lift)
    L_ell: float = 1.0    # Lipschitz constant of ell in x (uniform in a)
    L_f: float = 0.0      # Lipschitz constant of f in x
    M_f: float = 1.0      # sup |f|
    has_exact_V: bool = False

    def __init__(self, device, lam=1.0):
        self.device = device
        self.lam = lam
        self.actions = self._actions().to(device)   # (K,d)

    def _actions(self):
        raise NotImplementedError

    # ---- dynamics / cost (batched: x (B,d), a (B,d)) ----
    def f(self, x, a):
        return a

    def ell(self, x, a):
        raise NotImplementedError

    def step(self, x, a, h):
        """Exact flow for f=a (override for nonlinear f)."""
        return wrap(x + h * a)

    def running_cost(self, x, a, h, n_sub=8):
        """int_0^h e^{-lam s} ell(X_s,a) ds along the held-action trajectory (Simpson on exact flow)."""
        B, d = x.shape
        s = torch.linspace(0.0, h, 2 * n_sub + 1, device=x.device)
        xs = wrap(x[:, None, :] + s[None, :, None] * a[:, None, :])
        aa = a[:, None, :].expand_as(xs)
        c = self.ell(xs.reshape(-1, d), aa.reshape(-1, d)).view(B, -1)
        w = torch.ones_like(s)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w = w * (h / (2 * n_sub)) / 3.0
        return (c * (torch.exp(-self.lam * s) * w)[None, :]).sum(1)

    # ---- Hamiltonian over the finite action set ----
    def _ham_values(self, x, p):
        B, d = x.shape
        K = self.actions.shape[0]
        xa = x[:, None, :].expand(B, K, d).reshape(-1, d)
        aa = self.actions[None, :, :].expand(B, K, d).reshape(-1, d)
        pa = p[:, None, :].expand(B, K, d).reshape(-1, d)
        return (-self.ell(xa, aa) - (self.f(xa, aa) * pa).sum(-1)).view(B, K)

    def H(self, x, p):
        return self._ham_values(x, p).max(1).values

    def H_argmax(self, x, p):
        m, i = self._ham_values(x, p).max(1)
        return m, i

    def F(self, x, z, p):
        return self.lam * z + self.H(x, p)

    def V(self, x):
        raise NotImplementedError


class Ridge1D(Problem):
    """1D torus, f=a in {0,-1,1}, ell=g(x). Manufactured V = 1-|sin(pi x/2)|: one concave kink (peak) at 0."""
    dim = 1
    name = "ridge1d"
    has_exact_V = True

    def __init__(self, device, lam=1.0):
        super().__init__(device, lam)
        self.L_V = math.pi / 2
        self.L_ell = lam * math.pi / 2 + math.pi ** 2 / 4
        self.L_f = 0.0
        self.M_f = 1.0

    def _actions(self):
        return torch.tensor([[0.0], [-1.0], [1.0]])

    def V(self, x):
        return 1.0 - torch.abs(torch.sin(math.pi * x[:, 0] / 2))

    def g(self, x):
        return self.lam * self.V(x) + (math.pi / 2) * torch.abs(torch.cos(math.pi * x[:, 0] / 2))

    def ell(self, x, a):
        return self.g(x)


class Linf2D(Problem):
    """2D torus, f=a in {0,(+-1,0),(0,+-1)}, ell=g(x).
    V = 1 - max(|sin(pi x1/2)|,|sin(pi x2/2)|) = min of 4 smooth branches, each with a fixed optimal action.
    Kinks: ridge lines x1=0, x2=0 and the diagonals |sin x1|=|sin x2|; 4-fold corner at the origin."""
    dim = 2
    name = "linf2d"
    has_exact_V = True

    def __init__(self, device, lam=1.0):
        super().__init__(device, lam)
        self.L_V = math.pi / 2
        self.L_ell = lam * math.pi / 2 + math.pi ** 2 / 4
        self.L_f = 0.0
        self.M_f = 1.0

    def _actions(self):
        return torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    def branches(self, x):
        s1 = torch.sin(math.pi * x[:, 0] / 2)
        s2 = torch.sin(math.pi * x[:, 1] / 2)
        return torch.stack([1 - s1, 1 + s1, 1 - s2, 1 + s2], dim=1)   # branch k <-> action index k+1

    def V(self, x):
        return self.branches(x).min(1).values

    def g(self, x):
        c1 = torch.abs(torch.cos(math.pi * x[:, 0] / 2))
        c2 = torch.abs(torch.cos(math.pi * x[:, 1] / 2))
        return self.lam * self.V(x) + (math.pi / 2) * torch.minimum(c1, c2)

    def ell(self, x, a):
        return self.g(x)


class LinfND(Problem):
    """d-dim torus, f=a in {0, +-e_i}, ell=g(x).  V = 1 - max_i |sin(pi x_i/2)| = min of 2d smooth branches,
    g = lam V + (pi/2) min_i |cos(pi x_i/2)|,  H = -g + |p|_inf.  Ridges where the two largest |sin| tie; 2d-fold corner at 0."""
    name = "linfnd"
    has_exact_V = True

    def __init__(self, device, dim=4, lam=1.0):
        self.dim = dim
        super().__init__(device, lam)
        self.L_V = math.pi / 2
        self.L_ell = lam * math.pi / 2 + math.pi ** 2 / 4
        self.L_f = 0.0
        self.M_f = 1.0

    def _actions(self):
        eye = torch.eye(self.dim)
        return torch.cat([torch.zeros(1, self.dim), eye, -eye], 0)

    def branches(self, x):
        s = torch.sin(math.pi * x / 2)
        return torch.cat([1 - s, 1 + s], dim=1)           # branch i <-> +e_i, branch d+i <-> -e_i

    def V(self, x):
        return 1.0 - torch.abs(torch.sin(math.pi * x / 2)).max(1).values

    def g(self, x):
        return self.lam * self.V(x) + (math.pi / 2) * torch.abs(torch.cos(math.pi * x / 2)).min(1).values

    def ell(self, x, a):
        return self.g(x)


class Obstacle2D(Problem):
    """2D torus navigation: walls (high running cost) at x1=0 and x1=+-1 for |x2|<=wall_half, target disk at x_target.
    ell(x,a) = (1 + c_obs*obs(x)) * (1 - tgt(x)),  f=a, |a|<=1 on K directions + zero.
    The reference V is computed numerically (semi-Lagrangian value iteration on a fine grid)."""
    dim = 2
    name = "obstacle2d"
    has_exact_V = False

    def __init__(self, device, lam=0.5, K=16, c_obs=10.0, wall_half=0.5, wall_thick=0.08, ramp=0.03,
                 x_target=(0.3, 0.0), r_target=0.1, block=None):
        """block in {None, 'top', 'bottom'}: additionally wall off the top (x2 >= 0.5) or bottom (x2 <= -0.5) half of the
        passage band, leaving a single route (used for route-aligned branch critics)."""
        self.K = K
        super().__init__(device, lam)
        self.c_obs, self.wall_half, self.wall_thick, self.ramp = c_obs, wall_half, wall_thick, ramp
        self.x_target = torch.tensor(x_target, device=device)
        self.r_target = r_target
        self.block = block
        self.L_f, self.M_f = 0.0, 1.0
        self.L_ell = (1 + c_obs) / (4 * ramp)   # sigmoid ramp slope bound
        self.L_V = None
        self._Vgrid = None

    def _actions(self):
        th = torch.arange(self.K) * (2 * math.pi / self.K)
        return torch.cat([torch.zeros(1, 2), torch.stack([torch.cos(th), torch.sin(th)], 1)], 0)

    def obs(self, x):
        d1 = torch.minimum(torch.abs(wrap(x[:, :1]))[:, 0], torch.abs(wrap(x[:, :1] - 1.0))[:, 0])
        inside_x = torch.sigmoid((self.wall_thick / 2 - d1) / self.ramp)
        x2 = wrap(x[:, 1:2])[:, 0]
        inside_y = torch.sigmoid((self.wall_half - torch.abs(x2)) / self.ramp)
        if self.block == "top":
            inside_y = torch.maximum(inside_y, torch.sigmoid((x2 - self.wall_half) / self.ramp))
        elif self.block == "bottom":
            inside_y = torch.maximum(inside_y, torch.sigmoid((-x2 - self.wall_half) / self.ramp))
        return inside_x * inside_y

    def tgt(self, x):
        d = torch.sqrt((wrap(x - self.x_target[None]) ** 2).sum(-1) + 1e-12)
        return torch.sigmoid((self.r_target - d) / self.ramp)

    def ell(self, x, a):
        return (1.0 + self.c_obs * self.obs(x)) * (1.0 - self.tgt(x))

    def set_reference(self, Vgrid):
        self._Vgrid = Vgrid
        n = Vgrid.shape[0]
        dV = torch.stack([(Vgrid.roll(-1, k) - Vgrid.roll(1, k)) / (2 * 2.0 / n) for k in range(2)], 0)
        self.L_V = float(dV.pow(2).sum(0).sqrt().max())

    def V(self, x):
        assert self._Vgrid is not None, "call set_reference(Vgrid) first"
        return periodic_interp(self._Vgrid, x)


class Pendulum(Problem):
    """Torque-limited pendulum swing-up (gym constants: g=10, m=l=1, |u|<=2, |omega|<=8), embedded in the torus:
    x = (theta/pi, omega/omega_max) in [-1,1)^2; theta=0 is upright.  theta wraps, omega is clamped and carries a cost
    wall near |omega| = omega_max so that closed loops never reach the clamp.
    ell = (theta^2 + 0.1 omega^2 + 0.001 u^2)/16 + wall,  actions u in {-2, 0, 2}, RK4 flow.  Reference V by SL-VI."""
    dim = 2
    name = "pendulum"
    has_exact_V = False

    def __init__(self, device, lam=0.5, g=10.0, u_max=2.0, omega_max=8.0, c_wall=20.0, wall_start=0.9, ramp=0.02, n_sub=4):
        self.g, self.u_max, self.omega_max = g, u_max, omega_max
        self.c_wall, self.wall_start, self.ramp, self.n_sub = c_wall, wall_start, ramp, n_sub
        super().__init__(device, lam)
        self.M_f = math.sqrt((omega_max / math.pi) ** 2 + ((g + u_max) / omega_max) ** 2)
        self.L_f = max(omega_max / math.pi, g * math.pi / omega_max)
        self.L_ell = (2 * math.pi ** 2 + 0.2 * omega_max ** 2) / 16 + c_wall / (4 * ramp)
        self.L_V = None
        self._Vgrid = None

    def _actions(self):
        return torch.tensor([[0.0], [-self.u_max], [self.u_max]])

    def f(self, x, a):
        th = math.pi * x[:, 0]
        om = self.omega_max * x[:, 1]
        dth = om / math.pi
        dom = (self.g * torch.sin(th) + a[:, 0]) / self.omega_max
        return torch.stack([dth, dom], 1)

    def wall(self, x):
        return self.c_wall * torch.sigmoid((torch.abs(x[:, 1]) - self.wall_start) / self.ramp)

    def ell(self, x, a):
        th = math.pi * wrap(x[:, :1])[:, 0]
        om = self.omega_max * x[:, 1]
        return (th ** 2 + 0.1 * om ** 2 + 0.001 * a[:, 0] ** 2) / 16.0 + self.wall(x)

    def _project(self, x):
        return torch.stack([wrap(x[:, 0]), x[:, 1].clamp(-0.999, 0.999)], 1)

    def _flow(self, x, a, h):
        """RK4 with n_sub substeps; returns (x_h, discounted running cost) along the held-action trajectory."""
        dt = h / self.n_sub
        cost = torch.zeros(x.shape[0], device=x.device)
        s = 0.0
        for _ in range(self.n_sub):
            k1 = self.f(x, a)
            k2 = self.f(self._project(x + dt / 2 * k1), a)
            k3 = self.f(self._project(x + dt / 2 * k2), a)
            k4 = self.f(self._project(x + dt * k3), a)
            xm = self._project(x + dt / 2 * k1)
            xn = self._project(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            # Simpson on the substep with discount at the three nodes
            cost = cost + dt / 6 * (math.exp(-self.lam * s) * self.ell(x, a) + 4 * math.exp(-self.lam * (s + dt / 2)) * self.ell(xm, a)
                                    + math.exp(-self.lam * (s + dt)) * self.ell(xn, a))
            x = xn
            s += dt
        return x, cost

    def step(self, x, a, h):
        return self._flow(x, a, h)[0]

    def running_cost(self, x, a, h, n_sub=None):
        return self._flow(x, a, h)[1]

    def set_reference(self, Vgrid):
        self._Vgrid = Vgrid
        n = Vgrid.shape[0]
        dV = torch.stack([(Vgrid.roll(-1, k) - Vgrid.roll(1, k)) / (2 * 2.0 / n) for k in range(2)], 0)
        self.L_V = float(dV.pow(2).sum(0).sqrt().max())

    def V(self, x):
        assert self._Vgrid is not None, "call set_reference(Vgrid) first"
        return periodic_interp(self._Vgrid, self._project(x))


def make_problem(name, device, **kw):
    return {"ridge1d": Ridge1D, "linf2d": Linf2D, "obstacle2d": Obstacle2D, "linfnd": LinfND, "pendulum": Pendulum}[name](device, **kw)
