"""Reference solvers on regular grids (GPU): semi-Lagrangian value iteration, 1D spurious a.e. solutions."""
import math
import torch
from .torus import uniform_grid, periodic_interp, wrap


@torch.no_grad()
def sl_value_iteration(problem, n, h, iters=20000, tol=1e-7, n_sub=4, verbose=False, u0=None):
    """Semi-Lagrangian VI on the n^dim grid:  u <- min_a [ int_0^h e^{-lam s} ell ds + e^{-lam h} I[u](x + h f(x,a)) ].
    Monotone + consistent  =>  converges to the viscosity solution as h, delta -> 0 (Barles-Souganidis)."""
    dim, dev = problem.dim, problem.device
    pts, delta = uniform_grid(n, dim, dev)
    G = pts.shape[0]
    K = problem.actions.shape[0]
    xa = pts[:, None, :].expand(G, K, dim).reshape(-1, dim)
    aa = problem.actions[None, :, :].expand(G, K, dim).reshape(-1, dim)
    cost = problem.running_cost(xa, aa, h, n_sub=n_sub).view(G, K)
    xn = problem.step(xa, aa, h)
    gamma = math.exp(-problem.lam * h)
    u = torch.zeros(G, device=dev) if u0 is None else u0.clone().reshape(-1)
    shape = [n] * dim
    for it in range(iters):
        un = periodic_interp(u.view(*shape), xn).view(G, K)
        u_new = (cost + gamma * un).min(1).values
        diff = (u_new - u).abs().max().item()
        u = u_new
        if diff < tol:
            break
    if verbose:
        print(f"SL-VI n={n} h={h} iters={it+1} last diff={diff:.2e}")
    return pts, u, delta


@torch.no_grad()
def spurious_1d(problem, n, switches, w0_lo=-5.0, w0_hi=5.0, n_bisect=60):
    """Construct a periodic a.e. solution of  lam u + |u'| = g  in 1D by branch switching (RK4 on the grid).

    switches: list of (x_switch, sign) pairs sorted by x in (-1,1): the branch sign s in {+1,-1} means u' = s*(g - lam u)
    from that x onwards; the first sign applies from x=-1. Periodicity u(1)=u(-1) is enforced by bisection on u(-1)
    (u(1;w0) is monotone increasing in w0 because both branches are ODEs with Lipschitz right-hand sides).
    Returns grid points and values (n,).  Kinks appear where the sign switches (+ -> -: concave/peak, - -> +: convex/valley)
    and at x=+-1 if the closing slopes disagree.
    """
    import numpy as np
    dev = problem.device
    pts, delta = uniform_grid(n, 1, dev)
    xs = pts[:, 0]
    signs = torch.full((n,), float(switches[0][1]), device=dev)
    for xsw, s in switches[1:]:
        signs[xs >= xsw] = float(s)
    # g on nodes and midpoints (2n+1 points), evaluated once on the device, integrated on CPU with numpy
    xh = (-1.0 + (delta / 2) * torch.arange(2 * n + 1, device=dev, dtype=torch.float64)).float()
    gh = problem.g(xh[:, None]).double().cpu().numpy()
    sg = signs.double().cpu().numpy()
    lam = problem.lam

    def rhs(gi, u, s):
        return s * max(gi - lam * u, 0.0)

    def integrate(w0):
        u = np.empty(n + 1)
        u[0] = w0
        for i in range(n):
            s = sg[i]
            g0, gm, g1 = gh[2 * i], gh[2 * i + 1], gh[2 * i + 2]
            k1 = rhs(g0, u[i], s)
            k2 = rhs(gm, u[i] + delta / 2 * k1, s)
            k3 = rhs(gm, u[i] + delta / 2 * k2, s)
            k4 = rhs(g1, u[i] + delta * k3, s)
            u[i + 1] = u[i] + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

    def phi(w0):
        u = integrate(w0)
        return u[-1] - u[0]

    lo, hi = w0_lo, w0_hi
    plo, phi_hi = phi(lo), phi(hi)
    assert plo * phi_hi <= 0, f"no sign change of periodicity defect on [{lo},{hi}]: {plo:.3g},{phi_hi:.3g}"
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        pm = phi(mid)
        if pm * plo > 0:
            lo, plo = mid, pm
        else:
            hi = mid
    u = integrate(0.5 * (lo + hi))
    clamp_frac = float(np.mean((gh[0:2 * n:2] - lam * u[:-1]) < 0))
    ut = torch.tensor(u[:-1], device=dev, dtype=torch.float32)
    return pts, ut, float(abs(u[-1] - u[0])), clamp_frac


@torch.no_grad()
def spurious_family_1d(problem, n, c):
    """Exact a.e. solutions of  lam u + |u'| = g  for Ridge1D:  W_c = V - c*w,  c >= 0,
    w(x) = exp(-lam(1+x)) for x<0,  exp(-lam(1-x)) for x>=0  (w solves the linearized branch ODEs).
    W_c keeps the correct peak at 0 but acquires a spurious valley (convex kink, slopes -+ lam c) at x=+-1,
    so it is not a viscosity supersolution there; ||W_c - V||_inf = c and the lower contact residual is lam*c."""
    pts, _ = uniform_grid(n, 1, problem.device)
    x = pts[:, 0]
    w = torch.where(x < 0, torch.exp(-problem.lam * (1 + x)), torch.exp(-problem.lam * (1 - x)))
    return pts, problem.V(pts) - c * w
