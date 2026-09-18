"""Critic parametrizations on the torus."""
import math
import torch
import torch.nn as nn
from .torus import periodic_interp, uniform_grid


class FourierMLP(nn.Module):
    """Periodic MLP: x -> [sin(pi k x_i), cos(pi k x_i)]_{k<=n_freq} -> MLP -> scalar."""

    def __init__(self, dim, n_freq=6, width=128, depth=3, act="silu", out_scale=1.0, out_dim=1):
        super().__init__()
        self.dim, self.n_freq, self.out_dim = dim, n_freq, out_dim
        self.register_buffer("freqs", torch.arange(1, n_freq + 1).float() * math.pi)
        layers = []
        d_in = 2 * dim * n_freq
        for i in range(depth):
            layers.append(nn.Linear(d_in if i == 0 else width, width))
            layers.append(nn.SiLU() if act == "silu" else nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_scale = out_scale

    def features(self, x):
        z = x[..., :, None] * self.freqs                       # (B,d,F)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1).flatten(-2)

    def forward(self, x):
        out = self.out_scale * self.net(self.features(x))
        return out[..., 0] if self.out_dim == 1 else out


class QContinuous(nn.Module):
    """Q(x,a) for continuous actions: Fourier features of x concatenated with a, then an MLP."""

    def __init__(self, dim, act_dim, n_freq=12, width=256, depth=3):
        super().__init__()
        self.feat = FourierMLP(dim, n_freq, width, depth)   # reuse the feature map only
        d_in = 2 * dim * n_freq + act_dim
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(d_in if i == 0 else width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, a):
        return self.net(torch.cat([self.feat.features(x), a], -1))[..., 0]


class Actor(nn.Module):
    """Deterministic actor mu(x) in the unit disk: a = v/(1+|v|) with v = MLP(Fourier(x)) (smooth in x)."""

    def __init__(self, dim, act_dim, n_freq=12, width=128, depth=3):
        super().__init__()
        self.body = FourierMLP(dim, n_freq, width, depth, out_dim=act_dim)

    def forward(self, x):
        v = self.body(x)
        return v / (1.0 + v.norm(dim=-1, keepdim=True))


class DuelingMinQ(nn.Module):
    """Min-normalized dueling Q-network for cost minimization:
        Q(x,a) = V(x) + A(x,a),   A(x,a) = softplus(B(x,a)) - min_a' softplus(B(x,a')) >= 0,   min_a A(x,a) = 0.
    Hence min_a Q(x,a) = V(x) exactly: a fitted Q-iteration target  c + gamma min_a' Q_t(x',a')  reduces to  c + gamma V_t(x'),
    a single head, and the K advantage heads never enter a min in the backup (no min-backup underestimation bias)."""

    def __init__(self, dim, K, n_freq=12, width=128, depth=3):
        super().__init__()
        self.K = K
        self.v = FourierMLP(dim, n_freq, width, depth)
        self.b = FourierMLP(dim, n_freq, width, depth, out_dim=K)

    def value(self, x):
        return self.v(x)

    def advantage(self, x):
        sp = nn.functional.softplus(self.b(x))
        return sp - sp.min(-1, keepdim=True).values

    def forward(self, x):
        return self.v(x)[..., None] + self.advantage(x)


class DuelingMeanQ(nn.Module):
    """Mean-normalized dueling Q-network (Wang et al., 2016):  Q(x,a) = V(x) + B(x,a) - mean_a' B(x,a').
    min_a Q(x,a) = V(x) + min_a B - mean B != V(x), so the fitted Q-iteration target still minimizes over K heads."""

    def __init__(self, dim, K, n_freq=12, width=128, depth=3):
        super().__init__()
        self.K = K
        self.v = FourierMLP(dim, n_freq, width, depth)
        self.b = FourierMLP(dim, n_freq, width, depth, out_dim=K)

    def value(self, x):
        return self.forward(x).min(-1).values

    def advantage(self, x):
        b = self.b(x)
        return b - b.mean(-1, keepdim=True)

    def forward(self, x):
        return self.v(x)[..., None] + self.advantage(x)


class QMinValue(nn.Module):
    """V_Q(x) = min_a Q(x,a) for a Q-network with K outputs (used to extract V-based policies from a learned Q)."""

    def __init__(self, qnet):
        super().__init__()
        self.qnet = qnet

    def forward(self, x):
        return self.qnet(x).min(-1).values


class RouteMinCritic(nn.Module):
    """u = min_k u_k over given branch critics (e.g. values of route-constrained problems); same interface as MinBranchCritic."""

    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.K = len(heads)

    def branches(self, x):
        return torch.stack([h(x) for h in self.heads], dim=-1)

    def forward(self, x):
        return self.branches(x).min(-1).values


class MinBranchCritic(nn.Module):
    """u(x) = min_k u_k(x) with K smooth periodic branches (hard min in eval, optional soft-min in training)."""

    def __init__(self, dim, K=4, n_freq=6, width=96, depth=3, softmin_tau=0.0):
        super().__init__()
        self.K = K
        self.heads = nn.ModuleList([FourierMLP(dim, n_freq, width, depth) for _ in range(K)])
        self.softmin_tau = softmin_tau

    def branches(self, x):
        return torch.stack([h(x) for h in self.heads], dim=-1)   # (B,K)

    def forward(self, x):
        b = self.branches(x)
        if self.training and self.softmin_tau > 0:
            return -self.softmin_tau * torch.logsumexp(-b / self.softmin_tau, dim=-1)
        return b.min(-1).values


class GridFunction(nn.Module):
    """Periodic grid function with multilinear interpolation (reference solutions, spurious a.e. solutions)."""

    def __init__(self, vals):
        super().__init__()
        self.vals = nn.Parameter(vals.clone(), requires_grad=False)
        self.n = vals.shape[0]
        self.dim = vals.dim()

    def forward(self, x):
        return periodic_interp(self.vals, x)


@torch.no_grad()
def eval_on_grid(u, n, dim, device, chunk=1 << 18):
    """Evaluate a critic on the regular n^dim grid. Returns (pts (G,d), vals (G,), spacing)."""
    pts, delta = uniform_grid(n, dim, device)
    out = torch.empty(pts.shape[0], device=device)
    for i in range(0, pts.shape[0], chunk):
        out[i:i + chunk] = u(pts[i:i + chunk])
    return pts, out, delta


def grad_u(u, x, create_graph=False):
    """Autograd gradient of a scalar critic at x (B,d) -> (B,d)."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        val = u(x)
        g, = torch.autograd.grad(val.sum(), x, create_graph=create_graph)
    return val.detach() if not create_graph else val, g


def lipschitz_estimate(u, n, dim, device):
    """max |grad u| on a grid via finite differences of the grid evaluation (periodic)."""
    pts, vals, delta = eval_on_grid(u, n, dim, device)
    v = vals.view(*([n] * dim))
    g2 = torch.zeros_like(v)
    for k in range(dim):
        g2 = g2 + ((v.roll(-1, k) - v.roll(1, k)) / (2 * delta)) ** 2
    return float(g2.sqrt().max())
