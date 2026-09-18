"""Torus geometry T^d = (R/2Z)^d with coordinates in [-1,1)."""
import itertools
import torch


def wrap(d):
    """Wrap a displacement (or state) into [-1, 1) per coordinate."""
    return torch.remainder(d + 1.0, 2.0) - 1.0


def sqdist(x, y):
    """Squared torus distance between batches (B,d)."""
    return (wrap(x - y) ** 2).sum(-1)


def uniform_grid(n, dim, device):
    """Regular grid with n points per dimension on [-1,1)^dim. Returns (n^dim, dim) and spacing."""
    c = torch.linspace(-1.0, 1.0, n + 1, device=device)[:-1]
    mesh = torch.meshgrid(*([c] * dim), indexing="ij")
    pts = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    return pts, 2.0 / n


def random_points(B, dim, device, generator=None):
    return torch.rand(B, dim, device=device, generator=generator) * 2.0 - 1.0


def periodic_interp(vals, x):
    """Periodic multilinear interpolation. vals: (n,)*d tensor, x: (B,d) in [-1,1)."""
    d = x.shape[1]
    n = vals.shape[0]
    delta = 2.0 / n
    u = (wrap(x) + 1.0) / delta
    i0 = torch.floor(u).long()
    t = (u - i0.to(u.dtype)).clamp(0.0, 1.0)
    out = torch.zeros(x.shape[0], device=x.device, dtype=vals.dtype)
    for corner in itertools.product([0, 1], repeat=d):
        w = torch.ones(x.shape[0], device=x.device, dtype=vals.dtype)
        idx = []
        for k, c in enumerate(corner):
            w = w * (t[:, k] if c else (1.0 - t[:, k]))
            idx.append(torch.remainder(i0[:, k] + c, n))
        out = out + w * vals[tuple(idx)]
    return out
