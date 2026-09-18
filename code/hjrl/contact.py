"""Contact-point search for sup/inf-convolutions on the torus.

Upper contact (sign=+1):  C^+_eps(u;y) = argmax_z  u(z) - |z-y|^2/(2 eps),   p_+ = (x_+ - y)/eps
Lower contact (sign=-1):  C^-_eps(u;y) = argmin_z  u(z) + |z-y|^2/(2 eps),   p_- = (y - x_-)/eps
Both are argmax_z  sign*u(z) - |z-y|^2/(2 eps)  with  p = sign*(x - y)/eps.

Two solvers:
  * window_search: exhaustive search on a regular grid restricted to a torus window around y
    (reference solver; returns the whole objective window so that near-optimal sets can be inspected).
  * local_search: gradient ascent from y on the differentiable critic (the "practical", inexact solver).
"""
import itertools
import math
import torch
from .torus import wrap


def _offsets(w, dim, device):
    r = torch.arange(-w, w + 1, device=device)
    mesh = torch.meshgrid(*([r] * dim), indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)   # (W^d, d)


@torch.no_grad()
def window_search(grid_vals, n, dim, centers, eps, sign, radius, lifted=False):
    """Exhaustive search on the grid within a torus window of half-width `radius` around each center.

    grid_vals: (n^dim,) values of u on uniform_grid(n,dim) (ij order).  centers: (B,dim).
    Returns obj (B,W), z (B,W,dim): objective sign*u(z) - |z-y|^2/(2eps) at all window points.
    With lifted=True, retain Euclidean grid copies and their displacements,
    including both slopes at a torus cut locus; only values use periodic indices.
    """
    device = centers.device
    delta = 2.0 / n
    w = int(math.ceil(radius / delta))
    if not lifted:
        w = min(w, n // 2)
    off = _offsets(w, dim, device)                                   # (W,d)
    iy = torch.round((centers + 1.0) / delta).long()                 # (B,d)
    lifted_idx = iy[:, None, :] + off[None, :, :]
    idx = torch.remainder(lifted_idx, n)                            # (B,W,d)
    flat = torch.zeros(idx.shape[:2], dtype=torch.long, device=device)
    for k in range(dim):
        flat = flat * n + idx[..., k]
    z = -1.0 + delta * (lifted_idx if lifted else idx).to(centers.dtype)
    uz = grid_vals[flat]                                              # (B,W)
    dz = z - centers[:, None, :]
    if not lifted:
        dz = wrap(dz)
    obj = sign * uz - (dz ** 2).sum(-1) / (2 * eps)
    return obj, z, uz


def auto_chunk(n, dim, radius, budget=1.5e7, lifted=False):
    """Number of centers per chunk so that the window tensor stays around `budget` entries."""
    delta = 2.0 / n
    w = int(math.ceil(radius / delta))
    if not lifted:
        w = min(w, n // 2)
    W = (2 * w + 1) ** dim
    return max(1, int(budget // W))


@torch.no_grad()
def grid_contact(grid_vals, n, dim, centers, eps, sign, radius, chunk=None):
    """Best contact point per center (exhaustive within window). Returns x (B,d), p (B,d), u(x) (B,), obj (B,)."""
    xs, ps, us, objs = [], [], [], []
    chunk = chunk or auto_chunk(n, dim, radius)
    for i in range(0, centers.shape[0], chunk):
        y = centers[i:i + chunk]
        obj, z, uz = window_search(grid_vals, n, dim, y, eps, sign, radius)
        m, j = obj.max(1)
        x = z[torch.arange(z.shape[0], device=z.device), j]
        xs.append(x)
        ps.append(sign * wrap(x - y) / eps)
        us.append(uz[torch.arange(z.shape[0], device=z.device), j])
        objs.append(m)
    return torch.cat(xs), torch.cat(ps), torch.cat(us), torch.cat(objs)


def local_search(u, centers, eps, sign, steps=100, step_frac=0.5, n_restart=1, noise=0.0, generator=None):
    """Inexact gradient-ascent contact solver on a differentiable critic (returns one point per center).

    Iterates z <- z + eta*(sign*grad u(z) - wrap(z-y)/eps), eta = step_frac*eps (stable for |D^2 u| < 1/eps).
    With n_restart>1, random perturbed initializations of scale `noise` are tried and the best objective is kept.
    """
    B, d = centers.shape
    best_obj = torch.full((B,), -float("inf"), device=centers.device)
    best_z = centers.clone()
    eta = step_frac * eps
    for r in range(n_restart):
        z = centers.clone()
        if r > 0 and noise > 0:
            z = wrap(z + noise * torch.randn(B, d, device=z.device, generator=generator))
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            with torch.enable_grad():
                val = u(z)
                g, = torch.autograd.grad(val.sum(), z)
            z = wrap(z + eta * (sign * g - wrap(z - centers) / eps)).detach()
        with torch.no_grad():
            obj = sign * u(z) - (wrap(z - centers) ** 2).sum(-1) / (2 * eps)
            better = obj > best_obj
            best_obj = torch.where(better, obj, best_obj)
            best_z = torch.where(better[:, None], z, best_z)
    with torch.no_grad():
        p = sign * wrap(best_z - centers) / eps
        ux = u(best_z)
    return best_z, p, ux, best_obj
