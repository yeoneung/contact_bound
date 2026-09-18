"""Residual diagnostics and conditional bounds for exact arithmetic.

Grid contact bounds require a verified L_u and exact grid values. They do not
cover unsampled centers; use certificates.finite_center_bound for that term.
Historical result files used the pre-revision grid implementation.
"""
import math
import torch
from .torus import wrap
from .critic import grad_u
from .contact import window_search


# ---------------------------------------------------------------- AD residual
def ad_residual(problem, u, x):
    """|lam u(x) + H(x, grad u(x))| using autograd (defined a.e.)."""
    val, g = grad_u(u, x)
    with torch.no_grad():
        return (problem.lam * val + problem.H(x, g)).abs()


# ---------------------------------------------------------------- Bellman residual (semi-Lagrangian)
@torch.no_grad()
def bellman_target(problem, u, x, h, n_sub=4):
    """T_h u(x) = min_a [ int_0^h e^{-lam s} ell ds + e^{-lam h} u(X_h) ]  over the finite action set. Returns (T_h u, argmin index)."""
    B, d = x.shape
    K = problem.actions.shape[0]
    xa = x[:, None, :].expand(B, K, d).reshape(-1, d)
    aa = problem.actions[None, :, :].expand(B, K, d).reshape(-1, d)
    c = problem.running_cost(xa, aa, h, n_sub=n_sub)
    xn = problem.step(xa, aa, h)
    q = (c + math.exp(-problem.lam * h) * u(xn)).view(B, K)
    m, i = q.min(1)
    return m, i


@torch.no_grad()
def bellman_residual(problem, u, x, h, n_sub=4):
    """(u(x) - T_h u(x))/h  (signed)."""
    t, _ = bellman_target(problem, u, x, h, n_sub)
    return (u(x) - t) / h


# ---------------------------------------------------------------- contact residual
@torch.no_grad()
def contact_residual(problem, grid_vals, n, centers, eps, sign, L_u=None, radius=None, chunk=None,
                     certify=False):
    """Grid contact residual per center, optionally with a search-error bound.

    sign=+1: r = [F(x_+, u(x_+), p_+)]_+ ;  sign=-1: r = [-F(x_-, u(x_-), p_-)]_+ ,  F = lam z + H(x,p).
    Returns dict with x, p, ux, r_best (residual at the best grid contact),
    and if certify: r_cert (max residual over the Lipschitz-certified near-optimal set + discretization slack),
    n_near (multiplicity of the near-optimal set).
    certify=True requires a true global L_u, includes all relevant Euclidean
    lifts, and bounds contact-location error only. It does not certify center
    coverage, estimated constants, quadrature, or floating-point error.
    """
    dim = problem.dim
    delta = 2.0 / n
    if eps <= 0 or sign not in (-1, 1):
        raise ValueError("eps must be positive and sign must be -1 or 1")
    if certify and (L_u is None or not math.isfinite(L_u) or L_u < 0):
        raise ValueError("certify=True requires a finite nonnegative Lipschitz upper bound L_u")
    if radius is None:
        Lr = L_u if L_u is not None else problem.L_V
        radius = 2.0 * Lr * eps + 4 * delta
        if not certify:
            radius = min(1.0, radius)
    if certify and radius < 2.0 * L_u * eps + delta:
        raise ValueError("Certified window must contain all contacts and their nearest grid points")
    out = {k: [] for k in ["x", "p", "ux", "r_best", "r_cert", "n_near"]}
    from .contact import auto_chunk
    chunk = chunk or auto_chunk(n, dim, radius, budget=(6e6 if certify else 1.5e7), lifted=certify)
    for i in range(0, centers.shape[0], chunk):
        y = centers[i:i + chunk]
        obj, z, uz = window_search(grid_vals, n, dim, y, eps, sign, radius, lifted=certify)
        m, j = obj.max(1)
        ar = torch.arange(z.shape[0], device=z.device)
        x_lift = z[ar, j]
        x = wrap(x_lift)
        p = sign * ((x_lift - y) if certify else wrap(x_lift - y)) / eps
        ux = uz[ar, j]
        F = problem.F(x, ux, p)
        r_best = torch.clamp(sign * F, min=0.0)
        out["x"].append(x); out["p"].append(p); out["ux"].append(ux); out["r_best"].append(r_best)
        if certify:
            # radius is a coordinate half-width. Include rounding to the
            # nearest center node and convert it to a Euclidean radius.
            radius_euclidean = (math.ceil(radius / delta) + 0.5) * delta * math.sqrt(dim)
            L_psi = L_u + radius_euclidean / eps
            slack_obj = L_psi * delta * math.sqrt(dim)
            near = obj >= (m[:, None] - slack_obj)
            # residual over the near-optimal set
            zf = wrap(z.reshape(-1, dim))
            pf = sign * (z - y[:, None, :]).reshape(-1, dim) / eps
            Ff = problem.F(zf, uz.reshape(-1), pf).view(obj.shape)
            rf = torch.clamp(sign * Ff, min=0.0)
            rf = torch.where(near, rf, torch.full_like(rf, -1.0))
            r_near = rf.max(1).values
            # Lipschitz slack of F in (x,p) over half a grid cell: |dx| <= delta*sqrt(d)/2, |dp| <= |dx|/eps
            dx = delta * math.sqrt(dim) / 2
            pmax = radius_euclidean / eps
            slack_F = (problem.lam * L_u + problem.L_ell + problem.L_f * pmax) * dx + problem.M_f * dx / eps
            out["r_cert"].append(r_near + slack_F)
            out["n_near"].append(near.sum(1))
    res = {k: torch.cat(v) for k, v in out.items() if len(v) > 0}
    res["radius"] = radius
    return res


def certificate_bound(problem, R, eps):
    """Population formula, not a certificate for a finite-center residual.

    For finite centers use certificates.finite_center_bound with a verified
    fill-distance upper bound and a residual upper bound at every center.
    """
    return (R + 2 * problem.L_V * (problem.L_ell + 2 * problem.L_f * problem.L_V) * eps) / problem.lam


# ---------------------------------------------------------------- rollout (test-function TD) residual
@torch.no_grad()
def test_function_td(problem, x, ux, p, eps, h, sign, n_sub=8):
    """q_h^phi(x,a) for all actions a, with the quadratic contact test function
    phi(z) = u(x) + p.(z-x) + sign*|z-x|^2/(2 eps).  Returns (B,K)."""
    B, d = x.shape
    K = problem.actions.shape[0]
    xa = x[:, None, :].expand(B, K, d).reshape(-1, d)
    aa = problem.actions[None, :, :].expand(B, K, d).reshape(-1, d)
    c = problem.running_cost(xa, aa, h, n_sub=n_sub).view(B, K)
    xn = problem.step(xa, aa, h).view(B, K, d)
    dz = wrap(xn - x[:, None, :])
    phi_next = ux[:, None] + (p[:, None, :] * dz).sum(-1) + sign * (dz ** 2).sum(-1) / (2 * eps)
    return (c + math.exp(-problem.lam * h) * phi_next - ux[:, None]) / h


@torch.no_grad()
def rollout_residual(problem, x, ux, p, eps, h, sign, n_sub=8):
    """Upper (sign=+1): [ -inf_a q_h^{phi+} ]_+ ;  lower (sign=-1): [ inf_a q_h^{phi-} ]_+ ."""
    q = test_function_td(problem, x, ux, p, eps, h, sign, n_sub)
    qmin = q.min(1).values
    return torch.clamp(-sign * qmin, min=0.0), qmin
