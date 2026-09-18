"""Trajectory Bellman-residual bound (Theorem): terms computed along closed-loop sample-and-hold rollouts."""
import math
import torch
from .residuals import bellman_target


@torch.no_grad()
def rollout_terms(P, u, policy, x0, h, T, V_ref):
    """Run the sample-and-hold policy; return gap J^pi - V, the trajectory term sum_k gamma^k [T_h u - u]_+(x_k),
    the regret term sum_k gamma^k eta_k and the maximal single-step regret."""
    dev = x0.device
    gam = math.exp(-P.lam * h)
    x = x0.clone()
    B = x.shape[0]
    J = torch.zeros(B, device=dev); traj = torch.zeros(B, device=dev); eta_sum = torch.zeros(B, device=dev)
    eta_max = torch.zeros(B, device=dev)
    st = None
    n = int(round(T / h))
    for k in range(n):
        idx, st = policy(x, st)
        a = P.actions[idx]
        c = P.running_cost(x, a, h)
        xn = P.step(x, a, h)
        Tu, _ = bellman_target(P, u, x, h)
        ux = u(x)
        eta = torch.clamp(c + gam * u(xn) - Tu, min=0.0)
        J += gam ** k * c
        traj += gam ** k * torch.clamp(Tu - ux, min=0.0)
        eta_sum += gam ** k * eta
        eta_max = torch.maximum(eta_max, eta)
        x = xn
    J += gam ** n * V_ref(x)
    return {"gap": J - V_ref(x0), "traj": traj, "eta": eta_sum, "eta_max": eta_max}


@torch.no_grad()
def rollout_terms_branch(P, critic, policy, x0, h, T, V_ref, eps_list=(0.05, 0.2)):
    """Test-function versions of the trajectory term for a min-of-branches critic u = min_k u_k (Theorem, contact version).

    Along the closed loop, at x_k with active branch k*=argmin_k u_k(x_k):
      branch term : sum_k gamma^k [T_h u_{k*} - u_{k*}]_+(x_k)          (the active branch touches u from above)
      quad term   : sum_k gamma^k h [min_a q_h^{phi+}(x_k,a)]_+  with phi+ built from p = grad u_{k*}(x_k) and scale eps
    Both dominate the trajectory term sum_k gamma^k [T_h u - u]_+(x_k) pointwise.  Returns all sums and the gap."""
    from .critic import grad_u
    from .residuals import test_function_td
    dev = x0.device
    gam = math.exp(-P.lam * h)
    x = x0.clone()
    B = x.shape[0]
    J = torch.zeros(B, device=dev); traj = torch.zeros(B, device=dev); branch = torch.zeros(B, device=dev)
    quad = {e: torch.zeros(B, device=dev) for e in eps_list}
    st = None
    n = int(round(T / h))
    heads = list(critic.heads)
    for k in range(n):
        idx, st = policy(x, st)
        a = P.actions[idx]
        c = P.running_cost(x, a, h)
        xn = P.step(x, a, h)
        Tu, _ = bellman_target(P, critic, x, h)
        ux = critic(x)
        traj += gam ** k * torch.clamp(Tu - ux, min=0.0)
        # active branch and its residual / gradient
        b = critic.branches(x)
        kstar = b.argmin(1)
        Tk = torch.zeros(B, device=dev); uk = torch.zeros(B, device=dev); pk = torch.zeros(B, x.shape[1], device=dev)
        for j, head in enumerate(heads):
            m = kstar == j
            if m.any():
                Tj, _ = bellman_target(P, head, x[m], h)
                Tk[m] = Tj
                vj, gj = grad_u(head, x[m])
                uk[m] = vj; pk[m] = gj
        branch += gam ** k * torch.clamp(Tk - uk, min=0.0)
        for e in eps_list:
            q = test_function_td(P, x, ux, pk, e, h, +1)
            quad[e] += gam ** k * h * torch.clamp(q.min(1).values, min=0.0)
        J += gam ** k * c
        x = xn
    J += gam ** n * V_ref(x)
    out = {"gap": J - V_ref(x0), "traj": traj, "branch": branch}
    for e in eps_list:
        out[f"quad_eps{e}"] = quad[e]
    return out


@torch.no_grad()
def adversarial_potential(P, u, n, h, iters=20000, tol=1e-7, chunk=1 << 22):
    """Adversarial violation potential on the n^d grid (Theorem):
        delta(x) = v(x) + gamma * max_a delta(x_a^+),   v = [u - T_h u]_+ ,
    computed by value iteration with multilinear interpolation.  u - delta is a subsolution of T_h, so
    u <= V_h + delta and the policy bound holds with delta(x_0) in place of sup v/(1-gamma).
    Returns (pts, delta (G,), v (G,), iterations)."""
    from .torus import uniform_grid, periodic_interp
    dev = P.device
    d = P.dim
    pts, _ = uniform_grid(n, d, dev)
    G = pts.shape[0]
    K = P.actions.shape[0]
    gam = math.exp(-P.lam * h)
    # violation v on the grid
    v = torch.empty(G, device=dev)
    for i in range(0, G, chunk // K):
        x = pts[i:i + chunk // K]
        Tu, _ = bellman_target(P, u, x, h)
        v[i:i + chunk // K] = torch.clamp(u(x) - Tu, min=0.0)
    # next states for all actions
    xa = pts[:, None, :].expand(G, K, d).reshape(-1, d)
    aa = P.actions[None, :, :].expand(G, K, d).reshape(-1, d)
    xn = P.step(xa, aa, h)
    delta = v.clone()
    shape = [n] * d
    for it in range(iters):
        dn = periodic_interp(delta.view(*shape), xn).view(G, K).max(1).values
        new = v + gam * dn
        diff = float((new - delta).abs().max())
        delta = new
        if diff < tol:
            break
    return pts, delta, v, it + 1


@torch.no_grad()
def largest_subsolution(P, u, n, h, iters=20000, tol=1e-7, chunk=1 << 22):
    """Largest subsolution of T_h below u on the n^d grid: w_0 = u,  w_{k+1} = min(u, T_h w_k)  (monotone decreasing).
    rho = u - w* is the sharp replacement of the global term: u <= V_h + rho, and rho <= delta* <= sup v/(1-gamma).
    Returns (pts, w*, rho, iterations)."""
    from .torus import uniform_grid, periodic_interp
    dev = P.device
    d = P.dim
    pts, _ = uniform_grid(n, d, dev)
    G = pts.shape[0]
    K = P.actions.shape[0]
    gam = math.exp(-P.lam * h)
    ug = torch.empty(G, device=dev)
    for i in range(0, G, chunk):
        ug[i:i + chunk] = u(pts[i:i + chunk])
    xa = pts[:, None, :].expand(G, K, d).reshape(-1, d)
    aa = P.actions[None, :, :].expand(G, K, d).reshape(-1, d)
    C = torch.empty(G * K, device=dev)
    for i in range(0, G * K, chunk):
        C[i:i + chunk] = P.running_cost(xa[i:i + chunk], aa[i:i + chunk], h)
    C = C.view(G, K)
    xn = P.step(xa, aa, h)
    w = ug.clone()
    shape = [n] * d
    for it in range(iters):
        Tw = (C + gam * periodic_interp(w.view(*shape), xn).view(G, K)).min(1).values
        new = torch.minimum(ug, Tw)
        diff = float((w - new).abs().max())
        w = new
        if diff < tol:
            break
    return pts, w, ug - w, it + 1


@torch.no_grad()
def global_terms(P, u, h, X, V_ref):
    """s = sup [u - T_h u]_+, b_h = sup (T_h V - V), E = ||u - V||_inf on the sample X; returns (s, b_h, E, gamma)."""
    gam = math.exp(-P.lam * h)
    Tu, _ = bellman_target(P, u, X, h)
    s = float(torch.clamp(u(X) - Tu, min=0.0).max())
    TV, _ = bellman_target(P, V_ref, X, h)
    b = float((TV - V_ref(X)).max())
    E = float((u(X) - V_ref(X)).abs().max())
    return s, b, E, gam
