"""Trainers: PINN (AD residual), semi-Lagrangian TD (fitted VI), contact-TD (test-function TD at contact points), hybrid."""
import math
import time
import torch
from .torus import random_points, wrap
from .critic import grad_u, eval_on_grid
from .residuals import bellman_target, contact_residual, test_function_td
from .contact import grid_contact


@torch.no_grad()
def uniform_error(problem, u, n=2048):
    pts, vals, _ = eval_on_grid(u, n, problem.dim, problem.device)
    return float((vals - problem.V(pts)).abs().max()), float((vals - problem.V(pts)).abs().mean())


def make_sched(opt, steps, kind="const_decay", final_frac=0.1, decay_start=0.8):
    """'cosine': cosine annealing to 0 over all steps (fine for PINN regression).
    'const_decay': constant LR for the first `decay_start` fraction, then cosine decay to final_frac*lr.
    Fitted value iteration needs the network to keep moving by the value increments of every backup, so an LR that
    decays early freezes the iteration long before the Bellman contraction has converged."""
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    s0 = int(decay_start * steps)

    def f(s):
        if s < s0:
            return 1.0
        t = (s - s0) / max(1, steps - s0)
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


class History:
    def __init__(self):
        self.rows = []

    def log(self, **kw):
        self.rows.append(kw)

    def last(self):
        return self.rows[-1] if self.rows else {}


def _eval_row(problem, u, step, t0, env_samples, n_eval, extra=None):
    row = {"step": step, "time": time.time() - t0, "env_samples": env_samples}
    if problem.has_exact_V or getattr(problem, "_Vgrid", None) is not None:
        e_inf, e_1 = uniform_error(problem, u, n_eval)
        row.update(err_inf=e_inf, err_mean=e_1)
    if extra:
        row.update(extra)
    return row


def train_pinn(problem, critic, steps=5000, batch=4096, lr=1e-3, seed=0, log_every=250, n_eval=1024, hist=None,
               lr_sched="cosine", visc=0.0, visc_decay_frac=0.8):
    """Minimize mean (lam u + H(x, grad u) - nu Laplacian u)^2 on uniform samples. Model-based (H known); no env samples.
    visc>0 adds vanishing artificial viscosity nu(s) = visc * max(0, 1 - s/(visc_decay_frac*steps)) (Shilova et al. style
    epsilon-scheduling), which selects the viscosity solution among a.e. solutions of the first-order equation."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    sched = make_sched(opt, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    for s in range(steps + 1):
        x = random_points(batch, problem.dim, problem.device).requires_grad_(True)
        nu = visc * max(0.0, 1.0 - s / (visc_decay_frac * steps)) if visc > 0 else 0.0
        val = critic(x)
        g, = torch.autograd.grad(val.sum(), x, create_graph=True)
        F = problem.lam * val + problem.H(x, g)
        if nu > 0:
            lap = 0.0
            for k in range(problem.dim):
                gk, = torch.autograd.grad(g[:, k].sum(), x, create_graph=True)
                lap = lap + gk[:, k]
            F = F - nu * lap
        loss = (F ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps:
            hist.log(**_eval_row(problem, critic, s, t0, 0, n_eval, {"loss": loss.item()}))
    return hist


def train_sl_td(problem, critic, steps=5000, batch=4096, h=0.02, lr=1e-3, seed=0, log_every=250, n_eval=1024,
                replay=None, replay_frac=0.0, hist=None, target_every=20, lr_sched="const_decay", asym=(1.0, 1.0),
                branch_pde=0.0):
    """Fitted value iteration with semi-Lagrangian targets: loss = (u(x) - stopgrad T_h u_target(x))^2,
    where u_target is a frozen copy refreshed every `target_every` steps (without it the semi-gradient min-backup
    diverges: the min over K neighbours of an oscillating network is biased downward and feeds back on itself).
    asym=(w_plus, w_minus) weights the one-sided errors: w_plus*[u - T_h u]_+^2 + w_minus*[T_h u - u]_+^2; w_plus >> w_minus
    trains a *conservative* critic (approximate subsolution u <= T_h u), which removes the global term of the policy bound.
    Env samples per step: batch * K one-step rollouts of length h."""
    import copy
    torch.manual_seed(seed)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    sched = make_sched(opt, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    K = problem.actions.shape[0]
    env = 0
    target = copy.deepcopy(critic).eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)
    for s in range(steps + 1):
        if s % target_every == 0:
            target.load_state_dict(critic.state_dict())
        x = random_points(batch, problem.dim, problem.device)
        if replay is not None and replay_frac > 0 and replay.shape[0] > 0:
            m = int(batch * replay_frac)
            j = torch.randint(0, replay.shape[0], (m,), device=x.device)
            x = torch.cat([x[: batch - m], replay[j]], 0)
        tgt, _ = bellman_target(problem, target, x, h)
        env += x.shape[0] * K
        diff = critic(x) - tgt
        if asym == (1.0, 1.0):
            loss = (diff ** 2).mean()
        else:
            loss = (asym[0] * torch.clamp(diff, min=0.0) ** 2 + asym[1] * torch.clamp(-diff, min=0.0) ** 2).mean()
        if branch_pde > 0:
            # PDE residual of the ACTIVE branch of a min-of-branches critic:  lam u_k* + H(x, D u_k*) = 0 on its active region
            xg = x.detach().requires_grad_(True)
            b = critic.branches(xg)
            kstar = b.argmin(1, keepdim=True)
            uk = b.gather(1, kstar)[:, 0]
            gk, = torch.autograd.grad(uk.sum(), xg, create_graph=True)
            rpde = problem.lam * uk + problem.H(xg, gk)
            loss = loss + branch_pde * (rpde ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps:
            hist.log(**_eval_row(problem, critic, s, t0, env, n_eval, {"loss": loss.item()}))
    return hist


def train_sl_td_prio(problem, critic, steps=5000, batch=4096, h=0.05, lr=1e-3, seed=0, log_every=250, n_eval=1024,
                     hist=None, target_every=20, lr_sched="const_decay", prio_every=50, n_probe=16384, buf_size=4096,
                     prio_frac=0.5, jitter=0.02, anneal=False):
    """SL-TD with Bellman-residual-prioritized replay: every `prio_every` steps, evaluate |u - T_h u| on n_probe uniform
    states and keep the `buf_size` worst as a buffer; each minibatch draws prio_frac of its states from the buffer (with
    Gaussian jitter of scale `jitter`) and the rest uniformly.  Targets and losses are those of train_sl_td."""
    import copy
    torch.manual_seed(seed)
    dev = problem.device
    d = problem.dim
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    sched = make_sched(opt, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    K = problem.actions.shape[0]
    env = 0
    target = copy.deepcopy(critic).eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)
    buf = torch.empty(0, d, device=dev)
    for s in range(steps + 1):
        if s % target_every == 0:
            target.load_state_dict(critic.state_dict())
        if s % prio_every == 0:
            with torch.no_grad():
                xp = random_points(n_probe, d, dev)
                tp, _ = bellman_target(problem, critic, xp, h)
                res = (critic(xp) - tp).abs()
                buf = xp[res.topk(min(buf_size, n_probe)).indices]
            env += n_probe * K
        x = random_points(batch, d, dev)
        pf = prio_frac * max(0.0, 1.0 - s / (0.8 * steps)) if anneal else prio_frac
        if buf.shape[0] > 0 and pf > 0:
            m = int(batch * pf)
            j = torch.randint(0, buf.shape[0], (m,), device=dev)
            x = torch.cat([x[: batch - m], wrap(buf[j] + jitter * torch.randn(m, d, device=dev))], 0)
        tgt, _ = bellman_target(problem, target, x, h)
        env += x.shape[0] * K
        loss = ((critic(x) - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps:
            hist.log(**_eval_row(problem, critic, s, t0, env, n_eval, {"loss": loss.item()}))
    return hist


def train_fqi_q(problem, qnet, steps=10000, batch=4096, h=0.05, lr=1e-3, seed=0, log_every=500, n_eval=256,
                n_data=1 << 18, target_every=20, lr_sched="const_decay", hist=None, double=True):
    """Model-free fitted Q-iteration on a fixed batch of simulator transitions (x, a, c_h, x') with x uniform and every
    action tried once per state: Q(x,a) <- c_h + gamma Q_target(x', a*),  a* = argmin_a' Q_online(x', a') if double
    (double-DQN targets, which remove the min-backup underestimation bias of K independent outputs) else argmin of Q_target.
    No dynamics model is used in training. Q-greedy agrees with model-based lookahead only under Bellman consistency."""
    import copy
    torch.manual_seed(seed)
    dev = problem.device
    d = problem.dim
    K = problem.actions.shape[0]
    gam = math.exp(-problem.lam * h)
    with torch.no_grad():
        X = random_points(n_data, d, dev)
        xa = X[:, None, :].expand(n_data, K, d).reshape(-1, d)
        aa = problem.actions[None, :, :].expand(n_data, K, d).reshape(-1, d)
        C = problem.running_cost(xa, aa, h).view(n_data, K)
        Xn = problem.step(xa, aa, h).view(n_data, K, d)
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    sched = make_sched(opt, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    target = copy.deepcopy(qnet).eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)
    from .critic import QMinValue
    vq = QMinValue(qnet)
    for s in range(steps + 1):
        if s % target_every == 0:
            target.load_state_dict(qnet.state_dict())
        j = torch.randint(0, n_data, (batch,), device=dev)
        with torch.no_grad():
            xn = Xn[j].reshape(-1, d)
            qt = target(xn).view(batch, K, K)
            if double:
                astar = qnet(xn).view(batch, K, K).argmin(-1, keepdim=True)
                nxt = qt.gather(-1, astar)[..., 0]
            else:
                nxt = qt.min(-1).values
            tgt = C[j] + gam * nxt                                                            # (batch,K): target per action
        loss = ((qnet(X[j]) - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps:
            hist.log(**_eval_row(problem, vq, s, t0, n_data * K, n_eval, {"loss": loss.item()}))
    return hist


def train_ddpg_batch(problem, qnet, actor, steps=12000, batch=4096, h=0.05, lr=1e-3, seed=0, n_data=1 << 18,
                     target_every=20, log_every=1000, lr_sched="const_decay", hist=None, cand=None):
    """Batch (offline) deterministic actor-critic with continuous actions in the unit disk.
    Data: states x uniform, actions a uniform in the disk (half on the circle), one held-action transition each.
    Critic target: c_h + gamma * min over {mu_targ(x')} U cand of Q_targ(x', .)   (cand=None: plain DDPG target;
    cand=(K,d) candidate actions: continuous fitted Q-iteration, the actor being one of the candidates).
    Actor: minimize Q(x, mu(x)) by gradient in the action.  Model-free."""
    import copy
    torch.manual_seed(seed)
    dev = problem.device
    d = problem.dim
    gam = math.exp(-problem.lam * h)
    with torch.no_grad():
        X = random_points(n_data, d, dev)
        th = torch.rand(n_data, device=dev) * 2 * math.pi
        r = torch.where(torch.rand(n_data, device=dev) < 0.5, torch.ones(n_data, device=dev), torch.rand(n_data, device=dev).sqrt())
        A = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
        C = problem.running_cost(X, A, h)
        Xn = problem.step(X, A, h)
    opt_q = torch.optim.Adam(qnet.parameters(), lr=lr)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    sched_q = make_sched(opt_q, steps, lr_sched)
    sched_a = make_sched(opt_a, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    q_t = copy.deepcopy(qnet).eval(); a_t = copy.deepcopy(actor).eval()
    for m in (q_t, a_t):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    for s in range(steps + 1):
        if s % target_every == 0:
            q_t.load_state_dict(qnet.state_dict()); a_t.load_state_dict(actor.state_dict())
        j = torch.randint(0, n_data, (batch,), device=dev)
        with torch.no_grad():
            xn = Xn[j]
            if cand is None:
                y = C[j] + gam * q_t(xn, a_t(xn))
            else:
                Kc = cand.shape[0]
                cands = torch.cat([cand[None].expand(batch, Kc, d), a_t(xn)[:, None, :]], 1)
                xrep = xn[:, None, :].expand(batch, Kc + 1, d).reshape(-1, d)
                qn_t = q_t(xrep, cands.reshape(-1, d)).view(batch, Kc + 1)
                astar = qnet(xrep, cands.reshape(-1, d)).view(batch, Kc + 1).argmin(1, keepdim=True)   # double: argmin online
                y = C[j] + gam * qn_t.gather(1, astar)[:, 0]
        lq = ((qnet(X[j], A[j]) - y) ** 2).mean()
        opt_q.zero_grad(set_to_none=True); lq.backward(); opt_q.step(); sched_q.step()
        la = qnet(X[j], actor(X[j])).mean()
        opt_a.zero_grad(set_to_none=True); la.backward(); opt_a.step(); sched_a.step()
        if s % log_every == 0 or s == steps:
            hist.log(step=s, time=time.time() - t0, loss_q=lq.item(), loss_a=la.item())
    return hist


def _q_diff(problem, u_x, p, eps, h, sign, c, dz):
    """Differentiable q_h^phi(x,a) (B,K) as a function of the critic value u_x at the contact (p, dz, c fixed)."""
    gam = math.exp(-problem.lam * h)
    phi_inc = (p[:, None, :] * dz).sum(-1) + sign * (dz ** 2).sum(-1) / (2 * eps)
    return (c + gam * phi_inc + (gam - 1.0) * u_x[:, None]) / h


def train_contact_td(problem, critic, steps=3000, n_centers=1024, eps=0.05, h=0.01, grid_n=1024, lr=1e-3, seed=0,
                     log_every=100, n_eval=1024, replay_size=1024, replay_frac=0.5, radius=None, hist=None,
                     w_up=1.0, w_dn=1.0, anchor_sl=0.0, sl_h=0.02, lr_sched="const_decay"):
    """Contact-TD: at each step, evaluate the critic on the grid, find exact upper/lower contacts for a batch of centers,
    roll out every action for time h from each contact point, form test-function TD q_h^{phi+-}, and penalize violations
        upper: [ -min_a q_h^{phi+} ]_+^2 ,   lower: [ min_a q_h^{phi-} ]_+^2 ,
    using a semi-gradient through the critic value only; contact position and slope are deliberately frozen.
    Centers = uniform samples + replay of the worst violators (adaptive replay).
    anchor_sl>0 adds a semi-Lagrangian TD term on the same centers (hybrid variant)."""
    torch.manual_seed(seed)
    dev = problem.device
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    sched = make_sched(opt, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    K = problem.actions.shape[0]
    d = problem.dim
    env = 0
    replay = torch.empty(0, d, device=dev)
    gam = math.exp(-problem.lam * h)
    for s in range(steps + 1):
        y = random_points(n_centers, d, dev)
        if replay.shape[0] > 0 and replay_frac > 0:
            m = int(n_centers * replay_frac)
            j = torch.randint(0, replay.shape[0], (m,), device=dev)
            y = torch.cat([y[: n_centers - m], replay[j]], 0)
        _, grid_vals, _ = eval_on_grid(critic, grid_n, d, dev)
        rad = radius if radius is not None else min(1.0, 2.0 * problem.L_V * eps + 8.0 / grid_n)
        losses = []
        viol_all = []
        for sign, w in ((+1, w_up), (-1, w_dn)):
            x, p, ux, _ = grid_contact(grid_vals, grid_n, d, y, eps, sign, rad)
            with torch.no_grad():
                B = x.shape[0]
                xa = x[:, None, :].expand(B, K, d).reshape(-1, d)
                aa = problem.actions[None, :, :].expand(B, K, d).reshape(-1, d)
                c = problem.running_cost(xa, aa, h).view(B, K)
                dz = wrap(problem.step(xa, aa, h).view(B, K, d) - x[:, None, :])
                env += B * K
            u_x = critic(x)
            q = _q_diff(problem, u_x, p, eps, h, sign, c, dz)
            qmin = q.min(1).values
            viol = torch.clamp(-sign * qmin, min=0.0)
            losses.append(w * (viol ** 2).mean())
            viol_all.append((viol.detach(), x))
        loss = sum(losses)
        if anchor_sl > 0:
            tgt, _ = bellman_target(problem, critic, y, sl_h)
            env += y.shape[0] * K
            loss = loss + anchor_sl * ((critic(y) - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        # adaptive replay: keep the worst violating contact points as future centers
        with torch.no_grad():
            v = torch.cat([viol_all[0][0], viol_all[1][0]])
            xs = torch.cat([viol_all[0][1], viol_all[1][1]])
            k = min(replay_size // 4, xs.shape[0])
            top = v.topk(k).indices
            replay = torch.cat([replay, xs[top]], 0)[-replay_size:]
        if s % log_every == 0 or s == steps:
            hist.log(**_eval_row(problem, critic, s, t0, env, n_eval,
                                 {"loss": loss.item(), "viol_up": float(viol_all[0][0].max()),
                                  "viol_dn": float(viol_all[1][0].max())}))
    return hist


def train_hybrid(problem, critic, steps=5000, batch=4096, h=0.02, lr=1e-3, seed=0, log_every=250, n_eval=1024,
                 eps=0.05, audit_every=50, n_audit=2048, grid_n=1024, replay_size=2048, replay_frac=0.25,
                 hist=None, target_every=20, lr_sched="const_decay"):
    """SL-TD with contact-audit adaptive replay: every `audit_every` steps the critic is audited with exact contact
    residuals on random centers; the worst-violating contact points are pushed into a replay buffer that supplies a
    fraction of every SL-TD minibatch. (The 'smaller algorithm' fallback of the proposal: contact check as audit +
    adaptive replay, Bellman as the learning rule.)"""
    import copy
    torch.manual_seed(seed)
    dev = problem.device
    d = problem.dim
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    sched = make_sched(opt, steps, lr_sched)
    hist = hist or History()
    t0 = time.time()
    K = problem.actions.shape[0]
    env = 0
    replay = torch.empty(0, d, device=dev)
    target = copy.deepcopy(critic).eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)
    last_R = 0.0
    for s in range(steps + 1):
        if s % target_every == 0:
            target.load_state_dict(critic.state_dict())
        if s % audit_every == 0:
            with torch.no_grad():
                _, grid_vals, _ = eval_on_grid(critic, grid_n, d, dev)
                y = random_points(n_audit, d, dev)
                rad = min(1.0, 2.0 * problem.L_V * eps + 8.0 / grid_n)
                vs, xs = [], []
                for sign in (+1, -1):
                    x, p, ux, _ = grid_contact(grid_vals, grid_n, d, y, eps, sign, rad)
                    r = torch.clamp(sign * problem.F(x, ux, p), min=0.0)
                    vs.append(r); xs.append(x)
                v = torch.cat(vs); xs = torch.cat(xs)
                last_R = float(v.max())
                top = v.topk(min(replay_size // 4, xs.shape[0])).indices
                replay = torch.cat([replay, xs[top]], 0)[-replay_size:]
        x = random_points(batch, d, dev)
        if replay.shape[0] > 0 and replay_frac > 0:
            m = int(batch * replay_frac)
            j = torch.randint(0, replay.shape[0], (m,), device=dev)
            x = torch.cat([x[: batch - m], replay[j]], 0)
        tgt, _ = bellman_target(problem, target, x, h)
        env += x.shape[0] * K
        loss = ((critic(x) - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps:
            hist.log(**_eval_row(problem, critic, s, t0, env, n_eval, {"loss": loss.item(), "audit_R": last_R}))
    return hist


def contact_audit(problem, critic, eps, n_centers=4096, grid_n=2048, L_u=None, certify=True, seed=0):
    """Post-hoc audit: exact-contact residuals (upper/lower) on random centers + a full center grid; returns summary + bound."""
    g = torch.Generator(device=problem.device).manual_seed(seed)
    d = problem.dim
    _, grid_vals, _ = eval_on_grid(critic, grid_n, d, problem.device)
    from .critic import lipschitz_estimate
    if L_u is None:
        L_u = lipschitz_estimate(critic, grid_n, d, problem.device)
    y = random_points(n_centers, d, problem.device, generator=g)
    up = contact_residual(problem, grid_vals, grid_n, y, eps, +1, L_u=L_u, certify=certify)
    dn = contact_residual(problem, grid_vals, grid_n, y, eps, -1, L_u=L_u, certify=certify)
    key = "r_cert" if certify else "r_best"
    R = max(float(up[key].max()), float(dn[key].max()))
    from .residuals import certificate_bound
    return {"R_up": float(up[key].max()), "R_dn": float(dn[key].max()), "R": R, "L_u": L_u,
            "bound": certificate_bound(problem, R, eps), "eps": eps, "up": up, "dn": dn}
