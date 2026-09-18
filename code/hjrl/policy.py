"""Policy extraction from critics and sample-and-hold evaluation."""
import math
import torch
from .torus import wrap
from .critic import grad_u
from .residuals import bellman_target


class GradGreedy:
    """a = argmax_a { -ell(x,a) - f(x,a).grad u(x) }.  If prefer_zero, the zero action (index 0) is chosen whenever
    no action beats it by more than tie_tol (a 'symmetric tie-breaking' rule)."""

    def __init__(self, problem, u, tie_tol=0.0, prefer_zero=True):
        self.problem, self.u, self.tie_tol, self.prefer_zero = problem, u, tie_tol, prefer_zero

    def __call__(self, x, state=None):
        _, g = grad_u(self.u, x)
        with torch.no_grad():
            hv = self.problem._ham_values(x, g)
            m, i = hv.max(1)
            if self.prefer_zero:
                i = torch.where(m - hv[:, 0] <= self.tie_tol, torch.zeros_like(i), i)
        return i, state


class LookaheadGreedy:
    """a = argmin_a [ int_0^h e^{-lam s} ell ds + e^{-lam h} u(x + h f) ]  (semi-Lagrangian one-step lookahead)."""

    def __init__(self, problem, u, h):
        self.problem, self.u, self.h = problem, u, h

    def __call__(self, x, state=None):
        _, i = bellman_target(self.problem, self.u, x, self.h)
        return i, state


class QGreedy:
    """a = argmin_a Q(x,a) for a Q-network with K outputs (model-free one-step lookahead)."""

    def __init__(self, qnet):
        self.qnet = qnet

    def __call__(self, x, state=None):
        with torch.no_grad():
            return self.qnet(x).argmin(-1), state


class BranchActor:
    """Policy from a min-of-branches critic u = min_k u_k.

    rule='active'    : take the branch attaining the min at x (ties -> lowest index), act greedily on its gradient.
    rule='reachable' : among branches active at x within tol_active, keep only those that are the *strict* minimizer on a
                       positive fraction of probe points in the ball of radius r_probe (reachable-gradient filter);
                       score each candidate by its short-horizon cost (running cost + discounted critic after hold time h)
                       and keep the previous branch unless another one is better by more than `hysteresis` (per-trajectory state).
    """

    def __init__(self, problem, critic, rule="active", tol_active=1e-3, r_probe=0.05, n_probe=32, margin=1e-3,
                 hysteresis=0.0, h=0.05, generator=None):
        self.problem, self.critic, self.rule = problem, critic, rule
        self.tol_active, self.r_probe, self.n_probe, self.margin = tol_active, r_probe, n_probe, margin
        self.hysteresis, self.h, self.generator = hysteresis, h, generator

    def branch_actions(self, x):
        """Greedy action index for each branch k from its own gradient: (B,K)."""
        B = x.shape[0]
        acts = []
        for k, head in enumerate(self.critic.heads):
            _, g = grad_u(head, x)
            with torch.no_grad():
                acts.append(self.problem._ham_values(x, g).argmax(1))
        return torch.stack(acts, 1)

    @torch.no_grad()
    def reachable_mask(self, x):
        """(B,K) bool: branch k is the strict unique minimizer at some probe point near x."""
        B, d = x.shape
        K = self.critic.K
        probes = wrap(x[:, None, :] + self.r_probe * torch.randn(B, self.n_probe, d, device=x.device,
                                                                     generator=self.generator))
        b = self.critic.branches(probes.reshape(-1, d)).view(B, self.n_probe, K)
        sorted_b, _ = b.sort(-1)
        strict = b <= sorted_b[..., :1] + 0.0
        gap = sorted_b[..., 1] - sorted_b[..., 0]
        strict = strict & (gap[..., None] > self.margin)
        return strict.any(1)

    def __call__(self, x, state=None):
        B = x.shape[0]
        with torch.no_grad():
            b = self.critic.branches(x)
        acts = self.branch_actions(x)                                   # (B,K)
        if self.rule == "active":
            k = b.argmin(1)
            return acts[torch.arange(B), k], state
        # reachable rule
        with torch.no_grad():
            active = b <= b.min(1, keepdim=True).values + self.tol_active
            reach = self.reachable_mask(x)
            cand = active & reach
            none = ~cand.any(1)
            cand[none] = active[none]                                   # fall back to active set
            # score candidates by short-horizon lookahead cost with each branch's action
            K = self.critic.K
            d = x.shape[1]
            a = self.problem.actions[acts.reshape(-1)]
            xa = x[:, None, :].expand(B, K, d).reshape(-1, d)
            c = self.problem.running_cost(xa, a, self.h)
            xn = self.problem.step(xa, a, self.h)
            score = (c + math.exp(-self.problem.lam * self.h) * self.critic(xn)).view(B, K)
            score = torch.where(cand, score, torch.full_like(score, float("inf")))
            best_s, best_k = score.min(1)
            if state is None:
                state = best_k.clone()
            prev = state
            prev_s = score[torch.arange(B), prev]
            keep = prev_s <= best_s + self.hysteresis
            k = torch.where(keep, prev, best_k)
        return acts[torch.arange(B), k], k


@torch.no_grad()
def evaluate_policy(problem, policy, x0, tau, T, V_ref=None, n_sub=4, track_obs=False):
    """Sample-and-hold rollout: choose action every tau, hold it, accumulate discounted cost up to time T.
    Returns dict with J (B,), n_switch (B,), and if V_ref: gap = J - V_ref(x0). Tail is closed with V_ref(X_T) if given."""
    x = x0.clone()
    B = x.shape[0]
    J = torch.zeros(B, device=x.device)
    n_switch = torch.zeros(B, device=x.device)
    obs_time = torch.zeros(B, device=x.device)
    state = None
    prev = None
    n_steps = int(round(T / tau))
    for t in range(n_steps):
        idx, state = policy(x, state)
        a = problem.actions[idx]
        with torch.no_grad():
            J += math.exp(-problem.lam * t * tau) * problem.running_cost(x, a, tau, n_sub=n_sub)
            if track_obs and hasattr(problem, "obs"):
                obs_time += tau * (problem.obs(x) > 0.5).float()
            if prev is not None:
                n_switch += (idx != prev).float()
            prev = idx
            x = problem.step(x, a, tau)
    out = {"J": J, "n_switch": n_switch, "x_T": x, "obs_time": obs_time}
    if V_ref is not None:
        J_closed = J + math.exp(-problem.lam * T) * V_ref(x)
        out["J"] = J_closed
        out["gap"] = J_closed - V_ref(x0)
    return out
