"""E7: trajectory Bellman-residual bound for sample-and-hold policies.

For hold time h, gamma = e^{-lam h}, T_h w(x) = min_a [c_h(x,a) + gamma w(x_a^+)], any sample-and-hold policy pi with
trajectory x_k and per-step lookahead regret eta_k = c_h(x_k,a_k) + gamma u(x_{k+1}) - T_h u(x_k) >= 0:
    J^pi(x_0) - V(x_0) <= sum_k gamma^k ( [T_h u - u]_+(x_k) + eta_k ) + ( sup_x [u - T_h u]_+ + b_h ) / (1 - gamma),
    b_h = sup_x (T_h V - V).
Compare with the classical contraction bound: (b_h + sup_k eta_k + 2 gamma E)/(1-gamma), E = ||u - V||_inf.
Reports validity and tightness over many start states for several critics / policies / problems.
"""
import sys, json, time, math
sys.path.insert(0, ".")
import torch
from hjrl.problems import Ridge1D, Linf2D, Obstacle2D
from hjrl.critic import FourierMLP, MinBranchCritic, GridFunction, eval_on_grid
from hjrl.torus import uniform_grid, random_points, wrap
from hjrl.train import train_sl_td, train_pinn
from hjrl.solvers import sl_value_iteration
from hjrl.residuals import bellman_target
from hjrl.policy import GradGreedy, LookaheadGreedy, BranchActor

from hjrl.bound import rollout_terms, global_terms

dev = torch.device("cuda")
torch.manual_seed(0)


def evaluate(tag, P, critics, starts, X_global, V_ref, h=0.02, T=12.0, rows=None):
    for cname, u in critics.items():
        s, b, E, gam = global_terms(P, u, h, X_global, V_ref)
        pols = {"lookahead h=hold": lambda: LookaheadGreedy(P, u, h),
                "lookahead 5h": lambda: LookaheadGreedy(P, u, 5 * h),
                "grad-greedy tol0": lambda: GradGreedy(P, u, prefer_zero=True, tie_tol=0.0),
                "grad-greedy tol0.05": lambda: GradGreedy(P, u, prefer_zero=True, tie_tol=0.05)}
        if isinstance(u, MinBranchCritic):
            pols["branch reachable+hyst"] = lambda: BranchActor(P, u, rule="reachable", hysteresis=0.02, h=0.05)
        for pname, mk in pols.items():
            t = time.time()
            r = rollout_terms(P, u, mk(), starts, h, T, V_ref)
            bound = r["traj"] + r["eta"] + (s + b) / (1 - gam)
            bound5 = (b + r["eta_max"] + 2 * gam * E) / (1 - gam)
            gap = r["gap"]
            sig = gap > 0.01
            ratio = (bound / gap.clamp(min=1e-9))[sig]
            ratio5 = (bound5 / gap.clamp(min=1e-9))[sig]
            row = {"problem": tag, "critic": cname, "policy": pname, "h": h, "E": E, "s_sup_sub": s, "b_h": b,
                   "gap_max": float(gap.max()), "gap_mean": float(gap.mean()),
                   "traj_max": float(r["traj"].max()), "eta_max_sum": float(r["eta"].max()),
                   "bound_max": float(bound.max()), "valid_frac": float((gap <= bound + 1e-3).float().mean()),
                   "ratio_median": float(ratio.median()) if sig.any() else float("nan"),
                   "ratio_max": float(ratio.max()) if sig.any() else float("nan"),
                   "ratio5_median": float(ratio5.median()) if sig.any() else float("nan"),
                   "bound5_max": float(bound5.max()), "n_sig": int(sig.sum()), "n": int(gap.numel()), "time_s": time.time() - t}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


rows = []
which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("1d", "all"):
    P = Ridge1D(dev, lam=1.0)
    N = 8192
    pts, _ = uniform_grid(N, 1, dev)
    V = P.V(pts)
    crit = {"V+0.02 sin(8pi x)": GridFunction((V + 0.02 * torch.sin(8 * math.pi * pts[:, 0])).view(N)),
            "V+0.05 sin(16pi x)": GridFunction((V + 0.05 * torch.sin(16 * math.pi * pts[:, 0])).view(N))}
    for name, fn in [("SL-TD net", lambda c: train_sl_td(P, c, steps=4000, batch=4096, h=0.05, log_every=10 ** 6)),
                     ("PINN net", lambda c: train_pinn(P, c, steps=4000, batch=4096, log_every=10 ** 6)),
                     ("PINN+visc net", lambda c: train_pinn(P, c, steps=4000, batch=4096, log_every=10 ** 6, visc=0.05))]:
        torch.manual_seed(1)
        c = FourierMLP(1, n_freq=8, width=64, depth=3).to(dev)
        fn(c)
        crit[name] = c
    x0, _ = uniform_grid(2048, 1, dev)
    evaluate("ridge1d", P, crit, x0, pts, P.V, rows=rows)
    json.dump(rows, open("results/e7_policy_bound.json", "w"), indent=1)

if which in ("2d", "all"):
    P = Linf2D(dev, lam=1.0)
    N = 1024
    pts, _ = uniform_grid(N, 2, dev)
    V = P.V(pts)
    crit = {"V+0.02 sin(8pi x1)sin(8pi x2)": GridFunction((V + 0.02 * torch.sin(8 * math.pi * pts[:, 0]) * torch.sin(8 * math.pi * pts[:, 1])).view(N, N))}
    for name, fn in [("SL-TD net", lambda c: train_sl_td(P, c, steps=6000, batch=8192, h=0.05, log_every=10 ** 6)),
                     ("PINN+visc net", lambda c: train_pinn(P, c, steps=6000, batch=8192, log_every=10 ** 6, visc=0.05))]:
        torch.manual_seed(1)
        c = FourierMLP(2, n_freq=12, width=128, depth=3).to(dev)
        fn(c)
        crit[name] = c
    torch.manual_seed(1)
    mb = MinBranchCritic(2, K=8, n_freq=12, width=96, depth=3).to(dev)
    train_sl_td(P, mb, steps=6000, batch=8192, h=0.05, log_every=10 ** 6)
    mb.eval()
    crit["minK8 SL-TD"] = mb
    x0, _ = uniform_grid(48, 2, dev)
    t = torch.linspace(-0.95, 0.95, 64, device=dev)
    x0 = torch.cat([x0, torch.stack([t, t], 1), torch.stack([t, torch.zeros_like(t)], 1)], 0)
    Xg, _ = uniform_grid(512, 2, dev)
    evaluate("linf2d", P, crit, x0, Xg, P.V, rows=rows)
    json.dump(rows, open("results/e7_policy_bound.json", "w"), indent=1)

if which in ("obstacle", "all"):
    P = Obstacle2D(dev, lam=0.5, K=16)
    pts, Vg, _ = sl_value_iteration(P, 512, h=0.01, iters=30000, tol=1e-7)
    P.set_reference(Vg.view(512, 512))
    sd = torch.load("results/e5_critics.pt")
    crit = {}
    for name, ctor in [("MLP", lambda: FourierMLP(2, n_freq=12, width=128, depth=3)), ("minK2", lambda: MinBranchCritic(2, K=2, n_freq=12, width=96, depth=3)),
                       ("minK4", lambda: MinBranchCritic(2, K=4, n_freq=12, width=96, depth=3))]:
        m = ctor().to(dev); m.load_state_dict(sd[name]); m.eval(); crit[name] = m
    gs = torch.Generator(device=dev).manual_seed(3)
    x0 = torch.stack([-0.9 + 0.8 * torch.rand(2048, device=dev, generator=gs), -0.9 + 1.8 * torch.rand(2048, device=dev, generator=gs)], 1)
    tt = torch.linspace(-0.9, -0.12, 64, device=dev)
    x0 = torch.cat([x0, torch.stack([tt, torch.zeros_like(tt)], 1), torch.stack([tt, 0.01 * torch.ones_like(tt)], 1)], 0)
    Xg, _ = uniform_grid(512, 2, dev)
    evaluate("obstacle2d", P, crit, x0, Xg, P.V, h=0.02, T=16.0, rows=rows)
    json.dump(rows, open("results/e7_policy_bound.json" if which == "all" else "results/e7_policy_bound_obstacle.json", "w"), indent=1)

print("E7 done")
