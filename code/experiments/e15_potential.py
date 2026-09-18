"""E15: adversarial violation potential versus the global term sup[u - T_h u]_+/(1-gamma).

For the critics of E7 (1D, 2D, obstacle), compute delta* on the grid by max-value-iteration, evaluate delta*(x_0) on the
E7 start sets, and compare the certificate  traj + eta + delta*(x_0) + b_h/(1-gamma)  with the E7 certificate
traj + eta + (s + b_h)/(1-gamma):  validity, median/max ratio to the realized gap, and the tightening factor.
"""
import sys, json, time, math
sys.path.insert(0, ".")
import torch
from hjrl.problems import Ridge1D, Linf2D, Obstacle2D
from hjrl.critic import FourierMLP, MinBranchCritic, GridFunction
from hjrl.torus import uniform_grid, random_points, periodic_interp
from hjrl.train import train_sl_td, train_pinn
from hjrl.solvers import sl_value_iteration
from hjrl.policy import GradGreedy, LookaheadGreedy
from hjrl.bound import rollout_terms, global_terms, adversarial_potential, largest_subsolution

dev = torch.device("cuda")
torch.manual_seed(0)
rows = []
which = sys.argv[1] if len(sys.argv) > 1 else "all"


def evaluate(tag, P, critics, starts, Xg, V_ref, n_grid, h=0.02, T=12.0):
    for cname, u in critics.items():
        s, b, E, gam = global_terms(P, u, h, Xg, V_ref)
        t0 = time.time()
        pts, delta, v, its = adversarial_potential(P, u, n_grid, h)
        tpot = time.time() - t0
        t0 = time.time()
        _, wstar, rho, its2 = largest_subsolution(P, u, n_grid, h)
        trho = time.time() - t0
        shape = [n_grid] * P.dim
        d0 = periodic_interp(delta.view(*shape), starts)
        r0 = periodic_interp(rho.view(*shape), starts)
        with torch.no_grad():
            excess_true = torch.clamp(u(pts) - V_ref(pts), min=0.0)     # [u - V]_+ on the grid (rho should dominate u - V_h)
        base = {"problem": tag, "critic": cname, "E": E, "s": s, "b_h": b, "global_old": (s + b) / (1 - gam),
                "delta_max": float(delta.max()), "delta_mean": float(delta.mean()), "v_max": float(v.max()),
                "frac_v_pos": float((v > 1e-4).float().mean()), "delta0_med": float(d0.median()), "delta0_max": float(d0.max()),
                "rho_max": float(rho.max()), "rho_mean": float(rho.mean()), "rho0_med": float(r0.median()), "rho0_max": float(r0.max()),
                "excess_true_max": float(excess_true.max()), "excess_true_mean": float(excess_true.mean()),
                "tightening_delta": float((s / (1 - gam)) / max(float(delta.max()), 1e-9)),
                "tightening_rho": float((s / (1 - gam)) / max(float(rho.max()), 1e-9)),
                "potential_iters": its, "potential_s": tpot, "rho_iters": its2, "rho_s": trho}
        for pname, pol in [("lookahead h", LookaheadGreedy(P, u, h)), ("grad-greedy tol0", GradGreedy(P, u, prefer_zero=True, tie_tol=0.0))]:
            r = rollout_terms(P, u, pol, starts, h, T, V_ref)
            gap = r["gap"]; sig = gap > 0.01
            old = r["traj"] + r["eta"] + (s + b) / (1 - gam)
            new = r["traj"] + r["eta"] + d0 + b / (1 - gam)
            newr = r["traj"] + r["eta"] + r0 + b / (1 - gam)
            row = {**base, "policy": pname, "gap_max": float(gap.max()), "traj_max": float(r["traj"].max()),
                   "bound_old_max": float(old.max()), "bound_new_max": float(new.max()), "bound_rho_max": float(newr.max()),
                   "valid_old": float((gap <= old + 1e-3).float().mean()), "valid_new": float((gap <= new + 1e-3).float().mean()),
                   "valid_rho": float((gap <= newr + 1e-3).float().mean()),
                   "ratio_old_med": float((old / gap.clamp(min=1e-9))[sig].median()) if sig.any() else float("nan"),
                   "ratio_new_med": float((new / gap.clamp(min=1e-9))[sig].median()) if sig.any() else float("nan"),
                   "ratio_rho_med": float((newr / gap.clamp(min=1e-9))[sig].median()) if sig.any() else float("nan"),
                   "ratio_old_max": float((old / gap.clamp(min=1e-9))[sig].max()) if sig.any() else float("nan"),
                   "ratio_rho_max": float((newr / gap.clamp(min=1e-9))[sig].max()) if sig.any() else float("nan"), "n_sig": int(sig.sum())}
            rows.append(row)
            print(json.dumps(row), flush=True)


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
        c = FourierMLP(1, n_freq=8, width=64, depth=3).to(dev); fn(c); crit[name] = c
    x0, _ = uniform_grid(2048, 1, dev)
    evaluate("ridge1d", P, crit, x0, pts, P.V, N)
    json.dump(rows, open("results/e15_potential.json", "w"), indent=1)

if which in ("2d", "all"):
    P = Linf2D(dev, lam=1.0)
    N = 1024
    pts, _ = uniform_grid(N, 2, dev)
    V = P.V(pts)
    crit = {"V+0.02 sin(8pi x1)sin(8pi x2)": GridFunction((V + 0.02 * torch.sin(8 * math.pi * pts[:, 0]) * torch.sin(8 * math.pi * pts[:, 1])).view(N, N))}
    for name, fn in [("SL-TD net", lambda c: train_sl_td(P, c, steps=6000, batch=8192, h=0.05, log_every=10 ** 6)),
                     ("PINN+visc net", lambda c: train_pinn(P, c, steps=6000, batch=8192, log_every=10 ** 6, visc=0.05))]:
        torch.manual_seed(1)
        c = FourierMLP(2, n_freq=12, width=128, depth=3).to(dev); fn(c); crit[name] = c
    torch.manual_seed(1)
    mb = MinBranchCritic(2, K=8, n_freq=12, width=96, depth=3).to(dev)
    train_sl_td(P, mb, steps=6000, batch=8192, h=0.05, log_every=10 ** 6); mb.eval(); crit["minK8 SL-TD"] = mb
    x0, _ = uniform_grid(48, 2, dev)
    t = torch.linspace(-0.95, 0.95, 64, device=dev)
    x0 = torch.cat([x0, torch.stack([t, t], 1), torch.stack([t, torch.zeros_like(t)], 1)], 0)
    Xg, _ = uniform_grid(512, 2, dev)
    evaluate("linf2d", P, crit, x0, Xg, P.V, 512)
    json.dump(rows, open("results/e15_potential.json", "w"), indent=1)

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
    evaluate("obstacle2d", P, crit, x0, Xg, P.V, 512, h=0.02, T=16.0)
    json.dump(rows, open("results/e15_potential.json", "w"), indent=1)
print("E15 done")
