"""E5 with 3 critic-training seeds: obstacle navigation, main policies only. Saves results/e5_seeds.json."""
import sys, json, time, math
sys.path.insert(0, ".")
import torch
from hjrl.problems import Obstacle2D
from hjrl.solvers import sl_value_iteration
from hjrl.critic import FourierMLP, MinBranchCritic
from hjrl.train import train_sl_td, uniform_error, contact_audit
from hjrl.policy import GradGreedy, LookaheadGreedy, BranchActor, evaluate_policy

dev = torch.device("cuda")
torch.manual_seed(0)
P = Obstacle2D(dev, lam=0.5, K=16)
pts, Vg, _ = sl_value_iteration(P, 512, h=0.01, iters=30000, tol=1e-7)
P.set_reference(Vg.view(512, 512))
STEPS, H_SL, tau, T = 12000, 0.05, 0.02, 16.0
gs = torch.Generator(device=dev).manual_seed(3)
x0 = torch.stack([-0.9 + 0.8 * torch.rand(4096, device=dev, generator=gs), -0.9 + 1.8 * torch.rand(4096, device=dev, generator=gs)], 1)
tt = torch.linspace(-0.9, -0.12, 64, device=dev)
x_ridge = torch.cat([torch.stack([tt, torch.zeros_like(tt)], 1), torch.stack([tt, 0.01 * torch.ones_like(tt)], 1),
                     torch.stack([tt, -0.01 * torch.ones_like(tt)], 1)], 0)
rows = []
for seed in [1, 2, 3]:
    crits = {}
    torch.manual_seed(seed)
    net = FourierMLP(2, n_freq=12, width=128, depth=3).to(dev)
    train_sl_td(P, net, steps=STEPS, batch=8192, h=H_SL, seed=seed, log_every=STEPS, n_eval=256)
    crits["MLP"] = net
    for K in [2, 4]:
        torch.manual_seed(seed)
        mb = MinBranchCritic(2, K=K, n_freq=12, width=96, depth=3).to(dev)
        train_sl_td(P, mb, steps=STEPS, batch=8192, h=H_SL, seed=seed, log_every=STEPS, n_eval=256)
        mb.eval()
        crits[f"minK{K}"] = mb
    torch.save({k: v.state_dict() for k, v in crits.items()}, f"results/e5_critics_seed{seed}.pt")
    for name, u in crits.items():
        e_inf, e_mean = uniform_error(P, u, 512)
        aud = contact_audit(P, u, 0.05, n_centers=4096, grid_n=512, certify=False, seed=seed)
        diag = {"err_inf": e_inf, "err_mean": e_mean, "R_up": float(aud["up"]["r_best"].max()), "R_dn": float(aud["dn"]["r_best"].max())}
        pols = {"grad-greedy": lambda: GradGreedy(P, u, prefer_zero=True, tie_tol=0.0),
                "lookahead h=0.02": lambda: LookaheadGreedy(P, u, 0.02)}
        if isinstance(u, MinBranchCritic):
            pols["branch reachable+hyst"] = lambda: BranchActor(P, u, rule="reachable", hysteresis=0.02, h=0.05, r_probe=0.05)
        for pname, mk in pols.items():
            r = evaluate_policy(P, mk(), x0, tau, T, V_ref=P.V, track_obs=True)
            rr = evaluate_policy(P, mk(), x_ridge, tau, T, V_ref=P.V, track_obs=True)
            row = {"seed": seed, "critic": name, **diag, "policy": pname, "gap_max": float(r["gap"].max()), "gap_mean": float(r["gap"].mean()),
                   "frac_gap>0.1": float((r["gap"] > 0.1).float().mean()), "switch_mean": float(r["n_switch"].mean()),
                   "reach_frac": float((P.tgt(r["x_T"]) > 0.5).float().mean()), "ridge_gap_max": float(rr["gap"].max()),
                   "ridge_gap_mean": float(rr["gap"].mean()), "ridge_reach_frac": float((P.tgt(rr["x_T"]) > 0.5).float().mean()),
                   "ridge_frac>0.2": float((rr["gap"] > 0.2).float().mean())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open("results/e5_seeds.json", "w"), indent=1)
print("E5 seeds done")
