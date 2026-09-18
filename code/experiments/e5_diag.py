"""E5 diagnostic: retrain the min-branch K=4 critic (same seed/config as e5_obstacle.py), save critics, and inspect the
worst ridge start under the branch/gradient policies (where does the gap 0.86 come from?)."""
import sys, json, time, math
sys.path.insert(0, ".")
import torch
from hjrl.problems import Obstacle2D
from hjrl.solvers import sl_value_iteration
from hjrl.critic import FourierMLP, MinBranchCritic
from hjrl.train import train_sl_td, uniform_error
from hjrl.policy import GradGreedy, LookaheadGreedy, BranchActor, evaluate_policy
dev = torch.device("cuda")
torch.manual_seed(0)
P = Obstacle2D(dev, lam=0.5, K=16)
pts, Vg, delta = sl_value_iteration(P, 512, h=0.01, iters=30000, tol=1e-7)
P.set_reference(Vg.view(512, 512))
STEPS, H_SL = 12000, 0.05
crits = {}
torch.manual_seed(1)
net = FourierMLP(2, n_freq=12, width=128, depth=3).to(dev)
train_sl_td(P, net, steps=STEPS, batch=8192, h=H_SL, log_every=STEPS, n_eval=256)
crits["MLP"] = net
for K in [2, 4]:
    torch.manual_seed(1)
    mb = MinBranchCritic(2, K=K, n_freq=12, width=96, depth=3).to(dev)
    train_sl_td(P, mb, steps=STEPS, batch=8192, h=H_SL, log_every=STEPS, n_eval=256)
    mb.eval()
    crits[f"minK{K}"] = mb
torch.save({k: v.state_dict() for k, v in crits.items()}, "results/e5_critics.pt")
for k, v in crits.items():
    print(k, "err_inf/mean", uniform_error(P, v, 512), flush=True)

u = crits["minK4"]
tau, T = 0.02, 16.0
tt = torch.linspace(-0.9, -0.12, 64, device=dev)
x_ridge = torch.cat([torch.stack([tt, torch.zeros_like(tt)], 1), torch.stack([tt, 0.01 * torch.ones_like(tt)], 1),
                     torch.stack([tt, -0.01 * torch.ones_like(tt)], 1)], 0)
for pname, pol in [("grad-greedy", GradGreedy(P, u, prefer_zero=False)), ("branch reachable+hyst", BranchActor(P, u, rule="reachable", hysteresis=0.02, h=0.05)),
                   ("lookahead h=0.02", LookaheadGreedy(P, u, 0.02))]:
    r = evaluate_policy(P, pol, x_ridge, tau, T, V_ref=P.V)
    j = int(r["gap"].argmax())
    x0 = x_ridge[j:j + 1]
    print(f"{pname}: worst ridge start {x0.tolist()} gap={float(r['gap'][j]):.3f}  n_bad(gap>0.2)={int((r['gap']>0.2).sum())}", flush=True)
    # trace the trajectory of the worst start
    x = x0.clone(); st = None; traj = []
    for t in range(int(T / tau)):
        idx, st = pol(x, st)
        with torch.no_grad():
            b = u.branches(x)[0].tolist()
        traj.append((round(float(x[0, 0]), 3), round(float(x[0, 1]), 3), int(idx[0]), [round(v, 3) for v in b], int(st[0]) if st is not None else -1))
        x = P.step(x, P.actions[idx], tau)
    # print first 40 steps and a sample of later steps
    for row in traj[:40]:
        print("   ", row)
    print("    ... later:", traj[100], traj[300], traj[600], flush=True)
