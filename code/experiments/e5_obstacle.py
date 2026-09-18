"""E5: 2D torus navigation with walls (two routes around the central wall) - branch structure and policy extraction.

1. Reference V by semi-Lagrangian value iteration on a fine grid (GPU).
2. Critics trained by SL-TD: single Fourier-MLP and min-of-branches critics (K=2,4).
3. Policies: gradient-greedy, one-step lookahead, branch actor (active / reachable + hysteresis).
4. Metrics from start states on the left half and on the ridge line x2=0: performance gap vs V_ref, action switches,
   time spent inside walls, target reach rate.  Contact audit of the trained critics with the numerically estimated L_V.
"""
import sys, json, time, math
sys.path.insert(0, ".")
import torch, numpy as np
from hjrl.problems import Obstacle2D
from hjrl.solvers import sl_value_iteration
from hjrl.critic import GridFunction, FourierMLP, MinBranchCritic, eval_on_grid, lipschitz_estimate
from hjrl.torus import uniform_grid, random_points, wrap
from hjrl.train import train_sl_td, uniform_error, contact_audit
from hjrl.policy import GradGreedy, LookaheadGreedy, BranchActor, evaluate_policy
from hjrl.residuals import contact_residual, certificate_bound
from hjrl.plotting import setup, C

plt = setup()
dev = torch.device("cuda")
torch.manual_seed(0)
P = Obstacle2D(dev, lam=0.5, K=16)
out = {}

# ---------------------------------------------------------------- 1. reference solution
NREF = 512
t = time.time()
pts, Vg, delta = sl_value_iteration(P, NREF, h=0.01, iters=30000, tol=1e-7, verbose=True)
print(f"reference SL-VI: {time.time()-t:.1f}s  V range [{Vg.min():.3f},{Vg.max():.3f}]")
# refinement check against a coarser grid
pts2, Vg2, _ = sl_value_iteration(P, 256, h=0.02, iters=30000, tol=1e-7)
from hjrl.torus import periodic_interp
ref_diff = float((periodic_interp(Vg.view(NREF, NREF), pts2) - Vg2).abs().max())
print(f"ref consistency |V_512,h=0.01 - V_256,h=0.02|_inf = {ref_diff:.4f}")
P.set_reference(Vg.view(NREF, NREF))
print("estimated L_V =", P.L_V)
out["reference"] = {"n": NREF, "h": 0.01, "ref_diff_vs_256": ref_diff, "L_V": P.L_V, "Vmax": float(Vg.max())}

# contact residual of the reference itself (discrete solution of the continuous problem)
g = torch.Generator(device=dev).manual_seed(0)
y = random_points(4096, 2, dev, generator=g)
for eps in [0.02, 0.05, 0.1]:
    up = contact_residual(P, Vg, NREF, y, eps, +1, L_u=P.L_V)
    dn = contact_residual(P, Vg, NREF, y, eps, -1, L_u=P.L_V)
    print(f"reference contact residual eps={eps}: R_up={up['r_best'].max():.3f} R_dn={dn['r_best'].max():.3f} "
          f"(eps-term of bound (1) = {certificate_bound(P, 0.0, eps):.2f})")
    out[f"ref_contact_eps{eps}"] = {"R_up": float(up["r_best"].max()), "R_dn": float(dn["r_best"].max()), "eps_term": certificate_bound(P, 0.0, eps)}

# ---------------------------------------------------------------- 2. critics
critics = {}
STEPS = 12000
H_SL = 0.05
torch.manual_seed(1)
net = FourierMLP(2, n_freq=12, width=128, depth=3).to(dev)
h1 = train_sl_td(P, net, steps=STEPS, batch=8192, h=H_SL, log_every=STEPS // 10, n_eval=256)
critics["MLP"] = net
print("MLP final", h1.last(), flush=True)
for K in [2, 4]:
    torch.manual_seed(1)
    mb = MinBranchCritic(2, K=K, n_freq=12, width=96, depth=3).to(dev)
    hh = train_sl_td(P, mb, steps=STEPS, batch=8192, h=H_SL, log_every=STEPS // 10, n_eval=256)
    mb.eval()
    critics[f"min-branch K={K}"] = mb
    print(f"min-branch K={K} final", hh.last(), flush=True)
out["critics"] = {}
for name, u in critics.items():
    e_inf, e_mean = uniform_error(P, u, 512)
    aud = contact_audit(P, u, 0.05, n_centers=4096, grid_n=512, certify=False, seed=0)
    out["critics"][name] = {"err_inf": e_inf, "err_mean": e_mean, "R_up": float(aud["up"]["r_best"].max()),
                            "R_dn": float(aud["dn"]["r_best"].max()), "L_u": aud["L_u"]}
    print(name, out["critics"][name], flush=True)

# ---------------------------------------------------------------- 3./4. policies and evaluation
T = 16.0
tau = 0.02
gs = torch.Generator(device=dev).manual_seed(3)
x0 = torch.stack([-0.9 + 0.8 * torch.rand(4096, device=dev, generator=gs), -0.9 + 1.8 * torch.rand(4096, device=dev, generator=gs)], 1)
tt = torch.linspace(-0.9, -0.12, 64, device=dev)
x_ridge = torch.stack([tt, torch.zeros_like(tt)], 1)
x_ridge = torch.cat([x_ridge, torch.stack([tt, 0.01 * torch.ones_like(tt)], 1), torch.stack([tt, -0.01 * torch.ones_like(tt)], 1)], 0)


def reach_frac(xT):
    return float((P.tgt(xT) > 0.5).float().mean())


rows = []
traj_store = {}


def eval_all(name, u, pols):
    for pname, mk in pols.items():
        t0 = time.time()
        r = evaluate_policy(P, mk(), x0, tau, T, V_ref=P.V, track_obs=True)
        rr = evaluate_policy(P, mk(), x_ridge, tau, T, V_ref=P.V, track_obs=True)
        row = {"critic": name, "policy": pname, "gap_max": float(r["gap"].max()), "gap_mean": float(r["gap"].mean()),
               "gap_q90": float(r["gap"].quantile(0.9)), "frac_gap>0.1": float((r["gap"] > 0.1).float().mean()),
               "switch_mean": float(r["n_switch"].mean()), "obs_time_mean": float(r["obs_time"].mean()),
               "reach_frac": reach_frac(r["x_T"]), "ridge_gap_max": float(rr["gap"].max()), "ridge_gap_mean": float(rr["gap"].mean()),
               "ridge_switch_mean": float(rr["n_switch"].mean()), "ridge_reach_frac": reach_frac(rr["x_T"]), "time_s": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)


# optimal reference policy: lookahead on V_ref itself
eval_all("V_ref (grid)", P.V, {"lookahead h=0.02 on V_ref": lambda: LookaheadGreedy(P, P.V, 0.02)})
for name, u in critics.items():
    pols = {"grad-greedy (zero tie-break)": lambda: GradGreedy(P, u, prefer_zero=True, tie_tol=0.0),
            "grad-greedy (no zero pref)": lambda: GradGreedy(P, u, prefer_zero=False),
            "lookahead h=0.02": lambda: LookaheadGreedy(P, u, 0.02),
            "lookahead h=0.1": lambda: LookaheadGreedy(P, u, 0.1)}
    if isinstance(u, MinBranchCritic):
        pols.update({"branch: active": lambda: BranchActor(P, u, rule="active"),
                     "branch: reachable": lambda: BranchActor(P, u, rule="reachable", hysteresis=0.0, h=0.05, r_probe=0.05),
                     "branch: reachable + hyst 0.02": lambda: BranchActor(P, u, rule="reachable", hysteresis=0.02, h=0.05, r_probe=0.05)})
    eval_all(name, u, pols)
out["policies"] = rows
json.dump(out, open("results/e5_obstacle.json", "w"), indent=1)

# ---------------------------------------------------------------- figures: V_ref, trajectories, gap maps
def rollout_traj(policy, x, n_steps):
    xs = [x]
    st = None
    for _ in range(n_steps):
        idx, st = policy(xs[-1], st)
        xs.append(P.step(xs[-1], P.actions[idx], tau))
    return torch.stack(xs, 1)  # (B, n, 2)


starts = torch.tensor([[-0.5, 0.0], [-0.5, 0.02], [-0.5, -0.02], [-0.8, 0.3], [-0.3, -0.4]], device=dev)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
Vimg = Vg.view(NREF, NREF).T.cpu().numpy()
for ax, (name, mk) in zip(axes, [("V_ref lookahead", lambda: LookaheadGreedy(P, P.V, 0.02)),
                                 ("MLP grad-greedy", lambda: GradGreedy(P, critics["MLP"], prefer_zero=True)),
                                 ("min-branch K=2: reachable+hyst", lambda: BranchActor(P, critics["min-branch K=2"], rule="reachable", hysteresis=0.02, h=0.05))]):
    ax.imshow(Vimg, origin="lower", extent=[-1, 1, -1, 1], cmap="Blues", alpha=0.9)
    tr = rollout_traj(mk(), starts, 300).cpu().numpy()
    for b in range(tr.shape[0]):
        seg = tr[b]
        # break lines at wrap-around jumps
        jumps = np.where(np.abs(np.diff(seg, axis=0)).max(1) > 1.0)[0]
        prev = 0
        for jmp in list(jumps) + [len(seg) - 1]:
            ax.plot(seg[prev:jmp + 1, 0], seg[prev:jmp + 1, 1], color=C[1], lw=1.0)
            prev = jmp + 1
        ax.plot(seg[0, 0], seg[0, 1], "o", color=C[3], ms=4)
    ax.set_title(name, fontsize=9); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.grid(False)
fig.suptitle("E5: reference value (walls at x1=0,+-1; target at (0.3,0)) and sample-and-hold trajectories", fontsize=10)
fig.tight_layout(); fig.savefig("figures/e5_trajectories.png"); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
xg, _ = uniform_grid(96, 2, dev)
for ax, (name, mk) in zip(axes, [("MLP grad-greedy", lambda: GradGreedy(P, critics["MLP"], prefer_zero=True)),
                                 ("MLP lookahead h=0.02", lambda: LookaheadGreedy(P, critics["MLP"], 0.02)),
                                 ("K=2 branch active", lambda: BranchActor(P, critics["min-branch K=2"], rule="active")),
                                 ("K=2 reachable+hyst", lambda: BranchActor(P, critics["min-branch K=2"], rule="reachable", hysteresis=0.02, h=0.05))]):
    r = evaluate_policy(P, mk(), xg, tau, T, V_ref=P.V)
    im = ax.imshow(r["gap"].view(96, 96).T.cpu().numpy(), origin="lower", extent=[-1, 1, -1, 1], cmap="Oranges", vmin=0, vmax=1.0)
    ax.set_title(f"{name}: J-V (max {float(r['gap'].max()):.2f})", fontsize=8); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig("figures/e5_gapmaps.png"); plt.close(fig)
print("E5 done")
