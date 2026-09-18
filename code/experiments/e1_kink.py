"""E1: 1D kink discrimination on the torus.

Part A: analytic a.e. solutions of  lam u + |u'| = g  (true V, family W_c = V - c w with a spurious valley at +-1,
        and the mirrored solution with a valley at 0).  Compare AD residual, Bellman residual, contact residual.
Part B: PINN (AD loss) vs SL-TD training from many seeds: does AD training land on spurious kinks, and does the
        contact residual flag them?
"""
import sys, json, time, math
sys.path.insert(0, ".")
import torch, numpy as np
from hjrl.problems import Ridge1D
from hjrl.solvers import spurious_family_1d, spurious_1d
from hjrl.critic import GridFunction, FourierMLP, eval_on_grid, lipschitz_estimate
from hjrl.residuals import ad_residual, bellman_residual, contact_residual, certificate_bound
from hjrl.torus import random_points, uniform_grid
from hjrl.train import train_pinn, train_sl_td, uniform_error
from hjrl.plotting import setup, C

plt = setup()
dev = torch.device("cuda")
torch.manual_seed(0)
P = Ridge1D(dev, lam=1.0)
N = 8192
pts, delta = uniform_grid(N, 1, dev)
centers = pts.clone()                       # every grid point is a center
EPS = [0.02, 0.05, 0.2]
out = {"partA": [], "partB": []}


def residual_table(name, W, L_u):
    Wf = GridFunction(W.view(-1))
    xs = random_points(16384, 1, dev)
    ad = ad_residual(P, Wf, xs)
    row = {"name": name, "err_inf": float((W - P.V(pts)).abs().max()),
           "ad_median": float(ad.median()), "ad_q99": float(ad.quantile(0.99)), "ad_max": float(ad.max())}
    for h in [0.01, 0.05]:
        be = bellman_residual(P, Wf, pts, h)
        row[f"bellman_max_h{h}"] = float(be.abs().max())
    for eps in EPS:
        up = contact_residual(P, W, N, centers, eps, +1, L_u=L_u, certify=True)
        dn = contact_residual(P, W, N, centers, eps, -1, L_u=L_u, certify=True)
        row[f"Rup_eps{eps}"] = float(up["r_best"].max())
        row[f"Rdn_eps{eps}"] = float(dn["r_best"].max())
        row[f"Rcert_eps{eps}"] = float(max(up["r_cert"].max(), dn["r_cert"].max()))
        row[f"bound_best_eps{eps}"] = certificate_bound(P, max(row[f"Rup_eps{eps}"], row[f"Rdn_eps{eps}"]), eps)
        row[f"bound_cert_eps{eps}"] = certificate_bound(P, row[f"Rcert_eps{eps}"], eps)
    return row, Wf


# ------------------------------------------------------------------ Part A
cases = [("V (viscosity solution)", P.V(pts), P.L_V)]
for c in [0.05, 0.2, 1.0]:
    _, W = spurious_family_1d(P, N, c)
    cases.append((f"W_c, c={c} (spurious valley at +-1)", W, P.L_V + P.lam * c))
_, Wm, defect, cf = spurious_1d(P, N, [(-1.0, -1), (0.0, +1)])
cases.append(("mirror (spurious valley at 0)", Wm, 30.0))
profiles = {}
for name, W, L_u in cases:
    t = time.time()
    row, Wf = residual_table(name, W, L_u)
    row["time_s"] = time.time() - t
    out["partA"].append(row)
    print(json.dumps(row))
    if name.startswith("W_c, c=0.2") or name.startswith("V ") or name.startswith("mirror"):
        # residual profiles vs x (for the figure)
        eps = 0.05
        up = contact_residual(P, W, N, centers, eps, +1, L_u=L_u)
        dn = contact_residual(P, W, N, centers, eps, -1, L_u=L_u)
        profiles[name] = {"x": pts[:, 0].cpu().numpy(), "W": W.cpu().numpy(), "V": P.V(pts).cpu().numpy(),
                          "ad": ad_residual(P, Wf, pts).cpu().numpy(),
                          "bellman": bellman_residual(P, Wf, pts, 0.01).abs().cpu().numpy(),
                          "r_up": up["r_best"].cpu().numpy(), "r_dn": dn["r_best"].cpu().numpy()}

# ------------------------------------------------------------------ Part B: training from seeds
SEEDS = list(range(10))
learned = {"pinn": [], "sl_td": []}
for method in ["pinn", "sl_td"]:
    for seed in SEEDS:
        torch.manual_seed(seed)
        c = FourierMLP(1, n_freq=8, width=64, depth=3).to(dev)
        t = time.time()
        if method == "pinn":
            hist = train_pinn(P, c, steps=4000, batch=4096, lr=1e-3, seed=seed, log_every=1000)
        else:
            hist = train_sl_td(P, c, steps=4000, batch=4096, h=0.02, lr=1e-3, seed=seed, log_every=1000)
        _, vals, _ = eval_on_grid(c, N, 1, dev)
        L_u = lipschitz_estimate(c, N, 1, dev)
        xs = random_points(16384, 1, dev)
        ad = ad_residual(P, c, xs)
        row = {"method": method, "seed": seed, "err_inf": float((vals - P.V(pts)).abs().max()),
               "err_mean": float((vals - P.V(pts)).abs().mean()), "ad_mean": float(ad.mean()), "ad_max": float(ad.max()),
               "bellman_max_h0.01": float(bellman_residual(P, c, pts, 0.01).abs().max()), "L_u": L_u, "time_s": time.time() - t}
        for eps in EPS:
            up = contact_residual(P, vals, N, centers, eps, +1, L_u=L_u)
            dn = contact_residual(P, vals, N, centers, eps, -1, L_u=L_u)
            row[f"Rup_eps{eps}"] = float(up["r_best"].max())
            row[f"Rdn_eps{eps}"] = float(dn["r_best"].max())
            row[f"bound_best_eps{eps}"] = certificate_bound(P, max(row[f"Rup_eps{eps}"], row[f"Rdn_eps{eps}"]), eps)
        out["partB"].append(row)
        learned[method].append(vals.cpu().numpy())
        print(json.dumps(row))

json.dump(out, open("results/e1_kink.json", "w"), indent=1)

# ------------------------------------------------------------------ figures
x = pts[:, 0].cpu().numpy()
fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
for j, (name, pr) in enumerate(profiles.items()):
    ax = axes[0, j]
    ax.plot(x, pr["V"], color=C[0], label="V")
    if not name.startswith("V "):
        ax.plot(x, pr["W"], color=C[1], label="W")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("x"); ax.legend(loc="lower center", ncol=2)
    ax = axes[1, j]
    ax.plot(x, pr["ad"], color=C[3], label="AD |F(x,u,u')|")
    ax.plot(x, pr["bellman"], color=C[2], label="Bellman |u-T_h u|/h, h=0.01")
    ax.plot(x, pr["r_up"], color=C[0], label="contact upper [F]_+, eps=0.05")
    ax.plot(x, pr["r_dn"], color=C[1], label="contact lower [-F]_+, eps=0.05")
    ax.set_yscale("symlog", linthresh=1e-3); ax.set_xlabel("x (or center y)")
    if j == 0:
        ax.legend(fontsize=7, loc="upper center")
axes[1, 0].set_ylabel("residual")
fig.suptitle("E1: a.e. solutions of lam u + |u'| = g on the torus - which residual sees the wrong kink?", fontsize=10)
fig.tight_layout(); fig.savefig("figures/e1_profiles.png"); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
V = P.V(pts).cpu().numpy()
for j, method in enumerate(["pinn", "sl_td"]):
    ax = axes[j]
    ax.plot(x, V, color=INK if False else "#0b0b0b", lw=2.2, label="V")
    for k, vals in enumerate(learned[method]):
        ax.plot(x, vals, color=C[0] if method == "pinn" else C[2], lw=0.8, alpha=0.6, label=f"{method} seeds" if k == 0 else None)
    ax.set_title(f"{method}: 10 seeds", fontsize=9); ax.legend(); ax.set_xlabel("x")
ax = axes[2]
for method, col in [("pinn", C[0]), ("sl_td", C[2])]:
    rows = [r for r in out["partB"] if r["method"] == method]
    ax.scatter([r["err_inf"] for r in rows], [max(r["Rup_eps0.05"], r["Rdn_eps0.05"]) for r in rows], color=col, s=22, label=method)
ax.set_xlabel("||u - V||_inf"); ax.set_ylabel("contact residual R_eps (eps=0.05)"); ax.set_xscale("log"); ax.set_yscale("log")
ax.legend(); ax.set_title("residual vs true error", fontsize=9)
fig.tight_layout(); fig.savefig("figures/e1_learned.png"); plt.close(fig)
print("E1 done")
