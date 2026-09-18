"""Independent finite examples and regression checks; no neural training."""
from pathlib import Path
import json
import math
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from hjrl.certificates import (finite_center_bound, random_center_fill_bound,
                              rho_error_bound, quadratic_test_gap, rollout_tail_bound)
from hjrl.residuals import contact_residual, certificate_bound
from hjrl.contact import window_search
from hjrl.problems import Ridge1D

torch.set_num_threads(2)
ARTIFACT = {}


class ZeroCost:
    lam = 1.0
    L_V = L_ell = L_f = 0.0
    M_f = 1.0
    def __init__(self, dim=1):
        self.dim = dim
    def F(self, x, u, p):
        return u + p.abs().max(-1).values


class BoundsChecks(unittest.TestCase):
    def test_original_center_counterexample_both_signs(self):
        n, eps = 262144, 0.01
        x = -1 + 2 * torch.arange(n, dtype=torch.float64) / n
        centers = torch.tensor([[-1.0], [-0.5], [0.0], [0.5]], dtype=torch.float64)
        P = ZeroCost()
        values = []
        for critic_sign in [1, -1]:
            u = critic_sign * 0.1 * torch.sin(2 * math.pi * x).square()
            rs = [contact_residual(P, u, n, centers, eps, s, L_u=0.2*math.pi,
                                   certify=True) for s in [1, -1]]
            rb = max(float(r["r_best"].max()) for r in rs)
            rc = max(float(r["r_cert"].max()) for r in rs)
            self.assertLess(rb, 1e-12)
            old = certificate_bound(P, rc, eps)
            self.assertLess(old, 0.1)
            corrected = finite_center_bound(rc, eps, 0.25, lam=1, L_V=0,
                                            L_ell=0, L_f=0, M_f=1)
            self.assertGreaterEqual(corrected["bound"], 0.1)
            values.append({"critic_sign": critic_sign, "true_error": 0.1,
                           "population_formula_on_sampled_grid_bound": old, **corrected})
        ARTIFACT["missed_center_example"] = values

    def test_center_zero_fill_and_monotonicity(self):
        for lv in [0.0, 1.5]:
            vals = [finite_center_bound(0.2, 0.04, r, lam=0.5, L_V=lv,
                                       L_ell=2, L_f=0.4, M_f=1.3)
                    for r in [0, 1e-5, 0.01, 0.2]]
            self.assertAlmostEqual(vals[0]["center_correction"], 0)
            self.assertTrue(all(a["bound"] <= b["bound"] for a, b in zip(vals, vals[1:])))
        # The zero-dynamics edge case must not require division by M_f.
        bound = finite_center_bound(0, 0.03, 0.1, lam=1, L_V=0,
                                    L_ell=0, L_f=0, M_f=0)
        self.assertTrue(math.isfinite(bound["bound"]))
        with self.assertRaises(ValueError):
            finite_center_bound(0, 0, 0.1, lam=1, L_V=0, L_ell=0, L_f=0, M_f=1)

    def test_manufactured_1d_and_2d_critics(self):
        P = Ridge1D(torch.device("cpu"))
        n = 2048
        x = (-1 + 2*torch.arange(n, dtype=torch.float64)/n)[:, None]
        for eps in [0.015, 0.05]:
            for offset in [-0.2, 0.1]:
                u = P.V(x) + offset + 0.02*torch.sin(3*math.pi*x[:, 0])
                centers = (-1 + 2*torch.arange(32, dtype=torch.float64)/32)[:, None]
                lu = P.L_V + 0.06*math.pi
                residual = max(float(contact_residual(P, u, n, centers, eps, s,
                                      L_u=lu, certify=True)["r_cert"].max()) for s in [-1, 1])
                bound = finite_center_bound(residual, eps, 1/32, lam=P.lam, L_V=P.L_V,
                                            L_ell=P.L_ell, L_f=P.L_f, M_f=P.M_f)
                self.assertGreaterEqual(bound["bound"], abs(offset)+0.02)
        n, eps = 128, 0.02
        axis = -1 + 2*torch.arange(n, dtype=torch.float64)/n
        X, Y = torch.meshgrid(axis, axis, indexing="ij")
        u = (0.1*torch.sin(2*math.pi*X)*torch.sin(2*math.pi*Y)).flatten()
        c = torch.tensor([-1, -0.5, 0, 0.5], dtype=torch.float64)
        centers = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        P = ZeroCost(2)
        rc = max(float(contact_residual(P, u, n, centers, eps, s,
                       L_u=0.2*math.pi*math.sqrt(2), certify=True)["r_cert"].max()) for s in [-1, 1])
        b = finite_center_bound(rc, eps, math.sqrt(2)/4, lam=1, L_V=0,
                                L_ell=0, L_f=0, M_f=1)
        self.assertGreaterEqual(b["bound"], 0.1)

    def test_contact_lifts_and_missing_assumptions(self):
        n = 16
        values = torch.zeros(n, dtype=torch.float64)
        centers = torch.tensor([[0.0]], dtype=torch.float64)
        _, z, _ = window_search(values, n, 1, centers, 0.1, 1, 1.2, lifted=True)
        self.assertTrue(bool((z[..., 0] == -1).any()))
        self.assertTrue(bool((z[..., 0] == 1).any()))
        self.assertGreater(float(z.max()), 1)
        with self.assertRaises(ValueError):
            contact_residual(ZeroCost(), values, n, centers, 0.1, 1, certify=True)
        with self.assertRaises(ValueError):
            contact_residual(ZeroCost(), values, n, centers, 0.1, 1, L_u=1,
                             radius=0.01, certify=True)

    def test_grid_growth_counterexample_and_conditional_rate(self):
        a, spacing = 1e-6, 1e-4
        near_slack = (2-2*a)*spacing
        self.assertLess(a*0.9**2, near_slack)
        self.assertGreater(0.9, math.sqrt(2*near_slack))
        # Exact strongly concave objective with an off-node maximizer.
        xstar, kappa, B2, delta = 0.137, 3.0, 3.0, 0.05
        grid = np.arange(-1, 1.00001, delta)
        psi = -kappa*(grid-xstar)**2/2
        q = delta/2
        selected = grid[psi >= psi.max()-B2*q*q/2-1e-14]
        self.assertLessEqual(np.max(np.abs(selected-xstar)), math.sqrt(2*B2/kappa)*q)

    def test_quadratic_gap_and_time_scaling(self):
        c2, eps, h = math.pi**2, 1/math.pi**2, 0.1
        actual = 1+h*h/(2*eps)-math.cos(math.pi*h)
        corrected = quadratic_test_gap(h, eps, 1, C2=c2)
        self.assertGreater(actual, 0.09)
        self.assertLessEqual(actual, corrected)
        # At u=-|x|, p=0, the gap is first order; use the Lipschitz alternative.
        self.assertLessEqual(h+h*h/(2*eps), quadratic_test_gap(h, eps, 1, L_u=1))
        sums = [math.exp(-step)*quadratic_test_gap(step, eps, 1, C2=c2)/
                (1-math.exp(-step)) for step in [0.02, 0.01]]
        self.assertTrue(0.49 < sums[1]/sums[0] < 0.52)
        ARTIFACT["quadratic_gap_example"] = {"actual_gap": actual,
                                            "old_gap_upper_bound": 0,
                                            "corrected_gap_upper_bound": corrected,
                                            "halving_h_cumulative_ratio": sums[1]/sums[0]}

    def test_stationary_cap_lower_contact(self):
        w = 0.04
        c2 = (math.pi/2)**2*(1+1/w)
        eps = 0.9/c2
        x = np.linspace(-1, 1, 20001)
        u = 1-np.sqrt(np.sin(math.pi*x/2)**2+w*w)
        objective = u+x*x/(2*eps)
        self.assertEqual(int(np.argmin(objective)), 10000)
        residual_at_cap = math.pi/2+w
        self.assertGreaterEqual(residual_at_cap, math.pi/2-w)

    def test_random_center_probability_parameters(self):
        for dim in [1, 2, 4]:
            r1 = random_center_fill_bound(1000, dim, 0.05)
            r2 = random_center_fill_bound(10000, dim, 0.05)
            self.assertLessEqual(r2, r1)
            self.assertLessEqual(r1, math.sqrt(dim))
        with self.assertRaises(ValueError):
            random_center_fill_bound(3, 1, 1e-12)

    def test_clipped_iteration_and_finite_policy_bound(self):
        rng = np.random.default_rng(43)
        n, k, gam = 7, 3, 0.91
        nxt = rng.integers(n, size=(n, k))
        nxt[:, 0] = np.arange(n)
        costs = rng.uniform(0, 2, size=(n, k))
        T = lambda v: (costs+gam*v[nxt]).min(1)
        V = np.zeros(n)
        for _ in range(600):
            V = T(V)
        u = V+rng.normal(0, 2, n)
        Phi = lambda w: np.minimum(u, T(w))
        w = u.copy()
        for _ in range(600):
            w = Phi(w)
        rho = u-w
        self.assertTrue(np.all(w <= T(w)+1e-12))
        self.assertTrue(np.all(w <= V+1e-12))
        for _ in range(25):
            z = V-rng.uniform(10, 20)  # V minus a constant is a subsolution.
            if np.all(z <= u):
                self.assertTrue(np.all(z <= w+1e-12))
        W = u.copy()
        for _ in range(4):
            W = Phi(W)
        e = float(np.max(np.abs(W-Phi(W))))
        err = rho_error_bound(gam, 0, 0, 1, e)
        self.assertLessEqual(float(np.max(np.abs(W-w))), err+1e-12)
        pi = rng.integers(k, size=n)
        M = np.eye(n)
        M[np.arange(n), nxt[np.arange(n), pi]] -= gam
        J = np.linalg.solve(M, costs[np.arange(n), pi])
        tail = rollout_tail_bound(gam, 80, float(costs.max()), float(np.abs(u).max()))
        for start in range(n):
            x, total = start, 0.0
            for step in range(80):
                a = pi[x]
                eta = costs[x, a]+gam*u[nxt[x, a]]-T(u)[x]
                total += gam**step*(max(T(u)[x]-u[x], 0)+eta)
                x = nxt[x, a]
            self.assertLessEqual(J[start]-V[start], total+rho[start]+tail+1e-10)
        # Exact two-state example distinguishes rho from positive critic excess.
        u2 = np.array([0.0, -1.0])
        w2 = np.array([-gam, -1.0])
        self.assertTrue(np.allclose(np.minimum(u2, gam*w2[[1, 1]]), w2))
        ARTIFACT["rho_not_excess_example"] = {"rho_at_start": gam, "true_excess_at_start": 0}

    def test_rho_off_grid_interpolation_error(self):
        gam, spacing = 0.9, 0.25
        nodes = np.arange(-1, 1, spacing)
        u = lambda x: 0.2*np.sin(math.pi*x)
        # Zero dynamics/cost: w*=min(u,0), known exactly in the continuum.
        W = u(nodes).copy()
        for _ in range(4):
            W = np.minimum(u(nodes), gam*W)
        e = float(np.max(np.abs(W-np.minimum(u(nodes), gam*W))))
        xs = np.linspace(-1, 1, 1001, endpoint=False)
        iw = np.interp(xs, np.r_[nodes, 1], np.r_[W, W[0]])
        exact_w = np.minimum(u(xs), 0)
        bound = rho_error_bound(gam, 0.2*math.pi, spacing, 1, e)
        self.assertLessEqual(float(np.max(np.abs(iw-exact_w))), bound)

    def test_q_greedy_and_escape_margin(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            qtrue = rng.normal(size=5)
            qlearned = qtrue+rng.normal(scale=0.3, size=5)
            regret = qtrue[qlearned.argmin()]-qtrue.min()
            self.assertLessEqual(regret, 2*np.max(np.abs(qlearned-qtrue))+1e-12)
        # Label indexing alone allows regret to be positive.
        self.assertEqual(np.argmin([1.0, 0.0]), 1)
        self.assertEqual(np.argmin([0.0, 1.0]), 0)
        # Margin > 2 gamma E survives any bounded value perturbation.
        for _ in range(100):
            q = np.array([0, 0.6, 0.8])+0.9*rng.uniform(-0.2, 0.2, 3)
            self.assertEqual(int(q.argmin()), 0)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundsChecks))
    ARTIFACT["checks_run"] = result.testsRun
    ARTIFACT["successful"] = result.wasSuccessful()
    ARTIFACT["scope"] = "Analytic examples and finite numerical regression checks; not formal verification or neural retraining."
    Path(__file__).with_name("corrected_bounds_results.json").write_text(
        json.dumps(ARTIFACT, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
