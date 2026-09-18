"""Complete rational-arithmetic value certificate on a nonsmooth periodic example.

Run from submission/: python code/experiments/complete_contact_certificate.py
Only the standard library is needed. No trained network or reference solver is used.
All contact comparisons and residuals are exact integers/rationals; the square root
in the center correction is enclosed from above by a rational with denominator 10^12.
The timer covers contact enumeration, localization and the full error budget.
"""
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

ROOT = Path(__file__).resolve().parents[2]
# On [j,j+1], u=x^2+(B/5)x+D/5; u=(1-|x|)^2-|x|/5 on [-1,1].
BRANCHES = ((-2, 9, 3), (-1, 11, 5), (0, -11, 5), (1, -9, 3))
CASES = ((512, 16), (4096, 32), (32768, 64), (262144, 128), (2097152, 256))


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal": float(value)}


def sqrt_upper(value, digits=12):
    scale = 10**digits
    k = isqrt(value.numerator * scale * scale // value.denominator)
    upper = F(k + 1, scale)
    assert upper * upper >= value
    assert F(k, scale)**2 <= value
    return upper


def enumerate_contacts(n, m, phase=1):
    """Uniform centers (-n+2j+phase)/n; epsilon=1/m, location mesh 2/(8n).

    Each signed contact objective is strictly concave on each quadratic branch.
    Its clamped stationary point is the branch maximizer. Compare all four
    objective values exactly and retain every tie. Periodicity ensures every
    global contact has a nearest representative in [-2,2] for centers in [-1,1].
    """
    assert n > 0 and m > 2 and m % 2 == 0 and phase in (0, 1)
    mesh = 8 * n
    rounded_max = 0
    exact_max = F(0)
    max_distance = F(0)
    ties = 0
    for sigma in (-1, 1):
        q0 = 5 * (m - 2 * sigma)
        qden = n * q0
        exact_num = 0
        largest_displacement = 0
        for j in range(n):
            yn = -n + 2 * j + phase
            best = None
            winners = []
            for left, b, d in BRANCHES:
                xn = max(left*qden, min((left+1)*qden,
                                       5*m*yn + sigma*b*n))
                # Common positive objective denominator 5*qden^2.
                objective = (sigma*(5*xn*xn+b*xn*qden+d*qden*qden)
                             - (5*m//2)*(xn-yn*q0)**2)
                if best is None or objective > best:
                    best, winners = objective, [xn]
                elif objective == best and xn not in winners:
                    winners.append(xn)
            ties += len(winners) > 1
            for xn in winners:
                largest_displacement = max(largest_displacement, abs(xn-yn*q0))
                sn = abs((xn+qden) % (2*qden)-qden)
                fn = -10*qden+9*sn+5*m*abs(xn-yn*q0)
                exact_num = max(exact_num, sigma*fn)
                # Nearest mesh node; all quantities remain integers.
                node = (xn*mesh+qden)//(2*qden)
                xhat = 2*node
                assert abs(xhat*qden-xn*mesh) <= qden
                shat = abs((xhat+mesh) % (2*mesh)-mesh)
                fhat = -10*mesh+9*shat+5*m*abs(xhat-8*yn)
                rounded_max = max(rounded_max, sigma*fhat)
        exact_max = max(exact_max, F(exact_num, 5*qden))
        max_distance = max(max_distance, F(largest_displacement, qden))
    location_error = F(31+5*m, 5*mesh)  # Kq, K=Lu+Lell+1/eps, q=1/mesh
    rounded_residual = F(rounded_max, 5*mesh)
    assert exact_max <= rounded_residual+location_error
    assert max_distance <= F(22, 5*m)  # 2 Lu epsilon
    return dict(exact_sampled_residual=exact_max,
                rounded_residual=rounded_residual, location_error=location_error,
                center_residual_upper=rounded_residual+location_error,
                largest_contact_displacement=max_distance, tied_centers=ties)


def run_case(n, m):
    start = time.perf_counter()
    contact = enumerate_contacts(n, m)
    r, eps = F(1, n), F(1, m)
    zeta = 6*r+r*r/(2*eps)
    # lambda=1, LV=2, Lell=4, Lf=0, Mf=1, CV=16.
    center = 4*r+zeta+2*sqrt_upper(2*zeta*(6+m))
    scale = 16*eps
    bound = contact["center_residual_upper"]+scale+center
    assert bound >= F(1, 5)
    elapsed = time.perf_counter()-start
    values = {k: rational(v) if isinstance(v, F) else v for k, v in contact.items()}
    values.update({k: rational(v) for k, v in {
        "epsilon": eps, "fill_distance": r, "comparison_error": scale,
        "center_error_upper": center, "complete_bound_upper": bound,
        "true_error": F(1, 5), "effectivity_upper": 5*bound,
    }.items()})
    values.update(centers=n, location_mesh_nodes=8*n, reciprocal_scale=m,
                  branch_objective_evaluations=8*n, seconds=elapsed)
    return values


def independent_checks():
    """Fraction-based oracle, independent of the scaled-integer formulas.

    Check several phases/resolutions, every sign and all ties. Also verify the
    known residual at the valley and the viscosity inequalities at the cusp.
    """
    count = 0
    for n, m in ((7, 8), (16, 16), (31, 32)):
        for phase in (0, 1):
            ref = F(0)
            for j in range(n):
                y = F(-n+2*j+phase, n)
                for sigma in (-1, 1):
                    candidates = []
                    for left, b, d in BRANCHES:
                        x = max(F(left), min(F(left+1),
                                (y+F(sigma*b, 5*m))/(1-F(2*sigma, m))))
                        u = x*x+F(b, 5)*x+F(d, 5)
                        objective = sigma*u-F(m, 2)*(x-y)**2
                        s = abs((x+1) % 2-1)
                        g = (1-s)**2+2*(1-s)
                        residual = max(F(0), sigma*(u-g+abs(m*(x-y))))
                        candidates.append((objective, residual))
                    best = max(o for o, _ in candidates)
                    ref = max(ref, *(v for o, v in candidates if o == best))
            got = enumerate_contacts(n, m, phase)
            assert got["exact_sampled_residual"] == ref
            count += 1
    assert enumerate_contacts(16, 16, 0)["exact_sampled_residual"] == F(1, 5)
    # V(0)=1, g(0)=3, all upper-test slopes lie in [-2,2].
    assert all(1+abs(F(k, 10))-3 <= 0 for k in range(-20, 21))
    return {"independent_fraction_oracle_cases": count,
            "valley_residual_exact": "1/5", "checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checks = independent_checks()
    if args.check_only:
        print(json.dumps(checks, indent=2))
        return
    records = []
    for n, m in CASES:
        row = run_case(n, m)
        records.append(row)
        print(f"N={n}: bound={row['complete_bound_upper']['decimal']:.6f}, "
              f"ratio={row['effectivity_upper']['decimal']:.4f}, {row['seconds']:.3f}s", flush=True)
    output = {"problem": "QuadraticRidge1D", "critic": "(1-|x|)^2-|x|/5",
              "constants": {"lambda": 1, "L_V": 2, "L_ell": 4, "L_f": 0,
                            "M_f": 1, "L_u": "11/5", "C_V": 16},
              "center_phase": "half-cell shift; y_j=-1+(2j+1)/N",
              "arithmetic": "Exact integers and fractions; square root enclosed upward at 12 decimal places",
              "integration_error": 0, "reference_error": 0,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "processor": platform.processor()},
              "checks": checks, "records": records}
    path = ROOT/"results/complete_contact_certificate.json"
    path.write_text(json.dumps(output, indent=2)+"\n", encoding="utf8")
    print(path)


if __name__ == "__main__":
    main()
