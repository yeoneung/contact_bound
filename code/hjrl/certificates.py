"""Explicit constants in the corrected contact and policy bounds.

These scalar formulas require verified input upper bounds. They do not turn
sampled Lipschitz estimates, inexact contact searches, or floating-point output
into verified data. Historical experiment JSON files are left unchanged.
"""
import math


def _nonnegative(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def finite_center_bound(residual, eps, fill_distance, *, lam, L_V, L_ell, L_f, M_f):
    """Lemma (finite centers); residual must upper-bound both signs at all centers."""
    _nonnegative(residual=residual, fill_distance=fill_distance,
                 L_V=L_V, L_ell=L_ell, L_f=L_f, M_f=M_f)
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(lam) or lam <= 0:
        raise ValueError("eps and lam must be finite and positive")
    r = fill_distance
    D = 2 * L_V * eps + r
    P = D / eps
    K0 = L_ell + L_f * P
    zeta = 3 * L_V * r + r * r / (2 * eps)
    A0 = K0 + lam * L_V + M_f / eps
    population_term = 2 * L_V * (L_ell + 2 * L_f * L_V) * eps
    correction = K0 * D - population_term + lam * zeta + 2 * math.sqrt(2 * M_f * zeta * A0)
    return {"bound": (residual + population_term + correction) / lam,
            "population_term": population_term, "center_correction": correction,
            "zeta": zeta, "D": D, "P": P, "K0": K0, "A0": A0}


def random_center_fill_bound(n, dim, failure_probability):
    """Euclidean fill-distance upper bound for N uniform centers on [-1,1)^d."""
    if not isinstance(n, int) or n < 3 or not isinstance(dim, int) or dim < 1:
        raise ValueError("n >= 3 and dim >= 1 must be integers")
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must be in (0,1)")
    log_term = math.log(n) - math.log(failure_probability)
    if log_term > n:
        raise ValueError("The stated random-center proposition requires log(n/alpha) <= n")
    return min(math.sqrt(dim), 4 * math.sqrt(dim) * (log_term / n) ** (1 / dim))


def rho_error_bound(gamma, L_star, spacing, dim, iteration_residual,
                    *, at_nodes=False, operator_error=0.0):
    """Upper bound for |u-IW - rho|, using a fixed-point residual, not an iterate difference."""
    _nonnegative(L_star=L_star, spacing=spacing, iteration_residual=iteration_residual,
                 operator_error=operator_error)
    if not 0 <= gamma < 1 or not isinstance(dim, int) or dim < 1:
        raise ValueError("gamma must be in [0,1), dim a positive integer")
    interpolation = L_star * spacing * math.sqrt(dim) * (gamma if at_nodes else 1)
    return (iteration_residual + operator_error + interpolation) / (1 - gamma)


def quadratic_test_gap(h, eps, M_f, *, L_u=None, C2=None):
    """Upper gap at a successor, conditional on a valid upper test.

    C2 invokes a two-sided Hessian bound and p=Du(x); L_u invokes the general
    Lipschitz bound for p in D+u(x). Return the smaller applicable upper bound.
    """
    _nonnegative(h=h, M_f=M_f)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    bounds = []
    if L_u is not None:
        _nonnegative(L_u=L_u)
        bounds.append(2 * L_u * M_f * h + M_f * M_f * h * h / (2 * eps))
    if C2 is not None:
        _nonnegative(C2=C2)
        bounds.append((1 / eps + C2) * M_f * M_f * h * h / 2)
    if not bounds:
        raise ValueError("A Lipschitz or two-sided Hessian upper bound is required")
    return min(bounds)


def rollout_tail_bound(gamma, steps, cost_bound, critic_bound):
    """Tail for the finite telescoping policy bound."""
    _nonnegative(cost_bound=cost_bound, critic_bound=critic_bound)
    if not 0 <= gamma < 1 or not isinstance(steps, int) or steps < 0:
        raise ValueError("gamma must be in [0,1), steps a nonnegative integer")
    return gamma ** steps * (cost_bound / (1 - gamma) + critic_bound)
