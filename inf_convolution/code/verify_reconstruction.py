"""Arb certification of the corrected inf-convolution in the manuscript.

The verification covers the full interval, with exact rational cell endpoints.
It does not substitute sampled maxima for suprema. The nonsmooth join is
covered by the analytic viscosity argument in the manuscript.
"""
from pathlib import Path
from fractions import Fraction
from math import factorial
import argparse
import json
import time
import flint
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
PRECISION = 192
ROOT_BITS = 150
EXPORT_BITS = 100


def ball(q):
    q = Fraction(q)
    return arb(q.numerator) / q.denominator


def interval(lo, hi):
    return arb(ball((lo + hi)/2), ball((hi-lo)/2))


def ceil_int(v):
    return int(v.upper().ceil().unique_fmpz())


def floor_int(v):
    return int(v.lower().floor().unique_fmpz())


def upper_record(v):
    n = ceil_int(v * (1 << EXPORT_BITS))
    return dict(numerator=n, denominator=1 << EXPORT_BITS,
                decimal=n/(1 << EXPORT_BITS))


def lower_record(v):
    n = floor_int(v * (1 << EXPORT_BITS))
    return dict(numerator=n, denominator=1 << EXPORT_BITS,
                decimal=n/(1 << EXPORT_BITS))


def data(x, delta, profile='sine'):
    if profile == 'quadratic':
        s, ds = x-x*x/2, 1-x
        a = (s*s+delta*delta).sqrt()
        p = ds*s/a
        h = s/a-delta*delta*ds*ds/a**3
        return arb(1), s, ds, a, p, h
    assert profile == 'sine'
    k = arb.pi()/2
    s, c = (k*x).sin_cos()
    a = (s*s+delta*delta).sqrt()
    p = k*s*c/a
    h = k*k*(s*s/a-delta*delta*c*c/(a*a*a))
    return k, s, c, a, p, h


def root_bracket(delta, eps, profile='sine'):
    den = 1 << ROOT_BITS
    lo = Fraction(floor_int(eps*den/4), den)
    hi = Fraction(1)
    def f(x):
        return ball(x)-eps*data(ball(x), delta, profile)[4]
    assert f(lo) < 0 and f(hi) > 0
    while hi-lo > Fraction(1, den):
        mid = (lo+hi)/2
        value = f(mid)
        if value < 0:
            lo = mid
        elif value > 0:
            hi = mid
        else:
            raise ArithmeticError('Increase precision to decide root bracket.')
    assert f(lo) < 0 and f(hi) > 0
    return lo, hi


def theta_minus_sin(theta):
    # Taylor polynomial through degree 19 with a rigorous degree-21 remainder.
    t2 = theta*theta
    p = arb(0)
    for n in range(19, 2, -2):
        p = p*t2 + arb((-1)**((n-3)//2))/factorial(n)
    main = theta**3*p
    remainder = abs(theta)**21/factorial(21)
    return main + arb(0, remainder.upper())


def stable_quantities(x, delta, eps, profile='sine'):
    k, s, c, a, p, h = data(x, delta, profile)
    jacobian = 1+eps*h
    assert jacobian > 0 and 1+2*eps*h > 0
    e1 = k*c*delta**2/(a*(a+s))
    if profile == 'sine':
        theta = k*eps*p
        cm1 = -2*(theta/2).sin()**2
        tms = theta_minus_sin(theta)
        h_error = k*k*(s*delta**2/(a*(a+s))+delta**2*c*c/a**3)
        value_difference = -delta**2/(a+s)+s*cm1+c*tms-eps*p*e1
        gradient_difference = (e1+k*c*cm1-k*s*tms
                               +eps*p*h_error+eps**2*p*h*h/jacobian)
    else:
        h_error = delta**2/(a*(a+s))+delta**2*c*c/a**3
        value_difference = -delta**2/(a+s)-eps*p*e1-eps**2*p*p/2
        gradient_difference = e1+eps*p*h_error+eps**2*p*h*h/jacobian
    residual = value_difference-gradient_difference
    # Only y in [0,1] is needed. Clipping discards the tiny extension below
    # the root bracket, never an actual point of the reconstructed branch.
    y = (x-eps*p).max(arb(0)).min(arb(1))
    sy = (k*y).sin() if profile == 'sine' else y-y*y/2
    # The smoothing error decreases with the nonnegative sine value.
    # Endpoint evaluation avoids interval dependency in sqrt(s^2+d^2)-s.
    sy_lo = sy.lower() if sy.lower() > 0 else arb(0)
    sy_hi = sy.upper()
    def bias(t):
        return delta**2/((t*t+delta**2).sqrt()+t)
    smoothing_error = bias(sy_hi).union(bias(sy_lo))
    distance = value_difference+smoothing_error
    return residual, distance, value_difference, gradient_difference


def partition(start, subdivisions):
    left = start
    while left < 1:
        right = min(2*left, Fraction(1))
        for j in range(subdivisions):
            yield (left+(right-left)*j/subdivisions,
                   left+(right-left)*(j+1)/subdivisions)
        left = right


def certify(power, subdivisions=256, profile='sine'):
    started = time.perf_counter()
    ctx.prec = PRECISION
    delta_fraction = Fraction(1, 10**power)
    delta = ball(delta_fraction)
    eps = delta.sqrt()
    lo, hi = root_bracket(delta, eps, profile)
    rho, distance = arb(0), arb(0)
    count = 0
    for left, right in partition(lo, subdivisions):
        r, d, _, _ = stable_quantities(interval(left,right), delta, eps, profile)
        rho = rho.max(abs(r).upper())
        distance = distance.max(abs(d).upper())
        count += 1
    # Exact rational upward-rounded values form the saved certificate.
    rho_record, distance_record = upper_record(rho), upper_record(distance)
    total = Fraction(rho_record['numerator']+distance_record['numerator'],
                     1 << EXPORT_BITS)
    total_record = upper_record(ball(total))
    kappa = arb.pi()/2 if profile == 'sine' else arb(1)
    floor = arb(3)/4*(2*kappa)**(arb(1)/3)*delta**(arb(2)/3)
    assert ball(total) >= delta
    assert ball(total) < 30*delta
    return dict(profile=profile,delta_power=power, delta_numerator=1, delta_denominator=10**power,
                epsilon_rule='sqrt(delta)',
                root_lower=[lo.numerator,lo.denominator],
                root_upper=[hi.numerator,hi.denominator],
                subdivisions_per_doubling=subdivisions, cells=count,
                residual_upper=rho_record, distance_upper=distance_record,
                error_upper=total_record,
                reconstruction_effectivity_upper=upper_record(ball(total)/delta),
                contact_error_lower=lower_record(floor),
                contact_effectivity_lower=lower_record(floor/delta),
                elapsed_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subdivisions',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'results/reconstruction.json')
    args=parser.parse_args()
    rows=[]
    for power in (2,3,4,5,6,7,8):
        row=certify(power,args.subdivisions);rows.append(row)
        print(f"delta=1e-{power}: contact floor={row['contact_effectivity_lower']['decimal']:.6f}; "
              f"reconstruction upper={row['reconstruction_effectivity_upper']['decimal']:.6f}; "
              f"cells={row['cells']}; seconds={row['elapsed_seconds']:.3f}",flush=True)
    for power in (4,6,8):
        row=certify(power,args.subdivisions,'quadratic');rows.append(row)
        print(f"quadratic delta=1e-{power}: reconstruction upper="
              f"{row['reconstruction_effectivity_upper']['decimal']:.6f}",flush=True)
    report=dict(status='rigorous_interval_enclosures',arithmetic='Arb',
                python_flint_version=flint.__version__,precision_bits=PRECISION,
                root_width_bits=ROOT_BITS,export_denominator_bits=EXPORT_BITS,
                scope='One-dimensional periodic family in Theorems 1 and 2; analytic viscosity join proof required.',
                records=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
