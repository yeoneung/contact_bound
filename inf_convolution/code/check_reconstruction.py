"""Independent checks of identities, proof constants, and certificate replay.

These checks do not replace the proof or interval coverage. Direct residuals
are computed independently with mpmath, avoiding the cancellation formulas
used by the verifier. All saved certificates are then replayed in Arb.
"""
from pathlib import Path
from fractions import Fraction
import json
import mpmath as mp
from flint import arb,ctx
from verify_reconstruction import (ball,interval,root_bracket,stable_quantities,
                                   certify,PRECISION)

ROOT=Path(__file__).resolve().parents[1]


def exact_mp(q):
    q=Fraction(q)
    return mp.mpf(q.numerator)/q.denominator


def contains_mp(enclosure,value):
    # Enclose the independently computed decimal, allowing its 90-digit
    # working-precision error far below the precision of the tested cells.
    reference=arb(mp.nstr(value,85), '1e-78')
    return enclosure.contains(reference)


def direct(x,delta,profile):
    if profile=='sine':
        h=lambda z:mp.sin(mp.pi*z/2)
    else:
        h=lambda z:z-z*z/2
    u=lambda z:1-mp.sqrt(h(z)**2+delta**2)
    eps=mp.sqrt(delta)
    du=mp.diff(u,x);hu=mp.diff(u,x,2)
    y=x+eps*du
    w=u(x)+eps*du**2
    dw=du*(1+2*eps*hu)/(1+eps*hu)
    g=1-h(y)+mp.diff(h,y)
    return w+abs(dw)-g,w-u(y),w-(1-h(y)),dw+mp.diff(h,y)


def main():
    mp.mp.dps=90;ctx.prec=PRECISION
    report=json.loads((ROOT/'results/reconstruction.json').read_text())
    checks=[];identity_count=0
    for row in report['records']:
        profile=row['profile'];d=Fraction(1,row['delta_denominator'])
        lo,hi=root_bracket(ball(d),ball(d).sqrt(),profile)
        assert lo==Fraction(*row['root_lower']) and hi==Fraction(*row['root_upper'])
        # Interior points include the contact boundary layer and the smooth tail.
        for t in (Fraction(1,1000),Fraction(1,100),Fraction(1,4),Fraction(3,4)):
            x=hi+(1-hi)*t
            radius=min(Fraction(1,10**12),x/10,(1-x)/10)
            enclosure=stable_quantities(interval(x-radius,x+radius),ball(d),ball(d).sqrt(),profile)
            values=direct(exact_mp(x),exact_mp(d),profile)
            assert all(contains_mp(a,b) for a,b in zip(enclosure,values)),(profile,d,x)
            identity_count+=4
        replay=certify(row['delta_power'],row['subdivisions_per_doubling'],profile)
        for name in ('residual_upper','distance_upper','error_upper',
                     'reconstruction_effectivity_upper','contact_error_lower','contact_effectivity_lower'):
            assert replay[name]==row[name],(profile,row['delta_power'],name)
    checks.append(dict(name='independent_direct_residual_identities',count=identity_count,status='PASS'))
    checks.append(dict(name='root_brackets_and_all_rational_certificates_replayed',count=len(report['records']),status='PASS'))
    k=arb.pi()/2
    a0=k**4/2+(1+k*k)/20
    a1=k/2+k**3*(1+arb(1)/200)+k**5/(1-k*k/100)+k**5/2
    assert a0<arb(7)/2 and a1<21 and 2*a0+a1+1<30
    assert k*(1-k*k/200)/(1+arb(1)/100).sqrt()>1
    checks.append(dict(name='uniform_sinusoidal_proof_constants',status='PASS'))
    # The generic hypotheses also cover a profile not used by the verifier.
    h=lambda x:x-x**3/3
    delta=mp.mpf('1e-7');eps=mp.sqrt(delta)
    u=lambda x:1-mp.sqrt(h(x)**2+delta**2)
    x0=mp.findroot(lambda x:x+eps*mp.diff(u,x),(eps/2,2*eps))
    assert x0>eps/2 and x0<=eps
    for j in range(1,100):
        x=x0+(1-x0)*j/100
        assert 1+2*eps*mp.diff(u,x,2)>0
    checks.append(dict(name='additional_concave_profile_contact_diagnostic',status='PASS',
                       note='Numerical diagnostic; the general result rests on its analytic proof.'))
    (ROOT/'verification').mkdir(parents=True,exist_ok=True)
    (ROOT/'verification/checks.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n')
    for c in checks:print(c['name'],c['status'])


if __name__=='__main__':main()
