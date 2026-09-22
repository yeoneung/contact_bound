"""Interval hypotheses and independent 80-digit checks for asymmetric branches.

The pointwise derivative/root comparisons supplement the analytic proof. They
are not used as whole-domain numerical certificates in the benchmark tables.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import mpmath as mp
from flint import arb,ctx
from interval_utils import ball,interval,upper_record,lower_record

ROOT=Path(__file__).resolve().parents[1]
def mf(q):q=F(q);return mp.mpf(q.numerator)/q.denominator
def rational(r):return F(r['numerator'],r['denominator'])

def main():
    ctx.prec=192;mp.mp.dps=80;beta=F(1,20);bb=ball(beta);bm=mf(beta)
    xi=mp.findroot(lambda x:2*x-3*bm*x*x-6*bm,mp.mpf('.15'))
    center=F(int(mp.floor(xi*2**120)),2**120);rows=[];points=0
    for power in (8,12,16):
        delta=F(1,2**power);eps=F(1,2**(power//2));dd=ball(delta);ee=ball(eps)
        left=(center-4*eps,center-eps);right=(center+eps,center+4*eps)
        alpha=[arb(0),arb(0),arb(0)];L=arb(0);M=arb(0);j0=arb(10)
        def jet(x):
            d=2*x-3*bb*x*x-6*bb;dp=2-6*bb*x;dpp=-6*bb
            m=2-x*x+bb*x**3+6*bb*x;mp1=-2*x+3*bb*x*x+6*bb;mp2=-2+6*bb*x
            radius=(d*d+dd*dd).sqrt();p=mp1-d*dp/radius
            H=mp2-d*dpp/radius-dd*dd*dp*dp/radius**3
            return m-radius,p,H,d,dp,dpp,radius
        for lo,hi in (left,right):
            for k in range(256):
                x=interval(lo+(hi-lo)*k/256,lo+(hi-lo)*(k+1)/256)
                u,p,H,d,dp,dpp,r=jet(x);sign=arb(-1) if hi<center else arb(1);s=sign*d
                assert s>0
                defects=(-dd*dd/(r+s),sign*dd*dd*dp/(r*(r+s)),sign*dd*dd*dpp/(r*(r+s))-dd*dd*dp*dp/r**3)
                alpha=[a.max(abs(v).upper()) for a,v in zip(alpha,defects)]
                L=L.max(abs(p).upper());M=M.max(abs(H).upper());j0=j0.min((1+ee*H).lower())
        assert j0>0
        K2=ball(2+9*beta);K3=ball(6*beta)
        eta0=alpha[0]+ee*L*alpha[1]+K2*ee*ee*L*L/2
        eta1=alpha[1]+ee*L*alpha[2]+ee*ee*(L*M*M/j0+K3*L*L/2)
        window=(center-eps/2,center+eps/2);gamma=(4-12*bb*interval(*window)).lower()
        for lo,hi in (left,right):
            assert ball(lo)+ee*jet(ball(lo))[1]<ball(window[0])
            assert ball(hi)+ee*jet(ball(hi))[1]>ball(window[1])
        assert 2*eta0<gamma*ball(eps/2)-arb('1e-30') and 2*eta1<gamma
        dm=mf(delta);em=mf(eps)
        def dmp(x):return 2*x-3*bm*x*x-6*bm
        def mmp(x):return 2-x*x+bm*x**3+6*bm*x
        def ump(x):return mmp(x)-mp.sqrt(dmp(x)**2+dm**2)
        functions=[];observed=[mp.mpf(0),mp.mpf(0)]
        for side,ends in [(-1,left),(1,right)]:
            def make_branch(side):
                def w(y):
                    x=mp.findroot(lambda x:x+em*mp.diff(ump,x)-y,(xi+side*2*em,xi+side*mp.mpf('2.1')*em))
                    return ump(x)+em*mp.diff(ump,x)**2
                return w
            w=make_branch(side);functions.append(w)
            v=lambda x:mmp(x)-side*dmp(x)
            for k in range(1,8):
                x=mf(ends[0])+(mf(ends[1])-mf(ends[0]))*k/8;y=x+em*mp.diff(ump,x)
                for order,bound in enumerate((eta0,eta1)):
                    difference=abs(mp.diff(w,y,order)-mp.diff(v,y,order));observed[order]=max(observed[order],difference)
                    assert arb(mp.nstr(difference,65))<bound
                points+=1
        junction=mp.findroot(lambda y:functions[0](y)-functions[1](y),(xi-em/4,xi+em/4))
        shift=abs(junction-xi);bound=2*eta0/gamma
        assert arb(mp.nstr(shift,65))<bound
        rows.append(dict(delta=[delta.numerator,delta.denominator],epsilon=[eps.numerator,eps.denominator],
            alpha_upper=[upper_record(v) for v in alpha],L_upper=upper_record(L),M_upper=upper_record(M),
            jacobian_lower=lower_record(j0),eta0_upper=upper_record(eta0),eta1_upper=upper_record(eta1),
            slope_separation_lower=lower_record(gamma),junction_shift_upper=upper_record(bound),
            independent_observed_value_error=mp.nstr(observed[0],30),independent_observed_derivative_error=mp.nstr(observed[1],30),
            independent_junction_displacement=mp.nstr(shift,30)))
    report=dict(status='PASS',precision_bits=192,independent_precision_digits=80,independent_branch_points=points,
        independent_junctions=3,meaning='Interval bounds verify the hypotheses and constants. Pointwise inverse differentiation and root comparisons are auxiliary checks of the analytic stability estimate.',records=rows)
    (ROOT/'verification/branch_stability_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
