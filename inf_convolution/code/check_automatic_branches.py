"""Independent differentiation, global contacts, search replay and junction audit."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys
import mpmath as mp
import numpy as np
from flint import ctx,arb
from interval_utils import ball,interval,upper_record,lower_record
from asymmetric_input import construct,PeriodicSpline,solve,cost
from automatic_branches import discover,float_envelope
from certify_automatic_branches import branch_at,branch_range
from benchmark_automatic_branches import run

ROOT=Path(__file__).resolve().parents[1]

def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if not k.endswith('seconds') and k!='repetition'}
    if isinstance(value,list):return [stable(v) for v in value]
    return value

def mf(q):q=F(q);return mp.mpf(q.numerator)/q.denominator

def independent_jet(spline,eps):
    branches,_,_=discover(spline,eps);b=max(branches,key=lambda c:float(min(c.right,F(2))-max(c.left,F(0))))
    count=0
    for part in sorted(set([len(b.pieces)//3,2*len(b.pieces)//3])):
        a,c,i=b.pieces[part];x=(a+c)/2;y=x+eps*spline.point(x,1)
        co=spline.record['coefficient_numerators'];scale=mp.mpf(2)**spline.record['coefficient_bits']
        p,q,r,s=[mp.mpf(co[j%spline.n])/scale for j in [i-1,i,i+1,i+2]];h=mf(spline.h);ee=mf(eps)
        def u(xx):
            t=xx/h-i
            return (p*(1-t)**3+q*(4-6*t*t+3*t**3)+r*(1+3*t+3*t*t-3*t**3)+s*t**3)/6
        def inverse(yy):return mp.findroot(lambda xx:xx+ee*mp.diff(u,xx)-yy,(mf(x)-h/100,mf(x)+h/100))
        for kind,factor in [('uncorrected',mp.mpf('.5')),('corrected',mp.mpf(1))]:
            def value(yy):
                xx=inverse(yy);return u(xx)+factor*ee*mp.diff(u,xx)**2
            z=b.jet(y,y,part,kind)
            for order in range(3):
                reference=mp.diff(value,mf(y),order)
                # Some values are exact dyadic Arb points. A truncated decimal
                # cannot be contained in such a zero-radius point, so compare
                # with the independently computed decimal plus its rounding band.
                with ctx.workprec(256):reference_ball=arb(mp.nstr(reference,70),arb('1e-68'))
                assert z[order].overlaps(reference_ball) and z[order].rad()<arb('1e-45'),(kind,order,z[order],reference)
            count+=1
    return count

def independent_global_minimum(spline,eps):
    branches,_,_=discover(spline,eps);ee=mf(eps);h=mf(spline.h)
    for j in range(1,10):
        y=F(2*j,11);ym=mf(y);values=[]
        for i in range(-spline.n,2*spline.n):
            a,b,c,d=map(mf,spline.coeffs[i%spline.n]);A=3*ee*d/h;B=h+2*ee*c/h;C=i*h+ee*b/h-ym
            if A==0:roots=[] if B==0 else [-C/B]
            else:
                disc=B*B-4*A*C;roots=[] if disc<0 else [(-B+mp.sqrt(disc))/(2*A),(-B-mp.sqrt(disc))/(2*A)]
            for t in roots:
                if 0<=t<=1:
                    x=(i+t)*h;values.append(a+t*(b+t*(c+t*d))+(x-ym)**2/(2*ee))
        assert values
        exact=min(values);available=[b for b in branches if b.left<=y<=b.right]
        enclosed=branch_at(available[0],y,'uncorrected')[0]
        for b in available[1:]:enclosed=enclosed.min(branch_at(b,y,'uncorrected')[0])
        with ctx.workprec(256):reference_ball=arb(mp.nstr(exact,70),arb('1e-68'))
        assert enclosed.overlaps(reference_ball) and enclosed.rad()<arb('1e-45'),(spline.case,y,enclosed,exact)
    return 9

def junctions(spline,eps):
    branches,_,_=discover(spline,eps);y=np.arange(16385)*2/16384;out={}
    for kind in ['uncorrected','corrected']:
        _,_,which=float_envelope(branches,y,kind);events=[]
        for k in np.flatnonzero(which[:-1]!=which[1:]):
            left,right=F(int(k),8192),F(int(k+1),8192);a,b=branches[which[k]],branches[which[k+1]]
            if not(a.left<=left<=right<=a.right and b.left<=left<=right<=b.right):continue
            za=branch_range(a,left,right,kind);zb=branch_range(b,left,right,kind)
            if not za[1]-zb[1]>0:continue
            def difference(z):return branch_at(a,z,kind)[0]-branch_at(b,z,kind)[0]
            if not(difference(left)<0 and difference(right)>0):continue
            while right-left>F(1,2**85):
                mid=(left+right)/2;v=difference(mid)
                if v<0:left=mid
                elif v>0:right=mid
                else:raise ArithmeticError('Junction root requires greater precision.')
            za=branch_range(a,left,right,kind);zb=branch_range(b,left,right,kind)
            # Every other candidate is strictly above the two joining branches.
            roof=za[0].union(zb[0])
            for other in branches:
                if other is a or other is b:continue
                if other.left<=left<=right<=other.right:assert branch_range(other,left,right,kind)[0]>roof
            event=dict(lower=lower_record(ball(left)),upper=upper_record(ball(right)),
                       left_derivative=upper_record(za[1]),right_derivative=upper_record(zb[1]))
            if kind=='uncorrected':
                ca=branch_range(a,left,right,'corrected')[0];cb=branch_range(b,left,right,'corrected')[0]
                jump=abs(ca-cb);assert jump>0
                event['old_junction_value_jump_lower']=lower_record(jump)
                event['old_junction_value_jump_upper']=upper_record(jump)
            events.append(event)
        out[kind]=events
    assert out['uncorrected'] and len(out['uncorrected'])==len(out['corrected'])
    return out

def main():
    ctx.prec=192;mp.mp.dps=75;path=ROOT/'results/automatic_branches.json'
    data=json.loads(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==path.with_suffix('.sha256').read_text().strip()
    assert data['status']=='completed_certified_target_study'
    for name,expected in data['source_sha256'].items():assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected,name
    assert len(data['records'])==54 and data['repetitions']==3
    groups={}
    for row in data['records']:groups.setdefault((row['case'],row['method'],tuple(row['target'])),[]).append(row)
    attempts=completed=excluded=geometry_failures=0
    for key,rows in groups.items():
        assert len(rows)==3 and all(stable(r)==stable(rows[0]) for r in rows[1:])
        case,method,target=key;replayed=run(case,method,F(*target),max_n=max(data['grids']))
        assert stable(replayed)==stable(rows[0]),key
        assert rows[0]['status']=='target_certified'
        for step in rows[0]['history']:
            for c in step['attempts']:
                attempts+=1;completed+=c['status']=='certified';excluded+=c['status']=='above_target';geometry_failures+=c['status']=='not_certified'
        print('Replayed automatic search',case,method,target,'N',rows[0]['selected_n'],flush=True)
    points=global_points=0
    for case in data['cases']:
        # Independent synchronous iteration checks the label-setting input solver.
        n=64;g=cost(2*np.arange(n)/n,case);u=g.copy();h=2/n
        for _ in range(4*n):u=np.minimum(g,np.minimum((h*g+np.roll(u,1))/(1+h),(h*g+np.roll(u,-1))/(1+h)))
        assert np.max(abs(u-solve(n,case)))<1e-13
        s=PeriodicSpline(construct(256,case));eps=F(1,32)
        points+=independent_jet(s,eps);global_points+=independent_global_minimum(s,eps)
    s=PeriodicSpline(construct(256,'skew'));events=junctions(s,F(1,32))
    jr=dict(status='rigorous_junction_enclosures',benchmark_sha256=sha,case='skew',n=256,epsilon=[1,32],events=events)
    (ROOT/'results/automatic_junctions.json').write_text(json.dumps(jr,indent=2)+'\n')
    report=dict(status='PASS',benchmark_sha256=sha,replayed_searches=len(groups),timing_repetitions_checked=54,
                unique_attempts=attempts,certificates_replayed=completed,target_exclusions_replayed=excluded,
                geometry_failures_replayed=geometry_failures,independent_jet_points=points,
                independent_global_contact_points=global_points,independent_precision_digits=75,
                independent_node_solvers=3,junctions_replayed=len(events['corrected']))
    (ROOT/'verification/automatic_branch_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
