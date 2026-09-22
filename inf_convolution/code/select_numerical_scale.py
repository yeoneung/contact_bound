"""Select a scale solely from whole-domain certificates for a computed spline.

This module has no access to the reference solution or the actual error.
Admissibility is proved from rational spline coefficients. Numerical bounds
use Arb enclosures and adaptive subdivision, not sampled maxima.
"""
from fractions import Fraction
from pathlib import Path
import argparse,hashlib,json,time
from flint import arb,ctx
from numerical_input import Spline,quadratic_extrema,running_cost_arb
from interval_utils import ball,interval,upper_record,lower_record

ROOT=Path(__file__).resolve().parents[1]
ROOT_BITS=110


class Inadmissible(Exception):pass


def contact_geometry(spline,eps):
    n=spline.n
    f=lambda x:x+eps*spline.point(x,1)
    if 1+eps*spline.second[0]>=0:
        raise Inadmissible('No strict contact bifurcation at the origin.')
    first=next((j for j in range(1,n+1) if f(Fraction(j,n))>0),None)
    if first is None:raise Inadmissible('No positive contact crossing.')
    root_cell=first-1
    for i in range(root_cell):
        a,b,c,d=spline.coeffs[i]
        coeff=(Fraction(i,n)+eps*n*b,Fraction(1,n)+eps*n*2*c,eps*n*3*d)
        # At i=0 the left endpoint is zero. A negative initial derivative
        # and nonpositive quadratic maximum prove negativity in its interior.
        mn,mx=quadratic_extrema(coeff)
        if mx>0 or mn>=0 or (i>0 and mx>=0):
            raise Inadmissible('A pre-contact cell was not strictly negative.')
    left,right=Fraction(root_cell,n),Fraction(root_cell+1,n)
    if root_cell==0:
        _,b,c,d=spline.coeffs[0]
        a1=Fraction(1,n)+2*eps*n*c;a2=3*eps*n*d
        if not(a1<0 and a2>0):raise Inadmissible('First-cell crossing is not simple.')
        root=-a1/a2/n
        if not(0<root<right):raise Inadmissible('First-cell root outside the cell.')
        lo=hi=root
    else:
        if not(f(left)<0<f(right)):raise Inadmissible('Root lies on a knot or has wrong signs.')
        lo,hi=left,right
        while hi-lo>Fraction(1,1<<ROOT_BITS):
            mid=(lo+hi)/2;v=f(mid)
            if v<0:lo=mid
            elif v>0:hi=mid
            else:lo=hi=mid;break
    # A quadratic with opposite endpoint signs has exactly one interior zero.
    # Checking these linear derivative conditions covers the entire tail.
    min_j=None;min_twice=None
    for i in range(root_cell,n):
        a=max(lo,Fraction(i,n));b=Fraction(i+1,n)
        for x in (a,b):
            h=spline.local_value(i,x*n-i,2)
            j=1+eps*h;t=1+2*eps*h
            min_j=j if min_j is None else min(min_j,j)
            min_twice=t if min_twice is None else min(min_twice,t)
    if min_j<=0 or min_twice<=0:raise Inadmissible('The reconstructed branch is not admissible.')
    return dict(root_cell=root_cell,root_lower=[lo.numerator,lo.denominator],
                root_upper=[hi.numerator,hi.denominator],
                jacobian_lower=lower_record(ball(min_j)),
                derivative_factor_lower=lower_record(ball(min_twice)))


def evaluate_cell(spline,eps,left,right,i):
    x=interval(left,right);ee=ball(eps)
    u,du,hu=spline.local_arb(i,x)
    jl=1+ee*hu
    if not jl>0:raise ArithmeticError('Interval Jacobian not positive; subdivide.')
    # The contact coordinate map is already proved increasing.
    ul,dul,_=spline.local_arb(i,ball(left));ur,dur,_=spline.local_arb(i,ball(right))
    yl=ball(left)+ee*dul;yr=ball(right)+ee*dur
    y=yl.union(yr).max(arb(0)).min(arb(1))
    w=u+ee*du**2;dw=du*(1+2*ee*hu)/jl
    residual=w+abs(dw)-running_cost_arb(y)
    distance=w-spline.range(y)
    return residual,distance


def certify(spline,exponent,relative_tolerance=Fraction(1,50)):
    started=time.perf_counter();eps=Fraction(1,1<<exponent)
    geometry=contact_geometry(spline,eps)
    lo=Fraction(*geometry['root_lower']);hi=Fraction(*geometry['root_upper'])
    pending=[]
    for i in range(geometry['root_cell'],spline.n):
        pending.append((max(lo,Fraction(i,spline.n)),Fraction(i+1,spline.n),i,0))
    rlow=arb(0);dlow=arb(0);rup=arb(0);dup=arb(0);leaves=0;evaluations=0;depth_max=0
    # Establish global lower bounds before refining a nearly flat endpoint.
    for left,right,i,_ in pending:
        mid=(left+right)/2
        rp,dp=evaluate_cell(spline,eps,mid,mid,i)
        rlow=rlow.max(abs(rp).abs_lower());dlow=dlow.max(abs(dp).abs_lower())
    # Midpoint evaluations only guide refinement; they are never used as upper bounds.
    rtol=ball(relative_tolerance);atol=arb(1)/10**12
    while pending:
        left,right,i,depth=pending.pop();mid=(left+right)/2
        if mid<hi:raise ArithmeticError('Subdivision reached the root enclosure width.')
        rp,dp=evaluate_cell(spline,eps,mid,mid,i)
        rlow=rlow.max(abs(rp).abs_lower());dlow=dlow.max(abs(dp).abs_lower())
        try:
            rr,dd=evaluate_cell(spline,eps,left,right,i)
            ru=abs(rr).upper();du=abs(dd).upper()
            resolved=(ru<=rlow*(1+rtol)+atol and du<=dlow*(1+rtol)+atol)
        except ArithmeticError:
            resolved=False
        evaluations+=1
        if resolved:
            rup=rup.max(ru);dup=dup.max(du);leaves+=1;depth_max=max(depth_max,depth)
        else:
            if evaluations>300000 or depth>28:
                raise ArithmeticError(f'Verification budget exceeded: {i}, {depth}, {evaluations}, R={ru}, D={du}, lower={rlow}, {dlow}')
            pending.append((left,mid,i,depth+1));pending.append((mid,right,i,depth+1))
    rho=upper_record(rup);distance=upper_record(dup)
    total=Fraction(rho['numerator']+distance['numerator'],rho['denominator'])
    return dict(exponent=exponent,epsilon_numerator=1,epsilon_denominator=1<<exponent,
                geometry=geometry,residual_upper=rho,distance_upper=distance,
                error_upper=upper_record(ball(total)),leaves=leaves,evaluations=evaluations,
                maximum_depth=depth_max,elapsed_seconds=time.perf_counter()-started)


def select(record,exponents=range(2,15)):
    started=time.perf_counter();spline=Spline(record);candidates=[]
    for j in exponents:
        try:
            c=certify(spline,j);c['status']='certified';candidates.append(c)
            print(f"N={spline.n}, eps=2^-{j}: B <= {c['error_upper']['decimal']:.7g}, cells={c['leaves']}",flush=True)
        except Inadmissible as error:
            candidates.append(dict(exponent=j,status='inadmissible',reason=str(error)))
    valid=[c for c in candidates if c['status']=='certified']
    if not valid:raise RuntimeError('No scale was certified.')
    best=min(valid,key=lambda c:Fraction(c['error_upper']['numerator'],c['error_upper']['denominator']))
    return dict(n=spline.n,input_sha256=record['input_sha256'],candidates=candidates,
                selected_exponent=best['exponent'],selected_error_upper=best['error_upper'],
                selection_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',nargs='+',type=int,default=[64,128,256,512,1024])
    args=parser.parse_args();ctx.prec=192
    records=[]
    for n in args.n:
        path=ROOT/f'results/numerical_inputs/n{n}.json'
        record=json.loads(path.read_text());records.append(select(record))
    report=dict(status='selected_from_rigorous_bounds_without_reference_error',
                exponents=list(range(2,15)),selection_rule='minimum rational certified upper bound',
                relative_subdivision_tolerance=[1,50],absolute_subdivision_tolerance='1e-12',
                precision_bits=192,root_width_bits=ROOT_BITS,
                cost='2 + cos(pi*x) + cos(2*pi*x)/5',records=records)
    report['selection_source_sha256']={name:hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()
                                       for name in ['numerical_input.py','select_numerical_scale.py','interval_utils.py']}
    path=ROOT/'results/numerical_selection.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'results/numerical_selection.sha256').write_text(sha+'\n')
    print('Selection frozen before reference evaluation:',sha,flush=True)


if __name__=='__main__':main()
