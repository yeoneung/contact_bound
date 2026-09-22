"""Whole-period certification of automatically reconnected contact branches.

Every active branch is smooth. A strict dominance check removes artificial
branch endpoints from the envelope. Convexity of |p| then supplies the
viscosity subsolution inequality at every new junction; the supersolution
inequality follows by testing any active smooth branch.
"""
from fractions import Fraction as F
import time
from flint import arb
from interval_utils import ball,interval,upper_record,lower_record
from asymmetric_input import cost_arb
from automatic_branches import discover,BranchFailure,components

def branch_at(branch,y,kind):return branch.jet(y,y,branch.part(y),kind)

def branch_range(branch,left,right,kind):
    cuts=[left]+[e for e in branch.edges if left<e<right]+[right];out=None
    for a,b in zip(cuts,cuts[1:]):
        z=branch.jet(a,b,branch.part((a+b)/2),kind)
        out=z if out is None else tuple(v.union(w) for v,w in zip(out,z))
    return out

def continuity(branches,kind,periodic):
    if periodic:return dict(periodic_single_branch=True,endpoints=0)
    events=sorted({e for b in branches for e in [b.left,b.right] if 0<=e<2})
    margin=arb(10**6)
    for y in events:
        changes=[b for b in branches if y in [b.left,b.right]]
        through=[b for b in branches if b.left<y<b.right]
        if not through:raise BranchFailure('The retained contact images leave a gap.')
        vals=[branch_at(b,y,kind)[0] for b in through]
        incumbent=vals[0]
        for v in vals[1:]:incumbent=incumbent.min(v)
        for b in changes:
            gap=branch_at(b,y,kind)[0]-incumbent
            if not gap>0:raise BranchFailure('A branch endpoint is not strictly dominated after reconstruction.')
            margin=margin.min(gap.lower())
    # Domain endpoints partition the circle. Coverage is checked on every
    # resulting open interval as well as at the domain endpoints above.
    edges=sorted(set([F(0),F(2)]+events))
    for a,b in zip(edges,edges[1:]):
        m=(a+b)/2
        if not any(c.left<=m<=c.right for c in branches):raise BranchFailure('Uncovered center interval.')
    return dict(periodic_single_branch=False,endpoints=len(events),dominance_margin_lower=lower_record(margin))

def global_contacts(spline,eps,branches):
    """Exclude every omitted stationary contact or show negative second variation."""
    threshold=F(1,2);h=spline.h;pending=[];excluded=dominated=0
    for i in range(spline.n):
        a,b=i*h,(i+1)*h;jl=1+eps*spline.local_exact(i,F(0),2);jr=1+eps*spline.local_exact(i,F(1),2)
        if min(jl,jr)>=threshold:continue
        if max(jl,jr)>threshold:
            cross=a+h*(threshold-jl)/(jr-jl)
            if jl>=threshold:a=cross
            else:b=cross
        pending.append((a,b,i,0))
    ee=ball(eps);iterations=0
    while pending:
        a,b,i,depth=pending.pop();iterations+=1
        x=interval(a,b);u,p,H,_=spline.jet(i,x);J=1+ee*H
        if J<0:excluded+=1;continue
        y=x+ee*p;lq=y.lower().fmpq();hq=y.upper().fmpq()
        lo=F(int(lq.numerator),int(lq.denominator));hi=F(int(hq.numerator),int(hq.denominator))
        choices=[c for c in branches if c.left<=lo<=hi<=c.right]
        passed=False
        for c in choices:
            try:
                q,qp,_,_=branch_range(c,lo,hi,'uncorrected')
                mid=(a+b)/2;um,pm,_,_=spline.jet(i,ball(mid));ym=mid+eps*spline.point(mid,1)
                vm,_,_,_=branch_at(c,ym,'uncorrected')
                center=um+ee*pm**2/2-vm
                derivative=J*(p-qp)
                difference=center+interval(-(b-a)/2,(b-a)/2)*derivative
                if difference>0:passed=True;break
            except ArithmeticError:pass
        if passed:dominated+=1;continue
        if depth>=32 or iterations>=100000:raise BranchFailure('Completeness of the retained contact branches was not proved.')
        m=(a+b)/2;pending.extend([(a,m,i,depth+1),(m,b,i,depth+1)])
    return dict(omitted_negative_curvature_cells=excluded,omitted_dominated_cells=dominated,evaluations=iterations)

def initial_partition(branches):
    edges=sorted(set([F(0),F(2)]+[e for b in branches for e in b.edges if 0<e<2]))
    out=[]
    for a,b in zip(edges,edges[1:]):
        m=(a+b)/2;active=[(c,c.part(m)) for c in branches if c.left<=a<b<=c.right]
        if not active:raise BranchFailure('No smooth branch covers a center interval.')
        out.append((a,b,active,0))
    return out

def envelope_point(spline,entries,y,kind,values=None):
    if values is None:values=[c.jet(y,y,p,kind) for c,p in entries]
    w=values[0][0]
    for v in values[1:]:w=w.min(v[0])
    # Use a residual sample only when the active branch is unambiguous.
    winner=next((v for v in values if all(v is other or v[0]<other[0] for other in values)),None)
    distance=w-ball(spline.point(y))
    r=None if winner is None else winner[0]+abs(winner[1])-cost_arb(ball(y),spline.case)
    return distance,r

def enclose(spline,a,b,entries,kind,centers=None):
    mid=(a+b)/2;rad=interval(-(b-a)/2,(b-a)/2);y=interval(a,b)
    jets=[c.jet(a,b,p,kind) for c,p in entries]
    if centers is None:centers=[c.jet(mid,mid,p,kind) for c,p in entries]
    possible=[]
    for k,v in enumerate(jets):
        dominated=False
        for j,other in enumerate(jets):
            if j==k:continue
            difference=centers[k][0]-centers[j][0]+rad*(v[1]-other[1])
            if difference>0:dominated=True;break
        if not dominated:possible.append(k)
    if not possible:raise ArithmeticError('Inconsistent dominance enclosures.')
    residual=arb(0);distance=arb(0);u0=ball(spline.point(mid));du=spline.range(a,b,1)
    gm=cost_arb(ball(mid),spline.case);dg=cost_arb(y,spline.case,1)
    for k in possible:
        v,p,second,_=jets[k];vm,pm,_,_=centers[k]
        r0=vm+abs(pm)-gm
        sign=arb(1) if p>=0 else (arb(-1) if p<=0 else interval(F(-1),F(1)))
        dr=p+sign*second-dg
        rr=abs(r0+rad*dr).upper()
        dd=arb(0) if getattr(entries[k][0],'identical_input',False) else abs(vm-u0+rad*(p-du)).upper()
        residual=residual.max(rr);distance=distance.max(dd)
    return residual,distance,len(possible)

def certify(spline,exponent,kind,relative=F(1,50),absolute=F(1,10**10),eps=None,target=None):
    assert kind in ['uncorrected','corrected']
    started=time.perf_counter();eps=F(1,2**exponent) if eps is None else F(eps);geometry_started=started
    try:
        branches,periodic,count=discover(spline,eps)
        qc=continuity(branches,'uncorrected',periodic)
        completeness=global_contacts(spline,eps,branches)
        wc=qc if kind=='uncorrected' else continuity(branches,kind,periodic)
        pending=initial_partition(branches)
    except (BranchFailure,ArithmeticError) as e:
        return dict(status='not_certified',kind=kind,exponent=exponent,epsilon=[eps.numerator,eps.denominator],reason=str(e),total_seconds=time.perf_counter()-started)
    geometry_seconds=time.perf_counter()-geometry_started
    result=certify_norms(spline,pending,kind,relative,absolute,target)
    return dict(status='above_target' if result.get('target_excluded') else 'certified',kind=kind,exponent=exponent,
                epsilon=[eps.numerator,eps.denominator],branch_components=count,
                contact_completeness=completeness,uncorrected_continuity=qc,envelope_continuity=wc,
                **result,geometry_seconds=geometry_seconds,total_seconds=time.perf_counter()-started)

def certify_norms(spline,pending,kind,relative=F(1,50),absolute=F(1,10**10),target=None):
    rlow=arb(0);dlow=arb(0);rup=arb(0);dup=arb(0);leaves=evaluations=unresolved=0;maximum_depth=0
    saved_centers={}
    for a,b,entries,_ in pending:
        m=(a+b)/2;centers=[c.jet(m,m,p,kind) for c,p in entries];saved_centers[a,b]=centers
        d,r=envelope_point(spline,entries,m,kind,centers)
        if r is not None:rlow=rlow.max(abs(r).abs_lower())
        dlow=dlow.max(abs(d).abs_lower())
    if target is not None and rlow+dlow>ball(target):
        return dict(target_excluded=True,error_lower=lower_record(rlow+dlow),
                    residual_lower=lower_record(rlow),distance_lower=lower_record(dlow),seed_cells=len(pending))
    rel=ball(relative);atol=ball(absolute)
    while pending:
        a,b,entries,depth=pending.pop();evaluations+=1;mid=(a+b)/2
        centers=saved_centers.pop((a,b),None)
        if centers is None:centers=[c.jet(mid,mid,p,kind) for c,p in entries]
        d,r=envelope_point(spline,entries,mid,kind,centers)
        if r is not None:rlow=rlow.max(abs(r).abs_lower())
        dlow=dlow.max(abs(d).abs_lower())
        try:
            ru,dd,active=enclose(spline,a,b,entries,kind,centers)
            passed=ru<=rlow*(1+rel)+atol and dd<=dlow*(1+rel)+atol
        except ArithmeticError:passed=False
        if passed:
            rup=rup.max(ru);dup=dup.max(dd);leaves+=1;unresolved+=active>1;maximum_depth=max(maximum_depth,depth)
        else:
            if depth>=36 or evaluations>=200000:raise ArithmeticError(f'Norm subdivision budget exceeded: {a}, {b}, {depth}, {evaluations}.')
            pending.extend([(a,mid,entries,depth+1),(mid,b,entries,depth+1)])
    rho=upper_record(rup);dist=upper_record(dup)
    B=F(rho['numerator'],rho['denominator'])+F(dist['numerator'],dist['denominator'])
    return dict(residual_upper=rho,distance_upper=dist,error_upper=upper_record(ball(B)),
                residual_lower=lower_record(rlow),distance_lower=lower_record(dlow),
                leaves=leaves,evaluations=evaluations,junction_cells=unresolved,maximum_depth=maximum_depth)

def certify_polynomial(spline,width,degree=3,target=None):
    from direct_branch_polynomials import build
    started=time.perf_counter()
    try:pending,peaks=build(spline,width,degree)
    except BranchFailure as e:
        return dict(status='not_certified',kind='polynomial',width=width,degree=degree,reason=str(e),total_seconds=time.perf_counter()-started)
    geometry_seconds=time.perf_counter()-started
    result=certify_norms(spline,pending,'polynomial',target=target)
    return dict(status='above_target' if result.get('target_excluded') else 'certified',kind='polynomial',width=width,degree=degree,numerical_peaks=peaks,
                **result,geometry_seconds=geometry_seconds,total_seconds=time.perf_counter()-started)
