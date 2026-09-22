"""Post-refinement ranking, multiple starts, and complete finite search.

Only a rigorous lower bound exceeding the incumbent certificate can exclude
a candidate. Both initializations receive identical proposals and arithmetic.
"""
from fractions import Fraction as F
import time,json
import numpy as np
from flint import arb
import unknown_junction_2d as problem
from interval_utils import ball,lower_record
from residual_refinement_2d import refine,coefficients,rounded,certify_priority


def point_lower(inp,branches,n=65):
    axis=np.linspace(-1,1,n);x,y=np.meshgrid(axis,axis,indexing='ij');x=x.ravel();y=y.ravel()
    values=[];px=[];py=[]
    for branch in branches:
        c=np.array([[float(v) for v in row] for row in branch.rational])
        values.append(np.polynomial.polynomial.polyval2d(x,y,c))
        px.append(np.polynomial.polynomial.polyval2d(x,y,np.polynomial.polynomial.polyder(c,axis=0)))
        py.append(np.polynomial.polynomial.polyval2d(x,y,np.polynomial.polynomial.polyder(c,axis=1)))
    values=np.array(values);ids=np.arange(len(x));winner=values.argmin(axis=0);v=values[winner,ids]
    p=np.array(px)[winner,ids];q=np.array(py)[winner,ids];a,b,g=problem.data(x,y,inp.case)
    H=a*abs(p)+b*abs(q) if inp.case=='variable_l1' else np.hypot(a*p,b*q)
    R=abs(v+H-g);D=abs(v-inp.float_spline.ev(x,y))
    chosen=np.unique(np.r_[np.argsort(R)[-16:],np.argsort(D)[-16:]])
    rl=arb(0);dl=arb(0);bl=arb(0);points=[]
    for index in chosen:
        i,j=divmod(int(index),n);xx=-1+F(2*i,n-1);yy=-1+F(2*j,n-1)
        ci=min(inp.n-2,int((xx+1)//inp.h));cj=min(inp.n-2,int((yy+1)//inp.h))
        mx,my=ball(xx),ball(yy)
        jets=[tuple(b.horner(b.derivatives[k],mx,my) for k in ((0,0),(1,0),(0,1))) for b in branches];z=jets[0][0]
        for b in jets[1:]:z=z.min(b[0])
        poly=inp.coefficients(ci,cj)
        u=poly.horner(poly.derivatives[0,0],(mx-ball(-1+ci*inp.h))/ball(inp.h),(my-ball(-1+cj*inp.h))/ball(inp.h))
        dl=dl.max(abs(z-u).abs_lower())
        aa,bb,gg,*_=problem.data_arb(ball(xx),ball(yy),inp.case)
        for k,(vv,pp,qq,*_) in enumerate(jets):
            if all(k==l or vv<jj[0] for l,jj in enumerate(jets)):
                rl=rl.max(abs(vv+problem.hamiltonian(aa,bb,pp,qq,inp.case)-gg).abs_lower())
        if abs(xx)==1 or abs(yy)==1:bl=bl.max(abs(z).abs_lower())
        points.append([str(xx),str(yy)])
    return dict(bound=lower_record(dl+rl.max(bl)),residual=lower_record(rl),distance=lower_record(dl),points=points)


def pool(inp,kind):
    candidates=[];failures=[]
    for degree in (2,3,4,5):
        for scale in (.5,1.,2.):
            start=time.perf_counter()
            try:branches,seed=problem.fit(inp,degree,kind,scale)
            except (ArithmeticError,ValueError) as e:
                failures.append(dict(degree=degree,scale=scale,reason=str(e)));continue
            seed_score=problem.sampled_score(inp,branches)
            for restart in (0,1):
                base=branches
                if restart:
                    cs=coefficients(branches,degree)
                    rng=np.random.default_rng(20260920+100*degree+int(scale*10))
                    cs=cs+.01*rng.standard_normal(cs.shape)/np.sqrt(cs.shape[1])
                    base,_=rounded(cs,degree,'perturbed_seed')
                try:
                    updated,model=refine(inp,base,degree)
                    score=problem.sampled_score(inp,updated,n=81)
                    candidates.append((score,updated,dict(degree=degree,scale=scale,restart=restart,
                        seed_score=seed_score,seed=seed,model=model)))
                except (ArithmeticError,ValueError) as e:
                    failures.append(dict(degree=degree,scale=scale,restart=restart,reason=str(e)))
    return sorted(candidates,key=lambda z:z[0]),failures


def search(n,case,kind):
    start=time.perf_counter();inp=problem.Input(n,case);construction=time.perf_counter()-start
    candidates,failures=pool(inp,kind);proposal_seconds=time.perf_counter()-start-construction
    attempts=[];best=None;top3=None;targets={};cache={}
    for rank,(score,branches,record) in enumerate(candidates):
        lo=point_lower(inp,branches)
        if best is not None and F(lo['bound']['numerator'],lo['bound']['denominator'])>F(best['certificate']['input_error_upper']['numerator'],best['certificate']['input_error_upper']['denominator']):
            attempt=dict(status='excluded',rank=rank,sampled_score=score,lower=lo,candidate=record,
                         incumbent=best['certificate']['input_error_upper'],elapsed_seconds=time.perf_counter()-start)
        else:
            key=json.dumps(record['model']['branches'],sort_keys=True);reused=key in cache
            certificate=cache[key] if reused else certify_priority(inp,branches,budget=16000)
            cache[key]=certificate
            attempt=dict(status='certified',rank=rank,sampled_score=score,lower=lo,candidate=record,
                         certificate=certificate,reused_certificate=reused,elapsed_seconds=time.perf_counter()-start)
            if best is None or certificate['input_error_upper']['decimal']<best['certificate']['input_error_upper']['decimal']:best=attempt
            for target in (.12,.08,.04):
                if str(target) not in targets and certificate['input_error_upper']['decimal']<=target:
                    targets[str(target)]=dict(rank=rank,seconds=attempt['elapsed_seconds'],bound=certificate['input_error_upper']['decimal'])
        attempts.append(attempt)
        if rank==2:top3=best['certificate']['input_error_upper'] if best else None
    return dict(n=n,case=case,kind=kind,input=inp.record,attempts=attempts,selected=best,top3_bound=top3,
                targets=targets,failed_proposals=failures,construction_seconds=construction,proposal_seconds=proposal_seconds,
                total_seconds=time.perf_counter()-start)
