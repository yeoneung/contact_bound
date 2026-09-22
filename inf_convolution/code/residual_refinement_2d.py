"""Sequential minimax residual correction of boundary-factored branches.

Floating-point linear programs only propose coefficients. No sampled objective,
optimizer status or line-search decision is an error certificate. The final
rounded polynomials are checked by the same interval verifier as the controls.
"""
from fractions import Fraction as F
import heapq
import time
import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from flint import arb
from interval_utils import ball, upper_record, lower_record
from unknown_junction_2d import basis, data, Power, Verifier, boundary_bound
import unknown_junction_2d as problem


def matrices(x,y,degree):
    v,ids=basis(x,y,degree-1)
    dx,_=basis(x,y,degree-1,dx=1);dy,_=basis(x,y,degree-1,dy=1)
    out=[]
    for k in range(4):
        d=[1+x,1-x,1+y,1-y][k]
        sx=[1,-1,0,0][k];sy=[0,0,1,-1][k]
        out.append((d[:,None]*v,d[:,None]*dx+sx*v,d[:,None]*dy+sy*v,v))
    return out,ids


def coefficients(branches,degree):
    axis=np.linspace(-1,1,degree+3);x,y=np.meshgrid(axis,axis,indexing='ij')
    x=x.ravel();y=y.ravel();mats,_=matrices(x,y,degree)
    cs=[]
    for p,(v,*_) in zip(branches,mats):
        c=np.array([[float(a) for a in row] for row in p.rational])
        z=np.polynomial.polynomial.polyval2d(x,y,c)
        cs.append(np.linalg.lstsq(v,z,rcond=None)[0])
    return np.array(cs)


def rounded(cs,degree,kind='residual_refined'):
    ids=[(i,j) for i in range(degree) for j in range(degree-i)]
    branches=[];records=[]
    for label,c in enumerate(cs):
        powers=np.zeros((degree+1,degree+1))
        for a,(i,j) in zip(c,ids):
            ci=np.zeros(i+1);ci[-1]=1;cj=np.zeros(j+1);cj[-1]=1
            powers[:i+1,:j+1]+=a*np.outer(ch.cheb2poly(ci),ch.cheb2poly(cj))
        base=np.rint(powers*2**44).astype(np.int64);nums=base.copy()
        if label<2:nums[1:,:]+=[1,-1][label]*base[:-1,:]
        else:nums[:,1:]+=[1,-1][label-2]*base[:,:-1]
        branches.append(Power([[F(int(v),2**44) for v in row] for row in nums]))
        records.append(dict(label=label,bits=44,coefficients=nums.tolist()))
    return branches,dict(kind=kind,degree=degree,branches=records)


def evaluate(cs,mats,a,b,g,u,case,margin=0):
    v=np.array([m[0]@c for c,m in zip(cs,mats)])
    px=np.array([m[1]@c for c,m in zip(cs,mats)])
    py=np.array([m[2]@c for c,m in zip(cs,mats)])
    h=a*abs(px)+b*abs(py) if case=='variable_l1' else np.hypot(a*px,b*py)
    r=v+h-g;w=v.min(axis=0);active=v<=w+margin
    residual=float(np.max(abs(r)[active]));distance=float(np.max(abs(w-u)))
    return residual+distance,(v,px,py,r,active),residual,distance


def refine(inp,branches,degree,iterations=8,training_n=21,validation_n=81):
    start=time.perf_counter();cs=coefficients(branches,degree);nc=cs.shape[1]
    ax=np.linspace(-1,1,training_n);x,y=np.meshgrid(ax,ax,indexing='ij');x=x.ravel();y=y.ravel()
    va=np.linspace(-1,1,validation_n);vx,vy=np.meshgrid(va,va,indexing='ij');vx=vx.ravel();vy=vy.ravel()
    vm,_=matrices(vx,vy,degree);vd=problem.data(vx,vy,inp.case);vu=inp.float_spline.ev(vx,vy)
    score,_,_,_=evaluate(cs,vm,*vd,vu,inp.case)
    history=[];trust=.15
    for iteration in range(iterations):
        mats,_=matrices(x,y,degree);a,b,g=problem.data(x,y,inp.case);u=inp.float_spline.ev(x,y)
        _,(v,px,py,r,active),_,_=evaluate(cs,mats,a,b,g,u,inp.case,margin=.02)
        rows=[];rhs=[]
        def add(k,A,tcol,z):
            M=np.zeros((len(z),4*nc+2));M[:,k*nc:(k+1)*nc]=A;M[:,tcol]=-1
            rows.append(M);rhs.append(z)
        # Linearized branch residuals at current and nearly active branches.
        for k,(V,Dx,Dy,B) in enumerate(mats):
            mask=active[k]
            if inp.case=='variable_l1':hx=a*np.sign(px[k]);hy=b*np.sign(py[k])
            else:
                H=np.maximum(np.hypot(a*px[k],b*py[k]),1e-14)
                hx=a*a*px[k]/H;hy=b*b*py[k]/H
            J=V+hx[:,None]*Dx+hy[:,None]*Dy
            add(k,J[mask],-2,-r[k,mask]);add(k,-J[mask],-2,r[k,mask])
            # All branches above u-D, and one selected branch below u+D.
            add(k,-V,-1,v[k]-u)
            winner=np.argmin(v,axis=0)==k
            add(k,V[winner],-1,u[winner]-v[k,winner])
            # Positivity of boundary factors is only a sampled constraint.
            M=np.zeros((len(x),4*nc+2));M[:,k*nc:(k+1)*nc]=-B
            rows.append(M);rhs.append(B@cs[k]-.02)
        objective=np.r_[np.zeros(4*nc),1.,1.]
        opt=linprog(objective,A_ub=csr_matrix(np.vstack(rows)),b_ub=np.concatenate(rhs),
                    bounds=[(-trust,trust)]*(4*nc)+[(0,None),(0,None)],method='highs')
        entry=dict(iteration=iteration,lp_success=bool(opt.success),lp_message=opt.message,score_before=score,
                   training_points=len(x),trust=trust)
        if not opt.success:history.append(entry);break
        step=opt.x[:-2].reshape(4,nc);best=None
        for alpha in (1.,.5,.25,.125):
            trial=cs+alpha*step
            value,state,R,D=evaluate(trial,vm,*vd,vu,inp.case)
            if best is None or value<best[0]:best=(value,trial,alpha,state,R,D)
        value,trial,alpha,state,R,D=best
        entry.update(score_after=value,step=alpha,residual=R,distance=D)
        if value<score-1e-10:
            cs=trial;score=value;entry['accepted']=True;trust=min(.3,trust*1.25)
        else:entry['accepted']=False;trust*=.35
        history.append(entry)
        # Add worst validation points even after a rejected proposal. This is
        # an exchange of collocation constraints, not a rigorous error bound.
        vv,pp,qq,rr,aa=state
        worst=np.max(np.where(aa,abs(rr),0),axis=0)+abs(vv.min(axis=0)-vu)
        ii=np.argsort(worst)[-24:]
        points=np.unique(np.vstack([np.column_stack([x,y]),np.column_stack([vx[ii],vy[ii]])]),axis=0)
        x,y=points.T
        if trust<1e-5:break
    out,record=rounded(cs,degree)
    record.update(history=history,seconds=time.perf_counter()-start,sampled_score=score,
                  training_n=training_n,validation_n=validation_n,iterations=iterations)
    return out,record


def certify_priority(inp,branches,budget=16000,relative=F(1,20),absolute=F(1,1000000)):
    """Same enclosures as the control; split the largest remaining bounds first.

    Every leaf is retained. A work/depth cap can reduce sharpness but cannot
    remove an unverified part of the domain. Applies identically to all seeds.
    """
    start=time.perf_counter();ver=Verifier(inp,branches);leaves={};queue=[];counter=0
    rl=arb(0);dl=arb(0);bc=boundary_bound(branches);maxdepth=0
    def insert(box):
        nonlocal counter,rl,dl,maxdepth
        z=ver.box(box);rl=rl.max(z[2]);dl=dl.max(z[3]);key=counter;counter+=1
        leaves[key]=(box,z);maxdepth=max(maxdepth,box[-1])
        if box[-1]<12:
            heapq.heappush(queue,(-float(z[0]+z[1]),key))
    for i in range(inp.n-1):
        for j in range(inp.n-1):
            insert((-1+i*inp.h,-1+(i+1)*inp.h,-1+j*inp.h,-1+(j+1)*inp.h,i,j,0))
    stopped='budget'
    while queue and ver.evaluations+2<=max(budget,(inp.n-1)**2):
        _,key=heapq.heappop(queue);box,z=leaves[key]
        if z[0]<=rl*ball(1+relative)+ball(absolute) and z[1]<=dl*ball(1+relative)+ball(absolute):continue
        del leaves[key];xl,xr,yl,yr,i,j,d=box
        if xr-xl>=yr-yl:
            m=(xl+xr)/2;children=[(xl,m,yl,yr,i,j,d+1),(m,xr,yl,yr,i,j,d+1)]
        else:
            m=(yl+yr)/2;children=[(xl,xr,yl,m,i,j,d+1),(xl,xr,m,yr,i,j,d+1)]
        for child in children:insert(child)
    if not queue:stopped='tolerance_or_depth'
    ru=arb(0);du=arb(0);limited=0;junctions=0
    for box,z in leaves.values():
        ru=ru.max(z[0]);du=du.max(z[1]);junctions+=z[4]>1
        limited+=not(z[0]<=rl*ball(1+relative)+ball(absolute) and z[1]<=dl*ball(1+relative)+ball(absolute))
    return dict(status='certified',input_error_upper=upper_record(du+ru.max(bc)),
                reconstruction_error_upper=upper_record(ru.max(bc)),residual_upper=upper_record(ru),
                distance_upper=upper_record(du),boundary_upper=upper_record(bc),
                residual_lower=lower_record(rl),distance_lower=lower_record(dl),evaluations=ver.evaluations,
                leaves=len(leaves),junction_boxes=junctions,maximum_depth=maxdepth,
                tightness_limited_boxes=limited,stopping=stopped,seconds=time.perf_counter()-start)
