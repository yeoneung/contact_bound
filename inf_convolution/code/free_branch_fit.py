"""Discover connected smooth regions, fit an unrestricted number of branches.

No branch count, face association, junction coordinates or exact values are
supplied. Numerical region discovery and optimization only propose globally
smooth polynomials; reliability is established by a separate interval check.
"""
from fractions import Fraction as F
from itertools import product
import time
import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy.ndimage import label,generate_binary_structure
from scipy.optimize import minimize,LinearConstraint,linprog
from scipy.sparse import csr_matrix
from free_branch_problem import Polynomial

def exponents(d,degree):return [e for e in product(range(degree+1),repeat=d) if sum(e)<=degree]

def basis(x,degree,derivative=None):
    d=x.shape[1];ids=exponents(d,degree);derivative=derivative or [0]*d
    values=[]
    for e in ids:
        col=np.ones(len(x))
        for j,k in enumerate(e):
            c=np.zeros(k+1);c[-1]=1
            if derivative[j]:c=ch.chebder(c,m=derivative[j])
            col*=ch.chebval(x[:,j],c)
        values.append(col)
    return np.array(values).T

def matrices(x,degree):
    d=x.shape[1];V=basis(x,degree);grad=[]
    for j in range(d):
        nu=[0]*d;nu[j]=1;grad.append(basis(x,degree,nu))
    return V,np.stack(grad,axis=1)

def round_branches(cs,degree,d):
    ids=exponents(d,degree);out=[]
    for c in cs:
        powers=np.zeros((degree+1,)*d)
        for value,e in zip(c,ids):
            factors=[]
            for k in e:
                a=np.zeros(k+1);a[-1]=1;factors.append(ch.cheb2poly(a))
            term=factors[0]
            for a in factors[1:]:term=np.multiply.outer(term,a)
            slices=tuple(slice(0,k+1) for k in e);powers[slices]+=value*term
        nums=[int(round(v*2**44)) for v in powers.flat]
        out.append(Polynomial(np.array([F(n,2**44) for n in nums],dtype=object).reshape(powers.shape)))
    return out

def sample_grid(inp,n):
    axis=np.linspace(-1,1,n);points=np.array(list(product(axis,repeat=inp.d)))
    return points[inp.problem.inside(points,1e-10)]

def evaluate(cs,mats,problem,points,u,boundary):
    V,G=mats;values=cs@V.T;grad=np.einsum('kc,ndc->knd',cs,G)
    H,_=problem.H_float(grad);residual=values+H-problem.g_float(points)
    val=values.min(axis=0);winner=values.argmin(axis=0);idx=np.arange(len(points))
    R=float(np.max(abs(residual[winner,idx])));D=float(np.max(abs(val-u)))
    b=float(np.max(abs(np.min(cs@boundary.T,axis=0))))
    return D+max(R,b),dict(values=values,gradient=grad,residual=residual,R=R,D=D,b=b)

def fit(inp,degree=3,scale=1,kind='polynomial'):
    start=time.perf_counter();d=inp.d;h=float(inp.h);eps=scale*h
    axis=np.linspace(-1+h/2,1-h/2,inp.n-1);x=np.array(list(product(axis,repeat=d)))
    u,p,H=inp.floating(x);eigen=np.linalg.eigvalsh(H)[:,0]
    retained=inp.problem.inside(x,1e-10)&(1+eps*eigen>=.5)
    labs,count=label(retained.reshape((inp.n-1,)*d),structure=generate_binary_structure(d,1));labs=labs.ravel()
    nc=len(exponents(d,degree));regions=[];discarded=[]
    for k in range(1,count+1):
        ids=np.flatnonzero(labs==k)
        if len(ids)>=max(2*nc,8):regions.append(ids)
        else:discarded.append(len(ids))
    if not regions:raise ArithmeticError('No adequately sampled smooth component')
    if len(regions)>32:raise ArithmeticError('More than 32 detected components; computation budget exceeded')
    tx=x.copy();values=u.copy();grads=p.copy()
    if kind=='corrected':
        tx=x+eps*p;values=u+eps*np.sum(p*p,axis=1)
        J=np.eye(d)[None,:,:]+eps*H
        for ids in regions:grads[ids]=2*p[ids]-np.linalg.solve(J[ids],p[ids,:,None])[...,0]
    V,G=matrices(tx,degree);guard=sample_grid(inp,13 if d==2 else 9);B=basis(guard,degree)
    floor=inp.floating(guard,second=False)[0]-.2*h;cs=[];details=[]
    for ids in regions:
        A=np.vstack([V[ids],.15*G[ids].reshape(-1,nc)]);rhs=np.r_[values[ids],.15*grads[ids].reshape(-1)]
        c,_,rank,singular=np.linalg.lstsq(A,rhs,rcond=None)
        if rank<nc:raise ArithmeticError('Rank-deficient smooth component')
        normal=A.T@A/len(rhs)+1e-6*np.eye(nc);linear=A.T@rhs/len(rhs)
        c=np.linalg.solve(normal,linear)
        opt=minimize(lambda z:.5*z@normal@z-linear@z,c,jac=lambda z:normal@z-linear,
                     constraints=LinearConstraint(B,floor,np.inf),method='SLSQP',options=dict(maxiter=400,ftol=1e-12))
        if not opt.success:raise ArithmeticError('Component fit failed: '+opt.message)
        cs.append(opt.x);details.append(dict(samples=len(ids),condition=float(singular[0]/singular[-1]),iterations=int(opt.nit),ridge=1e-6))
    cs=np.array(cs);branches=round_branches(cs,degree,d)
    return cs,branches,dict(kind=kind,degree=degree,scale=scale,detected_components=count,retained_branches=len(cs),
        discarded_small_components=discarded,components=details,coefficients=cs.tolist(),branches=[p.record() for p in branches],seconds=time.perf_counter()-start)

def refine(inp,cs,degree,iterations=24):
    start=time.perf_counter();d=inp.d;K,nc=cs.shape;cs=cs.copy()
    x=sample_grid(inp,17 if d==2 else 9);vpoints=sample_grid(inp,49 if d==2 else 17)
    bpoints=inp.problem.boundary_points(17 if d==2 else 9);BM=basis(bpoints,degree)
    vm=matrices(vpoints,degree);vu=inp.floating(vpoints,second=False)[0]
    score,state=evaluate(cs,vm,inp.problem,vpoints,vu,BM);history=[];trust=.15
    for iteration in range(iterations):
        V,G=matrices(x,degree);u=inp.floating(x,second=False)[0]
        _,st=evaluate(cs,(V,G),inp.problem,x,u,BM);v=st['values'];p=st['gradient'];r=st['residual']
        active=v<=v.min(axis=0)+.02;_,hp=inp.problem.H_float(p);rows=[];rhs=[]
        def add(k,A,col,z):
            M=np.zeros((len(z),K*nc+2));M[:,k*nc:(k+1)*nc]=A;M[:,col]=-1
            rows.append(M);rhs.append(z)
        bv=cs@BM.T;bw=np.argmin(bv,axis=0)
        for k in range(K):
            J=V+np.einsum('nd,ndc->nc',hp[k],G);m=active[k]
            add(k,J[m],-2,-r[k,m]);add(k,-J[m],-2,r[k,m])
            add(k,-V,-1,v[k]-u);win=v.argmin(axis=0)==k;add(k,V[win],-1,u[win]-v[k,win])
            add(k,-BM,-2,bv[k]);win=bw==k;add(k,BM[win],-2,-bv[k,win])
        opt=linprog(np.r_[np.zeros(K*nc),1.,1.],A_ub=csr_matrix(np.vstack(rows)),b_ub=np.concatenate(rhs),
                    bounds=[(-trust,trust)]*(K*nc)+[(0,None),(0,None)],method='highs')
        entry=dict(iteration=iteration,success=bool(opt.success),message=opt.message,score_before=score,points=len(x))
        if not opt.success:history.append(entry);break
        step=opt.x[:-2].reshape(K,nc);best=None
        for alpha in (1.,.5,.25,.125):
            trial=cs+alpha*step;value,state=evaluate(trial,vm,inp.problem,vpoints,vu,BM)
            if best is None or value<best[0]:best=value,trial,state,alpha
        value,trial,state,alpha=best;entry.update(score_after=value,step=alpha)
        if value<score-1e-10:cs=trial;score=value;trust=min(.3,trust*1.25);entry['accepted']=True
        else:trust*=.35;entry['accepted']=False
        history.append(entry)
        values=state['values'];win=values.argmin(axis=0);ii=np.arange(len(vpoints))
        worst=abs(state['residual'][win,ii])+abs(values.min(axis=0)-vu);ids=np.argsort(worst)[-24:]
        x=np.unique(np.vstack([x,vpoints[ids]]),axis=0)
        if trust<1e-5:break
    branches=round_branches(cs,degree,d)
    return branches,dict(degree=degree,coefficients=cs.tolist(),branches=[p.record() for p in branches],
        history=history,sampled_score=score,seconds=time.perf_counter()-start,retained_branches=K)
