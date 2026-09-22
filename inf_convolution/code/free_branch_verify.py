"""Whole-domain Arb certificate for a minimum of any finite number of branches.

Cartesian cells covering a polytope are discarded only by exact halfspaces.
All unresolved cells and all boundary patches contribute to the upper bound.
"""
from fractions import Fraction as F
from itertools import product
import heapq,math,time
from flint import arb,ctx
from interval_utils import ball,interval,upper_record
from free_branch_bernstein import Distance


def maximum(xs):
    z=arb(0)
    for x in xs:z=z.max(x)
    return z


class Verifier:
    def __init__(self,inp,branches):
        self.inp=inp;self.problem=inp.problem;self.branches=branches;self.d=inp.d
        self.distance=Distance(inp,branches)
    def box(self,cell,lo,hi):
        center=[(a+b)/2 for a,b in zip(lo,hi)];radius=[(b-a)/2 for a,b in zip(lo,hi)]
        x=[interval(a,b) for a,b in zip(lo,hi)];m=[ball(a) for a in center]
        jets=[p.jet(x) for p in self.branches];mid=[p.jet(m,second=False) for p in self.branches]
        active=[]
        for k,(v,p,H) in enumerate(jets):
            if not any(k!=l and mid[k][0]-mid[l][0]+sum(interval(-r,r)*(a-b) for r,a,b in zip(radius,p,jets[l][1]))>0 for l in range(len(jets))):active.append(k)
        if not active:raise ArithmeticError('No potentially active branch')
        gc=self.problem.gpoly.at(m);_,gp,_=self.problem.gpoly.jet(x,second=False)
        residual=[];distance=[]
        for k in active:
            v,p,H=jets[k];vc,pc,_=mid[k]
            hc,_=self.problem.H_arb(pc);_,hp=self.problem.H_arb(p,H)
            rr=vc+hc-gc+sum(interval(-r,r)*(a+b-c) for r,a,b,c in zip(radius,p,hp,gp))
            residual.append(abs(rr));distance.append(self.distance.enclosure(cell,k,lo,hi))
        delta=distance[0]
        for value in distance[1:]:delta=delta.min(value)
        return maximum(residual),abs(delta),len(active)
    def boundary_box(self,face,lo,hi):
        origin,directions=self.problem.faces[face]
        t=[interval(a,b) for a,b in zip(lo,hi)];m=[ball((a+b)/2) for a,b in zip(lo,hi)]
        radii=[(b-a)/2 for a,b in zip(lo,hi)]
        x=[ball(a)+sum(ball(v[j])*s for v,s in zip(directions,t)) for j,a in enumerate(origin)]
        xc=[ball(a)+sum(ball(v[j])*s for v,s in zip(directions,m)) for j,a in enumerate(origin)]
        vals=[]
        for p in self.branches:
            _,grad,_=p.jet(x,second=False)
            value=p.at(xc)+sum(interval(-r,r)*sum(ball(a)*b for a,b in zip(v,grad)) for r,v in zip(radii,directions))
            vals.append(value)
        z=vals[0]
        for v in vals[1:]:z=z.min(v)
        return abs(z)


def encode(cell,lo,hi,depth=0):return dict(cell=list(cell),lo=[str(v) for v in lo],hi=[str(v) for v in hi],depth=depth)


def certify(inp,branches,budget=12000,boundary_budget=2048):
    ctx.prec=128;start=time.perf_counter();ver=Verifier(inp,branches);queue=[];outside=[];serial=0;evaluations=0
    def add(cell,lo,hi,depth):
        nonlocal serial,evaluations
        record=encode(cell,lo,hi,depth)
        if inp.problem.classify(lo,hi)<0:outside.append(record);return
        R,D,K=ver.box(cell,lo,hi);evaluations+=1
        heapq.heappush(queue,(-float((R+D).upper()),serial,cell,lo,hi,depth,R,D,K));serial+=1
    for cell in product(range(inp.n-1),repeat=inp.d):
        lo=tuple(-1+i*inp.h for i in cell);hi=tuple(a+inp.h for a in lo);add(cell,lo,hi,0)
    initial_evaluations=evaluations
    while evaluations+2<=max(budget,initial_evaluations) and queue:
        entry=heapq.heappop(queue);_,_,cell,lo,hi,depth,R,D,K=entry
        if depth>=16:heapq.heappush(queue,entry);break
        axis=max(range(inp.d),key=lambda j:hi[j]-lo[j]);mid=(lo[axis]+hi[axis])/2
        left=list(hi);left[axis]=mid;right=list(lo);right[axis]=mid
        add(cell,lo,tuple(left),depth+1);add(cell,tuple(right),hi,depth+1)
    R=maximum(e[6] for e in queue);D=maximum(e[7] for e in queue)
    leaves=[encode(e[2],e[3],e[4],e[5]) for e in queue]
    boundary=[];bs=0
    def badd(face,lo,hi,depth):
        nonlocal bs
        b=ver.boundary_box(face,lo,hi);heapq.heappush(boundary,(-float(b.upper()),bs,face,lo,hi,depth,b));bs+=1
    for face in range(len(inp.problem.faces)):badd(face,(F(0),)*(inp.d-1),(F(1),)*(inp.d-1),0)
    while bs+2<=boundary_budget:
        entry=heapq.heappop(boundary);_,_,face,lo,hi,depth,b=entry
        if depth>=16:heapq.heappush(boundary,entry);break
        axis=max(range(inp.d-1),key=lambda j:hi[j]-lo[j]);mid=(lo[axis]+hi[axis])/2
        left=list(hi);left[axis]=mid;right=list(lo);right[axis]=mid
        badd(face,lo,tuple(left),depth+1);badd(face,tuple(right),hi,depth+1)
    b=maximum(e[6] for e in boundary)
    return dict(residual=upper_record(R),distance=upper_record(D),boundary=upper_record(b),
        reconstruction_error_upper=upper_record(R.max(b)),input_error_upper=upper_record(D+R.max(b)),
        leaves=leaves,outside=outside,boundary_leaves=[dict(face=e[2],lo=[str(v) for v in e[3]],hi=[str(v) for v in e[4]],depth=e[5]) for e in boundary],
        evaluations=evaluations,boundary_evaluations=bs,initial_evaluations=initial_evaluations,
        branch_count=len(branches),seconds=time.perf_counter()-start,precision=128,budget=budget,boundary_budget=boundary_budget)
