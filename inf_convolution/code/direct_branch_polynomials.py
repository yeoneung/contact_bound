"""A direct competitor: one-sided polynomial extensions around detected maxima.

The maxima come from numerical nodes. Their locations are not supplied by the
equation or a reference solution. Every extension matches the input at its
outer endpoint; an endpoint dominance check makes the local minimum continuous.
"""
from fractions import Fraction as F
import numpy as np
from interval_utils import ball,interval
from automatic_branches import BranchFailure

class Polynomial:
    def __init__(self,anchor,coeffs,left,right,identical_input=False):
        self.anchor=anchor;self.coeffs=tuple(coeffs);self.acoeffs=tuple(ball(v) for v in coeffs)
        self.left=left;self.right=right;self.identical_input=identical_input
    def exact(self,y):
        t=y-self.anchor;a,b,c,d=self.coeffs;return a+t*(b+t*(c+t*d))
    def jet(self,left,right,part=0,kind=None):
        t=interval(left-self.anchor,right-self.anchor);a,b,c,d=self.acoeffs
        return a+t*(b+t*(c+t*d)),b+t*(2*c+3*d*t),2*c+6*d*t,interval(left,right)
    def float_values(self,y):
        t=y-float(self.anchor);a,b,c,d=map(float,self.coeffs)
        return a+t*(b+t*(c+t*d)),b+t*(2*c+3*d*t)

def build(spline,width,degree=3):
    h=spline.h;n=spline.n
    values=spline.record['numerical_node_numerators']
    peaks=[i for i,v in enumerate(values) if v>values[(i-1)%n] and v>=values[(i+1)%n]]
    if not peaks:raise BranchFailure('No numerical peak was detected.')
    gaps=[]
    for i in peaks:
        for shift in [-n,0,n]:
            a,b=(i+shift-width)*h,(i+shift+width)*h
            if b<=0 or a>=2:continue
            coeff=[]
            for x in [a,b]:
                # The third derivative is taken toward the smooth outer region.
                cell=x//h-(1 if x==a else 0)
                t=x/h-cell
                coeff.append([spline.local_exact(cell,t,k)/[1,1,2,6][k] if k<=degree else F(0) for k in range(4)])
            l=Polynomial(a,coeff[0],a,b);r=Polynomial(b,coeff[1],a,b)
            if l.exact(a)>r.exact(a) or r.exact(b)>l.exact(b):
                raise BranchFailure('Direct polynomial extensions do not join continuously.')
            gaps.append((a,b,l,r))
    gaps.sort(key=lambda v:v[0])
    if any(a[1]>b[0] for a,b in zip(gaps,gaps[1:])):raise BranchFailure('Detected polynomial reconstruction intervals overlap.')
    edges=sorted(set([F(0),F(2)]+[i*h for i in range(n+1)]+[x for a,b,_,_ in gaps for x in [a,b] if 0<x<2]))
    pending=[]
    for a,b in zip(edges,edges[1:]):
        m=(a+b)/2;gap=next((v for v in gaps if v[0]<=m<=v[1]),None)
        if gap:entries=[(gap[2],0),(gap[3],0)]
        else:
            i=m//h;c=spline.coeffs[i%n]
            entries=[(Polynomial(i*h,[c[k]/h**k for k in range(4)],a,b,True),0)]
        pending.append((a,b,entries,0))
    return pending,peaks

def float_envelope(pending,y):
    out=np.full(np.shape(y),np.nan);dout=out.copy()
    for a,b,entries,_ in pending:
        valid=(y>=float(a))&(y<=float(b));v=[];d=[]
        for p,_ in entries:
            vv,dd=p.float_values(y[valid]);v.append(vv);d.append(dd)
        idx=np.argmin(v,axis=0);out[valid]=np.take_along_axis(np.array(v),idx[None,:],axis=0)[0]
        dout[valid]=np.take_along_axis(np.array(d),idx[None,:],axis=0)[0]
    return out,dout
