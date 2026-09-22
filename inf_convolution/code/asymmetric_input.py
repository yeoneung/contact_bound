"""Periodic numerical inputs with no prescribed symmetry or junction location."""
from fractions import Fraction as F
import heapq,time
import numpy as np
from flint import arb
from interval_utils import ball,interval

CASES={
    'skew':[(1,1.,0.),(2,0.,.2)],
    'skew_strong':[(1,1.,0.),(2,0.,.35),(3,.1,0.)],
    'two_peaks':[(2,1.,0.),(1,0.,.3),(3,.15,0.)],
}

def cost(x,case='skew'):
    return 2+sum(a*np.cos(j*np.pi*x)+b*np.sin(j*np.pi*x) for j,a,b in CASES[case])

def cost_arb(x,case='skew',order=0):
    out=arb(2) if order==0 else arb(0)
    for j,a,b in CASES[case]:
        k=j*arb.pi();s,c=(k*x).sin_cos();aa=ball(F(str(a)));bb=ball(F(str(b)))
        if order==0:out+=aa*c+bb*s
        elif order==1:out+=k*(-aa*s+bb*c)
        else:raise ValueError(order)
    return out

def solve(n,case='skew'):
    """Label-setting solution of U_i=min(g_i,(h*g_i+U_neighbor)/(1+h))."""
    g=cost(2*np.arange(n)/n,case);h=2/n;u=g.copy();heap=[(float(v),i) for i,v in enumerate(u)]
    heapq.heapify(heap);settled=np.zeros(n,dtype=bool)
    while heap:
        value,i=heapq.heappop(heap)
        if settled[i] or value!=u[i]:continue
        settled[i]=True
        for j in [(i-1)%n,(i+1)%n]:
            candidate=(h*g[j]+value)/(1+h)
            if not settled[j] and candidate<u[j]:u[j]=candidate;heapq.heappush(heap,(float(candidate),j))
    residual=u+np.maximum(0,np.maximum(u-np.roll(u,1),u-np.roll(u,-1)))/h-g
    assert np.max(np.abs(residual))<1e-11
    return u

def construct(n,case='skew',bits=42):
    started=time.perf_counter();nodes=solve(n,case)
    # Rounded cardinal B-spline coefficients define an exact periodic C2
    # function. Solving for the coefficients does not locate any corner.
    coeff=np.fft.ifft(6*np.fft.fft(nodes)/(4+2*np.cos(2*np.pi*np.arange(n)/n))).real
    nums=[int(round(v*2**bits)) for v in coeff]
    return dict(n=n,case=case,coefficient_bits=bits,coefficient_numerators=nums,
                numerical_node_numerators=[int(round(v*2**bits)) for v in nodes],
                solve_and_fit_seconds=time.perf_counter()-started)

class PeriodicSpline:
    def __init__(self,record):
        self.record=record;self.n=n=record['n'];self.case=record['case'];self.h=h=F(2,n)
        a=[F(v,2**record['coefficient_bits']) for v in record['coefficient_numerators']]
        self.coeffs=[]
        for i in range(n):
            p,q,r,s=[a[j%n] for j in [i-1,i,i+1,i+2]]
            self.coeffs.append(((p+4*q+r)/6,(-p+r)/2,(p-2*q+r)/2,(-p+3*q-3*r+s)/6))
        self.acoeffs=[tuple(ball(c) for c in row) for row in self.coeffs]
        self.fcoeffs=np.array([[float(c) for c in row] for row in self.coeffs])
        for i in range(n):
            for order in range(3):assert self.local_exact(i,F(1),order)==self.local_exact((i+1)%n,F(0),order)

    def local_exact(self,i,t,order=0):
        a,b,c,d=self.coeffs[i%self.n];h=self.h
        if order==0:return a+t*(b+t*(c+t*d))
        if order==1:return (b+t*(2*c+3*d*t))/h
        if order==2:return (2*c+6*d*t)/h**2
        if order==3:return 6*d/h**3
        raise ValueError(order)

    def point(self,x,order=0):
        i=x//self.h;return self.local_exact(i,x/self.h-i,order)

    def jet(self,i,x):
        a,b,c,d=self.acoeffs[i%self.n];hh=ball(self.h);t=x/hh-i
        return a+t*(b+t*(c+t*d)),(b+t*(2*c+3*d*t))/hh,(2*c+6*d*t)/hh**2,6*d/hh**3

    def float_jet(self,x):
        h=float(self.h);xx=np.asarray(x);i=np.floor(xx/h).astype(int);t=xx/h-i
        a,b,c,d=np.moveaxis(self.fcoeffs[i%self.n],-1,0)
        return a+t*(b+t*(c+t*d)),(b+t*(2*c+3*d*t))/h,(2*c+6*d*t)/h**2,6*d/h**3

    def range(self,left,right,order=0):
        first=left//self.h;last=right//self.h;out=None
        for i in range(first,last+1):
            lo=max(left,i*self.h);hi=min(right,(i+1)*self.h)
            value=self.jet(i,interval(lo,hi))[order]
            out=value if out is None else out.union(value)
        return out
