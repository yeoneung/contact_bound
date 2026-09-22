"""Compute a periodic Godunov approximation and its exact clamped cubic spline.

Only the running cost is provided. No reference value function, uniform error,
or smoothing-error parameter is used by this module.
"""
from fractions import Fraction
from pathlib import Path
import hashlib,json
import numpy as np
from flint import arb,ctx
from interval_utils import ball,ceil_int,floor_int,interval,upper_record

ROOT=Path(__file__).resolve().parents[1]
DATA_BITS=44


def running_cost_float(x):
    return 2+np.cos(np.pi*x)+np.cos(2*np.pi*x)/5


def running_cost_arb(x):
    return 2+(arb.pi()*x).cos()+(2*arb.pi()*x).cos()/5


def construct(n):
    x=np.arange(n+1,dtype=float)/n
    g=running_cost_float(x)
    values=np.empty(n+1);values[-1]=g[-1]
    for i in range(n-1,-1,-1):values[i]=(g[i]/n+values[i+1])/(1+1/n)
    nums=[int(round(v*(1<<DATA_BITS))) for v in values]
    assert all(a>b for a,b in zip(nums[:-1],nums[1:]))
    record=dict(n=n,node_denominator_bits=DATA_BITS,node_numerators=nums,
                discretization='periodic Godunov upwind equation, solved on the even half-domain',
                interpolation='exact rational clamped C2 cubic spline, zero endpoint derivatives')
    encoded=json.dumps(record,sort_keys=True,separators=(',',':')).encode()
    record['input_sha256']=hashlib.sha256(encoded).hexdigest()
    return record


def quadratic_extrema(c,left=Fraction(0),right=Fraction(1)):
    c0,c1,c2=c
    def value(t):return c0+t*(c1+t*c2)
    vals=[value(left),value(right)]
    if c2:
        vertex=-c1/(2*c2)
        if left<vertex<right:vals.append(value(vertex))
    return min(vals),max(vals)


class Spline:
    def __init__(self,record):
        self.record=record;self.n=n=record['n'];self.step=Fraction(1,n)
        self.values=[Fraction(v,1<<record['node_denominator_bits']) for v in record['node_numerators']]
        vals=self.values
        # Exact tridiagonal system for knot second derivatives.
        diagonal=[Fraction(2)]+[Fraction(4)]*(n-1)+[Fraction(2)]
        rhs=[6*n*n*(vals[1]-vals[0])]
        rhs.extend(6*n*n*(vals[i-1]-2*vals[i]+vals[i+1]) for i in range(1,n))
        rhs.append(6*n*n*(vals[n-1]-vals[n]))
        for i in range(1,n+1):
            factor=1/diagonal[i-1]
            diagonal[i]-=factor;rhs[i]-=factor*rhs[i-1]
        second=[Fraction(0)]*(n+1);second[-1]=rhs[-1]/diagonal[-1]
        for i in range(n-1,-1,-1):second[i]=(rhs[i]-second[i+1])/diagonal[i]
        self.coeffs=[]
        for i in range(n):
            m0,m1=second[i:i+2];h2=self.step**2
            self.coeffs.append((vals[i],vals[i+1]-vals[i]-h2*(2*m0+m1)/6,
                                h2*m0/2,h2*(m1-m0)/6))
        self.second=second
        self.arb_coeffs=[tuple(ball(c) for c in row) for row in self.coeffs]
        assert self.point(Fraction(0),1)==0 and self.point(Fraction(1),1)==0
        # Check exact continuity, including both derivatives, at every knot.
        for i in range(n-1):
            for order in range(3):
                assert self.local_value(i,Fraction(1),order)==self.local_value(i+1,Fraction(0),order)
        self.monotone=all(quadratic_extrema((c[1],2*c[2],3*c[3]))[1]<=0 for c in self.coeffs)
        if not self.monotone:raise ValueError('Spline monotonicity was not verified.')

    def local_value(self,i,t,order=0):
        a,b,c,d=self.coeffs[i];n=self.n
        if order==0:return a+t*(b+t*(c+t*d))
        if order==1:return n*(b+t*(2*c+3*d*t))
        if order==2:return n*n*(2*c+6*d*t)
        raise ValueError(order)

    def point(self,x,order=0):
        i=min(self.n-1,max(0,int(x*self.n)))
        return self.local_value(i,x*self.n-i,order)

    def local_arb(self,i,x):
        a,b,c,d=self.arb_coeffs[i];n=self.n;t=x*n-i
        return a+t*(b+t*(c+t*d)),n*(b+t*(2*c+3*d*t)),n*n*(2*c+6*d*t)

    def range(self,y,order=0):
        # Each polynomial is evaluated only on the intersection with its cell.
        lo=max(0,min(self.n-1,floor_int(y.lower()*self.n)))
        hi=max(0,min(self.n-1,floor_int(y.upper()*self.n)))
        out=None
        for i in range(lo,hi+1):
            restricted=y.max(arb(i)/self.n).min(arb(i+1)/self.n)
            value=self.local_arb(i,restricted)[order]
            out=value if out is None else out.union(value)
        return out

    def discrete_residual(self):
        residual=arb(0);n=self.n
        for i in range(n):
            value=ball(self.values[i]+n*(self.values[i]-self.values[i+1]))-running_cost_arb(arb(i)/n)
            residual=residual.max(abs(value).upper())
        residual=residual.max(abs(ball(self.values[-1])-running_cost_arb(arb(1))).upper())
        return upper_record(residual)


def main():
    ctx.prec=192
    folder=ROOT/'results/numerical_inputs';folder.mkdir(parents=True,exist_ok=True)
    for n in (64,128,256,512,1024):
        record=construct(n);spline=Spline(record)
        record['discrete_residual_upper']=spline.discrete_residual()
        (folder/f'n{n}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(n,'nodes computed; exact C2 spline checked',flush=True)


if __name__=='__main__':main()
