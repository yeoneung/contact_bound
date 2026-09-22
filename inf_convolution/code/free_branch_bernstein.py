"""Arb Bernstein enclosures, including exact input/reconstruction cancellation."""
from itertools import product
from fractions import Fraction as F
from math import comb
import numpy as np
from flint import arb
from interval_utils import ball


def power_to_bernstein(c,lo,hi,degrees=None):
    d=c.ndim;degrees=degrees or tuple(n-1 for n in c.shape)
    out=np.empty(tuple(n+1 for n in degrees),dtype=object);out.fill(arb(0))
    for index in np.ndindex(c.shape):out[index]=ball(c[index])
    for axis,(a,b,n) in enumerate(zip(lo,hi,degrees)):
        aa=ball(a);width=ball(b-a);T=[]
        for j in range(n+1):
            T.append([sum(comb(k,m)*aa**(k-m)*width**m*ball(F(comb(j,m),comb(n,m))) for m in range(min(j,k)+1)) for k in range(n+1)])
        temp=np.moveaxis(out,axis,0);new=np.empty(temp.shape,dtype=object)
        for index in np.ndindex(temp.shape[1:]):
            row=[temp[(k,)+index] for k in range(n+1)]
            for j in range(n+1):new[(j,)+index]=sum(a*b for a,b in zip(T[j],row))
        out=np.moveaxis(new,0,axis)
    return out


def split_line(row,t):
    t=ball(t);one=1-t;work=list(row);left=[work[0]];right=[work[-1]]
    while len(work)>1:
        work=[one*a+t*b for a,b in zip(work,work[1:])];left.append(work[0]);right.append(work[-1])
    return left,list(reversed(right))


def restrict(c,lo,hi):
    out=c
    for axis,(a,b) in enumerate(zip(lo,hi)):
        if a==0 and b==1:continue
        temp=np.moveaxis(out,axis,0);new=np.empty(temp.shape,dtype=object)
        for index in np.ndindex(temp.shape[1:]):
            row=[temp[(k,)+index] for k in range(temp.shape[0])]
            if a>0:_,row=split_line(row,a)
            if b<1:row,_=split_line(row,(b-a)/(1-a))
            for k,v in enumerate(row):new[(k,)+index]=v
        out=np.moveaxis(new,0,axis)
    return out


class Distance:
    def __init__(self,inp,branches):self.inp=inp;self.branches=branches;self.cache={};self.input_cache={}
    def root(self,cell,k):
        key=(tuple(cell),k)
        if key not in self.cache:
            branch=self.branches[k];degrees=tuple(max(3,n-1) for n in branch.c.shape)
            lo=tuple(-1+i*self.inp.h for i in cell);hi=tuple(a+self.inp.h for a in lo)
            q=power_to_bernstein(branch.c,lo,hi,degrees)
            ikey=(tuple(cell),degrees)
            if ikey not in self.input_cache:
                self.input_cache[ikey]=power_to_bernstein(self.inp.coefficients(cell).c,(F(0),)*self.inp.d,(F(1),)*self.inp.d,degrees)
            self.cache[key]=q-self.input_cache[ikey]
        return self.cache[key]
    def enclosure(self,cell,k,lo,hi):
        base=[-1+i*self.inp.h for i in cell]
        a=[(v-z)/self.inp.h for v,z in zip(lo,base)];b=[(v-z)/self.inp.h for v,z in zip(hi,base)]
        c=restrict(self.root(cell,k),a,b);out=c.flat[0]
        for x in c.flat:out=out.union(x)
        return out
