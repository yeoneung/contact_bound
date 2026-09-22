"""Convex discounted PDEs on polytopes, with no prescribed solution branches."""
from fractions import Fraction as F
from itertools import product
import math,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from flint import arb
from interval_utils import ball,interval

DEFINITIONS={
 'pentagon':dict(vertices=[[-1,-.6],[-.2,-1],[1,-.3],[.7,1],[-.8,.8]],
    M=[[1,.2],[.1,.85]],drift=[.15,-.1],g=[([0,0],1),([1,0],.1),([1,1],.15),([0,2],.05)]),
 'heptagon':dict(vertices=[[-1,-.3],[-.7,-.9],[.2,-1],[1,-.4],[.9,.5],[.2,1],[-.8,.7]],
    velocities=[[1.12,.05],[.72,.95],[-.68,.75],[-.88,-.15],[-.08,-.95],[.92,-.55]],
    g=[([0,0],1),([2,0],.1),([0,2],.1),([1,1],.08)]),
 'coupled_3d':dict(dimension=3,M=[[1,.2,.1],[0,.85,.15],[.1,0,1.1]],drift=[.1,-.05,.08],
    g=[([0,0,0],1),([1,1,0],.1),([0,1,1],.08),([1,0,1],.06),([1,0,0],.05)]),
}

def rational(x):return F(str(x))

class Polynomial:
    def __init__(self,coefficients):
        self.c=np.asarray(coefficients,dtype=object);self.d=self.c.ndim;self.cache={}
        self.c=np.array([F(v) for v in self.c.flat],dtype=object).reshape(self.c.shape)
    @staticmethod
    def pack(c):
        if np.ndim(c)==0:
            value=c.item() if hasattr(c,'item') else c
            return ball(value) if value!=0 else None
        rows=[Polynomial.pack(a) for a in c]
        while rows and rows[-1] is None:rows.pop()
        return tuple(rows) if rows else None
    def derivative(self,orders):
        key=tuple(orders)
        if key not in self.cache:
            c=self.c
            for axis,order in enumerate(orders):
                for _ in range(order):
                    if c.shape[axis]<=1:
                        c=np.zeros(tuple(1 if k==axis else n for k,n in enumerate(c.shape)),dtype=object);break
                    c=np.take(c,np.arange(1,c.shape[axis]),axis=axis)
                    shape=[1]*self.d;shape[axis]=c.shape[axis]
                    c=c*np.arange(1,c.shape[axis]+1).reshape(shape)
            self.cache[key]=self.pack(c)
        return self.cache[key]
    @staticmethod
    def horner(c,x,axis=0):
        if c is None:return arb(0)
        if axis==len(x):return c
        out=arb(0)
        for a in reversed(c):out=out*x[axis]+Polynomial.horner(a,x,axis+1)
        return out
    def at(self,x,orders=None):return self.horner(self.derivative(orders or (0,)*self.d),x)
    def jet(self,x,second=True):
        value=self.at(x);grad=[];H=[]
        for k in range(self.d):
            nu=[0]*self.d;nu[k]=1;grad.append(self.at(x,nu))
        if second:
            for k in range(self.d):
                row=[]
                for l in range(self.d):
                    nu=[0]*self.d;nu[k]+=1;nu[l]+=1;row.append(self.at(x,nu))
                H.append(row)
        return value,grad,H
    def record(self,bits=44):
        nums=[int(v*2**bits) for v in self.c.flat]
        assert all(F(n,2**bits)==v for n,v in zip(nums,self.c.flat))
        return dict(shape=self.c.shape,bits=bits,numerators=nums)
    @classmethod
    def load(cls,r):return cls(np.array([F(n,2**r['bits']) for n in r['numerators']],dtype=object).reshape(r['shape']))

class Problem:
    def __init__(self,name):
        self.name=name;self.definition=definition=DEFINITIONS[name];self.d=definition.get('dimension',2)
        self.planes=[];self.faces=[]
        if 'vertices' in definition:
            vs=[[rational(a) for a in v] for v in definition['vertices']]
            for v,w in zip(vs,vs[1:]+vs[:1]):
                normal=[w[1]-v[1],v[0]-w[0]];offset=sum(a*b for a,b in zip(normal,v))
                self.planes.append((normal,offset));self.faces.append((v,[[w[i]-v[i] for i in range(2)]]))
            assert all(sum(a*b for a,b in zip(n,v))<=c for n,c in self.planes for v in vs)
        else:
            for axis in range(self.d):
                for sign in (-1,1):
                    normal=[F(0)]*self.d;normal[axis]=F(sign);self.planes.append((normal,F(1)))
                    origin=[F(-1)]*self.d;origin[axis]=F(sign);directions=[]
                    for k in range(self.d):
                        if k!=axis:
                            e=[F(0)]*self.d;e[k]=F(2);directions.append(e)
                    self.faces.append((origin,directions))
        self.normals=np.array([[float(v) for v in n] for n,c in self.planes]);self.offsets=np.array([float(c) for n,c in self.planes])
        self.polyhedral='velocities' in definition
        if self.polyhedral:self.velocities=np.array(definition['velocities']);self.av=[[ball(rational(v)) for v in row] for row in definition['velocities']]
        else:
            self.M=np.array(definition['M']);self.drift=np.array(definition['drift'])
            self.am=[[ball(rational(v)) for v in row] for row in definition['M']];self.ab=[ball(rational(v)) for v in definition['drift']]
        c=np.zeros((3,)*self.d,dtype=object)
        for exponent,value in definition['g']:c[tuple(exponent)]=rational(value)
        self.gpoly=Polynomial(c)
    def inside(self,x,margin=0):return np.all(x@self.normals.T<=self.offsets-margin,axis=-1)
    def classify(self,lo,hi):
        cut=False
        for normal,offset in self.planes:
            small=sum(a*(l if a>=0 else h) for a,l,h in zip(normal,lo,hi))
            large=sum(a*(h if a>=0 else l) for a,l,h in zip(normal,lo,hi))
            if small>offset:return -1
            if large>offset:cut=True
        return 0 if cut else 1
    def g_float(self,x):
        return sum(float(v)*np.prod(x**np.array(e),axis=-1) for e,v in self.definition['g'])
    def H_float(self,p):
        if self.polyhedral:
            v=p@self.velocities.T;k=np.argmax(v,axis=-1)
            return np.max(v,axis=-1),self.velocities[k]
        q=p@self.M.T;n=np.linalg.norm(q,axis=-1)
        return n+p@self.drift,(q@self.M)/np.maximum(n[...,None],1e-14)+self.drift
    def H_arb(self,p,Hess=None):
        if self.polyhedral:
            values=[sum(a*b for a,b in zip(v,p)) for v in self.av];out=values[0]
            for v in values[1:]:out=out.max(v)
            derivatives=[]
            if Hess is not None:
                active=[i for i,v in enumerate(values) if not any(v<w for w in values)]
                for j in range(self.d):
                    ds=[sum(v[i]*Hess[i][j] for i in range(self.d)) for k,v in enumerate(self.av) if k in active]
                    z=ds[0]
                    for v in ds[1:]:z=z.union(v)
                    derivatives.append(z)
            return out,derivatives
        q=[sum(a*b for a,b in zip(row,p)) for row in self.am];norm=sum(a**2 for a in q).sqrt()
        out=norm+sum(a*b for a,b in zip(self.ab,p));derivatives=[]
        if Hess is not None:
            for j in range(self.d):
                v=[sum(row[i]*Hess[i][j] for i in range(self.d)) for row in self.am]
                linear=sum(self.ab[i]*Hess[i][j] for i in range(self.d))
                z=sum(a*b for a,b in zip(q,v))/norm if norm>0 else interval(-1,1)*sum(abs(a) for a in v)
                derivatives.append(z+linear)
        return out,derivatives
    def boundary_points(self,n=13):
        parameters=np.array(list(product(np.linspace(0,1,n),repeat=self.d-1)));out=[]
        for origin,directions in self.faces:out.append(np.array(origin,float)+parameters@np.array(directions,float))
        return np.vstack(out)

def solve(problem,n):
    """Monotone semi-Lagrangian iteration; no exact solution or branch labels."""
    start=time.perf_counter();d=problem.d;axis=np.linspace(-1,1,n);grid=np.array(list(product(axis,repeat=d)))
    mask=problem.inside(grid,1e-10);x=grid[mask];h=2/(n-1)
    if problem.polyhedral:velocities=-problem.velocities
    else:
        if d==2:
            theta=2*np.pi*np.arange(48)/48;directions=np.column_stack([np.cos(theta),np.sin(theta)])
        else:
            k=np.arange(64);z=1-2*(k+.5)/64;theta=k*np.pi*(3-np.sqrt(5))
            directions=np.column_stack([np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z])
            directions=np.vstack([directions,np.eye(d),-np.eye(d)])
        velocities=-(directions@problem.M+problem.drift)
    step=.8*h/np.max(np.linalg.norm(velocities,axis=1));exit_times=[];ends=[]
    for velocity in velocities:
        rate=problem.normals@velocity;gap=problem.offsets-x@problem.normals.T
        exit_time=np.min(np.divide(gap,rate,out=np.full_like(gap,np.inf),where=rate>1e-14),axis=1)
        t=np.minimum(step,exit_time);exit_times.append(t);ends.append(exit_time<=step*(1+1e-12))
    t=np.array(exit_times);terminal=np.array(ends);q=x[None,:,:]+t[:,:,None]*velocities[:,None,:]
    scaled=np.clip((q+1)/h,0,n-1);left=np.minimum(np.floor(scaled).astype(int),n-2);frac=scaled-left
    indexes=[];weights=[]
    for corner in product((0,1),repeat=d):
        indexes.append(np.ravel_multi_index(tuple((left+corner)[...,j] for j in range(d)),(n,)*d))
        weights.append(np.prod(np.where(np.array(corner),frac,1-frac),axis=-1))
    indexes=np.array(indexes);weights=np.array(weights)
    g0=problem.g_float(x);g1=np.zeros_like(t);g2=np.zeros_like(t)
    for e,value in problem.definition['g']:
        e=np.array(e)
        for i in range(d):
            if e[i]:
                ee=e.copy();ee[i]-=1
                g1+=float(value)*e[i]*velocities[:,i,None]*np.prod(x**ee,axis=1)
                for j in range(d):
                    if ee[j]:
                        ff=ee.copy();ff[j]-=1
                        g2+=.5*float(value)*e[i]*ee[j]*velocities[:,i,None]*velocities[:,j,None]*np.prod(x**ff,axis=1)
    discount=np.exp(-t);running=g0*(-np.expm1(-t))+g1*(1-(t+1)*discount)+g2*(2-(t*t+2*t+2)*discount)
    u=np.zeros(len(grid));u[mask]=sum(abs(float(v)) for e,v in problem.definition['g'])
    for iteration in range(5000):
        continuation=np.sum(weights*u[indexes],axis=0);continuation[terminal]=0
        updated=np.min(running+discount*continuation,axis=0);change=float(np.max(abs(updated-u[mask])));u[mask]=updated
        if change<1e-11:break
    if change>=1e-11:raise ArithmeticError('Semi-Lagrangian iteration did not converge')
    return u.reshape((n,)*d),dict(iterations=iteration+1,discrete_update_defect=change,controls=len(velocities),seconds=time.perf_counter()-start)

HERMITE=np.array([[1,0,0,0],[0,0,1,0],[-3,3,-2,-1],[2,-2,1,1]],dtype=object)

class Input:
    def __init__(self,problem,n,record=None):
        self.problem=problem;self.d=problem.d;self.n=n;self.h=F(2,n-1);self.cache={}
        axis=np.linspace(-1,1,n);grid=np.array(list(product(axis,repeat=self.d)))
        if record is None:
            u,solver=solve(problem,n)
            sp=RegularGridInterpolator((axis,)*self.d,u,method='cubic',bounds_error=False,fill_value=None,solver_args=dict(rtol=1e-11,atol=1e-13))
            arrays=[]
            for nu in product((0,1),repeat=self.d):
                values=u.ravel() if not any(nu) else sp(grid,nu=nu)
                arrays.append(np.rint(values*2**42).astype(np.int64).reshape((n,)*self.d).tolist())
            record=dict(case=problem.name,n=n,dimension=self.d,bits=42,arrays=arrays,solver=solver)
        self.record=record;self.bits=record['bits'];self.arrays=[np.array(a,dtype=object) for a in record['arrays']]
        self.masks=list(product((0,1),repeat=self.d))
        u=np.asarray(self.arrays[0],dtype=float)/2**self.bits
        self.spline=RegularGridInterpolator((axis,)*self.d,u,method='cubic',bounds_error=False,fill_value=None,solver_args=dict(rtol=1e-11,atol=1e-13))
    def floating(self,x,second=True):
        v=self.spline(x);grad=[];H=[]
        for j in range(self.d):
            nu=[0]*self.d;nu[j]=1;grad.append(self.spline(x,nu=nu))
        if second:
            for i in range(self.d):
                row=[]
                for j in range(self.d):
                    nu=[0]*self.d;nu[i]+=1;nu[j]+=1;row.append(self.spline(x,nu=nu))
                H.append(row)
        return v,np.moveaxis(np.array(grad),0,-1),np.moveaxis(np.array(H),(0,1),(-2,-1)) if second else None
    def coefficients(self,cell):
        cell=tuple(cell)
        if cell in self.cache:return self.cache[cell]
        c=np.empty((4,)*self.d,dtype=object)
        for index in product(range(4),repeat=self.d):
            mask=tuple(i//2 for i in index);corner=tuple(k+i%2 for k,i in zip(cell,index))
            c[index]=F(int(self.arrays[self.masks.index(mask)][corner]),2**self.bits)*self.h**sum(mask)
        for axis in range(self.d):c=np.moveaxis(np.tensordot(HERMITE,c,axes=(1,axis)),0,axis)
        out=Polynomial(c);self.cache[cell]=out;return out
    def jet(self,x,cell):
        local=[(v-ball(-1+i*self.h))/ball(self.h) for v,i in zip(x,cell)]
        value,grad,_=self.coefficients(cell).jet(local,second=False)
        return value,[v/ball(self.h) for v in grad]
