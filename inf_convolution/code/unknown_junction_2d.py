"""Numerical branch discovery and certification without prescribed interfaces.

The input is an exact C1 bicubic Hermite interpolant of rounded numerical
nodes and derivatives. Four smooth polynomial branches are fitted to groups
identified from the numerical gradient. Their global minimum is continuous;
all possibly active branches are checked on every box, including junctions.
"""
from fractions import Fraction as F
import time,heapq
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize, LinearConstraint
from numpy.polynomial import chebyshev as ch
from flint import arb
from interval_utils import ball,interval,upper_record,lower_record

CASES=('variable_l1','anisotropic_l2','curved_l2')

def data(x,y,case):
    if case=='variable_l1':
        return 1+.15*x+.1*x*y,1-.15*y+.1*x*y,1+.2*x+.1*y+.15*x*y+.1*x*x
    if case=='anisotropic_l2':
        return 1.3+.2*y+.1*x*y,.8+.15*x-.1*x*y,1+.15*x-.1*y+.15*x*y
    if case=='curved_l2':
        return 1+.25*x*y+.1*y*y,1-.2*x*y+.15*x*x,1+.2*x+.1*y+.2*x*y+.15*(x*x+y*y)
    raise ValueError(case)

def data_arb(x,y,case):
    q=lambda n:ball(F(n))
    if case=='variable_l1':
        return (1+q('0.15')*x+q('.1')*x*y,1-q('.15')*y+q('.1')*x*y,
                1+q('.2')*x+q('.1')*y+q('.15')*x*y+q('.1')*x*x,
                q('.15')+q('.1')*y,q('.1')*x,q('.1')*y,-q('.15')+q('.1')*x,
                q('.2')+q('.15')*y+q('.2')*x,q('.1')+q('.15')*x)
    if case=='anisotropic_l2':
        return (q('1.3')+q('.2')*y+q('.1')*x*y,q('.8')+q('.15')*x-q('.1')*x*y,
                1+q('.15')*x-q('.1')*y+q('.15')*x*y,
                q('.1')*y,q('.2')+q('.1')*x,q('.15')-q('.1')*y,-q('.1')*x,
                q('.15')+q('.15')*y,-q('.1')+q('.15')*x)
    if case=='curved_l2':
        return (1+q('.25')*x*y+q('.1')*y*y,1-q('.2')*x*y+q('.15')*x*x,
                1+q('.2')*x+q('.1')*y+q('.2')*x*y+q('.15')*(x*x+y*y),
                q('.25')*y,q('.25')*x+q('.2')*y,-q('.2')*y+q('.3')*x,-q('.2')*x,
                q('.2')+q('.2')*y+q('.3')*x,q('.1')+q('.2')*x+q('.3')*y)
    raise ValueError(case)

def solve(n,case,tolerance=1e-12):
    """Monotone upwind solve with zero Dirichlet values on the square."""
    x=np.linspace(-1,1,n);X,Y=np.meshgrid(x,x,indexing='ij');a,b,g=data(X,Y,case);h=2/(n-1)
    u=g.copy();u[[0,-1],:]=0;u[:,[0,-1]]=0
    A=a[1:-1,1:-1]/h;B=b[1:-1,1:-1]/h;G=g[1:-1,1:-1]
    for iteration in range(20000):
        lx=np.minimum(u[:-2,1:-1],u[2:,1:-1]);ly=np.minimum(u[1:-1,:-2],u[1:-1,2:])
        vx=(G+A*lx)/(1+A);vy=(G+B*ly)/(1+B)
        if case=='variable_l1':z=np.minimum.reduce([G,vx,vy,(G+A*lx+B*ly)/(1+A+B)])
        else:
            # Roots of (G-z)^2=A^2(z-lx)^2+B^2(z-ly)^2,
            # with both upwind differences positive. Single-axis updates are
            # accepted only if the unused difference is nonpositive.
            qa=A*A+B*B-1;qb=2*(G-A*A*lx-B*B*ly);qc=A*A*lx*lx+B*B*ly*ly-G*G
            both=(-qb+np.sqrt(np.maximum(0,qb*qb-4*qa*qc)))/(2*qa)
            z=np.where(vx<=ly,vx,np.where(vy<=lx,vy,both));z=np.minimum(z,G)
        change=float(np.max(abs(z-u[1:-1,1:-1])));u[1:-1,1:-1]=z
        if change<tolerance:break
    if change>=tolerance:raise ArithmeticError('Numerical solver budget exhausted.')
    px=np.maximum(0,np.maximum(u[1:-1,1:-1]-u[:-2,1:-1],u[1:-1,1:-1]-u[2:,1:-1]))/h
    py=np.maximum(0,np.maximum(u[1:-1,1:-1]-u[1:-1,:-2],u[1:-1,1:-1]-u[1:-1,2:]))/h
    H=a[1:-1,1:-1]*px+b[1:-1,1:-1]*py if case=='variable_l1' else np.hypot(a[1:-1,1:-1]*px,b[1:-1,1:-1]*py)
    defect=float(np.max(abs(u[1:-1,1:-1]+H-G)))
    if defect>1e-9:raise ArithmeticError('Numerical upwind residual failed.')
    return x,u,dict(iterations=iteration+1,discrete_residual=defect)

HERMITE=np.array([[1,0,0,0],[0,0,1,0],[-3,3,-2,-1],[2,-2,1,1]],dtype=object)

class Input:
    def __init__(self,n,case,record=None):
        self.n=n;self.case=case;self.h=F(2,n-1);self.cache={}
        if record is None:
            grid,u,info=solve(n,case);sp=RectBivariateSpline(grid,grid,u,kx=3,ky=3,s=0)
            arrays=[u,sp(grid,grid,dx=1),sp(grid,grid,dy=1),sp(grid,grid,dx=1,dy=1)]
            # Tangential derivatives of the exact zero boundary are zero.
            arrays[1][:,[0,-1]]=0;arrays[2][[0,-1],:]=0
            record=dict(n=n,case=case,bits=42,arrays=[np.rint(v*2**42).astype(np.int64).tolist() for v in arrays],solver=info)
        self.record=record;self.bits=record['bits'];self.arrays=[np.asarray(a,dtype=object) for a in record['arrays']]
        self.float_spline=RectBivariateSpline(np.linspace(-1,1,n),np.linspace(-1,1,n),np.asarray(self.arrays[0],dtype=float)/2**self.bits,kx=3,ky=3,s=0)
    def coefficients(self,i,j):
        if (i,j) in self.cache:return self.cache[i,j]
        u,ux,uy,uxy=self.arrays;h=self.h;D=2**self.bits
        M=np.empty((4,4),dtype=object)
        for k in range(2):
            for l in range(2):
                M[k,l]=F(int(u[i+k,j+l]),D);M[k+2,l]=F(int(ux[i+k,j+l]),D)*h
                M[k,l+2]=F(int(uy[i+k,j+l]),D)*h;M[k+2,l+2]=F(int(uxy[i+k,j+l]),D)*h*h
        c=HERMITE@M@HERMITE.T;out=Power(c.tolist())
        self.cache[i,j]=out;return out
    def jet(self,x,y,i,j):
        xx=(x-ball(-1+i*self.h))/ball(self.h);yy=(y-ball(-1+j*self.h))/ball(self.h)
        v,p,q,*_=self.coefficients(i,j).jet(xx,yy)
        return v,p/ball(self.h),q/ball(self.h)

class Power:
    def __init__(self,coefficients):
        self.rational=[[F(c) for c in row] for row in coefficients]
        self.derivatives={}
        for dx,dy in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]:
            rows=[]
            for i in range(dx,len(coefficients)):
                row=[]
                for j in range(dy,len(coefficients[0])):
                    c=self.rational[i][j]
                    for k in range(dx):c*=i-k
                    for k in range(dy):c*=j-k
                    row.append(ball(c))
                rows.append(row)
            self.derivatives[dx,dy]=rows
    @staticmethod
    def horner(c,x,y):
        v=arb(0)
        for row in reversed(c):
            w=arb(0)
            for a in reversed(row):w=w*y+a
            v=v*x+w
        return v
    def jet(self,x,y):return tuple(self.horner(self.derivatives[k],x,y) for k in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)])

def basis(x,y,degree,dx=0,dy=0):
    ids=[(i,j) for i in range(degree+1) for j in range(degree+1-i)];cols=[]
    for i,j in ids:
        a=np.zeros(i+1);a[-1]=1;b=np.zeros(j+1);b[-1]=1
        if dx:a=ch.chebder(a,m=dx)
        if dy:b=ch.chebder(b,m=dy)
        cols.append(ch.chebval(x,a)*ch.chebval(y,b))
    return np.stack(cols,axis=1),ids

def fit(inp,degree=5,kind='polynomial',scale=1):
    started=time.perf_counter();h=float(inp.h);eps=scale*h
    # Interior sample points only; no exact values or interface coordinates.
    axis=np.linspace(-1+h/2,1-h/2,inp.n-1);X,Y=np.meshgrid(axis,axis,indexing='ij');x=X.ravel();y=Y.ravel();sp=inp.float_spline
    u=sp.ev(x,y);px=sp.ev(x,y,dx=1);py=sp.ev(x,y,dy=1)
    xx=sp.ev(x,y,dx=2);xy=sp.ev(x,y,dx=1,dy=1);yy=sp.ev(x,y,dy=2)
    # Orientation is inferred from the computed gradient, not the PDE solution.
    directions=np.array([px,-px,py,-py]);order=np.sort(directions,axis=0);labels=np.argmax(directions,axis=0)
    isolated=order[-1]>1.25*np.maximum(order[-2],1e-12)
    eig=(xx+yy-np.sqrt((xx-yy)**2+4*xy**2))/2
    retained=isolated & (1+eps*eig>=.5)
    if kind=='polynomial':tx=x;ty=y;v=u;gx=px;gy=py
    else:
        tx=x+eps*px;ty=y+eps*py;v=u+(.5 if kind=='uncorrected' else 1)*eps*(px*px+py*py)
        if kind=='uncorrected':gx=px;gy=py
        else:
            det=(1+eps*xx)*(1+eps*yy)-eps*eps*xy*xy
            invx=((1+eps*yy)*px-eps*xy*py)/det;invy=((1+eps*xx)*py-eps*xy*px)/det
            gx=2*px-invx;gy=2*py-invy
    V,ids=basis(tx,ty,degree-1);Dx,_=basis(tx,ty,degree-1,dx=1);Dy,_=basis(tx,ty,degree-1,dy=1)
    guard_axis=np.linspace(-1,1,17);guard_x,guard_y=np.meshgrid(guard_axis,guard_axis,indexing='ij')
    guard_x=guard_x.ravel();guard_y=guard_y.ravel();guard,_=basis(guard_x,guard_y,degree-1)
    floor=sp.ev(guard_x,guard_y)-.1*h
    records=[];branches=[]
    for label in range(4):
        mask=retained & (labels==label)
        if np.count_nonzero(mask)<len(ids):raise ArithmeticError('Insufficient samples for a branch.')
        distance=[tx+1,1-tx,ty+1,1-ty][label]
        guard_distance=[guard_x+1,1-guard_x,guard_y+1,1-guard_y][label]
        sx=[1,-1,0,0][label];sy=[0,0,1,-1][label]
        value_basis=distance[:,None]*V
        dx_basis=distance[:,None]*Dx+sx*V;dy_basis=distance[:,None]*Dy+sy*V
        weight=.15
        A=np.vstack([value_basis[mask],weight*dx_basis[mask],weight*dy_basis[mask]]);b=np.concatenate([v[mask],weight*gx[mask],weight*gy[mask]])
        c,_,rank,singular=np.linalg.lstsq(A,b,rcond=None)
        if rank<len(ids):raise ArithmeticError('Rank-deficient branch fit.')
        # These sampled inequalities stabilize extrapolation; they never
        # replace the subsequent whole-domain interval certificate.
        normal=A.T@A/len(b);rhs=A.T@b/len(b)
        objective=lambda z:float(.5*z@normal@z-rhs@z)
        gradient=lambda z:normal@z-rhs
        constraints=np.vstack([guard*guard_distance[:,None],guard])
        lower=np.concatenate([floor,np.full(len(guard),.1)])
        opt=minimize(objective,c,jac=gradient,constraints=LinearConstraint(constraints,lower,np.inf),
                     method='SLSQP',options=dict(maxiter=400,ftol=1e-12))
        if not opt.success:raise ArithmeticError('Constrained branch fit failed: '+opt.message)
        c=opt.x
        powers=np.zeros((degree+1,degree+1))
        for value,(i,j) in zip(c,ids):
            ci=np.zeros(i+1);ci[-1]=1;cj=np.zeros(j+1);cj[-1]=1
            outer=np.outer(ch.cheb2poly(ci),ch.cheb2poly(cj));powers[:i+1,:j+1]+=value*outer
        base_nums=np.rint(powers*2**44).astype(np.int64);nums=base_nums.copy()
        if label<2:nums[1:,:]+=sx*base_nums[:-1,:]
        else:nums[:,1:]+=sy*base_nums[:,:-1]
        branches.append(Power([[F(int(v),2**44) for v in row] for row in nums]))
        records.append(dict(label=label,bits=44,coefficients=nums.tolist(),samples=int(mask.sum()),condition=float(singular[0]/singular[-1]),fit_iterations=int(opt.nit)))
    return branches,dict(kind=kind,degree=degree,scale=scale,branches=records,fit_seconds=time.perf_counter()-started)

def load_branches(record):
    return [Power([[F(v,2**b['bits']) for v in row] for row in b['coefficients']]) for b in record['branches']]

def hamiltonian(a,b,px,py,case):
    return a*abs(px)+b*abs(py) if case=='variable_l1' else ((a*px)**2+(b*py)**2).sqrt()

class Verifier:
    def __init__(self,inp,branches):self.inp=inp;self.branches=branches;self.evaluations=0
    def box(self,box):
        self.evaluations+=1;xl,xr,yl,yr,i,j,depth=box
        xm=(xl+xr)/2;ym=(yl+yr)/2;x=interval(xl,xr);y=interval(yl,yr);mx=ball(xm);my=ball(ym)
        rx=interval(-(xr-xl)/2,(xr-xl)/2);ry=interval(-(yr-yl)/2,(yr-yl)/2)
        jets=[p.jet(x,y) for p in self.branches];centers=[p.jet(mx,my) for p in self.branches]
        possible=[k for k,v in enumerate(jets) if not any(k!=l and centers[k][0]-centers[l][0]+rx*(v[1]-w[1])+ry*(v[2]-w[2])>0 for l,w in enumerate(jets))]
        if not possible:raise ArithmeticError('Empty active set.')
        a,b,g,ax,ay,bx,by,gx,gy=data_arb(x,y,self.inp.case);am,bm,gm,*_=data_arb(mx,my,self.inp.case)
        um,_,_=self.inp.jet(mx,my,i,j);_,ux,uy=self.inp.jet(x,y,i,j)
        ru=arb(0);du=arb(0);rl=arb(0);val=centers[0][0]
        for c in centers[1:]:val=val.min(c[0])
        dl=abs(val-um).abs_lower()
        for k in possible:
            v,px,py,xx,xy,yy=jets[k];vm,pm,qm,*_=centers[k]
            R=vm+hamiltonian(am,bm,pm,qm,self.inp.case)-gm
            if self.inp.case=='variable_l1':
                sx=arb(1) if px>=0 else (arb(-1) if px<=0 else interval(-1,1))
                sy=arb(1) if py>=0 else (arb(-1) if py<=0 else interval(-1,1))
                hx=ax*abs(px)+a*sx*xx+bx*abs(py)+b*sy*xy
                hy=ay*abs(px)+a*sx*xy+by*abs(py)+b*sy*yy
            else:
                H=hamiltonian(a,b,px,py,self.inp.case)
                if H>0:
                    hx=(a*px*(ax*px+a*xx)+b*py*(bx*py+b*xy))/H
                    hy=(a*px*(ay*px+a*xy)+b*py*(by*py+b*yy))/H
                else:
                    lx=abs(ax*px+a*xx)+abs(bx*py+b*xy);ly=abs(ay*px+a*xy)+abs(by*py+b*yy)
                    hx=interval(-1,1)*lx;hy=interval(-1,1)*ly
            ru=ru.max(abs(R+rx*(px+hx-gx)+ry*(py+hy-gy)).upper())
            du=du.max(abs(vm-um+rx*(px-ux)+ry*(py-uy)).upper())
            if all(k==l or centers[k][0]<c[0] for l,c in enumerate(centers)):rl=rl.max(abs(R).abs_lower())
        return ru,du,rl,dl,len(possible)

def boundary_bound(branches,subdivisions=128):
    bound=arb(0)
    for side in range(4):
        restrictions=[]
        for p in branches:
            c=p.rational;sg=F(-1 if side in (0,2) else 1)
            if side<2:co=[sum(c[i][j]*sg**i for i in range(len(c))) for j in range(len(c[0]))]
            else:co=[sum(c[i][j]*sg**j for j in range(len(c[0]))) for i in range(len(c))]
            restrictions.append([ball(v) for v in co])
        for k in range(subdivisions):
            t=interval(F(-1)+F(2*k,subdivisions),F(-1)+F(2*(k+1),subdivisions))
            vals=[]
            for co in restrictions:
                value=arb(0)
                for a in reversed(co):value=value*t+a
                vals.append(value)
            v=vals[0]
            for w in vals[1:]:v=v.min(w)
            bound=bound.max(abs(v).upper())
    return bound

def sampled_score(inp,branches,n=65):
    """Proposal ranking only. Samples are never upper certificates."""
    axis=np.linspace(-1,1,n);x,y=np.meshgrid(axis,axis,indexing='ij');x=x.ravel();y=y.ravel();values=[];px=[];py=[]
    for p in branches:
        c=np.asarray([[float(v) for v in row] for row in p.rational])
        values.append(np.polynomial.polynomial.polyval2d(x,y,c))
        px.append(np.polynomial.polynomial.polyval2d(x,y,np.polynomial.polynomial.polyder(c,axis=0)))
        py.append(np.polynomial.polynomial.polyval2d(x,y,np.polynomial.polynomial.polyder(c,axis=1)))
    values=np.asarray(values);win=np.argmin(values,axis=0);ids=np.arange(len(x));v=values[win,ids]
    p=np.asarray(px)[win,ids];q=np.asarray(py)[win,ids];a,b,g=data(x,y,inp.case)
    H=a*abs(p)+b*abs(q) if inp.case=='variable_l1' else np.hypot(a*p,b*q)
    residual=float(np.max(abs(v+H-g)));distance=float(np.max(abs(v-inp.float_spline.ev(x,y))))
    return residual+distance

def certify(inp,branches,relative=F(1,20),absolute=F(1,1000000),budget=120000):
    started=time.perf_counter();ver=Verifier(inp,branches);rl=arb(0);dl=arb(0);ru=arb(0);du=arb(0);queue=[];leaves=junctions=0;maxdepth=0;limited=0
    for i in range(inp.n-1):
        for j in range(inp.n-1):
            box=(-1+i*inp.h,-1+(i+1)*inp.h,-1+j*inp.h,-1+(j+1)*inp.h,i,j,0)
            z=ver.box(box);rl=rl.max(z[2]);dl=dl.max(z[3]);queue.append((box,z))
    while queue:
        box,z=queue.pop()
        if z is None:z=ver.box(box);rl=rl.max(z[2]);dl=dl.max(z[3])
        tight=z[0]<=rl*ball(1+relative)+ball(absolute) and z[1]<=dl*ball(1+relative)+ball(absolute)
        # A subdivision limit affects tightness only: every unsplit box keeps
        # its rigorous upper enclosure, and is included in the final maximum.
        capped=box[-1]>=12 or ver.evaluations>=budget
        if tight or capped:
            limited+=not tight
            ru=ru.max(z[0]);du=du.max(z[1]);leaves+=1;junctions+=z[4]>1;maxdepth=max(maxdepth,box[-1]);continue
        xl,xr,yl,yr,i,j,d=box
        if xr-xl>=yr-yl:
            m=(xl+xr)/2;children=[(xl,m,yl,yr,i,j,d+1),(m,xr,yl,yr,i,j,d+1)]
        else:
            m=(yl+yr)/2;children=[(xl,xr,yl,m,i,j,d+1),(xl,xr,m,yr,i,j,d+1)]
        queue.extend((b,None) for b in children)
    bc=boundary_bound(branches);error=du+ru.max(bc)
    return dict(status='certified',input_error_upper=upper_record(error),reconstruction_error_upper=upper_record(ru.max(bc)),
        residual_upper=upper_record(ru),distance_upper=upper_record(du),boundary_upper=upper_record(bc),
        residual_lower=lower_record(rl),distance_lower=lower_record(dl),evaluations=ver.evaluations,leaves=leaves,
        junction_boxes=junctions,maximum_depth=maxdepth,tightness_limited_boxes=limited,seconds=time.perf_counter()-started)
