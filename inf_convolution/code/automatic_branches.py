"""Discover monotone contact branches of a periodic spline from its coefficients."""
from fractions import Fraction as F
import bisect
import numpy as np
from flint import arb
from interval_utils import ball,interval

class BranchFailure(Exception):pass

def components(spline,eps,threshold=F(1,2)):
    h=spline.h;pieces=[]
    for i in range(spline.n):
        jl=1+eps*spline.local_exact(i,F(0),2);jr=1+eps*spline.local_exact(i,F(1),2)
        if max(jl,jr)<threshold:continue
        a,b=i*h,(i+1)*h
        if min(jl,jr)<threshold:
            root=a+h*(threshold-jl)/(jr-jl)
            if jl<threshold:a=root
            else:b=root
        if a<b:pieces.append((a,b,i))
    if not pieces:raise BranchFailure('No interval has the required contact Jacobian.')
    blocks=[]
    for piece in pieces:
        if blocks and blocks[-1][-1][1]==piece[0]:blocks[-1].append(piece)
        else:blocks.append([piece])
    periodic=len(blocks)==1 and blocks[0][0][0]==0 and blocks[0][-1][1]==2
    if len(blocks)>1 and blocks[0][0][0]==0 and blocks[-1][-1][1]==2:
        joined=blocks[-1]+[(a+2,b+2,i+spline.n) for a,b,i in blocks[0]]
        blocks=blocks[1:-1]+[joined]
    return blocks,periodic

class Branch:
    def __init__(self,spline,eps,pieces,number):
        self.spline=spline;self.eps=eps;self.ee=ball(eps);self.number=number;self.pieces=pieces
        self.edges=[a+eps*spline.point(a,1) for a,_,_ in pieces]
        b=pieces[-1][1];self.edges.append(b+eps*spline.point(b,1))
        assert all(a<b for a,b in zip(self.edges,self.edges[1:]))
        self.left,self.right=self.edges[0],self.edges[-1]
        self.prepare_float()

    def prepare_float(self):
        self.fedges=np.array([float(e) for e in self.edges])
        self.fmid=np.array([float((a+b)/2) for a,b,_ in self.pieces])
        self.indices=np.array([i for _,_,i in self.pieces])

    def translated(self,k):
        if k==0:return self
        out=object.__new__(Branch);out.spline=self.spline;out.eps=self.eps;out.ee=self.ee;out.number=self.number
        out.pieces=[(a+2*k,b+2*k,i+k*self.spline.n) for a,b,i in self.pieces]
        out.edges=[e+2*k for e in self.edges];out.left=out.edges[0];out.right=out.edges[-1]
        out.prepare_float();return out

    def part(self,y):return max(0,min(len(self.pieces)-1,bisect.bisect_right(self.edges,y)-1))

    def inverse(self,y,part):
        a,b,i=self.pieces[part];s=self.spline;h=ball(s.h)
        mid=ball((a+b)/2);u,p,H,_=s.jet(i,mid)
        derivative=h*(1+self.ee*H);quadratic=3*self.ee*s.acoeffs[i%s.n][3]/h
        offset=y-mid-self.ee*p
        rad=derivative**2+4*quadratic*offset
        if not rad>0:raise ArithmeticError('Contact inverse discriminant needs subdivision.')
        return mid+h*2*offset/(derivative+rad.sqrt())

    def jet(self,left,right,part,kind):
        a,b,i=self.pieces[part]
        xl=self.inverse(ball(left),part);xr=self.inverse(ball(right),part);x=xl.union(xr)
        u,p,H,T=self.spline.jet(i,x);J=1+self.ee*H
        if not J>0:raise ArithmeticError('Contact Jacobian enclosure needs subdivision.')
        if kind=='uncorrected':value=u+self.ee*p**2/2;derivative=p;second=H/J
        elif kind=='corrected':
            value=u+self.ee*p**2;derivative=p*(2-1/J)
            second=(H*(2-1/J)+self.ee*p*T/J**2)/J
        else:raise ValueError(kind)
        return value,derivative,second,x

    def float_values(self,y,kind):
        yy=np.asarray(y);valid=(yy>=float(self.left))&(yy<=float(self.right))
        out=np.full(yy.shape,np.inf);derivative=np.full(yy.shape,np.nan)
        if not np.any(valid):return out,derivative
        z=yy[valid];part=np.clip(np.searchsorted(self.fedges,z,side='right')-1,0,len(self.pieces)-1)
        mid=self.fmid[part]
        u,p,H,_=self.spline.float_jet(mid);h=float(self.spline.h);eps=float(self.eps)
        indices=self.indices[part];A=3*eps*self.spline.fcoeffs[indices%self.spline.n,3]/h
        B=h*(1+eps*H);offset=z-mid-eps*p
        x=mid+h*2*offset/(B+np.sqrt(np.maximum(0,B*B+4*A*offset)))
        u,p,H,_=self.spline.float_jet(x);J=1+eps*H
        out[valid]=u+(1 if kind=='corrected' else .5)*eps*p*p
        derivative[valid]=p*(2-1/J) if kind=='corrected' else p
        return out,derivative

def discover(spline,eps):
    blocks,periodic=components(spline,eps);branches=[]
    nums=spline.record['coefficient_numerators'];bits=spline.record['coefficient_bits']
    lip=F(max(abs(a-b) for a,b in zip(nums,nums[1:]+nums[:1])),2**bits)/spline.h
    margin=eps*lip
    for number,block in enumerate(blocks):
        base=Branch(spline,eps,block,number)
        first=-int(-(-margin-base.right)/2//1);last=int((2+margin-base.left)/2//1)
        for k in range(first,last+1):branches.append(base.translated(k))
    return branches,periodic,len(blocks)

def float_envelope(branches,y,kind):
    values=[];derivatives=[]
    for b in branches:
        v,d=b.float_values(y,kind);values.append(v);derivatives.append(d)
    a=np.array(values);ds=np.array(derivatives);index=np.argmin(a,axis=0)
    return np.take_along_axis(a,index[None,:],axis=0)[0],np.take_along_axis(ds,index[None,:],axis=0)[0],index

def discover_float(spline,eps):
    """Cheap branch proposals for ordering candidates; never used as certificates."""
    n=spline.n;h=float(spline.h);ids=np.arange(n);left=ids*h;right=(ids+1)*h
    Hleft=spline.fcoeffs[:,2]*2/h**2;Hright=(2*spline.fcoeffs[:,2]+6*spline.fcoeffs[:,3])/h**2
    jl=1+eps*Hleft;jr=1+eps*Hright;keep=np.maximum(jl,jr)>=.5
    crossing=(np.minimum(jl,jr)<.5)&keep
    fraction=np.zeros(n);fraction[crossing]=(.5-jl[crossing])/(jr[crossing]-jl[crossing])
    left=np.where(crossing&(jl<.5),left+h*fraction,left)
    right=np.where(crossing&(jr<.5),ids*h+h*fraction,right)
    pieces=[(a,b,int(i)) for a,b,i in zip(left[keep],right[keep],ids[keep]) if b>a]
    blocks=[]
    for p in pieces:
        if blocks and abs(blocks[-1][-1][1]-p[0])<1e-14:blocks[-1].append(p)
        else:blocks.append([p])
    if len(blocks)>1 and blocks[0][0][0]==0 and blocks[-1][-1][1]==2:
        joined=blocks[-1]+[(a+2,b+2,i+n) for a,b,i in blocks[0]];blocks=blocks[1:-1]+[joined]
    result=[]
    for block in blocks:
        x=np.array([a for a,_,_ in block]+[block[-1][1]]);edges=x+eps*spline.float_jet(x)[1]
        for shift in range(-2,3):
            if edges[-1]+2*shift<0 or edges[0]+2*shift>2:continue
            b=object.__new__(Branch);b.spline=spline;b.eps=eps;b.pieces=block
            b.left=edges[0]+2*shift;b.right=edges[-1]+2*shift;b.fedges=edges+2*shift
            b.fmid=np.array([(a+c)/2+2*shift for a,c,_ in block]);b.indices=np.array([i+shift*n for _,_,i in block])
            result.append(b)
    return result
