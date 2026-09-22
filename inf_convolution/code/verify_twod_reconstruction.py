"""Whole-domain verification of the corrected two-dimensional inf-convolution.

The supplied bilinear approximation is the error target. A smoothed polynomial
fitted to its nodes supplies the contacts. No reference error is accessed here.
"""
from fractions import Fraction as F
from pathlib import Path
from math import comb
import argparse,hashlib,json,time
import numpy as np
from flint import arb,ctx
from interval_utils import ball,interval,upper_record,lower_record,floor_int
from twod_polynomials import power_coefficients,bernstein_bounds

ROOT=Path(__file__).resolve().parents[1]

def differentiate(p,axis):
    return {tuple(v-int(j==axis) for j,v in enumerate(k)):a*k[axis]
            for k,a in p.items() if k[axis]}

def evaluate(p,x,y):
    nx=max(k[0] for k in p);ny=max(k[1] for k in p);out=arb(0)
    for i in range(nx,-1,-1):
        row=arb(0)
        for j in range(ny,-1,-1):row=row*y+p.get((i,j),0)
        out=out*x+row
    return out

def restricted_bounds(p,limits):
    q={}
    for (i,j),c in p.items():
        for k in range(i+1):
            a=c*comb(i,k)*limits[0][0]**(i-k)*(limits[0][1]-limits[0][0])**k
            for l in range(j+1):
                v=a*comb(j,l)*limits[1][0]**(j-l)*(limits[1][1]-limits[1][0])**l
                q[k,l]=q.get((k,l),arb(0))+v
    return bernstein_bounds(q)

def problem_data(x,y):
    k=arb.pi()/2;c=(2*k*x).cos();d=(2*k*y).cos()
    speed=1+c*d/4
    cost=2-(k*x).sin()-(k*y).sin()+c*d/8
    cost+=speed*(k*((k*x).cos()+(k*y).cos())+k/4*((2*k*x).sin()*d+c*(2*k*y).sin()))
    return speed,cost

def problem_derivatives(x,y):
    k=arb.pi()/2;s=(k*x).sin();t=(k*y).sin();c=(k*x).cos();d=(k*y).cos()
    C=(2*k*x).cos();D=(2*k*y).cos();S=(2*k*x).sin();T=(2*k*y).sin()
    b=1+C*D/4;bx=-k*S*D/2;by=-k*C*T/2
    slopes=k*(c+d)+k/4*(S*D+C*T)
    sx=-k*k*s+k*k/2*(C*D-S*T);sy=-k*k*t+k*k/2*(C*D-S*T)
    gx=-k*c-k*S*D/4+bx*slopes+b*sx
    gy=-k*d-k*C*T/4+by*slopes+b*sy
    return b,bx,by,gx,gy

class Reconstruction:
    def __init__(self,record,exponent):
        self.record=record;self.n=record['n'];self.m=self.n//2
        self.nodes=np.array(record['node_numerators'],dtype=np.int64)
        co=record['polynomial'];matrix=np.array(co['numerators'],dtype=object).reshape(co['order']+1,co['order']+1)
        assert np.array_equal(matrix,matrix.T),'The strip proof uses coordinate-exchange symmetry.'
        self.eps=F(1,1<<exponent);self.ee=ball(self.eps);self.a=self.eps/2
        self.delta=arb(1)/record['smoothing_denominator'];self.s=(1+self.delta**2).sqrt()
        p=power_coefficients(record['polynomial'])
        assert all(i%2==j%2==0 for i,j in p)
        self.p={(i//2,j//2):v for (i,j),v in p.items()}
        self.px=differentiate(self.p,0);self.py=differentiate(self.p,1)
        self.pxx=differentiate(self.px,0);self.pyy=differentiate(self.py,1);self.pxy=differentiate(self.px,1)
        self.pxxx=differentiate(self.pxx,0);self.pxxy=differentiate(self.pxx,1)
        self.pxyy=differentiate(self.pyy,0);self.pyyy=differentiate(self.pyy,1)
        self.shifted={}

    def value(self,name,x,y):
        # Exact translation to a nearby dyadic center avoids cancellation in
        # power coefficients inherited from the Chebyshev fit. The polynomial
        # identity is valid on the entire input interval, without clipping.
        i=max(0,min(16,round(float(x.mid())*16)));j=max(0,min(16,round(float(y.mid())*16)))
        key=(name,i,j);a=arb(i)/16;b=arb(j)/16
        if key not in self.shifted:
            q={}
            for (r,s),c in getattr(self,name).items():
                for k in range(r+1):
                    for l in range(s+1):q[k,l]=q.get((k,l),arb(0))+c*comb(r,k)*comb(s,l)*a**(r-k)*b**(s-l)
            self.shifted[key]=q
        return evaluate(self.shifted[key],x-a,y-b)

    def jet(self,x,y,third=False):
        r=(x.lower()**2+self.delta**2).sqrt().union((x.upper()**2+self.delta**2).sqrt())
        q=(y.lower()**2+self.delta**2).sqrt().union((y.upper()**2+self.delta**2).sqrt())
        # This factored expression preserves t=0 at the periodic boundary.
        fx=(1+x)/(self.s+r);fy=(1+y)/(self.s+q)
        t=(self.s-r).max(arb(0));v=(self.s-q).max(arb(0));z=t**2;zz=v**2
        R=self.value('p',z,zz);Rx=self.value('px',z,zz);Ry=self.value('py',z,zz)
        Rxx=self.value('pxx',z,zz);Ryy=self.value('pyy',z,zz);Rxy=self.value('pxy',z,zz)
        A=2*fx*Rx*x/r;B=2*fy*Ry*y/q
        ux=-(1-x)*A;uy=-(1-y)*B
        hxx=(2*Rx+4*z*Rxx)*(x/r)**2-2*t*Rx*self.delta**2/r**3
        hyy=(2*Ry+4*zz*Ryy)*(y/q)**2-2*v*Ry*self.delta**2/q**3
        C=4*fx*fy*Rxy*x/r*y/q;hxy=(1-x)*(1-y)*C
        e=self.ee;jx=1+e*hxx;jy=1+e*hyy;jo=e*hxy;det=jx*jy-jo**2
        if not det>0:raise ArithmeticError('Jacobian determinant not yet positive.')
        Qx=((2*det-jy)*A+e*(1-y)**2*C*B)/det
        Qy=((2*det-jx)*B+e*(1-x)**2*C*A)/det
        wx=-(1-x)*Qx;wy=-(1-y)*Qy
        Y=x+e*ux;Z=y+e*uy;w=R+e*(ux**2+uy**2)
        out=dict(u=R,ux=ux,uy=uy,hxx=hxx,hyy=hyy,hxy=hxy,
                    jx=jx,jy=jy,jo=jo,det=det,qx=Qx,qy=Qy,
                    Y=Y,Z=Z,w=w,wx=wx,wy=wy)
        if third:
            Rxxx=self.value('pxxx',z,zz);Rxxy=self.value('pxxy',z,zz)
            Rxyy=self.value('pxyy',z,zz);Ryyy=self.value('pyyy',z,zz)
            tx=-x/r;vy=-y/q;txx=-self.delta**2/r**3;vyy=-self.delta**2/q**3
            txxx=3*self.delta**2*x/r**5;vyyy=3*self.delta**2*y/q**5
            out['xxx']=(12*t*Rxx+8*t**3*Rxxx)*tx**3+3*(2*Rx+4*z*Rxx)*tx*txx+2*t*Rx*txxx
            out['yyy']=(12*v*Ryy+8*v**3*Ryyy)*vy**3+3*(2*Ry+4*zz*Ryy)*vy*vyy+2*v*Ry*vyyy
            out['xxy']=(4*v*Rxy+8*z*v*Rxxy)*tx**2*vy+4*t*v*Rxy*txx*vy
            out['xyy']=(4*t*Rxy+8*t*zz*Rxyy)*tx*vy**2+4*t*v*Rxy*tx*vyy
        return out

    def residual_derivatives(self,z):
        e=self.ee;A=z['jx'];D=z['jy'];B=z['jo'];det=z['det']
        def inverse(v,w):return (D*v-B*w)/det,(A*w-B*v)/det
        h1,h2=inverse(z['ux'],z['uy'])
        b,bx,by,gx,gy=problem_derivatives(z['Y'],z['Z']);out=[]
        for hcol,tensor,jcol in [((z['hxx'],z['hxy']),(z['xxx'],z['xxy'],z['xyy']),(A,B)),
                                ((z['hxy'],z['hyy']),(z['xxy'],z['xyy'],z['yyy']),(B,D))]:
            ih=inverse(*hcol)
            it=inverse(tensor[0]*h1+tensor[1]*h2,tensor[1]*h1+tensor[2]*h2)
            vder=[2*hcol[k]-ih[k]+e*it[k] for k in range(2)]
            wder=(2*jcol[0]-(1 if len(out)==0 else 0))*z['ux']+(2*jcol[1]-(1 if len(out)==1 else 0))*z['uy']
            out.append(wder-(bx*jcol[0]+by*jcol[1])*(z['wx']+z['wy'])-b*sum(vder)-gx*jcol[0]-gy*jcol[1])
        return out

    def bilinear(self,x,y):
        m=self.m;n=self.n
        ix=range(floor_int((x.lower()+1)*m),floor_int((x.upper()+1)*m)+1)
        iy=range(floor_int((y.lower()+1)*m),floor_int((y.upper()+1)*m)+1)
        out=[None,None,None]
        for i in ix:
            s=(x*m+m-i).max(arb(0)).min(arb(1))
            for j in iy:
                t=(y*m+m-j).max(arb(0)).min(arb(1))
                a=int(self.nodes[i%n,j%n]);b=int(self.nodes[(i+1)%n,j%n])-a
                c=int(self.nodes[i%n,(j+1)%n])-a
                d=int(self.nodes[(i+1)%n,(j+1)%n])-a-b-c
                vals=[(a+b*s+c*t+d*s*t)/2**24,m*(b+d*t)/2**24,m*(c+d*s)/2**24]
                for k,v in enumerate(vals):out[k]=v if out[k] is None else out[k].union(v)
        return out

    def strip_condition(self):
        r=(ball(self.a)**2+self.delta**2).sqrt();t=self.s-r;t0=self.s-self.delta
        worst=arb(-100000)
        for k in range(16):
            lo,hi=restricted_bounds(self.px,[(t*t,t0*t0),(t0*t0*k/16,t0*t0*(k+1)/16)])
            v=1-self.ee*2*t*lo/r
            if not v<0:raise ArithmeticError('Pre-contact strip not excluded.')
            worst=worst.max(v.upper())
        return upper_record(worst)

    def monotonicity(self):
        # P_t/t=2 R_z, verified on a cube containing all regularized coordinates.
        lows=[]
        for p in [self.px,self.py]:
            lower=arb(100000)
            for i in range(8):
                for j in range(8):
                    lo,_=restricted_bounds(p,[(arb(i)/8,arb(i+1)/8),(arb(j)/8,arb(j+1)/8)])
                    if not lo>0:raise ArithmeticError('Polynomial monotonicity not proved.')
                    lower=lower.min(2*lo)
            lows.append(lower_record(lower))
        return lows

def split(box):
    a,b,c,d,depth=box
    if b-a>=d-c:
        m=(a+b)/2;return (a,m,c,d,depth+1),(m,b,c,d,depth+1)
    m=(c+d)/2;return (a,b,c,m,depth+1),(a,b,m,d,depth+1)

def initial_boxes(a,count=8):
    xs=[a+(1-a)*F(i,count) for i in range(count+1)]
    return [(xs[i],xs[i+1],xs[j],xs[j+1],0) for i in range(count) for j in range(count)]

def geometry(model):
    start=time.perf_counter();monotonicity=model.monotonicity();strip=model.strip_condition()
    pending=initial_boxes(model.a);lower={k:arb(100000) for k in ['jx','det','qx','qy']};leaves=0
    while pending:
        box=pending.pop();a,b,c,d,depth=box
        try:
            z=model.jet(interval(a,b),interval(c,d))
            passed=all(z[k]>0 for k in lower)
        except ArithmeticError:passed=False
        if passed:
            for k in lower:lower[k]=lower[k].min(z[k].lower())
            leaves+=1
        else:
            if depth>=24:raise ArithmeticError('Geometry subdivision failed.')
            pending.extend(split(box))
    return dict(monotonicity_lower=monotonicity,strip_factor_upper=strip,
                rectangle_lower=[model.a.numerator,model.a.denominator],
                lower_bounds={k:lower_record(v) for k,v in lower.items()},
                leaves=leaves,seconds=time.perf_counter()-start)

def bounds(model,relative_tolerance=F(1,20)):
    start=time.perf_counter();e=model.ee;rlow=arb(0);dlow=arb(0);rup=arb(0);dup=arb(0)
    pending=initial_boxes(model.a,16);leaves=pruned=evaluations=0;depth_max=0
    def center(a,b,c,d):
        x=ball((a+b)/2);y=ball((c+d)/2);z=model.jet(x,y)
        val=model.bilinear(z['Y'],z['Z'])[0];distance=z['w']-val
        valid=z['Y']>=0 and z['Z']>=0
        speed,cost=problem_data(z['Y'],z['Z'])
        residual=z['w']-speed*(z['wx']+z['wy'])-cost
        return z,distance,residual,valid
    for a,b,c,d,_ in pending:
        z,dd,rr,valid=center(a,b,c,d)
        if valid:rlow=rlow.max(abs(rr).abs_lower());dlow=dlow.max(abs(dd).abs_lower())
    rtol=ball(relative_tolerance);atol=arb(1)/10**8
    while pending:
        box=pending.pop();a,b,c,d,depth=box;evaluations+=1
        try:z=model.jet(interval(a,b),interval(c,d),third=True)
        except ArithmeticError:
            if depth>26:raise
            pending.extend(split(box));continue
        if z['Y']<0 or z['Z']<0:pruned+=1;continue
        zp,dp,rp,valid=center(a,b,c,d)
        if valid:rlow=rlow.max(abs(rp).abs_lower());dlow=dlow.max(abs(dp).abs_lower())
        speed,cost=problem_data(z['Y'].max(arb(0)).min(arb(1)),z['Z'].max(arb(0)).min(arb(1)))
        residual=z['w']-speed*(z['wx']+z['wy'])-cost;ru=abs(residual).upper()
        rx,ry=model.residual_derivatives(z)
        ru=ru.min((abs(rp)+abs(rx)*ball((b-a)/2)+abs(ry)*ball((d-c)/2)).upper())
        _,vx,vy=model.bilinear(z['Y'],z['Z'])
        # Centered mean-value enclosure of w(x)-U(F(x)), including grid joins.
        dx=(1+2*e*z['hxx'])*z['ux']+2*e*z['hxy']*z['uy']-z['jx']*vx-z['jo']*vy
        dy=2*e*z['hxy']*z['ux']+(1+2*e*z['hyy'])*z['uy']-z['jo']*vx-z['jy']*vy
        du=(abs(dp)+abs(dx)*ball((b-a)/2)+abs(dy)*ball((d-c)/2)).upper()
        if ru<=rlow*(1+rtol)+atol and du<=dlow*(1+rtol)+atol:
            rup=rup.max(ru);dup=dup.max(du);leaves+=1;depth_max=max(depth_max,depth)
        else:
            if depth>30 or evaluations>600000:raise ArithmeticError(f'Norm refinement budget: {evaluations}, {depth}')
            pending.extend(split(box))
        if evaluations%10000==0:print('  subdivision',evaluations,'pending',len(pending),'lower',float(rlow),float(dlow),'box',float(a),float(b),float(c),float(d),'upper',float(ru),float(du),'depth',depth,flush=True)
    rho=upper_record(rup);distance=upper_record(dup)
    return dict(residual_upper=rho,distance_upper=distance,error_upper=upper_record(ball(F(rho['numerator']+distance['numerator'],rho['denominator']))),
                lower_guides=dict(residual=lower_record(rlow),distance=lower_record(dlow)),
                leaves=leaves,pruned=pruned,evaluations=evaluations,maximum_depth=depth_max,seconds=time.perf_counter()-start)

def certify(record,exponent):
    model=Reconstruction(record,exponent);g=geometry(model)
    print('geometry',record['n'],exponent,g['leaves'],flush=True)
    return dict(exponent=exponent,geometry=g,**bounds(model))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--n',nargs='+',type=int,default=[32,64]);parser.add_argument('--exponents',nargs='+',type=int,default=[3,4,5,6,7]);parser.add_argument('--resume',action='store_true');args=parser.parse_args();ctx.prec=192
    path=ROOT/'results/twod_reconstruction.json'
    sources={f:hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest() for f in ['twod_input.py','twod_polynomials.py','verify_twod_reconstruction.py','interval_utils.py']}
    report=dict(status='incomplete',precision_bits=192,relative_tolerance=[1,20],absolute_tolerance='1e-8',source_sha256=sources,
                development='Order 4, smoothing n^-2 and the scale range were chosen after exploratory data-only calculations. All production scale candidates are retained.',records=[])
    if args.resume and path.exists():
        report=json.loads(path.read_text());assert report['source_sha256']==sources
    for n in args.n:
        inp=ROOT/f'results/twod_inputs/n{n}.json';record=json.loads(inp.read_text())
        row=next((r for r in report['records'] if r['n']==n),None)
        if row is None:row=dict(n=n,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),candidates=[]);report['records'].append(row)
        for j in args.exponents:
            if any(c['exponent']==j for c in row['candidates']):continue
            result=certify(record,j);row['candidates'].append(result)
            path.write_text(json.dumps(report,indent=2)+'\n')
            print(n,j,'B',result['error_upper']['decimal'],'rho',result['residual_upper']['decimal'],'distance',result['distance_upper']['decimal'],flush=True)
        best=min(row['candidates'],key=lambda c:F(c['error_upper']['numerator'],c['error_upper']['denominator']))
        row['selected_exponent']=best['exponent'];row['selected_error_upper']=best['error_upper']
    report['status']='rigorous_interval_enclosures';path.write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/twod_reconstruction.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
