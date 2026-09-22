"""Compare three certified bounds using the same data, precision and tolerances.

Direct reflection of the fitted polynomial, uncorrected inf-convolution, and
corrected inf-convolution all bound the error of the original bilinear input.
The two convolution methods select their scales independently from upper bounds.
Reference values and reference errors are not inputs to this module.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,platform,time
import numpy as np
from flint import ctx,arb
from interval_utils import ball,interval,upper_record,lower_record
from twod_input import fit
from verify_twod_reconstruction import Reconstruction,problem_data,problem_derivatives,initial_boxes,split

ROOT=Path(__file__).resolve().parents[1]
METHODS=('direct','uncorrected','corrected')
EXPONENTS=tuple(range(3,11))
RELATIVE_TOLERANCE=F(1,100)
ABSOLUTE_TOLERANCE=F(1,10**8)

class GeometryNotCertified(Exception):pass

class Candidate(Reconstruction):
    def __init__(self,record,method,exponent=None):
        super().__init__(record,exponent if exponent is not None else 0)
        assert method in METHODS;self.method=method
        if method=='direct':self.eps=F(0);self.ee=arb(0);self.a=F(0)

    def jet(self,x,y,third=False):
        if self.method!='direct':
            z=super().jet(x,y,third=third and self.method=='corrected')
            if self.method=='uncorrected':
                z['w']=z['u']+self.ee*(z['ux']**2+z['uy']**2)/2
                z['wx']=z['ux'];z['wy']=z['uy']
            return z
        t=1-x;v=1-y;s=t**2;r=v**2
        u=self.value('p',s,r);px=self.value('px',s,r);py=self.value('py',s,r)
        pxx=self.value('pxx',s,r);pyy=self.value('pyy',s,r);pxy=self.value('pxy',s,r)
        ux=-2*t*px;uy=-2*v*py
        return dict(u=u,w=u,ux=ux,uy=uy,wx=ux,wy=uy,
                    hxx=2*px+4*s*pxx,hyy=2*py+4*r*pyy,hxy=4*t*v*pxy,
                    jx=arb(1),jy=arb(1),jo=arb(0),det=arb(1),Y=x,Z=y)

    def parametric_gradient(self,z):
        factor={'direct':0,'uncorrected':1,'corrected':2}[self.method]*self.ee
        return ((1+factor*z['hxx'])*z['ux']+factor*z['hxy']*z['uy'],
                factor*z['hxy']*z['ux']+(1+factor*z['hyy'])*z['uy'])

    def residual_derivatives(self,z):
        if self.method=='corrected':return super().residual_derivatives(z)
        b,bx,by,gx,gy=problem_derivatives(z['Y'],z['Z'])
        px,py=self.parametric_gradient(z);summed=z['ux']+z['uy']
        rx=px-(bx*z['jx']+by*z['jo'])*summed-b*(z['hxx']+z['hxy'])-gx*z['jx']-gy*z['jo']
        ry=py-(bx*z['jo']+by*z['jy'])*summed-b*(z['hxy']+z['hyy'])-gx*z['jo']-gy*z['jy']
        return rx,ry

def prove_geometry(model):
    start=time.perf_counter();monotonicity=model.monotonicity()
    if model.method=='direct':return dict(monotonicity_lower=monotonicity,seconds=time.perf_counter()-start)
    failures=[]
    for multiplier in [F(1),F(1,2)]:
        model.a=model.eps*multiplier
        try:
            strip=model.strip_condition();pending=initial_boxes(model.a)
            keys=['jx','det']+(['qx','qy'] if model.method=='corrected' else [])
            lower={k:arb(100000) for k in keys};leaves=evaluations=0
            while pending:
                box=pending.pop();a,b,c,d,depth=box;evaluations+=1
                try:
                    z=model.jet(interval(a,b),interval(c,d));passed=all(z[k]>0 for k in keys)
                except ArithmeticError:passed=False
                if passed:
                    for k in keys:lower[k]=lower[k].min(z[k].lower())
                    leaves+=1
                else:
                    # A definitely negative point refutes this sufficient
                    # criterion on the proposed rectangle, not the method.
                    try:
                        point=model.jet(ball((a+b)/2),ball((c+d)/2))
                        if any(point[k]<0 for k in keys):raise GeometryNotCertified('A required geometry quantity is negative on this rectangle.')
                    except ArithmeticError:raise GeometryNotCertified('The proposed rectangle includes a singular contact map.')
                    if depth>=22 or evaluations>=60000:raise GeometryNotCertified('Geometry subdivision budget exceeded.')
                    pending.extend(split(box))
            return dict(monotonicity_lower=monotonicity,strip_factor_upper=strip,
                        rectangle_lower=[model.a.numerator,model.a.denominator],
                        lower_bounds={k:lower_record(v) for k,v in lower.items()},
                        leaves=leaves,preceding_rectangle_failures=failures,
                        seconds=time.perf_counter()-start)
        except (ArithmeticError,GeometryNotCertified) as error:
            failures.append(dict(multiplier=[multiplier.numerator,multiplier.denominator],reason=str(error)))
    raise GeometryNotCertified(json.dumps(failures,sort_keys=True))

def certify_norms(model):
    start=time.perf_counter();rlow=arb(0);dlow=arb(0);rup=arb(0);dup=arb(0)
    pending=initial_boxes(model.a,16);leaves=pruned=evaluations=depth_max=0
    def center(a,b,c,d):
        z=model.jet(ball((a+b)/2),ball((c+d)/2))
        dd=z['w']-model.bilinear(z['Y'],z['Z'])[0]
        speed,cost=problem_data(z['Y'],z['Z'])
        rr=z['w']-speed*(z['wx']+z['wy'])-cost
        return dd,rr,z['Y']>=0 and z['Z']>=0
    # Global lower bounds are established before depth-first refinement.
    for a,b,c,d,_ in pending:
        dd,rr,valid=center(a,b,c,d)
        if valid:rlow=rlow.max(abs(rr).abs_lower());dlow=dlow.max(abs(dd).abs_lower())
    if model.method!='direct':
        # Include rigorously valid points next to the mapped axes. Uniform
        # center samples can miss a maximum concentrated near a corner when
        # epsilon approaches the smoothing width. These remain lower bounds.
        probes=[None]+[model.a+(1-model.a)*F(i,8) for i in range(9)]
        for fixed in probes:
            lo,hi=model.a,F(1)
            for _ in range(64):
                mid=(lo+hi)/2;y=mid if fixed is None else fixed
                f=model.jet(ball(mid),ball(y))['Y']
                if f>0:hi=mid
                elif f<0:lo=mid
                else:break
            y=hi if fixed is None else fixed
            dd,rr,valid=center(hi,hi,y,y)
            if valid:rlow=rlow.max(abs(rr).abs_lower());dlow=dlow.max(abs(dd).abs_lower())
    relative=ball(RELATIVE_TOLERANCE);absolute=ball(ABSOLUTE_TOLERANCE)
    while pending:
        box=pending.pop();a,b,c,d,depth=box;evaluations+=1
        try:z=model.jet(interval(a,b),interval(c,d),third=True)
        except ArithmeticError:
            if depth>28:raise
            pending.extend(split(box));continue
        if z['Y']<0 or z['Z']<0:pruned+=1;continue
        dp,rp,valid=center(a,b,c,d)
        if valid:rlow=rlow.max(abs(rp).abs_lower());dlow=dlow.max(abs(dp).abs_lower())
        speed,cost=problem_data(z['Y'].max(arb(0)).min(arb(1)),z['Z'].max(arb(0)).min(arb(1)))
        direct_r=z['w']-speed*(z['wx']+z['wy'])-cost
        rx,ry=model.residual_derivatives(z)
        ru=abs(direct_r).upper().min((abs(rp)+abs(rx)*ball((b-a)/2)+abs(ry)*ball((d-c)/2)).upper())
        _,vx,vy=model.bilinear(z['Y'],z['Z']);px,py=model.parametric_gradient(z)
        dx=px-z['jx']*vx-z['jo']*vy;dy=py-z['jo']*vx-z['jy']*vy
        du=(abs(dp)+abs(dx)*ball((b-a)/2)+abs(dy)*ball((d-c)/2)).upper()
        if ru<=rlow*(1+relative)+absolute and du<=dlow*(1+relative)+absolute:
            rup=rup.max(ru);dup=dup.max(du);leaves+=1;depth_max=max(depth_max,depth)
        else:
            if depth>38 or evaluations>1000000:raise ArithmeticError(f'Norm subdivision budget: {evaluations}, {depth}, box={box}, upper={ru},{du}, lower={rlow},{dlow}')
            pending.extend(split(box))
    rho=upper_record(rup);distance=upper_record(dup)
    return dict(residual_upper=rho,distance_upper=distance,
                error_upper=upper_record(ball(F(rho['numerator']+distance['numerator'],rho['denominator']))),
                residual_lower=lower_record(rlow),distance_lower=lower_record(dlow),
                error_lower=lower_record(rlow+dlow),
                leaves=leaves,pruned=pruned,evaluations=evaluations,maximum_depth=depth_max,
                seconds=time.perf_counter()-start)

def certify(record,method,exponent=None):
    start=time.perf_counter();model=Candidate(record,method,exponent)
    try:g=prove_geometry(model)
    except GeometryNotCertified as error:
        return dict(method=method,exponent=exponent,status='not_certified',reason=str(error),total_seconds=time.perf_counter()-start)
    result=certify_norms(model)
    return dict(method=method,exponent=exponent,status='certified',geometry=g,**result,total_seconds=time.perf_counter()-start)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',nargs='+',type=int,default=[32,64])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--exponents',nargs='+',type=int,default=list(EXPONENTS))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();ctx.prec=192
    path=ROOT/'results/twod_method_comparison.json'
    files=['compare_twod_methods.py','verify_twod_reconstruction.py','twod_polynomials.py','twod_input.py','interval_utils.py']
    hashes={f:hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest() for f in files}
    report=dict(status='incomplete',source_sha256=hashes,precision_bits=192,
                exponents=list(EXPONENTS),relative_tolerance=[1,100],absolute_tolerance='1e-8',
                timing='One sequential full comparison run. Each method includes the common fit and its own complete scale search, geometry checks and norm verification. Input PDE solve, reference audit and imports are excluded.',
                environment=dict(python=platform.python_version(),platform=platform.platform(),processor='Intel Core i9-14900KF'),
                selection='Independent minimum rational upper bound for each method; no reference error.',records=[])
    if args.resume and path.exists():report=json.loads(path.read_text());assert report['source_sha256']==hashes
    for n in args.n:
        input_path=ROOT/f'results/twod_inputs/n{n}.json';record=json.loads(input_path.read_text())
        row=next((r for r in report['records'] if r['n']==n),None)
        if row is None:
            start=time.perf_counter();refitted=fit(np.asarray(record['node_numerators']));fitting=time.perf_counter()-start
            assert refitted==record['polynomial']
            row=dict(n=n,input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),fitting_seconds=fitting,methods=[]);report['records'].append(row)
        for method in args.methods:
            entry=next((r for r in row['methods'] if r['method']==method),None)
            if entry is None:entry=dict(method=method,candidates=[]);row['methods'].append(entry)
            for j in ([None] if method=='direct' else args.exponents):
                if any(r['exponent']==j for r in entry['candidates']):continue
                r=certify(record,method,j);entry['candidates'].append(r)
                path.write_text(json.dumps(report,indent=2)+'\n')
                print(n,method,j,r['status'],'B',r.get('error_upper',{}).get('decimal'),'seconds',round(r['total_seconds'],3),flush=True)
            valid=[r for r in entry['candidates'] if r['status']=='certified']
            assert valid,(n,method)
            chosen=min(valid,key=lambda r:F(r['error_upper']['numerator'],r['error_upper']['denominator']))
            entry.update(selected_exponent=chosen['exponent'],selected_error_upper=chosen['error_upper'],
                         total_seconds=row['fitting_seconds']+sum(r['total_seconds'] for r in entry['candidates']))
    report['status']='rigorous_interval_enclosures';path.write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/twod_method_comparison.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
