"""Replay saved exact reconstructions and independently check their algebra."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json
import numpy as np
import mpmath as mp
from flint import ctx
from interval_utils import ball,interval
from asymmetric_input import construct,PeriodicSpline
from robustness_design import install_cases
from adaptive_local import reconstruct,Context,build_patch,frac,Tapered
from certify_automatic_branches import certify_polynomial
from unknown_junction_2d import Input,Power,load_branches,certify,Verifier

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mq(x):
    x=F(x);return mp.mpf(x.numerator)/x.denominator
def contains(a,x):
    lo=a.lower().fmpq();hi=a.upper().fmpq()
    return mq(F(int(lo.numerator),int(lo.denominator)))<=x<=mq(F(int(hi.numerator),int(hi.denominator)))
def bounds(r,fields):return tuple((r[k]['numerator'],r[k]['denominator']) for k in fields)
def validate_file(name):
    p=ROOT/'results'/name;r=json.loads(p.read_text());assert r['status']=='completed'
    assert digest(p)==p.with_suffix('.sha256').read_text().strip()
    for source,sha in r['source_sha256'].items():assert digest(ROOT/'code'/source)==sha,(name,source)
    return r

def check_local(r):
    fields=['error_upper','residual_upper','distance_upper'];seen={};inputs={};replayed=independent=0
    records=[(row['input'],row['method'],row['result']) for row in r['fixed']]
    for row in r['targets']:
        if row['status']=='target_certified':
            assert frac(row['selected']['error_upper'])<=F(row['target'])
            assert row['selected_n']==row['input']['n']==row['history'][-1]['n']
            records.append((row['input'],row['method'],row['selected']))
        for step in row['history'][:-1]:
            result=step['result']
            assert result['status']!='certified' or frac(result['error_upper'])>F(row['target'])
    for record,method,saved in records:
        if saved['status']!='certified':continue
        key=(record['case'],record['n'])
        if key not in inputs:
            regenerated=construct(record['n'],record['case'])
            for field in ['coefficient_numerators','numerical_node_numerators']:assert regenerated[field]==record[field]
            inputs[key]=PeriodicSpline(record)
        spline=inputs[key]
        choice=saved.get('selected') if method!='global_polynomial' else [saved['width'],saved['degree']]
        skey=key+(method,json.dumps(choice,sort_keys=True))
        if skey in seen:
            assert bounds(saved,fields)==seen[skey];continue
        if method=='global_polynomial':new=certify_polynomial(spline,saved['width'],saved['degree'])
        else:new=reconstruct(spline,method=='adaptive_hybrid',selection=choice)
        assert new['status']=='certified'
        assert bounds(new,fields)==bounds(saved,fields),(key,method,bounds(new,fields),bounds(saved,fields))
        seen[skey]=bounds(saved,fields);replayed+=1
        # Independent differentiation of the tapered contact reconstruction.
        if method=='adaptive_hybrid':
            cc=Context(spline)
            for patch,candidate in zip(saved['history'],choice):
                if candidate['kind']!='corrected':continue
                pending=build_patch(cc,patch['peak'],patch['cap'],candidate)
                entries=[item for item in pending if any(isinstance(z,Tapered) for z,p in item[2])]
                for a,b,branches,_ in entries[::max(1,len(entries)//3)][:3]:
                    y=(a+b)/2
                    for z,part in branches[:2]:
                        if not isinstance(z,Tapered):continue
                        br=z.branch;lo,hi,i=br.pieces[part];eps=mq(br.eps);h=mq(spline.h)
                        coeff=list(map(mq,spline.coeffs[i%spline.n]));mid=mq((lo+hi)/2)
                        def uj(x):
                            t=x/h-i;A,B,C,D=coeff
                            return A+t*(B+t*(C+t*D)),(B+t*(2*C+3*D*t))/h,(2*C+6*D*t)/h**2
                        um,pm,Hm=uj(mid);linear=h*(1+eps*Hm);quadratic=3*eps*coeff[3]/h
                        local=list(map(mq,spline.coeffs[z.cell%spline.n]))
                        def fun(v):
                            off=v-mid-eps*pm;xx=mid+h*2*off/(linear+mp.sqrt(linear**2+4*quadratic*off))
                            uu,pp,_=uj(xx);w=uu+eps*pp*pp
                            t=v/h-z.cell;A,B,C,D=local;u=A+t*(B+t*(C+t*D))
                            s=(v-mq(z.a))/mq(z.b-z.a);theta=16*s*s*(1-s)**2
                            return u+theta*(w-u)
                        jets=z.jet(y,y,part)
                        for order in range(3):assert contains(jets[order],mp.diff(fun,mq(y),order))
                        independent+=1
    return dict(selected_reconstructions_replayed=replayed,distinct_inputs=len(inputs),independent_tapered_jets=independent,
                fixed_runs=len(r['fixed']),target_runs=len(r['targets']),source_sha256=r['source_sha256'])

def check_input(inp):
    """Exact coefficient identities, independent of point sampling."""
    n=inp.n-1;checks=0
    for i in range(n):
        for j in range(n):
            c=inp.coefficients(i,j).rational
            if i<n-1:
                d=inp.coefficients(i+1,j).rational
                for k in range(4):
                    assert sum(c[l][k] for l in range(4))==d[0][k]
                    assert sum(l*c[l][k] for l in range(1,4))==d[1][k]
                checks+=2
            if j<n-1:
                d=inp.coefficients(i,j+1).rational
                for k in range(4):
                    assert sum(c[k])==d[k][0]
                    assert sum(l*c[k][l] for l in range(1,4))==d[k][1]
                checks+=2
            if i==0:assert all(v==0 for v in c[0])
            if i==n-1:assert all(sum(c[l][k] for l in range(4))==0 for k in range(4))
            if j==0:assert all(row[0]==0 for row in c)
            if j==n-1:assert all(sum(row)==0 for row in c)
    return checks

def mp_input(inp,x,y,i,j):
    h=mq(inp.h);s=(x-(-1+i*h))/h;t=(y-(-1+j*h))/h
    basis=lambda z:[2*z**3-3*z*z+1,-2*z**3+3*z*z,z**3-2*z*z+z,z**3-z*z]
    bx,by=basis(s),basis(t);D=mp.mpf(2)**inp.bits;v=mp.mpf(0)
    for k in range(2):
        for l in range(2):
            u,ux,uy,uxy=[mp.mpf(int(a[i+k,j+l]))/D for a in inp.arrays]
            v+=u*bx[k]*by[l]+h*ux*bx[k+2]*by[l]+h*uy*bx[k]*by[l+2]+h*h*uxy*bx[k+2]*by[l+2]
    return v

def mp_data(x,y,case):
    q=mp.mpf
    if case=='variable_l1':return 1+q('.15')*x+q('.1')*x*y,1-q('.15')*y+q('.1')*x*y,1+q('.2')*x+q('.1')*y+q('.15')*x*y+q('.1')*x*x
    if case=='anisotropic_l2':return q('1.3')+q('.2')*y+q('.1')*x*y,q('.8')+q('.15')*x-q('.1')*x*y,1+q('.15')*x-q('.1')*y+q('.15')*x*y
    return 1+q('.25')*x*y+q('.1')*y*y,1-q('.2')*x*y+q('.15')*x*x,1+q('.2')*x+q('.1')*y+q('.2')*x*y+q('.15')*(x*x+y*y)

def independent_2d(inp,branches):
    functions=[]
    for branch in branches:
        c=[[mq(v) for v in row] for row in branch.rational]
        functions.append(lambda x,y,c=c:sum(v*x**i*y**j for i,row in enumerate(c) for j,v in enumerate(row)))
    ver=Verifier(inp,branches);count=0
    for k in range(12):
        i=(7*k+3)%(inp.n-1);j=(11*k+1)%(inp.n-1)
        box=(-1+i*inp.h,-1+(i+1)*inp.h,-1+j*inp.h,-1+(j+1)*inp.h,i,j,0);z=ver.box(box)
        x=box[0]+inp.h*F(3,7);y=box[2]+inp.h*F(2,5);xm,ym=mq(x),mq(y)
        iu=inp.jet(ball(x),ball(y),i,j)
        assert contains(iu[0],mp_input(inp,xm,ym,i,j))
        assert contains(iu[1],mp.diff(lambda t:mp_input(inp,t,ym,i,j),xm))
        assert contains(iu[2],mp.diff(lambda t:mp_input(inp,xm,t,i,j),ym))
        vals=[f(xm,ym) for f in functions];winner=min(range(4),key=lambda l:vals[l]);w=vals[winner];f=functions[winner]
        p=mp.diff(lambda t:f(t,ym),xm);q=mp.diff(lambda t:f(xm,t),ym);a,b,g=mp_data(xm,ym,inp.case)
        H=a*abs(p)+b*abs(q) if inp.case=='variable_l1' else mp.sqrt((a*p)**2+(b*q)**2)
        assert contains(interval(0,frac(dict(numerator=int(z[0].upper().fmpq().numerator),denominator=int(z[0].upper().fmpq().denominator)))),abs(w+H-g))
        ub=z[1].upper().fmpq();assert abs(w-mp_input(inp,xm,ym,i,j))<=mq(F(int(ub.numerator),int(ub.denominator)))
        count+=1
    return count

def junction_checks(branches):
    """Verify transverse branch crossings on rational horizontal slices."""
    polys=[np.asarray([[float(v) for v in row] for row in p.rational]) for p in branches];proofs=[]
    axis=np.linspace(-.9,.9,181)
    for yi in (-3,-2,-1,0,1,2,3):
        y=F(yi,5);vals=np.asarray([np.polynomial.polynomial.polyval2d(axis,np.full_like(axis,float(y)),c) for c in polys]);labels=np.argmin(vals,axis=0)
        for k in np.flatnonzero(labels[:-1]!=labels[1:]):
            a=F(-9,10)+F(int(k),100);b=a+F(1,100);left=int(labels[k]);right=int(labels[k+1]);m=(a+b)/2
            jets=[p.jet(interval(a,b),ball(y)) for p in branches];centers=[p.jet(ball(m),ball(y)) for p in branches]
            fa=branches[left].jet(ball(a),ball(y))[0]-branches[right].jet(ball(a),ball(y))[0]
            fb=branches[left].jet(ball(b),ball(y))[0]-branches[right].jet(ball(b),ball(y))[0]
            derivative=jets[left][1]-jets[right][1]
            if not (fa<0 and fb>0 and derivative>0):continue
            dominated=True
            for other in set(range(4))-{left,right}:
                diff=centers[other][0]-centers[left][0]+interval(-(b-a)/2,(b-a)/2)*(jets[other][1]-jets[left][1])
                if not diff>0:dominated=False
            if dominated:proofs.append(dict(y=str(y),x_interval=[str(a),str(b)],branches=[left,right],positive_transverse_derivative=True))
    return proofs

def check_twod(r):
    fields=['input_error_upper','reconstruction_error_upper','residual_upper','distance_upper','boundary_upper'];inputs={};seen={};replayed=point_checks=identities=0;crossings=[]
    for row in r['records']:
        key=(row['case'],row['n'])
        if key not in inputs:
            regenerated=Input(row['n'],row['case']);assert regenerated.record['arrays']==row['input']['arrays']
            inputs[key]=Input(row['n'],row['case'],row['input']);identities+=check_input(inputs[key])
        inp=inputs[key];best=min(row['attempts'],key=lambda a:F(a['certificate']['input_error_upper']['numerator'],a['certificate']['input_error_upper']['denominator']))
        assert best==row['selected']
        for attempt in row['attempts']:
            sig=key+(json.dumps(attempt['model']['branches'],sort_keys=True),)
            if sig in seen:assert bounds(attempt['certificate'],fields)==seen[sig];continue
            branches=load_branches(attempt['model']);new=certify(inp,branches,budget=16000)
            assert bounds(new,fields)==bounds(attempt['certificate'],fields),(key,row['method'])
            seen[sig]=bounds(new,fields);replayed+=1
            if attempt==best:
                point_checks+=independent_2d(inp,branches);proof=junction_checks(branches)
                assert len(proof)>0,(key,row['method'],'no transverse crossing verified')
                crossings.append(dict(case=row['case'],n=row['n'],method=row['method'],proofs=proof))
        print('verified',row['repetition'],row['case'],row['n'],row['method'],flush=True)
    return dict(candidate_certificates_replayed=replayed,distinct_inputs=len(inputs),exact_C1_coefficient_identities=identities,
        independent_80_digit_points=point_checks,junction_crossings=crossings,source_sha256=r['source_sha256'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--only',choices=['local','twod']);p.add_argument('--assemble',action='store_true');args=p.parse_args();ctx.prec=192;mp.mp.dps=80;install_cases();report={}
    if args.assemble:
        for name,datafile in [('local','adaptive_local.json'),('twod','unknown_junction_2d.json')]:
            saved=json.loads((ROOT/'verification'/f'adaptive_{name}_checks.json').read_text());validate_file(datafile)
            assert saved['status']=='PASS' and saved[name]['data_sha256']==digest(ROOT/'results'/datafile)
            report[name]=saved[name]
        report['status']='PASS';report['scope']='Assembly of separately completed replay and independent-check reports, with current data and source hashes revalidated.'
        (ROOT/'verification/adaptive_extensions_checks.json').write_text(json.dumps(report,indent=2)+'\n');return
    if args.only!='twod':
        r=validate_file('adaptive_local.json');report['local']=check_local(r);report['local']['data_sha256']=digest(ROOT/'results/adaptive_local.json')
    if args.only!='local':
        r=validate_file('unknown_junction_2d.json');report['twod']=check_twod(r);report['twod']['data_sha256']=digest(ROOT/'results/unknown_junction_2d.json')
    report['status']='PASS';report['scope']='Exact saved-certificate replay, coefficient identities and independent point checks. Point checks supplement the mathematical proof; they do not replace whole-domain verification.'
    path=ROOT/'verification'/('adaptive_extensions_checks.json' if args.only is None else f'adaptive_{args.only}_checks.json')
    path.write_text(json.dumps(report,indent=2)+'\n');print(path,flush=True)

if __name__=='__main__':main()
