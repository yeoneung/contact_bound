"""Replay all three-method certificates and independently check their formulas."""
from pathlib import Path
from fractions import Fraction as F
import contextlib,hashlib,io,json
import mpmath as mp
import numpy as np
from flint import arb,ctx
from interval_utils import ball,upper_record
from compare_twod_methods import Candidate,certify,METHODS,EXPONENTS
from check_twod_reconstruction import stable
from audit_twod_reference import main as audit_reference
from twod_solver import solve
from twod_input import fit

ROOT=Path(__file__).resolve().parents[1]

def rational(record):return F(record['numerator'],record['denominator'])

def independent(record,method,exponent=6):
    model=Candidate(record,method,None if method=='direct' else exponent)
    co=record['polynomial'];order=co['order'];den=mp.mpf(2)**co['denominator_bits']
    eta=mp.mpf(1)/record['smoothing_denominator'];eps=mp.mpf(0) if method=='direct' else mp.mpf(2)**-exponent
    def u(x,y):
        if method=='direct':t,s=1-x,1-y
        else:t,s=[mp.sqrt(1+eta**2)-mp.sqrt(v*v+eta**2) for v in [x,y]]
        return sum(mp.mpf(co['numerators'][i*(order+1)+j])/den*mp.chebyt(2*i,t)*mp.chebyt(2*j,s)
                   for i in range(order+1) for j in range(order+1))
    def quantities(x,y):
        p=mp.matrix([mp.diff(lambda v:u(v,y),x),mp.diff(lambda v:u(x,v),y)])
        H=mp.matrix([[mp.diff(u,(x,y),(2,0)),mp.diff(u,(x,y),(1,1))],
                     [mp.diff(u,(x,y),(1,1)),mp.diff(u,(x,y),(0,2))]])
        J=mp.eye(2)+eps*H;Y=mp.matrix([x,y])+eps*p
        gradient=(2*mp.eye(2)-J**-1)*p if method=='corrected' else p
        factor=mp.mpf(1) if method=='corrected' else mp.mpf('0.5')
        value=u(x,y)+factor*eps*(p[0]**2+p[1]**2)
        X,Z=Y;c,d=mp.cos(mp.pi*X),mp.cos(mp.pi*Z);k=mp.pi/2;b=1+c*d/4
        g=2-mp.sin(k*X)-mp.sin(k*Z)+c*d/8+b*(k*(mp.cos(k*X)+mp.cos(k*Z))+mp.pi/8*(mp.sin(mp.pi*X)*d+c*mp.sin(mp.pi*Z)))
        return value-b*(gradient[0]+gradient[1])-g,value,gradient
    for xx,yy in [('0.07','0.36'),('0.74','0.85')]:
        x,y=mp.mpf(xx),mp.mpf(yy);z=model.jet(arb(xx),arb(yy),third=True)
        _,value,gradient=quantities(x,y)
        def contains(a,v):assert a.contains(arb(mp.nstr(v,65))),(method,a,v)
        contains(z['w'],value)
        for name,v in zip(['wx','wy'],gradient):contains(z[name],v)
        for a,orders in zip(model.residual_derivatives(z),[(1,0),(0,1)]):
            contains(a,mp.diff(lambda s,t:quantities(s,t)[0],(x,y),orders))
        for a,orders in zip(model.parametric_gradient(z),[(1,0),(0,1)]):
            contains(a,mp.diff(lambda s,t:quantities(s,t)[1],(x,y),orders))

def main():
    ctx.prec=192;mp.mp.dps=70;path=ROOT/'results/twod_method_comparison.json'
    sha=hashlib.sha256(path.read_bytes()).hexdigest();data=json.loads(path.read_text())
    assert sha==(ROOT/'results/twod_method_comparison.sha256').read_text().strip()
    assert data['status']=='rigorous_interval_enclosures' and data['precision_bits']==192
    assert data['exponents']==list(EXPONENTS) and data['relative_tolerance']==[1,100]
    assert data['absolute_tolerance']=='1e-8'
    for name,expected in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected,name
    assert sorted(r['n'] for r in data['records'])==[32,64]
    count=excluded=0
    for row in data['records']:
        inp=ROOT/f"results/twod_inputs/n{row['n']}.json";record=json.loads(inp.read_text())
        assert hashlib.sha256(inp.read_bytes()).hexdigest()==row['input_sha256']
        nodes=np.asarray(record['node_numerators'])
        assert np.array_equal(nodes,solve(row['n'])) and fit(nodes)==record['polynomial']
        assert sorted(m['method'] for m in row['methods'])==sorted(METHODS)
        for entry in row['methods']:
            method=entry['method'];independent(record,method)
            assert [c['exponent'] for c in entry['candidates']]==([None] if method=='direct' else list(EXPONENTS))
            for candidate in entry['candidates']:
                replayed=certify(record,method,candidate['exponent'])
                assert stable(replayed)==stable(candidate),(row['n'],method,candidate['exponent'])
                if candidate['status']=='certified':
                    count+=1
                    for name in ['residual','distance']:
                        lo=rational(candidate[name+'_lower']);hi=rational(candidate[name+'_upper'])
                        assert 0<=lo<=hi and hi<=lo*F(101,100)+F(1,10**8)+F(1,2**95)
                else:excluded+=1
                print('Replayed',row['n'],method,candidate['exponent'],candidate['status'],flush=True)
            best=min((c for c in entry['candidates'] if c['status']=='certified'),key=lambda c:rational(c['error_upper']))
            assert entry['selected_exponent']==best['exponent'] and entry['selected_error_upper']==best['error_upper']
            assert abs(entry['total_seconds']-row['fitting_seconds']-sum(c['total_seconds'] for c in entry['candidates']))<1e-10
    # The preserved reference is evaluated only after all method selections
    # are frozen. It has no role in construction, geometry or scale choice.
    with contextlib.redirect_stdout(io.StringIO()):audit_reference()
    refs=json.loads((ROOT/'results/twod_reference_audit.json').read_text());assessed=[]
    for row in data['records']:
        ref=next(r for r in refs['records'] if r['n']==row['n']);methods=[]
        for entry in row['methods']:
            B=rational(entry['selected_error_upper']);lo=rational(ref['error_lower'])
            methods.append(dict(method=entry['method'],selected_exponent=entry['selected_exponent'],
                                error_upper=entry['selected_error_upper'],effectivity_upper=upper_record(ball(B/lo))))
        assessed.append(dict(n=row['n'],error_lower=ref['error_lower'],error_upper=ref['error_upper'],methods=methods))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    report=dict(status='PASS',comparison_sha256=sha,certificates_replayed=count,
                uncertified_candidates_replayed=excluded,numerical_inputs_recomputed=2,
                reference_intervals_replayed=2,independent_precision_digits=70,
                independent_formula_points=12,common_relative_tolerance=[1,100],
                method_selections_replayed=6,records=assessed)
    (ROOT/'verification/twod_method_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
