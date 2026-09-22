"""Independent formula checks and full replay of the coupled 2D certificates."""
from pathlib import Path
from fractions import Fraction as F
import contextlib,hashlib,io,json
import numpy as np
import mpmath as mp
from flint import ctx,arb
from twod_input import fit
from twod_solver import solve
from verify_twod_reconstruction import Reconstruction,certify
from audit_twod_reference import main as audit_reference

ROOT=Path(__file__).resolve().parents[1]

def stable(d):
    if isinstance(d,dict):return {k:stable(v) for k,v in d.items() if not k.endswith('seconds')}
    if isinstance(d,list):return [stable(v) for v in d]
    return d

def independent(record,exponent):
    model=Reconstruction(record,exponent);co=record['polynomial'];order=co['order']
    den=mp.mpf(2)**co['denominator_bits'];delta=mp.mpf(1)/record['smoothing_denominator'];eps=mp.mpf(1)/2**exponent
    def u(x,y):
        t=mp.sqrt(1+delta**2)-mp.sqrt(x*x+delta**2);s=mp.sqrt(1+delta**2)-mp.sqrt(y*y+delta**2)
        return sum(mp.mpf(co['numerators'][i*(order+1)+j])/den*mp.chebyt(2*i,t)*mp.chebyt(2*j,s) for i in range(order+1) for j in range(order+1))
    def quantities(x,y):
        grad=mp.matrix([mp.diff(lambda a:u(a,y),x),mp.diff(lambda b:u(x,b),y)])
        H=mp.matrix([[mp.diff(lambda a:u(a,y),x,2),mp.diff(u,(x,y),(1,1))],
                     [mp.diff(u,(x,y),(1,1)),mp.diff(lambda b:u(x,b),y,2)]])
        J=mp.eye(2)+eps*H;Y=mp.matrix([x,y])+eps*grad;dw=(2*mp.eye(2)-J**-1)*grad
        k=mp.pi/2;X,Z=Y;c=mp.cos(mp.pi*X);d=mp.cos(mp.pi*Z);b=1+c*d/4
        g=2-mp.sin(k*X)-mp.sin(k*Z)+c*d/8+b*(k*(mp.cos(k*X)+mp.cos(k*Z))+mp.pi/8*(mp.sin(mp.pi*X)*d+c*mp.sin(mp.pi*Z)))
        w=u(x,y)+eps*(grad[0]**2+grad[1]**2)
        return w-b*(dw[0]+dw[1])-g,w,dw
    for xx,yy in [('0.041','0.37'),('0.71','0.82')]:
        x=mp.mpf(xx);y=mp.mpf(yy);z=model.jet(arb(xx),arb(yy),third=True)
        def contains(a,v):assert a.contains(arb(mp.nstr(v,65))),(a,v)
        residual,w,dw=quantities(x,y)
        contains(z['w'],w);contains(z['wx'],dw[0]);contains(z['wy'],dw[1])
        for name,orders in [('xxx',(3,0)),('xxy',(2,1)),('xyy',(1,2)),('yyy',(0,3))]:
            contains(z[name],mp.diff(u,(x,y),orders))
        for a,orders in zip(model.residual_derivatives(z),[(1,0),(0,1)]):
            contains(a,mp.diff(lambda s,t:quantities(s,t)[0],(x,y),orders))
    # Check bilinear values and derivatives independently with exact fractions,
    # including both sides of a grid join and a reflected negative coordinate.
    for x,y in [(F(3,17),F(7,19)),(F(-1,64),F(5,32)),(F(1),F(1))]:
        m=model.m;n=model.n;i=int((x+1)*m//1);j=int((y+1)*m//1);s=(x+1)*m-i;t=(y+1)*m-j
        a=F(int(model.nodes[i%n,j%n]),2**24);b=F(int(model.nodes[(i+1)%n,j%n]),2**24)
        c=F(int(model.nodes[i%n,(j+1)%n]),2**24);d=F(int(model.nodes[(i+1)%n,(j+1)%n]),2**24)
        vals=[a*(1-s)*(1-t)+b*s*(1-t)+c*(1-s)*t+d*s*t,
              m*((b-a)*(1-t)+(d-c)*t),m*((c-a)*(1-s)+(d-b)*s)]
        actual=model.bilinear(arb(x.numerator)/x.denominator,arb(y.numerator)/y.denominator)
        for enclosing,exact in zip(actual,vals):assert enclosing.contains(arb(exact.numerator)/exact.denominator)

def main():
    ctx.prec=192;mp.mp.dps=70;path=ROOT/'results/twod_reconstruction.json'
    sha=hashlib.sha256(path.read_bytes()).hexdigest();data=json.loads(path.read_text())
    assert data['status']=='rigorous_interval_enclosures'
    assert sha==(ROOT/'results/twod_reconstruction.sha256').read_text().strip()
    for name,expected in data['source_sha256'].items():assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected
    count=0
    for row in data['records']:
        inp=ROOT/f"results/twod_inputs/n{row['n']}.json";record=json.loads(inp.read_text());nodes=np.array(record['node_numerators'])
        assert hashlib.sha256(inp.read_bytes()).hexdigest()==row['input_sha256']
        assert np.array_equal(solve(row['n']),nodes)
        assert fit(nodes)==record['polynomial']
        assert np.array_equal(nodes,nodes.T) and np.array_equal(nodes,nodes[np.mod(-np.arange(row['n']),row['n'])])
        independent(record,row['selected_exponent'])
        assert sorted(c['exponent'] for c in row['candidates'])==[3,4,5,6,7]
        for candidate in row['candidates']:
            with contextlib.redirect_stdout(io.StringIO()):replayed=certify(record,candidate['exponent'])
            assert stable(replayed)==stable(candidate),(row['n'],candidate['exponent'])
            count+=1;print('Replayed',row['n'],candidate['exponent'],flush=True)
        best=min(row['candidates'],key=lambda c:F(c['error_upper']['numerator'],c['error_upper']['denominator']))
        assert best['exponent']==row['selected_exponent'] and best['error_upper']==row['selected_error_upper']
    with contextlib.redirect_stdout(io.StringIO()):audit_reference()
    audit=json.loads((ROOT/'results/twod_reference_audit.json').read_text())
    assert audit['selection_sha256']==sha
    for row in audit['records']:
        upper=next(r['selected_error_upper'] for r in data['records'] if r['n']==row['n'])
        assert F(row['uncorrected']['bound_lower']['numerator'],row['uncorrected']['bound_lower']['denominator'])>F(upper['numerator'],upper['denominator'])
    assert audit['records'][0]['effectivity_upper']['decimal']<2.044
    assert audit['records'][1]['effectivity_upper']['decimal']<1.889
    report=dict(status='PASS',selection_sha256=sha,certificates_replayed=count,
                numerical_inputs_recomputed=2,reference_intervals_replayed=2,
                independent_precision_digits=70,independent_jet_points=4,
                independent_bilinear_points=6,uncorrected_comparisons_replayed=2)
    (ROOT/'verification').mkdir(exist_ok=True)
    (ROOT/'verification/twod_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
