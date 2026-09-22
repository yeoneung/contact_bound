"""Replay reference-free selection and check it against independent arithmetic."""
from fractions import Fraction
from pathlib import Path
import ast,hashlib,json
import mpmath as mp
from flint import arb,ctx
from numerical_input import construct,Spline
from select_numerical_scale import certify,contact_geometry,Inadmissible,evaluate_cell
from audit_numerical_reference import error_enclosure

ROOT=Path(__file__).resolve().parents[1]


def mpr(q):
    q=Fraction(q)
    return mp.mpf(q.numerator)/q.denominator


def independent_residual(spline,x,eps):
    n=spline.n
    def value(z):
        i=min(n-1,max(0,int(z*n)));t=z*n-i
        a,b,c,d=map(mpr,spline.coeffs[i])
        return a+b*t+c*t*t+d*t*t*t
    xm=mpr(x);em=mpr(eps)
    u=value(xm);du=mp.diff(value,xm);hu=mp.diff(value,xm,2)
    y=xm+em*du;w=u+em*du**2;dw=du*(1+2*em*hu)/(1+em*hu)
    g=2+mp.cos(mp.pi*y)+mp.cos(2*mp.pi*y)/5
    return w+abs(dw)-g,w-value(y)


def main():
    mp.mp.dps=90;ctx.prec=192
    path=ROOT/'results/numerical_selection.json';frozen=hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen==(ROOT/'results/numerical_selection.sha256').read_text().strip()
    selection=json.loads(path.read_text());audit=json.loads((ROOT/'results/numerical_reference_audit.json').read_text())
    assert audit['selection_sha256']==frozen
    for name,expected in selection['selection_source_sha256'].items():
        src=ROOT/'code'/name
        assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
        tree=ast.parse(src.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ['audit_numerical_reference','check_numerical_selection']
    count=0;rejected=0;identities=0
    for row,refrow in zip(selection['records'],audit['records']):
        n=row['n'];record=json.loads((ROOT/f'results/numerical_inputs/n{n}.json').read_text())
        replay_input=construct(n)
        assert replay_input['node_numerators']==record['node_numerators']
        assert replay_input['input_sha256']==record['input_sha256']==row['input_sha256']
        spline=Spline(record)
        assert spline.discrete_residual()==record['discrete_residual_upper']
        for i in range(n):assert spline.local_value(i,Fraction(1))==spline.values[i+1]
        for c in row['candidates']:
            if c['status']=='certified':
                replay=certify(spline,c['exponent'])
                for name in ['geometry','residual_upper','distance_upper','error_upper','leaves','evaluations','maximum_depth']:
                    assert replay[name]==c[name],(n,c['exponent'],name)
                count+=1
            else:
                try:contact_geometry(spline,Fraction(1,1<<c['exponent']))
                except Inadmissible:rejected+=1
                else:raise AssertionError('A rejected scale became admissible.')
        certified=[c for c in row['candidates'] if c['status']=='certified']
        best=min(certified,key=lambda c:Fraction(c['error_upper']['numerator'],c['error_upper']['denominator']))
        assert best['exponent']==row['selected_exponent']
        assert best['error_upper']==row['selected_error_upper']
        root_cell=best['geometry']['root_cell'];eps=Fraction(1,1<<best['exponent'])
        for i in sorted(set([root_cell+1,max(root_cell+1,n//3),max(root_cell+1,2*n//3),n-1])):
            x=Fraction(2*i+1,2*n);radius=Fraction(1,10**13*n)
            enclosed=evaluate_cell(spline,eps,x-radius,x+radius,i)
            values=independent_residual(spline,x,eps)
            for a,b in zip(enclosed,values):
                assert a.contains(arb(mp.nstr(b,85),'1e-78')),(n,i)
                identities+=1
        assert error_enclosure(spline)==refrow['reference_error']
        print(n,'all candidates and post-selection error replayed',flush=True)
    # Separate integral representation validates the reference formula numerically.
    def g(x):return 2+mp.cos(mp.pi*x)+mp.cos(2*mp.pi*x)/5
    def part(x):
        return 2+sum(a*(mp.cos(w*x)-w*mp.sin(w*x))/(1+w*w)
                     for w,a in [(mp.pi,mp.mpf(1)),(2*mp.pi,mp.mpf(1)/5)])
    for x in map(mp.mpf,['0','.13','.57','1']):
        closed=part(x)+(g(1)-part(1))*mp.exp(x-1)
        integral=mp.exp(x-1)*g(1)+mp.quad(lambda s:mp.exp(x-s)*g(s),[x,1])
        assert abs(closed-integral)<mp.mpf('1e-80')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen
    report=dict(status='PASS',inputs_reproduced=5,certificates_replayed=count,
                inadmissible_scales_replayed=rejected,independent_residual_identities=identities,
                reference_error_enclosures_replayed=5,independent_reference_integrals=4,
                selection_sha256=frozen,reference_not_imported_by_selection=True)
    (ROOT/'verification/numerical_selection_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
