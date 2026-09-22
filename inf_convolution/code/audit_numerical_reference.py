"""Evaluate the reference error only AFTER the selection file is frozen.

Nothing in the constructor or selector imports this module. The exact value
formula is used only here, for a post-selection assessment of effectivity.
"""
from fractions import Fraction
from pathlib import Path
import hashlib,json,time
from flint import arb,ctx
from interval_utils import ball,interval,upper_record,lower_record
from numerical_input import Spline,running_cost_arb

ROOT=Path(__file__).resolve().parents[1]


def particular(x):
    out=arb(2)
    for j,amplitude in [(1,arb(1)),(2,arb(1)/5)]:
        omega=j*arb.pi();s,c=(omega*x).sin_cos()
        out+=amplitude*(c-omega*s)/(1+omega*omega)
    return out


def reference_value(x):
    return particular(x)+(arb(6)/5-particular(arb(1)))*(x-1).exp()


def reference_derivative(x):return reference_value(x)-running_cost_arb(x)


def error_enclosure(spline):
    lower=arb(0);upper=arb(0);cells=0
    pending=[(Fraction(i,spline.n),Fraction(i+1,spline.n),i,0) for i in range(spline.n)]
    for i in range(spline.n+1):
        x=Fraction(i,spline.n)
        lower=lower.max(abs(ball(spline.point(x))-reference_value(ball(x))).abs_lower())
    rtol=arb(1)/1000;atol=arb(1)/10**14
    while pending:
        left,right,i,depth=pending.pop();mid=(left+right)/2
        value=spline.local_arb(i,ball(mid))[0]-reference_value(ball(mid))
        lower=lower.max(abs(value).abs_lower())
        x=interval(left,right)
        derivative=spline.local_arb(i,x)[1]-reference_derivative(x)
        enclosure=abs(value).upper()+ball((right-left)/2)*abs(derivative).upper()
        if enclosure<=lower*(1+rtol)+atol:
            upper=upper.max(enclosure);cells+=1
        else:
            if depth>30:raise ArithmeticError('Reference audit did not resolve the supremum.')
            pending.append((left,mid,i,depth+1));pending.append((mid,right,i,depth+1))
    return dict(lower=lower_record(lower),upper=upper_record(upper),cells=cells)


def main():
    ctx.prec=192
    path=ROOT/'results/numerical_selection.json'
    frozen=hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen==(ROOT/'results/numerical_selection.sha256').read_text().strip()
    selection=json.loads(path.read_text());rows=[]
    for name,expected in selection['selection_source_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected,name
    for row in selection['records']:
        started=time.perf_counter()
        record=json.loads((ROOT/f"results/numerical_inputs/n{row['n']}.json").read_text())
        assert record['input_sha256']==row['input_sha256']
        spline=Spline(record);error=error_enclosure(spline)
        b=row['selected_error_upper'];ub=ball(Fraction(b['numerator'],b['denominator']))
        lb=ball(Fraction(error['lower']['numerator'],error['lower']['denominator']))
        assert lb>0 and ub>ball(Fraction(error['upper']['numerator'],error['upper']['denominator']))
        baseline=abs(ball(spline.values[0])-running_cost_arb(arb(0)))
        rows.append(dict(n=row['n'],selected_exponent=row['selected_exponent'],
                         selected_error_upper=b,reference_error=error,
                         effectivity_upper=upper_record(ub/lb),
                         differential_residual_lower=lower_record(baseline),
                         reference_audit_seconds=time.perf_counter()-started))
        print(row['n'],'error in',error['lower']['decimal'],error['upper']['decimal'],
              'effectivity <=',rows[-1]['effectivity_upper']['decimal'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen
    report=dict(status='post_selection_reference_audit',selection_sha256=frozen,
                relative_error_enclosure_tolerance=[1,1000],records=rows)
    (ROOT/'results/numerical_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
