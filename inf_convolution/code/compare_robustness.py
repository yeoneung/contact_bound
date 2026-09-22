"""Full finite-list searches on identical inputs, with rigorous incumbent pruning."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time
import numpy as np
from flint import ctx
from asymmetric_input import construct,PeriodicSpline,cost
from automatic_branches import discover_float,float_envelope
from benchmark_automatic_branches import polynomial_proposal
from certify_automatic_branches import certify,certify_polynomial
from robustness_design import (install_cases,specification,ORIGINAL_CASES,NEW_CASES,
    STANDARD_SCALES,WIDE_SCALES,POLYNOMIAL_PARAMETERS,GRIDS,POLICY_GRIDS,TARGETS,BUDGETS)

ROOT=Path(__file__).resolve().parents[1]
SOURCE_NAMES=['robustness_design.py','compare_robustness.py','asymmetric_input.py',
    'automatic_branches.py','certify_automatic_branches.py','direct_branch_polynomials.py',
    'benchmark_automatic_branches.py','interval_utils.py']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rational(r):return F(r['numerator'],r['denominator'])
def key(case,n,method,scope):return f'{case}|{n}|{method}|{scope}'

def proposals(spline,method,scope):
    count=min(4*spline.n,32768);y=np.arange(count)*2/count
    u,du,_,_=spline.float_jet(y);g=cost(y,spline.case);rows=[]
    if method=='polynomial':
        for width,degree in POLYNOMIAL_PARAMETERS:
            z=polynomial_proposal(spline,y,u,du,width,degree)
            score=None if z is None else float(np.max(abs(z[0]-u))+np.max(abs(z[0]+abs(z[1])-g)))
            rows.append(dict(width=width,degree=degree,sampled_score=score))
    else:
        for ratio in WIDE_SCALES if scope=='wide' else STANDARD_SCALES:
            eps=ratio*spline.h;branches=discover_float(spline,float(eps))
            z=None if not branches else float_envelope(branches,y,method)
            score=None if z is None or not np.all(np.isfinite(z[0])) else float(np.max(abs(z[0]-u))+np.max(abs(z[0]+abs(z[1])-g)))
            rows.append(dict(ratio=[ratio.numerator,ratio.denominator],epsilon=[eps.numerator,eps.denominator],sampled_score=score))
    # Even failed floating-point proposals receive a rigorous attempt.
    return sorted(rows,key=lambda r:float('inf') if r['sampled_score'] is None else r['sampled_score'])

def search(case,n,method,scope):
    start=time.perf_counter();record=construct(n,case);spline=PeriodicSpline(record);constructed=time.perf_counter()
    ranked=proposals(spline,method,scope);ordered=time.perf_counter();best=None;rows=[]
    for rank,p in enumerate(ranked,1):
        cutoff=None if best is None else rational(best['error_upper']);t0=time.perf_counter()
        try:
            if method=='polynomial':r=certify_polynomial(spline,p['width'],p['degree'],target=cutoff)
            else:r=certify(spline,None,method,eps=F(*p['epsilon']),target=cutoff)
        except ArithmeticError as e:
            r=dict(status='not_certified',reason=str(e),total_seconds=time.perf_counter()-t0)
        r['proposal']=p;r['rank']=rank
        r['incumbent_cutoff']=None if cutoff is None else [cutoff.numerator,cutoff.denominator]
        rows.append(r)
        if r['status']=='certified' and (best is None or rational(r['error_upper'])<rational(best['error_upper'])):best=r
    return dict(case=case,n=n,method=method,scope=scope,input=record,
        status='certified' if best is not None else 'not_certified',selected=best,candidates=rows,
        construction_seconds=constructed-start,proposal_seconds=ordered-constructed,total_seconds=time.perf_counter()-start)

def earliest(row,target,budget):
    # Before the first certificate below target, pruning at an incumbent above
    # target cannot discard a successful candidate. Later candidates are irrelevant.
    for c in row['candidates'][:budget]:
        if c['status']=='certified' and rational(c['error_upper'])<=target:return c['rank']
        if c['status']=='above_target':
            cutoff=F(*c['incumbent_cutoff'])
            assert cutoff>target,('A successful earlier candidate must already have returned.',row['case'],row['n'])
    return None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['fixed','policy','all'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();ctx.prec=192;install_cases();path=ROOT/'results/robustness.json';design_path=ROOT/'results/robustness_design.json'
    spec=json.loads(json.dumps(specification()))
    if design_path.exists():assert json.loads(design_path.read_text())==spec
    else:design_path.write_text(json.dumps(spec,indent=2)+'\n')
    hashes={n:digest(ROOT/'code'/n) for n in SOURCE_NAMES}
    if a.resume and path.exists():
        report=json.loads(path.read_text());assert report['source_sha256']==hashes and report['design_sha256']==digest(design_path)
    else:report=dict(status='incomplete',design_sha256=digest(design_path),source_sha256=hashes,records=[],policy=[])
    cached={key(r['case'],r['n'],r['method'],r['scope']):r for r in report['records']}
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    def obtain(case,n,method,scope):
        k=key(case,n,method,scope)
        if k not in cached:
            r=search(case,n,method,scope);cached[k]=r;report['records'].append(r);save()
            best=r['selected'];print(case,n,method,scope,r['status'],None if best is None else best['error_upper']['decimal'],round(r['total_seconds'],3),flush=True)
        return cached[k]
    if a.stage in ['fixed','all']:
        for case in (*ORIGINAL_CASES,*NEW_CASES):
            for n in GRIDS:
                for method in ('polynomial','uncorrected','corrected'):
                    obtain(case,n,method,'wide' if case in ORIGINAL_CASES and method!='polynomial' else 'standard')
        report['fixed_complete']=True;save()
    if a.stage in ['policy','all']:
        policy=[]
        for case in ORIGINAL_CASES:
            for method in ('polynomial','uncorrected','corrected'):
                pending={(t,b) for t in TARGETS for b in BUDGETS}
                for n in POLICY_GRIDS:
                    row=obtain(case,n,method,'standard')
                    for target,budget in sorted(pending,reverse=True):
                        rank=earliest(row,target,budget)
                        if rank is not None:
                            policy.append(dict(case=case,method=method,target=[target.numerator,target.denominator],budget=budget,n=n,rank=rank))
                            pending.remove((target,budget))
                    if not pending:break
                for target,budget in sorted(pending):policy.append(dict(case=case,method=method,target=[target.numerator,target.denominator],budget=budget,n=None,rank=None))
                report['policy']=policy;save()
        report['policy_complete']=True
    report['status']='completed' if report.get('fixed_complete') and report.get('policy_complete') else 'partial'
    save();path.with_suffix('.sha256').write_text(digest(path)+'\n')
    print('Saved',len(report['records']),'full searches and',len(report['policy']),'policy choices.',flush=True)

if __name__=='__main__':main()
