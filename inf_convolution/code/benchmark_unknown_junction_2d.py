"""Complete proposal lists, then identical three-candidate certification budgets."""
from pathlib import Path
import argparse,hashlib,json,time,platform
import numpy as np
import scipy
from flint import ctx
from unknown_junction_2d import Input,fit,certify,sampled_score,CASES

ROOT=Path(__file__).resolve().parents[1]
METHODS=('polynomial','uncorrected','corrected')
SOURCES=('unknown_junction_2d.py','benchmark_unknown_junction_2d.py','interval_utils.py')

def search(n,case,kind):
    start=time.perf_counter();inp=Input(n,case);construction=time.perf_counter()-start;candidates=[];failures=[]
    for degree in (2,3,4,5):
        for scale in (.5,1.,2.):
            try:
                branches,model=fit(inp,degree,kind,scale);score=sampled_score(inp,branches)
                candidates.append((score,branches,model))
            except (ArithmeticError,ValueError) as e:failures.append(dict(degree=degree,scale=scale,reason=str(e)))
    candidates.sort(key=lambda z:z[0]);proposal_seconds=time.perf_counter()-start-construction
    attempts=[];best=None
    for score,branches,model in candidates[:3]:
        cert=certify(inp,branches,budget=16000)
        row=dict(sampled_score=score,model=model,certificate=cert);attempts.append(row)
        if cert['status']=='certified' and (best is None or cert['input_error_upper']['decimal']<best['certificate']['input_error_upper']['decimal']):best=row
    return dict(status='certified' if best is not None else 'not_certified',input=inp.record,selected=best,
        proposals=[dict(score=s,degree=m['degree'],scale=m['scale'],fit_seconds=m['fit_seconds']) for s,b,m in candidates],
        failed_proposals=failures,attempts=attempts,construction_seconds=construction,proposal_seconds=proposal_seconds,total_seconds=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=3);p.add_argument('--output',default='results/unknown_junction_2d.json');args=p.parse_args();ctx.prec=192
    path=ROOT/args.output;design=dict(cases=CASES,grids=[33,65],methods=METHODS,degrees=[2,3,4,5],scales=[.5,1,2],
        repetitions=args.repetitions,precision=192,verified_candidates=3,box_budget=16000,max_depth=12,
        interface_information='No exact solution, interface coordinates or reflection axes supplied. Four boundary-associated branches; their meeting curves are determined by their fitted values.',
        guarantee='Continuous PDE error of the exact C1 numerical input: distance + max(branch residual, boundary discrepancy). Every remaining box contributes a rigorous upper enclosure.',
        tolerance='5 percent plus 1e-6 is a subdivision stopping guide. Budget/depth-limited boxes retain their possibly wider rigorous upper bounds; this is not a guaranteed 5 percent supremum approximation.',
        comparison='All methods have the same polynomial degrees, retention scales, extrapolation constraints, sample ranking and three-candidate verification budget.',
        development='Three development problems; design frozen after pilot fits. Not a blinded generalization study. Degree seven was unstable in pilot extrapolation and is not in the final candidate list.',
        timing='Sequential cyclic method order; includes input solve, every proposal fit and all three interval verifications.')
    dp=path.with_name(path.stem+'_design.json')
    if dp.exists() and json.loads(dp.read_text())!=json.loads(json.dumps(design)):raise RuntimeError('Frozen design mismatch.')
    dp.write_text(json.dumps(design,indent=2)+'\n')
    report=dict(status='running',design=design,source_sha256={f:hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest() for f in SOURCES},
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,processor='Intel Core i9-14900KF'),records=[])
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for rep in range(args.repetitions):
        order=METHODS[rep%3:]+METHODS[:rep%3]
        for case in CASES:
            for n in [33,65]:
                for kind in order:
                    r=search(n,case,kind);report['records'].append(dict(case=case,n=n,method=kind,repetition=rep,**r));save()
                    value=None if r['selected'] is None else r['selected']['certificate']['input_error_upper']['decimal']
                    print(rep,case,n,kind,value,round(r['total_seconds'],3),flush=True)
    report['status']='completed';save();path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
