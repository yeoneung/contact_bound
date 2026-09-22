"""Fixed-input and target-cost comparisons with an equally adaptive control."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time,platform
import numpy as np
from flint import ctx
from asymmetric_input import construct,PeriodicSpline
from robustness_design import install_cases,NEW_CASES
from adaptive_local import reconstruct,frac
from benchmark_automatic_branches import rank_candidates
from certify_automatic_branches import certify_polynomial

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('adaptive_local.py','benchmark_adaptive_local.py','asymmetric_input.py','automatic_branches.py',
         'certify_automatic_branches.py','direct_branch_polynomials.py','benchmark_automatic_branches.py','interval_utils.py','robustness_design.py')
METHODS=('global_polynomial','adaptive_polynomial','adaptive_hybrid')

def evaluate(s,method,target=None):
    if method!='global_polynomial':return reconstruct(s,method=='adaptive_hybrid',target=target)
    started=time.perf_counter();ranking=rank_candidates(s,'polynomial');best=None;attempts=[]
    for p in ranking[:3]:
        r=certify_polynomial(s,p['width'],p['degree'],target=target);attempts.append(dict(candidate=p,certificate=r))
        if r['status']=='certified' and (best is None or frac(r['error_upper'])<frac(best['error_upper'])):best=r
        if best is not None and target is not None and frac(best['error_upper'])<=target:break
    if best is None:return dict(status='not_certified',attempts=attempts,total_seconds=time.perf_counter()-started)
    return dict(**{k:v for k,v in best.items() if k!='total_seconds'},attempts=attempts,total_seconds=time.perf_counter()-started)

def run_target(case,method,target,max_n):
    started=time.perf_counter();history=[]
    for n in (64,128,256,512,1024,2048,4096,8192):
        if n>max_n:break
        t=time.perf_counter();record=construct(n,case);s=PeriodicSpline(record);construction=time.perf_counter()-t
        r=evaluate(s,method,target);history.append(dict(n=n,construction_seconds=construction,result=r))
        if r['status']=='certified' and frac(r['error_upper'])<=target:
            return dict(status='target_certified',selected_n=n,selected=r,input=record,history=history,total_seconds=time.perf_counter()-started)
    return dict(status='budget_exhausted',history=history,total_seconds=time.perf_counter()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=3);p.add_argument('--max-n',type=int,default=8192)
    p.add_argument('--output',default='results/adaptive_local.json');args=p.parse_args();install_cases();ctx.prec=192
    path=ROOT/args.output;design=dict(cases=list(NEW_CASES),fixed_grids=[256,1024],targets=['.04','.02','.01'],
        methods=METHODS,repetitions=args.repetitions,max_n=args.max_n,precision=192,
        timing='Sequential cyclic method order. Target time includes input solves, proposals, failed attempts and preceding grids.',
        development='Design fixed after exploratory skew and two-dimensional pilot runs; these are development benchmarks, not blinded validation.',
        input_refinement='Uniform input-grid doubling; local adaptive reconstruction and interval subdivision. No claim of nonuniform input-mesh refinement.',
        fallback='If all three initial hybrid candidates fail on a patch, try the polynomial control candidates not yet checked, up to its first three. All fallback work is timed and recorded.',
        fairness='The polynomial control has the same local patch selection and interval cache as the hybrid. Global polynomial uses its existing 30-candidate ranking, checks at most three candidates, rejects by rigorous target lower bounds, and stops immediately once the target is certified. Local methods also reject regions by rigorous target lower bounds.')
    design_path=path.with_name(path.stem+'_design.json')
    if design_path.exists() and json.loads(design_path.read_text())!=json.loads(json.dumps(design)):raise RuntimeError('Frozen design mismatch.')
    design_path.write_text(json.dumps(design,indent=2)+'\n')
    report=dict(status='running',design=design,source_sha256={f:hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest() for f in SOURCES},
                environment=dict(python=platform.python_version(),numpy=np.__version__,processor='Intel Core i9-14900KF'),fixed=[],targets=[])
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for rep in range(args.repetitions):
        order=METHODS[rep%3:]+METHODS[:rep%3]
        for case in NEW_CASES:
            for n in [256,1024]:
                for method in order:
                    t=time.perf_counter();record=construct(n,case);s=PeriodicSpline(record);r=evaluate(s,method)
                    report['fixed'].append(dict(case=case,n=n,method=method,repetition=rep,input=record,result=r,total_seconds=time.perf_counter()-t))
                save()
            for target in [F(1,25),F(1,50),F(1,100)]:
                for method in order:
                    r=run_target(case,method,target,args.max_n)
                    report['targets'].append(dict(case=case,target=str(target),method=method,repetition=rep,**r))
                save()
            print(rep,case,'finished',flush=True)
    report['status']='completed';save();path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
