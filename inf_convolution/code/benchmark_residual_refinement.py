"""Prospective common-target comparison including all solves and attempts."""
from pathlib import Path
import argparse,hashlib,json,time,platform
import numpy as np
import scipy
from flint import ctx
from unknown_junction_2d import Input,fit,sampled_score,CASES
from residual_refinement_2d import refine,certify_priority
from residual_refinement_design import HELDOUT,install

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('unknown_junction_2d.py','interval_utils.py','residual_refinement_2d.py',
         'residual_refinement_design.py','benchmark_residual_refinement.py')
METHODS=('polynomial','refined_polynomial','refined_corrected')
TARGETS=(.12,.08,.04)

def search(case,method):
    started=time.perf_counter();targets={};history=[]
    for n in (17,33,65):
        inp=Input(n,case);candidates=[];failures=[]
        kind='corrected' if method=='refined_corrected' else 'polynomial'
        for degree in (2,3,4,5):
            for scale in (.5,1.,2.):
                try:
                    branches,model=fit(inp,degree,kind,scale)
                    score=sampled_score(inp,branches)
                    if np.isfinite(score):candidates.append((score,branches,model))
                    else:failures.append(dict(degree=degree,scale=scale,reason='Nonfinite sampled score'))
                except (ArithmeticError,ValueError) as e:failures.append(dict(degree=degree,scale=scale,reason=str(e)))
        candidates.sort(key=lambda z:z[0]);attempts=[]
        grid=dict(n=n,input=inp.record,failed_proposals=failures,
                  proposals=[dict(score=s,degree=m['degree'],scale=m['scale']) for s,b,m in candidates],attempts=attempts)
        history.append(grid)
        for score,branches,seed in candidates[:3]:
            model=seed;failure=None
            if method!='polynomial':
                try:branches,model=refine(inp,branches,seed['degree'])
                except (ArithmeticError,ValueError) as e:failure=str(e)
            cert=certify_priority(inp,branches,budget=16000)
            elapsed=time.perf_counter()-started
            attempt=dict(seed=seed,model=model,refinement_failure=failure,certificate=cert,elapsed_seconds=elapsed)
            attempts.append(attempt)
            for target in TARGETS:
                if str(target) not in targets and cert['input_error_upper']['decimal']<=target:
                    targets[str(target)]=dict(status='target_certified',n=n,seconds=elapsed,
                                              history_index=len(history)-1,attempt_index=len(attempts)-1,
                                              bound=cert['input_error_upper']['decimal'])
            if len(targets)==len(TARGETS):break
        if len(targets)==len(TARGETS):break
    for target in TARGETS:
        if str(target) not in targets:targets[str(target)]=dict(status='not_reached',seconds=time.perf_counter()-started,max_n=65)
    return dict(case=case,method=method,targets=targets,history=history,total_seconds=time.perf_counter()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--output',default='results/residual_refinement.json');args=p.parse_args()
    ctx.prec=192;install();path=ROOT/args.output
    design=dict(development_cases=list(CASES),prospective_cases=HELDOUT,methods=METHODS,
        targets=TARGETS,grids=[17,33,65],degrees=[2,3,4,5],scales=[.5,1,2],verified_candidates=3,
        repetitions=args.repetitions,precision=192,box_budget=16000,max_depth=12,
        refinement=dict(iterations=8,training_n=21,validation_n=81,near_active_margin=.02,exchange_points=24),
        certificate='Input error D+max(R,boundary), with all branches and boxes rigorously checked; samples only propose coefficients.',
        timing='Sequential cyclic order. Every solve, fit, failed attempt, refinement and verification included. A target stops at its first verified success; tighter targets continue along exactly the same path.',
        comparison='The polynomial control receives the same priority verifier and proposal budget. Refined polynomial is an equally optimized control for contact initialization.',
        scope='Four prescribed boundary-associated polynomial branches in two dimensions. Four additional coefficient problems fixed prospectively after the three-case pilot. No claim of general HJB superiority.')
    dp=path.with_name(path.stem+'_design.json')
    if dp.exists() and json.loads(dp.read_text())!=json.loads(json.dumps(design)):raise RuntimeError('Frozen design mismatch')
    dp.write_text(json.dumps(design,indent=2)+'\n')
    report=dict(status='running',design=design,source_sha256={s:hashlib.sha256((ROOT/'code'/s).read_bytes()).hexdigest() for s in SOURCES},
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,processor='Intel Core i9-14900KF'),records=[])
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for rep in range(args.repetitions):
        order=METHODS[rep%3:]+METHODS[:rep%3]
        for case in [*CASES,*HELDOUT]:
            for method in order:
                row=search(case,method);row['repetition']=rep;report['records'].append(row);save()
                print(rep,case,method,{k:(v['status'],v.get('n'),round(v['seconds'],3)) for k,v in row['targets'].items()},flush=True)
    report['status']='completed';save();path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
