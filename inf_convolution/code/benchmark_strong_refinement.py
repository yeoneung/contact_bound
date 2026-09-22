"""Frozen complete-search comparison on the existing seven coefficient cases."""
from pathlib import Path
import argparse,hashlib,json,time
from flint import ctx
from residual_refinement_design import install,HELDOUT
from unknown_junction_2d import CASES
from strong_refinement_search import search

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('unknown_junction_2d.py','interval_utils.py','residual_refinement_2d.py','residual_refinement_design.py',
         'strong_refinement_search.py','benchmark_strong_refinement.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--grids',nargs='+',type=int,default=[33,65]);p.add_argument('--output',default='results/strong_refinement.json')
    args=p.parse_args();ctx.prec=192;install();path=ROOT/args.output
    design=dict(cases=[*CASES,*HELDOUT],grids=args.grids,repetitions=args.repetitions,kinds=['polynomial','corrected'],
                degrees=[2,3,4,5],scales=[.5,1,2],starts=2,precision=192,box_budget=16000,
                ranking='All 24 successful degree-scale-start candidates are refined before ranking.',
                search='Complete finite search; exclusion only by a rigorous lower bound exceeding the incumbent upper certificate.',
                timing='Sequential alternating method order. Input generation, every fit and refinement, all lower checks and interval verification included.',
                scope='Existing development cases. Fixed-input comparisons and first target success at that input; not a new adaptive-grid timing study.')
    dp=path.with_name(path.stem+'_design.json')
    if dp.exists() and json.loads(dp.read_text())!=json.loads(json.dumps(design)):raise RuntimeError('Design changed')
    dp.write_text(json.dumps(design,indent=2)+'\n')
    report=dict(status='running',design=design,source_sha256={n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest() for n in SOURCES},records=[])
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for rep in range(args.repetitions):
        for case in design['cases']:
            for n in args.grids:
                for kind in (design['kinds'] if rep%2==0 else design['kinds'][::-1]):
                    row=search(n,case,kind);row['repetition']=rep;report['records'].append(row);save()
                    print(rep,case,n,kind,row['selected']['certificate']['input_error_upper']['decimal'],round(row['total_seconds'],2),flush=True)
    report['status']='completed';save();path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
