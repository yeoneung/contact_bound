"""Construct and certify one residual-refined two-dimensional approximation."""
from pathlib import Path
import argparse,json,time
from flint import ctx
from unknown_junction_2d import CASES,Input,fit
from residual_refinement_design import HELDOUT,install
from residual_refinement_2d import refine,certify_priority

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=[*CASES,*HELDOUT],default='anisotropic_l2')
    p.add_argument('--n',type=int,default=33)
    p.add_argument('--initialization',choices=['polynomial','corrected'],default='polynomial')
    p.add_argument('--degree',type=int,choices=[2,3,4,5],default=4)
    p.add_argument('--scale',type=float,choices=[.5,1.,2.],default=1.)
    p.add_argument('--output',default='research/residual_refinement_single.json')
    args=p.parse_args()
    if args.n<9:p.error('--n must be at least 9')
    ctx.prec=192;install();start=time.perf_counter()
    inp=Input(args.n,args.case);branches,seed=fit(inp,args.degree,args.initialization,args.scale)
    branches,model=refine(inp,branches,args.degree)
    certificate=certify_priority(inp,branches,budget=16000)
    result=dict(input=inp.record,seed=seed,model=model,certificate=certificate,total_seconds=time.perf_counter()-start,
                scope='One prescribed candidate. Its timing is not the full candidate-search benchmark.')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(file=str(path),input_error_upper=certificate['input_error_upper']['decimal'],
                         reconstruction_error_upper=certificate['reconstruction_error_upper']['decimal'],
                         total_seconds=result['total_seconds']),indent=2))

if __name__=='__main__':main()
