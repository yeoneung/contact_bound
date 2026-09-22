"""Development pilot, not a blinded or prospective benchmark."""
import argparse,json,time
from pathlib import Path
from free_branch_problem import Problem,Input
from free_branch_fit import fit,refine
from free_branch_verify import certify

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='pentagon');p.add_argument('--n',type=int,default=33)
    p.add_argument('--degrees',type=int,nargs='+',default=[2,3]);p.add_argument('--scales',type=float,nargs='+',default=[.5,1,2])
    p.add_argument('--budget',type=int,default=8000);p.add_argument('--verify',action='store_true');a=p.parse_args()
    start=time.perf_counter();inp=Input(Problem(a.case),a.n)
    out=dict(status='development',case=a.case,n=a.n,input=inp.record,records=[])
    print('input',a.case,a.n,inp.record['solver'],flush=True)
    for degree in a.degrees:
      for scale in a.scales:
       for kind in ('polynomial','corrected'):
        record=dict(degree=degree,scale=scale,kind=kind)
        try:
            cs,branches,model=fit(inp,degree,scale,kind);record['seed']=model
            branches,model=refine(inp,cs,degree);record['model']=model
            if a.verify:record['certificate']=certify(inp,branches,budget=a.budget)
            print(degree,scale,kind,'K',len(branches),'score',model['sampled_score'],
                  'bound',record.get('certificate',{}).get('input_error_upper',{}).get('decimal'),flush=True)
        except (ArithmeticError,ValueError) as e:
            record['failure']=str(e);print(degree,scale,kind,'FAIL',str(e),flush=True)
        out['records'].append(record);out['seconds']=time.perf_counter()-start
        (ROOT/'research'/f'free_branch_{a.case}_{a.n}.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
