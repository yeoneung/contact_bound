"""Fixed development comparison with automatic component/branch counts."""
from pathlib import Path
import json,hashlib,time
from flint import ctx
from free_branch_problem import Problem,Input,DEFINITIONS
from free_branch_fit import fit,refine
from free_branch_verify import certify

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('free_branch_problem.py','free_branch_fit.py','free_branch_verify.py','free_branch_bernstein.py',
         'benchmark_free_branches.py','interval_utils.py')
def main():
    ctx.prec=128;path=ROOT/'results/free_branches.json'
    design=dict(cases={'pentagon':33,'heptagon':65,'coupled_3d':17},degrees=[2,3],scales=[1,2,4],
        kinds=['polynomial','corrected'],iterations=24,ridge=1e-6,precision=128,
        box_budget={'2':12000,'3':20000},boundary_budget=2048,definitions=DEFINITIONS,
        selection='Refine and certify every successful degree-scale candidate. Select the smallest rigorous input-error upper bound separately for each initialization.',
        scope='Development experiments: geometry, fitting regularization, refinement budget and verifier were developed on these cases. Not a blinded generalization or timing benchmark.',
        branch_discovery='Connected components of cell centers passing the Hessian screen. No branch count, boundary-face association, analytic solution or junction coordinates supplied.')
    dp=path.with_name('free_branches_design.json')
    if dp.exists():assert json.loads(dp.read_text())==json.loads(json.dumps(design))
    dp.write_text(json.dumps(design,indent=2)+'\n')
    report=dict(status='running',design=design,source_sha256={n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest() for n in SOURCES},records=[])
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for case,n in design['cases'].items():
        start=time.perf_counter();inp=Input(Problem(case),n);generation=time.perf_counter()-start
        for kind in design['kinds']:
            row=dict(case=case,n=n,kind=kind,input=inp.record,input_seconds=generation,attempts=[])
            start=time.perf_counter()
            for degree in design['degrees']:
              for scale in design['scales']:
                entry=dict(degree=degree,scale=scale);t=time.perf_counter()
                try:
                    cs,branches,seed=fit(inp,degree,scale,kind);entry['seed']=seed
                    branches,model=refine(inp,cs,degree,iterations=design['iterations']);entry['model']=model
                    entry['certificate']=certify(inp,branches,budget=design['box_budget'][str(inp.d)],boundary_budget=design['boundary_budget'])
                    entry['status']='certified'
                except (ArithmeticError,ValueError) as e:entry.update(status='failed_proposal',reason=str(e))
                entry['seconds']=time.perf_counter()-t;row['attempts'].append(entry)
                print(case,n,kind,degree,scale,entry['status'],entry.get('model',{}).get('retained_branches'),
                      entry.get('certificate',{}).get('input_error_upper',{}).get('decimal'),flush=True)
            success=[(i,a) for i,a in enumerate(row['attempts']) if a['status']=='certified']
            row['selected_index']=min(success,key=lambda z:z[1]['certificate']['input_error_upper']['decimal'])[0] if success else None
            row['search_seconds']=time.perf_counter()-start;report['records'].append(row);save()
    report['status']='completed';save();path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
if __name__=='__main__':main()
