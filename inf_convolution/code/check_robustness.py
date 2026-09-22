"""Check search inventories, rigorous pruning, selections and independent formulas."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import argparse,hashlib,json
import mpmath as mp
import numpy as np
from flint import ctx
from asymmetric_input import construct,PeriodicSpline,solve,cost
from robustness_design import (install_cases,specification,ORIGINAL_CASES,NEW_CASES,
    GRIDS,POLICY_GRIDS,TARGETS,BUDGETS)
from compare_robustness import search,proposals,earliest,rational,key
from certify_automatic_branches import certify,certify_polynomial
from check_automatic_branches import stable,independent_jet,independent_global_minimum

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-replay',action='store_true');args=parser.parse_args()
    ctx.prec=192;mp.mp.dps=75;install_cases();path=ROOT/'results/robustness.json';data=json.loads(path.read_text());sha=digest(path)
    assert data['status']=='completed' and sha==path.with_suffix('.sha256').read_text().strip()
    assert json.loads((ROOT/'results/robustness_design.json').read_text())==json.loads(json.dumps(specification()))
    assert data['design_sha256']==digest(ROOT/'results/robustness_design.json')
    for name,value in data['source_sha256'].items():assert digest(ROOT/'code'/name)==value,name
    rows={key(r['case'],r['n'],r['method'],r['scope']):r for r in data['records']};assert len(rows)==len(data['records'])
    for case in (*ORIGINAL_CASES,*NEW_CASES):
        for n in GRIDS:
            for method in ['polynomial','uncorrected','corrected']:
                scope='wide' if case in ORIGINAL_CASES and method!='polynomial' else 'standard'
                assert key(case,n,method,scope) in rows
    count=Counter();replayed=0
    for row in data['records']:
        s=PeriodicSpline(row['input']);ranked=proposals(s,row['method'],row['scope']);assert len(ranked)==len(row['candidates'])
        incumbent=None
        for rank,(p,c) in enumerate(zip(ranked,row['candidates']),1):
            assert p==c['proposal'] and c['rank']==rank
            assert c['incumbent_cutoff']==(None if incumbent is None else [incumbent.numerator,incumbent.denominator])
            count[c['status']]+=1
            if c['status']=='above_target':assert incumbent is not None and rational(c['error_lower'])>incumbent
            if c['status']=='certified':
                B=rational(c['error_upper']);assert B==rational(c['residual_upper'])+rational(c['distance_upper'])
                for name in ['residual','distance']:
                    lo=rational(c[name+'_lower']);hi=rational(c[name+'_upper'])
                    assert 0<=lo<=hi<=F(51,50)*lo+F(1,10**10)+F(1,2**95)
                if incumbent is None or B<incumbent:incumbent=B
        certificates=[c for c in row['candidates'] if c['status']=='certified']
        best=min(certificates,key=lambda c:rational(c['error_upper'])) if certificates else None
        assert row['selected']==best
        if args.full_replay:
            r=search(row['case'],row['n'],row['method'],row['scope']);assert stable(r)==stable(row),(row['case'],row['n'],row['method'])
            replayed+=len(row['candidates'])
        elif best is not None:
            p=best['proposal'];cutoff=None if best['incumbent_cutoff'] is None else F(*best['incumbent_cutoff'])
            if row['method']=='polynomial':r=certify_polynomial(s,p['width'],p['degree'],target=cutoff)
            else:r=certify(s,None,row['method'],eps=F(*p['epsilon']),target=cutoff)
            expected={k:v for k,v in best.items() if k not in ['proposal','rank','incumbent_cutoff']}
            assert stable(r)==stable(expected);replayed+=1
        print('Replayed',row['case'],row['n'],row['method'],row['scope'],flush=True)
    assert len(data['policy'])==len(ORIGINAL_CASES)*3*len(TARGETS)*len(BUDGETS)
    for p in data['policy']:
        found=None
        for n in POLICY_GRIDS:
            k=key(p['case'],n,p['method'],'standard')
            if k not in rows:break
            rank=earliest(rows[k],F(*p['target']),p['budget'])
            if rank is not None:found=(n,rank);break
        assert found==(p['n'],p['rank']) if found is not None else p['n'] is None
    points=global_points=0
    for case in NEW_CASES:
        n=64;g=cost(2*np.arange(n)/n,case);u=g.copy();h=2/n
        for _ in range(4*n):u=np.minimum(g,np.minimum((h*g+np.roll(u,1))/(1+h),(h*g+np.roll(u,-1))/(1+h)))
        assert np.max(abs(u-solve(n,case)))<1e-13
        row=rows[key(case,256,'uncorrected','standard')];s=PeriodicSpline(row['input']);eps=F(*row['selected']['epsilon'])
        points+=independent_jet(s,eps);global_points+=independent_global_minimum(s,eps)
    report=dict(status='PASS',comparison_sha256=sha,design_sha256=data['design_sha256'],
        searches=len(rows),candidate_statuses=dict(count),replay_mode='all_candidates' if args.full_replay else 'selected_certificates',
        certificates_or_attempts_replayed=replayed,policy_choices_checked=len(data['policy']),
        independent_jet_points=points,independent_global_contact_points=global_points,independent_node_solvers=len(NEW_CASES))
    name='robustness_full_replay.json' if args.full_replay else 'robustness_checks.json'
    (ROOT/'verification'/name).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
