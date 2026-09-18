"""Recompute all six reported contact optima and both Bellman successor modes.

--full-operators also rebuilds the three associated flow/cost/hold enclosures.
The default verifies the saved operator arrays by hash and reuses them. Timing
fields are excluded from equality; every computed numerical field must match.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from flint import ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/experiments')]
from compare_variable_speed import digest,load_critic,contact_bound,build_operator,bellman_bound
from hjrl.verified_variable_speed import VariableSpeed


def equal_numeric(actual,expected):
    for k,v in actual.items():
        if 'seconds' not in k:assert v==expected[k],(k,v,expected[k])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full-operators',action='store_true');args=ap.parse_args()
    ctx.prec=128;data=json.loads((ROOT/'results/state_dependent_comparison.json').read_text())
    assert data['complete'] and len(data['records'])==156 and len(data['operators'])==30
    for path,h in data['hashes'].items():assert digest(ROOT/path)==h,path
    critics={name:load_critic(name) for name in ('MLP','Min2')}
    old=json.loads((ROOT/'results/estimator_comparison.json').read_text())
    errors={r['critic']:r['true_error_enclosure']['upper'] for r in old['records']}
    for r in data['records']:assert r['complete_bound']['upper']>=errors[r['critic']]
    operator_keys=set();recomputed=[]
    for beta in (0,1,2):
        for name,c in critics.items():
            pool=[r for r in data['records'] if r['beta_quarters']==beta and r['critic']==name]
            best=min((r for r in pool if r['method']=='contact'),key=lambda r:r['complete_bound']['upper'])
            new=contact_bound(VariableSpeed(beta),c,best['reciprocal_scale'],best['centers'])
            equal_numeric(new,best);recomputed.append(best['key'])
            best_b=min((r for r in pool if r['method']=='bellman'),key=lambda r:r['complete_bound']['upper'])
            operator_keys.add(best_b['operator_key'])
    rebuilt=[]
    for key in sorted(operator_keys):
        path=ROOT/'results/state_dependent_operators'/(key+'.npz')
        metadata=json.loads(path.with_suffix('.json').read_text())
        assert metadata['hashes']==data['hashes'] and digest(path)==metadata['sha256']
        arrays=dict(np.load(path,allow_pickle=False))
        p=VariableSpeed(metadata['beta_quarters'])
        if args.full_operators:
            actual,newmeta=build_operator(p,metadata['nodes'],metadata['reciprocal_hold'],metadata['panels'])
            for k in arrays:assert np.array_equal(actual[k],arrays[k]),(key,k)
            equal_numeric(newmeta,metadata);rebuilt.append(key)
        for row in data['records']:
            if row.get('operator_key')==key:
                new=bellman_bound(p,critics[row['critic']],arrays,metadata,row['successor_mode']=='derivative')
                equal_numeric(new,row);recomputed.append(row['key'])
    report={'successful':True,'all_non_timing_fields_match':True,'settings_recomputed':recomputed,
            'operator_enclosures_rebuilt':rebuilt,'cached_critic_enclosures_used':True,
            'global_neural_enclosures_rebuilt':False,'entire_parameter_sweep_repeated':False}
    out=ROOT/'checks/variable_speed_subset_reproduction.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
