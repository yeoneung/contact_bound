"""Recompute the settings determining the reported improved bounds from caches."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/experiments'))
from hjrl.verified_arb import SCALE
from improve_contact_estimator import envelope_bound
from refine_contact_stationarity import filtered_bound

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b):
    assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}

def main():
    ctx.prec=128
    shifted=json.loads((ROOT/'results/improved_contact_estimator.json').read_text())
    filtered=json.loads((ROOT/'results/stationary_contact_estimator.json').read_text())
    for data in (shifted,filtered):
        assert data['complete']
        for path,sha in data['hashes'].items():assert digest(ROOT/path)==sha,path
    runs=[]
    for row in shifted['records']:
        name=row['critic'];fine=[r for r in row['settings'] if r['grid_nodes']==65536]
        chosen={min(fine,key=lambda r:r['restricted_one_sided_bounds'][s]['upper'])['reciprocal_scale']
                for s in ('-1','1')}
        vp=ROOT/f'results/comparison_enclosures_{name}.npz'
        vm=json.loads(vp.with_suffix('.json').read_text())
        assert digest(vp)==vm['enclosures_sha256']
        assert digest(ROOT/'results/verified_neural_models.pt')==vm['weights_sha256']
        assert digest(ROOT/'code/hjrl/verified_arb.py')==vm['arithmetic_source_sha256']
        values=np.load(vp,allow_pickle=False)
        ulo,uhi=values['lower'],values['upper'];lu=arb(vm['L_u_numerator'])/SCALE
        for m in sorted(chosen):
            before=next(r for r in fine if r['reciprocal_scale']==m)
            same(before,envelope_bound(ulo,uhi,lu,m,16384))
            runs.append({'critic':name,'method':'shifted','reciprocal_scale':m})
        dp=ROOT/f'results/contact_derivatives_{name}.npz'
        dm=json.loads(dp.with_suffix('.json').read_text())
        assert dm['hashes']==filtered['hashes'] and digest(dp)==dm['sha256']
        derivatives=np.load(dp,allow_pickle=False)
        fr=next(r for r in filtered['records'] if r['critic']==name)
        chosen={min(fr['settings'],key=lambda r:r['one_sided_bounds'][s]['upper'])['reciprocal_scale']
                for s in ('-1','1')}
        for m in sorted(chosen):
            before=next(r for r in fr['settings'] if r['reciprocal_scale']==m)
            same(before,filtered_bound(ulo,uhi,derivatives['lower'],derivatives['upper'],lu,m,16384))
            runs.append({'critic':name,'method':'filtered','reciprocal_scale':m})
    report={'successful':True,'settings_recomputed':runs,'all_non_timing_fields_match':True,
            'cached_verified_enclosures_used':True,'full_derivative_verification_repeated':False,
            'entire_scale_sweep_repeated':False}
    (ROOT/'checks/improved_subset_reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
