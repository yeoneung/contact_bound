"""Independently reproduce two saved 2D configurations on the CPU."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from flint import ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from hjrl.verified_twod import sparse_contact_bound

ctx.prec=128
data=json.loads((ROOT/'results/twod_contact.json').read_text())
assert data['complete']
records=[]
for row in data['records']:
    path=ROOT/f"results/twod_critic_{row['critic_side']}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['coefficients_sha256']
    u=np.load(path)['nodes']
    saved=next(s for s in row['settings'] if s['location_side']==1024 and s['reciprocal_scale']==64)
    replay=sparse_contact_bound(u,1024,512,64,'numpy')
    for key in ['filtered_residual_numerators','filtered_candidate_counts','complete_bound','coverage_correction']:
        assert replay[key]==saved[key],(row['critic_side'],key)
    records.append(dict(critic_side=row['critic_side'],location_side=1024,center_side=512,reciprocal_scale=64,
                        bound=replay['complete_bound']['upper'],cpu_seconds=replay['verification_seconds']))
report=dict(successful=True,backend='numpy CPU',settings_recomputed=records)
(ROOT/'checks/twod_subset_reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
