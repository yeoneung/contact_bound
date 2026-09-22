"""Replay the preserved reference certificate only after scale selection is fixed."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
from flint import ctx,arb
from interval_utils import ball,upper_record,lower_record
from twod_polynomials import certify as certify_auxiliary,SCALE
from twod_reference_distance import bilinear_distance
from verify_twod_reconstruction import Reconstruction,problem_data

ROOT=Path(__file__).resolve().parents[1]

def uncorrected_lower(record,exponent,side=48):
    model=Reconstruction(record,exponent);rl=arb(0);dl=arb(0);points=0
    for i in range(side+1):
        for j in range(side+1):
            x=model.a+(1-model.a)*F(i,side);y=model.a+(1-model.a)*F(j,side)
            z=model.jet(ball(x),ball(y))
            if not(z['Y']>=0 and z['Z']>=0):continue
            q=z['u']+model.ee*(z['ux']**2+z['uy']**2)/2
            speed,cost=problem_data(z['Y'],z['Z'])
            r=q-speed*(z['ux']+z['uy'])-cost;d=q-model.bilinear(z['Y'],z['Z'])[0]
            rl=rl.max(abs(r).abs_lower());dl=dl.max(abs(d).abs_lower());points+=1
    return dict(residual_lower=lower_record(rl),distance_lower=lower_record(dl),
                bound_lower=lower_record(rl+dl),valid_points=points,point_grid_side=side+1)

def main():
    ctx.prec=192;path=ROOT/'results/twod_reconstruction.json'
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==(ROOT/'results/twod_reconstruction.sha256').read_text().strip()
    selection=json.loads(path.read_text());assert selection['status']=='rigorous_interval_enclosures'
    reference=json.loads((ROOT/'results/twod_auxiliary_reference.json').read_text())
    for aux in reference['auxiliary_functions']:
        assert certify_auxiliary(aux['polynomial'])==aux['certificate']
    selected=next(a for a in reference['auxiliary_functions'] if a['polynomial']['order']==6)
    rho=selected['certificate']['residual_bound']['upper_numerator'];rows=[]
    for row in sorted(selection['records'],key=lambda r:r['n']):
        n=row['n'];record=json.loads((ROOT/f'results/twod_inputs/n{n}.json').read_text())
        old=next(r for r in reference['records'] if r['approximation']==f'bilinear_{n}')
        distance=bilinear_distance(selected['polynomial'],np.array(record['node_numerators']),reference['design']['distance_interval_tolerance'])
        assert {k:v for k,v in distance.items() if k!='seconds'}=={k:v for k,v in old['distance'].items() if k!='seconds'}
        lo=F(max(0,distance['lower_numerator']-rho),SCALE);hi=F(distance['upper_numerator']+rho,SCALE)
        assert lo==F(old['error_interval']['lower_numerator'],SCALE) and hi==F(old['error_interval']['upper_numerator'],SCALE)
        B=F(row['selected_error_upper']['numerator'],row['selected_error_upper']['denominator'])
        baseline=uncorrected_lower(record,row['selected_exponent'])
        rows.append(dict(n=n,selected_exponent=row['selected_exponent'],
                         error_lower=lower_record(ball(lo)),error_upper=upper_record(ball(hi)),
                         effectivity_upper=upper_record(ball(B/lo)),uncorrected=baseline))
        print(n,'effectivity <=',float(B/lo),'uncorrected bound >=',baseline['bound_lower']['decimal'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    report=dict(status='PASS',selection_sha256=sha,auxiliary_certificates_replayed=3,
                reference='Preserved additional PDE approximation, with certified viscosity sub- and supersolutions; read after the new selection is frozen.',records=rows)
    (ROOT/'results/twod_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
