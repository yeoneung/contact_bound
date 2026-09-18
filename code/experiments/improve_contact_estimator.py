"""Verified residuals of ordered sup/inf envelopes, covering center cells.

The Hamiltonian is evaluated at the envelope center and the signed quadratic
penalty is retained. Coverage is incorporated in candidate selection; no sampled
center or contact is silently treated as exact. Specializes f=a, lambda=1,
H(y,p)=|p|-g(y) to the existing Ridge1D fixed checkpoints.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from flint import arb, ctx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hjrl.verified_arb import SCALE, ceil_int, floor_int, interval_dict, ridge_cost


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def envelope_bound(ulo,uhi,lu,m,n):
    start=time.perf_counter()
    M=len(ulo)
    q,r,eps=arb(1)/M,arb(1)/n,arb(1)/m
    # A global contact of an L_u-Lipschitz critic has |x-y| <= L_u*eps:
    # its touching quadratic has slope in D^+/-u, hence norm <= L_u.
    required=lu*eps+r+q
    assert required < 1, 'This implementation avoids cut-locus ambiguity'
    wing=ceil_int(required*M/2)+2
    radius=arb(2*(wing+2))/M
    theta=2*lu*r+r*r*m+(lu+radius*m)*q
    threshold=ceil_int(theta*SCALE)
    correction=lu*q+(arb.pi()/2+arb.pi()**2/4)*r+(1+radius)*m*(q+r)
    cap=lu.min(arb.pi()/2)
    restricted_radius=cap*eps+r+q
    restricted_theta=(lu+cap)*r+r*r*m+(lu+restricted_radius*m)*q
    restricted_threshold=ceil_int(restricted_theta*SCALE)
    restricted_correction=lu*q+(arb.pi()/2+arb.pi()**2/4)*r+(1+restricted_radius)*m*(q+r)
    D=max(M,n)
    assert D%M==0 and D%n==0
    penalty_unit=SCALE*m//(2*D*D)
    slope_unit=SCALE*m//D
    assert penalty_unit*2*D*D == SCALE*m and slope_unit*D==SCALE*m
    assert D*D*penalty_unit+int(max(abs(ulo).max(),abs(uhi).max()))<2**62
    yn=-D+(2*np.arange(n,dtype=np.int64)+1)*(D//n)
    values=[ridge_cost(arb(int(y))/D) for y in yn]
    glo=np.array([floor_int(v*SCALE) for v in values],dtype=np.int64)
    ghi=np.array([ceil_int(v*SCALE) for v in values],dtype=np.int64)
    offsets=np.arange(-wing,wing+1,dtype=np.int64)
    maxima={-1:0,1:0}
    restricted_maxima={-1:0,1:0}
    candidates={-1:0,1:0}
    for begin in range(0,n,64):
        js=np.arange(begin,min(begin+64,n),dtype=np.int64)
        base=(2*js+1)*M//(2*n)
        ix=(base[:,None]+offsets[None,:])%M
        xn=-D+2*ix*(D//M)
        displacement=(xn-yn[js,None]+D)%(2*D)-D
        penalty=displacement*displacement*penalty_unit
        slope=abs(displacement)*slope_unit
        for sigma in (-1,1):
            low=(ulo[ix] if sigma==1 else -uhi[ix])-penalty
            high=(uhi[ix] if sigma==1 else -ulo[ix])-penalty
            relevant=high>=low.max(axis=1)[:,None]-threshold
            residual=(uhi[ix]-glo[js,None]+slope-penalty if sigma==1
                      else ghi[js,None]-ulo[ix]-slope-penalty)
            maxima[sigma]=max(maxima[sigma],int(np.where(relevant,residual,0).max()))
            restricted=(high>=low.max(axis=1)[:,None]-restricted_threshold)
            restricted &= abs(displacement)<=ceil_int(restricted_radius*D)
            restricted_maxima[sigma]=max(restricted_maxima[sigma],int(np.where(restricted,residual,0).max()))
            candidates[sigma]+=int(relevant.sum())
    bounds={str(s):interval_dict(arb(maxima[s])/SCALE+correction) for s in (-1,1)}
    restricted_bounds={str(s):interval_dict(arb(restricted_maxima[s])/SCALE+restricted_correction) for s in (-1,1)}
    return {'reciprocal_scale':m,'grid_nodes':M,'centers':n,
            'threshold':interval_dict(theta),'threshold_integer':threshold,
            'window_radius':interval_dict(radius),'coverage_correction':interval_dict(correction),
            'candidate_residual_upper':{str(s):maxima[s]/SCALE for s in (-1,1)},
            'candidate_counts':{str(s):candidates[s] for s in (-1,1)},
            'one_sided_bounds':bounds,
            'complete_bound':interval_dict(arb(max(maxima.values()))/SCALE+correction),
            'slope_cap':interval_dict(cap),
            'restricted_threshold':interval_dict(restricted_theta),
            'restricted_coverage_correction':interval_dict(restricted_correction),
            'restricted_candidate_residual_upper':{str(s):restricted_maxima[s]/SCALE for s in (-1,1)},
            'restricted_one_sided_bounds':restricted_bounds,
            'restricted_complete_bound':interval_dict(arb(max(restricted_maxima.values()))/SCALE+restricted_correction),
            'seconds':time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--coarse-only',action='store_true')
    args=parser.parse_args()
    ctx.prec=128
    design_path=ROOT/'results/envelope_design.json'
    design=json.loads(design_path.read_text())
    hashes={str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__),design_path,ROOT/'code/hjrl/verified_arb.py',ROOT/'results/verified_neural_models.pt')}
    out=ROOT/'results/improved_contact_estimator.json'
    data={'hashes':hashes,'design':design,'precision_bits':128,'records':[]}
    if args.resume and out.exists():
        data=json.loads(out.read_text())
        assert data['hashes']==hashes
    def save():
        out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    for name in ('MLP','Min2'):
        cache=ROOT/f'results/comparison_enclosures_{name}.npz'
        meta=json.loads(cache.with_suffix('.json').read_text())
        assert digest(cache)==meta['enclosures_sha256']
        assert digest(ROOT/'results/verified_neural_models.pt')==meta['weights_sha256']
        assert digest(ROOT/'code/hjrl/verified_arb.py')==meta['arithmetic_source_sha256']
        arrays=np.load(cache,allow_pickle=False)
        lu=arb(meta['L_u_numerator'])/SCALE
        row=next((r for r in data['records'] if r['critic']==name),None)
        if row is None:
            row={'critic':name,'enclosure_sha256':digest(cache),'verified_L_u':interval_dict(lu),'settings':[]}
            data['records'].append(row)
        levels=design['grids'][:1] if args.coarse_only else design['grids']
        for grid in levels:
            M,n=grid['nodes'],grid['centers']
            for m in design['reciprocal_scales']:
                if any((r['reciprocal_scale'],r['grid_nodes'],r['centers'])==(m,M,n) for r in row['settings']):continue
                stride=65536//M
                result=envelope_bound(arrays['lower'][::stride],arrays['upper'][::stride],lu,m,n)
                row['settings'].append(result)
                save()
                print(name,'M',M,'N',n,'1/eps',m,'bounds',
                      {s:b['upper'] for s,b in result['restricted_one_sided_bounds'].items()},
                      'seconds',round(result['seconds'],2),flush=True)
    data['complete']=not args.coarse_only
    save()


if __name__=='__main__':main()
