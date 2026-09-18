"""Discard impossible contact cells using rigorously enclosed signed derivatives.

The filter is optional: a true contact slope must lie in the derivative hull on
its cell. Smooth-head Hessian bounds tighten those hulls; the minimum critic
itself is not assumed C^2. All prescribed scales use the same fixed grid.
"""
from pathlib import Path
import hashlib,json,sys,time,argparse
import numpy as np
import torch
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hjrl.verified_arb import SCALE,ceil_int,floor_int,interval_dict,ridge_cost
from hjrl.verified_stationarity import SecondOrderCritic


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def filtered_bound(ulo,uhi,dlo,dhi,lu,m,n):
    start=time.perf_counter()
    M=len(ulo)
    q,r,eps=arb(1)/M,arb(1)/n,arb(1)/m
    cap=lu.min(arb.pi()/2)
    R=cap*eps+r+q
    assert lu*eps+r+q<1
    wing=ceil_int((lu*eps+r+q)*M/2)+2
    theta=(lu+cap)*r+r*r*m+(lu+R*m)*q
    threshold=ceil_int(theta*SCALE)
    correction=lu*q+(arb.pi()/2+arb.pi()**2/4)*r+(1+R)*m*(q+r)
    margin=ceil_int(m*(q+r)*SCALE)
    D=max(M,n)
    penalty_unit,slope_unit=SCALE*m//(2*D*D),SCALE*m//D
    assert D%M==0 and D%n==0
    assert penalty_unit*2*D*D==SCALE*m and slope_unit*D==SCALE*m
    assert D*D*penalty_unit+int(max(abs(ulo).max(),abs(uhi).max()))<2**62
    yn=-D+(2*np.arange(n,dtype=np.int64)+1)*(D//n)
    g=[ridge_cost(arb(int(y))/D) for y in yn]
    glo=np.array([floor_int(v*SCALE) for v in g],dtype=np.int64)
    ghi=np.array([ceil_int(v*SCALE) for v in g],dtype=np.int64)
    offsets=np.arange(-wing,wing+1,dtype=np.int64)
    maxima={-1:0,1:0};unfiltered={-1:0,1:0};counts={-1:0,1:0};before_counts={-1:0,1:0}
    for begin in range(0,n,64):
        js=np.arange(begin,min(begin+64,n),dtype=np.int64)
        base=(2*js+1)*M//(2*n)
        ix=(base[:,None]+offsets[None,:])%M
        xn=-D+2*ix*(D//M)
        delta=(xn-yn[js,None]+D)%(2*D)-D
        penalty=delta*delta*penalty_unit
        slope=abs(delta)*slope_unit
        for sigma in (-1,1):
            low=(ulo[ix] if sigma==1 else -uhi[ix])-penalty
            high=(uhi[ix] if sigma==1 else -ulo[ix])-penalty
            relevant=high>=low.max(axis=1)[:,None]-threshold
            relevant &= abs(delta)<=ceil_int(R*D)
            residual=(uhi[ix]-glo[js,None]+slope-penalty if sigma==1
                      else ghi[js,None]-ulo[ix]-slope-penalty)
            before_counts[sigma]+=int(relevant.sum())
            unfiltered[sigma]=max(unfiltered[sigma],int(np.where(relevant,residual,0).max()))
            p=sigma*delta*slope_unit
            relevant &= (p+margin>=dlo[ix]) & (p-margin<=dhi[ix])
            counts[sigma]+=int(relevant.sum())
            maxima[sigma]=max(maxima[sigma],int(np.where(relevant,residual,0).max()))
    return {'reciprocal_scale':m,'grid_nodes':M,'centers':n,
            'coverage_correction':interval_dict(correction),
            'candidate_residual_upper':{str(s):maxima[s]/SCALE for s in (-1,1)},
            'one_sided_bounds':{str(s):interval_dict(arb(maxima[s])/SCALE+correction) for s in (-1,1)},
            'complete_bound':interval_dict(arb(max(maxima.values()))/SCALE+correction),
            'unfiltered_residual_upper':{str(s):unfiltered[s]/SCALE for s in (-1,1)},
            'candidate_counts_before':{str(s):before_counts[s] for s in (-1,1)},
            'candidate_counts_after':{str(s):counts[s] for s in (-1,1)},
            'seconds':time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-derivatives',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    ctx.prec=128
    weights=ROOT/'results/verified_neural_models.pt'
    states=torch.load(weights,map_location='cpu',weights_only=True)
    sources=[Path(__file__),ROOT/'code/hjrl/verified_stationarity.py',ROOT/'code/hjrl/verified_arb.py',weights]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    out=ROOT/'results/stationary_contact_estimator.json'
    data={'precision_bits':128,'hashes':hashes,'reciprocal_scales':[4,8,16,32,64],
          'nodes':65536,'centers':16384,'method':'Optional stationarity filter for the shifted residual',
          'development':'Added after the first shifted-residual scan. All five scales are retained; no model selection.',
          'records':[]}
    if args.resume and out.exists():
        data=json.loads(out.read_text());assert data['hashes']==hashes
    def save():out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    for name,branches in (('MLP',1),('Min2',2)):
        cache=ROOT/f'results/comparison_enclosures_{name}.npz'
        meta=json.loads(cache.with_suffix('.json').read_text())
        assert digest(cache)==meta['enclosures_sha256'] and digest(weights)==meta['weights_sha256']
        assert digest(ROOT/'code/hjrl/verified_arb.py')==meta['arithmetic_source_sha256']
        values=np.load(cache,allow_pickle=False)
        ulo,uhi=values['lower'],values['upper']
        lu=arb(meta['L_u_numerator'])/SCALE
        derivative_path=ROOT/f'results/contact_derivatives_{name}.npz'
        derivative_meta=derivative_path.with_suffix('.json')
        if args.reuse_derivatives and derivative_path.exists():
            dm=json.loads(derivative_meta.read_text())
            assert dm['hashes']==hashes and dm['sha256']==digest(derivative_path)
            d=np.load(derivative_path,allow_pickle=False)
            dlo,dhi=d['lower'],d['upper']
        else:
            net=SecondOrderCritic(states[name],branches)
            start=time.perf_counter();lower=[];higher=[]
            for begin in range(0,65536,256):
                centers=[arb(-65536+2*i)/65536 for i in range(begin,begin+256)]
                a,b=net.tight_derivative_boxes(centers,arb(1)/65536)
                lower.extend(a);higher.extend(b)
                if begin%16384==0:print(name,'derivative cells',begin,flush=True)
            dlo,dhi=np.array(lower,dtype=np.int64),np.array(higher,dtype=np.int64)
            np.savez_compressed(derivative_path,lower=dlo,upper=dhi)
            dm={'hashes':hashes,'sha256':digest(derivative_path),'nodes':65536,
                'cell_radius':'1/65536','seconds':time.perf_counter()-start,
                'maximum_derivative_interval_width':int((dhi-dlo).max())/SCALE}
            derivative_meta.write_text(json.dumps(dm,indent=2)+'\n')
            print(name,'derivative verification seconds',dm['seconds'],flush=True)
        row=next((r for r in data['records'] if r['critic']==name),None)
        if row is None:
            row={'critic':name,'value_enclosures_sha256':digest(cache),
                 'derivative_enclosures_sha256':digest(derivative_path),
                 'derivative_verification_seconds':dm['seconds'],'settings':[]}
            data['records'].append(row)
        for m in data['reciprocal_scales']:
            if any(r['reciprocal_scale']==m for r in row['settings']):continue
            result=filtered_bound(ulo,uhi,dlo,dhi,lu,m,16384)
            row['settings'].append(result);save()
            print(name,'1/eps',m,'filtered bounds',
                  {s:v['upper'] for s,v in result['one_sided_bounds'].items()},flush=True)
    data['complete']=True;save()


if __name__=='__main__':main()
