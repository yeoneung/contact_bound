"""Complete fixed-critic comparison on three state-dependent-speed problems.

All settings are fixed in results/state_dependent_design.json. Reuse of earlier
verified critic caches is explicit and their measured construction time is
charged to each standalone certificate. --resume validates all source hashes.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hjrl.verified_arb import SCALE,ceil_int,floor_int,interval_dict
from hjrl.verified_variable_speed import VariableSpeed


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ball(lo,hi):return arb(arb(int(lo)+int(hi))/(2*SCALE),arb(int(hi)-int(lo))/(2*SCALE))


def load_critic(name):
    vp=ROOT/f'results/comparison_enclosures_{name}.npz'
    dp=ROOT/f'results/contact_derivatives_{name}.npz'
    vm=json.loads(vp.with_suffix('.json').read_text())
    dm=json.loads(dp.with_suffix('.json').read_text())
    assert digest(vp)==vm['enclosures_sha256'] and digest(dp)==dm['sha256']
    assert digest(ROOT/'results/verified_neural_models.pt')==vm['weights_sha256']
    assert digest(ROOT/'code/hjrl/verified_arb.py')==vm['arithmetic_source_sha256']
    for p,h in dm['hashes'].items():assert digest(ROOT/p)==h,p
    v=np.load(vp,allow_pickle=False);d=np.load(dp,allow_pickle=False)
    return {'name':name,'ulo':v['lower'],'uhi':v['upper'],'dlo':d['lower'],'dhi':d['upper'],
            'lu':arb(vm['L_u_numerator'])/SCALE,'base_seconds':vm['evaluation_seconds'],
            'derivative_seconds':dm['seconds'],'value_sha256':digest(vp),'derivative_sha256':digest(dp)}


def contact_bound(problem,critic,m,n):
    start=time.perf_counter()
    ulo,uhi,dlo,dhi=(critic[k] for k in ('ulo','uhi','dlo','dhi'))
    lu=critic['lu'];M=len(ulo);D=max(M,n);B=2**24
    q,r,eps=arb(1)/M,arb(1)/n,arb(1)/m
    cap=lu.min(problem.lv);R=cap*eps+r+q
    assert lu*eps+r+q<1
    wing=ceil_int((lu*eps+r+q)*M/2)+2
    threshold=ceil_int(((lu+cap)*r+r*r*m+(lu+R*m)*q)*SCALE)
    correction=lu*q+(problem.le+problem.lf*R*m)*r+(problem.mf+R)*m*(q+r)
    margin=ceil_int(m*(q+r)*SCALE)
    pu=SCALE*m//(2*D*D);su=SCALE*m//D;bu=SCALE*m//(B*D)
    assert pu*2*D*D==SCALE*m and bu*B*D==SCALE*m and D%M==D%n==0
    yn=-D+(2*np.arange(n,dtype=np.int64)+1)*(D//n)
    g=[problem.cost(arb(int(y))/D) for y in yn]
    b=[problem.speed(arb(int(y))/D) for y in yn]
    glo=np.array([floor_int(v*SCALE) for v in g],dtype=np.int64)
    ghi=np.array([ceil_int(v*SCALE) for v in g],dtype=np.int64)
    blo=np.array([floor_int(v*B) for v in b],dtype=np.int64)
    bhi=np.array([ceil_int(v*B) for v in b],dtype=np.int64)
    # Establish integer product safety before the vectorized search.
    assert D*D*pu+int(max(abs(ulo).max(),abs(uhi).max()))+int(bhi.max())*D*bu<2**62
    offset=np.arange(-wing,wing+1,dtype=np.int64)
    maxima={-1:0,1:0};unfiltered={-1:0,1:0};counts={-1:0,1:0}
    filter_seconds=0.0
    for begin in range(0,n,64):
        js=np.arange(begin,min(begin+64,n),dtype=np.int64)
        base=(2*js+1)*M//(2*n)
        ix=(base[:,None]+offset[None,:])%M
        xn=-D+2*ix*(D//M)
        delta=(xn-yn[js,None]+D)%(2*D)-D
        penalty=delta*delta*pu
        for sigma in (-1,1):
            low=(ulo[ix] if sigma==1 else -uhi[ix])-penalty
            high=(uhi[ix] if sigma==1 else -ulo[ix])-penalty
            relevant=(high>=low.max(axis=1)[:,None]-threshold)&(abs(delta)<=ceil_int(R*D))
            residual=(uhi[ix]-glo[js,None]+bhi[js,None]*abs(delta)*bu-penalty if sigma==1
                      else ghi[js,None]-ulo[ix]-blo[js,None]*abs(delta)*bu-penalty)
            unfiltered[sigma]=max(unfiltered[sigma],int(np.where(relevant,residual,0).max()))
            fs=time.perf_counter()
            p=sigma*delta*su
            relevant &= (p+margin>=dlo[ix])&(p-margin<=dhi[ix])
            maxima[sigma]=max(maxima[sigma],int(np.where(relevant,residual,0).max()))
            counts[sigma]+=int(relevant.sum())
            filter_seconds+=time.perf_counter()-fs
    elapsed=time.perf_counter()-start
    row={'method':'contact','reciprocal_scale':m,'nodes':M,'centers':n,
         'coverage_correction':interval_dict(correction),'candidate_counts':{str(k):v for k,v in counts.items()},
         'one_sided_bounds':{str(s):interval_dict(arb(maxima[s])/SCALE+correction) for s in (-1,1)},
         'complete_bound':interval_dict(arb(max(maxima.values()))/SCALE+correction),
         'unfiltered_bound':interval_dict(arb(max(unfiltered.values()))/SCALE+correction),
         'method_seconds':elapsed,'filter_seconds':filter_seconds,
         'unfiltered_method_seconds':elapsed-filter_seconds,
         'unfiltered_standalone_seconds':critic['base_seconds']+elapsed-filter_seconds,
         'standalone_seconds':critic['base_seconds']+critic['derivative_seconds']+elapsed}
    return row


def build_operator(problem,n,m,panels,cells=4096):
    """Critic-independent flow and quadrature enclosures, with all costs timed."""
    start=time.perf_counter();M=65536
    h=arb(1)/m
    costs_lo=np.empty((n,3),np.int64);costs_hi=np.empty((n,3),np.int64)
    indices=np.empty((n,3),np.int64);delta_lo=np.empty((n,3),np.int64);delta_hi=np.empty((n,3),np.int64)
    max_radius=0
    for i in range(n):
        x=arb(-n+2*i)/n;theta=problem.phase(x)
        for column,a in enumerate((0,-1,1)):
            X=x if a==0 else problem.flow_from_phase(theta,a,h)
            j=floor_int((X+1)*M/2+arb(1)/2)
            delta=X-(-1+arb(2*j)/M)
            assert delta.abs_upper()<arb(3)/M
            c,error=problem.held_cost(x,a,h,panels,theta)
            max_radius=max(max_radius,ceil_int(error*SCALE))
            costs_lo[i,column],costs_hi[i,column]=floor_int(c*SCALE),ceil_int(c*SCALE)
            indices[i,column]=j%M
            delta_lo[i,column],delta_hi[i,column]=floor_int(delta*SCALE),ceil_int(delta*SCALE)
    flow_cost_seconds=time.perf_counter()-start
    hs=time.perf_counter();bh=problem.hold_bias(h,cells);hold_seconds=time.perf_counter()-hs
    arrays={'cost_lo':costs_lo,'cost_hi':costs_hi,'indices':indices,'delta_lo':delta_lo,'delta_hi':delta_hi}
    metadata={'beta_quarters':problem.beta_quarters,'nodes':n,'reciprocal_hold':m,'panels':panels,
              'cost_mode':'analytic' if problem.beta_quarters==0 else 'verified_piecewise_midpoint',
              'flow_cost_seconds':flow_cost_seconds,'hold_bias_seconds':hold_seconds,
              'operator_seconds':flow_cost_seconds+hold_seconds,
              'quadrature_radius_upper':max_radius/SCALE,'hold_bias':interval_dict(bh)}
    return arrays,metadata


def interval_product(alo,ahi,blo,bhi):
    # Python integers avoid overflow of the 2^-40 fixed-point products.
    alo,ahi,blo,bhi=(np.asarray(v,dtype=np.int64).astype(object) for v in (alo,ahi,blo,bhi))
    products=[alo*blo,alo*bhi,ahi*blo,ahi*bhi]
    lower=np.minimum.reduce(products)//SCALE
    upper=-((-np.maximum.reduce(products))//SCALE)
    return lower.astype(np.int64),upper.astype(np.int64)


def bellman_bound(problem,critic,arrays,metadata,derivative_mode=True):
    start=time.perf_counter();n=metadata['nodes'];m=metadata['reciprocal_hold']
    M=len(critic['ulo']);h=arb(1)/m;gamma=(-h).exp()
    ix=arrays['indices']
    # The three adjacent derivative cells cover the segment from the selected
    # node to every enclosed successor, including nearest-cell ties.
    if derivative_mode:
        dl=np.minimum.reduce([critic['dlo'][(ix+j)%M] for j in (-1,0,1)])
        dh=np.maximum.reduce([critic['dhi'][(ix+j)%M] for j in (-1,0,1)])
    else:
        lip=ceil_int(critic['lu']*SCALE)
        dl=np.full(ix.shape,-lip,dtype=np.int64);dh=np.full(ix.shape,lip,dtype=np.int64)
    vl,vh=interval_product(dl,dh,arrays['delta_lo'],arrays['delta_hi'])
    vl+=critic['ulo'][ix];vh+=critic['uhi'][ix]
    gl,gh=floor_int(gamma*SCALE),ceil_int(gamma*SCALE)
    ql,qh=interval_product(vl,vh,gl,gh)
    ql+=arrays['cost_lo'];qh+=arrays['cost_hi']
    tl,th=ql.min(axis=1),qh.min(axis=1)
    stride=M//n;assert stride*n==M
    residual=max(int((critic['uhi'][::stride]-tl).max()),int((th-critic['ulo'][::stride]).max()),0)
    coverage=problem.bellman_coverage(critic['lu'],h,n)
    bh=arb(metadata['hold_bias']['upper_numerator'])/SCALE
    bound=(arb(residual)/SCALE+coverage+bh)/(1-gamma)
    eval_seconds=time.perf_counter()-start
    return {'method':'bellman','nodes':n,'reciprocal_hold':m,'panels':metadata['panels'],
            'successor_mode':'derivative' if derivative_mode else 'lipschitz',
            'sample_residual_upper':residual/SCALE,'coverage_correction':interval_dict(coverage/(1-gamma)),
            'hold_correction':interval_dict(bh/(1-gamma)),
            'quadrature_radius_amplified':interval_dict((arb(int(round(metadata['quadrature_radius_upper']*SCALE)))/SCALE)/(1-gamma)),
            'successor_enclosure_width_upper':int((vh-vl).max())/SCALE,
            'complete_bound':interval_dict(bound),'operator_seconds':metadata['operator_seconds'],
            'residual_evaluation_seconds':eval_seconds,'method_seconds':metadata['operator_seconds']+eval_seconds,
            'standalone_seconds':critic['base_seconds']+(critic['derivative_seconds'] if derivative_mode else 0)+metadata['operator_seconds']+eval_seconds}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();ctx.prec=128
    design_path=ROOT/'results/state_dependent_design.json';design=json.loads(design_path.read_text())
    sources=[Path(__file__),ROOT/'code/hjrl/verified_variable_speed.py',ROOT/'code/hjrl/verified_arb.py',design_path]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    out=ROOT/'results/state_dependent_comparison.json'
    data={'design':design,'hashes':hashes,'precision_bits':128,'records':[],'operators':[],'complete':False}
    if args.resume and out.exists():
        data=json.loads(out.read_text());assert data['hashes']==hashes
    critics={name:load_critic(name) for name in design['critics']}
    data['critic_caches']={name:{k:c[k] for k in ('base_seconds','derivative_seconds','value_sha256','derivative_sha256')} for name,c in critics.items()}
    def save():out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    if args.smoke:
        p=VariableSpeed(1);c=critics['MLP'];arr,meta=build_operator(p,256,16,2,128)
        print(json.dumps(bellman_bound(p,c,arr,meta),indent=2));return
    cache_dir=ROOT/'results/state_dependent_operators';cache_dir.mkdir(exist_ok=True)
    for beta in design['beta_quarters']:
        problem=VariableSpeed(beta)
        for name,c in critics.items():
            for n in design['contact_centers']:
                for m in design['contact_reciprocal_scales']:
                    key=f'{beta}:{name}:contact:{n}:{m}'
                    if any(r['key']==key for r in data['records']):continue
                    row=contact_bound(problem,c,m,n);row.update(key=key,critic=name,beta_quarters=beta)
                    data['records'].append(row);save()
                    print(key,'bound',row['complete_bound']['upper'],'method seconds',round(row['method_seconds'],2),flush=True)
        for n in design['bellman_nodes']:
            for m in design['bellman_reciprocal_holds']:
                for panels in ([0] if beta==0 else design['quadrature_panels_per_smooth_segment']):
                    opkey=f'b{beta}_N{n}_h{m}_q{panels}'
                    cp=cache_dir/(opkey+'.npz');mp=cp.with_suffix('.json')
                    if args.resume and cp.exists() and mp.exists():
                        meta=json.loads(mp.read_text());assert meta['hashes']==hashes and digest(cp)==meta['sha256']
                        arrays=dict(np.load(cp,allow_pickle=False))
                    else:
                        arrays,meta=build_operator(problem,n,m,panels,design['hold_bias_cells'])
                        np.savez_compressed(cp,**arrays);meta.update(hashes=hashes,sha256=digest(cp))
                        mp.write_text(json.dumps(meta,indent=2)+'\n')
                    if not any(r['key']==opkey for r in data['operators']):
                        data['operators'].append({'key':opkey,**meta})
                    for name,c in critics.items():
                        for derivative_mode in (False,True):
                            mode='derivative' if derivative_mode else 'lipschitz'
                            key=f'{beta}:{name}:bellman:{n}:{m}:{panels}:{mode}'
                            if any(r['key']==key for r in data['records']):continue
                            row=bellman_bound(problem,c,arrays,meta,derivative_mode);row.update(key=key,critic=name,beta_quarters=beta,operator_key=opkey)
                            data['records'].append(row);save()
                            print(key,'bound',row['complete_bound']['upper'],'method seconds',round(row['method_seconds'],2),flush=True)
    data['complete']=True;save()


if __name__=='__main__':main()
