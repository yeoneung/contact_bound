"""Rigorous contact and closed-loop certificates for two freshly trained critics.

All model/cost/derivative enclosures use Arb. Contact-grid comparisons use exact
int64 arithmetic with explicit overflow guards. Approximate clipped iterates are
accepted only after their Bellman residual has been enclosed with Arb.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
from flint import arb, ctx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from hjrl.verified_arb import (CertifiedCritic, SCALE, ceil_int, floor_int, upper,
                              interval_dict, ridge_value, ridge_cost, held_cost)
from hjrl.critic import FourierMLP, MinBranchCritic

M=16384                 # neural evaluation and trajectory lattice
DERIVATIVE_CELLS=65536  # covers the entire torus
CENTERS=262144          # contact centers, half-cell shifted
SCALES=(64,16)          # reciprocal epsilon, fixed before evaluation
H=arb(1)/32
STEPS=512              # horizon 16, rigorous remaining tail
ACTION=(0,-1,1)


def ball(lo,hi):
    return arb(arb(int(lo+hi))/(2*SCALE),arb(int(hi-lo))/(2*SCALE))


def model_data(state,branches,name):
    model=CertifiedCritic(state,branches)
    start=time.perf_counter()
    lu=0
    for begin in range(0,DERIVATIVE_CELLS,512):
        xs=[arb(arb(-DERIVATIVE_CELLS+2*i+1)/DERIVATIVE_CELLS,
                arb(1)/DERIVATIVE_CELLS) for i in range(begin,min(begin+512,DERIVATIVE_CELLS))]
        _,bounds=model.evaluate(xs,True)
        lu=max(lu,max(bounds))
    print(name,'verified L_u',lu/SCALE,'derivative seconds',round(time.perf_counter()-start,2),flush=True)
    lo,hi=[],[]
    for begin in range(0,M,512):
        xs=[arb(-M+2*i)/M for i in range(begin,min(begin+512,M))]
        values=model.evaluate(xs)
        lo.extend(floor_int(v*SCALE) for v in values)
        hi.extend(ceil_int(v*SCALE) for v in values)
    return np.array(lo,dtype=np.int64),np.array(hi,dtype=np.int64),arb(lu)/SCALE,time.perf_counter()-start


def contact_bound(ulo,uhi,glo,ghi,lu,m):
    n=CENTERS
    eps=arb(1)/m
    q=arb(1)/M
    radius=2*lu*eps
    wing=min(M//2,ceil_int(radius*M/2)+2)
    offsets=np.arange(-wing,wing+1,dtype=np.int64)
    rwindow=min(arb(1),arb(2*(wing+2))/M)
    threshold=ceil_int(2*(lu+rwindow*m)*q*SCALE)
    penalty_unit=SCALE*m//(2*n*n)
    assert penalty_unit*2*n*n == SCALE*m
    slope_unit=SCALE*m//n
    assert slope_unit*n == SCALE*m
    assert n*n*penalty_unit+max(abs(ulo).max(),abs(uhi).max()) < 2**62
    maxres=0
    start=time.perf_counter()
    for begin in range(0,n,256):
        js=np.arange(begin,min(begin+256,n),dtype=np.int64)
        yn=-n+2*js+1
        base=js*M//n
        ix=(base[:,None]+offsets[None,:])%M
        xn=-n+2*ix*(n//M)
        displacement=(xn-yn[:,None]+n)%(2*n)-n
        penalty=displacement*displacement*penalty_unit
        slope=abs(displacement)*slope_unit
        for sigma in (-1,1):
            low=(ulo[ix] if sigma==1 else -uhi[ix])-penalty
            high=(uhi[ix] if sigma==1 else -ulo[ix])-penalty
            relevant=high >= low.max(axis=1)[:,None]-threshold
            residual=(uhi[ix]-glo[ix]+slope if sigma==1 else ghi[ix]-ulo[ix]-slope)
            maxres=max(maxres,int(np.where(relevant,residual,0).max()))
    k=lu+(arb.pi()/2+arb.pi()**2/4)+m
    loc=k*q
    lv=arb.pi()/2
    le=lv+arb.pi()**2/4
    cv=2*lv*le
    r=arb(1)/n
    zeta=3*lv*r+r*r/(2*eps)
    center=le*r+zeta+2*(2*zeta*(le+lv+m)).sqrt()
    bound=arb(maxres)/SCALE+loc+cv*eps+center
    return {"reciprocal_scale":m,"centers":n,"grid_nodes":M,
            "near_contact_residual_upper":interval_dict(arb(maxres)/SCALE),
            "location_correction":interval_dict(loc),"comparison_correction":interval_dict(cv*eps),
            "center_correction":interval_dict(center),"complete_bound":interval_dict(bound),
            "seconds":time.perf_counter()-start}


def hold_bias():
    """Continuous-state upper bound, independent of a reference value solver.

    Away from a valley by >=h, the optimal action is held exactly. Within d<h,
    compare stopping with moving through the valley, integrating its defect.
    """
    a=arb.pi()/2
    gamma=(-H).exp()
    best=arb(0)
    for i in range(4096):
        d=arb(H*(2*i+1)/8192,H/8192)
        stop=(1-gamma)*a*(a*d).sin()
        length=H-d
        integral=(a-(-length).exp()*((a*length).sin()+a*(a*length).cos()))/(1+a*a)
        move=2*a*(-d).exp()*integral
        best=best.max(upper(stop.min(move)))
    return upper(best)


def clipped_and_policies(state,branches,ulo,uhi,lu,costs,vref,bh):
    gamma=(-H).exp()
    gf=float(gamma)
    ug=[ball(int(l),int(h)) for l,h in zip(ulo,uhi)]
    ucenter=(ulo.astype(float)+uhi.astype(float))/(2*SCALE)
    shifts=np.array(ACTION)*(M//64) # h=1/32 on spacing 2/M
    successors=(np.arange(M)[:,None]+shifts[None,:])%M
    cf=np.array([[float(v) for v in row] for row in costs])
    # Clipping uses every second node; all its held successors remain nodes.
    node_successors=successors[::2]//2
    w=ucenter[::2].copy()
    for iteration in range(20000):
        new=np.minimum(ucenter[::2],(cf[::2]+gf*w[node_successors]).min(axis=1))
        diff=float(abs(new-w).max());w=new
        if diff<1e-12:break
    wb=[arb(float(v)) for v in w]
    ed=arb(0)
    for i in range(M//2):
        backup=costs[2*i][0]+gamma*wb[int(node_successors[i,0])]
        for a in (1,2):backup=backup.min(costs[2*i][a]+gamma*wb[int(node_successors[i,a])])
        clipped=ug[2*i].min(backup)
        ed=ed.max(upper((wb[i]-clipped).abs_upper()))
    lstar=lu.max(H*(arb.pi()/2+arb.pi()**2/4)/(1-gamma))
    stopping=upper(ed/(1-gamma))
    offgrid=upper(lstar*2/M) # half of the clipping-grid spacing 4/M
    # Exact node closure removes repeated spatial interpolation. Off-grid starts
    # still pay the one-time interpolation error; no hidden zero error assumption.
    cmin=[]
    global_sample=arb(0)
    for i in range(M):
        q=[costs[i][a]+gamma*ug[int(successors[i,a])] for a in range(3)]
        qmin=q[0].min(q[1]).min(q[2]);cmin.append(qmin)
        global_sample=global_sample.max(upper((ug[i]-qmin).max(arb(0))))
    s=upper(global_sample+(2*lu+H*(arb.pi()/2+arb.pi()**2/4))/M)
    global_term=upper((s+bh)/(1-gamma))
    # Freeze implemented action maps. Regret is then verified for their actual
    # choices, so no exact-real argmin or gradient-sign assumption is necessary.
    actions_lookahead=(cf+gf*ucenter[successors]).argmin(axis=1)
    torch.set_num_threads(4)
    model=(FourierMLP(1,n_freq=8,width=64,depth=3) if branches==1
           else MinBranchCritic(1,K=2,n_freq=8,width=64,depth=3)).double()
    model.load_state_dict(state);model.eval()
    x=torch.tensor((-1+2*np.arange(M)/M)[:,None],dtype=torch.float64,requires_grad=True)
    grad=torch.autograd.grad(model(x).sum(),x)[0].detach().numpy()[:,0]
    actions_gradient=np.where(grad>0,1,np.where(grad<0,2,0))
    unorm=arb(0)
    for v in ug:unorm=unorm.max(upper(v.abs_upper()))
    unorm=upper(unorm+lu/M)
    gmax=1+arb.pi()/2
    tail_bound=upper(gamma**STEPS*(gmax+unorm))
    cost_tail=upper(gamma**STEPS*gmax)
    starts=1+np.arange(64)*(M//64) # odd nodes: off the clipping mesh
    records=[]
    for pname,action_map in (("Lookahead",actions_lookahead),("Gradient",actions_gradient)):
        rows=[]
        for start in starts:
            i=int(start);discount=arb(1);traj=arb(0);regret=arb(0);jcost=arb(0)
            for step in range(STEPS):
                a=int(action_map[i]);j=int(successors[i,a])
                traj+=discount*(cmin[i]-ug[i]).max(arb(0))
                regret+=discount*(costs[i][a]+gamma*ug[j]-cmin[i]).max(arb(0))
                jcost+=discount*costs[i][a]
                discount*=gamma;i=j
            interp=(wb[int(start)//2]+wb[(int(start)//2+1)%(M//2)])/2
            rho=upper((ug[int(start)]-interp+stopping+offgrid).max(arb(0)))
            hold=upper(bh/(1-gamma))
            total=upper(traj+regret+rho+hold+tail_bound)
            original=upper(traj+regret+global_term+tail_bound)
            gaplo=(jcost-vref[int(start)]).lower()
            gaphi=(jcost+cost_tail-vref[int(start)]).upper()
            assert total >= gaphi
            rows.append({"start_index":int(start),"gap":interval_dict(gaplo.union(gaphi)),
                         "trajectory":interval_dict(traj),"regret":interval_dict(regret),
                         "rho_upper":interval_dict(rho),"complete_bound":interval_dict(total),
                         "global_bound":interval_dict(original)})
        records.append({"policy":pname,"starts":64,"rows":rows,
                        "maximum_gap_upper":max(r["gap"]["upper"] for r in rows),
                        "maximum_bound_upper":max(r["complete_bound"]["upper"] for r in rows),
                        "maximum_global_bound_upper":max(r["global_bound"]["upper"] for r in rows),
                        "maximum_regret_upper":max(r["regret"]["upper"] for r in rows)})
    return {"hold":interval_dict(H),"horizon":16,"clipping_nodes":M//2,
            "iterations":iteration+1,"operator_residual":interval_dict(ed),
            "stopping_correction":interval_dict(stopping),"off_grid_correction":interval_dict(offgrid),
            "L_star":interval_dict(lstar),"hold_bias_upper":interval_dict(bh),
            "hold_correction":interval_dict(bh/(1-gamma)),"trajectory_tail":interval_dict(tail_bound),
            "cost_tail":interval_dict(cost_tail),"operator_error_included_in_residual":True,
            "exact_successor_node_closure":True,"policies":records}


def main():
    ctx.prec=128
    weights=ROOT/"results/verified_neural_models.pt"
    states=torch.load(weights,map_location='cpu',weights_only=True)
    data={"precision_bits":128,"arithmetic":"Arb enclosures; exact fixed-point contact comparisons",
          "weights_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
          "derivative_cells":DERIVATIVE_CELLS,"evaluation_nodes":M,
          "mathematical_features":"sin(k*pi*x), cos(k*pi*x), exact stored dyadic weights",
          "records":[]}
    xs=[arb(-M+2*i)/M for i in range(M)]
    g=[ridge_cost(x) for x in xs]
    vlo=[ridge_value(x) for x in xs]
    glo=np.array([floor_int(v*SCALE) for v in g],dtype=np.int64)
    ghi=np.array([ceil_int(v*SCALE) for v in g],dtype=np.int64)
    costs=[[held_cost(x,a,H) for a in ACTION] for x in xs]
    bh=hold_bias()
    for name,branches in (("MLP",1),("Min2",2)):
        start=time.perf_counter()
        ulo,uhi,lu,evaluation_time=model_data(states[name],branches,name)
        row={"critic":name,"verified_L_u":interval_dict(lu),"evaluation_seconds":evaluation_time}
        sample_error=arb(0)
        for l,h,v in zip(ulo,uhi,vlo):sample_error=sample_error.max(abs(ball(int(l),int(h))-v))
        row["true_error_enclosure"]=interval_dict(sample_error.lower().union(
            sample_error.upper()+(lu+arb.pi()/2)/M))
        row["contacts"]=[]
        for m in SCALES:
            result=contact_bound(ulo,uhi,glo,ghi,lu,m)
            row["contacts"].append(result)
            print(name,'epsilon',1/m,'contact bound',result['complete_bound']['upper'],flush=True)
        row["policy"]=clipped_and_policies(states[name],branches,ulo,uhi,lu,costs,vlo,bh)
        row["total_seconds"]=time.perf_counter()-start
        data["records"].append(row)
        data["source_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
             (Path(__file__),ROOT/"code/hjrl/verified_arb.py")}
        (ROOT/"results/verified_neural_policy.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
        print(name,'policies',[(r['policy'],r['maximum_gap_upper'],r['maximum_bound_upper'])
                              for r in row['policy']['policies']],flush=True)


if __name__=='__main__':main()
