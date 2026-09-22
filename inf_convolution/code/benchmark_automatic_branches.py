"""Total cost of reaching a certified tolerance, without reference values.

Floating point proposals only order candidates. At most three candidates per
grid are verified. A result is accepted only after whole-period verification.
Every method includes the numerical solve, spline construction, proposal search,
all unsuccessful attempts, and the final certificate in its elapsed time.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time,platform
import numpy as np
from flint import ctx
from asymmetric_input import CASES,construct,PeriodicSpline,cost
from automatic_branches import discover_float,float_envelope
from certify_automatic_branches import certify,certify_polynomial

ROOT=Path(__file__).resolve().parents[1]
SCALES=tuple(range(4,41))
WIDTHS=tuple(range(2,17))
GRIDS=tuple(2**j for j in range(6,15))

def polynomial_proposal(spline,y,u,du,width,degree):
    n=spline.n;h=float(spline.h);nodes=np.asarray(spline.record['numerical_node_numerators'])
    peaks=np.flatnonzero((nodes>np.roll(nodes,1))&(nodes>=np.roll(nodes,-1)))
    w=u.copy();dw=du.copy();gaps=[]
    for i in peaks:
        for shift in [-n,0,n]:
            a,b=(i+shift-width)*h,(i+shift+width)*h
            if b<=0 or a>=2:continue
            gaps.append((a,b))
    gaps.sort()
    if not gaps or any(a[1]>b[0] for a,b in zip(gaps,gaps[1:])):return None
    for a,b in gaps:
        coeff=[]
        for x in [a,b]:
            i=int(round(x/h))-(1 if x==a else 0);t=x/h-i
            A,B,C,D=spline.fcoeffs[i%n]
            co=[A+t*(B+t*(C+t*D)),(B+t*(2*C+3*D*t))/h,(C+3*D*t)/h**2,D/h**3 if degree==3 else 0.]
            coeff.append(co)
        def val(co,t):A,B,C,D=co;return A+t*(B+t*(C+t*D))
        if coeff[0][0]>val(coeff[1],a-b) or coeff[1][0]>val(coeff[0],b-a):return None
        valid=(y>=a)&(y<=b);t=y[valid]-a;s=y[valid]-b
        l=val(coeff[0],t);r=val(coeff[1],s);choose=l<=r
        ld=coeff[0][1]+t*(2*coeff[0][2]+3*coeff[0][3]*t)
        rd=coeff[1][1]+s*(2*coeff[1][2]+3*coeff[1][3]*s)
        w[valid]=np.minimum(l,r);dw[valid]=np.where(choose,ld,rd)
    return w,dw

def rank_candidates(spline,method):
    y=np.arange(min(4*spline.n,32768))*2/min(4*spline.n,32768)
    u,du,_,_=spline.float_jet(y);g=cost(y,spline.case);out=[]
    if method=='polynomial':
        for width in WIDTHS:
            for degree in [2,3]:
                z=polynomial_proposal(spline,y,u,du,width,degree)
                if z is None:continue
                w,dw=z;B=float(np.max(abs(w-u))+np.max(abs(w+abs(dw)-g)))
                out.append(dict(width=width,degree=degree,sampled_bound=B))
    else:
        for k in SCALES:
            eps=F(k,4)*spline.h;branches=discover_float(spline,float(eps))
            if not branches:continue
            w,dw,_=float_envelope(branches,y,method)
            if not np.all(np.isfinite(w)):continue
            B=float(np.max(abs(w-u))+np.max(abs(w+abs(dw)-g)))
            out.append(dict(scale_quarters=k,epsilon=[eps.numerator,eps.denominator],sampled_bound=B))
    return sorted(out,key=lambda r:r['sampled_bound'])

def run(case,method,target,max_n=16384,attempts=3):
    started=time.perf_counter();history=[]
    for n in GRIDS:
        if n>max_n:break
        t0=time.perf_counter();record=construct(n,case);spline=PeriodicSpline(record);constructed=time.perf_counter()
        ranking=rank_candidates(spline,method);ranked=time.perf_counter();evaluations=[]
        for candidate in ranking[:attempts]:
            if method=='polynomial':r=certify_polynomial(spline,candidate['width'],candidate['degree'],target=target)
            else:r=certify(spline,None,method,eps=F(*candidate['epsilon']),target=target)
            r['proposal']=candidate;evaluations.append(r)
            if r['status']=='certified' and F(r['error_upper']['numerator'],r['error_upper']['denominator'])<=target:break
        step=dict(n=n,construction_seconds=constructed-t0,proposal_seconds=ranked-constructed,
                  proposals=len(ranking),attempts=evaluations,seconds=time.perf_counter()-t0)
        history.append(step)
        if evaluations and evaluations[-1]['status']=='certified':
            r=evaluations[-1]
            if F(r['error_upper']['numerator'],r['error_upper']['denominator'])<=target:
                return dict(case=case,method=method,target=[target.numerator,target.denominator],status='target_certified',
                            selected_n=n,selected=r,input=record,history=history,total_seconds=time.perf_counter()-started)
    return dict(case=case,method=method,target=[target.numerator,target.denominator],status='budget_exhausted',
                history=history,total_seconds=time.perf_counter()-started)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',default=list(CASES));parser.add_argument('--methods',nargs='+',default=['polynomial','uncorrected','corrected'])
    parser.add_argument('--targets',nargs='+',default=['.02','.01']);parser.add_argument('--max-n',type=int,default=16384)
    parser.add_argument('--repetitions',type=int,default=3);parser.add_argument('--output',default='results/automatic_branches.json')
    args=parser.parse_args();ctx.prec=192;rows=[];path=ROOT/args.output
    source_names=['asymmetric_input.py','automatic_branches.py','direct_branch_polynomials.py','certify_automatic_branches.py','benchmark_automatic_branches.py','interval_utils.py']
    report=dict(status='incomplete',precision_bits=192,relative_tolerance=[1,50],absolute_tolerance='1e-10',
                cases=args.cases,methods=args.methods,targets=args.targets,repetitions=args.repetitions,
                grids=[n for n in GRIDS if n<=args.max_n],contact_scale_quarters=list(SCALES),
                polynomial_half_widths=list(WIDTHS),polynomial_degrees=[2,3],verified_candidates_per_grid=3,
                proposal='Rank all prescribed candidates using sampled floating-point bounds; verify up to three in that order. Samples are not upper certificates. Search exhaustion is not an impossibility claim.',
                timing='Sequential runs with cyclic method order. Includes every grid solve, spline construction, proposal, failed attempt and final verification. Excludes imports and independent checks.',
                environment=dict(python=platform.python_version(),platform=platform.platform(),processor='Intel Core i9-14900KF'),
                source_sha256={name:hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() for name in source_names},records=rows)
    for repetition in range(args.repetitions):
        for case in args.cases:
            for target in args.targets:
                # Rotate method order to reduce systematic timing-order effects.
                methods=args.methods[repetition%len(args.methods):]+args.methods[:repetition%len(args.methods)]
                for method in methods:
                    row=run(case,method,F(target),args.max_n);row['repetition']=repetition;rows.append(row)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(report,indent=2)+'\n')
                    print(case,target,method,row['status'],row.get('selected_n'),round(row['total_seconds'],4),
                          row.get('selected',{}).get('error_upper',{}).get('decimal'),flush=True)
    report['status']='completed_certified_target_study';path.write_text(json.dumps(report,indent=2)+'\n')
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
