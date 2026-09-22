"""Local reconstruction and reusable interval enclosures on a fixed input.

Both the polynomial control and the hybrid use the same patches, proposals,
norm tolerance and cache. Contact branches are used as local reconstructions;
no assertion of global inf-convolution completeness is needed or made here.
"""
from fractions import Fraction as F
import bisect, time
import numpy as np
from flint import arb
from interval_utils import ball, interval, upper_record, lower_record
from asymmetric_input import cost, cost_arb
from automatic_branches import discover, discover_float, float_envelope, BranchFailure
from certify_automatic_branches import continuity
from direct_branch_polynomials import Polynomial

WIDTHS=(2,4,8,12,16)
SCALES=(1,2,3,4,6,8,10)

def frac(record):return F(record['numerator'],record['denominator'])

class Tapered:
    def __init__(self,branch,spline,a,b,cell):
        self.branch=branch;self.spline=spline;self.a=a;self.b=b;self.cell=cell
    def jet(self,left,right,part,kind=None):
        v,p,H,_=self.branch.jet(left,right,part,'corrected')
        x=interval(left,right);u,du,Hu,_=self.spline.jet(self.cell,x)
        t=(x-ball(self.a))/ball(self.b-self.a);s=1-t
        theta=16*t*t*s*s
        dt=32*t*s*(1-2*t)/ball(self.b-self.a)
        ddt=32*(1-6*t+6*t*t)/ball((self.b-self.a)**2)
        return u+theta*(v-u),du+dt*(v-u)+theta*(p-du),Hu+ddt*(v-u)+2*dt*(p-du)+theta*(H-Hu),x

class Context:
    def __init__(self,spline):
        self.s=spline;self.input_cache={};self.identity_cache={};self.geometry={};self.float_geometry={}
        self.cache_hits=0;self.enclosures=0
    def data(self,a,b):
        key=(a,b)
        if key in self.input_cache:self.cache_hits+=1;return self.input_cache[key]
        m=(a+b)/2;x=interval(a,b)
        out=(ball(self.s.point(m)),self.s.range(a,b,1),cost_arb(ball(m),self.s.case),cost_arb(x,self.s.case,1))
        self.input_cache[key]=out;return out
    def contacts(self,scale):
        if scale not in self.geometry:
            try:
                bs,periodic,_=discover(self.s,F(scale)*self.s.h)
                proof=continuity(bs,'corrected',periodic)
                # Shifted copies permit a patch crossing the periodic boundary.
                cover=[b.translated(k) for b in bs for k in (-1,0,1)]
                seen=set();unique=[]
                for b in cover:
                    key=(b.number,b.left,b.right)
                    if key not in seen:seen.add(key);unique.append(b)
                self.geometry[scale]=(unique,proof)
            except (BranchFailure,ArithmeticError) as e:self.geometry[scale]=str(e)
        out=self.geometry[scale]
        if isinstance(out,str):raise BranchFailure(out)
        return out
    def float_contacts(self,scale):
        if scale not in self.float_geometry:self.float_geometry[scale]=discover_float(self.s,scale*float(self.s.h))
        return self.float_geometry[scale]
    def enclose(self,a,b,entries):
        identity=len(entries)==1 and getattr(entries[0][0],'identical_input',False)
        if identity and (a,b) in self.identity_cache:
            self.cache_hits+=1;return self.identity_cache[a,b]
        self.enclosures+=1;m=(a+b)/2;rad=interval(-(b-a)/2,(b-a)/2)
        jets=[c.jet(a,b,p) for c,p in entries];centers=[c.jet(m,m,p) for c,p in entries]
        active=[]
        for k,v in enumerate(jets):
            if not any(k!=j and centers[k][0]-centers[j][0]+rad*(v[1]-w[1])>0 for j,w in enumerate(jets)):active.append(k)
        if not active:raise ArithmeticError('Empty active set.')
        u0,du,g0,dg=self.data(a,b);ru=arb(0);dd=arb(0)
        value=centers[0][0]
        for v in centers[1:]:value=value.min(v[0])
        dl=abs(value-u0).abs_lower();rl=arb(0)
        for k in active:
            v,p,H,_=jets[k];vm,pm,_,_=centers[k]
            sg=arb(1) if p>=0 else (arb(-1) if p<=0 else interval(F(-1),F(1)))
            r0=vm+abs(pm)-g0
            ru=ru.max(abs(r0+rad*(p+sg*H-dg)).upper())
            if not identity:dd=dd.max(abs(vm-u0+rad*(p-du)).upper())
            if all(k==j or centers[k][0]<w[0] for j,w in enumerate(centers)):rl=rl.max(abs(r0).abs_lower())
        out=(ru,dd,rl,dl,len(active))
        if identity:self.identity_cache[a,b]=out
        return out

def identity_piece(s,i):
    return Polynomial(i*s.h,[s.coeffs[i%s.n][k]/s.h**k for k in range(4)],i*s.h,(i+1)*s.h,True)

def patches(s):
    v=s.record['numerical_node_numerators'];n=s.n
    peaks=[i for i in range(n) if v[i]>v[(i-1)%n] and v[i]>=v[(i+1)%n]]
    out=[]
    for k,i in enumerate(peaks):
        gap=min((i-peaks[k-1])%n or n,(peaks[(k+1)%len(peaks)]-i)%n or n)
        width=min(16,(gap-1)//2)
        if width>=2:out.append((i,width))
    return out

def build_patch(ctx,peak,cap,candidate):
    s=ctx.s;h=s.h;width=candidate['width'];a=(peak-width)*h;b=(peak+width)*h
    outer_a=(peak-cap)*h;outer_b=(peak+cap)*h;kind=candidate['kind']
    if kind=='polynomial':
        coeff=[]
        for x,i in [(a,peak-width-1),(b,peak+width)]:
            t=x/h-i
            coeff.append([s.local_exact(i,t,k)/[1,1,2,6][k] if k<=candidate['degree'] else F(0) for k in range(4)])
        l=Polynomial(a,coeff[0],a,b);r=Polynomial(b,coeff[1],a,b)
        if l.exact(a)>r.exact(a) or r.exact(b)>l.exact(b):raise BranchFailure('Polynomial endpoint dominance failed.')
        branches=None
    else:
        branches,_=ctx.contacts(candidate['scale'])
    edges={outer_a,outer_b,a,b}|{i*h for i in range(peak-cap,peak+cap+1)}
    if branches:edges|={e for c in branches for e in c.edges if a<e<b}
    edges=sorted(edges);pending=[]
    for lo,hi in zip(edges,edges[1:]):
        m=(lo+hi)/2;i=m//h
        if m<a or m>b:entries=[(identity_piece(s,i),0)]
        elif kind=='polynomial':entries=[(l,0),(r,0)]
        else:
            entries=[(Tapered(c,s,a,b,i),c.part(m)) for c in branches if c.left<=lo<hi<=c.right]
            if not entries:raise BranchFailure('Contact coverage failed on local patch.')
        pending.append((lo,hi,entries,0))
    return pending

def certify_region(ctx,pending,incumbent=None,max_evaluations=40000):
    started=time.perf_counter();rl=arb(0);dl=arb(0);ru=arb(0);du=arb(0);leaves=0;evaluations=0
    queue=[]
    # All seed midpoints are examined before norm subdivision.
    for a,b,entries,depth in pending:
        try:z=ctx.enclose(a,b,entries)
        except ArithmeticError:z=None
        if z is not None:rl=rl.max(z[2]);dl=dl.max(z[3])
        queue.append((a,b,entries,depth,z))
    while queue:
        if incumbent is not None and rl+dl>ball(incumbent):
            return dict(status='excluded',error_lower=lower_record(rl+dl),evaluations=evaluations,seconds=time.perf_counter()-started)
        a,b,entries,depth,z=queue.pop();evaluations+=1
        if z is None:
            try:z=ctx.enclose(a,b,entries)
            except ArithmeticError:z=None
        if z is not None:
            rl=rl.max(z[2]);dl=dl.max(z[3]);passed=z[0]<=rl*ball(F(51,50))+ball(F(1,10**10)) and z[1]<=dl*ball(F(51,50))+ball(F(1,10**10))
        else:passed=False
        if passed:ru=ru.max(z[0]);du=du.max(z[1]);leaves+=1
        else:
            if depth>=32 or evaluations>=max_evaluations:
                return dict(status='budget_exhausted',evaluations=evaluations,seconds=time.perf_counter()-started)
            m=(a+b)/2;queue.extend([(a,m,entries,depth+1,None),(m,b,entries,depth+1,None)])
    return dict(status='certified',residual_upper=upper_record(ru),distance_upper=upper_record(du),
                error_upper=upper_record(ru+du),residual_lower=lower_record(rl),distance_lower=lower_record(dl),
                evaluations=evaluations,leaves=leaves,seconds=time.perf_counter()-started)

def rank_patch(ctx,peak,cap,hybrid):
    s=ctx.s;h=float(s.h);y=(peak+np.linspace(-cap,cap,8*cap+1))*h
    u,up,_,_=s.float_jet(y);g=cost(y,s.case);out=[]
    for width in sorted(set(w for w in WIDTHS if w<=cap)|{cap}):
        a=(peak-width)*h;b=(peak+width)*h;valid=(y>=a)&(y<=b)
        for degree in (2,3):
            co=[]
            for x,i in [(F(peak-width)*s.h,peak-width-1),(F(peak+width)*s.h,peak+width)]:
                co.append([float(s.local_exact(i,x/s.h-i,k))/[1,1,2,6][k] if k<=degree else 0. for k in range(4)])
            def values(c,t):return c[0]+t*(c[1]+t*(c[2]+t*c[3])),c[1]+t*(2*c[2]+3*c[3]*t)
            if co[0][0]>values(co[1],a-b)[0] or co[1][0]>values(co[0],b-a)[0]:continue
            lv,lp=values(co[0],y[valid]-a);rv,rp=values(co[1],y[valid]-b)
            w=u.copy();p=up.copy();w[valid]=np.minimum(lv,rv);p[valid]=np.where(lv<=rv,lp,rp)
            out.append(dict(kind='polynomial',width=width,degree=degree,sampled=float(np.max(abs(w-u))+np.max(abs(w+abs(p)-g)))))
        if hybrid:
            for scale in SCALES:
                bs=ctx.float_contacts(scale)
                if not bs:continue
                v,p,_=float_envelope(bs,np.mod(y[valid],2),'corrected')
                if not np.all(np.isfinite(v)):continue
                t=(y[valid]-a)/(b-a);theta=16*t*t*(1-t)**2;dt=32*t*(1-t)*(1-2*t)/(b-a)
                w=u.copy();wp=up.copy();w[valid]=u[valid]+theta*(v-u[valid]);wp[valid]=up[valid]+dt*(v-u[valid])+theta*(p-up[valid])
                out.append(dict(kind='corrected',width=width,scale=scale,sampled=float(np.max(abs(w-u))+np.max(abs(w+abs(wp)-g)))))
    return sorted(out,key=lambda r:r['sampled'])

def reconstruct(spline,hybrid=True,attempts=3,selection=None,target=None):
    started=time.perf_counter();ctx=Context(spline);patch_list=patches(spline);chosen=[];history=[];occupied=set()
    for peak,cap in patch_list:occupied|={i%spline.n for i in range(peak-cap,peak+cap)}
    outside=[(i*spline.h,(i+1)*spline.h,[(identity_piece(spline,i),0)],0) for i in range(spline.n) if i not in occupied]
    outside_cert=certify_region(ctx,outside,incumbent=target)
    if outside_cert['status']!='certified':return dict(status='above_target' if outside_cert['status']=='excluded' else 'not_certified',outside=outside_cert,history=history,total_seconds=time.perf_counter()-started)
    for k,(peak,cap) in enumerate(patch_list):
        candidates=rank_patch(ctx,peak,cap,hybrid) if selection is None else [selection[k]]
        best=None;records=[];trial_list=list(candidates[:attempts]);fallback_used=False
        for candidate in trial_list:
            incumbent=target if best is None else frac(best['certificate']['error_upper'])
            if target is not None and incumbent is not None:incumbent=min(target,incumbent)
            try:r=certify_region(ctx,build_patch(ctx,peak,cap,candidate),incumbent)
            except (ArithmeticError,BranchFailure) as e:r=dict(status='not_certified',reason=str(e))
            row=dict(candidate=candidate,certificate=r);records.append(row)
            if r['status']=='certified' and (best is None or frac(r['error_upper'])<frac(best['certificate']['error_upper'])):best=row
            if best is None and hybrid and selection is None and candidate is trial_list[-1] and not fallback_used:
                # Contact proposals must not prevent access to the polynomial
                # control's candidates when the initial local search fails.
                fallback_used=True
                alternatives=[c for c in candidates if c['kind']=='polynomial'][:attempts]
                trial_list.extend(c for c in alternatives if c not in trial_list)
        history.append(dict(peak=peak,cap=cap,attempts=records,polynomial_fallback=fallback_used))
        if best is None:return dict(status='not_certified',history=history,total_seconds=time.perf_counter()-started)
        chosen.append(best)
    certs=[row['certificate'] for row in chosen]+[outside_cert]
    rho=max(frac(r['residual_upper']) for r in certs);dist=max(frac(r['distance_upper']) for r in certs)
    return dict(status='certified',kind='adaptive_hybrid' if hybrid else 'adaptive_polynomial',
        error_upper=upper_record(ball(rho+dist)),residual_upper=upper_record(ball(rho)),distance_upper=upper_record(ball(dist)),
        patches=len(patch_list),corrected_patches=sum(row['candidate']['kind']=='corrected' for row in chosen),
        selected=[row['candidate'] for row in chosen],history=history,outside=outside_cert,
        cache_hits=ctx.cache_hits,enclosures=ctx.enclosures,total_seconds=time.perf_counter()-started)
