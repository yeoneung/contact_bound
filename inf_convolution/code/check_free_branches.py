"""Exact coverage/C1 checks, certificate replay and independent 80-digit checks."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import factorial,comb
import json,hashlib
import numpy as np
import mpmath as mp
from flint import arb,ctx
from interval_utils import ball,upper_record
from free_branch_problem import Problem,Input,Polynomial
from free_branch_verify import Verifier,maximum
from free_branch_bernstein import power_to_bernstein,restrict

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fraction(r):return F(r['numerator'],r['denominator'])
def mq(x):
    q=F(x);return mp.mpf(q.numerator)/q.denominator
def contains(a,x):
    lo=a.lower().fmpq();hi=a.upper().fmpq()
    return mq(F(int(lo.numerator),int(lo.denominator)))<=x<=mq(F(int(hi.numerator),int(hi.denominator)))


def polynomial(c,x,nu=None):
    nu=nu or (0,)*len(x);out=mp.mpf(0)
    for index in np.ndindex(c.shape):
        if all(k>=r for k,r in zip(index,nu)):
            out+=mq(c[index])*mp.fprod(factorial(k)//factorial(k-r)*v**(k-r) for k,r,v in zip(index,nu,x))
    return out


def bernstein_checks():
    count=0
    for d in (2,3):
        c=np.array([F((17*i)%23-11,64) for i in range(4**d)],dtype=object).reshape((4,)*d)
        lo=[F(-3,4)]*d;hi=[F(2,5)]*d;b=power_to_bernstein(c,lo,hi)
        aa=[F(1,4)]*d;bb=[F(3,4)]*d;restricted=restrict(b,aa,bb)
        p=Polynomial(c)
        for t in product((F(0),F(1,3),F(1)),repeat=d):
            x=[a+(z-a)*v for a,z,v in zip(lo,hi,t)];mx=[mq(v) for v in x]
            bv=sum(b[index]*ball(F(np.prod([comb(3,k)*u**k*(1-u)**(3-k) for k,u in zip(index,t)],dtype=object))) for index in np.ndindex(b.shape))
            assert contains(bv,polynomial(c,mx))
            s=[a+(z-a)*v for a,z,v in zip(aa,bb,t)];xx=[a+(z-a)*v for a,z,v in zip(lo,hi,s)]
            rv=sum(restricted[index]*ball(F(np.prod([comb(3,k)*u**k*(1-u)**(3-k) for k,u in zip(index,t)],dtype=object))) for index in np.ndindex(b.shape))
            assert contains(rv,polynomial(c,[mq(v) for v in xx]))
            jets=p.jet([ball(v) for v in x]);orders=[(0,)*d]
            for j in range(d):nu=[0]*d;nu[j]=1;orders.append(tuple(nu))
            for i in range(d):
                for j in range(d):nu=[0]*d;nu[i]+=1;nu[j]+=1;orders.append(tuple(nu))
            for a,nu in zip([jets[0],*jets[1],*(v for row in jets[2] for v in row)],orders):assert contains(a,polynomial(c,mx,nu))
            count+=1
    return count


def c1_checks(inp):
    count=0
    for cell in product(range(inp.n-1),repeat=inp.d):
      left=inp.coefficients(cell).c
      for axis in range(inp.d):
        if cell[axis]>=inp.n-2:continue
        neighbor=list(cell);neighbor[axis]+=1;right=inp.coefficients(tuple(neighbor)).c
        for tangential in product(range(4),repeat=inp.d-1):
          for order in (0,1):
            values=[]
            for k in range(order,4):
                index=list(tangential);index.insert(axis,k)
                values.append(left[tuple(index)]*F(factorial(k),factorial(k-order)))
            index=list(tangential);index.insert(axis,order)
            assert sum(values)==right[tuple(index)]*factorial(order);count+=1
    return count


def partition(records,lo,hi,depth=0):
    assert records
    if len(records)==1 and records[0][0]==lo and records[0][1]==hi:
        assert records[0][2]==depth;return 1
    axis=max(range(len(lo)),key=lambda j:hi[j]-lo[j]);mid=(lo[axis]+hi[axis])/2
    left=[r for r in records if r[1][axis]<=mid];right=[r for r in records if r[0][axis]>=mid]
    assert len(left)+len(right)==len(records)
    lh=list(hi);lh[axis]=mid;rl=list(lo);rl[axis]=mid
    return partition(left,lo,tuple(lh),depth+1)+partition(right,tuple(rl),hi,depth+1)


def point_checks(inp,branches,cert):
    rng=np.random.default_rng(817);count=0
    for _ in range(80):
        point=[F(int(v),128) for v in rng.integers(-128,129,size=inp.d)]
        if inp.problem.classify(point,point)<0:continue
        cell=tuple(min(inp.n-2,int((v+1)//inp.h)) for v in point);x=[mq(v) for v in point]
        vals=[polynomial(p.c,x) for p in branches];k=min(range(len(vals)),key=lambda k:vals[k]);z=vals[k]
        gradient=[]
        for j in range(inp.d):nu=[0]*inp.d;nu[j]=1;gradient.append(polynomial(branches[k].c,x,nu))
        definition=inp.problem.definition
        if 'velocities' in definition:H=max(sum(mp.mpf(str(a))*b for a,b in zip(v,gradient)) for v in definition['velocities'])
        else:
            H=mp.sqrt(sum(sum(mp.mpf(str(a))*b for a,b in zip(row,gradient))**2 for row in definition['M']))
            H+=sum(mp.mpf(str(a))*b for a,b in zip(definition['drift'],gradient))
        g=sum(mp.mpf(str(v))*mp.fprod(a**k for a,k in zip(x,e)) for e,v in definition['g'])
        local=[(mq(v)-mq(-1+i*inp.h))/mq(inp.h) for v,i in zip(point,cell)]
        u=polynomial(inp.coefficients(cell).c,local)
        assert abs(z+H-g)<=mq(fraction(cert['residual']))
        assert abs(z-u)<=mq(fraction(cert['distance']))
        for p in branches:
            jets=p.jet([ball(v) for v in point]);assert contains(jets[0],polynomial(p.c,x))
            for j in range(inp.d):nu=[0]*inp.d;nu[j]=1;assert contains(jets[1][j],polynomial(p.c,x,nu))
        count+=1
    return count


def replay(inp,branches,cert):
    ctx.prec=cert['precision'];ver=Verifier(inp,branches);groups={};R=arb(0);D=arb(0)
    for record in cert['leaves']+cert['outside']:
        cell=tuple(record['cell']);lo=tuple(F(x) for x in record['lo']);hi=tuple(F(x) for x in record['hi'])
        assert all(-1+i*inp.h<=a<b<=-1+(i+1)*inp.h for i,a,b in zip(cell,lo,hi))
        groups.setdefault(cell,[]).append((lo,hi,record['depth']))
    assert len(groups)==(inp.n-1)**inp.d
    for cell,records in groups.items():
        lo=tuple(-1+i*inp.h for i in cell);hi=tuple(a+inp.h for a in lo);partition(records,lo,hi)
    for record in cert['outside']:assert inp.problem.classify([F(x) for x in record['lo']],[F(x) for x in record['hi']])<0
    for record in cert['leaves']:
        r,d,k=ver.box(tuple(record['cell']),[F(x) for x in record['lo']],[F(x) for x in record['hi']]);R=R.max(r);D=D.max(d)
    groups={};b=arb(0)
    for record in cert['boundary_leaves']:
        face=record['face'];lo=tuple(F(x) for x in record['lo']);hi=tuple(F(x) for x in record['hi'])
        groups.setdefault(face,[]).append((lo,hi,record['depth']));b=b.max(ver.boundary_box(face,lo,hi))
    assert len(groups)==len(inp.problem.faces)
    for records in groups.values():partition(records,(F(0),)*(inp.d-1),(F(1),)*(inp.d-1))
    for field,value in [('residual',R),('distance',D),('boundary',b),('reconstruction_error_upper',R.max(b)),('input_error_upper',D+R.max(b))]:
        assert fraction(upper_record(value))<=fraction(cert[field]),field
    return len(cert['leaves'])+len(cert['outside'])+len(cert['boundary_leaves'])


def main():
    ctx.prec=128;mp.mp.dps=80;path=ROOT/'results/free_branches.json';r=json.loads(path.read_text())
    assert r['status']=='completed' and digest(path)==path.with_suffix('.sha256').read_text().strip()
    for n,sha in r['source_sha256'].items():assert digest(ROOT/'code'/n)==sha
    counts=dict(bernstein_polynomial_points=bernstein_checks(),C1_identities=0,covered_leaves=0,independent_points=0,replayed_certificates=0,selection_decisions=0)
    inputs={}
    for row in r['records']:
        key=(row['case'],row['n'])
        if key not in inputs:
            inp=Input(Problem(row['case']),row['n'],row['input']);assert Input(inp.problem,inp.n).record['arrays']==inp.record['arrays']
            counts['C1_identities']+=c1_checks(inp);inputs[key]=inp
        inp=inputs[key];valid=[(i,a) for i,a in enumerate(row['attempts']) if a['status']=='certified']
        chosen=min(valid,key=lambda z:fraction(z[1]['certificate']['input_error_upper']))[0];assert chosen==row['selected_index']
        counts['selection_decisions']+=1;entry=row['attempts'][chosen];branches=[Polynomial.load(p) for p in entry['model']['branches']]
        counts['covered_leaves']+=replay(inp,branches,entry['certificate'])
        counts['independent_points']+=point_checks(inp,branches,entry['certificate']);counts['replayed_certificates']+=1
        print('verified',*key,row['kind'],len(branches),flush=True)
    report=dict(status='PASS',data_sha256=digest(path),**counts,
        scope='All method selections; each selected certificate fully replayed over exact covering partitions including cut cells and boundary patches; exact C1 interfaces; independent 80-digit polynomial/Bernstein and pointwise PDE checks. Point checks supplement, rather than replace, the domain certificates.')
    (ROOT/'verification/free_branch_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
