"""Replay selected certificates, verify box coverage, and check independent jets."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
import mpmath as mp
from flint import ctx
import residual_refinement_2d as refinement
from residual_refinement_design import HELDOUT,install
from unknown_junction_2d import Input,load_branches,Verifier
import check_adaptive_extensions as independent

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('input_error_upper','reconstruction_error_upper','residual_upper','distance_upper','boundary_upper')

def fraction(r):return F(r['numerator'],r['denominator'])
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def basis_checks():
    count=0
    for degree in (2,3,4,5):
        ids=[(i,j) for i in range(degree) for j in range(degree-i)]
        cs=np.array([[(k+1)*(label+1)/128 for k in range(len(ids))] for label in range(4)])
        branches,_=refinement.rounded(cs,degree)
        for x,y in [(F(137,1000),F(-211,1000)),(F(-2,5),F(3,7))]:
            mx,my=independent.mq(x),independent.mq(y)
            mats,_=refinement.matrices(np.array([float(x)]),np.array([float(y)]),degree)
            for label,p in enumerate(branches):
                def exact(a,b):
                    distance=[1+a,1-a,1+b,1-b][label]
                    return distance*sum(mp.mpf(float(c))*mp.chebyt(i,a)*mp.chebyt(j,b) for c,(i,j) in zip(cs[label],ids))
                expected=[mp.diff(exact,(mx,my),(dx,dy)) for dx,dy in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]]
                jets=p.jet(independent.ball(x),independent.ball(y))
                assert all(independent.contains(a,b) for a,b in zip(jets,expected))
                for k in range(3):assert abs(float((mats[label][k]@cs[label])[0])-float(expected[k]))<1e-12
                count+=1
    return count

class CoverageVerifier(Verifier):
    last=None
    def __init__(self,*args):
        super().__init__(*args);self.partition={};self.children={};CoverageVerifier.last=self
    def box(self,box):
        xl,xr,yl,yr,i,j,d=box;h=self.inp.h
        if d:
            if d%2:
                width=2*(xr-xl);base=-1+i*h;left=base+((xl-base)//width)*width
                parent=(left,left+width,yl,yr,i,j,d-1)
            else:
                height=2*(yr-yl);base=-1+j*h;left=base+((yl-base)//height)*height
                parent=(xl,xr,left,left+height,i,j,d-1)
            self.children.setdefault(parent,[]).append(box)
            self.partition.pop(parent,None)
        assert box not in self.partition
        self.partition[box]=(xr-xl)*(yr-yl)
        return super().box(box)

def main():
    ctx.prec=192;mp.mp.dps=80;install()
    path=ROOT/'results/residual_refinement.json';r=json.loads(path.read_text())
    assert r['status']=='completed' and digest(path)==path.with_suffix('.sha256').read_text().strip()
    for name,sha in r['source_sha256'].items():assert digest(ROOT/'code'/name)==sha
    old_mp=independent.mp_data
    def mp_data(x,y,case):
        if case not in HELDOUT:return old_mp(x,y,case)
        def p(row):
            c=[mp.mpf(str(v)) for v in row]
            return c[0]+c[1]*x+c[2]*y+c[3]*x*y+c[4]*x*x+c[5]*y*y
        return tuple(p(row) for row in HELDOUT[case])
    independent.mp_data=mp_data;refinement.Verifier=CoverageVerifier
    inputs={};seen={};counts=dict(replayed=0,point_checks=0,C1_identities=0,coverage_leaves=0,target_decisions=0,
                                independent_basis_jets=basis_checks())
    for row in r['records']:
        selected=set()
        flat=[(i,j,a) for i,g in enumerate(row['history']) for j,a in enumerate(g['attempts'])]
        for target in r['design']['targets']:
            qualifying=[(i,j,a) for i,j,a in flat if fraction(a['certificate']['input_error_upper'])<=F(str(target))]
            saved=row['targets'][str(target)]
            if qualifying:
                i,j,a=qualifying[0]
                assert saved['status']=='target_certified'
                assert (i,j)==(saved['history_index'],saved['attempt_index'])
                assert saved['seconds']==a['elapsed_seconds'] and saved['n']==row['history'][i]['n']
                selected.add((i,j))
            else:
                assert saved['status']=='not_reached'
                if flat:
                    i,j,_=min(flat,key=lambda z:fraction(z[2]['certificate']['input_error_upper']))
                    selected.add((i,j))
            counts['target_decisions']+=1
        for i,j in sorted(selected):
            grid=row['history'][i];attempt=grid['attempts'][j];key=(row['case'],grid['n'])
            sig=key+(json.dumps(attempt['model']['branches'],sort_keys=True),)
            saved=tuple(fraction(attempt['certificate'][k]) for k in FIELDS)
            if sig in seen:assert saved==seen[sig];continue
            if key not in inputs:
                inp=Input(grid['n'],row['case'],grid['input'])
                assert Input(grid['n'],row['case']).record['arrays']==inp.record['arrays']
                counts['C1_identities']+=independent.check_input(inp);inputs[key]=inp
            inp=inputs[key];branches=load_branches(attempt['model'])
            # Exact zero on each branch's assigned boundary, after rounding.
            for label,p in enumerate(branches):
                c=p.rational;sg=F(-1 if label in (0,2) else 1)
                if label<2:assert all(sum(c[k][l]*sg**k for k in range(len(c)))==0 for l in range(len(c[0])))
                else:assert all(sum(row[l]*sg**l for l in range(len(row)))==0 for row in c)
            replay=refinement.certify_priority(inp,branches,budget=16000)
            assert tuple(fraction(replay[k]) for k in FIELDS)==saved,(key,row['method'])
            partition=CoverageVerifier.last.partition
            assert sum(partition.values())==4 and len(partition)==replay['leaves']
            assert all(a>=-1 and b<=1 and c>=-1 and d<=1 and a<b and c<d for a,b,c,d,*_ in partition)
            for parent,children in CoverageVerifier.last.children.items():
                assert len(children)==2
                a,b,c,d,*_=parent;one,two=sorted(children)
                assert sum((z[1]-z[0])*(z[3]-z[2]) for z in children)==(b-a)*(d-c)
                assert all(a<=z[0]<z[1]<=b and c<=z[2]<z[3]<=d for z in children)
                assert one[1]<=two[0] or one[3]<=two[2]
            counts['coverage_leaves']+=len(partition)
            counts['point_checks']+=independent.independent_2d(inp,branches)
            counts['replayed']+=1;seen[sig]=saved
            print('verified',row['case'],grid['n'],row['method'],flush=True)
    counts['distinct_inputs']=len(inputs)
    report=dict(status='PASS',data_sha256=digest(path),source_sha256=r['source_sha256'],**counts,
                scope='All target decisions, unique first-success certificates and best available unsuccessful-path certificates. Exact dyadic box partition coverage, Hermite continuity and assigned boundary identities, plus independent 80-digit point checks.')
    (ROOT/'verification/residual_refinement_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
