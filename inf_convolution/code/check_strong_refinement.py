"""Replay complete-search exclusions and selected/first-target certificates."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
import mpmath as mp
from flint import ctx
import residual_refinement_2d as refinement
from residual_refinement_design import install,HELDOUT
from unknown_junction_2d import Input,load_branches
from strong_refinement_search import point_lower
from check_residual_refinement import CoverageVerifier,FIELDS,fraction,digest
import check_adaptive_extensions as independent

ROOT=Path(__file__).resolve().parents[1]
def main():
    ctx.prec=192;mp.mp.dps=80;install();path=ROOT/'results/strong_refinement.json';data=json.loads(path.read_text())
    assert data['status']=='completed' and digest(path)==path.with_suffix('.sha256').read_text().strip()
    for name,sha in data['source_sha256'].items():assert digest(ROOT/'code'/name)==sha
    old=independent.mp_data
    def mp_data(x,y,case):
        if case not in HELDOUT:return old(x,y,case)
        return tuple(sum(mp.mpf(str(c))*v for c,v in zip(row,[1,x,y,x*y,x*x,y*y])) for row in HELDOUT[case])
    independent.mp_data=mp_data;refinement.Verifier=CoverageVerifier
    inputs={};replayed={};lowers={};counts=dict(paths=0,selected_certificates_replayed=0,exclusions_replayed=0,
        C1_identities=0,independent_points=0,coverage_leaves=0,target_decisions=0)
    repeated={}
    for row in data['records']:
        key=(row['case'],row['n']);pathkey=key+(row['kind'],)
        if key not in inputs:
            inp=Input(row['n'],row['case'],row['input']);assert Input(row['n'],row['case']).record['arrays']==inp.record['arrays']
            inputs[key]=inp;counts['C1_identities']+=independent.check_input(inp)
        inp=inputs[key];assert inp.record['arrays']==row['input']['arrays']
        incumbent=None;selected=[];targets={};deterministic=[]
        for rank,a in enumerate(row['attempts']):
            assert a['rank']==rank
            sig=key+(json.dumps(a['candidate']['model']['branches'],sort_keys=True),)
            if sig not in lowers:lowers[sig]=point_lower(inp,load_branches(a['candidate']['model']))
            assert lowers[sig]==a['lower']
            if a['status']=='excluded':
                assert incumbent and fraction(a['incumbent'])==fraction(incumbent['certificate']['input_error_upper'])
                assert fraction(a['lower']['bound'])>fraction(a['incumbent']);counts['exclusions_replayed']+=1
            else:
                assert a['status']=='certified'
                if incumbent is None or fraction(a['certificate']['input_error_upper'])<fraction(incumbent['certificate']['input_error_upper']):incumbent=a
                for target in (.12,.08,.04):
                    if str(target) not in targets and fraction(a['certificate']['input_error_upper'])<=F(str(target)):
                        targets[str(target)]=dict(rank=rank,seconds=a['elapsed_seconds'],bound=a['certificate']['input_error_upper']['decimal']);selected.append(a)
            if rank==2:assert row['top3_bound']==incumbent['certificate']['input_error_upper']
            deterministic.append((a['status'],a['candidate']['degree'],a['candidate']['scale'],a['candidate']['restart'],
                                  a.get('certificate',{}).get('input_error_upper'),a['lower']))
        assert row['selected']==incumbent and row['targets']==targets;selected.append(incumbent)
        if pathkey in repeated:assert deterministic==repeated[pathkey]
        else:repeated[pathkey]=deterministic
        counts['target_decisions']+=3
        for a in selected:
            sig=key+(json.dumps(a['candidate']['model']['branches'],sort_keys=True),)
            saved=tuple(fraction(a['certificate'][f]) for f in FIELDS)
            if sig in replayed:assert replayed[sig]==saved;continue
            branches=load_branches(a['candidate']['model']);cert=refinement.certify_priority(inp,branches,budget=16000)
            assert tuple(fraction(cert[f]) for f in FIELDS)==saved
            coverage=CoverageVerifier.last;assert sum(coverage.partition.values())==4 and len(coverage.partition)==cert['leaves']
            for parent,children in coverage.children.items():
                assert len(children)==2
                xl,xr,yl,yr,*_=parent;one,two=sorted(children)
                assert sum((z[1]-z[0])*(z[3]-z[2]) for z in children)==(xr-xl)*(yr-yl)
                assert all(xl<=z[0]<z[1]<=xr and yl<=z[2]<z[3]<=yr for z in children)
                assert one[1]<=two[0] or one[3]<=two[2]
            counts['coverage_leaves']+=len(coverage.partition)
            counts['independent_points']+=independent.independent_2d(inp,branches)
            counts['selected_certificates_replayed']+=1;replayed[sig]=saved
        counts['paths']+=1;print('verified',row['repetition'],*pathkey,flush=True)
    counts['unique_lower_checks']=len(lowers)
    report=dict(status='PASS',data_sha256=digest(path),**counts,
        scope='All saved lower bounds and exclusion/selection/target decisions; unique selected and first-target certificates replayed with full domain coverage; all input C1 identities and supplementary independent point checks.')
    (ROOT/'verification/strong_refinement_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
