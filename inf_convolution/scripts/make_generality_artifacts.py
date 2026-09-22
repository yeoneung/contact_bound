"""Generate the stronger-control and varying-geometry summaries from saved data."""
from pathlib import Path
import csv,hashlib,json,statistics

ROOT=Path(__file__).resolve().parents[1]
LABELS={'variable_l1':'I','anisotropic_l2':'II','curved_l2':'III','holdout_a':'IV','holdout_b':'V','holdout_c':'VI','holdout_d':'VII'}
def up4(r):return f"{((r['numerator']*10000+r['denominator']-1)//r['denominator'])/10000:.4f}"
def load(name):
    p=ROOT/'results'/name;r=json.loads(p.read_text());assert r['status']=='completed'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==p.with_suffix('.sha256').read_text().strip()
    return r
def write_csv(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    strong=load('strong_refinement.json');free=load('free_branches.json');groups={};rows=[]
    for r in strong['records']:groups.setdefault((r['case'],r['n'],r['kind']),[]).append(r)
    for case in strong['design']['cases']:
      for n in strong['design']['grids']:
        row=dict(case=case,n=n)
        for kind in strong['design']['kinds']:
            group=groups[case,n,kind];assert len(group)==strong['design']['repetitions']
            bounds=[r['selected']['certificate']['input_error_upper']['decimal'] for r in group];assert max(bounds)==min(bounds)
            row[kind+'_bound']=bounds[0];row[kind+'_seconds']=statistics.median(r['total_seconds'] for r in group)
            row[kind+'_bound_up4']=up4(group[0]['selected']['certificate']['input_error_upper'])
            row[kind+'_top3']=group[0]['top3_bound']['decimal']
        row['speed_ratio']=row['polynomial_seconds']/row['corrected_seconds'];rows.append(row)
    write_csv('strong_refinement_summary.csv',rows)
    target_rows=[];target_ratios=[]
    for case in strong['design']['cases']:
      for n in strong['design']['grids']:
       for target in ('0.12','0.08','0.04'):
        row=dict(case=case,n=n,target=target)
        for kind in strong['design']['kinds']:
            group=groups[case,n,kind];success=[r['targets'].get(target) for r in group]
            assert all(v is not None for v in success) or all(v is None for v in success)
            row[kind+'_seconds']=statistics.median(v['seconds'] for v in success) if success[0] else ''
        if all(row[k+'_seconds']!='' for k in strong['design']['kinds']):
            row['speed_ratio']=row['polynomial_seconds']/row['corrected_seconds'];target_ratios.append(row['speed_ratio'])
        else:row['speed_ratio']=''
        target_rows.append(row)
    write_csv('strong_refinement_targets.csv',target_rows)
    improved=sum(r[k+'_bound']<r[k+'_top3'] for r in rows for k in strong['design']['kinds'])
    summary=dict(paths=len(strong['records']),problem_inputs=len(rows),
        corrected_smaller_bound=sum(r['corrected_bound']<r['polynomial_bound'] for r in rows),
        corrected_faster=sum(r['speed_ratio']>1 for r in rows),median_speed_ratio=statistics.median(r['speed_ratio'] for r in rows),
        common_fixed_input_targets=len(target_ratios),corrected_faster_matched_targets=sum(v>1 for v in target_ratios),
        matched_target_median_speed_ratio=statistics.median(target_ratios),
        complete_search_improves_top3=improved,largest_top3_ratio=max(r[k+'_top3']/r[k+'_bound'] for r in rows for k in strong['design']['kinds']),
        attempts=sum(len(r['attempts']) for r in strong['records']),exclusions=sum(a['status']=='excluded' for r in strong['records'] for a in r['attempts']),
        failures=sum(len(r['failed_proposals']) for r in strong['records']))
    (ROOT/'results/strong_refinement_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Complete searches at $n=33$ on the seven coefficient problems. Each degree--scale pair is refined from two starts before ranking. $B$ is the input-error upper bound, rounded upward; $T$ is the median time to complete the search, in seconds, from three sequential runs. Both initializations use the same refinement and verifier.}\label{tab:strongcontrol}',
        r'\medskip',r'\begin{tabular}{crrrr}',r'\toprule',r'Case & $B_{P_R}$ & $B_{W_R}$ & $T_{P_R}$ & $T_{W_R}$ \\',r'\midrule']
    for r in rows:table.append(LABELS[r['case']]+f" & {r['polynomial_bound_up4']} & {r['corrected_bound_up4']} & {r['polynomial_seconds']:.2f} & {r['corrected_seconds']:.2f}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'manuscript/tables/strong_control.tex').write_text('\n'.join(table)+'\n')
    frows=[]
    for row in free['records']:
        a=row['attempts'][row['selected_index']];c=a['certificate']
        frows.append(dict(case=row['case'],dimension=row['input']['dimension'],n=row['n'],kind=row['kind'],degree=a['degree'],scale=a['scale'],
            branches=a['model']['retained_branches'],residual=c['residual']['decimal'],distance=c['distance']['decimal'],boundary=c['boundary']['decimal'],bound=c['input_error_upper']['decimal'],
            bound_up4=up4(c['input_error_upper']),proposals=len(row['attempts']),failed_proposals=sum(a['status']!='certified' for a in row['attempts'])))
    write_csv('free_branches_summary.csv',frows)
    ftable=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Certification with branch counts determined from the numerical input. Each initialization selects a reconstruction with $K$ branches and input-error upper bound $B$, rounded upward. All successful proposals are refined and verified. These are development cases.}\label{tab:freebranches}',
        r'\medskip',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Domain & $d$ & $n$ & $K_P$ & $K_W$ & $B_{P_R}$ & $B_{W_R}$ \\',r'\midrule']
    names={'pentagon':'Pentagon','heptagon':'Heptagon','coupled_3d':'Coupled cube'}
    for case in free['design']['cases']:
        p=next(r for r in frows if r['case']==case and r['kind']=='polynomial');w=next(r for r in frows if r['case']==case and r['kind']=='corrected')
        ftable.append(names[case]+f" & {p['dimension']} & {p['n']} & {p['branches']} & {w['branches']} & {p['bound_up4']} & {w['bound_up4']}"+r' \\')
    ftable += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'manuscript/tables/free_branches.tex').write_text('\n'.join(ftable)+'\n')
    prose=(
        r'Refining every candidate before ranking tests whether selecting only three candidates affects the comparison. At $n=33$, each of the twelve degree--scale pairs receives two starts: the fit and a common seeded perturbation of its boundary-factored coefficients. Both initializations receive up to eight refinement iterations per start. '
        r'Refined sampled scores determine verification order for all $24$ candidates per initialization. A pointwise Arb lower bound for $D+\max(R,b_\partial)$ can exclude a candidate only if it exceeds the best certified upper bound. Otherwise the same full-domain verifier is run. Exact duplicate coefficient vectors reuse their certificate. The elapsed time includes every proposal, refinement, exclusion and verification.'+'\n\n'+
        r'\input{tables/strong_control}'+'\n\n'+
        f"Complete search improves the bound beyond the first three refined candidates on ${improved}$ of the $14$ input--initialization pairs. "
        f"Contact initialization gives the smaller selected bound on ${summary['corrected_smaller_bound']}$ of seven inputs and completes the search faster on ${summary['corrected_faster']}$, with median time ratio ${summary['median_speed_ratio']:.2f}$. "
        f"For targets $0.12$, $0.08$ and $0.04$ reached by both methods on this fixed input, it is faster on ${summary['corrected_faster_matched_targets']}/{summary['common_fixed_input_targets']}$ pairs, with median time ratio ${summary['matched_target_median_speed_ratio']:.2f}$. "
        r'Table~\ref{tab:strongcontrol} reports the time to complete the search; target times stop at the first certified success, after all proposals have been refined. Both time ratios use polynomial time divided by contact time. These comparisons use the existing coefficient cases on a fixed input grid.'+'\n')
    (ROOT/'manuscript/sections/generality_comparison.tex').write_text(prose)
    notes=['# Stronger control and automatic branch counts','',
        'Both studies retain failed proposals and compare the same certified continuum input-error quantity. The first uses seven existing coefficient cases and a fixed input grid, not unseen problems or adaptive-grid target timings. The second consists of development problems; it supports feasibility across the implemented geometries, not general superiority.','',
        f"Complete search: {summary['corrected_smaller_bound']}/7 smaller bounds and {summary['corrected_faster']}/7 faster complete searches for contact initialization; median time ratio {summary['median_speed_ratio']:.3f}. The complete search improves on the first three refined candidates in {improved}/14 cases.",
        f"Matched fixed-input targets: contact initialization is faster on {summary['corrected_faster_matched_targets']}/{summary['common_fixed_input_targets']} common successes, median time ratio {summary['matched_target_median_speed_ratio']:.3f}. All proposal/refinement costs precede the first certificate and are included.",
        '', '| Case | Initialization | Dimension / grid | Detected K | Certified input error |','|---|---|---:|---:|---:|']
    for r in frows:notes.append(f"| {r['case']} | {r['kind']} | {r['dimension']} / {r['n']} | {r['branches']} | {r['bound']:.8f} |")
    notes += ['','Bounds are absolute uniform-error bounds, not measured errors or relative error percentages. Rounding for this table does not replace the rational endpoints in the data.',
              '', 'See `GENERALITY_STUDY.md` for the algorithms, scope, failures, and reproduction commands.']
    (ROOT/'documents/GENERALITY_RESULTS.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2));print(json.dumps(frows,indent=2))
if __name__=='__main__':main()
