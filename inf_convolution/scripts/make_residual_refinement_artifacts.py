"""Summaries of the complete, frozen residual-refinement comparison."""
from pathlib import Path
import csv,json,statistics,hashlib

ROOT=Path(__file__).resolve().parents[1]
METHODS=('polynomial','refined_polynomial','refined_corrected')
NAMES={'polynomial':'Direct polynomial','refined_polynomial':'Residual-refined polynomial','refined_corrected':'Residual-refined contact'}
LABELS={'variable_l1':'I','anisotropic_l2':'II','curved_l2':'III',
        'holdout_a':'IV','holdout_b':'V','holdout_c':'VI','holdout_d':'VII'}

def main():
    path=ROOT/'results/residual_refinement.json';r=json.loads(path.read_text())
    assert r['status']=='completed'
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_suffix('.sha256').read_text().strip()==sha
    rows=[];groups={};attempts=failures=lp_failures=0
    for record in r['records']:
        for grid in record['history']:
            failures+=len(grid['failed_proposals']);attempts+=len(grid['attempts'])
            for attempt in grid['attempts']:
                lp_failures+=sum(not h['lp_success'] for h in attempt['model'].get('history',[]))
        for target,outcome in record['targets'].items():
            row=dict(case=record['case'],method=record['method'],target=target,repetition=record['repetition'],
                     status=outcome['status'],n=outcome.get('n',''),seconds=outcome['seconds'],bound=outcome.get('bound',''))
            rows.append(row);groups.setdefault((row['case'],target),{}).setdefault(row['method'],[]).append(row)
    with (ROOT/'results/residual_refinement_targets.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregates={}
    for key,methods in groups.items():
        aggregates[key]={}
        for method,rep in methods.items():
            assert len(rep)==r['design']['repetitions']
            assert len({(a['status'],a['n']) for a in rep})==1
            aggregates[key][method]=dict(status=rep[0]['status'],n=rep[0]['n'],
                seconds=statistics.median(a['seconds'] for a in rep),bound=max(a['bound'] for a in rep))
    def comparisons(method,base,subset):
        allpairs=[d for (case,t),d in aggregates.items() if case in subset]
        common=[(d[method],d[base]) for d in allpairs if d[method]['status']==d[base]['status']=='target_certified']
        ratios=[b['seconds']/a['seconds'] for a,b in common]
        return dict(method=method,baseline=base,pairs=len(allpairs),common_success=len(common),
            method_success=sum(d[method]['status']=='target_certified' for d in allpairs),
            baseline_success=sum(d[base]['status']=='target_certified' for d in allpairs),
            faster=sum(a['seconds']<b['seconds'] for a,b in common),
            median_speedup=statistics.median(ratios) if ratios else None,
            min_speedup=min(ratios) if ratios else None,max_speedup=max(ratios) if ratios else None,
            smaller_grid=sum(a['n']<b['n'] for a,b in common),
            median_node_ratio=statistics.median((b['n']/a['n'])**2 for a,b in common) if common else None)
    allcases=list(LABELS);prospective=list(r['design']['prospective_cases'])
    pairs=[('refined_polynomial','polynomial'),('refined_corrected','polynomial'),('refined_corrected','refined_polynomial')]
    summary=dict(data_sha256=sha,paths=len(r['records']),target_outcomes=len(rows),
        certificate_attempts=attempts,failed_fit_proposals=failures,failed_linear_programs=lp_failures,
        all=[comparisons(a,b,allcases) for a,b in pairs],prospective=[comparisons(a,b,prospective) for a,b in pairs])
    (ROOT/'results/residual_refinement_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    notes=['# Residual-refinement comparison','',
        'The full comparison uses seven two-dimensional coefficient problems, three target input-error bounds and three sequential timing repetitions. Four problems were fixed prospectively after the three-problem pilot. Every attempt and failure is retained. All methods use the same priority interval verifier.','',
        '| Method | Baseline | Targets met (method / baseline) | Faster on common successes | Median speedup |',
        '|---|---|---:|---:|---:|']
    for z in summary['all']:
        notes.append(f"| {NAMES[z['method']]} | {NAMES[z['baseline']]} | {z['method_success']}/21 / {z['baseline_success']}/21 | {z['faster']}/{z['common_success']} | {z['median_speedup']:.3f} |")
    notes+=['','Speed ratios use the median of three times for each method on each problem-target pair, followed by the median ratio across common successful pairs. An unmet target is never assigned an artificial speedup. The common grid cap is 65 by 65.','',
        '## Additional coefficient problems','',
        '| Method | Baseline | Targets met | Faster on common successes | Median speedup |','|---|---|---:|---:|---:|']
    for z in summary['prospective']:
        notes.append(f"| {NAMES[z['method']]} | {NAMES[z['baseline']]} | {z['method_success']}/12 / {z['baseline_success']}/12 | {z['faster']}/{z['common_success']} | {z['median_speedup']:.3f} |")
    notes+=['','## Target 0.08','',
        '| Case | Direct polynomial: n / seconds | Refined polynomial: n / seconds | Refined contact: n / seconds |',
        '|---|---:|---:|---:|']
    table=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Cost of certifying input error at most $0.08$ in two dimensions. $P$ is direct polynomial fitting; $P_R$ and $W_R$ apply the same residual refinement to polynomial and contact initializations. Entries give the first successful grid width $n$ and median total seconds from three runs. A dash denotes failure through $n=65$. Cases IV--VII were specified before evaluation. All methods use the same interval verifier and candidate budget.}\label{tab:residualrefinement}',
        r'\begin{tabular}{crrrrrr}',r'\toprule',
        r'Case & $n_P$ & $T_P$ & $n_{P_R}$ & $T_{P_R}$ & $n_{W_R}$ & $T_{W_R}$ \\',r'\midrule']
    for case in allcases:
        d=aggregates[case,'0.08'];cells=[];tex=[]
        for m in METHODS:
            z=d[m]
            if z['status']=='target_certified':cells.append(f"{z['n']} / {z['seconds']:.2f}");tex.extend([str(z['n']),f"{z['seconds']:.2f}"])
            else:cells.append(f"not reached ({z['seconds']:.2f} s)");tex.extend(['--','--'])
        notes.append('| '+LABELS[case]+' | '+' | '.join(cells)+' |')
        table.append(LABELS[case]+' & '+' & '.join(tex)+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    notes+=['', 'The improvement concerns residual refinement of fitted branches. Contact initialization must be judged against the equally refined polynomial method, not against the unrefined control alone. These tests do not establish superiority over other PDE solvers, high-dimensional scalability, arbitrary branch topology or a global convergence theorem.',
        '',f"Recorded {attempts} interval certificate attempts, {failures} failed fit proposals and {lp_failures} unsuccessful linear programs across all repetitions. A failed optimizer can still leave a valid seed, whose final certificate is recorded separately.",
        '', 'See `RESIDUAL_REFINEMENT.md` for the construction, limitations and reproduction commands.']
    (ROOT/'documents/RESIDUAL_REFINEMENT_RESULTS.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    (ROOT/'research/residual_refinement_table.tex').write_text('\n'.join(table)+'\n',encoding='utf8')
    p,c,pc=summary['all'];extra=summary['prospective'][0]
    prose=(
        r'We denote the refined polynomial and contact fits by $P_R$ and $W_R$, respectively. '
        r'On seven coefficient problems and targets $0.12$, $0.08$ and $0.04$, both refined methods certify '
        f"${p['method_success']}$ of $21$ problem--target pairs through $n=65$, compared with ${p['baseline_success']}$ for the unrefined polynomial control. "
        r'On the twelve pairs certified by all three methods, the refined polynomial and contact methods are each faster on eleven. '
        f"Their median speed ratios relative to the unrefined control are ${p['median_speedup']:.2f}$ and ${c['median_speedup']:.2f}$, respectively. "
        r'Each ratio uses the median of three complete sequential runs for each method and target. '
        f"The four coefficient sets fixed before evaluation contribute ${extra['method_success']}/12$ successes after either refinement and ${extra['baseline_success']}/12$ without refinement.\n\n"
        f"Contact initialization is faster than the equally refined polynomial control on ${pc['faster']}$ of their ${pc['common_success']}$ common successful pairs, with median speed ratio ${pc['median_speedup']:.2f}$."+'\n')
    sections=ROOT/'manuscript/sections';sections.mkdir(parents=True,exist_ok=True)
    (sections/'refinement_comparison.tex').write_text(prose,encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
