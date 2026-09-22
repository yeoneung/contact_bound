"""Tables and complete CSVs for fixed-input and search-policy comparisons."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import csv,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from robustness_design import NEW_CASES,ORIGINAL_CASES,GRIDS

def rat(r):return F(r['numerator'],r['denominator'])
def upward(q,digits=6):
    n=-((-q.numerator*10**digits)//q.denominator)
    return f'{n//10**digits}.{n%10**digits:0{digits}d}'
def write_csv(name,rows):
    with (ROOT/'results'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    data=json.loads((ROOT/'results/robustness.json').read_text());assert data['status']=='completed'
    records={(r['case'],r['n'],r['method'],r['scope']):r for r in data['records']}
    def row(case,n,method):
        scope='wide' if case in ORIGINAL_CASES and method!='polynomial' else 'standard'
        return records[case,n,method,scope]
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Certification of the same input spline. Each contact method examines 100 prescribed scales; the polynomial method examines 30 width--degree pairs. The smallest certified upper bounds are rounded upward. The last column gives their uncorrected-to-corrected ratio, computed before display rounding.}\label{tab:numericalselection}',
        r'\begin{tabular}{crrrrr}',r'\toprule',
        r'Cost & $n$ & Polynomial & Uncorrected & Corrected & $\widehat B_q/\widehat B_w$ \\',r'\midrule']
    fixed=[]
    for case,label in zip(ORIGINAL_CASES,'ABC'):
        for n in GRIDS:
            group={m:row(case,n,m) for m in ['polynomial','uncorrected','corrected']}
            bounds={m:rat(v['selected']['error_upper']) for m,v in group.items()}
            vals=[label,str(n)]+[upward(bounds[m]) for m in group]+[f"{float(bounds['uncorrected']/bounds['corrected']):.2f}"]
            lines.append(' & '.join(vals)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'manuscript/tables/numerical_selection.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{All twelve members of \eqref{eq:frequencyfamily}, on both grids. Each row summarizes four $(a,b)$ pairs. Counts compare the selected upper bounds. Time ratios are medians over four complete searches, including construction, with one sequential run per input and method.}\label{tab:robustness}',
        r'\begin{tabular}{rrrrrr}',r'\toprule',
        r'$m$ & $n$ & $\widehat B_w<\widehat B_q$ & $\widehat B_w<\widehat B_P$ & Median $\widehat B_w/\widehat B_q$ & Median $t_w/t_P$ \\',r'\midrule']
    summary=[]
    for frequency in (1,2,4):
        for n in GRIDS:
            cases=[c for c,v in NEW_CASES.items() if v['frequency']==frequency];ratios=[];times=[];qcount=pcount=0
            for case in cases:
                p,q,w=[row(case,n,m) for m in ['polynomial','uncorrected','corrected']]
                pb,qb,wb=[rat(z['selected']['error_upper']) for z in [p,q,w]]
                qcount+=wb<qb;pcount+=wb<pb;ratios.append(float(wb/qb));times.append(w['total_seconds']/p['total_seconds'])
            med=statistics.median(ratios);tm=statistics.median(times)
            lines.append(f'{frequency} & {n} & {qcount}/4 & {pcount}/4 & {med:.2f} & {tm:.2f}'+r' \\')
            summary.append(dict(frequency=frequency,n=n,inputs=4,corrected_smaller_than_uncorrected=qcount,
                corrected_smaller_than_polynomial=pcount,median_bound_ratio=med,median_time_ratio=tm))
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'manuscript/tables/robustness.tex').write_text('\n'.join(lines)+'\n')
    candidates=[]
    for r in data['records']:
        s=r['selected'];p=s['proposal'] if s else {};statuses=Counter(c['status'] for c in r['candidates'])
        fixed.append(dict(case=r['case'],n=r['n'],method=r['method'],scope=r['scope'],status=r['status'],
            epsilon=str(p.get('epsilon','')),width=p.get('width',''),degree=p.get('degree',''),
            residual_upper=s['residual_upper']['decimal'] if s else '',distance_upper=s['distance_upper']['decimal'] if s else '',
            bound_upper=s['error_upper']['decimal'] if s else '',construction_seconds=r['construction_seconds'],
            proposal_seconds=r['proposal_seconds'],total_seconds=r['total_seconds'],
            certified=statuses['certified'],pruned=statuses['above_target'],not_certified=statuses['not_certified']))
        for c in r['candidates']:
            p=c['proposal'];candidates.append(dict(case=r['case'],n=r['n'],method=r['method'],scope=r['scope'],rank=c['rank'],
                epsilon=str(p.get('epsilon','')),width=p.get('width',''),degree=p.get('degree',''),status=c['status'],
                residual_upper=c.get('residual_upper',{}).get('decimal',''),distance_upper=c.get('distance_upper',{}).get('decimal',''),
                bound_upper=c.get('error_upper',{}).get('decimal',''),bound_lower=c.get('error_lower',{}).get('decimal',''),
                cutoff=str(c['incumbent_cutoff']),seconds=c['total_seconds'],reason=c.get('reason','')))
    write_csv('robustness_selected.csv',fixed);write_csv('robustness_candidates.csv',candidates);write_csv('robustness_summary.csv',summary)
    write_csv('robustness_policy.csv',[dict(**{k:v for k,v in p.items() if k!='target'},target=float(F(*p['target']))) for p in data['policy']])
    print('Generated fixed-input and robustness tables and four complete CSV files.')

if __name__=='__main__':main()
