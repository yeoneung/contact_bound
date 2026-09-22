"""Generate manuscript artifacts only from the saved rational certificates."""
from pathlib import Path
from fractions import Fraction
import csv,json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def rational(record):
    return Fraction(record['numerator'],record['denominator'])


def directed(q, digits=3, upper=True):
    scale=10**digits
    n=(q*scale).numerator//(q*scale).denominator
    if upper and Fraction(n,scale)<q:n+=1
    return f'{n//scale}.{n%scale:0{digits}d}'


def main():
    (ROOT/'manuscript/tables').mkdir(parents=True,exist_ok=True)
    (ROOT/'figures').mkdir(parents=True,exist_ok=True)
    report=json.loads((ROOT/'results/reconstruction.json').read_text())
    rows=report['records'];sine=[r for r in rows if r['profile']=='sine']
    lines=[r'\begin{table}[t]',r'\centering',
           r'\caption{Sinusoidal profile. The contact column is a lower bound for the optimized effectivity index. All reconstruction columns are verified upper bounds; displayed decimals are rounded in the indicated direction.}\label{tab:reconstruction}',
           r'\begin{tabular}{rrrrrr}',r'\toprule',
           r'$\delta$ & Contact $\ge$ & $\widehat\rho/\delta\le$ & $\widehat d/\delta\le$ & $\widehat B/\delta\le$ & Cells \\',r'\midrule']
    export=[]
    for r in rows:
        delta=Fraction(1,r['delta_denominator'])
        entry=dict(profile=r['profile'],delta_power=r['delta_power'],
                   contact_effectivity_lower=float(rational(r['contact_effectivity_lower'])),
                   residual_effectivity_upper=float(rational(r['residual_upper'])/delta),
                   distance_effectivity_upper=float(rational(r['distance_upper'])/delta),
                   reconstruction_effectivity_upper=float(rational(r['reconstruction_effectivity_upper'])),
                   cells=r['cells'],elapsed_seconds=r['elapsed_seconds'])
        export.append(entry)
        if r['profile']=='sine':
            values=[f"$10^{{-{r['delta_power']}}}$",
                    directed(rational(r['contact_effectivity_lower']),upper=False),
                    directed(rational(r['residual_upper'])/delta),
                    directed(rational(r['distance_upper'])/delta),
                    directed(rational(r['reconstruction_effectivity_upper'])),str(r['cells'])]
            lines.append(' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'manuscript/tables/reconstruction.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (ROOT/'results/reconstruction.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.0,4.2),layout='constrained')
    xs=[10.**-r['delta_power'] for r in sine]
    ys=[float(rational(r['contact_effectivity_lower'])) for r in sine]
    bs=[float(rational(r['reconstruction_effectivity_upper'])) for r in sine]
    ax.loglog(xs,ys,'o--',color='#194f84',label='Contact estimate: proven lower bound',linewidth=1.8,markersize=5)
    ax.loglog(xs,bs,'s-',color='#bd5319',label='Reconstruction: verified upper bound',linewidth=1.8,markersize=5)
    ax.axhline(30,color='#747474',linestyle=':',linewidth=1.4,label='Reconstruction: uniform analytic bound')
    ax.set_xlim(1.7e-2,6e-9);ax.set_ylim(2,850)
    ax.set_xlabel(r'Actual uniform error $\delta$');ax.set_ylabel('Error bound / actual error')
    ax.grid(which='major',alpha=.16)
    ax.legend(loc='upper left',frameon=False,fontsize=9)
    fig.savefig(ROOT/'figures/effectivity.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(ROOT/'figures/effectivity.png',dpi=220)
    plt.close(fig)
    make_numerical_table()
    make_automatic_table()
    from make_robustness_artifacts import main as make_robustness
    make_robustness()
    make_twod_table()
    from make_extension_artifacts import main as make_extensions
    make_extensions()
    from make_residual_refinement_artifacts import main as make_refinement
    make_refinement()
    from make_generality_artifacts import main as make_generality
    make_generality()
    print('Generated six tables, two figures, and all comparison CSV files.')


def make_numerical_table():
    selection=json.loads((ROOT/'results/numerical_selection.json').read_text())
    audit=json.loads((ROOT/'results/numerical_reference_audit.json').read_text())
    lines=[r'\begin{table}[t]',r'\centering',
           r'\caption{Computed cubic splines. Scales are selected using only certified upper bounds. The error intervals and effectivity indices are evaluated afterward. Bounds and interval endpoints are rounded outward.}\label{tab:numericalselection}',
           r'\begin{tabular}{rrrrr}',r'\toprule',
           r'$N$ & $\eps_*$ & $10^3\widehat B_{\eps_*}$ & $10^3\norm{u_N-V}_\infty\in$ & Effectivity $\le$ \\',r'\midrule']
    records=[];candidates=[]
    for sel,ref in zip(selection['records'],audit['records']):
        assert sel['n']==ref['n'] and sel['selected_exponent']==ref['selected_exponent']
        err=ref['reference_error'];n=sel['n']
        vals=[str(n),f"$2^{{-{sel['selected_exponent']}}}$",
              directed(rational(sel['selected_error_upper'])*1000),
              '['+directed(rational(err['lower'])*1000,4,False)+', '+directed(rational(err['upper'])*1000,4)+']',
              directed(rational(ref['effectivity_upper']))]
        lines.append(' & '.join(vals)+r' \\')
        chosen=next(c for c in sel['candidates'] if c['exponent']==sel['selected_exponent'])
        records.append(dict(n=n,epsilon_exponent=sel['selected_exponent'],
                            selected_error_upper=float(rational(sel['selected_error_upper'])),
                            reference_error_lower=float(rational(err['lower'])),
                            reference_error_upper=float(rational(err['upper'])),
                            effectivity_upper=float(rational(ref['effectivity_upper'])),
                            selected_cells=chosen['leaves'],
                            certified_candidates=sum(c['status']=='certified' for c in sel['candidates']),
                            selection_seconds=sel['selection_seconds']))
        for c in sel['candidates']:
            candidates.append(dict(n=n,exponent=c['exponent'],status=c['status'],
                                   selected=c['exponent']==sel['selected_exponent'],
                                   certified_upper=c.get('error_upper',{}).get('decimal',''),
                                   reason=c.get('reason','')))
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    # Keep the old symmetric-input CSVs reproducible. The current Table 2 is
    # generated from the automatic asymmetric study below.
    for name,rows in [('numerical_selection.csv',records),('numerical_candidates.csv',candidates)]:
        with (ROOT/'results'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_automatic_table():
    data=json.loads((ROOT/'results/automatic_branches.json').read_text());groups={}
    for row in data['records']:groups.setdefault((row['case'],tuple(row['target']),row['method']),[]).append(row)
    lines=[r'\begin{table}[t]',r'\centering',
           r'\caption{Automatic certification of asymmetric inputs. Each entry is the selected number of grid points, followed by the median total time in seconds over three runs. Every method attains the stated error tolerance $\tau$. Times include all preceding grids, proposals and failed attempts.}\label{tab:numericalselection}',
           r'\begin{tabular}{ccrrr}',r'\toprule',
           r'Cost & $\tau$ & Polynomial & Uncorrected & Corrected \\',r'\midrule']
    summary=[];repeats=[];attempts=[]
    for case,label in [('skew','A'),('skew_strong','B'),('two_peaks','C')]:
        for target in [(1,50),(1,100)]:
            values=[label,f'{float(Fraction(*target)):.2f}']
            for method in ['polynomial','uncorrected','corrected']:
                rows=groups[case,target,method];first=rows[0]
                assert all(r['status']=='target_certified' and r['selected_n']==first['selected_n'] for r in rows)
                times=[r['total_seconds'] for r in rows];seconds=statistics.median(times)
                values.append(f"{first['selected_n']} ({seconds:.3f})")
                summary.append(dict(case=case,target=float(Fraction(*target)),method=method,n=first['selected_n'],
                                    bound_upper=first['selected']['error_upper']['decimal'],median_seconds=seconds,
                                    minimum_seconds=min(times),maximum_seconds=max(times)))
                for row in rows:
                    repeats.append(dict(case=case,target=float(Fraction(*target)),method=method,repetition=row['repetition'],n=row['selected_n'],total_seconds=row['total_seconds']))
                    for step in row['history']:
                        for c in step['attempts']:
                            attempts.append(dict(case=case,target=float(Fraction(*target)),method=method,repetition=row['repetition'],n=step['n'],
                                                 status=c['status'],epsilon=str(c.get('epsilon','')),width=c.get('width',''),degree=c.get('degree',''),
                                                 upper=c.get('error_upper',{}).get('decimal',''),lower=c.get('error_lower',{}).get('decimal',''),
                                                 seconds=c['total_seconds'],reason=c.get('reason','')))
            lines.append(' & '.join(values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'manuscript/tables/numerical_selection.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for name,rows in [('automatic_branches.csv',summary),('automatic_timings.csv',repeats),('automatic_attempts.csv',attempts)]:
        with (ROOT/'results'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_twod_table():
    data=json.loads((ROOT/'results/twod_method_comparison.json').read_text())
    audit=json.loads((ROOT/'verification/twod_method_checks.json').read_text())
    lines=[r'\begin{table}[t]',r'\centering',
           r'\caption{Coupled two-dimensional problem. Each row certifies the error of the same bilinear input $U_n$. The two convolution methods select their scales independently from the same eight candidates. All suprema use $1\%+10^{-8}$ tolerance. Error bounds and effectivity indices are rounded upward; time includes fitting and the complete search, including unsuccessful attempts.}\label{tab:twodreconstruction}',
           r'\begin{tabular}{rlrrrr}',r'\toprule',
           r'$n$ & Function & $\eps_*$ & Bound $\le$ & Effectivity $\le$ & Time (s) \\',r'\midrule']
    export=[];candidates=[]
    for row in sorted(data['records'],key=lambda r:r['n']):
        ref=next(r for r in audit['records'] if r['n']==row['n'])
        if export:lines.append(r'\midrule')
        for entry in row['methods']:
            method=entry['method'];j=entry['selected_exponent']
            assessed=next(r for r in ref['methods'] if r['method']==method)
            selected=next(c for c in entry['candidates'] if c['exponent']==j)
            label={'direct':r'$R_n$','uncorrected':r'$q_{\eps_*}$','corrected':r'$w_{\eps_*}$'}[method]
            lines.append(' & '.join([str(row['n']),label,'---' if j is None else f'$2^{{-{j}}}$',
                                     directed(rational(entry['selected_error_upper']),5),
                                     directed(rational(assessed['effectivity_upper'])),f"{entry['total_seconds']:.1f}"])+r' \\')
            export.append(dict(n=row['n'],method=method,epsilon_exponent=j,
                               residual_upper=selected['residual_upper']['decimal'],distance_upper=selected['distance_upper']['decimal'],
                               error_upper=entry['selected_error_upper']['decimal'],reference_lower=ref['error_lower']['decimal'],
                               reference_upper=ref['error_upper']['decimal'],effectivity_upper=assessed['effectivity_upper']['decimal'],
                               certified_candidates=sum(c['status']=='certified' for c in entry['candidates']),
                               fitting_and_search_seconds=entry['total_seconds']))
            for c in entry['candidates']:
                candidates.append(dict(n=row['n'],method=method,exponent=c['exponent'],status=c['status'],
                                       selected=c['exponent']==j,error_lower=c.get('error_lower',{}).get('decimal',''),
                                       residual_upper=c.get('residual_upper',{}).get('decimal',''),
                                       distance_upper=c.get('distance_upper',{}).get('decimal',''),
                                       error_upper=c.get('error_upper',{}).get('decimal',''),
                                       geometry_seconds=c.get('geometry',{}).get('seconds',''),
                                       norm_seconds=c.get('seconds',''),total_seconds=c['total_seconds'],reason=c.get('reason','')))
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'manuscript/tables/twod_reconstruction.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for name,rows in [('twod_reconstruction.csv',export),('twod_candidates.csv',candidates)]:
        with (ROOT/'results'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
