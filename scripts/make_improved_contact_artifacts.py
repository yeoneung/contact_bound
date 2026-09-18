"""Generate data-derived plots and CSV summaries from saved experiment results."""
from pathlib import Path
import csv,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def up(x):return f'{math.ceil(x*1e4)/1e4:.4f}'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def best_signed(settings,key):
    best={s:min(settings,key=lambda r:r[key][s]['upper']) for s in ('-1','1')}
    return {'bound':max(best[s][key][s]['upper'] for s in best),
            'reciprocal_scales':{s:best[s]['reciprocal_scale'] for s in best}}


def main():
    shift_path=ROOT/'results/improved_contact_estimator.json'
    filter_path=ROOT/'results/stationary_contact_estimator.json'
    shifted=json.loads(shift_path.read_text());filtered=json.loads(filter_path.read_text())
    reference=json.loads((ROOT/'results/estimator_comparison.json').read_text())
    assert shifted['complete'] and filtered['complete']
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.1),layout='constrained',sharey=True)
    summaries=[];csv_rows=[]
    for ax,row in zip(axes,shifted['records']):
        name=row['critic']
        extra=next(r for r in filtered['records'] if r['critic']==name)
        old=next(r for r in reference['records'] if r['critic']==name)
        coarse=[c for c in row['settings'] if c['grid_nodes']==16384]
        fine=[c for c in row['settings'] if c['grid_nodes']==65536]
        assert len(coarse)==len(fine)==len(extra['settings'])==5
        summary={'critic':name,'value_only':best_signed(fine,'restricted_one_sided_bounds'),
                 'filtered':best_signed(extra['settings'],'one_sided_bounds'),
                 'unrestricted':best_signed(fine,'one_sided_bounds'),
                 'old_best':min(c['complete_bound']['upper'] for c in old['contacts']),
                 'old_initial_grid_best':min(c['complete_bound']['upper'] for c in old['contacts']
                                            if c['grid_nodes']==16384 and c['centers']==262144),
                 'bellman_best':min(c['complete_bound']['upper'] for c in old['bellman']),
                 'extra_derivative_seconds':extra['derivative_verification_seconds']}
        summaries.append(summary)
        ms=[4,8,16,32,64]
        ax.plot(ms,[c['restricted_complete_bound']['upper'] for c in coarse],'--o',color='#c36022',markersize=3,label='Shifted: coarse grid')
        ax.plot(ms,[c['restricted_complete_bound']['upper'] for c in fine],'-o',color='#c36022',markersize=3,label='Shifted: fine grid')
        ax.plot(ms,[c['complete_bound']['upper'] for c in extra['settings']],'-s',color='#1464a0',markersize=3,label='Fine grid + stationarity')
        ax.axhline(summary['old_best'],color='#777777',linestyle='-.',label='Best previous contact bound')
        ax.axhline(summary['bellman_best'],color='#257449',linestyle=':',label='Best tested Bellman bound')
        ax.set_xscale('log',base=2);ax.set_xticks(ms,[f'$1/{m}$' for m in ms])
        ax.set(title=name,xlabel=r'Contact scale $\varepsilon$');ax.grid(alpha=.18)
        for c,f,s in zip(coarse,fine,extra['settings']):
            assert c['reciprocal_scale']==f['reciprocal_scale']==s['reciprocal_scale']
            csv_rows.append({'critic':name,'reciprocal_scale':f['reciprocal_scale'],
                             'coarse_shifted_upper':c['restricted_complete_bound']['upper'],
                             'fine_unrestricted_upper':f['complete_bound']['upper'],
                             'fine_shifted_lower_side':f['restricted_one_sided_bounds']['-1']['upper'],
                             'fine_shifted_upper_side':f['restricted_one_sided_bounds']['1']['upper'],
                             'filtered_lower_side':s['one_sided_bounds']['-1']['upper'],
                             'filtered_upper_side':s['one_sided_bounds']['1']['upper'],
                             'shifted_seconds':f['seconds'],'filtered_seconds':s['seconds'],
                             'extra_derivative_seconds':extra['derivative_verification_seconds']})
    axes[0].set_ylabel('Complete value-error upper bound')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=7,frameon=False)
    path=ROOT/'figures/improved_contact.pdf'
    fig.savefig(path,dpi=600,metadata={'CreationDate':None,'ModDate':None});fig.savefig(path.with_suffix('.png'),dpi=300)
    plt.close(fig)
    with path.with_suffix('.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print('Saved plots and CSV summaries to figures/')


if __name__=='__main__':main()
