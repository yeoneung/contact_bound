"""Generate data-derived plots and CSV summaries from saved experiment results."""
from pathlib import Path
import csv,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def up(x):return f'{math.ceil(x*1e4)/1e4:.4f}'


def candidates(records):
    out=[]
    for r in records:
        if r['method']=='contact':
            for mode in ('values','filtered'):
                out.append({'key':r['key']+':'+mode,'method':'Contact','variant':mode,
                            'bound':r['unfiltered_bound' if mode=='values' else 'complete_bound']['upper'],
                            'seconds':r['unfiltered_standalone_seconds' if mode=='values' else 'standalone_seconds']})
        else:
            out.append({'key':r['key'],'method':'Bellman','variant':r['successor_mode'],
                        'bound':r['complete_bound']['upper'],'seconds':r['standalone_seconds']})
    return out


def frontier(rows):
    out=[];best=float('inf')
    for r in sorted(rows,key=lambda r:(r['seconds'],r['bound'])):
        if r['bound']<best:
            out.append(r);best=r['bound']
    return out


def main():
    source=ROOT/'results/state_dependent_comparison.json';data=json.loads(source.read_text())
    assert data['complete'] and len(data['records'])==156 and len(data['operators'])==30
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.0,4.5),layout='constrained',sharey=True)
    summaries=[];csvrows=[]
    for rownum,name in enumerate(('MLP','Min2')):
        for col,beta in enumerate((0,1,2)):
            ax=axes[rownum,col]
            records=[r for r in data['records'] if r['critic']==name and r['beta_quarters']==beta]
            choices=candidates(records)
            for r in choices:csvrows.append({'critic':name,'beta':beta/4,**r})
            best={};targets={};tuning={}
            for method,color,marker in [('Contact','#b35b22','o'),('Bellman','#176d91','s')]:
                rs=[r for r in choices if r['method']==method]
                fr=frontier(rs);best[method]=min(rs,key=lambda r:(r['bound'],r['seconds']))
                ax.scatter([r['seconds'] for r in rs],[r['bound'] for r in rs],s=9,alpha=.3,color=color,marker=marker)
                ax.step([r['seconds'] for r in fr],[r['bound'] for r in fr],where='post',color=color,
                        marker=marker,markersize=3,label=method)
                targets[method]={str(t):min((r['seconds'] for r in rs if r['bound']<=t),default=None)
                                 for t in data['design']['target_bounds']}
                cache=data['critic_caches'][name]
                base=cache['base_seconds']+cache['derivative_seconds']
                if method=='Contact':tuning[method]=base+sum(r['method_seconds'] for r in records if r['method']=='contact')
                else:
                    ops={r['operator_key'] for r in records if r['method']=='bellman'}
                    tuning[method]=base+sum(r['operator_seconds'] for r in data['operators'] if r['key'] in ops)+sum(
                        r['residual_evaluation_seconds'] for r in records if r['method']=='bellman')
            ax.set_title(name+rf', $\beta={beta/4:g}$')
            ax.grid(alpha=.15);ax.set_xscale('log')
            if rownum==1:ax.set_xlabel('Standalone verification time (s)')
            if col==0:ax.set_ylabel('Complete error upper bound')
            summary={'critic':name,'beta':beta/4,'best':best,'target_seconds':targets,'full_sweep_seconds':tuning}
            summaries.append(summary)
            fields=[name,f'{beta/4:g}',up(best['Contact']['bound']),f"{best['Contact']['seconds']:.1f}",
                    up(best['Bellman']['bound']),f"{best['Bellman']['seconds']:.1f}"]
            for method in ('Contact','Bellman'):
                t=targets[method]['0.8'];fields.append('--' if t is None else f'{t:.1f}')
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=2,frameon=False)
    figure=ROOT/'figures/state_dependent_budget.pdf'
    fig.savefig(figure,dpi=600,metadata={'CreationDate':None,'ModDate':None});fig.savefig(figure.with_suffix('.png'),dpi=300);plt.close(fig)
    with figure.with_suffix('.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    print('Saved plots and CSV summaries to figures/')


if __name__=='__main__':main()
