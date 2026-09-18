"""Generate data-derived plots and CSV summaries from saved experiment results."""
from pathlib import Path
import csv, hashlib, json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from hjrl.verified_twod import BASE, problem_float


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def up(x):return f'{math.ceil(x*10000)/10000:.4f}'


def main():
    source=ROOT/'results/twod_contact.json'
    data=json.loads(source.read_text())
    assert data['complete']
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.9),layout='constrained')
    rows=[];summaries=[]
    for color,row in zip(['#1464a0','#bd5726'],data['records']):
        n0=row['critic_side']
        best=[]
        for m in sorted({s['location_side'] for s in row['settings']}):
            subset=[s for s in row['settings'] if s['location_side']==m]
            best.append(min(subset,key=lambda s:s['complete_bound']['upper']))
        chosen=min(best,key=lambda s:s['complete_bound']['upper'])
        summaries.append(dict(critic_side=n0,reference_error=row['reference_error'],
                              best_bound=chosen['complete_bound']['upper'],
                              best_location_side=chosen['location_side'],best_center_side=chosen['center_side'],
                              best_reciprocal_scale=chosen['reciprocal_scale'],
                              best_seconds=chosen['standalone_seconds'],
                              ratio_lower=chosen['complete_bound']['upper']/row['reference_error']['upper'],
                              ratio_upper=chosen['complete_bound']['upper']/row['reference_error']['lower']))
        axes[1].plot([s['location_side'] for s in best],[s['complete_bound']['upper'] for s in best],
                     '-o',color=color,markersize=4,label=f'Critic {n0}'+'$^2$: certificate')
        axes[1].axhline(row['reference_error']['upper'],color=color,linestyle='--',linewidth=1,
                       label=f'Critic {n0}'+'$^2$: reference error')
        for s in row['settings']:
            entry=dict(critic_side=n0,location_side=s['location_side'],center_side=s['center_side'],
                       reciprocal_scale=s['reciprocal_scale'],bound=s['complete_bound']['upper'],
                       coverage_correction=s['coverage_correction']['upper'],
                       lower_side=s['filtered_one_sided_bounds']['-1']['upper'],
                       upper_side=s['filtered_one_sided_bounds']['1']['upper'],
                       seconds=s['standalone_seconds'],reference_error_lower=row['reference_error']['lower'],
                       reference_error_upper=row['reference_error']['upper'])
            rows.append(entry)
    n0=data['records'][0]['critic_side']
    u=np.load(ROOT/f'results/twod_critic_{n0}.npz')['nodes']/BASE
    x=-1+2*np.arange(n0)/n0
    xx,yy=np.meshgrid(x,x,indexing='ij')
    error=u-problem_float(xx,yy)[0]
    periodic_error=np.pad(error,((0,1),(0,1)),mode='wrap')
    xp=np.linspace(-1,1,n0+1)
    im=axes[0].pcolormesh(xp,xp,periodic_error.T,cmap='magma',shading='nearest',rasterized=True)
    axes[0].set(xlabel='$x_1$',ylabel='$x_2$',title=f'Critic {n0}'+'$^2$: nodal error',xlim=(-1,1),ylim=(-1,1),aspect='equal')
    fig.colorbar(im,ax=axes[0],fraction=.046,pad=.03)
    axes[1].set_xscale('log',base=2)
    grid_sides=sorted({r['location_side'] for r in rows})
    axes[1].set_xticks(grid_sides,[str(x) for x in grid_sides])
    axes[1].set(xlabel='Contact-location nodes per coordinate',ylabel='Uniform error / upper bound',title='Complete 2D verification')
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=.18)
    axes[1].legend(fontsize=7,frameon=False,loc='lower left',bbox_to_anchor=(0,.15))
    figure=ROOT/'figures/twod_contact.pdf'
    fig.savefig(figure,dpi=600,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(figure.with_suffix('.png'),dpi=300)
    plt.close(fig)
    with figure.with_suffix('.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved plots and CSV summaries to figures/')


if __name__=='__main__':main()
