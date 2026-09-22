"""Generate extension tables and summaries directly from frozen records."""
from pathlib import Path
import csv,json,statistics,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def write_csv(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    for name in ['documents','results','manuscript/tables']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    local=json.loads((ROOT/'results/adaptive_local.json').read_text());twod=json.loads((ROOT/'results/unknown_junction_2d.json').read_text())
    assert local['status']==twod['status']=='completed'
    fixed=[];target=[];two=[];groups={}
    for row in local['fixed']:
        r=row['result'];fixed.append(dict(case=row['case'],n=row['n'],method=row['method'],repetition=row['repetition'],status=r['status'],
            bound=r.get('error_upper',{}).get('decimal',''),corrected_patches=r.get('corrected_patches',0),total_seconds=row['total_seconds']))
    for row in local['targets']:
        target.append(dict(case=row['case'],target=row['target'],method=row['method'],repetition=row['repetition'],status=row['status'],
            n=row.get('selected_n',''),bound=row.get('selected',{}).get('error_upper',{}).get('decimal',''),total_seconds=row['total_seconds']))
        groups.setdefault((row['case'],row['target']),{}).setdefault(row['method'],[]).append(row)
    comparisons=[]
    for method,base in [('adaptive_polynomial','global_polynomial'),('adaptive_hybrid','global_polynomial'),('adaptive_hybrid','adaptive_polynomial')]:
        pairs=[(statistics.median(x['total_seconds'] for x in d[method]),statistics.median(x['total_seconds'] for x in d[base])) for d in groups.values()
               if all(x['status']=='target_certified' for x in d[method]+d[base])]
        comparisons.append(dict(method=method,baseline=base,successful_pairs=len(pairs),faster=sum(a<b for a,b in pairs),median_speedup=statistics.median(b/a for a,b in pairs)))
    grouped={}
    for row in twod['records']:
        selected=row['selected'];c=selected['certificate'] if selected else {}
        out=dict(case=row['case'],n=row['n'],method=row['method'],repetition=row['repetition'],status=row['status'],
            bound=c.get('input_error_upper',{}).get('decimal',''),residual=c.get('residual_upper',{}).get('decimal',''),
            distance=c.get('distance_upper',{}).get('decimal',''),boundary=c.get('boundary_upper',{}).get('decimal',''),
            degree=selected['model']['degree'] if selected else '',scale=selected['model']['scale'] if selected else '',
            junction_boxes=c.get('junction_boxes',''),limited_boxes=c.get('tightness_limited_boxes',''),total_seconds=row['total_seconds'])
        two.append(out);grouped.setdefault((row['case'],row['n']),{}).setdefault(row['method'],[]).append(out)
    write_csv('adaptive_local_fixed.csv',fixed);write_csv('adaptive_local_targets.csv',target);write_csv('adaptive_local_comparison.csv',comparisons)
    write_csv('unknown_junction_2d.csv',two)
    first=[r for r in fixed if r['repetition']==0];byfixed={}
    for row in first:byfixed.setdefault((row['case'],row['n']),{})[row['method']]=row
    local_pairs=[(d['adaptive_hybrid'],d['adaptive_polynomial']) for d in byfixed.values() if d['adaptive_hybrid']['status']==d['adaptive_polynomial']['status']=='certified']
    summary=dict(target_comparisons=comparisons,local_fixed=dict(inputs=len(byfixed),hybrid_certified=sum(d['adaptive_hybrid']['status']=='certified' for d in byfixed.values()),
        polynomial_certified=sum(d['adaptive_polynomial']['status']=='certified' for d in byfixed.values()),
        hybrid_smaller_than_adaptive_polynomial=sum(a['bound']<b['bound']-1e-12 for a,b in local_pairs),
        hybrid_equal_to_adaptive_polynomial=sum(abs(a['bound']-b['bound'])<=1e-12 for a,b in local_pairs),
        corrected_patches=sum(d['adaptive_hybrid']['corrected_patches'] for d in byfixed.values())),
        twod=dict(inputs=len(grouped),certified_runs=sum(r['status']=='certified' for r in two),total_runs=len(two)))
    cbetter=qbetter=0
    for d in grouped.values():
        cbetter+=d['corrected'][0]['bound']<d['polynomial'][0]['bound'];qbetter+=d['corrected'][0]['bound']<d['uncorrected'][0]['bound']
    summary['twod'].update(corrected_smaller_than_polynomial=cbetter,corrected_smaller_than_uncorrected=qbetter)
    (ROOT/'results/adaptive_extension_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels={'variable_l1':'I','anisotropic_l2':'II','curved_l2':'III'}
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Bounds for the same two-dimensional Hermite inputs, rounded upward. $P$, $Q$ and $W$ denote direct, uncorrected and corrected branch fits. Each method ranks twelve degree--scale pairs and verifies up to three. Times are medians of three sequential runs, including input generation, all fits and all verification attempts.}\label{tab:twodreconstruction}',
        r'\begin{tabular}{crrrr@{\hspace{2em}}rrr}',r'\toprule',r'Case & $n$ & $B_P$ & $B_Q$ & $B_W$ & $T_P$ (s) & $T_Q$ (s) & $T_W$ (s) \\',r'\midrule']
    for (case,n),d in grouped.items():
        methods=['polynomial','uncorrected','corrected']
        bs=[math.ceil(max(r['bound'] for r in d[m])*10000)/10000 for m in methods]
        ts=[statistics.median(r['total_seconds'] for r in d[m]) for m in methods]
        lines.append(f"{labels[case]} & {n} & "+' & '.join(f'{v:.4f}' for v in bs)+' & '+' & '.join(f'{v:.2f}' for v in ts)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'manuscript/tables/twod_reconstruction.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    notes=['# Adaptive extension results','',
        'All comparisons below use the final frozen records. The target benchmark gives the original polynomial method early rejection and immediate stopping at a successful certificate.','',
        '| Method | Baseline | Faster targets | Median speedup |','|---|---|---:|---:|']
    for d in comparisons:notes.append(f"| {d['method']} | {d['baseline']} | {d['faster']}/{d['successful_pairs']} | {d['median_speedup']:.3f} |")
    notes+=['',f"On the 24 fixed inputs, the local hybrid certifies {summary['local_fixed']['hybrid_certified']}; the equally adaptive polynomial control certifies {summary['local_fixed']['polynomial_certified']}. The hybrid has a smaller bound on {summary['local_fixed']['hybrid_smaller_than_adaptive_polynomial']} common certified inputs, and selects {summary['local_fixed']['corrected_patches']} corrected patches in the first repetition.",
        '',f"The two-dimensional study certifies {summary['twod']['certified_runs']}/{summary['twod']['total_runs']} complete runs on six inputs. The corrected fit gives a smaller selected upper bound than the direct fit on {cbetter}/6 inputs and than the uncorrected fit on {qbetter}/6.",
        '', 'These results do not establish general practical superiority of contact correction. The local polynomial control is necessary to distinguish implementation savings from the contribution of the correction. The two-dimensional experiment establishes certification without prescribed junction coordinates for four fitted boundary-associated branches; arbitrary topology and asymptotic efficiency are not established.',
        '', 'CSV files retain per-run results, chosen parameters and all timing repetitions. JSON files also retain every proposal and attempted certificate.']
    (ROOT/'documents/ADAPTIVE_EXTENSION_RESULTS.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    make_figure(twod)
    print(json.dumps(summary,indent=2))

def make_figure(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
    axis=np.linspace(-1,1,321);x,y=np.meshgrid(axis,axis,indexing='ij')
    names=['variable_l1','anisotropic_l2','curved_l2']
    for ax,case,label in zip(axes,names,['I','II','III']):
        row=next(r for r in report['records'] if r['case']==case and r['n']==65 and r['method']=='corrected' and r['repetition']==0)
        vals=[]
        for b in row['selected']['model']['branches']:
            c=np.asarray(b['coefficients'],dtype=float)/2**b['bits'];vals.append(np.polynomial.polynomial.polyval2d(x,y,c))
        vals=np.asarray(vals);v=np.min(vals,axis=0)
        surface=ax.pcolormesh(x,y,v,cmap='viridis',shading='auto',rasterized=True)
        for a in range(4):
            for b in range(a+1,4):
                other=np.min(vals[[k for k in range(4) if k not in (a,b)]],axis=0)
                diff=np.ma.masked_where(np.minimum(vals[a],vals[b])>other,vals[a]-vals[b])
                if diff.count() and diff.min()<0<diff.max():ax.contour(x,y,diff,levels=[0],colors='white',linewidths=.9)
        ax.set(title=f'Case {label}',xlabel=r'$x_1$',ylabel=r'$x_2$',aspect='equal',xlim=(-1,1),ylim=(-1,1),xticks=[-1,0,1],yticks=[-1,0,1])
        colorbar=fig.colorbar(surface,ax=ax,shrink=.75,pad=.02);colorbar.ax.yaxis.set_major_locator(MaxNLocator(3))
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'unknown_junctions.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(folder/'unknown_junctions.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
