"""Thin interim report adapter: standardized CSVs and optional existing snapshots.

No run-name parsing, event discovery, training, or new evaluation. Individual
seeds and unweighted seed means are shown; no significance testing.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT = Path(__file__).parent / 'data/dgc_interim_20260911'
COLORS = {'DIRECT_WORKER':'#0072B2', 'DIRECT_DG':'#D55E00', 'WAYPOINT_DG':'#009E73'}
LABELS = {'DIRECT_WORKER':'Direct: worker', 'DIRECT_DG':'Direct: DG + worker', 'WAYPOINT_DG':'Waypoint: DG + worker'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path)
    args=parser.parse_args()
    font=Path(font_manager.findfont('DejaVu Sans',fallback_to_default=False))
    assert font.is_file() and font.suffix in ('.ttf','.otf')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'axes.titlesize':18,
                         'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,
                         'legend.fontsize':15,'pdf.fonttype':42,'svg.fonttype':'none'})
    figroot=ROOT/'figures'; figroot.mkdir(exist_ok=True)
    p=pd.read_csv(ROOT/'spatial/per_snapshot.csv')
    matched=p[p.target_env_steps==25000000].copy()
    assert len(matched)==27 and matched.run_name.nunique()==27
    metrics=['active_only_map_cosine','unique_active_peak_bins','mono_field_unit_fraction',
             'silent_unit_fraction','stationary_step_fraction','visited_cell_fraction',
             'graph_prospective_success_fraction','graph_grounded_controllability',
             'graph_reachable_pair_fraction','graph_reliable_edge_count']
    matched[['run_name','base','capacity','seed']+metrics].to_csv(ROOT/'matched25m_per_run.csv',index=False)
    matched.groupby(['base','capacity'])[metrics].agg(['mean','std','min','max']).to_csv(ROOT/'matched25m_summary.csv')
    for name,panels in [('control_and_overlap',[
        ('graph_prospective_success_fraction','Success on previously known edges',1),
        ('active_only_map_cosine','Active-only map cosine',1)]),('field_quality_and_movement',[
        ('mono_field_unit_fraction','Single-field fraction (eligible units)',1),
        ('stationary_step_fraction','Stationary decision fraction',1)])]:
        fig,axs=plt.subplots(1,2,figsize=(13,5.8))
        for ax,(metric,label,scale) in zip(axs,panels):
            for k,(base,color) in enumerate(COLORS.items()):
                q=matched[matched.base==base]
                xs=np.arange(3)+(k-1)*0.15
                means=q.groupby('capacity')[metric].mean().reindex([16,32,64])
                ax.plot(xs,means*scale,'-',color=color,linewidth=2,label=LABELS[base])
                for seed,marker in [(8,'o'),(99,'s'),(123,'^')]:
                    values=q[q.seed==seed].set_index('capacity')[metric].reindex([16,32,64])
                    ax.scatter(xs,values*scale,marker=marker,s=55,color=color,edgecolor='white',linewidth=.5,zorder=3)
            ax.set_xticks(range(3),['16','32','64']); ax.set_xlabel('DG units'); ax.set_ylabel(label)
            ax.set_ylim(0,1); ax.grid(axis='y',alpha=.18); ax.spines[['top','right']].set_visible(False)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        fig.text(.5,.02,'25M frames; points: seeds 8 (circle), 99 (square), 123 (triangle); lines: seed means',ha='center',fontsize=14)
        fig.subplots_adjust(top=.84,bottom=.21,left=.08,right=.98,wspace=.32)
        for ext in ['png','pdf']: fig.savefig(figroot/f'{name}.{ext}',dpi=100)
        plt.close(fig)
    peak_rows=[]
    if args.raw_root:
        for path in sorted(args.raw_root.rglob('*.npz')):
            with np.load(path) as data:
                maps=data['rate_maps']; occupancy=data['occupancy']; run=str(data['run_name'].item())
                n=maps.shape[-1]; assert maps.shape[:2]==occupancy.shape
                active=(data['dg_activity']>0).any(axis=0)
                peaks=np.argmax(maps.reshape(-1,n),axis=0)[active]
                ids,counts=np.unique(peaks,return_counts=True); order=np.argsort(-counts)
                peak_rows.append(dict(run_name=run,active_units=int(active.sum()),unique_peak_bins=len(ids),
                    largest_peak_cluster_units=int(counts.max()),largest_peak_cluster_fraction=float(counts.max()/active.sum()),
                    top3_peak_cluster_units=int(counts[order[:3]].sum())))
                for start in range(0,n,16):
                    fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained')
                    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#d9d9d9')
                    for ax,unit in zip(axes.flat,range(start,min(start+16,n))):
                        m=maps[:,:,unit]; peak=m.max()
                        normalized=m/peak if peak>0 else m
                        im=ax.imshow(np.ma.array(normalized,mask=occupancy==0),origin='lower',
                            vmin=0,vmax=1,cmap=cmap,interpolation='nearest',extent=(100,2000,100,2000))
                        ax.set_title(f'DG {unit}'); ax.set_xticks([100,2000]); ax.set_yticks([100,2000])
                    for ax in axes.flat[min(16,n-start):]: ax.set_visible(False)
                    fig.colorbar(im,ax=list(axes.flat),shrink=.55,label='Activity / unit peak (gray: unvisited)')
                    fig.suptitle(f'{run}\n25M snapshot · units {start}–{min(start+15,n-1)}',fontsize=18)
                    for ext in ['png','pdf']: fig.savefig(figroot/f'{run}_fields_{start//16+1}.{ext}',dpi=100)
                    plt.close(fig)
    if peak_rows: pd.DataFrame(peak_rows).to_csv(ROOT/'selected_peak_clusters.csv',index=False)
    (ROOT/'figure_metadata.json').write_text(json.dumps({'font':str(font),'aggregation':'unweighted mean of three seeds; individual seeds shown',
        'snapshot_target_frames':25000000,'snapshot_window_decisions':100000,'map_normalization':'each unit divided by its peak; unvisited bins masked',
        'spatial_data':'spatial/per_snapshot.csv; exact raw paths in spatial/snapshot_inventory.csv',
        'limits':'policy-driven thresholded maps; no pre-threshold arrays or fixed-panel stability; graph success is observational'},indent=2)+'\n')


if __name__=='__main__': main()
