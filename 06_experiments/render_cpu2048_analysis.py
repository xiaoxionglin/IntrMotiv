"""CPU comparison adapter over canonical StudySpecs and collect-spatial tables.

Run on NEMO2 with the repository on PYTHONPATH. Raw snapshots stay there.
Uses the existing spatial map implementation; only report composition is new.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.collections import LineCollection
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spatial_contract import SpatialBounds, spatial_rate_maps


def save(fig, path):
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.'+ext), dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--spec-root', type=Path, default=Path('hpc_runs/studies'))
    a = ap.parse_args()
    font = Path(font_manager.findfont('DejaVu Sans', fallback_to_default=False))
    assert font.suffix.lower() in ('.ttf', '.otf')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,
        'axes.titlesize':18,'xtick.labelsize':14,'ytick.labelsize':14,
        'pdf.fonttype':42})
    records, provenance, declarations = [], [], []
    for spec in sorted(a.spec_root.glob('controller_cpu*.study.json')):
        study = load_study(spec)
        provenance.append(study.provenance())
        runs = {r.name:r for r in study.expand_runs()}
        frame = pd.read_csv(a.root/spec.stem/'per_snapshot.csv')
        for r in runs.values():
            args = dict(x[2:].split('=',1) for x in r.args if '=' in x)
            row = {'run_name':r.name,'base':r.base,'controller':r.factors['controller'],
                'seed':r.seed,'cadence':int(args['controller_decisions_per_update']),
                'study_id':study.raw['study_id']}
            declarations.append(row)
        frame['cadence'] = frame.run_name.map({r['run_name']:r['cadence'] for r in declarations})
        records.append(frame)
    df = pd.concat(records, ignore_index=True)
    declared = pd.DataFrame(declarations)
    assert declared.run_name.nunique()==len(declared)==24
    common = set.intersection(*(set(g.target_env_steps) for _,g in df.groupby('run_name')))
    assert df.run_name.nunique()==24 and common, 'No common all-run milestone'
    target = max(common)
    matched = df[df.target_env_steps==target].copy()
    assert len(matched)==24
    def label(row):
        arch = 'Direct F16' if row['base']=='DIRECT_F16' else 'Waypoint F64'
        learner = row['controller'].replace('_HER','+HER')
        return f"{arch} · {learner} · cadence {int(row['cadence'])}"
    matched['label'] = matched.apply(label,axis=1)
    df.to_csv(a.root/'all_snapshots.csv',index=False)
    declared.to_csv(a.root/'declared_runs.csv',index=False)
    matched.to_csv(a.root/'matched_per_run.csv',index=False)
    metrics = ['active_only_map_cosine','silent_unit_fraction','unique_active_peak_bins',
        'mono_field_unit_fraction','active_unit_mean_spatial_information','stationary_step_fraction',
        'path_efficiency','visited_cell_fraction','graph_prospective_success_fraction',
        'graph_reliable_global_efficiency','graph_grounded_controllability','graph_reliable_edge_count']
    matched.groupby('label')[metrics].agg(['mean','std']).to_csv(a.root/'matched_summary.csv')
    figdir = a.root/'figures'; figdir.mkdir(exist_ok=True)
    for metric in metrics:
        fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
        for y,(name,g) in enumerate(matched.groupby('label',sort=True)):
            for seed,marker in [(8,'o'),(99,'s'),(123,'^')]:
                ax.scatter(g[g.seed==seed][metric], [y], marker=marker,s=65,color='#0072B2')
            ax.plot([g[metric].mean()]*2,[y-.16,y+.16],color='#D55E00',lw=3)
        ax.set_yticks(range(8),sorted(matched.label.unique())); ax.invert_yaxis()
        ax.set_xlabel(metric.replace('_',' ')); ax.set_title(f'{target/1e6:g}M frames · all seeds')
        ax.grid(axis='x',alpha=.2)
        save(fig,figdir/metric)
    atlas=['# CPU2048 spatial atlas', '',f'Common snapshot: {target/1e6:g}M frames. All 24 runs; no seed selection.',
        '', 'Fields: activity divided by each unit’s peak; gray means unvisited. This displays shape, not amplitude.',
        'Trajectory overlay retains all segments. Four example segments are selected at evenly spaced indices, without behavior-based screening.',
        'Graph matrix shows recorded prospective hits / attempts; gray means unattempted, not failed.', '']
    edges = pd.concat([pd.read_csv(p) for p in a.root.glob('*.study/graph_edge.csv')],ignore_index=True)
    for _,row in matched.sort_values(['label','seed']).iterrows():
        run=row.run_name; folder=figdir/run; folder.mkdir(exist_ok=True)
        title=f"{row.label}\nseed {row.seed} · {target/1e6:g}M frames"
        with np.load(row.snapshot_path,allow_pickle=False) as data:
            pose=data['pose']; seg=data['segment_id']; activity=data['dg_activity']
            bounds=SpatialBounds(*data['bounds'].tolist())
            maps,occ,_=spatial_rate_maps(pose,activity,bounds,int(data['grain']))
            extent=(bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max)
            atlas += [f'## {row.label} — seed {row.seed}', '']
            for start in range(0,len(maps),16):
                fig,axes=plt.subplots(4,4,figsize=(13,13),layout='constrained')
                cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#ddd')
                for ax,u in zip(axes.flat,range(start,min(start+16,len(maps)))):
                    m=maps[u]; peak=m.max(); norm=m/peak if peak>0 else m
                    im=ax.imshow(np.ma.array(norm,mask=occ==0),origin='lower',extent=extent,
                        cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
                    ax.set_title(f'DG {u}'); ax.set_xticks([100,2000]); ax.set_yticks([100,2000])
                fig.suptitle(title); fig.colorbar(im,ax=list(axes.flat),shrink=.6,label='Activity / unit peak')
                stem=f'fields_{start//16+1}'; save(fig,folder/stem)
                atlas += [f'![Fields {start}–{min(start+15,len(maps)-1)}](figures/{run}/{stem}.png)', '']
            starts=np.r_[0,np.flatnonzero(seg[1:]!=seg[:-1])+1]; ends=np.r_[starts[1:],len(seg)]
            lines=[pose[s:e,:2] for s,e in zip(starts,ends) if e-s>1]
            fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained')
            im=axs[0].imshow(np.ma.masked_equal(occ,0),origin='lower',extent=extent,cmap='cividis')
            fig.colorbar(im,ax=axs[0],shrink=.65,label='Retained observations')
            axs[1].add_collection(LineCollection(lines,colors='#0072B2',alpha=.12,linewidths=.65))
            for ax in axs: ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='x',ylabel='y')
            axs[0].set_title('Occupancy'); axs[1].set_title(f'All {len(lines)} segments')
            fig.suptitle(title); save(fig,folder/'trajectory')
            atlas += [f'![Occupancy and trajectory](figures/{run}/trajectory.png)', '']
            fig,axs=plt.subplots(2,2,figsize=(10,10),layout='constrained')
            for ax,idx in zip(axs.flat,np.linspace(0,len(lines)-1,4,dtype=int)):
                line=lines[idx]; ax.plot(line[:,0],line[:,1],color='#0072B2'); ax.scatter(*line[0],color='#009E73'); ax.scatter(*line[-1],marker='x',color='#D55E00')
                ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',title=f'Segment {idx} · {len(line)} samples')
            fig.suptitle(title+'\nGreen: start · orange: end'); save(fig,folder/'segments')
            atlas += [f'![Individual segments](figures/{run}/segments.png)', '']
        e=edges[(edges.run_name==run)&(edges.target_env_steps==target)]
        n=int(row.capacity); matrix=np.full((n,n),np.nan)
        for edge in e.itertuples():
            if edge.prospective_attempts>0:
                matrix[int(edge.source_unit),int(edge.target_unit)]=edge.prospective_successes/edge.prospective_attempts
        fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
        cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#ddd')
        im=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap=cmap,interpolation='nearest')
        ax.set(xlabel='Target DG unit',ylabel='Source DG unit',title=title)
        fig.colorbar(im,ax=ax,shrink=.7,label='Prospective hits / attempts')
        save(fig,folder/'graph')
        atlas += [f'![Directed graph outcomes](figures/{run}/graph.png)', '']
        print(run,flush=True)
    (a.root/'atlas.md').write_text('\n'.join(atlas))
    (a.root/'provenance.json').write_text(json.dumps({'studies':provenance,'target':int(target),
        'font':str(font),'n_runs':24,'map_scaling':'per-unit peak normalization',
        'summary':'all seed points; orange tick=unweighted mean; no significance tests'},indent=2))


if __name__=='__main__': main()
