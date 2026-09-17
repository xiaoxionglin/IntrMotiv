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
from hpc_runs.intrmotiv_study import WORKFLOW_VERSION, load_study
from hpc_runs.intrmotiv_study.spatial import (
    ATLAS_FIGURE_STYLE, render_place_field_contact_sheets,
    render_occupancy_trajectory, render_trajectory_segments, render_graph_outcomes,
)


def save(fig, path):
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.'+ext), dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--spec-root', type=Path, default=Path('hpc_runs/studies'))
    a = ap.parse_args()
    prior_path = a.root/'provenance.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    collection_versions = prior.get('collection_workflow_versions', sorted({
        item['workflow_version'] for item in prior.get('studies', [])
    }))
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
        'Trajectory colors distinguish independent segments in storage order, not speed or global time. Circles mark starts, crosses mark ends, and arrows show heading. No segments are joined.',
        'Four example segments are selected at evenly spaced indices, without behavior-based screening.',
        'Graph matrix shows recorded prospective hits / attempts; gray means unattempted, not failed.', '']
    for _,row in matched.sort_values(['label','seed']).iterrows():
        run=row.run_name; folder=figdir/run; folder.mkdir(exist_ok=True)
        title=f"{row.label}\nseed {row.seed} · {target/1e6:g}M frames"
        with np.load(row.snapshot_path,allow_pickle=False) as data:
            atlas += [f'## {row.label} — seed {row.seed}', '']
            pages = render_place_field_contact_sheets(data, folder/'fields', title=title)
            # Keep existing report URLs stable while delegating page rendering.
            for page, path in enumerate((p for p in pages if p.suffix == '.png'), start=1):
                for suffix in ('.png', '.pdf'):
                    path.with_suffix(suffix).replace(folder/f'fields_{page}{suffix}')
                atlas += [f'![Place fields](figures/{run}/fields_{page}.png)', '']
            render_occupancy_trajectory(data, folder/'trajectory', title=title)
            atlas += [f'![Occupancy and trajectory](figures/{run}/trajectory.png)', '']
            render_trajectory_segments(data, folder/'segments', title=title)
            atlas += [f'![Individual segments](figures/{run}/segments.png)', '']
            render_graph_outcomes(data['control_prospective_attempts'],
                data['control_prospective_successes'], folder/'graph', title=title)
        atlas += [f'![Directed graph outcomes](figures/{run}/graph.png)', '']
        print(run,flush=True)
    (a.root/'atlas.md').write_text('\n'.join(atlas))
    (a.root/'provenance.json').write_text(json.dumps({'studies':provenance,'target':int(target),
        'font':str(font),'n_runs':24,'map_scaling':'per-unit peak normalization',
        'figure_style':ATLAS_FIGURE_STYLE,
        'figure_workflow_version':WORKFLOW_VERSION,
        'collection_workflow_versions':collection_versions,
        'summary':'all seed points; orange tick=unweighted mean; no significance tests'},indent=2))


if __name__=='__main__': main()
