"""Plot corridor results from canonical CSV/history exports; never reread event files.

Each dot is one map seed. Online curves use 2M-frame bins of logged episode
statistics; condition curves weight each of the three layouts equally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study import load_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()
    study = load_study(args.study)
    runs = study.expand_runs()
    bases = list(dict.fromkeys(r.base for r in runs))
    openness = sorted({r.factors['openness'] for r in runs})
    seeds = sorted({r.factors['map_seed'] for r in runs})
    labels = {'SAT': 'SAT FiLM', 'DGP': 'DGP joint', 'WAYPOINT_HER': 'Waypoint F64 HER'}
    colors = ['#0072B2', '#D55E00', '#009E73']
    font = font_manager.findfont('DejaVu Sans', fallback_to_default=False)
    assert Path(font).suffix.lower() in {'.ttf', '.otf'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16,
                         'axes.titlesize': 17, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14,
                         'legend.fontsize': 14, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    out = args.data / 'figures'
    out.mkdir(exist_ok=True)
    online = pd.read_csv(args.data / 'online/per_run.csv')
    spatial = pd.read_csv(args.data / 'spatial/per_snapshot.csv')
    terminal = spatial[spatial.target_env_steps == 100_000_000].copy()
    assert len(terminal) == len(online) == len(runs) == 27
    geometry = json.loads((args.study.parent / 'assets/corridor_geometry/maps.json').read_text())['maps']
    areas = {(r['map_seed'], r['wall_removal_probability']): r['accessible_cells'] for r in geometry}
    online['accessible_cells'] = [areas[(r.map_seed, r.openness)] for r in online.itertuples()]
    online['mean_accessible_cells_visited'] = online.accessible_coverage_fraction * online.accessible_cells
    metrics = ['active_only_map_cosine', 'silent_unit_fraction', 'unique_active_peak_bins',
               'active_unit_mean_spatial_information', 'graph_reliable_edge_count',
               'graph_largest_strong_component_size', 'stationary_step_fraction',
               'graph_reachable_pair_fraction', 'graph_prospective_success_fraction']
    merged = online[['run_name', 'base', 'map_seed', 'openness', 'accessible_cells',
                     'accessible_coverage_auc', 'coverage_auc', 'accessible_coverage_fraction',
                     'mean_accessible_cells_visited']].merge(terminal[['run_name', *metrics]],
                                                           on='run_name', validate='one_to_one')
    merged.to_csv(args.data / 'layout_results.csv', index=False)
    measures = ['accessible_coverage_auc', 'coverage_auc', 'accessible_coverage_fraction',
                'mean_accessible_cells_visited', *metrics]
    paired = []
    for (base, seed), frame in merged.groupby(['base', 'map_seed']):
        frame = frame.set_index('openness')
        for q in openness[:-1]:
            row = {'base': base, 'map_seed': seed, 'openness': q, 'reference_openness': openness[-1]}
            row.update({m: float(frame.loc[q, m] - frame.loc[openness[-1], m]) for m in measures})
            paired.append(row)
    pd.DataFrame(paired).to_csv(args.data / 'paired_layout_effects.csv', index=False)
    summary = merged.groupby(['base', 'openness'])[measures].agg(['mean', 'min', 'max'])
    summary.to_csv(args.data / 'descriptive_summary.csv')
    units = pd.read_csv(args.data / 'spatial_details/per_unit.csv')
    units = units.merge(spatial[['run_name', 'target_env_steps', 'base', 'map_seed', 'openness']],
                        on=['run_name', 'target_env_steps'], validate='many_to_one')
    active = units[(units.target_env_steps == 100_000_000) & (units.active_fraction > 0)].copy()
    active['single_component'] = active.geometry_field_components_half_peak == 1
    components = active.groupby(['run_name', 'base', 'map_seed', 'openness']).agg(
        mean_traversable_components=('geometry_field_components_half_peak', 'mean'),
        single_traversable_component_fraction=('single_component', 'mean'),
        active_units=('unit_id', 'count')).reset_index()
    components.to_csv(args.data / 'traversable_field_summary.csv', index=False)

    def save(fig, name):
        fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1), sharey=True)
    for ax, base in zip(axes, bases):
        frame = merged[merged.base == base]
        for seed, color, marker in zip(seeds, colors, ['o', 's', '^']):
            d = frame[frame.map_seed == seed].sort_values('openness')
            ax.plot(range(len(openness)), d.accessible_coverage_auc * 100,
                    marker=marker, color=color, linewidth=1.7, markersize=7, label=str(seed))
        ax.set_title(labels[base])
        ax.set_xticks(range(3), ['Corridor\n0.00', 'Intermediate\n0.35', 'Open\n0.75'])
        ax.set_ylim(0, float(merged.accessible_coverage_auc.max()) * 110)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Episode coverage AUC\n(% of accessible area)')
    axes[-1].legend(title='Map seed', loc='best')
    fig.suptitle('Final training window: 95–100M environment frames', y=1.02)
    fig.tight_layout()
    save(fig, 'final_coverage')

    all_curves = []
    tag = study.analysis['window_metrics']['accessible_coverage_auc']
    for run in runs:
        d = pd.read_csv(args.data / 'online/histories' / f'{run.name}.csv')
        d = d[(d.tag == tag) & (d.step > 0) & (d.step <= 100_000_000)].copy()
        d['frame_bin'] = ((d.step - 1) // 2_000_000 + 1) * 2
        curve = d.groupby('frame_bin').value.agg(['mean', 'count']).reset_index()
        curve['base'], curve['openness'], curve['map_seed'] = run.base, run.factors['openness'], run.factors['map_seed']
        curve['run_name'] = run.name
        all_curves.append(curve)
    curves = pd.concat(all_curves, ignore_index=True)
    curves.to_csv(args.data / 'learning_curve_bins.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1), sharey=True)
    for ax, base in zip(axes, bases):
        for q, color in zip(openness, colors):
            d = curves[(curves.base == base) & (curves.openness == q)]
            for seed in seeds:
                one = d[d.map_seed == seed]
                ax.plot(one.frame_bin, one['mean'] * 100, color=color, alpha=.2, linewidth=.8)
            group = d.groupby('frame_bin')['mean'].agg(['mean', 'count'])
            group = group[group['count'] == len(seeds)]
            ax.plot(group.index, group['mean'] * 100, color=color, linewidth=2.4, label=f'q={q:.2f}')
        ax.set_title(labels[base]); ax.set_xlabel('Environment frames (millions)')
        ax.set_xlim(0, 100); ax.set_ylim(0, float(curves['mean'].max()) * 110); ax.grid(alpha=.2)
    axes[0].set_ylabel('Episode coverage AUC\n(% of accessible area)')
    axes[-1].legend(loc='best')
    fig.suptitle('Training curves: 2M-frame bins; thin lines show individual layouts', y=1.02)
    fig.tight_layout(); save(fig, 'learning_curves')

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 9), sharex=True, sharey='row')
    for col, base in enumerate(bases):
        frame = merged[merged.base == base]
        for seed, color, marker in zip(seeds, colors, ['o', 's', '^']):
            d = frame[frame.map_seed == seed].sort_values('openness')
            for row, metric in enumerate(['active_only_map_cosine', 'stationary_step_fraction']):
                axes[row, col].plot(range(3), d[metric], color=color, marker=marker, linewidth=1.7, label=str(seed))
        axes[0, col].set_title(labels[base])
        axes[1, col].set_xticks(range(3), ['Corridor\n0.00', 'Intermediate\n0.35', 'Open\n0.75'])
        for ax in axes[:, col]: ax.grid(axis='y', alpha=.2); ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel('Active-only map cosine\n(lower = less overlap)')
    axes[1, 0].set_ylabel('Stationary-step fraction')
    axes[0, -1].legend(title='Map seed', loc='best')
    fig.suptitle('100M online snapshots: field overlap and movement', y=1.01)
    fig.tight_layout(); save(fig, 'spatial_diagnostics')
    metadata = {'study_sha256': study.fingerprint, 'schema': study.raw['schema'],
                'workflow_version': '1.10.1', 'font_path': font, 'replicate_unit': 'map seed',
                'training_seed': 99, 'n_layouts': 3, 'terminal_window': [95000000, 100000000],
                'curve_bin_width': 2000000, 'curves': 'event means per run/bin; equal-layout condition mean',
                'uncertainty': 'individual layouts shown; no inferential intervals',
                'spatial_protocol': 'training-trajectory online snapshots, not frozen evaluations',
                'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_hashes': {str(p.relative_to(args.data)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [args.data/'online/per_run.csv', args.data/'spatial/per_snapshot.csv', args.data/'spatial_details/per_unit.csv']}}
    (out / 'figure_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
