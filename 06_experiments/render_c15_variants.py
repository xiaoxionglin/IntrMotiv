"""C15 lineage survey over validated studies and existing telemetry only.

StudySpecs supply identities and experimental factors. Canonical scalar/spatial
exports supply control metrics. Shared poster renderers supply the diagnostic
figures; online windows and frozen probes are never pooled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study.spec import load_study
from hpc_runs.intrmotiv_study.spatial import render_graph_outcomes
from analyze_place_field_manifest import multilevel_field_structure
from render_flat_goal_comparison import (
    setup_style, short_return, place_field_figure, trajectory_occupancy_figure,
    dg_kernel_figure, render_dg_peak_map, COLORS, FIGURE_SCALE,
)
from render_goal_option_trajectory import render as render_trajectory, trajectory_segments
from render_poster_frozen import render_flow, flow_cells

STUDIES = {
    'directional_predictive_recruitment': 'DPR',
    'source_credit_retirement': 'Source credit',
    'saturday_batch': 'Saturday',
    'dg_policy_gradient_first_outcome': 'DGP',
    'ca3_feedback_predictive_dg': 'CPD',
    'persistent_intrinsic_control_c15': 'Persistent C15',
}
CORE = {'C14': 'CCR_C14_TOPOLOGY_VISIT_O1',
        'C15': 'CCR_C15_TOPOLOGY_UCB_DIRECT_O1',
        'C16': 'CCR_C16_TOPOLOGY_UCB_WAYPOINT_O1'}
GRAPH_METRICS = ['grounded_controllability', 'prospective_success', 'graph_reachability']
SEED_COLORS = {8: '#0072B2', 99: '#D55E00', 123: '#009E73'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry(root: Path) -> pd.DataFrame:
    records = []
    for name, family in STUDIES.items():
        source = Path(f'hpc_runs/studies/{name}.study.json')
        study = load_study(root / source)
        for run in study.expand_runs():
            if run.base not in ('C15', 'C15_CONTINUE'):
                continue
            settings = dict(arg.split('=', 1) for arg in run.args if '=' in arg)
            records.append({
                'family': family, 'condition': run.condition, 'seed': run.seed,
                'run_name': run.name, 'short_label': ' '.join(run.factor_labels.values()).replace('_', ' ') or run.base,
                'study_source': str(source), 'study_schema': study.raw['schema'],
                'workflow_version': study.declared_workflow_version, 'study_sha256': study.fingerprint,
                'factors': json.dumps(run.factors, sort_keys=True),
                'manager_mode': settings.get('--hrl_manager_mode', ''),
                'target_selection': settings.get('--hrl_direct_target_selection', ''),
                'control_outcome': settings.get('--hrl_control_outcome', ''),
                'goal_conditioning': settings.get('--hrl_goal_conditioning', ''),
                'ppo_dg_gradient': settings.get('--ppo_dg_gradient', ''),
                'dg_units': int(settings.get('--Hippo_n_feature', 16)),
                'encoder_credit': settings.get('--encoder_reward_recipient', ''),
                'recruitment_rule': settings.get('--dg_recruitment_victim_rule', ''),
                'endpoint_gate': settings.get('--dg_recruitment_endpoint_gate', ''),
                'context_feedback': settings.get('--dg_context_feedback', ''),
                'context_history': settings.get('--dg_context_history', ''),
                'context_gradient': settings.get('--dg_context_gradient', ''),
                'transition_predictor': settings.get('--dg_transition_prediction', ''),
            })
    for code, condition in CORE.items():
        for seed in (8, 99, 123):
            records.append({'family': 'Corrected core', 'condition': condition, 'seed': seed,
                            'run_name': f'{condition}_S{seed}', 'short_label': code,
                            'study_source': '06_experiments/corrected_core_reevaluation_20260901.md',
                            'manager_mode': 'frontier_waypoint' if code == 'C16' else 'frontier_direct',
                            'target_selection': 'visit novelty' if code == 'C14' else 'UCB',
                            'dg_units': 16})
    frame = pd.DataFrame(records)
    if frame.run_name.duplicated().any():
        raise ValueError('Study registry has duplicate run identities')
    return frame


def read_input(root: Path, output: Path, relative: str, audit: list) -> pd.DataFrame:
    path = root / relative
    destination = output / 'source_inputs' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(path.read_bytes())
    audit.append({'original_file': str(path), 'repository_file': relative,
                  'sha256': sha(path), 'pinned_file': str(destination.relative_to(output))})
    return pd.read_csv(path)


def control_tables(root: Path, output: Path, runs: pd.DataFrame, audit: list) -> pd.DataFrame:
    survey = read_input(root, output, '06_experiments/data/poster_missing_analyses_20260926/cross_run_scatter/all_run_metrics.csv', audit)
    online = survey[(survey.protocol == 'online_latest_saved_window') & survey.run_name.isin(runs.run_name)].copy()
    online['protocol'] = 'online_100k_training_window'
    online['evaluation_age'] = online.frames
    pieces = [online]
    for family, source in [
        ('Saturday', '06_experiments/results/recent_batches_audit_20260906/saturday_terminal_per_run.csv'),
        ('DPR', '06_experiments/results/recent_batches_audit_20260906/dpr_terminal_per_run.csv')]:
        original = read_input(root, output, source, audit)
        selected = original[original.run_name.isin(runs.run_name)].copy()
        selected = selected.rename(columns={'reachable_pair_fraction': 'graph_reachability',
                                            'mono_field_fraction': 'mono_fraction_eligible',
                                            'active_map_cosine': 'map_cosine'})
        selected['source_csv'] = source
        selected['source_artifact'] = selected.get('wandb_id', selected.run_name)
        selected['protocol'] = 'final_training_scalar_window'
        selected['evaluation_age'] = selected.max_env_steps
        selected['prospective_success'] = np.nan  # Option success is a different estimand.
        pieces.append(selected)
    core = read_input(root, output, '06_experiments/results/corrected_core_reevaluation_20260902/per_run_terminal_10m.csv', audit)
    selected = core[core.condition.isin(CORE)].copy()
    selected['condition'] = selected.condition.map(CORE)
    selected['run_name'] = selected.condition + '_S' + selected.seed.astype(str)
    selected['protocol'] = 'final_10m_training_scalar_window'
    selected['evaluation_age'] = 100040704
    selected['source_csv'] = '06_experiments/results/corrected_core_reevaluation_20260902/per_run_terminal_10m.csv'
    pieces.append(selected)
    metrics = pd.concat(pieces, ignore_index=True)
    metrics = metrics.drop(columns=['family', 'condition', 'seed', 'dg_units'], errors='ignore').merge(
        runs[['run_name', 'condition', 'seed', 'family', 'short_label', 'dg_units']], on='run_name', validate='many_to_one')
    for name in GRAPH_METRICS:
        if name not in metrics: metrics[name] = np.nan
    metrics.to_csv(output / 'control_per_run.csv', index=False)
    summary = metrics.groupby(['family', 'protocol', 'condition'], sort=False)[GRAPH_METRICS].agg(['mean', 'std', 'count', 'min', 'max'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / 'control_by_condition.csv', index=False)
    return metrics


def metric_sheet(data: pd.DataFrame, metrics: list[tuple[str, str]], destination: Path,
                 title: str, *, percentage: bool = True) -> None:
    conditions = data[['condition', 'short_label']].drop_duplicates().sort_values('short_label')
    order = conditions.condition.tolist()
    height = max(3.4, 1.6 + .34 * len(order))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, height), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (metric, label) in zip(axes, metrics):
        for index, condition in enumerate(order):
            group = data[data.condition == condition]
            values = pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors='coerce')
            if not values.notna().any():
                ax.text(.5, index, 'unavailable', transform=ax.get_yaxis_transform(),
                        ha='center', va='center', fontsize=12, color='#777777')
                continue
            multiplier = 100 if percentage else 1
            for seed, value in zip(group.seed, values):
                if np.isfinite(value):
                    dy = {8: -.08, 99: 0, 123: .08}.get(int(seed), 0)
                    ax.scatter(value * multiplier, index + dy, s=35, color=SEED_COLORS.get(int(seed), '#555555'), zorder=3)
            ax.plot([values.mean() * multiplier] * 2, [index-.18, index+.18], lw=2, color='black')
        ax.set(title=label, xlabel='Percent' if percentage else label)
        if percentage: ax.set_xlim(-2, 102)
        ax.grid(axis='x', alpha=.2)
        ax.set_yticks(range(len(order)), conditions.short_label.tolist(), fontsize=12)
        ax.tick_params(labelsize=12)
    axes[0].invert_yaxis()
    fig.suptitle(title + '\nBlue: seed 8 · orange: 99 · green: 123 · black: mean', fontsize=14)
    fig.savefig(destination, bbox_inches='tight')
    plt.close(fig)


def declared_contrasts(root: Path, output: Path, data: pd.DataFrame) -> None:
    """Use the original StudySpec contrasts and the canonical pairing engine."""
    from hpc_runs.intrmotiv_study.analysis import linear_contrasts

    details, summaries = [], []
    for study_name, family in STUDIES.items():
        study = load_study(root / f'hpc_runs/studies/{study_name}.study.json')
        lookup = {run.name: run for run in study.expand_runs()}
        contrasts = study.raw.get('analysis', {}).get('contrasts', [])
        if not contrasts: continue
        for protocol, group in data[data.family == family].groupby('protocol'):
            columns = [m for m in [*GRAPH_METRICS, 'coverage_auc', 'option_success', 'target_hit_lift']
                       if m in group and group[m].notna().any()]
            records = [{**row, **lookup[row['run_name']].context} for row in group.to_dict('records')]
            detailed, aggregate = linear_contrasts(
                records, columns, study.raw['analysis'].get('contrast_group_by', []),
                study.raw['analysis'].get('replicate_by', ['seed']), contrasts)
            binding = {'family': family, 'protocol': protocol, 'study_sha256': study.fingerprint,
                       'workflow_version': study.declared_workflow_version, 'study_schema': study.raw['schema']}
            details.extend({**binding, **row} for row in detailed)
            summaries.extend({**binding, **row} for row in aggregate)
    pd.DataFrame(details).to_csv(output / 'declared_paired_contrasts_per_seed.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'declared_paired_contrasts_summary.csv', index=False)


def command_evidence(root: Path, output: Path, runs: pd.DataFrame, audit: list) -> pd.DataFrame:
    """Keep bounded command probes separate from training-time graph counters."""
    rows = []
    source_root = root / '06_experiments/data/poster_missing_analyses_20260926/command_json'
    for path in sorted(source_root.rglob('intervention_summary.json')):
        record = json.loads(path.read_text())
        if record.get('condition') not in set(runs.condition): continue
        relative = path.relative_to(root)
        destination = output / 'source_inputs' / relative
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(path.read_bytes())
        audit.append({'original_file': str(path), 'sha256': sha(path), 'pinned_file': str(destination.relative_to(output))})
        rows.append({'condition': record['condition'], 'seed': record['seed'],
                     'checkpoint_frames': record['checkpoint_frames'], 'trial_count': record['trial_count'],
                     'executed_success': record['executed_target_success_rate'],
                     'shuffled_success': record['matched_shuffled_target_success_rate'],
                     'executed_minus_shuffled': record['executed_target_success_rate']-record['matched_shuffled_target_success_rate'],
                     'action_sensitivity': record['mean_counterfactual_action_sensitivity'],
                     'complete_pair_fraction': record['ordered_pairs_complete']/record['ordered_pairs_eligible'],
                     'original_file': str(path), 'checkpoint': record['checkpoint']})
    result = pd.DataFrame(rows).merge(runs[['condition', 'seed', 'family', 'short_label']],
                                      on=['condition', 'seed'], validate='one_to_one')
    result.to_csv(output / 'command_evidence_per_run.csv', index=False)
    for family, group in result.groupby('family'):
        metric_sheet(group, [('executed_success', 'Executed target success'),
                             ('shuffled_success', 'Shuffled target success'),
                             ('complete_pair_fraction', 'Complete ordered pairs')],
                     output / f'{family.lower()}_command_evidence.svg',
                     family + ' · bounded frozen command probes')
    return result


def raw_inventory(roots: list[Path], runs: pd.DataFrame) -> dict[tuple, tuple]:
    """Identify saved archives from their metadata; never infer factors from filenames."""
    names = runs.set_index('run_name')[['condition', 'seed', 'family', 'short_label']].to_dict('index')
    result = {}
    for root in roots:
        if not root.exists(): continue
        paths = sorted(root.rglob('*.npz'))
        for path in paths:
            if path.name != 'place_fields.npz' and not path.name.startswith('snapshot_'): continue
            with np.load(path, allow_pickle=False) as z:
                if 'run_name' in z:
                    name = str(z['run_name']); protocol = 'online_100k_training_window'; age = int(z['actual_env_steps'])
                elif 'checkpoint' in z:
                    name = next((p.name.removeprefix('00_') for p in Path(str(z['checkpoint'])).parents
                                 if p.name.removeprefix('00_') in names), '')
                    protocol = 'frozen_10k_policy_probe'; age = int(Path(str(z['checkpoint'])).stem.rsplit('_', 1)[-1])
                else: continue
                if name not in names: continue
                expected_age = 100040704 if names[name]['family'] == 'Corrected core' else (75005952 if protocol.startswith('online') else 75038720)
                # Exploratory DPR probes are unmatched checkpoint ages, retained separately.
                if names[name]['family'] == 'DPR': protocol = 'frozen_unmatched_age_probe'
                elif age != expected_age: continue
                if protocol.startswith('frozen') and not path.with_name('pose.csv').exists(): continue
                key = (name, protocol)
                if key not in result or age > result[key][1]: result[key] = (path, age)
    return result


def diagnostics(path: Path, age: int, record: dict, protocol: str, output: Path) -> dict:
    short = record['short_label']; seed = int(record['seed']); condition = record['condition']
    label = f'{short} · S{seed}'
    destination = output / 'per_run' / f'{record["run_name"]}_{age}_{"online" if protocol.startswith("online") else "frozen"}'
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as z:
        occupancy, maps, info, active = [z[k].copy() for k in ['occupancy', 'rate_maps', 'spatial_information', 'active_fraction']]
        if 'pose' in z:
            pose = pd.DataFrame({'x': z['pose'][:, 0], 'y': z['pose'][:, 1], 'rot_y': z['pose'][:, 2],
                                 'agent': 0, 'num_traj': z['segment_id'], 'frame': np.arange(len(z['pose']))})
        else: pose = pd.read_csv(path.with_name('pose.csv'))
        eligible, _, _, score, mono = multilevel_field_structure(maps, occupancy, active)
        peak_bins = [np.unravel_index(np.nanargmax(maps[:, :, u]), occupancy.shape)
                     if np.nanmax(maps[:, :, u]) > 0 else (-1, -1) for u in range(len(active))]
        unit_table = pd.DataFrame({'unit': range(len(active)), 'active_fraction': active,
                                  'peak_x_bin': [p[0] for p in peak_bins], 'peak_y_bin': [p[1] for p in peak_bins],
                                  'field_eligible': eligible, 'mono_field': mono, 'mono_score': score})
        unit_table.to_csv(destination / 'dg_units.csv', index=False)
        for evidence, numerator in [('stored', 'control_edge_confidence'), ('prospective', 'control_prospective_successes')]:
            denominator = 'control_attempts' if evidence == 'stored' else 'control_prospective_attempts'
            if numerator in z and denominator in z:
                # Frozen buffers can differ by tiny floating-point roundoff only.
                attempts = z[denominator].astype(float); successes = z[numerator].astype(float)
                if np.any(successes > attempts + 1e-5): raise ValueError(f'Invalid {evidence} graph counts: {path}')
                successes = np.minimum(successes, attempts)
                with plt.rc_context({'svg.fonttype': 'none'}):
                    render_graph_outcomes(attempts, successes, destination / f'{evidence}_graph',
                                          title=label + f'\n{evidence.capitalize()} graph',
                                          ratio_label='Stored confidence / attempts' if evidence == 'stored' else 'Prospective hits / attempts',
                                          figure_scale=.58, formats=('svg',))
    setup_style()
    COLORS[short] = '#0072B2'
    for scope in ('full', 'first_episode'):
        render_trajectory(pose, destination / f'trajectory_{scope}.svg', label, scope, (100, 2000, 100, 2000),
                          sampling_label='training window' if protocol.startswith('online') else 'archived probe',
                          figure_scale=FIGURE_SCALE, compact_title=label)
    trajectory_occupancy_figure(pose, occupancy, destination, short, seed)
    render_flow(pose, destination / 'flow.svg', label + ' · flow', figure_scale=FIGURE_SCALE)
    flow_cells(pose).to_csv(destination / 'flow_cells.csv', index=False)
    limits = place_field_figure(maps, occupancy, info, destination, short, seed)
    pd.DataFrame(limits).to_csv(destination / 'place_field_scales.csv', index=False)
    dg_kernel_figure(maps, occupancy, destination, short, seed)
    selected = unit_table[(unit_table.active_fraction > 0) & (unit_table.peak_x_bin >= 0)]
    peak_record = SimpleNamespace(condition=short, seed=seed)
    render_dg_peak_map(selected, occupancy, peak_record, len(active), destination / 'all_active_dg_peaks.svg', 'all active DG peaks')
    render_dg_peak_map(selected[selected.mono_field], occupancy, peak_record, len(active), destination / 'mono_field_peaks.svg', 'Mono-field peaks')
    # Use contiguous segment runs for loop denominators, even if IDs are reused.
    pieces = list(trajectory_segments(pose))
    contiguous = pd.concat([segment.assign(num_traj=i) for i, segment in enumerate(pieces)], ignore_index=True)
    result = {**record, 'protocol': protocol, 'evaluation_age': age, 'observations': len(pose),
              'contiguous_segments': len(pieces), 'source_npz': str(path), 'source_sha256': sha(path),
              'source_pose': str(path.with_name('pose.csv')) if path.with_name('pose.csv').exists() else 'pose + segment_id within NPZ',
              'source_pose_sha256': sha(path.with_name('pose.csv')) if path.with_name('pose.csv').exists() else sha(path),
              'visited_fraction': np.mean(occupancy > 0), 'dg_units': len(active),
              'mono_units': int(mono.sum()), 'distinct_peak_bins': len(selected[['peak_x_bin', 'peak_y_bin']].drop_duplicates()),
              'figure_dir': str(destination.relative_to(output)), 'mobile_loop_definition': 'displacement <100, path >500; within contiguous segment only'}
    for lag in (20, 40):
        try: result.update(short_return(contiguous, lag))
        except ValueError:
            result[f'return_{lag}_mobile'] = np.nan
            result[f'windows_{lag}_mobile'] = 0
    return result


def diagnostic_index(output: Path, runs: pd.DataFrame, control: pd.DataFrame,
                     probes: pd.DataFrame) -> None:
    """Regenerate the complete SVG/source index and all declared-run availability."""
    availability = runs[['family', 'condition', 'seed', 'run_name']].copy()
    availability['control_record_available'] = availability.run_name.isin(control.run_name)
    availability['local_trajectory_available'] = availability.run_name.isin(probes.run_name)
    availability['frozen_probe_available'] = availability.run_name.isin(
        probes[probes.protocol.str.startswith('frozen')].run_name)
    availability['online_trajectory_available'] = availability.run_name.isin(
        probes[probes.protocol.str.startswith('online')].run_name)
    availability.to_csv(output / 'run_availability.csv', index=False)
    lines = ['# C15 variant diagnostic index', '',
             'Every listed run uses unchanged saved data. Full paths break at contiguous episode/segment boundaries. SVG text stays editable.', '',
             '| Variant / seed | Protocol / age | Full trajectory | First segment | Occupancy + path | Flow | Top four fields | All DG peaks | Mono peaks | DG kernel | Graph | Original local data |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in probes.sort_values(['family', 'condition', 'protocol', 'seed']).itertuples():
        folder = row.figure_dir
        links = [f'[SVG]({folder}/{name}.svg)' for name in
                 ['trajectory_full', 'trajectory_first_episode', 'trajectory_occupancy', 'flow',
                  'place_fields', 'all_active_dg_peaks', 'mono_field_peaks', 'dg_kernel']]
        graph_links = [f'[{evidence}]({folder}/{evidence}_graph.svg)' for evidence in ['stored', 'prospective']
                       if (output / folder / f'{evidence}_graph.svg').exists()]
        source = f'[NPZ]({row.source_npz})'
        if row.source_pose != 'pose + segment_id within NPZ': source += f' · [pose]({row.source_pose})'
        cells = [f'{row.family} {row.short_label} S{row.seed}', f'{row.protocol} · {row.evaluation_age:,}',
                 *links, ' · '.join(graph_links) or 'not saved', source]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'run_index.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--raw-roots', nargs='+', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalogue-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    setup_style(); runs = registry(args.input_root)
    runs.to_csv(args.output / 'variant_registry.csv', index=False)
    audit = []; metrics = control_tables(args.input_root, args.output, runs, audit)
    command_evidence(args.input_root, args.output, runs, audit)
    declared_contrasts(args.input_root, args.output, metrics)
    core = metrics[metrics.family == 'Corrected core']
    metric_sheet(core, [('coverage_auc', 'Coverage AUC'), ('unique_cells', 'Unique cells'),
                        ('target_hit_lift', 'Target-hit lift'), ('option_success_fraction', 'Option success')],
                 args.output / 'corrected_core_training_comparison.svg',
                 'C14 / C15 / C16 · final 10M training window', percentage=False)
    (args.output / 'input_provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
    for (family, protocol), data in metrics.groupby(['family', 'protocol']):
        if any(data[m].notna().any() for m in GRAPH_METRICS):
            stem = family.lower().replace(' ', '_') + '_' + protocol
            metric_sheet(data, list(zip(GRAPH_METRICS, ['Grounded score', 'Prospective success', 'Graph reachability'])),
                         args.output / f'{stem}_control.svg', f'{family} · {protocol.replace("_", " ")}')
    if args.catalogue_only: return
    inventory = raw_inventory(args.raw_roots, runs)
    lookup = runs.set_index('run_name').to_dict('index')
    results = []
    for (name, protocol), (path, age) in inventory.items():
        record = {**lookup[name], 'run_name': name}
        print(f'Rendering {name} {protocol}', flush=True)
        results.append(diagnostics(path, age, record, protocol, args.output))
    detailed = pd.DataFrame(results)
    previous_path = args.output / 'trajectory_per_run.csv'
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        if 'original_cluster_npz' in previous:
            detailed = detailed.merge(previous[['run_name', 'protocol', 'evaluation_age', 'original_cluster_npz']],
                                      on=['run_name', 'protocol', 'evaluation_age'], how='left', validate='one_to_one')
    detailed.to_csv(previous_path, index=False)
    diagnostic_index(args.output, runs, metrics, detailed)
    for (family, protocol), data in detailed.groupby(['family', 'protocol']):
        metric_sheet(data, [('visited_fraction', 'Visited arena bins'), ('return_20_mobile', '20-step return'),
                            ('return_40_mobile', '40-step return')],
                     args.output / f'{family.lower().replace(" ", "_")}_{protocol}_behavior.svg',
                     f'{family} · {protocol.replace("_", " ")}')
    print(f'{runs.condition.nunique()} variants; {len(metrics)} control records; {len(detailed)} detailed run/protocol rows.')


if __name__ == '__main__': main()
