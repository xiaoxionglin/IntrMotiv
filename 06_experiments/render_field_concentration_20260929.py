"""Poster candidates for continuous field concentration, using local archives.

Exact run/model age/protocol keys bind raw maps to the pinned survey. Capacity
stays at 16; absent raw maps remain absent. The broad dominance-only comparison
reuses canonical per-unit CSV exports instead of inventing map concentration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from compose_a0_poster_20260929 import (
    ROOT, SCATTER, RESULTS, TABLE, SVG, COLORS, PT_MM, dots, embed, np, pd,
    plt, sha, style, text, wrap_lines,
)
from render_poster_candidate_plots_20260929 import (
    Candidate, Gallery, LARGE_MM, ROW_MM, SMALL_MM, paired, scalar_axes, verify_and_render,
)
from hpc_runs.intrmotiv_study.field_concentration import calculate_field_concentration
from hpc_runs.intrmotiv_study.version import WORKFLOW_VERSION
from analyze_place_field_manifest import multilevel_field_structure
from matplotlib.lines import Line2D

DEFAULT_OUT = ROOT/'05_plans/poster_20260929/continuous_fields'
KEYS = ['run_name', 'policy_id', 'frames', 'protocol']
MEASURES = ['concentration', 'entropy_concentration', 'effective_area_fraction',
            'mass80_area_fraction', 'dominance', 'concentration_min5']
CPD_COLORS = ['#444444', '#0072B2', '#D55E00', '#009E73']
CPD_LABELS = {
    'CPD_C15_BASE': 'Baseline', 'CPD_C15_ADD_CA3_DIR': 'CA3 feedback',
    'CPD_C15_GATE_ACT_DIR': 'Goal context', 'CPD_C15_GATE_ACT_DIR_GOAL': '+ predictor',
}
COMMAND_GROUPS = {
    'DGP': {'DGP_C15_FIRST_JOINT_LEG': 'First', 'DGP_C15_HIT_JOINT_LEG': 'Hit'},
    'Saturday': {'SAT_C15_SRC_MON_FILM': 'Source', 'SAT_C15_ARR_MON_FILM': 'Arrival'},
}


def checkpoint_identity(checkpoint: str, names: set[str]) -> tuple[str, int]:
    """Read identity from the recorded checkpoint directory, not a condition parser."""
    path = Path(checkpoint)
    name = next((p.name.removeprefix('00_') for p in path.parents
                 if p.name.removeprefix('00_') in names), '')
    return name, int(path.stem.rsplit('_', 1)[-1])


def collect_maps(g: Gallery, survey: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units, availability, seen = [], [], {}
    names = set(survey.run_name)
    directories = [ROOT/'06_experiments/data/c15_variants_20260928/raw',
        ROOT/'06_experiments/data/flat_goal_comparison_20260927/corrected_core_candidates_20260902_place_fields/raw',
        ROOT/'06_experiments/data/poster_missing_analyses_20260926/common_fields',
        RESULTS/'raw_snapshots', RESULTS/'frozen_raw']
    for directory in directories:
        for path in sorted(directory.rglob('*.npz')):
            with np.load(path, allow_pickle=False) as data:
                if 'rate_maps' not in data or 'occupancy' not in data:
                    continue
                panel = str(data['observation_panel']) if 'observation_panel' in data else ''
                if 'run_name' in data:
                    name, age = str(data['run_name']), int(data['actual_env_steps'])
                    protocol = 'online_latest_saved_window'
                    policy = int(data['policy_id'])
                elif 'checkpoint' in data:
                    if 'checkpoint_p0' not in Path(str(data['checkpoint'])).parts:
                        raise ValueError(f'Frozen policy identity is not p0: {path}')
                    name, age = checkpoint_identity(str(data['checkpoint']), names)
                    protocol = ('common_replay_with_frozen_outcomes' if panel
                                else 'frozen_latest_archived_probe')
                    policy = 0
                else:
                    continue
                matched = survey[(survey.run_name == name) & (survey.frames == age)
                                 & (survey.protocol == protocol)]
                audit = dict(file=str(path.relative_to(ROOT)), run_name=name, frames=age,
                             protocol=protocol, raw_capacity=data['rate_maps'].shape[-1])
                if len(matched) != 1:
                    availability.append({**audit, 'status': 'no_exact_latest_survey_match'})
                    continue
                row = matched.iloc[0]
                if row.dg_units != 16:
                    availability.append({**audit, 'status': 'separate_capacity_not_plotted'})
                    continue
                if protocol.startswith('common'):
                    if Path(panel).name != Path(str(row.representation_panel)).name:
                        raise ValueError(f'Observation panel mismatch: {path}')
                maps = data['rate_maps'].astype(float)
                occupancy = data['occupancy'].astype(float)
                # Legacy offline maps mark unknown bins NaN; preserve observed zeros.
                maps[occupancy == 0] = 0
                active = data['active_fraction'].astype(float)
                if maps.shape[-1] != int(row.dg_units):
                    raise ValueError(f'Capacity mismatch: {path}')
                key = (name, policy, age, protocol)
                if key in seen:
                    if not np.allclose(seen[key][0], maps, equal_nan=True) or not np.array_equal(seen[key][1], occupancy):
                        raise ValueError(f'Duplicate maps disagree: {path}')
                    availability.append({**audit, 'status': 'duplicate_same_maps'})
                    continue
                seen[key] = (maps, occupancy)
                source_hash = sha(path)
                g.sources[str(path.relative_to(ROOT))] = source_hash
                if protocol.startswith('online'):
                    eligible = data['field_eligible'].astype(bool)
                    dominance = data['field_mono_score'].astype(float)
                else:
                    eligible, _, _, dominance, _ = multilevel_field_structure(maps, occupancy, active)
                shape = calculate_field_concentration(maps, occupancy)
                sensitivity = calculate_field_concentration(maps, occupancy, minimum_bin_observations=5)
                for unit in range(maps.shape[-1]):
                    units.append({**dict(zip(KEYS, key)), 'unit': unit,
                        'field_eligible': bool(eligible[unit]), 'active_fraction': active[unit],
                        'dominance': dominance[unit] if eligible[unit] else np.nan,
                        **{k: values[unit] for k, values in shape.items() if k != 'supported_bins'},
                        'concentration_min5': sensitivity['concentration'][unit],
                        'supported_bins': int(shape['supported_bins']),
                        'supported_bins_min5': int(sensitivity['supported_bins']),
                        'unit_source': str(path.relative_to(ROOT)), 'source_kind': 'raw_maps',
                        'source_sha256': source_hash, 'map_array': 'rate_maps',
                        'occupancy_array': 'occupancy'})
                availability.append({**audit, 'status': 'included',
                    'eligible_units': int(eligible.sum()), 'supported_bins': int(shape['supported_bins']),
                    'source_sha256': source_hash})
    return pd.DataFrame(units), pd.DataFrame(availability)


def collect_unit_tables(g: Gallery, survey: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    """Only match saved unit rows at the exact latest age, with a fixed capacity."""
    paths = [ROOT/'06_experiments/data/navigation8_algorithm_screen_interim_20260916/online_spatial/per_unit.csv',
        ROOT/'06_experiments/data/dgc_interim_20260911/spatial/per_unit.csv']
    latest = survey[(survey.protocol == 'online_latest_saved_window') & (survey.dg_units == 16)
                    & (survey.geometry_group == 'legacy_19x19')][KEYS].copy()
    pieces = [maps]
    for path in paths:
        original = g.read(path).rename(columns={'actual_env_steps': 'frames', 'unit_id': 'unit',
                                                'mono_field_score': 'dominance'})
        original['protocol'] = 'online_latest_saved_window'
        selected = original.merge(latest, on=KEYS, validate='many_to_one')
        if selected.empty:
            continue
        selected['unit_source'] = str(path.relative_to(ROOT))
        selected['source_sha256'] = sha(path)
        selected['source_kind'] = 'canonical_unit_csv_dominance_only'
        pieces.append(selected[KEYS+['unit', 'field_eligible', 'dominance', 'active_fraction',
                                     'unit_source', 'source_sha256', 'source_kind']])
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(KEYS+['unit']).any():
        raise ValueError('Duplicate exact unit keys')
    return result


def aggregate_units(units: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in units.groupby(KEYS):
        valid = group[group.field_eligible.astype(bool)]
        record = {**dict(zip(KEYS, key)), 'total_units': len(group),
                  'eligible_units': len(valid), 'silent_units': int((group.active_fraction == 0).sum()),
                  'unit_source': group.unit_source.iloc[0], 'source_kind': group.source_kind.iloc[0]}
        for measure in MEASURES:
            values = valid[measure].dropna()
            record[measure] = values.mean() if len(values) else np.nan
            record[measure+'_median'] = values.median() if len(values) else np.nan
        for name in ['supported_bins', 'supported_bins_min5']:
            record[name] = group[name].iloc[0]
        rows.append(record)
    result = pd.DataFrame(rows).merge(survey, on=KEYS, validate='one_to_one')
    if (result.total_units != result.dg_units).any():
        raise ValueError('Incomplete unit records or capacity mismatch')
    return result


def rho(data: pd.DataFrame, x: str, y: str) -> float:
    selected = data[[x, y]].dropna()
    if len(selected) < 3 or selected[x].nunique() < 2 or selected[y].nunique() < 2:
        return np.nan
    return float(selected.corr(method='spearman').iloc[0, 1])


def correlation_tables(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    outcomes = ['exploration_coverage', 'prospective_success', 'graph_reachability',
                'map_cosine', 'stationary_fraction', 'path_efficiency', 'executed_minus_shuffled',
                'coverage_auc_terminal', 'action_sensitivity_terminal', 'return_40_mobile']
    for (protocol, family), group in runs.groupby(['protocol', 'family']):
        for aggregation in ['run_seed', 'three_seed_variant_mean']:
            if aggregation == 'three_seed_variant_mean':
                counts = group.groupby('condition').seed.nunique()
                ages = group.groupby('condition').frames.nunique()
                complete = counts[(counts == 3) & (ages == 1)].index
                selected = group[group.condition.isin(complete)].groupby('condition').mean(numeric_only=True)
            else:
                selected = group
            for measure in MEASURES:
                for outcome in outcomes:
                    valid = selected[[measure, outcome]].dropna()
                    records.append({'protocol': protocol, 'family': family,
                        'aggregation': aggregation, 'measure': measure, 'outcome': outcome,
                        'n': len(valid), 'variants': group.loc[group[measure].notna(), 'condition'].nunique(),
                        'spearman_rho': rho(valid, measure, outcome)})
    return pd.DataFrame(records)


def command_summary_data(g: Gallery, survey: pd.DataFrame) -> pd.DataFrame:
    """Recover all saved seeds without substituting a mean score for minimum D.

    Replay-summary `frames` counts observations, whereas checkpoint_frames
    identifies model age. Keep both, and audit against the pinned survey and
    replay manifest before joining frozen command outcomes.
    """
    base = ROOT/'06_experiments/data/poster_missing_analyses_20260926'
    replay_path = base/'replay_reduced_derived.csv'
    command_path = base/'control_interventions_per_run.csv'
    replay = g.read(replay_path).rename(columns={
        'frames': 'replay_observations', 'checkpoint_frames': 'frames',
        'mean_dominant_component_mass': 'mean_threshold_dominance'})
    command = g.read(command_path).rename(columns={
        'checkpoint_frames': 'frames', 'protocol': 'command_protocol',
        'executed_minus_shuffled': 'command_lift'})
    keys = ['condition', 'seed', 'frames']
    metadata = survey[survey.protocol.eq('common_replay_with_frozen_outcomes')
                      & survey.family.isin(COMMAND_GROUPS)].copy()
    result = metadata.merge(replay[keys+['replay_observations', 'field_eligible_units',
                                        'mean_threshold_dominance']],
                            on=keys, validate='one_to_one')
    result = result.merge(command, on=keys, validate='one_to_one', suffixes=('', '_command'))
    if len(result) != 12 or not (result.frames == 75038720).all() or not (result.dg_units == 16).all():
        raise ValueError('Expected 12 same-age DG-16 command/replay models')
    for family, groups in COMMAND_GROUPS.items():
        selected = result[result.family == family]
        if set(selected.condition) != set(groups) or not all(
            set(group.seed) == {8, 99, 123} for _, group in selected.groupby('condition')
        ):
            raise ValueError(f'Incomplete three-seed command comparison: {family}')
    if not (result.field_eligible_units > 0).all() or not result.mean_threshold_dominance.between(0, 1).all():
        raise ValueError('Undefined or out-of-range continuous dominance summary')
    if not np.allclose(result.command_lift, result.executed_minus_shuffled) or not np.allclose(
        result.command_lift, result.executed_success-result.matched_shuffled_success
    ):
        raise ValueError('Command lift disagrees with saved success rates or canonical survey')
    for column in ['executed_success', 'matched_shuffled_success']:
        if column+'_command' in result and not np.allclose(result[column], result[column+'_command']):
            raise ValueError(f'Command summary disagrees with canonical survey: {column}')
    panel_path = base/'fullrange_plan/kernel_fullrange_manifest.tsv'
    g.sources[str(panel_path.relative_to(ROOT))] = sha(panel_path)
    panels = pd.read_csv(panel_path, sep='\t').rename(columns={'checkpoint_frames': 'frames'})
    panels = result[keys].merge(panels[keys+['panel']], on=keys, validate='one_to_one')
    result = result.merge(panels, on=keys, validate='one_to_one')
    if result.panel.nunique() != 1 or not (result.panel == result.representation_panel).all():
        raise ValueError('Replay panel provenance disagrees; do not infer it from family names')
    if not (result.replay_observations == result.representation_observations).all():
        raise ValueError('Replay observation counts disagree')
    # Verify the archived summary formula on every locally staged seed-99 map.
    result['raw_summary_crosscheck'] = False
    for path in sorted((base/'common_fields/replay_reduced/raw').rglob('place_fields.npz')):
        with np.load(path, allow_pickle=False) as raw:
            name, age = checkpoint_identity(str(raw['checkpoint']), set(result.run_name))
            selected = result[(result.run_name == name) & (result.frames == age)]
            if len(selected) != 1:
                raise ValueError(f'No exact command-summary identity for {path}')
            row = selected.iloc[0]
            eligible, _, mass, _, _ = multilevel_field_structure(
                raw['rate_maps'], raw['occupancy'], raw['active_fraction'])
            if not np.isclose(mass[:, eligible].mean(), row.mean_threshold_dominance, atol=1e-7):
                raise ValueError(f'Mean-threshold dominance disagrees: {path}')
            if str(raw['observation_panel']) != row.panel or int(eligible.sum()) != row.field_eligible_units:
                raise ValueError(f'Raw replay metadata disagrees: {path}')
            result.loc[selected.index, 'raw_summary_crosscheck'] = True
            g.sources[str(path.relative_to(ROOT))] = sha(path)
    if result.raw_summary_crosscheck.sum() != 4:
        raise ValueError('Expected four local seed-99 raw-summary crosschecks')
    result['dominance_source'] = str(replay_path.relative_to(ROOT))
    result['command_source'] = str(command_path.relative_to(ROOT))
    result.to_csv(g.out/'command_dominance_per_run.csv', index=False)
    stats = []
    for family, groups in COMMAND_GROUPS.items():
        selected = result[result.family == family]
        first, second = groups
        seed_pairs = selected.pivot(index='seed', columns='condition', values='command_lift')
        differences = seed_pairs[first]-seed_pairs[second]
        stats.append({'family': family, 'n': len(selected), 'variants': 2, 'seeds_per_variant': 3,
            'spearman_rho': rho(selected, 'mean_threshold_dominance', 'command_lift'),
            'first_condition': first, 'second_condition': second,
            'first_mean_lift_pp': float(seed_pairs[first].mean()*100),
            'second_mean_lift_pp': float(seed_pairs[second].mean()*100),
            'paired_lift_difference_pp': {str(k): float(v*100) for k, v in differences.items()},
            'first_higher_seeds': int((differences > 0).sum()),
            'all_lifts_positive': bool((selected.command_lift > 0).all())})
    (g.out/'command_dominance_statistics.json').write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)+'\n')
    return result


def axes_grid():
    fig, axes = plt.subplots(2, 2, figsize=tuple(v/25.4 for v in LARGE_MM))
    fig.subplots_adjust(left=.115, right=.96, bottom=.17, top=.80, wspace=.50, hspace=1.15)
    return fig, axes.ravel()


def scatter(g: Gallery, key: str, ax, data: pd.DataFrame, x: str, y: str,
            xlabel: str, ylabel: str, title: str, groups: dict, colors: dict,
            factor=1., correlation=True):
    valid = data.dropna(subset=[x, y])
    for index, (condition, label) in enumerate(groups.items()):
        selected = valid[valid.condition == condition]
        ax.scatter(selected[x], selected[y]*factor, s=78, marker=['o', 's', '^', 'D', 'P', 'X'][index % 6],
                   color=colors.get(condition, '#555555'), edgecolor='white', linewidth=.5, alpha=.85)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    r = rho(valid, x, y)
    suffix = f'n={len(valid)}' + (f' · ρ={r:.2f}' if correlation and np.isfinite(r) else '')
    # One title line leaves room for the next row's labels at fixed poster size.
    display_title = title.split(' ', 1)[0]+'  '+suffix if correlation else title
    ax.set_title(display_title, loc='left', pad=12)
    ax.grid(color='#eeeeee', zorder=0)
    provenance = {'protocol': valid.protocol.iloc[0] if len(valid) else '',
                  'model_frames': int(valid.frames.iloc[0]) if len(valid) else '',
                  'capacity': 16}
    g.record(key, title, x, valid, x, **provenance,
             unit='mean across eligible DG units', x_axis=True)
    g.record(key, title, y, valid, y, factor=factor, **provenance, y_axis=True)
    if y in ('exploration_coverage', 'prospective_success', 'return_40_mobile'):
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
    if x == 'dominance':
        ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1])
    elif x.startswith('concentration'):
        ax.set_xlim(.65, 1); ax.set_xticks([.7, .8, .9, 1])
    else:
        ax.set_xlim(left=0)


def legend(fig, groups: dict, colors: dict, *, two_rows=False):
    handles = [Line2D([], [], marker=['o','s','^','D','P','X'][i % 6], color=colors[k],
                      ls='', markersize=10, label=v) for i, (k, v) in enumerate(groups.items())]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, 1.01),
               ncol=3 if two_rows else len(groups), frameon=False, handletextpad=.25, columnspacing=.6)


def core_candidates(g: Gallery, runs: pd.DataFrame, flat: pd.DataFrame):
    core = runs[(runs.family == 'Corrected core') & runs.code.isin(COLORS)
                & runs.protocol.eq('frozen_latest_archived_probe')].copy()
    core['condition'] = core.code
    key = '01_continuous_fields_core'
    fig, axes = scalar_axes(2)
    for ax, measure, label in zip(axes, ['concentration', 'dominance'],
                                  ['Concentration', 'Dominance']):
        dots(ax, core, measure, tuple(COLORS))
        for collection in ax.collections:
            collection.set_sizes([78])
        ax.set(title=label, ylim=(0, 1), yticks=[0, .5, 1])
        g.record(key, label.replace('\n', ' '), measure, core, measure,
                 protocol='frozen_10k', unit='mean of eligible DG units', capacity=16)
    g.finish(fig, Candidate(key, 'Continuous field structure', SMALL_MM, 'Compact replacement for strict counts',
        'C15 explores more with broader DG activity. Three seeds per condition.',
        'Points are run means across eligible DG units; black ticks average three training seeds. '
        'C is gain-invariant spatial concentration; D is minimum dominant-component mass across '
        '30/50/70% peak thresholds. Frozen 10k probes, 100,040,704-frame models, DG 16.',
        'One-condition contrasts are descriptive; maps came from each policy’s own trajectory. '
        'Concentration does not imply a single contiguous field.',
        'Does a usable internal goal require a compact field, or a predictable transition?'))
    key = '02_core_concentration_and_behavior'
    fig, axes = axes_grid()
    for ax, x, y, xl, yl, title, factor in zip(axes,
        ['concentration', 'dominance', 'concentration', 'dominance'],
        ['coverage_auc_terminal', 'coverage_auc_terminal', 'return_40_mobile', 'return_40_mobile'],
        ['Concentration', 'Dominance']*2,
        ['Coverage\nAUC', 'Coverage\nAUC', '40-step\nreturn', '40-step\nreturn'],
        ['a  Exploration', 'b  Exploration', 'c  Repeated paths', 'd  Repeated paths'], [1,1,1,1]):
        scatter(g, key, ax, core, x, y, xl, yl, title, COLORS, COLORS, factor)
    legend(fig, {c:c for c in COLORS}, COLORS)
    g.finish(fig, Candidate(key, 'Field structure versus behavior', LARGE_MM, 'Discussion candidate',
        'Broader exploration need not accompany sharper fields. Field shape and loops are measured in frozen probes.',
        'One point per training seed, three seeds each for C01/C05/C15. Shape and 40-step mobile returns '
        'use archived 10k frozen probes; coverage AUC is the final 10M training window for the same checkpoint. '
        'Returns require displacement <100 and path >500 within one contiguous segment. ρ is descriptive Spearman correlation.',
        'Nine seeds mix three designs. Training AUC and frozen paths have different observation protocols; '
        'unit concentration is conditioned on visited locations, and does not measure diversity across units.',
        'Can distributed landmark codes support exploration without conventional single place fields?'))


def cpd_candidates(g: Gallery, runs: pd.DataFrame):
    data = runs[(runs.family == 'CPD') & runs.protocol.eq('online_latest_saved_window')]
    groups = CPD_LABELS
    colors = dict(zip(groups, CPD_COLORS))
    if len(data) != 12 or data.condition.nunique() != 4 or not (data.frames == 75005952).all():
        raise ValueError('Expected four CPD variants with three seeds at the same age')
    key = '03_concentration_exploration_control'
    fig, axes = axes_grid()
    for ax, x, y, xl, yl, title in zip(axes,
        ['concentration', 'concentration', 'dominance', 'dominance'],
        ['exploration_coverage', 'prospective_success']*2,
        ['Concentration']*2+['Dominance']*2,
        ['Visited bins\n(fraction)', 'Target hits /\nattempts']*2,
        ['a  Exploration', 'b  Recorded control', 'c  Exploration', 'd  Recorded control']):
        scatter(g, key, ax, data, x, y, xl, yl, title, groups, colors)
    legend(fig, groups, colors)
    g.finish(fig, Candidate(key, 'Concentration versus exploration and control', LARGE_MM, 'Main continuous scatter candidate',
        'Within four CA3-feedback variants, neither concentration score orders recorded target hits strongly.',
        'DG 16, four variants × three training seeds, all 75,005,952 frames. Each point averages '
        'eligible DG units. Maps and visited coverage use a recent 100k training window; target-event '
        'hits/attempts are historical counters saved with that model. All y axes retain 0–1; C x axes '
        'show 0.65–1 for detail, D x axes 0–1. ρ is a descriptive seed-level rank association.',
        'Only four CPD variants have locally available maps; this is a availability-defined subset, '
        'not the full architectural survey. Target hits are internal DG events, not verified physical arrival. '
        'Coverage is nearly saturated and field sampling is policy-driven.',
        'Should field sharpness predict internal recognition, executable commands, or both?'))


def broad_dominance_candidate(g: Gallery, runs: pd.DataFrame):
    key = '04_dominance_across_variants'
    fig, axes = axes_grid()
    fig.text(.53, .97, 'CA3 feedback: top · Algorithm screen: bottom', ha='center', va='top')
    panel = 0
    for family, prefix in [('CPD', 'CA3 feedback'), ('Navigation8', 'Algorithm screen')]:
        data = runs[(runs.family == family) & runs.protocol.eq('online_latest_saved_window')]
        groups = (CPD_LABELS if family == 'CPD' else
                  dict((c, c.removeprefix('N8_').replace('_', ' ')) for c in data.condition.unique()))
        colors = {c: ('#0072B2' if family == 'CPD' else '#A6761D') for c in groups}
        for outcome, label in [('exploration_coverage', 'Visited bins\n(fraction)'),
                               ('prospective_success', 'Target hits /\nattempts')]:
            scatter(g, key, axes[panel], data, 'dominance', outcome, 'Dominance',
                    label, f'{"abcd"[panel]}  {prefix}', groups, colors)
            panel += 1
    g.finish(fig, Candidate(key, 'Dominance across architectural variants', LARGE_MM, 'Broader survey supplement',
        'Dominance–exploration ordering changes by family. Target-hit associations remain weak.',
        'CA3-feedback: four variants, 12 seeds. Algorithm screen: six variants, 18 seeds for coverage; '
        'four goal variants, 12 seeds for target hits. All DG 16 at 75,005,952 frames. Means use '
        'eligible units; shape markers identify variants within each family. This plot includes '
        'canonical unit CSVs where raw maps were unavailable; it measures component dominance, not occupied area.',
        'Six worker-reference seeds have no target-event counters and are omitted only from that panel. '
        'Different control outcomes and recent-map/historical-counter scopes limit between-family comparisons.',
        'Does one field per unit matter after the controller and goal interface are held fixed?'))


def command_candidate(g: Gallery, runs: pd.DataFrame):
    key = '05_concentration_command_comparisons'
    fig, axes = axes_grid()
    fig.text(.53, .97, 'Two variants per family · one seed per variant', ha='center', va='top')
    colors = {'DGP_C15_FIRST_JOINT_LEG':'#0072B2', 'DGP_C15_HIT_JOINT_LEG':'#D55E00',
              'SAT_C15_SRC_MON_FILM':'#0072B2', 'SAT_C15_ARR_MON_FILM':'#D55E00'}
    for col, family in enumerate(['DGP', 'Saturday']):
        data = runs[(runs.family == family) & runs.protocol.eq('common_replay_with_frozen_outcomes')]
        if len(data) != 2 or data.seed.nunique() != 1 or data.frames.nunique() != 1:
            raise ValueError('Expected two fixed-seed same-age command examples per family')
        groups = {c: c for c in colors if c in set(data.condition)}
        for row, (measure, label) in enumerate([('concentration','Concentration'),
                                               ('dominance','Dominance')]):
            scatter(g, key, axes[2*row+col], data, measure, 'executed_minus_shuffled', label,
                    'Command\nlift (p.p.)',
                    f'{"abcd"[2*row+col]}  {"DG policy" if family == "DGP" else "Credit assignment"}',
                    groups, colors, factor=100, correlation=False)
            axes[2*row+col].axhline(0, color='#777777', lw=1.1, ls='--')
            axes[2*row+col].set_ylim(-5, 35)
            axes[2*row+col].set_yticks([0, 15, 30])
    g.finish(fig, Candidate(key, 'Matched commands: exploratory examples', LARGE_MM, 'Optional, too few runs for a trend claim',
        'Higher concentration accompanies lower command benefit in both available pairs. One seed per variant.',
        'DG 16, seed 99, 75,038,720-frame models. Left: first-outcome (blue circle), '
        'target-hit (orange square). Right: source credit (blue circle), arrival credit (orange square). '
        'C and D are measured from the same 10,001-observation recorded replay panel across both families. '
        'Command benefit is executed success minus matched shuffled-command success '
        'at the same checkpoint, in percentage points.',
        'Only two variants × one seed per family; there is no rank coefficient or fitted trend. '
        'The pattern motivates discussion, not a conclusion that sharper fields impair control. '
        'Command success follows the saved intervention’s internal target-event rule.',
        'Is the useful representation property field sharpness, or controllable transitions between landmarks?'))


def replicated_command_candidates(g: Gallery, data: pd.DataFrame):
    key = '06_dominance_command_all_saved_seeds'
    fig, axes = plt.subplots(1, 2, figsize=tuple(v/25.4 for v in ROW_MM))
    fig.subplots_adjust(left=.12, right=.96, bottom=.43, top=.79, wspace=.50)
    for ax, (family, groups), letter in zip(axes, COMMAND_GROUPS.items(), 'ab'):
        selected = data[data.family == family]
        colors = dict(zip(groups, ['#0072B2', '#D55E00']))
        scatter(g, key, ax, selected, 'mean_threshold_dominance', 'command_lift',
                'Mean dominance', 'Lift\n(p.p.)', f'{letter}  {family}', groups, colors,
                factor=100, correlation=False)
        ax.set_title(f'{letter}  {"DG policy" if family == "DGP" else "Credit assignment"}',
                     loc='left', pad=12)
        ax.set(xlim=(0, 1), xticks=[0, .5, 1], ylim=(0, 35), yticks=[0, 15, 30])
        ax.axhline(0, color='#777777', lw=1.1, ls='--')
        # The caption names both marker identities; an in-panel legend at 30 pt
        # would cover points on this compact physical canvas.
    g.finish(fig, Candidate(key, 'Dominance and commands: all saved seeds', ROW_MM,
        'Replicated discussion supplement',
        'Three seeds per variant. Blue circles: First/Source; orange squares: Hit/Arrival. '
        'Mean-threshold dominance gives different control ordering across families.',
        'Each point is one DG-16 model at 75,038,720 frames: two variants × three seeds per family. '
        'All representations use the same 10,001-observation replay. Mean dominance averages '
        'largest-component mass across eligible units and 30/50/70% peak thresholds; it differs '
        'from minimum-threshold D in candidates 01–05. Lift is executed minus matched shuffled target-event '
        'success in percentage points. Blue circle: First or Source; orange square: Hit or Arrival.',
        'Two selected variants per family do not constitute a full architectural sweep. '
        'Descriptive rank associations are −0.31 (DG policy) and +0.77 (credit assignment), each n=6. '
        'Seed replication removes the common negative pattern seen in candidate 05; no causal field-shape claim follows.',
        'Which representation properties predict controllability after accounting for architecture and seed?'))
    key = '07_command_lift_seed_pairs'
    fig, axes = scalar_axes(2)
    for ax, (family, groups), letter in zip(axes, COMMAND_GROUPS.items(), 'ab'):
        selected = data[data.family == family]
        labels = tuple('Arr.' if label == 'Arrival' else label for label in groups.values())
        paired(ax, selected, 'command_lift', tuple(groups), labels, factor=100)
        for collection in ax.collections:
            collection.set_sizes([78])
        ax.set(title=f'{letter}  {"DG policy" if family == "DGP" else "Credit"}',
               ylabel='Lift\n(p.p.)' if letter == 'a' else '', ylim=(0, 35), yticks=[0, 15, 30])
        ax.axhline(0, color='#777777', lw=1.1, ls='--')
        g.record(key, family, 'command_lift', selected, 'command_lift', factor=100,
                 protocol='frozen_matched_commands', model_frames=75038720, capacity=16,
                 unit='percentage points', sample_unit='one training seed')
    g.finish(fig, Candidate(key, 'Commands show target specificity', SMALL_MM,
        'Preferred command result',
        'Executed commands beat matched shuffled targets in all 12 models. First-outcome lift exceeds target-hit lift in all three seed pairs.',
        'DG 16 at 75,038,720 frames. Points are three training seeds per variant; gray lines connect '
        'matching seed IDs and black bars show means. Lift: executed target-event success minus '
        'matched shuffled-command success. DG-policy means: First 24.5 versus Hit 15.2 p.p.; '
        'credit means: Source 22.1 versus Arrival 20.7 p.p. Zero means no command advantage.',
        'The shuffled comparator is matched retrospectively from completed trials, not identical-start '
        'physical navigation. Trial eligibility and deadline rules follow the saved intervention; '
        'ordered-pair coverage is incomplete. Three seed pairs support a descriptive comparison, not a significance claim.',
        'Do learned commands identify specific internal targets, and how should this translate to physical navigation?'))


def compose_gallery(g: Gallery):
    # Put the replicated command result before the optional seed-99 example.
    for page, candidates in enumerate([g.candidates[:4], list(reversed(g.candidates[4:]))]):
        root = ET.Element(f'{{{SVG}}}svg', {'width':'841mm','height':'740mm','viewBox':'0 0 841 740'})
        ET.SubElement(root, f'{{{SVG}}}rect', {'width':'841','height':'740','fill':'white'})
        text(root, f'title-{page}', 20, 32, ['Continuous place-field candidates'], size=48, bold=True)
        text(root, f'legend-{page}', 20, 53,
             ['Original physical sizes • plot text 30 pt • captions 40 pt • DG 16'], size=30)
        for i, c in enumerate(candidates):
            x, y = 20+(i % 2)*418, 85+(i//2)*330
            lines = wrap_lines(c.key[:2]+'. '+c.title, 383, 48, bold=True)
            text(root, c.key+'-title', x, y, lines, size=48, step=19, bold=True)
            py = y+19*(len(lines)-1)+12
            height = embed(root, g.out/f'{c.key}.svg', c.key, x+(383-c.size_mm[0])/2, py, c.size_mm[0])
            caption = wrap_lines(c.short_caption, 383, 40)
            if py+height+19+(len(caption)-1)*18 > y+317:
                raise ValueError(f'Gallery card overflow: {c.key}')
            text(root, c.key+'-caption', x, py+height+19, caption, size=40, step=18)
        path = g.out/f'gallery_continuous_{page+1}.svg'
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        g.figures.append(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    font = style(); g = Gallery(args.out)
    poster = ROOT/'05_plans/poster_20260929/layout_versions/Bernstein2026_A_system_transfer.svg'
    poster_hash = sha(poster)
    survey = g.read(SCATTER)
    # The frozen summary omits policy_id; all selected checkpoint paths are p0.
    survey['policy_id'] = survey.policy_id.fillna(0).astype(int)
    maps, availability = collect_maps(g, survey)
    units = collect_unit_tables(g, survey, maps)
    runs = aggregate_units(units, survey)
    flat = g.read(TABLE)
    core_names = {}
    for row in flat.itertuples():
        name, _ = checkpoint_identity(row.checkpoint, set(survey.run_name))
        core_names[name] = row.condition
    runs['code'] = runs.run_name.map(core_names)
    runs = runs.merge(flat[['condition','seed','checkpoint_frames','coverage_auc_terminal',
                           'action_sensitivity_terminal','return_40_mobile']].rename(
                             columns={'condition':'code', 'checkpoint_frames':'frames'}),
                      how='left', on=['code','seed','frames'], validate='many_to_one')
    trajectory = g.read(RESULTS/'c15_variants/trajectory_per_run.csv')
    # Keep the prior run/protocol trajectory evidence available for interpretation.
    protocol_lookup = {'online_100k_training_window':'online_latest_saved_window',
                       'frozen_10k_policy_probe':'frozen_latest_archived_probe'}
    trajectory['protocol'] = trajectory.protocol.map(protocol_lookup)
    trajectory = trajectory.rename(columns={'evaluation_age':'frames','return_40_mobile':'trajectory_return40'})
    runs = runs.merge(trajectory[['run_name','frames','protocol','trajectory_return40']],
                      how='left', on=['run_name','frames','protocol'], validate='many_to_one')
    runs['return_40_mobile'] = runs.return_40_mobile.fillna(runs.trajectory_return40)
    units.to_csv(args.out/'per_unit.csv', index=False)
    runs.to_csv(args.out/'per_run.csv', index=False)
    availability.to_csv(args.out/'raw_map_availability.csv', index=False)
    stats = correlation_tables(runs)
    stats.to_csv(args.out/'correlations.csv', index=False)
    # Every survey run has an explicit measurement-availability row.
    available = runs[KEYS+['source_kind']].assign(continuous_dominance_available=True,
                                                concentration_available=runs.concentration.notna())
    full = survey[survey.dg_units == 16].merge(available, on=KEYS, how='left', validate='one_to_one')
    for c in ['continuous_dominance_available','concentration_available']:
        full[c] = full[c].eq(True)
    full[KEYS+['condition','family','geometry_group','continuous_dominance_available',
              'concentration_available','source_kind']].to_csv(args.out/'survey_availability.csv', index=False)
    core_candidates(g, runs, flat)
    cpd_candidates(g, runs)
    broad_dominance_candidate(g, runs)
    command_candidate(g, runs)
    commands = command_summary_data(g, survey)
    replicated_command_candidates(g, commands)
    compose_gallery(g)
    pd.DataFrame(g.points).to_csv(args.out/'plotted_points.csv', index=False)
    for path in [Path(__file__), ROOT/'hpc_runs/intrmotiv_study/field_concentration.py',
                 ROOT/'06_experiments/analyze_place_field_manifest.py']:
        g.sources[str(path.relative_to(ROOT))] = sha(path)
    manifest = {'schema':'intrmotiv/continuous-field-candidates/v1', 'workflow_version':WORKFLOW_VERSION,
        'study_schema':'intrmotiv/study/v1', 'study_sha256_note':'Retained per-run in per_run.csv; no new study',
        'font_path':font, 'plot_font_pt':30, 'caption_font_pt':40, 'marker_area_pt2':78,
        'map_definition':'unsmoothed, occupancy-corrected rate_maps; equal spatial-bin weights',
        'support':'occupancy >=1; same eligibility with >=5 support sensitivity',
        'aggregation':'equal-weight mean of canonically eligible units within each run',
        'concentration_formula':'p = r/sum(r); Aeff = 1/sum(p*p); C=(N-Aeff)/(N-1)',
        'dominance_definition':'minimum largest-component share at 30/50/70% peak thresholds',
        'mean_threshold_dominance_definition':'mean largest-component share across thresholds and eligible units; separate from minimum D',
        'command_summary_models':len(commands),
        'command_shared_replay_panel':str(commands.panel.iloc[0]),
        'sources':g.sources, 'candidates':[asdict(c) for c in g.candidates],
        'poster_sha256_before':poster_hash,
        'numpy_version':np.__version__, 'pandas_version':pd.__version__,
        'matplotlib_version':plt.matplotlib.__version__}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    verify_and_render(g, poster, poster_hash)
    print('Available means:', runs.groupby(['protocol','family']).size().to_dict())
    print('Outputs:', args.out)


if __name__ == '__main__':
    main()
