"""Render optional poster claims from pinned local summaries, without editing the poster.

Use SF_git Python. Physical sizes and editable typography reuse the current A0
composer; comparisons retain individual downstream/training seeds. The source
DG in each transfer architecture is one fixed representation, not three
independent pretraining replicates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from compose_a0_poster_20260929 import (
    ROOT, RESULTS, SCATTER, TABLE, SVG, COLORS, FAMILY_COLORS,
    PLOT_FONT_PT, BODY_FONT_PT, HEADING_FONT_PT, PT_MM,
    dots, embed, plt, np, pd, save, sha, style, text, wrap_lines,
)
from matplotlib.lines import Line2D

DATA = ROOT / '06_experiments/data/poster_missing_analyses_20260926'
DEFAULT_OUT = ROOT / '05_plans/poster_20260929/candidate_plots'
SMALL_MM = (191.5, 76.2)
ROW_MM = (383., 76.2)
LARGE_MM = (383., 165.1)
TRANSFER_SEEDS = (42, 1234, 9999)
TRANSFER_AGES = {'D50': 75_022_336, 'D51': 75_038_720}
ARM_COLORS = ('#0072B2', '#D55E00')


@dataclass
class Candidate:
    key: str
    title: str
    size_mm: tuple[float, float]
    rank: str
    short_caption: str
    caption: str
    limitation: str
    question: str


class Gallery:
    """Keep plotted values, source fingerprints, and editorial claims together."""

    def __init__(self, out: Path):
        self.out = out
        self.sources = {}
        self.points = []
        self.candidates = []
        self.figures = []

    def read(self, path: Path) -> pd.DataFrame:
        self.sources[str(path.relative_to(ROOT))] = sha(path)
        return pd.read_csv(path)

    def record(self, candidate: str, panel: str, metric: str, data: pd.DataFrame,
               column: str, factor=1., **metadata) -> None:
        for _, row in data.iterrows():
            self.points.append({
                'candidate': candidate, 'panel': panel, 'metric': metric,
                'condition': row.get('condition', ''),
                'seed': int(row.seed) if pd.notna(row.seed) else '',
                'raw_value': float(row[column]), 'display_factor': factor,
                'displayed_value': float(row[column]) * factor,
                'cue': row.get('cue', ''), 'training_age_m': row.get('age_m', ''),
                'run_name': row.get('run_name', ''), 'family': row.get('family', ''),
                'spatial_score': row.get('spatial_information', ''),
                'dominant_mass_cutoff': row.get('dominant_mass_cutoff', ''),
                'trials': row.get('trials', ''),
                'study_schema': row.get('study_schema', ''),
                'workflow_version': row.get('workflow_version', ''),
                'study_sha256': row.get('study_sha256', ''),
                **metadata,
            })

    def finish(self, fig, candidate: Candidate) -> None:
        # No tight bounding box: Inkscape must receive the intended print size.
        path = self.out / f'{candidate.key}.svg'
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        outside = []
        for artist in fig.findobj(match=plt.Text):
            if not artist.get_visible() or not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
                outside.append(artist.get_text())
        if outside:
            raise ValueError(f'{candidate.key}: text outside figure: {outside}')
        save(fig, path)
        root = ET.parse(path).getroot()
        root.set('width', f'{candidate.size_mm[0]}mm')
        root.set('height', f'{candidate.size_mm[1]}mm')
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        self.figures.append(path)
        self.candidates.append(candidate)


def scalar_axes(count=2):
    size = SMALL_MM if count == 2 else ROW_MM
    fig, axes = plt.subplots(1, count, figsize=tuple(v / 25.4 for v in size), layout='constrained')
    return fig, axes


def paired(ax, data, key, conditions, labels, factor=1., single_source=False):
    """Seed-paired points and mean bars; one fixed source DG is drawn once."""
    values = data.pivot(index='seed', columns='condition', values=key).sort_index()
    if values.isna().any().any() or len(values) != 3:
        raise ValueError(f'Expected three complete seed pairs: {key}')
    offsets = np.array([-.12, 0, .12])
    if not single_source:
        for offset, (_, row) in zip(offsets, values.iterrows()):
            ax.plot([offset, 1+offset], row[list(conditions)].to_numpy(float)*factor,
                    color='#aaaaaa', lw=1.1, zorder=1)
    for i, condition in enumerate(conditions):
        y = values[condition].to_numpy(float)*factor
        x = i+offsets
        if single_source and i == 0:
            if np.ptp(y) > 1e-10:
                raise ValueError('Frozen source representation unexpectedly differs across seeds')
            x, y = np.array([0.]), y[:1]
        ax.scatter(x, y, s=56, marker=('o', 's')[i], color=ARM_COLORS[i], zorder=3)
        ax.plot([i-.19, i+.19], [y.mean()]*2, color='black', lw=2.2, zorder=4)
    ax.set(xticks=[0, 1], xticklabels=labels, xlim=(-.5, 1.5))
    ax.grid(axis='y', color='#e5e5e5', zorder=0)


def heldout(g: Gallery, architecture: str, developmental=False):
    folder = ('heldout_50m' if developmental else
              {'D50': 'heldout_d50_75m_latest', 'D51': 'heldout_d51_75m'}[architecture])
    path = DATA / folder / 'analysis/heldout_paired_trials.csv'
    raw = g.read(path)
    raw = raw[raw.architecture == architecture].copy()
    if raw.duplicated(['seed', 'requested_seed']).any():
        raise ValueError('Duplicate matched-reset trial')
    if tuple(sorted(raw.seed.unique())) != TRANSFER_SEEDS:
        raise ValueError('Wrong transfer training seeds')
    expected_age = 50_003_968 if developmental else TRANSFER_AGES[architecture]
    for arm in ('source', 'random'):
        if not (raw[f'checkpoint_frames_{arm}'] == expected_age).all():
            raise ValueError('Mixed checkpoint ages in heldout comparison')
    for key in ('engine_seed', 'number_instruction', 'reward_center_x', 'reward_center_y',
                'policy_seed', 'start_x', 'start_y'):
        if not (raw[f'{key}_source'] == raw[f'{key}_random']).all():
            raise ValueError(f'Unmatched trial field: {key}')
    if not (raw.groupby('seed').size() == 40).all():
        raise ValueError('Expected 40 paired reset trials per training seed')
    rows = []
    for seed, group in raw.groupby('seed'):
        for arm in ('source', 'random'):
            rows.append({'condition': arm, 'seed': seed, 'success': group[f'{arm}_success'].mean(),
                         'trials': len(group), 'checkpoint_frames': expected_age})
    return pd.DataFrame(rows), raw, path


def transfer_candidates(g: Gallery):
    heldouts, trials, replays = {}, {}, {}
    fig, axes = scalar_axes(4)
    key = '01_transfer_spatial_score_and_success'
    for i, architecture in enumerate(('D50', 'D51')):
        stem = {'D50': 'd50_75m_latest', 'D51': 'd51_75m'}[architecture]
        path = DATA / f'{stem}_allcue_replay_derived.csv'
        replay = g.read(path)
        if not (replay.checkpoint_frames == TRANSFER_AGES[architecture]).all():
            raise ValueError('Wrong common-replay checkpoint')
        # Exact condition labels are declared here; no run-name metadata parsing.
        replay['condition'] = replay.condition.map({
            f'CR5C_{architecture}_W_SOURCE_DG': 'source',
            f'CR5C_{architecture}_W_RAND_DG': 'random',
        })
        if replay.condition.isna().any():
            raise ValueError('Unknown transfer condition')
        replays[architecture] = replay
        heldouts[architecture], trials[architecture], trial_path = heldout(g, architecture)
        ax = axes[2*i]
        paired(ax, replay, 'active_unit_mean_si_bits', ('source', 'random'), ('Src', 'Rnd'),
               single_source=True)
        ax.set(title=f'{chr(65+2*i)}  {architecture}\nSpatial score', ylim=(0, .08), yticks=[0, .04, .08])
        displayed_replay = pd.concat([replay[replay.condition == 'source'].iloc[:1].assign(seed=np.nan),
                                      replay[replay.condition == 'random']])
        g.record(key, architecture+' score', 'spatial score', displayed_replay, 'active_unit_mean_si_bits',
                 protocol='identical all-five-cue observation/action history', source=str(path.relative_to(ROOT)),
                 checkpoint_frames=TRANSFER_AGES[architecture], dg_units=64,
                 source_replication='one frozen source representation per architecture')
        ax = axes[2*i+1]
        paired(ax, heldouts[architecture], 'success', ('source', 'random'), ('Src', 'Rnd'), 100)
        ax.set(title=f'{chr(66+2*i)}  {architecture}\nSuccess (%)', ylim=(0, 100), yticks=[0, 50, 100])
        g.record(key, architecture+' success', 'heldout physical success', heldouts[architecture], 'success', 100,
                 protocol='40 paired resets per downstream seed, 1800 decisions',
                 source=str(trial_path.relative_to(ROOT)), checkpoint_frames=TRANSFER_AGES[architecture], dg_units=64)
    g.finish(fig, Candidate(key, 'Higher spatial scores do not ensure transfer gains', ROW_MM, 'First choice',
        'DG 64; one source representation per architecture. Three downstream seeds; 40 matched resets each.',
        'Frozen pretrained source DG versus calibrated random DG, with fresh downstream worker, reward manager, '
        'and empty graph in both arms. A/C: common five-cue replay spatial score, with one source point and '
        'three random representations. B/D: per-training-seed physical reward success over 40 exactly matched '
        'reset trials, 1800-decision horizon. Src/circles: pretrained source; Rnd/squares: calibrated random; black marks: means. '
        'Heldout success is 50.0% versus 58.3% in D50 and 47.5% versus 61.7% in D51.',
        'One pretrained source checkpoint per architecture is reused across downstream seeds. D50 success '
        'differences are -25.0, -7.5, +7.5 percentage points; D51 differences are -22.5, -10.0, -10.0. '
        'Spatial score also depends on activation amplitude. This is transfer within the same visual geometry, '
        'not generalization to a new visual environment or a causal mediation test.',
        'Which properties of a learned landmark code make it reusable for externally rewarded control?'))

    key = '02_transfer_training_reward'
    reward_path = DATA / 'reward_auc_windows_paired.csv'
    reward = g.read(reward_path)
    reward = reward[reward.horizon_frames == 75_000_000]
    fig, axes = scalar_axes()
    for ax, architecture in zip(axes, ('D50', 'D51')):
        group = reward[reward.architecture == architecture]
        data = pd.concat([group[['seed', col]].rename(columns={col: 'reward'}).assign(condition=arm)
                          for arm, col in [('source', 'source_mean_reward_auc'), ('random', 'random_mean_reward_auc')]])
        paired(ax, data, 'reward', ('source', 'random'), ('Src', 'Rnd'), 1000)
        ax.set(title=f'{architecture}  Reward\n/ step × 10³', ylim=(0, .4), yticks=[0, .2, .4])
        g.record(key, architecture, '0–75M reward AUC / horizon', data, 'reward', 1000,
                 protocol='canonical 0–75M scalar history', source=str(reward_path.relative_to(ROOT)), dg_units=64)
        if not (group.source_minus_random < 0).all():
            raise ValueError('Claim requires lower source reward in all paired seeds')
    g.finish(fig, Candidate(key, 'Pretrained DG accumulates less transfer reward', SMALL_MM, 'Strong compact alternative',
        'Source reward is lower in all six paired seeds over 0–75M frames. Lines connect downstream seeds.',
        'Mean reward per step from the canonical reward AUC divided by the 75M-frame horizon, displayed '
        '×1000. Three paired downstream training seeds per architecture; points are seeds, connecting lines '
        'preserve seed pairing, and black marks show means. Src/circles: source; Rnd/squares: random. Source-minus-random differences average '
        '-0.074 × 10⁻³ in D50 and -0.058 × 10⁻³ in D51. Both arms freeze DG 64.',
        'This measures reward collected during transfer training, not heldout trial success. Early reward '
        'differences are mixed; the claim concerns the full 0–75M history and the two selected fixed sources.',
        'Could an intrinsic code favor familiar transitions that are unhelpful for the reward task?'))

    key = '03_transfer_by_reward_cue'
    fig, axes = plt.subplots(1, 2, figsize=tuple(v/25.4 for v in LARGE_MM), sharey=True)
    fig.subplots_adjust(left=.17, right=.985, top=.86, bottom=.34, wspace=.22)
    cue_counts = {}
    for ax, architecture in zip(axes, ('D50', 'D51')):
        raw = trials[architecture]
        counts = raw.groupby('cue').size()
        cue_counts[architecture] = counts.to_dict()
        seed_cues = raw.groupby(['seed', 'cue']).source_minus_random_success.mean().reset_index()
        seed_cues['trials'] = raw.groupby(['seed', 'cue']).size().to_numpy()
        seed_cues['condition'] = 'source minus random'
        for cue, group in seed_cues.groupby('cue'):
            y = group.sort_values('seed').source_minus_random_success.to_numpy()*100
            ax.scatter(cue + np.array([-.10, 0, .10]), y, s=65, color=ARM_COLORS[0], zorder=3)
            ax.plot([cue-.22, cue+.22], [y.mean()]*2, color='black', lw=2.2, zorder=4)
        ax.axhline(0, color='#777777', lw=1.5, ls='--')
        ax.set(title=architecture, xticks=list(counts.index),
               xticklabels=[f'{cue}\n({n})' for cue, n in counts.items()],
               xlabel='Reward cue (n pairs)', ylim=(-105, 105), yticks=[-100, 0, 100])
        g.record(key, architecture, 'paired success difference by cue', seed_cues,
                 'source_minus_random_success', 100,
                 source=str((DATA / {'D50':'heldout_d50_75m_latest','D51':'heldout_d51_75m'}[architecture] /
                             'analysis/heldout_paired_trials.csv').relative_to(ROOT)),
                 protocol='within-seed matched reset trials; each cue separately', dg_units=64)
    axes[0].set_ylabel('Success difference\n(pp)')
    fig.text(.5, .035, 'Dots: training seeds; black marks: means', ha='center', fontsize=PLOT_FONT_PT)
    g.finish(fig, Candidate(key, 'Transfer performance varies across reward cues', LARGE_MM, 'Optional transfer detail',
        'Paired success differences by cue. Cue 1 has only six trial pairs; negative values favor random DG.',
        'At the 75M endpoints, dots show each downstream training seed’s source-minus-random physical-success '
        'difference for each reward cue, in percentage points (pp). Black marks show means. The same resets '
        'and reward sites are paired between arms. Per architecture there are 6/30/30/36/18 trial pairs for '
        'cues 1–5, distributed equally over the three training seeds.',
        'Cue sample sizes are unequal. Overall success in candidate 01 weights trials as recorded, not cues '
        'equally. Cue 1 is particularly uncertain; these cue differences do not identify a representation mechanism.',
        'Does a useful landmark code need uniform goal coverage, or can a few poorly represented sites dominate failure?'))

    key = '04_transfer_development'
    fig, axes = scalar_axes()
    for ax, architecture in zip(axes, ('D50', 'D51')):
        earlier, _, earlier_path = heldout(g, architecture, developmental=True)
        records = []
        for age, data in [(50, earlier), (75, heldouts[architecture])]:
            wide = data.pivot(index='seed', columns='condition', values='success')
            for seed, row in wide.iterrows():
                records.append({'condition': 'source minus random', 'seed': seed, 'age_m': age,
                                'difference': row.source-row.random})
        data = pd.DataFrame(records)
        for seed, rows in data.groupby('seed'):
            ax.plot(rows.age_m, rows.difference*100, color='#999999', lw=1.1, marker='o', markersize=np.sqrt(56), zorder=2)
        means = data.groupby('age_m').difference.mean()
        ax.plot(means.index, means*100, color='black', lw=2.2, marker='_', markersize=16, zorder=4)
        ax.axhline(0, color='#777777', ls='--', lw=1.5)
        ax.set(title=f'{architecture}  Δ\nSuccess (pp)', xticks=[50, 75], xlim=(46, 79),
               ylim=(-40, 40), yticks=[-40, 0, 40])
        g.record(key, architecture, 'source minus random heldout success', data, 'difference', 100,
                 source=str(earlier_path.relative_to(ROOT))+' and corresponding 75M paired trials',
                 protocol='same 40-reset protocol at 50M and 75M', dg_units=64)
    g.finish(fig, Candidate(key, 'D51’s transfer deficit grows by 75M', SMALL_MM, 'Optional learning dynamics',
        'Horizontal axis: 50M and 75M training frames. Lines track paired success differences for each seed.',
        'Source-minus-random heldout success in percentage points at 50,003,968 frames and the 75M '
        'endpoints. Gray lines track the three downstream training seeds; black marks show means. '
        'D50 changes from +0.8 to -8.3 pp, D51 from -0.8 to -14.2 pp. In D51 the paired difference '
        'declines in every seed. Evaluations use the same 40-reset protocol and 1800-decision horizon.',
        'Two checkpoints describe developmental change rather than a full learning curve. The 50M snapshot '
        'does not replace the completed 75M endpoint. D50 remains mixed across seeds.',
        'When should we evaluate transfer: early adaptation, accumulated reward, or final heldout control?'))


def architecture_candidates(g: Gallery):
    key = '05_predictor_loops_and_coverage'
    path = RESULTS / 'c15_variants/trajectory_per_run.csv'
    raw = g.read(path)
    conditions = ('CPD_C15_GATE_ACT_DIR', 'CPD_C15_GATE_ACT_DIR_GOAL')
    data = raw[raw.condition.isin(conditions) & (raw.protocol == 'frozen_10k_policy_probe') &
               (raw.evaluation_age == 75_038_720)].copy()
    fig, axes = scalar_axes()
    for ax, metric, title in zip(axes, ('return_20_mobile', 'visited_fraction'),
                               ('A  Return (%)\n20 decisions', 'B  Visited\nbins (%)')):
        paired(ax, data, metric, conditions, ('None', 'Goal'), 100)
        ax.set(title=title, ylim=(0, 100), yticks=[0, 50, 100])
        g.record(key, title, metric, data, metric, 100, source=str(path.relative_to(ROOT)),
                 protocol='frozen 10k-decision policy probe', checkpoint_frames=75_038_720, dg_units=16)
    g.finish(fig, Candidate(key, 'A goal predictor accompanies more loops', SMALL_MM, 'Good discussion candidate',
        'Goal predictor: more returns and less coverage in every paired seed. DG 16; three seeds.',
        'Matched CPD GATE ACT DIR designs without (None) and with (Goal) a goal predictor at '
        '75,038,720 frames. Frozen 10k-decision probes; dots are seeds 8/99/123, lines join training-seed '
        'pairs, black marks show means. Mobile-window 20-decision returns increase from 62.4% to 83.2%; '
        'visited arena-bin fraction decreases from 69.3% to 55.3%. Both directions hold in all three seeds.',
        'Probes match training age, environment and length, but not stochastic trajectories or starts. '
        'This predictor ablation is inside an already goal-conditioned controller; it is not an ablation '
        'of goal conditioning itself. Stabilizing familiar cycles is a hypothesis.',
        'Can better prediction reinforce familiar cycles rather than discovery?'))

    key = '06_graph_and_recorded_outcomes'
    path = RESULTS / 'c15_variants/control_per_run.csv'
    raw = g.read(path)
    conditions = ('DGP_C15_FIRST_JOINT_LEG', 'DGP_C15_HIT_JOINT_LEG')
    data = raw[raw.condition.isin(conditions) & (raw.protocol == 'online_100k_training_window') &
               (raw.frames == 75_005_952)].copy()
    fig, axes = scalar_axes()
    for ax, metric, title in zip(axes, ('prospective_success', 'graph_reachability'),
                               ('A  Event hits\n(%)', 'B  Reachable\npairs (%)')):
        paired(ax, data, metric, conditions, ('FIRST', 'HIT'), 100)
        ax.set(title=title, ylim=(0, 105), yticks=[0, 50, 100])
        g.record(key, title, metric, data, metric, 100, source=str(path.relative_to(ROOT)),
                 protocol='online 100k-observation window; graph counters retain accumulated history',
                 checkpoint_frames=75_005_952, dg_units=16)
    g.finish(fig, Candidate(key, 'Graph connectivity and recorded outcomes diverge', SMALL_MM, 'Use with explicit definitions',
        'DG 16; three paired seeds. FIRST/HIT also change the outcome rule, so event-hit definitions differ.',
        'DGP FIRST versus HIT JOINT LEG at the shared 75,005,952-frame online snapshot. Three '
        'training seeds, paired by seed; black marks show means. Recorded prospective successes/attempts '
        'are 58.9% versus 50.8%, while stored reachable ordered node pairs are 4.7% versus 96.1%. '
        'Every seed shows opposite orderings. FIRST/HIT are declared worker-outcome rules.',
        'FIRST/HIT change the success definition. This comparison illustrates different diagnostics, '
        'not improved physical control under one shared success rule. Graph counters are accumulated '
        'event summaries; connectivity alone does not establish executable edges.',
        'Which stored edges correspond to actions the agent can actually execute?'))

    key = '07_exploration_and_compact_fields'
    mono_path = RESULTS / 'flat_goal_comparison/mono_field_peak_counts.csv'
    mono = g.read(mono_path)
    method_path = RESULTS / 'flat_goal_comparison/mono_field_method.json'
    g.sources[str(method_path.relative_to(ROOT))] = sha(method_path)
    method = json.loads(method_path.read_text())
    if (not (mono.checkpoint_frames == 100_040_704).all() or not (mono.dg_units == 16).all()
            or mono.mono_units.sum() != 4 or method['min_dominant_component_mass_at_each_threshold'] != .8):
        raise ValueError('Mono-field source or primary criterion changed')
    matched = g.read(TABLE)
    conditions = ('C01', 'C05', 'C15')
    data = matched[matched.condition.isin(conditions)].merge(
        mono[['condition', 'seed', 'mono_units']], on=['condition', 'seed'], validate='one_to_one')
    fig, axes = scalar_axes()
    dots(axes[0], data, 'coverage_auc_terminal', conditions)
    axes[0].set(title='A  Coverage AUC\n(cells)', ylim=(0, 100), yticks=[0, 50, 100])
    dots(axes[1], data, 'mono_units', conditions)
    axes[1].set(title='B  Single-field\nunits /16', ylim=(-.08, 2), yticks=[0, 1, 2])
    g.record(key, 'A', 'terminal online coverage AUC', data, 'coverage_auc_terminal',
             source=str(TABLE.relative_to(ROOT)), protocol='terminal online summaries', checkpoint_frames=100_040_704, dg_units=16)
    g.record(key, 'B', 'strict mono-field count per 16 DG units', data, 'mono_units',
             source=str(mono_path.relative_to(ROOT)), protocol='archived 10k-decision frozen field probes',
             checkpoint_frames=100_040_704, dg_units=16)
    g.finish(fig, Candidate(key, 'Broader exploration need not mean compact fields', SMALL_MM, 'Connects directly to the main story',
        'C15 explores more; strict single-field counts remain low. Only 4 of 144 DG units qualify.',
        'Matched C01/C05/C15 seeds 8/99/123 at 100,040,704 frames. A: terminal online coverage AUC. '
        'B: strict mono-field count per 16 DG units from archived frozen policy probes. '
        'Dots: seeds; black marks: means. Across seeds the counts are 1/48, 2/48 and 1/48 units. '
        'The classifier requires ≥80% dominant-component mass at 30/50/70% peak thresholds, with '
        'occupancy-corrected smoothing and the saved activity eligibility checks.',
        'The two diagnostics use different observation protocols. Sparse strict counts depend on '
        'the field criterion and policy sampling; they do not establish an absence of spatial coding '
        'or equivalence across conditions. See candidate 07b for criterion sensitivity.',
        'Can broad or multifield landmark events still support navigation through sequence context?'))
    field_sensitivity(g)
    within_family(g)


def field_sensitivity(g: Gallery):
    key = '07b_field_criterion_sensitivity'
    path = RESULTS / 'flat_goal_comparison/mono_field_sensitivity.csv'
    data = g.read(path)
    fig, ax = plt.subplots(figsize=tuple(v/25.4 for v in LARGE_MM), layout='constrained')
    for condition, group in data.groupby('condition'):
        for seed, rows in group.groupby('seed'):
            ax.scatter(rows.dominant_mass_cutoff*100, rows.qualifying_units/rows.dg_units*100,
                       color=COLORS[condition], alpha=.5, s=56, zorder=2)
        means = group.groupby('dominant_mass_cutoff').qualifying_units.mean()/16*100
        ax.plot(means.index*100, means, color=COLORS[condition], lw=2.2,
                marker={'C01':'o','C05':'s','C15':'^'}[condition], markersize=np.sqrt(56),
                label=condition, zorder=3)
    ax.axvline(80, ls='--', color='#777777', lw=1.5)
    ax.set(title='Single-field classification depends on the criterion', xlabel='Dominant component mass cutoff (%)',
           ylabel='Qualifying units (%)', xlim=(17, 93), xticks=[20, 40, 60, 80],
           ylim=(-3, 103), yticks=[0, 50, 100])
    ax.legend(frameon=False)
    data['fraction'] = data.qualifying_units/data.dg_units
    g.record(key, 'A', 'qualifying units / all DG units', data, 'fraction', 100,
             source=str(path.relative_to(ROOT)), protocol='same frozen maps; varied dominant mass criterion',
             checkpoint_frames=100_040_704, dg_units=16)
    g.finish(fig, Candidate(key, 'Field counts depend on the compactness criterion', LARGE_MM, 'Supplement to 07; not a new main claim',
        'Same DG-16 maps and seeds; only the dominant mass cutoff varies. Dashed line: strict 80% criterion.',
        'The same C01/C05/C15 frozen maps are classified at dominant-component mass cutoffs 20–90%, '
        'retaining the same three peak thresholds and activity eligibility. Dots: individual seed '
        'fractions; colored lines: seed means; denominator: all 16 DG units. At 50% mass, totals are '
        '7/48, 16/48 and 7/48; at 80%, 1/48, 2/48 and 1/48.',
        'These are repeated classifications of the same maps, not independent experiments. '
        'A relaxed cutoff does not imply a compact single field. The strict criterion remains the '
        'declared primary definition; this panel shows why that choice matters.',
        'What field compactness would actually be necessary for a useful internal goal?'))


def within_family(g: Gallery):
    key = '08_architecture_conditioned_scatter'
    raw = g.read(SCATTER)
    data = raw[(raw.geometry_group == 'legacy_19x19') & (raw.dg_units == 16) &
               (raw.protocol == 'online_latest_saved_window')].dropna(
                   subset=['spatial_information', 'unique_peak_bins', 'prospective_success']).copy()
    if len(data) != 168 or data.run_name.duplicated().any():
        raise ValueError('Survey cohort differs from the latest poster')
    fig, axes = plt.subplots(1, 3, figsize=tuple(v/25.4 for v in LARGE_MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.985, top=.78, bottom=.43, wspace=.18)
    statistics = []
    for ax, family in zip(axes, ('ALL', 'CPD', 'DGP')):
        group = data if family == 'ALL' else data[data.family == family]
        rho = float(group.spatial_information.corr(group.prospective_success, method='spearman'))
        for name, rows in group.groupby('family'):
            ax.scatter(rows.spatial_information, rows.prospective_success*100, s=65, alpha=.8,
                       color=FAMILY_COLORS[name], marker={'CPD':'o','DGP':'s','DGC':'^',
                       'CPU cadence':'D','Navigation8':'P','Source credit':'X'}[name], zorder=3)
        title = {'ALL':'Pooled', 'CPD':'CA3 feedback', 'DGP':'DG policy'}[family]
        displayed_rho = 0. if abs(rho) < .005 else rho
        ax.set(title=f'{title}\nn={len(group)}, ρ={displayed_rho:.2f}', xlim=(0, .56), xticks=[0, .2, .4],
               ylim=(0, 100), yticks=[0, 50, 100], xlabel='Spatial score')
        ax.grid(color='#e5e5e5', zorder=0)
        ax.tick_params(axis='x', pad=12)
        g.record(key, family, 'recorded prospective success', group, 'prospective_success', 100,
                 source=str(SCATTER.relative_to(ROOT)), protocol='latest saved online window', dg_units=16)
        statistics.append({'family':family, 'n':len(group), 'rho':rho,
                           'minimum_frames':int(group.frames.min()), 'maximum_frames':int(group.frames.max())})
    axes[0].set_ylabel('Recorded hits (%)')
    labels = {'CPD':'CA3 feedback','DGP':'DG policy','DGC':'Goal interfaces',
              'CPU cadence':'Controllers','Navigation8':'Algorithm screen','Source credit':'Source credit'}
    markers = {'CPD':'o','DGP':'s','DGC':'^','CPU cadence':'D','Navigation8':'P','Source credit':'X'}
    handles = [Line2D([], [], marker=markers[name], color=FAMILY_COLORS[name], linestyle='none',
                      markersize=8, label=labels[name]) for name in sorted(data.family.unique())]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015), ncol=3,
               frameon=False, fontsize=PLOT_FONT_PT, handletextpad=.4, columnspacing=.9)
    (g.out / 'within_family_statistics.json').write_text(json.dumps(statistics, indent=2)+'\n')
    g.finish(fig, Candidate(key, 'The pooled spatial association weakens within families', LARGE_MM, 'Alternative framing for section 3',
        'Capacity fixed at DG 16. Strong pooled association; little monotonic relation within CPD or DGP.',
        'Same 168-run cohort as the current poster: latest saved online windows, legacy arena, DG 16. '
        'Left: all 56 variants; middle: CA3-feedback family (CPD, 81 runs); right: DG-policy family '
        '(DGP, 24 runs). Dots: runs, with the same family color/shape mapping as the current poster. '
        'Spearman correlations are 0.583, 0.094 and -0.001. The pooled panel supplies context rather '
        'than an additional independent dataset. No regression or trend line is fitted.',
        'Ages and success rules differ across families; target counters retain accumulated history while '
        'fields summarize recent policy observations. This is a descriptive survey, not independent '
        'replication of each architecture or evidence that spatial score has no within-family effect.',
        'Is the pooled correlation driven by the representation metric or by the surrounding controller design?'))


def compose_sheets(g: Gallery):
    """Two selection sheets; each imported figure remains at 100% print size."""
    sets = [('gallery_transfer', 'Candidate transfer claims', g.candidates[:4]),
            ('gallery_architecture', 'Candidate architecture claims',
             [c for c in g.candidates if c.key[:2] in {'05','06','07','08'} and not c.key.startswith('07b')])]
    for stem, heading, candidates in sets:
        root = ET.Element(f'{{{SVG}}}svg', {'width':'841mm','height':'710mm','viewBox':'0 0 841 710'})
        ET.SubElement(root, f'{{{SVG}}}rect', {'width':'841','height':'710','fill':'white'})
        text(root, stem+'-title', 20, 30, [heading], size=HEADING_FONT_PT, bold=True)
        text(root, stem+'-subtitle', 20, 53, ['Selection sheet • original panel sizes • plot text 30 pt • captions 40 pt'], size=PLOT_FONT_PT)
        for i, candidate in enumerate(candidates):
            x, y = (20 if i % 2 == 0 else 438), 82+(i//2)*310
            lines = wrap_lines(candidate.key[:2]+'. '+candidate.title, 383, HEADING_FONT_PT, bold=True)
            text(root, candidate.key+'-title', x, y, lines, size=HEADING_FONT_PT, step=19, bold=True)
            plot_y = y + (len(lines)-1)*19 + 10
            plot_x = x+(383-candidate.size_mm[0])/2
            h = embed(root, g.out/f'{candidate.key}.svg', candidate.key, plot_x, plot_y, candidate.size_mm[0])
            caption_lines = wrap_lines(candidate.short_caption, 383, BODY_FONT_PT)
            caption_y = plot_y+h+19
            if caption_y+(len(caption_lines)-1)*18 > y+287:
                raise ValueError(f'Caption exceeds sheet card: {candidate.key}')
            text(root, candidate.key+'-caption', x, caption_y, caption_lines, size=BODY_FONT_PT, step=18)
        target = g.out/f'{stem}.svg'
        ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
        g.figures.append(target)


def write_notes(g: Gallery, font_path: str, poster_hash: str):
    notes = ['# Optional poster plot gallery', '',
        'The current poster was not edited. Import individual SVGs into Inkscape at **100% physical size**: '
        'small pairs 191.5 × 76.2 mm, four-panel rows 383 × 76.2 mm, larger blocks 383 × 165.1 mm. '
        'Every plot label is 30 pt DejaVu Sans; seed markers are 56 pt² and scatter markers 65 pt², '
        'matching the latest poster additions. Selection-sheet explanations are 40 pt. SVG text is editable.', '',
        'Start with **01**, or use **02** for a compact transfer result. **05** is the clearest additional '
        'architecture ablation. **07** connects to C01/C05/C15. **08** can replace the existing section-3 '
        'framing. **03/04** are optional transfer detail; **06** needs the definition caveat beside it.', '',
        '![Transfer candidates](gallery_transfer.png)', '',
        '![Architecture candidates](gallery_architecture.png)', '',
        '| ID | Candidate | Physical size (mm) | Suggested use |',
        '| --- | --- | --- | --- |']
    for c in g.candidates:
        notes.append(f'| [{c.key.split("_")[0]}]({c.key}.svg) | {c.title} | {c.size_mm[0]:g} × {c.size_mm[1]:g} | {c.rank} |')
    notes.extend(['', '## Comparison setup', '',
        'Transfer keeps DG capacity at 64 and compares **frozen pretrained source DG** with a **frozen '
        'calibrated random DG**. Each architecture has one selected source checkpoint reused across '
        'three downstream seeds (42/1234/9999), not three independently pretrained DGs. Both arms '
        'start with a fresh worker, fresh reward manager and empty graph. The task has five invisible '
        'reward cells cued by a stable number instruction, within the original visual arena. '
        'D50 and D51 identify the two source architectures, not two goal locations in this five-cue task. '
        'Exact endpoints: D50 75,022,336 frames; D51 75,038,720. '
        'See [task and matched-control setup](../../../06_experiments/cued_reward5_transfer_20260925.md) '
        'and the [completed endpoint report](../../../06_experiments/results/A0_poster_analysis_20260926/batch_summary.md).', '',
        'The architecture comparisons use DG 16. Capacity is not pooled across the transfer and '
        'architecture candidates. Gray lines join training-seed pairs; black short marks show means. '
        'No significance tests or confidence intervals are implied by these three-seed displays. '
        'Spatial score includes activation amplitude and is not labeled as normalized spatial information. '
        'In the point table, the fixed source-representation point has a blank downstream seed; its three '
        'identical replay rows are drawn once. Source-row inputs remain pinned in the manifest.', ''])
    for c in g.candidates:
        notes.extend([f'## {c.key.split("_")[0]}. {c.title}', '',
                      f'[Editable SVG]({c.key}.svg) · [PNG preview]({c.key}.png)', '',
                      f'**Suggested caption:** {c.caption}', '', f'**Claim limit:** {c.limitation}', '',
                      f'**Visitor question:** {c.question}', ''])
    notes.extend(['## Regeneration and evidence', '',
        'Run from the repository root:', '', '```bash',
        '/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/render_poster_candidate_plots_20260929.py',
        '```', '',
        '[plotted_points.csv](plotted_points.csv) maps seed values to panels, units, protocols and '
        'input tables. [manifest.json](manifest.json) pins source hashes and includes claims/captions. '
        '[quality_checks.json](quality_checks.json) records physical sizes, editable fonts and SVG validity. '
        'Heldout summaries are regenerated from exact paired-reset trials; unmatched starts, wrong ages '
        'and duplicate trial keys fail the render.', '',
        '**Reusable lesson:** Reuse the current poster’s style and physical canvas sizes. Pin completed '
        '75M transfer inputs rather than older 50M illustrations. Inspect original trial structure before '
        'calling seed variation representation replication. Render the compact panels directly at final '
        'size; shrinking a large figure would also shrink its text.', ''])
    (g.out/'README.md').write_text('\n'.join(notes))
    manifest = {'schema':'intrmotiv/poster-candidate-gallery/v1', 'sources':g.sources,
                'font_path':font_path, 'matplotlib_version':plt.matplotlib.__version__,
                'numpy_version':np.__version__, 'pandas_version':pd.__version__,
                'plot_font_pt':PLOT_FONT_PT, 'caption_font_pt':BODY_FONT_PT,
                'transfer_study':{'schema':'intrmotiv/study/v1', 'workflow_version':'1.12.0',
                    'study_id':'cued_reward5_frozen_dg_controls_20260925',
                    'study_sha256':'e3d4f5a1d29240dcb415b9ee8a163f236b1b41c145f6c766f4f3056698a1a32a',
                    'source':'06_experiments/cued_reward5_transfer_20260925.md'},
                'poster_sha256_before':poster_hash, 'candidates':[asdict(c) for c in g.candidates],
                'study_metadata_note':'StudySpec fingerprints retained in canonical experiment reports; no new study or telemetry.'}
    (g.out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    pd.DataFrame(g.points).to_csv(g.out/'plotted_points.csv', index=False)


def verify_and_render(g: Gallery, poster: Path, original_hash: str):
    checks = []
    for path in g.figures:
        root = ET.parse(path).getroot()
        ids = [el.get('id') for el in root.iter() if el.get('id')]
        text_elements = list(root.iter(f'{{{SVG}}}text'))
        if not text_elements or len(ids) != len(set(ids)):
            raise ValueError(f'Invalid editable SVG: {path}')
        # Matplotlib uses points in its viewBox; selection-sheet text uses mm.
        font_sizes = []
        for el in text_elements:
            css = el.get('style', '')
            # New Matplotlib writes e.g. "font-size: 30px".
            match = re.search(r'font-size:\s*([\d.]+)(?:px|pt)', css)
            if match:
                size = float(match[1])
                if path.stem.startswith('gallery_') and size < 20:
                    size /= PT_MM
                font_sizes.append(size)
        if len(font_sizes) != len(text_elements) or min(font_sizes) < PLOT_FONT_PT-.01:
            raise ValueError(f'Unexpected SVG font sizes: {path}, {font_sizes}')
        if any(el.tag == f'{{{SVG}}}image' for el in root.iter()):
            raise ValueError('Candidate plots must remain vector')
        references = {ref for el in root.iter() for value in el.attrib.values()
                      for ref in re.findall(r'url\(#([^)]+)\)', value)}
        if references.difference(ids):
            raise ValueError(f'Unresolved SVG references: {path}')
        preview = path.with_suffix('.png')
        # 150 dpi print-size previews; overview sheets use 1800px for browsing.
        width_mm = float(root.get('width').removesuffix('mm'))
        export_width = 1800 if path.stem.startswith('gallery_') else round(width_mm/25.4*150)
        result = subprocess.run(['inkscape', str(path), '--export-area-page', '--export-type=png',
                                 f'--export-width={export_width}', f'--export-filename={preview}'],
                                capture_output=True, text=True)
        if result.returncode or not preview.is_file():
            raise RuntimeError(f'Inkscape export failed: {result.stderr}')
        checks.append({'file':path.name, 'width':root.get('width'), 'height':root.get('height'),
                       'text_elements':len(text_elements), 'minimum_font_pt':min(font_sizes),
                       'unique_ids':True, 'native_vector':True, 'preview':preview.name})
    if sha(poster) != original_hash:
        raise ValueError('Current poster changed during candidate generation')
    (g.out/'quality_checks.json').write_text(json.dumps({'poster_unchanged':True, 'figures':checks}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    font_path = style()
    poster = ROOT/'05_plans/poster_20260929/Bernstein2026_IntrMotiv_filled.svg'
    original_hash = sha(poster)
    g = Gallery(args.out)
    for reference in [ROOT/'06_experiments/cued_reward5_transfer_20260925.md',
                      RESULTS/'batch_summary.md']:
        g.sources[str(reference.relative_to(ROOT))] = sha(reference)
    transfer_candidates(g)
    architecture_candidates(g)
    compose_sheets(g)
    write_notes(g, font_path, original_hash)
    verify_and_render(g, poster, original_hash)
    print(f'{len(g.candidates)} candidate plots, two selection sheets: {args.out}')


if __name__ == '__main__':
    main()
