"""Poster-size WORKER versus RAND_DG comparison from pinned W&B histories.

Reuse the current A0 plotting style, shared frame integrals, seed-paired study
contrasts, and scoped SVG importer. This is a transfer-package comparison:
WORKER brings frozen DG, a trainable pretrained worker, and a source graph.
No training, heldout evaluation, or current-poster mutation is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from render_poster_candidate_plots_20260929 import (
    ARM_COLORS, BODY_FONT_PT, Candidate, Gallery, HEADING_FONT_PT, LARGE_MM,
    PLOT_FONT_PT, ROOT, ROW_MM, SVG, TRANSFER_SEEDS, embed, np, paired, pd,
    plt, scalar_axes, sha, style, text, verify_and_render, wrap_lines,
)
from analyze_poster_reward_auc import binned_curve, mean_reward_auc, mean_reward_window
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts, summarize_records

DEFAULT_DATA = ROOT/'06_experiments/data/worker_random_transfer_20260929'
DEFAULT_OUT = ROOT/'05_plans/poster_20260929/worker_random_transfer'
ARMS = ('W_WORKER', 'W_RAND_DG')
LABELS = ('WORKER', 'RAND_DG')
HORIZON = 75_000_000
BIN_EDGES = np.arange(0, HORIZON+1, 2_500_000)
METRICS = ('early_0_10m', 'full_0_75m', 'terminal_65_75m')


def analyze(g: Gallery, data_dir: Path):
    """Validate declared identities and pin frame-weighted plotted quantities."""
    path = data_dir/'collection_manifest.json'
    g.sources[str(path.relative_to(ROOT))] = sha(path)
    manifest = json.loads(path.read_text())
    if len(manifest['runs']) != 12:
        raise ValueError('Expected two arms × two architectures × three downstream seeds')
    declared = {}
    for study in manifest['studies']:
        path = data_dir/study['pinned_study']
        g.sources[str(path.relative_to(ROOT))] = sha(path)
        spec = load_study(path)
        if spec.fingerprint != study['study_sha256']:
            raise ValueError('Pinned study fingerprint mismatch')
        declared.update({r.name:r for r in spec.expand_runs() if r.factors['arm'] == study['selected_arm']})
    rows, curves = [], []
    for record in manifest['runs']:
        run = declared[record['run_name']]
        if record['seed'] != run.seed or record['condition'] != run.condition or record['arm'] != run.factors['arm']:
            raise ValueError('History identity disagrees with validated StudySpec')
        history = g.read(data_dir/record['history']).sort_values('_step')
        history = history.drop_duplicates('train/env_steps', keep='last').sort_values('train/env_steps')
        steps, rewards = history['train/env_steps'].to_numpy(float), history[manifest['metric']].to_numpy(float)
        if steps[-1] < .995 * HORIZON:
            raise ValueError('Incomplete comparison history')
        metadata = {key:record[key] for key in (
            'run_name', 'architecture', 'arm', 'seed', 'study_schema', 'workflow_version',
            'study_sha256', 'source_checkpoint_sha256', 'wandb_url', 'last_logged_frames',
            'endpoint_carry_frames', 'maximum_logging_gap_frames')}
        metadata['condition'] = record['arm']
        metrics = dict(zip(METRICS, (
            mean_reward_auc(steps, rewards, 10_000_000),
            mean_reward_auc(steps, rewards, HORIZON),
            mean_reward_window(steps, rewards, 65_000_000, HORIZON))))
        rows.append({**metadata, **metrics})
        bins = binned_curve(steps, rewards, HORIZON, BIN_EDGES)
        if not np.isclose(np.average(bins, weights=np.diff(BIN_EDGES)), metrics['full_0_75m'], atol=1e-12):
            raise ValueError('Curve integral differs from summary point')
        for low, high, reward in zip(BIN_EDGES[:-1], BIN_EDGES[1:], bins):
            curves.append({**metadata, 'low_frames':int(low), 'high_frames':int(high),
                           'age_m':(low+high)/2e6, 'reward':reward})
    frame, curve = pd.DataFrame(rows), pd.DataFrame(curves)
    if frame.duplicated(['architecture','arm','seed']).any():
        raise ValueError('Duplicate seed/arm cell')
    for (architecture, arm), group in frame.groupby(['architecture','arm']):
        if tuple(sorted(group.seed)) != TRANSFER_SEEDS or arm not in ARMS or architecture not in ('D50','D51'):
            raise ValueError('Unexpected comparison cell')
    pairs, contrasts = linear_contrasts(rows, METRICS, ['architecture'], ['seed'], [{
        'name':'WORKER_minus_RAND_DG', 'terms':[{'weight':1, 'where':{'arm':ARMS[0]}},
                                               {'weight':-1, 'where':{'arm':ARMS[1]}}]}])
    summaries = summarize_records(rows, ['architecture','arm'], METRICS)
    frame.to_csv(g.out/'reward_per_run.csv', index=False)
    curve.to_csv(g.out/'reward_binned_curves.csv', index=False)
    pd.DataFrame(pairs).to_csv(g.out/'paired_differences.csv', index=False)
    pd.DataFrame(contrasts).to_csv(g.out/'paired_summary.csv', index=False)
    pd.DataFrame(summaries).to_csv(g.out/'arm_summary.csv', index=False)
    return manifest, frame, curve


def summaries(g: Gallery, frame: pd.DataFrame):
    key = '01_paired_reward_summary'
    fig, axes = scalar_axes(4)
    for i, architecture in enumerate(('D50','D51')):
        selected = frame[frame.architecture == architecture]
        for j, (metric, window) in enumerate([('full_0_75m','0–75M'), ('terminal_65_75m','65–75M')]):
            ax = axes[2*i+j]
            paired(ax, selected, metric, ARMS, ('Wkr','Rnd'), factor=1000)
            ax.set(title=f'{architecture} {window}\nReward × 10³', ylim=(0,.6), yticks=[0,.3,.6])
            g.record(key, architecture+' '+window, metric, selected, metric, 1000,
                     dg_units=64, protocol='frame-weighted logged environment reward per step')
    g.finish(fig, Candidate(key, 'Reward during transfer training', ROW_MM, 'Compact main comparison',
        'Mean reward / step × 10³. Wkr: WORKER; Rnd: RAND_DG. Lines join three downstream seeds; black marks show means.',
        'WORKER transfers frozen learned DG, a trainable pretrained worker, and the source graph. RAND_DG '
        'uses calibrated frozen random DG, a fresh trainable worker, and an empty initial graph. Both '
        'receive 75M downstream frames. Panels show full-training (0–75M) and late-training (65–75M) '
        'mean logged environment reward per step, ×1000. Blue circles: WORKER; orange squares: RAND_DG. '
        'Points are seeds 42/1234/9999; gray lines join seeds; black marks are arm means.',
        'WORKER has higher arm means, but wins in only two of three seed pairs per architecture in both '
        'windows. This tests transfer of the joint system. It does not identify a DG-specific benefit '
        'or establish statistical significance with three seeds.',
        'Which interactions between landmark code, worker, and graph make transferred control useful?'))


def curves(g: Gallery, curve: pd.DataFrame):
    key = '02_learning_curves'
    fig, axes = plt.subplots(1, 2, figsize=tuple(v/25.4 for v in LARGE_MM), sharey=True)
    fig.subplots_adjust(left=.135, right=.96, top=.88, bottom=.32, wspace=.27)
    for ax, architecture in zip(axes, ('D50','D51')):
        group = curve[curve.architecture == architecture]
        for arm, color, label, linestyle in zip(ARMS, ARM_COLORS, LABELS, ('-', '--')):
            selected = group[group.arm == arm]
            for _, run in selected.groupby('seed'):
                ax.plot(run.age_m, run.reward*1000, color=color, lw=1.15, alpha=.45, ls=linestyle)
            mean = selected.groupby('age_m').reward.mean()
            ax.plot(mean.index, mean*1000, color=color, lw=3.2, ls=linestyle,
                    marker=('o' if arm == ARMS[0] else 's'), markersize=7.5, markevery=4, label=label)
        ax.set(title=architecture, xlabel='Frames (millions)', xlim=(0,75), xticks=[0,25,50,75],
               ylim=(0,.7), yticks=[0,.35,.7])
        ax.grid(color='#e5e5e5')
        g.record(key, architecture, '2.5M-bin mean reward per step', group, 'reward', 1000,
                 dg_units=64, protocol='preceding-interval frame weighting, no extra smoothing')
    axes[0].set_ylabel('Reward / step\n× 10³')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.015),
               ncol=2, frameon=False, handlelength=1.5, columnspacing=1.0)
    g.finish(fig, Candidate(key, 'Transfer gains depend on the downstream seed', LARGE_MM, 'Main temporal evidence',
        'Thick curves: three-seed means; thin curves: individual seeds. Frame-weighted 2.5M bins, with no extra smoothing.',
        'Logged environment reward per step, ×1000, averaged in 2.5M-frame bins. Each thin line is '
        'one downstream training seed; thick lines are means over the same three seeds. Blue solid '
        'circles: WORKER; orange dashed squares: RAND_DG. The curve integrals reproduce the paired '
        '0–75M summary points. Shared axes preserve the architectural comparison.',
        'The thin-line spread shows observed seed variation, not a confidence band. WORKER has lower '
        'mean reward in the first 10M frames in both architectures, so an immediate reward head start '
        'is not supported. One selected source checkpoint per architecture was reused across seeds.',
        'What determines whether an exploration-trained controller adapts or remains tied to its source behavior?'))
    for architecture in ('D50','D51'):
        key = f'03_{architecture.lower()}_paired_seed_curves'
        fig, axes = plt.subplots(1, 3, figsize=tuple(v/25.4 for v in ROW_MM), layout='constrained', sharey=True)
        selected = curve[curve.architecture == architecture]
        for ax, seed in zip(axes, TRANSFER_SEEDS):
            group = selected[selected.seed == seed]
            for arm, color, linestyle in zip(ARMS, ARM_COLORS, ('-', '--')):
                run = group[group.arm == arm]
                ax.plot(run.age_m, run.reward*1000, color=color, lw=2.2, ls=linestyle)
            ax.set(title=f'{architecture} · {seed}', xlabel='Frames (M)', xlim=(0,75),
                   xticks=[0,40,75], ylim=(0,.7), yticks=[0,.35,.7])
            ax.grid(color='#e5e5e5')
            g.record(key, f'{architecture} seed {seed}', '2.5M-bin mean reward per step', group, 'reward', 1000,
                     dg_units=64, protocol='one paired downstream seed per panel')
        axes[0].set_ylabel('Reward\n× 10³')
        g.finish(fig, Candidate(key, f'{architecture}: paired learning curves', ROW_MM,
            'Supporting seed detail',
            'Blue solid: WORKER; orange dashed: RAND_DG. Seeds 42 and 1234 gain over the full run; seed 9999 loses.',
            f'{architecture}, the same three downstream seed pairs separated into panels. Reward / step '
            '×1000, frame-weighted 2.5M bins. Blue solid is WORKER, orange dashed is RAND_DG. '
            'Panels preserve identical axes and use all observations rather than selecting successful runs.',
            'These are the same data as the main curves, not an additional replication. '
            'Each architecture has one source checkpoint, so source-checkpoint robustness remains untested.',
            'Can the initial graph or worker state predict the low-transfer seed before long training?'))


def sheet(g: Gallery):
    root = ET.Element(f'{{{SVG}}}svg', {'width':'841mm','height':'710mm','viewBox':'0 0 841 710'})
    ET.SubElement(root, f'{{{SVG}}}rect', {'width':'841','height':'710','fill':'white'})
    text(root, 'heading', 20, 30, ['WORKER versus RAND_DG: system transfer'], HEADING_FONT_PT, bold=True)
    text(root, 'subtitle', 20, 55, ['Original print sizes • plot labels 30 pt • captions 40 pt • current poster preserved'], PLOT_FONT_PT)
    for i, candidate in enumerate(g.candidates):
        x, y = (20 if i%2 == 0 else 438), (85 if i<2 else 402)
        title_lines = wrap_lines(candidate.title, 383, HEADING_FONT_PT, bold=True)
        text(root, candidate.key+'-title', x, y, title_lines, HEADING_FONT_PT, step=19, bold=True)
        plot_y = y+(len(title_lines)-1)*19+10
        height = embed(root, g.out/f'{candidate.key}.svg', candidate.key, x, plot_y, candidate.size_mm[0])
        lines = wrap_lines(candidate.short_caption, 383, BODY_FONT_PT)
        caption_y = plot_y+height+19
        if caption_y+(len(lines)-1)*18 > (377 if i<2 else 690):
            raise ValueError('Selection sheet caption exceeds its card')
        text(root, candidate.key+'-caption', x, caption_y, lines, BODY_FONT_PT, step=18)
    path = g.out/'gallery_worker_random.svg'
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    g.figures.append(path)


def notes(g: Gallery, collection: dict, frame: pd.DataFrame, font_path: str, poster_hash: str):
    means = frame.groupby(['architecture','arm'])[list(METRICS)].mean()
    lines = ['# WORKER versus RAND_DG: joint-system transfer', '',
        '![Comparison sheet](gallery_worker_random.png)', '',
        '**Recommended poster claim:** Transferring the learned DG–worker–graph system raises mean '
        'reward during adaptation, with seed-dependent outcomes.', '',
        'Use **01** for a compact result and **02** to show the temporal and seed variation. '
        'The two **03** figures are optional detail, using the same data. The current poster is unchanged. '
        'Import these editable SVGs at 100% physical size: 383 × 76.2 mm for scalar/seed rows and '
        '383 × 165.1 mm for the main learning curves. Plot labels are 30 pt DejaVu Sans; paired '
        'markers are 56 pt²; curve mean markers are 7.5 pt. Sheet captions are 40 pt.', '',
        '## What is compared', '',
        '| Component | WORKER | RAND_DG |', '| --- | --- | --- |',
        '| DG 64 | Frozen pretrained projection | Frozen random projection, BN calibrated on an unlabeled minibatch |',
        '| Worker | Pretrained, trainable during adaptation | Fresh, trainable |',
        '| Initial graph | Source graph transferred; reward values reset | Empty |',
        '| Reward manager | Fresh | Fresh |',
        '| Visual trunk | Fixed ImageNet ResNet-18 through layer 2 | Same |', '',
        'Task, downstream architecture, seeds (42, 1234, 9999), and 75M downstream budget are matched. '
        'The task has five invisible reward locations cued by a number instruction, in the source visual '
        'arena. The comparison tests whether a learned control package is reusable; it does not isolate '
        'a DG effect. WORKER includes source pretraining, so total lifetime compute is not equal. '
        'D50 and D51 use one selected pretrained source checkpoint each, reused across downstream seeds.', '',
        '## Results', '',
        'Values below are mean logged environment reward / step ×1000, not physical heldout success rates.', '',
        '| Architecture | Window | WORKER | RAND_DG | Mean relative difference | Seed pairs won |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for architecture in ('D50','D51'):
        for metric, label in zip(METRICS, ('0–10M (early)','0–75M (full)','65–75M (late)')):
            w, r = means.loc[(architecture,ARMS[0]),metric], means.loc[(architecture,ARMS[1]),metric]
            paired_values = frame[frame.architecture == architecture].pivot(index='seed', columns='arm',values=metric)
            wins = int((paired_values[ARMS[0]] > paired_values[ARMS[1]]).sum())
            lines.append(f'| {architecture} | {label} | {1000*w:.4f} | {1000*r:.4f} | {(w/r-1)*100:+.1f}% | {wins}/3 |')
    lines.extend(['',
        'Seeds 42 and 1234 favor WORKER over the full and late windows in both architectures. Seed '
        '9999 favors RAND_DG in both. Early arm means favor RAND_DG, so “immediate reward head start” '
        'or “consistently better transfer” would overstate these data. No significance or confidence '
        'interval is claimed from three seed pairs. The late window reflects reward during ongoing '
        'training, not an independent policy evaluation.', '',
        '**Heldout success remains pending:** Existing local matched-reset trials compare SOURCE_DG '
        'with RAND_DG and cannot be relabeled WORKER. Matching WORKER heldout evaluations were not '
        'available locally, and remote SSH did not respond. No new evaluation was launched.', '',
        '## Measurement and provenance', '',
        'Both arms use unsampled `wandb.Api.scan_history` exports from the same backend. Frame '
        'progress is `train/env_steps`; W&B `_step` is a logging index. Values retain W&B frame '
        'precision rather than inventing exact checkpoint ages. Each logged reward mean is assigned '
        'to its preceding logging interval, matching the original AUC convention. Repeated frame '
        'values keep the last log. Full, early, late, and binned summaries share the same interval '
        'integrator. Curves use frame-weighted 2.5M bins, with no additional smoothing.', '',
        'This is an approximation to reward accumulated during training from logged mean telemetry, '
        'not a sum of raw episode returns. The latest included value is carried to a window boundary. '
        'Two completed runs last log at 74,989,570 frames, requiring 10,430 carried frames (0.014% '
        'of the full horizon; 0.104% of the late window). The maximum logging gap is 65,540 frames. '
        'Every run passes the existing 99.5% horizon-coverage threshold and is marked finished.', '',
        '[Raw histories and validated StudySpecs](../../../06_experiments/data/worker_random_transfer_20260929/) '
        'are pinned with collection configs and study fingerprints. [manifest.json](manifest.json) '
        'hashes every input. [reward_per_run.csv](reward_per_run.csv) and '
        '[paired_differences.csv](paired_differences.csv) retain all seed values; '
        '[reward_binned_curves.csv](reward_binned_curves.csv) maps bin locations to curves. '
        'The canonical study analysis engine computes paired contrasts. '
        '[quality_checks.json](quality_checks.json) verifies editable typography, unique IDs, '
        'native vectors, and preservation of the current poster.', ''])
    for c in g.candidates:
        lines.extend([f'## {c.title}', '', f'[Editable SVG]({c.key}.svg) · [PNG]({c.key}.png)', '',
                      f'**Caption:** {c.caption}', '', f'**Claim limit:** {c.limitation}', '',
                      f'**Discussion question:** {c.question}', ''])
    lines.extend(['## Regeneration', '', 'From the repository root:', '', '```bash',
        '/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/render_worker_random_transfer_20260929.py',
        '```', '', 'Recollection is optional; it requires the existing W&B login:', '', '```bash',
        '/home/xiaoxiong/miniforge3/envs/SF_git/bin/python 06_experiments/collect_worker_random_transfer_20260929.py --refresh',
        '```', '', '**Reusable lesson:** Validate run identity against the original StudySpec; use one '
        'logging backend and its frame field for both arms. Pin histories once, reuse final-size '
        'figure helpers, and make the curve integral agree with summary points. Avoid revisiting '
        'an unresponsive SSH connection when existing logs answer the requested training comparison.', ''])
    (g.out/'README.md').write_text('\n'.join(lines))
    pd.DataFrame(g.points).to_csv(g.out/'plotted_points.csv', index=False)
    report = {'schema':'intrmotiv/worker-random-poster-comparison/v1', 'sources':g.sources,
              'studies':collection['studies'], 'font_path':font_path, 'plot_font_pt':PLOT_FONT_PT,
              'caption_font_pt':BODY_FONT_PT, 'paired_marker_area_pt2':56,
              'mean_curve_marker_diameter_pt':7.5, 'frame_bins':BIN_EDGES.tolist(),
              'backend':collection['backend'], 'reward_tag':collection['metric'],
              'poster_sha256_before':poster_hash, 'candidates':[asdict(c) for c in g.candidates],
              'matplotlib_version':plt.matplotlib.__version__, 'numpy_version':np.__version__,
              'pandas_version':pd.__version__, 'heldout_worker_available':False}
    (g.out/'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    font = style()
    poster = ROOT/'05_plans/poster_20260929/Bernstein2026_IntrMotiv_filled.svg'
    original_hash = sha(poster)
    g = Gallery(args.out)
    for path in (Path(__file__), *(Path(__file__).with_name(name) for name in (
            'analyze_poster_reward_auc.py', 'render_poster_candidate_plots_20260929.py',
            'compose_a0_poster_20260929.py', 'collect_worker_random_transfer_20260929.py'))):
        g.sources[str(path.relative_to(ROOT))] = sha(path)
    collection, frame, curve = analyze(g, args.data)
    summaries(g, frame)
    curves(g, curve)
    sheet(g)
    notes(g, collection, frame, font, original_hash)
    verify_and_render(g, poster, original_hash)
    print(f'{len(g.candidates)} plots and selection sheet: {args.out}')
    print(frame.groupby(['architecture','arm'])[list(METRICS)].mean().mul(1000).round(4).to_string())


if __name__ == '__main__':
    main()
