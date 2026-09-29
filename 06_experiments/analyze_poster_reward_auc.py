"""Paired five-cue reward curves from the canonical exported scalar histories.

The scalar collector already resolved TensorBoard files and run identity. This
adapter integrates its environment-reward mean on a fixed shared frame window.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / 'data' / 'poster_missing_analyses_20260926'
HISTORIES = ROOT / 'reward_histories'
HORIZONS = (10_000_000, 20_000_000, 50_000_000, 75_000_000)
TAG = 'intrmotiv/reward/environment_mean'


def reward_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: dict[int, float] = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['tag'] == TAG:
                values[int(row['step'])] = float(row['value'])
    if not values:
        raise ValueError(f'No environment-reward values: {path}')
    steps = np.array(sorted(values), dtype=float)
    rewards = np.array([values[int(step)] for step in steps], dtype=float)
    return steps, rewards


def reward_intervals(steps: np.ndarray, rewards: np.ndarray, end: float):
    """Return one consistent frame-weighted approximation for scalar histories.

    Attribute each logged mean to its preceding logging interval. Only samples
    at or before ``end`` enter the integral; carry the final included value to
    the boundary. This preserves the original AUC convention. Logged means are
    telemetry summaries, so the integral approximates collected reward rather
    than reconstructing exact episode returns.
    """
    steps, rewards = np.asarray(steps, float), np.asarray(rewards, float)
    if (steps.ndim != 1 or rewards.shape != steps.shape or not len(steps)
            or not np.isfinite(steps).all() or not np.isfinite(rewards).all()
            or steps[0] < 0 or np.any(np.diff(steps) <= 0) or end <= 0):
        raise ValueError('Reward history requires finite, strictly ordered frame samples and positive end')
    selected = steps <= end
    steps, rewards = steps[selected], rewards[selected]
    if not len(steps):
        raise ValueError('No reward samples in requested AUC window')
    left = np.r_[0.0, steps[:-1]]
    right = steps.copy()
    if steps[-1] < end:
        left = np.r_[left, steps[-1]]
        right = np.r_[right, end]
        rewards = np.r_[rewards, rewards[-1]]
    return left, right, rewards


def mean_reward_auc(steps: np.ndarray, rewards: np.ndarray, end: int) -> float:
    left, right, value = reward_intervals(steps, rewards, end)
    return float(np.sum((right - left) * value) / end)


def binned_curve(steps: np.ndarray, rewards: np.ndarray, end: int, bin_edges: np.ndarray) -> np.ndarray:
    left, right, value = reward_intervals(steps, rewards, end)
    bin_edges = np.asarray(bin_edges, float)
    if (bin_edges.ndim != 1 or len(bin_edges) < 2 or not np.isfinite(bin_edges).all()
            or np.any(np.diff(bin_edges) <= 0) or bin_edges[0] < 0 or bin_edges[-1] > end):
        raise ValueError('Bin edges must increase within the integration window')
    result = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        overlap = np.maximum(0, np.minimum(right, hi) - np.maximum(left, lo))
        result.append(float(np.sum(overlap * value) / (hi - lo)))
    return np.array(result)


def mean_reward_window(steps: np.ndarray, rewards: np.ndarray, low: int, high: int) -> float:
    if not 0 <= low < high:
        raise ValueError('Reward window requires 0 <= low < high')
    if steps[-1] < .995 * high:
        raise ValueError('Run has not reached the requested terminal window')
    left, right, value = reward_intervals(steps, rewards, high)
    overlap = np.maximum(0, np.minimum(right, high) - np.maximum(left, low))
    return float(np.sum(overlap * value) / (high - low))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--end', type=int, default=75_000_000,
                        help='Shared AUC endpoint in environment frames')
    args = parser.parse_args()
    end = args.end
    bin_edges = np.linspace(0, end, int(round(end / 2_500_000)) + 1)
    rows = []
    curves = {}
    window_rows = []
    terminal_rows = []
    for path in sorted(HISTORIES.glob('CR5C_*.csv')):
        name = path.stem
        parts = name.split('_')
        architecture = parts[1]
        arm = 'SOURCE_DG' if 'SOURCE_DG' in name else 'RAND_DG'
        seed = int(parts[-1][1:])
        steps, rewards = reward_series(path)
        if steps[-1] < .995 * end:
            raise ValueError(f'{name} stops at {steps[-1]:.0f}, short of {end} frames')
        auc = mean_reward_auc(steps, rewards, end)
        curve = binned_curve(steps, rewards, end, bin_edges)
        if not np.isclose(curve.mean(), auc, atol=1e-10):
            raise ValueError(f'Curve/AUC integral mismatch: {name}')
        rows.append(dict(architecture=architecture, arm=arm, seed=seed,
                         horizon_frames=end, mean_reward_auc=auc,
                         last_scalar_step=int(steps[-1]), source_file=path.name))
        curves[architecture, arm, seed] = curve
        for horizon in HORIZONS:
            if steps[-1] >= .995 * horizon:
                window_rows.append(dict(architecture=architecture, arm=arm, seed=seed,
                                        horizon_frames=horizon,
                                        mean_reward_auc=mean_reward_auc(steps, rewards, horizon),
                                        last_scalar_step=int(steps[-1]), source_file=path.name))
        if steps[-1] >= .995 * 75_000_000:
            terminal_rows.append(dict(architecture=architecture, arm=arm, seed=seed,
                                      low_frames=65_000_000, high_frames=75_000_000,
                                      mean_environment_reward=mean_reward_window(steps, rewards, 65_000_000, 75_000_000),
                                      source_file=path.name))
    if len(rows) != 12:
        raise ValueError(f'Expected 12 histories, found {len(rows)}')
    with (ROOT / 'reward_auc_per_run.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / 'reward_auc_windows_per_run.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    if terminal_rows:
        with (ROOT / 'reward_terminal_65_75m_per_run.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(terminal_rows[0]))
            writer.writeheader()
            writer.writerows(terminal_rows)

    paired = []
    for architecture in ('D50', 'D51'):
        for seed in (42, 1234, 9999):
            source = next(row for row in rows if row['architecture'] == architecture and row['arm'] == 'SOURCE_DG' and row['seed'] == seed)
            random = next(row for row in rows if row['architecture'] == architecture and row['arm'] == 'RAND_DG' and row['seed'] == seed)
            paired.append(dict(architecture=architecture, seed=seed,
                               source_mean_reward_auc=source['mean_reward_auc'],
                               random_mean_reward_auc=random['mean_reward_auc'],
                               source_minus_random=source['mean_reward_auc'] - random['mean_reward_auc']))
    with (ROOT / 'reward_auc_paired.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    window_pairs = []
    for architecture in ('D50', 'D51'):
        for horizon in HORIZONS:
            for seed in (42, 1234, 9999):
                selected = [row for row in window_rows if row['architecture'] == architecture and row['seed'] == seed and row['horizon_frames'] == horizon]
                if len(selected) != 2:
                    continue
                source = next(row for row in selected if row['arm'] == 'SOURCE_DG')
                random = next(row for row in selected if row['arm'] == 'RAND_DG')
                window_pairs.append(dict(architecture=architecture, seed=seed,
                                         horizon_frames=horizon,
                                         source_mean_reward_auc=source['mean_reward_auc'],
                                         random_mean_reward_auc=random['mean_reward_auc'],
                                         source_minus_random=source['mean_reward_auc']-random['mean_reward_auc']))
    with (ROOT / 'reward_auc_windows_paired.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_pairs[0]))
        writer.writeheader()
        writer.writerows(window_pairs)

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2e6
    for ax, architecture in zip(axes, ('D50', 'D51')):
        for arm, color in [('SOURCE_DG', '#1768a0'), ('RAND_DG', '#c65b29')]:
            matrix = np.stack([curves[architecture, arm, seed] for seed in (42, 1234, 9999)])
            mean = matrix.mean(axis=0)
            ax.plot(centers, mean, color=color, linewidth=2.4, label=arm.replace('_', ' '))
            ax.fill_between(centers, matrix.min(axis=0), matrix.max(axis=0), color=color, alpha=.15)
        ax.set_title(f'{architecture}: frozen DG')
        ax.set_xlabel('Training frames (millions)')
        ax.set_xlim(0, end / 1e6)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Environment reward per step')
    axes[1].legend(frameon=False)
    figure_stem = ROOT / f'reward_curves_0_{end//1_000_000}m'
    fig.savefig(figure_stem.with_suffix('.png'), dpi=180)
    fig.savefig(figure_stem.with_suffix('.svg'))
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey=True,
                             constrained_layout=True)
    for column, architecture in enumerate(('D50', 'D51')):
        for row, seed in enumerate((42, 1234, 9999)):
            ax = axes[row, column]
            for arm, color in [('SOURCE_DG', '#1768a0'), ('RAND_DG', '#c65b29')]:
                ax.plot(centers, curves[architecture, arm, seed], color=color,
                        linewidth=2, label=arm.replace('_', ' '))
            ax.set_title(f'{architecture}, seed {seed}')
            ax.grid(alpha=.25)
            if row == 2:
                ax.set_xlabel('Training frames (millions)')
            if column == 0:
                ax.set_ylabel('Environment reward per step')
    axes[0, 1].legend(frameon=False)
    paired_stem = ROOT / f'reward_curves_paired_0_{end//1_000_000}m'
    fig.savefig(paired_stem.with_suffix('.png'), dpi=180)
    fig.savefig(paired_stem.with_suffix('.svg'))
    plt.close(fig)
    print('Wrote paired AUC and reward curve')
    for row in paired:
        print(row)


if __name__ == '__main__':
    main()
