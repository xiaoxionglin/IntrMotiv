"""Render frozen trajectory, occupancy, flow, and stored graph diagnostics.

The canonical spatial module renders trajectories and directed outcome matrices.
Only the occupancy flow field is computed here, from consecutive frozen-policy
poses within one physical episode. This never crosses episode boundaries.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study.spatial import render_graph_outcomes, render_occupancy_trajectory


BOUNDS = (100.0, 2000.0)
GRAIN = 19
BIN_WIDTH = (BOUNDS[1] - BOUNDS[0]) / GRAIN


def flow_cells(pose: pd.DataFrame) -> pd.DataFrame:
    current = pose.iloc[:-1].reset_index(drop=True)
    following = pose.iloc[1:].reset_index(drop=True)
    same_episode = ((current['agent'].to_numpy() == following['agent'].to_numpy()) &
                    (current['num_traj'].to_numpy() == following['num_traj'].to_numpy()))
    dx = following['x'].to_numpy() - current['x'].to_numpy()
    dy = following['y'].to_numpy() - current['y'].to_numpy()
    distance = np.hypot(dx, dy)
    xbin = np.floor((current['x'].to_numpy() - BOUNDS[0]) / BIN_WIDTH).astype(int)
    ybin = np.floor((current['y'].to_numpy() - BOUNDS[0]) / BIN_WIDTH).astype(int)
    valid = same_episode & (distance <= 250) & (xbin >= 0) & (xbin < GRAIN) & (ybin >= 0) & (ybin < GRAIN)
    bins = np.stack((xbin[valid], ybin[valid]), axis=1)
    table = pd.DataFrame({'xbin': bins[:, 0], 'ybin': bins[:, 1],
                          'dx': dx[valid], 'dy': dy[valid], 'distance': distance[valid]})
    result = table.groupby(['xbin', 'ybin'], as_index=False).agg(
        transitions=('distance', 'size'), mean_dx=('dx', 'mean'),
        mean_dy=('dy', 'mean'), mean_displacement=('distance', 'mean'))
    result['coherence'] = np.hypot(result['mean_dx'], result['mean_dy']) / np.maximum(result['mean_displacement'], 1e-8)
    result['x_center'] = BOUNDS[0] + (result['xbin'] + .5) * BIN_WIDTH
    result['y_center'] = BOUNDS[0] + (result['ybin'] + .5) * BIN_WIDTH
    return result


def render_flow(pose: pd.DataFrame, output: Path, title: str, figure_scale: float = 1.0) -> pd.DataFrame:
    cells = flow_cells(pose)
    occupancy, _, _ = np.histogram2d(pose['x'], pose['y'], bins=GRAIN, range=[BOUNDS, BOUNDS])
    # Histograms are [x,y], so transpose for imshow's [row,col] convention.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'axes.titlesize': 16, 'axes.labelsize': 14,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12})
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(np.log1p(occupancy.T), extent=(*BOUNDS, *BOUNDS), origin='lower',
              cmap='Greys', alpha=.42, interpolation='nearest')
    shown = cells[cells['transitions'] >= 5]
    norm = np.hypot(shown['mean_dx'], shown['mean_dy'])
    direction_x = shown['mean_dx'] / np.maximum(norm, 1e-8)
    direction_y = shown['mean_dy'] / np.maximum(norm, 1e-8)
    length = 15 + 65 * np.clip(shown['mean_displacement'], 0, 100) / 100
    arrows = ax.quiver(shown['x_center'], shown['y_center'], length * direction_x,
                       length * direction_y, shown['coherence'], cmap='viridis', clim=(0, 1),
                       angles='xy', scale_units='xy', scale=1, width=.005)
    fig.colorbar(arrows, ax=ax, shrink=.75, label='Flow coherence')
    ax.set(xlim=BOUNDS, ylim=BOUNDS, aspect='equal', xlabel='x (DMLab units)',
           ylabel='y (DMLab units)', title=title)
    if figure_scale < 1:
        fig.canvas.draw()
        arrow_width_inches = .005 * ax.get_window_extent().width / fig.dpi
        arrows.units = "inches"
        arrows.width = arrow_width_inches
        fig.set_size_inches(8 * figure_scale, 7 * figure_scale)
    fig.savefig(output, dpi=180, bbox_inches="tight" if figure_scale < 1 else None)
    plt.close(fig)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for artifact in sorted(args.input_dir.glob('*/place_fields.npz')):
        run_dir = artifact.parent
        label = run_dir.name
        pose = pd.read_csv(run_dir / 'pose.csv')
        pose_array = pose[['x', 'y', 'rot_y']].to_numpy(dtype=np.float32)
        segment = (pose['agent'].astype(np.int64) * 1_000_000 + pose['num_traj'].astype(np.int64)).to_numpy()
        payload = {'pose': pose_array, 'segment_id': segment,
                   'dg_activity': np.zeros((len(pose), 1), dtype=np.float32),
                   'bounds': np.array([*BOUNDS, *BOUNDS]), 'grain': GRAIN,
                   'run_name': label}
        run_out = args.output_dir / label
        run_out.mkdir(parents=True, exist_ok=True)
        short_title = label.rsplit('__', 1)[-1]
        match = re.fullmatch(r'(.+)_S(\d+)_(\d+)', short_title)
        if match:
            short_title = (match.group(1).replace('_', ' ') +
                           f'\nseed {match.group(2)} · {int(match.group(3))/1e6:.1f}M frames')
        render_occupancy_trajectory(payload, run_out / 'occupancy_trajectory', title=short_title)
        cells = render_flow(pose, run_out / 'occupancy_flow.png', short_title)
        cells.to_csv(run_out / 'occupancy_flow_cells.csv', index=False)
        with np.load(artifact, allow_pickle=False) as values:
            if 'control_attempts' in values and 'control_edge_confidence' in values:
                attempts = values['control_attempts']
                confidence = values['control_edge_confidence']
                excess = confidence - attempts
                if np.any(excess > 1e-6):
                    raise ValueError(f'Material graph-count violation in {label}: {excess.max()}')
                rounded_cells = int((excess > 0).sum())
                # Legacy decayed float buffers can differ by a few ULPs at
                # effectively zero counts; satisfy the canonical renderer's
                # exact bound without changing any meaningful graph evidence.
                render_graph_outcomes(attempts, np.minimum(confidence, attempts),
                                      run_out / 'stored_graph', title=short_title,
                                      ratio_label='Stored edge confidence / attempts')
                edge_count = int((attempts > 0).sum())
                confidence_fraction = float(confidence.sum() / attempts.sum()) if attempts.sum() else float('nan')
            else:
                edge_count = 0
                confidence_fraction = float('nan')
                rounded_cells = 0
        rows.append({'label': label, 'observations': len(pose),
                     'flow_cells_at_least_5': int((cells['transitions'] >= 5).sum()),
                     'mean_flow_coherence': float(np.average(cells['coherence'], weights=cells['transitions'])),
                     'stored_graph_attempted_edges': edge_count,
                     'stored_graph_confidence_fraction': confidence_fraction,
                     'graph_roundoff_clipped_cells': rounded_cells})
    if rows:
        with (args.output_dir / 'behavior_graph_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Rendered {len(rows)} frozen-policy runs')


if __name__ == '__main__':
    main()
