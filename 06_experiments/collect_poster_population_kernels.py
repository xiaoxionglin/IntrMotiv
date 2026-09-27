"""Replay a common observation panel and map DG, CA3, and decoder-1 geometry.

This uses the established IntrMotiv checkpoint loader and observation replay.
It adds a read-only decoder call after each replayed core step so the first
decoder hidden layer is measured on exactly the same state/observation history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sf_working_directories.IntrMotiv.evaluation.observation_panel import replay_observations
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env


def decoder_one_module(actor: nn.Module) -> tuple[str, nn.Module]:
    decoder = getattr(actor, 'decoder', None)
    if decoder is None:
        raise ValueError('Policy has no decoder module')
    activations = (nn.ReLU, nn.ELU, nn.GELU, nn.Tanh, nn.LeakyReLU, nn.SiLU)
    for name, module in decoder.named_modules():
        if isinstance(module, activations):
            return f'decoder.{name}', module
    return 'decoder', decoder


def rate_maps(pose, activity: np.ndarray, grain: int = 19):
    edges = np.linspace(100.0, 2000.0, grain + 1)
    xy = pose[['x', 'y']].to_numpy()
    occupancy = np.histogramdd(xy, (edges, edges))[0]
    xbin = np.searchsorted(edges, xy[:, 0], side='right') - 1
    ybin = np.searchsorted(edges, xy[:, 1], side='right') - 1
    valid = (xbin >= 0) & (xbin < grain) & (ybin >= 0) & (ybin < grain)
    sums = np.zeros((grain, grain, activity.shape[1]), dtype=np.float64)
    np.add.at(sums, (xbin[valid], ybin[valid]), activity[valid])
    with np.errstate(divide='ignore', invalid='ignore'):
        maps = sums / occupancy[:, :, None]
    maps[occupancy == 0] = np.nan
    return occupancy, maps


def correlation_kernel(maps: np.ndarray, occupancy: np.ndarray, radius: int | None = None,
                       min_visits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of population vectors at matched spatial offsets.

    Each eligible cell pair has equal weight. A cell is eligible only when the
    common panel visited it at least ``min_visits`` times and its population
    vector has nonzero variance. This follows Jannek's kernel construction
    while excluding unvisited/undefined vectors from the average.
    """
    if radius is None:
        radius = max(maps.shape[:2]) - 1
    if radius < 0 or radius > max(maps.shape[:2]) - 1:
        raise ValueError(f"Radius {radius} is outside this map's displacement range")
    mean = np.nanmean(maps, axis=2, keepdims=True)
    std = np.nanstd(maps, axis=2, keepdims=True)
    eligible = (occupancy >= min_visits) & np.isfinite(std[..., 0]) & (std[..., 0] > 1e-8)
    z = (maps - mean) / np.maximum(std, 1e-8)
    h, w, features = z.shape
    kernel = np.full((2 * radius + 1, 2 * radius + 1), np.nan, dtype=np.float32)
    pairs = np.zeros_like(kernel, dtype=np.int32)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x0, x1 = max(0, -dx), min(h, h - dx)
            y0, y1 = max(0, -dy), min(w, w - dy)
            good = eligible[x0:x1, y0:y1] & eligible[x0+dx:x1+dx, y0+dy:y1+dy]
            pairs[dx+radius, dy+radius] = int(good.sum())
            if good.any():
                a = z[x0:x1, y0:y1][good]
                b = z[x0+dx:x1+dx, y0+dy:y1+dy][good]
                kernel[dx+radius, dy+radius] = float(np.mean(np.sum(a*b, axis=1) / features))
    return kernel, pairs


def heading_matched_kernel(pose, activity: np.ndarray, radius: int | None = None):
    """Average same-heading population kernels over four 90-degree yaw bins."""
    heading = np.floor(np.mod(pose['rot_y'].to_numpy() + 180.0, 360.0) / 90.0).astype(int)
    kernels, pair_counts = [], []
    for bucket in range(4):
        selected = heading == bucket
        occupancy, maps = rate_maps(pose.loc[selected], activity[selected])
        kernel, pairs = correlation_kernel(maps, occupancy, radius=radius, min_visits=2)
        kernels.append(kernel)
        pair_counts.append(pairs)
    with np.errstate(invalid='ignore'):
        result = np.nanmean(np.stack(kernels), axis=0)
    return result, np.sum(pair_counts, axis=0)


def radial_profile(kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    dx, dy = np.mgrid[-radius:radius+1, -radius:radius+1]
    distance = np.hypot(dx, dy)
    max_rounded_distance = int(np.floor(distance.max() + .5))
    return np.array([
        np.nanmean(kernel[(distance >= max(0, r - .5)) & (distance < r + .5)])
        for r in range(max_rounded_distance + 1)
    ], dtype=np.float32)


def length_scale_bins(profile: np.ndarray) -> float | None:
    below = np.flatnonzero(np.isfinite(profile[1:]) & (profile[1:] <= np.exp(-1)))
    return float(below[0] + 1) if len(below) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-cue', action='store_true',
                        help='Also compute separate kernels for each instruction on this panel')
    args = parser.parse_args()
    cfg, env, _env_info, actor, checkpoint, device = load_policy_env(
        args.run_dir, 10_000, False, 0, args.checkpoint
    )
    env.close()
    decoder_name, decoder_module = decoder_one_module(actor)
    core_snapshots: list[np.ndarray] = []
    decoder_snapshots: list[np.ndarray] = []
    original_core = actor.forward_core
    capturing_decoder = False
    decoder_calls: list[np.ndarray] = []
    decoder_calls_per_step: list[int] = []
    multi_call_max_difference = 0.0

    def forward_core_with_decoder(head, state):
        nonlocal capturing_decoder, multi_call_max_difference
        output, next_state = original_core(head, state)
        core_snapshots.append(output.detach().cpu().numpy().copy())
        # A values-only tail runs the actual decoder path, without sampling an
        # action or modifying the recurrent state used by panel replay.
        capturing_decoder = True
        decoder_calls.clear()
        try:
            actor.forward_tail(output, values_only=True, sample_actions=False)
        finally:
            capturing_decoder = False
        decoder_calls_per_step.append(len(decoder_calls))
        if not decoder_calls:
            raise ValueError("Read-only policy tail did not activate decoder-1")
        if len(decoder_calls) > 1:
            difference = max(float(np.max(np.abs(decoder_calls[0] - later)))
                             for later in decoder_calls[1:])
            multi_call_max_difference = max(multi_call_max_difference, difference)
        # Sample Factory's generic MLP reuses one activation module across
        # hidden layers. The hook then fires twice, and the first invocation
        # is decoder layer 1. The later activation is a deeper layer.
        decoder_snapshots.append(decoder_calls[0])
        return output, next_state

    def decoder_hook(_module, _inputs, output):
        if not capturing_decoder:
            return
        if isinstance(output, (tuple, list)):
            output = output[0]
        decoder_calls.append(output.detach().cpu().numpy().copy())

    actor.forward_core = forward_core_with_decoder
    handle = decoder_module.register_forward_hook(decoder_hook)
    try:
        pose, dg, _logits = replay_observations(actor, cfg, args.panel, device)
    finally:
        handle.remove()
        actor.forward_core = original_core
    core = np.concatenate(core_snapshots)
    decoder = np.concatenate(decoder_snapshots)
    if len(core) != len(pose) or len(decoder) != len(pose) or len(dg) != len(pose):
        raise ValueError(f'Population alignment failed: pose={len(pose)}, DG={len(dg)}, '
                         f'CA3={len(core)}, decoder={len(decoder)}')
    n = int(cfg.Hippo_n_feature)
    e = int(cfg.Hippo_R + cfg.Hippo_L - 1)
    ca3 = core[:, :n*e]
    if ca3.shape[1] != n*e:
        raise ValueError('Canonical CA3 trace is shorter than declared')
    populations = {'dg': dg, 'ca3': ca3, 'decoder_1': decoder}
    output = {}
    grid_bins = 19
    max_offset_bins = grid_bins - 1
    reference_far_offset_bins = 6
    summary = {'checkpoint': str(checkpoint), 'panel': str(args.panel),
               'decoder_1_module': decoder_name, 'observations': len(pose),
               'grid_bins_per_axis': grid_bins,
               'bin_width_dmlab_units': 100,
               'max_offset_bins_per_axis': max_offset_bins,
               'reference_far_offset_bins': reference_far_offset_bins,
               'decoder_calls_per_step': sorted(set(decoder_calls_per_step)),
               'decoder_multi_call_max_difference': multi_call_max_difference,
               'feature_counts': {name: values.shape[1] for name, values in populations.items()}}
    cue_values = None
    if args.per_cue:
        with np.load(args.panel, allow_pickle=False) as panel:
            if 'obs_INSTR' not in panel:
                raise ValueError('Per-cue analysis requires obs_INSTR in the common panel')
            cue_values = panel['obs_INSTR'].reshape(-1)
        if len(cue_values) != len(pose):
            raise ValueError('Panel instruction labels and replayed poses do not align')
        summary['per_cue'] = {}
    for name, values in populations.items():
        occupancy, maps = rate_maps(pose, values)
        kernel, pairs = correlation_kernel(maps, occupancy)
        heading_kernel, heading_pairs = heading_matched_kernel(pose, values)
        output[f'{name}_kernel'] = kernel
        output[f'{name}_pair_count'] = pairs
        output[f'{name}_heading_kernel'] = heading_kernel
        output[f'{name}_heading_pair_count'] = heading_pairs
        output[f'{name}_radial_profile'] = radial_profile(kernel)
        output[f'{name}_heading_radial_profile'] = radial_profile(heading_kernel)
        summary[f'{name}_length_scale_bins'] = length_scale_bins(output[f'{name}_radial_profile'])
        summary[f'{name}_heading_length_scale_bins'] = length_scale_bins(output[f'{name}_heading_radial_profile'])
        near = np.array([kernel[max_offset_bins+dx, max_offset_bins+dy]
                         for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))])
        far = np.array([kernel[max_offset_bins+dx, max_offset_bins+dy]
                        for dx,dy in ((reference_far_offset_bins,0),(-reference_far_offset_bins,0),
                                      (0,reference_far_offset_bins),(0,-reference_far_offset_bins))])
        summary[f'{name}_near_corr'] = float(np.nanmean(near))
        summary[f'{name}_far_corr'] = float(np.nanmean(far))
        summary[f'{name}_near_minus_far'] = summary[f'{name}_near_corr'] - summary[f'{name}_far_corr']
        heading_near = np.array([heading_kernel[max_offset_bins+dx, max_offset_bins+dy]
                                 for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))])
        heading_far = np.array([heading_kernel[max_offset_bins+dx, max_offset_bins+dy]
                                for dx,dy in ((reference_far_offset_bins,0),(-reference_far_offset_bins,0),
                                              (0,reference_far_offset_bins),(0,-reference_far_offset_bins))])
        summary[f'{name}_heading_near_corr'] = float(np.nanmean(heading_near))
        summary[f'{name}_heading_far_corr'] = float(np.nanmean(heading_far))
        summary[f'{name}_heading_near_minus_far'] = (summary[f'{name}_heading_near_corr'] -
                                                      summary[f'{name}_heading_far_corr'])
        if cue_values is not None:
            for cue_id in sorted(np.unique(cue_values).astype(int)):
                selected = cue_values == cue_id
                cue_occupancy, cue_maps = rate_maps(pose.loc[selected], values[selected])
                cue_kernel, cue_pairs = correlation_kernel(cue_maps, cue_occupancy, min_visits=2)
                output[f'{name}_cue_{cue_id}_kernel'] = cue_kernel
                output[f'{name}_cue_{cue_id}_pair_count'] = cue_pairs
                near_indices = tuple((max_offset_bins + dx, max_offset_bins + dy)
                                     for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)))
                far_indices = tuple((max_offset_bins + dx, max_offset_bins + dy)
                                    for dx, dy in ((reference_far_offset_bins,0),
                                                   (-reference_far_offset_bins,0),
                                                   (0,reference_far_offset_bins),
                                                   (0,-reference_far_offset_bins)))
                near = np.array([cue_kernel[index] for index in near_indices])
                far = np.array([cue_kernel[index] for index in far_indices])
                near_count = int(sum(cue_pairs[index] for index in near_indices))
                far_count = int(sum(cue_pairs[index] for index in far_indices))
                summary['per_cue'].setdefault(str(cue_id), {'observations': int(selected.sum())})[name] = {
                    'visited_cells': int((cue_occupancy > 0).sum()),
                    'near_pairs': near_count,
                    'far_pairs': far_count,
                    'near_corr': float(np.nanmean(near)),
                    'far_corr': float(np.nanmean(far)),
                    'near_minus_far': float(np.nanmean(near) - np.nanmean(far)),
                }
    output['occupancy'] = occupancy
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
