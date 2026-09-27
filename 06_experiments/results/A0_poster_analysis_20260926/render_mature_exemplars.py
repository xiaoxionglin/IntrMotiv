"""Render mature online-snapshot exemplars with canonical atlas recipes."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hpc_runs.intrmotiv_study import spatial
from hpc_runs.intrmotiv_study.spatial_contract import SpatialBounds, calculate_spatial_metrics


HERE = Path(__file__).resolve().parent
OUT = HERE / "poster_candidates"
OUT.mkdir(exist_ok=True)
plt.rcParams["svg.fonttype"] = "none"


def svg_save(fig, stem, pyplot):
    path = stem.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    pyplot.close(fig)
    return [path]


# Reuse the canonical figure construction; only switch its output format.
spatial._save_figure = svg_save


def local_flow(payload, stem):
    """Valid within-segment displacement, averaged per source spatial bin."""
    pose = payload["pose"][:, :2]
    bounds = payload["bounds"]
    grain = int(payload["grain"])
    step = (bounds[1] - bounds[0]) / grain
    xy = np.floor((pose - [bounds[0], bounds[2]]) / step).astype(int)
    valid = (payload["segment_id"][1:] == payload["segment_id"][:-1]) & ~payload["dones"][:-1]
    valid &= (xy[:-1] >= 0).all(axis=1) & (xy[:-1] < grain).all(axis=1)
    valid &= (xy[1:] >= 0).all(axis=1) & (xy[1:] < grain).all(axis=1)
    displacement = pose[1:] - pose[:-1]
    valid &= np.linalg.norm(displacement, axis=1) < float(payload["max_segment_jump_distance"])
    grid = xy[:-1][valid]
    delta = displacement[valid]
    count = np.zeros((grain, grain))
    vector = np.zeros((grain, grain, 2))
    length = np.zeros((grain, grain))
    np.add.at(count, (grid[:, 0], grid[:, 1]), 1)
    np.add.at(vector, (grid[:, 0], grid[:, 1]), delta)
    np.add.at(length, (grid[:, 0], grid[:, 1]), np.linalg.norm(delta, axis=1))
    mean = np.divide(vector, count[..., None], out=np.zeros_like(vector), where=count[..., None] > 0)
    coherence = np.divide(np.linalg.norm(vector, axis=2), length, out=np.zeros_like(count), where=length > 0)
    min_count = 20
    centers = bounds[0] + (np.arange(grain) + .5) * step
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(np.ma.masked_equal(count.T, 0), origin="lower", extent=bounds, cmap="cividis")
    x, y = np.meshgrid(centers, centers, indexing="ij")
    shown = count >= min_count
    magnitude = np.linalg.norm(mean, axis=2)
    direction = np.divide(mean, magnitude[..., None], out=np.zeros_like(mean), where=magnitude[..., None] > 0)
    arrows = direction * (0.55 * step * coherence[..., None])
    ax.quiver(x[shown], y[shown], arrows[..., 0][shown], arrows[..., 1][shown],
              color="white", angles="xy", scale_units="xy", scale=1, width=.004)
    ax.set(xlabel="x (DMLab units)", ylabel="y (DMLab units)",
           title=f"Occupancy flow (n ≥ {min_count})")
    fig.colorbar(image, ax=ax, label="Source-bin transition count")
    svg_save(fig, stem, plt)
    return int(valid.sum()), float(np.nanmean(coherence[count >= min_count]))


def dg_kernel(payload, stem):
    maps = payload["rate_maps"]
    occupancy = payload["occupancy"]
    eligible = occupancy >= 5
    radius = 6
    kernel = np.full((13, 13), np.nan)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x0, x1 = max(0, -dx), min(19, 19 - dx)
            y0, y1 = max(0, -dy), min(19, 19 - dy)
            good = eligible[x0:x1, y0:y1] & eligible[x0+dx:x1+dx, y0+dy:y1+dy]
            if not good.any():
                continue
            a = maps[x0:x1, y0:y1][good]
            b = maps[x0+dx:x1+dx, y0+dy:y1+dy][good]
            a = (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1, keepdims=True), 1e-8)
            b = (b - b.mean(1, keepdims=True)) / np.maximum(b.std(1, keepdims=True), 1e-8)
            kernel[dx+radius, dy+radius] = np.mean((a*b).mean(1))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(kernel.T, origin="lower", extent=(-6.5, 6.5, -6.5, 6.5), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(xlabel="Δx (bins)", ylabel="Δy (bins)", title="DG spatial kernel")
    fig.colorbar(image, ax=ax, label="Population-vector correlation")
    svg_save(fig, stem, plt)
    near = np.nanmean([kernel[7, 6], kernel[5, 6], kernel[6, 7], kernel[6, 5]])
    far = np.nanmean([kernel[12, 6], kernel[0, 6], kernel[6, 12], kernel[6, 0]])
    return float(near), float(far)


rows = []
for path in sorted((HERE / "raw_snapshots").glob("*.npz")):
    with np.load(path) as archive:
        payload = {name: archive[name] for name in archive.files}
    name = path.stem
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    spatial.render_place_field_contact_sheets(payload, directory / "place_fields")
    spatial.render_occupancy_trajectory(payload, directory / "trajectory")
    spatial.render_trajectory_segments(payload, directory / "segments")
    spatial.render_graph_outcomes(payload["control_prospective_attempts"],
                                  payload["control_prospective_successes"],
                                  directory / "graph_outcomes", title=f"{name}: attempted directed transitions")
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(payload["graph_reliable_adjacency"], cmap="Greys", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set(xlabel="Target DG unit", ylabel="Source DG unit", title="Reliable directed edges")
    svg_save(fig, directory / "graph_reliable_mask", plt)
    transitions, coherence = local_flow(payload, directory / "flow")
    near, far = dg_kernel(payload, directory / "dg_kernel")
    metrics = calculate_spatial_metrics(
        payload["pose"], payload["dg_activity"], payload["dones"], payload["segment_id"],
        SpatialBounds(*map(float, payload["bounds"])), int(payload["grain"]),
        float(payload["stationary_distance"]),
    )
    rows.append({"run_name": str(payload["run_name"]), "checkpoint_frames": int(payload["actual_env_steps"]),
                 "samples": len(payload["pose"]), "valid_transitions": transitions,
                 "flow_coherence": coherence, "active_units": int(np.sum(payload["active_fraction"] > 0)),
                 "map_cosine": metrics["active_only_map_cosine"],
                 "mono_field_fraction": metrics["mono_field_unit_fraction"],
                 "spatial_information": metrics["active_unit_mean_spatial_information"],
                 "visited_cell_fraction": metrics["visited_cell_fraction"],
                 "distinct_peak_bins": int(metrics["unique_active_peak_bins"]),
                 "reliable_edges": int(payload["graph_reliable_edge_count"]),
                 "reachable_pair_fraction": float(payload["graph_reachable_pair_fraction"]),
                 "prospective_success_fraction": float(payload["graph_prospective_success_fraction"]),
                 "dg_kernel_near": near, "dg_kernel_far": far})
with (HERE / "mature_exemplar_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
