"""Render 75M place fields, reliable graph, and trajectory for selected outliers.

Run on NEMO2 with the SFgit environment. Each page uses the retained 100k
policy-driven samples from one canonical online-spatial snapshot. Place-field
maps are occupancy-normalized and divided by each unit's own peak for shape
comparison. The reliable graph uses a circular, non-spatial node layout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from hpc_runs.intrmotiv_study.spatial_contract import (
    SpatialBounds,
    calculate_spatial_metrics,
)


RUNS = (
    ("DGP focal", "intrmotiv_dg_policy_gradient_first_outcome_20260906", "DGP_C15_HIT_JOINT_LEG_S123"),
    ("DGP FIRST", "intrmotiv_dg_policy_gradient_first_outcome_20260906", "DGP_C15_FIRST_JOINT_LEG_S99"),
    ("DGP late coverage", "intrmotiv_dg_policy_gradient_first_outcome_20260906", "DGP_C15_FIRST_JOINT_LEG_S8"),
    ("CPD temporal", "intrmotiv_ca3_feedback_predictive_dg_20260907", "CPD_C15_GATE_CA3_BPTT_S99"),
    ("CPD mono-field", "intrmotiv_ca3_feedback_predictive_dg_20260907", "CPD_C15_GATE_ACT_DIR_GOAL_S8"),
    ("CPD grounded", "intrmotiv_ca3_feedback_predictive_dg_20260907", "CPD_C15_GATE_ACT_DIR_S99"),
    ("SCR spatial", "intrmotiv_source_credit_retirement_20260904", "SCR_C15_ARR_DIRS_S123"),
    ("SAT spatial", "intrmotiv_saturday_batch_20260905", "SAT_C15_ARR_DIRO_FILM_S8"),
    ("SAT low overlap", "intrmotiv_saturday_batch_20260905", "SAT_C15_ARR_MON_FILM_S123"),
    ("SAT exploration", "intrmotiv_saturday_batch_20260905", "SAT_C15_SRC_MON_FILM_S8"),
)


def configure_style() -> str:
    font_path = Path(fm.findfont(
        fm.FontProperties(family="DejaVu Sans"), fallback_to_default=False
    ))
    if not font_path.is_file() or font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"required scalable font unavailable: {font_path}")
    fm.fontManager.addfont(font_path)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 19,
        "axes.titlesize": 21, "axes.labelsize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
        "figure.titlesize": 27, "axes.linewidth": 1.1,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    return str(font_path)


def scalar(data: dict[str, np.ndarray], key: str):
    return np.asarray(data[key]).item()


def load_snapshot(root: Path, batch: str, run: str) -> tuple[Path, dict[str, np.ndarray]]:
    paths = list((root / batch / run / "policy_00").glob("snapshot_target_000075000000_actual_*.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"{run}: expected one 75M snapshot, found {len(paths)}")
    with np.load(paths[0], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data["pose"].shape != (100_000, 3) or data["dg_activity"].shape != (100_000, 16):
        raise RuntimeError(f"{run}: unexpected retained-array shapes")
    if str(scalar(data, "run_name")) != run or int(scalar(data, "target_env_steps")) != 75_000_000:
        raise RuntimeError(f"{run}: snapshot identity mismatch")
    return paths[0], data


def arrow(ax, start, end, color, width, rad):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=width,
        color=color, alpha=0.55, connectionstyle=f"arc3,rad={rad}",
        shrinkA=8, shrinkB=8, zorder=1,
    ))


def page(short: str, run: str, data: dict[str, np.ndarray], output: Path) -> dict[str, object]:
    maps = np.asarray(data["smoothed_rate_maps"], dtype=float)
    occupancy = np.asarray(data["occupancy"], dtype=float)
    mono = np.asarray(data["field_mono"], dtype=bool)
    eligible = np.asarray(data["field_eligible"], dtype=bool)
    bounds = np.asarray(data["bounds"], dtype=float)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    metrics = calculate_spatial_metrics(
        data["pose"], data["dg_activity"], data["dones"], data["segment_id"],
        SpatialBounds(*bounds.tolist()), int(scalar(data, "grain")),
    )

    fig = plt.figure(figsize=(22, 15), layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.90))
    outer = fig.add_gridspec(1, 2, width_ratios=(1.65, 1.0))
    field_grid = outer[0].subgridspec(4, 4, wspace=0.06, hspace=0.18)
    right = outer[1].subgridspec(2, 1, hspace=0.18)
    field_axes = []
    image = None
    cmap = mpl.colormaps["cividis"].copy(); cmap.set_bad("#e0e0e0")
    for unit in range(16):
        ax = fig.add_subplot(field_grid[unit // 4, unit % 4]); field_axes.append(ax)
        field = maps[:, :, unit]
        peak = float(np.nanmax(field))
        normalized = field / peak if peak > 0 else np.zeros_like(field)
        image = ax.imshow(np.ma.array(normalized, mask=occupancy == 0), origin="lower",
            extent=extent, interpolation="nearest", cmap=cmap, vmin=0, vmax=1, aspect="equal")
        mark = "●" if mono[unit] else ("○" if eligible[unit] else "×")
        ax.set_title(f"DG {unit}  {mark}", pad=2, fontsize=17)
        ax.set_xticks([]); ax.set_yticks([])
    assert image is not None
    cbar = fig.colorbar(image, ax=field_axes, location="bottom", shrink=0.62, pad=0.02, aspect=35)
    cbar.set_label("Within-unit normalized occupancy-corrected DG activity")
    cbar.set_ticks([0, .5, 1])

    graph_ax = fig.add_subplot(right[0])
    adjacency = np.asarray(data["graph_reliable_adjacency"], dtype=bool)
    reliability = np.asarray(data["graph_edge_reliability"], dtype=float)
    visits = np.asarray(data["control_node_visits"], dtype=float)
    theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 16, endpoint=False)
    positions = np.column_stack((np.cos(theta), np.sin(theta)))
    edge_cmap = mpl.colormaps["viridis"]
    edge_norm = mpl.colors.Normalize(.5, 1.0)
    for source, target in np.argwhere(adjacency):
        value = float(np.clip(reliability[source, target], .5, 1))
        reverse = adjacency[target, source]
        rad = .12 if reverse and source < target else (-.12 if reverse else .035)
        arrow(graph_ax, positions[source], positions[target], edge_cmap(edge_norm(value)),
              .45 + 1.1 * (value - .5) / .5, rad)
    sizes = 230 + 530 * np.sqrt(visits / max(1.0, visits.max()))
    graph_ax.scatter(positions[:, 0], positions[:, 1], s=sizes, facecolor="#fafafa",
        edgecolor=np.where(mono, "#D55E00", "#263238"), linewidth=2.1, zorder=3)
    for node, (x, y) in enumerate(positions):
        graph_ax.text(x, y, str(node), ha="center", va="center", fontsize=15, zorder=4)
    graph_ax.set(xlim=(-1.22, 1.22), ylim=(-1.22, 1.22), aspect="equal",
        title="b  Reliable directed graph\n(circular non-spatial layout)")
    graph_ax.axis("off")
    edge_scalar = mpl.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    graph_bar = fig.colorbar(edge_scalar, ax=graph_ax, shrink=.68, pad=.01)
    graph_bar.set_label("Posterior edge reliability")
    graph_bar.set_ticks([.5, .75, 1])

    trajectory_ax = fig.add_subplot(right[1])
    pose = np.asarray(data["pose"], dtype=float)
    segment = np.asarray(data["segment_id"])
    trajectory_ax.imshow(np.log1p(occupancy), origin="lower", extent=extent,
        interpolation="nearest", cmap="Greys", alpha=.48, aspect="equal")
    index = np.arange(0, pose.shape[0] - 1, 10)
    index = index[segment[index] == segment[index + 1]]
    lines = np.stack((pose[index, :2], pose[index + 1, :2]), axis=1)
    progress = 100 * index / max(1, pose.shape[0] - 1)
    collection = LineCollection(lines, cmap="cividis", norm=mpl.colors.Normalize(0, 100),
        linewidth=.75, alpha=.72, rasterized=True)
    collection.set_array(progress); trajectory_ax.add_collection(collection)
    trajectory_ax.set(xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]), aspect="equal",
        title="c  Policy-driven trajectory", xlabel="x (environment units)", ylabel="y (environment units)")
    time_bar = fig.colorbar(collection, ax=trajectory_ax, shrink=.68, pad=.01)
    time_bar.set_label("Position in retained window (%)")

    edge_count = int(np.asarray(data["graph_reliable_edge_count"]).item())
    mono_count = int(mono.sum())
    info = float(metrics["active_unit_mean_spatial_information"])
    cosine = float(metrics["active_only_map_cosine"])
    grounded = float(np.asarray(data["graph_grounded_controllability"]).item())
    fig.suptitle(
        f"{short}: {run} · 75M\n"
        f"100k samples · mono {mono_count}/16 · information {info:.3f} · map cosine {cosine:.3f} · "
        f"reliable edges {edge_count}/240 · grounded score {grounded:.3f}", y=.975)
    fig.text(.02, .895, "a  Place fields", fontsize=21)
    stem = output / run.lower()
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "label": short, "run_name": run, "retained_samples": int(pose.shape[0]),
        "trajectory_segments": int(np.unique(segment).size), "mono_field_units": mono_count,
        "eligible_units": int(eligible.sum()), "mean_spatial_information": info,
        "active_only_map_cosine": cosine, "unique_peak_bins": int(metrics["unique_active_peak_bins"]),
        "visited_cell_fraction": float(metrics["visited_cell_fraction"]),
        "reliable_edges": edge_count, "reachable_pair_fraction": float(np.asarray(data["graph_reachable_pair_fraction"]).item()),
        "grounded_controllability": grounded,
    }


def overview(rows: list[dict[str, object]], output: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True, constrained_layout=True)
    values = (
        ([100 * float(row["mono_field_units"]) / float(row["eligible_units"]) for row in rows], "Single-field units (%)", (0, 75)),
        ([float(row["active_only_map_cosine"]) for row in rows], "Active-only map cosine", (0, .45)),
        ([float(row["grounded_controllability"]) for row in rows], "Grounded graph score", (0, .55)),
    )
    for ax, (x, title, limits) in zip(axes, values):
        ax.scatter(x, y, s=85, color="#0072B2")
        ax.set(title=title, xlim=limits, yticks=y, yticklabels=labels)
        ax.grid(axis="x", color="#dddddd", lw=.8)
    axes[0].invert_yaxis()
    fig.suptitle("Selected late-training outliers · terminal 100k-sample snapshots")
    fig.savefig(output / "selected_outliers_overview.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "selected_outliers_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    font = configure_style(); rows = []; sources = []
    for short, batch, run in RUNS:
        path, data = load_snapshot(args.snapshot_root, batch, run)
        rows.append(page(short, run, data, args.output)); sources.append(str(path))
    overview(rows, args.output)
    with (args.output / "selected_outlier_spatial_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "gallery_manifest.json").write_text(json.dumps({
        "target_env_steps": 75_000_000, "font": font, "sources": sources,
        "sample_unit": "one retained policy-driven behavior sample",
        "retained_samples_per_run": 100_000,
        "place_field_transform": "cached occupancy-normalized smoothed rate maps, each divided by its own maximum for display; unvisited bins masked",
        "graph_transform": "cached reliable directed graph in circular non-spatial layout; orange node borders mark mono-field units; node size indicates visits",
        "trajectory_transform": "raw pose, every tenth adjacent within-segment line drawn over log1p occupancy and colored by retained-window time",
        "selection": [run for _, _, run in RUNS],
        "limitation": "policy-driven online windows; not fixed-trajectory stability or causal intervention evidence",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
