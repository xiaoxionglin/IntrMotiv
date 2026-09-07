"""Render selected 75M CPD online-spatial snapshots.

The place-field panels normalize each unit by its own maximum solely to compare
spatial shape. Activity amplitude and graph metrics are exported separately.
Trajectories and graphs use the raw snapshot arrays without smoothing or
subsampling except for drawing every tenth adjacent trajectory segment.
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


CONDITIONS = (
    ("base.npz", "BASE", "Baseline"),
    ("p_pass.npz", "P-PASS", "Passive predictor"),
    ("gate_ca3_dir_pass.npz", "GATE+P-PASS", "Gated CA3 feedback + predictor"),
    ("add_ca3_dir_pass.npz", "ADD+P-PASS", "Additive CA3 feedback + predictor"),
)


def configure_style() -> None:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.is_file():
        raise RuntimeError(f"required scalable font is unavailable: {font_path}")
    fm.fontManager.addfont(font_path)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 23,
        "axes.titlesize": 26,
        "axes.labelsize": 27,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 22,
        "figure.titlesize": 31,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def load_snapshots(input_dir: Path) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    loaded = []
    for filename, short, long_name in CONDITIONS:
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        if payload["pose"].shape != (100_000, 3):
            raise RuntimeError(f"unexpected pose shape in {path}: {payload['pose'].shape}")
        if payload["smoothed_rate_maps"].shape != (19, 19, 16):
            raise RuntimeError(
                f"unexpected rate-map shape in {path}: {payload['smoothed_rate_maps'].shape}"
            )
        loaded.append((short, long_name, payload))
    return loaded


def save_both(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_place_fields(short: str, long_name: str, data: dict[str, np.ndarray], output: Path) -> None:
    maps = np.asarray(data["smoothed_rate_maps"], dtype=float)
    occupancy = np.asarray(data["occupancy"]) > 0
    bounds = np.asarray(data["bounds"], dtype=float)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    active = np.asarray(data["active_fraction"], dtype=float)
    info = np.asarray(data["spatial_information"], dtype=float)
    mono = np.asarray(data["field_mono"], dtype=bool)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for unit, ax in enumerate(axes.flat):
        field = maps[:, :, unit]
        peak = float(np.nanmax(field))
        normalized = field / peak if peak > 0 else np.zeros_like(field)
        masked = np.ma.array(normalized, mask=~occupancy)
        ax.set_facecolor("#ededed")
        image = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="cividis",
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        marker = " · mono" if mono[unit] else ""
        ax.set_title(f"DG {unit}{marker}", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.90, wspace=0.16, hspace=0.24)
    colorbar_ax = fig.add_axes((0.89, 0.18, 0.025, 0.60))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Within-unit normalized activity")
    colorbar.set_ticks([0, 0.5, 1])
    fig.suptitle(
        f"{short} · seed 99 · 75M",
        y=0.975,
    )
    fig.supxlabel("x position (DMLab units)")
    fig.supylabel("y position (DMLab units)")
    save_both(fig, output / f"cpd_75m_place_fields_{short.lower().replace('+', '_')}")


def plot_trajectories(
    snapshots: list[tuple[str, str, dict[str, np.ndarray]]], output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    norm = mpl.colors.Normalize(0, 100)
    cmap = mpl.colormaps["cividis"]
    final_collection = None
    for ax, (short, long_name, data) in zip(axes.flat, snapshots):
        pose = np.asarray(data["pose"], dtype=float)
        segment = np.asarray(data["segment_id"])
        occupancy = np.asarray(data["occupancy"], dtype=float)
        bounds = np.asarray(data["bounds"], dtype=float)
        extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
        ax.imshow(
            np.log1p(occupancy),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="Greys",
            alpha=0.42,
            aspect="equal",
        )
        index = np.arange(0, pose.shape[0] - 1, 10)
        contiguous = segment[index] == segment[index + 1]
        index = index[contiguous]
        lines = np.stack((pose[index, :2], pose[index + 1, :2]), axis=1)
        progress = 100 * index / max(1, pose.shape[0] - 1)
        final_collection = LineCollection(
            lines, cmap=cmap, norm=norm, linewidth=1.1, alpha=0.82, rasterized=True
        )
        final_collection.set_array(progress)
        ax.add_collection(final_collection)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_title(short)
        ax.set_xlabel("x (DMLab units)")
        ax.set_ylabel("y (DMLab units)")
    assert final_collection is not None
    colorbar = fig.colorbar(final_collection, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Position within retained 100k-sample window (%)")
    fig.suptitle("Policy-driven trajectories · seed 99 · 75M", y=1.015)
    save_both(fig, output / "cpd_75m_trajectories")


def curved_arrow(ax: plt.Axes, start: np.ndarray, end: np.ndarray, color, width: float, rad: float) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=25,
        linewidth=width,
        color=color,
        alpha=0.82,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=14,
        shrinkB=14,
        zorder=1,
    )
    ax.add_patch(arrow)


def plot_graphs(
    snapshots: list[tuple[str, str, dict[str, np.ndarray]]], output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 16, endpoint=False)
    positions = np.column_stack((np.cos(theta), np.sin(theta)))
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(0.5, 1.0)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for ax, (short, long_name, data) in zip(axes.flat, snapshots):
        adjacency = np.asarray(data["graph_reliable_adjacency"], dtype=bool)
        reliability = np.asarray(data["graph_edge_reliability"], dtype=float)
        visits = np.asarray(data["control_node_visits"], dtype=float)
        for source, target in np.argwhere(adjacency):
            value = float(np.clip(reliability[source, target], 0.5, 1.0))
            reverse = adjacency[target, source]
            rad = 0.17 if reverse and source < target else (-0.17 if reverse else 0.06)
            curved_arrow(
                ax,
                positions[source],
                positions[target],
                cmap(norm(value)),
                1.8 + 4.2 * (value - 0.5) / 0.5,
                rad,
            )
        scaled = 500 + 1_100 * np.sqrt(visits / max(1.0, visits.max()))
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=scaled,
            facecolor="#f7f7f7",
            edgecolor="#1f2933",
            linewidth=2.2,
            zorder=3,
        )
        for node, (x, y) in enumerate(positions):
            ax.text(x, y, str(node), ha="center", va="center", fontsize=23, zorder=4)
        ax.set_xlim(-1.32, 1.32)
        ax.set_ylim(-1.32, 1.32)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(short)
    colorbar = fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Posterior edge reliability")
    fig.suptitle(
        "Reliable controllability graphs · seed 99 · 75M",
        y=1.015,
    )
    save_both(fig, output / "cpd_75m_reliable_graphs")


def write_summary(
    snapshots: list[tuple[str, str, dict[str, np.ndarray]]], output: Path
) -> None:
    rows = []
    for short, long_name, data in snapshots:
        rows.append({
            "condition": short,
            "description": long_name,
            "run_name": str(data["run_name"]),
            "target_env_steps": int(data["target_env_steps"]),
            "actual_env_steps": int(data["actual_env_steps"]),
            "retained_samples": int(data["pose"].shape[0]),
            "trajectory_segments": int(np.unique(data["segment_id"]).size),
            "visited_bins": int((data["occupancy"] > 0).sum()),
            "mean_active_fraction": float(np.mean(data["active_fraction"])),
            "mean_spatial_information": float(np.mean(data["spatial_information"])),
            "mono_field_fraction": float(np.mean(data["field_mono"])),
            "median_peak_nearest_neighbor_distance": float(
                np.nanmedian(data["field_dominant_peak_nearest_neighbor_distance"])
            ),
            "reliable_edge_count": int(data["graph_reliable_edge_count"]),
            "reachable_pair_fraction": float(data["graph_reachable_pair_fraction"]),
            "reliable_global_efficiency": float(data["graph_reliable_global_efficiency"]),
            "prospective_success_fraction": float(data["graph_prospective_success_fraction"]),
            "spatial_endpoint_valid_fraction": float(data["graph_spatial_endpoint_valid_fraction"]),
            "grounded_controllability": float(data["graph_grounded_controllability"]),
        })
    with (output / "cpd_75m_selected_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "selection_rule": (
            "Seed 99 at 75M for the predefined mechanistic sequence BASE, passive predictor-only, "
            "gated CA3 direct feedback plus passive predictor, and additive CA3 direct feedback "
            "plus passive predictor."
        ),
        "place_field_transform": (
            "Cached occupancy-normalized smoothed rate maps; each unit divided by its own maximum "
            "for spatial-shape display only; unvisited bins masked."
        ),
        "trajectory_transform": (
            "Raw retained pose; every tenth adjacent within-segment line drawn and colored by time; "
            "occupancy shown as log(1 + count)."
        ),
        "graph_transform": (
            "Cached reliable directed adjacency in a circular non-spatial layout; edge color and "
            "width encode posterior reliability; node size encodes visits."
        ),
        "limitations": [
            "Policy-driven online windows are not fixed-trajectory checkpoint rollouts.",
            "Representative visualizations use one predeclared trajectory seed (99).",
            "No graph has spatially valid endpoints under the cached grounding contract.",
        ],
        "inputs": [filename for filename, _, _ in CONDITIONS],
    }
    (output / "cpd_75m_selected_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(args.input_dir)
    for short, long_name, data in snapshots:
        plot_place_fields(short, long_name, data, args.output_dir)
    plot_trajectories(snapshots, args.output_dir)
    plot_graphs(snapshots, args.output_dir)
    write_summary(snapshots, args.output_dir)


if __name__ == "__main__":
    main()
