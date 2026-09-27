"""Compare corrected-core flat and goal-conditioned terminal probes.

Inputs are unchanged canonical place-field probes and final-window scalar tables.
The short-return measure uses within-episode windows, so respawns cannot count
as loops. This adapter reuses the established flow and DG-kernel calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_poster_population_kernels import correlation_kernel
from render_poster_frozen import flow_cells, render_flow


DATA = Path("06_experiments/data/flat_goal_comparison_20260927")
DEFAULT_OUTPUT = Path("06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison")
REMOTE_ANALYSIS = Path("/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/"
                       "train_dir/analysis")
CONDITIONS = {"C01": "Flat", "C02": "Goal: delayed", "C03": "Goal: immediate",
              "C05": "Goal + DG regularization", "C15": "Goal + UCB frontier"}
COLORS = {"C01": "#5E5E5E", "C02": "#0072B2", "C03": "#56B4E9",
          "C05": "#009E73", "C15": "#D55E00"}


def setup_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "svg.fonttype": "none"})


def short_return(pose: pd.DataFrame, lag: int) -> dict[str, float]:
    """Return within 100 position units after traversing over 500 in lag steps."""
    xy = pose[["x", "y"]].to_numpy(dtype=float)
    episode = pose["num_traj"].to_numpy()
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    traveled = np.convolve(steps, np.ones(lag), mode="valid")
    displacement = np.linalg.norm(xy[lag:] - xy[:-lag], axis=1)
    same_episode = episode[lag:] == episode[:-lag]
    mobile = same_episode & (traveled > 500)
    if not mobile.any():
        raise ValueError("No mobile within-episode windows")
    return {f"return_{lag}_all": float(np.mean((displacement[same_episode] < 100)
                                              & (traveled[same_episode] > 500))),
            f"return_{lag}_mobile": float(np.mean(displacement[mobile] < 100)),
            f"mobile_{lag}_fraction": float(np.mean(mobile[same_episode])),
            f"windows_{lag}_mobile": int(mobile.sum())}


def build_table(input_root: Path, output: Path) -> pd.DataFrame:
    probe_root = input_root / "corrected_core_candidates_20260902_place_fields"
    scalar = pd.read_csv("06_experiments/results/corrected_core_reevaluation_20260902/"
                         "per_run_terminal_10m.csv")
    fields = pd.read_csv(probe_root / "summary/derived_place_field_metrics.csv")
    rows = []
    for path in sorted((probe_root / "raw").glob("*100m/pose.csv")):
        name = path.parent.name
        condition = next((key for key in CONDITIONS if f"CCR_{key}_" in name), None)
        if condition is None:
            continue
        seed = int(name.split("_S", 1)[1].split("__", 1)[0])
        pose = pd.read_csv(path)
        archive = path.with_name("place_fields.npz")
        with np.load(archive, allow_pickle=False) as values:
            checkpoint = str(values["checkpoint"])
            occupancy = values["occupancy"]
        if not checkpoint.endswith("100040704.pth"):
            raise ValueError(f"Unexpected checkpoint age: {checkpoint}")
        s = scalar[(scalar.condition == condition) & (scalar.seed == seed)].iloc[0]
        f = fields[(fields.condition.str.contains(condition.lower())) &
                   (fields.seed == seed) & (fields.checkpoint_frames == 100040704)]
        if len(f) != 1:
            raise ValueError(f"Expected one field summary for {condition} seed {seed}, found {len(f)}")
        f = f.iloc[0]
        move = np.linalg.norm(np.diff(pose[["x", "y"]].to_numpy(), axis=0), axis=1)
        same = np.diff(pose.num_traj.to_numpy()) == 0
        row = {"condition": condition, "seed": seed, "checkpoint_frames": 100040704,
               "trajectory_samples": len(pose),
               "pose_file": str(REMOTE_ANALYSIS / "corrected_core_candidates_20260902_place_fields"
                                / path.relative_to(probe_root)),
               "field_file": str(REMOTE_ANALYSIS / "corrected_core_candidates_20260902_place_fields"
                                 / archive.relative_to(probe_root)),
               "checkpoint": checkpoint,
               "visited_bins_probe": int((occupancy > 0).sum()),
               "stationary_fraction_probe": float(np.mean(move[same] < 1)),
               "coverage_auc_terminal": float(s.coverage_auc),
               "unique_cells_terminal": float(s.unique_cells),
               "target_hit_lift_terminal": float(s.target_hit_lift),
               "action_sensitivity_terminal": float(s.action_sensitivity),
               "known_edge_fraction_terminal": float(s.known_edge_fraction),
               "active_unit_si_bits_probe": float(f.active_unit_mean_si_bits),
               "active_map_cosine_probe": float(f.active_map_cosine_mean),
               "active_peak_bins_probe": int(f.active_unique_peak_bins)}
        row.update(short_return(pose, 20))
        row.update(short_return(pose, 40))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["condition", "seed"])
    if len(result) != 15:
        raise ValueError(f"Expected 15 run rows, found {len(result)}")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "matched_terminal_per_run.csv", index=False)
    return result


def summary_figure(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    metrics = [("coverage_auc_terminal", "Coverage AUC / episode", None),
               ("unique_cells_terminal", "Unique cells / episode", None),
               ("return_20_mobile", "20-decision short-return fraction", (0, 1)),
               ("return_40_mobile", "40-decision short-return fraction", (0, 1))]
    for ax, (metric, ylabel, ylim) in zip(axes.flat, metrics):
        for condition in CONDITIONS:
            group = data[data.condition == condition].sort_values("seed")
            xs = np.full(len(group), list(CONDITIONS).index(condition), dtype=float)
            ax.scatter(xs + np.array([-0.09, 0, 0.09]), group[metric],
                       c=COLORS[condition], s=54, zorder=3)
            ax.plot([xs[0] - .15, xs[0] + .15], [group[metric].mean()] * 2,
                    color="black", lw=2, zorder=4)
        ax.set(xlim=(-.5, 4.5), ylabel=ylabel, ylim=ylim)
        ax.set_xticks(range(5), ["Flat", "Delayed", "Immediate", "DG reg.", "UCB"],
                      rotation=15, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Corrected core · flat versus goal-conditioned packages · 100.04M frames\n"
                 "Dots: seeds 8, 99, 123; black line: three-seed mean")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def exemplar(data: pd.DataFrame, input_root: Path, output: Path, condition: str,
             seed: int = 99) -> None:
    record = data[(data.condition == condition) & (data.seed == seed)].iloc[0]
    local_raw = input_root / "corrected_core_candidates_20260902_place_fields/raw"
    local_run = local_raw / Path(record.pose_file).parent.name
    pose = pd.read_csv(local_run / "pose.csv")
    with np.load(local_run / "place_fields.npz", allow_pickle=False) as values:
        occupancy = values["occupancy"]
        maps = values["rate_maps"]
        information = values["spatial_information"]
    destination = output / f"{condition.lower()}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)

    # Reuse the established flow computation and renderer, preserving its cell contract.
    flow_cells(pose).to_csv(destination / "flow_cells.csv", index=False)
    render_flow(pose, destination / "flow.svg", f"{CONDITIONS[condition]} · seed {seed} · local flow")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    occupancy_image = axes[0].imshow(np.ma.masked_equal(occupancy.T, 0), origin="lower",
                                     cmap="cividis", extent=(100, 2000, 100, 2000),
                                     interpolation="nearest")
    fig.colorbar(occupancy_image, ax=axes[0], label="Recorded observations", shrink=.8)
    for _, segment in pose.groupby("num_traj", sort=False):
        axes[1].plot(segment.x, segment.y, lw=.55, alpha=.65, color=COLORS[condition])
    for ax in axes:
        ax.set(xlim=(100, 2000), ylim=(100, 2000), aspect="equal",
               xlabel="x (DMLab units)", ylabel="y (DMLab units)")
    axes[0].set_title("Occupancy · white means unvisited")
    axes[1].set_title(f"Trajectory · {pose.num_traj.nunique()} reset segments")
    fig.suptitle(f"{CONDITIONS[condition]} · seed {seed} · 10,000-decision frozen probe")
    fig.savefig(destination / "trajectory_occupancy.svg", bbox_inches="tight")
    plt.close(fig)


    valid = np.flatnonzero(np.isfinite(information) & (np.nanmax(maps, axis=(0, 1)) > 0))
    selected = valid[np.argsort(information[valid])[-4:][::-1]]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    vmax = float(np.nanmax(maps[:, :, selected]))
    for ax, unit in zip(axes.flat, selected):
        field_image = ax.imshow(np.ma.masked_where(occupancy.T == 0, maps[:, :, unit].T),
                                origin="lower", vmin=0, vmax=vmax, cmap="viridis",
                                interpolation="nearest")
        ax.set(title=f"DG {unit} · SI {information[unit]:.2f} bit", xlabel="x bin", ylabel="y bin")
    fig.colorbar(field_image, ax=axes, label="Mean DG activation", shrink=.75)
    fig.suptitle(f"{CONDITIONS[condition]} · seed {seed} · top four DG maps; shared scale")
    fig.savefig(destination / "place_fields.svg", bbox_inches="tight")
    plt.close(fig)

    kernel, pairs = correlation_kernel(maps, occupancy, radius=8, min_visits=5)
    kernel[pairs < 10] = np.nan
    kernel[8, 8] = np.nan
    np.savez_compressed(destination / "dg_kernel.npz", correlation=kernel, pairs=pairs)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(kernel.T), origin="lower", extent=(-8.5, 8.5, -8.5, 8.5),
                      cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="DG population-vector Pearson correlation")
    ax.set(xlabel="x offset (100-unit bins)", ylabel="y offset (100-unit bins)",
           title=f"{condition} seed {seed} · DG spatial kernel")
    fig.savefig(destination / "dg_kernel.svg", bbox_inches="tight")
    plt.close(fig)


def graph_figure(input_root: Path, output: Path) -> None:
    """Show the checkpoint-stored edge evidence for the two goal exemplars."""
    with (input_root / "stored_graph_seed99.json").open() as stream:
        records = json.load(stream)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, record in zip(axes, records):
        attempts = np.asarray(record["control_attempts"], dtype=float)
        confidence = np.asarray(record["edge_confidence"], dtype=float)
        ratio = np.divide(confidence, attempts, out=np.zeros_like(confidence), where=attempts > 0)
        np.fill_diagonal(ratio, np.nan)
        ratio[attempts == 0] = np.nan
        image = ax.imshow(np.ma.masked_invalid(ratio), cmap="viridis", vmin=0, vmax=1,
                          interpolation="nearest")
        ax.set(xlabel="Target DG", ylabel="Source DG", title=f"{CONDITIONS[record['condition']]} · seed 99")
        ax.set_xticks(range(0, 16, 3))
        ax.set_yticks(range(0, 16, 3))
    fig.colorbar(image, ax=axes, label="Stored confidence / attempts", shrink=.78)
    fig.suptitle("Checkpoint graph evidence · white: no attempted directed edge")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def older_batch_check(input_root: Path, output: Path) -> None:
    """Record a different-generation seed-99 comparison as a boundary check."""
    rows = []
    for condition, fragment in (("flat_encourage_sim", "flat_encourage_sim"),
                                ("global_hrl_hl5000_sim", "global_hrl_hl5000_sim")):
        matches = list((input_root / "raw").glob(f"*__{fragment}__100M/pose.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one old-batch pose for {condition}: {matches}")
        row = {"condition": condition, "seed": 99,
               "pose_file": str(REMOTE_ANALYSIS / "place_fields_architecture_trajectory_20260824"
                                / matches[0].relative_to(input_root))}
        pose = pd.read_csv(matches[0])
        row.update(short_return(pose, 20))
        row.update(short_return(pose, 40))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    setup_style()
    data = build_table(args.input, args.output)
    summary_figure(data, args.output / "architecture_summary.svg")
    for condition in ("C01", "C05", "C15"):
        exemplar(data, args.input, args.output, condition)
    graph_figure(args.input, args.output / "stored_graphs_seed99.svg")
    older_batch_check(args.input, args.output / "older_batch_seed99_check.csv")
    print(data.groupby("condition")[["coverage_auc_terminal", "return_20_mobile",
                                     "return_40_mobile"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
