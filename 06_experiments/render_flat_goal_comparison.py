"""Compare corrected-core flat and goal-conditioned terminal probes.

Inputs are unchanged canonical place-field probes and final-window scalar tables.
The short-return measure uses within-episode windows, so respawns cannot count
as loops. This adapter reuses the established flow and DG-kernel calculations.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from render_poster_frozen import flow_cells, render_flow
from render_goal_option_trajectory import (
    render as render_trajectory, trajectory_segments, validate_complete_event_stream,
)
from analyze_place_field_manifest import (
    multilevel_field_structure, FIELD_MIN_ACTIVE_OBSERVATIONS, FIELD_MIN_ACTIVE_BINS,
    FIELD_THRESHOLD_FRACTIONS, FIELD_MONO_MASS_FRACTION,
)
from prepare_poster_exemplar_gallery import mono_peak_panel


DATA = Path("06_experiments/data/flat_goal_comparison_20260927")
DEFAULT_OUTPUT = Path("06_experiments/results/A0_poster_analysis_20260926/flat_goal_comparison")
REMOTE_ANALYSIS = Path("/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/"
                       "train_dir/analysis")
CONDITIONS = {"C01": "Non-goal-conditioned", "C02": "Goal: delayed", "C03": "Goal: immediate",
              "C05": "Goal + DG regularization", "C15": "Goal + UCB frontier"}
COLORS = {"C01": "#5E5E5E", "C02": "#0072B2", "C03": "#56B4E9",
          "C05": "#009E73", "C15": "#D55E00"}

# Compact canvases preserve all physical font, stroke, and marker sizes.
FIGURE_SCALE = 1 / np.sqrt(3)


def setup_style() -> None:
    from matplotlib.font_manager import findfont
    font_path = Path(findfont("DejaVu Sans", fallback_to_default=False))
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise ValueError(f"Expected a scalable font, found {font_path}")
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


def summary_figure(data: pd.DataFrame, output: Path, conditions=None, *,
                   width_scale: float = 1.0) -> None:
    conditions = list(CONDITIONS) if conditions is None else list(conditions)
    fig, axes = plt.subplots(2, 2, figsize=(14 * FIGURE_SCALE * width_scale, 10 * FIGURE_SCALE), constrained_layout=True)
    metrics = [("coverage_auc_terminal", "Coverage AUC", None),
               ("unique_cells_terminal", "Unique cells", None),
               ("return_20_mobile", "20-step return", (0, 1)),
               ("return_40_mobile", "40-step return", (0, 1))]
    for ax, (metric, ylabel, ylim) in zip(axes.flat, metrics):
        for condition in conditions:
            group = data[data.condition == condition].sort_values("seed")
            xs = np.full(len(group), conditions.index(condition), dtype=float)
            ax.scatter(xs + np.array([-0.09, 0, 0.09]), group[metric],
                       c=COLORS[condition], s=54, zorder=3)
            ax.plot([xs[0] - .15, xs[0] + .15], [group[metric].mean()] * 2,
                    color="black", lw=2, zorder=4)
        ax.set(xlim=(-.5, len(conditions)-.5), ylabel=ylabel, ylim=ylim)
        labels = {"C01": "C01\nNon-goal-\nconditioned", "C02": "C02\nDelayed",
                  "C03": "C03\nImmediate", "C05": "C05\nGoal + DG reg.",
                  "C15": "C15\nGoal + UCB"}
        ax.set_xticks(range(len(conditions)), conditions if FIGURE_SCALE < 1 else [labels[c] for c in conditions],
                      fontsize=16, ha="center")
        ax.tick_params(axis="y", labelsize=16)
        ax.yaxis.label.set_size(16)
        ax.grid(axis="y", alpha=.2)
    title = "Corrected core · 100.04M frames\nSeeds 8 / 99 / 123 · black line: seed mean"
    if width_scale < .8:
        title = "Corrected core · 100.04M frames\nSeeds 8 / 99 / 123\nBlack line: seed mean"
    fig.suptitle(title, fontsize=18)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def paired_changes(data: pd.DataFrame, output: Path) -> None:
    """Compare each goal-conditioned seed with its C01 training-seed baseline."""
    baseline = data[data.condition == "C01"].set_index("seed")
    rows = []
    for record in data[data.condition.isin(["C05", "C15"])].itertuples():
        base = baseline.loc[record.seed]
        rows.append({"condition": record.condition, "baseline": "C01", "seed": record.seed,
                     "coverage_auc_change": record.coverage_auc_terminal - base.coverage_auc_terminal,
                     "unique_cells_change": record.unique_cells_terminal - base.unique_cells_terminal,
                     "return20_change_pp": 100 * (record.return_20_mobile - base.return_20_mobile),
                     "return40_change_pp": 100 * (record.return_40_mobile - base.return_40_mobile),
                     "probe_visited_bins_change": record.visited_bins_probe - base.visited_bins_probe})
    pd.DataFrame(rows).to_csv(output, index=False)


def mono_field_outputs(data: pd.DataFrame, input_root: Path, output: Path) -> None:
    """Verify historical classifications and render all comparison maps locally."""
    gallery = output.parent / "exemplar_gallery/historical_extension"
    inventory = pd.read_csv(gallery / "historical_exemplar_inventory.csv")
    raw = input_root / "corrected_core_candidates_20260902_place_fields/raw"
    unit_tables, counts = [], []
    for record in data[data.condition.isin(["C01", "C05", "C15"])].itertuples():
        path = raw / Path(record.field_file).parent.name / "place_fields.npz"
        with np.load(path, allow_pickle=False) as values:
            maps, occupancy = values["rate_maps"], values["occupancy"]
            active = values["active_fraction"]
            if str(values["checkpoint"]) != record.checkpoint:
                raise ValueError(f"Checkpoint mismatch in {path}")
            eligible, _, _, score, mono = multilevel_field_structure(maps, occupancy, active)
            peaks = np.asarray([
                np.unravel_index(np.nanargmax(maps[:, :, u]), occupancy.shape)
                if np.nanmax(maps[:, :, u]) > 0 else (-1, -1)
                for u in range(len(active))
            ])
        units = pd.DataFrame({"condition": record.condition, "seed": record.seed,
                              "checkpoint_frames": record.checkpoint_frames,
                              "unit": np.arange(len(active)), "field_eligible": eligible,
                              "mono_field": mono, "mono_score": score,
                              "active_fraction": active, "peak_x_bin": peaks[:, 0],
                              "peak_y_bin": peaks[:, 1]})
        units["peak_x_position"] = np.where(peaks[:, 0] >= 0, 150 + 100 * peaks[:, 0], np.nan)
        units["peak_y_position"] = np.where(peaks[:, 1] >= 0, 150 + 100 * peaks[:, 1], np.nan)
        units["original_field_file"] = record.field_file
        selected = units[units.field_eligible & units.mono_field & (units.active_fraction > 0)]
        bins = len(selected[["peak_x_bin", "peak_y_bin"]].drop_duplicates())
        existing = inventory[(inventory.checkpoint == record.checkpoint) &
                             (inventory.seed == record.seed) &
                             (inventory.protocol == "frozen_10k_policy_probe")]
        if len(existing) == 1:
            saved = existing.iloc[0]
            if (int(saved.mono_units), int(saved.eligible_units), int(saved.mono_peak_bins)) != (
                    len(selected), int(eligible.sum()), bins):
                raise ValueError(f"Existing mono-field metrics differ for {record.condition}/{record.seed}")
            figure = Path(saved.figure_dir) / "mono_field_peaks.svg"
            if not figure.is_file():
                raise FileNotFoundError(figure)
            source_figure = os.path.relpath(figure, output)
        else:
            if len(existing) > 1:
                raise ValueError("Ambiguous historical mono-field figure")
            source_figure = None
        figure = output / f"{record.condition.lower()}_seed{record.seed}/mono_field_peaks.svg"
        figure.parent.mkdir(parents=True, exist_ok=True)
        # Historical maps use Times New Roman; keep that styling local.
        historical_style = {"font.family": "Times New Roman", "font.size": 12,
                            "axes.titlesize": 13, "axes.labelsize": 12,
                            "xtick.labelsize": 12, "ytick.labelsize": 12}
        if source_figure is not None:
            from matplotlib.font_manager import findfont
            findfont("Times New Roman", fallback_to_default=False)
        with plt.rc_context(historical_style if source_figure is not None else {}):
            canvas = (4.1, 4.5) if source_figure is not None else (7, 7.3)
            fig, ax = plt.subplots(figsize=tuple(side * FIGURE_SCALE for side in canvas),
                                   layout="constrained")
            mono_peak_panel(ax, units, occupancy,
                            f"{record.condition} · S{record.seed}\n{len(selected)}/16 mono · {bins} bins")
            if source_figure is not None:
                for text in ax.texts:
                    if text.get_text() == "No mono-field units":
                        text.set_text("No mono-field\nunits")
            fig.savefig(figure, bbox_inches="tight")
            plt.close(fig)
        counts.append({"condition": record.condition, "seed": record.seed,
                       "checkpoint_frames": record.checkpoint_frames, "dg_units": len(active),
                       "eligible_units": int(eligible.sum()), "mono_units": len(selected),
                       "mono_fraction_all_dg": len(selected) / len(active),
                       "mono_peak_bins": bins, "figure": os.path.relpath(figure, output),
                       "reused_existing_figure": False, "verified_source_figure": source_figure,
                       "original_field_file": record.field_file,
                       "field_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        unit_tables.append(units)
    all_units = pd.concat(unit_tables, ignore_index=True)
    all_units.to_csv(output / "mono_field_units.csv", index=False)
    all_units[all_units.field_eligible & all_units.mono_field & (all_units.active_fraction > 0)].to_csv(
        output / "mono_field_peak_locations.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "mono_field_peak_counts.csv", index=False)
    method = {
        "schema": "intrmotiv/mono-field-peak-comparison/v1",
        "checkpoint_frames": 100040704,
        "protocol": "Archived stochastic 10000-decision frozen probes; no new rollouts",
        "classification": "Existing multilevel_field_structure: occupancy-corrected binomial smoothing, 8-connected components",
        "eligible_min_active_observations": FIELD_MIN_ACTIVE_OBSERVATIONS,
        "eligible_min_active_bins": FIELD_MIN_ACTIVE_BINS,
        "threshold_peak_fractions": list(FIELD_THRESHOLD_FRACTIONS),
        "min_dominant_component_mass_at_each_threshold": FIELD_MONO_MASS_FRACTION,
        "peak_location": "Maximum of the original unsmoothed occupancy-corrected rate map; ties use first NumPy argmax bin",
        "grid": {"grain": 19, "bounds": [100, 2000, 100, 2000],
                 "bin_center_position": "150 + 100 * bin_index"},
        "fraction_denominator": "All 16 DG units; eligible-unit counts exported separately",
        "plotting_helper": "prepare_poster_exemplar_gallery.mono_peak_panel",
        "classifier": "analyze_place_field_manifest.multilevel_field_structure",
    }
    (output / "mono_field_method.json").write_text(json.dumps(method, indent=2) + "\n")


def pack_peak_labels(ax: plt.Axes, labels: list) -> None:
    """Place unit labels near their peaks without covering labels or markers.

    Try a fixed set of nearby positions in physical points. The score penalizes
    overlapping labels, covering peak markers, and leaving the plotting area;
    neither data coordinates nor text sizes change.
    """
    from matplotlib.transforms import Bbox

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    point_to_pixel = ax.figure.dpi / 72
    radius = np.sqrt(95) / 2 * point_to_pixel
    marker_boxes = []
    for label in labels:
        x, y = ax.transData.transform(label.xy)
        marker_boxes.append(Bbox.from_extents(x-radius, y-radius, x+radius, y+radius))
    offsets = [(x, y) for distance in (7, 14, 22) for x, y in
               [(-distance, distance), (distance, distance), (-distance, -distance),
                (distance, -distance), (0, distance), (0, -distance),
                (-distance, 0), (distance, 0)]]
    def overlap(first, second):
        return max(0., min(first.x1, second.x1)-max(first.x0, second.x0)) * max(
            0., min(first.y1, second.y1)-max(first.y0, second.y0))
    placed = []
    # Long labels (coincident units) have fewer viable positions, so place them first.
    for label in sorted(labels, key=lambda item: -len(item.get_text())):
        best = None
        for dx, dy in [label.get_position(), *offsets]:
            label.set_position((dx, dy))
            label.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
            label.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
            box = label.get_window_extent(renderer).padded(point_to_pixel)
            outside = box.width*box.height - overlap(box, ax.bbox)
            collisions = sum(overlap(box, previous) for previous in placed)
            covered = sum(overlap(box, marker) for marker in marker_boxes)
            cost = 100*outside + 100*collisions + 10*covered + .01*(dx*dx+dy*dy)
            candidate = (cost, dx, dy, label.get_ha(), label.get_va(), box)
            if best is None or cost < best[0]:
                best = candidate
        _, dx, dy, ha, va, box = best
        label.set_position((dx, dy)); label.set_ha(ha); label.set_va(va)
        placed.append(box)


def render_dg_peak_map(selected: pd.DataFrame, occupancy: np.ndarray, record,
                       total_units: int, figure: Path, view_title: str) -> None:
    """Draw sampled peak bins with unit IDs and explicit collisions."""
    positions = selected.groupby(["peak_x_bin", "peak_y_bin"], as_index=False).agg(
        unit_ids=("unit", lambda v: ",".join(map(str, v))))
    fig, ax = plt.subplots(figsize=(7 * FIGURE_SCALE, 7.3 * FIGURE_SCALE), layout="constrained")
    palette = matplotlib.colors.ListedColormap(["#eef3f7"])
    palette.set_bad("#c9ced3")
    ax.imshow(np.where(occupancy > 0, 1., np.nan).T, origin="lower", cmap=palette,
              vmin=0, vmax=1, extent=(0, 19, 0, 19), interpolation="nearest")
    labels = []
    for row in positions.itertuples():
        ax.scatter(row.peak_x_bin + .5, row.peak_y_bin + .5, s=95,
                   color=COLORS[record.condition], edgecolor="white", linewidth=.8)
        # Left-edge peaks have labels inside the arena; others extend left.
        left = row.peak_x_bin < 3
        top = row.peak_y_bin >= 17
        labels.append(ax.annotate(row.unit_ids, (row.peak_x_bin + .5, row.peak_y_bin + .5),
                    xytext=(7 if left else -7, -7 if top else 6), textcoords="offset points",
                    ha="left" if left else "right", va="top" if top else "bottom", fontsize=12,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .4}))
    ax.set(xlim=(0, 19), ylim=(0, 19), xticks=[0, 9, 18], yticks=[0, 9, 18],
           xlabel="x bin", ylabel="y bin",
           title=f"{record.condition} · S{record.seed} · "
                 f"{'DG peaks' if view_title == 'all active DG peaks' else '30% dominance'}\n"
                 f"{len(selected)}/{total_units} units · {len(positions)} bins")
    ax.set_aspect("equal")
    pack_peak_labels(ax, labels)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)


def peak_sensitivity_outputs(data: pd.DataFrame, input_root: Path, output: Path) -> None:
    """Compare fixed dominance cutoffs and export peaks without field filtering.

    A peak is the maximum sampled mean activation, not proof of a single field.
    Keep eligibility fixed when changing dominance; include weakly sampled units
    in the separate all-active view and export their eligibility for inspection.
    """
    units = pd.read_csv(output / "mono_field_units.csv")
    raw = input_root / "corrected_core_candidates_20260902_place_fields/raw"
    sensitivity, counts, peak_tables = [], [], []
    cutoffs = (.20, .30, .40, .50, .60, .70, .80, .90)
    for record in data[data.condition.isin(["C01", "C05", "C15"])].itertuples():
        group = units[(units.condition == record.condition) & (units.seed == record.seed)].copy()
        path = raw / Path(record.field_file).parent.name / "place_fields.npz"
        with np.load(path, allow_pickle=False) as values:
            occupancy, maps = values["occupancy"], values["rate_maps"]
            group["peak_mean_activation"] = [
                float(maps[int(row.peak_x_bin), int(row.peak_y_bin), int(row.unit)])
                if row.peak_x_bin >= 0 else np.nan for row in group.itertuples()]
            group["peak_bin_observations"] = [
                int(occupancy[int(row.peak_x_bin), int(row.peak_y_bin)])
                if row.peak_x_bin >= 0 else 0 for row in group.itertuples()]
        selected = group[(group.active_fraction > 0) & (group.peak_x_bin >= 0) &
                         (group.peak_mean_activation > 0)]
        peak_tables.append(selected)
        for cutoff in cutoffs:
            qualifying = group.field_eligible & (group.mono_score >= cutoff)
            sensitivity.append({"condition": record.condition, "seed": record.seed,
                                "dominant_mass_cutoff": cutoff, "qualifying_units": int(qualifying.sum()),
                                "dg_units": len(group), "eligible_units": int(group.field_eligible.sum())})
        figure = output / f"{record.condition.lower()}_seed{record.seed}/all_active_dg_peaks.svg"
        render_dg_peak_map(selected, occupancy, record, len(group), figure, "all active DG peaks")
        relaxed = selected[selected.field_eligible & (selected.mono_score >= .30)]
        relaxed_figure = output / f"{record.condition.lower()}_seed{record.seed}/relaxed_30_dg_peaks.svg"
        render_dg_peak_map(relaxed, occupancy, record, len(group), relaxed_figure,
                           "DG peaks · 30% dominance")
        counts.append({"condition": record.condition, "seed": record.seed,
                       "active_units_with_peak": len(selected), "distinct_peak_bins": len(selected[["peak_x_bin", "peak_y_bin"]].drop_duplicates()),
                       "eligible_units": int(group.field_eligible.sum()), "dg_units": len(group),
                       "figure": str(figure.relative_to(output)),
                       "relaxed_30_units": len(relaxed),
                       "relaxed_30_figure": str(relaxed_figure.relative_to(output)),
                       "original_field_file": record.field_file})
    pd.concat(peak_tables, ignore_index=True).to_csv(output / "all_active_dg_peak_locations.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "all_active_dg_peak_counts.csv", index=False)
    sweep = pd.DataFrame(sensitivity)
    sweep.to_csv(output / "mono_field_sensitivity.csv", index=False)
    (output / "peak_view_method.json").write_text(json.dumps({
        "schema": "intrmotiv/dg-peak-sensitivity/v1",
        "dominant_mass_cutoffs": list(cutoffs), "relaxed_map_cutoff": .30,
        "classification_source": "mono_field_method.json",
        "source_hashes": "mono_field_peak_counts.csv",
        "eligibility": "Unchanged across the dominance sweep; not required for all-active maps",
        "all_active_selection": "active_fraction > 0 and a positive original rate-map maximum",
        "peak_statistic": "Original unsmoothed occupancy-corrected mean activation; first NumPy argmax on ties",
        "peak_bin_observations": "Saved occupancy count in the selected bin",
        "interpretation": "Relaxed dominance and all-active argmax are descriptive, not validated monofield classifications"
    }, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9 * FIGURE_SCALE, 5.5 * FIGURE_SCALE), layout="constrained")
    for condition in ("C01", "C05", "C15"):
        subset = sweep[sweep.condition == condition]
        for seed, group in subset.groupby("seed"):
            ax.plot(100 * group.dominant_mass_cutoff, group.qualifying_units,
                    color=COLORS[condition], alpha=.35, linewidth=1, marker=".")
        mean = subset.groupby("dominant_mass_cutoff").qualifying_units.mean()
        ax.plot(100 * mean.index, mean.values, color=COLORS[condition], marker="o",
                linewidth=2, label=condition)
    ax.set(xlabel="Required dominant mass (%)",
           ylabel="DG units (out of 16)", xticks=[20, 40, 60, 80],
           title="Field sensitivity · three seeds")
    ax.legend(title="Lines: seeds / mean", loc="upper right")
    fig.savefig(output / "mono_field_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)


def trajectory_occupancy_figure(pose: pd.DataFrame, occupancy: np.ndarray, destination: Path,
                                condition: str, seed: int) -> None:
    """Render occupancy beside continuous within-episode paths, preserving every pose."""
    fig, axes = plt.subplots(1, 2, figsize=(13 * FIGURE_SCALE, 6 * FIGURE_SCALE), constrained_layout=True)
    occupancy_image = axes[0].imshow(np.ma.masked_equal(occupancy.T, 0), origin="lower",
                                     cmap="cividis", extent=(100, 2000, 100, 2000),
                                     interpolation="nearest")
    fig.colorbar(occupancy_image, ax=axes[0], label="Observations", shrink=.8)
    for segment in trajectory_segments(pose):
        line, = axes[1].plot(segment.x, segment.y, lw=.55, alpha=.65, color=COLORS[condition])
        line.get_path().should_simplify = False
    for ax in axes:
        ax.set(xlim=(100, 2000), ylim=(100, 2000), aspect="equal",
               xlabel="x (DMLab units)", ylabel="y (DMLab units)")
    axes[0].set_title("Occupancy")
    axes[1].set_title(f"Trajectory · {pose.num_traj.nunique()} segments")
    fig.suptitle(f"{condition} · S{seed} · 10,000-decision frozen probe")
    fig.savefig(destination / "trajectory_occupancy.svg", bbox_inches="tight")
    plt.close(fig)




def place_field_figure(maps: np.ndarray, occupancy: np.ndarray, information: np.ndarray,
                       destination: Path, condition: str, seed: int, *,
                       individual_scale: bool = True) -> list[dict]:
    """Render the same top-four units with shared or independent raw activation limits."""
    valid = np.flatnonzero(np.isfinite(information) & (np.nanmax(maps, axis=(0, 1)) > 0))
    selected = valid[np.argsort(information[valid])[-4:][::-1]]
    fig, axes = plt.subplots(2, 2, figsize=(9 * FIGURE_SCALE, 9 * FIGURE_SCALE),
                             sharex=True, sharey=True, constrained_layout=True)
    shared_max = float(np.nanmax(maps[:, :, selected]))
    limits = []
    for index, (ax, unit) in enumerate(zip(axes.flat, selected)):
        vmax = float(np.nanmax(maps[:, :, unit])) if individual_scale else shared_max
        limits.append({"condition": condition, "seed": seed, "rank": index + 1,
                       "unit": int(unit), "spatial_score": float(information[unit]),
                       "color_min": 0., "color_max": vmax})
        field_image = ax.imshow(np.ma.masked_where(occupancy.T == 0, maps[:, :, unit].T),
                                origin="lower", vmin=0, vmax=vmax, cmap="viridis",
                                interpolation="nearest")
        ax.set(title=f"DG {unit}: {information[unit]:.2f}",
               xlabel="x bin" if index >= 2 else None,
               ylabel="y bin" if index % 2 == 0 else None)
        if individual_scale:
            fig.colorbar(field_image, ax=ax, ticks=[0, vmax], format="%.2g",
                         shrink=.75, fraction=.07, pad=.035)
    if not individual_scale:
        fig.colorbar(field_image, ax=axes, label="Mean DG activation", shrink=.75)
    if individual_scale:
        fig.suptitle(f"{condition} · S{seed} · top four DG maps\nMean activation · each scale: 0 to unit max")
    else:
        fig.suptitle(f"{condition} · S{seed} · top four DG maps · shared scale")
    filename = "place_fields.svg" if individual_scale else "place_fields_shared_scale.svg"
    fig.savefig(destination / filename, bbox_inches="tight")
    if individual_scale:
        # Keep the explicit report links compatible with the primary figure.
        (destination / "place_fields_individual_scale.svg").write_bytes(
            (destination / filename).read_bytes())
    plt.close(fig)

    return limits


def exemplar(data: pd.DataFrame, input_root: Path, output: Path, condition: str,
             seed: int = 99, *, reuse_saved_kernel: bool = False) -> None:
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
    for scope in ("full", "first_episode"):
        render_trajectory(pose, destination / f"trajectory_{scope}.svg",
                          f"{condition} · {CONDITIONS[condition]} · seed {seed}",
                          scope, (100, 2000, 100, 2000), sampling_label="archived probe",
                          figure_scale=FIGURE_SCALE, compact_title=f"{condition} · S{seed}")

    # Reuse the established flow computation and renderer, preserving its cell contract.
    flow_cells(pose).to_csv(destination / "flow_cells.csv", index=False)
    render_flow(pose, destination / "flow.svg", f"{condition} · S{seed} · local flow",
                figure_scale=FIGURE_SCALE)

    trajectory_occupancy_figure(pose, occupancy, destination, condition, seed)

    place_field_figure(maps, occupancy, information, destination, condition, seed, individual_scale=False)
    place_field_figure(maps, occupancy, information, destination, condition, seed, individual_scale=True)

    saved_kernel = destination / "dg_kernel.npz"
    if reuse_saved_kernel and saved_kernel.exists():
        with np.load(saved_kernel) as values:
            kernel, pairs = values["correlation"], values["pairs"]
    else:
        from collect_poster_population_kernels import correlation_kernel
        kernel, pairs = correlation_kernel(maps, occupancy, radius=8, min_visits=5)
        kernel[pairs < 10] = np.nan
        kernel[8, 8] = np.nan
        np.savez_compressed(saved_kernel, correlation=kernel, pairs=pairs)
    fig, ax = plt.subplots(figsize=(8 * FIGURE_SCALE, 6 * FIGURE_SCALE), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(kernel.T), origin="lower", extent=(-8.5, 8.5, -8.5, 8.5),
                      cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="DG population-vector r")
    ax.set(xlabel="x offset (bins)", ylabel="y offset (bins)",
           title=f"{condition} seed {seed} · DG spatial kernel")
    fig.savefig(destination / "dg_kernel.svg", bbox_inches="tight")
    plt.close(fig)


def graph_figure(input_root: Path, output: Path, seed: int = 99,
                 graph_path: Path | None = None) -> None:
    """Show the checkpoint-stored edge evidence for the two goal exemplars."""
    with (graph_path or input_root / "stored_graph_seed99.json").open() as stream:
        records = json.load(stream)
    fig, axes = plt.subplots(1, 2, figsize=(12 * FIGURE_SCALE, 5.5 * FIGURE_SCALE), constrained_layout=True)
    records = [r for r in records if r["seed"] == seed]
    if len(records) != 2:
        raise ValueError(f"Expected C05/C15 graph records for seed {seed}")
    for ax, record in zip(axes, records):
        attempts = np.asarray(record["control_attempts"], dtype=float)
        confidence = np.asarray(record["edge_confidence"], dtype=float)
        ratio = np.divide(confidence, attempts, out=np.zeros_like(confidence), where=attempts > 0)
        np.fill_diagonal(ratio, np.nan)
        ratio[attempts == 0] = np.nan
        image = ax.imshow(np.ma.masked_invalid(ratio), cmap="viridis", vmin=0, vmax=1,
                          interpolation="nearest")
        ax.set(xlabel="Target DG", ylabel="Source DG", title=f"{record['condition']} · S{seed}")
        ax.set_xticks(range(0, 16, 3))
        ax.set_yticks(range(0, 16, 3))
    fig.colorbar(image, ax=axes, label="Confidence / attempts", shrink=.78)
    fig.suptitle("Stored graphs · white: no attempt")
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


def load_goal_trajectory_streams(input_root: Path, output: Path) -> dict[str, pd.DataFrame]:
    """Validate full replay streams against published summaries and event tables."""
    streams = {}
    for condition in ("C05", "C15"):
        folder = f"{condition.lower()}_seed99"
        source = input_root / folder / "pose_events.csv"
        if not source.exists():
            raise FileNotFoundError(f"Stage the full {source}; set --goal-events-input. "
                                    "Do not substitute the sparse option_events.csv.")
        pose = pd.read_csv(source)
        summary = json.loads((output / folder / "event_summary.json").read_text())
        validate_complete_event_stream(pose, summary["recorded_observations"])
        if int(pose.option_start.sum()) != summary["option_starts"] or int(pose.goal_hit.sum()) != summary["goal_hits"]:
            raise ValueError(f"Event counts differ from the published summary for {condition}")
        if pose.num_traj.nunique() != summary["reset_segments"]:
            raise ValueError(f"Episode counts differ for {condition}")
        expected_events = pd.read_csv(output / folder / "option_events.csv")
        actual_events = pose.loc[pose.option_start | pose.goal_hit, expected_events.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_events, expected_events, check_dtype=False)
        pose.attrs["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        streams[condition] = pose
    return streams


def render_goal_trajectory_streams(streams: dict[str, pd.DataFrame], output: Path) -> None:
    """Render complete replay paths and save their source/segmentation record."""
    provenance = json.loads((output / "goal_option_provenance.json").read_text())
    records = []
    for condition, pose in streams.items():
        destination = output / f"{condition.lower()}_seed99"
        plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13})
        for scope in ("full", "first_episode"):
            render_trajectory(pose, destination / f"trajectory_option_events_{scope}.svg",
                              f"{condition} · S99", scope, (100, 2000, 100, 2000),
                              figure_scale=FIGURE_SCALE, compact_title=f"{condition} · S99")
        first = next(trajectory_segments(pose))
        records.append({"condition": condition, "seed": 99, "observations": len(pose),
                        "first_episode_observations": len(first),
                        "contiguous_paths": sum(1 for _ in trajectory_segments(pose)),
                        "source_sha256": pose.attrs["source_sha256"],
                        "option_starts": int(pose.option_start.sum()), "goal_hits": int(pose.goal_hit.sum()),
                        "original_pose_events_file": str(Path(provenance["analysis_root"]) /
                                                          "output/raw" / destination.name / "pose_events.csv")})
    (output / "goal_option_trajectory_rendering.json").write_text(json.dumps({
        "schema": "intrmotiv/goal-option-trajectory-render/v1",
        "line_source": "Complete observation-time pose_events.csv, not event-only rows",
        "segmentation": "Separate agents and contiguous episode runs; never connect across reset labels or skipped frames",
        "validation": "Observation count, reset count, and event rows match published summary and marker CSV",
        "runs": records,
    }, indent=2) + "\n")


def main() -> None:
    global FIGURE_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exemplar-seeds", nargs="+", type=int, default=[99])
    parser.add_argument("--graphs", type=Path, help="Compact checkpoint graph JSON")
    parser.add_argument("--figure-scale", type=float, default=FIGURE_SCALE,
                        help="Canvas width/height multiplier; text and strokes remain unchanged")
    parser.add_argument("--individual-fields-only", action="store_true",
                        help="Update default top-four fields with separate per-unit 0-to-max scales")
    parser.add_argument("--goal-events-input", type=Path,
                        help="Staged full event replay streams: c05_seed99/pose_events.csv and c15_seed99/pose_events.csv")
    parser.add_argument("--trajectories-only", action="store_true",
                        help="Re-render archived trajectories and full goal-event replays only")
    parser.add_argument("--compact-only", action="store_true",
                        help="Re-render saved comparison plots with compact canvases")
    parser.add_argument("--mono-only", action="store_true",
                        help="Reuse saved comparison rows and add/link mono-field peak maps")
    args = parser.parse_args()
    if not np.isfinite(args.figure_scale) or args.figure_scale <= 0:
        parser.error("--figure-scale must be finite and positive")
    FIGURE_SCALE = args.figure_scale
    setup_style()
    if args.individual_fields_only:
        data = pd.read_csv(args.output / "matched_terminal_per_run.csv")
        plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})
        limits = []
        for row in data[data.condition.isin(["C01", "C05", "C15"])].itertuples():
            raw = args.input / "corrected_core_candidates_20260902_place_fields/raw" / Path(row.field_file).parent.name
            with np.load(raw / "place_fields.npz", allow_pickle=False) as values:
                rows = place_field_figure(values["rate_maps"], values["occupancy"], values["spatial_information"],
                                          args.output / f"{row.condition.lower()}_seed{row.seed}",
                                          row.condition, row.seed, individual_scale=True)
            source_hash = hashlib.sha256((raw / "place_fields.npz").read_bytes()).hexdigest()
            for entry in rows:
                entry["field_sha256"] = source_hash
                entry["original_field_file"] = row.field_file
                entry["figure"] = f"{row.condition.lower()}_seed{row.seed}/place_fields_individual_scale.svg"
            limits.extend(rows)
        pd.DataFrame(limits).to_csv(args.output / "place_field_individual_scales.csv", index=False)
        return
    if args.compact_only or args.trajectories_only:
        # Validate before replacing any SVGs, so missing streams cannot leave a partial rerender.
        streams = load_goal_trajectory_streams(args.goal_events_input or args.input / "goal_option_events", args.output)
    if args.trajectories_only:
        data = pd.read_csv(args.output / "matched_terminal_per_run.csv")
        for condition in ("C01", "C05", "C15"):
            for seed in (8, 99, 123):
                row = data[(data.condition == condition) & (data.seed == seed)].iloc[0]
                raw = args.input / "corrected_core_candidates_20260902_place_fields/raw" / Path(row.pose_file).parent.name
                pose = pd.read_csv(raw / "pose.csv")
                if condition != "C01" or seed != 8:
                    # These published archived plots inherit the flow renderer's style.
                    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})
                for scope in ("full", "first_episode"):
                    render_trajectory(pose, args.output / f"{condition.lower()}_seed{seed}/trajectory_{scope}.svg",
                                      f"{condition} · S{seed}", scope, (100, 2000, 100, 2000),
                                      sampling_label="archived probe", figure_scale=FIGURE_SCALE,
                                      compact_title=f"{condition} · S{seed}")
                with np.load(raw / "place_fields.npz", allow_pickle=False) as values:
                    occupancy = values["occupancy"]
                plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})
                trajectory_occupancy_figure(pose, occupancy, args.output / f"{condition.lower()}_seed{seed}",
                                            condition, seed)
        render_goal_trajectory_streams(streams, args.output)
        return
    if args.compact_only:
        data = pd.read_csv(args.output / "matched_terminal_per_run.csv")
        mono_field_outputs(data, args.input, args.output)
        peak_sensitivity_outputs(data, args.input, args.output)
        summary_figure(data, args.output / "architecture_summary.svg")
        summary_figure(data, args.output / "c01_c05_c15_summary.svg", ("C01", "C05", "C15"), width_scale=.6)
        for condition in ("C01", "C05", "C15"):
            for seed in (8, 99, 123):
                exemplar(data, args.input, args.output, condition, seed, reuse_saved_kernel=True)
        for seed in (8, 99, 123):
            graph_figure(args.input, args.output / f"stored_graphs_seed{seed}.svg", seed, args.graphs)
        render_goal_trajectory_streams(streams, args.output)
        return
    if args.mono_only:
        mono_field_outputs(pd.read_csv(args.output / "matched_terminal_per_run.csv"),
                           args.input, args.output)
        peak_sensitivity_outputs(pd.read_csv(args.output / "matched_terminal_per_run.csv"),
                                 args.input, args.output)
        return
    data = build_table(args.input, args.output)
    mono_field_outputs(data, args.input, args.output)
    peak_sensitivity_outputs(data, args.input, args.output)
    paired_changes(data, args.output / "c01_paired_changes.csv")
    summary_figure(data, args.output / "architecture_summary.svg")
    summary_figure(data, args.output / "c01_c05_c15_summary.svg", ("C01", "C05", "C15"), width_scale=.6)
    for condition in ("C01", "C05", "C15"):
        for seed in args.exemplar_seeds:
            exemplar(data, args.input, args.output, condition, seed)
    for seed in args.exemplar_seeds:
        graph_figure(args.input, args.output / f"stored_graphs_seed{seed}.svg", seed, args.graphs)
    older_batch_check(args.input, args.output / "older_batch_seed99_check.csv")
    print(data.groupby("condition")[["coverage_auc_terminal", "return_20_mobile",
                                     "return_40_mobile"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
