"""Render the final-size COSYNE results figure.

All plotted comparisons come from archived seed-level tables. The selected
seed-8 example uses only the lightweight peak coordinates and occupancy mask
derived from two hash-verified NEMO2 evaluator arrays; raw NPZs are not stored
in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd


DRAFT = Path(__file__).resolve().parent
ROOT = DRAFT.parents[1]
RESULTS = ROOT / "06_experiments" / "results"
FIGURES = DRAFT / "figures"
DATA = DRAFT / "data"
SOURCES = {
    "online": RESULTS / "cosyne_2027_online_wandb" / "arm_summaries.csv",
    "behavior": DATA / "fresh_behavior_training_seeds.csv",
    "behavior_absolute": DATA / "fresh_behavior_decoder1_absolute.csv",
    "encoder": RESULTS / "cosyne_2027_checkpoint_fields" / "common_panel_analysis" / "paired_encoder_fields.csv",
    "peaks": RESULTS / "cosyne_2027_checkpoint_fields" / "common_panel_analysis" / "paired_peak_spread_by_seed.csv",
    "example": DATA / "c01_seed8_peak_example.json",
    "decoder": RESULTS / "cosyne_2027_temporal_credit_layers" / "paired_by_seed.csv",
    "worker": RESULTS / "cosyne_2027_online_wandb" / "paired_contrasts_by_seed.csv",
    "longer_credit_online": RESULTS.parent / "dg_goal_control" / "results" / "episode_long_comparison_20261009" / "longer_credit_100m_paired.csv",
    "longer_credit_fields": RESULTS.parent / "dg_goal_control" / "results" / "episode_long_comparison_20261009" / "longer_credit_100m_frozen_fields_paired.csv",
}
COLOR = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73",
         "P": "#7B4FA3"}
GRAY = "#5B646C"
FIGURE_HEIGHT = 4.48
CONTENT_SCALE = 0.88  # Preserve horizontal panel widths and type size.
CONTENT_OFFSET = -0.43  # Crop unused space below the bottom-row labels.


def style() -> str:
    font = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    if Path(font).suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"Arial did not resolve to a scalable font: {font}")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "svg.hashsalt": "COSYNE-2027-submission-panels",
        "pdf.fonttype": 42,
    })
    return font


def save(fig: plt.Figure, stem: str, directory: Path = FIGURES) -> None:
    directory.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        target = directory / f"{stem}.{suffix}"
        metadata = ({"Date": None} if suffix == "svg" else
                    {"CreationDate": None, "ModDate": None} if suffix == "pdf" else
                    None)
        fig.savefig(target, dpi=300,
                    metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in
                                        target.read_text().splitlines()) + "\n")
    plt.close(fig)


def axes_in(fig: plt.Figure, x: float, y: float, width: float, height: float):
    return fig.add_axes([x / 7.5, content_y(y) / FIGURE_HEIGHT,
                         width / 7.5, height * CONTENT_SCALE / FIGURE_HEIGHT])


def content_y(y: float) -> float:
    return y * CONTENT_SCALE + CONTENT_OFFSET


def section(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(x / 7.5, content_y(y) / FIGURE_HEIGHT,
             label, fontsize=10, fontweight="bold",
             color="#20272C", va="top")


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, pad=1.5)
    ax.grid(axis="y", color="#E6E8E9", lw=0.5, zorder=0)


def paired_rows(ax: plt.Axes, values: pd.DataFrame, groups: list[str],
                control_column: str, experimental_column: str,
                *, labels: bool, xlim: tuple[float, float],
                marker_size: float = 10, mode: str = "paired") -> None:
    """Show vertical paired values or paired-difference boxes with seed dots."""
    assert mode in {"paired", "box"}
    observed = values[[control_column, experimental_column]].to_numpy(dtype=float)
    assert np.isfinite(observed).all()
    if mode == "paired":
        assert observed.min() >= xlim[0] - 1e-9 and observed.max() <= xlim[1] + 1e-9, (
            f"Observed values {observed.min():.4g}–{observed.max():.4g} "
            f"exceed vertical limits {xlim}")
    for row, group in enumerate(groups):
        data = values[values.group == group].sort_values("seed")
        if data.empty:
            continue
        x_positions = row + np.linspace(-0.12, 0.12, len(data))
        color = COLOR[group.split("-")[0]]
        if mode == "paired":
            for x, control, experimental in zip(
                    x_positions, data[control_column], data[experimental_column]):
                ax.plot([x, x], [control, experimental], color="#BDC5C9",
                        lw=0.55, zorder=1)
            ax.scatter(x_positions, data[control_column], s=marker_size,
                       marker="s", facecolor="white", edgecolor=GRAY,
                       linewidth=0.75, zorder=3)
            ax.scatter(x_positions, data[experimental_column], s=marker_size,
                       color=color, linewidth=0, zorder=4)
        else:
            differences = (data[experimental_column] - data[control_column]).to_numpy()
            box = ax.boxplot([differences], positions=[row], widths=0.48,
                             whis=(0, 100), showfliers=False, patch_artist=True,
                             manage_ticks=False)
            box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.8)
            for item in box["whiskers"] + box["caps"]:
                item.set(color=color, linewidth=0.75)
            box["medians"][0].set(color=color, linewidth=1.1)
            ax.scatter(x_positions, differences, s=marker_size * 0.7,
                       color=color, edgecolor="white", linewidth=0.2, zorder=4)
    ax.set_xlim(-0.42, len(groups) - 0.58)
    if mode == "paired":
        ax.set_ylim(*xlim)
    else:
        differences = (values[experimental_column] - values[control_column]).to_numpy()
        extent = max(abs(float(differences.min())), abs(float(differences.max())))
        extent = max(extent * 1.18, 0.01)
        ax.set_ylim(-extent, extent)
        ax.axhline(0, color="#7D858A", lw=0.7, zorder=1)
    short = {"C05-wT": "05T", "C05-wC": "05C",
             "C05-eT": "05T", "C05-eC": "05C"}
    ax.set_xticks(range(len(groups)),
                  [short.get(group, group) for group in groups] if labels else [""] * len(groups))
    ax.tick_params(axis="x", labelsize=8 if len(groups) > 3 else 9)
    finish_axes(ax)


def peak_grid(peaks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    centers = np.arange(150.0, 2000.0, 100.0)
    gx, gy = np.meshgrid(centers, centers)
    squared = (gx[None] - peaks[:, 0, None, None]) ** 2
    squared += (gy[None] - peaks[:, 1, None, None]) ** 2
    grid = np.exp(-squared / (2 * 100.0**2)).sum(axis=0)
    grid[~mask] = 0
    return grid / grid.sum()


def results_figure(*, mode: str = "paired", output_dir: Path = FIGURES,
                   stem: str = "results_sections") -> dict[str, float]:
    online = pd.read_csv(SOURCES["online"])
    behavior = pd.read_csv(SOURCES["behavior"])
    behavior_absolute = pd.read_csv(SOURCES["behavior_absolute"])
    encoder = pd.read_csv(SOURCES["encoder"])
    peaks = pd.read_csv(SOURCES["peaks"])
    decoder = pd.read_csv(SOURCES["decoder"])
    worker = pd.read_csv(SOURCES["worker"])
    longer_online = pd.read_csv(SOURCES["longer_credit_online"])
    longer_fields = pd.read_csv(SOURCES["longer_credit_fields"])
    example = json.loads(SOURCES["example"].read_text())
    mask = np.asarray(example["visited_bin_mask"], dtype=bool)
    assert mask.shape == (19, 19) and mask.sum() == 242
    baseline_area = peaks[(peaks.contrast == "C01") &
                          (peaks.metric == "effective_area_fraction") &
                          (peaks.cutoff == 0.0025)]
    selected = baseline_area[baseline_area.seed == example["seed"]].iloc[0]
    assert len(baseline_area) == 5
    assert selected.difference == baseline_area.difference.max()
    assert np.isclose(selected.temporal, example["arms"]["temporal"]["effective_area_fraction"])
    assert np.isclose(selected.constant, example["arms"]["constant"]["effective_area_fraction"])

    fresh = behavior[(behavior.cohort == "fresh") &
                     (behavior.layer == "decoder_1")].dropna(
                         subset=["information_difference", "common_split_half_difference"])
    keys = ["condition", "family", "encoder_credit", "worker_bonus", "training_seed"]
    checked = behavior_absolute.merge(fresh, on=keys, validate="one_to_one")
    assert len(checked) == 38
    assert np.allclose(checked.own_info - checked.random_info,
                       checked.information_difference, atol=1e-10)
    assert np.allclose(checked.own_reliability - checked.random_reliability,
                       checked.common_split_half_difference, atol=1e-10)

    fig = plt.figure(figsize=(7.5, FIGURE_HEIGHT), facecolor="white")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], linestyle="none", marker="s", markersize=4,
               markerfacecolor="white", markeredgecolor=GRAY, label="Control"),
        Line2D([], [], linestyle="none", marker="o", markersize=4,
               color="#30363B", label="Condition"),
    ]
    if mode == "paired":
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 1.005),
                   ncol=2, frameon=False, fontsize=9, handletextpad=0.2,
                   columnspacing=0.5)

    # A: five-seed online course. B: absolute random-control versus own-policy
    # decoder values, with two repeats averaged inside each training seed.
    section(fig, 0.12, 5.46, "A  Spatial emergence")
    section(fig, 2.24, 5.46, "B  Behavior shapes spatial coding")
    ax = axes_in(fig, 0.58, 4.24, 1.42, 0.94)
    for family, worker_arm in (("C01", "inapplicable"), ("C05", "temporal"),
                               ("C15", "temporal")):
        rows = online[(online.family == family) &
                      (online.encoder_credit == "temporal") &
                      (online.worker_bonus == worker_arm) &
                      (online.metric == "spatial_information")].sort_values("env_steps")
        assert len(rows) > 50 and rows.n.eq(5).all()
        x = rows.env_steps.to_numpy() / 1e6
        mean = rows["mean"].to_numpy()
        sem = rows["sem"].to_numpy()
        ax.plot(x, mean, lw=1.1, color=COLOR[family], label=family)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.13, color=COLOR[family])
    ax.set(xlim=(0, 100), xlabel="Frames (millions)", ylabel="DG score")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3,
              handlelength=0.7, columnspacing=0.3, handletextpad=0.2,
              borderpad=0)
    finish_axes(ax)

    behavior_rows = behavior_absolute.rename(columns={"training_seed": "seed"}).copy()
    behavior_rows["group"] = behavior_rows.family
    behavior_groups = ["C01", "C05", "C15"]
    for x0, control, experimental, title, limits, xlabel in [
        (2.92, "random_info", "own_info", "Readout MLP layer 1", (0, 1.9), "Bits/activation"),
        (5.23, "random_reliability", "own_reliability",
         "Map reliability", (0, 0.9), "Split-half correlation"),
    ]:
        ax = axes_in(fig, x0, 4.20, 1.43, 0.98)
        paired_rows(ax, behavior_rows, behavior_groups, control, experimental,
                    labels=True, xlim=limits, marker_size=8, mode=mode)
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_ylabel(xlabel, fontsize=8, labelpad=1)

    fig.add_artist(plt.Line2D([0.02, 0.98],
                              [content_y(3.85) / FIGURE_HEIGHT] * 2,
                              transform=fig.transFigure, color="#CCD1D4", lw=0.6))
    section(fig, 0.12, 3.76, "C–D  Temporal push: DG fields")
    fig.text(0.38 / 7.5, content_y(3.51) / FIGURE_HEIGHT,
             "C  Selected C01 example: seed 8 (largest of 5 pairs)",
             fontsize=9, color=GRAY)

    # C: peak-coordinate example. D: original absolute values for every
    # matched encoder pair, grouped by architecture and fixed worker arm.
    grids = {arm: peak_grid(np.asarray(example["arms"][arm]["peaks_xy"]), mask)
             for arm in ("temporal", "constant")}
    vmax = max(np.nanmax(value) for value in grids.values())
    for x0, arm, label in [(0.38, "temporal", "Temporal"),
                           (1.62, "constant", "Constant")]:
        ax = axes_in(fig, x0, 2.51, 1.04, 0.83)
        grid = np.ma.masked_where(~mask, grids[arm])
        cmap = plt.get_cmap("Greys").copy()
        cmap.set_bad("#DDDDDD")
        ax.imshow(grid, origin="lower", extent=(100, 2000, 100, 2000),
                  cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        xy = np.asarray(example["arms"][arm]["peaks_xy"])
        ax.scatter(xy[:, 0], xy[:, 1], s=8, color="#D55E00",
                   edgecolor="white", linewidth=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        area = example["arms"][arm]["effective_area_fraction"]
        ax.set_title(f"{label} ({area:.3f})", fontsize=9, pad=2)

    contrasts = ["C01", "C05 · temporal hit", "C05 · constant hit", "C15"]
    row_names = ["C01", "C05-wT", "C05-wC", "C15"]
    metrics = [
        ("DG score", encoder, "mean_spatial_information_bits", (0, 0.23)),
        ("Map cosine", encoder, "active_map_cosine_mean", (0, 0.17)),
        ("Peak area", peaks, "effective_area_fraction", (0, 0.48)),
    ]
    for index, (title, source, metric, limits) in enumerate(metrics):
        x0 = (3.62, 4.91, 6.20)[index]
        ax = axes_in(fig, x0, 2.43, 0.86, 0.96)
        subset = source[source.metric == metric].copy()
        if source is peaks:
            subset = subset[subset.cutoff == 0.0025]
        subset["group"] = subset.contrast.map(dict(zip(contrasts, row_names)))
        assert len(subset) == 20 and subset.group.notna().all()
        assert np.allclose(subset.temporal - subset.constant,
                           subset.difference, atol=1e-10)
        paired_rows(ax, subset, row_names, "constant", "temporal",
                    labels=True, xlim=limits, marker_size=10, mode=mode)
        ax.set_title(title, fontsize=10, pad=2)
        ax.locator_params(axis="y", nbins=3)

    fig.add_artist(plt.Line2D([0.02, 0.98],
                              [content_y(2.20) / FIGURE_HEIGHT] * 2,
                              transform=fig.transFigure, color="#CCD1D4", lw=0.6))
    section(fig, 0.12, 2.10, "E  Temporal push: readout")
    section(fig, 2.72, 2.10, "F  Temporal pull: two interventions")
    downstream = decoder[(decoder.factor == "encoder") &
                         (decoder.layer == "decoder_1") &
                         (decoder.metric == "mean_bits_per_activation_eligible")].copy()
    downstream["group"] = downstream.apply(
        lambda row: (f"C05-w{row.fixed_arm[0].upper()}"
                     if row.family == "C05" else row.family), axis=1)
    decoder_groups = ["C01", "C05-wT", "C05-wC", "C15"]
    assert downstream.groupby("group").size().reindex(decoder_groups).tolist() == [5, 5, 5, 5]
    assert downstream.groupby("group").difference.apply(
        lambda values: values.gt(0).sum()).reindex(decoder_groups).tolist() == [5, 5, 4, 4]
    assert np.allclose(downstream.temporal - downstream.constant,
                       downstream.difference, atol=1e-10)
    ax = axes_in(fig, 0.67, 0.89, 1.63, 0.88)
    paired_rows(ax, downstream, decoder_groups, "constant", "temporal",
                labels=True, xlim=(0, 3.0), marker_size=10, mode=mode)
    ax.set_title("Readout MLP layer 1", fontsize=10, pad=2)
    ax.set_ylabel("Bits/activation", fontsize=8, labelpad=1)

    pull = worker[(worker.contrast == "temporal_minus_constant_hit_bonus") &
                  (worker.family == "C05") &
                  (worker.metric.isin(["visited_cell_fraction",
                                     "spatial_information"]))].copy()
    pull["group"] = pull.fixed_factor.map({"temporal": "C05-eT", "constant": "C05-eC"})
    assert pull.groupby(["metric", "group"]).size().eq(5).all() and len(pull) == 20
    assert np.allclose(pull["first"] - pull["second"], pull.difference, atol=1e-10)
    for x0, metric, title, limits, xlabel, labels in [
        (3.02, "visited_cell_fraction", "Hit: coverage", (0.52, 0.92), "Visited frac.", True),
        (4.07, "spatial_information", "Hit: DG score", (0.08, 0.23), "DG score", False),
    ]:
        ax = axes_in(fig, x0, 0.89, 0.79, 0.88)
        paired_rows(ax, pull[pull.metric == metric], ["C05-eT", "C05-eC"],
                    "second", "first", labels=True, xlim=limits, marker_size=10,
                    mode=mode)
        ax.set_title(title, fontsize=9, pad=2)
        ax.locator_params(axis="y", nbins=3)

    # Longer PPO credit changes gamma and GAE lambda together. Its online
    # coverage and policy-driven frozen-field samples are separate evaluations.
    longer = longer_online.merge(longer_fields, on=["dg_family", "seed"],
                                 validate="one_to_one")
    assert len(longer) == 6 and longer.groupby("dg_family").size().eq(3).all()
    assert np.allclose(longer.longer_coverage_auc - longer.baseline_coverage_auc,
                       longer.coverage_auc_delta, atol=1e-10)
    assert np.allclose(longer.learned_si_bits_after - longer.learned_si_bits_before,
                       longer.learned_si_bits_delta, atol=1e-10)
    assert longer.coverage_auc_delta.gt(0).all()
    assert longer.learned_si_bits_delta.lt(0).all()
    longer["group"] = longer.dg_family.map({"Prescribed": "P", "C15": "C15"})
    assert longer.group.notna().all()
    for x0, control, experimental, title, limits, xlabel, labels in [
        (5.35, "baseline_coverage_auc", "longer_coverage_auc",
         "γ,λ↑: coverage", (40, 95), "AUC (points)", True),
        (6.46, "learned_si_bits_before", "learned_si_bits_after",
         "γ,λ↑: DG info", (0.09, 0.22), "SI (bits)", False),
    ]:
        ax = axes_in(fig, x0, 0.89, 0.78, 0.88)
        paired_rows(ax, longer, ["P", "C15"], control, experimental,
                    labels=True, xlim=limits, marker_size=11, mode=mode)
        ax.set_title(title, fontsize=9, pad=2)
        ax.locator_params(axis="y", nbins=3)

    save(fig, stem, output_dir)
    return {
        "selected_seed": 8,
        "selected_peak_area_temporal": example["arms"]["temporal"]["effective_area_fraction"],
        "selected_peak_area_constant": example["arms"]["constant"]["effective_area_fraction"],
        "behavior_models": len(behavior_absolute),
        "decoder_delta_by_group": downstream.groupby("group").difference.mean().to_dict(),
        "worker_delta_by_metric_and_group": {
            f"{metric}:{group}": float(value) for (metric, group), value in
            pull.groupby(["metric", "group"]).difference.mean().items()},
        "longer_credit_delta_by_family": {
            family: {"coverage_auc": float(rows.coverage_auc_delta.mean()),
                     "learned_si_bits": float(rows.learned_si_bits_delta.mean())}
            for family, rows in longer.groupby("dg_family")},
        "result_panel_width_inches": {
            "behavior": 1.43, "dg": 0.86, "decoder": 1.63,
            "worker": 0.79, "longer_credit": 0.78},
        "orientation": "vertical",
        "summary_mark": "paired seeds" if mode == "paired" else "paired-difference boxes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("pairs", "boxes"),
                        help="Export a compact layout candidate without changing the submission figure")
    args = parser.parse_args()
    font = style()
    if args.candidate:
        output_dir = DRAFT / "figure2_candidates"
        mode = "paired" if args.candidate == "pairs" else "box"
        stem = f"{'05_compact_vertical_pairs' if mode == 'paired' else '06_compact_vertical_boxes'}"
        summary = results_figure(mode=mode, output_dir=output_dir, stem=stem)
        print(json.dumps(summary, indent=2))
        return
    summary = results_figure()
    manifest = {
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in SOURCES.items()},
        "font_path": font,
        "results_size_inches": [7.5, FIGURE_HEIGHT],
        "figure_values": summary,
        "peak_example_choice": "largest C01 peak-area difference among five paired seeds",
        "behavior_absolute_source": (
            "6412596:06_experiments/results/cosyne_2027_behavior_secondary/"
            "fresh_pair_analysis/paired_evaluation_repeats.csv"),
    }
    (DRAFT / "PANEL_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
