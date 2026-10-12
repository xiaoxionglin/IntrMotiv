"""Render the final-size COSYNE results figure.

All plotted comparisons come from archived seed-level tables. The selected
seed-8 example uses only the lightweight peak coordinates and occupancy mask
derived from two hash-verified NEMO2 evaluator arrays; raw NPZs are not stored
in this repository.
"""

from __future__ import annotations

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
}
COLOR = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73"}
GRAY = "#5B646C"
FIGURE_HEIGHT = 4.95
CONTENT_SCALE = 0.88  # Fit all data panels at their original widths and type sizes.
CONTENT_OFFSET = 0.05


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


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        target = FIGURES / f"{stem}.{suffix}"
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
    ax.grid(axis="x", color="#E6E8E9", lw=0.5, zorder=0)


def paired_rows(ax: plt.Axes, values: pd.DataFrame, groups: list[str],
                control_column: str, experimental_column: str,
                *, labels: bool, xlim: tuple[float, float],
                marker_size: float = 10) -> None:
    """Show actual control and experimental values joined within training seed."""
    for row, group in enumerate(groups):
        data = values[values.group == group].sort_values("seed")
        if data.empty:
            continue
        y_positions = row + np.linspace(-0.17, 0.17, len(data))
        color = COLOR["C01" if group == "C01" else
                      "C15" if group == "C15" else "C05"]
        for y, control, experimental in zip(
                y_positions, data[control_column], data[experimental_column]):
            ax.plot([control, experimental], [y, y], color="#BDC5C9",
                    lw=0.55, zorder=1)
        ax.scatter(data[control_column], y_positions, s=marker_size,
                   marker="s", facecolor="white", edgecolor=GRAY,
                   linewidth=0.75, zorder=3)
        ax.scatter(data[experimental_column], y_positions, s=marker_size,
                   color=color, linewidth=0, zorder=4)
    ax.set_ylim(-0.42, len(groups) - 0.58)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_yticks(range(len(groups)), groups if labels else [""] * len(groups))
    finish_axes(ax)


def paired_seeds(ax: plt.Axes, values: pd.DataFrame, control_column: str,
                 experimental_column: str, *, xlim: tuple[float, float],
                 color: str) -> None:
    """Compact one-factor control-versus-condition plot with paired seeds."""
    data = values.sort_values("seed")
    y_positions = np.arange(len(data))
    for y, control, experimental in zip(
            y_positions, data[control_column], data[experimental_column]):
        ax.plot([control, experimental], [y, y], color="#BDC5C9",
                lw=0.65, zorder=1)
    ax.scatter(data[control_column], y_positions, s=13, marker="s",
               facecolor="white", edgecolor=GRAY, linewidth=0.85, zorder=3)
    ax.scatter(data[experimental_column], y_positions, s=13, color=color,
               linewidth=0, zorder=4)
    ax.set_ylim(-0.45, len(data) - 0.55)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    finish_axes(ax)


def peak_grid(peaks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    centers = np.arange(150.0, 2000.0, 100.0)
    gx, gy = np.meshgrid(centers, centers)
    squared = (gx[None] - peaks[:, 0, None, None]) ** 2
    squared += (gy[None] - peaks[:, 1, None, None]) ** 2
    grid = np.exp(-squared / (2 * 100.0**2)).sum(axis=0)
    grid[~mask] = 0
    return grid / grid.sum()


def results_figure() -> dict[str, float]:
    online = pd.read_csv(SOURCES["online"])
    behavior = pd.read_csv(SOURCES["behavior"])
    behavior_absolute = pd.read_csv(SOURCES["behavior_absolute"])
    encoder = pd.read_csv(SOURCES["encoder"])
    peaks = pd.read_csv(SOURCES["peaks"])
    decoder = pd.read_csv(SOURCES["decoder"])
    worker = pd.read_csv(SOURCES["worker"])
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
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 1.005),
               ncol=2, frameon=False, fontsize=9, handletextpad=0.2,
               columnspacing=0.5)

    # A: five-seed online course. B: absolute random-control versus own-policy
    # decoder values, with two repeats averaged inside each training seed.
    section(fig, 0.12, 5.46, "A  Spatial coding emerges")
    section(fig, 2.24, 5.46, "B  Policy shapes expression")
    ax = axes_in(fig, 0.58, 4.17, 1.42, 1.01)
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
    ax.set(xlim=(0, 100), xlabel="Steps (millions)", ylabel="DG score")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3,
              handlelength=0.7, columnspacing=0.3, handletextpad=0.2,
              borderpad=0)
    finish_axes(ax)

    behavior_rows = behavior_absolute.rename(columns={"training_seed": "seed"}).copy()
    behavior_rows["group"] = behavior_rows.family
    behavior_groups = ["C01", "C05", "C15"]
    for x0, control, experimental, title, limits, xlabel in [
        (2.92, "random_info", "own_info", "Decoder info", (0, 1.9), "Bits/activation"),
        (5.23, "random_reliability", "own_reliability",
         "Map reliability", (0, 0.9), "Split-half correlation"),
    ]:
        ax = axes_in(fig, x0, 4.20, 1.43, 0.98)
        paired_rows(ax, behavior_rows, behavior_groups, control, experimental,
                    labels=x0 < 4, xlim=limits, marker_size=8)
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_xlabel(xlabel, fontsize=9, labelpad=1)

    fig.add_artist(plt.Line2D([0.02, 0.98],
                              [content_y(3.85) / FIGURE_HEIGHT] * 2,
                              transform=fig.transFigure, color="#CCD1D4", lw=0.6))
    section(fig, 0.12, 3.76, "C–D  Temporal push on shared observations")
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
    row_names = ["C01", "C05-T", "C05-C", "C15"]
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
        assert len(subset) == 18 and subset.group.notna().all()
        assert np.allclose(subset.temporal - subset.constant,
                           subset.difference, atol=1e-10)
        paired_rows(ax, subset, row_names, "constant", "temporal",
                    labels=index == 0, xlim=limits, marker_size=10)
        ax.set_title(title, fontsize=10, pad=2)
        ax.tick_params(axis="x", labelsize=9)
        ax.locator_params(axis="x", nbins=2)

    fig.add_artist(plt.Line2D([0.02, 0.98],
                              [content_y(2.20) / FIGURE_HEIGHT] * 2,
                              transform=fig.transFigure, color="#CCD1D4", lw=0.6))
    section(fig, 0.12, 2.10, "E  Push reaches the readout")
    section(fig, 3.30, 2.10, "F  Temporal-hit coverage")
    downstream = decoder[(decoder.family == "C01") &
                         (decoder.factor == "encoder") &
                         (decoder.layer == "decoder_1") &
                         (decoder.metric == "mean_bits_per_activation_eligible")]
    assert len(downstream) == 5 and downstream.difference.gt(0).all()
    assert np.allclose(downstream.temporal - downstream.constant,
                       downstream.difference, atol=1e-10)
    ax = axes_in(fig, 0.60, 0.73, 1.20, 1.04)
    paired_seeds(ax, downstream, "constant", "temporal",
                 xlim=(0, 2.35), color=COLOR["C01"])
    ax.set_title("Baseline · 5 pairs", fontsize=10, pad=2)
    ax.set_xlabel("Decoder bits/activation", fontsize=9, labelpad=1)
    fig.text(2.04 / 7.5, content_y(1.40) / FIGURE_HEIGHT,
             "5/5 temporal higher\nmean gain +0.68", fontsize=9,
             va="top", color=GRAY)

    pull = worker[(worker.contrast == "temporal_minus_constant_hit_bonus") &
                  (worker.family == "C05") &
                  (worker.metric == "visited_cell_fraction")]
    assert len(pull) == 10
    for x0, fixed, label in [(3.88, "temporal", "Temporal encoder"),
                             (5.78, "constant", "Constant encoder")]:
        subset = pull[pull.fixed_factor == fixed].copy()
        assert len(subset) == 5
        assert np.allclose(subset["first"] - subset["second"],
                           subset.difference, atol=1e-10)
        ax = axes_in(fig, x0, 0.73, 1.10, 1.04)
        paired_seeds(ax, subset, "second", "first",
                     xlim=(0, 0.95), color=COLOR["C05"])
        ax.set_title(label, fontsize=10, pad=2)
        ax.set_xlabel("Visited fraction", fontsize=9, labelpad=1)

    save(fig, "results_sections")
    return {
        "selected_seed": 8,
        "selected_peak_area_temporal": example["arms"]["temporal"]["effective_area_fraction"],
        "selected_peak_area_constant": example["arms"]["constant"]["effective_area_fraction"],
        "behavior_models": len(behavior_absolute),
        "decoder_c01_mean_delta": float(downstream.difference.mean()),
        "result_panel_width_inches": {
            "behavior": 1.43, "dg": 0.86, "decoder": 1.20, "worker": 1.10},
    }


def main() -> None:
    font = style()
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
