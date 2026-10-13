"""Show every high-credit DG peak pair and the spacing-cutoff diagnostic.

Filled circles are the 0.25%-active, field-eligible peaks used by the Gaussian
effective-area measure. An x marks an eligible but lower-activity peak excluded
from area. Coincident peaks are shown by a larger circle containing their count.
The gray density is the same equal-mass Gaussian field used for effective area.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_high_credit_compact import COLOR, HERE, SEEDS, save, style
from plot_high_credit_representation import (
    load_peak_inputs, paired_data, peak_metric, selected_peak_coordinates,
)


ARMS = ("CONSTANT", "TEMPORAL")
METRIC_AREA = "effective_peak_area_fraction"
METRIC_CUTOFF_DISTANCE = "cutoff_median_nearest_peak_distance"
METRIC_ALL_DISTANCE = "median_dominant_peak_nearest_neighbor_distance"


def source_data() -> tuple[dict[tuple[str, int, str], dict], np.ndarray, pd.DataFrame]:
    records, mask = load_peak_inputs()
    indexed = {(row["family"], int(row["seed"]), row["worker_bonus"]): row
               for row in records}
    paired, _ = paired_data()
    paired = paired.set_index(["family", "seed"])
    expected = {(family, seed) for family in ("C05", "C15") for seed in SEEDS}
    if set(paired.index) != expected or len(paired) != len(expected):
        raise ValueError("Expected exactly ten matched high-credit seed pairs")
    if len(indexed) != 20:
        raise ValueError("Expected exactly twenty archived peak snapshots")
    return indexed, mask, paired


def draw_peak_map(ax: plt.Axes, row: dict, mask: np.ndarray,
                  density_limit: float, color: str) -> None:
    qualified, excluded = selected_peak_coordinates(row)
    density = peak_metric.peak_grid(qualified, mask, peak_metric.SIGMA)
    covered = np.ma.masked_where(~mask, density)
    gray = plt.get_cmap("Greys").copy()
    gray.set_bad("#E4E7E9")
    ax.imshow(covered, origin="lower", extent=(100, 2000, 100, 2000),
              cmap=gray, vmin=0, vmax=density_limit, interpolation="nearest")
    positions, multiplicities = np.unique(qualified, axis=0, return_counts=True)
    for (x, y), count in zip(positions, multiplicities):
        ax.scatter(x, y, s=55 + 22 * (count - 1), facecolor=color,
                   edgecolor="white", linewidth=.9, zorder=4)
        if count > 1:
            ax.text(x, y, str(count), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=5)
    if len(excluded):
        ax.scatter(excluded[:, 0], excluded[:, 1], marker="x", s=70,
                   color="#B31942", linewidth=1.8, zorder=6)
    ax.set_xlim(100, 2000)
    ax.set_ylim(100, 2000)
    ax.set_aspect("equal")
    ax.set_xticks((100, 1000, 1900))
    ax.set_yticks((100, 1000, 1900))
    ax.tick_params(labelsize=12, length=2)
    for spine in ax.spines.values():
        spine.set_color("#6F777D")
        spine.set_linewidth(.7)


def plot_family(family: str, snapshots: dict, mask: np.ndarray,
                paired: pd.DataFrame, density_limit: float) -> None:
    color = COLOR[family]
    display = "UG/C05" if family == "C05" else "FG/C15"
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 15.0), facecolor="white")
    fig.subplots_adjust(left=.10, right=.86, bottom=.105, top=.84,
                        hspace=.56, wspace=.23)
    fig.suptitle(f"{display}: 100M-frame DG peak locations", x=.10, y=.985,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(.10, .957,
             "Constant vs temporal worker bonus · shared 318-bin mask and color scale",
             ha="left", va="top", fontsize=12)
    fig.text(.10, .936,
             "A = effective area fraction; NN = median nearest-peak distance after the 0.25% cutoff",
             ha="left", va="top", fontsize=12)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=color, markeredgecolor="white",
               label="Peak used for coverage and cutoff-matched spacing"),
        Line2D([], [], marker="x", linestyle="none", markersize=8,
               color="#B31942", label="Eligible peak below 0.25% activity"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.10, .92),
               frameon=False, fontsize=12, ncol=1)
    fig.text(.29, .867, "Constant worker", ha="center", fontsize=12,
             fontweight="bold")
    fig.text(.67, .867, "Temporal worker", ha="center", fontsize=12,
             fontweight="bold")
    for row_index, seed in enumerate(SEEDS):
        pair = paired.loc[(family, seed)]
        for column, arm in enumerate(ARMS):
            ax = axes[row_index, column]
            record = snapshots[(family, seed, arm)]
            draw_peak_map(ax, record, mask, density_limit, color)
            prefix = arm.lower()
            area = float(pair[f"{prefix}_{METRIC_AREA}"])
            nearest = float(pair[f"{prefix}_{METRIC_CUTOFF_DISTANCE}"])
            ax.set_title(f"s{seed} · A {area:.3f} · NN {nearest:.0f}",
                         fontsize=12, pad=5)
            if column == 0:
                ax.set_ylabel("y (DMLab units)", fontsize=12)
            if row_index == 4:
                ax.set_xlabel("x (DMLab units)", fontsize=12)
    scale = plt.cm.ScalarMappable(norm=plt.Normalize(0, density_limit),
                                  cmap="Greys")
    bar = fig.colorbar(scale, ax=axes, location="right", fraction=.018,
                       pad=.025, shrink=.78)
    bar.set_label("Gaussian peak density (mass/bin)", fontsize=12)
    bar.ax.tick_params(labelsize=12)
    fig.text(.10, .022,
             "Circle numbers mark coincident peaks; crosses mark eligible units excluded by the coverage cutoff.\n"
             "Unit identities are not matched across independently trained networks.",
             fontsize=12, va="bottom")
    save(fig, f"high_credit_peak_locations_{family.lower()}", HERE)


def plot_difference_diagnostic(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7), sharex=True, sharey=True,
                             facecolor="white")
    fig.subplots_adjust(left=.11, right=.98, bottom=.20, top=.76, wspace=.19)
    fig.suptitle("Why peak area and nearest-peak distance can disagree",
                 x=.11, y=.98, ha="left", fontsize=15, fontweight="bold")
    fig.text(.11, .91, "Temporal − constant worker bonus · each label is a training seed",
             fontsize=12, va="top")
    for ax, metric, title in zip(
        axes,
        (METRIC_ALL_DISTANCE, METRIC_CUTOFF_DISTANCE),
        ("All field-eligible peaks (original)", "Same 0.25% cutoff as area"),
    ):
        for family in ("C05", "C15"):
            family_rows = paired.loc[family]
            x = family_rows[f"delta_{METRIC_AREA}"].to_numpy(float)
            y = family_rows[f"delta_{metric}"].to_numpy(float)
            ax.scatter(x, y, s=58, color=COLOR[family],
                       label="UG/C05" if family == "C05" else "FG/C15", zorder=4)
            for seed, xx, yy in zip(SEEDS, x, y):
                offset = ((4, -14) if family == "C05" and seed == 123 else
                          (4, 6) if family == "C05" and seed == 41 else
                          (4, 5))
                ax.annotate(str(seed), (xx, yy), xytext=offset,
                            textcoords="offset points", fontsize=10,
                            color=COLOR[family])
        ax.axhline(0, color="#8A9297", linewidth=.8)
        ax.axvline(0, color="#8A9297", linewidth=.8)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xlabel("Δ effective peak-area fraction", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Δ median nearest-peak distance (DMLab units)", fontsize=12)
    axes[0].legend(loc="lower right", frameon=False, fontsize=11)
    axes[0].set_xlim(-.11, .22)
    axes[0].set_ylim(-160, 160)
    fig.text(.11, .05,
             "C05 seed 99: area +0.019; spacing +115 with all eligible peaks, but 0 with the area cutoff.",
             fontsize=12, va="bottom")
    save(fig, "high_credit_peak_area_vs_spacing", HERE)


def plot_examples(snapshots: dict, mask: np.ndarray, paired: pd.DataFrame,
                  density_limit: float) -> None:
    """Place the two clearest mechanisms at a useful side-by-side viewing size."""
    cases = (("C05", 99), ("C15", 123))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.3), facecolor="white")
    fig.subplots_adjust(left=.11, right=.96, bottom=.18, top=.80,
                        hspace=.43, wspace=.23)
    fig.suptitle("The same peaks, two different measures of spread", x=.11, y=.98,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.11, .92, "Circles: peaks used for area and cutoff-matched spacing · red ×: eligible peak below 0.25% activity",
             fontsize=11, va="top")
    fig.text(.32, .85, "Constant worker", ha="center", fontsize=12,
             fontweight="bold")
    fig.text(.75, .85, "Temporal worker", ha="center", fontsize=12,
             fontweight="bold")
    for row_index, (family, seed) in enumerate(cases):
        pair = paired.loc[(family, seed)]
        for column, arm in enumerate(ARMS):
            ax = axes[row_index, column]
            draw_peak_map(ax, snapshots[(family, seed, arm)], mask,
                          density_limit, COLOR[family])
            prefix = arm.lower()
            area = float(pair[f"{prefix}_{METRIC_AREA}"])
            nearest = float(pair[f"{prefix}_{METRIC_CUTOFF_DISTANCE}"])
            ax.set_title(f"{family} s{seed} · area {area:.3f} · NN {nearest:.0f}",
                         fontsize=12, pad=5)
            if column == 0:
                ax.set_ylabel("y (DMLab units)", fontsize=12)
            if row_index == 1:
                ax.set_xlabel("x (DMLab units)", fontsize=12)
    fig.text(.11, .025,
             "C05 s99: the +115-unit original NN difference vanishes with the area cutoff.\n"
             "C15 s123: all 16 peaks pass; local NN rises by 100 units while global area falls by 0.028.",
             fontsize=11, va="bottom")
    save(fig, "high_credit_peak_examples", HERE)


def main() -> None:
    style()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    snapshots, mask, paired = source_data()
    density_limit = max(float(peak_metric.peak_grid(
        selected_peak_coordinates(record)[0], mask, peak_metric.SIGMA).max())
        for record in snapshots.values())
    for family in ("C05", "C15"):
        plot_family(family, snapshots, mask, paired, density_limit)
    plot_examples(snapshots, mask, paired, density_limit)
    plot_difference_diagnostic(paired)


if __name__ == "__main__":
    main()
