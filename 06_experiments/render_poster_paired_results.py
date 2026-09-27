"""Render seed-paired poster summaries from manifest-derived evaluation metrics.

Each dot is one independently trained seed, joined across the two arms.
No inferential significance is implied for the three-seed comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#D55E00"


def paired_panel(ax, table: pd.DataFrame, first: str, second: str, metric: str,
                 labels: tuple[str, str], ylabel: str, title: str, limit: tuple[float, float]):
    left = table[table.condition == first].set_index("seed")
    right = table[table.condition == second].set_index("seed")
    seeds = sorted(left.index.intersection(right.index))
    for seed in seeds:
        a, b = left.loc[seed, metric], right.loc[seed, metric]
        ax.plot([0, 1], [a, b], color="#777777", linewidth=1.4, alpha=.8, zorder=1)
        ax.scatter(0, a, s=95, color=BLUE, edgecolor="white", linewidth=.8, zorder=2)
        ax.scatter(1, b, s=95, color=ORANGE, edgecolor="white", linewidth=.8, zorder=2)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-.3, 1.35)
    ax.set_ylim(*limit)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.metrics)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    paired_panel(axes[0], data, "SAT_C15_ARR_MON_FILM", "SAT_C15_SRC_MON_FILM",
                 "dg_map_cosine_active", ("ARR", "SRC"), "Active DG map cosine",
                 "a  Saturday C15 · 75M", (0, .35))
    paired_panel(axes[1], data, "CPU2048_DIRECT_F16_DDQN_HER",
                 "CPU2048_WAYPOINT_DECODER_F64_DDQN_HER", "dg_distinct_peak_bins",
                 ("Direct F16", "Waypoint F64"), "Distinct DG peak cells",
                 "b  CPU2048 HER · 300M", (0, 64))
    paired_panel(axes[2], data, "DGP_C15_HIT_JOINT_LEG", "DGP_C15_FIRST_JOINT_LEG",
                 "stored_graph_confidence_fraction", ("HIT", "FIRST"),
                 "Stored edge confidence / attempts",
                 "c  DGP C15 · 75M", (0, 1))
    fig.suptitle("Frozen DG maps and stored checkpoint graphs · three paired seeds", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
