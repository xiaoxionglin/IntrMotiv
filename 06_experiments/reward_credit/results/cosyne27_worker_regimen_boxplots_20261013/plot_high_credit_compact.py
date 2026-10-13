"""Compact Figure-2-style box plots for the high-policy-credit worker study.

Read the canonical paired seed table in this directory. Only the high-credit
rows are plotted; each difference is temporal minus constant worker bonus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "07_abstracts/cosyne_2027_submission_draft"))
from render_submission_panels import COLOR, finish_axes, save, style  # noqa: E402


SEEDS = (8, 17, 41, 99, 123)
METRICS = (
    ("delta_spatial_information_bits", "A  DG information", r"$\Delta$ bits"),
    ("delta_coverage_auc", "B  Coverage AUC", r"$\Delta$ AUC"),
    ("delta_correct_hit_observations", "C  Correct hits", r"$\Delta$ observations"),
    ("delta_reliable_edges", "D  Graph edges", r"$\Delta$ edges"),
)
OUTPUT_STEM = "high_credit_worker_compact"


def high_credit_data() -> pd.DataFrame:
    data = pd.read_csv(HERE / "paired_seed_differences.csv")
    data = data[data.policy_credit == "high"].copy()
    expected = {(family, seed) for family in ("C05", "C15") for seed in SEEDS}
    actual = set(zip(data.family, data.seed))
    if len(data) != len(expected) or actual != expected:
        raise ValueError("Expected five distinct high-credit pairs per architecture")
    if not np.isfinite(data[[metric for metric, _, _ in METRICS]].to_numpy()).all():
        raise ValueError("Non-finite high-credit paired difference")
    return data


def draw_metric(ax: plt.Axes, data: pd.DataFrame, metric: str,
                title: str, ylabel: str) -> None:
    all_values = []
    for position, family in enumerate(("C05", "C15")):
        values = (data[data.family == family].set_index("seed")
                  .loc[list(SEEDS), metric].to_numpy(dtype=float))
        all_values.extend(values)
        color = COLOR[family]
        box = ax.boxplot([values], positions=[position], widths=0.48,
                         whis=(0, 100), showfliers=False, patch_artist=True,
                         manage_ticks=False)
        box["boxes"][0].set(facecolor=color, edgecolor=color,
                             alpha=0.18, linewidth=0.8)
        for item in box["whiskers"] + box["caps"]:
            item.set(color=color, linewidth=0.75)
        box["medians"][0].set(color=color, linewidth=1.1)
        ax.scatter(position + np.linspace(-0.12, 0.12, len(SEEDS)), values,
                   s=18, color=color, edgecolor="white", linewidth=0.3,
                   zorder=4)
    extent = max(abs(min(all_values)), abs(max(all_values))) * 1.18
    ax.set_ylim(-extent, extent)
    ax.set_xlim(-0.42, 1.58)
    ax.axhline(0, color="#7D858A", lw=0.7, zorder=1)
    ax.set_xticks((0, 1), ("C05", "C15"))
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    finish_axes(ax)


def main() -> None:
    data = high_credit_data()
    style()  # Reuse the submitted Figure 2 font, architecture colors, and axes.
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 2.50), facecolor="white")
    fig.text(0.03, 0.96,
             "High policy credit: temporal − constant worker bonus  "
             r"($\gamma=0.999$, $\lambda=0.995$)",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    for position, (metric, title, ylabel) in enumerate(METRICS):
        ax = fig.add_axes([(0.75 + position * 1.80) / 7.5,
                           0.28 / 2.50, 1.20 / 7.5, 1.53 / 2.50])
        draw_metric(ax, data, metric, title, ylabel)
    save(fig, OUTPUT_STEM, HERE)


if __name__ == "__main__":
    main()
