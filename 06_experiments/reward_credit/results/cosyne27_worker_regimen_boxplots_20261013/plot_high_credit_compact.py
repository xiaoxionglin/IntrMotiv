"""Compact Figure-2-style box plots for the high-policy-credit worker study.

Read the canonical paired seed table in this directory. Only the high-credit
rows are plotted; each difference is temporal minus constant worker bonus.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator, FuncFormatter
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
RELATIVE_OUTPUT_STEM = "high_credit_worker_cplus1_compact"
SOURCE_COMMIT = "a26852f"
SOURCE_DIRECTORY = "06_experiments/reward_credit/results/cosyne27_worker_g999_l995_20261012"
RELATIVE_METRICS = (
    METRICS[0], METRICS[1],
    ("relative_correct_hit_observations", "C  Correct hits", "Relative change (%)"),
    ("relative_reliable_edges", "D  Graph edges", "Relative change (%)"),
)


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


def source_table(name: str) -> pd.DataFrame:
    """Read the pinned high-credit source without copying a second data bundle."""
    path = f"{SOURCE_DIRECTORY}/{name}"
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT,
        check=True, capture_output=True,
    )
    return pd.read_csv(io.BytesIO(result.stdout))


def relative_count_data(data: pd.DataFrame) -> pd.DataFrame:
    """Add seed-paired (temporal - constant)/(constant + one count)."""
    rows = []
    for family in ("C05", "C15"):
        slug = family.lower()
        sources = (
            ("correct_hit_observations", "count_observations",
             source_table(f"{slug}_worker_hit_weighted_90_100m.csv")),
            ("reliable_edges", "graph_reliable_edge_count",
             source_table(f"{slug}_spatial_per_snapshot.csv")),
        )
        for measure, source_column, source in sources:
            if measure == "reliable_edges":
                source = source[source.target_env_steps == 100_000_000]
            if len(source) != 2 * len(SEEDS):
                raise ValueError(f"Expected ten source rows for {family} {measure}")
            pairs = source.pivot(index="seed", columns="worker_bonus",
                                 values=source_column)
            if set(pairs.index) != set(SEEDS) or set(pairs.columns) != {"TEMPORAL", "CONSTANT"}:
                raise ValueError(f"Incomplete pairs for {family} {measure}")
            for seed in SEEDS:
                constant = float(pairs.loc[seed, "CONSTANT"])
                temporal = float(pairs.loc[seed, "TEMPORAL"])
                if not np.isfinite([constant, temporal]).all() or constant < 0:
                    raise ValueError(f"Invalid count for {family} seed {seed} {measure}")
                expected_delta = data.loc[(data.family == family) &
                                          (data.seed == seed), f"delta_{measure}"].item()
                if not np.isclose(temporal - constant, expected_delta, atol=1e-6):
                    raise ValueError(f"Source contrast mismatch for {family} seed {seed} {measure}")
                rows.append({"family": family, "seed": seed, "measure": measure,
                             "constant": constant, "temporal": temporal,
                             "relative_change_cplus1": (temporal - constant) / (constant + 1)})
    relative = pd.DataFrame(rows)
    relative.to_csv(HERE / "high_credit_cplus1_seed_values.csv", index=False)
    for measure in ("correct_hit_observations", "reliable_edges"):
        column = f"relative_{measure}"
        values = relative[relative.measure == measure][
            ["family", "seed", "relative_change_cplus1"]].rename(
                columns={"relative_change_cplus1": column})
        data = data.merge(values, on=["family", "seed"], validate="one_to_one")
    return data


def draw_values(ax: plt.Axes, data: pd.DataFrame, metric: str,
                scale: float = 1.0) -> list[float]:
    all_values = []
    for position, family in enumerate(("C05", "C15")):
        values = (data[data.family == family].set_index("seed")
                  .loc[list(SEEDS), metric].to_numpy(dtype=float) * scale)
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
    ax.set_xlim(-0.42, 1.58)
    ax.axhline(0, color="#7D858A", lw=0.7, zorder=1)
    finish_axes(ax)
    return all_values


def draw_metric(ax: plt.Axes, data: pd.DataFrame, metric: str,
                title: str, ylabel: str) -> None:
    all_values = draw_values(ax, data, metric)
    extent = max(abs(min(all_values)), abs(max(all_values))) * 1.18
    ax.set_ylim(-extent, extent)
    ax.set_xticks((0, 1), ("C05", "C15"))
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))


def draw_broken_relative_metric(fig: plt.Figure, data: pd.DataFrame,
                                position: int, metric: str,
                                title: str, ylabel: str) -> None:
    """Keep the central ±100% range legible while retaining all larger dots."""
    left = 0.75 + position * 1.80
    width = 1.20
    lower = fig.add_axes([left / 7.5, 0.28 / 2.50,
                          width / 7.5, 0.88 / 2.50])
    upper = fig.add_axes([left / 7.5, 1.28 / 2.50,
                          width / 7.5, 0.53 / 2.50], sharex=lower)
    draw_values(lower, data, metric, scale=100.0)
    draw_values(upper, data, metric, scale=100.0)
    upper_limit = 620 if metric == "relative_correct_hit_observations" else 330
    upper_ticks = (200, 400, 600) if upper_limit == 620 else (200, 300)
    lower.set_ylim(-105, 105)
    upper.set_ylim(110, upper_limit)
    lower.yaxis.set_major_locator(FixedLocator((-100, 0, 100)))
    upper.yaxis.set_major_locator(FixedLocator(upper_ticks))
    for ax in (lower, upper):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    lower.set_xticks((0, 1), ("C05", "C15"))
    upper.tick_params(axis="x", bottom=False, labelbottom=False)
    lower.spines["top"].set_visible(False)
    upper.spines["bottom"].set_visible(False)
    upper.set_title(title, loc="left", fontweight="bold", pad=3)
    fig.text((left - 0.49) / 7.5, 1.05 / 2.50, ylabel,
             rotation=90, va="center", ha="center", fontsize=12)
    # Diagonal marks identify the omitted 105–110% interval and scale jump.
    for ax, y in ((lower, 1), (upper, 0)):
        for x in (0, 1):
            ax.plot((x - 0.025, x + 0.025), (y - 0.035, y + 0.035),
                    transform=ax.transAxes, color="#20272C", lw=0.8,
                    clip_on=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-counts", action="store_true",
                        help="Use (T-C)/(C+1) for panels C and D; retain A and B")
    args = parser.parse_args()
    data = high_credit_data()
    if args.relative_counts:
        data = relative_count_data(data)
    metrics = RELATIVE_METRICS if args.relative_counts else METRICS
    style()  # Reuse the submitted Figure 2 font, architecture colors, and axes.
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 2.50), facecolor="white")
    heading = ("High policy credit: C–D relative to constant arm  "
               if args.relative_counts else
               "High policy credit: temporal − constant worker bonus  ")
    fig.text(0.03, 0.96, heading + r"($\gamma=0.999$, $\lambda=0.995$)",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    if args.relative_counts:
        fig.text(0.03, 0.88, r"C–D: $100(T-C)/(C+\epsilon)$, $\epsilon=1$ count",
                 va="top", fontsize=12, color="#20272C")
    for position, (metric, title, ylabel) in enumerate(metrics):
        if args.relative_counts and position >= 2:
            draw_broken_relative_metric(fig, data, position, metric, title, ylabel)
        else:
            ax = fig.add_axes([(0.75 + position * 1.80) / 7.5,
                               0.28 / 2.50, 1.20 / 7.5, 1.53 / 2.50])
            draw_metric(ax, data, metric, title, ylabel)
    save(fig, RELATIVE_OUTPUT_STEM if args.relative_counts else OUTPUT_STEM, HERE)


if __name__ == "__main__":
    main()
