"""Compact high-credit contrasts with a family-specific graph-edge axis.

C is the paired change in commanded target hits per 1,000 eligible logged
events. D is the original paired change in reliable graph-edge count. The
default preserves the earlier dual-axis C/D candidate; --shared-hit-axis
places both C families on one scale while retaining D's two zero-aligned axes.
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_high_credit_compact import (
    COLOR,
    HERE,
    SEEDS,
    SOURCE_COMMIT,
    draw_metric,
    finish_axes,
    high_credit_data,
    save,
    source_table,
    style,
)


OUTPUT_STEM = "high_credit_worker_dual_axis_hit_rate_compact"
SHARED_HIT_OUTPUT_STEM = "high_credit_worker_shared_hit_rate_dual_edges_compact"
HIT_RATE_TABLE = HERE / "high_credit_hit_rate_pairs.csv"
PANEL_POSITIONS = ((0.72, 1.12), (2.34, 1.12), (4.02, 1.15), (5.95, 1.15))


def paired_hit_rates() -> pd.DataFrame:
    """Compute ratios of summed numerator and event count over the 90–100M window."""
    rows: list[dict[str, float | int | str]] = []
    for family in ("C05", "C15"):
        source = source_table(f"{family.lower()}_online_per_run.csv")
        expected = {(seed, arm) for seed in SEEDS for arm in ("TEMPORAL", "CONSTANT")}
        if len(source) != len(expected) or set(zip(source.seed, source.worker_bonus)) != expected:
            raise ValueError(f"Incomplete online source for {family}")
        count_columns = ("target_hit_numerator__n", "target_hit_event_count__n")
        if not (source[count_columns[0]] == source[count_columns[1]]).all():
            raise ValueError(f"Unequal numerator/denominator sample counts for {family}")
        if not (source[count_columns[0]] > 0).all():
            raise ValueError(f"Empty online hit-rate window for {family}")
        if not ((source.target_hit_event_count > 0) &
                (source.target_hit_numerator >= 0) &
                (source.target_hit_numerator <= source.target_hit_event_count)).all():
            raise ValueError(f"Invalid hit-rate components for {family}")
        source = source.copy()
        source["hits_per_1000_events"] = (
            1000 * source.target_hit_numerator / source.target_hit_event_count
        )
        by_pair = source.set_index(["seed", "worker_bonus"])
        for seed in SEEDS:
            temporal = by_pair.loc[(seed, "TEMPORAL")]
            constant = by_pair.loc[(seed, "CONSTANT")]
            if (temporal.window_low != constant.window_low or
                    temporal.window_high != constant.window_high):
                raise ValueError(f"Mismatched hit-rate windows for {family} seed {seed}")
            rows.append({
                "family": family,
                "seed": seed,
                "temporal_hit_numerator_mean": temporal.target_hit_numerator,
                "temporal_eligible_event_count_mean": temporal.target_hit_event_count,
                "constant_hit_numerator_mean": constant.target_hit_numerator,
                "constant_eligible_event_count_mean": constant.target_hit_event_count,
                "logged_updates_per_arm": int(temporal.target_hit_numerator__n),
                "temporal_hits_per_1000_events": temporal.hits_per_1000_events,
                "constant_hits_per_1000_events": constant.hits_per_1000_events,
                "delta_hits_per_1000_events": (temporal.hits_per_1000_events -
                                               constant.hits_per_1000_events),
                "source_commit": SOURCE_COMMIT,
            })
    result = pd.DataFrame(rows)
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Non-finite paired hit rate")
    result.to_csv(HIT_RATE_TABLE, index=False)
    return result


def nice_limit(values: np.ndarray) -> float:
    """Round a zero-centered family limit up to a short printed tick value."""
    target = float(np.max(np.abs(values))) * 1.10
    scale = 10 ** math.floor(math.log10(target))
    for multiple in (1, 1.5, 2, 3, 5, 6, 10):
        limit = multiple * scale
        if limit >= target:
            return limit
    raise AssertionError("Unreachable nice-limit case")


def draw_family_box(ax: plt.Axes, values: np.ndarray, family: str,
                    position: int) -> None:
    color = COLOR[family]
    box = ax.boxplot([values], positions=[position], widths=0.44,
                     whis=(0, 100), showfliers=False, patch_artist=True,
                     manage_ticks=False)
    box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.18,
                         linewidth=0.8)
    for artist in box["whiskers"] + box["caps"]:
        artist.set(color=color, linewidth=0.75)
    box["medians"][0].set(color=color, linewidth=1.1)
    ax.scatter(position + np.linspace(-0.12, 0.12, len(SEEDS)), values,
               s=18, color=color, edgecolor="white", linewidth=0.3, zorder=4)
    ax.scatter(position + 0.18, float(np.mean(values)), marker="D", s=28,
               facecolor="white", edgecolor="#20272C", linewidth=1.2,
               zorder=5)


def draw_dual_panel(fig: plt.Figure, data: pd.DataFrame, slot: int,
                    column: str, title: str) -> None:
    left, width = PANEL_POSITIONS[slot]
    ax_c05 = fig.add_axes([left / 7.5, 0.28 / 2.50,
                           width / 7.5, 1.53 / 2.50])
    ax_c15 = ax_c05.twinx()
    ax_c15.patch.set_visible(False)
    for axis, family, position in ((ax_c05, "C05", 0), (ax_c15, "C15", 1)):
        values = (data[data.family == family].set_index("seed")
                  .loc[list(SEEDS), column].to_numpy(dtype=float))
        draw_family_box(axis, values, family, position)
        limit = nice_limit(values)
        axis.set_ylim(-limit, limit)  # Both zero lines align physically.
        axis.set_xlim(-0.42, 1.58)
        axis.set_yticks((-limit, 0, limit))
        axis.tick_params(axis="y", colors=COLOR[family], length=2.2,
                         width=0.6, pad=1.5)
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_color("#20272C")
        axis.spines["left" if family == "C05" else "right"].set_color(COLOR[family])
    ax_c05.spines["right"].set_visible(False)
    ax_c15.spines["left"].set_visible(False)
    ax_c05.axhline(0, color="#7D858A", lw=0.7, zorder=1)
    ax_c05.grid(axis="y", color="#E6E8E9", lw=0.5, zorder=0)
    ax_c05.set_xticks((0, 1), ("C05", "C15"))
    for label, family in zip(ax_c05.get_xticklabels(), ("C05", "C15")):
        label.set_color(COLOR[family])
    ax_c05.tick_params(axis="x", length=2.2, width=0.6, pad=1.5)
    ax_c15.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_c05.set_title(title, loc="left", fontweight="bold", pad=3)


def draw_shared_hit_panel(fig: plt.Figure, data: pd.DataFrame) -> None:
    """Draw both hit-rate contrasts against one genuinely comparable scale."""
    left, width = PANEL_POSITIONS[2]
    ax = fig.add_axes([left / 7.5, 0.28 / 2.50,
                       width / 7.5, 1.53 / 2.50])
    all_values = []
    for position, family in enumerate(("C05", "C15")):
        values = (data[data.family == family].set_index("seed")
                  .loc[list(SEEDS), "delta_hits_per_1000_events"].to_numpy(dtype=float))
        draw_family_box(ax, values, family, position)
        all_values.extend(values)
    limit = 5 * math.ceil(1.08 * max(abs(value) for value in all_values) / 5)
    ax.set_ylim(-limit, limit)
    ax.set_xlim(-0.42, 1.58)
    ax.set_yticks((-limit, 0, limit))
    ax.axhline(0, color="#7D858A", lw=0.7, zorder=1)
    ax.set_xticks((0, 1), ("C05", "C15"))
    for label, family in zip(ax.get_xticklabels(), ("C05", "C15")):
        label.set_color(COLOR[family])
    ax.set_title("C  Hit rate", loc="left", fontweight="bold", pad=3)
    ax.set_ylabel(r"$\Delta$ hits / 1k", labelpad=1)
    finish_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-hit-axis", action="store_true",
                        help="Give C05 and C15 one hit-rate axis; keep D's dual axes")
    args = parser.parse_args()
    data = high_credit_data()
    hit_rates = paired_hit_rates()
    data = data.merge(hit_rates[["family", "seed", "delta_hits_per_1000_events"]],
                      on=["family", "seed"], validate="one_to_one")
    style()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 2.50), facecolor="white")
    fig.text(0.03, 0.96,
             "High policy credit: temporal − constant worker bonus  "
             r"($\gamma=0.999$, $\lambda=0.995$)",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    subtitle = (
        "C: shared Δ hits / 1k eligible events.  "
        "D: separate Δ-edge scales (C05 left, C15 right)."
        if args.shared_hit_axis else
        "C: Δ hits / 1k eligible events; D: Δ edges.  "
        "C05 left (orange), C15 right (green)."
    )
    fig.text(0.03, 0.88, subtitle,
             va="top", fontsize=12, color="#20272C")
    for slot, (metric, title, ylabel) in enumerate((
        ("delta_spatial_information_bits", "A  DG information", r"$\Delta$ bits"),
        ("delta_coverage_auc", "B  Coverage AUC", r"$\Delta$ AUC"),
    )):
        left, width = PANEL_POSITIONS[slot]
        ax = fig.add_axes([left / 7.5, 0.28 / 2.50,
                           width / 7.5, 1.53 / 2.50])
        draw_metric(ax, data, metric, title, ylabel)
    if args.shared_hit_axis:
        draw_shared_hit_panel(fig, data)
    else:
        draw_dual_panel(fig, data, 2, "delta_hits_per_1000_events", "C  Hit rate")
    draw_dual_panel(fig, data, 3, "delta_reliable_edges", "D  Graph edges")
    save(fig, SHARED_HIT_OUTPUT_STEM if args.shared_hit_axis else OUTPUT_STEM, HERE)


if __name__ == "__main__":
    main()
