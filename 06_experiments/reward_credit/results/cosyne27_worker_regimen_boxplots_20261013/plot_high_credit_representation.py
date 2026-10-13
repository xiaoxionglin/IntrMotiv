"""Plot additional high-credit DG representation contrasts at 100M frames.

Each point is one matched training seed's temporal-minus-constant worker-bonus
difference. Both arms use temporal encoder credit and their own-policy 100k-
decision spatial snapshots. Read the source tables from the pinned Git commit.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_high_credit_compact import (
    HERE, SEEDS, SOURCE_COMMIT, finish_axes, high_credit_data,
    save, source_table, style,
)
from plot_high_credit_dual_axes import draw_family_box


METRICS = (
    ("active_only_map_cosine", "A  Map overlap", r"$\Delta$ cosine"),
    ("unique_active_peak_bins", "B  Distinct peaks", r"$\Delta$ bins"),
    ("mono_field_unit_fraction", "C  Single fields", r"$\Delta$ fraction"),
    ("median_dominant_peak_nearest_neighbor_distance", "D  Peak spacing",
     r"$\Delta$ DMLab units"),
)
TABLE = HERE / "high_credit_representation_pairs.csv"
OUTPUT_STEM = "high_credit_worker_representation_compact"


def paired_data() -> pd.DataFrame:
    """Retain both arms and seed differences for all snapshot diagnostics."""
    rows = []
    expected = {(seed, arm) for seed in SEEDS for arm in ("TEMPORAL", "CONSTANT")}
    source_columns = (
        "active_unit_mean_spatial_information", "active_only_map_cosine",
        "unique_active_peak_bins", "mono_field_unit_fraction",
        "median_dominant_peak_nearest_neighbor_distance", "visited_cell_fraction",
    )
    existing = high_credit_data().set_index(["family", "seed"])
    for family in ("C05", "C15"):
        source = source_table(f"{family.lower()}_spatial_per_snapshot.csv")
        source = source[source.target_env_steps == 100_000_000]
        if len(source) != len(expected) or set(zip(source.seed, source.worker_bonus)) != expected:
            raise ValueError(f"Incomplete 100M snapshots for {family}")
        indexed = source.set_index(["seed", "worker_bonus"])
        for seed in SEEDS:
            temporal = indexed.loc[(seed, "TEMPORAL")]
            constant = indexed.loc[(seed, "CONSTANT")]
            row = {"family": family, "seed": seed, "source_commit": SOURCE_COMMIT,
                   "snapshot_target_frames": 100_000_000,
                   "snapshot_decisions": int(temporal.window_limit)}
            for metric in source_columns:
                t, c = float(temporal[metric]), float(constant[metric])
                if not np.isfinite([t, c]).all():
                    raise ValueError(f"Non-finite {metric} for {family} seed {seed}")
                row[f"temporal_{metric}"] = t
                row[f"constant_{metric}"] = c
                row[f"delta_{metric}"] = t - c
            expected_si = existing.loc[(family, seed), "delta_spatial_information_bits"]
            if not np.isclose(row["delta_active_unit_mean_spatial_information"],
                              expected_si, atol=1e-6):
                raise ValueError(f"Spatial-information source mismatch for {family} seed {seed}")
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TABLE, index=False)
    return result


def main() -> None:
    data = paired_data()
    style()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 2.50), facecolor="white")
    fig.text(0.03, 0.96, "High policy credit: temporal − constant worker bonus",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    fig.text(0.03, 0.88,
             "DG maps from own-policy 100k-decision snapshots at 100M frames",
             va="top", fontsize=12, color="#20272C")
    for index, (metric, title, ylabel) in enumerate(METRICS):
        left = 0.66 + 1.79 * index
        ax = fig.add_axes([left / 7.5, 0.28 / 2.5, 1.20 / 7.5, 1.53 / 2.5])
        all_values = []
        for position, family in enumerate(("C05", "C15")):
            values = (data[data.family == family].set_index("seed")
                      .loc[list(SEEDS), f"delta_{metric}"].to_numpy(dtype=float))
            draw_family_box(ax, values, family, position)
            all_values.extend(values)
        extent = max(abs(value) for value in all_values) * 1.20
        ax.set_ylim(-extent, extent)
        ax.set_xlim(-0.42, 1.58)
        ax.axhline(0, color="#7D858A", lw=0.7, zorder=1)
        ax.set_xticks((0, 1), ("C05", "C15"))
        ax.set_title(title, loc="left", fontweight="bold", pad=3)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
        finish_axes(ax)
    save(fig, OUTPUT_STEM, HERE)


if __name__ == "__main__":
    main()
