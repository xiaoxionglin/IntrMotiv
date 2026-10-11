"""Summarize shared-observation peak geometry with training seed as the unit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_factors import factors


METRICS = (
    "active_units",
    "unique_peak_bins",
    "coverage_entropy_normalized",
    "coverage_effective_area_fraction",
    "coverage_gini",
    "mean_pairwise_peak_distance_bins",
    "mean_nearest_neighbor_peak_distance_bins",
    "active_only_map_cosine",
    "mean_information_amplitude_weighted",
    "mean_information_per_activation_bits",
    "field_eligible_units",
    "mono_field_fraction_of_eligible",
)


def paired(rows: pd.DataFrame, factor: str, left: str, right: str) -> pd.DataFrame:
    keys = ["family", "seed", "activity_cutoff_fraction", "gaussian_sigma_bins"]
    keys.append("worker_bonus" if factor == "encoder_credit" else "encoder_credit")
    left_rows = rows.loc[rows[factor] == left, keys + list(METRICS)]
    right_rows = rows.loc[rows[factor] == right, keys + list(METRICS)]
    joined = left_rows.merge(right_rows, on=keys, validate="one_to_one", suffixes=(f"_{left}", f"_{right}"))
    for metric in METRICS:
        joined[f"{metric}_delta"] = joined[f"{metric}_{right}"] - joined[f"{metric}_{left}"]
    joined.insert(0, "contrast", f"{right}_minus_{left}")
    return joined


def plot_primary(rows: pd.DataFrame, output: Path) -> None:
    chosen = rows.loc[
        np.isclose(rows.activity_cutoff_fraction, 0.0025)
        & np.isclose(rows.gaussian_sigma_bins, 1.0)
    ]
    groups = [("C01", "inapplicable", "C01"),
              ("C05", "constant", "C05\nW constant"),
              ("C05", "temporal", "C05\nW temporal"),
              ("C15", "temporal", "C15")]
    panels = [("active_units", "Valid DG units", (0, 17)),
              ("coverage_effective_area_fraction", "Peak coverage effective area", (.2, .55)),
              ("active_only_map_cosine", "Active-only map cosine", (0, .2)),
              ("mean_information_per_activation_bits", "Information per activation (bits)", (1.8, 4.4))]
    colors = {"constant": "#2C6593", "temporal": "#B34C34"}
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=10, label=f"{name.title()} encoder")
               for name, color in colors.items()]
    for suffix, selected_panels in (("geometry", panels[:2]), ("functional", panels[2:])):
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 11), constrained_layout=True)
        for ax, (metric, ylabel, limits) in zip(axes, selected_panels):
            for index, (family, worker, label) in enumerate(groups):
                group = chosen.loc[(chosen.family == family) & (chosen.worker_bonus == worker)]
                common_seeds = set(group.loc[group.encoder_credit == "constant", "seed"]) & set(group.loc[group.encoder_credit == "temporal", "seed"])
                for seed in sorted(common_seeds):
                    pair = group.loc[group.seed == seed].set_index("encoder_credit")
                    ax.plot([index - .14, index + .14], [pair.loc["constant", metric], pair.loc["temporal", metric]],
                            color="#B8B8B8", linewidth=1.3, zorder=1)
                for encoder, offset in (("constant", -.14), ("temporal", .14)):
                    values = group.loc[group.encoder_credit == encoder, metric].dropna().to_numpy()
                    ax.scatter(np.full(len(values), index + offset), values, s=43, color=colors[encoder], alpha=.55, zorder=2)
                    if len(values):
                        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                        ax.errorbar(index + offset, values.mean(), yerr=sem, fmt="o", markersize=11,
                                    capsize=5, capthick=2, linewidth=2, color=colors[encoder], zorder=3)
                if metric == panels[0][0]:
                    ax.text(index, 16.3, f"paired n={len(common_seeds)}", ha="center", fontsize=17)
            ax.set_xticks(range(len(groups)), [label for _, _, label in groups])
            ax.tick_params(labelsize=19)
            ax.set_ylabel(ylabel, fontsize=21)
            ax.set_xlim(-.5, 3.5)
            ax.set_ylim(*limits)
            ax.grid(axis="y", alpha=.18)
        axes[0].legend(handles=handles, loc="lower left" if suffix == "geometry" else "upper left", fontsize=18)
        fig.suptitle("Identical-observation DG maps · activity >0.25% · Gaussian sigma 1 bin", fontsize=21)
        destination = output.with_name(f"common_peak_{suffix}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(destination.with_suffix(".png"), dpi=190, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_sensitivity", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.run_sensitivity)
    rows[["family", "encoder_credit", "worker_bonus"]] = rows.condition.apply(lambda value: pd.Series(factors(value)))
    rows.to_csv(args.output_dir / "peak_run_factors.csv", index=False)
    group_keys = ["family", "encoder_credit", "worker_bonus", "activity_cutoff_fraction", "gaussian_sigma_bins"]
    summary = rows.groupby(group_keys, dropna=False)[list(METRICS)].agg(["count", "mean", "std", "sem"])
    summary.columns = ["__".join(column) for column in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "peak_condition_summary.csv", index=False)
    paired(rows, "encoder_credit", "constant", "temporal").to_csv(args.output_dir / "peak_paired_encoder.csv", index=False)
    c05 = rows.loc[rows.family == "C05"]
    paired(c05, "worker_bonus", "constant", "temporal").to_csv(args.output_dir / "peak_paired_worker.csv", index=False)
    interaction = c05.pivot(index=["seed", "activity_cutoff_fraction", "gaussian_sigma_bins"],
                            columns=["encoder_credit", "worker_bonus"], values=list(METRICS))
    differences = pd.DataFrame(index=interaction.index)
    for metric in METRICS:
        values = interaction[metric]
        differences[f"{metric}_interaction"] = (
            values["temporal", "temporal"] - values["constant", "temporal"]
            - values["temporal", "constant"] + values["constant", "constant"]
        )
    differences.reset_index().to_csv(args.output_dir / "peak_c05_interaction.csv", index=False)
    plot_primary(rows, args.output_dir.parent / "figures" / "common_peak_comparisons")


if __name__ == "__main__":
    main()
