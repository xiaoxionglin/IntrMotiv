"""Rank saved historical DG spatial snapshots for poster follow-up.

This is a read-only adapter over canonical summary tables. It does not select
checkpoints for matched comparisons or infer all-DG mono counts where only the
eligible-unit fraction was saved. Refresh exact checkpoint files before any
cross-condition replay or frozen-policy evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCES = (
    ("CPD C15", "results/late_outliers_20260908/cpd_snapshots/per_snapshot.csv"),
    ("Navigation8", "data/navigation8_algorithm_screen_interim_20260916/online_spatial/per_snapshot.csv"),
)


def load_online(project: Path, family: str, relative_path: str) -> pd.DataFrame:
    table = pd.read_csv(project / relative_path)
    table = table.sort_values("actual_env_steps").groupby(
        ["condition", "seed"], as_index=False).tail(1).copy()
    return pd.DataFrame({
        "family": family,
        "condition": table.condition,
        "seed": table.seed.astype(int),
        "checkpoint_frames": table.actual_env_steps.astype(int),
        "mono_fraction_eligible": table.mono_field_unit_fraction,
        "mono_units": pd.NA,
        "all_dg_units": 16,
        "active_fraction": table.active_unit_fraction,
        "distinct_all_peak_bins": table.unique_active_peak_bins,
        "visited_cell_fraction": table.visited_cell_fraction,
        "source_protocol": "online 100k training-window spatial snapshot",
        "source_csv": relative_path,
    })


def load_corrected_c15(project: Path) -> pd.DataFrame:
    relative = "assets/corrected_core_c15_place_fields_20260908/c15_metrics.csv"
    table = pd.read_csv(project / relative)
    table = table.sort_values("checkpoint_frames").groupby(
        ["condition", "seed"], as_index=False).tail(1)
    return pd.DataFrame({
        "family": "Corrected core",
        "condition": "CCR_C15_TOPOLOGY_UCB_DIRECT_O1",
        "seed": table.seed.astype(int),
        "checkpoint_frames": table.checkpoint_frames.astype(int),
        "mono_fraction_eligible": table.mono_field_fraction,
        "mono_units": (table.mono_field_fraction * table.field_eligible_units).round().astype(int),
        "all_dg_units": 16,
        "active_fraction": table.active_units / 16,
        "distinct_all_peak_bins": table.active_unique_peak_bins,
        "visited_cell_fraction": table.visited_cell_fraction,
        "source_protocol": "frozen-policy 10k-decision historical evaluator",
        "source_csv": relative,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("06_experiments"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = pd.concat([load_online(args.project, *source) for source in SOURCES] +
                     [load_corrected_c15(args.project)], ignore_index=True)
    ranking = runs.groupby(["family", "condition", "source_protocol"],
                           as_index=False).agg(
        seeds=("seed", "nunique"),
        mono_fraction_eligible_mean=("mono_fraction_eligible", "mean"),
        mono_fraction_eligible_min=("mono_fraction_eligible", "min"),
        mono_fraction_eligible_max=("mono_fraction_eligible", "max"),
        distinct_all_peak_bins_mean=("distinct_all_peak_bins", "mean"),
        visited_cell_fraction_mean=("visited_cell_fraction", "mean"),
    ).sort_values(["mono_fraction_eligible_mean", "mono_fraction_eligible_min"],
                  ascending=False)
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / "historical_candidate_runs.csv", index=False)
    ranking.to_csv(args.output / "historical_candidate_ranking.csv", index=False)
    print(f"Screened {len(runs)} latest saved rows across {len(ranking)} conditions")


if __name__ == "__main__":
    main()
