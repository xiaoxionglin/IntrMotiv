"""Join canonical native-policy 100M place-field rows without duplicate seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_factors import factors


COHORTS = (
    "c01_temporal_s3", "c01_constant_s3", "c01_encoder_s2",
    "c05_pushpull_s3", "c05_pushpull_s2", "c15_encoder_s2", "c15_available",
)

METRICS = (
    "visited_cell_fraction", "mean_active_fraction", "silent_units",
    "active_unit_mean_si_bits", "active_map_cosine_mean",
    "active_unique_peak_bins", "active_peak_bin_entropy",
    "active_pairwise_peak_distance_bins", "field_eligible_units",
    "mono_field_fraction", "prethreshold_map_cosine_mean",
)


def paired(rows: pd.DataFrame, factor: str, left: str, right: str) -> pd.DataFrame:
    keys = ["family", "seed", "worker_bonus" if factor == "encoder_credit" else "encoder_credit"]
    left_rows = rows.loc[rows[factor] == left, keys + list(METRICS)]
    right_rows = rows.loc[rows[factor] == right, keys + list(METRICS)]
    result = left_rows.merge(right_rows, on=keys, validate="one_to_one", suffixes=(f"_{left}", f"_{right}"))
    for metric in METRICS:
        result[f"{metric}_delta"] = result[f"{metric}_{right}"] - result[f"{metric}_{left}"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_dir", type=Path)
    parser.add_argument("source_map", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for cohort in COHORTS:
        data = pd.read_csv(args.native_dir / cohort / "derived_place_field_metrics.csv")
        data = data.loc[data.target_frames == 100_000_000].copy()
        data.insert(0, "cohort", cohort)
        parts.append(data)
    joined = pd.concat(parts, ignore_index=True)
    sources = pd.read_csv(args.source_map)[
        ["label_suffix", "study_id", "study_sha256", "source_base_commit", "recovery_patch_sha256"]
    ]
    joined = joined.merge(sources, on="label_suffix", how="left", validate="one_to_one")
    if joined.study_sha256.isna().any():
        raise ValueError("Missing StudySpec fingerprint for a native terminal row")
    joined[["family", "encoder_credit", "worker_bonus"]] = joined.condition.apply(lambda value: pd.Series(factors(value)))
    if len(joined) != 38 or joined.duplicated(["condition", "seed"]).any():
        raise ValueError(f"Expected 38 unique verified 100M endpoints, got {len(joined)}")
    joined.to_csv(args.output_dir / "native_terminal_seed_metrics.csv", index=False)
    keys = ["family", "encoder_credit", "worker_bonus"]
    summary = joined.groupby(keys)[list(METRICS)].agg(["count", "mean", "std", "sem"])
    summary.columns = ["__".join(column) for column in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "native_terminal_condition_summary.csv", index=False)
    paired(joined, "encoder_credit", "constant", "temporal").to_csv(
        args.output_dir / "native_terminal_paired_encoder.csv", index=False)
    paired(joined.loc[joined.family == "C05"], "worker_bonus", "constant", "temporal").to_csv(
        args.output_dir / "native_terminal_paired_worker.csv", index=False)


if __name__ == "__main__":
    main()
