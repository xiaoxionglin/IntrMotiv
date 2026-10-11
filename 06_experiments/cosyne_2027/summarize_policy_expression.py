"""Summarize alternative own-policy versus uniform-random field outcomes.

Evaluation seeds are repeated probes of each training seed, so the three
training seeds per family are the replication unit. These September checkpoints
form a separate source cohort from the fresh COSYNE training matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PREFIX_COLUMNS = (
    "difference", "bits_per_step_difference", "mean_activation_difference",
    "shared_bins", "eligible_units", "own_active_units", "random_active_units",
)
FULL_COLUMNS = (
    "split_half_map_correlation", "active_only_map_cosine",
    "distinct_peak_bins", "mean_dominant_mass_fraction",
    "mean_component_count", "mean_motion_per_decision", "stationary_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(args.input / "paired_prefixes.csv")
    paired = paired[(paired.random_policy == "uniform") & (paired.decisions == 50_000) & paired.supported]
    if len(paired) != 3 * 3 * 2 * 4:
        raise ValueError("Expected 72 supported 50k paired layer probes")
    paired.to_csv(args.output / "own_uniform_50k_probe_rows.csv", index=False)
    group = ["condition", "seed", "layer"]
    seed = paired.groupby(group)[list(PREFIX_COLUMNS)].mean().reset_index()

    full = pd.read_csv(args.input / "per_run.csv")
    full = full[full.policy.isin(["own", "uniform"])]
    identity = ["condition", "seed", "eval_seed", "layer"]
    wide = full.pivot(index=identity, columns="policy", values=list(FULL_COLUMNS))
    if wide.isna().all(axis=1).any():
        raise ValueError("Own/uniform full-run probe pair missing")
    changes = pd.DataFrame(index=wide.index)
    for metric in FULL_COLUMNS:
        changes[metric + "_own_minus_uniform"] = wide[(metric, "own")] - wide[(metric, "uniform")]
    changes.reset_index().to_csv(args.output / "own_uniform_full_probe_rows.csv", index=False)
    seed = seed.merge(changes.groupby(level=[0, 1, 3]).mean().reset_index(), on=group, validate="one_to_one")
    seed.to_csv(args.output / "own_uniform_training_seed_means.csv", index=False)
    summary = seed.groupby(["condition", "layer"]).agg({
        **{name: ["mean", "std"] for name in PREFIX_COLUMNS},
        **{name + "_own_minus_uniform": ["mean", "std"] for name in FULL_COLUMNS},
    })
    summary.columns = ["__".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(args.output / "own_uniform_three_seed_summary.csv", index=False)
    print("training-seed rows", len(seed))


if __name__ == "__main__":
    main()
