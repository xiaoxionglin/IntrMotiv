"""Join canonical COSYNE snapshots and retain seed-level candidate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "active_unit_mean_spatial_information",
    "active_unit_fraction",
    "active_only_map_cosine",
    "unique_active_peak_bins",
    "mono_field_unit_fraction",
    "visited_cell_fraction",
    "path_efficiency",
    "mean_physical_step_distance",
    "stationary_step_fraction",
    "graph_reliable_edge_count",
    "graph_reachable_pair_fraction",
    "graph_reliable_global_efficiency",
    "graph_prospective_attempt_count",
    "graph_prospective_success_fraction",
    "graph_grounded_controllability",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(args.catalog).fillna("")
    catalog = catalog[
        catalog.cohort.isin(["cosyne", "cosyne_replacement"])
        & catalog.attempt_status.str.startswith("verified_100m")
    ]
    snapshots = []
    for directory in sorted(args.snapshot_root.iterdir()):
        path = directory / "per_snapshot.csv"
        if not path.exists():
            continue
        metadata = json.loads((directory / "analysis_manifest.json").read_text())
        table = pd.read_csv(path)
        table["study_id"] = metadata["study_id"]
        table["study_sha256"] = metadata["study_sha256"]
        table["snapshot_complete"] = metadata["complete"]
        snapshots.append(table)
    data = pd.concat(snapshots, ignore_index=True)
    data = data.merge(
        catalog[["run_name", "study_id", "study_sha256", "attempt_status", "job_id", "source_base_commit", "recovery_patch_sha256"]],
        on=["run_name", "study_id", "study_sha256"], how="inner", validate="many_to_one",
    )
    data["encoder_credit"] = data["encoder_credit"].str.lower()
    data["worker_bonus"] = data["worker_bonus"].str.lower()
    if data.duplicated(["run_name", "target_env_steps"]).any():
        raise ValueError("Duplicate snapshot age for a run")
    observed = set(data.run_name)
    missing_runs = set(catalog.run_name) - observed
    if missing_runs:
        raise ValueError(f"Missing snapshot runs: {sorted(missing_runs)}")
    data.to_csv(args.output / "snapshot_long.csv", index=False)
    terminal = data[data.target_env_steps == 100_000_000].copy()
    if terminal.run_name.nunique() != len(catalog):
        raise ValueError("Missing terminal snapshot")
    terminal.to_csv(args.output / "terminal_seed_metrics.csv", index=False)

    group = ["family", "encoder_credit", "worker_bonus"]
    results = terminal.groupby(group).agg(
        n_seeds=("seed", "nunique"),
        **{f"{name}__mean": (name, "mean") for name in METRICS},
        **{f"{name}__sd": (name, "std") for name in METRICS},
    ).reset_index()
    for name in METRICS:
        results[f"{name}__sem"] = results[f"{name}__sd"] / np.sqrt(results.n_seeds)
    results.to_csv(args.output / "terminal_condition_summary.csv", index=False)

    pairs = []
    for (family, worker, seed), frame in terminal.groupby(["family", "worker_bonus", "seed"]):
        by_mode = frame.set_index("encoder_credit")
        if not {"temporal", "constant"}.issubset(by_mode.index):
            continue
        row = {"family": family, "worker_bonus": worker, "seed": seed,
               "temporal_run": by_mode.loc["temporal", "run_name"],
               "constant_run": by_mode.loc["constant", "run_name"]}
        row.update({f"{name}__temporal_minus_constant":
                    by_mode.loc["temporal", name] - by_mode.loc["constant", name]
                    for name in METRICS})
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(args.output / "paired_encoder_terminal.csv", index=False)

    worker_pairs = []
    for (encoder, seed), frame in terminal[terminal.family == "C05"].groupby(["encoder_credit", "seed"]):
        by_mode = frame.set_index("worker_bonus")
        if not {"temporal", "constant"}.issubset(by_mode.index):
            continue
        row = {"family": "C05", "encoder_credit": encoder, "seed": seed}
        row.update({f"{name}__temporal_minus_constant":
                    by_mode.loc["temporal", name] - by_mode.loc["constant", name]
                    for name in METRICS})
        worker_pairs.append(row)
    pd.DataFrame(worker_pairs).to_csv(args.output / "paired_worker_terminal.csv", index=False)

    complete_cells = catalog.groupby(["family", "encoder_credit", "worker_bonus"]).seed.nunique()
    print("terminal runs", len(terminal), "observed per cell", complete_cells.to_dict())


if __name__ == "__main__":
    main()
