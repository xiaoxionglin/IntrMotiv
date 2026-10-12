"""Render selected COSYNE directed prospective-outcome matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hpc_runs.intrmotiv_study.spatial import render_graph_outcomes


SELECTED = {
    "cosyne27_c05_pushpull_s3_20261010": (
        "COS27_C05_TEMPORAL_TEMPORAL_S99",
        "COS27_C05_CONSTANT_TEMPORAL_S99",
        "COS27_C05_TEMPORAL_CONSTANT_S99",
        "COS27_C05_CONSTANT_CONSTANT_S99",
    ),
    "cosyne27_c15_encoder_s2_20261010": (
        "COS27_C15_TEMPORAL_S17", "COS27_C15_CONSTANT_S17",
    ),
    "cosyne27_c15_encoder_s3_20261010": ("COS27_C15_TEMPORAL_S99",),
    "cosyne27_c15_constant_recovery_20261011": ("COS27_REC_C15_CONSTANT_S99",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for study, names in SELECTED.items():
        for name in names:
            files = list((args.snapshot_root / study / name / "policy_00").glob(
                "snapshot_target_000100000000_actual_*.npz"
            ))
            if len(files) != 1:
                raise ValueError(f"Expected one 100M snapshot for {name}, found {len(files)}")
            with np.load(files[0], allow_pickle=False) as snapshot:
                attempts = snapshot["control_prospective_attempts"]
                successes = snapshot["control_prospective_successes"]
            render_graph_outcomes(
                attempts, successes, args.output / name,
                title=f"{name} · 100M frames",
            )
            print(name, "attempted edges", int((attempts > 0).sum()), flush=True)


if __name__ == "__main__":
    main()
