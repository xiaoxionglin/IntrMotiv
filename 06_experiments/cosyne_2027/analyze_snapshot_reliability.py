"""Estimate DG map repeatability from alternating contiguous behavior segments.

The 100k-observation retained snapshot carries aligned pose, activity, resets,
and segment IDs. Alternating complete contiguous segments balances time while
keeping each segment intact. Unit correlations use only bins with at least five
visits in each half; fewer than 20 shared bins or zero map variance is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study.spatial_contract import SpatialBounds, spatial_rate_maps


def split_segments(segment_id: np.ndarray, dones: np.ndarray) -> np.ndarray:
    breaks = np.empty(len(segment_id), dtype=bool)
    breaks[0] = True
    breaks[1:] = (segment_id[1:] != segment_id[:-1]) | dones[:-1]
    return (np.cumsum(breaks) - 1) % 2 == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_table", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(args.snapshot_table)
    unit_rows = []
    summary_rows = []
    for index, item in inventory.iterrows():
        with np.load(item.snapshot_path, allow_pickle=False) as snapshot:
            pose = snapshot["pose"]
            dg = snapshot["dg_activity"]
            segments = snapshot["segment_id"]
            dones = snapshot["dones"]
            bounds = SpatialBounds(*map(float, snapshot["bounds"]))
            grain = int(snapshot["grain"])
        first = split_segments(segments, dones)
        halves = [spatial_rate_maps(pose[selection], dg[selection], bounds, grain)
                  for selection in (first, ~first)]
        common = (halves[0][1] >= 5) & (halves[1][1] >= 5)
        correlations = []
        common_metadata = {
            "run_name": item.run_name, "study_id": item.study_id,
            "study_sha256": item.study_sha256, "family": item.family,
            "encoder_credit": item.encoder_credit,
            "worker_bonus": item.worker_bonus, "seed": item.seed,
            "target_env_steps": item.target_env_steps,
            "actual_env_steps": item.actual_env_steps,
            "first_half_samples": int(first.sum()),
            "second_half_samples": int((~first).sum()),
            "shared_bins_at_least_five_visits": int(common.sum()),
        }
        for unit in range(dg.shape[1]):
            x = halves[0][0][unit, common].astype(float)
            y = halves[1][0][unit, common].astype(float)
            corr = np.nan
            if len(x) >= 20 and np.std(x) > 0 and np.std(y) > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
            correlations.append(corr)
            unit_rows.append({**common_metadata, "unit_id": unit,
                              "split_segment_map_correlation": corr})
        finite = np.asarray(correlations)[np.isfinite(correlations)]
        summary_rows.append({**common_metadata,
                             "eligible_units": len(finite),
                             "mean_split_segment_map_correlation": float(finite.mean()) if len(finite) else np.nan,
                             "median_split_segment_map_correlation": float(np.median(finite)) if len(finite) else np.nan})
        if (index + 1) % 20 == 0:
            print(index + 1, "snapshots", flush=True)
    pd.DataFrame(unit_rows).to_csv(args.output / "split_segment_reliability_per_unit.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output / "split_segment_reliability_per_snapshot.csv", index=False)
    print("completed", len(summary_rows), "snapshots", flush=True)


if __name__ == "__main__":
    main()
