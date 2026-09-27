"""Summarize spatial-correlation kernels across the full 19-by-19 map.

Displacements are the unit of each band mean. A displacement needs at least
ten eligible cell pairs in that run. The CSV keeps support counts beside each
correlation so sparse long-distance estimates are not presented as dense maps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


LAYERS = ("dg", "ca3", "decoder_1")
BANDS = (("local_1_3", 1, 4), ("near_4_6", 4, 7),
         ("middle_7_12", 7, 13), ("long_13_18", 13, 19),
         ("diagonal_19_25", 19, 26))


def summarize_kernel(kernel: np.ndarray, pairs: np.ndarray, minimum_pairs: int = 10) -> list[dict]:
    if kernel.shape != (37, 37) or pairs.shape != kernel.shape:
        raise ValueError(f"Expected a full 37-by-37 kernel and counts, got {kernel.shape}")
    dx, dy = np.mgrid[-18:19, -18:19]
    distance = np.hypot(dx, dy)
    records = []
    for name, start, stop in BANDS:
        in_band = (distance >= start) & (distance < stop)
        supported = in_band & (pairs >= minimum_pairs) & np.isfinite(kernel)
        values = kernel[supported]
        records.append({"distance_band": name, "offsets_possible": int(in_band.sum()),
                        "offsets_supported": int(supported.sum()),
                        "eligible_cell_pairs": int(pairs[supported].sum()),
                        "mean_correlation": float(values.mean()) if len(values) else np.nan})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    records = []
    for row in manifest:
        path = args.kernels / f"{row['label_suffix']}.npz"
        with np.load(path, allow_pickle=False) as data:
            for layer in LAYERS:
                for scope in ("all_cues", "heading_matched", "cue_1", "cue_2",
                              "cue_3", "cue_4", "cue_5"):
                    stem = layer if scope == "all_cues" else (f"{layer}_heading" if scope == "heading_matched"
                                                              else f"{layer}_{scope}")
                    kernel_key = f"{stem}_kernel"
                    if kernel_key not in data:
                        if scope != "all_cues" and row["per_cue"] == "false":
                            continue
                        raise ValueError(f"Missing {kernel_key}: {path}")
                    for band in summarize_kernel(data[kernel_key], data[f"{stem}_pair_count"]):
                        records.append({"condition": row["condition"], "seed": int(row["seed"]),
                                        "checkpoint_frames": int(row["checkpoint_frames"]),
                                        "role": row["analysis_role"], "label_suffix": row["label_suffix"],
                                        "layer": layer, "scope": scope, **band})
    per_run = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output / "fullrange_bands_per_run.csv", index=False)
    group_keys = ["role", "condition", "layer", "scope", "distance_band"]
    aggregate = per_run.groupby(group_keys, as_index=False).agg(
        trained_seeds=("seed", "nunique"),
        offsets_possible=("offsets_possible", "first"),
        offsets_supported_mean=("offsets_supported", "mean"),
        eligible_cell_pairs_mean=("eligible_cell_pairs", "mean"),
        mean_correlation=("mean_correlation", "mean"),
        seeds_with_estimate=("mean_correlation", "count"))
    aggregate.to_csv(args.output / "fullrange_bands_by_condition.csv", index=False)
    print(f"Summarized {len(manifest)} full-range kernel runs")


if __name__ == "__main__":
    main()
