"""Summarize layerwise transfer kernels with equal weight for each cue.

The shared 10,001-observation panel is not cue-balanced. This adapter averages
the five separately estimated cue kernels per run, and reports their pair
counts so sparse cue evidence remains visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAYERS = ("dg", "ca3", "decoder_1")
MEASURES = ("near_corr", "far_corr", "near_minus_far")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    records = []
    for row in manifest:
        path = args.kernels / f"{row['label_suffix']}.json"
        result = json.loads(path.read_text())
        cues = result.get("per_cue", {})
        if set(cues) != {"1", "2", "3", "4", "5"}:
            raise ValueError(f"Incomplete cue set: {path}: {set(cues)}")
        record = {"condition": row["condition"], "seed": int(row["seed"]),
                  "checkpoint_frames": int(row["checkpoint_frames"]),
                  "label_suffix": row["label_suffix"], "observations": result["observations"]}
        for cue, evidence in cues.items():
            record[f"cue_{cue}_observations"] = evidence["observations"]
            for layer in LAYERS:
                layer_data = evidence[layer]
                record[f"{layer}_cue_{cue}_near_pairs"] = layer_data["near_pairs"]
                record[f"{layer}_cue_{cue}_far_pairs"] = layer_data["far_pairs"]
        for layer in LAYERS:
            for measure in MEASURES:
                values = [cues[str(cue)][layer][measure] for cue in range(1, 6)]
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {layer}/{measure} in {path}")
                record[f"{layer}_{measure}_cue_equal"] = float(np.mean(values))
            record[f"{layer}_near_minus_far_all_observations"] = result[f"{layer}_near_minus_far"]
            record[f"{layer}_far_corr_all_observations"] = result[f"{layer}_far_corr"]
        records.append(record)
    per_run = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output / "cue_kernel_per_run.csv", index=False)
    paired = []
    architectures = sorted({condition.split("_")[1] for condition in per_run.condition})
    for architecture in architectures:
        source = per_run[per_run.condition == f"CR5C_{architecture}_W_SOURCE_DG"].set_index("seed")
        random = per_run[per_run.condition == f"CR5C_{architecture}_W_RAND_DG"].set_index("seed")
        if len(source) != 3 or set(source.index) != set(random.index):
            raise ValueError(f"Incomplete paired seeds for {architecture}")
        for seed in sorted(source.index):
            for layer in LAYERS:
                for measure in ("near_minus_far_cue_equal", "far_corr_cue_equal",
                                "near_minus_far_all_observations", "far_corr_all_observations"):
                    key = f"{layer}_{measure}"
                    paired.append({"architecture": architecture, "seed": seed,
                                   "layer": layer, "metric": measure,
                                   "source": source.loc[seed, key],
                                   "random": random.loc[seed, key],
                                   "source_minus_random": source.loc[seed, key] - random.loc[seed, key]})
    pd.DataFrame(paired).to_csv(args.output / "cue_kernel_paired.csv", index=False)
    print(f"Summarized {len(per_run)} cue-complete kernel runs")


if __name__ == "__main__":
    main()
