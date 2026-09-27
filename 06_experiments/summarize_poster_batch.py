"""Combine canonical poster-evaluation artifacts by explicit manifest row.

This keeps policy-driven behavior separate from common-panel representation.
The script reads existing evaluator outputs and writes small tabular summaries;
it does not launch an environment or infer metadata from run names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRASTS = (
    ("Saturday ARR-SRC", "SAT_C15_ARR_MON_FILM", "SAT_C15_SRC_MON_FILM"),
    ("CPU2048 Direct-Waypoint", "CPU2048_DIRECT_F16_DDQN_HER", "CPU2048_WAYPOINT_DECODER_F64_DDQN_HER"),
    ("DGP HIT-FIRST", "DGP_C15_HIT_JOINT_LEG", "DGP_C15_FIRST_JOINT_LEG"),
    ("D50 Source-Random", "CR5C_D50_W_SOURCE_DG", "CR5C_D50_W_RAND_DG"),
    ("D51 Source-Random", "CR5C_D51_W_SOURCE_DG", "CR5C_D51_W_RAND_DG"),
)
METRICS = (
    "dg_active_units", "dg_map_cosine_active", "dg_spatial_information_active_mean",
    "dg_normalized_si_bits_per_activity",
    "dg_distinct_peak_bins", "visited_cell_fraction", "stored_graph_confidence_fraction",
    "stored_graph_attempted_edges", "mean_flow_coherence", "dg_near_minus_far",
    "ca3_near_minus_far", "decoder_1_near_minus_far", "dg_heading_near_minus_far",
    "ca3_heading_near_minus_far", "decoder_1_heading_near_minus_far",
)


def active_map_cosine(maps: np.ndarray, occupancy: np.ndarray, active: np.ndarray) -> float:
    activity = np.nan_to_num(maps[occupancy > 0][:, active].T, nan=0.0)
    if len(activity) < 2:
        return float("nan")
    norms = np.linalg.norm(activity, axis=1)
    positive = norms > 0
    activity = activity[positive] / norms[positive, None]
    if len(activity) < 2:
        return float("nan")
    cosine = activity @ activity.T
    return float(cosine[np.triu_indices(len(activity), k=1)].mean())


def field_metrics(path: Path) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as data:
        occupancy = data["occupancy"]
        maps = data["rate_maps"]
        fractions = data["active_fraction"]
        information = data["spatial_information"]
        active = fractions > 0
        peaks = np.nanargmax(maps[:, :, active].reshape(-1, int(active.sum())), axis=0) if active.any() else []
        attempted = data["control_attempts"] if "control_attempts" in data else np.array([])
        confidence = data["control_edge_confidence"] if "control_edge_confidence" in data else np.array([])
        return {
            "dg_active_units": int(active.sum()),
            "dg_silent_units": int((~active).sum()),
            "dg_map_cosine_active": active_map_cosine(maps, occupancy, active),
            "dg_spatial_information_active_mean": float(information[active].mean()) if active.any() else float("nan"),
            "dg_normalized_si_bits_per_activity": (float(np.mean(information[active] / fractions[active]))
                                                    if active.any() else float("nan")),
            "dg_distinct_peak_bins": int(len(set(np.asarray(peaks).tolist()))),
            "visited_cell_fraction": float((occupancy > 0).mean()),
            "stored_graph_attempted_edges": int((attempted > 0).sum()) if attempted.size else 0,
            "stored_graph_confidence_fraction": (float(confidence.sum() / attempted.sum())
                                                  if attempted.size and attempted.sum() else float("nan")),
        }


def indexed_artifact(directory: Path, suffix: str, filename: str) -> Path | None:
    matches = list(directory.glob(f"*__{suffix}/{filename}"))
    if len(matches) > 1:
        raise ValueError(f"Multiple {filename} artifacts for {suffix}")
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.root / "staged_manifest.tsv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    flow_file = args.root / "frozen/figures/behavior_graph_summary.csv"
    flow = pd.read_csv(flow_file).set_index("label") if flow_file.exists() else pd.DataFrame()
    records = []
    for index, row in enumerate(manifest):
        suffix = row["label_suffix"]
        record = {"row": index, "condition": row["condition"], "seed": int(row["seed"]),
                  "checkpoint_frames": int(row["checkpoint_frames"]), "checkpoint": row["checkpoint"],
                  "label_suffix": suffix}
        frozen = indexed_artifact(args.root / "frozen/raw", suffix, "place_fields.npz")
        if frozen:
            record.update(field_metrics(frozen))
            record["frozen_complete"] = True
            if not flow.empty and frozen.parent.name in flow.index:
                record["mean_flow_coherence"] = float(flow.loc[frozen.parent.name, "mean_flow_coherence"])
        else:
            record["frozen_complete"] = False
        for panel_name, rows in (("reduced", set(range(0, 6)) | set(range(12, 18))),
                                 ("cpu", set(range(6, 12)) | set(range(18, 21))),
                                 ("cue", set(range(21, 33)))):
            if index not in rows:
                continue
            replay = indexed_artifact(args.root / f"replay_{panel_name}/raw", suffix, "place_fields.npz")
            if replay:
                common = field_metrics(replay)
                record.update({f"replay_{key}": value for key, value in common.items()
                               if key.startswith("dg_") or key == "visited_cell_fraction"})
                record["replay_complete"] = True
            else:
                record["replay_complete"] = False
            kernel_file = args.root / f"kernels_{panel_name}/results/{suffix}.json"
            if kernel_file.exists():
                kernel = json.loads(kernel_file.read_text())
                for key, value in kernel.items():
                    if key.endswith(("near_minus_far", "length_scale_bins")):
                        record[key] = value
                record["kernel_complete"] = True
            else:
                record["kernel_complete"] = False
        records.append(record)
    per_run = pd.DataFrame(records)
    per_run.to_csv(args.out_dir / "per_run.csv", index=False)
    paired = []
    for label, a, b in CONTRASTS:
        first = per_run[per_run.condition == a].set_index("seed")
        second = per_run[per_run.condition == b].set_index("seed")
        for seed in sorted(first.index.intersection(second.index)):
            for metric in METRICS + tuple(f"replay_{x}" for x in METRICS if x.startswith("dg_")):
                if metric in first.columns and metric in second.columns:
                    av, bv = first.loc[seed, metric], second.loc[seed, metric]
                    paired.append({"contrast": label, "seed": seed, "metric": metric,
                                   "first": av, "second": bv, "first_minus_second": av - bv})
    pd.DataFrame(paired).to_csv(args.out_dir / "paired.csv", index=False)
    print(f"Summarized {sum(per_run.frozen_complete)} frozen, "
          f"{sum(per_run.get('replay_complete', []))} replay, "
          f"{sum(per_run.get('kernel_complete', []))} kernel runs")


if __name__ == "__main__":
    main()
