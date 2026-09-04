#!/usr/bin/env python3
"""Build the interactive DPR seed-99 place-field and graph viewer.

Inputs are the canonical telemetry manifest, the standardized per-run factor
table, and manifest-driven `place_fields.npz` artifacts. No factor metadata is
recovered from run or directory names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


REC_LABEL = {"monitor": "MON", "directional": "DIR", "predictive": "PRED"}
GOAL_LABEL = {"legacy": "LEG", "target_id_film": "FILM"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--per-run", type=Path, required=True)
    result.add_argument("--template", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def quantize_maps(rate_maps: np.ndarray) -> list[list[int]]:
    maps = []
    for unit in range(rate_maps.shape[-1]):
        values = np.nan_to_num(
            np.maximum(rate_maps[:, :, unit], 0.0), nan=0.0, posinf=0.0, neginf=0.0
        )
        peak = float(values.max())
        scaled = np.rint(values / peak * 255.0).astype(np.uint8) if peak > 0 else np.zeros_like(values, np.uint8)
        maps.append(scaled.reshape(-1).astype(int).tolist())
    return maps


def main() -> None:
    args = parser().parse_args()
    factors = pd.read_csv(args.per_run)[
        ["condition", "base", "recruitment", "goal_conditioning"]
    ].drop_duplicates()
    if factors.condition.duplicated().any():
        raise SystemExit("Condition metadata is not unique in the canonical per-run table")
    factor_by_condition = factors.set_index("condition").to_dict("index")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if int(row["seed"]) == 99 and int(row["target_frames"]) == 75_000_000]
    if len(selected) != 18:
        raise SystemExit(f"Expected 18 terminal seed-99 rows, found {len(selected)}")

    payload: dict[str, object] = {}
    for row in selected:
        condition = row["condition"]
        factor = factor_by_condition[condition]
        rec = REC_LABEL[factor["recruitment"]]
        goal = GOAL_LABEL[factor["goal_conditioning"]]
        key = f'{factor["base"]}_{rec}_{goal}'
        matches = list(args.raw_dir.glob(f'*{row["label_suffix"]}/place_fields.npz'))
        if len(matches) != 1:
            raise SystemExit(f"Expected one artifact for {row['label_suffix']}, found {len(matches)}")
        with np.load(matches[0], allow_pickle=False) as artifact:
            rate_maps = np.asarray(artifact["rate_maps"], dtype=float)
            occupancy = np.asarray(artifact["occupancy"], dtype=float) > 0
            confidence = np.nan_to_num(np.asarray(artifact["control_edge_confidence"], dtype=float))
            attempts = np.nan_to_num(np.asarray(artifact["control_attempts"], dtype=float))
            tctrl = np.nan_to_num(np.asarray(artifact["control_tctrl"], dtype=float))
            posterior = (confidence + 1.0) / (attempts + 2.0)
            reliable = (tctrl > 0) & (confidence >= 0.5) & (posterior >= 0.5)
            np.fill_diagonal(reliable, False)
            attempted = (attempts > 0) | (confidence > 0) | (tctrl > 0)
            np.fill_diagonal(attempted, False)
            incoming = np.where(reliable, confidence, 0.0).sum(axis=0)
            edges = []
            for source, target in np.argwhere(attempted):
                edges.append({
                    "source": int(source),
                    "target": int(target),
                    "attempted": True,
                    "reliable": bool(reliable[source, target]),
                    "confidence": round(float(confidence[source, target]), 3),
                    "attempts": round(float(attempts[source, target]), 3),
                    "posterior": round(float(posterior[source, target]), 3),
                    "tctrl": round(float(tctrl[source, target]), 2),
                })
            active = np.nan_to_num(np.asarray(artifact["active_fraction"], dtype=float))
            payload[key] = {
                "label": f'{factor["base"]} · {rec} · {"FiLM" if goal == "FILM" else "LEG"}',
                "frames": int(row["checkpoint_frames"]),
                "maps": quantize_maps(rate_maps),
                "occupancy": occupancy.reshape(-1).astype(int).tolist(),
                "active": np.round(active, 5).tolist(),
                "maxActive": round(float(max(1e-6, active.max())), 5),
                "si": np.round(
                    np.nan_to_num(np.asarray(artifact["spatial_information"], dtype=float)), 4
                ).tolist(),
                "edges": edges,
                "reliableCount": int(reliable.sum()),
                "incoming": np.round(incoming, 3).tolist(),
                "inDegree": reliable.sum(axis=0).astype(int).tolist(),
                "outDegree": reliable.sum(axis=1).astype(int).tolist(),
            }

    template = args.template.read_text(encoding="utf-8")
    if template.count("__DPR_DATA__") != 1:
        raise SystemExit("Viewer template must contain exactly one __DPR_DATA__ marker")
    output = template.replace(
        "__DPR_DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    if args.output.stat().st_size >= 1_000_000:
        raise SystemExit(f"Viewer exceeds 1 MB: {args.output.stat().st_size} bytes")
    print(f"wrote {len(payload)} conditions, {args.output.stat().st_size} bytes: {args.output}")


if __name__ == "__main__":
    main()
