#!/usr/bin/env python3
"""Embed selected aligned rollout trajectories and DG rate maps in the viewer."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import numpy as np


RUNS = {
    "GSR_C13_D4_H10K_S99": {
        "label": "C13 · D4 · H10k · seed 99",
        "note": "selective, no recruitment",
        "checkpoint": 73.859072,
        "assignments": 0,
        "si": 0.1773,
        "cosine": 0.0515,
        "peaks": 16,
    },
    "GSR_C05_D8_H10K_S8": {
        "label": "C05 · D8 · H10k · seed 8",
        "note": "15 assignments",
        "checkpoint": 74.219520,
        "assignments": 15,
        "si": 0.1106,
        "cosine": 0.0946,
        "peaks": 16,
    },
    "GSR_C05_D8_H5K_S123": {
        "label": "C05 · D8 · H5k · seed 123",
        "note": "lower peak diversity",
        "checkpoint": 73.924608,
        "assignments": 3,
        "si": 0.1046,
        "cosine": 0.1800,
        "peaks": 11,
    },
    "GSR_C15_D4_H10K_S99": {
        "label": "C15 · D4 · H10k · seed 99",
        "note": "diverse, no recruitment",
        "checkpoint": 75.792384,
        "assignments": 0,
        "si": 0.1465,
        "cosine": 0.0687,
        "peaks": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-stride", type=int, default=5)
    return parser.parse_args()


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def locate_run(root: Path, prefix: str) -> Path:
    matches = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix + "__"))
    if len(matches) != 1:
        raise ValueError(f"Expected one directory matching {prefix!r}, found {len(matches)}")
    return matches[0]


def rate_map_payload(run_dir: Path) -> tuple[str, str]:
    with np.load(run_dir / "place_fields.npz", allow_pickle=False) as artifact:
        occupancy = artifact["occupancy"] > 0
        rate_maps = artifact["rate_maps"]
    if occupancy.shape != (19, 19) or rate_maps.shape != (19, 19, 16):
        raise ValueError(f"Unexpected place-field shapes in {run_dir}")

    mask = np.ascontiguousarray(occupancy[::-1].astype(np.uint8))
    mask_bool = mask.astype(bool)
    quantized_units = []
    for unit in range(16):
        values = np.nan_to_num(rate_maps[:, :, unit][::-1], nan=0.0)
        maximum = float(values[mask_bool].max()) if mask_bool.any() else 0.0
        scaled = values / maximum if maximum > 0 else values
        quantized_units.append(np.clip(np.rint(scaled * 254), 0, 254).astype(np.uint8))
    return encode(mask), encode(np.stack(quantized_units))


def pose_payload(run_dir: Path, stride: int) -> str:
    if stride < 1:
        raise ValueError("pose stride must be positive")
    with (run_dir / "pose.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No pose rows in {run_dir}")
    required = {"x", "y", "num_traj"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing pose columns in {run_dir}")

    payload = bytearray()
    for index, row in enumerate(rows):
        if index % stride and index != len(rows) - 1:
            continue
        x = round((float(row["x"]) - 100.0) / 1900.0 * 255.0)
        y = round((float(row["y"]) - 100.0) / 1900.0 * 255.0)
        episode = int(float(row["num_traj"]))
        payload.extend((max(0, min(255, x)), max(0, min(255, y)), max(0, min(255, episode))))
    return base64.b64encode(payload).decode("ascii")


def main() -> None:
    args = parse_args()
    data = []
    for prefix, metadata in RUNS.items():
        run_dir = locate_run(args.input_root, prefix)
        mask, maps = rate_map_payload(run_dir)
        data.append(
            {
                "id": prefix,
                **metadata,
                "mask": mask,
                "maps": maps,
                "pose": pose_payload(run_dir, args.pose_stride),
            }
        )

    template = args.template.read_text()
    placeholder = "__RUN_DATA__"
    if template.count(placeholder) != 1:
        raise ValueError(f"Template must contain exactly one {placeholder} placeholder")
    output = template.replace(placeholder, json.dumps(data, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(args.output)


if __name__ == "__main__":
    main()
