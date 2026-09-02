#!/usr/bin/env python3
"""Summarize the persistent HRL graph stored in IntrMotiv checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


PREFIX = "core.policy_graph."


def parse_checkpoint(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    return label, Path(path)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def summarize(label: str, path: Path, threshold: float) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    tctrl = model[PREFIX + "tctrl"].detach().cpu().numpy()
    confidence = model[PREFIX + "edge_confidence"].detach().cpu().numpy()
    attempts = model[PREFIX + "control_attempts"].detach().cpu().numpy()
    node_visits = model[PREFIX + "node_visits"].detach().cpu().numpy()

    off_diagonal = ~np.eye(tctrl.shape[0], dtype=bool)
    known = off_diagonal & (tctrl > 0) & (confidence >= threshold)
    known_times = tctrl[known]
    known_confidence = confidence[known]
    known_attempts = attempts[known]

    reciprocal = known & known.T
    source_nodes = np.any(known, axis=1)
    target_nodes = np.any(known, axis=0)
    source_outdegree = known.sum(axis=1)
    target_indegree = known.sum(axis=0)
    nonzero_indegree = target_indegree[target_indegree > 0]

    pose_valid = model.get(PREFIX + "pose_valid")
    valid_pose_count = int(pose_valid.sum().item()) if pose_valid is not None else 0

    return {
        "label": label,
        "checkpoint": str(path),
        "env_steps": int(checkpoint["env_steps"]),
        "field_count": int(tctrl.shape[0]),
        "known_edges": int(known.sum()),
        "known_edge_fraction": float(known.sum() / off_diagonal.sum()),
        "source_nodes": int(source_nodes.sum()),
        "target_nodes": int(target_nodes.sum()),
        "target_ids": ";".join(str(index) for index in np.flatnonzero(target_nodes)),
        "target_indegrees": ";".join(str(int(value)) for value in nonzero_indegree),
        "source_outdegree_median": quantile(source_outdegree, 0.5),
        "max_target_indegree": int(target_indegree.max()),
        "max_target_edge_fraction": float(target_indegree.max() / known.sum()) if known.any() else 0.0,
        "reciprocal_directed_edges": int(reciprocal.sum()),
        "reciprocal_fraction_of_known": float(reciprocal.sum() / known.sum()) if known.any() else 0.0,
        "tctrl_min": float(known_times.min()) if known_times.size else float("nan"),
        "tctrl_q25": quantile(known_times, 0.25),
        "tctrl_median": quantile(known_times, 0.5),
        "tctrl_mean": float(known_times.mean()) if known_times.size else float("nan"),
        "tctrl_q75": quantile(known_times, 0.75),
        "tctrl_max": float(known_times.max()) if known_times.size else float("nan"),
        "tctrl_le_5_fraction": float(np.mean(known_times <= 5)) if known_times.size else float("nan"),
        "tctrl_le_8_fraction": float(np.mean(known_times <= 8)) if known_times.size else float("nan"),
        "tctrl_lt_10_fraction": float(np.mean(known_times < 10)) if known_times.size else float("nan"),
        "confidence_median": quantile(known_confidence, 0.5),
        "attempts_median": quantile(known_attempts, 0.5),
        "node_visits_min": float(node_visits.min()),
        "node_visits_median": quantile(node_visits, 0.5),
        "node_visits_max": float(node_visits.max()),
        "valid_landmark_poses": valid_pose_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [summarize(label, path, args.confidence_threshold) for label, path in args.checkpoint]
    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
