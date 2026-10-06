"""Summarize the declared 75M frozen coverage and command panels.

The manifests supply condition, seed, and checkpoint identities. Raw evaluation
artifacts remain on NEMO2; copy only their small JSON summaries into the
matching local evaluation/raw directories before running this adapter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CUES = ("rich", "control")


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def unique_artifact(folder: Path, label: str, filename: str) -> Path:
    matches = list(folder.glob(f"*{label}/{filename}"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} for {label} in {folder}; found {len(matches)}")
    return matches[0]


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def coverage_rows() -> list[dict]:
    rows = []
    for cue in CUES:
        base = ROOT / f"{cue}_frozen"
        for item in manifest_rows(base / "trajectory_manifest.tsv"):
            label = item["label_suffix"]
            raw = base / "evaluation" / "raw"
            policy = read_json(unique_artifact(raw, label, "policy_episode_coverage.json"))
            random = read_json(unique_artifact(raw, label, "uniform_random_episode_coverage.json"))
            for payload, name in ((policy, "policy"), (random, "uniform_random")):
                if payload["schema"] != "intrmotiv/episode-coverage/v1":
                    raise ValueError(f"Unexpected coverage schema for {label}")
                if payload["policy"] != name or not payload["frozen_state_verified"]:
                    raise ValueError(f"Invalid frozen {name} payload for {label}")
                if len(payload["episodes"]) != 20:
                    raise ValueError(f"Expected 20 {name} episodes for {label}")
            if policy["geometry_sha256"] != random["geometry_sha256"]:
                raise ValueError(f"Geometry mismatch for {label}")
            p = {e["reset_seed"]: e for e in policy["episodes"]}
            q = {e["reset_seed"]: e for e in random["episodes"]}
            if sorted(p) != sorted(q) or len(p) != 20:
                raise ValueError(f"Reset-seed mismatch for {label}")
            if any(p[s]["action_seed"] != q[s]["action_seed"] for s in p):
                raise ValueError(f"Action-seed mismatch for {label}")
            paired = [p[s]["accessible_coverage_auc"] - q[s]["accessible_coverage_auc"] for s in p]
            rows.append(
                {
                    "cue": cue,
                    "family": item["family"],
                    "seed": item["seed"],
                    "checkpoint_frames": item["checkpoint_frames"],
                    "label_suffix": label,
                    "geometry_sha256": policy["geometry_sha256"],
                    "episodes": len(p),
                    "policy_coverage_auc": policy["mean_accessible_coverage_auc"],
                    "uniform_random_coverage_auc": random["mean_accessible_coverage_auc"],
                    "paired_policy_minus_random_auc": sum(paired) / len(paired),
                }
            )
    if len(rows) != 12:
        raise ValueError(f"Expected 12 frozen coverage rows, found {len(rows)}")
    if len({row["geometry_sha256"] for row in rows}) != 1:
        raise ValueError("Frozen coverage rows do not share one verified geometry")
    if {row["checkpoint_frames"] for row in rows} != {"75005952"}:
        raise ValueError("Frozen coverage rows do not share the exact 75M checkpoint")
    return rows


def intervention_rows() -> list[dict]:
    rows = []
    for cue in CUES:
        base = ROOT / f"{cue}_four_source"
        for item in manifest_rows(base / "intervention_manifest.tsv"):
            label = item["label_suffix"]
            summary = read_json(unique_artifact(base / "interventions" / "raw", label, "intervention_summary.json"))
            if summary["protocol"] != "landmark-matched-commands-v1":
                raise ValueError(f"Unexpected intervention protocol for {label}")
            if not all(summary[k] for k in ("exact_start_verified", "policy_frozen", "graph_frozen")):
                raise ValueError(f"Unverified frozen intervention for {label}")
            if summary["checkpoint"] != item["checkpoint"]:
                raise ValueError(f"Checkpoint mismatch for {label}")
            rows.append(
                {
                    "cue": cue,
                    "family": item["family"],
                    "seed": item["seed"],
                    "checkpoint_frames": item["checkpoint_frames"],
                    "label_suffix": label,
                    **{key: summary[key] for key in (
                        "supported_sources", "sources_evaluated", "source_panel_coverage",
                        "starts_evaluated", "paired_comparisons", "paired_arrival_lift",
                        "mean_initial_action_total_variation", "censored_rows", "rows",
                    )},
                    "missing": json.dumps(summary["missing"], separators=(",", ":")),
                }
            )
    if len(rows) != 12:
        raise ValueError(f"Expected 12 intervention rows, found {len(rows)}")
    if {row["checkpoint_frames"] for row in rows} != {"75005952"}:
        raise ValueError("Intervention rows do not share the exact 75M checkpoint")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    write_csv(ROOT / "frozen_coverage_per_run.csv", coverage_rows())
    write_csv(ROOT / "four_source_interventions_per_run.csv", intervention_rows())
