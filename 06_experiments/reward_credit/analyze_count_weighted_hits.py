"""Summarize hit-conditional learner scalars from canonical exported histories.

The standardized online collector reports arithmetic means of scalar events.
Elapsed decisions and reward magnitude are conditional on correct hits, so this
adapter weights each same-step mean by its correct-hit count. The input
``per_run.csv`` supplies StudySpec identities and the declared common window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TAGS = {
    "count": "intrmotiv/hrl/control/correct_count",
    "elapsed": "intrmotiv/hrl/control/correct_elapsed_mean",
    "reward": "intrmotiv/hrl/control/correct_reward_magnitude_mean",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_history(path: Path, low: int, high: int) -> dict[str, float | int]:
    history = pd.read_csv(path)
    selected = history.loc[
        history["tag"].isin(TAGS.values()) & history["step"].between(low, high),
        ["tag", "step", "value"],
    ]
    if selected.duplicated(["tag", "step"]).any():
        raise ValueError(f"duplicate conditional metric step in {path}")
    aligned = selected.pivot(index="step", columns="tag", values="value")
    if set(aligned.columns) != set(TAGS.values()) or aligned[TAGS["count"]].isna().any():
        raise ValueError(f"correct-hit counts are missing at a selected step in {path}")
    counts = aligned[TAGS["count"]].to_numpy(dtype=float)
    elapsed = aligned[TAGS["elapsed"]].to_numpy(dtype=float)
    reward = aligned[TAGS["reward"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.allclose(counts, np.rint(counts)):
        raise ValueError(f"invalid correct-hit counts in {path}")
    active = counts > 0
    if not active.any() or not np.isfinite(elapsed[active]).all() or not np.isfinite(reward[active]).all():
        raise ValueError(f"missing finite hit-conditional values in {path}")
    total = float(counts.sum())
    return {
        "logged_steps": len(aligned),
        "positive_hit_steps": int(active.sum()),
        "correct_hit_observations": int(total),
        "hit_weighted_elapsed_decisions": float(np.dot(counts[active], elapsed[active]) / total),
        "hit_weighted_reward_magnitude": float(np.dot(counts[active], reward[active]) / total),
        "mean_correct_count_per_step": float(counts.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_collection", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--remote-source", required=True, help="Authoritative workspace history folder")
    args = parser.parse_args()

    source = args.history_collection
    manifest_path = source / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["collection"]["exported_histories"] is None:
        raise ValueError("canonical histories were not exported")
    rows = pd.read_csv(source / "per_run.csv")
    if len(rows) != manifest["expected_runs"] or rows["run_name"].duplicated().any():
        raise ValueError("canonical run count or identities do not match the manifest")
    if rows[["window_low", "window_high"]].drop_duplicates().shape[0] != 1:
        raise ValueError("runs do not share one window")

    records = []
    history_hashes = {}
    event_sources = {}
    for run in rows.itertuples(index=False):
        path = source / "histories" / f"{run.run_name}.csv"
        source_record = source / "histories" / f"{run.run_name}.json"
        metrics = summarize_history(path, int(run.window_low), int(run.window_high))
        if metrics["logged_steps"] != int(run.correct_outcome_count__n):
            raise ValueError(f"logged-step count mismatch for {run.run_name}")
        if not np.isclose(metrics["mean_correct_count_per_step"], run.correct_outcome_count, atol=1e-6):
            raise ValueError(f"correct-count mean mismatch for {run.run_name}")
        records.append({
            "run_name": run.run_name,
            "condition": run.condition,
            "seed": int(run.seed),
            "worker_bonus": run.worker_bonus,
            "window_low": int(run.window_low),
            "window_high": int(run.window_high),
            **metrics,
        })
        history_hashes[run.run_name] = sha256(path)
        event_sources[run.run_name] = json.loads(source_record.read_text())["sources"]

    per_run = pd.DataFrame.from_records(records).sort_values(["seed", "worker_bonus"])
    if per_run.groupby("seed")["worker_bonus"].apply(set).ne({"TEMPORAL", "CONSTANT"}).any():
        raise ValueError("every seed must have temporal and constant worker arms")
    paired = []
    metrics = ("correct_hit_observations", "hit_weighted_elapsed_decisions", "hit_weighted_reward_magnitude")
    for seed, group in per_run.groupby("seed"):
        arms = group.set_index("worker_bonus")
        paired.append({"seed": int(seed), **{
            f"temporal_minus_constant_{metric}": float(arms.loc["TEMPORAL", metric] - arms.loc["CONSTANT", metric])
            for metric in metrics
        }})
    paired_frame = pd.DataFrame.from_records(paired).sort_values("seed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output_dir / "hit_weighted_per_run.csv", index=False)
    paired_frame.to_csv(args.output_dir / "hit_weighted_paired.csv", index=False)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps({
        "schema": manifest["schema"],
        "workflow_version": manifest["workflow_version"],
        "study_id": manifest["study_id"],
        "study_sha256": manifest["study_sha256"],
        "source_root": args.remote_source,
        "collector_manifest_sha256": sha256(manifest_path),
        "collector_per_run_sha256": sha256(source / "per_run.csv"),
        "history_csv_sha256_by_run": history_hashes,
        "event_sources_by_run": event_sources,
        "run_count": len(per_run),
        "paired_seed_count": len(paired_frame),
        "interpretation": "Repeated learner-batch DG detector-hit observations; no independently mapped physical arrivals.",
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
