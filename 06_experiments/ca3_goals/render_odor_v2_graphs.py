"""Render declared terminal graph evidence with the shared study atlas renderer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spatial import render_graph_outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--target", type=int, default=75_000_000)
    args = parser.parse_args()

    study = load_study(args.study)
    if study.provenance()["study_sha256"] != "c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de":
        raise ValueError("Study fingerprint differs from the launched batch")
    workspace = Path(study.workspace_root).resolve()
    args.snapshot_root.resolve().relative_to(workspace)
    args.out_dir.resolve().relative_to(workspace)

    rows = []
    for run in study.expand_runs():
        if run.seed != args.seed:
            continue
        pattern = (
            f"{run.name}_/analysis/online_spatial/unbatched/{run.name}/policy_00/"
            f"snapshot_target_{args.target:012d}_actual_*.npz"
        )
        matches = list(args.snapshot_root.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one snapshot for {run.name}, found {len(matches)}")
        with np.load(matches[0], allow_pickle=False) as snapshot:
            if str(snapshot["run_name"].item()) != run.name:
                raise ValueError(f"Snapshot identity mismatch: {matches[0]}")
            attempts = np.asarray(snapshot["control_prospective_attempts"], dtype=float)
            successes = np.asarray(snapshot["control_prospective_successes"], dtype=float)
            reliable_edges = int(np.asarray(snapshot["graph_reliable_edge_count"]).item())
            reachable_fraction = float(np.asarray(snapshot["graph_reachable_pair_fraction"]).item())

        output = args.out_dir / run.name / f"target_{args.target:012d}_policy_00_graph_outcomes"
        render_graph_outcomes(
            attempts,
            successes,
            output,
            title=f"{run.condition} · seed {run.seed} · {args.target // 1_000_000}M",
        )
        off_diagonal = ~np.eye(attempts.shape[0], dtype=bool)
        weights = attempts[off_diagonal]
        positive = weights[weights > 0]
        probability = positive / positive.sum() if positive.size else np.array([])
        rows.append(
            {
                "run_name": run.name,
                "condition": run.condition,
                "seed": run.seed,
                "odor": run.factors["odor"],
                "goal_set": run.factors["goal_set"],
                "target_frames": args.target,
                "attempted_directed_edges": int(positive.size),
                "effective_attempted_edges": float(np.exp(-np.sum(probability * np.log(probability))))
                if positive.size else 0.0,
                "reliable_edges": reliable_edges,
                "reachable_pair_fraction": reachable_fraction,
                "prospective_attempts": float(weights.sum()),
                "prospective_successes": float(successes[off_diagonal].sum()),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "graph_snapshot_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rendered graph outcomes for {len(rows)} declared runs")


if __name__ == "__main__":
    main()
