"""Aggregate frozen matched-command DG detector hits by training seed.

The evaluator calls its scalar `paired_arrival_lift`, but `hit` is a DG
exclusive target event. This script names it detector-hit lift and retains
the physical endpoints without claiming an independently mapped arrival.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_factors import factors


COHORTS = ("c05_pushpull_s3", "c05_pushpull_s2", "c15_encoder_s2", "c15_available")
HORIZONS = (64, 128, 256, 900)


def trial_metrics(trials: pd.DataFrame) -> dict[str, float]:
    if not trials.exact_start_verified.all():
        raise ValueError("A matched-command start did not verify")
    keys = ["source", "repeat", "target"]
    group_sizes = trials.groupby(keys).size()
    if not group_sizes.eq(3).all():
        raise ValueError("Expected three executed commands per source/repeat/target")
    result: dict[str, float] = {"trial_rows": len(trials)}
    endpoint_distances = []
    for _, start in trials.groupby(["source", "repeat"]):
        commands = start.drop_duplicates("command").sort_values("command")
        endpoints = np.stack([np.asarray(json.loads(value), dtype=float)[:2] for value in commands.endpoint])
        for index in range(len(endpoints)):
            for other in range(index + 1, len(endpoints)):
                endpoint_distances.append(float(np.linalg.norm(endpoints[index] - endpoints[other])))
    result["mean_command_endpoint_separation_xy"] = float(np.mean(endpoint_distances)) if endpoint_distances else np.nan
    for horizon in HORIZONS:
        column = f"hit_{horizon}"
        censored = f"censored_{horizon}"
        result[f"commanded_hit_{horizon}"] = float(trials.loc[trials.commanded, column].mean())
        result[f"alternate_hit_{horizon}"] = float(trials.loc[~trials.commanded, column].mean())
        result[f"censored_fraction_{horizon}"] = float(trials[censored].mean())
        differences = []
        for _, group in trials.groupby(keys):
            command = group.loc[group.commanded, column]
            alternate = group.loc[~group.commanded, column]
            if len(command) != 1 or len(alternate) != 2:
                raise ValueError("Expected one commanded and two alternate trajectories")
            differences.append(float(command.iloc[0]) - float(alternate.mean()))
        result[f"paired_detector_hit_lift_{horizon}"] = float(np.mean(differences))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="NEMO2 analysis root with cohort intervention manifests and control/raw")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--use-merged-shards", action="store_true",
                        help="Include audited control_merged/raw rows when the ordinary output is absent")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for cohort in COHORTS:
        manifest_path = args.root / cohort / "intervention_manifest.tsv"
        if not manifest_path.exists():
            continue
        study = json.loads((args.root / cohort / "study_manifest.json").read_text())
        manifest = pd.read_csv(manifest_path, sep="\t")
        for row_index, record in enumerate(manifest.itertuples(index=False)):
            raw_dir = args.root / cohort / "control" / "raw" / record.label_suffix
            merged_dir = args.root / cohort / "control_merged" / "raw" / record.label_suffix
            if args.use_merged_shards and (merged_dir / "intervention_summary.json").exists():
                if (raw_dir / "intervention_summary.json").exists():
                    raise ValueError(f"Both ordinary and merged output exist for {record.label_suffix}")
                raw_dir = merged_dir
            trial_path = raw_dir / "intervention_trials.csv"
            summary_path = raw_dir / "intervention_summary.json"
            if not trial_path.exists() or not summary_path.exists():
                missing.append((cohort, record.label_suffix))
                continue
            summary = json.loads(summary_path.read_text())
            expected = {"schema": study["schema"], "workflow_version": study["workflow_version"],
                        "study_id": study["study_id"], "study_sha256": study["study_sha256"],
                        "condition": record.condition, "seed": record.seed,
                        "checkpoint_frames": record.checkpoint_frames,
                        "checkpoint": record.checkpoint, "manifest_row": row_index}
            for key, value in expected.items():
                if summary.get(key) != value:
                    raise ValueError(f"{key} mismatch for {record.label_suffix}")
            trials = pd.read_csv(trial_path)
            family, encoder, worker = factors(record.condition)
            row = {"cohort": cohort, "condition": record.condition, "family": family,
                   "encoder_credit": encoder, "worker_bonus": worker, "seed": record.seed,
                   "target_frames": record.target_frames, "checkpoint_frames": record.checkpoint_frames,
                   "label_suffix": record.label_suffix, "study_id": summary["study_id"],
                   "study_sha256": summary["study_sha256"], "schema": summary["schema"],
                   "workflow_version": summary["workflow_version"],
                   "output_kind": "source_shard_merge" if raw_dir == merged_dir else "ordinary",
                   "sources_evaluated": summary["sources_evaluated"],
                   "supported_sources": summary["supported_sources"],
                   "source_panel_coverage": summary["source_panel_coverage"],
                   "starts_evaluated": summary["starts_evaluated"],
                   "missing_reasons": json.dumps(summary["missing"]),
                   "mean_initial_action_total_variation": summary["mean_initial_action_total_variation"],
                   **trial_metrics(trials)}
            rows.append(row)
    if missing and not args.allow_incomplete:
        raise ValueError(f"Missing {len(missing)} intervention outputs; first: {missing[:3]}")
    pd.DataFrame(missing, columns=["cohort", "label_suffix"]).to_csv(args.output_dir / "missing_control_rows.csv", index=False)
    if not rows:
        raise ValueError("No completed intervention rows")
    data = pd.DataFrame(rows)
    if data.duplicated(["condition", "seed", "target_frames"]).any():
        raise ValueError("Duplicate condition/seed/checkpoint intervention")
    data.to_csv(args.output_dir / "control_seed_checkpoint_metrics.csv", index=False)
    metrics = [column for column in data if column.startswith(("paired_detector_hit_lift_", "commanded_hit_", "alternate_hit_", "censored_fraction_"))]
    keys = ["family", "encoder_credit", "worker_bonus", "target_frames"]
    aggregate = data.groupby(keys)[metrics].agg(["count", "mean", "std", "sem"])
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(args.output_dir / "control_condition_summary.csv", index=False)
    print(f"Completed {len(data)} intervention rows; missing {len(missing)}")


if __name__ == "__main__":
    main()
