"""Analyze completed own/uniform traces with training-seed replication.

Incomplete evaluation pairs remain explicit in coverage.csv. Raw traces and
all intermediate arrays stay in the active NEMO2 workspace; only small tables
are intended for the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from hpc_runs.behavior_field_analysis import read_trace
from hpc_runs.behavior_field_metrics import (
    LAYERS, comparison_details, paired_information,
    paired_split_half_reliability, split_half_reliability,
)


def csv_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_rows(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest: Path, raw_root: Path, output: Path, cohort: str,
            provenance: Path | None = None) -> dict[str, object]:
    entries = csv_rows(manifest, delimiter="\t")
    meta = {row["label"]: row for row in csv_rows(provenance)} if provenance else {}
    groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for entry in entries:
        if entry["policy"] not in ("own", "uniform"):
            continue
        key = entry["condition"], entry["seed"], entry["eval_seed"]
        if entry["policy"] in groups.setdefault(key, {}):
            raise ValueError(f"Duplicate evaluation arm: {key}")
        groups[key][entry["policy"]] = entry
    output.mkdir(parents=True, exist_ok=True)
    coverage: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for (condition, seed, eval_seed), arms in sorted(groups.items()):
        if set(arms) != {"own", "uniform"}:
            raise ValueError(f"Manifest pair incomplete: {(condition, seed, eval_seed)}")
        folders = {policy: raw_root / row["label"] for policy, row in arms.items()}
        complete = all((folder / "trace.npz").is_file() and
                       (folder / "metadata.json").is_file() for folder in folders.values())
        coverage.append({"cohort": cohort, "condition": condition, "training_seed": seed,
                         "eval_seed": eval_seed, "own_complete": (folders["own"] / "trace.npz").is_file(),
                         "uniform_complete": (folders["uniform"] / "trace.npz").is_file(),
                         "paired_complete": complete})
        if not complete:
            continue
        own = read_trace(folders["own"], "own")
        uniform = read_trace(folders["uniform"], "uniform")
        if meta and (meta[arms["own"]["label"]]["study_sha256"] !=
                     meta[arms["uniform"]["label"]]["study_sha256"]):
            raise ValueError("Own and uniform arms have different StudySpecs")
        for layer in LAYERS:
            info = paired_information(comparison_details(own["pose"], own[layer]),
                                      comparison_details(uniform["pose"], uniform[layer]))
            native_own = split_half_reliability(own["pose"], own[layer], own["done"])
            native_uniform = split_half_reliability(uniform["pose"], uniform[layer],
                                                    uniform["done"])
            common = paired_split_half_reliability(own, uniform, layer)
            stricter = ({cutoff: paired_split_half_reliability(own, uniform, layer, cutoff)
                         for cutoff in (25, 50)} if layer == "decoder_1" else {})
            source = meta.get(arms["own"]["label"], {})
            comparisons.append({
                "cohort": cohort, "condition": condition, "family": source.get("family", condition),
                "encoder_credit": source.get("encoder_credit", "temporal"),
                "worker_bonus": source.get("worker_bonus", "inapplicable" if condition == "C01" else "temporal"),
                "training_seed": seed, "eval_seed": eval_seed, "layer": layer,
                "training_job_id": source.get("training_job_id", ""),
                "study_id": source.get("study_id", "intrmotiv_corrected_core_reevaluation_20260901"),
                "study_schema": source.get("study_schema", "historical-manifest"),
                "workflow_version": source.get("workflow_version", "pre-StudySpec"),
                "study_sha256": source.get("study_sha256", ""),
                "source_base_commit": source.get("source_base_commit", "unavailable"),
                "checkpoint_frames": arms["own"]["checkpoint_frames"],
                "information_supported": info["supported"],
                "information_shared_bins": info["shared_bins"],
                "information_eligible_units": info["eligible_units"],
                "own_bits_per_activation": info["own_bits_per_activation"],
                "uniform_bits_per_activation": info["random_bits_per_activation"],
                "information_difference": info["difference"],
                "bits_per_step_difference": info["bits_per_step_difference"],
                "mean_activation_difference": info["mean_activation_difference"],
                "native_own_split_half": native_own,
                "native_uniform_split_half": native_uniform,
                "native_split_half_difference": (native_own - native_uniform
                                                 if native_own is not None and native_uniform is not None else None),
                "common_split_half_supported": common["supported"],
                "common_split_half_bins": common["shared_bins"],
                "common_split_half_units": common["eligible_units"],
                "own_half_bins": common.get("own_half_bins"),
                "uniform_half_bins": common.get("uniform_half_bins"),
                "common_own_split_half": common["own"],
                "common_uniform_split_half": common["uniform"],
                "common_split_half_difference": common["difference"],
                "common_split_half_bins_25": stricter.get(25, {}).get("shared_bins"),
                "common_split_half_units_25": stricter.get(25, {}).get("eligible_units"),
                "common_split_half_difference_25": stricter.get(25, {}).get("difference"),
                "common_split_half_bins_50": stricter.get(50, {}).get("shared_bins"),
                "common_split_half_units_50": stricter.get(50, {}).get("eligible_units"),
                "common_split_half_difference_50": stricter.get(50, {}).get("difference"),
            })
        print(f"analyzed {condition} training seed {seed} evaluation seed {eval_seed}", flush=True)
    write_rows(output / "coverage.csv", coverage,
               ("cohort", "condition", "training_seed", "eval_seed", "own_complete",
                "uniform_complete", "paired_complete"))
    if comparisons:
        write_rows(output / "paired_evaluation_repeats.csv", comparisons, tuple(comparisons[0]))
    summary = {"schema": "intrmotiv/behavior-field-pairs/v1", "cohort": cohort,
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "complete_evaluation_pairs": sum(row["paired_complete"] for row in coverage),
               "expected_evaluation_pairs": len(coverage),
               "common_support": "same bins with >=10 visits in each policy and each episode-disjoint half; identical full-run eligible units with finite correlations in both policies; decoder 1 also checked at 25 and 50 visits per half",
               "limitations": "spatial-bin weights are equal, but within-bin sample counts and temporal inputs remain policy dependent"}
    (output / "analysis_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.manifest, args.raw_root, args.output,
                             args.cohort, args.provenance)))


if __name__ == "__main__":
    main()
