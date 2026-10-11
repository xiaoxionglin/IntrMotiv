"""Summarize frozen-policy evidence without mixing sources or evaluation repeats."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "06_experiments/results/cosyne_2027_behavior_secondary"
TRAINING = ROOT / "06_experiments/cosyne_2027/training_runs.csv"
EVALUATION_SOURCES = {
    "historical": RESULTS / "historical_pair_analysis/paired_evaluation_repeats.csv",
    "fresh": RESULTS / "fresh_pair_analysis/paired_evaluation_repeats.csv",
}
REPEAT_METRICS = (
    "information_difference", "bits_per_step_difference",
    "mean_activation_difference", "native_split_half_difference",
    "common_split_half_difference", "information_shared_bins",
    "information_eligible_units", "common_split_half_bins",
    "common_split_half_units",
    "common_split_half_bins_25", "common_split_half_units_25",
    "common_split_half_difference_25", "common_split_half_bins_50",
    "common_split_half_units_50", "common_split_half_difference_50",
)


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, entries: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


def chosen_models() -> list[dict[str, str]]:
    attempts = [row for row in rows(TRAINING)
                if row["cohort"] in {"historical", "cosyne", "cosyne_replacement"}]
    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in attempts:
        cohort = "historical" if row["cohort"] == "historical" else "fresh"
        key = (cohort, row["family"], row["seed"],
               row["encoder_credit"], row["worker_bonus"])
        if key not in by_key or row["cohort"] == "cosyne_replacement":
            by_key[key] = row
    if len(by_key) != 49:
        raise ValueError(f"Expected 9 historical and 40 fresh models, found {len(by_key)}")
    return [by_key[key] for key in sorted(by_key)]


def complete_mean(repeats: list[dict[str, str]], metric: str) -> float | None:
    if {int(row["eval_seed"]) for row in repeats} != {51_000, 52_000} or len(repeats) != 2:
        return None
    values = [row.get(metric, "") for row in repeats]
    if any(value in ("", "None") for value in values):
        return None
    numeric = np.asarray([float(value) for value in values])
    return float(numeric.mean()) if np.isfinite(numeric).all() else None


def main() -> None:
    repeat_rows = [row for source in EVALUATION_SOURCES.values() for row in rows(source)
                   if row["layer"] == "decoder_1"]
    repeat_key = lambda row: (row["cohort"], row["condition"], row["training_seed"])
    by_repeat: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in repeat_rows:
        by_repeat.setdefault(repeat_key(row), []).append(row)
    records: list[dict[str, object]] = []
    for model in chosen_models():
        cohort = "historical" if model["cohort"] == "historical" else "fresh"
        group = by_repeat.get((cohort, model["condition"], model["seed"]), [])
        record: dict[str, object] = {
            "cohort": cohort, "condition": model["condition"],
            "family": model["family"], "encoder_credit": model["encoder_credit"],
            "worker_bonus": model["worker_bonus"], "training_seed": model["seed"],
            "training_job_id": model["job_id"], "training_status": model["attempt_status"],
            "study_id": model["study_id"], "study_schema": model["schema"],
            "workflow_version": model["workflow_version"],
            "study_sha256": model["study_sha256"],
            "source_base_commit": model["source_base_commit"],
            "source_patch_sha256": model["source_patch_sha256"],
            "recovery_patch_sha256": model["recovery_patch_sha256"],
            "completed_eval_repeats": len({row["eval_seed"] for row in group}),
        }
        for metric in REPEAT_METRICS:
            record[metric] = complete_mean(group, metric)
        records.append(record)
    write(RESULTS / "all_training_seeds.csv", records, tuple(records[0]))
    summaries = []
    factors = sorted({(row["cohort"], row["condition"], row["family"],
                       row["encoder_credit"], row["worker_bonus"]) for row in records})
    for cohort, condition, family, credit, worker in factors:
        group = [row for row in records if (row["cohort"], row["condition"], row["family"],
                                               row["encoder_credit"], row["worker_bonus"]) ==
                 (cohort, condition, family, credit, worker)]
        for metric in ("information_difference", "native_split_half_difference",
                       "common_split_half_difference"):
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summaries.append({"cohort": cohort, "condition": condition, "family": family,
                              "encoder_credit": credit, "worker_bonus": worker,
                              "metric": metric, "planned_training_seeds": len(group),
                              "complete_training_seeds": len(values),
                              "positive_training_seeds": sum(value > 0 for value in values),
                              "mean_difference": float(np.mean(values)) if values else None})
    write(RESULTS / "factor_summary.csv", summaries, tuple(summaries[0]))
    (RESULTS / "factor_summary_manifest.json").write_text(json.dumps({
        "schema": "cosyne27_behavior_training_seeds_v1",
        "replication": "two paired evaluation seeds averaged within each independent training seed",
        "source_policy": "historical and fresh cohorts are separate; no pooled effect estimate",
        "rows": len(records),
        "workflow_versions": sorted({str(row["workflow_version"]) for row in records}),
        "study_provenance": sorted({(str(row["study_schema"]), str(row["workflow_version"]),
                                     str(row["study_sha256"])) for row in records}),
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (TRAINING, *EVALUATION_SOURCES.values()) if path.exists()},
    }, indent=2) + "\n")
    print(f"summarized {len(records)} planned training models and {len(repeat_rows)} decoder-1 evaluation pairs")


if __name__ == "__main__":
    main()
