"""Rebuild the COSYNE run catalog from immutable study and submission records.

This script indexes attempts, including failed attempts and their replacements.
Scientific factors come from StudySpec metadata; job IDs and output locations
come from the submitted manifests. It never scans or copies training data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDIES = ROOT / "hpc_runs/studies"
OLD_RESULTS = ROOT / "06_experiments/reward_credit/results/corrected_core_hit_only_20261009"
NEW_RESULTS = ROOT / "06_experiments/reward_credit/results/cosyne27_staged_push_pull_20261010"
HIGH_DISCOUNT_RESULTS = ROOT / "06_experiments/reward_credit/results/cosyne27_high_discount_hit_timing_20261012"

# Cohort-level sources only. Individual identities and factors are generated.
COHORTS = (
    ("old_ablation", "corrected_core_hit_only_first_20261009", OLD_RESULTS / "first_wave_submitted_jobs.tsv"),
    ("old_ablation", "corrected_core_encoder_worker_later_20261009", OLD_RESULTS / "later_wave_submitted_jobs.tsv"),
    ("old_replacement", "corrected_core_c15_hitonly_recovery_production_20261010", None),
    ("old_replacement", "corrected_core_c15_zero_constant_recovery_production_20261010", None),
    ("cosyne", "cosyne27_c05_pushpull_s3_20261010", NEW_RESULTS / "c05_s3_jobs.tsv"),
    ("cosyne", "cosyne27_c05_pushpull_s2_20261010", NEW_RESULTS / "c05_s2_jobs.tsv"),
    ("cosyne", "cosyne27_c01_temporal_s3_20261010", NEW_RESULTS / "c01_temporal_s3_jobs.tsv"),
    ("cosyne", "cosyne27_c01_constant_s3_20261010", NEW_RESULTS / "c01_constant_s3_jobs.tsv"),
    ("cosyne", "cosyne27_c01_encoder_s2_20261010", NEW_RESULTS / "c01_encoder_s2_jobs.tsv"),
    ("cosyne", "cosyne27_c15_encoder_s3_20261010", NEW_RESULTS / "c15_encoder_s3_jobs.tsv"),
    ("cosyne", "cosyne27_c15_encoder_s2_20261010", NEW_RESULTS / "c15_encoder_s2_jobs.tsv"),
    ("cosyne_replacement", "cosyne27_c15_constant_recovery_20261011", NEW_RESULTS / "c15_constant_recovery_20261011_jobs.tsv"),
    ("cosyne_new", "cosyne27_c01_no_credit_s5_20261012", NEW_RESULTS / "c01_no_credit_s5_20261012_jobs.tsv"),
    ("cosyne_new", "cosyne27_c15_constant_hit_s5_20261012", NEW_RESULTS / "c15_constant_hit_s5_20261012_jobs.tsv"),
    ("cosyne_long_credit", "cosyne27_c05_long_credit_worker_s5_20261012", HIGH_DISCOUNT_RESULTS / "production_submitted_jobs.tsv"),
)

RECOVERY_JOBS = {
    "corrected_core_c15_hitonly_recovery_production_20261010": "8324628",
    "corrected_core_c15_zero_constant_recovery_production_20261010": "8324629",
}

FIELDNAMES = (
    "cohort", "study_id", "study_path", "schema", "workflow_version", "study_sha256",
    "run_name", "condition", "family", "seed", "encoder_credit", "ppo_dg_gradient",
    "worker_bonus", "job_id", "attempt_status", "replaces_job_id", "replaces_run_name",
    "submitted_manifest", "train_root", "run_dir", "source_base_commit",
    "source_patch_sha256", "recovery_patch_sha256", "report_path",
)

ACTIVE_TRAIN_DIR = "/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir"
HISTORICAL_TRAIN_DIR = "/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir"


def relative(path: Path | None) -> str:
    return str(path.relative_to(ROOT)) if path else ""


def manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    rows = {record["experiment"]: record for record in records}
    if len(rows) != len(records):
        raise ValueError(f"duplicate experiment in {path}")
    return rows


def report_for(cohort: str) -> str:
    reports = {
        "historical": "06_experiments/corrected_core_reevaluation_20260901.md",
        "old_ablation": "06_experiments/reward_credit/corrected_core_encoder_worker_ablation_20261010.md",
        "old_replacement": "06_experiments/reward_credit/corrected_core_c15_failed_run_recovery_20261010.md",
        "cosyne": "06_experiments/reward_credit/cosyne27_staged_push_pull_submission_20261010.md",
        "cosyne_replacement": "06_experiments/reward_credit/cosyne27_staged_push_pull_submission_20261010.md",
        "cosyne_new": "06_experiments/reward_credit/cosyne27_missing_training_20261012.md",
        "cosyne_long_credit": "06_experiments/reward_credit/cosyne27_high_discount_hit_timing_20261012.md",
    }
    return reports[cohort]


def historical_rows() -> list[dict[str, str]]:
    provenance_path = OLD_RESULTS / "historical_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    manifest_path = OLD_RESULTS / "historical_jobs.tsv"
    manifest = {record["job_id"]: record for record in manifest_rows(manifest_path).values()}
    rows = []
    for ref in provenance["references"]:
        job = manifest[ref["job_id"]]
        rows.append(dict(
            cohort="historical", study_id="intrmotiv_corrected_core_reevaluation_20260901",
            study_path=relative(provenance_path), schema="historical-manifest",
            workflow_version="pre-StudySpec", study_sha256=provenance["manifest_sha256"],
            run_name=job["experiment"].removeprefix("00_"), condition=ref["family"],
            family=ref["family"], seed=str(ref["seed"]), encoder_credit="temporal",
            ppo_dg_gradient="stop", worker_bonus="temporal" if ref["family"] != "C01" else "inapplicable",
            job_id=ref["job_id"], attempt_status="verified_100m_historical",
            replaces_job_id="", replaces_run_name="", submitted_manifest=relative(manifest_path),
            train_root=job["train_root"],
            run_dir=f"{HISTORICAL_TRAIN_DIR}/{job['train_root']}/{job['experiment']}",
            source_base_commit="unavailable", source_patch_sha256="unavailable",
            recovery_patch_sha256="", report_path=report_for("historical"),
        ))
    if len(rows) != 9:
        raise ValueError("historical reference set changed")
    return rows


def study_rows() -> list[dict[str, str]]:
    rows = []
    for cohort, study_name, manifest_path in COHORTS:
        study_path = STUDIES / f"{study_name}.study.json"
        spec = load_study(study_path)
        manifest = manifest_rows(manifest_path) if manifest_path else {}
        expected = set()
        for run in spec.expand_runs():
            experiment = f"00_{run.name}"
            expected.add(experiment)
            job = manifest.get(experiment)
            if manifest_path and job is None:
                raise ValueError(f"missing submitted job for {experiment}")
            metadata = run.metadata
            status = "verified_100m"
            if cohort in ("cosyne_new", "cosyne_long_credit"):
                status = "running_2026-10-12"
            elif cohort == "old_replacement":
                status = "verified_100m_2026-10-11"
            replaced_job = metadata.get("recovery_original_job", "")
            if "recovery_original_jobs" in metadata:
                seed_jobs = dict(zip(
                    (int(value) for value in str(metadata["recovery_original_seeds"]).split(",")),
                    str(metadata["recovery_original_jobs"]).split(","), strict=True,
                ))
                replaced_job = seed_jobs[run.seed]
            train_root = job["train_root"] if job else f"{spec.batch_name}/{run.name}_"
            rows.append(dict(
                cohort=cohort, study_id=spec.study_id, study_path=relative(study_path),
                schema=spec.raw["schema"], workflow_version=spec.declared_workflow_version,
                study_sha256=spec.fingerprint, run_name=run.name, condition=run.condition,
                family=str(metadata.get("family", "")), seed=str(run.seed),
                encoder_credit=str(metadata.get("encoder_credit", metadata.get("credit_encoder_credit", metadata.get("encoder_credit_encoder_credit", "")))),
                ppo_dg_gradient=str(metadata.get("ppo_dg_gradient", metadata.get("credit_ppo_dg_gradient", metadata.get("encoder_credit_ppo_dg_gradient", "")))),
                worker_bonus=("inapplicable" if metadata.get("family") == "C01" else
                              str(metadata.get("worker_bonus", metadata.get("worker_bonus_worker_bonus", "")))),
                job_id=job["job_id"] if job else RECOVERY_JOBS[study_name],
                attempt_status=status,
                replaces_job_id=str(replaced_job),
                replaces_run_name=str(metadata.get("recovery_original_run", "")),
                submitted_manifest=relative(manifest_path),
                train_root=train_root, run_dir=f"{ACTIVE_TRAIN_DIR}/{train_root}/00_{run.name}",
                source_base_commit=str(metadata.get("source_base_commit", "")),
                source_patch_sha256=str(metadata.get("source_patch_sha256", "")),
                recovery_patch_sha256=str(metadata.get("recovery_patch_sha256", "")),
                report_path=report_for(cohort),
            ))
        if manifest and set(manifest) != expected:
            raise ValueError(f"manifest has unexpected jobs: {set(manifest) - expected}")
    by_job = {row["job_id"]: row for row in rows}
    for replacement in (row for row in rows if row["replaces_job_id"]):
        original = by_job[replacement["replaces_job_id"]]
        original["attempt_status"] = "failed_replaced"
        replacement["replaces_run_name"] = original["run_name"]
    return rows


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = historical_rows() + study_rows()
    if len(rows) != 129 or len({row["job_id"] for row in rows}) != 129:
        raise ValueError("expected 129 uniquely submitted training attempts")
    write_csv(HERE / "training_runs.csv", FIELDNAMES, rows)

    with (STUDIES / "behavior_field_expression_20261007.tsv").open(newline="") as stream:
        evaluations = list(csv.DictReader(stream, delimiter="\t"))
    historical = {(row["family"], row["seed"]): row for row in rows if row["cohort"] == "historical"}
    indexed = []
    for evaluation in evaluations:
        parent = historical[(evaluation["condition"], evaluation["seed"])]
        indexed.append({**evaluation, "training_job_id": parent["job_id"],
                        "training_run_name": parent["run_name"],
                        "source_manifest": "hpc_runs/studies/behavior_field_expression_20261007.tsv"})
    if len(indexed) != 54 or len({row["label"] for row in indexed}) != 54:
        raise ValueError("expected 54 unique frozen-policy evaluations")
    write_csv(HERE / "frozen_policy_evaluations.csv", tuple(indexed[0]), indexed)
    print(f"Indexed {len(rows)} training attempts and {len(indexed)} frozen-policy evaluations")


if __name__ == "__main__":
    main()
