"""Join canonical field manifests and select available C15 checkpoints.

The original C15 three-seed StudySpec cannot render while two failed original
runs are present: fallback checkpoint selection creates duplicate labels. This
adapter keeps the validated manifests for complete cohorts and uses the same
canonical checkpoint selector for completed C15 attempts only. A provenance
sidecar retains each row's original StudySpec fingerprint and source patch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import (
    checkpoint_frames,
    select_checkpoints,
)


COMPLETE_COHORTS = (
    "c01_temporal_s3", "c01_constant_s3", "c01_encoder_s2",
    "c05_pushpull_s3", "c05_pushpull_s2", "c15_encoder_s2",
)
FIELDS = (
    "condition", "family", "schedule", "feedback", "half_life", "seed",
    "target_frames", "checkpoint_frames", "checkpoint", "run_dir", "label_suffix",
)
AGES = (5_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("analysis_root", type=Path)
    args = parser.parse_args()
    with args.catalog.open(newline="") as stream:
        catalog = list(csv.DictReader(stream))
    valid = [row for row in catalog if row["cohort"] in {"cosyne", "cosyne_replacement"}
             and row["attempt_status"].startswith("verified_100m")]
    by_cell = {(row["condition"], row["seed"]): row for row in valid}
    if len(by_cell) != len(valid):
        raise ValueError("More than one verified attempt for a condition and seed")

    all_rows: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []
    for cohort in COMPLETE_COHORTS:
        folder = args.analysis_root / cohort
        manifest = json.loads((folder / "study_manifest.json").read_text())
        for row in read_tsv(folder / "analysis_manifest.tsv"):
            key = (row["condition"], row["seed"])
            attempt = by_cell[key]
            all_rows.append(row)
            sources.append({
                "label_suffix": row["label_suffix"], "run_name": attempt["run_name"],
                "study_id": attempt["study_id"],
                "study_sha256": attempt["study_sha256"],
                "source_base_commit": attempt["source_base_commit"],
                "recovery_patch_sha256": attempt["recovery_patch_sha256"],
            })
            if attempt["study_sha256"] != manifest["study_sha256"]:
                raise ValueError(f"Study fingerprint mismatch for {key}")

    # Original C15 three-seed source plus any verified replacements. The
    # StudySpec still determines these identities via the run catalog.
    c15_three_seed = [row for row in valid if row["family"] == "C15"
                       and row["seed"] in {"8", "99", "123"}]
    for attempt in c15_three_seed:
        run_dir = Path(attempt["run_dir"])
        targets = AGES if attempt["seed"] == "99" else (100_000_000,)
        for target, checkpoint in select_checkpoints(run_dir, targets):
            actual = checkpoint_frames(checkpoint)
            if abs(actual - target) > 200_000:
                raise ValueError(f"Missing intended checkpoint for {attempt['run_name']}: {target} -> {actual}")
            row = {name: "" for name in FIELDS}
            row.update({
                "condition": attempt["condition"], "family": "C15",
                "seed": attempt["seed"], "target_frames": str(target),
                "checkpoint_frames": str(actual), "checkpoint": str(checkpoint),
                "run_dir": str(run_dir),
                "label_suffix": f"{attempt['condition']}__s{attempt['seed']}__f{actual}",
            })
            all_rows.append(row)
            sources.append({
                "label_suffix": row["label_suffix"], "run_name": attempt["run_name"],
                "study_id": attempt["study_id"],
                "study_sha256": attempt["study_sha256"],
                "source_base_commit": attempt["source_base_commit"],
                "recovery_patch_sha256": attempt["recovery_patch_sha256"],
            })

    labels = [row["label_suffix"] for row in all_rows]
    if len(labels) != len(set(labels)):
        raise ValueError("Field manifest has duplicate labels")
    terminal = [row for row in all_rows if int(row["target_frames"]) == 100_000_000]
    if len(terminal) != len(valid):
        raise ValueError(f"Expected {len(valid)} valid terminal rows, found {len(terminal)}")
    all_rows.sort(key=lambda row: (row["family"], row["condition"], int(row["seed"]), int(row["target_frames"])))
    terminal.sort(key=lambda row: (row["family"], row["condition"], int(row["seed"])))
    write_tsv(args.analysis_root / "available_analysis_manifest.tsv", all_rows)
    write_tsv(args.analysis_root / "available_terminal_manifest.tsv", terminal)
    source_by_label = {row["label_suffix"]: row for row in sources}
    c15_three_seed_rows = [
        row for row in all_rows
        if row["family"] == "C15"
        and int(row["seed"]) in {8, 99, 123}
        and source_by_label[row["label_suffix"]]["study_id"]
        in {"cosyne27_c15_encoder_s3_20261010", "cosyne27_c15_constant_recovery_20261011"}
    ]
    write_tsv(args.analysis_root / "c15_available_manifest.tsv", c15_three_seed_rows)
    write_tsv(
        args.analysis_root / "c15_trajectory_manifest.tsv",
        [row for row in c15_three_seed_rows if int(row["seed"]) == 99],
    )
    with (args.analysis_root / "manifest_sources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sources[0]))
        writer.writeheader()
        writer.writerows(sorted(sources, key=lambda row: row["label_suffix"]))
    print(f"Available field rows: {len(all_rows)}; terminal rows: {len(terminal)}; C15 three-seed rows: {len(c15_three_seed_rows)}")


if __name__ == "__main__":
    main()
