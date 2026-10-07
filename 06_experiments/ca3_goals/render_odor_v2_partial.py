"""Render only available seed-99 milestone rows from the immutable v2 StudySpec.

This temporary adapter uses the canonical StudySpec expansion, run discovery,
checkpoint selector, and manifest writer. It does not modify the study file.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.discovery import discover_run_directories
from hpc_runs.intrmotiv_study.telemetry import (
    CheckpointRecord,
    build_place_field_manifests,
    write_manifest,
)
from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import (
    checkpoint_frames,
    select_checkpoints,
)


def main() -> None:
    spec_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    targets = [int(value) for value in sys.argv[3:]]
    if not targets or targets != sorted(set(targets)):
        raise ValueError("Targets must be sorted, unique, and nonempty")

    study = load_study(spec_path)
    if not set(targets).issubset(set(study.telemetry["target_frames"])):
        raise ValueError("Targets must be declared by the immutable StudySpec")
    workspace = Path(study.workspace_root).resolve()
    output_dir.relative_to(workspace)

    # Only the trajectory seed is due before the terminal checkpoint. The
    # intervention contract remains in the immutable source StudySpec.
    partial_telemetry = {
        **study.telemetry,
        "target_frames": targets,
        "terminal_seeds": [],
        "intervention": None,
    }
    partial_study = replace(study, telemetry=partial_telemetry)
    run_directories = discover_run_directories(study, Path(study.output_root))
    trajectory_seed = int(study.telemetry["trajectory_seed"])

    inventory = []
    for run in study.expand_runs():
        if run.seed != trajectory_seed:
            continue
        run_dir = run_directories[run.name]
        for target, checkpoint in select_checkpoints(run_dir, target_frames=targets):
            actual = checkpoint_frames(checkpoint)
            if abs(actual - target) > target // 100:
                raise ValueError(f"No checkpoint within 1% of {target}: {run.name} selected {actual}")
            inventory.append(CheckpointRecord(run.name, target, actual, checkpoint, run_dir))

    rows, trajectory = build_place_field_manifests(partial_study, inventory)
    expected = sum(run.seed == trajectory_seed for run in study.expand_runs()) * len(targets)
    if len(rows) != expected or rows != trajectory:
        raise ValueError(f"Expected {expected} trajectory rows, got {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(output_dir / "analysis_manifest.tsv", rows)
    (output_dir / "study_manifest_partial.json").write_text(
        json.dumps(
            {
                **study.provenance(),
                "source_study": str(spec_path),
                "subset_target_frames": targets,
                "trajectory_seed": trajectory_seed,
                "analysis_rows": len(rows),
                "reason": "Declared milestones available before the full 75M telemetry contract can render",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Rendered {len(rows)} canonical manifest rows for targets {targets} in {output_dir}")


if __name__ == "__main__":
    main()
