"""Cache COSYNE online spatial histories with explicit training-run provenance.

The per-run W&B history is queried once. Rows are joined to the immutable run
catalog using StudySpec identity, seed, condition, and experiment name; failed
attempts are never silently substituted for replacement runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import wandb


PROJECT = "xiaoxionglin-bernstein-center-freiburg/SF_IntrMotiv_PushPullAblation"
PREFIX = "intrmotiv/online/"
METRICS = (
    "place_field/active_unit_mean_spatial_information",
    "place_field/active_only_map_cosine",
    "place_field/active_unit_fraction",
    "place_field/silent_unit_fraction",
    "place_field/unique_active_peak_bins",
    "place_field/mono_field_unit_fraction",
    "place_field/median_dominant_peak_nearest_neighbor_distance",
    "place_field/mean_primary_secondary_peak_distance",
    "place_field/visited_cell_fraction",
    "place_field/valid_sample_count",
    "trajectory/path_efficiency",
    "trajectory/mean_physical_step_distance",
    "trajectory/stationary_step_fraction",
    "trajectory/mean_absolute_circular_yaw_change",
)
HISTORY_KEYS = ["train/env_steps", *(PREFIX + name for name in METRICS)]


def selected_attempts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row
        for row in rows
        if row["cohort"] in {"cosyne", "cosyne_replacement"}
        and row["attempt_status"].startswith("verified_100m")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    attempts = selected_attempts(args.catalog)
    by_study: dict[str, list[dict[str, str]]] = {}
    for row in attempts:
        by_study.setdefault(row["study_id"], []).append(row)

    api = wandb.Api(timeout=45)
    manifest: list[dict[str, str | int]] = []
    for study_id, study_rows in by_study.items():
        indexed = {
            (row["run_name"], row["condition"], int(row["seed"])): row
            for row in study_rows
        }
        seen: set[tuple[str, str, int]] = set()
        runs = api.runs(PROJECT, filters={"config.study_id": study_id}, per_page=100)
        for run in runs:
            key = (
                str(run.config.get("experiment", "")).removeprefix("00_"),
                str(run.config.get("study_condition", "")),
                int(run.config.get("seed", -1)),
            )
            if key not in indexed:
                continue
            if key in seen:
                raise ValueError(f"Duplicate W&B run for {key}")
            seen.add(key)
            catalog_row = indexed[key]
            history_path = args.output / f"{catalog_row['run_name']}.csv"
            with history_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=HISTORY_KEYS)
                writer.writeheader()
                count = 0
                for history in run.scan_history(keys=HISTORY_KEYS, page_size=1000):
                    writer.writerow({name: history.get(name) for name in HISTORY_KEYS})
                    count += 1
            manifest.append(
                {
                    "study_id": study_id,
                    "study_sha256": catalog_row["study_sha256"],
                    "run_name": catalog_row["run_name"],
                    "condition": catalog_row["condition"],
                    "family": catalog_row["family"],
                    "seed": int(catalog_row["seed"]),
                    "encoder_credit": catalog_row["encoder_credit"],
                    "worker_bonus": catalog_row["worker_bonus"],
                    "source_base_commit": catalog_row["source_base_commit"],
                    "recovery_patch_sha256": catalog_row["recovery_patch_sha256"],
                    "wandb_path": f"{PROJECT}/{run.id}",
                    "wandb_state": run.state,
                    "history_rows": count,
                    "history_file": history_path.name,
                }
            )
            print(study_id, catalog_row["run_name"], count, flush=True)
        missing = set(indexed) - seen
        if missing:
            raise ValueError(f"Missing W&B runs in {study_id}: {sorted(missing)}")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
