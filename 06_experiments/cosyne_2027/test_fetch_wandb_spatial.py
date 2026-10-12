"""Focused provenance checks for the local W&B history cache."""

import csv
import json
from pathlib import Path

from fetch_wandb_spatial import HISTORY_KEYS, reusable_cache


def test_cache_reuses_matching_history_and_rejects_changed_source(tmp_path: Path) -> None:
    attempt = {
        "study_id": "recovery",
        "study_sha256": "study-a",
        "run_name": "C15_S99",
        "condition": "constant",
        "family": "C15",
        "seed": "99",
        "encoder_credit": "constant",
        "worker_bonus": "temporal",
        "source_base_commit": "source-a",
        "recovery_patch_sha256": "patch-a",
    }
    history = tmp_path / "C15_S99.csv"
    with history.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HISTORY_KEYS)
        writer.writerow([100_000_000, *([0.1] * (len(HISTORY_KEYS) - 1))])
    record = {**attempt, "seed": 99, "history_file": history.name, "history_rows": 1}
    (tmp_path / "manifest.json").write_text(json.dumps([record]))

    reused, missing = reusable_cache(tmp_path, [attempt])
    assert [item["run_name"] for item in reused] == ["C15_S99"]
    assert missing == []

    changed = {**attempt, "recovery_patch_sha256": "patch-b"}
    reused, missing = reusable_cache(tmp_path, [changed])
    assert reused == []
    assert missing == [changed]


def test_cache_rejects_truncated_history(tmp_path: Path) -> None:
    attempt = {
        "study_id": "recovery", "study_sha256": "study-a", "run_name": "C15_S8",
        "condition": "constant", "family": "C15", "seed": "8",
        "encoder_credit": "constant", "worker_bonus": "temporal",
        "source_base_commit": "source-a", "recovery_patch_sha256": "patch-a",
    }
    (tmp_path / "C15_S8.csv").write_text(",".join(HISTORY_KEYS) + "\n")
    record = {**attempt, "seed": 8, "history_file": "C15_S8.csv", "history_rows": 1}
    (tmp_path / "manifest.json").write_text(json.dumps([record]))

    reused, missing = reusable_cache(tmp_path, [attempt])
    assert reused == []
    assert missing == [attempt]
