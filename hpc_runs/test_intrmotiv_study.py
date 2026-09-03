from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from hpc_runs.graph_stabilized_recruitment_manifest import rows as legacy_rows
from hpc_runs.intrmotiv_study import SCHEMA_ID, SpecError, WORKFLOW_VERSION, load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts, summarize_records
from hpc_runs.intrmotiv_study.sample_factory import build_run_description
from hpc_runs.intrmotiv_study.submission import audit_submission
from hpc_runs.intrmotiv_study.telemetry import (
    MANIFEST_COLUMNS,
    CheckpointRecord,
    build_place_field_manifests,
)
from hpc_runs.intrmotiv_study.tensorboard import latest_at_or_before, mean_in_window


SPEC_PATH = (
    Path(__file__).with_name("studies")
    / "graph_stabilized_recruitment.study.json"
)


@dataclass(frozen=True)
class Event:
    step: int
    value: float
    wall_time: float = 0.0


class StudySpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)

    def test_real_factorial_study_expands_to_unique_runs(self):
        runs = self.study.expand_runs()
        self.assertEqual(self.study.expected_runs, 36)
        self.assertEqual(len({run.name for run in runs}), 36)
        self.assertEqual(runs[0].name, "GSR_C05_D4_H5K_S8")
        self.assertEqual(runs[-1].name, "GSR_C15_D8_H10K_S123")
        self.assertIn("--dg_recruitment_redundancy_max_steps=4", runs[0].args)
        self.assertIn("--seed=8", runs[0].args)
        self.assertEqual(self.study.raw["schema"], SCHEMA_ID)
        self.assertEqual(self.study.declared_workflow_version, "1.0.0")
        self.assertEqual(WORKFLOW_VERSION, "1.2.0")
        self.assertEqual(len(self.study.fingerprint), 64)

    def test_machine_readable_schema_is_valid_json(self):
        schema_path = Path(__file__).with_name("intrmotiv_study") / "study.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], SCHEMA_ID)
        self.assertIn("training", schema["properties"])

    def test_historical_manifest_is_now_a_thin_compatibility_adapter(self):
        rows = legacy_rows()
        self.assertEqual(len(rows), 36)
        self.assertEqual(rows[0]["name"], "GSR_C05_D4_H5K_S8")
        self.assertNotIn("--seed=8", rows[0]["args"])
        self.assertEqual(rows[0]["context_controls"], [
            "original_C05_seed8",
            "original_C05_seed99",
            "original_C05_seed123",
        ])

    def test_incorrect_expected_product_is_rejected(self):
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        raw["expected_runs"] = 35
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(SpecError, "Cartesian product contains 36"):
                load_study(path)

    def test_duplicate_cli_flags_are_rejected(self):
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        raw["training"]["common_args"].append("--seed=456")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(SpecError, "duplicate flags"):
                load_study(path)

    def test_supplemental_study_cannot_be_submitted_as_complete(self):
        with self.assertRaisesRegex(RuntimeError, "not a complete Sample Factory"):
            build_run_description(self.study)


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)
        self.records = []
        for run in self.study.expand_runs():
            distance = int(run.factors["redundancy_max_steps"])
            half_life = int(run.factors["half_life_events"])
            self.records.append({
                "run_name": run.name,
                "base": run.base,
                "seed": run.seed,
                **run.factors,
                "score": distance + half_life / 10_000 + run.seed / 1_000,
            })

    def test_group_summary_has_mean_sd_and_count(self):
        summary = summarize_records(
            self.records,
            ["base", "redundancy_max_steps", "half_life_events"],
            ["score"],
        )
        self.assertEqual(len(summary), 12)
        self.assertEqual({row["score__n"] for row in summary}, {3})
        self.assertTrue(all(row["score__sd"] > 0 for row in summary))

    def test_explicit_factorial_contrasts_are_seed_paired(self):
        detailed, summary = linear_contrasts(
            self.records,
            ["score"],
            ["base"],
            ["seed"],
            self.study.analysis["contrasts"],
        )
        self.assertEqual(len(detailed), 27)
        by_name = {row["contrast"]: row for row in summary if row["base"] == "C05"}
        self.assertAlmostEqual(by_name["D8_minus_D4"]["mean"], 4.0)
        self.assertAlmostEqual(by_name["H10k_minus_H5k"]["mean"], 0.5)
        self.assertAlmostEqual(by_name["interaction"]["mean"], 0.0)
        self.assertEqual(by_name["interaction"]["n"], 3)

    def test_contrast_refuses_ambiguous_terms(self):
        bad = [{
            "name": "ambiguous",
            "terms": [{"weight": 1, "where": {"redundancy_max_steps": 8}}],
        }]
        with self.assertRaisesRegex(SpecError, "selected 2 rows"):
            linear_contrasts(self.records, ["score"], ["base"], ["seed"], bad)


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)

    def test_standard_protocol_builds_five_plus_two_rows_per_condition(self):
        inventory = []
        targets = self.study.telemetry["target_frames"]
        for run in self.study.expand_runs():
            for target in targets:
                inventory.append(CheckpointRecord(
                    run_name=run.name,
                    target_frames=target,
                    checkpoint_frames=target + run.seed,
                    checkpoint=Path(
                        f"/work/classic/fr_xl1014-train/checkpoints/{run.name}/{target}.pth"
                    ),
                    run_dir=Path(f"/work/classic/fr_xl1014-train/runs/{run.name}"),
                ))
        rows, trajectory = build_place_field_manifests(
            self.study, inventory, require_checkpoint_files=False
        )
        self.assertEqual(len(rows), 84)
        self.assertEqual(len(trajectory), 60)
        self.assertEqual(tuple(rows[0]), MANIFEST_COLUMNS)
        self.assertEqual(len({row["label_suffix"] for row in rows}), 84)
        self.assertEqual({row["seed"] for row in trajectory}, {"99"})

    def test_nonworkspace_checkpoint_is_rejected(self):
        run = self.study.expand_runs()[0]
        inventory = [CheckpointRecord(
            run_name=run.name,
            target_frames=target,
            checkpoint_frames=target,
            checkpoint=Path(f"/home/fr/fr_xl1014/{target}.pth"),
            run_dir=Path("/home/fr/fr_xl1014/run"),
        ) for target in self.study.telemetry["target_frames"]]
        with self.assertRaisesRegex(SpecError, "outside the workspace"):
            build_place_field_manifests(
                self.study, inventory, require_checkpoint_files=False
            )


class SubmissionAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = load_study(SPEC_PATH)

    def _write_jobs(self, path: Path, remove_first_arg: bool = False) -> None:
        fields = [
            "job_id", "status", "experiment", "train_root", "sbatch_file",
            "stdout", "stderr", "command",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for index, run in enumerate(self.study.expand_runs(), start=1000):
                args = list(run.args)
                if remove_first_arg and index == 1000:
                    args.pop(0)
                root = "/work/classic/fr_xl1014-train/study"
                writer.writerow({
                    "job_id": str(index),
                    "status": "submitted",
                    "experiment": f"00_{run.name}",
                    "train_root": f"batch/{run.name}",
                    "sbatch_file": f"{root}/sbatch_{run.name}.sh",
                    "stdout": f"{root}/{run.name}-%j.out",
                    "stderr": f"{root}/{run.name}-%j.err",
                    "command": " ".join([
                        *args,
                        f"--experiment=00_{run.name}",
                        f"--train_dir={root}/{run.name}",
                    ]),
                })

    def test_submission_audit_matches_matrix_commands_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            self._write_jobs(path)
            result = audit_submission(self.study, path, require_submitted=True)
        self.assertTrue(result["submitted_complete"])
        self.assertTrue(result["commands_match_study"])
        self.assertEqual(result["status_counts"], {"submitted": 36})

    def test_submission_audit_rejects_missing_study_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            self._write_jobs(path, remove_first_arg=True)
            with self.assertRaisesRegex(SpecError, "missing study arguments"):
                audit_submission(self.study, path, require_submitted=True)


class TensorBoardPrimitiveTests(unittest.TestCase):
    def test_window_mean_and_latest_are_shared_primitives(self):
        events = [Event(1, 1.0, 1.0), Event(2, 3.0, 2.0), Event(3, 8.0, 3.0)]
        self.assertEqual(mean_in_window(events, 1, 2), (2.0, 2))
        self.assertEqual(latest_at_or_before(events, 2), (3.0, 2))
        missing, count = mean_in_window(events, 10, 20)
        self.assertTrue(math.isnan(missing))
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
