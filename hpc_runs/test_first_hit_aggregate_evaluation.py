"""Pure artifact/causal-estimand tests; no native environment or trained model."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from hpc_runs.first_hit_control.aggregate_evaluation import (
    aggregate, arrival_interval, contrast_summary, file_hash, first_before_interval, read_shard,
)
from hpc_runs.first_hit_control.dmlab_evaluate import EvaluationConfig
from hpc_runs.first_hit_control.provenance import audit_source_snapshots, snapshot_sources


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def artificial_rows(source, *, selective_candidate=False, missing=False):
    """Both goalblind actors follow the same cyclic physical winner path.

    Each command terminates its own path when that command is reached. This is
    the explicit counterexample where naive own-stop lift is positive without
    command-dependent navigation. The optional candidate goes directly to its
    commanded goal, providing a distinct FIRST-order positive control.
    """
    rows = []
    for policy in ("source", "first_hit"):
        for command in range(4):
            if missing:
                for target in range(4):
                    rows.append(dict(policy=policy, source=source, repeat=0, command=command, target=target,
                        commanded=command == target, issued=False, missing=True, missing_reason="start_not_discovered",
                        initially_at_target=target == source, initially_at_command=command == source,
                        hit=False, censored=False))
                continue
            sequence = [source]
            if command != source:
                if policy == "first_hit" and selective_candidate:
                    sequence.append(command)
                else:
                    sequence += [(source + offset) % 4 for offset in range(1, (command - source) % 4 + 1)]
            elapsed = len(sequence) - 1
            for target in range(4):
                hit_time = sequence.index(target) if target in sequence else None
                row = dict(policy=policy, source=source, repeat=0, prefix_seed=41000 + source,
                    prefix_actions=[source], command=command, target=target, commanded=command == target,
                    issued=True, missing=False, missing_reason=None, exact_start_verified=True, horizon=3,
                    elapsed=elapsed, hit=hit_time is not None, hit_time=hit_time,
                    exclusive_hit=hit_time is not None, exclusive_hit_time=hit_time,
                    initially_at_target=target == source, initially_at_command=command == source,
                    command_hit=True, physical_done=False, successor_unknown=False, censored=False,
                    timeout=False, first_distinct=sequence[1] if elapsed else None,
                    initial_scores=[1., 0., 0., 0.] if elapsed else None,
                    executed_actions=[0] * elapsed, stopped_at_other_command=hit_time is None)
                if command == target:
                    row["trajectory"] = [{"position": [100. * winner, 0., 0.], "dg": [0.] * 4} for winner in sequence]
                rows.append(row)
    return rows


class AggregateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.source = self.runtime / "evaluator.py"
        self.source.write_text("# immutable fake evaluator source\n")

    def shard(self, source, *, selective_candidate=False, missing=False):
        directory = self.root / f"shard_{source}"
        directory.mkdir()
        config = asdict(EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=3, repeats=1,
                                         max_sources=1, focus_sources=(source,)))
        config["focus_sources"] = [source]
        config["preflight_targets"] = []
        layout = dict(width=8, goal_start=4, goals=4)
        snapshot = snapshot_sources([self.source], directory / "source_snapshot")
        manifest = dict(schema="intrmotiv/matched-first-hit-prerun/v1", evaluation_config=config,
            source_root=str(self.runtime), source_snapshot=snapshot, source_checkpoint_sha256="a" * 64,
            original_config_sha256="b" * 64, canonical_source_sha256="c" * 64, callback_source_sha256="d" * 64,
            fitted_metadata=dict(source_checkpoint_sha256="a" * 64, source_config_sha256="b" * 64,
                                 checkpoint_sha256="e" * 64, primary_detector="winner", layout=layout),
            evaluation_overrides=dict(dmlab_level_cache_path=str(directory / "cache"), transfer_freeze_dg=True))
        write_json(directory / "pre_run_manifest.json", manifest)
        with (directory / "trials.jsonl").open("w") as stream:
            for row in artificial_rows(source, selective_candidate=selective_candidate, missing=missing):
                stream.write(json.dumps(row) + "\n")
        (directory / "matched_starts.pt").write_bytes(b"opaque fixture; aggregator checks bytes and never unpickles")
        frozen = {key: "f" * 64 for key in ("actor", "graph", "readout", "q")}
        summary = dict(schema="intrmotiv/matched-first-hit-evaluation/v2", config=config,
            source_panel=[source], declared_dg_vocabulary=list(range(4)), observed_targets=list(range(4)),
            eligible_targets=list(range(4)), unobserved_goal_ids=[], eligibility=dict(layout=layout),
            primary_detector="winner", policies_frozen=True, graph_frozen=True, q_frozen=True,
            checkpoint_unchanged=True, source_configuration_unchanged=True,
            frozen_sha256_before=frozen, frozen_sha256_after=frozen,
            trials_sha256=file_hash(directory / "trials.jsonl"), matched_starts_sha256=file_hash(directory / "matched_starts.pt"),
            source_audit=audit_source_snapshots(directory / "source_snapshot"), action_protocol="deterministic argmax",
            command_protocol="Stop at own winner including initial", terminal_protocol="Unknown terminal is censored")
        write_json(directory / "summary.json", summary)
        return directory

    def shards(self, **kwargs):
        return [self.shard(source, **kwargs) for source in range(4)]

    def mutate_summary(self, directory, change):
        path = directory / "summary.json"
        value = json.loads(path.read_text())
        change(value)
        write_json(path, value)

    def rewrite_rows(self, directory, change):
        path = directory / "trials.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.mutate_summary(directory, lambda summary: summary.update(trials_sha256=file_hash(path)))

    def test_goalblind_ownstop_lift_is_positive_but_competing_first_is_zero(self):
        result = aggregate(self.shards(), self.root / "aggregate")
        for policy in ("source", "first_hit"):
            self.assertGreater(result["outcomes"][policy]["paired_lift_censored_as_unconfirmed"], 0)
            contrast = result["competing_target_FIRST"][policy]["overall"]
            self.assertEqual(contrast["declared_comparisons"], 24)
            self.assertEqual(contrast["point_estimate_if_fully_identified"], 0)
            self.assertEqual(contrast["source_cluster_bootstrap"]["identification_endpoint_envelope"], [0, 0])
        self.assertEqual(result["retained_light_trial_rows"], 128)
        self.assertFalse(result["heavy_trajectory_action_arrays_retained"])
        self.assertEqual(len((self.root / "aggregate/per_goal.csv").read_text().splitlines()), 5)
        self.assertEqual(len((self.root / "aggregate/source_target.csv").read_text().splitlines()), 17)
        self.assertEqual(len((self.root / "aggregate/competing_targets.csv").read_text().splitlines()), 25)
        import csv
        with (self.root / "aggregate/competing_targets.csv").open() as stream:
            matrix = list(csv.DictReader(stream))
        self.assertTrue(all(int(row["declared_comparisons"]) == 2 for row in matrix))
        self.assertTrue(all(float(row["mean_lower_bound"]) == float(row["mean_upper_bound"]) == 0 for row in matrix))
        self.assertTrue(result["analysis_source_audit"]["archive_valid"])

    def test_selective_ordering_positive_control_separate_from_own_arrival(self):
        result = aggregate(self.shards(selective_candidate=True), self.root / "aggregate")
        self.assertEqual(result["paired_candidate_minus_source"]["overall"]["point_estimate_if_fully_identified"], 0)
        self.assertEqual(result["competing_target_FIRST"]["source"]["overall"]["point_estimate_if_fully_identified"], 0)
        self.assertEqual(result["competing_target_FIRST"]["first_hit"]["overall"]["point_estimate_if_fully_identified"], 1)

    def test_missing_source_is_retained_as_unknown_in_capacity_and_contrasts(self):
        directories = [self.shard(source, missing=source == 3) for source in range(4)]
        result = aggregate(directories, self.root / "aggregate")
        own = result["outcomes"]["first_hit"]["capacity_counts"]
        self.assertEqual(own["declared_nontrivial_own_rows"], 12)
        self.assertEqual(own["missing_nontrivial_own_rows"], 3)
        self.assertEqual((own["success_lower_bound_declared"], own["success_upper_bound_declared"]), (.75, 1))
        contrast = result["paired_candidate_minus_source"]["overall"]
        self.assertEqual((contrast["mean_lower_bound"], contrast["mean_upper_bound"]), (-.25, .25))
        self.assertIsNone(contrast["point_estimate_if_fully_identified"])
        self.assertNotIn("complete_case_mean", contrast)
        self.assertNotIn("complete_case_interval", contrast["source_cluster_bootstrap"])
        self.assertEqual(result["competing_target_FIRST"]["first_hit"]["overall"]["missing_comparisons"], 6)

    def test_streamed_reader_discards_heavy_arrays(self):
        rows = read_shard(self.shard(0)).rows
        self.assertTrue(rows)
        self.assertFalse(any(key in row for row in rows for key in ("trajectory", "executed_actions", "prefix_actions")))

    def test_trial_tamper_detected_before_parsing(self):
        path = self.shard(0)
        with (path / "trials.jsonl").open("a") as stream:
            stream.write("not JSON\n")
        with self.assertRaisesRegex(ValueError, "Trial bytes differ"):
            read_shard(path)

    def test_missing_or_duplicate_trial_rows_fail_complete_cartesian_contract(self):
        path = self.shard(0)
        self.rewrite_rows(path, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "Incomplete declared"):
            read_shard(path)

    def test_duplicate_row_is_not_an_extra_replicate(self):
        path = self.shard(0)
        self.rewrite_rows(path, lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            read_shard(path)

    def test_wrong_fitted_identity_rejects_cross_shard_pooling(self):
        paths = self.shards()
        path = paths[1] / "pre_run_manifest.json"
        data = json.loads(path.read_text())
        data["fitted_metadata"]["checkpoint_sha256"] = "9" * 64
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "identities do not match"):
            aggregate(paths, self.root / "aggregate")

    def test_runtime_archive_tamper_rejected(self):
        directory = self.shard(0)
        archived = next((directory / "source_snapshot/sources").rglob("evaluator.py"))
        archived.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Archived evaluator sources"):
            read_shard(directory)

    def test_mutated_q_or_start_artifact_rejected(self):
        directory = self.shard(0)
        self.mutate_summary(directory, lambda value: value["frozen_sha256_after"].update(q="1" * 64))
        with self.assertRaisesRegex(ValueError, "Model, graph or Q changed"):
            read_shard(directory)

    def test_matched_start_bytes_are_validated_without_unpickling(self):
        directory = self.shard(0)
        (directory / "matched_starts.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "matched-start artifact"):
            read_shard(directory)

    def test_command_prefix_mismatch_rejected_even_with_rehashed_trials(self):
        directory = self.shard(0)
        self.rewrite_rows(directory, lambda rows: rows[-1].update(prefix_actions=[99]))
        with self.assertRaisesRegex(ValueError, "different physical prefixes"):
            read_shard(directory)

    def test_partial_command_issuance_cannot_masquerade_as_a_matched_block(self):
        directory = self.shard(0)
        def change(rows):
            rows[-1].update(issued=False, missing=True, missing_reason="decision_budget", hit=False, censored=False)
        self.rewrite_rows(directory, change)
        with self.assertRaisesRegex(ValueError, "mixes issued and missing"):
            read_shard(directory)

    def test_source_partition_must_be_complete_and_disjoint(self):
        paths = self.shards()
        with self.assertRaisesRegex(ValueError, "four distinct"):
            aggregate(paths[:3] + [paths[0]], self.root / "aggregate")
        # Metadata and rows consistently claiming source2 twice still fail the
        # cross-shard partition gate, rather than creating an extra replicate.
        path = paths[3]
        import shutil
        shutil.rmtree(path)
        shutil.copytree(paths[2], path)
        with self.assertRaisesRegex(ValueError, "disjoint complete"):
            aggregate(paths, self.root / "aggregate")

    def test_unobserved_target_keeps_complete_capacity_denominator(self):
        paths = [self.shard(source, missing=source == 3) for source in range(4)]
        for path in paths:
            self.rewrite_rows(path, lambda rows: rows.__setitem__(slice(None),
                [row for row in rows if row["command"] != 3 and row["target"] != 3]))
            self.mutate_summary(path, lambda summary: summary.update(observed_targets=[0, 1, 2],
                eligible_targets=[0, 1, 2], unobserved_goal_ids=[3]))
        result = aggregate(paths, self.root / "aggregate")
        goal = result["outcomes"]["first_hit"]["per_goal"]["3"]
        self.assertEqual(goal["declared_nontrivial_own_rows"], 3)
        self.assertEqual(goal["missing_nontrivial_own_rows"], 3)
        self.assertEqual(goal["issued_nontrivial_own_rows"], 0)
        self.assertEqual((goal["success_lower_bound_declared"], goal["success_upper_bound_declared"]), (0, 1))
        self.assertEqual(result["competing_target_FIRST"]["first_hit"]["overall"]["declared_comparisons"], 24)

    def test_first_event_known_before_later_unknown_terminal(self):
        missing = dict(issued=True, hit_time=None, censored=True, elapsed=3)
        prior = dict(issued=True, hit_time=1, censored=False, elapsed=3)
        self.assertEqual(first_before_interval(missing, prior, 3), (0, 0))
        self.assertEqual(first_before_interval(prior, missing, 3), (1, 1))
        self.assertEqual(first_before_interval(missing, missing, 3), (0, 1))
        clean = dict(issued=True, hit_time=None, censored=False, elapsed=3)
        self.assertEqual(first_before_interval(clean, clean, 3), (0, 0))
        self.assertEqual(first_before_interval(None, clean, 3), (0, 1))
        with self.assertRaisesRegex(ValueError, "same time"):
            first_before_interval(prior, prior, 3)

    def test_censoring_and_missing_are_unknown_not_zero(self):
        self.assertEqual(arrival_interval(None), (0, 1))
        self.assertEqual(arrival_interval(dict(issued=False)), (0, 1))
        self.assertEqual(arrival_interval(dict(issued=True, hit=False, censored=True)), (0, 1))
        self.assertEqual(arrival_interval(dict(issued=True, hit=True, censored=False)), (1, 1))

    def test_bootstrap_clusters_sources_not_repeated_goal_rows(self):
        records = [dict(source=source, lower=value, upper=value) for source, value in enumerate([-1, 0, 0, 1])]
        first = contrast_summary(records, list(range(4)))
        repeated = contrast_summary(records * 100, list(range(4)))
        self.assertEqual(first["source_cluster_bootstrap"]["identification_endpoint_envelope"],
                         repeated["source_cluster_bootstrap"]["identification_endpoint_envelope"])
        self.assertEqual(first["source_cluster_bootstrap"]["seed"], 123)
        self.assertEqual(first["source_cluster_bootstrap"]["replicates"], 10000)
        self.assertEqual(first, contrast_summary(records, list(range(4))))

    def test_never_overwrite_completed_analysis(self):
        paths = self.shards()
        output = self.root / "aggregate"
        aggregate(paths, output)
        with self.assertRaises(FileExistsError):
            aggregate(paths, output)


if __name__ == "__main__":
    unittest.main()
