"""Certified trajectory sources preserve separate old/new policy cohorts."""

from contextlib import closing
import json

import pytest

from hpc_runs.first_hit_control.aggregate_evaluation import (
    certified_trial_sources, file_hash, iter_certified_trial_rows, trial_source_records,
)


def write_rows(directory, name, rows):
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return file_hash(path)


def reused_fixture(root):
    old, new = root / "original", root / "candidate"
    old_hash = write_rows(old, "trials.jsonl", [dict(policy="source", marker="original_source"),
        dict(policy="first_hit", marker="old_finite_head_never_import")])
    new_hash = write_rows(new, "candidate_trials.jsonl", [dict(policy="first_hit", marker="new_stationary")])
    receipt = dict(schema="intrmotiv/reused-control-evaluation/v1", directory=str(new), trials_sha256=new_hash,
                   candidate_trials_file="candidate_trials.jsonl",
                   imported_control=dict(directory=str(old), trials_sha256=old_hash))
    receipt["trial_sources"] = trial_source_records(receipt)
    return receipt


def test_legacy_minimal_receipt_preserves_both_policies(tmp_path):
    rows = [dict(policy="source"), dict(policy="first_hit")]
    sha = write_rows(tmp_path, "trials.jsonl", rows)
    receipt = dict(directory=str(tmp_path), trials_sha256=sha)
    sources = certified_trial_sources(receipt)
    assert len(sources) == 1 and sources[0].policies == ("source", "first_hit")
    assert sources[0].path == tmp_path / "trials.jsonl" and sources[0].sha256 == sha
    assert list(iter_certified_trial_rows(receipt)) == rows


def test_reused_sources_exclude_old_finite_head_and_preserve_raw_rows(tmp_path):
    receipt = reused_fixture(tmp_path)
    sources = certified_trial_sources(receipt)
    assert [s.policies for s in sources] == [("source",), ("first_hit",)]
    assert [s.cohort for s in sources] == ["imported_source_control", "new_stationary_candidate"]
    rows = list(iter_certified_trial_rows(receipt))
    assert [row["marker"] for row in rows] == ["original_source", "new_stationary"]
    assert all(set(row) == {"policy", "marker"} for row in rows)


def test_explicit_filter_cannot_broaden_imported_source(tmp_path):
    receipt = reused_fixture(tmp_path)
    receipt["trial_sources"][0]["policies"] = ["source", "first_hit"]
    with pytest.raises(ValueError, match="Explicit trial sources"):
        certified_trial_sources(receipt)


def test_old_and_new_file_tampering_rejected_before_reading(tmp_path):
    receipt = reused_fixture(tmp_path)
    path = tmp_path / "original/trials.jsonl"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="Trial bytes"):
        list(iter_certified_trial_rows(receipt))


def test_foreign_policy_in_candidate_file_fails_even_with_updated_hash(tmp_path):
    receipt = reused_fixture(tmp_path)
    receipt["trials_sha256"] = write_rows(tmp_path / "candidate", "candidate_trials.jsonl", [dict(policy="source")])
    receipt["trial_sources"] = trial_source_records(receipt)
    with pytest.raises(ValueError, match="Unexpected policy"):
        list(iter_certified_trial_rows(receipt))


def test_closing_a_partial_stream_runs_final_integrity_check(tmp_path):
    receipt = reused_fixture(tmp_path)
    with pytest.raises(ValueError, match="Trial bytes changed"):
        with closing(iter_certified_trial_rows(receipt)) as stream:
            assert next(stream)["marker"] == "original_source"
            path = tmp_path / "original/trials.jsonl"
            path.write_text(path.read_text()+"\n")


def test_optional_completed_receipt_hashes_are_checked(tmp_path):
    receipt = reused_fixture(tmp_path)
    summary = tmp_path / "original/summary.json"
    summary.write_text("{}\n")
    receipt["imported_control"]["summary_sha256"] = file_hash(summary)
    summary.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="receipt changed"):
        certified_trial_sources(receipt)


def test_nested_reused_control_cannot_hide_another_candidate_cohort(tmp_path):
    receipt = reused_fixture(tmp_path)
    receipt["imported_control"]["schema"] = "intrmotiv/reused-control-evaluation/v1"
    with pytest.raises(ValueError, match="original completed"):
        trial_source_records(receipt)
