"""Separate-cohort provenance and fake native hooks; no native execution."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hpc_runs.first_hit_control.aggregate_evaluation import (
    _canonical_hash, _read_trials, aggregate, arrival_interval, file_hash, first_before_interval, read_shard,
)
from hpc_runs.first_hit_control.block_journal import CompleteBlockJournal, validate_complete_journal
from hpc_runs.first_hit_control.dmlab_evaluate import EvaluationConfig, execute_trial
from hpc_runs.first_hit_control.provenance import audit_source_snapshots, snapshot_sources
from hpc_runs.first_hit_control.reused_control import (
    PRERUN, SCHEMA, ReusedControlEvaluator, candidate_hashes, read_reused_shard,
    run_on_nemo, signature_hash, validate_candidate_binding, verify_engine_sources,
    validate_effective_plan_binding,
)
from hpc_runs import test_first_hit_aggregate_evaluation as original_artifacts
from hpc_runs.test_first_hit_aggregate_evaluation import artificial_rows, write_json
from hpc_runs.test_first_hit_dmlab_evaluate import FakeAPI, FakeAdapter


def fake_native(*, terminal=False, cap=10000):
    api, model = FakeAPI(physical_done=terminal, cap=cap), FakeAdapter(horizon=3)
    model.action_parameterization = nn.Linear(1, 2)
    model.eval().requires_grad_(False)
    def signature(obs, state):
        return dict(obs=torch.as_tensor(obs), __state=state.clone(),
                    __position=torch.tensor([obs, 0., 0.]), __rotation=torch.zeros(3))
    api["signature"] = signature
    start = api.start()
    config = asdict(EvaluationConfig(horizon=3, discovery_episodes=1, prefix_cap=6, repeats=1,
                                     max_sources=1, focus_sources=(0,)))
    summary = dict(config=config, eligible_targets=[0, 1, 2], source_panel=[0],
                   observed_targets=[0, 1, 2], declared_dg_vocabulary=[0, 1, 2], unobserved_goal_ids=[],
                   primary_detector="winner", secondary_detector="exclusive", command_protocol="own stop",
                   terminal_protocol="unknown terminal censor", frozen_sha256_before=candidate_hashes(api["actor"], model))
    return api, model, start, summary


def callback_fixture(tmp_path, *, missing=False, terminal=False, cap=10000):
    api, model, start, summary = fake_native(terminal=terminal, cap=cap)
    rows = []
    # Source-only rows establish the old start block, never candidate outcomes.
    for command in summary["eligible_targets"]:
        rows += execute_trial(api, model, policy="source", start=start, command=command,
                              targets=summary["eligible_targets"], horizon=3)
    if missing:
        summary["source_panel"].append(1)
        for command in range(3):
            for target in range(3):
                rows.append(dict(policy="source", source=1, repeat=0, command=command, target=target,
                    issued=False, missing_reason="decision_budget"))
    path = tmp_path / "trials.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    baseline = SimpleNamespace(directory=tmp_path, summary=summary, receipt=dict(trials_sha256=file_hash(path)))
    api.count = 0
    api["actor"].inputs.clear()
    return api, model, baseline, {0: [start]}


def test_candidate_only_reuses_all_blocks_without_discovery_or_source_actions(tmp_path):
    api, model, baseline, panel = callback_fixture(tmp_path, missing=True)
    # Source replay() internally calls its established fresh closure; any
    # discovery caller using the public entry would fail this fixture.
    api["fresh"] = lambda *_: (_ for _ in ()).throw(AssertionError("No discovery"))
    rows, summary = ReusedControlEvaluator(model, baseline, panel)(api)
    assert len(rows) == 2 * 3 * 3 and all(row["policy"] == "first_hit" for row in rows)
    assert len([r for r in rows if not r["issued"]]) == 9
    assert all(r["missing_reason"] == "decision_budget" for r in rows if not r["issued"])
    assert not api["actor"].inputs and api.closed
    assert summary["source_control_rerun"] is False and summary["discovery_decisions"] == 0
    assert summary["frozen_sha256_before"] == summary["frozen_sha256_after"]


def test_exact_full_recurrent_signature_and_donor_hash_required(tmp_path):
    api, model, baseline, panel = callback_fixture(tmp_path)
    panel[0][0]["signature"]["__state"] += 1
    with pytest.raises(RuntimeError, match="signature differs exactly"):
        ReusedControlEvaluator(model, baseline, panel)(api)
    api, model, start, summary = fake_native()
    baseline.summary["frozen_sha256_before"]["actor"] = "changed"
    with pytest.raises(ValueError, match="actor/graph"):
        ReusedControlEvaluator(model, baseline, {0:[start]})(api)


def test_retained_prefix_mismatch_and_unissued_block_never_recovered(tmp_path):
    api, model, baseline, panel = callback_fixture(tmp_path, missing=True)
    panel[0][0]["prefix"] = [1]
    with pytest.raises(ValueError, match="retained matched start"):
        ReusedControlEvaluator(model, baseline, panel)
    panel[0][0]["prefix"] = []
    extra = dict(panel[0][0], source=1)
    panel[1] = [extra]
    rows, _ = ReusedControlEvaluator(model, baseline, panel)(api)
    assert all(not r["issued"] for r in rows if r["source"] == 1)


def test_unknown_terminal_and_missing_keep_identification_bounds(tmp_path):
    api, model, baseline, panel = callback_fixture(tmp_path, terminal=True)
    rows, _ = ReusedControlEvaluator(model, baseline, panel)(api)
    row = next(r for r in rows if r["command"] == 2 and r["target"] == 2)
    competitor = next(r for r in rows if r["command"] == 2 and r["target"] == 1)
    assert arrival_interval(row) == (0., 1.)
    assert first_before_interval(row, competitor, 3) == (0., 1.)
    assert arrival_interval(dict(issued=False)) == (0., 1.)


def test_readout_head_mutation_fails_and_small_budget_does_not_drop_block(tmp_path):
    api, model, baseline, panel = callback_fixture(tmp_path)
    forward = model.forward
    def changing(*args):
        with torch.no_grad():
            model.action_parameterization.weight.add_(1)
        return forward(*args)
    model.forward = changing
    with pytest.raises(RuntimeError, match="mutated"):
        ReusedControlEvaluator(model, baseline, panel)(api)
    api, model, baseline, panel = callback_fixture(tmp_path)
    api["decision_cap"] = 1
    with pytest.raises(RuntimeError, match="complete command block"):
        ReusedControlEvaluator(model, baseline, panel)(api)


@pytest.fixture
def artifacts():
    fixture = original_artifacts.AggregateEvaluationTests()
    fixture.setUp()
    yield fixture
    fixture.doCleanups()


def reused_artifact(fixture, source, *, missing=False):
    olddir = fixture.shard(source, missing=missing)
    baseline = read_shard(olddir)
    directory = fixture.root / f"candidate_{source}"
    directory.mkdir()
    config = dict(baseline.summary["config"], discovery_episodes=0)
    fit_config = dict(seed=99, updates=30000, batch_size=128, horizon=3, learning_rate=.0003,
                      target_update_period=100, holdout_fraction=.1)
    metadata = dict(baseline.manifest["fitted_metadata"], kind="stationary_projected_first_hit", gamma=.99,
        bootstrap_projection=[0,1], collection_evaluation_overrides=dict(baseline.manifest["evaluation_overrides"],
            dmlab_level_cache_path=str(fixture.root / "same_fitting_replay_cache")),
        fit=dict(config=fit_config, final_readout_sha256="1"*64))
    plan = dict(donor=dict(checkpoint_sha256="a"*64, config_sha256="b"*64), fit=fit_config,
                evaluation=dict(horizon=3, repeats=1, source_shards=[[i] for i in range(4)]))
    validation = dict(effective_plan_sha256=_canonical_hash(plan), donor_study_sha256="9"*64)
    metadata.update(validation)
    rows = [r for r in artificial_rows(source, selective_candidate=True, missing=missing) if r["policy"] == "first_hit"]
    journal = CompleteBlockJournal(directory, [(source,0)], [(c,t) for c in range(4) for t in range(4)])
    journal.append((source,0), rows)
    journal.finalize()
    trial = directory / "candidate_trials.jsonl"
    snapshot = snapshot_sources([fixture.source], directory / "source_snapshot")
    frozen = {key: "f"*64 for key in ("actor", "graph", "readout", "action_head")}
    signatures = [] if missing else [dict(source=source, repeat=0, seed=41000+source,
        prefix_length=1, prefix_sha256=_canonical_hash([source]), full_signature_sha256="f"*64)]
    manifest = dict(schema=PRERUN, source_root=str(fixture.runtime), source_snapshot=snapshot,
        imported_control=baseline.receipt, imported_control_identity=baseline.identity, engine_identity={},
        fitted_metadata=metadata, analysis_plan=plan,
        plan_validation=validation,
        effective_plan_sha256=_canonical_hash(plan), evaluation_config=config, matched_start_signatures=signatures)
    summary = dict(baseline.summary, schema=SCHEMA, config=config, frozen_sha256_before=frozen,
        frozen_sha256_after=frozen, readout_frozen=True, action_head_frozen=True,
        source_control_rerun=False, discovery_decisions=0, matched_start_signatures=signatures,
        restored_readout_sha256="1"*64, fitted_checkpoint_unchanged=True,
        source_audit=audit_source_snapshots(directory / "source_snapshot"), trials_sha256=file_hash(trial))
    write_json(directory / "pre_run_manifest.json", manifest)
    write_json(directory / "summary.json", summary)
    return directory


def test_model_free_loader_imports_source_only_and_explicit_cohort(artifacts, monkeypatch):
    directory = reused_artifact(artifacts, 0)
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No deserialization")))
    shard = read_reused_shard(directory)
    assert len(shard.rows) == 32
    assert shard.identity["cohort_contract"] == "imported_source_and_new_stationary_candidate"
    assert len([r for r in shard.rows if r["policy"] == "source"]) == 16
    assert len([r for r in shard.rows if r["policy"] == "first_hit"]) == 16
    assert not (directory / "trials.jsonl").exists()


def test_original_preflight_import_is_explicit_and_keeps_capacity(artifacts):
    directory = artifacts.shard(2)
    summary_path, manifest_path = directory / "summary.json", directory / "pre_run_manifest.json"
    summary, manifest = json.loads(summary_path.read_text()), json.loads(manifest_path.read_text())
    config = dict(summary["config"], phase="preflight", preflight_targets=[0,1])
    summary.update(config=config, eligible_targets=[0,1])
    manifest["evaluation_config"] = config
    path = directory / "trials.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row["command"] in (0,1) and row["target"] in (0,1)]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    summary["trials_sha256"] = file_hash(path)
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="full evaluation"):
        read_shard(directory)
    imported = read_shard(directory, allow_preflight=True)
    assert len(imported.rows) == 8 and imported.summary["declared_dg_vocabulary"] == [0,1,2,3]


def test_model_free_reducer_preserves_missing_capacity_and_separate_provenance(artifacts):
    directories = [reused_artifact(artifacts, i, missing=i==3) for i in range(4)]
    result = aggregate(directories, artifacts.root / "reused_aggregate", bootstrap_replicates=20)
    assert result["status"] == "validated_completed_reused_control_shards"
    assert result["cohort_contract"] == "imported_source_and_new_stationary_candidate"
    assert result["outcomes"]["first_hit"]["capacity_counts"]["missing_nontrivial_own_rows"] == 3
    assert result["competing_target_FIRST"]["source"]["overall"]["identified_comparisons"] < 24
    assert all("imported_control" in receipt for receipt in result["input_receipts"])


def test_candidate_prefix_and_cartesian_mismatch_rejected(artifacts):
    directory = reused_artifact(artifacts, 0)
    path = directory / "candidate_trials.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    for row in rows:
        row["prefix_actions"] = [99]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    summary = json.loads((directory / "summary.json").read_text())
    summary["trials_sha256"] = file_hash(path)
    write_json(directory / "summary.json", summary)
    journal = json.loads((directory / "block_journal.json").read_text())
    journal["sha256"] = summary["trials_sha256"]
    write_json(directory / "block_journal.json", journal)
    with pytest.raises(ValueError, match="prefixes differ"):
        read_reused_shard(directory)
    path.write_text("\n".join(original.splitlines()[:-1])+"\n")
    summary["trials_sha256"] = file_hash(path)
    write_json(directory / "summary.json", summary)
    journal["sha256"] = summary["trials_sha256"]
    write_json(directory / "block_journal.json", journal)
    with pytest.raises(ValueError, match="Incomplete declared"):
        read_reused_shard(directory)


def test_old_summary_hash_and_candidate_donor_mismatch_rejected(artifacts):
    directory = reused_artifact(artifacts, 0)
    manifest_path = directory / "pre_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fitted_metadata"]["source_checkpoint_sha256"] = "changed"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="disagree"):
        read_reused_shard(directory)
    manifest["fitted_metadata"]["source_checkpoint_sha256"] = "a"*64
    write_json(manifest_path, manifest)
    old_summary = Path(manifest["imported_control"]["directory"]) / "summary.json"
    old_summary.write_text(old_summary.read_text()+" ")
    with pytest.raises(ValueError, match="receipt changed"):
        read_reused_shard(directory)


def test_signature_dtype_and_native_guard():
    signature = dict(obs=torch.tensor([1]), __state=torch.ones(1), __position=torch.zeros(3), __rotation=torch.zeros(3))
    assert signature_hash(signature) != signature_hash(dict(signature, __state=torch.ones(1, dtype=torch.float64)))
    with pytest.raises(ValueError, match="full recurrent"):
        signature_hash(dict(obs=torch.tensor([1])))
    with pytest.raises(RuntimeError, match="allocated NEMO"):
        run_on_nemo(SimpleNamespace())
    plan = {"runtime": {"source_release": "fixed_release"}}
    validation = dict(effective_plan_sha256=_canonical_hash(plan), donor_study_sha256="9"*64)
    with pytest.raises(ValueError, match="effective_plan_sha256"):
        validate_effective_plan_binding(dict(validation, effective_plan_sha256="changed"), plan, validation)


def test_missing_full_state_receipt_and_engine_source_drift_rejected(artifacts, tmp_path):
    directory = reused_artifact(artifacts, 0)
    for filename in ("pre_run_manifest.json", "summary.json"):
        path = directory / filename
        value = json.loads(path.read_text())
        value["matched_start_signatures"] = []
        write_json(path, value)
    with pytest.raises(ValueError, match="full-state signature"):
        read_reused_shard(directory)
    import hashlib
    relative = "sf_working_directories/IntrMotiv/evaluation/target_control_interventions.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("unchanged canonical engine\n")
    baseline = SimpleNamespace(identity=dict(sources={relative:file_hash(path),
        "canonical_callback_overlay":hashlib.sha256(b"overlay").hexdigest()}))
    path.write_text("changed engine\n")
    with pytest.raises(ValueError, match="engine semantics changed"):
        verify_engine_sources(baseline, tmp_path, "overlay")


def test_complete_block_journal_is_durable_partial_and_never_resumes(tmp_path):
    journal = CompleteBlockJournal(tmp_path, [(0,0), (1,0)], [(0,0)])
    row = dict(source=0, repeat=0, command=0, target=0)
    receipt = journal.append((0,0), [row])
    assert receipt["status"] == "partial" and receipt["aggregatable"] is False
    assert json.loads((tmp_path / "candidate_trials.jsonl.partial").read_text()) == row
    with pytest.raises(ValueError, match="missing declared blocks"):
        journal.finalize()
    with pytest.raises(ValueError, match="diagnostic only"):
        validate_complete_journal(tmp_path, [(0,0),(1,0)], 1, receipt["sha256"])
    with pytest.raises(FileExistsError, match="resume"):
        CompleteBlockJournal(tmp_path, [(0,0),(1,0)], [(0,0)])
    with pytest.raises(ValueError, match="complete declared block"):
        journal.append((1,0), [])
    journal.append((1,0), [dict(row, source=1)])
    complete = journal.finalize()
    assert validate_complete_journal(tmp_path, [(0,0),(1,0)], 1, complete["sha256"])["status"] == "complete"
    assert not (tmp_path / "candidate_trials.jsonl.partial").exists()


def test_callback_journals_only_completed_blocks_and_releases_heavy_rows(tmp_path):
    api, model, baseline, panel = callback_fixture(tmp_path, missing=True)
    journal = CompleteBlockJournal(tmp_path, [(0,0),(1,0)], [(c,t) for c in range(3) for t in range(3)])
    rows, _ = ReusedControlEvaluator(model, baseline, panel, journal=journal)(api)
    assert len(rows) == 18 and not any("trajectory" in row or "executed_actions" in row for row in rows)
    status = json.loads((tmp_path / "block_journal.json").read_text())
    assert status["completed_blocks"] == [[0,0],[1,0]] and status["status"] == "partial"
    persisted = [json.loads(line) for line in journal.partial.read_text().splitlines()]
    assert any("trajectory" in row for row in persisted)
