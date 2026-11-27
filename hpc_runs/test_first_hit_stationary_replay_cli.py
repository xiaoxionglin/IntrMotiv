"""Original-class tensor-only integration, provenance and native sampler reuse."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.eligibility import certify_slice_only_relabel
from hpc_runs.first_hit_control.replay_cli import file_hash
from hpc_runs.first_hit_control.replay_fit import FitConfig, episode_split, fit_replay, replay_batch, state_digest
from hpc_runs.first_hit_control.stationary_replay_cli import (
    RUNTIME_FILES, construct_stationary, load_fitted_stationary, run_fit, validate_plan,
)
from hpc_runs.first_hit_control.targets import all_goal_hits


@pytest.fixture
def source_replay(tmp_path):
    from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
    from sf_working_directories.IntrMotiv.dmlab.custom_decoder import MlpDecoderJit
    from sample_factory.model.action_parameterization import ActionParameterizationDefault
    source = Path(os.environ.get("INTRMOTIV_RUNTIME_SOURCE", "/home/xiaoxiong/SFgit/SF_hipposlam"))
    if not source.is_dir():
        pytest.skip("Runtime source unavailable")
    torch.manual_seed(99)
    cfg = SimpleNamespace(Hippo_R=2, Hippo_L=3, Hippo_n_feature=3, hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer", hrl_target_timing="immediate", hrl_manager_mode="visit_direct",
        hrl_goal_conditioning="legacy", dg_goal_input="none", decoder_mlp_layers=[16, 16],
        nonlinearity="relu", use_jit=False)
    core = SimpleSequenceWithBypassCore(cfg, 5)
    decoder = MlpDecoderJit(cfg, core.get_out_size())
    head = ActionParameterizationDefault(cfg, 16, gym.spaces.Discrete(2))
    certificate = certify_slice_only_relabel(core, decoder, source_root=source, input_kind="canonical_core_output")
    snapshot = {"decoder_state_dict": decoder.state_dict(), "action_head_state_dict": head.state_dict(),
        "decoder_class": "MlpDecoderJit", "action_head_class": "ActionParameterizationDefault",
        "decoder_config": {k: getattr(cfg, k) for k in ("decoder_mlp_layers", "nonlinearity", "use_jit")},
        "layout": certificate["layout"], "action_count": 2}
    path = tmp_path / "replay"
    path.mkdir()
    count, width = 24, core.get_out_size()
    np.savez_compressed(path / "replay.npz", core=np.zeros((count, width), np.float32),
        successor_core=np.zeros((count, width), np.float32), action=np.arange(count) % 2,
        successor_dg=np.eye(3, dtype=np.float32)[np.arange(count) % 3], physical_done=np.ones(count, bool),
        successor_valid=np.ones(count, bool), successor_core_valid=np.zeros(count, bool), episode_id=np.arange(count))
    torch.save(snapshot, path / "decoder_state.pt")
    metadata = {"schema": "intrmotiv/frozen-first-hit-replay/v1", "model_unchanged": True,
        "model_sha256_before": "synthetic_fixture", "model_sha256_after": "synthetic_fixture",
        "native_reward_zero_every_step": True, "reset_observations_used_as_successors": 0,
        "primary_detector": "winner", "layout": certificate["layout"], "eligibility": certificate,
        "checkpoint_sha256": "synthetic_not_a_native_checkpoint", "source_run": "synthetic_fixture",
        "source_config_sha256": "a" * 64, "evaluation_overrides": {"transfer_freeze_dg": True},
        "dataset_sha256": file_hash(path / "replay.npz"), "decoder_state_sha256": file_hash(path / "decoder_state.pt")}
    (path / "metadata.json").write_text(json.dumps(metadata))
    return source, path, snapshot, metadata, decoder, head


def adapter_snapshot(source_replay, horizon=2):
    _, _, source, metadata, _, _ = source_replay
    return dict(source, eligibility=metadata["eligibility"], fit_config={"horizon": horizon}, gamma=.99)


def test_original_source_initialization_reload_clock_invariance_and_input_preservation(source_replay, tmp_path):
    source, replay, snapshot, meta, decoder, head = source_replay
    output = tmp_path / "fit"
    originals = (state_digest(decoder), state_digest(head))
    torch.set_num_threads(1)
    result = run_fit(replay, output, source, FitConfig(updates=3, batch_size=4, horizon=2, log_every=3))
    model, metadata = load_fitted_stationary(output, source)
    assert (result["source_initialization"]["decoder_state_sha256"],
            result["source_initialization"]["action_head_state_sha256"]) == originals
    assert (state_digest(decoder), state_digest(head)) == originals
    assert type(model.decoder).__name__ == "MlpDecoderJit"
    assert type(model.action_parameterization).__name__ == "ActionParameterizationDefault"
    assert not hasattr(model, "q_heads") and all(not p.requires_grad for p in model.parameters())
    assert result["source_audit"]["archive_valid"] and result["source_audit"]["current_sources_unchanged"]
    assert metadata["kind"] == "stationary_projected_first_hit" and metadata["gamma"] == .99
    assert metadata["source_config_sha256"] == meta["source_config_sha256"]
    assert result["fit"]["goal_labels_seen"] == 3 * 4 * 3
    x = torch.randn(3, model.layout.width)
    original_x = x.clone()
    goals = torch.tensor([0, 1, 2])
    captured = []
    handle = model.decoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    q = model(x, goals, torch.tensor([0, 1, 256]))
    handle.remove()
    assert torch.equal(q, model(x, goals, torch.tensor([256, 0, 1])))
    assert torch.equal(x, original_x)
    start = model.layout.goal_start
    assert torch.equal(captured[0][:, :start], x[:, :start])
    assert torch.equal(captured[0][:, start:], torch.eye(3))
    assert torch.equal(model.act(x, goals, None), q.argmax(-1))
    # Different GEMM batch sizes can change last-bit floating point rounding.
    torch.testing.assert_close(model.all_goals(x, None)[torch.arange(3), goals], q, rtol=1e-6, atol=1e-7)
    with (output / "adapter.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_fitted_stationary(output, source)


def test_scripted_original_modules_are_leaf_trainable_and_targets_construct_fresh(source_replay):
    source = source_replay[0]
    snapshot = adapter_snapshot(source_replay)
    snapshot["decoder_config"] = dict(snapshot["decoder_config"], use_jit=True)
    model = construct_stationary(snapshot, source)
    assert all(p.is_leaf and p.requires_grad for p in model.parameters())
    with np.load(source_replay[1] / "replay.npz") as arrays:
        data = dict(arrays)
    from hpc_runs.first_hit_control.detectors import detector_hits
    from hpc_runs.first_hit_control.stationary_runtime_adapter import stationary_replay_batch
    result = fit_replay(model, data, FitConfig(updates=2, batch_size=4, horizon=2),
        lambda dg: detector_hits(dg, "winner"), action_count=2, batch_builder=stationary_replay_batch,
        target_factory=lambda: construct_stationary(snapshot, source))
    assert result["initial_readout_sha256"] != result["final_readout_sha256"]


@pytest.mark.parametrize("change", ["geometry", "source_hash", "suffix"])
def test_certificate_fail_closed_without_dummy_value_heads(source_replay, change):
    snapshot = copy.deepcopy(adapter_snapshot(source_replay))
    cert = snapshot["eligibility"]
    if change == "geometry":
        cert["effective_flags"]["hrl_landmark_geometry"] = "se2"
    elif change == "source_hash":
        cert["source_sha256"][next(iter(cert["source_sha256"]))] = "0" * 64
    else:
        cert["half_open_slices"]["physical_bypass"][1] -= 1
    with pytest.raises(ValueError):
        construct_stationary(snapshot, source_replay[0])


def test_actual_native_shape_has_no_added_parameters(source_replay):
    from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
    from sf_working_directories.IntrMotiv.dmlab.custom_decoder import MlpDecoderJit
    source = source_replay[0]
    cfg = SimpleNamespace(Hippo_R=8, Hippo_L=64, Hippo_n_feature=16, hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer", hrl_target_timing="immediate", hrl_manager_mode="visit_direct",
        hrl_goal_conditioning="legacy", dg_goal_input="none", decoder_mlp_layers=[128, 128],
        nonlinearity="relu", use_jit=False)
    core = SimpleSequenceWithBypassCore(cfg, 29)
    decoder = MlpDecoderJit(cfg, core.get_out_size())
    cert = certify_slice_only_relabel(core, decoder, source_root=source, input_kind="canonical_core_output")
    snapshot = dict(layout=cert["layout"], decoder_config=dict(decoder_mlp_layers=[128, 128], nonlinearity="relu", use_jit=False),
                    eligibility=cert, action_count=5, fit_config={"horizon": 256}, gamma=.99)
    model = construct_stationary(snapshot, source)
    assert cert["layout"] == dict(width=1165, goal_start=1149, goals=16)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 166405
    assert set(model.action_parameterization.state_dict()) == {"distribution_linear.weight", "distribution_linear.bias"}


def test_restore_arithmetic_archive_and_objective_are_bound(source_replay, tmp_path):
    source, replay = source_replay[:2]
    output = tmp_path / "fit"
    run_fit(replay, output, source, FitConfig(updates=1, batch_size=4, horizon=2))
    relocated = tmp_path / "source"
    for relative in RUNTIME_FILES:
        destination = relocated / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    load_fitted_stationary(output, relocated)
    path = relocated / "sample_factory/algo/utils/action_distributions.py"
    path.write_text(path.read_text() + "\n# arithmetic changed\n")
    with pytest.raises(ValueError, match="arithmetic.*action_distributions"):
        load_fitted_stationary(output, relocated)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["gamma"] = .98
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="objective/layout"):
        load_fitted_stationary(output, source)
    metadata["gamma"] = .99
    metadata_path.write_text(json.dumps(metadata))
    manifest = json.loads((output / "source_snapshot/manifest.json").read_text())
    (output / "source_snapshot" / manifest["sources"][0]["snapshot_path"]).write_text("corrupt")
    with pytest.raises(ValueError, match="source archive"):
        load_fitted_stationary(output, source)


def test_fit_default_hook_is_numerically_identical_and_records_native_batch_stream():
    from hpc_runs.test_first_hit_replay_fit import model, replay
    config = FitConfig(updates=4, batch_size=5, horizon=2, target_update_period=2)
    torch.manual_seed(41)
    first, second, data = model(), model(), replay()
    second.load_state_dict(first.state_dict())
    result = fit_replay(first, data, config, all_goal_hits, target_factory=model)
    explicit = fit_replay(second, data, config, all_goal_hits, target_factory=model,
                          action_count=2, batch_builder=replay_batch)
    assert state_digest(first) == state_digest(second)
    assert result["trace"] == explicit["trace"]
    assert result["batch_provenance"] == explicit["batch_provenance"]
    train, _ = episode_split(data["episode_id"], config.holdout_fraction, config.seed)
    train = train[data["successor_valid"][train]]
    rng = np.random.default_rng(config.seed)
    rows, clocks = [], []
    for _ in range(config.updates):
        rows.extend(rng.choice(train, config.batch_size, replace=True))
        clocks.extend(rng.integers(1, config.horizon + 1, config.batch_size))
    assert result["batch_provenance"]["factual_indices_sha256"] == hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()
    assert result["batch_provenance"]["consumed_deadlines_sha256"] == hashlib.sha256(np.asarray(clocks, dtype="<i8").tobytes()).hexdigest()


def test_plan_checks_budgets_and_replay_pins_before_fit(source_replay, tmp_path):
    _, replay, _, metadata, _, _ = source_replay
    config = FitConfig(updates=3, batch_size=4, horizon=2)
    plan = {"schema": "intrmotiv/reused-replay-control-plan/v1", "analysis_id": "fixture", "workflow_version": "fixture",
        "fit": dict(seed=99, updates=3, batch_size=4, horizon=2, learning_rate=.0003, target_update_period=100, holdout_fraction=.1),
        "replay": dict(dataset_sha256=metadata["dataset_sha256"], metadata_sha256=file_hash(replay / "metadata.json"),
                       decoder_state_sha256=file_hash(replay / "decoder_state.pt")),
        "objective": {"gamma": .99}, "donor": dict(checkpoint_sha256=metadata["checkpoint_sha256"], config_sha256=metadata["source_config_sha256"])}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert validate_plan(path, metadata, config, .99, replay)["phase"] == "full"
    with pytest.raises(ValueError, match="updates"):
        validate_plan(path, metadata, FitConfig(updates=4, batch_size=4, horizon=2), .99, replay)
    with pytest.raises(ValueError, match="objective"):
        validate_plan(path, metadata, config, .98, replay)


def test_production_entry_requires_canonical_validation_and_exact_release(monkeypatch, tmp_path):
    from hpc_runs.first_hit_control import stationary_replay_cli as cli, reused_plan
    called = []
    source = tmp_path / "declared_source"
    plan_path = tmp_path / "plan.json"
    validation = {"effective_plan_sha256": "a" * 64, "donor_study_sha256": "b" * 64}
    def load(path):
        called.append(("validate", path))
        return {"runtime": {"source_release": str(source)}, "fit": {"device": "cpu", "threads": 4}}, validation
    monkeypatch.setattr(reused_plan, "load_plan", load)
    monkeypatch.setattr(cli, "run_fit", lambda *args, **kwargs: called.append(("fit", kwargs)))
    args = ["stationary_replay_cli", "--replay", str(tmp_path / "replay"), "--output", str(tmp_path / "out"),
            "--source-root", str(source), "--analysis-plan", str(plan_path)]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert [row[0] for row in called] == ["validate", "fit"]
    assert called[1][1]["canonical_plan_validation"] == validation
    args[6] = str(tmp_path / "wrong_source")  # value after --source-root
    called.clear()
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="source root"):
        cli.main()
    assert [row[0] for row in called] == ["validate"]
