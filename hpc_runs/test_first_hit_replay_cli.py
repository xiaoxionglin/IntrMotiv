"""A model-only round trip of the actual source legacy readout and certificates."""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.eligibility import certify_slice_only_relabel
from hpc_runs.first_hit_control.replay_cli import file_hash, load_fitted_adapter, run_fit
from hpc_runs.first_hit_control.replay_fit import FitConfig


@pytest.fixture
def source_replay(tmp_path):
    from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
    from sf_working_directories.IntrMotiv.dmlab.custom_decoder import MlpDecoderJit

    source = Path(os.environ.get("INTRMOTIV_RUNTIME_SOURCE", "/home/xiaoxiong/SFgit/SF_hipposlam"))
    if not source.is_dir():
        pytest.skip("Runtime source unavailable")
    cfg = SimpleNamespace(Hippo_R=2, Hippo_L=3, Hippo_n_feature=3, hrl_controllable_graph=True,
                          hrl_graph_memory="policy_buffer", hrl_target_timing="immediate",
                          hrl_manager_mode="visit_direct", hrl_goal_conditioning="legacy", dg_goal_input="none",
                          decoder_mlp_layers=[16, 16], nonlinearity="relu", use_jit=False)
    core = SimpleSequenceWithBypassCore(cfg, 5)
    decoder = MlpDecoderJit(cfg, core.get_out_size())
    certificate = certify_slice_only_relabel(core, decoder, source_root=source, input_kind="canonical_core_output")
    path = tmp_path / "replay"
    path.mkdir()
    count, width = 16, core.get_out_size()
    np.savez_compressed(path / "replay.npz", core=np.zeros((count, width), np.float32),
                        successor_core=np.zeros((count, width), np.float32), action=np.arange(count) % 2,
                        successor_dg=np.eye(3, dtype=np.float32)[np.arange(count) % 3],
                        physical_done=np.ones(count, bool), successor_valid=np.ones(count, bool),
                        successor_core_valid=np.zeros(count, bool), episode_id=np.arange(count))
    torch.save({"decoder_state_dict": decoder.state_dict(), "decoder_class": "MlpDecoderJit",
                "decoder_config": {name: getattr(cfg, name) for name in ("decoder_mlp_layers", "nonlinearity", "use_jit")},
                "layout": certificate["layout"], "action_count": 2}, path / "decoder_state.pt")
    metadata = {"schema": "intrmotiv/frozen-first-hit-replay/v1", "model_unchanged": True,
                "model_sha256_before": "unit_fixture", "model_sha256_after": "unit_fixture",
                "native_reward_zero_every_step": True, "reset_observations_used_as_successors": 0,
                "primary_detector": "winner", "layout": certificate["layout"], "eligibility": certificate,
                "checkpoint_sha256": "fixture_not_a_DMLab_checkpoint", "source_run": "unit_fixture",
                "source_config_sha256": "a" * 64, "evaluation_overrides": {"transfer_freeze_dg": True},
                "dataset_sha256": file_hash(path / "replay.npz"), "decoder_state_sha256": file_hash(path / "decoder_state.pt")}
    (path / "metadata.json").write_text(json.dumps(metadata))
    return source, path


def test_actual_legacy_fit_reload_and_tamper_rejection(source_replay, tmp_path):
    source, replay = source_replay
    output = tmp_path / "fit"
    torch.set_num_threads(1)
    report = run_fit(replay, output, source, FitConfig(updates=3, batch_size=4, horizon=2, log_every=3))
    adapter, metadata = load_fitted_adapter(output, source)
    assert type(adapter.decoder).__name__ == "MlpDecoderJit"
    assert not adapter.training and all(not p.requires_grad for p in adapter.parameters())
    assert metadata["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert metadata["source_audit"]["archive_valid"]
    assert metadata["source_audit"]["current_sources_unchanged"]
    assert metadata["source_config_sha256"] == "a" * 64
    assert metadata["collection_evaluation_overrides"] == {"transfer_freeze_dg": True}
    assert metadata["fit"]["goal_labels_seen"] == 3 * 4 * 3
    with (output / "adapter.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_fitted_adapter(output, source)


def test_restore_rejects_changed_arithmetic_and_corrupt_archive(source_replay, tmp_path):
    source, replay = source_replay
    output = tmp_path / "fit"
    torch.set_num_threads(1)
    run_fit(replay, output, source, FitConfig(updates=1, batch_size=4, horizon=2))
    relocated = tmp_path / "source"
    target = relocated / "sf_working_directories/IntrMotiv/dmlab"
    target.mkdir(parents=True)
    for name in ("custom_core.py", "custom_decoder.py", "controller_q.py"):
        shutil.copy2(source / "sf_working_directories/IntrMotiv/dmlab" / name, target / name)
    load_fitted_adapter(output, relocated)
    with (target / "controller_q.py").open("a") as handle:
        handle.write("\n# changed source bytes\n")
    with pytest.raises(ValueError, match="Restored source.*controller_q"):
        load_fitted_adapter(output, relocated)
    manifest = json.loads((output / "source_snapshot/manifest.json").read_text())
    archived = output / "source_snapshot" / manifest["sources"][0]["snapshot_path"]
    archived.write_text("corrupted source archive")
    with pytest.raises(ValueError, match="source archive"):
        load_fitted_adapter(output, source)


def test_sum_zero_without_per_step_reward_certificate_is_rejected(source_replay, tmp_path):
    source, replay = source_replay
    path = replay / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.pop("native_reward_zero_every_step")
    metadata["native_reward_sum"] = 0.0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="per-step native"):
        run_fit(replay, tmp_path / "invalid", source, FitConfig(updates=1, horizon=2))
