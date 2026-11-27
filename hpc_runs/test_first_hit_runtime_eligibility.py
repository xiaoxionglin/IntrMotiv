"""Structural guards against relabeling a goal-dependent stored worker state."""

from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from hpc_runs.first_hit_control.eligibility import certify_slice_only_relabel
from hpc_runs.first_hit_control.runtime_adapter import runtime_factories


SOURCE = Path(os.environ.get("INTRMOTIV_RUNTIME_SOURCE", "/home/xiaoxiong/SFgit/SF_hipposlam"))


@pytest.fixture
def actual_core_decoder():
    if not SOURCE.is_dir():
        pytest.skip("Desktop runtime checkout unavailable")
    from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore

    cfg = SimpleNamespace(
        Hippo_R=8, Hippo_L=64, Hippo_n_feature=16, hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer", hrl_target_timing="immediate", hrl_manager_mode="visit_direct",
        hrl_goal_conditioning="target_id_film", dg_goal_input="none",
    )
    core = SimpleSequenceWithBypassCore(cfg, 26)
    decoder_class, _, _ = runtime_factories(SOURCE)
    return core, decoder_class(core)


def test_actual_scalar_core_layout_is_structurally_eligible(actual_core_decoder):
    core, decoder = actual_core_decoder
    result = certify_slice_only_relabel(core, decoder, source_root=SOURCE, input_kind="canonical_core_output")
    assert result["half_open_slices"] == {"canonical_ca3": [0, 1136], "physical_bypass": [1136, 1146], "goal": [1146, 1162]}
    assert result["structural_eligibility"] and len(result["remaining_gates"]) == 3


@pytest.mark.parametrize("field,value,flag", [
    ("landmark_geometry", "se2", "hrl_landmark_geometry"),
    ("behavior_mode_condition", True, "hrl_behavior_mode_condition"),
    ("motion_policy_input", True, "hrl_motion_policy_input"),
    ("action_path_integration", True, "hrl_action_path_integration"),
    ("readout_mode", "worker", "ca3_state_readout_mode"),
    ("worker_goal_mode", "raw_ca3", "ca3_worker_goal_mode"),
    ("context_feedback_mode", "additive", "dg_context_feedback"),
])
def test_rejects_unqualified_goal_and_context_variants(actual_core_decoder, field, value, flag):
    core, decoder = actual_core_decoder
    setattr(core, field, value)
    with pytest.raises(ValueError, match=flag):
        certify_slice_only_relabel(core, decoder, source_root=SOURCE, input_kind="canonical_core_output")


def test_rejects_goal_write_and_worker_cache(actual_core_decoder):
    core, decoder = actual_core_decoder
    core.cfg.dg_goal_input = "write"
    with pytest.raises(ValueError, match="dg_goal_input"):
        certify_slice_only_relabel(core, decoder, source_root=SOURCE, input_kind="canonical_core_output")
    core.cfg.dg_goal_input = "none"
    with pytest.raises(ValueError, match="unverified"):
        certify_slice_only_relabel(core, decoder, source_root=SOURCE, input_kind="controller_worker_state")


def test_rejects_undeclared_suffix(actual_core_decoder):
    core, decoder = actual_core_decoder
    core.total_output_size += 4
    with pytest.raises(ValueError, match="suffix"):
        certify_slice_only_relabel(core, decoder, source_root=SOURCE, input_kind="canonical_core_output")
