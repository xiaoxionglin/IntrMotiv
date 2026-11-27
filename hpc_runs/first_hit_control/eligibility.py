"""Fail-closed structural eligibility for slice-only goal relabeling.

This does not certify a trained checkpoint. It identifies the one existing
physical-state layout whose conditioning semantics the adapter supports. Real
collection must additionally prove frozen DG/BatchNorm and actor/replay parity.
Unsupported variants require a separately verified reconstruction, not a flag
that bypasses these checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hpc_runs.first_hit_control.runtime_adapter import GoalLayout


def certify_slice_only_relabel(core, decoder, *, source_root: str | Path, input_kind: str) -> dict:
    """Check an actual unchanged runtime core/decoder and report exact slices.

    Only full ``canonical_core_output`` is accepted. Stored worker state, raw
    recurrent state, and auxiliary worker views are not interchangeable with it.
    This first qualification supports SimpleSequenceWithBypassCore with a
    TargetFiLMDecoder or its original MlpDecoderJit, ordinary scalar CA3,
    unchanged bypass, and one goal suffix. The decoder type must agree with cfg.
    """
    if input_kind != "canonical_core_output":
        raise ValueError("Require canonical_core_output; stored worker/recurrent states are unverified")
    cfg = getattr(core, "cfg", None)
    if cfg is None:
        raise ValueError("Actual runtime core configuration is required")
    effective = {
        "dg_goal_input": getattr(cfg, "dg_goal_input", "none"),
        "hrl_landmark_geometry": getattr(core, "landmark_geometry", getattr(cfg, "hrl_landmark_geometry", "none")),
        "hrl_behavior_mode_condition": bool(getattr(core, "behavior_mode_condition", getattr(cfg, "hrl_behavior_mode_condition", False))),
        "hrl_motion_policy_input": bool(getattr(core, "motion_policy_input", getattr(cfg, "hrl_motion_policy_input", False))),
        "hrl_action_path_integration": bool(getattr(core, "action_path_integration", getattr(cfg, "hrl_action_path_integration", False))),
        "ca3_state_readout_mode": getattr(core, "readout_mode", getattr(cfg, "ca3_state_readout_mode", "off")),
        "ca3_worker_goal_mode": getattr(core, "worker_goal_mode", getattr(cfg, "ca3_worker_goal_mode", "target_id")),
        "dg_context_feedback": getattr(core, "context_feedback_mode", getattr(cfg, "dg_context_feedback", "none")),
        "dg_orthogonal_recruitment": bool(getattr(cfg, "dg_orthogonal_recruitment", False)),
    }
    expected = {
        "dg_goal_input": "none", "hrl_landmark_geometry": "none", "hrl_behavior_mode_condition": False,
        "hrl_motion_policy_input": False, "hrl_action_path_integration": False, "ca3_state_readout_mode": "off",
        "ca3_worker_goal_mode": "target_id", "dg_context_feedback": "none", "dg_orthogonal_recruitment": False,
    }
    violations = [f"{key}={effective[key]!r}" for key in expected if effective[key] != expected[key]]
    if violations:
        raise ValueError("Unqualified goal/state reconstruction: " + ", ".join(violations))
    root = Path(source_root).resolve(strict=True)
    paths = {}
    conditioning = getattr(cfg, "hrl_goal_conditioning", "legacy")
    if conditioning not in ("legacy", "target_id_film"):
        raise ValueError(f"Unverified hrl_goal_conditioning={conditioning!r}")
    decoder_name, decoder_file = (
        ("MlpDecoderJit", "custom_decoder.py") if conditioning == "legacy"
        else ("TargetFiLMDecoder", "custom_actor_critic.py")
    )
    for instance, class_name, filename in (
        (core, "SimpleSequenceWithBypassCore", "custom_core.py"),
        (decoder, decoder_name, decoder_file),
    ):
        if type(instance).__name__ != class_name:
            raise ValueError(f"Unverified runtime class {type(instance).__name__}; require {class_name}")
        code = getattr(type(instance).__init__, "__code__", None)
        actual = Path(code.co_filename).resolve(strict=True) if code is not None else None
        expected_path = (root / "sf_working_directories/IntrMotiv/dmlab" / filename).resolve(strict=True)
        if actual != expected_path:
            raise ValueError(f"Runtime source mismatch for {class_name}")
        paths[str(actual)] = hashlib.sha256(actual.read_bytes()).hexdigest()
    width, start, goals = int(core.get_out_size()), int(core.target_condition_start), int(core.Hippo_n_feature)
    layout = GoalLayout(width, start, goals)
    if width != start + goals:
        raise ValueError("Unverified suffix after goal slice; geometry/mode/history reconstruction required")
    ca3 = int(core.core_output_size)
    if ca3 != goals * int(core.expanded_length) or int(core.base_state_size) != start:
        raise ValueError("Unverified scalar CA3/bypass layout")
    if conditioning == "target_id_film":
        if decoder.target_condition_start != start or decoder.n_targets != goals:
            raise ValueError("Core and decoder disagree about the goal slice")
    elif decoder.core_input_size != width:
        raise ValueError("Original legacy decoder input width differs from complete core output")
    return {
        "schema": "intrmotiv/slice-only-relabel-eligibility/v1",
        "structural_eligibility": True,
        "input_kind": input_kind,
        "core_class": type(core).__name__, "decoder_class": type(decoder).__name__,
        "hrl_goal_conditioning": conditioning,
        "layout": {"width": layout.width, "goal_start": layout.goal_start, "goals": layout.goals},
        "half_open_slices": {"canonical_ca3": [0, ca3], "physical_bypass": [ca3, start], "goal": [start, width]},
        "effective_flags": effective,
        "source_sha256": paths,
        "remaining_gates": ["Frozen source DG weights and BatchNorm buffers", "Actual actor/replay state parity", "Genuine terminal successor transport"],
        "unsupported_state_note": "GoalConditionedDGCore splits worker memory at canonical_state_size-behavior_goal_state_size; that goal-written worker block is not canonical scalar CA3. The controller_worker_state cache alone does not certify its meaning.",
    }


def validate_legacy_certificate(decoder, certificate, *, source_root) -> GoalLayout:
    """Verify a saved scalar-CA3 legacy layout and exact source decoder.

    Shared by finite-deadline and original-action-head stationary adapters;
    no value head or native environment is required to validate eligibility.
    """
    required = {
        "schema": "intrmotiv/slice-only-relabel-eligibility/v1",
        "structural_eligibility": True, "input_kind": "canonical_core_output",
        "core_class": "SimpleSequenceWithBypassCore", "decoder_class": "MlpDecoderJit",
        "hrl_goal_conditioning": "legacy",
    }
    if any(certificate.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported or incomplete collector eligibility certificate")
    flags = {
        "dg_goal_input": "none", "hrl_landmark_geometry": "none", "hrl_behavior_mode_condition": False,
        "hrl_motion_policy_input": False, "hrl_action_path_integration": False, "ca3_state_readout_mode": "off",
        "ca3_worker_goal_mode": "target_id", "dg_context_feedback": "none", "dg_orthogonal_recruitment": False,
    }
    if certificate.get("effective_flags") != flags:
        raise ValueError("Certificate contains unqualified goal-dependent physical inputs")
    layout_data = certificate.get("layout", {})
    if set(layout_data) != {"width", "goal_start", "goals"} or any(type(v) is not int for v in layout_data.values()):
        raise ValueError("Certificate requires an integer complete core layout")
    layout = GoalLayout(**layout_data)
    slices = certificate.get("half_open_slices", {})
    ca3 = slices.get("canonical_ca3", [])
    if (layout.width != layout.goal_start + layout.goals or len(ca3) != 2 or ca3[0] != 0
            or type(ca3[1]) is not int or not 0 < ca3[1] <= layout.goal_start or ca3[1] % layout.goals
            or slices.get("physical_bypass") != [ca3[1], layout.goal_start]
            or slices.get("goal") != [layout.goal_start, layout.width]):
        raise ValueError("Certificate contains inconsistent physical/goal slices")
    directory = Path(source_root).resolve(strict=True) / "sf_working_directories/IntrMotiv/dmlab"
    recorded = certificate.get("source_sha256", {})
    for name in ("custom_core.py", "custom_decoder.py"):
        matches = [digest for path, digest in recorded.items()
                   if str(path).endswith("/sf_working_directories/IntrMotiv/dmlab/" + name)]
        if len(matches) != 1 or hashlib.sha256((directory / name).read_bytes()).hexdigest() != matches[0]:
            raise ValueError(f"Certified source hash mismatch for {name}")
    code = getattr(type(decoder).__init__, "__code__", None)
    if (type(decoder).__name__ != "MlpDecoderJit" or code is None
            or Path(code.co_filename).resolve(strict=True) != (directory / "custom_decoder.py").resolve(strict=True)
            or decoder.core_input_size != layout.width):
        raise ValueError("Supplied decoder differs from the certified original legacy decoder")
    return layout
