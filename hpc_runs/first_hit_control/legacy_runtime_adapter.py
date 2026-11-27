"""First-hit acting on a certified source legacy decoder, without replacing it."""

import copy
import hashlib
from pathlib import Path

from torch import nn

from hpc_runs.first_hit_control.eligibility import certify_slice_only_relabel
from hpc_runs.first_hit_control.runtime_adapter import FixedGoalFirstHitAdapter, GoalLayout


class CertifiedLegacyFirstHitAdapter(FixedGoalFirstHitAdapter):
    """Reuse the original MlpDecoderJit and inherited first-hit query methods.

    Only the readout copies are trainable. No new FiLM/target-trace decoder is
    introduced. Core eligibility is required explicitly because this decoder
    itself does not carry goal-slice metadata.
    """

    def __init__(self, core, decoder, q_heads, horizon_scale: int, *, source_root):
        certificate = certify_slice_only_relabel(
            core, decoder, source_root=source_root, input_kind="canonical_core_output"
        )
        self._initialize(decoder, q_heads, certificate, horizon_scale)

    @classmethod
    def from_certificate(cls, decoder, q_heads, certificate, horizon_scale: int, *, source_root):
        """Restore a readout from a hash-verified collector eligibility record.

        This avoids reconstructing a native environment just to fit replay. The
        caller must also verify the replay/decoder snapshot hashes recorded by
        that collector. A structural certificate is not independent evidence
        that arbitrary replay tensors came from the certified physical core.
        Recorded absolute source paths are remapped beneath the supplied root,
        permitting an immutable source release to be moved between hosts.
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
        result = cls.__new__(cls)
        result._initialize(decoder, q_heads, certificate, horizon_scale)
        return result

    def _initialize(self, decoder, q_heads, certificate, horizon_scale):
        nn.Module.__init__(self)
        if certificate["decoder_class"] != "MlpDecoderJit":
            raise ValueError("This adapter requires the existing legacy decoder")
        if not isinstance(horizon_scale, int) or isinstance(horizon_scale, bool) or horizon_scale < 1:
            raise ValueError("horizon_scale must be a positive integer")
        if q_heads.auxiliary is None or q_heads.main.in_features != decoder.get_out_size():
            raise ValueError("Compatible existing deadline head is required")
        self.decoder = copy.deepcopy(decoder).requires_grad_(True)
        self.q_heads = copy.deepcopy(q_heads)
        self.q_heads.auxiliary.requires_grad_(True)
        self.q_heads.main.requires_grad_(False)
        self.layout = GoalLayout(**certificate["layout"])
        self.horizon_scale = horizon_scale
        self.eligibility = copy.deepcopy(certificate)
