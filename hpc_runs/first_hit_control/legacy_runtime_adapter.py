"""First-hit acting on a certified source legacy decoder, without replacing it."""

import copy

from torch import nn

from hpc_runs.first_hit_control.eligibility import certify_slice_only_relabel, validate_legacy_certificate
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
        validate_legacy_certificate(decoder, certificate, source_root=source_root)
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
