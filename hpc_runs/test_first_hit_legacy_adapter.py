from types import SimpleNamespace
from pathlib import Path
import copy
import os

import torch
import pytest

from hpc_runs.first_hit_control.detectors import detector_hits
from hpc_runs.first_hit_control.legacy_runtime_adapter import CertifiedLegacyFirstHitAdapter
from hpc_runs.first_hit_control.runtime_adapter import fixed_goal_input, runtime_factories


def test_source_detector_predicates_preserve_multi_activation_and_ties():
    dg = torch.tensor([[0., 0., 0.], [2., 3., 0.], [2., 2., 0.], [0., 0., 4.]])
    assert detector_hits(dg, "winner").tolist() == [[False, False, False], [False, True, False], [True, False, False], [False, False, True]]
    assert detector_hits(dg, "exclusive").sum().item() == 1
    assert detector_hits(dg, "positive").sum().item() == 5
    with pytest.raises(ValueError):
        detector_hits(dg, "implicit")


def test_original_legacy_decoder_and_head_are_reused_without_source_mutation():
    from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
    from sf_working_directories.IntrMotiv.dmlab.custom_decoder import MlpDecoderJit

    source = Path(os.environ.get("INTRMOTIV_RUNTIME_SOURCE", "/home/xiaoxiong/SFgit/SF_hipposlam"))
    cfg = SimpleNamespace(
        Hippo_R=8, Hippo_L=64, Hippo_n_feature=16, hrl_controllable_graph=True,
        hrl_graph_memory="policy_buffer", hrl_target_timing="immediate", hrl_manager_mode="visit_direct",
        hrl_goal_conditioning="legacy", dg_goal_input="none", decoder_mlp_layers=[128, 128],
        nonlinearity="relu", use_jit=False,
    )
    core = SimpleSequenceWithBypassCore(cfg, 26)
    decoder = MlpDecoderJit(cfg, core.get_out_size()).eval().requires_grad_(False)
    _, q_class, _ = runtime_factories(source)
    q = q_class(128, 5, auxiliary=True).eval().requires_grad_(False)
    adapter = CertifiedLegacyFirstHitAdapter(core, decoder, q, 64, source_root=source)
    state = torch.randn(3, core.get_out_size(), requires_grad=True)
    goals, h = torch.tensor([0, 5, 15]), torch.tensor([1, 16, 64])
    expected = q.hindsight(decoder(fixed_goal_input(state, goals, adapter.layout)), h, 64)
    assert torch.equal(expected, adapter(state, goals, h))
    assert type(adapter.decoder).__name__ == "MlpDecoderJit"
    adapter(state, goals, h).sum().backward()
    assert state.grad is None and all(p.grad is None and not p.requires_grad for p in decoder.parameters())
    assert all(p.grad is not None and p.requires_grad for p in adapter.decoder.parameters())
    restored = CertifiedLegacyFirstHitAdapter.from_certificate(
        decoder, q, adapter.eligibility, 64, source_root=source
    )
    assert torch.equal(expected, restored(state, goals, h))
    # Relocation uses the source-relative filename and content hash, not the
    # collector host's original absolute checkout path.
    moved = copy.deepcopy(adapter.eligibility)
    moved["source_sha256"] = {k.replace(str(source), "/old/collector/source"): v
                              for k, v in moved["source_sha256"].items()}
    assert torch.equal(expected, CertifiedLegacyFirstHitAdapter.from_certificate(
        decoder, q, moved, 64, source_root=source
    )(state, goals, h))
    for mutation in ("hash", "layout", "flags", "schema"):
        broken = copy.deepcopy(adapter.eligibility)
        if mutation == "hash":
            broken["source_sha256"][next(iter(broken["source_sha256"]))] = "0" * 64
        elif mutation == "layout":
            broken["layout"]["width"] += 1
        elif mutation == "flags":
            broken["effective_flags"]["dg_goal_input"] = "target_trace"
        else:
            broken["schema"] = "unverified"
        with pytest.raises(ValueError):
            CertifiedLegacyFirstHitAdapter.from_certificate(decoder, q, broken, 64, source_root=source)
