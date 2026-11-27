"""Certified stationary first-hit readout using only the original actor layers.

Remaining time is an evaluator-interface argument, never a neural input or
target boundary. The physical source core stays frozen; only independent
copies of its existing legacy decoder and categorical action head are trained.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from .detectors import detector_hits
from .eligibility import validate_legacy_certificate
from .runtime_adapter import fixed_goal_input
from .targets import first_hit_targets


KIND = "stationary_projected_first_hit"


class StationaryFirstHitAdapter(nn.Module):
    """Own newly instantiated source-class modules, without an auxiliary head.

    Callers must instantiate independent modules and load both original state
    dictionaries. Taking ownership rather than deep-copying also avoids JIT's
    nonleaf-parameter deepcopy problem when constructing a target network.
    """

    def __init__(self, decoder, action_parameterization, certificate, horizon_scale,
                 *, source_root, gamma=.99):
        super().__init__()
        self.layout = validate_legacy_certificate(decoder, certificate, source_root=source_root)
        if not isinstance(horizon_scale, int) or isinstance(horizon_scale, bool) or horizon_scale < 1:
            raise ValueError("horizon_scale is a positive external evaluation deadline")
        if not 0 < gamma < 1:
            raise ValueError("Stationary gamma must be between zero and one")
        expected = Path(source_root).resolve() / "sample_factory/model/action_parameterization.py"
        code = getattr(type(action_parameterization).__init__, "__code__", None)
        if (type(action_parameterization).__name__ != "ActionParameterizationDefault" or code is None
                or Path(code.co_filename).resolve() != expected.resolve()):
            raise ValueError("Require the exact original ActionParameterizationDefault source class")
        linear = action_parameterization.distribution_linear
        if linear.in_features != decoder.get_out_size() or linear.out_features < 1:
            raise ValueError("Original action head and decoder dimensions disagree")
        self.decoder = decoder.requires_grad_(True)
        self.action_parameterization = action_parameterization.requires_grad_(True)
        self.action_count = int(linear.out_features)
        self.horizon_scale = horizon_scale
        self.gamma = float(gamma)
        self.kind = KIND
        self.eligibility = copy.deepcopy(certificate)
        self.ca3_width = certificate["half_open_slices"]["canonical_ca3"][1]
        self.expanded_length = self.ca3_width // self.layout.goals

    def forward(self, core, goals, remaining=None):
        # In particular h=0 does NOT zero the stationary value. The evaluator
        # stops externally, and every live replay transition bootstraps.
        hidden = self.decoder(fixed_goal_input(core, goals, self.layout))
        raw, _ = self.action_parameterization(hidden)
        return raw

    def all_goals(self, core, remaining=None):
        batch = len(core)
        goals = torch.arange(self.layout.goals, device=core.device).repeat(batch)
        values = self(core.repeat_interleave(self.layout.goals, dim=0), goals)
        return values.reshape(batch, self.layout.goals, self.action_count)

    def act(self, core, goals, remaining=None):
        return self(core, goals, remaining).argmax(-1)

    def current_dg(self, core):
        """The certified core's detector slot, not raw head DG or source_goal."""
        return core[:, :self.ca3_width].reshape(-1, self.layout.goals, self.expanded_length)[:, :, 0]


def stationary_replay_batch(model, target, data, indices, remaining, detector):
    """Projected discounted Double-Q labels on factual actions and all goals.

    Hit precedes physical termination; unknown successors stay unknown. A
    known hit/terminal needs no successor core. Current-goal masking always
    uses the canonical positive argmax winner, including multi-activation.
    ``remaining`` is deliberately ignored; the shared sampler still draws it.
    """
    device = next(model.parameters()).device
    if model.gamma != target.gamma:
        raise ValueError("Online and target stationary discounts differ")
    to_tensor = lambda key: torch.as_tensor(data[key][indices], device=device)
    current, successor = to_tensor("core").float(), to_tensor("successor_core").float()
    done, valid, core_valid = (to_tensor(key) for key in
                              ("physical_done", "successor_valid", "successor_core_valid"))
    hits = detector(to_tensor("successor_dg").float())
    with torch.no_grad():
        online_next = model.all_goals(successor)
        target_next = target.all_goals(successor)
        # Do not let clipping conceal NaNs/infinities in a corrupt raw head.
        if not torch.isfinite(target_next).all():
            raise FloatingPointError("Nonfinite raw stationary target values")
        # Sentinel 2 enables the existing target helper's live bootstrap. It is
        # not the external deadline and never enters a neural input.
        sentinel = torch.full((len(indices),), 2, dtype=torch.long, device=device)
        labels, usable = first_hit_targets(hits, done, sentinel, online_next,
            model.gamma * target_next.clamp(0, 1), successor_valid=valid)
        needs_core = ~done[:, None] & ~hits
        usable &= ~needs_core | core_valid[:, None]
        usable &= ~detector_hits(model.current_dg(current), "winner")
        labels = labels.masked_fill(~usable, torch.nan)
    values = model.all_goals(current)
    actions = to_tensor("action").long()
    selected = values.gather(-1, actions[:, None, None].expand(-1, model.layout.goals, 1)).squeeze(-1)
    return selected, labels, usable
