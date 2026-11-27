"""Opt-in experiment adapter reusing the actual IntrMotiv decoder and Q head.

This is an offline qualification boundary, not a production actor patch. It
copies the supplied decoder/head, detaches physical inputs, and explicitly acts
through the existing deadline-conditioned auxiliary. Original policy modules,
CA3 state, goal manager, representation, and runtime files are never modified.
The caller must verify that every retained prefix/suffix is goal-independent.
Runtime variants with target-relative geometry or manager-mode suffixes need
their own audited conditioning path; this slice-only adapter does not qualify
counterfactual relabeling of those variants.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GoalLayout:
    """Declared goal slice in a complete existing controller input vector."""

    width: int
    goal_start: int
    goals: int

    def __post_init__(self):
        if self.goals < 1 or self.goal_start < 0 or self.goal_start + self.goals > self.width:
            raise ValueError("Goal slice must be contained in the core output")


def fixed_goal_input(core_output: Tensor, goal_ids: Tensor, layout: GoalLayout) -> Tensor:
    """Copy immutable physical inputs and replace ONLY the declared goal slice."""
    if core_output.ndim != 2 or core_output.shape[1] != layout.width:
        raise ValueError("core_output width does not match declared layout")
    if not core_output.is_floating_point() or not torch.isfinite(core_output).all():
        raise ValueError("core_output must be finite floating point")
    if goal_ids.shape != (core_output.shape[0],) or goal_ids.is_floating_point() or goal_ids.dtype == torch.bool:
        raise ValueError("goal_ids must be an integer [batch] tensor")
    if goal_ids.device != core_output.device or ((goal_ids < 0) | (goal_ids >= layout.goals)).any():
        raise ValueError("goal_ids must be valid identities on the input device")
    result = core_output.detach().clone()
    goal = result[:, layout.goal_start : layout.goal_start + layout.goals]
    goal.zero_().scatter_(1, goal_ids.long()[:, None], 1)
    return result


def runtime_factories(source_root: str | Path) -> tuple[Callable, Callable, dict]:
    """Load exact source files, with path/hash provenance and no copied classes.

    Runtime dependencies must already be importable in the chosen environment.
    The decoder file imports Sample Factory, so this fails rather than silently
    substituting a hand-written decoder when that environment is unavailable.
    """
    root = Path(source_root).resolve(strict=True)
    directory = root / "sf_working_directories/IntrMotiv/dmlab"
    modules, provenance = [], {"source_root": str(root), "files": {}}
    for name in ("custom_actor_critic", "controller_q"):
        path = (directory / f"{name}.py").resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(f"intrmotiv_first_hit_{name}_{digest[:12]}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load runtime source {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
        provenance["files"][str(path)] = digest
    return modules[0].TargetFiLMDecoder, modules[1].ControllerQHeads, provenance


def create_runtime_components(
    state_width: int,
    goal_count: int,
    action_count: int,
    *,
    hidden_size: int = 128,
    source_root: str | Path,
) -> tuple[nn.Module, nn.Module, GoalLayout, dict]:
    """Instantiate unchanged runtime classes for a state-prefix/goal-suffix toy.

    This convenience layout contains no bypass suffix. For a real checkpoint,
    construct the adapter from its actual decoder and actual declared layout.
    """
    if min(state_width, goal_count, action_count, hidden_size) < 1:
        raise ValueError("Component sizes must be positive")
    decoder_class, heads_class, provenance = runtime_factories(source_root)
    layout = GoalLayout(state_width + goal_count, state_width, goal_count)
    core = SimpleNamespace(
        Hippo_n_feature=goal_count,
        target_condition_start=state_width,
        get_out_size=lambda: layout.width,
        cfg=SimpleNamespace(fixed_task_goal_mixture=False),
    )
    decoder = decoder_class(core, hidden_size=hidden_size)
    heads = heads_class(hidden_size, action_count, auxiliary=True)
    return decoder, heads, layout, provenance


class FixedGoalFirstHitAdapter(nn.Module):
    """Independent trainable readout on frozen existing physical state.

    The original decoder and Q modules are deep-copied, so adapting this readout
    cannot mutate a pretrained policy. All source inputs are detached. The
    existing main continuing-return head is retained for parity but frozen and
    unused. The auxiliary's deadline interactions are used explicitly as Q_h.
    The outputs are raw regression values, not certified probabilities.
    """

    def __init__(self, decoder: nn.Module, q_heads: nn.Module, layout: GoalLayout, horizon_scale: int):
        super().__init__()
        if not isinstance(horizon_scale, int) or isinstance(horizon_scale, bool) or horizon_scale < 1:
            raise ValueError("horizon_scale must be a positive integer")
        if getattr(q_heads, "auxiliary", None) is None:
            raise ValueError("Existing deadline-conditioned auxiliary is required")
        if getattr(decoder, "target_condition_start", None) != layout.goal_start:
            raise ValueError("Decoder goal offset differs from declared layout")
        if getattr(decoder, "n_targets", None) != layout.goals:
            raise ValueError("Decoder goal count differs from declared layout")
        if q_heads.main.in_features != decoder.get_out_size():
            raise ValueError("Decoder and Q hidden widths differ")
        self.decoder = copy.deepcopy(decoder)
        self.q_heads = copy.deepcopy(q_heads)
        # Frozen checkpoint loaders disable every source parameter. This fresh
        # readout copy must explicitly restore only its intended trainable path.
        self.decoder.requires_grad_(True)
        self.q_heads.auxiliary.requires_grad_(True)
        self.q_heads.main.requires_grad_(False)
        self.layout, self.horizon_scale = layout, horizon_scale

    def forward(self, core_output: Tensor, goal_ids: Tensor, remaining: Tensor) -> Tensor:
        if remaining.shape != (core_output.shape[0],):
            raise ValueError("remaining must have shape [batch]")
        if remaining.is_floating_point() or remaining.dtype == torch.bool or (remaining < 0).any():
            raise ValueError("remaining must contain nonnegative integer deadlines")
        if remaining.device != core_output.device:
            raise ValueError("remaining must share the input device")
        hidden = self.decoder(fixed_goal_input(core_output, goal_ids, self.layout))
        values = self.q_heads.hindsight(hidden, remaining, self.horizon_scale)
        return torch.where(remaining[:, None] > 0, values, torch.zeros_like(values))

    def all_goals(self, core_output: Tensor, remaining: Tensor) -> Tensor:
        """Enumerate the declared goals independently of transition outcomes."""
        batch = len(core_output)
        ids = torch.arange(self.layout.goals, device=core_output.device).repeat(batch)
        expanded = core_output.repeat_interleave(self.layout.goals, dim=0)
        clocks = remaining.repeat_interleave(self.layout.goals)
        return self(expanded, ids, clocks).reshape(batch, self.layout.goals, -1)

    def greedy_actions(self, core_output: Tensor, goal_ids: Tensor, remaining: Tensor, action_mask=None) -> Tensor:
        """Select from Q_h; h=0 is a stop boundary, never an executable action."""
        if (remaining <= 0).any():
            raise ValueError("Expired deadlines cannot select an action")
        values = self(core_output, goal_ids, remaining)
        if action_mask is not None:
            if action_mask.shape != values.shape or action_mask.dtype != torch.bool:
                raise ValueError("action_mask must be boolean [batch, actions]")
            if not action_mask.any(-1).all():
                raise ValueError("No supported action for a requested decision")
            values = values.masked_fill(~action_mask, -torch.inf)
        return values.argmax(-1)
