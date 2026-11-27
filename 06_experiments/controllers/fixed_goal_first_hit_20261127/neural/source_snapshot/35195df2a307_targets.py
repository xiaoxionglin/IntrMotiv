"""Same-goal first-arrival backups from factual, outcome-independent replay.

Every transition is paired with the same complete ordered goal set. A missing
terminal observation is unknown, not a failed goal. Buffer boundaries are not
physical termination and must not be passed as ``physical_done``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def all_goal_hits(successor_dg: Tensor) -> Tensor:
    """Label every frozen DG detector; simultaneous active identities all hit.

    The agreed objective is entry into the detector set, not the older exclusive
    first-distinct option outcome. No future-achieved-goal sampling occurs here.
    """
    if successor_dg.ndim != 2 or successor_dg.shape[1] < 1:
        raise ValueError("successor_dg must have shape [batch, goals]")
    if not torch.isfinite(successor_dg).all() or (successor_dg < 0).any():
        raise ValueError("DG activity must be finite and nonnegative")
    return successor_dg.detach() > 0


def first_hit_targets(
    successor_hits: Tensor,
    physical_done: Tensor,
    remaining: Tensor,
    next_online: Tensor,
    next_target: Tensor,
    *,
    successor_valid: Tensor | None = None,
    next_action_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return detached all-goal Double-Q targets and their usable-label mask.

    Values have shape [B,G,A], hits [B,G], and terminal/deadline arrays [B].
    Both successor heads must already be evaluated at the SAME goal and h-1.
    ``successor_valid`` [B] says a genuine successor detector observation is
    available, including on a physical terminal transition. Missing observations
    yield unusable labels, even when termination is known. Horizon zero is the
    analytic boundary and needs no observed successor.

    The optional mask [B,A] or [B,G,A] excludes unsupported successor actions.
    No supported live action makes the bootstrap label unusable; it does not
    turn unknown action consequences into zero success probability. Immediate
    success, physical failure, and h=1 do not require a bootstrap action.

    No discount, exploration reward, manager successor goal, or first-distinct
    failure is included. Targets are in [0,1] when the supplied target values
    are in [0,1]; the existing linear runtime head is not automatically bounded.
    """
    if next_online.ndim != 3 or next_online.shape != next_target.shape:
        raise ValueError("next values must share shape [batch, goals, actions]")
    batch, goals, actions = next_online.shape
    if min(goals, actions) < 1 or successor_hits.shape != (batch, goals):
        raise ValueError("successor_hits must have shape [batch, goals]")
    if physical_done.shape != (batch,) or remaining.shape != (batch,):
        raise ValueError("physical_done and remaining must have shape [batch]")
    if successor_hits.dtype != torch.bool or physical_done.dtype != torch.bool:
        raise ValueError("hits and physical_done must be boolean")
    if remaining.dtype == torch.bool or remaining.is_floating_point() or (remaining < 0).any():
        raise ValueError("remaining must contain nonnegative integer deadlines")
    tensors = (next_target, successor_hits, physical_done, remaining)
    if any(t.device != next_online.device for t in tensors):
        raise ValueError("All target inputs must share a device")
    if not next_online.is_floating_point() or next_online.dtype != next_target.dtype:
        raise ValueError("next values must share a floating dtype")
    if not torch.isfinite(next_online).all() or not torch.isfinite(next_target).all():
        raise ValueError("next values must be finite")
    if successor_valid is None:
        successor_valid = torch.ones(batch, dtype=torch.bool, device=next_online.device)
    if successor_valid.shape != (batch,) or successor_valid.dtype != torch.bool:
        raise ValueError("successor_valid must be boolean [batch]")
    if successor_valid.device != next_online.device:
        raise ValueError("successor_valid must share the values device")
    if next_action_mask is None:
        allowed = torch.ones_like(next_online, dtype=torch.bool)
    else:
        allowed = next_action_mask
        if allowed.shape == (batch, actions):
            allowed = allowed[:, None, :].expand_as(next_online)
        if allowed.shape != next_online.shape or allowed.dtype != torch.bool:
            raise ValueError("next_action_mask must be boolean [B,A] or [B,G,A]")
        if allowed.device != next_online.device:
            raise ValueError("next_action_mask must share the values device")
    with torch.no_grad():
        active = remaining[:, None] > 0
        needs_bootstrap = (remaining[:, None] > 1) & ~physical_done[:, None] & ~successor_hits
        selected = next_online.masked_fill(~allowed, -torch.inf).argmax(-1, keepdim=True)
        continuation = next_target.gather(-1, selected).squeeze(-1)
        targets = torch.where(successor_hits, torch.ones_like(continuation), torch.zeros_like(continuation))
        targets = torch.where(needs_bootstrap, continuation, targets)
        targets = torch.where(active, targets, torch.zeros_like(targets))
        usable = ~active | (successor_valid[:, None] & (~needs_bootstrap | allowed.any(-1)))
        # NaN is deliberate: callers cannot accidentally train missing labels as
        # negatives without explicitly ignoring the supplied usability mask.
        return targets.masked_fill(~usable, torch.nan), usable
