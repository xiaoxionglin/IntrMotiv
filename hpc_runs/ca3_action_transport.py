"""Low-rank action transport within CA3 traces, preserving landmark and age.

The first state channel follows the original scalar DG register exactly.
Only additional tag channels are transformed. Matrices are learned; no action
kinematics, coordinates or endpoint labels are used to construct them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from parallel_action_register_probe import advance_ca3


class LowRankActionTransport(nn.Module):
    """Column-vector maps T_a = I + B diag(e_a) A or I + B_a A_a.

    Action ``action_count`` is the existing all-zero/no-op sentinel and uses
    the identity exactly. A shared identity base makes shared rank-one action
    maps commute. Higher shared rank permits, but does not guarantee,
    noncommuting maps. Per-action factors avoid their common-range constraint.
    """

    def __init__(self, tag_dim: int, rank: int, action_count: int = 3, mode: str = "shared"):
        super().__init__()
        if not 1 <= rank <= tag_dim or mode not in ("shared", "per_action"):
            raise ValueError("Require 1 <= rank <= tag_dim and shared/per_action mode")
        self.tag_dim, self.rank = tag_dim, rank
        self.action_count, self.mode = action_count, mode
        shape_a = (rank, tag_dim) if mode == "shared" else (action_count, rank, tag_dim)
        shape_b = (tag_dim, rank) if mode == "shared" else (action_count, tag_dim, rank)
        self.factor_a = nn.Parameter(torch.randn(shape_a) * .2)
        self.factor_b = nn.Parameter(torch.randn(shape_b) * .05)
        self.action_embedding = nn.Parameter(torch.randn(action_count, rank)) if mode == "shared" else None

    def matrices(self) -> Tensor:
        if self.mode == "shared":
            delta = torch.einsum("kr,ar,rj->akj", self.factor_b, self.action_embedding, self.factor_a)
        else:
            delta = self.factor_b @ self.factor_a
        identity = torch.eye(self.tag_dim, dtype=delta.dtype, device=delta.device)
        return torch.cat((identity[None] + delta, identity[None]), dim=0)

    def forward(self, tags: Tensor, actions: Tensor) -> Tensor:
        matrices = self.matrices()[actions.long()]
        return torch.einsum("b...j,bkj->b...k", tags, matrices)


class ActionConditionedCA3(nn.Module):
    """Reference full state [B,F,E,1+K], with unaltered scalar channel 0."""

    def __init__(self, tag_dim: int, rank: int, repeat: int, mode: str = "shared"):
        super().__init__()
        self.transport = LowRankActionTransport(tag_dim, rank, mode=mode)
        self.repeat = repeat
        initial = torch.zeros(tag_dim)
        initial[0] = 1
        self.register_buffer("initial_tag", initial)

    def advance(self, state: Tensor, dg: Tensor, actions: Tensor) -> Tensor:
        raw = advance_ca3(state[..., 0], dg, self.repeat)
        previous = torch.cat((torch.zeros_like(state[:, :, :1, 1:]), state[:, :, :-1, 1:]), dim=2)
        tags = self.transport(previous, actions)
        injection = torch.zeros_like(tags)
        injection[:, :, :self.repeat] = dg[:, :, None, None] * self.initial_tag
        return torch.cat((raw[..., None], tags + injection), dim=-1)

    def pooled_without_expiry(self, dg: Tensor, actions: Tensor, expanded: int) -> Tensor:
        """Exact normalized sum over tag age slots, only before any expiry.

        dg is [B,T+1,F], actions [B,T]. Since shifting loses no nonzero
        coordinate when T+repeat <= E, sum(tags)/repeat evolves as
        c_{t+1,i} = T_a c_{t,i} + u_{t+1,i} c0. This avoids materializing
        many identical, zero or redundant age slots during this small probe.
        Original scalar CA3 features are retained separately by the readout.
        """
        if actions.shape != dg.shape[:2][:1] + (dg.size(1) - 1,):
            raise ValueError("Require dg [B,T+1,F] and actions [B,T]")
        if actions.size(1) + self.repeat > expanded:
            raise ValueError("Pooled shortcut would omit tag expiry; use the full recurrence")
        tags = dg[:, 0, :, None] * self.initial_tag
        matrices = self.transport.matrices()
        for step in range(actions.size(1)):
            matrix = matrices[actions[:, step].long()]
            tags = torch.bmm(tags, matrix.transpose(1, 2)) + dg[:, step + 1, :, None] * self.initial_tag
        return tags


def inputs_from_registers(data: dict, config: dict) -> tuple[Tensor, Tensor]:
    """Recover factual DG/action inputs from existing registers, not labels.

    DG's R-wide injection is a causal box convolution over age. Its inverse
    is u[k] = X[k]-X[k-1]+u[k-R]. This exact finite-window inverse is valid
    here because the source memory starts at zero and no event has expired.
    """
    steps, repeat = config["steps"], config["repeat"]
    if steps + repeat > config["expanded"]:
        raise ValueError("Probe input recovery requires no expired DG events")
    channels = [0, 1, 2, 3, config["dg_count"] - 1]
    state = data["dg_state"][:, channels]
    recovered = torch.zeros(state.size(0), len(channels), steps + 1)
    for age in range(steps + 1):
        recovered[:, :, age] = state[:, :, age]
        if age:
            recovered[:, :, age] -= state[:, :, age - 1]
        if age >= repeat:
            recovered[:, :, age] += recovered[:, :, age - repeat]
    dg = recovered.flip(-1).transpose(1, 2).contiguous()
    history = data["action_state"][:, -steps:]
    actions = history.argmax(-1)
    actions = torch.where(history.sum(-1) > 0, actions, history.size(-1))
    return dg, actions


class TransportReadout(nn.Module):
    """Jointly learned transport and endpoint head, without an action register."""

    def __init__(self, base_size: int, landmark_count: int, tag_dim: int, rank: int,
                 repeat: int, expanded: int, mode: str = "shared", hidden: int = 64):
        super().__init__()
        self.core = ActionConditionedCA3(tag_dim, rank, repeat, mode)
        self.expanded = expanded
        size = base_size + landmark_count * tag_dim
        self.head = (nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, 2))
                     if hidden else nn.Linear(size, 2))

    def forward(self, base: Tensor, dg: Tensor, actions: Tensor) -> Tensor:
        tags = self.core.pooled_without_expiry(dg, actions, self.expanded)
        return self.head(torch.cat((base, tags.flatten(1)), dim=1))
