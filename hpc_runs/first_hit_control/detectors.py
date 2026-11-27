"""Explicit source landmark-event semantics, independent of queried goals."""

import torch
from torch import Tensor

from hpc_runs.first_hit_control.targets import all_goal_hits


def detector_hits(dg: Tensor, mode: str) -> Tensor:
    """Label all goals under the declared immutable detector predicate.

    ``winner`` reproduces C05 visit-direct: positive argmax, including a winner
    during multi-activation. ``exclusive`` reproduces the frontier event and
    older strict evaluation: exactly one positive DG. ``positive`` labels every
    positive unit and is a distinct outcome definition, never a silent fallback.
    PyTorch argmax selects the first index for ties, matching the source helper.
    """
    active = all_goal_hits(dg)
    if mode == "positive":
        return active
    if mode == "exclusive":
        return active & active.sum(-1, keepdim=True).eq(1)
    if mode == "winner":
        winner = torch.zeros_like(active).scatter_(1, dg.detach().argmax(-1, keepdim=True), True)
        return winner & active.any(-1, keepdim=True)
    raise ValueError("mode must be winner, exclusive, or positive")
