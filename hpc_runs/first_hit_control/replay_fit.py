"""Fit an independent first-arrival readout from certified frozen transitions.

This module never constructs an environment or changes the source policy. The
caller supplies a copied readout and a frozen replay bundle. Training enumerates
all goal labels for each factual action, including failed outcomes.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
import torch

from .targets import first_hit_targets


@dataclass(frozen=True)
class FitConfig:
    seed: int = 99
    updates: int = 5000
    batch_size: int = 128
    horizon: int = 256
    learning_rate: float = 0.0003
    target_update_period: int = 100
    holdout_fraction: float = 0.1
    log_every: int = 100

    def __post_init__(self):
        for key in ("updates", "batch_size", "horizon", "target_update_period", "log_every"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.learning_rate or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in [0,1)")


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_replay(data, layout, action_count):
    """Validate the causal array contract; provenance is checked by the loader."""
    required = ("core", "action", "successor_core", "successor_dg", "physical_done",
                "successor_valid", "successor_core_valid", "episode_id")
    if any(key not in data for key in required):
        raise ValueError(f"Missing replay keys: {sorted(set(required) - set(data))}")
    count = len(data["action"])
    if count < 1 or any(len(data[key]) != count for key in required):
        raise ValueError("Replay arrays must have the same nonzero length")
    for key in ("core", "successor_core"):
        if data[key].shape != (count, layout.width) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} shape or values")
    if data["successor_dg"].shape != (count, layout.goals):
        raise ValueError("Successor DG shape differs from goal vocabulary")
    if not np.isfinite(data["successor_dg"]).all() or (data["successor_dg"] < 0).any():
        raise ValueError("Successor DG must be finite and nonnegative")
    for key in ("physical_done", "successor_valid", "successor_core_valid"):
        if data[key].shape != (count,) or data[key].dtype != np.bool_:
            raise ValueError(f"{key} must be boolean [N]")
    for key in ("action", "episode_id"):
        if data[key].shape != (count,) or not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"{key} must be integer [N]")
    if (data["action"] < 0).any() or (data["action"] >= action_count).any():
        raise ValueError("Replay contains an action outside the source interface")
    return count


def episode_split(episode_ids, fraction, seed):
    """Split whole recorded episodes; no row from an episode crosses the split."""
    episodes = np.unique(episode_ids)
    if len(episodes) < 2 or fraction == 0:
        return np.arange(len(episode_ids)), np.array([], dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(episodes)
    size = min(len(episodes) - 1, max(1, round(fraction * len(episodes))))
    holdout = np.isin(episode_ids, shuffled[:size])
    return np.flatnonzero(~holdout), np.flatnonzero(holdout)


def replay_batch(model, target, data, indices, remaining, detector):
    """Build factual-action, all-goal predictions and detached Bellman labels.

    ``detector`` is explicitly supplied from the source's goal-event contract.
    A valid terminal DG observation needs no successor core for bootstrapping.
    Unavailable physical successors stay unknown instead of becoming negatives.
    """
    device = next(model.parameters()).device
    to_tensor = lambda key: torch.as_tensor(data[key][indices], device=device)
    current = to_tensor("core").float()
    successor = to_tensor("successor_core").float()
    clocks = torch.as_tensor(remaining, dtype=torch.long, device=device)
    done = to_tensor("physical_done")
    valid = to_tensor("successor_valid")
    core_valid = to_tensor("successor_core_valid")
    hits = detector(to_tensor("successor_dg").float())
    with torch.no_grad():
        online_next = model.all_goals(successor, clocks - 1)
        target_next = target.all_goals(successor, clocks - 1)
        labels, usable = first_hit_targets(hits, done, clocks, online_next, target_next,
                                           successor_valid=valid)
        needs_core = (clocks[:, None] > 1) & ~done[:, None] & ~hits
        usable &= ~needs_core | core_valid[:, None]
        labels = labels.masked_fill(~usable, torch.nan)
    values = model.all_goals(current, clocks)
    actions = to_tensor("action").long()
    selected = values.gather(-1, actions[:, None, None].expand(-1, model.layout.goals, 1)).squeeze(-1)
    return selected, labels, usable


def fit_replay(model, data, config: FitConfig, detector, *, target_factory=None, progress=None):
    """Optimize only the supplied copied readout; holdout is diagnostic only."""
    if model.horizon_scale != config.horizon:
        raise ValueError("Model and fit deadline scales differ")
    count = validate_replay(data, model.layout, model.q_heads.main.out_features)
    train, holdout = episode_split(data["episode_id"], config.holdout_fraction, config.seed)
    # Rows without any certified successor observation cannot supervise a goal.
    train_usable = train[data["successor_valid"][train]]
    if len(train_usable) == 0:
        raise ValueError("No certified successor transitions in training episodes")
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    initial = state_digest(model)
    if target_factory is None:
        target = copy.deepcopy(model)
        if any(not parameter.is_leaf for parameter in target.parameters()):
            raise ValueError("Supply target_factory for a scripted decoder with nonleaf deep-copy parameters")
    else:
        # Constructing a target must not perturb the optimizer's random stream.
        with torch.random.fork_rng():
            target = target_factory()
        target.to(next(model.parameters()).device)
        target.load_state_dict(model.state_dict(), strict=True)
    target = target.eval().requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable copied readout parameters")
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    traces, skipped, labels_seen = [], 0, 0
    started = time.monotonic()
    model.train()
    for update in range(1, config.updates + 1):
        indices = rng.choice(train_usable, config.batch_size, replace=True)
        remaining = rng.integers(1, config.horizon + 1, config.batch_size)
        prediction, labels, usable = replay_batch(model, target, data, indices, remaining, detector)
        if not usable.any():
            skipped += 1
            continue
        loss = torch.nn.functional.mse_loss(prediction[usable], labels[usable])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite first-hit loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError(f"Nonfinite first-hit gradient at update {update}")
        optimizer.step()
        labels_seen += int(usable.sum())
        if update % config.target_update_period == 0:
            target.load_state_dict(model.state_dict())
        if update == 1 or update % config.log_every == 0 or update == config.updates:
            row = {"update": update, "training_mse": float(loss.detach()),
                   "prediction_min": float(prediction.detach()[usable].min()),
                   "prediction_max": float(prediction.detach()[usable].max()),
                   "usable_goal_labels": labels_seen}
            traces.append(row)
            if progress:
                progress(row)
    model.eval()
    diagnostic = None
    if len(holdout):
        # No selection, early stopping, or hyperparameter change uses this loss.
        with torch.no_grad():
            indices = rng.choice(holdout, min(len(holdout), 2048), replace=False)
            clocks = rng.integers(1, config.horizon + 1, len(indices))
            prediction, labels, usable = replay_batch(model, target, data, indices, clocks, detector)
            diagnostic = {"sampled_transitions": len(indices), "usable_labels": int(usable.sum()),
                          "td_mse": float(((prediction[usable] - labels[usable]) ** 2).mean()) if usable.any() else None}
    return {"schema": "intrmotiv/frozen-first-hit-fit/v1", "config": asdict(config),
            "transitions": count, "training_transitions": len(train), "holdout_transitions": len(holdout),
            "certified_training_successors": len(train_usable), "updates_without_usable_labels": skipped,
            "goal_labels_seen": labels_seen, "trainable_parameters": sum(p.numel() for p in parameters),
            "initial_readout_sha256": initial, "final_readout_sha256": state_digest(model),
            "seconds": time.monotonic() - started, "trace": traces, "holdout_diagnostic": diagnostic,
            "interpretation": "Offline fitting diagnostics only; executed matched-goal control is required"}
