"""Causal replay and whole-episode fitting contracts, without DMLab."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hpc_runs.first_hit_control.replay_fit import FitConfig, episode_split, fit_replay, replay_batch, validate_replay
from hpc_runs.first_hit_control.runtime_adapter import FixedGoalFirstHitAdapter, GoalLayout
from hpc_runs.first_hit_control.targets import all_goal_hits


class Decoder(nn.Module):
    target_condition_start = 2
    n_targets = 2

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, values):
        return self.linear(values).relu()

    def get_out_size(self):
        return 8


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Linear(8, 2)
        self.auxiliary = nn.Linear(17, 2)

    def hindsight(self, hidden, remaining, scale):
        clock = remaining[:, None].to(hidden) / scale
        return self.auxiliary(torch.cat((hidden, hidden * clock, clock), -1))


def model():
    return FixedGoalFirstHitAdapter(Decoder(), Heads(), GoalLayout(4, 2, 2), 2)


def replay():
    # Two physical contexts, each repeated in independent episodes; one action
    # deterministically reaches each detector. No state IDs enter the learner.
    contexts = np.repeat(np.eye(2, dtype=np.float32), 8, axis=0)
    actions = np.tile([0, 1], 8).astype(np.int64)
    return {"core": np.concatenate((contexts, np.zeros_like(contexts)), axis=1),
            "successor_core": np.zeros((16, 4), np.float32),
            "successor_dg": np.eye(2, dtype=np.float32)[actions], "action": actions,
            "physical_done": np.ones(16, bool), "successor_valid": np.ones(16, bool),
            "successor_core_valid": np.zeros(16, bool), "episode_id": np.arange(16, dtype=np.int64)}


def test_split_never_leaks_a_recorded_episode():
    ids = np.repeat(np.arange(7), 4)
    train, holdout = episode_split(ids, .25, 99)
    assert set(ids[train]).isdisjoint(ids[holdout])
    assert sorted(np.r_[train, holdout].tolist()) == list(range(len(ids)))


def test_terminal_hits_do_not_require_a_successor_core():
    source, data = model(), replay()
    prediction, target, valid = replay_batch(source, source, data, np.arange(4), np.full(4, 2), all_goal_hits)
    assert valid.all()
    assert torch.equal(target, torch.tensor(data["successor_dg"][:4]))
    assert prediction.requires_grad and not target.requires_grad


def test_missing_live_core_masks_only_targets_requiring_bootstrap():
    source, data = model(), replay()
    data["physical_done"][:] = False
    _, targets, valid = replay_batch(source, source, data, np.arange(2), np.array([1, 2]), all_goal_hits)
    assert valid[0].all()  # h=1: known non-hit is a certified deadline failure.
    assert valid[1].tolist() == [False, True]
    assert torch.isnan(targets[1, 0]) and targets[1, 1] == 1


def test_fit_changes_only_readout_and_keeps_replay_immutable():
    torch.set_num_threads(1)
    data, source = replay(), model()
    before = {key: value.copy() for key, value in data.items()}
    result = fit_replay(source, data, FitConfig(updates=100, batch_size=16, horizon=2,
                        learning_rate=.01, target_update_period=10, log_every=100), all_goal_hits,
                        target_factory=model)
    assert result["initial_readout_sha256"] != result["final_readout_sha256"]
    assert result["goal_labels_seen"] == 100 * 16 * 2
    assert result["trace"][-1]["training_mse"] < .02
    assert result["holdout_transitions"] > 0
    assert all(np.array_equal(data[key], value) for key, value in before.items())


def test_missing_successors_are_not_trainable_negatives():
    data = replay()
    data["successor_valid"][:] = False
    with pytest.raises(ValueError, match="No certified"):
        fit_replay(model(), data, FitConfig(updates=1, horizon=2), all_goal_hits)


def test_invalid_action_or_shape_fails_before_training():
    data = replay()
    data["action"][0] = 2
    with pytest.raises(ValueError, match="action outside"):
        validate_replay(data, GoalLayout(4, 2, 2), 2)
    data = replay()
    data["successor_core"] = data["successor_core"][:, :3]
    with pytest.raises(ValueError, match="successor_core"):
        validate_replay(data, GoalLayout(4, 2, 2), 2)
