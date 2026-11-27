"""Stationary target semantics without a native environment or donor data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hpc_runs.first_hit_control.detectors import detector_hits
from hpc_runs.first_hit_control.stationary_runtime_adapter import stationary_replay_batch


class TableModel(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.layout = SimpleNamespace(goals=3)
        self.gamma = .99

    def all_goals(self, core, remaining=None):
        return self.values[None].expand(len(core), -1, -1)

    def current_dg(self, core):
        return core[:, :9].reshape(-1, 3, 3)[:, :, 0]


def batch_fixture():
    count = 7
    data = dict(core=np.zeros((count, 12), np.float32), successor_core=np.zeros((count, 12), np.float32),
        successor_dg=np.zeros((count, 3), np.float32), action=np.arange(count) % 3,
        physical_done=np.array([False, True, True, True, False, False, False]),
        successor_valid=np.array([True, True, True, False, True, True, True]),
        successor_core_valid=np.array([True, False, False, False, False, True, True]),
        source_goal=np.full(count, 2), truncated=np.ones(count, bool))
    data["core"][0, [0, 3]] = [2, 3]  # Only g1 wins, though g0 is also active.
    data["core"][5, [0, 3]] = [2, 2]  # Tie picks first, g0.
    data["successor_dg"][1, 2] = 4
    data["successor_dg"][4, 1] = 3
    online = TableModel([[4, 5, 1], [2, 5, 1], [2, 5, 1]])
    target = TableModel([[2, .4, -3], [0, 1.02, 0], [0, -.5, 0]])
    return data, online, target


def test_raw_argmax_project_before_discount_factual_action_and_current_winner():
    data, online, target = batch_fixture()
    selected, labels, usable = stationary_replay_batch(online, target, data, np.arange(7), np.zeros(7),
                                                     lambda dg: detector_hits(dg, "winner"))
    assert selected[0].tolist() == [4, 2, 2]  # The factual a0, not greedy a1.
    assert selected[1].tolist() == [5, 5, 5]
    assert selected.requires_grad and not labels.requires_grad
    assert usable[0].tolist() == [True, False, True]
    assert labels[0, 0].item() == pytest.approx(.396)  # Raw argmax selects target .4, not 2.
    assert labels[6].tolist() == pytest.approx([.396, .99, 0])
    assert usable[5].tolist() == [False, True, True]  # Canonical tie and no all-positive mask.
    assert torch.isnan(labels[0, 1]) and labels[0, 2] == 0


def test_terminal_hit_missing_observation_and_missing_core_remain_distinct():
    data, online, target = batch_fixture()
    _, labels, usable = stationary_replay_batch(online, target, data, np.arange(7), np.full(7, 1),
                                              lambda dg: detector_hits(dg, "winner"))
    assert usable[1].all() and labels[1].tolist() == [0, 0, 1]  # Hit wins on terminal.
    assert usable[2].all() and labels[2].tolist() == [0, 0, 0]
    assert not usable[3].any() and torch.isnan(labels[3]).all()
    assert usable[4].tolist() == [False, True, False]
    assert labels[4, 1] == 1
    # External h=1 and the extra truncation flag do not turn live nonhit into failure.
    assert labels[6, 0].item() == pytest.approx(.396)


def test_external_deadlines_cannot_change_predictions_targets_or_masks():
    data, online, target = batch_fixture()
    first = stationary_replay_batch(online, target, data, np.arange(7), np.zeros(7),
                                   lambda dg: detector_hits(dg, "winner"))
    second = stationary_replay_batch(online, target, data, np.arange(7), np.full(7, 256),
                                    lambda dg: detector_hits(dg, "winner"))
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, equal_nan=True, rtol=0, atol=0)


def test_nonfinite_target_cannot_be_hidden_by_projection():
    data, online, target = batch_fixture()
    with torch.no_grad():
        target.values[0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="Nonfinite raw"):
        stationary_replay_batch(online, target, data, np.arange(7), None,
                                lambda dg: detector_hits(dg, "winner"))
