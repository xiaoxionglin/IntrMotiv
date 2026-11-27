import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.collector import PendingReplay, RewardAudit, encode_terminal_core


def test_pending_pair_has_exact_one_step_successor_and_keeps_negative_goals():
    replay = PendingReplay(2, 4, 2)
    replay.begin(np.arange(4), 1, 3, 0, 1)
    replay.finish(physical_done=False, successor_core=np.arange(4) + 10, successor_dg=[0, 2], successor_episode=3)
    replay.begin(np.arange(4) + 10, 0, 3, 1, 0)
    replay.finish(physical_done=True, successor_core=np.arange(4) + 20, successor_dg=[3, 0])
    assert replay.index == 2 and replay.pending is None
    assert np.array_equal(replay.data["successor_core"][0], replay.data["core"][1])
    assert replay.data["successor_valid"].tolist() == [True, True]
    assert replay.data["successor_core_valid"].tolist() == [True, True]
    assert replay.data["physical_done"].tolist() == [False, True]
    assert replay.data["successor_dg"].tolist() == [[0, 2], [3, 0]]


def test_missing_terminal_is_unknown_and_cannot_accept_reset_as_live_successor():
    replay = PendingReplay(2, 4, 2)
    replay.begin(np.zeros(4), 0, 0, 0, -1)
    with pytest.raises(ValueError, match="reset"):
        replay.finish(physical_done=False, successor_core=np.ones(4), successor_dg=[1, 0], successor_episode=1)
    replay.finish(physical_done=True)
    assert not replay.data["successor_valid"][0]
    replay.begin(np.zeros(4), 0, 1, 0, -1)
    with pytest.raises(RuntimeError, match="finalized"):
        replay.begin(np.zeros(4), 0, 1, 1, -1)
    replay.finish(physical_done=True, successor_dg=[0, 0])
    assert replay.data["successor_valid"].tolist() == [False, True]


def test_reward_cancellation_and_missing_capture_do_not_qualify():
    audit = RewardAudit()
    audit.record({"first_hit_raw_reward": 1.0}, 0.2)
    audit.record({"first_hit_raw_reward": -1.0}, -0.2)
    assert audit.raw_sum == 0 and audit.raw_abs_sum == 2 and audit.raw_nonzero_count == 2
    with pytest.raises(RuntimeError, match="Missing native"):
        audit.record({}, 0)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        audit.record({"first_hit_raw_reward": float("nan")}, 0)


def test_terminal_detector_uses_actual_core_on_cloned_pre_reset_state():
    class Actor:
        def forward_head(self, obs):
            return obs * 2

        def forward_core(self, head, state):
            state.add_(3)
            return head + state, state

    state = torch.ones(1, 3)
    result = encode_terminal_core(Actor(), torch.ones(1, 3), state)
    assert torch.equal(state, torch.ones(1, 3))
    assert torch.equal(result, torch.full((1, 3), 6.))
