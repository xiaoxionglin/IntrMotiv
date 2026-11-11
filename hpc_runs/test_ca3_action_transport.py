"""Contracts for low-rank transport and the exact finite-window fast path."""

import pytest
import torch

from ca3_action_transport import ActionConditionedCA3, LowRankActionTransport, TransportReadout, inputs_from_registers
from parallel_action_register_probe import advance_ca3, build_dataset
from test_parallel_action_register_probe import small_config


@pytest.mark.parametrize("mode,rank", [("shared", 1), ("shared", 2), ("shared", 4), ("per_action", 2)])
def test_original_dg_channel_and_landmark_identity_are_preserved(mode, rank):
    torch.manual_seed(2)
    core = ActionConditionedCA3(4, rank, 3, mode)
    state = torch.zeros(2, 3, 12, 5)
    raw = torch.zeros(2, 3, 12)
    for step in range(6):
        dg = torch.zeros(2, 3)
        dg[:, 1] = float(step == 0)
        actions = torch.tensor([step % 3, (step + 1) % 3])
        original = state.clone()
        raw = advance_ca3(raw, dg, 3)
        updated = core.advance(state, dg, actions)
        assert torch.equal(updated[..., 0], raw)
        assert updated[:, [0, 2]].eq(0).all()
        assert torch.equal(original, state)
        state = updated


@pytest.mark.parametrize("mode", ["shared", "per_action"])
def test_pooled_fast_path_matches_full_shift_register_with_overlapping_events(mode):
    torch.manual_seed(5)
    core = ActionConditionedCA3(4, 2, 3, mode)
    dg, actions = torch.rand(2, 8, 3), torch.randint(4, (2, 7))
    full = torch.zeros(2, 3, 12, 5)
    full = core.advance(full, dg[:, 0], torch.full((2,), 3))
    for step in range(7):
        full = core.advance(full, dg[:, step + 1], actions[:, step])
    pooled = core.pooled_without_expiry(dg, actions, 12)
    torch.testing.assert_close(pooled, full[..., 1:].sum(2) / 3)
    with pytest.raises(ValueError, match="expiry"):
        core.pooled_without_expiry(dg, actions, 8)


def test_shared_rank_one_commutes_but_rank_two_can_distinguish_order():
    torch.manual_seed(3)
    rank_one = LowRankActionTransport(4, 1).double()
    matrices = rank_one.matrices()
    torch.testing.assert_close(matrices[0] @ matrices[1], matrices[1] @ matrices[0], atol=1e-12, rtol=1e-12)
    rank_two = LowRankActionTransport(4, 2)
    with torch.no_grad():
        rank_two.factor_a.copy_(torch.tensor([[1., 1., 0., 0.], [0., 1., 0., 0.]]))
        rank_two.factor_b.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]]))
        rank_two.action_embedding.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
    matrices = rank_two.matrices()
    assert not torch.allclose(matrices[0] @ matrices[1], matrices[1] @ matrices[0])
    assert torch.equal(matrices[3], torch.eye(4))


def test_gradients_reach_all_action_factors_without_modifying_factual_inputs():
    torch.manual_seed(3)
    model = TransportReadout(2, 3, 4, 2, 2, 12)
    dg = torch.rand(5, 6, 3)
    actions = torch.randint(3, (5, 5))
    original = dg.clone()
    model(torch.rand(5, 2), dg, actions).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert torch.equal(dg, original)


def test_recovered_inputs_recreate_original_ca3_without_pose_or_word_metadata():
    config = small_config()
    config["anchor_age"] = 6
    data = build_dataset(config)
    expected = data["dg_state"][:, [0, 1, 2, 3, 15]].clone()
    data["rows"], data["targets"] = None, None
    dg, actions = inputs_from_registers(data, config)
    state = torch.zeros_like(expected)
    for step in range(dg.size(1)):
        state = advance_ca3(state, dg[:, step], config["repeat"])
    assert torch.equal(state, expected)
    assert set(actions.flatten().tolist()) == {0, 1, 2, 3}
