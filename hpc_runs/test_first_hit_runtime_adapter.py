"""Boundary qualification against the installed, unchanged IntrMotiv classes."""

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hpc_runs.first_hit_control.runtime_adapter import (
    FixedGoalFirstHitAdapter,
    GoalLayout,
    create_runtime_components,
    fixed_goal_input,
    runtime_factories,
)
from hpc_runs.first_hit_control.targets import all_goal_hits, first_hit_targets


SOURCE = Path(os.environ.get("INTRMOTIV_RUNTIME_SOURCE", "/home/xiaoxiong/SFgit/SF_hipposlam"))


def test_all_goal_target_semantics_and_double_q_selection():
    # Last axis deliberately disagrees between online and target argmax.
    online = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]).expand(5, -1, -1).clone().requires_grad_()
    target = torch.tensor([[[0.4, 0.7], [0.6, 0.3]]]).expand_as(online).clone().requires_grad_()
    hits = torch.tensor([[True, False], [True, False], [False, False], [True, True], [True, False]])
    done = torch.tensor([True, False, False, True, False])
    remaining = torch.tensor([3, 1, 3, 0, 3])
    values, usable = first_hit_targets(hits, done, remaining, online, target)
    expected = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.4, 0.3], [0.0, 0.0], [1.0, 0.3]])
    assert torch.equal(values, expected)
    assert usable.all() and not values.requires_grad


def test_missing_terminal_successor_is_unknown_not_failure():
    hits = torch.tensor([[False, False], [True, False], [False, False]])
    values = torch.zeros(3, 2, 2)
    targets, usable = first_hit_targets(
        hits, torch.tensor([True, True, True]), torch.tensor([2, 2, 0]), values, values,
        successor_valid=torch.tensor([False, True, False]),
    )
    assert torch.isnan(targets[0]).all() and not usable[0].any()
    assert targets[1].tolist() == [1.0, 0.0] and usable[1].all()
    assert targets[2].tolist() == [0.0, 0.0] and usable[2].all()


def test_support_mask_does_not_turn_unknown_continuations_into_failures():
    values = torch.tensor([[[0.8, 0.2], [0.6, 0.3]]]).expand(3, -1, -1)
    hits = torch.tensor([[False, False], [True, False], [False, False]])
    mask = torch.tensor([[False, True], [False, False], [False, False]])
    targets, usable = first_hit_targets(
        hits, torch.tensor([False, False, False]), torch.tensor([3, 3, 1]), values, values,
        next_action_mask=mask,
    )
    assert torch.equal(targets[0], torch.tensor([0.2, 0.3]))
    assert targets[1, 0] == 1 and usable[1, 0]
    assert torch.isnan(targets[1, 1]) and not usable[1, 1]
    assert targets[2].eq(0).all() and usable[2].all()


def test_all_goal_labels_keep_negatives_and_simultaneous_detector_hits():
    dg = torch.tensor([[0.0, 2.0, 3.0], [0.0, 0.0, 0.0]], requires_grad=True)
    hits = all_goal_hits(dg)
    assert hits.tolist() == [[False, True, True], [False, False, False]]
    assert not hits.requires_grad
    with pytest.raises(ValueError, match="nonnegative"):
        all_goal_hits(torch.tensor([[-1.0]]))


def test_goal_overwrite_preserves_prefix_suffix_and_input_gradients():
    layout = GoalLayout(11, 5, 3)
    states = torch.arange(22, dtype=torch.float32).reshape(2, 11).requires_grad_()
    before = states.detach().clone()
    output = fixed_goal_input(states, torch.tensor([2, 0]), layout)
    assert torch.equal(states, before)
    assert torch.equal(output[:, :5], before[:, :5])
    assert torch.equal(output[:, 8:], before[:, 8:])
    assert output[:, 5:8].tolist() == [[0, 0, 1], [1, 0, 0]]
    assert not output.requires_grad
    successor = fixed_goal_input(states + 100, torch.tensor([2, 0]), layout)
    assert torch.equal(output[:, 5:8], successor[:, 5:8])


@pytest.fixture
def actual_components():
    if not SOURCE.is_dir():
        pytest.skip("Set INTRMOTIV_RUNTIME_SOURCE to an existing source checkout")
    decoder_class, q_class, provenance = runtime_factories(SOURCE)
    layout = GoalLayout(19, 12, 4)  # extra suffix verifies true goal slicing
    core = SimpleNamespace(
        Hippo_n_feature=4, target_condition_start=12, get_out_size=lambda: 19,
        cfg=SimpleNamespace(fixed_task_goal_mixture=False),
    )
    torch.manual_seed(27)
    decoder, heads = decoder_class(core, hidden_size=32), q_class(32, 5, auxiliary=True)
    return decoder, heads, layout, provenance


def test_actual_runtime_shape_and_forward_parity(actual_components):
    decoder, heads, layout, provenance = actual_components
    adapter = FixedGoalFirstHitAdapter(decoder, heads, layout, 64)
    states, goals, h = torch.randn(3, layout.width), torch.tensor([0, 2, 3]), torch.tensor([1, 16, 64])
    expected = heads.hindsight(decoder(fixed_goal_input(states, goals, layout)), h, 64)
    assert torch.equal(adapter(states, goals, h), expected)
    assert adapter(states, goals, h).shape == (3, 5)
    assert heads.main.weight.shape == (5, 32)
    assert heads.auxiliary.weight.shape == (5, 65)
    assert len(provenance["files"]) == 2
    assert all(len(sha) == 64 for sha in provenance["files"].values())


def test_existing_clock_head_cannot_represent_a_two_step_arrival_curve(actual_components):
    # This limitation matters to the scientific interpretation: reusing the
    # head proves interface minimality, not arbitrary-deadline expressiveness.
    decoder, heads, layout, _ = actual_components
    state = torch.randn(1, layout.width).expand(3, -1)
    goals = torch.zeros(3, dtype=torch.long)
    hidden = decoder(fixed_goal_input(state, goals, layout))
    values = heads.hindsight(hidden, torch.tensor([1, 2, 3]), 8)
    assert torch.allclose(values[0] - 2 * values[1] + values[2], torch.zeros(5), atol=1e-6)
    arrival_in_two = torch.tensor([0.0, 1.0, 1.0])
    assert arrival_in_two[0] - 2 * arrival_in_two[1] + arrival_in_two[2] == -1


def test_head_routing_horizon_zero_and_goal_enumeration(actual_components):
    decoder, heads, layout, _ = actual_components
    with torch.no_grad():
        heads.main.weight.zero_(); heads.main.bias.copy_(torch.tensor([9, 0, 0, 0, 0]))
        heads.auxiliary.weight.zero_(); heads.auxiliary.bias.copy_(torch.tensor([0, 0, 8, 0, 0]))
    adapter = FixedGoalFirstHitAdapter(decoder, heads, layout, 64)
    states, goals, h = torch.randn(2, layout.width), torch.tensor([1, 3]), torch.tensor([4, 8])
    assert adapter.greedy_actions(states, goals, h).tolist() == [2, 2]
    assert adapter(states, goals, torch.zeros_like(h)).eq(0).all()
    with pytest.raises(ValueError, match="Expired"):
        adapter.greedy_actions(states, goals, torch.zeros_like(h))
    enumerated = adapter.all_goals(states, h)
    assert enumerated.shape == (2, 4, 5)
    for goal in range(4):
        assert torch.equal(enumerated[:, goal], adapter(states, torch.full_like(goals, goal), h))


def test_training_copied_adapter_leaves_source_and_physical_state_unchanged(actual_components):
    decoder, heads, layout, _ = actual_components
    decoder.eval().requires_grad_(False)
    heads.eval().requires_grad_(False)
    source_before = [deepcopy(module.state_dict()) for module in (decoder, heads)]
    adapter = FixedGoalFirstHitAdapter(decoder, heads, layout, 8)
    state = torch.randn(5, layout.width, requires_grad=True)
    state_before = state.detach().clone()
    goals, h = torch.tensor([0, 1, 2, 3, 0]), torch.tensor([1, 2, 4, 8, 8])
    optim = torch.optim.Adam((p for p in adapter.parameters() if p.requires_grad), lr=0.01)
    for _ in range(2):
        loss = (adapter(state, goals, h) - 1).square().mean()
        optim.zero_grad(); loss.backward(); optim.step()
    assert state.grad is None and torch.equal(state, state_before)
    for module, before in zip((decoder, heads), source_before):
        assert all(torch.equal(value, before[key]) for key, value in module.state_dict().items())
        assert all(p.grad is None for p in module.parameters())
        assert all(not p.requires_grad for p in module.parameters())
        assert not module.training
    assert torch.equal(adapter.q_heads.main.weight, heads.main.weight)
    assert not torch.equal(adapter.q_heads.auxiliary.weight, heads.auxiliary.weight)
    assert all(p.requires_grad and p.grad is not None for p in adapter.decoder.parameters())
    assert all(p.requires_grad and p.grad is not None for p in adapter.q_heads.auxiliary.parameters())


def test_factory_uses_actual_source_classes():
    if not SOURCE.is_dir():
        pytest.skip("Runtime source is unavailable")
    decoder, heads, layout, _ = create_runtime_components(16 * 71, 16, 5, source_root=SOURCE)
    assert decoder.get_out_size() == 128 and layout.goal_start == 1136
    assert heads.auxiliary.weight.shape == (5, 257)


@pytest.mark.parametrize("remaining", [torch.tensor([-1]), torch.tensor([0.5]), torch.tensor([True])])
def test_rejects_invalid_deadlines(remaining):
    with pytest.raises(ValueError, match="integer"):
        first_hit_targets(torch.zeros(1, 2, dtype=torch.bool), torch.tensor([False]), remaining,
                          torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))
