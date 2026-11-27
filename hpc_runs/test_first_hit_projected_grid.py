"""Small numerical/interface fixtures; not the declared six efficacy fits."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.projected_grid import (
    BatchAudit, ClippedTarget, ENVIRONMENT, collect_replay, finite_view, fixed_config,
    fresh_residuals, load_environment, prepare, projected_batch,
)
from hpc_runs.first_hit_control.stationary_readout import (
    StationaryReadout, fit_stationary, original_classes, stationary_batch, summarize_trials,
)
from hpc_runs.first_hit_control.neural import evaluate_readout


@pytest.fixture
def environment(tmp_path):
    archived = tmp_path / "environment.py"
    archived.write_bytes(ENVIRONMENT.read_bytes())
    return load_environment(archived)


def simple_data():
    return dict(ca3=np.zeros((2, 2, 8), np.float32), successor_ca3=np.zeros((2, 2, 8), np.float32),
                successor_dg=np.zeros((2, 2), np.float32), action=np.array([0, 1]), terminated=np.array([False, True]))


def test_projection_precedes_gamma_and_does_not_clip_online_argmax():
    class Online(torch.nn.Module):
        def forward(self, memory, goals):
            return torch.tensor([[1.1, 1.2]]).expand(len(goals), -1)
    class Target(torch.nn.Module):
        def forward(self, memory, goals):
            return torch.tensor([[.2, 1.4]]).expand(len(goals), -1)
    data = simple_data()
    data["successor_dg"][0, 0] = 1
    _, target, usable = projected_batch(Online(), Target(), data, np.arange(2), (0, 1))
    torch.testing.assert_close(target, torch.tensor([[1., .99], [0., 0.]]))
    assert usable.all()
    _, unprojected, _ = stationary_batch(Online(), Target(), data, np.arange(2), (0, 1))
    assert unprojected[0, 1] > 1  # Clipping gamma*Q would incorrectly give1.


def test_known_scalar_selfloop_witness_and_negative_values():
    class Constant(torch.nn.Module):
        def forward(self, memory, goals):
            return torch.tensor([[1., 1.3]]).expand(len(goals), -1)
    data = simple_data()
    _, target, _ = projected_batch(Constant(), Constant(), data, np.arange(2), (0, 1))
    assert target[0, 0].item() == pytest.approx(.99)
    class Negative(torch.nn.Module):
        def forward(self, memory, goals):
            return torch.full((len(goals), 2), -5.)
    assert ClippedTarget(Negative())(None, torch.tensor([0])).max() == 0


def test_canonical_steps_preserve_wall_selfloops_and_replay_cuts(environment):
    env = finite_view(environment)
    assert env.n_states == 16 and env.n_actions == 4
    for action in ("S", "W"):
        assert env.transition[0, environment.actions.index(action), 0] == 1
        assert not environment.is_valid(environment.states[0], action)
    config = replace(fixed_config(), transitions=64)
    data = collect_replay(environment, config, 8)
    assert not data["terminated"].any()
    assert np.flatnonzero(data["truncated"]).tolist() == [31, 63]
    assert len(set(data["episode_id"])) == 2
    for state, action, successor in zip(data["state"], data["action"], data["next_state"]):
        assert environment.step(environment.states[state], environment.actions[action]) == environment.states[successor]
    np.testing.assert_array_equal(data["ca3"][:, :, 0], np.eye(16)[data["state"]])
    np.testing.assert_array_equal(data["successor_ca3"][:, :, 0], data["successor_dg"])


def _factory(config):
    decoder, action, _ = original_classes()
    settings = dict(decoder_mlp_layers=[128, 128], nonlinearity="relu", use_jit=True)
    return lambda: StationaryReadout(16, config, decoder, action, settings, action_count=4)


def test_latent_node_id_permutation_cannot_change_projected_updates(environment):
    torch.set_num_threads(1)
    config = replace(fixed_config(), transitions=64, updates=3, batch_size=8)
    data = collect_replay(environment, config, 8)
    changed = {key: value.copy() for key, value in data.items()}
    changed["state"][:] = -999
    changed["next_state"][:] = 999999
    first, _ = fit_stationary(_factory(config), data, tuple(range(16)), config, 8, batch_function=projected_batch)
    second, _ = fit_stationary(_factory(config), changed, tuple(range(16)), config, 8, batch_function=projected_batch)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_paired_actual_initialization_batches_and_masks_are_identical(environment):
    torch.set_num_threads(1)
    config = replace(fixed_config(), transitions=64, updates=3, batch_size=8)
    data = collect_replay(environment, config, 99)
    audits, counts = [], []
    for condition in ("unprojected", "projected"):
        audit = BatchAudit(condition)
        model, stats = fit_stationary(_factory(config), data, tuple(range(16)), config, 99, batch_function=audit)
        assert set(dict(model.named_children())) == {"decoder", "action_parameterization"}
        audits.append(audit.summary()); counts.append(stats["used_noninitial_goal_labels"])
    assert audits[0] == audits[1] and counts == [3*8*15, 3*8*15]


def test_fresh_final_operator_diagnostic_separates_projection():
    class Constant(torch.nn.Module):
        def forward(self, memory, goals):
            return torch.tensor([[1., 1.3]]).expand(len(goals), -1)
    raw = fresh_residuals(Constant(), simple_data(), (0, 1), "unprojected")
    projected = fresh_residuals(Constant(), simple_data(), (0, 1), "projected")
    assert raw["overall"]["target_max"] > 1
    assert projected["overall"]["target_max"] == pytest.approx(.99)
    assert raw["overall"]["mse"] != projected["overall"]["mse"]


def test_all_queries_and_competing_first_denominators(environment):
    class Constant(torch.nn.Module):
        def forward(self, memory, goals, remaining, route="first_hit"):
            return torch.tensor([[0., 0., 1., 0.]])
    env, config = finite_view(environment), fixed_config()
    rows = evaluate_readout(Constant(), env, np.eye(16, dtype=np.float32), tuple(range(16)), config, 8)
    summary, first = summarize_trials(rows, env, config)
    assert summary["n_nontrivial_pairs"] == 240
    assert summary["initially_satisfied_queries"] == 16 and summary["declared_queries"] == 256
    assert len(first) == 3360 and all(row["contrast"] == 0 for row in first)


def test_protocol_is_saved_before_dataset_construction(tmp_path, monkeypatch):
    from hpc_runs.first_hit_control import projected_grid as module
    output = tmp_path / "prospective_order_fixture"
    def stop_before_data(environment, config, seed):
        manifest = json.loads((output / "pre_data_manifest.json").read_text())
        assert manifest["data_status"] == "not_constructed"
        assert (output / "source_snapshot/manifest.json").is_file()
        assert not list(output.glob("*.npz"))
        raise RuntimeError("deliberate fixture stop before data")
    monkeypatch.setattr(module, "collect_replay", stop_before_data)
    with pytest.raises(RuntimeError, match="deliberate fixture"):
        prepare(output)
