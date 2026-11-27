"""Meaningful contracts for the learned, frozen-CA3 first-hit experiment."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from hpc_runs.first_hit_control.finite import FiniteMDP
from hpc_runs.first_hit_control.neural import (
    NeuralConfig, RUNTIME_ROOT, SharedReadout, collect_trajectory_data,
    evaluate_readout, evaluation_starts, observe, sparse_context_environment,
    train_readout,
)
from hpc_runs.first_hit_control.runtime_adapter import runtime_factories


class NeuralFirstHitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.factories = runtime_factories(RUNTIME_ROOT)[:2]

    def test_sparse_history_disambiguates_identical_current_dg(self):
        env, dg, _ = sparse_context_environment()
        config = NeuralConfig()
        starts = list(evaluation_starts(env, dg, config))
        self.assertTrue(torch.equal(starts[0][1][:, :, 0], starts[1][1][:, :, 0]))
        self.assertFalse(torch.equal(starts[0][1], starts[1][1]))
        self.assertEqual(starts[0][1][0, 0, 1], 1)
        self.assertEqual(starts[1][1][0, 1, 1], 1)
        self.assertEqual(env.transition[2, 0, 4], 1)
        self.assertEqual(env.transition[3, 1, 4], 1)

    def test_recorded_successor_uses_exact_ca3_and_physical_terminal(self):
        env, dg, _ = sparse_context_environment()
        config = replace(NeuralConfig(), transitions=64)
        data = collect_trajectory_data(env, dg, 8, config)
        for index in range(len(data["state"])):
            current = torch.tensor(data["ca3"][index:index + 1])
            expected = observe(current, dg[data["next_state"][index]], config.repeat)
            self.assertTrue(torch.equal(expected, torch.tensor(data["successor_ca3"][index:index + 1])))
            self.assertEqual(data["terminated"][index], env.terminal[data["next_state"][index]])

    def test_actual_deadline_head_is_affine_before_zero_boundary(self):
        config = NeuralConfig()
        model = SharedReadout(4, config, *self.factories)
        memory = torch.randn(1, 4, config.memory_length)
        goal = torch.tensor([2])
        q1, q2, q3 = [model(memory, goal, torch.tensor([h])) for h in (1, 2, 3)]
        torch.testing.assert_close(q1 + q3, 2 * q2)
        self.assertTrue(model(memory, goal, torch.tensor([0])).eq(0).all())
        self.assertGreater(model.decoder.target_modulation.numel(), 0)

    def test_goal_permutation_changes_only_goal_modulation_row(self):
        config = NeuralConfig()
        model = SharedReadout(4, config, *self.factories)
        memory = torch.randn(1, 4, config.memory_length)
        before = memory.clone()
        with torch.no_grad():
            model.decoder.target_modulation[2].fill_(0.5)
            model.decoder.target_modulation[3].fill_(-0.5)
        q2 = model(memory, torch.tensor([2]), torch.tensor([2]))
        q3 = model(memory, torch.tensor([3]), torch.tensor([2]))
        self.assertFalse(torch.equal(q2, q3))
        self.assertTrue(torch.equal(memory, before))

    def test_alternative_command_stops_before_original_target(self):
        env = FiniteMDP("stop_test", np.eye(3)[np.array([[1, 1, 1], [2, 2, 2], [2, 2, 2]])],
                        np.zeros(3, dtype=bool), (1, 2), (0,))
        dg = np.eye(3, dtype=np.float32)

        class ConstantAction(torch.nn.Module):
            def forward(self, memory, goals, remaining, route="first_hit"):
                return torch.tensor([[1., 0., 0.]]).repeat(len(memory), 1)

        rows = evaluate_readout(ConstantAction(), env, dg, (1, 2), replace(NeuralConfig(), horizon=2), 8)
        original = next(row for row in rows if row["route"] == "first_hit" and row["start"] == 0 and row["goal"] == 2)
        shuffled = next(row for row in rows if row["route"] == "shuffled_goal" and row["start"] == 0 and row["goal"] == 2)
        self.assertEqual(original["success"], 1)
        self.assertEqual(shuffled["success"], 0)
        self.assertEqual(shuffled["decisions"], 1)

    def test_replay_td_updates_ignore_latent_node_labels(self):
        env, dg, goals = sparse_context_environment()
        config = replace(NeuralConfig(), transitions=128, updates=3, batch_size=16)
        data = collect_trajectory_data(env, dg, 8, config)
        relabeled = {name: array.copy() for name, array in data.items()}
        permutation = np.array([1, 0, 3, 2, 4, 5])
        relabeled["state"] = permutation[relabeled["state"]]
        relabeled["next_state"] = permutation[relabeled["next_state"]]
        first = train_readout(env, dg, goals, data, config, 8, self.factories, "replay_td")[0]
        second = train_readout(env, dg, goals, relabeled, config, 8, self.factories, "replay_td")[0]
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)

    def test_scripted_legacy_decoder_has_independent_trainable_target_path(self):
        from hpc_runs.first_hit_control.legacy_readout import legacy_factory

        factory, _ = legacy_factory({"decoder_mlp_layers": [128, 128], "nonlinearity": "relu", "use_jit": True})
        env, dg, goals = sparse_context_environment()
        config = replace(NeuralConfig(), transitions=128, updates=2, batch_size=16)
        data = collect_trajectory_data(env, dg, 8, config)
        model, _, stats = train_readout(env, dg, goals, data, config, 8, (factory, self.factories[1]), "replay_td")
        self.assertEqual(stats["updates"], 2)
        self.assertTrue(all(parameter.is_leaf for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
