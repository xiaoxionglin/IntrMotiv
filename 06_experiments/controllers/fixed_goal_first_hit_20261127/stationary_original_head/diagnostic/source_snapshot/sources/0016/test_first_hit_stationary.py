"""Contracts for the prospective original-head, stationary graph comparison."""

from dataclasses import replace
import json
import unittest

import numpy as np
import torch

from hpc_runs.first_hit_control.finite import FiniteMDP, make_fork_cycle
from hpc_runs.first_hit_control.neural import (
    NeuralConfig, collect_trajectory_data, evaluate_readout, sparse_context_environment,
)
from hpc_runs.first_hit_control.stationary_readout import (
    StationaryReadout, competing_first_from_trials, fit_stationary, original_classes,
    shortest_distances, stationary_batch, stationary_targets, summarize_trials,
)


class StationaryFirstHitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.decoder, cls.action, _ = original_classes()
        cls.settings = dict(decoder_mlp_layers=[128, 128], nonlinearity="relu", use_jit=True)

    def factory(self, config=NeuralConfig(), features=4):
        return lambda: StationaryReadout(features, config, self.decoder, self.action, self.settings)

    def test_exact_original_classes_raw_outputs_and_no_added_head(self):
        model = self.factory()()
        self.assertEqual(type(model.decoder).__name__, "MlpDecoderJit")
        self.assertEqual(type(model.action_parameterization).__name__, "ActionParameterizationDefault")
        self.assertEqual(set(dict(model.named_children())), {"decoder", "action_parameterization"})
        self.assertEqual(sum(p.numel() for p in model.action_parameterization.parameters()), 3 * 129)
        self.assertEqual(set(model.action_parameterization.state_dict()), {"distribution_linear.weight", "distribution_linear.bias"})
        with torch.no_grad():
            model.action_parameterization.distribution_linear.weight.zero_()
            model.action_parameterization.distribution_linear.bias.copy_(torch.tensor([-2., 3., 8.]))
        memory = torch.randn(2, 4, 8)
        torch.testing.assert_close(model(memory, torch.tensor([2, 3])), torch.tensor([[-2., 3., 8.], [-2., 3., 8.]]))

    def test_remaining_is_not_a_model_input_and_ca3_is_immutable(self):
        model = self.factory()()
        memory = torch.randn(2, 4, 8)
        before = memory.clone()
        expected = model(memory, torch.tensor([2, 3]), torch.tensor([0, 1]))
        torch.testing.assert_close(expected, model(memory, torch.tensor([2, 3]), torch.tensor([100, -9])), rtol=0, atol=0)
        model(memory, torch.tensor([2, 3]), route="memoryless")
        self.assertTrue(torch.equal(before, memory))

    def test_discounted_same_goal_double_q_terminal_success_precedence(self):
        online = torch.tensor([[[9., 1.], [1., 9.]], [[1., 2.], [2., 1.]]])
        target = torch.tensor([[[.7, .2], [.3, .8]], [[3., 4.], [5., 6.]]])
        hits = torch.tensor([[False, False], [True, False]])
        value, usable = stationary_targets(hits, torch.tensor([False, True]), online, target)
        torch.testing.assert_close(value, torch.tensor([[.693, .792], [1., 0.]]))
        self.assertTrue(usable.all())
        self.assertFalse(value.requires_grad)

    def test_factual_action_all_goals_and_initial_command_mask(self):
        class Fake(torch.nn.Module):
            def forward(self, memory, goals):
                # Distinct goal and action contributions make axis mistakes visible.
                return goals[:, None].float() * 10 + torch.arange(3)[None].float()

        data = dict(ca3=np.zeros((2, 4, 8), dtype=np.float32), successor_ca3=np.zeros((2, 4, 8), dtype=np.float32),
                    successor_dg=np.zeros((2, 4), dtype=np.float32), action=np.array([1, 2]), terminated=np.array([False, True]))
        data["ca3"][0, 2, 0] = 1
        data["successor_dg"][1, 3] = 1
        prediction, labels, usable = stationary_batch(Fake(), Fake(), data, np.arange(2), (2, 3))
        torch.testing.assert_close(prediction, torch.tensor([[21., 31.], [22., 32.]]))
        torch.testing.assert_close(labels, torch.tensor([[21.78, 31.68], [0., 1.]]))
        self.assertEqual(usable.tolist(), [[False, True], [True, True]])

    def test_updates_ignore_latent_state_ids_and_scripted_target_is_trainable(self):
        config = replace(NeuralConfig(), transitions=64, updates=3, batch_size=8)
        env, dg, goals = sparse_context_environment()
        data = collect_trajectory_data(env, dg, 8, config)
        relabeled = {name: value.copy() for name, value in data.items()}
        relabeled["state"][:] = -999
        relabeled["next_state"][:] = 999999
        first, stats = fit_stationary(self.factory(config), data, goals, config, 8)
        second, _ = fit_stationary(self.factory(config), relabeled, goals, config, 8)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(p.is_leaf and torch.isfinite(p).all() for p in first.parameters()))
        self.assertEqual(stats["continuing_loss_labels"], 0)

    def test_competing_first_removes_goalblind_own_stop_artifact(self):
        env = FiniteMDP("loop", np.eye(3)[np.array([[1, 1, 1], [2, 2, 2], [0, 0, 0]])],
                        np.zeros(3, dtype=bool), (0, 1, 2), (0,))

        class Constant(torch.nn.Module):
            def forward(self, memory, goals, remaining, route="first_hit"):
                return torch.tensor([[1., 0., 0.]])

        rows = evaluate_readout(Constant(), env, np.eye(3, dtype=np.float32), (0, 1, 2), replace(NeuralConfig(), horizon=3), 8)
        first = competing_first_from_trials(rows, env.goal_states, 3)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(row["contrast"] == 0 for row in first))
        own = next(r for r in rows if r["route"] == "first_hit" and r["start"] == 2 and r["goal"] == 1)
        alt = next(r for r in rows if r["route"] == "shuffled_goal" and r["start"] == 2 and r["goal"] == 1)
        self.assertEqual(own["success"], 1)
        self.assertEqual(alt["success"], 0)
        self.assertEqual(json.loads(alt["actions"]), [])

    def test_shortest_oracle_has_terminal_success_and_failure(self):
        env, _, _ = sparse_context_environment()
        distances = shortest_distances(env)
        self.assertEqual(distances[0, 4], 0)
        self.assertTrue(np.isinf(distances[0, 5]))
        self.assertEqual(distances[0, 2], 1)
        self.assertEqual(distances[0, 0], 2)

    def test_full_goal_denominator_and_terminal_sparse_memoryless_bound(self):
        env, dg, goals = sparse_context_environment()
        config = NeuralConfig()

        class HistoryRule(torch.nn.Module):
            def forward(self, memory, goals, remaining, route="first_hit"):
                cue_b = memory[:, 1, :].sum(-1) > 0
                if route == "memoryless":
                    cue_b = torch.zeros_like(cue_b)
                choice = cue_b.long() ^ (goals == 3).long()
                return torch.nn.functional.one_hot(choice, 3).float()

        rows = evaluate_readout(HistoryRule(), env, dg, goals, config, 8)
        summary, first = summarize_trials(rows, env, config)
        self.assertEqual(summary["declared_queries"], 4)
        self.assertEqual(summary["first_hit_success"], 1)
        self.assertEqual(summary["shortest_success_fraction"], 1)
        self.assertEqual(summary["competing_first_contrast"], 1)
        self.assertEqual(summary["per_route_success"]["memoryless"], .5)
        self.assertEqual(summary["memoryless_competing_first_contrast"], 0)
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
