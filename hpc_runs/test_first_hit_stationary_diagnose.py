"""Saved-path diagnostic checks: no new routes and exact full-memory support."""

import json
import unittest

import numpy as np
import torch

from hpc_runs.first_hit_control.finite import FiniteMDP
from hpc_runs.first_hit_control.neural import NeuralConfig, observe
from hpc_runs.first_hit_control.stationary_diagnose import audit_failed_path


class StationaryPathAuditTests(unittest.TestCase):
    def fixture(self):
        env = FiniteMDP("wait", np.eye(2)[np.array([[1, 0, 0], [1, 1, 1]])],
                        np.array([False, True]), (1,), (0,))
        dg = np.eye(2, dtype=np.float32)
        memory = observe(torch.zeros(1, 2, 8), dg[0], 1)
        distinct_history = memory.clone()
        distinct_history[0, 1, 3] = 1
        data = dict(ca3=torch.cat((memory, memory, distinct_history)).numpy(),
                    state=np.array([0, 0, 0]), action=np.array([0, 1, 1]))
        row = dict(route="first_hit", success="0", start="0", goal="1",
                   prefix="[0]", states="[0, 0, 0]", actions="[1, 1]")

        class Waiting(torch.nn.Module):
            def forward(self, memory, goal):
                return torch.tensor([[1., 1.02, .9]])

        return Waiting(), row, env, dg, (1,), data

    def test_first_wrong_ranking_and_full_ca3_support(self):
        audit = audit_failed_path(*self.fixture(), NeuralConfig(), .99)
        first = audit["first_wrong_ranking"]
        self.assertEqual(first["step"], 0)
        self.assertEqual(first["shortest_actions"], [0])
        self.assertAlmostEqual(first["oracle_shortest_minus_chosen"], .01)
        self.assertAlmostEqual(first["predicted_chosen_minus_best_shortest"], .02, places=6)
        self.assertEqual(first["full_ca3_exact_replay_rows"], 2)
        self.assertEqual(first["full_ca3_exact_factual_action_rows"], 1)
        self.assertEqual(first["full_ca3_exact_shortest_action_rows"], {"0": 1})
        self.assertTrue(first["chosen_successor_is_self_loop"])
        self.assertEqual(audit["strict_wrong_ranking_steps"], 2)

    def test_rejects_changed_action_in_saved_trace(self):
        args = list(self.fixture())
        args[1]["actions"] = "[2, 1]"
        with self.assertRaisesRegex(ValueError, "reproduce"):
            audit_failed_path(*args, NeuralConfig(), .99)

    def test_rejects_nonfactual_path_and_success_path(self):
        args = list(self.fixture())
        args[1]["states"] = json.dumps([0, 1, 0])
        with self.assertRaisesRegex(ValueError, "contains its requested goal"):
            audit_failed_path(*args, NeuralConfig(), .99)


if __name__ == "__main__":
    unittest.main()
