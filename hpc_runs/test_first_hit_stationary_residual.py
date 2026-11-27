"""Final-online residuals must use factual successor history and full masks."""

import unittest

import numpy as np
import torch

from hpc_runs.first_hit_control.stationary_residual import complete_replay_values, residual_metrics


class FinalStationaryResidualTests(unittest.TestCase):
    def test_same_physical_node_different_history_uses_actual_successor(self):
        class History(torch.nn.Module):
            def forward(self, memory, goals):
                value = 1 + memory[:, 0, 2] * .04 + memory[:, 0, 1] * .01
                return torch.stack((torch.ones_like(value), value), dim=1)

        current, successor = np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.float32)
        current[0, 0, 1], successor[0, 0, 2] = 1, 1
        data = dict(ca3=current, successor_ca3=successor, action=np.array([1]),
                    successor_dg=np.zeros((1, 2), np.float32), terminated=np.array([False]))
        prediction, labels, usable = complete_replay_values(History(), data, (1,), .99)
        self.assertAlmostEqual(float(prediction[0, 0]), 1.01, places=6)
        self.assertAlmostEqual(float(labels[0, 0]), .99 * 1.04, places=6)
        self.assertEqual(int(usable.sum()), 1)
        self.assertAlmostEqual(residual_metrics(prediction, labels, usable)["mse"], (1.01-.99*1.04)**2, places=7)

    def test_complete_batches_preserve_current_goal_mask_and_terminal_hit(self):
        class Constant(torch.nn.Module):
            def forward(self, memory, goals):
                return torch.ones((len(memory), 2)) * 1.2

        memory = np.zeros((3, 2, 3), np.float32)
        memory[0, 1, 0] = 1
        hits = np.zeros((3, 2), np.float32)
        hits[1, 1] = 1
        data = dict(ca3=memory, successor_ca3=np.zeros_like(memory), action=np.zeros(3, int),
                    successor_dg=hits, terminated=np.array([False, True, True]))
        prediction, labels, usable = complete_replay_values(Constant(), data, (1,), .99, batch_size=2)
        self.assertEqual(usable[:, 0].tolist(), [False, True, True])
        self.assertEqual(labels[1:, 0].tolist(), [1., 0.])
        metrics = residual_metrics(prediction, labels, usable)
        self.assertEqual(metrics["labels"], 2)
        self.assertEqual(metrics["omitted_labels"], 1)


if __name__ == "__main__":
    unittest.main()
