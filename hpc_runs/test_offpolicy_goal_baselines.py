import unittest

import numpy as np
import torch

from hpc_runs.offpolicy_goal_baselines import (
    ContrastiveGoalAgent,
    EpisodeReplay,
    farthest_point_indices,
    floyd_warshall_next,
)


class OffPolicyGoalBaselineTest(unittest.TestCase):
    def test_replay_future_pairs_are_ordered_and_bounded(self):
        replay = EpisodeReplay(100, seed=3)
        features = np.arange(11, dtype=np.float32)[:, None]
        replay.add(features, np.arange(10), np.zeros((11, 3)))
        batch = replay.sample(64, max_future=4)
        self.assertTrue(np.all(batch["future_goal"][:, 0] > batch["state"][:, 0]))
        self.assertTrue(np.all(batch["offset"] <= 4))

    def test_crl_shapes_and_backward(self):
        agent = ContrastiveGoalAgent(7, 5, hidden_dim=16, repr_dim=4, action_dim=3)
        state = torch.randn(8, 7)
        goal = torch.randn(8, 7)
        action = torch.randint(0, 5, (8,))
        critic, actor, metrics = agent.losses(state, action, goal, torch.ones(8), goal.roll(1, 0))
        (critic + actor).backward()
        self.assertEqual(agent.all_action_scores(state, goal).shape, (8, 5))
        self.assertIn("critic_accuracy", metrics)

    def test_farthest_points(self):
        points = np.asarray([[0.0], [1.0], [10.0], [11.0]])
        chosen = farthest_point_indices(points, 2)
        self.assertEqual(chosen.tolist(), [0, 3])

    def test_directed_shortest_path(self):
        inf = np.inf
        cost = np.asarray([[0, 1, 10], [inf, 0, 2], [inf, inf, 0]], dtype=float)
        distance, next_hop = floyd_warshall_next(cost)
        self.assertEqual(distance[0, 2], 3)
        self.assertEqual(next_hop[0, 2], 1)


if __name__ == "__main__":
    unittest.main()
