import unittest

import numpy as np
import torch

from hpc_runs.offpolicy_goal_baselines import (
    ContrastiveGoalAgent,
    EpisodeReplay,
    farthest_point_indices,
    floyd_warshall_next,
)
from hpc_runs.offpolicy_goal_baselines.planner import LandmarkPlanner
from hpc_runs.offpolicy_goal_baselines.train import FrozenVisualFeatures
from hpc_runs.offpolicy_goal_baselines.train_parallel import _select_info


class OffPolicyGoalBaselineTest(unittest.TestCase):
    def test_frozen_feature_batch_encodes_once_and_appends_instruction(self):
        class MeanTrunk(torch.nn.Module):
            def forward(self, image):
                return image.mean(dim=(-2, -1))

        extractor = object.__new__(FrozenVisualFeatures)
        extractor.trunk = MeanTrunk()
        extractor.device = torch.device("cpu")
        extractor.number_instruction_coef = 9.0
        observations = {
            "obs": np.stack(
                (np.full((3, 2, 2), 255, dtype=np.uint8), np.zeros((3, 2, 2), dtype=np.uint8))
            ),
            "instruction": np.asarray([1, 3]),
            "telemetry_pose": np.zeros((2, 3), dtype=np.float32),
        }
        features = extractor.batch(observations)
        self.assertEqual(features.shape, (2, 6))
        np.testing.assert_allclose(features[0], [1, 1, 1, 9, 0, 0])
        np.testing.assert_allclose(features[1], [0, 0, 0, 0, 0, 9])

    def test_vector_info_prefers_terminal_episode_info(self):
        infos = {
            "score": np.asarray([1.0, 2.0]),
            "_score": np.asarray([True, True]),
            "final_info": np.asarray([None, {"episode_extra_stats": {"coverage": 7.0}}], dtype=object),
            "_final_info": np.asarray([False, True]),
        }
        self.assertEqual(_select_info(infos, 0, 2)["score"], 1.0)
        self.assertEqual(
            _select_info(infos, 1, 2)["episode_extra_stats"]["coverage"], 7.0
        )
        nested_infos = {
            "final_info": {
                "episode_extra_stats": {
                    "coverage": np.asarray([5.0, 7.0]),
                    "_coverage": np.asarray([True, True]),
                },
                "_episode_extra_stats": np.asarray([True, True]),
            },
            "_final_info": np.asarray([True, True]),
        }
        self.assertEqual(
            _select_info(nested_infos, 1, 2)["episode_extra_stats"]["coverage"], 7.0
        )

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

    def test_planner_uses_landmark_outside_local_horizon(self):
        class DistanceAgent:
            def eval(self):
                return self

            def temporal_distance(self, state, goal):
                return torch.log1p(torch.abs(goal[:, 0] - state[:, 0]))

        planner = LandmarkPlanner(landmark_count=2, candidates=2, neighbors=1, local_horizon=2)
        planner.features = np.asarray([[3.0], [7.0]], dtype=np.float32)
        planner.graph_cost = np.asarray([[0.0, 4.0], [np.inf, 0.0]])
        subgoal = planner.subgoal(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([10.0], dtype=np.float32),
            DistanceAgent(),
            torch.device("cpu"),
        )
        np.testing.assert_array_equal(subgoal, np.asarray([3.0], dtype=np.float32))
        self.assertEqual(planner.landmark_subgoals, 1)


if __name__ == "__main__":
    unittest.main()
