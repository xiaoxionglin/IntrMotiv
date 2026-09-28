"""Regression checks for complete trajectory input and episode boundaries."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import matplotlib.pyplot as plt

from render_goal_option_trajectory import render, trajectory_segments, validate_complete_event_stream


class TrajectoryRenderingTests(unittest.TestCase):
    def pose(self):
        return pd.DataFrame({"frame": range(6), "agent": [0] * 6,
                             "num_traj": [0, 0, 0, 1, 1, 1],
                             "x": [100, 150, 200, 1800, 1850, 1900],
                             "y": [100, 150, 100, 1800, 1850, 1800],
                             "option_start": [True, False, False, True, False, False],
                             "goal_hit": [False, False, True, False, False, True]})

    def test_sparse_event_table_cannot_be_drawn_as_a_trajectory(self):
        pose = self.pose()
        events = pose[pose.option_start | pose.goal_hit]
        with self.assertRaisesRegex(ValueError, "pose_events.csv"):
            validate_complete_event_stream(events)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist.svg"
            with self.assertRaisesRegex(ValueError, "pose_events.csv"):
                render(events, output, "Test", "full", (100, 2000, 100, 2000))
            self.assertFalse(output.exists())

    def test_reused_episode_ids_and_skipped_frames_break_paths(self):
        pose = self.pose().iloc[:5].copy()
        pose["frame"] = [0, 1, 2, 3, 5]
        pose["num_traj"] = [0, 0, 1, 0, 0]
        paths = [segment.frame.tolist() for segment in trajectory_segments(pose)]
        self.assertEqual(paths, [[0, 1], [2], [3], [5]])

    def test_agents_have_separate_paths(self):
        pose = self.pose().iloc[:4].copy()
        pose["agent"] = [0, 1, 0, 1]
        pose["frame"] = [0, 0, 1, 1]
        pose["num_traj"] = 0
        paths = list(trajectory_segments(pose))
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(segment.agent.nunique() == 1 for segment in paths))

    def test_renderer_does_not_connect_episode_end_to_next_start(self):
        pose = self.pose()
        validate_complete_event_stream(pose, expected_observations=6)
        fig, ax = plt.subplots()
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                "render_goal_option_trajectory.plt.subplots", return_value=(fig, ax)
            ), patch("render_goal_option_trajectory.plt.close"):
                render(pose, Path(directory) / "trajectory.svg", "Test", "full", (100, 2000, 100, 2000))
            self.assertEqual([list(line.get_xdata()) for line in ax.lines],
                             [[100, 150, 200], [1800, 1850, 1900]])
            self.assertTrue(all(not line.get_path().should_simplify for line in ax.lines))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
