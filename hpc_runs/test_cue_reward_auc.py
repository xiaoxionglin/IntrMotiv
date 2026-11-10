"""Sparse reward events must survive plotting and analysis resolution changes."""

import unittest

from hpc_runs.analyze_cue_transfer_results import normalized_reward_auc


class RewardAUCTest(unittest.TestCase):
    def test_spike_between_plot_points_is_retained(self):
        self.assertAlmostEqual(normalized_reward_auc([0, 1, 2, 100], [0, 10, 0, 0], 0, 100), 0.1)

    def test_window_cut_through_triangle(self):
        self.assertAlmostEqual(normalized_reward_auc([0, 2, 4], [0, 4, 0], 1, 3), 3.0)

    def test_endpoint_hold_is_explicit(self):
        self.assertAlmostEqual(normalized_reward_auc([2, 4], [3, 3], 0, 10), 3.0)


if __name__ == "__main__":
    unittest.main()
