"""Check interval direction and agreement between reward summaries and curves."""
import unittest
import numpy as np
from analyze_poster_reward_auc import mean_reward_auc, binned_curve, mean_reward_window


class RewardIntegralTests(unittest.TestCase):
    def test_nonuniform_bins_reproduce_auc(self):
        steps, rewards = np.array([0., 10., 20., 30.]), np.array([99., 1., 3., 2.])
        edges = np.array([0., 15., 22., 30.])
        curve = binned_curve(steps, rewards, 30, edges)
        np.testing.assert_allclose(curve, [5/3, 19/7, 2])
        self.assertAlmostEqual(np.average(curve, weights=np.diff(edges)), 2)
        self.assertAlmostEqual(mean_reward_auc(steps, rewards, 30), 2)
        self.assertAlmostEqual(mean_reward_window(steps, rewards, 10, 30), 2.5)

    def test_endpoint_carry_preserves_existing_auc_convention(self):
        steps, rewards = np.array([10., 20., 30.]), np.array([1., 3., 999.])
        self.assertAlmostEqual(mean_reward_auc(steps, rewards, 25), 55/25)
        np.testing.assert_allclose(binned_curve(steps, rewards, 25, np.array([0, 10, 25])), [1, 3])

    def test_window_excludes_future_samples(self):
        steps = np.array([10., 20., 30., 40.])
        self.assertAlmostEqual(mean_reward_window(steps, np.array([1., 3., 2., 999.]), 20, 30), 2)

    def test_duplicate_frames_and_invalid_bins_fail(self):
        with self.assertRaises(ValueError):
            mean_reward_auc(np.array([0., 0.]), np.array([1., 2.]), 20)
        with self.assertRaises(ValueError):
            binned_curve(np.array([0., 10.]), np.array([1., 2.]), 20, np.array([0, 20, 10]))

    def test_unfinished_window_fails(self):
        with self.assertRaises(ValueError):
            mean_reward_window(np.array([0., 10.]), np.array([1., 2.]), 10, 30)


if __name__ == '__main__':
    unittest.main()
