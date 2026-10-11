"""Guard the training-seed aggregation against evaluation-repeat inflation."""

import importlib
import unittest


class BehaviorSeedSummaryTests(unittest.TestCase):
    def test_requires_both_finite_evaluation_repeats(self):
        complete_mean = importlib.import_module(
            "06_experiments.cosyne_2027.summarize_behavior_expression"
        ).complete_mean
        first = {"eval_seed": "51000", "information_difference": "0.2"}
        second = {"eval_seed": "52000", "information_difference": "0.4"}
        self.assertIsNone(complete_mean([first], "information_difference"))
        self.assertAlmostEqual(complete_mean([first, second], "information_difference"), 0.3)
        second["information_difference"] = "nan"
        self.assertIsNone(complete_mean([first, second], "information_difference"))


if __name__ == "__main__":
    unittest.main()
