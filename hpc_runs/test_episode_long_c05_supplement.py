"""Guard the expensive C05 supplement's declared C15 comparison."""

import unittest
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study


STUDIES = Path(__file__).resolve().parent / "studies"
TRACKING_KEYS = {"study_id", "study_condition", "study_base", "wandb_group", "wandb_tags"}
SCIENTIFIC_DIFFERENCES = {
    "dg_global_punishment_coeff": ("0.01", "0.0"),
    "dg_row_repulsion_coeff": ("1.0", "0.0"),
    "hrl_manager_mode": ("visit_direct", "frontier_direct"),
}


def options(run):
    result = {}
    for arg in run.args:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            result[key] = value
    return result


class EpisodeLongC05SupplementTest(unittest.TestCase):
    def test_every_c05_seed_matches_c15_except_declared_family_settings(self):
        supplement = load_study(STUDIES / "episode_long_c05_supplement_20261008.study.json")
        original = load_study(STUDIES / "episode_long_dg_controller_20261008.study.json")
        c05 = {run.seed: run for run in supplement.expand_runs()}
        c15 = {run.seed: run for run in original.expand_runs() if run.factors["arm"] == "c15_film"}
        self.assertEqual(set(c05), {8, 99, 123})
        self.assertEqual(set(c05), set(c15))

        for seed in sorted(c05):
            with self.subTest(seed=seed):
                left, right = options(c05[seed]), options(c15[seed])
                for key in TRACKING_KEYS:
                    left.pop(key, None)
                    right.pop(key, None)
                differences = {key: (left.get(key), right.get(key))
                               for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
                self.assertEqual(differences, SCIENTIFIC_DIFFERENCES)


if __name__ == "__main__":
    unittest.main()
