"""Pure synthetic projection/mask/selection contracts; no native readout load."""

import unittest
import numpy as np
import torch
from types import SimpleNamespace

from hpc_runs.first_hit_control.affine_deadline_audit import (
    audit_selected, boundary_metrics, project_curve, select_rows, selected_row_receipt,
)
from hpc_runs.first_hit_control.replay_fit import episode_split


class AffineDeadlineTests(unittest.TestCase):
    def test_affine_exact_targets_have_zero_floor_and_exact_excess(self):
        h = np.arange(1, 257)
        target = .3 + .5 * h / 256
        prediction = target + .2
        row = project_curve(prediction, target, h)
        self.assertLess(row["affine_floor_mse"], 1e-28)
        self.assertAlmostEqual(row["excess_mse"], .04)
        self.assertAlmostEqual(row["total_mse"], .04)
        self.assertLess(abs(row["decomposition_remainder"]), 1e-14)

    def test_piecewise_boundary_has_irreducible_floor_and_known_projection(self):
        h = np.arange(1, 5)
        target = np.array([0., 1., 1., 1.])
        row = project_curve(.3*h, target, h)
        self.assertAlmostEqual(row["affine_floor_mse"], .075)
        self.assertAlmostEqual(row["excess_mse"], 0)
        self.assertAlmostEqual(row["intercept"], 0)
        self.assertAlmostEqual(row["normalized_deadline_slope"], 1.2)

    def test_weighted_masks_equal_independent_least_squares(self):
        h = np.arange(1, 9, dtype=float)
        target = np.array([0., .4, np.nan, 1., .2, .8, .1, .7])
        weight = np.array([2., 0., 99., 1., 3., 1., 4., 2.])
        usable = np.array([True, True, False, True, True, True, True, True])
        prediction = .2 + .1*h
        row = project_curve(prediction, target, h, weights=weight, usable=usable)
        selected = usable & (weight > 0)
        design = np.column_stack([np.ones(len(h)), h/8])
        coefficients = np.linalg.lstsq(design[selected] * np.sqrt(weight[selected, None]),
                                       target[selected] * np.sqrt(weight[selected]), rcond=None)[0]
        expected = design @ coefficients
        floor = np.average((expected[selected] - target[selected])**2, weights=weight[selected])
        self.assertAlmostEqual(row["affine_floor_mse"], floor)
        self.assertEqual(row["usable_points"], 6)
        self.assertLess(abs(row["decomposition_remainder"]), 1e-14)

    def test_unequal_analytic_h0_is_excluded(self):
        h = np.arange(5, dtype=float)
        prediction = np.array([-1000., .3, .6, .9, 1.2])
        target = np.array([999., 0., 1., 1., 1.])
        with_zero = project_curve(prediction, target, h)
        without = project_curve(prediction[1:], target[1:], h[1:])
        self.assertEqual(with_zero["excluded_h0_points"], 1)
        self.assertEqual(with_zero["total_mse"], without["total_mse"])
        self.assertEqual(with_zero["affine_floor_mse"], without["affine_floor_mse"])

    def test_nonaffine_deployed_predictions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "affine-deadline"):
            project_curve(np.array([0., 1., 1., 1.]), np.zeros(4), np.arange(1, 5))

    def test_one_usable_point_is_retained_not_dropped(self):
        row = project_curve(np.arange(1, 5)*.2, np.array([1., np.nan, np.nan, np.nan]),
                            np.arange(1, 5), usable=np.array([True, False, False, False]))
        self.assertEqual(row["projection_rank"], 1)
        self.assertEqual(row["usable_points"], 1)
        self.assertEqual(row["affine_floor_mse"], 0)
        self.assertAlmostEqual(row["excess_mse"], .64)

    def test_uniform_selection_preserves_episode_split_and_does_not_filter_core(self):
        data = dict(episode_id=np.repeat(np.arange(20), 20), successor_valid=np.ones(400, bool),
                    successor_core_valid=np.zeros(400, bool))
        cfg = dict(holdout_fraction=.2, seed=99)
        protocol = dict(sampling=dict(seed=123, train_rows=32, holdout_rows=32))
        selected = select_rows(data, cfg, protocol)
        self.assertEqual(len(selected["train"]), 32)
        self.assertEqual(len(np.unique(selected["holdout"])), 32)
        self.assertFalse(set(data["episode_id"][selected["train"]]) & set(data["episode_id"][selected["holdout"]]))
        rng = np.random.default_rng(123)
        train, holdout = episode_split(data["episode_id"], .2, 99)
        np.testing.assert_array_equal(selected["train"], rng.choice(train, 32, replace=False))
        np.testing.assert_array_equal(selected["holdout"], rng.choice(holdout, 32, replace=False))
        data["successor_valid"][:] = False
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_rows(data, cfg, protocol)

    def test_row_hash_binds_values_and_boundary_zero_baseline(self):
        data = dict(action=np.array([1]), episode_id=np.array([4]), core=np.array([[1., 2.]]),
                    successor_valid=np.array([True]), successor_core_valid=np.array([False]), physical_done=np.array([True]))
        first = selected_row_receipt(data, {"train": np.array([0])})
        data["core"][0, 1] = 3
        second = selected_row_receipt(data, {"train": np.array([0])})
        self.assertNotEqual(first["rows"][0]["content_sha256"], second["rows"][0]["content_sha256"])
        metrics = boundary_metrics(np.array([.5, .5, .5]), np.array([1., 0., 0.]), np.ones(3, bool))
        self.assertAlmostEqual(metrics["all"]["zero_prediction_mse"], 1/3)
        self.assertAlmostEqual(metrics["all"]["mse"], .25)

    def test_synthetic_adapter_keeps_all_curves_and_canonical_core_masks(self):
        class FakeAffineAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_heads = torch.nn.Module()
                self.q_heads.main = torch.nn.Linear(1, 5)
                self.layout = SimpleNamespace(width=32, goals=16)
                self.horizon_scale = 256
                self.eligibility = {"half_open_slices": {"canonical_ca3": [0, 16]}}

            def all_goals(self, core, clocks):
                values = (.1 + .001 * clocks[:, None, None] + .01 * torch.arange(16)[None, :, None]
                          + .02 * torch.arange(5)[None, None, :])
                return torch.where(clocks[:, None, None] > 0, values, torch.zeros_like(values))

        model = FakeAffineAdapter().eval().requires_grad_(False)
        core = np.zeros((64, 32), np.float32)
        core[np.arange(64), np.arange(64) % 16] = 1
        dg = np.zeros((64, 16), np.float32); dg[:, 3] = 1
        data = dict(core=core, successor_core=core.copy(), successor_dg=dg, action=np.arange(64) % 5,
            episode_id=np.arange(64), physical_done=np.zeros(64, bool), successor_valid=np.ones(64, bool),
            successor_core_valid=np.ones(64, bool))
        data["successor_core_valid"][0] = False
        data["physical_done"][1] = True
        protocol = dict(curves=dict(horizons=list(range(1, 257)), goals=list(range(16)), declared_curves=1024),
                        projection=dict(affinity_tolerance=dict(absolute=2e-5, relative=2e-5),
                                        remainder_tolerance=dict(absolute=1e-8, relative=1e-6)))
        rows, summary = audit_selected(model, data, {"train": np.arange(32), "holdout": np.arange(32, 64)}, protocol, "winner")
        self.assertEqual(len(rows), 1024)
        self.assertEqual(sum(row["initially_at_goal"] for row in rows), 64)
        # One nonterminal row lacks successor core: h1 and immediate-hit labels
        # remain valid, other h>1 goal labels are masked, not trained as zero.
        self.assertEqual(sum(row["projection_rank"] == 1 for row in rows), 15)
        self.assertEqual(summary["declared_curve_points"], 262144)
        self.assertEqual(summary["panels"]["train"]["h1"]["all"]["labels"], 512)
        self.assertTrue(summary["readout_unchanged"])


if __name__ == "__main__":
    unittest.main()
