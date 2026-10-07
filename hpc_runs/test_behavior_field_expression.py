"""Focused regression checks for frozen-policy field probes."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import csv

import numpy as np
import torch
from torch import nn

from hpc_runs.behavior_field_metrics import (
    comparison_details, information_scores, layer_details, normalized_information,
    paired_information,
)
from hpc_runs.behavior_field_probe import (
    assert_alignment, capture_decoder_module, choose_executed_action,
    configure_workspace_paths,
    frozen_digest, reset_recurrent_state,
)
from hpc_runs.behavior_field_study import CONDITIONS, SEEDS, FIELDS, make_rows, read_manifest
from hpc_runs.behavior_field_analysis import condition_summaries


class BehaviorFieldTests(unittest.TestCase):
    def test_information_is_gain_invariant(self):
        self.assertEqual(information_scores(np.array([[2.0, 0.0]]),
                                            np.ones((1, 2))), (1.0, 1.0))
        self.assertEqual(information_scores(np.array([[4.0, 0.0]]),
                                            np.ones((1, 2))), (1.0, 2.0))
        rate = np.array([[0.1, 0.3], [0.0, 0.6]])
        weight = np.ones((2, 2))
        self.assertAlmostEqual(normalized_information(rate, weight),
                               normalized_information(rate * 17, weight))
        per_activation, per_step = information_scores(rate, weight)
        scaled_activation, scaled_step = information_scores(rate * 17, weight)
        self.assertAlmostEqual(per_activation, scaled_activation)
        self.assertAlmostEqual(scaled_step, per_step * 17)
        self.assertAlmostEqual(per_step, per_activation * rate.mean())

    def test_common_support_and_same_units(self):
        occupancy = np.full((19, 19), 10)
        rate = np.ones((19, 19, 2))
        rate[0, 0, 0] = 9
        own = {"occupancy": occupancy, "rate_maps": rate,
               "field_eligible": np.array([True, True])}
        random = {"occupancy": occupancy.copy(), "rate_maps": rate * 2,
                  "field_eligible": np.array([True, False])}
        result = paired_information(own, random)
        self.assertEqual(result["shared_bins"], 361)
        self.assertEqual(result["eligible_units"], 1)
        self.assertAlmostEqual(result["difference"], 0)
        self.assertAlmostEqual(result["bits_per_step_difference"],
                               -result["own_bits_per_step"])
        self.assertAlmostEqual(result["mean_activation_difference"],
                               -result["own_mean_activation"])
        random["occupancy"][:] = 0
        self.assertIsNone(paired_information(own, random)["difference"])
        self.assertIsNone(paired_information(own, random)["bits_per_step_difference"])
        self.assertIsNone(paired_information(own, random)["mean_activation_difference"])

    def test_fast_prefix_maps_match_canonical_contract(self):
        rng = np.random.default_rng(9)
        pose = np.column_stack((rng.uniform(100, 2000, 400),
                                rng.uniform(100, 2000, 400), np.zeros(400)))
        activity = rng.exponential(size=(400, 3)).astype(np.float32)
        activity[rng.random(activity.shape) < .3] = 0
        slow = layer_details(pose, activity)
        fast = comparison_details(pose, activity)
        np.testing.assert_array_equal(slow["occupancy"], fast["occupancy"])
        np.testing.assert_allclose(slow["rate_maps"], fast["rate_maps"], atol=1e-6)
        np.testing.assert_array_equal(slow["field_eligible"], fast["field_eligible"])

    def test_action_override_and_episode_reset(self):
        rng = np.random.default_rng(123)
        self.assertEqual(choose_executed_action("own", 4, rng, 8, 0, None), 4)
        actions = []
        previous = None
        for t in range(16):
            previous = choose_executed_action("persistent8", 4, rng, 8, t, previous)
            actions.append(previous)
        self.assertEqual(len(set(actions[:8])), 1)
        self.assertEqual(len(set(actions[8:])), 1)
        state = torch.ones((1, 4))
        self.assertTrue(torch.equal(reset_recurrent_state(state, True), torch.zeros_like(state)))
        self.assertRaises(RuntimeError, assert_alignment, {"a": [0], "b": []}, 1)

    def test_two_decoder_hooks_and_frozen_digest(self):
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                activation = nn.ReLU()
                self.decoder = nn.Sequential(nn.Linear(3, 3), activation,
                                             nn.Linear(3, 3), activation)
                self.bn = nn.BatchNorm1d(3)

        actor = Actor().eval()
        name, module = capture_decoder_module(actor)
        self.assertEqual(name, "decoder.1")
        outputs = []
        handle = module.register_forward_hook(lambda _m, _i, output: outputs.append(output.detach().clone()))
        actor.decoder(torch.ones((1, 3)))
        handle.remove()
        self.assertEqual(len(outputs), 2)
        initial = frozen_digest(actor)
        with torch.no_grad():
            actor.bn.running_mean.add_(1)
        self.assertNotEqual(initial, frozen_digest(actor))

    def test_historical_output_paths_are_overridden_before_env(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = SimpleNamespace(train_dir="/retired/train", dmlab_level_cache_path="/retired/cache",
                                  wandb_dir="/retired/wandb", with_wandb=True)
            paths = configure_workspace_paths(cfg, root / "runs" / "run", root / "analysis", root)
            self.assertEqual(cfg.train_dir, str(root / "runs"))
            self.assertEqual(cfg.dmlab_level_cache_path, str(root / "analysis" / "dmlab_cache"))
            self.assertFalse(cfg.with_wandb)
            self.assertTrue(all(path.startswith(str(root)) for path in paths.values()))

    def test_manifest_requires_complete_paired_matrix(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "poster.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("condition", "seed", "checkpoint_frames", "checkpoint"))
                writer.writeheader()
                for condition in CONDITIONS:
                    for seed in SEEDS:
                        writer.writerow({"condition": condition, "seed": seed,
                                         "checkpoint_frames": 100_040_704,
                                         "checkpoint": f"/retired/{CONDITIONS[condition]}_S{seed}/checkpoint_100040704.pth"})
            rows = make_rows(source, root)
            self.assertEqual(len(rows), 54)
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(read_manifest(manifest, root, require_inputs=False)), 54)
            rows[-1]["policy"] = "own"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            self.assertRaises(ValueError, read_manifest, manifest, root, False)

    def test_condition_summary_does_not_count_eval_repeats_as_seeds(self):
        rows = [{"condition": "C01", "seed": "8", "eval_seed": str(eval_seed),
                 "random_policy": "uniform", "layer": "dg", "decisions": prefix,
                 "supported": True, "difference": .2,
                 "bits_per_step_difference": -.3,
                 "mean_activation_difference": .4}
                for eval_seed in (51000, 52000) for prefix in (40000, 50000)]
        summary = next(row for row in condition_summaries(rows)
                       if row["condition"] == "C01" and row["random_policy"] == "uniform"
                       and row["layer"] == "dg")
        self.assertEqual(summary["supported_training_seeds"], 1)
        self.assertFalse(summary["positive_all_three"])
        self.assertEqual(summary["bits_per_step_supported_training_seeds"], 1)
        self.assertAlmostEqual(summary["bits_per_step_mean_difference"], -.3)
        self.assertFalse(summary["bits_per_step_positive_all_three"])
        self.assertAlmostEqual(summary["activation_mean_difference"], .4)


if __name__ == "__main__":
    unittest.main()
