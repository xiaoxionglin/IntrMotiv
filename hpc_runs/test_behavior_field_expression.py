"""Focused regression checks for frozen-policy field probes."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from hpc_runs.behavior_field_metrics import normalized_information, paired_information
from hpc_runs.behavior_field_probe import (
    assert_alignment, capture_decoder_module, choose_executed_action,
    frozen_digest, reset_recurrent_state,
)


class BehaviorFieldTests(unittest.TestCase):
    def test_information_is_gain_invariant(self):
        rate = np.array([[0.1, 0.3], [0.0, 0.6]])
        weight = np.ones((2, 2))
        self.assertAlmostEqual(normalized_information(rate, weight),
                               normalized_information(rate * 17, weight))

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
        random["occupancy"][:] = 0
        self.assertIsNone(paired_information(own, random)["difference"])

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


if __name__ == "__main__":
    unittest.main()
