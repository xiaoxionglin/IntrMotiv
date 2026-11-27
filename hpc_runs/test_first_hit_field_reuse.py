"""Saved field mapping and complete stored-graph denominator contracts."""

import unittest
import numpy as np

from hpc_runs.first_hit_control.field_reuse import graph_summary, image_values


class FieldReuseTests(unittest.TestCase):
    def test_historical_xy_transposes_and_unvisited_is_masked(self):
        maps = np.arange(12).reshape(2, 3, 2)
        occupancy = np.ones((2, 3)); occupancy[1, 0] = 0
        displayed = image_values(maps, occupancy, 1)
        self.assertEqual(displayed.shape, (3, 2))
        self.assertEqual(displayed[2, 0], maps[0, 2, 1])
        self.assertTrue(displayed.mask[0, 1])

    def test_success_mass_not_probability_and_all_graph_entries_preserved(self):
        data = {key: np.zeros((16, 16)) for key in ("control_attempts", "control_edge_confidence", "control_tctrl")}
        data["control_attempts"][0, 1] = 10
        data["control_edge_confidence"][0, 1] = 6
        data["control_tctrl"][0, 1] = 4
        rows, summary = graph_summary(data)
        self.assertEqual(len(rows), 256)
        self.assertEqual(sum(r["diagonal"] for r in rows), 16)
        self.assertEqual(rows[1]["stored_success_mass"], 6)
        self.assertAlmostEqual(rows[1]["stored_smoothed_success_ratio"], 7/12, places=6)
        self.assertTrue(rows[1]["canonical_reliable_edge"])
        self.assertFalse(summary["prospective_evidence_available"])
        self.assertFalse(any("prospective_success" in key or "grounded" in key for key in summary))


if __name__ == "__main__":
    unittest.main()
