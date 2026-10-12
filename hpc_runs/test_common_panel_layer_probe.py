"""Focused checks for common-panel layer information units and activity gates."""

import numpy as np

from hpc_runs.common_panel_layer_probe import summarize_layer


def test_gain_changes_amplitude_score_but_not_bits_per_activation():
    pose = np.zeros((120, 3), dtype=np.float32)
    pose[:, 0] = 150 + (np.arange(120) % 4) * 100
    pose[:, 1] = 150
    activity = np.zeros((120, 2), dtype=np.float32)
    activity[:, 0] = np.where(np.arange(120) % 4 == 0, 2.0, 0.25)
    first, occupancy = summarize_layer(pose, activity)
    scaled, scaled_occupancy = summarize_layer(pose, activity * 3)
    assert np.array_equal(occupancy, scaled_occupancy)
    assert first["visited_bins"] == 4
    assert first["eligible_units"] == scaled["eligible_units"] == 1
    assert first["silent_units"] == scaled["silent_units"] == 1
    assert np.isclose(first["mean_bits_per_activation_eligible"],
                      scaled["mean_bits_per_activation_eligible"])
    assert np.isclose(3 * first["mean_amplitude_weighted_bits_all"],
                      scaled["mean_amplitude_weighted_bits_all"])
