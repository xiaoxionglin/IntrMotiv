"""Verify factual residual accounting without creating a native environment."""

import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.fit_diagnostics import diagnose, residual_statistics
from hpc_runs.test_first_hit_replay_fit import model, replay


def test_residual_groups_preserve_labels_and_frozen_model():
    torch.set_num_threads(1)
    readout = model().eval().requires_grad_(False)
    data = replay()
    originals = {key: value.copy() for key, value in data.items()}
    # An unknown terminal cannot become a target zero in any stratum.
    data["successor_valid"][0] = False
    originals["successor_valid"][0] = False
    result = diagnose(readout, data, dict(horizon=2, holdout_fraction=0, seed=99), "winner", samples_per_bin=32)
    assert result["readout_unchanged"]
    assert all(row["missing"] for row in result["panels"]["holdout"])
    for row in result["panels"]["train"]:
        assert row["all"]["labels"] == 30
        assert sum(group["labels"] for group in row["per_goal"]) == 30
        assert sum(group["labels"] for group in row["per_action"]) == 30
        assert row["immediate_hit"]["labels"] == row["not_immediate_hit"]["labels"] == 15
    assert all(np.array_equal(value, data[key]) for key, value in originals.items())
    assert result == diagnose(readout, data, dict(horizon=2, holdout_fraction=0, seed=99), "winner", samples_per_bin=32)


def test_unknown_only_statistics_stay_missing():
    result = residual_statistics(np.array([.5]), np.array([np.nan]), np.array([False]))
    assert result["labels"] == 0 and result["mse"] is None
    with pytest.raises(ValueError, match="frozen evaluation"):
        diagnose(model(), replay(), dict(horizon=2, holdout_fraction=.1, seed=99), "winner")
