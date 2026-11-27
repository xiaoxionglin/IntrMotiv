"""Focused post-hoc adapter checks; these are not efficacy fits."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.neural import evaluate_readout, state_digest
from hpc_runs.first_hit_control.projected_regression import (
    InitialAudit, PROTOCOL_SHA, experiments, prepare, reconstruct_sampling,
    regression_config, validate_reference,
)
from hpc_runs.first_hit_control.provenance import audit_source_snapshots
from hpc_runs.first_hit_control.stationary_readout import (
    StationaryReadout, fit_stationary, original_classes, summarize_trials,
)


def test_protocol_reference_source_and_exact_old_settings():
    protocol, old, matched = validate_reference()
    config = regression_config()
    assert config.horizon == 6 and config.updates == 2500
    assert config.seeds == (8, 99, 123) and config.batch_size == 256
    assert protocol["new_fit_count"] == 6 and old["gamma"] == .99
    assert len(matched) == 10


def test_every_declared_task_query_and_first_denominator():
    class Constant(torch.nn.Module):
        def forward(self, memory, goals, remaining=None, route="first_hit"):
            return torch.tensor([[1., 0., 0.]]).expand(len(goals), -1)
    expected = [(42, 7, 210), (4, 0, 4)]
    for (env, dg, goals, _), (count, initial, first_count) in zip(experiments(), expected):
        rows = evaluate_readout(Constant(), env, dg, goals, regression_config(), 8)
        summary, first = summarize_trials(rows, env, regression_config())
        assert summary["n_nontrivial_pairs"] == count
        assert summary["initially_satisfied_queries"] == initial
        assert len(first) == first_count
        assert all(row["contrast"] == 0 for row in first)


def test_reconstructed_rng_masks_match_observed_new_fit(tmp_path):
    torch.set_num_threads(1)
    config = replace(regression_config(), updates=3, batch_size=8)
    data = dict(ca3=np.zeros((12, 2, 8), np.float32),
                successor_ca3=np.zeros((12, 2, 8), np.float32),
                successor_dg=np.zeros((12, 2), np.float32),
                action=np.arange(12) % 3, terminated=np.zeros(12, bool))
    data["ca3"][:6, 0, 0] = 1
    data["successor_dg"][1::2, 1] = 1
    goals, seed = (0, 1), 99
    decoder, action, _ = original_classes()
    settings = dict(decoder_mlp_layers=[128, 128], nonlinearity="relu", use_jit=True)
    factory = lambda: StationaryReadout(2, config, decoder, action, settings)
    audit = InitialAudit(tmp_path / "initial.pt")
    model, _ = fit_stationary(factory, data, goals, config, seed, batch_function=audit)
    expected = reconstruct_sampling(data, goals, config, seed)
    assert all(audit.summary()[key] == value for key, value in expected.items())
    restored = factory()
    restored.load_state_dict(torch.load(tmp_path / "initial.pt", weights_only=True))
    assert state_digest(restored) == audit.initial
    assert state_digest(model) != audit.initial
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        assert state_digest(factory()) == audit.initial


def test_preparation_archives_before_fitting_and_refuses_overwrite(tmp_path):
    output = tmp_path / "prepared"
    result = prepare(output)
    assert result["protocol_sha256"] == PROTOCOL_SHA
    assert result["new_fit_count"] == 6 and len(result["datasets"]) == 6
    assert "not recorded historical hashes" in result["matching_scope"]
    assert not (output / "fit_started.json").exists()
    assert not list(output.rglob("*.pt"))
    assert json.loads((output / "prepared.json").read_text())["gamma"] == .99
    audit = audit_source_snapshots(output / "source_snapshot")
    assert audit["archive_valid"] and audit["current_sources_unchanged"]
    with pytest.raises(FileExistsError):
        prepare(output)
