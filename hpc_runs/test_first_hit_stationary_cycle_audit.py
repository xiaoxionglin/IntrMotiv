"""Synthetic exact-cycle witnesses; never load the twelve audited policies."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from hpc_runs.first_hit_control.finite import FiniteMDP
from hpc_runs.first_hit_control.neural import observe
from hpc_runs.first_hit_control.projected_regression import experiments, regression_config
from hpc_runs.first_hit_control.stationary_cycle_audit import (
    PROTOCOL_SHA, augmented_key, audit_query, prepare, summarize, validate_finite_system,
)


def system():
    transitions = np.eye(2)[np.array([[0, 1], [1, 1]])]
    env = FiniteMDP("cycle_fixture", transitions, np.array([False, True]), (1,), (0,))
    return env, np.eye(2, dtype=np.float32)


class AlwaysWait(torch.nn.Module):
    def forward(self, memory, goals):
        return torch.tensor([[1., 0.]])


class AlwaysGo(torch.nn.Module):
    def forward(self, memory, goals):
        return torch.tensor([[0., 1.]])


def test_physical_node_repeat_is_not_a_cycle_when_history_changes():
    class WaitUntilHistory(torch.nn.Module):
        def forward(self, memory, goals):
            return torch.tensor([[0., 1.]]) if memory[0, 0, 2] > 0 else torch.tensor([[1., 0.]])
    env, dg = system()
    memory = observe(torch.zeros(1, 2, 8), dg[0], 1)
    row = audit_query(WaitUntilHistory(), env, dg, 0, memory, 1, 1, cap=100)
    assert row["outcome"] == "hit" and row["first_hit_latency"] == 3
    assert json.loads(row["states"]) == [0, 0, 0, 1]
    assert row["cycle_period"] is None


def test_full_history_recurrence_proves_cycle_and_preserves_witness():
    env, _ = system()
    dg = np.zeros((2, 2), np.float32)
    memory = torch.zeros(1, 2, 8)
    memory[0, 0, 0] = 1  # A cue expires while physical node stays the same.
    row = audit_query(AlwaysWait(), env, dg, 0, memory, 1, 1, cap=100)
    assert row["outcome"] == "proven_cycle_nonarrival"
    assert row["cycle_preperiod"] == 8 and row["cycle_period"] == 1
    assert row["decisions"] == 9
    assert bytes.fromhex(row["repeat_ca3_hex"]) == torch.zeros_like(memory).numpy().tobytes()
    assert augmented_key(0, memory, 1) != augmented_key(0, memory, 0)
    assert augmented_key(0, memory, 1) != augmented_key(1, memory, 1)


def test_hit_precedence_over_terminal_and_cap_including_t0():
    env, dg = system()
    memory = torch.zeros(1, 2, 8)
    row = audit_query(AlwaysGo(), env, dg, 0, memory, 1, 1, cap=1)
    assert row["outcome"] == "hit" and row["first_hit_latency"] == 1
    initial = audit_query(AlwaysGo(), env, dg, 1, memory, 1, 1, cap=0)
    assert initial["outcome"] == "hit" and initial["first_hit_latency"] == 0
    assert initial["initially_satisfied"]


def test_fixed_cap_is_unknown_even_after_physical_node_repeats():
    env, _ = system()
    memory = torch.zeros(1, 2, 8)
    memory[0, 0, 0] = 1
    row = audit_query(AlwaysWait(), env, np.zeros((2, 2), np.float32), 0, memory, 1, 1, cap=3)
    assert row["outcome"] == "unknown_cap" and row["decisions"] == 3
    assert row["cycle_preperiod"] is None and row["cycle_period"] is None
    assert summarize([row])["noninitial"]["unknown_cap"] == 1


def test_repeat_at_cap_is_still_a_proven_cycle():
    env, _ = system()
    row = audit_query(AlwaysWait(), env, np.zeros((2, 2), np.float32), 0,
                      torch.zeros(1, 2, 8), 1, 1, cap=1)
    assert row["outcome"] == "proven_cycle_nonarrival" and row["cycle_period"] == 1


def test_non_goal_physical_terminal_is_failure_without_cycle_claim():
    env, dg = system()
    row = audit_query(AlwaysWait(), env, dg, 1, torch.zeros(1, 2, 8), 0, 0, cap=0)
    assert row["outcome"] == "physical_terminal_nonarrival" and row["terminal_time"] == 0
    assert row["cycle_period"] is None


def test_state_space_contract_and_declared_denominators():
    env, dg = system()
    assert validate_finite_system(env, dg, regression_config())["dg_alphabet_including_padding"] == 3
    with pytest.raises(ValueError, match="L8/R1"):
        validate_finite_system(env, dg, replace(regression_config(), repeat=2))
    noisy = env.transition.copy()
    noisy[0, 0] = [.5, .5]
    stochastic = FiniteMDP(env.name, noisy, env.terminal, env.goal_states, env.start_states)
    with pytest.raises(ValueError, match="deterministic"):
        validate_finite_system(stochastic, dg, regression_config())
    # Two conditions times three seeds, without selecting checkpoints.
    assert 2 * 3 * (49 + 4) == 318
    for env, dg, _, _ in experiments():
        assert validate_finite_system(env, dg, regression_config())["conservative_augmented_state_bound_per_goal"] > 100000


def test_archive_is_prepared_without_restoring_or_running_models(tmp_path, monkeypatch):
    from hpc_runs.first_hit_control import stationary_cycle_audit as module
    def forbidden(*args, **kwargs):
        raise AssertionError("No actual policy construction/rollout is allowed during preparation")
    monkeypatch.setattr(module, "StationaryReadout", forbidden)
    monkeypatch.setattr(module, "audit_query", forbidden)
    result = prepare(tmp_path / "audit")
    assert result["protocol_sha256"] == PROTOCOL_SHA and len(result["cases"]) == 12
    entries = result["source_snapshot"]["sources"]
    assert sum(entry["source_path"].endswith("/readout.pt") for entry in entries) == 12
    assert sum(entry["source_path"].endswith("/trials.csv") for entry in entries) == 12
    assert not (tmp_path / "audit/audit_started.json").exists()
