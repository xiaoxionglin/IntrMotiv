"""Synthetic mapping checks; these fixtures are not experimental results."""

import copy
import csv
import json

import numpy as np
import pytest

from hpc_runs.first_hit_control.aggregate_evaluation import POLICIES, file_hash
from hpc_runs.first_hit_control.plot_evaluation import (
    goal_mapping, matrix_mapping, select_trajectories, stream_selected_trials,
    trajectory_mapping,
)


def aggregate_fixture(n=16):
    result = dict(identities=dict(goal_vocabulary=list(range(n)), observed_targets=[0, 1]),
                  outcomes={}, competing_target_FIRST={})
    for policy in POLICIES:
        result["outcomes"][policy] = {"per_goal": {}}
        result["competing_target_FIRST"][policy] = {"per_goal": {}}
        for g in range(n):
            result["outcomes"][policy]["per_goal"][str(g)] = dict(
                success_lower_bound_declared=0, success_upper_bound_declared=1,
                declared_nontrivial_own_rows=3*(n-1), missing_nontrivial_own_rows=3*(n-1),
                censored_nontrivial_own_rows=0)
            result["competing_target_FIRST"][policy]["per_goal"][str(g)] = dict(
                mean_lower_bound=-1, mean_upper_bound=1, declared_comparisons=100,
                identified_comparisons=0)
    return result


def matrix_fixture(n=3):
    rows = []
    for s in range(n):
        for g in range(n):
            row = dict(source=s, goal=g, initially_satisfied=s == g, declared_histories=3)
            for p in POLICIES:
                row.update({f"{p}_success_lower": None if s == g else 0,
                            f"{p}_success_upper": None if s == g else 1,
                            f"{p}_missing": 2, f"{p}_censored": 1})
            rows.append(row)
    return rows


def test_all_capacity_goals_and_unknown_intervals_remain_visible():
    result = goal_mapping(aggregate_fixture())
    assert len(result) == 32 and {r["goal"] for r in result} == set(range(16))
    unknown = next(r for r in result if r["goal"] == 15)
    assert not unknown["observed"]
    assert unknown["arrival"] == [0., 1.] and unknown["FIRST"] == [-1., 1.]


def test_source_row_target_column_and_excluded_diagonal():
    rows = matrix_fixture()
    selected = next(r for r in rows if r["source"] == 1 and r["goal"] == 2)
    selected.update(source_success_lower=1/3, source_success_upper=2/3,
                    source_missing=0, source_censored=1)
    result = matrix_mapping(rows, [0, 1, 2])
    assert result["source"]["lower"][1, 2] == 1/3
    assert result["source"]["lower"][2, 1] == 0
    assert result["source"]["upper"][2, 1] == 1
    assert np.isnan(result["source"]["lower"].diagonal()).all()
    assert result["source"]["missing"][2, 1] == 2


def test_bounds_must_reconcile_and_full_matrix_cannot_drop_cells():
    rows = matrix_fixture()
    with pytest.raises(ValueError, match="Full capacity"):
        matrix_mapping(rows[:-1], [0, 1, 2])
    rows[1]["source_success_upper"] = .5
    with pytest.raises(ValueError, match="reconcile"):
        matrix_mapping(rows, [0, 1, 2])


def test_example_selection_ignores_success_and_keeps_unassessed_commands():
    rows = [dict(source=2, repeat=0, issued=True, command=3, hit=False),
            dict(source=1, repeat=0, issued=False, command=3, hit=False),
            dict(source=0, repeat=1, issued=True, command=3, hit=True)]
    selection = select_trajectories(rows, list(range(16)))
    assert selection["source"] == 2 and selection["repeat"] == 0
    assert selection["commands"] == [0, 6, 10, 15]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["hit"] = not row["hit"]
    assert select_trajectories(changed, list(range(16))) == selection
    missing = trajectory_mapping(None, policy="source", command=15, bounds={}, jump_distance=250)
    assert not missing["issued"] and "Unassessed" in missing["reason"]


def test_no_issued_repeat_zero_source_has_no_outcome_fallback():
    selection = select_trajectories([dict(source=0, repeat=1, issued=True)], list(range(16)))
    assert selection["source"] is None and selection["commands"] == []


def test_stream_reuses_only_actual_selected_own_paths_and_checks_hash(tmp_path):
    path = tmp_path / "trials.jsonl"
    rows = [dict(policy="source", source=2, repeat=0, command=6, commanded=True, issued=False),
            dict(policy="source", source=2, repeat=0, command=6, commanded=False, issued=False),
            dict(policy="source", source=3, repeat=0, command=6, commanded=True, issued=False)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    receipt = dict(directory=str(tmp_path), trials_sha256=file_hash(path))
    selected, hashes = stream_selected_trials([receipt], dict(source=2, repeat=0, commands=[6]))
    assert set(selected) == {("source", 2, 0, 6)} and hashes[str(path)] == file_hash(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Trial bytes"):
        stream_selected_trials([receipt], dict(source=2, repeat=0, commands=[6]))


def test_unknown_terminal_endpoint_and_large_jump_never_draw_fake_line():
    row = dict(issued=True, elapsed=3, physical_done=True, trajectory_end_observed=False,
               trajectory_id="synthetic", hit=False, hit_time=None,
               trajectory=[dict(position=[100, 100, 0]), dict(position=[110, 100, 0]),
                           dict(position=[1000, 1000, 0])])
    item = trajectory_mapping(row, policy="source", command=2,
        bounds=dict(x_min=100, x_max=2000, y_min=100, y_max=2000), jump_distance=250)
    assert len(item["segments"]) == 2 and item["segments"][0] == [[100., 100.], [110., 100.]]
    assert not item["endpoint_observed"] and item["omitted_unknown_terminal_steps"] == 1
    assert item["last_observed"] == [1000., 1000.]
    bad = copy.deepcopy(row)
    bad["trajectory"].append(dict(position=[100, 100, 0]))
    with pytest.raises(ValueError, match="endpoint"):
        trajectory_mapping(bad, policy="source", command=2, bounds={}, jump_distance=250)


def test_completed_artifact_hash_binding_without_rendering_result_figures(tmp_path, monkeypatch):
    from hpc_runs.first_hit_control import plot_evaluation as module
    aggregate = aggregate_fixture(3)
    aggregate.update(schema="intrmotiv/first-hit-evaluation-aggregate/v1",
                     status="validated_completed_shards", input_receipts=[])
    table = tmp_path / "source_target.csv"
    matrix = matrix_fixture()
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader(); writer.writerows(matrix)
    aggregate["outputs"] = {"source_target.csv": file_hash(table)}
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate))
    trials = tmp_path / "per_trial.csv"
    trials.write_text("source,repeat,issued\n0,0,False\n")
    traces = dict(schema="intrmotiv/first-hit-trace-analysis/v1",
        input_hashes={str(aggregate_path): file_hash(aggregate_path)},
        outputs={"per_trial.csv": file_hash(trials)},
        bounds=dict(x_min=100, x_max=2000, y_min=100, y_max=2000),
        trajectory_segmentation=dict(maximum_contiguous_jump=250))
    trace_path = tmp_path / "summary.json"
    trace_path.write_text(json.dumps(traces))
    fake_font = tmp_path / "font_fixture.ttf"
    fake_font.write_bytes(b"synthetic hash-only fixture; renderer is mocked")
    monkeypatch.setattr(module, "_runtime", lambda: (None, fake_font))
    for name in ("plot_goal_intervals", "plot_matrices", "plot_trajectory_pair"):
        monkeypatch.setattr(module, name, lambda *args: [])
    result = module.render(aggregate_path, trace_path, tmp_path / "mapping_only")
    assert not result["outputs"] and len(result["per_goal"]) == 6
    assert result["trajectory_selection"]["source"] is None
    assert result["matrices"]["source"]["lower"][0][0] is None
    table.write_text("corrupted\n")
    with pytest.raises(ValueError, match="Changed completed analysis"):
        module.render(aggregate_path, trace_path, tmp_path / "rejected")
