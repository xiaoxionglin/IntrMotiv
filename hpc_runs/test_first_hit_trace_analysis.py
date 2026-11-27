"""Scientific contract checks for post-hoc path and footprint analysis."""

import copy
import json

import numpy as np
import pytest

from hpc_runs.first_hit_control.trace_analysis import analyze, footprint_events, summarize_trace, trace_arrays
from hpc_runs.first_hit_control.aggregate_evaluation import file_hash
from hpc_runs.intrmotiv_study.spatial_contract import SpatialBounds


def row_for(winners, *, done=False, command=2, positions=None):
    elapsed = len(winners) - 1 + int(done)
    trace = []
    for t, winner in enumerate(winners):
        dg = np.zeros(3)
        if winner is not None:
            dg[winner] = 1
        trace.append(dict(elapsed=t, remaining=8-t, action=1 if t < elapsed else None,
                          position=[100 + t, 100, 0] if positions is None else positions[t],
                          rotation=[0, 90, 0], dg=dg.tolist(),
                          winner=[] if winner is None else [winner],
                          exclusive=[] if winner is None else [winner]))
    hit_times = [t for t, winner in enumerate(winners) if winner == command]
    return dict(policy="first_hit", source=0, repeat=0, command=command, issued=True, commanded=True,
                missing=False, missing_reason=None, initially_at_command=command == 0,
                horizon=8, elapsed=elapsed, physical_done=done, censored=done and not hit_times,
                hit=bool(hit_times), hit_time=hit_times[0] if hit_times else None,
                trajectory=trace, trajectory_id="test", trajectory_end_observed=not done,
                executed_actions=[1] * elapsed)


def test_terminal_unknown_has_no_reset_edge_or_endpoint():
    row = row_for([0, 1], done=True)
    summary, pose, _, _, edges = summarize_trace(row, 3)
    assert pose.shape == (2, 3)
    assert summary["unknown_terminal_step"] == 1
    assert not summary["endpoint_is_true_trial_end"]
    assert edges.sum() == 1 and edges[0, 1] == 1
    bad = copy.deepcopy(row)
    bad["trajectory"].append(copy.deepcopy(bad["trajectory"][0]))
    with pytest.raises(ValueError, match="terminal successor"):
        trace_arrays(bad, 3)


def test_silent_steps_remain_in_factual_graph():
    row = row_for([0, None, 1, None, 1, 2])
    summary, _, _, _, edges = summarize_trace(row, 3)
    assert edges[0, 3] == 1 and edges[3, 1] == 2 and edges[1, 3] == 1 and edges[1, 2] == 1
    assert edges[0, 1] == 0
    assert summary["first_winner_times_json"] == "[0, 2, 5]"
    assert summary["detector_reentries_json"] == "[0, 1, 0]"


def test_large_jump_segments_geometry_but_retains_factual_observed_edge():
    row = row_for([0, 1, 2], positions=[[100, 100, 0], [1000, 100, 0], [1001, 100, 0]])
    summary, _, _, _, edges = summarize_trace(row, 3)
    assert summary["observed_contiguous_path_length"] == 1
    assert summary["relocation_step_pairs"] == 1
    assert edges.sum() == 2


def test_detector_tie_uses_first_argmax_and_exclusive_separate():
    row = row_for([0, 2])
    row["trajectory"][0]["dg"] = [1, 1, 0]
    row["trajectory"][0]["exclusive"] = []
    trace_arrays(row, 3)
    row["trajectory"][0]["winner"] = [1]
    with pytest.raises(ValueError, match="detector events"):
        trace_arrays(row, 3)


def test_action_and_first_hit_alignment_rejected():
    row = row_for([0, 2])
    row["trajectory"][-1]["action"] = 1
    with pytest.raises(ValueError, match="next actions"):
        trace_arrays(row, 3)
    row = row_for([0, 2, 2])
    with pytest.raises(ValueError, match="stop on first"):
        trace_arrays(row, 3)


def test_initial_satisfied_path_has_zero_steps():
    row = row_for([0], command=0)
    summary, _, _, _, edges = summarize_trace(row, 3)
    assert summary["elapsed"] == 0 and summary["hit_time"] == 0 and edges.sum() == 0


def test_historical_xy_masks_do_not_transpose_canonical_yx_bins():
    components = np.zeros((2, 2, 3), dtype=int)
    components[1, 0, 2] = 1
    occupancy = np.ones((2, 2))
    occupancy[0, 1] = 0
    field = dict(bounds=SpatialBounds(0, 2, 0, 2), occupancy=occupancy, components=components)
    pose = np.array([[1.5, .5, 0], [.5, 1.5, 0], [3, 0, 0]])
    result = footprint_events(pose, 2, field)
    assert result["half_peak_footprint_initial"]
    assert result["half_peak_footprint_first_observed_time"] == 0
    assert result["half_peak_footprint_last_observed"] is None
    assert result["half_peak_footprint_unmapped_in_bounds_samples"] == 1
    assert result["half_peak_footprint_out_of_bounds_samples"] == 1


def test_closed_upper_bound_is_last_bin():
    field = dict(bounds=SpatialBounds(0, 2, 0, 2), occupancy=np.ones((2, 2)),
                 components=np.ones((2, 2, 3), int))
    result = footprint_events(np.array([[2, 2, 0]]), 1, field)
    assert result["half_peak_footprint_last_observed"]


@pytest.mark.parametrize("reused", [False, True])
def test_complete_denominators_and_no_duplicate_initial_occupancy(tmp_path, reused):
    rows = []
    for policy in ("source", "first_hit"):
        for source in range(3):
            for command in range(3):
                if source:
                    row = dict(policy=policy, source=source, repeat=0, command=command,
                               issued=False, commanded=True, missing=True, missing_reason="no_history",
                               initially_at_command=source == command, hit=False, hit_time=None)
                else:
                    row = row_for([0] if command != 1 else [0, 1], command=command, done=command == 2)
                    row["policy"] = policy
                rows.append(row)
    shard = tmp_path / "shard"
    shard.mkdir()
    trial_file = shard / "trials.jsonl"
    trial_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = dict(directory=str(shard), trials_sha256=file_hash(trial_file))
    if reused:
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        candidate_file = candidate / "candidate_trials.jsonl"
        candidate_file.write_text("".join(json.dumps(r) + "\n" for r in rows if r["policy"] == "first_hit"))
        # Old candidate paths deliberately lack the trace contract. The new
        # analysis must import only the original source policy from this file.
        old_rows = [r if r["policy"] == "source" else dict(policy="first_hit", obsolete=True) for r in rows]
        trial_file.write_text("".join(json.dumps(r) + "\n" for r in old_rows))
        receipt["trials_sha256"] = file_hash(trial_file)
        receipt = dict(schema="intrmotiv/reused-control-evaluation/v1", directory=str(candidate),
                       trials_sha256=file_hash(candidate_file), candidate_trials_file=candidate_file.name,
                       imported_control=receipt)
    aggregate = dict(schema="intrmotiv/first-hit-evaluation-aggregate/v1", status="validated_completed_shards",
                     identities=dict(goal_vocabulary=[0, 1, 2], observed_targets=[0, 1, 2], config=dict(repeats=1)),
                     input_receipts=[receipt])
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate))
    result = analyze(aggregate_path, tmp_path / "analysis")
    assert result["declared_command_rows"] == 18
    for policy in ("source", "first_hit"):
        summary = result["policies"][policy]
        assert summary["issued_nontrivial_commands"] == 2
        assert summary["executed_nontrivial_decisions"] == 2
        assert summary["observed_nontrivial_path_samples"] == 3
        assert summary["observed_adjacent_winner_pairs"] == 1
    with np.load(tmp_path / "analysis" / "spatial_graphs.npz") as arrays:
        assert arrays["first_hit_occupancy"].sum() == 3
        assert arrays["first_hit_observed_winner_transitions"][0, 1] == 1
    with pytest.raises(FileExistsError):
        analyze(aggregate_path, tmp_path / "analysis")
    trial_file.write_text(trial_file.read_text() + "\n")
    with pytest.raises(ValueError, match="Trial bytes"):
        analyze(aggregate_path, tmp_path / "altered")
