"""Replay exposure tests using arrays only; no native environment imports."""

import copy
import json

import numpy as np
import pytest

from hpc_runs.first_hit_control.replay_fit import episode_split
from hpc_runs.first_hit_control.support_analysis import analyze_replay, file_hash, run_analysis


def fixture():
    layout = {"width": 9, "goal_start": 7, "goals": 2}
    core = np.zeros((6, 9), np.float32)
    # CA3 width6 / two goals => expanded_length3, then one physical bypass.
    core[:, [0, 3]] = [[2, 1], [0, 0], [0, 3], [2, 2], [0, 0], [1, 0]]
    core[:, [1, 2, 4, 5, 6, 7, 8]] = 50  # History, bypass and goal must not label current DG.
    data = dict(core=core, action=np.array([0, 1, 1, 0, 0, 1], np.int64),
                successor_core=np.zeros_like(core),
                successor_dg=np.array([[2, 1], [0, 0], [0, 3], [2, 2], [0, 0], [0, 0]], np.float32),
                physical_done=np.array([False, True, False, True, False, True]),
                successor_valid=np.array([True, False, True, True, True, True]),
                successor_core_valid=np.array([True, False, True, False, True, False]),
                episode_id=np.repeat(np.arange(3), 2))
    metadata = dict(schema="intrmotiv/frozen-first-hit-replay/v1", model_unchanged=True,
                    model_sha256_before="frozen_fixture", model_sha256_after="frozen_fixture",
                    native_reward_zero_every_step=True, reset_observations_used_as_successors=0,
                    layout=layout, action_count=2, primary_detector="winner",
                    eligibility=dict(schema="intrmotiv/slice-only-relabel-eligibility/v1",
                                     structural_eligibility=True, input_kind="canonical_core_output", layout=layout,
                                     half_open_slices=dict(canonical_ca3=[0, 6], physical_bypass=[6, 7], goal=[7, 9])))
    return data, metadata


def test_actual_slot_zero_winners_ties_and_unknown_are_distinct():
    data, metadata = fixture()
    before = {key: value.copy() for key, value in data.items()}
    for array in data.values():
        array.setflags(write=False)
    report = analyze_replay(data, metadata, split_seed=99, holdout_fraction=.34)
    all_rows = report["panels"]["all"]
    assert report["expanded_length"] == 3
    assert all_rows["successor_positive_counts"] == {"winner": [2, 1], "exclusive": [0, 1]}
    assert all_rows["successor_positive_episode_counts"] == {"winner": [2, 1], "exclusive": [0, 1]}
    assert all_rows["current_dg_winner_action_counts"] == [[2, 1], [0, 1]]
    assert all_rows["current_dg_silent_action_counts"] == [1, 1]
    assert all_rows["current_dg_unknown_action_counts"] == [0, 0]
    assert all_rows["unknown_terminal_successor_rows"] == 1
    assert all_rows["certified_silent_successor_rows"] == 2
    assert all_rows["certified_multi_active_successor_rows"] == 2
    assert all_rows["current_dg_multi_active_rows"] == 2
    assert all(np.array_equal(data[key], values) for key, values in before.items())


def test_split_matches_fitter_and_preserves_complete_denominators():
    data, metadata = fixture()
    report = analyze_replay(data, metadata, split_seed=8, holdout_fraction=.34)
    train, holdout = episode_split(data["episode_id"], .34, 8)
    assert report["panels"]["train"]["rows"] == len(train)
    assert report["panels"]["holdout"]["episode_ids"] == np.unique(data["episode_id"][holdout]).tolist()
    assert set(report["panels"]["train"]["episode_ids"]).isdisjoint(report["panels"]["holdout"]["episode_ids"])
    for name in ("all", "train", "holdout"):
        panel = report["panels"][name]
        counted = np.asarray(panel["current_dg_winner_action_counts"]).sum(0)
        counted += panel["current_dg_silent_action_counts"]
        counted += panel["current_dg_unknown_action_counts"]
        assert counted.tolist() == panel["action_counts"]
        assert panel["certified_successor_rows"] + panel["unknown_successor_rows"] == panel["rows"]


def test_single_episode_has_explicit_empty_holdout_and_no_observed_goals():
    data, metadata = fixture()
    data["episode_id"][:] = 0
    data["successor_dg"][:] = 0
    report = analyze_replay(data, metadata)
    assert report["split"]["holdout_available"] is False
    assert report["panels"]["holdout"]["rows"] == 0
    assert report["panels"]["all"]["goals_without_winner_positive"] == [0, 1]
    assert report["panels"]["all"]["unknown_successor_rows"] == 1


def test_bad_certificate_and_live_successor_gap_are_rejected():
    data, metadata = fixture()
    bad = copy.deepcopy(metadata)
    bad["eligibility"]["half_open_slices"]["canonical_ca3"] = [0, 5]
    with pytest.raises(ValueError, match="scalar CA3"):
        analyze_replay(data, bad)
    bad = copy.deepcopy(metadata)
    bad["native_reward_zero_every_step"] = False
    with pytest.raises(ValueError, match="per-step native"):
        analyze_replay(data, bad)
    data["physical_done"][1] = False
    with pytest.raises(ValueError, match="successor validity"):
        analyze_replay(data, metadata)


def test_file_entry_verifies_hash_and_refuses_overwrite(tmp_path):
    data, metadata = fixture()
    replay = tmp_path / "replay"
    replay.mkdir()
    dataset = replay / "replay.npz"
    np.savez_compressed(dataset, **data)
    metadata["dataset_sha256"] = file_hash(dataset)
    (replay / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "report"
    result = run_analysis(replay, output)
    assert json.loads((output / "exposure.json").read_text())["panels"] == result["panels"]
    with pytest.raises(FileExistsError):
        run_analysis(replay, output)
    with dataset.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="collector certificate"):
        run_analysis(replay, tmp_path / "tampered_report")
