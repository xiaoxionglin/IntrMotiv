"""Scientific contract tests: splits, action timing and matched counterfactuals."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from parallel_action_register_probe import ACTIONS, audit, build_dataset, decode_registers, endpoint, matched_pairs


def small_config():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "06_experiments/ca3_goals/parallel_action_register_probe_20261111/config.json").read_text())
    config.update(train_max_length=4, longer_lengths=[5], longer_words_per_length=30)
    return config


def test_counterfactual_endpoint_really_depends_on_order_and_landmark():
    assert endpoint("FL", 0).tolist() == [1, 0]
    assert endpoint("LF", 0).tolist() == [0, 1]
    assert endpoint("FL", 1).tolist() == [0, 1]
    assert endpoint("LF", 1).tolist() == [-1, 0]


def test_registers_are_sufficient_without_word_or_heading_metadata():
    config = small_config()
    data = build_dataset(config)
    expected = data["targets"].numpy().copy()
    data["rows"] = None
    assert np.array_equal(decode_registers(data, config), expected)


def test_splits_and_matching_cannot_leak_words_or_counts():
    config = small_config()
    data = build_dataset(config)
    checks = audit(data, config)
    assert checks["matched_pair_counts"]["unseen_words"] > 0
    assert checks["matched_pair_counts"]["longer_words"] > 0
    pairs = matched_pairs(data, "unseen_words")
    i, j = pairs.T
    assert torch.equal(data["dg_state"][i], data["dg_state"][j])
    assert torch.equal(data["features"]["unordered"][i], data["features"]["unordered"][j])


def test_history_contains_past_actions_in_causal_order_and_no_label_input():
    config = small_config()
    data = build_dataset(config)
    for row, history in zip(data["rows"], data["action_state"]):
        decoded = "".join(ACTIONS[int(slot.argmax())] if slot.any() else "W" for slot in history[-config["steps"]:])
        assert decoded == row["prefix"] + row["suffix"]
        assert history[:-config["steps"]].eq(0).all()
    old_features = {name: value.clone() for name, value in data["features"].items()}
    data["targets"].fill_(999)
    for name in old_features:
        assert torch.equal(old_features[name], data["features"][name])


def test_current_dg_register_unchanged_when_action_order_changes():
    config = small_config()
    data = build_dataset(config)
    i, j = matched_pairs(data, "train")[0]
    original = copy.deepcopy(data["dg_state"][i])
    assert torch.equal(original, data["dg_state"][j])
    assert not torch.equal(data["action_state"][i], data["action_state"][j])


def test_age_matched_control_covers_every_suffix_position_without_target_leakage():
    config = small_config()
    config["anchor_age"] = 6
    data = build_dataset(config)
    assert audit(data, config)["oracle_exact"]
    for row in data["rows"]:
        assert len(row["suffix"]) == 6
        assert row["suffix"].replace("W", "") == row["word"]
    train_history = data["action_state"][data["indices"]["train"], -6:]
    assert train_history.sum(dim=0).gt(0).all()
