"""Coarse exposure diagnostics for certified frozen replay, without an environment.

DG winner counts aggregate many physical and recurrent states. They are not a
Markov-state/action coverage certificate and do not establish controllability,
reachability from every start, or sufficiency of the frozen representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .detectors import detector_hits
from .replay_fit import episode_split, validate_replay
from .runtime_adapter import GoalLayout


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def certified_layout(metadata):
    """Require the collector's explicit full-core scalar CA3 layout."""
    if metadata.get("schema") != "intrmotiv/frozen-first-hit-replay/v1":
        raise ValueError("Unknown frozen replay schema")
    if (metadata.get("model_unchanged") is not True
            or not metadata.get("model_sha256_before")
            or metadata["model_sha256_before"] != metadata.get("model_sha256_after")):
        raise ValueError("Frozen model certificate is missing or inconsistent")
    if metadata.get("native_reward_zero_every_step") is not True:
        raise ValueError("A per-step native reward-free certificate is required")
    if metadata.get("reset_observations_used_as_successors") != 0:
        raise ValueError("Reset observations cannot enter factual replay")
    if metadata.get("primary_detector") not in ("winner", "exclusive"):
        raise ValueError("The source winner/exclusive detector contract must be explicit")
    certificate = metadata.get("eligibility", {})
    if (certificate.get("schema") != "intrmotiv/slice-only-relabel-eligibility/v1"
            or certificate.get("structural_eligibility") is not True
            or certificate.get("input_kind") != "canonical_core_output"
            or certificate.get("layout") != metadata.get("layout")):
        raise ValueError("Require the collector's canonical full-core layout certificate")
    values = metadata["layout"]
    if (not isinstance(values, dict) or set(values) != {"width", "goal_start", "goals"}
            or any(type(value) is not int for value in values.values())):
        raise ValueError("The full-core layout must contain exact integer dimensions")
    layout = GoalLayout(**values)
    slices = certificate.get("half_open_slices", {})
    ca3 = slices.get("canonical_ca3", [])
    if (len(ca3) != 2 or ca3[0] != 0 or type(ca3[1]) is not int
            or not 0 < ca3[1] <= layout.goal_start or ca3[1] % layout.goals
            or layout.width != layout.goal_start + layout.goals
            or slices.get("physical_bypass") != [ca3[1], layout.goal_start]
            or slices.get("goal") != [layout.goal_start, layout.width]):
        raise ValueError("Unverified scalar CA3/physical-bypass/goal slices")
    actions = metadata.get("action_count")
    if type(actions) is not int or actions < 1:
        raise ValueError("The certified source action count is required")
    return layout, ca3[1] // layout.goals, actions


def analyze_replay(data, metadata, *, split_seed=99, holdout_fraction=.1):
    """Summarize immutable arrays; split episodes exactly as the replay fitter.

    Current DG is the post-processing CA3 slot zero. All current core rows in
    certified replay are known; successor validity is independent, so an
    unknown terminal observation is never counted as a silent observation.
    """
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0,1)")
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")
    layout, expanded, actions = certified_layout(metadata)
    count = validate_replay(data, layout, actions)
    valid, core_valid, done = (data[key] for key in
                              ("successor_valid", "successor_core_valid", "physical_done"))
    if np.any(core_valid & ~valid) or np.any(~done & (~valid | ~core_valid)):
        raise ValueError("Replay has inconsistent factual successor validity")
    if np.any(data["episode_id"] < 0):
        raise ValueError("Episode identities must be nonnegative")
    current = data["core"][:, :layout.goals * expanded].reshape(count, layout.goals, expanded)[:, :, 0]
    if np.any(current < 0):
        raise ValueError("Current post-processing DG must be nonnegative")
    # Canonical detector helpers do not mutate their inputs. Copy only the small
    # DG matrix so read-only replay memory maps remain safe for torch views.
    current_tensor = torch.from_numpy(current.copy())
    successor_tensor = torch.from_numpy(data["successor_dg"].copy())
    current_hits = detector_hits(current_tensor, "winner").numpy()
    successor_hits = {mode: detector_hits(successor_tensor, mode).numpy() & valid[:, None]
                      for mode in ("winner", "exclusive")}
    current_active = (current > 0).sum(1)
    successor_active = (data["successor_dg"] > 0).sum(1)
    train, holdout = episode_split(data["episode_id"], holdout_fraction, split_seed)
    panels = {"all": np.arange(count), "train": train, "holdout": holdout}
    summary = {}
    for name, indices in panels.items():
        episode_ids = data["episode_id"][indices]
        panel_actions = data["action"][indices]
        positive_counts, positive_episodes = {}, {}
        for mode, hits in successor_hits.items():
            selected = hits[indices]
            positive_counts[mode] = selected.sum(0).tolist()
            positive_episodes[mode] = [int(len(np.unique(episode_ids[selected[:, goal]])))
                                       for goal in range(layout.goals)]
        by_winner = np.zeros((layout.goals, actions), np.int64)
        selected_current = current_hits[indices]
        row, goal = np.nonzero(selected_current)
        np.add.at(by_winner, (goal, panel_actions[row]), 1)
        silent = current_active[indices] == 0
        silent_actions = np.bincount(panel_actions[silent], minlength=actions)
        summary[name] = dict(
            rows=len(indices), episodes=int(len(np.unique(episode_ids))),
            episode_ids=np.unique(episode_ids).tolist(),
            action_counts=np.bincount(panel_actions, minlength=actions).tolist(),
            physical_terminal_rows=int(done[indices].sum()),
            unknown_terminal_successor_rows=int((done[indices] & ~valid[indices]).sum()),
            unknown_successor_rows=int((~valid[indices]).sum()),
            certified_successor_rows=int(valid[indices].sum()),
            certified_silent_successor_rows=int((valid[indices] & (successor_active[indices] == 0)).sum()),
            certified_multi_active_successor_rows=int((valid[indices] & (successor_active[indices] > 1)).sum()),
            successor_positive_counts=positive_counts,
            successor_positive_episode_counts=positive_episodes,
            goals_without_winner_positive=[goal for goal, number in enumerate(positive_counts["winner"]) if number == 0],
            current_dg_winner_action_counts=by_winner.tolist(),
            current_dg_silent_action_counts=silent_actions.tolist(),
            current_dg_unknown_action_counts=[0] * actions,
            current_dg_silent_rows=int(silent.sum()), current_dg_unknown_rows=0,
            current_dg_multi_active_rows=int((current_active[indices] > 1).sum()),
        )
    return dict(
        schema="intrmotiv/frozen-replay-exposure/v1",
        interpretation="Coarse DG/action exposure diagnostic; not a Markov-state/action coverage certificate",
        limitations=["DG identities pool distinct physical and recurrent states",
                     "Observed visits do not prove control or reachability from every start",
                     "Missing successors are unknown, not certified failures or silent DG",
                     "No result here establishes frozen-memory sufficiency"],
        split=dict(seed=split_seed, holdout_fraction=holdout_fraction,
                   method="Canonical replay_fit.episode_split, whole episodes",
                   holdout_available=bool(len(holdout))),
        layout=metadata["layout"], expanded_length=expanded,
        declared_goal_ids=list(range(layout.goals)), action_ids=list(range(actions)),
        current_detector="Actual scalar CA3[:, goal, 0]; positive argmax, first-index tie break",
        action_matrix_axes=["current_dg_winner", "executed_action"],
        successor_detectors=["winner", "exclusive"],
        primary_detector=metadata.get("primary_detector"),
        panels=summary,
    )


def run_analysis(replay_directory, output, *, split_seed=99, holdout_fraction=.1):
    """Verify immutable collection inputs, then write a new diagnostic artifact."""
    replay_directory, output = Path(replay_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing exposure reports; choose a new directory")
    metadata_path, dataset = replay_directory / "metadata.json", replay_directory / "replay.npz"
    input_hashes = {str(path): file_hash(path) for path in (metadata_path, dataset)}
    metadata = json.loads(metadata_path.read_text())
    if input_hashes[str(dataset)] != metadata.get("dataset_sha256"):
        raise ValueError("Replay bytes differ from the collector certificate")
    with np.load(dataset, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    result = analyze_replay(data, metadata, split_seed=split_seed, holdout_fraction=holdout_fraction)
    if any(file_hash(path) != expected for path, expected in input_hashes.items()):
        raise RuntimeError("Certified replay inputs changed during analysis")
    package = Path(__file__).parent
    result["provenance"] = dict(
        input_files=input_hashes, source_checkpoint_sha256=metadata.get("checkpoint_sha256"),
        source_config_sha256=metadata.get("source_config_sha256"),
        analysis_source_sha256={name: file_hash(package / name) for name in
                                ("support_analysis.py", "replay_fit.py", "detectors.py", "targets.py", "runtime_adapter.py")},
    )
    output.mkdir(parents=True)
    (output / "exposure.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "rows": result["panels"]["all"]["rows"],
                      "train_goals_without_positive": result["panels"]["train"]["goals_without_winner_positive"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=99)
    parser.add_argument("--holdout-fraction", type=float, default=.1)
    args = parser.parse_args()
    run_analysis(args.replay, args.output, split_seed=args.split_seed, holdout_fraction=args.holdout_fraction)


if __name__ == "__main__":
    main()
