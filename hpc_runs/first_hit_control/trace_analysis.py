"""Analyze certified command paths using canonical spatial binning.

This consumes completed aggregation receipts; it never loads an actor or
environment. Winner transitions, command outcomes, and stored graph estimates
remain separate objects. Occupancy is descriptive of own-command-stopped paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from hpc_runs.intrmotiv_study import spatial_contract
from hpc_runs.intrmotiv_study.spatial_contract import SpatialBounds, spatial_rate_maps

from .aggregate_evaluation import POLICIES, file_hash
from .detectors import detector_hits
from .provenance import audit_source_snapshots, snapshot_sources


def trace_arrays(row, goals):
    """Certify observation/action alignment, including an unknown terminal end."""
    trace = row.get("trajectory")
    if not row["issued"] or not row["commanded"] or not isinstance(trace, list) or not trace:
        raise ValueError("Require one issued own-command trajectory")
    elapsed, done = row["elapsed"], row["physical_done"]
    expected = elapsed + 1 - int(done)
    if len(trace) != expected or row["trajectory_end_observed"] != (not done):
        raise ValueError("Unknown terminal successor must be omitted, not autoreset")
    if [sample["elapsed"] for sample in trace] != list(range(expected)):
        raise ValueError("Trajectory observation times are misaligned")
    if [sample["remaining"] for sample in trace] != [row["horizon"] - t for t in range(expected)]:
        raise ValueError("Trajectory deadline annotation disagrees")
    actions = row["executed_actions"]
    if len(actions) != elapsed or [sample["action"] for sample in trace] != actions + ([] if done else [None]):
        raise ValueError("Trajectory next actions disagree with executed actions")
    position = np.asarray([sample["position"] for sample in trace], dtype=float)
    rotation = np.asarray([sample["rotation"] for sample in trace], dtype=float)
    dg = np.asarray([sample["dg"] for sample in trace], dtype=np.float32)
    if (position.shape != (expected, 3) or rotation.shape != (expected, 3)
            or dg.shape != (expected, goals) or not np.isfinite(position).all()
            or not np.isfinite(rotation).all() or not np.isfinite(dg).all() or (dg < 0).any()):
        raise ValueError("Invalid pose or processed DG dimensions/values")
    winner = detector_hits(torch.from_numpy(dg), "winner").numpy()
    exclusive = detector_hits(torch.from_numpy(dg), "exclusive").numpy()
    for mode, values in (("winner", winner), ("exclusive", exclusive)):
        if [sample[mode] for sample in trace] != [np.flatnonzero(v).tolist() for v in values]:
            raise ValueError("Saved detector events disagree with processed DG")
    if np.flatnonzero(winner[0]).tolist() != [row["source"]]:
        raise ValueError("Saved trace does not start at the declared source winner")
    times = np.flatnonzero(winner[:, row["command"]])
    hit_time = int(times[0]) if len(times) else None
    if hit_time != row["hit_time"] or bool(len(times)) != row["hit"]:
        raise ValueError("Requested first-hit outcome disagrees with trace")
    if hit_time is not None and hit_time != elapsed:
        raise ValueError("Command path did not stop on first winner arrival")
    return np.column_stack((position[:, :2], rotation[:, 1])), dg, winner


def footprint_events(pose, goal, field):
    """Secondary physical descriptor using all saved half-peak components.

    Historical field arrays are [x,y,unit], unlike canonical [unit,y,x]
    rate-map outputs. Unmapped and outside positions remain explicitly unknown.
    """
    bounds = field["bounds"]
    grain = field["occupancy"].shape[0]
    _, _, valid = spatial_rate_maps(pose, np.zeros((len(pose), 1)), bounds, grain)
    inside, known = np.zeros(len(pose), bool), np.zeros(len(pose), bool)
    indices = np.flatnonzero(valid)
    if len(indices):
        xy = pose[indices, :2]
        bins = np.floor((xy - [bounds.x_min, bounds.y_min]) /
                        [bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min] * grain).astype(int)
        bins = np.clip(bins, 0, grain - 1)
        x, y = bins.T
        known[indices] = field["occupancy"][x, y] > 0
        inside[indices] = known[indices] & (field["components"][x, y, goal] > 0)
    first = np.flatnonzero(inside)
    return dict(half_peak_footprint_first_observed_time=int(first[0]) if len(first) else None,
                half_peak_footprint_initial=bool(inside[0]) if known[0] else None,
                half_peak_footprint_last_observed=bool(inside[-1]) if known[-1] else None,
                half_peak_footprint_unknown_samples=int((~known).sum()),
                half_peak_footprint_out_of_bounds_samples=int((~valid).sum()),
                half_peak_footprint_unmapped_in_bounds_samples=int((valid & ~known).sum()))


def summarize_trace(row, goals, field=None, jump_distance=250.0):
    """Describe a path without joining resets or unmarked large relocations."""
    pose, dg, winner = trace_arrays(row, goals)
    distance = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
    # This is the existing telemetry segmentation threshold, not a control rule.
    if not np.isfinite(jump_distance) or jump_distance <= 0:
        raise ValueError("Invalid existing telemetry jump distance")
    continuous = distance <= jump_distance
    winners = np.where(winner.any(1), winner.argmax(1), goals)
    edges = np.zeros((goals + 1, goals + 1), dtype=np.int64)
    # A real teleport remains an observed factual detector transition. The
    # spatial path is segmented separately to avoid drawing an invented line.
    np.add.at(edges, (winners[:-1], winners[1:]), 1)
    first_times = [int(indices[0]) if len(indices) else None
                   for indices in (np.flatnonzero(winner[:, goal]) for goal in range(goals))]
    entries = np.r_[winner[:1], winner[1:] & ~winner[:-1]].sum(0)
    item = {key: row.get(key) for key in ("policy", "source", "repeat", "prefix_seed", "command",
            "issued", "missing", "missing_reason", "initially_at_command", "hit", "hit_time",
            "elapsed", "horizon", "physical_done", "censored", "trajectory_id")}
    item.update(observed_samples=len(pose), observed_step_pairs=len(distance),
                unknown_terminal_step=int(row["physical_done"]),
                relocation_step_pairs=int((~continuous).sum()),
                observed_adjacent_endpoint_distance=float(distance.sum()),
                observed_contiguous_path_length=float(distance[continuous].sum()),
                observed_stationary_step_pairs=int((distance[continuous] <= 1.0).sum()),
                observed_distinct_winners=int(winner.any(0).sum()),
                observed_winner_changes=int(((winners[1:] != winners[:-1]) & continuous).sum()),
                observed_silent_samples=int((~winner.any(1)).sum()),
                first_winner_times_json=json.dumps(first_times),
                detector_reentries_json=json.dumps(np.maximum(entries - 1, 0).tolist()),
                action_counts_json=json.dumps({str(a): row["executed_actions"].count(a)
                                               for a in sorted(set(row["executed_actions"]))}),
                last_observed_x=float(pose[-1, 0]), last_observed_y=float(pose[-1, 1]),
                endpoint_is_true_trial_end=not row["physical_done"])
    if field is not None:
        item.update(footprint_events(pose, row["command"], field))
    return item, pose, dg, winner, edges


def load_field(audit_path, field_path, goals):
    audit, field_path = json.loads(Path(audit_path).read_text()), Path(field_path)
    declared = audit["artifacts"]["common_panel_fields"]
    if file_hash(field_path) != declared["sha256"]:
        raise ValueError("Field bytes differ from the exact-checkpoint reuse audit")
    with np.load(field_path, allow_pickle=False) as arrays:
        occupancy = arrays["occupancy"].copy()
        labels = arrays["post_inhibition_field_component_labels"].copy()
    shape = audit["mapping_contract"]["shape"]
    if occupancy.shape != tuple(shape[:2]) or labels.shape != (3, *shape) or shape[-1] != goals:
        raise ValueError("Unexpected historical [threshold,x,y,unit] field layout")
    mapping = audit["mapping_contract"]
    if mapping["existing_component_masks"]["threshold_fractions"] != [.3, .5, .7]:
        raise ValueError("The saved half-peak threshold identity is not certified")
    bounds = mapping["world_bounds"]
    return dict(occupancy=occupancy, components=labels[1],
                bounds=SpatialBounds(*bounds["x"], *bounds["y"]))


def analyze(aggregate_path, output, *, field_audit=None, field_npz=None):
    """Keep every declared command row; stream heavy traces only once each."""
    aggregate_path, output = Path(aggregate_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous trajectory analysis")
    aggregate = json.loads(aggregate_path.read_text())
    if (aggregate.get("schema") != "intrmotiv/first-hit-evaluation-aggregate/v1"
            or aggregate.get("status") != "validated_completed_shards"):
        raise ValueError("Run canonical completed-shard aggregation first")
    identity = aggregate["identities"]
    goals = len(identity["goal_vocabulary"])
    if identity["goal_vocabulary"] != list(range(goals)):
        raise ValueError("Require complete indexed DG vocabulary")
    if bool(field_audit) != bool(field_npz):
        raise ValueError("Provide the field audit and its exact NPZ together")
    field = load_field(field_audit, field_npz, goals) if field_audit else None
    if field_audit:
        audit = json.loads(Path(field_audit).read_text())
        if (audit.get("all_recorded_checks_passed") is not True
                or audit["artifacts"]["checkpoint"]["sha256"] != identity["source_checkpoint_sha256"]
                or audit["artifacts"]["source_config"]["sha256"] != identity["source_config_sha256"]):
            raise ValueError("Field audit and evaluated donor identities disagree")
        bounds = field["bounds"]
    else:
        bounds = SpatialBounds()
    input_hashes = {str(aggregate_path): file_hash(aggregate_path)}
    for path in (field_audit, field_npz):
        if path:
            input_hashes[str(Path(path).resolve())] = file_hash(path)
    sources = [Path(__file__), Path(__file__).with_name("aggregate_evaluation.py"),
               Path(__file__).with_name("detectors.py"), Path(__file__).with_name("targets.py"),
               Path(__file__).with_name("provenance.py"), Path(spatial_contract.__file__)]
    output.mkdir(parents=True)
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    records, panels = [], {policy: dict(pose=[], dg=[], winner=[], edges=np.zeros((goals + 1, goals + 1), np.int64))
                            for policy in POLICIES}
    seen = set()
    for receipt in aggregate["input_receipts"]:
        trial_path = Path(receipt["directory"]) / "trials.jsonl"
        if file_hash(trial_path) != receipt["trials_sha256"]:
            raise ValueError("Trial bytes differ from completed aggregate")
        input_hashes[str(trial_path)] = receipt["trials_sha256"]
        with trial_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if not row["commanded"]:
                    continue
                key = tuple(row[k] for k in ("policy", "source", "repeat", "command"))
                if key in seen:
                    raise ValueError("Duplicate executed command")
                seen.add(key)
                if not row["issued"]:
                    records.append({k: row.get(k) for k in ("policy", "source", "repeat", "command",
                        "issued", "missing", "missing_reason", "initially_at_command", "hit", "hit_time")})
                    continue
                item, pose, dg, winner, edges = summarize_trace(row, goals, field)
                records.append(item)
                # Initial-satisfied queries have no executed path; keep their
                # rows but exclude their duplicated t0 from behavior maps.
                if not row["initially_at_command"]:
                    panel = panels[row["policy"]]
                    for name, values in (("pose", pose), ("dg", dg), ("winner", winner)):
                        panel[name].append(values)
                    panel["edges"] += edges
    expected = {(p, s, r, g) for p in POLICIES for s in identity["goal_vocabulary"]
                for r in range(identity["config"]["repeats"]) for g in identity["observed_targets"]}
    if seen != expected:
        raise ValueError("Incomplete declared own-command rows")
    arrays, policy_summary = {}, {}
    for policy, panel in panels.items():
        pose = np.concatenate(panel["pose"]) if panel["pose"] else np.zeros((0, 3))
        dg = np.concatenate(panel["dg"]) if panel["dg"] else np.zeros((0, goals))
        winner = np.concatenate(panel["winner"]) if panel["winner"] else np.zeros((0, goals))
        maps, occupancy, in_bounds = spatial_rate_maps(pose, dg, bounds, 19)
        winner_maps, _, _ = spatial_rate_maps(pose, winner.astype(float), bounds, 19)
        arrays.update({f"{policy}_mean_dg": maps, f"{policy}_winner_probability": winner_maps,
                       f"{policy}_occupancy": occupancy, f"{policy}_observed_winner_transitions": panel["edges"]})
        policy_summary[policy] = dict(observed_nontrivial_path_samples=len(pose),
            issued_nontrivial_commands=sum(r["policy"] == policy and r["issued"] and
                                           not r["initially_at_command"] for r in records),
            executed_nontrivial_decisions=sum(r["elapsed"] for r in records if r["policy"] == policy
                                             and r["issued"] and not r["initially_at_command"]),
            samples_outside_bounds=int((~in_bounds).sum()), visited_bins=int((occupancy > 0).sum()),
            total_bins=int(occupancy.size), observed_adjacent_winner_pairs=int(panel["edges"].sum()))
    columns = list(dict.fromkeys(key for row in records for key in row))
    with (output / "per_trial.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "spatial_graphs.npz", **arrays)
    if any(file_hash(path) != value for path, value in input_hashes.items()):
        raise RuntimeError("Analysis inputs changed during processing")
    source_audit = audit_source_snapshots(output / "source_snapshot")
    if not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"]:
        raise RuntimeError("Analysis source changed during processing")
    result = dict(schema="intrmotiv/first-hit-trace-analysis/v1", input_hashes=input_hashes,
        source_snapshot=snapshot, source_audit=source_audit, declared_command_rows=len(records),
        policies=policy_summary, bounds=vars(bounds), grain=19,
        goal_capacity_status={str(g): "observed_and_declared" if g in identity["observed_targets"]
                              else "unobserved_unassessed" for g in identity["goal_vocabulary"]},
        map_axes=["DG_identity", "y_bin", "x_bin"], occupancy_axes=["y_bin", "x_bin"],
        winner_transition_axes=["current_winner_or_silent", "next_winner_or_silent"], silent_index=goals,
        trajectory_segmentation=dict(maximum_contiguous_jump=250.0, stationary_distance=1.0,
            source="reusable_place_field_telemetry.md descriptive jump convention; does not filter detector edges"),
        interpretation=["Own-command-stopped occupancy is exposure weighted and not a matched-time exploration comparison.",
            "Maps count t0 and every known successor observation, including observed endpoints; they do not count decisions. Unvisited bins are zero-filled and must be masked by occupancy.",
            "Winner transition counts are observed adjacent events, not goal-conditioned control probabilities.",
            "Unknown terminal successors never create detector edges; observed large jumps remain factual detector edges but are excluded from spatial line lengths.",
            "Initial-satisfied rows and missing commands are retained; neither enters behavior maps.",
            "Half-peak continuous-map membership is a secondary physical descriptor, not the winner detector or unique destination.",
            "Censored final poses are last observed positions, never certified physical endpoints."],
        outputs={name: file_hash(output / name) for name in ("per_trial.csv", "spatial_graphs.npz")})
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-audit", type=Path)
    parser.add_argument("--field-npz", type=Path)
    args = parser.parse_args()
    result = analyze(args.aggregate, args.output, field_audit=args.field_audit, field_npz=args.field_npz)
    print(json.dumps({"output": str(args.output), "declared_commands": result["declared_command_rows"]}))


if __name__ == "__main__":
    main()
