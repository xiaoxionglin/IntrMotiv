"""Post-hoc exact-cycle diagnostic of fixed deterministic toy controllers.

No optimizer, exploration, or native environment is run. A physical node alone
is not the controller state: the complete CA3 register and fixed command are
included in the exact recurrence key. The original six-step results stay fixed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .neural import digest, evaluation_starts, observe, state_digest, write_csv
from .projected_grid import archived_path
from .projected_regression import ROOT, STUDY, experiments, regression_config, validate_reference
from .provenance import audit_source_snapshots, snapshot_sources
from .stationary_readout import StationaryReadout, original_classes, shortest_distances


PROTOCOL = STUDY / "runtime/stationary_cycle_audit_protocol.json"
PROTOCOL_SHA = "c7d4664b2864276bd5496154efabf30cd695ad8b0f08d458024a9be3096ce301"


def augmented_key(state, memory, goal):
    """Exact equality, with no tolerances, rounding, or node-only collapse."""
    array = memory.detach().cpu().contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite CA3 cannot certify an exact deterministic cycle")
    return int(state), int(goal), array.dtype.str, tuple(array.shape), array.tobytes()


def validate_finite_system(env, dg, config):
    """The R1 adapter is a finite-alphabet shift register, without overlap."""
    if config.repeat != 1 or config.memory_length != 8:
        raise ValueError("Only the declared L8/R1 memory is certified")
    if not np.isin(env.transition, [0., 1.]).all() or not np.all(env.transition.sum(-1) == 1):
        raise ValueError("Exact cycle proof requires deterministic physical transitions")
    if dg.ndim != 2 or dg.shape[0] != env.n_states or not np.isin(dg, [0., 1.]).all() or np.any(dg.sum(-1) > 1):
        raise ValueError("Expected finite zero/one-hot synthetic DG observations")
    alphabet = len({row.tobytes() for row in np.concatenate((dg, np.zeros_like(dg[:1])))})
    return dict(dg_alphabet_including_padding=alphabet,
                conservative_augmented_state_bound_per_goal=env.n_states * alphabet ** config.memory_length)


@torch.no_grad()
def audit_query(model, env, dg, initial, initial_memory, goal_state, goal_id, *, cap, repeat=1):
    """Stop on a hit, physical failure, proven exact cycle, or unknown cap.

    Terminal and hit checks precede recurrence and cap checks, including after
    the final allowed action. No deadline is passed to the stationary policy.
    """
    if not isinstance(cap, int) or cap < 0:
        raise ValueError("The audit cap must be a nonnegative integer")
    state, memory = int(initial), initial_memory.clone()
    seen, states, actions = {}, [state], []
    initial_key = augmented_key(state, memory, goal_id)
    result = dict(start=state, goal=int(goal_state), goal_id=int(goal_id),
                  initially_satisfied=state == goal_state, first_hit_latency=None,
                  terminal_time=None, cycle_preperiod=None, cycle_period=None,
                  repeat_state=None, repeat_ca3_dtype=None, repeat_ca3_shape=None,
                  repeat_ca3_hex=None, repeat_ca3_sha256=None,
                  initial_ca3_sha256=hashlib.sha256(initial_key[-1]).hexdigest())
    while True:
        elapsed = len(actions)
        if state == goal_state:
            result.update(outcome="hit", first_hit_latency=elapsed)
            break
        if env.terminal[state]:
            result.update(outcome="physical_terminal_nonarrival", terminal_time=elapsed)
            break
        key = augmented_key(state, memory, goal_id)
        if key in seen:
            first = seen[key]
            result.update(outcome="proven_cycle_nonarrival", cycle_preperiod=first,
                          cycle_period=elapsed-first, repeat_state=state,
                          repeat_ca3_dtype=key[2], repeat_ca3_shape=list(key[3]),
                          repeat_ca3_hex=key[-1].hex(),
                          repeat_ca3_sha256=hashlib.sha256(key[-1]).hexdigest())
            break
        if elapsed >= cap:
            result.update(outcome="unknown_cap")
            break
        seen[key] = elapsed
        values = model(memory, torch.tensor([goal_id]))
        if values.shape != (1, env.n_actions) or not torch.isfinite(values).all():
            raise ValueError("Expected finite raw action values for exactly one query")
        action = int(values.argmax(-1).item())
        probability = env.transition[state, action]
        if not np.isin(probability, [0., 1.]).all() or probability.sum() != 1:
            raise ValueError("Cannot certify recurrence of a stochastic trajectory")
        state = int(probability.argmax())
        memory = observe(memory, dg[state], repeat)
        states.append(state)
        actions.append(action)
    result.update(decisions=len(actions), states=json.dumps(states), actions=json.dumps(actions),
                  distinct_augmented_states=len(seen))
    return result


def summarize(rows):
    """Preserve initial, noninitial, resolved and unknown denominators."""
    def count(panel):
        result = {name: sum(r["outcome"] == name for r in panel) for name in
                  ("hit", "physical_terminal_nonarrival", "proven_cycle_nonarrival", "unknown_cap")}
        result["queries"] = len(panel)
        latencies = [r["first_hit_latency"] for r in panel if r["outcome"] == "hit"]
        result["maximum_observed_hit_latency"] = max(latencies) if latencies else None
        return result
    noninitial = [row for row in rows if not row["initially_satisfied"]]
    return dict(all=count(rows), noninitial=count(noninitial),
                initially_satisfied=count([row for row in rows if row["initially_satisfied"]]),
                per_goal={str(goal): count([row for row in noninitial if row["goal"] == goal])
                          for goal in sorted({row["goal"] for row in rows})})


def validate_protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Fixed cycle protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol["inputs"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Declared source result/checkpoint changed: {relative}")
    _, old, sources = validate_reference()
    return protocol, old, sources


def prepare(output):
    """Archive all checkpoints/sources/old traces before any audit rollout."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous cycle audit outputs")
    protocol, old, sources = validate_protocol()
    paths = [Path(__file__), Path(__file__).parents[1] / "test_first_hit_stationary_cycle_audit.py", PROTOCOL]
    paths += [Path(__file__).with_name(name) for name in
              ("projected_regression.py", "projected_grid.py", "provenance.py", "aggregate_evaluation.py", "dmlab_evaluate.py")]
    paths += [Path(path) for path in sources]
    paths += [ROOT / path for path in protocol["inputs"]]
    cases = []
    for condition in protocol["conditions"]:
        original = json.loads((STUDY / condition / "results.json").read_text())["runs"]
        for record in original:
            folder = STUDY / condition / record["environment"] / f"seed_{record['seed']}"
            traces, checkpoint = folder / "trials.csv", folder / "readout.pt"
            paths.append(traces)
            cases.append(dict(condition=condition, environment=record["environment"], seed=record["seed"],
                              checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                              traces=str(traces), traces_sha256=digest(traces),
                              original_h6_success=record["first_hit_success"],
                              original_h6_gate=record["all_pair_gate_passed"]))
    if len(cases) != 12 or len({(c["condition"], c["environment"], c["seed"]) for c in cases}) != 12:
        raise ValueError("Expected exactly the twelve declared checkpoints")
    output.mkdir(parents=True)
    snapshot = snapshot_sources(list(dict.fromkeys(paths)), output / "source_snapshot")
    prepared = dict(schema="intrmotiv/fixed-weight-cycle-prepared/v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    protocol_sha256=PROTOCOL_SHA, cases=cases, source_snapshot=snapshot,
                    config=asdict(regression_config()), decoder_settings=old["decoder_settings"],
                    stage="before model restoration or actual checkpoint rollouts")
    (output / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def run_prepared(output):
    output = Path(output).resolve()
    prepared = json.loads((output / "prepared.json").read_text())
    protocol, old, _ = validate_protocol()
    audit = audit_source_snapshots(output / "source_snapshot")
    if not audit["archive_valid"] or not audit["current_sources_unchanged"]:
        raise ValueError("Cycle audit sources or input bytes changed")
    config = regression_config()
    if prepared["config"] != json.loads(json.dumps(asdict(config))) or prepared["decoder_settings"] != old["decoder_settings"]:
        raise ValueError("Stationary reconstruction settings changed")
    with (output / "audit_started.json").open("x") as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(), prepared_sha256=digest(output / "prepared.json")), stream)
    decoder, action, _ = original_classes()
    torch.set_num_threads(1)
    environments = {env.name: (env, dg, goals) for env, dg, goals, _ in experiments()}
    all_rows, summaries = [], []
    for case in prepared["cases"]:
        env, dg, goals = environments[case["environment"]]
        finite = validate_finite_system(env, dg, config)
        model = StationaryReadout(dg.shape[1], config, decoder, action, old["decoder_settings"], action_count=env.n_actions)
        checkpoint = archived_path(output, prepared, case["checkpoint"])
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
        model.eval().requires_grad_(False)
        before = state_digest(model)
        distances = shortest_distances(env)
        with archived_path(output, prepared, case["traces"]).open() as stream:
            prior = {(int(r["start"]), int(r["goal"])): r for r in csv.DictReader(stream) if r["route"] == "first_hit"}
        rows = []
        for initial, memory, prefix in evaluation_starts(env, dg, config):
            for goal_index, goal in enumerate(env.goal_states):
                row = audit_query(model, env, dg, initial, memory, goal, goals[goal_index],
                                  cap=protocol["evaluation"]["cap_decisions"], repeat=config.repeat)
                row.update(condition=case["condition"], environment=env.name, seed=case["seed"],
                           prefix=json.dumps(prefix), shortest_distance=int(distances[goal_index, initial]),
                           original_h6_success=None, original_h6_prefix_identical=True)
                if initial != goal:
                    original = prior.pop((initial, int(goal)))
                    old_actions, old_states = json.loads(original["actions"]), json.loads(original["states"])
                    new_actions, new_states = json.loads(row["actions"]), json.loads(row["states"])
                    shared = min(len(old_actions), len(new_actions))
                    if old_actions[:shared] != new_actions[:shared] or old_states[:shared+1] != new_states[:shared+1]:
                        raise ValueError("Fixed-weight audit differs from original six-step trajectory prefix")
                    row["original_h6_success"] = int(original["success"])
                    hit_by_six = row["outcome"] == "hit" and row["first_hit_latency"] <= config.horizon
                    if hit_by_six != bool(int(original["success"])):
                        raise ValueError("Cycle audit changed a recorded H6 outcome")
                rows.append(row)
        if prior:
            raise ValueError("A declared prior noninitial query was omitted")
        if before != state_digest(model):
            raise ValueError("Cycle audit changed fixed model tensors")
        expected = 49 if env.name == "fork_all_landmarks" else 4
        if len(rows) != expected:
            raise ValueError("Cycle audit query denominator differs")
        folder = output / case["condition"] / env.name / f"seed_{case['seed']}"
        folder.mkdir(parents=True)
        write_csv(folder / "queries.csv", rows)
        summary = dict(**case, **summarize(rows), finite_state=finite,
                       model_state_sha256=before, model_unchanged=True, original_h6_prefixes_identical=True,
                       queries_sha256=digest(folder / "queries.csv"))
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps({key: summary[key] for key in ("condition", "environment", "seed", "noninitial")}), flush=True)
    totals = summarize(all_rows)
    # Integer goal labels are local to each task; do not pool fork goal4 with
    # sparse goal4 as though they were the same event. Per-goal counts remain
    # in each checkpoint's summary, where the environment is unambiguous.
    totals.pop("per_goal")
    expected_counts = protocol["evaluation"]
    if (totals["all"]["queries"] != expected_counts["query_count_total"]
            or totals["noninitial"]["queries"] != expected_counts["query_count_noninitial"]
            or totals["initially_satisfied"]["queries"] != expected_counts["query_count_initially_satisfied"]):
        raise ValueError("Complete declared audit denominators changed")
    final_audit = audit_source_snapshots(output / "source_snapshot")
    if not final_audit["archive_valid"] or not final_audit["current_sources_unchanged"]:
        raise ValueError("Inputs or source files changed during cycle audit")
    results = dict(schema="intrmotiv/fixed-weight-exact-cycle-results/v1", runs=summaries, totals=totals,
                   protocol_sha256=PROTOCOL_SHA, prepared_sha256=digest(output / "prepared.json"),
                   source_and_input_audit=final_audit,
                   scope="Post-hoc fixed-weight diagnosis; all original H6 gates retained unchanged")
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    result = prepare(args.output) if args.prepare_only else run_prepared(args.output)
    print(json.dumps({"schema": result["schema"], "output": str(args.output)}), flush=True)
