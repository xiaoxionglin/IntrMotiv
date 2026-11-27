"""Post-audit competing-FIRST analysis of already saved fully observed toy paths.

No environment, trained model, optimizer, or policy replay is constructed.
Only the all-landmarks fork datasets qualify: each node ID is exactly one DG
identity, and every path includes every visited decision-time state. The
shuffled_goal rows are excluded because their goal field is not their command.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .aggregate_evaluation import first_before_interval


ROUTES = ("first_hit", "continuing_main", "memoryless")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_first(states, target, competitor):
    """Independent trace oracle: scan until the first competing DG event."""
    if target == competitor or states[0] in (target, competitor):
        raise ValueError("Require distinct goals, neither initially satisfied")
    for state in states[1:]:
        if state in (target, competitor):
            return int(state == target)
    return 0


def classified_first(row, target, competitor, horizon):
    """Cross-check the production interval classifier on fully known paths."""
    def event(goal):
        return dict(issued=True, censored=False, elapsed=row["decisions"],
                    hit_time=row["states"].index(goal) if goal in row["states"] else None)
    interval = first_before_interval(event(target), event(competitor), horizon)
    oracle = trace_first(row["states"], target, competitor)
    if interval != (float(oracle), float(oracle)):
        raise RuntimeError("Independent statepath oracle and production FIRST classifier disagree")
    return oracle


def read_paths(path, goals, horizon, expected_seed):
    """Validate complete saved command paths, own stopping and matched prefixes."""
    index, prefixes = {}, {}
    skipped = 0
    with Path(path).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw["route"] not in ROUTES:
                skipped += 1
                continue
            row = {key: int(raw[key]) for key in ("start", "goal", "seed", "success", "decisions", "start_at_commanded_goal")}
            row.update(route=raw["route"], states=json.loads(raw["states"]),
                       actions=json.loads(raw["actions"]), prefix=json.loads(raw["prefix"]))
            if raw["environment"] != "fork_all_landmarks" or row["seed"] != expected_seed:
                raise ValueError("Only the declared all-DG fork run can use state IDs as detector identities")
            if (row["start"] not in goals or row["goal"] not in goals or row["start"] == row["goal"]
                    or row["start_at_commanded_goal"] != 0 or row["success"] not in (0, 1)):
                raise ValueError("Invalid nontrivial own-command identity")
            if (not row["states"] or len(row["states"]) != len(row["actions"]) + 1
                    or row["states"][0] != row["start"] or len(row["actions"]) != row["decisions"]
                    or not 1 <= row["decisions"] <= horizon):
                raise ValueError("Saved state/action/deadline trace is incomplete")
            if (any(type(state) is not int or state not in goals for state in row["states"])
                    or any(type(action) is not int or action not in (0, 1, 2) for action in row["actions"])):
                raise ValueError("Trace lies outside the declared DG/action interface")
            if not row["prefix"] or row["prefix"][-1] != row["start"]:
                raise ValueError("Saved initial recurrent prefix is missing")
            if row["start"] in prefixes and prefixes[row["start"]] != row["prefix"]:
                raise ValueError("Commands or routes used different initial histories")
            prefixes[row["start"]] = row["prefix"]
            reached = row["goal"] in row["states"]
            if reached != bool(row["success"]):
                raise ValueError("Saved success disagrees with the complete state trace")
            if reached and row["states"].index(row["goal"]) != len(row["states"]) - 1:
                raise ValueError("Saved own-command trajectory continued after its own hit")
            if not reached and row["decisions"] != horizon:
                raise ValueError("A nonterminal fork path ended early without its own target")
            key = (row["route"], row["start"], row["goal"])
            if key in index:
                raise ValueError("Duplicate saved own-command trajectory")
            index[key] = row
    expected = {(route, start, goal) for route in ROUTES for start in goals for goal in goals if start != goal}
    if set(index) != expected:
        raise ValueError("Saved paths do not cover every declared route and nontrivial start-goal pair")
    return index, skipped


def analyze_csv(path, *, goals, horizon, seed):
    index, skipped = read_paths(path, goals, horizon, seed)
    summaries = {}
    for route in ROUTES:
        comparisons = []
        for start in goals:
            for goal in goals:
                for competitor in goals:
                    if start in (goal, competitor) or goal == competitor:
                        continue
                    own, alternate = index[(route, start, goal)], index[(route, start, competitor)]
                    left = classified_first(own, goal, competitor, horizon)
                    right = classified_first(alternate, goal, competitor, horizon)
                    shorter = min(len(own["actions"]), len(alternate["actions"]))
                    prefix_equal = (own["actions"][:shorter] == alternate["actions"][:shorter]
                                    and own["states"][:shorter + 1] == alternate["states"][:shorter + 1])
                    comparisons.append(dict(start=start, goal=goal, competitor=competitor,
                        difference=left - right,
                        naive=int(goal in own["states"]) - int(goal in alternate["states"]),
                        prefix_equal=prefix_equal))
        differences = [row["difference"] for row in comparisons]
        naive = [row["naive"] for row in comparisons]
        selected = [row for (name, _, _), row in index.items() if name == route]
        summaries[route] = dict(
            own_command_paths=len(selected), own_command_successes=sum(row["success"] for row in selected),
            declared_competing_pairs=len(comparisons), identified_competing_pairs=len(comparisons),
            competing_FIRST_numerator=sum(differences), competing_FIRST_mean=sum(differences) / len(differences),
            difference_counts={str(value): differences.count(value) for value in (-1, 0, 1)},
            stopped_task_lift_numerator=sum(naive), stopped_task_lift_mean=sum(naive) / len(naive),
            stopped_task_lift_interpretation="Secondary; can be positive solely because each command stops at its own hit",
            different_prefix_pairs=sum(not row["prefix_equal"] for row in comparisons),
            per_goal={str(goal): dict(comparisons=sum(row["goal"] == goal for row in comparisons),
                numerator=sum(row["difference"] for row in comparisons if row["goal"] == goal)) for goal in goals},
            ordered_goal_competitor_matrix=[[
                None if goal == competitor else sum(row["difference"] for row in comparisons
                    if row["goal"] == goal and row["competitor"] == competitor) / (len(goals) - 2)
                for competitor in goals] for goal in goals],
            matrix_axes=["goal", "competitor"], matrix_comparisons_per_nondiagonal_cell=len(goals) - 2,
            classifier_oracle_agreement=True,
        )
        if route == "continuing_main":
            summaries[route]["goalblind_null_check"] = all(value == 0 for value in differences) and all(
                row["prefix_equal"] for row in comparisons)
    return dict(seed=seed, horizon=horizon, goals=list(goals), saved_rows_analyzed=len(index),
                other_route_rows_excluded=skipped, routes=summaries)


def reanalyze(experiment_root, output):
    """Write a new lightweight report and verify that all inputs stay unchanged."""
    root, output = Path(experiment_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous analyses; choose a new output file")
    from . import aggregate_evaluation
    source_paths = [Path(__file__), Path(aggregate_evaluation.__file__)]
    sources_before = {str(path): file_hash(path) for path in source_paths}
    inputs, suites = {}, {}
    for name, suite_dir, manifest_path in (
        ("film", root / "neural_all_landmarks", root / "neural_all_landmarks/pre_run_manifest.json"),
        ("legacy", root / "neural_legacy_r1/fork_all_landmarks", root / "neural_legacy_r1/pre_run_manifest.json"),
    ):
        inputs[str(manifest_path)] = file_hash(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        results = []
        for seed in config["seeds"]:
            path = suite_dir / f"seed_{seed}" / "trials.csv"
            inputs[str(path)] = file_hash(path)
            result = analyze_csv(path, goals=tuple(range(7)), horizon=config["horizon"], seed=seed)
            result["source_csv"] = str(path)
            result["source_csv_sha256"] = inputs[str(path)]
            results.append(result)
        totals = {}
        for route in ROUTES:
            rows = [result["routes"][route] for result in results]
            count = sum(row["declared_competing_pairs"] for row in rows)
            numerator = sum(row["competing_FIRST_numerator"] for row in rows)
            totals[route] = dict(seeds=len(rows), comparisons=count, numerator=numerator,
                                competing_FIRST_mean=numerator / count,
                                stopped_task_lift_mean=sum(row["stopped_task_lift_numerator"] for row in rows) / count)
        suites[name] = dict(results=results, totals=totals)
    if any(file_hash(path) != digest for path, digest in {**inputs, **sources_before}.items()):
        raise RuntimeError("Saved inputs or analysis sources changed during reanalysis")
    report = dict(schema="intrmotiv/toy-competing-first-reanalysis/v1",
        analyzed_utc=datetime.now(timezone.utc).isoformat(),
        designation="Post-audit reanalysis of saved trajectories; not an originally preregistered toy metric",
        no_new_training=True, no_environment_execution=True, no_policy_replay=True,
        definition="P(g before c within H | command g) minus P(g before c within H | command c), distinct g,c neither initially satisfied",
        observation_contract="All seven fork states are distinct DG identities; saved decision-time state paths are complete",
        excluded_routes="shuffled_goal: stored goal labels identify evaluated targets, not actual issued commands",
        classifier_check="Independent scan-to-first-competing-state oracle agrees with aggregate_evaluation.first_before_interval for every outcome",
        limitations=["Same previously collected trajectories and fitted policies; no new independent evidence or tuning",
                     "Continuing main is a synthetic goal-blind negative, not a routing-only historical-policy control",
                     "Zero individual pair contrast can reflect unavoidable landmark order, even with perfect goal arrival",
                     "This fully observed fork does not test general partial-observation memory sufficiency"],
        input_files=inputs, analysis_sources=sources_before, suites=suites)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: suite["totals"] for name, suite in suites.items()}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reanalyze(args.experiment_root, args.output)


if __name__ == "__main__":
    main()
