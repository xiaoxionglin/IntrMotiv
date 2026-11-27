"""Validate and aggregate immutable source-sharded first-hit evaluations.

This is model-free analysis: no environment, checkpoint deserialization, or
learning. Trial bytes are verified before JSONL rows are read; large paths and
action lists are discarded after consistency checks. Missing observations stay
unknown. In particular, command-dependent stopping can inflate ordinary arrival
lift even for a goal-blind policy. The competing-target FIRST contrast avoids
that artifact by comparing the same event under two actually executed commands.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .dmlab_evaluate import summarize_rows
from .provenance import audit_source_snapshots, snapshot_sources


POLICIES = ("source", "first_hit")
LIGHT_FIELDS = {
    "policy", "source", "repeat", "command", "target", "commanded", "issued", "missing", "missing_reason",
    "prefix_seed", "exact_start_verified", "horizon", "elapsed", "hit", "hit_time", "exclusive_hit",
    "exclusive_hit_time", "initially_at_target", "initially_at_command", "command_hit", "physical_done",
    "successor_unknown", "censored", "timeout", "first_distinct", "initial_scores", "stopped_at_other_command",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _reject_nonfinite(value):
    raise ValueError(f"Nonfinite trial value: {value}")


@dataclass
class Shard:
    directory: Path
    summary: dict
    manifest: dict
    identity: dict
    rows: list
    receipt: dict


def _source_identity(directory, manifest):
    archive = directory / "source_snapshot"
    audit = audit_source_snapshots(archive, check_current=False)
    if not audit["archive_valid"]:
        raise ValueError("Archived evaluator sources failed integrity validation")
    saved = _read_json(archive / "manifest.json")
    if saved != manifest.get("source_snapshot"):
        raise ValueError("Pre-run source declaration differs from the saved source archive")
    root = Path(manifest["source_root"])
    files = {}
    for entry in saved["sources"]:
        path = Path(entry["source_path"])
        if path.name == "canonical_target_control_with_callback.py":
            key = "canonical_callback_overlay"
        elif path.is_relative_to(root):
            key = path.relative_to(root).as_posix()
        else:
            raise ValueError(f"Unrecognized evaluator source outside declared runtime: {path}")
        if key in files:
            raise ValueError("Ambiguous archived source identity")
        files[key] = entry["sha256"]
    return files


def _check_bool(row, key):
    if type(row.get(key)) is not bool:
        raise ValueError(f"Trial {key} must be a JSON boolean")


def _read_trials(path, summary):
    expected_hash = summary["trials_sha256"]
    if file_hash(path) != expected_hash:
        raise ValueError("Trial bytes differ from the completed shard receipt")
    config = summary["config"]
    sources, targets = set(summary["source_panel"]), set(summary["eligible_targets"])
    repeats, horizon = config["repeats"], config["horizon"]
    if any(type(value) is not int or value < 1 for value in (repeats, horizon)):
        raise ValueError("Declared repeats and horizon must be positive integers")
    expected = {(policy, source, repeat, command, target) for policy in POLICIES for source in sources
                for repeat in range(repeats) for command in targets for target in targets}
    seen, rows, path_checks, starts, issuance = set(), [], {}, {}, {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank trial record at line {line_number}")
            row = json.loads(line, parse_constant=_reject_nonfinite)
            for key in ("source", "repeat", "command", "target"):
                if type(row.get(key)) is not int:
                    raise ValueError(f"Trial {key} must be an integer")
            key = tuple(row[k] for k in ("policy", "source", "repeat", "command", "target"))
            if key not in expected or key in seen:
                raise ValueError("Trial key is undeclared or duplicated")
            seen.add(key)
            for field in ("commanded", "issued", "missing", "hit", "censored", "initially_at_target", "initially_at_command"):
                _check_bool(row, field)
            if (row["commanded"] != (row["command"] == row["target"])
                    or row["missing"] == row["issued"]
                    or row["initially_at_target"] != (row["source"] == row["target"])
                    or row["initially_at_command"] != (row["source"] == row["command"])):
                raise ValueError("Trial command, initial-goal, or issued/missing flags disagree")
            block_key = (row["source"], row["repeat"])
            block_status = (row["issued"], None if row["issued"] else row.get("missing_reason"))
            if block_key in issuance and issuance[block_key] != block_status:
                raise ValueError("Matched source-history block mixes issued and missing command paths")
            issuance[block_key] = block_status
            if not row["issued"]:
                if row["hit"] or row["censored"] or not row.get("missing_reason"):
                    raise ValueError("An unissued command must remain explicitly missing")
            else:
                for field in ("exact_start_verified", "exclusive_hit", "physical_done", "successor_unknown", "command_hit"):
                    _check_bool(row, field)
                if not row["exact_start_verified"] or row["horizon"] != horizon:
                    raise ValueError("Executed command lacks the declared exact start or horizon")
                elapsed = row["elapsed"]
                if type(elapsed) is not int or not 0 <= elapsed <= horizon:
                    raise ValueError("Invalid elapsed trial deadline")
                hit_time = row.get("hit_time")
                if row["hit"] != (hit_time is not None) or (hit_time is not None and
                        (type(hit_time) is not int or not 0 <= hit_time <= elapsed)):
                    raise ValueError("Hit flag/time disagree")
                if (hit_time == 0) != row["initially_at_target"]:
                    raise ValueError("Initial winner and first-hit time disagree")
                if row["censored"] != (row["successor_unknown"] and not row["hit"]):
                    raise ValueError("Censoring must mean an unresolved terminal successor")
                if row["initially_at_command"] and elapsed != 0:
                    raise ValueError("An initially satisfied command must not act")
                if row["commanded"] and row["hit"] and elapsed != hit_time:
                    raise ValueError("Own-command trial continued after its hit")
                if row["commanded"] and row["command_hit"] != row["hit"]:
                    raise ValueError("Own-command hit disagrees with path stopping status")
                prefix = row.get("prefix_actions")
                actions = row.get("executed_actions")
                if not isinstance(prefix, list) or not isinstance(actions, list) or len(actions) != elapsed:
                    raise ValueError("Executed prefix/action trace is missing or misaligned")
                start_key = (row["source"], row["repeat"])
                start_signature = (row["prefix_seed"], _canonical_hash(prefix))
                if start_key in starts and starts[start_key] != start_signature:
                    raise ValueError("Commands or policies used different physical prefixes")
                starts[start_key] = start_signature
                path_key = key[:4]
                signature = (start_signature, _canonical_hash(actions), elapsed, row["command_hit"],
                             row["physical_done"], row["successor_unknown"])
                if path_key in path_checks and path_checks[path_key] != signature:
                    raise ValueError("Cross-target rows disagree about their executed command path")
                path_checks[path_key] = signature
            rows.append({key: value for key, value in row.items() if key in LIGHT_FIELDS})
    if seen != expected:
        raise ValueError(f"Incomplete declared trial Cartesian product: expected {len(expected)}, found {len(seen)}")
    # Second hash detects files being changed during the streaming parse.
    if file_hash(path) != expected_hash:
        raise ValueError("Trial bytes changed while aggregation was reading them")
    return rows


def read_shard(directory):
    directory = Path(directory).resolve(strict=True)
    summary_path, manifest_path = directory / "summary.json", directory / "pre_run_manifest.json"
    summary, manifest = _read_json(summary_path), _read_json(manifest_path)
    if summary.get("schema") != "intrmotiv/matched-first-hit-evaluation/v2":
        raise ValueError("Require the qualified v2 evaluation outcome contract")
    if manifest.get("schema") != "intrmotiv/matched-first-hit-prerun/v1":
        raise ValueError("Missing canonical pre-run identity")
    if summary["config"] != manifest["evaluation_config"] or summary["config"]["phase"] != "full":
        raise ValueError("Require the declared full evaluation configuration")
    sources = summary["source_panel"]
    if (not sources or any(type(source) is not int for source in sources)
            or len(sources) != len(set(sources)) or sorted(sources) != sorted(summary["config"]["focus_sources"])):
        raise ValueError("Require a unique explicitly declared source shard")
    fit = manifest["fitted_metadata"]
    if (fit["source_checkpoint_sha256"] != manifest["source_checkpoint_sha256"]
            or fit["source_config_sha256"] != manifest["original_config_sha256"]
            or fit["primary_detector"] != "winner" or summary["primary_detector"] != "winner"):
        raise ValueError("Source donor or detector identity disagrees with fitted replay")
    if fit["layout"] != summary["eligibility"]["layout"]:
        raise ValueError("Fitted and executed goal vocabularies differ")
    goals = fit["layout"]["goals"]
    if type(goals) is not int or goals < 1:
        raise ValueError("DG capacity must be a positive integer")
    if summary["declared_dg_vocabulary"] != list(range(goals)):
        raise ValueError("Incomplete declared DG capacity")
    observed, targets = summary["observed_targets"], summary["eligible_targets"]
    if (any(type(goal) is not int for goal in observed + targets)
            or len(observed) != len(set(observed)) or sorted(targets) != sorted(observed)
            or not set(observed) <= set(range(goals))):
        raise ValueError("Full targets must equal the complete observed vocabulary")
    if sorted(summary["unobserved_goal_ids"]) != sorted(set(range(goals)) - set(observed)):
        raise ValueError("Unobserved target denominator is inconsistent")
    for key in ("policies_frozen", "graph_frozen", "q_frozen", "checkpoint_unchanged", "source_configuration_unchanged"):
        if summary.get(key) is not True:
            raise ValueError(f"Shard lacks successful frozen evaluation gate: {key}")
    source_audit = summary.get("source_audit", {})
    if source_audit.get("archive_valid") is not True or source_audit.get("current_sources_unchanged") is not True:
        raise ValueError("Evaluation did not certify source integrity at completion")
    if summary["frozen_sha256_before"] != summary["frozen_sha256_after"]:
        raise ValueError("Model, graph or Q changed during evaluation")
    if set(summary["frozen_sha256_before"]) != {"actor", "graph", "readout", "q"}:
        raise ValueError("Incomplete frozen state identity")
    if file_hash(directory / "matched_starts.pt") != summary["matched_starts_sha256"]:
        raise ValueError("Saved matched-start artifact failed hash validation")
    source_identity = _source_identity(directory, manifest)
    config = {k: v for k, v in summary["config"].items() if k != "focus_sources"}
    overrides = {k: v for k, v in manifest["evaluation_overrides"].items() if k != "dmlab_level_cache_path"}
    identity = dict(source_checkpoint_sha256=manifest["source_checkpoint_sha256"],
        source_config_sha256=manifest["original_config_sha256"], fitted_checkpoint_sha256=fit["checkpoint_sha256"],
        fitted_metadata_sha256=_canonical_hash(fit), source_root=manifest["source_root"], sources=source_identity,
        canonical_source_sha256=manifest["canonical_source_sha256"], callback_source_sha256=manifest["callback_source_sha256"],
        frozen_state=summary["frozen_sha256_before"], config=config, overrides=overrides,
        goal_vocabulary=summary["declared_dg_vocabulary"], observed_targets=sorted(observed),
        detector=summary["primary_detector"], action_protocol=summary["action_protocol"],
        command_protocol=summary["command_protocol"], terminal_protocol=summary["terminal_protocol"])
    rows = _read_trials(directory / "trials.jsonl", summary)
    receipt = dict(directory=str(directory), summary_sha256=file_hash(summary_path),
        pre_run_manifest_sha256=file_hash(manifest_path), trials_sha256=summary["trials_sha256"],
        matched_starts_sha256=summary["matched_starts_sha256"],
        archived_sources_manifest_sha256=file_hash(directory / "source_snapshot/manifest.json"),
        source_ids=sources, trial_rows=len(rows))
    return Shard(directory, summary, manifest, identity, rows, receipt)


def arrival_interval(row):
    """A missing or censored failure is unknown, never a confirmed zero."""
    if row is None or not row["issued"]:
        return 0.0, 1.0
    if row["hit"]:
        return 1.0, 1.0
    return (0.0, 1.0) if row["censored"] else (0.0, 0.0)


def first_before_interval(target_row, competitor_row, horizon):
    """First member of {target, competitor} under one actually issued command.

    Once either member has appeared its order is identified, including when a
    later unrelated terminal successor is unavailable. Both targets must differ
    from the initial winner; callers exclude those trivial comparisons.
    """
    if target_row is None or competitor_row is None or not target_row["issued"] or not competitor_row["issued"]:
        return 0.0, 1.0
    target_time, competitor_time = target_row.get("hit_time"), competitor_row.get("hit_time")
    if target_time is not None and competitor_time is not None and target_time == competitor_time:
        raise ValueError("Distinct winner targets cannot first occur at the same time")
    if target_time is not None and (competitor_time is None or target_time < competitor_time):
        return 1.0, 1.0
    if competitor_time is not None:
        return 0.0, 0.0
    if target_row["censored"] or competitor_row["censored"]:
        return 0.0, 1.0
    if target_row["elapsed"] == horizon:
        return 0.0, 0.0
    raise ValueError("A command stopped before either competing target without a declared censor")


def _contrast_record(source, repeat, goal, left, right, **extra):
    return dict(source=source, repeat=repeat, goal=goal, lower=left[0] - right[1],
                upper=left[1] - right[0], **extra)


def paired_arrival_records(index, sources, goals, repeats):
    records = []
    for source in sources:
        for repeat in range(repeats):
            for goal in goals:
                if source == goal:
                    continue
                candidate = index.get(("first_hit", source, repeat, goal, goal))
                baseline = index.get(("source", source, repeat, goal, goal))
                records.append(_contrast_record(source, repeat, goal, arrival_interval(candidate), arrival_interval(baseline),
                    missing=any(r is None or not r["issued"] for r in (candidate, baseline)),
                    censored=any(r is not None and r["issued"] and r["censored"] for r in (candidate, baseline))))
    return records


def competing_first_records(index, policy, sources, goals, repeats, horizon):
    records = []
    for source in sources:
        for repeat in range(repeats):
            for goal in goals:
                for competitor in goals:
                    if source in (goal, competitor) or goal == competitor:
                        continue
                    own_rows = (index.get((policy, source, repeat, goal, goal)),
                                index.get((policy, source, repeat, goal, competitor)))
                    alternate_rows = (index.get((policy, source, repeat, competitor, goal)),
                                      index.get((policy, source, repeat, competitor, competitor)))
                    own = first_before_interval(*own_rows, horizon)
                    alternate = first_before_interval(*alternate_rows, horizon)
                    terminal_unknown = lambda interval, paths: interval[0] != interval[1] and any(
                        row is not None and row["issued"] and row["censored"] for row in paths)
                    records.append(_contrast_record(source, repeat, goal, own, alternate, competitor=competitor,
                        missing=any(row is None or not row["issued"] for row in own_rows + alternate_rows),
                        censored=terminal_unknown(own, own_rows) or terminal_unknown(alternate, alternate_rows)))
    return records


def _contrast_counts(records):
    lower = np.asarray([r["lower"] for r in records], dtype=float)
    upper = np.asarray([r["upper"] for r in records], dtype=float)
    known = lower == upper
    result = dict(declared_comparisons=len(records), identified_comparisons=int(known.sum()),
        unresolved_comparisons=int((~known).sum()), missing_comparisons=sum(r.get("missing", False) for r in records),
        censored_comparisons=sum(r.get("censored", False) for r in records),
        mean_lower_bound=float(lower.mean()) if len(records) else None,
        mean_upper_bound=float(upper.mean()) if len(records) else None,
        point_estimate_if_fully_identified=float(lower.mean()) if len(records) and known.all() else None)
    return result, lower, upper, known


def contrast_summary(records, sources, *, bootstrap_replicates=10000, bootstrap_seed=123):
    """Descriptive source-cluster uncertainty conditional on this donor/panel.

    Bootstrap interval endpoints over all declared comparisons. No selected
    complete-case estimate or interval is substituted for missing trajectories.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    result, lower, upper, known = _contrast_counts(records)
    cluster = np.zeros((len(sources), 4), dtype=float)
    for i, source in enumerate(sources):
        selected = np.asarray([r["source"] == source for r in records], dtype=bool)
        cluster[i] = (lower[selected].sum(), upper[selected].sum(), selected.sum(), (selected & known).sum())
    with_data = int((cluster[:, 2] > 0).sum())
    known_clusters = int((cluster[:, 3] > 0).sum())
    bootstrap = dict(unit="source identity; retain all histories/goals within each sampled source",
        seed=bootstrap_seed, replicates=bootstrap_replicates, declared_source_clusters=len(sources),
        clusters_with_comparisons=with_data, clusters_with_identified_comparisons=known_clusters,
        interval_level=0.95, identification_endpoint_envelope=None,
        interpretation="Descriptive source-cluster intervals conditional on one frozen donor, one fit, and this start protocol; no replicated training evidence. Every declared comparison stays in the bounds; per-goal intervals are unadjusted descriptive intervals.")
    if len(records) and with_data >= 2:
        draws = np.random.default_rng(bootstrap_seed).integers(0, len(sources), (bootstrap_replicates, len(sources)))
        samples = cluster[draws].sum(axis=1)
        valid = samples[:, 2] > 0
        bootstrap["identification_endpoint_envelope"] = [float(np.quantile(samples[valid, 0] / samples[valid, 2], .025)),
            float(np.quantile(samples[valid, 1] / samples[valid, 2], .975))]
        bootstrap["bootstrap_draws_with_no_comparisons"] = int((~valid).sum())
        bootstrap["bootstrap_draws_with_no_identified_comparisons"] = int((samples[:, 3] == 0).sum())
    result["source_cluster_bootstrap"] = bootstrap
    return result


def _per_goal_contrasts(records, goals, sources, settings):
    return {str(goal): contrast_summary([r for r in records if r["goal"] == goal], sources, **settings) for goal in goals}


def _flat_counts(index, policy, source, goal, repeats):
    rows = [index.get((policy, source, repeat, goal, goal)) for repeat in range(repeats)]
    intervals = [arrival_interval(row) for row in rows]
    trivial = source == goal
    return {f"{policy}_issued": sum(r is not None and r["issued"] for r in rows),
        f"{policy}_missing": sum(r is None or not r["issued"] for r in rows),
        f"{policy}_censored": sum(r is not None and r["issued"] and r["censored"] for r in rows),
        f"{policy}_confirmed_hits": sum(r is not None and r["issued"] and r["hit"] for r in rows),
        f"{policy}_success_lower": None if trivial else float(np.mean([x[0] for x in intervals])),
        f"{policy}_success_upper": None if trivial else float(np.mean([x[1] for x in intervals]))}


def aggregate(shard_directories, output, *, bootstrap_replicates=10000, bootstrap_seed=123):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite a completed analysis directory")
    analysis_sources = [Path(__file__), Path(__file__).with_name("dmlab_evaluate.py"),
                        Path(__file__).with_name("provenance.py")]
    analysis_source_hashes = {str(path): file_hash(path) for path in analysis_sources}
    directories = [Path(path).resolve() for path in shard_directories]
    if len(directories) != 4 or len(set(directories)) != 4:
        raise ValueError("Require exactly four distinct completed source shards")
    shards = [read_shard(directory) for directory in directories]
    identity = shards[0].identity
    if any(shard.identity != identity for shard in shards[1:]):
        raise ValueError("Shard fitted/source/configuration identities do not match")
    goals = identity["goal_vocabulary"]
    sources = [source for shard in shards for source in shard.summary["source_panel"]]
    if len(sources) != len(set(sources)) or sorted(sources) != goals:
        raise ValueError("Source shards must form a disjoint complete declared source partition")
    output.mkdir(parents=True)
    source_snapshot = snapshot_sources(analysis_sources, output / "source_snapshot")
    if any(entry["sha256"] != analysis_source_hashes[entry["source_path"]] for entry in source_snapshot["sources"]):
        raise RuntimeError("Analysis sources changed during input validation")
    write_protocol = dict(schema="intrmotiv/evaluation-aggregation-protocol/v1",
        input_receipts=[shard.receipt for shard in shards], source_snapshot=source_snapshot,
        bootstrap=dict(replicates=bootstrap_replicates, seed=bootstrap_seed, cluster="source identity"))
    (output / "protocol.json").write_text(json.dumps(write_protocol, indent=2) + "\n")
    sources.sort()
    rows = [row for shard in shards for row in shard.rows]
    index = {tuple(row[k] for k in ("policy", "source", "repeat", "command", "target")): row for row in rows}
    repeats, horizon = identity["config"]["repeats"], identity["config"]["horizon"]
    settings = dict(bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)
    outcomes = summarize_rows(rows, goal_vocabulary=goals,
        declared_starts=[(source, repeat) for source in sources for repeat in range(repeats)],
        observed_targets=identity["observed_targets"])
    for result in outcomes.values():
        result["arrival_lift_interpretation"] = "Stopped-task arrival contrast only. Positive lift can arise from own-command stopping with identical goal-blind actions; use competing-target FIRST for selective navigation evidence."
    paired = paired_arrival_records(index, sources, goals, repeats)
    competing = {policy: competing_first_records(index, policy, sources, goals, repeats, horizon) for policy in POLICIES}
    paired_per_goal = _per_goal_contrasts(paired, goals, sources, settings)
    first = {policy: dict(overall=contrast_summary(records, sources, **settings),
                         per_goal=_per_goal_contrasts(records, goals, sources, settings))
             for policy, records in competing.items()}
    result = dict(schema="intrmotiv/first-hit-evaluation-aggregate/v1", status="validated_completed_shards",
        identities=identity, input_receipts=[shard.receipt for shard in shards],
        source_partition=[shard.summary["source_panel"] for shard in shards], retained_light_trial_rows=len(rows),
        heavy_trajectory_action_arrays_retained=False, outcomes=outcomes,
        paired_candidate_minus_source=dict(overall=contrast_summary(paired, sources, **settings), per_goal=paired_per_goal),
        competing_target_FIRST=first,
        competing_target_definition="For ordered distinct goals g,c, neither initially satisfied: P(g before c within H | command g) minus the same event under command c, on the exact matched start. Known first c is zero even if later terminal is unknown; neither at an uncertified terminal is unknown.",
        scientific_limits=["One pretrained donor and one fitted initialization; source clusters are not independent training seeds.",
            "Missing and censored comparisons stay within identification bounds; they are never confirmed failures.",
            "Positive competing-FIRST contrast supports command-dependent ordering on tested paths; zero contrast does not refute goal reaching when a competing target is an unavoidable intermediate. No every-pair-positive gate is justified.",
            "Ordinary own-stop arrival lift and initial action sensitivity alone do not establish selective navigation.",
            "Bootstrap intervals are conditional finite-panel uncertainty, and per-goal intervals are descriptive without multiplicity adjustment."])
    per_goal_rows = []
    for goal in goals:
        row = dict(goal=goal, observed=goal in identity["observed_targets"])
        for policy in POLICIES:
            counts = outcomes[policy]["per_goal"][str(goal)]
            row.update({f"{policy}_{key}": counts[key] for key in ("declared_nontrivial_own_rows", "issued_nontrivial_own_rows",
                "missing_nontrivial_own_rows", "censored_nontrivial_own_rows", "successful_nontrivial_own_rows",
                "success_lower_bound_declared", "success_upper_bound_declared")})
            contrast = first[policy]["per_goal"][str(goal)]
            row.update({f"{policy}_FIRST_{key}": contrast[key] for key in ("declared_comparisons", "identified_comparisons", "mean_lower_bound", "mean_upper_bound")})
        row.update({f"candidate_minus_source_{key}": paired_per_goal[str(goal)][key] for key in
                    ("declared_comparisons", "identified_comparisons", "mean_lower_bound", "mean_upper_bound")})
        per_goal_rows.append(row)
    source_target_rows = []
    for source in sources:
        for goal in goals:
            row = dict(source=source, goal=goal, declared_histories=repeats, initially_satisfied=source == goal)
            for policy in POLICIES:
                row.update(_flat_counts(index, policy, source, goal, repeats))
                ordering = [r for r in competing[policy] if r["source"] == source and r["goal"] == goal]
                row[f"{policy}_FIRST_comparisons"] = len(ordering)
                row[f"{policy}_FIRST_identified"] = sum(r["lower"] == r["upper"] for r in ordering)
                row[f"{policy}_FIRST_lower"] = float(np.mean([r["lower"] for r in ordering])) if ordering else None
                row[f"{policy}_FIRST_upper"] = float(np.mean([r["upper"] for r in ordering])) if ordering else None
            selected = [r for r in paired if r["source"] == source and r["goal"] == goal]
            row["candidate_minus_source_lower"] = float(np.mean([r["lower"] for r in selected])) if selected else None
            row["candidate_minus_source_upper"] = float(np.mean([r["upper"] for r in selected])) if selected else None
            source_target_rows.append(row)
    competing_target_rows = []
    for policy in POLICIES:
        for goal in goals:
            for competitor in goals:
                if goal != competitor:
                    selected = [r for r in competing[policy] if r["goal"] == goal and r["competitor"] == competitor]
                    counts, _, _, _ = _contrast_counts(selected)
                    competing_target_rows.append(dict(policy=policy, goal=goal, competitor=competitor, **counts))
    tables = (("per_goal.csv", per_goal_rows), ("source_target.csv", source_target_rows),
              ("competing_targets.csv", competing_target_rows))
    for name, table in tables:
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    result["outputs"] = {name: file_hash(output / name) for name, _ in tables}
    result["analysis_sources"] = analysis_source_hashes
    result["analysis_source_audit"] = audit_source_snapshots(output / "source_snapshot")
    if not result["analysis_source_audit"]["archive_valid"] or not result["analysis_source_audit"]["current_sources_unchanged"]:
        raise RuntimeError("Analysis sources changed during aggregation")
    result["analysis_protocol_sha256"] = file_hash(output / "protocol.json")
    (output / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.shards, args.output)
    print(json.dumps({"output": str(args.output), "trial_rows": result["retained_light_trial_rows"],
                     "paired_bounds": result["paired_candidate_minus_source"]["overall"]}))


if __name__ == "__main__":
    main()
