"""Matched frozen-policy evaluation through the canonical DMLab engine helpers.

Importing this module never imports or constructs DMLab. ``install_callback``
adds one dispatch point to a saved copy of the existing evaluator; its fresh
engine, action preprocessing, recurrent prefix replay, and state signatures are
reused unchanged. Only an allocated NEMO job may invoke the executable entry.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .detectors import detector_hits
from .runtime_adapter import fixed_goal_input


def module_hash(module):
    """Include names, dtypes, shapes, parameters and persistent graph buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_source_config(source_config, fitted_metadata, evaluation_overrides):
    """Bind effective perception, action timing, and assets to collection.

    Only the per-run cache destination may differ. In particular, normalization,
    detector settings, action repeat, freezing, and the runfiles asset path may
    not drift behind an unchanged checkpoint filename or tensor layout.
    """
    actual = hashlib.sha256(Path(source_config).read_bytes()).hexdigest()
    if actual != fitted_metadata.get("source_config_sha256"):
        raise ValueError("Source configuration differs from frozen replay collection")
    collected = fitted_metadata.get("collection_evaluation_overrides")
    if not isinstance(collected, dict):
        raise ValueError("Fitted metadata lacks collection evaluation overrides")
    semantic = lambda values: {key: value for key, value in values.items() if key != "dmlab_level_cache_path"}
    if semantic(collected) != semantic(evaluation_overrides):
        raise ValueError("Evaluation overrides differ from frozen replay collection")
    return actual


def install_callback(source: str) -> str:
    """Fail closed if the canonical helper contract changes; never edit its file."""
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "run_landmark_matched_interventions"]
    if len(matches) != 1:
        raise ValueError("Canonical matched evaluator is missing or duplicated")
    function = matches[0]
    nested = {node.name for node in function.body if isinstance(node, ast.FunctionDef)}
    if not {"fresh", "advance", "encode", "signature", "replay"}.issubset(nested):
        raise ValueError("Canonical engine/prefix helper contract changed")
    lines = source.splitlines(keepends=True)
    block = "".join(lines[function.lineno - 1:function.end_lineno])
    signature_anchor = "    discovery_multiplier=2,\n"
    dispatch_anchor = "    try:\n        with torch.no_grad():\n"
    if block.count(signature_anchor) != 1 or block.count(dispatch_anchor) != 1:
        raise ValueError("Canonical callback insertion anchors changed")
    if "first_hit_evaluator" in block:
        raise ValueError("Canonical source is already patched")
    block = block.replace(signature_anchor, signature_anchor + "    first_hit_evaluator=None,\n")
    dispatch = '''    if first_hit_evaluator is not None:
        return first_hit_evaluator(dict(
            cfg=cfg, actor=actor, graph=graph, checkpoint=checkpoint,
            device=device, decision_cap=decision_cap, fresh=fresh,
            advance=advance, encode=encode, signature=signature, replay=replay,
            decisions=lambda: decisions, close=lambda: env.close(),
        ))

'''
    block = block.replace(dispatch_anchor, dispatch + dispatch_anchor)
    patched = "".join(lines[:function.lineno - 1]) + block + "\n" + "".join(lines[function.end_lineno:])
    ast.parse(patched)
    return patched


@dataclass(frozen=True)
class EvaluationConfig:
    horizon: int = 256
    discovery_episodes: int = 32
    prefix_cap: int = 512
    repeats: int = 3
    max_sources: int = 16
    discovery_seed: int = 31000
    phase: str = "full"
    preflight_targets: tuple[int, ...] = ()
    focus_sources: tuple[int, ...] = ()

    def __post_init__(self):
        for key in ("horizon", "discovery_episodes", "prefix_cap", "repeats", "max_sources"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.phase not in ("preflight", "full"):
            raise ValueError("phase must be preflight or full")
        if self.phase == "full" and self.preflight_targets:
            raise ValueError("Full evaluation cannot truncate the observed target vocabulary")
        if self.phase == "preflight" and (len(set(self.preflight_targets)) != 2 or self.max_sources != 1):
            raise ValueError("Bounded preflight requires exactly two targets and one source")


def _same_signature(left, right):
    return left.keys() == right.keys() and all(torch.equal(torch.as_tensor(left[k]), torch.as_tensor(right[k]))
                                              for k in left)


def _frozen_hashes(actor, adapter):
    return {"actor": module_hash(actor), "graph": module_hash(actor.core.policy_graph),
            "readout": module_hash(adapter), "q": module_hash(adapter.q_heads)}


def _goal_ids(dg, mode):
    return torch.nonzero(detector_hits(dg, mode)[0]).flatten().cpu().tolist()


def current_core_dg(actor, core_output):
    """Read actual post-processing DG in CA3 slot zero, matching the manager.

    The canonical encoder helper also returns raw visual head DG. Refractory
    filtering can change that value before the core injects it, so raw head DG
    is not an interchangeable arrival detector.
    """
    core = actor.core
    goals, expanded = int(core.Hippo_n_feature), int(core.expanded_length)
    width = int(core.core_output_size)
    if width != goals * expanded or core_output.ndim != 2 or core_output.shape[1] < width:
        raise ValueError("Unverified scalar CA3 layout for current landmark detector")
    return core_output[:, :width].reshape(-1, goals, expanded)[:, :, 0]


def execute_trial(api, adapter, *, policy, start, command, targets, horizon):
    """Execute one command once, then score every target on that same path.

    Own winner success is an absorbing stop. The exclusive predicate is a
    secondary annotation of this primary-policy trajectory, not a second run.
    Canonical ``advance`` lacks certified terminal observations: a physical
    boundary therefore censors unresolved arrivals and never scores reset DG.
    """
    actor, device = api["actor"], api["device"]
    obs, state = api["replay"](start["seed"], start["prefix"])
    if not _same_signature(start["signature"], api["signature"](obs, state)):
        raise RuntimeError("Physical observation or recurrent prefix state did not reproduce exactly")
    _, core, state = api["encode"](obs, state, command)
    dg = current_core_dg(actor, core)
    initial_winners = set(_goal_ids(dg, "winner"))
    initial_exclusive = set(_goal_ids(dg, "exclusive"))
    winner_times = {g: 0 if g in initial_winners else None for g in targets}
    exclusive_times = {g: 0 if g in initial_exclusive else None for g in targets}
    already_command = command in initial_winners
    own_hit, physical_done, elapsed = already_command, False, 0
    first_distinct = None
    initial_scores, actions = None, []
    ids = torch.tensor([command], dtype=torch.long, device=device)
    # Both routes receive the same complete core vector and the same goal slice.
    for step in range(horizon if not already_command else 0):
        conditioned = fixed_goal_input(core, ids, adapter.layout)
        if policy == "source":
            scores = actor.forward_tail(conditioned, values_only=False, sample_actions=False)["action_logits"]
        elif policy == "first_hit":
            scores = adapter(conditioned, ids, torch.tensor([horizon - step], device=device))
        else:
            raise ValueError("Unknown evaluation policy")
        if scores.ndim != 2 or scores.shape != (1, actor.action_space.n) or not torch.isfinite(scores).all():
            raise ValueError("Invalid acting logits/Q values")
        if initial_scores is None:
            initial_scores = scores[0].cpu().tolist()
        action = int(scores.argmax(-1).item())
        actions.append(action)
        obs, physical_done = api["advance"](action)
        elapsed = step + 1
        if physical_done:
            break  # Returned obs may be autoreset: do not label it as arrival.
        _, core, state = api["encode"](obs, state, command)
        dg = current_core_dg(actor, core)
        winners, exclusive = _goal_ids(dg, "winner"), _goal_ids(dg, "exclusive")
        for g in winners:
            if g not in initial_winners and first_distinct is None:
                first_distinct = g
            if g in winner_times and winner_times[g] is None:
                winner_times[g] = elapsed
        for g in exclusive:
            if g in exclusive_times and exclusive_times[g] is None:
                exclusive_times[g] = elapsed
        own_hit = command in winners
        if own_hit:
            break
    rows = []
    for target in targets:
        hit = winner_times[target] is not None
        censored = physical_done and not hit
        rows.append(dict(
            policy=policy, source=start["source"], repeat=start["repeat"], prefix_seed=start["seed"],
            prefix_actions=start["prefix"], command=command, target=target, commanded=command == target,
            issued=True, missing=False, missing_reason=None, exact_start_verified=True,
            horizon=horizon, elapsed=elapsed, hit=hit, hit_time=winner_times[target],
            exclusive_hit=exclusive_times[target] is not None, exclusive_hit_time=exclusive_times[target],
            initially_at_target=target in initial_winners, initially_at_command=already_command,
            command_hit=own_hit, physical_done=physical_done, successor_unknown=physical_done,
            censored=censored, timeout=not hit and not own_hit and not physical_done,
            stopped_at_other_command=not hit and own_hit, first_distinct=first_distinct,
            initial_scores=initial_scores, executed_actions=actions,
        ))
    return rows


def summarize_rows(rows):
    """Report all declared/issued cases; censoring never silently disappears."""
    summaries = {}
    for policy in ("source", "first_hit"):
        selected = [r for r in rows if r["policy"] == policy]
        own = [r for r in selected if r["commanded"]]
        nontrivial = [r for r in own if not r.get("initially_at_target", False)]
        issued = [r for r in nontrivial if r["issued"]]
        differences, lift_lower, lift_upper = [], [], []
        for primary in issued:
            alternatives = [r for r in selected if r["issued"] and not r["commanded"]
                            and not r["initially_at_command"] and not r["initially_at_target"]
                            and all(r[k] == primary[k] for k in ("source", "repeat", "target"))]
            if alternatives:
                differences.append(float(primary["hit"]) - float(np.mean([r["hit"] for r in alternatives])))
                lift_lower.append(float(primary["hit"]) - float(np.mean([r["hit"] or r["censored"] for r in alternatives])))
                lift_upper.append(float(primary["hit"] or primary["censored"]) - float(np.mean([r["hit"] for r in alternatives])))
        summaries[policy] = dict(
            declared_own_rows=len(own), initially_satisfied_own_rows=sum(r.get("initially_at_target", False) for r in own),
            declared_nontrivial_own_rows=len(nontrivial), issued_nontrivial_own_rows=len(issued),
            missing_nontrivial_own_rows=sum(not r["issued"] for r in nontrivial),
            censored_nontrivial_own_rows=sum(r["censored"] for r in issued),
            successful_nontrivial_own_rows=sum(r["hit"] for r in issued),
            success_lower_bound_declared=sum(r["hit"] for r in issued) / len(nontrivial) if nontrivial else None,
            success_lower_bound_issued=sum(r["hit"] for r in issued) / len(issued) if issued else None,
            success_upper_bound_declared=(sum(r["hit"] or r["censored"] for r in issued)
                                          + len(nontrivial) - len(issued)) / len(nontrivial) if nontrivial else None,
            exclusive_successful_nontrivial_own_rows=sum(r["exclusive_hit"] for r in issued),
            paired_lift_censored_as_unconfirmed=float(np.mean(differences)) if differences else None,
            paired_lift_identification_lower=float(np.mean(lift_lower)) if lift_lower else None,
            paired_lift_identification_upper=float(np.mean(lift_upper)) if lift_upper else None,
            n_lift_pairs=len(differences),
            executed_action_trials=len({(r["source"], r["repeat"], r["command"]) for r in selected if r["issued"]}),
        )
    return summaries


class MatchedFirstHitEvaluator:
    """Callback consuming canonical engine/prefix operations; no environment code."""

    def __init__(self, adapter, config: EvaluationConfig, *, observed_targets=()):
        self.adapter, self.config = adapter, config
        self.observed_targets = set(int(g) for g in observed_targets)

    def __call__(self, api):
        actor, config, adapter = api["actor"], self.config, self.adapter
        n = adapter.layout.goals
        if adapter.horizon_scale != config.horizon:
            raise ValueError("Evaluation deadline differs from fitted readout deadline")
        if any(g < 0 or g >= n for g in (*self.observed_targets, *config.preflight_targets, *config.focus_sources)):
            raise ValueError("Declared target/source is outside the DG vocabulary")
        if actor.training or adapter.training:
            raise ValueError("Both source actor and candidate must already be in evaluation mode")
        before = _frozen_hashes(actor, adapter)
        observed = self.observed_targets | set(torch.nonzero(api["graph"].node_visits >= 1).flatten().cpu().tolist())
        panel, rows, missing = {}, [], []
        discovery_missing = False
        try:
            with torch.no_grad():
                # Discovery is reward-free random action replay, independent of
                # target-command outcomes and passive graph connectivity.
                for attempt in range(config.discovery_episodes):
                    if api["decisions"]() + config.prefix_cap > api["decision_cap"] // 4:
                        discovery_missing = True
                        break
                    seed = config.discovery_seed + attempt
                    obs, state = api["fresh"](seed)
                    prefix = []
                    random = np.random.default_rng(seed)
                    retained_this_episode = set()
                    for _ in range(config.prefix_cap):
                        signature = api["signature"](obs, state)
                        _, core, state = api["encode"](obs, state)
                        dg = current_core_dg(actor, core)
                        winners = _goal_ids(dg, "winner")
                        observed.update(winners)
                        for source in winners:
                            if ((not config.focus_sources or source in config.focus_sources)
                                    and source not in retained_this_episode
                                    and len(panel.get(source, [])) < config.repeats
                                    and (source in panel or len(panel) < config.max_sources)):
                                panel.setdefault(source, []).append(dict(source=source, repeat=len(panel.get(source, [])),
                                    seed=seed, prefix=list(prefix), signature=signature))
                                retained_this_episode.add(source)
                        action = int(random.integers(actor.action_space.n))
                        obs, done = api["advance"](action)
                        prefix.append(action)
                        if done:
                            break
                targets = sorted(observed) if config.phase == "full" else list(config.preflight_targets)
                if any(g not in observed for g in targets):
                    raise ValueError("Preflight targets must be observed under the declared winner predicate")
                if config.phase == "full" and config.max_sources < len(observed) and not config.focus_sources:
                    raise ValueError("Full panel cannot silently cap the observed source vocabulary")
                sources = sorted(config.focus_sources or (observed if config.phase == "full" else panel))
                if config.phase == "preflight" and not sources:
                    sources = [-1]  # Keep the one requested but undiscovered start in the denominator.
                if not targets or not sources:
                    missing.append(dict(reason="no_observed_targets_or_sources"))
                for source in sources:
                    for rep in range(config.repeats):
                        starts = panel.get(source, [])
                        start = starts[rep] if rep < len(starts) else None
                        # Reserve both policies and every requested command as a
                        # matched block, so budget exhaustion cannot favor one.
                        required = 2 * len(targets) * (len(start["prefix"]) + config.horizon) if start else 0
                        reason = ("start_not_discovered" if start is None else
                                  "decision_budget" if api["decisions"]() + required > api["decision_cap"] else None)
                        if reason:
                            missing.append(dict(source=source, repeat=rep, reason=reason))
                            for policy in ("source", "first_hit"):
                                for command in targets:
                                    for target in targets:
                                        rows.append(dict(policy=policy, source=source, repeat=rep, command=command,
                                            target=target, commanded=command == target, issued=False, missing=True,
                                            missing_reason=reason, initially_at_target=target == source,
                                            initially_at_command=command == source, hit=False, censored=False))
                            continue
                        for policy in ("source", "first_hit"):
                            for command in targets:
                                rows.extend(execute_trial(api, adapter, policy=policy, start=start, command=command,
                                                          targets=targets, horizon=config.horizon))
        finally:
            api["close"]()
        after = _frozen_hashes(actor, adapter)
        if before != after:
            raise RuntimeError(f"Evaluation mutated source actor, graph or readout: before={before}, after={after}")
        self.retained_starts = panel
        return rows, dict(schema="intrmotiv/matched-first-hit-evaluation/v1", config=asdict(config),
            checkpoint=str(api["checkpoint"]), decisions=api["decisions"](), eligible_targets=targets,
            observed_targets=sorted(observed), declared_dg_vocabulary=list(range(n)),
            unobserved_goal_ids=sorted(set(range(n)) - observed), source_panel=sources, missing=missing,
            discovery_budget_exhausted=discovery_missing, frozen_sha256_before=before, frozen_sha256_after=after,
            policies_frozen=True, graph_frozen=True, q_frozen=True, outcomes=summarize_rows(rows),
            action_protocol="deterministic argmax source logits versus deterministic argmax first-hit Q",
            primary_detector="winner", secondary_detector="exclusive on primary-stopped trajectory",
            terminal_protocol="Uncertified terminal successor is censored; autoreset observations never score",
            command_protocol="Stop on own winner including t=0; alternative arrivals score only through own-hit stop",
            target_protocol="All observed winner identities, no passive-edge filter" if config.phase == "full"
                            else "Explicit two-goal integration preflight; not all-goal scientific evaluation")


def run_on_nemo(args):
    """Executable integration boundary; native imports occur only after guards."""
    import importlib.util
    import os
    import sys

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Native evaluation requires an allocated NEMO Slurm compute job")
    workspace = Path(os.environ.get("INTRMOTIV_WORKSPACE_ROOT", "/work/classic/fr_xl1014-corridor-geometry")).resolve()
    if not str(workspace).startswith("/work/"):
        raise ValueError("NEMO workspace must be an allocated /work path")
    output, source_root = args.output.resolve(), args.source_root.resolve(strict=True)
    if not output.is_relative_to(workspace) or output.exists():
        raise ValueError("Require a new evaluation output directory in the active allocated workspace")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Source checkpoint differs from the reviewed donor")
    config = EvaluationConfig(horizon=args.horizon, discovery_episodes=args.discovery_episodes,
        prefix_cap=args.prefix_cap, repeats=args.repeats, max_sources=args.max_sources,
        discovery_seed=args.discovery_seed, phase=args.phase, preflight_targets=tuple(args.targets),
        focus_sources=tuple(args.sources))
    output.mkdir(parents=True)
    sys.path.insert(0, str(source_root))
    from .eligibility import certify_slice_only_relabel
    from .provenance import snapshot_sources, audit_source_snapshots
    from .replay_cli import load_fitted_adapter

    adapter, fitted_metadata = load_fitted_adapter(args.fitted, source_root, device="cpu")
    if fitted_metadata.get("source_checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("Fitted readout is not bound to this donor checkpoint")
    if fitted_metadata.get("primary_detector") != "winner":
        raise ValueError("Fitted readout detector differs from C05 winner evaluation")
    adapter.eval().requires_grad_(False)
    canonical = source_root / "sf_working_directories/IntrMotiv/evaluation/target_control_interventions.py"
    patched = install_callback(canonical.read_text())
    patched_path = output / "canonical_target_control_with_callback.py"
    patched_path.write_text(patched)
    local = Path(__file__).parent
    snapshot_paths = [local / name for name in ("dmlab_evaluate.py", "detectors.py", "targets.py",
        "runtime_adapter.py", "legacy_runtime_adapter.py", "eligibility.py", "replay_cli.py", "provenance.py")]
    snapshot_paths += [canonical, patched_path,
        source_root / "sf_working_directories/IntrMotiv/evaluation/place_fields.py"]
    snapshot_paths += [source_root / "sf_working_directories/IntrMotiv/dmlab" / name for name in
                       ("custom_actor_critic.py", "custom_core.py", "custom_decoder.py", "controller_q.py")]
    snapshot = snapshot_sources(snapshot_paths, output / "source_snapshot")
    source_config = args.run_dir / "config.json"
    original = json.loads(source_config.read_text())
    if original.get("env") != "openfield_map2_fixed_loc3_fixedlength_noreward":
        raise ValueError("Qualification requires the reviewed C05 reward-free source level")
    overrides = dict(dmlab_use_level_cache=False, dmlab_level_cache_path=str(output / "cache"),
        dmlab_runfiles_path=str(args.runfiles.resolve(strict=True)), online_spatial_telemetry=False,
        with_wandb=False, transfer_freeze_dg=True)
    source_config_sha256 = verify_source_config(source_config, fitted_metadata, overrides)
    loader = output / "loader"
    loader.mkdir()
    (loader / "config.json").write_text(json.dumps(dict(original, **overrides), indent=2) + "\n")
    receipt = dict(schema="intrmotiv/matched-first-hit-prerun/v1", source_root=str(source_root),
        source_checkpoint=str(args.checkpoint), source_checkpoint_sha256=args.checkpoint_sha256,
        fitted_directory=str(args.fitted.resolve()), fitted_metadata=fitted_metadata,
        original_config_sha256=source_config_sha256, evaluation_overrides=overrides,
        canonical_source_sha256=digest(canonical), callback_source_sha256=digest(patched_path),
        source_snapshot=snapshot, evaluation_config=asdict(config), decision_cap=args.decision_cap)
    (output / "pre_run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # The only native import/construction path in this module.
    import deepmind_lab
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    spec = importlib.util.spec_from_file_location("intrmotiv_first_hit_canonical_evaluator", patched_path)
    canonical_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = canonical_module
    spec.loader.exec_module(canonical_module)
    cfg, env, env_info, actor, checkpoint, device = canonical_module.load_policy_env(
        loader, args.decision_cap, True, 0, args.checkpoint)
    actor.eval().requires_grad_(False)
    try:
        certificate = certify_slice_only_relabel(actor.core, actor.decoder, source_root=source_root,
                                                input_kind="canonical_core_output")
        if certificate["layout"] != asdict(adapter.layout):
            raise ValueError("Source full-core layout differs from fitted readout")
        if actor.action_space.n != adapter.q_heads.main.out_features or env.num_agents != 1:
            raise ValueError("Source and fitted readout action interfaces differ")
        callback = MatchedFirstHitEvaluator(adapter, config)
        rows, summary = canonical_module.run_landmark_matched_interventions(
            cfg, env, env_info, actor, checkpoint, device, args.decision_cap, deterministic=True,
            first_hit_evaluator=callback)
    except BaseException:
        env.close()
        raise
    summary["eligibility"] = certificate
    summary["source_audit"] = audit_source_snapshots(output / "source_snapshot")
    summary["checkpoint_unchanged"] = digest(args.checkpoint) == args.checkpoint_sha256
    summary["source_configuration_unchanged"] = verify_source_config(source_config, fitted_metadata, overrides) == source_config_sha256
    if not summary["source_audit"]["archive_valid"] or not summary["source_audit"]["current_sources_unchanged"]:
        raise RuntimeError("Evaluation source bytes changed during execution")
    if not summary["checkpoint_unchanged"]:
        raise RuntimeError("Source checkpoint bytes changed during execution")
    with (output / "trials.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    torch.save(callback.retained_starts, output / "matched_starts.pt")
    summary["trials_sha256"] = digest(output / "trials.jsonl")
    summary["matched_starts_sha256"] = digest(output / "matched_starts.pt")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "outcomes": summary["outcomes"], "decisions": summary["decisions"]}))
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "run-dir", "checkpoint", "fitted", "output", "runfiles"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--targets", type=int, nargs="*", default=[])
    parser.add_argument("--sources", type=int, nargs="*", default=[])
    for name, default in (("horizon", 256), ("discovery-episodes", 32), ("prefix-cap", 512),
                          ("repeats", 1), ("max-sources", 1), ("discovery-seed", 31000),
                          ("decision-cap", 100000)):
        parser.add_argument("--" + name, type=int, default=default)
    run_on_nemo(parser.parse_args())


if __name__ == "__main__":
    main()
