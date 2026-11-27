"""Candidate-only evaluation against an explicitly imported frozen control.

The original completed source-policy trajectories are never re-executed or
relabeled as newly collected. Their complete source/history/command panel and
missing blocks determine the candidate panel before any candidate outcome.
Aggregation reads separate verified cohorts; it never deserializes a model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .aggregate_evaluation import (LIGHT_FIELDS, Shard, _canonical_hash, _read_json, _read_trials,
                                  _source_identity, file_hash, read_shard, trial_source_records)
from .block_journal import CompleteBlockJournal, validate_complete_journal
from .dmlab_evaluate import execute_trial, install_callback, module_hash, summarize_rows, verify_source_config
from .provenance import audit_source_snapshots, snapshot_sources


SCHEMA = "intrmotiv/reused-control-evaluation/v1"
PRERUN = "intrmotiv/reused-control-prerun/v1"
SIGNATURE_FIELDS = {"__state", "__position", "__rotation"}


def _engine_file(relative):
    return relative == "canonical_callback_overlay" or relative.startswith("sf_working_directories/") or relative in (
        "hpc_runs/first_hit_control/dmlab_evaluate.py", "hpc_runs/first_hit_control/detectors.py",
        "hpc_runs/first_hit_control/runtime_adapter.py")


def signature_hash(signature):
    """Hash full physical observation and full recurrent state, never CA3 only."""
    if not isinstance(signature, dict) or not SIGNATURE_FIELDS < set(signature):
        raise ValueError("Retained start lacks physical pose, observation, or full recurrent state")
    digest = hashlib.sha256()
    for name, value in sorted(signature.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite matched-start signature")
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def baseline_source_blocks(baseline):
    """Retain every source block; read no candidate outcomes for selection."""
    blocks = {}
    path = baseline.directory / "trials.jsonl"
    if file_hash(path) != baseline.receipt["trials_sha256"]:
        raise ValueError("Imported baseline trial bytes changed")
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["policy"] != "source":
                continue
            key = (row["source"], row["repeat"])
            record = dict(issued=row["issued"], missing_reason=row.get("missing_reason"))
            if row["issued"]:
                record.update(seed=row["prefix_seed"], prefix=row["prefix_actions"])
            if key in blocks and blocks[key] != record:
                raise ValueError("Imported source commands do not share one start/issuance block")
            blocks[key] = record
    expected = {(s, r) for s in baseline.summary["source_panel"] for r in range(baseline.summary["config"]["repeats"])}
    if set(blocks) != expected or file_hash(path) != baseline.receipt["trials_sha256"]:
        raise ValueError("Incomplete or changed imported source blocks")
    return blocks


def certify_retained_starts(baseline, panel):
    """Every issued original block must match the saved seed/prefix/signature."""
    blocks = baseline_source_blocks(baseline)
    retained, receipt = {}, []
    if not isinstance(panel, dict):
        raise ValueError("Matched starts must be the original dictionary of retained histories")
    for source, starts in panel.items():
        if type(source) is not int or not isinstance(starts, list):
            raise ValueError("Invalid retained-start panel")
        for start in starts:
            key = (start["source"], start["repeat"])
            if key in retained or start["source"] != source:
                raise ValueError("Duplicate or mislabeled retained history")
            if type(start["seed"]) is not int or not isinstance(start["prefix"], list) or any(type(a) is not int for a in start["prefix"]):
                raise ValueError("Retained prefix actions and seed must be exact integers")
            retained[key] = start
            receipt.append(dict(source=source, repeat=start["repeat"], seed=start["seed"],
                                prefix_sha256=_canonical_hash(start["prefix"]), prefix_length=len(start["prefix"]),
                                full_signature_sha256=signature_hash(start["signature"])))
    for key, block in blocks.items():
        if block["issued"]:
            start = retained.get(key)
            if start is None or start["seed"] != block["seed"] or start["prefix"] != block["prefix"]:
                raise ValueError("Issued baseline block differs from retained matched start")
    # Discovery can retain extra starts outside an issued panel. Keep their
    # signatures in the receipt, but never expand the predeclared source panel.
    return blocks, retained, sorted(receipt, key=lambda r: (r["source"], r["repeat"]))


def validate_candidate_binding(baseline, metadata, plan):
    """Certify donor/config/vocabulary/terminal semantics across both cohorts."""
    old = baseline.manifest["fitted_metadata"]
    for key in ("source_checkpoint_sha256", "source_config_sha256", "layout", "primary_detector"):
        if metadata.get(key) != old.get(key):
            raise ValueError(f"Imported source and new candidate disagree: {key}")
    donor = plan["donor"]
    if (metadata["source_checkpoint_sha256"] != donor["checkpoint_sha256"]
            or metadata["source_config_sha256"] != donor["config_sha256"]):
        raise ValueError("Candidate/source donor differs from declared plan")
    if (metadata.get("kind") != "stationary_projected_first_hit" or metadata.get("gamma") != .99
            or metadata.get("bootstrap_projection") != [0, 1] or metadata["primary_detector"] != "winner"):
        raise ValueError("Require the declared projected stationary raw-head candidate")
    semantic = lambda values: {k: v for k, v in values.items() if k != "dmlab_level_cache_path"}
    if semantic(metadata["collection_evaluation_overrides"]) != semantic(baseline.manifest["evaluation_overrides"]):
        raise ValueError("Candidate collection and imported source engine semantics differ")
    horizon = baseline.summary["config"]["horizon"]
    if metadata["fit"]["config"]["horizon"] != horizon:
        raise ValueError("Candidate external horizon differs from imported source")
    phase = baseline.summary["config"]["phase"]
    if phase == "preflight":
        q = plan["qualification"]
        expected = dict(trials_sha256=q["source_trials_sha256"], summary_sha256=q["source_summary_sha256"],
                        matched_starts_sha256=q["matched_starts_sha256"])
        if any(baseline.receipt[k] != v for k, v in expected.items()):
            raise ValueError("Qualification source receipt differs from the exact nonempty-prefix probe")
        if horizon != q["horizon_external_only"] or baseline.summary["source_panel"] != [2] or baseline.summary["eligible_targets"] != [0, 1]:
            raise ValueError("Qualification source panel differs")
    else:
        expected = plan["evaluation"]
        if horizon != expected["horizon"] or baseline.summary["config"]["repeats"] != expected["repeats"]:
            raise ValueError("Full source horizon/repeats differ from declared plan")
        if baseline.summary["source_panel"] not in expected["source_shards"]:
            raise ValueError("Imported source partition is undeclared")
    settings = dict(plan["fit"])
    if phase == "preflight":
        settings.update({key: plan["qualification"][key] for key in ("seed", "updates", "batch_size", "learning_rate")})
        settings["horizon"] = plan["qualification"]["horizon_external_only"]
    for key in ("seed", "updates", "batch_size", "horizon", "learning_rate", "target_update_period", "holdout_fraction"):
        if metadata["fit"]["config"][key] != settings[key]:
            raise ValueError(f"Candidate fit differs from declared budget: {key}")
    return phase


def candidate_hashes(actor, adapter):
    return dict(actor=module_hash(actor), graph=module_hash(actor.core.policy_graph),
                readout=module_hash(adapter), action_head=module_hash(adapter.action_parameterization))


def validate_effective_plan_binding(metadata, plan, validation):
    if validation.get("effective_plan_sha256") != _canonical_hash(plan):
        raise ValueError("Effective plan fingerprint differs from validated settings")
    for key in ("effective_plan_sha256", "donor_study_sha256"):
        if metadata.get(key) != validation.get(key) or not metadata.get(key):
            raise ValueError(f"Candidate and evaluation plan binding differ: {key}")


class ReusedControlEvaluator:
    """Use only retained starts; execute_trial supplies unchanged native hooks."""
    def __init__(self, adapter, baseline, panel, *, journal=None):
        self.adapter, self.baseline = adapter, baseline
        self.blocks, self.starts, self.start_receipt = certify_retained_starts(baseline, panel)
        self.journal = journal

    def __call__(self, api):
        model, baseline, actor = self.adapter, self.baseline, api["actor"]
        summary = baseline.summary
        horizon, targets = summary["config"]["horizon"], summary["eligible_targets"]
        if model.training or actor.training or model.horizon_scale != horizon:
            raise ValueError("Frozen actors and identical external horizon are required")
        before = candidate_hashes(actor, model)
        old_frozen = summary["frozen_sha256_before"]
        if any(before[key] != old_frozen[key] for key in ("actor", "graph")):
            raise ValueError("Loaded source actor/graph differs from imported control")
        rows = []
        try:
            with torch.no_grad():
                for key, block in sorted(self.blocks.items()):
                    source, repeat = key
                    block_rows = []
                    if not block["issued"]:
                        for command in targets:
                            for target in targets:
                                block_rows.append(dict(policy="first_hit", source=source, repeat=repeat, command=command,
                                    target=target, commanded=command == target, issued=False, missing=True,
                                    missing_reason=block["missing_reason"], initially_at_target=target == source,
                                    initially_at_command=command == source, hit=False, censored=False))
                    else:
                        start = self.starts[key]
                        required = len(targets) * (len(start["prefix"]) + horizon)
                        if api["decisions"]() + required > api["decision_cap"]:
                            raise RuntimeError("Candidate budget cannot cover an originally issued complete command block")
                        original_signature = api["signature"]
                        expected = signature_hash(start["signature"])
                        def checked_signature(obs, state):
                            actual = original_signature(obs, state)
                            if signature_hash(actual) != expected:
                                raise RuntimeError("Reused physical observation/full recurrent signature differs exactly")
                            return actual
                        trial_api = dict(api, signature=checked_signature)
                        for command in targets:
                            block_rows.extend(execute_trial(trial_api, model, policy="first_hit", start=start,
                                                            command=command, targets=targets, horizon=horizon))
                    if self.journal is not None:
                        self.journal.append(key, block_rows)
                        rows.extend({k:v for k,v in row.items() if k in LIGHT_FIELDS} for row in block_rows)
                    else:
                        rows.extend(block_rows)
        finally:
            api["close"]()
        after = candidate_hashes(actor, model)
        if before != after:
            raise RuntimeError("Candidate evaluation mutated actor, graph, readout, or original action head")
        comparison_config = dict(summary["config"], discovery_episodes=0)
        return rows, dict(schema=SCHEMA, config=comparison_config, source_panel=summary["source_panel"],
            eligible_targets=targets, observed_targets=summary["observed_targets"],
            declared_dg_vocabulary=summary["declared_dg_vocabulary"], unobserved_goal_ids=summary["unobserved_goal_ids"],
            primary_detector=summary["primary_detector"], secondary_detector=summary["secondary_detector"],
            command_protocol=summary["command_protocol"], terminal_protocol=summary["terminal_protocol"],
            action_protocol="Imported original source raw argmax versus newly evaluated stationary raw-head argmax",
            decisions=api["decisions"](), discovery_decisions=0, source_control_rerun=False,
            frozen_sha256_before=before, frozen_sha256_after=after,
            policies_frozen=True, graph_frozen=True, readout_frozen=True, action_head_frozen=True,
            matched_start_signatures=self.start_receipt,
            outcomes=summarize_rows(rows, goal_vocabulary=summary["declared_dg_vocabulary"],
                declared_starts=list(self.blocks), observed_targets=summary["observed_targets"]))


def _compare_candidate_rows(baseline, candidate):
    original = {tuple(r[k] for k in ("source", "repeat", "command", "target")): r
                for r in baseline.rows if r["policy"] == "source"}
    for row in candidate:
        key = tuple(row[k] for k in ("source", "repeat", "command", "target"))
        source = original.pop(key, None)
        if source is None or row["issued"] != source["issued"] or (not row["issued"] and row["missing_reason"] != source["missing_reason"]):
            raise ValueError("Candidate changed an imported source cell or missing block")
    if original:
        raise ValueError("Candidate omitted imported source cells")


def validate_start_receipt(blocks, receipt):
    """Validate portable full-state signature receipts without unpickling starts."""
    indexed = {}
    for entry in receipt:
        key = (entry["source"], entry["repeat"])
        if key in indexed:
            raise ValueError("Duplicate retained-start signature receipt")
        digest = entry.get("full_signature_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid full-state signature checksum")
        indexed[key] = entry
    for key, block in blocks.items():
        if block["issued"]:
            entry = indexed.get(key)
            if (entry is None or entry["seed"] != block["seed"]
                    or entry["prefix_sha256"] != _canonical_hash(block["prefix"])
                    or entry["prefix_length"] != len(block["prefix"])):
                raise ValueError("Missing or mismatched full-state signature for an issued original block")


def read_reused_shard(directory, *, allow_preflight=False):
    """Model-free separate-cohort validation before canonical metric reduction."""
    directory = Path(directory).resolve(strict=True)
    summary, manifest = _read_json(directory / "summary.json"), _read_json(directory / "pre_run_manifest.json")
    if summary.get("schema") != SCHEMA or manifest.get("schema") != PRERUN:
        raise ValueError("Missing explicit reused-control cohort contract")
    baseline = read_shard(manifest["imported_control"]["directory"], allow_preflight=allow_preflight)
    if baseline.summary.get("schema") != "intrmotiv/matched-first-hit-evaluation/v2":
        raise ValueError("Imported control must be an original completed native shard")
    if baseline.receipt != manifest["imported_control"]:
        raise ValueError("Imported source receipt changed after candidate preparation")
    if baseline.identity != manifest["imported_control_identity"]:
        raise ValueError("Imported source/config/engine identity differs from declared cohort")
    plan = manifest["analysis_plan"]
    if _canonical_hash(plan) != manifest["effective_plan_sha256"]:
        raise ValueError("Effective reused-control plan changed")
    validate_effective_plan_binding(manifest["fitted_metadata"], plan, manifest["plan_validation"])
    validate_candidate_binding(baseline, manifest["fitted_metadata"], plan)
    expected_config = dict(baseline.summary["config"], discovery_episodes=0)
    if summary["config"] != expected_config or manifest["evaluation_config"] != expected_config:
        raise ValueError("Candidate comparison configuration changed")
    for key in ("source_panel", "eligible_targets", "observed_targets", "declared_dg_vocabulary", "unobserved_goal_ids",
                "primary_detector", "command_protocol", "terminal_protocol"):
        if summary[key] != baseline.summary[key]:
            raise ValueError(f"Candidate changed imported comparison contract: {key}")
    for key in ("policies_frozen", "graph_frozen", "readout_frozen", "action_head_frozen", "checkpoint_unchanged", "source_configuration_unchanged"):
        if summary.get(key) is not True:
            raise ValueError(f"Candidate lacks frozen gate: {key}")
    if summary.get("source_control_rerun") is not False or summary.get("discovery_decisions") != 0:
        raise ValueError("Reused-control comparison must not rerun source or discovery")
    frozen = summary["frozen_sha256_before"]
    if set(frozen) != {"actor", "graph", "readout", "action_head"} or frozen != summary["frozen_sha256_after"]:
        raise ValueError("Candidate frozen readout/head identity changed")
    if any(frozen[k] != baseline.summary["frozen_sha256_before"][k] for k in ("actor", "graph")):
        raise ValueError("Candidate source actor differs from imported source")
    if summary["matched_start_signatures"] != manifest["matched_start_signatures"]:
        raise ValueError("Retained-start signatures changed during execution")
    if summary["eligibility"]["layout"] != manifest["fitted_metadata"]["layout"]:
        raise ValueError("Executed core layout differs from fitted candidate")
    if (summary["restored_readout_sha256"] != manifest["fitted_metadata"]["fit"]["final_readout_sha256"]
            or summary.get("fitted_checkpoint_unchanged") is not True):
        raise ValueError("Executed candidate differs from the fitted final checkpoint")
    if summary["source_audit"].get("archive_valid") is not True or summary["source_audit"].get("current_sources_unchanged") is not True:
        raise ValueError("Candidate source integrity gate failed")
    sources = _source_identity(directory, manifest)
    expected_engine = {key:sha for key,sha in baseline.identity["sources"].items() if _engine_file(key)}
    if manifest["engine_identity"] != expected_engine or any(sources.get(key) != sha for key,sha in expected_engine.items()):
        raise ValueError("Candidate archived engine bytes differ from imported source semantics")
    validate_complete_journal(directory,
        [(s,r) for s in sorted(summary["source_panel"]) for r in range(summary["config"]["repeats"])],
        len(summary["eligible_targets"]) ** 2, summary["trials_sha256"])
    rows = _read_trials(directory / "candidate_trials.jsonl", summary, policies=("first_hit",))
    _compare_candidate_rows(baseline, rows)
    # The heavy candidate prefix is checked against the imported source prefix
    # independently of the target scoring, without loading matched_starts.pt.
    blocks = baseline_source_blocks(baseline)
    validate_start_receipt(blocks, manifest["matched_start_signatures"])
    with (directory / "candidate_trials.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["issued"]:
                block = blocks[row["source"], row["repeat"]]
                if row["prefix_seed"] != block["seed"] or row["prefix_actions"] != block["prefix"]:
                    raise ValueError("Candidate and imported control physical prefixes differ")
    imported = [row for row in baseline.rows if row["policy"] == "source"]
    identity = dict(baseline.identity)
    identity.update(cohort_contract="imported_source_and_new_stationary_candidate", source_root=manifest["source_root"],
        sources=sources, original_control_sources=baseline.identity["sources"],
        config={k:v for k,v in summary["config"].items() if k != "focus_sources"},
        fitted_checkpoint_sha256=manifest["fitted_metadata"]["checkpoint_sha256"],
        fitted_metadata_sha256=_canonical_hash(manifest["fitted_metadata"]), frozen_state=frozen,
        action_protocol=summary["action_protocol"], effective_plan_sha256=manifest["effective_plan_sha256"])
    receipt = dict(directory=str(directory), schema=SCHEMA, summary_sha256=file_hash(directory / "summary.json"),
        pre_run_manifest_sha256=file_hash(directory / "pre_run_manifest.json"),
        trials_sha256=summary["trials_sha256"], candidate_trials_file="candidate_trials.jsonl",
        imported_control=baseline.receipt, matched_starts_sha256=baseline.receipt["matched_starts_sha256"],
        source_ids=summary["source_panel"], trial_rows=len(rows)+len(imported),
        archived_sources_manifest_sha256=file_hash(directory / "source_snapshot/manifest.json"))
    receipt["trial_sources"] = trial_source_records(receipt)
    return Shard(directory, summary, manifest, identity, imported+rows, receipt)


def verify_engine_sources(baseline, source_root, callback_text):
    """Require unchanged canonical native code, action/prefix logic and detector."""
    root = Path(source_root)
    checked = {}
    for relative, expected in baseline.identity["sources"].items():
        if relative == "canonical_callback_overlay":
            actual = hashlib.sha256(callback_text.encode()).hexdigest()
        elif _engine_file(relative):
            actual = file_hash(root / relative)
        else:
            continue  # The old finite-head loader is intentionally not reused.
        if actual != expected:
            raise ValueError(f"Imported control engine semantics changed: {relative}")
        checked[relative] = actual
    required = {"canonical_callback_overlay", "hpc_runs/first_hit_control/dmlab_evaluate.py",
                "sf_working_directories/IntrMotiv/evaluation/target_control_interventions.py",
                "sf_working_directories/IntrMotiv/evaluation/place_fields.py"}
    if not required <= set(checked):
        raise ValueError("Imported source archive lacks required native engine/prefix evidence")
    return checked


def run_on_nemo(args):
    """Only an allocated NEMO job may instantiate engines; never local smoke."""
    import importlib.util
    import os
    import sys

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Candidate native evaluation requires an allocated NEMO Slurm job")
    workspace = Path(os.environ.get("INTRMOTIV_WORKSPACE_ROOT", "/work/classic/fr_xl1014-corridor-geometry")).resolve()
    if not str(workspace).startswith("/work/"):
        raise ValueError("Native outputs must be in the allocated /work workspace")
    source_root, output = args.source_root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(workspace):
        raise ValueError("Require a new candidate output in the allocated workspace")
    from .reused_plan import load_plan
    plan, validation = load_plan(args.plan, source_release=str(source_root))
    if not output.is_relative_to(Path(plan["runtime"]["output_root"])):
        raise ValueError("Candidate output is outside this declared study")
    if args.checkpoint_sha256 != plan["donor"]["checkpoint_sha256"] or file_hash(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Source donor checkpoint differs from declaration")
    baseline = read_shard(args.baseline, allow_preflight=True)
    if baseline.summary["schema"] != "intrmotiv/matched-first-hit-evaluation/v2":
        raise ValueError("Source reuse requires a completed original two-policy shard")
    expected_root = (Path(plan["qualification"]["source_evaluation"]) if baseline.summary["config"]["phase"] == "preflight"
                     else Path(plan["parent_analysis"]["output"]))
    if not args.baseline.resolve().is_relative_to(expected_root):
        raise ValueError("Imported source is outside the declared original analysis")
    sys.path.insert(0, str(source_root))
    from .stationary_replay_cli import load_fitted_stationary
    from .replay_fit import state_digest
    from .eligibility import certify_slice_only_relabel

    adapter, metadata = load_fitted_stationary(args.fitted, source_root, device="cpu")
    validate_effective_plan_binding(metadata, plan, validation)
    phase = validate_candidate_binding(baseline, metadata, plan)
    expected_cap = plan["qualification"]["decision_cap"] if phase == "preflight" else plan["evaluation"]["decision_cap_per_shard"]
    if args.decision_cap != expected_cap:
        raise ValueError("Candidate decision guard differs from declared plan")
    if metadata["analysis_plan"]["sha256"] != file_hash(args.plan):
        raise ValueError("Candidate was fitted under a different declared analysis plan")
    adapter.eval().requires_grad_(False)
    # The original starts contain tensors only. Deserialize only here, after
    # native/job guards and original completion/hash validation; never in reducer.
    panel = torch.load(baseline.directory / "matched_starts.pt", map_location="cpu", weights_only=True)
    callback = ReusedControlEvaluator(adapter, baseline, panel)
    if baseline.summary["config"]["phase"] == "preflight":
        issued = [r for r in callback.start_receipt if (r["source"], r["repeat"]) in callback.blocks
                  and callback.blocks[r["source"], r["repeat"]]["issued"]]
        if len(issued) != 1 or issued[0]["seed"] != 31003 or issued[0]["prefix_length"] != 20:
            raise ValueError("Qualification must replay the exact twenty-action retained prefix")
    canonical = source_root / "sf_working_directories/IntrMotiv/evaluation/target_control_interventions.py"
    patched = install_callback(canonical.read_text())
    engine_identity = verify_engine_sources(baseline, source_root, patched)
    source_config = args.run_dir / "config.json"
    original = _read_json(source_config)
    overrides = dict(dmlab_use_level_cache=False, dmlab_level_cache_path=str(output / "cache"),
                     dmlab_runfiles_path=str(args.runfiles.resolve(strict=True)), online_spatial_telemetry=False,
                     with_wandb=False, transfer_freeze_dg=True)
    config_sha = verify_source_config(source_config, metadata, overrides)
    output.mkdir(parents=True)
    overlay = output / "canonical_target_control_with_callback.py"
    overlay.write_text(patched)
    local = Path(__file__).parent
    source_paths = [local / name for name in ("reused_control.py", "reused_plan.py", "aggregate_evaluation.py",
        "dmlab_evaluate.py", "detectors.py", "runtime_adapter.py", "stationary_replay_cli.py", "stationary_runtime_adapter.py",
        "replay_cli.py", "replay_fit.py", "eligibility.py", "provenance.py", "targets.py", "block_journal.py")]
    source_paths += [args.plan.resolve(), overlay]
    source_paths += [source_root / path for path in engine_identity if path != "canonical_callback_overlay"]
    source_paths += [source_root / "sample_factory/model/action_parameterization.py"]
    snapshot = snapshot_sources(list(dict.fromkeys(source_paths)), output / "source_snapshot")
    loader = output / "loader"
    loader.mkdir()
    (loader / "config.json").write_text(json.dumps(dict(original, **overrides), indent=2) + "\n")
    manifest = dict(schema=PRERUN, source_root=str(source_root), source_snapshot=snapshot,
        imported_control=baseline.receipt, imported_control_identity=baseline.identity,
        fitted_directory=str(args.fitted.resolve()), fitted_metadata=metadata,
        original_config_sha256=config_sha, source_checkpoint_sha256=args.checkpoint_sha256,
        evaluation_overrides=overrides, engine_identity=engine_identity, analysis_plan=plan,
        plan_validation=validation, declared_plan_sha256=file_hash(args.plan), effective_plan_sha256=_canonical_hash(plan),
        evaluation_config=dict(baseline.summary["config"], discovery_episodes=0), decision_cap=args.decision_cap,
        matched_start_signatures=callback.start_receipt,
        cohorts=dict(source="Imported original source-policy rows only, by immutable reference",
                     first_hit="New stationary candidate-only trajectories; no discovery or original control rerun"))
    (output / "pre_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    callback.journal = CompleteBlockJournal(output, sorted(callback.blocks),
        [(command,target) for command in baseline.summary["eligible_targets"] for target in baseline.summary["eligible_targets"]])
    import deepmind_lab
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    spec = importlib.util.spec_from_file_location("intrmotiv_reused_control_canonical", overlay)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg, env, info, actor, checkpoint, device = module.load_policy_env(loader, args.decision_cap, True, 0, args.checkpoint)
    actor.eval().requires_grad_(False)
    try:
        certificate = certify_slice_only_relabel(actor.core, actor.decoder, source_root=source_root, input_kind="canonical_core_output")
        if certificate["layout"] != asdict(adapter.layout) or actor.action_space.n != adapter.action_count or env.num_agents != 1:
            raise ValueError("Native source core/action interface differs from stationary readout")
        rows, summary = module.run_landmark_matched_interventions(cfg, env, info, actor, checkpoint, device,
            args.decision_cap, deterministic=True, first_hit_evaluator=callback)
    except BaseException:
        env.close()
        raise
    summary.update(eligibility=certificate, source_audit=audit_source_snapshots(output / "source_snapshot"),
        checkpoint_unchanged=file_hash(args.checkpoint) == args.checkpoint_sha256,
        source_configuration_unchanged=verify_source_config(source_config, metadata, overrides) == config_sha,
        fitted_checkpoint_unchanged=file_hash(args.fitted / "adapter.pt") == metadata["checkpoint_sha256"],
        restored_readout_sha256=state_digest(adapter))
    if (not summary["source_audit"]["archive_valid"] or not summary["source_audit"]["current_sources_unchanged"]
            or not summary["checkpoint_unchanged"] or not summary["fitted_checkpoint_unchanged"]):
        raise RuntimeError("Candidate input/source bytes changed during evaluation")
    if read_shard(args.baseline, allow_preflight=True).receipt != baseline.receipt:
        raise RuntimeError("Imported control changed during candidate execution")
    callback.journal.finalize()
    summary["trials_sha256"] = file_hash(output / "candidate_trials.jsonl")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    # Model-free validation of the completed, explicitly separate cohorts.
    read_reused_shard(output, allow_preflight=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "run-dir", "checkpoint", "fitted", "baseline", "output", "runfiles", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--decision-cap", type=int, default=350000)
    result = run_on_nemo(parser.parse_args())
    print(json.dumps({"schema": result["schema"], "decisions": result["decisions"], "source_control_rerun": False}))


if __name__ == "__main__":
    main()
