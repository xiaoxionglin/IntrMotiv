"""Validate and render a readout comparison reusing factual replay and controls.

The parent plan and canonical StudySpec remain the donor authorities. This
adapter describes only new readout fitting and candidate paths. It never
submits jobs, imports an environment, or calls missing controls fresh data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

from . import analysis_plan as parent_api
from .replay_fit import FitConfig


SCHEMA = "intrmotiv/reused-replay-control-plan/v1"


def _read_pinned(root, relative, expected):
    path = parent_api._local(root, relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Pinned local artifact changed: {relative}")
    return json.loads(path.read_text())


def validate_plan(plan, repository_root):
    """Check scientific identities and bounds without claiming remote health."""
    root = Path(repository_root)
    if plan.get("schema") != SCHEMA or plan.get("analysis_version") != "1.0.0":
        raise ValueError("Unsupported reused-control plan")
    reference = plan["parent_analysis"]
    parent = _read_pinned(root, reference["path"], reference["sha256"])
    _, parent_validation = parent_api.load_plan(
        root / reference["path"], repository_root=root, source_release=reference["source_release"])
    if parent_validation["effective_plan_sha256"] != reference["effective_plan_sha256"]:
        raise ValueError("Actual parent release differs from the completed main plan")
    release = _read_pinned(root, reference["release_record"]["path"], reference["release_record"]["sha256"])
    if release["source_release"] != reference["source_release"] or reference["output"] != parent["runtime"]["output_root"]:
        raise ValueError("Parent source or output identity differs")
    for key in ("workflow_version", "donor_study", "donor", "source_interface"):
        if plan[key] != parent[key]:
            raise ValueError(f"Reused comparison changes parent {key}")
    if plan.get("contract", {}).get("kind") != "artifact_dependency_graph":
        raise ValueError("Reused artifacts must not be represented as fresh pretraining runs")

    runtime, fit, evaluate = (plan[key] for key in ("runtime", "fit", "evaluation"))
    workspace = parent_api._absolute(runtime["workspace_root"], "workspace")
    if runtime["workspace_root"] != parent["runtime"]["workspace_root"]:
        raise ValueError("Allocated workspace differs from parent")
    for key in ("source_release", "output_root", "torch_cache"):
        parent_api._inside(runtime[key], workspace, key)
    for path in runtime["dependency_roots"]:
        parent_api._inside(path, workspace, "runtime dependency")
    parent_api._absolute(runtime["python"], "read-only Python executable")
    source, output = map(PurePosixPath, (runtime["source_release"], runtime["output_root"]))
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and bulk output paths overlap")
    if str(source) == reference["source_release"] or str(output) == reference["output"]:
        raise ValueError("A new candidate must preserve the original source and output")

    replay, qualification = plan["replay"], plan["qualification"]
    if replay["directory"] != reference["output"] + "/replay" or replay["new_collection_decisions"] != 0:
        raise ValueError("This comparison reuses the completed main replay without collection")
    if replay["qualification_replay_not_pooled"] is not True or qualification["new_collection_decisions"] != 0:
        raise ValueError("Qualification data must remain separate and already collected")
    for key in ("dataset_sha256", "decoder_state_sha256", "metadata_sha256"):
        parent_api._sha(replay[key], key)
    for key in ("replay", "source_evaluation"):
        parent_api._inside(qualification[key], workspace, "qualification " + key)
    for key in ("replay_dataset_sha256", "replay_metadata_sha256", "source_trials_sha256",
                "source_summary_sha256", "matched_starts_sha256"):
        parent_api._sha(qualification[key], key)
    if qualification["replay"] == replay["directory"]:
        raise ValueError("Integration replay and scientific replay must remain distinct")
    config_keys = ("seed", "updates", "batch_size", "horizon", "learning_rate",
                   "target_update_period", "holdout_fraction", "threads", "device")
    if any(fit[key] != parent["fit"][key] for key in config_keys):
        raise ValueError("Full fitting budgets must match the declared parent comparison")
    fixed = FitConfig(**{key: fit[key] for key in config_keys if key not in ("threads", "device")})
    if plan["objective"]["gamma"] != .99 or plan["model"]["added_inference_parameters"] != 0:
        raise ValueError("Only the declared projected original-head design is qualified")
    if plan["model"]["trainable_parameters"] != 166405:
        raise ValueError("Original native readout parameter count differs")
    if (qualification["updates"], qualification["batch_size"], qualification["horizon_external_only"],
            qualification["learning_rate"], qualification["seed"], qualification["decision_cap"]) != (100, 64, 64, .0003, 99, 4096):
        raise ValueError("Integration qualification settings differ from declaration")
    for key in ("source_shards", "repeats", "prefix_cap", "decision_cap_per_shard"):
        if evaluate[key] != parent["evaluation"][key]:
            raise ValueError(f"Reused query panel changes {key}")
    if (evaluate["horizon"] != fixed.horizon or evaluate["discovery_episodes"] != 0
            or evaluate["policy_count"] != 1 or evaluate["analysis_policy_count"] != 2
            or evaluate["independent_jobs"] is not True):
        raise ValueError("Execute only one new candidate on the existing horizon and starts")

    evidence = {Path(path).parent.name: _read_pinned(root, path, sha)
                for path, sha in plan["local_evidence"].items() if path.endswith("results.json")}
    # The theory/protocol files also remain byte pinned, although not all are JSON.
    for relative, expected in plan["local_evidence"].items():
        if hashlib.sha256(parent_api._local(root, relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Local design evidence changed")
    if evidence["projected_grid_r1"]["all_pair_gate_by_condition"]["projected"] is not True:
        raise ValueError("The separately declared grid gate did not pass")
    cycles = evidence["stationary_cycle_audit_r1"]
    selected = [row for row in cycles["runs"] if row["condition"] == "projected_regression_r1"]
    if len(selected) != 6 or sum(row["noninitial"]["queries"] for row in selected) != 138:
        raise ValueError("The complete fixed-policy reaching audit is required")
    if any(row["noninitial"]["hit"] != row["noninitial"]["queries"]
           or not row["model_unchanged"] or not row["original_h6_prefixes_identical"] for row in selected):
        raise ValueError("The reaching audit has unresolved, failed, or changed policies")
    if evidence["projected_regression_r1"]["all_six_gates_passed"] is not False:
        raise ValueError("Preserve the original failed H6 result rather than replacing its gate")
    goals = plan["source_interface"]["goal_count"]
    costs = [len(shard) * evaluate["repeats"] * goals * (evaluate["prefix_cap"] + fixed.horizon)
             for shard in evaluate["source_shards"]]
    if max(costs) > evaluate["decision_cap_per_shard"]:
        raise ValueError("Candidate budget cannot cover all declared paths")
    return dict(donor_run_spec=parent_validation["donor_run_spec"],
        donor_study_sha256=parent_validation["donor_study_sha256"],
        canonical_workflow_version=parent_validation["canonical_workflow_version"],
        parent_effective_plan_sha256=reference["effective_plan_sha256"],
        budgets=dict(new_collection_decisions=0, new_discovery_decisions=0,
            full_fit_transition_draws=fixed.updates * fixed.batch_size,
            maximum_full_fit_goal_labels=fixed.updates * fixed.batch_size * goals,
            new_candidate_trials=goals ** 2 * evaluate["repeats"],
            new_noninitial_trials=goals * (goals-1) * evaluate["repeats"],
            worst_case_candidate_decisions_per_shard=costs),
        original_h6_gate_passed=evidence["projected_regression_r1"]["all_six_gates_passed"],
        preserved_failed_gate="Original H6 regression; new advancement decision uses complete fixed-policy reaching and separate grid test",
        execution_gates="Static validation only. Remote bytes, completed baseline artifacts, allocated qualification, and frozen execution remain required.")


def load_plan(path, *, repository_root=None, source_release=None):
    path = Path(path).resolve()
    root = Path(repository_root) if repository_root else Path(__file__).resolve().parents[2]
    raw = json.loads(path.read_text())
    effective = copy.deepcopy(raw)
    if source_release is not None:
        effective["runtime"]["source_release"] = str(source_release)
    validation = validate_plan(effective, root)
    return effective, dict(validation, schema=SCHEMA, source_plan=str(path),
        source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        declared_plan_sha256=parent_api.fingerprint(raw), effective_plan_sha256=parent_api.fingerprint(effective))


def render_commands(plan, validation):
    """Print exact new-stage arguments and completed-artifact dependencies."""
    runtime, donor, fit = (plan[key] for key in ("runtime", "donor", "fit"))
    output = PurePosixPath(runtime["output_root"])
    release, python = runtime["source_release"], runtime["python"]
    plan_path = str(PurePosixPath(release) / "hpc_runs/studies/stationary_projected_c05_s99_20261127.analysis.json")
    stages = []
    def stage(name, module, argv, dependencies, destination, native=False):
        argv = [python, "-m", "hpc_runs.first_hit_control." + module, *map(str, argv)]
        stages.append(dict(stage_id=name, argv=argv, shell_display=shlex.join(argv),
            depends_on=dependencies, output=str(destination), native_environment=native))
    # The strict replay/config contract includes the original runfiles path.
    # Reusing replay therefore also reuses its phase-specific native assets;
    # rebuilding equivalent files under the new output would violate that
    # contract before any environment is created.
    qualification_assets = PurePosixPath(plan["qualification"]["replay"]).parent / "runfiles"
    main_assets = PurePosixPath(plan["replay"]["directory"]).parent / "runfiles"
    for name, replay, settings, horizon, dependencies in (
        ("qualification_fit", plan["qualification"]["replay"], plan["qualification"], 64, ["certified_qualification_replay"]),
        ("fit", plan["replay"]["directory"], fit, fit["horizon"], ["reviewed_native_qualification", "certified_main_replay"])):
        stage(name, "stationary_replay_cli", ["--source-root", release, "--replay", replay,
            "--output", output / name, "--seed", settings["seed"], "--updates", settings["updates"],
            "--batch-size", settings["batch_size"], "--horizon", horizon, "--learning-rate", settings["learning_rate"],
            "--gamma", plan["objective"]["gamma"], "--threads", fit["threads"], "--device", fit["device"],
            "--analysis-plan", plan_path],
            dependencies, output / name)
    common = ["--source-root", release, "--run-dir", donor["run_dir"], "--checkpoint", donor["checkpoint"],
              "--checkpoint-sha256", donor["checkpoint_sha256"], "--plan", plan_path]
    stage("qualification_evaluate", "reused_control", common + ["--fitted", output / "qualification_fit",
        "--baseline", plan["qualification"]["source_evaluation"], "--output", output / "qualification_evaluation",
        "--runfiles", qualification_assets, "--decision-cap", plan["qualification"]["decision_cap"]],
        ["certified_qualification_assets", "qualification_fit", "completed_nonempty_prefix_baseline"],
        output / "qualification_evaluation", True)
    for index, sources in enumerate(plan["evaluation"]["source_shards"]):
        name = f"evaluation_sources_{index:02d}"
        stage(name, "reused_control", common + ["--fitted", output / "fit", "--baseline",
            PurePosixPath(plan["parent_analysis"]["output"]) / name, "--output", output / name,
            "--runfiles", main_assets, "--decision-cap", plan["evaluation"]["decision_cap_per_shard"]],
            ["certified_main_assets", "fit", f"completed_main_shard_{index:02d}"], output / name, True)
    return dict(schema="intrmotiv/reused-control-rendered-plan/v1", print_only=True,
        jobs_submitted=False, validation=validation, working_directory=release, stages=stages,
        launcher_requirements=["Ordinary allocated Slurm jobs; all caches/logs/temp/data in declared workspace",
            "Review exact immutable source release and print-only arguments before submission",
            "Candidate evaluations depend on successful fit and completed corresponding original shard",
            "Reuse each replay phase's original runfiles path; the native configuration guard verifies it against fitted metadata",
            "No original-policy execution, discovery, collection, or unreviewed automatic retries"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "render"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--source-release", type=Path)
    args = parser.parse_args()
    plan, validation = load_plan(args.plan, repository_root=args.repository_root, source_release=args.source_release)
    print(json.dumps(validation if args.command == "validate" else render_commands(plan, validation), indent=2))
