"""Validate and print a dependent frozen-replay analysis without submitting it.

The canonical StudySpec remains the donor identity authority. Its Cartesian
Sample Factory RunSpecs describe independent training runs; collection ->
exposure -> copied-readout fitting -> shared-artifact evaluation is a different
contract. This adapter adds only that artifact DAG and renders existing CLIs.
It does not fabricate training RunSpecs, execute commands, or import DMLab.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

from hpc_runs.intrmotiv_study import WORKFLOW_VERSION, load_study
from .replay_fit import FitConfig


SCHEMA = "intrmotiv/frozen-readout-analysis-plan/v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a SHA-256")


def _absolute(value, label):
    if not isinstance(value, str) or not value.startswith("/") or ".." in PurePosixPath(value).parts:
        raise ValueError(f"{label} must be an absolute path without parent traversal")
    return PurePosixPath(value)


def _inside(value, workspace, label):
    path = _absolute(value, label)
    if path == workspace or not path.is_relative_to(workspace):
        raise ValueError(f"{label} must be inside the allocated workspace")
    return path


def _local(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Repository references must be relative without parent traversal")
    return root / path


def validate_plan(plan, repository_root):
    """Validate static identities and bounded budgets; no remote existence claim."""
    root = Path(repository_root)
    if plan.get("schema") != SCHEMA or plan.get("analysis_version") != "1.0.0":
        raise ValueError("Unsupported frozen-readout analysis contract")
    if plan.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError("Analysis workflow version differs from the canonical local package")
    if plan.get("contract", {}).get("kind") != "artifact_dependency_graph":
        raise ValueError("This plan is an artifact dependency graph, not independent training runs")
    reference, donor = plan["donor_study"], plan["donor"]
    study = load_study(_local(root, reference["path"]))
    if study.fingerprint != reference["sha256"]:
        raise ValueError("Donor StudySpec fingerprint differs from the pinned plan")
    if study.declared_workflow_version != reference["declared_workflow_version"]:
        raise ValueError("Historical donor workflow version differs")
    if type(donor["seed"]) is not int:
        raise ValueError("Donor seed must be an integer")
    matches = [run for run in study.expand_runs() if run.condition == donor["condition"] and run.seed == donor["seed"]]
    if len(matches) != 1:
        raise ValueError("Donor selector must identify exactly one canonical RunSpec")
    run = matches[0]
    runtime, interface, collect, fit, evaluate = (plan[key] for key in
                                                ("runtime", "source_interface", "collection", "fit", "evaluation"))
    workspace = _absolute(runtime["workspace_root"], "workspace_root")
    if not workspace.is_relative_to(PurePosixPath("/work")) or str(workspace) != study.workspace_root:
        raise ValueError("Use the donor's active allocated /work workspace")
    for name in ("source_release", "output_root", "torch_cache"):
        _inside(runtime[name], workspace, name)
    dependencies = runtime["dependency_roots"]
    if not dependencies or len(set(dependencies)) != len(dependencies):
        raise ValueError("Declare unique runtime dependency roots")
    for path in dependencies:
        _inside(path, workspace, "runtime dependency")
    _absolute(runtime["python"], "read-only Python executable")
    output, source = PurePosixPath(runtime["output_root"]), PurePosixPath(runtime["source_release"])
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Bulk outputs and immutable source release must be separate")
    run_dir = _inside(donor["run_dir"], workspace, "donor run")
    expected_run = PurePosixPath(study.output_root) / (run.name + "_") / ("00_" + run.name)
    if run_dir != expected_run:
        raise ValueError("Donor path differs from the selected canonical run identity")
    checkpoint = _inside(donor["checkpoint"], workspace, "donor checkpoint")
    if not checkpoint.is_relative_to(run_dir / "checkpoint_p0"):
        raise ValueError("Checkpoint must belong to the selected donor run")
    for key in ("checkpoint_sha256", "config_sha256"):
        _sha(donor[key], key)
    receipt_ref = plan["qualification_protocol"]
    receipt_path = _local(root, receipt_ref["path"])
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != receipt_ref["sha256"]:
        raise ValueError("Qualification protocol bytes differ from the reviewed receipt")
    receipt = json.loads(receipt_path.read_text())
    if any(str(donor[key]) != str(receipt["donor"][key]) for key in
           ("condition", "seed", "run_dir", "checkpoint", "checkpoint_sha256", "config_sha256")):
        raise ValueError("Donor checkpoint/config selection differs from qualification")
    goals = _positive(interface["goal_count"], "goal_count")
    actions = _positive(interface["action_count"], "action_count")
    repeat = _positive(interface["action_repeat"], "action_repeat")
    cfg = receipt["donor"]["config"]
    if (goals != cfg["Hippo_n_feature"] or repeat != cfg["env_frameskip"]
            or interface["environment"] != cfg["env"] or interface["primary_detector"] != "winner"
            or interface["secondary_detector"] != "exclusive"):
        raise ValueError("Declared source detector/environment/timing differs from qualification")
    if not cfg["dmlab_reduced_action_set"] or actions != 5:
        raise ValueError("This qualified donor uses the unchanged five-action interface")
    _positive(collect["decisions"], "collection decisions")
    if type(collect["seed"]) is not int or not 0 <= collect["random_action_probability"] <= 1:
        raise ValueError("Invalid collection seed or mixture probability")
    if collect["reuse_qualification_data"] is not False:
        raise ValueError("Qualification replay cannot be silently pooled into the new fixed dataset")
    fixed = FitConfig(**{key: fit[key] for key in
                         ("seed", "updates", "batch_size", "horizon", "learning_rate", "target_update_period", "holdout_fraction")})
    defaults = FitConfig()
    if (fixed.target_update_period != defaults.target_update_period
            or fixed.holdout_fraction != defaults.holdout_fraction):
        raise ValueError("Frozen fit CLI does not expose changed target period or holdout fraction")
    _positive(fit["threads"], "fit threads")
    if fit["device"] != "cpu" or type(fit["seed"]) is not int:
        raise ValueError("The reviewed fit uses a fixed integer seed and bounded CPU execution")
    if fixed.updates // fixed.target_update_period < fixed.horizon:
        raise ValueError("Declared target generations are shorter than the finite-horizon budget")
    _positive(plan["fit_diagnostics"]["samples_per_bin"], "diagnostic samples per bin")
    for name in ("repeats", "discovery_episodes", "prefix_cap", "decision_cap_per_shard"):
        _positive(evaluate[name], name)
    if type(evaluate["discovery_seed"]) is not int or evaluate["policy_count"] != 2 or evaluate["independent_jobs"] is not True:
        raise ValueError("Require fixed discovery seeds and two-policy independent source-shard jobs")
    shards = evaluate["source_shards"]
    if not isinstance(shards, list) or not shards or any(not isinstance(shard, list) or not shard for shard in shards):
        raise ValueError("Source shards must be nonempty lists")
    flattened = [goal for shard in shards for goal in shard]
    if any(type(goal) is not int for goal in flattened) or sorted(flattened) != list(range(goals)):
        raise ValueError("Source shards must partition every declared DG identity exactly once")
    discovery_cost = evaluate["discovery_episodes"] * evaluate["prefix_cap"]
    cost = [discovery_cost + len(shard) * evaluate["repeats"] * goals * evaluate["policy_count"]
            * (evaluate["prefix_cap"] + fixed.horizon) for shard in shards]
    if discovery_cost > evaluate["decision_cap_per_shard"] // 4:
        raise ValueError("Discovery exceeds the evaluator's quarter-budget reserve")
    if max(cost) > evaluate["decision_cap_per_shard"]:
        raise ValueError("Decision cap cannot cover all declared matched blocks")
    return dict(donor_run_spec=run.as_dict(), donor_study_sha256=study.fingerprint,
                canonical_workflow_version=WORKFLOW_VERSION,
                budgets=dict(collection_decisions=collect["decisions"],
                    transition_draws=fixed.updates * fixed.batch_size,
                    maximum_goal_labels=fixed.updates * fixed.batch_size * goals,
                    target_refreshes=fixed.updates // fixed.target_update_period,
                    horizon_frames=fixed.horizon * repeat,
                    discovery_decisions_per_shard=discovery_cost,
                    worst_case_decisions_per_shard=cost,
                    total_evaluation_cap=len(shards) * evaluate["decision_cap_per_shard"],
                    command_trials_per_policy=goals * goals * evaluate["repeats"],
                    noninitial_own_trials_per_policy=goals * (goals - 1) * evaluate["repeats"],
                    cross_target_rows_per_policy=goals ** 3 * evaluate["repeats"]),
                warning="Static budget and provenance validation only; remote bytes, symlinks, runtime health and capability remain execution gates")


def load_plan(path, *, repository_root=None, source_release=None):
    source = Path(path).resolve()
    root = Path(repository_root) if repository_root else Path(__file__).resolve().parents[2]
    raw = json.loads(source.read_text())
    effective = copy.deepcopy(raw)
    if source_release is not None:
        effective["runtime"]["source_release"] = str(source_release)
    validation = validate_plan(effective, root)
    return effective, dict(validation, schema=SCHEMA, source_plan=str(source),
                           declared_plan_sha256=fingerprint(raw), effective_plan_sha256=fingerprint(effective))


def render_commands(plan, validation):
    """Return argv arrays plus dependencies; never invoke a shell or scheduler."""
    runtime, donor, collect, fit, evaluate = (plan[key] for key in ("runtime", "donor", "collection", "fit", "evaluation"))
    output, python, release = PurePosixPath(runtime["output_root"]), runtime["python"], runtime["source_release"]
    paths = {name: str(output / name) for name in ("runfiles", "replay", "exposure", "fit", "fit_diagnostics")}
    stages = []

    def stage(identifier, module, arguments, dependencies, artifact, native=False):
        argv = [python, "-m", module] + [str(value) for value in arguments]
        stages.append(dict(stage_id=identifier, depends_on=dependencies, argv=argv,
                           shell_display=shlex.join(argv), output=artifact, native_environment=native))

    stage("runfiles", "hpc_runs.intrmotiv_study.runfiles", ["--source", release, "--output", paths["runfiles"]], [], paths["runfiles"])
    common = ["--source-root", release, "--run-dir", donor["run_dir"], "--checkpoint", donor["checkpoint"],
              "--checkpoint-sha256", donor["checkpoint_sha256"], "--runfiles", paths["runfiles"]]
    stage("collect", "hpc_runs.first_hit_control.collector", common + ["--config-sha256", donor["config_sha256"],
          "--output", paths["replay"], "--decisions", collect["decisions"], "--seed", collect["seed"],
          "--random-action-probability", collect["random_action_probability"]], ["runfiles"], paths["replay"], True)
    stage("support", "hpc_runs.first_hit_control.support_analysis", ["--replay", paths["replay"], "--output", paths["exposure"],
          "--split-seed", fit["seed"], "--holdout-fraction", fit["holdout_fraction"]], ["collect"], paths["exposure"])
    stage("fit", "hpc_runs.first_hit_control.replay_cli", ["--source-root", release, "--replay", paths["replay"],
          "--output", paths["fit"], "--updates", fit["updates"], "--batch-size", fit["batch_size"], "--horizon", fit["horizon"],
          "--learning-rate", fit["learning_rate"], "--seed", fit["seed"], "--threads", fit["threads"], "--device", fit["device"]],
          ["collect", "support"], paths["fit"])
    stage("fit_diagnostics", "hpc_runs.first_hit_control.fit_diagnostics", ["--replay", paths["replay"],
          "--fitted", paths["fit"], "--source-root", release, "--output", paths["fit_diagnostics"],
          "--samples-per-bin", plan["fit_diagnostics"]["samples_per_bin"], "--threads", fit["threads"]],
          ["fit"], paths["fit_diagnostics"])
    for index, sources in enumerate(evaluate["source_shards"]):
        destination = str(output / f"evaluation_sources_{index:02d}")
        stage(f"evaluate_{index:02d}", "hpc_runs.first_hit_control.dmlab_evaluate", common + ["--fitted", paths["fit"],
              "--output", destination, "--phase", "full", "--sources", *sources, "--horizon", fit["horizon"],
              "--discovery-episodes", evaluate["discovery_episodes"], "--prefix-cap", evaluate["prefix_cap"],
              "--repeats", evaluate["repeats"], "--max-sources", len(sources), "--discovery-seed", evaluate["discovery_seed"],
              "--decision-cap", evaluate["decision_cap_per_shard"]], ["fit_diagnostics", "support", "runfiles"], destination, True)
    environment = dict(INTRMOTIV_WORKSPACE_ROOT=runtime["workspace_root"], INTRMOTIV_RUNTIME_SOURCE=release,
                       PYTHONPATH=":".join([release, *runtime["dependency_roots"]]),
                       TORCH_HOME=runtime["torch_cache"], PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled")
    for variable, directory in (("XDG_CACHE_HOME", "cache"), ("MPLCONFIGDIR", "matplotlib"),
                                ("DMLAB_CACHE_DIR", "dmlab_cache"), ("WANDB_DIR", "wandb"),
                                ("WANDB_CACHE_DIR", "wandb_cache"), ("WANDB_DATA_DIR", "wandb_data")):
        environment[variable] = str(output / directory)
    return dict(schema="intrmotiv/frozen-readout-rendered-plan/v1", print_only=True,
                analysis_id=plan["analysis_id"], validation=validation, working_directory=release,
                environment=environment,
                per_job_temporary_path_template=runtime["workspace_root"] + "/tmp/first_hit_{job_id}",
                launcher_requirements=["Allocated ordinary NEMO Slurm jobs, never login-node native execution",
                                       "Set TMPDIR/TMP/TEMP from the short workspace job path; no home output/cache paths",
                                       "Review immutable source release and qualification gates before submission"],
                stages=stages)


def render_slurm(plan, validation, release_record, *, collect_fit_hours=12, evaluation_hours=12):
    """Render reviewable ordinary jobs from argv, without scheduler operations.

    Resources are launch settings, separate from the scientific plan. Walltime
    may be reduced from the reviewed twelve-hour reservation; the argv remain
    identical. The supplied release record binds the already staged runtime,
    which need not contain this local-only command renderer.
    """
    for value in (collect_fit_hours, evaluation_hours):
        if type(value) is not int or not 1 <= value <= 12:
            raise ValueError("Walltime must be an integer from one through twelve hours")
    rendered = render_commands(plan, validation)
    runtime, donor = plan["runtime"], plan["donor"]
    if plan["fit"]["threads"] > 4:
        raise ValueError("Reviewed four-CPU reservation cannot serve the requested fit threads")
    if release_record.get("source_release") != runtime["source_release"]:
        raise ValueError("Staged release record differs from the effective source release")
    for key in ("archive_sha256", "baseline_tree_sha256"):
        _sha(release_record.get(key), key)
    commit = release_record.get("candidate_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("The staged runtime requires an exact committed source identity")
    root = PurePosixPath(runtime["output_root"])
    if any(c.isspace() for c in str(root) + runtime["source_release"]):
        raise ValueError("Slurm directive paths must not contain whitespace")
    setup, evaluations = [], []
    for stage in rendered["stages"]:
        (evaluations if stage["argv"][2] == "hpc_runs.first_hit_control.dmlab_evaluate" else setup).append(stage)
    if len(setup) != 5 or len(evaluations) != len(plan["evaluation"]["source_shards"]):
        raise ValueError("Unexpected setup/evaluation artifact stages")
    completed = set()
    for stage in setup:
        if not set(stage["depends_on"]).issubset(completed):
            raise ValueError("Setup commands are not topologically ordered")
        completed.add(stage["stage_id"])
    if any(not set(stage["depends_on"]).issubset(completed) for stage in evaluations):
        raise ValueError("An evaluation depends on an unsupported job")
    grouped = [("collect_fit", setup, 16, collect_fit_hours, [])]
    grouped += [(stage["stage_id"], [stage], 8, evaluation_hours, ["collect_fit"]) for stage in evaluations]
    jobs = []
    for identifier, stages, memory, hours, dependencies in grouped:
        lines = ["#!/bin/bash", "# Generated print-only payload; submit only after reviewed qualification passes.",
                 "# Ordinary job; evaluation requires afterok on the actual collect_fit job ID.",
                 f"#SBATCH --job-name=fh_main_{identifier}", "#SBATCH --partition=cpu",
                 f"#SBATCH --time={hours:02d}:00:00", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4",
                 f"#SBATCH --mem={memory}G", f"#SBATCH --chdir={runtime['source_release']}",
                 f"#SBATCH --output={root}/slurm/{identifier}_%j.out",
                 f"#SBATCH --error={root}/slurm/{identifier}_%j.err", "", "set -euo pipefail",
                 ': "${SLURM_JOB_ID:?Run only as the reviewed NEMO Slurm compute job}"',
                 "workspace=" + shlex.quote(runtime["workspace_root"]),
                 "source_release=" + shlex.quote(runtime["source_release"]),
                 "main_root=" + shlex.quote(str(root)),
                 "python=" + shlex.quote(runtime["python"]), ""]
        lines += ["export " + key + "=" + shlex.quote(value) for key, value in rendered["environment"].items()]
        lines += ["export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
                  'export TMPDIR="$workspace/tmp/fh_${SLURM_JOB_ID}"', 'export TMP="$TMPDIR" TEMP="$TMPDIR"',
                  'for output_path in "$main_root" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TORCH_HOME"; do',
                  '    [[ "$(readlink -m "$output_path")" == "$workspace"/* ]] || exit 2', "done",
                  '[[ "$(readlink -m "$source_release")" == "$workspace"/* ]] || exit 2',
                  'mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"',
                  'test -s "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"',
                  'test -x "$python"', 'cd "$source_release"', "",
                  "# Verify the immutable runtime receipt and exact donor before native imports."]
        release_pins = {key: release_record[key] for key in
                        ("candidate_commit", "archive_sha256", "baseline_tree_sha256")}
        lines += ['"$python" - "$source_release/FIRST_HIT_RELEASE.json" "$source_release/FIRST_HIT_BASELINE.json" <<\'PY\'',
                  "import json, sys", "record = json.load(open(sys.argv[1]))", "expected = " + repr(release_pins),
                  "if any(record.get(key) != value for key, value in expected.items()):",
                  "    raise SystemExit('Runtime release differs from reviewed launch record')",
                  "baseline = json.load(open(sys.argv[2]))",
                  "if baseline.get('release') != " + repr(release_record["source_release"]) + ":",
                  "    raise SystemExit('Runtime source directory differs from reviewed baseline')", "PY"]
        checks = [(donor["checkpoint_sha256"], donor["checkpoint"]),
                  (donor["config_sha256"], str(PurePosixPath(donor["run_dir"]) / "config.json"))]
        for digest, filename in checks:
            lines.append("printf '%s\\n' " + shlex.quote(digest + "  " + filename) + " | sha256sum --check --status")
        if dependencies:
            lines += ["", "# Artifact gates supplement the required afterok scheduler dependency."]
            for artifact in ("replay/metadata.json", "exposure/exposure.json", "fit/metadata.json", "fit/adapter.pt", "fit_diagnostics/residuals.json"):
                lines.append("test -s " + shlex.quote(str(root / artifact)))
        lines += ["", "# Keep every stage fresh; partial runs require a separately reviewed recovery."]
        lines += ["test ! -e " + shlex.quote(stage["output"]) for stage in stages]
        for stage in stages:
            lines += ["", "# stage: " + stage["stage_id"], shlex.join(stage["argv"])]
        script = "\n".join(lines) + "\n"
        jobs.append(dict(job_id=identifier, depends_on_jobs=dependencies,
                         dependency_kind="afterok" if dependencies else None,
                         script_name=identifier + ".sbatch", script=script,
                         script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                         resources=dict(partition="cpu", cpus=4, memory_gb=memory, hours=hours),
                         stages=[dict(stage_id=stage["stage_id"], argv=stage["argv"],
                                      argv_sha256=fingerprint(stage["argv"])) for stage in stages]))
    return dict(schema="intrmotiv/frozen-readout-slurm-bundle/v1", print_only=True, jobs_submitted=False,
                qualification_gate="Actual integration qualification must pass and be reviewed before submission",
                source_release_record=release_record, rendered_plan=rendered, jobs=jobs,
                remote_launch_directory=str(root / "launch"),
                directories_before_submission=[str(root / "launch"), str(root / "slurm")])


def write_launch_bundle(plan, validation, release_record, directory, **resources):
    """Write only new review artifacts; no remote access or job submission."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Preserve reviewed launch bundles; choose a new directory")
    bundle = render_slurm(plan, validation, release_record, **resources)
    directory.mkdir(parents=True)
    artifacts = {"protocol.json": plan, "rendered_plan.json": bundle["rendered_plan"],
                 "source_release.json": release_record}
    for name, content in artifacts.items():
        (directory / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    for job in bundle["jobs"]:
        (directory / job["script_name"]).write_text(job["script"])
    manifest = {key: value for key, value in bundle.items() if key not in ("rendered_plan", "jobs")}
    manifest["jobs"] = [{key: value for key, value in job.items() if key != "script"} for job in bundle["jobs"]]
    manifest["artifacts_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(directory.iterdir()) if path.is_file()}
    manifest["rendered_plan_sha256"] = manifest["artifacts_sha256"]["rendered_plan.json"]
    manifest["effective_plan_sha256"] = validation["effective_plan_sha256"]
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "render", "render-slurm"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--source-release", type=Path)
    parser.add_argument("--release-receipt", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--collect-fit-hours", type=int, default=12)
    parser.add_argument("--evaluation-hours", type=int, default=12)
    args = parser.parse_args()
    plan, validation = load_plan(args.plan, repository_root=args.repository_root, source_release=args.source_release)
    if args.command == "render-slurm":
        if args.release_receipt is None:
            parser.error("render-slurm requires the reviewed --release-receipt")
        receipt = json.loads(args.release_receipt.read_text())
        record = receipt.get("release", receipt)
        resources = dict(collect_fit_hours=args.collect_fit_hours, evaluation_hours=args.evaluation_hours)
        result = (write_launch_bundle(plan, validation, record, args.output_directory, **resources)
                  if args.output_directory is not None else render_slurm(plan, validation, record, **resources))
    else:
        if args.output_directory is not None or args.release_receipt is not None:
            parser.error("Launch artifact options apply only to render-slurm")
        result = validation if args.command == "validate" else render_commands(plan, validation)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
