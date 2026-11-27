"""Fit/reload the original action head on certified frozen replay, without DMLab."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from .detectors import detector_hits
from .provenance import audit_source_snapshots, snapshot_sources
from .replay_cli import file_hash, source_class
from .replay_fit import FitConfig, episode_split, fit_replay, state_digest
from .stationary_runtime_adapter import KIND, StationaryFirstHitAdapter, stationary_replay_batch


RUNTIME_FILES = (
    "sf_working_directories/IntrMotiv/dmlab/custom_core.py",
    "sf_working_directories/IntrMotiv/dmlab/custom_decoder.py",
    "sample_factory/model/action_parameterization.py",
    "sample_factory/algo/utils/action_distributions.py",
    "sample_factory/model/model_utils.py",
    "sample_factory/model/decoder.py",
    "sample_factory/algo/utils/torch_utils.py",
)
PACKAGE_FILES = (
    "stationary_runtime_adapter.py", "stationary_replay_cli.py", "replay_fit.py",
    "eligibility.py", "runtime_adapter.py", "detectors.py", "targets.py",
    "provenance.py", "replay_cli.py", "legacy_runtime_adapter.py",
    "reused_plan.py", "analysis_plan.py",
)
DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "studies/stationary_projected_c05_s99_20261127.analysis.json"


def source_files(source_root):
    package = Path(__file__).parent
    return {**{f"hpc_runs/first_hit_control/{name}": package / name for name in PACKAGE_FILES},
            **{name: Path(source_root) / name for name in RUNTIME_FILES}}


def verify_imported_arithmetic(source_root):
    """Dependencies may relocate, but their loaded bytes must match the source."""
    import sample_factory.algo.utils.action_distributions as distributions
    import sample_factory.algo.utils.torch_utils as torch_utils
    import sample_factory.model.model_utils as model_utils
    import sample_factory.model.decoder as decoder
    for module in (distributions, torch_utils, model_utils, decoder):
        relative = module.__name__.replace(".", "/") + ".py"
        if file_hash(module.__file__) != file_hash(Path(source_root) / relative):
            raise ValueError(f"Imported arithmetic differs from declared source: {relative}")


def construct_stationary(snapshot, source_root, device="cpu"):
    """Construct exact original classes only; never create an environment/core."""
    verify_imported_arithmetic(source_root)
    cfg = SimpleNamespace(**snapshot["decoder_config"])
    decoder = source_class(source_root, "custom_decoder.py", "MlpDecoderJit")(cfg, snapshot["layout"]["width"])
    path = Path(source_root) / "sample_factory/model/action_parameterization.py"
    spec = importlib.util.spec_from_file_location(f"stationary_action_{file_hash(path)[:12]}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action = module.ActionParameterizationDefault(cfg, decoder.get_out_size(), gym.spaces.Discrete(snapshot["action_count"]))
    return StationaryFirstHitAdapter(decoder, action, snapshot["eligibility"], snapshot["fit_config"]["horizon"],
        source_root=source_root, gamma=snapshot["gamma"]).to(device)


def verify_restore_sources(directory, source_root, metadata):
    archive = Path(directory) / "source_snapshot"
    manifest_path = archive / "manifest.json"
    if file_hash(manifest_path) != metadata.get("source_snapshot_manifest_sha256"):
        raise ValueError("Stationary source manifest hash mismatch")
    if not audit_source_snapshots(archive, check_current=False)["archive_valid"]:
        raise ValueError("Stationary source archive is invalid")
    manifest = json.loads(manifest_path.read_text())
    for relative, current in source_files(source_root).items():
        suffix = Path(relative).parts
        entries = [row for row in manifest["sources"] if Path(row["source_path"]).parts[-len(suffix):] == suffix]
        if len(entries) != 1 or file_hash(current) != entries[0]["sha256"]:
            raise ValueError(f"Restored stationary arithmetic differs from fit: {relative}")
    verify_imported_arithmetic(source_root)


def load_fitted_stationary(directory, source_root, device="cpu"):
    """Restore an immutable model with source arithmetic and objective binding."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != "intrmotiv/fitted-first-hit-readout/v1" or metadata.get("kind") != KIND:
        raise ValueError("Unknown stationary fitted readout kind/schema")
    verify_restore_sources(directory, source_root, metadata)
    checkpoint = directory / "adapter.pt"
    if file_hash(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Stationary checkpoint hash mismatch")
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (snapshot.get("schema") != "intrmotiv/stationary-first-hit-readout-weights/v1"
            or snapshot.get("kind") != KIND or snapshot.get("gamma") != metadata.get("gamma")
            or snapshot.get("bootstrap_projection") != [0., 1.] or metadata.get("bootstrap_projection") != [0., 1.]
            or snapshot.get("layout") != metadata.get("layout")
            or snapshot.get("fit_config") != metadata["fit"]["config"]):
        raise ValueError("Stationary checkpoint objective/layout certificate differs from metadata")
    model = construct_stationary(snapshot, source_root, device)
    model.load_state_dict(snapshot["model_state"], strict=True)
    model.eval().requires_grad_(False)
    if state_digest(model) != metadata["fit"]["final_readout_sha256"]:
        raise ValueError("Restored stationary tensors differ from fitted state")
    return model, metadata


def validate_plan(plan_path, metadata, config, gamma, replay_directory):
    """Bind a production invocation to the current predeclared dependency plan."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "intrmotiv/reused-replay-control-plan/v1":
        raise ValueError("Unknown stationary analysis plan schema")
    if metadata["dataset_sha256"] == plan["replay"]["dataset_sha256"]:
        expected = plan["fit"]
        replay = plan["replay"]
        pins = (replay["dataset_sha256"], replay["metadata_sha256"], replay["decoder_state_sha256"])
        phase = "full"
    else:
        expected = dict(plan["fit"], **plan["qualification"])
        expected["horizon"] = plan["qualification"]["horizon_external_only"]
        pins = (expected["replay_dataset_sha256"], expected["replay_metadata_sha256"], plan["replay"]["decoder_state_sha256"])
        phase = "qualification"
    for key in ("seed", "updates", "batch_size", "horizon", "learning_rate", "target_update_period", "holdout_fraction"):
        if getattr(config, key) != expected[key]:
            raise ValueError(f"Fit {key} differs from declared {phase} budget")
    paths = [Path(replay_directory) / name for name in ("replay.npz", "metadata.json", "decoder_state.pt")]
    if any(file_hash(path) != expected_hash for path, expected_hash in zip(paths, pins)):
        raise ValueError("Replay input differs from declared stationary plan")
    if (gamma != plan["objective"]["gamma"] or metadata["checkpoint_sha256"] != plan["donor"]["checkpoint_sha256"]
            or metadata["source_config_sha256"] != plan["donor"]["config_sha256"]):
        raise ValueError("Stationary objective or donor differs from declaration")
    return {"path": str(Path(plan_path).resolve()), "sha256": file_hash(plan_path), "phase": phase,
            "analysis_id": plan["analysis_id"], "schema": plan["schema"], "workflow_version": plan["workflow_version"]}


def run_fit(replay_directory, output, source_root, config, *, device="cpu", gamma=.99,
            analysis_plan=None, canonical_plan_validation=None):
    """Fit a copied decoder AND copied pretrained action head on fixed replay.

    ``analysis_plan=None`` permits synthetic fixture tests. The command-line
    entry always supplies a validated production plan before any fitting.
    """
    replay_directory, output = Path(replay_directory).resolve(), Path(output).resolve()
    source_root = Path(source_root).resolve(strict=True)
    if output.exists():
        raise FileExistsError("Preserve existing stationary outputs; choose a new directory")
    if not 0 < gamma < 1:
        raise ValueError("Stationary gamma must be between zero and one")
    data_path, state_path = replay_directory / "replay.npz", replay_directory / "decoder_state.pt"
    metadata_path = replay_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "intrmotiv/frozen-first-hit-replay/v1":
        raise ValueError("Unknown replay schema")
    if not metadata.get("model_unchanged") or metadata["model_sha256_before"] != metadata["model_sha256_after"]:
        raise ValueError("Replay representation was not certified frozen")
    if metadata.get("native_reward_zero_every_step") is not True:
        raise ValueError("A per-step native reward-free certificate is required")
    if metadata.get("reset_observations_used_as_successors") != 0:
        raise ValueError("Reset observations entered replay")
    if metadata.get("primary_detector") != "winner":
        raise ValueError("This stationary native design requires the canonical winner detector")
    config_hash = metadata.get("source_config_sha256", "")
    if len(config_hash) != 64 or any(c not in "0123456789abcdef" for c in config_hash):
        raise ValueError("Source configuration SHA-256 is required")
    if not isinstance(metadata.get("evaluation_overrides"), dict):
        raise ValueError("Collection evaluation overrides are required")
    if file_hash(data_path) != metadata["dataset_sha256"] or file_hash(state_path) != metadata["decoder_state_sha256"]:
        raise ValueError("Replay or source readout hash differs from collection certificate")
    declaration = validate_plan(analysis_plan, metadata, config, gamma, replay_directory) if analysis_plan else None
    if canonical_plan_validation is not None:
        if declaration is None or canonical_plan_validation["source_file_sha256"] != declaration["sha256"]:
            raise ValueError("Canonical plan validation differs from the supplied declaration")
        declaration.update({key: canonical_plan_validation[key] for key in
            ("effective_plan_sha256", "declared_plan_sha256", "donor_study_sha256", "canonical_workflow_version")})
    source = torch.load(state_path, map_location="cpu", weights_only=True)
    if (source.get("decoder_class") != "MlpDecoderJit" or source.get("action_head_class") != "ActionParameterizationDefault"
            or source["layout"] != metadata["layout"] or source["layout"] != metadata["eligibility"]["layout"]):
        raise ValueError("Require both original source classes and their complete certified layout")
    snapshot = {"schema": "intrmotiv/stationary-first-hit-readout-weights/v1", "kind": KIND,
        "gamma": float(gamma), "bootstrap_projection": [0., 1.], "fit_config": asdict(config),
        "layout": source["layout"], "action_count": source["action_count"],
        "decoder_config": source["decoder_config"], "eligibility": metadata["eligibility"]}
    if declaration and (snapshot["layout"] != {"width": 1165, "goal_start": 1149, "goals": 16}
                        or snapshot["action_count"] != 5 or snapshot["decoder_config"]["decoder_mlp_layers"] != [128, 128]):
        raise ValueError("Declared native source dimensions differ from certified original head")
    output.mkdir(parents=True)
    files = list(source_files(source_root).values()) + ([Path(analysis_plan).resolve()] if analysis_plan else [])
    snapshot_sources(files, output / "source_snapshot")
    inputs = {str(path): file_hash(path) for path in (data_path, state_path, metadata_path)}
    protocol = {"schema": "intrmotiv/stationary-first-hit-fit-protocol/v1", "kind": KIND,
        "fit_config": asdict(config), "gamma": gamma, "bootstrap_projection": [0., 1.], "projection_order": "clip before gamma",
        "analysis_plan": declaration, "input_files": inputs, "device": device,
        "initialization": "Both original pretrained decoder and action head; no added neural parameters",
        "current_goal_mask": "Canonical winner from certified CA3 slot zero; not all positive units or source_goal",
        "sampling": "Shared native episode split and factual sampler; consume unused deadline draws; fixed updates",
        "selection": "Final weights only; whole-episode holdout is diagnostic"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.manual_seed(config.seed)
    model = construct_stationary(snapshot, source_root, device)
    model.decoder.load_state_dict(source["decoder_state_dict"], strict=True)
    model.action_parameterization.load_state_dict(source["action_head_state_dict"], strict=True)
    for module, key in ((model.decoder, "decoder_state_dict"), (model.action_parameterization, "action_head_state_dict")):
        for name, actual in module.state_dict().items():
            expected = source[key][name]
            if actual.dtype != expected.dtype or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
                raise ValueError(f"Original pretrained tensor did not copy exactly: {key}/{name}")
    initial = {"verified": True, "decoder_class": source["decoder_class"], "action_head_class": source["action_head_class"],
        "decoder_state_sha256": state_digest(model.decoder), "action_head_state_sha256": state_digest(model.action_parameterization),
        "action_head_class_sha256": file_hash(source_root / "sample_factory/model/action_parameterization.py"),
        "action_distributions_sha256": file_hash(source_root / "sample_factory/algo/utils/action_distributions.py"),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if declaration and initial["trainable_parameters"] != 166405:
        raise ValueError("Original native readout parameter count differs from declaration")
    (output / "initialization.json").write_text(json.dumps(initial, indent=2) + "\n")
    with np.load(data_path) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    fit = fit_replay(model, data, config, partial(detector_hits, mode="winner"),
        target_factory=partial(construct_stationary, snapshot, source_root, device),
        action_count=model.action_count, batch_builder=stationary_replay_batch,
        progress=lambda row: print(json.dumps(row), flush=True))
    snapshot["model_state"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint = output / "adapter.pt"
    torch.save(snapshot, checkpoint)
    source_audit = audit_source_snapshots(output / "source_snapshot")
    if not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"]:
        raise RuntimeError("Declared stationary source bytes changed during fitting")
    if any(file_hash(path) != digest for path, digest in inputs.items()):
        raise RuntimeError("Declared replay changed during fitting")
    with torch.no_grad():
        activity = torch.as_tensor(data["successor_dg"][data["successor_valid"]])
        exposures = {mode: detector_hits(activity, mode).sum(0).tolist() for mode in ("winner", "exclusive")}
        core = torch.as_tensor(data["core"])
        winners = detector_hits(model.current_dg(core), "winner")
        train, holdout = episode_split(data["episode_id"], config.holdout_fraction, config.seed)
        masks = {name: winners[indices].sum(0).tolist() for name, indices in
                 (("all_replay", np.arange(len(core))), ("train", train), ("holdout", holdout))}
    result = {"schema": "intrmotiv/fitted-first-hit-readout/v1", "kind": KIND, "gamma": gamma,
        "bootstrap_projection": [0., 1.], "fit": fit, "input_files": inputs, "analysis_plan": declaration,
        "effective_plan_sha256": declaration.get("effective_plan_sha256") if declaration else None,
        "donor_study_sha256": declaration.get("donor_study_sha256") if declaration else None,
        "protocol_sha256": file_hash(output / "protocol.json"), "checkpoint_sha256": file_hash(checkpoint),
        "source_audit": source_audit, "source_snapshot_manifest_sha256": file_hash(output / "source_snapshot/manifest.json"),
        "source_initialization": initial, "initialization_sha256": file_hash(output / "initialization.json"),
        "primary_detector": "winner", "goal_positive_observations": exposures, "current_winner_rows_by_goal": masks,
        "source_checkpoint_sha256": metadata["checkpoint_sha256"], "source_config_sha256": metadata["source_config_sha256"],
        "collection_evaluation_overrides": metadata["evaluation_overrides"], "source_run": metadata["source_run"],
        "layout": source["layout"], "replay_directory": str(replay_directory), "source_root": str(source_root)}
    (output / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    reloaded, _ = load_fitted_stationary(output, source_root, device)
    if state_digest(reloaded) != state_digest(model):
        raise RuntimeError("Saved stationary readout did not reload exactly")
    print(json.dumps({"output": str(output), "reload_exact": True, "fit_seconds": fit["seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "output", "source-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=256, help="External evaluation deadline; never a neural input")
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    # Production must validate the canonical donor StudySpec and all referenced
    # evidence, rather than relying only on this module's fixture-sized checks.
    from .reused_plan import load_plan
    plan, validation = load_plan(args.analysis_plan)
    if args.source_root.resolve() != Path(plan["runtime"]["source_release"]).resolve():
        raise ValueError("Supplied source root differs from the declared immutable release")
    if args.device != plan["fit"]["device"] or args.threads != plan["fit"]["threads"]:
        raise ValueError("Production device/thread resources differ from declaration")
    torch.set_num_threads(args.threads)
    config = FitConfig(seed=args.seed, updates=args.updates, batch_size=args.batch_size,
                       horizon=args.horizon, learning_rate=args.learning_rate)
    run_fit(args.replay, args.output, args.source_root, config, device=args.device, gamma=args.gamma,
            analysis_plan=args.analysis_plan, canonical_plan_validation=validation)


if __name__ == "__main__":
    main()
