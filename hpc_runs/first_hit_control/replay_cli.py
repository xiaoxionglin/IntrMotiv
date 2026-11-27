"""Fit/reload a copied legacy readout from certified reward-free replay only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .detectors import detector_hits
from .legacy_runtime_adapter import CertifiedLegacyFirstHitAdapter
from .provenance import audit_source_snapshots, snapshot_sources
from .replay_fit import FitConfig, fit_replay, state_digest


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_class(source_root, filename, classname):
    path = Path(source_root) / "sf_working_directories/IntrMotiv/dmlab" / filename
    spec = importlib.util.spec_from_file_location(f"first_hit_frozen_{path.stem}_{file_hash(path)[:12]}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, classname)


def construct_adapter(snapshot, source_root, device="cpu"):
    """Instantiate only decoder/value modules; no native environment is loaded."""
    decoder_class = source_class(source_root, "custom_decoder.py", "MlpDecoderJit")
    head_class = source_class(source_root, "controller_q.py", "ControllerQHeads")
    decoder = decoder_class(SimpleNamespace(**snapshot["decoder_config"]), snapshot["layout"]["width"])
    # Frozen source copies are leaves; training may subsequently enable the copy.
    decoder.requires_grad_(False)
    heads = head_class(decoder.get_out_size(), snapshot["action_count"], auxiliary=True).requires_grad_(False)
    adapter = CertifiedLegacyFirstHitAdapter.from_certificate(
        decoder, heads, snapshot["eligibility"], snapshot["fit_config"]["horizon"], source_root=source_root,
    )
    return adapter.to(device)


def verify_restore_sources(directory, source_root, metadata):
    """Bind restored arithmetic to the archived fit, allowing host relocation.

    Tensor equality alone cannot detect a changed forward method. The required
    model modules are compared by package-relative identity, not original host
    paths; the complete archive is checked independently for corruption.
    """
    archive = Path(directory) / "source_snapshot"
    manifest_path = archive / "manifest.json"
    if file_hash(manifest_path) != metadata.get("source_snapshot_manifest_sha256"):
        raise ValueError("Fit source manifest hash mismatch")
    if not audit_source_snapshots(archive, check_current=False)["archive_valid"]:
        raise ValueError("Fit source archive is invalid")
    manifest = json.loads(manifest_path.read_text())
    package = Path(__file__).parent
    runtime = Path(source_root) / "sf_working_directories/IntrMotiv/dmlab"
    required = [package / name for name in (
        "runtime_adapter.py", "legacy_runtime_adapter.py", "eligibility.py", "detectors.py",
    )] + [runtime / name for name in ("custom_core.py", "custom_decoder.py", "controller_q.py")]
    for current in required:
        suffix = current.parts[-3:] if current.parent == package else current.parts[-4:]
        entries = [row for row in manifest["sources"]
                   if Path(row["source_path"]).parts[-len(suffix):] == suffix]
        if len(entries) != 1 or file_hash(current) != entries[0]["sha256"]:
            raise ValueError(f"Restored source differs from fit: {current.name}")


def load_fitted_adapter(directory: Path, source_root: Path, device="cpu"):
    """Load the exact evaluated checkpoint; reject edited weights or metadata."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != "intrmotiv/fitted-first-hit-readout/v1":
        raise ValueError("Unknown fitted readout schema")
    verify_restore_sources(directory, source_root, metadata)
    checkpoint = directory / "adapter.pt"
    if file_hash(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("First-hit readout checkpoint hash mismatch")
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if snapshot.get("schema") != "intrmotiv/first-hit-readout-weights/v1":
        raise ValueError("Unknown readout weights schema")
    model = construct_adapter(snapshot, source_root, device)
    model.load_state_dict(snapshot["model_state"], strict=True)
    model.eval().requires_grad_(False)
    if state_digest(model) != metadata["fit"]["final_readout_sha256"]:
        raise ValueError("Restored readout tensors differ from fitted state")
    return model, metadata


def run_fit(replay_directory, output, source_root, config, *, device="cpu"):
    replay_directory, output = Path(replay_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing fit outputs; choose a new directory")
    source_root = Path(source_root).resolve(strict=True)
    data_path, state_path = replay_directory / "replay.npz", replay_directory / "decoder_state.pt"
    input_metadata_path = replay_directory / "metadata.json"
    metadata = json.loads(input_metadata_path.read_text())
    if metadata.get("schema") != "intrmotiv/frozen-first-hit-replay/v1":
        raise ValueError("Unknown replay schema")
    if not metadata.get("model_unchanged") or metadata["model_sha256_before"] != metadata["model_sha256_after"]:
        raise ValueError("Replay representation was not certified frozen")
    if metadata.get("native_reward_zero_every_step") is not True:
        raise ValueError("A per-step native reward-free certificate is required")
    if metadata.get("reset_observations_used_as_successors") != 0:
        raise ValueError("Reset observations entered factual replay")
    config_hash = metadata.get("source_config_sha256", "")
    if len(config_hash) != 64 or any(c not in "0123456789abcdef" for c in config_hash):
        raise ValueError("A source configuration SHA-256 is required")
    if not isinstance(metadata.get("evaluation_overrides"), dict):
        raise ValueError("Collection evaluation overrides are required")
    if file_hash(data_path) != metadata["dataset_sha256"] or file_hash(state_path) != metadata["decoder_state_sha256"]:
        raise ValueError("Replay or source decoder hash differs from collection certificate")
    source = torch.load(state_path, map_location="cpu", weights_only=True)
    if source["decoder_class"] != "MlpDecoderJit" or source["layout"] != metadata["layout"]:
        raise ValueError("This fit requires the certified source legacy decoder and layout")
    mode = metadata["primary_detector"]
    if mode not in ("winner", "exclusive"):
        raise ValueError("Production source detector must be explicit winner or exclusive")
    snapshot = {"schema": "intrmotiv/first-hit-readout-weights/v1", "fit_config": asdict(config),
                "layout": source["layout"], "action_count": source["action_count"],
                "decoder_config": source["decoder_config"], "eligibility": metadata["eligibility"]}
    output.mkdir(parents=True)
    source_files = list(Path(__file__).parent.glob("*.py"))
    runtime = source_root / "sf_working_directories/IntrMotiv/dmlab"
    source_files.extend(runtime / name for name in ("custom_decoder.py", "controller_q.py", "custom_core.py"))
    snapshot_sources(source_files, output / "source_snapshot")
    input_files = {str(path): file_hash(path) for path in (data_path, state_path, input_metadata_path)}
    protocol = {"schema": "intrmotiv/first-hit-fit-protocol/v1", "fit_config": asdict(config),
                "input_files": input_files, "detector": mode, "device": device,
                "initialized_from": "Source pretrained decoder copied; new deadline Q head; original actor unchanged",
                "goal_sampling": "Every detector identity on each factual transition",
                "deadline_sampling": "Uniform integers from 1 through declared horizon",
                "selection": "Fixed update count; whole-episode holdout is diagnostic only",
                "scope": "Readout consolidation after reward-free exploration; requires executed-control evaluation"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.manual_seed(config.seed)
    model = construct_adapter(snapshot, source_root, device)
    model.decoder.load_state_dict(source["decoder_state_dict"], strict=True)
    with np.load(data_path) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    fitter = partial(construct_adapter, snapshot, source_root, device)
    fit = fit_replay(model, data, config, partial(detector_hits, mode=mode), target_factory=fitter,
                     progress=lambda row: print(json.dumps(row), flush=True))
    snapshot["model_state"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint = output / "adapter.pt"
    torch.save(snapshot, checkpoint)
    source_audit = audit_source_snapshots(output / "source_snapshot")
    if not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"]:
        raise RuntimeError("Declared source bytes changed during fitting")
    if any(file_hash(path) != expected for path, expected in input_files.items()):
        raise RuntimeError("A declared replay input changed during fitting")
    valid = data["successor_valid"]
    with torch.no_grad():
        activity = torch.as_tensor(data["successor_dg"][valid])
        exposures = {name: detector_hits(activity, name).sum(0).tolist() for name in ("winner", "exclusive")}
    result = {"schema": "intrmotiv/fitted-first-hit-readout/v1", "fit": fit,
              "input_files": input_files, "protocol_sha256": file_hash(output / "protocol.json"),
              "checkpoint_sha256": file_hash(checkpoint), "source_audit": source_audit,
              "source_snapshot_manifest_sha256": file_hash(output / "source_snapshot/manifest.json"),
              "primary_detector": mode, "goal_positive_observations": exposures,
              "source_checkpoint_sha256": metadata["checkpoint_sha256"],
              "source_config_sha256": metadata["source_config_sha256"],
              "collection_evaluation_overrides": metadata["evaluation_overrides"],
              "source_run": metadata["source_run"], "layout": source["layout"],
              "replay_directory": str(replay_directory), "source_root": str(source_root)}
    (output / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    reloaded, _ = load_fitted_adapter(output, source_root, device)
    if state_digest(reloaded) != state_digest(model):
        raise RuntimeError("Saved first-hit model did not reload exactly")
    print(json.dumps({"output": str(output), "reload_exact": True, "fit_seconds": fit["seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("replay", "output", "source-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = FitConfig(seed=args.seed, updates=args.updates, batch_size=args.batch_size,
                       horizon=args.horizon, learning_rate=args.learning_rate)
    run_fit(args.replay, args.output, args.source_root, config, device=args.device)


if __name__ == "__main__":
    main()
