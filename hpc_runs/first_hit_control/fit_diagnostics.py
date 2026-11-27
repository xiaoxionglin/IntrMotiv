"""Frozen final-readout Bellman diagnostics; never used for model selection.

These residuals bootstrap from the same fitted model. They measure consistency
on recorded factual actions, not calibrated hitting probabilities or coverage.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from .detectors import detector_hits
from .replay_cli import file_hash, load_fitted_adapter
from .replay_fit import episode_split, replay_batch, state_digest, validate_replay


def residual_statistics(prediction, target, mask):
    count = int(mask.sum())
    if not count:
        return dict(labels=0, mse=None, prediction_min=None, prediction_max=None, outside_unit_interval=None)
    values = prediction[mask]
    return dict(labels=count, mse=float(((values - target[mask]) ** 2).mean()),
                prediction_min=float(values.min()), prediction_max=float(values.max()),
                outside_unit_interval=float(((values < 0) | (values > 1)).mean()))


def diagnose(model, data, fit_config, detector_mode, *, samples_per_bin=1024, seed=20261127):
    """Use fixed independent diagnostic draws, preserving model and data."""
    if type(samples_per_bin) is not int or samples_per_bin < 1:
        raise ValueError("samples_per_bin must be positive")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Diagnostics require a frozen evaluation readout")
    action_count = model.q_heads.main.out_features
    validate_replay(data, model.layout, action_count)
    horizon = fit_config["horizon"]
    if model.horizon_scale != horizon:
        raise ValueError("Fit horizon and readout scale differ")
    train, holdout = episode_split(data["episode_id"], fit_config["holdout_fraction"], fit_config["seed"])
    bins = [(lo, min(hi, horizon)) for lo, hi in ((1, 1), (2, 4), (5, 16), (17, 64), (65, horizon))
            if lo <= horizon]
    detector = partial(detector_hits, mode=detector_mode)
    random = np.random.default_rng(seed)
    before = state_digest(model)
    panels = {}
    with torch.no_grad():
        for name, available in (("train", train), ("holdout", holdout)):
            rows = []
            for low, high in bins:
                if not len(available):
                    rows.append(dict(deadlines=[low, high], sampled_rows=0, missing=True))
                    continue
                selected = random.choice(available, min(samples_per_bin, len(available)), replace=False)
                remaining = random.integers(low, high + 1, len(selected))
                prediction, target, usable = replay_batch(model, model, data, selected, remaining, detector)
                prediction, target, usable = (x.cpu().numpy() for x in (prediction, target, usable))
                hits = detector(torch.as_tensor(data["successor_dg"][selected].copy())).numpy()
                actions = data["action"][selected]
                row = dict(deadlines=[low, high], sampled_rows=len(selected), missing=False,
                           all=residual_statistics(prediction, target, usable),
                           immediate_hit=residual_statistics(prediction, target, usable & hits),
                           not_immediate_hit=residual_statistics(prediction, target, usable & ~hits))
                row["per_goal"] = [residual_statistics(prediction[:, g], target[:, g], usable[:, g])
                                   for g in range(model.layout.goals)]
                row["per_action"] = [residual_statistics(prediction, target, usable & (actions == a)[:, None])
                                     for a in range(action_count)]
                rows.append(row)
            panels[name] = rows
    if state_digest(model) != before:
        raise RuntimeError("Diagnostic execution mutated the fitted readout")
    return dict(schema="intrmotiv/final-first-hit-residual/v1", panels=panels,
                seed=seed, samples_per_bin=samples_per_bin, readout_unchanged=True,
                readout_sha256=before, primary_detector=detector_mode,
                interpretation="Final-model self-bootstrapped factual Bellman residual; not calibration or control evidence",
                selection="Diagnostic only; final update-count checkpoint selected before this computation")


def run_diagnostics(replay, fitted, source_root, output, *, samples_per_bin=1024):
    replay, fitted, output = Path(replay).resolve(), Path(fitted).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous diagnostics; choose a new output directory")
    model, metadata = load_fitted_adapter(fitted, Path(source_root), device="cpu")
    data_path, input_metadata = replay / "replay.npz", replay / "metadata.json"
    for path in (data_path, input_metadata):
        if file_hash(path) != metadata["input_files"].get(str(path)):
            raise ValueError("Replay input differs from the fitted data")
    with np.load(data_path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    result = diagnose(model, data, metadata["fit"]["config"], metadata["primary_detector"],
                      samples_per_bin=samples_per_bin)
    result["input_files"] = {str(path): file_hash(path) for path in
                              (data_path, input_metadata, fitted / "metadata.json", fitted / "adapter.pt")}
    package = Path(__file__).parent
    result["source_sha256"] = {name: file_hash(package / name) for name in
                               ("fit_diagnostics.py", "replay_fit.py", "targets.py", "detectors.py")}
    output.mkdir(parents=True)
    (output / "residuals.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "readout_unchanged": True}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "fitted", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples-per-bin", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    run_diagnostics(args.replay, args.fitted, args.source_root, args.output, samples_per_bin=args.samples_per_bin)


if __name__ == "__main__":
    main()
