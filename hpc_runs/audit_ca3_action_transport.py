"""Replay learned checkpoints and audit comparison artifacts without refitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ca3_action_transport import TransportReadout, inputs_from_registers
from parallel_action_register_probe import ROOT, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.directory
    spec = json.loads((directory / "spec.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    for source, expected in {**provenance["sources"], **provenance["parent_artifacts"]}.items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source/parent artifact changed: {source}")
    parent = ROOT / spec["parent_directory"]
    config = json.loads((parent / "config.json").read_text())
    data = build_dataset(config)
    dg, actions = inputs_from_registers(data, config)
    archive = np.load(directory / "predictions.npz")
    # Verify baselines were copied exactly, including their target row order.
    with np.load(parent / "predictions.npz") as original:
        assert np.array_equal(original["targets"], archive["targets"])
        for name in spec["baseline_models"]:
            for seed in config["fit_seeds"]:
                key = f"{name}__seed{seed}"
                assert np.array_equal(original[key], archive[key]), key
    checks, weight_hashes = [], {}
    selected = torch.tensor(data["indices"]["longer_words"][:32])
    for variant in spec["variants"]:
        for seed in config["fit_seeds"]:
            key = f"{variant['id']}__seed{seed}"
            model = TransportReadout(data["features"]["landmark"].size(1), dg.size(2), spec["tag_dim"],
                                     variant["rank"], config["repeat"], config["expanded"],
                                     variant["mode"], variant["hidden"])
            weight_path = directory / "weights" / f"{key}.pt"
            model.load_state_dict(torch.load(weight_path, weights_only=True))
            weight_hashes[weight_path.name] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
            with torch.no_grad():
                prediction = model(data["features"]["landmark"], dg, actions) * config["target_scale"]
                torch.testing.assert_close(prediction, torch.tensor(archive[key]), atol=1e-5, rtol=1e-5)
                # Independently execute the full landmark x age x tag state.
                full = torch.zeros(len(selected), dg.size(2), config["expanded"], spec["tag_dim"] + 1)
                full = model.core.advance(full, dg[selected, 0], torch.full((len(selected),), 3))
                for step in range(config["steps"]):
                    full = model.core.advance(full, dg[selected, step + 1], actions[selected, step])
                expected = data["dg_state"][selected][:, [0, 1, 2, 3, 15]]
                assert torch.equal(full[..., 0], expected), key
                tags = full[..., 1:].sum(2) / config["repeat"]
                full_prediction = model.head(torch.cat((data["features"]["landmark"][selected], tags.flatten(1)), dim=1)) * config["target_scale"]
                torch.testing.assert_close(full_prediction, prediction[selected], atol=2e-4, rtol=2e-4)
            checks.append({"model": key, "checkpoint_replay": True, "full_reference_matches": True,
                           "original_ca3_exact": True})
    # Endpoint metrics recomputed directly in NumPy, without calling the
    # fitting script's scoring function.
    for row in csv.DictReader((directory / "metrics.csv").open()):
        key = f"{row['model']}__seed{row['seed']}"
        indices = data["indices"][row["split"]]
        prediction, target = archive[key][indices], archive["targets"][indices]
        rmse = np.sqrt(((prediction - target) ** 2).sum(axis=1).mean())
        correct = (np.round(prediction) == target).all(axis=1).mean()
        assert abs(float(rmse) - float(row["position_rmse"])) < 1e-6
        assert abs(float(correct) - float(row["rounded_endpoint_accuracy"])) < 1e-10
        assert np.isfinite(prediction).all()
    result = {"parent_and_source_hashes": True, "baseline_predictions_identical": True,
              "independent_endpoint_metrics": True, "checkpoints": checks, "weight_sha256": weight_hashes,
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "replay_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(checks)} checkpoints replayed; full CA3 matches; baselines identical; endpoint metrics verified")


if __name__ == "__main__":
    main()
