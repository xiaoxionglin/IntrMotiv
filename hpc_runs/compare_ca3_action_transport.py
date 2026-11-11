"""Fit learned CA3 action transport against the frozen parallel-register probe.

The declarative synthetic spec references the parent's authoritative data and
training settings. Data construction, pairs and metrics are reused; no SF or
Slurm workflow is introduced. Test outputs are evaluated only after selecting
each checkpoint by held-out validation words.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ca3_action_transport import TransportReadout, inputs_from_registers
from parallel_action_register_probe import ROOT, audit, build_dataset, digest, matched_pairs, metrics


def fit(variant: dict, seed: int, data: dict, parent_config: dict, spec: dict, output: Path):
    torch.manual_seed(seed)
    base = data["features"]["landmark"]
    dg, actions = inputs_from_registers(data, parent_config)
    y = data["targets"] / parent_config["target_scale"]
    train, valid = (torch.tensor(data["indices"][key]) for key in ("train", "validation"))
    model = TransportReadout(base.size(1), dg.size(2), spec["tag_dim"], variant["rank"],
                             parent_config["repeat"], parent_config["expanded"],
                             variant["mode"], variant["hidden"])
    # Keep the same readout initialization across recurrent variants with the
    # same head shape, independently of how many random factors were drawn.
    torch.manual_seed(seed + 10000)
    for layer in model.head.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=parent_config["learning_rate"])
    best_loss, best_state, best_epoch = float("inf"), None, None
    trace, started = [], time.monotonic()
    for epoch in range(parent_config["epochs"] + 1):
        if epoch % parent_config["validation_every"] == 0 or epoch == parent_config["epochs"]:
            with torch.no_grad():
                valid_loss = (model(base[valid], dg[valid], actions[valid]) - y[valid]).square().mean().item()
                train_loss = (model(base[train], dg[train], actions[train]) - y[train]).square().mean().item()
            if not np.isfinite(valid_loss):
                raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
            trace.append({"epoch": epoch, "train_mse_scaled": train_loss, "validation_mse_scaled": valid_loss})
            if valid_loss < best_loss:
                best_loss, best_epoch = valid_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % 100 == 0:
                print(json.dumps({"variant": variant["id"], "seed": seed, "epoch": epoch,
                                  "train_mse_scaled": train_loss, "validation_mse_scaled": valid_loss}), flush=True)
        if epoch == parent_config["epochs"]:
            break
        order = train[torch.randperm(len(train))]
        for batch in order.split(parent_config["batch_size"]):
            loss = (model(base[batch], dg[batch], actions[batch]) - y[batch]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.load_state_dict(best_state)
    with torch.no_grad():
        prediction = (model(base, dg, actions) * parent_config["target_scale"]).numpy()
        matrices = model.core.transport.matrices()
        commutators = [torch.linalg.matrix_norm(matrices[a] @ matrices[b] - matrices[b] @ matrices[a]).item()
                       for a in range(3) for b in range(a + 1, 3)]
        tags = model.core.pooled_without_expiry(dg, actions, parent_config["expanded"])
    key = f"{variant['id']}__seed{seed}"
    torch.save(model.state_dict(), output / "weights" / f"{key}.pt")
    recurrence_parameters = sum(p.numel() for p in model.core.parameters())
    record = {"model": variant["id"], "seed": seed, "variant": variant,
              "best_epoch": best_epoch, "seconds": time.monotonic() - started,
              "parameters": sum(p.numel() for p in model.parameters()),
              "recurrence_parameters": recurrence_parameters,
              "head_input_size": base.size(1) + dg.size(2) * spec["tag_dim"],
              "max_commutator_frobenius": max(commutators),
              "max_action_spectral_radius": max(torch.linalg.eigvals(matrix).abs().max().item() for matrix in matrices),
              "max_tag_abs": tags.abs().max().item(), "matrices": matrices.tolist(), "trace": trace}
    print(json.dumps({k: v for k, v in record.items() if k not in ("matrices", "trace", "variant")}), flush=True)
    return prediction, record


def run(spec_path: Path, dry_run: bool = False) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = json.loads(spec_path.read_text())
    parent = ROOT / spec["parent_directory"]
    parent_config = json.loads((parent / "config.json").read_text())
    data = build_dataset(parent_config)
    checks = audit(data, parent_config)
    if checks["dataset_sha256"] != spec["expected_parent_dataset_sha256"]:
        raise ValueError("Generated data differ from the pinned parent dataset")
    if json.loads((parent / "dataset.json").read_text()) != json.loads(json.dumps(data["rows"])):
        raise ValueError("Parent manifest row order or contents changed")
    with np.load(parent / "predictions.npz") as previous:
        if not np.array_equal(previous["targets"], data["targets"].numpy()):
            raise ValueError("Parent targets changed")
    print(json.dumps({"spec_sha256": digest(spec), "audit": checks,
                      "variants": spec["variants"], "parent_training": parent_config}), flush=True)
    if dry_run:
        return
    output = ROOT / spec["output_directory"]
    (output / "weights").mkdir(parents=True, exist_ok=True)
    pairs = {split: matched_pairs(data, split) for split in data["indices"]}
    predictions, records, scores = {}, [], []
    # Reuse prior predictions verbatim. Never retrain a favorable baseline or
    # compare against a different data split, seed set or fitting budget.
    with np.load(parent / "predictions.npz") as previous:
        for name in spec["baseline_models"]:
            for seed in parent_config["fit_seeds"]:
                key = f"{name}__seed{seed}"
                predictions[key] = previous[key].copy()
    for variant in spec["variants"]:
        for seed in parent_config["fit_seeds"]:
            prediction, record = fit(variant, seed, data, parent_config, spec, output)
            predictions[f"{variant['id']}__seed{seed}"] = prediction
            records.append(record)
            (output / "fit_records.json").write_text(json.dumps(records, indent=2) + "\n")
    for key, prediction in predictions.items():
        name, seed = key.split("__seed")
        for split, indices in data["indices"].items():
            scores.append({"model": name, "seed": int(seed), "split": split,
                           **metrics(prediction, data, indices, pairs[split])})
    np.savez_compressed(output / "predictions.npz", targets=data["targets"].numpy(), **predictions)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, list(scores[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scores)
    for name, value in (("audit.json", checks), ("spec.json", spec)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    sources = [Path(__file__), ROOT / "hpc_runs/ca3_action_transport.py", ROOT / "hpc_runs/parallel_action_register_probe.py",
               ROOT / "exports/intrmotiv_prediction_starter/intrmotiv_transfer/state.py"]
    artifacts = [parent / name for name in ("predictions.npz", "config.json", "dataset.json", "provenance.json")]
    provenance = {"schema": spec["schema"], "spec_sha256": digest(spec), "python": sys.version,
                  "torch": torch.__version__, "numpy": np.__version__,
                  "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  "parent_artifacts": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
                  "scope": "Learned action-conditioned recurrence and endpoint readout; no learned future-DG generator, policy or DMLab test."}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.spec, args.dry_run)


if __name__ == "__main__":
    main()
