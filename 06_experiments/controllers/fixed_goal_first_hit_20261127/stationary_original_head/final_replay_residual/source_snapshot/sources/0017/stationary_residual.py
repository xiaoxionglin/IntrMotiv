"""Read-only final-network Bellman residual audit on complete cached replay.

The online and target functions are the same saved final network. This differs
from the historical last minibatch, which used an older target and pre-update
online weights. No training, target projection, or environment rollout occurs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .neural import NeuralConfig, digest, observe, sparse_context_environment, state_digest
from .provenance import audit_source_snapshots, snapshot_sources
from .stationary_readout import StationaryReadout, original_classes, stationary_batch


def residual_metrics(prediction, labels, usable):
    """Describe exactly the supplied valid factual-action labels."""
    predicted = np.asarray(prediction, dtype=np.float64)[usable]
    target = np.asarray(labels, dtype=np.float64)[usable]
    if not len(predicted) or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Residual metrics require finite usable labels")
    error = predicted - target
    return dict(labels=int(usable.sum()), omitted_labels=int(usable.size - usable.sum()),
        mse=float(np.mean(error ** 2)), rmse=float(np.sqrt(np.mean(error ** 2))),
        mean_residual=float(error.mean()), maximum_absolute_residual=float(np.max(np.abs(error))),
        prediction_min=float(predicted.min()), prediction_max=float(predicted.max()),
        factual_predictions_above_one=int(np.sum(predicted > 1)),
        target_min=float(target.min()), target_max=float(target.max()))


@torch.no_grad()
def complete_replay_values(model, data, goal_ids, gamma, batch_size=256):
    """Evaluate every row once, retaining the original current-goal mask."""
    pieces = []
    for offset in range(0, len(data["action"]), batch_size):
        rows = np.arange(offset, min(offset + batch_size, len(data["action"])))
        output = stationary_batch(model, model, data, rows, goal_ids, gamma)
        pieces.append(tuple(tensor.detach().cpu().numpy() for tensor in output))
    return tuple(np.concatenate([part[index] for part in pieces]) for index in range(3))


def final_sample_indices(config, seed, row_count, goal_count):
    """Reproduce sampling indices only; no parameter or optimizer operations."""
    rng = np.random.default_rng(seed + 10000)
    for _ in range(config.updates):
        rows = rng.integers(row_count, size=config.batch_size)
        rng.integers(goal_count, size=config.batch_size)
        rng.integers(1, config.horizon + 1, size=config.batch_size)
    return rows


def run(run_directory, output):
    root, output = Path(run_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve prior residual audit")
    audit = audit_source_snapshots(root / "source_snapshot")
    if not audit["archive_valid"] or not audit["current_sources_unchanged"]:
        raise ValueError("Exact fitted-run source bytes are required")
    manifest_path, results_path = root / "pre_run_manifest.json", root / "results.json"
    manifest, result = [json.loads(path.read_text()) for path in (manifest_path, results_path)]
    if digest(manifest_path) != result["pre_run_manifest_sha256"]:
        raise ValueError("Fitted results and original manifest disagree")
    summary = next(row for row in result["runs"] if row["environment"] == "sparse_context" and row["seed"] == 99)
    folder = root / "sparse_context/seed_99"
    checkpoint, trials = folder / "readout.pt", folder / "trials.csv"
    dataset = next(Path(path) for path in manifest["datasets"] if Path(path).parent.name == "seed_99"
                   and Path(path).parent.parent.name == "sparse_context")
    expected = {str(checkpoint): summary["checkpoint_sha256"], str(dataset): manifest["datasets"][str(dataset)]}
    if any(digest(path) != sha for path, sha in expected.items()):
        raise ValueError("Saved checkpoint or replay differs from its certificate")
    inputs = {str(path): digest(path) for path in (manifest_path, results_path, checkpoint, trials, dataset)}
    sources = [Path(row["source_path"]) for row in manifest["source_snapshot"]["sources"]]
    sources += [Path(__file__), Path(__file__).parents[1] / "test_first_hit_stationary_residual.py"]
    output.mkdir(parents=True)
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    config, gamma = NeuralConfig(**manifest["config"]), manifest["gamma"]
    protocol = dict(schema="intrmotiv/stationary-final-residual/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        source_snapshot=snapshot, inputs=inputs, selected_run="sparse_context/seed_99",
        target_contract="Same final saved network for online argmax and target value; actual stored successor CA3; same goal; absorbing hit/physical termination; original current-goal mask",
        scope="All cached factual transitions plus the already recorded failed-path start; no training, projection, or rollout",
        historical_loss_contract="Recorded last minibatch uses pre-update-2500 online weights and target copied after update2400; those old weights are unavailable, so no exact historical-target recomputation is claimed")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(1)
    decoder, action, _ = original_classes()
    env, dg, goals = sparse_context_environment()
    model = StationaryReadout(dg.shape[1], config, decoder, action, manifest["decoder_settings"], env.n_actions)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    before = state_digest(model)
    with np.load(dataset) as saved:
        data = {key: saved[key] for key in saved.files}
    prediction, labels, usable = complete_replay_values(model, data, goals, gamma)
    per_goal = {str(goal): residual_metrics(prediction[:, i], labels[:, i], usable[:, i])
                for i, goal in enumerate(goals)}
    last_rows = final_sample_indices(config, 99, len(data["action"]), len(goals))
    with trials.open() as stream:
        failure = next(row for row in csv.DictReader(stream) if row["route"] == "first_hit" and not int(row["success"]))
    memory = torch.zeros(1, dg.shape[1], config.memory_length)
    for state in json.loads(failure["prefix"]):
        memory = observe(memory, dg[state], config.repeat)
    exact = np.all(data["ca3"] == memory.numpy()[0], axis=(1, 2))
    command = env.goal_states.index(int(failure["goal"]))
    chosen = json.loads(failure["actions"])[0]
    matching = np.flatnonzero(exact & (data["action"] == chosen))
    if not len(matching):
        raise ValueError("Recorded witness action lacks exact CA3 replay support")
    successor_memories = data["successor_ca3"][matching]
    if not np.all(successor_memories == successor_memories[0]):
        raise ValueError("This deterministic witness has multiple represented successors")
    successor = torch.as_tensor(successor_memories[:1])
    with torch.no_grad():
        current_q = model(memory, torch.tensor([goals[command]]))[0].numpy()
        next_q = model(successor, torch.tensor([goals[command]]))[0].numpy()
    at = int(matching[0])
    witness = dict(start=int(failure["start"]), physical_goal=int(failure["goal"]), detector_goal=goals[command],
        chosen_action=chosen, full_ca3_matching_rows=int(exact.sum()), exact_input_action_counts={str(a): int(np.sum(exact & (data["action"] == a))) for a in range(env.n_actions)},
        current_q=current_q.tolist(), actual_successor_q=next_q.tolist(),
        current_ca3_nonzero=np.argwhere(memory.numpy()[0] != 0).tolist(),
        successor_ca3_nonzero=np.argwhere(successor.numpy()[0] != 0).tolist(),
        physical_state_unchanged=bool(data["state"][at] == data["next_state"][at]),
        full_ca3_unchanged=bool(np.array_equal(data["ca3"][at], data["successor_ca3"][at])),
        final_target=float(labels[at, command]), final_prediction=float(prediction[at, command]),
        actual_fresh_residual=float(prediction[at, command] - labels[at, command]),
        squared_actual_fresh_residual=float((float(prediction[at, command]) - float(labels[at, command])) ** 2),
        invalid_same_physical_node_shortcut_residual=float((1 - gamma) * current_q[chosen]),
        matching_wait_residuals=residual_metrics(prediction[matching, command], labels[matching, command], usable[matching, command]))
    source_audit = audit_source_snapshots(output / "source_snapshot")
    unchanged = all(digest(path) == sha for path, sha in inputs.items())
    if state_digest(model) != before or not unchanged or not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"]:
        raise RuntimeError("Read-only residual audit changed or lost provenance")
    receipt = dict(schema="intrmotiv/stationary-final-residual-results/v1", protocol_sha256=digest(output / "protocol.json"),
        source_audit=source_audit, inputs_unchanged=unchanged, model_unchanged=True,
        replay_rows=len(data["action"]), goal_ids=list(goals),
        complete_replay=residual_metrics(prediction, labels, usable), per_goal=per_goal,
        same_last_sample_indices_with_final_weights=residual_metrics(prediction[last_rows], labels[last_rows], usable[last_rows]),
        recorded_last_stale_target_minibatch_mse=summary["training_trace"][-1]["first_hit_mse"],
        historical_target_weights_available=False, witness=witness,
        target_generation_diagnostic=dict(refreshes=config.updates // config.target_update_period,
            gamma=gamma, exact_contraction_factor_after_25_backups=gamma ** (config.updates // config.target_update_period),
            finite_horizon=6, interpretation="25 neural target refreshes are not 25 exact Bellman backups. The exact stationary contraction bound would retain up to gamma^25 of initial sup error; finite-H6 exact backups have an acyclic time boundary. Neither fact certifies neural convergence or explains failure uniquely."))
    (output / "results.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.run_directory, args.output)
    print(json.dumps({"complete_replay_mse": receipt["complete_replay"]["mse"],
                      "recorded_last_stale_mse": receipt["recorded_last_stale_target_minibatch_mse"]}))
