"""Declared post-hoc projected-target regression; never retrain old references."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .finite import FiniteMDP, make_fork_cycle
from .neural import (NeuralConfig, digest, evaluate_readout,
                     sparse_context_environment, state_digest, write_csv)
from .projected_grid import BatchAudit, GAMMA, fresh_residuals
from .provenance import audit_source_snapshots, snapshot_sources
from .stationary_readout import StationaryReadout, fit_stationary, original_classes, stationary_calibration, summarize_trials


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "06_experiments/controllers/fixed_goal_first_hit_20261127"
PROTOCOL = STUDY / "runtime/projected_regression_protocol.json"
PROTOCOL_SHA = "addba931cb6918beb03531b0682735f78eebd1016169bd72f327e9452e43b3f0"
REFERENCE = STUDY / "stationary_original_head"


def regression_config():
    return replace(NeuralConfig(), methods=("stationary_replay_td",))


def experiments():
    parent = make_fork_cycle()
    fork = FiniteMDP("fork_all_landmarks", parent.transition, parent.terminal,
                     tuple(range(parent.n_states)), parent.start_states)
    sparse, dg, goals = sparse_context_environment()
    return ((fork, np.eye(fork.n_states, dtype=np.float32), fork.goal_states, "fork_cycle"),
            (sparse, dg, goals, "sparse_context"))


class InitialAudit(BatchAudit):
    """Save actual NEW initial tensors at the first pre-update batch callback."""
    def __init__(self, destination):
        super().__init__("projected")
        self.destination = Path(destination)

    def __call__(self, model, target, data, rows, goals, gamma):
        if self.initial is None:
            torch.save(model.state_dict(), self.destination)
        return super().__call__(model, target, data, rows, goals, gamma)


def reconstruct_sampling(data, goals, config, seed):
    """Reconstruct old code/seed draws; not an observed historical hash.

    The known finite replay has genuine successors and no action restrictions,
    so the usable-label mask is exactly the current-goal exclusion mask.
    """
    rng = np.random.default_rng(seed + 10000)
    rows_hash, masks_hash = hashlib.sha256(), hashlib.sha256()
    for _ in range(config.updates):
        rows = rng.integers(len(data["action"]), size=config.batch_size)
        rng.integers(len(goals), size=config.batch_size)
        rng.integers(1, config.horizon + 1, size=config.batch_size)
        rows_hash.update(np.asarray(rows, dtype=np.int64).tobytes())
        masks_hash.update((data["ca3"][rows][:, goals, 0] <= 0).tobytes())
    return dict(batch_rows_sha256=rows_hash.hexdigest(), usable_masks_sha256=masks_hash.hexdigest())


def validate_reference():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Declared post-hoc protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for relative, sha in protocol["inputs"].items():
        if digest(ROOT / relative) != sha:
            raise ValueError(f"Declared reference changed: {relative}")
    old = json.loads((REFERENCE / "pre_run_manifest.json").read_text())
    config = regression_config()
    if json.loads(json.dumps(asdict(config))) != old["config"] or old["gamma"] != GAMMA:
        raise ValueError("Historical training settings differ")
    # Relevant original decoder, model construction, fitting and state update
    # code must still match the historical source archive byte for byte.
    required = {"stationary_readout.py", "neural.py", "finite.py", "runtime_adapter.py", "targets.py",
                "state.py", "custom_decoder.py", "action_parameterization.py", "model_utils.py",
                "action_distributions.py"}
    matched = {}
    for entry in old["source_snapshot"]["sources"]:
        original = Path(entry["source_path"])
        if original.name in required:
            if digest(original) != entry["sha256"]:
                raise ValueError(f"Historical fit source differs: {original}")
            matched[str(original)] = entry["sha256"]
    if not required.issubset({Path(path).name for path in matched}):
        raise ValueError("Historical source record lacks required reconstruction evidence")
    return protocol, old, matched


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new regression output directory")
    protocol, old, matched = validate_reference()
    datasets = {}
    for _, _, _, source in experiments():
        for seed in regression_config().seeds:
            path = STUDY / "neural" / source / f"seed_{seed}" / "reward_free_trajectories.npz"
            if digest(path) != old["datasets"][str(path)]:
                raise ValueError("Historical factual replay changed")
            datasets[f"{source}:{seed}"] = dict(path=str(path), sha256=digest(path))
    sources = [Path(__file__), ROOT / "hpc_runs/test_first_hit_projected_regression.py", PROTOCOL,
               Path(__file__).with_name("projected_grid.py"), Path(__file__).with_name("provenance.py"),
               Path(__file__).with_name("aggregate_evaluation.py"), Path(__file__).with_name("dmlab_evaluate.py")]
    sources += [Path(path) for path in matched]
    sources += [ROOT / relative for relative in protocol["inputs"]]
    sources = list(dict.fromkeys(sources))
    output.mkdir(parents=True)
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    prepared = dict(schema="intrmotiv/projected-posthoc-prepared/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=PROTOCOL_SHA, source_snapshot=snapshot, config=asdict(regression_config()), gamma=GAMMA,
        decoder_settings=old["decoder_settings"], datasets=datasets, historical_source_hashes_matched=matched,
        reference_results_sha256=digest(REFERENCE / "results.json"), new_fit_count=6,
        matching_scope="New initial tensors/rows/masks observed. Historical matching reconstructed from unchanged code/config/data/seed, not recorded historical hashes; assumes same library RNG semantics.",
        runtime_versions=dict(torch=str(torch.__version__), numpy=np.__version__),
        interpretation="Post-hoc regression after fresh-grid validation; no independent validation or selection claim")
    (output / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def run_prepared(output):
    output = Path(output).resolve()
    prepared = json.loads((output / "prepared.json").read_text())
    _, old, _ = validate_reference()
    audit = audit_source_snapshots(output / "source_snapshot")
    if not audit["archive_valid"] or not audit["current_sources_unchanged"]:
        raise ValueError("Frozen regression source changed")
    config = regression_config()
    if json.loads(json.dumps(asdict(config))) != prepared["config"]:
        raise ValueError("Fixed regression settings changed")
    for dataset in prepared["datasets"].values():
        if digest(Path(dataset["path"])) != dataset["sha256"]:
            raise ValueError("Historical replay changed after preparation")
    with (output / "fit_started.json").open("x") as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(), prepared_sha256=digest(output / "prepared.json")), stream)
    references = json.loads((REFERENCE / "results.json").read_text())["runs"]
    decoder, action, _ = original_classes()
    torch.set_num_threads(1)
    records = []
    for env, dg, goals, source in experiments():
        for seed in config.seeds:
            dataset = prepared["datasets"][f"{source}:{seed}"]
            with np.load(dataset["path"], allow_pickle=False) as loaded:
                data = {key: loaded[key].copy() for key in loaded.files if key != "dg_observations"}
            folder = output / env.name / f"seed_{seed}"
            folder.mkdir(parents=True)
            factory = lambda: StationaryReadout(dg.shape[1], config, decoder, action, old["decoder_settings"], action_count=3)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                reconstructed_initial_hash = state_digest(factory())
            sampling = reconstruct_sampling(data, goals, config, seed)
            batch_audit = InitialAudit(folder / "initial_readout.pt")
            model, training = fit_stationary(factory, data, goals, config, seed, GAMMA, batch_function=batch_audit)
            actual = batch_audit.summary()
            if actual["initial_parameter_sha256"] != reconstructed_initial_hash or any(actual[k] != v for k, v in sampling.items()):
                raise ValueError("New observed initialization/batches/masks differ from reference-code reconstruction")
            checkpoint = folder / "readout.pt"
            torch.save(model.state_dict(), checkpoint)
            before = state_digest(model)
            trials = evaluate_readout(model, env, dg, goals, config, seed)
            summary, first = summarize_trials(trials, env, config)
            expected = 42 if env.name == "fork_all_landmarks" else 4
            if summary["n_nontrivial_pairs"] != expected or len(first) != (210 if expected == 42 else 4):
                raise ValueError("Post-hoc query denominator changed")
            calibration = stationary_calibration(model, env, dg, goals, config, GAMMA)
            residuals = {operator: fresh_residuals(model, data, goals, operator) for operator in ("unprojected", "projected")}
            reloaded = factory()
            reloaded.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
            reloaded.eval()
            repeated = evaluate_readout(reloaded, env, dg, goals, config, seed)
            summarize_trials(repeated, env, config)
            if before != state_digest(model) or trials != repeated:
                raise ValueError("Frozen/saved-weight evaluation failed")
            baseline = [r for r in references if r["environment"] == env.name and r["seed"] == seed]
            if len(baseline) != 1 or training["used_noninitial_goal_labels"] != baseline[0]["used_noninitial_goal_labels"]:
                raise ValueError("Historical usable-label budget differs")
            write_csv(folder / "trials.csv", trials)
            write_csv(folder / "competing_first.csv", first)
            write_csv(folder / "calibration.csv", calibration)
            (folder / "fresh_residuals.json").write_text(json.dumps(residuals, indent=2, allow_nan=False) + "\n")
            record = dict(environment=env.name, seed=seed, condition="projected_posthoc", **training, **summary, **actual,
                worst_goal_success=min(g["success"] for g in summary["per_goal"].values()),
                maximum_start_query_absolute_calibration_error=max(abs(r["error"]) for r in calibration),
                reference_unprojected={key: baseline[0][key] for key in ("first_hit_success", "shortest_success_fraction",
                    "competing_first_contrast", "per_goal", "checkpoint_sha256", "used_noninitial_goal_labels")},
                historical_initial_match="reconstructed from code/config/seed, not observed historical hash",
                historical_batch_match="reconstructed RNG consumption/current-goal mask, not observed historical hash",
                reconstructed_initial_parameter_sha256=reconstructed_initial_hash,
                reconstructed_sampling=sampling, reconstruction_matches_new_observed=True,
                checkpoint_sha256=digest(checkpoint), initial_checkpoint_sha256=digest(folder / "initial_readout.pt"),
                dataset_sha256=dataset["sha256"], evaluation_state_hash_unchanged=True,
                saved_checkpoint_trials_exactly_reproduced=True,
                own_operator_fresh_residual=residuals["projected"]["overall"])
            (folder / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            records.append(record)
            print(json.dumps({key: record[key] for key in ("environment", "seed", "first_hit_success",
                "worst_goal_success", "shortest_success_fraction", "competing_first_contrast", "training_seconds")}), flush=True)
    result = dict(schema="intrmotiv/projected-posthoc-results/v1", scope="Explicitly post-hoc regression, not independent validation",
        runs=records, all_six_gates_passed=all(r["all_pair_gate_passed"] for r in records),
        protocol_sha256=PROTOCOL_SHA, prepared_sha256=digest(output / "prepared.json"),
        source_audit=audit_source_snapshots(output / "source_snapshot"),
        datasets_unchanged=all(digest(Path(d["path"])) == d["sha256"] for d in prepared["datasets"].values()),
        historical_reference_unchanged=digest(REFERENCE / "results.json") == prepared["reference_results_sha256"])
    if not result["source_audit"]["archive_valid"] or not result["source_audit"]["current_sources_unchanged"] or not result["datasets_unchanged"] or not result["historical_reference_unchanged"]:
        raise RuntimeError("Regression source/data/reference bytes changed")
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    result = prepare(args.output) if args.prepare_only else run_prepared(args.output)
    print(json.dumps({"output": str(args.output), "schema": result["schema"]}), flush=True)
