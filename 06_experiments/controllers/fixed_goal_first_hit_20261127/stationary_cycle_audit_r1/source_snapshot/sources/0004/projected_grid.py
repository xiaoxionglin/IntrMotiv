"""Fixed paired projected/unprojected bootstrap test on a fresh GeoMem grid.

The canonical environment is imported from an exact source snapshot. Collection
uses its actual step method, including wall self-loops. Only bootstrap target
values differ between conditions; no inference module or input is added.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from .finite import FiniteMDP
from .neural import NeuralConfig, RUNTIME_ROOT, STATE_PATH, digest, evaluate_readout, observe, state_digest, write_csv
from .provenance import audit_source_snapshots, snapshot_sources
from .stationary_readout import (StationaryReadout, original_classes, fit_stationary,
                                stationary_batch, stationary_calibration, summarize_trials)


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/projected_bootstrap_design.md"
DESIGN_SHA = "959a03bce0acffd7e95a58e1392b3854086034a88dca94ead52986e130315e64"
ENVIRONMENT = Path("/home/xiaoxiong/Desktop/Projects/geomem/geomem-env/src/geomem_env/environment.py")
ENVIRONMENT_SHA = "6354015e2960e9170c97f37851cce9124b6f49d97f2fde17301f84406b653015"
CONDITIONS = ("unprojected", "projected")
SETTINGS = dict(decoder_mlp_layers=[128, 128], nonlinearity="relu", use_jit=True)
GAMMA = .99


def fixed_config():
    return replace(NeuralConfig(), horizon=16, methods=CONDITIONS)


class ClippedTarget(nn.Module):
    """Training-only target view; online selection and deployed outputs stay raw."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).clamp(0, 1)


def projected_batch(model, target_model, data, rows, goal_ids, gamma=GAMMA):
    # stationary_batch applies gamma AFTER this projection and selects the
    # target action using the unchanged raw online output.
    return stationary_batch(model, ClippedTarget(target_model), data, rows, goal_ids, gamma)


class BatchAudit:
    """Record actual initialization, sampled rows and usable masks during fit."""
    def __init__(self, condition):
        if condition not in CONDITIONS:
            raise ValueError("Unknown declared condition")
        self.batch = projected_batch if condition == "projected" else stationary_batch
        self.rows, self.masks = hashlib.sha256(), hashlib.sha256()
        self.initial = None
        self.calls = 0

    def __call__(self, model, target, data, rows, goals, gamma):
        if self.initial is None:
            self.initial = state_digest(model)
            if self.initial != state_digest(target):
                raise ValueError("Initial online and target parameters differ")
        self.rows.update(np.asarray(rows, dtype=np.int64).tobytes())
        prediction, labels, usable = self.batch(model, target, data, rows, goals, gamma)
        self.masks.update(usable.detach().cpu().numpy().tobytes())
        self.calls += 1
        return prediction, labels, usable

    def summary(self):
        return dict(initial_parameter_sha256=self.initial, batch_rows_sha256=self.rows.hexdigest(),
                    usable_masks_sha256=self.masks.hexdigest(), actual_batch_calls=self.calls)


def archived_path(output, manifest, original):
    entries = [e for e in manifest["source_snapshot"]["sources"] if e["source_path"] == str(Path(original).resolve())]
    if len(entries) != 1:
        raise ValueError("Required source is not uniquely archived")
    path = output / "source_snapshot" / entries[0]["snapshot_path"]
    if digest(path) != entries[0]["sha256"]:
        raise ValueError("Archived source bytes changed")
    return path


def load_environment(path):
    """Import the archived canonical file, registering dataclass type metadata."""
    if digest(path) != ENVIRONMENT_SHA:
        raise ValueError("GeoMem environment source differs from prospective design")
    name = f"projected_grid_geomem_{ENVIRONMENT_SHA[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.clamped_lattice(width=4, height=4)


def finite_view(environment):
    """Exact transition table for shared rollout/oracle code, never TD targets."""
    states, actions = environment.states, environment.actions
    index = {state: i for i, state in enumerate(states)}
    if len(states) != 16 or len(actions) != 4:
        raise ValueError("The prospective grid has exactly sixteen nodes/four actions")
    transition = np.zeros((16, 4, 16))
    for state in states:
        for action_id, action in enumerate(actions):
            transition[index[state], action_id, index[environment.step(state, action)]] = 1
    goals = tuple(range(16))
    return FiniteMDP("geomem_clamped_4x4", transition, np.zeros(16, bool), goals, goals)


def collect_replay(environment, config, seed):
    """Uniform actions and uniform reset nodes; every cut is a collector cut."""
    rng = np.random.default_rng(seed)
    states, actions = environment.states, environment.actions
    index = {state: i for i, state in enumerate(states)}
    dg = np.eye(len(states), dtype=np.float32)
    arrays = {k: [] for k in ("state", "next_state", "action", "ca3", "successor_ca3", "successor_dg",
                              "terminated", "truncated", "episode_id")}
    while len(arrays["action"]) < config.transitions:
        episode = len(arrays["action"]) // config.episode_limit
        state = states[int(rng.integers(len(states)))]
        memory = observe(torch.zeros(1, len(states), config.memory_length), dg[index[state]], config.repeat)
        for step in range(config.episode_limit):
            action = int(rng.integers(len(actions)))
            successor = environment.step(state, actions[action])
            successor_memory = observe(memory, dg[index[successor]], config.repeat)
            cut = step == config.episode_limit - 1 or len(arrays["action"]) == config.transitions - 1
            values = dict(state=index[state], next_state=index[successor], action=action,
                ca3=memory.numpy()[0].copy(), successor_ca3=successor_memory.numpy()[0].copy(),
                successor_dg=dg[index[successor]].copy(), terminated=False, truncated=cut, episode_id=episode)
            for key, value in values.items():
                arrays[key].append(value)
            state, memory = successor, successor_memory
            if len(arrays["action"]) == config.transitions:
                break
    return {key: np.asarray(values) for key, values in arrays.items()}


def source_paths():
    names = ("projected_grid.py", "stationary_readout.py", "neural.py", "finite.py", "runtime_adapter.py",
             "targets.py", "provenance.py", "aggregate_evaluation.py", "dmlab_evaluate.py")
    paths = [Path(__file__).with_name(name) for name in names]
    paths += [ENVIRONMENT, DESIGN, STATE_PATH, ROOT / "hpc_runs/test_first_hit_projected_grid.py",
              DESIGN.with_name("stationary_action_head_audit.json")]
    paths += [RUNTIME_ROOT / path for path in (
        "sf_working_directories/IntrMotiv/dmlab/custom_decoder.py", "sample_factory/model/action_parameterization.py",
        "sample_factory/model/model_utils.py", "sample_factory/model/decoder.py",
        "sample_factory/algo/utils/action_distributions.py", "sample_factory/algo/utils/torch_utils.py")]
    return paths


def prepare(output):
    """Archive source/protocol BEFORE data; save data hashes BEFORE any fit."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    if digest(DESIGN) != DESIGN_SHA or digest(ENVIRONMENT) != ENVIRONMENT_SHA:
        raise ValueError("Frozen design/environment changed")
    output.mkdir(parents=True)
    snapshot = snapshot_sources(source_paths(), output / "source_snapshot")
    config = fixed_config()
    manifest = dict(schema="intrmotiv/projected-grid-protocol/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        config=asdict(config), gamma=GAMMA, decoder_settings=SETTINGS, source_snapshot=snapshot,
        design_sha256=DESIGN_SHA, environment_sha256=ENVIRONMENT_SHA,
        data_status="not_constructed", environment="GeoMem clamped_lattice(width4,height4); actual Environment.step",
        observation="Fixed synthetic16-wayDG, exact CA3L8R1; fresh toy weights, no learned visual fields or donor transfer",
        conditions=list(CONDITIONS), contrast="Only projected condition clamps target network output to[0,1] before gamma; raw online selection unchanged",
        controls="Both conditions omit initially satisfied labels, use only stationary first-hit loss, and share exact initial tensors/data/batches",
        outcomes="240nontrivial queries+16analytic initial stops;3360ordered competingFIRST contrasts; everypair>=.9 perseed; reportworstgoal+shortest",
        evaluation="Shared immutable evaluator additionally retains its fixed memoryless/goalzero/cyclic-command annotations; no extra fitted models",
        fixed_final_weights=True, early_stopping=False, hyperparameter_sweep=False,
        native_environment_runs=False, numerical_threads=1,
        runtime_versions=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__))
    (output / "pre_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # The environment is loaded from archive only after this manifest exists.
    environment = load_environment(archived_path(output, manifest, ENVIRONMENT))
    env = finite_view(environment)
    datasets = {}
    for seed in config.seeds:
        data = collect_replay(environment, config, seed)
        path = output / f"reward_free_seed_{seed}.npz"
        np.savez_compressed(path, **data)
        exposure = np.zeros((env.n_states, env.n_actions), int)
        np.add.at(exposure, (data["state"], data["action"]), 1)
        datasets[str(seed)] = dict(path=str(path), sha256=digest(path), transitions=len(data["action"]),
            node_action_counts=exposure.tolist(), node_action_pairs_covered=int((exposure > 0).sum()),
            physical_done_count=int(data["terminated"].sum()), collector_cut_count=int(data["truncated"].sum()),
            self_loop_rows=int((data["state"] == data["next_state"]).sum()))
    prepared = dict(schema="intrmotiv/projected-grid-prepared/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        pre_data_manifest_sha256=digest(output / "pre_data_manifest.json"), datasets=datasets,
        source_audit=audit_source_snapshots(output / "source_snapshot"))
    if not prepared["source_audit"]["archive_valid"] or not prepared["source_audit"]["current_sources_unchanged"]:
        raise RuntimeError("Source changed during data preparation")
    (output / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


@torch.no_grad()
def fresh_residuals(model, data, goals, condition, gamma=GAMMA):
    """Final weights on BOTH sides of each backup; replay diagnostics, not holdout."""
    batch = projected_batch if condition == "projected" else stationary_batch
    collected = {key: [] for key in ("predictions", "targets", "mask")}
    for start in range(0, len(data["action"]), 256):
        rows = np.arange(start, min(start + 256, len(data["action"])))
        prediction, target, mask = batch(model, model, data, rows, goals, gamma)
        for key, value in zip(collected, (prediction, target, mask)):
            collected[key].append(value.cpu().numpy())
    prediction, target, mask = (np.concatenate(collected[k]) for k in collected)
    residual = prediction - target
    def stats(selected):
        p, t, e = prediction[selected], target[selected], residual[selected]
        return dict(labels=int(selected.sum()), mse=float(np.mean(e**2)), maximum_absolute_residual=float(np.abs(e).max()),
            prediction_min=float(p.min()), prediction_max=float(p.max()), target_min=float(t.min()), target_max=float(t.max()),
            predictions_above_one=int((p > 1).sum()), predictions_below_zero=int((p < 0).sum()))
    by_goal = {}
    for column, goal in enumerate(goals):
        selected = np.zeros_like(mask)
        selected[:, column] = mask[:, column]
        by_goal[str(goal)] = stats(selected)
    return dict(operator=condition, same_final_model_on_both_sides=True, scope="all cached factual replay; current-goal labels omitted",
                overall=stats(mask), per_goal=by_goal)


def run_prepared(output):
    output = Path(output).resolve()
    protocol = json.loads((output / "pre_data_manifest.json").read_text())
    prepared = json.loads((output / "prepared.json").read_text())
    if prepared["pre_data_manifest_sha256"] != digest(output / "pre_data_manifest.json"):
        raise ValueError("Pre-data protocol changed")
    config = fixed_config()
    if (json.loads(json.dumps(asdict(config))) != protocol["config"] or protocol["design_sha256"] != DESIGN_SHA
            or protocol["gamma"] != GAMMA or protocol["decoder_settings"] != SETTINGS):
        raise ValueError("Fixed settings changed")
    audit = audit_source_snapshots(output / "source_snapshot")
    if not audit["archive_valid"] or not audit["current_sources_unchanged"]:
        raise ValueError("Frozen fit sources differ")
    for dataset in prepared["datasets"].values():
        if digest(Path(dataset["path"])) != dataset["sha256"]:
            raise ValueError("Prepared data changed")
    with (output / "fit_started.json").open("x") as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(), prepared_sha256=digest(output / "prepared.json")), stream, indent=2)
    torch.set_num_threads(1)
    environment = load_environment(archived_path(output, protocol, ENVIRONMENT))
    env, goals, dg = finite_view(environment), tuple(range(16)), np.eye(16, dtype=np.float32)
    decoder, action, _ = original_classes()
    factory = lambda: StationaryReadout(16, config, decoder, action, SETTINGS, action_count=4)
    records, pair_checks = [], []
    for seed in config.seeds:
        with np.load(prepared["datasets"][str(seed)]["path"], allow_pickle=False) as loaded:
            data = {key: loaded[key].copy() for key in loaded.files}
        pair = []
        for condition in CONDITIONS:
            folder = output / f"seed_{seed}" / condition
            folder.mkdir(parents=True)
            batch_audit = BatchAudit(condition)
            model, training = fit_stationary(factory, data, goals, config, seed, GAMMA, batch_function=batch_audit)
            checkpoint = folder / "readout.pt"
            torch.save(model.state_dict(), checkpoint)
            before = state_digest(model)
            trials = evaluate_readout(model, env, dg, goals, config, seed)
            summary, first = summarize_trials(trials, env, config)
            if summary["n_nontrivial_pairs"] != 240 or summary["declared_queries"] != 256 or len(first) != 3360:
                raise ValueError("Declared complete query denominator changed")
            calibration = stationary_calibration(model, env, dg, goals, config, GAMMA)
            residuals = {operator: fresh_residuals(model, data, goals, operator) for operator in CONDITIONS}
            reloaded = factory()
            reloaded.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
            reloaded.eval()
            repeated = evaluate_readout(reloaded, env, dg, goals, config, seed)
            summarize_trials(repeated, env, config)
            if before != state_digest(model) or trials != repeated:
                raise ValueError("Frozen model or saved-weight replay audit failed")
            write_csv(folder / "trials.csv", trials)
            write_csv(folder / "competing_first.csv", first)
            write_csv(folder / "calibration.csv", calibration)
            (folder / "fresh_residuals.json").write_text(json.dumps(residuals, indent=2, allow_nan=False) + "\n")
            record = dict(condition=condition, seed=seed, **training, **summary, **batch_audit.summary(),
                worst_goal_success=min(g["success"] for g in summary["per_goal"].values()),
                maximum_fresh_start_value_error=max(abs(row["error"]) for row in calibration),
                maximum_value_error_scope="960 action values on declared fresh-current-observation starts only; not all CA3 histories",
                checkpoint_sha256=digest(checkpoint), dataset_sha256=prepared["datasets"][str(seed)]["sha256"],
                evaluation_state_hash_unchanged=True, saved_checkpoint_trials_exactly_reproduced=True,
                own_operator_fresh_residual=residuals[condition]["overall"])
            (folder / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            records.append(record); pair.append(record)
            print(json.dumps({k: record[k] for k in ("seed", "condition", "first_hit_success", "worst_goal_success",
                             "shortest_success_fraction", "competing_first_contrast", "training_seconds")}), flush=True)
        keys = ("initial_parameter_sha256", "batch_rows_sha256", "usable_masks_sha256", "used_noninitial_goal_labels", "dataset_sha256")
        if any(pair[0][key] != pair[1][key] for key in keys):
            raise ValueError("Paired initialization/data/batches/usable labels differ")
        pair_checks.append(dict(seed=seed, matched_keys=list(keys), all_equal=True,
                                projected_minus_unprojected_success=pair[1]["first_hit_success"]-pair[0]["first_hit_success"]))
    result = dict(schema="intrmotiv/projected-grid-results/v1", runs=records, paired_checks=pair_checks,
        pre_data_manifest_sha256=digest(output / "pre_data_manifest.json"), prepared_sha256=digest(output / "prepared.json"),
        source_audit=audit_source_snapshots(output / "source_snapshot"),
        datasets_unchanged=all(digest(Path(d["path"])) == d["sha256"] for d in prepared["datasets"].values()),
        all_pair_gate_by_condition={condition: all(r["all_pair_gate_passed"] for r in records if r["condition"] == condition)
                                   for condition in CONDITIONS},
        interpretation="Six fixed fits only; fresh separately-trained fully observed graph; syntheticDG; no policy transfer, hidden-topology generalization or DMLab claim")
    if not result["source_audit"]["archive_valid"] or not result["source_audit"]["current_sources_unchanged"] or not result["datasets_unchanged"]:
        raise RuntimeError("Frozen source or data changed during the six fits")
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
