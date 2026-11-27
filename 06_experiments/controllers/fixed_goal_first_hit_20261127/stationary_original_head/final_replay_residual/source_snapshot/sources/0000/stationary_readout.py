"""Prospective stationary first-hit test using the original action-head class.

Only the legacy MLP and original categorical parameterization are instantiated.
Their raw outputs are regressed as discounted absorbing values; no clock or
auxiliary head enters the network. Data, CA3, input conditioning, deterministic
rollouts, and FIRST event classification reuse the preceding qualification.
The old optimizer cannot be reused unchanged: it always constructs two heads
and jointly trains a continuing loss. The compact loop here accepts an explicit
batch function and trains only the prespecified stationary objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .aggregate_evaluation import first_before_interval
from .finite import FiniteMDP, make_fork_cycle
from .neural import (NeuralConfig, RUNTIME_ROOT, STATE_PATH, SharedReadout, digest,
                     evaluate_readout, evaluation_starts, sparse_context_environment,
                     state_digest, write_csv)
from .provenance import audit_source_snapshots, snapshot_sources
from .runtime_adapter import GoalLayout
from .targets import first_hit_targets


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/minimal_stationary_head_design.md"
DESIGN_SHA256 = "dca22976dd83869f6bb3343357a811911c42a9458e63d57d3b1a5f1ea45eee2f"
DONOR_AUDIT = DESIGN.with_name("stationary_action_head_audit.json")


def original_classes(source_root=RUNTIME_ROOT):
    """Load the exact source classes; do not reconstruct their neural layers."""
    paths = [Path(source_root) / "sf_working_directories/IntrMotiv/dmlab/custom_decoder.py",
             Path(source_root) / "sample_factory/model/action_parameterization.py"]
    classes = []
    for path, name in zip(paths, ("MlpDecoderJit", "ActionParameterizationDefault")):
        spec = importlib.util.spec_from_file_location(f"stationary_original_{name}_{digest(path)[:12]}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes.append(getattr(module, name))
    return (*classes, paths)


class StationaryReadout(SharedReadout):
    """Reuse full-input conditioning, replacing the two heads by the original.

    ``remaining`` exists solely for compatibility with the shared evaluator and
    never enters the decoder or output layer. The goal-zeroed route is an input
    intervention on this same frozen network, not a separately trained policy.
    """

    def __init__(self, feature_count, config, decoder_class, action_class, decoder_settings, action_count=3):
        nn.Module.__init__(self)
        width = feature_count * config.memory_length
        self.layout = GoalLayout(width + feature_count, width, feature_count)
        cfg = SimpleNamespace(**decoder_settings)
        self.decoder = decoder_class(cfg, self.layout.width)
        self.action_parameterization = action_class(cfg, self.decoder.get_out_size(), gym.spaces.Discrete(action_count))

    def forward(self, ca3, goals, remaining=None, route="first_hit"):
        if route not in ("first_hit", "shuffled_goal", "memoryless", "continuing_main"):
            raise ValueError(f"Unknown evaluation route {route}")
        hidden = self.hidden(ca3, goals, remove_history=route == "memoryless", no_goal=route == "continuing_main")
        raw, _ = self.action_parameterization(hidden)
        return raw


def stationary_targets(hits, done, next_online, next_target, gamma=.99):
    """Reuse factual-label/terminal checks with discounted Double-Q continuation.

    Constant 2 is an internal bootstrap-enabled sentinel for the canonical
    target helper, not a clock supplied to either neural network. Success has
    precedence over physical termination. Every successor retains the same g.
    """
    if not 0 < gamma < 1:
        raise ValueError("Stationary contraction requires 0 < gamma < 1")
    sentinel = torch.full((len(hits),), 2, dtype=torch.long, device=hits.device)
    return first_hit_targets(hits, done, sentinel, next_online, gamma * next_target)


def stationary_batch(model, target_model, data, rows, goal_ids, gamma=.99):
    """Use only recorded CA3, action, DG and physical termination, never IDs."""
    goals = torch.as_tensor(goal_ids).repeat(len(rows))
    current_memory = torch.as_tensor(data["ca3"][rows], dtype=torch.float32)
    current = current_memory.repeat_interleave(len(goal_ids), 0)
    successor = torch.as_tensor(data["successor_ca3"][rows], dtype=torch.float32).repeat_interleave(len(goal_ids), 0)
    with torch.no_grad():
        online_next = model(successor, goals).reshape(len(rows), len(goal_ids), -1)
        target_next = target_model(successor, goals).reshape_as(online_next)
        hits = torch.as_tensor(data["successor_dg"][rows][:, goal_ids] > 0)
        done = torch.as_tensor(data["terminated"][rows])
        labels, usable = stationary_targets(hits, done, online_next, target_next, gamma)
        # Initially satisfied commands stop without taking an action. Values
        # for departing from such states are outside the prospective equation.
        usable &= current_memory[:, goal_ids, 0] <= 0
    values = model(current, goals).reshape(len(rows), len(goal_ids), -1)
    actions = torch.as_tensor(data["action"][rows]).long()
    selected = values.gather(-1, actions[:, None, None].expand(-1, len(goal_ids), 1)).squeeze(-1)
    return selected, labels, usable


def fit_stationary(factory, data, goal_ids, config, seed, gamma=.99, *, batch_function=stationary_batch):
    """Fixed-budget learner; batches preserve the earlier sampled-row sequence."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 10000)
    model = factory()
    # Scripted MLP deepcopy can create nonleaf parameters. Construct a fresh
    # target without consuming the learner RNG stream, then strictly copy it.
    with torch.random.fork_rng(devices=[]):
        target = factory()
    target.load_state_dict(model.state_dict(), strict=True)
    target.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    before = time.monotonic()
    trace, used_labels = [], 0
    for update in range(config.updates):
        rows = rng.integers(len(data["action"]), size=config.batch_size)
        # Consume the preceding learner's unused goal/deadline draws solely to
        # match the exact subsequent row batches. Neither is a neural input.
        rng.integers(len(goal_ids), size=config.batch_size)
        rng.integers(1, config.horizon + 1, size=config.batch_size)
        prediction, labels, usable = batch_function(model, target, data, rows, goal_ids, gamma)
        if not usable.any():
            raise ValueError("No noninitial first-hit labels in the sampled batch")
        loss = nn.functional.mse_loss(prediction[usable], labels[usable])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite stationary Bellman loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        used_labels += int(usable.sum())
        if (update + 1) % config.target_update_period == 0:
            target.load_state_dict(model.state_dict())
        if update % 250 == 0 or update == config.updates - 1:
            trace.append({"update": update + 1, "first_hit_mse": float(loss.detach())})
    model.eval()
    return model, {"updates": config.updates, "training_seconds": time.monotonic() - before,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "action_head_parameters": sum(p.numel() for p in model.action_parameterization.parameters()),
                   "enumerated_goal_labels": config.updates * config.batch_size * len(goal_ids),
                   "used_noninitial_goal_labels": used_labels, "continuing_loss_labels": 0,
                   "replay_transitions": len(data["action"]), "training_trace": trace}


def shortest_distances(env):
    """Independent deterministic oracle by reverse shortest-path relaxation."""
    if not np.allclose(env.transition.max(-1), 1):
        raise ValueError("Shortest-path diagnostics require deterministic dynamics")
    successor = env.transition.argmax(-1)
    distance = np.full((len(env.goal_states), env.n_states), np.inf)
    for index, goal in enumerate(env.goal_states):
        distance[index, goal] = 0
        for _ in range(env.n_states):
            previous = distance[index].copy()
            for state in range(env.n_states):
                if state != goal and not env.terminal[state]:
                    distance[index, state] = min(previous[state], 1 + previous[successor[state]].min())
            if np.array_equal(previous, distance[index]):
                break
    return distance


@torch.no_grad()
def stationary_calibration(model, env, dg, goal_ids, config, gamma):
    """Finite declared-start diagnostics; not a global approximation certificate."""
    distances = shortest_distances(env)
    rows = []
    for state, memory, _ in evaluation_starts(env, dg, config):
        for index, goal in enumerate(env.goal_states):
            if state == goal:
                continue
            prediction = model(memory, torch.tensor([goal_ids[index]]))[0].numpy()
            for action, next_state in enumerate(env.transition[state].argmax(-1)):
                truth = (1.0 if next_state == goal else
                         gamma ** distances[index, next_state] if not env.terminal[next_state] else 0.0)
                rows.append(dict(start=state, goal=int(goal), action=action, prediction=float(prediction[action]),
                                 exact=float(truth), error=float(prediction[action] - truth)))
    return rows


def competing_first_from_trials(trials, goals, horizon, route="first_hit"):
    """Apply canonical FIRST classifier to two actual own-command trajectories."""
    own = {(int(row["start"]), int(row["goal"])): row for row in trials if row["route"] == route}
    result = []
    for source in sorted({key[0] for key in own}):
        for goal in goals:
            for competitor in goals:
                if goal == competitor or source in (goal, competitor):
                    continue
                events = []
                for command in (goal, competitor):
                    row = own[source, command]
                    path = json.loads(row["states"])
                    scalar = lambda target: dict(issued=True, hit_time=path.index(target) if target in path else None,
                                                  censored=False, elapsed=int(row["decisions"]))
                    events.append(first_before_interval(scalar(goal), scalar(competitor), horizon))
                if any(low != high for low, high in events):
                    raise ValueError("Fully observed deterministic toy comparisons must be identified")
                result.append(dict(source=source, goal=int(goal), competitor=int(competitor),
                                   own_event=events[0][0], alternative_event=events[1][0],
                                   contrast=events[0][0] - events[1][0]))
    return result


def summarize_trials(trials, env, config):
    own = [row for row in trials if row["route"] == "first_hit"]
    distances = shortest_distances(env)
    goal_index = {goal: index for index, goal in enumerate(env.goal_states)}
    for row in trials:
        row["shortest_distance"] = int(distances[goal_index[row["goal"]], row["start"]])
        row["shortest_success"] = int(row["success"] and row["decisions"] == row["shortest_distance"])
    first = competing_first_from_trials(trials, env.goal_states, config.horizon)
    memoryless = competing_first_from_trials(trials, env.goal_states, config.horizon, "memoryless")
    initial_count = sum(not env.terminal[state] and state in env.goal_states for state in range(env.n_states)) if env.name != "sparse_context" else 0
    # t0 stops are analytic and not executed: shared evaluator excludes them.
    return dict(n_nontrivial_pairs=len(own), initially_satisfied_queries=initial_count,
                declared_queries=len(own) + initial_count,
                first_hit_success=float(np.mean([r["success"] for r in own])),
                worst_pair_success=min(r["success"] for r in own),
                all_pair_gate_passed=all(r["success"] >= config.pair_success_gate for r in own),
                shortest_success_fraction=float(np.mean([r["shortest_success"] for r in own])),
                mean_decisions_nontrivial_queries=float(np.mean([r["decisions"] for r in own])),
                mean_shortest_distance=float(np.mean([r["shortest_distance"] for r in own])),
                n_competing_comparisons=len(first), competing_first_contrast=float(np.mean([r["contrast"] for r in first])),
                memoryless_competing_first_contrast=float(np.mean([r["contrast"] for r in memoryless])),
                per_goal={str(goal): dict(queries=sum(r["goal"] == goal for r in own),
                    success=float(np.mean([r["success"] for r in own if r["goal"] == goal])),
                    shortest_success=float(np.mean([r["shortest_success"] for r in own if r["goal"] == goal])))
                    for goal in env.goal_states},
                per_route_success={route: float(np.mean([r["success"] for r in trials if r["route"] == route]))
                    for route in ("first_hit", "shuffled_goal", "continuing_main", "memoryless")}), first


def run(output, original, legacy, theory_sha256=DESIGN_SHA256):
    output, original, legacy = map(lambda p: Path(p).resolve(), (output, original, legacy))
    if output.exists():
        raise FileExistsError("Refusing to overwrite a stationary run")
    if digest(DESIGN) != theory_sha256:
        raise ValueError("Prospective theory note changed before the declared run")
    config = replace(NeuralConfig(), methods=("stationary_replay_td",))
    gamma = .99
    old = json.loads((legacy / "pre_run_manifest.json").read_text())
    settings = old["decoder_settings"]
    keys = ("seeds", "horizon", "memory_length", "repeat", "hidden_size", "transitions", "updates", "batch_size", "learning_rate", "target_update_period")
    current_config = json.loads(json.dumps(asdict(config)))
    if any(current_config[key] != old["config"][key] for key in keys):
        raise ValueError("Stationary budgets differ from the declared legacy baseline")
    decoder, action, class_paths = original_classes()
    donor_audit = json.loads(DONOR_AUDIT.read_text())
    for relative, expected in donor_audit["source_files"].items():
        if digest(RUNTIME_ROOT / relative) != expected["donor_sha256"] or expected["donor_sha256"] != expected["r3_sha256"]:
            raise ValueError("Local original class source differs from certified donor")
    parent = make_fork_cycle()
    full = FiniteMDP("fork_all_landmarks", parent.transition, parent.terminal, tuple(range(parent.n_states)), parent.start_states)
    sparse, sparse_dg, sparse_goals = sparse_context_environment()
    experiments = [(full, np.eye(full.n_states, dtype=np.float32), full.goal_states, "fork_cycle"),
                   (sparse, sparse_dg, sparse_goals, "sparse_context")]
    datasets = {}
    original_manifest = json.loads((original / "pre_run_manifest.json").read_text())
    for _, _, _, source in experiments:
        for seed in config.seeds:
            path = original / source / f"seed_{seed}" / "reward_free_trajectories.npz"
            if digest(path) != original_manifest["datasets"][str(path)]:
                raise ValueError("Cached factual dataset differs from original manifest")
            datasets[str(path)] = digest(path)
    comparisons = {str(path): digest(path) for path in legacy.glob("*/seed_*/trials.csv")}
    if len(comparisons) != 6:
        raise ValueError("All six frozen finite-horizon comparisons are required")
    output.mkdir(parents=True)
    source_names = ("stationary_readout.py", "neural.py", "finite.py", "runtime_adapter.py", "targets.py",
                    "provenance.py", "aggregate_evaluation.py", "dmlab_evaluate.py")
    sources = [Path(__file__).with_name(name) for name in source_names] + class_paths + [STATE_PATH, DESIGN, DONOR_AUDIT, legacy / "pre_run_manifest.json"]
    sources += [RUNTIME_ROOT / "sample_factory/algo/utils/action_distributions.py",
                RUNTIME_ROOT / "sample_factory/model/model_utils.py", ROOT / "hpc_runs/test_first_hit_stationary.py"]
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    manifest = dict(schema="intrmotiv/stationary-original-head/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        config=asdict(config), gamma=gamma, decoder_settings=settings, source_snapshot=snapshot, datasets=datasets,
        theory_sha256=theory_sha256, donor_audit_sha256=digest(DONOR_AUDIT), finite_horizon_comparisons=comparisons,
        source_classes=[decoder.__name__, action.__name__], action_count=3, donor_action_count=5,
        training="Factual-action all-goal stationary Double-Q MSE; no continuing loss; initially satisfied commands excluded",
        sampling="Same row batches as legacy trainer, including consumption of unused goal/deadline RNG draws; no neural clock input",
        controls="All own commands; cyclic alternate; same network with memory zeroed or goal input zeroed; no additional fitted heads",
        gates="Every declared nontrivial pair >=.9 success in each seed; report latency and all ordered competing FIRST; no tuning",
        differences_from_legacy=["Stationary discounted objective replaces finite deadline objective", "No auxiliary head or neural clock",
            "No jointly optimized continuing loss", "Current already-satisfied goal rows omitted from regression"],
        limitations=["Fresh toy weights using actual classes, not donor checkpoint transfer; toy3 versus native5 actions",
            "Fork all7 identities; sparse two target identities plus two cue identities",
            "Finite deterministic graphs: fork fully observed, sparse task history conditioned; no DMLab or stochastic probability-optimality claim",
            "Sparse evaluation has fresh cues; finite CA3 expiry remains a limit", "Training budgets match updates/batches, not FLOPs or active labels"])
    (output / "pre_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(1)
    results = []
    for env, dg, goals, source in experiments:
        for seed in config.seeds:
            folder = output / env.name / f"seed_{seed}"
            folder.mkdir(parents=True)
            with np.load(original / source / f"seed_{seed}" / "reward_free_trajectories.npz") as loaded:
                data = {key: loaded[key] for key in loaded.files if key != "dg_observations"}
            factory = lambda: StationaryReadout(dg.shape[1], config, decoder, action, settings, env.transition.shape[1])
            model, stats = fit_stationary(factory, data, goals, config, seed, gamma)
            checkpoint = folder / "readout.pt"
            torch.save(model.state_dict(), checkpoint)
            before = state_digest(model)
            trials = evaluate_readout(model, env, dg, goals, config, seed)
            summary, first = summarize_trials(trials, env, config)
            calibrated = stationary_calibration(model, env, dg, goals, config, gamma)
            unchanged = before == state_digest(model)
            reloaded = factory()
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            reloaded.eval()
            repeated = evaluate_readout(reloaded, env, dg, goals, config, seed)
            summarize_trials(repeated, env, config)
            if not unchanged or repeated != trials:
                raise RuntimeError("Frozen evaluation or saved-weight reproduction failed")
            write_csv(folder / "trials.csv", trials)
            write_csv(folder / "competing_first.csv", first)
            write_csv(folder / "calibration.csv", calibrated)
            import csv
            with (legacy / env.name / f"seed_{seed}" / "trials.csv").open() as stream:
                old_trials = list(csv.DictReader(stream))
            old_own = [r for r in old_trials if r["route"] == "first_hit"]
            old_first = competing_first_from_trials(old_trials, env.goal_states, config.horizon)
            legacy_stats = dict(success=float(np.mean([int(r["success"]) for r in old_own])),
                mean_decisions_nontrivial_queries=float(np.mean([int(r["decisions"]) for r in old_own])),
                shortest_success_fraction=float(np.mean([int(r["success"]) and int(r["decisions"]) ==
                    shortest_distances(env)[env.goal_states.index(int(r["goal"])), int(r["start"])] for r in old_own])),
                competing_first_contrast=float(np.mean([r["contrast"] for r in old_first])))
            record = dict(environment=env.name, seed=seed, **stats, **summary,
                source_head_state_keys=list(model.action_parameterization.state_dict()),
                positive_successor_exposures=(data["successor_dg"][:, goals] > 0).sum(0).tolist(),
                initial_goal_rows_masked_per_goal=(data["ca3"][:, goals, 0] > 0).sum(0).tolist(),
                sampled_initial_goal_labels_omitted=stats["enumerated_goal_labels"] - stats["used_noninitial_goal_labels"],
                maximum_start_query_absolute_calibration_error=max(abs(r["error"]) for r in calibrated),
                checkpoint_sha256=digest(checkpoint), evaluation_state_hash_unchanged=unchanged,
                saved_checkpoint_trials_exactly_reproduced=True, finite_horizon_comparison=legacy_stats,
                route_aliases={"continuing_main": "same stationary network with goal input zeroed; no continuing training"})
            (folder / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            results.append(record)
            print(json.dumps({key: record[key] for key in ("environment", "seed", "first_hit_success", "shortest_success_fraction", "competing_first_contrast", "training_seconds")}), flush=True)
    result = dict(schema="intrmotiv/stationary-original-head-results/v1", runs=results,
        all_pair_gates_passed=all(r["all_pair_gate_passed"] for r in results),
        pre_run_manifest_sha256=digest(output / "pre_run_manifest.json"),
        source_audit=audit_source_snapshots(output / "source_snapshot"),
        datasets_unchanged=all(digest(path) == sha for path, sha in datasets.items()),
        comparisons_unchanged=all(digest(path) == sha for path, sha in comparisons.items()))
    if not result["source_audit"]["archive_valid"] or not result["source_audit"]["current_sources_unchanged"] or not result["datasets_unchanged"] or not result["comparisons_unchanged"]:
        raise RuntimeError("Declared source or input bytes changed during the six fits")
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.original, args.legacy)
