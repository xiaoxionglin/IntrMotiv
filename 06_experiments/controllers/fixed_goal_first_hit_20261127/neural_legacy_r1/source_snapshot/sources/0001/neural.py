"""Frozen-CA3 readout qualification on reward-free finite trajectories.

This separates empirical-value distillation from actual factual-replay TD.
Physical state IDs are used only by distillation and evaluation; the TD targets
use stored CA3, action, successor DG, and physical termination. Inputs are frozen
synthetic DG observations processed by the unchanged CA3 update. The actual runtime
TargetFiLMDecoder and ControllerQHeads are loaded, not reimplemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .finite import FiniteMDP, first_hit_q, fit_transition_model, make_fork_cycle
from .runtime_adapter import GoalLayout, fixed_goal_input, runtime_factories
from .targets import first_hit_targets


ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "exports/intrmotiv_prediction_starter/intrmotiv_transfer/state.py"
THEORY_PATH = ROOT / "02_algorithm/fixed_goal_first_hit_control.md"
RUNTIME_ROOT = Path("/home/xiaoxiong/SFgit/SF_hipposlam")


def load_ca3_update():
    spec = importlib.util.spec_from_file_location("first_hit_exact_ca3_state", STATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.advance_ca3


advance_ca3 = load_ca3_update()


@dataclass(frozen=True)
class NeuralConfig:
    seeds: tuple[int, ...] = (8, 99, 123)
    horizon: int = 6
    memory_length: int = 8
    repeat: int = 1
    hidden_size: int = 128
    transitions: int = 4096
    episode_limit: int = 32
    updates: int = 2500
    batch_size: int = 256
    learning_rate: float = 0.003
    continuing_discount: float = 0.9
    pair_success_gate: float = 0.9
    methods: tuple[str, ...] = ("empirical_value_distillation", "replay_td")
    target_update_period: int = 100


def sparse_context_environment():
    """Equal blank observations require different actions after distinct cues."""
    next_states = np.array([[2, 2, 2], [3, 3, 3], [4, 5, 2],
                            [5, 4, 3], [4, 4, 4], [5, 5, 5]])
    transition = np.eye(6)[next_states]
    env = FiniteMDP("sparse_context", transition, np.array([0, 0, 0, 0, 1, 1], dtype=bool),
                    (4, 5), (0, 1))
    dg = np.zeros((6, 4), dtype=np.float32)
    dg[0, 0] = dg[1, 1] = dg[4, 2] = dg[5, 3] = 1
    return env, dg, (2, 3)


def experiment_environments():
    env = make_fork_cycle()
    yield env, np.eye(len(env.terminal), dtype=np.float32), tuple(env.goal_states), "distinct_dg_positive_control"
    env, dg, goal_ids = sparse_context_environment()
    yield env, dg, goal_ids, "history_required_sparse_dg"


def observe(memory, observation, repeat):
    with torch.no_grad():
        return advance_ca3(memory, torch.as_tensor(observation)[None], repeat)


def collect_trajectory_data(env, dg, seed, config):
    """Collect uniform random actions before any target goal is assigned."""
    rng = np.random.default_rng(seed)
    arrays = {name: [] for name in ("state", "action", "next_state", "terminated", "episode_id", "ca3", "successor_ca3", "successor_dg")}
    episode = 0
    while len(arrays["state"]) < config.transitions:
        state = int(rng.choice(env.start_states))
        memory = observe(torch.zeros(1, dg.shape[1], config.memory_length), dg[state], config.repeat)
        for _ in range(config.episode_limit):
            action = int(rng.integers(env.transition.shape[1]))
            successor = int(rng.choice(len(env.terminal), p=env.transition[state, action]))
            successor_memory = observe(memory, dg[successor], config.repeat)
            for key, value in (("state", state), ("action", action), ("next_state", successor),
                               ("terminated", bool(env.terminal[successor])), ("episode_id", episode),
                               ("ca3", memory.numpy()[0].copy()), ("successor_ca3", successor_memory.numpy()[0].copy()),
                               ("successor_dg", dg[successor].copy())):
                arrays[key].append(value)
            state = successor
            memory = successor_memory
            if env.terminal[state] or len(arrays["state"]) >= config.transitions:
                break
        episode += 1
    return {name: np.asarray(values) for name, values in arrays.items()}


def continuing_values(env, transition, discount):
    """Synthetic negative: collect any goal event, continuing across commands.

    This is an explicit goal-independent reward control, not a recreation of
    historical IntrMotiv bonuses. Scale by (1-gamma) for bounded regression.
    """
    reward = np.isin(np.arange(len(env.terminal)), env.goal_states).astype(float)
    values = np.zeros(len(env.terminal))
    for _ in range(1000):
        q = np.einsum("sak,k->sa", transition, (1 - discount) * reward + discount * (~env.terminal) * values)
        updated = q.max(-1)
        if np.max(np.abs(updated - values)) < 1e-12:
            break
        values = updated
    return q


class SharedReadout(nn.Module):
    """Two existing heads, one unchanged runtime FiLM; routing is explicit."""
    def __init__(self, feature_count, config, decoder_factory, heads_factory):
        super().__init__()
        width = feature_count * config.memory_length
        core = SimpleNamespace(Hippo_n_feature=feature_count, target_condition_start=width,
                               get_out_size=lambda: width + feature_count,
                               cfg=SimpleNamespace(fixed_task_goal_mixture=False))
        self.decoder = decoder_factory(core, hidden_size=config.hidden_size)
        self.heads = heads_factory(config.hidden_size, 3, auxiliary=True)
        self.layout = GoalLayout(width + feature_count, width, feature_count)
        self.horizon = config.horizon

    def hidden(self, ca3, goals, remove_history=False, no_goal=False):
        if remove_history:
            ca3 = ca3.clone()
            ca3[:, :, 1:] = 0
        flat = ca3.flatten(1)
        complete = torch.cat((flat, flat.new_zeros(len(flat), self.layout.goals)), -1)
        conditioned = fixed_goal_input(complete, goals, self.layout)
        if no_goal:
            conditioned[:, self.layout.goal_start:] = 0
        return self.decoder(conditioned)

    def forward(self, ca3, goals, remaining, route="first_hit"):
        hidden = self.hidden(ca3, goals, remove_history=route == "memoryless", no_goal=route == "continuing_main")
        if route == "continuing_main":
            return self.heads(hidden)
        values = self.heads.hindsight(hidden, remaining, self.horizon)
        return torch.where(remaining[:, None] > 0, values, torch.zeros_like(values))


def train_readout(env, dg, goal_ids, data, config, seed, factories, method):
    """Fit empirical Bellman table values; never update on evaluation trials."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 10000)
    model = SharedReadout(dg.shape[1], config, *factories)
    transition, support = fit_transition_model(env, data)
    target = first_hit_q(env, transition, config.horizon, support)
    main_target = continuing_values(env, transition, config.continuing_discount)
    memory = torch.as_tensor(data["ca3"], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # TorchScript MLP parameters can become nonleaf tensors under deepcopy.
    # Recreate the same architecture and copy values, preserving the caller's
    # random stream; this also gives an independent, explicitly frozen target.
    with torch.random.fork_rng(devices=[]):
        target_model = SharedReadout(dg.shape[1], config, *factories)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    target_model.requires_grad_(False)
    traces = []
    before = time.monotonic()
    for update in range(config.updates):
        rows = rng.integers(len(data["state"]), size=config.batch_size)
        goals = rng.integers(len(goal_ids), size=config.batch_size)
        remaining = rng.integers(1, config.horizon + 1, size=config.batch_size)
        identifiers = torch.tensor(np.asarray(goal_ids)[goals])
        clocks = torch.tensor(remaining)
        if method == "empirical_value_distillation":
            desired = torch.tensor(target[remaining, goals, data["state"][rows]], dtype=torch.float32)
            q = model(memory[rows], identifiers, clocks)
            main_q = model(memory[rows], identifiers, clocks, "continuing_main")
            expected_main = torch.tensor(main_target[data["state"][rows]], dtype=torch.float32)
            loss_hit = nn.functional.mse_loss(q, desired)
            loss_main = nn.functional.mse_loss(main_q, expected_main)
        elif method == "replay_td":
            # Only factual stored representations, actions, detector events and
            # physical termination enter these targets. No state IDs or table
            # values from the empirical-model upper bound are read here.
            ids = torch.tensor(goal_ids).repeat(config.batch_size)
            current = memory[rows].repeat_interleave(len(goal_ids), 0)
            successor = torch.tensor(data["successor_ca3"][rows]).repeat_interleave(len(goal_ids), 0)
            expanded_clocks = clocks.repeat_interleave(len(goal_ids))
            with torch.no_grad():
                online_next = model(successor, ids, expanded_clocks - 1).reshape(config.batch_size, len(goal_ids), -1)
                target_next = target_model(successor, ids, expanded_clocks - 1).reshape_as(online_next)
                hits = torch.tensor(data["successor_dg"][rows][:, goal_ids] > 0)
                done = torch.tensor(data["terminated"][rows])
                desired, usable = first_hit_targets(hits, done, clocks, online_next, target_next)
                if not usable.all():
                    raise ValueError("Synthetic recorded transitions must have certified successors")
                next_main_online = model(successor[::len(goal_ids)], identifiers, clocks - 1, "continuing_main")
                next_main_target = target_model(successor[::len(goal_ids)], identifiers, clocks - 1, "continuing_main")
                greedy = next_main_online.argmax(-1, keepdim=True)
                expected_main = ((1 - config.continuing_discount) * hits.any(-1).float()
                                 + config.continuing_discount * (~done).float() * next_main_target.gather(1, greedy).flatten())
            actions = torch.tensor(data["action"][rows])
            q = model(current, ids, expanded_clocks).reshape(config.batch_size, len(goal_ids), -1)
            selected = q.gather(-1, actions[:, None, None].expand(-1, len(goal_ids), 1)).squeeze(-1)
            main_q = model(memory[rows], identifiers, clocks, "continuing_main").gather(1, actions[:, None]).flatten()
            loss_hit = nn.functional.mse_loss(selected, desired)
            loss_main = nn.functional.mse_loss(main_q, expected_main)
        else:
            raise ValueError(f"Unknown method: {method}")
        loss = loss_hit + loss_main
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (update + 1) % config.target_update_period == 0:
            target_model.load_state_dict(model.state_dict())
        if update % 250 == 0 or update == config.updates - 1:
            traces.append({"update": update + 1, "first_hit_mse": float(loss_hit.detach()), "continuing_mse": float(loss_main.detach())})
    model.eval()
    stats = {"training_seconds": time.monotonic() - before, "parameters": sum(p.numel() for p in model.parameters()),
             "updates": config.updates, "regression_action_targets_first_hit": config.updates * config.batch_size * (3 if method == "empirical_value_distillation" else len(goal_ids)),
             "regression_action_targets_continuing": config.updates * config.batch_size * (3 if method == "empirical_value_distillation" else 1),
             "replay_transitions": len(data["state"]), "supported_state_actions": int(support.sum()),
             "training_trace": traces}
    return model, target, stats


def evaluation_starts(env, dg, config):
    if env.name == "sparse_context":
        for source, state in ((0, 2), (1, 3)):
            memory = observe(torch.zeros(1, dg.shape[1], config.memory_length), dg[source], config.repeat)
            yield state, observe(memory, dg[state], config.repeat), [source, state]
    else:
        for state in range(len(env.terminal)):
            if not env.terminal[state]:
                yield state, observe(torch.zeros(1, dg.shape[1], config.memory_length), dg[state], config.repeat), [state]


@torch.no_grad()
def evaluate_readout(model, env, dg, goal_ids, config, seed):
    rows = []
    for route in ("first_hit", "shuffled_goal", "continuing_main", "memoryless"):
        for initial, initial_memory, prefix in evaluation_starts(env, dg, config):
            for goal_index, target_state in enumerate(env.goal_states):
                if initial == target_state:
                    continue  # Trivial initial-goal cases are counted separately.
                state, memory = initial, initial_memory.clone()
                command = (goal_index + 1) % len(goal_ids) if route == "shuffled_goal" else goal_index
                action_trace = []
                states = [state]
                hit = False
                first_distinct_goal = -1
                for remaining in range(config.horizon, 0, -1):
                    if state == env.goal_states[command]:
                        break
                    outputs = model(memory, torch.tensor([goal_ids[command]]), torch.tensor([remaining]), route)
                    action = int(outputs.argmax(-1))
                    # Environments in this learned-decoder qualification are deterministic.
                    probability = env.transition[state, action]
                    if not np.isclose(probability.max(), 1):
                        raise ValueError("Neural deterministic qualification cannot silently sample stochastic transitions")
                    state = int(probability.argmax())
                    states.append(state)
                    action_trace.append(action)
                    memory = observe(memory, dg[state], config.repeat)
                    if state in env.goal_states and first_distinct_goal == -1:
                        first_distinct_goal = int(state)
                    if state == target_state:
                        hit = True
                        break
                    if state == env.goal_states[command]:
                        # An alternative command stops when IT is reached;
                        # never continue a controller after its own stop event.
                        break
                    if env.terminal[state]:
                        break
                rows.append({"environment": env.name, "seed": seed, "route": route, "start": initial,
                             "goal": int(target_state), "success": int(hit), "decisions": len(action_trace),
                             "start_at_commanded_goal": int(initial == env.goal_states[command]),
                             "prefix": json.dumps(prefix), "states": json.dumps(states), "actions": json.dumps(action_trace),
                             "first_distinct_goal": first_distinct_goal})
    return rows


@torch.no_grad()
def calibration(model, env, dg, goal_ids, config, target):
    rows = []
    for state, memory, _ in evaluation_starts(env, dg, config):
        for goal_index, goal in enumerate(env.goal_states):
            if state == goal:
                continue
            for remaining in range(1, config.horizon + 1):
                prediction = model(memory, torch.tensor([goal_ids[goal_index]]), torch.tensor([remaining]))[0].numpy()
                exact = target[remaining, goal_index, state]
                for action, (value, truth) in enumerate(zip(prediction, exact)):
                    rows.append({"start": state, "goal": int(goal), "remaining": remaining, "action": action,
                                 "prediction": float(value), "empirical_exact": float(truth), "error": float(value - truth)})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def audit_saved_runs(output):
    """Replay saved policies; verify weights, datasets and archived provenance."""
    output = Path(output).resolve()
    manifest = json.loads((output / "pre_run_manifest.json").read_text())
    result = json.loads((output / "results.json").read_text())
    config_values = manifest["config"]
    config_values["seeds"] = tuple(config_values["seeds"])
    config_values["methods"] = tuple(config_values["methods"])
    config = NeuralConfig(**config_values)
    torch.set_num_threads(1)
    factories = runtime_factories(RUNTIME_ROOT)[:2]
    environments = {env.name: (env, dg, goal_ids) for env, dg, goal_ids, _ in experiment_environments()}
    checks = []
    for run_record in result["runs"]:
        env, dg, goal_ids = environments[run_record["environment"]]
        folder = output / env.name / f"seed_{run_record['seed']}" / run_record["method"]
        model = SharedReadout(dg.shape[1], config, *factories)
        model.load_state_dict(torch.load(folder / "readout.pt", map_location="cpu", weights_only=True))
        model.eval()
        before = state_digest(model)
        actual = evaluate_readout(model, env, dg, goal_ids, config, run_record["seed"])
        expected = list(csv.DictReader((folder / "trials.csv").open()))
        equal = len(actual) == len(expected) and all({key: str(value) for key, value in a.items()} == b for a, b in zip(actual, expected))
        checks.append({"environment": env.name, "seed": run_record["seed"], "method": run_record["method"],
                       "checkpoint_sha256": digest(folder / "readout.pt"), "state_sha256_before": before,
                       "state_sha256_after": state_digest(model), "saved_trials_exactly_reproduced": equal})
    dataset_checks = {name: digest(name) == expected for name, expected in manifest["datasets"].items()}
    source_checks, drift = {}, {}
    for name, expected in manifest["source_sha256"].items():
        archived = output / "source_snapshot" / (expected[:12] + "_" + Path(name).name)
        source_checks[name] = archived.exists() and digest(archived) == expected
        if digest(name) != expected:
            drift[name] = {"pre_run_sha256": expected, "current_sha256": digest(name), "exact_archived_source": str(archived)}
    expired = {}
    for seed in config.seeds:
        with np.load(output / "sparse_context" / f"seed_{seed}" / "reward_free_trajectories.npz") as data:
            blank = np.isin(data["state"], (2, 3))
            absent = (data["ca3"][:, :2].sum((1, 2)) == 0) & blank
            expired[str(seed)] = {"expired_cue_rows": int(absent.sum()), "blank_state_rows": int(blank.sum())}
    audit = {"schema": "intrmotiv/first-hit-neural-audit/v1", "saved_checkpoint_replays": checks,
             "dataset_checks": dataset_checks, "exact_archived_source_checks": source_checks,
             "source_drift_receipt": drift, "sparse_context_expired_cue_rows": expired,
             "all_passed": all(dataset_checks.values()) and all(source_checks.values())
                           and all(c["saved_trials_exactly_reproduced"] and c["state_sha256_before"] == c["state_sha256_after"] for c in checks),
             "clarifications": ["Original manifest claim text predates adding replay_td; methods and recorded code identify actual TD versus distillation.",
                                "Continuing-main negative removes the command AND changes the objective; it is not a routing-only causal contrast.",
                                "Distillation and TD have identical optimizer steps but different label/action supervision counts, recorded per run.",
                                "Primary sparse starts have fresh cues and H6<E8. Expired-cue replay observations can be irreducibly aliased; no universal memory sufficiency is claimed."]}
    (output / "replay_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def run(output, config=NeuralConfig()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pre_run_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a preregistered run directory")
    torch.set_num_threads(1)
    decoder_factory, heads_factory, runtime = runtime_factories(RUNTIME_ROOT)
    files = [Path(__file__), Path(__file__).with_name("finite.py"), Path(__file__).with_name("runtime_adapter.py"), Path(__file__).with_name("targets.py"), STATE_PATH, THEORY_PATH]
    manifest = {"schema": "intrmotiv/first-hit-neural/v1", "config": asdict(config), "runtime": runtime,
                "source_sha256": {str(path): digest(path) for path in files},
                "claim": "Supervised empirical finite-horizon value approximation with actual runtime FiLM/heads; not online TD convergence or DMLab proof",
                "controls": "Same trained decoder and data; first-hit auxiliary deployed versus independently trained continuing main with zero command; shuffled command and memoryless interventions",
                "evaluation_updates": 0, "neural_input": "Exact CA3 from synthetic frozen DG observations plus original target onehot and deadline; no physical node IDs"}
    prepared = []
    for env, dg, goal_ids, role in experiment_environments():
        for seed in config.seeds:
            folder = output / env.name / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            data = collect_trajectory_data(env, dg, seed, config)
            path = folder / "reward_free_trajectories.npz"
            np.savez_compressed(path, **data, dg_observations=dg)
            prepared.append((env, dg, goal_ids, role, seed, folder, data))
    manifest["datasets"] = {str(item[5] / "reward_free_trajectories.npz"): digest(item[5] / "reward_free_trajectories.npz") for item in prepared}
    (output / "pre_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    all_rows, summaries = [], []
    runs = [(env, dg, goal_ids, role, seed, folder / method, data, method)
            for env, dg, goal_ids, role, seed, folder, data in prepared for method in config.methods]
    for env, dg, goal_ids, role, seed, folder, data, method in runs:
        folder.mkdir(parents=True, exist_ok=True)
        model, target, stats = train_readout(env, dg, goal_ids, data, config, seed, (decoder_factory, heads_factory), method)
        rows = evaluate_readout(model, env, dg, goal_ids, config, seed)
        calibrated = calibration(model, env, dg, goal_ids, config, target)
        write_csv(folder / "trials.csv", rows)
        write_csv(folder / "deadline_calibration.csv", calibrated)
        torch.save(model.state_dict(), folder / "readout.pt")
        first = [row for row in rows if row["route"] == "first_hit"]
        shuffled = [row for row in rows if row["route"] == "shuffled_goal"]
        informative = [(a, b) for a, b in zip(first, shuffled) if not b["start_at_commanded_goal"]]
        summary = {"environment": env.name, "role": role, "seed": seed, "method": method, **stats,
                   "per_route_success": {route: float(np.mean([row["success"] for row in rows if row["route"] == route])) for route in ("first_hit", "shuffled_goal", "continuing_main", "memoryless")},
                   "worst_pair_success": min(row["success"] for row in first),
                   "paired_command_lift": float(np.mean([a["success"] - b["success"] for a, b in informative])),
                   "n_lift_pairs": len(informative),
                   "n_nontrivial_pairs": len(first),
                   "initial_goal_trials_excluded": sum(state in env.goal_states for state, _, _ in evaluation_starts(env, dg, config)),
                   "maximum_absolute_calibration_error": max(abs(row["error"]) for row in calibrated),
                   "pair_gate_passed": all(row["success"] >= config.pair_success_gate for row in first)}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        all_rows.extend([{**row, "method": method} for row in rows])
        summaries.append(summary)
        print(json.dumps({key: summary[key] for key in ("environment", "seed", "method", "per_route_success", "pair_gate_passed", "training_seconds")}), flush=True)
    write_csv(output / "all_trials.csv", all_rows)
    result = {"schema": "intrmotiv/first-hit-neural-results/v1", "pre_run_manifest_sha256": digest(output / "pre_run_manifest.json"),
              "all_pair_gates_passed": all(row["pair_gate_passed"] for row in summaries), "runs": summaries,
              "limitations": ["Synthetic DG identities are fixed, not learned image representations.",
                              "Empirical-value distillation uses hidden state indices to look up upper-bound targets; replay_td uses only factual stored transitions and learned bootstrap values.",
                              "All observed state-goal combinations receive synthetic labels; this is not held-out-goal generalization.",
                              "Existing deadline head is affine in remaining time at fixed state/goal; arbitrary finite-horizon values are not exactly representable.",
                              "Continuing-main control uses a declared synthetic goal-independent reward, not historical IntrMotiv bonus reconstruction."]}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps({"all_passed": audit_saved_runs(args.output)["all_passed"]}))
    else:
        run(args.output)


if __name__ == "__main__":
    main()
