"""Exact finite-system tests of fixed-goal first-hit control.

These are tabular mechanism tests, not simulations of IntrMotiv's PPO learner.
The learner receives only independent, reward-free one-step observations. The
environment transition matrix is used solely for executing/evaluating frozen
policies and for an explicitly identified oracle.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FiniteMDP:
    name: str
    transition: np.ndarray  # [state, action, successor]
    terminal: np.ndarray  # terminal *physical* successor, success checked first
    goal_states: tuple[int, ...]
    start_states: tuple[int, ...]

    def __post_init__(self):
        p = np.asarray(self.transition)
        if p.ndim != 3 or p.shape[0] != p.shape[2] or p.shape[1] < 1:
            raise ValueError("Expected transition[state, action, successor]")
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1):
            raise ValueError("Transition probabilities must be finite, nonnegative, normalized")
        if np.asarray(self.terminal).shape != (p.shape[0],):
            raise ValueError("Physical terminal mask must have one entry per state")
        if len(set(self.goal_states)) != len(self.goal_states) or not self.goal_states:
            raise ValueError("Expected distinct goal states")
        if any(s < 0 or s >= p.shape[0] for s in self.goal_states + self.start_states):
            raise ValueError("Goal/start outside state space")

    @property
    def n_states(self):
        return self.transition.shape[0]

    @property
    def n_actions(self):
        return self.transition.shape[1]

    @property
    def goal_detector(self):
        result = np.zeros((self.n_states, len(self.goal_states)), dtype=bool)
        result[list(self.goal_states), np.arange(len(self.goal_states))] = True
        return result


def _deterministic(name, successors, goals, starts, terminals=()):
    successors = np.asarray(successors, dtype=np.int64)
    p = np.zeros((*successors.shape, successors.shape[0]), dtype=np.float64)
    states, actions = np.indices(successors.shape)
    p[states, actions, successors] = 1
    terminal = np.zeros(successors.shape[0], dtype=bool)
    terminal[list(terminals)] = True
    return FiniteMDP(name, p, terminal, tuple(goals), tuple(starts))


def make_fork_cycle():
    """Two branches, reversible routes and self-loops; seven DG identities."""
    return _deterministic(
        "fork_cycle",
        [(1, 2, 0), (3, 0, 1), (0, 4, 2), (5, 1, 3),
         (2, 6, 4), (3, 5, 5), (6, 4, 6)],
        goals=(5, 6), starts=range(7),
    )


def make_stochastic_witness():
    """At H=1 risky succeeds with .1; at H=2 the safe route succeeds with 1."""
    p = np.zeros((4, 2, 4), dtype=np.float64)
    p[0, 0, 2], p[0, 0, 3] = 0.1, 0.9
    p[0, 1, 1] = 1
    p[1, :, 2] = 1
    p[2, :, 2] = 1
    p[3, :, 3] = 1
    return FiniteMDP("stochastic_risky_safe", p, np.array([False, False, True, True]), (2,), (0,))


def make_alias_witness():
    """States zero/one have identical observations and opposite correct actions."""
    return _deterministic(
        "aliased_context", [(2, 3), (3, 2), (2, 2), (3, 3)],
        goals=(2,), starts=(0, 1), terminals=(2, 3),
    )


def collect_one_step_data(env, seed, samples_per_action=100, omit_pairs=()):
    """Sample a stratified reward-free dataset with an independent reset per row.

    Resetting to each nonterminal state is an explicit finite-test capability,
    not assumed to be available in DMLab exploration. No trajectory has two
    actions, so paths longer than one step can only be obtained by composition.
    ``terminated`` is physical termination; ``truncated`` marks collector cuts
    and must NOT be used as a Bellman terminal mask.
    """
    if samples_per_action < 1:
        raise ValueError("samples_per_action must be positive")
    rng = np.random.default_rng(seed)
    omitted = set(omit_pairs)
    rows = []
    for s in range(env.n_states):
        if env.terminal[s]:
            continue
        for a in range(env.n_actions):
            if (s, a) in omitted:
                continue
            for successor in rng.choice(env.n_states, samples_per_action, p=env.transition[s, a]):
                rows.append((s, a, int(successor), bool(env.terminal[successor])))
    rng.shuffle(rows)
    return {
        "state": np.array([r[0] for r in rows], dtype=np.int64),
        "action": np.array([r[1] for r in rows], dtype=np.int64),
        "next_state": np.array([r[2] for r in rows], dtype=np.int64),
        "terminated": np.array([r[3] for r in rows], dtype=bool),
        "truncated": np.ones(len(rows), dtype=bool),
        "episode_id": np.arange(len(rows), dtype=np.int64),
    }


def fit_transition_model(env, data):
    """Empirical transition frequencies; unsupported rows remain exactly zero."""
    count = np.zeros_like(env.transition, dtype=np.int64)
    fields = [np.asarray(data[k]) for k in ("state", "action", "next_state", "terminated")]
    if len({len(v) for v in fields}) != 1:
        raise ValueError("Transition fields have different lengths")
    states, actions, successors, terminated = fields
    if len(states):
        if ((states < 0) | (states >= env.n_states)).any() or ((successors < 0) | (successors >= env.n_states)).any():
            raise ValueError("State outside environment")
        if ((actions < 0) | (actions >= env.n_actions)).any():
            raise ValueError("Action outside environment")
        if not np.array_equal(terminated, env.terminal[successors]):
            raise ValueError("Physical terminal labels disagree with environment")
        if env.terminal[states].any():
            raise ValueError("Physical terminal states must not generate outgoing data")
        np.add.at(count, (states, actions, successors), 1)
    total = count.sum(-1)
    p = np.divide(count, total[..., None], out=np.zeros_like(env.transition), where=total[..., None] > 0)
    return p, total > 0


def _validate_model(env, transition, support):
    p = np.asarray(transition, dtype=np.float64)
    if p.shape != env.transition.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid empirical transition array")
    mask = np.asarray(support, dtype=bool) if support is not None else np.ones(p.shape[:2], dtype=bool)
    if mask.shape != p.shape[:2]:
        raise ValueError("Support mask has wrong shape")
    if not np.allclose(p.sum(-1)[mask], 1):
        raise ValueError("Every supported transition row must sum to one")
    return p, mask


def first_hit_q(env, transition, horizon, support=None, behavior_policy=None):
    """Return Q[remaining, goal, state, action] from empirical one-step data.

    Default backups maximize same-goal success. Supplying a fixed behavior
    policy instead predicts its successor success after the first action;
    greedy deployment of those predictions is a separate comparison.
    Unsupported action values are -inf at positive horizons. Rows with no
    supported action have zero continuation, and are reported in evaluation.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    p, mask = _validate_model(env, transition, support)
    if behavior_policy is not None:
        behavior_policy = np.asarray(behavior_policy, dtype=np.float64)
        if behavior_policy.shape != mask.shape or (behavior_policy < 0).any() or not np.allclose(behavior_policy.sum(-1), 1):
            raise ValueError("Invalid fixed behavior policy")
        if np.any((behavior_policy > 0) & ~mask & ~env.terminal[:, None]):
            raise ValueError("Fixed behavior uses unsupported actions")
    q = np.zeros((horizon + 1, len(env.goal_states), env.n_states, env.n_actions))
    hit = env.goal_detector.T.astype(np.float64)
    live = ~env.terminal
    for h in range(1, horizon + 1):
        if behavior_policy is None:
            value = q[h - 1].max(-1)
            value = np.where(np.isfinite(value), value, 0)
        else:
            values = np.where(np.isfinite(q[h - 1]), q[h - 1], 0)
            value = (values * behavior_policy[None]).sum(-1)
        target = hit + (1 - hit) * live[None] * value
        q[h] = np.einsum("san,gn->gsa", p, target)
        q[h, :, env.terminal, :] = 0
        q[h] = np.where(mask[None], q[h], -np.inf)
    return q


def continuing_manager_q(env, transition, horizon, support=None, manager_goal=0):
    """Explicit control: manager replaces the command after every decision.

    Reward is one when the current command is met; future rewards use a fixed
    manager-selected goal and can recur. This deliberately simple continuing
    objective is a counterexample, NOT a simulation of IntrMotiv's manager.
    """
    p, mask = _validate_model(env, transition, support)
    if not 0 <= manager_goal < len(env.goal_states):
        raise ValueError("manager goal outside vocabulary")
    q = np.zeros((horizon + 1, len(env.goal_states), env.n_states, env.n_actions))
    hit = env.goal_detector.T.astype(np.float64)
    for h in range(1, horizon + 1):
        continuation = q[h - 1, manager_goal].max(-1)
        continuation = np.where(np.isfinite(continuation), continuation, 0)
        target = hit + (~env.terminal)[None] * continuation[None]
        q[h] = np.einsum("san,gn->gsa", p, target)
        q[h, :, env.terminal, :] = 0
        q[h] = np.where(mask[None], q[h], -np.inf)
    return q


def greedy_policy(q):
    """Stable lowest-index tie-breaking; fully unsupported rows choose action 0."""
    actions = np.asarray(q).argmax(-1)
    return np.eye(q.shape[-1], dtype=np.float64)[actions]


def exhaustive_oracle(env, horizon):
    """Independent recursive enumeration of possible successor/action branches."""
    @lru_cache(None)
    def value(h, state, goal):
        if h == 0 or env.terminal[state]:
            return 0.0
        choices = []
        for action in range(env.n_actions):
            total = 0.0
            for successor, probability in enumerate(env.transition[state, action]):
                outcome = 1.0 if successor == goal else value(h - 1, successor, goal)
                total += probability * outcome
            choices.append(total)
        return max(choices)
    return np.array([[value(horizon, s, g) for s in range(env.n_states)] for g in env.goal_states])


def evaluate_goal_policy(env, policy, horizon, support=None):
    """Execute a frozen finite-horizon policy against the true kernel exactly.

    Arrays are [command, evaluated_goal, initial_state]. Every alternative
    command is executed and stops on its own commanded hit, rather than being
    inferred from action differences. Reaching the evaluated target first is
    counted as success; reaching only the commanded alternative is failure for
    that evaluated target. An alternative already achieved at the start stops
    at time zero. Latency
    mass is E[first-hit time * success]; divide by success only for reporting
    conditional latency. Starts already at goal have no automatic t=0 success
    and are separately flagged by the caller.
    """
    policy = np.asarray(policy, dtype=np.float64)
    shape = (horizon + 1, len(env.goal_states), env.n_states, env.n_actions)
    if policy.shape != shape or not np.isfinite(policy).all() or (policy < 0).any() or not np.allclose(policy.sum(-1), 1):
        raise ValueError("Policy must be normalized [remaining, command, state, action]")
    masks = np.ones((env.n_states, env.n_actions), dtype=bool) if support is None else np.asarray(support, bool)
    result = np.zeros((len(env.goal_states), len(env.goal_states), env.n_states))
    latency = np.zeros_like(result)
    unsupported = np.zeros_like(result)
    for command in range(len(env.goal_states)):
        for goal_id, goal in enumerate(env.goal_states):
            hit = np.arange(env.n_states) == goal
            command_hit = np.arange(env.n_states) == env.goal_states[command]
            live = (~hit & ~command_hit & ~env.terminal).astype(float)
            success = np.zeros(env.n_states)
            mass = np.zeros(env.n_states)
            bad = np.zeros(env.n_states)
            for h in range(1, horizon + 1):
                target = hit + live * success
                # Add one decision on every successful trajectory suffix.
                latency_target = hit + live * (mass + success)
                qsuccess = np.einsum("san,n->sa", env.transition, target)
                qlatency = np.einsum("san,n->sa", env.transition, latency_target)
                later_bad = np.einsum("san,n->sa", env.transition, live * bad)
                qbad = np.where(masks, later_bad, 1.0)
                success = (policy[h, command] * qsuccess).sum(-1)
                mass = (policy[h, command] * qlatency).sum(-1)
                bad = (policy[h, command] * qbad).sum(-1)
                success[env.terminal] = mass[env.terminal] = bad[env.terminal] = 0
                if command != goal_id:
                    success[command_hit] = mass[command_hit] = bad[command_hit] = 0
            result[command, goal_id] = success
            latency[command, goal_id] = mass
            unsupported[command, goal_id] = bad
    return {"success": result, "latency_mass": latency, "unsupported_action_probability": unsupported}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(output, theory_path, seeds=(8, 99, 123), samples_per_action=100):
    output, theory_path = Path(output).resolve(), Path(theory_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not theory_path.is_file():
        raise ValueError("Pre-existing theory file is required")
    source_path = Path(__file__).resolve()
    config = {"seeds": seeds, "samples_per_action": samples_per_action,
              "horizons": [1, 2, 4, 8], "collection": "independent stratified one-step resets",
              "method": "empirical tabular finite-horizon dynamic programming",
              "continuing_manager": "retarget to goal zero after EVERY decision; repeated hit reward",
              "evaluation_protocol": "stop at commanded hit; exclude initially-satisfied alternatives from primary lift",
              "no_policy_or_representation_learning_at_evaluation": True}
    before = {"schema": "intrmotiv/first-hit-finite/v1", "config": config,
              "config_sha256": canonical_hash(config),
              "theory_path": str(theory_path), "theory_sha256": hashlib.sha256(theory_path.read_bytes()).hexdigest(),
              "source_path": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_path.parent, text=True).strip()}
    # This receipt is persisted before collecting or fitting any experiment.
    _write_json(output / "prerun.json", before)
    rows, trials, datasets, audits = [], [], [], []
    for env in (make_fork_cycle(), make_stochastic_witness()):
        for seed in seeds:
            variants = [("complete", ())]
            if env.name == "fork_cycle":
                variants.append(("missing_hub_right", ((0, 1),)))
            for support_name, omitted in variants:
                data = collect_one_step_data(env, seed, samples_per_action, omitted)
                data_id = f"{env.name}_{support_name}_s{seed}"
                np.savez_compressed(output / f"{data_id}.npz", **data)
                model, support = fit_transition_model(env, data)
                datasets.append({"dataset": data_id, "payload_sha256": canonical_hash(data),
                                 "npz_sha256": hashlib.sha256((output / f"{data_id}.npz").read_bytes()).hexdigest(),
                                 "rows": len(data["state"]), "max_actions_per_episode": 1,
                                 "support_pairs": int(support.sum()),
                                 "required_nonterminal_pairs": int((~env.terminal).sum() * env.n_actions),
                                 "environment_sha256": canonical_hash({"transition": env.transition, "terminal": env.terminal,
                                                                       "goals": env.goal_states, "starts": env.start_states}),
                                 "omitted_pairs": omitted})
                for horizon in config["horizons"]:
                    first = first_hit_q(env, model, horizon, support)
                    continuing = continuing_manager_q(env, model, horizon, support)
                    uniform = np.ones((env.n_states, env.n_actions)) / env.n_actions
                    covered_uniform = support.astype(float)
                    counts = covered_uniform.sum(-1, keepdims=True)
                    covered_uniform = np.divide(covered_uniform, counts, out=uniform.copy(), where=counts > 0)
                    fixed = first_hit_q(env, model, horizon, support, covered_uniform)
                    policies = {"first_hit": greedy_policy(first),
                                "fixed_policy_successor_greedy": greedy_policy(fixed),
                                "continuing_retarget_each_step": greedy_policy(continuing),
                                "uniform_random": np.broadcast_to(uniform, first.shape).copy()}
                    oracle = exhaustive_oracle(env, horizon)
                    oracle_q = first_hit_q(env, env.transition, horizon)
                    oracle_error = float(np.max(np.abs(oracle_q[horizon].max(-1) - oracle)))
                    if oracle_error > 1e-12:
                        raise AssertionError("Vectorized backup disagrees with independent recursive oracle")
                    audits.append({"environment": env.name, "seed": seed, "support": support_name,
                                   "horizon": horizon, "exact_oracle_max_error": oracle_error})
                    for method, policy in policies.items():
                        evaluation = evaluate_goal_policy(env, policy, horizon, support)
                        for goal_id, goal in enumerate(env.goal_states):
                            for state in env.start_states:
                                for command in range(len(env.goal_states)):
                                    trials.append({"environment": env.name, "dataset": data_id,
                                                   "seed": seed, "support": support_name,
                                                   "horizon": horizon, "method": method,
                                                   "start": state, "evaluated_goal_id": goal_id,
                                                   "commanded_goal_id": command,
                                                   "start_at_evaluated_goal": state == goal,
                                                   "start_at_commanded_goal": state == env.goal_states[command],
                                                   "success_probability": float(evaluation["success"][command, goal_id, state]),
                                                   "success_latency_mass": float(evaluation["latency_mass"][command, goal_id, state])})
                                probability = evaluation["success"][goal_id, goal_id, state]
                                alternatives = [evaluation["success"][command, goal_id, state]
                                                for command in range(len(env.goal_states))
                                                if command != goal_id and state != env.goal_states[command]]
                                alternative = float(np.mean(alternatives)) if alternatives else None
                                rows.append({"environment": env.name, "dataset": data_id, "seed": seed,
                                             "support": support_name, "horizon": horizon, "method": method,
                                             "start": state, "goal_id": goal_id, "goal_state": goal,
                                             "already_at_goal": state == goal,
                                             "oracle_reachable_within_horizon": oracle[goal_id, state] > 0,
                                             "empirical_supported_reachable": first[horizon, goal_id, state].max() > 0,
                                             "success_probability": float(probability),
                                             "oracle_success": float(oracle[goal_id, state]),
                                             "oracle_regret": float(oracle[goal_id, state] - probability),
                                             "alternative_command_success": alternative,
                                             "eligible_alternative_commands": len(alternatives),
                                             "command_lift": None if alternative is None else float(probability - alternative),
                                             "success_conditional_latency": None if probability == 0 else float(evaluation["latency_mass"][goal_id, goal_id, state] / probability),
                                             "unsupported_action_probability": float(evaluation["unsupported_action_probability"][goal_id, goal_id, state])})
    summaries = []
    groups = sorted({(r["environment"], r["support"], r["seed"], r["horizon"], r["method"]) for r in rows})
    for key in groups:
        group = [r for r in rows if (r["environment"], r["support"], r["seed"], r["horizon"], r["method"]) == key]
        eligible = [r for r in group if not r["already_at_goal"]]
        reachable = [r for r in eligible if r["oracle_reachable_within_horizon"]]
        lift = [r["command_lift"] for r in eligible if r["command_lift"] is not None]
        summaries.append(dict(zip(("environment", "support", "seed", "horizon", "method"), key),
                              declared_pairs=len(group), already_at_goal_pairs=len(group)-len(eligible),
                              evaluated_pairs=len(eligible), oracle_reachable_pairs=len(reachable),
                              unsupported_or_unreachable_empirical_pairs=sum(not r["empirical_supported_reachable"] for r in eligible),
                              mean_success=float(np.mean([r["success_probability"] for r in eligible])),
                              worst_pair_success=min(r["success_probability"] for r in eligible),
                              worst_reachable_pair_success=min((r["success_probability"] for r in reachable), default=None),
                              max_oracle_regret=max(r["oracle_regret"] for r in eligible),
                              n_lift_pairs=len(lift),
                              mean_command_lift=float(np.mean(lift)) if lift else None))
    alias = make_alias_witness()
    alias_policies = []
    for action in range(alias.n_actions):
        policy = np.zeros((2, 1, alias.n_states, alias.n_actions))
        policy[..., action] = 1
        values = evaluate_goal_policy(alias, policy, 1)["success"][0, 0, list(alias.start_states)]
        alias_policies.append({"shared_action": action, "context_success": values, "mean_success": float(values.mean())})
    alias_result = {"shared_observation_codes": [0, 0, 1, 2], "contexts": [0, 1],
                    "shared_deterministic_policy_enumeration": alias_policies,
                    "best_shared_mean_success": max(row["mean_success"] for row in alias_policies),
                    "one_bit_context_oracle_success": exhaustive_oracle(alias, 1)[0, :2],
                    "interpretation": "Exact information boundary, not a neural representation-learning result"}
    stochastic = []
    env = make_stochastic_witness()
    for seed in seeds:
        data = collect_one_step_data(env, seed, samples_per_action)
        p, support = fit_transition_model(env, data)
        positive_mask = data["next_state"] == env.goal_states[0]
        positive_data = {k: v[positive_mask] for k, v in data.items()}
        positive_p, positive_support = fit_transition_model(env, positive_data)
        positive_q = first_hit_q(env, positive_p, 2, positive_support)
        evaluation = evaluate_goal_policy(env, greedy_policy(positive_q), 2, positive_support)
        stochastic.append({"seed": seed, "risky_trials": int(((data["state"] == 0) & (data["action"] == 0)).sum()),
                           "risky_successes": int(((data["state"] == 0) & (data["action"] == 0) & positive_mask).sum()),
                           "empirical_risky_success": p[0, 0, 2], "true_risky_success": 0.1,
                           "positive_only_estimated_risky_success": positive_p[0, 0, 2],
                           "positive_only_policy_true_success_h2": evaluation["success"][0, 0, 0],
                           "all_outcomes_policy_true_success_h2": evaluate_goal_policy(env, greedy_policy(first_hit_q(env, p, 2, support)), 2, support)["success"][0, 0, 0],
                           "selection": "one-step hit-only subset; illustration of outcome-conditioning bias, not a full HER implementation"})
    # Same missing-action data are compatible with a world where the omitted
    # hub-right action is a self-loop. Thus its consequence is unidentified.
    env = make_fork_cycle()
    other = env.transition.copy()
    other[0, 1] = 0
    other[0, 1, 0] = 1
    omission_witness = {"omitted_pair": [0, 1], "observed_kernel_max_difference": float(np.max(np.abs(np.delete((env.transition-other).reshape(-1, env.n_states), 1, axis=0)))),
                        "omitted_kernel_total_variation": float(np.abs(env.transition[0, 1]-other[0, 1]).sum()/2),
                        "interpretation": "No offline learner can identify this action from the shared observed rows"}
    _write_csv(output / "per_pair.csv", rows)
    _write_csv(output / "command_trials.csv", trials)
    _write_csv(output / "summary.csv", summaries)
    _write_json(output / "dataset_manifest.json", datasets)
    _write_json(output / "oracle_audits.json", audits)
    _write_json(output / "alias_witness.json", alias_result)
    _write_json(output / "stochastic_witness.json", stochastic)
    _write_json(output / "missing_support_witness.json", omission_witness)
    final = {**before, "datasets": len(datasets), "per_pair_rows": len(rows),
             "summary_rows": len(summaries), "oracle_max_error": max(r["exact_oracle_max_error"] for r in audits),
             "theory_sha256_after": hashlib.sha256(theory_path.read_bytes()).hexdigest(),
             "source_sha256_after": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    if final["theory_sha256_after"] != before["theory_sha256"] or final["source_sha256_after"] != before["source_sha256"]:
        raise RuntimeError("Theory or simulation source changed during execution")
    _write_json(output / "provenance.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theory", type=Path, required=True)
    parser.add_argument("--samples-per-action", type=int, default=100)
    args = parser.parse_args()
    result = run_suite(args.output, args.theory, samples_per_action=args.samples_per_action)
    print(json.dumps({k: result[k] for k in ("datasets", "per_pair_rows", "summary_rows", "oracle_max_error")}, indent=2))


if __name__ == "__main__":
    main()
