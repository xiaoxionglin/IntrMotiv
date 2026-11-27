"""Post-hoc ranking audit of saved failed paths, without new policy rollouts.

Replay only observations already stored in the evaluated path to reconstruct
the exact CA3 input. Saved weights supply scores; graph arrays supply diagnostic
shortest-route labels. No training, action execution, or new environment path
is generated. This is an observational error audit, not a causal ablation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .finite import FiniteMDP, make_fork_cycle
from .neural import NeuralConfig, digest, observe, sparse_context_environment, state_digest
from .provenance import audit_source_snapshots, snapshot_sources
from .stationary_readout import StationaryReadout, original_classes, shortest_distances


def oracle_action_values(env, state, goal_index, gamma, distances):
    successors = env.transition[state].argmax(-1)
    goal = env.goal_states[goal_index]
    return np.asarray([1.0 if nxt == goal else gamma ** distances[goal_index, nxt]
                       if not env.terminal[nxt] else 0.0 for nxt in successors])


@torch.no_grad()
def audit_failed_path(model, row, env, dg, goal_ids, data, config, gamma):
    """Validate a recorded failed path and return its first strict misranking."""
    if row["route"] != "first_hit" or int(row["success"]):
        raise ValueError("This diagnostic accepts only recorded own-command failures")
    prefix, path, actions = [json.loads(row[key]) for key in ("prefix", "states", "actions")]
    goal, start = int(row["goal"]), int(row["start"])
    if not prefix or prefix[-1] != start or path[0] != start or len(path) != len(actions) + 1:
        raise ValueError("Recorded prefix, state, and action alignment failed")
    if goal in path:
        raise ValueError("A failed path already contains its requested goal")
    goal_index = env.goal_states.index(goal)
    distances = shortest_distances(env)
    memory = torch.zeros(1, dg.shape[1], config.memory_length)
    for state in prefix:
        memory = observe(memory, dg[state], config.repeat)
    errors = []
    frozen = state_digest(model)
    for step, action in enumerate(actions):
        state, successor = path[step:step + 2]
        if env.transition[state, action, successor] != 1:
            raise ValueError("Recorded path is not a factual deterministic transition")
        scores = model(memory, torch.tensor([goal_ids[goal_index]]))[0].numpy()
        if int(scores.argmax()) != action:
            raise ValueError("Saved weights do not reproduce this recorded action")
        exact = oracle_action_values(env, state, goal_index, gamma, distances)
        optimal = np.flatnonzero(exact == exact.max())
        successor_memory = observe(memory, dg[successor], config.repeat)
        if action not in optimal:
            successor_scores = model(successor_memory, torch.tensor([goal_ids[goal_index]]))[0].numpy()
            self_label = (1.0 if successor == goal else 0.0 if env.terminal[successor]
                          else gamma * float(successor_scores.max()))
            equal = np.all(data["ca3"] == memory.numpy()[0], axis=(1, 2))
            exact_actions = data["action"][equal]
            successors = env.transition[state].argmax(-1)
            length = [1 if nxt == goal else 1 + distances[goal_index, nxt]
                      if not env.terminal[nxt] else np.inf for nxt in successors]
            errors.append(dict(step=step, state=state, requested_goal=goal, chosen_action=action,
                shortest_actions=optimal.tolist(), predicted_values=scores.tolist(), oracle_values=exact.tolist(),
                predicted_chosen_minus_best_shortest=float(scores[action] - scores[optimal].max()),
                oracle_shortest_minus_chosen=float(exact.max() - exact[action]),
                final_network_self_backup_label=self_label,
                final_network_self_backup_residual=float(scores[action] - self_label),
                shortest_remaining_decisions=int(distances[goal_index, state]),
                chosen_best_total_remaining_decisions=int(length[action]) if np.isfinite(length[action]) else None,
                chosen_successor_is_self_loop=successor == state,
                current_latent_state_previously_seen_in_path=state in path[:step],
                chosen_successor_already_in_path=successor in path[:step + 1],
                full_ca3_exact_replay_rows=int(equal.sum()),
                full_ca3_exact_factual_action_rows=int(np.sum(exact_actions == action)),
                full_ca3_exact_shortest_action_rows={str(a): int(np.sum(exact_actions == a)) for a in optimal},
                latent_states_sharing_exact_replay_ca3=sorted(np.unique(data["state"][equal]).tolist()),
                current_dg=memory[0, :, 0].tolist()))
        memory = successor_memory
    if state_digest(model) != frozen:
        raise RuntimeError("Inference changed saved model state")
    if not errors:
        raise ValueError("A reachable failed path within the declared sufficient deadline has no strict ranking error")
    return dict(start=start, goal=goal, recorded_path=path, recorded_actions=actions,
                recorded_prefix=prefix, path_repeats_latent_state=len(set(path)) != len(path),
                strict_wrong_ranking_steps=len(errors), first_wrong_ranking=errors[0],
                all_recorded_actions_reproduced=True, model_unchanged=True)


def run(run_directory, output):
    root, output = Path(run_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve completed diagnostic artifacts")
    previous_audit = audit_source_snapshots(root / "source_snapshot")
    if not previous_audit["archive_valid"] or not previous_audit["current_sources_unchanged"]:
        raise ValueError("Restore exact fitted-run source bytes before this inference audit")
    manifest_path, results_path = root / "pre_run_manifest.json", root / "results.json"
    manifest, results = [json.loads(path.read_text()) for path in (manifest_path, results_path)]
    if results["pre_run_manifest_sha256"] != digest(manifest_path):
        raise ValueError("Completed results do not bind this preregistration")
    config, gamma = NeuralConfig(**manifest["config"]), manifest["gamma"]
    decoder, action, _ = original_classes()
    fork = make_fork_cycle()
    full = FiniteMDP("fork_all_landmarks", fork.transition, fork.terminal, tuple(range(fork.n_states)), fork.start_states)
    sparse, sparse_dg, sparse_goals = sparse_context_environment()
    environments = {full.name: (full, np.eye(full.n_states, dtype=np.float32), full.goal_states),
                    sparse.name: (sparse, sparse_dg, sparse_goals)}
    inputs = {str(path): digest(path) for path in (manifest_path, results_path)}
    inputs.update(manifest["datasets"])
    for record in results["runs"]:
        folder = root / record["environment"] / f"seed_{record['seed']}"
        checkpoint = folder / "readout.pt"
        if digest(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Saved readout differs from completed run")
        inputs[str(checkpoint)] = record["checkpoint_sha256"]
        inputs[str(folder / "trials.csv")] = digest(folder / "trials.csv")
    if any(digest(path) != sha for path, sha in inputs.items()):
        raise ValueError("Recorded diagnostic input changed before analysis")
    output.mkdir(parents=True)
    sources = [Path(item["source_path"]) for item in manifest["source_snapshot"]["sources"]]
    sources += [Path(__file__), Path(__file__).parents[1] / "test_first_hit_stationary_diagnose.py"]
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    protocol = dict(schema="intrmotiv/stationary-saved-path-audit/v1", created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, source_snapshot=snapshot,
        scope="Every recorded failed own-command path; no new rollouts, training, or parameter selection",
        ranking="First action with strictly worse exact discounted latent-state route value; compare full CA3 byte equality against all recorded replay rows",
        limitations=["Post-hoc observational analysis; selected failure paths are not an unbiased error-rate sample",
            "Final recorded minibatch TD MSE is not a full-replay, held-out, or uniform Bellman residual",
            "No unique attribution to clock, capacity, objective, representation, or optimization",
            "Latent states appear only in diagnostic oracle and stored path checks, never model inputs"])
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(1)
    records = []
    for summary in results["runs"]:
        env, dg, goals = environments[summary["environment"]]
        folder = root / env.name / f"seed_{summary['seed']}"
        dataset = next(Path(path) for path in manifest["datasets"] if
                       Path(path).parent.name == f"seed_{summary['seed']}" and
                       Path(path).parent.parent.name == ("fork_cycle" if env.name == full.name else env.name))
        with np.load(dataset) as saved:
            data = {key: saved[key] for key in ("ca3", "state", "action")}
        model = StationaryReadout(dg.shape[1], config, decoder, action, manifest["decoder_settings"], env.n_actions)
        model.load_state_dict(torch.load(folder / "readout.pt", map_location="cpu", weights_only=True), strict=True)
        model.eval()
        with (folder / "trials.csv").open() as stream:
            failed = [row for row in csv.DictReader(stream) if row["route"] == "first_hit" and not int(row["success"])]
        audits = [audit_failed_path(model, row, env, dg, goals, data, config, gamma) for row in failed]
        records.append(dict(environment=env.name, seed=summary["seed"], failed_paths=len(audits),
                            final_recorded_minibatch_td_mse=summary["training_trace"][-1]["first_hit_mse"], paths=audits))
    source_audit = audit_source_snapshots(output / "source_snapshot")
    unchanged = all(digest(path) == sha for path, sha in inputs.items())
    if not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"] or not unchanged:
        raise RuntimeError("Diagnostic source or data changed during analysis")
    result = dict(schema="intrmotiv/stationary-saved-path-audit-results/v1", runs=records,
        failed_paths=sum(row["failed_paths"] for row in records),
        protocol_sha256=digest(output / "protocol.json"), source_audit=source_audit, inputs_unchanged=unchanged)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_directory, args.output)
    print(json.dumps({"failed_paths_audited": result["failed_paths"], "output": str(args.output)}))
