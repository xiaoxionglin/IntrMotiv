"""Prospective all-seven-DG scope extension of the frozen-CA3 replay-TD test.

Reuses the exact saved reward-free data, trainer, environment, and budgets from
the two-target qualification. Only the eligible target vocabulary expands. This
is not compute-matched: all-goal labels increase from two to seven per row.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from .finite import FiniteMDP, make_fork_cycle
from .neural import (
    NeuralConfig, RUNTIME_ROOT, STATE_PATH, THEORY_PATH, SharedReadout,
    calibration, digest, evaluate_readout, state_digest, train_readout, write_csv,
)
from .runtime_adapter import runtime_factories
from .provenance import audit_source_snapshots, snapshot_sources


def run(output, original):
    output, original = Path(output).resolve(), Path(original).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pre_run_manifest.json").exists():
        raise FileExistsError("All-landmark extension is immutable; choose a new output path")
    config = replace(NeuralConfig(), methods=("replay_td",))
    torch.set_num_threads(1)
    decoder, heads, runtime = runtime_factories(RUNTIME_ROOT)
    parent = make_fork_cycle()
    env = FiniteMDP("fork_all_landmarks", parent.transition, parent.terminal, tuple(range(parent.n_states)), parent.start_states)
    dg = np.eye(env.n_states, dtype=np.float32)
    goal_ids = tuple(range(env.n_states))
    original_manifest = json.loads((original / "pre_run_manifest.json").read_text())
    paths = {seed: original / "fork_cycle" / f"seed_{seed}" / "reward_free_trajectories.npz" for seed in config.seeds}
    for path in paths.values():
        if digest(path) != original_manifest["datasets"][str(path)]:
            raise ValueError("Parent reward-free dataset failed its recorded hash")
    sources = [Path(__file__), Path(__file__).with_name("neural.py"), Path(__file__).with_name("finite.py"),
               Path(__file__).with_name("runtime_adapter.py"), Path(__file__).with_name("targets.py"),
               Path(__file__).with_name("provenance.py"), STATE_PATH, THEORY_PATH]
    # Exact source bytes are captured before the first optimizer update.
    archive = output / "source_snapshot"
    sources.extend(Path(name) for name in runtime["files"])
    source_records = snapshot_sources(sources, archive)
    manifest = {"schema": "intrmotiv/first-hit-all-landmarks/v1", "config": asdict(config), "runtime": runtime,
                "source_records": source_records, "datasets": {str(path): digest(path) for path in paths.values()},
                "original_pre_run_sha256": digest(original / "pre_run_manifest.json"),
                "intervention": "Expand target vocabulary from DG5/DG6 to every one of the seven fixed DG identities; reuse exact original data",
                "primary_gate": "Every one of 42 nontrivial directed start-goal pairs succeeds with probability at least .9 in each seed; no evaluation updates",
                "budget_change": "Same2500updates/batch256/data4096/hidden128/H6; firsthit target labels perupdate increase2to7, so not compute-matched",
                "supervision": "Factual replay TD only; no empirical-model value labels enter the training loss"}
    (output / "pre_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for seed in config.seeds:
        folder = output / f"seed_{seed}"
        folder.mkdir()
        with np.load(paths[seed]) as loaded:
            data = {name: loaded[name] for name in loaded.files if name != "dg_observations"}
        model, exact, stats = train_readout(env, dg, goal_ids, data, config, seed, (decoder, heads), "replay_td")
        checkpoint = folder / "readout.pt"
        torch.save(model.state_dict(), checkpoint)
        before = state_digest(model)
        trials = evaluate_readout(model, env, dg, goal_ids, config, seed)
        unchanged = before == state_digest(model)
        write_csv(folder / "trials.csv", trials)
        calibrated = calibration(model, env, dg, goal_ids, config, exact)
        write_csv(folder / "deadline_calibration.csv", calibrated)
        reloaded = SharedReadout(dg.shape[1], config, decoder, heads)
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        reloaded.eval()
        reproduced = evaluate_readout(reloaded, env, dg, goal_ids, config, seed) == trials
        own = [row for row in trials if row["route"] == "first_hit"]
        alternative = [row for row in trials if row["route"] == "shuffled_goal"]
        eligible = [(a, b) for a, b in zip(own, alternative) if not b["start_at_commanded_goal"]]
        record = {"seed": seed, **stats, "n_nontrivial_pairs": len(own), "already_at_goal_pairs_excluded": len(goal_ids),
                  "n_lift_pairs": len(eligible), "first_hit_success": float(np.mean([row["success"] for row in own])),
                  "worst_pair_success": min(row["success"] for row in own),
                  "per_goal_success": {str(goal): float(np.mean([row["success"] for row in own if row["goal"] == goal])) for goal in goal_ids},
                  "paired_command_lift": float(np.mean([a["success"] - b["success"] for a, b in eligible])),
                  "per_route_success": {route: float(np.mean([row["success"] for row in trials if row["route"] == route])) for route in ("first_hit", "shuffled_goal", "memoryless", "continuing_main")},
                  "maximum_absolute_calibration_error": max(abs(row["error"]) for row in calibrated),
                  "pair_gate_passed": all(row["success"] >= config.pair_success_gate for row in own),
                  "checkpoint_sha256": digest(checkpoint), "evaluation_state_hash_unchanged": unchanged,
                  "saved_checkpoint_trials_exactly_reproduced": reproduced}
        (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps({key: record[key] for key in ("seed", "first_hit_success", "worst_pair_success", "paired_command_lift", "pair_gate_passed", "training_seconds")}), flush=True)
    result = {"schema": "intrmotiv/first-hit-all-landmarks-results/v1", "pre_run_manifest_sha256": digest(output / "pre_run_manifest.json"),
              "runs": records, "all_pair_gates_passed": all(record["pair_gate_passed"] for record in records),
              "source_audit": audit_source_snapshots(archive, check_current=True),
              "datasets_unchanged": all(digest(path) == sha for path, sha in manifest["datasets"].items()),
              "limitations": ["Synthetic frozen one-hot DG identities make current observations sufficient; sparse-history evidence is from the separate two-target toy.",
                              "No learned DMLab visual representation or production control is tested.",
                              "The main-head negative also removes goal input and is not a routing-only causal comparison.",
                              "Success at deadline6 does not establish calibrated success probabilities at every deadline."]}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output, arguments.original)
