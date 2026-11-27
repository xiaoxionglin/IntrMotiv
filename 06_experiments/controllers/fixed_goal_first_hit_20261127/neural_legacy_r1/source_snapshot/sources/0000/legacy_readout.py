"""Current legacy decoder qualification; pure finite graphs, no DMLab runtime.

Use the actual MlpDecoderJit with its existing concatenated target onehot.
The recorded trajectories, TD learner, CA3 update, and evaluator are reused.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .finite import FiniteMDP, make_fork_cycle
from .neural import (
    NeuralConfig, RUNTIME_ROOT, STATE_PATH, THEORY_PATH, SharedReadout,
    calibration, digest, evaluate_readout, sparse_context_environment,
    state_digest, train_readout, write_csv,
)
from .provenance import audit_source_snapshots, snapshot_sources
from .runtime_adapter import runtime_factories


def legacy_factory(decoder_settings):
    """Adapt the actual legacy constructor to the shared experiment factory."""
    source = RUNTIME_ROOT / "sf_working_directories/IntrMotiv/dmlab/custom_decoder.py"
    spec = importlib.util.spec_from_file_location("first_hit_original_legacy_decoder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def create(core, hidden_size=128):
        if decoder_settings["decoder_mlp_layers"][-1] != hidden_size:
            raise ValueError("Existing decoder output must match the declared Q head")
        return module.MlpDecoderJit(SimpleNamespace(**decoder_settings), core.get_out_size())

    return create, source


def run(output, original, donor_metadata):
    output, original = Path(output).resolve(), Path(original).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pre_run_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a legacy decoder qualification")
    donor_metadata = Path(donor_metadata).resolve(strict=True)
    donor = json.loads(donor_metadata.read_text())
    settings = {key: donor[key] for key in ("decoder_mlp_layers", "nonlinearity", "use_jit")}
    if donor["hrl_goal_conditioning"] != "legacy":
        raise ValueError("The followup must retain a verified legacy donor interface")
    config = replace(NeuralConfig(), methods=("replay_td",), hidden_size=settings["decoder_mlp_layers"][-1])
    torch.set_num_threads(1)
    factory, decoder_source = legacy_factory(settings)
    _, heads, runtime = runtime_factories(RUNTIME_ROOT)
    parent = make_fork_cycle()
    full = FiniteMDP("fork_all_landmarks", parent.transition, parent.terminal, tuple(range(parent.n_states)), parent.start_states)
    sparse, sparse_dg, sparse_goal_ids = sparse_context_environment()
    experiments = [(full, np.eye(full.n_states, dtype=np.float32), full.goal_states, "fork_cycle"),
                   (sparse, sparse_dg, sparse_goal_ids, "sparse_context")]
    original_manifest = json.loads((original / "pre_run_manifest.json").read_text())
    datasets = {}
    for _, _, _, source_env in experiments:
        for seed in config.seeds:
            path = original / source_env / f"seed_{seed}" / "reward_free_trajectories.npz"
            if digest(path) != original_manifest["datasets"][str(path)]:
                raise ValueError("Original data changed before legacy followup")
            datasets[str(path)] = digest(path)
    sources = [Path(__file__), Path(__file__).with_name("neural.py"), Path(__file__).with_name("finite.py"),
               Path(__file__).with_name("runtime_adapter.py"), Path(__file__).with_name("targets.py"),
               Path(__file__).with_name("provenance.py"), STATE_PATH, THEORY_PATH, decoder_source, donor_metadata]
    sources.extend(Path(name) for name in runtime["files"])
    source_snapshot = snapshot_sources(sources, output / "source_snapshot")
    manifest = {"schema": "intrmotiv/first-hit-legacy-readout/v1", "config": asdict(config), "decoder_settings": settings,
                "donor_metadata": str(donor_metadata), "donor_metadata_sha256": digest(donor_metadata),
                "source_snapshot": source_snapshot, "datasets": datasets,
                "runtime_class": "Actual MlpDecoderJit; full existing state plus target onehot; no FiLM",
                "primary_gate": "Every declared nontrivial pair success>=.9 in each seed, frozen evaluation",
                "scope": "All7 fixed DG targets on fork graph;2 target DGs+2 history cues on sparse context task",
                "limits": "Fresh readout weights with donor architecture, not transfer of pretrained decoder weights; parameter counts differ from FiLM, no efficiency claim"}
    (output / "pre_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for env, dg, goal_ids, source_env in experiments:
        for seed in config.seeds:
            folder = output / env.name / f"seed_{seed}"
            folder.mkdir(parents=True)
            with np.load(original / source_env / f"seed_{seed}" / "reward_free_trajectories.npz") as loaded:
                data = {key: loaded[key] for key in loaded.files if key != "dg_observations"}
            model, exact, stats = train_readout(env, dg, goal_ids, data, config, seed, (factory, heads), "replay_td")
            checkpoint = folder / "readout.pt"
            torch.save(model.state_dict(), checkpoint)
            before = state_digest(model)
            trials = evaluate_readout(model, env, dg, goal_ids, config, seed)
            unchanged = before == state_digest(model)
            write_csv(folder / "trials.csv", trials)
            calibrated = calibration(model, env, dg, goal_ids, config, exact)
            write_csv(folder / "deadline_calibration.csv", calibrated)
            reloaded = SharedReadout(dg.shape[1], config, factory, heads)
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            reloaded.eval()
            reproduced = evaluate_readout(reloaded, env, dg, goal_ids, config, seed) == trials
            own = [row for row in trials if row["route"] == "first_hit"]
            alternative = [row for row in trials if row["route"] == "shuffled_goal"]
            eligible = [(a, b) for a, b in zip(own, alternative) if not b["start_at_commanded_goal"]]
            record = {"environment": env.name, "seed": seed, **stats, "n_nontrivial_pairs": len(own),
                      "n_lift_pairs": len(eligible), "first_hit_success": float(np.mean([row["success"] for row in own])),
                      "worst_pair_success": min(row["success"] for row in own),
                      "per_goal_success": {str(goal): float(np.mean([row["success"] for row in own if row["goal"] == goal])) for goal in env.goal_states},
                      "paired_command_lift": float(np.mean([a["success"] - b["success"] for a, b in eligible])),
                      "per_route_success": {route: float(np.mean([row["success"] for row in trials if row["route"] == route])) for route in ("first_hit", "shuffled_goal", "memoryless", "continuing_main")},
                      "maximum_absolute_calibration_error": max(abs(row["error"]) for row in calibrated),
                      "pair_gate_passed": all(row["success"] >= config.pair_success_gate for row in own),
                      "checkpoint_sha256": digest(checkpoint), "evaluation_state_hash_unchanged": unchanged,
                      "saved_checkpoint_trials_exactly_reproduced": reproduced}
            (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(json.dumps({key: record[key] for key in ("environment", "seed", "first_hit_success", "pair_gate_passed", "training_seconds")}), flush=True)
    result = {"schema": "intrmotiv/first-hit-legacy-readout-results/v1", "runs": records,
              "pre_run_manifest_sha256": digest(output / "pre_run_manifest.json"),
              "all_pair_gates_passed": all(row["pair_gate_passed"] for row in records),
              "source_audit": audit_source_snapshots(output / "source_snapshot", check_current=True),
              "datasets_unchanged": all(digest(path) == sha for path, sha in datasets.items()),
              "limitations": manifest["limits"]}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--donor-metadata", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.original, args.donor_metadata)
