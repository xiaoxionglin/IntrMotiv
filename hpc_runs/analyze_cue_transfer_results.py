"""Join the two fixed cue-transfer studies without deriving factors from names.

Run after canonical collect-online and analyze_place_field_manifest. Raw trial
files remain in the workspace. This adapter retains cue denominators, matched
reset evidence, seed-level outcomes and censored decision times.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study import load_study


def table(path, sep=","):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter=sep))


def normalized_reward_auc(x, y, low, high):
    """Integrate all logged points, including sparse spikes between plot bins.

    Duplicate steps must be averaged by the caller. Outside logged support,
    hold the nearest endpoint; callers retain support bounds in their output.
    """
    x, y = np.asarray(x), np.asarray(y)
    assert high > low and len(x) == len(y) and np.all(np.diff(x) > 0)
    knots = np.r_[low, x[(x > low) & (x < high)], high]
    return float(np.trapz(np.interp(knots, x, y), knots) / (high - low))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-online", action="store_true", help="Aggregate completed evaluations before online collection finishes"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes, per_cue, sources, resets, curves, fields, trajectories = [], [], [], [], [], [], []
    reference = None
    representatives = {}
    selected = {
        "CT_D51_FRESH_SIMPLEX",
        "CT_D51_FRESH_FREE",
        "CT_D51_SOURCE_SIMPLEX",
        "CT_D51_SOURCE_FREE",
        "CE_SOURCE_FREE_BN",
        "CE_SOURCE_FREE_PPO",
        "CE_SOURCE_FREE_JOINT",
        "CE_RANDOM_FROZEN_FREE",
    }
    reset_keys = [
        "requested_seed",
        "engine_seed",
        "number_instruction",
        "reward_center_x",
        "reward_center_y",
        "policy_seed",
        "start_x",
        "start_y",
    ]
    for study_id in ["cue_transfer_20261104", "cue_transfer_expanded_20261104"]:
        spec_path = args.source / "hpc_runs/studies" / (study_id + ".study.json")
        study = load_study(spec_path)
        runs = {(run.condition, run.seed): run for run in study.expand_runs()}
        frozen = args.train_root / "analysis" / study_id / "frozen"
        manifest = table(frozen / "terminal_heldout_manifest.tsv", "\t")
        online = args.output / study_id / "online"
        for row in manifest:
            run = runs[(row["condition"], int(row["seed"]))]
            meta = {
                "study": study_id,
                "condition": run.condition,
                "seed": run.seed,
                "run_name": run.name,
                **run.factors,
                **run.metadata,
            }
            assert int(row["checkpoint_frames"]) == 25001984
            for protocol in ["correct", "shuffled_cue", "zero_ca3", "zero_depth"]:
                base = frozen
                if study_id == "cue_transfer_20261104" and protocol != "correct":
                    base = args.train_root / "analysis/cue_transfer_expanded_20261104/parent_interventions"
                suffix = "" if protocol == "correct" else "__" + protocol
                path = base / "heldout/results" / (row["label_suffix"] + suffix + ".csv")
                trials = table(path)
                assert len(trials) == 100 and len({x["requested_seed"] for x in trials}) == 100
                key = [tuple(x[k] for k in reset_keys) for x in trials]
                if reference is None:
                    reference = key
                resets.append({**meta, "protocol": protocol, "matches_common_resets": key == reference})
                assert key == reference, str(path)
                assert all(x["checkpoint"] == row["checkpoint"] for x in trials)
                if protocol == "shuffled_cue":
                    assert all(x["policy_number_instruction"] != x["number_instruction"] for x in trials)
                elif "policy_number_instruction" in trials[0]:
                    assert all(x["policy_number_instruction"] == x["number_instruction"] for x in trials)
                cue_rates, cue_times = [], []
                for cue in range(1, 6):
                    subset = [x for x in trials if int(x["number_instruction"]) == cue]
                    assert subset
                    success = [x["physical_success"].lower() == "true" for x in subset]
                    rate = np.mean(success)
                    # Failures are censored at the common 1800-decision horizon.
                    times = [int(x["decisions_to_termination"]) if ok else 1800 for x, ok in zip(subset, success)]
                    cue_rates.append(rate)
                    cue_times.append(np.mean(times))
                    per_cue.append(
                        {
                            **meta,
                            "protocol": protocol,
                            "cue": cue,
                            "n": len(subset),
                            "successes": sum(success),
                            "success_rate": rate,
                            "censored_mean_decisions": np.mean(times),
                        }
                    )
                outcomes.append(
                    {
                        **meta,
                        "protocol": protocol,
                        "success_equal_cue": np.mean(cue_rates),
                        "success_pooled": np.mean([x["physical_success"].lower() == "true" for x in trials]),
                        "censored_decisions_equal_cue": np.mean(cue_times),
                    }
                )
                sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if not args.skip_online:
                history = pd.read_csv(online / "histories" / (run.name + ".csv"))
                reward = history[history.tag == "intrmotiv/reward/environment_mean"]
                # Duplicate logged steps use their mean, then linear interpolation.
                reward = reward.groupby("step", as_index=False).value.mean().sort_values("step")
                x, y = reward.step.to_numpy(), reward.value.to_numpy()
                assert len(x) > 2
                grid = np.arange(0, 25000001, 250000)
                # The plot is a 250k-frame bin mean, not subsampled sparse events.
                values = [normalized_reward_auc(x, y, max(0, step - 250000), max(250000, step)) for step in grid]
                auc = {}
                for high in [5000000, 10000000, 24000000, 25000000]:
                    auc["reward_auc_mean_" + str(high)] = normalized_reward_auc(x, y, 0, high)
                for step, value in zip(grid, values):
                    curves.append(
                        {
                            **meta,
                            "frames": step,
                            "reward_mean": value,
                            **auc,
                            "first_reward_log_step": int(x[0]),
                            "last_reward_log_step": int(x[-1]),
                        }
                    )
        frame = pd.read_csv(args.output / study_id / "fields/derived_place_field_metrics.csv")
        fields.append(frame.assign(study=study_id))
        from sf_working_directories.IntrMotiv.evaluation.analyze_place_field_manifest import artifact_for_suffix

        for row in table(frozen / "analysis_manifest.tsv", "\t"):
            artifact = artifact_for_suffix(frozen / "fields/raw", row["label_suffix"])
            pose = pd.read_csv(artifact.parent / "pose.csv")
            xy = pose[["x", "y"]].to_numpy()
            same_segment = np.diff(pose.num_traj.to_numpy()) == 0
            trajectories.append(
                {
                    **row,
                    "study": study_id,
                    "pose_rows": len(pose),
                    "segments": pose.num_traj.nunique(),
                    "path_length_world_units": np.linalg.norm(np.diff(xy, axis=0), axis=1)[same_segment].sum(),
                    "columns": ",".join(pose.columns),
                    "artifact": str(artifact),
                }
            )
            if row["condition"] in selected and int(row["seed"]) == 42 and int(row["target_frames"]) == 25000000:
                with np.load(artifact, allow_pickle=False) as data:
                    for key in [
                        "occupancy",
                        "rate_maps",
                        "pre_threshold_rate_maps",
                        "spatial_information",
                        "active_fraction",
                        "bounds",
                    ]:
                        representatives[row["condition"] + "__" + key] = data[key]
                representatives[row["condition"] + "__pose"] = pose[["x", "y", "num_traj"]].to_numpy()
    assert len(outcomes) == 324 and len(trajectories) == 189
    pd.DataFrame(outcomes).to_csv(args.output / "heldout_per_run.csv", index=False)
    pd.DataFrame(per_cue)[
        [
            "study",
            "condition",
            "seed",
            "protocol",
            "cue",
            "n",
            "successes",
            "success_rate",
            "censored_mean_decisions",
        ]
    ].to_csv(args.output / "heldout_per_cue.csv", index=False)
    if curves:
        frame = pd.DataFrame(curves)
        keep = [
            "study",
            "condition",
            "seed",
            "frames",
            "reward_mean",
            *[key for key in frame if key.startswith("reward_auc_mean_")],
            "first_reward_log_step",
            "last_reward_log_step",
        ]
        frame[keep].to_csv(args.output / "reward_curves.csv", index=False)
    pd.concat(fields).to_csv(args.output / "field_metrics.csv", index=False)
    pd.DataFrame(trajectories).to_csv(args.output / "trajectory_inventory.csv", index=False)
    pd.DataFrame(resets)[["study", "condition", "seed", "protocol", "matches_common_resets"]].to_csv(
        args.output / "matched_reset_audit.csv", index=False
    )
    np.savez_compressed(args.output / "representative_maps_and_paths.npz", **representatives)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "common_reset_count": 100,
                "matched_protocol_files": len(resets),
                "horizon_decisions": 1800,
                "terminal_checkpoint_frames": 25001984,
                "reward_curve_method": (
                    "mean duplicate steps; integrate all native event knots for normalized trapezoidal AUC; "
                    "plot preceding250k-frame bin means; edge hold outside logged support"
                ),
                "graph": "inapplicable: graph disabled in all conditions",
            },
            indent=2,
        )
        + "\n"
    )
    summary = pd.DataFrame(outcomes).groupby(["condition", "protocol"]).success_equal_cue.agg(["mean", "std"])
    summary.to_csv(args.output / "heldout_condition_summary.csv")
    print(summary.xs("correct", level="protocol").sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()
