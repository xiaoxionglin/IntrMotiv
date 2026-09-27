"""Aggregate paired five-cue held-out episodes from the canonical evaluator.

The matched unit is (architecture, downstream seed, requested reset seed).
Success-only time to reward is reported together with a horizon-censored mean
decision count, so failures do not silently disappear from timing summaries.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    trials = []
    missing = []
    for row in manifest:
        path = args.results / f"{row['label_suffix']}.csv"
        if not path.exists():
            missing.append(path.name)
            continue
        data = pd.read_csv(path)
        architecture = row["condition"].split("_")[1]
        arm = "SOURCE_DG" if "SOURCE_DG" in row["condition"] else "RAND_DG"
        data["architecture"] = architecture
        data["arm"] = arm
        data["seed"] = int(row["seed"])
        data["checkpoint_frames"] = int(row["checkpoint_frames"])
        data["label_suffix"] = row["label_suffix"]
        trials.append(data)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} held-out files: {missing}")
    all_trials = pd.concat(trials, ignore_index=True)
    all_trials["censored_decisions"] = all_trials["decisions_to_termination"]
    keys = ["architecture", "seed", "requested_seed"]
    first = all_trials[all_trials.arm == "SOURCE_DG"].set_index(keys)
    second = all_trials[all_trials.arm == "RAND_DG"].set_index(keys)
    if set(first.index) != set(second.index):
        raise ValueError("Source and random arms do not have matched reset keys")
    paired = first.join(second, lsuffix="_source", rsuffix="_random", validate="one_to_one")
    for column in ("engine_seed", "number_instruction", "start_x", "start_y",
                   "reward_center_x", "reward_center_y"):
        if not np.array_equal(paired[f"{column}_source"], paired[f"{column}_random"]):
            raise ValueError(f"Held-out starts/cues are not matched: {column}")
    paired["cue"] = paired["number_instruction_source"].astype(int)
    paired["source_success"] = paired["physical_success_source"].astype(int)
    paired["random_success"] = paired["physical_success_random"].astype(int)
    paired["source_minus_random_success"] = paired["source_success"] - paired["random_success"]
    paired["source_minus_random_censored_decisions"] = (paired["censored_decisions_source"] -
                                                           paired["censored_decisions_random"])
    args.output.mkdir(parents=True, exist_ok=True)
    paired.reset_index().to_csv(args.output / "heldout_paired_trials.csv", index=False)
    groups = []
    for (architecture, arm, cue), data in all_trials.groupby(["architecture", "arm", "number_instruction"]):
        successful = data[data.physical_success.astype(bool)]
        groups.append({"architecture": architecture, "arm": arm, "cue": int(cue),
                       "episodes": len(data), "trained_seeds": data.seed.nunique(),
                       "successes": len(successful), "success_fraction": len(successful) / len(data),
                       "mean_censored_decisions": data.censored_decisions.mean(),
                       "median_decisions_to_reward_when_successful": (successful.decisions_to_termination.median()
                                                                      if len(successful) else np.nan),
                       "median_seconds_to_reward_when_successful": (successful.time_to_reward_seconds.median()
                                                                    if len(successful) else np.nan)})
    pd.DataFrame(groups).to_csv(args.output / "heldout_by_cue.csv", index=False)
    aggregate = []
    for (architecture, arm), data in all_trials.groupby(["architecture", "arm"]):
        aggregate.append({"architecture": architecture, "arm": arm, "episodes": len(data),
                          "success_fraction": data.physical_success.mean(),
                          "mean_censored_decisions": data.censored_decisions.mean()})
    pd.DataFrame(aggregate).to_csv(args.output / "heldout_overall.csv", index=False)
    print(f"Analyzed {len(all_trials)} episodes and {len(paired)} exact start/cue pairs")


if __name__ == "__main__":
    main()
