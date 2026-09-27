"""Summarize frozen matched-command interventions without mixing protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Directory containing protocol-named subdirectories of JSON summaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.inputs.glob("*/raw/*/intervention_summary.json")):
        data = json.loads(path.read_text())
        eligible = int(data["ordered_pairs_eligible"])
        completed = int(data["ordered_pairs_complete"])
        records.append({"protocol": path.parts[-4], "condition": data["condition"],
                        "seed": int(data["seed"]),
                        "checkpoint_frames": int(data["checkpoint_frames"]),
                        "trial_count": int(data["trial_count"]),
                        "eligible_pairs": eligible, "complete_pairs": completed,
                        "complete_pair_fraction": completed / eligible if eligible else float("nan"),
                        "executed_success": data["executed_target_success_rate"],
                        "matched_shuffled_success": data["matched_shuffled_target_success_rate"],
                        "executed_minus_shuffled": (data["executed_target_success_rate"] -
                                                    data["matched_shuffled_target_success_rate"]),
                        "action_sensitivity": data["mean_counterfactual_action_sensitivity"],
                        "decision_cap": int(data["decision_cap"]),
                        "summary_file": str(path.relative_to(args.inputs))})
    if not records:
        raise ValueError(f"No intervention summaries in {args.inputs}")
    per_run = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output / "control_interventions_per_run.csv", index=False)
    columns = ["trial_count", "complete_pair_fraction", "executed_success",
               "matched_shuffled_success", "executed_minus_shuffled", "action_sensitivity"]
    aggregate = per_run.groupby(["protocol", "condition"], as_index=False)[columns].mean()
    aggregate["trained_seeds"] = [len(per_run[(per_run.protocol == row.protocol) &
                                              (per_run.condition == row.condition)])
                                  for row in aggregate.itertuples()]
    aggregate.to_csv(args.output / "control_interventions_summary.csv", index=False)
    print(f"Summarized {len(per_run)} intervention runs")


if __name__ == "__main__":
    main()
