"""Summarize canonical TensorBoard histories at the ablation's five milestones.

The input directories are outputs of ``intrmotiv_study collect-online`` with
``--export-histories``. This reads cached events; it never reloads TensorBoard.
Each row is one run and one milestone. The last 10M frames define mature
windows, while the 5M health window is 1–5M frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MILESTONES = (5_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)
TAGS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "dg_density": "intrmotiv/dg/density",
    "encoder_event_count": "intrmotiv/encoder/dominant_event_count",
    "encoder_feedback": "intrmotiv/encoder/feedback_on_dominant_event_mean",
    "target_hit_numerator": "intrmotiv/hrl/target_hit_numerator",
    "correct_outcome_count": "intrmotiv/hrl/control/correct_count",
    "correct_elapsed_mean": "intrmotiv/hrl/control/correct_elapsed_mean",
    "target_action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "advantage_std": "train/adv_std",
    "dg_silent_fraction": "intrmotiv/dg/silent_unit_fraction",
}


def summarize_run(history_path: Path, run: pd.Series) -> list[dict]:
    history = pd.read_csv(history_path, usecols=("tag", "step", "value"))
    history = history.loc[history.tag.isin(TAGS.values())]
    rows = []
    for milestone in MILESTONES:
        low = max(1_000_000, milestone - 10_000_000)
        selected = history.loc[(history.step >= low) & (history.step <= milestone)]
        result = {
            "run_name": run.run_name,
            "family": run.family,
            "seed": int(run.seed),
            "credit": run.credit,
            "worker_bonus": run.worker_bonus,
            "milestone": milestone,
            "window_low": low,
            "window_high": milestone,
        }
        for name, tag in TAGS.items():
            values = selected.loc[selected.tag == tag, "value"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            result[name] = float(finite.mean()) if len(finite) else np.nan
            result[f"{name}_n"] = len(finite)
            result[f"{name}_nonfinite"] = len(values) - len(finite)
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("online_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    results = []
    for online_dir in args.online_dirs:
        per_run = pd.read_csv(online_dir / "per_run.csv")
        for _, run in per_run.iterrows():
            history_path = online_dir / "histories" / f"{run.run_name}.csv"
            if not history_path.is_file():
                raise FileNotFoundError(history_path)
            results.extend(summarize_run(history_path, run))
    output = pd.DataFrame(results).sort_values(
        ["family", "worker_bonus", "credit", "seed", "milestone"]
    )
    expected = 44 * len(MILESTONES)
    if len(output) != expected or output[["run_name", "milestone"]].duplicated().any():
        raise ValueError(f"Expected {expected} unique run-milestone rows, got {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
