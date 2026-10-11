"""Count-weight conditional hit magnitudes from cached TensorBoard histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TAGS = {
    "correct_count": "intrmotiv/hrl/control/correct_count",
    "censored_count": "intrmotiv/hrl/control/censored_count",
    "reward_magnitude_mean": "intrmotiv/hrl/control/correct_reward_magnitude_mean",
    "correct_elapsed_mean": "intrmotiv/hrl/control/correct_elapsed_mean",
    "target_hit_event_count": "intrmotiv/hrl/target_hit_event_count",
    "encoder_event_count": "intrmotiv/encoder/dominant_event_count",
}


def weighted(value: np.ndarray, count: np.ndarray) -> float:
    positive = count > 0
    if not positive.any():
        return float("nan")
    if not np.isfinite(value[positive]).all():
        raise ValueError("Nonfinite conditional mean on a positive-count update")
    return float(np.dot(value[positive], count[positive]) / count[positive].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalar_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in sorted(args.scalar_root.iterdir()):
        path = directory / "per_run.csv"
        if not path.exists():
            continue
        run_table = pd.read_csv(path)
        for _, run in run_table.iterrows():
            if run.family == "C01":
                continue
            history = pd.read_csv(directory / "histories" / f"{run.run_name}.csv")
            history = history[history.step.between(90_000_000, 100_000_000) & history.tag.isin(TAGS.values())]
            if history.duplicated(["tag", "step"]).any():
                raise ValueError(f"Duplicate conditional scalar step in {run.run_name}")
            wide = history.pivot(index="step", columns="tag", values="value")
            missing = set(TAGS.values()) - set(wide)
            if missing:
                raise ValueError(f"Missing scalar tags in {run.run_name}: {missing}")
            wide = wide.dropna(subset=[TAGS["correct_count"], TAGS["reward_magnitude_mean"], TAGS["correct_elapsed_mean"]])
            count = wide[TAGS["correct_count"]].to_numpy(dtype=float)
            row = {"run_name": run.run_name, "condition": run.condition,
                   "family": run.family, "encoder_credit": str(run.encoder_credit).lower(),
                   "worker_bonus": str(run.worker_bonus).lower(), "seed": int(run.seed),
                   "study_cohort": directory.name, "window_low": 90_000_000,
                   "window_high": 100_000_000, "updates": len(wide),
                   "updates_with_correct_hits": int((count > 0).sum()),
                   "correct_hit_observation_count": float(count.sum()),
                   "censored_observation_count": float(wide[TAGS["censored_count"]].sum()),
                   "target_hit_event_observation_count": float(wide[TAGS["target_hit_event_count"]].sum()),
                   "encoder_event_observation_count": float(wide[TAGS["encoder_event_count"]].sum()),
                   "correct_reward_magnitude_count_weighted": weighted(
                       wide[TAGS["reward_magnitude_mean"]].to_numpy(dtype=float), count),
                   "correct_elapsed_count_weighted": weighted(
                       wide[TAGS["correct_elapsed_mean"]].to_numpy(dtype=float), count)}
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "conditional_hit_seed_metrics.csv", index=False)
    group = result.groupby(["family", "encoder_credit", "worker_bonus"])
    group.agg({name: ["mean", "std"] for name in (
        "correct_hit_observation_count", "target_hit_event_observation_count",
        "encoder_event_observation_count", "correct_reward_magnitude_count_weighted",
        "correct_elapsed_count_weighted",
    )}).to_csv(args.output / "conditional_hit_cell_summary.csv")
    print("processed", len(result), "goal-conditioned runs")


if __name__ == "__main__":
    main()
