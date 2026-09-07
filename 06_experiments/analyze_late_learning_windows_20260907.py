"""Audit apparent post-50M learning in the recent completed studies."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wandb


ENTITY = "xiaoxionglin-bernstein-center-freiburg"
STUDIES = (
    (
        "DPR",
        "SF_IntrMotiv_DirectionalPredictiveRecruitment",
        "intrmotiv_directional_predictive_recruitment_20260904",
        re.compile(r"DPR_(C05|C13|C15)_(MON|DIR|PRED)_(LEG|FILM)_S(8|99|123)"),
    ),
    (
        "SAT",
        "SF_IntrMotiv_SaturdayBatch",
        "intrmotiv_saturday_batch_20260905",
        re.compile(r"SAT_C15_(ARR|SRC)_(MON|DIRO|PREDO)_(LEG|FILM)_S(8|99|123)"),
    ),
)

METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "option_success": "intrmotiv/hrl/option_success_fraction",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "reachable_pairs": "intrmotiv/hrl/reliable/reachable_pair_fraction",
}

RATE_METRICS = {
    "target_num": "intrmotiv/hrl/target_hit_numerator",
    "target_count": "intrmotiv/hrl/target_hit_event_count",
    "shuffle_num": "intrmotiv/hrl/shuffled_hit_numerator",
    "shuffle_count": "intrmotiv/hrl/shuffled_hit_event_count",
}

WINDOWS = ((5, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def fetch(run, family: str, pattern: re.Pattern[str]) -> list[dict[str, object]]:
    match = pattern.search(run.name)
    if match is None:
        raise ValueError(f"cannot parse {run.name}")
    factor1, factor2, goal, seed = match.groups()
    requested = {**METRICS, **(RATE_METRICS if family == "SAT" else {})}
    frame = pd.DataFrame(run.history(
        keys=["train/env_steps", *requested.values()],
        samples=5_000,
        pandas=False,
    )).rename(columns={value: key for key, value in requested.items()})
    if "train/env_steps" not in frame:
        raise RuntimeError(f"{run.name}: missing train/env_steps")
    rows = []
    for low_m, high_m in WINDOWS:
        low, high = low_m * 1_000_000, high_m * 1_000_000
        window = frame.loc[(frame["train/env_steps"] >= low) & (frame["train/env_steps"] < high)]
        if family == "SAT":
            target_rate = ratio(window["target_num"].sum(), window["target_count"].sum())
            shuffle_rate = ratio(window["shuffle_num"].sum(), window["shuffle_count"].sum())
        else:
            target_rate = shuffle_rate = np.nan
        row = {
            "family": family,
            "run_name": match.group(0),
            "factor1": factor1,
            "factor2": factor2,
            "goal_conditioning": goal,
            "seed": int(seed),
            "window_low_m": low_m,
            "window_high_m": high_m,
            "samples": len(window),
            "coverage_auc": window["coverage_auc"].mean(),
            "option_success": window["option_success"].mean(),
            "action_sensitivity": window["action_sensitivity"].mean(),
            "target_rate": target_rate,
            "shuffle_rate": shuffle_rate,
            "target_advantage": ratio(target_rate, shuffle_rate) - 1,
            "reachable_pairs": window["reachable_pairs"].mean(),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_experiments/results/late_learning_audit_20260907"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    rows: list[dict[str, object]] = []
    for family, project, group, pattern in STUDIES:
        runs = list(api.runs(f"{ENTITY}/{project}", filters={"group": group}, per_page=100))
        expected = 54 if family == "DPR" else 36
        if len(runs) != expected:
            raise RuntimeError(f"{family}: expected {expected} runs, found {len(runs)}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, run, family, pattern) for run in runs]
            for future in as_completed(futures):
                rows.extend(future.result())

    frame = pd.DataFrame(rows).sort_values(
        ["family", "run_name", "window_low_m"]
    )
    frame.to_csv(args.output_dir / "per_run_window.csv", index=False)
    metrics = [
        "coverage_auc", "option_success", "action_sensitivity",
        "target_rate", "shuffle_rate", "target_advantage", "reachable_pairs",
    ]
    summary = (
        frame.groupby(["family", "factor1", "factor2", "goal_conditioning", "window_low_m"])[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "condition_window_summary.csv", index=False)

    early = frame.loc[frame.window_low_m == 45].set_index("run_name")
    late = frame.loc[frame.window_low_m == 65].set_index("run_name")
    common = early.index.intersection(late.index)
    deltas = late.loc[common, metrics] - early.loc[common, metrics]
    identity = late.loc[common, ["family", "factor1", "factor2", "goal_conditioning", "seed"]]
    delta_frame = identity.join(deltas.add_prefix("delta_"), how="inner").reset_index()
    delta_frame.to_csv(args.output_dir / "late_deltas_45_55_to_65_75.csv", index=False)

    threshold = 0.05
    delta_frame["advantage_crossed_above_zero"] = (
        (early.loc[common, "target_advantage"].to_numpy() <= 0)
        & (late.loc[common, "target_advantage"].to_numpy() > 0)
    )
    delta_frame["advantage_crossed_above_5pct"] = (
        (early.loc[common, "target_advantage"].to_numpy() <= threshold)
        & (late.loc[common, "target_advantage"].to_numpy() > threshold)
    )
    delta_frame.to_csv(args.output_dir / "late_deltas_with_crossings.csv", index=False)


if __name__ == "__main__":
    main()
