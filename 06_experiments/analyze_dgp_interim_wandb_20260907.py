"""Interim trajectory audit for the DGP first-outcome production study.

This is deliberately diagnostic rather than a replacement for the canonical
TensorBoard collector.  It uses W&B to compare early learning windows while
the jobs are still running and records the exact StudySpec fingerprint.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from hpc_runs.intrmotiv_study import load_study


ENTITY = "xiaoxionglin-bernstein-center-freiburg"
PROJECT = "SF_IntrMotiv_DGPolicyGradientFirstOutcome"
GROUP = "intrmotiv_dg_policy_gradient_first_outcome_20260906"
STUDY = Path("hpc_runs/studies/dg_policy_gradient_first_outcome.study.json")
NAME = re.compile(r"DGP_C15_(HIT|FIRST)_(STOP|JOINT)_(LEG|FILM)_S(8|99|123)")

METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "dg_density": "intrmotiv/dg/density",
    "silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "usage_entropy": "intrmotiv/dg/usage_entropy",
    "option_success": "intrmotiv/hrl/option_success_fraction",
    "action_tv": "intrmotiv/hrl/goal_condition/action_probability_tv",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "target_num": "intrmotiv/hrl/target_hit_numerator",
    "target_count": "intrmotiv/hrl/target_hit_event_count",
    "shuffle_num": "intrmotiv/hrl/shuffled_hit_numerator",
    "shuffle_count": "intrmotiv/hrl/shuffled_hit_event_count",
    "correct": "intrmotiv/hrl/control/correct_count",
    "wrong": "intrmotiv/hrl/control/wrong_count",
    "timeout": "intrmotiv/hrl/control/timeout_count",
    "command_entropy": "intrmotiv/hrl/control/normalized_command_entropy",
    "pair_coverage": "intrmotiv/hrl/control/observed_pair_coverage",
    "candidate_pairs": "intrmotiv/hrl/control/local_candidate_pair_count",
    "candidate_sources": "intrmotiv/hrl/control/local_candidate_source_fraction",
    "candidate_count": "intrmotiv/hrl/control/local_candidate_count_mean",
    "behavior_candidate_count": "intrmotiv/hrl/control/behavior_candidate_count_mean",
    "commanded_num": "intrmotiv/hrl/control/commanded_numerator",
    "commanded_count": "intrmotiv/hrl/control/commanded_event_count",
    "first_shuffle_num": "intrmotiv/hrl/control/shuffled_numerator",
    "first_shuffle_count": "intrmotiv/hrl/control/shuffled_event_count",
    "largest_scc": "intrmotiv/hrl/reliable/largest_scc",
    "reachable_pairs": "intrmotiv/hrl/reliable/reachable_pair_fraction",
    "top3_incoming": "intrmotiv/hrl/reliable/top3_incoming_confidence_share",
    "ppo_dg_norm": "intrmotiv/dg/gradient/ppo_norm",
    "encoder_dg_norm": "intrmotiv/dg/gradient/encoder_norm",
    "gradient_ratio": "intrmotiv/dg/gradient/ppo_encoder_ratio",
    "gradient_cosine": "intrmotiv/dg/gradient/cosine",
    "row_conflict": "intrmotiv/dg/gradient/row_conflict_fraction",
}

WINDOWS = [(1_000_000, 5_000_000), (5_000_000, 10_000_000),
           (10_000_000, 15_000_000), (15_000_000, 20_000_000),
           (20_000_000, 23_000_000)]


def parse_name(name: str) -> dict[str, object]:
    match = NAME.search(name)
    if match is None:
        raise ValueError(f"cannot parse run name: {name}")
    outcome, gradient, goal, seed = match.groups()
    return {
        "worker_outcome": outcome,
        "ppo_dg_gradient": gradient,
        "goal_conditioning": goal,
        "seed": int(seed),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def fetch(run) -> list[dict[str, object]]:
    frame = run.history(
        keys=["train/env_steps", *METRICS.values()],
        x_axis="train/env_steps",
        samples=10_000,
        pandas=True,
    )
    frame = frame.rename(columns={value: key for key, value in METRICS.items()})
    identity = parse_name(run.name)
    rows: list[dict[str, object]] = []
    for low, high in WINDOWS:
        window = frame.loc[(frame["train/env_steps"] >= low) & (frame["train/env_steps"] < high)]
        row: dict[str, object] = {
            "run_name": NAME.search(run.name).group(0),
            **identity,
            "window_low": low,
            "window_high": high,
            "samples": len(window),
            "run_state": run.state,
            "max_env_steps": float(frame["train/env_steps"].max()),
        }
        for metric in METRICS:
            row[metric] = float(window[metric].dropna().mean()) if metric in window else np.nan
        row["target_rate"] = safe_ratio(window["target_num"].sum(), window["target_count"].sum())
        row["shuffle_rate"] = safe_ratio(window["shuffle_num"].sum(), window["shuffle_count"].sum())
        row["target_advantage"] = safe_ratio(row["target_rate"], row["shuffle_rate"]) - 1
        row["first_commanded_rate"] = safe_ratio(
            window["commanded_num"].sum(), window["commanded_count"].sum()
        )
        row["first_shuffle_rate"] = safe_ratio(
            window["first_shuffle_num"].sum(), window["first_shuffle_count"].sum()
        )
        row["first_advantage"] = (
            safe_ratio(row["first_commanded_rate"], row["first_shuffle_rate"]) - 1
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_experiments/results/dgp_interim_20260907"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=50))
    if len(runs) != 24:
        raise RuntimeError(f"expected 24 runs, found {len(runs)}")
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, run) for run in runs]
        for future in as_completed(futures):
            rows.extend(future.result())
    per_run = pd.DataFrame(rows).sort_values(
        ["window_low", "worker_outcome", "ppo_dg_gradient", "goal_conditioning", "seed"]
    )
    per_run.to_csv(args.output_dir / "per_run_window.csv", index=False)

    metrics = [
        "coverage_auc", "dg_density", "silent_fraction", "usage_entropy",
        "option_success", "action_tv", "action_sensitivity", "target_rate",
        "shuffle_rate", "target_advantage", "correct", "wrong", "timeout",
        "command_entropy", "pair_coverage", "candidate_pairs", "candidate_sources",
        "candidate_count", "behavior_candidate_count", "first_commanded_rate",
        "first_shuffle_rate", "first_advantage", "largest_scc", "reachable_pairs",
        "top3_incoming", "ppo_dg_norm", "encoder_dg_norm", "gradient_ratio",
        "gradient_cosine", "row_conflict",
    ]
    summary = (
        per_run.groupby(
            ["window_low", "window_high", "worker_outcome", "ppo_dg_gradient", "goal_conditioning"],
            dropna=False,
        )[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "condition_window_summary.csv", index=False)

    factor_summary = (
        per_run.groupby(["window_low", "window_high", "worker_outcome", "ppo_dg_gradient"])[metrics]
        .mean()
        .reset_index()
    )
    factor_summary.to_csv(args.output_dir / "outcome_gradient_window_summary.csv", index=False)

    study = load_study(STUDY)
    metadata = {
        **study.provenance(),
        "wandb_entity": ENTITY,
        "wandb_project": PROJECT,
        "wandb_group": GROUP,
        "windows": WINDOWS,
        "purpose": "interim learning-trajectory diagnosis; canonical TensorBoard analysis remains authoritative",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
