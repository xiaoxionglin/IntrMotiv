#!/usr/bin/env python3
"""Reproducible terminal analysis for the completed Saturday IntrMotiv study.

The script uses final 5M-step W&B histories for rapidly varying online metrics,
the synchronized 75M online place-field snapshot for spatial metrics, and final
cumulative counters for recruitment/replay bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


ENTITY = "xiaoxionglin-bernstein-center-freiburg"
PROJECT = "SF_IntrMotiv_SaturdayBatch"
GROUP = "intrmotiv_saturday_batch_20260905"
STUDY_SHA256 = "dcbce50207b64b452c19bc1bf408a549ff10fb4b664b3296383b1b03e5be31cb"
RUN_RE = re.compile(
    r"^\d+_(SAT_C15_(ARR|SRC)_(MON|DIRO|PREDO)_(LEG|FILM)_S(8|99|123))_"
)

STEP = "train/env_steps"
HISTORY_METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "dg_density": "intrmotiv/dg/density",
    "dg_silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "dg_usage_entropy": "intrmotiv/dg/usage_entropy",
    "option_success": "intrmotiv/hrl/option_success_fraction",
    "action_probability_tv": "intrmotiv/hrl/goal_condition/action_probability_tv",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "target_num": "intrmotiv/hrl/target_hit_numerator",
    "target_count": "intrmotiv/hrl/target_hit_event_count",
    "shuffled_num": "intrmotiv/hrl/shuffled_hit_numerator",
    "shuffled_count": "intrmotiv/hrl/shuffled_hit_event_count",
    "largest_scc": "intrmotiv/hrl/reliable/largest_scc",
    "reachable_pair_fraction": "intrmotiv/hrl/reliable/reachable_pair_fraction",
    "outgoing_node_fraction": "intrmotiv/hrl/reliable/outgoing_node_fraction",
    "top3_incoming_share": "intrmotiv/hrl/reliable/top3_incoming_confidence_share",
    "reliable_global_efficiency": "intrmotiv/hrl/summary/reliable_global_efficiency",
    "grounded_controllability": "intrmotiv/hrl/summary/grounded_controllability",
    "candidate_endpoints": "intrmotiv/dg/recruitment/candidate_count",
    "silent_endpoints": "intrmotiv/dg/recruitment/silent_endpoint_count",
    "active_endpoints": "intrmotiv/dg/recruitment/active_endpoint_count",
    "eligible_victim_endpoints": "intrmotiv/dg/recruitment/eligible_victim_endpoint_count",
    "residual_pass": "intrmotiv/dg/recruitment/residual_pass_count",
    "replacement_conversion": "intrmotiv/dg/recruitment/replacement_conversion",
    "attempt_coverage": "intrmotiv/dg/recruitment/attempt_coverage_fraction",
    "fully_tested_count": "intrmotiv/dg/recruitment/fully_tested_count",
    "predictive_gap": "intrmotiv/dg/recruitment/predictive_reliability_gap",
    "predictive_eligible_count": "intrmotiv/dg/recruitment/predictive_eligible_count",
    "bad_source_count": "intrmotiv/dg/recruitment/bad_source_count",
    "zero_outdegree_count": "intrmotiv/dg/recruitment/zero_outdegree_count",
    "credit_replay_match": "intrmotiv/encoder/credit/replay_match",
    "behavior_replay_mismatch": "intrmotiv/hrl/behavior_replay_mismatch",
    "stale_generation_fraction": "intrmotiv/replay/stale_generation_rejected_fraction",
}

SNAPSHOT_METRICS = {
    "spatial_information": "intrmotiv/online/place_field/active_unit_mean_spatial_information",
    "active_map_cosine": "intrmotiv/online/place_field/active_only_map_cosine",
    "mono_field_fraction": "intrmotiv/online/place_field/mono_field_unit_fraction",
    "active_unit_fraction": "intrmotiv/online/place_field/active_unit_fraction",
    "unique_peak_bins": "intrmotiv/online/place_field/unique_active_peak_bins",
    "path_efficiency": "intrmotiv/online/trajectory/path_efficiency",
    "spatial_target_steps": "intrmotiv/online/window/target_env_steps",
}

CUMULATIVE_METRICS = {
    "recruitment_total": "intrmotiv/dg/recruitment/total",
    "repeat_total": "intrmotiv/dg/recruitment/repeat_total",
    "goal_adapter_reset_total": "intrmotiv/dg/recruitment/goal_adapter_reset_total",
    "dropped_rollouts_total": "intrmotiv/replay/dropped_rollouts_total",
    "dropped_decisions_total": "intrmotiv/replay/dropped_decisions_total",
    "deferred_updates_total": "intrmotiv/replay/deferred_updates_total",
}


def parse_run(name: str) -> dict[str, object]:
    match = RUN_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected run name: {name}")
    canonical, credit, retirement, goal, seed = match.groups()
    return {
        "run_name": canonical,
        "encoder_credit": credit,
        "retirement": retirement,
        "goal_conditioning": goal,
        "seed": int(seed),
    }


def finite_mean(frame: pd.DataFrame, key: str) -> float:
    values = pd.to_numeric(frame.get(key), errors="coerce")
    return float(values.mean()) if values.notna().any() else math.nan


def fetch_one(run, terminal_start: float) -> dict[str, object]:
    identity = parse_run(run.name)
    keys = [STEP, *HISTORY_METRICS.values()]
    history = pd.DataFrame(run.history(samples=5000, keys=keys, pandas=False))
    if STEP not in history:
        raise RuntimeError(f"{run.name}: missing {STEP}")
    history[STEP] = pd.to_numeric(history[STEP], errors="coerce")
    window = history.loc[history[STEP] >= terminal_start].copy()
    if window.empty:
        raise RuntimeError(f"{run.name}: no observations at or above {terminal_start:g}")
    result = {
        **identity,
        "wandb_name": run.name,
        "wandb_id": run.id,
        "state": run.state,
        "max_env_steps": float(history[STEP].max()),
        "window_start": terminal_start,
        "window_rows": int(len(window)),
    }
    for short, tag in HISTORY_METRICS.items():
        result[short] = finite_mean(window, tag)
    for short in [
        "candidate_endpoints",
        "silent_endpoints",
        "active_endpoints",
        "eligible_victim_endpoints",
        "residual_pass",
        "replacement_conversion",
    ]:
        values = pd.to_numeric(history.get(HISTORY_METRICS[short]), errors="coerce")
        result[f"{short}_history_sum"] = float(values.sum(min_count=1))
        result[f"{short}_nonzero_reports"] = int((values.fillna(0) > 0).sum())

    def aggregate_rate(num_tag: str, den_tag: str) -> tuple[float, float, float]:
        num = pd.to_numeric(window.get(num_tag), errors="coerce").sum(min_count=1)
        den = pd.to_numeric(window.get(den_tag), errors="coerce").sum(min_count=1)
        rate = float(num / den) if pd.notna(num) and pd.notna(den) and den > 0 else math.nan
        return float(num), float(den), rate

    tnum, tden, target_rate = aggregate_rate(
        HISTORY_METRICS["target_num"], HISTORY_METRICS["target_count"]
    )
    snum, sden, shuffled_rate = aggregate_rate(
        HISTORY_METRICS["shuffled_num"], HISTORY_METRICS["shuffled_count"]
    )
    result.update(
        target_num_sum=tnum,
        target_event_count_sum=tden,
        target_hit_rate_aggregated=target_rate,
        shuffled_num_sum=snum,
        shuffled_event_count_sum=sden,
        shuffled_hit_rate_aggregated=shuffled_rate,
        target_hit_rate_delta=target_rate - shuffled_rate,
        target_hit_advantage=(target_rate / shuffled_rate - 1.0)
        if shuffled_rate > 0
        else math.nan,
    )
    summary = dict(run.summary)
    for short, tag in SNAPSHOT_METRICS.items():
        result[short] = summary.get(tag, math.nan)
    for short, tag in CUMULATIVE_METRICS.items():
        result[short] = summary.get(tag, math.nan)
    return result


def paired_contrast(
    frame: pd.DataFrame,
    factor: str,
    level_a: str,
    level_b: str,
    metrics: list[str],
) -> pd.DataFrame:
    other = [
        column
        for column in ["encoder_credit", "retirement", "goal_conditioning", "seed"]
        if column != factor
    ]
    left = frame.loc[frame[factor] == level_a, other + metrics].set_index(other)
    right = frame.loc[frame[factor] == level_b, other + metrics].set_index(other)
    common = left.index.intersection(right.index)
    rows = []
    for metric in metrics:
        delta = right.loc[common, metric] - left.loc[common, metric]
        rows.append(
            {
                "contrast": f"{level_b}-{level_a}",
                "metric": metric,
                "n_pairs": int(delta.notna().sum()),
                "mean_delta": float(delta.mean()),
                "sd_delta": float(delta.std(ddof=1)),
                "positive_pairs": int((delta > 0).sum()),
                "negative_pairs": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def condition_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(
        ["encoder_credit", "retirement", "goal_conditioning"], sort=True
    )[metrics]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).add_suffix("_sd")
    count = grouped.count().add_suffix("_n")
    return pd.concat([mean, sd, count], axis=1).reset_index()


def plot_family(
    frame: pd.DataFrame,
    panels: list[tuple[str, str, float]],
    output_stem: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    retirements = ["MON", "DIRO", "PREDO"]
    x = np.arange(len(retirements))
    styles = {
        ("ARR", "LEG"): ("#0072B2", "o", "-"),
        ("ARR", "FILM"): ("#56B4E9", "s", "-"),
        ("SRC", "LEG"): ("#D55E00", "o", "--"),
        ("SRC", "FILM"): ("#E69F00", "s", "--"),
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(12.8, 5.2), constrained_layout=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (metric, title, scale) in zip(axes, panels):
        for (credit, goal), (color, marker, linestyle) in styles.items():
            means, sds = [], []
            for retirement in retirements:
                values = frame.loc[
                    (frame.encoder_credit == credit)
                    & (frame.goal_conditioning == goal)
                    & (frame.retirement == retirement),
                    metric,
                ].astype(float) * scale
                means.append(values.mean())
                sds.append(values.std(ddof=1))
                jitter = -0.055 if goal == "LEG" else 0.055
                ax.scatter(
                    np.full(len(values), x[retirements.index(retirement)] + jitter),
                    values,
                    color=color,
                    alpha=0.32,
                    s=25,
                    linewidths=0,
                    zorder=2,
                )
            ax.errorbar(
                x,
                means,
                yerr=sds,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                capsize=3,
                label=f"{credit}-{goal}",
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xticks(x, retirements)
        ax.set_xlabel("retirement")
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Saturday batch at 75M (points: seeds; bars: mean ± SD, n=3)", y=0.985)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_experiments/results/recent_batches_audit_20260906"),
    )
    parser.add_argument("--terminal-start", type=float, default=70_000_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=120)
    runs = list(
        api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"group": GROUP},
            per_page=100,
        )
    )
    if len(runs) != 36:
        raise RuntimeError(f"Expected 36 runs, found {len(runs)}")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, run, args.terminal_start): run for run in runs}
        for future in as_completed(futures):
            records.append(future.result())
    frame = pd.DataFrame(records).sort_values(
        ["encoder_credit", "retirement", "goal_conditioning", "seed"]
    )
    if set(frame.state) != {"finished"}:
        raise RuntimeError(f"Non-finished states: {frame.state.value_counts().to_dict()}")
    # Scalar logging may stop one learner report before the exact training cap.
    # The synchronized spatial snapshot below remains exactly targeted at 75M.
    if frame.max_env_steps.min() < 74_900_000:
        raise RuntimeError(f"Minimum terminal step {frame.max_env_steps.min():g}")
    if not np.allclose(frame.spatial_target_steps.astype(float), 75_000_000):
        raise RuntimeError("Not every online place-field snapshot is synchronized at 75M")

    metrics = [
        "coverage_auc",
        "dg_density",
        "dg_silent_fraction",
        "dg_usage_entropy",
        "spatial_information",
        "active_map_cosine",
        "mono_field_fraction",
        "active_unit_fraction",
        "unique_peak_bins",
        "path_efficiency",
        "option_success",
        "action_probability_tv",
        "action_sensitivity",
        "target_hit_rate_aggregated",
        "shuffled_hit_rate_aggregated",
        "target_hit_rate_delta",
        "target_hit_advantage",
        "largest_scc",
        "reachable_pair_fraction",
        "outgoing_node_fraction",
        "top3_incoming_share",
        "reliable_global_efficiency",
        "grounded_controllability",
        "attempt_coverage",
        "fully_tested_count",
        "predictive_gap",
        "credit_replay_match",
        "behavior_replay_mismatch",
        "stale_generation_fraction",
        "recruitment_total",
        "repeat_total",
        "goal_adapter_reset_total",
        "dropped_rollouts_total",
        "dropped_decisions_total",
        "deferred_updates_total",
    ]
    frame.to_csv(args.output_dir / "saturday_terminal_per_run.csv", index=False)
    condition_summary(frame, metrics).to_csv(
        args.output_dir / "saturday_terminal_condition_summary.csv", index=False
    )

    contrasts = [
        paired_contrast(frame, "encoder_credit", "ARR", "SRC", metrics),
        paired_contrast(frame, "goal_conditioning", "LEG", "FILM", metrics),
        paired_contrast(
            frame.loc[frame.retirement.isin(["MON", "DIRO"])],
            "retirement",
            "MON",
            "DIRO",
            metrics,
        ),
        paired_contrast(
            frame.loc[frame.retirement.isin(["MON", "PREDO"])],
            "retirement",
            "MON",
            "PREDO",
            metrics,
        ),
    ]
    pd.concat(contrasts, ignore_index=True).to_csv(
        args.output_dir / "saturday_paired_main_effects.csv", index=False
    )

    plot_family(
        frame,
        [
            ("spatial_information", "Spatial information", 1.0),
            ("active_map_cosine", "Across-unit map cosine ↓", 1.0),
            ("mono_field_fraction", "Mono-field units (%)", 100.0),
        ],
        args.output_dir / "saturday_representation",
    )
    plot_family(
        frame,
        [
            ("action_probability_tv", "Target action TV", 1.0),
            ("target_hit_advantage", "Target vs shuffled advantage (%)", 100.0),
            ("option_success", "Intentional option success", 1.0),
        ],
        args.output_dir / "saturday_control",
    )
    plot_family(
        frame,
        [
            ("reachable_pair_fraction", "Reliable reachable-pair fraction", 1.0),
            ("top3_incoming_share", "Top-three incoming share ↓", 1.0),
            ("grounded_controllability", "Grounded controllability", 1.0),
        ],
        args.output_dir / "saturday_graph",
    )

    metadata = {
        "study": "saturday_batch_20260905",
        "study_sha256": STUDY_SHA256,
        "wandb_entity": ENTITY,
        "wandb_project": PROJECT,
        "wandb_group": GROUP,
        "sample": "36 finished runs; 12 matched cells; seeds 8, 99, 123",
        "window": "arithmetic means over train/env_steps >= 70M; event rates use pooled numerators/denominators within each run",
        "spatial": "final synchronized online 10k-decision snapshot targeted at 75M",
        "uncertainty": "figures show across-seed standard deviation; paired contrasts retain matched seed and all other factors",
        "transforms": "target_hit_advantage = target_rate / shuffled_rate - 1; percentages multiply fractions by 100",
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
