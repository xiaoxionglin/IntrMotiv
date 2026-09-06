#!/usr/bin/env python3
"""Largest-common-step diagnostic for the canceled edge-exploration study."""

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
PROJECT = "SF_IntrMotiv_ControllabilityEdgeExploration"
GROUP = "intrmotiv_controllability_edge_exploration_20260903"
STUDY_SHA256 = "4b940a37e09bdabc7efd8bbedc21053194b5288543fb22d32d0c9bc1323e9734"
RUN_RE = re.compile(
    r"^\d+_(CEE_(X0|X1)_(SH|SEP)_(NODE|EDGE)_(G0|G1)_S(8|99|123))_"
)
STEP = "train/env_steps"
METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "dg_density": "intrmotiv/dg/density",
    "dg_silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "dg_usage_entropy": "intrmotiv/dg/usage_entropy",
    "option_success": "intrmotiv/hrl/option_success_fraction",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "free_mode_fraction": "intrmotiv/hrl/mode/free_fraction",
    "goal_mode_fraction": "intrmotiv/hrl/mode/goal_fraction",
    "probe_mode_fraction": "intrmotiv/hrl/mode/probe_fraction",
    "probe_success_rate": "intrmotiv/hrl/edge/probe_success_rate",
    "probe_timeout_rate": "intrmotiv/hrl/edge/probe_timeout_rate",
    "promotion_rate": "intrmotiv/hrl/edge/promotion_rate",
    "edge_reliability": "intrmotiv/hrl/edge_reliability_mean",
    "largest_scc": "intrmotiv/hrl/reliable/largest_scc",
    "reachable_pair_fraction": "intrmotiv/hrl/reliable/reachable_pair_fraction",
    "outgoing_node_fraction": "intrmotiv/hrl/reliable/outgoing_node_fraction",
    "top3_incoming_share": "intrmotiv/hrl/reliable/top3_incoming_confidence_share",
    "goal_loop_fraction": "intrmotiv/hrl/by_mode/goal_loop_fraction",
    "goal_straightness": "intrmotiv/hrl/by_mode/goal_straightness",
    "probe_loop_fraction": "intrmotiv/hrl/by_mode/probe_loop_fraction",
    "probe_straightness": "intrmotiv/hrl/by_mode/probe_straightness",
    "behavior_replay_mismatch": "intrmotiv/hrl/behavior_replay_mismatch",
}
CUMULATIVE = {
    "recruitment_total": "intrmotiv/dg/recruitment/total",
    "repeat_total": "intrmotiv/dg/recruitment/repeat_total",
}


def parse(name: str) -> dict[str, object]:
    match = RUN_RE.match(name)
    if not match:
        raise ValueError(name)
    canonical, representation, head, manager, geometry, seed = match.groups()
    return {
        "run_name": canonical,
        "representation": representation,
        "exploration_head": head,
        "manager_objective": manager,
        "geometry": geometry,
        "seed": int(seed),
    }


def mean(frame: pd.DataFrame, tag: str) -> float:
    value = pd.to_numeric(frame.get(tag), errors="coerce")
    return float(value.mean()) if value.notna().any() else math.nan


def fetch(run, start: float, end: float) -> dict[str, object]:
    frame = pd.DataFrame(
        run.history(
            samples=5000,
            keys=[STEP, *METRICS.values(), *CUMULATIVE.values()],
            pandas=False,
        )
    )
    frame[STEP] = pd.to_numeric(frame[STEP], errors="coerce")
    window = frame.loc[(frame[STEP] >= start) & (frame[STEP] <= end)]
    if window.empty:
        raise RuntimeError(f"{run.name}: empty {start:g}--{end:g} window")
    record = {
        **parse(run.name),
        "wandb_name": run.name,
        "state": run.state,
        "observed_max_steps": float(frame[STEP].max()),
        "aligned_start": start,
        "aligned_end": end,
        "window_rows": len(window),
    }
    for short, tag in METRICS.items():
        record[short] = mean(window, tag)
    for short, tag in CUMULATIVE.items():
        values = pd.to_numeric(frame.loc[frame[STEP] <= end, tag], errors="coerce").dropna()
        record[short] = float(values.iloc[-1]) if not values.empty else math.nan
    return record


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRICS) + list(CUMULATIVE)
    specifications = [
        ("representation", "X0", "X1"),
        ("exploration_head", "SH", "SEP"),
        ("manager_objective", "NODE", "EDGE"),
        ("geometry", "G0", "G1"),
    ]
    output = []
    all_factors = [
        "representation",
        "exploration_head",
        "manager_objective",
        "geometry",
        "seed",
    ]
    for factor, a, b in specifications:
        index = [item for item in all_factors if item != factor]
        left = frame.loc[frame[factor] == a].set_index(index)
        right = frame.loc[frame[factor] == b].set_index(index)
        common = left.index.intersection(right.index)
        for metric in metrics:
            delta = right.loc[common, metric] - left.loc[common, metric]
            output.append(
                {
                    "contrast": f"{b}-{a}",
                    "metric": metric,
                    "n_pairs": int(delta.notna().sum()),
                    "mean_delta": float(delta.mean()),
                    "positive_pairs": int((delta > 0).sum()),
                    "negative_pairs": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(output)


def plot(frame: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panels = [
        ("dg_silent_fraction", "Silent DG fraction"),
        ("recruitment_total", "Total replacements"),
        ("option_success", "Intentional option success"),
        ("reachable_pair_fraction", "Reliable reachable-pair fraction"),
    ]
    managers = ["NODE", "EDGE"]
    x = np.arange(2)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=False)
    for ax, (metric, title) in zip(axes.flat, panels):
        for representation, color, marker in [
            ("X0", "#0072B2", "o"),
            ("X1", "#D55E00", "s"),
        ]:
            means, sds = [], []
            for manager in managers:
                values = frame.loc[
                    (frame.representation == representation)
                    & (frame.manager_objective == manager),
                    metric,
                ]
                means.append(values.mean())
                sds.append(values.std(ddof=1))
                jitter = -0.04 if representation == "X0" else 0.04
                ax.scatter(
                    np.full(len(values), x[managers.index(manager)] + jitter),
                    values,
                    color=color,
                    alpha=0.22,
                    s=22,
                    linewidths=0,
                )
            ax.errorbar(
                x,
                means,
                yerr=sds,
                color=color,
                marker=marker,
                linewidth=2,
                capsize=3,
                label=representation,
            )
        ax.set_title(title)
        ax.set_xticks(x, managers)
        ax.set_xlabel("manager objective")
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.suptitle("Canceled edge-exploration study at largest common step (mean ± SD)", y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_experiments/results/recent_batches_audit_20260906"),
    )
    parser.add_argument("--window-width", type=float, default=5_000_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=100))
    if len(runs) != 48:
        raise RuntimeError(f"Expected 48 runs, got {len(runs)}")
    summary_steps = [float(run.summary[STEP]) for run in runs]
    aligned_end = min(summary_steps)
    aligned_start = aligned_end - args.window_width
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, run, aligned_start, aligned_end): run for run in runs
        }
        for future in as_completed(futures):
            records.append(future.result())
    frame = pd.DataFrame(records).sort_values(
        ["representation", "exploration_head", "manager_objective", "geometry", "seed"]
    )
    frame.to_csv(args.output_dir / "edge_aligned_per_run.csv", index=False)
    metrics = list(METRICS) + list(CUMULATIVE)
    grouped = frame.groupby(
        ["representation", "exploration_head", "manager_objective", "geometry"]
    )[metrics]
    pd.concat(
        [grouped.mean().add_suffix("_mean"), grouped.std(ddof=1).add_suffix("_sd")],
        axis=1,
    ).reset_index().to_csv(args.output_dir / "edge_aligned_condition_summary.csv", index=False)
    paired_effects(frame).to_csv(args.output_dir / "edge_aligned_paired_effects.csv", index=False)
    plot(frame, args.output_dir / "edge_aligned")
    metadata = {
        "study": "controllability_edge_exploration_20260903",
        "study_sha256": STUDY_SHA256,
        "state": "all 48 jobs deliberately canceled; uneven terminal steps",
        "aligned_end": aligned_end,
        "aligned_start": aligned_start,
        "sample": "48 runs; factor-effect pairs preserve all other factors and seed",
        "uncertainty": "figure shows SD after pooling head, geometry, and seed within representation × manager; this is diagnostic, not a production contrast",
    }
    (args.output_dir / "edge_aligned_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
