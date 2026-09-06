#!/usr/bin/env python3
"""Final 70--75M aligned analysis of the completed DPR study."""

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
PROJECT = "SF_IntrMotiv_DirectionalPredictiveRecruitment"
GROUP = "intrmotiv_directional_predictive_recruitment_20260904"
STUDY_SHA256 = "72b0ac2d04ad7a297a674f96d4f32c85d48dcf89a9fc62abb7243adb22ea53aa"
RUN_RE = re.compile(
    r"^\d+_(DPR_(C05|C13|C15)_(MON|DIR|PRED)_(LEG|FILM)_S(8|99|123))_"
)
STEP = "train/env_steps"
METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "dg_density": "intrmotiv/dg/density",
    "dg_silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "dg_usage_entropy": "intrmotiv/dg/usage_entropy",
    "unit_duty_cycle_max": "intrmotiv/dg/unit_duty_cycle_max",
    "option_success": "intrmotiv/hrl/option_success_fraction",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "target_hit_rate": "intrmotiv/hrl/target_hit_rate",
    "target_hit_lift": "intrmotiv/hrl/target_hit_lift",
    "largest_scc": "intrmotiv/hrl/reliable/largest_scc",
    "reachable_pair_fraction": "intrmotiv/hrl/reliable/reachable_pair_fraction",
    "outgoing_node_fraction": "intrmotiv/hrl/reliable/outgoing_node_fraction",
    "top3_incoming_share": "intrmotiv/hrl/reliable/top3_incoming_confidence_share",
    "attempt_coverage": "intrmotiv/dg/recruitment/attempt_coverage_fraction",
    "fully_tested_count": "intrmotiv/dg/recruitment/fully_tested_count",
    "zero_outdegree_count": "intrmotiv/dg/recruitment/zero_outdegree_count",
    "predictive_eligible_count": "intrmotiv/dg/recruitment/predictive_eligible_count",
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
    canonical, base, recruitment, goal, seed = match.groups()
    return {
        "run_name": canonical,
        "base": base,
        "recruitment": recruitment,
        "goal_conditioning": goal,
        "seed": int(seed),
    }


def mean(frame: pd.DataFrame, tag: str) -> float:
    value = pd.to_numeric(frame.get(tag), errors="coerce")
    return float(value.mean()) if value.notna().any() else math.nan


def fetch(run, terminal_start: float) -> dict[str, object]:
    frame = pd.DataFrame(
        run.history(samples=5000, keys=[STEP, *METRICS.values()], pandas=False)
    )
    frame[STEP] = pd.to_numeric(frame[STEP], errors="coerce")
    window = frame.loc[frame[STEP] >= terminal_start]
    if window.empty:
        raise RuntimeError(f"{run.name}: empty terminal window")
    record = {
        **parse(run.name),
        "wandb_name": run.name,
        "wandb_id": run.id,
        "state": run.state,
        "max_env_steps": float(frame[STEP].max()),
        "window_rows": int(len(window)),
    }
    for short, tag in METRICS.items():
        record[short] = mean(window, tag)
    summary = dict(run.summary)
    for short, tag in CUMULATIVE.items():
        record[short] = summary.get(tag, math.nan)
    return record


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRICS) + list(CUMULATIVE)
    grouped = frame.groupby(["base", "recruitment", "goal_conditioning"])[metrics]
    return pd.concat(
        [grouped.mean().add_suffix("_mean"), grouped.std(ddof=1).add_suffix("_sd")],
        axis=1,
    ).reset_index()


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRICS) + list(CUMULATIVE)
    specifications = [
        ("goal_conditioning", "LEG", "FILM"),
        ("recruitment", "MON", "DIR"),
        ("recruitment", "MON", "PRED"),
    ]
    output = []
    for factor, a, b in specifications:
        subset = frame.loc[frame[factor].isin([a, b])]
        index = [x for x in ["base", "recruitment", "goal_conditioning", "seed"] if x != factor]
        left = subset.loc[subset[factor] == a].set_index(index)
        right = subset.loc[subset[factor] == b].set_index(index)
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
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panels = [
        ("coverage_auc", "Coverage AUC"),
        ("option_success", "Intentional option success"),
        ("action_sensitivity", "Target logit sensitivity"),
        ("reachable_pair_fraction", "Reliable reachable-pair fraction"),
    ]
    recruitments = ["MON", "DIR", "PRED"]
    x = np.arange(3)
    palette = {"C05": "#009E73", "C13": "#D55E00", "C15": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), constrained_layout=False)
    for ax, (metric, title) in zip(axes.flat, panels):
        for base in ["C05", "C13", "C15"]:
            for goal, linestyle, marker, offset in [
                ("LEG", "--", "o", -0.045),
                ("FILM", "-", "s", 0.045),
            ]:
                means, sds = [], []
                for recruitment in recruitments:
                    values = frame.loc[
                        (frame.base == base)
                        & (frame.goal_conditioning == goal)
                        & (frame.recruitment == recruitment),
                        metric,
                    ]
                    means.append(values.mean())
                    sds.append(values.std(ddof=1))
                    ax.scatter(
                        np.full(len(values), x[recruitments.index(recruitment)] + offset),
                        values,
                        color=palette[base],
                        alpha=0.25,
                        s=20,
                        linewidths=0,
                    )
                ax.errorbar(
                    x,
                    means,
                    yerr=sds,
                    color=palette[base],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.8,
                    capsize=3,
                    label=f"{base}-{goal}",
                )
        ax.set_title(title)
        ax.set_xticks(x, recruitments)
        ax.set_xlabel("recruitment")
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
    )
    fig.suptitle("Directional/predictive batch, 70–75M (mean ± SD, n=3)", y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
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
    parser.add_argument("--terminal-start", type=float, default=70_000_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    runs = list(
        api.runs(
            f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=100
        )
    )
    if len(runs) != 54:
        raise RuntimeError(f"Expected 54 runs, got {len(runs)}")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, run, args.terminal_start): run for run in runs}
        for future in as_completed(futures):
            records.append(future.result())
    frame = pd.DataFrame(records).sort_values(
        ["base", "recruitment", "goal_conditioning", "seed"]
    )
    if set(frame.state) != {"finished"} or frame.max_env_steps.min() < 74_900_000:
        raise RuntimeError("DPR terminal alignment/completion check failed")
    frame.to_csv(args.output_dir / "dpr_terminal_per_run.csv", index=False)
    summarize(frame).to_csv(args.output_dir / "dpr_terminal_condition_summary.csv", index=False)
    contrasts(frame).to_csv(args.output_dir / "dpr_paired_main_effects.csv", index=False)
    plot(frame, args.output_dir / "dpr_terminal")
    metadata = {
        "study": "directional_predictive_recruitment_20260904",
        "study_sha256": STUDY_SHA256,
        "sample": "54 finished runs; 18 cells; seeds 8, 99, 123",
        "window": "arithmetic means over train/env_steps >= 70M",
        "uncertainty": "figures show across-seed SD; contrasts retain seed and all other factors",
        "caution": "action_sensitivity is the legacy logit diagnostic; target_hit_lift is unstable and is not treated as causal evidence",
    }
    (args.output_dir / "dpr_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
