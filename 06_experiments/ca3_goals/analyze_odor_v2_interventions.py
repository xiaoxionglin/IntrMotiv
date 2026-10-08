"""Analyze the two frozen odor/CA3 command-intervention protocols.

Inputs are exact manifest row outputs from the established evaluator. The
production StudySpec supplies every condition and seed identity; a separate
evaluation-only StudySpec changes only the first-distinct stopping rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts, summarize_records


PRODUCTION_SHA = "c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de"
FIRST_DISTINCT_SHA = "602ea435864c7a193d0563c365e406c93be91d80aceb3cfdbc8b5da8be1b8000"
SELECTORS = ("all16", "random4", "hebb4", "random8", "hebb8")
METRICS = ("command_success", "matched_shuffle_success", "command_minus_shuffle",
           "timeout_rate", "censored_rate", "wrong_first_rate", "action_sensitivity",
           "mean_deadline", "deadline_64_fraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--target-hit-raw", type=Path, required=True)
    parser.add_argument("--first-distinct-raw", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_protocol(raw: Path, protocol: str, expected: dict[tuple[str, int], object]) -> list[dict]:
    summaries = sorted(raw.glob("*/intervention_summary.json"))
    if len(summaries) != len(expected):
        raise ValueError(f"{protocol}: expected {len(expected)} summaries, found {len(summaries)}")
    expected_sha = PRODUCTION_SHA if protocol == "target_hit" else FIRST_DISTINCT_SHA
    records = []
    seen = set()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        key = (summary["condition"], int(summary["seed"]))
        if key not in expected or key in seen:
            raise ValueError(f"{protocol}: unexpected or duplicate condition/seed {key}")
        seen.add(key)
        if summary["study_sha256"] != expected_sha or summary["termination"] != protocol:
            raise ValueError(f"{protocol}: wrong study or stopping rule in {summary_path}")
        remote_raw = Path(summary["manifest"]).parent / "interventions" / "raw"
        if int(summary["checkpoint_frames"]) != 75_005_952:
            raise ValueError(f"{protocol}: unexpected checkpoint in {summary_path}")
        if not (summary["policy_frozen"] and summary["graph_frozen"] and summary["dg_frozen"]):
            raise ValueError(f"{protocol}: unfrozen evaluator in {summary_path}")
        trial_path = summary_path.with_name("intervention_trials.csv")
        trials = pd.read_csv(trial_path)
        if len(trials) != summary["trial_count"] or trials.empty:
            raise ValueError(f"{protocol}: trial count mismatch in {trial_path}")
        if not trials.source.between(0, 15).all() or not trials.target.between(0, 15).all():
            raise ValueError(f"{protocol}: invalid DG identity in {trial_path}")
        if (trials.source == trials.target).any():
            raise ValueError(f"{protocol}: self-target in {trial_path}")
        if trials.deadline.isna().any() or (trials.deadline <= 0).any():
            raise ValueError(f"{protocol}: invalid deadline in {trial_path}")
        pair_counts = trials.groupby(["source", "target"]).size()
        if (pair_counts > 5).any() or int((pair_counts == 5).sum()) != summary["ordered_pairs_complete"]:
            raise ValueError(f"{protocol}: pair count mismatch in {trial_path}")
        if summary["ordered_pairs_eligible"] != 240:
            raise ValueError(f"{protocol}: expected all 240 directed pairs")
        expected_reasons = {"target_hit", "timeout", "censored_boundary"} if protocol == "target_hit" else {
            "first_distinct", "timeout", "censored_boundary"}
        if not set(trials.completion_reason).issubset(expected_reasons):
            raise ValueError(f"{protocol}: unexpected completion reason in {trial_path}")
        success = float(trials.success.mean())
        shuffled = float(trials.shuffled_success.mean())
        if not np.isclose(success, summary["executed_target_success_rate"]) or not np.isclose(
                shuffled, summary["matched_shuffled_target_success_rate"]):
            raise ValueError(f"{protocol}: summary/trial success disagreement in {trial_path}")
        wrong_first = ((trials.completion_reason == "first_distinct") & (trials.outcome != trials.target))
        run = expected[key]
        records.append({
            "protocol": protocol, "condition": run.condition, "seed": run.seed,
            **run.factors,
            "checkpoint_frames": int(summary["checkpoint_frames"]),
            "trial_count": len(trials),
            "complete_pairs": int(summary["ordered_pairs_complete"]),
            "command_success": success,
            "matched_shuffle_success": shuffled,
            "command_minus_shuffle": success - shuffled,
            "timeout_rate": float((trials.completion_reason == "timeout").mean()),
            "censored_rate": float((trials.completion_reason == "censored_boundary").mean()),
            "wrong_first_rate": float(wrong_first.mean()) if protocol == "first_distinct" else np.nan,
            "action_sensitivity": float(summary["mean_counterfactual_action_sensitivity"]),
            "mean_deadline": float(trials.deadline.mean()),
            "deadline_64_fraction": float((trials.deadline == 64).mean()),
            "summary_workspace_path": str(remote_raw / summary_path.parent.name / summary_path.name),
            "trials_workspace_path": str(remote_raw / summary_path.parent.name / trial_path.name),
            "summary_sha256": sha256(summary_path),
            "trials_sha256": sha256(trial_path),
        })
    if seen != set(expected):
        raise ValueError(f"{protocol}: missing runs: {sorted(set(expected) - seen)}")
    return records


def plot_protocols(frame: pd.DataFrame, out_path: Path) -> None:
    font_path = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"),
                                           fallback_to_default=False))
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"A scalable font is required, found {font_path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    protocols = [name for name in ("target_hit", "first_distinct") if name in set(frame.protocol)]
    fig, axes = plt.subplots(len(protocols), 2, figsize=(15, 6 * len(protocols)), squeeze=False)
    for row, protocol in enumerate(protocols):
        for col, odor in enumerate(("off", "on")):
            ax = axes[row, col]
            data = frame[(frame.protocol == protocol) & (frame.odor == odor)]
            centers = np.arange(len(SELECTORS))
            for shift, metric, label, color in ((-0.17, "command_success", "Commanded", "#2675ad"),
                                                (0.17, "matched_shuffle_success", "Shuffled label", "#ca6b17")):
                groups = [data[data.goal_set == selector][metric].to_numpy() for selector in SELECTORS]
                means = [values.mean() for values in groups]
                errors = [values.std(ddof=1) / np.sqrt(len(values)) for values in groups]
                ax.bar(centers + shift, means, width=0.32, color=color, label=label)
                ax.errorbar(centers + shift, means, yerr=errors, fmt="none", color="black", capsize=3)
            ax.set_xticks(centers, [s.upper() for s in SELECTORS], rotation=25, ha="right")
            ax.set_ylim(0, 0.85 if protocol == "target_hit" else 0.11)
            if protocol == "first_distinct":
                ax.axhline(1 / 15, color="#555555", linestyle=":", linewidth=2,
                           label="Uniform 1/15 reference" if col == 0 else None)
            ax.set_title(f"{protocol.replace('_', ' ').title()} · odor {odor.upper()}")
            ax.set_ylabel("Trial fraction" if col == 0 else "")
            ax.grid(axis="y", alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=14)
    fig.suptitle("Frozen command probes · mean ± SEM across four seeds", fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def first_outcome_by_target(raw: Path, expected: dict[tuple[str, int], object]) -> pd.DataFrame:
    """Retain wrong-goal identity, including timeouts/censoring as outcome -1."""
    rows = []
    for summary_path in sorted(raw.glob("*/intervention_summary.json")):
        summary = json.loads(summary_path.read_text())
        run = expected[(summary["condition"], int(summary["seed"]))]
        trials = pd.read_csv(summary_path.with_name("intervention_trials.csv"), usecols=["target", "outcome"])
        counts = trials.groupby(["target", "outcome"]).size()
        for target in range(16):
            denominator = int((trials.target == target).sum())
            if denominator == 0:
                raise ValueError(f"No first-distinct trials for target {target} in {summary_path}")
            for outcome in range(-1, 16):
                count = int(counts.get((target, outcome), 0))
                rows.append({"condition": run.condition, "seed": run.seed, **run.factors,
                             "target": target, "first_outcome": outcome,
                             "count": count, "target_trials": denominator})
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["condition", "odor", "goal_set", "target", "first_outcome"],
                            as_index=False)[["count", "target_trials"]].sum()
    grouped["fraction"] = grouped["count"] / grouped["target_trials"]
    return grouped


def plot_first_outcomes(frame: pd.DataFrame, odor: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, selector in zip(axes, ("random4", "hebb4")):
        group = frame[(frame.odor == odor) & (frame.goal_set == selector)
                      & (frame.first_outcome >= 0)]
        matrix = group.pivot(index="target", columns="first_outcome", values="fraction")
        matrix = matrix.reindex(index=range(16), columns=range(16), fill_value=0)
        image = ax.imshow(matrix.to_numpy(), origin="lower", vmin=0, vmax=0.12,
                          cmap="viridis", interpolation="nearest", aspect="equal")
        ax.plot([-0.5, 15.5], [-0.5, 15.5], color="white", linestyle=":", linewidth=2)
        ax.set_title(selector.upper(), fontsize=30)
        ax.set_xticks([0, 4, 8, 12, 15])
        ax.set_yticks([0, 4, 8, 12, 15])
        ax.tick_params(labelsize=27)
        ax.set_xlabel("First distinct DG identity", fontsize=29)
    axes[0].set_ylabel("Commanded DG identity", fontsize=29)
    fig.subplots_adjust(left=0.09, right=0.82, bottom=0.16, top=0.76, wspace=0.18)
    colorbar_axis = fig.add_axes([0.86, 0.19, 0.025, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Fraction of trials", fontsize=27)
    colorbar.set_ticks([0, 0.03, 0.06, 0.09, 0.12])
    colorbar.ax.tick_params(labelsize=24)
    fig.suptitle(f"Odor {odor.upper()} · first distinct outcomes at 75M", fontsize=31, y=0.97)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    study = load_study(args.study)
    if study.provenance()["study_sha256"] != PRODUCTION_SHA:
        raise ValueError("Production StudySpec fingerprint changed")
    expected = {(run.condition, run.seed): run for run in study.expand_runs()}
    records = read_protocol(args.target_hit_raw, "target_hit", expected)
    if args.first_distinct_raw:
        records.extend(read_protocol(args.first_distinct_raw, "first_distinct", expected))
    frame = pd.DataFrame(records).sort_values(["protocol", "odor", "goal_set", "seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "interventions_per_run.csv", index=False)
    groups = []
    contrasts_per_seed = []
    contrasts_summary = []
    for protocol, subset in frame.groupby("protocol"):
        data = subset.to_dict("records")
        metrics = [m for m in METRICS if not (m == "wrong_first_rate" and protocol == "target_hit")]
        groups.extend({"protocol": protocol, **row} for row in summarize_records(data, ["odor", "goal_set"], metrics))
        detail, summary = linear_contrasts(data, metrics,
                                           study.analysis.get("contrast_group_by", []),
                                           study.analysis.get("replicate_by", ["seed"]),
                                           study.analysis["contrasts"])
        contrasts_per_seed.extend({"protocol": protocol, **row} for row in detail)
        contrasts_summary.extend({"protocol": protocol, **row} for row in summary)
    pd.DataFrame(groups).to_csv(args.out_dir / "interventions_condition_summary.csv", index=False)
    pd.DataFrame(contrasts_per_seed).to_csv(args.out_dir / "interventions_paired_contrasts_per_seed.csv", index=False)
    pd.DataFrame(contrasts_summary).to_csv(args.out_dir / "interventions_paired_contrasts_summary.csv", index=False)
    plot_protocols(frame, args.out_dir / "interventions_success.png")
    if args.first_distinct_raw:
        outcomes = first_outcome_by_target(args.first_distinct_raw, expected)
        outcomes.to_csv(args.out_dir / "first_outcome_by_target.csv", index=False)
        for odor in ("off", "on"):
            plot_first_outcomes(outcomes, odor, args.out_dir / f"first_outcomes_{odor}.png")
    provenance = {"schema": study.raw["schema"], "workflow_version": study.declared_workflow_version,
                  "production_study_sha256": PRODUCTION_SHA,
                  "first_distinct_analysis_study_sha256": FIRST_DISTINCT_SHA if args.first_distinct_raw else None,
                  "protocols": sorted(frame.protocol.unique()), "runs_per_protocol": len(expected),
                  "source_roots": {protocol: sorted(set(group.summary_workspace_path.map(
                      lambda path: str(Path(path).parents[1]))))
                      for protocol, group in frame.groupby("protocol")}}
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Analyzed {len(frame)} frozen intervention runs across {len(set(frame.protocol))} protocol(s)")


if __name__ == "__main__":
    main()
