"""Summarize canonical odor/CA3 field trajectories and graph-edge exposure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from hpc_runs.intrmotiv_study import load_study


SELECTORS = ("all16", "random4", "hebb4", "random8", "hebb8")
COLORS = {"all16": "#242424", "random4": "#2675ad", "hebb4": "#2675ad",
          "random8": "#ca6b17", "hebb8": "#ca6b17"}
STYLES = {"all16": "-", "random4": "-", "hebb4": "--", "random8": "-", "hebb8": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def graph_exposure(snapshots: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    terminal = snapshots[snapshots.target_env_steps == 75_000_000].copy()
    terminal_edges = edges[(edges.target_env_steps == 75_000_000)
                           & (edges.source_unit != edges.target_unit)]
    rows = []
    for name, group in terminal_edges.groupby("run_name"):
        attempts = group.prospective_attempts.to_numpy(dtype=float)
        positive = attempts[attempts > 0]
        probability = positive / positive.sum() if positive.size else np.array([])
        rows.append({"run_name": name,
                     "attempted_directed_edges": len(positive),
                     "effective_attempted_edges": float(np.exp(-(probability * np.log(probability)).sum()))
                     if positive.size else 0.0,
                     "prospective_attempts_from_edges": float(attempts.sum())})
    result = terminal.merge(pd.DataFrame(rows), on="run_name", validate="one_to_one")
    if len(result) != 40 or terminal_edges.groupby("run_name").size().ne(240).any():
        raise ValueError("Expected 40 terminal graphs with 240 directed off-diagonal edges each")
    if not np.allclose(result.graph_prospective_attempt_count,
                       result.prospective_attempts_from_edges, atol=0.1):
        raise ValueError("Edge attempts disagree with canonical graph diagnostics")
    return result


def plot_graph(snapshots: pd.DataFrame, output: Path) -> None:
    summary = snapshots.groupby(["odor", "goal_set", "target_env_steps"], sort=False).graph_reachable_pair_fraction.agg(
        mean="mean", sd="std", n="size").reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for axis, odor in zip(axes, ("off", "on")):
        for selector in SELECTORS:
            group = summary[(summary.odor == odor) & (summary.goal_set == selector)].sort_values(
                "target_env_steps")
            x = group.target_env_steps.to_numpy() / 1e6
            y = group["mean"].to_numpy()
            sem = group.sd.to_numpy() / np.sqrt(group.n.to_numpy())
            axis.plot(x, y, color=COLORS[selector], linestyle=STYLES[selector],
                      marker="o", linewidth=2.6, label=selector.upper())
            axis.fill_between(x, y - sem, y + sem, color=COLORS[selector], alpha=0.1)
        axis.set(title=f"Odor {odor.upper()}", xlabel="Training frames (millions)",
                 xlim=(0, 78), ylim=(0, 1.05))
        axis.set_xticks([5, 10, 25, 50, 75])
        axis.tick_params(labelsize=15)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Reachable directed goal-pair fraction", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=15, ncol=5,
                  loc="lower center", bbox_to_anchor=(0.5, 0.015))
    figure.suptitle("Reliable policy graph: mean ± SEM across four seeds", fontsize=20)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.88, wspace=0.09)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_fields(fields: pd.DataFrame, conditions: dict[str, tuple[str, str]], output: Path) -> None:
    frame = fields[fields.seed == 99].copy()
    frame[["odor", "goal_set"]] = frame.condition.map(conditions).apply(pd.Series)
    if len(frame) != 50:
        raise ValueError("Expected five checkpoint field rows for each seed-99 condition")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metrics = (("active_map_cosine_mean", "Active-only map cosine"),
               ("visited_cells", "Visited spatial bins"))
    for row_index, odor in enumerate(("off", "on")):
        for col_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, col_index]
            for selector in SELECTORS:
                group = frame[(frame.odor == odor) & (frame.goal_set == selector)].sort_values(
                    "target_frames")
                axis.plot(group.target_frames / 1e6, group[metric],
                          color=COLORS[selector], linestyle=STYLES[selector],
                          marker="o", linewidth=2.4, label=selector.upper())
            axis.set_title(f"Odor {odor.upper()} · {label}", fontsize=19)
            axis.set_xticks([5, 10, 25, 50, 75])
            axis.tick_params(labelsize=15)
            axis.grid(alpha=0.25)
            if col_index == 0:
                axis.set_ylabel(label, fontsize=17)
            if row_index == 1:
                axis.set_xlabel("Training frames (millions)", fontsize=17)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=15, ncol=5,
                  loc="lower center", bbox_to_anchor=(0.5, 0.015))
    figure.suptitle("Seed-99 policy rollouts · each checkpoint uses its own trajectory", fontsize=20)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.90, wspace=0.16, hspace=0.28)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    study = load_study(args.study)
    if study.provenance()["study_sha256"] != "c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de":
        raise ValueError("Study fingerprint differs from the launched batch")
    font_path = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"),
                                           fallback_to_default=False))
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"A scalable font is required, found {font_path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17})

    snapshots = pd.read_csv(args.snapshots)
    fields = pd.read_csv(args.fields)
    edges = pd.read_csv(args.edges, usecols=["run_name", "target_env_steps", "source_unit",
                                                 "target_unit", "prospective_attempts"])
    if len(snapshots) != 200 or len(fields) != 70:
        raise ValueError("Expected 200 online snapshots and 70 offline field rollouts")
    conditions = {run.condition: (run.factors["odor"], run.factors["goal_set"])
                  for run in study.expand_runs()}
    terminal = graph_exposure(snapshots, edges)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    terminal.to_csv(args.out_dir / "graph_terminal_exposure.csv", index=False)
    plot_graph(snapshots, args.out_dir / "graph_reachability.png")
    plot_fields(fields, conditions, args.out_dir / "place_field_checkpoint_trajectory.png")
    print("Analyzed 200 graph snapshots and 70 offline field rollouts")


if __name__ == "__main__":
    main()
