"""Render the Navigation8 algorithm-screen interim figures from canonical CSVs.

This is deliberately a report adapter: run discovery, snapshot validation,
place-field measurements, graph extraction, and trajectory segmentation are
performed by ``hpc_runs.intrmotiv_study collect-spatial``.  The adapter only
selects the common 5M/25M/75M milestones, keeps every seed visible, and renders
configuration-labelled summaries suitable for the accompanying experiment note.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent / "data" / "navigation8_algorithm_screen_interim_20260916"
INPUT = ROOT / "online_spatial" / "per_snapshot.csv"
OUT = ROOT / "figures"

ORDER = [
    "SCR_ARR_DIRS", "SAT_ARR_DIRO_FILM", "DGP_HIT_JOINT_LEG",
    "CPD_GATE_CA3_BPTT", "W_REF_STOP", "W_REF_JOINT",
]
LABELS = {
    "SCR_ARR_DIRS": "SCR: arrival-dir, silent gate",
    "SAT_ARR_DIRO_FILM": "SAT: arrival-dir, open FiLM",
    "DGP_HIT_JOINT_LEG": "DGP: hit, joint legacy",
    "CPD_GATE_CA3_BPTT": "CPD: CA3 gate + BPTT",
    "W_REF_STOP": "W-ref: stop-gradient",
    "W_REF_JOINT": "W-ref: joint gradient",
}
COLORS = {
    "SCR_ARR_DIRS": "#0072B2", "SAT_ARR_DIRO_FILM": "#E69F00",
    "DGP_HIT_JOINT_LEG": "#009E73", "CPD_GATE_CA3_BPTT": "#D55E00",
    "W_REF_STOP": "#7A7A7A", "W_REF_JOINT": "#CC79A7",
}
MARKERS = {8: "o", 99: "s", 123: "^"}
TARGETS = [5_000_000, 25_000_000, 75_000_000]


def setup_style() -> Path:
    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if not font.is_file() or font.suffix not in {".ttf", ".otf"}:
        raise RuntimeError("A scalable DejaVu Sans font is required for figures")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    return font


def finish(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def strip(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.7)


def terminal_representation(df: pd.DataFrame) -> None:
    q = df[df.target_env_steps.eq(75_000_000)]
    panels = [
        ("active_only_map_cosine", "Active-only map cosine\n(lower = less overlap)", (0, 0.65)),
        ("mono_field_unit_fraction", "Mono-field eligible-unit fraction", (0, 1)),
        ("active_unit_mean_spatial_information", "Amplitude-weighted spatial score", (0, None)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    xs = np.arange(len(ORDER))
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for index, base in enumerate(ORDER):
            values = q[q.base.eq(base)].set_index("seed")[metric]
            for seed, marker in MARKERS.items():
                ax.scatter(index, values.loc[seed], color=COLORS[base], marker=marker,
                           s=70, edgecolor="white", linewidth=0.6, zorder=3)
            ax.plot([index - 0.16, index + 0.16], [values.mean()] * 2,
                    color=COLORS[base], linewidth=2.2, zorder=2)
        ax.set_ylabel(ylabel); ax.set_xticks(xs, [LABELS[x] for x in ORDER], rotation=42, ha="right")
        ax.set_xlim(-0.55, len(ORDER) - 0.45); ax.set_ylim(*ylim); strip(ax)
    fig.text(0.5, 0.01, "75M cached online-spatial snapshot; points: seeds 8 / 99 / 123; horizontal segments: unweighted seed mean (n=3).",
             ha="center", fontsize=12)
    fig.subplots_adjust(bottom=0.34, wspace=0.35)
    finish(fig, "representation_at_75m")


def seed99_trajectories(df: pd.DataFrame) -> None:
    q = df[df.seed.eq(99)]
    panels = [
        ("active_only_map_cosine", "Active-only map cosine\n(lower = less overlap)", (0, 0.7)),
        ("mono_field_unit_fraction", "Mono-field eligible-unit fraction", (0, 1)),
        ("stationary_step_fraction", "Stationary decision fraction", (0, 1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharex=True)
    x = np.array(TARGETS) / 1e6
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for base in ORDER:
            values = q[q.base.eq(base)].set_index("target_env_steps")[metric].reindex(TARGETS)
            ax.plot(x, values, marker="s", markersize=5.5, color=COLORS[base], linewidth=2,
                    label=LABELS[base])
        ax.set_xlabel("Training frames (millions)"); ax.set_ylabel(ylabel); ax.set_xticks(x)
        ax.set_ylim(*ylim); strip(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.03, "Seed 99 only; each point is the cached latest-100k-decision behavior window at the requested milestone.",
             ha="center", fontsize=12)
    fig.subplots_adjust(top=0.78, bottom=0.19, wspace=0.34)
    finish(fig, "seed99_spatial_trajectory")


def graph_trajectory(df: pd.DataFrame) -> None:
    graph_bases = ORDER[:4]
    q = df[df.base.isin(graph_bases)]
    panels = [
        ("graph_prospective_success_fraction", "Previously-known edge success", (0, 1)),
        ("graph_reliable_global_efficiency", "Reliable global efficiency", (0, 1)),
        ("graph_grounded_controllability", "Grounded controllability", (0, 1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharex=True)
    x = np.array(TARGETS) / 1e6
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for base in graph_bases:
            summary = q[q.base.eq(base)].groupby("target_env_steps")[metric].agg(["mean", "std"]).reindex(TARGETS)
            ax.errorbar(x, summary["mean"], yerr=summary["std"], marker="o", capsize=3,
                        color=COLORS[base], linewidth=2, label=LABELS[base])
        ax.set_xlabel("Training frames (millions)"); ax.set_ylabel(ylabel); ax.set_xticks(x)
        ax.set_ylim(*ylim); strip(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.03, "Graph-enabled configurations only; points are unweighted three-seed means and bars are sample SD. W-ref configurations have no graph payload.",
             ha="center", fontsize=12)
    fig.subplots_adjust(top=0.78, bottom=0.19, wspace=0.34)
    finish(fig, "graph_trajectory")


def main() -> None:
    font = setup_style()
    df = pd.read_csv(INPUT)
    required = len(ORDER) * len(MARKERS) * len(TARGETS)
    if len(df) != required or df.run_name.nunique() != 18:
        raise RuntimeError(f"Expected {required} matched snapshots from 18 runs, found {len(df)} rows / {df.run_name.nunique()} runs")
    if sorted(df.target_env_steps.unique()) != TARGETS:
        raise RuntimeError("Unexpected milestone set")
    terminal_representation(df)
    seed99_trajectories(df)
    graph_trajectory(df)
    (ROOT / "figure_metadata.json").write_text(json.dumps({
        "input": str(INPUT), "study_id": "navigation8_algorithm_screen_20260909",
        "study_sha256": "c435ddac609945336d1e42ca16ed0bcc8fd2d46be13eef167a0085b39b682094",
        "milestones": TARGETS, "replication": "seeds 8, 99, 123",
        "font": str(font), "representation_summary": "75M unweighted seed mean with all seed points",
        "graph_summary": "three-seed mean and sample SD; graph-enabled configurations only",
        "trajectory": "seed 99 cached online-spatial windows; policy-driven, not a fixed-trajectory stability test",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
