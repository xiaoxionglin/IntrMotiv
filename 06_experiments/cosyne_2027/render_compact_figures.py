"""Render letter-page COSYNE panels from cached, provenance-pinned CSVs.

All contrasts keep the training seed as the independent observation. This
script changes presentation only; the upstream analysis tables stay intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "06_experiments/results"
OUT = RESULTS / "cosyne_2027_compact_figures"
SOURCES = {
    "online": RESULTS / "cosyne_2027_online_wandb/arm_summaries.csv",
    "online_pairs": RESULTS / "cosyne_2027_online_wandb/paired_contrasts_by_seed.csv",
    "behavior": RESULTS / "cosyne_2027_behavior_secondary/paired_training_seeds.csv",
    "encoder": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/paired_encoder_fields.csv",
    "peaks": RESULTS / "cosyne_2027_checkpoint_fields/common_panel_analysis/paired_peak_spread_by_seed.csv",
    "credit_presence": ROOT / "06_experiments/reward_credit/results/corrected_core_encoder_worker_20261010/common_panel_encoder_pairs.csv",
}
COLORS = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73"}
FACTOR_ORDER = ("C01", "C05 · temporal hit", "C05 · constant hit", "C15")
FACTOR_LABELS = ("C01", "C05 T", "C05 C", "C15")
BEHAVIOR_PANEL_SIZE_IN = (1.5, 1.8)


def style() -> str:
    path = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    if Path(path).suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"Arial did not resolve to a scalable font: {path}")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "svg.fonttype": "none", "svg.hashsalt": "IntrMotiv-COSYNE-2027-compact",
        "pdf.fonttype": 42,
    })
    return path


def export(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path, dpi=300, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def seed_differences(ax: plt.Axes, frame: pd.DataFrame, groups: tuple[str, ...],
                     *, group_column: str, labels: tuple[str, ...],
                     colors: tuple[str, ...]) -> None:
    ax.axhline(0, color="#888888", lw=0.8, zorder=0)
    for x, (group, color) in enumerate(zip(groups, colors)):
        values = frame[frame[group_column] == group].sort_values("seed")
        if len(values) not in {2, 3, 5} or values.seed.nunique() != len(values):
            raise ValueError(f"unexpected paired-seed count for {group}: {len(values)}")
        offsets = np.linspace(-0.13, 0.13, len(values))
        y = values.difference.to_numpy(float)
        ax.scatter(x + offsets, y, s=17, color=color, linewidth=0,
                   zorder=3)
        ax.plot([x - 0.16, x + 0.16], [np.mean(y)] * 2,
                color="black", lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups)), labels)
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)


def online_course(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.27, top=0.95)
    for family in ("C01", "C05", "C15"):
        rows = data[(data.family == family) & (data.encoder_credit == "temporal")
                    & (data.metric == "spatial_information")
                    & (data.worker_bonus == ("inapplicable" if family == "C01" else "temporal"))]
        if len(rows) != 99 or not rows.n.eq(5).all():
            raise ValueError(f"incomplete five-seed online course for {family}")
        rows = rows.sort_values("env_steps")
        x = rows.env_steps.to_numpy(float) / 1e6
        mean = rows["mean"].to_numpy(float)
        sem = rows["sem"].to_numpy(float)
        ax.plot(x, mean, lw=1.25, color=COLORS[family], label=family)
        ax.fill_between(x, mean - sem, mean + sem, color=COLORS[family], alpha=0.16,
                        linewidth=0)
        ax.text(102, mean[-1], family, color=COLORS[family], va="center", fontsize=10)
    ax.set(xlim=(0, 135), ylim=(0, 0.23), xlabel="Steps (M)", ylabel="DG score")
    ax.set_xticks((0, 50, 100))
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    export(fig, "01_online_spatial_course")


def behavior_expression(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(2 * BEHAVIOR_PANEL_SIZE_IN[0],
                                            BEHAVIOR_PANEL_SIZE_IN[1]))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.24, top=0.81, wspace=0.48)
    specs = (
        ("bits_per_activation", "a  Δ info/act."),
        ("split_half_map_correlation", "b  Δ reliability"),
    )
    for ax, (metric, label) in zip(axes, specs):
        rows = data[(data.layer == "decoder_1") & (data.metric == metric)]
        seed_differences(ax, rows, ("C01", "C05", "C15"), group_column="condition",
                         labels=("C01", "C05", "C15"),
                         colors=tuple(COLORS.values()))
        ax.set_title(label, loc="left", pad=8)
    export(fig, "02_behavior_expression")


def credit_presence(data: pd.DataFrame) -> None:
    rows = data[data.contrast == "none_stop_minus_constant"].copy()
    rows["group"] = rows.family + rows.worker_bonus.map({
        "flat": "", "temporal": "T", "zero": "Z"})
    rows["difference"] = -rows.active_unit_mean_si_bits_delta
    groups = ("C01", "C05T", "C05Z", "C15T", "C15Z")
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.25, top=0.79)
    seed_differences(ax, rows, groups, group_column="group",
                     labels=("01", "05T", "05Z", "15T", "15Z"),
                     colors=(COLORS["C01"], COLORS["C05"], COLORS["C05"],
                             COLORS["C15"], COLORS["C15"]))
    ax.set_title("Constant − no credit", loc="left", pad=8)
    ax.set_ylabel("Δ DG info (bits)")
    export(fig, "02_credit_presence")


def encoder_credit(fields: pd.DataFrame, peaks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(5.7, 2.1))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.25, top=0.79, wspace=0.33)
    specs = (
        (fields, "mean_spatial_information_bits", "a  Δ DG info (bits)"),
        (fields, "active_map_cosine_mean", "b  Δ map cosine"),
        (peaks[peaks.cutoff == 0.0025], "effective_area_fraction",
         "c  Δ peak area"),
    )
    colors = (COLORS["C01"], "#E69F00", COLORS["C05"], COLORS["C15"])
    for ax, (table, metric, label) in zip(axes, specs):
        rows = table[table.metric == metric]
        seed_differences(ax, rows, FACTOR_ORDER, group_column="contrast",
                         labels=("C01", "C05\nT", "C05\nC", "C15"), colors=colors)
        ax.set_title(label, loc="left", pad=8)
    export(fig, "03_encoder_credit_factorized")


def worker_bonus(data: pd.DataFrame) -> None:
    rows = data[data.contrast == "temporal_minus_constant_hit_bonus"]
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.05))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.81, wspace=0.35)
    specs = (("visited_cell_fraction", "a  Δ visited fraction"),
             ("spatial_information", "b  Δ online DG score"))
    for ax, (metric, label) in zip(axes, specs):
        selected = rows[rows.metric == metric]
        seed_differences(ax, selected, ("temporal", "constant"),
                         group_column="fixed_factor", labels=("T", "C"),
                         colors=("#0072B2", "#D55E00"))
        ax.set_title(label, loc="left", pad=8)
        ax.set_xlabel("Encoder credit")
    export(fig, "04_worker_timing_factorized")


def main() -> None:
    font_path = style()
    tables = {name: pd.read_csv(path) for name, path in SOURCES.items()}
    online_course(tables["online"])
    behavior_expression(tables["behavior"])
    credit_presence(tables["credit_presence"])
    encoder_credit(tables["encoder"], tables["peaks"])
    worker_bonus(tables["online_pairs"])
    metadata = {
        "schema": "cosyne27_compact_figures_v3", "font": "Arial",
        "verified_font_file": font_path, "font_pt": 10,
        "default_width_inches_per_panel": 1.9,
        "behavior_panel_size_inches": BEHAVIOR_PANEL_SIZE_IN,
        "unit_of_replication": "training seed",
        "matplotlib_version": matplotlib.__version__,
        "pandas_version": pd.__version__,
        "mean_marker": "black horizontal line; individual seeds are colored points",
        "online_uncertainty": "mean ± 1 SEM across five training seeds at each 1M-step grid point",
        "sources": {name: {"path": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in SOURCES.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
