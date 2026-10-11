"""Combine canonical frozen-checkpoint field summaries for the COSYNE matrix.

The manifest-driven evaluator supplies the metrics. This adapter joins its
lightweight CSVs by explicit condition and seed, then renders paired contrasts.
Each checkpoint uses its own 10k-decision policy rollout; these are not common
observations across conditions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_wandb_online import export, style, write_csv


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_experiments/results/cosyne_2027_checkpoint_fields"
STUDIES = (
    "c01_temporal_s3", "c01_constant_s3", "c01_encoder_s2",
    "c05_pushpull_s3", "c05_pushpull_s2", "c15_available", "c15_encoder_s2",
)
METRICS = {
    "mean_spatial_information_bits": "Mean DG spatial information (bits)",
    "mean_active_fraction": "Mean DG active fraction",
    "active_map_cosine_mean": "Active-only map cosine",
    "active_unique_peak_bins": "Distinct active peak bins",
    "active_pairwise_peak_distance_bins": "Mean active peak distance (bins)",
    "mono_field_fraction": "Single-field fraction of eligible units",
    "prethreshold_map_cosine_mean": "Pre-threshold map cosine",
    "field_eligible_units": "Field-eligible DG units / 16",
    "visited_cell_fraction": "Visited-bin fraction",
}
CONTRASTS = (
    ("C01", "inapplicable", "C01"),
    ("C05", "temporal", "C05 · temporal hit"),
    ("C05", "constant", "C05 · constant hit"),
    ("C15", "temporal", "C15"),
)
PALETTE = {
    "temporal": "#0072B2", "constant": "#D55E00",
}


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    parts = []
    sources = []
    for study in STUDIES:
        path = OUT / study / "summary/derived_place_field_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing completed study summary: {path}")
        frame = pd.read_csv(path)
        frame["source_study"] = study
        parts.append(frame)
        sources.append({"study": study, "path": str(path.relative_to(ROOT)),
                        "sha256": source_digest(path), "rows": len(frame)})
    all_rows = pd.concat(parts, ignore_index=True)
    if all_rows.duplicated(["condition", "seed", "target_frames"]).any():
        raise ValueError("duplicate frozen checkpoints across study summaries")
    catalog = pd.read_csv(ROOT / "06_experiments/cosyne_2027/training_runs.csv")
    expected = catalog[(catalog.cohort == "cosyne")
                       & (catalog.attempt_status == "verified_100m")]
    factors = expected[["condition", "seed", "family", "encoder_credit",
                        "worker_bonus"]].drop_duplicates()
    if factors.duplicated(["condition", "seed"]).any():
        raise ValueError("run catalog has duplicate factor metadata")
    all_rows = all_rows.drop(columns=["family"]).merge(
        factors, on=["condition", "seed"], how="left", validate="many_to_one")
    if all_rows[["family", "encoder_credit", "worker_bonus"]].isna().any().any():
        raise ValueError("frozen checkpoint lacks StudySpec-derived factor metadata")
    if all_rows.groupby(["condition", "seed"]).target_frames.max().lt(100_000_000).any():
        raise ValueError("terminal checkpoint missing")
    expected_keys = set(zip(expected.condition, expected.seed))
    observed = all_rows[all_rows.target_frames == 100_000_000]
    observed_keys = set(zip(observed.condition, observed.seed))
    if expected_keys != observed_keys or len(observed) != len(expected_keys):
        raise ValueError("terminal checkpoint rows do not match the indexed verified runs")
    return all_rows, sources


def paired_terminal(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    terminal = rows[rows.target_frames == 100_000_000]
    pairs = []
    for family, worker, label in CONTRASTS:
        subset = terminal[(terminal.family == family) & (terminal.worker_bonus == worker)]
        for seed, frame in subset.groupby("seed"):
            if set(frame.encoder_credit) != {"temporal", "constant"}:
                continue
            temporal = frame.set_index("encoder_credit").loc["temporal"]
            constant = frame.set_index("encoder_credit").loc["constant"]
            for metric in METRICS:
                pairs.append({"family": family, "worker_bonus": worker,
                              "contrast": label, "seed": int(seed), "metric": metric,
                              "temporal": temporal[metric], "constant": constant[metric],
                              "difference": temporal[metric] - constant[metric]})
    by_seed = pd.DataFrame(pairs)
    summary = (by_seed.groupby(["family", "worker_bonus", "contrast", "metric"],
                               as_index=False)
               .agg(n=("seed", "size"), mean_difference=("difference", "mean"),
                    sd_difference=("difference", "std"),
                    positive_pairs=("difference", lambda s: int((s > 0).sum())),
                    negative_pairs=("difference", lambda s: int((s < 0).sum()))))
    return by_seed, summary


def paired_worker(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair temporal versus constant hit timing within each C05 encoder arm."""
    terminal = rows[(rows.target_frames == 100_000_000) & (rows.family == "C05")]
    records = []
    for encoder, frame in terminal.groupby("encoder_credit"):
        for seed, seed_rows in frame.groupby("seed"):
            if set(seed_rows.worker_bonus) != {"temporal", "constant"}:
                continue
            temporal = seed_rows.set_index("worker_bonus").loc["temporal"]
            constant = seed_rows.set_index("worker_bonus").loc["constant"]
            for metric in METRICS:
                records.append({"encoder_credit": encoder, "seed": int(seed),
                                "metric": metric, "temporal": temporal[metric],
                                "constant": constant[metric],
                                "difference": temporal[metric] - constant[metric]})
    by_seed = pd.DataFrame(records)
    summary = (by_seed.groupby(["encoder_credit", "metric"], as_index=False)
               .agg(n=("seed", "size"), mean_difference=("difference", "mean"),
                    sd_difference=("difference", "std"),
                    positive_pairs=("difference", lambda s: int((s > 0).sum())),
                    negative_pairs=("difference", lambda s: int((s < 0).sum()))))
    return by_seed, summary


def plot_pairs(pairs: pd.DataFrame) -> None:
    style()
    for metric, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        subset = pairs[pairs.metric == metric]
        ax.axhline(0, color="#777777", linewidth=1, zorder=0)
        for x, (_, _, label) in enumerate(CONTRASTS):
            values = subset[subset.contrast == label].sort_values("seed")
            if values.empty:
                continue
            jitter = np.linspace(-0.13, 0.13, len(values))
            color = "#009E73" if label.startswith("C15") else (
                "#D55E00" if "constant hit" in label else "#0072B2")
            ax.scatter(x + jitter, values.difference, color=color, s=37,
                       edgecolor="white", linewidth=0.6, zorder=2)
            ax.plot([x - 0.21, x + 0.21], [values.difference.mean()] * 2,
                    color="black", linewidth=2, zorder=3)
            ax.text(x, values.difference.max(), f"n={len(values)}", ha="center",
                    va="bottom", fontsize=12)
        ax.set(title=title, ylabel="Temporal − constant",
               xticks=range(len(CONTRASTS)),
               xticklabels=[item[2].replace(" · ", "\n") for item in CONTRASTS])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        fig.suptitle("Paired 100M frozen-checkpoint fields", fontsize=14)
        export(fig, OUT / "figures" / f"paired_{metric}")


def plot_worker_pairs(pairs: pd.DataFrame) -> None:
    style()
    for metric, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
        subset = pairs[pairs.metric == metric]
        ax.axhline(0, color="#777777", linewidth=1, zorder=0)
        for x, encoder in enumerate(("temporal", "constant")):
            values = subset[subset.encoder_credit == encoder].sort_values("seed")
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(x + jitter, values.difference, color=PALETTE[encoder],
                       s=37, edgecolor="white", linewidth=0.6, zorder=2)
            ax.plot([x - 0.19, x + 0.19], [values.difference.mean()] * 2,
                    color="black", linewidth=2, zorder=3)
        ax.set(title=title, ylabel="Temporal − constant hit bonus",
               xticks=(0, 1), xticklabels=("Temporal encoder", "Constant encoder"),
               xlim=(-0.6, 1.6))
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        fig.suptitle("C05 worker timing: five paired seeds", fontsize=14)
        export(fig, OUT / "figures" / f"worker_{metric}")


def plot_factorized(rows: pd.DataFrame) -> None:
    style()
    terminal = rows[rows.target_frames == 100_000_000]
    for metric, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        for x, (family, worker, label) in enumerate(CONTRASTS):
            subset = terminal[(terminal.family == family) & (terminal.worker_bonus == worker)]
            for credit, offset in (("temporal", -0.16), ("constant", 0.16)):
                values = subset[subset.encoder_credit == credit].sort_values("seed")
                if values.empty:
                    continue
                jitter = np.linspace(-0.07, 0.07, len(values))
                ax.scatter(x + offset + jitter, values[metric],
                           color=PALETTE[credit], s=32, alpha=0.8,
                           edgecolor="white", linewidth=0.5, zorder=2)
                ax.plot([x + offset - 0.09, x + offset + 0.09],
                        [values[metric].mean()] * 2, color=PALETTE[credit],
                        linewidth=2.2, zorder=3)
        ax.set(title=title, xticks=range(len(CONTRASTS)),
               xticklabels=[item[2].replace(" · ", "\n") for item in CONTRASTS])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        handles = [plt.Line2D([0], [0], color=PALETTE[arm], marker="o",
                              linestyle="none", label=f"{arm.title()} encoder")
                   for arm in ("temporal", "constant")]
        ax.legend(handles=handles, loc="best", frameon=False)
        fig.suptitle("All available 100M arms", fontsize=14)
        export(fig, OUT / "figures" / f"factorized_{metric}")


def plot_checkpoint_course(rows: pd.DataFrame) -> None:
    style()
    colors = {
        ("temporal", "temporal"): "#0072B2",
        ("temporal", "constant"): "#56B4E9",
        ("constant", "temporal"): "#D55E00",
        ("constant", "constant"): "#E69F00",
        ("temporal", "inapplicable"): "#0072B2",
        ("constant", "inapplicable"): "#D55E00",
    }
    for seed in (17, 99):
        selected = rows[(rows.seed == seed) & (rows.target_frames.isin(
            [5_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000]))]
        for family in ("C01", "C05", "C15"):
            fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.2), layout="constrained", sharex=True)
            for row_idx, metric in enumerate(("mean_spatial_information_bits",
                                              "active_map_cosine_mean")):
                ax = axes[row_idx]
                subset = selected[selected.family == family]
                for (credit, worker), arm in subset.groupby(["encoder_credit", "worker_bonus"]):
                    arm = arm.sort_values("target_frames")
                    if len(arm) != 5:
                        continue
                    label = f"{credit.title()} encoder"
                    if family == "C05":
                        label += f", {worker} hit"
                    ax.plot(arm.target_frames / 1e6, arm[metric], marker="o",
                            color=colors[(credit, worker)], linewidth=2, label=label)
                ax.set(title=METRICS[metric],
                       xlabel="Frames (millions)" if row_idx == 1 else None,
                       ylabel=METRICS[metric],
                       xlim=(0, 105))
                ax.grid(color="#dddddd", linewidth=0.6)
                if row_idx == 0:
                    ax.legend(fontsize=11, frameon=False, loc="best")
            fig.suptitle(f"{family} seed {seed} checkpoint course", fontsize=14)
            export(fig, OUT / "figures" / f"{family.lower()}_seed{seed}_checkpoint_course")


def main() -> None:
    rows, sources = read_data()
    pairs, summary = paired_terminal(rows)
    worker_pairs, worker_summary = paired_worker(rows)
    write_csv(rows, OUT / "all_checkpoint_fields.csv")
    write_csv(pairs, OUT / "paired_terminal_fields.csv")
    write_csv(summary, OUT / "paired_terminal_field_summary.csv")
    write_csv(worker_pairs, OUT / "paired_worker_fields.csv")
    write_csv(worker_summary, OUT / "paired_worker_field_summary.csv")
    plot_pairs(pairs)
    plot_worker_pairs(worker_pairs)
    plot_factorized(rows)
    plot_checkpoint_course(rows)
    metadata = {"schema": "cosyne27_checkpoint_fields_v1",
                "source": "manifest-driven 10k-decision policy-specific checkpoint evaluation",
                "sources": sources, "checkpoint_rows": len(rows),
                "paired_seed_metric_rows": len(pairs),
                "paired_worker_seed_metric_rows": len(worker_pairs),
                "condition_counts": rows[rows.target_frames == 100_000_000]
                    .groupby("condition").size().to_dict()}
    (OUT / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
