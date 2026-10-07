"""Analyze paired odor/goal-selector coverage curves from canonical histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from hpc_runs.intrmotiv_study import load_study


BIN_FRAMES = 5_000_000
HORIZON_FRAMES = 75_000_000
WINDOWS = {"full": (0, 75), "early": (0, 10), "terminal": (70, 75)}
SELECTORS = ("all16", "random4", "hebb4", "random8", "hebb8")
COLORS = {"all16": "#242424", "random4": "#2675ad", "hebb4": "#2675ad",
          "random8": "#ca6b17", "hebb8": "#ca6b17"}
STYLES = {"all16": "-", "random4": "-", "hebb4": "--", "random8": "-", "hebb8": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def coverage_bins(study, histories: Path) -> pd.DataFrame:
    tag = study.analysis["window_metrics"]["coverage_auc"]
    rows = []
    for run in study.expand_runs():
        history = pd.read_csv(histories / f"{run.name}.csv", usecols=["tag", "step", "value"])
        values = history.loc[history.tag == tag, ["step", "value"]].copy()
        values = values[np.isfinite(values.value) & values.step.between(0, HORIZON_FRAMES)]
        # Keep repeated event steps, as the standardized exporter does.
        values["bin"] = np.minimum((values.step // BIN_FRAMES).astype(int), 14)
        grouped = values.groupby("bin").value.agg(["mean", "size"])
        if set(grouped.index) != set(range(15)):
            raise ValueError(f"Coverage history lacks a 5M bin: {run.name}")
        for bin_index, row in grouped.iterrows():
            rows.append({"run_name": run.name, "condition": run.condition,
                         "seed": run.seed, "odor": run.factors["odor"],
                         "goal_set": run.factors["goal_set"],
                         "bin": int(bin_index), "bin_start_m": int(bin_index) * 5,
                         "bin_end_m": (int(bin_index) + 1) * 5,
                         "coverage_auc": float(row["mean"]), "samples": int(row["size"])})
    result = pd.DataFrame(rows)
    if len(result) != 600 or result.groupby(["condition", "bin"]).size().ne(4).any():
        raise ValueError("Expected 40 runs, four seeds per condition, and 15 bins")
    return result


def paired_contrasts(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookups = {(row.odor, row.goal_set, row.seed, row.window): row.coverage_auc
               for row in windows.itertuples(index=False)}
    definitions = []
    for odor in ("off", "on"):
        for k in (4, 8):
            definitions.append((f"HEBB{k}−RANDOM{k} | {odor}", (odor, f"hebb{k}"),
                                (odor, f"random{k}")))
            definitions.append((f"RANDOM{k}−ALL16 | {odor}", (odor, f"random{k}"),
                                (odor, "all16")))
        for selector in ("random", "hebb"):
            definitions.append((f"{selector.upper()}4−{selector.upper()}8 | {odor}",
                                (odor, f"{selector}4"), (odor, f"{selector}8")))
    for goal in SELECTORS:
        definitions.append((f"ON−OFF | {goal.upper()}", ("on", goal), ("off", goal)))
    rows = []
    for name, left, right in definitions:
        for window in WINDOWS:
            for seed in (8, 99, 123, 2026):
                rows.append({"contrast": name, "window": window, "seed": seed,
                             "difference": lookups[(*left, seed, window)]
                             - lookups[(*right, seed, window)]})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["contrast", "window"], sort=False).difference.agg(
        mean="mean", sd="std", n="size").reset_index()
    summary["ci95_low"] = summary["mean"] - 3.182446 * summary.sd / np.sqrt(summary.n)
    summary["ci95_high"] = summary["mean"] + 3.182446 * summary.sd / np.sqrt(summary.n)
    return detail, summary


def render_curve(curves: pd.DataFrame, output: Path) -> None:
    font_path = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"),
                                           fallback_to_default=False))
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"A scalable font is required, found {font_path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17})
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for axis, odor in zip(axes, ("off", "on")):
        for selector in SELECTORS:
            part = curves[(curves.odor == odor) & (curves.goal_set == selector)].sort_values("bin")
            x = part.bin.to_numpy() * 5 + 2.5
            y = part["mean"].to_numpy()
            error = part["sd"].to_numpy() / 2  # SEM across four paired seeds
            axis.plot(x, y, color=COLORS[selector], linestyle=STYLES[selector],
                      linewidth=2.6, label=selector.upper())
            axis.fill_between(x, y - error, y + error, color=COLORS[selector], alpha=0.10)
        axis.set_title(f"Odor {odor.upper()}", fontsize=20)
        axis.set_xlabel("Training frames (millions)", fontsize=18)
        axis.set_xlim(0, 75)
        axis.set_xticks([0, 10, 25, 50, 75])
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=15)
    axes[0].set_ylabel("External coverage AUC", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=15, ncol=5,
                  loc="lower center", bbox_to_anchor=(0.5, 0.015))
    figure.suptitle("Coverage during training: mean ± SEM across four seeds", fontsize=20)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.88, wspace=0.09)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    study = load_study(args.study)
    if study.provenance()["study_sha256"] != "c325acb58098b763c5a8e7c0d97a07bca676eaa8dde4e262b6c9dffc4a94d7de":
        raise ValueError("Study fingerprint differs from the launched 40-run study")
    bins = coverage_bins(study, args.histories)
    curves = bins.groupby(["odor", "goal_set", "condition", "bin"], sort=False).coverage_auc.agg(
        mean="mean", sd="std", n="size").reset_index()
    windows = []
    for name, (low, high) in WINDOWS.items():
        subset = bins[(bins.bin_start_m >= low) & (bins.bin_end_m <= high)]
        frame = subset.groupby(["run_name", "condition", "seed", "odor", "goal_set"],
                               sort=False).coverage_auc.mean().reset_index()
        frame["window"] = name
        windows.append(frame)
    windows = pd.concat(windows, ignore_index=True)
    detail, summary = paired_contrasts(windows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("coverage_bins.csv", bins), ("coverage_curves.csv", curves),
                        ("coverage_windows.csv", windows), ("coverage_paired.csv", detail),
                        ("coverage_contrasts.csv", summary)):
        table.to_csv(args.out_dir / name, index=False)
    render_curve(curves, args.out_dir / "coverage_learning_curves.png")
    print(f"Analyzed {bins.run_name.nunique()} runs and {len(summary)} paired contrast summaries")


if __name__ == "__main__":
    main()
