"""Summarize cached W&B spatial histories and render the COSYNE learning curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SI = "place_field/active_unit_mean_spatial_information"
COSINE = "place_field/active_only_map_cosine"
PALETTE = {"temporal": "#0072B2", "constant": "#D55E00"}


def collect_cached_histories(source: Path) -> pd.DataFrame:
    manifest = json.loads((source / "manifest.json").read_text())
    rows = []
    for record in manifest:
        history = pd.read_csv(source / record["history_file"])
        history.columns = [name.removeprefix("intrmotiv/online/") for name in history]
        history = history.rename(columns={"train/env_steps": "env_steps"})
        history["target_million_steps"] = np.rint(history.env_steps / 1e6).astype(int)
        if history.target_million_steps.duplicated().any():
            raise ValueError(f"Repeated million-step bin in {record['run_name']}")
        if not set(history.target_million_steps).issubset(set(range(1, 101))):
            raise ValueError(f"Unexpected environment steps in {record['run_name']}")
        for field in ("run_name", "study_id", "study_sha256", "family", "condition", "seed", "encoder_credit", "worker_bonus", "wandb_path"):
            history[field] = record[field]
        rows.append(history)
    return pd.concat(rows, ignore_index=True)


def summarize(histories: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["family", "encoder_credit", "worker_bonus", "target_million_steps"]
    metric_columns = [c for c in histories if c.startswith(("place_field/", "trajectory/"))]
    result = histories.groupby(id_columns, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        mean_env_steps=("env_steps", "mean"),
        **{f"{name}__mean": (name, "mean") for name in metric_columns},
        **{f"{name}__sd": (name, "std") for name in metric_columns},
    ).reset_index()
    for name in metric_columns:
        result[f"{name}__sem"] = result[f"{name}__sd"] / np.sqrt(result.n_seeds)
    return result


def plot_metric(summary: pd.DataFrame, output: Path, metric: str, ylabel: str) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 19,
        "axes.labelsize": 20, "axes.titlesize": 22,
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 16, "svg.fonttype": "none",
    })
    panels = (("C01", "inapplicable", "C01"),
              ("C05", "constant", "C05 · constant worker bonus"),
              ("C05", "temporal", "C05 · temporal worker bonus"),
              ("C15", "temporal", "C15"))
    fig, axes = plt.subplots(4, 1, figsize=(10, 14.5), sharex=True, sharey=True, layout="constrained")
    for ax, (family, worker_timing, title) in zip(axes, panels):
        frame = summary[(summary.family == family) & (summary.worker_bonus == worker_timing)]
        for (encoder, worker), curve in frame.groupby(["encoder_credit", "worker_bonus"], sort=False):
            curve = curve.sort_values("target_million_steps")
            x = curve["mean_env_steps"].to_numpy(dtype=float) / 1e6
            y = curve[f"{metric}__mean"].to_numpy(dtype=float)
            sem = curve[f"{metric}__sem"].to_numpy(dtype=float)
            label = f"{encoder.title()} encoder (n={int(curve.n_seeds.min())})"
            ax.plot(x, y, color=PALETTE[encoder], lw=2.5, label=label)
            ax.fill_between(x, y - sem, y + sem, color=PALETTE[encoder], alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.legend(loc="upper left", frameon=False)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Environment steps (millions)")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    histories = collect_cached_histories(args.source)
    summary = summarize(histories)
    histories.to_csv(args.output / "online_spatial_history_long.csv", index=False)
    summary.to_csv(args.output / "online_spatial_history_summary.csv", index=False)
    plot_metric(summary, args.output / "spatial_information_learning_curves", SI,
                "DG information\n(amplitude-weighted)")
    plot_metric(summary, args.output / "active_only_cosine_learning_curves", COSINE,
                "Active-only DG\nmap cosine")
    print("histories", len(histories), "summary rows", len(summary))


if __name__ == "__main__":
    main()
