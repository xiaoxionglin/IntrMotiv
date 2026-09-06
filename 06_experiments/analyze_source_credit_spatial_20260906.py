#!/usr/bin/env python3
"""Analyze completed source-credit spatial/grounded telemetry at 75M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "active_unit_mean_spatial_information",
    "active_only_map_cosine",
    "mono_field_unit_fraction",
    "graph_reliable_edge_count",
    "graph_largest_strong_component_size",
    "graph_reachable_pair_fraction",
    "graph_prospective_success_fraction",
    "graph_grounded_controllability",
    "path_efficiency",
]
LABELS = {
    "monitor": "MON",
    "dir_silent": "DIRS",
    "dir_open": "DIRO",
    "pred_silent": "PREDS",
    "pred_open": "PREDO",
}
STUDY_SHA256 = "aa34bb2ef868df37dbafc38e7c4b8dc5c9cc684e5f8dba5e05a586d068ad8a0d"


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    specs = [
        ("encoder_credit", "arrival", "source", ["retirement", "seed"]),
        ("retirement", "dir_silent", "dir_open", ["encoder_credit", "seed"]),
        ("retirement", "pred_silent", "pred_open", ["encoder_credit", "seed"]),
    ]
    for factor, a, b, index in specs:
        left = frame.loc[frame[factor] == a].set_index(index)
        right = frame.loc[frame[factor] == b].set_index(index)
        common = left.index.intersection(right.index)
        for metric in METRICS:
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
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panels = [
        ("active_unit_mean_spatial_information", "Spatial information"),
        ("active_only_map_cosine", "Across-unit map cosine ↓"),
        ("mono_field_unit_fraction", "Mono-field unit fraction"),
        ("graph_grounded_controllability", "Grounded controllability"),
        ("graph_prospective_success_fraction", "Prospective edge success"),
        ("graph_reachable_pair_fraction", "Reliable reachable pairs"),
    ]
    order = ["monitor", "dir_silent", "dir_open", "pred_silent", "pred_open"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=False)
    for ax, (metric, title) in zip(axes.flat, panels):
        for credit, color, marker in [
            ("arrival", "#0072B2", "o"),
            ("source", "#D55E00", "s"),
        ]:
            means, sds = [], []
            for retirement in order:
                values = frame.loc[
                    (frame.encoder_credit == credit) & (frame.retirement == retirement),
                    metric,
                ]
                means.append(values.mean())
                sds.append(values.std(ddof=1))
                ax.scatter(
                    np.full(len(values), x[order.index(retirement)]),
                    values,
                    color=color,
                    alpha=0.28,
                    s=25,
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
                label=credit.upper(),
            )
        ax.set_title(title)
        ax.set_xticks(x, [LABELS[item] for item in order], rotation=20)
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Source-credit/retirement telemetry at 75M (mean ± SD, n=3)", y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/source_credit_retirement_spatial_diag_20260905/per_snapshot.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_experiments/results/recent_batches_audit_20260906"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    terminal = frame.loc[frame.target_env_steps == 75_000_000].copy()
    if len(terminal) != 30:
        raise RuntimeError(f"Expected 30 terminal rows, got {len(terminal)}")
    terminal.to_csv(args.output_dir / "source_credit_terminal_spatial_per_run.csv", index=False)
    grouped = terminal.groupby(["encoder_credit", "retirement"])[METRICS]
    pd.concat(
        [grouped.mean().add_suffix("_mean"), grouped.std(ddof=1).add_suffix("_sd")],
        axis=1,
    ).reset_index().to_csv(
        args.output_dir / "source_credit_terminal_spatial_summary.csv", index=False
    )
    contrasts(terminal).to_csv(
        args.output_dir / "source_credit_spatial_paired_effects.csv", index=False
    )
    plot(terminal, args.output_dir / "source_credit_spatial")
    metadata = {
        "study": "source_credit_retirement_20260904",
        "study_sha256": STUDY_SHA256,
        "sample": "30 runs; 10 matched cells; seeds 8, 99, 123",
        "snapshot": "manifest-driven 100k-decision evaluation at the terminal checkpoint targeted at 75M",
        "uncertainty": "figure shows across-seed SD; contrasts retain seed and the other factor",
        "source_table": str(args.input),
    }
    (args.output_dir / "source_credit_spatial_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
