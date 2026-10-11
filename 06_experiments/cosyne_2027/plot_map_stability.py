"""Plot canonical own-policy map correlation to the terminal checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PANELS = [
    ("C01 · seed 99", "c01_temporal_s3", "c01_constant_s3", [
        ("COS27_C01_CONSTANT", "Constant encoder", "#2C6593"),
        ("COS27_C01_TEMPORAL", "Temporal encoder", "#B34C34"),
    ]),
    ("C05 · constant worker · seed 99", "c05_pushpull_s3", None, [
        ("COS27_C05_CONSTANT_CONSTANT", "Constant encoder", "#2C6593"),
        ("COS27_C05_TEMPORAL_CONSTANT", "Temporal encoder", "#B34C34"),
    ]),
    ("C05 · temporal worker · seed 99", "c05_pushpull_s3", None, [
        ("COS27_C05_CONSTANT_TEMPORAL", "Constant encoder", "#2C6593"),
        ("COS27_C05_TEMPORAL_TEMPORAL", "Temporal encoder", "#B34C34"),
    ]),
    ("C15 · seed 17", "c15_encoder_s2", None, [
        ("COS27_C15_CONSTANT", "Constant encoder", "#2C6593"),
        ("COS27_C15_TEMPORAL", "Temporal encoder", "#B34C34"),
    ]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stability_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, root, constant_root, conditions) in zip(axes.flat, PANELS):
        for condition, label, color in conditions:
            cohort = constant_root if constant_root and "CONSTANT" in condition else root
            path = args.stability_dir / cohort / condition / "rate_map_stability_summary.csv"
            rows = pd.read_csv(path)
            ax.plot(rows.checkpoint_frames / 1e6, rows.mean_correlation, "o-", linewidth=2.5,
                    markersize=9, color=color, label=label)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(-.1, 1.04)
        ax.set_xticks([5, 25, 50, 75, 100])
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Map correlation to 100M", fontsize=18)
    for ax in axes[1]:
        ax.set_xlabel("Training frames (millions)", fontsize=18)
    fig.suptitle("Native-policy DG map similarity · occupied-bin overlap", fontsize=22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
