"""Render exact-reset paired five-cue success rates at a matched checkpoint.

The per-cue denominators are displayed because cue frequency depends on the
environment's seeded starts, and a 40-trial evaluation need not balance cues.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-d50", required=True, type=Path)
    parser.add_argument("--paired-d51", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 16, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True,
                             sharey=True)
    arms = (("source_success", "Source DG", "#346c94"),
            ("random_success", "Random DG", "#c27739"))
    for ax, (name, path) in zip(axes, (("D50 · 75.02M", args.paired_d50),
                                        ("D51 · 75.04M", args.paired_d51))):
        trials = pd.read_csv(path)
        if trials["seed"].nunique() != 3 or trials["requested_seed"].nunique() != 40:
            raise ValueError(f"Expected three training seeds and 40 paired reset seeds: {path}")
        if len(trials) != 120:
            raise ValueError(f"Expected 120 paired endpoint trials: {path}")
        grouped = trials.groupby("cue")
        if set(grouped.groups) != {1, 2, 3, 4, 5}:
            raise ValueError(f"Missing a cue: {path}")
        cue = np.arange(1, 6)
        width = .37
        for offset, (column, label, color) in zip((-.5, .5), arms):
            values = grouped[column].mean().reindex(cue).to_numpy()
            ax.bar(cue + offset * width, values, width, label=label, color=color)
        counts = grouped.size().reindex(cue).to_numpy()
        for x, count in zip(cue, counts):
            ax.text(x, 1.02, f"n={count}", ha="center", va="bottom", fontsize=12)
        ax.set_title(name)
        ax.set_xticks(cue, [f"Cue {x}" for x in cue])
        ax.set_ylim(0, 1.14)
        ax.set_xlabel("Heldout instruction cue")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Physical reward success fraction")
    axes[1].legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, -.16), frameon=False)
    fig.suptitle("Five-cue heldout transfer · matched reset seeds · 3 training seeds",
                 fontsize=17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
