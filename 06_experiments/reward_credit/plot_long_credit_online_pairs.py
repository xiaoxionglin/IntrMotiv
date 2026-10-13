"""Render absolute five-seed C05 worker-timing comparisons for a report page."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


METRICS = (
    ("correct_hit_observations", "Correct-hit observations\n(repeated learner batches)"),
    ("hit_weighted_elapsed_decisions", "Elapsed decisions per correct hit\n(count weighted)"),
    ("coverage_auc", "Online coverage AUC"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hit_weighted_csv", type=Path)
    parser.add_argument("online_per_run_csv", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()

    font_path = Path(font_manager.findfont("Arial", fallback_to_default=False))
    if not font_path.is_file() or font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError("A scalable Arial TTF/OTF is required")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })

    hits = pd.read_csv(args.hit_weighted_csv)
    online = pd.read_csv(args.online_per_run_csv)[["run_name", "coverage_auc"]]
    data = hits.merge(online, on="run_name", validate="one_to_one")
    if data.groupby("seed")["worker_bonus"].apply(set).ne({"TEMPORAL", "CONSTANT"}).any():
        raise ValueError("expected one constant and one temporal row per seed")
    seeds = sorted(data.seed.unique())
    if len(seeds) != 5:
        raise ValueError("expected five paired training seeds")

    figure, axes = plt.subplots(3, 1, figsize=(6.2, 8.4), constrained_layout=True)
    for index, (ax, (metric, ylabel)) in enumerate(zip(axes, METRICS)):
        endpoints = []
        for seed in seeds:
            pair = data.loc[data.seed == seed].set_index("worker_bonus")
            constant = float(pair.loc["CONSTANT", metric])
            temporal = float(pair.loc["TEMPORAL", metric])
            ax.plot([0, 1], [constant, temporal], color="#A5B0B7", linewidth=1.3, zorder=1)
            ax.scatter([0], [constant], marker="s", s=74, facecolors="white",
                       edgecolors="#63747E", linewidths=1.5, zorder=2)
            ax.scatter([1], [temporal], marker="o", s=74, facecolors="#30363B",
                       edgecolors="#30363B", linewidths=1, zorder=2)
            endpoints.append((temporal, seed))
        upper = float(data[metric].max()) * 1.14
        minimum_label_gap = upper * 0.055
        labeled = []
        for actual, seed in sorted(endpoints):
            label_y = max(actual, labeled[-1][0] + minimum_label_gap) if labeled else actual
            labeled.append((label_y, actual, seed))
        if labeled[-1][0] > upper * 0.95:
            shift = labeled[-1][0] - upper * 0.95
            labeled = [(label_y - shift, actual, seed) for label_y, actual, seed in labeled]
        for label_y, actual, seed in labeled:
            ax.plot([1.035, 1.10], [actual, label_y], color="#84929B", linewidth=0.8)
            ax.text(1.12, label_y, str(seed), ha="left", va="center", fontsize=12,
                    color="#30363B")
        ax.set_xlim(-0.4, 1.46)
        ax.set_ylim(0, upper)
        ax.set_xticks([0, 1], ["Constant", "Temporal"])
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.99, "abc"[index], transform=ax.transAxes, va="top",
                ha="left", fontweight="bold", fontsize=16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=4, width=0.8)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_stem.with_suffix(".svg")
    figure.savefig(svg_path, metadata={"Creator": "IntrMotiv"})
    # Matplotlib leaves spaces at SVG path line ends; remove them for clean diffs.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300)
    plt.close(figure)
    args.output_stem.with_name(args.output_stem.name + "_manifest.json").write_text(json.dumps({
        "hit_weighted_csv": str(args.hit_weighted_csv),
        "hit_weighted_csv_sha256": sha256(args.hit_weighted_csv),
        "online_per_run_csv": str(args.online_per_run_csv),
        "online_per_run_csv_sha256": sha256(args.online_per_run_csv),
        "font_path": str(font_path),
        "font_size_pt": 15,
        "figure_inches": [6.2, 8.4],
        "dpi_png": 300,
        "training_seeds": [int(seed) for seed in seeds],
        "pairing": "constant and temporal worker bonus within training seed",
        "panels": [{"column": metric, "label": ylabel} for metric, ylabel in METRICS],
        "sample_unit": "training seed",
        "meaning": "Online training observations, not independent physical arrivals or executed-command trials.",
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
