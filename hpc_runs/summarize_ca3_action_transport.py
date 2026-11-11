"""Summarize learned transport and saved parallel predictions without refits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

from parallel_action_register_probe import ROOT, matched_pairs, metrics
from summarize_parallel_action_register_probe import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary = summarize(directory)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    spec = json.loads((directory / "spec.json").read_text())
    parent = ROOT / spec["parent_directory"]
    rows = json.loads((parent / "dataset.json").read_text())
    for row in rows:
        row["counts"] = tuple(row["counts"])
    archive = np.load(directory / "predictions.npz")
    data = {"rows": rows, "targets": torch.tensor(archive["targets"])}
    records = []
    for length in range(2, 10):
        split = "unseen_words" if length <= 6 else "longer_words"
        indices = np.array([i for i, row in enumerate(rows) if row["split"] == split and len(row["word"]) == length])
        data["indices"] = {split: indices}
        pairs = matched_pairs(data, split)
        for key in archive.files:
            if key == "targets":
                continue
            name, seed = key.split("__seed")
            records.append({"model": name, "seed": int(seed), "length": length,
                            **metrics(archive[key], data, indices, pairs)})
    with (directory / "metrics_by_length.csv").open("w") as stream:
        fields = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    font_path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if not font_path.is_file() or font_path.suffix.lower() not in (".ttf", ".otf"):
        raise RuntimeError("A scalable TTF or OTF font is required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    names = {"parallel_mlp64": "Parallel + MLP 64", "shared_r2_mlp64": "Shared rank 2 + MLP 64",
             "shared_r4_mlp64": "Shared rank 4 + MLP 64", "per_action_r2_mlp64": "Per-action rank 2 + MLP 64",
             "per_action_r2_linear": "Per-action rank 2 + linear"}
    colors = ("#666666", "#0072B2", "#CC79A7", "#D55E00", "#009E73")
    markers = ("s", "o", "D", "^", "v")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, metric, label in zip(axes, ("position_rmse", "rounded_endpoint_accuracy"),
                                 ("Endpoint RMSE (grid steps)", "Correct endpoint (%)")):
        factor = 100 if metric.endswith("accuracy") else 1
        for (name, legend), color, marker in zip(names.items(), colors, markers):
            means = []
            for length in range(2, 10):
                selected = [row for row in records if row["model"] == name and row["length"] == length]
                values = [row[metric] * factor for row in selected]
                means.append(np.mean(values))
                ax.scatter(length + np.linspace(-.04, .04, len(values)), values, c=color, s=18, alpha=.35)
            ax.plot(range(2, 10), means, color=color, marker=marker, label=legend, lw=1.8, ms=6)
        ax.axvline(6.5, color="black", ls="--", lw=1)
        ax.set_xticks(range(2, 10))
        ax.set_xlabel("Non-no-op actions since landmark")
        ax.set_ylabel(label)
        ax.set_xlim(1.8, 9.2)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.text(.02, 1.02, "Held-out words", transform=ax.transAxes, fontsize=12)
        ax.text(.70, 1.02, "New lengths", transform=ax.transAxes, fontsize=12)
    axes[1].set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=12,
               bbox_to_anchor=(.52, 1.01))
    fig.subplots_adjust(top=.74, bottom=.19, left=.08, right=.99, wspace=.27)
    fig.text(.08, .045, "Dots: three fit seeds on one fixed split. Lines: means. Landmark age = 9 throughout; train on 2–6 actions.", fontsize=12)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(directory / f"length_generalization.{extension}", dpi=120 if extension == "png" else 180)
    plt.close(fig)
    svg = directory / "length_generalization.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    manifest = {"source": str(Path(__file__).relative_to(ROOT)),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "font": str(font_path), "plot": "Per-length endpoint metrics; individual optimizer seeds and unweighted seed means. No CIs.",
                "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (directory / "metrics.csv", directory / "predictions.npz", parent / "dataset.json")}}
    (directory / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {split: {metric: round(summary[name][split][metric]["mean"], 5)
                                   for metric in ("position_rmse", "rounded_endpoint_accuracy", "pair_direction_accuracy")}
                            for split in ("unseen_words", "longer_words")} for name in summary}, indent=2))


if __name__ == "__main__":
    main()
