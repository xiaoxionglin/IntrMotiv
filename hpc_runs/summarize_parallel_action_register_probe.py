"""Render readout-probe evidence from saved predictions, without refitting."""

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

from parallel_action_register_probe import ROOT


def summarize(directory: Path) -> dict:
    rows = list(csv.DictReader((directory / "metrics.csv").open()))
    result = {}
    for model in dict.fromkeys(row["model"] for row in rows):
        result[model] = {}
        for split in dict.fromkeys(row["split"] for row in rows):
            selected = [row for row in rows if row["model"] == model and row["split"] == split]
            result[model][split] = {}
            for metric in ("position_rmse", "rounded_endpoint_accuracy", "pair_direction_accuracy"):
                values = [float(row[metric]) for row in selected]
                result[model][split][metric] = {"mean": float(np.mean(values)),
                                               "min": min(values), "max": max(values), "values": values}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    summaries = {"variable_age": summarize(directory), "matched_age": summarize(directory / "age_matched")}
    (directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    font_path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if not font_path.is_file() or font_path.suffix.lower() not in (".ttf", ".otf"):
        raise RuntimeError("A verified scalable font is required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    names = {"unordered_mlp64": "Unordered + MLP 64", "parallel_linear": "Parallel + linear",
             "parallel_mlp64": "Parallel + MLP 64", "parallel_mlp128": "Parallel + MLP 128"}
    colors = ("#666666", "#CC79A7", "#0072B2", "#D55E00")
    markers = ("s", "D", "o", "^")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for column, (protocol, title) in enumerate((("variable_age", "Landmark age follows word length"),
                                               ("matched_age", "Landmark age fixed at 9 steps"))):
        for row, metric in enumerate(("pair_direction_accuracy", "position_rmse")):
            ax = axes[row, column]
            for offset, ((name, label), color, marker) in enumerate(zip(names.items(), colors, markers)):
                positions = np.array((0., 1.)) + (offset - 1.5) * .14
                factor = 100 if row == 0 else 1
                for x, split in zip(positions, ("unseen_words", "longer_words")):
                    stats = summaries[protocol][name][split][metric]
                    values = np.array(stats["values"]) * factor
                    ax.scatter(x + np.linspace(-.025, .025, len(values)), values,
                               c=color, marker=marker, s=42, alpha=.6, zorder=3)
                    ax.plot([x - .045, x + .045], [stats["mean"] * factor] * 2, color=color, lw=2.5)
                if row == 0 and column == 0:
                    ax.scatter([], [], c=color, marker=marker, label=label)
            ax.set_xlim(-.4, 1.4)
            ax.set_xticks((0, 1), ("Unseen words\n2–6 actions", "Longer words\n7–9 actions"))
            ax.grid(axis="y", alpha=.2)
            if row == 0:
                ax.axhline(50, color="black", ls="--", lw=1)
                ax.set_ylim(45, 103)
                ax.set_title(title, fontsize=14, pad=12)
            else:
                ax.set_ylim(0, 3)
            ax.text(0, 1.04, "abcd"[row * 2 + column], transform=ax.transAxes, fontweight="bold")
    axes[0, 0].set_ylabel("Order-effect direction accuracy (%)")
    axes[1, 0].set_ylabel("Endpoint RMSE (grid steps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.52, 1.01), fontsize=13)
    fig.subplots_adjust(top=.83, bottom=.13, left=.12, right=.98, hspace=.25, wspace=.12)
    fig.text(.12, .025, "Dots: three optimizer seeds on one fixed split. Bars: means. Dashed line: tied predictions score 50%.", fontsize=12)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"generalization.{extension}", dpi=120 if extension == "png" else 180)
    svg = directory / "generalization.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    inputs = [directory / "metrics.csv", directory / "age_matched/metrics.csv"]
    manifest = {"source": str(Path(__file__).relative_to(ROOT)),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "font": str(font_path), "inputs": {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                "transformation": "Unweighted mean across three optimizer seeds; each seed also plotted. No confidence intervals."}
    (directory / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
