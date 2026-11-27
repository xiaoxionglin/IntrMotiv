"""Render the declared replay-TD intervention results without refitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def render(source: Path, output: Path):
    results = json.loads(source.read_text())
    rows = [row for row in results["runs"] if row["method"] == "replay_td"]
    environments = ("fork_cycle", "sparse_context")
    seeds = (8, 99, 123)
    routes = ("first_hit", "shuffled_goal", "memoryless", "continuing_main")
    labels = ("Correct\ngoal", "Shuffled\ngoal", "No\nhistory", "Blind\ncontrol")
    titles = ("Branching graph: 12 pairs / seed", "Memory task: 4 pairs / seed")
    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if font.suffix.lower() not in (".ttf", ".otf") or not font.is_file():
        raise RuntimeError("A verified scalable font is required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.3), sharey=True)
    plotted = []
    for ax, environment, title in zip(axes, environments, titles):
        for seed_index, seed in enumerate(seeds):
            selected = [row for row in rows if row["environment"] == environment and row["seed"] == seed]
            if len(selected) != 1:
                raise ValueError(f"Expected exactly one result for {environment}/{seed}")
            row = selected[0]
            values = np.asarray([row["per_route_success"][route] for route in routes])
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Invalid success probabilities")
            ax.scatter(np.arange(len(routes)) + (seed_index - 1) * .15, values,
                       color=("#0072B2", "#D55E00", "#009E73")[seed_index],
                       marker=("o", "s", "^")[seed_index], s=75, edgecolor="white", linewidth=.5,
                       label=f"Seed {seed}", zorder=3)
            plotted.extend({"environment": environment, "seed": seed, "route": route,
                            "success_fraction": float(value), "pairs": row["n_nontrivial_pairs"]}
                           for route, value in zip(routes, values))
        ax.set_title(title, pad=15)
        ax.set_xticks(range(len(routes)), labels)
        ax.set_xlim(-.5, len(routes) - .5)
        ax.set_ylim(-.05, 1.08)
        ax.set_yticks([0, .5, 1], ["0", "50", "100"])
        ax.grid(axis="y", color=".87", linewidth=.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes:
        ax.set_ylabel("Successful pairs (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.13, right=.97, top=.93, bottom=.16, hspace=.70)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"replay_control.{extension}", dpi=140, facecolor="white")
    plt.close(fig)
    mapping = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "font": str(font), "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
               "matplotlib_version": matplotlib.__version__, "rows": plotted,
               "transformation": "success fraction shown as percent; horizontal seed offsets only",
               "uncertainty": "Individual fit/data seeds; no error bars or independence claim for repeated graph tasks",
               "scope": "Factual replay-TD, synthetic frozen DG, two eligible goal identities per environment. Goal-blind control also removes goal input; not routing-only."}
    (output / "replay_control.mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
