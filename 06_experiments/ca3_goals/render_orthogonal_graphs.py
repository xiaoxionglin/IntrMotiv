"""Render the four-goal 75M graph comparison from canonical edge summaries.

The input is the seed-99 subset of canonical ``collect-spatial --include-details``
edge tables. A cell means prospectively attempted and/or stored as reliable;
it does not assert successful navigation, especially for episode-long runs
whose episode-end unresolved commands are absent from the outcome denominator.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parent / "results" / "orthogonal_film_dg_20261008"
PANELS = (
    ("Finite prescribed", "Finite learned-4"),
    ("Episode prescribed", "Episode learned-4"),
)


def main() -> None:
    edges = pd.read_csv(RESULTS / "graph_seed99_75m_adjacency.csv")
    assert len(edges) == 48
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 21,
        "axes.titlesize": 22,
        "axes.labelsize": 21,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "pdf.fonttype": 42,
    })
    colors = ("#e6e6e6", "#E69F00", "#0072B2")
    fig, axes = plt.subplots(2, 2, figsize=(10, 11))
    for ax, panel in zip(axes.flat, sum(PANELS, ())):
        selected = edges.loc[edges.panel.eq(panel)]
        assert len(selected) == 12
        matrix = np.full((4, 4), np.nan)
        for row in selected.itertuples(index=False):
            matrix[int(row.source_unit), int(row.target_unit)] = (
                2 if bool(row.reliable) else 1 if row.prospective_attempts > 0 else 0
            )
        image = np.ma.masked_invalid(matrix)
        ax.imshow(image, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax.set_title(panel)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel("Target DG unit")
        ax.set_ylabel("Source DG unit")
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.20, hspace=0.36, wspace=0.42)
    fig.legend(
        handles=(
            Patch(facecolor=colors[0], label="Unattempted"),
            Patch(facecolor=colors[1], label="Attempted, not reliable"),
            Patch(facecolor=colors[2], label="Stored reliable"),
        ),
        loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False, fontsize=18,
    )
    stem = RESULTS / "graph_seed99_75m_adjacency"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
