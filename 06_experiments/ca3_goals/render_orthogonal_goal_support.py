"""Show observed first-four DG goal support in the finite seed-99 frozen rollout.

Input bins are derived from post-replacement ``rate_maps`` and ``occupancy``
in the canonical 75M, 10k-decision place-field NPZs. A blue bin means the DG
unit was positive at least once there; it does not establish a stable field.
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


def main() -> None:
    cells = pd.read_csv(RESULTS / "frozen_finite_s99_goal_support.csv")
    assert len(cells) == 2 * 4 * 19 * 19
    colors = ("#dddddd", "#ffffff", "#0072B2")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 24,
        "axes.titlesize": 26,
        "figure.titlesize": 29,
        "pdf.fonttype": 42,
    })
    for arm in ("Prescribed", "Learned-4"):
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for unit in range(4):
            ax = axes.flat[unit]
            block = cells.loc[cells.arm.eq(arm) & cells.unit.eq(unit)]
            assert len(block) == 19 * 19
            state = np.zeros((19, 19), dtype=int)
            state[block.y_bin, block.x_bin] = np.where(
                block.goal_event_observed, 2, np.where(block.visited, 1, 0)
            )
            ax.imshow(state, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5,
                      origin="lower", interpolation="nearest")
            ax.set_title(f"DG {unit}: {int((state == 2).sum())} bins")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{arm} DG · 75M frozen")
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.20,
                            hspace=0.28, wspace=0.12)
        fig.legend(
            handles=(Patch(facecolor=colors[0], label="Unvisited"),
                     Patch(facecolor=colors[1], edgecolor="#999999", label="Visited, no event"),
                     Patch(facecolor=colors[2], label="Goal event observed")),
            loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2,
            frameon=False, fontsize=19,
        )
        suffix = "oracle" if arm == "Prescribed" else "learned4"
        stem = RESULTS / f"frozen_finite_s99_{suffix}_goal_support"
        fig.savefig(stem.with_suffix(".png"), dpi=200)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
