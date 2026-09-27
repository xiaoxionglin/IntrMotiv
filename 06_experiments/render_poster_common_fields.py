"""Render a compact DG field comparison from two canonical common-panel archives.

Each arm shows its four active units with the highest spatial information.
This selection is descriptive; population summaries use every active unit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fields(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        maps = data["rate_maps"].copy()
        occupancy = data["occupancy"].copy()
        information = data["spatial_information"].copy()
        active = data["active_fraction"] > 0
    if maps.ndim != 3 or maps.shape[:2] != occupancy.shape:
        raise ValueError(f"Map/occupancy shape mismatch: {path}")
    candidates = np.flatnonzero(active & np.isfinite(information))
    if len(candidates) < 4:
        raise ValueError(f"Fewer than four active DG units: {path}")
    selected = candidates[np.argsort(information[candidates])[-4:][::-1]]
    return maps, occupancy, information, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    first = fields(args.first)
    second = fields(args.second)
    if not np.array_equal(first[1], second[1]):
        raise ValueError("Two arms must have identical common-panel occupancy")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 13, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    color = plt.get_cmap("viridis").copy()
    color.set_bad("#d9d9d9")
    for row, (maps, occupancy, information, selected) in enumerate((first, second)):
        for column, unit in enumerate(selected):
            ax = axes[row, column]
            field = maps[:, :, unit]
            peak = np.nanmax(field)
            relative = field / peak if peak > 0 else field
            image = ax.imshow(np.ma.array(relative, mask=occupancy == 0), origin="lower",
                              cmap=color, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"DG {unit} · SI {information[unit]:.2f} bits")
            ax.set_xlabel("x bin")
            if column == 0:
                ax.set_ylabel((args.first_label if row == 0 else args.second_label) + "\ny bin")
    fig.colorbar(image, ax=axes, shrink=.78, label="Activity / unit peak")
    fig.suptitle(args.title + "\nSame 10,001-observation history · top four active units by SI")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
