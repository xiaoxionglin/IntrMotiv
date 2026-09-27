"""Render seed-averaged DG, CA3, and decoder-1 kernels from common replay.

The seed is the averaging unit. A displacement is shown only when at least two
seeds have ten or more valid occupied-cell pairs there. Kernel construction is
in ``collect_poster_population_kernels.py``; this file only presents it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def kernel_for_condition(manifest: list[dict], directory: Path, condition: str,
                         layer: str, heading: bool, role: str) -> tuple[np.ndarray, int]:
    suffix = "_heading" if heading else ""
    pieces = []
    for row in manifest:
        if row["condition"] != condition or row["analysis_role"] != role:
            continue
        path = directory / f"{row['label_suffix']}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            kernel = data[f"{layer}{suffix}_kernel"].copy()
            counts = data[f"{layer}{suffix}_pair_count"]
            kernel[counts < 10] = np.nan
            pieces.append(kernel)
    if len(pieces) != 3:
        raise ValueError(f"Expected three {condition} seed kernels, found {len(pieces)}")
    stack = np.stack(pieces)
    valid = np.isfinite(stack).sum(axis=0) >= 2
    finite = np.isfinite(stack)
    average = np.divide(np.nansum(stack, axis=0), finite.sum(axis=0),
                        out=np.full(stack.shape[1:], np.nan), where=finite.sum(axis=0) > 0)
    average[~valid] = np.nan
    return average, len(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--condition-a", required=True)
    parser.add_argument("--condition-b", required=True)
    parser.add_argument("--role-a", default="endpoint")
    parser.add_argument("--role-b", default="endpoint")
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--heading-matched", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 16, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.fonttype": "none"})
    layers = (("dg", "DG"), ("ca3", "CA3"), ("decoder_1", "Decoder-1"))
    arms = ((args.condition_a, args.label_a, args.role_a),
            (args.condition_b, args.label_b, args.role_b))
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True,
                             sharex=True, sharey=True)
    scale = TwoSlopeNorm(vmin=-.2, vcenter=0, vmax=1)
    image = None
    palette = plt.get_cmap("RdBu_r").copy()
    palette.set_bad("#d7d7d7")
    for row_index, (layer, layer_name) in enumerate(layers):
        for column_index, (condition, arm_name, role) in enumerate(arms):
            ax = axes[row_index, column_index]
            kernel, seeds = kernel_for_condition(manifest, args.kernels, condition,
                                                 layer, args.heading_matched, role)
            display = kernel.copy()
            radius = display.shape[0] // 2
            if display.shape != (2 * radius + 1, 2 * radius + 1):
                raise ValueError(f"Expected a square odd-sized kernel, got {display.shape}")
            display[radius, radius] = np.nan  # Identity self-correlation is always one.
            extent = (-radius - .5, radius + .5, -radius - .5, radius + .5)
            image = ax.imshow(display.T, origin="lower", extent=extent,
                              cmap=palette, norm=scale, interpolation="nearest")
            ax.set_aspect("equal")
            ax.set_title(f"{layer_name} · {arm_name}", loc="left")
            tick_step = max(1, radius // 3)
            ticks = sorted(set([-radius, -2 * tick_step, -tick_step, 0,
                                tick_step, 2 * tick_step, radius]))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlabel("East–west offset (100-unit bins)")
            ax.set_ylabel("North–south offset (100-unit bins)")
    fig.colorbar(image, ax=axes, label="Mean population-vector correlation",
                 shrink=.62, pad=.025)
    suffix = " · heading matched" if args.heading_matched else ""
    fig.suptitle(f"{args.title}{suffix}\nFull ±{radius}-bin range · common history · mean of 3 seeds · gray: fewer than 2 seeds with ≥10 pairs, or self",
                 fontsize=17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
