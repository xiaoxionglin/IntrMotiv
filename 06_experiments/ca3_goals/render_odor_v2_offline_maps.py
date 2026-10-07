"""Render readable 10k-decision DG maps from the established evaluator NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def draw_maps(
    maps: np.ndarray,
    occupancy: np.ndarray,
    bounds: np.ndarray,
    output: Path,
    title: str,
    *,
    signed: bool,
) -> None:
    cmap = plt.get_cmap("RdBu_r" if signed else "viridis").copy()
    cmap.set_bad("#d9d9d9")
    observed = np.where(occupancy[:, :, None] > 0, maps, np.nan)
    if signed:
        scale = float(np.nanmax(np.abs(observed)))
        vmin, vmax = -scale, scale
    else:
        peaks = np.nanmax(observed, axis=(0, 1))
        observed = np.divide(observed, peaks, out=np.full_like(observed, np.nan), where=peaks > 0)
        vmin, vmax = 0.0, 1.0

    figure, axes = plt.subplots(4, 4, figsize=(14, 13), sharex=True, sharey=True)
    for unit, axis in enumerate(axes.flat):
        image = axis.imshow(
            np.ma.masked_invalid(observed[:, :, unit]),
            origin="lower",
            extent=bounds,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(f"DG {unit}", fontsize=18)
        axis.set_xticks([100, 1050, 2000])
        axis.set_yticks([100, 1050, 2000])
        axis.tick_params(labelsize=14)
    for axis in axes[-1]:
        axis.set_xlabel("X position", fontsize=16)
    for axis in axes[:, 0]:
        axis.set_ylabel("Y position", fontsize=16)
    figure.subplots_adjust(left=0.08, right=0.87, bottom=0.07, top=0.90, wspace=0.13, hspace=0.27)
    colorbar_axis = figure.add_axes((0.89, 0.22, 0.018, 0.58))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Pre-threshold logit" if signed else "Activity / unit peak", fontsize=17)
    colorbar.ax.tick_params(labelsize=15)
    figure.suptitle(title + " · 10k decisions · gray: unvisited", fontsize=20)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    font_path = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"),
                                           fallback_to_default=False))
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"A scalable font is required, found {font_path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16})
    with np.load(args.input, allow_pickle=False) as data:
        occupancy = np.asarray(data["occupancy"])
        bounds = np.asarray(data["bounds"], dtype=float)
        rate_maps = np.asarray(data["rate_maps"], dtype=float)
        logits = np.asarray(data["pre_threshold_rate_maps"], dtype=float)
    if rate_maps.shape != logits.shape or rate_maps.shape != (19, 19, 16):
        raise ValueError("Expected matched 19-by-19 maps for 16 DG units")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_maps(rate_maps, occupancy, bounds, args.output_dir / f"{args.label}_place_fields.png",
              args.label.replace("_", " ") + " · thresholded DG", signed=False)
    draw_maps(logits, occupancy, bounds, args.output_dir / f"{args.label}_pre_threshold_logits.png",
              args.label.replace("_", " ") + " · pre-threshold DG", signed=True)


if __name__ == "__main__":
    main()
