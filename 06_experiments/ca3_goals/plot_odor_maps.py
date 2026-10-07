"""Render the fixed, pre-noise odor fields from the study calibration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "odor_gain_calibration_20261007.json"
OUTPUT = HERE / "figures" / "odor_clean_fields_v2_20261007.png"


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    centers = np.asarray(calibration["odor_centers"], dtype=float)
    sigma = float(calibration["odor_spatial_sigma"])
    if centers.shape != (4, 2) or sigma <= 0:
        raise ValueError("Expected four two-dimensional odor centers and a positive width")

    font_path = Path(
        font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"),
            fallback_to_default=False,
        )
    )
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"A scalable font is required, found {font_path}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18})

    coordinate = np.linspace(100, 2000, 401)
    x, y = np.meshgrid(coordinate, coordinate)
    fields = np.exp(
        -((x[None] - centers[:, 0, None, None]) ** 2
          + (y[None] - centers[:, 1, None, None]) ** 2)
        / (2 * sigma**2)
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(
            fields[index],
            origin="lower",
            extent=(100, 2000, 100, 2000),
            vmin=0,
            vmax=1,
            cmap="viridis",
            interpolation="bilinear",
        )
        axis.plot(*centers[index], marker="+", color="white", markersize=16, mew=3)
        axis.set_title(
            f"Odor {index + 1}: center ({centers[index, 0]:.0f}, {centers[index, 1]:.0f})",
            fontsize=20,
        )
        axis.set_xticks([100, 575, 1050, 1525, 2000])
        axis.set_yticks([100, 575, 1050, 1525, 2000])
        axis.tick_params(labelsize=16)
    for axis in axes[-1]:
        axis.set_xlabel("X position", fontsize=19)
    for axis in axes[:, 0]:
        axis.set_ylabel("Y position", fontsize=19)
    figure.subplots_adjust(left=0.08, right=0.82, bottom=0.08, top=0.91, wspace=0.19, hspace=0.18)
    colorbar_axis = figure.add_axes((0.855, 0.22, 0.025, 0.58))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Clean odor intensity", fontsize=19)
    colorbar.ax.tick_params(labelsize=16)
    figure.suptitle("Four fixed spatial odor fields (before observation noise)", fontsize=23)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=200)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
