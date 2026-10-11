"""Render equal-weight DG peak coverage on the shared observation support.

The input tables come from analyze_common_peaks.py. No model replay or map
resampling happens here; each selected unit contributes one Gaussian at its
maximum supported-bin response. The same density scale is used in each figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def density(peaks: np.ndarray, support: np.ndarray, sigma: float) -> np.ndarray:
    grid = np.indices(support.shape, dtype=float)
    result = np.zeros(support.shape, dtype=float)
    for row, col in peaks:
        result += np.exp(-((grid[0] - row) ** 2 + (grid[1] - col) ** 2) / (2 * sigma**2))
    result[~support] = np.nan
    result /= np.nansum(result)
    return result


def panel(
    rows: pd.DataFrame,
    support: np.ndarray,
    selections: list[tuple[str, int, str]],
    outfile: Path,
    sigma: float,
    cutoff: float,
    figure_title: str,
) -> None:
    surfaces = []
    for condition, seed, title in selections:
        selected = rows.loc[(rows.condition == condition) & (rows.seed == seed)]
        selected = selected.loc[(selected.active_fraction > cutoff) & (selected.peak_rate > 0)]
        peaks = selected[["peak_row", "peak_col"]].to_numpy(dtype=float)
        if not len(peaks):
            raise ValueError(f"No valid peaks: {condition}, seed {seed}")
        surfaces.append((density(peaks, support, sigma), peaks, title, len(peaks)))
    maximum = max(float(np.nanmax(surface)) for surface, _, _, _ in surfaces)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.3), constrained_layout=True)
    for ax, (surface, peaks, title, count) in zip(axes.flat, surfaces):
        # The evaluator's histogramdd bins x on axis 0 and y on axis 1.
        image = ax.imshow(surface.T, cmap="magma", vmin=0, vmax=maximum, origin="lower")
        ax.scatter(peaks[:, 0], peaks[:, 1], s=43, c="#56E4FF", edgecolor="#112233", linewidth=0.75)
        ax.set_title(f"{title}\n{count} valid peaks", fontsize=21)
        ax.set_xlabel("Position bin x", fontsize=19)
        ax.set_ylabel("Position bin y", fontsize=19)
        ax.tick_params(labelsize=16)
        ax.set_xticks([0, 5, 10, 15, 18])
        ax.set_yticks([0, 5, 10, 15, 18])
    bar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015)
    bar.set_label("Peak density on observed bins", fontsize=19)
    bar.ax.tick_params(labelsize=16)
    fig.suptitle(f"{figure_title} · sigma {sigma:g} bin · activity > {cutoff * 100:g}%", fontsize=22)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile.with_suffix(".svg"))
    fig.savefig(outfile.with_suffix(".png"), dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("peak_units", type=Path)
    parser.add_argument("support_mask", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = pd.read_csv(args.peak_units)
    support = np.loadtxt(args.support_mask, delimiter=",").astype(bool)
    if support.shape != (19, 19):
        raise ValueError(f"Unexpected support shape {support.shape}")
    panel(rows, support, [
        ("COS27_C01_CONSTANT", 99, "C01 constant encoder"),
        ("COS27_C01_TEMPORAL", 99, "C01 temporal encoder"),
        ("COS27_C15_CONSTANT", 17, "C15 constant encoder"),
        ("COS27_C15_TEMPORAL", 17, "C15 temporal encoder"),
    ], args.output_dir / "common_peak_density_c01_c15", 1.0, 0.0025,
          "DG peak density · C01 seed 99 / C15 seed 17")
    panel(rows, support, [
        ("COS27_C05_CONSTANT_CONSTANT", 99, "Constant encoder\nConstant worker"),
        ("COS27_C05_TEMPORAL_CONSTANT", 99, "Temporal encoder\nConstant worker"),
        ("COS27_C05_CONSTANT_TEMPORAL", 99, "Constant encoder\nTemporal worker"),
        ("COS27_C05_TEMPORAL_TEMPORAL", 99, "Temporal encoder\nTemporal worker"),
    ], args.output_dir / "common_peak_density_c05", 1.0, 0.0025,
          "C05 seed 99 · DG peak density")


if __name__ == "__main__":
    main()
