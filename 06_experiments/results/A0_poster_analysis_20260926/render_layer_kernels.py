"""Render saved common-history DG, CA3, and decoder-1 spatial kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none"})
LAYERS = (("dg", "DG"), ("ca3", "CA3"), ("decoder_1", "Decoder-1"))
SHORT_NAMES = {
    "CPU2048_DIRECT_F16_DDQN_S99_300007424": "Direct F16 DDQN · seed 99 · 300M",
    "CPU2048_WAYPOINT_DECODER_F64_DDQN_S99_300007424": "Waypoint F64 DDQN · seed 99 · 300M",
    "FSCS_WAYPOINT_DECODER_F64_PPO_S123_192266240": "Full-system PPO · seed 123 · 192M",
}


parser = argparse.ArgumentParser()
parser.add_argument("--kernels", type=Path, default=HERE / "kernels")
parser.add_argument("--output-root", type=Path, default=HERE / "poster_candidates")
parser.add_argument("--labels", nargs="*", help="Optional NPZ stems to render")
args = parser.parse_args()

for path in sorted(args.kernels.glob("*.npz")):
    if args.labels and path.stem not in args.labels:
        continue
    with np.load(path) as archive:
        data = {name: archive[name] for name in archive.files}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for ax, (key, label) in zip(axes, LAYERS):
        kernel = data[f"{key}_kernel"]
        pairs = data[f"{key}_pair_count"]
        radius = kernel.shape[0] // 2
        mask = pairs.T < 10
        mask[kernel.shape[0] // 2, kernel.shape[1] // 2] = True
        image = ax.imshow(np.ma.array(kernel.T, mask=mask),
                          origin="lower", extent=(-radius-.5, radius+.5,
                                                   -radius-.5, radius+.5),
                          cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        ax.set(xlabel="Δx (100-unit bins)", ylabel="Δy (100-unit bins)", title=label)
    fig.colorbar(image, ax=axes, shrink=.75, label="Population-vector correlation")
    fig.suptitle(f"Common-history spatial kernels · {SHORT_NAMES.get(path.stem, path.stem)} · full ±{radius} bins", fontsize=16)
    output = args.output_root / path.stem / "layerwise_kernels.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    dx, dy = np.meshgrid(np.arange(-radius, radius+1),
                         np.arange(-radius, radius+1), indexing="ij")
    radius = np.hypot(dx, dy)
    for key, label in LAYERS:
        kernel = data[f"{key}_kernel"]
        pairs = data[f"{key}_pair_count"]
        centers, values = [], []
        for distance in range(0, int(np.floor(radius.max()+.5))+1):
            chosen = (radius >= distance-.5) & (radius < distance+.5) & (pairs >= 10)
            if chosen.any():
                centers.append(distance)
                values.append(float(np.nanmean(kernel[chosen])))
        ax.plot(centers, values, marker="o", label=label)
    ax.set(xlabel="Displacement radius (100-unit bins)", ylabel="Population-vector correlation",
           title="Radial spatial-kernel profile", ylim=(-.1, 1.05))
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output.with_name("layerwise_radial.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
