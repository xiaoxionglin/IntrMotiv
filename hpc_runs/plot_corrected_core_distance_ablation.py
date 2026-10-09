"""Render paired 100M DG maps and frozen-policy paths for the hit-bonus study.

Inputs are the unmodified evaluation NPZ and pose CSV files bundled in the
report's visual-input archive. The common-panel maps share exact observations;
the pose CSV files are separate 10k-decision frozen-policy rollouts.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "06_experiments/ca3_goals"
INPUT = REPORT / "results/corrected_core_hit_only_20261009/distance_ablation_visual_inputs.zip"
OUTPUT = REPORT / "assets/corrected_core_distance_reward_ablation_20261009"
CONDITIONS = ("historical", "hit_only")
LABELS = {"historical": "Temporal bonus", "hit_only": "Hit only"}
ARENA = (100, 2000)


def read_pair(archive: ZipFile, family: str, seed: int, condition: str) -> tuple[dict, pd.DataFrame]:
    prefix = f"{family}/s{seed}/{condition}"
    with np.load(BytesIO(archive.read(f"{prefix}/field_maps.npz"))) as loaded:
        fields = {name: loaded[name] for name in (
            "occupancy", "post_inhibition_rate_maps", "post_inhibition_spatial_information",
            "post_inhibition_field_eligible", "post_inhibition_field_mono",
        )}
    pose = pd.read_csv(BytesIO(archive.read(f"{prefix}/pose.csv")))
    assert len(pose) == 10001
    return fields, pose


def occupancy(pose: pd.DataFrame) -> np.ndarray:
    counts, _, _ = np.histogram2d(
        pose.x, pose.y, bins=19, range=(ARENA, ARENA),
    )
    return counts.T


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def footprint_figure(archive: ZipFile, family: str, seeds: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(len(seeds), 2, figsize=(10, 5.0 * len(seeds)),
                             layout="constrained", squeeze=False)
    image = None
    for row, seed in enumerate(seeds):
        for col, condition in enumerate(CONDITIONS):
            _, pose = read_pair(archive, family, seed, condition)
            counts = occupancy(pose)
            ax = axes[row, col]
            image = ax.imshow(
                np.ma.masked_equal(counts, 0), origin="lower", extent=(*ARENA, *ARENA),
                cmap="cividis", norm=LogNorm(vmin=1, vmax=300), interpolation="nearest",
            )
            ax.set_title(f"{family.upper()} seed {seed} · {LABELS[condition]}\n"
                         f"{np.count_nonzero(counts)}/361 bins visited", fontsize=16)
            ax.set(xlabel="x (arena units)", ylabel="y (arena units)")
            ax.set_xticks((100, 1000, 2000))
            ax.set_yticks((100, 1000, 2000))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.55,
                 label="Observations per visited bin (log scale)")
    save(fig, f"{family}_paired_100m_trajectory_footprints")


def first_episode_figure(archive: ZipFile) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
    for ax, condition in zip(axes, CONDITIONS):
        _, pose = read_pair(archive, "c05", 99, condition)
        first = pose.loc[pose.num_traj == pose.num_traj.iloc[0]]
        assert len(first) == 900
        ax.plot(first.x, first.y, color="#0072B2", lw=1.0, alpha=0.8)
        ax.scatter(first.x.iloc[0], first.y.iloc[0], color="#009E73", s=75,
                   marker="o", zorder=3, label="Start")
        ax.scatter(first.x.iloc[-1], first.y.iloc[-1], color="#D55E00", s=85,
                   marker="x", linewidths=2.4, zorder=3, label="End")
        ax.set(xlim=ARENA, ylim=ARENA, aspect="equal", xlabel="x (arena units)",
               ylabel="y (arena units)")
        ax.set_title(f"C05 seed 99 · {LABELS[condition]}", fontsize=16)
        ax.legend(loc="lower right", fontsize=13)
    save(fig, "c05_s99_paired_first_episode_paths")


def field_atlas_figure(archive: ZipFile, family: str, seed: int) -> None:
    # Each unit uses its own peak normalization, as in the canonical contact
    # sheet. Units are not matched identities across independently trained runs.
    fig, axes = plt.subplots(4, 8, figsize=(18, 9.5), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    for block, condition in enumerate(CONDITIONS):
        fields, _ = read_pair(archive, family, seed, condition)
        maps = fields["post_inhibition_rate_maps"]
        mask = fields["occupancy"] == 0
        for unit in range(16):
            ax = axes[unit // 4, block * 4 + unit % 4]
            this_map = maps[:, :, unit]
            peak = float(this_map.max())
            normed = this_map / peak if peak > 0 else this_map
            ax.imshow(np.ma.array(normed, mask=mask), origin="lower", cmap=cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            note = "silent" if peak == 0 else ("single" if fields["post_inhibition_field_mono"][unit] else "")
            ax.set_title(f"DG {unit}" + (f" · {note}" if note else ""), fontsize=12)
            ax.set_xticks(())
            ax.set_yticks(())
    fig.text(0.27, 1.01, "Temporal bonus", ha="center", fontsize=18, weight="bold")
    fig.text(0.73, 1.01, "Hit only", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, -0.01,
             f"{family.upper()} seed {seed} · identical 10,001-observation panel · gray: unvisited · color: activity / each unit peak",
             ha="center", fontsize=15)
    save(fig, f"{family}_s{seed}_paired_100m_common_panel_fields")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "pdf.fonttype": 42})
    with ZipFile(INPUT) as archive:
        reference = None
        for family, seeds in (("c05", (8, 99, 123)), ("c15", (8, 123))):
            for seed in seeds:
                for condition in CONDITIONS:
                    fields, _ = read_pair(archive, family, seed, condition)
                    if reference is None:
                        reference = fields["occupancy"]
                    assert np.array_equal(reference, fields["occupancy"]), (
                        "Common-observation panel occupancy changed", family, seed, condition
                    )
        assert np.count_nonzero(reference) == 242
        footprint_figure(archive, "c05", (8, 99, 123))
        footprint_figure(archive, "c15", (8, 123))
        first_episode_figure(archive)
        field_atlas_figure(archive, "c05", 99)
        field_atlas_figure(archive, "c15", 8)


if __name__ == "__main__":
    main()
