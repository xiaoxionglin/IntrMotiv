"""Render paired 100M DG maps and frozen-policy paths for the hit-bonus study.

Inputs are the unmodified evaluation NPZ and pose CSV files bundled in the
report's visual-input archive. The common-panel maps share exact observations;
the pose CSV files are separate 10k-decision frozen-policy rollouts.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "06_experiments/ca3_goals"
INPUT = REPORT / "results/corrected_core_hit_only_20261009/distance_ablation_visual_inputs.zip"
OUTPUT = REPORT / "assets/corrected_core_distance_reward_ablation_20261009"
RESULTS = REPORT / "results/corrected_core_hit_only_20261009"
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


def peak_center_records(archive: ZipFile) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One strongest observed post-inhibition map bin for each active DG unit."""
    rows: list[dict] = []
    summaries: list[dict] = []
    for family, seeds in (("c05", (8, 99, 123)), ("c15", (8, 123))):
        for seed in seeds:
            for condition in CONDITIONS:
                fields, _ = read_pair(archive, family, seed, condition)
                maps = fields["post_inhibition_rate_maps"]
                visited = fields["occupancy"] > 0
                assert maps.shape == (19, 19, 16)
                centers = []
                for unit in range(16):
                    observed_map = np.where(visited, maps[:, :, unit], -np.inf)
                    peak = float(observed_map.max())
                    active = peak > 0
                    row, col = np.unravel_index(int(observed_map.argmax()), (19, 19)) if active else (-1, -1)
                    if active:
                        centers.append((int(row), int(col)))
                    rows.append({
                        "family": family.upper(), "seed": seed, "condition": condition,
                        "unit": unit, "active": active,
                        "peak_row": int(row) if active else np.nan,
                        "peak_col": int(col) if active else np.nan,
                        "x_center": 150 + int(col) * 100 if active else np.nan,
                        "y_center": 150 + int(row) * 100 if active else np.nan,
                        "peak_rate": peak if active else 0.0,
                        "outer_border": bool(row in (0, 18) or col in (0, 18)) if active else False,
                        "field_eligible": bool(fields["post_inhibition_field_eligible"][unit]),
                        "single_field": bool(fields["post_inhibition_field_mono"][unit]),
                    })
                points = np.asarray(centers, dtype=float)
                distances = np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=-1))
                upper = distances[np.triu_indices(len(points), k=1)]
                summaries.append({
                    "family": family.upper(), "seed": seed, "condition": condition,
                    "active_units": len(centers), "unique_peak_bins": len(set(centers)),
                    "outer_border_peaks": sum(r in (0, 18) or c in (0, 18) for r, c in centers),
                    "mean_pairwise_peak_distance_bins": float(upper.mean()),
                })
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def draw_peak_map(ax: plt.Axes, fields: dict, centers: pd.DataFrame,
                  family: str, seed: int, condition: str) -> None:
    visited = fields["occupancy"] > 0
    ax.imshow(np.where(visited, 1.0, np.nan), origin="lower", extent=(*ARENA, *ARENA),
              cmap="Greys", vmin=0, vmax=2.6, interpolation="nearest")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in centers.itertuples():
        if row.active:
            groups[(int(row.peak_row), int(row.peak_col))].append(int(row.unit))
    for (row, col), units in groups.items():
        x, y = 150 + col * 100, 150 + row * 100
        shared = len(units) > 1
        size = 750 if len(units) >= 3 else (420 if shared else 250)
        label = "\n".join(map(str, units)) if len(units) >= 3 else ",".join(map(str, units))
        ax.scatter(x, y, s=size,
                   c="#B44F10" if shared else "#006BA4", edgecolors="white",
                   linewidths=1.0, zorder=3)
        ax.text(x, y, label, color="white", ha="center", va="center",
                fontsize=8 if len(units) >= 3 else (9 if shared else 10),
                linespacing=0.8, weight="bold", zorder=4)
    active = int(centers.active.sum())
    border = int(centers.outer_border.sum())
    ax.set_title(f"{family.upper()} seed {seed} · {LABELS[condition]}\n"
                 f"{active} active · {len(groups)} peak bins · {border} outer-border peaks",
                 fontsize=14)
    ax.set(xlim=(50, 2050), ylim=(50, 2050), aspect="equal",
           xlabel="x (arena units)", ylabel="y (arena units)")
    ax.set_xticks((100, 1000, 2000))
    ax.set_yticks((100, 1000, 2000))


def peak_center_figures(archive: ZipFile) -> None:
    centers, summary = peak_center_records(archive)
    centers.to_csv(RESULTS / "peak_centers_common_panel100.csv", index=False)
    summary.to_csv(RESULTS / "peak_center_summary_common_panel100.csv", index=False)
    for family, seeds in (("c05", (8, 99, 123)), ("c15", (8, 123))):
        for seed in seeds:
            fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="constrained")
            for ax, condition in zip(axes, CONDITIONS):
                fields, _ = read_pair(archive, family, seed, condition)
                selection = centers.loc[(centers.family == family.upper()) &
                                        (centers.seed == seed) &
                                        (centers.condition == condition)]
                draw_peak_map(ax, fields, selection, family, seed, condition)
            save(fig, f"{family}_s{seed}_paired_100m_peak_centers")
        fig, axes = plt.subplots(len(seeds), 2, figsize=(11, 5.5 * len(seeds)),
                                 layout="constrained", squeeze=False)
        for row, seed in enumerate(seeds):
            for col, condition in enumerate(CONDITIONS):
                fields, _ = read_pair(archive, family, seed, condition)
                selection = centers.loc[(centers.family == family.upper()) &
                                        (centers.seed == seed) &
                                        (centers.condition == condition)]
                draw_peak_map(axes[row, col], fields, selection, family, seed, condition)
        save(fig, f"{family}_all_paired_100m_peak_centers")


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
        peak_center_figures(archive)


if __name__ == "__main__":
    main()
