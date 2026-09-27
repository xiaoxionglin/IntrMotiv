"""Render SVGs from canonical frozen place-field NPZ and pose.csv outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "svg.fonttype": "none"})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fields(archive, out):
    maps = archive["raw_dg_rate_maps"]
    occupancy = archive["raw_dg_occupancy"]
    count = maps.shape[-1]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    for start in range(0, count, 16):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
        for offset, ax in enumerate(axes.flat):
            unit = start + offset
            if unit >= count:
                ax.set_visible(False)
                continue
            values = maps[..., unit]
            peak = np.nanmax(values)
            normalized = values / peak if peak > 0 else values
            image = ax.imshow(np.ma.array(normalized.T, mask=(occupancy.T == 0)), origin="lower",
                              extent=(100, 2000, 100, 2000), vmin=0, vmax=1, cmap=cmap,
                              interpolation="nearest")
            silent = archive["raw_dg_active_fraction"][unit] == 0
            ax.set_title(f"DG unit {unit}" + (" · silent" if silent else ""))
            ax.set_xticks((100, 2000))
            ax.set_yticks((100, 2000))
        fig.suptitle(f"Frozen-policy DG place fields · units {start}–{min(start+15, count-1)}")
        fig.colorbar(image, ax=list(axes.flat), shrink=.65, label="Activity / unit peak (gray: unvisited)")
        save(fig, out / f"frozen_fields_page{start//16+1:02d}.svg")


def trajectory_and_flow(rows, out):
    xy = np.array([(float(row["x"]), float(row["y"])) for row in rows])
    episode = np.array([(int(row["agent"]), int(row["num_traj"])) for row in rows])
    grain = 19
    edges = np.linspace(100, 2000, grain+1)
    occupancy, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(edges, edges))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    image = ax0.imshow(np.ma.masked_equal(occupancy.T, 0), origin="lower",
                       extent=(100, 2000, 100, 2000), cmap="cividis", interpolation="nearest")
    fig.colorbar(image, ax=ax0, label="Frozen observations per bin")
    ax0.set_title("Occupancy")
    keys = list(dict.fromkeys(map(tuple, episode)))
    for index, key in enumerate(keys):
        part = xy[np.all(episode == key, axis=1)]
        ax1.plot(part[:, 0], part[:, 1], linewidth=.8, alpha=.6,
                 color=plt.get_cmap("turbo")(index/max(1,len(keys)-1)))
    ax1.set_title(f"{len(keys)} independent episodes")
    for ax in (ax0, ax1):
        ax.set(xlim=(100, 2000), ylim=(100, 2000), aspect="equal",
               xlabel="x (DMLab units)", ylabel="y (DMLab units)")
    save(fig, out / "frozen_trajectory.svg")

    same = np.all(episode[1:] == episode[:-1], axis=1)
    delta = xy[1:] - xy[:-1]
    same &= np.linalg.norm(delta, axis=1) < 200
    bins = np.searchsorted(edges, xy[:-1], side="right") - 1
    same &= (bins >= 0).all(axis=1) & (bins < grain).all(axis=1)
    bins, delta = bins[same], delta[same]
    count = np.zeros((grain, grain))
    total = np.zeros((grain, grain, 2))
    path_length = np.zeros((grain, grain))
    np.add.at(count, (bins[:, 0], bins[:, 1]), 1)
    np.add.at(total, (bins[:, 0], bins[:, 1]), delta)
    np.add.at(path_length, (bins[:, 0], bins[:, 1]), np.linalg.norm(delta, axis=1))
    coherence = np.divide(np.linalg.norm(total, axis=2), path_length,
                          out=np.zeros_like(count), where=path_length > 0)
    norm = np.linalg.norm(total, axis=2)
    direction = np.divide(total, norm[..., None], out=np.zeros_like(total), where=norm[..., None] > 0)
    arrows = direction * (55 * coherence[..., None])
    centers = (edges[1:] + edges[:-1]) / 2
    x, y = np.meshgrid(centers, centers, indexing="ij")
    shown = count >= 5
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(np.ma.masked_equal(count.T, 0), origin="lower",
                      extent=(100, 2000, 100, 2000), cmap="cividis")
    ax.quiver(x[shown], y[shown], arrows[..., 0][shown], arrows[..., 1][shown],
              color="white", angles="xy", scale_units="xy", scale=1, width=.004)
    ax.set(xlabel="x (DMLab units)", ylabel="y (DMLab units)", title="Frozen occupancy flow (n ≥ 5)")
    fig.colorbar(image, ax=ax, label="Source-bin transition count")
    save(fig, out / "frozen_flow.svg")
    return len(rows), len(keys), float((occupancy > 0).mean()), float(np.mean(coherence[shown]))


summaries = []
for directory in sorted((HERE / "frozen_raw").glob("*")):
    if not (directory / "place_fields.npz").is_file():
        continue
    out = HERE / "poster_candidates" / ("direct_f16_ddqn_s99_300m" if "DIRECT" in directory.name else "waypoint_f64_ddqn_s99_300m")
    with np.load(directory / "place_fields.npz") as archive:
        fields(archive, out)
        active_mask = archive["raw_dg_active_fraction"] > 0
        active = int(active_mask.sum())
        mono = float(np.mean(archive["raw_dg_field_mono"][archive["raw_dg_field_eligible"]]))
        maps = archive["raw_dg_rate_maps"]
        occupied = archive["raw_dg_occupancy"] > 0
        vectors = np.nan_to_num(maps[occupied][:, active_mask]).T
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors[norms > 0] / norms[norms > 0, None]
        similarities = vectors @ vectors.T
        map_cosine = float(similarities[np.triu_indices(len(vectors), 1)].mean())
        spatial_information = float(np.mean(archive["raw_dg_spatial_information"][active_mask]))
        peak_bins = int(np.unique(np.nanargmax(maps.reshape(-1, maps.shape[-1])[:, active_mask], axis=0)).size)
    with (directory / "pose.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    observations, episodes, coverage, coherence = trajectory_and_flow(rows, out)
    summaries.append(dict(run=directory.name, observations=observations, episodes=episodes,
                          coverage_fraction=coverage, mean_flow_coherence=coherence,
                          active_units=active, mono_field_fraction=mono,
                          active_only_map_cosine=map_cosine,
                          active_mean_spatial_information=spatial_information,
                          distinct_peak_bins=peak_bins))
if summaries:
    with (HERE / "frozen_exemplar_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # A compact poster panel selected by a declared rule: top four active
    # units by spatial information within each fixed mature checkpoint.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), constrained_layout=True)
    for row_index, key in enumerate(("DIRECT", "WAYPOINT")):
        directory = next(p for p in (HERE / "frozen_raw").glob("*") if key in p.name)
        with np.load(directory / "place_fields.npz") as archive:
            maps = archive["raw_dg_rate_maps"]
            occupancy = archive["raw_dg_occupancy"]
            information = archive["raw_dg_spatial_information"]
            active = np.flatnonzero(archive["raw_dg_active_fraction"] > 0)
            selected = active[np.argsort(information[active])[-4:][::-1]]
            for ax, unit in zip(axes[row_index], selected):
                values = maps[..., unit]
                peak = np.nanmax(values)
                normalized = values / peak if peak > 0 else values
                image = ax.imshow(np.ma.array(normalized.T, mask=occupancy.T == 0),
                                  origin="lower", extent=(100, 2000, 100, 2000),
                                  cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"{key.title()} u{unit} · {information[unit]:.3f} bits", fontsize=14)
                ax.set_xticks((100, 2000))
                ax.set_yticks((100, 2000))
    fig.colorbar(image, ax=list(axes.flat), shrink=.75, label="Activity / unit peak (gray: unvisited)")
    fig.suptitle("Mature frozen DG fields · top four active units by spatial information", fontsize=18)
    save(fig, HERE / "poster_candidates" / "frozen_top_information_fields.svg")
