"""Render standalone, provenance-checked trajectory thumbnail candidates.

Preparation extracts only 65 intact fragments from each selected 100M-frame
online-spatial snapshot, plus one archived verified-hit episode per HRL arm.
Normal rendering reads the small local archives and never needs NEMO2 access.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "selected_fragments.npz"
HIT_INPUTS = HERE / "verified_hit_episode_inputs.npz"
MANIFEST = HERE / "manifest.json"
FAMILIES = ("NG", "UG", "FG")
COLORS = {"NG": "#0072B2", "UG": "#D55E00", "FG": "#009E73"}
RUNS = {
    "NG": ("COS27_C01_TEMPORAL_S99", "245224fb0aadae446ececc8d7612323523f7d259786eb496f5276d3b2be3ddde"),
    "UG": ("COS27_C05_TEMPORAL_TEMPORAL_S123", "03096201542cf4e9681a6560fd4d0a058eec7af4987a81be5f5aea7675a67b33"),
    "FG": ("COS27_C15_TEMPORAL_S17", "bf977d531dfa1486af0442d10af69d8da0636b481d45ea7c5bdbf532279499b6"),
}
COHORTS = {
    "NG": "cosyne27_c01_temporal_s3_20261010",
    "UG": "cosyne27_c05_pushpull_s3_20261010",
    "FG": "cosyne27_c15_encoder_s2_20261010",
}
ONLINE_ROOT = ("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/"
               "train_dir/analysis/online_spatial")
HIT_ROOT = ("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/"
            "train_dir/analysis/flat_goal_option_events_20260928/output/raw")
HIT_SOURCES = {
    "C05": ("c05_seed99_pose_events.csv", "3a50f9e55f9ff6044131f8cfc0357802aec44403c77487d90c05de44f6c116b7", 128, 44),
    "C15": ("c15_seed99_pose_events.csv", "da1745c40a98721932546135315ddc0fa3f8ede353713e71a403153fd3eb8c22", 12, 1),
}
ARCHIVE_COUNT = 65
VARIANTS = {"09_fragments": 9, "33_fragments": 33, "65_fragments": 65}
BOUNDS = (100.0, 2000.0, 100.0, 2000.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contiguous_slices(segment_id: np.ndarray, dones: np.ndarray) -> list[slice]:
    """Use the same boundary rule as the canonical spatial trajectory renderer."""
    starts = np.r_[0, np.flatnonzero((segment_id[1:] != segment_id[:-1]) |
                                      dones[:-1].astype(bool)) + 1]
    ends = np.r_[starts[1:], len(segment_id)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def prepare(raw_root: Path) -> None:
    """Hash-check full sources, then archive only the selected plot coordinates."""
    fragments: dict[str, np.ndarray] = {}
    for family_index, (family, (run_name, expected_hash)) in enumerate(RUNS.items()):
        source = raw_root / family / "online_spatial.npz"
        if sha256(source) != expected_hash:
            raise ValueError(f"Online-spatial snapshot hash mismatch: {source}")
        with np.load(source, allow_pickle=False) as payload:
            pose = np.asarray(payload["pose"], dtype=np.float32)
            segment_id = np.asarray(payload["segment_id"])
            dones = np.asarray(payload["dones"])
            bounds = tuple(np.asarray(payload["bounds"], dtype=float))
            window = (int(payload["window_start_env_steps"]),
                      int(payload["window_end_env_steps"]))
            recorded_run = str(payload["run_name"])
        if (pose.shape != (100000, 3) or segment_id.shape != (len(pose),) or
                dones.shape != (len(pose),) or bounds != BOUNDS or
                recorded_run != run_name or window != (99207936, 100007936)):
            raise ValueError(f"Unexpected {family} snapshot contract")
        all_slices = contiguous_slices(segment_id, dones)
        eligible = [(ordinal, part) for ordinal, part in enumerate(all_slices)
                    if part.stop - part.start > 1]
        # Random sampling avoids periodic aliasing with the parallel actor order.
        # The first 9 and 33 draws form nested, predeclared smaller candidates.
        chosen = np.random.default_rng(2027 + family_index).permutation(len(eligible))[:ARCHIVE_COUNT]
        if len(np.unique(chosen)) != ARCHIVE_COUNT:
            raise ValueError(f"Repeated selected fragment for {family}")
        picked = [eligible[index] for index in chosen]
        positions = [pose[part, :2] for _, part in picked]
        offsets = np.r_[0, np.cumsum([len(part) for part in positions])]
        fragments[f"xy_{family}"] = np.concatenate(positions).astype(np.float32)
        fragments[f"offsets_{family}"] = offsets.astype(np.int32)
        fragments[f"ordinals_{family}"] = np.array([ordinal for ordinal, _ in picked], dtype=np.int32)
        fragments[f"start_rows_{family}"] = np.array([part.start for _, part in picked], dtype=np.int32)
        fragments[f"total_segments_{family}"] = np.array(len(all_slices), dtype=np.int32)
        fragments[f"eligible_segments_{family}"] = np.array(len(eligible), dtype=np.int32)
    np.savez_compressed(INPUTS, **fragments)

    verified: dict[str, np.ndarray] = {}
    for family, (filename, expected_hash, expected_starts, expected_hits) in HIT_SOURCES.items():
        source = raw_root / filename
        if sha256(source) != expected_hash:
            raise ValueError(f"Verified hit stream hash mismatch: {source}")
        pose = pd.read_csv(source)
        if len(pose) != 10001 or not {"frame", "num_traj", "x", "y", "option_start", "goal_hit"}.issubset(pose):
            raise ValueError(f"Unexpected verified-hit pose stream: {source}")
        episode = pose.loc[pose.num_traj.eq(1)].copy()
        if (len(episode) != 900 or not np.array_equal(episode.frame.to_numpy(), np.arange(900, 1800)) or
                int(episode.option_start.sum()) != expected_starts or
                int(episode.goal_hit.sum()) != expected_hits or
                not episode.loc[episode.goal_hit, "option_start"].all()):
            raise ValueError(f"Unexpected episode-1 event alignment: {source}")
        verified[f"xy_{family}"] = episode[["x", "y"]].to_numpy(np.float32)
        verified[f"option_start_{family}"] = episode.option_start.to_numpy(bool)
        verified[f"goal_hit_{family}"] = episode.goal_hit.to_numpy(bool)
    np.savez_compressed(HIT_INPUTS, **verified)


def configure_style() -> str:
    font = Path(font_manager.findfont("Arial", fallback_to_default=False))
    if font.suffix.lower() not in {".ttf", ".otf"} or not font.is_file():
        raise RuntimeError("A verified scalable Arial font is required")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 12, "axes.titlesize": 12,
        "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "cosyne27-standalone-trajectory-candidates-v1",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return str(font)


def save(fig: plt.Figure, stem: str) -> list[str]:
    outputs = []
    for extension in ("svg", "pdf", "png"):
        path = HERE / f"{stem}.{extension}"
        metadata = ({"Date": "2026-10-14"} if extension == "svg" else
                    {"CreationDate": datetime(2026, 10, 14)} if extension == "pdf" else None)
        fig.savefig(path, dpi=300, metadata=metadata)
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                            encoding="utf-8")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def segment_coordinates(payload: np.lib.npyio.NpzFile, family: str, count: int) -> list[np.ndarray]:
    points = payload[f"xy_{family}"]
    offsets = payload[f"offsets_{family}"]
    return [points[int(offsets[index]):int(offsets[index + 1])]
            for index in range(count)]


def draw_path(ax: plt.Axes, paths: list[np.ndarray], color: str) -> None:
    for part in paths:
        ax.plot(part[:, 0], part[:, 1], color=color, lw=.85,
                alpha=.83, solid_capstyle="round", zorder=2)
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[2], BOUNDS[3])
    ax.set_aspect("equal")
    ax.set_xticks((100, 1000, 1900))
    ax.set_yticks((100, 1000, 1900))
    ax.tick_params(length=2, width=.6, labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(.7)
        spine.set_color("#606970")


def render_fragment_variants(payload: np.lib.npyio.NpzFile) -> list[str]:
    outputs = []
    for stem, count in VARIANTS.items():
        fig = plt.figure(figsize=(7.5, 3.15))
        fig.text(.075, .97, f"Trajectory fragments · {count} per architecture",
                 ha="left", va="top", fontsize=14, fontweight="bold")
        fig.text(.075, .89, "100k observations near the 100M checkpoint · fixed 100–2000 world extent",
                 ha="left", va="top", fontsize=12)
        for index, family in enumerate(FAMILIES):
            ax = fig.add_axes([.075 + index * .305, .18, .245, .59])
            draw_path(ax, segment_coordinates(payload, family, count), COLORS[family])
            ax.set_title(f"{family} · seed {dict(NG=99, UG=123, FG=17)[family]}", pad=6)
            ax.set_xlabel("x (DMLab units)", labelpad=0)
            if index == 0:
                ax.set_ylabel("y (DMLab units)", labelpad=0)
            else:
                ax.set_yticklabels([])
        outputs.extend(save(fig, stem))
        for family in FAMILIES:
            tile = plt.figure(figsize=(2.2, 2.2))
            ax = tile.add_axes([.03, .03, .94, .94])
            draw_path(ax, segment_coordinates(payload, family, count), COLORS[family])
            ax.set_axis_off()
            outputs.extend(save(tile, f"{family}_{stem}_tile"))
    return outputs


def render_verified_hits(payload: np.lib.npyio.NpzFile) -> list[str]:
    """Use older aligned event streams; never claim hits for the selected runs."""
    outputs = []
    fig = plt.figure(figsize=(7.5, 4.5))
    fig.text(.075, .98, "Verified option-target hits on archived paths",
             ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(.075, .91, "Earlier C05/C15 seed-99 probes · episode 1, 900 observations each",
             ha="left", va="top", fontsize=12)
    legend = [
        Line2D([], [], color="#59636A", lw=1.1, label="Observed path"),
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor="none", markeredgecolor="#505B63",
               label="Option start without hit"),
        Line2D([], [], marker="*", linestyle="none", markersize=10,
               markerfacecolor="#B31942", markeredgecolor="white",
               label="Verified option-target hit"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.075, .86),
               frameon=False, fontsize=12, ncol=3, columnspacing=.8)
    for index, family in enumerate(("C05", "C15")):
        ax = fig.add_axes([.105 + .465 * index, .22, .31, .48])
        xy = payload[f"xy_{family}"]
        start = payload[f"option_start_{family}"]
        hit = payload[f"goal_hit_{family}"]
        draw_path(ax, [xy], COLORS["UG" if family == "C05" else "FG"])
        candidate = start & ~hit
        ax.scatter(xy[candidate, 0], xy[candidate, 1], s=18, marker="o",
                   facecolor="none", edgecolor="#505B63", linewidth=.75, zorder=4)
        ax.scatter(xy[hit, 0], xy[hit, 1], s=64, marker="*",
                   facecolor="#B31942", edgecolor="white", linewidth=.4, zorder=5)
        ax.set_title(f"{family} seed 99 · {int(hit.sum())} hits", pad=5)
        ax.set_xlabel("x (DMLab units)", labelpad=0)
        if index == 0:
            ax.set_ylabel("y (DMLab units)", labelpad=0)
        else:
            ax.set_yticklabels([])
        tile = plt.figure(figsize=(2.2, 2.2))
        tile_ax = tile.add_axes([.03, .03, .94, .94])
        draw_path(tile_ax, [xy], COLORS["UG" if family == "C05" else "FG"])
        tile_ax.scatter(xy[candidate, 0], xy[candidate, 1], s=13, marker="o",
                        facecolor="none", edgecolor="#505B63", linewidth=.6, zorder=4)
        tile_ax.scatter(xy[hit, 0], xy[hit, 1], s=50, marker="*",
                        facecolor="#B31942", edgecolor="white", linewidth=.4, zorder=5)
        tile_ax.set_axis_off()
        outputs.extend(save(tile, f"archived_hits_{family}_seed99_tile"))
    fig.text(.075, .035,
             "Earlier runs: these verified hits do not belong to the selected Figure 2 paths.",
             ha="left", va="bottom", fontsize=12)
    outputs.extend(save(fig, "archived_verified_goal_hits"))
    return outputs


def render() -> None:
    font = configure_style()
    with np.load(INPUTS, allow_pickle=False) as fragments:
        outputs = render_fragment_variants(fragments)
        selection = {
            family: {
                "run_name": RUNS[family][0],
                "source_npz_path": (f"{ONLINE_ROOT}/{COHORTS[family]}/{RUNS[family][0]}/"
                                    "policy_00/snapshot_target_000100000000_actual_000100007936.npz"),
                "source_npz_sha256": RUNS[family][1],
                "total_segments": int(fragments[f"total_segments_{family}"]),
                "eligible_segments": int(fragments[f"eligible_segments_{family}"]),
                "selected_segment_ordinals": fragments[f"ordinals_{family}"].astype(int).tolist(),
                "selected_start_rows": fragments[f"start_rows_{family}"].astype(int).tolist(),
                "observations_per_variant": {
                    name: int(np.diff(fragments[f"offsets_{family}"][:count + 1]).sum())
                    for name, count in VARIANTS.items()
                },
            }
            for family in FAMILIES
        }
    with np.load(HIT_INPUTS, allow_pickle=False) as hits:
        outputs.extend(render_verified_hits(hits))
    manifest = {
        "schema": "cosyne27/standalone-trajectory-candidates/v1",
        "current_main_checked": "b5ef23351152755714dfd478c326b146329ce68b",
        "selection": "All canonical non-singleton contiguous fragments (segment_id changes or previous done splits a fragment); take the first 65 positions in a uniform random permutation without replacement, using independent fixed NumPy seeds 2027, 2028, 2029 for NG, UG, FG. The 9- and 33-fragment variants use the first 9 and 33 draws, so sets are nested. No selection by spatial coverage or hit count; no paths are joined across fragments.",
        "source_window_env_frames": [99207936, 100007936],
        "source_window_observations": 100000,
        "bounds_xy_dmlab": list(BOUNDS),
        "variants": {name: {"count_per_run": count} for name, count in VARIANTS.items()},
        "selected_inputs": selection,
        "selected_fragment_archive_sha256": sha256(INPUTS),
        "verified_hit_archive_sha256": sha256(HIT_INPUTS),
        "verified_hit_sources": {
            family: {"source_pose_events_sha256": item[1], "episode_num_traj": 1,
                     "source_pose_events_path": f"{HIT_ROOT}/{item[0].split('_pose_events.csv')[0]}/pose_events.csv",
                     "episode_observations": 900, "option_starts": item[2], "verified_hits": item[3]}
            for family, item in HIT_SOURCES.items()
        },
        "selected_run_hit_coordinates": "unavailable: the three selected online-spatial NPZs have no observation-aligned target-hit flags; the selected UG frozen pose has no hit flags either",
        "verified_hit_semantics": "HRLStateLayout.target_hit core flag, aligned with x/y at the same observation; option starts without this flag are not hits",
        "archive_note": "The verified-hit example uses older C05/C15 seed-99 probes and is not interchangeable with the selected NG/UG/FG Figure 2 trajectories.",
        "font_path": font,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "files": {filename: sha256(HERE / filename) for filename in outputs},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-from", type=Path,
                        help="Temporary directory containing verified source NPZ and CSV files")
    args = parser.parse_args()
    if args.prepare_from is not None:
        prepare(args.prepare_from)
    render()


if __name__ == "__main__":
    main()
