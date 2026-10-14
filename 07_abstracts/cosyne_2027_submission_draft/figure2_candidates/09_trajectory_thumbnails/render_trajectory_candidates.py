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
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
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
    "FG": ("COS27_C15_TEMPORAL_S123", "38f1504d0b918c2d3c5261cee4bac482b89a27c4d99cbe77d355869d842ad9a6"),
}
COHORTS = {
    "NG": "cosyne27_c01_temporal_s3_20261010",
    "UG": "cosyne27_c05_pushpull_s3_20261010",
    "FG": "cosyne27_c15_encoder_s3_20261010",
}
FG_CANDIDATES = {
    8: ("cosyne27_c15_encoder_s3_20261010", "66de1d84bd71075c549ab4e6683b4b9cc90aed653e7c936b8c31a07756cf35e3"),
    17: ("cosyne27_c15_encoder_s2_20261010", "bf977d531dfa1486af0442d10af69d8da0636b481d45ea7c5bdbf532279499b6"),
    41: ("cosyne27_c15_encoder_s2_20261010", "714852a4aa7e3a6485e9e1c0819e7be90abce4cbc1c977cee9b344c9d2d437e0"),
    99: ("cosyne27_c15_encoder_s3_20261010", "1c3458c6682da7c4e0fe93e7c9f5d4c20ef08f9bf5f2402e8d3174ea53c2bfc0"),
    123: ("cosyne27_c15_encoder_s3_20261010", "38f1504d0b918c2d3c5261cee4bac482b89a27c4d99cbe77d355869d842ad9a6"),
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
VARIANTS = {"65_fragments": ARCHIVE_COUNT}
BOUNDS = (100.0, 2000.0, 100.0, 2000.0)
STEP_NORM = Normalize(vmin=0, vmax=99999)
PANEL_PT = 38


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


def occupancy_metrics(xy: np.ndarray) -> tuple[int, float, float]:
    """N_eff/visited for equal 19-by-19 world bins; larger means more even use."""
    counts = np.histogram2d(xy[:, 0], xy[:, 1], bins=19,
                            range=((100, 2000), (100, 2000)))[0].ravel()
    visited = counts[counts > 0]
    probabilities = visited / visited.sum()
    effective_bins = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return len(visited), effective_bins, effective_bins / len(visited)


def read_snapshot(source: Path, expected_hash: str, run_name: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError(f"Unexpected snapshot contract: {source}")
    return pose, segment_id, dones


def select_slices(segment_id: np.ndarray, dones: np.ndarray, rng_seed: int
                  ) -> tuple[list[slice], list[tuple[int, slice]], list[tuple[int, slice]]]:
    all_slices = contiguous_slices(segment_id, dones)
    eligible = [(ordinal, part) for ordinal, part in enumerate(all_slices)
                if part.stop - part.start > 1]
    # Independent fixed permutations avoid aliasing with parallel actor order.
    chosen = np.random.default_rng(rng_seed).permutation(len(eligible))[:ARCHIVE_COUNT]
    if len(np.unique(chosen)) != ARCHIVE_COUNT:
        raise ValueError("Repeated selected fragment")
    return all_slices, eligible, [eligible[index] for index in chosen]


def prepare(raw_root: Path) -> None:
    """Hash-check full sources, then archive only the selected plot coordinates."""
    fragments: dict[str, np.ndarray] = {}
    screening = []
    for seed, (_, source_hash) in FG_CANDIDATES.items():
        source = raw_root / "C15_candidates" / f"S{seed}.npz"
        pose, segment_id, dones = read_snapshot(
            source, source_hash, f"COS27_C15_TEMPORAL_S{seed}")
        _, _, picked = select_slices(segment_id, dones, 2029)
        sampled_xy = np.concatenate([pose[part, :2] for _, part in picked])
        screening.append((seed, occupancy_metrics(pose[:, :2]),
                          occupancy_metrics(sampled_xy)))
    fragments["fg_screen_seeds"] = np.array([row[0] for row in screening], dtype=np.int32)
    for name, position in (("full", 1), ("sample", 2)):
        fragments[f"fg_screen_{name}_visited"] = np.array(
            [row[position][0] for row in screening], dtype=np.int32)
        fragments[f"fg_screen_{name}_effective"] = np.array(
            [row[position][1] for row in screening], dtype=np.float64)
        fragments[f"fg_screen_{name}_uniformity"] = np.array(
            [row[position][2] for row in screening], dtype=np.float64)
    for family_index, (family, (run_name, expected_hash)) in enumerate(RUNS.items()):
        source = (raw_root / "C15_candidates" / "S123.npz" if family == "FG"
                  else raw_root / family / "online_spatial.npz")
        pose, segment_id, dones = read_snapshot(source, expected_hash, run_name)
        all_slices, eligible, picked = select_slices(segment_id, dones, 2027 + family_index)
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


def segment_coordinates(payload: np.lib.npyio.NpzFile, family: str, count: int
                        ) -> list[tuple[np.ndarray, np.ndarray]]:
    points = payload[f"xy_{family}"]
    offsets = payload[f"offsets_{family}"]
    starts = payload[f"start_rows_{family}"]
    return [(points[int(offsets[index]):int(offsets[index + 1])],
             int(starts[index]) + np.arange(int(offsets[index + 1] - offsets[index])))
            for index in range(count)]


def draw_gradient_path(ax: plt.Axes,
                       paths: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Color every within-fragment decision step by its original 100k row."""
    segments = np.concatenate([
        np.stack((xy[:-1], xy[1:]), axis=1) for xy, _ in paths])
    steps = np.concatenate([rows[:-1] + .5 for _, rows in paths])
    if len(segments) != len(steps) or steps.min() < 0 or steps.max() > 99999:
        raise ValueError("Invalid original-window step indices")
    lines = LineCollection(segments, cmap="viridis", norm=STEP_NORM,
                           linewidths=.55, alpha=1.0, capstyle="round")
    lines.set_array(steps)
    ax.add_collection(lines)
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[2], BOUNDS[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((100, 100), 1900, 1900, fill=False,
                               edgecolor="#606970", linewidth=.4))


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
        width_pt, height_pt = 270, 140
        fig = plt.figure(figsize=(width_pt / 72, height_pt / 72))
        fig.text(10 / width_pt, 136 / height_pt,
                 f"Trajectory fragments · {count} per architecture",
                 ha="left", va="top", fontsize=12, fontweight="bold")
        for index, family in enumerate(FAMILIES):
            x_pt = 26 + index * 90
            ax = fig.add_axes([x_pt / width_pt, 72 / height_pt,
                               PANEL_PT / width_pt, PANEL_PT / height_pt])
            draw_gradient_path(ax, segment_coordinates(payload, family, count))
            fig.text((x_pt + PANEL_PT / 2) / width_pt, 117 / height_pt,
                     f"{family} · seed {dict(NG=99, UG=123, FG=123)[family]}",
                     ha="center", va="center", fontsize=12)
        color_axis = fig.add_axes([50 / width_pt, 39 / height_pt,
                                   170 / width_pt, 6 / height_pt])
        colorbar = fig.colorbar(ScalarMappable(norm=STEP_NORM, cmap="viridis"),
                                cax=color_axis, orientation="horizontal")
        colorbar.set_ticks([0, 50000, 99999])
        colorbar.set_ticklabels(["0", "50,000", "99,999"])
        colorbar.ax.tick_params(labelsize=12, length=2, width=.5, pad=1)
        colorbar.outline.set_linewidth(.4)
        fig.text(227 / width_pt, 42 / height_pt, "step", ha="left", va="center",
                 fontsize=12)
        outputs.extend(save(fig, stem))
        for family in FAMILIES:
            tile = plt.figure(figsize=(PANEL_PT / 72, PANEL_PT / 72))
            ax = tile.add_axes([0, 0, 1, 1])
            draw_gradient_path(ax, segment_coordinates(payload, family, count))
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
        hit_count = int(hit.sum())
        ax.set_title(f"{family} seed 99 · {hit_count} {'hit' if hit_count == 1 else 'hits'}", pad=5)
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
        fg_screen = {
            str(int(seed)): {
                "run_name": f"COS27_C15_TEMPORAL_S{int(seed)}",
                "source_npz_path": (
                    f"{ONLINE_ROOT}/{FG_CANDIDATES[int(seed)][0]}/"
                    f"COS27_C15_TEMPORAL_S{int(seed)}/policy_00/"
                    "snapshot_target_000100000000_actual_000100007936.npz"),
                "source_npz_sha256": FG_CANDIDATES[int(seed)][1],
                "full_window": {
                    "visited_bins": int(fragments["fg_screen_full_visited"][index]),
                    "effective_bins": float(fragments["fg_screen_full_effective"][index]),
                    "effective_fraction_of_visited": float(
                        fragments["fg_screen_full_uniformity"][index]),
                },
                "sampled_65_fragments": {
                    "visited_bins": int(fragments["fg_screen_sample_visited"][index]),
                    "effective_bins": float(fragments["fg_screen_sample_effective"][index]),
                    "effective_fraction_of_visited": float(
                        fragments["fg_screen_sample_uniformity"][index]),
                },
            }
            for index, seed in enumerate(fragments["fg_screen_seeds"])
        }
    with np.load(HIT_INPUTS, allow_pickle=False) as hits:
        outputs.extend(render_verified_hits(hits))
    manifest = {
        "schema": "cosyne27/standalone-trajectory-candidates/v2",
        "current_main_checked": "a49c677277e9812b19389afb3615b29120b01142",
        "selection": "All canonical non-singleton contiguous fragments (segment_id changes or previous done splits a fragment); take the first 65 positions in a uniform random permutation without replacement, using independent fixed NumPy seeds 2027, 2028, 2029 for NG, UG, FG. No fragment selection by spatial coverage or hit count; no paths are joined across fragments.",
        "fg_seed_choice": "Seed 123 selected at the author's direction. Its full-window occupancy is more even than the prior seed 17, although seeds 8 and 99 rank higher by this metric. All five comparable C15 temporal-credit snapshots were screened; path efficiency was not used.",
        "fg_seed_screen_method": "For each of five complete 100000-observation C15 snapshots, histogram x/y into fixed 19x19 bins across [100,2000] on each axis; visited bins have positive count; effective bins = exp(-sum(p_i log p_i)); uniformity = effective bins / visited bins. The sampled-65 score uses the exact fixed-seed fragment set. Larger uniformity is more even occupancy.",
        "fg_seed_screen": fg_screen,
        "source_window_env_frames": [99207936, 100007936],
        "source_window_observations": 100000,
        "step_color": "Viridis sequential color, with shared normalization 0 to 99999 across NG/UG/FG. Each within-fragment line segment uses the midpoint of its two original observation row indices in the 100000-observation source window; later rows are brighter. The colorbar label is step. Sampling order does not set the color.",
        "bounds_xy_dmlab": list(BOUNDS),
        "variants": {name: {"count_per_run": count, "panel_square_points": PANEL_PT,
                            "sheet_points": [270, 140], "tile_points": [PANEL_PT, PANEL_PT]}
                     for name, count in VARIANTS.items()},
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
