"""Render slide-sized IntrMotiv cards from the pinned COSYNE difference table.

Only presentation geometry changes here. The paired differences and study
protocols are defined by the source table and its adjacent manifest/README.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "07_abstracts/cosyne_2027_submission_draft/figure2_candidates/08_focused_canvas/plotted_seed_differences.csv"
SOURCE_MANIFEST = SOURCE.parent / "manifest.json"
TRAJECTORY_INPUT = ROOT / "07_abstracts/cosyne_2027_submission_draft/figure2_candidates/09_trajectory_thumbnails/selected_fragments.npz"
TRAJECTORY_MANIFEST = TRAJECTORY_INPUT.parent / "manifest.json"
PALETTE_SOURCE = ROOT / "07_abstracts/cosyne_2027_submission_draft/figure_palette_candidates/manifest.json"
PALETTE = json.loads(PALETTE_SOURCE.read_text())["palettes"]["priority_rgb_orange_violet"]["colors"]
COLORS = {"C01": PALETTE["ng"], "C05": PALETTE["ug"], "C15": PALETTE["fg"]}
LABELS = {"C01": "NG", "C05": "UG", "C15": "FG"}
INK = "#183043"
FONT = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")

CARDS = {
    "01_frozen_behavior": {
        "arm": "behavior",
        "heading": "Frozen weights · learned policy − random actions",
        "footer": "38 complete models · two paired 50k-decision repeats per model · common spatial bins and DG units",
        "metrics": [
            ("dg_information", "DG spatial information", "Difference (bits per decision)"),
            ("map_overlap", "DG map overlap", "Difference (mean pairwise cosine)"),
        ],
    },
    "02_encoder_credit": {
        "arm": "push",
        "heading": "Encoder push · temporal − mean-matched constant credit",
        "footer": "Five matched training seeds per architecture · identical 10,001-observation terminal replay",
        "metrics": [
            ("dg_information", "DG spatial information", "Difference (bits per decision)"),
            ("map_overlap", "DG map overlap", "Difference (mean pairwise cosine)"),
        ],
    },
    "03_graph_control_backup": {
        "arm": "push",
        "heading": "Encoder push · policy-sampled graph and command diagnostics",
        "footer": "Five matched seeds per goal architecture · separate policy-dependent measurements",
        "metrics": [
            ("graph_edges", "Reliable stored edges", "Difference (edge count)"),
            ("hit_rate", "Commanded detector hits", "Difference (hits per 1,000 eligible events)"),
        ],
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_editable_svg(fig, path: Path) -> None:
    fig.savefig(path, metadata={"Date": None})
    # Matplotlib writes spaces before path-data newlines. Normalize them so
    # generated vectors pass the same whitespace check as handwritten sources.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def load_values() -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    with SOURCE.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["arm"], row["metric"], row["family"])
            observation = (*key, row["seed"])
            if observation in seen:
                raise ValueError(f"Duplicate plotted observation: {observation}")
            seen.add(observation)
            value = float(row["delta"])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite plotted value: {observation}")
            values[key].append(value)
    expected = json.loads(SOURCE_MANIFEST.read_text())["sample_counts"]
    for card in CARDS.values():
        families = ["C05", "C15"] if card["metrics"][0][0] == "graph_edges" else ["C01", "C05", "C15"]
        for metric, _, _ in card["metrics"]:
            for family in families:
                key = (card["arm"], metric, family)
                if len(values[key]) != expected[f"{card['arm']}/{metric}/{family}"]:
                    raise ValueError(f"Focused-manifest sample count mismatch: {key}")
    return values


def draw_panel(ax, values, arm, metric, title, ylabel, families):
    arrays = [np.asarray(values[(arm, metric, family)]) for family in families]
    if any(len(a) == 0 for a in arrays):
        raise ValueError(f"Missing data for {arm}/{metric}")
    positions = np.arange(1, len(families) + 1)
    for pos, family, arr in zip(positions, families, arrays):
        color = COLORS[family]
        # Fixed offsets reveal coincident observations without adding stochastic jitter.
        offsets = np.linspace(-0.105, 0.105, len(arr))
        ax.scatter(pos + offsets, arr, s=78, facecolor=color, edgecolor=INK,
                   linewidth=0.7, alpha=0.82, zorder=4)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        ax.vlines(pos, arr.min(), arr.max(), color=INK, lw=1.4, zorder=2)
        ax.add_patch(plt.Rectangle((pos - 0.19, q1), 0.38, q3 - q1,
                                   facecolor="white", edgecolor=INK, lw=1.4, zorder=3))
        ax.hlines(median, pos - 0.19, pos + 0.19, color=INK, lw=2.2, zorder=5)
    outer = max(abs(v) for a in arrays for v in a)
    limit = outer * 1.22 if outer else 1.0
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#7B8991", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0.55, len(families) + 0.45)
    ax.set_xticks(positions, [f"{LABELS[f]}  n={len(a)}" for f, a in zip(families, arrays)])
    ax.set_ylabel(ylabel, fontsize=15, labelpad=8)
    ax.set_title(title, fontsize=19, fontweight="bold", loc="left", pad=15)
    ax.tick_params(axis="both", labelsize=14.5, length=4, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#6F7F88")
    ax.spines[["left", "bottom"]].set_linewidth(1)
    ax.yaxis.grid(True, color="#E2E8EB", lw=0.8)
    ax.set_axisbelow(True)


def render(name: str, spec: dict, values: dict) -> dict:
    families = ["C05", "C15"] if name.endswith("backup") else ["C01", "C05", "C15"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), dpi=180)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.095, right=0.98, top=0.79, bottom=0.215, wspace=0.32)
    fig.text(0.02, 0.965, spec["heading"], ha="left", va="top",
             fontsize=22, fontweight="bold", color=INK)
    fig.text(0.98, 0.965, "ONGOING · INTRMOTIV", ha="right", va="top",
             fontsize=14.5, color="#586674")
    for ax, (metric, title, ylabel) in zip(axes, spec["metrics"]):
        draw_panel(ax, values, spec["arm"], metric, title, ylabel, families)
    fig.text(0.02, 0.07, spec["footer"], ha="left", va="center", fontsize=14.5, color=INK)
    fig.text(0.02, 0.027, "Points = models/seeds · box = IQR and median · whiskers = observed range · dashed line = 0 · no significance test",
             ha="left", va="center", fontsize=14.5, color="#586674")
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    save_editable_svg(fig, svg)
    fig.savefig(png, dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return {
        "arm": spec["arm"],
        "metrics": [metric for metric, _, _ in spec["metrics"]],
        "families": families,
        "sample_counts": {f"{metric}/{family}": len(values[(spec["arm"], metric, family)])
                          for metric, _, _ in spec["metrics"] for family in families},
        "svg_sha256": sha256(svg),
        "png_sha256": sha256(png),
    }


def render_trajectories() -> dict:
    """Enlarge the archived random fragment sample without reselection."""
    name = "04_trajectory_examples"
    norm = Normalize(vmin=0, vmax=63)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.5), dpi=180)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.79, bottom=0.20, wspace=0.18)
    fig.text(0.02, 0.965, "Illustrative paths during online exploration", ha="left",
             va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.98, 0.965, "ONGOING · INTRMOTIV", ha="right", va="top",
             fontsize=14.5, color="#586674")
    selected = {}
    trajectory_source = json.loads(TRAJECTORY_MANIFEST.read_text())
    if sha256(TRAJECTORY_INPUT) != trajectory_source["selected_fragment_archive_sha256"]:
        raise ValueError("Trajectory archive differs from its source manifest")
    with np.load(TRAJECTORY_INPUT, allow_pickle=False) as archive:
        for ax, family in zip(axes, ["C01", "C05", "C15"]):
            label = LABELS[family]
            xy = archive[f"xy_{label}"]
            offsets = archive[f"offsets_{label}"]
            if len(offsets) != 66 or offsets[0] != 0 or offsets[-1] != len(xy):
                raise ValueError(f"Invalid 65-fragment archive for {label}")
            line_segments = []
            local_steps = []
            for start, stop in zip(offsets[:-1], offsets[1:]):
                path = xy[int(start):int(stop)]
                if len(path) > 64 or len(path) < 2:
                    raise ValueError(f"Invalid fragment length for {label}")
                line_segments.extend(np.stack([path[:-1], path[1:]], axis=1))
                local_steps.extend(np.arange(len(path) - 1) + 0.5)
            collection = LineCollection(line_segments, cmap=cmap, norm=norm,
                                        linewidth=1.25, alpha=0.82, rasterized=False)
            collection.set_array(np.asarray(local_steps))
            ax.add_collection(collection)
            ax.set_xlim(100, 2000)
            ax.set_ylim(100, 2000)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([500, 1000, 1500, 2000])
            ax.set_yticks([500, 1000, 1500, 2000])
            ax.tick_params(axis="both", labelsize=14.5, colors=INK)
            ax.set_title(f"{label} · seed {str(archive[f'run_name_{label}']).rsplit('S', 1)[-1]}",
                         color=COLORS[family], fontsize=19, fontweight="bold", pad=12)
            for spine in ax.spines.values():
                spine.set_color("#6F7F88")
            selected[label] = {
                "run_name": str(archive[f"run_name_{label}"]),
                "source_npz_sha256": str(archive[f"source_hash_{label}"]),
                "fragments": len(offsets) - 1,
                "observations": len(xy),
            }
            if (selected[label]["run_name"] != trajectory_source["selected_inputs"][label]["run_name"] or
                    len(xy) != trajectory_source["selected_inputs"][label]["observations_per_variant"]["65_fragments"]):
                raise ValueError(f"Trajectory selection differs from its source manifest: {label}")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=axes, orientation="horizontal", fraction=0.042,
                            pad=0.09, aspect=60)
    colorbar.set_label("Decision index within each fragment (resets at every boundary)",
                       fontsize=14.5, color=INK)
    colorbar.ax.tick_params(labelsize=13.5, colors=INK)
    fig.text(0.02, 0.033,
             "65 randomly selected complete fragments per run · same 100–2000 world bounds · selected seeds are examples, not group estimates",
             ha="left", va="center", fontsize=14.0, color="#586674")
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    save_editable_svg(fig, svg)
    fig.savefig(png, dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return {
        "input_npz": str(TRAJECTORY_INPUT.relative_to(ROOT)),
        "input_npz_sha256": sha256(TRAJECTORY_INPUT),
        "source_manifest": str(TRAJECTORY_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(TRAJECTORY_MANIFEST),
        "selection": selected,
        "world_bounds_xy": [100, 2000, 100, 2000],
        "local_step_color_range": [0, 63],
        "svg_sha256": sha256(svg),
        "png_sha256": sha256(png),
    }


def main() -> None:
    if not FONT.is_file():
        raise FileNotFoundError(f"Required scalable font absent: {FONT}")
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
        "svg.fonttype": "none",  # Editable vector text.
        "svg.hashsalt": "sab2026-intrmotiv-assets-v1",
        "text.color": INK,
        "axes.labelcolor": INK,
    })
    values = load_values()
    records = {name: render(name, spec, values) for name, spec in CARDS.items()}
    records["04_trajectory_examples"] = render_trajectories()
    manifest = {
        "schema": "intrmotiv/sab2026-slide-assets/v1",
        "source_table": str(SOURCE.relative_to(ROOT)),
        "source_table_sha256": sha256(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "palette_manifest": str(PALETTE_SOURCE.relative_to(ROOT)),
        "palette_manifest_sha256": sha256(PALETTE_SOURCE),
        "font": str(FONT),
        "matplotlib": matplotlib.__version__,
        "canvas_inches": [12.2, 5.5],
        "preview_dpi": 180,
        "cards": records,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
