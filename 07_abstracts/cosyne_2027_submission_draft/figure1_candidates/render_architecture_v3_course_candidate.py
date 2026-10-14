"""Render a separate Figure 1 candidate from architecture v3 and audited data.

The three DG examples are selected for distinct maximum-response locations
within the C15 temporal, seed-123 terminal probe. Their source atlas uses
per-unit visited-bin min–max color scaling, not a shared amplitude scale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, NullLocator
import numpy as np
import pandas as pd
from PIL import Image

from render_architecture_with_fields import DRAFT, HERE, style
from render_submission_panels import COLOR, SOURCES


STEM = "04_architecture_v3_dg_reward_fields"
ROOT = DRAFT.parents[1]
ARCHITECTURE = DRAFT / "architecture_v3.svg"
ATLAS = HERE / "full_seed_atlases_minmax/c15_temporal_s123.png"
UNIT_METRICS = HERE / "full_seed_atlases/terminal_unit_metrics.csv"
SEED_METRICS = HERE / "full_seed_atlases/terminal_seed_metrics.csv"
REWARD = ROOT / "06_experiments/results/cosyne_2027_intrinsic_reward_course/arm_summaries.csv"
UNITS = (4, 1, 11)
PEAK_CELLS_XY = {4: (2, 4), 1: (6, 15), 11: (17, 3)}
FAMILIES = (("C01", "inapplicable", "NG"),
            ("C05", "temporal", "UG"),
            ("C15", "temporal", "FG"))
SIZE_IN = (7.5, 1.9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_architecture() -> Image.Image:
    """Render the unchanged authored v3 artwork at >600 dpi in the figure."""
    with tempfile.TemporaryDirectory(prefix="figure1_v3_") as temporary:
        image_path = Path(temporary) / "architecture.png"
        subprocess.run(
            ["inkscape", str(ARCHITECTURE), f"--export-filename={image_path}",
             "--export-width=3500"], check=True, capture_output=True, text=True)
        with Image.open(image_path) as source:
            return source.convert("RGB")


def course(ax: plt.Axes, table: pd.DataFrame, metric: str, *, log: bool) -> None:
    for family, worker, label in FAMILIES:
        rows = table[(table.family == family) &
                     (table.encoder_credit == "temporal") &
                     (table.worker_bonus == worker) &
                     (table.metric == metric)].sort_values("env_steps")
        if len(rows) != 99 or not rows.n.eq(5).all():
            raise ValueError(f"Incomplete {metric} course: {family}")
        x = rows.env_steps.to_numpy() / 1e6
        mean = rows["mean"].to_numpy()
        sem = rows["sem"].to_numpy()
        if log and np.any(mean - sem <= 0):
            raise ValueError(f"Log course has nonpositive uncertainty bound: {family}")
        ax.plot(x, mean, color=COLOR[family], lw=1.15, label=label)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=COLOR[family], alpha=0.17, linewidth=0)
    ax.set_xlim(0, 100)
    ax.set_xticks((0, 100))
    ax.tick_params(axis="both", labelsize=8, length=2, width=0.6, pad=1)
    ax.grid(axis="y", color="#E6E8E9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 5)
        ax.set_yticks((0.01, 1))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("reward/step", fontsize=8)
    else:
        ax.set_ylim(0, 0.22)
        ax.set_yticks((0, 0.2))
        ax.set_ylabel("bits/step", fontsize=8)


def field_tiles() -> dict[int, Image.Image]:
    with Image.open(ATLAS) as source:
        if source.size != (1280, 1414):
            raise ValueError(f"Unexpected source atlas size: {source.size}")
        tiles = {}
        for unit in UNITS:
            row, column = divmod(unit, 4)
            left, top = 65 + 320 * column, 144 + 300 * row
            tile = source.crop((left, top, left + 190, top + 190)).convert("RGB")
            brightest = np.all(np.asarray(tile) == (253, 231, 37), axis=2)
            peak_y, peak_x = np.where(brightest)
            if (len(peak_x) != 100 or
                    set(zip(peak_x // 10, peak_y // 10)) != {PEAK_CELLS_XY[unit]}):
                raise ValueError(f"Peak cell changed for DG {unit}")
            tiles[unit] = tile
        return tiles


def verify_selection() -> dict[int, dict]:
    seed = pd.read_csv(SEED_METRICS)
    rows = seed[(seed.condition == "COS27_C15_TEMPORAL") & (seed.seed == 123)]
    if len(rows) != 1 or int(rows.iloc[0].visited_cells) != 307:
        raise ValueError("Unexpected C15 seed-123 probe or occupancy")
    units = pd.read_csv(UNIT_METRICS)
    rows = units[(units.condition == "COS27_C15_TEMPORAL") &
                 (units.seed == 123) & units.unit.isin(UNITS)]
    if set(rows.unit) != set(UNITS) or not rows.field_eligible.all():
        raise ValueError("Selected example units are missing or ineligible")
    result = {}
    for _, row in rows.iterrows():
        result[int(row.unit)] = {
            "spatial_information_bits": float(row.spatial_information_bits),
            "active_fraction": float(row.active_fraction),
            "mono_score": float(row.mono_score),
            "strict_monofield": bool(row.mono_field),
            "peak_cell_xy": list(PEAK_CELLS_XY[int(row.unit)]),
        }
    return result


def main() -> None:
    font = style()
    metrics = verify_selection()
    architecture = original_architecture()
    online = pd.read_csv(SOURCES["online"])
    reward = pd.read_csv(REWARD)
    tiles = field_tiles()

    fig = plt.figure(figsize=SIZE_IN, facecolor="white")
    # Keep the original 5.41-inch artwork at native size. The remaining
    # 2.09 inches hold two narrow course plots above three unlabeled maps.
    ax_art = fig.add_axes([0, 0, 5.40856 / SIZE_IN[0], 1])
    ax_art.imshow(architecture, interpolation="lanczos")
    ax_art.axis("off")
    # The v3 title touches the top-left edge; place A on the image corner.
    fig.text(0.040 / SIZE_IN[0], 1.16 / 1.9, "A", va="top",
             fontsize=9.5, fontweight="bold", color="#20272C",
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

    fig.text(5.52 / SIZE_IN[0], 1.84 / 1.9, "B  DG info", va="top",
             fontsize=9.5, fontweight="bold", color="#20272C")
    ax_info = fig.add_axes([5.70 / SIZE_IN[0], 0.99 / 1.9,
                            0.60 / SIZE_IN[0], 0.47 / 1.9])
    course(ax_info, online, "spatial_information", log=False)
    for x, family, label in ((5.64, "C01", "NG"), (5.97, "C05", "UG"),
                             (6.30, "C15", "FG")):
        fig.text(x / SIZE_IN[0], 1.62 / 1.9, label, va="top", fontsize=8,
                 fontweight="bold", color=COLOR[family])

    fig.text(6.67 / SIZE_IN[0], 1.84 / 1.9,
             "C  Intrinsic\nreward (log)", va="top", fontsize=8.8,
             linespacing=0.9, fontweight="bold", color="#20272C")
    ax_reward = fig.add_axes([6.88 / SIZE_IN[0], 0.99 / 1.9,
                              0.56 / SIZE_IN[0], 0.47 / 1.9])
    course(ax_reward, reward, "intrinsic_mean", log=True)
    fig.text(6.49 / SIZE_IN[0], 0.81 / 1.9, "Training frames (M)",
             ha="center", va="center", fontsize=8, color="#20272C")

    fig.text(5.52 / SIZE_IN[0], 0.73 / 1.9, "D  Example DG place field", va="top",
             fontsize=9.5, fontweight="bold", color="#20272C")
    for index, unit in enumerate(UNITS):
        x = 5.54 + 0.65 * index
        ax = fig.add_axes([x / SIZE_IN[0], 0.005 / 1.9,
                           0.54 / SIZE_IN[0], 0.54 / 1.9])
        ax.imshow(tiles[unit], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_edgecolor("#77858A")

    for suffix in ("pdf", "svg", "png"):
        destination = HERE / f"{STEM}.{suffix}"
        metadata = ({"Date": None} if suffix == "svg" else
                    {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
        fig.savefig(destination, dpi=600 if suffix != "png" else 300,
                    metadata=metadata)
        if suffix == "svg":
            destination.write_text("\n".join(line.rstrip() for line in
                                            destination.read_text().splitlines()) + "\n")
    plt.close(fig)

    (HERE / f"{STEM}_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/figure1-v3-course-candidate/v1",
        "figure": STEM,
        "size_inches": list(SIZE_IN),
        "font": font,
        "architecture": "unchanged v3 SVG rendered at 3500 pixels wide",
        "panel_a_label": "small white badge on the upper-left observation image",
        "courses": "arithmetic five-seed mean ± SEM, same verified runs, 1–99M frames",
        "reward_metric": "intrmotiv/reward/intrinsic_mean over 2048 valid transitions per update, zeros included",
        "reward_log_scale": "log10 display axis; arithmetic means and additive SEM unchanged",
        "fields": {"condition": "COS27_C15_TEMPORAL", "seed": 123,
                   "checkpoint_frames": 100007936, "probe_decisions": 10001,
                   "visited_bins": 307, "total_bins": 361,
                   "source_scale": "per-unit visited-bin min–max; gray unvisited",
                   "units": metrics},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in (ARCHITECTURE, SOURCES["online"], REWARD,
                                       ATLAS, UNIT_METRICS, SEED_METRICS)},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
