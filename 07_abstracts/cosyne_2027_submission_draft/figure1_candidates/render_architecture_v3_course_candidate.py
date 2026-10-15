"""Render a separate Figure 1 candidate from architecture v3 and audited data.

The three DG examples are selected for distinct maximum-response locations
within the C15 temporal, seed-123 terminal probe. Their source atlas uses
per-unit visited-bin min–max color scaling, not a shared amplitude scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, NullLocator
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from render_architecture_with_fields import DRAFT, HERE, style
from render_submission_panels import COLOR, SOURCES


STEM = "04_architecture_v3_dg_reward_fields"
RELATIVE_STEM = "05_architecture_v3_relative_reward"
GOAL_GATE_STEM = "06_architecture_v3_goal_gate_header"
ROOT = DRAFT.parents[1]
ARCHITECTURE = DRAFT / "architecture_v3.svg"
GOAL_GATE_ICON = HERE / "goal_policy_icon_options/simple/05_side_gate.svg"
ATLAS = HERE / "full_seed_atlases_minmax/c15_temporal_s123.png"
UNIT_METRICS = HERE / "full_seed_atlases/terminal_unit_metrics.csv"
SEED_METRICS = HERE / "full_seed_atlases/terminal_seed_metrics.csv"
REWARD = ROOT / "06_experiments/results/cosyne_2027_intrinsic_reward_course/arm_summaries.csv"
REWARD_SEEDS = REWARD.with_name("interpolated_by_seed.csv")
ONLINE_SEEDS = ROOT / "06_experiments/results/cosyne_2027_online_wandb/interpolated_by_seed.csv"
UNITS = (4, 1, 11)
PEAK_CELLS_XY = {4: (2, 4), 1: (6, 15), 11: (17, 3)}
FAMILIES = (("C01", "inapplicable", "NG"),
            ("C05", "temporal", "UG"),
            ("C15", "temporal", "FG"))
SIZE_IN = (7.5, 1.9)
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ARCHITECTURE_WIDTH_IN = 5.4085569
VIS_TITLE_SHIFT_MM = 3.8
SMOOTHING_SIGMA_M_FRAMES = 1.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inline_architecture(composite_svg: Path, *, harmonize_labels: bool = False) -> None:
    """Replace Matplotlib's placeholder with v3's native SVG elements.

    The source SVG uses IDs for markers, glyphs, and filters. Its IDs are
    disjoint from Matplotlib's IDs, so preserve them and their references.
    """
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    composite = ET.parse(composite_svg)
    root = composite.getroot()
    target = next((node for node in root.iter() if node.get("id") == "axes_1"), None)
    if target is None:
        raise ValueError("Architecture placeholder axes missing")
    for child in list(target):
        target.remove(child)

    source = ET.parse(ARCHITECTURE).getroot()
    if source.get("viewBox") != "0 0 137.37734 48.268308":
        raise ValueError("Architecture v3 viewBox changed")
    nested = ET.Element(f"{{{SVG_NS}}}svg", {
        "id": "architecture_v3_vector",
        "x": "0", "y": "0",
        "width": f"{ARCHITECTURE_WIDTH_IN * 72:.6f}",
        "height": f"{SIZE_IN[1] * 72:.6f}",
        "viewBox": source.get("viewBox"),
        "preserveAspectRatio": "xMinYMin meet",
    })
    for child in source:
        if child.tag.rsplit("}", 1)[-1] != "namedview":
            nested.append(child)

    title = next(node for node in nested.iter()
                 if node.tag == f"{{{SVG_NS}}}text"
                 and "Vis. Encoder" in "".join(node.itertext()))
    # Shift only the heading, leaving the modules and all arrow geometry at
    # their authored size. The text and its two ancestors each have an x scale.
    source_units_per_mm = 1 / (0.26458333 * 0.28110339 * 5.1396257)
    title.set("x", f"{float(title.get('x')) + VIS_TITLE_SHIFT_MM * source_units_per_mm:.6f}")
    for node in nested.iter():
        if node.tag == f"{{{SVG_NS}}}text" and "temopral" in "".join(node.itertext()):
            for descendant in node.iter():
                if descendant.text:
                    descendant.text = descendant.text.replace("temopral", "temporal")
    if harmonize_labels:
        harmonize_architecture_labels(nested)

    target.append(nested)
    all_ids = [node.get("id") for node in root.iter() if node.get("id")]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate SVG IDs after architecture import")
    composite.write(composite_svg, encoding="utf-8", xml_declaration=True)
    composite_svg.write_text("\n".join(line.rstrip() for line in
                                       composite_svg.read_text().splitlines()) + "\n")


def harmonize_architecture_labels(nested: ET.Element) -> None:
    """Use Figure 2's TS/Policy vocabulary in the current candidate only."""
    readout_labels = [node for node in nested.iter()
                      if node.tag == f"{{{SVG_NS}}}text"
                      and "Readout" in "".join(node.itertext())]
    if len(readout_labels) != 1:
        raise ValueError("Expected one architecture Readout heading")
    for descendant in readout_labels[0].iter():
        if descendant.text:
            descendant.text = descendant.text.replace("Readout", "Policy")
        if descendant.get("x") is not None:
            # Center the shorter replacement over the same readout wedge.
            descendant.set("x", f"{float(descendant.get('x')) + 6.3:.6f}")

    # Three original delta glyphs label the same temporal-separation motif:
    # the central example, push, and pull. Preserve their neighboring arrows
    # and DG event circles while replacing only each glyph with editable text.
    glyph_href = f"{{{XLINK_NS}}}href"
    parent_by_child = {child: parent for parent in nested.iter() for child in parent}
    delta_glyphs = [node for node in nested.iter()
                    if node.tag == f"{{{SVG_NS}}}use"
                    and node.get(glyph_href) == "#ldg_STIXGeneral-Regular-394"]
    if len(delta_glyphs) != 3:
        raise ValueError("Expected three architecture temporal-separation glyphs")
    for index, glyph in enumerate(delta_glyphs, start=1):
        transform = glyph.get("transform", "")
        match = re.fullmatch(r"matrix\(([^)]+)\)", transform)
        if match is None:
            raise ValueError(f"Unexpected delta glyph transform: {transform}")
        values = [float(value) for value in re.split(r"[,\s]+", match.group(1))]
        if (len(values) != 6 or
                not np.allclose(values[:4], [0.20044281, 0, 0, -0.20044281])):
            raise ValueError(f"Unexpected delta glyph scale: {transform}")
        replacement = ET.Element(f"{{{SVG_NS}}}text", {
            "id": f"temporal_separation_ts_{index}",
            "x": f"{values[4] - 2:.6f}", "y": f"{values[5]:.6f}",
            "font-family": "Helvetica, Arial, sans-serif",
            "font-size": "14", "fill": "#252525",
        })
        replacement.text = "TS"
        parent = parent_by_child[glyph]
        parent.insert(list(parent).index(glyph), replacement)
        parent.remove(glyph)


def inline_goal_gate(composite_svg: Path) -> None:
    """Place the editable side-gate artwork in B's top legend row."""
    tree = ET.parse(composite_svg)
    root = tree.getroot()
    source = ET.parse(GOAL_GATE_ICON).getroot()
    if source.get("viewBox") != "0 0 53 20":
        raise ValueError("Goal-gate icon geometry changed")
    icon_labels = {node.text: node for node in source.iter()
                   if node.tag == f"{{{SVG_NS}}}text"
                   and node.text in ("Δ", "π")}
    if set(icon_labels) != {"Δ", "π"}:
        raise ValueError("Goal-gate icon labels changed")
    icon_labels["Δ"].text = "TS"
    icon_labels["Δ"].set("font-size", "4.3")
    icon_labels["π"].text = "Pol."
    icon_labels["π"].set("font-size", "4.2")
    width_in = 1.23
    top_in = 1.84
    nested = ET.SubElement(root, f"{{{SVG_NS}}}svg", {
        "id": "goal_gate_header_vector",
        "x": f"{5.88 * 72:.6f}",
        "y": f"{(SIZE_IN[1] - top_in) * 72:.6f}",
        "width": f"{width_in * 72:.6f}",
        "height": f"{width_in * 20 / 53 * 72:.6f}",
        "viewBox": source.get("viewBox"),
    })
    nested.extend(list(source))
    ids = [node.get("id") for node in root.iter() if node.get("id")]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate SVG IDs after goal-gate import")
    tree.write(composite_svg, encoding="utf-8", xml_declaration=True)


def course(ax: plt.Axes, table: pd.DataFrame, metric: str, *, log: bool,
           relative_to_mean: bool = False, thin_lines: bool = False,
           seed_table: pd.DataFrame | None = None,
           compact_ylabel: bool = False) -> dict[str, float]:
    """Plot means and SEM, optionally scaled by each condition's own mean.

    The denominator is the mean of the 99 plotted five-seed means. Scaling
    the mean and SEM by the same positive constant preserves the within-
    condition trajectory. When requested, smoothing happens separately for
    each seed before re-aggregating the mean and SEM.
    """
    normalizers = {}
    if relative_to_mean:
        ax.axhline(1, color="#70777B", lw=0.7, ls=(0, (2, 2)), zorder=0)
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
        denominator = float(np.mean(mean)) if relative_to_mean else 1.0
        if seed_table is not None:
            if not np.allclose(np.diff(x), 1, rtol=0, atol=1e-9):
                raise ValueError(f"Smoothing requires a one-million-frame grid: {family}")
            seeds = seed_table[(seed_table.family == family) &
                               (seed_table.metric == metric)]
            if "encoder_credit" in seeds:
                seeds = seeds[(seeds.encoder_credit == "temporal") &
                              (seeds.worker_bonus == worker)]
            curves = seeds.pivot(index="env_steps", columns="seed", values="value")
            if (curves.shape != (99, 5) or
                    not np.array_equal(curves.index.to_numpy(), rows.env_steps.to_numpy()) or
                    not np.all(np.isfinite(curves.to_numpy()))):
                raise ValueError(f"Incomplete seed curves: {family}, {metric}")
            if (not np.allclose(curves.mean(axis=1), mean, rtol=0, atol=1e-8) or
                    not np.allclose(curves.sem(axis=1), sem, rtol=0, atol=1e-8)):
                raise ValueError(f"Seed curves disagree with archived summary: {family}, {metric}")
            smoothed = gaussian_filter1d(curves.to_numpy(),
                                         sigma=SMOOTHING_SIGMA_M_FRAMES,
                                         axis=0, mode="nearest")
            mean = smoothed.mean(axis=1)
            sem = smoothed.std(axis=1, ddof=1) / np.sqrt(smoothed.shape[1])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"Invalid reward normalization: {family}")
        normalizers[family] = denominator
        mean = mean / denominator
        sem = sem / denominator
        if log and np.any(mean - sem <= 0):
            raise ValueError(f"Log course has nonpositive uncertainty bound: {family}")
        ax.plot(x, mean, color=COLOR[family], lw=0.45 if thin_lines else 1.15,
                label=label)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=COLOR[family], alpha=0.10 if thin_lines else 0.17,
                        linewidth=0)
    ax.set_xlim(0, 100)
    ax.set_xticks((0, 100))
    ax.tick_params(axis="both", labelsize=8, length=2, width=0.6, pad=1)
    ax.grid(axis="y", color="#E6E8E9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if relative_to_mean:
        ax.set_ylim(0, 2)
        ax.set_yticks((0, 1, 2))
        ax.set_ylabel("Reward\n/ mean" if compact_ylabel else
                      "Intrinsic reward\n/ own mean", fontsize=8)
    elif log:
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
    return normalizers


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
            # The archived atlas draws increasing world y downward in its
            # raster. The architecture maze and trajectory panels show world
            # y upward, so align the field presentation here.
            tiles[unit] = tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
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


def main(*, relative_reward: bool = False,
         goal_gate_header: bool = False,
         output_stem: str | None = None) -> None:
    if goal_gate_header and not relative_reward:
        raise ValueError("The goal-gate layout requires the relative-reward courses")
    stem = output_stem or (GOAL_GATE_STEM if goal_gate_header else
                           RELATIVE_STEM if relative_reward else STEM)
    if output_stem is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_stem):
        raise ValueError("--output-stem must be a simple filename stem")
    if (stem == GOAL_GATE_STEM or goal_gate_header) and (HERE / f"{stem}.svg").exists():
        raise FileExistsError(
            f"{stem}.svg already exists. The 06 SVG is an author-edited base; "
            "use a new --output-stem for generated alternatives."
        )
    font = style()
    metrics = verify_selection()
    online = pd.read_csv(SOURCES["online"])
    reward = pd.read_csv(REWARD)
    online_seeds = pd.read_csv(ONLINE_SEEDS) if relative_reward else None
    reward_seeds = pd.read_csv(REWARD_SEEDS) if relative_reward else None
    tiles = field_tiles()

    fig = plt.figure(figsize=SIZE_IN, facecolor="white")
    # Keep the original 5.41-inch artwork at native size. The remaining
    # 2.09 inches hold two narrow course plots above three unlabeled maps.
    # The placeholder axes is filled with the source SVG after Matplotlib
    # finishes its independent plots and labels.
    ax_art = fig.add_axes([0, 0, ARCHITECTURE_WIDTH_IN / SIZE_IN[0], 1])
    ax_art.axis("off")
    fig.text(0.035 / SIZE_IN[0], 1.86 / 1.9, "A", va="top",
             fontsize=9.5, fontweight="bold", color="#20272C")

    if goal_gate_header:
        fig.text(5.52 / SIZE_IN[0], 1.84 / 1.9, "B", va="top",
                 fontsize=9.5, fontweight="bold", color="#20272C")
    elif relative_reward:
        fig.text(5.52 / SIZE_IN[0], 1.84 / 1.9, "B", va="top",
                 fontsize=9.5, fontweight="bold", color="#20272C")
    else:
        fig.text(5.52 / SIZE_IN[0], 1.84 / 1.9, "B  DG info", va="top",
                 fontsize=9.5, fontweight="bold", color="#20272C")
    course_height = 0.45 if goal_gate_header else 0.69 if relative_reward else 0.47
    course_bottom = 0.92 if goal_gate_header else 0.99
    info_left = 5.85 if relative_reward else 5.70
    info_width = 0.59 if goal_gate_header else 0.52 if relative_reward else 0.60
    ax_info = fig.add_axes([info_left / SIZE_IN[0], course_bottom / 1.9,
                            info_width / SIZE_IN[0], course_height / 1.9])
    course(ax_info, online, "spatial_information", log=False,
           thin_lines=relative_reward, seed_table=online_seeds)
    if relative_reward:
        ax_info.set_ylabel("Spatial info", fontsize=8)
    if goal_gate_header:
        legend_positions = ((7.22, 1.81), (7.22, 1.66), (7.22, 1.51))
    else:
        legend_x = ((5.82, 6.10, 6.37) if relative_reward else
                    (5.64, 5.97, 6.30))
        legend_y = 1.84 if relative_reward else 1.62
        legend_positions = tuple((x, legend_y) for x in legend_x)
    for (x, y), (family, label) in zip(legend_positions,
                                       (("C01", "NG"), ("C05", "UG"),
                                        ("C15", "FG"))):
        fig.text(x / SIZE_IN[0], y / 1.9, label, va="top", fontsize=8,
                 fontweight="bold", color=COLOR[family])

    if not goal_gate_header:
        reward_heading = "C" if relative_reward else "C  Intrinsic\nreward (log)"
        fig.text(6.67 / SIZE_IN[0], 1.84 / 1.9,
                 reward_heading, va="top", fontsize=9.5 if relative_reward else 8.8,
                 linespacing=0.9, fontweight="bold", color="#20272C")
    ax_reward = fig.add_axes([6.88 / SIZE_IN[0], course_bottom / 1.9,
                              0.56 / SIZE_IN[0], course_height / 1.9])
    reward_normalizers = course(ax_reward, reward, "intrinsic_mean", log=True,
                                relative_to_mean=relative_reward,
                                thin_lines=relative_reward,
                                seed_table=reward_seeds,
                                compact_ylabel=goal_gate_header)
    if goal_gate_header:
        # Two short x labels sit beneath their respective axes; a centered
        # label would collide with the second plot's vertical reward label.
        for center_x in (6.145, 7.16):
            fig.text(center_x / SIZE_IN[0], 0.75 / 1.9, "Frames (M)",
                     ha="center", va="center", fontsize=8, color="#20272C")
    else:
        fig.text(6.49 / SIZE_IN[0], 0.81 / 1.9,
                 "Training frames (M)",
                 ha="center", va="center", fontsize=8, color="#20272C")

    field_heading = "C" if goal_gate_header else "D  Example DG place field"
    fig.text(5.52 / SIZE_IN[0], (0.67 if goal_gate_header else 0.73) / 1.9,
             field_heading, va="top", fontsize=9.5, fontweight="bold",
             color="#20272C")
    for index, unit in enumerate(UNITS):
        x = 5.54 + 0.65 * index
        ax = fig.add_axes([x / SIZE_IN[0], 0.035 / 1.9,
                           0.54 / SIZE_IN[0], 0.54 / 1.9])
        ax.imshow(tiles[unit], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_edgecolor("#77858A")

    svg = HERE / f"{stem}.svg"
    fig.savefig(svg, dpi=300, metadata={"Date": None})
    plt.close(fig)
    inline_architecture(svg, harmonize_labels=goal_gate_header)
    if goal_gate_header:
        inline_goal_gate(svg)
    for suffix, options in (
        ("pdf", ["--export-type=pdf"]),
        ("png", [f"--export-width={round(SIZE_IN[0] * 300)}"]),
    ):
        destination = HERE / f"{stem}.{suffix}"
        subprocess.run(["inkscape", str(svg), *options,
                        f"--export-filename={destination}"],
                       check=True, capture_output=True, text=True)

    manifest = {
        "schema": "intrmotiv/figure1-v3-course-candidate/v1",
        "figure": stem,
        "size_inches": list(SIZE_IN),
        "font": font,
        "architecture": "source v3 SVG elements inlined as vectors; original embedded scene and maze images preserved",
        "architecture_arrangement": {"vis_encoder_heading_shift_mm": VIS_TITLE_SHIFT_MM,
                                     "push_legend_typo_corrected": True,
                                     **({"readout_heading": "Policy",
                                         "three_temporal_symbols": "TS"}
                                        if goal_gate_header else {})},
        "panel_a_label": "vector letter preceding the shifted Vis. Encoder heading",
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
                                       ATLAS, UNIT_METRICS, SEED_METRICS,
                                       *([GOAL_GATE_ICON] if goal_gate_header else []))},
    }
    if relative_reward:
        manifest["course_smoothing"] = {
            "applies_to": (["B spatial information", "B intrinsic reward"]
                           if goal_gate_header else
                           ["B spatial information", "C intrinsic reward"]),
            "method": "Gaussian filter on each seed's 1-million-frame grid before recomputing mean and SEM",
            "sigma_million_frames": SMOOTHING_SIGMA_M_FRAMES,
            "edge_mode": "nearest",
            "line_width_points": 0.45,
            "band_alpha": 0.10,
        }
        manifest["source_sha256"].update({
            str(path.relative_to(ROOT)): sha256(path)
            for path in (ONLINE_SEEDS, REWARD_SEEDS)
        })
        manifest["reward_display"] = {
            "transformation": "five-seed mean and SEM at each training point divided by the arithmetic mean of the 99 five-seed means for that condition",
            "normalizer_by_family": reward_normalizers,
            "unit": "fold of own-condition mean",
            "reference_line": 1,
            "normalizer_uncertainty": "treated as a descriptive fixed scaling constant",
        }
        manifest.pop("reward_log_scale")
        manifest["reward_axis_scale"] = "linear display axis of within-condition fold change"
    if goal_gate_header:
        manifest["panel_layout"] = {
            "B": "spatial-information and relative-reward courses share one panel; side-gate icon occupies its top row with the NG/UG/FG legend stacked at its right",
            "C": "three C15 seed-123 DG response maps; panel letter only, no heading",
            "icon": "05_side_gate.svg inlined as native vectors at 1.23 inches wide, with TS and Pol. labels",
        }
    (HERE / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-reward", action="store_true",
                        help="Save a separate version with reward curves divided by each condition's own mean")
    parser.add_argument("--goal-gate-header", action="store_true",
                        help="Save a separate B/C merged version with the side-gate icon above both curves")
    parser.add_argument("--output-stem",
                        help="New filename stem for a generated alternative (required to avoid replacing the author-edited 06 SVG)")
    args = parser.parse_args()
    main(relative_reward=args.relative_reward or args.goal_gate_header,
         goal_gate_header=args.goal_gate_header,
         output_stem=args.output_stem)
