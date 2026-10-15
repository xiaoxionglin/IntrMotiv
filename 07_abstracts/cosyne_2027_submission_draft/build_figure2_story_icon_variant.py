"""Build the E-free Figure 2 story with metric icons beside A–D.

This variant reuses all quantitative panels from the story figure. Its eight
schematic icons are purely explanatory and remain editable vector content.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from matplotlib import colormaps

from build_figure2_metric_icons import HEIGHT_PT, ICONS, RED, WIDTH_PT, OVERLAP_WIDTH_PT
from figure2_icon_candidates.spatial_information.build_spatial_information_icons import (
    SELECTED_FIGURE_AMPLITUDE, perlin_dg_maps,
)
from build_figure2_story import (
    HERE, ROOT, behavior_differences, box_summary, export, layout_base,
    selected_trajectories, sha256, story_overlay, terminal_rows,
)
from figure2_whisker_transform import DIFFERENCES, apply_tukey_whiskers
from figure2_three_category_pull import (
    PAIRS, PAIRS_SHA256, SHARED_MASK_SHA256,
    add_ng_na_to_edge_panels, add_ng_pull_representation,
)


OUTPUT = HERE / "Figure2_story_icons.svg"
MANIFEST = HERE / "Figure2_story_icons_manifest.json"
ICON_SOURCE = HERE / "build_figure2_metric_icons.py"
STORY_SOURCE = HERE / "build_figure2_story.py"
SCALE_RESULTS = 57 / HEIGHT_PT
RESULT_ICON_WIDTH = WIDTH_PT * SCALE_RESULTS
SPATIAL_ICON_POSITIONS = {
    "behavior": (396, 35), "push": (171, 106), "pull": (171, 171),
}
RESULT_POSITIONS = {
    "push": {"overlap": (279, 100.5), "coverage": (401.5, 100.5),
             "edges": (511, 100.5)},
    "pull": {"overlap": (279, 165.5), "coverage": (401.5, 165.5),
             "edges": (511, 165.5)},
}
RELIABILITY_POSITION = (505.5, 24.5)
PATH_EFFICIENCY_POSITION = (74.5, 27)
TRAJECTORY_CENTERS = (129, 171, 213)
TRAJECTORY_SIZE = 39
TRAJECTORY_CANDIDATES = HERE / "figure2_candidates/09_trajectory_thumbnails"
TRAJECTORY_TILES = {
    family: TRAJECTORY_CANDIDATES / f"{family}_65_fragments_tile.svg"
    for family in ("NG", "UG", "FG")
}


def trajectory_tile(family: str, x: float, y: float, size: int) -> str:
    """Embed the selected fragment tile as vector paths at the existing size."""
    root = ET.parse(TRAJECTORY_TILES[family]).getroot()
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    for child in list(root):
        if child.tag == f"{{{svg_ns}}}metadata":
            root.remove(child)
    ids = {
        element.attrib["id"]: f"trajectory-{family}-{element.attrib['id']}"
        for element in root.iter() if "id" in element.attrib
    }
    for element in root.iter():
        for key, value in tuple(element.attrib.items()):
            if key == "id":
                element.set(key, ids[value])
            else:
                for old, new in ids.items():
                    value = value.replace(f"url(#{old})", f"url(#{new})")
                    if value == f"#{old}":
                        value = f"#{new}"
                element.set(key, value)
    root.set("id", f"figure2a-{family}-trajectory")
    root.set("x", f"{x:g}")
    root.set("y", f"{y:g}")
    root.set("width", f"{size:g}")
    root.set("height", f"{size:g}")
    return ET.tostring(root, encoding="unicode")


def trajectory_step_scale() -> str:
    """Show the local 0–63 step colors used independently in each fragment."""
    x0, y0, width, height = 170, 78, 50, 4
    parts = ['<g id="figure2a-step-scale" aria-label="Step within each trajectory fragment">',
             '<text x="137" y="82.5" class="mini-label">step</text>',
             '<text x="163" y="82.5" class="mini-label">0</text>']
    cmap = colormaps["viridis"]
    for step in range(64):
        rgb = cmap(step / 63)[:3]
        color = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
        parts.append(f'<rect x="{x0 + step * width / 64:.5f}" y="{y0}" '
                     f'width="{width / 64 + .01:.5f}" height="{height}" '
                     f'fill="{color}" stroke="none"/>')
    parts.append('<text x="223" y="82.5" class="mini-label">63</text></g>')
    return "\n".join(parts)


def unique_icon(name: str, instance: str) -> str:
    """Namespace SVG definitions so repeated icons remain independent."""
    body = ICONS[name]()
    identifiers = re.findall(r'\bid="([^"]+)"', body)
    for identifier in identifiers:
        unique = f'{identifier}-{instance}'
        body = body.replace(f'id="{identifier}"', f'id="{unique}"')
        body = body.replace(f'url(#{identifier})', f'url(#{unique})')
    return body


def spatial_icon_group(row: str, dg_red: str = RED) -> str:
    """Embed the selected single-DG map at the same scale in B, C, and D."""
    x, y = SPATIAL_ICON_POSITIONS[row]
    identifier = f"{row}-spatial-information-icon"
    return (
        f'<g id="{identifier}" '
        f'aria-label="Spatial-information schematic for panel {row}" '
        f'transform="translate({x:g},{y:g}) scale({SCALE_RESULTS:.8f})">'
        f'{perlin_dg_maps(dg_red, id_prefix=identifier, amplitude=SELECTED_FIGURE_AMPLITUDE)}'
        '</g>'
    )


def icon_overlay() -> str:
    parts = ['<g id="figure2-metric-icon-overlay" '
             'aria-label="Schematic metric icons; upper example has higher raw metric value">']
    for row, metrics in RESULT_POSITIONS.items():
        for metric, (x, y) in metrics.items():
            scale = 1.0 if metric == "overlap" else SCALE_RESULTS
            parts.append(
                f'<g id="{row}-{metric}-icon" '
                f'transform="translate({x:g},{y:g}) scale({scale:.8f})">'
                f'{unique_icon(metric, f"{row}-{metric}")}</g>'
            )
    x, y = RELIABILITY_POSITION
    for row in SPATIAL_ICON_POSITIONS:
        parts.append(spatial_icon_group(row))
    parts.append(f'<g id="behavior-reliability-icon" transform="translate({x:g},{y:g})">'
                 f'{unique_icon("reliability", "behavior")}</g>')
    x, y = PATH_EFFICIENCY_POSITION
    parts.append(f'<g id="exploration-path-efficiency-icon" '
                 f'transform="translate({x:g},{y:g})">'
                 f'{unique_icon("path_efficiency", "exploration")}</g>')
    parts.append('</g>')
    return '\n'.join(parts)


def align_y_label_spacing(svg: str) -> str:
    """Keep visible B–D y labels the same distance from their axes.

    The source edge labels sit closer to the axis than the other C/D labels,
    while the two new B labels sit farther away. These small translations
    change only the labels and leave every plot rectangle in place.
    """
    for row in ("push", "pull"):
        identifier = f"{row}_graph_edges_text_6"
        svg, count = re.subn(
            rf'(<g\s+id="{identifier}")',
            r'\1 transform="translate(-6.5,0)"', svg, count=1,
        )
        if count != 1:
            raise ValueError(f"Expected one edge ylabel: {identifier}")
    # This source overlap label alone has a closer baseline in the pull row.
    svg, count = re.subn(
        r'(<g\s+id="pull_map_overlap_text_6")',
        r'\1 transform="translate(-4.5,0)"', svg, count=1,
    )
    if count != 1:
        raise ValueError("Expected one pull overlap ylabel")
    for metric in ("dg_information", "dg_reliability"):
        identifier = f"behavior-{metric}-ylabel"
        svg, count = re.subn(
            rf'(<text id="{identifier}"[^>]*>.*?</text>)',
            r'<g transform="translate(4.5,0)">\1</g>',
            svg, count=1, flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError(f"Expected one behavior ylabel: {identifier}")
    return svg


def build() -> None:
    svg = layout_base(extra_plot_shift=1, metric_extra_shifts=(0, 6, 28, 40))
    svg = add_ng_na_to_edge_panels(svg)
    svg, imported_box_summaries = apply_tukey_whiskers(svg)
    svg, ng_box_summaries = add_ng_pull_representation(svg)
    imported_box_summaries.extend(ng_box_summaries)
    old_name = 'sodipodi:docname="Figure2_story.svg"'
    if svg.count(old_name) != 1:
        raise ValueError("Could not identify the story canvas document name")
    svg = svg.replace(old_name, 'sodipodi:docname="Figure2_story_icons.svg"', 1)
    behavior = behavior_differences()
    overlay = story_overlay(
        terminal_rows(), selected_trajectories(), behavior,
        trajectory_tile_size=TRAJECTORY_SIZE,
        trajectory_centers=TRAJECTORY_CENTERS,
        path_metric_shift=1,
        elbow_arrows=True, paired_peak_layout=True,
        show_panel_e=False, behavior_layout_shift=40,
        information_shift=10, reliability_shift=25,
        whisker_rule="tukey",
        trajectory_renderer=trajectory_tile,
        extra_overlay=icon_overlay() + trajectory_step_scale(),
    )
    svg, count = re.subn(r'</svg>\s*$', overlay + '\n</svg>\n', svg)
    if count != 1:
        raise ValueError("Could not append the Figure 2 icon overlay")
    svg = align_y_label_spacing(svg)
    export(svg, OUTPUT)

    MANIFEST.write_text(json.dumps({
        "schema": "cosyne27-figure2-story-icon-variant/v1",
        "parent_story_source": str(STORY_SOURCE.relative_to(ROOT)),
        "parent_story_source_sha256": sha256(STORY_SOURCE),
        "icon_source": str(ICON_SOURCE.relative_to(ROOT)),
        "icon_source_sha256": sha256(ICON_SOURCE),
        "physical_size_points": [540, 230],
        "removed_panel": "E",
        "quantitative_panels": "D representation plots add five NG paired differences from the audited shared-mask source; existing UG/FG values and y scales remain unchanged. B–D statistical plot dimensions are retained; A trajectory tiles are 39 points square and display 65 complete fragments",
        "boxplot_whisker_rule": "lowest and highest observed values within Q1 - 1.5 IQR and Q3 + 1.5 IQR; all observations remain as dots",
        "imported_box_source": {
            "path": str(DIFFERENCES.relative_to(ROOT)),
            "sha256": sha256(DIFFERENCES),
        },
        "imported_box_summaries": imported_box_summaries,
        "pull_shared_mask_source": {
            "path": str(PAIRS.relative_to(ROOT)),
            "sha256": PAIRS_SHA256,
            "shared_mask_sha256": SHARED_MASK_SHA256,
            "shared_mask_bins": 318,
            "sample_count_per_family": 5,
            "interventions": {
                "NG": "temporal versus calibrated constant flat-policy reward",
                "UG/FG": "temporal versus constant worker-hit reward with temporal DG credit fixed",
            },
        },
        "behavior_box_outlier_counts": {
            metric: {
                family: sum(value < summary[0] or value > summary[4]
                            for value in values)
                for family, values in groups.items()
                for summary in [box_summary(values, "tukey")]
            } for metric, groups in behavior.items()
        },
        "trajectory_source_protocol": "100M-frame checkpoint, 100k-observation own-policy online-spatial window; 65 sampled complete fragments per architecture; viridis maps local step 0–63 within each fragment",
        "trajectory_tiles": {family: {
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        } for family, path in TRAJECTORY_TILES.items()},
        "trajectory_candidate_manifest": {
            "path": str((TRAJECTORY_CANDIDATES / "manifest.json").relative_to(ROOT)),
            "sha256": sha256(TRAJECTORY_CANDIDATES / "manifest.json"),
        },
        "trajectory_centers_points": TRAJECTORY_CENTERS,
        "trajectory_tile_size_points": TRAJECTORY_SIZE,
        "path_metric_shift_points": 1,
        "ablation_schematic": "TS lowered; directed arrows terminate on the bottom edges of DG and policy",
        "behavior_layout_shift_points": 40,
        "behavior_information_shift_points": 10,
        "behavior_reliability_shift_points": 25,
        "result_group_shifts_points": {
            "dg_information": 1, "map_overlap": 7,
            "peak_coverage": 29, "graph_edges": 41,
        },
        "result_icon_size_points": {
            "overlap": [OVERLAP_WIDTH_PT, 20 * 72 / 25.4],
            "coverage_and_edges": [RESULT_ICON_WIDTH, 57],
        },
        "result_icon_positions_points": RESULT_POSITIONS,
        "spatial_information_icon_positions_points": SPATIAL_ICON_POSITIONS,
        "spatial_information_icon_scale": SCALE_RESULTS,
        "spatial_information_icon_amplitude": SELECTED_FIGURE_AMPLITUDE,
        "spatial_information_icon_baseline_opacity": 0.1,
        "graph_edge_ng_category": "N/A in C and D; no NG graph edges, box, dots, or numeric zero",
        "reliability_icon_size_points": [WIDTH_PT, 20 * 72 / 25.4],
        "reliability_icon_position_points": RELIABILITY_POSITION,
        "path_efficiency_icon_size_points": [WIDTH_PT, 20 * 72 / 25.4],
        "path_efficiency_icon_position_points": PATH_EFFICIENCY_POSITION,
        "icon_interpretation": "top is a higher raw metric value; upward arrow is a scale cue, not an observed condition effect",
        "icons": {name: {
            "path": str((HERE / f"Figure2_metric_icon_{name}.svg").relative_to(ROOT)),
            "sha256": sha256(HERE / f"Figure2_metric_icon_{name}.svg"),
        } for name in ICONS},
    }, indent=2) + '\n')


if __name__ == '__main__':
    build()
