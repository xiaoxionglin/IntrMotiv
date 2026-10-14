"""Build the E-free Figure 2 story with metric icons beside B, C, and D.

This variant reuses all quantitative panels from the story figure. Its seven
schematic icons are purely explanatory and remain editable vector content.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from build_figure2_metric_icons import HEIGHT_PT, ICONS, WIDTH_PT
from build_figure2_story import (
    HERE, ROOT, behavior_differences, export, layout_base,
    selected_trajectories, sha256, story_overlay, terminal_rows,
)


OUTPUT = HERE / "Figure2_story_icons.svg"
MANIFEST = HERE / "Figure2_story_icons_manifest.json"
ICON_SOURCE = HERE / "build_figure2_metric_icons.py"
STORY_SOURCE = HERE / "build_figure2_story.py"
SCALE_RESULTS = 57 / HEIGHT_PT
RESULT_ICON_WIDTH = WIDTH_PT * SCALE_RESULTS
RESULT_POSITIONS = {
    "push": {"overlap": (178, 100.5), "coverage": (295, 100.5),
             "edges": (411, 100.5)},
    "pull": {"overlap": (178, 165.5), "coverage": (295, 165.5),
             "edges": (411, 165.5)},
}
RELIABILITY_POSITION = (384, 24.5)


def unique_icon(name: str, instance: str) -> str:
    """Namespace SVG definitions so repeated icons remain independent."""
    body = ICONS[name]()
    identifiers = re.findall(r'\bid="([^"]+)"', body)
    for identifier in identifiers:
        unique = f'{identifier}-{instance}'
        body = body.replace(f'id="{identifier}"', f'id="{unique}"')
        body = body.replace(f'url(#{identifier})', f'url(#{unique})')
    return body


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
    parts.append(f'<g id="behavior-reliability-icon" transform="translate({x:g},{y:g})">'
                 f'{unique_icon("reliability", "behavior")}</g>')
    parts.append('</g>')
    return '\n'.join(parts)


def build() -> None:
    svg = layout_base(extra_plot_shift=1, metric_extra_shifts=(0, 16, 32, 48))
    old_name = 'sodipodi:docname="Figure2_story.svg"'
    if svg.count(old_name) != 1:
        raise ValueError("Could not identify the story canvas document name")
    svg = svg.replace(old_name, 'sodipodi:docname="Figure2_story_icons.svg"', 1)
    overlay = story_overlay(
        terminal_rows(), selected_trajectories(), behavior_differences(),
        trajectory_tile_size=38, elbow_arrows=True, paired_peak_layout=True,
        show_panel_e=False, behavior_layout_shift=30,
        compact_behavior_schematic=True, reliability_shift=33,
        extra_overlay=icon_overlay(),
    )
    svg, count = re.subn(r'</svg>\s*$', overlay + '\n</svg>\n', svg)
    if count != 1:
        raise ValueError("Could not append the Figure 2 icon overlay")
    export(svg, OUTPUT)

    MANIFEST.write_text(json.dumps({
        "schema": "cosyne27-figure2-story-icon-variant/v1",
        "parent_story_source": str(STORY_SOURCE.relative_to(ROOT)),
        "parent_story_source_sha256": sha256(STORY_SOURCE),
        "icon_source": str(ICON_SOURCE.relative_to(ROOT)),
        "icon_source_sha256": sha256(ICON_SOURCE),
        "physical_size_points": [540, 230],
        "removed_panel": "E",
        "quantitative_panels": "A–D data unchanged; B–D statistical plot dimensions retained; A trajectory tiles are 38 points square; C/D components redistributed horizontally",
        "behavior_layout_shift_points": 30,
        "compact_behavior_schematic": True,
        "behavior_reliability_shift_points": 33,
        "result_group_shifts_points": {
            "dg_information": 1, "map_overlap": 17,
            "peak_coverage": 33, "graph_edges": 49,
        },
        "result_icon_size_points": {
            "overlap": [WIDTH_PT, 20 * 72 / 25.4],
            "coverage_and_edges": [RESULT_ICON_WIDTH, 57],
        },
        "result_icon_positions_points": RESULT_POSITIONS,
        "reliability_icon_size_points": [WIDTH_PT, 20 * 72 / 25.4],
        "reliability_icon_position_points": RELIABILITY_POSITION,
        "icon_interpretation": "top is a higher raw metric value; upward arrow is a scale cue, not an observed condition effect",
        "icons": {name: {
            "path": str((HERE / f"Figure2_metric_icon_{name}.svg").relative_to(ROOT)),
            "sha256": sha256(HERE / f"Figure2_metric_icon_{name}.svg"),
        } for name in ICONS},
    }, indent=2) + '\n')


if __name__ == '__main__':
    build()
