"""Build the selected Figure 2 with A–D metric icons in their y-axis gutters.

The H-palette preview is the immutable layout input. This builder changes
horizontal plot/label/icon positions and icon scale; plotted points, boxes,
axes, and vertical coordinates retain their source values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "figure_palette_candidates/priority_rgb_orange_violet_figure2.svg"
OUTPUT = HERE / "Figure2.svg"
MANIFEST = HERE / "Figure2_layout_manifest.json"
SVG_PT_PER_INKSCAPE_PX = 540 / (7.5 * 96)
LABEL_ICON_GAP = 2.0
ICON_TICK_GAP = 2.0

# The two-map icon needs more room than the smaller trial icon. Shift that
# column and the columns after it equally, preserving their mutual spacing.
PLOT_SHIFTS = {"dg_information": 10.0, "map_overlap": 16.0,
               "peak_coverage": 10.0, "graph_edges": 10.0}
PATH_EFFICIENCY_SHIFT = 24.0
BEHAVIOR_SHIFT = 8.0
ICON_WIDTHS = {"spatial": 12.0, "coverage": 14.0, "edges": 14.0}
CD_SPATIAL_LABEL_DOWN_PT = 2.0


def query_boxes(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Read actual rendered bounds, including rotated labels and map art."""
    result = subprocess.run(
        ["inkscape", str(path), "--query-all"], capture_output=True,
        text=True, check=True,
    )
    return {
        row[0]: tuple(float(value) * SVG_PT_PER_INKSCAPE_PX for value in row[1:5])
        for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 5
    }


def move_element(
    svg: str, identifier: str, shift_x: float, *, shift_y: float = 0.0,
    tag: str = "g",
) -> str:
    """Translate one existing SVG group or text without changing descendants."""
    pattern = rf'<{tag}\s+id="{re.escape(identifier)}"[^>]*>'

    def replace(match: re.Match[str]) -> str:
        opening = match.group(0)
        translation = f"translate({shift_x:.6f},{shift_y:.6f})"
        if 'transform="' in opening:
            return opening.replace('transform="', f'transform="{translation} ', 1)
        return opening[:-1] + f' transform="{translation}">'

    svg, count = re.subn(pattern, replace, svg, count=1)
    if count != 1:
        raise ValueError(f"Expected exactly one {tag} with id={identifier}")
    return svg


def move_behavior_heading(svg: str, shift_x: float) -> str:
    """Move the two source B heading texts, which have no element IDs."""
    for opening in ('<text x="237" y="12" class="story-letter">',
                    '<text x="254" y="12" class="story-heading">'):
        if svg.count(opening) != 1:
            raise ValueError(f"Expected one B heading element: {opening}")
        svg = svg.replace(
            opening, opening[:-1] + f' transform="translate({shift_x:.6f},0)">', 1,
        )
    return svg


def shift_component(svg: str, identifier: str, shift_x: float) -> str:
    """Move one imported Matplotlib plot without changing its local geometry."""
    pattern = (rf'(<g\s+id="{re.escape(identifier)}"[^>]*'
               rf'transform="translate\()([0-9.]+)(,[-0-9.]+\)")')
    svg, count = re.subn(
        pattern,
        lambda match: (match.group(1) +
                       f"{float(match.group(2)) + shift_x:.8f}" +
                       match.group(3)),
        svg, count=1,
    )
    if count != 1:
        raise ValueError(f"Expected one translated plot: {identifier}")
    return svg


def place_icon(
    svg: str, identifier: str, source_box: tuple[float, float, float, float],
    target_left: float, target_center_y: float, target_width: float,
) -> str:
    """Retain the icon's aspect ratio while fitting its measured artwork."""
    pattern = (rf'(<g\s+id="{re.escape(identifier)}"[^>]*'
               rf'transform="translate\()([-0-9.]+),([-0-9.]+)\)'
               rf'(?: scale\(([0-9.]+)\))?(")')

    def replace(match: re.Match[str]) -> str:
        old_x, old_y = map(float, match.group(2, 3))
        old_scale = float(match.group(4) or 1)
        new_scale = old_scale * target_width / source_box[2]
        body_min_x = (source_box[0] - old_x) / old_scale
        body_min_y = (source_box[1] - old_y) / old_scale
        new_height = source_box[3] * new_scale / old_scale
        new_x = target_left - body_min_x * new_scale
        new_y = target_center_y - new_height / 2 - body_min_y * new_scale
        return (match.group(1) + f"{new_x:.6f},{new_y:.6f}) "
                f"scale({new_scale:.8f})" + match.group(5))

    svg, count = re.subn(pattern, replace, svg, count=1)
    if count != 1:
        raise ValueError(f"Expected one movable icon: {identifier}")
    return svg


def tick_text_ids(root: ET.Element, prefix: str) -> list[str]:
    """Find the numeric y tick labels inside the source axis group."""
    axis_id = f"{prefix}_matplotlib.axis_2"
    axis = next((node for node in root.iter() if node.get("id") == axis_id), None)
    if axis is None:
        raise ValueError(f"Missing y axis: {axis_id}")
    return [node.get("id") for tick in axis
            if "_ytick_" in tick.get("id", "")
            for node in tick.iter()
            if node.get("id", "").startswith(f"{prefix}_text_")]


def build() -> None:
    svg = SOURCE.read_text()
    root = ET.fromstring(svg)
    boxes = query_boxes(SOURCE)
    records = []
    overlap_extra = boxes["push-overlap-icon"][2] - 18.0
    reliability_extra = boxes["behavior-reliability-icon"][2] - 18.0
    plot_shifts = {
        metric: shift + (overlap_extra if metric != "dg_information" else 0.0)
        for metric, shift in PLOT_SHIFTS.items()
    }
    # Align B's last plotted column with the rightmost C/D column. Translate
    # the complete B panel so its schematic and title keep their spacing.
    cd_right = max(
        boxes[f"component_{row}_graph_edges"][0]
        + boxes[f"component_{row}_graph_edges"][2]
        + plot_shifts["graph_edges"]
        for row in ("push", "pull")
    )
    behavior_right = (boxes["behavior-dg_reliability"][0]
                      + boxes["behavior-dg_reliability"][2]
                      + BEHAVIOR_SHIFT + reliability_extra)
    behavior_align_shift = cd_right - behavior_right
    svg = move_behavior_heading(svg, behavior_align_shift)
    svg = move_element(svg, "behavior-ablation-schematic", behavior_align_shift)

    specs = [
        # panel, metric, icon, plot group, y label, numeric tick labels,
        # plot shift, target icon artwork width in SVG points.
        ("A", "path_efficiency", "exploration-path-efficiency-icon",
         "panel-a-path-efficiency", "panel-a-path-efficiency-ylabel",
         [f"panel-a-path-efficiency-ytick-{i}" for i in range(3)],
         PATH_EFFICIENCY_SHIFT, boxes["exploration-path-efficiency-icon"][2]),
        ("B", "dg_information", "behavior-spatial-information-icon",
         "behavior-dg_information", "behavior-dg_information-ylabel",
         [f"behavior-dg_information-ytick-{i}" for i in range(3)],
         BEHAVIOR_SHIFT + behavior_align_shift, ICON_WIDTHS["spatial"]),
        ("B", "dg_reliability", "behavior-reliability-icon",
         "behavior-dg_reliability", "behavior-dg_reliability-ylabel",
         [f"behavior-dg_reliability-ytick-{i}" for i in range(3)],
         BEHAVIOR_SHIFT + reliability_extra + behavior_align_shift,
         boxes["behavior-reliability-icon"][2]),
    ]
    for row in ("push", "pull"):
        panel = "C" if row == "push" else "D"
        for metric, icon_kind in (("dg_information", "spatial"),
                                  ("map_overlap", "overlap"),
                                  ("peak_coverage", "coverage"),
                                  ("graph_edges", "edges")):
            prefix = f"{row}_{metric}"
            label_index = (7 if row == "push" else 6)
            if metric == "graph_edges":
                label_index = 6
            specs.append((
                panel, metric, f"{row}-{icon_kind if icon_kind != 'spatial' else 'spatial-information'}-icon",
                f"component_{prefix}", f"{prefix}_text_{label_index}",
                tick_text_ids(root, prefix), plot_shifts[metric],
                boxes[f"{row}-overlap-icon"][2] if icon_kind == "overlap"
                else ICON_WIDTHS[icon_kind],
            ))

    for panel, metric, icon_id, plot_id, label_id, tick_ids, plot_shift, icon_width in specs:
        if plot_id.startswith("component_"):
            svg = shift_component(svg, plot_id, plot_shift)
            axis_box = boxes[f"{plot_id.removeprefix('component_')}_patch_2"]
            axis_center_y = axis_box[1] + axis_box[3] / 2
            label_shift_with_plot = plot_shift
        else:
            svg = move_element(svg, plot_id, plot_shift)
            if panel == "A":
                axis_center_y = (36 + 69) / 2
                label_shift_with_plot = plot_shift
            else:
                axis_center_y = 30 + 47.52 / 2
                label_shift_with_plot = 0.0
        tick_left = min(boxes[tick_id][0] for tick_id in tick_ids) + plot_shift
        icon_left = tick_left - ICON_TICK_GAP - icon_width
        label_right = boxes[label_id][0] + boxes[label_id][2] + label_shift_with_plot
        target_label_right = icon_left - LABEL_ICON_GAP
        label_shift_y = (CD_SPATIAL_LABEL_DOWN_PT
                         if panel in ("C", "D") and metric == "dg_information"
                         else 0.0)
        svg = move_element(svg, label_id, target_label_right - label_right,
                           shift_y=label_shift_y,
                           tag="text" if panel in ("A", "B") else "g")
        svg = place_icon(svg, icon_id, boxes[icon_id], icon_left,
                         axis_center_y, icon_width)
        records.append({"panel": panel, "metric": metric, "icon_id": icon_id,
                        "icon_width_pt": icon_width, "plot_shift_pt": plot_shift,
                        "target_icon_left_pt": round(icon_left, 3),
                        "target_label_right_pt": round(target_label_right, 3),
                        "label_vertical_shift_pt": label_shift_y})

    OUTPUT.write_text(svg)
    for extension in ("pdf", "png"):
        command = ["inkscape", str(OUTPUT),
                   f"--export-filename={OUTPUT.with_suffix('.' + extension)}"]
        if extension == "png":
            command.append("--export-width=1800")
        subprocess.run(command, capture_output=True, text=True, check=True)
    MANIFEST.write_text(json.dumps({
        "schema": "intrmotiv/figure2-layout/v1",
        "source": str(SOURCE.relative_to(HERE)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_svg_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "output_pdf_sha256": hashlib.sha256(OUTPUT.with_suffix('.pdf').read_bytes()).hexdigest(),
        "output_png_sha256": hashlib.sha256(OUTPUT.with_suffix('.png').read_bytes()).hexdigest(),
        "canvas_width_pt": 540,
        "arrangement": "panel A-D icons between their own y labels and numeric tick labels",
        "behavior_panel_shift_pt": round(behavior_align_shift, 3),
        "source_data_unchanged": True,
        "placements": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    build()
