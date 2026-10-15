"""Apply Tukey whiskers to the imported C/D boxes in the current Figure 2.

The author's SVG holds editable Matplotlib paths. This adapter checks each
stored box against the archived five-seed differences before moving only its
whisker and cap endpoints. Quartiles, medians, dots, axes, and layout stay put.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re

from build_figure2_story import box_summary


HERE = Path(__file__).resolve().parent
DIFFERENCES = HERE / "figure2_candidates/08_focused_canvas/plotted_seed_differences.csv"
METRICS = ("dg_information", "map_overlap", "peak_coverage", "graph_edges")
VERTICAL = re.compile(r"([Mm]) ([-\d.]+),([-\d.]+) ([Vv]) ([-\d.]+)")
HORIZONTAL = re.compile(r"([Mm]) ([-\d.]+),([-\d.]+) ([Hh]) ([-\d.]+)")


def path_data(svg: str, element_id: str) -> str:
    pattern = (rf'<g\b[^>]*\bid="{re.escape(element_id)}"[^>]*>\s*'
               r'<path\s+d="([^"]+)"')
    matches = re.findall(pattern, svg, flags=re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected one editable path for {element_id}; found {len(matches)}")
    return matches[0]


def replace_path_data(svg: str, element_id: str, old: str, new: str) -> str:
    pattern = (rf'(<g\b[^>]*\bid="{re.escape(element_id)}"[^>]*>\s*'
               rf'<path\s+d="){re.escape(old)}(")')
    svg, count = re.subn(pattern, lambda match: match.group(1) + new + match.group(2),
                         svg, count=1, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Could not update {element_id}")
    return svg


def vertical_coordinates(path: str) -> tuple[float, float, float]:
    match = VERTICAL.fullmatch(path)
    if match is None:
        raise ValueError(f"Unexpected vertical whisker path: {path}")
    x, start, endpoint = map(float, (match.group(2), match.group(3), match.group(5)))
    return x, start, start + endpoint if match.group(4) == "v" else endpoint


def cap_coordinates(path: str) -> tuple[float, float, float]:
    match = HORIZONTAL.fullmatch(path)
    if match is None:
        raise ValueError(f"Unexpected whisker cap path: {path}")
    left, y, right = map(float, (match.group(2), match.group(3), match.group(5)))
    return left, y, left + right if match.group(4) == "h" else right


def apply_tukey_whiskers(svg: str) -> tuple[str, list[dict[str, object]]]:
    """Reposition imported C/D whiskers after validating their source values."""
    with DIFFERENCES.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summaries: list[dict[str, object]] = []
    for arm in ("push", "pull"):
        for metric in METRICS:
            families = (("C01", "C05", "C15") if arm == "push" and metric != "graph_edges"
                        else ("C05", "C15"))
            first_line = 8 if len(families) == 3 else 7
            for index, family in enumerate(families):
                selected = [row for row in rows if row["arm"] == arm and
                            row["metric"] == metric and row["family"] == family]
                if len(selected) != 5 or len({row["seed"] for row in selected}) != 5:
                    raise ValueError(f"Expected five distinct seeds for {arm}/{metric}/{family}")
                values = [float(row["delta"]) for row in selected]
                low, q1, _, q3, high = box_summary(values, "tukey")
                prefix = f"{arm}_{metric}_line2d_"
                identifiers = [prefix + str(first_line + 4 * index + offset)
                               for offset in range(4)]
                paths = [path_data(svg, identifier) for identifier in identifiers]
                (x_low, q1_y, min_y), (x_high, q3_y, max_y) = map(
                    vertical_coordinates, paths[:2])
                (cap_low_x, cap_low_y, cap_low_right), (cap_high_x, cap_high_y,
                                                       cap_high_right) = map(cap_coordinates, paths[2:])
                if max(values) == min(values):
                    raise ValueError(f"Cannot calibrate constant values for {arm}/{metric}/{family}")
                slope = (max_y - min_y) / (max(values) - min(values))
                intercept = min_y - slope * min(values)
                if (abs(q1_y - (slope * q1 + intercept)) > 0.01 or
                        abs(q3_y - (slope * q3 + intercept)) > 0.01 or
                        abs(cap_low_y - min_y) > 0.01 or
                        abs(cap_high_y - max_y) > 0.01 or
                        abs(x_low - x_high) > 0.01):
                    raise ValueError(f"SVG box disagrees with archived data: {arm}/{metric}/{family}")
                low_y, high_y = slope * low + intercept, slope * high + intercept
                if low != min(values):
                    replacements = ((0, f"M {x_low:.6f},{q1_y:.6f} V {low_y:.6f}"),
                                    (2, f"M {cap_low_x:.6f},{low_y:.6f} H {cap_low_right:.6f}"))
                    for offset, replacement in replacements:
                        svg = replace_path_data(svg, identifiers[offset], paths[offset], replacement)
                if high != max(values):
                    replacements = ((1, f"M {x_high:.6f},{q3_y:.6f} V {high_y:.6f}"),
                                    (3, f"M {cap_high_x:.6f},{high_y:.6f} H {cap_high_right:.6f}"))
                    for offset, replacement in replacements:
                        svg = replace_path_data(svg, identifiers[offset], paths[offset], replacement)
                summaries.append({"arm": arm, "metric": metric, "family": family,
                                  "sample_count": len(values),
                                  "outlier_count": sum(value < low or value > high for value in values),
                                  "whisker_low": low, "whisker_high": high})
    return svg, summaries
