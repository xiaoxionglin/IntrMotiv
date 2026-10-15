"""Add validated NG policy-reward pairs to Figure 2D's editable SVG plots.

The author's imported D plots already contain the final UG/FG values. This
adapter checks those values against the audited shared-mask table, moves their
glyphs horizontally to the three-category positions used in C, then draws the
NG box and five seed dots at the unchanged D y scale. Graph edges remain N/A.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from build_figure2_story import box_summary
from figure2_whisker_transform import DIFFERENCES


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRS = (ROOT / "06_experiments/reward_credit/results/"
         "cosyne27_c01_flat_policy_reward_20261015/figure2d_shared_mask_pairs.csv")
PAIRS_SHA256 = "1f8644a774e18ceb9890c80498f96599565368e6756da9f8d31b80e00b1d8395"
ARM_VALUES_SHA256 = "e5785e82ac758ff62ea294dbd269d21d4efd634ec4306e751b10802c1c3d7265"
PROVENANCE_SHA256 = "a2ac3fffc3416d5832eb78792c34f785402a8ea417c546df464468189f925fe6"
SHARED_MASK_SHA256 = "fc9481bbb296fae9facc6890267a828266e5d0554493ee98d471e4cc7a454e5a"
SEEDS = (8, 17, 41, 99, 123)
FAMILIES = {"C01": ("NG", "flat_policy_reward", "False"),
            "C05": ("UG", "worker_hit_bonus", "True"),
            "C15": ("FG", "worker_hit_bonus", "True")}
METRIC_COLUMNS = {
    "dg_information": "active_unit_mean_spatial_information_bits",
    "map_overlap": "active_only_map_cosine",
    "peak_coverage": "eligible_peak_area_fraction",
}
NG_SOURCE_COLOR = "#0072B2"  # Recolored to H-palette slate by the palette builder.
NG_X, UG_X, FG_X = 74.668966, 91.8, 108.931034
OLD_CENTERS = (78.726318, 104.873690)
BOX_WIDTH = 4.79669
BOX_WIDTH_SCALE = BOX_WIDTH / 7.32127


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_pairs(path: Path = PAIRS) -> dict[tuple[str, int], dict[str, str]]:
    """Verify pinned input, pair arithmetic, cohort, and prior UG/FG values."""
    if sha256(path) != PAIRS_SHA256:
        raise ValueError(f"Unexpected Figure 2D pair table hash: {path}")
    companion_hashes = {
        "figure2d_shared_mask_arm_values.csv": ARM_VALUES_SHA256,
        "figure2d_shared_mask_provenance.json": PROVENANCE_SHA256,
    }
    for name, expected in companion_hashes.items():
        if sha256(path.with_name(name)) != expected:
            raise ValueError(f"Unexpected Figure 2D companion hash: {name}")
    provenance = json.loads(path.with_name("figure2d_shared_mask_provenance.json").read_text())
    if (provenance.get("schema") != "intrmotiv/figure2d-shared-mask-source/v1" or
            provenance.get("shared_mask_bins") != 318 or
            provenance.get("shared_mask_sha256") != SHARED_MASK_SHA256):
        raise ValueError("Figure 2D shared-mask provenance changed")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row["family"], int(row["seed"])): row for row in rows}
    expected_keys = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if len(rows) != 15 or set(keyed) != expected_keys:
        raise ValueError("Figure 2D requires five unique pairs in each family")
    for (family, _), row in keyed.items():
        display, intervention, graph_applicable = FAMILIES[family]
        if (row["display_name"], row["intervention_factor"],
                row["graph_edges_applicable"]) != (display, intervention, graph_applicable):
            raise ValueError(f"Inconsistent Figure 2D intervention: {family}")
        if (int(row["shared_mask_bins"]) != 318 or
                row["shared_mask_sha256"] != SHARED_MASK_SHA256):
            raise ValueError(f"Figure 2D mask differs for {family}/{row['seed']}")
        for column in METRIC_COLUMNS.values():
            temporal = float(row[f"temporal_{column}"])
            constant = float(row[f"constant_{column}"])
            delta = float(row[f"delta_{column}"])
            if (not all(map(math.isfinite, (temporal, constant, delta))) or
                    not math.isclose(delta, temporal - constant, abs_tol=1e-12)):
                raise ValueError(f"Invalid {family}/{row['seed']}/{column} pair")

    with DIFFERENCES.open(newline="") as stream:
        prior = list(csv.DictReader(stream))
    prior_values = {
        (row["family"], int(row["seed"].split("S")[-1]), row["metric"]): float(row["delta"])
        for row in prior if row["arm"] == "pull"
    }
    for (family, seed), row in keyed.items():
        if family == "C01":
            continue
        for metric, column in METRIC_COLUMNS.items():
            old = prior_values[(family, seed, metric)]
            new = float(row[f"delta_{column}"])
            if not math.isclose(old, new, abs_tol=1e-12):
                raise ValueError(f"Existing {family}/{seed}/{metric} point changed")
    return keyed


def _insert_once(svg: str, pattern: str, insertion: str, *, description: str) -> str:
    svg, count = re.subn(pattern, lambda match: match.group(0) + insertion,
                         svg, count=1)
    if count != 1:
        raise ValueError(f"Expected one {description}")
    return svg


def shift_two_category_component(svg: str, prefix: str) -> str:
    """Make a third x slot while keeping UG/FG y coordinates and glyph sizes."""
    for index, (old_x, new_x) in enumerate(zip(OLD_CENTERS, (UG_X, FG_X)), 1):
        shift = new_x - old_x
        identifier = f"{prefix}_xtick_{index}"
        svg, count = re.subn(
            rf'(<g\s+id="{identifier}")',
            rf'\1 transform="translate({shift:.6f},0)"', svg, count=1)
        if count != 1:
            raise ValueError(f"Expected one x tick: {identifier}")
        identifier = f"{prefix}_PathCollection_{index}"
        next_identifier = (f"{prefix}_PathCollection_2" if index == 1
                           else f"{prefix}_patch_5")
        pattern = (rf'(<g\s+id="{identifier}">)(.*?)'
                   rf'(?=<g\s+id="{next_identifier}")')

        def remap_markers(match: re.Match[str]) -> str:
            body, count = re.subn(
                r'(<use\b[^>]*\bx=")([0-9.]+)(")',
                lambda point: (point.group(1) +
                               f"{new_x + BOX_WIDTH_SCALE * (float(point.group(2)) - old_x):.6f}" +
                               point.group(3)),
                match.group(2))
            if count != 5:
                raise ValueError(f"Expected five paired markers: {identifier}")
            return match.group(1) + body

        svg, count = re.subn(pattern, remap_markers, svg, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError(f"Expected one marker collection: {identifier}")
        box_groups = (("patch_5", 7, 8, 9, 10, 15) if index == 1 else
                      ("patch_6", 11, 12, 13, 14, 16))
        for part in box_groups:
            suffix = f"line2d_{part}" if isinstance(part, int) else part
            identifier = f"{prefix}_{suffix}"
            offset = new_x - BOX_WIDTH_SCALE * old_x
            transform = f"matrix({BOX_WIDTH_SCALE:.8f},0,0,1,{offset:.8f},0)"
            svg, count = re.subn(rf'(<g\s+id="{identifier}")',
                                 rf'\1 transform="{transform}"', svg, count=1)
            if count != 1:
                raise ValueError(f"Expected one editable box group: {identifier}")

    tick = (
        f'<g id="{prefix}_xtick_ng">'
        f'<path d="M {NG_X:.6f},106.56 V 108.56" '
        'style="fill:none;stroke:#000000;stroke-width:0.8"/>'
        f'<text x="{NG_X:.6f}" y="115.28625" '
        'style="font-size:9px;font-family:Arial;text-anchor:middle">NG</text>'
        '</g>')
    return _insert_once(svg, rf'<g\s+id="{prefix}_matplotlib\.axis_1">', tick,
                        description=f"x axis for {prefix}")


def add_ng_na_to_edge_panels(svg: str) -> str:
    """Keep NG graph edges explicitly inapplicable in C and D."""
    for row in ("push", "pull"):
        prefix = f"{row}_graph_edges"
        svg = shift_two_category_component(svg, prefix)
        unavailable = (
            f'<g id="{prefix}_ng_unavailable" aria-label="NG: no goal graph">'
            f'<rect x="{NG_X-7:.6f}" y="77.9" width="14" height="9.5" '
            'fill="white" stroke="none"/>'
            f'<text x="{NG_X:.6f}" y="85.2" '
            'style="font-size:7px;font-family:Arial;text-anchor:middle;fill:#68757d">'
            'N/A</text></g>')
        svg, count = re.subn(rf'(?=<g\s+id="{prefix}_PathCollection_1"[^>]*>)',
                             unavailable, svg, count=1)
        if count != 1:
            raise ValueError(f"Expected one data group for {prefix}")
    return svg


def _y_scale(svg: str, prefix: str) -> tuple[float, float]:
    """Calibrate from the retained zero and positive y tick positions."""
    root = ET.fromstring(svg)
    ticks = {}
    for index in (2, 3):
        identifier = f"{prefix}_ytick_{index}"
        group = next((node for node in root.iter() if node.get("id") == identifier), None)
        if group is None:
            raise ValueError(f"Missing y tick {identifier}")
        mark = next(node for node in group.iter() if node.tag.endswith("}use"))
        label = next(node for node in group.iter() if node.tag.endswith("}text"))
        ticks[index] = (float("".join(label.itertext())), float(mark.get("y")))
    (zero, zero_y), (positive, positive_y) = ticks[2], ticks[3]
    if abs(zero) > 1e-12 or positive <= 0 or not positive_y < zero_y:
        raise ValueError(f"Unexpected y-axis calibration for {prefix}")
    return zero_y, (positive_y - zero_y) / positive


def add_ng_pull_representation(
    svg: str, *, source_path: Path = PAIRS,
) -> tuple[str, list[dict[str, object]]]:
    """Draw three-category D plots using the audited NG five-seed pairs."""
    pairs = checked_pairs(source_path)
    summaries = []
    for metric, column in METRIC_COLUMNS.items():
        prefix = f"pull_{metric}"
        zero_y, slope = _y_scale(svg, prefix)
        values = [float(pairs[("C01", seed)][f"delta_{column}"]) for seed in SEEDS]
        coordinates = [zero_y + slope * value for value in values]
        if any(not 59.04 <= y <= 106.56 for y in coordinates):
            raise ValueError(f"NG values outside existing {metric} axis")
        svg = shift_two_category_component(svg, prefix)
        markers = [f'<g id="{prefix}_ng_points" aria-label="NG paired seed differences">']
        for index, (seed, y) in enumerate(zip(SEEDS, coordinates)):
            x = NG_X + 0.5995 * (index - 2)
            markers.append(
                f'<circle id="{prefix}_ng_seed_{seed}" cx="{x:.6f}" cy="{y:.6f}" '
                f'r="1.936492" fill="{NG_SOURCE_COLOR}" fill-opacity="0.7"/>')
        markers.append('</g>')
        svg, count = re.subn(rf'(?=<g\s+id="{prefix}_PathCollection_1"[^>]*>)',
                             ''.join(markers), svg, count=1)
        if count != 1:
            raise ValueError(f"Expected data group for {prefix}")

        low, q1, median, q3, high = box_summary(values, "tukey")
        y = lambda value: zero_y + slope * value
        left, right = NG_X - BOX_WIDTH / 2, NG_X + BOX_WIDTH / 2
        cap_left, cap_right = NG_X - BOX_WIDTH / 4, NG_X + BOX_WIDTH / 4
        box = (
            f'<g id="{prefix}_ng_box" aria-label="NG Tukey box and whiskers">'
            f'<path d="M {NG_X:.6f},{y(q1):.6f} V {y(low):.6f} '
            f'M {NG_X:.6f},{y(q3):.6f} V {y(high):.6f} '
            f'M {cap_left:.6f},{y(low):.6f} H {cap_right:.6f} '
            f'M {cap_left:.6f},{y(high):.6f} H {cap_right:.6f}" '
            f'fill="none" stroke="{NG_SOURCE_COLOR}" stroke-width="1"/>'
            f'<rect x="{left:.6f}" y="{y(q3):.6f}" width="{BOX_WIDTH:.6f}" '
            f'height="{y(q1)-y(q3):.6f}" fill="white" '
            f'stroke="{NG_SOURCE_COLOR}" stroke-width="1"/>'
            f'<path d="M {left:.6f},{y(median):.6f} H {right:.6f}" '
            f'fill="none" stroke="{NG_SOURCE_COLOR}" stroke-width="1.2"/>'
            '</g>')
        svg, count = re.subn(rf'(?=<g\s+id="{prefix}_patch_5"[^>]*>)',
                             box, svg, count=1)
        if count != 1:
            raise ValueError(f"Expected box group for {prefix}")
        summaries.append({
            "arm": "pull", "metric": metric, "family": "C01",
            "sample_count": 5, "positive_count": sum(value > 0 for value in values),
            "outlier_count": sum(value < low or value > high for value in values),
            "whisker_low": low, "whisker_high": high,
            "mean_delta": sum(values) / len(values),
        })
    return svg, summaries
