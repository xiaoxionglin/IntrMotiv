"""Render data-free, print-size DG spatial-information icon candidates.

Each 10 × 20 mm icon compares one DG unit under equal spatial occupancy and
equal total activation. The upper example has greater location specificity.
Nothing here represents a measured condition or a temporal-credit effect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE.parents[1]
PALETTE_MANIFEST = FIGURE_DIR / "figure_palette_candidates" / "manifest.json"
SELECTED_FIGURE = FIGURE_DIR / "Figure2.svg"

# Reuse the existing icon export and SVG wrapper so page size and PDF/PNG
# settings stay in sync with the other Figure 2 metric icons.
sys.path.insert(0, str(FIGURE_DIR))
from build_figure2_metric_icons import INK, MUTED, svg_document, write_and_export  # noqa: E402


PAGE_WIDTH = 10 * 72 / 25.4
FRAME_X = 4.5
FRAME_WIDTH = 19.3
FRAME_TOPS = (2.0, 35.4)
GRID = "#d5dde1"
PALE = "#f8f9fa"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_dg_red() -> tuple[str, str]:
    manifest = json.loads(PALETTE_MANIFEST.read_text())
    selected = manifest["selected_palette"]
    return selected, manifest["palettes"][selected]["colors"]["dg_red"]


def common_elements(dg_red: str) -> str:
    """Two equal arenas, the same DG-unit mark, and an upward comparison arrow."""
    parts = []
    for y in FRAME_TOPS:
        parts.append(
            f'<rect x="{FRAME_X}" y="{y}" width="{FRAME_WIDTH}" '
            f'height="{FRAME_WIDTH}" rx="0.8" fill="white" '
            f'stroke="{MUTED}" stroke-width="0.65"/>'
        )
        parts.append(
            f'<circle cx="25.45" cy="{y + 2.6:.2f}" r="1.35" '
            f'fill="{dg_red}" stroke="{INK}" stroke-width="0.35"/>'
        )
    parts.append(
        f'<path d="M 14.15,32.8 V 25.0 M 11.8,27.4 L 14.15,25.0 '
        f'L 16.5,27.4" fill="none" stroke="{MUTED}" stroke-width="1.05"/>'
    )
    return "".join(parts)


def rate_map(dg_red: str, *, id_prefix: str = "") -> str:
    """One localized high-rate patch versus a spatially uniform low rate.

    Four fully colored cells above and sixteen quarter-opacity cells below
    depict equal summed activation on the same 4 × 4 occupancy grid.
    """
    parts = [common_elements(dg_red)]
    for row_index, y in enumerate(FRAME_TOPS):
        for r in range(4):
            for c in range(4):
                concentrated = row_index == 0 and r in (1, 2) and c in (2, 3)
                opacity = 1.0 if concentrated else 0.25 if row_index == 1 else 0.0
                x_cell = 5.45 + c * 4.4
                y_cell = y + 0.95 + r * 4.4
                parts.append(
                    f'<rect x="{x_cell:.2f}" y="{y_cell:.2f}" '
                    f'width="4.0" height="4.0" fill="{PALE}" '
                    f'stroke="{GRID}" stroke-width="0.30"/>'
                )
                if opacity:
                    parts.append(
                        f'<rect x="{x_cell:.2f}" y="{y_cell:.2f}" '
                        f'width="4.0" height="4.0" fill="{dg_red}" '
                        f'fill-opacity="{opacity:g}"/>'
                    )
    return "".join(parts)


def dg_events(dg_red: str, *, id_prefix: str = "") -> str:
    """Nine events from the same unit in one bin versus across nine bins."""
    parts = [common_elements(dg_red)]
    for y in FRAME_TOPS:
        for k in (1, 2):
            offset = FRAME_WIDTH * k / 3
            parts.append(
                f'<path d="M {FRAME_X + offset:.2f},{y:.2f} '
                f'V {y + FRAME_WIDTH:.2f} M {FRAME_X:.2f},{y + offset:.2f} '
                f'H {FRAME_X + FRAME_WIDTH:.2f}" '
                f'fill="none" stroke="{GRID}" stroke-width="0.4"/>'
            )
    top_points = ((12.75 + 1.4 * c, 9.0 + 1.4 * r)
                  for r in range(3) for c in range(3))
    bottom_points = ((FRAME_X + FRAME_WIDTH * (c + 0.5) / 3,
                      FRAME_TOPS[1] + FRAME_WIDTH * (r + 0.5) / 3)
                     for r in range(3) for c in range(3))
    for x, y in (*top_points, *bottom_points):
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="0.72" '
            f'fill="{dg_red}" stroke="{INK}" stroke-width="0.17"/>'
        )
    return "".join(parts)


def position_profile(dg_red: str, *, id_prefix: str = "") -> str:
    """Four position-bin response bars with equal sum in both examples."""
    parts = [common_elements(dg_red)]
    profiles = ((0.05, 0.05, 0.05, 0.85), (0.25, 0.25, 0.25, 0.25))
    for y, profile in zip(FRAME_TOPS, profiles):
        baseline = y + 14.8
        parts.append(
            f'<path d="M 6.4,{baseline:.2f} H 21.9" '
            f'fill="none" stroke="{INK}" stroke-width="0.55"/>'
        )
        for index, rate in enumerate(profile):
            x = 7.0 + index * 4.05
            height = 12.0 * rate
            parts.append(
                f'<rect x="{x:.2f}" y="{baseline - height:.2f}" '
                f'width="2.6" height="{height:.2f}" fill="{dg_red}"/>'
                f'<rect x="{x + 0.55:.2f}" y="{y + 16.5:.2f}" '
                f'width="1.5" height="1.5" fill="{PALE}" '
                f'stroke="{MUTED}" stroke-width="0.35"/>'
            )
    return "".join(parts)


def soft_dg_maps(dg_red: str, *, id_prefix: str = "field") -> str:
    """One DG map per row in the overlap and reliability icons' soft style.

    The upper field is narrow. The lower map has a broad weak field rather
    than a perfectly uniform fill. The SVG opacities give approximately equal
    mean activity on the two equal-area maps.
    """
    narrow_id = f"{id_prefix}-narrow-dg"
    broad_id = f"{id_prefix}-broad-dg"
    parts = [
        '<defs>',
        f'<radialGradient id="{narrow_id}">',
        f'<stop offset="0" stop-color="{dg_red}" stop-opacity="0.85"/>',
        f'<stop offset="0.5" stop-color="{dg_red}" stop-opacity="0.35"/>',
        f'<stop offset="1" stop-color="{dg_red}" stop-opacity="0"/>',
        '</radialGradient>',
        f'<radialGradient id="{broad_id}">',
        f'<stop offset="0" stop-color="{dg_red}" stop-opacity="0.20"/>',
        f'<stop offset="0.6" stop-color="{dg_red}" stop-opacity="0.08"/>',
        f'<stop offset="1" stop-color="{dg_red}" stop-opacity="0"/>',
        '</radialGradient></defs>',
    ]
    card_size = 14.5
    card_x = (PAGE_WIDTH - card_size) / 2
    for row, y in enumerate((2.3, 35.0)):
        baseline_opacity = 0.05 if row == 0 else 0.07
        parts.append(
            f'<rect x="{card_x:.2f}" y="{y}" width="{card_size}" '
            f'height="{card_size}" fill="{dg_red}" '
            f'fill-opacity="{baseline_opacity:g}"/>'
        )
        parts.append(
            f'<circle cx="{PAGE_WIDTH / 2:.2f}" cy="{y + card_size / 2:.2f}" '
            f'r="{4.0 if row == 0 else 6.5}" '
            f'fill="url(#{narrow_id if row == 0 else broad_id})"/>'
        )
        parts.extend((
            f'<rect x="{card_x:.2f}" y="{y}" width="{card_size}" '
            f'height="{card_size}" fill="none" '
            f'stroke="{MUTED}" stroke-width="0.6"/>',
            f'<circle cx="{card_x + card_size + 2.0:.2f}" '
            f'cy="{y + 2.7:.2f}" r="1.25" fill="{dg_red}" '
            f'stroke="{INK}" stroke-width="0.35"/>',
        ))
    parts.append(
        f'<path d="M 14.17,31.5 V 25.2 M 11.8,27.6 L 14.17,25.2 '
        f'L 16.54,27.6" fill="none" stroke="{MUTED}" stroke-width="1.05"/>'
    )
    return "".join(parts)


CANDIDATES = (
    ("01_rate_map", "Rate map", rate_map),
    ("02_dg_events", "DG events", dg_events),
    ("03_position_profile", "Position profile", position_profile),
    ("04_soft_dg_maps", "Soft DG maps", soft_dg_maps),
)


def comparison_sheet(dg_red: str) -> str:
    """Show enlarged vectors and true-size copies of every candidate."""
    width_pt = 160 * 72 / 25.4
    height_pt = 76 * 72 / 25.4
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="160mm" '
        f'height="76mm" viewBox="0 0 {width_pt:.4f} {height_pt:.4f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width_pt / 2:.2f}" y="15" text-anchor="middle" '
        f'font-family="Arial" font-size="12" fill="{INK}">'
        'One DG unit: spatial information'
        '</text>',
        f'<text x="{width_pt / 2:.2f}" y="30" text-anchor="middle" '
        f'font-family="Arial" font-size="12" fill="{MUTED}">'
        'Equal occupancy and total activity; top is more selective'
        '</text>',
    ]
    for index, (_, label, factory) in enumerate(CANDIDATES):
        center_x = 57.0 + index * 113.0
        enlarged_body = factory(dg_red, id_prefix=f"{index}-enlarged")
        actual_body = factory(dg_red, id_prefix=f"{index}-actual")
        parts.extend((
            f'<text x="{center_x:.2f}" y="46" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="{INK}">{label}</text>',
            f'<g transform="translate({center_x - PAGE_WIDTH * 0.7:.2f},52) '
            f'scale(1.4)">{enlarged_body}</g>',
            f'<g transform="translate({center_x - PAGE_WIDTH / 2:.2f},139)">'
            f'{actual_body}</g>',
            f'<text x="{center_x:.2f}" y="209" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="{MUTED}">10 × 20 mm</text>',
        ))
    parts.append('</svg>\n')
    return "".join(parts)


def build() -> None:
    selected_palette, dg_red = selected_dg_red()
    for stem, _, factory in CANDIDATES:
        write_and_export(HERE / f"{stem}.svg", svg_document(
            factory(dg_red, id_prefix=stem), 20))
    write_and_export(HERE / "comparison.svg", comparison_sheet(dg_red), png_width=2200)
    manifest = {
        "schema": "intrmotiv/figure2-spatial-information-icons/v1",
        "type": "data-free illustration; no measured result direction",
        "size_mm": [10, 20],
        "selected_palette": selected_palette,
        "single_dg_identity_color": dg_red,
        "example_order": "upper: higher raw spatial information; lower: lower",
        "controlled_assumptions": ["equal spatial occupancy", "equal total activation"],
        "reference_sha256": {
            str(SELECTED_FIGURE.relative_to(FIGURE_DIR)): sha256(SELECTED_FIGURE),
            str(PALETTE_MANIFEST.relative_to(FIGURE_DIR)): sha256(PALETTE_MANIFEST),
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    build()
