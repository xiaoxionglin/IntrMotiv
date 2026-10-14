"""Render compact, editable NG/UG/FG icon alternatives for Figure 1.

Each strip contains three icons in NG, UG, FG order. The same
policy input appears in every design; only the lower pull-eligibility motif
changes. FG changes goal selection, not the goal-matching pull rule.
"""

from __future__ import annotations

from pathlib import Path
import sys

from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1]
sys.path.insert(0, str(DRAFT))
from build_figure2_metric_icons import (  # noqa: E402
    DG_COLORS, INK, MUTED, write_and_export,
)


DG_BLUE, DG_RED, DG_YELLOW = DG_COLORS[:3]
PULL = "#236d9d"  # Match the blue pull path of architecture v3.
LIGHT = "#d9e1e5"
PALE = "#f7f9fa"
CONDITIONS = ("NG", "UG", "FG")
CELL_WIDTH_MM = 24
CELL_HEIGHT_MM = 20
STRIP_WIDTH_MM = CELL_WIDTH_MM * len(CONDITIONS)
COMPACT_WIDTH_MM = 53  # Fits the 2.09-inch right side of Figure 1.
COMPACT_HEIGHT_MM = CELL_HEIGHT_MM * COMPACT_WIDTH_MM / STRIP_WIDTH_MM


def validate_font() -> None:
    """Fail instead of silently substituting a bitmap font in the preview."""
    path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if not path.is_file() or path.suffix.lower() not in {".ttf", ".otf"}:
        raise RuntimeError(f"Scalable DejaVu Sans font unavailable: {path}")


def circle(x: float, y: float, radius: float, fill: str, *,
           stroke: str = INK, width: float = 0.35) -> str:
    return (f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width:g}"/>')


def path(data: str, color: str = INK, width: float = 0.45, *,
         dash: str = "") -> str:
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{data}" fill="none" stroke="{color}" '
            f'stroke-width="{width:g}"{pattern}/>')


def arrow(x: float, y_start: float, y_end: float, *, color: str = PULL) -> str:
    return path(f"M{x:g},{y_start:g} V{y_end:g} "
                f"M{x-0.72:g},{y_end-0.72:g} L{x:g},{y_end:g} "
                f"L{x+0.72:g},{y_end-0.72:g}", color, 0.55)


def selected_ring(x: float, y: float, radius: float = 1.85) -> str:
    return circle(x, y, radius, "none", stroke=INK, width=0.35)


def selector(condition: str) -> str:
    """Uniform candidates or an open graph leaf choose the red DG goal."""
    if condition == "NG":
        return (circle(11.4, 6.7, 1.28, "white", stroke=MUTED, width=0.32)
                + path("M10.5,7.6 L12.3,5.8", MUTED, 0.35))
    if condition == "UG":
        source = ''.join(circle(x, 6.7, 0.55, color, width=0.2)
                         for x, color in ((3.5, DG_BLUE), (5.0, DG_RED),
                                          (6.5, DG_YELLOW)))
        source += path("M7.4,6.7 H9.3 M8.7,6.1 L9.3,6.7 L8.7,7.3",
                       MUTED, 0.38)
    elif condition == "FG":
        source = (path("M3.3,6.7 L5.4,5.2 M3.3,6.7 L5.4,8.2 "
                       "M5.4,5.2 L7.2,5.2", MUTED, 0.42)
                  + circle(3.3, 6.7, 0.52, INK, width=0.2)
                  + circle(5.4, 8.2, 0.48, MUTED, width=0.2)
                  + circle(5.4, 5.2, 0.48, MUTED, width=0.2)
                  + circle(7.2, 5.2, 0.68, "white", stroke=INK, width=0.34)
                  + path("M7.9,5.2 L9.2,6.4 M8.6,6.0 L9.2,6.4 "
                         "L9.1,5.6", MUTED, 0.38))
    else:
        raise ValueError(condition)
    return (source + selected_ring(11.4, 6.7, 1.65)
            + circle(11.4, 6.7, 1.1, DG_RED)
            + path("M13.1,6.7 H16.8 M16.2,6.1 L16.8,6.7 L16.2,7.3",
                   INK, 0.43))


def policy_top(condition: str) -> str:
    """CA3 input is common; a goal token enters only UG and FG readouts."""
    return (
        path("M8.6,3.8 H16.8 M16.2,3.2 L16.8,3.8 L16.2,4.4", MUTED, 0.40)
        + '<path d="M17.2,2.6 L21.5,3.3 L21.5,7.8 L17.2,8.5 Z" '
          f'fill="{PALE}" stroke="{INK}" stroke-width="0.45"/>'
        + f'<text x="19.35" y="6.75" text-anchor="middle" '
          f'font-family="DejaVu Sans" font-size="4.2" fill="{INK}">π</text>'
        + selector(condition)
    )


def two_lanes(condition: str) -> str:
    dots = ''.join(circle(x, 11.7, 1.28, color)
                   for x, color in ((4.4, DG_BLUE), (12, DG_RED),
                                    (19.6, DG_YELLOW)))
    if condition != "NG":
        dots += selected_ring(12, 11.7)
    eligible = (4.4, 12, 19.6) if condition == "NG" else (12,)
    return dots + ''.join(arrow(x, 13.45, 17.2) for x in eligible)


def timeline(condition: str) -> str:
    line = path("M2.2,12.1 H21.9 M21.2,11.45 L21.9,12.1 L21.2,12.75",
                MUTED, 0.34)
    events = ((4.3, DG_BLUE), (9.4, DG_RED), (14.5, DG_YELLOW),
              (19.6, DG_RED))
    line += ''.join(circle(x, 12.1, 1.08, color) for x, color in events)
    if condition != "NG":
        line += selected_ring(9.4, 12.1, 1.55)
        line += selected_ring(19.6, 12.1, 1.55)
    eligible = (4.3, 9.4, 14.5, 19.6) if condition == "NG" else (9.4, 19.6)
    return line + ''.join(arrow(x, 13.7, 17.6) for x in eligible)


def goal_gate(condition: str) -> str:
    dots = ''.join(circle(4.2, y, 0.9, color)
                   for y, color in ((9.8, DG_BLUE), (12.0, DG_RED),
                                    (14.2, DG_YELLOW)))
    if condition != "NG":
        dots += selected_ring(4.2, 12.0, 1.35)
    gate = ('<rect x="15.2" y="9.7" width="3.7" height="4.6" rx="0.7" '
            f'fill="white" stroke="{PULL}" stroke-width="0.48"/>')
    if condition != "NG":
        gate += circle(17.05, 12.0, 0.73, DG_RED, width=0.2)
    for source_y, target_y in ((9.8, 10.7), (12.0, 12.0),
                               (14.2, 13.3)):
        active = condition == "NG" or source_y == 12.0
        target = 15.2 if active else 14.2
        gate += path(f"M5.35,{source_y:g} L{target:g},{target_y:g}",
                     PULL if active else LIGHT, 0.42 if active else 0.32)
    return (dots + gate
            + path("M18.9,12 H22.1 M21.4,11.3 L22.1,12 L21.4,12.7",
                   PULL, 0.62))


VARIANTS = {
    "01_two_lanes": ("Two lanes", two_lanes),
    "02_event_timeline": ("Timeline", timeline),
    "03_goal_gate": ("Goal gate", goal_gate),
}


def icon_group(condition: str, bottom_factory, x_offset: float,
               id_prefix: str = "") -> str:
    return (
        f'<g id="{id_prefix}{condition}" transform="translate({x_offset:g},0)" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f'<title>{condition}: policy goal input and pull-eligible DG events</title>'
        f'<rect x="0.45" y="0.45" width="23.1" height="19.1" rx="1" '
        f'fill="white" stroke="{LIGHT}" stroke-width="0.32"/>'
        + policy_top(condition) + bottom_factory(condition) + '</g>'
    )


def strip_body(bottom_factory, id_prefix: str = "") -> str:
    return ''.join(icon_group(condition, bottom_factory, i * CELL_WIDTH_MM,
                              id_prefix)
                   for i, condition in enumerate(CONDITIONS))


def svg_document(body: str, width_mm: float, height_mm: float, *,
                 view_width: float | None = None,
                 view_height: float | None = None) -> str:
    view_width = width_mm if view_width is None else view_width
    view_height = height_mm if view_height is None else view_height
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:g}mm" '
            f'height="{height_mm:g}mm" viewBox="0 0 {view_width:g} {view_height:g}">'
            f'{body}</svg>\n')


def overview() -> str:
    parts = ['<rect width="110" height="86" fill="white"/>',
             f'<text x="2" y="6.8" font-family="DejaVu Sans" '
             f'font-size="4.25" font-weight="bold" fill="{INK}">'
             'Goal conditioning and pull eligibility</text>']
    for i, condition in enumerate(CONDITIONS):
        parts.append(f'<text x="{46+24*i}" y="15.0" text-anchor="middle" '
                     f'font-family="DejaVu Sans" font-size="4.25" '
                     f'font-weight="bold" fill="{INK}">{condition}</text>')
    for i, (_, (label, factory)) in enumerate(VARIANTS.items()):
        y = 18.5 + 22 * i
        parts.append(f'<text x="2" y="{y+11:g}" font-family="DejaVu Sans" '
                     f'font-size="4.25" fill="{INK}">{label}</text>')
        parts.append(f'<g transform="translate(34,{y:g})">'
                     f'{strip_body(factory, id_prefix=f"variant-{i+1}-")}</g>')
    return svg_document(''.join(parts), 110, 86)


def main() -> None:
    validate_font()
    HERE.mkdir(parents=True, exist_ok=True)
    for name, (_, factory) in VARIANTS.items():
        write_and_export(HERE / f"{name}.svg",
                         svg_document(strip_body(factory), COMPACT_WIDTH_MM,
                                      COMPACT_HEIGHT_MM,
                                      view_width=STRIP_WIDTH_MM,
                                      view_height=CELL_HEIGHT_MM),
                         png_width=1800)
    write_and_export(HERE / "overview.svg", overview(), png_width=2200)


if __name__ == "__main__":
    main()
