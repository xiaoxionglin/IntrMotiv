"""Render compact alternatives for one selector feeding policy and pull.

The SVGs are schematic: NG omits the selector output and opens the pull gate;
UG and FG differ only in how they choose g. Their gate tests the destination j
of the DG transition, not its source i.
"""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from render_goal_policy_icons import (  # noqa: E402
    INK, PALE, PULL, path, svg_document, validate_font,
    write_and_export,
)


def rect(x: float, y: float, width: float, height: float, *,
         stroke: str = INK, fill: str = "white", radius: float = 0.8) -> str:
    return (f'<rect x="{x:g}" y="{y:g}" width="{width:g}" '
            f'height="{height:g}" rx="{radius:g}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="0.42"/>')


def label(x: float, y: float, value: str, size: float = 3.3, *,
          color: str = INK, anchor: str = "middle", weight: str = "normal") -> str:
    return (f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
            f'font-family="DejaVu Sans" font-size="{size:g}" '
            f'font-weight="{weight}" fill="{color}">{value}</text>')


def arrow(data: str, *, color: str = INK, width: float = 0.55) -> str:
    return path(data, color, width)


def transition_label(x: float, y: float) -> str:
    """Keep the source and destination indices native, editable SVG text."""
    return (f'<text x="{x:g}" y="{y:g}" font-family="DejaVu Sans" '
            f'font-size="4.2" fill="{INK}">Δ'
            '<tspan font-size="2.7" baseline-shift="sub">i→j</tspan>'
            '</text>')


def branch() -> str:
    """One goal token branches to the policy and the destination gate."""
    body = rect(1.5, 3.0, 17, 6.6, fill=PALE)
    body += label(10, 7.2, "select", 4.3, weight="bold")
    body += rect(38, 3.2, 9.3, 6.2, fill=PALE)
    body += label(42.65, 7.5, "π", 5.3)
    body += arrow("M18.5,6.3 H37.8 M37.1,5.65 L37.8,6.3 L37.1,6.95")
    body += label(25.8, 5.0, "g", 3.6)
    body += arrow("M29.1,6.3 V11.1 H38.3 V12.5")
    body += rect(32, 12.5, 13, 5.8, stroke=PULL)
    body += label(38.5, 16.5, "j = g", 4.2)
    body += transition_label(2.3, 16.2)
    body += arrow("M15.3,15.4 H31.8 M31.1,14.75 L31.8,15.4 L31.1,16.05")
    body += arrow("M45.2,15.4 H51.5 M50.8,14.7 L51.5,15.4 L50.8,16.1",
                  color=PULL, width=0.68)
    return svg_document(body, 53, 20)


def stacked() -> str:
    """A vertical split makes the policy and pull uses of g explicit."""
    body = rect(2, 6.2, 16.4, 7.2, fill=PALE)
    body += label(10.2, 10.7, "select", 4.3, weight="bold")
    body += label(23.7, 5.0, "g", 3.6)
    body += arrow("M18.5,9.8 H25.3 V5.8 H35.8 M35.1,5.1 L35.8,5.8 L35.1,6.5")
    body += arrow("M25.3,9.8 V11.2 H41.1 V12.8 M40.4,12.1 L41.1,12.8 L41.8,12.1")
    body += rect(36.1, 2.8, 10.0, 6.0, fill=PALE)
    body += label(41.1, 7.1, "π", 5.3)
    body += rect(34.5, 13.0, 12.5, 6.0, stroke=PULL)
    body += label(40.75, 17.0, "j = g", 4.2)
    body += transition_label(2.3, 18.3)
    body += arrow("M15.3,17.5 H34.3 M33.6,16.8 L34.3,17.5 L33.6,18.2")
    body += arrow("M47.2,16 H51.5 M50.8,15.3 L51.5,16 L50.8,16.7",
                  color=PULL, width=0.68)
    return svg_document(body, 53, 20)


OPTIONS = {
    "01_branch": ("Branch", branch),
    "02_stacked": ("Stacked", stacked),
}


def overview() -> str:
    body = '<rect width="82" height="74" fill="white"/>'
    body += label(2, 7, "One goal selector, two uses", 4.1,
                  anchor="start", weight="bold")
    for index, (name, (title, factory)) in enumerate(OPTIONS.items()):
        y = 12 + index * 23
        body += label(2, y + 12, title, 3.4, anchor="start")
        body += f'<g id="{name}" transform="translate(27,{y:g})">'
        body += factory().split('>', 1)[1].rsplit('</svg>', 1)[0]
        body += '</g>'
    body += label(2, 58, 'NG: no goal; all j pass', 3.0, anchor="start")
    body += label(2, 62.5, 'UG: uniform DG goal; j = g', 3.0, anchor="start")
    body += label(2, 67, 'FG: frontier DG goal; j = g', 3.0, anchor="start")
    body += label(2, 71.5, 'Blue arrow: pull-eligible signal', 2.8,
                  color=PULL, anchor="start")
    return svg_document(body, 82, 74)


def main() -> None:
    validate_font()
    HERE.mkdir(parents=True, exist_ok=True)
    for name, (_, factory) in OPTIONS.items():
        write_and_export(HERE / f"{name}.svg", factory(), png_width=1800)
    write_and_export(HERE / "overview.svg", overview(), png_width=2000)


if __name__ == "__main__":
    main()
