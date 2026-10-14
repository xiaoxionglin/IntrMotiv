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
    DG_RED, DG_YELLOW, INK, MUTED, PALE, PULL, circle, path, svg_document, validate_font,
    write_and_export,
)

TEMPORAL = "#257873"  # Rounded temporal-separation frame in architecture v3.
READOUT = "#136199"   # Policy readout wedge in architecture v3.


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


def policy_wedge(x: float, y: float, width: float, height: float) -> str:
    """Use the original readout's blue, tapered module outline."""
    right = x + width
    bottom = y + height
    shape = (f'<path d="M{x:g},{y:g} L{right:g},{y+1.0:g} '
             f'L{right:g},{bottom-1.0:g} L{x:g},{bottom:g} Z" '
             f'fill="white" stroke="{READOUT}" stroke-width="0.55" '
             'stroke-linejoin="round"/>')
    return shape + label(x + width / 2, y + height * 0.7, "π", 5.3)


def delta_frame(x: float, y: float, width: float, height: float) -> str:
    """Use the original temporal-separation panel's teal rounded outline."""
    return rect(x, y, width, height, stroke=TEMPORAL, radius=1.7)


def transition_label(x: float, y: float) -> str:
    """Keep the source and destination indices native, editable SVG text."""
    return (f'<text x="{x:g}" y="{y:g}" font-family="DejaVu Sans" '
            f'font-size="4.2" fill="{INK}">Δ'
            '<tspan font-size="2.7" baseline-shift="sub">i→j</tspan>'
            '</text>')


def branch() -> str:
    """One goal token branches to the policy and the destination gate."""
    body = rect(1.5, 3.0, 17, 6.6, fill=PALE)
    body += label(10, 7.2, "Goal", 4.3, weight="bold")
    body += policy_wedge(38, 3.2, 9.3, 6.2)
    body += arrow("M18.5,6.3 H37.8 M37.1,5.65 L37.8,6.3 L37.1,6.95")
    body += label(25.8, 5.0, "g", 3.6)
    body += arrow("M29.1,6.3 V11.1 H38.3 V12.5")
    body += rect(32, 12.5, 13, 5.8, stroke=PULL)
    body += label(38.5, 16.5, "j = g", 4.2)
    body += delta_frame(1.3, 12.1, 12.3, 6.5)
    body += transition_label(2.3, 16.2)
    body += arrow("M15.3,15.4 H31.8 M31.1,14.75 L31.8,15.4 L31.1,16.05")
    body += arrow("M45.2,15.4 H51.5 M50.8,14.7 L51.5,15.4 L50.8,16.1",
                  color=PULL, width=0.68)
    return svg_document(body, 53, 20)


def stacked() -> str:
    """A vertical split makes the policy and pull uses of g explicit."""
    body = rect(2, 6.2, 16.4, 7.2, fill=PALE)
    body += label(10.2, 10.7, "Goal", 4.3, weight="bold")
    body += label(23.7, 5.0, "g", 3.6)
    body += arrow("M18.5,9.8 H25.3 V5.8 H35.8 M35.1,5.1 L35.8,5.8 L35.1,6.5")
    body += arrow("M25.3,9.8 V11.2 H41.1 V12.8 M40.4,12.1 L41.1,12.8 L41.8,12.1")
    body += policy_wedge(36.1, 2.8, 10.0, 6.0)
    body += rect(34.5, 13.0, 12.5, 6.0, stroke=PULL)
    body += label(40.75, 17.0, "j = g", 4.2)
    body += delta_frame(1.3, 13.8, 12.3, 5.9)
    body += transition_label(2.3, 18.3)
    body += arrow("M15.3,17.5 H34.3 M33.6,16.8 L34.3,17.5 L33.6,18.2")
    body += arrow("M47.2,16 H51.5 M50.8,15.3 L51.5,16 L50.8,16.7",
                  color=PULL, width=0.68)
    return svg_document(body, 53, 20)


def readout_elbow(*, event_dots: bool) -> str:
    """Keep the original Δ-below-readout placement and upward pull elbow."""
    body = rect(1.5, 2.1, 14.2, 6.0, fill=PALE)
    body += label(8.6, 6.3, "Goal", 4.3, weight="bold")
    body += policy_wedge(39.2, 2.0, 10.0, 6.1)
    body += arrow("M15.9,5.1 H39.0 M38.3,4.4 L39,5.1 L38.3,5.8")
    body += label(22.4, 3.9, "g", 3.6)
    body += arrow("M22.8,5.1 V10.8 H30.0 V12.3 ")

    body += rect(24.8, 12.5, 12.6, 5.8, stroke=PULL)
    body += label(31.1, 16.5, "j = g", 4.2)
    if event_dots:
        body += delta_frame(1.2, 12.2, 20.2, 7.4)
        body += label(2.3, 16.8, "Δ", 4.8, anchor="start")
        body += circle(10.0, 15.4, 1.25, DG_RED)
        body += circle(16.5, 15.4, 1.25, DG_YELLOW)
        body += arrow("M11.5,15.4 H14.8 M14.1,14.7 L14.8,15.4 L14.1,16.1",
                      width=0.42)
        body += arrow("M19.2,13.9 V17.4 M18.6,16.8 L19.2,17.4 L19.8,16.8",
                      width=0.42)
        body += label(10.0, 19.0, "i", 2.7)
        body += label(16.5, 19.0, "j", 2.7)
        input_start = 21.6
    else:
        body += delta_frame(1.2, 12.2, 13.3, 6.7)
        body += transition_label(2.3, 16.4)
        input_start = 15.3
    body += arrow(f"M{input_start:g},15.4 H24.6 "
                  "M23.9,14.7 L24.6,15.4 L23.9,16.1")
    body += arrow("M37.6,15.4 H46.0 V8.6 "
                  "M45.3,9.3 L46,8.6 L46.7,9.3", color=PULL, width=0.68)
    return svg_document(body, 53, 20)


def labeled_elbow() -> str:
    return readout_elbow(event_dots=False)


def dg_event_elbow() -> str:
    return readout_elbow(event_dots=True)


def modulated_elbow(*, explicit_j_input: bool) -> str:
    """Keep the Δ-to-policy arrow intact; modulate it from a side branch."""
    body = rect(1.5, 2.1, 14.2, 6.0, fill=PALE)
    body += label(8.6, 6.3, "Goal", 4.3, weight="bold")
    body += policy_wedge(39.2, 2.0, 10.0, 6.1)
    body += arrow("M15.9,5.1 H39.0 M38.3,4.4 L39,5.1 L38.3,5.8")
    body += label(22.4, 3.9, "g", 3.6)

    body += delta_frame(1.2, 12.2, 20.2, 7.4)
    body += label(2.3, 16.8, "Δ", 4.8, anchor="start")
    body += circle(10.0, 15.4, 1.25, DG_RED)
    body += circle(16.5, 15.4, 1.25, DG_YELLOW)
    body += arrow("M11.5,15.4 H14.8 M14.1,14.7 L14.8,15.4 L14.1,16.1",
                  width=0.42)
    body += arrow("M19.2,13.9 V17.4 M18.6,16.8 L19.2,17.4 L19.8,16.8",
                  width=0.42)
    body += label(10.0, 19.0, "i", 2.7)
    body += label(16.5, 19.0, "j", 2.7)

    # The blue route is one SVG path from Δ to the readout, including its elbow.
    body += arrow("M21.5,15.4 H46 V8.6 M45.3,9.3 L46,8.6 L46.7,9.3",
                  color=PULL, width=0.68)
    body += rect(28, 8.3, 12, 4.2, stroke=INK, fill="white", radius=1.8)
    body += label(34, 11.3, "j = g", 3.4)
    body += arrow("M22.8,5.1 V7.5 H34 V8.2", width=0.43)
    if explicit_j_input:
        body += arrow("M16.5,13.9 V10.4 H27.8 "
                      "M27.2,9.9 L27.8,10.4 L27.2,10.9", width=0.36)
    body += path("M34,12.5 V13.7", MUTED, 0.45, dash="0.55,0.45")
    body += arrow("M33.5,13.2 L34,13.7 L34.5,13.2", color=MUTED,
                  width=0.38)
    body += circle(34, 15.4, 1.65, "white", stroke=PULL, width=0.55)
    body += label(34, 16.6, "×", 3.9, color=PULL)
    return svg_document(body, 53, 20)


def implicit_mask() -> str:
    return modulated_elbow(explicit_j_input=False)


def explicit_mask() -> str:
    return modulated_elbow(explicit_j_input=True)


OPTIONS = {
    "01_branch": ("Branch", branch),
    "02_stacked": ("Stacked", stacked),
    "03_labeled_elbow": ("Readout elbow", labeled_elbow),
    "04_dg_event_elbow": ("DG event elbow", dg_event_elbow),
}

GATE_OPTIONS = {
    "05_side_gate": ("Side gate", implicit_mask),
    "06_two_input_gate": ("Two inputs", explicit_mask),
}


def overview() -> str:
    body = '<rect width="82" height="120" fill="white"/>'
    body += label(2, 7, "One goal selector, two uses", 4.1,
                  anchor="start", weight="bold")
    for index, (name, (title, factory)) in enumerate(OPTIONS.items()):
        y = 12 + index * 23
        body += label(2, y + 12, title, 3.4, anchor="start")
        body += f'<g id="{name}" transform="translate(27,{y:g})">'
        body += factory().split('>', 1)[1].rsplit('</svg>', 1)[0]
        body += '</g>'
    body += label(2, 103, 'NG: no goal; all j pass', 3.0, anchor="start")
    body += label(2, 107.5, 'UG: uniform DG goal; j = g', 3.0, anchor="start")
    body += label(2, 112, 'FG: frontier DG goal; j = g', 3.0, anchor="start")
    body += label(2, 116.5, 'Blue arrow: pull-eligible signal', 2.8,
                  color=PULL, anchor="start")
    return svg_document(body, 82, 120)


def gate_comparison() -> str:
    body = '<rect width="82" height="71" fill="white"/>'
    body += label(2, 7, "Pull path with a separate goal gate", 3.6,
                  anchor="start", weight="bold")
    for index, (name, (title, factory)) in enumerate(GATE_OPTIONS.items()):
        y = 12 + index * 23
        body += label(2, y + 12, title, 3.4, anchor="start")
        body += f'<g id="{name}" transform="translate(27,{y:g})">'
        body += factory().split('>', 1)[1].rsplit('</svg>', 1)[0]
        body += '</g>'
    body += label(2, 60, 'Blue: pull from Δ to policy; × applies the mask',
                  2.9, color=PULL, anchor="start")
    body += label(2, 65, 'Side control: j = g (NG uses an always-open gate)',
                  2.9, anchor="start")
    return svg_document(body, 82, 71)


def main() -> None:
    validate_font()
    HERE.mkdir(parents=True, exist_ok=True)
    for name, (_, factory) in {**OPTIONS, **GATE_OPTIONS}.items():
        write_and_export(HERE / f"{name}.svg", factory(), png_width=1800)
    write_and_export(HERE / "overview.svg", overview(), png_width=2000)
    write_and_export(HERE / "gate_comparison.svg", gate_comparison(), png_width=2000)


if __name__ == "__main__":
    main()
