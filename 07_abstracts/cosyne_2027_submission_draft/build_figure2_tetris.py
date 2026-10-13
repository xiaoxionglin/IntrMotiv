"""Reflow the author's Figure 2 sketch into a shallow 7.5-inch-wide figure.

All ten result plots and three mechanism drawings retain their original size.
The C15 seed-123 maps occupy the spare right side of the first row. The source
SVG remains untouched, and all result objects remain editable vector groups.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from build_figure2_fiddle_with_fields import (
    SOURCE, UNITS, fix_pdf_viewbox, map_data_uri, repair_readout_label,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Figure2_tetris.svg"
SIDEBAR_OUTPUT = HERE / "Figure2_tetris_sidebar.svg"


def move_group(svg: str, group_id: str, dx: float, dy: float) -> str:
    """Move one top-level editable group without altering any of its content."""
    pattern = re.compile(
        rf'(<g\s+id="{re.escape(group_id)}"[^>]*?transform="translate\()'
        r'([-\d.]+),([-\d.]+)(\)")', re.DOTALL)
    matches = list(pattern.finditer(svg))
    if len(matches) != 1:
        raise ValueError(f"Expected one transform for {group_id}, found {len(matches)}")

    def shifted(match: re.Match[str]) -> str:
        return (f"{match.group(1)}{float(match.group(2)) + dx:.8f},"
                f"{float(match.group(3)) + dy:.8f}{match.group(4)}")

    return pattern.sub(shifted, svg, count=1)


def top_row_maps() -> str:
    contents = [
        '<g id="compact-map-examples" inkscape:groupmode="layer" '
        'inkscape:label="Panel labels, policy ablation, and C15 fields">',
        '<style>.compact-heading{font:700 12px Arial,sans-serif;fill:#25313a}'
        '.panel-letter{font:700 13px Arial,sans-serif;fill:#25313a}'
        '.compact-map-label{font:12px Arial,sans-serif;fill:#25313a}</style>',
        '<text x="6" y="2" class="panel-letter">A</text>',
        '<text x="23" y="2" class="compact-heading">Behavior</text>',
        '<text x="6" y="69" class="panel-letter">B</text>',
        '<text x="23" y="69" class="compact-heading">Temporal DG credit</text>',
        '<text x="6" y="134" class="panel-letter">C</text>',
        '<text x="23" y="134" class="compact-heading">Temporal hit credit</text>',
        '<text x="389" y="2" class="panel-letter">D</text>',
        '<text x="405" y="2" class="compact-heading">C15 · temporal · s123</text>',
        # Panel A contrasts the learned policy with random actions; the cross
        # marks interruption of the Policy-to-Environment feedback loop.
        '<path id="policy-ablation-cross" d="M 114,6 L 126,18 M 114,18 L 126,6" '
        'fill="none" stroke="#a12629" stroke-width="2.5" '
        'stroke-linecap="round"/>',
    ]
    for x, unit in zip((389, 439, 489), UNITS):
        contents.extend((
            f'<image x="{x}" y="9" width="46" height="46" '
            f'preserveAspectRatio="none" xlink:href="{map_data_uri(unit)}"/>',
            f'<rect x="{x}" y="9" width="46" height="46" fill="none" '
            'stroke="#5c6870" stroke-width="0.6"/>',
            f'<text x="{x + 23}" y="68" text-anchor="middle" '
            f'class="compact-map-label">DG {unit}</text>',
        ))
    contents.append('</g>')
    return "\n".join(contents)


def sidebar_slots() -> str:
    """Draw two compact mechanism icons and two honest, empty analysis slots."""
    items = ['<g id="sidebar-layout-guides">',
             '<style>.mini-label{font:9px Arial,sans-serif;fill:#25313a}'
             '.slot-label{font:9px Arial,sans-serif;fill:#56636c}</style>']
    for y, crossed_connection in ((79, "left"), (144, "right")):
        items.extend((
            f'<rect x="4" y="{y}" width="17" height="20" rx="3" '
            'fill="#c58c8c" stroke="#933333" stroke-width="0.8"/>',
            f'<rect x="29" y="{y}" width="17" height="20" rx="3" '
            'fill="#edf4f2" stroke="#257873" stroke-width="0.8"/>',
            f'<rect x="54" y="{y}" width="20" height="20" rx="3" '
            'fill="#abc9e0" stroke="#16639a" stroke-width="0.8"/>',
            f'<text x="12.5" y="{y+13}" text-anchor="middle" class="mini-label">DG</text>',
            f'<text x="37.5" y="{y+13}" text-anchor="middle" class="mini-label">TS</text>',
            f'<text x="64" y="{y+13}" text-anchor="middle" class="mini-label">Pol.</text>',
            f'<path d="M 21,{y+10} H 29 M 46,{y+10} H 54" '
            'stroke="#26323a" stroke-width="0.8"/>',
        ))
        cross_x = 25 if crossed_connection == "left" else 50
        items.append(
            f'<path d="M {cross_x-3},{y+5} L {cross_x+3},{y+15} '
            f'M {cross_x-3},{y+15} L {cross_x+3},{y+5}" '
            'stroke="#a12629" stroke-width="1.5" fill="none"/>')
    for y, title, detail in ((78, "Trajectory", "candidate"),
                              (143, "Peak spread", "candidate")):
        items.extend((
            f'<rect x="462" y="{y}" width="73" height="51" rx="2" '
            'fill="#f7f9fa" stroke="#9ba8b0" stroke-width="0.7" '
            'stroke-dasharray="3 2"/>',
            f'<text x="498.5" y="{y+21}" text-anchor="middle" '
            f'class="slot-label">{title}</text>',
            f'<text x="498.5" y="{y+34}" text-anchor="middle" '
            f'class="slot-label">{detail}</text>',
        ))
    items.append('</g>')
    return "\n".join(items)


def make_sidebar_variant(svg: str) -> str:
    """Free a right-side column without changing the size of any data plot."""
    for prefix in ("push", "pull"):
        for metric in ("dg_information", "map_overlap", "peak_coverage", "graph_edges"):
            svg = move_group(svg, f"component_{prefix}_{metric}", -80, 0)
    for group_id in ("g701", "g702"):
        svg, count = re.subn(rf'(<g\s+id="{group_id}")',
                             r'\1 style="display:none"', svg, count=1)
        if count != 1:
            raise ValueError(f"Could not hide the original {group_id} schematic")
    # The B x-axis tick labels reach into the next row's left margin. This
    # shorter heading keeps the original meaning without covering those ticks.
    svg = svg.replace('>Temporal hit credit</text>', '>Hit credit</text>', 1)
    return re.sub(r"</svg>\s*$", sidebar_slots() + "\n</svg>\n", svg)


def export(svg: str, path: Path) -> None:
    path.write_text(svg)
    for extension, width in (("pdf", None), ("png", "1800")):
        command = ["inkscape", str(path),
                   f"--export-filename={path.with_suffix('.' + extension)}"]
        if width is not None:
            command.append(f"--export-width={width}")
        subprocess.run(command, check=True, capture_output=True, text=True)


def build() -> None:
    svg = SOURCE.read_text()
    expected = ('width="6.7790866in"', 'height="2.8933051in"',
                'viewBox="0 0 488.09423 208.31796"')
    if any(svg.count(marker) != 1 for marker in expected):
        raise ValueError("The author SVG geometry changed; review before reflowing")
    # Keep the plot glyphs at the author's original size; raise only the 8 pt
    # tick/annotation text to the previously accepted 9 pt minimum.
    svg = svg.replace("font-size:8px", "font-size:9px")
    # The hand-edited source label ends mid-word at "split-half r". Use a
    # shorter, complete axis label; the caption defines split-half reliability.
    svg = repair_readout_label(svg)

    # The first result plot retains its original position. Successive plots
    # shift right into formerly unused page width; none is resized.
    x_offsets = (0, 15, 30, 47)
    metrics = ("dg_information", "map_overlap", "peak_coverage", "graph_edges")
    for prefix, row_shift in (("push", -10), ("pull", -16)):
        for metric, dx in zip(metrics, x_offsets):
            svg = move_group(svg, f"component_{prefix}_{metric}", dx, row_shift)
    svg = svg.replace('id="g701"><g',
                      'id="g701" transform="translate(0,-10)"><g', 1)
    svg = move_group(svg, "g702", 0, -16)

    svg = svg.replace('width="6.7790866in"', 'width="7.5in"', 1)
    svg = svg.replace('height="2.8933051in"', 'height="210pt"', 1)
    svg = svg.replace('viewBox="0 0 488.09423 208.31796"',
                      'viewBox="0 -12 540 210"', 1)
    svg = svg.replace('sodipodi:docname="Figure2_handmade.svg"',
                      'sodipodi:docname="Figure2_tetris.svg"', 1)
    if len(re.findall(r"</svg>\s*$", svg)) != 1:
        raise ValueError("Could not find a unique SVG closing tag")
    svg = re.sub(r"</svg>\s*$", top_row_maps() + "\n</svg>\n", svg)
    sidebar_svg = make_sidebar_variant(svg)
    sidebar_svg = sidebar_svg.replace('sodipodi:docname="Figure2_tetris.svg"',
                                      'sodipodi:docname="Figure2_tetris_sidebar.svg"', 1)
    export(fix_pdf_viewbox(svg, "540", "210", 12), OUTPUT)
    export(fix_pdf_viewbox(sidebar_svg, "540", "210", 12), SIDEBAR_OUTPUT)


if __name__ == "__main__":
    build()
