"""Build four small, data-free icons for Figure 2's metric labels.

Cosine, coverage, and edges are 10 by 30 mm; reliability is 10 by 20 mm.
The larger contact sheet is only a preview.
"""

from __future__ import annotations

from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
WIDTH_PT = 10 * 72 / 25.4
HEIGHT_PT = 30 * 72 / 25.4
INK = "#27343c"
MUTED = "#919ba2"
ACCENT = "#555f66"


def svg_document(body: str, height_mm: float = 30) -> str:
    height_pt = height_mm * 72 / 25.4
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="{height_mm:g}mm" '
        f'viewBox="0 0 {WIDTH_PT:.6f} {height_pt:.6f}">'
        '<g stroke-linecap="round" stroke-linejoin="round">'
        f'{body}</g></svg>\n'
    )


def frames() -> str:
    return (
        f'<rect x="2.52" y="4.5" width="23.3" height="23.3" rx="1.3" '
        f'fill="white" stroke="{INK}" stroke-width="0.75"/>'
        f'<rect x="2.52" y="57.2" width="23.3" height="23.3" rx="1.3" '
        f'fill="white" stroke="{INK}" stroke-width="0.75"/>'
    )


def comparison_link() -> str:
    """Point toward the higher statistic, matching an upward plot axis."""
    return (
        f'<path d="M 14.17,52.7 V 32.2 M 11.8,34.7 L 14.17,32.2 L 16.54,34.7" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.05"/>'
    )


def cosine_icon() -> str:
    """Two pairs of DG place fields: one similar, one spatially different."""
    fields = (
        ("similar-a", 4.10, 10.30, 8.35, 14.70),
        ("similar-b", 15.10, 10.30, 19.35, 14.70),
        ("different-a", 4.10, 63.00, 6.90, 65.80),
        ("different-b", 15.10, 63.00, 21.45, 69.35),
    )
    gradient = (
        '<defs><radialGradient id="cosine-rate">'
        f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.67"/>'
        f'<stop offset="0.58" stop-color="{ACCENT}" stop-opacity="0.22"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/>'
        '</radialGradient></defs>'
    )
    patches = ''.join(
        f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="url(#cosine-rate)"/>'
        f'<path d="M {x},{y+9.2:.2f} V {y+1.7:.2f} M {x},{y+9.2:.2f} H {x+7.5:.2f}" '
        f'fill="none" stroke="{MUTED}" stroke-width="0.52"/>'
        for name, x, y, cx, cy in fields
    )
    return gradient + patches + comparison_link()


def coverage_icon() -> str:
    """The same number of peaks, clustered and spatially dispersed."""
    gradient = (
        '<defs><radialGradient id="coverage-blur">'
        f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.29"/>'
        f'<stop offset="0.55" stop-color="{ACCENT}" stop-opacity="0.1"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/>'
        '</radialGradient>'
        '<clipPath id="coverage-top"><rect x="2.52" y="4.5" width="23.3" height="23.3"/></clipPath>'
        '<clipPath id="coverage-bottom"><rect x="2.52" y="57.2" width="23.3" height="23.3"/></clipPath>'
        '</defs>'
    )
    dispersed = ((7.0, 10.3), (21.0, 11.5), (8.1, 23.1), (20.7, 23.4))
    clustered = ((11.3, 66.7), (13.8, 69.5), (16.0, 65.7), (17.0, 71.6))
    def draw_peaks(points: tuple[tuple[float, float], ...], clip_id: str) -> str:
        return (f'<g clip-path="url(#{clip_id})">' + ''.join(
            f'<circle cx="{x}" cy="{y}" r="5.1" fill="url(#coverage-blur)"/>'
            f'<circle cx="{x}" cy="{y}" r="1.15" fill="{INK}" '
            'stroke="white" stroke-width="0.35"/>'
            for x, y in points
        ) + '</g>')

    peaks = draw_peaks(dispersed, 'coverage-top') + draw_peaks(clustered, 'coverage-bottom')
    return gradient + frames() + peaks + comparison_link()


def graph_arrows(edges: tuple[tuple[int, int], ...], y_offset: float) -> str:
    """Draw directed reliable edges between four landmark nodes."""
    nodes = ((7.5, 10), (20.5, 10), (7.5, 22), (20.5, 22))
    paths = []
    for source, target in edges:
        x1, y1 = nodes[source]
        x2, y2 = nodes[target]
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        start_x, start_y = x1 + 2.9 * ux, y1 + y_offset + 2.9 * uy
        end_x, end_y = x2 - 3.3 * ux, y2 + y_offset - 3.3 * uy
        # A tiny V at the destination remains visible at the intended size.
        wing_x, wing_y = end_x - 2.1 * ux, end_y - 2.1 * uy
        px, py = -uy * 1.1, ux * 1.1
        paths.append(
            f'<path d="M {start_x:.2f},{start_y:.2f} L {end_x:.2f},{end_y:.2f} '
            f'M {wing_x+px:.2f},{wing_y+py:.2f} L {end_x:.2f},{end_y:.2f} '
            f'L {wing_x-px:.2f},{wing_y-py:.2f}" fill="none" '
            f'stroke="{INK}" stroke-width="1.05"/>'
        )
    circles = ''.join(
        f'<circle cx="{x}" cy="{y+y_offset}" r="2.35" fill="white" '
        f'stroke="{INK}" stroke-width="0.95"/>'
        for x, y in nodes
    )
    return ''.join(paths) + circles


def edges_icon() -> str:
    """Two stored directed graphs with different reliable-edge counts."""
    return (
        frames()
        + graph_arrows(((0, 1), (0, 2), (1, 3), (2, 3)), 0)
        + graph_arrows(((0, 1), (1, 3)), 52.7)
        + comparison_link()
    )


def reliability_icon() -> str:
    """A single DG field compared across two episode-disjoint halves."""
    fields = (
        (4.10, 4.5, 8.35, 9.0), (15.10, 4.5, 19.35, 9.0),
        (4.10, 42.0, 6.90, 44.8), (15.10, 42.0, 21.45, 48.35),
    )
    out = [
        '<defs><radialGradient id="reliability-rate">'
        f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.67"/>'
        f'<stop offset="0.58" stop-color="{ACCENT}" stop-opacity="0.22"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/>'
        '</radialGradient></defs>',
    ]
    for x, y, cx, cy in fields:
        out.append(f'<rect x="{x}" y="{y}" width="9.2" height="9.2" '
                   f'fill="white" stroke="{MUTED}" stroke-width="0.55"/>')
        out.append(f'<circle cx="{cx}" cy="{cy}" r="4.0" '
                   'fill="url(#reliability-rate)"/>')
    out.append(
        f'<path d="M 14.17,37.3 V 19.3 M 11.8,21.8 L 14.17,19.3 L 16.54,21.8" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.05"/>'
    )
    return ''.join(out)


ICONS = {
    "cosine": cosine_icon,
    "coverage": coverage_icon,
    "edges": edges_icon,
    "reliability": reliability_icon,
}


def export(svg_path: Path, png_width: int = 400) -> None:
    for extension in ("pdf", "png"):
        command = ["inkscape", str(svg_path),
                   f"--export-filename={svg_path.with_suffix('.' + extension)}"]
        if extension == "png":
            command.append(f"--export-width={png_width}")
        subprocess.run(command, capture_output=True, text=True, check=True)


def write_and_export(path: Path, svg: str, png_width: int = 400) -> None:
    """Keep stable exports when the editable source has not changed."""
    if (path.is_file() and path.read_text() == svg
            and path.with_suffix('.pdf').is_file()
            and path.with_suffix('.png').is_file()):
        return
    path.write_text(svg)
    export(path, png_width=png_width)


def build() -> None:
    for name, factory in ICONS.items():
        path = HERE / f"Figure2_metric_icon_{name}.svg"
        write_and_export(path, svg_document(factory(), 20 if name == 'reliability' else 30))

    # Enlarged inspection sheet; use standalone files for physical dimensions.
    locations = (("cosine", 13), ("coverage", 115), ("edges", 217),
                 ("reliability", 319))
    sheet_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="147mm" height="70mm" '
        'viewBox="0 0 417 198"><rect width="417" height="198" fill="white"/>',
        '<style>text{font:11px Arial,sans-serif;fill:#27343c}</style>',
    ]
    for name, x in locations:
        top = 34 if name == 'reliability' else 8
        sheet_parts.append(f'<g transform="translate({x},{top}) scale(1.9)">'
                           f'{ICONS[name]()}')
        sheet_parts.append('</g>')
        sheet_parts.append(f'<text x="{x+27}" y="185" text-anchor="middle">'
                           f'{"Map cosine" if name == "cosine" else "Peak coverage" if name == "coverage" else "Reliable edges" if name == "edges" else "Reliability"}'
                           '</text>')
    sheet_parts.append('</svg>\n')
    sheet = HERE / "Figure2_metric_icons_preview.svg"
    write_and_export(sheet, ''.join(sheet_parts), png_width=1600)


if __name__ == "__main__":
    build()
