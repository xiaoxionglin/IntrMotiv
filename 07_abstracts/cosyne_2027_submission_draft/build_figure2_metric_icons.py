"""Build three small, data-free icons for Figure 2's metric labels.

Each standalone SVG/PDF is exactly 10 by 30 mm. The larger contact sheet is
only a preview; the manuscript can use the individual vector assets.
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


def svg_document(body: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="30mm" '
        f'viewBox="0 0 {WIDTH_PT:.6f} {HEIGHT_PT:.6f}">'
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
    """A two-headed link means comparison, never a result direction."""
    return (
        f'<path d="M 14.17,32.2 V 52.7 M 11.8,34.7 L 14.17,32.2 L 16.54,34.7 '
        f'M 11.8,50.2 L 14.17,52.7 L 16.54,50.2" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.05"/>'
    )


def cosine_icon() -> str:
    """Two spatial rate maps whose distributed patterns enter a cosine."""
    gradient = (
        '<defs><radialGradient id="cosine-rate">'
        f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.67"/>'
        f'<stop offset="0.58" stop-color="{ACCENT}" stop-opacity="0.22"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/>'
        '</radialGradient></defs>'
    )
    patches = ''.join(
        f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" '
        'fill="url(#cosine-rate)"/>'
        for x, y, rx, ry in (
            (10.2, 12.8, 7.2, 6.6), (18.9, 21.2, 5.9, 5.2),
            (11.4, 65.8, 7.2, 6.6), (18.1, 73.4, 5.9, 5.2),
        )
    )
    return gradient + frames() + patches + comparison_link()


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
    clustered = ((11.3, 14.0), (13.8, 16.8), (16.0, 13.0), (17.0, 18.9))
    dispersed = ((7.0, 63.0), (21.0, 64.2), (8.1, 75.8), (20.7, 76.1))
    def draw_peaks(points: tuple[tuple[float, float], ...], clip_id: str) -> str:
        return (f'<g clip-path="url(#{clip_id})">' + ''.join(
            f'<circle cx="{x}" cy="{y}" r="5.1" fill="url(#coverage-blur)"/>'
            f'<circle cx="{x}" cy="{y}" r="1.15" fill="{INK}" '
            'stroke="white" stroke-width="0.35"/>'
            for x, y in points
        ) + '</g>')

    peaks = draw_peaks(clustered, 'coverage-top') + draw_peaks(dispersed, 'coverage-bottom')
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
        + graph_arrows(((0, 1), (1, 3)), 0)
        + graph_arrows(((0, 1), (0, 2), (1, 3), (2, 3)), 52.7)
        + comparison_link()
    )


ICONS = {
    "cosine": cosine_icon,
    "coverage": coverage_icon,
    "edges": edges_icon,
}


def export(svg_path: Path, png_width: int = 400) -> None:
    for extension in ("pdf", "png"):
        command = ["inkscape", str(svg_path),
                   f"--export-filename={svg_path.with_suffix('.' + extension)}"]
        if extension == "png":
            command.append(f"--export-width={png_width}")
        subprocess.run(command, capture_output=True, text=True, check=True)


def build() -> None:
    for name, factory in ICONS.items():
        path = HERE / f"Figure2_metric_icon_{name}.svg"
        path.write_text(svg_document(factory()))
        export(path)

    # Enlarged inspection sheet; standalone icons retain their 10 × 30 mm size.
    locations = (("cosine", 13), ("coverage", 116), ("edges", 219))
    sheet_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="110mm" height="70mm" '
        'viewBox="0 0 312 198"><rect width="312" height="198" fill="white"/>',
        '<style>text{font:11px Arial,sans-serif;fill:#27343c}</style>',
    ]
    for name, x in locations:
        sheet_parts.append(f'<g transform="translate({x},8) scale(1.9)">'
                           f'{ICONS[name]()}')
        sheet_parts.append('</g>')
        sheet_parts.append(f'<text x="{x+27}" y="185" text-anchor="middle">'
                           f'{"Map cosine" if name == "cosine" else "Peak coverage" if name == "coverage" else "Reliable edges"}'
                           '</text>')
    sheet_parts.append('</svg>\n')
    sheet = HERE / "Figure2_metric_icons_preview.svg"
    sheet.write_text(''.join(sheet_parts))
    export(sheet, png_width=1600)


if __name__ == "__main__":
    build()
