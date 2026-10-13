"""Extend the author's editable Figure 2 layout with verified C15 DG maps.

The original SVG is copied unchanged inside the new canvas. Only a new lower
strip is added, so the author's plot positions and editable vector groups stay
intact. The four dashed cards reserve space for optional metric illustrations.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import re
import subprocess

from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Figure2_fiddle.svg"
OUTPUT = HERE / "Figure2_fiddle_with_fields.svg"
ATLAS = (HERE / "figure1_candidates" / "full_seed_atlases" /
         "c15_temporal_s123.png")


# These units illustrate a compact field (4) and two other high-information
# responses (7, 15) in the same frozen network and probe. They were selected
# qualitatively and are not an estimate of field prevalence.
UNITS = (4, 7, 15)


def map_data_uri(unit: int) -> str:
    with Image.open(ATLAS) as source:
        if source.size != (1200, 1206):
            raise ValueError(f"Unexpected C15 atlas dimensions: {source.size}")
        tile = source.crop((18 + 300 * (unit % 4),
                            126 + 270 * (unit // 4),
                            209 + 300 * (unit % 4),
                            317 + 270 * (unit // 4))).convert("RGB")
    buffer = BytesIO()
    tile.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def lower_strip() -> str:
    maps = []
    for x, unit in zip((10, 76, 142), UNITS):
        uri = map_data_uri(unit)
        maps.extend((
            f'<image x="{x}" y="236" width="56" height="56" '
            f'preserveAspectRatio="none" xlink:href="{uri}"/>',
            f'<rect x="{x}" y="236" width="56" height="56" fill="none" '
            'stroke="#5c6870" stroke-width="0.6"/>',
            f'<text x="{x + 28}" y="307" text-anchor="middle" '
            f'class="field-label">DG {unit}</text>',
        ))
    placeholders = []
    for x, line1, line2 in (
        (228, "Split-half", "reliability"),
        (290, "Map", "overlap"),
        (352, "Peak", "coverage"),
        (414, "Graph", "edges"),
    ):
        placeholders.extend((
            f'<rect x="{x}" y="236" width="59" height="56" rx="4" '
            'fill="#f7f9fa" stroke="#82919b" stroke-width="0.7" '
            'stroke-dasharray="3 2"/>',
            f'<text x="{x + 29.5}" y="262" text-anchor="middle" '
            f'class="placeholder-label">{line1}</text>',
            f'<text x="{x + 29.5}" y="278" text-anchor="middle" '
            f'class="placeholder-label">{line2}</text>',
        ))
    return "\n".join((
        '<g id="illustrative-fields-and-icon-slots">',
        '<style>.strip-heading{font:700 12px Arial,sans-serif;fill:#25313a}'
        '.field-label{font:12px Arial,sans-serif;fill:#25313a}'
        '.placeholder-label{font:12px Arial,sans-serif;fill:#56636c}</style>',
        '<path d="M 6,216 H 482" fill="none" stroke="#aeb9bf" stroke-width="0.7"/>',
        '<text x="10" y="229" class="strip-heading">C15 temporal DG · seed 123</text>',
        '<text x="228" y="229" class="strip-heading">Metric illustrations · slots</text>',
        *maps,
        *placeholders,
        '</g>',
    ))


def main() -> None:
    source = SOURCE.read_text()
    expected = ('width="6.7790866in"', 'height="2.8933051in"',
                'viewBox="0 0 488.09423 208.31796"')
    if any(source.count(marker) != 1 for marker in expected):
        raise ValueError("The author's SVG geometry changed; review the new layout first")
    # The supplied crop cuts the upper ends of two vertical axis labels. Add
    # eight points above it without resizing or moving any existing panel.
    source = source.replace('height="2.8933051in"', 'height="4.486111in"', 1)
    source = source.replace('viewBox="0 0 488.09423 208.31796"',
                            'viewBox="0 -8 488.09423 323"', 1)
    source = source.replace('sodipodi:docname="Figure2_handmade.svg"',
                            'sodipodi:docname="Figure2_fiddle_with_fields.svg"', 1)
    if len(re.findall(r"</svg>\s*$", source)) != 1:
        raise ValueError("Could not find a unique SVG closing tag")
    output = re.sub(r"</svg>\s*$", lower_strip() + "\n</svg>\n", source)
    OUTPUT.write_text(output)
    for extension, width in (("pdf", None), ("png", "1800")):
        command = ["inkscape", str(OUTPUT),
                   f"--export-filename={OUTPUT.with_suffix('.' + extension)}"]
        if width is not None:
            command.append(f"--export-width={width}")
        subprocess.run(command, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
