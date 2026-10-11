"""Compose the user's supplied SVG elements into a compact COSYNE figure.

Each panel is rendered directly from the supplied SVG at 400 pixels per inch.
The source artwork is retained beside this script, including its vector text,
formulas, screenshots, and diagrams. Inkscape changes text placement when the
whole source is cloned under a transformed SVG group, so the assembled file
embeds high-resolution crops instead of altering the original objects.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY / "architecture_supplied_elements.svg"
OUTPUT = DIRECTORY / "architecture_from_supplied.svg"
SOURCE_SHA256 = "f8c3c029f1c46f6dc5294482203bb7b061a39529f7dea31aa677ad05d04a0910"

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


def tag(name: str) -> str:
    return f"{{{SVG}}}{name}"


def add_crop(
    figure: ET.Element,
    original: ET.Element,
    temporary_dir: Path,
    *,
    name: str,
    source_pixels: tuple[float, float, float, float],
    destination: tuple[int, int, int, int],
) -> None:
    """Crop one source region and place its 2× render at final figure scale."""

    px_to_mm = 25.4 / 96.0  # Original Inkscape page uses millimetres.
    source_box = tuple(value * px_to_mm for value in source_pixels)
    target_x, target_y, target_width, target_height = destination
    cropped = copy.deepcopy(original)
    cropped.set("viewBox", " ".join(f"{value:.5f}" for value in source_box))
    cropped.set("width", str(2 * target_width))
    cropped.set("height", str(2 * target_height))

    crop_svg = temporary_dir / f"{name}.svg"
    crop_png = temporary_dir / f"{name}.png"
    ET.ElementTree(cropped).write(crop_svg, encoding="utf-8", xml_declaration=True)
    subprocess.run(
        [
            "inkscape",
            str(crop_svg),
            "--export-type=png",
            f"--export-filename={crop_png}",
            f"--export-width={2 * target_width}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    embedded_png = base64.b64encode(crop_png.read_bytes()).decode("ascii")
    ET.SubElement(
        figure,
        tag("image"),
        {
            "id": name,
            "x": str(target_x),
            "y": str(target_y),
            "width": str(target_width),
            "height": str(target_height),
            f"{{{XLINK}}}href": f"data:image/png;base64,{embedded_png}",
        },
    )


def render() -> None:
    source_bytes = SOURCE.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(f"Supplied SVG changed: {actual_sha256}")
    original = ET.fromstring(source_bytes)
    source_group = next(
        child for child in original if child.tag == tag("g") and child.get("id") == "g1-5"
    )
    # These poster objects lie outside the composed panels and leave clipped
    # fragments at the upper or lower edge of the main stream.
    for child in list(source_group):
        if child.get("id") in {
            "text131-6-1-1-7-5-2-3-2-8-1-1-1-4-9",
            "use18-18-7",
        }:
            source_group.remove(child)

    figure = ET.Element(
        tag("svg"),
        {
            "width": "7.2in",
            "height": "4.1in",
            "viewBox": "0 0 1440 820",
            "role": "img",
            "aria-labelledby": "figure-title figure-description",
        },
    )
    ET.SubElement(figure, tag("title"), {"id": "figure-title"}).text = (
        "IntrMotiv architecture using supplied elements"
    )
    ET.SubElement(figure, tag("desc"), {"id": "figure-description"}).text = (
        "The supplied visual encoder, DG, CA3, readout, action and environment "
        "artwork forms a compact top flow. A short label identifies DG as a "
        "learned sparse projection and CA3 as a fixed shift register. Below "
        "it are the supplied DG sparsifier, CA3 temporal traces, and red and "
        "blue learning formulas."
    )
    ET.SubElement(figure, tag("rect"), {"width": "1440", "height": "820", "fill": "#fff"})
    # This short functional label is the only new figure annotation. W, B,
    # and the full DG/CA3 updates are deferred to the detailed explanation.
    ET.SubElement(
        figure,
        tag("text"),
        {
            "x": "720",
            "y": "483",
            "text-anchor": "middle",
            "font-family": "Arial, sans-serif",
            "font-size": "34",
            "fill": "#172c3c",
        },
    ).text = "DG: learned sparse projection  →  CA3: fixed shift register"

    with tempfile.TemporaryDirectory(prefix="cosyne_architecture_") as temporary:
        temp_dir = Path(temporary)
        crops = [
            # Original main stream, environment return arrow, and loss key.
            ("supplied-main-stream", (48, 230, 1548, 480), (27, 12, 1386, 430)),
            # Original visual intuitions: DG threshold and CA3 temporal trace.
            ("supplied-sparsifier", (805, 735, 395, 225), (29, 509, 350, 199)),
            ("supplied-ca3-traces", (395, 1080, 310, 210), (410, 509, 280, 190)),
            ("supplied-ca3-caption", (465, 1288, 420, 110), (360, 694, 390, 102)),
            # The original full red DG and blue policy formulas.
            ("supplied-dg-formula", (1235, 805, 600, 105), (790, 522, 600, 105)),
            ("supplied-policy-formula", (875, 1455, 465, 125), (840, 660, 490, 132)),
        ]
        for name, source_pixels, destination in crops:
            add_crop(
                figure,
                original,
                temp_dir,
                name=name,
                source_pixels=source_pixels,
                destination=destination,
            )
    ET.indent(figure, space="  ")
    ET.ElementTree(figure).write(OUTPUT, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    render()
