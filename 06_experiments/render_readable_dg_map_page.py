#!/usr/bin/env python3
"""Render one readable four-unit DG-map page from a telemetry artifact.

The page preserves the raw evaluator's occupancy masking and uses a single
colour scale across its four displayed units.  Invoke it once for each
consecutive four-unit page so a report can retain a compact 2 x 2 layout
without shrinking unit labels into illegibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--unit-start", type=int, choices=(0, 4, 8, 12), required=True)
    parser.add_argument("--pre-threshold", action="store_true")
    return parser.parse_args()


def configure_font() -> None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return
    raise RuntimeError("No scalable DejaVu Sans font found; refusing bitmap font fallback")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in FONT_CANDIDATES:
        path = candidate.with_name(name)
        if path.exists():
            return ImageFont.truetype(path, size=size)
    raise RuntimeError("No scalable DejaVu Sans font found; refusing bitmap font fallback")


def interpolate(stops: tuple[tuple[int, int, int], ...], value: np.ndarray) -> np.ndarray:
    scaled = np.clip(value, 0.0, 1.0) * (len(stops) - 1)
    index = np.minimum(scaled.astype(int), len(stops) - 2)
    fraction = (scaled - index)[..., None]
    palette = np.asarray(stops, dtype=np.float64)
    return np.round(palette[index] * (1.0 - fraction) + palette[index + 1] * fraction).astype(np.uint8)


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, used_font: ImageFont.FreeTypeFont, width: int) -> None:
    box = draw.textbbox((0, 0), text, font=used_font)
    draw.text(((width - (box[2] - box[0])) / 2, y), text, fill="#0f172a", font=used_font)


def main() -> None:
    args = parse_args()
    configure_font()
    artifact = np.load(args.artifact, allow_pickle=False)
    occupancy = artifact["occupancy"]
    maps = artifact["pre_threshold_rate_maps"] if args.pre_threshold else artifact["rate_maps"]
    units = range(args.unit_start, args.unit_start + 4)
    selected = maps[:, :, list(units)]
    valid = occupancy > 0

    if args.pre_threshold:
        limit = float(np.nanmax(np.abs(selected[valid])))
        normalised = lambda values: (values + limit) / (2.0 * limit)
        palette = ((49, 54, 149), (116, 173, 209), (247, 247, 247), (244, 165, 130), (165, 0, 38))
    else:
        vmax = float(np.nanmax(selected[valid]))
        normalised = lambda values: values / vmax
        palette = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))

    width, height = 1280, 1400
    panel, left, top, gap = 480, 90, 160, 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_centered(draw, args.title, 36, font(52, bold=True), width)
    for index, unit in enumerate(units):
        row, column = divmod(index, 2)
        x = left + column * (panel + gap)
        y = top + row * (panel + gap)
        rgb = interpolate(palette, normalised(np.nan_to_num(maps[:, :, unit], nan=0.0)))
        rgb[~valid] = (209, 213, 219)
        map_image = Image.fromarray(rgb[::-1], mode="RGB").resize((panel, panel), Image.Resampling.NEAREST)
        image.paste(map_image, (x, y))
        draw.rectangle((x, y, x + panel, y + panel), outline="#475569", width=3)
        label = f"DG {unit:02d}"
        box = draw.textbbox((0, 0), label, font=font(48, bold=True))
        draw.text((x + (panel - (box[2] - box[0])) / 2, y + panel + 17), label, fill="#0f172a", font=font(48, bold=True))
    label = "Shared scale; gray = unvisited position cell"
    draw_centered(draw, label, 1340, font(34), width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
