"""Render a per-unit min–max variant of every COSYNE terminal DG atlas.

Read the archived numerical NPZ maps rather than recoloring the original
shared-scale PNGs. Orientation, occupancy masking, and the five-stop viridis
palette match the canonical evaluator; only color normalization changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE_INDEX = HERE / "full_seed_atlases/terminal_seed_metrics.csv"
UNIT_METRICS = HERE / "full_seed_atlases/terminal_unit_metrics.csv"
OUTPUT = HERE / "full_seed_atlases_minmax"
STOPS = np.asarray(((68, 1, 84), (59, 82, 139), (33, 145, 140),
                    (94, 201, 98), (253, 231, 37)), dtype=float)
GRAY = (235, 235, 235)
PANEL_WIDTH = 320
PANEL_HEIGHT = 300
TITLE_HEIGHT = 144
FOOTER_HEIGHT = 70
MAP_PIXELS = 190
MAP_OFFSET = (PANEL_WIDTH - MAP_PIXELS) // 2


def scalable_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if not path.is_file():
        raise FileNotFoundError(f"A scalable font is required: {path}")
    return ImageFont.truetype(str(path), size=size)


def raw_npz_relative(source_atlas: str) -> Path:
    atlas = Path(source_atlas)
    if not atlas.name.startswith("place_fields_") or atlas.suffix != ".png":
        raise ValueError(f"Unexpected atlas name: {atlas}")
    label = atlas.stem[len("place_fields_"):]
    return atlas.parent.parent / "raw" / label / "place_fields.npz"


def colorize(normalized: np.ndarray, visited: np.ndarray) -> Image.Image:
    """Apply the canonical palette with gray for bins outside the probe path."""
    clipped = np.clip(np.where(visited, normalized, 0.0), 0.0, 1.0) * 4
    index = np.minimum(np.floor(clipped).astype(int), 3)
    fraction = clipped - index
    rgb = np.rint(STOPS[index] * (1.0 - fraction[..., None]) +
                   STOPS[index + 1] * fraction[..., None]).astype(np.uint8)
    rgb[~visited] = GRAY
    tile = Image.fromarray(rgb, mode="RGB")
    return tile.resize((MAP_PIXELS, MAP_PIXELS), Image.Resampling.NEAREST)


def render_atlas(row: pd.Series, npz_path: Path,
                 expected_units: pd.DataFrame) -> list[dict]:
    with np.load(npz_path, allow_pickle=False) as source:
        occupancy = source["occupancy"]
        maps = source["rate_maps"]
        information = source["spatial_information"]
        activity = source["active_fraction"]
    if (occupancy.shape != (19, 19) or maps.shape != (19, 19, 16) or
            information.shape != (16,) or activity.shape != (16,)):
        raise ValueError(f"Unexpected DG map dimensions: {npz_path}")
    if int(occupancy.sum()) != 10001 or int((occupancy > 0).sum()) != row.visited_cells:
        raise ValueError(f"Probe length or occupancy changed: {npz_path}")
    expected_units = expected_units.sort_values("unit")
    if not np.array_equal(expected_units.unit.to_numpy(), np.arange(16)):
        raise ValueError(f"Missing per-unit metrics: {row.condition} seed {row.seed}")
    if not np.allclose(information, expected_units.spatial_information_bits,
                       rtol=1e-9, atol=1e-9):
        raise ValueError(f"Spatial scores differ from archived metrics: {npz_path}")
    if not np.allclose(activity, expected_units.active_fraction,
                       rtol=1e-9, atol=1e-9):
        raise ValueError(f"Activity differs from archived metrics: {npz_path}")

    image = Image.new("RGB", (4 * PANEL_WIDTH,
                              TITLE_HEIGHT + 4 * PANEL_HEIGHT + FOOTER_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = scalable_font(30, bold=True)
    text_font = scalable_font(21)
    unit_font = scalable_font(22, bold=True)
    family = row.family
    condition = (f"{row.encoder_credit} DG / {row.worker_bonus} worker"
                 if family == "C05" else f"{row.encoder_credit} DG credit")
    title = f"{family} · {condition} · seed {int(row.seed)}"
    draw.text((24, 11), title, font=title_font, fill="black")
    draw.text((24, 54),
              f"All 16 DG maps · 10,001 decisions · {int(row.visited_cells)}/361 bins visited",
              font=text_font, fill="black")
    draw.text((24, 84),
              "Each tile: visited-bin min → 0, max → 1; gray = unvisited",
              font=text_font, fill="black")
    scale_rows = []
    for unit in range(16):
        row_index, column = divmod(unit, 4)
        x = column * PANEL_WIDTH + MAP_OFFSET
        y = TITLE_HEIGHT + row_index * PANEL_HEIGHT
        rate_map = maps[:, :, unit].T
        visited = (occupancy.T > 0) & np.isfinite(rate_map)
        if not np.any(visited):
            raise ValueError(f"No finite visited map bins: {npz_path}, DG {unit}")
        minimum = float(rate_map[visited].min())
        maximum = float(rate_map[visited].max())
        normalized = np.zeros_like(rate_map, dtype=float)
        if maximum > minimum:
            normalized[visited] = ((rate_map[visited] - minimum) /
                                   (maximum - minimum))
        tile = colorize(normalized, visited)
        image.paste(tile, (x, y))
        draw.rectangle((x, y, x + MAP_PIXELS, y + MAP_PIXELS),
                       outline=(80, 80, 80), width=1)
        center = column * PANEL_WIDTH + PANEL_WIDTH // 2
        draw.text((center, y + 199), f"DG {unit:02d} · SI {information[unit]:.3f}",
                  font=unit_font, fill="black", anchor="mt")
        draw.text((center, y + 231), f"active {100 * activity[unit]:.1f}%",
                  font=text_font, fill="black", anchor="mt")
        draw.text((center, y + 262), f"raw min {minimum:.2f} · max {maximum:.2f}",
                  font=text_font, fill="black", anchor="mt")
        scale_rows.append({
            "family": family, "condition": row.condition,
            "encoder_credit": row.encoder_credit,
            "worker_bonus": row.worker_bonus, "seed": int(row.seed),
            "unit": unit, "raw_min": minimum, "raw_max": maximum,
            "constant_map": maximum == minimum,
            "visited_bins": int(visited.sum()),
            "spatial_information_bits": float(information[unit]),
            "active_fraction": float(activity[unit]),
        })
    draw.text((24, TITLE_HEIGHT + 4 * PANEL_HEIGHT + 11),
              "Color shows each unit's spatial shape; use the printed raw range to compare amplitude.",
              font=text_font, fill="black")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT / row.file, optimize=True)
    return scale_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-root", type=Path, required=True,
                        help="Local read-only copy of the canonical terminal NPZs")
    args = parser.parse_args()
    terminal = pd.read_csv(SOURCE_INDEX)
    units = pd.read_csv(UNIT_METRICS)
    if len(terminal) != 40 or len(units) != 640:
        raise ValueError("Expected complete 40-run and 640-unit source indexes")
    manifest = {"schema": "intrmotiv/cosyne-terminal-atlas-minmax/v1",
                "normalization": "(rate - visited_min) / (visited_max - visited_min); constant maps show 0",
                "rate_maps": "post-threshold occupancy-corrected DG maps",
                "occupancy": "unvisited bins gray; [x,y,unit] source transposed for image display",
                "source_index": str(SOURCE_INDEX.relative_to(HERE)), "atlases": []}
    scale_rows = []
    for _, row in terminal.iterrows():
        relative = raw_npz_relative(row.source_relative_path)
        npz_path = args.npz_root / relative
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        selected = units[(units.condition == row.condition) & (units.seed == row.seed)]
        scale_rows.extend(render_atlas(row, npz_path, selected))
        output = OUTPUT / row.file
        manifest["atlases"].append({
            "condition": row.condition, "seed": int(row.seed),
            "output": output.name,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_npz_relative_path": str(relative),
            "source_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        })
    pd.DataFrame(scale_rows).to_csv(OUTPUT / "unit_scale_ranges.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
