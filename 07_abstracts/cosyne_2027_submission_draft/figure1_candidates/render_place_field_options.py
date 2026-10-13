"""Render five auditable four-map alternatives for compact Figure 1 panel C.

The source figure panels and their metrics are archived. No evaluator, new
rollout, submitted figure, or Overleaf package is changed by this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from render_architecture_with_fields import DRAFT, style


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "place_field_options"
RESULTS = DRAFT.parents[1] / "06_experiments" / "results"
ARCHIVE = RESULTS / "A0_poster_analysis_20260926" / "flat_goal_comparison"
COSYNE = RESULTS / "cosyne_2027_checkpoint_fields"

# One four-unit alternative per independently sampled frozen-policy probe.
# Current-atlas units balance strict mono-field examples with high spatial
# information; the historical exports contain only their archived top four.
OPTIONS = (
    ("01_cosyne_c15_temporal_s123", "COSYNE C15 temporal · seed 123",
     "cosyne", "c15_available", "COS27_C15_TEMPORAL", 123,
     (4, 3, 15, 12)),
    ("02_cosyne_c05_const_temporal_s99", "COSYNE C05 constant/temporal · seed 99",
     "cosyne", "c05_pushpull_s3", "COS27_C05_CONSTANT_TEMPORAL", 99,
     (5, 4, 15, 12)),
    ("03_archived_c15_s8", "Earlier C15 · seed 8",
     "archive", "", "C15", 8, (15, 10, 7, 14)),
    ("04_archived_c05_s123", "Earlier C05 · seed 123",
     "archive", "", "C05", 123, (6, 1, 2, 3)),
    ("05_archived_c01_s99", "Earlier C01 · seed 99",
     "archive", "", "C01", 99, (1, 2, 12, 13)),
)


def matching_cosyne_files(batch: str, condition: str, seed: int) -> tuple[Path, Path, Path]:
    summary = COSYNE / batch / "summary"
    matches = list(summary.glob(
        f"place_fields_{condition}_S{seed}__*__{condition}__s{seed}__f100007936.png"))
    if len(matches) != 1:
        raise ValueError(f"Expected one atlas for {condition} seed {seed}: {matches}")
    return (matches[0], summary / "per_unit_place_field_metrics.csv",
            summary / "place_field_summary.csv")


def cosyne_option(batch: str, condition: str, seed: int, units: tuple[int, ...]):
    atlas, unit_csv, probe_csv = matching_cosyne_files(batch, condition, seed)
    unit_table = pd.read_csv(unit_csv)
    suffix = next((s for s in unit_table.label_suffix.dropna().unique()
                   if atlas.stem.endswith(s)), None)
    if suffix is None:
        raise ValueError(f"No matching unit rows for {atlas}")
    selected = unit_table[(unit_table.label_suffix == suffix) &
                          (unit_table.unit.isin(units))].set_index("unit")
    if set(selected.index) != set(units) or not selected.field_eligible.all():
        raise ValueError(f"Missing or ineligible selected unit: {atlas}")
    probe = pd.read_csv(probe_csv)
    matching_probe = probe[probe.label.map(lambda value: atlas.stem.endswith(value))]
    if len(matching_probe) != 1 or int(matching_probe.iloc[0].frames) != 10001:
        raise ValueError(f"Unexpected probe summary for {atlas}")
    with Image.open(atlas) as source:
        if source.size != (1200, 1206):
            raise ValueError(f"Unexpected atlas dimensions: {atlas}")
        tiles = {unit: source.crop((18 + 300 * (unit % 4),
                                    126 + 270 * (unit // 4),
                                    209 + 300 * (unit % 4),
                                    317 + 270 * (unit // 4))).convert("RGB")
                 for unit in units}
    metrics = {unit: {
        "si": float(selected.loc[unit, "spatial_information_bits"]),
        "mono_score": float(selected.loc[unit, "mono_score"]),
        "mono_field": bool(selected.loc[unit, "mono_field"]),
        "active_fraction": float(selected.loc[unit, "active_fraction"]),
    } for unit in units}
    return (tiles, metrics, int(matching_probe.iloc[0].visited_cells),
            "shared within-run scale; gray means unvisited",
            [atlas, unit_csv, probe_csv])


def archive_option(condition: str, seed: int, units: tuple[int, ...]):
    source_svg = ARCHIVE / f"{condition.lower()}_seed{seed}" / "place_fields_individual_scale.svg"
    unit_csv = ARCHIVE / "mono_field_units.csv"
    rank_csv = ARCHIVE / "place_field_individual_scales.csv"
    probe_csv = ARCHIVE / "matched_terminal_per_run.csv"
    ranked = pd.read_csv(rank_csv)
    ranked = ranked[(ranked.condition == condition) & (ranked.seed == seed)].sort_values("rank")
    if tuple(ranked.unit) != units:
        raise ValueError(f"Archived tile order changed: {source_svg}")
    selected = pd.read_csv(unit_csv)
    selected = selected[(selected.condition == condition) &
                        (selected.seed == seed) &
                        (selected.unit.isin(units))].set_index("unit")
    if set(selected.index) != set(units):
        raise ValueError(f"Missing archived unit rows: {source_svg}")
    probe = pd.read_csv(probe_csv)
    matching_probe = probe[(probe.condition == condition) & (probe.seed == seed)]
    if len(matching_probe) != 1 or int(matching_probe.iloc[0].trajectory_samples) != 10001:
        raise ValueError(f"Unexpected archived probe: {source_svg}")
    svg_root = ET.parse(source_svg).getroot()
    px_per_svg_unit = 1200 / float(svg_root.attrib["viewBox"].split()[2])
    bounds = []
    for axis_number in range(1, 5):
        axis = next(node for node in svg_root.iter()
                    if node.attrib.get("id") == f"axes_{axis_number}")
        patch = next(node for node in axis
                     if node.attrib.get("id", "").startswith("patch_"))
        outline = next(node for node in patch if node.tag.endswith("path"))
        numbers = [float(value) for value in re.findall(
            r"-?\d+(?:\.\d+)?", outline.attrib["d"])]
        points = list(zip(numbers[::2], numbers[1::2]))
        bounds.append(tuple(round(extreme * px_per_svg_unit) for extreme in (
            min(x for x, _ in points), min(y for _, y in points),
            max(x for x, _ in points), max(y for _, y in points))))
    if any(abs((box[2] - box[0]) - (box[3] - box[1])) > 1 for box in bounds):
        raise ValueError(f"Non-square archived map bounds: {source_svg}")
    with tempfile.TemporaryDirectory(prefix="cosyne_field_option_") as temporary:
        rendered = Path(temporary) / "tiles.png"
        subprocess.run(["inkscape", str(source_svg),
                        f"--export-filename={rendered}", "--export-width=1200"],
                       check=True, capture_output=True, text=True)
        with Image.open(rendered) as source:
            if source.width != 1200 or source.height < 1064:
                raise ValueError(f"Unexpected archived panel dimensions: {source_svg}")
            # The SVG axes bounds vary across runs because colorbar labels
            # affect the saved canvas width. Derive each crop from its own
            # axes patch, excluding tick marks while retaining the full map.
            tiles = {unit: source.crop(box).convert("RGB")
                     for unit, box in zip(units, bounds)}
    metrics = {unit: {
        "si": float(ranked.set_index("unit").loc[unit, "spatial_score"]),
        "mono_score": float(selected.loc[unit, "mono_score"]),
        "mono_field": bool(selected.loc[unit, "mono_field"]),
        "active_fraction": float(selected.loc[unit, "active_fraction"]),
    } for unit in units}
    return (tiles, metrics, int(matching_probe.iloc[0].visited_bins_probe),
            "independent per-unit scales", [source_svg, rank_csv, unit_csv, probe_csv])


def render_option(name: str, title: str, tiles: dict, metrics: dict,
                  visited_bins: int) -> None:
    fig = plt.figure(figsize=(7.5, 3.4), facecolor="white")
    fig.text(0.025, 0.956, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.025, 0.853,
             f"100M-frame frozen probe · 10,001 decisions · visited {visited_bins}/361 bins",
             fontsize=12, va="top")
    for position, (unit, tile) in enumerate(tiles.items()):
        x = 0.025 + position * 0.246
        ax = fig.add_axes([x, 0.324, 0.215, 0.465])
        ax.imshow(tile, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#61717A")
            spine.set_linewidth(0.6)
        row = metrics[unit]
        fig.text(x + 0.1075, 0.286, f"DG {unit}", fontsize=12,
                 ha="center", va="top", fontweight="bold")
        fig.text(x + 0.1075, 0.215, f"SI {row['si']:.3f}",
                 fontsize=12, ha="center", va="top")
        fig.text(x + 0.1075, 0.145,
                 f"mono {row['mono_score']:.2f}" +
                 (" (strict)" if row["mono_field"] else ""),
                 fontsize=12, ha="center", va="top",
                 color="#16664B" if row["mono_field"] else "#59666D")
        fig.text(x + 0.1075, 0.075,
                 f"active {100 * row['active_fraction']:.1f}%",
                 fontsize=12, ha="center", va="top")
    for suffix in ("pdf", "png"):
        destination = OUTPUT / f"{name}.{suffix}"
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(destination, dpi=300, metadata=metadata)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    font = style()
    rows = []
    manifest = {"schema": "intrmotiv/figure1-field-options/v1", "font": font,
                "options": []}
    for name, title, kind, batch, condition, seed, units in OPTIONS:
        if kind == "cosyne":
            tiles, metrics, visited, scale_note, sources = cosyne_option(
                batch, condition, seed, units)
        else:
            tiles, metrics, visited, scale_note, sources = archive_option(
                condition, seed, units)
            scale_note += "; white means unvisited"
        render_option(name, title, tiles, metrics, visited)
        entry = {"name": name, "title": title, "kind": kind,
                 "condition": condition, "seed": seed,
                 "checkpoint_frames": 100007936 if kind == "cosyne" else 100040704,
                 "visited_bins": visited, "total_bins": 361,
                 "source_scale": scale_note,
                 "units": {str(unit): metrics[unit] for unit in units},
                 "source_sha256": {str(path.relative_to(DRAFT.parents[1])):
                                    hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sources}}
        manifest["options"].append(entry)
        for unit in units:
            rows.append({"option": name, "kind": kind, "condition": condition,
                         "seed": seed, "checkpoint_frames": entry["checkpoint_frames"],
                         "visited_bins": visited, "unit": unit, **metrics[unit]})
    pd.DataFrame(rows).to_csv(OUTPUT / "selected_units.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    previews = []
    for name, *_ in OPTIONS:
        with Image.open(OUTPUT / f"{name}.png") as source:
            previews.append(source.convert("RGB").resize(
                (1000, round(source.height * 1000 / source.width)),
                Image.Resampling.LANCZOS))
    separator = 16
    overview = Image.new("RGB", (1000, sum(im.height for im in previews) +
                                  separator * (len(previews) - 1)), "white")
    y = 0
    for preview in previews:
        overview.paste(preview, (0, y))
        y += preview.height + separator
    overview.save(OUTPUT / "all_five_overview.png")


if __name__ == "__main__":
    main()
