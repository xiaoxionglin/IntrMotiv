"""Render Figure 1 candidate 2 with the author's original architecture art.

The full author SVG has equations outside its diagram. This adapter crops the
diagram without modifying that source, then reuses the current candidate's
quantitative and field panels on the right.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from render_architecture_with_fields import (
    DRAFT, FIELD_SOURCE, FIELD_TABLE, HERE, PROBE_TABLE, SOURCES,
    example_fields, spatial_emergence, style,
)


STEM = "02_original_architecture_spatial_right"
ARCHITECTURE_SOURCE = DRAFT / "figures" / "architecture_v2.svg"


def load_original_artwork() -> Image.Image:
    """Crop the original modules and illustrations, excluding all equations."""
    with tempfile.TemporaryDirectory(prefix="cosyne_original_architecture_") as temporary:
        rendered = Path(temporary) / "architecture.png"
        subprocess.run(
            ["inkscape", str(ARCHITECTURE_SOURCE),
             f"--export-filename={rendered}", "--export-width=3600"],
            check=True, capture_output=True, text=True)
        source = Image.open(rendered).convert("RGB")
        assert source.width == 3600 and source.height >= 2260
        # Coordinates are twice those measured on the stable 1800-pixel
        # export. This crop retains the original sensory scene, encoder, DG,
        # CA3, readout, maze, temporal-separation inset, arrows, and legend;
        # the separate loss and state equations lie farther down the page.
        art = source.crop((180, 1320, 3090, 2260))
        # The top-right observation-update formula is inside the diagram's
        # rectangular crop. Cover only that formula, above the feedback arrow.
        ImageDraw.Draw(art).rectangle((2180, 60, 2800, 176), fill="white")
    return art


def original_artwork(fig: plt.Figure) -> None:
    """Show the author's architecture art at the candidate-2 dimensions."""
    art = load_original_artwork()

    fig.text(0.015, 0.967, "A  Push–pull navigation architecture",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    ax = fig.add_axes([0.004, 0.285, 0.605, 0.615])
    ax.imshow(art, interpolation="lanczos")
    ax.axis("off")

    # Keep the critical architecture roles readable at the candidate's
    # compact print width; the original internal SVG labels are smaller.
    for y, label in (
        (0.244, "Fixed: ResNet-18 visual trunk and CA3 memory."),
        (0.165, "Learned: DG projection, batch normalization, readout."),
        (0.086, "Red push trains DG; blue pull shapes the policy."),
    ):
        fig.text(0.017, y, label, fontsize=12, color="#1A3448", va="center")


def main() -> None:
    font = style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 3.12), facecolor="white")
    original_artwork(fig)
    spatial_emergence(fig)
    example_fields(fig)
    for suffix in ("pdf", "svg", "png"):
        target = HERE / f"{STEM}.{suffix}"
        metadata = ({"Date": None} if suffix == "svg" else
                    {"CreationDate": None, "ModDate": None} if suffix == "pdf"
                    else None)
        fig.savefig(target, dpi=600 if suffix != "png" else 300,
                    metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in
                                        target.read_text().splitlines()) + "\n")
    plt.close(fig)

    inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in (
                  ("original_architecture", ARCHITECTURE_SOURCE),
                  ("online", SOURCES["online"]),
                  ("archived_place_fields", FIELD_SOURCE),
                  ("field_rank_table", FIELD_TABLE),
                  ("probe_table", PROBE_TABLE),
              )}
    manifest = {"figure": STEM, "size_inches": [7.5, 3.12],
                "font": font, "sources_sha256": inputs,
                "pdf_svg_raster_dpi": 600,
                "architecture_export_width_px": 3600,
                "architecture_crop_px": [180, 1320, 3090, 2260],
                "architecture_formula_mask_in_crop_px": [2180, 60, 2800, 176],
                "separate_archived_example": True}
    (HERE / f"{STEM}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
