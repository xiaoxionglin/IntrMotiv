"""Render Figure 1 candidate 3: large original art, compact results.

This adapter changes only a separate Figure 1 candidate. It reuses the
author's architecture artwork and the audited online and place-field sources.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from PIL import ImageDraw

from render_architecture_with_fields import (
    FIELD_SOURCE, FIELD_TABLE, HERE, INK, PROBE_TABLE, SOURCES,
    load_archived_field_tiles, style,
)
from render_original_architecture_candidate import (
    ARCHITECTURE_SOURCE, load_original_artwork,
)
from render_submission_panels import COLOR


STEM = "03_architecture_priority_compact_results"
FIGURE_SIZE = (7.5, 3.12)
PANEL_TITLE_SIZE = 11


def architecture(fig: plt.Figure) -> None:
    fig.text(0.012, 0.967, "A  Push–pull navigation architecture",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    ax = fig.add_axes([0.003, 0.293, 0.772, 0.625])
    art = load_original_artwork()
    # The authored layout was lettered for a much wider page. Blank only the
    # word-label regions, then redraw the same names at 10 pt. Coordinates are
    # from the 770-pixel inspection export of the 2910-pixel-wide crop.
    scale = art.width / 770
    masks = (
        (120, 26, 615, 47, "white"),  # original legend
        (143, 53, 251, 75, "white"),   # module names
        (280, 53, 318, 76, "white"),
        (399, 53, 451, 76, "white"),
        (525, 53, 608, 77, "white"),
        (50, 157, 119, 176, "white"), # input dimensions
        (143, 101, 236, 142, "white"),
        (261, 146, 355, 166, "white"),
        (300, 119, 343, 138, "white"),
        (477, 82, 503, 143, "white"),
        (519, 97, 611, 116, "white"),
        (519, 118, 598, 138, "white"),
        (325, 179, 472, 210, (237, 244, 242)),
        (672, 178, 760, 203, "white"),
    )
    draw = ImageDraw.Draw(art)
    for x0, y0, x1, y1, fill in masks:
        draw.rectangle((round(x0 * scale), round(y0 * scale),
                        round(x1 * scale), round(y1 * scale)), fill=fill)
    ax.imshow(art, interpolation="lanczos")

    def label(x: float, y: float, value: str, *, size: float = 10,
              rotation: float = 0, color: str = "#183043") -> None:
        ax.text(x * scale, y * scale, value, ha="center", va="center",
                fontsize=size, rotation=rotation, color=color,
                linespacing=0.9)

    for x, edge, text_x, name in (
        (124, "#20272C", 188, "Fixed"),
        (225, "#A33C3C", 341, "temporal push loss"),
        (435, "#17679C", 540, "temporal pull loss"),
    ):
        ax.add_patch(Rectangle((x * scale, 29 * scale), 29 * scale,
                               11 * scale, fill=False, edgecolor=edge,
                               linewidth=1.0))
        label(text_x, 35, name, color="#20272C")
    for x, y, value in (
        (196, 65, "Vis. Encoder"), (299, 65, "DG"),
        (424, 65, "CA3"), (566, 65, "Readout"),
        (84, 165, "96×72×3"),
        (190, 119, "Layer 2\nResNet-18"),
        (305, 157, "Sparsifier"), (315, 129, "~2.5%"),
        (574, 107, "2×MLP (128)"), (557, 127, "Action(5)"),
        (399, 195, "Temporal\nseparation"),
        (719, 190, "DG event"),
    ):
        label(x, y, value)
    label(489, 112, "Flatten", rotation=90)
    ax.axis("off")
    for y, label in (
        (0.245, "Fixed: ResNet-18 visual trunk and CA3 memory."),
        (0.165, "Learned: DG projection, batch normalization, readout."),
        (0.085, "Red push trains DG; blue pull shapes the policy."),
    ):
        fig.text(0.015, y, label, fontsize=11, color=INK, va="center")


def spatial_emergence(fig: plt.Figure) -> None:
    """Plot the five-seed online spatial information in a narrow top strip."""
    online = pd.read_csv(SOURCES["online"])
    ax = fig.add_axes([0.831, 0.681, 0.145, 0.189])
    arms = (("C01", "inapplicable"), ("C05", "temporal"),
            ("C15", "temporal"))
    for family, worker_arm in arms:
        rows = online[(online.family == family) &
                      (online.encoder_credit == "temporal") &
                      (online.worker_bonus == worker_arm) &
                      (online.metric == "spatial_information")].sort_values("env_steps")
        assert len(rows) == 99 and rows.n.eq(5).all()
        x = rows.env_steps.to_numpy() / 1e6
        mean, sem = rows["mean"].to_numpy(), rows["sem"].to_numpy()
        ax.plot(x, mean, color=COLOR[family], linewidth=1.1, label=family)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=COLOR[family], alpha=0.17, linewidth=0)
    ax.set(xlim=(0, 100), ylim=(0, 0.22), xticks=[0, 50, 100],
           yticks=[0, 0.1, 0.2], xlabel="Frames (M)", ylabel="bits/step")
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(labelsize=8.5, length=2, width=0.6, pad=1)
    ax.grid(axis="y", color="#E6E8E9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 1.04), ncol=3,
              frameon=False, fontsize=8.5, handlelength=0.65,
              handletextpad=0.15, columnspacing=0.27, borderaxespad=0)
    fig.text(0.79, 0.967, "B  Spatial rise", va="top",
             fontsize=PANEL_TITLE_SIZE, fontweight="bold", color="#20272C")


def example_fields(fig: plt.Figure) -> None:
    """Show all four ranked example maps on independent color scales."""
    tiles = load_archived_field_tiles()
    fig.text(0.79, 0.565, "C  Example DG fields", va="top",
             fontsize=PANEL_TITLE_SIZE, fontweight="bold", color="#20272C")
    for name, x, y in (
        ("DG 7", 0.798, 0.310), ("DG 11", 0.894, 0.310),
        ("DG 1", 0.798, 0.055), ("DG 0", 0.894, 0.055),
    ):
        ax = fig.add_axes([x, y, 0.084, 0.202])
        ax.imshow(tiles[name], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#9CA9AD")
        fig.text(x + 0.042, y - 0.009, name, ha="center", va="top",
                 fontsize=8.5, color=INK)


def main() -> None:
    font = style()
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    architecture(fig)
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
    manifest = {
        "figure": STEM, "size_inches": list(FIGURE_SIZE), "font": font,
        "sources_sha256": inputs, "pdf_svg_raster_dpi": 600,
        "architecture_export_width_px": 3600,
        "architecture_crop_px": [180, 1320, 3090, 2260],
        "architecture_formula_mask_in_crop_px": [2180, 60, 2800, 176],
        "redrawn_architecture_word_labels_pt": 10,
        "original_artwork_elements_preserved": True,
        "field_units": [7, 11, 1, 0],
        "field_color_scale": "independent per unit, zero-based",
    }
    (HERE / f"{STEM}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
