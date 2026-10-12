"""Render an independent, compact Figure 1 layout experiment.

The submitted architecture, Figure 2, TeX document, and ZIP are never written.
Numerical and archived image sources are shared with render_submission_panels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
from PIL import Image

DRAFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRAFT))
from render_submission_panels import COLOR, SOURCES, style  # noqa: E402


HERE = Path(__file__).resolve().parent
STEM = "01_architecture_spatial_right"
INK = "#1A3448"
RED = "#A33C3C"
BLUE = "#17679C"
ARCHIVE = (DRAFT.parents[1] / "06_experiments" / "results" /
           "A0_poster_analysis_20260926" / "flat_goal_comparison")
FIELD_SOURCE = ARCHIVE / "c15_seed99" / "place_fields_individual_scale.svg"
FIELD_TABLE = ARCHIVE / "place_field_individual_scales.csv"
PROBE_TABLE = ARCHIVE / "matched_terminal_per_run.csv"


def box(ax, x, y, width, height, lines, *, edge=INK, fill="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.2, edgecolor=edge, facecolor=fill))
    ax.text(x + width / 2, y + height / 2, lines, ha="center", va="center",
            fontsize=12, color=INK, linespacing=1.12)


def arrow(ax, start, end, color=INK, *, bend=0, width=1.3):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        connectionstyle=f"arc3,rad={bend}", linewidth=width, color=color))


def architecture(fig):
    ax = fig.add_axes([0.015, 0.045, 0.59, 0.91])
    ax.set_xlim(0, 4.45)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    ax.text(0.0, 2.96, "A  Vision to action", va="top", fontsize=12,
            fontweight="bold", color="#20272C")

    # Four modules preserve the original causal flow without shrinking its
    # detailed vector diagram or retaining the equations that occupied it.
    module_y = 1.60
    module_height = 0.88
    modules = [
        (0.00, 0.89, "Visual\ntrunk", INK, "#F5F7F8"),
        (1.12, 0.89, "Sparse\nDG", RED, "#F9F1F1"),
        (2.24, 0.80, "CA3\nmemory", INK, "#F5F7F8"),
        (3.27, 1.11, "Readout\nMLP", BLUE, "#F0F5F9"),
    ]
    for x, width, label, edge, fill in modules:
        box(ax, x, module_y, width, module_height, label, edge=edge, fill=fill)
    for start, end in [((0.89, 2.04), (1.11, 2.04)),
                       ((2.01, 2.04), (2.23, 2.04)),
                       ((3.04, 2.04), (3.26, 2.04))]:
        arrow(ax, start, end)

    ax.text(0.03, 2.58, "RGB", fontsize=12, color=INK)
    ax.text(1.30, 2.58, "sparse events", fontsize=12, color=RED)
    ax.text(3.54, 2.58, "5 actions", fontsize=12, color=BLUE)
    box(ax, 0.04, 0.11, 2.09, 0.73,
        "Push: strengthen\nlater DG events", edge=RED, fill="#F9F1F1")
    box(ax, 2.28, 0.11, 2.09, 0.73,
        "Pull: favor shorter\nevent transitions", edge=BLUE, fill="#F0F5F9")
    arrow(ax, (1.54, 0.86), (1.54, 1.58), RED, width=1.5)
    arrow(ax, (3.81, 0.86), (3.81, 1.58), BLUE, width=1.5)


def spatial_emergence(fig):
    online = pd.read_csv(SOURCES["online"])
    ax = fig.add_axes([0.735, 0.585, 0.245, 0.225])
    for family, worker_arm in (("C01", "inapplicable"),
                               ("C05", "temporal"), ("C15", "temporal")):
        rows = online[(online.family == family) &
                      (online.encoder_credit == "temporal") &
                      (online.worker_bonus == worker_arm) &
                      (online.metric == "spatial_information")].sort_values("env_steps")
        assert len(rows) == 99 and rows.n.eq(5).all()
        x = rows.env_steps.to_numpy() / 1e6
        mean = rows["mean"].to_numpy()
        sem = rows["sem"].to_numpy()
        ax.plot(x, mean, color=COLOR[family], linewidth=1.35, label=family)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=COLOR[family], alpha=0.16, linewidth=0)
    ax.set(xlim=(0, 100), ylim=(0, 0.22), xticks=[0, 50, 100],
           yticks=[0, 0.1, 0.2], xlabel="Frames (millions)",
           ylabel="bits/step")
    ax.grid(axis="y", color="#E6E8E9", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.6, pad=2)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.07), ncol=3,
              frameon=False, fontsize=12, handlelength=0.7,
              handletextpad=0.25, columnspacing=0.45, borderaxespad=0)
    fig.text(0.625, 0.967, "B  Spatial coding emerges", va="top",
             fontsize=12, fontweight="bold", color="#20272C")


def example_fields(fig):
    """Crop two audited C15 seed-99 maps from the higher-coverage probe."""
    probe = pd.read_csv(PROBE_TABLE)
    row = probe[(probe.condition == "C15") & (probe.seed == 99)].squeeze()
    assert int(row.visited_bins_probe) == 310
    scales = pd.read_csv(FIELD_TABLE)
    selected = scales[(scales.condition == "C15") & (scales.seed == 99)]
    assert list(selected.unit) == [7, 11, 1, 0]
    # These stable export coordinates omit axes, titles, and the original
    # per-unit colorbars. The caption states that the scales are independent.
    crops = {"DG 7": (147, 211, 525, 590),
             "DG 0": (663, 684, 1043, 1064)}
    with tempfile.TemporaryDirectory(prefix="cosyne_figure1_fields_") as temporary:
        rendered = Path(temporary) / "fields.png"
        subprocess.run(["inkscape", str(FIELD_SOURCE),
                        f"--export-filename={rendered}", "--export-width=1200"],
                       check=True, capture_output=True, text=True)
        source = Image.open(rendered).convert("RGB")
        assert source.width == 1200 and source.height >= 1064
        tiles = {name: source.crop(bounds).copy() for name, bounds in crops.items()}
    fig.text(0.625, 0.415, "C  Example DG place fields", va="top",
             fontsize=12, fontweight="bold", color="#20272C")
    for x, name in ((0.674, "DG 7"), (0.831, "DG 0")):
        ax = fig.add_axes([x, 0.095, 0.13, 0.27])
        ax.imshow(tiles[name], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_edgecolor("#9CA9AD")
        fig.text(x + 0.065, 0.044, name, ha="center", va="bottom", fontsize=12,
                 color=INK)


def main():
    font = style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 3.12), facecolor="white")
    architecture(fig)
    spatial_emergence(fig)
    example_fields(fig)
    for suffix in ("pdf", "svg", "png"):
        target = HERE / f"{STEM}.{suffix}"
        metadata = ({"Date": None} if suffix == "svg" else
                    {"CreationDate": None, "ModDate": None} if suffix == "pdf"
                    else None)
        fig.savefig(target, dpi=300, metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in
                                        target.read_text().splitlines()) + "\n")
    plt.close(fig)
    inputs = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in (("online", SOURCES["online"]),
                                 ("archived_place_fields", FIELD_SOURCE),
                                 ("field_rank_table", FIELD_TABLE),
                                 ("probe_table", PROBE_TABLE))}
    manifest = {"figure": STEM, "size_inches": [7.5, 3.12],
                "font": font, "sources_sha256": inputs,
                "separate_archived_example": True}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
