"""Render screened, eight-unit DG galleries for each COSYNE architecture.

These are selection sheets, not replacement submission figures. Each row is
one saved terminal policy probe; the source atlas and evaluator tables remain
untouched. Selections exclude the visually diffuse maps in these six atlases.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from render_architecture_with_fields import DRAFT, style
from render_place_field_options import cosyne_option


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "place_field_options" / "eight_per_architecture"

# Four examples per policy probe keep the color scale and visited-bin mask
# consistent within a row. The two rows of a sheet are separate trained runs.
SHEETS = (
    ("c01", "C01 · direct exploration", (
        ("temporal DG credit", "c01_temporal_s3", "COS27_C01_TEMPORAL", 99,
         (6, 15, 5, 0)),
        ("constant DG credit", "c01_constant_s3", "COS27_C01_CONSTANT", 99,
         (2, 10, 7, 15)),
    )),
    ("c05", "C05 · DG-target navigation", (
        ("constant DG / temporal worker", "c05_pushpull_s3",
         "COS27_C05_CONSTANT_TEMPORAL", 99, (5, 12, 7, 6)),
        ("temporal DG / temporal worker", "c05_pushpull_s3",
         "COS27_C05_TEMPORAL_TEMPORAL", 99, (11, 7, 3, 5)),
    )),
    ("c15", "C15 · frontier-selected DG targets", (
        ("temporal DG credit", "c15_available", "COS27_C15_TEMPORAL", 123,
         (4, 3, 7, 11)),
        ("constant DG credit", "c15_available", "COS27_C15_CONSTANT", 123,
         (0, 11, 3, 14)),
    )),
)


def draw_sheet(name: str, title: str, rows: list[dict]) -> None:
    fig = plt.figure(figsize=(7.5, 8.2), facecolor="white")
    fig.text(0.03, 0.984, title, va="top", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.948,
             "Eight screened DG maps · terminal 100M-frame frozen-policy probes",
             va="top", fontsize=12)
    for row_index, row in enumerate(rows):
        label_y = (0.892, 0.477)[row_index]
        map_y = (0.661, 0.246)[row_index]
        fig.text(0.03, label_y,
                 f"{row['label']} · seed {row['seed']} · "
                 f"{row['visited']}/361 bins visited ({row['visited'] / 361:.0%})",
                 va="top", fontsize=12, fontweight="bold")
        for column, unit in enumerate(row["units"]):
            x = 0.030 + column * 0.247
            ax = fig.add_axes([x, map_y, 0.205, 0.1875])
            ax.imshow(row["tiles"][unit], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#5C6870")
                spine.set_linewidth(0.7)
            metric = row["metrics"][unit]
            center = x + 0.1025
            fig.text(center, map_y - 0.010, f"DG {unit}", ha="center", va="top",
                     fontsize=12, fontweight="bold")
            fig.text(center, map_y - 0.047,
                     f"SI {metric['si']:.3f} bits", ha="center", va="top", fontsize=12)
            fig.text(center, map_y - 0.084,
                     f"mono {metric['mono_score']:.2f}" +
                     (" · strict" if metric["mono_field"] else ""),
                     ha="center", va="top", fontsize=12,
                     color="#16664B" if metric["mono_field"] else "#4D5961")
            fig.text(center, map_y - 0.121,
                     f"active {100 * metric['active_fraction']:.1f}%",
                     ha="center", va="top", fontsize=12)
    fig.text(0.03, 0.048,
             "Gray = unvisited. Color is comparable within each row only.",
             va="top", fontsize=12)
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(OUTPUT / f"{name}_eight_screened_fields.{suffix}", dpi=240,
                    metadata=metadata)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = style()
    manifest = {"schema": "intrmotiv/figure1-eight-field-gallery/v1",
                "font": font, "sheets": []}
    selection_rows = []
    for name, title, specifications in SHEETS:
        rows = []
        manifest_rows = []
        for label, batch, condition, seed, units in specifications:
            tiles, metrics, visited, scale_note, sources = cosyne_option(
                batch, condition, seed, units)
            rows.append({"label": label, "condition": condition, "seed": seed,
                         "units": units, "tiles": tiles, "metrics": metrics,
                         "visited": visited})
            manifest_rows.append({
                "label": label, "condition": condition, "seed": seed,
                "checkpoint_frames": 100007936, "probe_decisions": 10001,
                "visited_bins": visited, "total_bins": 361,
                "source_scale": scale_note,
                "units": {str(unit): metrics[unit] for unit in units},
                "source_sha256": {
                    str(path.relative_to(DRAFT.parents[1])):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources},
            })
            for unit in units:
                selection_rows.append({
                    "architecture": name.upper(), "row_label": label,
                    "condition": condition, "seed": seed,
                    "checkpoint_frames": 100007936, "probe_decisions": 10001,
                    "visited_bins": visited, "unit": unit, **metrics[unit],
                })
        draw_sheet(name, title, rows)
        manifest["sheets"].append({"architecture": name.upper(),
                                   "title": title, "rows": manifest_rows})
    pd.DataFrame(selection_rows).to_csv(OUTPUT / "selected_units.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
