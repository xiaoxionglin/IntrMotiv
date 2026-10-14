"""Build a Figure 2 story variant with three matched-run peak-spread maps.

The maps use the same training runs as the panel-A trajectory examples, but
their peak coordinates come from separate terminal 100k-decision online
snapshots. The existing story figure remains a separate selectable artifact.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from build_figure2_story import (
    COLORS, HERE, ROOT, RUNS, behavior_differences, export, layout_base,
    selected_trajectories, sha256, story_overlay, terminal_rows,
)


RESULTS = ROOT / "06_experiments/results/cosyne_2027_snapshot_peak_spread"
UNITS = {
    "NG": RESULTS / "inputs/c01_temporal_s3_per_unit.csv",
    "UG": RESULTS / "inputs/c05_pushpull_s3_per_unit.csv",
    "FG": RESULTS / "inputs/c15_encoder_s2_per_unit.csv",
}
SPREAD_TABLE = RESULTS / "per_snapshot_cutoff_spread.csv"
MASK_PATH = RESULTS / "common_visited_mask.json"
ANALYSIS_SOURCE = ROOT / "06_experiments/cosyne_2027/analyze_snapshot_peak_spread.py"
OUTPUT = HERE / "Figure2_story_peaks.svg"
DATA = HERE / "Figure2_story_peaks_data.csv"
MANIFEST = HERE / "Figure2_story_peaks_manifest.json"
CUTOFF = 0.0025
TARGET_STEPS = 100_000_000
DENSITY_TOP = 0.04  # shared normalized peak mass per 100-by-100 spatial bin
MAP_LEFT = 477.0
MAP_SIZE = 58.0
MAP_TOPS = {"NG": 26.0, "UG": 97.0, "FG": 168.0}


def peak_analysis():
    """Reuse the established density and effective-area calculations."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("06_experiments.cosyne_2027.analyze_snapshot_peak_spread")


def map_inputs() -> tuple[dict[str, dict], list[dict[str, object]]]:
    analysis = peak_analysis()
    mask_record = json.loads(MASK_PATH.read_text())
    mask = np.asarray(mask_record["mask"], dtype=bool)
    if mask.shape != (19, 19) or int(mask.sum()) != 318:
        raise ValueError("The shared terminal visited-area mask changed")
    summary = pd.read_csv(SPREAD_TABLE)
    maps = {}
    peak_rows = []
    for family, run_name in RUNS.items():
        frame = pd.read_csv(UNITS[family])
        frame = frame[(frame.run_name == run_name)
                      & (frame.target_env_steps == TARGET_STEPS)].sort_values("unit_id")
        if len(frame) != 16 or frame.unit_id.nunique() != 16:
            raise ValueError(f"Expected 16 terminal DG units for {run_name}")
        finite = np.isfinite(frame[["dominant_peak_x", "dominant_peak_y"]]).all(axis=1)
        valid = ((frame.active_fraction >= CUTOFF)
                 & frame.field_eligible.astype(bool) & finite)
        selected = frame.loc[valid]
        xy = selected[["dominant_peak_x", "dominant_peak_y"]].to_numpy(float)
        grid = analysis.peak_grid(xy, mask)
        effective_area = analysis.spread_metrics(xy, mask)["effective_area_fraction"]
        expected = summary[(summary.run_name == run_name)
                           & (summary.target_env_steps == TARGET_STEPS)
                           & np.isclose(summary.cutoff, CUTOFF)]
        if len(expected) != 1 or not math.isclose(
                effective_area, float(expected.iloc[0].effective_area_fraction),
                rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Peak-spread value disagrees with the archived table: {run_name}")
        if not np.isclose(grid.sum(), 1.0) or np.nanmax(grid) > DENSITY_TOP:
            raise ValueError(f"Peak-density normalization or shared scale changed: {run_name}")
        maps[family] = {"run_name": run_name, "xy": xy, "grid": grid,
                        "mask": mask, "effective_area": effective_area,
                        "valid_peaks": len(xy)}
        for row in selected.itertuples():
            peak_rows.append({"family": family, "run_name": run_name,
                              "unit_id": int(row.unit_id),
                              "dominant_peak_x": float(row.dominant_peak_x),
                              "dominant_peak_y": float(row.dominant_peak_y),
                              "active_fraction": float(row.active_fraction)})
    return maps, peak_rows


def peak_panel(maps: dict[str, dict]) -> str:
    """Draw three vector maps on a single spatial and density scale."""
    cell = MAP_SIZE / 19
    items = ['<g id="peak-spread-panel" aria-label="Terminal DG peak distributions">']
    for family in ("NG", "UG", "FG"):
        source = maps[family]
        top = MAP_TOPS[family]
        items.append(f'<g id="peak-map-{family}">')
        items.append(f'<text x="{MAP_LEFT:g}" y="{top-3:g}" class="peak-label" '
                     f'fill="{COLORS[family]}">{family}</text>')
        items.append(f'<text x="{MAP_LEFT+MAP_SIZE:g}" y="{top-3:g}" '
                     'text-anchor="end" class="peak-label" fill="#25313a">'
                     f'{source["effective_area"]:.2f}</text>')
        for row in range(19):
            for column in range(19):
                if source["mask"][row, column]:
                    intensity = min(float(source["grid"][row, column]) / DENSITY_TOP, 1.0)
                    gray = round(255 - 200 * intensity)
                    fill = f'#{gray:02x}{gray:02x}{gray:02x}'
                else:
                    fill = '#ededed'
                x = MAP_LEFT + column * cell
                y = top + (18 - row) * cell
                items.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{cell:.3f}" '
                             f'height="{cell:.3f}" fill="{fill}"/>')
        for peak_x, peak_y in source["xy"]:
            x = MAP_LEFT + (peak_x - 100) * MAP_SIZE / 1900
            y = top + MAP_SIZE - (peak_y - 100) * MAP_SIZE / 1900
            items.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="1.4" '
                         f'fill="{COLORS[family]}" stroke="white" stroke-width="0.45"/>')
        items.append(f'<rect x="{MAP_LEFT:g}" y="{top:g}" width="{MAP_SIZE:g}" '
                     f'height="{MAP_SIZE:g}" fill="none" stroke="#5c6870" '
                     'stroke-width="0.6"/>')
        items.append('</g>')
    items.append('</g>')
    return '\n'.join(items)


def build() -> None:
    maps, peak_rows = map_inputs()
    svg = layout_base(extra_plot_shift=-4)
    old_name = 'sodipodi:docname="Figure2_story.svg"'
    if svg.count(old_name) != 1:
        raise ValueError("Could not identify the story canvas document name")
    svg = svg.replace(old_name,
                      'sodipodi:docname="Figure2_story_peaks.svg"', 1)
    overlay = story_overlay(terminal_rows(), selected_trajectories(),
                            behavior_differences(), peak_panel=peak_panel(maps),
                            trajectory_tile_size=42, elbow_arrows=True)
    svg = re.sub(r'</svg>\s*$', overlay + '\n</svg>\n', svg)
    export(svg, OUTPUT)
    with DATA.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(peak_rows[0]))
        writer.writeheader()
        writer.writerows(peak_rows)
    MANIFEST.write_text(json.dumps({
        "schema": "cosyne27-figure2-story-peak-variant/v1",
        "parent_figure": "Figure2_story.svg",
        "physical_size_points": [540, 230],
        "trajectory_tile_points": 42,
        "peak_map_size_points": MAP_SIZE,
        "peak_map_data": str(DATA.relative_to(ROOT)),
        "sample_protocol": "terminal 100k-decision online snapshots; separate from panel-A 10k frozen-policy trajectory probes",
        "target_env_steps": TARGET_STEPS,
        "field_eligibility": "at least 20 active observations in at least three spatial bins",
        "active_fraction_cutoff": CUTOFF,
        "gaussian_sigma_dmlab_units": 100,
        "map_world_bounds_dmlab_units": [100, 2000, 100, 2000],
        "shared_density_display_range_mass_per_bin": [0, DENSITY_TOP],
        "shared_visited_mask": str(MASK_PATH.relative_to(ROOT)),
        "shared_visited_mask_sha256": sha256(MASK_PATH),
        "derived_peak_spread_table": str(SPREAD_TABLE.relative_to(ROOT)),
        "derived_peak_spread_table_sha256": sha256(SPREAD_TABLE),
        "canonical_peak_analysis": str(ANALYSIS_SOURCE.relative_to(ROOT)),
        "canonical_peak_analysis_sha256": sha256(ANALYSIS_SOURCE),
        "examples": {family: {
            "run_name": maps[family]["run_name"],
            "source_per_unit_table": str(UNITS[family].relative_to(ROOT)),
            "source_per_unit_sha256": sha256(UNITS[family]),
            "valid_peaks": maps[family]["valid_peaks"],
            "effective_area_fraction": maps[family]["effective_area"],
        } for family in ("NG", "UG", "FG")},
    }, indent=2) + '\n')


if __name__ == '__main__':
    build()
