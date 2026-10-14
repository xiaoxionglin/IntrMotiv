"""Build a Figure 2 variant with three paired temporal/control DG peak maps.

The temporal maps use the panel-A trajectory runs; controls match their
architecture and seed. Peak coordinates come from separate terminal
100k-decision online snapshots. The field-map figure remains selectable.
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
RUN_PAIRS = {
    "NG": {"Temporal": RUNS["NG"], "Control": "COS27_C01_CONSTANT_S99"},
    "UG": {"Temporal": RUNS["UG"], "Control": "COS27_C05_CONSTANT_TEMPORAL_S123"},
    "FG": {"Temporal": RUNS["FG"], "Control": "COS27_C15_CONSTANT_S17"},
}
UNITS = {
    ("NG", "Temporal"): RESULTS / "inputs/c01_temporal_s3_per_unit.csv",
    ("NG", "Control"): RESULTS / "inputs/c01_constant_s3_per_unit.csv",
    ("UG", "Temporal"): RESULTS / "inputs/c05_pushpull_s3_per_unit.csv",
    ("UG", "Control"): RESULTS / "inputs/c05_pushpull_s3_per_unit.csv",
    ("FG", "Temporal"): RESULTS / "inputs/c15_encoder_s2_per_unit.csv",
    ("FG", "Control"): RESULTS / "inputs/c15_encoder_s2_per_unit.csv",
}
SPREAD_TABLE = RESULTS / "per_snapshot_cutoff_spread.csv"
MASK_PATH = RESULTS / "common_visited_mask.json"
ANALYSIS_SOURCE = ROOT / "06_experiments/cosyne_2027/analyze_snapshot_peak_spread.py"
OUTPUT = HERE / "Figure2_story_peaks.svg"
DATA = HERE / "Figure2_story_peaks_data.csv"
MANIFEST = HERE / "Figure2_story_peaks_manifest.json"
CUTOFF = 0.0025
TARGET_STEPS = 100_000_000
MAP_LEFTS = {"Temporal": 445.0, "Control": 493.0}
MAP_SIZE = 43.0
MAP_TOPS = {"NG": 41.0, "UG": 105.0, "FG": 169.0}


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
    for family, pair in RUN_PAIRS.items():
        maps[family] = {}
        for credit, run_name in pair.items():
            frame = pd.read_csv(UNITS[family, credit])
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
            if not np.isclose(grid.sum(), 1.0):
                raise ValueError(f"Peak-density normalization changed: {run_name}")
            maps[family][credit] = {"run_name": run_name, "xy": xy, "grid": grid,
                                    "mask": mask, "effective_area": effective_area,
                                    "valid_peaks": len(xy)}
            for row in selected.itertuples():
                peak_rows.append({"family": family, "credit": credit,
                                  "run_name": run_name, "unit_id": int(row.unit_id),
                                  "dominant_peak_x": float(row.dominant_peak_x),
                                  "dominant_peak_y": float(row.dominant_peak_y),
                                  "active_fraction": float(row.active_fraction)})
    return maps, peak_rows


def peak_panel(maps: dict[str, dict]) -> str:
    """Draw same-seed peak locations without a spatial-bin grid."""
    items = ['<g id="peak-spread-panel" aria-label="Terminal DG peak distributions: temporal DG credit versus constant-credit control">']
    for credit, left in MAP_LEFTS.items():
        items.append(f'<text x="{left+MAP_SIZE/2:g}" y="28" text-anchor="middle" '
                     f'class="peak-label" fill="#25313a">{credit}</text>')
    for family in ("NG", "UG", "FG"):
        top = MAP_TOPS[family]
        items.append(f'<text x="{MAP_LEFTS["Temporal"]:g}" y="{top-3:g}" class="peak-label" '
                     f'fill="{COLORS[family]}">{family}</text>')
        for credit, left in MAP_LEFTS.items():
            source = maps[family][credit]
            items.append(f'<g id="peak-map-{family}-{credit.lower()}">')
            items.append(f'<rect x="{left:g}" y="{top:g}" width="{MAP_SIZE:g}" '
                         f'height="{MAP_SIZE:g}" fill="white"/>')
            for peak_x, peak_y in source["xy"]:
                x = left + (peak_x - 100) * MAP_SIZE / 1900
                y = top + MAP_SIZE - (peak_y - 100) * MAP_SIZE / 1900
                items.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="1.3" '
                             f'fill="{COLORS[family]}" stroke="white" stroke-width="0.35"/>')
            items.append(f'<rect x="{left:g}" y="{top:g}" width="{MAP_SIZE:g}" '
                         f'height="{MAP_SIZE:g}" fill="none" stroke="#5c6870" '
                         'stroke-width="0.6"/>')
            items.append(f'<text x="{left+MAP_SIZE/2:g}" y="{top+MAP_SIZE+10:g}" '
                         'text-anchor="middle" class="peak-label" fill="#25313a">'
                         f'{source["effective_area"]:.2f}</text>')
            items.append('</g>')
    items.append('</g>')
    return '\n'.join(items)


def build() -> None:
    maps, peak_rows = map_inputs()
    svg = layout_base(extra_plot_shift=-4, metric_extra_shifts=(0, -9, -18, -27))
    old_name = 'sodipodi:docname="Figure2_story.svg"'
    if svg.count(old_name) != 1:
        raise ValueError("Could not identify the story canvas document name")
    svg = svg.replace(old_name,
                      'sodipodi:docname="Figure2_story_peaks.svg"', 1)
    overlay = story_overlay(terminal_rows(), selected_trajectories(),
                            behavior_differences(), peak_panel=peak_panel(maps),
                            trajectory_tile_size=38, elbow_arrows=True,
                            paired_peak_layout=True)
    svg = re.sub(r'</svg>\s*$', overlay + '\n</svg>\n', svg)
    export(svg, OUTPUT)
    with DATA.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(peak_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(peak_rows)
    MANIFEST.write_text(json.dumps({
        "schema": "cosyne27-figure2-story-peak-variant/v2",
        "parent_figure": "Figure2_story.svg",
        "physical_size_points": [540, 230],
        "trajectory_tile_points": 38,
        "peak_map_size_points": MAP_SIZE,
        "peak_map_arrangement": "three architecture rows (NG, UG, FG) by two credit columns (Temporal, Control)",
        "peak_map_data": str(DATA.relative_to(ROOT)),
        "sample_protocol": "terminal 100k-decision online snapshots; separate from panel-A 10k frozen-policy trajectory probes",
        "target_env_steps": TARGET_STEPS,
        "field_eligibility": "at least 20 active observations in at least three spatial bins",
        "active_fraction_cutoff": CUTOFF,
        "gaussian_sigma_dmlab_units": 100,
        "map_world_bounds_dmlab_units": [100, 2000, 100, 2000],
        "rendered_map_marks": "DG dominant peak coordinates only; no spatial-bin grid or density shading",
        "shared_visited_mask": str(MASK_PATH.relative_to(ROOT)),
        "shared_visited_mask_sha256": sha256(MASK_PATH),
        "derived_peak_spread_table": str(SPREAD_TABLE.relative_to(ROOT)),
        "derived_peak_spread_table_sha256": sha256(SPREAD_TABLE),
        "canonical_peak_analysis": str(ANALYSIS_SOURCE.relative_to(ROOT)),
        "canonical_peak_analysis_sha256": sha256(ANALYSIS_SOURCE),
        "controls": "same training seed and architecture, constant instead of temporal DG credit; C05 worker-hit credit is temporal in both columns",
        "examples": {family: {credit: {
            "run_name": maps[family][credit]["run_name"],
            "source_per_unit_table": str(UNITS[family, credit].relative_to(ROOT)),
            "source_per_unit_sha256": sha256(UNITS[family, credit]),
            "valid_peaks": maps[family][credit]["valid_peaks"],
            "effective_area_fraction": maps[family][credit]["effective_area"],
        } for credit in ("Temporal", "Control")} for family in ("NG", "UG", "FG")},
    }, indent=2) + '\n')


if __name__ == '__main__':
    build()
