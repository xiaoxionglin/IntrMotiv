"""Build the three-row Figure 2 layout and swappable behavior metrics.

The push and pull result plots come from the author's editable Figure 2 canvas.
The behavior results are redrawn from archived paired tables at compact width.
The trajectory thumbnails show independent 10k
frozen-policy probes at 100M training frames; the summary charts use separate
100k online-spatial windows for five terminal seeds per architecture.
"""

from __future__ import annotations

import base64
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

from PIL import Image

from build_figure2_fiddle_with_fields import ATLAS, map_data_uri
from build_figure2_tetris import OUTPUT as BASE_SVG, move_group


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = (ROOT / "06_experiments/results/cosyne_2027_checkpoint_fields/"
        "graph_trajectory_summary/trajectory_by_snapshot.csv")
TRAJECTORIES = (ROOT / "06_experiments/results/cosyne_2027_checkpoint_fields/"
                "selected_figures/trajectories")
SELECTED = TRAJECTORIES.parent / "selected_figure_sources.csv"
STORY = HERE / "Figure2_story.svg"
GALLERY = HERE / "Figure2_behavior_candidate_stats.svg"
MANIFEST = HERE / "Figure2_story_manifest.json"
DG_INFORMATION = HERE / "figure2_candidates/08_focused_canvas/fresh_behavior_dg_bits_per_step.csv"
DG_RELIABILITY = HERE / "data/fresh_behavior_dg_reliability.csv"

COLORS = {"NG": "#0072B2", "UG": "#D55E00", "FG": "#009E73"}
# Match the plotting rectangles in the existing push and pull SVG components.
RESULT_AXIS_WIDTH = 49.68
RESULT_AXIS_HEIGHT = 47.52
TRAJECTORY_TILE_SIZE = 46
FIELD_UNITS = (4, 7, 12, 15)
RUNS = {
    "NG": "COS27_C01_TEMPORAL_S99",
    "UG": "COS27_C05_TEMPORAL_TEMPORAL_S123",
    "FG": "COS27_C15_TEMPORAL_S17",
}
UG_TILE = HERE / "figure2_candidates/UG_s123_100m_trajectory_tile.png"
UG_FULL = HERE / "figure2_candidates/UG_s123_100m_trajectory.png"
UG_TILE_SHA256 = "1c1af9ec4ee2fe2e8ecc2446caf7c0fadca901b863c38bc57adfa58cd254786e"
UG_FULL_SHA256 = "6dfa9ca75a5a74e5c3b6555b91ea6b1d1445d57e71c2a5db71a76ce604f83820"
UG_POSE_SHA256 = "27c283ba992897e36a989acd56d4c5373b8d8353d2af7b47276f19d317b00a85"
METRICS = {
    "path_efficiency": ("Path efficiency", 0.0, 0.5, (0.0, 0.25, 0.5)),
    "visited_cell_fraction": ("Visited bins (fraction)", 0.8, 0.9, (0.8, 0.85, 0.9)),
    "stationary_step_fraction": ("Stationary steps (fraction)", 0.0, 0.3, (0.0, 0.15, 0.3)),
    "mean_physical_step_distance": ("Step distance (DMLab units)", 0.0, 45.0, (0.0, 20.0, 40.0)),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal_rows() -> dict[str, list[dict[str, str]]]:
    """Select the same temporal design in each family, five seeds each."""
    cohorts = {"C01": "NG", "C05": "UG", "C15": "FG"}
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in cohorts.values()}
    with DATA.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["target_env_steps"] != "100000000":
                continue
            if row["encoder_credit"] != "TEMPORAL":
                continue
            if row["family"] == "C05" and row["worker_bonus"] != "TEMPORAL":
                continue
            if row["family"] not in cohorts:
                continue
            rows[cohorts[row["family"]]].append(row)
    expected = {8, 17, 41, 99, 123}
    for family, values in rows.items():
        seeds = {int(row["seed"]) for row in values}
        if len(values) != 5 or seeds != expected:
            raise ValueError(f"Unexpected terminal cohort for {family}: {seeds}")
        values.sort(key=lambda row: int(row["seed"]))
        for row in values:
            for metric in METRICS:
                value = float(row[metric])
                if not (float("-inf") < value < float("inf")):
                    raise ValueError(f"Nonfinite {metric}: {row['run_name']}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def label(value: float) -> str:
    return f"{value:g}"


def behavior_differences() -> dict[str, dict[str, list[float]]]:
    """Join DG information and reliability on the same 38 frozen models."""
    def load(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if row.get("cohort", "fresh") == "fresh"
                and row.get("layer", "dg") == "dg" and row["completed_eval_repeats"] == "2"]
        keyed = {(row["condition"], row["family"], int(row["training_seed"])): row
                 for row in rows}
        if len(keyed) != len(rows) or len(keyed) != 38:
            raise ValueError(f"Expected 38 unique complete models in {path}")
        return keyed

    dg = load(DG_INFORMATION)
    reliability = load(DG_RELIABILITY)
    if dg.keys() != reliability.keys():
        raise ValueError("DG information and reliability cohorts do not match")
    samples = {metric: {family: [] for family in ("NG", "UG", "FG")}
               for metric in ("dg_information", "dg_reliability")}
    names = {"C01": "NG", "C05": "UG", "C15": "FG"}
    for identity in sorted(dg):
        family = names[identity[1]]
        samples["dg_information"][family].append(float(dg[identity]["bits_per_step_difference"]))
        samples["dg_reliability"][family].append(
            float(reliability[identity]["common_split_half_difference"]))
    for metric, families in samples.items():
        if {family: len(values) for family, values in families.items()} != {"NG": 10, "UG": 20, "FG": 8}:
            raise ValueError(f"Unexpected sample counts for {metric}")
        if any(not -1000 < value < 1000 for values in families.values() for value in values):
            raise ValueError(f"Nonfinite or implausible {metric}")
    return samples


def behavior_card(metric: str, samples: dict[str, list[float]], x: int) -> str:
    """Draw an editable box/dot chart near the width of C and D's originals."""
    specs = {
        "dg_information": (-0.02, 0.12, (0.0, 0.05, 0.1)),
        "dg_reliability": (-0.05, 0.55, (0.0, 0.25, 0.5)),
    }
    minimum, maximum, ticks = specs[metric]
    # Reserve space for the vertical metric name and its numeric tick labels.
    left, top = x + 35, 30
    right, bottom = left + RESULT_AXIS_WIDTH, top + RESULT_AXIS_HEIGHT

    def scale(value: float) -> float:
        if not minimum <= value <= maximum:
            raise ValueError(f"{metric} value {value:g} outside plotted range")
        return bottom - (value - minimum) * (bottom - top) / (maximum - minimum)

    out = [f'<g id="behavior-{metric}" aria-label="{metric}, learned minus random actions">',
           f'<path d="M {left},{top} V {bottom} H {right}" class="behavior-axis"/>']
    for tick_index, tick in enumerate(ticks):
        ty = scale(tick)
        out.extend((
            f'<path d="M {left-2},{ty:.3f} H {left}" class="behavior-axis"/>',
            f'<text id="behavior-{metric}-ytick-{tick_index}" '
            f'x="{left-3}" y="{ty+3:.3f}" text-anchor="end" '
            f'class="behavior-tick">{tick:g}</text>',
        ))
    if minimum < 0 < maximum:
        out.append(f'<path d="M {left},{scale(0):.3f} H {right}" '
                   'stroke="#879096" stroke-width="0.55"/>')
    for index, family in enumerate(("NG", "UG", "FG")):
        cx = left + RESULT_AXIS_WIDTH * (index + 0.5) / 3
        values = samples[family]
        color = COLORS[family]
        for dot_index, value in enumerate(values):
            jitter = ((dot_index * 7) % 9 - 4) * 0.42
            out.append(f'<circle cx="{cx+jitter:.3f}" cy="{scale(value):.3f}" '
                       f'r="1.15" fill="{color}" fill-opacity="0.7"/>')
        low, q1, median, q3, high = (min(values), percentile(values, .25),
                                      percentile(values, .5), percentile(values, .75), max(values))
        out.extend((
            f'<path d="M {cx:.3f},{scale(low):.3f} V {scale(high):.3f} '
            f'M {cx-2.5:.3f},{scale(low):.3f} H {cx+2.5:.3f} '
            f'M {cx-2.5:.3f},{scale(high):.3f} H {cx+2.5:.3f}" '
            f'stroke="{color}" stroke-width="0.8" fill="none"/>',
            f'<rect x="{cx-3:.3f}" y="{scale(q3):.3f}" width="6" '
            f'height="{scale(q1)-scale(q3):.3f}" fill="white" stroke="{color}" stroke-width="0.9"/>',
            f'<path d="M {cx-3:.3f},{scale(median):.3f} H {cx+3:.3f}" stroke="{color}" stroke-width="0.9"/>',
            f'<text x="{cx:.3f}" y="{bottom+9:.3f}" text-anchor="middle" '
            f'class="behavior-tick">{family}</text>',
        ))
    out.append('</g>')
    return "\n".join(out)


def metric_card(metric: str, rows: dict[str, list[dict[str, str]]],
                x: float, y: float, width: float = 300, height: float = 90) -> str:
    """Render one editable vector box/dot chart at its final physical size."""
    title, minimum, maximum, ticks = METRICS[metric]
    left, right = x + 35, x + width - 9
    top, bottom = y + 23, y + height - 18

    def scale(value: float) -> float:
        if not minimum <= value <= maximum:
            raise ValueError(f"{metric} value {value} outside the printed axis")
        return bottom - (value - minimum) * (bottom - top) / (maximum - minimum)

    out = [f'<g id="metric-{metric}" aria-label="{title}, five seeds per architecture">',
           f'<text x="{x+7:g}" y="{y+12:g}" class="metric-title">{title}</text>',
           f'<text x="{x+width-9:g}" y="{y+12:g}" text-anchor="end" '
           'class="metric-note">n = 5 per design</text>',
           f'<path d="M {left:g},{top:g} V {bottom:g} H {right:g}" '
           'stroke="#26323a" stroke-width="0.65" fill="none"/>']
    for tick in ticks:
        ty = scale(tick)
        out.extend((
            f'<path d="M {left-3:g},{ty:g} H {left:g}" stroke="#26323a" stroke-width="0.65"/>',
            f'<text x="{left-5:g}" y="{ty+3:g}" text-anchor="end" class="metric-tick">{label(tick)}</text>',
        ))
    plot_width = right - left
    for index, family in enumerate(("NG", "UG", "FG")):
        cx = left + plot_width * (index + 0.5) / 3
        values = [float(row[metric]) for row in rows[family]]
        low, q1, middle, q3, high = (min(values), percentile(values, 0.25),
                                      percentile(values, 0.5), percentile(values, 0.75),
                                      max(values))
        color = COLORS[family]
        out.extend((
            f'<path d="M {cx:g},{scale(low):g} V {scale(high):g} '
            f'M {cx-5:g},{scale(low):g} H {cx+5:g} '
            f'M {cx-5:g},{scale(high):g} H {cx+5:g}" '
            f'stroke="{color}" stroke-width="1" fill="none"/>',
            f'<rect x="{cx-7:g}" y="{scale(q3):g}" width="14" '
            f'height="{scale(q1)-scale(q3):g}" fill="white" '
            f'stroke="{color}" stroke-width="1.3"/>',
            f'<path d="M {cx-7:g},{scale(middle):g} H {cx+7:g}" '
            f'stroke="{color}" stroke-width="1.3"/>',
            f'<text x="{cx:g}" y="{bottom+12:g}" text-anchor="middle" '
            f'class="metric-category">{family}</text>',
        ))
        for value, jitter in zip(values, (-5, -2.5, 0, 2.5, 5)):
            out.append(f'<circle cx="{cx+jitter:g}" cy="{scale(value):g}" r="1.7" '
                       f'fill="{color}" fill-opacity="0.72"/>')
    out.append('</g>')
    return "\n".join(out)


def poster_dot_metric(rows: dict[str, list[dict[str, str]]]) -> str:
    """Draw A's compact seed dots and black mean bars as on the Bernstein poster."""
    left, top, bottom = 35, 36, 69
    minimum, maximum = 0.0, 0.5

    def scale(value: float) -> float:
        if not minimum <= value <= maximum:
            raise ValueError(f"Path efficiency {value} outside the printed axis")
        return bottom - (value - minimum) * (bottom - top) / (maximum - minimum)

    out = ['<g id="panel-a-path-efficiency" aria-label="Path efficiency, five seeds per design">',
           '<text id="panel-a-path-efficiency-ylabel" x="9" y="52.5" '
           'text-anchor="middle" transform="rotate(-90 9 52.5)" '
           'class="poster-stat-title">Path efficiency</text>',
           f'<path d="M {left},{top} V {bottom} H 71" stroke="#171717" '
           'stroke-width="0.7" fill="none"/>']
    for tick_index, tick in enumerate((0.0, 0.25, 0.5)):
        ty = scale(tick)
        out.extend((
            f'<path d="M {left-2},{ty:g} H {left}" stroke="#171717" stroke-width="0.7"/>',
            f'<text id="panel-a-path-efficiency-ytick-{tick_index}" '
            f'x="{left-4}" y="{ty+3:g}" text-anchor="end" '
            f'class="poster-stat-tick">{tick:g}</text>',
        ))
    for family, cx in (("NG", 39), ("UG", 53), ("FG", 67)):
        values = [float(row["path_efficiency"]) for row in rows[family]]
        for value, jitter in zip(values, (-3.8, -1.9, 0, 1.9, 3.8)):
            out.append(f'<circle cx="{cx+jitter:g}" cy="{scale(value):g}" r="1.55" '
                       f'fill="{COLORS[family]}"/>')
        mean_y = scale(sum(values) / len(values))
        out.extend((
            f'<path d="M {cx-4},{mean_y:g} H {cx+4}" stroke="#171717" stroke-width="1"/>',
            f'<text id="panel-a-path-efficiency-category-{family}" '
            f'x="{cx}" y="80" text-anchor="middle" '
            f'class="poster-stat-tick">{family}</text>',
        ))
    out.append('</g>')
    return "\n".join(out)


def trajectory_uri(path: Path, expected_sha256: str) -> str:
    if sha256(path) != expected_sha256:
        raise ValueError(f"Trajectory source hash changed: {path}")
    with Image.open(path) as source:
        if source.size == (466, 466):
            tile = source.convert("RGB")
        elif source.width in (1425, 1429) and source.height in (688, 690):
            # The archived source's right panel is a square trajectory map.
            crop = (902, 147, 1368, 613) if source.width == 1425 else (908, 151, 1368, 611)
            tile = source.crop(crop).convert("RGB")
        else:
            raise ValueError(f"Unexpected trajectory image size: {source.size}")
    buffer = BytesIO()
    tile.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def selected_trajectories() -> dict[str, dict[str, str]]:
    records = {}
    with SELECTED.open(newline="") as stream:
        by_run = {row["run_name"]: row for row in csv.DictReader(stream)
                  if row["kind"] == "trajectory"}
    for family, run_name in RUNS.items():
        if family == "UG":
            if not UG_FULL.is_file() or not UG_TILE.is_file():
                raise FileNotFoundError("Selected UG seed-123 canonical render is missing")
            if sha256(UG_TILE) != UG_TILE_SHA256 or sha256(UG_FULL) != UG_FULL_SHA256:
                raise ValueError("Selected UG seed-123 render changed; verify its pose source")
            records[family] = {
                "run_name": run_name,
                "sha256": UG_TILE_SHA256,
                "local_path": str(UG_TILE.relative_to(ROOT)),
                "full_plot_path": str(UG_FULL.relative_to(ROOT)),
                "full_plot_sha256": UG_FULL_SHA256,
                "source_pose_sha256": UG_POSE_SHA256,
                "source_pose_path": (
                    "/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/"
                    "train_dir/analysis/cosyne27_checkpoint_20261011/c05_pushpull_s3/raw/"
                    "COS27_C05_TEMPORAL_TEMPORAL_S123__00_COS27_C05_TEMPORAL_TEMPORAL_S123__"
                    "COS27_C05_TEMPORAL_TEMPORAL__s123__f100007936/pose.csv"
                ),
                "renderer": "hpc_runs.intrmotiv_study.spatial.render_occupancy_trajectory",
            }
            continue
        record = by_run.get(run_name)
        if record is None:
            raise ValueError(f"No archived selected trajectory for {run_name}")
        path = TRAJECTORIES / f"{run_name}.png"
        records[family] = {"run_name": run_name, "sha256": record["sha256"],
                           "local_path": str(path.relative_to(ROOT))}
    return records


def hide_group(svg: str, group_id: str) -> str:
    svg, count = re.subn(rf'(<g\s+id="{re.escape(group_id)}")',
                         r'\1 display="none"', svg, count=1)
    if count != 1:
        raise ValueError(f"Could not hide group {group_id}")
    return svg


def behavior_schematic(compact: bool = False) -> str:
    """Match the rounded-box ablation style, using only black/gray ink."""
    if compact:
        # Save ten points beside the information y-label without reducing type.
        env_left, env_width = 232, 16
        dg_left, dg_width = 250, 14
        policy_left, policy_width = 266, 17
    else:
        env_left, env_width = 232, 17
        dg_left, dg_width = 254, 16
        policy_left, policy_width = 275, 18
    env_center = env_left + env_width / 2
    dg_center = dg_left + dg_width / 2
    policy_center = policy_left + policy_width / 2
    arc_left = env_center if compact else 240
    return "\n".join((
        '<g id="behavior-ablation-schematic">',
        f'<rect x="{env_left}" y="39" width="{env_width}" height="18" rx="3" class="black-box"/>',
        f'<rect x="{dg_left}" y="39" width="{dg_width}" height="18" rx="3" class="black-box"/>',
        f'<rect x="{policy_left}" y="39" width="{policy_width}" height="18" rx="3" class="black-box"/>',
        f'<text x="{env_center:g}" y="51" text-anchor="middle" class="box-label">Env</text>',
        f'<text x="{dg_center:g}" y="51" text-anchor="middle" class="box-label">DG</text>',
        f'<text x="{policy_center:g}" y="51" text-anchor="middle" class="box-label">Pol.</text>',
        f'<path d="M {env_left+env_width},48 H {dg_left} '
        f'M {dg_left+dg_width},48 H {policy_left}" class="black-arrow"/>',
        f'<path d="M {dg_left-2},45.5 L {dg_left},48 L {dg_left-2},50.5 '
        f'M {policy_left-2},45.5 L {policy_left},48 L {policy_left-2},50.5" '
        'fill="none" stroke="#26323a" stroke-width="0.9"/>',
        f'<path d="M {policy_center:g},38 C {policy_center-2:g},28 '
        f'{arc_left:g},28 {arc_left:g},38" class="black-arrow"/>',
        f'<path d="M {arc_left-3:g},34 L {arc_left:g},38 L {arc_left+3:g},34" fill="none" '
        'stroke="#26323a" stroke-width="0.9"/>',
        f'<path d="M {dg_center-4:g},28 L {dg_center+4:g},36 '
        f'M {dg_center-4:g},36 L {dg_center+4:g},28" fill="none" '
        'stroke="#171717" stroke-width="1.6"/>',
        '</g>',
    ))


def credit_schematic(y: int, crossed: str, name: str,
                     elbow_arrows: bool = False) -> str:
    """Compact the DG/TS/Policy motif while retaining both directed arrows."""
    if elbow_arrows:
        # The lower TS box leaves a narrow column for the result plots. The
        # crossed connection stays visible on the vertical leg of its arrow.
        cross_x = 25 if crossed == "push" else 51
        return "\n".join((
            f'<g id="{name}-schematic">',
            f'<rect x="4" y="{y}" width="19" height="21" rx="4" '
            'fill="#c58c8c" stroke="#933333" stroke-width="0.9"/>',
            f'<rect x="29" y="{y+12}" width="19" height="21" rx="4" '
            'fill="#edf4f2" stroke="#257873" stroke-width="0.9"/>',
            f'<rect x="53" y="{y}" width="22" height="21" rx="4" '
            'fill="#abc9e0" stroke="#16639a" stroke-width="0.9"/>',
            f'<text x="13.5" y="{y+14}" text-anchor="middle" class="mini-label">DG</text>',
            f'<text x="38.5" y="{y+26}" text-anchor="middle" class="mini-label">TS</text>',
            f'<text x="64" y="{y+14}" text-anchor="middle" class="mini-label">Pol.</text>',
            f'<path d="M 29,{y+22.5} H 25 V {y+10.5} H 23" '
            'fill="none" stroke="#933333" stroke-width="1.2"/>',
            f'<path d="M 26,{y+8} L 23,{y+10.5} L 26,{y+13}" '
            'fill="none" stroke="#933333" stroke-width="1"/>',
            f'<path d="M 48,{y+22.5} H 51 V {y+10.5} H 53" '
            'fill="none" stroke="#16639a" stroke-width="1.2"/>',
            f'<path d="M 50,{y+8} L 53,{y+10.5} L 50,{y+13}" '
            'fill="none" stroke="#16639a" stroke-width="1"/>',
            f'<path d="M {cross_x-4},{y+15} L {cross_x+4},{y+25} '
            f'M {cross_x-4},{y+25} L {cross_x+4},{y+15}" '
            'stroke="#171717" stroke-width="1.7" fill="none"/>',
            '</g>',
        ))
    cross_x = 27 if crossed == "push" else 56
    return "\n".join((
        f'<g id="{name}-schematic">',
        f'<rect x="4" y="{y}" width="19" height="21" rx="4" '
        'fill="#c58c8c" stroke="#933333" stroke-width="0.9"/>',
        f'<rect x="32" y="{y}" width="19" height="21" rx="4" '
        'fill="#edf4f2" stroke="#257873" stroke-width="0.9"/>',
        f'<rect x="60" y="{y}" width="22" height="21" rx="4" '
        'fill="#abc9e0" stroke="#16639a" stroke-width="0.9"/>',
        f'<text x="13.5" y="{y+14}" text-anchor="middle" class="mini-label">DG</text>',
        f'<text x="41.5" y="{y+14}" text-anchor="middle" class="mini-label">TS</text>',
        f'<text x="71" y="{y+14}" text-anchor="middle" class="mini-label">Pol.</text>',
        f'<path d="M 32,{y+10.5} H 23" stroke="#933333" stroke-width="1.2"/>',
        f'<path d="M 27,{y+8} L 23,{y+10.5} L 27,{y+13}" '
        'fill="none" stroke="#933333" stroke-width="1"/>',
        f'<path d="M 51,{y+10.5} H 60" stroke="#16639a" stroke-width="1.2"/>',
        f'<path d="M 56,{y+8} L 60,{y+10.5} L 56,{y+13}" '
        'fill="none" stroke="#16639a" stroke-width="1"/>',
        f'<path d="M {cross_x-4},{y+5} L {cross_x+4},{y+16} '
        f'M {cross_x-4},{y+16} L {cross_x+4},{y+5}" '
        'stroke="#171717" stroke-width="1.7" fill="none"/>',
        '</g>',
    ))


def story_overlay(rows: dict[str, list[dict[str, str]]],
                  selected: dict[str, dict[str, str]],
                  behavior: dict[str, dict[str, list[float]]],
                  *, peak_panel: str | None = None,
                  trajectory_tile_size: int = TRAJECTORY_TILE_SIZE,
                  elbow_arrows: bool = False,
                  paired_peak_layout: bool = False,
                  show_panel_e: bool = True,
                  behavior_layout_shift: float = 0,
                  compact_behavior_schematic: bool = False,
                  reliability_shift: float = 0,
                  extra_overlay: str = '') -> str:
    peak_style = ('.peak-heading{font:700 9px Arial,sans-serif;fill:#25313a}'
                  '.peak-label{font:700 9px Arial,sans-serif}'
                  if peak_panel is not None else '')
    out = ['<g id="story-overlay">',
           '<style>.story-heading{font:700 12px Arial,sans-serif;fill:#25313a}'
           '.story-letter{font:700 13px Arial,sans-serif;fill:#25313a}'
           '.story-label{font:10px Arial,sans-serif;fill:#25313a}'
           '.field-label{font:9px Arial,sans-serif;fill:#25313a}'
           '.mini-label{font:9px Arial,sans-serif;fill:#25313a}'
           '.box-label{font:9px Arial,sans-serif;fill:#171717}'
           '.black-box{fill:#f4f4f4;stroke:#171717;stroke-width:1}'
           '.black-arrow{fill:none;stroke:#26323a;stroke-width:1}'
           '.poster-stat-title{font:700 9px Arial,sans-serif;fill:#25313a}'
           '.poster-stat-tick{font:9px Arial,sans-serif;fill:#25313a}'
           '.metric-title{font:700 10px Arial,sans-serif;fill:#25313a}'
           '.metric-note,.metric-tick,.metric-category{font:9px Arial,sans-serif;fill:#25313a}'
           '.behavior-y-label{font:10px Arial,sans-serif;fill:#25313a}'
           '.behavior-tick{font:9px Arial,sans-serif;fill:#25313a}'
           '.behavior-axis{stroke:#25313a;stroke-width:0.65;fill:none}'
           f'{peak_style}</style>',
           '<text x="6" y="12" class="story-letter">A</text>',
           '<text x="23" y="12" class="story-heading">Exploration</text>',
           '<text x="6" y="94" class="story-letter">C</text>',
           '<text x="23" y="94" class="story-heading">Push ablation</text>',
           '<text x="6" y="159" class="story-letter">D</text>',
           '<text x="23" y="159" class="story-heading">Pull ablation</text>']
    if show_panel_e:
        out.extend((
            f'<text x="{440 if paired_peak_layout else 469}" y="12" class="story-letter">E</text>',
            (f'<text id="peak-spread-heading" x="{454 if paired_peak_layout else 483}" y="12" '
             'class="peak-heading">Peak spread</text>'
             if peak_panel is not None else
             '<text x="486" y="12" class="story-heading">Fields</text>'),
        ))
    centers = (95, 136, 177) if paired_peak_layout else (97, 150, 203)
    for center_x, family in zip(centers, ("NG", "UG", "FG")):
        x = center_x - trajectory_tile_size / 2
        out.extend((
            f'<text x="{center_x}" y="26" text-anchor="middle" class="story-label" '
            f'fill="{COLORS[family]}">{family}</text>',
            f'<image x="{x:g}" y="32" width="{trajectory_tile_size}" '
            f'height="{trajectory_tile_size}" '
            'preserveAspectRatio="none" '
            f'xlink:href="{trajectory_uri(ROOT / selected[family]["local_path"], selected[family]["sha256"])}"/>',
            f'<rect x="{x:g}" y="32" width="{trajectory_tile_size}" '
            f'height="{trajectory_tile_size}" '
            'fill="none" stroke="#5c6870" stroke-width="0.6"/>',
        ))
    out.append(poster_dot_metric(rows))
    behavior_shift = (-34 if paired_peak_layout else 0) + behavior_layout_shift
    out.extend((
        f'<text x="{231 + behavior_shift}" y="12" class="story-letter">B</text>',
        f'<text x="{248 + behavior_shift}" y="12" class="story-heading">Behavior ablation</text>',
    ))
    if behavior_shift:
        out.append(f'<g transform="translate({behavior_shift},0)">'
                   f'{behavior_schematic(compact_behavior_schematic)}</g>')
    else:
        out.append(behavior_schematic(compact_behavior_schematic))
    for metric, x in (("dg_information", 296), ("dg_reliability", 386)):
        metric_shift = reliability_shift if metric == 'dg_reliability' else 0
        out.append(behavior_card(metric, behavior[metric], x + behavior_shift + metric_shift))
    # Keep B's metric names on the y axes, as in the push and pull rows.
    for metric, x, measure in (("dg_information", 296, "Δ info (bits/step)"),
                               ("dg_reliability", 386, "Δ reliability")):
        x += behavior_shift + (reliability_shift if metric == 'dg_reliability' else 0)
        y_center = 30 + RESULT_AXIS_HEIGHT / 2
        out.append(f'<text id="behavior-{metric}-ylabel" '
                   f'x="{x+6}" y="{y_center:.3f}" text-anchor="middle" '
                   f'transform="rotate(-90 {x+6} {y_center:.3f})" '
                   f'class="behavior-y-label">{measure}</text>')
    out.append(credit_schematic(105, "push", "push", elbow_arrows))
    out.append(credit_schematic(170, "pull", "pull", elbow_arrows))
    if show_panel_e and peak_panel is None:
        for y, unit in zip((20, 72, 124, 176), FIELD_UNITS):
            out.extend((
                f'<image x="487" y="{y}" width="40" height="40" '
                f'preserveAspectRatio="none" xlink:href="{map_data_uri(unit)}"/>',
                f'<rect x="487" y="{y}" width="40" height="40" '
                'fill="none" stroke="#5c6870" stroke-width="0.6"/>',
                f'<text x="507" y="{y+49}" text-anchor="middle" '
                f'class="field-label">DG {unit}</text>',
            ))
    elif show_panel_e:
        out.append(peak_panel)
    if extra_overlay:
        out.append(extra_overlay)
    out.append('</g>')
    return "\n".join(out)


def export(svg: str, path: Path) -> None:
    # Inkscape embeds fresh PDF metadata on every export. Preserve stable
    # candidate PDFs when their editable SVG and rendered files are unchanged.
    if (path.is_file() and path.read_text() == svg
            and path.with_suffix(".pdf").is_file()
            and path.with_suffix(".png").is_file()):
        return
    path.write_text(svg)
    for extension, width in (("pdf", None), ("png", "1800")):
        command = ["inkscape", str(path),
                   f"--export-filename={path.with_suffix('.' + extension)}"]
        if width:
            command.append(f"--export-width={width}")
        subprocess.run(command, capture_output=True, text=True, check=True)


def standalone(body: str, width: int, height: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">'
            '<style>.metric-title{font:700 10px Arial,sans-serif;fill:#25313a}'
            '.metric-note,.metric-tick,.metric-category{font:9px Arial,sans-serif;fill:#25313a}'
            '</style>' + body + '</svg>\n')


def layout_base(extra_plot_shift: float = 0,
                metric_extra_shifts: tuple[float, float, float, float] = (0, 0, 0, 0)) -> str:
    """Prepare the editable source canvas shared by the two story variants."""
    if not BASE_SVG.is_file():
        raise FileNotFoundError("Build the compact Figure 2 source before the story layout")
    svg = BASE_SVG.read_text()
    for old, new in (('height="210pt"', 'height="230pt"'),
                     ('viewBox="0 0 540 210"', 'viewBox="0 0 540 230"'),
                     ('id="pdf-origin-fix" transform="translate(0,12)"',
                      'id="pdf-origin-fix" transform="translate(0,20)"'),
                     ('sodipodi:docname="Figure2_tetris.svg"',
                      'sodipodi:docname="Figure2_story.svg"')):
        if svg.count(old) != 1:
            raise ValueError(f"Base layout changed: expected {old}")
        svg = svg.replace(old, new, 1)
    for group in ("g699", "g701", "g702", "compact-map-examples",
                  "component_behavior_dg_information", "component_behavior_readout_reliability"):
        svg = hide_group(svg, group)
    # These are display names; the stored metric remains pairwise map cosine.
    # The source SVG has path-converted per-letter text, so replace only the
    # two y-label groups rather than altering statistical plot geometry.
    for prefix, label_index in (("push", 7), ("pull", 6)):
        pattern = (rf'<g\s+id="{prefix}_map_overlap_text_{label_index}"><g\s+'
                   r'transform="(rotate\([^"]+\))"[^>]*><text[^>]*>.*?</text></g></g>')
        def new_label(match: re.Match[str]) -> str:
            return (f'<g id="{prefix}_map_overlap_text_{label_index}">'
                    f'<g transform="{match.group(1)}"><text x="-6" y="-0.703125" '
                    'style="font-size:10px;font-family:Arial">Δ overlap</text></g></g>')
        svg, count = re.subn(pattern, new_label, svg, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError(f"Could not rename {prefix} overlap label")
    for prefix in ("push", "pull"):
        for metric, metric_shift in zip(
                ("dg_information", "map_overlap", "peak_coverage", "graph_edges"),
                metric_extra_shifts):
            svg = move_group(svg, f"component_{prefix}_{metric}",
                             -68 + extra_plot_shift + metric_shift, 10)
    return svg


def build() -> None:
    rows = terminal_rows()
    selected = selected_trajectories()
    behavior = behavior_differences()
    svg = layout_base()
    svg = re.sub(r"</svg>\s*$", story_overlay(rows, selected, behavior) + "\n</svg>\n", svg)
    export(svg, STORY)

    gallery_cards = []
    for index, metric in enumerate(METRICS):
        x, y = 270 * (index % 2), 95 * (index // 2)
        gallery_cards.append(metric_card(metric, rows, x, y, 265, 91))
        export(standalone(metric_card(metric, rows, 0, 0), 300, 90),
               HERE / f"Figure2_behavior_{metric}.svg")
    export(standalone("\n".join(gallery_cards), 540, 190),
           GALLERY)
    MANIFEST.write_text(json.dumps({
        "schema": "cosyne27-figure2-story/v2",
        "source_trajectory_table": str(DATA.relative_to(ROOT)),
        "source_trajectory_table_sha256": sha256(DATA),
        "sample_unit": "training seed; five per architecture",
        "metric_window": "latest 100,000 online own-policy observations at 100M training frames",
        "trajectory_example_window": "separate 10,001-observation frozen own-policy probe at the 100M checkpoint",
        "cohorts": {family: [row["run_name"] for row in values]
                    for family, values in rows.items()},
        "example_trajectories": selected,
        "field_example_atlas": str(ATLAS.relative_to(ROOT)),
        "field_example_atlas_sha256": sha256(ATLAS),
        "field_example_units": list(FIELD_UNITS),
        "frozen_behavior_comparison": {
            "source_dg_table": str(DG_INFORMATION.relative_to(ROOT)),
            "source_dg_table_sha256": sha256(DG_INFORMATION),
            "source_dg_reliability_table": str(DG_RELIABILITY.relative_to(ROOT)),
            "source_dg_reliability_table_sha256": sha256(DG_RELIABILITY),
            "reliability_archive_commit": "6412596",
            "reliability_archive_path": "06_experiments/results/cosyne_2027_behavior_secondary/all_layer_training_seeds.csv",
            "reliability_archive_sha256": "1a601d3501de92e4187ba53c3bca00ca587441808ac4e06626b0e7ae48755c29",
            "sample_unit": "terminal frozen model; two paired 50,000-decision probes averaged within model",
            "sample_counts": {family: len(behavior["dg_information"][family])
                              for family in ("NG", "UG", "FG")},
            "difference": "learned-policy minus uniform-random-action value",
            "metrics": ["DG spatial information (bits/step)",
                        "DG common-support split-half spatial-map reliability"],
        },
        "path_efficiency_definition": "sum of per-segment end-to-end displacement divided by total path length",
        "image_crop": "right trajectory panel or selected square tile, fixed world extent, no color change",
    }, indent=2) + "\n")


if __name__ == "__main__":
    build()
