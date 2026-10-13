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
READOUT = HERE / "data/fresh_behavior_decoder1_absolute.csv"
ALL_LAYER = HERE / "data/fresh_behavior_training_seeds.csv"
DG_INFORMATION = HERE / "figure2_candidates/08_focused_canvas/fresh_behavior_dg_bits_per_step.csv"

COLORS = {"NG": "#0072B2", "UG": "#D55E00", "FG": "#009E73"}
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
    """Join three B metrics on exactly the same 38 completed frozen models."""
    def load(path: Path, allowed_layer: str | None = None) -> dict[tuple[str, str, int], dict[str, str]]:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if allowed_layer is not None:
            rows = [row for row in rows if row["layer"] == allowed_layer and row["cohort"] == "fresh"]
        rows = [row for row in rows if row.get("completed_eval_repeats", "2") == "2"]
        keyed = {(row["condition"], row["family"], int(row["training_seed"])): row
                 for row in rows}
        if len(keyed) != len(rows) or len(keyed) != 38:
            raise ValueError(f"Expected 38 unique complete models in {path}")
        return keyed

    dg = load(DG_INFORMATION)
    ca1 = load(ALL_LAYER, "decoder_1")
    reliability = load(READOUT)
    if dg.keys() != ca1.keys() or dg.keys() != reliability.keys():
        raise ValueError("The DG, CA1, and reliability cohorts do not match")
    samples = {metric: {family: [] for family in ("NG", "UG", "FG")}
               for metric in ("dg_information", "ca1_information", "ca1_reliability")}
    names = {"C01": "NG", "C05": "UG", "C15": "FG"}
    for identity in sorted(dg):
        family = names[identity[1]]
        samples["dg_information"][family].append(float(dg[identity]["bits_per_step_difference"]))
        samples["ca1_information"][family].append(float(ca1[identity]["bits_per_step_difference"]))
        read = reliability[identity]
        samples["ca1_reliability"][family].append(
            float(read["own_reliability"]) - float(read["random_reliability"]))
    for metric, families in samples.items():
        if {family: len(values) for family, values in families.items()} != {"NG": 10, "UG": 20, "FG": 8}:
            raise ValueError(f"Unexpected sample counts for {metric}")
        if any(not -1000 < value < 1000 for values in families.values() for value in values):
            raise ValueError(f"Nonfinite or implausible {metric}")
    return samples


def behavior_card(metric: str, samples: dict[str, list[float]], x: int) -> str:
    """Draw a narrow, editable box/dot chart; layer labels are added later as SVG text."""
    specs = {
        "dg_information": (-0.02, 0.12, (0.0, 0.05, 0.1)),
        "ca1_information": (-0.1, 0.4, (-0.1, 0.15, 0.4)),
        "ca1_reliability": (0.0, 0.5, (0.0, 0.25, 0.5)),
    }
    minimum, maximum, ticks = specs[metric]
    left, right, top, bottom = x + 25, x + 66, 41, 70

    def scale(value: float) -> float:
        if not minimum <= value <= maximum:
            raise ValueError(f"{metric} value {value:g} outside plotted range")
        return bottom - (value - minimum) * (bottom - top) / (maximum - minimum)

    out = [f'<g id="behavior-{metric}" aria-label="{metric}, learned minus random actions">',
           f'<path d="M {left},{top} V {bottom} H {right}" class="behavior-axis"/>']
    for tick in ticks:
        ty = scale(tick)
        out.extend((
            f'<path d="M {left-2},{ty:.3f} H {left}" class="behavior-axis"/>',
            f'<text x="{left-3}" y="{ty+3:.3f}" text-anchor="end" '
            f'class="behavior-tick">{tick:g}</text>',
        ))
    if minimum < 0 < maximum:
        out.append(f'<path d="M {left},{scale(0):.3f} H {right}" '
                   'stroke="#879096" stroke-width="0.55"/>')
    for index, family in enumerate(("NG", "UG", "FG")):
        cx = left + 6 + index * 14
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
            f'<text x="{cx:.3f}" y="80" text-anchor="middle" class="behavior-tick">{family}</text>',
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
    left, top, bottom = 22, 36, 69
    minimum, maximum = 0.0, 0.5

    def scale(value: float) -> float:
        if not minimum <= value <= maximum:
            raise ValueError(f"Path efficiency {value} outside the printed axis")
        return bottom - (value - minimum) * (bottom - top) / (maximum - minimum)

    out = ['<g id="panel-a-path-efficiency" aria-label="Path efficiency, five seeds per design">',
           '<text x="7" y="29" class="poster-stat-title">Path efficiency</text>',
           f'<path d="M {left},{top} V {bottom} H 64" stroke="#171717" '
           'stroke-width="0.7" fill="none"/>']
    for tick in (0.0, 0.25, 0.5):
        ty = scale(tick)
        out.extend((
            f'<path d="M {left-2},{ty:g} H {left}" stroke="#171717" stroke-width="0.7"/>',
            f'<text x="{left-4}" y="{ty+3:g}" text-anchor="end" '
            f'class="poster-stat-tick">{tick:g}</text>',
        ))
    for family, cx in (("NG", 29), ("UG", 44), ("FG", 59)):
        values = [float(row["path_efficiency"]) for row in rows[family]]
        for value, jitter in zip(values, (-3.8, -1.9, 0, 1.9, 3.8)):
            out.append(f'<circle cx="{cx+jitter:g}" cy="{scale(value):g}" r="1.55" '
                       f'fill="{COLORS[family]}"/>')
        mean_y = scale(sum(values) / len(values))
        out.extend((
            f'<path d="M {cx-5},{mean_y:g} H {cx+5}" stroke="#171717" stroke-width="1"/>',
            f'<text x="{cx}" y="80" text-anchor="middle" class="poster-stat-tick">{family}</text>',
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


def behavior_schematic() -> str:
    """Match the rounded-box ablation style, using only black/gray ink."""
    return "\n".join((
        '<g id="behavior-ablation-schematic" transform="translate(-42,0)">',
        '<rect x="232" y="39" width="17" height="18" rx="3" class="black-box"/>',
        '<rect x="254" y="39" width="16" height="18" rx="3" class="black-box"/>',
        '<rect x="275" y="39" width="18" height="18" rx="3" class="black-box"/>',
        '<text x="240.5" y="51" text-anchor="middle" class="box-label">Env</text>',
        '<text x="262" y="51" text-anchor="middle" class="box-label">DG</text>',
        '<text x="284" y="51" text-anchor="middle" class="box-label">Pol.</text>',
        '<path d="M 249,48 H 254 M 270,48 H 275" class="black-arrow"/>',
        '<path d="M 251,45.5 L 254,48 L 251,50.5 M 272,45.5 L 275,48 L 272,50.5" '
        'fill="none" stroke="#26323a" stroke-width="0.9"/>',
        '<path d="M 284,38 C 282,28 240,28 240,38" class="black-arrow"/>',
        '<path d="M 237,34 L 240,38 L 243,34" fill="none" '
        'stroke="#26323a" stroke-width="0.9"/>',
        '<path d="M 258,28 L 266,36 M 258,36 L 266,28" fill="none" '
        'stroke="#171717" stroke-width="1.6"/>',
        '</g>',
    ))


def credit_schematic(y: int, crossed: str, name: str) -> str:
    """Compact the original DG/TS/Policy motif while retaining both arrows."""
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
                  behavior: dict[str, dict[str, list[float]]]) -> str:
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
           '.behavior-title{font:700 9px Arial,sans-serif;fill:#25313a}'
           '.behavior-unit{font:9px Arial,sans-serif;fill:#25313a}'
           '.behavior-tick{font:9px Arial,sans-serif;fill:#25313a}'
           '.behavior-axis{stroke:#25313a;stroke-width:0.65;fill:none}'
           '</style>',
           '<text x="6" y="12" class="story-letter">A</text>',
           '<text x="23" y="12" class="story-heading">Exploration</text>',
           '<text x="190" y="12" class="story-letter">B</text>',
           '<text x="207" y="12" class="story-heading">Behavior ablation</text>',
           '<text x="6" y="94" class="story-letter">C</text>',
           '<text x="23" y="94" class="story-heading">Push ablation</text>',
           '<text x="6" y="159" class="story-letter">D</text>',
           '<text x="23" y="159" class="story-heading">Pull ablation</text>',
           '<text x="469" y="12" class="story-letter">E</text>',
           '<text x="486" y="12" class="story-heading">Fields</text>']
    for x, family in zip((67, 107, 147), ("NG", "UG", "FG")):
        run_name = RUNS[family]
        seed = run_name.rsplit("S", 1)[-1]
        out.extend((
            f'<text x="{x+19}" y="26" text-anchor="middle" class="story-label" '
            f'fill="{COLORS[family]}">{family}</text>',
            f'<text x="{x+19}" y="36" text-anchor="middle" class="story-label">s{seed}</text>',
            f'<image x="{x}" y="40" width="38" height="38" '
            'preserveAspectRatio="none" '
            f'xlink:href="{trajectory_uri(ROOT / selected[family]["local_path"], selected[family]["sha256"])}"/>',
            f'<rect x="{x}" y="40" width="38" height="38" '
            'fill="none" stroke="#5c6870" stroke-width="0.6"/>',
        ))
    out.append(poster_dot_metric(rows))
    out.append(behavior_schematic())
    for metric, x in (("dg_information", 255), ("ca1_information", 328),
                      ("ca1_reliability", 401)):
        out.append(behavior_card(metric, behavior[metric], x))
    # Layer and unit labels are SVG text added after the quantitative marks.
    for x, layer, measure in ((255, "DG", "info · bit/step"),
                              (328, "CA1", "info · bit/step"),
                              (401, "CA1", "reliability")):
        out.extend((f'<text x="{x+36}" y="25" text-anchor="middle" class="behavior-title">{layer}</text>',
                    f'<text x="{x+36}" y="35" text-anchor="middle" class="behavior-unit">{measure}</text>'))
    out.append(credit_schematic(105, "push", "push"))
    out.append(credit_schematic(170, "pull", "pull"))
    for y, unit in zip((20, 72, 124, 176), FIELD_UNITS):
        out.extend((
            f'<image x="487" y="{y}" width="40" height="40" '
            f'preserveAspectRatio="none" xlink:href="{map_data_uri(unit)}"/>',
            f'<rect x="487" y="{y}" width="40" height="40" '
            'fill="none" stroke="#5c6870" stroke-width="0.6"/>',
            f'<text x="507" y="{y+49}" text-anchor="middle" '
            f'class="field-label">DG {unit}</text>',
        ))
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


def build() -> None:
    if not BASE_SVG.is_file():
        raise FileNotFoundError("Build the compact Figure 2 source before the story layout")
    rows = terminal_rows()
    selected = selected_trajectories()
    behavior = behavior_differences()
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
    for prefix in ("push", "pull"):
        for metric in ("dg_information", "map_overlap", "peak_coverage", "graph_edges"):
            svg = move_group(svg, f"component_{prefix}_{metric}", -68, 10)
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
        "schema": "cosyne27-figure2-story/v1",
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
            "source_all_layer_table": str(ALL_LAYER.relative_to(ROOT)),
            "source_all_layer_table_sha256": sha256(ALL_LAYER),
            "source_readout_table": str(READOUT.relative_to(ROOT)),
            "source_readout_table_sha256": sha256(READOUT),
            "sample_unit": "terminal frozen model; two paired 50,000-decision probes averaged within model",
            "sample_counts": {family: len(behavior["dg_information"][family])
                              for family in ("NG", "UG", "FG")},
            "difference": "learned-policy minus uniform-random-action value",
            "metrics": ["DG spatial information (bits/step)",
                        "readout MLP layer 1 spatial information (bits/step)",
                        "readout MLP layer 1 split-half spatial-map reliability"],
        },
        "path_efficiency_definition": "sum of per-segment end-to-end displacement divided by total path length",
        "image_crop": "right trajectory panel or selected square tile, fixed world extent, no color change",
    }, indent=2) + "\n")


if __name__ == "__main__":
    build()
