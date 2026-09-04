#!/usr/bin/env python3
"""Render readable interim diagnostics for the DPR StudySpec batch.

Data source:
  results/directional_predictive_recruitment_interim_20260904/per_run.csv

The script uses only explicit StudySpec columns. It never recovers factors by
parsing run names. SVG is the vector master; Inkscape renders matching PNGs.
"""

from __future__ import annotations

import html
from pathlib import Path
import subprocess

import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results/directional_predictive_recruitment_interim_20260904/per_run.csv"
OUTPUT_DIR = HERE / "assets/directional_predictive_recruitment_interim_20260904"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
WIDTH, HEIGHT = 2200, 1220
COLORS = {
    "C05": "#0072B2", "C13": "#D55E00", "C15": "#009E73",
    "monitor": "#6B7280", "directional": "#7E57C2", "predictive": "#CC79A7",
}
REC_LABEL = {"monitor": "MON", "directional": "DIR", "predictive": "PRED"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self) -> None:
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text { font-family: "DejaVu Sans", sans-serif; fill: #202124; } '
            '.axis { stroke: #3C4043; stroke-width: 3; } .grid { stroke: #DADCE0; stroke-width: 2; } '
            '.pair { stroke: #AAB0B6; stroke-width: 3; opacity: 0.48; }</style>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(self, x: float, y: float, value: object, size: int = 42, anchor: str = "start",
             weight: int = 400, rotate: float | None = None, fill: str | None = None) -> None:
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        color = f' fill="{fill}"' if fill else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}"{transform}{color}>{esc(value)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str = "axis", **attrs: object) -> None:
        extra = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>')

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str, sw: float = 3,
               opacity: float = 1.0) -> None:
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str, sw: float = 3,
             opacity: float = 1.0) -> None:
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def scale(value: float, lo: float, hi: float, start: float, end: float) -> float:
    return start + (value - lo) / (hi - lo) * (end - start)


def axes(svg: SVG, x0: float, xticks: list[tuple[float, str]], yticks: list[tuple[float, str]],
         xlim: tuple[float, float], ylim: tuple[float, float], xlabel: str, ylabel: str) -> tuple:
    left, right, top, bottom = x0 + 145, x0 + 1040, 275, 895
    for value, label in yticks:
        y = scale(value, ylim[0], ylim[1], bottom, top)
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 20, y + 15, label, size=40, anchor="end")
    for value, label in xticks:
        x = scale(value, xlim[0], xlim[1], left, right)
        svg.line(x, bottom, x, bottom + 12)
        svg.text(x, bottom + 58, label, size=38, anchor="middle")
    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text((left + right) / 2, 1035, xlabel, size=44, anchor="middle")
    svg.text(x0 + 42, (top + bottom) / 2, ylabel, size=44, anchor="middle", rotate=-90)
    return left, right, top, bottom


def marker(svg: SVG, x: float, y: float, color: str, goal: str, size: float = 12) -> None:
    if goal == "legacy":
        svg.circle(x, y, size, "white", color, sw=4, opacity=0.9)
    else:
        svg.rect(x - size, y - size, 2 * size, 2 * size, color, color, sw=2, opacity=0.86)


def title(svg: SVG, main: str) -> None:
    svg.text(WIDTH / 2, 62, main, size=58, anchor="middle", weight=700)
    svg.text(WIDTH / 2, 118, "54 runs; aligned means over 62.5–67.5M environment steps",
             size=40, anchor="middle", fill="#5F6368")


def panel_title(svg: SVG, x: float, letter: str, text: str) -> None:
    svg.text(x + 25, 210, letter, size=58, weight=700)
    svg.text(x + 88, 210, text, size=48, weight=600)


def legend_shape(svg: SVG, x: float, y: float) -> None:
    marker(svg, x, y, "#3C4043", "legacy", 11)
    svg.text(x + 25, y + 15, "LEG", size=38)
    marker(svg, x + 150, y, "#3C4043", "target_id_film", 11)
    svg.text(x + 175, y + 15, "FiLM", size=38)


def render(svg: SVG, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path.write_text(svg.finish(), encoding="utf-8")
    subprocess.run(["inkscape", str(svg_path), "--export-type=png", f"--export-filename={png_path}",
                    f"--export-width={WIDTH}"], check=True)
    print(svg_path)
    print(png_path)


def graph_and_intervention(df: pd.DataFrame) -> None:
    svg = SVG()
    title(svg, "Graph structure, control, and realized recruitment")

    panel_title(svg, 0, "a", "Connectivity ≠ target control")
    left, right, top, bottom = axes(
        svg, 0,
        [(0, "0"), (0.25, ".25"), (0.5, ".50"), (0.75, ".75"), (1, "1")],
        [(0, "0"), (.02, "2"), (.04, "4"), (.06, "6"), (.08, "8")],
        (0, 1), (0, .08), "Reliable reachable-pair fraction", "Mean |Δ action logit| × 100",
    )
    for base, group in df.groupby("base"):
        for _, pair in group.groupby(["seed", "recruitment"]):
            legacy = pair[pair.goal_conditioning == "legacy"]
            film = pair[pair.goal_conditioning == "target_id_film"]
            if len(legacy) == len(film) == 1:
                p1, p2 = legacy.iloc[0], film.iloc[0]
                svg.line(scale(p1.reachable_pair_fraction, 0, 1, left, right),
                         scale(p1.target_action_sensitivity, 0, .08, bottom, top),
                         scale(p2.reachable_pair_fraction, 0, 1, left, right),
                         scale(p2.target_action_sensitivity, 0, .08, bottom, top), cls="pair")
        for _, row in group.iterrows():
            marker(svg, scale(row.reachable_pair_fraction, 0, 1, left, right),
                   scale(row.target_action_sensitivity, 0, .08, bottom, top),
                   COLORS[base], row.goal_conditioning)

    panel_title(svg, 1100, "b", "Recruitment was mostly inactive")
    left, right, top, bottom = axes(
        svg, 1100, [(0, "C05"), (1, "C13"), (2, "C15")],
        [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5")],
        (-.45, 2.45), (0, 5.25), "Representation base", "Mean cumulative counter (window)",
    )
    offsets = {"monitor": -.21, "directional": 0, "predictive": .21}
    seed_jitter = {8: -.04, 99: 0, 123: .04}
    goal_jitter = {"legacy": -.018, "target_id_film": .018}
    for base_index, base in enumerate(["C05", "C13", "C15"]):
        for rec in ["monitor", "directional", "predictive"]:
            subset = df[(df.base == base) & (df.recruitment == rec)]
            center = base_index + offsets[rec]
            mean_y = scale(float(subset.recruitment_total.mean()), 0, 5.25, bottom, top)
            mean_x = scale(center, -.45, 2.45, left, right)
            svg.line(mean_x - 18, mean_y, mean_x + 18, mean_y, cls="axis",
                     stroke=COLORS[rec], stroke_width=7)
            for _, row in subset.iterrows():
                xval = center + seed_jitter[int(row.seed)] + goal_jitter[row.goal_conditioning]
                marker(svg, scale(xval, -.45, 2.45, left, right),
                       scale(row.recruitment_total, 0, 5.25, bottom, top),
                       COLORS[rec], row.goal_conditioning)

    y = 1145
    for i, base in enumerate(["C05", "C13", "C15"]):
        x = 105 + i * 165
        svg.circle(x, y, 11, COLORS[base], COLORS[base])
        svg.text(x + 22, y + 14, base, size=38)
    legend_shape(svg, 620, y)
    for i, rec in enumerate(["monitor", "directional", "predictive"]):
        x = 1300 + i * 250
        svg.circle(x, y, 11, COLORS[rec], COLORS[rec])
        svg.text(x + 22, y + 14, REC_LABEL[rec], size=38)
    render(svg, "interim_graph_control_62p5m_67p5m")


def representation_and_film(df: pd.DataFrame) -> None:
    svg = SVG()
    title(svg, "Representation health and FiLM goal conditioning")

    panel_title(svg, 0, "a", "C13 representation collapses")
    left, right, top, bottom = axes(
        svg, 0, [(0, "C05"), (1, "C13"), (2, "C15")],
        [(0, "0"), (.1, "10"), (.2, "20"), (.3, "30"), (.4, "40")],
        (-.45, 2.45), (0, .45), "Representation base", "Silent DG units (%)",
    )
    offsets = {"monitor": -.21, "directional": 0, "predictive": .21}
    seed_jitter = {8: -.04, 99: 0, 123: .04}
    goal_jitter = {"legacy": -.018, "target_id_film": .018}
    for base_index, base in enumerate(["C05", "C13", "C15"]):
        for rec in ["monitor", "directional", "predictive"]:
            subset = df[(df.base == base) & (df.recruitment == rec)]
            center = base_index + offsets[rec]
            mean_y = scale(float(subset.dg_silent_fraction.mean()), 0, .45, bottom, top)
            mean_x = scale(center, -.45, 2.45, left, right)
            svg.line(mean_x - 18, mean_y, mean_x + 18, mean_y, cls="axis",
                     stroke=COLORS[rec], stroke_width=7)
            for _, row in subset.iterrows():
                xval = center + seed_jitter[int(row.seed)] + goal_jitter[row.goal_conditioning]
                marker(svg, scale(xval, -.45, 2.45, left, right),
                       scale(row.dg_silent_fraction, 0, .45, bottom, top),
                       COLORS[rec], row.goal_conditioning)

    panel_title(svg, 1100, "b", "FiLM changes C15 actions weakly")
    x_positions = [(0, "LEG"), (1, "FiLM"), (2.5, "LEG"), (3.5, "FiLM"),
                   (5, "LEG"), (6, "FiLM")]
    left, right, top, bottom = axes(
        svg, 1100, x_positions,
        [(0, "0"), (.005, ".5"), (.010, "1.0"), (.015, "1.5")],
        (-.45, 6.45), (0, .016), "MON / DIR / PRED groups (left to right)", "Mean |Δ action logit| × 100",
    )
    for group_index, rec in enumerate(["monitor", "directional", "predictive"]):
        x_legacy, x_film = group_index * 2.5, group_index * 2.5 + 1
        subset = df[(df.base == "C15") & (df.recruitment == rec)]
        for seed in [8, 99, 123]:
            p1 = subset[(subset.seed == seed) & (subset.goal_conditioning == "legacy")].iloc[0]
            p2 = subset[(subset.seed == seed) & (subset.goal_conditioning == "target_id_film")].iloc[0]
            x1, x2 = scale(x_legacy, -.45, 6.45, left, right), scale(x_film, -.45, 6.45, left, right)
            y1 = scale(p1.target_action_sensitivity, 0, .016, bottom, top)
            y2 = scale(p2.target_action_sensitivity, 0, .016, bottom, top)
            svg.line(x1, y1, x2, y2, cls="axis", stroke=COLORS[rec], stroke_width=4, opacity=.72)
            marker(svg, x1, y1, COLORS[rec], "legacy")
            marker(svg, x2, y2, COLORS[rec], "target_id_film")

    y = 1145
    for i, rec in enumerate(["monitor", "directional", "predictive"]):
        x = 120 + i * 220
        svg.circle(x, y, 11, COLORS[rec], COLORS[rec])
        svg.text(x + 22, y + 14, REC_LABEL[rec], size=38)
    legend_shape(svg, 850, y)
    svg.text(1320, y + 14, "lines pair identical base/rule/seed", size=36, fill="#5F6368")
    render(svg, "interim_representation_film_62p5m_67p5m")


def main() -> None:
    if not FONT_PATH.is_file():
        raise SystemExit(f"Scalable font is required but missing: {FONT_PATH}")
    if not INPUT.is_file():
        raise SystemExit(f"Missing canonical per-run table: {INPUT}")
    df = pd.read_csv(INPUT)
    required = {"base", "seed", "recruitment", "goal_conditioning", "reachable_pair_fraction",
                "target_action_sensitivity", "recruitment_total", "dg_silent_fraction"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if len(df) != 54:
        raise SystemExit(f"Expected 54 StudySpec rows, found {len(df)}")
    graph_and_intervention(df)
    representation_and_film(df)


if __name__ == "__main__":
    main()
