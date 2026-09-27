"""Render editable SVG trajectories with exact GCRL option-event markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def render(pose: pd.DataFrame, destination: Path, title: str, scope: str,
           bounds: tuple[float, float, float, float], sampling_label: str = "replay") -> None:
    if scope == "first_episode":
        pose = pose.loc[pose.num_traj == pose.num_traj.iloc[0]].copy()
        scope_title = f"First episode ({len(pose):,} decisions)"
    else:
        scope_title = f"Full {sampling_label} ({len(pose):,} decisions)"

    fig, ax = plt.subplots(figsize=(8.5, 8.0), layout="constrained")
    for _, episode in pose.groupby(["agent", "num_traj"], sort=False):
        ax.plot(episode.x, episode.y, color="#606973", lw=1.1, alpha=0.46, zorder=1)

    event_columns = {"option_start", "goal_hit"}
    if event_columns & set(pose.columns) and not event_columns <= set(pose.columns):
        raise ValueError("Option-start and goal-hit columns must be supplied together")
    has_events = event_columns <= set(pose.columns)
    starts = pose.loc[pose.option_start] if has_events else pose.iloc[:0]
    hits = pose.loc[pose.goal_hit] if has_events else pose.iloc[:0]
    start_size = 24 if scope == "full" else 56
    start_alpha = 0.65 if scope == "full" else 1.0
    if has_events:
        ax.scatter(starts.x, starts.y, s=start_size, marker="o", facecolor="none",
                   edgecolor="#0b69a3", linewidth=1.4, alpha=start_alpha, zorder=3,
                   label=f"Option start ({len(starts)})")
        ax.scatter(hits.x, hits.y, s=118, marker="*", facecolor="#d38a00",
                   edgecolor="#754b00", linewidth=0.55, zorder=4,
                   label=f"Goal hit ({len(hits)})")
    ax.scatter(pose.iloc[0].x, pose.iloc[0].y, s=72, marker="s",
               facecolor="#303941", edgecolor="white", linewidth=0.9,
               zorder=5, label=f"{sampling_label.capitalize()} start")
    ax.set(xlabel="Arena x (DMLab units)", ylabel="Arena y (DMLab units)",
           xlim=bounds[:2], ylim=bounds[2:])
    ax.set_title(f"{title}\n{scope_title}", pad=13)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d6dade", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, fontsize=12, columnspacing=0.9, handletextpad=0.5)
    ax.tick_params(labelsize=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--bounds", nargs=4, type=float,
                        default=(100, 2000, 100, 2000),
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Arena bounds; default matches the corrected-core open field")
    args = parser.parse_args()
    pose = pd.read_csv(args.input)
    required = {"x", "y", "agent", "num_traj", "option_start", "goal_hit"}
    if missing := required - set(pose.columns):
        raise ValueError(f"Missing pose/event columns: {sorted(missing)}")
    if pose.empty:
        raise ValueError("Empty trajectory")
    font_path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if font_path.suffix.lower() not in {".ttf", ".otf"}:
        raise ValueError(f"Expected a scalable font, found {font_path}")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                                "axes.titlesize": 14, "axes.labelsize": 13,
                                "svg.fonttype": "none"})
    for scope in ("full", "first_episode"):
        render(pose, args.output_dir / f"trajectory_option_events_{scope}.svg",
               args.title, scope, tuple(args.bounds))


if __name__ == "__main__":
    main()
