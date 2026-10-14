"""Collect and plot the verified COSYNE intrinsic-reward training course.

Collection uses the exact run identities already audited for the online DG
course. Rendering reads only the archived reward histories and copies the
physical layout and drawing style of ``01_online_spatial_course.svg``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, NullLocator
import numpy as np
import pandas as pd

import plot_wandb_online as online
import render_compact_figures as compact


ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_RUNS = ROOT / "06_experiments/results/cosyne_2027_online_wandb/collection_manifest.json"
OUTPUT = ROOT / "06_experiments/results/cosyne_2027_intrinsic_reward_course"
HISTORIES = OUTPUT / "histories"
METRIC_TAG = "intrmotiv/reward/intrinsic_mean"
SUM_TAG = "intrmotiv/reward/intrinsic_sum"
NONZERO_TAG = "intrmotiv/reward/intrinsic_nonzero_fraction"
STEP_TAG = "train/env_steps"
SEEDS = {8, 17, 41, 99, 123}
FAMILIES = ("C01", "C05", "C15")
LABELS = {"C01": "NG", "C05": "UG", "C15": "FG"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_runs() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select the same three intact five-seed cells as the DG course."""
    source = json.loads(ARCHIVED_RUNS.read_text())
    if source["schema"] != "intrmotiv/cosyne-wandb-online/v1":
        raise ValueError("Unexpected archived W&B run manifest")
    selected = [row for row in source["runs"]
                if row["encoder_credit"] == "temporal"
                and row["worker_bonus"] == ("inapplicable" if row["family"] == "C01"
                                             else "temporal")]
    actual = {family: {int(row["seed"]) for row in selected if row["family"] == family}
              for family in FAMILIES}
    if len(selected) != 15 or any(seeds != SEEDS for seeds in actual.values()):
        raise ValueError(f"Expected five verified runs per intact design: {actual}")
    # The catalog also contains other, later experiments, so its whole-file
    # hash may change while these verified run identities remain fixed.
    with (ROOT / source["catalog_path"]).open(newline="") as stream:
        current = {row["run_name"]: row for row in csv.DictReader(stream)}
    for row in selected:
        catalog = current.get(row["run_name"])
        if catalog is None or (
                catalog["study_id"], catalog["study_sha256"], catalog["seed"],
                catalog["family"], catalog["encoder_credit"], catalog["worker_bonus"],
                catalog["attempt_status"]) != (
                row["study_id"], row["study_sha256"], str(row["seed"]),
                row["family"], row["encoder_credit"], row["worker_bonus"],
                "verified_100m"):
            raise ValueError(f"Verified run identity changed: {row['run_name']}")
    return source, selected


def collect(*, workers: int) -> None:
    """Read only reward scalar histories, without replacing the DG cache."""
    import wandb

    source, selected = selected_runs()
    api = wandb.Api(timeout=120)
    remote = {run.id: run for run in api.runs(
        source["project"],
        filters={"config.study_id": {"$in": sorted({row["study_id"] for row in selected})}},
        per_page=100,
    )}
    HISTORIES.mkdir(parents=True, exist_ok=True)

    def fetch(row: dict[str, object]) -> dict[str, object]:
        run = remote.get(row["wandb_id"])
        if run is None or run.state != "finished":
            raise ValueError(f"Missing finished W&B run: {row['run_name']}")
        config = run.config
        if (str(config.get("study_id")), str(config.get("experiment")),
                str(config.get("seed"))) != (
                row["study_id"], f"00_{row['run_name']}", str(row["seed"])):
            raise ValueError(f"W&B configuration mismatch: {row['run_name']}")
        records = list(run.scan_history(
            keys=["_step", STEP_TAG, METRIC_TAG, SUM_TAG, NONZERO_TAG],
            page_size=1000,
        ))
        frame = pd.DataFrame(records).rename(columns={
            "_step": "wandb_step", STEP_TAG: "env_steps", METRIC_TAG: "intrinsic_mean",
            SUM_TAG: "intrinsic_sum", NONZERO_TAG: "intrinsic_nonzero_fraction",
        })
        columns = ["wandb_step", "env_steps", "intrinsic_mean", "intrinsic_sum",
                   "intrinsic_nonzero_fraction"]
        if not set(columns).issubset(frame):
            raise ValueError(f"Incomplete reward streams: {row['run_name']}")
        frame = (frame[columns].apply(pd.to_numeric, errors="raise")
                 .sort_values("wandb_step")
                 .drop_duplicates("env_steps", keep="last")
                 .sort_values("env_steps"))
        if len(frame) < 2500 or frame.env_steps.iloc[0] > 1_000_000 or frame.env_steps.iloc[-1] < 99_000_000:
            raise ValueError(f"Insufficient reward course: {row['run_name']}")
        if not np.isfinite(frame[columns]).all().all():
            raise ValueError(f"Nonfinite reward course: {row['run_name']}")
        if not frame.intrinsic_nonzero_fraction.between(0, 1).all():
            raise ValueError(f"Invalid reward nonzero fraction: {row['run_name']}")
        # The mean is over the learner's valid transition batch, including
        # zero-reward decisions; the paired sum provides a direct audit.
        nonzero = frame.intrinsic_mean.abs() > 1e-6
        denominators = (frame.loc[nonzero, "intrinsic_sum"] /
                        frame.loc[nonzero, "intrinsic_mean"])
        if denominators.empty or not np.isfinite(denominators).all():
            raise ValueError(f"No usable reward mean/sum pairs: {row['run_name']}")
        if not np.allclose(denominators, 2048, rtol=1e-5, atol=.02):
            raise ValueError(f"Reward mean is not over 2,048 decisions: {row['run_name']}")
        path = HISTORIES / f"{row['run_name']}.csv"
        online.write_csv(frame, path)
        return {
            "run_name": row["run_name"], "family": row["family"],
            "seed": int(row["seed"]), "study_id": row["study_id"],
            "study_sha256": row["study_sha256"], "wandb_id": row["wandb_id"],
            "wandb_url": row["wandb_url"], "history_path": str(path.relative_to(ROOT)),
            "history_sha256": sha256(path), "history_rows": len(frame),
            "first_env_steps": int(frame.env_steps.iloc[0]),
            "last_env_steps": int(frame.env_steps.iloc[-1]),
            "reward_sum_over_mean_min": float(denominators.min()),
            "reward_sum_over_mean_max": float(denominators.max()),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, row) for row in selected]):
            result = future.result()
            rows.append(result)
            print(f"Archived {result['run_name']}: {result['history_rows']} reward points", flush=True)
    rows.sort(key=lambda row: (row["family"], row["seed"]))
    (OUTPUT / "collection_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/cosyne-intrinsic-reward-course/v1",
        "source": "unsampled wandb.Api.scan_history; verified exact run identities",
        "project": source["project"], "step_tag": STEP_TAG,
        "metric_tag": METRIC_TAG, "sum_tag": SUM_TAG, "nonzero_tag": NONZERO_TAG,
        "comparison_run_manifest": str(ARCHIVED_RUNS.relative_to(ROOT)),
        "comparison_run_manifest_sha256": sha256(ARCHIVED_RUNS),
        "catalog_path": source["catalog_path"],
        "catalog_sha256": sha256(ROOT / source["catalog_path"]),
        "rows": rows,
    }, indent=2) + "\n")


def render() -> None:
    manifest_path = OUTPUT / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _, selected = selected_runs()
    if (manifest["schema"] != "intrmotiv/cosyne-intrinsic-reward-course/v1"
            or manifest["comparison_run_manifest_sha256"] != sha256(ARCHIVED_RUNS)
            or manifest["metric_tag"] != METRIC_TAG):
        raise ValueError("Intrinsic-reward collection manifest is stale")
    selected_ids = {row["run_name"] for row in selected}
    if {row["run_name"] for row in manifest["rows"]} != selected_ids:
        raise ValueError("The reward and DG courses use different runs")

    grid = np.arange(1_000_000, 100_000_000, 1_000_000)
    records = []
    for row in manifest["rows"]:
        path = ROOT / row["history_path"]
        if sha256(path) != row["history_sha256"]:
            raise ValueError(f"Reward history changed: {row['run_name']}")
        frame = pd.read_csv(path)
        values = online.interpolate_history(frame, "intrinsic_mean", grid)
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete 1–99M course: {row['run_name']}")
        records.extend({"run_name": row["run_name"], "family": row["family"],
                        "seed": row["seed"], "metric": "intrinsic_mean",
                        "env_steps": int(step), "value": float(value)}
                       for step, value in zip(grid, values))
    by_seed = pd.DataFrame(records)
    online.write_csv(by_seed, OUTPUT / "interpolated_by_seed.csv")
    summary = online.summary_by_arm(by_seed.assign(
        encoder_credit="temporal",
        worker_bonus=np.where(by_seed.family == "C01", "inapplicable", "temporal"),
    ))
    online.write_csv(summary, OUTPUT / "arm_summaries.csv")
    if len(summary) != 3 * len(grid) or not summary.n.eq(5).all():
        raise ValueError("Each plotted point needs five independent training seeds")

    font = compact.style()

    def draw_course(ax: plt.Axes) -> tuple[float, float]:
        """Keep marks, labels, and seed summaries identical across scales."""
        plotted = {}
        for family in FAMILIES:
            rows = summary[summary.family == family].sort_values("env_steps")
            x = rows.env_steps.to_numpy(float) / 1e6
            mean = rows["mean"].to_numpy(float)
            sem = rows["sem"].to_numpy(float)
            plotted[family] = (mean, sem)
            ax.plot(x, mean, lw=1.25, color=compact.COLORS[family], label=LABELS[family])
            ax.fill_between(x, mean - sem, mean + sem, color=compact.COLORS[family],
                            alpha=0.16, linewidth=0)
        for family in FAMILIES:
            ax.text(102, plotted[family][0][-1], LABELS[family], color=compact.COLORS[family],
                    va="center", fontsize=10)
        ax.set(xlim=(0, 135), xlabel="Steps (M)", ylabel="Mean intrinsic reward")
        ax.set_xticks((0, 50, 100))
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
        return (min(float(np.min(mean - sem)) for mean, sem in plotted.values()),
                max(float(np.max(mean + sem)) for mean, sem in plotted.values()))

    def export_course(fig: plt.Figure, name: str) -> None:
        stem = OUTPUT / name
        fig.savefig(stem.with_suffix(".svg"), dpi=300, metadata={"Date": None})
        svg_path = stem.with_suffix(".svg")
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        fig.savefig(stem.with_suffix(".pdf"), dpi=300,
                    metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)

    linear_name = "01b_online_intrinsic_reward_course"
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.27, top=0.95)
    low, high = draw_course(ax)
    # Intrinsic reward is signed in general; preserve negative values if this
    # cohort contains any instead of silently clipping the linear course.
    y_bottom = min(0.0, low * 1.06) if low < 0 else 0.0
    y_top = max(0.1, high * 1.06)
    ax.set_ylim(y_bottom, y_top)
    export_course(fig, linear_name)

    log_name = "01c_online_intrinsic_reward_course_log"
    if low <= 0:
        raise ValueError("A log-y course would hide a nonpositive mean or SEM bound")
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.27, top=0.95)
    draw_course(ax)
    log_limits = (0.01, 5.0)
    if low < log_limits[0] or high > log_limits[1]:
        raise ValueError("Log-y axis would clip an uncertainty band")
    ax.set_yscale("log")
    ax.set_ylim(*log_limits)
    ax.set_yticks((0.01, 0.1, 1.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(NullLocator())
    export_course(fig, log_name)
    (OUTPUT / "render_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/cosyne-intrinsic-reward-course-figure/v2",
        "figures": {"linear": linear_name, "log_y": log_name},
        "physical_size_inches": [1.9, 1.9], "font": font,
        "comparison_figure": "06_experiments/results/cosyne_2027_compact_figures/01_online_spatial_course.svg",
        "metric_tag": METRIC_TAG,
        "aggregation": "W&B logged per-update mean over valid transitions, including zero-reward transitions; plotted mean and SEM across five training seeds at each interpolated 1M-step point",
        "grid_millions": [1, 99], "seeds_per_family": 5,
        "source_collection_manifest_sha256": sha256(manifest_path),
        "source_summary_sha256": sha256(OUTPUT / "arm_summaries.csv"),
        "reward_sum_over_mean_range": [
            min(row["reward_sum_over_mean_min"] for row in manifest["rows"]),
            max(row["reward_sum_over_mean_max"] for row in manifest["rows"]),
        ],
        "rendered_y_limits": [y_bottom, y_top],
        "log_y_limits": list(log_limits),
        "minimum_mean_minus_sem": low,
        "log_axis_note": "Log transformation applies only to the display axis; arithmetic seed means and additive SEM bands are unchanged. Zero and negative values cannot appear on this variant.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="refresh reward histories from W&B")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.collect:
        collect(workers=args.workers)
    render()
