"""Archive unsampled COSYNE W&B spatial histories and render reproducible plots.

Collection uses exact StudySpec-derived identities in training_runs.csv. Rendering
uses only saved CSV files and never calls W&B, so figures can be revised cheaply.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study import load_study


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "06_experiments/cosyne_2027/training_runs.csv"
OUTPUT = ROOT / "06_experiments/results/cosyne_2027_online_wandb"
TAG_PREFIX = "intrmotiv/online/place_field/"
STEP_TAG = "train/env_steps"
METRICS = {
    "spatial_information": "active_unit_mean_spatial_information",
    "active_fraction": "active_unit_fraction",
    "silent_fraction": "silent_unit_fraction",
    "map_cosine": "active_only_map_cosine",
    "unique_peak_bins": "unique_active_peak_bins",
    "mono_field_fraction": "mono_field_unit_fraction",
    "peak_nearest_neighbor_distance": "median_dominant_peak_nearest_neighbor_distance",
    "primary_secondary_peak_distance": "mean_primary_secondary_peak_distance",
    "visited_cell_fraction": "visited_cell_fraction",
    "valid_sample_count": "valid_sample_count",
    "in_bounds_fraction": "in_bounds_fraction",
}
PLOT_METRICS = {
    "spatial_information": "DG spatial information",
    "active_fraction": "Active DG units (fraction)",
    "map_cosine": "Active-unit map cosine",
    "unique_peak_bins": "Distinct DG peak bins",
    "peaks_per_active_unit": "Distinct peaks per active DG unit",
    "mono_field_fraction": "Single-field fraction",
    "peak_nearest_neighbor_distance": "Nearest peak distance (DMLab units)",
    "primary_secondary_peak_distance": "Peak-component distance (DMLab units)",
    "visited_cell_fraction": "Visited arena cells (fraction)",
}
COLORS = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73"}
ARM_STYLES = {
    ("temporal", "temporal"): ("Temporal encoder · temporal hit", "-", "#0072B2"),
    ("temporal", "constant"): ("Temporal encoder · constant hit", "--", "#56B4E9"),
    ("constant", "temporal"): ("Constant encoder · temporal hit", "-.", "#D55E00"),
    ("constant", "constant"): ("Constant encoder · constant hit", ":", "#E69F00"),
    ("temporal", "inapplicable"): ("Temporal encoder", "-", "#0072B2"),
    ("constant", "inapplicable"): ("Constant encoder", "--", "#D55E00"),
}


def read_catalog() -> list[dict[str, str]]:
    with CATALOG.open(newline="") as stream:
        return list(csv.DictReader(stream))


def wandb_project(rows: list[dict[str, str]]) -> str:
    projects = set()
    for study_path in sorted({row["study_path"] for row in rows}):
        spec = load_study(ROOT / study_path)
        args = spec.expand_runs()[0].args
        settings = dict(arg[2:].split("=", 1) for arg in args if arg.startswith("--") and "=" in arg)
        projects.add((settings["wandb_user"], settings["wandb_project"]))
    if len(projects) != 1:
        raise ValueError(f"expected one W&B project, got {projects}")
    return "/".join(projects.pop())


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(*, refresh: bool, workers: int) -> None:
    import wandb

    selected = [row for row in read_catalog() if row["cohort"] == "cosyne"
                and row["attempt_status"] == "verified_100m"]
    if len(selected) != 38:
        raise ValueError(f"expected 38 verified original attempts, got {len(selected)}")
    project = wandb_project(selected)
    api = wandb.Api(timeout=120)
    study_ids = sorted({row["study_id"] for row in selected})
    remote = list(api.runs(project, filters={"config.study_id": {"$in": study_ids}}, per_page=100))
    matches = {}
    for run in remote:
        key = (str(run.config.get("study_id")), str(run.config.get("experiment")),
               str(run.config.get("seed")))
        matches.setdefault(key, []).append(run)
    pairs = []
    for row in selected:
        key = (row["study_id"], f"00_{row['run_name']}", row["seed"])
        found = matches.get(key, [])
        if len(found) != 1:
            raise ValueError(f"expected one W&B run for {key}, got {len(found)}")
        if found[0].state != "finished":
            raise ValueError(f"W&B run {found[0].id} is {found[0].state}")
        pairs.append((row, found[0]))

    tags = [TAG_PREFIX + name for name in METRICS.values()]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def fetch(pair: tuple[dict[str, str], object]) -> dict[str, object]:
        row, run = pair
        path = OUTPUT / "histories" / f"{row['run_name']}.csv"
        if not path.exists() or refresh:
            records = list(run.scan_history(keys=["_step", STEP_TAG, *tags], page_size=1000))
            if not records:
                raise ValueError(f"empty W&B spatial history: {row['run_name']}")
            frame = pd.DataFrame(records).rename(columns={STEP_TAG: "env_steps", "_step": "wandb_step",
                                                         **{TAG_PREFIX + tag: metric for metric, tag in METRICS.items()}})
            frame = frame.sort_values("wandb_step").drop_duplicates("env_steps", keep="last")
            frame = frame.sort_values("env_steps")[["wandb_step", "env_steps", *METRICS]]
            frame["env_steps"] = pd.to_numeric(frame["env_steps"], errors="raise")
            if frame["env_steps"].isna().any() or frame["env_steps"].iloc[-1] < 99_500_000:
                raise ValueError(f"incomplete spatial history: {row['run_name']}")
            if not np.isfinite(frame["spatial_information"]).all():
                raise ValueError(f"nonfinite spatial information: {row['run_name']}")
            write_csv(frame, path)
        frame = pd.read_csv(path)
        return {
            "run_name": row["run_name"], "study_id": row["study_id"],
            "study_sha256": row["study_sha256"], "family": row["family"],
            "seed": int(row["seed"]), "encoder_credit": row["encoder_credit"],
            "worker_bonus": row["worker_bonus"], "wandb_id": run.id,
            "wandb_url": run.url, "wandb_state": run.state,
            "history_path": str(path.relative_to(ROOT)), "history_sha256": sha256(path),
            "history_rows": len(frame), "first_env_steps": int(frame["env_steps"].iloc[0]),
            "last_env_steps": int(frame["env_steps"].iloc[-1]),
        }

    metadata = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, pair) for pair in pairs]
        for future in as_completed(futures):
            record = future.result()
            metadata.append(record)
            print(f"archived {record['run_name']}: {record['history_rows']} rows", flush=True)
    metadata.sort(key=lambda record: (record["family"], record["encoder_credit"],
                                      record["worker_bonus"], record["seed"]))
    (OUTPUT / "collection_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/cosyne-wandb-online/v1", "project": project,
        "source": "unsampled wandb.Api.scan_history; exact StudySpec study_id/experiment/seed match",
        "step_tag": STEP_TAG, "metric_tags": {key: TAG_PREFIX + value for key, value in METRICS.items()},
        "catalog_path": str(CATALOG.relative_to(ROOT)), "catalog_sha256": sha256(CATALOG),
        "runs": metadata,
    }, indent=2) + "\n")
    print(f"archived {len(metadata)} histories to {OUTPUT}")


def interpolate_history(frame: pd.DataFrame, metric: str, grid: np.ndarray) -> np.ndarray:
    points = frame[["env_steps", metric]].apply(pd.to_numeric, errors="coerce").dropna()
    points = points[np.isfinite(points[metric])]
    result = np.full(len(grid), np.nan)
    if len(points) < 2:
        return result
    valid = (grid >= points.env_steps.iloc[0]) & (grid <= points.env_steps.iloc[-1])
    result[valid] = np.interp(grid[valid], points.env_steps, points[metric])
    return result


def summary_by_arm(interpolated: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_keys = ["family", "encoder_credit", "worker_bonus", "metric", "env_steps"]
    for keys, frame in interpolated.groupby(group_keys, sort=True):
        values = frame.value.dropna().to_numpy(float)
        n = len(values)
        if n:
            records.append(dict(zip(group_keys, keys), n=n, mean=float(values.mean()),
                                sem=float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan))
    return pd.DataFrame(records)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def draw_curve(ax, frame: pd.DataFrame, *, label: str, color: str, linestyle: str) -> None:
    frame = frame.sort_values("env_steps")
    x = frame.env_steps.to_numpy(float) / 1_000_000
    mean = frame["mean"].to_numpy(float)
    sem = frame["sem"].to_numpy(float)
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.8)
    mask = np.isfinite(sem)
    ax.fill_between(x, mean - sem, mean + sem, where=mask, color=color, alpha=0.18,
                    linewidth=0)


def export(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", dpi=300)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def render() -> None:
    manifest = json.loads((OUTPUT / "collection_manifest.json").read_text())
    runs = manifest["runs"]
    if len(runs) != 38:
        raise ValueError("render requires all 38 completed original histories")
    grid = np.arange(1_000_000, 100_000_001, 1_000_000)
    records = []
    for row in runs:
        frame = pd.read_csv(ROOT / row["history_path"])
        frame["peaks_per_active_unit"] = frame.unique_peak_bins / (16 * frame.active_fraction).replace(0, np.nan)
        for metric in PLOT_METRICS:
            values = interpolate_history(frame, metric, grid)
            for env_steps, value in zip(grid, values):
                if np.isfinite(value):
                    records.append({"run_name": row["run_name"], "study_id": row["study_id"],
                                    "family": row["family"], "seed": row["seed"],
                                    "encoder_credit": row["encoder_credit"],
                                    "worker_bonus": row["worker_bonus"],
                                    "metric": metric, "env_steps": int(env_steps),
                                    "value": float(value)})
    by_seed = pd.DataFrame(records)
    write_csv(by_seed, OUTPUT / "interpolated_by_seed.csv")
    summary = summary_by_arm(by_seed)
    write_csv(summary, OUTPUT / "arm_summaries.csv")

    style()
    main = summary[(summary.metric == "spatial_information")
                   & (summary.encoder_credit == "temporal")
                   & (summary.worker_bonus.isin(["temporal", "inapplicable"]))]
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
    for family in ("C01", "C05", "C15"):
        curve = main[main.family == family]
        if not set(curve.n) == {5}:
            raise ValueError(f"{family} intact curve does not contain all five seeds")
        draw_curve(ax, curve, label=f"{family} (n=5)", color=COLORS[family], linestyle="-")
    ax.set(xlabel="Environment frames (millions)",
           ylabel=PLOT_METRICS["spatial_information"], xlim=(0, 100))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best")
    export(fig, OUTPUT / "figures" / "intact_spatial_information_mean_sem")

    for metric, ylabel in PLOT_METRICS.items():
        fig, axes = plt.subplots(3, 1, figsize=(7.4, 11.2), layout="constrained", sharex=True)
        for ax, family in zip(axes, ("C01", "C05", "C15")):
            subset = summary[(summary.metric == metric) & (summary.family == family)]
            for (credit, bonus), curve in subset.groupby(["encoder_credit", "worker_bonus"], sort=True):
                label, line, color = ARM_STYLES[(credit, bonus)]
                counts = sorted(set(curve.n))
                nlabel = str(counts[0]) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
                draw_curve(ax, curve, label=f"{label} (n={nlabel})", color=color,
                           linestyle=line)
            ax.set(ylabel=ylabel, xlim=(0, 100))
            if metric == "spatial_information":
                ax.set_ylim(bottom=0)
            ax.text(0.01, 0.98, family, transform=ax.transAxes, va="top", fontweight="bold")
            ax.legend(frameon=False, loc="best")
        axes[-1].set_xlabel("Environment frames (millions)")
        export(fig, OUTPUT / "figures" / f"factorized_{metric}")

    terminal = by_seed[by_seed.env_steps.between(90_000_000, 99_000_000)]
    terminal = terminal.groupby(["run_name", "study_id", "family", "seed", "encoder_credit",
                                 "worker_bonus", "metric"], as_index=False).value.mean()
    write_csv(terminal, OUTPUT / "terminal_90_99m_by_seed.csv")

    pairs = []
    for family, bonus in (("C01", "inapplicable"), ("C05", "temporal"),
                          ("C05", "constant"), ("C15", "temporal")):
        subset = terminal[(terminal.family == family) & (terminal.worker_bonus == bonus)]
        wide = subset.pivot_table(index=["family", "seed", "metric"],
                                  columns="encoder_credit", values="value")
        for (family_value, seed, metric), row in wide.dropna(subset=["temporal", "constant"]).iterrows():
            pairs.append({"contrast": "temporal_minus_constant_encoder", "family": family_value,
                          "fixed_factor": bonus, "seed": seed, "metric": metric,
                          "first": row["temporal"], "second": row["constant"],
                          "difference": row["temporal"] - row["constant"]})
    for credit in ("temporal", "constant"):
        subset = terminal[(terminal.family == "C05") & (terminal.encoder_credit == credit)]
        wide = subset.pivot_table(index=["family", "seed", "metric"],
                                  columns="worker_bonus", values="value")
        for (family_value, seed, metric), row in wide.dropna(subset=["temporal", "constant"]).iterrows():
            pairs.append({"contrast": "temporal_minus_constant_hit_bonus", "family": family_value,
                          "fixed_factor": credit, "seed": seed, "metric": metric,
                          "first": row["temporal"], "second": row["constant"],
                          "difference": row["temporal"] - row["constant"]})
    paired = pd.DataFrame(pairs)
    write_csv(paired, OUTPUT / "paired_contrasts_by_seed.csv")
    paired_summary = paired.groupby(["contrast", "family", "fixed_factor", "metric"]).difference.agg(
        n="count", mean="mean", sd="std", positive=lambda values: int((values > 0).sum()),
        negative=lambda values: int((values < 0).sum())).reset_index()
    paired_summary["sem"] = paired_summary.sd / np.sqrt(paired_summary.n)
    write_csv(paired_summary, OUTPUT / "paired_contrast_summaries.csv")
    (OUTPUT / "render_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/cosyne-wandb-render/v1",
        "source_collection_sha256": sha256(OUTPUT / "collection_manifest.json"),
        "alignment": "linear interpolation of each unsampled run history at 1M-frame grid; no smoothing or extrapolation",
        "uncertainty": "across-training-seed sample standard deviation / sqrt(n) at each frame",
        "primary_metric": "spatial_information is the amplitude-weighted online DG score over active units, not bits per activation",
        "n": "original validated runs only: five per complete arm; C15 constant encoder has three until two replacements finish",
        "matplotlib": matplotlib.__version__,
    }, indent=2) + "\n")
    print(f"rendered cached figures and tables in {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "render"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "collect":
        collect(refresh=args.refresh, workers=args.workers)
    else:
        render()


if __name__ == "__main__":
    main()
