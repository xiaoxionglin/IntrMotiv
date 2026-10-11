"""Analyze cached COSYNE 100k-window DG peak spread without NEMO2 reruns.

Inputs are archived canonical collect-spatial per-unit and per-snapshot CSVs.
The pooled terminal visited-cell mask is a common spatial *domain*, not a
common observation sequence. A field is eligible by the canonical minimum of
20 active observations in at least three spatial bins.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_wandb_online import ARM_STYLES, COLORS, export, style


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_experiments/results/cosyne_2027_snapshot_peak_spread"
INPUT = OUT / "inputs"
CUTOFFS = (0.0, 0.00025, 0.0025, 0.01)
CUTOFF_LABELS = ("0%", "0.025%", "0.25%", "1%")
SIGMA = 100.0  # one 19-by-19 spatial bin in DMLab coordinate units
SIGMA_SENSITIVITY = (50.0, 100.0, 150.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    unit_files = sorted(INPUT.glob("*_per_unit.csv"))
    snapshot_files = sorted(INPUT.glob("*_per_snapshot.csv"))
    if len(unit_files) != 7 or len(snapshot_files) != 7:
        raise ValueError("expected the seven original COSYNE study tables")
    units = pd.concat((pd.read_csv(path) for path in unit_files), ignore_index=True)
    snapshots = pd.concat((pd.read_csv(path) for path in snapshot_files), ignore_index=True)
    if len(units) != 38 * 5 * 16 or len(snapshots) != 38 * 5:
        raise ValueError("source snapshot inventory changed; inspect before rendering")
    keys = ["run_name", "target_env_steps"]
    if units.groupby(keys).size().ne(16).any() or snapshots.duplicated(keys).any():
        raise ValueError("incomplete or duplicated per-unit snapshot records")
    mask_record = json.loads((OUT / "common_visited_mask.json").read_text())
    mask = np.asarray(mask_record["mask"], dtype=bool)
    if mask.shape != (19, 19) or mask_record["terminal_snapshots"] != 38:
        raise ValueError("unexpected pooled terminal visited-bin mask")
    return units, snapshots, mask


def peak_grid(xy: np.ndarray, mask: np.ndarray,
              sigma: float = SIGMA) -> np.ndarray:
    """Equal-mass Gaussian kernel at each eligible peak, pooled on fixed bins."""
    centers = np.arange(150.0, 2000.0, 100.0)
    xgrid, ygrid = np.meshgrid(centers, centers)
    if not len(xy):
        return np.full(mask.shape, np.nan)
    squared = (xgrid[None] - xy[:, 0, None, None]) ** 2
    squared += (ygrid[None] - xy[:, 1, None, None]) ** 2
    grid = np.exp(-squared / (2 * sigma**2)).sum(axis=0)
    grid[~mask] = 0
    total = grid.sum()
    return grid / total if total > 0 else np.full(mask.shape, np.nan)


def spread_metrics(xy: np.ndarray, mask: np.ndarray,
                   sigma: float = SIGMA) -> dict[str, float]:
    if len(xy) < 2:
        mean_pairwise = math.nan
        nearest = math.nan
    else:
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        mean_pairwise = float(distances[np.triu_indices(len(xy), 1)].mean())
        np.fill_diagonal(distances, np.inf)
        nearest = float(np.median(distances.min(axis=1)))
    grid = peak_grid(xy, mask, sigma)
    if not np.isfinite(grid).all():
        return dict(mean_pairwise_peak_distance=mean_pairwise,
                    median_nearest_peak_distance=nearest,
                    effective_area_fraction=math.nan, peak_density_gini=math.nan)
    probabilities = grid[mask]
    entropy = -float(np.sum(probabilities[probabilities > 0]
                            * np.log(probabilities[probabilities > 0])))
    effective_area = float(np.exp(entropy) / mask.sum())
    ordered = np.sort(probabilities)
    n = len(ordered)
    gini = float(2 * np.dot(np.arange(1, n + 1), ordered) / n - (n + 1) / n)
    return dict(mean_pairwise_peak_distance=mean_pairwise,
                median_nearest_peak_distance=nearest,
                effective_area_fraction=effective_area, peak_density_gini=gini)


def analyze(units: pd.DataFrame, snapshots: pd.DataFrame,
            mask: np.ndarray) -> pd.DataFrame:
    metadata = snapshots[[
        "run_name", "target_env_steps", "actual_env_steps", "family", "seed",
        "encoder_credit_encoder_credit", "worker_bonus", "visited_cell_fraction",
        "active_only_map_cosine", "active_unit_mean_spatial_information",
        "mono_field_unit_fraction", "mean_physical_step_distance",
        "stationary_step_fraction", "path_efficiency", "graph_available",
        "graph_reliable_global_efficiency", "graph_grounded_controllability",
    ]].rename(columns={"encoder_credit_encoder_credit": "encoder_credit"})
    metadata["encoder_credit"] = metadata["encoder_credit"].str.lower()
    metadata["worker_bonus"] = metadata["worker_bonus"].str.lower()
    records = []
    for (run_name, target), frame in units.groupby(["run_name", "target_env_steps"]):
        if len(frame) != 16 or frame.unit_id.nunique() != 16:
            raise ValueError(f"not 16 distinct DG units in {run_name}, {target}")
        meta = metadata[(metadata.run_name == run_name)
                        & (metadata.target_env_steps == target)]
        if len(meta) != 1:
            raise ValueError(f"missing metadata for {run_name}, {target}")
        base = meta.iloc[0].to_dict()
        fraction = frame.active_fraction.to_numpy(float)
        eligible = frame.field_eligible.to_numpy(bool)
        finite = np.isfinite(frame[["dominant_peak_x", "dominant_peak_y"]]).all(axis=1)
        for cutoff in CUTOFFS:
            active = fraction > cutoff if cutoff == 0 else fraction >= cutoff
            valid = active & eligible & finite
            xy = frame.loc[valid, ["dominant_peak_x", "dominant_peak_y"]].to_numpy(float)
            source_si = frame.loc[active, "spatial_information"].to_numpy(float)
            records.append({
                **base, "cutoff": cutoff, "active_units": int(active.sum()),
                "practically_silent_units": int((~active).sum()),
                "valid_peaks": int(valid.sum()),
                "distinct_valid_peak_bins": int(len(set(map(tuple, np.round(xy).astype(int))))),
                "mean_active_spatial_score": float(source_si.mean()) if len(source_si) else math.nan,
                **spread_metrics(xy, mask),
            })
    result = pd.DataFrame(records)
    if len(result) != 38 * 5 * len(CUTOFFS):
        raise ValueError("derived snapshot inventory incomplete")
    return result


def paired_terminal(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    terminal = frame[frame.target_env_steps == 100_000_000]
    metrics = (
        "active_units", "practically_silent_units", "valid_peaks",
        "distinct_valid_peak_bins", "mean_active_spatial_score",
        "mean_pairwise_peak_distance", "median_nearest_peak_distance",
        "effective_area_fraction", "peak_density_gini",
        "visited_cell_fraction", "active_only_map_cosine",
        "mono_field_unit_fraction", "mean_physical_step_distance",
        "stationary_step_fraction", "path_efficiency",
        "graph_reliable_global_efficiency", "graph_grounded_controllability",
    )
    pairs = []
    for (family, worker, seed, cutoff), group in terminal.groupby(
        ["family", "worker_bonus", "seed", "cutoff"]
    ):
        modes = group.set_index("encoder_credit")
        if not {"temporal", "constant"}.issubset(modes.index):
            continue
        for metric in metrics:
            temporal = float(modes.loc["temporal", metric])
            constant = float(modes.loc["constant", metric])
            pairs.append({
                "family": family, "worker_bonus": worker, "seed": seed,
                "cutoff": cutoff, "metric": metric, "temporal": temporal,
                "constant": constant, "difference": temporal - constant,
            })
    per_seed = pd.DataFrame(pairs)
    summary = per_seed.groupby(["family", "worker_bonus", "cutoff", "metric"]).difference.agg(
        n="count", mean="mean", sd="std",
        positive=lambda x: int((x > 0).sum()),
        negative=lambda x: int((x < 0).sum()),
    ).reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    return per_seed, summary


def draw_factor_curves(frame: pd.DataFrame, metric: str, ylabel: str,
                       stem: str, *, factor: str, levels: tuple,
                       labels: tuple[str, ...], xlabel: str) -> None:
    terminal = frame[frame.target_env_steps == 100_000_000]
    style()
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 12.0), sharex=True,
                             layout="constrained")
    for ax, family in zip(axes, ("C01", "C05", "C15")):
        for (credit, worker), group in terminal[terminal.family == family].groupby(
            ["encoder_credit", "worker_bonus"]
        ):
            means = group.groupby(factor)[metric].agg(["mean", "std", "count"])
            _, linestyle, color = ARM_STYLES[(credit, worker)]
            label = (f"{credit.title()} encoder" if worker == "inapplicable"
                     else f"{credit.title()} encoder · {worker} hit")
            x = np.arange(len(levels))
            y = means["mean"].reindex(levels).to_numpy(float)
            sem = (means["std"] / np.sqrt(means["count"])).reindex(levels).to_numpy(float)
            n = int(group[group[factor] == levels[0]].shape[0])
            ax.plot(x, y, color=color, linestyle=linestyle, marker="o",
                    linewidth=1.8, label=f"{label} (n={n})")
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.96, family, transform=ax.transAxes, va="top",
                fontweight="bold")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=2, borderaxespad=0)
    axes[-1].set_xticks(np.arange(len(levels)), labels)
    axes[-1].set_xlabel(xlabel)
    export(fig, OUT / "figures" / stem)


def bandwidth_sensitivity(units: pd.DataFrame, derived: pd.DataFrame,
                          mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = derived[(derived.target_env_steps == 100_000_000)
                   & (derived.cutoff == 0.0025)].set_index("run_name")
    rows = []
    for run_name, frame in units[units.target_env_steps == 100_000_000].groupby("run_name"):
        valid = ((frame.active_fraction >= 0.0025)
                 & frame.field_eligible.astype(bool)
                 & np.isfinite(frame[["dominant_peak_x", "dominant_peak_y"]]).all(axis=1))
        xy = frame.loc[valid, ["dominant_peak_x", "dominant_peak_y"]].to_numpy(float)
        metadata = base.loc[run_name]
        for sigma in SIGMA_SENSITIVITY:
            spread = spread_metrics(xy, mask, sigma)
            rows.append({"run_name": run_name, "target_env_steps": 100_000_000,
                         "family": metadata.family, "seed": metadata.seed,
                         "encoder_credit": metadata.encoder_credit,
                         "worker_bonus": metadata.worker_bonus, "sigma": sigma,
                         "effective_area_fraction": spread["effective_area_fraction"],
                         "peak_density_gini": spread["peak_density_gini"]})
    by_run = pd.DataFrame(rows)
    pairs = []
    for (family, worker, seed, sigma), frame in by_run.groupby(
        ["family", "worker_bonus", "seed", "sigma"]
    ):
        modes = frame.set_index("encoder_credit")
        if not {"temporal", "constant"}.issubset(modes.index):
            continue
        for metric in ("effective_area_fraction", "peak_density_gini"):
            pairs.append({"family": family, "worker_bonus": worker,
                          "seed": seed, "sigma": sigma, "metric": metric,
                          "difference": float(modes.loc["temporal", metric]
                                              - modes.loc["constant", metric])})
    return by_run, pd.DataFrame(pairs)


def draw_examples(units: pd.DataFrame, mask: np.ndarray) -> None:
    examples = (("C01", 99, "inapplicable"), ("C05", 99, "temporal"),
                ("C15", 123, "temporal"))
    styles = {"temporal": "#0072B2", "constant": "#D55E00"}
    style()
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 11.4), layout="constrained")
    heatmap = None
    for row, (family, seed, worker) in enumerate(examples):
        for col, mode in enumerate(("temporal", "constant")):
            ax = axes[row, col]
            label = f"COS27_{family}_{mode.upper()}"
            if family == "C05":
                label += f"_{worker.upper()}"
            run_name = f"{label}_S{seed}"
            frame = units[(units.run_name == run_name)
                          & (units.target_env_steps == 100_000_000)]
            if len(frame) != 16:
                raise ValueError(f"missing example: {run_name}")
            active = frame.active_fraction >= 0.0025
            valid = active & frame.field_eligible.astype(bool)
            xy = frame.loc[valid, ["dominant_peak_x", "dominant_peak_y"]].to_numpy(float)
            grid = peak_grid(xy, mask)
            heatmap = ax.imshow(np.ma.masked_where(~mask, grid), origin="lower",
                      extent=(100, 2000, 100, 2000), cmap="Greys", vmin=0, vmax=0.11,
                      interpolation="nearest", aspect="equal")
            ax.scatter(xy[:, 0], xy[:, 1], s=60, color=styles[mode],
                       edgecolors="white", linewidths=0.9, zorder=3)
            ax.set(xlim=(100, 2000), ylim=(100, 2000),
                   title=f"{family} S{seed} · {mode} ({len(xy)})")
            if col == 0:
                ax.set_ylabel("Y (DMLab units)")
            if row == 2:
                ax.set_xlabel("X (DMLab units)")
    fig.colorbar(heatmap, ax=axes, shrink=0.65,
                 label="Normalized peak mass per spatial bin")
    export(fig, OUT / "figures" / "example_valid_peaks_gaussian")


def draw_paired_checks(paired: pd.DataFrame) -> None:
    """Show behavior and stored-graph differences that could confound peaks."""
    panels = (
        ("visited_cell_fraction", "Visited-cell fraction"),
        ("active_only_map_cosine", "Active-only map cosine"),
        ("mean_physical_step_distance", "Mean physical step (DMLab units)"),
        ("graph_reliable_global_efficiency", "Stored graph efficiency"),
    )
    groups = (("C01", "inapplicable", "C01\nflat"),
              ("C05", "temporal", "C05\nT hit"),
              ("C05", "constant", "C05\nC hit"),
              ("C15", "temporal", "C15\nT hit"))
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.1), layout="constrained")
    selected = paired[paired.cutoff == 0.0025]
    for ax, (metric, title) in zip(axes.flat, panels):
        for x, (family, worker, _) in enumerate(groups):
            if metric.startswith("graph_") and family == "C01":
                continue  # flat C01 has no commanded-goal graph
            rows = selected[(selected.metric == metric)
                            & (selected.family == family)
                            & (selected.worker_bonus == worker)].sort_values("seed")
            values = rows.difference.to_numpy(float)
            if not len(values):
                raise ValueError(f"missing paired metric: {family}, {worker}, {metric}")
            offsets = np.linspace(-0.14, 0.14, len(values))
            ax.scatter(x + offsets, values, s=45, color=COLORS[family],
                       edgecolors="white", linewidths=0.8, zorder=3)
            sem = values.std(ddof=1) / math.sqrt(len(values))
            ax.errorbar(x, values.mean(), yerr=sem, marker="_", markersize=17,
                        color="black", capsize=4, zorder=4)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(len(groups)), [g[2] for g in groups])
        ax.set_ylabel("Temporal − constant")
        ax.set_title(title, loc="left", fontweight="bold")
    export(fig, OUT / "figures" / "paired_behavior_graph_checks")


def main() -> None:
    units, snapshots, mask = load_sources()
    derived = analyze(units, snapshots, mask)
    derived.to_csv(OUT / "per_snapshot_cutoff_spread.csv", index=False)
    per_seed, summary = paired_terminal(derived)
    per_seed.to_csv(OUT / "paired_terminal_by_seed.csv", index=False)
    summary.to_csv(OUT / "paired_terminal_summary.csv", index=False)
    draw_paired_checks(per_seed)
    for metric, ylabel, stem in (
        ("valid_peaks", "Valid DG peak count", "valid_peaks_by_cutoff"),
        ("mean_pairwise_peak_distance", "Mean peak distance (DMLab units)",
         "pairwise_distance_by_cutoff"),
        ("effective_area_fraction", "Effective visited-area fraction",
         "gaussian_effective_area_by_cutoff"),
        ("peak_density_gini", "Peak-density Gini", "gaussian_gini_by_cutoff"),
    ):
        draw_factor_curves(derived, metric, ylabel, stem, factor="cutoff",
                           levels=CUTOFFS, labels=CUTOFF_LABELS,
                           xlabel="Minimum DG firing frequency")
    bandwidth_runs, bandwidth_pairs = bandwidth_sensitivity(units, derived, mask)
    bandwidth_runs.to_csv(OUT / "bandwidth_by_run.csv", index=False)
    bandwidth_pairs.to_csv(OUT / "bandwidth_paired_by_seed.csv", index=False)
    draw_factor_curves(bandwidth_runs, "effective_area_fraction",
                       "Effective visited-area fraction", "gaussian_bandwidth_sensitivity",
                       factor="sigma", levels=SIGMA_SENSITIVITY,
                       labels=tuple(str(int(s)) for s in SIGMA_SENSITIVITY),
                       xlabel="Gaussian width (DMLab units)")
    draw_examples(units, mask)
    studies = {}
    for path in sorted(INPUT.glob("*_analysis_manifest.json")):
        source = json.loads(path.read_text())
        studies[source["study_id"]] = {
            "schema": source["schema"],
            "workflow_version": source["workflow_version"],
            "study_sha256": source["study_sha256"],
            "observed_snapshots": source["observed_snapshots"],
            "expected_snapshots": source["expected_snapshots"],
            "complete": source["complete"],
        }
    (OUT / "analysis_manifest.json").write_text(json.dumps({
        "schema": "cosyne27/snapshot-peak-spread/v1",
        "workflow_version": "1.14.1",
        "source": "canonical collect-spatial 100k behavior snapshots",
        "input_sha256": {p.name: sha256(p) for p in sorted(INPUT.iterdir())},
        "source_studies": studies,
        "mask_sha256": sha256(OUT / "common_visited_mask.json"),
        "terminal_snapshots": 38, "snapshots_all_ages": 190,
        "cutoff_fractions": CUTOFFS,
        "cutoff_rule": "positive activity at 0; >= cutoff otherwise",
        "valid_peak_rule": "canonical eligible: >=20 active observations in >=3 bins; finite dominant peak; meets activity cutoff",
        "gaussian_sigma_dmlab_units": SIGMA,
        "gaussian_sigma_sensitivity_dmlab_units": SIGMA_SENSITIVITY,
        "coverage": "exp(Shannon entropy of equal-mass Gaussian peak density on pooled visited bins) / pooled visited-bin count",
        "pooled_mask_caveat": "union of visited bins from all 38 terminal own-policy snapshots; not a fixed observation sequence or exact traversability mask",
        "unit_of_replication": "training seed paired within family and fixed worker bonus",
        "matplotlib": matplotlib.__version__,
    }, indent=2) + "\n")
    print(f"Analyzed {len(derived)} snapshot-cutoff cells and {len(per_seed)} paired metric cells")


if __name__ == "__main__":
    main()
