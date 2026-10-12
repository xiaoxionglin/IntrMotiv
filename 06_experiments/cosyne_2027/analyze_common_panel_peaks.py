"""Measure eligible DG peak spread on the fixed COSYNE observation panel.

The raw evaluator NPZs stay in the allocated NEMO2 workspace. Pass a temporary
local copy of the 40 terminal NPZs with --raw-root to regenerate these small
tables and figures. All peak densities use the same 242-bin panel occupancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze_checkpoint_fields import (
    CONTRASTS,
    OUT,
    ROOT,
    attach_catalog_factors,
)
from .analyze_snapshot_peak_spread import peak_grid, spread_metrics
from .plot_wandb_online import export, style, write_csv


SOURCE = OUT / "common_panel/summary/derived_place_field_metrics.csv"
DEST = OUT / "common_panel_analysis"
CUTOFFS = (0.0, 0.00025, 0.0025, 0.01)
CUTOFF_LABELS = ("0%", "0.025%", "0.25%", "1%")
WIDTHS = (50.0, 100.0, 150.0)
REMOTE_RAW = ("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/"
              "train_dir/analysis/cosyne27_checkpoint_20261011/common_panel/raw")


def read_npzs(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    source = pd.read_csv(SOURCE)
    if len(source) != 40 or source.label.nunique() != 40:
        raise ValueError("expected exactly 40 named common-panel source rows")
    records = []
    widths = []
    provenance = []
    example = {}
    occupancy_hashes = set()
    panel_paths = set()
    for row in source.itertuples(index=False):
        path = raw_root / row.label / "place_fields.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as array:
            occupancy = array["occupancy"] > 0
            occupancy_hashes.add(hashlib.sha256(array["occupancy"].tobytes()).hexdigest())
            panel_paths.add(str(array["observation_panel"]))
            if int(occupancy.sum()) != 242:
                raise ValueError(f"not 242 shared visited bins: {path}")
            active = array["active_fraction"]
            eligible = array["post_inhibition_field_eligible"]
            xy = array["post_inhibition_field_dominant_peak_xy"]
            if active.shape != (16,) or eligible.shape != (16,) or xy.shape != (16, 2):
                raise ValueError(f"unexpected 16-unit field arrays: {path}")
            finite = np.isfinite(xy).all(axis=1)
            for cutoff in CUTOFFS:
                keep = eligible & finite & (active > cutoff if cutoff == 0 else active >= cutoff)
                peaks = xy[keep]
                records.append({"condition": row.condition, "seed": row.seed,
                                "target_frames": row.target_frames, "label": row.label,
                                "cutoff": cutoff, "valid_peaks": int(keep.sum()),
                                **spread_metrics(peaks, occupancy, sigma=100.0)})
                if cutoff == 0.0025:
                    for sigma in WIDTHS:
                        widths.append({"condition": row.condition, "seed": row.seed,
                                       "target_frames": row.target_frames, "label": row.label,
                                       "cutoff": cutoff, "sigma_dmlab_units": sigma,
                                       "valid_peaks": int(keep.sum()),
                                       **spread_metrics(peaks, occupancy, sigma=sigma)})
                    if row.condition in {"COS27_C01_TEMPORAL", "COS27_C01_CONSTANT"} and row.seed == 99:
                        example[row.condition] = peaks.copy()
                        example["mask"] = occupancy.copy()
        provenance.append({"label": row.label,
                           "remote_npz": f"{REMOTE_RAW}/{row.label}/place_fields.npz",
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(occupancy_hashes) != 1 or len(panel_paths) != 1:
        raise ValueError("replay arrays do not share an identical panel and occupancy")
    certificate = json.loads((DEST / "occupancy_certificate.json").read_text())
    if (list(occupancy_hashes) != certificate["occupancy_sha256"]
            or list(panel_paths) != certificate["panel_paths"]):
        raise ValueError("NPZs disagree with archived common-panel certificate")
    write_csv(pd.DataFrame(provenance), DEST / "npz_source_manifest.csv")
    return (attach_catalog_factors(pd.DataFrame(records), terminal_repeats=len(CUTOFFS)),
            attach_catalog_factors(pd.DataFrame(widths), terminal_repeats=len(WIDTHS)),
            example)


def pair(rows: pd.DataFrame, group_by: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = []
    for family, worker, label in CONTRASTS:
        arm = rows[(rows.family == family) & (rows.worker_bonus == worker)]
        for (value, seed), frame in arm.groupby([group_by, "seed"]):
            if set(frame.encoder_credit) != {"temporal", "constant"}:
                continue
            by_credit = frame.set_index("encoder_credit")
            for metric in ("valid_peaks", "effective_area_fraction",
                           "peak_density_gini", "mean_pairwise_peak_distance",
                           "median_nearest_peak_distance"):
                result.append({"contrast": label, "family": family,
                               "worker_bonus": worker, group_by: value,
                               "seed": int(seed), "metric": metric,
                               "temporal": by_credit.loc["temporal", metric],
                               "constant": by_credit.loc["constant", metric],
                               "difference": (by_credit.loc["temporal", metric]
                                              - by_credit.loc["constant", metric])})
    paired = pd.DataFrame(result)
    summary = (paired.groupby(["contrast", "family", "worker_bonus", group_by,
                               "metric"], as_index=False)
               .agg(n=("seed", "size"), mean_difference=("difference", "mean"),
                    positive_pairs=("difference", lambda s: int((s > 0).sum())),
                    negative_pairs=("difference", lambda s: int((s < 0).sum()))))
    return paired, summary


def plot_cutoff_pairs(paired: pd.DataFrame) -> None:
    style()
    colors = {"C01": "#0072B2", "C05 · temporal hit": "#56B4E9",
              "C05 · constant hit": "#D55E00", "C15": "#009E73"}
    for _, _, label in CONTRASTS:
        frame = paired[(paired.contrast == label)
                       & (paired.metric == "effective_area_fraction")]
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        ax.axhline(0, color="#777777", linewidth=1)
        for seed, course in frame.groupby("seed"):
            course = course.sort_values("cutoff")
            ax.plot(range(len(CUTOFFS)), course.difference, marker="o",
                    color=colors[label], alpha=0.34, linewidth=1.2)
        mean = frame.groupby("cutoff").difference.mean().reindex(CUTOFFS)
        ax.plot(range(len(CUTOFFS)), mean, marker="o", color=colors[label],
                linewidth=2.8, label=f"Paired mean (n={frame.seed.nunique()})")
        ax.set(title=f"{label}: shared-panel peak area",
               xlabel="Minimum DG active fraction", ylabel="Temporal − constant area",
               xticks=range(len(CUTOFFS)), xticklabels=CUTOFF_LABELS)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False)
        slug = label.lower().replace(" · ", "_").replace(" ", "_")
        export(fig, DEST / "figures" / f"{slug}_effective_area_by_cutoff")


def plot_factorized_area(rows: pd.DataFrame) -> None:
    style()
    selected = rows[rows.cutoff == 0.0025]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    colors = {"temporal": "#0072B2", "constant": "#D55E00"}
    for x, (family, worker, label) in enumerate(CONTRASTS):
        subset = selected[(selected.family == family) & (selected.worker_bonus == worker)]
        for credit, offset in (("temporal", -0.16), ("constant", 0.16)):
            values = subset[subset.encoder_credit == credit].sort_values("seed")
            jitter = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(x + offset + jitter, values.effective_area_fraction,
                       color=colors[credit], s=35, edgecolor="white",
                       linewidth=0.5, zorder=2)
            ax.plot([x + offset - 0.09, x + offset + 0.09],
                    [values.effective_area_fraction.mean()] * 2,
                    color=colors[credit], linewidth=2.2, zorder=3)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color=colors[arm],
                          label=f"{arm.title()} encoder") for arm in colors]
    ax.legend(handles=handles, frameon=False, loc="best")
    ax.set(title="All arms · shared-panel Gaussian peak area",
           ylabel="Effective area / 242 visited bins", xticks=range(len(CONTRASTS)),
           xticklabels=[item[2].replace(" · ", "\n") for item in CONTRASTS])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle("Equal-mass eligible peaks · 0.25% cutoff", fontsize=14)
    export(fig, DEST / "figures/factorized_peak_effective_area")


def plot_example(example: dict[str, np.ndarray]) -> None:
    if not {"COS27_C01_TEMPORAL", "COS27_C01_CONSTANT", "mask"} <= set(example):
        raise ValueError("missing C01 seed-99 common-panel peak example")
    style()
    mask = example["mask"]
    arms = (("COS27_C01_TEMPORAL", "Temporal"),
            ("COS27_C01_CONSTANT", "Constant"))
    grids = [peak_grid(example[condition], mask, sigma=100.0) for condition, _ in arms]
    scale = max(float(np.nanmax(grid)) for grid in grids)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 4.4), layout="constrained")
    for ax, (condition, title), grid in zip(axes, arms, grids):
        plot_values = np.ma.masked_where(~mask, grid)
        cmap = plt.get_cmap("Greys").copy()
        cmap.set_bad("#dddddd")
        image = ax.imshow(plot_values, origin="lower", extent=(100, 2000, 100, 2000),
                          cmap=cmap, vmin=0, vmax=scale)
        xy = example[condition]
        ax.scatter(xy[:, 0], xy[:, 1], s=46, color="#D55E00",
                   edgecolor="white", linewidth=0.8)
        ax.set(title=f"{title} · {len(xy)} peaks", xlabel="Arena x",
               ylabel="Arena y", xlim=(100, 2000), ylim=(100, 2000))
    fig.colorbar(image, ax=axes, shrink=0.75, label="Normalized Gaussian peak density")
    fig.suptitle("C01 seed 99 · same observations · 0.25% cutoff", fontsize=14)
    export(fig, DEST / "figures/c01_seed99_common_peak_density")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Temporary local copy of common_panel/raw")
    args = parser.parse_args()
    rows, widths, example = read_npzs(args.raw_root)
    paired, summary = pair(rows, "cutoff")
    paired_width, summary_width = pair(widths, "sigma_dmlab_units")
    write_csv(rows, DEST / "peak_spread_by_run_cutoff.csv")
    write_csv(paired, DEST / "paired_peak_spread_by_seed.csv")
    write_csv(summary, DEST / "paired_peak_spread_summary.csv")
    write_csv(widths, DEST / "peak_spread_width_sensitivity.csv")
    write_csv(paired_width, DEST / "paired_peak_width_sensitivity.csv")
    write_csv(summary_width, DEST / "paired_peak_width_summary.csv")
    plot_cutoff_pairs(paired)
    plot_factorized_area(rows)
    plot_example(example)
    metadata = {"schema": "cosyne27_common_panel_peak_spread_v1",
                "source_metric": "post-inhibition field-eligible peak XY",
                "equal_peak_mass": True, "shared_visited_bins": 242,
                "cutoffs": CUTOFFS, "sigma_dmlab_units": WIDTHS,
                "source_table": str(SOURCE.relative_to(ROOT)),
                "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "npz_count": len(rows) // len(CUTOFFS)}
    (DEST / "peak_spread_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
