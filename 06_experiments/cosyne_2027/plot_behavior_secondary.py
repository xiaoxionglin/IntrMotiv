"""Plot additional own-policy versus uniform-random representation readouts.

Uses only pinned local 50k probe tables. Evaluation repeats are averaged within
each training seed; the training seed is the unit displayed and summarized.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "06_experiments/results/behavior_field_expression_20261007"
OUT = ROOT / "06_experiments/results/cosyne_2027_behavior_secondary"
COLORS = {"C01": "#0072B2", "C05": "#D55E00", "C15": "#009E73"}
LAYERS = ("dg", "ca3", "decoder_1", "decoder_2")
NATIVE_METRICS = (
    "split_half_map_correlation", "active_only_map_cosine",
    "mean_dominant_mass_fraction", "mean_component_count", "distinct_peak_bins",
)
SHARED_METRICS = {
    "difference": "bits_per_activation",
    "bits_per_step_difference": "bits_per_step",
    "mean_activation_difference": "mean_activation",
}


def paired_native() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "per_run.csv")
    frame = frame[frame.policy.isin(("own", "uniform"))]
    if len(frame) != 144:
        raise ValueError(f"expected 144 own/uniform layer rows, got {len(frame)}")
    keys = ["condition", "seed", "eval_seed", "layer"]
    records = []
    for metric in NATIVE_METRICS:
        wide = frame.pivot(index=keys, columns="policy", values=metric)
        if wide.isna().any().any() or len(wide) != 72:
            raise ValueError(f"incomplete native pairs for {metric}")
        for key, row in wide.iterrows():
            records.append(dict(zip(keys, key), metric=metric, own=row["own"],
                                uniform=row["uniform"], difference=row["own"] - row["uniform"],
                                support="policy-native occupancy"))
    return pd.DataFrame(records)


def paired_shared() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "paired_prefixes.csv")
    frame = frame[(frame.random_policy == "uniform") & (frame.decisions == 50_000)]
    if len(frame) != 72 or not frame.supported.all():
        raise ValueError("incomplete supported 50k common-support pairs")
    records = []
    keys = ["condition", "seed", "eval_seed", "layer"]
    for column, metric in SHARED_METRICS.items():
        for row in frame.itertuples(index=False):
            record = {key: getattr(row, key) for key in keys}
            record.update(metric=metric, difference=getattr(row, column),
                          shared_bins=row.shared_bins, eligible_units=row.eligible_units,
                          support="same units and equally weighted shared bins")
            records.append(record)
    return pd.DataFrame(records)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def draw_seed_panel(ax, data: pd.DataFrame, *, metric: str, ylabel: str) -> None:
    for x, condition in enumerate(("C01", "C05", "C15")):
        values = data[(data.metric == metric) & (data.condition == condition)].sort_values("seed")
        if len(values) != 3:
            raise ValueError(f"expected three training seeds: {condition}, {metric}")
        xpoints = np.array([-0.13, 0, 0.13]) + x
        y = values.difference.to_numpy(float)
        ax.scatter(xpoints, y, color=COLORS[condition], s=45, zorder=3,
                   edgecolors="white", linewidths=0.7)
        mean = float(y.mean())
        sem = float(y.std(ddof=1) / math.sqrt(3))
        ax.errorbar(x, mean, yerr=sem, color="black", marker="_", markersize=17,
                    capsize=4, linewidth=1.3, zorder=4)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(range(3), ("C01", "C05", "C15"))
    ax.set_ylabel(ylabel)


def save(fig, name: str) -> None:
    path = OUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", dpi=300)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_probe = pd.concat([paired_native(), paired_shared()], ignore_index=True)
    by_probe.to_csv(OUT / "paired_evaluation_repeats.csv", index=False, lineterminator="\n")
    keys = ["condition", "seed", "layer", "metric", "support"]
    by_seed = by_probe.groupby(keys, as_index=False).difference.mean()
    if by_seed.groupby(["condition", "layer", "metric"]).size().ne(3).any():
        raise ValueError("missing training-seed contrasts")
    by_seed.to_csv(OUT / "paired_training_seeds.csv", index=False, lineterminator="\n")
    summary = by_seed.groupby(["condition", "layer", "metric", "support"]).difference.agg(
        n="count", mean="mean", sd="std", positive=lambda x: int((x > 0).sum()),
        negative=lambda x: int((x < 0).sum())).reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    summary.to_csv(OUT / "summary_by_layer.csv", index=False, lineterminator="\n")

    style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.1), layout="constrained")
    for ax, layer in zip(axes.flat, LAYERS):
        draw_seed_panel(ax, by_seed[by_seed.layer == layer], metric="split_half_map_correlation",
                        ylabel="Δ split-half correlation")
        ax.set_title({"dg": "DG", "ca3": "CA3", "decoder_1": "Decoder 1",
                      "decoder_2": "Decoder 2"}[layer], loc="left", fontweight="bold")
    save(fig, "own_minus_random_map_reliability")

    panels = (
        ("bits_per_activation", "Spatial info (bits/activation)"),
        ("split_half_map_correlation", "Split-half map correlation"),
        ("active_only_map_cosine", "Active-unit map cosine"),
        ("mean_activation", "Mean activation (shared support)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.1), layout="constrained")
    decoder = by_seed[by_seed.layer == "decoder_1"]
    for ax, (metric, title) in zip(axes.flat, panels):
        draw_seed_panel(ax, decoder, metric=metric, ylabel="Own − random")
        ax.set_title(title, loc="left", fontweight="bold")
    save(fig, "own_minus_random_decoder1_candidates")

    (OUT / "analysis_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/cosyne-behavior-secondary/v1",
        "source_native": str((SOURCE / "per_run.csv").relative_to(ROOT)),
        "source_shared": str((SOURCE / "paired_prefixes.csv").relative_to(ROOT)),
        "comparison": "own-policy minus uniform-random at 50k decisions",
        "replication": "two evaluation seeds averaged within each of three training seeds",
        "native_support_caveat": "split-half reliability and map cosine use each policy's own occupied bins; changes in sampled trajectories can affect them",
        "shared_support": "information and mean activation use identical eligible units and equally weighted common spatial bins",
        "uncertainty": "sample standard deviation across three training-seed differences / sqrt(3)",
        "matplotlib": matplotlib.__version__,
    }, indent=2) + "\n")
    print(f"wrote secondary behavior figures and tables to {OUT}")


if __name__ == "__main__":
    main()
