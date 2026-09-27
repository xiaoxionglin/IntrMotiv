"""Summarize checkpoint-stored directed graphs with canonical diagnostics.

Frozen place-field archives contain stored graph buffers but no fresh
prospective rollout counters. Only mono-field DG nodes get spatial coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hpc_runs.intrmotiv_study.spatial_contract import calculate_graph_diagnostics


METRICS = (
    "graph_reliable_edge_count", "graph_reliable_edge_density",
    "graph_largest_strong_component_size", "graph_reachable_pair_fraction",
    "graph_max_total_degree_fraction", "graph_spatial_endpoint_valid_fraction",
    "graph_reliable_edge_peak_distance_mean", "graph_grounded_controllability",
)


def graph_figure(adjacency: np.ndarray, valid: np.ndarray, peaks: np.ndarray,
                 output: Path, title: str) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    coords = peaks[valid]
    indices = np.flatnonzero(valid)
    for source in indices:
        for target in indices:
            if adjacency[source, target]:
                ax.annotate("", xy=peaks[target], xytext=peaks[source],
                            arrowprops={"arrowstyle": "->", "color": "#777777",
                                        "alpha": .45, "lw": 1.2})
    if len(coords):
        ax.scatter(coords[:, 0], coords[:, 1], s=95, c="#0072B2",
                   edgecolor="white", linewidth=.7, zorder=3)
    ax.set(xlabel="x (DMLab units)", ylabel="y (DMLab units)",
           title=title + f"\n{len(indices)} mono-field nodes / {len(valid)} DG units")
    ax.set_xlim(0, 1800)
    ax.set_ylim(0, 1800)
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Canonical frozen output containing raw/<run>/place_fields.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-label", action="append", default=[])
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    records = []
    for row in manifest:
        label = row["label_suffix"]
        matches = list((args.input_dir / "raw").glob(f"*__{label}/place_fields.npz"))
        if len(matches) != 1:
            raise ValueError(f"Expected one frozen archive for {label}: {matches}")
        with np.load(matches[0], allow_pickle=False) as data:
            tctrl = data["control_tctrl"]
            confidence = data["control_edge_confidence"]
            attempts = data["control_attempts"]
            valid = data["raw_dg_field_mono"].astype(bool)
            peaks = data["raw_dg_field_dominant_peak_xy"]
            if "control_prospective_attempts" in data or "control_prospective_successes" in data:
                raise ValueError(f"Unexpected prospective counters need separate analysis: {label}")
        config = json.loads((Path(row["run_dir"]) / "config.json").read_text())
        diagnostics = calculate_graph_diagnostics(
            tctrl, confidence, attempts, field_eligible=valid, dominant_peak_xy=peaks,
            confidence_threshold=float(config["hrl_edge_confidence_threshold"]),
            reliability_threshold=float(config["hrl_edge_reliability_threshold"]),
        )
        record = {"condition": row["condition"], "seed": int(row["seed"]),
                  "checkpoint_frames": int(row["checkpoint_frames"]),
                  "label_suffix": label, "dg_nodes": len(valid),
                  "mono_field_nodes": int(valid.sum()),
                  "stored_attempted_edges": int((attempts > 0).sum()),
                  "stored_confidence_per_attempt": (float(confidence.sum() / attempts.sum())
                                                    if attempts.sum() else float("nan"))}
        for key in METRICS:
            record[key] = float(np.asarray(diagnostics[key]))
        records.append(record)
        if label in args.figure_label:
            graph_figure(diagnostics["graph_reliable_adjacency"], valid, peaks,
                         args.output_dir / "figures" / f"{label}_spatial_graph.svg",
                         f"{row['condition'].split('_')[0]} · seed {row['seed']} · "
                         f"{int(row['checkpoint_frames']) / 1e6:.1f}M")
    unknown = set(args.figure_label) - {r["label_suffix"] for r in records}
    if unknown:
        raise ValueError(f"Unknown figure labels: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "stored_graph_per_run.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Summarized {len(records)} checkpoint-stored graphs")


if __name__ == "__main__":
    main()
