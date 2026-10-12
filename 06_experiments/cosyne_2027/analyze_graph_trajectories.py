"""Summarize canonical COSYNE retained trajectories and directed graph outcomes.

Run this beside the NEMO2 analysis root. Full graph-edge tables and snapshots
remain in the allocated workspace; only compact summaries and selected plots
are intended for the repository. Condition factors come from validated
collect-spatial rows, never from a run-name parser.
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


COHORTS = (
    "c01_constant_s3", "c01_encoder_s2", "c01_temporal_s3",
    "c05_pushpull_s2", "c05_pushpull_s3", "c15_encoder_s2",
    "c15_encoder_s3", "c15_recovery",
)
TARGETS = (5_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)
GRAPH_COLUMNS = (
    "graph_reliable_edge_count", "graph_reachable_pair_fraction",
    "graph_prospective_attempt_count", "graph_prospective_success_fraction",
)
TRAJECTORY_COLUMNS = (
    "visited_cell_fraction", "mean_physical_step_distance",
    "stationary_step_fraction", "path_efficiency",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    snapshots = []
    edges = []
    sources = []
    for cohort in COHORTS:
        folder = root / cohort / "online_spatial"
        manifest_path = folder / "analysis_manifest.json"
        snapshot_path = folder / "per_snapshot.csv"
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != "intrmotiv/study/v1" or not manifest["include_details"]:
            raise ValueError(f"Unexpected spatial contract: {manifest_path}")
        frame = pd.read_csv(snapshot_path)
        if len(frame) != manifest["per_snapshot_rows"]:
            raise ValueError(f"Snapshot count mismatch: {cohort}")
        frame["source_cohort"] = cohort
        snapshots.append(frame)
        source = {
            "cohort": cohort, "schema": manifest["schema"],
            "workflow_version": manifest["workflow_version"],
            "study_sha256": manifest["study_sha256"],
            "study_id": manifest["study_id"],
            "snapshot_rows": len(frame),
            "per_snapshot_sha256": digest(snapshot_path),
            "analysis_manifest_sha256": digest(manifest_path),
        }
        edge_path = folder / "graph_edge.csv"
        if edge_path.exists():
            edge = pd.read_csv(edge_path)
            if len(edge) != manifest["graph_edge_rows"]:
                raise ValueError(f"Edge count mismatch: {cohort}")
            edge["source_cohort"] = cohort
            edges.append(edge)
            source["graph_edge_rows"] = len(edge)
            source["graph_edge_sha256"] = digest(edge_path)
        else:
            if frame.graph_available.fillna(0).ne(0).any():
                raise ValueError(f"Graph rows absent for graph-available cohort {cohort}")
            source["graph_edge_rows"] = 0
        sources.append(source)
    all_snapshots = pd.concat(snapshots, ignore_index=True)
    all_edges = pd.concat(edges, ignore_index=True)
    key = ["run_name", "target_env_steps"]
    if len(all_snapshots) != 200 or all_snapshots.duplicated(key).any():
        raise ValueError("Expected 200 unique retained snapshots from 40 runs")
    if sorted(all_snapshots.target_env_steps.unique()) != list(TARGETS):
        raise ValueError("Unexpected snapshot ages")
    if len(all_edges) != 150 * 240 or all_edges.groupby(key).size().ne(240).any():
        raise ValueError("Expected all 240 directed nonself edges at 150 graph snapshots")
    return all_snapshots, all_edges, sources


def graph_support(snapshots: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    key = ["run_name", "target_env_steps"]
    if (edges.source_unit == edges.target_unit).any():
        raise ValueError("Self edge in directed graph table")
    if (edges.prospective_successes > edges.prospective_attempts).any():
        raise ValueError("More prospective successes than attempts")
    if (edges.prospective_attempts < 0).any() or (edges.attempts < 0).any():
        raise ValueError("Negative directed-edge exposure")
    edge = edges.assign(
        stored_attempted=edges.attempts.gt(0).astype(int),
        prospective_attempted=edges.prospective_attempts.gt(0).astype(int),
        prospective_succeeded=edges.prospective_successes.gt(0).astype(int),
    )
    grouped = edge.groupby(key, as_index=False).agg(
        stored_attempted_edges=("stored_attempted", "sum"),
        prospective_attempted_edges=("prospective_attempted", "sum"),
        prospective_succeeded_edges=("prospective_succeeded", "sum"),
        prospective_attempts=("prospective_attempts", "sum"),
        prospective_successes=("prospective_successes", "sum"),
    )
    grouped["unattempted_prospective_edges"] = 240 - grouped.prospective_attempted_edges
    grouped["prospective_success_fraction"] = (
        grouped.prospective_successes / grouped.prospective_attempts.replace(0, np.nan)
    )
    metadata = snapshots.loc[snapshots.graph_available.eq(1), key + [
        "condition", "family", "seed", "encoder_credit", "worker_bonus",
        "actual_env_steps", "graph_reliable_edge_count",
        "graph_reachable_pair_fraction", "graph_reliable_global_efficiency",
        "graph_largest_strong_component_size", "graph_prospective_attempt_count",
        "graph_prospective_success_count", "graph_prospective_success_fraction",
    ]]
    result = metadata.merge(grouped, on=key, how="left", validate="one_to_one")
    if len(result) != 150 or result.prospective_attempted_edges.isna().any():
        raise ValueError("Graph metadata and edges do not cover the same snapshots")
    if not np.allclose(result.prospective_attempts, result.graph_prospective_attempt_count):
        raise ValueError("Prospective attempts disagree with canonical snapshot summary")
    if not np.allclose(result.prospective_successes, result.graph_prospective_success_count):
        raise ValueError("Prospective successes disagree with canonical snapshot summary")
    return result


def paired(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    keys = ["family", "worker_bonus", "seed", "target_env_steps"]
    values = frame.copy()
    values["encoder_credit"] = values.encoder_credit.str.lower()
    wide = values.pivot(index=keys, columns="encoder_credit", values=list(metrics))
    rows = []
    for key, record in wide.iterrows():
        if any(pd.isna(record[(metric, arm)]) for metric in metrics for arm in ("temporal", "constant")):
            continue
        row = dict(zip(keys, key))
        for metric in metrics:
            row[f"temporal_{metric}"] = record[(metric, "temporal")]
            row[f"constant_{metric}"] = record[(metric, "constant")]
            row[f"delta_{metric}"] = record[(metric, "temporal")] - record[(metric, "constant")]
        rows.append(row)
    return pd.DataFrame(rows)


def worker_paired(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    values = frame.loc[frame.family.eq("C05")].copy()
    values["encoder_credit"] = values.encoder_credit.str.lower()
    values["worker_bonus"] = values.worker_bonus.str.lower()
    keys = ["encoder_credit", "seed", "target_env_steps"]
    wide = values.pivot(index=keys, columns="worker_bonus", values=list(metrics))
    rows = []
    for key, record in wide.iterrows():
        row = dict(zip(keys, key))
        for metric in metrics:
            row[f"temporal_{metric}"] = record[(metric, "temporal")]
            row[f"constant_{metric}"] = record[(metric, "constant")]
            row[f"delta_{metric}"] = record[(metric, "temporal")] - record[(metric, "constant")]
        rows.append(row)
    return pd.DataFrame(rows)


def arm_summary(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    keys = ["condition", "family", "encoder_credit", "worker_bonus", "target_env_steps"]
    summary = frame.groupby(keys, as_index=False).agg(n_seeds=("seed", "nunique"))
    for metric in metrics:
        grouped = frame.groupby(keys)[metric]
        summary[f"{metric}_mean"] = grouped.mean().to_numpy()
        summary[f"{metric}_sem"] = grouped.sem().to_numpy()
    return summary


def course_figure(frame: pd.DataFrame, metric: str, title: str, ylabel: str,
                  output: Path, *, graph_only: bool = False) -> None:
    families = ("C05", "C15") if graph_only else ("C01", "C05", "C15")
    color = {"temporal": "#245c9f", "constant": "#c15b27"}
    linestyle = {"temporal": "-", "constant": "--", "inapplicable": "-"}
    for family in families:
        fig, axis = plt.subplots(figsize=(9, 6.2), constrained_layout=True)
        subset = frame.loc[frame.family.eq(family)].copy()
        subset["encoder_credit"] = subset.encoder_credit.str.lower()
        subset["worker_bonus"] = subset.worker_bonus.str.lower()
        for (encoder, worker), group in subset.groupby(["encoder_credit", "worker_bonus"]):
            count = group.run_name.nunique()
            label = encoder.title() + (f" / {worker} hit" if family == "C05" else "")
            label += f" (n={count})"
            values = group.groupby("target_env_steps")[metric]
            mean = values.mean()
            sem = values.sem().fillna(0)
            x = np.asarray(mean.index) / 1e6
            axis.fill_between(x, (mean - sem).to_numpy(), (mean + sem).to_numpy(),
                              color=color[encoder], alpha=0.12, linewidth=0)
            axis.plot(x, mean.values, label=label,
                      color=color[encoder], linestyle=linestyle[worker],
                      linewidth=2.2, marker="o", markersize=4.5)
            if family == "C05":
                # Distinguish the two worker arms without altering the encoder color.
                axis.lines[-1].set_alpha(1 if worker == "temporal" else 0.65)
        axis.set_title(f"{title} · {family}", fontsize=17)
        axis.set_xlabel("Training frames (millions)", fontsize=14)
        axis.set_ylabel(ylabel, fontsize=14)
        if metric == "visited_cell_fraction":
            axis.set_ylim(0.8, 0.9)
        axis.tick_params(labelsize=12)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=11, loc="best", frameon=False)
        family_output = output.with_name(f"{output.name}_{family.lower()}")
        for extension in ("png", "svg"):
            target = family_output.with_suffix("." + extension)
            fig.savefig(target, dpi=220, metadata={"Date": None})
            if extension == "svg":
                target.write_text("\n".join(line.rstrip() for line in
                                            target.read_text().splitlines()) + "\n")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "cosyne27-graph-trajectory"})
    snapshots, edges, sources = read_sources(args.analysis_root)
    graph = graph_support(snapshots, edges)
    trajectory = snapshots[["run_name", "condition", "family", "seed",
                            "encoder_credit", "worker_bonus", "target_env_steps",
                            "actual_env_steps", *TRAJECTORY_COLUMNS]].copy()
    graph.to_csv(args.output_dir / "directed_graph_by_snapshot.csv", index=False)
    trajectory.to_csv(args.output_dir / "trajectory_by_snapshot.csv", index=False)
    graph_metrics = ("stored_attempted_edges", "prospective_attempted_edges",
                     "prospective_success_fraction", "graph_reliable_edge_count",
                     "graph_reachable_pair_fraction")
    graph_pairs = paired(graph, graph_metrics)
    trajectory_pairs = paired(trajectory, TRAJECTORY_COLUMNS)
    if len(graph_pairs) != 75 or len(trajectory_pairs) != 100:
        raise ValueError("Paired seed inventory differs from 50 C05 + 25 C15 graph and 100 trajectory rows")
    graph_pairs.to_csv(
        args.output_dir / "paired_directed_graph.csv", index=False)
    trajectory_pairs.to_csv(
        args.output_dir / "paired_trajectory.csv", index=False)
    worker_paired(graph, graph_metrics).to_csv(
        args.output_dir / "paired_worker_directed_graph.csv", index=False)
    worker_paired(trajectory, TRAJECTORY_COLUMNS).to_csv(
        args.output_dir / "paired_worker_trajectory.csv", index=False)
    arm_summary(graph, graph_metrics).to_csv(
        args.output_dir / "graph_arm_by_age.csv", index=False)
    arm_summary(trajectory, TRAJECTORY_COLUMNS).to_csv(
        args.output_dir / "trajectory_arm_by_age.csv", index=False)
    course_figure(trajectory, "visited_cell_fraction", "Retained 100k-window occupancy",
                  "Visited fraction of 361 bins", args.output_dir / "trajectory_visited_course")
    course_figure(graph, "prospective_attempted_edges", "Directed prospective support",
                  "Attempted edges of 240", args.output_dir / "graph_attempted_course",
                  graph_only=True)
    course_figure(graph, "graph_reachable_pair_fraction", "Stored directed reachability",
                  "Reachable ordered-pair fraction", args.output_dir / "graph_reachability_course",
                  graph_only=True)
    (args.output_dir / "source_manifest.json").write_text(json.dumps({
        "schema": "cosyne27-graph-trajectory/v1", "sample_unit": "training seed",
        "trajectory_snapshots": len(trajectory), "graph_snapshots": len(graph),
        "directed_edge_rows": len(edges), "sources": sources,
        "graph_qualification": "Stored graph and prospective detector events; no executed command inference",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
