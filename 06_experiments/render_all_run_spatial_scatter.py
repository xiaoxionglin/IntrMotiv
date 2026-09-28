"""Harmonize saved run summaries and plot representation/control/exploration.

No environment, checkpoint, or event file is evaluated. Canonical spatial CSVs
are discovered by their column contract, deduplicated by run/policy/frame, and
reduced to the latest saved window per run. Frozen probes and common-history
representation panels are separate data strata with explicit provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study.spec import load_study, SpecError
from hpc_runs.intrmotiv_study.version import WORKFLOW_VERSION
from prepare_poster_exemplar_gallery import save, style


CANONICAL_COLUMNS = {"snapshot_path", "actual_env_steps", "run_name",
                     "unique_active_peak_bins", "active_unit_mean_spatial_information",
                     "visited_cell_fraction"}
ALIASES = {
    "spatial_information": "active_unit_mean_spatial_information",
    "unique_peak_bins": "unique_active_peak_bins",
    "exploration_coverage": "visited_cell_fraction",
    "map_cosine": "active_only_map_cosine",
    "graph_reachability": "graph_reachable_pair_fraction",
    "prospective_success": "graph_prospective_success_fraction",
    "prospective_attempts": "graph_prospective_attempt_count",
    "grounded_controllability": "graph_grounded_controllability",
    "stationary_fraction": "stationary_step_fraction",
    "path_efficiency": "path_efficiency",
    "mono_fraction_eligible": "mono_field_unit_fraction",
}
LABELS = {
    "spatial_information": "Mean active DG spatial score",
    "unique_peak_bins": "Distinct active DG peak bins",
    "map_cosine": "Active DG map cosine",
    "peak_bins_per_dg": "Distinct peak bins / all DG units",
    "mono_fraction_all": "Mono-field units / all DG units",
    "graph_reachability": "Stored-graph reachable pairs (fraction)",
    "prospective_success": "Prospective successes / attempts",
    "grounded_controllability": "Grounded controllability",
    "exploration_coverage": "Visited grid cells (fraction)",
    "executed_minus_shuffled": "Command success − shuffled success",
}
FAMILY_COLORS = {
    "CPD": "#0072B2", "DGP": "#D55E00", "DGC": "#009E73",
    "CPU cadence": "#CC79A7", "CA3 state": "#6A51A3",
    "Navigation8": "#A6761D", "Source credit": "#E69F00",
    "Persistent control": "#6B8E23", "Five-cue transfer": "#56B4E9",
    "Saturday": "#882255", "Corrected core": "#666666",
    "Mature reference": "#222222", "Corridor layouts": "#999933",
    "Other": "#888888",
}


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_registry(root: Path) -> tuple[dict, list[dict]]:
    """Recover capacities from validated StudySpec arguments, never run labels."""
    registry: dict[str, list[dict]] = {}
    audit = []
    for path in sorted((root / "hpc_runs/studies").glob("*.study.json")):
        try:
            study = load_study(path)
            for run in study.expand_runs():
                settings = dict(arg.split("=", 1) for arg in run.args if "=" in arg)
                value = settings.get("--Hippo_n_feature")
                if value:
                    registry.setdefault(run.name, []).append({
                        "dg_units": int(value), "study_sha256": study.fingerprint,
                        "study_schema": study.raw["schema"],
                        "study_workflow_version": study.declared_workflow_version,
                        "study_source": str(path.relative_to(root)),
                        "architecture": run.context.get("base_family"),
                        "wall_removal_probability": run.context.get("openness"),
                    })
        except (SpecError, ValueError, KeyError) as error:
            audit.append({"source": str(path.relative_to(root)), "status": "invalid_study",
                          "detail": str(error)})
    return registry, audit


def family_for_study(study: str) -> str:
    prefixes = {
        "intrmotiv_ca3_feedback": "CPD", "intrmotiv_dg_policy": "DGP",
        "intrmotiv_dg_capacity": "DGC", "controller_cpu": "CPU cadence",
        "ca3_state_goal": "CA3 state", "intrmotiv_navigation8": "Navigation8",
        "intrmotiv_source_credit": "Source credit",
        "intrmotiv_persistent": "Persistent control", "corridor_geometry": "Corridor layouts",
        "unbatched": "Five-cue transfer",
    }
    return next((label for prefix, label in prefixes.items() if study.startswith(prefix)), "Other")


def online_tables(root: Path, registry: dict, audit: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    for directory in (root / "06_experiments/data", root / "06_experiments/results"):
        for path in sorted(directory.rglob("*.csv")):
            if "cross_run_scatter" in path.relative_to(root).parts:
                continue
            try:
                columns = pd.read_csv(path, nrows=0).columns
            except (pd.errors.EmptyDataError, UnicodeDecodeError):
                continue
            if not CANONICAL_COLUMNS.issubset(columns):
                continue
            raw = pd.read_csv(path)
            frame = raw[["run_name", "condition", "seed", "policy_id", "actual_env_steps",
                         "snapshot_path"]].rename(columns={"actual_env_steps": "frames"})
            for metric, column in ALIASES.items():
                frame[metric] = pd.to_numeric(raw[column], errors="coerce") if column in raw else np.nan
            frame["protocol"] = "online_latest_saved_window"
            frame["representation_protocol"] = "policy_driven_training_window"
            frame["graph_available"] = raw.get("graph_available", pd.Series(False, index=raw.index))
            unavailable = ~frame.graph_available.fillna(False).astype(bool)
            frame.loc[unavailable, ["graph_reachability", "prospective_success",
                                    "prospective_attempts", "grounded_controllability"]] = np.nan
            frame.loc[frame.prospective_attempts <= 0,
                      ["prospective_success", "grounded_controllability"]] = np.nan
            frame["observations"] = raw.get("valid_sample_count", np.nan)
            frame["source_csv"] = str(path.relative_to(root))
            frame["source_priority"] = len(raw.columns)
            frame["source_artifact"] = frame.snapshot_path
            frame["study_batch"] = frame.snapshot_path.map(
                lambda value: str(value).split("/online_spatial/")[-1].split("/")[0])
            frame["family"] = frame.study_batch.map(family_for_study)
            frame["geometry_group"] = np.where(frame.family == "Corridor layouts",
                                                 "multiple_corridor_layouts", "legacy_19x19")
            for column in ("openness", "map_seed", "window_limit", "environment"):
                frame[column] = raw[column] if column in raw else np.nan
            frame["dg_units"] = pd.to_numeric(raw.get("capacity", pd.Series(np.nan, index=raw.index)),
                                                errors="coerce")
            for index, row in frame.iterrows():
                records = registry.get(row.run_name, [])
                capacities = {entry["dg_units"] for entry in records}
                if len(capacities) == 1:
                    value = capacities.pop()
                    if pd.notna(row.dg_units) and row.dg_units != value:
                        raise ValueError(f"Conflicting capacity for {row.run_name}")
                    frame.loc[index, "dg_units"] = value
                frame.loc[index, "study_sha256"] = "|".join(sorted({r["study_sha256"] for r in records}))
                if row.family == "Corridor layouts":
                    architectures = {r["architecture"] for r in records if r.get("architecture")}
                    probabilities = {r["wall_removal_probability"] for r in records
                                     if r.get("wall_removal_probability") is not None}
                    if len(architectures) != 1 or probabilities != {row.openness}:
                        raise ValueError(f"Corridor factors disagree with StudySpec: {row.run_name}")
                    frame.loc[index, "architecture"] = architectures.pop()
            pieces.append(frame)
            audit.append({"source": str(path.relative_to(root)), "status": "canonical_online",
                          "rows": len(raw), "sha256": fingerprint(path)})
    combined = pd.concat(pieces, ignore_index=True)
    keys = ["run_name", "policy_id", "frames"]
    # Prefer the export with richer metadata, then its stable source path. This
    # choice is independent of metric values and is recorded in provenance.
    combined = combined.sort_values(["source_priority", "source_csv"], ascending=[False, True])
    for metric in ALIASES:
        bounds = combined.groupby(keys)[metric].agg(["min", "max"])
        inconsistent = bounds[(bounds["max"] - bounds["min"]).abs() > 1e-5]
        if len(inconsistent):
            raise ValueError(f"Duplicate snapshot exports disagree on {metric}: {inconsistent.head()}")
    snapshots = combined.groupby(keys, as_index=False, sort=False).first()
    snapshots["duplicate_export_count"] = combined.groupby(keys).size().reindex(
        pd.MultiIndex.from_frame(snapshots[keys])).to_numpy()
    latest = snapshots.sort_values("frames").groupby(["run_name", "policy_id"], as_index=False).tail(1)
    return latest, snapshots


def family_for_poster(condition: str) -> str:
    # This is a display-only family label; conditions/seeds are read unchanged.
    return next((label for prefix, label in {
        "SAT_": "Saturday", "CPU2048": "CPU cadence", "DGP_": "DGP",
        "DGC_": "DGC", "CR5C_": "Five-cue transfer", "FSCS_": "Mature reference",
        "CCR_": "Corrected core", "CPD_": "CPD",
    }.items() if condition.startswith(prefix)), "Other")


def frozen_tables(root: Path, audit: list[dict]) -> pd.DataFrame:
    report = root / "06_experiments/results/A0_poster_analysis_20260926"
    poster_path = report / "exemplar_gallery/exemplar_inventory.csv"
    raw = pd.read_csv(poster_path)
    poster = raw[["condition", "seed", "checkpoint_frames"]].rename(columns={"checkpoint_frames": "frames"})
    mapping = {"spatial_information": "active_unit_mean_si_bits", "unique_peak_bins": "active_unique_peak_bins",
               "exploration_coverage": "visited_cell_fraction", "graph_reachability": "graph_reachable_pair_fraction",
               "dg_units": "dg_units_total", "mono_fraction_all": "mono_field_fraction_all_dg",
               "mono_peak_bins": "mono_field_peak_bins", "map_cosine": "active_map_cosine_mean"}
    for metric, column in mapping.items():
        poster[metric] = raw[column]
    poster["run_name"] = raw.condition + "_S" + raw.seed.astype(str)
    poster["source_artifact"] = raw.label_suffix
    poster["source_csv"] = str(poster_path.relative_to(root))
    poster["family"] = raw.condition.map(family_for_poster)
    poster["source_priority"] = 2
    pieces = [poster]
    core_path = root / "06_experiments/data/flat_goal_comparison_20260927/corrected_core_candidates_20260902_place_fields/summary/derived_place_field_metrics.csv"
    if core_path.exists():
        raw = pd.read_csv(core_path)
        core = raw[["condition", "seed", "checkpoint_frames"]].rename(columns={"checkpoint_frames": "frames"})
        for metric, column in {"spatial_information": "active_unit_mean_si_bits",
                               "unique_peak_bins": "active_unique_peak_bins",
                               "exploration_coverage": "visited_cell_fraction",
                               "map_cosine": "active_map_cosine_mean"}.items():
            core[metric] = raw[column]
        core["run_name"] = raw.run_dir.map(lambda path: Path(str(path)).name.removeprefix("00_"))
        core["dg_units"] = raw.active_units + raw.silent_units
        core["source_csv"] = str(core_path.relative_to(root))
        core["source_artifact"] = raw.checkpoint
        core["family"] = "Corrected core"
        core["source_priority"] = 1
        pieces.append(core)
    hist_path = report / "exemplar_gallery/historical_extension/historical_exemplar_inventory.csv"
    raw = pd.read_csv(hist_path)
    raw = raw[raw.protocol.str.startswith("frozen")].copy()
    hist = raw[["condition", "seed", "run_name", "frames", "dg_units"]].copy()
    for metric, column in {"spatial_information": "active_mean_si_bits", "unique_peak_bins": "active_unique_peak_bins",
                           "exploration_coverage": "visited_fraction", "map_cosine": "active_map_cosine",
                           "mono_fraction_all": "mono_fraction_all_dg", "mono_peak_bins": "mono_peak_bins",
                           "graph_reachability": "stored_graph_reachable_pair_fraction"}.items():
        hist[metric] = raw[column]
    # Frozen CPD has stored buffers, but no prospective attempts. The canonical
    # graph helper's zero in that case is not a measured success probability.
    hist["source_artifact"] = raw.raw_source
    hist["source_csv"] = str(hist_path.relative_to(root))
    hist["family"] = raw.condition.map(family_for_poster)
    hist["source_priority"] = 3
    pieces.append(hist)
    for path in [poster_path, hist_path, core_path]:
        if path.exists():
            audit.append({"source": str(path.relative_to(root)), "status": "frozen_probe_summary",
                          "rows": len(pd.read_csv(path)), "sha256": fingerprint(path)})
    combined = pd.concat(pieces, ignore_index=True).sort_values(["source_priority", "frames"], ascending=False)
    merged = combined.groupby(["run_name", "frames"], as_index=False, sort=False).first()
    merged = merged.sort_values("frames").groupby("run_name", as_index=False).tail(1)
    merged["protocol"] = "frozen_latest_archived_probe"
    merged["representation_protocol"] = "policy_driven_frozen_probe"
    merged["geometry_group"] = "legacy_19x19"
    merged["observations"] = 10001
    return merged


def common_replay_rows(root: Path, frozen: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    base = root / "06_experiments/data/poster_missing_analyses_20260926"
    files = ("replay_reduced_derived.csv", "replay_cpu_derived.csv",
             "dgc_direct_latest_replay_derived.csv", "dgc_waypoint_replay_derived.csv",
             "d50_75m_latest_allcue_replay_derived.csv", "d51_75m_allcue_replay_derived.csv",
             "mature_replay_derived.csv")
    pieces = []
    for name in files:
        path = base / name
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        replay = raw[["condition", "seed", "checkpoint_frames"]].rename(columns={"checkpoint_frames": "frames"})
        replay["replay_spatial_information"] = raw.active_unit_mean_si_bits
        replay["replay_unique_peak_bins"] = raw.active_unique_peak_bins
        replay["representation_panel_coverage"] = raw.visited_cell_fraction
        replay["representation_observations"] = raw.frames
        replay["representation_source_csv"] = str(path.relative_to(root))
        pieces.append(replay)
        audit.append({"source": str(path.relative_to(root)), "status": "common_replay_summary",
                      "rows": len(raw), "sha256": fingerprint(path)})
    representations = pd.concat(pieces, ignore_index=True).drop_duplicates(["condition", "seed", "frames"])
    manifests = []
    for name in ("kernel_fullrange_manifest.tsv", "kernel_fullrange_d50_endpoint_manifest.tsv",
                 "kernel_fullrange_d51_endpoint_manifest.tsv"):
        path = base / "fullrange_plan" / name
        manifest = pd.read_csv(path, sep="\t").rename(columns={"checkpoint_frames": "frames",
                                                               "panel": "representation_panel"})
        manifests.append(manifest[["condition", "seed", "frames", "representation_panel"]])
        audit.append({"source": str(path.relative_to(root)), "status": "representation_panel_manifest",
                      "rows": len(manifest), "sha256": fingerprint(path)})
    panels = pd.concat(manifests, ignore_index=True).drop_duplicates(["condition", "seed", "frames"], keep="last")
    representations = representations.merge(panels, how="left", on=["condition", "seed", "frames"],
                                            validate="one_to_one")
    rows = frozen.merge(representations, on=["condition", "seed", "frames"], how="inner", validate="one_to_one")
    rows["spatial_information"] = rows.replay_spatial_information
    rows["unique_peak_bins"] = rows.replay_unique_peak_bins
    rows["protocol"] = "common_replay_with_frozen_outcomes"
    rows["representation_protocol"] = "common_observation_action_history"
    if rows.representation_panel.isna().any():
        raise ValueError("Common replay row lacks authoritative observation-panel provenance")
    rows["mono_fraction_all"] = np.nan  # Frozen-policy fields are a different input distribution.
    control_path = base / "control_interventions_per_run.csv"
    controls = pd.read_csv(control_path).rename(columns={"checkpoint_frames": "frames"})
    rows = rows.merge(controls[["condition", "seed", "frames", "executed_minus_shuffled",
                                "executed_success", "matched_shuffled_success"]],
                      how="left", on=["condition", "seed", "frames"], validate="one_to_one")
    audit.append({"source": str(control_path.relative_to(root)), "status": "matched_command_intervention",
                  "rows": len(controls), "sha256": fingerprint(control_path)})
    return rows


def draw_scatter(ax, data: pd.DataFrame, x: str, y: str, title: str,
                 color_column: str = "family", colors: dict | None = None,
                 marker_column: str = "dg_units", markers: dict | None = None) -> int:
    usable = data.dropna(subset=[x, y])
    colors = FAMILY_COLORS if colors is None else colors
    markers = {16: "o", 32: "^", 64: "s"} if markers is None else markers
    for family, group in usable.groupby(color_column):
        for index, (capacity, subset) in enumerate(group.groupby(marker_column, dropna=False)):
            marker = markers.get(capacity, "D")
            ax.scatter(subset[x], subset[y], s=40, marker=marker,
                       color=colors.get(family, "#888888"), alpha=.72,
                       edgecolor="white", linewidth=.5,
                       label=family if index == 0 else "_nolegend_", rasterized=False)
    ax.set(xlabel=LABELS[x], ylabel=LABELS[y], title=f"{title} · n={len(usable)}")
    if x not in ("executed_minus_shuffled",):
        ax.set_xlim(left=-.02 if usable[x].max() <= 1 else -.5)
    if y in ("graph_reachability", "exploration_coverage", "prospective_success", "grounded_controllability"):
        ax.set_ylim(-.025, 1.025)
    if x in ("peak_bins_per_dg", "mono_fraction_all", "graph_reachability", "prospective_success"):
        ax.set_xlim(-.025, 1.025)
    if y == "executed_minus_shuffled":
        ax.axhline(0, color="#777777", linewidth=1.2)
    ax.grid(alpha=.18, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    if usable.empty:
        ax.text(.5, .5, "Metric unavailable\nfor this saved protocol", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#666666")
        ax.set_xlim(0, 1)
    return len(usable)


def capacity_legend(data: pd.DataFrame) -> dict[str, Line2D]:
    return {f"DG {int(units)}": Line2D([], [], color="#777777", marker=marker,
                                       linestyle="none", markersize=6)
            for units, marker in ((16, "o"), (32, "^"), (64, "s"))
            if units in data.dg_units.values}


def sheet(data: pd.DataFrame, pairs: list[tuple[str, str, str]], output: Path,
          title: str, columns: int = 2, zoom_coverage: bool = False,
          encoding: dict | None = None) -> None:
    style()
    rows = int(np.ceil(len(pairs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.9 * rows),
                             squeeze=False, constrained_layout=True)
    legend_handles = {}
    for ax, (x, y, label) in zip(axes.flat, pairs):
        draw_scatter(ax, data, x, y, label, **(encoding or {}))
        if zoom_coverage and y == "exploration_coverage":
            limits = data.dropna(subset=[x, y])[y]
            ax.set_ylim(max(0, limits.min() - .015), min(1, limits.max() + .015))
        handles, labels = ax.get_legend_handles_labels()
        legend_handles.update(zip(labels, handles))
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    if encoding:
        legend_handles = {name: Line2D([], [], color=color, linewidth=3)
                          for name, color in encoding["colors"].items()}
        legend_handles.update({name: Line2D([], [], color="#777777", marker=marker,
                                            linestyle="none", markersize=6)
                               for name, marker in encoding["markers"].items()})
    else:
        legend_handles.update(capacity_legend(data))
    fig.suptitle(title, fontsize=14)
    fig.legend(legend_handles.values(), legend_handles.keys(), loc="upper center",
               bbox_to_anchor=(.5, -.005), ncol=3, frameon=False, fontsize=12)
    save(fig, output)
    # Keep a readable legend outside the scientific panels and export each
    # panel independently for poster selection. No downscaled dense legend.
    for x, y, label in pairs:
        fig, ax = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
        draw_scatter(ax, data, x, y, label, **(encoding or {}))
        if zoom_coverage and y == "exploration_coverage":
            limits = data.dropna(subset=[x, y])[y]
            ax.set_ylim(max(0, limits.min() - .015), min(1, limits.max() + .015))
        save(fig, output.parent / "panels" / f"{output.stem}__{x}_vs_{y}.svg")
    fig, ax = plt.subplots(figsize=(8.6, max(1.4, .38 * len(legend_handles))), constrained_layout=True)
    ax.axis("off")
    ax.legend(legend_handles.values(), legend_handles.keys(), loc="center", ncol=2,
              frameon=False, title=title.split("\n")[0])
    save(fig, output.with_name(output.stem + "_legend.svg"))


def within_family_control(data: pd.DataFrame, output: Path) -> None:
    """Expose family separation behind a pooled representation/control trend."""
    families = sorted(data.family.unique())
    style()
    fig, axes = plt.subplots(int(np.ceil(len(families) / 3)), 3,
                             figsize=(12.6, 3.9 * int(np.ceil(len(families) / 3))),
                             squeeze=False, constrained_layout=True)
    for ax, family in zip(axes.flat, families):
        subset = data[data.family == family]
        usable = subset.dropna(subset=["spatial_information", "prospective_success"])
        rho = usable.spatial_information.corr(usable.prospective_success, method="spearman") \
            if usable.spatial_information.nunique() > 1 and usable.prospective_success.nunique() > 1 else np.nan
        title = f"{family} · rho={rho:.2f}" if np.isfinite(rho) else family
        draw_scatter(ax, subset, "spatial_information", "prospective_success", title)
        panel, single = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
        draw_scatter(single, subset, "spatial_information", "prospective_success", title)
        save(panel, output.parent / "panels" / f"online_within_family__{family.lower().replace(' ', '_')}.svg")
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.suptitle("Within-family online association · each point is a run · descriptive Spearman rho",
                 fontsize=14)
    handles = capacity_legend(data)
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               bbox_to_anchor=(.5, -.005), ncol=3, frameon=False, fontsize=12)
    save(fig, output)


def peak_capacity_contrast(data: pd.DataFrame, output: Path) -> None:
    """Place raw and capacity-normalized peak counts side by side."""
    style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8), constrained_layout=True)
    handles = {}
    for row_index, (protocol, label) in enumerate((
        ("online_latest_saved_window", "Online windows"),
        ("frozen_latest_archived_probe", "Frozen probes"),
    )):
        subset = data[(data.protocol == protocol) & (data.geometry_group == "legacy_19x19")]
        for ax, x in zip(axes[row_index], ("unique_peak_bins", "peak_bins_per_dg")):
            usable = subset.dropna(subset=[x, "graph_reachability"])
            rho = usable[x].corr(usable.graph_reachability, method="spearman")
            draw_scatter(ax, subset, x, "graph_reachability", f"{label} · rho={rho:.2f}")
            artists, labels = ax.get_legend_handles_labels()
            handles.update(zip(labels, artists))
    handles.update(capacity_legend(data))
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               bbox_to_anchor=(.5, -.005), ncol=3, frameon=False, fontsize=12)
    fig.suptitle("Raw peak count and capacity-normalized diversity · descriptive run survey",
                 fontsize=14)
    save(fig, output)


def corridor_comparisons(data: pd.DataFrame, output: Path, data_output: Path,
                         pairs: list[tuple[str, str, str]]) -> None:
    """Expose wall-removal probability and retain architecture/layout replication.

    Layout seeds are geometric replicates, all with training seed 99. The
    connected traces identify equal layout seeds across probabilities; they
    are descriptive and do not imply nested wall sets or independent training
    replications. Means are shown without inferential error bars.
    """
    data = data.copy()
    if data.frames.nunique() != 1:
        raise ValueError("Corridor contrast requires one shared saved age")
    if data.duplicated(["architecture", "openness", "map_seed", "seed"]).any():
        raise ValueError("Duplicated corridor run factors")
    architecture_labels = {"SAT": "SAT arrival F16", "DGP": "DGP hit F16",
                           "WAYPOINT_HER": "Waypoint HER F64"}
    data["architecture_label"] = data.architecture.map(architecture_labels)
    data["wall_removal_probability"] = data.openness
    data["wall_removal_label"] = data.openness.map(lambda p: f"Wall removal {p:.0%}")
    probabilities = sorted(data.openness.unique())
    colors = dict(zip([f"Wall removal {p:.0%}" for p in probabilities],
                      ["#440154", "#21918C", "#E69F00"]))
    markers = dict(zip(architecture_labels.values(), ["o", "^", "s"]))
    sheet(data, pairs, output / "corridor_layouts.svg",
          "Corridor layouts · color: wall-removal probability · shape: architecture",
          encoding={"color_column": "wall_removal_label", "colors": colors,
                    "marker_column": "architecture_label", "markers": markers})
    metric_groups = {
        "representation": ["spatial_information", "unique_peak_bins", "peak_bins_per_dg", "map_cosine"],
        "control_exploration": ["graph_reachability", "prospective_success",
                                "grounded_controllability", "exploration_coverage"],
    }
    metrics = sum(metric_groups.values(), [])
    data.to_csv(data_output / "corridor_probability_per_run.csv", index=False)
    long = data.melt(id_vars=["architecture", "openness", "map_seed", "seed", "frames"],
                     value_vars=metrics, var_name="metric", value_name="value")
    summary = long.groupby(["architecture", "openness", "metric"]).value.agg(
        n="count", mean="mean", std_across_layouts="std", minimum="min", maximum="max").reset_index()
    summary.to_csv(data_output / "corridor_probability_summary.csv", index=False)
    paired = data.pivot(index=["architecture", "map_seed", "seed"], columns="openness", values=metrics)
    differences = pd.DataFrame({metric: paired[(metric, probabilities[-1])] - paired[(metric, probabilities[0])]
                                for metric in metrics}).reset_index()
    differences["probability_from"] = probabilities[0]
    differences["probability_to"] = probabilities[-1]
    differences.to_csv(data_output / "corridor_probability_paired_differences.csv", index=False)
    architecture_colors = dict(zip(architecture_labels.values(), ["#0072B2", "#D55E00", "#009E73"]))

    def draw_probability(ax, metric):
        for offset, architecture in zip([-.025, 0, .025], architecture_labels.values()):
            subset = data[data.architecture_label == architecture]
            color = architecture_colors[architecture]
            for _, group in subset.groupby("map_seed"):
                group = group.sort_values("openness")
                ax.plot(group.openness + offset, group[metric], color=color, alpha=.25,
                        linewidth=1, marker=markers[architecture], markersize=5)
            average = subset.groupby("openness")[metric].mean()
            ax.plot(average.index + offset, average.values, color=color, linewidth=1.5,
                    marker=markers[architecture], markersize=7, label=architecture)
        ax.set(xlabel="Wall-removal probability", ylabel=LABELS[metric],
               title="3 layout seeds per architecture × probability", xlim=(-.10, .85))
        ax.set_xticks(probabilities, [f"{p:.0%}" for p in probabilities])
        if metric in ("graph_reachability", "prospective_success", "grounded_controllability",
                      "exploration_coverage", "peak_bins_per_dg", "map_cosine"):
            ax.set_ylim(-.025, 1.025)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.18, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)

    for kind, selected in metric_groups.items():
        style()
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8), constrained_layout=True)
        for ax, metric in zip(axes.flat, selected):
            draw_probability(ax, metric)
            panel, single = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
            draw_probability(single, metric)
            save(panel, output / "panels" / f"corridor_probability__{metric}.svg")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.005),
                   ncol=3, frameon=False, fontsize=12)
        fig.suptitle("Wall removal · faint: layout seeds · bold: mean · training seed 99", fontsize=14)
        save(fig, output / f"corridor_probability_{kind}.svg")


def associations(data: pd.DataFrame) -> pd.DataFrame:
    pairs = [(x, y) for x in ("spatial_information", "unique_peak_bins", "peak_bins_per_dg", "mono_fraction_all")
             for y in ("graph_reachability", "prospective_success", "grounded_controllability", "exploration_coverage")]
    pairs += [("graph_reachability", "exploration_coverage"),
              ("prospective_success", "exploration_coverage"),
              ("spatial_information", "executed_minus_shuffled")]
    records = []
    for (protocol, geometry), group in data.groupby(["protocol", "geometry_group"]):
        groups = [("all_families_descriptive", "all", group)]
        groups += [(family, "all", subset) for family, subset in group.groupby("family")]
        groups += [(family, f"DG{int(capacity)}", subset)
                   for (family, capacity), subset in group.groupby(["family", "dg_units"])]
        for family, capacity_stratum, subset in groups:
            for x, y in pairs:
                usable = subset.dropna(subset=[x, y])
                if len(usable) < 3:
                    continue
                rho = usable[x].corr(usable[y], method="spearman") if usable[x].nunique() > 1 and usable[y].nunique() > 1 else np.nan
                records.append({"protocol": protocol, "geometry_group": geometry, "family": family,
                                "capacity_stratum": capacity_stratum,
                                "x": x, "y": y, "n_run_rows": len(usable), "spearman_rho": rho})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path,
                        help="Replot from the pinned input_sources directory rather than live summary exports")
    args = parser.parse_args()
    audit = []
    input_root = args.source_bundle or args.root
    registry_root = input_root if (input_root / "hpc_runs/studies").exists() else args.root
    registry, study_audit = canonical_registry(registry_root)
    audit.extend(study_audit)
    online, snapshots = online_tables(input_root, registry, audit)
    frozen = frozen_tables(input_root, audit)
    capacities = frozen.set_index("run_name").dg_units
    online["dg_units"] = online.dg_units.fillna(online.run_name.map(capacities))
    replay = common_replay_rows(input_root, frozen, audit)
    data = pd.concat([online, frozen, replay], ignore_index=True)
    data["representation_observations"] = data.representation_observations.fillna(data.observations)
    data["outcome_observations"] = data.observations
    for column in ("mono_fraction_all", "executed_minus_shuffled", "prospective_success",
                   "prospective_attempts", "grounded_controllability"):
        if column not in data:
            data[column] = np.nan
    data["peak_bins_per_dg"] = data.unique_peak_bins / data.dg_units
    data = data.sort_values(["protocol", "family", "run_name"])
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data / "all_run_metrics.csv", index=False)
    snapshots.to_csv(args.data / "online_snapshot_inventory.csv", index=False)
    pd.DataFrame(audit).to_csv(args.data / "source_audit.csv", index=False)
    stats = associations(data)
    stats.to_csv(args.data / "descriptive_associations.csv", index=False)
    pairs = [("spatial_information", "graph_reachability", "Representation and stored graph"),
             ("unique_peak_bins", "graph_reachability", "Peak diversity and stored graph"),
             ("spatial_information", "exploration_coverage", "Representation and exploration"),
             ("unique_peak_bins", "exploration_coverage", "Peak diversity and exploration")]
    for protocol, group in data[data.geometry_group == "legacy_19x19"].groupby("protocol"):
        title = protocol.replace("_", " ").capitalize()
        sheet(group, pairs, args.output / f"{protocol}.svg", title)
        if protocol == "online_latest_saved_window":
            extra = [(x, y, label) for x in ("spatial_information", "unique_peak_bins")
                     for y, label in (("prospective_success", "Measured prospective control"),
                                      ("grounded_controllability", "Spatially grounded control"))]
            sheet(group, extra, args.output / "online_prospective_control.svg",
                  "Online saved windows · measured prospective control")
            sheet(group, [pair for pair in pairs if pair[1] == "exploration_coverage"],
                  args.output / "online_exploration_detail.svg",
                  "Online exploration detail · all points included · expanded coverage scale",
                  zoom_coverage=True)
            within_family_control(group, args.output / "online_within_family_control.svg")
        elif protocol == "common_replay_with_frozen_outcomes":
            sheet(group, [(x, "executed_minus_shuffled", "Matched command intervention")
                          for x in ("spatial_information", "unique_peak_bins")],
                  args.output / "common_replay_command_control.svg",
                  "Common-history representation · commanded minus shuffled success")
        normal = [("peak_bins_per_dg", y, "Capacity-normalized peak diversity")
                  for y in ("graph_reachability", "exploration_coverage")]
        if group.mono_fraction_all.notna().any():
            normal += [("mono_fraction_all", y, "Mono-field fraction of all DG")
                       for y in ("graph_reachability", "exploration_coverage")]
        sheet(group, normal, args.output / f"{protocol}_normalized.svg", title + " · capacity and mono fields")
    geometry = data[data.geometry_group == "multiple_corridor_layouts"]
    if len(geometry):
        corridor_comparisons(geometry, args.output, args.data, pairs)
    peak_capacity_contrast(data, args.output / "peak_capacity_contrast.svg")
    for protocol in ("online_latest_saved_window", "frozen_latest_archived_probe"):
        subset = data[(data.geometry_group == "legacy_19x19") & (data.protocol == protocol)]
        control_pairs = [("graph_reachability", "exploration_coverage", "Stored graph and exploration")]
        if protocol.startswith("online"):
            control_pairs.append(("prospective_success", "exploration_coverage", "Prospective control and exploration"))
        sheet(subset, control_pairs, args.output / f"{protocol}_control_exploration.svg",
              protocol.replace("_", " ").capitalize() + " · control and exploration",
              columns=len(control_pairs))
    metadata = {"analysis_schema": "intrmotiv/cross-run-scatter/v1", "workflow_version": WORKFLOW_VERSION,
                "sample_unit": "one run/policy at its latest saved observation window or archived frozen probe",
                "selection": "latest available age per run and protocol; common replay must join exact probe age",
                "input_sources": audit, "rows_by_protocol": data.groupby("protocol").size().to_dict(),
                "unique_run_names": data.run_name.nunique(), "online_unique_snapshots": len(snapshots),
                "font": "Times New Roman, verified scalable font through shared poster style",
                "matplotlib_version": matplotlib.__version__, "pandas_version": pd.__version__,
                "numpy_version": np.__version__}
    relevant_studies = {entry["study_sha256"]: {key: value for key, value in entry.items()
                                              if key not in ("architecture", "wall_removal_probability")}
                        for name in data.run_name.unique()
                        for entry in registry.get(name, [])}
    metadata["resolved_studies"] = list(relevant_studies.values())
    if not args.source_bundle:
        bundle = args.data / "input_sources"
        paths = {entry["source"] for entry in audit if entry.get("sha256")}
        paths.update(entry["study_source"] for entry in relevant_studies.values())
        for relative in sorted(paths):
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.root / relative, destination)
    metadata["pinned_input_bundle"] = str(args.data / "input_sources")
    (args.data / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"rows_by_protocol": metadata["rows_by_protocol"],
                      "unique_run_names": metadata["unique_run_names"], "snapshot_rows": len(snapshots)}))


if __name__ == "__main__":
    main()
