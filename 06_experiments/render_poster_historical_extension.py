"""Render saved historical spatial snapshots for the poster candidate gallery.

Online 100k training windows and frozen 10k policy probes remain separate
protocols. This adapter reuses the canonical spatial renderers and the poster
gallery's peak-map style; it never launches an environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study.spatial import render_graph_outcomes, render_occupancy_trajectory
from analyze_place_field_manifest import multilevel_field_structure, pairwise_map_cosine, peak_statistics
from prepare_poster_exemplar_gallery import mono_peak_panel, save, style
from render_poster_frozen import render_flow


ONLINE = {
    "N8_W_REF_STOP", "N8_W_REF_JOINT", "N8_SCR_ARR_DIRS",
    "CPD_C15_BASE", "CPD_C15_GATE_ACT_DIR_GOAL",
    "CPD_C15_GATE_ACT_DIR", "CPD_C15_ADD_CA3_DIR",
}
LEGACY_ANALYSIS = Path("/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis")
ACTIVE_HISTORICAL = Path("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/"
                         "train_dir/analysis/poster_missing_analyses_20260926/historical_extension")
FROZEN = {"CCR_C04_DIRECT_IMMEDIATE_ITER", "CCR_C05_DIRECT_IMMEDIATE_G001_R100",
          "CCR_C15_TOPOLOGY_UCB_DIRECT_O1"} | {name for name in ONLINE if name.startswith("CPD_")}
PLOT_LABELS = {
    "CCR_C04_DIRECT_IMMEDIATE_ITER": "C04 iterative",
    "CCR_C05_DIRECT_IMMEDIATE_G001_R100": "C05 direct",
    "CCR_C15_TOPOLOGY_UCB_DIRECT_O1": "C15 topology",
    "CPD_C15_ADD_CA3_DIR": "ADD CA3 DIR",
    "CPD_C15_BASE": "BASE",
    "CPD_C15_GATE_ACT_DIR": "GATE ACT DIR",
    "CPD_C15_GATE_ACT_DIR_GOAL": "GATE DIR GOAL",
    "N8_SCR_ARR_DIRS": "SCR ARR DIRS",
    "N8_W_REF_JOINT": "REF JOINT",
    "N8_W_REF_STOP": "REF STOP",
}


def selected_files(root: Path, protocol: str) -> list[Path]:
    files = sorted(root.rglob("*.npz"))
    chosen = []
    for path in files:
        if protocol == "online":
            if path.parent.name != "policy_00":
                continue
            with np.load(path, allow_pickle=False) as data:
                run = str(data["run_name"])
                if any(run == f"{condition}_S{seed}" for condition in ONLINE for seed in (8, 99, 123)):
                    chosen.append(path)
        else:
            with np.load(path, allow_pickle=False) as data:
                run = frozen_run_name(data)
                frame = int(Path(str(data["checkpoint"])).stem.rsplit("_", 1)[-1])
            required_frame = 75_038_720 if run.startswith("CPD_") else 100_040_704
            if frame != required_frame:
                continue
            if any(run == f"{condition}_S{seed}" for condition in FROZEN for seed in (8, 99, 123)):
                chosen.append(path)
    if protocol == "online" and len(chosen) != 21:
        raise ValueError(f"Expected 21 online files, found {len(chosen)}")
    if protocol == "frozen":
        ccr = sum("CCR_" in path.parent.name for path in chosen)
        cpd = len(chosen) - ccr
        if ccr != 9 or cpd not in (0, 12):
            raise ValueError(f"Expected 9 CCR and 0 or 12 CPD frozen files, found {ccr} and {cpd}")
    return chosen


def frozen_run_name(data: np.lib.npyio.NpzFile) -> str:
    for parent in Path(str(data["checkpoint"])).parents:
        if parent.name.startswith("00_"):
            return parent.name[3:]
    raise ValueError("Checkpoint path contains no run directory")


def plot_fields(maps: np.ndarray, occupancy: np.ndarray, information: np.ndarray,
                output: Path, title: str) -> None:
    style()
    valid = np.flatnonzero(np.isfinite(information) & (np.nanmax(maps, axis=(0, 1)) > 0))
    selected = valid[np.argsort(information[valid])[-4:][::-1]]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.8), constrained_layout=True)
    for ax, unit in zip(axes.flat, selected):
        image = np.ma.array(maps[:, :, unit].T, mask=(occupancy.T == 0))
        ax.imshow(image, origin="lower", cmap="viridis", interpolation="nearest",
                  extent=(0, 19, 0, 19), aspect="equal")
        ax.set(title=f"DG {unit} · SI {information[unit]:.2f} bit", xlabel="x bin", ylabel="y bin")
        ax.set_xticks((0, 9, 18))
        ax.set_yticks((0, 9, 18))
    fig.suptitle(title + " · top four active DG maps")
    save(fig, output)


def plot_mono(occupancy: np.ndarray, eligible: np.ndarray, mono: np.ndarray,
              peaks: np.ndarray, active: np.ndarray, output: Path, title: str) -> int:
    style()
    units = pd.DataFrame({"field_eligible": eligible, "mono_field": mono,
                          "peak_x_bin": peaks[:, 0], "peak_y_bin": peaks[:, 1],
                          "active_fraction": active})
    fig, ax = plt.subplots(figsize=(4.1, 4.5), constrained_layout=True)
    mono_peak_panel(ax, units, occupancy,
                    f"{title}\nMono-field DG: {mono.sum()}/{len(mono)}")
    save(fig, output)
    qualifying = units[units.field_eligible & units.mono_field & (units.active_fraction > 0)]
    return len(qualifying[["peak_x_bin", "peak_y_bin"]].drop_duplicates())


def pose_frame(pose: np.ndarray, segment_id: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": pose[:, 0], "y": pose[:, 1], "rot_y": pose[:, 2],
                         "agent": np.zeros(len(pose), dtype=np.int32),
                         "num_traj": segment_id})


def render_command_diagnostics(attempts: np.ndarray, successes: np.ndarray,
                               reliable: np.ndarray, output: Path, title: str) -> int:
    """Summarize each source DG command's attempted and successful transitions."""
    style()
    source_attempts = attempts.sum(axis=1)
    source_successes = successes.sum(axis=1)
    fractions = np.divide(source_successes, source_attempts,
                          out=np.full(len(source_attempts), np.nan), where=source_attempts > 0)
    table = pd.DataFrame({"source_dg_command": np.arange(len(source_attempts)),
                          "prospective_attempts": source_attempts,
                          "prospective_successes": source_successes,
                          "prospective_success_fraction": fractions,
                          "reliable_outgoing_edges": reliable.sum(axis=1)})
    table.to_csv(output.with_suffix(".csv"), index=False)
    fig, left = plt.subplots(figsize=(8.3, 4.2), constrained_layout=True)
    x = np.arange(len(source_attempts))
    left.bar(x, source_attempts, color="#9ecae1", label="Attempted transitions")
    left.set(xlabel="Source DG command", ylabel="Prospective attempts", xticks=x[::2],
             title=title + " · command-specific control")
    right = left.twinx()
    right.plot(x, fractions, color="#D55E00", marker="o", linewidth=1.3,
               label="Conditional success")
    right.set(ylabel="Success / attempts", ylim=(0, 1))
    save(fig, output.with_suffix(".svg"))
    return int((source_attempts > 0).sum())


def render_stored_command_diagnostics(attempts: np.ndarray, confidence: np.ndarray,
                                      output: Path, title: str) -> int:
    """Display checkpoint-stored source-command evidence without calling it success."""
    style()
    source_attempts = attempts.sum(axis=1)
    source_confidence = confidence.sum(axis=1)
    ratio = np.divide(source_confidence, source_attempts,
                      out=np.full(len(source_attempts), np.nan), where=source_attempts > 0)
    pd.DataFrame({"source_dg_command": np.arange(len(source_attempts)),
                  "stored_attempts": source_attempts,
                  "stored_confidence": source_confidence,
                  "stored_confidence_per_attempt": ratio}).to_csv(
                      output.with_suffix(".csv"), index=False)
    fig, left = plt.subplots(figsize=(8.3, 4.2), constrained_layout=True)
    x = np.arange(len(source_attempts))
    left.bar(x, source_attempts, color="#9ecae1")
    left.set(xlabel="Source DG command", ylabel="Stored attempts", xticks=x[::2],
             title=title + " · stored command evidence")
    right = left.twinx()
    right.plot(x, ratio, color="#D55E00", marker="o", linewidth=1.3)
    right.set(ylabel="Stored confidence / attempts", ylim=(0, 1))
    save(fig, output.with_suffix(".svg"))
    return int((source_attempts > 0).sum())


def render_goal_segments(source: Path, output: Path, title: str) -> dict[str, float]:
    """Summarize which frozen-policy commands were issued and for how long."""
    data = json.loads(source.read_text())
    frame = pd.DataFrame(data["segments"])
    if frame.empty or not {"goal", "decisions"}.issubset(frame):
        return {}
    grouped = frame.groupby("goal", as_index=False).agg(
        segments=("decisions", "size"), mean_decisions=("decisions", "mean"),
        timeout_reselections=("same_goal_timeouts", "sum"))
    grouped.to_csv(output.with_suffix(".csv"), index=False)
    style()
    fig, left = plt.subplots(figsize=(8.3, 4.2), constrained_layout=True)
    left.bar(grouped.goal, grouped.segments, color="#9ecae1")
    left.set(xlabel="Commanded DG goal", ylabel="Goal segments",
             xticks=np.arange(0, int(frame.goal.max()) + 1, 2),
             title=title + " · frozen-policy command allocation")
    right = left.twinx()
    right.plot(grouped.goal, grouped.mean_decisions, color="#D55E00", marker="o",
               linewidth=1.3)
    right.set_ylabel("Mean decisions per segment")
    save(fig, output.with_suffix(".svg"))
    return {"goal_segments_total": int(len(frame)),
            "goals_sampled": int(grouped.goal.nunique()),
            "mean_goal_segment_decisions": float(frame.decisions.mean()),
            "goal_timeout_reselections": int(data.get("timeout_reselections", 0))}


def process_online(path: Path, output: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        run = str(data["run_name"])
        condition, seed = run.rsplit("_S", 1)
        frames = int(data["actual_env_steps"])
        label = f"{run}_{frames}"
        directory = output / "per_run" / label
        directory.mkdir(parents=True, exist_ok=True)
        occupancy = data["occupancy"].copy()
        maps = data["rate_maps"].copy()
        info = data["spatial_information"].copy()
        active = data["active_fraction"].copy()
        eligible = data["field_eligible"].copy()
        mono = data["field_mono"].copy()
        peaks = data["field_dominant_peak_bin"].copy()
        pose = data["pose"].copy()
        segments = data["segment_id"].copy()
        graph = "control_attempts" in data.files
        graph_metrics = {}
        if graph:
            attempts = data["control_attempts"].copy()
            confidence = data["control_edge_confidence"].copy()
            prospective_attempts = data["control_prospective_attempts"].copy()
            prospective_success = data["control_prospective_successes"].copy()
            reliable = data["graph_reliable_adjacency"].copy()
            graph_metrics = {
                "graph_reachable_pair_fraction": float(data["graph_reachable_pair_fraction"]),
                "graph_reliable_edge_count": int(data["graph_reliable_edge_count"]),
                "prospective_attempts": float(prospective_attempts.sum()),
                "prospective_success_fraction": float(prospective_success.sum() / prospective_attempts.sum())
                if prospective_attempts.sum() > 0 else np.nan,
            }
        payload = {"pose": pose, "segment_id": segments, "dones": data["dones"].copy(),
                   "dg_activity": data["dg_activity"].copy(), "bounds": data["bounds"].copy(),
                   "grain": int(data["grain"]), "run_name": run,
                   "target_env_steps": int(data["target_env_steps"]), "actual_env_steps": frames}
    plot_fields(maps, occupancy, info, directory / "top_four_fields.svg", run)
    bins = plot_mono(occupancy, eligible, mono, peaks, active,
                     directory / "mono_field_peaks.svg", run)
    render_occupancy_trajectory(payload, directory / "occupancy_trajectory", title=run)
    frame = pose_frame(pose, segments)
    cells = render_flow(frame, directory / "occupancy_flow.png", run)
    cells.to_csv(directory / "occupancy_flow_cells.csv", index=False)
    if graph:
        render_graph_outcomes(attempts, np.minimum(confidence, attempts),
                              directory / "stored_graph", title=run,
                              ratio_label="Stored edge confidence / attempts")
        render_graph_outcomes(prospective_attempts, prospective_success,
                              directory / "prospective_graph", title=run,
                              ratio_label="Prospective hits / attempts")
        graph_metrics["command_nodes_with_prospective_attempts"] = render_command_diagnostics(
            prospective_attempts, prospective_success, reliable,
            directory / "command_diagnostics", run)
    return {"condition": condition, "seed": int(seed), "run_name": run,
            "protocol": "online_100k_training_window", "frames": frames,
            "raw_source": str(LEGACY_ANALYSIS / "online_spatial" /
                              path.relative_to(Path("/tmp/intrmotiv_online_npz"))),
            "dg_units": len(active), "active_units": int((active > 0).sum()),
            "eligible_units": int(eligible.sum()), "mono_units": int(mono.sum()),
            "mono_fraction_all_dg": float(mono.sum() / len(mono)),
            "mono_peak_bins": bins, "visited_fraction": float((occupancy > 0).sum() / occupancy.size),
            "active_mean_si_bits": float(info[active > 0].mean()),
            "active_unique_peak_bins": peak_statistics(maps, np.flatnonzero(active > 0),
                                                        require_positive=True)[0],
            "active_map_cosine": pairwise_map_cosine(maps, occupancy, np.flatnonzero(active > 0)),
            "flow_cells_at_least_5": int((cells.transitions >= 5).sum()),
            "mean_flow_coherence": float(np.average(cells.coherence, weights=cells.transitions)),
            "graph_available": graph, **graph_metrics, "figure_dir": str(directory)}


def process_frozen(path: Path, output: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        run = frozen_run_name(data)
        occupancy = data["occupancy"].copy()
        maps = data["rate_maps"].copy()
        info = data["spatial_information"].copy()
        active = data["active_fraction"].copy()
        checkpoint = str(data["checkpoint"])
        if "raw_dg_field_mono" in data.files:
            eligible = data["raw_dg_field_eligible"].copy()
            mono = data["raw_dg_field_mono"].copy()
            peaks = data["raw_dg_field_dominant_peak_bin"].copy()
        else:
            eligible, _, _, _, mono = multilevel_field_structure(maps, occupancy, active)
            peaks = np.asarray([np.unravel_index(np.nanargmax(maps[:, :, unit]), occupancy.shape)
                                for unit in range(len(active))])
        graph = "control_attempts" in data.files
        if graph:
            attempts = data["control_attempts"].copy()
            confidence = data["control_edge_confidence"].copy()
    condition, seed = run.rsplit("_S", 1)
    frames = int(Path(checkpoint).stem.rsplit("_", 1)[-1])
    directory = output / "per_run" / f"{run}_{frames}"
    directory.mkdir(parents=True, exist_ok=True)
    plot_fields(maps, occupancy, info, directory / "top_four_fields.svg", run)
    bins = plot_mono(occupancy, eligible, mono, peaks, active,
                     directory / "mono_field_peaks.svg", run)
    pose_csv = path.parent / "pose.csv"
    pose = pd.read_csv(pose_csv)
    segment = (pose.agent.astype(np.int64) * 1_000_000 + pose.num_traj.astype(np.int64)).to_numpy()
    payload = {"pose": pose[["x", "y", "rot_y"]].to_numpy(dtype=np.float32),
               "segment_id": segment, "dg_activity": np.zeros((len(pose), 1), dtype=np.float32),
               "bounds": np.array([100, 2000, 100, 2000]), "grain": 19,
               "run_name": run, "target_env_steps": frames, "actual_env_steps": frames}
    render_occupancy_trajectory(payload, directory / "occupancy_trajectory", title=run)
    cells = render_flow(pose, directory / "occupancy_flow.png", run)
    cells.to_csv(directory / "occupancy_flow_cells.csv", index=False)
    goal_file = path.parent / "goal_behavior_diagnostics.json"
    goal_metrics = render_goal_segments(goal_file, directory / "goal_segments", run) if goal_file.is_file() else {}
    if graph:
        render_graph_outcomes(attempts, np.minimum(confidence, attempts),
                              directory / "stored_graph", title=run,
                              ratio_label="Stored edge confidence / attempts")
        command_nodes = render_stored_command_diagnostics(
            attempts, confidence, directory / "stored_command_diagnostics", run)
    else:
        command_nodes = 0
    return {"condition": condition, "seed": int(seed), "run_name": run,
            "protocol": "frozen_10k_policy_probe", "frames": frames,
            "checkpoint": checkpoint,
            "raw_source": str((ACTIVE_HISTORICAL / "frozen_cpd/raw" if run.startswith("CPD_")
                               else LEGACY_ANALYSIS / "corrected_core_candidates_20260902_place_fields/raw") /
                              path.parent.name / "place_fields.npz"),
            "dg_units": len(active), "active_units": int((active > 0).sum()),
            "eligible_units": int(eligible.sum()), "mono_units": int(mono.sum()),
            "mono_fraction_all_dg": float(mono.sum() / len(mono)),
            "mono_peak_bins": bins, "visited_fraction": float((occupancy > 0).sum() / occupancy.size),
            "active_mean_si_bits": float(info[active > 0].mean()),
            "active_unique_peak_bins": peak_statistics(maps, np.flatnonzero(active > 0),
                                                        require_positive=True)[0],
            "active_map_cosine": pairwise_map_cosine(maps, occupancy, np.flatnonzero(active > 0)),
            "flow_cells_at_least_5": int((cells.transitions >= 5).sum()),
            "mean_flow_coherence": float(np.average(cells.coherence, weights=cells.transitions)),
            "graph_available": graph,
            "stored_attempted_edges": int((attempts > 0).sum()) if graph else np.nan,
            "stored_confidence_per_attempt": float(confidence.sum() / attempts.sum())
            if graph and attempts.sum() else np.nan,
            "command_nodes_with_stored_attempts": command_nodes,
            **goal_metrics,
            "figure_dir": str(directory)}


def peak_rows(paths: list[Path], protocol: str) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if protocol.startswith("online"):
                run = str(data["run_name"])
                frame = int(data["actual_env_steps"])
                active = data["active_fraction"]
                eligible = data["field_eligible"]
                mono = data["field_mono"]
                peaks = data["field_dominant_peak_bin"]
                information = data["spatial_information"]
            else:
                run = frozen_run_name(data)
                frame = int(Path(str(data["checkpoint"])).stem.rsplit("_", 1)[-1])
                active = data["active_fraction"]
                information = data["spatial_information"]
                if "raw_dg_field_mono" in data.files:
                    eligible = data["raw_dg_field_eligible"]
                    mono = data["raw_dg_field_mono"]
                    peaks = data["raw_dg_field_dominant_peak_bin"]
                else:
                    eligible, _, _, _, mono = multilevel_field_structure(
                        data["rate_maps"], data["occupancy"], active)
                    peaks = np.asarray([np.unravel_index(np.nanargmax(data["rate_maps"][:, :, unit]),
                                                        data["occupancy"].shape)
                                        for unit in range(len(active))])
            condition, seed = run.rsplit("_S", 1)
            for unit, (xbin, ybin) in enumerate(peaks):
                rows.append({"protocol": protocol, "condition": condition, "seed": int(seed),
                             "run_name": run, "frames": frame, "unit": unit,
                             "active_fraction": float(active[unit]),
                             "field_eligible": bool(eligible[unit]),
                             "mono_field": bool(mono[unit]),
                             "peak_x_bin": int(xbin), "peak_y_bin": int(ybin),
                             "spatial_information_bits": float(information[unit])})
    return rows


def render_aggregate(table: pd.DataFrame, output: Path) -> None:
    style()
    table = table.copy()
    table["family"] = table.condition.map(lambda name: name.split("_", 1)[0])
    for (protocol, family), subset in table.groupby(["protocol", "family"]):
        conditions = sorted(subset.condition.unique())
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(conditions) * 1.4), 4.6),
                                 constrained_layout=True)
        positions = np.arange(len(conditions))
        for axis, (metric, title) in zip(axes, (
            ("mono_fraction_all_dg", "Mono-field DG / all DG"),
            ("mono_peak_bins", "Distinct mono-field peak bins"))):
            means = [subset.loc[subset.condition == condition, metric].mean() for condition in conditions]
            axis.bar(positions, means, color="#9ecae1", width=.62, edgecolor="#4a6680")
            for index, condition in enumerate(conditions):
                values = subset.loc[subset.condition == condition, metric].to_numpy()
                axis.scatter(np.full(len(values), index) + np.linspace(-.12, .12, len(values)), values,
                             color="#0072B2", edgecolor="white", linewidth=.5, zorder=3)
            axis.set(xticks=positions, xticklabels=[PLOT_LABELS[name] for name in conditions],
                     title=title)
            axis.tick_params(axis="x", labelrotation=35)
            if metric == "mono_fraction_all_dg":
                axis.set_ylim(0, max(.55, max(means) * 1.3))
            else:
                axis.set_ylim(bottom=0)
        fig.suptitle(f"{family} · {protocol.replace('_', ' ')} · each point is one seed")
        suffix = "frozen" if protocol.startswith("frozen") else "online"
        save(fig, output / f"aggregate_{family.lower()}_{suffix}.svg")


def render_spatial_graph_scatter(table: pd.DataFrame, output: Path) -> None:
    """Show seed variation in online mono-field fraction and graph reach."""
    subset = table[table.protocol.str.startswith("online") &
                   table.graph_reachable_pair_fraction.notna()]
    style()
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, .8, subset.condition.nunique()))
    for color, (condition, group) in zip(colors, subset.groupby("condition")):
        ax.scatter(group.mono_fraction_all_dg, group.graph_reachable_pair_fraction,
                   s=75, color=color, edgecolor="white", linewidth=.8,
                   label=PLOT_LABELS[condition])
        for row in group.itertuples():
            if row.mono_fraction_all_dg < .3 and row.graph_reachable_pair_fraction < .35:
                continue
            ax.annotate(f"S{row.seed}", (row.mono_fraction_all_dg,
                                          row.graph_reachable_pair_fraction),
                        xytext=(4, 4), textcoords="offset points", fontsize=12)
    ax.set(xlabel="Mono-field DG / all DG", ylabel="Reachable ordered node pairs",
           xlim=(-.025, .55), ylim=(-.04, 1.04),
           title="Historical online 75M window · each point is one seed")
    ax.grid(alpha=.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    save(fig, output / "historical_mono_graph_scatter.svg")


def render_kernel_bands(path: Path, output: Path) -> None:
    """Plot three-seed full-range correlations from the shared kernel summary."""
    rows = pd.read_csv(path)
    rows = rows[(rows.scope == "all_cues") & (rows.distance_band == "long_13_18")]
    style()
    for family in ("CPD", "CCR"):
        subset = rows[rows.condition.str.startswith(family + "_")]
        conditions = sorted(subset.condition.unique())
        fig, axes = plt.subplots(1, 3, figsize=(max(9.3, len(conditions) * 2.6), 4.5),
                                 constrained_layout=True)
        for ax, (layer, label) in zip(axes, (("dg", "DG"), ("ca3", "CA3"),
                                            ("decoder_1", "Decoder-1"))):
            layer_rows = subset[subset.layer == layer]
            seed_colors = {8: "#0072B2", 99: "#D55E00", 123: "#009E73"}
            for seed, seed_rows in layer_rows.groupby("seed"):
                values = seed_rows.set_index("condition").reindex(conditions).mean_correlation
                color = seed_colors[int(seed)]
                ax.plot(range(len(conditions)), values, color=color, lw=1.3,
                        alpha=.8, zorder=1, label=f"Seed {seed}")
                ax.scatter(range(len(conditions)), values, color=color, s=38,
                           edgecolor="white", linewidth=.5, zorder=2)
            ax.set(xticks=range(len(conditions)),
                   xticklabels=[PLOT_LABELS[name] for name in conditions],
                   title=label, ylim=(-.04, 1.02))
            ax.tick_params(axis="x", labelrotation=35)
            ax.grid(axis="y", alpha=.2)
            if ax is axes[0]:
                ax.set_ylabel("Population correlation, 13–18 bins")
                ax.legend(frameon=False, loc="upper right", fontsize=12)
        fig.suptitle(f"{family} · exact shared checkpoint · common history · paired seeds")
        save(fig, output / f"kernel_long_{family.lower()}.svg")


def render_corrected_core_dynamics(root: Path, output: Path) -> None:
    """Show older C05/C15 seed-99 probes only as learning dynamics."""
    rows = []
    for path in root.rglob("place_fields.npz"):
        with np.load(path, allow_pickle=False) as data:
            run = frozen_run_name(data)
            if run not in ("CCR_C05_DIRECT_IMMEDIATE_G001_R100_S99",
                           "CCR_C15_TOPOLOGY_UCB_DIRECT_O1_S99"):
                continue
            frame = int(Path(str(data["checkpoint"])).stem.rsplit("_", 1)[-1])
            eligible, _, _, _, mono = multilevel_field_structure(
                data["rate_maps"], data["occupancy"], data["active_fraction"])
            rows.append({"run_name": run, "checkpoint_frames": frame,
                         "mono_units": int(mono.sum()), "dg_units": len(mono),
                         "eligible_units": int(eligible.sum()),
                         "visited_fraction": float((data["occupancy"] > 0).sum() /
                                                   data["occupancy"].size)})
    table = pd.DataFrame(rows).sort_values(["run_name", "checkpoint_frames"])
    if len(table) < 4:
        return
    table.to_csv(output / "c05_c15_seed99_learning_dynamics.csv", index=False)
    style()
    fig, ax = plt.subplots(figsize=(6.7, 4.3), constrained_layout=True)
    for name, group in table.groupby("run_name"):
        label = "C05 direct" if "C05" in name else "C15 topology"
        ax.plot(group.checkpoint_frames / 1e6, group.mono_units / group.dg_units,
                marker="o", label=label, linewidth=1.3)
    ax.set(xlabel="Checkpoint age (million frames)",
           ylabel="Mono-field DG / all DG", ylim=(-.008, .16),
           title="C05/C15 seed 99 · archived frozen probes · dynamics only")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, output / "c05_c15_seed99_learning_dynamics.svg")


def write_run_index(table: pd.DataFrame, output: Path) -> None:
    lines = ["# Historical poster candidate run index", "",
             "Every row links the saved-map and behavior analyses for one seed. "
             "Online rows use the retained 100k-sample training window ending at "
             "75,005,952 frames; frozen rows use the exact shared 75,038,720- or "
             "100,040,704-frame checkpoint and a 10k-decision policy probe. "
             "These protocols are separate.", "",
             "| Run | Mono DG / all | Peak bins | Coverage | Spatial score | Flow coherence | Analyses |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in table.itertuples():
        folder = f"per_run/{row.run_name}_{row.frames}"
        links = [f"[fields]({folder}/top_four_fields.svg)",
                 f"[mono peaks]({folder}/mono_field_peaks.svg)",
                 f"[trajectory]({folder}/occupancy_trajectory.png)",
                 f"[flow]({folder}/occupancy_flow.png)"]
        if row.graph_available:
            links.append(f"[stored graph]({folder}/stored_graph.png)")
            if row.protocol.startswith("online"):
                links.extend([f"[prospective graph]({folder}/prospective_graph.png)",
                              f"[commands]({folder}/command_diagnostics.svg)"])
            else:
                links.append(f"[stored commands]({folder}/stored_command_diagnostics.svg)")
        if pd.notna(getattr(row, "goal_segments_total", np.nan)):
            links.append(f"[issued goals]({folder}/goal_segments.svg)")
        kernel_frame = 75_038_720 if row.condition.startswith("CPD_") else 100_040_704
        if row.condition.startswith(("CPD_", "CCR_")):
            kernel_folder = f"per_run/{row.run_name}_{kernel_frame}"
            links.extend([f"[terminal layer kernel]({kernel_folder}/layerwise_kernels.svg)",
                          f"[radial profile]({kernel_folder}/layerwise_radial.svg)"])
        lines.append(f"| {row.run_name} ({row.frames:,}) | {row.mono_units}/{row.dg_units} "
                     f"({row.mono_fraction_all_dg:.1%}) | {row.mono_peak_bins} | "
                     f"{row.visited_fraction:.1%} | {row.active_mean_si_bits:.3f} | "
                     f"{row.mean_flow_coherence:.3f} | {', '.join(links)} |")
    (output / "run_index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-bands", type=Path,
                        help="Full-range per-run band CSV from the common-history replay")
    parser.add_argument("--stored-graph-summary", type=Path,
                        help="Canonical graph summary for frozen exact-checkpoint CPD runs")
    args = parser.parse_args()
    style()
    args.output.mkdir(parents=True, exist_ok=True)
    online_paths = selected_files(args.online_root, "online")
    frozen_paths = selected_files(args.frozen_root, "frozen")
    rows = [process_online(path, args.output) for path in online_paths]
    rows += [process_frozen(path, args.output) for path in frozen_paths]
    table = pd.DataFrame(rows).sort_values(["protocol", "condition", "seed"])
    if args.stored_graph_summary:
        graph = pd.read_csv(args.stored_graph_summary)
        metrics = [column for column in graph.columns if column.startswith("graph_")]
        graph = graph[["condition", "seed", "checkpoint_frames", *metrics]].rename(
            columns={column: f"stored_{column}" for column in metrics})
        table = table.merge(graph, how="left", left_on=["condition", "seed", "frames"],
                            right_on=["condition", "seed", "checkpoint_frames"],
                            validate="many_to_one").drop(columns="checkpoint_frames")
    table.to_csv(args.output / "historical_exemplar_inventory.csv", index=False)
    pd.DataFrame(peak_rows(online_paths, "online_100k_training_window") +
                 peak_rows(frozen_paths, "frozen_10k_policy_probe")).to_csv(
                     args.output / "historical_peak_positions.csv", index=False)
    render_aggregate(table, args.output)
    render_spatial_graph_scatter(table, args.output)
    if args.kernel_bands:
        render_kernel_bands(args.kernel_bands, args.output)
    render_corrected_core_dynamics(args.frozen_root, args.output)
    write_run_index(table, args.output)
    # The shared spatial renderer also emits PDFs; keep one compact PNG/SVG
    # gallery copy per panel in the vault and leave bulk source arrays remote.
    for pdf in args.output.rglob("*.pdf"):
        pdf.unlink()
    print(f"Rendered {len(table)} historical candidate runs")


if __name__ == "__main__":
    main()
