"""Plot completed first-hit aggregates without loading a policy or environment.

Identification bounds retain missing/censored outcomes. Source-history examples
are selected from issuance metadata, never successes. This is a thin consumer
of aggregate_evaluation and trace_analysis artifacts, not another evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import textwrap

import numpy as np

from hpc_runs.intrmotiv_study.spatial import _figure_runtime
from .aggregate_evaluation import POLICIES, file_hash


LABELS = {"source": "Original actor", "first_hit": "First-hit actor"}
COLORS = {"source": "#0072B2", "first_hit": "#D55E00"}


def _boolean(value):
    if value in (True, "True", "true"):
        return True
    if value in (False, "False", "false"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def _interval(lower, upper, limits):
    if lower is None or upper is None or lower == "" or upper == "":
        raise ValueError("A nontrivial declared outcome needs identification bounds")
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or not limits[0] <= lower <= upper <= limits[1]:
        raise ValueError("Invalid identification interval")
    return [lower, upper]


def goal_mapping(aggregate):
    """Keep every capacity ID and copy exact precomputed identification bounds."""
    goals = aggregate["identities"]["goal_vocabulary"]
    if goals != list(range(len(goals))) or len(goals) < 2:
        raise ValueError("Require a complete indexed DG vocabulary")
    result = []
    for goal in goals:
        for policy in POLICIES:
            arrival = aggregate["outcomes"][policy]["per_goal"][str(goal)]
            first = aggregate["competing_target_FIRST"][policy]["per_goal"][str(goal)]
            result.append(dict(goal=goal, policy=policy,
                observed=goal in aggregate["identities"]["observed_targets"],
                arrival=_interval(arrival["success_lower_bound_declared"], arrival["success_upper_bound_declared"], (0, 1)),
                arrival_declared=arrival["declared_nontrivial_own_rows"],
                arrival_missing=arrival["missing_nontrivial_own_rows"],
                arrival_censored=arrival["censored_nontrivial_own_rows"],
                FIRST=_interval(first["mean_lower_bound"], first["mean_upper_bound"], (-1, 1)),
                FIRST_declared=first["declared_comparisons"], FIRST_identified=first["identified_comparisons"],
                FIRST_bootstrap=first.get("source_cluster_bootstrap")))
    return result


def matrix_mapping(rows, goals):
    """Rows are sources, columns are commands; diagonal is excluded, not zero."""
    n = len(goals)
    if goals != list(range(n)):
        raise ValueError("Indexed complete goal vocabulary required")
    matrices = {p: {name: np.zeros((n, n), dtype=int if name in ("missing", "censored", "histories") else float)
                   for name in ("lower", "upper", "missing", "censored", "histories")} for p in POLICIES}
    seen = set()
    for row in rows:
        source, goal = int(row["source"]), int(row["goal"])
        if (source, goal) in seen or source not in goals or goal not in goals:
            raise ValueError("Duplicate or undeclared source-target cell")
        seen.add((source, goal))
        if _boolean(row["initially_satisfied"]) != (source == goal):
            raise ValueError("Initial-goal diagonal disagrees")
        for policy in POLICIES:
            values = matrices[policy]
            values["histories"][source, goal] = int(row["declared_histories"])
            for name in ("missing", "censored"):
                values[name][source, goal] = int(row[f"{policy}_{name}"])
            if source == goal:
                values["lower"][source, goal] = values["upper"][source, goal] = np.nan
            else:
                lower, upper = _interval(row[f"{policy}_success_lower"], row[f"{policy}_success_upper"], (0, 1))
                values["lower"][source, goal], values["upper"][source, goal] = lower, upper
                count = values["histories"][source, goal]
                unknown = values["missing"][source, goal] + values["censored"][source, goal]
                if count <= 0 or unknown < 0 or unknown > count or not np.isclose(upper - lower, unknown / count):
                    raise ValueError("Cell bounds do not reconcile with missing/censored histories")
    if seen != {(s, g) for s in goals for g in goals}:
        raise ValueError("Full capacity source-target matrix required")
    return matrices


def select_trajectories(per_trial, goals):
    """Smallest issued source at repeat zero, then four evenly spaced IDs.

    Selection reads no outcome field. Unobserved or unissued chosen commands
    remain selected and will be drawn as missing; there is no fallback.
    """
    sources = sorted({int(row["source"]) for row in per_trial
                      if int(row["repeat"]) == 0 and _boolean(row["issued"])})
    if not sources:
        return dict(source=None, repeat=0, commands=[], reason="No issued source at repeat zero")
    source = sources[0]
    candidates = [g for g in goals if g != source]
    positions = np.rint(np.linspace(0, len(candidates) - 1, min(4, len(candidates)))).astype(int)
    return dict(source=source, repeat=0, commands=[candidates[i] for i in positions],
                rule="Minimum issued source at repeat0; round(linspace(0,N-1,min(4,N))) over sorted noninitial capacity IDs; no outcome selection")


def stream_selected_trials(receipts, selection):
    """Retain only selected own-command rows while streaming certified JSONL."""
    wanted = {(p, selection["source"], 0, g) for p in POLICIES for g in selection["commands"]}
    selected, hashes = {}, {}
    if not wanted:
        return selected, hashes
    for receipt in receipts:
        path = Path(receipt["directory"]) / "trials.jsonl"
        if file_hash(path) != receipt["trials_sha256"]:
            raise ValueError("Trial bytes differ from completed aggregation")
        hashes[str(path.resolve())] = receipt["trials_sha256"]
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["policy"], row["source"], row["repeat"], row["command"])
                if row["commanded"] and key in wanted:
                    if key in selected:
                        raise ValueError("Duplicate selected own-command trace")
                    selected[key] = row
    return selected, hashes


def trajectory_mapping(row, *, policy, command, bounds, jump_distance):
    """Map an already certified path, keeping an unknown final endpoint unknown."""
    if row is None or not row["issued"]:
        return dict(policy=policy, command=command, issued=False,
                    reason="Unassessed command identity" if row is None else row.get("missing_reason", "Unissued"))
    trace = row.get("trajectory")
    if not trace:
        raise ValueError("Issued selected path has no observation trace")
    xy = np.asarray([item["position"][:2] for item in trace], dtype=float)
    if xy.shape != (len(trace), 2) or not np.isfinite(xy).all():
        raise ValueError("Selected path has invalid physical coordinates")
    elapsed = int(row["elapsed"])
    if len(trace) != elapsed + 1 - int(row["physical_done"]):
        raise ValueError("Selected path contains a missing or autoreset endpoint")
    if row["trajectory_end_observed"] != (not row["physical_done"]):
        raise ValueError("Selected endpoint certification disagrees")
    breaks = np.flatnonzero(np.linalg.norm(np.diff(xy, axis=0), axis=1) > jump_distance) + 1
    segments = [part.tolist() for part in np.split(xy, breaks) if len(part)]
    outside = ((xy[:, 0] < bounds["x_min"]) | (xy[:, 0] > bounds["x_max"]) |
               (xy[:, 1] < bounds["y_min"]) | (xy[:, 1] > bounds["y_max"]))
    return dict(policy=policy, command=command, issued=True, trajectory_id=row["trajectory_id"],
                elapsed=elapsed, hit=bool(row["hit"]), hit_time=row["hit_time"],
                endpoint_observed=bool(row["trajectory_end_observed"]),
                segments=segments, initial=xy[0].tolist(), last_observed=xy[-1].tolist(),
                out_of_bounds_samples=int(outside.sum()), omitted_unknown_terminal_steps=int(row["physical_done"]))


def _runtime():
    plt = _figure_runtime()
    from matplotlib import font_manager
    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    # At 100 dpi this is at least 41.7 displayed pixels on a 1000px canvas:
    # about 13.6 physical points on the user's 220-PPI report display.
    plt.rcParams.update({"font.size": 30, "axes.titlesize": 30, "axes.labelsize": 30,
                         "xtick.labelsize": 30, "ytick.labelsize": 30, "legend.fontsize": 30,
                         "svg.fonttype": "none"})
    return plt, font


def _save(fig, output, stem, plt):
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"{stem}.{extension}"
        # Fixed ten-inch canvas at 100 dpi: 1000px wide, without tight-bbox rescaling.
        fig.savefig(path, dpi=100, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_goal_intervals(mapping, goals, output, metric, plt):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 16), layout="constrained")
    for row in mapping:
        lower, upper = row[metric]
        offset = -.16 if row["policy"] == "source" else .16
        y, color = row["goal"] + offset, COLORS[row["policy"]]
        ax.plot([lower, upper], [y, y], color=color, linewidth=3,
                linestyle="-" if row["policy"] == "source" else "--")
        ax.plot([lower, upper], [y, y], "|", color=color, markersize=11, markeredgewidth=1.5)
        if lower == upper:
            ax.plot(lower, y, "o" if row["policy"] == "source" else "s", color=color, markersize=7)
    ax.set_yticks(goals, [str(g) for g in goals])
    ax.set_ylim(len(goals) - .4, -.6)
    ax.set_ylabel("Commanded DG identity")
    if metric == "arrival":
        ax.set_xlim(-.025, 1.025)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel("Arrival fraction\nIdentification bounds")
        ax.set_title("All declared nontrivial queries")
    else:
        ax.set_xlim(-1.025, 1.025)
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.axvline(0, color=".55", linewidth=1, zorder=0)
        ax.set_xlabel("Competing-target FIRST\nIdentification bounds")
        ax.set_title("Command-dependent ordering on matched starts")
    ax.grid(axis="x", color=".88", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=COLORS[p], linewidth=3,
                       linestyle="-" if p == "source" else "--", label=LABELS[p]) for p in POLICIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2, frameon=False)
    return _save(fig, output, "per_goal_success" if metric == "arrival" else "per_goal_FIRST", plt)


def plot_matrices(matrices, goals, output, plt):
    from matplotlib.patches import Patch, Rectangle
    fig, axes = plt.subplots(2, 1, figsize=(10, 20))
    fig.subplots_adjust(left=.18, right=.96, top=.90, bottom=.14, hspace=.42)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    ticks = np.unique(np.rint(np.linspace(0, len(goals) - 1, min(3, len(goals)))).astype(int))
    for ax, policy in zip(axes, POLICIES):
        matrix = matrices[policy]
        image = ax.imshow(np.ma.masked_invalid(matrix["lower"]), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        for source, goal in np.argwhere(matrix["upper"] > matrix["lower"]):
            ax.add_patch(Rectangle((goal - .5, source - .5), 1, 1, facecolor="none",
                                   edgecolor="#f0f0f0", linewidth=0, hatch="xxx"))
        ax.set(title=LABELS[policy], xlabel="Commanded DG identity", ylabel="Source DG identity")
        ax.set_xticks(ticks, [str(goals[t]) for t in ticks])
        ax.set_yticks(ticks, [str(goals[t]) for t in ticks])
    fig.colorbar(image, ax=axes, orientation="horizontal", fraction=.04, pad=.10,
                 ticks=[0, .5, 1], label="Arrival lower bound")
    fig.legend(handles=[Patch(facecolor="#777777", edgecolor="white", hatch="xxx", label="Unresolved histories"),
                        Patch(facecolor="#d9d9d9", label="Initially at goal")],
               loc="upper center", ncol=1, frameon=False)
    return _save(fig, output, "source_target_success", plt)


def plot_trajectory_pair(pair, selection, command, bounds, output, plt):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 1, figsize=(10, 21))
    fig.subplots_adjust(left=.20, right=.96, top=.88, bottom=.15, hspace=.48)
    for ax, policy in zip(axes, POLICIES):
        item = pair[policy]
        ax.set(xlim=(bounds["x_min"], bounds["x_max"]), ylim=(bounds["y_min"], bounds["y_max"]),
               xlabel="World x (DMLab units)", ylabel="World y (DMLab units)", title=LABELS[policy])
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(bounds["x_min"], bounds["x_max"], 3))
        ax.set_yticks(np.linspace(bounds["y_min"], bounds["y_max"], 3))
        if not item["issued"]:
            ax.set_title(f"{LABELS[policy]}\nNot issued", fontsize=30)
            ax.text(.5, .5, textwrap.fill(str(item["reason"]).replace("_", " "), width=22),
                    ha="center", va="center", transform=ax.transAxes, fontsize=30, wrap=True)
            continue
        for part in item["segments"]:
            xy = np.asarray(part)
            ax.plot(xy[:, 0], xy[:, 1], color=COLORS[policy], linewidth=1.6)
        ax.plot(*item["initial"], "o", color="black", markersize=8)
        marker = "*" if item["hit"] else "s" if item["endpoint_observed"] else "^"
        ax.plot(*item["last_observed"], marker, markersize=12, markeredgecolor="black",
                markerfacecolor=COLORS[policy] if item["hit"] else "white")
        status = "Arrival" if item["hit"] else "No arrival by stop" if item["endpoint_observed"] else "Terminal unknown"
        ax.set_title(f"{LABELS[policy]}\n{status}; {item['elapsed']} decisions", fontsize=30)
    fig.suptitle(f"Source {selection['source']}, history 0\nCommand {command}", fontsize=30)
    handles = [Line2D([], [], marker=m, linestyle="none", color="black", markerfacecolor="black" if m == "o" else "white",
                       markersize=9, label=label) for m, label in (("o", "Start"), ("*", "Arrival"),
                       ("s", "Observed stop"), ("^", "Last observed"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    return _save(fig, output, f"trajectory_source_{selection['source']}_command_{command}", plt)


def _read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def render(aggregate_path, trace_path, output):
    """Render only hash-linked, completed analysis outputs into a fresh folder."""
    aggregate_path, trace_path, output = [Path(p).resolve() for p in (aggregate_path, trace_path, output)]
    if output.exists():
        raise FileExistsError("Preserve earlier figure exports")
    from hpc_runs.intrmotiv_study import spatial
    sources = {str(p): file_hash(p) for p in (Path(__file__).resolve(), Path(spatial.__file__).resolve())}
    aggregate, traces = json.loads(aggregate_path.read_text()), json.loads(trace_path.read_text())
    if (aggregate.get("schema") != "intrmotiv/first-hit-evaluation-aggregate/v1"
            or aggregate.get("status") != "validated_completed_shards"
            or traces.get("schema") != "intrmotiv/first-hit-trace-analysis/v1"):
        raise ValueError("Require completed canonical aggregate and trace analysis")
    if file_hash(aggregate_path) not in traces["input_hashes"].values():
        raise ValueError("Trace analysis is not bound to this aggregate")
    hashes = {str(p): file_hash(p) for p in (aggregate_path, trace_path)}
    for parent, receipt, name in ((aggregate_path.parent, aggregate, "source_target.csv"),
                                  (trace_path.parent, traces, "per_trial.csv")):
        path = parent / name
        if file_hash(path) != receipt["outputs"][name]:
            raise ValueError(f"Changed completed analysis table: {name}")
        hashes[str(path)] = receipt["outputs"][name]
    goals = aggregate["identities"]["goal_vocabulary"]
    per_goal = goal_mapping(aggregate)
    matrices = matrix_mapping(_read_csv(aggregate_path.parent / "source_target.csv"), goals)
    selection = select_trajectories(_read_csv(trace_path.parent / "per_trial.csv"), goals)
    selected, trial_hashes = stream_selected_trials(aggregate["input_receipts"], selection)
    hashes.update(trial_hashes)
    bounds, pairs = traces["bounds"], {}
    for command in selection["commands"]:
        if command in aggregate["identities"]["observed_targets"] and any(
                (p, selection["source"], 0, command) not in selected for p in POLICIES):
            raise ValueError("A declared selected command row is absent from certified traces")
        pairs[command] = {p: trajectory_mapping(selected.get((p, selection["source"], 0, command)),
            policy=p, command=command, bounds=bounds,
            jump_distance=traces["trajectory_segmentation"]["maximum_contiguous_jump"]) for p in POLICIES}
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    plt, font = _runtime()
    outputs = plot_goal_intervals(per_goal, goals, output, "arrival", plt)
    outputs += plot_goal_intervals(per_goal, goals, output, "FIRST", plt)
    outputs += plot_matrices(matrices, goals, output, plt)
    for command, pair in pairs.items():
        outputs += plot_trajectory_pair(pair, selection, command, bounds, output, plt)
    # NaN means the excluded initial-goal diagonal; JSON carries explicit null.
    matrix_json = {p: {k: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]
                       for k, a in matrix.items()} for p, matrix in matrices.items()}
    import matplotlib
    mapping = dict(schema="intrmotiv/first-hit-evaluation-figures/v1", inputs=hashes, sources=sources,
        font=dict(path=str(font), sha256=file_hash(font), minimum_points=30, minimum_display_pixels=41.7,
                  canvas_width_px=1000, intended_display_ppi=220),
        matplotlib_version=matplotlib.__version__, per_goal=per_goal, matrices=matrix_json,
        evaluation_config=aggregate["identities"].get("config", {}),
        trajectory_selection=selection, trajectories=pairs,
        statistical_units="One frozen donor and one fitted model; source-history queries and ordered competing target pairs, not independent training seeds.",
        captions={
            "per_goal_success": "Segments are identification bounds over every declared nontrivial source-history query for each command. Missing and unresolved terminal outcomes may lie anywhere in their bounds. A point appears only when the interval is identified. These are not confidence intervals.",
            "per_goal_FIRST": "Segments bound the same competing-target FIRST event under command g minus command c, averaged over all declared competitors and matched starts. Third landmarks are allowed. These are identification bounds, not confidence intervals. Zero does not rule out reaching when an intermediate target is unavoidable; no every-pair-positive requirement applies.",
            "source_target_success": "Color is confirmed arrivals divided by all declared histories, fixed scale0–1. Hatching marks an interval with unresolved missing/censored histories; the mapped upper bound and separate counts remain in this JSON. Gray diagonal is initially satisfied and excluded, not failed. Cells are all capacity source-command combinations.",
            "trajectories": "Examples use the smallest issued source at repeat0 and four evenly spaced noninitial capacity command IDs, selected without outcomes. Both actors retain the same selected queries, including missing trials. Physical axes are fixed to the canonical environment bounds. Lines connect observed positions only within the existing jump segmentation; triangles mark a last observed position before an unknown terminal successor. Pose is telemetry, not a controller input. Coordinates outside the declared bounds are counted in the mapping and are not rescaled into view."},
        outputs={p.name: file_hash(p) for p in outputs},
        limitations=["Own-command stopping produces different path lengths; selected paths are descriptive examples.",
                     "No new trajectories, model evaluations, thresholds or outcome-selected examples.",
                     "Source-cluster bootstrap metadata is preserved but not plotted as if it were an identification interval."])
    if any(file_hash(path) != sha for path, sha in (hashes | sources).items()):
        raise RuntimeError("Plot inputs or sources changed during rendering")
    mapping["captions"]["per_goal_success"] += " Declared nontrivial queries per goal: " + ", ".join(
        str(n) for n in sorted({r["arrival_declared"] for r in per_goal})) + "; each actor is shown separately."
    mapping["captions"]["per_goal_FIRST"] += " Declared ordered comparisons per goal: " + ", ".join(
        str(n) for n in sorted({r["FIRST_declared"] for r in per_goal})) + "; per-goal counts remain in the mapping."
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2, allow_nan=False) + "\n")
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True, help="trace_analysis summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.aggregate, args.traces, args.output)
