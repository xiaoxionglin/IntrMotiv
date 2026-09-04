"""Post-hoc collection and selected rendering for online spatial snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .spatial_contract import (
    SNAPSHOT_SCHEMA,
    SpatialBounds,
    SpatialContractError,
    calculate_spatial_metrics,
    load_spatial_snapshot,
    spatial_rate_maps,
)
from .spec import SpecError, StudySpec


SPATIAL_METRICS = (
    "valid_sample_count",
    "in_bounds_fraction",
    "visited_cell_fraction",
    "active_unit_fraction",
    "silent_unit_fraction",
    "active_unit_mean_spatial_information",
    "active_only_map_cosine",
    "unique_active_peak_bins",
    "mean_physical_step_distance",
    "stationary_step_fraction",
    "path_efficiency",
    "mean_absolute_circular_yaw_change",
)
DEFAULT_TARGETS = (25_000_000, 50_000_000, 75_000_000, 100_000_000)


def _inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _scalar(payload: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(payload[key])
    if value.size != 1:
        raise SpecError(f"snapshot metadata {key!r} must be scalar")
    return value.item()


def expected_spatial_targets(study: StudySpec) -> tuple[int, ...]:
    explicit = study.telemetry.get("online_spatial_target_frames")
    if explicit is None:
        standard = study.telemetry.get("target_frames", DEFAULT_TARGETS)
        explicit = [value for value in standard if int(value) in DEFAULT_TARGETS]
    try:
        targets = tuple(int(value) for value in explicit)
    except (TypeError, ValueError) as error:
        raise SpecError("telemetry online spatial targets must be integers") from error
    if not targets or len(set(targets)) != len(targets) or any(value <= 0 for value in targets):
        raise SpecError("telemetry online spatial targets must be unique positive integers")
    if any(value % 25_000_000 for value in targets):
        raise SpecError("online spatial targets must follow the 25M-frame cadence")
    return targets


def discover_spatial_snapshots(
    study: StudySpec,
    snapshot_root: Path,
    *,
    require_workspace: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    snapshot_root = Path(snapshot_root)
    workspace = Path(study.workspace_root)
    if require_workspace and not _inside(snapshot_root, workspace):
        raise SpecError(f"snapshot root {snapshot_root} is outside workspace {workspace}")
    expected_runs = {run.name for run in study.expand_runs()}
    expected_targets = set(expected_spatial_targets(study))
    discovered: list[tuple[Path, dict[str, Any]]] = []
    identities: set[tuple[str, int, int]] = set()
    for path in sorted(snapshot_root.rglob("*.npz")):
        if require_workspace and not _inside(path, workspace):
            raise SpecError(f"snapshot {path} is outside workspace {workspace}")
        try:
            payload = load_spatial_snapshot(path)
        except (OSError, SpatialContractError, ValueError) as error:
            raise SpecError(f"invalid online spatial snapshot {path}: {error}") from error
        run_name = str(_scalar(payload, "run_name"))
        policy_id = int(_scalar(payload, "policy_id"))
        target = int(_scalar(payload, "target_env_steps"))
        if run_name not in expected_runs:
            raise SpecError(f"snapshot {path} declares unexpected run {run_name!r}")
        if target not in expected_targets:
            raise SpecError(f"snapshot {path} has unexpected target {target}")
        identity = (run_name, policy_id, target)
        if identity in identities:
            raise SpecError(f"duplicate spatial snapshot identity {identity}")
        identities.add(identity)
        discovered.append((path, payload))
    if not discovered:
        raise SpecError(f"no {SNAPSHOT_SCHEMA} snapshots found under {snapshot_root}")
    return discovered


def collect_spatial_records(
    study: StudySpec,
    snapshot_root: Path,
    *,
    require_workspace: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    runs = {run.name: run for run in study.expand_runs()}
    records: list[dict[str, Any]] = []
    snapshots = discover_spatial_snapshots(study, snapshot_root, require_workspace=require_workspace)
    observed: set[tuple[str, int, int]] = set()
    policies: set[int] = set()
    for path, payload in snapshots:
        run_name = str(_scalar(payload, "run_name"))
        run = runs[run_name]
        policy_id = int(_scalar(payload, "policy_id"))
        target = int(_scalar(payload, "target_env_steps"))
        policies.add(policy_id)
        observed.add((run_name, policy_id, target))
        bounds_array = np.asarray(payload["bounds"], dtype=np.float32)
        bounds = SpatialBounds(*[float(value) for value in bounds_array])
        metrics = calculate_spatial_metrics(
            payload["pose"],
            payload["dg_activity"],
            payload["dones"],
            payload["segment_id"],
            bounds,
            int(_scalar(payload, "grain")),
            float(np.asarray(payload.get("stationary_distance", 1.0)).item()),
        )
        records.append({
            "run_name": run_name,
            "condition": run.condition,
            "base": run.base,
            "seed": run.seed,
            **run.factors,
            **run.metadata,
            "policy_id": policy_id,
            "target_env_steps": target,
            "actual_env_steps": int(_scalar(payload, "actual_env_steps")),
            "window_limit": int(_scalar(payload, "window_limit")),
            "environment": str(_scalar(payload, "environment")),
            "frameskip": int(_scalar(payload, "frameskip")),
            "snapshot_path": str(path.resolve()),
            **metrics,
        })
    records.sort(key=lambda row: (row["run_name"], row["policy_id"], row["target_env_steps"]))

    targets = expected_spatial_targets(study)
    policy_ids = sorted(policies or {0})
    missing = [
        {"run_name": run.name, "policy_id": policy, "target_env_steps": target}
        for run in study.expand_runs()
        for policy in policy_ids
        for target in targets
        if (run.name, policy, target) not in observed
    ]
    inventory = [{
        "run_name": row["run_name"],
        "policy_id": row["policy_id"],
        "target_env_steps": row["target_env_steps"],
        "snapshot_path": row["snapshot_path"],
    } for row in records]
    status = {
        **study.provenance(),
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "expected_targets": list(targets),
        "policies": policy_ids,
        "observed_snapshots": len(records),
        "expected_snapshots": len(study.expand_runs()) * len(policy_ids) * len(targets),
        "complete": not missing,
        "missing": missing,
    }
    return records, inventory, status


def summarize_spatial_records(
    records: Sequence[Mapping[str, Any]], group_by: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def summarize(fields: Sequence[str]) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for record in records:
            key = tuple(record[field] for field in fields)
            groups.setdefault(key, []).append(record)
        rows: list[dict[str, Any]] = []
        for key in sorted(groups, key=lambda item: tuple(map(str, item))):
            members = groups[key]
            row = dict(zip(fields, key))
            for metric in SPATIAL_METRICS:
                values = [float(member[metric]) for member in members if math.isfinite(float(member[metric]))]
                row[f"{metric}__mean"] = fmean(values) if values else math.nan
                row[f"{metric}__sd"] = stdev(values) if len(values) > 1 else math.nan
                row[f"{metric}__n"] = len(values)
            rows.append(row)
        return rows

    suffix = ("policy_id", "target_env_steps")
    return summarize((*group_by, *suffix)), summarize((*group_by, "seed", *suffix))


def _figure_runtime():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    font_path = Path(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False
    ))
    if font_path.suffix.lower() not in {".ttf", ".otf"} or not font_path.is_file():
        raise RuntimeError("a verified scalable DejaVu Sans TTF/OTF font is required")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def _save_figure(fig, stem: Path, plt) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=100, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def render_place_field_contact_sheets(payload: Mapping[str, Any], output_stem: Path) -> list[Path]:
    plt = _figure_runtime()
    bounds_values = np.asarray(payload["bounds"], dtype=float)
    bounds = SpatialBounds(*bounds_values.tolist())
    maps, occupancy, in_bounds = spatial_rate_maps(
        payload["pose"], payload["dg_activity"], bounds, int(_scalar(payload, "grain"))
    )
    active = (np.asarray(payload["dg_activity"]) > 0).any(axis=0)
    units = np.flatnonzero(active)
    if not units.size:
        units = np.arange(min(1, maps.shape[0]))
    outputs: list[Path] = []
    mask = occupancy == 0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    for page, start in enumerate(range(0, units.size, 16), start=1):
        page_units = units[start : start + 16]
        fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
        image = None
        for ax, unit in zip(axes.flat, page_units):
            image = ax.imshow(
                np.ma.array(maps[unit], mask=mask),
                origin="lower",
                extent=(bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max),
                cmap=cmap,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(f"DG unit {int(unit)}")
            ax.set_xticks((bounds.x_min, bounds.x_max))
            ax.set_yticks((bounds.y_min, bounds.y_max))
        for ax in axes.flat[len(page_units) :]:
            ax.set_visible(False)
        if image is not None:
            fig.colorbar(image, ax=list(axes.flat), shrink=0.65, label="Mean thresholded DG activity")
        fig.suptitle(
            f"{_scalar(payload, 'run_name')} · target {int(_scalar(payload, 'target_env_steps')):,} · "
            f"active units {int(active.sum())}/{active.size} · page {page}"
        )
        outputs.extend(_save_figure(fig, output_stem.with_name(f"{output_stem.name}_page{page:02d}"), plt))
    return outputs


def render_occupancy_trajectory(payload: Mapping[str, Any], output_stem: Path) -> list[Path]:
    plt = _figure_runtime()
    pose = np.asarray(payload["pose"], dtype=np.float32)
    segments = np.asarray(payload["segment_id"], dtype=np.int64)
    bounds_values = np.asarray(payload["bounds"], dtype=float)
    bounds = SpatialBounds(*bounds_values.tolist())
    _, occupancy, _ = spatial_rate_maps(
        pose, payload["dg_activity"], bounds, int(_scalar(payload, "grain"))
    )
    fig, (occupancy_ax, trajectory_ax) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#eeeeee")
    image = occupancy_ax.imshow(
        np.ma.masked_equal(occupancy, 0),
        origin="lower",
        extent=(bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max),
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    fig.colorbar(image, ax=occupancy_ax, shrink=0.78, label="Observations per visited bin")
    occupancy_ax.set_title("Occupancy (unvisited masked)")
    occupancy_ax.set_xlabel("x (DMLab units)")
    occupancy_ax.set_ylabel("y (DMLab units)")

    starts = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1]
    ends = np.r_[starts[1:], pose.shape[0]]
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(1, len(starts))))
    for color, start, end in zip(colors, starts, ends):
        trajectory_ax.plot(pose[start:end, 0], pose[start:end, 1], color=color, linewidth=1.4, alpha=0.8)
        trajectory_ax.scatter(pose[start, 0], pose[start, 1], color=color, marker="o", s=24)
        trajectory_ax.scatter(pose[end - 1, 0], pose[end - 1, 1], color=color, marker="x", s=30)
    stride = max(1, pose.shape[0] // 100)
    yaw = np.deg2rad(pose[::stride, 2])
    trajectory_ax.quiver(
        pose[::stride, 0], pose[::stride, 1], np.cos(yaw), np.sin(yaw),
        color="#202020", angles="xy", scale_units="xy", scale=0.012, width=0.0025, alpha=0.65,
    )
    trajectory_ax.set_xlim(bounds.x_min, bounds.x_max)
    trajectory_ax.set_ylim(bounds.y_min, bounds.y_max)
    trajectory_ax.set_aspect("equal")
    trajectory_ax.set_title(f"Trajectory · {len(starts)} independent segments")
    trajectory_ax.set_xlabel("x (DMLab units)")
    trajectory_ax.set_ylabel("y (DMLab units)")
    fig.suptitle(
        f"{_scalar(payload, 'run_name')} · target {int(_scalar(payload, 'target_env_steps')):,} · "
        f"actual {int(_scalar(payload, 'actual_env_steps')):,}"
    )
    return _save_figure(fig, output_stem, plt)


def render_selected_snapshots(
    snapshots: Iterable[tuple[Path, Mapping[str, Any]]],
    output_dir: Path,
    selected_runs: Sequence[str],
    selected_targets: Sequence[int],
) -> list[Path]:
    if not selected_runs:
        return []
    run_set = set(selected_runs)
    target_set = set(selected_targets)
    outputs: list[Path] = []
    matched: set[tuple[str, int]] = set()
    for _, payload in snapshots:
        run_name = str(_scalar(payload, "run_name"))
        target = int(_scalar(payload, "target_env_steps"))
        if run_name not in run_set or (target_set and target not in target_set):
            continue
        policy = int(_scalar(payload, "policy_id"))
        matched.add((run_name, target))
        stem_dir = output_dir / "figures" / run_name
        prefix = f"target_{target:012d}_policy_{policy:02d}"
        outputs.extend(render_place_field_contact_sheets(payload, stem_dir / f"{prefix}_place_fields"))
        outputs.extend(render_occupancy_trajectory(payload, stem_dir / f"{prefix}_trajectory"))
    if target_set:
        missing = sorted({(run, target) for run in run_set for target in target_set} - matched)
    else:
        matched_runs = {run for run, _ in matched}
        missing = sorted((run, -1) for run in run_set - matched_runs)
    if missing:
        raise SpecError(f"selected run/target snapshots were not found: {missing}")
    return outputs
