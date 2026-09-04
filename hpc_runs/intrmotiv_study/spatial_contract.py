"""Shared contract and calculations for compact online spatial telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


SNAPSHOT_SCHEMA = "intrmotiv/online-spatial/v1"
DEFAULT_GRAIN = 19
DEFAULT_BOUNDS = (100.0, 2000.0, 100.0, 2000.0)
SNAPSHOT_REQUIRED_ARRAYS = (
    "pose",
    "dg_activity",
    "actions",
    "dones",
    "segment_id",
    "policy_version",
)


class SpatialContractError(ValueError):
    """Raised when an online spatial payload violates the v1 contract."""


@dataclass(frozen=True)
class SpatialBounds:
    x_min: float = DEFAULT_BOUNDS[0]
    x_max: float = DEFAULT_BOUNDS[1]
    y_min: float = DEFAULT_BOUNDS[2]
    y_max: float = DEFAULT_BOUNDS[3]

    def __post_init__(self) -> None:
        values = np.asarray((self.x_min, self.x_max, self.y_min, self.y_max), dtype=np.float64)
        if not np.isfinite(values).all():
            raise SpatialContractError("spatial bounds must be finite")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise SpatialContractError("spatial bounds require min < max on both axes")

    def as_array(self) -> np.ndarray:
        return np.asarray((self.x_min, self.x_max, self.y_min, self.y_max), dtype=np.float32)


def _aligned_arrays(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    segment_id: np.ndarray,
    policy_version: np.ndarray,
) -> tuple[np.ndarray, ...]:
    pose = np.asarray(pose, dtype=np.float32)
    dg_activity = np.asarray(dg_activity, dtype=np.float32)
    actions = np.asarray(actions)
    dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
    segment_id = np.asarray(segment_id, dtype=np.int64).reshape(-1)
    policy_version = np.asarray(policy_version, dtype=np.int64).reshape(-1)
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise SpatialContractError(f"pose must have shape [N, 3], got {pose.shape}")
    if dg_activity.ndim != 2:
        raise SpatialContractError(f"dg_activity must have shape [N, units], got {dg_activity.shape}")
    n = pose.shape[0]
    if n == 0:
        raise SpatialContractError("spatial telemetry cannot be empty")
    if actions.ndim == 0 or actions.shape[0] != n:
        raise SpatialContractError("actions must have N rows")
    for name, value in (
        ("dg_activity", dg_activity),
        ("dones", dones),
        ("segment_id", segment_id),
        ("policy_version", policy_version),
    ):
        if value.shape[0] != n:
            raise SpatialContractError(f"{name} has {value.shape[0]} rows; expected {n}")
    if not np.isfinite(pose).all():
        raise SpatialContractError("pose contains non-finite values")
    if not np.isfinite(dg_activity).all():
        raise SpatialContractError("dg_activity contains non-finite values")
    if (dg_activity < 0).any():
        raise SpatialContractError("thresholded DG activity must be non-negative")
    return pose, dg_activity, actions, dones, segment_id, policy_version


def spatial_rate_maps(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    bounds: SpatialBounds = SpatialBounds(),
    grain: int = DEFAULT_GRAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return occupancy-corrected unit maps, occupancy, and in-bounds mask."""

    pose = np.asarray(pose, dtype=np.float32)
    activity = np.asarray(dg_activity, dtype=np.float32)
    if grain <= 0:
        raise SpatialContractError("grain must be positive")
    if pose.ndim != 2 or pose.shape[1] != 3 or activity.ndim != 2 or activity.shape[0] != pose.shape[0]:
        raise SpatialContractError("pose and DG activity must be aligned [N, 3] and [N, units] arrays")
    if not np.isfinite(pose).all() or not np.isfinite(activity).all():
        raise SpatialContractError("pose and DG activity must be finite")

    x = pose[:, 0]
    y = pose[:, 1]
    in_bounds = (
        (x >= bounds.x_min)
        & (x <= bounds.x_max)
        & (y >= bounds.y_min)
        & (y <= bounds.y_max)
    )
    occupancy = np.zeros((grain, grain), dtype=np.int64)
    rate_sums = np.zeros((activity.shape[1], grain, grain), dtype=np.float64)
    if in_bounds.any():
        x_bin = np.floor((x[in_bounds] - bounds.x_min) / (bounds.x_max - bounds.x_min) * grain).astype(int)
        y_bin = np.floor((y[in_bounds] - bounds.y_min) / (bounds.y_max - bounds.y_min) * grain).astype(int)
        x_bin = np.clip(x_bin, 0, grain - 1)
        y_bin = np.clip(y_bin, 0, grain - 1)
        np.add.at(occupancy, (y_bin, x_bin), 1)
        bounded_activity = activity[in_bounds]
        for unit in range(activity.shape[1]):
            np.add.at(rate_sums[unit], (y_bin, x_bin), bounded_activity[:, unit])

    rate_maps = np.divide(
        rate_sums,
        occupancy[None, :, :],
        out=np.zeros_like(rate_sums),
        where=occupancy[None, :, :] > 0,
    ).astype(np.float32)
    return rate_maps, occupancy, in_bounds


def _spatial_information(rate_map: np.ndarray, occupancy: np.ndarray) -> float:
    total = float(occupancy.sum())
    if total <= 0:
        return 0.0
    probability = occupancy.astype(np.float64) / total
    mean_rate = float(np.sum(rate_map * probability))
    if mean_rate <= 0:
        return 0.0
    positive = (rate_map > 0) & (probability > 0)
    ratio = rate_map[positive] / mean_rate
    return float(np.sum(rate_map[positive] * probability[positive] * np.log2(ratio)))


def calculate_spatial_metrics(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    dones: np.ndarray,
    segment_id: np.ndarray,
    bounds: SpatialBounds = SpatialBounds(),
    grain: int = DEFAULT_GRAIN,
    stationary_distance: float = 1.0,
) -> dict[str, float]:
    """Calculate monitoring metrics from exact behavior-time pose and DG activity."""

    if stationary_distance < 0 or not np.isfinite(stationary_distance):
        raise SpatialContractError("stationary_distance must be finite and non-negative")
    pose = np.asarray(pose, dtype=np.float32)
    activity = np.asarray(dg_activity, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
    segment_id = np.asarray(segment_id, dtype=np.int64).reshape(-1)
    if pose.ndim != 2 or activity.ndim != 2 or pose.shape[0] != activity.shape[0]:
        raise SpatialContractError("pose and DG activity arrays are not aligned")
    if dones.shape[0] != pose.shape[0] or segment_id.shape[0] != pose.shape[0]:
        raise SpatialContractError("trajectory arrays are not aligned")
    if not np.isfinite(pose).all() or not np.isfinite(activity).all():
        raise SpatialContractError("pose and DG activity must be finite")

    rate_maps, occupancy, in_bounds = spatial_rate_maps(pose, activity, bounds, grain)
    bounded_activity = activity[in_bounds]
    active = (bounded_activity > 0).any(axis=0)
    n_units = int(activity.shape[1])
    n_active = int(active.sum())
    active_maps = rate_maps[active]
    information = [_spatial_information(unit_map, occupancy) for unit_map in active_maps]

    occupied = occupancy.reshape(-1) > 0
    map_cosine = 0.0
    if n_active >= 2 and occupied.any():
        vectors = active_maps.reshape(n_active, -1)[:, occupied].astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        usable = norms > 0
        vectors = vectors[usable]
        norms = norms[usable]
        if vectors.shape[0] >= 2:
            similarity = (vectors @ vectors.T) / np.outer(norms, norms)
            upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
            map_cosine = float(upper.mean()) if upper.size else 0.0

    unique_peaks = 0
    if n_active:
        unique_peaks = int(np.unique(active_maps.reshape(n_active, -1).argmax(axis=1)).size)

    transition = (segment_id[1:] == segment_id[:-1]) & ~dones[:-1]
    delta_xy = pose[1:, :2] - pose[:-1, :2]
    step_distance = np.linalg.norm(delta_xy, axis=1)[transition]
    yaw_delta = ((pose[1:, 2] - pose[:-1, 2] + 180.0) % 360.0) - 180.0
    yaw_delta = np.abs(yaw_delta[transition])

    total_path = float(step_distance.sum())
    total_displacement = 0.0
    for segment in np.unique(segment_id):
        indices = np.flatnonzero(segment_id == segment)
        if indices.size >= 2:
            total_displacement += float(np.linalg.norm(pose[indices[-1], :2] - pose[indices[0], :2]))

    n = int(pose.shape[0])
    return {
        "valid_sample_count": float(n),
        "in_bounds_fraction": float(in_bounds.mean()) if n else 0.0,
        "visited_cell_fraction": float((occupancy > 0).sum() / occupancy.size),
        "active_unit_fraction": float(n_active / n_units) if n_units else 0.0,
        "silent_unit_fraction": float(1.0 - n_active / n_units) if n_units else 0.0,
        "active_unit_mean_spatial_information": float(np.mean(information)) if information else 0.0,
        "active_only_map_cosine": map_cosine,
        "unique_active_peak_bins": float(unique_peaks),
        "mean_physical_step_distance": float(step_distance.mean()) if step_distance.size else 0.0,
        "stationary_step_fraction": float((step_distance <= stationary_distance).mean()) if step_distance.size else 0.0,
        "path_efficiency": float(total_displacement / total_path) if total_path > 0 else 0.0,
        "mean_absolute_circular_yaw_change": float(yaw_delta.mean()) if yaw_delta.size else 0.0,
    }


class OnlineSpatialWindow:
    """A bounded latest-N buffer that preserves rollout and episode segments."""

    def __init__(self, limit: int):
        if limit <= 0:
            raise SpatialContractError("window limit must be positive")
        self.limit = int(limit)
        self._next_segment = 0
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return 0 if not self._arrays else int(self._arrays["pose"].shape[0])

    def append_rollouts(
        self,
        pose: np.ndarray,
        dg_activity: np.ndarray,
        actions: np.ndarray,
        dones: np.ndarray,
        policy_version: np.ndarray,
        valids: np.ndarray,
    ) -> int:
        pose = np.asarray(pose, dtype=np.float32)
        activity = np.asarray(dg_activity, dtype=np.float32)
        actions = np.asarray(actions)
        dones = np.asarray(dones, dtype=np.bool_)
        versions = np.asarray(policy_version)
        valids = np.asarray(valids, dtype=np.bool_)
        if pose.ndim != 3 or pose.shape[-1] != 3 or activity.ndim != 3:
            raise SpatialContractError("rollout pose/activity must have shapes [B,T,3] and [B,T,units]")
        leading = pose.shape[:2]
        for name, value in (
            ("dg_activity", activity),
            ("actions", actions),
            ("dones", dones),
            ("policy_version", versions),
            ("valids", valids),
        ):
            if value.shape[:2] != leading:
                raise SpatialContractError(f"{name} rollout shape {value.shape[:2]} != {leading}")

        selected: dict[str, list[np.ndarray | Any]] = {
            "pose": [], "dg_activity": [], "actions": [], "dones": [],
            "segment_id": [], "policy_version": [],
        }
        for row in range(leading[0]):
            segment = self._next_segment
            self._next_segment += 1
            previous_valid = False
            for step in range(leading[1]):
                if not valids[row, step] or not np.isfinite(pose[row, step]).all():
                    if previous_valid:
                        segment = self._next_segment
                        self._next_segment += 1
                    previous_valid = False
                    continue
                if not previous_valid and selected["pose"]:
                    # A gap never inherits the segment preceding it.
                    segment = self._next_segment
                    self._next_segment += 1
                selected["pose"].append(pose[row, step])
                selected["dg_activity"].append(activity[row, step])
                selected["actions"].append(actions[row, step])
                selected["dones"].append(bool(dones[row, step]))
                selected["segment_id"].append(segment)
                selected["policy_version"].append(int(round(float(versions[row, step]))))
                previous_valid = True
                if dones[row, step]:
                    segment = self._next_segment
                    self._next_segment += 1
                    previous_valid = False

        if not selected["pose"]:
            return 0
        incoming = {
            "pose": np.asarray(selected["pose"], dtype=np.float32),
            "dg_activity": np.asarray(selected["dg_activity"], dtype=np.float32),
            "actions": np.asarray(selected["actions"]),
            "dones": np.asarray(selected["dones"], dtype=np.bool_),
            "segment_id": np.asarray(selected["segment_id"], dtype=np.int64),
            "policy_version": np.asarray(selected["policy_version"], dtype=np.int64),
        }
        if self._arrays:
            incoming = {key: np.concatenate((self._arrays[key], value), axis=0) for key, value in incoming.items()}
        self._arrays = {key: value[-self.limit :] for key, value in incoming.items()}
        return len(selected["pose"])

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._arrays.items()}


def validate_snapshot_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(np.asarray(payload.get("schema", "")).item())
    if schema != SNAPSHOT_SCHEMA:
        raise SpatialContractError(f"schema must be {SNAPSHOT_SCHEMA!r}, got {schema!r}")
    missing = [key for key in SNAPSHOT_REQUIRED_ARRAYS if key not in payload]
    if missing:
        raise SpatialContractError(f"snapshot is missing arrays: {missing}")
    arrays = _aligned_arrays(*(payload[key] for key in SNAPSHOT_REQUIRED_ARRAYS))
    result = dict(payload)
    for key, value in zip(SNAPSHOT_REQUIRED_ARRAYS, arrays):
        result[key] = value
    required_metadata = (
        "target_env_steps", "actual_env_steps", "window_limit", "window_start_env_steps",
        "window_end_env_steps", "policy_id", "run_name", "environment", "frameskip", "grain", "bounds",
    )
    absent = [key for key in required_metadata if key not in payload]
    if absent:
        raise SpatialContractError(f"snapshot is missing metadata: {absent}")
    bounds_array = np.asarray(payload["bounds"], dtype=np.float32).reshape(-1)
    if bounds_array.shape != (4,):
        raise SpatialContractError("bounds must contain x_min, x_max, y_min, y_max")
    SpatialBounds(*[float(item) for item in bounds_array])
    if int(np.asarray(payload["grain"]).item()) <= 0:
        raise SpatialContractError("grain must be positive")
    if int(np.asarray(payload["target_env_steps"]).item()) <= 0:
        raise SpatialContractError("target_env_steps must be positive")
    if int(np.asarray(payload["actual_env_steps"]).item()) < int(np.asarray(payload["target_env_steps"]).item()):
        raise SpatialContractError("actual_env_steps cannot precede target_env_steps")
    return result


def load_spatial_snapshot(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return validate_snapshot_payload({key: data[key] for key in data.files})


def write_spatial_snapshot_atomic(output_dir: Path, payload: Mapping[str, Any]) -> tuple[Path, bool]:
    """Write a compressed target/actual snapshot atomically without replacement."""

    validated = validate_snapshot_payload(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = int(np.asarray(validated["target_env_steps"]).item())
    actual = int(np.asarray(validated["actual_env_steps"]).item())
    run_name = str(np.asarray(validated["run_name"]).item())
    policy_id = int(np.asarray(validated["policy_id"]).item())
    existing = sorted(output_dir.glob(f"snapshot_target_{target:012d}_actual_*.npz"))
    if existing:
        for path in existing:
            prior = load_spatial_snapshot(path)
            if str(np.asarray(prior["run_name"]).item()) != run_name or int(np.asarray(prior["policy_id"]).item()) != policy_id:
                raise SpatialContractError(f"existing target has conflicting identity: {path}")
        return existing[0], False

    destination = output_dir / f"snapshot_target_{target:012d}_actual_{actual:012d}.npz"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".online-spatial-", suffix=".npz", dir=output_dir, delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **validated)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            prior = load_spatial_snapshot(destination)
            if str(np.asarray(prior["run_name"]).item()) != run_name or int(np.asarray(prior["policy_id"]).item()) != policy_id:
                raise SpatialContractError(f"concurrent snapshot has conflicting identity: {destination}")
            return destination, False
        return destination, True
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
