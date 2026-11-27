"""Frozen self-target audit of the deployed affine deadline interface.

No learner or environment is constructed. The CLI restores only the existing
decoder/value adapter on the approved analysis host. Synthetic tests exercise
the projection, masks and selection without loading that adapter.
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .detectors import detector_hits
from .provenance import audit_source_snapshots, snapshot_sources
from .replay_cli import file_hash, load_fitted_adapter
from .replay_fit import episode_split, replay_batch, state_digest, validate_replay


PROTOCOL_SHA256 = "71b359d2c4d94e43025abed8f0fa035678ad544040d8096813306a9834296f84"


def project_curve(prediction, target, horizons, *, weights=None, usable=None,
                  affinity_atol=2e-5, affinity_rtol=2e-5,
                  remainder_atol=1e-8, remainder_rtol=1e-6):
    """Weighted affine floor and excess; the analytic h=0 is never included.

    Coefficients use x=h/max(h). A one-point valid curve is retained with slope
    zero and rank one. The target is an immutable supplied array: projection
    neither changes a network nor recomputes targets from the fitted line.
    """
    q, y, h = [np.asarray(value, dtype=np.float64) for value in (prediction, target, horizons)]
    if q.ndim != 1 or y.shape != q.shape or h.shape != q.shape or not np.isfinite(h).all():
        raise ValueError("Curves and deadlines must be aligned one-dimensional arrays")
    if (h < 0).any() or len(np.unique(h)) != len(h):
        raise ValueError("Deadlines must be distinct and nonnegative")
    positive = h > 0
    if positive.sum() < 2 or not np.isfinite(q[positive]).all():
        raise ValueError("At least two finite positive-deadline predictions are required")
    w = np.ones_like(h) if weights is None else np.asarray(weights, dtype=np.float64)
    mask = np.ones_like(positive) if usable is None else np.asarray(usable)
    if w.shape != q.shape or not np.isfinite(w).all() or (w < 0).any() or mask.shape != q.shape or mask.dtype != np.bool_:
        raise ValueError("Weights must be finite nonnegative and usability boolean")
    valid = positive & mask & (w > 0)
    if not valid.any() or not np.isfinite(y[valid]).all():
        raise ValueError("At least one finite usable positive-deadline label is required")
    first, last = np.flatnonzero(positive)[np.argsort(h[positive])[[0, -1]]]
    endpoint_line = q[first] + (q[last] - q[first]) * (h - h[first]) / (h[last] - h[first])
    affinity_error = float(np.max(np.abs(q[positive] - endpoint_line[positive])))
    affinity_tolerance = affinity_atol + affinity_rtol * float(np.max(np.abs(q[positive])))
    if affinity_error > affinity_tolerance:
        raise ValueError("Actual predictions violate the declared affine-deadline tolerance")
    x = h / h[positive].max()
    total_weight = float(w[valid].sum())
    normal = w[valid] / total_weight
    mean_x, mean_y = np.sum(normal * x[valid]), np.sum(normal * y[valid])
    variance = float(np.sum(normal * (x[valid] - mean_x) ** 2))
    rank = 2 if variance > np.finfo(float).eps else 1
    slope = float(np.sum(normal * (x[valid] - mean_x) * (y[valid] - mean_y)) / variance) if rank == 2 else 0.0
    intercept = float(mean_y - slope * mean_x)
    projected = intercept + slope * x
    mse = lambda error: float(np.sum(normal * error[valid] ** 2))
    total, floor, excess = mse(q-y), mse(projected-y), mse(q-projected)
    remainder = total - floor - excess
    tolerance = remainder_atol + remainder_rtol * max(1., abs(total), abs(floor), abs(excess))
    if abs(remainder) > tolerance:
        raise ValueError("Affine decomposition remainder exceeds the declared tolerance")
    return dict(positive_deadlines=int(positive.sum()), usable_points=int(valid.sum()),
        masked_positive_points=int((positive & ~valid).sum()), excluded_h0_points=int((h == 0).sum()),
        total_weight=total_weight, projection_rank=rank, intercept=intercept, normalized_deadline_slope=slope,
        total_mse=total, affine_floor_mse=floor, excess_mse=excess,
        decomposition_remainder=remainder, remainder_tolerance=tolerance,
        maximum_affinity_error=affinity_error, affinity_tolerance=affinity_tolerance)


def select_rows(data, fit_config, protocol):
    """Uniform fixed-size selection within the exact prior episode split."""
    setting = protocol["sampling"]
    train, holdout = episode_split(data["episode_id"], fit_config["holdout_fraction"], fit_config["seed"])
    rng = np.random.default_rng(setting["seed"])
    result = {}
    for name, population in (("train", train), ("holdout", holdout)):
        eligible = population[data["successor_valid"][population]]
        count = setting[f"{name}_rows"]
        if len(eligible) < count:
            raise ValueError(f"Insufficient valid-successor rows in {name}; refusing a smaller panel")
        result[name] = rng.choice(eligible, count, replace=False)
    return result


def selected_row_receipt(data, selection):
    """Bind exact sampled row contents without writing raw representations."""
    count = len(data["action"])
    keys = sorted(key for key, value in data.items() if value.ndim and len(value) == count)
    rows = []
    for panel, indices in selection.items():
        for index in indices:
            digest = hashlib.sha256()
            for key in keys:
                value = np.ascontiguousarray(data[key][index])
                digest.update(key.encode()); digest.update(value.dtype.str.encode())
                digest.update(str(value.shape).encode()); digest.update(value.tobytes())
            rows.append(dict(panel=panel, replay_index=int(index), episode_id=int(data["episode_id"][index]),
                factual_action=int(data["action"][index]), successor_valid=bool(data["successor_valid"][index]),
                successor_core_valid=bool(data["successor_core_valid"][index]), physical_done=bool(data["physical_done"][index]),
                content_sha256=digest.hexdigest()))
    return dict(schema="intrmotiv/selected-replay-rows/v1", hashed_fields=keys, rows=rows)


def verify_target_helpers(fitted, expected_manifest_sha256):
    """Bind the target arithmetic to the original fit, not merely its weights."""
    manifest_path = Path(fitted) / "source_snapshot/manifest.json"
    if file_hash(manifest_path) != expected_manifest_sha256:
        raise ValueError("Original fit source manifest bytes changed")
    manifest = json.loads(manifest_path.read_text())
    package = Path(__file__).parent
    for name in ("replay_fit.py", "targets.py", "detectors.py"):
        matches = [entry for entry in manifest["sources"]
                   if Path(entry["source_path"]).parts[-2:] == ("first_hit_control", name)]
        if len(matches) != 1 or file_hash(package / name) != matches[0]["sha256"]:
            raise ValueError(f"Audit target arithmetic differs from fitted source: {name}")


def aggregate_curves(rows):
    """Aggregate by usable deadline weight, preserving declared curve counts."""
    weight = sum(row["total_weight"] for row in rows)
    result = dict(curves=len(rows), usable_points=sum(row["usable_points"] for row in rows),
        masked_positive_points=sum(row["masked_positive_points"] for row in rows),
        rank_one_curves=sum(row["projection_rank"] == 1 for row in rows),
        initially_at_goal_curves=sum(row["initially_at_goal"] for row in rows), total_weight=weight)
    for field in ("total_mse", "affine_floor_mse", "excess_mse", "decomposition_remainder"):
        result[field] = sum(row[field] * row["total_weight"] for row in rows) / weight
    result["maximum_absolute_remainder"] = max(abs(row["decomposition_remainder"]) for row in rows)
    result["maximum_affinity_error"] = max(row["maximum_affinity_error"] for row in rows)
    return result


def boundary_metrics(q, y, mask):
    """The h1 labels are observed events, with no bootstrap contribution."""
    q, y, mask = np.asarray(q), np.asarray(y), np.asarray(mask)
    def section(chosen):
        if not chosen.any():
            return dict(labels=0, mse=None)
        prediction, target = q[chosen], y[chosen]
        return dict(labels=int(chosen.sum()), mse=float(np.mean((prediction-target) ** 2)),
            zero_prediction_mse=float(np.mean(target ** 2)), prediction_min=float(prediction.min()),
            prediction_max=float(prediction.max()), outside_unit_interval=float(np.mean((prediction < 0) | (prediction > 1))))
    if not np.isin(y[mask], [0., 1.]).all():
        raise ValueError("h1 targets must be observed binary goal events")
    return dict(all=section(mask), immediate_hit=section(mask & (y == 1)), not_immediate_hit=section(mask & (y == 0)))


@torch.no_grad()
def audit_selected(model, data, selection, protocol, detector_mode):
    """Materialize fixed target curves first, then apply only algebraic fits."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The audit requires frozen evaluation weights")
    horizon = protocol["curves"]["horizons"]
    if horizon != list(range(1, 257)) or model.horizon_scale != 256 or model.layout.goals != 16:
        raise ValueError("The declared native audit requires H256 and all16 goals")
    indices = np.concatenate(list(selection.values()))
    validate_replay(data, model.layout, model.q_heads.main.out_features)
    before = state_digest(model)
    detector = partial(detector_hits, mode=detector_mode)
    predictions, targets, masks = [], [], []
    for remaining in horizon:
        q, target, usable = replay_batch(model, model, data, indices, np.full(len(indices), remaining), detector)
        predictions.append(q.cpu().numpy().astype(np.float64))
        targets.append(target.cpu().numpy().astype(np.float64))
        masks.append(usable.cpu().numpy())
    # Targets are now fixed. No model evaluation occurs during projection.
    predictions, targets, masks = map(np.asarray, (predictions, targets, masks))
    ca3_end = model.eligibility["half_open_slices"]["canonical_ca3"][1]
    current_dg = data["core"][indices, :ca3_end].reshape(len(indices), model.layout.goals, -1)[:, :, 0]
    current_hits = detector(torch.as_tensor(current_dg.copy())).cpu().numpy()
    panel_names = [panel for panel, chosen in selection.items() for _ in chosen]
    rows = []
    affinity, remainder = [protocol["projection"][key] for key in ("affinity_tolerance", "remainder_tolerance")]
    for offset, index in enumerate(indices):
        for goal in protocol["curves"]["goals"]:
            metrics = project_curve(predictions[:, offset, goal], targets[:, offset, goal], horizon,
                usable=masks[:, offset, goal], affinity_atol=affinity["absolute"], affinity_rtol=affinity["relative"],
                remainder_atol=remainder["absolute"], remainder_rtol=remainder["relative"])
            rows.append(dict(panel=panel_names[offset], replay_index=int(index), goal=goal,
                factual_action=int(data["action"][index]), initially_at_goal=bool(current_hits[offset, goal]), **metrics))
    panels = {}
    for name in selection:
        chosen = np.asarray(panel_names) == name
        subset = [row for row in rows if row["panel"] == name]
        panels[name] = dict(aggregate=aggregate_curves(subset),
            per_goal={str(goal): aggregate_curves([row for row in subset if row["goal"] == goal]) for goal in range(16)},
            h1=boundary_metrics(predictions[0, chosen], targets[0, chosen], masks[0, chosen]))
    if state_digest(model) != before:
        raise RuntimeError("The affine audit mutated the fitted readout")
    if len(rows) != protocol["curves"]["declared_curves"]:
        raise ValueError("Audit lost declared query curves")
    return rows, dict(panels=panels, readout_unchanged=True, readout_sha256=before,
        target_array_sha256=hashlib.sha256(np.ascontiguousarray(targets).tobytes()).hexdigest(),
        usable_mask_sha256=hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest(),
        declared_curves=len(rows), declared_curve_points=int(targets.size))


def run(replay, fitted, source_root, protocol_path, output):
    replay, fitted, source_root, protocol_path, output = map(lambda value: Path(value).resolve(),
        (replay, fitted, source_root, protocol_path, output))
    if output.exists():
        raise FileExistsError("Preserve previous affine audit results")
    if file_hash(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Protocol differs from the reviewed preimplementation declaration")
    protocol = json.loads(protocol_path.read_text())
    metadata_path, data_path = fitted / "metadata.json", replay / "replay.npz"
    input_paths = [data_path, replay / "metadata.json", metadata_path, fitted / "adapter.pt"]
    declared = protocol["provenance"]["source_fit_inputs"]
    for path in input_paths:
        if str(path) not in declared or file_hash(path) != declared[str(path)]:
            raise ValueError(f"Input differs from the preregistered native fit: {path}")
    inputs = {str(path): file_hash(path) for path in input_paths + [protocol_path]}
    metadata = json.loads(metadata_path.read_text())
    fit_config = metadata["fit"]["config"]
    if fit_config != protocol["provenance"]["source_fit_config"]:
        raise ValueError("The original fit configuration changed")
    if metadata["source_snapshot_manifest_sha256"] != protocol["provenance"]["fit_source_snapshot_manifest_sha256"]:
        raise ValueError("Fit source archive identity changed")
    verify_target_helpers(fitted, metadata["source_snapshot_manifest_sha256"])
    with np.load(data_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    selection = select_rows(data, fit_config, protocol)
    output.mkdir(parents=True)
    package = Path(__file__).parent
    sources = [package / name for name in ("affine_deadline_audit.py", "replay_cli.py", "replay_fit.py", "targets.py",
        "detectors.py", "runtime_adapter.py", "legacy_runtime_adapter.py", "eligibility.py", "provenance.py")]
    sources += [source_root / "sf_working_directories/IntrMotiv/dmlab" / name
                for name in ("custom_core.py", "custom_decoder.py", "controller_q.py")]
    sources += [protocol_path, package.parent / "test_first_hit_affine_deadline_audit.py"]
    snapshot = snapshot_sources(sources, output / "source_snapshot")
    selected_path = output / "selected_rows.json"
    selected_path.write_text(json.dumps(selected_row_receipt(data, selection), indent=2) + "\n")
    preaudit = dict(schema="intrmotiv/affine-deadline-preaudit/v1", protocol_sha256=PROTOCOL_SHA256,
        input_files=inputs, source_snapshot=snapshot, selected_rows_sha256=file_hash(selected_path),
        stage="Written before adapter loading and all prediction/target inference")
    (output / "preaudit.json").write_text(json.dumps(preaudit, indent=2) + "\n")
    torch.set_num_threads(4)
    model, _ = load_fitted_adapter(fitted, source_root, device="cpu")
    if state_digest(model) != protocol["provenance"]["fit_readout_state_sha256"]:
        raise ValueError("Saved readout state differs from the preregistered diagnostic")
    rows, summary = audit_selected(model, data, selection, protocol, metadata["primary_detector"])
    with (output / "per_query.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    source_audit = audit_source_snapshots(output / "source_snapshot")
    unchanged = all(file_hash(path) == sha for path, sha in inputs.items())
    unchanged &= file_hash(selected_path) == preaudit["selected_rows_sha256"]
    if not source_audit["archive_valid"] or not source_audit["current_sources_unchanged"] or not unchanged:
        raise RuntimeError("Audit source or fit input changed during analysis")
    summary.update(schema="intrmotiv/affine-deadline-audit/v1", source_audit=source_audit, input_files_unchanged=unchanged,
        preaudit_sha256=file_hash(output / "preaudit.json"), per_query_sha256=file_hash(output / "per_query.csv"),
        interpretation=protocol["interpretation"] + ["Initially satisfied commands remain in the original training-label panel; their values are outside executed navigation, which stops at time zero."],
        native_environment_executed=False, training_updates=0)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "fitted", "source-root", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.replay, args.fitted, args.source_root, args.protocol, args.output)
    print(json.dumps({"curves": result["declared_curves"], "readout_unchanged": result["readout_unchanged"]}))
