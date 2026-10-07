"""Shared spatial metrics for frozen representation probes.

Report the canonical amplitude-weighted score (bits per decision) and its
gain-invariant normalization (bits per unit of mean activation) together.
"""

from __future__ import annotations

import numpy as np

from hpc_runs.intrmotiv_study.spatial_contract import (
    SpatialBounds,
    calculate_place_field_details,
)


LAYERS = ("dg", "ca3", "decoder_1", "decoder_2")
PREFIXES = (10_000, 20_000, 30_000, 40_000, 50_000)
MIN_SHARED_VISITS = 10
MIN_SHARED_BINS = 20


def layer_details(pose: np.ndarray, activity: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate canonical 19x19 maps, retaining the original signed trace.

    DG is already nonnegative. A signed downstream trace is rectified only for
    rate/field metrics; callers preserve its raw values in the rollout artifact.
    """
    activity = np.asarray(activity, dtype=np.float32)
    if activity.ndim != 2 or not np.isfinite(activity).all():
        raise ValueError("activity must be a finite [decision, unit] array")
    return calculate_place_field_details(
        np.asarray(pose, dtype=np.float32), np.maximum(activity, 0), SpatialBounds(), 19
    )


def comparison_details(pose: np.ndarray, activity: np.ndarray) -> dict[str, np.ndarray]:
    """Only the canonical map/eligibility arrays needed at each prefix.

    Component labeling across 1,136 CA3 channels is reserved for complete
    50k maps; repeating it at every prefix adds cost without changing SI.
    """
    pose = np.asarray(pose, dtype=np.float32)
    activity = np.maximum(np.asarray(activity, dtype=np.float32), 0)
    if pose.ndim != 2 or pose.shape[1] != 3 or activity.ndim != 2 or len(pose) != len(activity):
        raise ValueError("Unaligned pose and activity")
    x = np.floor((pose[:, 0] - 100) / 100).astype(int)
    y = np.floor((pose[:, 1] - 100) / 100).astype(int)
    valid = (x >= 0) & (x < 19) & (y >= 0) & (y < 19)
    bins = y[valid] * 19 + x[valid]
    values = activity[valid]
    occupancy = np.bincount(bins, minlength=361).reshape(19, 19)
    maps = np.zeros((361, values.shape[1]), dtype=np.float32)
    eligible = np.zeros(values.shape[1], dtype=bool)
    for unit in range(values.shape[1]):
        sums = np.bincount(bins, weights=values[:, unit], minlength=361)
        maps[:, unit] = np.divide(sums, occupancy.ravel(), out=np.zeros(361),
                                  where=occupancy.ravel() > 0)
        active = values[:, unit] > 0
        eligible[unit] = active.sum() >= 20 and np.unique(bins[active]).size >= 3
    return {"occupancy": occupancy, "rate_maps": maps.reshape(19, 19, -1),
            "field_eligible": eligible}


def information_scores(rate_map: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return Skaggs bits per activation and bits per decision on given weights."""
    rate = np.asarray(rate_map, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if rate.shape != weight.shape or (rate < 0).any() or (weight < 0).any():
        raise ValueError("rate and weight must be same-shape nonnegative arrays")
    total = weight.sum()
    if total <= 0:
        return float("nan"), float("nan")
    p = weight / total
    mean = float(np.sum(p * rate))
    if mean <= 0:
        return float("nan"), float("nan")
    positive = (p > 0) & (rate > 0)
    bits_per_step = float(np.sum(
        p[positive] * rate[positive] * np.log2(rate[positive] / mean)
    ))
    return bits_per_step / mean, bits_per_step


def normalized_information(rate_map: np.ndarray, weights: np.ndarray) -> float:
    """Skaggs bits per activation on explicitly chosen spatial weights."""
    return information_scores(rate_map, weights)[0]


def paired_information(
    own: dict[str, np.ndarray], random: dict[str, np.ndarray]
) -> dict[str, object]:
    """Compare identical units on spatial bins sampled by both policies.

    Uniform weighting over shared bins removes policy occupancy weights from
    the comparison. A probe without at least 20 supported bins is undefined.
    """
    own_occ = np.asarray(own["occupancy"])
    random_occ = np.asarray(random["occupancy"])
    if own_occ.shape != (19, 19) or random_occ.shape != own_occ.shape:
        raise ValueError("Both occupancy grids must be 19x19")
    shared = (own_occ >= MIN_SHARED_VISITS) & (random_occ >= MIN_SHARED_VISITS)
    own_eligible = np.asarray(own["field_eligible"], dtype=bool)
    random_eligible = np.asarray(random["field_eligible"], dtype=bool)
    if own_eligible.shape != random_eligible.shape:
        raise ValueError("The same model must have the same unit count")
    eligible = own_eligible & random_eligible
    result: dict[str, object] = {
        "shared_bins": int(shared.sum()),
        "eligible_units": int(eligible.sum()),
        "own_bins": int((own_occ > 0).sum()),
        "random_bins": int((random_occ > 0).sum()),
        "own_active_units": int(own_eligible.sum()),
        "random_active_units": int(random_eligible.sum()),
        "supported": bool(shared.sum() >= MIN_SHARED_BINS and eligible.any()),
        "own_bits_per_activation": None,
        "random_bits_per_activation": None,
        "difference": None,
        "own_bits_per_step": None,
        "random_bits_per_step": None,
        "bits_per_step_difference": None,
        "own_mean_activation": None,
        "random_mean_activation": None,
        "mean_activation_difference": None,
    }
    if not result["supported"]:
        return result
    weights = shared.astype(np.float64)
    own_map = np.asarray(own["rate_maps"])
    random_map = np.asarray(random["rate_maps"])
    own_scores = []
    random_scores = []
    own_step_scores = []
    random_step_scores = []
    own_means = []
    random_means = []
    for unit in np.flatnonzero(eligible):
        a, a_step = information_scores(own_map[:, :, unit], weights)
        b, b_step = information_scores(random_map[:, :, unit], weights)
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(a_step) and np.isfinite(b_step):
            own_scores.append(a)
            random_scores.append(b)
            own_step_scores.append(a_step)
            random_step_scores.append(b_step)
            own_means.append(float(np.mean(own_map[:, :, unit][shared])))
            random_means.append(float(np.mean(random_map[:, :, unit][shared])))
    result["eligible_units"] = len(own_scores)
    if not own_scores:
        result["supported"] = False
        return result
    result["own_bits_per_activation"] = float(np.mean(own_scores))
    result["random_bits_per_activation"] = float(np.mean(random_scores))
    result["difference"] = float(np.mean(np.asarray(own_scores) - random_scores))
    result["own_bits_per_step"] = float(np.mean(own_step_scores))
    result["random_bits_per_step"] = float(np.mean(random_step_scores))
    result["bits_per_step_difference"] = float(np.mean(
        np.asarray(own_step_scores) - random_step_scores
    ))
    result["own_mean_activation"] = float(np.mean(own_means))
    result["random_mean_activation"] = float(np.mean(random_means))
    result["mean_activation_difference"] = float(np.mean(
        np.asarray(own_means) - random_means
    ))
    return result


def split_half_reliability(pose: np.ndarray, activity: np.ndarray, dones: np.ndarray) -> float | None:
    """Mean unit map correlation between episode-disjoint temporal halves."""
    ends = np.flatnonzero(np.asarray(dones, dtype=bool)) + 1
    boundaries = ends[(ends > 0) & (ends < len(pose))]
    if not len(boundaries):
        return None
    split = int(boundaries[np.argmin(abs(boundaries - len(pose) / 2))])
    first = comparison_details(pose[:split], activity[:split])
    second = comparison_details(pose[split:], activity[split:])
    shared = (first["occupancy"] >= MIN_SHARED_VISITS) & (second["occupancy"] >= MIN_SHARED_VISITS)
    if shared.sum() < MIN_SHARED_BINS:
        return None
    correlations = []
    for unit in range(activity.shape[1]):
        a = first["rate_maps"][:, :, unit][shared]
        b = second["rate_maps"][:, :, unit][shared]
        if np.std(a) > 0 and np.std(b) > 0:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(correlations)) if correlations else None
