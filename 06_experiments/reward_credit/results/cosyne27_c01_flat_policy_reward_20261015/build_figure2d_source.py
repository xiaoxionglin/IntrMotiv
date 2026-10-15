"""Build comparable UG, FG, and NG Figure 2D DG source values.

All 30 terminal own-policy snapshots supply one pooled visited-bin mask. Peak
area uses the already published Gaussian field metric with the same cutoff.
The two UG/FG arms vary worker-hit credit; the NG arms vary flat policy reward.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIOR = ROOT / "06_experiments/reward_credit/results/cosyne27_worker_regimen_boxplots_20261013"
sys.path.insert(0, str(ROOT))

from hpc_runs.intrmotiv_study import load_study  # noqa: E402


PEAK_METRIC = importlib.import_module("06_experiments.cosyne_2027.analyze_snapshot_peak_spread")
STUDY = ROOT / "hpc_runs/studies/cosyne27_c01_flat_reward_s5_20261015.study.json"
PRIOR_INPUTS = PRIOR / "high_credit_peak_snapshot_inputs.json"
PRIOR_PAIRS = PRIOR / "high_credit_representation_pairs.csv"
NG_INPUTS = HERE / "ng_peak_snapshot_inputs.json"
NG_SPATIAL = HERE / "ng_spatial_per_snapshot.csv"
NG_MANIFEST = HERE / "ng_spatial_analysis_manifest.json"
ARM_OUTPUT = HERE / "figure2d_shared_mask_arm_values.csv"
PAIR_OUTPUT = HERE / "figure2d_shared_mask_pairs.csv"
PROVENANCE_OUTPUT = HERE / "figure2d_shared_mask_provenance.json"
SEEDS = (8, 17, 41, 99, 123)
ARMS = ("TEMPORAL", "CONSTANT")
FAMILIES = ("C05", "C15", "C01")
DISPLAY = {"C05": "UG", "C15": "FG", "C01": "NG"}
CUTOFF = 0.0025


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed_records(path: Path, arm_key: str) -> dict[tuple[str, int, str], dict]:
    records = json.loads(path.read_text())["snapshots"]
    index = {(row["family"], int(row["seed"]), row[arm_key]): row
             for row in records}
    if len(index) != len(records):
        raise ValueError(f"Duplicate peak input identity in {path}")
    return index


def shared_mask(records: list[dict]) -> tuple[np.ndarray, str]:
    masks = []
    for record in records:
        bits = record["visited_mask_bits"]
        if len(bits) != 361 or set(bits) - {"0", "1"}:
            raise ValueError(f"Invalid visited-bin mask: {record['run_name']}")
        mask = np.fromiter((bit == "1" for bit in bits), bool, count=361).reshape(19, 19)
        if int(mask.sum()) != int(record["visited_bin_count"]):
            raise ValueError(f"Visited-bin count mismatch: {record['run_name']}")
        masks.append(mask)
    pooled = np.logical_or.reduce(masks)
    return pooled, hashlib.sha256(pooled.astype(np.uint8).tobytes()).hexdigest()


def peak_area(record: dict, mask: np.ndarray) -> tuple[float, int]:
    fractions = np.asarray(record["active_fraction"], dtype=float)
    eligible = np.asarray(record["field_eligible"], dtype=bool)
    coords = np.asarray([[math.nan if value is None else value for value in pair]
                         for pair in record["field_dominant_peak_xy"]], dtype=float)
    if (fractions.shape != (16,) or eligible.shape != (16,) or coords.shape != (16, 2)
            or not np.isfinite(fractions).all()):
        raise ValueError(f"Invalid DG peak input: {record['run_name']}")
    selected = eligible & (fractions >= CUTOFF) & np.isfinite(coords).all(axis=1)
    xy = coords[selected]
    if len(xy) < 2 or ((xy < 100) | (xy >= 2000)).any():
        raise ValueError(f"Insufficient bounded eligible peaks: {record['run_name']}")
    area = PEAK_METRIC.spread_metrics(xy, mask, PEAK_METRIC.SIGMA)["effective_area_fraction"]
    if not math.isfinite(area) or not 0 < area <= 1:
        raise ValueError(f"Invalid effective peak area: {record['run_name']}")
    return area, len(xy)


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    study = load_study(STUDY)
    ng_manifest = json.loads(NG_MANIFEST.read_text())
    ng_inputs = json.loads(NG_INPUTS.read_text())
    if (not ng_manifest["complete"] or ng_manifest["observed_snapshots"] != 50
            or ng_manifest["study_sha256"] != study.fingerprint
            or ng_inputs["study_sha256"] != study.fingerprint):
        raise ValueError("NG canonical collection or peak archive is incomplete")

    prior = indexed_records(PRIOR_INPUTS, "worker_bonus")
    ng = indexed_records(NG_INPUTS, "arm")
    expected = {(family, seed, arm) for family in FAMILIES
                for seed in SEEDS for arm in ARMS}
    records = {**prior, **ng}
    if len(records) != 30 or set(records) != expected:
        raise ValueError("Expected 30 distinct UG/FG/NG terminal snapshots")
    mask, mask_hash = shared_mask(list(records.values()))

    prior_pairs = pd.read_csv(PRIOR_PAIRS).set_index(["family", "seed"])
    if len(prior_pairs) != 10:
        raise ValueError("Expected ten pinned UG/FG pairs")
    ng_spatial = pd.read_csv(NG_SPATIAL)
    ng_spatial = ng_spatial[ng_spatial.target_env_steps == 100_000_000]
    if len(ng_spatial) != 10:
        raise ValueError("Expected ten NG terminal spatial rows")
    ng_spatial = ng_spatial.set_index(["seed", "flat_policy_reward"])

    arms = []
    for family in FAMILIES:
        for seed in SEEDS:
            for arm in ARMS:
                record = records[(family, seed, arm)]
                area, count = peak_area(record, mask)
                if family == "C01":
                    source = ng_spatial.loc[(seed, arm)]
                    if (str(source.run_name) != record["run_name"]
                            or str(source.snapshot_path) != record["snapshot_path"]
                            or int(source.window_limit) != 100_000
                            or int(source.graph_available) != 0):
                        raise ValueError(f"NG source mismatch: {record['run_name']}")
                    info = float(source.active_unit_mean_spatial_information)
                    cosine = float(source.active_only_map_cosine)
                    visited = float(source.visited_cell_fraction)
                    factor = "flat_policy_reward"
                    source_id = study.fingerprint
                else:
                    source = prior_pairs.loc[(family, seed)]
                    prefix = arm.lower()
                    if (str(source[f"{prefix}_snapshot_sha256"])
                            != record["snapshot_sha256"]):
                        raise ValueError(f"UG/FG snapshot hash mismatch: {record['run_name']}")
                    if not math.isclose(area, float(source[f"{prefix}_effective_peak_area_fraction"]),
                                        abs_tol=1e-12):
                        raise ValueError(f"UG/FG peak area changed: {record['run_name']}")
                    info = float(source[f"{prefix}_active_unit_mean_spatial_information"])
                    cosine = float(source[f"{prefix}_active_only_map_cosine"])
                    visited = float(source[f"{prefix}_visited_cell_fraction"])
                    factor = "worker_hit_bonus"
                    source_id = str(source.source_commit)
                values = (info, cosine, visited, area)
                if not all(map(math.isfinite, values)):
                    raise ValueError(f"Nonfinite DG metric: {record['run_name']}")
                arms.append({
                    "family": family, "display_name": DISPLAY[family], "seed": seed,
                    "arm": arm, "intervention_factor": factor,
                    "run_name": record["run_name"], "source_id": source_id,
                    "snapshot_sha256": record["snapshot_sha256"],
                    "target_env_steps": 100_000_000,
                    "snapshot_decisions": 100_000,
                    "active_unit_mean_spatial_information_bits": info,
                    "active_only_map_cosine": cosine,
                    "eligible_peak_area_fraction": area,
                    "eligible_peak_count": count,
                    "visited_cell_fraction": visited,
                    "shared_mask_bins": int(mask.sum()),
                    "shared_mask_sha256": mask_hash,
                    "graph_edges_applicable": family != "C01",
                })
    arm_frame = pd.DataFrame(arms)

    pair_rows = []
    metrics = ("active_unit_mean_spatial_information_bits", "active_only_map_cosine",
               "eligible_peak_area_fraction", "visited_cell_fraction")
    for (family, seed), frame in arm_frame.groupby(["family", "seed"], sort=False):
        indexed = frame.set_index("arm")
        row = {"family": family, "display_name": DISPLAY[family], "seed": seed,
               "intervention_factor": indexed.loc["TEMPORAL", "intervention_factor"],
               "temporal_run_name": indexed.loc["TEMPORAL", "run_name"],
               "constant_run_name": indexed.loc["CONSTANT", "run_name"],
               "temporal_snapshot_sha256": indexed.loc["TEMPORAL", "snapshot_sha256"],
               "constant_snapshot_sha256": indexed.loc["CONSTANT", "snapshot_sha256"],
               "source_id": indexed.loc["TEMPORAL", "source_id"],
               "shared_mask_bins": int(mask.sum()), "shared_mask_sha256": mask_hash,
               "graph_edges_applicable": family != "C01"}
        for metric in metrics:
            temporal = float(indexed.loc["TEMPORAL", metric])
            constant = float(indexed.loc["CONSTANT", metric])
            row[f"temporal_{metric}"] = temporal
            row[f"constant_{metric}"] = constant
            row[f"delta_{metric}"] = temporal - constant
        for arm in ARMS:
            row[f"{arm.lower()}_eligible_peak_count"] = int(indexed.loc[arm, "eligible_peak_count"])
        pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)
    provenance = {
        "schema": "intrmotiv/figure2d-shared-mask-source/v1",
        "ng_study_schema": study.provenance()["schema"],
        "ng_workflow_version": study.declared_workflow_version,
        "ng_study_sha256": study.fingerprint,
        "ug_fg_source_commit": "a26852f",
        "protocol": "own-policy 100000-decision terminal snapshots at 100M frames; matched training seeds, different observations",
        "shared_mask_definition": "union of occupied bins across all 30 terminal UG, FG, and NG snapshots",
        "shared_mask_bins": int(mask.sum()),
        "shared_mask_sha256": mask_hash,
        "peak_activity_cutoff_fraction": CUTOFF,
        "field_min_active_observations": 20,
        "field_min_active_bins": 3,
        "gaussian_sigma_dmlab_units": PEAK_METRIC.SIGMA,
        "peak_area_definition": "exp(Shannon entropy of equal-mass eligible-peak Gaussian density) / shared_mask_bins",
        "graph_edges_for_ng": "inapplicable: no commanded DG goals or goal graph",
        "input_sha256": {path.name: sha256(path) for path in
                         (PRIOR_INPUTS, PRIOR_PAIRS, NG_INPUTS, NG_SPATIAL, NG_MANIFEST)},
    }
    arm_frame.to_csv(ARM_OUTPUT, index=False, float_format="%.15g")
    pair_frame.to_csv(PAIR_OUTPUT, index=False, float_format="%.15g")
    PROVENANCE_OUTPUT.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return arm_frame, pair_frame, provenance


if __name__ == "__main__":
    arm_values, pairs, source = build()
    print(f"{len(arm_values)} arms, {len(pairs)} seed pairs, "
          f"{source['shared_mask_bins']} shared visited bins")
