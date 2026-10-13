"""Derive paired DG map overlap and peak coverage from completed frozen probes.

This is a thin adapter over the established behavior-field map builder and
COSYNE peak-spread metric. Run it on a compute node against the archived 50k
traces; it never starts an environment or loads a training checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hpc_runs.behavior_field_metrics import comparison_details


field_metrics = importlib.import_module("06_experiments.analyze_place_field_manifest")
peak_metrics = importlib.import_module(
    "06_experiments.cosyne_2027.analyze_snapshot_peak_spread"
)

MIN_VISITS = 10
ACTIVITY_CUTOFF = 0.0025
EVAL_SEEDS = {51_000, 52_000}
POLICIES = {"own", "uniform"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace(folder: Path, policy: str, eval_seed: int) -> tuple[dict, dict]:
    metadata_path = folder / "metadata.json"
    trace_path = folder / "trace.npz"
    metadata = json.loads(metadata_path.read_text())
    if (metadata["schema"] != "intrmotiv/behavior-field-probe/v1" or
            metadata["policy"] != policy or metadata["eval_seed"] != eval_seed or
            metadata["decisions"] != 50_000 or metadata["layer_widths"]["dg"] != 16):
        raise ValueError(f"Unexpected probe metadata: {folder.name}")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in ("pose", "dg")}
    if trace["pose"].shape != (50_000, 3) or trace["dg"].shape != (50_000, 16):
        raise ValueError(f"Unexpected trace shape: {folder.name}")
    return trace, {"metadata_sha256": sha256(metadata_path),
                   "trace_sha256": sha256(trace_path),
                   "checkpoint_sha256": metadata["checkpoint_sha256"],
                   "frozen_model_sha256": metadata["frozen_model_sha256"]}


def peak_coordinates(maps: np.ndarray, units: np.ndarray,
                     common_mask: np.ndarray) -> np.ndarray:
    """Take each selected unit's largest rate on the same spatial support."""
    coordinates = []
    for unit in np.flatnonzero(units):
        values = np.where(common_mask, maps[:, :, unit], -np.inf)
        y, x = np.unravel_index(int(np.argmax(values)), (19, 19))
        if not np.isfinite(values[y, x]) or values[y, x] <= 0:
            raise ValueError(f"Selected DG unit {unit} has no positive shared peak")
        coordinates.append((150 + 100 * x, 150 + 100 * y))
    return np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)


def pair_geometry(own: dict, random: dict) -> dict[str, float | int]:
    own_maps = comparison_details(own["pose"], own["dg"])
    random_maps = comparison_details(random["pose"], random["dg"])
    common = ((own_maps["occupancy"] >= MIN_VISITS) &
              (random_maps["occupancy"] >= MIN_VISITS))
    eligible = (own_maps["field_eligible"] & random_maps["field_eligible"])
    if common.sum() < 20 or eligible.sum() < 2:
        raise ValueError("Insufficient shared bins or jointly eligible DG units")

    # Cosine uses jointly eligible units, without the peak cutoff. The shared
    # metric excludes any pair containing a zero-norm map on common support.
    own_cosine = field_metrics.pairwise_map_cosine(
        own_maps["rate_maps"], common, np.flatnonzero(eligible))
    random_cosine = field_metrics.pairwise_map_cosine(
        random_maps["rate_maps"], common, np.flatnonzero(eligible))

    # The field-area contrast retains exactly the same DG identities and mask
    # in both arms, requiring the 0.25% activity cutoff in both 50k probes.
    own_fraction = (own["dg"] > 0).mean(axis=0)
    random_fraction = (random["dg"] > 0).mean(axis=0)
    peak_units = eligible & (own_fraction >= ACTIVITY_CUTOFF) & (
        random_fraction >= ACTIVITY_CUTOFF)
    if peak_units.sum() < 2:
        raise ValueError("Fewer than two jointly qualified DG peaks")
    own_peaks = peak_coordinates(own_maps["rate_maps"], peak_units, common)
    random_peaks = peak_coordinates(random_maps["rate_maps"], peak_units, common)
    own_area = peak_metrics.spread_metrics(own_peaks, common)["effective_area_fraction"]
    random_area = peak_metrics.spread_metrics(random_peaks, common)["effective_area_fraction"]
    if not np.isfinite([own_cosine, random_cosine, own_area, random_area]).all():
        raise ValueError("Nonfinite behavior geometry metric")
    return {
        "shared_bins": int(common.sum()),
        "jointly_eligible_units": int(eligible.sum()),
        "jointly_qualified_peaks": int(peak_units.sum()),
        "own_map_cosine": float(own_cosine),
        "random_map_cosine": float(random_cosine),
        "map_cosine_difference": float(own_cosine - random_cosine),
        "own_peak_coverage": float(own_area),
        "random_peak_coverage": float(random_area),
        "peak_coverage_difference": float(own_area - random_area),
    }


def collect(provenance: Path, raw_root: Path, output: Path) -> None:
    with provenance.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["policy"] in POLICIES]
    if len(rows) != 152 or len({row["label"] for row in rows}) != 152:
        raise ValueError("Expected 38 models × 2 repeats × 2 policies")
    keyed = {(row["run_name"], int(row["eval_seed"]), row["policy"]): row
             for row in rows}
    models = sorted({row["run_name"] for row in rows})
    if len(keyed) != 152 or len(models) != 38:
        raise ValueError("Duplicated or incomplete frozen-probe identity")

    records = []
    input_records = []
    for model in models:
        for eval_seed in sorted(EVAL_SEEDS):
            pair = {policy: keyed[(model, eval_seed, policy)] for policy in POLICIES}
            traces = {}
            identities = {}
            for policy in sorted(POLICIES):
                row = pair[policy]
                traces[policy], identities[policy] = load_trace(
                    raw_root / row["label"], policy, eval_seed)
                input_records.append({"label": row["label"], "run_name": model,
                                      "eval_seed": eval_seed, "policy": policy,
                                      **identities[policy]})
            if (identities["own"]["checkpoint_sha256"] !=
                    identities["uniform"]["checkpoint_sha256"] or
                    identities["own"]["frozen_model_sha256"] !=
                    identities["uniform"]["frozen_model_sha256"]):
                raise ValueError(f"Frozen-model mismatch: {model}, {eval_seed}")
            own = pair["own"]
            random = pair["uniform"]
            factor_keys = ("family", "training_seed", "encoder_credit",
                           "worker_bonus", "study_id", "study_sha256")
            if any(own[key] != random[key] for key in factor_keys):
                raise ValueError(f"Factor mismatch: {model}, {eval_seed}")
            records.append({"condition": model.rsplit("_S", 1)[0],
                            **{key: own[key] for key in factor_keys},
                            "eval_seed": eval_seed,
                            "checkpoint_sha256": identities["own"]["checkpoint_sha256"],
                            "frozen_model_sha256": identities["own"]["frozen_model_sha256"],
                            **pair_geometry(traces["own"], traces["uniform"])})
    repeats = pd.DataFrame(records).sort_values(["family", "condition",
                                                  "training_seed", "eval_seed"])
    if len(repeats) != 76 or repeats.groupby(["condition", "training_seed"]).size().ne(2).any():
        raise ValueError("Incomplete paired evaluation repeats")
    group_keys = ["condition", "family", "training_seed", "encoder_credit",
                  "worker_bonus", "study_id", "study_sha256"]
    model_rows = repeats.groupby(group_keys, as_index=False)[[
        "shared_bins", "jointly_eligible_units", "jointly_qualified_peaks",
        "own_map_cosine", "random_map_cosine", "map_cosine_difference",
        "own_peak_coverage", "random_peak_coverage", "peak_coverage_difference",
    ]].mean()
    if model_rows.groupby("family").size().to_dict() != {"C01": 10, "C05": 20, "C15": 8}:
        raise ValueError("Unexpected model cohort")
    output.mkdir(parents=True, exist_ok=True)
    repeats.to_csv(output / "behavior_geometry_by_repeat.csv", index=False)
    model_rows.to_csv(output / "behavior_geometry_by_model.csv", index=False)
    pd.DataFrame(input_records).to_csv(output / "behavior_geometry_inputs.csv", index=False)
    (output / "behavior_geometry_manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/figure2-frozen-behavior-geometry/v1",
        "provenance_sha256": sha256(provenance),
        "raw_probes": len(input_records), "paired_repeats": len(repeats),
        "complete_models": len(model_rows),
        "comparison": "own policy minus uniform random; two repeats averaged within model",
        "spatial_support": "bins with >=10 visits in both policies, separately for each paired repeat",
        "dg_units": "same units field-eligible in both arms (>=20 active observations in >=3 bins)",
        "map_cosine": "mean over distinct pairs of jointly eligible DG rate maps on common bins; zero-norm map pairs excluded",
        "peak_coverage": "equal-mass Gaussian density (sigma=100 DMLab units) of dominant shared-bin peaks; jointly eligible units with >=0.25% active decisions in both arms; exp(entropy)/common-bin count",
        "source_code_sha256": {name: sha256(Path(module.__file__)) for name, module in (
            ("behavior_field_metrics", importlib.import_module("hpc_runs.behavior_field_metrics")),
            ("pairwise_map_cosine", field_metrics),
            ("peak_spread", peak_metrics),
        )},
    }, indent=2) + "\n")
    print(f"Saved {len(model_rows)} models, {len(repeats)} paired repeats")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.provenance, args.raw_root, args.output)


if __name__ == "__main__":
    main()
