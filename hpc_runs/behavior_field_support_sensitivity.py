"""Re-score completed 50k maps at stricter shared-bin visit thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hpc_runs.behavior_field_metrics import LAYERS, paired_information
from hpc_runs.behavior_field_study import read_manifest


VISIT_THRESHOLDS = (10, 25, 50, 100)


def load_map_details(path: Path, layer: str) -> dict[str, np.ndarray]:
    """Read the saved canonical map and eligibility for one layer."""
    with np.load(path, allow_pickle=False) as archive:
        occupancy = archive[f"{layer}_occupancy"].copy()
        rate_maps = archive[f"{layer}_rate_maps"].copy()
        eligible = archive[f"{layer}_field_eligible"].copy()
    return {"occupancy": occupancy, "rate_maps": rate_maps,
            "field_eligible": eligible}


def mask_low_visit_bins(details: dict[str, np.ndarray], visits: int) -> dict[str, np.ndarray]:
    occupancy = details["occupancy"].copy()
    occupancy[occupancy < visits] = 0
    return {**details, "occupancy": occupancy}


def analyze(manifest: Path, maps_root: Path, output: Path,
            workspace_root: Path) -> list[dict]:
    rows = read_manifest(manifest, workspace_root, require_inputs=False)
    lookup = {(r["condition"], r["seed"], r["eval_seed"], r["policy"]): r
              for r in rows}
    results = []
    for condition in ("C01", "C05", "C15"):
        for seed in ("8", "99", "123"):
            for eval_seed in ("51000", "52000"):
                own = lookup[(condition, seed, eval_seed, "own")]
                own_path = maps_root / f"{own['label']}_maps.npz"
                for policy in ("uniform", "persistent8"):
                    other = lookup[(condition, seed, eval_seed, policy)]
                    other_path = maps_root / f"{other['label']}_maps.npz"
                    for layer in LAYERS:
                        own_details = load_map_details(own_path, layer)
                        other_details = load_map_details(other_path, layer)
                        for visits in VISIT_THRESHOLDS:
                            metrics = paired_information(
                                mask_low_visit_bins(own_details, visits),
                                mask_low_visit_bins(other_details, visits),
                            )
                            results.append({"condition": condition, "seed": seed,
                                            "eval_seed": eval_seed,
                                            "random_policy": policy, "layer": layer,
                                            "min_visits_per_policy": visits, **metrics})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.manifest, args.maps_root, args.output, args.workspace_root)
    print(f"Wrote {len(rows)} sensitivity comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
