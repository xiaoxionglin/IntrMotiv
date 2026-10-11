"""Measure DG peak geometry on one identical observation panel.

Raw maps remain in the NEMO2 analysis workspace. The emitted per-unit rows,
common support mask, run summaries, and paired differences reproduce every
peak-location and Gaussian coverage figure without another model replay.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CUTOFFS = (0.0, 0.00025, 0.0025, 0.01)
BANDWIDTHS = (0.5, 1.0, 2.0)  # 19-by-19 map bins


def read_table(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No output rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    total = ordered.sum()
    if total <= 0:
        return float("nan")
    n = len(ordered)
    return float(np.dot(2 * np.arange(1, n + 1) - n - 1, ordered) / (n * total))


def spread(peaks: np.ndarray, support: np.ndarray, sigma: float) -> dict[str, float]:
    if len(peaks) == 0:
        return {"coverage_entropy_normalized": np.nan,
                "coverage_effective_area_fraction": np.nan,
                "coverage_gini": np.nan}
    grid = np.indices(support.shape, dtype=float)
    density = np.zeros(support.shape, dtype=float)
    for row, col in peaks:
        density += np.exp(-((grid[0] - row) ** 2 + (grid[1] - col) ** 2) / (2 * sigma**2))
    values = density[support]
    probabilities = values / values.sum()
    positive = probabilities > 0
    entropy = float(-np.dot(probabilities[positive], np.log(probabilities[positive])))
    n = len(values)
    return {"coverage_entropy_normalized": entropy / np.log(n),
            "coverage_effective_area_fraction": float(np.exp(entropy) / n),
            "coverage_gini": gini(values)}


def geometry(peaks: np.ndarray) -> tuple[float, float]:
    if len(peaks) < 2:
        return np.nan, np.nan
    distances = np.linalg.norm(peaks[:, None, :] - peaks[None, :, :], axis=-1)
    triangular = distances[np.triu_indices(len(peaks), k=1)]
    np.fill_diagonal(distances, np.inf)
    return float(triangular.mean()), float(distances.min(axis=1).mean())


def map_cosine(maps: np.ndarray) -> float:
    if maps.shape[1] < 2:
        return np.nan
    norms = np.linalg.norm(maps, axis=0)
    normalized = np.divide(maps, norms, out=np.zeros_like(maps), where=norms > 0)
    cosine = normalized.T @ normalized
    return float(cosine[np.triu_indices(maps.shape[1], k=1)].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_table(args.manifest, delimiter="\t")
    sources = {row["label_suffix"]: row for row in read_table(args.sources)}
    unit_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    reference_support = None
    reference_occupancy = None
    for record in manifest:
        label = record["label_suffix"]
        artifacts = list(args.raw_dir.glob(f"*__{label}/place_fields.npz"))
        if len(artifacts) != 1:
            raise ValueError(f"Expected exactly one replay for {label}, found {len(artifacts)}")
        with np.load(artifacts[0], allow_pickle=False) as data:
            occupancy = np.asarray(data["occupancy"])
            maps = np.asarray(data["rate_maps"], dtype=float)
            fractions = np.asarray(data["active_fraction"], dtype=float)
            information = np.asarray(data["spatial_information"], dtype=float)
            eligible = np.asarray(data["post_inhibition_field_eligible"], dtype=bool)
            mono = np.asarray(data["post_inhibition_field_mono"], dtype=bool)
        support = occupancy > 0
        if reference_support is None:
            reference_support = support.copy()
            reference_occupancy = occupancy.copy()
        elif not np.array_equal(occupancy, reference_occupancy):
            raise ValueError(f"Observation occupancy differs for {label}")
        if maps.shape[:2] != support.shape or maps.shape[2] != len(fractions):
            raise ValueError(f"Map shape mismatch for {label}")
        values = maps[support]
        maxima = np.nanmax(values, axis=0)
        valid_rate = np.isfinite(maxima) & (maxima > 0)
        masked = np.where(support[:, :, None], maps, -np.inf)
        peak_flat = np.argmax(masked.reshape(-1, maps.shape[2]), axis=0)
        peaks = np.column_stack(np.unravel_index(peak_flat, support.shape))
        mean_rate = np.dot(occupancy[support] / occupancy[support].sum(), values)
        information_per_activation = np.divide(
            information, mean_rate,
            out=np.full_like(information, np.nan), where=mean_rate > 0,
        )
        source = sources[label]
        common = {"label_suffix": label, "run_name": source["run_name"],
                  "condition": record["condition"], "family": record["family"],
                  "seed": int(record["seed"]), "study_id": source["study_id"],
                  "study_sha256": source["study_sha256"],
                  "recovery_patch_sha256": source["recovery_patch_sha256"],
                  "checkpoint_frames": int(record["checkpoint_frames"])}
        for unit, (row, col) in enumerate(peaks):
            unit_rows.append({**common, "unit_id": unit,
                              "peak_row": int(row) if valid_rate[unit] else "",
                              "peak_col": int(col) if valid_rate[unit] else "",
                              "peak_x_bin": int(row) if valid_rate[unit] else "",
                              "peak_y_bin": int(col) if valid_rate[unit] else "",
                              "peak_rate": float(maxima[unit]),
                              "mean_rate": float(mean_rate[unit]),
                              "active_fraction": float(fractions[unit]),
                              "information_amplitude_weighted": float(information[unit]),
                              "information_per_activation_bits": float(information_per_activation[unit]),
                              "field_eligible": int(eligible[unit]),
                              "field_mono": int(mono[unit]),
                              **{f"valid_at_{cutoff:g}": int(bool(valid_rate[unit] and fractions[unit] > cutoff))
                                 for cutoff in CUTOFFS}})
        for cutoff in CUTOFFS:
            selected = valid_rate & (fractions > cutoff)
            active_peaks = peaks[selected]
            pairwise, nearest = geometry(active_peaks)
            active_values = values[:, selected]
            for bandwidth in BANDWIDTHS:
                run_rows.append({**common, "activity_cutoff_fraction": cutoff,
                                 "gaussian_sigma_bins": bandwidth,
                                 "observation_count": int(occupancy.sum()),
                                 "supported_bins": int(support.sum()),
                                 "dg_capacity": maps.shape[2],
                                 "active_units": int(selected.sum()),
                                 "silent_or_below_cutoff_units": int((~selected).sum()),
                                 "unique_peak_bins": len(set(map(tuple, active_peaks))),
                                 "mean_pairwise_peak_distance_bins": pairwise,
                                 "mean_nearest_neighbor_peak_distance_bins": nearest,
                                 **spread(active_peaks, support, bandwidth),
                                 "active_only_map_cosine": map_cosine(active_values),
                                 "mean_active_fraction": float(fractions[selected].mean()) if selected.any() else np.nan,
                                 "mean_peak_rate": float(maxima[selected].mean()) if selected.any() else np.nan,
                                 "mean_information_amplitude_weighted": float(information[selected].mean()) if selected.any() else np.nan,
                                 "mean_information_per_activation_bits": float(information_per_activation[selected].mean()) if selected.any() else np.nan,
                                 "field_eligible_units": int((eligible & selected).sum()),
                                 "mono_field_fraction_of_eligible": float(mono[eligible & selected].mean())
                                 if (eligible & selected).any() else np.nan})
        print(label, "active @0.25%", int((valid_rate & (fractions > .0025)).sum()), flush=True)
    write_table(args.output / "peak_units.csv", unit_rows)
    write_table(args.output / "peak_run_sensitivity.csv", run_rows)
    np.savetxt(args.output / "common_support_mask.csv", reference_support.astype(int), fmt="%d", delimiter=",")
    print("completed", len(manifest), "replays; supported bins", int(reference_support.sum()))


if __name__ == "__main__":
    main()
