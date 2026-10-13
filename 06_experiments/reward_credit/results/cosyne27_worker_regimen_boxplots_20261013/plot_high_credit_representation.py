"""Render five paired high-credit worker representation measures.

The Gaussian area calculation reuses the original COSYNE peak-spread metric
and the lightweight inputs archived from the same 20 own-policy snapshots.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_high_credit_compact import (
    COLOR, HERE, ROOT, SEEDS, SOURCE_COMMIT, high_credit_data,
    save, source_table, style,
)


sys.path.insert(0, str(ROOT))
peak_metric = importlib.import_module(
    "06_experiments.cosyne_2027.analyze_snapshot_peak_spread")

PEAK_INPUTS = HERE / "high_credit_peak_snapshot_inputs.json"
TABLE = HERE / "high_credit_representation_pairs.csv"
OUTPUT_STEM = "high_credit_worker_representation_compact"
CARD_DIRECTORY = HERE / "representation_cards"
CARD_WIDTH, CARD_HEIGHT = 2.25, 1.95
PLOT_WIDTH, PLOT_HEIGHT = 1.20, 1.00
DATA_LEFT, DATA_BOTTOM, DATA_WIDTH, DATA_HEIGHT = .93, .47, .695, .66
PEAK_CUTOFF = .0025
DISPLAY_NAMES = {"C05": "UG", "C15": "FG"}
METRICS = (
    ("active_only_map_cosine", "A  DG map overlap", "cosine", "map_overlap"),
    ("effective_peak_area_fraction", "B  Field-peak coverage", "area fraction", "peak_coverage"),
    ("unique_active_peak_bins", "C  Distinct peak bins", "bins", "distinct_peak_bins"),
    ("mono_field_unit_fraction", "D  Single fields", "fraction", "single_fields"),
    ("cutoff_median_nearest_peak_distance", "E  Peak spacing", "DMLab units", "peak_spacing"),
)


def load_peak_inputs() -> tuple[list[dict], np.ndarray]:
    """Validate the archived 20 snapshots and return their common spatial mask."""
    if not PEAK_INPUTS.is_file():
        raise FileNotFoundError(f"Missing {PEAK_INPUTS}; collect the peak inputs first")
    source = json.loads(PEAK_INPUTS.read_text())
    if (source["schema"] != "intrmotiv/high-credit-peak-inputs/v1" or
            source["source_commit"] != SOURCE_COMMIT):
        raise ValueError("Peak-input schema or pinned source commit changed")
    records = source["snapshots"]
    expected = {(family, seed, arm) for family in ("C05", "C15")
                for seed in SEEDS for arm in ("TEMPORAL", "CONSTANT")}
    actual = {(row["family"], row["seed"], row["worker_bonus"])
              for row in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("Expected exactly 20 high-credit terminal peak inputs")
    masks = []
    for record in records:
        bits = record["visited_mask_bits"]
        if len(bits) != 361 or set(bits) - {"0", "1"}:
            raise ValueError("Invalid 19-by-19 visited-bin mask")
        masks.append(np.fromiter((bit == "1" for bit in bits),
                                 dtype=bool, count=361).reshape(19, 19))
    shared_mask = np.logical_or.reduce(masks)
    return records, shared_mask


def selected_peak_coordinates(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage-qualified peaks and eligible peaks below its activity cutoff."""
    fractions = np.asarray(record["active_fraction"], dtype=float)
    eligible = np.asarray(record["field_eligible"], dtype=bool)
    coordinates = np.array([[np.nan if value is None else value for value in pair]
                            for pair in record["field_dominant_peak_xy"]], dtype=float)
    if (fractions.shape != (16,) or eligible.shape != (16,) or
            coordinates.shape != (16, 2) or not np.isfinite(fractions).all()):
        raise ValueError(f"Invalid DG-unit input in {record['run_name']}")
    finite_eligible = eligible & np.isfinite(coordinates).all(axis=1)
    qualified = finite_eligible & (fractions >= PEAK_CUTOFF)
    selected = coordinates[qualified]
    excluded = coordinates[finite_eligible & ~qualified]
    if len(selected) < 2 or ((selected < 100) | (selected >= 2000)).any():
        raise ValueError(f"Insufficient bounded peaks in {record['run_name']}")
    return selected, excluded


def peak_area_data() -> tuple[pd.DataFrame, dict]:
    """Apply the canonical area and nearest-distance metrics to the same peaks."""
    records, shared_mask = load_peak_inputs()
    mask_hash = hashlib.sha256(shared_mask.astype(np.uint8).tobytes()).hexdigest()
    rows = []
    for record in records:
        xy, excluded = selected_peak_coordinates(record)
        metrics = peak_metric.spread_metrics(xy, shared_mask, peak_metric.SIGMA)
        uncut = peak_metric.spread_metrics(
            np.concatenate((xy, excluded)), shared_mask,
            peak_metric.SIGMA)["median_nearest_peak_distance"]
        area = metrics["effective_area_fraction"]
        nearest = metrics["median_nearest_peak_distance"]
        if not np.isfinite([area, nearest, uncut]).all() or not (0 < area <= 1):
            raise ValueError(f"Invalid peak geometry in {record['run_name']}")
        rows.append({"family": record["family"], "seed": record["seed"],
                     "worker_bonus": record["worker_bonus"],
                     "effective_peak_area_fraction": area,
                     "cutoff_median_nearest_peak_distance": nearest,
                     "uncut_median_nearest_peak_distance": uncut,
                     "eligible_peak_count": int(len(xy)),
                     "snapshot_sha256": record["snapshot_sha256"]})
    provenance = {
        "mask_definition": "union of occupied bins across all 20 high-credit 100M snapshots",
        "shared_mask_bins": int(shared_mask.sum()),
        "shared_mask_sha256": mask_hash,
        "activity_cutoff_fraction": PEAK_CUTOFF,
        "minimum_active_observations": 20,
        "minimum_active_bins": 3,
        "gaussian_sigma_dmlab_units": peak_metric.SIGMA,
        "area_definition": "exp(Shannon entropy of equal-mass eligible-peak Gaussian density) / shared_mask_bins",
        "peak_inputs_sha256": hashlib.sha256(PEAK_INPUTS.read_bytes()).hexdigest(),
    }
    return pd.DataFrame(rows), provenance


def paired_data() -> tuple[pd.DataFrame, dict]:
    """Preserve both arms and matched differences for all five diagnostics."""
    area, provenance = peak_area_data()
    area = area.set_index(["family", "seed", "worker_bonus"])
    rows = []
    expected = {(seed, arm) for seed in SEEDS for arm in ("TEMPORAL", "CONSTANT")}
    source_columns = (
        "active_unit_mean_spatial_information", "active_only_map_cosine",
        "unique_active_peak_bins", "mono_field_unit_fraction",
        "median_dominant_peak_nearest_neighbor_distance", "visited_cell_fraction",
    )
    existing = high_credit_data().set_index(["family", "seed"])
    for family in ("C05", "C15"):
        source = source_table(f"{family.lower()}_spatial_per_snapshot.csv")
        source = source[source.target_env_steps == 100_000_000]
        if len(source) != len(expected) or set(zip(source.seed, source.worker_bonus)) != expected:
            raise ValueError(f"Incomplete 100M snapshots for {family}")
        indexed = source.set_index(["seed", "worker_bonus"])
        for seed in SEEDS:
            temporal = indexed.loc[(seed, "TEMPORAL")]
            constant = indexed.loc[(seed, "CONSTANT")]
            row = {"family": family, "seed": seed, "source_commit": SOURCE_COMMIT,
                   "snapshot_target_frames": 100_000_000,
                   "snapshot_decisions": int(temporal.window_limit),
                   "peak_mask_bins": provenance["shared_mask_bins"],
                   "peak_mask_sha256": provenance["shared_mask_sha256"]}
            for metric in source_columns:
                t, c = float(temporal[metric]), float(constant[metric])
                if not np.isfinite([t, c]).all():
                    raise ValueError(f"Non-finite {metric} for {family} seed {seed}")
                row[f"temporal_{metric}"] = t
                row[f"constant_{metric}"] = c
                row[f"delta_{metric}"] = t - c
            for arm, label in (("TEMPORAL", "temporal"), ("CONSTANT", "constant")):
                peak = area.loc[(family, seed, arm)]
                if not np.isclose(
                        float(peak.uncut_median_nearest_peak_distance),
                        float(indexed.loc[(seed, arm),
                                          "median_dominant_peak_nearest_neighbor_distance"]),
                        atol=1e-4):
                    raise ValueError(f"Uncut peak-spacing source mismatch for {family} seed {seed} {arm}")
                row[f"{label}_effective_peak_area_fraction"] = float(
                    peak.effective_peak_area_fraction)
                row[f"{label}_cutoff_median_nearest_peak_distance"] = float(
                    peak.cutoff_median_nearest_peak_distance)
                row[f"{label}_eligible_peak_count"] = int(peak.eligible_peak_count)
                row[f"{label}_snapshot_sha256"] = str(peak.snapshot_sha256)
            row["delta_effective_peak_area_fraction"] = (
                row["temporal_effective_peak_area_fraction"] -
                row["constant_effective_peak_area_fraction"])
            row["delta_cutoff_median_nearest_peak_distance"] = (
                row["temporal_cutoff_median_nearest_peak_distance"] -
                row["constant_cutoff_median_nearest_peak_distance"])
            expected_si = existing.loc[(family, seed), "delta_spatial_information_bits"]
            if not np.isclose(row["delta_active_unit_mean_spatial_information"],
                              expected_si, atol=1e-6):
                raise ValueError(f"Spatial-information source mismatch for {family} seed {seed}")
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TABLE, index=False)
    return result, provenance


def draw_card(fig: plt.Figure, data: pd.DataFrame, metric: str,
              title: str, unit: str, x: float, y: float) -> tuple[plt.Axes, plt.Text]:
    """Use the loose-canvas card dimensions, text, alpha, and summary layers."""
    width, height = fig.get_figwidth(), fig.get_figheight()
    words, lines, current = title.split(), [], []
    for word in words:
        if len(" ".join(current + [word])) > 20 and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    lines.append(" ".join(current))
    label = fig.text((x + .045 * CARD_WIDTH) / width,
                     (y + CARD_HEIGHT - .05) / height,
                     "\n".join(lines), ha="left", va="top", fontsize=12,
                     fontweight="bold", color="#20272C")
    ax = fig.add_axes([(x + DATA_LEFT) / width, (y + DATA_BOTTOM) / height,
                       DATA_WIDTH / width, DATA_HEIGHT / height])
    all_values = []
    for position, family in enumerate(("C05", "C15")):
        values = (data[data.family == family].set_index("seed")
                  .loc[list(SEEDS), f"delta_{metric}"].to_numpy(dtype=float))
        all_values.extend(values)
        color = COLOR[family]
        ax.scatter(np.full(len(values), position) + np.linspace(-.075, .075, len(values)),
                   values, s=15, color=color, alpha=.7, edgecolors="none", zorder=2)
        ax.boxplot([values], positions=[position], widths=.28, patch_artist=True,
                   showfliers=False, whis=(0, 100), manage_ticks=False, zorder=4,
                   boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1},
                   whiskerprops={"color": color, "linewidth": 1},
                   capprops={"color": color, "linewidth": 1},
                   medianprops={"color": color, "linewidth": 1.3})
        ax.scatter([position], [values.mean()], marker="D", s=20,
                   facecolors="white", edgecolors=color, linewidths=1, zorder=6)
    extent = max(abs(value) for value in all_values) * 1.13
    ax.set_ylim(-extent, extent)
    ax.set_xlim(-.55, 1.55)
    ax.axhline(0, color="#7D858A", lw=.7, zorder=1)
    ax.set_xticks((0, 1), (DISPLAY_NAMES["C05"], DISPLAY_NAMES["C15"]))
    ax.tick_params(axis="x", labelsize=8, length=2, pad=1)
    ax.tick_params(axis="y", labelsize=8, length=2, pad=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_ylabel(r"$\Delta$ " + unit, fontsize=10, labelpad=-1.5)
    ax.spines[["top", "right"]].set_visible(False)
    return ax, label


def check_card_bounds(fig: plt.Figure, cards: list[tuple[plt.Axes, plt.Text]]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, title in cards:
        plot = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        heading = title.get_window_extent(renderer).transformed(
            fig.dpi_scale_trans.inverted())
        if (plot.width > PLOT_WIDTH + .001 or plot.height > PLOT_HEIGHT + .001 or
                plot.y1 >= heading.y0):
            raise ValueError(f"Plot region {plot.width:.3f} × {plot.height:.3f} in "
                             "exceeds standard or overlaps its title")


def render(data: pd.DataFrame) -> None:
    style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.hashsalt": "COSYNE-2027-high-credit-representation"})
    CARD_DIRECTORY.mkdir(exist_ok=True)
    for metric, title, unit, slug in METRICS:
        fig = plt.figure(figsize=(CARD_WIDTH, CARD_HEIGHT), facecolor="white")
        check_card_bounds(fig, [draw_card(fig, data, metric, title, unit, 0, 0)])
        save(fig, slug, CARD_DIRECTORY)
    fig = plt.figure(figsize=(7.5, 5.0), facecolor="white")
    fig.text(.03, .985, "High policy credit: temporal − constant worker bonus",
             va="top", fontsize=12, fontweight="bold", color="#20272C")
    fig.text(.03, .94, "DG representation in 100M-frame own-policy snapshots · five paired seeds per family",
             va="top", fontsize=10, color="#20272C")
    positions = ((.05, 2.55), (2.55, 2.55), (5.05, 2.55),
                 (1.30, .20), (3.80, .20))
    cards = [draw_card(fig, data, metric, title, unit, *position)
             for (metric, title, unit, _), position in zip(METRICS, positions)]
    check_card_bounds(fig, cards)
    save(fig, OUTPUT_STEM, HERE)


def main() -> None:
    data, provenance = paired_data()
    (HERE / "high_credit_peak_area_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    render(data)
    for family in ("C05", "C15"):
        values = data.loc[data.family == family,
                          "delta_effective_peak_area_fraction"].to_numpy(float)
        print(f"{family}: area Δ mean {values.mean():+.6f}; "
              f"positive {int((values > 0).sum())}/{len(values)}")
    print(f"Shared high-credit occupancy mask: {provenance['shared_mask_bins']} bins")


if __name__ == "__main__":
    main()
