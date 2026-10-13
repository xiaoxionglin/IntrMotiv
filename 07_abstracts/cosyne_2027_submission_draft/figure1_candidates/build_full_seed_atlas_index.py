"""Copy and index all 40 existing COSYNE terminal DG atlas PNGs.

The input directory is a read-only rsync copy of the canonical NEMO2
checkpoint-analysis root. Run identity and factors come from the validated
all-checkpoint table, not from parsing filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "06_experiments/results/cosyne_2027_checkpoint_fields"
OUTPUT = HERE / "full_seed_atlases"
MINMAX_OUTPUT = HERE / "full_seed_atlases_minmax"
INDEX = HERE / "full_place_field_atlases.md"
REMOTE_ROOT = ("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/"
               "SF_hipposlam/train_dir/analysis/cosyne27_checkpoint_20261011")
SEEDS = (8, 17, 41, 99, 123)
CONDITIONS = (
    ("C01", "temporal", "inapplicable", "Temporal DG; direct exploration"),
    ("C01", "constant", "inapplicable", "Constant DG; direct exploration"),
    ("C05", "temporal", "temporal", "Temporal DG / temporal worker"),
    ("C05", "constant", "temporal", "Constant DG / temporal worker"),
    ("C05", "temporal", "constant", "Temporal DG / constant worker"),
    ("C05", "constant", "constant", "Constant DG / constant worker"),
    ("C15", "temporal", "temporal", "Temporal DG; frontier-selected targets"),
    ("C15", "constant", "temporal", "Constant DG; frontier-selected targets"),
)


def source_relative(row: pd.Series) -> Path:
    folder = Path(str(row.source_study))
    if row.source_study == "c15_recovery":
        folder /= "field"
    return folder / "summary" / f"place_fields_{row.label}.png"


def output_name(row: pd.Series) -> str:
    factors = [row.family.lower(), row.encoder_credit]
    if row.family == "C05":
        factors.append(row.worker_bonus)
    return "_".join(factors) + f"_s{int(row.seed)}.png"


def write_unit_table(terminal: pd.DataFrame, recovery_csv: Path) -> list[dict]:
    """Join all 640 terminal unit rows to the canonical factor table."""
    studies = sorted(set(terminal.source_study))
    parts = []
    sources = []
    for study in studies:
        source = (recovery_csv if study == "c15_recovery" else
                  RESULTS / study / "summary/per_unit_place_field_metrics.csv")
        rows = pd.read_csv(source)
        rows = rows[rows.target_frames == 100_000_000].copy()
        rows["source_study"] = study
        parts.append(rows)
        sources.append({"study": study,
                        "path": (f"{REMOTE_ROOT}/c15_recovery/field/analysis/"
                                 "per_unit_place_field_metrics.csv"
                                 if study == "c15_recovery" else
                                 str(source.relative_to(ROOT))),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    unit_rows = pd.concat(parts, ignore_index=True)
    factors = terminal[["condition", "seed", "family", "encoder_credit",
                        "worker_bonus", "source_study"]].drop_duplicates()
    unit_rows = unit_rows.drop(columns=["family"], errors="ignore").merge(
        factors, on=["condition", "seed", "source_study"], how="inner",
        validate="many_to_one")
    if (len(unit_rows) != 640 or
            unit_rows.duplicated(["condition", "seed", "unit"]).any() or
            not unit_rows.groupby(["condition", "seed"]).size().eq(16).all()):
        raise ValueError("Expected 16 unique DG units for each of 40 terminal runs")
    columns = ["family", "encoder_credit", "worker_bonus", "condition",
               "seed", "unit", "spatial_information_bits", "active_fraction",
               "field_eligible", "mono_field", "mono_score",
               "field_components_30pct", "field_components_50pct",
               "field_components_70pct", "field_spread", "source_study"]
    unit_rows[columns].sort_values(
        ["family", "encoder_credit", "worker_bonus", "seed", "unit"]
    ).to_csv(OUTPUT / "terminal_unit_metrics.csv", index=False)
    return sources


def write_index(records: pd.DataFrame) -> None:
    minmax_paths = [MINMAX_OUTPUT / file for file in records.file]
    minmax_count = sum(path.is_file() for path in minmax_paths)
    if minmax_count not in (0, len(records)):
        raise ValueError(f"Only {minmax_count}/{len(records)} min–max atlases exist")
    lines = [
        "# All COSYNE terminal DG place-field atlases",
        "",
        "These are the **unfiltered complete atlases for all 40 terminal networks** "
        "in the main COSYNE C01/C05/C15 field comparison: five seeds (8, 17, "
        "41, 99, 123) in each of eight factor cells. Every PNG displays all "
        "16 DG units, numbered 0–15 in row-major order. No seed or unit was "
        "removed because of its field statistics.",
        "",
        "## Architecture notes",
        "",
        "All three architectures share a fixed ImageNet-pretrained ResNet-18 "
        "trunk through layer 2, trainable DG projection with batch "
        "normalization, fixed CA3-like sequence memory, and learned readout. "
        "**C01** explores directly. **C05** navigates toward an internally "
        "detected DG target. **C15** adds frontier-based selection of DG "
        "targets. The [original diagram](../figures/architecture_v2.svg) "
        "shows the common modules.",
        "",
        "## Select from every seed",
        "",
        "Each cell links the complete 16-unit atlas in both scale conventions "
        "and states its visited spatial bins out of 361. Select by architecture, "
        "DG credit, worker "
        "credit, seed, and unit number—for example, `C05 temporal/temporal, "
        "seed 123, DG 7`. C01 has no target worker; both C15 rows have the "
        "same frontier-target architecture and temporal worker credit.",
        "",
        "| Architecture | Condition | " + " | ".join(f"Seed {seed}" for seed in SEEDS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in SEEDS) + " |",
    ]
    for family, encoder, worker, label in CONDITIONS:
        cells = []
        for seed in SEEDS:
            matches = records[(records.family == family) &
                              (records.encoder_credit == encoder) &
                              (records.worker_bonus == worker) &
                              (records.seed == seed)]
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate atlas: {family}/{encoder}/{worker}/{seed}")
            row = matches.iloc[0]
            links = f"[shared](full_seed_atlases/{row.file})"
            if minmax_count:
                links += f" · [min–max](full_seed_atlases_minmax/{row.file})"
            cells.append(f"{links} · {int(row.visited_cells)} bins")
        lines.append(f"| {family} | {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## What the maps can and cannot show",
        "",
        "Each atlas comes from one 10,001-decision policy-driven frozen-model "
        "probe near 100 million training frames. Gray bins were unvisited. "
        "The original **shared** renderer divides every rate by the 98th "
        "percentile of all finite values pooled across that run's 16 unit "
        "maps, then clips the display to 0–1. Thus color has one scale "
        "within a run, but not across runs. The separate **min–max** renderer "
        "sets each unit's visited-bin minimum to 0 and maximum to 1, "
        "without percentile clipping; all 640 minima happen to be zero. "
        "Its maps preserve the original occupancy mask and orientation and "
        "print each unit's raw range below the tile. Min–max color exposes "
        "shape but cannot compare response amplitude across units. The "
        "[per-unit scale table](full_seed_atlases_minmax/unit_scale_ranges.csv) "
        "and [min–max source hashes](full_seed_atlases_minmax/manifest.json) "
        "record the exact transformation; the "
        "[renderer](render_minmax_terminal_atlases.py) rebuilds these maps "
        "from the numerical NPZs. Probe occupancy "
        "and sampled trajectories differ across networks; these images are "
        "for selecting illustrative units, not for a controlled contrast. "
        "Use the [all-seed metric table](full_seed_atlases/terminal_seed_metrics.csv) "
        "for exact spatial-information, mono-field, activity, and coverage "
        "values. The [per-unit table](full_seed_atlases/terminal_unit_metrics.csv) "
        "lists all 640 unit scores, activity fractions, and mono-field "
        "measurements. The [source manifest](full_seed_atlases/manifest.json) "
        "records remote paths and PNG hashes. The original "
        "[checkpoint analysis](../../../06_experiments/cosyne_2027/checkpoint_fields_20261011.md) "
        "reports paired statistical results.",
        "",
        "The earlier six-image gallery used C01/C05 seed 99 and C15 seed "
        "123 because those example atlases had been copied locally. That "
        "gallery was **not an all-seed ranking**. In the original peak "
        "illustration, C15 seed 123 was used because constant-credit seed "
        "99 lacked its original terminal checkpoint; a separately indexed "
        "recovery later supplied it. Some omitted seeds have stronger "
        "statistics: C01 temporal seed 123 has mean DG spatial information "
        "0.217 versus 0.190 for seed 99; C05 temporal/temporal seed 123 "
        "visits 312 bins and has a 0.133 strict mono-field fraction versus "
        "190 bins and 0.000 for seed 99. These examples also differ on "
        "other measures, so neither seed is universally better.",
        "",
        "C15 constant-credit seeds 8 and 99 are recovery runs with a "
        "distinct source lineage, recorded in the manifest and the "
        "[run catalog](../../../06_experiments/cosyne_2027/training_runs.csv). "
        "The atlases were copied from the allocated NEMO2 workspace; no "
        "new evaluation or training was run.",
        "",
    ]
    INDEX.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True,
                        help="Local rsync copy of the canonical checkpoint-analysis root")
    parser.add_argument("--recovery-unit-csv", type=Path, required=True,
                        help="Rsync copy of the C15 recovery per-unit metric CSV")
    args = parser.parse_args()
    terminal = pd.read_csv(RESULTS / "all_checkpoint_fields.csv")
    terminal = terminal[terminal.target_frames == 100_000_000].copy()
    if len(terminal) != 40 or terminal.duplicated(["condition", "seed"]).any():
        raise ValueError("Expected 40 unique terminal condition/seed rows")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in terminal.iterrows():
        relative = source_relative(row)
        source = args.download_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as atlas:
            if atlas.size != (1200, 1206):
                raise ValueError(f"Unexpected source atlas dimensions: {source}")
            atlas.verify()
        destination = OUTPUT / output_name(row)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Existing destination differs: {destination}")
        shutil.copyfile(source, destination)
        records.append({
            "family": row.family,
            "encoder_credit": row.encoder_credit,
            "worker_bonus": row.worker_bonus,
            "condition": row.condition,
            "seed": int(row.seed),
            "file": destination.name,
            "visited_cells": int(row.visited_cells),
            "mean_spatial_information_bits": float(row.mean_spatial_information_bits),
            "max_spatial_information_bits": float(row.max_spatial_information_bits),
            "mono_field_fraction": float(row.mono_field_fraction),
            "mean_active_fraction": float(row.mean_active_fraction),
            "field_eligible_units": int(row.field_eligible_units),
            "source_study": row.source_study,
            "source_relative_path": str(relative),
            "source_remote_path": f"{REMOTE_ROOT}/{relative}",
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT / "terminal_seed_metrics.csv", index=False)
    unit_sources = write_unit_table(terminal, args.recovery_unit_csv)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "schema": "intrmotiv/cosyne-terminal-atlas-index/v1",
        "source_metrics": str((RESULTS / "all_checkpoint_fields.csv").relative_to(ROOT)),
        "source_metrics_sha256": hashlib.sha256(
            (RESULTS / "all_checkpoint_fields.csv").read_bytes()).hexdigest(),
        "remote_root": REMOTE_ROOT,
        "unit_metric_sources": unit_sources,
        "atlases": records,
    }, indent=2) + "\n")
    write_index(result)


if __name__ == "__main__":
    main()
