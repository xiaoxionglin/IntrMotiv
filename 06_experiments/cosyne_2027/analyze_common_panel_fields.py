"""Pair COSYNE field metrics on one saved, identical observation sequence.

The canonical manifest-driven evaluator has already computed each checkpoint's
field metrics. This adapter joins StudySpec factors and renders the same
contrasts as the policy-specific field analysis for a direct source comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .analyze_checkpoint_fields import (
    OUT,
    ROOT,
    attach_catalog_factors,
    paired_terminal,
    paired_worker,
    plot_factorized,
    plot_pairs,
    plot_worker_pairs,
)
from .plot_wandb_online import write_csv


SOURCE = OUT / "common_panel/summary/derived_place_field_metrics.csv"
MANIFEST = OUT / "available_terminal_manifest.tsv"
ANALYSIS = OUT / "common_panel_analysis"
CERTIFICATE = ANALYSIS / "occupancy_certificate.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    certificate = json.loads(CERTIFICATE.read_text())
    if (certificate["replay_files"] != 38
            or len(certificate["panel_paths"]) != 1
            or len(certificate["occupancy_sha256"]) != 1
            or certificate["evaluated_observations"] != [10001]
            or certificate["visited_bins"] != [242]):
        raise ValueError("shared-observation certificate is incomplete")

    source = pd.read_csv(SOURCE)
    manifest = pd.read_csv(MANIFEST, sep="\t")
    if len(source) != 38 or len(manifest) != 38:
        raise ValueError("expected the 38 available terminal checkpoints")
    if set(source.label_suffix) != set(manifest.label_suffix):
        raise ValueError("source and validated terminal manifest disagree")
    if source.visited_cells.nunique() != 1 or source.frames.nunique() != 1:
        raise ValueError("common-panel support is not identical")
    rows = attach_catalog_factors(source)
    pairs, summary = paired_terminal(rows)
    worker_pairs, worker_summary = paired_worker(rows)

    write_csv(rows, ANALYSIS / "shared_observation_fields.csv")
    write_csv(pairs, ANALYSIS / "paired_encoder_fields.csv")
    write_csv(summary, ANALYSIS / "paired_encoder_summary.csv")
    write_csv(worker_pairs, ANALYSIS / "paired_worker_fields.csv")
    write_csv(worker_summary, ANALYSIS / "paired_worker_summary.csv")
    plot_pairs(pairs, out_dir=ANALYSIS,
               title_prefix="Paired 100M fields on identical observations")
    plot_worker_pairs(worker_pairs, out_dir=ANALYSIS)
    plot_factorized(rows, out_dir=ANALYSIS,
                    title_prefix="All available 100M arms · same observations")

    metadata = {
        "schema": "cosyne27_common_panel_fields_v1",
        "source_protocol": "one saved 10001-observation panel replayed on frozen checkpoints",
        "panel_path": certificate["panel_paths"][0],
        "occupancy_sha256": certificate["occupancy_sha256"][0],
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                    for p in (SOURCE, MANIFEST, CERTIFICATE)],
        "terminal_rows": len(rows),
        "paired_encoder_seed_metric_rows": len(pairs),
        "paired_worker_seed_metric_rows": len(worker_pairs),
    }
    (ANALYSIS / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
