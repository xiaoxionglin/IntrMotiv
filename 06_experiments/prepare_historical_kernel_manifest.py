"""Build a reviewed historical kernel manifest from the exact checkpoint audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PANEL = Path("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/"
             "poster_missing_analyses_20260926/panels/reduced_sat.npz")
SHARED_GROUPS = {"CPD_C15", "CCR_C04_C05_C15"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.audit.open(newline="") as stream:
        audit = list(csv.DictReader(stream, delimiter="\t"))
    selected = [row for row in audit if row["group"] in SHARED_GROUPS]
    if len(selected) != 21 or any(not row["latest_shared_checkpoint"] for row in selected):
        raise ValueError("Historical CPD/CCR audit lacks 21 exact shared checkpoints")
    rows = []
    for row in selected:
        frame = int(row["latest_shared_frames"])
        rows.append({"condition": row["condition"], "seed": row["seed"],
                     "checkpoint_frames": frame,
                     "checkpoint": row["latest_shared_checkpoint"],
                     "run_dir": row["run_dir"],
                     "label_suffix": f"{row['condition']}_S{row['seed']}_{frame}",
                     "panel": PANEL, "source_kind": "shared", "per_cue": "false",
                     "analysis_role": "endpoint", "max_offset_bins_per_axis": 18})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"schema": "poster_kernel_manifest_review_v1",
                "combined_manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "row_count": len(rows), "groups": sorted(SHARED_GROUPS),
                "panel": str(PANEL), "common_age_source": str(args.audit)}
    args.output.with_suffix(".review.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Reviewed {len(rows)} exact-age kernel rows")


if __name__ == "__main__":
    main()
