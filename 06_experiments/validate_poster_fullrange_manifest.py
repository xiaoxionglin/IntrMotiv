"""Print-only review of staged poster kernel rows before Slurm submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if digest != metadata["combined_manifest_sha256"]:
        raise ValueError("Combined manifest SHA-256 mismatch")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != metadata["row_count"]:
        raise ValueError("Manifest row count mismatch")
    labels = {row["label_suffix"] for row in rows}
    if len(labels) != len(rows):
        raise ValueError("Duplicate output labels")
    for index, row in enumerate(rows):
        for key in ("run_dir", "checkpoint", "panel"):
            path = Path(row[key])
            if not path.is_relative_to(args.workspace_root) or not path.exists():
                raise ValueError(f"Row {index} {key} missing or outside workspace: {path}")
        if row["checkpoint_frames"] not in Path(row["checkpoint"]).name:
            raise ValueError(f"Row {index} checkpoint frame mismatch")
        if int(row["max_offset_bins_per_axis"]) != 18:
            raise ValueError(f"Row {index} does not request full range")
        if row["source_kind"] not in {"shared", "cue"} or row["per_cue"] not in {"true", "false"}:
            raise ValueError(f"Row {index} source or cue setting invalid")
    print(f"PRINT-ONLY REVIEW PASS: {len(rows)} rows, {len(labels)} unique labels")
    print("Panels:", ", ".join(sorted({Path(row["panel"]).name for row in rows})))
    for role in sorted({row["analysis_role"] for row in rows}):
        print(f"{role}: {sum(row['analysis_role'] == role for row in rows)} rows")
    print("Manifest SHA-256:", digest)


if __name__ == "__main__":
    main()
