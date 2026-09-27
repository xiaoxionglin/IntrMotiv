"""Submit reviewed poster-kernel manifest rows as independent Slurm jobs.

Defaults to print-only. Recorded job IDs make interrupted submissions resumable
without launching a duplicate row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--sbatch-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--skip-indices", default="", help="Comma-separated already-submitted row indices")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != metadata["combined_manifest_sha256"]:
        raise ValueError("Manifest SHA-256 differs from the print-only review")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != metadata["row_count"]:
        raise ValueError("Manifest row count changed")
    skipped = {int(value) for value in args.skip_indices.split(",") if value}
    job_map = args.output_root / "submitted_jobs.tsv"
    if job_map.exists():
        with job_map.open(newline="") as stream:
            skipped.update(int(row["row_index"]) for row in csv.DictReader(stream, delimiter="\t"))
    priority = {"endpoint": 0, "endpoint_unmatched_age": 1,
                "endpoint_exemplar": 2, "developmental": 3}
    planned = sorted(((index, row) for index, row in enumerate(rows) if index not in skipped),
                     key=lambda item: (priority[item[1]["analysis_role"]], item[0]))
    for index, row in planned:
        print(f"{index}\t{row['analysis_role']}\t{row['label_suffix']}\t{row['panel']}", flush=True)
    print(f"Reviewed {len(planned)} unsubmitted independent rows", flush=True)
    if not args.submit:
        return
    (args.output_root / "slurm").mkdir(parents=True, exist_ok=True)
    job_exists = job_map.exists()
    with job_map.open("a", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        if not job_exists:
            writer.writerow(("row_index", "job_id", "label_suffix"))
        for index, row in planned:
            command = ["sbatch", "--parsable", f"--job-name=poster-kfull-{index}",
                       f"--output={args.output_root}/slurm/%j.out",
                       f"--error={args.output_root}/slurm/%j.err",
                       str(args.sbatch_script), str(args.manifest), str(index),
                       row["panel"], str(args.output_root), row["source_kind"], row["per_cue"]]
            job_id = subprocess.check_output(command, text=True).strip()
            writer.writerow((index, job_id, row["label_suffix"]))
            stream.flush()
            print(f"Submitted row {index}: {job_id}", flush=True)


if __name__ == "__main__":
    main()
