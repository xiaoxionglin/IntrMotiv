"""Print-only by default: submit missing C05 common-panel layer replay jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import shlex
import subprocess


WORKSPACE = Path("/work/classic/fr_xl1014-corridor-geometry")
RUNTIME = Path("/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--dg-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    for path in (args.manifest, args.panel, args.dg_root, args.output_root, args.source):
        if not path.resolve().is_relative_to(WORKSPACE):
            raise ValueError(f"Path outside active workspace: {path}")
    if not args.manifest.is_file() or not args.panel.is_file() or not args.dg_root.is_dir():
        raise FileNotFoundError("Required manifest, panel, or DG reference is missing")
    runner = args.source / "hpc_runs/run_common_panel_layer_single.sh"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    with args.manifest.open(newline="") as stream:
        rows = list(enumerate(csv.DictReader(stream, delimiter="\t")))
    selected = [(index, row) for index, row in rows
                if row["condition"].startswith("COS27_C05_") and
                int(row["target_frames"]) == 100_000_000]
    if len(selected) != 20 or len({(row["condition"], row["seed"]) for _, row in selected}) != 20:
        raise ValueError("Expected exactly 20 original C05 terminal manifest rows")
    active = subprocess.check_output(["squeue", "-u", "fr_xl1014", "-h", "-o", "%j"], text=True).splitlines()
    commands = []
    for index, row in selected:
        label = f'{row["condition"]}__s{row["seed"]}__f{row["checkpoint_frames"]}'
        job_name = f"cos27-layer-{index}"
        destination = args.output_root / "raw" / f"{label}.json"
        if destination.exists():
            continue
        if job_name in active:
            raise RuntimeError(f"Replay already queued or running: {job_name}")
        cmd = ["sbatch", f"--job-name={job_name}", "--partition=cpu",
               "--cpus-per-task=4", "--mem=16G", "--time=01:00:00",
               f"--output={args.output_root / 'slurm' / (job_name + '-%j.out')}",
               str(runner), str(args.manifest), str(index), str(args.panel),
               str(args.dg_root), str(args.output_root), str(args.source), str(RUNTIME)]
        commands.append((index, label, cmd))
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(f"Manifest SHA-256 {digest}; 20 C05 rows; {len(commands)} missing jobs", flush=True)
    for _, _, cmd in commands:
        print(shlex.join(cmd), flush=True)
    if not args.submit:
        return
    if not commands:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "slurm").mkdir(exist_ok=True)
    submission = args.output_root / "submission_manifest.tsv"
    if submission.exists():
        raise FileExistsError(submission)
    with submission.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("row", "label", "job_id", "command"))
        for index, label, cmd in commands:
            answer = subprocess.check_output(cmd, text=True).strip()
            match = re.fullmatch(r"Submitted batch job (\d+)", answer)
            if match is None:
                raise RuntimeError(f"Unexpected sbatch response: {answer}")
            writer.writerow((index, label, match.group(1), shlex.join(cmd)))
            stream.flush()
            print(answer, label, flush=True)


if __name__ == "__main__":
    main()
