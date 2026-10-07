"""Manifest and ordinary-job launcher for frozen behavior-field probes.

Use ``prepare`` on the poster's pinned nine-run terminal CSV, then ``submit``
on NEMO2. Submission is print-only unless ``--submit`` is given.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


CONDITIONS = {
    "C01": "CCR_C01_FLAT_CTRL",
    "C05": "CCR_C05_DIRECT_IMMEDIATE_G001_R100",
    "C15": "CCR_C15_TOPOLOGY_UCB_DIRECT_O1",
}
SEEDS = (8, 99, 123)
EVAL_SEEDS = (51_000, 52_000)
POLICIES = ("own", "uniform", "persistent8")
CHECKPOINT_FRAMES = 100_040_704
DECISIONS = 50_000
FIELDS = ("condition", "seed", "eval_seed", "policy", "decisions", "checkpoint_frames",
          "run_dir", "checkpoint", "label")


def make_rows(source: Path, workspace: Path) -> list[dict[str, str]]:
    with source.open(newline="") as stream:
        selected = {(row["condition"], int(row["seed"])): row
                    for row in csv.DictReader(stream)
                    if row["condition"] in CONDITIONS and int(row["seed"]) in SEEDS}
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if selected.keys() != expected:
        raise ValueError(f"Missing poster models: {expected - selected.keys()}")
    rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            source_row = selected[(condition, seed)]
            if int(source_row["checkpoint_frames"]) != CHECKPOINT_FRAMES:
                raise ValueError("Poster model checkpoint age mismatch")
            checkpoint = Path(source_row["checkpoint"])
            if CONDITIONS[condition] + f"_S{seed}" not in str(checkpoint):
                raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
            staged_run = workspace / "inputs" / f"{condition}_S{seed}_" / f"00_{CONDITIONS[condition]}_S{seed}"
            staged_checkpoint = staged_run / "checkpoint_p0" / checkpoint.name
            for eval_seed in EVAL_SEEDS:
                for policy in POLICIES:
                    rows.append({
                        "condition": condition, "seed": str(seed),
                        "eval_seed": str(eval_seed), "policy": policy,
                        "decisions": str(DECISIONS), "checkpoint_frames": str(CHECKPOINT_FRAMES),
                        "run_dir": str(staged_run), "checkpoint": str(staged_checkpoint),
                        "label": f"{condition}_S{seed}_E{eval_seed}_{policy}",
                    })
    return rows


def read_manifest(path: Path, workspace_root: Path, require_inputs: bool) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected manifest columns")
        rows = list(reader)
    if len(rows) != 54 or len({row["label"] for row in rows}) != 54:
        raise ValueError("Expected 54 distinct rows: nine models x two evaluation seeds x three policies")
    actual = {(row["condition"], int(row["seed"]), int(row["eval_seed"]), row["policy"])
              for row in rows}
    expected = {(condition, seed, eval_seed, policy)
                for condition in CONDITIONS for seed in SEEDS
                for eval_seed in EVAL_SEEDS for policy in POLICIES}
    if actual != expected:
        raise ValueError(f"Incomplete or repeated probe matrix: missing={expected - actual}, extra={actual - expected}")
    for row in rows:
        if int(row["decisions"]) != DECISIONS or int(row["checkpoint_frames"]) != CHECKPOINT_FRAMES:
            raise ValueError(f"Protocol mismatch in {row['label']}")
        if row["condition"] not in CONDITIONS or int(row["seed"]) not in SEEDS or int(row["eval_seed"]) not in EVAL_SEEDS:
            raise ValueError(f"Identity mismatch in {row['label']}")
        for key in ("run_dir", "checkpoint"):
            value = Path(row[key]).resolve()
            if not value.is_relative_to(workspace_root.resolve()):
                raise ValueError(f"{key} escapes active workspace: {value}")
            if require_inputs and not (value.is_dir() if key == "run_dir" else value.is_file()):
                raise FileNotFoundError(value)
    return rows


def submit(args: argparse.Namespace) -> None:
    root = args.workspace_root.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    runner = args.runner.resolve(strict=True)
    source = args.runtime_source.resolve(strict=True)
    overlay = args.overlay.resolve(strict=True)
    for path in (manifest, output, runner, overlay):
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact escapes active workspace: {path}")
    rows = read_manifest(manifest, root, require_inputs=True)
    indices = list(range(len(rows))) if args.rows == "all" else [int(s) for s in args.rows.split(",")]
    if any(i < 0 or i >= len(rows) for i in indices) or len(indices) != len(set(indices)):
        raise ValueError("Invalid or repeated row index")
    if args.arm != "all":
        indices = [i for i in indices if rows[i]["policy"] == args.arm]
    commands = []
    for i in indices:
        row = rows[i]
        destination = output / "raw" / row["label"]
        if destination.exists():
            raise FileExistsError(destination)
        command = ["sbatch", f"--job-name=bf-{row['label']}", "--partition=cpu",
                   "--cpus-per-task=4", "--mem=16G", f"--time={args.time_limit}",
                   f"--output={output / 'slurm' / (row['label'] + '-%j.out')}",
                   "--export=ALL," + ",".join((
                       f"BF_WORKSPACE={root}", f"BF_OUTPUT={output}",
                       f"BF_RUNTIME={source}", f"BF_OVERLAY={overlay}")),
                   str(runner), str(manifest), str(i)]
        commands.append((i, command))
    print(f"Validated {len(commands)} ordinary jobs; manifest SHA-256 "
          f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}")
    for _, command in commands:
        print(shlex.join(command))
    if not args.submit:
        return
    for folder in (output / "raw", output / "slurm", output / "tmp", output / "cache",
                   output / "dmlab_cache", output / "wandb"):
        folder.mkdir(parents=True, exist_ok=True)
    record = output / f"submission_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.tsv"
    with record.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("row", "label", "job_id", "command"))
        for i, command in commands:
            response = subprocess.check_output(command, text=True).strip()
            match = re.fullmatch(r"Submitted batch job (\d+)", response)
            if not match:
                raise RuntimeError(f"Unexpected sbatch response: {response}")
            writer.writerow((i, rows[i]["label"], match.group(1), shlex.join(command)))
            stream.flush()
            time.sleep(1)
    print(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--workspace-root", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--runner", type=Path, required=True)
    launch.add_argument("--runtime-source", type=Path, required=True)
    launch.add_argument("--overlay", type=Path, required=True)
    launch.add_argument("--rows", default="all")
    launch.add_argument("--arm", choices=("all",) + POLICIES, default="all")
    launch.add_argument("--time-limit", default="08:00:00")
    launch.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        rows = make_rows(args.source, args.workspace)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, delimiter="\t", fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
    else:
        submit(args)


if __name__ == "__main__":
    main()
