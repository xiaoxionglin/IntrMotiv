"""Review and submit ordinary jobs for frozen common-panel layer replays.

The checkpoint manifest, rather than run-name parsing, selects the evaluation
matrix. Output JSON is written only by a successful compute-node replay.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import getpass
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess


WORKSPACE = Path("/work/classic/fr_xl1014-corridor-geometry")


def reviewed_commands(args: argparse.Namespace) -> list[tuple[int, str, list[str]]]:
    root = args.workspace.resolve(strict=True)
    if root != WORKSPACE:
        raise ValueError(f"Unexpected workspace: {root}")
    paths = (args.manifest, args.panel, args.dg_root, args.output,
             args.source, args.runtime)
    for path in paths[:-1]:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Path outside workspace: {path}")
    for path in (args.manifest, args.panel, args.source / "hpc_runs/common_panel_layer_probe.py",
                 args.source / "hpc_runs/run_common_panel_layer_single.sh"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.dg_root.is_dir() or not args.runtime.is_dir():
        raise FileNotFoundError("Missing DG reference or runtime source")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or len({(r["condition"], r["seed"]) for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate checkpoint manifest")
    selected = {int(value) for value in args.rows.split(",")} if args.rows else set(range(len(rows)))
    if any(index < 0 or index >= len(rows) for index in selected):
        raise ValueError("Row index outside manifest")
    families = set(args.families.split(","))
    if not families <= {"C01", "C05", "C15"}:
        raise ValueError("Unsupported family selection")
    commands = []
    for index in sorted(selected):
        row = rows[index]
        family = row["family"]
        if family not in families:
            continue
        if int(row["target_frames"]) != 100_000_000:
            raise ValueError(f"Nonterminal row {index}")
        if not Path(row["checkpoint"]).is_file() or not Path(row["run_dir"]).is_dir():
            raise FileNotFoundError(f"Missing checkpoint or run directory for row {index}")
        label = f'{row["condition"]}__s{row["seed"]}__f{row["checkpoint_frames"]}'
        destination = args.output / "raw" / f"{label}.json"
        if destination.exists():
            if args.skip_complete:
                item = json.loads(destination.read_text())
                if item.get("checkpoint") != row["checkpoint"] or item.get("condition") != row["condition"]:
                    raise ValueError(f"Existing output has wrong identity: {destination}")
                continue
            raise FileExistsError(destination)
        job_name = f"cos27-layer-{family.lower()}-{row['seed']}-{index}"
        command = [
            "sbatch", "--parsable", f"--job-name={job_name}", "--partition=cpu",
            "--cpus-per-task=4", "--mem=16G", "--time=01:00:00",
            f"--output={args.output / 'slurm' / (label + '-%j.out')}",
            str(args.source / "hpc_runs/run_common_panel_layer_single.sh"),
            str(args.manifest), str(index), str(args.panel), str(args.dg_root),
            str(args.output), str(args.source), str(args.runtime),
        ]
        commands.append((index, job_name, command))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dg-root", type=Path, required=True)
    parser.add_argument("--output", "--output-root", dest="output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--families", default="C01,C05,C15")
    parser.add_argument("--rows", help="Comma-separated zero-based manifest rows")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    commands = reviewed_commands(args)
    print(f"Reviewed {len(commands)} jobs; manifest SHA-256 {sha256(args.manifest.read_bytes()).hexdigest()}")
    for index, _name, command in commands:
        print(f"row {index}: {shlex.join(command)}")
    if not args.submit:
        return
    for folder in ("raw", "slurm", "tmp", "dmlab_cache", "cache", "wandb"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    queued = set(subprocess.check_output(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%j"], text=True
    ).splitlines())
    if any(name in queued for _, name, _ in commands):
        raise RuntimeError("A selected layer replay is already queued")
    ledger = args.output / f"submission_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.tsv"
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("row", "job_name", "job_id", "command"))
        for index, name, command in commands:
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {job_id}")
            writer.writerow((index, name, job_id, shlex.join(command)))
            stream.flush()
    print(ledger)


if __name__ == "__main__":
    main()
