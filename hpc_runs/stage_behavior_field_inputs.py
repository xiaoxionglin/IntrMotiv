"""Copy only poster terminal configs/checkpoints into the active NEMO2 workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from hpc_runs.behavior_field_study import CONDITIONS, SEEDS, CHECKPOINT_FRAMES


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage(source: Path, destination: Path, workspace_root: Path) -> list[dict]:
    destination = destination.resolve()
    if not destination.is_relative_to(workspace_root.resolve(strict=True)):
        raise ValueError("Staging destination must be inside the active workspace")
    with source.open(newline="") as stream:
        selected = [row for row in csv.DictReader(stream)
                    if row["condition"] in CONDITIONS and int(row["seed"]) in SEEDS]
    if len(selected) != 9:
        raise ValueError(f"Expected nine unique poster rows, found {len(selected)}")
    records = []
    for row in selected:
        condition, seed = row["condition"], int(row["seed"])
        if int(row["checkpoint_frames"]) != CHECKPOINT_FRAMES:
            raise ValueError("Checkpoint age mismatch")
        original = Path(row["checkpoint"])
        config = original.parent.parent / "config.json"
        if not original.is_file() or not config.is_file():
            raise FileNotFoundError(original if not original.is_file() else config)
        run = destination / f"{condition}_S{seed}_" / f"00_{CONDITIONS[condition]}_S{seed}"
        for file, target in ((config, run / "config.json"),
                             (original, run / "checkpoint_p0" / original.name)):
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = digest(file)
            if target.exists():
                if digest(target) != expected:
                    raise FileExistsError(f"Existing staged file differs: {target}")
            else:
                temporary = target.with_suffix(target.suffix + ".partial")
                if temporary.exists():
                    temporary.unlink()
                shutil.copyfile(file, temporary)
                if digest(temporary) != expected:
                    raise IOError(f"Copy hash mismatch: {temporary}")
                temporary.replace(target)
            records.append({"condition": condition, "seed": seed,
                            "source": str(file), "staged": str(target), "sha256": expected})
    (destination.parent / "staged_inputs.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"files": len(stage(args.source, args.destination, args.workspace_root))}))


if __name__ == "__main__":
    main()
