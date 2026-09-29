"""Stage small historical checkpoints for canonical frozen policy evaluation.

The legacy allocation is read-only. The evaluator receives a compact config
and exact checkpoint copy inside the active workspace; all output stays there.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    active = Path("/work/classic/fr_xl1014-corridor-geometry")
    old = Path("/work/classic/fr_xl1014-train")
    if not inside(args.output_root, active):
        raise ValueError("Historical frozen output must be in the active workspace")
    with args.audit.open(newline="") as stream:
        audit = list(csv.DictReader(stream, delimiter="\t"))
    selected = [item for item in audit if item["group"] == "CPD_C15"]
    if len(selected) != 12:
        raise ValueError("Expected twelve CPD C15 terminal rows")
    rows = []
    mapping = []
    for item in selected:
        source_run = Path(item["run_dir"])
        source_checkpoint = Path(item["latest_shared_checkpoint"])
        if not inside(source_run, old) or not inside(source_checkpoint, old):
            raise ValueError(f"Historical input outside legacy workspace: {source_run}")
        if not source_checkpoint.is_file():
            raise FileNotFoundError(source_checkpoint)
        frame = int(item["latest_shared_frames"])
        run_name = f"{item['condition']}_S{item['seed']}"
        label = f"{run_name}_{frame}"
        staged_run = args.output_root / "inputs" / label / f"00_{run_name}"
        staged_checkpoint = staged_run / "checkpoint_p0" / source_checkpoint.name
        staged_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for name in ("config.json",):
            shutil.copy2(source_run / name, staged_run / name)
        shutil.copy2(source_checkpoint, staged_checkpoint)
        rows.append({"condition": item["condition"], "family": "CPD_C15",
                     "schedule": "poster", "feedback": "na", "half_life": "na",
                     "seed": item["seed"], "target_frames": 75_000_000,
                     "checkpoint_frames": frame, "checkpoint": staged_checkpoint,
                     "run_dir": staged_run, "label_suffix": label})
        mapping.append({"label_suffix": label, "source_checkpoint": str(source_checkpoint),
                        "staged_checkpoint": str(staged_checkpoint)})
    manifest = args.output_root / "frozen_cpd_manifest.tsv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "frozen_cpd_staging.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Staged {len(rows)} CPD C15 checkpoints at {rows[0]['checkpoint_frames']} frames")


if __name__ == "__main__":
    main()
