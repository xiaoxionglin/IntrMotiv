"""Inventory every saved checkpoint for the explicit poster manifest runs.

Direct checkpoint files and retained milestone files both count. The maximum
intersection of saved frame counts defines each comparison's latest shared age.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FRAME = re.compile(r"checkpoint_\d+_(\d+)\.pth$")


def group(condition: str) -> str:
    if condition.startswith("SAT_"):
        return "SAT_ARR_SRC"
    if condition.startswith("CPU2048_") and "DDQN_HER" in condition:
        return "CPU2048_HER"
    if condition.startswith("DGP_"):
        return "DGP_HIT_FIRST"
    if condition.startswith("CR5C_D50_"):
        return "D50_SOURCE_RANDOM"
    if condition.startswith("CR5C_D51_"):
        return "D51_SOURCE_RANDOM"
    if condition == "DGC_DIRECT_WORKER_F16":
        return "DGC_DIRECT"
    if condition == "DGC_WAYPOINT_DG_F64":
        return "DGC_WAYPOINT"
    return "MATURE_EXEMPLAR"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = {}
    for manifest in args.manifest:
        with manifest.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                key = (row["condition"], int(row["seed"]))
                if key not in selected or int(row["checkpoint_frames"]) > int(selected[key]["checkpoint_frames"]):
                    selected[key] = row
    inventories = []
    for (condition, seed), row in sorted(selected.items()):
        checkpoint_dir = Path(row["run_dir"]) / "checkpoint_p0"
        saved = {}
        for path in checkpoint_dir.glob("checkpoint_*.pth"):
            match = FRAME.fullmatch(path.name)
            if match:
                saved[int(match.group(1))] = str(path)
        for path in (checkpoint_dir / "milestones").glob("checkpoint_*.pth"):
            match = FRAME.fullmatch(path.name)
            if match:
                saved.setdefault(int(match.group(1)), str(path))
        if not saved:
            raise ValueError(f"No checkpoints: {checkpoint_dir}")
        selected_frame = int(row["checkpoint_frames"])
        if selected_frame not in saved or not Path(row["checkpoint"]).exists():
            raise ValueError(f"Selected checkpoint missing: {condition} S{seed} {selected_frame}")
        inventories.append({"comparison_group": group(condition), "condition": condition,
                            "seed": seed, "run_dir": row["run_dir"],
                            "saved_checkpoint_count": len(saved),
                            "latest_saved_checkpoint_frames": max(saved),
                            "selected_analysis_checkpoint_frames": selected_frame,
                            "selected_analysis_checkpoint": row["checkpoint"],
                            "saved_checkpoint_frames": ",".join(map(str, sorted(saved))),
                            "saved_frame_set": set(saved)})
    groups = sorted(set(row["comparison_group"] for row in inventories))
    for name in groups:
        members = [row for row in inventories if row["comparison_group"] == name]
        common = set.intersection(*(row["saved_frame_set"] for row in members))
        latest = max(common) if common else None
        for row in members:
            row["latest_exact_shared_checkpoint_frames"] = latest if latest is not None else ""
            if latest is not None and row["selected_analysis_checkpoint_frames"] != latest:
                raise ValueError(f"Selected {row['condition']} S{row['seed']} at "
                                 f"{row['selected_analysis_checkpoint_frames']} instead of shared {latest}")
    for row in inventories:
        row.pop("saved_frame_set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventories[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(inventories)
    print(f"Inventoried {len(inventories)} selected runs in {len(groups)} groups")


if __name__ == "__main__":
    main()
