"""Print exact shared checkpoint ages for historical poster candidate groups.

Uses only checkpoint filenames and Python's standard library so it can run on
the NEMO2 login node without loading a model or entering DMLab.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path("/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir")
GROUPS = {
    "N8_REFERENCE": ("intrmotiv_navigation8_algorithm_screen_20260909",
                     ("N8_W_REF_STOP", "N8_W_REF_JOINT")),
    "N8_SCR": ("intrmotiv_navigation8_algorithm_screen_20260909", ("N8_SCR_ARR_DIRS",)),
    "CPD_C15": ("intrmotiv_ca3_feedback_predictive_dg_20260907",
                ("CPD_C15_BASE", "CPD_C15_GATE_ACT_DIR_GOAL",
                 "CPD_C15_GATE_ACT_DIR", "CPD_C15_ADD_CA3_DIR")),
    "CCR_C04_C05_C15": ("intrmotiv_corrected_core_reevaluation_20260901",
                        ("CCR_C04_DIRECT_IMMEDIATE_ITER", "CCR_C05_DIRECT_IMMEDIATE_G001_R100",
                         "CCR_C15_TOPOLOGY_UCB_DIRECT_O1")),
}
FRAME = re.compile(r"checkpoint_\d+_(\d+)\.pth")


def main() -> None:
    rows = []
    for group, (batch, conditions) in GROUPS.items():
        inventory = []
        for condition in conditions:
            for seed in (8, 99, 123):
                run = f"{condition}_S{seed}"
                run_dir = ROOT / batch / f"{run}_" / f"00_{run}"
                checkpoint_dir = run_dir / "checkpoint_p0"
                files = list(checkpoint_dir.glob("checkpoint_*.pth"))
                files += list((checkpoint_dir / "milestones").glob("checkpoint_*.pth"))
                saved = {int(match.group(1)): path for path in files
                         if (match := FRAME.fullmatch(path.name))}
                if not saved:
                    raise FileNotFoundError(checkpoint_dir)
                inventory.append((condition, seed, run_dir, saved))
        shared = set.intersection(*(set(item[3]) for item in inventory))
        latest_shared = max(shared) if shared else None
        for condition, seed, run_dir, saved in inventory:
            rows.append({"group": group, "condition": condition, "seed": seed,
                         "run_dir": run_dir, "latest_saved_frames": max(saved),
                         "latest_shared_frames": latest_shared or "",
                         "latest_shared_checkpoint": saved[latest_shared] if latest_shared else "",
                         "saved_checkpoint_count": len(saved)})
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
