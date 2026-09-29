"""Export compact stored graph arrays from corrected-core checkpoints.

Run with the NEMO2 project Python environment and redirect stdout to JSON.
No checkpoint or training output is modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path("/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/"
            "intrmotiv_corrected_core_reevaluation_20260901")
PREFIX = "core.policy_graph."


def main() -> None:
    rows = []
    for condition in ("C05", "C15"):
        matches = list(ROOT.glob(f"CCR_{condition}_*_S99_/00_CCR_{condition}_*_S99/"
                                 "checkpoint_p0/checkpoint_*_100040704.pth"))
        if len(matches) != 1:
            raise ValueError(f"Expected one terminal checkpoint for {condition}: {matches}")
        path = matches[0]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = checkpoint["model"]
        row = {"condition": condition, "seed": 99, "checkpoint": str(path),
               "frames": int(checkpoint["env_steps"])}
        for name in ("tctrl", "edge_confidence", "control_attempts"):
            row[name] = model[PREFIX + name].detach().cpu().numpy().tolist()
        rows.append(row)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
