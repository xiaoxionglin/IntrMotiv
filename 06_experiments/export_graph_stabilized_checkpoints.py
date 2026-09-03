#!/usr/bin/env python3
"""Export compact graph tensors from the latest GSR checkpoint in each run."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


RUN_RE = re.compile(r"GSR_(C05|C13|C15)_D(4|8)_H(5|10)K_S(8|99|123)$")
POLICY_PREFIX = "core.policy_graph."
RECRUITMENT_PREFIX = "core.passive_recruitment_graph."
DG_PREFIX = "encoder.DG_projection."


def array(model: dict[str, torch.Tensor], key: str, digits: int = 6) -> list:
    value = model[key].detach().cpu().numpy()
    return value.round(digits).tolist()


def scalar(model: dict[str, torch.Tensor], key: str) -> int | float:
    value = model[key].detach().cpu().item()
    return int(value) if isinstance(value, int) else float(value)


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list(run_dir.glob("checkpoint_p0/checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint below {run_dir}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def export_run(run_dir: Path) -> dict[str, object]:
    run_name = run_dir.name.removeprefix("00_")
    match = RUN_RE.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Unexpected run name: {run_name}")
    backbone, d_value, half_life, seed = match.groups()
    checkpoint_path = latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    result: dict[str, object] = {
        "run": run_name,
        "backbone": backbone,
        "d": int(d_value),
        "half_life_k": int(half_life),
        "seed": int(seed),
        "env_steps": int(checkpoint["env_steps"]),
        "checkpoint": str(checkpoint_path),
        "complete": int(checkpoint["env_steps"]) >= 100_000_000,
        "policy_confidence": array(model, POLICY_PREFIX + "edge_confidence"),
        "policy_time": array(model, POLICY_PREFIX + "tctrl"),
        "policy_attempts": array(model, POLICY_PREFIX + "control_attempts"),
        "node_visits": array(model, POLICY_PREFIX + "node_visits"),
        "passive_confidence": array(model, POLICY_PREFIX + "passive_confidence"),
        "passive_time": array(model, POLICY_PREFIX + "passive_time"),
        "passive_path_length": array(model, POLICY_PREFIX + "passive_path_length"),
        "recruitment_confidence": array(model, RECRUITMENT_PREFIX + "confidence"),
        "recruitment_time": array(model, RECRUITMENT_PREFIX + "elapsed"),
        "birth_support": array(model, RECRUITMENT_PREFIX + "birth_support"),
        "row_assignments": array(model, DG_PREFIX + "recruitment_row_counts", digits=0),
        "recruitment_total": scalar(model, DG_PREFIX + "recruitment_count"),
        "repeat_total": scalar(model, DG_PREFIX + "recruitment_repeat_count"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    run_dirs = sorted(args.batch_root.glob("GSR_*_/*"))
    rows = [export_run(run_dir) for run_dir in run_dirs if RUN_RE.fullmatch(run_dir.name.removeprefix("00_"))]
    if len(rows) != 36:
        raise SystemExit(f"Expected 36 GSR runs, found {len(rows)}")
    payload = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows),
        "policy_known_threshold": 0.5,
        "recruitment_threshold": 0.25,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "n_runs": len(rows),
        "complete": sum(bool(row["complete"]) for row in rows),
        "min_env_steps": min(int(row["env_steps"]) for row in rows),
        "max_env_steps": max(int(row["env_steps"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
