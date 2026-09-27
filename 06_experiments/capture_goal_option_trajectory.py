"""Capture exact option-start and target-hit flags during a frozen DG probe.

The canonical Sample Factory ``rollout_dg`` remains the rollout backend. This
adapter only registers a read-only hook on the policy core and writes aligned
event columns beside the canonical pose stream. Run it on a DMLab compute node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sf_working_directories.IntrMotiv.dmlab.hrl_controllable_graph import HRLStateLayout
from sf_working_directories.IntrMotiv.evaluation import place_fields


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(run_dir: Path, checkpoint: Path, output_dir: Path, decisions: int) -> dict:
    """Run the canonical frozen probe and align core flags with observation poses."""
    records: list[np.ndarray] = []
    handles = []
    original_loader = place_fields.load_policy_env

    def load_with_event_hook(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        actor = loaded[3]
        core = dict(actor.named_modules())["core"]
        if not getattr(core, "hrl_enabled", False):
            raise ValueError("This checkpoint has no HRL options")
        layout = HRLStateLayout(int(core.Hippo_n_feature))

        def record_flags(module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError("Unexpected core output; cannot align option events")
            hrl_state = module._split_state(output[1])[1]
            values = hrl_state[:, [layout.option_reset, layout.target_hit,
                                   layout.option_expired]].detach().cpu().numpy()
            records.append(values.copy())

        handles.append(core.register_forward_hook(record_flags))
        return loaded

    try:
        with patch.object(place_fields, "load_policy_env", load_with_event_hook):
            _cfg, loaded_checkpoint, pose, _dg, _logits, arrays = place_fields.rollout_dg(
                run_dir, decisions, deterministic=False, checkpoint_rank=0,
                checkpoint_path=checkpoint,
            )
    finally:
        for handle in handles:
            handle.remove()

    if not records:
        raise ValueError("The core hook captured no option events")
    flags = np.concatenate(records, axis=0)
    if len(flags) != len(pose):
        raise ValueError(f"Option flags and poses differ: {len(flags)} versus {len(pose)}")
    if "behavior_goal_ids" not in arrays or len(arrays["behavior_goal_ids"]) != len(pose):
        raise ValueError("Canonical evaluator did not expose aligned goal identities")
    enriched = pose.copy()
    enriched["option_start"] = flags[:, 0] > 0
    enriched["goal_hit"] = flags[:, 1] > 0
    enriched["option_timeout"] = flags[:, 2] > 0
    enriched["new_goal_id"] = np.asarray(arrays["behavior_goal_ids"], dtype=int)
    events = enriched[enriched.option_start | enriched.goal_hit].copy()
    if not enriched.option_start.any():
        raise ValueError("No option start was observed; check the HRL state hook")

    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "pose_events.csv", index=False)
    events.to_csv(output_dir / "option_events.csv", index=False)
    summary = {
        "schema": "intrmotiv/goal-option-trajectory/v1",
        "run_dir": str(run_dir),
        "checkpoint": str(loaded_checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "requested_decisions": decisions,
        "recorded_observations": len(enriched),
        "option_starts": int(enriched.option_start.sum()),
        "goal_hits": int(enriched.goal_hit.sum()),
        "overlapping_start_and_hit": int((enriched.option_start & enriched.goal_hit).sum()),
        "option_timeouts": int(enriched.option_timeout.sum()),
        "reset_segments": int(enriched.num_traj.nunique()),
        "alignment": "Flags and x/y are from the same observation-time core call; a hit can also start the next option",
        "event_semantics": {
            "option_start": "HRLStateLayout.option_reset in the core output state",
            "goal_hit": "HRLStateLayout.target_hit in the core output state",
            "option_timeout": "HRLStateLayout.option_expired in the core output state",
        },
    }
    (output_dir / "event_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(capture(args.run_dir, args.checkpoint, args.output_dir, args.decisions), indent=2))


if __name__ == "__main__":
    main()
