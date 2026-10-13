"""Extract completed FG encoder-pair hit counts from existing TensorBoard logs.

Run on NEMO2 with the already completed terminal manifest. This reads logs
only; it does not run a simulator or change training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "correct_count": "intrmotiv/hrl/control/correct_count",
    "censored_count": "intrmotiv/hrl/control/censored_count",
    "reward_magnitude_mean": "intrmotiv/hrl/control/correct_reward_magnitude_mean",
    "correct_elapsed_mean": "intrmotiv/hrl/control/correct_elapsed_mean",
    "target_hit_event_count": "intrmotiv/hrl/target_hit_event_count",
    "encoder_event_count": "intrmotiv/encoder/dominant_event_count",
}
LOW, HIGH = 90_000_000, 100_000_000
SEEDS = {8, 99, 123}


def run_counts(row: dict[str, str]) -> dict:
    summary = Path(row["run_dir"]) / ".summary" / "0"
    accumulator = EventAccumulator(str(summary), size_guidance={"scalars": 0})
    accumulator.Reload()
    missing = set(TAGS.values()) - set(accumulator.Tags().get("scalars", []))
    if missing:
        raise ValueError(f"{row['run_dir']} lacks tags: {sorted(missing)}")
    histories = {}
    for name, tag in TAGS.items():
        events = [(int(value.step), float(value.value))
                  for value in accumulator.Scalars(tag)
                  if LOW <= int(value.step) <= HIGH]
        history = dict(events)
        if len(history) != len(events):
            raise ValueError(f"Duplicate {tag} update in {row['run_dir']}")
        histories[name] = history
    steps = sorted(set.intersection(*(set(history) for history in histories.values())))
    if len(steps) < 300 or not all(math.isfinite(history[step])
                                  for history in histories.values() for step in steps):
        raise ValueError(f"Incomplete or nonfinite FG history: {row['run_dir']}")
    correct = sum(histories["correct_count"][step] for step in steps)
    eligible = sum(histories["target_hit_event_count"][step] for step in steps)
    if eligible <= 0:
        raise ValueError(f"Zero eligible target-hit events: {row['run_dir']}")
    event_files = sorted(summary.glob("events*"))
    return {
        "condition": row["condition"], "seed": int(row["seed"]),
        "run_dir": row["run_dir"], "updates": len(steps),
        "first_step": steps[0], "last_step": steps[-1],
        "correct_hit_observation_count": correct,
        "target_hit_event_observation_count": eligible,
        "censored_observation_count": sum(
            histories["censored_count"][step] for step in steps),
        "event_files": [{"path": str(path), "size": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns}
                        for path in event_files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        runs = [row for row in csv.DictReader(stream, delimiter="\t")
                if row["family"] == "C15" and int(row["seed"]) in SEEDS]
    expected = {(f"COS27_C15_{credit}", seed)
                for credit in ("TEMPORAL", "CONSTANT") for seed in SEEDS}
    if {(row["condition"], int(row["seed"])) for row in runs} != expected or len(runs) != 6:
        raise ValueError("Expected exactly six completed FG encoder-arm rows")
    result = {"schema": "intrmotiv/focused-canvas-additional-fg-hits/v1",
              "window_low": LOW, "window_high": HIGH,
              "rows": [run_counts(row) for row in runs]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(result['rows'])} completed FG run summaries to {args.output}")


if __name__ == "__main__":
    main()
