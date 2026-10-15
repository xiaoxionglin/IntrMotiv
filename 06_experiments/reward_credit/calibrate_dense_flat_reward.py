"""Pin a mean-matched constant for the dense C01 policy reward.

The reward summaries come from the learner's valid-decision batch. The sum and
mean tags certify the number of decisions represented by each logged block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hpc_runs.intrmotiv_study import load_study


REWARD_MEAN_TAG = "intrmotiv/reward/intrinsic_mean"
REWARD_SUM_TAG = "intrmotiv/reward/intrinsic_sum"
REWARD_NONZERO_TAG = "intrmotiv/reward/intrinsic_nonzero_fraction"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_by_step(accumulator: EventAccumulator, tag: str) -> dict[int, float]:
    events = accumulator.Scalars(tag)
    steps = [int(event.step) for event in events]
    if len(steps) != len(set(steps)):
        raise ValueError(f"Duplicate steps in {tag}; audit resumes before calibration")
    return {int(event.step): float(event.value) for event in events}


def calibrate(summary_dir: Path, study_path: Path, job_id: int, high_frame: int) -> dict:
    study = load_study(study_path)
    runs = study.expand_runs()
    if len(runs) != 1 or runs[0].seed != 99:
        raise ValueError("Reference StudySpec must have exactly the seed-99 temporal run")
    if runs[0].factors.get("flat_policy_reward") != "TEMPORAL":
        raise ValueError("Reference run must use temporal flat-policy reward")
    event_files = sorted(summary_dir.glob("events*"))
    if len(event_files) != 1:
        raise ValueError("Expected one fresh event file; audit any restarts explicitly")
    accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    if not {REWARD_MEAN_TAG, REWARD_SUM_TAG, REWARD_NONZERO_TAG} <= available:
        raise ValueError("Reward mean/sum/nonzero tags are missing")
    means = scalar_by_step(accumulator, REWARD_MEAN_TAG)
    sums = scalar_by_step(accumulator, REWARD_SUM_TAG)
    nonzero = scalar_by_step(accumulator, REWARD_NONZERO_TAG)
    selected_steps = sorted(step for step in means if 0 < step <= high_frame)
    if (len(selected_steps) < 30
            or selected_steps != sorted(step for step in sums if 0 < step <= high_frame)
            or selected_steps != sorted(step for step in nonzero if 0 < step <= high_frame)):
        raise ValueError("Reference reward window is incomplete or scalar steps differ")

    valid_per_block = 2048  # The pinned C01 StudySpec uses one valid 2,048-decision batch.
    eligible_counts = []
    for step in selected_steps:
        mean, total, fraction = means[step], sums[step], nonzero[step]
        if not all(math.isfinite(value) for value in (mean, total, fraction)):
            raise ValueError(f"Nonfinite reward at frame {step}")
        if not math.isclose(total, mean * valid_per_block, rel_tol=2e-6, abs_tol=2e-3):
            raise ValueError(f"Reward mean/sum denominator differs from {valid_per_block} at {step}")
        eligible = round(fraction * valid_per_block)
        if not 0 <= eligible <= valid_per_block or not math.isclose(
            fraction * valid_per_block, eligible, rel_tol=0, abs_tol=2e-3
        ):
            raise ValueError(f"Invalid reward-eligible decision count at {step}")
        if eligible == 0 and not math.isclose(total, 0.0, abs_tol=2e-3):
            raise ValueError(f"Reward appears without an eligible decision at {step}")
        eligible_counts.append(eligible)
    values = [means[step] for step in selected_steps]
    total_eligible = sum(eligible_counts)
    if total_eligible == 0:
        raise ValueError("No reward-eligible decisions in the calibration window")
    total_reward = math.fsum(sums[step] for step in selected_steps)
    constant = total_reward / total_eligible
    if not 0 < constant <= 7.1 or max(values) <= min(values):
        raise ValueError("The flat temporal reward is not positive and variable as expected")
    event_file = event_files[0]
    return {
        "schema": "intrmotiv/flat-policy-reward-calibration/v2",
        "study_id": study.study_id,
        "study_sha256": study.fingerprint,
        "study_schema": study.provenance()["schema"],
        "workflow_version": study.declared_workflow_version,
        "run_name": runs[0].name,
        "job_id": job_id,
        "event_file": str(event_file),
        "event_file_sha256": file_sha256(event_file),
        "window_frames": [0, high_frame],
        "first_logged_step": selected_steps[0],
        "last_logged_step": selected_steps[-1],
        "logged_blocks": len(selected_steps),
        "valid_decisions_per_block": valid_per_block,
        "valid_decisions_total": len(selected_steps) * valid_per_block,
        "reward_eligible_decisions": total_eligible,
        "reward_eligible_fraction": total_eligible / (len(selected_steps) * valid_per_block),
        "unconditional_reward_mean": total_reward / (len(selected_steps) * valid_per_block),
        "reward_mean_min": min(values),
        "reward_mean_max": max(values),
        "flat_policy_reward_constant": constant,
        "constant_unit": "post-scale intrinsic reward per reward-eligible policy decision",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_dir", type=Path)
    parser.add_argument("study_path", type=Path)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--high-frame", type=int, default=2_000_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = calibrate(args.summary_dir, args.study_path, args.job_id, args.high_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"constant={result['flat_policy_reward_constant']:.12g} from {result['logged_blocks']} blocks")


if __name__ == "__main__":
    main()
