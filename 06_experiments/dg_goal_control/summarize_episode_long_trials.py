"""Summarize completed exact-start rows without changing their evaluation contract.

The inputs are copied raw outputs from the canonical
``landmark-matched-commands-v1`` evaluator. The source CSVs and checkpoints
stay in the allocated NEMO2 workspace; this adapter exports only compact
per-window results and file hashes for the unified scientific report.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    provenance: list[dict] = []
    for summary_path in sorted(root.glob("*/intervention_summary.json")):
        trial_path = summary_path.with_name("intervention_trials.csv")
        summary = json.loads(summary_path.read_text())
        trials = pd.read_csv(trial_path)
        assert summary["protocol"] == "landmark-matched-commands-v1"
        assert summary["exact_start_verified"] and summary["policy_frozen"]
        assert summary["graph_frozen"] and len(trials) == summary["rows"]
        assert trials.exact_start_verified.all()
        commanded = trials.loc[trials.commanded.eq(True)]
        alternatives = trials.loc[trials.commanded.eq(False)]
        assert len(commanded) > 0 and len(alternatives) == 2 * len(commanded)
        for horizon in summary["horizons"]:
            column = f"hit_by_{horizon}"
            yes_hits = commanded.loc[commanded[column].eq(True)]
            alt_hits = alternatives.loc[alternatives[column].eq(True)]
            rows.append(
                {
                    "run_name": summary["condition"],
                    "condition": summary["condition"],
                    "seed": summary["seed"],
                    "study_id": summary["study_id"],
                    "study_sha256": summary["study_sha256"],
                    "checkpoint_frames": summary["checkpoint_frames"],
                    "horizon": horizon,
                    "sources_discovered": summary["supported_sources"],
                    "sources_requested": summary["max_sources"],
                    "starts": summary["starts_evaluated"],
                    "paired_targets": len(commanded),
                    "commanded_hits": len(yes_hits),
                    "alternative_hits": len(alt_hits),
                    "alternative_trials": len(alternatives),
                    "commanded_rate": commanded[column].mean(),
                    "alternative_rate": alternatives[column].mean(),
                    "paired_lift": commanded[column].mean() - alternatives[column].mean(),
                    "commanded_median_hit_time": yes_hits.hit_time.median(),
                    "alternative_median_hit_time": alt_hits.hit_time.median(),
                    "censored_rows": summary["censored_rows"],
                    "total_rows": len(trials),
                    "initial_action_tv": summary["mean_initial_action_total_variation"],
                    "populated_physical_distance_rows": trials.physical_start_distance.notna().sum(),
                }
            )
        provenance.append(
            {
                "label": summary_path.parent.name,
                "condition": summary["condition"],
                "seed": summary["seed"],
                "manifest": summary["manifest"],
                "manifest_row": summary["manifest_row"],
                "checkpoint": summary["checkpoint"],
                "summary_sha256": digest(summary_path),
                "trials_sha256": digest(trial_path),
            }
        )
    return rows, provenance


def paired_horizon_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair training seeds while retaining each arm's executed-trial denominator.

    Commands are exactly matched to alternatives *within* an arm. Starts need
    not be identical *between* independently trained finite and episode arms.
    """
    arms = (
        ("Prescribed", "OFDG_ORACLE", "OELDG_ORACLE_FILM"),
        ("Learned-4 detector", "OFDG_LEARNED", "OELDG_LEARNED4_FILM"),
    )
    paired = []
    for family, finite, episode in arms:
        for seed in (8, 99, 123):
            # Finite-arm intervention trials stop at 256 decisions; 900 is
            # available only in the episode-long arm and has no paired control.
            for horizon in (64, 128, 256):
                finite_rows = frame.loc[
                    frame.run_name.eq(finite) & frame.seed.eq(seed) & frame.horizon.eq(horizon)
                ]
                episode_rows = frame.loc[
                    frame.run_name.eq(episode) & frame.seed.eq(seed) & frame.horizon.eq(horizon)
                ]
                assert len(finite_rows) == len(episode_rows) == 1
                left, right = finite_rows.iloc[0], episode_rows.iloc[0]
                assert left.paired_targets == right.paired_targets == 96
                paired.append({
                    "family": family, "seed": seed, "horizon": horizon,
                    "finite_run": finite, "episode_run": episode,
                    "finite_lift": left.paired_lift,
                    "episode_lift": right.paired_lift,
                    "episode_minus_finite_lift": right.paired_lift - left.paired_lift,
                    "finite_commanded_rate": left.commanded_rate,
                    "episode_commanded_rate": right.commanded_rate,
                    "finite_alternative_rate": left.alternative_rate,
                    "episode_alternative_rate": right.alternative_rate,
                    "finite_censored_rows": left.censored_rows,
                    "episode_censored_rows": right.censored_rows,
                    "trials_per_arm": left.total_rows,
                })
    assert len(paired) == 18
    return pd.DataFrame(paired)


def audit_cross_arm_starts(finite_root: Path, episode_root: Path) -> pd.DataFrame:
    """Certify shared physical starts for prescribed fields across arms.

    Learned DG source IDs name separately learned events, so overlap of reset
    seeds is reported but physical-state parity is not assumed for that family.
    """
    selected = {}
    for root in (finite_root, episode_root):
        for summary_path in sorted(root.glob("*/intervention_summary.json")):
            summary = json.loads(summary_path.read_text())
            key = (summary["condition"], int(summary["seed"]))
            if key[0] in ("OFDG_ORACLE", "OELDG_ORACLE_FILM",
                          "OFDG_LEARNED", "OELDG_LEARNED4_FILM"):
                selected[key] = summary_path.with_name("intervention_trials.csv")
    rows = []
    for family, finite, episode in (
        ("Prescribed", "OFDG_ORACLE", "OELDG_ORACLE_FILM"),
        ("Learned-4 detector", "OFDG_LEARNED", "OELDG_LEARNED4_FILM"),
    ):
        for seed in (8, 99, 123):
            starts = []
            for condition in (finite, episode):
                trials = pd.read_csv(selected[(condition, seed)])
                keys = ["source", "repeat", "prefix_seed"]
                assert trials.groupby(keys).start_position.nunique().eq(1).all()
                distinct = trials[["source", "repeat", "prefix_seed", "start_position"]]
                distinct = distinct.drop_duplicates(keys)
                assert len(distinct) == 32
                starts.append(distinct)
            joined = starts[0].merge(starts[1],
                                     on=["source", "repeat", "prefix_seed"],
                                     suffixes=("_finite", "_episode"),
                                     validate="one_to_one")
            distances = np.array([
                np.linalg.norm(
                    np.asarray(json.loads(a), dtype=float) -
                    np.asarray(json.loads(b), dtype=float)
                )
                for a, b in zip(joined.start_position_finite,
                                joined.start_position_episode)
            ])
            if family == "Prescribed":
                assert len(joined) == 32 and np.allclose(distances, 0, atol=1e-5)
            rows.append({
                "family": family, "seed": seed,
                "finite_starts": len(starts[0]),
                "episode_starts": len(starts[1]),
                "matched_source_repeat_reset_keys": len(joined),
                "identical_physical_starts": int((distances < 1e-5).sum()),
                "median_position_distance": float(np.median(distances)) if len(distances) else np.nan,
                "maximum_position_distance": float(np.max(distances)) if len(distances) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-root", required=True, type=Path)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--c05-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    all_rows: list[dict] = []
    all_provenance: list[dict] = []
    for root in (args.finite_root, args.episode_root, args.c05_root):
        rows, provenance = summarize(root)
        all_rows.extend(rows)
        all_provenance.extend(provenance)
    assert all_rows and all_provenance
    frame = pd.DataFrame(all_rows).sort_values(["condition", "seed", "horizon"])
    paired = paired_horizon_contrasts(frame)
    start_parity = audit_cross_arm_starts(args.finite_root, args.episode_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "matched_command_75m_per_horizon.csv", index=False
    )
    paired.to_csv(
        args.output_dir / "matched_command_75m_paired_horizon_effects.csv", index=False
    )
    start_parity.to_csv(
        args.output_dir / "matched_command_75m_start_parity.csv", index=False
    )
    pd.DataFrame(all_provenance).sort_values(["condition", "seed"]).to_csv(
        args.output_dir / "matched_command_raw_provenance.csv", index=False
    )
    print(f"summarized {len(all_provenance)} completed checkpoint outputs")


if __name__ == "__main__":
    main()
