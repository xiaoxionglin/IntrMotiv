"""Validate and merge bounded exact-start source shards for the 75M comparison.

This is a thin adapter for the canonical landmark-matched-commands-v1 trial
contract. Source eligibility and reset seeds come from a panel independently
observed in two completed fixed-field runs, never from intervention success.
Bulk trial files stay in the allocated workspace; only compact summaries and
hashes are copied into the vault.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_outputs(root: Path) -> list[Path]:
    return sorted(root.glob("*/intervention_summary.json"))


def copy_completed(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for summary in completed_outputs(source):
        target = destination / summary.parent.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(summary.parent, target,
                        ignore=shutil.ignore_patterns("*.npz", "*.png", "*.log"))


def source_panel(path: Path) -> dict[int, list[int]]:
    rows = pd.read_csv(path).sort_values(["source", "repeat"])
    assert len(rows) == 32
    assert rows.source_panel_verified_in.eq("episode_s99_and_finite_s123").all()
    panel = {}
    for source in range(4):
        subset = rows.loc[rows.source.eq(source)]
        assert subset.repeat.tolist() == list(range(8))
        panel[source] = subset.prefix_seed.astype(int).tolist()
        assert len(set(panel[source])) == 8
    return panel


def load_shard(record: dict, panel: dict[int, list[int]]) -> tuple[pd.DataFrame, dict, dict]:
    output = Path(record["output_dir"])
    summaries = completed_outputs(output / "raw")
    assert len(summaries) == 1, (output, len(summaries))
    summary_path = summaries[0]
    trial_path = summary_path.with_name("intervention_trials.csv")
    summary = json.loads(summary_path.read_text())
    trials = pd.read_csv(trial_path)
    source = int(record["source_id"])
    expected = panel[source]
    assert summary["protocol"] == "landmark-matched-commands-v1"
    assert summary["focus_sources"] == [source]
    assert summary["discovery_seeds"] == expected
    assert summary["checkpoint"] == record["checkpoint"]
    assert summary["exact_start_verified"] and summary["policy_frozen"]
    assert summary["graph_frozen"] and trials.exact_start_verified.all()
    assert summary["sources_evaluated"] == 1 and summary["supported_sources"] == 1
    assert summary["starts_evaluated"] == 8 and summary["paired_comparisons"] == 24
    assert summary["rows"] == len(trials) == 72
    assert not summary["missing"], (output, summary["missing"])
    assert trials.source.eq(source).all()
    starts = trials[["source", "repeat", "prefix_seed"]].drop_duplicates().sort_values("repeat")
    assert starts.repeat.tolist() == list(range(8))
    assert starts.prefix_seed.astype(int).tolist() == expected
    for (_, _, _), group in trials.groupby(["source", "repeat", "target"]):
        assert len(group) == 3 and group.commanded.sum() == 1
        assert group.command.nunique() == 3
    provenance = dict(
        protocol=record["protocol"], seed=int(record["seed"]), source_id=source,
        job_id=record["job_id"], checkpoint=record["checkpoint"],
        summary_sha256=sha256(summary_path), trials_sha256=sha256(trial_path),
        decisions=summary["decisions"], rows=len(trials),
    )
    return trials, summary, provenance


def merge_group(records: list[dict], panel: dict[int, list[int]],
                panel_hash: str, destination: Path) -> list[dict]:
    assert {int(record["source_id"]) for record in records} == set(range(4))
    loaded = [load_shard(record, panel) for record in sorted(records, key=lambda row: int(row["source_id"]))]
    trials = pd.concat([item[0] for item in loaded], ignore_index=True)
    summaries = [item[1] for item in loaded]
    for key in ("condition", "seed", "study_id", "study_sha256", "checkpoint",
                "checkpoint_frames", "manifest", "manifest_row", "horizons",
                "fixed_evaluation_horizon", "target_selection", "goal_count"):
        assert all(summary[key] == summaries[0][key] for summary in summaries)
    assert len(trials) == 288 and trials[["source", "repeat", "command", "target"]].duplicated().sum() == 0
    differences = []
    for _, group in trials.groupby(["source", "repeat", "target"]):
        commanded = group.loc[group.commanded.eq(True)]
        alternatives = group.loc[group.commanded.eq(False)]
        assert len(commanded) == 1 and len(alternatives) == 2
        differences.append(float(commanded.hit.iloc[0]) - float(alternatives.hit.mean()))
    assert len(differences) == 96
    merged = summaries[0].copy()
    merged.update(
        max_sources=4,
        supported_sources=4,
        source_panel_coverage=1.0,
        sources_evaluated=4,
        starts_evaluated=32,
        paired_comparisons=96,
        paired_arrival_lift=float(np.mean(differences)),
        rows=len(trials),
        censored_rows=sum(int(summary["censored_rows"]) for summary in summaries),
        decisions=sum(int(summary["decisions"]) for summary in summaries),
        ambiguous_discovery_observations=sum(int(summary["ambiguous_discovery_observations"]) for summary in summaries),
        ambiguous_trial_observations=sum(int(summary["ambiguous_trial_observations"]) for summary in summaries),
        mean_initial_action_total_variation=float(np.mean([
            summary["mean_initial_action_total_variation"] for summary in summaries
        ])),
        focus_sources=None,
        discovery_seeds=None,
        source_seed_panel_sha256=panel_hash,
        source_shard_job_ids=[record["job_id"] for record in records],
        trial_status="paired_trials_available",
        missing=[],
    )
    label = completed_outputs(Path(records[0]["output_dir"]) / "raw")[0].parent.name
    target = destination / label
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    trials.sort_values(["source", "repeat", "target", "command"]).to_csv(
        target / "intervention_trials.csv", index=False
    )
    (target / "intervention_summary.json").write_text(json.dumps(merged, indent=2) + "\n")
    return [item[2] for item in loaded]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", required=True, type=Path)
    parser.add_argument("--source-panel", required=True, type=Path)
    parser.add_argument("--finite-original", required=True, type=Path)
    parser.add_argument("--episode-original", required=True, type=Path)
    parser.add_argument("--c05-original", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    panel = source_panel(args.source_panel)
    with args.submission_manifest.open(newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert len(records) == 16
    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        key = (record["protocol"], int(record["seed"]))
        assert key in {("episode", 8), ("episode", 123), ("finite", 8), ("finite", 99)}
        grouped.setdefault(key, []).append(record)
    assert len(grouped) == 4 and all(len(group) == 4 for group in grouped.values())
    for protocol, original in (("finite", args.finite_original),
                               ("episode", args.episode_original),
                               ("c05", args.c05_original)):
        copy_completed(original, args.output_root / protocol)
    provenance = []
    for (protocol, _), group in sorted(grouped.items()):
        provenance.extend(merge_group(group, panel, sha256(args.source_panel),
                                      args.output_root / protocol))
    pd.DataFrame(provenance).sort_values(["protocol", "seed", "source_id"]).to_csv(
        args.output_root / "source_shard_raw_provenance.csv", index=False
    )
    print("Merged 16 validated shards into four complete fixed-field panels")


if __name__ == "__main__":
    main()
