"""Audit and merge a complete source-sharded matched-command manifest row.

The study manifest declares the source count and intervention settings. Each
shard remains in the NEMO2 workspace; the merged output is written only after
every source and matched-command triplet passes validation. A detector hit is
the evaluator's DG-exclusive target event, not physical arrival.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def merge_row(
    study_manifest_path: Path,
    intervention_manifest_path: Path,
    row_index: int,
    shard_root: Path,
    output_root: Path,
) -> Path:
    study = json.loads(study_manifest_path.read_text())
    settings = study["intervention"]
    manifest = pd.read_csv(intervention_manifest_path, sep="\t")
    require(0 <= row_index < len(manifest), f"Manifest row {row_index} is out of range")
    record = manifest.iloc[row_index]
    label = str(record.label_suffix)
    source_count = int(settings["max_sources"])
    repeats = int(settings["repeats"])
    targets = int(settings["targets_per_source"])
    require(settings["evaluation"] == "landmark-matched-commands-v1", "Unexpected control protocol")
    require(settings["target_selection"] == "all_eligible", "Unexpected target selection")
    require(settings["alternative_targets"] == targets, "Expected all three commands per target")

    frames = []
    summaries = []
    provenance = []
    for source in range(source_count):
        raw = shard_root / f"source_{source:02d}" / "raw" / label
        summary_path = raw / "intervention_summary.json"
        trials_path = raw / "intervention_trials.csv"
        require(summary_path.is_file() and trials_path.is_file(), f"Missing source {source}: {raw}")
        summary = json.loads(summary_path.read_text())
        trials = pd.read_csv(trials_path)
        expected = {
            "protocol": "landmark-matched-commands-v1",
            "schema": study["schema"],
            "workflow_version": study["workflow_version"],
            "study_id": study["study_id"],
            "study_sha256": study["study_sha256"],
            "condition": str(record.condition),
            "seed": int(record.seed),
            "checkpoint_frames": int(record.checkpoint_frames),
            "checkpoint": str(record.checkpoint),
            "manifest": str(intervention_manifest_path),
            "manifest_row": row_index,
            "target_selection": settings["target_selection"],
            "discovery_multiplier": int(settings["discovery_multiplier"]),
            "fixed_evaluation_horizon": int(settings["fixed_evaluation_horizon"]),
            "horizons": list(settings["horizons"]),
            "targets_per_source": targets,
            "repeats_requested": repeats,
            "focus_sources": [source],
            "sources_evaluated": 1,
            "supported_sources": 1,
            "starts_evaluated": repeats,
            "paired_comparisons": repeats * targets,
            "rows": repeats * targets * targets,
            "missing": [],
            "exact_start_verified": True,
            "policy_frozen": True,
            "graph_frozen": True,
        }
        for key, value in expected.items():
            require(summary.get(key) == value, f"Source {source}: {key} mismatch: {summary.get(key)!r}")
        require(len(trials) == expected["rows"], f"Source {source}: trial count mismatch")
        require(trials.source.eq(source).all(), f"Source {source}: wrong source ID")
        require(trials.exact_start_verified.all(), f"Source {source}: unverified start")
        require(not trials[["source", "repeat", "target", "command"]].duplicated().any(),
                f"Source {source}: duplicate command trial")
        require(sorted(trials.repeat.unique().tolist()) == list(range(repeats)),
                f"Source {source}: repeat panel mismatch")
        for repeat, start in trials.groupby("repeat"):
            require(start.prefix_seed.nunique() == 1, f"Source {source}, repeat {repeat}: changed start seed")
            require(start.target.nunique() == targets, f"Source {source}, repeat {repeat}: target count mismatch")
        for key, group in trials.groupby(["source", "repeat", "target"]):
            require(len(group) == targets and int(group.commanded.sum()) == 1,
                    f"Source {source}: incomplete matched-command triplet {key}")
            require(group.command.nunique() == targets, f"Source {source}: duplicate command {key}")
        frames.append(trials)
        summaries.append(summary)
        provenance.append({
            "source": source,
            "summary_sha256": file_sha256(summary_path),
            "trials_sha256": file_sha256(trials_path),
            "decisions": int(summary["decisions"]),
            "rows": len(trials),
            "prefix_seeds": json.dumps(trials[["repeat", "prefix_seed"]]
                                       .drop_duplicates().sort_values("repeat").prefix_seed.astype(int).tolist()),
        })

    all_trials = pd.concat(frames, ignore_index=True)
    require(not all_trials[["source", "repeat", "target", "command"]].duplicated().any(),
            "Duplicate trials across source shards")
    require(len(all_trials) == source_count * repeats * targets * targets,
            "Merged trial count mismatch")
    paired_hits = []
    for key, group in all_trials.groupby(["source", "repeat", "target"]):
        commanded = group.loc[group.commanded, "hit"]
        alternatives = group.loc[~group.commanded, "hit"]
        require(len(commanded) == 1 and len(alternatives) == targets - 1,
                f"Incomplete detector-hit contrast {key}")
        paired_hits.append(float(commanded.iloc[0]) - float(alternatives.mean()))

    merged = summaries[0].copy()
    merged.update(
        focus_sources=None,
        max_sources=source_count,
        supported_sources=source_count,
        source_panel_coverage=1.0,
        sources_evaluated=source_count,
        starts_evaluated=source_count * repeats,
        paired_comparisons=len(paired_hits),
        paired_arrival_lift=sum(paired_hits) / len(paired_hits),
        rows=len(all_trials),
        decisions=sum(int(item["decisions"]) for item in summaries),
        mean_initial_action_total_variation=sum(
            float(item["mean_initial_action_total_variation"]) for item in summaries
        ) / source_count,
        source_shard_count=source_count,
        source_shard_provenance="source_shard_provenance.csv",
    )
    destination = output_root / "raw" / label
    require(not destination.exists(), f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{label}.", dir=destination.parent) as temporary:
        staging = Path(temporary) / label
        staging.mkdir()
        all_trials.sort_values(["source", "repeat", "target", "command"]).to_csv(
            staging / "intervention_trials.csv", index=False
        )
        pd.DataFrame(provenance).to_csv(staging / "source_shard_provenance.csv", index=False)
        (staging / "intervention_summary.json").write_text(json.dumps(merged, indent=2) + "\n")
        staging.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", required=True, type=Path)
    parser.add_argument("--intervention-manifest", required=True, type=Path)
    parser.add_argument("--row", required=True, type=int)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(merge_row(args.study_manifest, args.intervention_manifest, args.row,
                    args.shard_root, args.output_root))


if __name__ == "__main__":
    main()
