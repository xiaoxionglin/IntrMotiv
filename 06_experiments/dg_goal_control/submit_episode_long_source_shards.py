"""Review and submit missing 75M fixed-field exact-start evaluator shards.

This study-specific adapter consumes original immutable intervention manifests
and the outcome-blind source reset seeds verified in two complete panels. The
first episode/seed-8/source-0 shard is separately qualified and must be
validated before the remaining fifteen are released.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


WORK = Path("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir")
ANALYSIS = WORK / "analysis"
ROOT = ANALYSIS / "episode_long_unified_20261009/exact_start_shards_75m_20261009"
SOURCE = Path("/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_orthogonal_exact_start_seed_panel_20261009")
RUNNER = SOURCE / "sf_working_directories/IntrMotiv/evaluation/run_target_control_intervention_single.sh"
SOURCE_PANEL = ANALYSIS / "episode_long_unified_20261009/source_seed_panel_75m.csv"
MANIFESTS = {
    "episode": ANALYSIS / "orthogonal_film_75m_eval_20261008/episode/telemetry/intervention_manifest.tsv",
    "finite": ANALYSIS / "orthogonal_film_finite_20261008/telemetry/intervention_manifest.tsv",
}
PANEL = (("episode", 8, 0), ("episode", 123, 2), ("finite", 8, 3), ("finite", 99, 4))
QUALIFICATION_OUTPUT = ROOT / "qual_seeded_episode_s8_source0"
QUALIFICATION_JOB_ID = "8322566"


def read_row(path: Path, index: int) -> dict:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))[index]


def qualified_record(checkpoint: str, seeds: list[int]) -> dict:
    summaries = list((QUALIFICATION_OUTPUT / "raw").glob("*/intervention_summary.json"))
    assert len(summaries) == 1, "The first source shard has not completed"
    summary = json.loads(summaries[0].read_text())
    expected = {
        "condition": "OELDG_ORACLE_FILM", "seed": 8,
        "checkpoint": checkpoint, "focus_sources": [0],
        "discovery_seeds": seeds, "supported_sources": 1,
        "sources_evaluated": 1, "starts_evaluated": 8,
        "paired_comparisons": 24, "rows": 72,
    }
    for key, value in expected.items():
        assert summary[key] == value, (key, summary[key], value)
    assert summary["exact_start_verified"] and summary["policy_frozen"]
    assert summary["graph_frozen"] and not summary["missing"]
    trial_file = summaries[0].parent / "intervention_trials.csv"
    with trial_file.open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    assert len(trials) == 72
    starts = {(int(row["repeat"]), int(row["prefix_seed"])) for row in trials}
    assert starts == set(enumerate(seeds))
    groups: dict[tuple[int, int], list[dict]] = {}
    for trial in trials:
        assert int(trial["source"]) == 0 and trial["exact_start_verified"] == "True"
        groups.setdefault((int(trial["repeat"]), int(trial["target"])), []).append(trial)
    assert len(groups) == 24
    for group in groups.values():
        assert len(group) == 3 and len({row["command"] for row in group}) == 3
        assert sum(row["commanded"] == "True" for row in group) == 1
    return dict(protocol="episode", seed=8, source_id=0, row_index=0,
                checkpoint=checkpoint,
                discovery_seeds=":".join(str(seed) for seed in seeds),
                output_dir=str(QUALIFICATION_OUTPUT), job_id=QUALIFICATION_JOB_ID)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--include-qualified-shard", action="store_true",
                        help="Rerun episode seed 8 source 0 if the qualification was incomplete")
    args = parser.parse_args()
    with SOURCE_PANEL.open(newline="") as handle:
        panel = list(csv.DictReader(handle))
    assert len(panel) == 32
    selected_seeds = {}
    for source_id in range(4):
        selected = [row for row in panel if int(row["source"]) == source_id]
        assert [int(row["repeat"]) for row in selected] == list(range(8))
        assert all(row["source_panel_verified_in"] == "episode_s99_and_finite_s123"
                   for row in selected)
        selected_seeds[source_id] = [int(row["prefix_seed"]) for row in selected]
    records = []
    for protocol, seed, index in PANEL:
        manifest = MANIFESTS[protocol]
        row = read_row(manifest, index)
        expected_condition = "OELDG_ORACLE_FILM" if protocol == "episode" else "OFDG_ORACLE"
        assert row["condition"] == expected_condition and int(row["seed"]) == seed
        assert Path(row["checkpoint"]).is_file()
        if args.submit and protocol == "episode" and seed == 8 and not args.include_qualified_shard:
            records.append(qualified_record(row["checkpoint"], selected_seeds[0]))
        for source_id in range(4):
            if protocol == "episode" and seed == 8 and source_id == 0 and not args.include_qualified_shard:
                continue
            seed_export = ":".join(str(seed_value) for seed_value in selected_seeds[source_id])
            name = f"{protocol}_s{seed}_source{source_id}"
            output = ROOT / name
            command = [
                "sbatch", "--parsable", "--partition=cpu", "--cpus-per-task=4",
                "--mem=16G", "--time=03:00:00", f"--job-name=eldg_{protocol[0]}{seed}_{source_id}",
                f"--output={output}/slurm/%j.out", f"--error={output}/slurm/%j.err",
                "--export=ALL,"
                f"INTRMOTIV_RUNTIME_SOURCE={SOURCE},"
                f"INTERVENTION_FOCUS_SOURCES={source_id},"
                f"INTERVENTION_DISCOVERY_SEEDS={seed_export},"
                "PLACE_FIELD_MAX_FRAMES=200000",
                str(RUNNER), str(manifest), str(index), str(output),
            ]
            print(shlex.join(command), flush=True)
            if args.submit:
                (output / "slurm").mkdir(parents=True, exist_ok=False)
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                records.append(dict(protocol=protocol, seed=seed, source_id=source_id,
                                    row_index=index, checkpoint=row["checkpoint"],
                                    discovery_seeds=seed_export,
                                    output_dir=str(output), job_id=result.stdout.strip()))
    if args.submit:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = ROOT / f"submission_manifest_{timestamp}.tsv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        print(f"Recorded {len(records)} source shards including the qualified first shard: {path}", flush=True)


if __name__ == "__main__":
    main()
