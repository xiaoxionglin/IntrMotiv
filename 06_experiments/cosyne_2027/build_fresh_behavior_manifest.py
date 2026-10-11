"""Derive frozen own/uniform probes from canonical COSYNE checkpoint records.

The terminal place-field manifest supplies the selected checkpoint. The source
map and training catalog supply the validated StudySpec identity and factors.
No run identity is inferred from a label or silently recovered from a failed run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from hpc_runs.behavior_field_study import DECISIONS, EVAL_SEEDS, FIELDS


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "06_experiments/results/cosyne_2027_behavior_secondary"
TERMINAL = RESULTS / "fresh_terminal_manifest_20261011.tsv"
SOURCES = RESULTS / "fresh_manifest_sources_20261011.csv"
TRAINING = ROOT / "06_experiments/cosyne_2027/training_runs.csv"
MANIFEST = ROOT / "hpc_runs/studies/cosyne27_fresh_behavior_expression_20261011.tsv"
PROVENANCE = RESULTS / "fresh_probe_provenance.csv"
WORKSPACE = Path("/work/classic/fr_xl1014-corridor-geometry")


def read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]],
               *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    terminal = read_rows(TERMINAL, delimiter="\t")
    sources = {row["label_suffix"]: row for row in read_rows(SOURCES)}
    training = {(row["study_id"], row["run_name"]): row
                for row in read_rows(TRAINING) if row["cohort"] == "cosyne"}
    probes: list[dict[str, str]] = []
    provenance: list[dict[str, str]] = []
    model_keys: set[tuple[str, str]] = set()
    for checkpoint in terminal:
        source = sources[checkpoint["label_suffix"]]
        run = training[(source["study_id"], source["run_name"])]
        if (run["study_sha256"] != source["study_sha256"] or
                run["source_base_commit"] != source["source_base_commit"]):
            raise ValueError(f"StudySpec or source fingerprint mismatch: {source['run_name']}")
        if (run["run_dir"] != checkpoint["run_dir"] or
                run["condition"] != checkpoint["condition"] or
                run["family"] != checkpoint["family"] or
                run["attempt_status"] != "verified_100m"):
            raise ValueError(f"Run or completion mismatch: {source['run_name']}")
        key = (run["condition"], run["seed"])
        if key in model_keys:
            raise ValueError(f"Duplicate terminal model: {key}")
        model_keys.add(key)
        if int(checkpoint["target_frames"]) != 100_000_000 or int(checkpoint["checkpoint_frames"]) < 100_000_000:
            raise ValueError(f"Wrong checkpoint age: {key}")
        for field in ("checkpoint", "run_dir"):
            if not Path(checkpoint[field]).is_relative_to(WORKSPACE):
                raise ValueError(f"{field} outside active workspace: {key}")
        for eval_seed in EVAL_SEEDS:
            for policy in ("own", "uniform"):
                label = f"{run['condition']}__s{run['seed']}__e{eval_seed}__{policy}"
                probe = dict(condition=run["condition"], seed=run["seed"],
                             eval_seed=str(eval_seed), policy=policy,
                             decisions=str(DECISIONS),
                             checkpoint_frames=checkpoint["checkpoint_frames"],
                             run_dir=checkpoint["run_dir"], checkpoint=checkpoint["checkpoint"],
                             label=label)
                probes.append(probe)
                provenance.append({"label": label, "cohort": run["cohort"],
                                   "family": run["family"], "training_seed": run["seed"],
                                   "eval_seed": str(eval_seed), "policy": policy,
                                   "encoder_credit": run["encoder_credit"],
                                   "worker_bonus": run["worker_bonus"],
                                   "training_job_id": run["job_id"],
                                   "run_name": run["run_name"],
                                   "study_id": run["study_id"],
                                   "study_schema": run["schema"],
                                   "workflow_version": run["workflow_version"],
                                   "study_sha256": run["study_sha256"],
                                   "source_base_commit": run["source_base_commit"],
                                   "source_patch_sha256": run["source_patch_sha256"],
                                   "checkpoint_frames": checkpoint["checkpoint_frames"],
                                   "checkpoint": checkpoint["checkpoint"]})
    if len(model_keys) != 38 or len(probes) != 152:
        raise ValueError(f"Expected 38 complete fresh models and 152 probes; found {len(model_keys)}, {len(probes)}")
    return probes, provenance


def main() -> None:
    probes, provenance = build()
    write_rows(MANIFEST, FIELDS, probes, delimiter="\t")
    fields = tuple(provenance[0])
    write_rows(PROVENANCE, fields, provenance)
    metadata = {
        "schema": "cosyne27_fresh_behavior_expression_v1",
        "training_unit": "independent training seed within architecture and factor cell",
        "evaluation_repeats": list(EVAL_SEEDS),
        "policies": ["own", "uniform"],
        "decisions_per_probe": DECISIONS,
        "available_models": 38,
        "planned_models": 40,
        "missing_models": "C15 constant-credit seeds 8 and 99; replacement jobs 8329014 and 8329015 running at plan time",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (TERMINAL, SOURCES, TRAINING)},
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(PROVENANCE.read_bytes()).hexdigest(),
        "study_schemas": sorted({row["study_schema"] for row in provenance}),
        "workflow_versions": sorted({row["workflow_version"] for row in provenance}),
    }
    (RESULTS / "fresh_probe_plan.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"models": 38, "probes": len(probes), "sha256": metadata["manifest_sha256"]}))


if __name__ == "__main__":
    main()
