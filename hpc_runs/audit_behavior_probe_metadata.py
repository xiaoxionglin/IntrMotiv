"""Audit frozen policy-probe identities and workspace paths without loading traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def audit(manifest: Path, raw_root: Path, workspace: Path) -> dict[str, object]:
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    root = workspace.resolve(strict=True)
    complete = 0
    models: dict[tuple[str, str], tuple[str, str]] = {}
    missing = []
    for row in rows:
        folder = raw_root / row["label"]
        trace = folder / "trace.npz"
        metadata_path = folder / "metadata.json"
        if not trace.is_file() or not metadata_path.is_file():
            missing.append(row["label"])
            continue
        metadata = json.loads(metadata_path.read_text())
        if (metadata["policy"] != row["policy"] or
                metadata["eval_seed"] != int(row["eval_seed"]) or
                metadata["decisions"] != int(row["decisions"]) or
                metadata["checkpoint"] != row["checkpoint"] or
                metadata["run_dir"] != row["run_dir"]):
            raise ValueError(f"Manifest/probe metadata mismatch: {row['label']}")
        if metadata["policy"] == "own" and metadata["proposed_executed_disagreements"] != 0:
            raise ValueError(f"Own-policy action mismatch: {row['label']}")
        if metadata["policy"] == "uniform" and metadata["proposed_executed_disagreements"] == 0:
            raise ValueError(f"Uniform arm did not change executed actions: {row['label']}")
        if any(not Path(path).resolve().is_relative_to(root)
               for path in metadata["runtime_paths"].values()):
            raise ValueError(f"Output path escapes active workspace: {row['label']}")
        identity = metadata["checkpoint_sha256"], metadata["frozen_model_sha256"]
        model = row["condition"], row["seed"]
        if model in models and models[model] != identity:
            raise ValueError(f"Checkpoint or frozen weights differ within training model: {model}")
        models[model] = identity
        complete += 1
    return {"schema": "intrmotiv/behavior-probe-audit/v1",
            "expected_probes": len(rows), "complete_probes": complete,
            "missing_labels": missing, "consistent_training_models": len(models),
            "all_complete": complete == len(rows),
            "workspace": str(root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.workspace.resolve(strict=True)):
        raise ValueError("Audit output outside active workspace")
    result = audit(args.manifest, args.raw_root, args.workspace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "missing_labels"}))


if __name__ == "__main__":
    main()
