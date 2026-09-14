"""Summarize completed and active profile directories without loading checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    rows = []
    for directory in sorted(Path(root).iterdir()):
        if not (directory / "manifest.json").is_file():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        samples = []
        for line in (directory / "resources.jsonl").read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # sampler may be in the middle of appending its latest line
        result = json.loads((directory / "summary.json").read_text()) if (directory / "summary.json").exists() else None
        row = dict(profile=directory.name, workers=manifest["workers"], batch_size=manifest["batch_size"],
                   concurrency=manifest["concurrency"], results=result, sample_count=len(samples))
        if samples:
            row.update(cpu_p95=float(np.percentile([s["cpu_percent"] for s in samples], 95)),
                       minimum_available_ram_gib=min(s["available_ram_gib"] for s in samples),
                       latest_runs=samples[-1].get("runs", []), gpus={})
            for gpu in samples[-1]["gpus"]:
                index = gpu["index"]
                history = [g for s in samples for g in s["gpus"] if g["index"] == index]
                row["gpus"][str(int(index))] = dict(
                    utilization_mean=float(np.mean([g["utilization"] for g in history])),
                    utilization_p95=float(np.percentile([g["utilization"] for g in history], 95)),
                    minimum_free_mib=min(g["free_mib"] for g in history))
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
