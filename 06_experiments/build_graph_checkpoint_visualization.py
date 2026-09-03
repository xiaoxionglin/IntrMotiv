#!/usr/bin/env python3
"""Build the inline learned-DG-graph visualization from checkpoint JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEEP = {
    "run", "backbone", "d", "half_life_k", "seed", "env_steps", "complete",
    "policy_confidence", "policy_time", "node_visits", "passive_confidence",
    "passive_time", "recruitment_confidence", "recruitment_time",
    "birth_support", "row_assignments", "recruitment_total", "repeat_total",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    compact = {
        "snapshot_utc": payload["snapshot_utc"],
        "policy_known_threshold": payload["policy_known_threshold"],
        "recruitment_threshold": payload["recruitment_threshold"],
        "runs": [{key: run[key] for key in KEEP} for run in payload["runs"]],
    }
    template = args.template.read_text()
    marker = "__GRAPH_DATA__"
    if template.count(marker) != 1:
        raise SystemExit(f"Expected exactly one {marker} marker")
    rendered = template.replace(marker, json.dumps(compact, separators=(",", ":")))
    if len(rendered.encode()) >= 1_000_000:
        raise SystemExit("Rendered visualization exceeds the 1 MB inline limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output} ({len(rendered.encode()):,} bytes)")


if __name__ == "__main__":
    main()
