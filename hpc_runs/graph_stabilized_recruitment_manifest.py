"""Generate the production matrix for Graph-Stabilized Recruitment.

This manifest intentionally leaves the corrected-core backbone command bundles
opaque: those bundles are defined in the authoritative NEMO2 checkout and are
not present in this vault repository.  The generated rows contain the exact
new flags and can be joined to the existing C05/C13/C15 bundles there.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


BATCH_NAME = "intrmotiv_graph_stabilized_recruitment_20260903"
BACKBONES = ("C05", "C13", "C15")
REDUNDANCY_THRESHOLDS = (4, 8)
HALF_LIVES = (5_000, 10_000)
SEEDS = (8, 99, 123)


def rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for backbone, threshold, half_life, seed in product(
        BACKBONES, REDUNDANCY_THRESHOLDS, HALF_LIVES, SEEDS
    ):
        result.append({
            "name": f"GSR_{backbone}_D{threshold}_H{half_life // 1000}k_S{seed}",
            "batch": BATCH_NAME,
            "backbone": backbone,
            "seed": seed,
            "args": [
                "--dg_orthogonal_recruitment_mode=graph",
                "--dg_recruitment_connectivity_threshold=0.25",
                f"--dg_recruitment_redundancy_max_steps={threshold}",
                f"--dg_recruitment_passive_half_life_events={half_life}",
                # Policy-buffer HRL uses this same half-life for Tctrl.
                f"--hrl_fast_weight_half_life_options={half_life}",
            ],
            "context_controls": [f"original_{backbone}_seed{control_seed}" for control_seed in SEEDS],
        })
    assert len(result) == 36
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON instead of stdout")
    args = parser.parse_args()
    payload = {"batch": BATCH_NAME, "rows": rows(), "count": 36}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
