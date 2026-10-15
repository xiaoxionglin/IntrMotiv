"""Archive the lightweight peak inputs from the ten NG terminal snapshots.

The raw 100k-decision NPZ files remain on NEMO2. Reuse the checked exporter
for the existing UG/FG snapshots so the same field and geometry contracts
apply before computing a pooled Figure 2 spatial mask.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIOR = ROOT / "06_experiments/reward_credit/results/cosyne27_worker_regimen_boxplots_20261013"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PRIOR))

from hpc_runs.intrmotiv_study import load_study  # noqa: E402
from collect_high_credit_peak_inputs import (  # noqa: E402
    REMOTE_PYTHON,
    REMOTE_SCRIPT,
    WORKSPACE_PREFIX,
)


STUDY = ROOT / "hpc_runs/studies/cosyne27_c01_flat_reward_s5_20261015.study.json"
SPATIAL = HERE / "ng_spatial_per_snapshot.csv"
OUTPUT = HERE / "ng_peak_snapshot_inputs.json"
SEEDS = {8, 17, 41, 99, 123}
ARMS = {"TEMPORAL", "CONSTANT"}


def expected_sources() -> list[dict]:
    """Select exactly the declared ten terminal runs from canonical collection."""
    study = load_study(STUDY)
    rows = pd.read_csv(SPATIAL)
    rows = rows[rows.target_env_steps == 100_000_000]
    expected = {run.name: (run.seed, run.factors["flat_policy_reward"])
                for run in study.expand_runs()}
    if (len(rows) != 10 or set(rows.run_name) != set(expected)
            or set(rows.seed) != SEEDS
            or set(rows.flat_policy_reward) != ARMS):
        raise ValueError("The ten declared NG terminal snapshots are incomplete")
    sources = []
    for row in rows.itertuples():
        seed, arm = expected[row.run_name]
        path = str(row.snapshot_path)
        if (int(row.seed) != seed or row.flat_policy_reward != arm
                or int(row.window_limit) != 100_000
                or not path.startswith(WORKSPACE_PREFIX)
                or not path.endswith(".npz")):
            raise ValueError(f"Snapshot identity or workspace mismatch: {row.run_name}")
        sources.append({"family": "C01", "seed": seed, "arm": arm,
                        "run_name": row.run_name, "snapshot_path": path})
    return sorted(sources, key=lambda row: (row["seed"], row["arm"]))


def collect(host: str) -> dict:
    sources = expected_sources()
    encoded = base64.b64encode(json.dumps(sources).encode()).decode()
    program = REMOTE_SCRIPT.replace("__ENCODED_SOURCES__", encoded)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         host, REMOTE_PYTHON, "-"],
        input=program, text=True, capture_output=True, check=True,
    )
    records = json.loads(completed.stdout)["snapshots"]
    identity = ("family", "seed", "arm", "run_name", "snapshot_path")
    if ([tuple(record[key] for key in identity) for record in records]
            != [tuple(source[key] for key in identity) for source in sources]):
        raise ValueError("Remote peak inputs differ from the declared source rows")
    result = {"schema": "intrmotiv/ng-flat-reward-peak-inputs/v1",
              "study_sha256": load_study(STUDY).fingerprint,
              "snapshots": records}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="nemo2")
    args = parser.parse_args()
    result = collect(args.host)
    print(f"Archived {len(result['snapshots'])} NG terminal peak inputs to {OUTPUT}")
