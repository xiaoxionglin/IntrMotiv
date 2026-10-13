"""Archive the small peak inputs from pinned high-credit 100M NPZ snapshots.

Raw snapshots stay in the NEMO2 workspace. The output contains only occupied-bin
masks, field eligibility, active fractions, peak coordinates, and source hashes.
It can be used to reproduce peak-area summaries without another SSH session.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

from plot_high_credit_compact import HERE, SEEDS, SOURCE_COMMIT, source_table


OUTPUT = HERE / "high_credit_peak_snapshot_inputs.json"
WORKSPACE_PREFIX = (
    "/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/"
    "train_dir/analysis/online_spatial/"
)
REMOTE_PYTHON = "/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python"

REMOTE_SCRIPT = r'''
import base64
import hashlib
import json
from pathlib import Path
import numpy as np

sources = json.loads(base64.b64decode("__ENCODED_SOURCES__"))
records = []
for source in sources:
    path = Path(source["snapshot_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with np.load(path, allow_pickle=False) as artifact:
        if (str(artifact["run_name"].item()) != source["run_name"] or
                int(artifact["target_env_steps"]) != 100_000_000 or
                int(artifact["window_limit"]) != 100_000 or
                int(artifact["grain"]) != 19):
            raise ValueError(f"Snapshot identity or geometry mismatch: {path}")
        occupancy = np.asarray(artifact["occupancy"])
        fractions = np.asarray(artifact["active_fraction"], dtype=float)
        eligible = np.asarray(artifact["field_eligible"], dtype=bool)
        peaks = np.asarray(artifact["field_dominant_peak_xy"], dtype=float)
        bounds = np.asarray(artifact["bounds"], dtype=float)
        if (occupancy.shape != (19, 19) or fractions.shape != (16,) or
                eligible.shape != (16,) or peaks.shape != (16, 2) or
                not np.allclose(bounds, [100, 2000, 100, 2000]) or
                int(artifact["field_min_active_observations"]) != 20 or
                int(artifact["field_min_active_bins"]) != 3):
            raise ValueError(f"Unexpected field contract: {path}")
        if (not np.isfinite(fractions).all() or
                (fractions < 0).any() or (fractions > 1).any()):
            raise ValueError(f"Invalid active fractions: {path}")
        mask = occupancy > 0
        coordinates = [[float(value) if np.isfinite(value) else None
                        for value in pair] for pair in peaks]
        records.append({
            **source,
            "snapshot_sha256": digest.hexdigest(),
            "actual_env_steps": int(artifact["actual_env_steps"]),
            "visited_mask_bits": "".join("1" if value else "0"
                                         for value in mask.flat),
            "visited_bin_count": int(mask.sum()),
            "active_fraction": fractions.tolist(),
            "field_eligible": eligible.tolist(),
            "field_dominant_peak_xy": coordinates,
        })
print(json.dumps({"snapshots": records}, sort_keys=True, allow_nan=False))
'''


def pinned_sources() -> list[dict]:
    """Use the source table's declared NPZ path and condition metadata."""
    sources = []
    expected = {(family, seed, arm) for family in ("C05", "C15")
                for seed in SEEDS for arm in ("TEMPORAL", "CONSTANT")}
    for family in ("C05", "C15"):
        table = source_table(f"{family.lower()}_spatial_per_snapshot.csv")
        for row in table[table.target_env_steps == 100_000_000].itertuples():
            path = str(row.snapshot_path)
            if not path.startswith(WORKSPACE_PREFIX) or not path.endswith(".npz"):
                raise ValueError(f"Snapshot is outside the allocated analysis workspace: {path}")
            sources.append({"family": family, "seed": int(row.seed),
                            "worker_bonus": str(row.worker_bonus),
                            "run_name": str(row.run_name),
                            "snapshot_path": path})
    actual = {(row["family"], row["seed"], row["worker_bonus"])
              for row in sources}
    if len(sources) != len(expected) or actual != expected:
        raise ValueError("Expected exactly 20 high-credit terminal snapshots")
    return sorted(sources, key=lambda row: (row["family"], row["seed"],
                                             row["worker_bonus"]))


def collect(host: str) -> dict:
    sources = pinned_sources()
    encoded = base64.b64encode(json.dumps(sources).encode()).decode()
    program = REMOTE_SCRIPT.replace("__ENCODED_SOURCES__", encoded)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         host, REMOTE_PYTHON, "-"],
        input=program, text=True, capture_output=True, check=True,
    )
    result = json.loads(completed.stdout)
    records = result["snapshots"]
    if [tuple(record[key] for key in ("family", "seed", "worker_bonus", "run_name", "snapshot_path"))
            for record in records] != [tuple(source[key] for key in
                                            ("family", "seed", "worker_bonus", "run_name", "snapshot_path"))
                                      for source in sources]:
        raise ValueError("Remote peak inputs do not match the pinned source table")
    result.update({"schema": "intrmotiv/high-credit-peak-inputs/v1",
                   "source_commit": SOURCE_COMMIT,
                   "description": "Lightweight high-credit 100M own-policy peak inputs; raw NPZs remain on NEMO2"})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="nemo2")
    args = parser.parse_args()
    result = collect(args.host)
    print(f"Archived {len(result['snapshots'])} lightweight peak inputs to {OUTPUT}")
