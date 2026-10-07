#!/usr/bin/env bash
set -euo pipefail

manifest=${1:?manifest required}
index=${2:?zero-based row required}
root=${BF_WORKSPACE:?active workspace root required}
output=${BF_OUTPUT:?output root required}
runtime=${BF_RUNTIME:?runtime source required}
overlay=${BF_OVERLAY:?probe source required}
for path in "$manifest" "$output" "$overlay"; do
  case "$path" in "$root"/*) ;; *) echo "Path outside active workspace: $path" >&2; exit 2 ;; esac
done
export TMPDIR="$output/tmp"
export DMLAB_CACHE_DIR="$output/dmlab_cache"
export XDG_CACHE_HOME="$output/cache"
export WANDB_DIR="$output/wandb"
export WANDB_MODE=disabled
export TORCH_HOME="$root/IntrMotiv/SF_hipposlam/runtime/cache/torch"
export PYTHONPATH="$overlay:$runtime${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$TORCH_HOME"
cd "$runtime"
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python - "$manifest" "$index" "$output" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

manifest, index, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
with manifest.open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
row = rows[index]
decisions = __import__("os").environ.get("BF_DECISIONS_OVERRIDE", row["decisions"])
command = [sys.executable, "-m", "hpc_runs.behavior_field_probe",
           "--run-dir", row["run_dir"], "--checkpoint", row["checkpoint"],
           "--policy", row["policy"], "--eval-seed", row["eval_seed"],
           "--decisions", decisions, "--output", str(output / "raw" / row["label"]),
           "--workspace-root", __import__("os").environ["BF_WORKSPACE"]]
subprocess.run(command, check=True)
PY
