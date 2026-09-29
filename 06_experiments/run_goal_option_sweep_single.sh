#!/usr/bin/env bash
set -euo pipefail

# Custom worker for the canonical independent-job place-field submitter.
manifest=${1:?manifest required}
row_index=${2:?row index required}
output_dir=${3:?output directory required}
workspace_root=${INTRMOTIV_WORKSPACE_ROOT:-/work/classic/fr_xl1014-corridor-geometry}
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
runtime_source=${INTRMOTIV_RUNTIME_SOURCE:-/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam}
capture_script=${GOAL_OPTION_CAPTURE_SCRIPT:?staged capture script required}

for path in "$manifest" "$output_dir" "$capture_script"; do
  case "$path" in "$workspace_root"/*) ;; *) echo "Path outside active workspace: $path" >&2; exit 2 ;; esac
done
mapfile -t fields < <("$python" - "$manifest" "$row_index" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
row = rows[int(sys.argv[2])]
for key in ("checkpoint", "run_dir", "label_suffix"):
    print(row[key])
PY
)
checkpoint=${fields[0]}
run_dir=${fields[1]}
label=${fields[2]}
for path in "$checkpoint" "$run_dir"; do
  case "$path" in "$workspace_root"/*) ;; *) echo "Input outside active workspace: $path" >&2; exit 2 ;; esac
done
test -f "$checkpoint"
test -d "$run_dir"
test -f "$capture_script"

export TMPDIR="$workspace_root/tmp/goal_option_${SLURM_JOB_ID:-local}_${row_index}"
export DMLAB_CACHE_DIR="$output_dir/dmlab_cache"
export XDG_CACHE_HOME="$output_dir/cache"
export WANDB_DIR="$output_dir/wandb"
export WANDB_MODE=disabled
export TORCH_HOME="$workspace_root/IntrMotiv/SF_hipposlam/runtime/cache/torch"
mkdir -p "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$output_dir/raw/$label"
export PYTHONPATH="$runtime_source${PYTHONPATH:+:$PYTHONPATH}"
terminal_binding=${INTRMOTIV_TERMINAL_BINDING:-$workspace_root/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1}
if [[ -d $terminal_binding ]]; then
  export PYTHONPATH="$runtime_source:$terminal_binding${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "$runtime_source"
"$python" "$capture_script" \
  --run-dir "$run_dir" --checkpoint "$checkpoint" \
  --output-dir "$output_dir/raw/$label" \
  --decisions "${PLACE_FIELD_MAX_FRAMES:-10000}"
