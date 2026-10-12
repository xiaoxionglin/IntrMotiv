#!/usr/bin/env bash
# Ordinary compute-node job for one validated terminal-manifest row.
set -euo pipefail

manifest=${1:?manifest required}
row=${2:?zero-based row required}
panel=${3:?panel required}
dg_root=${4:?archived DG root required}
output=${5:?output root required}
source=${6:?pinned source root required}
runtime=${7:?runtime checkout required}
workspace=/work/classic/fr_xl1014-corridor-geometry
for path in "$manifest" "$panel" "$dg_root" "$output" "$source"; do
  case "$path" in "$workspace"/*) ;; *) echo "Path outside active workspace: $path" >&2; exit 2 ;; esac
done
[[ $row =~ ^[0-9]+$ ]] || { echo "Invalid row: $row" >&2; exit 2; }
[[ -f $manifest && -f $panel && -d $dg_root && -f $source/hpc_runs/common_panel_layer_probe.py ]] || exit 2
[[ -f $runtime/sf_working_directories/IntrMotiv/evaluation/place_fields.py ]] || exit 2
export TMPDIR=$output/tmp
export DMLAB_CACHE_DIR=$output/dmlab_cache
export XDG_CACHE_HOME=$output/cache
export WANDB_DIR=$output/wandb
export TORCH_HOME=$workspace/IntrMotiv/SF_hipposlam/runtime/cache/torch
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
terminal_binding=$workspace/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1
if [[ -d $terminal_binding ]]; then
  export PYTHONPATH=$source:$runtime:$terminal_binding${PYTHONPATH:+:$PYTHONPATH}
else
  export PYTHONPATH=$source:$runtime${PYTHONPATH:+:$PYTHONPATH}
fi
mkdir -p "$output/raw" "$output/slurm" "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"
cd "$runtime"
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python "$source/hpc_runs/common_panel_layer_probe.py" \
  --manifest "$manifest" --row "$row" --panel "$panel" \
  --dg-root "$dg_root" --output-root "$output" --workspace-root "$workspace"
