#!/bin/bash
# Compute-node exact-reload certificate for one immutable C15 qualification checkpoint.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 RUN_DIR CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

run_dir=$1
checkpoint=$2
output_dir=$3
source_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_corrected_core_failed_c15_recovery_20261010
runtime_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime

source /home/fr/fr_xl1014/miniforge3/etc/profile.d/conda.sh
conda activate SFgit
export PYTHONPATH="$runtime_root/torch_cuda_2_9_1:$runtime_root/controller_terminal_binding_v1:$source_root:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$runtime_root/cache"
export MPLCONFIGDIR="$runtime_root/matplotlib"
export TMPDIR="/work/classic/fr_xl1014-corridor-geometry/tmp/intrmotiv_reload_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$source_root"
python -m hpc_runs.intrmotiv_study.checkpoint_reload \
  --run-dir "$run_dir" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_dir"
