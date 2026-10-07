#!/usr/bin/env bash
set -euo pipefail

workspace=${BF_WORKSPACE:?active workspace root required}
study=${BF_STUDY:?study folder required}
runtime=${BF_RUNTIME:?runtime source required}
overlay=${BF_OVERLAY:?analysis source required}
case "$study" in "$workspace"/*) ;; *) echo "Study outside active workspace" >&2; exit 2 ;; esac
case "$overlay" in "$workspace"/*) ;; *) echo "Source outside active workspace" >&2; exit 2 ;; esac
export TMPDIR="$study/analysis_tmp"
export XDG_CACHE_HOME="$study/analysis_cache"
export MPLCONFIGDIR="$study/analysis_cache/matplotlib"
export PYTHONPATH="$overlay:$runtime${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$study/summary"
cd "$runtime"
extra_args=()
if [[ "${BF_PAIRS_ONLY:-0}" == 1 ]]; then
  extra_args+=(--pairs-only)
fi
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.behavior_field_analysis \
  --manifest "$study/behavior_field_expression_20261007.tsv" \
  --raw-root "$study/probes/raw" \
  --output "$study/summary" \
  --workspace-root "$workspace" \
  "${extra_args[@]}"
