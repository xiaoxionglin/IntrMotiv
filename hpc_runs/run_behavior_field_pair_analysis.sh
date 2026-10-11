#!/usr/bin/env bash
set -euo pipefail

workspace=${BF_WORKSPACE:?active workspace required}
output=${BF_OUTPUT:?workspace analysis output required}
overlay=${BF_OVERLAY:?workspace source overlay required}
runtime=${BF_RUNTIME:?runtime source required}
manifest=${1:?manifest required}
raw_root=${2:?raw probe root required}
cohort=${3:?cohort required}
provenance=${4:-}
for path in "$output" "$overlay" "$manifest" "$raw_root"; do
  case "$path" in "$workspace"/*) ;; *) echo "Path outside active workspace: $path" >&2; exit 2 ;; esac
done
if [[ -n "$provenance" ]]; then
  case "$provenance" in "$workspace"/*) ;; *) echo "Provenance outside active workspace" >&2; exit 2 ;; esac
fi
export TMPDIR="$output/tmp"
export XDG_CACHE_HOME="$output/cache"
export PYTHONPATH="$overlay:$runtime${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
args=()
if [[ -n "$provenance" ]]; then args+=(--provenance "$provenance"); fi
cd "$runtime"
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.behavior_field_pair_analysis \
  --manifest "$manifest" --raw-root "$raw_root" --output "$output" \
  --cohort "$cohort" "${args[@]}"
