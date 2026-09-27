#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --account=bw21j002
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail

run_dir=${1:?run directory}
checkpoint=${2:?checkpoint}
output=${3:?output NPZ}
panel=${4:?common observation panel}
workspace_root=/work/classic/fr_xl1014-corridor-geometry
analysis_root=$workspace_root/IntrMotiv/SF_hipposlam/train_dir/analysis/A0_poster_20260926
runtime_source=/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
export INTRMOTIV_WORKSPACE_ROOT=$workspace_root
export TMPDIR=$analysis_root/tmp/$SLURM_JOB_ID
export DMLAB_CACHE_DIR=$analysis_root/dmlab_cache/$SLURM_JOB_ID
export XDG_CACHE_HOME=$analysis_root/cache/$SLURM_JOB_ID
export WANDB_MODE=disabled
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$runtime_source${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$(dirname "$output")"
cd "$runtime_source"
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  "$analysis_root/collect_poster_population_kernels.py" \
  --run-dir "$run_dir" --checkpoint "$checkpoint" --panel "$panel" --output "$output"
