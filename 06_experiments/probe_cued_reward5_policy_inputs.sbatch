#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --account=bw21j002
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

run_dir=${1:?run directory}
checkpoint=${2:?checkpoint}
output=${3:?output JSON}
decisions=${4:-1024}

workspace_root=/work/classic/fr_xl1014-fixed-reward-transfer
analysis_root=$workspace_root/IntrMotiv/SF_hipposlam/train_dir/analysis/cued_reward5_frozen_dg_controls_20260925/input_sensitivity_20260926
export INTRMOTIV_WORKSPACE_ROOT=$workspace_root
export TMPDIR=$analysis_root/tmp/$SLURM_JOB_ID
export DMLAB_CACHE_DIR=$analysis_root/dmlab_cache/$SLURM_JOB_ID
export XDG_CACHE_HOME=$analysis_root/cache/$SLURM_JOB_ID
export WANDB_MODE=disabled
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$(dirname "$output")"

cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_cued_reward5_frozen_dg_20260925
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  "$analysis_root/probe_cued_reward5_policy_inputs.py" \
  --run-dir "$run_dir" --checkpoint "$checkpoint" --output "$output" \
  --decisions "$decisions"
