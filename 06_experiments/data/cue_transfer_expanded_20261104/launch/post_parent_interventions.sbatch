#!/bin/bash
#SBATCH --job-name=cue-parent-interventions
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
workspace=/work/classic/fr_xl1014-corridor-geometry
release=$workspace/IntrMotiv/source_cue_expanded_analysis_fix_20261104
runtime=$workspace/IntrMotiv/SF_hipposlam/runtime
export INTRMOTIV_WORKSPACE_ROOT=$workspace INTRMOTIV_RUNTIME_ROOT=$runtime
export INTRMOTIV_RUNTIME_SOURCE=$release
export PYTHONPATH=$runtime/torch_cuda_2_9_1:$runtime/controller_terminal_binding_v1:$release
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=$workspace/tmp/cepost_${SLURM_JOB_ID}
export XDG_CACHE_HOME=$runtime/cache MPLCONFIGDIR=$runtime/matplotlib
export TORCH_HOME=$runtime/cache/torch WANDB_MODE=disabled
export WANDB_DIR=$workspace/IntrMotiv/SF_hipposlam/train_dir/analysis/cue_transfer_expanded_20261104/wandb
export DMLAB_CACHE_DIR=$workspace/IntrMotiv/SF_hipposlam/train_dir/analysis/cue_transfer_expanded_20261104/dmlab_cache
mkdir -p "$TMPDIR" "$WANDB_DIR" "$DMLAB_CACHE_DIR"
cd "$release"
exec /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python hpc_runs/post_cue_transfer_20261104.py \
  --source "$release" --python /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  --study "$release/hpc_runs/studies/cue_transfer_20261104.study.json" \
  --heldout-protocols shuffled_cue,zero_ca3,zero_depth --trials 100 --seed-base 61000 --horizon 1800 \
  --heldout-time-limit 08:00:00 --skip-fields --output-root "$workspace/IntrMotiv/SF_hipposlam/train_dir/analysis/cue_transfer_expanded_20261104/parent_interventions" --submit
