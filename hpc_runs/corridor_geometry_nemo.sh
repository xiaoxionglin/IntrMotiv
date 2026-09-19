#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --time=$TIMEOUT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --mem=$MEMORY
set -euo pipefail
source /home/fr/fr_xl1014/miniforge3/etc/profile.d/conda.sh
conda activate SFgit
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_corridor_20260919
export PYTHONPATH="$$PWD:/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/corridor_20260919/cache
export TORCH_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache/torch
export MPLCONFIGDIR="$$XDG_CACHE_HOME/matplotlib"
export WANDB_CACHE_DIR="$$XDG_CACHE_HOME/wandb"
export WANDB_DATA_DIR="$$XDG_CACHE_HOME/wandb_data"
export WANDB_DIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/corridor_geometry_20260919/wandb
export TMPDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/corridor_20260919/tmp/$${SLURM_JOB_ID}
mkdir -p "$$TMPDIR" "$$XDG_CACHE_HOME" "$$MPLCONFIGDIR" "$$WANDB_CACHE_DIR" "$$WANDB_DATA_DIR" "$$WANDB_DIR"
exec python -m sf_working_directories.IntrMotiv.dmlab.train_hipposlam $CMD
