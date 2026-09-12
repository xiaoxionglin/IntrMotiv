#!/bin/bash
#SBATCH --job-name=intrmotiv-replay-profile
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller-stable-head-updates-%j.out
#SBATCH --error=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller-stable-head-updates-%j.err
set -euo pipefail
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_stable_head_20260912
export PYTHONPATH="$PWD:/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1:/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/matplotlib
export TMPDIR=/work/classic/fr_xl1014-train/tmp/restore_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller_stable_head_updates.py
