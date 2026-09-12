#!/bin/bash
#SBATCH --job-name=intrmotiv-exact-restore
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller-exact-restore-%j.out
#SBATCH --error=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller-exact-restore-%j.err
set -euo pipefail
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_padded_20260912
export PYTHONPATH="$PWD:/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/matplotlib
export TMPDIR=/work/classic/fr_xl1014-train/tmp/restore_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
for index in 0 1 2 3 4 5; do
 /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller_exact_restore.py "$index"
done
