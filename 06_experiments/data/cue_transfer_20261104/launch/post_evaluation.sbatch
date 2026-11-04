#!/bin/bash
#SBATCH --job-name=cue-transfer-evaluation
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
workspace=/work/classic/fr_xl1014-corridor-geometry
release=$workspace/IntrMotiv/source_cue_analysis_20261104
runtime=$workspace/IntrMotiv/SF_hipposlam/runtime
export INTRMOTIV_WORKSPACE_ROOT=$workspace INTRMOTIV_RUNTIME_ROOT=$runtime
export INTRMOTIV_RUNTIME_SOURCE=$release
export PYTHONPATH=$runtime/torch_cuda_2_9_1:$runtime/controller_terminal_binding_v1:$release
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=$workspace/tmp/ctpost_${SLURM_JOB_ID}
export XDG_CACHE_HOME=$runtime/cache MPLCONFIGDIR=$runtime/matplotlib
export TORCH_HOME=$runtime/cache/torch WANDB_MODE=disabled
mkdir -p "$TMPDIR"
cd "$release"
exec /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python hpc_runs/post_cue_transfer_20261104.py --source "$release" --python /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python --submit
