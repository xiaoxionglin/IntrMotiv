#!/bin/bash
# Fixed stationary comparison; complete prior tests/fit plus separate native qualification are required.
#SBATCH --job-name=fh_stationary_main_fit
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/slurm/main_fit_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/slurm/main_fit_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Run only as the reviewed ordinary Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
source_release=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1
output_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1
launch=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/launch_v3
source_manifest=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/launch_v1/release_manifest.json
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
export INTRMOTIV_WORKSPACE_ROOT=$workspace
export INTRMOTIV_RUNTIME_SOURCE=$source_release INTRMOTIV_RUNTIME_ROOT=$source_release
export PYTHONPATH=$source_release:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=$workspace/tmp/fh_stationary_${SLURM_JOB_ID}
export TMP=$TMPDIR TEMP=$TMPDIR
export XDG_CACHE_HOME=$output_root/cache MPLCONFIGDIR=$output_root/matplotlib
export DMLAB_CACHE_DIR=$output_root/dmlab_cache
export WANDB_DIR=$output_root/wandb WANDB_CACHE_DIR=$output_root/wandb_cache WANDB_DATA_DIR=$output_root/wandb_data WANDB_MODE=disabled
export TORCH_HOME=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/cache/torch
for path in "$source_release" "$output_root" "$launch" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TORCH_HOME"; do
 [[ "$(readlink -m "$path")" == "$workspace"/* ]] || exit 2
done
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
test -s "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/fit
mkdir -p "$output_root/main_job_checks"
cd "$source_release"
"$python" "$launch/verify_release.py" "$source_manifest" 8a6a791418c65e494a9c037ae49510272bcb426965eaf9102ea581da70fc7067 > "$output_root/main_job_checks/main_fit_source_before.json"
"$python" "$launch/qualification_gate.py" "$output_root" > "$output_root/main_job_checks/main_fit_qualification_gate.json"
# Exact corrected canonical stage: fit
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.stationary_replay_cli --source-root /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1 --replay /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/fit --seed 99 --updates 30000 --batch-size 128 --horizon 256 --learning-rate 0.0003 --gamma 0.99 --threads 4 --device cpu --analysis-plan /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1/hpc_runs/studies/stationary_projected_c05_s99_20261127.analysis.json
"$python" "$launch/verify_release.py" "$source_manifest" 8a6a791418c65e494a9c037ae49510272bcb426965eaf9102ea581da70fc7067 > "$output_root/main_job_checks/main_fit_source_after.json"
