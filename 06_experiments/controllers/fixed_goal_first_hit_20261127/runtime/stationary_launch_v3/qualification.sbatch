#!/bin/bash
# Native-only continuation: reuse successful test/fit stages of failed job8950995; no refit or new source control.
#SBATCH --job-name=fh_stationary_native_q3
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/slurm/qualification_native_r3_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/slurm/qualification_native_r3_%j.err
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
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_native_r3
test ! -e "$output_root/qualification_evaluation"
mkdir /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_native_r3
cd "$source_release"
"$python" "$launch/verify_release.py" "$source_manifest" 8a6a791418c65e494a9c037ae49510272bcb426965eaf9102ea581da70fc7067 > /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_native_r3/source_before.json
"$python" "$launch/completed_stage_guard.py" "$launch/launch_manifest.json" 2e91bdf82ab9976cdb4e38bf2f9d9970b1a398e06f10aa4a33af9ad945f798b9 > /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_native_r3/prerequisites.json
# Exact corrected canonical stage: qualification_evaluate
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.reused_control --source-root /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1 --run-dir /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99 --checkpoint /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99/checkpoint_p0/milestones/checkpoint_000006104_100007936.pth --checkpoint-sha256 f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38 --plan /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1/hpc_runs/studies/stationary_projected_c05_s99_20261127.analysis.json --fitted /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_fit --baseline /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r3/evaluation_nonempty_prefix --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_evaluation --runfiles /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r3/runfiles --decision-cap 4096
"$python" "$launch/verify_release.py" "$source_manifest" 8a6a791418c65e494a9c037ae49510272bcb426965eaf9102ea581da70fc7067 > /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_native_r3/source_after.json
"$python" "$launch/completed_stage_guard.py" "$launch/launch_manifest.json" 2e91bdf82ab9976cdb4e38bf2f9d9970b1a398e06f10aa4a33af9ad945f798b9 after > /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/qualification_native_r3/completion.json
