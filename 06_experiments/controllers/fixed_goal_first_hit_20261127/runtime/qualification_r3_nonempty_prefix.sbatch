#!/bin/bash
# Print-only qualification payload. Ordinary independent job; no array.
#SBATCH --job-name=first_hit_nonempty_prefix
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r3/slurm/%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r3/slurm/%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Run only as the reviewed Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
source_release=$workspace/IntrMotiv/source_first_hit_control_20261127_r3
runtime_root=$workspace/IntrMotiv/SF_hipposlam/runtime
qualification_root=$workspace/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r3
donor_run=$workspace/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99
donor_checkpoint=$donor_run/checkpoint_p0/milestones/checkpoint_000006104_100007936.pth
checkpoint_sha256=f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38
config_sha256=a18da40ae1fbbd428e8c61c8cefcb79827fd66fa3d1f8646ba7201369df563fd
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python

# Reassert after the node prolog; short Unix socket paths remain in workspace.
export INTRMOTIV_WORKSPACE_ROOT=$workspace
export INTRMOTIV_RUNTIME_SOURCE=$source_release
export PYTHONPATH=$source_release:$runtime_root/torch_cuda_2_9_1:$runtime_root/controller_terminal_binding_v1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=$workspace/tmp/fhp_$SLURM_JOB_ID
export TMP=$TMPDIR TEMP=$TMPDIR
export XDG_CACHE_HOME=$qualification_root/cache
export MPLCONFIGDIR=$qualification_root/matplotlib
export DMLAB_CACHE_DIR=$qualification_root/dmlab_cache
export WANDB_DIR=$qualification_root/wandb
export WANDB_CACHE_DIR=$qualification_root/wandb_cache
export WANDB_DATA_DIR=$qualification_root/wandb_data
export WANDB_MODE=disabled
export TORCH_HOME=$runtime_root/cache/torch
for output_path in "$qualification_root" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TORCH_HOME"; do
    [[ "$(readlink -m "$output_path")" == "$workspace"/* ]] || exit 2
done
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
test -s "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
cd "$source_release"

# Reuse the completed r3 fit and the same frozen runtime/assets. This is an
# evaluator-only probe; source2 differs from the observed initial source4.
test -f "$qualification_root/runfiles/provenance.json"
"$python" -m hpc_runs.first_hit_control.dmlab_evaluate \
    --source-root "$source_release" --run-dir "$donor_run" --checkpoint "$donor_checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha256" --fitted "$qualification_root/fit" \
    --runfiles "$qualification_root/runfiles" --output "$qualification_root/evaluation_nonempty_prefix" \
    --phase preflight --targets 0 1 --sources 2 --max-sources 1 --horizon 64 \
    --discovery-episodes 8 --prefix-cap 256 --repeats 1 --decision-cap 10000 --discovery-seed 31000
