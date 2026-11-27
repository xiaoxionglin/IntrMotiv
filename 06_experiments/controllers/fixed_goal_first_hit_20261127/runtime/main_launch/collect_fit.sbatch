#!/bin/bash
# Generated print-only payload; submit only after reviewed qualification passes.
# Ordinary job; evaluation requires afterok on the actual collect_fit job ID.
#SBATCH --job-name=fh_main_collect_fit
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/collect_fit_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/collect_fit_%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Run only as the reviewed NEMO Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
source_release=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3
main_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python

export INTRMOTIV_WORKSPACE_ROOT=/work/classic/fr_xl1014-corridor-geometry
export INTRMOTIV_RUNTIME_SOURCE=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3
export PYTHONPATH=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1
export TORCH_HOME=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/cache/torch
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export XDG_CACHE_HOME=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/matplotlib
export DMLAB_CACHE_DIR=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/dmlab_cache
export WANDB_DIR=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/wandb
export WANDB_CACHE_DIR=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/wandb_cache
export WANDB_DATA_DIR=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/wandb_data
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$workspace/tmp/fh_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
for output_path in "$main_root" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TORCH_HOME"; do
    [[ "$(readlink -m "$output_path")" == "$workspace"/* ]] || exit 2
done
[[ "$(readlink -m "$source_release")" == "$workspace"/* ]] || exit 2
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DMLAB_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
test -s "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
test -x "$python"
cd "$source_release"

# Verify the immutable runtime receipt and exact donor before native imports.
"$python" - "$source_release/FIRST_HIT_RELEASE.json" "$source_release/FIRST_HIT_BASELINE.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
expected = {'candidate_commit': '10cd0e574cb3943e244d8f5847aef7aecff87dfa', 'archive_sha256': '9d54907ddca6e686488e755775877ea4b0bbfa455fe304e52d6aff42dfcc1823', 'baseline_tree_sha256': '0b6010c806ba2fb98c482240e61e50c6566f745756f6fab2c9df13d5d6e64ba8'}
if any(record.get(key) != value for key, value in expected.items()):
    raise SystemExit('Runtime release differs from reviewed launch record')
baseline = json.load(open(sys.argv[2]))
if baseline.get('release') != '/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3':
    raise SystemExit('Runtime source directory differs from reviewed baseline')
PY
printf '%s\n' 'f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99/checkpoint_p0/milestones/checkpoint_000006104_100007936.pth' | sha256sum --check --status
printf '%s\n' 'a18da40ae1fbbd428e8c61c8cefcb79827fd66fa3d1f8646ba7201369df563fd  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99/config.json' | sha256sum --check --status

# Keep every stage fresh; partial runs require a separately reviewed recovery.
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/runfiles
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/exposure
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit_diagnostics

# stage: runfiles
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.intrmotiv_study.runfiles --source /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3 --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/runfiles

# stage: collect
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.collector --source-root /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3 --run-dir /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99 --checkpoint /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99/checkpoint_p0/milestones/checkpoint_000006104_100007936.pth --checkpoint-sha256 f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38 --runfiles /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/runfiles --config-sha256 a18da40ae1fbbd428e8c61c8cefcb79827fd66fa3d1f8646ba7201369df563fd --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay --decisions 100000 --seed 73002 --random-action-probability 0.2

# stage: support
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.support_analysis --replay /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/exposure --split-seed 99 --holdout-fraction 0.1

# stage: fit
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.replay_cli --source-root /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3 --replay /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit --updates 30000 --batch-size 128 --horizon 256 --learning-rate 0.0003 --seed 99 --threads 4 --device cpu

# stage: fit_diagnostics
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.fit_diagnostics --replay /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay --fitted /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit --source-root /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3 --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit_diagnostics --samples-per-bin 1024 --threads 4
