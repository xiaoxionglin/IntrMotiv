#!/bin/bash
# Generated print-only payload; submit only after reviewed qualification passes.
# Ordinary job; evaluation requires afterok on the actual collect_fit job ID.
#SBATCH --job-name=fh_main_evaluate_01
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/evaluate_01_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/evaluate_01_%j.err

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

# Artifact gates supplement the required afterok scheduler dependency.
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay/metadata.json
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/exposure/exposure.json
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit/metadata.json
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit/adapter.pt
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit_diagnostics/residuals.json

# Keep every stage fresh; partial runs require a separately reviewed recovery.
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/evaluation_sources_01

# stage: evaluate_01
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.dmlab_evaluate --source-root /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3 --run-dir /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99 --checkpoint /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/cosyne27_c05_pushpull_s3_20261010/COS27_C05_TEMPORAL_TEMPORAL_S99_/00_COS27_C05_TEMPORAL_TEMPORAL_S99/checkpoint_p0/milestones/checkpoint_000006104_100007936.pth --checkpoint-sha256 f35464c7e058df3ac51b676d5560470aa3cddd3c57f02ed1c0012a5fbcfaed38 --runfiles /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/runfiles --fitted /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/evaluation_sources_01 --phase full --sources 4 5 6 7 --horizon 256 --discovery-episodes 96 --prefix-cap 512 --repeats 3 --max-sources 4 --discovery-seed 41000 --decision-cap 350000
