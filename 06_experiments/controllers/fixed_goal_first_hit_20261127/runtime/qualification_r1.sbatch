#!/bin/bash
# Print-only qualification payload. Ordinary independent job; no array.
#SBATCH --job-name=first_hit_c05_s99_q1
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r1/slurm/%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r1/slurm/%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Run only as the reviewed Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
source_release=$workspace/IntrMotiv/source_first_hit_control_20261127_r1
runtime_root=$workspace/IntrMotiv/SF_hipposlam/runtime
qualification_root=$workspace/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/qualification_r1
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
export TMPDIR=$workspace/tmp/fh_$SLURM_JOB_ID
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

# Canonical runfiles overlay keeps native assets and Lua provenance explicit.
"$python" -m hpc_runs.intrmotiv_study.runfiles --source "$source_release" --output "$qualification_root/runfiles"
"$python" -m hpc_runs.first_hit_control.collector \
    --source-root "$source_release" --run-dir "$donor_run" --checkpoint "$donor_checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha256" --config-sha256 "$config_sha256" \
    --runfiles "$qualification_root/runfiles" --output "$qualification_root/replay" \
    --decisions 5000 --seed 73001 --random-action-probability 0.2
"$python" -m hpc_runs.first_hit_control.replay_cli \
    --source-root "$source_release" --replay "$qualification_root/replay" --output "$qualification_root/fit" \
    --updates 100 --batch-size 64 --horizon 64 --seed 99 --threads 4 --device cpu

# Target identities are fixed from source-winner observations before command
# trials. This deliberately tiny panel qualifies plumbing, not efficacy.
"$python" - "$qualification_root/replay/replay.npz" > "$qualification_root/preflight_targets.txt" <<'PY'
import sys
import numpy as np
import torch
from hpc_runs.first_hit_control.detectors import detector_hits
with np.load(sys.argv[1]) as data:
    hits = detector_hits(torch.from_numpy(data['successor_dg'][data['successor_valid']]), 'winner')
observed = torch.nonzero(hits.any(0)).flatten().tolist()
if len(observed) < 2:
    raise RuntimeError('Fewer than two winner landmarks observed; qualification cannot form its panel')
for goal in observed[:2]:
    print(goal)
PY
readarray -t preflight_targets < "$qualification_root/preflight_targets.txt"
"$python" -m hpc_runs.first_hit_control.dmlab_evaluate \
    --source-root "$source_release" --run-dir "$donor_run" --checkpoint "$donor_checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha256" \
    --fitted "$qualification_root/fit" --runfiles "$qualification_root/runfiles" \
    --output "$qualification_root/evaluation" --phase preflight --targets "${preflight_targets[@]}" \
    --horizon 64 --discovery-episodes 4 --prefix-cap 256 --repeats 1 --max-sources 1 \
    --discovery-seed 31000 --decision-cap 10000
