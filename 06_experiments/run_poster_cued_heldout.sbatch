#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

manifest=${1:?manifest TSV}
row_index=${2:?zero-based manifest row}
trials=${3:?matched reset count}
horizon=${4:?decision horizon}
output_root=${5:?workspace output root}

workspace_root=/work/classic/fr_xl1014-corridor-geometry
source_root=$workspace_root/IntrMotiv/source_poster_20260926/cue
runfiles=/work/classic/fr_xl1014-fixed-reward-transfer/IntrMotiv/SF_hipposlam/runtime/fixed_reward_transfer_runfiles
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
export INTRMOTIV_WORKSPACE_ROOT=$workspace_root
export TMPDIR=$output_root/tmp/$SLURM_JOB_ID
export DMLAB_CACHE_DIR=$output_root/dmlab_cache/$SLURM_JOB_ID
export XDG_CACHE_HOME=$output_root/cache/$SLURM_JOB_ID
export TORCH_HOME=$workspace_root/IntrMotiv/SF_hipposlam/runtime/cache/torch
export WANDB_MODE=disabled
export PYTHONPATH=$source_root${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$output_root/results"

mapfile -t fields < <("$python" - "$manifest" "$row_index" <<'PY'
import csv
import sys
with open(sys.argv[1], newline='') as stream:
    rows = list(csv.DictReader(stream, delimiter='\t'))
row = rows[int(sys.argv[2])]
for key in ('run_dir', 'checkpoint', 'label_suffix'):
    print(row[key])
PY
)
run_dir=${fields[0]}
checkpoint=${fields[1]}
label=${fields[2]}
for path in "$run_dir" "$checkpoint" "$output_root"; do
    case "$path" in
        "$workspace_root"/*) ;;
        *) echo "Input or output outside the active workspace: $path" >&2; exit 2 ;;
    esac
done
cd "$source_root"
"$python" sf_working_directories/IntrMotiv/evaluation/fixed_reward_heldout.py \
    --run-dir "$run_dir" --checkpoint "$checkpoint" --runfiles "$runfiles" \
    --site cued5 --out "$output_root/results/$label.csv" \
    --trials "$trials" --horizon "$horizon" --seed-base 61000 --stochastic
