#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail

manifest=${1:?staged manifest TSV}
row_index=${2:?zero-based row}
panel=${3:?observation panel}
output_root=${4:?workspace output root}
source_kind=${5:?shared or cue}
per_cue=${6:-false}

workspace_root=/work/classic/fr_xl1014-corridor-geometry
legacy_input_root=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir
case "$source_kind" in shared|cue) ;; *) echo "Invalid source kind" >&2; exit 2 ;; esac
source_root=$workspace_root/IntrMotiv/source_poster_20260926/$source_kind
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
for path in "$run_dir" "$checkpoint"; do
    case "$path" in
        "$workspace_root"/*|"$legacy_input_root"/*) ;;
        *) echo "Training input outside recognized workspaces: $path" >&2; exit 2 ;;
    esac
done
for path in "$panel" "$output_root"; do
    case "$path" in
        "$workspace_root"/*) ;;
        *) echo "Panel or output outside the active workspace: $path" >&2; exit 2 ;;
    esac
done
cd "$source_root"
extra_args=()
if [[ "$per_cue" == true ]]; then
    extra_args+=(--per-cue)
fi
"$python" "$output_root/../scripts/collect_poster_population_kernels.py" \
    --run-dir "$run_dir" --checkpoint "$checkpoint" --panel "$panel" \
    --output "$output_root/results/$label.npz" "${extra_args[@]}"
