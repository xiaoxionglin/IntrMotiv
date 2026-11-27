#!/bin/bash
# Print-only reviewed payload; ordinary CPU job, no environment execution.
#SBATCH --job-name=fh_analysis_main
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_analysis_20261127_r1
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/fh_analysis_main_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/fh_analysis_main_%j.err
#SBATCH --dependency=afterok:8950234:8950235:8950236:8950237
set -euo pipefail
: "${SLURM_JOB_ID:?Use the reviewed ordinary Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
source_release=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_analysis_20261127_r1
main_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1
launch=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/analysis_launch_v1
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
export PYTHONPATH="$source_release:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$main_root/analysis_cache_v1"
export MPLCONFIGDIR="$main_root/analysis_matplotlib_v1"
export TMPDIR="$workspace/tmp/fh_analysis_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for path in "$source_release" "$launch" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"; do
 [[ "$(readlink -m "$path")" == "$workspace"/* ]] || exit 2
done
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$source_release"
printf '%s\n' '1dd89f949a72890a15e70a788248acf3c4e5647b94df413a5d0255d48332677d  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/analysis_launch_v1/verify_release.py' | sha256sum --check --status
"$python" "$launch/verify_release.py" "$launch/release_manifest.json" 31242fb5f293b96cf3e29ae4d9efa68cb45075722c59cb0c6ab28e20b0e91e72
# Submission must also add afterok on the successful qualification job ID.
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/analysis_qualification_v1/probe/qualification.json
"$python" - /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/analysis_qualification_v1/pytest.txt /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/analysis_qualification_v1/probe/qualification.json <<'CHECK'
import json,pathlib,sys
assert '36 passed' in pathlib.Path(sys.argv[1]).read_text()
r=json.loads(pathlib.Path(sys.argv[2]).read_text());assert r['native_execution'] is False
assert sum(v['issued_own_paths'] for v in r['panels'].values())==8
CHECK
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/aggregate_v1
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/trace_analysis_v1
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/figures_v1
"$python" -m hpc_runs.first_hit_control.aggregate_evaluation --shards /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/evaluation_sources_00 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/evaluation_sources_01 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/evaluation_sources_02 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/evaluation_sources_03 --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/aggregate_v1
"$python" -m hpc_runs.first_hit_control.trace_analysis --aggregate /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/aggregate_v1/aggregate.json --field-audit "$source_release/06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/field_reuse_audit.json" --field-npz /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cosyne27_checkpoint_20261011/common_panel/raw/COS27_C05_TEMPORAL_TEMPORAL_S99__00_COS27_C05_TEMPORAL_TEMPORAL_S99__COS27_C05_TEMPORAL_TEMPORAL__s99__f100007936/place_fields.npz --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/trace_analysis_v1
"$python" -m hpc_runs.first_hit_control.plot_evaluation --aggregate /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/aggregate_v1/aggregate.json --traces /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/trace_analysis_v1/summary.json --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/figures_v1
"$python" "$launch/verify_release.py" "$launch/release_manifest.json" 31242fb5f293b96cf3e29ae4d9efa68cb45075722c59cb0c6ab28e20b0e91e72
