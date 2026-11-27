#!/bin/bash
# Dependency IDs resolved from the actual four candidate evaluation submissions.
# Pure analysis of explicit imported-source/new-candidate cohorts. No actors or engines.
#SBATCH --job-name=fh_stationary_analysis
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/slurm/stationary_analysis_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/slurm/stationary_analysis_%j.err
#SBATCH --dependency=afterok:8951060:8951061:8951062:8951063
set -euo pipefail
: "${SLURM_JOB_ID:?Use the reviewed ordinary Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
source_release=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_stationary_20261127_r1
output_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1
launch=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_launch_v1
execution=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_execution_v1
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
export INTRMOTIV_WORKSPACE_ROOT="$workspace"
export INTRMOTIV_RUNTIME_SOURCE="$source_release" INTRMOTIV_RUNTIME_ROOT="$source_release"
export PYTHONPATH="$source_release:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export XDG_CACHE_HOME="$output_root/analysis_cache_v1"
export MPLCONFIGDIR="$output_root/analysis_matplotlib_v1"
export TORCH_HOME=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/cache/torch
export TMPDIR="$workspace/tmp/fh_stationary_analysis_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for path in "$source_release" "$output_root" "$launch" "$execution" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"; do
 [[ "$(readlink -m "$path")" == "$workspace"/* ]] || exit 2
done
test ! -e "$execution"
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/aggregate_v1
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/trace_analysis_v1
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/figures_v1
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_00/summary.json
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_01/summary.json
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_02/summary.json
test -s /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_03/summary.json
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
mkdir "$execution"
cd "$source_release"
printf '%s\n' '84c393fff8ff1883ee38873461c8a70cae62eb51046ca43072b4e3a66556f510  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_launch_v1/verify_release.py' | sha256sum --check --status
printf '%s\n' '292e90a402fba6c072bf040b884ace07d866a4ef0d9253519cf6d4c39488e7dc  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_launch_v1/field_reuse_audit.json' | sha256sum --check --status
printf '%s\n' '9e8ea512f63bfa1c585397d3fbace2b3043ccb6b8b30bb1ab218d7b8b03f1e79  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_launch_v1/commands.json' | sha256sum --check --status
printf '%s\n' 'a7848b1e8ed0cd526bd87c90d41f348af4008c7f1a8074a51801756df2dc3eb7  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cosyne27_checkpoint_20261011/common_panel/raw/COS27_C05_TEMPORAL_TEMPORAL_S99__00_COS27_C05_TEMPORAL_TEMPORAL_S99__COS27_C05_TEMPORAL_TEMPORAL__s99__f100007936/place_fields.npz' | sha256sum --check --status
"$python" "$launch/verify_release.py" "$launch/source_manifest.json" 8a6a791418c65e494a9c037ae49510272bcb426965eaf9102ea581da70fc7067 > "$execution/source_before.json"
# Canonical pure stage: aggregate
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.aggregate_evaluation --shards /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_00 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_01 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_02 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/evaluation_sources_03 --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/aggregate_v1
# Canonical pure stage: traces
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.trace_analysis --aggregate /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/aggregate_v1/aggregate.json --field-audit /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_launch_v1/field_reuse_audit.json --field-npz /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cosyne27_checkpoint_20261011/common_panel/raw/COS27_C05_TEMPORAL_TEMPORAL_S99__00_COS27_C05_TEMPORAL_TEMPORAL_S99__COS27_C05_TEMPORAL_TEMPORAL__s99__f100007936/place_fields.npz --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/trace_analysis_v1
# Canonical pure stage: figures
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.first_hit_control.plot_evaluation --aggregate /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/aggregate_v1/aggregate.json --traces /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/trace_analysis_v1/summary.json --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/figures_v1
"$python" "$launch/verify_release.py" "$launch/source_manifest.json" 8a6a791418c65e494a9c037ae49510272bcb426965eaf9102ea581da70fc7067 > "$execution/source_after.json"
printf '%s\n' '292e90a402fba6c072bf040b884ace07d866a4ef0d9253519cf6d4c39488e7dc  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/stationary_projected_r1/analysis_launch_v1/field_reuse_audit.json' 'a7848b1e8ed0cd526bd87c90d41f348af4008c7f1a8074a51801756df2dc3eb7  /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cosyne27_checkpoint_20261011/common_panel/raw/COS27_C05_TEMPORAL_TEMPORAL_S99__00_COS27_C05_TEMPORAL_TEMPORAL_S99__COS27_C05_TEMPORAL_TEMPORAL__s99__f100007936/place_fields.npz' | sha256sum --check --status
