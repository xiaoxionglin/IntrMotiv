#!/bin/bash
# Reviewed pure-tensor postfit diagnostic; no training or environment execution.
#SBATCH --job-name=fh_affine_audit
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_affine_audit_20261127_r1
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/affine_deadline_audit_%j.out
#SBATCH --error=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/slurm/affine_deadline_audit_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Use the reviewed ordinary Slurm compute job}"
workspace=/work/classic/fr_xl1014-corridor-geometry
audit_source=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_affine_audit_20261127_r1
model_source=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_first_hit_control_20261127_r3
launch=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_launch_v1
python=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
export PYTHONPATH="$audit_source:/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_cache_v1
export MPLCONFIGDIR=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_matplotlib_v1
export TORCH_HOME=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/runtime/cache/torch
export TMPDIR="$workspace/tmp/fh_affine_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for path in "$audit_source" "$model_source" "$launch" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME" /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_v1 /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_tests_v1; do
 [[ "$(readlink -m "$path")" == "$workspace"/* ]] || exit 2
done
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$audit_source"
"$python" - "$launch/release_manifest.json" 0917b9e246c2c2694cebf74c9ab3f9bbeb878976e4aee2019f1fdee253a0adb8 <<'VERIFY'
import hashlib,json,pathlib,sys
manifest=pathlib.Path(sys.argv[1]);assert hashlib.sha256(manifest.read_bytes()).hexdigest()==sys.argv[2]
m=json.loads(manifest.read_text());root=pathlib.Path(m['source_release']);overlay=m['overlay_files_sha256']
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in overlay}
assert hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['base_tree_sha256']
for relative,sha in overlay.items():assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==sha
print(json.dumps({'all_base_files_unchanged':True,'base_files':len(files),'overlay_files':len(overlay)}))
VERIFY
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_v1
test ! -e /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_tests_v1
mkdir -p /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_tests_v1
"$python" -m pytest -q -p no:cacheprovider --basetemp /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_tests_v1/tmp hpc_runs/test_first_hit_affine_deadline_audit.py | tee /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_tests_v1/pytest.txt
"$python" -m hpc_runs.first_hit_control.affine_deadline_audit --replay /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/replay --fitted /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/fit --source-root "$model_source" --protocol "$audit_source/06_experiments/controllers/fixed_goal_first_hit_20261127/runtime/affine_deadline_protocol.json" --output /work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/first_hit_control_20261127/main_r1/affine_audit_v1
"$python" - "$launch/release_manifest.json" 0917b9e246c2c2694cebf74c9ab3f9bbeb878976e4aee2019f1fdee253a0adb8 <<'VERIFY'
import hashlib,json,pathlib,sys
manifest=pathlib.Path(sys.argv[1]);assert hashlib.sha256(manifest.read_bytes()).hexdigest()==sys.argv[2]
m=json.loads(manifest.read_text());root=pathlib.Path(m['source_release']);overlay=m['overlay_files_sha256']
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in overlay}
assert hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['base_tree_sha256']
for relative,sha in overlay.items():assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==sha
print(json.dumps({'all_base_files_unchanged':True,'base_files':len(files),'overlay_files':len(overlay)}))
VERIFY
