#!/bin/bash
#SBATCH --job-name=intrmotiv-exact-restore
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller-r4-exact-restore-%j.out
#SBATCH --error=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller-r4-exact-restore-%j.err
set -euo pipefail
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_publication_20260912
export PYTHONPATH="$PWD:/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/torch_cuda_2_9_1:/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/matplotlib
export TMPDIR=/work/classic/fr_xl1014-train/tmp/restore_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
for index in 0 1 2 3 4 5; do
 /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/controller_r4_exact_restore.py "$index"
done
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python - <<'PYEND'
from pathlib import Path
import json
base=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/restart_baselines/intrmotiv_full_system_controller_preflight_20260912_r4')
rows=json.loads((base/'baselines.json').read_text())
result=dict(schema='intrmotiv/checkpoint-reload/v1',runs=[json.loads((base/(r['run']+'.reload.json')).read_text()) for r in rows])
assert len(result['runs'])==6 and all(r['exact_restore'] for r in result['runs'])
(base/'reload_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
PYEND
