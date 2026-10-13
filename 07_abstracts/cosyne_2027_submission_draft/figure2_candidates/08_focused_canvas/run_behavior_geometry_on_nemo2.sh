#!/usr/bin/env bash
#SBATCH --job-name=cos27-behavior-geometry
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cosyne27_behavior_expression_20261011/figure2_geometry/geometry-%j.out

set -euo pipefail

analysis_root=/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/SF_hipposlam/train_dir/analysis/cosyne27_behavior_expression_20261011
geometry_root="$analysis_root/figure2_geometry"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$geometry_root/mplconfig"
export PYTHONPATH="$geometry_root/source:$analysis_root/source"

/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  "$geometry_root/source/collect_behavior_geometry.py" \
  --provenance "$analysis_root/fresh_probe_provenance.csv" \
  --raw-root "$analysis_root/probes/raw" \
  --output "$geometry_root/results"
