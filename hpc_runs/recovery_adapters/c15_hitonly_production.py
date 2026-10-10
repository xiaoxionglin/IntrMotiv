"""Production run for the C15 hit-only recovery."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).resolve().parents[1] / "studies/corrected_core_c15_hitonly_recovery_production_20261010.study.json"
RUN_DESCRIPTION = build_run_description(load_study(SPEC))
