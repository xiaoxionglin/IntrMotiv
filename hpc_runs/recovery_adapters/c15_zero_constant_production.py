"""Production run for the C15 zero-bonus constant-credit recovery."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

SPEC = Path(__file__).resolve().parents[1] / "studies/corrected_core_c15_zero_constant_recovery_production_20261010.study.json"
RUN_DESCRIPTION = build_run_description(load_study(SPEC))
