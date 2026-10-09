"""Held second release of the corrected-core distance-reward ablation."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


STUDY = load_study(Path(__file__).with_name("studies") / "corrected_core_encoder_worker_later_20261009.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
