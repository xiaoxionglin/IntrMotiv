"""Thin Sample Factory adapter; the canonical StudySpec owns all cells."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description

STUDY = load_study(Path(__file__).with_name("studies") / "ca3_action_transport_qualification_20261113.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
