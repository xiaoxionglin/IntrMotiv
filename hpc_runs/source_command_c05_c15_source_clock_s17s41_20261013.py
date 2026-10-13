"""Canonical StudySpec adapter for the five-seed source-clock comparison."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).with_name("studies") / "source_command_c05_c15_source_clock_s17s41_20261013.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
