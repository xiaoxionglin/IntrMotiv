"""Sample Factory adapter for the CPU2048 push–pull reference calibration."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


STUDY = load_study(Path(__file__).with_name("studies") / "push_pull_cpu2048_calibration_20261007.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
