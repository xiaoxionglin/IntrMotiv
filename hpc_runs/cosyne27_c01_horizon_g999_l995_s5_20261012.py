"""Sample Factory adapter for cosyne27_c01_horizon_g999_l995_s5_20261012."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


STUDY = load_study(Path(__file__).with_name("studies") / "cosyne27_c01_horizon_g999_l995_s5_20261012.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
