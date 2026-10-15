"""Sample Factory adapter for the C01 flat-policy temporal calibration."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


STUDY = load_study(Path(__file__).with_name("studies") / "cosyne27_c01_flat_reward_qtemp_20261015.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
