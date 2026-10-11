"""Sample Factory adapter for five fresh C01 no-credit controls."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


STUDY = load_study(Path(__file__).with_name("studies") / "cosyne27_c01_no_credit_s5_20261012.study.json")
RUN_DESCRIPTION = build_run_description(STUDY)
