"""Matched C05/C15 constant-hit reward controls at 100M frames."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = Path(__file__).with_name("studies") / "source_command_c05_c15_constant_hit_s3_20261012.study.json"
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
