"""Import the actual canonical execution entry points without making an env.

AST callback checks alone miss dependencies of imported evaluator helpers. This
regression uses the staged source path, so an incomplete source overlay fails
before a Slurm/native qualification is submitted.
"""

import importlib
import os
from pathlib import Path

import pytest


SOURCE = Path(os.environ.get("INTRMOTIV_RUNTIME_SOURCE", "/home/xiaoxiong/SFgit/SF_hipposlam")).resolve()


@pytest.mark.parametrize("name,entrypoint", [
    ("sf_shared.checkpoints", "load_checkpoint_dict"),
    ("sf_working_directories.IntrMotiv.evaluation.place_fields", "load_policy_env"),
    ("sf_working_directories.IntrMotiv.evaluation.target_control_interventions", "run_landmark_matched_interventions"),
])
def test_canonical_execution_imports_resolve_inside_declared_release(name, entrypoint):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(SOURCE)
    assert callable(getattr(module, entrypoint))
