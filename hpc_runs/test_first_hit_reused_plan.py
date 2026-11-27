"""Static reused-control contract checks; no fitting, environment or SSH."""

import copy
import json
from pathlib import Path

import pytest

from hpc_runs.first_hit_control.reused_plan import load_plan, render_commands, validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "hpc_runs/studies/stationary_projected_c05_s99_20261127.analysis.json"


def test_real_parent_study_and_complete_reused_query_budget():
    plan, check = load_plan(PLAN)
    assert check["donor_run_spec"]["name"] == "COS27_C05_TEMPORAL_TEMPORAL_S99"
    assert check["canonical_workflow_version"] == "1.15.2"
    assert check["budgets"]["new_collection_decisions"] == 0
    assert check["budgets"]["new_discovery_decisions"] == 0
    assert check["budgets"]["new_candidate_trials"] == 768
    assert check["budgets"]["new_noninitial_trials"] == 720
    assert check["budgets"]["worst_case_candidate_decisions_per_shard"] == [147456] * 4
    assert check["original_h6_gate_passed"] is False
    rendered = render_commands(plan, check)
    stages = {stage["stage_id"]: stage for stage in rendered["stages"]}
    assert rendered["print_only"] and not rendered["jobs_submitted"]
    assert len(stages) == 8
    assert sum(stage["native_environment"] for stage in stages.values()) == 5
    assert all("collector" not in stage["argv"][2] for stage in stages.values())
    for index in range(4):
        stage = stages[f"evaluation_sources_{index:02d}"]
        assert f"completed_main_shard_{index:02d}" in stage["depends_on"]
        assert "--baseline" in stage["argv"] and "--plan" in stage["argv"]
        assert "--discovery-episodes" not in stage["argv"]
    assert "reviewed_native_qualification" in stages["fit"]["depends_on"]
    assert "--analysis-plan" in stages["fit"]["argv"]


@pytest.mark.parametrize("section,key,value", [
    ("runtime", "output_root", "/home/fr/fr_xl1014/bulk"),
    ("fit", "updates", 30001),
    ("objective", "gamma", 1.0),
    ("model", "added_inference_parameters", 128),
    ("evaluation", "discovery_episodes", 96),
    ("evaluation", "policy_count", 2),
    ("evaluation", "source_shards", [[0, 1], [2, 3]]),
    ("replay", "new_collection_decisions", 1000),
    ("qualification", "updates", 200),
    ("parent_analysis", "effective_plan_sha256", "0" * 64),
])
def test_reject_changed_scientific_scope_and_unsafe_paths(section, key, value):
    plan = json.loads(PLAN.read_text())
    plan[section][key] = value
    with pytest.raises(ValueError):
        validate_plan(plan, ROOT)


def test_reject_reusing_active_output_or_mixing_qualification_replay():
    plan = json.loads(PLAN.read_text())
    changed = copy.deepcopy(plan)
    changed["runtime"]["output_root"] = plan["parent_analysis"]["output"]
    with pytest.raises(ValueError, match="preserve the original"):
        validate_plan(changed, ROOT)
    changed = copy.deepcopy(plan)
    changed["qualification"]["replay"] = plan["replay"]["directory"]
    with pytest.raises(ValueError, match="remain distinct"):
        validate_plan(changed, ROOT)


def test_source_override_changes_effective_identity_without_editing_plan():
    before = PLAN.read_bytes()
    plan, old = load_plan(PLAN)
    alternate = plan["runtime"]["source_release"] + "_review_only"
    revised, current = load_plan(PLAN, source_release=alternate)
    assert current["declared_plan_sha256"] == old["declared_plan_sha256"]
    assert current["effective_plan_sha256"] != old["effective_plan_sha256"]
    assert revised["runtime"]["source_release"] == alternate
    assert PLAN.read_bytes() == before


def test_tampered_evidence_fingerprint_is_rejected():
    plan = json.loads(PLAN.read_text())
    path = next(path for path in plan["local_evidence"] if "projected_grid_r1/results" in path)
    plan["local_evidence"][path] = "0" * 64
    with pytest.raises(ValueError, match="Pinned local artifact changed"):
        validate_plan(plan, ROOT)
