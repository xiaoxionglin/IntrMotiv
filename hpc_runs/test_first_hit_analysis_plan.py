"""Static artifact-plan validation; no remote access, submission or environment."""

import hashlib
import json
from pathlib import Path
import shlex

import pytest

from hpc_runs.first_hit_control.analysis_plan import load_plan, render_commands, render_slurm, validate_plan, write_launch_bundle


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "hpc_runs/studies/fixed_goal_first_hit_c05_s99_20261127.analysis.json"


def test_canonical_donor_and_full_budget_are_bound():
    plan, result = load_plan(PLAN)
    assert result["donor_run_spec"]["name"] == "COS27_C05_TEMPORAL_TEMPORAL_S99"
    assert result["donor_study_sha256"] == plan["donor_study"]["sha256"]
    budget = result["budgets"]
    assert budget["collection_decisions"] == 100000
    assert budget["maximum_goal_labels"] == 61440000
    assert budget["target_refreshes"] == 300
    assert budget["worst_case_decisions_per_shard"] == [344064] * 4
    assert budget["total_evaluation_cap"] == 1400000
    assert budget["noninitial_own_trials_per_policy"] == 720


def test_renderer_has_real_dependencies_and_identical_shared_artifacts():
    plan, validation = load_plan(PLAN)
    rendered = render_commands(plan, validation)
    assert rendered["print_only"] is True
    stages = {stage["stage_id"]: stage for stage in rendered["stages"]}
    assert len(stages) == 9
    assert stages["support"]["depends_on"] == ["collect"]
    assert stages["fit"]["depends_on"] == ["collect", "support"]
    assert stages["fit_diagnostics"]["depends_on"] == ["fit"]
    diagnostic = stages["fit_diagnostics"]["argv"]
    assert diagnostic[diagnostic.index("--samples-per-bin") + 1] == "1024"
    evaluated_sources = []
    for stage in rendered["stages"]:
        argv = stage["argv"]
        assert argv[:2] == [plan["runtime"]["python"], "-m"]
        assert "sbatch" not in argv
        if stage["stage_id"].startswith("evaluate_"):
            assert argv[argv.index("--fitted") + 1] == stages["fit"]["output"]
            assert argv[argv.index("--runfiles") + 1] == stages["runfiles"]["output"]
            start = argv.index("--sources") + 1
            evaluated_sources += list(map(int, argv[start:argv.index("--horizon")]))
            assert argv[argv.index("--horizon") + 1] == str(plan["fit"]["horizon"])
    assert sorted(evaluated_sources) == list(range(16))


@pytest.mark.parametrize("change,match", [
    (lambda p: p["donor_study"].update(sha256="0" * 64), "fingerprint"),
    (lambda p: p["donor"].update(seed=777), "exactly one"),
    (lambda p: p["donor"].update(config_sha256="0" * 64), "differs from qualification"),
    (lambda p: p["runtime"].update(output_root="/home/fr/fr_xl1014/bulk"), "allocated workspace"),
    (lambda p: p["evaluation"]["source_shards"][0].__setitem__(0, 1), "partition every"),
    (lambda p: p["evaluation"].update(decision_cap_per_shard=300000), "all declared matched blocks"),
    (lambda p: p["fit"].update(updates=5000), "target generations"),
    (lambda p: p["fit"].update(target_update_period=50), "does not expose"),
    (lambda p: p["collection"].update(reuse_qualification_data=True), "silently pooled"),
])
def test_invalid_identity_storage_partition_or_budget_fails(change, match):
    plan, _ = load_plan(PLAN)
    change(plan)
    with pytest.raises(ValueError, match=match):
        validate_plan(plan, ROOT)


def test_release_override_is_explicit_and_changes_effective_fingerprint_only():
    original, initial = load_plan(PLAN)
    release = original["runtime"]["source_release"] + "_audited"
    changed, audited = load_plan(PLAN, source_release=release)
    assert changed["runtime"]["source_release"] == release
    assert initial["declared_plan_sha256"] == audited["declared_plan_sha256"]
    assert initial["effective_plan_sha256"] != audited["effective_plan_sha256"]
    assert original == load_plan(PLAN)[0]
    with pytest.raises(ValueError, match="allocated workspace"):
        load_plan(PLAN, source_release="/home/fr/source")


def release_record(plan):
    return dict(candidate_commit="a" * 40, archive_sha256="b" * 64,
                baseline_tree_sha256="c" * 64, source_release=plan["runtime"]["source_release"])


def test_slurm_jobs_keep_exact_argv_resources_and_runtime_guards():
    plan, validation = load_plan(PLAN)
    bundle = render_slurm(plan, validation, release_record(plan))
    original = {row["stage_id"]: row["argv"] for row in render_commands(plan, validation)["stages"]}
    assert bundle["jobs_submitted"] is False
    assert len(bundle["jobs"]) == 5
    assert len(bundle["jobs"][0]["stages"]) == 5
    for index, job in enumerate(bundle["jobs"]):
        assert job["resources"] == dict(partition="cpu", cpus=4, memory_gb=16 if index == 0 else 8, hours=12)
        assert job["depends_on_jobs"] == ([] if index == 0 else ["collect_fit"])
        assert "SLURM_JOB_ID:?" in job["script"]
        assert '"$(readlink -m "$output_path")"' in job["script"]
        assert "FIRST_HIT_RELEASE.json" in job["script"]
        assert "FIRST_HIT_BASELINE.json" in job["script"]
        assert "#SBATCH --array" not in job["script"]
        lines = job["script"].splitlines()
        for stage in job["stages"]:
            line = lines[lines.index("# stage: " + stage["stage_id"]) + 1]
            assert shlex.split(line) == original[stage["stage_id"]] == stage["argv"]


def test_launch_bundle_records_real_file_hashes_and_refuses_overwrite(tmp_path):
    plan, validation = load_plan(PLAN)
    output = tmp_path / "launch"
    manifest = write_launch_bundle(plan, validation, release_record(plan), output)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name, expected in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        write_launch_bundle(plan, validation, release_record(plan), output)
    with pytest.raises(ValueError, match="staged runtime|Staged release"):
        record = release_record(plan)
        record["source_release"] += "_other"
        render_slurm(plan, validation, record)
    with pytest.raises(ValueError, match="twelve"):
        render_slurm(plan, validation, release_record(plan), evaluation_hours=13)


def test_guard_uses_real_release_and_baseline_manifest_contract(tmp_path, monkeypatch):
    plan, validation = load_plan(PLAN)
    record = release_record(plan)
    script = render_slurm(plan, validation, record)["jobs"][0]["script"]
    guard = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    release = tmp_path / "FIRST_HIT_RELEASE.json"
    baseline = tmp_path / "FIRST_HIT_BASELINE.json"
    # The remote release manifest has no source_release key. Its baseline
    # manifest binds the absolute staged directory under the key 'release'.
    release.write_text(json.dumps({key: value for key, value in record.items() if key != "source_release"}))
    baseline.write_text(json.dumps({"release": record["source_release"]}))
    monkeypatch.setattr("sys.argv", ["-", str(release), str(baseline)])
    exec(compile(guard, "<rendered-runtime-identity-guard>", "exec"), {})
    baseline.write_text(json.dumps({"release": "/work/different_source"}))
    with pytest.raises(SystemExit, match="source directory"):
        exec(compile(guard, "<rendered-runtime-identity-guard>", "exec"), {})
