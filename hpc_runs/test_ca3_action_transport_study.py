"""Scientific matrix contracts for the authorized 80 production cells."""

from collections import Counter
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.analysis import linear_contrasts


SPEC = Path(__file__).with_name("studies") / "ca3_action_transport_80_20261113.study.json"


def test_exactly_eight_conditions_and_ten_paired_seeds():
    study = load_study(SPEC)
    runs = study.expand_runs()
    assert study.expected_runs == len(runs) == len({r.name for r in runs}) == 80
    assert len(study.seeds) == 10
    assert set(Counter(r.condition for r in runs).values()) == {10}
    assert len({r.condition for r in runs}) == 8
    for condition in {r.condition for r in runs}:
        assert {r.seed for r in runs if r.condition == condition} == set(study.seeds)


def test_only_declared_memory_treatment_differs_within_seed():
    study = load_study(SPEC)
    variable_flags = {
        "--ca3_action_memory", "--ca3_action_memory_rank", "--ca3_action_memory_hidden_size",
        "--study_condition", "--wandb_tags",
    }
    for seed in study.seeds:
        settings = [
            {k: v for k, v in (arg.split("=", 1) for arg in run.args) if k not in variable_flags}
            for run in study.expand_runs() if run.seed == seed
        ]
        assert all(s == settings[0] for s in settings)
        common = settings[0]
        assert common["--transfer_freeze_dg"] == "True"
        assert common["--transfer_calibrate_frozen_dg"] == "False"
        assert common["--dg_orthogonal_recruitment"] == "False"
        assert common["--ppo_dg_gradient"] == "stop"
        assert float(common["--encoder_grad_coeff"]) == 0
        assert common["--encoder_conv_architecture"] == "layer2_resnet18"
        assert common["--hrl_goal_conditioning"] == "target_id_film"
        assert common["--hrl_action_path_integration"] == "False"
        assert common["--fixed_task_conditioning"] == "False"


def test_contrasts_select_one_cell_per_paired_seed():
    study = load_study(SPEC)
    records = [dict(base=r.base, seed=r.seed, **r.factors, dummy=0.0) for r in study.expand_runs()]
    detail, summary = linear_contrasts(
        records, ["dummy"], ["base"], ["seed"], study.analysis["contrasts"]
    )
    assert len(detail) == len(study.analysis["contrasts"]) * 10
    assert all(row["n"] == 10 for row in summary)


def test_telemetry_retains_all_terminal_replicates_and_five_developmental_ages():
    study = load_study(SPEC)
    assert len(study.telemetry["target_frames"]) == 5
    assert study.telemetry["trajectory_seed"] == 99
    assert set(study.telemetry["terminal_seeds"]) == set(study.seeds)
    assert study.telemetry["online_spatial_target_frames"] == study.telemetry["target_frames"]
