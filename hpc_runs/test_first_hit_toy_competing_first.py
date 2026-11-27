"""Independent saved-path oracle checks, without training or environment calls."""

import pytest

from hpc_runs.first_hit_control.toy_competing_first import classified_first, trace_first


def test_goalblind_stopped_paths_have_zero_first_but_positive_naive_arrival():
    command_g = dict(states=[0, 1, 2], actions=[0, 0], decisions=2)
    command_c = dict(states=[0, 1], actions=[0], decisions=1)
    assert int(2 in command_g["states"]) - int(2 in command_c["states"]) == 1
    assert classified_first(command_g, 2, 1, 3) - classified_first(command_c, 2, 1, 3) == 0


def test_direct_command_selection_has_positive_first_and_third_cues_are_allowed():
    command_g = dict(states=[0, 3, 2], actions=[0, 0], decisions=2)
    command_c = dict(states=[0, 1], actions=[1], decisions=1)
    assert classified_first(command_g, 2, 1, 3) - classified_first(command_c, 2, 1, 3) == 1
    assert trace_first([0, 3, 3, 3], 2, 1) == 0
    with pytest.raises(ValueError, match="neither initially"):
        trace_first([0, 1], 0, 1)
