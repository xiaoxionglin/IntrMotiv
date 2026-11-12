"""Check that coordinate recovery discovers, rather than assumes, kinematics."""

import numpy as np
import pytest

from inspect_ca3_action_transport import planar_action_maps, recover_coordinates, trace_word


def test_recovers_unknown_well_conditioned_linear_basis_without_labels():
    rng = np.random.default_rng(5)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    basis = np.diag([.8, 1.1, 1.4, 1.8]) @ orthogonal
    canonical = planar_action_maps()
    latent = {name: np.linalg.solve(basis, matrix @ basis) for name, matrix in canonical.items()}
    readout = np.eye(4)[:2] @ basis
    result = recover_coordinates(latent, readout)
    np.testing.assert_allclose(result["coordinate_map"], basis, atol=1e-12)
    assert max(result["canonical_frobenius_errors"].values()) < 1e-12


def test_turn_mismatch_is_not_hidden_by_forward_based_coordinate_choice():
    maps = planar_action_maps()
    maps["L"] = np.eye(4)  # a model that has not learned turning
    result = recover_coordinates(maps, np.eye(4)[:2])
    assert result["canonical_frobenius_errors"]["L"] > 1


def test_rejects_unidentified_coordinates_instead_of_forcing_a_pose_interpretation():
    maps = {name: np.eye(4) for name in "FLR"}
    with pytest.raises(ValueError, match="independent"):
        recover_coordinates(maps, np.eye(4)[:2])


def test_action_order_changes_endpoint_and_updates_have_rank_two():
    maps = planar_action_maps()
    initial = np.array([0., 0., 1., 0.])
    first = trace_word("FLF", maps, initial, np.eye(4))
    second = trace_word("LFF", maps, initial, np.eye(4))
    np.testing.assert_allclose(first["position"][-1], [1, 1])
    np.testing.assert_allclose(second["position"][-1], [0, 2])
    for matrix in maps.values():
        assert np.linalg.matrix_rank(matrix - np.eye(4)) == 2


def test_separate_rank_one_maps_can_be_order_sensitive():
    first = np.array([[1., 1.], [0., 1.]])
    second = np.array([[1., 0.], [1., 1.]])
    assert np.linalg.matrix_rank(first - np.eye(2)) == 1
    assert np.linalg.matrix_rank(second - np.eye(2)) == 1
    assert not np.allclose(first @ second, second @ first)
