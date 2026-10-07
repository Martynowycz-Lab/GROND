"""Cropped DIALS crystal models must retain the geometry of their original frames."""

import numpy as np
import pytest

from grond.io.dials_geometry import _a_scan_points_for_current_scan, _rotation_matrix, dials_sg_statistics


def metadata(raw, start, stop):
    return {"A_scan_points": raw, "scan": {"array_range": [start, stop], "image_range": [start + 1, stop]}}


def test_interior_origin_below_gross_error_threshold():
    # The old three-candidate heuristic accepted offset20 because its excitation
    # error passed0.005 A^-1. Only the interior offset15 matches the source model.
    raw = np.repeat(np.eye(3)[None, :, :], 134, axis=0)
    raw[:, 0, 0] = np.arange(134)
    tested = []

    def score(slice):
        offset = int(slice[0, 0, 0])
        tested.append(offset)
        return abs(offset - 15) * 1e-4 + 1e-7

    selected, contract = _a_scan_points_for_current_scan(metadata(raw, 20, 123), scorer=score)
    assert tested == list(range(31))
    assert contract["source_scan_point_origin"] == 15
    np.testing.assert_array_equal(selected, raw[15:119])


def test_predicted_bragg_geometry_recovers_interior_origin():
    # Integer (1,0,0) reflections lie on the Ewald sphere at each known frame.
    # Applying the wrong A-model origin rotates them away from that sphere.
    base = np.diag([0.1, 0.1, 0.1])
    base[2, 0] = 40 - np.sqrt(40**2 - 0.1**2)
    raw = np.array([_rotation_matrix(np.array([0, 1, 0]), -(j - 2) * 0.1) @ base for j in range(8)])
    meta = metadata(raw, 7, 11)
    meta.update(
        {
            "beam": {"s0": [0, 0, -40]},
            "goniometer": {"rotation_axis": [0, 1, 0]},
            "scan": {**meta["scan"], "oscillation": [0, 0.1]},
        }
    )
    z = np.arange(7, 12)
    arrays = {
        "hkl": np.tile([1, 0, 0], (len(z), 1)),
        "xyzcal": np.column_stack([np.zeros(len(z)), np.zeros(len(z)), z]),
    }
    result = dials_sg_statistics(arrays, meta, np.ones(len(z), dtype=bool))
    assert result["status"] == "ok"
    assert result["a_scan_contract"]["source_scan_point_origin"] == 2
    assert result["p95_abs_sg_inv_A"] < 1e-12


def test_distinct_equal_scores_require_explicit_contract():
    raw = np.repeat(np.eye(3)[None, :, :], 7, axis=0)
    raw[:, 0, 0] += np.arange(7) * 0.01
    with pytest.raises(ValueError, match="Ambiguous"):
        _a_scan_points_for_current_scan(metadata(raw, 0, 3), scorer=lambda a: 1e-7)
    selected, c = _a_scan_points_for_current_scan(metadata(raw, 0, 3), source_index_offset=2)
    assert c["source_scan_point_origin"] == 2
    np.testing.assert_array_equal(selected, raw[2:6])


def test_equivalent_static_slices_are_not_physically_ambiguous():
    raw = np.repeat(np.eye(3)[None, :, :], 7, axis=0)
    selected, c = _a_scan_points_for_current_scan(metadata(raw, 4, 7), scorer=lambda a: 1e-7)
    assert selected.shape == (4, 3, 3)
    np.testing.assert_array_equal(selected, raw[:4])


@pytest.mark.parametrize("count", [1, 5])
def test_unambiguous_models_need_no_scorer(count):
    raw = np.repeat(np.eye(3)[None, :, :], count, axis=0)

    def forbidden(a):
        raise AssertionError("Unambiguous model must not be scored")

    selected, c = _a_scan_points_for_current_scan(metadata(raw, 12, 16), scorer=forbidden)
    np.testing.assert_array_equal(selected, raw)
    assert c["geometry_candidate_scores"] == []


def test_incomplete_scan_model_is_rejected():
    with pytest.raises(ValueError, match="complete scan"):
        _a_scan_points_for_current_scan(metadata(np.repeat(np.eye(3)[None, :, :], 3, axis=0), 0, 5))


@pytest.mark.parametrize("offset", [-1, 4, 1.5])
def test_invalid_override_is_rejected(offset):
    raw = np.repeat(np.eye(3)[None, :, :], 7, axis=0)
    with pytest.raises(ValueError, match="integer"):
        _a_scan_points_for_current_scan(metadata(raw, 0, 3), source_index_offset=offset)


def test_crop_without_geometry_does_not_guess_from_frame_zero():
    with pytest.raises(ValueError, match="predicted-row geometry"):
        _a_scan_points_for_current_scan(metadata(np.repeat(np.eye(3)[None, :, :], 7, axis=0), 2, 5))
