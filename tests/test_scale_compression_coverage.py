import numpy as np
import pytest

from grond.legacy.engine.scale_compression_polish import (
    fit_group_detector_scales,
    fit_group_scales,
)


def test_compact_group_without_work_rows_borrows_work_global_scale():
    observations = np.asarray([20.0, 20.0, 5000.0])
    sigma = np.ones(3)
    raw_prediction = np.full(3, 10.0)
    groups = np.asarray([0, 0, 1])
    work = np.asarray([True, True, False])

    scaled, scales = fit_group_scales(
        observations,
        sigma,
        raw_prediction,
        groups,
        work,
        "native",
        0.0,
        0.0,
        0.01,
    )

    assert scales == {"0": 2.0, "1": 2.0}
    np.testing.assert_allclose(scaled, [20.0, 20.0, 20.0])


def test_compact_detector_bin_seen_only_in_free_rows_stays_finite_without_leakage():
    n_work = 10
    sigma = np.ones(n_work + 1)
    raw_prediction = np.full(n_work + 1, 10.0)
    groups = np.zeros(n_work + 1, dtype=int)
    detector_bins = np.asarray([0] * n_work + [50])
    work = np.asarray([True] * n_work + [False])

    def fit(free_observation: float):
        observations = np.asarray([20.0] * n_work + [free_observation])
        return fit_group_detector_scales(
            observations,
            sigma,
            raw_prediction,
            groups,
            detector_bins,
            work,
            "native",
            0.0,
            0.0,
            0.01,
        )

    scaled, frame_scales, detector_scales = fit(5000.0)
    changed_scaled, changed_frame_scales, changed_detector_scales = fit(0.001)

    np.testing.assert_allclose(scaled, np.full(n_work + 1, 20.0))
    np.testing.assert_allclose(changed_scaled, scaled)
    assert frame_scales == changed_frame_scales == {"0": 2.0}
    assert detector_scales == changed_detector_scales
    assert len(detector_scales) == 64
    assert detector_scales["0"] == detector_scales["50"] == 1.0
    assert set(detector_scales.values()) == {1.0}


def test_compact_scale_fit_fails_closed_without_two_fit_rows():
    with pytest.raises(ValueError, match="at least two scale-fit rows"):
        fit_group_scales(
            np.asarray([20.0, 5000.0]),
            np.ones(2),
            np.full(2, 10.0),
            np.asarray([0, 1]),
            np.asarray([True, False]),
            "native",
            0.0,
            0.0,
            0.01,
        )
