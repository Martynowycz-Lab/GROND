import numpy as np
import pytest

from grond.io.dials_bridge import _dials_sigma_multiplier, dials_sg_statistics


def test_dials_sg_statistics_uses_scan_point_geometry():
    arrays = {
        "hkl": np.array([[0, 0, 0], [1, 0, 0]], dtype=int),
        "xyzcal": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    }
    metadata = {
        "A_scan_points": [
            [[0.10, 0.0, 0.0], [0.0, 0.10, 0.0], [0.0, 0.0, 0.10]],
            [[0.20, 0.0, 0.0], [0.0, 0.10, 0.0], [0.0, 0.0, 0.10]],
        ],
        "goniometer": {"rotation_axis": [1.0, 0.0, 0.0]},
        "beam": {"s0": [0.0, 0.0, -40.0]},
        "scan": {"oscillation": [0.0, 0.0]},
    }

    stats = dials_sg_statistics(arrays, metadata, np.array([True, True]))

    assert stats["status"] == "ok"
    assert stats["rows"] == 2
    assert stats["max_abs_sg_inv_A"] > 0.0


def test_dials_sigma_auto_keeps_plausible_gain_integrated_rows_raw():
    arrays = {"partiality": np.array([0.99, 0.99, 0.99], dtype=float)}
    metadata = {"detector": {"gain": 204.0}}
    iobs = np.array([100.0, 120.0, 80.0], dtype=float)
    sigma = np.array([10.0, 10.0, 10.0], dtype=float)

    multiplier, summary = _dials_sigma_multiplier(
        arrays,
        metadata,
        iobs,
        sigma,
        mode="sum_complete",
        sigma_mode="auto",
        sigma_scale=None,
    )

    assert multiplier == 1.0
    assert summary["source"] == "auto_raw"
    assert summary["i_over_sigma_after_median"] == pytest.approx(summary["i_over_sigma_before_median"])


def test_dials_sigma_auto_applies_detector_gain_for_grossly_overprecise_rows():
    arrays = {"partiality": np.array([0.99, 0.99, 0.99], dtype=float)}
    metadata = {"detector": {"gain": 204.0}}
    iobs = np.array([10000.0, 12000.0, 8000.0], dtype=float)
    sigma = np.array([10.0, 10.0, 10.0], dtype=float)

    multiplier, summary = _dials_sigma_multiplier(
        arrays,
        metadata,
        iobs,
        sigma,
        mode="sum_complete",
        sigma_mode="auto",
        sigma_scale=None,
    )

    assert multiplier == pytest.approx(np.sqrt(204.0))
    assert summary["source"] == "auto_detector_gain_gross_overprecision"
    assert summary["i_over_sigma_after_median"] < summary["i_over_sigma_before_median"]


def test_dials_sigma_auto_prefers_empirical_duplicate_scatter_when_available():
    hkl = []
    iobs = []
    sigma = []
    for h in range(1, 36):
        hkl.extend([(h, 0, 0), (-h, 0, 0)])
        iobs.extend([100.0 + h, 112.0 + h])
        sigma.extend([1.0, 1.0])
    arrays = {
        "hkl": np.asarray(hkl, dtype=int),
        "partiality": np.full(len(iobs), 0.99, dtype=float),
        "integrated_sum": np.ones(len(iobs), dtype=bool),
    }
    metadata = {"detector": {"gain": 204.0}, "space_group": "P 2 2 2"}

    multiplier, summary = _dials_sigma_multiplier(
        arrays,
        metadata,
        np.asarray(iobs, dtype=float),
        np.asarray(sigma, dtype=float),
        mode="sum_complete",
        sigma_mode="auto",
        sigma_scale=None,
    )

    assert summary["source"] == "auto_empirical_replicate_variance"
    assert multiplier > 1.0
    assert summary["empirical_variance_model"]["usable_groups"] == 35
    assert np.nanmedian(summary["_sigma_effective"]) > 1.0


def test_dials_sigma_raw_keeps_variances_unchanged():
    arrays = {"partiality": np.array([0.99, 0.99], dtype=float)}
    metadata = {"detector": {"gain": 204.0}}
    multiplier, summary = _dials_sigma_multiplier(
        arrays,
        metadata,
        np.array([100.0, 120.0], dtype=float),
        np.array([10.0, 10.0], dtype=float),
        mode="sum_complete",
        sigma_mode="raw",
        sigma_scale=None,
    )

    assert multiplier == 1.0
    assert summary["source"] == "raw"
