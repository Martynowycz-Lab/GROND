import json

import numpy as np
import pytest

from grond.io.dials_bridge import (
    _choose_intensity,
    _dials_sigma_multiplier,
    _rotation_matrix,
    convert_dials_to_grond,
    dials_sg_statistics,
)
from grond.legacy.engine.bloch_engine import XDSData


def test_sum_complete_uses_lp_over_qe_and_preserves_recorded_intensity():
    arrays = {
        "i_sum": np.array([10.0]),
        "var_sum": np.array([4.0]),
        "lp": np.array([2.0]),
        "qe": np.array([0.5]),
        "partiality": np.array([0.8]),
    }

    iobs, sigma, correction, peak, note = _choose_intensity(arrays, "sum_complete", eps=1e-12)

    assert correction[0] == pytest.approx(4.0)
    assert iobs[0] == pytest.approx(50.0)
    assert sigma[0] == pytest.approx(10.0)
    assert peak[0] == pytest.approx(80.0)
    assert iobs[0] * peak[0] / 100.0 / correction[0] == pytest.approx(10.0)
    assert "lp/qe" in note


def test_bridge_writes_qe_correction_full_geometry_and_raw_sigma_default(tmp_path, monkeypatch):
    n = 1
    arrays = {
        "hkl": np.array([[1, 0, 0]], dtype=int),
        "i_sum": np.array([10.0]),
        "var_sum": np.array([4.0]),
        "i_scale": np.array([np.nan]),
        "var_scale": np.array([np.nan]),
        "lp": np.array([2.0]),
        "qe": np.array([0.5]),
        "partiality": np.array([0.8]),
        "xyzcal": np.array([[1.0, 2.0, 0.5]]),
        "xyzobs": np.array([[1.1, 2.1, 0.6]]),
        "bbox": np.array([[0, 1, 0, 1, 0, 1]], dtype=int),
        "panel": np.zeros(n, dtype=int),
        "integrated_sum": np.ones(n, dtype=bool),
        "integrated_prf": np.zeros(n, dtype=bool),
        "dont_integrate": np.zeros(n, dtype=bool),
        "scaled": np.zeros(n, dtype=bool),
    }
    for flag in (
        "overloaded",
        "in_powder_ring",
        "excluded_for_scaling",
        "bad_for_scaling",
        "outlier_in_scaling",
        "background_includes_bad_pixels",
        "foreground_includes_bad_pixels",
        "includes_bad_pixels",
        "bad_shoebox",
        "failed_during_background_modelling",
        "failed_during_summation",
    ):
        arrays[flag] = np.zeros(n, dtype=bool)
    fixed = _rotation_matrix(np.array([0.0, 0.0, 1.0]), 12.0)
    setting = _rotation_matrix(np.array([0.0, 1.0, 0.0]), -7.0)
    axis_datum = np.array([1.0, 0.0, 0.0])
    metadata = {
        "space_group": "P 1",
        "unit_cell": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        "A_static": np.diag([0.1, 0.1, 0.1]).tolist(),
        "A_scan_points": [np.diag([0.1, 0.1, 0.1]).tolist()],
        "beam": {"wavelength_A": 0.025, "s0": [0.0, 0.0, -40.0]},
        "goniometer": {
            "rotation_axis": (setting @ axis_datum).tolist(),
            "rotation_axis_datum": axis_datum.tolist(),
            "fixed_rotation": fixed.tolist(),
            "setting_rotation": setting.tolist(),
        },
        "scan": {"image_range": [1, 1], "oscillation": [0.0, 1.0]},
        "detector": {"image_size_px": [10, 10], "pixel_size_mm": [0.01, 0.01], "gain": 2.0},
    }
    monkeypatch.setattr(
        "grond.io.dials_bridge._run_dials_helper",
        lambda *_args, **_kwargs: (arrays, metadata),
    )

    output = convert_dials_to_grond("input.refl", "input.expt", tmp_path / "bridge")
    data = XDSData(output.hkl)
    summary = json.loads(output.summary.read_text())

    assert data.iobs[0] == pytest.approx(50.0)
    assert data.sigma[0] == pytest.approx(10.0)
    assert data.rlp[0] == pytest.approx(4.0)
    assert data.recorded_proxy()[0] == pytest.approx(10.0)
    assert data.fixed_rotation == pytest.approx(fixed)
    assert data.setting_rotation == pytest.approx(setting)
    assert summary["sigma_model"]["mode"] == "raw"
    assert summary["sigma_model"]["source"] == "raw"


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


def test_dials_sg_statistics_uses_full_setting_rotation_fixed_rotation_geometry():
    axis_datum = np.array([1.0, 0.0, 0.0])
    fixed = _rotation_matrix(np.array([0.0, 0.0, 1.0]), 27.0)
    setting = _rotation_matrix(np.array([0.0, 1.0, 0.0]), -18.0)
    a_matrix = np.diag([0.1, 0.125, 0.15])
    hkl = np.array([2.0, 1.0, -1.0])
    omega = 13.0
    s0 = np.array([0.0, 0.0, -40.0])
    axis_lab = setting @ axis_datum
    arrays = {
        "hkl": hkl[None, :].astype(int),
        "xyzcal": np.array([[0.0, 0.0, 1.5]], dtype=float),
    }
    metadata = {
        "A_scan_points": [a_matrix.tolist()],
        "goniometer": {
            "rotation_axis": axis_lab.tolist(),
            "rotation_axis_datum": axis_datum.tolist(),
            "fixed_rotation": fixed.tolist(),
            "setting_rotation": setting.tolist(),
        },
        "beam": {"s0": s0.tolist()},
        "scan": {"oscillation": [10.0, 2.0]},
    }

    stats = dials_sg_statistics(arrays, metadata, np.array([True]))

    g = setting @ _rotation_matrix(axis_datum, omega) @ fixed @ a_matrix @ hkl
    expected = (np.dot(s0, s0) - np.dot(s0 + g, s0 + g)) / (2.0 * np.linalg.norm(s0))
    assert stats["median_sg_inv_A"] == pytest.approx(expected, abs=1e-12)


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
