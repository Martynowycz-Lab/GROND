import json

import numpy as np
import pytest

from grond.io.dials_bridge import (
    _choose_intensity,
    _dials_row_selection_keep,
    _dials_sigma_multiplier,
    _require_valid_dials_geometry,
    _rotation_matrix,
    convert_dials_to_grond,
    dials_sg_statistics,
)
from grond.legacy.engine.bloch_engine import XDSData


def test_dials_scaled_row_selection_is_independent_of_intensity_mode():
    arrays = {
        "integrated_sum": np.array([True, True, False, True]),
        "scaled": np.array([True, False, True, True]),
    }

    np.testing.assert_array_equal(
        _dials_row_selection_keep(arrays, 4, "integrated"),
        np.array([True, True, False, True]),
    )
    np.testing.assert_array_equal(
        _dials_row_selection_keep(arrays, 4, "scaled"),
        np.array([True, False, False, True]),
    )


def test_dials_scaled_row_selection_fails_closed_without_scaled_rows():
    arrays = {
        "integrated_sum": np.array([True, True]),
        "scaled": np.array([False, False]),
    }

    with pytest.raises(ValueError, match="contains no rows flagged"):
        _dials_row_selection_keep(arrays, 2, "scaled")


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
        "inverse_scale_factor": np.array([1.25]),
        "inverse_scale_factor_variance": np.array([0.04]),
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
        "space_group_number": 1,
        "experiment_identifier": "crystal-uuid-1",
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
        # Place the synthetic calculated centroid on the Ewald crossing for
        # the fixed/setting rotations above so the bridge geometry gate is real.
        "scan": {"image_range": [1, 1], "oscillation": [-35.36177806085131, 1.0]},
        "detector": {"image_size_px": [10, 10], "pixel_size_mm": [0.01, 0.01], "gain": 2.0},
        "profile": {
            "model": "GaussianRSProfileModel",
            "parameters": {
                "__id__": "gaussian_rs",
                "sigma_b": 0.0012,
                "sigma_m": 0.0023,
            },
        },
        "frame_sliced_shoebox": {
            "available": True,
            "used_for_intensity": False,
            "status": "recorded_for_future_rocking_profile_import",
        },
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
    assert data.header["SPACE_GROUP_NUMBER"] == "1"
    assert data.header["DIALS_INDEXING_BASIS"] == "miller_index_from_source_reflection_table"
    assert data.header["DIALS_EXPERIMENT_IDENTIFIER"] == "crystal-uuid-1"
    assert summary["geometry"]["space_group_number"] == 1
    assert summary["geometry"]["experiment_identifier"] == "crystal-uuid-1"
    assert data.header["DIALS_PROFILE_MODEL"] == "GaussianRSProfileModel"
    profile = data.integration_profile_provenance()
    assert profile["dials"]["profile_model"] == "GaussianRSProfileModel"
    assert profile["dials"]["profile_parameters"]["sigma_b"] == pytest.approx(0.0012)
    assert profile["dials"]["gaussian_rs_sigma_m_deg"] == pytest.approx([0.0023])
    assert profile["optional_prior"]["projected_mosaicity_sigma_deg"] == pytest.approx(0.0023)
    assert summary["geometry"]["profile"]["parameters"]["sigma_m"] == pytest.approx(0.0023)
    assert data.header["DIALS_FRAME_SLICED_SHOEBOX_AVAILABLE"] == "TRUE"
    assert profile["dials"]["frame_sliced_shoebox_available"] is True
    control = summary["dials_inverse_scale_control"]
    assert control["status"] == "recorded_fixed_control"
    assert control["available_rows"] == 1
    assert data.header["DIALS_INVERSE_SCALE_CONTROL_FILE"] == "DIALS_INVERSE_SCALE_CONTROL.npz"
    with np.load(tmp_path / "bridge" / "DIALS_INVERSE_SCALE_CONTROL.npz") as loaded:
        assert loaded["inverse_scale_factor"][0] == pytest.approx(1.25)
        assert loaded["inverse_scale_factor_variance"][0] == pytest.approx(0.04)


def test_bridge_preserves_cropped_dials_scan_origin(tmp_path, monkeypatch):
    n = 1
    first = np.diag([0.10, 0.11, 0.12])
    second = np.diag([0.20, 0.21, 0.22])
    reciprocal_length_at_centroid = 0.5 * (first[1, 1] + second[1, 1])
    bragg_angle = np.degrees(np.arcsin(reciprocal_length_at_centroid / (2.0 * 40.0)))
    oscillation_width = 0.888
    arrays = {
        "hkl": np.array([[0, 1, 0]], dtype=int),
        "i_sum": np.array([10.0]),
        "var_sum": np.array([4.0]),
        "i_scale": np.array([np.nan]),
        "var_scale": np.array([np.nan]),
        "lp": np.array([1.0]),
        "qe": np.array([1.0]),
        "partiality": np.array([1.0]),
        "xyzcal": np.array([[1.0, 2.0, 5.5]]),
        "xyzobs": np.array([[1.1, 2.1, 5.6]]),
        "bbox": np.array([[0, 1, 0, 1, 5, 7]], dtype=int),
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
    metadata = {
        "space_group": "P 1",
        "unit_cell": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        "A_static": first.tolist(),
        "A_scan_points": [first.tolist(), second.tolist()],
        "beam": {"wavelength_A": 0.025, "s0": [0.0, 0.0, -40.0]},
        "goniometer": {
            "rotation_axis": [1.0, 0.0, 0.0],
            "rotation_axis_datum": [1.0, 0.0, 0.0],
            "fixed_rotation": np.eye(3).tolist(),
            "setting_rotation": np.eye(3).tolist(),
        },
        "scan": {
            "image_range": [6, 136],
            "array_range": [5, 136],
            "oscillation": [bragg_angle - 0.5 * oscillation_width, oscillation_width],
        },
        "detector": {"image_size_px": [10, 10], "pixel_size_mm": [0.01, 0.01], "gain": 2.0},
    }
    monkeypatch.setattr(
        "grond.io.dials_bridge._run_dials_helper",
        lambda *_args, **_kwargs: (arrays, metadata),
    )

    output = convert_dials_to_grond("input.refl", "input.expt", tmp_path / "bridge")
    data = XDSData(output.hkl)
    summary = json.loads(output.summary.read_text())

    assert data.starting_frame == pytest.approx(5.0)
    assert data.zcal[0] == pytest.approx(5.5)
    assert data.omega_from_z(5.5) == pytest.approx(bragg_angle)
    np.testing.assert_allclose(data.A_for_omega(data.omega_from_z(5.5)), 0.5 * (first + second))
    assert summary["geometry"]["scan_image_range"] == [6, 136]
    assert summary["geometry"]["scan_array_range"] == [5, 136]
    assert summary["geometry"]["starting_frame"] == 5
    assert summary["sg_validation"]["starting_frame"] == 5
    assert summary["sg_validation"]["status"] == "ok"

    metadata["scan"]["oscillation"] = [-55.56, oscillation_width]
    failed_dir = tmp_path / "failed_bridge"
    with pytest.raises(ValueError, match="Refusing to write GROND rows"):
        convert_dials_to_grond("input.refl", "input.expt", failed_dir)
    failed_summary = json.loads((failed_dir / "dials_bridge_summary.json").read_text())
    assert failed_summary["status"] == "failed"
    assert failed_summary["sg_validation"]["status"] == "error"
    assert not (failed_dir / "INTEGRATE.HKL").exists()


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


def test_full_scan_a_points_are_cropped_to_nonzero_scan_origin():
    from grond.io.dials_bridge import _a_scan_points_for_current_scan

    full = np.asarray([np.eye(3) * (1.0 + 0.01 * i) for i in range(148)])
    metadata = {
        "A_scan_points": full.tolist(),
        "scan": {
            "image_range": [27, 96],
            "array_range": [26, 96],
            "oscillation": [11.3204, 0.4354],
        },
    }

    selected, contract = _a_scan_points_for_current_scan(metadata)

    assert selected.shape == (71, 3, 3)
    np.testing.assert_allclose(selected[0], full[26])
    np.testing.assert_allclose(selected[-1], full[96])
    assert contract == {
        "source": "cropped_from_full_scan_model",
        "raw_scan_points": 148,
        "written_scan_points": 71,
        "source_scan_point_origin": 26,
        "scan_array_range": [26, 96],
        "geometry_candidate_scores": [],
    }


def test_scan_a_geometry_scoring_can_align_a_model_with_nonzero_source_origin():
    from grond.io.dials_bridge import _a_scan_points_for_current_scan

    full = np.asarray([np.eye(3) * (1.0 + 0.01 * i) for i in range(144)])
    metadata = {
        "A_scan_points": full.tolist(),
        "scan": {
            "image_range": [94, 147],
            "array_range": [93, 147],
            "oscillation": [40.4922, 0.4354],
        },
    }

    def score(candidate):
        return abs(float(candidate[0, 0, 0]) - float(full[89, 0, 0]))

    selected, contract = _a_scan_points_for_current_scan(metadata, scorer=score)

    np.testing.assert_allclose(selected[0], full[89])
    np.testing.assert_allclose(selected[-1], full[143])
    assert contract["source"] == "geometry_selected_cropped_scan_model"
    assert contract["source_scan_point_origin"] == 89
    assert contract["geometry_candidate_scores"][0]["source_index_offset"] == 89


def test_dials_sg_statistics_normalizes_cropped_scan_z_and_fails_closed_without_it():
    reciprocal_length = 0.1
    beam_length = 40.0
    bragg_angle = np.degrees(np.arcsin(reciprocal_length / (2.0 * beam_length)))
    oscillation_width = 0.888
    metadata = {
        "A_scan_points": [np.diag([reciprocal_length] * 3).tolist()],
        "goniometer": {
            "rotation_axis": [1.0, 0.0, 0.0],
            "rotation_axis_datum": [1.0, 0.0, 0.0],
            "fixed_rotation": np.eye(3).tolist(),
            "setting_rotation": np.eye(3).tolist(),
        },
        "beam": {"s0": [0.0, 0.0, -beam_length]},
        "scan": {
            "image_range": [6, 136],
            "array_range": [5, 136],
            "oscillation": [bragg_angle - 0.5 * oscillation_width, oscillation_width],
        },
    }
    arrays = {
        "hkl": np.array([[0, 1, 0]], dtype=int),
        "xyzcal": np.array([[0.0, 0.0, 5.5]], dtype=float),
    }

    corrected = dials_sg_statistics(arrays, metadata, np.array([True]))

    assert corrected["status"] == "ok"
    assert corrected["starting_frame"] == 5
    assert corrected["median_abs_sg_inv_A"] == pytest.approx(0.0, abs=1e-12)

    broken_metadata = dict(metadata)
    broken_metadata["scan"] = {
        "image_range": [1, 136],
        "array_range": [0, 136],
        "oscillation": metadata["scan"]["oscillation"],
    }
    broken = dials_sg_statistics(arrays, broken_metadata, np.array([True]))

    assert broken["status"] == "error"
    assert broken["p95_abs_sg_inv_A"] > broken["limits_inv_A"]["p95_abs_sg"]
    with pytest.raises(ValueError, match="Refusing to write GROND rows"):
        _require_valid_dials_geometry(broken)


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
