from types import SimpleNamespace

import numpy as np

from grond.legacy.engine import native_row_regularize as native
from grond.legacy.engine.native_row_regularize import (
    automatic_frame_offset_grid,
    fit_group_scales,
    kernel_variance_terms,
    native_metrics,
    rank_trials,
    resolve_frame_offset_grid,
    xds_frame_datum_summary,
)


def test_automatic_frame_grid_tracks_xds_datum_with_symmetric_residual_probes():
    assert automatic_frame_offset_grid(1.3447) == [1.0, 1.4, 1.8]
    assert automatic_frame_offset_grid(0.0) == [-0.4, 0.0, 0.4]
    assert automatic_frame_offset_grid(float("nan")) == [0.0, 0.2]


def test_xds_frame_datum_requires_many_consistent_geometry_roots(monkeypatch):
    data = SimpleNamespace(
        zcal=np.arange(20, dtype=float),
        hkl=np.zeros((20, 3), dtype=int),
        oscillation_range=-0.2,
        omega_from_z=lambda z: float(z),
    )
    monkeypatch.setattr(native, "_xds_header_bragg_omega", lambda data, hkl, omega: omega - 0.25)

    reliable = xds_frame_datum_summary(data, np.arange(20))
    underdetermined = xds_frame_datum_summary(data, np.arange(1))
    fallback_grid, fallback = resolve_frame_offset_grid("auto", data, np.arange(1))

    assert reliable["status"] == "ok"
    assert reliable["median_offset_frames"] == 1.25
    assert underdetermined["status"] == "unreliable"
    assert fallback["status"] == "unreliable"
    assert fallback_grid == [0.0, 0.2]


def test_work_fitted_scale_is_invariant_to_free_observations():
    prediction = np.full(4, 10.0)
    sigma = np.ones(4)
    groups = np.zeros(4, dtype=int)
    work = np.asarray([True, True, False, False])

    _, original = fit_group_scales(
        np.asarray([20.0, 20.0, 30.0, 30.0]),
        sigma,
        prediction,
        groups,
        work,
        0.0,
        0.0,
    )
    _, changed = fit_group_scales(
        np.asarray([20.0, 20.0, 3000.0, 0.01]),
        sigma,
        prediction,
        groups,
        work,
        0.0,
        0.0,
    )

    assert original == changed == {0: 2.0}


def test_group_without_work_rows_borrows_finite_global_scale():
    observations = np.asarray([20.0, 20.0, 50.0])
    prediction = np.asarray([10.0, 10.0, 10.0])
    groups = np.asarray([0, 0, 1])
    work = np.asarray([True, True, False])

    scaled, scales = fit_group_scales(
        observations,
        np.ones(3),
        prediction,
        groups,
        work,
        0.0,
        0.0,
    )

    assert scales == {0: 2.0, 1: 2.0}
    np.testing.assert_allclose(scaled, [20.0, 20.0, 20.0])


def test_native_metrics_excludes_nonpositive_sigmas():
    result = native_metrics(
        np.asarray([10.0, 10.0, 10.0]),
        np.asarray([1.0, 0.0, -1.0]),
        np.asarray([10.0, 10.0, 10.0]),
        np.ones(3, dtype=bool),
        0.0,
        0.0,
    )

    assert result["n"] == 1


def test_native_sigma_terms_map_to_frozen_bloch_config_names():
    assert kernel_variance_terms(
        {"sigma_a": 0.1, "sigma_b_corr": 0.2, "sigma_c_peak": 0.3}
    ) == {
        "variance_a": 0.1,
        "variance_b_corr": 0.2,
        "variance_c_peak": 0.3,
    }


def test_work_trial_ranking_does_not_use_free_rows_as_tie_breaker():
    def trial(label, work, free):
        return {
            "label": label,
            "metrics": {
                "work": {"dyn": {"wR2_all": work}},
                "free": {"dyn": {"wR2_all": free}},
            },
        }

    original = [trial("first", 0.2, 0.9), trial("second", 0.2, 0.1)]
    changed_free = [trial("first", 0.2, 0.01), trial("second", 0.2, 9.0)]

    assert rank_trials(original, "work_wR2")[0]["label"] == "first"
    assert rank_trials(changed_free, "work_wR2")[0]["label"] == "first"
