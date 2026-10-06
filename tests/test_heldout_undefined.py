"""Undefined amplitude ratios must not become perfect scores or confidence intervals."""

import numpy as np
import pytest

from grond.reports.heldout import _split_block
from grond.validation.heldout import paired_orbit_bootstrap, r_metrics
from grond.workflows.hand_statistics import heldout_hand_rule


def bootstrap(y, sigma, paired):
    y, sigma = np.asarray(y), np.asarray(sigma)
    return paired_orbit_bootstrap(y, sigma, y * 0.81, y if paired else None,
                                  np.array(["a", "b"]), np.zeros(2, dtype=int), np.ones(2, dtype=bool),
                                  n_boot=200, seed=4)


@pytest.mark.parametrize("paired", [False, True])
def test_no_observed_rows_have_no_observed_score_or_interval(paired):
    result = bootstrap([1.0, 2.0], [1.0, 1.0], paired)
    observed = result["r_obs"]

    assert np.isnan(observed["delta" if paired else "value"])
    assert np.all(np.isnan(observed["ci95"]))
    assert observed["ci_status"] == "no_positive_observed_amplitude"
    assert observed["n_undefined_resamples"] == 200
    assert np.all(np.isfinite(result["r_all"]["ci95"]))


@pytest.mark.parametrize("paired", [False, True])
def test_draws_without_observed_rows_do_not_create_zero_residuals(paired):
    observed = bootstrap([100.0, 1.0], [1.0, 1.0], paired)["r_obs"]

    assert observed["delta" if paired else "value"] == pytest.approx(10.0)
    assert 0 < observed["n_undefined_resamples"] < 200
    assert observed["ci_status"] == "undefined_bootstrap_resamples"
    assert np.all(np.isnan(observed["ci95"]))


def test_report_explains_an_unavailable_observed_interval():
    y, sigma = np.array([100.0, 1.0]), np.ones(2)
    result = _split_block(y, sigma, y * 0.81, np.array(["a", "b"]), np.zeros(2, dtype=int), np.ones(2, dtype=bool))

    assert result["r_obs"] == pytest.approx(10.0)
    assert "r_obs: undefined_bootstrap_resamples" in result["ci_status"]
    assert result["bootstrap_diagnostics"]["r_obs"]["n_undefined_resamples"] > 0


def test_an_unavailable_interval_cannot_support_a_hand_switch():
    interval = bootstrap([100.0, 1.0], [1.0, 1.0], True)["r_obs"]
    accepted, reasons = heldout_hand_rule({"orbit_sign_z": 10.0, "delta_r_obs": interval}, orbit_sign_z_min=3.0)
    assert not accepted
    assert any("unavailable" in reason for reason in reasons)


def test_amplitude_metrics_and_bootstrap_are_invariant_to_common_intensity_units():
    y, sigma = np.array([100.0, 25.0]), np.array([1.0, 1.0])
    baseline = bootstrap(y, sigma, True)
    tiny = bootstrap(y * 1e-30, sigma * 1e-30, True)
    assert r_metrics(y * 1e-30, sigma * 1e-30, y * 0.81e-30)["r_all"] == pytest.approx(10.0)
    for metric in ("r_all", "r_obs"):
        assert tiny[metric]["delta"] == pytest.approx(baseline[metric]["delta"])
        assert tiny[metric]["ci95"] == pytest.approx(baseline[metric]["ci95"])
