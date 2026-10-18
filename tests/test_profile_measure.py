"""Direct-angle/Sg measure equivalence on independently integrable rocking curves, without fitted scales."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from grond.bloch.config import Config, frozen_kernel_contract
from grond.bloch.errors import LocalSgCrossingError
from grond.engine.bloch_engine import XDSBlochPort


def port(*, coordinate="direct_angle", normalization="target_sg_equivalent_v1", derivative=0.01,
         curvature=0.0, points=80, accumulation="sum", rule="gauss_legendre", bound=4.0):
    result = object.__new__(XDSBlochPort)
    result.cfg = Config(profile_coordinate=coordinate, profile_normalization=normalization,
                        profile_quadrature=rule, profile_accumulation=accumulation, n_points=points,
                        sg_window=0.004, sg_window_mode="dials_bbox", profile_angle_max_half_width_deg=bound)
    result.data = SimpleNamespace(oscillation_range=0.5)
    result.center, result.center_sg = 7.0, 0.0003

    def excitation(omega):
        x = float(omega) - result.center
        return result.center_sg + derivative * x + curvature * x*x

    def root(_row, _hkl, _center, target, _meta):
        delta = float(target) - result.center_sg
        if curvature == 0.0:
            return result.center + delta / derivative
        roots = np.roots([curvature, derivative, -delta])
        return result.center + float(roots[np.argmin(np.abs(roots))])

    result.sg_for_hkls = lambda _hkl, omega: np.asarray([excitation(omega)])
    result.sg_derivative_per_deg = lambda _hkl, omega: derivative + 2*curvature*(float(omega)-result.center)
    result.solve_sg_profile_target = root
    return result


def samples(p, width=0.006):
    return p.row_profile_omega_points(0, np.array([1, 0, 0]), p.center, p.center_sg, width)


@pytest.mark.parametrize("accumulation", ["sum", "mean"])
@pytest.mark.parametrize("curvature", [0.0, 0.002])
def test_zero_cone_direct_angle_matches_sg_and_independent_angular_integral(accumulation, curvature):
    direct = port(curvature=curvature, accumulation=accumulation)
    target = port(coordinate="target_sg", normalization="legacy", curvature=curvature, accumulation=accumulation)
    omega, weights, residual, meta, _ = samples(direct)
    sg_omega, sg_weights, *_ = samples(target)
    def intensity(w):
        return np.exp(-35*(w-direct.center-0.04)**2) + 0.3*np.cos(8*(w-direct.center))**2
    lower, upper = meta["sg_profile_angle_interval_deg"]
    total = 80.0 if accumulation == "sum" else 1.0
    expected = total * 1.5 * quad(intensity, lower, upper, epsabs=1e-13)[0] / (upper-lower)
    assert weights @ intensity(omega) == pytest.approx(expected, rel=2e-12)
    assert sg_weights @ intensity(sg_omega) == pytest.approx(expected, rel=2e-12)
    assert sum(weights) == pytest.approx(total * 1.5)
    assert np.max(abs(residual)) < 1e-12
    assert meta["sg_profile_normalization"] == "target_sg_equivalent_v1"
    assert not meta["sg_profile_angle_half_width_clipped"]
    if curvature:
        assert abs((lower+upper)/2-direct.center) > 0.01  # The old symmetric linearized interval is not equivalent.


@pytest.mark.parametrize("derivative", [0.002, 0.01, -0.01])
@pytest.mark.parametrize("width", [0.004, 0.008])
def test_born_lorentz_completed_row_and_absolute_handoff_need_no_fitted_scale(derivative, width):
    p = port(derivative=derivative)
    omega, weights, _, meta, _ = samples(p, width)
    thickness, h10 = 100.0, 0.0001

    def born(w):
        s = p.center_sg + derivative*(w-p.center)
        return (2*np.pi*h10*thickness)**2 * np.sinc(s*thickness)**2

    lower, upper = meta["sg_profile_angle_interval_deg"]
    # In a locally linear scan, this is the Lorentz-corrected angle integral in the legacy global units Q/S0.
    expected = 80 / 0.004 * abs(derivative) * quad(born, lower, upper, epsabs=1e-13)[0]
    predicted = weights @ born(omega)
    assert predicted == pytest.approx(expected, rel=2e-12)
    kappa_fc2, observed = 250.0, 120.0
    assert kappa_fc2 * observed / predicted == pytest.approx(kappa_fc2 * observed / expected, rel=2e-12)


def test_legacy_direct_angle_values_remain_unchanged_and_new_mode_refuses_clipping():
    legacy = port(normalization="legacy", bound=0.12, points=7, rule="uniform")
    omega, weights, residual, meta, _ = samples(legacy)
    np.testing.assert_array_equal(omega, legacy.center + np.linspace(-0.12, 0.12, 7))
    np.testing.assert_array_equal(weights, np.ones(7))
    assert len(residual) == 0 and meta["sg_profile_angle_half_width_clipped"]
    equivalent = port(bound=0.12)
    with pytest.raises(LocalSgCrossingError, match="refusing to clip"):
        samples(equivalent)


def test_endpoint_interval_refuses_an_unverified_branch():
    p = port()
    p.same_branch_interval_error = lambda *_: "derivative reverses inside interval"
    with pytest.raises(LocalSgCrossingError, match="selected branch"):
        samples(p)


def test_full_cone_retains_one_fixed_angular_interval_and_normalized_orientation_weights():
    p = port(points=20)
    p.cfg = replace(p.cfg, mosaicity_mode="gaussian_cone", mosaicity_sigma_deg=0.1,
                    mosaicity_points=3, mosaicity_support_sigma_deg=0.1)
    omega, tilts, weights, _, meta, mosaic = p.row_profile_orientation_points(
        0, np.array([1, 0, 0]), p.center, p.center_sg, 0.006)
    assert len(omega) == 20*9 and mosaic["mosaicity_cone_points"] == 9
    np.testing.assert_allclose(omega.reshape(20, 9), omega.reshape(20, 9)[:, :1]*np.ones((1, 9)))
    assert weights.sum() == pytest.approx(20*1.5)
    # Independent Gaussian second moments: each angular coordinate has variance sigma^2.
    assert weights @ np.sum(tilts**2, axis=1) / weights.sum() == pytest.approx(2*0.1**2)
    assert meta["sg_profile_normalization"] == "target_sg_equivalent_v1"


@pytest.mark.parametrize("rule", ["uniform", "trapezoid", "gauss_legendre"])
def test_changing_q_only_changes_declared_sum_for_constant_integrand(rule):
    low = port(points=60, rule=rule)
    high = port(points=120, rule=rule)
    _, low_weights, *_ = samples(low)
    _, high_weights, *_ = samples(high)
    assert high_weights.sum() / low_weights.sum() == pytest.approx(2.0)
    assert high_weights.sum()/120 == pytest.approx(low_weights.sum()/60)
    # A common count factor cancels in q, but must still be accounted for in the absolute Icalc handoff.
    assert (0.7*high_weights.sum())/(0.2*high_weights.sum()) == pytest.approx(
        (0.7*low_weights.sum())/(0.2*low_weights.sum()))


def test_normalization_is_explicit_in_config_contract_and_cache_identity():
    with pytest.raises(ValueError, match="direct_angle"):
        Config(profile_normalization="target_sg_equivalent_v1")
    with pytest.raises(ValueError, match="direct_angle"):
        Config(mosaicity_mode="gaussian_cone", profile_coordinate="target_sg")
    p = port()
    assert frozen_kernel_contract(p.cfg)["geometry"]["profile_normalization"] == "target_sg_equivalent_v1"
    p.geometry_path, p.dose_ref_frame = None, 0.0
    p.model = {"cell": [10., 10., 10., 90., 90., 90.]}
    p.data = SimpleNamespace(iobs=np.ones(2), verify_source_files=lambda: {"hkl": "input-hash"},
                             hkl_transform_spec="identity", wavelength=0.025)
    p.cfg = replace(p.cfg, row_plan_cache="readwrite")
    current, fields = p.row_plan_cache_key(np.array([0, 1]))
    p.cfg = replace(p.cfg, profile_normalization="legacy")
    legacy, _ = p.row_plan_cache_key(np.array([0, 1]))
    assert fields["profile_normalization"] == "target_sg_equivalent_v1" and current != legacy
