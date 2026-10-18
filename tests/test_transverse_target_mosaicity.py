"""Re-centred transverse mosaic spread (mosaicity_mode="transverse_target")."""

from __future__ import annotations

import numpy as np
import pytest

from grond.bloch.config import Config
from grond.engine.bloch_engine import XDSBlochPort


def stub_port(sigma: float, points: int = 5, n_points: int = 5):
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(n_points=n_points, profile_coordinate="target_sg", mosaicity_mode="transverse_target",
                      mosaicity_sigma_deg=sigma, mosaicity_points=points)
    requested = np.linspace(-0.002, 0.002, n_points)
    port.row_profile_sg_points = lambda *_a: (requested, np.ones(n_points), {})
    a, omega_b = 0.02, 10.0          # dSg/domega (1/A per deg) and Bragg angle of the untilted target
    port.solve_sg_profile_target = lambda _r, _h, _oc, target, _m: omega_b + float(target) / a

    def sg_at(hkls, omega, along, transverse):
        # a transverse tilt moves the target's Sg (first and second order); along must stay zero here
        assert along == 0.0
        return np.asarray([a * (float(omega) - omega_b) + 0.013 * transverse + 0.4 * transverse ** 2])

    port.sg_for_hkls_at_orientation = sg_at
    return port, requested


def test_every_tilted_sample_reproduces_the_untilted_target_excitation_errors():
    port, requested = stub_port(0.15)
    omegas, tilts, weights, resid, pmeta, mmeta = port.row_profile_orientation_points(0, np.array([1, 0, 0]), 10.0, 0.0, 0.004)
    assert len(omegas) == 5 * 5 and tilts.shape == (25, 2)
    assert np.all(tilts[:, 0] == 0.0)                          # only transverse tilts
    assert np.max(np.abs(resid)) < 1e-10                        # target Sg identical for every block
    sg = np.asarray([port.sg_for_hkls_at_orientation(None, o, 0.0, t)[0] for o, (_z, t) in zip(omegas, tilts, strict=True)])
    assert np.allclose(sg, np.tile(requested, 5), atol=1e-10)
    assert weights.sum() == pytest.approx(5.0)                  # summed sub-exposure normalisation kept
    assert len(np.unique(np.round(tilts[:, 1], 12))) == 5 and mmeta["mosaicity_kernel_points"] == 5


def test_zero_width_equals_untilted_target_profile():
    port, requested = stub_port(0.0)
    omegas, tilts, weights, resid, _p, _m = port.row_profile_orientation_points(0, np.array([1, 0, 0]), 10.0, 0.0, 0.004)
    assert np.allclose(omegas, 10.0 + requested / 0.02) and np.all(tilts == 0.0)
    assert np.allclose(weights, np.ones(5))


def test_transverse_target_requires_target_sg_profiles():
    with pytest.raises(ValueError, match="target_sg"):
        Config(profile_coordinate="direct_angle", mosaicity_mode="transverse_target", mosaicity_sigma_deg=0.1)


def test_adaptive_window_weights_integrate_rather_than_average():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(n_points=5, sg_window=0.004, sg_window_mode="dials_bbox", sg_window_min=0.003, sg_window_max=0.03)
    assert port.window_integral_factor(0.012) == pytest.approx(3.0)
    port.cfg = Config(n_points=5, sg_window=0.004)
    assert port.window_integral_factor(0.004) == 1.0
