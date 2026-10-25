"""Target-Sg rocking measure on an independently integrable rocking curve, without fitted scales."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from grond.bloch.config import Config
from grond.engine.bloch_engine import XDSBlochPort


def port(*, derivative=0.01, curvature=0.0, points=80, accumulation="sum", rule="gauss_legendre"):
    result = object.__new__(XDSBlochPort)
    result.cfg = Config(profile_quadrature=rule, profile_accumulation=accumulation, n_points=points,
                        sg_window=0.004, sg_window_mode="dials_bbox")
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
def test_target_sg_profile_matches_independent_angular_integral(accumulation, curvature):
    target = port(curvature=curvature, accumulation=accumulation)
    sg_omega, sg_weights, *_ = samples(target)
    def intensity(w):
        return np.exp(-35*(w-target.center-0.04)**2) + 0.3*np.cos(8*(w-target.center))**2
    lower, upper = sorted(target.solve_sg_profile_target(0, None, target.center, target.center_sg + 0.5*side*0.006, {})
                          for side in (-1, 1))
    total = 80.0 if accumulation == "sum" else 1.0
    expected = total * 1.5 * quad(intensity, lower, upper, epsabs=1e-13)[0] / (upper-lower)
    assert sg_weights @ intensity(sg_omega) == pytest.approx(expected, rel=2e-12)
