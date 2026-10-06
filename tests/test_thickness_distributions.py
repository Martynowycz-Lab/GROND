"""Physical thickness distributions on a support: hat weights, gamma characteristic function, auto evaluator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma as gamma_dist

from grond.bloch import thickness as T
from grond.regularize import nested_selection as NS
from tests._nested_support import G, random_spectra

REFERENCE_GRID = np.arange(0.0, 12000.0 + 1.0, 2.0)


@pytest.mark.parametrize("family,params", [
    ("gamma", {"mean": 700.0, "shape": 0.7}),
    ("gamma", {"mean": 1500.0, "shape": 6.0}),
    ("gamma", {"mean": 300.0, "shape": 0.15}),
    ("wedge", {"t0": 1250.0, "grad": 0.9}),
    ("wedge", {"t0": 5000.0, "grad": 1.0}),
])
def test_hat_weights_are_reference_weights_renormalised_on_the_support(family, params):
    grid = T.support_grid(0.0, 12000.0, 2.0)
    assert np.array_equal(grid, REFERENCE_GRID)
    reference = G.distribution_weights(family, params, REFERENCE_GRID)
    mine = T.distribution_weights(family, params, grid)
    assert mine.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(mine, np.maximum(reference, 0.0) / np.maximum(reference, 0.0).sum(), rtol=1e-12, atol=1e-18)
    if family == "wedge":  # fully inside [0, 12000]: the reference is already normalised
        assert np.allclose(mine, reference, rtol=1e-12, atol=1e-18)


def test_node_absorption_matches_reference_analytic_tilt_to_interpolation_accuracy():
    spec = random_spectra(3, n_beams=8, seed=21)
    table = NS.thickness_table(spec, NS.ThicknessSupport(0.0, 12000.0, 2.0), hands=(1, -1))
    gtab = G.thickness_table(spec, dt=2.0, t_max=12000.0)
    for fam, prm in (("wedge", {"t0": 1250.0, "grad": 0.9}), ("wedge", {"t0": 800.0, "grad": 0.5})):
        for hand in (1, -1):
            mine, _born = NS.table_row_intensities(table, {"family": fam, **prm}, mu=1e-4, hand=hand)
            reference = G.table_intensities(gtab, hand, G.distribution_weights(fam, prm, REFERENCE_GRID, mu=1e-4))
            assert np.allclose(mine, reference, rtol=1e-5)
    w0 = T.distribution_weights("wedge", {"t0": 1250.0, "grad": 0.9}, REFERENCE_GRID)
    w1 = T.distribution_weights("wedge", {"t0": 1250.0, "grad": 0.9}, REFERENCE_GRID, mu=1e-4)
    assert np.allclose(w1, w0 * np.exp(-4 * np.pi * 1e-4 * REFERENCE_GRID), rtol=1e-14, atol=0)


def _dense_truncated_average(spec, family, params, t_min, t_max, hand, n=20000):
    """Midpoint reference of the support-truncated, renormalised average of I(t)."""
    edges = np.linspace(t_min, t_max, n + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    if family == "gamma":
        k, m = params["shape"], params["mean"]
        p = np.diff(gamma_dist.cdf(edges, k, scale=m / k))
    else:
        lo, hi = params["t0"] * (1 - params["grad"]), params["t0"] * (1 + params["grad"])
        p = np.diff(np.clip(edges, lo, hi)) / (hi - lo)
    node = G.node_intensities(spec, mid, hands=(hand,))
    return np.sum(node[hand] * p[None, :], axis=1) / p.sum()


@pytest.mark.parametrize("family,params", [
    ("gamma", {"mean": 300.0, "shape": 0.5}),     # a third of its mass below 10 nm
    ("gamma", {"mean": 3000.0, "shape": 1.0}),    # a tail beyond 1 um
    ("wedge", {"t0": 150.0, "grad": 0.9}),        # crosses t_min
    ("wedge", {"t0": 6000.0, "grad": 0.9}),       # crosses t_max
])
def test_table_is_the_truncated_renormalised_average(family, params):
    spec = random_spectra(2, n_beams=7, seed=31)
    support = NS.ThicknessSupport()
    assert T.mass_outside_support(family, params, support.t_min_A, support.t_max_A) > 1e-3
    table = NS.thickness_table(spec, support, hands=(1, -1))
    for hand in (1, -1):
        got, _born = NS.table_row_intensities(table, {"family": family, **params}, hand=hand)
        ref = _dense_truncated_average(spec, family, params, support.t_min_A, support.t_max_A, hand)
        assert np.allclose(got, ref, rtol=2e-3)


def _adaptive_gamma(spec, params, mu, hand, row=0, lo=0.0, hi=None):
    k, th = params["shape"], params["mean"] / params["shape"]
    hi = gamma_dist.ppf(1 - 1e-13, k, scale=th) if hi is None else hi
    f = lambda t: G.node_intensities({key: v for key, v in spec.items()}, np.array([t]), hands=(hand,))[hand][row, 0] \
        * np.exp(-4 * np.pi * mu * t) * gamma_dist.pdf(t, k, scale=th)  # noqa: E731
    split = min(50.0, 0.1 * params["mean"])
    if lo < split:
        return quad(f, lo, split, limit=400, points=[1e-6, 1e-3, 1e-1, 1.0])[0] + quad(f, split, hi, limit=4000)[0]
    return quad(f, lo, hi, limit=4000)[0]


@pytest.mark.parametrize("params,mu", [({"mean": 700.0, "shape": 0.7}, 0.0), ({"mean": 1500.0, "shape": 6.0}, 5e-5),
                                       ({"mean": 300.0, "shape": 0.15}, 0.0)])
def test_characteristic_function_matches_the_reference_and_is_exact(params, mu):
    spec = random_spectra(1, n_beams=10, n_points=2, seed=41)
    out = NS.cf_row_intensities(spec, {"family": "gamma", **params}, mu, hands=(1, -1))
    for hand in (1, -1):
        reference = G.gamma_cf_intensities(spec, {"family": "gamma", **params, "path": {"model": "none"}}, mu, hand)
        assert out[hand][0] == pytest.approx(float(reference[0]), rel=1e-12)
        assert out[hand][0] == pytest.approx(_adaptive_gamma(spec, params, mu, hand), rel=1e-6)


@pytest.mark.parametrize("delta", [0.0, 1e-12, 3e-8, 1e-6, 1e-5, 3e-4, 2e-3, -7e-4])
@pytest.mark.parametrize("shape,mean", [(0.3, 400.0), (1.5, 900.0), (40.0, 2000.0)])
def test_gamma_born_average_is_exact(delta, shape, mean):
    theta = mean / shape
    got = float(T.gamma_cf_born(2.5e-5, delta, shape, theta))

    def born(t):
        return (2 * np.pi) ** 2 * 2.5e-5 * t ** 2 * np.sinc(delta * t) ** 2 * gamma_dist.pdf(t, shape, scale=theta)

    hi = gamma_dist.ppf(1 - 1e-14, shape, scale=theta)
    ref = quad(born, 0.0, hi, limit=2000, points=np.linspace(0, hi, 50)[1:-1])[0]
    assert got == pytest.approx(ref, rel=1e-7)


def test_gamma_born_is_continuous_across_the_series_branch():
    k, theta = 3.0, 300.0
    edge = 1e-3 / ((k + 3.0) * 2 * np.pi * theta)
    below = float(T.gamma_cf_born(1.0, edge * (1 - 1e-9), k, theta))
    above = float(T.gamma_cf_born(1.0, edge * (1 + 1e-9), k, theta))
    assert below == pytest.approx(above, rel=1e-12)


def test_auto_resolves_to_cf_only_when_the_gamma_mass_outside_is_negligible():
    t_min, t_max = 100.0, 10000.0
    wide = {"mean": 3000.0, "shape": 10.0}
    narrow = {"mean": 900.0, "shape": 1.5}
    assert T.mass_outside_support("gamma", wide, t_min, t_max) < T.CF_MASS_TOLERANCE
    assert T.mass_outside_support("gamma", narrow, t_min, t_max) > 0.04
    assert T.resolve_evaluator("gamma", wide, "auto", t_min, t_max) == "cf"
    assert T.resolve_evaluator("gamma", narrow, "auto", t_min, t_max) == "table"
    assert T.resolve_evaluator("wedge", {"t0": 1000.0, "grad": 0.5}, "auto", t_min, t_max) == "table"
    assert T.resolve_evaluator("gamma", narrow, "cf", t_min, t_max) == "cf"
    with pytest.raises(ValueError, match="gamma family only"):
        T.resolve_evaluator("wedge", {"t0": 1000.0, "grad": 0.5}, "cf", t_min, t_max)
    # Where auto picks cf, the untruncated average equals the truncated renormalised one to the outside mass.
    spec = random_spectra(1, n_beams=8, n_points=1, seed=51)
    cf = NS.cf_row_intensities(spec, {"family": "gamma", **wide})[1][0]
    k, th = wide["shape"], wide["mean"] / wide["shape"]
    mass = gamma_dist.cdf(t_max, k, scale=th) - gamma_dist.cdf(t_min, k, scale=th)
    truncated = _adaptive_gamma(spec, wide, 0.0, 1, lo=t_min, hi=t_max) / mass
    assert abs(cf - truncated) <= T.mass_outside_support("gamma", wide, t_min, t_max)
    table = NS.row_intensities(spec, {"family": "gamma", **wide}, evaluator="table")["dyn"][0]
    assert table == pytest.approx(cf, rel=3e-3)  # 2 A interpolation only


def test_single_thickness_wedge_is_linear_interpolation():
    t = T.support_grid(100.0, 10000.0, 2.0)
    w = T.distribution_weights("wedge", {"t0": 1233.0, "grad": 0.0}, t)
    j = int(np.flatnonzero(w)[0])
    assert np.count_nonzero(w) == 2 and t[j] == 1232.0
    assert w[j] == pytest.approx(0.5) and w[j + 1] == pytest.approx(0.5)
    assert T.distribution_weights("wedge", {"t0": 10000.0, "grad": 0.0}, t)[-1] == 1.0


@pytest.mark.parametrize("call,match", [
    (lambda: T.support_grid(100.0, 10001.0, 2.0), "integer number"),
    (lambda: T.support_grid(100.0, 100.0, 2.0), "exceed its minimum"),
    (lambda: T.support_grid(-1.0, 100.0, 1.0), "non-negative"),
    (lambda: T.support_grid(0.0, 1e7, 1.0), "limit"),
    (lambda: T.distribution_weights("wedge", {"t0": 20000.0, "grad": 0.1}, T.support_grid(100.0, 10000.0, 2.0)), "no mass"),
    (lambda: T.distribution_weights("wedge", {"t0": 1000.0, "grad": 1.2}, T.support_grid(100.0, 10000.0, 2.0)), r"\[0, 1\]"),
    (lambda: T.distribution_weights("gamma", {"mean": 1000.0, "shape": 0.0}, T.support_grid(100.0, 10000.0, 2.0)), "shape"),
    (lambda: T.canonical_family("lognormal"), "unknown thickness family"),
    (lambda: T.resolve_evaluator("gamma", {"mean": 1.0, "shape": 1.0}, "quadrature", 0.0, 1.0), "unknown support"),
])
def test_distributions_fail_closed(call, match):
    with pytest.raises(ValueError, match=match):
        call()
