import numpy as np

from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, frozen_kernel_contract


def test_matrix_convention_is_recorded_in_kernel_contract():
    default = frozen_kernel_contract(Config())
    alternate = frozen_kernel_contract(Config(matrix_convention="i_minus_j"))
    assert default["hamiltonian"]["matrix_convention"] == "j_minus_i"
    assert default["hamiltonian"]["offdiagonal_sign"] == "basis[j] - basis[i]"
    assert alternate["hamiltonian"]["matrix_convention"] == "i_minus_j"
    assert alternate["hamiltonian"]["offdiagonal_sign"] == "basis[i] - basis[j]"


def profile_points(cfg: Config):
    port = object.__new__(XDSBlochPort)
    port.cfg = cfg
    return port.row_profile_sg_points(0, np.asarray([1, 0, 0]), 0.004)


def test_profile_quadrature_defaults_to_summed_subexposures():
    points, weights, meta = profile_points(Config(n_points=5))
    assert len(points) == 5
    assert np.isclose(float(np.sum(weights)), 5.0)
    assert meta["sg_profile_accumulation"] == "sum"
    assert meta["sg_profile_quadrature"] == "uniform"
    assert meta["sg_profile_weight_sum"] == 5.0


def test_profile_quadrature_mean_is_legacy_normalized_average():
    _, weights, meta = profile_points(Config(n_points=5, profile_accumulation="mean"))
    assert np.isclose(float(np.sum(weights)), 1.0)
    assert meta["sg_profile_accumulation"] == "mean"


def test_profile_quadrature_rules_have_stable_total_weight():
    for rule in ["uniform", "trapezoid", "gauss_legendre"]:
        points, weights, meta = profile_points(Config(n_points=7, profile_quadrature=rule))
        assert len(points) == 7
        assert np.isclose(float(np.sum(weights)), 7.0)
        assert meta["sg_profile_quadrature"] == rule


def test_single_profile_point_is_centered():
    points, weights, _ = profile_points(Config(n_points=1))
    assert np.allclose(points, [0.0])
    assert np.allclose(weights, [1.0])
