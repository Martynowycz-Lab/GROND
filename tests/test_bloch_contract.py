import numpy as np
from scipy.linalg import expm

from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    _jax_matrix_bucket_size,
    _real_embedding_generator_batch,
    _stack_square_matrices,
    frozen_kernel_contract,
    promoted_bethe_effective_hamiltonian,
    row_plan_cache_validation_errors,
)


def test_matrix_convention_is_recorded_in_kernel_contract():
    default = frozen_kernel_contract(Config())
    alternate = frozen_kernel_contract(Config(matrix_convention="i_minus_j"))
    assert default["hamiltonian"]["matrix_convention"] == "j_minus_i"
    assert default["hamiltonian"]["offdiagonal_sign"] == "basis[j] - basis[i]"
    assert alternate["hamiltonian"]["matrix_convention"] == "i_minus_j"
    assert alternate["hamiltonian"]["offdiagonal_sign"] == "basis[i] - basis[j]"


def test_row_plan_cache_validation_rejects_stale_metadata():
    indices = np.asarray([3, 7, 11], dtype=np.int64)
    meta = {
        "cache_schema": 1,
        "key": "expected",
        "hkl_sha256": "abc",
        "hkl_transform": "as_input",
        "indices_sha256": "idx",
        "n_basis": 24,
        "n_points": 10,
        "profile_quadrature": "uniform",
        "profile_accumulation": "sum",
    }
    valid_cache = {
        "meta": dict(meta),
        "indices": indices.copy(),
        "basis_offsets": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "basis_flat": np.zeros((6, 3), dtype=np.int32),
        "profile_offsets": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "profile_weights_flat": np.ones(3, dtype=float),
        "sg_offsets": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "sg_basis_flat": np.zeros(6, dtype=float),
        "diff_offsets": np.asarray([0, 4, 8, 12], dtype=np.int64),
        "diff_idx_flat": np.zeros(12, dtype=np.int64),
        "unique_diffs": np.zeros((1, 3), dtype=np.int32),
        "unique_inv_d2": np.zeros(1, dtype=float),
        "center_sg": np.zeros(3, dtype=float),
        "omega_center": np.zeros(3, dtype=float),
        "omega_bragg": np.zeros(3, dtype=float),
        "omega_delta": np.zeros(3, dtype=float),
        "sg_window": np.ones(3, dtype=float),
        "sg_domega": np.ones(3, dtype=float),
        "sg_dframe": np.ones(3, dtype=float),
        "sg_window_raw": np.ones(3, dtype=float),
        "sg_window_half_frames": np.ones(3, dtype=float),
        "sg_window_clipped": np.zeros(3, dtype=bool),
        "sg_jitter_sigma": np.zeros(3, dtype=float),
        "sg_jitter_d_inv": np.zeros(3, dtype=float),
        "sg_jitter_frame_fraction": np.zeros(3, dtype=float),
        "sg_jitter_kernel_points": np.ones(3, dtype=np.int32),
        "sg_profile_points_total": np.ones(3, dtype=np.int32),
        "sg_profile_weight_sum": np.ones(3, dtype=float),
        "sg_residual_max": np.zeros(3, dtype=float),
        "dose_delta_b": np.zeros(3, dtype=float),
    }
    assert row_plan_cache_validation_errors(valid_cache, indices, "expected", meta) == []

    stale_cache = dict(valid_cache)
    stale_cache["meta"] = dict(meta, n_basis=30)
    errors = row_plan_cache_validation_errors(stale_cache, indices, "expected", meta)
    assert "metadata mismatch: n_basis" in errors

    broken_cache = dict(valid_cache)
    broken_cache.pop("sg_basis_flat")
    broken_cache["basis_offsets"] = np.asarray([0, 1, 3, 5], dtype=np.int64)
    errors = row_plan_cache_validation_errors(broken_cache, indices, "expected", meta)
    assert any("missing arrays" in err for err in errors)
    assert "basis offsets do not contain direct and target beams for every row" in errors


def test_cached_static_hamiltonian_uses_frozen_dose_delta_b():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(dose_b_mode="linear_frame", dose_b_per_frame=99.0)
    port.data = type("DummyData", (), {"k": 2.0})()

    basis = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=int)
    diff_idx = np.asarray([0, 1, 2, 3], dtype=np.int64)
    unique_diffs = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0]], dtype=np.int32)
    unique_inv_d2 = np.asarray([0.0, 4.0, 4.0, 0.0], dtype=float)
    unique_ug = np.asarray([0.0, 1.0 + 0.0j, 1.0 + 0.0j, 0.0], dtype=np.complex128)

    H = port.cached_static_hamiltonian(
        basis,
        diff_idx,
        unique_diffs,
        unique_inv_d2,
        unique_ug,
        row_idx=0,
        absorb=False,
        dose_delta_b=2.0,
    )
    expected = np.exp(-2.0) / (2.0 * port.data.k)
    assert np.isclose(H[0, 1], expected)
    assert np.isclose(H[1, 0], expected)


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


def test_expm_multiply_backend_matches_eigensolver_on_hermitian_matrix():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    H = (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    port = object.__new__(XDSBlochPort)
    port.cfg = Config(compute_backend="scipy")
    port.thick = np.asarray([300.0, 600.0, 900.0], dtype=float)
    port.thick_w = np.asarray([0.2, 0.5, 0.3], dtype=float)

    reference_single = port.propagate_direct_to_target(H, 700.0)
    reference_average = port.propagate_direct_to_target_thickness_average(H)

    port.cfg = Config(compute_backend="scipy_expm_multiply")
    trial_single = port.propagate_direct_to_target(H, 700.0)
    trial_average = port.propagate_direct_to_target_thickness_average(H)

    assert np.isclose(trial_single, reference_single, rtol=1e-9, atol=1e-11)
    assert np.isclose(trial_average, reference_average, rtol=1e-9, atol=1e-11)


def test_real_embedding_generator_matches_complex_expm_column():
    rng = np.random.default_rng(321)
    raw = 0.001 * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    H = (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))
    thickness = 350.0

    embedded = _real_embedding_generator_batch(H[None, :, :], thickness, "float64")[0]
    real_col = expm(embedded)[:, 0]
    trial = real_col[1] + 1j * real_col[H.shape[0] + 1]

    complex_col = expm((2j * np.pi * thickness) * H)[:, 0]
    assert np.isclose(abs(trial) ** 2, abs(complex_col[1]) ** 2, rtol=1e-11, atol=1e-13)


def test_bethe_backend_is_exact_when_strong_block_contains_all_beams():
    rng = np.random.default_rng(1234)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    H = 0.001 * (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    port = object.__new__(XDSBlochPort)
    port.thick = np.asarray([300.0, 600.0, 900.0], dtype=float)
    port.thick_w = np.asarray([0.2, 0.5, 0.3], dtype=float)
    port.cfg = Config(compute_backend="scipy")
    reference = port.propagate_direct_to_target_thickness_average(H)

    port.cfg = Config(compute_backend="scipy_bethe", bethe_strong_count=H.shape[0])
    trial = port.propagate_direct_to_target_thickness_average(H)

    assert np.isclose(trial, reference, rtol=1e-12, atol=1e-14)


def test_bethe_variants_return_finite_effective_hamiltonians():
    rng = np.random.default_rng(4321)
    raw = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    H = 0.001 * (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    for variant in ["truncate", "schur0", "schur_target", "schur_mean"]:
        cfg = Config(bethe_strong_count=4, bethe_variant=variant, bethe_promote_denom=0.0)
        effective = promoted_bethe_effective_hamiltonian(H, cfg)
        assert effective.shape == (4, 4)
        assert np.all(np.isfinite(effective))


def test_padded_jax_backend_defaults_to_eight_beam_buckets():
    assert _jax_matrix_bucket_size(Config(compute_backend="jax_cpu"), "jax_cpu") == 0
    assert _jax_matrix_bucket_size(Config(compute_backend="jax_cpu_padded"), "jax_cpu_padded") == 8
    assert _jax_matrix_bucket_size(Config(compute_backend="jax_mps_real_expm_padded"), "jax_mps_real_expm_padded") == 8
    assert (
        _jax_matrix_bucket_size(
            Config(compute_backend="jax_cpu_padded", jax_matrix_bucket_size=16),
            "jax_cpu_padded",
        )
        == 16
    )


def test_stack_square_matrices_pads_with_uncoupled_zero_beams():
    a = np.eye(2, dtype=np.complex128)
    b = np.ones((3, 3), dtype=np.complex128)
    stack = _stack_square_matrices([a, b], pad_to=4)
    assert stack.shape == (2, 4, 4)
    assert np.allclose(stack[0, :2, :2], a)
    assert np.allclose(stack[0, 2:, :], 0.0)
    assert np.allclose(stack[0, :, 2:], 0.0)
    assert np.allclose(stack[1, :3, :3], b)
