"""Compact profile matrices preserve native precision, values and independent ownership."""
import numpy as np
import pytest

from grond.bloch import hamiltonian as H


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_lazy_profile_matches_independent_dense_construction(dtype):
    rng = np.random.default_rng(870)
    static = (rng.normal(size=(5, 5)) + 1j*rng.normal(size=(5, 5))).astype(dtype)
    excitation = rng.normal(size=(9, 5))
    expected = []
    for row in excitation:
        matrix = static.copy()
        matrix[np.diag_indices(5)] += row.astype(np.float64 if dtype == np.complex128 else np.float32)
        expected.append(matrix)
    expected = np.asarray(expected)
    profile = H.ProfileHamiltonians(static, excitation)
    assert profile.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(profile), expected)
    np.testing.assert_array_equal(profile[-1], expected[-1])
    np.testing.assert_array_equal(np.asarray(profile[1::2]), expected[1::2])
    np.testing.assert_array_equal(profile.diagonals(), np.diagonal(expected, axis1=-2, axis2=-1))
    static[:] = 0
    excitation[:] = 0
    modified = profile[0]
    modified[:] = 0
    np.testing.assert_array_equal(np.asarray(profile), expected)


def test_large_profile_builds_only_the_requested_matrix(monkeypatch):
    calls = []
    original = H.hamiltonian_with_excitation

    def counted(static, excitation):
        calls.append(len(excitation))
        return original(static, excitation)

    monkeypatch.setattr(H, 'hamiltonian_with_excitation', counted)
    profile = H.ProfileHamiltonians(np.zeros((600, 600), dtype=np.complex128), np.zeros((720, 600)))
    assert calls == []
    assert profile.storage_nbytes < 10*1024**2
    assert np.prod(profile.shape)*16 > 3.8*1024**3
    assert profile[719].shape == (600, 600)
    assert calls == [600]


def test_empty_slice_retains_its_matrix_shape():
    profile = H.ProfileHamiltonians(np.zeros((3, 3), dtype=np.complex128), np.zeros((5, 3)))
    assert np.asarray(profile[:0]).shape == (0, 3, 3)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_pets_compact_plan_preserves_absorption_without_dense_stack(monkeypatch, dtype):
    from types import SimpleNamespace

    from scipy.linalg import expm

    from grond.polish.crystals import compact_row_plan

    static = np.array([[0.004j, 0.002+0.001j], [0.002-0.001j, 0.004j]], dtype=dtype)
    excitation = np.array([[0.0, 0.003], [0.0, -0.008], [0.0, 0.01]])
    profile = H.ProfileHamiltonians(static, excitation)
    weights = np.array([0.2, 0.5, 0.3])
    dense = np.asarray(profile).astype(np.complex128)

    def propagate(matrix):
        return abs(expm(2j * np.pi * 7.0 * matrix)[1, 0]) ** 2

    expected = weights @ np.array([propagate(matrix) for matrix in dense])

    def forbid_stack(*args, **kwargs):
        raise AssertionError('PETS plan must not allocate a full profile stack')

    monkeypatch.setattr(H.ProfileHamiltonians, '__array__', forbid_stack)
    port = SimpleNamespace(
        data=SimpleNamespace(pets_exact_a_center=np.eye(3)),
        row_prediction_plan=lambda index: {
            'support_hkls': [[0, 0, 0], [1, 2, 3]],
            '_dyn_matrices': profile, '_profile_weights': weights,
        },
        propagate_direct_to_target_thickness_average=propagate,
        absorption_enabled=lambda: True,
        row_delta_b=lambda index: 0.5,
    )
    result = compact_row_plan(port, 0, predict=True)
    np.testing.assert_array_equal(result['sg'], np.diagonal(dense, axis1=1, axis2=2).real)
    np.testing.assert_allclose(result['native_intensity'], expected, rtol=1e-14)
    assert result['mu'] == float(static[0, 0].imag)
    assert result['dose_delta_b_A2'] == 0.5


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_hermitian_eigh_uses_the_real_solver_only_for_real_matrices(dtype):
    from scipy.linalg import eigh

    from grond.bloch.numerics import hermitian_eigh

    rng = np.random.default_rng(3)
    a = rng.normal(size=(40, 40))
    real = ((a + a.T) / 2).astype(dtype)  # centrosymmetric model: real potential coefficients
    lam, vec = hermitian_eigh(real)
    assert vec.dtype == real.dtype
    tol = 1e-4 if dtype == np.complex64 else 1e-11
    np.testing.assert_allclose(real @ vec, vec * lam, atol=tol)
    np.testing.assert_allclose(lam, eigh(real.astype(np.complex128), eigvals_only=True), atol=tol)

    b = rng.normal(size=(40, 40))
    chiral = (real + 1e-2j * (b - b.T) / 2).astype(dtype)
    for got, ref in zip(hermitian_eigh(chiral), eigh(chiral), strict=True):
        np.testing.assert_array_equal(got, ref)  # complex matrices take the unchanged complex solver
