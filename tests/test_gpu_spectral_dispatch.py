"""Accelerator dispatch and row-order contracts, checked against independent CPU exponentials.

The NumPy-backed JAX stub exercises dispatch, padding and cache ownership without needing a local GPU. Real GPU
parity and compiled cuSOLVER execution remain separate required campaign checks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from grond.bloch import backends as B


@pytest.fixture
def jax_stub(monkeypatch):
    calls = {"compiled": [], "eigh_dtypes": [], "eig": []}

    def eigh(matrices):
        calls["eigh_dtypes"].append(matrices.dtype)
        return np.linalg.eigh(matrices)

    def general_eig(matrices, **kwargs):
        calls["eig"].append(kwargs)
        return np.linalg.eig(matrices)

    def forbidden_svd_condition(*args, **kwargs):
        raise AssertionError("general GPU propagation must use the inverse-residual bound, not an extra SVD")

    def pmap(function, *, in_axes, devices):
        assert in_axes == (0, None, None)
        calls["compiled"].append(tuple(device.platform for device in devices))

        def mapped(shards, thickness, weights):
            assert len(shards) == len(devices)
            return np.stack([function(shard, thickness, weights) for shard in shards])

        return mapped

    jnp = SimpleNamespace(**{name: getattr(np, name) for name in (
        "pi", "exp", "sum", "real", "abs", "conj", "isfinite", "maximum", "finfo", "float64", "where", "nan", "eye", "inf"
    )})
    jnp.linalg = SimpleNamespace(eigh=eigh, eig=np.linalg.eig, inv=np.linalg.inv,
                                 cond=forbidden_svd_condition, norm=np.linalg.norm)
    jax = SimpleNamespace(pmap=pmap, lax=SimpleNamespace(linalg=SimpleNamespace(
        eig=general_eig, EigImplementation=SimpleNamespace(CUSOLVER="explicit-cusolver"))))
    devices = tuple(SimpleNamespace(platform="gpu", id=i) for i in range(2))
    monkeypatch.setattr(B, "_jax_modules", lambda backend: (jax, jnp))
    monkeypatch.setattr(B, "_jax_devices", lambda backend: devices)
    monkeypatch.setattr(B, "_JAX_PROPAGATORS", {})
    return calls, jax, devices


def reference(matrices, thickness, weights):
    return np.array([sum(w * abs(expm(2j * np.pi * t * matrix)[1, 0])**2
                         for t, w in zip(thickness, weights, strict=True)) for matrix in matrices])


def elastic_batch(count=5):
    rng = np.random.default_rng(58)
    matrices = rng.normal(size=(count, 6, 6)) * 0.0003
    return (matrices + matrices.swapaxes(-2, -1)).astype(np.complex128)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_real_threshold_matches_cpu_criterion_per_matrix(dtype):
    matrices = np.tile(np.eye(2, dtype=dtype), (2, 1, 1))
    eps = np.finfo(matrices.real.dtype).eps
    matrices[0, 0, 1] = 4j * eps
    matrices[0, 1, 0] = -4j * eps
    assert B._jax_eigensolver_input(matrices, absorption=False).dtype == np.float64
    matrices[0, 0, 1] = 9j * eps
    matrices[0, 1, 0] = -9j * eps
    # An unrelated large real matrix must not loosen the first matrix's threshold.
    matrices[1] *= 1e8
    result = B._jax_eigensolver_input(matrices, absorption=False)
    assert result.dtype == np.complex128
    np.testing.assert_array_equal(result, matrices)


def test_absorption_never_discards_imaginary_entries():
    matrices = elastic_batch(1)
    matrices[0] += 1e-20j * np.eye(6)
    result = B._jax_eigensolver_input(matrices, absorption=True)
    assert result.dtype == np.complex128
    np.testing.assert_array_equal(result, matrices)


def test_real_elastic_dispatch_keeps_order_padding_and_thickness_average(jax_stub):
    calls, _, _ = jax_stub
    matrices = elastic_batch()
    thickness, weights = np.array([0., 100., 900., 5000.]), np.array([0.1, 0.2, 0.3, 0.4])
    actual = B._jax_propagate_batch(matrices, thickness, weights, "jax_cuda", False, batch_size=4)
    np.testing.assert_allclose(actual, reference(matrices, thickness, weights), rtol=1e-12, atol=1e-14)
    assert all(dtype == np.float64 for dtype in calls["eigh_dtypes"])
    assert calls["compiled"] == [("gpu", "gpu")]


def test_complex_elastic_retains_phase_and_has_separate_jit_cache(jax_stub):
    calls, _, _ = jax_stub
    matrices = elastic_batch(1)
    thickness, weights = np.array([500., 1500.]), np.array([0.4, 0.6])
    real_values = B._jax_propagate_batch(matrices, thickness, weights, "jax_cuda", False, 1)
    # This diagonal unitary phase change represents an origin change: eigenvalues and intensities are invariant,
    # but discarding its complex coupling phases gives a different physical matrix.
    phase = np.exp(1j * np.linspace(0.0, 1.3, 6))
    translated = matrices * phase[None, :, None] * phase.conj()[None, None, :]
    values = B._jax_propagate_batch(translated, thickness, weights, "jax_cuda", False, 1)
    np.testing.assert_allclose(values, reference(translated, thickness, weights), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(values, real_values, rtol=1e-12, atol=1e-14)
    assert calls["eigh_dtypes"] == [np.dtype("float64"), np.dtype("complex128")]
    assert len(B._JAX_PROPAGATORS) == 2


def test_gpu_absorption_uses_explicit_general_cusolver(jax_stub):
    calls, _, _ = jax_stub
    matrices = elastic_batch(2)
    rng = np.random.default_rng(19)
    loss_factor = rng.normal(size=(6, 6)) * 0.0001
    loss = loss_factor @ loss_factor.T + 2e-5 * np.eye(6)
    matrices += 1j * loss
    thickness, weights = np.array([0., 500., 1200.]), np.array([0.1, 0.4, 0.5])
    values = B._jax_propagate_batch(matrices, thickness, weights, "jax_cuda", True, 2)
    np.testing.assert_allclose(values, reference(matrices, thickness, weights), rtol=1e-12, atol=1e-14)
    assert not calls["eigh_dtypes"]
    assert calls["eig"] and all(call == {
        "compute_left_eigenvectors": False, "compute_right_eigenvectors": True,
        "implementation": "explicit-cusolver",
    } for call in calls["eig"])


def test_unsupported_gpu_general_eig_fails_before_compilation(jax_stub):
    calls, jax, _ = jax_stub
    del jax.lax.linalg.EigImplementation
    with pytest.raises(NotImplementedError, match="explicit cuSOLVER"):
        B._jax_propagate_batch(elastic_batch(1), [300.], [1.], "jax_cuda", True, 1)
    assert not calls["compiled"]
    assert B._jax_general_eig_implementation(jax, [SimpleNamespace(platform="cpu")]) is None


def test_nonfinite_and_invalid_matrices_are_rejected():
    with pytest.raises(ValueError, match="direct and target"):
        B._jax_eigensolver_input(np.zeros((1, 1, 1)), absorption=False)
    matrices = elastic_batch(1)
    matrices[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        B._jax_eigensolver_input(matrices, absorption=False)


@pytest.mark.parametrize("condition", [1.0, 1e4, 1e8, 0.99e10, 1.01e10, 1e12])
def test_inverse_condition_bound_covers_nearly_parallel_eigenvectors(condition):
    # Two unit-length columns; singular values and hence cond_2 are fixed by their included angle.
    cosine = (condition**2 - 1.0) / (condition**2 + 1.0)
    sine = 2.0 * condition / (condition**2 + 1.0)
    vectors = np.array([[1.0, cosine], [0.0, sine]], dtype=complex)
    bound = B._inverse_condition_upper_bound(vectors, np.linalg.inv(vectors))
    assert bound >= np.linalg.cond(vectors)
    if condition > 1e10:
        assert bound > 1e10


def test_condition_bound_corrects_an_inaccurate_inverse_at_the_acceptance_threshold():
    condition = 1.005e10
    vectors = np.diag([1.0, 1.0 / condition]).astype(complex)
    approximate = np.diag([1.0, 0.99 * condition]).astype(complex)
    assert np.linalg.norm(vectors, "fro") * np.linalg.norm(approximate, "fro") < 1e10
    assert B._inverse_condition_upper_bound(vectors, approximate) > 1e10


def test_uncertain_inverse_bound_rejects_without_svd():
    matrices = np.array([np.eye(4), np.diag([1.0, 0.1, 0.001, 1e-8])], dtype=complex)
    inverses = np.linalg.inv(matrices)
    bounds = B._inverse_condition_upper_bound(matrices, inverses)
    assert np.all(bounds >= np.linalg.cond(matrices))
    assert np.isinf(B._inverse_condition_upper_bound(matrices[0], np.zeros((4, 4), dtype=complex)))
    # Near-singular inverse estimates whose roundoff allowance alone reaches one also fail closed.
    uncertain = np.diag([1.0, 1e-18]).astype(complex)
    assert np.isinf(B._inverse_condition_upper_bound(uncertain, np.linalg.inv(uncertain)))
