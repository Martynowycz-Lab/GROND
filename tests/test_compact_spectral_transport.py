"""The compact profile representation reproduces each dense native matrix and every spectral guard."""

from types import SimpleNamespace

import numpy as np
import pytest

from grond.bloch import backends, spectral_gpu
from grond.bloch.hamiltonian import ProfileHamiltonians
from grond.legacy.engine import spectra_stage as stage
from tests.test_device_spectra import device_stub  # noqa: F401  (fixture)


def _port(matrices, weights, batch_size=2):
    cfg = SimpleNamespace(compute_backend="jax_cuda", jax_batch_size=batch_size)
    return SimpleNamespace(cfg=cfg, absorption_enabled=lambda: False,
                           row_prediction_plan=lambda row: {"_dyn_matrices": matrices, "_profile_weights": weights})


def _forbidden(*args, **kwargs):
    raise AssertionError("this call must not happen")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("complex_phase", [False, True])
@pytest.mark.parametrize("absorptive", [False, True])
def test_native_diagonal_rounding_and_hermitian_split_match_dense(dtype, complex_phase, absorptive):
    rng = np.random.default_rng(618)
    a = rng.normal(size=(7, 7))
    static = (a + a.T).astype(dtype)
    if complex_phase:
        phase = np.exp(0.34j * np.arange(7))
        static = (phase[:, None] * static * phase.conj()[None, :]).astype(dtype)
    static[np.diag_indices(7)] = np.linspace(1e5, 3e5, 7) + (2e-5j if absorptive else 0)
    excitation = -static.diagonal().real + rng.normal(scale=0.01, size=(9, 7))
    matrices = ProfileHamiltonians(static, excitation)
    indices = [8, 1, 5, 0]
    off, diagonals, h10, delta, mu = stage._compact_spectral_inputs(matrices, indices, absorptive)
    for j, i in enumerate(indices):
        native = np.asarray(matrices[i], dtype=complex)
        if absorptive:
            expected, expected_mu = stage.split_uniform_absorption(native)
        else:
            expected, expected_mu = 0.5 * (native + native.conj().T), 0.0
        assembled = off.copy()
        assembled[np.diag_indices(7)] = diagonals[j]
        np.testing.assert_array_equal(assembled, expected)
        assert h10 == float(abs(native[1, 0]) ** 2)
        assert delta[j] == float(np.real(native[1, 1] - native[0, 0]))
        assert mu == expected_mu


def test_native32_addition_is_not_replaced_by_double_addition():
    static = np.diag(np.array([1e8, 2e8], dtype=np.complex64))
    matrices = ProfileHamiltonians(static, [[1.0, 3.0], [5.0, 7.0]])
    _, diagonal, *_ = stage._compact_spectral_inputs(matrices, [0, 1], False)
    expected = np.array([np.diag(matrices[i]).real for i in range(2)], dtype=float)
    np.testing.assert_array_equal(diagonal, expected)
    double_sum = static.diagonal().real.astype(float)[None, :] + matrices.excitation.astype(float)
    assert not np.array_equal(diagonal, double_sum)


def test_uniform_loss_check_uses_each_matrix_scale():
    static = np.array([[0.0, 0.001 + 1e-7j], [0.001 + 1e-7j, 0.0]], complex)
    matrices = ProfileHamiltonians(static, [[100.0, 100.0], [0.0, 0.0]])
    stage.split_uniform_absorption(matrices[0])  # the relative tolerance accepts the high-norm matrix
    with pytest.raises(ValueError, match="non-uniform"):
        stage.split_uniform_absorption(matrices[1])
    with pytest.raises(ValueError, match="non-uniform"):
        stage._compact_spectral_inputs(matrices, [0, 1], True)


@pytest.mark.parametrize("where", ["static", "excitation"])
def test_nonfinite_compact_input_is_rejected(where):
    static = np.eye(3, dtype=complex)
    excitation = np.zeros((2, 3))
    if where == "static":
        static[0, 1] = np.nan
    else:
        excitation[1, 0] = np.inf
    matrices = ProfileHamiltonians(static, excitation)
    with pytest.raises(ValueError, match="finite"):
        stage._compact_spectral_inputs(matrices, [0, 1], False)


def test_compact_kernel_reuses_static_upload_and_keeps_order(device_stub, monkeypatch):  # noqa: F811
    rng = np.random.default_rng(16)
    a = rng.normal(size=(6, 6)) * 0.001
    static = (a + a.T).astype(complex)
    np.fill_diagonal(static, 0)
    diagonal = rng.normal(size=(7, 6)) * 0.001
    uploads = []

    def upload(jax, matrix, devices):
        uploads.append((matrix.copy(), len(devices)))
        return np.broadcast_to(matrix, (len(devices), *matrix.shape))

    monkeypatch.setattr(spectral_gpu, "_put_static_on_devices", upload)
    values, coupling = spectral_gpu.hermitian_spectra_from_static(static, diagonal, "jax_cuda", 4)
    assert len(uploads) == 1 and uploads[0][1] == 2
    dense = np.broadcast_to(static, (7, 6, 6)).copy()
    dense[:, np.arange(6), np.arange(6)] = diagonal
    expected_values, expected_vectors = np.linalg.eigh(dense.real)
    np.testing.assert_allclose(values, expected_values, atol=1e-15)
    np.testing.assert_allclose(coupling, expected_vectors[:, 1, :] * expected_vectors[:, 0, :], atol=1e-14)
    assert device_stub["matrices"] == 8  # exactly one padded slot in the final device batch

    changed = static.copy()
    changed[0, 1] += 0.0003
    changed[1, 0] += 0.0003
    updated, _ = spectral_gpu.hermitian_spectra_from_static(changed, diagonal, "jax_cuda", 4)
    assert len(uploads) == 2
    dense[:, 0, 1] += 0.0003
    dense[:, 1, 0] += 0.0003
    np.testing.assert_allclose(updated, np.linalg.eigvalsh(dense.real), atol=1e-15)


def test_compact_real_gate_does_not_hide_low_norm_matrix(device_stub):  # noqa: F811
    static = np.array([[0.0, 1e-4 + 1e-18j], [1e-4 - 1e-18j, 0.0]], complex)
    diagonal = np.array([[100.0, 100.0], [0.0, 0.0]])
    spectral_gpu.hermitian_spectra_from_static(static, diagonal, "jax_cuda", 2)
    assert set(device_stub["dtypes"]) == {np.dtype("complex128")}


def test_profile_path_does_not_materialize_dense_host_matrices(device_stub, monkeypatch):  # noqa: F811
    static = np.array([[0.0, 0.001], [0.001, 0.0]], complex)
    matrices = ProfileHamiltonians(static, np.array([[0.0, v] for v in np.linspace(-0.01, 0.01, 5)]))
    monkeypatch.setattr(ProfileHamiltonians, "__getitem__", _forbidden)
    result = stage.row_spectrum(_port(matrices, np.ones(5)), 7)
    assert result[2].shape == (5, 2)


def test_complete_donor_never_uploads_or_builds_a_matrix(device_stub, monkeypatch):  # noqa: F811
    static = np.array([[0.0, 0.001], [0.001, 0.0]], complex)
    matrices = ProfileHamiltonians(static, np.array([[0.0, v] for v in [0.01, 0.02, 0.03]]))
    keys = stage.matrix_digests(matrices, False)
    donor = {key: stage.matrix_spectrum(matrices[i], False) for i, key in enumerate(keys)}
    monkeypatch.setattr(ProfileHamiltonians, "__getitem__", _forbidden)
    monkeypatch.setattr(spectral_gpu, "hermitian_spectra_from_static", _forbidden)
    result = stage.row_spectrum(_port(matrices, np.ones(3)), 7, donor)
    assert result[8] == 3 and not device_stub["dtypes"]


def test_partial_donor_preparation_does_not_build_all_profile_diagonals(monkeypatch):
    matrices = ProfileHamiltonians(np.eye(3, dtype=np.complex64), np.zeros((10000, 3), dtype=np.float32))
    indices = [9999, 7]
    expected = np.array([np.diag(matrices[i]) for i in indices], dtype=complex)
    # Enough memory for the static buffers and the selected points, not for ten thousand diagonals.
    monkeypatch.setattr(backends, "_jax_host_buffer_bytes", lambda: (8 * 3 * 3 + 4 * len(indices) * 3) * 16)
    monkeypatch.setattr(ProfileHamiltonians, "diagonals", _forbidden)
    _, diagonal, *_ = stage._compact_spectral_inputs(matrices, indices, False)
    np.testing.assert_array_equal(diagonal, expected.real)
