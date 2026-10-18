"""Host/device packing preserves each profile node and pads only a final device tail."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from grond.bloch import backends, spectral_gpu
from grond.bloch.hamiltonian import ProfileHamiltonians
from grond.legacy.engine import spectra_stage


@pytest.fixture
def three_devices(monkeypatch):
    calls = {'solved_matrices': 0, 'device_groups': [], 'host_chunks': []}
    devices = tuple(SimpleNamespace(platform='gpu', id=i) for i in range(3))

    def eigh(matrices):
        calls['solved_matrices'] += len(matrices)
        return np.linalg.eigh(matrices)

    def pmap(function, *, devices):
        def mapped(shards):
            calls['device_groups'].append(tuple(device.id for device in devices))
            pieces = [function(shard) for shard in shards]
            return tuple(np.stack([piece[i] for piece in pieces]) for i in range(2))
        return mapped

    jax = SimpleNamespace(pmap=pmap)
    jnp = SimpleNamespace(linalg=SimpleNamespace(eigh=eigh), conj=np.conj)
    monkeypatch.setattr(spectral_gpu, '_jax_devices', lambda backend: devices)
    monkeypatch.setattr(spectral_gpu, '_jax_modules', lambda backend: (jax, jnp))
    monkeypatch.setattr(spectral_gpu, '_SOLVERS', {})
    return calls


@pytest.mark.parametrize('beams,expected', [(1000, 15), (2000, 3)])
def test_one_gib_host_buffer_feeds_three_devices_without_partial_middle_chunks(three_devices, beams, expected):
    count = spectral_gpu.spectral_host_chunk_size(beams, 540, 'jax_cuda', 0, 2**30)
    assert count == expected and count % 3 == 0
    assert 4 * count * beams**2 * np.dtype(complex).itemsize <= 2**30


def test_small_host_buffer_requires_explicit_device_reduction(three_devices):
    with pytest.raises(MemoryError, match='3 active devices'):
        spectral_gpu.spectral_host_chunk_size(2000, 540, 'jax_cuda', 0, 2**28)
    assert spectral_gpu.spectral_host_chunk_size(2000, 540, 'jax_cuda', 1, 2**28) == 1
    with pytest.raises(ValueError, match='nonnegative'):
        spectral_gpu.spectral_host_chunk_size(1000, 540, 'jax_cuda', -1, 2**30)


def profile(count, complex_phase=False):
    rng = np.random.default_rng(81)
    a = rng.normal(size=(6, 6)) * .0002
    static = (a + a.T).astype(complex)
    if complex_phase:
        phase = np.exp(.31j * np.arange(6))
        static = phase[:, None] * static * phase.conj()[None, :]
    excitation = rng.normal(size=(count, 6)) * .0005
    return ProfileHamiltonians(static, excitation)


@pytest.mark.parametrize('count,batch,solved', [(7, 6, 9), (15, 12, 15), (27, 12, 27)])
@pytest.mark.parametrize('complex_phase', [False, True])
def test_device_tail_is_smaller_than_nominal_batch_and_preserves_order(three_devices, count, batch, solved, complex_phase):
    matrices = np.asarray(profile(count, complex_phase))[::-1]
    values, coupling = spectral_gpu.hermitian_spectra(matrices, 'jax_cuda', batch)
    assert three_devices['solved_matrices'] == solved
    np.testing.assert_allclose(values, np.linalg.eigvalsh(matrices), rtol=1e-12, atol=1e-15)
    times = np.array([20., 180., 950.])
    got = abs(np.sum(coupling[:, :, None] * np.exp(2j*np.pi*values[:, :, None]*times), axis=1))**2
    expected = np.array([[abs(expm(2j*np.pi*t*matrix)[1, 0])**2 for t in times] for matrix in matrices])
    np.testing.assert_allclose(got, expected, rtol=2e-11, atol=1e-14)


def test_host_chunks_do_not_repeat_padding_and_donor_holes_keep_original_order(three_devices, monkeypatch):
    matrices = list(profile(31, True))  # Exercise the retained generic dense fallback.
    weights = np.linspace(.2, 1.1, 31)
    cfg = SimpleNamespace(compute_backend='jax_cuda', jax_batch_size=6)
    port = SimpleNamespace(cfg=cfg, absorption_enabled=lambda: False,
                           row_prediction_plan=lambda row: {'_dyn_matrices': matrices, '_profile_weights': weights})
    # Eight matrices fit by memory, but six form two complete three-device batches.
    monkeypatch.setattr(backends, '_jax_host_buffer_bytes', lambda: 4 * 8 * 6**2 * 16)
    original = spectral_gpu.hermitian_spectra

    def counted(batch, *args):
        three_devices['host_chunks'].append(len(batch))
        return original(batch, *args)

    monkeypatch.setattr(spectral_gpu, 'hermitian_spectra', counted)
    keys = spectra_stage.matrix_digests(matrices, False)
    donor_indices = set(range(0, 31, 3))
    donor = {keys[i]: spectra_stage.matrix_spectrum(matrices[i], False) for i in donor_indices}
    record = spectra_stage.row_spectrum(port, 17, donor)
    missing = 31 - len(donor_indices)
    assert record[8] == len(donor_indices)
    assert three_devices['host_chunks'] == [6, 6, 6, 2]
    assert three_devices['solved_matrices'] == missing
    assert all(group == (0, 1, 2) for group in three_devices['device_groups'][:-1])
    assert three_devices['device_groups'][-1] == (0, 1)
    np.testing.assert_array_equal(record[1], weights)
    times = np.array([30., 400.])
    for i, matrix in enumerate(matrices):
        amplitude = record[3][i] @ np.exp(2j*np.pi*record[2][i, :, None]*times)
        expected = np.array([abs(expm(2j*np.pi*t*matrix)[1, 0])**2 for t in times])
        np.testing.assert_allclose(abs(amplitude)**2, expected, rtol=2e-11, atol=1e-14)
