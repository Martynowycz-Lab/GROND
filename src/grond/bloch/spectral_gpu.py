"""Device eigensystems for full-matrix elastic and uniform-loss row spectra.

Only eigenvalues and incident-to-target couplings return to the host. General
absorptive matrices must use a separate complex spectral contract.
"""

from __future__ import annotations

import math

import numpy as np

from grond.bloch.backends import _elastic_matrix_dtype, _jax_devices, _jax_modules

_SOLVERS = {}


def spectral_host_chunk_size(n, n_rows, backend, batch_size, host_budget_bytes):
    """Bound dense host buffers and keep complete chunks divisible across active devices.

    Four complex buffers conservatively cover retained matrices and validation
    temporaries. Device solver workspaces and the source row plan are additional.
    """
    requested = int(batch_size or 0)
    if requested < 0:
        raise ValueError("spectral batch size must be nonnegative")
    devices = len(_jax_devices(backend))
    active = min(devices, int(n_rows), requested or devices)
    if int(n) < 1 or active < 1:
        raise ValueError("spectral chunks need a positive matrix size, row count and device count")
    capacity = int(host_budget_bytes) // (4 * int(n) * int(n) * np.dtype(complex).itemsize)
    if capacity < active:
        required = 4 * active * int(n) * int(n) * np.dtype(complex).itemsize
        raise MemoryError(
            f"spectral host budget cannot feed {active} active devices; need at least {required} bytes, "
            "increase GROND_JAX_HOST_BUFFER_GIB or explicitly reduce devices/batch size"
        )
    return min(int(n_rows), capacity // active * active)


def hermitian_spectra(matrices, backend: str, batch_size: int = 0):
    """Return eigenvalues and V[1,j] conj(V[0,j]) on the requested JAX devices.

    The caller supplies Hermitian matrices after separating any uniform loss.
    Batch size counts matrices across all visible devices. The default bounds
    input plus estimated solver workspace conservatively for large bases.
    """
    source = np.asarray(matrices)
    dtype = _elastic_matrix_dtype(source)
    n_rows, n, _ = source.shape
    if not n_rows:
        return np.empty((0, n)), np.empty((0, n), dtype=complex)
    norm = np.max(np.abs(source), axis=(-2, -1))
    error = np.max(np.abs(source - source.conj().transpose(0, 2, 1)), axis=(-2, -1))
    tolerance = 8 * np.finfo(source.real.dtype).eps * norm
    if np.any(error > tolerance):
        raise ValueError("Hermitian GPU spectra received a non-Hermitian matrix")
    H = np.asarray(source.real if dtype == np.float64 else source, dtype=dtype)
    jax, jnp = _jax_modules(backend)
    devices = _jax_devices(backend)
    requested = int(batch_size or 0)
    if requested < 0:
        raise ValueError("spectral batch size must be nonnegative")
    if not requested:
        per_device = min(8, max(1, (256 * 2**20) // (8 * n * n * H.dtype.itemsize)))
        requested = per_device * len(devices)
    devices = devices[:min(len(devices), n_rows, requested)]
    device_count = len(devices)
    chunk = max(device_count, requested // device_count * device_count)
    chunk = min(chunk, math.ceil(n_rows / device_count) * device_count)
    values = np.empty((n_rows, n), dtype=float)
    coupling = np.empty((n_rows, n), dtype=complex)
    for start in range(0, n_rows, chunk):
        batch = H[start:start + chunk]
        count = len(batch)
        # A smaller final batch needs only device-count padding, not a full nominal batch.
        per_device = math.ceil(count / device_count)
        solved_count = per_device * device_count
        if count < solved_count:
            batch = np.concatenate((batch, np.repeat(batch[-1:], solved_count - count, axis=0)))
        key = (backend, H.dtype.str, n, per_device, tuple(str(d) for d in devices))
        if key not in _SOLVERS:
            def solve(shard):
                eigenvalues, vectors = jnp.linalg.eigh(shard)
                return eigenvalues, vectors[:, 1, :] * jnp.conj(vectors[:, 0, :])

            _SOLVERS[key] = jax.pmap(solve, devices=devices)
        lam, c = _SOLVERS[key](batch.reshape(device_count, per_device, n, n))
        values[start:start + count] = np.asarray(lam).reshape(solved_count, n)[:count]
        coupling[start:start + count] = np.asarray(c).reshape(solved_count, n)[:count]
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coupling)):
        raise FloatingPointError("GPU spectra contain non-finite eigenvalues or couplings")
    return values, coupling
