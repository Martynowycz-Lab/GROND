#!/usr/bin/env python3
"""Standalone XDS Bloch-wave port.

This keeps the audited row-contract conventions explicit in the XDS-facing
calculator:

* signed observation HKLs are preserved;
* off-diagonal coupling uses U(h_j - h_i), not U(h_i - h_j);
* the rocking path is sampled in target Sg and converted back to XDS omega;
* all basis-beam Sg values are recomputed at each omega;
* Born and Bloch predictions are scaled and scored on identical row masks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import gemmi
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig, eigh, expm
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

try:
    from grond.paths import runtime_cache_dir
except ImportError:  # Direct script execution from the legacy engine directory.

    def runtime_cache_dir(*parts: str) -> Path:
        override = os.environ.get("GROND_CACHE_DIR", "").strip()
        root = Path(override).expanduser() if override else Path.home() / ".cache" / "grond"
        return root.joinpath(*parts)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_XDS_ASCII = Path("XDS_ASCII.HKL")
DEFAULT_RES = Path("model.res")

_WORKER_PORT = None
_JAX_MODULES = {}
_JAX_PROPAGATORS = {}
_JAX_REAL_EXPM_PROPAGATORS = {}
JAX_BACKENDS = {"jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_gpu"}
JAX_PADDED_BACKENDS = {
    "jax_padded",
    "jax_auto_padded",
    "jax_cpu_padded",
    "jax_mps_padded",
    "jax_cuda_padded",
    "jax_gpu_padded",
}
JAX_REAL_EXPM_BACKENDS = {"jax_cpu_real_expm", "jax_mps_real_expm"}
JAX_REAL_EXPM_PADDED_BACKENDS = {"jax_mps_real_expm_padded"}
ALL_JAX_BACKENDS = JAX_BACKENDS | JAX_PADDED_BACKENDS | JAX_REAL_EXPM_BACKENDS | JAX_REAL_EXPM_PADDED_BACKENDS
COMPUTE_BACKENDS = [
    "scipy",
    "scipy_bethe",
    "scipy_expm",
    "scipy_expm_multiply",
    "jax",
    "jax_auto",
    "jax_cpu",
    "jax_mps",
    "jax_cuda",
    "jax_padded",
    "jax_auto_padded",
    "jax_cpu_padded",
    "jax_mps_padded",
    "jax_cuda_padded",
    "jax_cpu_real_expm",
    "jax_mps_real_expm",
    "jax_mps_real_expm_padded",
]
COMPUTE_BACKEND_HELP = ", ".join(COMPUTE_BACKENDS)
ROW_PLAN_CACHE_SCHEMA = 6
UG_CACHE_SCHEMA = 4
# Account only for coordinate text rounding, never SHELXL's much wider SPEC
# constraint-search radius. This is used consistently for occupancy conversion
# and structure-factor expansion (including six-decimal thirds).
SITE_SYMMETRY_TOLERANCE_FRACTIONAL = 2e-6
ELECTRON_COMPTON_WAVELENGTH_A = 0.024263102
DEFAULT_MAX_DENSE_UG_GIB = 2.0
# The final complex128 cube is 16 bytes per voxel. Construction also holds
# several real and complex vectorized work arrays at once; measured peaks are
# about 170 bytes per voxel for the anisotropic path. These bounds leave
# deliberate headroom for allocator and decompression overhead.
DENSE_UG_BUILD_BYTES_PER_VOXEL = 256
DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL = 64


class LocalSgCrossingError(ValueError):
    """A requested target Sg has no crossing inside the local geometry bound."""


class InfeasibleSgJitterProfileError(ValueError):
    """An active Sg-jitter candidate requests an unreachable quadrature point."""


class NonPassiveAbsorptionError(ValueError):
    """An optical-potential candidate would amplify some incident state.

    ``details`` is JSON-compatible and survives multiprocessing exception
    transfer, so an optional grid candidate can be rejected with evidence.
    """

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = dict(details or {})


def validate_passive_absorption(H, *, mode="unknown", row_index=None):
    """Require nonnegative loss for exp(2*pi*i*H*t), without changing H.

    With H=A+iB, d||psi||^2/dt=-4*pi*psi^dagger*B*psi. Gershgorin
    bounds avoid another eigensolve when positive diagonal loss suffices.
    This check is performed on the static matrix once per row, because adding
    real excitation errors at the rocking points cannot change B.
    """

    matrix = np.asarray(H, dtype=np.complex128)
    loss = (matrix - matrix.conj().T) / (2j)
    magnitude = float(np.linalg.norm(loss, ord=np.inf))
    tolerance = 1e-12 + 1e-10 * magnitude
    diagonal = np.real(np.diag(loss))
    radii = np.sum(np.abs(loss), axis=1) - np.abs(np.diag(loss))
    bound = float(np.min(diagonal - radii))
    if bound >= -tolerance:
        return
    minimum = float(np.linalg.eigvalsh(loss)[0])
    if minimum < -tolerance:
        details = {
            "reason": "nonpassive_absorption",
            "absorption_mode": str(mode),
            "row_index": None if row_index is None else int(row_index),
            "basis_size": int(len(matrix)),
            "minimum_loss_eigenvalue_inv_A": minimum,
            "minimum_additional_uniform_mu_inv_A": -minimum,
            "tolerance_inv_A": tolerance,
        }
        raise NonPassiveAbsorptionError(
            f"absorption candidate {mode!r} is not passive: minimum loss eigenvalue "
            f"{minimum:.6g} 1/A. Use an elastic candidate or a physically supported "
            "positive absorptive potential; increasing a fit parameter merely to "
            "pass this check is not an absorption model.",
            details=details,
        )


def dense_ug_memory_plan(h_lim: int, k_lim: int, l_lim: int) -> dict[str, int | list[int]]:
    """Estimate dense U_g grid storage and peak working memory before allocation."""

    limits = tuple(int(value) for value in (h_lim, k_lim, l_lim))
    if any(value < 0 for value in limits):
        raise ValueError(f"dense U_g limits must be nonnegative, got {limits}")
    shape = tuple(2 * value + 1 for value in limits)
    voxels = math.prod(shape)
    return {
        "shape": list(shape),
        "voxels": int(voxels),
        "cube_bytes": int(voxels * np.dtype(np.complex128).itemsize),
        "estimated_cache_load_peak_bytes": int(voxels * DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL),
        "estimated_build_peak_bytes": int(voxels * DENSE_UG_BUILD_BYTES_PER_VOXEL),
    }


def dense_ug_memory_limit_bytes() -> int:
    """Return the dense U_g working-memory ceiling, with an explicit env override."""

    raw = os.environ.get("GROND_MAX_DENSE_UG_GIB", str(DEFAULT_MAX_DENSE_UG_GIB)).strip()
    try:
        gib = float(raw)
    except ValueError as exc:
        raise ValueError("GROND_MAX_DENSE_UG_GIB must be a positive finite number") from exc
    if not math.isfinite(gib) or gib <= 0.0:
        raise ValueError("GROND_MAX_DENSE_UG_GIB must be a positive finite number")
    return int(gib * (1024**3))


def enforce_dense_ug_memory_limit(
    plan: dict[str, int | list[int]], *, operation: str
) -> dict[str, int | str | list[int]]:
    """Fail before a dense U_g allocation whose estimated peak exceeds the limit."""

    estimate_key = {
        "build": "estimated_build_peak_bytes",
        "cache load": "estimated_cache_load_peak_bytes",
    }.get(operation)
    if estimate_key is None:
        raise ValueError(f"unknown dense U_g memory operation {operation!r}")
    estimated = int(plan[estimate_key])
    limit = dense_ug_memory_limit_bytes()
    status: dict[str, int | str | list[int]] = {
        **plan,
        "operation": operation,
        "estimated_peak_bytes": estimated,
        "limit_bytes": limit,
    }
    if estimated > limit:
        shape = " x ".join(str(value) for value in plan["shape"])
        gib = 1024**3
        raise MemoryError(
            f"Dense U_g {operation} refused before allocation: grid {shape} "
            f"({int(plan['voxels']):,} voxels, {int(plan['cube_bytes']) / gib:.2f} GiB stored) "
            f"has an estimated {estimated / gib:.2f} GiB peak, above the {limit / gib:.2f} GiB "
            "safety limit. Increase ug_d_min to reduce the reciprocal grid, or set "
            "GROND_MAX_DENSE_UG_GIB to a larger value only after confirming available memory."
        )
    return status


def _init_prediction_worker(port):
    global _WORKER_PORT
    _WORKER_PORT = port


def _jax_backend_base(backend: str) -> str:
    """Return the accelerator selector for a possibly padded JAX backend name."""

    key = str(backend or "jax").lower()
    if key.endswith("_padded"):
        key = key[: -len("_padded")]
    if key.endswith("_real_expm"):
        key = key[: -len("_real_expm")]
    if key == "jax_auto":
        key = "jax"
    if key == "jax_gpu":
        key = "jax_cuda"
    return key


def _jax_backend_uses_matrix_padding(backend: str) -> bool:
    """Return whether the backend should bucket/pad matrix dimensions for JAX."""

    return str(backend or "").lower() in (JAX_PADDED_BACKENDS | JAX_REAL_EXPM_PADDED_BACKENDS)


def _jax_backend_uses_real_expm(backend: str) -> bool:
    """Return whether the backend uses the experimental real embedded expm path."""

    return str(backend or "").lower() in (JAX_REAL_EXPM_BACKENDS | JAX_REAL_EXPM_PADDED_BACKENDS)


def _jax_matrix_bucket_size(cfg: Config | None, backend: str) -> int:
    """Return the matrix-dimension bucket for padded JAX backends.

    Exact-size JAX grouping is still the default for ordinary JAX backends.
    The padded backends default to multiples of 8 because representative
    large-support benchmarks showed that this keeps compilation count low
    without paying too much extra eigensolver work from full max-size padding.
    """

    requested = int(getattr(cfg, "jax_matrix_bucket_size", 0) or 0) if cfg is not None else 0
    if requested > 0:
        return requested
    return 8 if _jax_backend_uses_matrix_padding(backend) else 0


def _jax_modules(backend: str):
    """Import JAX lazily for optional experimental Bloch propagation backends."""

    key = _jax_backend_base(backend)
    if key in _JAX_MODULES:
        return _JAX_MODULES[key]
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if key == "jax_cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif key == "jax_mps":
        os.environ.setdefault("JAX_PLATFORMS", "mps")
    elif key in {"jax_cuda", "jax_gpu"}:
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except Exception as exc:
        raise NotImplementedError(
            f"compute_backend={backend!r} requested JAX, but JAX could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _JAX_MODULES[key] = (jax, jnp)
    return _JAX_MODULES[key]


def _jax_module_key(backend: str) -> str:
    return _jax_backend_base(backend)


def _stack_square_matrices(matrices, pad_to: int | None = None) -> np.ndarray:
    """Stack square complex matrices, padding smaller matrices with zero beams."""

    matrices = list(matrices)
    if not matrices:
        return np.zeros((0, 0, 0), dtype=np.complex128)
    sizes = [int(np.asarray(H).shape[0]) for H in matrices]
    n = int(pad_to or max(sizes))
    out = np.zeros((len(matrices), n, n), dtype=np.complex128)
    for i, H in enumerate(matrices):
        H = np.asarray(H, dtype=np.complex128)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"expected square matrix, got shape {H.shape}")
        if H.shape[0] > n:
            raise ValueError(f"matrix dimension {H.shape[0]} exceeds requested pad_to={n}")
        out[i, : H.shape[0], : H.shape[1]] = H
    return out


def _auto_jax_batch_size(n: int, n_devices: int) -> int:
    """Conservative chunk size for batched complex eigensolvers."""

    n = max(1, int(n))
    n_devices = max(1, int(n_devices))
    if n <= 48:
        base = 1536
    elif n <= 80:
        base = 768
    elif n <= 128:
        base = 192
    elif n <= 192:
        base = 96
    elif n <= 256:
        base = 48
    else:
        base = 24
    return max(n_devices, int(math.ceil(base / n_devices)) * n_devices)


def _jax_real_expm_order() -> int:
    """Fixed Taylor order for the experimental real-embedded JAX expm backend."""

    return max(4, int(os.environ.get("GROND_JAX_REAL_EXPM_ORDER", "24")))


def _jax_real_expm_theta() -> float:
    """Scaling threshold for the experimental real-embedded JAX expm backend."""

    return max(1e-6, float(os.environ.get("GROND_JAX_REAL_EXPM_THETA", "0.5")))


def _jax_real_dtype_name(backend: str) -> str:
    """Use float32 on MPS because current JAX-Metal rejects float64."""

    return "float32" if _jax_backend_base(backend) == "jax_mps" else "float64"


def _real_embedding_generator_batch(H_batch: np.ndarray, thickness: float, dtype_name: str) -> np.ndarray:
    """Return real matrices for exp((2*pi*i*thickness)*H).

    For K = A + iB, the block matrix [[A, -B], [B, A]] propagates the real and
    imaginary parts of the complex wavefunction without asking the accelerator
    for complex eigensolvers.  This is an experimental Apple-Metal route, not a
    replacement for the SciPy eigensolver reference.
    """

    H_batch = np.asarray(H_batch, dtype=np.complex128)
    if H_batch.ndim != 3 or H_batch.shape[1] != H_batch.shape[2]:
        raise ValueError(f"expected a stack of square matrices, got shape {H_batch.shape}")
    n_batch, n, _ = H_batch.shape
    dtype = np.float32 if dtype_name == "float32" else np.float64
    factor = 2.0 * math.pi * float(thickness)
    real_part = (-factor * H_batch.imag).astype(dtype, copy=False)
    imag_part = (factor * H_batch.real).astype(dtype, copy=False)
    out = np.zeros((n_batch, 2 * n, 2 * n), dtype=dtype)
    out[:, :n, :n] = real_part
    out[:, :n, n:] = -imag_part
    out[:, n:, :n] = imag_part
    out[:, n:, n:] = real_part
    return out


def _jax_real_expm_scale_steps(real_batch: np.ndarray) -> int:
    """Choose conservative scaling steps for a fixed Taylor expm approximation."""

    if real_batch.size == 0:
        return 0
    norm_inf = float(np.max(np.sum(np.abs(real_batch), axis=2)))
    if not np.isfinite(norm_inf) or norm_inf <= 0.0:
        return 0
    return max(0, int(math.ceil(math.log2(norm_inf / _jax_real_expm_theta()))))


def _jax_real_expm_column0_batch(real_batch, backend: str, scale_steps: int, order: int, batch_size: int = 0):
    """Evaluate target-beam intensity using real scaling/squaring expm batches."""

    key = _jax_module_key(backend)
    jax, jnp = _jax_modules(key)
    devices = tuple(jax.devices())
    if not devices:
        raise NotImplementedError(f"compute_backend={backend!r} found no JAX devices")
    real_batch = np.asarray(real_batch)
    if real_batch.ndim != 3 or real_batch.shape[1] != real_batch.shape[2]:
        raise ValueError(f"expected a stack of square real matrices, got shape {real_batch.shape}")
    n_real = int(real_batch.shape[0])
    if n_real == 0:
        return np.zeros(0, dtype=float)
    n2 = int(real_batch.shape[1])
    if n2 < 4 or n2 % 2:
        raise ValueError(f"real embedded matrix dimension must be even and at least 4, got {n2}")
    dtype_name = "float32" if real_batch.dtype == np.float32 else "float64"
    cache_key = (key, "real_expm", n2, int(scale_steps), int(order), dtype_name)

    if cache_key not in _JAX_REAL_EXPM_PROPAGATORS:
        dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
        inv_scale = dtype(1.0 / float(2**int(scale_steps)))

        def _eval(Rs):
            A = Rs * inv_scale
            eye = jnp.eye(n2, dtype=dtype)
            term = jnp.broadcast_to(eye, Rs.shape)
            result = term
            for k in range(1, int(order) + 1):
                term = jnp.matmul(term, A) / dtype(k)
                result = result + term
            for _ in range(int(scale_steps)):
                result = jnp.matmul(result, result)
            n = n2 // 2
            real_target = result[:, 1, 0]
            imag_target = result[:, n + 1, 0]
            return real_target * real_target + imag_target * imag_target

        _JAX_REAL_EXPM_PROPAGATORS[cache_key] = jax.jit(_eval)

    fn = _JAX_REAL_EXPM_PROPAGATORS[cache_key]
    chunk = int(batch_size) if int(batch_size or 0) > 0 else _auto_jax_batch_size(n2 // 2, len(devices))
    chunk = max(1, int(chunk))
    out = np.zeros(n_real, dtype=float)
    for start in range(0, n_real, chunk):
        sub = real_batch[start : start + chunk]
        vals = np.asarray(fn(jnp.asarray(sub, dtype=jnp.float32 if dtype_name == "float32" else jnp.float64)))
        out[start : start + len(sub)] = vals[: len(sub)]
    return out


def _jax_real_expm_propagate_batch(H_batch, thick, thick_w, backend: str, batch_size: int = 0):
    """Experimental real-embedded JAX matrix exponential propagation."""

    H_batch = np.asarray(H_batch, dtype=np.complex128)
    if H_batch.ndim != 3 or H_batch.shape[1] != H_batch.shape[2]:
        raise ValueError(f"expected a stack of square matrices, got shape {H_batch.shape}")
    out = np.zeros(int(H_batch.shape[0]), dtype=float)
    dtype_name = _jax_real_dtype_name(backend)
    order = _jax_real_expm_order()
    for t, w in zip(np.asarray(thick, dtype=float), np.asarray(thick_w, dtype=float), strict=True):
        real_batch = _real_embedding_generator_batch(H_batch, float(t), dtype_name)
        scale_steps = _jax_real_expm_scale_steps(real_batch)
        vals = _jax_real_expm_column0_batch(real_batch, backend, scale_steps, order, batch_size)
        out += float(w) * vals
    return out


def _jax_propagate_batch(H_batch, thick, thick_w, backend: str, absorption: bool, batch_size: int = 0):
    """Return thickness-averaged direct-to-target intensities for a matrix batch.

    The input matrices must all have the same dimension.  Rows are split across
    all visible JAX devices with pmap; each device evaluates a batched eigensolver
    over its shard.  This is intentionally isolated from the SciPy reference path
    so a failing accelerator cannot silently alter production calculations.
    """

    if _jax_backend_uses_real_expm(backend):
        return _jax_real_expm_propagate_batch(H_batch, thick, thick_w, backend, batch_size=batch_size)

    key = _jax_module_key(backend)
    jax, jnp = _jax_modules(key)
    devices = tuple(jax.devices())
    if not devices:
        raise NotImplementedError(f"compute_backend={backend!r} found no JAX devices")
    H_batch = np.asarray(H_batch, dtype=np.complex128)
    if H_batch.ndim != 3 or H_batch.shape[1] != H_batch.shape[2]:
        raise ValueError(f"expected a stack of square matrices, got shape {H_batch.shape}")
    n_real = int(H_batch.shape[0])
    if n_real == 0:
        return np.zeros(0, dtype=float)
    n_devices = len(devices)
    n = int(H_batch.shape[1])
    chunk = int(batch_size) if int(batch_size or 0) > 0 else _auto_jax_batch_size(n, n_devices)
    chunk = max(n_devices, int(math.ceil(chunk / n_devices)) * n_devices)
    per_device = max(1, chunk // n_devices)
    cache_key = (key, bool(absorption), n, len(thick), n_devices, per_device, tuple(str(d) for d in devices))

    if cache_key not in _JAX_PROPAGATORS:
        if absorption:
            def _eval_shard(Hs, tvals, wvals):
                vals, vecs = jnp.linalg.eig(Hs)
                inv_vecs = jnp.linalg.inv(vecs)
                coupling = vecs[:, 1, :] * inv_vecs[:, :, 0]
                phase = jnp.exp((2j * jnp.pi) * vals[:, :, None] * tvals[None, None, :])
                amp = jnp.sum(coupling[:, :, None] * phase, axis=1)
                per_t = jnp.real(jnp.abs(amp) ** 2)
                return jnp.sum(per_t * wvals[None, :], axis=1)
        else:
            def _eval_shard(Hs, tvals, wvals):
                vals, vecs = jnp.linalg.eigh(Hs)
                coupling = vecs[:, 1, :] * jnp.conj(vecs[:, 0, :])
                phase = jnp.exp((2j * jnp.pi) * vals[:, :, None] * tvals[None, None, :])
                amp = jnp.sum(coupling[:, :, None] * phase, axis=1)
                per_t = jnp.real(jnp.abs(amp) ** 2)
                return jnp.sum(per_t * wvals[None, :], axis=1)

        _JAX_PROPAGATORS[cache_key] = jax.pmap(_eval_shard, in_axes=(0, None, None), devices=devices)

    fn = _JAX_PROPAGATORS[cache_key]
    tvals = jnp.asarray(np.asarray(thick, dtype=float), dtype=jnp.float64)
    wvals = jnp.asarray(np.asarray(thick_w, dtype=float), dtype=jnp.float64)
    out = np.zeros(n_real, dtype=float)
    for start in range(0, n_real, chunk):
        sub = H_batch[start : start + chunk]
        pad = chunk - len(sub)
        if pad:
            sub = np.concatenate([sub, np.repeat(sub[-1:, :, :], pad, axis=0)], axis=0)
        shard = sub.reshape(n_devices, per_device, n, n)
        vals = np.asarray(fn(jnp.asarray(shard, dtype=jnp.complex128), tvals, wvals)).reshape(-1)
        out[start : start + min(chunk, n_real - start)] = vals[: min(chunk, n_real - start)]
    return out


def jax_bloch_available(backend: str = "jax") -> dict:
    """Return whether JAX can run the complex eigensolvers needed by GROND."""

    out = {"backend": backend, "available": False}
    try:
        jax, jnp = _jax_modules(backend)
        out["default_backend"] = str(jax.default_backend())
        out["devices"] = [str(d) for d in jax.devices()]
        if _jax_backend_uses_real_expm(backend):
            H = np.eye(4, dtype=np.complex128)
            H[0, 1] = 0.001 + 0.002j
            H[1, 0] = np.conj(H[0, 1])
            vals = _jax_real_expm_propagate_batch(
                H[None, :, :],
                np.asarray([100.0], dtype=float),
                np.asarray([1.0], dtype=float),
                backend,
                batch_size=1,
            )
            out["available"] = True
            out["real_expm_dtype"] = _jax_real_dtype_name(backend)
            out["real_expm_test"] = float(vals[0])
            return out
        H = jnp.asarray(np.eye(4, dtype=np.complex128) + (0.01 + 0.02j) * np.ones((4, 4), dtype=np.complex128))
        Hh = H + jnp.conj(H.T)
        vals_h, _ = jnp.linalg.eigh(Hh)
        vals, _ = jnp.linalg.eig(H)
        vals_h.block_until_ready()
        vals.block_until_ready()
        out["available"] = True
        out["eig_dtype"] = str(vals.dtype)
        out["eigh_dtype"] = str(vals_h.dtype)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    return out


def _predict_row_worker(idx):
    if _WORKER_PORT is None:
        raise RuntimeError("prediction worker was not initialized")
    return int(idx), _WORKER_PORT.row_prediction(int(idx))


def _predict_row_cache_worker(pos):
    if _WORKER_PORT is None:
        raise RuntimeError("prediction worker was not initialized")
    return _WORKER_PORT.row_prediction_from_active_cache(int(pos))


def _ordered_prediction_attempt(indices, attempt_results):
    """Validate one complete backend attempt before exposing any row results.

    Multiprocessing iterators can yield valid rows before a worker or context
    fails.  Keeping those rows would make a later context retry or serial
    fallback count and apply some rows twice.  This helper is the transaction
    boundary: an attempt is committed only when it returned every requested row
    exactly once.
    """

    requested = np.asarray(indices, dtype=int)
    position = {int(idx): int(pos) for pos, idx in enumerate(requested)}
    if len(position) != len(requested):
        raise ValueError("prediction row indices must be unique")
    ordered = [None] * len(requested)
    for idx, result in attempt_results:
        idx = int(idx)
        pos = position.get(idx)
        if pos is None:
            raise RuntimeError(f"prediction returned unexpected row index {idx}")
        if ordered[pos] is not None:
            raise RuntimeError(f"prediction returned duplicate row index {idx}")
        ordered[pos] = result
    missing = [int(requested[pos]) for pos, result in enumerate(ordered) if result is None]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:8])
        raise RuntimeError(
            f"prediction attempt returned {len(requested) - len(missing)}/{len(requested)} rows; "
            f"missing row indices: {preview}"
        )
    return ordered


def row_plan_cache_validation_errors(cache: dict, indices, expected_key: str, expected_meta: dict) -> list[str]:
    """Return reasons a row-plan cache payload cannot be trusted."""

    required = {
        "indices",
        "basis_offsets",
        "basis_flat",
        "profile_offsets",
        "profile_weights_flat",
        "profile_omega_flat",
        "profile_tilts_flat",
        "sg_offsets",
        "sg_basis_flat",
        "diff_offsets",
        "diff_idx_flat",
        "unique_diffs",
        "unique_inv_d2",
        "center_sg",
        "omega_center",
        "omega_bragg",
        "omega_delta",
        "sg_window",
        "sg_domega",
        "sg_dframe",
        "sg_window_raw",
        "sg_window_half_frames",
        "sg_window_clipped",
        "sg_jitter_sigma",
        "sg_jitter_d_inv",
        "sg_jitter_frame_fraction",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "sg_profile_same_branch_continuation_used",
        "sg_profile_same_branch_continuation_segments",
        "sg_residual_max",
        "dose_delta_b",
    }
    errors: list[str] = []
    missing = sorted(k for k in required if k not in cache)
    if missing:
        errors.append(f"missing arrays: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    meta = cache.get("meta")
    if not isinstance(meta, dict):
        errors.append("missing metadata")
        meta = {}
    if meta.get("key") != expected_key:
        errors.append("cache key mismatch")
    if int(meta.get("cache_schema", -1)) != int(expected_meta.get("cache_schema", ROW_PLAN_CACHE_SCHEMA)):
        errors.append("cache schema mismatch")
    for field, expected_value in expected_meta.items():
        if meta.get(field) != expected_value:
            errors.append(f"metadata mismatch: {field}")
    if "indices" in cache and not np.array_equal(np.asarray(cache["indices"], dtype=np.int64), np.asarray(indices, dtype=np.int64)):
        errors.append("index order mismatch")
    if "basis_offsets" in cache:
        offsets = np.asarray(cache["basis_offsets"], dtype=np.int64)
        if len(offsets) != len(indices) + 1:
            errors.append("basis offset count mismatch")
        elif offsets.ndim != 1 or offsets[0] != 0 or np.any(offsets < 0):
            errors.append("basis offsets must be a nonnegative vector starting at zero")
        elif np.any(np.diff(offsets) <= 1):
            errors.append("basis offsets do not contain direct and target beams for every row")
        elif "basis_flat" in cache and offsets[-1] != len(cache["basis_flat"]):
            errors.append("basis offsets do not span basis_flat")
    if "basis_flat" in cache:
        basis = np.asarray(cache["basis_flat"])
        if (
            basis.ndim != 2
            or basis.shape[1:] != (3,)
            or not np.all(np.isfinite(basis))
            or not np.all(basis == np.rint(basis))
        ):
            errors.append("basis_flat must be a finite integer N by 3 array")
    if "profile_offsets" in cache:
        offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        if len(offsets) != len(indices) + 1:
            errors.append("profile offset count mismatch")
        elif offsets.ndim != 1 or offsets[0] != 0 or np.any(offsets < 0):
            errors.append("profile offsets must be a nonnegative vector starting at zero")
        elif np.any(np.diff(offsets) <= 0):
            errors.append("profile offsets contain an empty row profile")
        elif "profile_weights_flat" in cache and offsets[-1] != len(cache["profile_weights_flat"]):
            errors.append("profile offsets do not span profile_weights_flat")
    if "profile_weights_flat" in cache:
        weights = np.asarray(cache["profile_weights_flat"], dtype=float)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            errors.append("profile weights contain non-finite or negative values")
        if "profile_offsets" in cache:
            offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
            if (
                len(offsets) == len(indices) + 1
                and offsets.ndim == 1
                and offsets[0] == 0
                and offsets[-1] == len(weights)
                and any(float(np.sum(weights[offsets[i] : offsets[i + 1]])) <= 0.0 for i in range(len(indices)))
            ):
                errors.append("profile weights contain an all-zero row profile")
    if "profile_omega_flat" in cache:
        values = np.asarray(cache["profile_omega_flat"], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            errors.append("profile omega values must be a finite vector")
        elif "profile_offsets" in cache:
            offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
            if offsets[-1] != len(values):
                errors.append("profile offsets do not span profile_omega_flat")
    if "profile_tilts_flat" in cache:
        values = np.asarray(cache["profile_tilts_flat"], dtype=float)
        if values.ndim != 2 or values.shape[1:] != (2,) or not np.all(
            np.isfinite(values)
        ):
            errors.append("profile tilts must be a finite N by 2 array")
        elif "profile_offsets" in cache:
            offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
            if offsets[-1] != len(values):
                errors.append("profile offsets do not span profile_tilts_flat")
    if "sg_offsets" in cache:
        offsets = np.asarray(cache["sg_offsets"], dtype=np.int64)
        if (
            len(offsets) != len(indices) + 1
            or offsets.ndim != 1
            or offsets[0] != 0
            or np.any(offsets < 0)
            or np.any(np.diff(offsets) <= 0)
        ):
            errors.append("Sg offsets are incomplete")
        elif "sg_basis_flat" in cache and offsets[-1] != len(cache["sg_basis_flat"]):
            errors.append("Sg offsets do not span sg_basis_flat")
    if "sg_basis_flat" in cache:
        sg_basis = np.asarray(cache["sg_basis_flat"], dtype=float)
        if sg_basis.ndim != 1 or not np.all(np.isfinite(sg_basis)):
            errors.append("Sg basis contains non-finite values or has the wrong shape")
    if "diff_offsets" in cache:
        offsets = np.asarray(cache["diff_offsets"], dtype=np.int64)
        if (
            len(offsets) != len(indices) + 1
            or offsets.ndim != 1
            or offsets[0] != 0
            or np.any(offsets < 0)
            or np.any(np.diff(offsets) < 4)
        ):
            errors.append("difference offsets are incomplete")
        elif "diff_idx_flat" in cache and offsets[-1] != len(cache["diff_idx_flat"]):
            errors.append("difference offsets do not span diff_idx_flat")
    if "unique_diffs" in cache:
        diffs = np.asarray(cache["unique_diffs"])
        if diffs.ndim != 2 or diffs.shape[1:] != (3,) or not np.all(np.isfinite(diffs)):
            errors.append("unique_diffs must be a finite N by 3 array")
        if "unique_inv_d2" in cache:
            inv_d2 = np.asarray(cache["unique_inv_d2"], dtype=float)
            if inv_d2.shape != (len(diffs),) or not np.all(np.isfinite(inv_d2)) or np.any(inv_d2 < 0.0):
                errors.append("unique_inv_d2 is incomplete or invalid")
        if "diff_idx_flat" in cache:
            diff_indices = np.asarray(cache["diff_idx_flat"], dtype=np.int64)
            if np.any(diff_indices < 0) or np.any(diff_indices >= len(diffs)):
                errors.append("difference indices are out of range")
    if all(name in cache for name in ("basis_offsets", "profile_offsets", "sg_offsets", "diff_offsets")):
        basis_offsets = np.asarray(cache["basis_offsets"], dtype=np.int64)
        profile_offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        sg_offsets = np.asarray(cache["sg_offsets"], dtype=np.int64)
        diff_offsets = np.asarray(cache["diff_offsets"], dtype=np.int64)
        if all(
            offsets.ndim == 1
            and len(offsets) == len(indices) + 1
            and offsets[0] == 0
            and np.all(offsets >= 0)
            for offsets in (basis_offsets, profile_offsets, sg_offsets, diff_offsets)
        ):
            basis_counts = np.diff(basis_offsets)
            profile_counts = np.diff(profile_offsets)
            if not np.array_equal(np.diff(sg_offsets), basis_counts * profile_counts):
                errors.append("Sg offsets do not contain one basis vector per profile point")
            if not np.array_equal(np.diff(diff_offsets), basis_counts * basis_counts):
                errors.append("difference offsets do not contain one index per basis-matrix entry")
    finite_row_arrays = (
        "center_sg",
        "omega_center",
        "sg_window",
        "sg_domega",
        "sg_dframe",
        "sg_window_raw",
        "sg_window_half_frames",
        "sg_window_clipped",
        "sg_jitter_sigma",
        "sg_jitter_d_inv",
        "sg_jitter_frame_fraction",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "sg_profile_same_branch_continuation_used",
        "sg_profile_same_branch_continuation_segments",
        "sg_residual_max",
        "dose_delta_b",
    )
    optional_nan_contracts = {
        # These are audit-only quantities which are undefined for the stated
        # physical contract.  The cache builder records that fact as NaN.  Do
        # not relax validation for any geometry-defining array or for a mode
        # in which the quantity is applicable.
        "sg_window_half_frames": (
            "sg_window_mode",
            {"fixed", "rlp_peak"},
        ),
        "sg_jitter_d_inv": (
            "sg_jitter_mode",
            {"none"},
        ),
        "sg_jitter_frame_fraction": (
            "sg_jitter_mode",
            {"none"},
        ),
    }
    for name in finite_row_arrays:
        if name not in cache:
            continue
        values = np.asarray(cache[name])
        if values.shape != (len(indices),):
            errors.append(f"{name} is incomplete or non-finite")
            continue
        optional_contract = optional_nan_contracts.get(name)
        allows_nan = False
        if optional_contract is not None:
            contract_field, nonapplicable_values = optional_contract
            allows_nan = str(expected_meta.get(contract_field, "")).lower() in nonapplicable_values
        numeric = np.asarray(values, dtype=float)
        valid = ~np.isinf(numeric) if allows_nan else np.isfinite(numeric)
        if not np.all(valid):
            errors.append(f"{name} is incomplete or non-finite")
    if "omega_bragg" in cache or "omega_delta" in cache:
        root_available = np.asarray(
            cache.get(
                "omega_bragg_root_available",
                np.isfinite(np.asarray(cache.get("omega_bragg", []), dtype=float)),
            ),
            dtype=bool,
        )
        if root_available.shape != (len(indices),):
            errors.append("omega_bragg_root_available is incomplete")
        for name in ("omega_bragg", "omega_delta"):
            values = np.asarray(cache.get(name, []), dtype=float)
            if values.shape != (len(indices),):
                errors.append(f"{name} is incomplete or non-finite")
                continue
            if root_available.shape == values.shape and np.any(
                root_available & ~np.isfinite(values)
            ):
                errors.append(f"{name} is non-finite despite an available local root")
            if np.any(np.isinf(values)):
                errors.append(f"{name} contains an infinite diagnostic")
    if all(
        name in cache
        for name in (
            "sg_profile_same_branch_continuation_used",
            "sg_profile_same_branch_continuation_segments",
        )
    ):
        continuation_used = np.asarray(cache["sg_profile_same_branch_continuation_used"], dtype=bool)
        continuation_segments = np.asarray(cache["sg_profile_same_branch_continuation_segments"], dtype=np.int64)
        if np.any(continuation_segments < 1):
            errors.append("profile continuation segment counts must be positive")
        if np.any(continuation_used & (continuation_segments < 2)):
            errors.append("profile continuation rows must record at least two strict local segments")
        if np.any(~continuation_used & (continuation_segments != 1)):
            errors.append("direct profile rows must record exactly one local segment")
    if "profile_offsets" in cache:
        offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        if (
            offsets.ndim == 1
            and len(offsets) == len(indices) + 1
            and offsets[0] == 0
            and np.all(offsets >= 0)
        ):
            profile_counts = np.diff(offsets)
            if "sg_profile_points_total" in cache and not np.array_equal(
                np.asarray(cache["sg_profile_points_total"], dtype=np.int64),
                profile_counts,
            ):
                errors.append("profile point totals do not match profile offsets")
            if "profile_weights_flat" in cache and "sg_profile_weight_sum" in cache:
                weights = np.asarray(cache["profile_weights_flat"], dtype=float)
                if offsets[-1] == len(weights):
                    sums = np.asarray(
                        [np.sum(weights[offsets[i] : offsets[i + 1]]) for i in range(len(indices))],
                        dtype=float,
                    )
                    if not np.allclose(
                        sums,
                        np.asarray(cache["sg_profile_weight_sum"], dtype=float),
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        errors.append("profile weight sums do not match cached row metadata")
    return errors


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)


def relativistic_gamma_from_wavelength(wavelength_A: float) -> float:
    """Return the electron Lorentz factor implied by its de Broglie wavelength."""

    wavelength_A = float(wavelength_A)
    if not math.isfinite(wavelength_A) or wavelength_A <= 0.0:
        raise ValueError("electron wavelength must be finite and positive")
    return math.sqrt(1.0 + (ELECTRON_COMPTON_WAVELENGTH_A / wavelength_A) ** 2)


def parse_hkl_transform(spec: str | None) -> tuple[np.ndarray, str]:
    """Parse a signed axis permutation mapping XDS HKLs into model HKLs.

    The returned matrix T uses column-vector convention:
    h_model = T @ h_xds.  For row arrays this is hkl_model = hkl_xds @ T.T.
    """
    from grond.hkl_transform import parse_hkl_transform_info

    info = parse_hkl_transform_info(spec)
    return np.asarray(info.matrix, dtype=int), info.canonical


def cell_from_real_matrix(real_matrix: np.ndarray) -> gemmi.UnitCell:
    real = np.asarray(real_matrix, dtype=float)
    a_vec, b_vec, c_vec = real[:, 0], real[:, 1], real[:, 2]

    def length(v):
        return float(np.linalg.norm(v))

    def angle(u, v):
        denom = max(length(u) * length(v), 1e-30)
        cosang = float(np.dot(u, v) / denom)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))

    return gemmi.UnitCell(
        length(a_vec),
        length(b_vec),
        length(c_vec),
        angle(b_vec, c_vec),
        angle(a_vec, c_vec),
        angle(a_vec, b_vec),
    )


def join_shelx_continuations(lines):
    out, pending = [], ""
    for raw in lines:
        line = raw.rstrip()
        pending = (pending + " " + line.strip()) if pending else line
        if pending.rstrip().endswith("="):
            pending = pending.rstrip()[:-1].strip()
            continue
        out.append(pending)
        pending = ""
    if pending:
        out.append(pending)
    return out


def parse_sfac(tokens):
    if len(tokens) < 10:
        raise ValueError(
            "SFAC records used for Bloch calculations must include an element and "
            "at least four explicit scattering-factor coefficient pairs"
        )
    element = tokens[1]
    try:
        vals = [float(v) for v in tokens[2:]]
    except ValueError as exc:
        raise ValueError(f"non-numeric SFAC coefficients for element {element}") from exc
    if not all(math.isfinite(value) for value in vals):
        raise ValueError(f"non-finite SFAC coefficients for element {element}")
    if not any(value != 0.0 for value in vals):
        raise ValueError(f"all-zero SFAC coefficients for element {element}")
    pairs = [(vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]), (vals[6], vals[7])]
    c = vals[8] if len(vals) > 8 else 0.0
    return {"element": element, "pairs": pairs, "c": c, "raw": vals}


def gemmi_op_record(text: str):
    op = gemmi.Op(str(text).strip().replace(" ", "").lower())
    den = float(op.DEN)
    rot = (np.asarray(op.rot, dtype=float) / den).tolist()
    trans = (np.asarray(op.tran, dtype=float) / den).tolist()
    trans = [float(v % 1.0) for v in trans]
    return {"rot": rot, "trans": trans, "triplet": op.triplet()}


def lattice_translations_from_latt(latt_abs: int):
    # SHELX LATT codes. Most data here are primitive, but keeping centering
    # explicit prevents silent errors when this tool is reused.
    mapping = {
        1: [(0.0, 0.0, 0.0)],
        2: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],  # I
        3: [(0.0, 0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)],  # R
        4: [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],  # F
        5: [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],  # A
        6: [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],  # B
        7: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],  # C
    }
    return mapping.get(int(latt_abs), [(0.0, 0.0, 0.0)])


def unique_symmetry_records(records):
    out = []
    seen = set()
    for rec in records:
        rot = np.asarray(rec["rot"], dtype=float)
        trans = np.asarray(rec["trans"], dtype=float) % 1.0
        key = tuple(np.rint(rot * 24).astype(int).ravel().tolist() + np.rint(trans * 24).astype(int).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append({"rot": rot.tolist(), "trans": trans.tolist(), "triplet": rec.get("triplet", "")})
    return out


def shelx_symmetry_records(latt: int, symm_texts):
    base = [gemmi_op_record("x,y,z")]
    base.extend(gemmi_op_record(text) for text in symm_texts)
    if int(latt) > 0:
        centrosym = []
        for rec in base:
            centrosym.append(rec)
            centrosym.append({"rot": (-np.asarray(rec["rot"], dtype=float)).tolist(), "trans": (-np.asarray(rec["trans"], dtype=float)).tolist(), "triplet": ""})
        base = centrosym

    records = []
    for rec in base:
        rot = np.asarray(rec["rot"], dtype=float)
        trans = np.asarray(rec["trans"], dtype=float)
        for cent in lattice_translations_from_latt(abs(int(latt))):
            records.append({"rot": rot.tolist(), "trans": (trans + np.asarray(cent, dtype=float)).tolist(), "triplet": rec.get("triplet", "")})
    return unique_symmetry_records(records)


def cif_symmetry_records(block):
    rows = []
    for tag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
        _, rows = loop_rows(block, tag)
        if rows:
            break
    records = []
    for row in rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        text = values_by_tag.get("_space_group_symop_operation_xyz") or values_by_tag.get(
            "_symmetry_equiv_pos_as_xyz"
        )
        if text:
            records.append(gemmi_op_record(str(text).strip().strip("'\"")))
    if records:
        return unique_symmetry_records(records)
    if rows:
        raise ValueError("CIF declares a symmetry-operation loop but contains no usable operations")

    declared_groups = []
    number = block.find_value("_space_group_IT_number") or block.find_value("_symmetry_Int_Tables_number")
    if number:
        parsed_number = cif_number(number)
        if not math.isfinite(parsed_number) or parsed_number != round(parsed_number):
            raise ValueError(f"CIF declares an invalid space-group number {number!r}")
        sg = gemmi.find_spacegroup_by_number(int(parsed_number))
        if sg is None:
            raise ValueError(f"CIF declares unknown space-group number {number!r}")
        declared_groups.append(("IT number", sg))

    for tag in (
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M",
        "_space_group_name_H-M_ref",
    ):
        raw = block.find_value(tag)
        if not raw:
            continue
        name = str(raw).strip().strip("'\"")
        sg = gemmi.find_spacegroup_by_name(name)
        if sg is None:
            raise ValueError(f"CIF declares an unrecognized space-group name {name!r} in {tag}")
        declared_groups.append((tag, sg))

    for tag in ("_space_group_name_Hall", "_symmetry_space_group_name_Hall"):
        raw = block.find_value(tag)
        if not raw:
            continue
        name = str(raw).strip().strip("'\"")
        sg = next((candidate for candidate in gemmi.spacegroup_table() if candidate.hall == name), None)
        if sg is None:
            raise ValueError(f"CIF declares an unrecognized Hall space-group name {name!r} in {tag}")
        declared_groups.append((tag, sg))

    if declared_groups:
        numbers = {int(group.number) for _source, group in declared_groups}
        if len(numbers) != 1:
            declarations = ", ".join(f"{source}={group.xhm()}" for source, group in declared_groups)
            raise ValueError(f"CIF space-group declarations disagree: {declarations}")
        sg = declared_groups[0][1]
        return unique_symmetry_records(gemmi_op_record(op.triplet()) for op in sg.operations())
    return [gemmi_op_record("x,y,z")]


SHELX_NON_ATOM_RECORDS = {
    "ACTA",
    "AFIX",
    "ANIS",
    "BASF",
    "BIND",
    "BLOC",
    "BOND",
    "BUMP",
    "CELL",
    "CGLS",
    "CHIV",
    "CONF",
    "CONN",
    "DAMP",
    "DANG",
    "DEFS",
    "DELU",
    "DFIX",
    "DISP",
    "EADP",
    "END",
    "EQIV",
    "EXTI",
    "EXYZ",
    "FEND",
    "FLAT",
    "FMAP",
    "FRAG",
    "FREE",
    "FVAR",
    "GRID",
    "HFIX",
    "HKLF",
    "HOPE",
    "HTAB",
    "ISOR",
    "L.S.",
    "LATT",
    "LAUE",
    "LIST",
    "MERG",
    "MOLE",
    "MORE",
    "MOVE",
    "MPLA",
    "NCSY",
    "NEUT",
    "OMIT",
    "PART",
    "PLAN",
    "PRIG",
    "REM",
    "RESI",
    "RIGU",
    "RTAB",
    "SADI",
    "SAME",
    "SFAC",
    "SHEL",
    "SIMU",
    "SIZE",
    "SPEC",
    "SQUEEZE",
    "STIR",
    "SUMP",
    "SWAT",
    "SYMM",
    "TEMP",
    "TIME",
    "TITL",
    "TWIN",
    "UNIT",
    "WGHT",
    "WPDB",
    "ZERR",
}


def shelx_parameter(value: float, free_variables: list[float], *, field: str) -> float:
    """Resolve SHELX's ``10*m+p`` parameter encoding.

    The first FVAR value is the overall scale.  Parameter references therefore
    start at ``m=2``.  A negative ``m`` denotes the complementary value.  Fixed
    values use ``m=1`` and are returned without the added ten.
    """

    raw = float(value)
    if not math.isfinite(raw):
        raise ValueError(f"non-finite SHELX {field}: {raw!r}")
    mode = math.trunc(raw / 10.0)
    multiplier = raw - 10.0 * mode
    if mode == 0 or abs(multiplier) >= 5.0:
        return raw
    if mode == 1:
        return multiplier
    if mode == -1:
        raise ValueError(f"invalid SHELX {field} free-variable reference {raw:g}: FVAR 1 is the scale")

    index = abs(mode) - 1
    if index >= len(free_variables):
        raise ValueError(
            f"SHELX {field} value {raw:g} references FVAR {abs(mode)}, "
            f"but only {len(free_variables)} FVAR values were supplied"
        )
    free_value = float(free_variables[index])
    if not math.isfinite(free_value):
        raise ValueError(f"SHELX FVAR {abs(mode)} is not finite")
    if mode > 1:
        return multiplier * free_value
    return multiplier * (free_value - 1.0)


def is_shelx_q_peak(label: str) -> bool:
    """Return whether a SHELX atom-like label is a residual-density Q peak."""

    return re.fullmatch(r"Q\d+[A-Z]?", str(label).strip(), flags=re.IGNORECASE) is not None


def cif_number(value, default=float("nan")):
    text = str(value).strip().strip("'\"")
    if not text or text in {".", "?"}:
        return float(default)
    if "(" in text:
        text = text.split("(", 1)[0]
    try:
        return float(text)
    except ValueError:
        return float(default)


def parse_res(path: Path):
    lines = join_shelx_continuations(path.read_text(errors="replace").splitlines())
    cell = None
    sfac = []
    atoms = []
    current_parent_ueq = None
    latt = 1
    symm_texts = []
    free_variables = []
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0].upper() == "FVAR":
            try:
                free_variables.extend(float(value) for value in tokens[1:])
            except ValueError as exc:
                raise ValueError(f"invalid FVAR instruction in {path}: {line.strip()}") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("REM"):
            continue
        tokens = stripped.split()
        key = tokens[0].upper()
        if key == "CELL":
            if len(tokens) < 8:
                raise ValueError(f"CELL instruction in {path} does not contain six cell parameters")
            cell = tuple(float(v) for v in tokens[2:8])
            if (
                not all(math.isfinite(value) for value in cell)
                or any(value <= 0.0 for value in cell[:3])
                or any(not 0.0 < value < 180.0 for value in cell[3:])
            ):
                raise ValueError(f"CELL instruction in {path} contains invalid cell parameters")
        elif key == "SFAC":
            sfac.append(parse_sfac(tokens))
        elif key == "LATT" and len(tokens) > 1:
            latt = int(float(tokens[1]))
        elif key == "SYMM":
            symm_texts.append(stripped[4:].strip())
        elif key in {"END", "HKLF"}:
            if key == "HKLF":
                break
        elif key in SHELX_NON_ATOM_RECORDS:
            continue
        elif len(tokens) >= 7 and tokens[1].lstrip("-").isdigit():
            if is_shelx_q_peak(tokens[0]):
                continue
            sfac_index = int(tokens[1]) - 1
            if not 0 <= sfac_index < len(sfac):
                raise ValueError(
                    f"atom {tokens[0]} in {path} references SFAC {sfac_index + 1}, "
                    f"but {len(sfac)} scattering-factor records were read"
                )
            vals = [shelx_parameter(float(v), free_variables, field=f"{tokens[0]} ADP") for v in tokens[6:]]
            atom = {
                "name": tokens[0],
                "sfac_index": sfac_index,
                "element": sfac[sfac_index]["element"],
                "x": shelx_parameter(float(tokens[2]), free_variables, field=f"{tokens[0]} x"),
                "y": shelx_parameter(float(tokens[3]), free_variables, field=f"{tokens[0]} y"),
                "z": shelx_parameter(float(tokens[4]), free_variables, field=f"{tokens[0]} z"),
                "occ": shelx_parameter(float(tokens[5]), free_variables, field=f"{tokens[0]} occupancy"),
                "shelx_sof_encoded": float(tokens[5]),
                "raw_adp": vals,
            }
            numeric_atom_values = [
                atom["x"],
                atom["y"],
                atom["z"],
                atom["occ"],
                *vals,
            ]
            if not all(math.isfinite(float(value)) for value in numeric_atom_values):
                raise ValueError(f"atom {tokens[0]} in {path} contains non-finite parameters")
            if len(vals) >= 6:
                atom.update({"adp_type": "anis", "u11": vals[0], "u22": vals[1], "u33": vals[2], "u23": vals[3], "u13": vals[4], "u12": vals[5]})
                if cell is None:
                    raise ValueError(f"CELL must precede anisotropic atom {tokens[0]} in {path}")
                current_parent_ueq = float(np.trace(shelx_u_cart_matrix(atom, cell))) / 3.0
            else:
                atom["adp_type"] = "iso"
                uiso = vals[0] if vals else 0.05
                if uiso < 0.0 and current_parent_ueq is not None:
                    uiso = abs(uiso) * current_parent_ueq
                elif uiso < 0.0:
                    uiso = abs(uiso) * 0.05
                atom["uiso"] = float(uiso)
                if atom["element"].upper() != "H" and uiso > 0.0:
                    current_parent_ueq = float(uiso)
            atoms.append(atom)
    if cell is None:
        raise ValueError(f"No CELL line in {path}")
    if not sfac:
        raise ValueError(f"No explicit electron-scattering SFAC records in {path}")
    if not atoms:
        raise ValueError(f"No atoms found before HKLF in {path}")
    model = {"cell": cell, "sfac": sfac, "atoms": atoms, "model_source": "shelx_res", "ignored_exti": None, "symmetry_ops": shelx_symmetry_records(latt, symm_texts), "shelx_latt": latt}
    operation_count = len(model["symmetry_ops"])
    for atom in atoms:
        multiplicity = len(list(unique_symmetry_mates(atom, model)))
        if not multiplicity or operation_count % multiplicity:
            raise ValueError(
                f"inconsistent special-position multiplicity for {atom['name']} in {path}: "
                f"{multiplicity} sites from {operation_count} symmetry operations"
            )
        # SHELX's SOF already contains special/general multiplicity. The
        # internal model, like coreCIF, stores physical site occupancy because
        # fcalc expands unique sites. Retain both the encoded and decoded source
        # values so free-variable and partial-occupancy provenance is not lost.
        atom["shelx_sof"] = float(atom["occ"])
        atom["site_symmetry_order"] = operation_count // multiplicity
        atom["site_multiplicity"] = multiplicity
        atom["occ"] = atom["shelx_sof"] * atom["site_symmetry_order"]
    return model


def loop_rows(block, required_tag):
    col = block.find_loop(required_tag)
    if not col:
        return [], []
    loop = col.get_loop()
    tags = [str(t) for t in loop.tags]
    rows = []
    for i in range(loop.length()):
        rows.append({tag: loop[i, j] for j, tag in enumerate(tags)})
    return tags, rows


def parse_cif(path: Path):
    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    cell = (
        cif_number(block.find_value("_cell_length_a")),
        cif_number(block.find_value("_cell_length_b")),
        cif_number(block.find_value("_cell_length_c")),
        cif_number(block.find_value("_cell_angle_alpha")),
        cif_number(block.find_value("_cell_angle_beta")),
        cif_number(block.find_value("_cell_angle_gamma")),
    )
    if (
        not all(np.isfinite(cell))
        or any(value <= 0.0 for value in cell[:3])
        or any(not 0.0 < value < 180.0 for value in cell[3:])
    ):
        raise ValueError(f"No usable cell constants in {path}")

    coefficient_kind = str(
        block.find_value("_grond_scattering_factor_kind")
        or block.find_value("_atom_type_scat_source")
        or ""
    ).strip().strip("'\"").lower()
    if coefficient_kind not in {"electron", "electron_diffraction", "electron-diffraction"}:
        raise ValueError(
            f"CIF {path} does not identify its scattering coefficients as electron factors; "
            "add '_grond_scattering_factor_kind electron' or use a SHELX RES/INS model "
            "with explicit electron SFAC records"
        )

    sfac_tags, sfac_rows = loop_rows(block, "_atom_type_symbol")
    coefficient_tags = {
        *(f"_atom_type_scat_Cromer_Mann_a{i}" for i in range(1, 5)),
        *(f"_atom_type_scat_Cromer_Mann_b{i}" for i in range(1, 5)),
    }
    if sfac_rows and not {tag.lower() for tag in coefficient_tags}.issubset(
        {tag.lower() for tag in sfac_tags}
    ):
        raise ValueError(
            f"CIF {path} does not contain explicit electron scattering-factor coefficients; "
            "supply a CIF with coefficient columns or a SHELX RES/INS model with numeric SFAC records"
        )
    sfac = []
    symbol_to_index = {}
    for row in sfac_rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        symbol = str(values_by_tag.get("_atom_type_symbol", "")).strip().strip("'\"")
        if not symbol:
            continue
        pairs = []
        for i in range(1, 5):
            a = cif_number(
                values_by_tag.get(f"_atom_type_scat_cromer_mann_a{i}", 0.0),
                0.0,
            )
            b = cif_number(
                values_by_tag.get(f"_atom_type_scat_cromer_mann_b{i}", 0.0),
                0.0,
            )
            pairs.append((a, b))
        c = cif_number(values_by_tag.get("_atom_type_scat_cromer_mann_c", 0.0), 0.0)
        coefficients = [value for pair in pairs for value in pair] + [c]
        if not all(math.isfinite(value) for value in coefficients) or not any(
            value != 0.0 for value in coefficients
        ):
            raise ValueError(f"CIF {path} has invalid or all-zero scattering coefficients for {symbol}")
        symbol_to_index[symbol.upper()] = len(sfac)
        sfac.append({"element": symbol, "pairs": pairs, "c": c, "raw": [v for pair in pairs for v in pair] + [c]})
    if not sfac:
        raise ValueError(f"No _atom_type scattering-factor loop found in {path}")

    _, aniso_rows = loop_rows(block, "_atom_site_aniso_label")
    aniso = {}
    for row in aniso_rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        label = str(values_by_tag.get("_atom_site_aniso_label", "")).strip().strip("'\"")
        if not label:
            continue
        forms = {}
        for kind, factor in (("u", 1.0), ("b", 1.0 / (8.0 * math.pi**2))):
            tags = {ij: f"_atom_site_aniso_{kind}_{ij}" for ij in ("11", "22", "33", "12", "13", "23")}
            if any(tag in values_by_tag for tag in tags.values()):
                values = {f"u{ij}": factor * cif_number(values_by_tag.get(tag)) for ij, tag in tags.items()}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"CIF {path} has incomplete or non-finite {kind.upper()}ij for atom {label}")
                forms[kind] = {"adp_type": "anis", "adp_source": f"cif_{kind}ani", **values}
        if not forms:
            raise ValueError(f"CIF {path} has an anisotropic record without Uij or Bij for atom {label}")
        if "u" in forms and "b" in forms and not np.allclose(
            [forms["u"][f"u{ij}"] for ij in ("11", "22", "33", "12", "13", "23")],
            [forms["b"][f"u{ij}"] for ij in ("11", "22", "33", "12", "13", "23")],
            rtol=1e-4, atol=1e-6,
        ):
            raise ValueError(f"CIF {path} declares inconsistent Uij and Bij for atom {label}")
        aniso[label] = forms

    _, atom_rows = loop_rows(block, "_atom_site_label")
    atoms = []
    for row in atom_rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        label = str(values_by_tag.get("_atom_site_label", "")).strip().strip("'\"")
        element = str(values_by_tag.get("_atom_site_type_symbol", "")).strip().strip("'\"")
        if not label or not element:
            continue
        sfac_index = symbol_to_index.get(element.upper())
        if sfac_index is None:
            raise ValueError(f"Atom {label} uses element {element}, but no matching _atom_type row was found in {path}")
        atom = {
            "name": label,
            "sfac_index": int(sfac_index),
            "element": element,
            "x": cif_number(values_by_tag.get("_atom_site_fract_x")),
            "y": cif_number(values_by_tag.get("_atom_site_fract_y")),
            "z": cif_number(values_by_tag.get("_atom_site_fract_z")),
            "occ": cif_number(values_by_tag.get("_atom_site_occupancy", 1.0), 1.0),
        }
        def adp_label(tag, values=values_by_tag):
            value = str(values.get(tag, "")).strip().strip("'\"").lower()
            return "" if value in {"", ".", "?"} else value

        declared = adp_label("_atom_site_adp_type")
        alternate = adp_label("_atom_site_thermal_displace_type")
        if declared and alternate and declared != alternate:
            raise ValueError(f"CIF {path} has conflicting ADP type labels for atom {label}")
        declared = declared or alternate
        if declared and declared not in {"uani", "bani", "uiso", "biso"}:
            raise ValueError(f"CIF {path} has unsupported ADP type {declared!r} for atom {label}")
        if label not in aniso:
            supplied_iso = {}
            for kind in ("u", "b"):
                tag = f"_atom_site_{kind}_iso_or_equiv"
                if tag in values_by_tag:
                    value = cif_number(values_by_tag[tag])
                    if not math.isfinite(value):
                        raise ValueError(f"CIF {path} has a non-finite or malformed isotropic displacement value {tag} for atom {label}")
                    supplied_iso[kind] = value if kind == "u" else value / (8.0 * math.pi**2)
            if len(supplied_iso) == 2 and not math.isclose(supplied_iso["u"], supplied_iso["b"], rel_tol=1e-4, abs_tol=1e-6):
                raise ValueError(f"CIF {path} declares inconsistent Uiso and Biso for atom {label}")
        if not declared:
            if label in aniso:
                declared = "uani" if "u" in aniso[label] else "bani"
            elif math.isfinite(cif_number(values_by_tag.get("_atom_site_u_iso_or_equiv"))):
                declared = "uiso"
            elif math.isfinite(cif_number(values_by_tag.get("_atom_site_b_iso_or_equiv"))):
                declared = "biso"
        if declared in {"uani", "bani"}:
            if declared[0] not in aniso.get(label, {}):
                raise ValueError(f"CIF {path} declares {declared} but lacks matching anisotropic values for atom {label}")
            atom.update(aniso[label][declared[0]])
        elif declared in {"uiso", "biso"}:
            value = cif_number(values_by_tag.get(f"_atom_site_{declared[0]}_iso_or_equiv"))
            if not math.isfinite(value):
                raise ValueError(f"CIF {path} declares {declared} but lacks its displacement value for atom {label}")
            atom.update(adp_type="iso", uiso=value if declared == "uiso" else value / (8.0 * math.pi**2), adp_source=f"cif_{declared}")
        else:
            # Preserve the documented historical fallback only when no ADP
            # representation was supplied, never for an unsupported one.
            atom.update(adp_type="iso", uiso=0.05, adp_source="default_uiso")
        numeric_atom_values = [
            value
            for key, value in atom.items()
            if key in {"x", "y", "z", "occ", "uiso", "u11", "u22", "u33", "u12", "u13", "u23"}
        ]
        if not all(math.isfinite(float(value)) for value in numeric_atom_values):
            raise ValueError(f"atom {label} in CIF {path} contains non-finite parameters")
        atoms.append(atom)
    if not atoms:
        raise ValueError(f"No _atom_site loop found in {path}")

    exti = block.find_value("_ref" + "ine_ls_extinction_coef") or block.find_value("_shelx_extinction_method")
    return {"cell": cell, "sfac": sfac, "atoms": atoms, "model_source": "cif", "ignored_exti": str(exti) if exti else None, "symmetry_ops": cif_symmetry_records(block)}


def parse_model(path: Path):
    if path.suffix.lower() == ".cif":
        return parse_cif(path)
    return parse_res(path)


def alternate_hand_model(model):
    out = copy.deepcopy(model)
    for atom in out["atoms"]:
        atom["x"] = (-float(atom["x"])) % 1.0
        atom["y"] = (-float(atom["y"])) % 1.0
        atom["z"] = (-float(atom["z"])) % 1.0
    for operation in out.get("symmetry_ops", []):
        operation["trans"] = [(-float(value)) % 1.0 for value in operation.get("trans", (0.0, 0.0, 0.0))]
    return out


def model_content_hash(model):
    payload = {
        "cell": [round(float(v), 8) for v in model.get("cell", [])],
        "sfac": model.get("sfac", []),
        "atoms": [
            {
                "sfac_index": int(a["sfac_index"]),
                "x": round(float(a["x"]), 8),
                "y": round(float(a["y"]), 8),
                "z": round(float(a["z"]), 8),
                "occ": round(float(a["occ"]), 8),
                "adp_type": a.get("adp_type", ""),
                "uiso": round(float(a.get("uiso", 0.0)), 8),
                "u11": round(float(a.get("u11", 0.0)), 8),
                "u22": round(float(a.get("u22", 0.0)), 8),
                "u33": round(float(a.get("u33", 0.0)), 8),
                "u23": round(float(a.get("u23", 0.0)), 8),
                "u13": round(float(a.get("u13", 0.0)), 8),
                "u12": round(float(a.get("u12", 0.0)), 8),
            }
            for a in model.get("atoms", [])
        ],
        "symmetry_ops": [
            {
                "rot": [[round(float(v), 8) for v in row] for row in op.get("rot", [])],
                "trans": [round(float(v) % 1.0, 8) for v in op.get("trans", [])],
            }
            for op in model.get("symmetry_ops", [])
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def symmetry_ops_p212121():
    return [
        (np.diag([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])),
        (np.diag([-1.0, -1.0, 1.0]), np.array([0.5, 0.0, 0.5])),
        (np.diag([-1.0, 1.0, -1.0]), np.array([0.0, 0.5, 0.5])),
        (np.diag([1.0, -1.0, -1.0]), np.array([0.5, 0.5, 0.0])),
    ]


def model_symmetry_ops(model):
    records = model.get("symmetry_ops") or []
    if not records:
        return [(np.eye(3), np.zeros(3))]
    return [(np.asarray(op["rot"], dtype=float), np.asarray(op["trans"], dtype=float)) for op in records]


def reciprocal_lengths(cell_tuple):
    uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
    rec = uc.reciprocal()
    return float(rec.a), float(rec.b), float(rec.c)


def inv_d2_from_hkl_array(h, k, l, cell_tuple):
    uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
    rec = uc.reciprocal()
    ast, bst, cst = float(rec.a), float(rec.b), float(rec.c)
    alst, best, gast = (math.radians(float(v)) for v in (rec.alpha, rec.beta, rec.gamma))
    return (
        (h * ast) ** 2
        + (k * bst) ** 2
        + (l * cst) ** 2
        + 2.0 * h * k * ast * bst * math.cos(gast)
        + 2.0 * h * l * ast * cst * math.cos(best)
        + 2.0 * k * l * bst * cst * math.cos(alst)
    )


def sfac_value(sfac, s2):
    return float(sum(a * math.exp(-b * s2) for a, b in sfac["pairs"]) + sfac.get("c", 0.0))


def sfac_value_array(sfac, s2):
    out = np.zeros_like(s2, dtype=float) + float(sfac.get("c", 0.0))
    for a, b in sfac["pairs"]:
        out += float(a) * np.exp(-float(b) * s2)
    return out


def cell_orthogonalization_matrix(cell_tuple):
    """Return the matrix that maps fractional coordinates to Cartesian Angstroms."""

    uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
    axes = (
        uc.orthogonalize(gemmi.Fractional(1, 0, 0)),
        uc.orthogonalize(gemmi.Fractional(0, 1, 0)),
        uc.orthogonalize(gemmi.Fractional(0, 0, 1)),
    )
    return np.asarray([[axis.x, axis.y, axis.z] for axis in axes], dtype=float).T


def shelx_u_matrix(atom):
    """Return a SHELX/CIF crystallographic U tensor in symmetric matrix form."""

    return np.asarray(
        [
            [float(atom["u11"]), float(atom["u12"]), float(atom["u13"])],
            [float(atom["u12"]), float(atom["u22"]), float(atom["u23"])],
            [float(atom["u13"]), float(atom["u23"]), float(atom["u33"])],
        ],
        dtype=float,
    )


def shelx_u_cart_matrix(atom, cell_tuple):
    """Convert crystallographic Uij values to a Cartesian displacement tensor.

    SHELX and coreCIF store Uij in the crystallographic convention used by the
    structure-factor exponent: the reciprocal-vector components are
    ``(h*a*, k*b*, l*c*)``.  Geometry restraints, by contrast, require a true
    Cartesian tensor.  This conversion is identity only for orthogonal cells.
    """

    orth = cell_orthogonalization_matrix(cell_tuple)
    reciprocal_scale = np.diag(reciprocal_lengths(cell_tuple))
    transform = orth @ reciprocal_scale
    return transform @ shelx_u_matrix(atom) @ transform.T


def atom_dw(atom, hkl, cell_tuple):
    h, k, l = (float(v) for v in hkl)
    if atom.get("adp_type") == "anis":
        ast, bst, cst = reciprocal_lengths(cell_tuple)
        scaled_hkl = np.asarray([h * ast, k * bst, l * cst], dtype=float)
        arg = float(scaled_hkl @ shelx_u_matrix(atom) @ scaled_hkl)
        return math.exp(-2.0 * math.pi * math.pi * arg)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, cell_tuple)
    return math.exp(-8.0 * math.pi * math.pi * float(atom.get("uiso", 0.05)) * s2)


def atom_dw_array(atom, h, k, l, s2, cell_tuple):
    if atom.get("adp_type") == "anis":
        ast, bst, cst = reciprocal_lengths(cell_tuple)
        u11, u22, u33 = float(atom["u11"]), float(atom["u22"]), float(atom["u33"])
        u12, u13, u23 = float(atom["u12"]), float(atom["u13"]), float(atom["u23"])
        sx = np.asarray(h, dtype=float) * ast
        sy = np.asarray(k, dtype=float) * bst
        sz = np.asarray(l, dtype=float) * cst
        arg = (
            sx * sx * u11
            + sy * sy * u22
            + sz * sz * u33
            + 2.0 * sx * sy * u12
            + 2.0 * sx * sz * u13
            + 2.0 * sy * sz * u23
        )
        return np.exp(-2.0 * math.pi * math.pi * arg)
    return np.exp(-8.0 * math.pi * math.pi * float(atom.get("uiso", 0.05)) * s2)


def unique_symmetry_mates(atom, model):
    """Yield unique sites with only a tight coordinate-rounding tolerance.

    Periodic differences handle positions written as 0.999999 versus zero.
    This is not a near-special-position constraint or coordinate adjustment.
    """

    xyz = np.asarray([atom["x"], atom["y"], atom["z"]], dtype=float)
    seen = []
    for rotation, translation in model_symmetry_ops(model):
        position = np.mod(rotation @ xyz + translation, 1.0)
        if any(
            np.max(np.abs((position - prior) - np.rint(position - prior)))
            <= SITE_SYMMETRY_TOLERANCE_FRACTIONAL
            for prior in seen
        ):
            continue
        seen.append(position)
        yield position, rotation


def fcalc_res(model, hkl):
    h, k, l = (int(v) for v in hkl)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, model["cell"])
    total = 0.0 + 0.0j
    for atom in model["atoms"]:
        sf = sfac_value(model["sfac"][atom["sfac_index"]], s2)
        for pos, rot in unique_symmetry_mates(atom, model):
            mate_hkl = np.asarray(rot, dtype=float).T @ np.asarray([h, k, l], dtype=float)
            dw = atom_dw(atom, mate_hkl, model["cell"])
            phase = 2.0 * math.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            total += atom["occ"] * sf * dw * complex(math.cos(phase), math.sin(phase))
    return total


def fcalc_res_grid(model, h, k, l):
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    l = np.asarray(l, dtype=float)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, model["cell"])
    total = np.zeros(h.shape, dtype=np.complex128)
    for atom in model["atoms"]:
        sf = sfac_value_array(model["sfac"][atom["sfac_index"]], s2)
        for pos, rot in unique_symmetry_mates(atom, model):
            mate_h = rot[0, 0] * h + rot[1, 0] * k + rot[2, 0] * l
            mate_k = rot[0, 1] * h + rot[1, 1] * k + rot[2, 1] * l
            mate_l = rot[0, 2] * h + rot[1, 2] * k + rot[2, 2] * l
            mate_s2 = 0.25 * inv_d2_from_hkl_array(
                mate_h, mate_k, mate_l, model["cell"]
            )
            dw = atom_dw_array(atom, mate_h, mate_k, mate_l, mate_s2, model["cell"])
            amp = float(atom["occ"]) * sf * dw
            phase = 2.0 * math.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            total += amp * np.exp(1j * phase)
    return total


DEFAULT_MAX_SUPPORT_SG = 0.01


@dataclass
class Config:
    n_basis: int = 20
    n_points: int = 20
    sg_window: float = 0.006
    sg_window_mode: str = "fixed"
    sg_window_min: float = 0.003
    sg_window_max: float = 0.012
    sg_window_scale: float = 1.0
    sg_window_frame_floor: float = 0.50
    sg_jitter_mode: str = "none"
    sg_jitter_sigma0: float = 0.0
    sg_jitter_sigma_d_inv: float = 0.0
    sg_jitter_sigma_frame: float = 0.0
    sg_jitter_points: int = 5
    sg_jitter_student_df: float = 4.0
    sg_jitter_width_factor: float = 3.0
    mosaicity_mode: str = "none"
    mosaicity_sigma_deg: float = 0.0
    mosaicity_max_sigma_deg: float = 0.5
    mosaicity_points: int = 3
    mosaicity_support_sigma_deg: float = 0.0
    mosaicity_support_points: int = 5
    mosaicity_support_width_sigma: float = 4.0
    geometry_path_file: str = ""
    profile_basis_mode: str = "center"
    profile_coordinate: str = "target_sg"
    profile_angle_max_half_width_deg: float = 2.0
    profile_quadrature: str = "uniform"
    profile_accumulation: str = "sum"
    topup_d_min: float = 0.8
    max_support_sg: float = DEFAULT_MAX_SUPPORT_SG
    thickness_A: float = 1240.0
    thickness_model: str = "uniform_gradient"
    thickness_grad_frac: float = 0.25
    thickness_sigma_frac: float = 0.0
    thickness_points: int = 7
    gamma: float | None = None
    group_width_deg: float = 3.0
    free_fraction: float = 0.10
    split_seed: int = 42
    variance_a: float = 0.0
    variance_b_corr: float = 0.0
    variance_c_peak: float = 0.0
    frame_offset: float = 0.0
    ug_d_min: float = 0.30
    absorption_mode: str = "none"
    absorption_mu: float = 0.0
    absorption_frac: float = 0.0
    absorption_mu_g2: float = 0.0
    absorption_mu_theta2: float = 0.0
    absorption_tds_b: float = 0.0
    dose_b_mode: str = "none"
    dose_b_per_frame: float = 0.0
    dose_b_ref_frame: float = float("nan")
    compute_backend: str = "scipy"
    jax_batch_size: int = 0
    jax_matrix_bucket_size: int = 0
    bethe_strong_count: int = 48
    bethe_variant: str = "schur_target"
    bethe_promote_denom: float = 0.005
    bethe_denom_eps: float = 1e-5
    matrix_convention: str = "j_minus_i"
    row_plan_cache: str = "off"
    ug_storage_mode: str = "auto"
    paired_inverse_hand: bool = False

    def __post_init__(self):
        integer_minima = {
            "n_basis": (self.n_basis, 2),
            "n_points": (self.n_points, 1),
            "sg_jitter_points": (self.sg_jitter_points, 1),
            "mosaicity_points": (self.mosaicity_points, 1),
            "mosaicity_support_points": (self.mosaicity_support_points, 2),
            "thickness_points": (self.thickness_points, 1),
            "jax_batch_size": (self.jax_batch_size, 0),
            "jax_matrix_bucket_size": (self.jax_matrix_bucket_size, 0),
            "bethe_strong_count": (self.bethe_strong_count, 2),
        }
        for name, (value, minimum) in integer_minima.items():
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")

        ug_storage_mode = str(self.ug_storage_mode or "auto").lower()
        if ug_storage_mode not in {"auto", "dense", "sparse_active"}:
            raise ValueError(
                "ug_storage_mode must be auto, dense, or sparse_active"
            )

        positive_fields = {
            "sg_window": self.sg_window,
            "sg_window_min": self.sg_window_min,
            "sg_window_max": self.sg_window_max,
            "sg_window_scale": self.sg_window_scale,
            "sg_jitter_width_factor": self.sg_jitter_width_factor,
            "topup_d_min": self.topup_d_min,
            "max_support_sg": self.max_support_sg,
            "mosaicity_max_sigma_deg": self.mosaicity_max_sigma_deg,
            "mosaicity_support_width_sigma": self.mosaicity_support_width_sigma,
            "profile_angle_max_half_width_deg": self.profile_angle_max_half_width_deg,
            "thickness_A": self.thickness_A,
            "group_width_deg": self.group_width_deg,
            "ug_d_min": self.ug_d_min,
            "bethe_denom_eps": self.bethe_denom_eps,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gamma is not None and (
            not math.isfinite(float(self.gamma)) or float(self.gamma) <= 1.0
        ):
            raise ValueError("gamma must be greater than 1 when explicitly supplied")

        nonnegative_fields = {
            "sg_window_frame_floor": self.sg_window_frame_floor,
            "sg_jitter_sigma0": self.sg_jitter_sigma0,
            "sg_jitter_sigma_d_inv": self.sg_jitter_sigma_d_inv,
            "sg_jitter_sigma_frame": self.sg_jitter_sigma_frame,
            "mosaicity_sigma_deg": self.mosaicity_sigma_deg,
            "mosaicity_support_sigma_deg": self.mosaicity_support_sigma_deg,
            "thickness_grad_frac": self.thickness_grad_frac,
            "thickness_sigma_frac": self.thickness_sigma_frac,
            "variance_a": self.variance_a,
            "variance_b_corr": self.variance_b_corr,
            "variance_c_peak": self.variance_c_peak,
            "absorption_mu": self.absorption_mu,
            "absorption_frac": self.absorption_frac,
            "absorption_mu_g2": self.absorption_mu_g2,
            "absorption_mu_theta2": self.absorption_mu_theta2,
            "absorption_tds_b": self.absorption_tds_b,
            "dose_b_per_frame": self.dose_b_per_frame,
            "bethe_promote_denom": self.bethe_promote_denom,
        }
        for name, value in nonnegative_fields.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.mosaicity_sigma_deg) > float(self.mosaicity_max_sigma_deg):
            raise ValueError(
                "mosaicity_sigma_deg exceeds the explicit mosaicity_max_sigma_deg bound"
            )
        if float(self.mosaicity_support_sigma_deg) > float(self.mosaicity_max_sigma_deg):
            raise ValueError(
                "mosaicity_support_sigma_deg exceeds the explicit mosaicity_max_sigma_deg bound"
            )
        if (
            str(self.mosaicity_mode).lower() == "gaussian_cone"
            and float(self.mosaicity_sigma_deg)
            > float(self.mosaicity_support_sigma_deg) + 1e-15
        ):
            raise ValueError(
                "gaussian_cone requires mosaicity_support_sigma_deg to be at least "
                "mosaicity_sigma_deg so candidate widths share one enclosing beam basis"
            )
        if (
            str(self.mosaicity_mode).lower() == "gaussian_cone"
            and str(self.profile_coordinate).lower() != "direct_angle"
        ):
            raise ValueError(
                "gaussian_cone currently requires profile_coordinate=direct_angle; "
                "this keeps the physical scan interval explicit and avoids extra Sg roots"
            )

        choices = {
            "sg_window_mode": (
                self.sg_window_mode,
                {"fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"},
            ),
            "sg_jitter_mode": (self.sg_jitter_mode, {"none", "gaussian", "student_t"}),
            "mosaicity_mode": (
                self.mosaicity_mode,
                {"none", "gaussian_omega", "gaussian_cone"},
            ),
            "profile_coordinate": (
                self.profile_coordinate,
                {"target_sg", "direct_angle"},
            ),
            "profile_basis_mode": (
                self.profile_basis_mode,
                {"center", "common_profile"},
            ),
            "profile_quadrature": (
                self.profile_quadrature,
                {"uniform", "trapezoid", "gauss_legendre", "legendre"},
            ),
            "profile_accumulation": (self.profile_accumulation, {"sum", "mean"}),
            "thickness_model": (
                self.thickness_model,
                {"single", "uniform_gradient", "gaussian", "gaussian_gradient"},
            ),
            "absorption_mode": (
                self.absorption_mode,
                {
                    "none",
                    "mean",
                    "beam_dependent",
                    "offdiag_fraction",
                    "offdiag_tds",
                    "midgley_like",
                    "midgley",
                },
            ),
            "dose_b_mode": (self.dose_b_mode, {"none", "linear_frame"}),
            "compute_backend": (
                self.compute_backend,
                set(COMPUTE_BACKENDS) | {"bethe", "scipy_bethe_promoted"},
            ),
            "bethe_variant": (
                self.bethe_variant,
                {"truncate", "schur0", "schur_target", "schur_mean"},
            ),
            "matrix_convention": (self.matrix_convention, {"j_minus_i", "i_minus_j"}),
            "row_plan_cache": (
                self.row_plan_cache,
                {"off", "exact", "readwrite", "rebuild"},
            ),
        }
        for name, (value, allowed) in choices.items():
            if str(value).lower() not in allowed:
                raise ValueError(f"unsupported {name} {value!r}; expected one of {sorted(allowed)}")

        if self.sg_window_min > self.sg_window_max:
            raise ValueError("sg_window_min must not exceed sg_window_max")
        fixed_width = float(self.sg_window) * float(self.sg_window_scale)
        if self.sg_window_mode == "fixed" and not self.sg_window_min <= fixed_width <= self.sg_window_max:
            raise ValueError(
                "fixed sg_window*sg_window_scale lies outside sg_window_min/sg_window_max; "
                "change the bounds explicitly instead of accepting silent clipping"
            )
        if self.thickness_grad_frac > 1.0:
            raise ValueError("thickness_grad_frac must not exceed 1")
        if not math.isfinite(float(self.sg_jitter_student_df)) or self.sg_jitter_student_df <= 1.0:
            raise ValueError("sg_jitter_student_df must be greater than 1")
        if not 0.0 < float(self.free_fraction) < 1.0:
            raise ValueError("free_fraction must be between 0 and 1")
        if not math.isfinite(float(self.frame_offset)):
            raise ValueError("frame_offset must be finite")
        if not (math.isnan(float(self.dose_b_ref_frame)) or math.isfinite(float(self.dose_b_ref_frame))):
            raise ValueError("dose_b_ref_frame must be finite or NaN")
        if bool(self.paired_inverse_hand):
            absorptive_values = (
                self.absorption_mu,
                self.absorption_frac,
                self.absorption_mu_g2,
                self.absorption_mu_theta2,
                self.absorption_tds_b,
            )
            if str(self.absorption_mode).lower() != "none" or any(
                abs(float(value)) > 0.0 for value in absorptive_values
            ):
                raise ValueError(
                    "paired_inverse_hand requires absorption_mode=none and zero absorptive coefficients"
                )
            if str(self.compute_backend).lower() not in {"scipy", "numpy_scipy_cpu"}:
                raise ValueError(
                    "paired_inverse_hand currently requires the full SciPy Hermitian backend"
                )


def frozen_kernel_contract(cfg: Config | None = None):
    """Machine-readable record of the audited Bloch convention.

    This is intentionally descriptive rather than configurable.  The XDS raw
    dynrow work should test observation contracts, not silently mutate the
    production Bloch kernel.
    """
    convention = str(getattr(cfg, "matrix_convention", "j_minus_i") if cfg is not None else "j_minus_i")
    offdiag_expr = "U(h_j - h_i) / (2*k)" if convention == "j_minus_i" else "U(h_i - h_j) / (2*k)"
    offdiag_sign = "basis[j] - basis[i]" if convention == "j_minus_i" else "basis[i] - basis[j]"
    return {
        "name": "audited_signed_row_simple_hermitian",
        "structure_factors": {
            "internal_occupancy": "physical site occupancy; parsed SHELX SOF multiplied by the site-symmetry order; coreCIF occupancy unchanged",
            "site_rounding_tolerance_fractional": SITE_SYMMETRY_TOLERANCE_FRACTIONAL,
            "displacements": "crystallographic Uij or Uiso in A^2; CIF Bij/Biso converted by division by 8*pi^2",
        },
        "geometry": {
            "g(alpha)": "S @ R(alpha) @ F @ A(alpha) @ h (S=F=identity for ordinary XDS)",
            "Sg": "(k^2 - |s0 + g(alpha)|^2) / (2*k)",
            "profile_mode": "target_sg_to_angle or explicit direct-angle integration",
            "profile_coordinate": (
                None if cfg is None else str(getattr(cfg, "profile_coordinate", "target_sg"))
            ),
            "profile_quadrature": None if cfg is None else str(getattr(cfg, "profile_quadrature", "uniform")),
            "profile_accumulation": None if cfg is None else str(getattr(cfg, "profile_accumulation", "sum")),
            "profile_weight_mode": "rocking-profile samples are summed by default; legacy normalized averaging is available as profile_accumulation=mean",
            "profile_coordinate_jacobian": "target-Sg quadrature weights include relative abs(domega/dSg) so exposure sampling is uniform in angle",
            "profile_width_mode": "fixed unless cfg.sg_window_mode explicitly requests an auditable XDS metadata-derived per-row width",
            "orientation_distribution": (
                "optional target-Sg jitter, projected one-axis mosaic convolution, or "
                "an incoherent two-axis Gaussian orientation cone with one beam basis "
                "fixed over the widest declared cone; all disabled by default"
            ),
            "geometry_path": (
                "optional fixed geometry-only Chebyshev slew and two-axis tilt contract; "
                "never fitted from intensities inside the Bloch calculation"
            ),
            "basis_sg_recompute": "all basis Sg recomputed at every solved target-Sg angle",
            "bragg_root_search": "strict local angular bound around the recorded row geometry",
            "base_profile_root_search": "exact target-Sg continuation through strict local steps on one derivative-sign-certified Ewald branch",
            "jitter_root_search": "active optional jitter targets retain the strict single local angular bound and reject the candidate if unreachable",
        },
        "hamiltonian": {
            "matrix_convention": convention,
            "diagonal": "H_ii = S_i",
            "offdiagonal": offdiag_expr,
            "offdiagonal_sign": offdiag_sign,
            "default_convention_note": "j_minus_i is the production signed-row convention; i_minus_j is an explicit raw-forward-model diagnostic.",
            "dose_b": "optional frame-dependent isotropic Debye-Waller attenuation U_g -> U_g exp(-DeltaB/(4d_g^2)); default none",
            "hermitian_fill": "H_ji = conj(H_ij)",
            "propagation": "psi(t) = exp(2*pi*i*H*t) psi(0)",
            "paired_inverse_hand": (
                None
                if cfg is None
                else {
                    "enabled": bool(getattr(cfg, "paired_inverse_hand", False)),
                    "method": "I_inverse(t)=I_original(-t) with one elastic Hermitian eigensystem",
                    "absorption_allowed": False,
                }
            ),
            "absorption": "optional non-Hermitian optical-potential diagnostics; default none. Modes are mean, beam_dependent, offdiag_fraction, offdiag_tds, and midgley_like.",
            "absorption_passivity": "the static loss operator (H-H^dagger)/(2i) must be positive semidefinite within numerical tolerance; nonpassive grid candidates are rejected",
            "rejected_nonhermitian_modes": [
                "row-specific imaginary diagonal terms",
                "EXTI-like extinction inside the raw Bloch matrix",
                "per-row fitted absorptive form factors",
            ],
            "solver": "scipy.linalg.eigh/eig by default; optional experimental scipy expm/expm_multiply, promoted Bethe/Schur, and JAX batched paths only when explicitly requested and probed",
        },
        "basis": {
            "mode": "direct_target_topup",
            "profile_basis_mode": None if cfg is None else str(cfg.profile_basis_mode),
            "first_beams": ["direct (0,0,0)", "target signed hkl"],
            "topup_d_min_A": None if cfg is None else float(cfg.topup_d_min),
            "max_support_sg_inv_A": None if cfg is None else float(cfg.max_support_sg),
            "score": "abs(U_g)^2 / (abs(Sg) + 1e-5)",
            "rejected_modes": ["row contract measured-intensity ranking", "frame_measured_plus_topup"],
        },
        "row_identity": {
            "signed_hkl_required": True,
            "asu_mapping": False,
            "friedel_merge": False,
            "p222_canonicalization": False,
        },
    }


BETHE_BACKENDS = {"scipy_bethe", "bethe", "scipy_bethe_promoted"}


def safe_complex_inverse(values: np.ndarray, eps: float) -> np.ndarray:
    """Return a reciprocal with bounded near-zero denominators."""

    values = np.asarray(values, dtype=np.complex128)
    out = np.empty_like(values)
    mag = np.abs(values)
    ok = mag >= float(eps)
    out[ok] = 1.0 / values[ok]
    if np.any(~ok):
        small = values[~ok]
        phase = np.ones(len(small), dtype=np.complex128)
        nonzero = np.abs(small) > 0.0
        phase[nonzero] = small[nonzero] / np.abs(small[nonzero])
        out[~ok] = phase / float(eps)
    return out


def promoted_bethe_effective_hamiltonian(H: np.ndarray, cfg: Config) -> np.ndarray:
    """Return a near-resonance-promoted strong-beam Schur Hamiltonian.

    The strong/weak-beam split follows a practical suggestion from Takanori
    Nakane; this implementation is GROND's independent Schur-complement
    realization and is kept as an explicit backend for agreement testing.
    """

    H = np.asarray(H, dtype=np.complex128)
    n = int(H.shape[0])
    m = max(2, min(int(getattr(cfg, "bethe_strong_count", 48)), n))
    if m >= n:
        return H

    variant = str(getattr(cfg, "bethe_variant", "schur_target") or "schur_target").lower()
    promote_denom = max(0.0, float(getattr(cfg, "bethe_promote_denom", 0.005)))
    eps = max(float(getattr(cfg, "bethe_denom_eps", 1e-5)), 1e-30)
    if variant not in {"truncate", "schur0", "schur_target", "schur_mean"}:
        raise ValueError(
            f"unknown bethe_variant={variant!r}; expected truncate, schur0, schur_target, or schur_mean"
        )

    if promote_denom > 0.0 and variant in {"schur0", "schur_target", "schur_mean"}:
        strong_diag0 = np.diag(H[:m, :m]).astype(np.complex128)
        weak_diag0 = np.diag(H[m:, m:]).astype(np.complex128)
        if variant == "schur0":
            energy0 = 0.0 + 0.0j
        elif variant == "schur_target":
            energy0 = H[1, 1]
        else:
            energy0 = np.mean(strong_diag0)
        promote = np.abs(energy0 - weak_diag0) < promote_denom
        if np.any(promote):
            weak_indices = np.arange(m, n)
            promoted = weak_indices[promote]
            remaining = weak_indices[~promote]
            perm = np.concatenate([np.arange(m), promoted, remaining])
            H = H[np.ix_(perm, perm)]
            m = int(m + len(promoted))
            if m >= n:
                return H

    Hss = H[:m, :m].copy()
    if variant == "truncate":
        return Hss

    Hsw = H[:m, m:]
    Hws = H[m:, :m]
    weak_diag = np.diag(H[m:, m:]).astype(np.complex128)
    if variant == "schur0":
        energy = 0.0 + 0.0j
    elif variant == "schur_target":
        energy = Hss[1, 1]
    else:
        energy = np.mean(np.diag(Hss).astype(np.complex128))
    denom = energy - weak_diag
    return Hss + (Hsw * safe_complex_inverse(denom, eps)[None, :]) @ Hws


def rotation_matrix_from_header(header: dict[str, str], key: str, path: Path) -> np.ndarray:
    """Parse an optional proper 3x3 rotation matrix from an XDS-like header."""

    text = str(header.get(key, "")).strip()
    if not text:
        return np.eye(3, dtype=float)
    values = np.asarray([float(value) for value in text.split()], dtype=float)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain nine finite values in {path}")
    matrix = values.reshape(3, 3)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{key} is not an orthogonal rotation matrix in {path}")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{key} is not a proper rotation matrix in {path}")
    return matrix


class XDSData:
    def __init__(self, hkl_path: Path, hkl_transform: str | None = None):
        self.path = Path(hkl_path)
        self.header = {}
        self.hkl_transform_spec = "identity"
        self.hkl_transform_matrix = np.eye(3, dtype=int)
        rows = []
        with open(self.path) as f:
            for line in f:
                if line.startswith("!"):
                    if "=" in line:
                        key, val = line[1:].split("=", 1)
                        self.header[key.strip()] = val.strip()
                    continue
                p = line.split()
                if len(p) < 11:
                    continue
                row = [float(x) for x in p[:21]]
                if len(row) == 12:
                    # XDS_ASCII.HKL (MERGE=FALSE) keeps signed, unmerged rows
                    # but only writes H K L I SIGMA XD YD ZD RLP PEAK CORR PSI.
                    # Pad to the INTEGRATE.HKL layout so downstream geometry and
                    # target code can use either file without changing row logic.
                    h, k, l, iobs, sig, xd, yd, zd, rlp, peak, corr, psi = row
                    row = [
                        h, k, l, iobs, sig, xd, yd, zd, rlp, peak, corr,
                        float("nan"), float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), float("nan"), float("nan"),
                        psi, 0.0,
                    ]
                elif len(row) < 21:
                    continue
                rows.append(tuple(row))
        if not rows:
            raise ValueError(f"no numeric reflection rows found in {self.path}")
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 21:
            raise ValueError(f"unexpected reflection-table shape {arr.shape} in {self.path}")
        if not np.all(arr[:, :3] == np.rint(arr[:, :3])):
            raise ValueError(f"non-integer Miller indices found in {self.path}")
        self.hkl = arr[:, :3].astype(int)
        self.iobs = arr[:, 3].astype(float)
        self.sigma = arr[:, 4].astype(float)
        self.xcal = arr[:, 5].astype(float)
        self.ycal = arr[:, 6].astype(float)
        self.zcal = arr[:, 7].astype(float)
        self.rlp = arr[:, 8].astype(float)
        self.peak = arr[:, 9].astype(float)
        self.corr = arr[:, 10].astype(float)
        self.maxc = arr[:, 11].astype(float)
        self.xobs = arr[:, 12].astype(float)
        self.yobs = arr[:, 13].astype(float)
        self.zobs = arr[:, 14].astype(float)
        # INTEGRATE.HKL uses an all-zero observed centroid when XOBS/YOBS/ZOBS
        # are unavailable. It is not a measurement at the detector origin.
        # Normalize that sentinel once so geometry diagnostics leave the row
        # unscored while XCAL/YCAL/ZCAL remain available to the Bloch model.
        missing_observed_centroid = (
            (self.xobs == 0.0) & (self.yobs == 0.0) & (self.zobs == 0.0)
        )
        self.xobs[missing_observed_centroid] = np.nan
        self.yobs[missing_observed_centroid] = np.nan
        self.zobs[missing_observed_centroid] = np.nan
        self.alf0 = arr[:, 15].astype(float)
        self.bet0 = arr[:, 16].astype(float)
        self.alf1 = arr[:, 17].astype(float)
        self.bet1 = arr[:, 18].astype(float)
        self.psi = arr[:, 19].astype(float)
        self.iseg = arr[:, 20].astype(int)

        required = {
            "wavelength": ("X-RAY_WAVELENGTH", "WAVELENGTH"),
            "starting frame": ("STARTING_FRAME",),
            "starting angle": ("STARTING_ANGLE",),
            "oscillation range": ("OSCILLATION_RANGE",),
            "rotation axis": ("ROTATION_AXIS",),
            "incident beam direction": ("INCIDENT_BEAM_DIRECTION",),
            "unit-cell constants": ("UNIT_CELL_CONSTANTS",),
            "unit-cell a axis": ("UNIT_CELL_A-AXIS",),
            "unit-cell b axis": ("UNIT_CELL_B-AXIS",),
            "unit-cell c axis": ("UNIT_CELL_C-AXIS",),
        }
        missing = [
            label
            for label, aliases in required.items()
            if not any(str(self.header.get(alias, "")).strip() for alias in aliases)
        ]
        if missing:
            raise ValueError(f"missing required XDS header fields in {self.path}: {', '.join(missing)}")

        wavelength_text = self.header.get("X-RAY_WAVELENGTH") or self.header["WAVELENGTH"]
        self.wavelength = float(wavelength_text.split()[0])
        if not math.isfinite(self.wavelength) or self.wavelength <= 0.0:
            raise ValueError(f"XDS wavelength must be finite and positive in {self.path}")
        self.k = 1.0 / self.wavelength
        self.starting_frame = float(self.header["STARTING_FRAME"].split()[0])
        self.starting_angle = float(self.header["STARTING_ANGLE"].split()[0])
        self.oscillation_range = float(self.header["OSCILLATION_RANGE"].split()[0])
        if not all(math.isfinite(v) for v in (self.starting_frame, self.starting_angle, self.oscillation_range)):
            raise ValueError(f"XDS scan metadata must be finite in {self.path}")
        if self.oscillation_range == 0.0:
            raise ValueError(f"XDS oscillation range must be nonzero in {self.path}")
        axis_text = self.header.get(
            "GROND_ROTATION_AXIS_DATUM",
            self.header.get("DIALS_ROTATION_AXIS_DATUM", self.header["ROTATION_AXIS"]),
        )
        self.axis = np.array([float(x) for x in axis_text.split()[:3]], dtype=float)
        axis_norm = float(np.linalg.norm(self.axis))
        if self.axis.shape != (3,) or not np.all(np.isfinite(self.axis)) or axis_norm <= 0.0:
            raise ValueError(f"XDS rotation axis must be a finite nonzero 3-vector in {self.path}")
        self.axis = self.axis / axis_norm
        rotation_axis_lab = np.array(
            [float(x) for x in self.header["ROTATION_AXIS"].split()[:3]], dtype=float
        )
        rotation_axis_lab_norm = float(np.linalg.norm(rotation_axis_lab))
        if (
            rotation_axis_lab.shape != (3,)
            or not np.all(np.isfinite(rotation_axis_lab))
            or rotation_axis_lab_norm <= 0.0
        ):
            raise ValueError(f"XDS rotation axis must be a finite nonzero 3-vector in {self.path}")
        self.rotation_axis_lab = rotation_axis_lab / rotation_axis_lab_norm
        fixed_rotation_key = (
            "GROND_FIXED_ROTATION" if self.header.get("GROND_FIXED_ROTATION") else "DIALS_FIXED_ROTATION"
        )
        setting_rotation_key = (
            "GROND_SETTING_ROTATION"
            if self.header.get("GROND_SETTING_ROTATION")
            else "DIALS_SETTING_ROTATION"
        )
        self.fixed_rotation = rotation_matrix_from_header(self.header, fixed_rotation_key, self.path)
        self.setting_rotation = rotation_matrix_from_header(self.header, setting_rotation_key, self.path)
        self.axis_cross = np.array(
            [
                [0.0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0.0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0.0],
            ],
            dtype=float,
        )
        self.axis_cross2 = self.axis_cross @ self.axis_cross
        self.axis_eye = np.eye(3, dtype=float)
        beam_dir = np.array([float(x) for x in self.header["INCIDENT_BEAM_DIRECTION"].split()[:3]], dtype=float)
        beam_norm = float(np.linalg.norm(beam_dir))
        if beam_dir.shape != (3,) or not np.all(np.isfinite(beam_dir)) or beam_norm <= 0.0:
            raise ValueError(f"XDS incident beam direction must be a finite nonzero 3-vector in {self.path}")
        self.s0 = self.k * beam_dir / beam_norm
        cell_vals = [float(x) for x in self.header["UNIT_CELL_CONSTANTS"].split()[:6]]
        self.cell = gemmi.UnitCell(*cell_vals)
        va = np.array([float(x) for x in self.header["UNIT_CELL_A-AXIS"].split()[:3]], dtype=float)
        vb = np.array([float(x) for x in self.header["UNIT_CELL_B-AXIS"].split()[:3]], dtype=float)
        vc = np.array([float(x) for x in self.header["UNIT_CELL_C-AXIS"].split()[:3]], dtype=float)
        self.real_matrix = np.stack([va, vb, vc], axis=1)
        if not np.all(np.isfinite(self.real_matrix)):
            raise ValueError(f"XDS unit-cell axes contain non-finite values in {self.path}")
        try:
            self.A = np.linalg.inv(self.real_matrix).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"XDS unit-cell axes are singular in {self.path}") from exc
        self.A_at_scan_points = None
        self.a_scan_points_path = None
        scan_a_file = self.header.get("DIALS_A_AT_SCAN_POINTS_FILE", "").split()
        if scan_a_file:
            p = Path(scan_a_file[0])
            if not p.is_absolute():
                p = self.path.parent / p
            self.a_scan_points_path = p.resolve()
            mats = []
            with p.open() as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        vals = [float(x) for x in line.split()]
                    except ValueError as exc:
                        raise ValueError(
                            f"DIALS A scan-point line {line_number} in {p} contains nonnumeric values"
                        ) from exc
                    if len(vals) != 9:
                        raise ValueError(
                            f"DIALS A scan-point line {line_number} in {p} must contain exactly nine values"
                        )
                    matrix = np.asarray(vals, dtype=float).reshape(3, 3)
                    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-15:
                        raise ValueError(
                            f"DIALS A scan-point line {line_number} in {p} must be finite and nonsingular"
                        )
                    mats.append(matrix)
            if not mats:
                raise ValueError(f"declared DIALS A scan-point file {p} contains no matrices")
            self.A_at_scan_points = np.asarray(mats, dtype=float)
        self.pets_exact_geometry_path = None
        self.pets_exact_a_center = None
        self.pets_integration_semiangle_deg = None
        pets_geometry_file = self.header.get("PETS_EXACT_GEOMETRY_FILE", "").split()
        if pets_geometry_file:
            p = Path(pets_geometry_file[0])
            if not p.is_absolute():
                p = self.path.parent / p
            self.pets_exact_geometry_path = p.resolve()
            with np.load(self.pets_exact_geometry_path, allow_pickle=False) as loaded:
                required = {"a_center", "integration_semiangle_deg", "source_zone_id"}
                missing = sorted(required.difference(loaded.files))
                if missing:
                    raise ValueError(
                        f"PETS exact geometry is missing arrays: {', '.join(missing)}"
                    )
                centers = np.asarray(loaded["a_center"], dtype=float)
                semiangles = np.asarray(
                    loaded["integration_semiangle_deg"], dtype=float
                )
                zone_ids = np.asarray(loaded["source_zone_id"], dtype=np.int64)
            if centers.ndim != 3 or centers.shape[1:] != (3, 3):
                raise ValueError("PETS exact A centers must have shape (N,3,3)")
            if semiangles.shape != (len(centers),) or zone_ids.shape != (len(centers),):
                raise ValueError("PETS exact geometry arrays must have one row per zone")
            if (
                not np.all(np.isfinite(centers))
                or not np.all(np.isfinite(semiangles))
                or np.any(semiangles <= 0.0)
            ):
                raise ValueError("PETS exact geometry must be finite with positive semiangles")
            if not np.array_equal(zone_ids, np.arange(1, len(zone_ids) + 1)):
                raise ValueError(
                    "PETS exact geometry source zone IDs must be consecutive from one"
                )
            self.pets_exact_a_center = centers
            self.pets_integration_semiangle_deg = semiangles
        self.variance_model = tuple(float(v) for v in self.header.get("VARIANCE_MODEL", "1.0 0.0").split()[:2])
        self.apply_hkl_transform(hkl_transform)

    def apply_hkl_transform(self, spec: str | None):
        T, canonical = parse_hkl_transform(spec)
        self.hkl_transform_spec = canonical
        self.hkl_transform_matrix = T
        self.raw_hkl = self.hkl.copy()
        self.raw_A = np.asarray(self.A, dtype=float).copy()
        self.raw_real_matrix = np.asarray(self.real_matrix, dtype=float).copy()
        self.raw_cell = gemmi.UnitCell(
            self.cell.a,
            self.cell.b,
            self.cell.c,
            self.cell.alpha,
            self.cell.beta,
            self.cell.gamma,
        )
        self.axis_derived_source_cell = cell_from_real_matrix(self.raw_real_matrix)
        self.axis_derived_source_cell_basis = "UNIT_CELL_[ABC]-AXIS"
        self.working_cell_basis = "axis_derived_oriented_model_basis"
        # T is an exact signed permutation, so T^-1 = T.T.  Apply that
        # algebra directly rather than introducing avoidable floating matrix
        # inversions.  UNIT_CELL_CONSTANTS and the oriented axes are rounded
        # independently by XDS-like writers; the Ewald geometry uses A, so the
        # working cell for every candidate must come from those same axes.
        # raw_cell remains the unedited UNIT_CELL_CONSTANTS provenance.
        inv_t = T.T.astype(float)
        self.hkl = (self.raw_hkl @ T.T).astype(int)
        self.A = self.raw_A @ inv_t
        self.real_matrix = self.raw_real_matrix @ inv_t
        self.cell = cell_from_real_matrix(self.real_matrix)
        if self.A_at_scan_points is not None:
            self.A_at_scan_points = np.einsum(
                "nij,jk->nik",
                np.asarray(self.A_at_scan_points, dtype=float),
                inv_t,
            )
        if self.pets_exact_a_center is not None:
            self.pets_exact_a_center = np.einsum(
                "nij,jk->nik",
                np.asarray(self.pets_exact_a_center, dtype=float),
                inv_t,
            )
        self.header["HKL_TRANSFORM_TO_MODEL_BASIS"] = canonical
        self.header["UNIT_CELL_CONSTANTS_MODEL_BASIS"] = (
            f"{self.cell.a:.6f} {self.cell.b:.6f} {self.cell.c:.6f} "
            f"{self.cell.alpha:.6f} {self.cell.beta:.6f} {self.cell.gamma:.6f}"
        )

    def omega_from_z(self, z):
        return self.starting_angle + (float(z) - self.starting_frame) * self.oscillation_range

    def z_from_omega(self, omega):
        return self.starting_frame + (float(omega) - self.starting_angle) / self.oscillation_range

    def A_for_omega(self, omega):
        if self.A_at_scan_points is None:
            return self.A
        scan_index = self.z_from_omega(omega) - self.starting_frame
        if scan_index <= 0:
            return self.A_at_scan_points[0]
        if scan_index >= len(self.A_at_scan_points) - 1:
            return self.A_at_scan_points[-1]
        lo = int(math.floor(scan_index))
        frac = float(scan_index) - lo
        return (1.0 - frac) * self.A_at_scan_points[lo] + frac * self.A_at_scan_points[lo + 1]

    def integration_profile_provenance(self):
        """Return profile/mosaic metadata without treating it as fitted physics.

        XDS reflecting range and beam divergence describe the integration
        profile, not a uniquely separable crystal mosaicity.  DIALS profile
        dictionaries are likewise retained in their native representation.
        The values are therefore suitable as reported provenance or bounded
        prior centres, but are never activated implicitly by the Bloch engine.
        """

        def first_float(key):
            text = str(self.header.get(key, "") or "").strip()
            if not text:
                return None
            try:
                value = float(text.split()[0])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        profile_json = str(self.header.get("DIALS_PROFILE_PARAMETERS_JSON", "") or "").strip()
        dials_profile = None
        if profile_json:
            try:
                dials_profile = json.loads(profile_json)
            except json.JSONDecodeError:
                dials_profile = {"status": "unparsed", "raw": profile_json}
        reflecting = first_float("REFLECTING_RANGE_E.S.D.")
        divergence = first_float("BEAM_DIVERGENCE_E.S.D.")
        dials_sigma_m_values = []
        if isinstance(dials_profile, dict) and dials_profile.get("__id__") == "gaussian_rs":
            raw_sigma_m = dials_profile.get("sigma_m")
            if not isinstance(raw_sigma_m, (list, tuple)):
                raw_sigma_m = [raw_sigma_m]
            for raw_value in raw_sigma_m:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0.0:
                    dials_sigma_m_values.append(value)
        if dials_sigma_m_values:
            prior_sigma = float(np.median(dials_sigma_m_values))
            prior_source = "DIALS GaussianRS sigma_m (degrees)"
        else:
            prior_sigma = reflecting
            prior_source = "XDS REFLECTING_RANGE_E.S.D. (degrees)"
        return {
            "source": str(self.header.get("GROND_SOURCE", "XDS") or "XDS").strip(),
            "profile_fitting": str(self.header.get("PROFILE_FITTING", "") or "").strip() or None,
            "xds": {
                "reflecting_range_esd_deg": reflecting,
                "beam_divergence_esd_deg": divergence,
            },
            "dials": {
                "profile_model": str(self.header.get("DIALS_PROFILE_MODEL", "") or "").strip() or None,
                "profile_parameters": dials_profile,
                "gaussian_rs_sigma_m_deg": dials_sigma_m_values or None,
                "frame_sliced_shoebox_available": (
                    str(
                        self.header.get(
                            "DIALS_FRAME_SLICED_SHOEBOX_AVAILABLE", "FALSE"
                        )
                    ).strip().upper()
                    == "TRUE"
                ),
            },
            "optional_prior": {
                "projected_mosaicity_sigma_deg": prior_sigma,
                "source": prior_source,
                "status": "provenance_only_not_activated",
                "warning": (
                    "The recorded integration-profile width is a prior centre, not an automatic "
                    "Bloch nuisance value. XDS reflecting range can contain multiple broadening "
                    "terms, while DIALS GaussianRS sigma_m remains tied to its integration model."
                ),
            },
        }

    def recorded_proxy(self):
        return self.iobs * np.maximum(self.peak, 0.0) / 100.0 / np.maximum(self.rlp, 1e-12)


def _rotation_matrix_axis_angle(axis, angle_deg):
    """Return a right-handed lab-frame rotation matrix."""

    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or norm <= 0.0:
        raise ValueError("geometry-path rotation axis must be a finite nonzero 3-vector")
    axis = axis / norm
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    theta = math.radians(float(angle_deg))
    return np.eye(3, dtype=float) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)


def load_geometry_path_contract(path):
    """Load one geometry-only smooth scan path with strict provenance checks."""

    text = str(path or "").strip()
    if not text:
        return None
    resolved = Path(text).expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if int(payload.get("schema_version", -1)) != 1 or payload.get("kind") != "grond_geometry_path":
        raise ValueError("geometry path must be a schema-1 grond_geometry_path JSON file")
    if payload.get("basis") != "chebyshev":
        raise ValueError("geometry path currently requires Chebyshev coefficients")
    center = float(payload.get("frame_center"))
    half_range = float(payload.get("frame_half_range"))
    if not math.isfinite(center) or not math.isfinite(half_range) or half_range <= 0.0:
        raise ValueError("geometry path frame centre/range must be finite and positive")
    coefficients = payload.get("coefficients") or {}
    checked = {}
    for name in ("slew_frames", "tilt_x_deg", "tilt_y_deg"):
        values = np.asarray(coefficients.get(name, []), dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"geometry path {name} coefficients must be a finite vector")
        checked[name] = values
    axes = payload.get("tilt_axes_lab") or []
    if len(axes) != 2:
        raise ValueError("geometry path must define two lab-frame tilt axes")
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (2, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("geometry path tilt axes must be a finite 2 by 3 array")
    axis_norms = np.linalg.norm(axes, axis=1)
    if np.any(axis_norms <= 0.0):
        raise ValueError("geometry path tilt axes must be nonzero")
    axes /= axis_norms[:, None]
    bounds = payload.get("bounds") or {}
    max_slew = float(bounds.get("max_abs_slew_frames", float("nan")))
    max_tilt = float(bounds.get("max_abs_tilt_deg", float("nan")))
    if not math.isfinite(max_slew) or max_slew <= 0.0:
        raise ValueError("geometry path must record a positive finite slew bound")
    if not math.isfinite(max_tilt) or max_tilt <= 0.0:
        raise ValueError("geometry path must record a positive finite tilt bound")
    sampled_x = np.linspace(-1.0, 1.0, 1001)
    sampled = {
        name: np.polynomial.chebyshev.chebval(sampled_x, values)
        for name, values in checked.items()
    }
    if float(np.max(np.abs(sampled["slew_frames"]))) > max_slew + 1e-10:
        raise ValueError("geometry path slew exceeds its declared bound")
    for name in ("tilt_x_deg", "tilt_y_deg"):
        if float(np.max(np.abs(sampled[name]))) > max_tilt + 1e-10:
            raise ValueError(f"geometry path {name} exceeds its declared bound")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "frame_center": center,
        "frame_half_range": half_range,
        "coefficients": checked,
        "tilt_axes_lab": axes,
        "source_hkl_sha256": str(payload.get("source_hkl_sha256", "") or ""),
        "bounds": {"max_abs_slew_frames": max_slew, "max_abs_tilt_deg": max_tilt},
        "fit_source": payload.get("fit_source"),
        "validation": payload.get("validation"),
    }


def validate_ug_cube(cube: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """Return a finite complex128 U_g cube of exactly the expected shape."""

    cube = np.asarray(cube)
    if cube.shape != expected_shape:
        raise ValueError(f"U_g cache shape {cube.shape} does not match expected {expected_shape}")
    if cube.dtype != np.dtype(np.complex128):
        raise ValueError(f"U_g cache dtype {cube.dtype} is not complex128")
    flat = cube.reshape(-1)
    for start in range(0, len(flat), 1_000_000):
        if not np.isfinite(flat[start : start + 1_000_000]).all():
            raise ValueError("U_g cache contains non-finite values")
    return cube


def load_validated_ug_cache(
    path: Path,
    *,
    expected_lims: tuple[int, int, int],
    expected_meta: dict,
) -> np.ndarray:
    """Load one U_g cache only after its schema, shape, metadata, and values agree."""

    expected_shape = tuple(2 * int(value) + 1 for value in expected_lims)
    expected_bytes = math.prod(expected_shape) * np.dtype(np.complex128).itemsize
    with zipfile.ZipFile(path) as archive:
        cube_member = archive.getinfo("ug_cube.npy")
        if not expected_bytes <= cube_member.file_size <= expected_bytes + 65536:
            raise ValueError(
                f"U_g cache member size {cube_member.file_size} does not match expected grid bytes {expected_bytes}"
            )
        for name, maximum in (("lims.npy", 65536), ("meta_json.npy", 1024 * 1024)):
            member = archive.getinfo(name)
            if member.file_size > maximum:
                raise ValueError(f"U_g cache member {name} is unexpectedly large ({member.file_size} bytes)")

    with np.load(path, allow_pickle=False) as loaded:
        required = {"ug_cube", "lims", "meta_json"}
        missing = sorted(required.difference(loaded.files))
        if missing:
            raise ValueError(f"U_g cache is missing arrays: {', '.join(missing)}")
        lims = tuple(int(value) for value in np.asarray(loaded["lims"]).reshape(-1))
        if lims != expected_lims:
            raise ValueError(f"U_g cache limits {lims} do not match expected {expected_lims}")
        meta_value = np.asarray(loaded["meta_json"])
        if meta_value.ndim != 0:
            raise ValueError("U_g cache metadata must be a scalar JSON value")
        meta_text = meta_value.item()
        if isinstance(meta_text, bytes):
            meta_text = meta_text.decode("utf-8")
        meta = json.loads(str(meta_text))
        if meta != expected_meta:
            raise ValueError("U_g cache metadata does not match the requested model/configuration")
        cube = np.asarray(loaded["ug_cube"])

    return validate_ug_cube(cube, expected_shape)


def write_ug_cache_atomic(
    path: Path,
    *,
    cube: np.ndarray,
    lims: tuple[int, int, int],
    meta: dict,
) -> None:
    """Write a U_g cache through a same-directory temporary file and atomic replace."""

    expected_shape = tuple(2 * int(value) + 1 for value in lims)
    cube = validate_ug_cube(cube, expected_shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
    try:
        np.savez_compressed(
            tmp,
            ug_cube=cube,
            lims=np.asarray(lims, dtype=np.int64),
            meta_json=json.dumps(meta, sort_keys=True),
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class XDSBlochPort:
    def __init__(self, data: XDSData, res_model, cfg: Config, cache_dir: Path | None = None, res_hash: str | None = None):
        self.data = data
        self.model = copy.deepcopy(res_model)
        self.cfg = cfg
        self.geometry_path = load_geometry_path_contract(
            getattr(cfg, "geometry_path_file", "")
        )
        if self.geometry_path is not None:
            expected_hkl_hash = str(self.geometry_path.get("source_hkl_sha256", ""))
            actual_hkl_hash = sha256(self.data.path)
            if not expected_hkl_hash:
                raise ValueError("geometry path is missing its source HKL hash")
            if expected_hkl_hash != actual_hkl_hash:
                raise ValueError(
                    "geometry path source HKL hash does not match the current reflection table"
                )
        self.gamma = (
            relativistic_gamma_from_wavelength(data.wavelength)
            if cfg.gamma is None
            else float(cfg.gamma)
        )
        self.res_hash = res_hash or "unknown"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model["cell"] = (data.cell.a, data.cell.b, data.cell.c, data.cell.alpha, data.cell.beta, data.cell.gamma)
        self.h_lim, self.k_lim, self.l_lim = data.cell.get_hkl_limits(float(self.cfg.ug_d_min))
        requested_ug_storage = str(
            getattr(self.cfg, "ug_storage_mode", "auto") or "auto"
        ).lower()
        row_plan_mode = str(
            getattr(self.cfg, "row_plan_cache", "off") or "off"
        ).lower()
        self.ug_storage_mode = (
            "sparse_active"
            if requested_ug_storage == "auto" and row_plan_mode != "off"
            else ("dense" if requested_ug_storage == "auto" else requested_ug_storage)
        )
        if self.ug_storage_mode == "sparse_active":
            self.ug_cube = None
            self.ug_cache_status = {
                "status": "sparse_active",
                "allocation": {
                    "dense_cube_skipped": True,
                    "reason": "row-plan cache evaluates only its exact active HKL differences",
                },
            }
        else:
            self.ug_cube = self.load_or_build_ug_cube()
        self.pool_hkl, self.pool_ug = self.build_pool()
        self.pool_inv_d2 = self.inv_d2_for_hkls(self.pool_hkl)
        self.thick, self.thick_w = self.thickness_grid()
        self.group_indices = self.make_groups()
        valid_rlp = np.asarray(self.data.rlp, dtype=float)
        valid_rlp = valid_rlp[np.isfinite(valid_rlp) & (valid_rlp > 0.0)]
        self.median_rlp = float(np.median(valid_rlp)) if valid_rlp.size else 1.0
        z_valid = np.asarray(self.data.zcal, dtype=float)
        z_valid = z_valid[np.isfinite(z_valid)]
        self.zcal_min = float(np.min(z_valid)) if z_valid.size else 0.0
        self.zcal_max = float(np.max(z_valid)) if z_valid.size else self.zcal_min
        self.zcal_span = max(self.zcal_max - self.zcal_min, 1e-12)
        self.dose_ref_frame = float(getattr(self.cfg, "dose_b_ref_frame", float("nan")))
        if not np.isfinite(self.dose_ref_frame):
            self.dose_ref_frame = float(np.min(z_valid)) if z_valid.size else 0.0

    def ug_cache_key(self):
        cell = self.model["cell"]
        payload = {
            "cache_schema": UG_CACHE_SCHEMA,
            "res_sha256": self.res_hash,
            "model_content_sha256": model_content_hash(self.model),
            "cell": [round(float(v), 6) for v in cell],
            "lims": [int(self.h_lim), int(self.k_lim), int(self.l_lim)],
            "gamma": float(self.gamma),
            "gamma_source": "wavelength" if self.cfg.gamma is None else "explicit_config",
            "ug_d_min_A": float(self.cfg.ug_d_min),
            "formula": "gamma*Fe/(pi*V); explicit electron SFAC; parsed model symmetry",
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16], payload

    def load_or_build_ug_cube(self):
        if self.cache_dir is None:
            cube = self.build_ug_cube()
            self.ug_cache_status = {
                "status": "disabled",
                "allocation": getattr(self, "ug_allocation_status", {}),
            }
            return cube
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key, meta = self.ug_cache_key()
        path = self.cache_dir / f"ug_cube_{key}.npz"
        lims = (int(self.h_lim), int(self.k_lim), int(self.l_lim))
        plan = dense_ug_memory_plan(*lims)
        cache_meta = {**meta, "key": key}
        rebuild_reason = ""
        if path.exists():
            self.ug_allocation_status = enforce_dense_ug_memory_limit(plan, operation="cache load")
            try:
                cube = load_validated_ug_cache(
                    path,
                    expected_lims=lims,
                    expected_meta=cache_meta,
                )
            except Exception as exc:
                rebuild_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
            else:
                self.ug_cache_status = {
                    "status": "hit",
                    "path": str(path),
                    "key": key,
                    "meta": meta,
                    "allocation": self.ug_allocation_status,
                }
                return cube
        cube = self.build_ug_cube()
        write_ug_cache_atomic(path, cube=cube, lims=lims, meta=cache_meta)
        self.ug_cache_status = {
            "status": "rebuilt" if rebuild_reason else "built",
            "path": str(path),
            "key": key,
            "meta": meta,
            "allocation": getattr(self, "ug_allocation_status", {}),
        }
        if rebuild_reason:
            self.ug_cache_status["rebuild_reason"] = rebuild_reason
        return cube

    def build_ug_cube(self):
        plan = dense_ug_memory_plan(self.h_lim, self.k_lim, self.l_lim)
        self.ug_allocation_status = enforce_dense_ug_memory_limit(plan, operation="build")
        h_vals = np.arange(-self.h_lim, self.h_lim + 1, dtype=float)
        k_vals = np.arange(-self.k_lim, self.k_lim + 1, dtype=float)
        l_vals = np.arange(-self.l_lim, self.l_lim + 1, dtype=float)
        hh, kk, ll = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
        factor = float(self.gamma) / (math.pi * self.data.cell.volume)
        cube = factor * fcalc_res_grid(self.model, hh, kk, ll)
        return validate_ug_cube(cube, tuple(int(value) for value in plan["shape"]))

    def ug(self, hkl):
        h, k, l = (int(v) for v in hkl)
        i, j, m = h + self.h_lim, k + self.k_lim, l + self.l_lim
        if i < 0 or j < 0 or m < 0 or i > 2 * self.h_lim or j > 2 * self.k_lim or m > 2 * self.l_lim:
            return 0.0 + 0.0j
        if self.ug_cube is None:
            factor = float(self.gamma) / (math.pi * self.data.cell.volume)
            value = fcalc_res_grid(
                self.model,
                np.asarray([h], dtype=float),
                np.asarray([k], dtype=float),
                np.asarray([l], dtype=float),
            )
            return complex(factor * np.asarray(value).reshape(-1)[0])
        return self.ug_cube[i, j, m]

    def ug_array(self, hkls):
        hkls = np.asarray(hkls, dtype=int)
        shape = hkls.shape[:-1]
        flat = hkls.reshape(-1, 3)
        i = flat[:, 0] + self.h_lim
        j = flat[:, 1] + self.k_lim
        m = flat[:, 2] + self.l_lim
        ok = (
            (i >= 0)
            & (j >= 0)
            & (m >= 0)
            & (i <= 2 * self.h_lim)
            & (j <= 2 * self.k_lim)
            & (m <= 2 * self.l_lim)
        )
        out = np.zeros(len(flat), dtype=np.complex128)
        if self.ug_cube is None:
            if np.any(ok):
                factor = float(self.gamma) / (math.pi * self.data.cell.volume)
                out[ok] = factor * fcalc_res_grid(
                    self.model,
                    flat[ok, 0],
                    flat[ok, 1],
                    flat[ok, 2],
                )
        else:
            out[ok] = self.ug_cube[i[ok], j[ok], m[ok]]
        return out.reshape(shape)

    def inv_d2_for_hkls(self, hkls):
        hkls = np.asarray(hkls, dtype=float)
        if hkls.ndim == 1:
            g = self.data.A @ hkls
            return float(np.dot(g, g))
        g = (self.data.A @ hkls.T).T
        return np.sum(g * g, axis=1)

    def row_delta_b(self, row_idx):
        mode = str(getattr(self.cfg, "dose_b_mode", "none") or "none").lower()
        if mode == "none" or abs(float(getattr(self.cfg, "dose_b_per_frame", 0.0))) <= 0.0:
            return 0.0
        if mode != "linear_frame":
            raise ValueError(f"unknown dose_b_mode={self.cfg.dose_b_mode!r}")
        dz = max(0.0, float(self.data.zcal[int(row_idx)]) - self.dose_ref_frame)
        return float(getattr(self.cfg, "dose_b_per_frame", 0.0)) * dz

    def dose_attenuation(self, hkl, row_idx):
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return 1.0
        inv_d2 = self.inv_d2_for_hkls(hkl)
        return float(math.exp(-0.25 * delta_b * inv_d2))

    def dose_attenuation_for_inv_d2(self, inv_d2, row_idx):
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return np.ones_like(inv_d2, dtype=float)
        return np.exp(-0.25 * delta_b * np.asarray(inv_d2, dtype=float))

    def ug_for_row(self, hkl, row_idx):
        return self.ug(hkl) * self.dose_attenuation(hkl, row_idx)

    def ug_for_row_array(self, hkls, row_idx):
        vals = self.ug_array(hkls)
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return vals
        inv_d2 = self.inv_d2_for_hkls(np.asarray(hkls, dtype=int).reshape(-1, 3)).reshape(vals.shape)
        return vals * np.exp(-0.25 * delta_b * inv_d2)

    def build_pool(self):
        h_lim, k_lim, l_lim = self.data.cell.get_hkl_limits(float(self.cfg.topup_d_min))
        k_grid, l_grid = np.meshgrid(
            np.arange(-k_lim, k_lim + 1, dtype=int),
            np.arange(-l_lim, l_lim + 1, dtype=int),
            indexing="ij",
        )
        hkls = []
        uvals = []
        inv_d2_limit = 1.0 / (float(self.cfg.topup_d_min) ** 2)
        for h in range(-h_lim, h_lim + 1):
            h_grid = np.full_like(k_grid, h)
            inv_d2 = inv_d2_from_hkl_array(h_grid, k_grid, l_grid, self.model["cell"])
            keep = (inv_d2 > 0.0) & (inv_d2 <= inv_d2_limit * (1.0 + 1e-12))
            if not np.any(keep):
                continue
            block = np.column_stack((h_grid[keep], k_grid[keep], l_grid[keep])).astype(int)
            hkls.append(block)
            uvals.append(np.asarray(self.ug_array(block), dtype=np.complex128))
        if not hkls:
            return np.zeros((0, 3), dtype=int), np.zeros(0, dtype=np.complex128)
        return np.concatenate(hkls, axis=0), np.concatenate(uvals)

    def thickness_grid(self):
        model = str(getattr(self.cfg, "thickness_model", "uniform_gradient") or "uniform_gradient").lower()
        n = max(1, int(self.cfg.thickness_points))
        t0 = float(self.cfg.thickness_A)
        if model == "single" or n <= 1:
            return np.array([t0], dtype=float), np.array([1.0], dtype=float)
        if model == "uniform_gradient":
            if self.cfg.thickness_grad_frac <= 0.0:
                return np.array([t0], dtype=float), np.array([1.0], dtype=float)
            x, w = leggauss(n)
            t = t0 * (1.0 + float(self.cfg.thickness_grad_frac) * x)
        elif model == "gaussian":
            sigma = abs(float(getattr(self.cfg, "thickness_sigma_frac", 0.0))) * t0
            if sigma <= 0.0:
                return np.array([t0], dtype=float), np.array([1.0], dtype=float)
            x, w = hermgauss(n)
            t = t0 + math.sqrt(2.0) * sigma * x
            w = w / math.sqrt(math.pi)
        elif model == "gaussian_gradient":
            sigma = abs(float(getattr(self.cfg, "thickness_sigma_frac", 0.0))) * t0
            grad = float(self.cfg.thickness_grad_frac)
            if sigma <= 0.0 and grad <= 0.0:
                return np.array([t0], dtype=float), np.array([1.0], dtype=float)
            xg, wg = leggauss(n)
            xh, wh = hermgauss(n)
            centers = t0 * (1.0 + grad * xg)
            wh = wh / math.sqrt(math.pi)
            t = (centers[:, None] + math.sqrt(2.0) * sigma * xh[None, :]).reshape(-1)
            w = (wg[:, None] * wh[None, :]).reshape(-1)
        else:
            raise ValueError(f"unknown thickness_model={self.cfg.thickness_model!r}")
        ok = np.asarray(t, dtype=float) > 0.0
        if not np.any(ok):
            return np.array([self.cfg.thickness_A], dtype=float), np.array([1.0], dtype=float)
        t = np.asarray(t, dtype=float)[ok]
        w = np.asarray(w, dtype=float)[ok]
        return t, w / np.sum(w)

    def make_groups(self):
        if getattr(self.data, "pets_exact_a_center", None) is not None:
            return np.rint(np.asarray(self.data.zcal, dtype=float)).astype(int) - 1
        omegas = np.array([self.omega_from_z(z) for z in self.data.zcal])
        return ((omegas - np.min(omegas)) // float(self.cfg.group_width_deg)).astype(int)

    def geometry_path_at_z(self, z):
        """Return fixed geometry-only corrections at a nominal frame."""

        if getattr(self, "geometry_path", None) is None:
            return 0.0, 0.0, 0.0
        contract = self.geometry_path
        x = (float(z) - float(contract["frame_center"])) / float(
            contract["frame_half_range"]
        )
        x = float(np.clip(x, -1.0, 1.0))
        coeff = contract["coefficients"]
        return tuple(
            float(np.polynomial.chebyshev.chebval(x, coeff[name]))
            for name in ("slew_frames", "tilt_x_deg", "tilt_y_deg")
        )

    def omega_from_z(self, z):
        nominal_z = float(z)
        if getattr(self, "geometry_path", None) is None:
            return self.data.omega_from_z(nominal_z + float(self.cfg.frame_offset))
        slew_frames, _, _ = self.geometry_path_at_z(nominal_z)
        return self.data.omega_from_z(
            nominal_z + float(self.cfg.frame_offset) + float(slew_frames)
        )

    def nominal_z_from_corrected_omega(self, omega_deg):
        """Invert the smooth slew map for profile samples by fixed-point iteration."""

        base = float(self.data.z_from_omega(omega_deg)) - float(self.cfg.frame_offset)
        if getattr(self, "geometry_path", None) is None:
            return base
        z = base
        for _ in range(8):
            slew_frames, _, _ = self.geometry_path_at_z(z)
            updated = base - float(slew_frames)
            if abs(updated - z) <= 1e-10:
                break
            z = updated
        return float(z)

    def gvecs(self, hkls, omega_deg):
        if getattr(self, "geometry_path", None) is None:
            A = self.data.A_for_omega(omega_deg)
        else:
            nominal_z = self.nominal_z_from_corrected_omega(omega_deg)
            A = self.data.A_for_omega(self.data.omega_from_z(nominal_z))
        theta = np.radians(float(omega_deg))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        R = self.data.axis_eye + s * self.data.axis_cross + (1.0 - c) * self.data.axis_cross2
        transform = self.data.setting_rotation @ R @ self.data.fixed_rotation @ A
        g = (transform @ np.asarray(hkls, dtype=float).T).T
        if getattr(self, "geometry_path", None) is None:
            return g
        _, tilt_x_deg, tilt_y_deg = self.geometry_path_at_z(nominal_z)
        axes = self.geometry_path["tilt_axes_lab"]
        correction = _rotation_matrix_axis_angle(
            axes[1], tilt_y_deg
        ) @ _rotation_matrix_axis_angle(axes[0], tilt_x_deg)
        return (correction @ g.T).T

    def mosaicity_tangent_axes(self):
        """Return deterministic along-scan and transverse lab-frame axes.

        A small crystal rotation about the incident-beam direction does not
        change excitation errors, so an incoherent orientation distribution
        needs only the two axes tangent to the beam sphere.  The first axis is
        the scan axis projected into that plane; the second completes a
        right-handed orthonormal pair.  This keeps the two fitted components
        physically interpretable and avoids an arbitrary detector convention.
        """

        beam = np.asarray(self.data.s0, dtype=float).copy()
        beam /= max(float(np.linalg.norm(beam)), 1e-30)
        scan = np.asarray(self.data.axis, dtype=float).copy()
        scan -= float(np.dot(scan, beam)) * beam
        if float(np.linalg.norm(scan)) <= 1e-12:
            seed = np.asarray([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(seed, beam))) > 0.9:
                seed = np.asarray([0.0, 1.0, 0.0], dtype=float)
            scan = seed - float(np.dot(seed, beam)) * beam
        along = scan / float(np.linalg.norm(scan))
        transverse = np.cross(beam, along)
        transverse /= float(np.linalg.norm(transverse))
        return along, transverse

    def gvecs_at_orientation(
        self,
        hkls,
        omega_deg,
        mosaic_along_deg=0.0,
        mosaic_transverse_deg=0.0,
    ):
        """Return reciprocal vectors at one explicit mosaic orientation."""

        g = np.asarray(self.gvecs(hkls, omega_deg), dtype=float)
        along_deg = float(mosaic_along_deg)
        transverse_deg = float(mosaic_transverse_deg)
        if along_deg == 0.0 and transverse_deg == 0.0:
            return g
        along, transverse = self.mosaicity_tangent_axes()
        correction = _rotation_matrix_axis_angle(
            transverse, transverse_deg
        ) @ _rotation_matrix_axis_angle(along, along_deg)
        return (correction @ g.T).T

    def sg_for_hkls(self, hkls, omega_deg):
        g = self.gvecs(hkls, omega_deg)
        return (self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)) / (2.0 * self.data.k)

    def sg_for_hkls_at_orientation(
        self,
        hkls,
        omega_deg,
        mosaic_along_deg=0.0,
        mosaic_transverse_deg=0.0,
    ):
        if float(mosaic_along_deg) == 0.0 and float(mosaic_transverse_deg) == 0.0:
            return self.sg_for_hkls(hkls, omega_deg)
        g = self.gvecs_at_orientation(
            hkls,
            omega_deg,
            mosaic_along_deg,
            mosaic_transverse_deg,
        )
        return (self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)) / (
            2.0 * self.data.k
        )

    def _fixed_xds_analytic_geometry_data(self):
        """Return fixed XDS geometry, or ``None`` when exact algebra is invalid."""

        data = getattr(self, "data", None)
        header = getattr(data, "header", {})
        source_text = str(header.get("GROND_SOURCE", "") or "").strip().upper()
        if source_text:
            source_kind = source_text
        else:
            source_kind = (
                "DIALS"
                if any(str(key).upper().startswith("DIALS_") for key in header)
                else "XDS"
            )
        required = (
            "fixed_rotation",
            "setting_rotation",
            "axis",
            "s0",
            "A",
            "k",
        )
        if (
            source_kind != "XDS"
            or getattr(data, "A_at_scan_points", None) is not None
            or getattr(self, "geometry_path", None) is not None
            or any(not hasattr(data, name) for name in required)
        ):
            return None
        return data

    def _analytic_fixed_xds_omega_for_sg(
        self,
        hkl,
        omega0,
        target_sg,
        max_delta_deg,
    ):
        """Return the exact local Sg root for a fixed XDS orientation matrix.

        This path is valid only when the source is XDS, ``A`` does not vary
        over the scan, and no fitted geometry path is active.  It solves the
        sinusoidal Ewald equation directly and retains the same nearest-local-
        branch contract as the numerical fallback.
        """

        data = self._fixed_xds_analytic_geometry_data()
        if data is None:
            raise NotImplementedError

        hkl_array = np.asarray(hkl, dtype=float)
        g0 = np.asarray(data.fixed_rotation, dtype=float) @ np.asarray(
            data.A, dtype=float
        ) @ hkl_array
        axis = np.asarray(data.axis, dtype=float)
        beam = np.asarray(data.setting_rotation, dtype=float).T @ np.asarray(
            data.s0, dtype=float
        )
        parallel = axis * float(np.dot(axis, g0))
        perpendicular = g0 - parallel
        cross = np.cross(axis, g0)
        constant = float(np.dot(beam, parallel))
        cosine = float(np.dot(beam, perpendicular))
        sine = float(np.dot(beam, cross))
        radius = math.hypot(cosine, sine)
        right_hand_side = (
            -float(data.k) * float(target_sg)
            - 0.5 * float(np.dot(g0, g0))
            - constant
        )
        if radius <= 1e-15:
            raise LocalSgCrossingError(
                "fixed XDS geometry has no orientation-dependent Sg crossing"
            )
        ratio = right_hand_side / radius
        if ratio < -1.0 - 1e-12 or ratio > 1.0 + 1e-12:
            raise LocalSgCrossingError(
                "fixed XDS geometry has no real Sg crossing"
            )
        ratio = float(np.clip(ratio, -1.0, 1.0))
        phase = math.atan2(sine, cosine)
        offset = math.acos(ratio)
        omega0_radians = math.radians(float(omega0))
        candidates = []
        for base in (phase - offset, phase + offset):
            central_turn = round((omega0_radians - base) / (2.0 * math.pi))
            for turn in (central_turn - 1, central_turn, central_turn + 1):
                candidate = math.degrees(base + 2.0 * math.pi * turn)
                if abs(candidate - float(omega0)) <= float(max_delta_deg) + 1e-10:
                    candidates.append(candidate)
        if not candidates:
            raise LocalSgCrossingError(
                "fixed XDS geometry has no crossing on the requested local branch"
            )
        return float(min(candidates, key=lambda value: abs(value - float(omega0))))

    def _numerical_omega_for_sg(
        self,
        hkl,
        omega0,
        target_sg,
        max_delta_deg,
    ):
        """Numerical local-root fallback for scan-varying or fitted geometry."""

        hkl_array = np.asarray([hkl], dtype=int)

        def residual(omega):
            return float(self.sg_for_hkls(hkl_array, float(omega))[0]) - target_sg

        offsets = np.linspace(-max_delta_deg, max_delta_deg, 65)
        values = np.asarray([residual(omega0 + offset) for offset in offsets], dtype=float)
        close = np.where(np.abs(values) <= 1e-10)[0]
        if len(close):
            best = int(close[np.argmin(np.abs(offsets[close]))])
            return omega0 + float(offsets[best])
        brackets = []
        for i in range(len(offsets) - 1):
            if values[i] * values[i + 1] < 0.0:
                midpoint = 0.5 * (offsets[i] + offsets[i + 1])
                brackets.append((abs(midpoint), i))
        if not brackets:
            raise LocalSgCrossingError(
                f"no local Sg={target_sg:.6g} crossing for HKL {tuple(int(v) for v in hkl)} "
                f"within {max_delta_deg:.6g} degrees of omega={omega0:.6g}"
            )
        _, index = min(brackets)
        lower = omega0 + float(offsets[index])
        upper = omega0 + float(offsets[index + 1])
        omega = float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=100))
        final_residual = residual(omega)
        if abs(final_residual) > 1e-9 or abs(omega - omega0) > max_delta_deg + 1e-10:
            raise ValueError(
                f"local Sg solver failed for HKL {tuple(int(v) for v in hkl)}: "
                f"residual={final_residual:.3g}, delta_omega={omega - omega0:.3g} degrees"
            )
        return omega

    def solve_omega_for_sg(self, hkl, omega0, target_sg, max_delta_deg=None):
        """Solve one local Ewald crossing without jumping to another branch."""

        hkl_array = np.asarray([hkl], dtype=int)
        omega0 = float(omega0)
        target_sg = float(target_sg)

        def residual(omega):
            return float(self.sg_for_hkls(hkl_array, float(omega))[0]) - target_sg

        initial = residual(omega0)
        if abs(initial) <= 1e-10:
            return omega0
        oscillation = abs(float(self.data.oscillation_range))
        if max_delta_deg is None:
            max_delta_deg = max(2.0 * oscillation, min(2.0, 10.0 * oscillation))
        max_delta_deg = float(max_delta_deg)
        if not math.isfinite(max_delta_deg) or max_delta_deg <= 0.0:
            raise ValueError("local Sg root-search range must be finite and positive")

        try:
            omega = self._analytic_fixed_xds_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        except NotImplementedError:
            return self._numerical_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        except LocalSgCrossingError:
            # Preserve the established numerical boundary behavior close to a
            # tangency or a rounded XDS limit.  Ordinary reachable roots never
            # enter this slow path.
            return self._numerical_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )

        final_residual = residual(omega)
        if abs(final_residual) > 1e-9:
            return self._numerical_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        return omega

    def row_diagnostic_omega_bragg(self, hkl, omega_center):
        """Return the reporting-only zero-Sg angle and its availability.

        Propagation uses the recorded scan-centered Sg profile and never this
        diagnostic. Coarse-sliced DIALS shoeboxes can legitimately span a
        crossing just outside the deliberately strict local-root window. In
        that case retain the observation and report the unavailable diagnostic
        as NaN instead of aborting or changing row membership.
        """

        if str(getattr(self.cfg, "profile_coordinate", "target_sg")) == "direct_angle":
            return float(omega_center), True
        try:
            return float(self.solve_omega_for_sg(hkl, omega_center, 0.0)), True
        except LocalSgCrossingError:
            return float("nan"), False

    def solve_sg_profile_target(self, row_idx, hkl, omega_center, target_sg, jitter_meta):
        """Solve one profile target while preserving the strict local-root contract.

        A nonzero jitter kernel is a nuisance candidate, not required geometry.
        If one of its convolved off-center samples has no crossing inside the
        audited local angular bound, the whole jitter candidate is infeasible.
        The caller can then retain ordinary or otherwise feasible candidates;
        it must not widen the bound or silently truncate and renormalize the
        requested orientation distribution.
        """

        try:
            return self.solve_omega_for_sg(hkl, omega_center, target_sg)
        except LocalSgCrossingError as exc:
            mode = str(jitter_meta.get("sg_jitter_mode", "none") or "none").lower()
            sigma = float(jitter_meta.get("sg_jitter_sigma_inv_A", 0.0) or 0.0)
            if mode == "none" or sigma <= 0.0:
                omega, segments = self.solve_same_branch_sg_profile_target(
                    hkl,
                    omega_center,
                    target_sg,
                    initial_error=exc,
                )
                jitter_meta["sg_profile_same_branch_continuation_used"] = True
                jitter_meta["sg_profile_same_branch_continuation_segments"] = max(
                    int(jitter_meta.get("sg_profile_same_branch_continuation_segments", 1)),
                    int(segments),
                )
                return omega
            raise InfeasibleSgJitterProfileError(
                "infeasible Sg-jitter quadrature target for "
                f"row {int(row_idx)}, HKL {tuple(int(v) for v in hkl)}: "
                f"mode={mode}, sigma={sigma:.6g} 1/A, target Sg={float(target_sg):.6g} 1/A; "
                f"{exc}. Reject this nuisance candidate; the local root-search bound is unchanged."
            ) from exc

    def solve_same_branch_sg_profile_target(
        self,
        hkl,
        omega_center,
        target_sg,
        *,
        initial_error,
    ):
        """Continue a required base-profile target along one monotonic Ewald branch.

        ``solve_omega_for_sg`` intentionally keeps a short angular bound because
        it is also used to validate the Bragg crossing assigned to an observed
        row.  A configured base Sg profile is different: its exact support can
        legitimately extend farther in angle when ``dSg/domega`` is small.  We
        reach that support through strict local solves for successively closer
        Sg targets, and reject any path that reverses angle or crosses a
        stationary point.  Active jitter remains subject to the single local
        bound in ``solve_sg_profile_target`` because it is an optional nuisance
        candidate rather than required base-profile support.
        """

        hkl_array = np.asarray([hkl], dtype=int)
        omega_center = float(omega_center)
        target_sg = float(target_sg)
        center_sg = float(self.sg_for_hkls(hkl_array, omega_center)[0])
        target_delta = target_sg - center_sg
        derivative0 = float(self.sg_derivative_per_deg(hkl, omega_center))
        derivative_tol = 1e-12
        if (
            not math.isfinite(center_sg)
            or not math.isfinite(target_delta)
            or not math.isfinite(derivative0)
            or abs(derivative0) <= derivative_tol
        ):
            raise LocalSgCrossingError(
                f"cannot continue Sg={target_sg:.6g} for HKL {tuple(int(v) for v in hkl)} "
                f"from omega={omega_center:.6g}: the local Ewald branch has "
                f"dSg/domega={derivative0:.6g} 1/A/degree"
            ) from initial_error

        expected_angle_direction = math.copysign(1.0, target_delta / derivative0)
        oscillation = abs(float(self.data.oscillation_range))
        strict_local_bound = max(2.0 * oscillation, min(2.0, 10.0 * oscillation))
        max_segments = 8
        max_total_delta_deg = max_segments * strict_local_bound
        last_reason = str(initial_error)
        for segments in (2, 4, max_segments):
            omega = omega_center
            branch_ok = True
            for step_index in range(1, segments + 1):
                intermediate_target = center_sg + target_delta * (step_index / segments)
                try:
                    solved = self.solve_omega_for_sg(hkl, omega, intermediate_target)
                except LocalSgCrossingError as exc:
                    last_reason = str(exc)
                    branch_ok = False
                    break
                angle_step = float(solved - omega)
                derivative = float(self.sg_derivative_per_deg(hkl, solved))
                total_angle_delta = abs(float(solved - omega_center))
                if (
                    not math.isfinite(derivative)
                    or abs(derivative) <= derivative_tol
                    or derivative * derivative0 <= 0.0
                    or angle_step * expected_angle_direction < -1e-10
                    or total_angle_delta > max_total_delta_deg + 1e-10
                ):
                    last_reason = (
                        "continuation would leave the monotonic local Ewald branch: "
                        f"omega={solved:.6g}, dSg/domega={derivative:.6g} 1/A/degree"
                    )
                    branch_ok = False
                    break
                interval_error = self.same_branch_interval_error(
                    hkl,
                    omega,
                    solved,
                    derivative0,
                    strict_local_bound,
                )
                if interval_error:
                    last_reason = interval_error
                    branch_ok = False
                    break
                omega = float(solved)
            if branch_ok:
                final_residual = float(self.sg_for_hkls(hkl_array, omega)[0]) - target_sg
                if abs(final_residual) <= 1e-9:
                    return omega, segments
                last_reason = f"final continuation residual was {final_residual:.6g} 1/A"

        raise LocalSgCrossingError(
            f"no same-branch Sg={target_sg:.6g} continuation for HKL "
            f"{tuple(int(v) for v in hkl)} from omega={omega_center:.6g} after "
            f"up to {max_segments} strict local segments and "
            f"{max_total_delta_deg:.6g} degrees total; {last_reason}"
        ) from initial_error

    def same_branch_interval_error(
        self,
        hkl,
        omega_a,
        omega_b,
        reference_derivative,
        strict_local_bound,
    ):
        """Return an explanation if one continuation interval is not monotonic.

        Endpoint derivative checks alone can miss a reversal and a second turn
        inside one successful root bracket.  Sample no more coarsely than the
        strict solver's 65-point grid, insert every DIALS scan-matrix boundary,
        and require both finite-difference slopes and midpoint derivatives to
        retain the reference branch sign.
        """

        omega_a = float(omega_a)
        omega_b = float(omega_b)
        if abs(omega_b - omega_a) <= 1e-12:
            return None
        grid_step = 2.0 * float(strict_local_bound) / 64.0
        sample_count = max(2, int(math.ceil(abs(omega_b - omega_a) / grid_step)) + 1)
        samples = list(np.linspace(min(omega_a, omega_b), max(omega_a, omega_b), sample_count))
        scan_matrices = getattr(self.data, "A_at_scan_points", None)
        starting_angle = getattr(self.data, "starting_angle", None)
        oscillation = getattr(self.data, "oscillation_range", None)
        if scan_matrices is not None and starting_angle is not None and oscillation is not None:
            for index in range(len(scan_matrices)):
                boundary = float(starting_angle) + index * float(oscillation)
                if samples[0] < boundary < samples[-1]:
                    samples.append(boundary)
        samples = np.asarray(sorted(set(samples)), dtype=float)
        hkl_array = np.asarray([hkl], dtype=int)
        values = np.asarray(
            [float(self.sg_for_hkls(hkl_array, omega)[0]) for omega in samples],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return "continuation encountered a non-finite Sg value inside a local segment"
        reference_sign = math.copysign(1.0, float(reference_derivative))
        slopes = np.diff(values) / np.diff(samples)
        slope_tol = 1e-12
        if np.any(~np.isfinite(slopes)) or np.any(reference_sign * slopes <= slope_tol):
            bad = int(np.where((~np.isfinite(slopes)) | (reference_sign * slopes <= slope_tol))[0][0])
            return (
                "continuation reverses or stalls inside a local segment: "
                f"omega interval {samples[bad]:.6g}..{samples[bad + 1]:.6g}, "
                f"finite-difference dSg/domega={slopes[bad]:.6g} 1/A/degree"
            )
        midpoints = 0.5 * (samples[:-1] + samples[1:])
        midpoint_derivatives = np.asarray(
            [float(self.sg_derivative_per_deg(hkl, omega)) for omega in midpoints],
            dtype=float,
        )
        if np.any(~np.isfinite(midpoint_derivatives)) or np.any(
            reference_sign * midpoint_derivatives <= slope_tol
        ):
            bad = int(
                np.where(
                    (~np.isfinite(midpoint_derivatives))
                    | (reference_sign * midpoint_derivatives <= slope_tol)
                )[0][0]
            )
            return (
                "continuation derivative reverses or stalls inside a local segment: "
                f"omega={midpoints[bad]:.6g}, "
                f"dSg/domega={midpoint_derivatives[bad]:.6g} 1/A/degree"
            )
        return None

    def _analytic_fixed_xds_sg_derivative_per_deg(self, hkl, omega_deg):
        """Exact dSg/domega for the same fixed-XDS contract as the root solver."""

        data = self._fixed_xds_analytic_geometry_data()
        if data is None:
            raise NotImplementedError
        g0 = np.asarray(data.fixed_rotation, dtype=float) @ np.asarray(
            data.A, dtype=float
        ) @ np.asarray(hkl, dtype=float)
        theta = math.radians(float(omega_deg))
        cosine = math.cos(theta)
        sine = math.sin(theta)
        axis = np.asarray(data.axis, dtype=float)
        rotated = (
            cosine * g0
            + sine * np.cross(axis, g0)
            + (1.0 - cosine) * axis * float(np.dot(axis, g0))
        )
        derivative = np.cross(axis, rotated)
        beam = np.asarray(data.setting_rotation, dtype=float).T @ np.asarray(
            data.s0, dtype=float
        )
        per_radian = -float(np.dot(beam, derivative)) / float(data.k)
        return per_radian * (math.pi / 180.0)

    def sg_derivative_per_deg(self, hkl, omega_deg):
        try:
            return self._analytic_fixed_xds_sg_derivative_per_deg(hkl, omega_deg)
        except NotImplementedError:
            pass
        step = 1e-4
        hp = np.asarray([hkl], dtype=int)
        sp = float(self.sg_for_hkls(hp, float(omega_deg) + step)[0])
        sm = float(self.sg_for_hkls(hp, float(omega_deg) - step)[0])
        return (sp - sm) / (2.0 * step)

    def row_sg_window(self, row_idx, hkl, omega_bragg):
        """Return an auditable per-row target-Sg window.

        The default remains the audited fixed window.  Adaptive modes are
        deliberately simple XDS metadata probes, not fitted scale parameters.
        They alter which part of the synthetic rocking curve is averaged.
        """
        mode = str(self.cfg.sg_window_mode or "fixed").lower()
        base = float(self.cfg.sg_window)
        deriv_deg = abs(float(self.sg_derivative_per_deg(hkl, omega_bragg)))
        deriv_frame = deriv_deg * abs(float(self.data.oscillation_range))
        dz = abs(float(self.data.zobs[row_idx] - self.data.zcal[row_idx]))
        peak_frac = max(float(self.data.peak[row_idx]) / 100.0, 1e-6)
        rlp_ratio = max(float(self.data.rlp[row_idx]), 1e-12) / max(self.median_rlp, 1e-12)
        half_frames = float(self.cfg.sg_window_frame_floor)

        if mode == "fixed":
            raw = base
            half_frames_used = float("nan")
            source = "global"
        elif mode == "zobs_residual":
            half_frames_used = max(half_frames, dz)
            raw = 2.0 * deriv_frame * half_frames_used
            source = "2*abs(dSg/dframe)*max(frame_floor,abs(ZOBS-ZCAL))"
        elif mode == "zobs_plus_floor":
            half_frames_used = half_frames + dz
            raw = 2.0 * deriv_frame * half_frames_used
            source = "2*abs(dSg/dframe)*(frame_floor+abs(ZOBS-ZCAL))"
        elif mode == "rlp_peak":
            half_frames_used = float("nan")
            raw = base * math.sqrt(max(rlp_ratio, 1e-12) / peak_frac)
            source = "sg_window*sqrt((RLP/median_RLP)/(PEAK/100))"
        elif mode == "dials_bbox":
            z0 = float(self.data.alf0[row_idx])
            z1 = float(self.data.bet0[row_idx])
            if np.isfinite(z0) and np.isfinite(z1) and z1 > z0:
                half_frames_used = max(half_frames, 0.5 * (z1 - z0))
                source = "2*abs(dSg/dframe)*max(frame_floor,0.5*(DIALS bbox z1-z0))"
            else:
                half_frames_used = half_frames
                source = "2*abs(dSg/dframe)*frame_floor; DIALS bbox z-range unavailable"
            raw = 2.0 * deriv_frame * half_frames_used
        else:
            raise ValueError(f"unknown sg_window_mode={self.cfg.sg_window_mode!r}")

        scaled = float(raw) * float(self.cfg.sg_window_scale)
        clipped = float(np.clip(scaled, float(self.cfg.sg_window_min), float(self.cfg.sg_window_max)))
        return clipped, {
            "sg_window_mode": mode,
            "sg_window_raw_inv_A": float(raw),
            "sg_window_source": source,
            "sg_dSg_domega_inv_A_per_deg": float(deriv_deg),
            "sg_dSg_dframe_inv_A": float(deriv_frame),
            "sg_window_half_frames": float(half_frames_used),
            "sg_window_zobs_minus_zcal_frame": float(self.data.zobs[row_idx] - self.data.zcal[row_idx]),
            "sg_window_rlp_ratio": float(rlp_ratio),
            "sg_window_peak_fraction": float(peak_frac),
            "sg_window_clipped": bool(clipped != scaled),
        }

    def row_sg_jitter_sigma(self, row_idx, hkl):
        """Return the target-Sg jitter sigma for an observation.

        This is an orientation-distribution diagnostic, not a hidden scale.  It
        broadens the target-Sg quadrature by averaging nearby offsets and keeps
        the ordinary uniform-Sg profile as the base distribution.
        """
        mode = str(getattr(self.cfg, "sg_jitter_mode", "none") or "none").lower()
        if mode == "none":
            return 0.0, {
                "sg_jitter_mode": mode,
                "sg_jitter_sigma_inv_A": 0.0,
                "sg_jitter_d_inv_A": float("nan"),
                "sg_jitter_frame_fraction": float("nan"),
            }
        d = float(self.data.cell.calculate_d(tuple(int(v) for v in hkl)))
        d_inv = 1.0 / max(d, 1e-12)
        frame_frac = (float(self.data.zcal[row_idx]) - self.zcal_min) / self.zcal_span
        raw = (
            float(getattr(self.cfg, "sg_jitter_sigma0", 0.0))
            + float(getattr(self.cfg, "sg_jitter_sigma_d_inv", 0.0)) * d_inv
            + float(getattr(self.cfg, "sg_jitter_sigma_frame", 0.0)) * frame_frac
        )
        sigma = max(0.0, float(raw))
        return sigma, {
            "sg_jitter_mode": mode,
            "sg_jitter_sigma_inv_A": float(sigma),
            "sg_jitter_d_inv_A": float(d_inv),
            "sg_jitter_frame_fraction": float(frame_frac),
        }

    def row_profile_sg_points(self, row_idx, hkl, sg_window_i):
        """Build weighted target-Sg profile samples, optionally convolved by jitter."""
        n_base = max(1, int(self.cfg.n_points))
        half_width = 0.5 * float(sg_window_i)
        rule = str(getattr(self.cfg, "profile_quadrature", "uniform") or "uniform").lower()
        accumulation = str(getattr(self.cfg, "profile_accumulation", "sum") or "sum").lower()
        if accumulation not in {"sum", "mean"}:
            raise ValueError(f"unknown profile_accumulation={accumulation!r}; expected sum or mean")
        if n_base == 1:
            base = np.asarray([0.0], dtype=float)
            base_w = np.asarray([1.0], dtype=float)
        elif rule == "uniform":
            base = np.linspace(-half_width, half_width, n_base)
            base_w = np.ones(n_base, dtype=float)
        elif rule == "trapezoid":
            base = np.linspace(-half_width, half_width, n_base)
            base_w = np.ones(n_base, dtype=float)
            base_w[0] = 0.5
            base_w[-1] = 0.5
            base_w *= n_base / max(float(np.sum(base_w)), 1e-30)
        elif rule in {"gauss_legendre", "legendre"}:
            nodes, weights = leggauss(n_base)
            base = half_width * nodes
            base_w = np.asarray(weights, dtype=float)
            base_w *= n_base / max(float(np.sum(base_w)), 1e-30)
        else:
            raise ValueError(f"unknown profile_quadrature={rule!r}; expected uniform, trapezoid, or gauss_legendre")
        if accumulation == "mean":
            base_w = base_w / max(float(np.sum(base_w)), 1e-30)
        sigma, meta = self.row_sg_jitter_sigma(row_idx, hkl)
        meta["sg_profile_same_branch_continuation_used"] = False
        meta["sg_profile_same_branch_continuation_segments"] = 1
        mode = str(meta["sg_jitter_mode"]).lower()
        if mode == "none" or sigma <= 0.0:
            meta.update(
                {
                    "sg_profile_quadrature": rule,
                    "sg_profile_accumulation": accumulation,
                    "sg_profile_weight_sum": float(np.sum(base_w)),
                    "sg_jitter_kernel_points": 1,
                    "sg_profile_points_total": int(len(base)),
                    "sg_jitter_offset_abs_p95_inv_A": 0.0,
                }
            )
            return base, base_w, meta

        n_kernel = max(1, int(getattr(self.cfg, "sg_jitter_points", 5)))
        if mode == "gaussian":
            if n_kernel == 1:
                offsets = np.asarray([0.0], dtype=float)
                kernel_w = np.asarray([1.0], dtype=float)
            else:
                nodes, weights = hermgauss(n_kernel)
                offsets = math.sqrt(2.0) * sigma * nodes
                kernel_w = weights / math.sqrt(math.pi)
        elif mode == "student_t":
            if n_kernel == 1:
                offsets = np.asarray([0.0], dtype=float)
                kernel_w = np.asarray([1.0], dtype=float)
            else:
                width = max(float(getattr(self.cfg, "sg_jitter_width_factor", 3.0)), 1e-9) * sigma
                offsets = np.linspace(-width, width, n_kernel)
                nu = max(float(getattr(self.cfg, "sg_jitter_student_df", 4.0)), 1.01)
                x = offsets / max(sigma, 1e-12)
                kernel_w = (1.0 + (x * x) / nu) ** (-(nu + 1.0) / 2.0)
        else:
            raise ValueError(f"unknown sg_jitter_mode={getattr(self.cfg, 'sg_jitter_mode', None)!r}")

        kernel_w = np.asarray(kernel_w, dtype=float)
        kernel_w /= max(float(np.sum(kernel_w)), 1e-30)
        requested = (base[:, None] + offsets[None, :]).reshape(-1)
        weights = (base_w[:, None] * kernel_w[None, :]).reshape(-1)
        abs_offsets = np.abs(offsets)
        meta.update(
            {
                "sg_profile_quadrature": rule,
                "sg_profile_accumulation": accumulation,
                "sg_profile_weight_sum": float(np.sum(weights)),
                "sg_jitter_kernel_points": int(len(offsets)),
                "sg_profile_points_total": int(len(requested)),
                "sg_jitter_offset_abs_p95_inv_A": float(np.quantile(abs_offsets, 0.95)) if len(abs_offsets) else 0.0,
            }
        )
        return requested, weights, meta

    def row_mosaicity_angle_points(self):
        """Return a normalized projected angular mosaic distribution.

        These offsets are applied directly to the crystal rotation angle after
        each base target-Sg point has been solved.  Unlike the legacy Sg-jitter
        proxy, every finite angular sample is geometrically defined and no
        additional Ewald-root search is required.
        """

        mode = str(getattr(self.cfg, "mosaicity_mode", "none") or "none").lower()
        sigma = float(getattr(self.cfg, "mosaicity_sigma_deg", 0.0))
        if mode == "none" or sigma <= 0.0:
            return (
                np.asarray([0.0], dtype=float),
                np.asarray([1.0], dtype=float),
                {
                    "mosaicity_mode": "none" if sigma <= 0.0 else mode,
                    "mosaicity_sigma_deg": 0.0 if sigma <= 0.0 else sigma,
                    "mosaicity_kernel_points": 1,
                    "mosaicity_offset_abs_p95_deg": 0.0,
                },
            )
        if mode == "gaussian_cone":
            return (
                np.asarray([0.0], dtype=float),
                np.asarray([1.0], dtype=float),
                {
                    "mosaicity_mode": mode,
                    "mosaicity_sigma_deg": sigma,
                    "mosaicity_kernel_points": 1,
                    "mosaicity_offset_abs_p95_deg": 0.0,
                },
            )
        if mode != "gaussian_omega":
            raise ValueError(f"unknown mosaicity_mode={mode!r}")
        n = max(1, int(getattr(self.cfg, "mosaicity_points", 3)))
        if n == 1:
            offsets = np.asarray([0.0], dtype=float)
            weights = np.asarray([1.0], dtype=float)
        else:
            nodes, raw_weights = hermgauss(n)
            offsets = math.sqrt(2.0) * sigma * np.asarray(nodes, dtype=float)
            weights = np.asarray(raw_weights, dtype=float) / math.sqrt(math.pi)
            weights /= max(float(np.sum(weights)), 1e-30)
        return offsets, weights, {
            "mosaicity_mode": mode,
            "mosaicity_sigma_deg": sigma,
            "mosaicity_kernel_points": int(len(offsets)),
            "mosaicity_offset_abs_p95_deg": float(
                np.quantile(np.abs(offsets), 0.95)
            ),
        }

    def row_mosaicity_cone_points(self):
        """Return tensor Gauss-Hermite points for a 2D Gaussian cone.

        The two rotations are incoherent mosaic orientations.  Intensities,
        not amplitudes, are averaged over this normalized distribution.
        """

        mode = str(getattr(self.cfg, "mosaicity_mode", "none") or "none").lower()
        sigma = float(getattr(self.cfg, "mosaicity_sigma_deg", 0.0))
        if mode != "gaussian_cone" or sigma <= 0.0:
            return (
                np.zeros((1, 2), dtype=float),
                np.asarray([1.0], dtype=float),
                {
                    "mosaicity_cone_points": 1,
                    "mosaicity_cone_quadrature": "zero_width",
                    "mosaicity_cone_weight_sum": 1.0,
                },
            )
        n = max(1, int(getattr(self.cfg, "mosaicity_points", 3)))
        if n == 1:
            offsets = np.asarray([0.0], dtype=float)
            weights = np.asarray([1.0], dtype=float)
        else:
            nodes, raw_weights = hermgauss(n)
            offsets = math.sqrt(2.0) * sigma * np.asarray(nodes, dtype=float)
            weights = np.asarray(raw_weights, dtype=float) / math.sqrt(math.pi)
            weights /= max(float(np.sum(weights)), 1e-30)
        support_sigma = float(
            getattr(self.cfg, "mosaicity_support_sigma_deg", 0.0)
        )
        support_width = float(
            getattr(self.cfg, "mosaicity_support_width_sigma", 4.0)
        )
        if float(np.max(np.abs(offsets))) > support_sigma * support_width + 1e-12:
            raise ValueError(
                "mosaicity quadrature extends beyond the declared common-cone "
                "basis envelope; increase mosaicity_support_sigma_deg or "
                "mosaicity_support_width_sigma"
            )
        points = np.stack(
            np.meshgrid(offsets, offsets, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        pair_weights = np.outer(weights, weights).reshape(-1)
        pair_weights /= max(float(np.sum(pair_weights)), 1e-30)
        return points, pair_weights, {
            "mosaicity_cone_points": int(len(points)),
            "mosaicity_cone_quadrature": f"tensor_gauss_hermite_{n}x{n}",
            "mosaicity_cone_weight_sum": float(np.sum(pair_weights)),
            "mosaicity_cone_offset_radius_p95_deg": float(
                np.quantile(np.linalg.norm(points, axis=1), 0.95)
            ),
        }

    def row_profile_orientation_points(
        self, row_idx, hkl, omega_center, center_sg, sg_window_i
    ):
        """Return explicit scan angles and two-axis mosaic orientations."""

        (
            base_omegas,
            base_weights,
            sg_resid,
            profile_meta,
            mosaicity_meta,
        ) = self.row_profile_omega_points(
            row_idx, hkl, omega_center, center_sg, sg_window_i
        )
        cone_points, cone_weights, cone_meta = self.row_mosaicity_cone_points()
        count_base = len(base_omegas)
        count_cone = len(cone_points)
        omegas = np.repeat(np.asarray(base_omegas, dtype=float), count_cone)
        tilts = np.tile(np.asarray(cone_points, dtype=float), (count_base, 1))
        weights = (
            np.asarray(base_weights, dtype=float)[:, None]
            * np.asarray(cone_weights, dtype=float)[None, :]
        ).reshape(-1)
        profile_meta = dict(profile_meta)
        profile_meta.update(
            {
                "sg_profile_points_total": int(len(weights)),
                "sg_profile_weight_sum": float(np.sum(weights)),
            }
        )
        mosaicity_meta = {**mosaicity_meta, **cone_meta}
        if str(getattr(self.cfg, "mosaicity_mode", "none")).lower() == "gaussian_cone":
            mosaicity_meta["mosaicity_kernel_points"] = int(count_cone)
            mosaicity_meta["mosaicity_offset_abs_p95_deg"] = float(
                cone_meta.get("mosaicity_cone_offset_radius_p95_deg", 0.0)
            )
        return omegas, tilts, weights, sg_resid, profile_meta, mosaicity_meta

    def row_profile_omega_points(self, row_idx, hkl, omega_center, center_sg, sg_window_i):
        """Build solved base-profile angles convolved with direct angular mosaicity."""

        coordinate = str(
            getattr(self.cfg, "profile_coordinate", "target_sg") or "target_sg"
        ).lower()
        if coordinate == "direct_angle":
            jitter_sigma, jitter_meta = self.row_sg_jitter_sigma(row_idx, hkl)
            if jitter_sigma > 0.0:
                raise ValueError(
                    "direct-angle profile quadrature cannot be combined with target-Sg jitter"
                )
            n_base = max(1, int(self.cfg.n_points))
            derivative = abs(float(self.sg_derivative_per_deg(hkl, omega_center)))
            if not math.isfinite(derivative) or derivative <= 1e-12:
                raise ValueError("direct-angle profile requires a finite nonzero dSg/domega")
            raw_half_width_deg = 0.5 * float(sg_window_i) / derivative
            maximum = float(getattr(self.cfg, "profile_angle_max_half_width_deg", 2.0))
            half_width_deg = min(raw_half_width_deg, maximum)
            rule = str(
                getattr(self.cfg, "profile_quadrature", "uniform") or "uniform"
            ).lower()
            accumulation = str(
                getattr(self.cfg, "profile_accumulation", "sum") or "sum"
            ).lower()
            if n_base == 1:
                base_offsets = np.asarray([0.0], dtype=float)
                base_weights = np.asarray([1.0], dtype=float)
            elif rule == "uniform":
                base_offsets = np.linspace(-half_width_deg, half_width_deg, n_base)
                base_weights = np.ones(n_base, dtype=float)
            elif rule == "trapezoid":
                base_offsets = np.linspace(-half_width_deg, half_width_deg, n_base)
                base_weights = np.ones(n_base, dtype=float)
                base_weights[[0, -1]] = 0.5
                base_weights *= n_base / max(float(np.sum(base_weights)), 1e-30)
            elif rule in {"gauss_legendre", "legendre"}:
                nodes, weights = leggauss(n_base)
                base_offsets = half_width_deg * np.asarray(nodes, dtype=float)
                base_weights = np.asarray(weights, dtype=float)
                base_weights *= n_base / max(float(np.sum(base_weights)), 1e-30)
            else:
                raise ValueError(f"unknown profile_quadrature={rule!r}")
            if accumulation == "mean":
                base_weights /= max(float(np.sum(base_weights)), 1e-30)
            elif accumulation != "sum":
                raise ValueError(f"unknown profile_accumulation={accumulation!r}")
            angular_offsets, angular_weights, mosaicity_meta = (
                self.row_mosaicity_angle_points()
            )
            sample_omegas = (
                float(omega_center)
                + base_offsets[:, None]
                + np.asarray(angular_offsets, dtype=float)[None, :]
            ).reshape(-1)
            profile_weights = (
                base_weights[:, None]
                * np.asarray(angular_weights, dtype=float)[None, :]
            ).reshape(-1)
            jitter_meta.update(
                {
                    "sg_profile_coordinate": "direct_angle",
                    "sg_profile_quadrature": rule,
                    "sg_profile_accumulation": accumulation,
                    "sg_profile_angle_jacobian": False,
                    "sg_profile_angle_half_width_deg": float(half_width_deg),
                    "sg_profile_angle_raw_half_width_deg": float(raw_half_width_deg),
                    "sg_profile_angle_half_width_clipped": bool(
                        half_width_deg < raw_half_width_deg
                    ),
                    "sg_profile_weight_sum": float(np.sum(profile_weights)),
                    "sg_profile_points_total": int(len(profile_weights)),
                    "sg_profile_same_branch_continuation_used": False,
                    "sg_profile_same_branch_continuation_segments": 1,
                    "sg_profile_root_search": "not_used_direct_angle",
                    "sg_jitter_kernel_points": 1,
                    "sg_jitter_offset_abs_p95_inv_A": 0.0,
                }
            )
            return (
                np.asarray(sample_omegas, dtype=float),
                np.asarray(profile_weights, dtype=float),
                np.asarray([], dtype=float),
                jitter_meta,
                mosaicity_meta,
            )
        if coordinate != "target_sg":
            raise ValueError(f"unknown profile_coordinate={coordinate!r}")

        requested, base_weights, jitter_meta = self.row_profile_sg_points(
            row_idx, hkl, sg_window_i
        )
        physical_targets = float(center_sg) + np.asarray(requested, dtype=float)
        base_omegas = []
        root_residuals = []
        for target_sg in physical_targets:
            omega = self.solve_sg_profile_target(
                row_idx,
                hkl,
                omega_center,
                target_sg,
                jitter_meta,
            )
            base_omegas.append(float(omega))
            solved = float(
                self.sg_for_hkls(np.asarray([hkl], dtype=int), omega)[0]
            )
            root_residuals.append(solved - float(target_sg))

        base_weights = np.asarray(base_weights, dtype=float)
        jacobian = 1.0 / np.maximum(
            np.abs(
                [self.sg_derivative_per_deg(hkl, omega) for omega in base_omegas]
            ),
            1e-12,
        )
        original_weight_sum = float(np.sum(base_weights))
        base_weights *= jacobian
        base_weights *= original_weight_sum / max(
            float(np.sum(base_weights)), 1e-30
        )
        angular_offsets, angular_weights, mosaicity_meta = (
            self.row_mosaicity_angle_points()
        )
        sample_omegas = (
            np.asarray(base_omegas, dtype=float)[:, None]
            + np.asarray(angular_offsets, dtype=float)[None, :]
        ).reshape(-1)
        profile_weights = (
            base_weights[:, None] * np.asarray(angular_weights, dtype=float)[None, :]
        ).reshape(-1)
        jitter_meta.update(
            {
                "sg_profile_coordinate": "target_sg",
                "sg_profile_angle_jacobian": True,
                "sg_profile_jacobian_ratio": float(
                    np.max(jacobian) / max(np.min(jacobian), 1e-30)
                ),
                "sg_profile_weight_sum": float(np.sum(profile_weights)),
                "sg_profile_points_total": int(len(profile_weights)),
            }
        )
        return (
            np.asarray(sample_omegas, dtype=float),
            np.asarray(profile_weights, dtype=float),
            np.asarray(root_residuals, dtype=float),
            jitter_meta,
            mosaicity_meta,
        )

    def select_basis(self, target_hkl, omega_bragg, row_idx=None):
        sg_pool = self.sg_for_hkls(self.pool_hkl, omega_bragg)
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(self.pool_inv_d2, row_idx)
        score = np.abs(pool_ug) ** 2 / (np.abs(sg_pool) + 1e-5)
        ok = np.abs(sg_pool) <= float(self.cfg.max_support_sg)
        basis = [(0, 0, 0), tuple(int(v) for v in target_hkl)]
        seen = set(basis)
        candidates = np.flatnonzero(ok)
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed == 0:
            return np.asarray(basis, dtype=int)
        take = min(len(candidates), needed + 1)
        if take < len(candidates):
            local = np.argpartition(score[candidates], -take)[-take:]
            candidates = candidates[local]
        candidates = candidates[np.lexsort((candidates, -score[candidates]))]
        for idx in candidates:
            h = tuple(int(v) for v in self.pool_hkl[idx])
            if h in seen:
                continue
            basis.append(h)
            seen.add(h)
            if len(basis) >= int(self.cfg.n_basis):
                break
        return np.asarray(basis, dtype=int)

    def cone_support_orientations(self, hkl, omega_center, sg_window_i):
        """Return a fixed orientation envelope for common-basis selection.

        The envelope depends on the declared maximum cone width, not the
        candidate width or evaluation quadrature.  Consequently sigma=0 and
        every tested finite sigma use exactly the same per-row beam basis.
        """

        sigma = float(getattr(self.cfg, "mosaicity_support_sigma_deg", 0.0))
        n = max(2, int(getattr(self.cfg, "mosaicity_support_points", 5)))
        width = float(getattr(self.cfg, "mosaicity_support_width_sigma", 4.0))
        derivative = abs(float(self.sg_derivative_per_deg(hkl, omega_center)))
        if not math.isfinite(derivative) or derivative <= 1e-12:
            raise ValueError(
                "common-cone basis requires a finite nonzero dSg/domega"
            )
        raw_half_width = 0.5 * float(sg_window_i) / derivative
        maximum = float(
            getattr(self.cfg, "profile_angle_max_half_width_deg", 2.0)
        )
        half_width = min(raw_half_width, maximum)
        omega_points = np.linspace(
            float(omega_center) - half_width,
            float(omega_center) + half_width,
            n,
        )
        cone_axis = (
            np.asarray([0.0], dtype=float)
            if sigma <= 0.0
            else np.linspace(-width * sigma, width * sigma, n)
        )
        cone_points = np.stack(
            np.meshgrid(cone_axis, cone_axis, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        return omega_points, cone_points, {
            "mosaicity_basis_mode": "common_maximum_cone",
            "mosaicity_support_sigma_deg": sigma,
            "mosaicity_support_points_per_axis": int(n),
            "mosaicity_support_width_sigma": width,
            "mosaicity_support_orientation_count": int(
                len(omega_points) * len(cone_points)
            ),
            "mosaicity_support_profile_half_width_deg": float(half_width),
        }

    def select_basis_common_cone(
        self, target_hkl, omega_center, sg_window_i, row_idx=None
    ):
        """Select one beam basis over the widest declared 2D mosaic cone."""

        if str(getattr(self.cfg, "mosaicity_mode", "none")).lower() != "gaussian_cone":
            return self.select_basis(target_hkl, omega_center, row_idx=row_idx), {}
        omega_points, cone_points, meta = self.cone_support_orientations(
            target_hkl, omega_center, sg_window_i
        )
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(
                self.pool_inv_d2, row_idx
            )
        coupling = np.abs(pool_ug) ** 2
        best_score = np.zeros(len(self.pool_hkl), dtype=float)
        minimum_abs_sg = np.full(len(self.pool_hkl), np.inf, dtype=float)
        for omega in omega_points:
            for along_deg, transverse_deg in cone_points:
                sg_pool = self.sg_for_hkls_at_orientation(
                    self.pool_hkl,
                    float(omega),
                    float(along_deg),
                    float(transverse_deg),
                )
                abs_sg = np.abs(sg_pool)
                minimum_abs_sg = np.minimum(minimum_abs_sg, abs_sg)
                best_score = np.maximum(
                    best_score, coupling / (abs_sg + 1e-5)
                )
        ok = minimum_abs_sg <= float(self.cfg.max_support_sg)
        basis = [(0, 0, 0), tuple(int(v) for v in target_hkl)]
        seen = set(basis)
        candidates = np.flatnonzero(ok)
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed == 0:
            return np.asarray(basis, dtype=int), meta
        take = min(len(candidates), needed + 1)
        if take < len(candidates):
            local = np.argpartition(best_score[candidates], -take)[-take:]
            candidates = candidates[local]
        candidates = candidates[
            np.lexsort((candidates, -best_score[candidates]))
        ]
        for idx in candidates:
            h = tuple(int(v) for v in self.pool_hkl[idx])
            if h in seen:
                continue
            basis.append(h)
            seen.add(h)
            if len(basis) >= int(self.cfg.n_basis):
                break
        return np.asarray(basis, dtype=int), meta

    def select_basis_for_profile(
        self,
        target_hkl,
        omega_center,
        sg_window_i,
        solved_omegas,
        mosaic_tilts,
        row_idx=None,
    ):
        """Select one frozen basis over the declared rocking-profile support.

        ``center`` preserves the historical row-center selection. The
        high-fidelity ``common_profile`` mode evaluates a fixed support grid
        spanning the complete solved rocking interval and retains every beam
        that enters the Sg support at any support orientation. The support
        grid size is independent of propagation quadrature Q.

        A two-axis Gaussian cone already owns a wider, candidate-invariant
        support construction and continues to use that implementation.
        """

        if str(getattr(self.cfg, "mosaicity_mode", "none")).lower() == "gaussian_cone":
            return self.select_basis_common_cone(
                target_hkl,
                omega_center,
                sg_window_i,
                row_idx=row_idx,
            )
        mode = str(
            getattr(self.cfg, "profile_basis_mode", "center") or "center"
        ).lower()
        if mode == "center":
            return self.select_basis(
                target_hkl,
                omega_center,
                row_idx=row_idx,
            ), {
                "profile_basis_mode": "center",
                "profile_basis_orientation_count": 1,
            }
        if mode != "common_profile":
            raise ValueError(f"unknown profile_basis_mode={mode!r}")

        omegas = np.asarray(solved_omegas, dtype=float).reshape(-1)
        tilts = np.asarray(mosaic_tilts, dtype=float).reshape(-1, 2)
        if len(omegas) == 0 or len(omegas) != len(tilts):
            raise ValueError(
                "common-profile basis requires matching nonempty rocking angles and tilts"
            )
        support_count = max(
            2,
            int(getattr(self.cfg, "mosaicity_support_points", 5)),
        )
        if float(np.max(omegas) - np.min(omegas)) <= 1e-12:
            support_omegas = np.asarray([float(omegas[0])], dtype=float)
        else:
            support_omegas = np.linspace(
                float(np.min(omegas)),
                float(np.max(omegas)),
                support_count,
            )
        support_tilts = np.zeros((len(support_omegas), 2), dtype=float)
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(
                self.pool_inv_d2,
                row_idx,
            )
        coupling = np.abs(pool_ug) ** 2
        best_score = np.zeros(len(self.pool_hkl), dtype=float)
        minimum_abs_sg = np.full(len(self.pool_hkl), np.inf, dtype=float)
        for omega, (along_deg, transverse_deg) in zip(
            support_omegas,
            support_tilts,
            strict=True,
        ):
            sg_pool = self.sg_for_hkls_at_orientation(
                self.pool_hkl,
                float(omega),
                float(along_deg),
                float(transverse_deg),
            )
            abs_sg = np.abs(sg_pool)
            minimum_abs_sg = np.minimum(minimum_abs_sg, abs_sg)
            best_score = np.maximum(
                best_score,
                coupling / (abs_sg + 1e-5),
            )
        candidates = np.flatnonzero(
            minimum_abs_sg <= float(self.cfg.max_support_sg)
        )
        basis = [(0, 0, 0), tuple(int(value) for value in target_hkl)]
        seen = set(basis)
        eligible = {
            tuple(int(value) for value in self.pool_hkl[index])
            for index in candidates
        }
        full_union_count = int(len(seen | eligible))
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed:
            take = min(len(candidates), needed + 1)
            if take < len(candidates):
                local = np.argpartition(best_score[candidates], -take)[-take:]
                candidates = candidates[local]
            candidates = candidates[
                np.lexsort((candidates, -best_score[candidates]))
            ]
            for index in candidates:
                candidate = tuple(int(value) for value in self.pool_hkl[index])
                if candidate in seen:
                    continue
                basis.append(candidate)
                seen.add(candidate)
                if len(basis) >= int(self.cfg.n_basis):
                    break
        return np.asarray(basis, dtype=int), {
            "profile_basis_mode": "common_profile",
            "profile_basis_orientation_count": int(len(support_omegas)),
            "profile_basis_omega_min_deg": float(np.min(support_omegas)),
            "profile_basis_omega_max_deg": float(np.max(support_omegas)),
            "profile_basis_candidate_count": int(len(eligible)),
            "profile_basis_full_union_count": full_union_count,
            "profile_basis_cap": int(self.cfg.n_basis),
            "profile_basis_cap_truncated": bool(len(basis) < full_union_count),
        }

    def absorption_enabled(self):
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        return mode != "none" and (
            abs(float(getattr(self.cfg, "absorption_mu", 0.0))) > 0.0
            or abs(float(getattr(self.cfg, "absorption_frac", 0.0))) > 0.0
            or abs(float(getattr(self.cfg, "absorption_mu_g2", 0.0))) > 0.0
            or abs(float(getattr(self.cfg, "absorption_mu_theta2", 0.0))) > 0.0
        )

    def beam_absorption_mu(self, basis):
        """Beam-specific diagonal absorptive potential in Hamiltonian units.

        The returned values are added as ``+i*mu_i``. With the project
        convention ``exp(2*pi*i*H*t)``, positive ``mu_i`` damps amplitudes as
        ``exp(-2*pi*mu_i*t)``.  ``absorption_mu`` is the uniform term.  The
        optional slopes are global, auditable damping probes versus |g|^2 and
        the small Bragg angle squared; they are not row-wise fitted knobs.
        """
        inv_d2 = np.asarray(self.inv_d2_for_hkls(np.asarray(basis, dtype=int)), dtype=float)
        sin_theta = np.sqrt(np.maximum(inv_d2, 0.0)) / max(2.0 * float(self.data.k), 1e-30)
        theta = np.arcsin(np.clip(sin_theta, 0.0, 1.0))
        return (
            float(getattr(self.cfg, "absorption_mu", 0.0))
            + float(getattr(self.cfg, "absorption_mu_g2", 0.0)) * inv_d2
            + float(getattr(self.cfg, "absorption_mu_theta2", 0.0)) * theta * theta
        )

    def offdiag_absorption_eta(self, diff):
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        eta = float(getattr(self.cfg, "absorption_frac", 0.0))
        if eta == 0.0:
            return 0.0
        if mode in {"offdiag_fraction", "midgley_like", "midgley"}:
            return eta
        if mode == "offdiag_tds":
            inv_d2 = float(self.inv_d2_for_hkls(diff))
            b_tds = max(float(getattr(self.cfg, "absorption_tds_b", 0.0)), 0.0)
            return eta * (1.0 - math.exp(-0.25 * b_tds * inv_d2))
        return 0.0

    def offdiag_absorption_eta_array(self, diffs):
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        eta = float(getattr(self.cfg, "absorption_frac", 0.0))
        diffs = np.asarray(diffs, dtype=int)
        if eta == 0.0:
            return np.zeros(diffs.shape[:-1], dtype=float)
        if mode in {"offdiag_fraction", "midgley_like", "midgley"}:
            return np.full(diffs.shape[:-1], eta, dtype=float)
        if mode == "offdiag_tds":
            inv_d2 = self.inv_d2_for_hkls(diffs.reshape(-1, 3)).reshape(diffs.shape[:-1])
            b_tds = max(float(getattr(self.cfg, "absorption_tds_b", 0.0)), 0.0)
            return eta * (1.0 - np.exp(-0.25 * b_tds * inv_d2))
        return np.zeros(diffs.shape[:-1], dtype=float)

    def hamiltonian_static_part(self, basis, absorb: bool = True, row_idx=None):
        """Return the row-static Hamiltonian terms for a fixed beam basis.

        For a given row and support basis the off-diagonal Fourier-potential
        couplings do not depend on the target-Sg quadrature point.  Only the
        diagonal excitation errors change as the crystal is rocked through the
        profile.  Keeping this static part separate avoids rebuilding the same
        dense U(h_j-h_i) block for every profile sample.
        """

        n = len(basis)
        H = np.zeros((n, n), dtype=np.complex128)
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        if absorb and mode in {"mean", "midgley_like", "midgley"}:
            # Positive imaginary diagonal gives exp(2*pi*i*(S+i*mu)*t),
            # i.e. uniform elastic-beam damping exp(-2*pi*mu*t).
            H[np.diag_indices(n)] += 1j * float(getattr(self.cfg, "absorption_mu", 0.0))
        elif absorb and mode == "beam_dependent":
            H[np.diag_indices(n)] += 1j * self.beam_absorption_mu(basis)
        convention = str(getattr(self.cfg, "matrix_convention", "j_minus_i") or "j_minus_i")
        if convention not in {"j_minus_i", "i_minus_j"}:
            raise ValueError(
                f"unknown matrix_convention={convention!r}; expected j_minus_i or i_minus_j"
            )
        # Production signed-row convention: use U(h_j - h_i).
        # The alternate sign is kept as an auditable raw-forward-model test.
        diffs = basis[None, :, :] - basis[:, None, :] if convention == "j_minus_i" else basis[:, None, :] - basis[None, :, :]
        u_mat = self.ug_array(diffs) if row_idx is None else self.ug_for_row_array(diffs, row_idx)
        if absorb:
            eta = self.offdiag_absorption_eta_array(diffs)
            if np.any(eta):
                # Absorptive Fourier coefficients are represented as a
                # same-phase imaginary potential. V(-g)=conj(U_g)+i*eta*conj(U_g),
                # not conj(V(g)), so the matrix is intentionally non-Hermitian.
                # The midgley_like mode is an optical-potential diagnostic
                # following Colmey, Doherty, Malik, and Midgley's absorption
                # analysis for electron-diffraction Bloch row modelling; it is gated rather
                # than assumed because absorption can be negligible for many
                # routine light-atom integrated-intensity regularizations.
                u_mat = u_mat + 1j * eta * u_mat
        np.fill_diagonal(u_mat, 0.0 + 0.0j)
        H += u_mat / (2.0 * self.data.k)
        if absorb and self.absorption_enabled():
            validate_passive_absorption(H, mode=mode, row_index=row_idx)
        return H

    def hamiltonian_from_static(self, static_H, sg_basis):
        H = np.array(static_H, dtype=np.complex128, copy=True)
        diag = np.diag_indices(H.shape[0])
        H[diag] += np.asarray(sg_basis, dtype=float)
        return H

    def hamiltonian(self, basis, sg_basis, absorb: bool = True, row_idx=None):
        static_H = self.hamiltonian_static_part(basis, absorb=absorb, row_idx=row_idx)
        return self.hamiltonian_from_static(static_H, sg_basis)

    @staticmethod
    def _nonhermitian_spectral_target_coupling(H):
        """Return a stable eigensystem coupling, or ``None`` when direct expm is safer."""

        H = np.asarray(H, dtype=np.complex128)
        try:
            vals, vecs = eig(H)
            condition = float(np.linalg.cond(vecs))
            if not math.isfinite(condition) or condition > 1e10:
                return None
            inv_vecs = np.linalg.inv(vecs)
            reconstruction = vecs @ np.diag(vals) @ inv_vecs
            relative_error = float(
                np.linalg.norm(reconstruction - H)
                / max(np.linalg.norm(H), np.finfo(float).eps)
            )
            if not math.isfinite(relative_error) or relative_error > 1e-8:
                return None
        except np.linalg.LinAlgError:
            return None
        coupling = vecs[1, :] * inv_vecs[:, 0]
        return np.asarray(vals, dtype=np.complex128), np.asarray(coupling, dtype=np.complex128)

    @staticmethod
    def _uniform_scalar_absorption_decomposition(H):
        """Return ``(Hermitian H0, mu)`` when ``H = H0 + i*mu*I`` exactly.

        A uniform imaginary potential commutes with the elastic Hamiltonian.
        Its wave-amplitude effect is the scalar ``exp(-2*pi*mu*t)`` and its
        intensity effect is ``exp(-4*pi*mu*t)``.  Recognizing this case avoids
        a general complex eigensolve without approximating the physics.
        """

        matrix = np.asarray(H, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return None
        diagonal_imaginary = np.imag(np.diag(matrix))
        if not np.all(np.isfinite(diagonal_imaginary)):
            return None
        mu = float(np.mean(diagonal_imaginary))
        elastic = matrix.copy()
        elastic[np.diag_indices(len(elastic))] -= 1j * mu
        if not np.allclose(elastic, elastic.conj().T, atol=1e-12, rtol=0.0):
            return None
        return elastic, mu

    @staticmethod
    def _nonhermitian_target_intensities(H, thicknesses):
        """Propagate a non-Hermitian matrix, falling back for defective eigensystems."""

        H = np.asarray(H, dtype=np.complex128)
        thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=float))
        spectral = XDSBlochPort._nonhermitian_spectral_target_coupling(H)

        if spectral is None:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            return np.asarray(
                [abs((expm((2j * np.pi * float(t)) * H) @ psi0)[1]) ** 2 for t in thicknesses],
                dtype=float,
            )

        vals, coupling = spectral
        phase = np.exp((2j * np.pi) * vals[:, None] * thicknesses[None, :])
        return np.asarray(np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2, dtype=float)

    def propagate_direct_to_target(self, H, thickness):
        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend in BETHE_BACKENDS:
            H = promoted_bethe_effective_hamiltonian(H, self.cfg)
            backend = "scipy"
        if backend in {"scipy", "numpy_scipy_cpu"}:
            if self.absorption_enabled():
                uniform = self._uniform_scalar_absorption_decomposition(H)
                if uniform is None:
                    return float(
                        self._nonhermitian_target_intensities(H, [thickness])[0]
                    )
                elastic, mu = uniform
                vals, vecs = eigh(elastic)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
                phase = np.exp(2j * np.pi * vals * thickness)
                return float(
                    abs(np.sum(coupling * phase)) ** 2
                    * math.exp(-4.0 * math.pi * mu * float(thickness))
                )
            else:
                vals, vecs = eigh(H)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
            phase = np.exp(2j * np.pi * vals * thickness)
            return abs(np.sum(coupling * phase)) ** 2
        if backend in {"scipy_expm", "expm"}:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            psi = expm((2j * np.pi * float(thickness)) * H) @ psi0
            return float(abs(psi[1]) ** 2)
        if backend in {"scipy_expm_multiply", "expm_multiply"}:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            psi = expm_multiply((2j * np.pi * float(thickness)) * H, psi0)
            return float(abs(psi[1]) ** 2)
        if backend in ALL_JAX_BACKENDS:
            if _jax_backend_uses_real_expm(backend):
                vals = _jax_real_expm_propagate_batch(
                    np.asarray(H, dtype=np.complex128)[None, :, :],
                    np.asarray([float(thickness)], dtype=float),
                    np.asarray([1.0], dtype=float),
                    backend,
                    int(getattr(self.cfg, "jax_batch_size", 0)),
                )
                return float(vals[0])
            try:
                _jax, jnp = _jax_modules(_jax_module_key(backend))
                Hj = jnp.asarray(H, dtype=jnp.complex128)
                if self.absorption_enabled():
                    vals, vecs = jnp.linalg.eig(Hj)
                    inv_vecs = jnp.linalg.inv(vecs)
                    coupling = vecs[1, :] * inv_vecs[:, 0]
                else:
                    vals, vecs = jnp.linalg.eigh(Hj)
                    coupling = vecs[1, :] * jnp.conj(vecs[0, :])
                phase = jnp.exp((2j * np.pi * float(thickness)) * vals)
                intensity = jnp.real(jnp.abs(jnp.sum(coupling * phase)) ** 2)
                return float(np.asarray(intensity))
            except Exception as exc:
                raise NotImplementedError(
                    f"compute_backend={backend!r} failed in the JAX Bloch eigensolver. "
                    "This backend must match SciPy complex eig/eigh before it can be used "
                    f"for GROND production runs. Error: {type(exc).__name__}: {exc}"
                ) from exc
        raise NotImplementedError(
            f"unknown compute_backend={backend!r}; expected one of {COMPUTE_BACKEND_HELP}"
        )

    def propagate_direct_to_target_thickness_average(self, H):
        """Propagate one Bloch matrix and average over the configured thickness quadrature."""

        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend in BETHE_BACKENDS:
            H = promoted_bethe_effective_hamiltonian(H, self.cfg)
            backend = "scipy"
        if backend in {"scipy", "numpy_scipy_cpu"}:
            if self.absorption_enabled():
                uniform = self._uniform_scalar_absorption_decomposition(H)
                if uniform is None:
                    per_t = self._nonhermitian_target_intensities(H, self.thick)
                    return float(np.sum(self.thick_w * per_t))
                elastic, mu = uniform
                vals, vecs = eigh(elastic)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
                phase = np.exp(
                    (2j * np.pi) * vals[:, None] * self.thick[None, :]
                )
                per_t = np.abs(
                    np.sum(coupling[:, None] * phase, axis=0)
                ) ** 2
                per_t *= np.exp(-4.0 * np.pi * mu * self.thick)
                return float(np.sum(self.thick_w * per_t))
            else:
                vals, vecs = eigh(H)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
            phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
            per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
            return float(np.sum(self.thick_w * per_t))
        if backend in {"scipy_expm", "expm", "scipy_expm_multiply", "expm_multiply"}:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            per_t = []
            for t in self.thick:
                if backend in {"scipy_expm", "expm"}:
                    psi = expm((2j * np.pi * float(t)) * H) @ psi0
                else:
                    psi = expm_multiply((2j * np.pi * float(t)) * H, psi0)
                per_t.append(float(abs(psi[1]) ** 2))
            return float(np.sum(self.thick_w * np.asarray(per_t, dtype=float)))
        if backend in ALL_JAX_BACKENDS:
            vals = _jax_propagate_batch(
                np.asarray(H, dtype=np.complex128)[None, :, :],
                self.thick,
                self.thick_w,
                backend,
                self.absorption_enabled(),
                int(getattr(self.cfg, "jax_batch_size", 0)),
            )
            return float(vals[0])
        raise NotImplementedError(
            f"unknown compute_backend={backend!r}; expected one of {COMPUTE_BACKEND_HELP}"
        )

    def propagate_direct_to_target_thickness_average_pair(self, H):
        """Return original- and inverse-hand intensities from one eigensystem.

        For elastic scattering the inverted-coordinate Hamiltonian is ``H*``.
        With a real incident direct-beam vector, propagation through ``H*`` at
        positive thickness has the same intensity as propagation through ``H``
        at negative thickness.  The two signs therefore share one Hermitian
        eigendecomposition.  Absorptive optical potentials break this identity
        and are rejected rather than silently approximated.
        """

        if self.absorption_enabled():
            raise ValueError(
                "paired inverse-hand propagation requires absorption_mode=none and zero "
                "absorptive coefficients; run explicit coordinate-inverted branches when "
                "absorption is active"
            )
        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend not in {"scipy", "numpy_scipy_cpu"}:
            raise NotImplementedError(
                "paired inverse-hand propagation requires the full audited SciPy Hermitian "
                "eigensolver; choose --compute-backend scipy"
            )
        vals, vecs = eigh(np.asarray(H, dtype=np.complex128))
        coupling = vecs[1, :] * np.conj(vecs[0, :])
        phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
        forward = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
        reverse = np.abs(np.sum(coupling[:, None] * np.conj(phase), axis=0)) ** 2
        return (
            float(np.sum(self.thick_w * forward)),
            float(np.sum(self.thick_w * reverse)),
        )

    def pets_exact_orientation_samples(self, row_idx):
        """Return the PETS-owned ``Rx(delta) @ A_zone`` rocking path."""

        centers = getattr(self.data, "pets_exact_a_center", None)
        semiangles = getattr(self.data, "pets_integration_semiangle_deg", None)
        if centers is None or semiangles is None:
            raise ValueError("PETS exact orientation samples require PETS geometry metadata")
        ordinal = int(round(float(self.data.zcal[row_idx]))) - 1
        if not 0 <= ordinal < len(centers):
            raise ValueError(
                f"PETS row {row_idx} uses zone ordinal {ordinal + 1}, outside 1..{len(centers)}"
            )
        center = np.asarray(centers[ordinal], dtype=float)
        semiangle = float(semiangles[ordinal])
        count = max(1, int(self.cfg.n_points))
        offsets = (
            np.asarray([0.0], dtype=float)
            if count == 1
            else np.linspace(-semiangle, semiangle, count)
        )
        matrices = np.asarray(
            [
                _rotation_matrix_axis_angle(np.asarray([1.0, 0.0, 0.0]), offset)
                @ center
                for offset in offsets
            ],
            dtype=float,
        )
        weights = np.ones(len(offsets), dtype=float)
        accumulation = str(
            getattr(self.cfg, "profile_accumulation", "sum") or "sum"
        ).lower()
        if accumulation == "mean":
            weights /= float(np.sum(weights))
        elif accumulation != "sum":
            raise ValueError(f"unknown profile_accumulation={accumulation!r}")
        return matrices, offsets, weights, semiangle, ordinal

    def pets_exact_sg_for_hkls(self, hkls, orientation):
        g = (np.asarray(orientation, dtype=float) @ np.asarray(hkls, dtype=float).T).T
        return (
            self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)
        ) / (2.0 * self.data.k)

    def pets_exact_select_common_basis(self, target_hkl, orientations, row_idx):
        """Select one scored beam union over the full PETS rocking wedge."""

        pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(
            self.pool_inv_d2, row_idx
        )
        coupling = np.abs(pool_ug) ** 2
        best_score = np.zeros(len(self.pool_hkl), dtype=float)
        minimum_abs_sg = np.full(len(self.pool_hkl), np.inf, dtype=float)
        for orientation in np.asarray(orientations, dtype=float):
            abs_sg = np.abs(
                self.pets_exact_sg_for_hkls(self.pool_hkl, orientation)
            )
            minimum_abs_sg = np.minimum(minimum_abs_sg, abs_sg)
            best_score = np.maximum(best_score, coupling / (abs_sg + 1e-5))
        candidates = np.flatnonzero(
            minimum_abs_sg <= float(self.cfg.max_support_sg)
        )
        basis = [(0, 0, 0), tuple(int(value) for value in target_hkl)]
        seen = set(basis)
        eligible_candidates = {
            tuple(int(value) for value in self.pool_hkl[index])
            for index in candidates
        }
        eligible_candidate_count = int(len(candidates))
        full_union_count = int(len(seen | eligible_candidates))
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed:
            take = min(len(candidates), needed + 1)
            if take < len(candidates):
                local = np.argpartition(best_score[candidates], -take)[-take:]
                candidates = candidates[local]
            candidates = candidates[
                np.lexsort((candidates, -best_score[candidates]))
            ]
            for index in candidates:
                candidate = tuple(int(value) for value in self.pool_hkl[index])
                if candidate in seen:
                    continue
                basis.append(candidate)
                seen.add(candidate)
                if len(basis) >= int(self.cfg.n_basis):
                    break
        return np.asarray(basis, dtype=int), {
            "pets_basis_mode": "one_common_scored_union_over_full_recorded_wedge",
            "pets_basis_orientation_count": int(len(orientations)),
            "pets_basis_candidate_count": eligible_candidate_count,
            "pets_basis_full_union_count": full_union_count,
            "pets_basis_ranked_candidate_count": int(len(candidates)),
            "pets_basis_cap": int(self.cfg.n_basis),
            "pets_basis_cap_truncated": bool(len(basis) < full_union_count),
        }

    def pets_exact_row_prediction_plan(self, row_idx):
        """Build one exact PETS direct-angle row plan without Sg root solving."""

        hkl = np.asarray(self.data.hkl[row_idx], dtype=int)
        orientations, offsets, profile_weights, semiangle, ordinal = (
            self.pets_exact_orientation_samples(row_idx)
        )
        center = np.asarray(self.data.pets_exact_a_center[ordinal], dtype=float)
        center_sg = float(
            self.pets_exact_sg_for_hkls(np.asarray([hkl]), center)[0]
        )
        derivative_step = 1e-4
        plus = _rotation_matrix_axis_angle(
            np.asarray([1.0, 0.0, 0.0]), derivative_step
        ) @ center
        minus = _rotation_matrix_axis_angle(
            np.asarray([1.0, 0.0, 0.0]), -derivative_step
        ) @ center
        sg_derivative = float(
            (
                self.pets_exact_sg_for_hkls(np.asarray([hkl]), plus)[0]
                - self.pets_exact_sg_for_hkls(np.asarray([hkl]), minus)[0]
            )
            / (2.0 * derivative_step)
        )
        sg_window = 2.0 * abs(sg_derivative) * semiangle
        basis, basis_meta = self.pets_exact_select_common_basis(
            hkl, orientations, row_idx
        )
        if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(
            basis[1], hkl
        ):
            raise RuntimeError(
                "PETS frozen basis must start with direct beam then exact signed target HKL"
            )
        dynamic_static = self.hamiltonian_static_part(
            basis, absorb=True, row_idx=row_idx
        )
        born_static = (
            self.hamiltonian_static_part(basis, absorb=False, row_idx=row_idx)
            if self.absorption_enabled()
            else dynamic_static
        )
        h10_born = born_static[1, 0]
        dynamic_matrices = []
        born_values = []
        born_deltas = []
        for orientation in orientations:
            sg_basis = self.pets_exact_sg_for_hkls(basis, orientation)
            dynamic_matrices.append(
                self.hamiltonian_from_static(dynamic_static, sg_basis)
            )
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            born_deltas.append(delta)
            sinc = np.sinc(delta * self.thick)
            born_t = (
                (2.0 * math.pi) ** 2
                * (abs(h10_born) ** 2)
                * (self.thick**2)
                * (sinc**2)
            )
            born_values.append(float(np.sum(self.thick_w * born_t)))
        return {
            "_dyn_matrices": dynamic_matrices,
            "_profile_weights": np.asarray(profile_weights, dtype=float),
            "_born_vals": np.asarray(born_values, dtype=float),
            "_born_h10_abs2": float(abs(h10_born) ** 2),
            "_born_deltas": np.asarray(born_deltas, dtype=float),
            "_sg_resid": np.asarray([], dtype=float),
            "basis_size": int(len(basis)),
            "support_hkls": [
                [int(value) for value in row] for row in basis.tolist()
            ],
            "center_sg_inv_A": center_sg,
            "omega_center_deg": float("nan"),
            "omega_bragg_deg": float("nan"),
            "omega_bragg_minus_center_deg": float("nan"),
            "sg_window_inv_A": sg_window,
            "dose_delta_b_A2": float(self.row_delta_b(row_idx)),
            "thickness_model": str(
                getattr(self.cfg, "thickness_model", "uniform_gradient")
            ),
            "thickness_sigma_frac": float(
                getattr(self.cfg, "thickness_sigma_frac", 0.0)
            ),
            "sg_profile_coordinate": "pets_exact_direct_angle",
            "sg_profile_quadrature": "uniform_linspace",
            "sg_profile_accumulation": str(
                getattr(self.cfg, "profile_accumulation", "sum")
            ),
            "sg_profile_angle_half_width_deg": semiangle,
            "sg_profile_points_total": int(len(orientations)),
            "sg_profile_weight_sum": float(np.sum(profile_weights)),
            "sg_dSg_domega_inv_A_per_deg": sg_derivative,
            "sg_dSg_dframe_inv_A": sg_derivative
            * float(self.data.oscillation_range),
            "sg_window_raw_inv_A": sg_window,
            "sg_window_half_frames": semiangle
            / max(abs(float(self.data.oscillation_range)), 1e-30),
            "sg_window_clipped": False,
            "sg_jitter_sigma_inv_A": 0.0,
            "sg_jitter_d_inv_A": float("nan"),
            "sg_jitter_frame_fraction": float("nan"),
            "sg_jitter_kernel_points": 1,
            "sg_profile_same_branch_continuation_used": False,
            "sg_profile_same_branch_continuation_segments": 1,
            "pets_zone_ordinal": int(ordinal + 1),
            "pets_rocking_offsets_deg": [float(value) for value in offsets],
            **basis_meta,
        }

    def row_prediction_plan(self, row_idx):
        """Build one row's Bloch matrices and metadata before eigensolver evaluation."""

        if getattr(self.data, "pets_exact_a_center", None) is not None:
            return self.pets_exact_row_prediction_plan(row_idx)

        hkl = self.data.hkl[row_idx]
        zcal = self.data.zcal[row_idx]
        omega_center = self.omega_from_z(zcal)
        center_sg = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega_center)[0])
        omega_bragg, omega_bragg_root_available = self.row_diagnostic_omega_bragg(
            hkl, omega_center
        )
        sg_window_i, sg_width_meta = self.row_sg_window(row_idx, hkl, omega_center)
        (
            solved_omegas,
            mosaic_tilts,
            profile_weights,
            sg_resid,
            jitter_meta,
            mosaicity_meta,
        ) = self.row_profile_orientation_points(
            row_idx, hkl, omega_center, center_sg, sg_window_i
        )
        basis, cone_basis_meta = self.select_basis_for_profile(
            hkl,
            omega_center,
            sg_window_i,
            solved_omegas,
            mosaic_tilts,
            row_idx=row_idx,
        )
        if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(basis[1], np.asarray(hkl, dtype=int)):
            raise RuntimeError("Frozen kernel basis contract violated: basis must start with direct beam then exact signed target hkl.")
        H_dyn_static = self.hamiltonian_static_part(basis, absorb=True, row_idx=row_idx)
        if self.absorption_enabled():
            H_born_static = self.hamiltonian_static_part(basis, absorb=False, row_idx=row_idx)
        else:
            H_born_static = H_dyn_static
        h10_born = H_born_static[1, 0]
        dyn_matrices, born_vals, born_deltas = [], [], []
        for omega, (along_deg, transverse_deg) in zip(
            solved_omegas, mosaic_tilts, strict=True
        ):
            sg_basis = self.sg_for_hkls_at_orientation(
                basis, omega, along_deg, transverse_deg
            )
            H = self.hamiltonian_from_static(H_dyn_static, sg_basis)
            dyn_matrices.append(H)
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            born_deltas.append(delta)
            sinc = np.sinc(delta * self.thick)
            born_t = (2.0 * math.pi) ** 2 * (abs(h10_born) ** 2) * (self.thick**2) * (sinc**2)
            born_vals.append(float(np.sum(self.thick_w * born_t)))
        return {
            "_dyn_matrices": dyn_matrices,
            "_profile_weights": np.asarray(profile_weights, dtype=float),
            "_born_vals": np.asarray(born_vals, dtype=float),
            "_born_h10_abs2": float(abs(h10_born) ** 2),
            "_born_deltas": np.asarray(born_deltas, dtype=float),
            "_sg_resid": np.asarray(sg_resid, dtype=float),
            "basis_size": int(len(basis)),
            "support_hkls": [[int(v) for v in row] for row in basis.tolist()],
            "center_sg_inv_A": center_sg,
            "omega_center_deg": float(omega_center),
            "omega_bragg_deg": float(omega_bragg),
            "omega_bragg_minus_center_deg": float(omega_bragg - omega_center),
            "omega_bragg_root_available": bool(omega_bragg_root_available),
            "sg_window_inv_A": float(sg_window_i),
            "dose_delta_b_A2": float(self.row_delta_b(row_idx)),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
            **sg_width_meta,
            **jitter_meta,
            **mosaicity_meta,
            **cone_basis_meta,
        }

    def finish_row_prediction_plan(self, plan, dyn_vals):
        profile_weights = np.asarray(plan["_profile_weights"], dtype=float)
        born_vals = np.asarray(plan["_born_vals"], dtype=float)
        sg_resid = np.asarray(plan["_sg_resid"], dtype=float)
        out = {k: v for k, v in plan.items() if not k.startswith("_")}
        out.update(
            {
                "dyn": float(np.sum(profile_weights * np.asarray(dyn_vals, dtype=float))),
                "born": float(np.sum(profile_weights * born_vals)),
                "max_sg_solve_residual": float(np.max(np.abs(sg_resid))) if len(sg_resid) else 0.0,
            }
        )
        return out

    def row_prediction(self, row_idx):
        plan = self.row_prediction_plan(row_idx)
        if bool(getattr(self.cfg, "paired_inverse_hand", False)):
            pairs = [
                self.propagate_direct_to_target_thickness_average_pair(H)
                for H in plan["_dyn_matrices"]
            ]
            out = self.finish_row_prediction_plan(plan, [pair[0] for pair in pairs])
            weights = np.asarray(plan["_profile_weights"], dtype=float)
            out["dyn_inverse"] = float(
                np.sum(weights * np.asarray([pair[1] for pair in pairs], dtype=float))
            )
            return out
        dyn_vals = [self.propagate_direct_to_target_thickness_average(H) for H in plan["_dyn_matrices"]]
        return self.finish_row_prediction_plan(plan, dyn_vals)

    def row_plan_cache_mode(self) -> str:
        """Return the requested frozen row-plan cache mode."""

        mode = str(getattr(self.cfg, "row_plan_cache", "off") or "off").lower()
        if mode in {"0", "false", "no", "none"}:
            return "off"
        if mode in {"1", "true", "yes", "on"}:
            return "readwrite"
        if mode not in {"off", "exact", "readwrite", "rebuild"}:
            raise ValueError(
                f"unknown row_plan_cache={mode!r}; expected off, exact, readwrite, or rebuild"
            )
        return mode

    def row_plan_cache_key(self, indices):
        """Return a geometry/config cache key for frozen row plans.

        ``exact`` keys the cache to the current structural model.  It is safe
        for repeated nuisance calculations at fixed coordinates because only
        thickness, scale, or other non-basis terms change.  ``readwrite``
        intentionally omits the current model and retains the older opt-in
        frozen-basis behavior for native finite-difference experiments.
        """

        indices = np.asarray(indices, dtype=np.int64)
        cfg_fields = {
            "cache_schema": ROW_PLAN_CACHE_SCHEMA,
            "hkl_sha256": sha256(self.data.path),
            "hkl_transform": self.data.hkl_transform_spec,
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "n_basis": int(self.cfg.n_basis),
            "n_points": int(self.cfg.n_points),
            "sg_window": float(self.cfg.sg_window),
            "sg_window_mode": str(self.cfg.sg_window_mode),
            "sg_window_min": float(self.cfg.sg_window_min),
            "sg_window_max": float(self.cfg.sg_window_max),
            "sg_window_scale": float(self.cfg.sg_window_scale),
            "sg_window_frame_floor": float(self.cfg.sg_window_frame_floor),
            "sg_jitter_mode": str(self.cfg.sg_jitter_mode),
            "sg_jitter_sigma0": float(self.cfg.sg_jitter_sigma0),
            "sg_jitter_sigma_d_inv": float(self.cfg.sg_jitter_sigma_d_inv),
            "sg_jitter_sigma_frame": float(self.cfg.sg_jitter_sigma_frame),
            "sg_jitter_points": int(self.cfg.sg_jitter_points),
            "sg_jitter_student_df": float(self.cfg.sg_jitter_student_df),
            "sg_jitter_width_factor": float(self.cfg.sg_jitter_width_factor),
            "mosaicity_mode": str(self.cfg.mosaicity_mode),
            "mosaicity_sigma_deg": float(self.cfg.mosaicity_sigma_deg),
            "mosaicity_max_sigma_deg": float(self.cfg.mosaicity_max_sigma_deg),
            "mosaicity_points": int(self.cfg.mosaicity_points),
            "mosaicity_support_sigma_deg": float(
                self.cfg.mosaicity_support_sigma_deg
            ),
            "mosaicity_support_points": int(self.cfg.mosaicity_support_points),
            "mosaicity_support_width_sigma": float(
                self.cfg.mosaicity_support_width_sigma
            ),
            "profile_coordinate": str(self.cfg.profile_coordinate),
            "profile_basis_mode": str(self.cfg.profile_basis_mode),
            "profile_angle_max_half_width_deg": float(
                self.cfg.profile_angle_max_half_width_deg
            ),
            "geometry_path_sha256": (
                str(self.geometry_path["sha256"])
                if self.geometry_path is not None
                else ""
            ),
            "profile_quadrature": str(self.cfg.profile_quadrature),
            "profile_accumulation": str(self.cfg.profile_accumulation),
            "topup_d_min": float(self.cfg.topup_d_min),
            "ug_d_min": float(self.cfg.ug_d_min),
            "max_support_sg": float(self.cfg.max_support_sg),
            "frame_offset": float(self.cfg.frame_offset),
            "matrix_convention": str(self.cfg.matrix_convention),
            "dose_b_mode": str(self.cfg.dose_b_mode),
            "dose_b_per_frame": float(self.cfg.dose_b_per_frame),
            "dose_b_ref_frame": float(self.dose_ref_frame),
            "cell": [float(v) for v in self.model["cell"]],
            "wavelength": float(self.data.wavelength),
            "scan_matrix_sha256": (
                sha256(self.data.a_scan_points_path)
                if self.data.a_scan_points_path is not None and self.data.a_scan_points_path.exists()
                else ""
            ),
            "pets_exact_geometry_sha256": (
                sha256(self.data.pets_exact_geometry_path)
                if self.data.pets_exact_geometry_path is not None
                and self.data.pets_exact_geometry_path.exists()
                else ""
            ),
        }
        if self.row_plan_cache_mode() == "exact":
            cfg_fields.update(
                {
                    "basis_model_content_sha256": model_content_hash(self.model),
                    "basis_res_sha256": self.res_hash,
                    "basis_policy": "exact_fixed_model",
                }
            )
        text = json.dumps(cfg_fields, sort_keys=True, allow_nan=True)
        return hashlib.sha256(text.encode()).hexdigest()[:20], cfg_fields

    def row_plan_cache_path(self, key: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("row-plan cache requires cache_dir")
        return self.cache_dir / "row_plans" / f"row_plan_{key}.npz"

    def build_row_plan_cache(self, indices):
        """Build immutable row geometry and basis plans for later reuse."""

        indices = np.asarray(indices, dtype=int)
        key, meta = self.row_plan_cache_key(indices)
        meta = dict(meta)
        meta.update(
            {
                "key": key,
                "basis_model_content_sha256": model_content_hash(self.model),
                "basis_res_sha256": self.res_hash,
                "basis_policy": (
                    "exact_fixed_model"
                    if self.row_plan_cache_mode() == "exact"
                    else "frozen_from_cache_builder_model"
                ),
            }
        )
        unique_diff_index: dict[tuple[int, int, int], int] = {}
        unique_diffs: list[tuple[int, int, int]] = []
        basis_offsets = [0]
        basis_flat = []
        profile_offsets = [0]
        profile_weights_flat = []
        profile_omega_flat = []
        profile_tilts_flat = []
        sg_offsets = [0]
        sg_basis_flat = []
        diff_offsets = [0]
        diff_idx_flat = []
        center_sg = []
        omega_center = []
        omega_bragg = []
        omega_delta = []
        omega_bragg_root_available = []
        sg_window = []
        sg_domega = []
        sg_dframe = []
        sg_window_raw = []
        sg_window_half_frames = []
        sg_window_clipped = []
        sg_jitter_sigma = []
        sg_jitter_d_inv = []
        sg_jitter_frame_fraction = []
        sg_jitter_kernel_points = []
        sg_profile_points_total = []
        sg_profile_weight_sum = []
        sg_profile_same_branch_continuation_used = []
        sg_profile_same_branch_continuation_segments = []
        sg_residual_max = []
        dose_delta_b = []
        convention = str(getattr(self.cfg, "matrix_convention", "j_minus_i") or "j_minus_i")
        if convention not in {"j_minus_i", "i_minus_j"}:
            raise ValueError(f"unknown matrix_convention={convention!r}; expected j_minus_i or i_minus_j")

        t0 = time.perf_counter()
        for pos, row_idx in enumerate(indices):
            row_idx = int(row_idx)
            hkl = self.data.hkl[row_idx]
            if getattr(self.data, "pets_exact_a_center", None) is not None:
                orientations, profile_angles, profile_weights, semiangle, ordinal = (
                    self.pets_exact_orientation_samples(row_idx)
                )
                profile_omegas = np.asarray(profile_angles, dtype=float)
                profile_tilts = np.zeros((len(profile_omegas), 2), dtype=float)
                center = np.asarray(
                    self.data.pets_exact_a_center[ordinal], dtype=float
                )
                c_sg = float(
                    self.pets_exact_sg_for_hkls(
                        np.asarray([hkl], dtype=int), center
                    )[0]
                )
                derivative_step = 1e-4
                plus = _rotation_matrix_axis_angle(
                    np.asarray([1.0, 0.0, 0.0]), derivative_step
                ) @ center
                minus = _rotation_matrix_axis_angle(
                    np.asarray([1.0, 0.0, 0.0]), -derivative_step
                ) @ center
                derivative = float(
                    (
                        self.pets_exact_sg_for_hkls(
                            np.asarray([hkl], dtype=int), plus
                        )[0]
                        - self.pets_exact_sg_for_hkls(
                            np.asarray([hkl], dtype=int), minus
                        )[0]
                    )
                    / (2.0 * derivative_step)
                )
                sg_window_i = 2.0 * abs(derivative) * semiangle
                basis, _pets_basis_meta = self.pets_exact_select_common_basis(
                    hkl, orientations, row_idx
                )
                sg_basis_rows = [
                    np.asarray(
                        self.pets_exact_sg_for_hkls(basis, orientation),
                        dtype=float,
                    )
                    for orientation in orientations
                ]
                # These legacy fields are not used for exact PETS propagation,
                # but the cache validator requires finite fixed-width arrays.
                omega_c = 0.0
                omega_b = 0.0
                omega_b_available = False
                sg_resid_values = np.asarray([], dtype=float)
                sg_width_meta = {
                    "sg_dSg_domega_inv_A_per_deg": derivative,
                    "sg_dSg_dframe_inv_A": derivative
                    * float(self.data.oscillation_range),
                    "sg_window_raw_inv_A": sg_window_i,
                    "sg_window_half_frames": semiangle
                    / max(abs(float(self.data.oscillation_range)), 1e-30),
                    "sg_window_clipped": False,
                }
                jitter_meta = {
                    "sg_jitter_sigma_inv_A": 0.0,
                    "sg_jitter_d_inv_A": float("nan"),
                    "sg_jitter_frame_fraction": float("nan"),
                    "sg_jitter_kernel_points": 1,
                    "sg_profile_points_total": int(len(orientations)),
                    "sg_profile_weight_sum": float(np.sum(profile_weights)),
                    "sg_profile_same_branch_continuation_used": False,
                    "sg_profile_same_branch_continuation_segments": 1,
                }
            else:
                zcal = self.data.zcal[row_idx]
                omega_c = self.omega_from_z(zcal)
                c_sg = float(
                    self.sg_for_hkls(np.asarray([hkl], dtype=int), omega_c)[0]
                )
                omega_b, omega_b_available = self.row_diagnostic_omega_bragg(
                    hkl, omega_c
                )
                sg_window_i, sg_width_meta = self.row_sg_window(
                    row_idx, hkl, omega_c
                )
                (
                    solved_omegas,
                    mosaic_tilts,
                    profile_weights,
                    sg_resid_values,
                    jitter_meta,
                    _mosaicity_meta,
                ) = self.row_profile_orientation_points(
                    row_idx, hkl, omega_c, c_sg, sg_window_i
                )
                profile_omegas = np.asarray(solved_omegas, dtype=float)
                profile_tilts = np.asarray(mosaic_tilts, dtype=float).reshape(-1, 2)
                basis, _cone_basis_meta = self.select_basis_for_profile(
                    hkl,
                    omega_c,
                    sg_window_i,
                    solved_omegas,
                    mosaic_tilts,
                    row_idx=row_idx,
                )
                sg_basis_rows = []
                for omega, (along_deg, transverse_deg) in zip(
                    solved_omegas, mosaic_tilts, strict=True
                ):
                    sg_basis = self.sg_for_hkls_at_orientation(
                        basis, omega, along_deg, transverse_deg
                    )
                    sg_basis_rows.append(np.asarray(sg_basis, dtype=float))
            if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(basis[1], np.asarray(hkl, dtype=int)):
                raise RuntimeError("Frozen row-plan basis contract violated: basis must start with direct beam then exact signed target hkl.")

            diffs = basis[None, :, :] - basis[:, None, :] if convention == "j_minus_i" else basis[:, None, :] - basis[None, :, :]
            row_diff_idx = []
            for diff in diffs.reshape(-1, 3):
                tup = tuple(int(v) for v in diff)
                idx = unique_diff_index.get(tup)
                if idx is None:
                    idx = len(unique_diffs)
                    unique_diff_index[tup] = idx
                    unique_diffs.append(tup)
                row_diff_idx.append(idx)

            basis_flat.extend(tuple(int(v) for v in b) for b in basis)
            basis_offsets.append(len(basis_flat))
            if not (
                len(profile_weights) == len(profile_omegas) == len(profile_tilts)
            ):
                raise RuntimeError(
                    "profile weights, orientations, and tilts must have identical lengths"
                )
            profile_weights_flat.extend(float(v) for v in profile_weights)
            profile_omega_flat.extend(float(value) for value in profile_omegas)
            profile_tilts_flat.extend(
                (float(row[0]), float(row[1])) for row in profile_tilts
            )
            profile_offsets.append(len(profile_weights_flat))
            sg_basis_arr = np.asarray(sg_basis_rows, dtype=float)
            sg_basis_flat.extend(float(v) for v in sg_basis_arr.reshape(-1))
            sg_offsets.append(len(sg_basis_flat))
            diff_idx_flat.extend(int(v) for v in row_diff_idx)
            diff_offsets.append(len(diff_idx_flat))

            center_sg.append(c_sg)
            omega_center.append(float(omega_c))
            omega_bragg.append(float(omega_b))
            omega_delta.append(float(omega_b - omega_c))
            omega_bragg_root_available.append(bool(omega_b_available))
            sg_window.append(float(sg_window_i))
            sg_domega.append(float(sg_width_meta["sg_dSg_domega_inv_A_per_deg"]))
            sg_dframe.append(float(sg_width_meta["sg_dSg_dframe_inv_A"]))
            sg_window_raw.append(float(sg_width_meta["sg_window_raw_inv_A"]))
            sg_window_half_frames.append(float(sg_width_meta["sg_window_half_frames"]))
            sg_window_clipped.append(bool(sg_width_meta["sg_window_clipped"]))
            sg_jitter_sigma.append(float(jitter_meta["sg_jitter_sigma_inv_A"]))
            sg_jitter_d_inv.append(float(jitter_meta["sg_jitter_d_inv_A"]))
            sg_jitter_frame_fraction.append(float(jitter_meta["sg_jitter_frame_fraction"]))
            sg_jitter_kernel_points.append(int(jitter_meta["sg_jitter_kernel_points"]))
            sg_profile_points_total.append(int(jitter_meta["sg_profile_points_total"]))
            sg_profile_weight_sum.append(float(jitter_meta["sg_profile_weight_sum"]))
            sg_profile_same_branch_continuation_used.append(
                bool(jitter_meta["sg_profile_same_branch_continuation_used"])
            )
            sg_profile_same_branch_continuation_segments.append(
                int(jitter_meta["sg_profile_same_branch_continuation_segments"])
            )
            sg_residual_max.append(
                float(np.max(np.abs(sg_resid_values)))
                if len(sg_resid_values)
                else 0.0
            )
            dose_delta_b.append(float(self.row_delta_b(row_idx)))
            if (pos + 1) % 500 == 0 or pos + 1 == len(indices):
                elapsed = time.perf_counter() - t0
                rate = (pos + 1) / max(elapsed, 1e-12)
                eta = (len(indices) - (pos + 1)) / max(rate, 1e-12)
                print(f"built row plans {pos + 1}/{len(indices)} | rate={rate:.2f}/s | ETA={eta/60:.1f}m", flush=True)

        unique_diffs_arr = np.asarray(unique_diffs, dtype=np.int32).reshape(-1, 3)
        unique_inv_d2 = self.inv_d2_for_hkls(unique_diffs_arr) if len(unique_diffs_arr) else np.zeros(0, dtype=float)
        return {
            "meta": meta,
            "indices": indices.astype(np.int64),
            "basis_offsets": np.asarray(basis_offsets, dtype=np.int64),
            "basis_flat": np.asarray(basis_flat, dtype=np.int32).reshape(-1, 3),
            "profile_offsets": np.asarray(profile_offsets, dtype=np.int64),
            "profile_weights_flat": np.asarray(profile_weights_flat, dtype=float),
            "profile_omega_flat": np.asarray(profile_omega_flat, dtype=float),
            "profile_tilts_flat": np.asarray(profile_tilts_flat, dtype=float).reshape(
                -1, 2
            ),
            "sg_offsets": np.asarray(sg_offsets, dtype=np.int64),
            "sg_basis_flat": np.asarray(sg_basis_flat, dtype=float),
            "diff_offsets": np.asarray(diff_offsets, dtype=np.int64),
            "diff_idx_flat": np.asarray(diff_idx_flat, dtype=np.int64),
            "unique_diffs": unique_diffs_arr,
            "unique_inv_d2": np.asarray(unique_inv_d2, dtype=float),
            "center_sg": np.asarray(center_sg, dtype=float),
            "omega_center": np.asarray(omega_center, dtype=float),
            "omega_bragg": np.asarray(omega_bragg, dtype=float),
            "omega_delta": np.asarray(omega_delta, dtype=float),
            "omega_bragg_root_available": np.asarray(
                omega_bragg_root_available, dtype=bool
            ),
            "sg_window": np.asarray(sg_window, dtype=float),
            "sg_domega": np.asarray(sg_domega, dtype=float),
            "sg_dframe": np.asarray(sg_dframe, dtype=float),
            "sg_window_raw": np.asarray(sg_window_raw, dtype=float),
            "sg_window_half_frames": np.asarray(sg_window_half_frames, dtype=float),
            "sg_window_clipped": np.asarray(sg_window_clipped, dtype=bool),
            "sg_jitter_sigma": np.asarray(sg_jitter_sigma, dtype=float),
            "sg_jitter_d_inv": np.asarray(sg_jitter_d_inv, dtype=float),
            "sg_jitter_frame_fraction": np.asarray(sg_jitter_frame_fraction, dtype=float),
            "sg_jitter_kernel_points": np.asarray(sg_jitter_kernel_points, dtype=np.int32),
            "sg_profile_points_total": np.asarray(sg_profile_points_total, dtype=np.int32),
            "sg_profile_weight_sum": np.asarray(sg_profile_weight_sum, dtype=float),
            "sg_profile_same_branch_continuation_used": np.asarray(
                sg_profile_same_branch_continuation_used,
                dtype=bool,
            ),
            "sg_profile_same_branch_continuation_segments": np.asarray(
                sg_profile_same_branch_continuation_segments,
                dtype=np.int32,
            ),
            "sg_residual_max": np.asarray(sg_residual_max, dtype=float),
            "dose_delta_b": np.asarray(dose_delta_b, dtype=float),
        }

    def save_row_plan_cache(self, path: Path, cache: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        save = {k: v for k, v in cache.items() if k != "meta"}
        np.savez_compressed(tmp, **save, meta_json=json.dumps(cache.get("meta", {}), sort_keys=True, allow_nan=True))
        os.replace(tmp, path)

    def load_row_plan_cache_file(self, path: Path) -> dict:
        with np.load(path, allow_pickle=False) as loaded:
            cache = {k: loaded[k] for k in loaded.files if k != "meta_json"}
            cache["meta"] = json.loads(str(loaded["meta_json"]))
        return cache

    def load_or_build_row_plan_cache(self, indices):
        mode = self.row_plan_cache_mode()
        if mode == "off":
            return None
        key, meta = self.row_plan_cache_key(indices)
        path = self.row_plan_cache_path(key)
        if mode in {"exact", "readwrite"} and path.exists():
            try:
                cache = self.load_row_plan_cache_file(path)
                errors = row_plan_cache_validation_errors(cache, indices, key, meta)
                if not errors:
                    cache["status"] = {
                        "status": "hit",
                        "mode": mode,
                        "path": str(path),
                        "key": key,
                    }
                    return cache
                print(
                    "Ignoring stale or incomplete row-plan cache "
                    f"{path}: {'; '.join(errors[:4])}{'; ...' if len(errors) > 4 else ''}",
                    flush=True,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(
                    f"Ignoring unreadable row-plan cache {path}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        cache = self.build_row_plan_cache(indices)
        self.save_row_plan_cache(path, cache)
        cache["status"] = {
            "status": "built" if mode in {"exact", "readwrite"} else "rebuilt",
            "mode": mode,
            "path": str(path),
            "key": key,
        }
        return cache

    def current_unique_ug(self, unique_diffs: np.ndarray) -> np.ndarray:
        """Compute current-model U_g only for the cache's unique differences."""

        unique_diffs = np.asarray(unique_diffs, dtype=int).reshape(-1, 3)
        out = np.zeros(len(unique_diffs), dtype=np.complex128)
        if len(unique_diffs) == 0:
            return out
        h = unique_diffs[:, 0]
        k = unique_diffs[:, 1]
        l = unique_diffs[:, 2]
        ok = (
            (h >= -self.h_lim)
            & (h <= self.h_lim)
            & (k >= -self.k_lim)
            & (k <= self.k_lim)
            & (l >= -self.l_lim)
            & (l <= self.l_lim)
        )
        if np.any(ok):
            factor = float(self.gamma) / (math.pi * self.data.cell.volume)
            out[ok] = factor * fcalc_res_grid(self.model, h[ok], k[ok], l[ok])
        return out

    def cached_static_hamiltonian(self, basis, diff_idx, unique_diffs, unique_inv_d2, unique_ug, row_idx, absorb: bool, dose_delta_b=None):
        n = len(basis)
        H = np.zeros((n, n), dtype=np.complex128)
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        if absorb and mode in {"mean", "midgley_like", "midgley"}:
            H[np.diag_indices(n)] += 1j * float(getattr(self.cfg, "absorption_mu", 0.0))
        elif absorb and mode == "beam_dependent":
            H[np.diag_indices(n)] += 1j * self.beam_absorption_mu(basis)
        u_mat = np.asarray(unique_ug[diff_idx], dtype=np.complex128).reshape(n, n).copy()
        delta_b = float(self.row_delta_b(row_idx) if dose_delta_b is None else dose_delta_b)
        if delta_b != 0.0:
            inv_d2 = np.asarray(unique_inv_d2[diff_idx], dtype=float).reshape(n, n)
            u_mat *= np.exp(-0.25 * delta_b * inv_d2)
        if absorb:
            diffs = np.asarray(unique_diffs[diff_idx], dtype=int).reshape(n, n, 3)
            eta = self.offdiag_absorption_eta_array(diffs)
            if np.any(eta):
                u_mat = u_mat + 1j * eta * u_mat
        np.fill_diagonal(u_mat, 0.0 + 0.0j)
        H += u_mat / (2.0 * self.data.k)
        if absorb and self.absorption_enabled():
            validate_passive_absorption(H, mode=mode, row_index=row_idx)
        return H

    def row_prediction_from_cache(self, cache: dict, pos: int, unique_ug: np.ndarray):
        row_idx = int(cache["indices"][pos])
        b0, b1 = int(cache["basis_offsets"][pos]), int(cache["basis_offsets"][pos + 1])
        p0, p1 = int(cache["profile_offsets"][pos]), int(cache["profile_offsets"][pos + 1])
        s0, s1 = int(cache["sg_offsets"][pos]), int(cache["sg_offsets"][pos + 1])
        d0, d1 = int(cache["diff_offsets"][pos]), int(cache["diff_offsets"][pos + 1])
        basis = np.asarray(cache["basis_flat"][b0:b1], dtype=int)
        n = len(basis)
        weights = np.asarray(cache["profile_weights_flat"][p0:p1], dtype=float)
        sg_basis_rows = np.asarray(cache["sg_basis_flat"][s0:s1], dtype=float).reshape(len(weights), n)
        diff_idx = np.asarray(cache["diff_idx_flat"][d0:d1], dtype=np.int64)
        H_dyn_static = self.cached_static_hamiltonian(
            basis,
            diff_idx,
            cache["unique_diffs"],
            cache["unique_inv_d2"],
            unique_ug,
            row_idx,
            absorb=True,
            dose_delta_b=float(cache["dose_delta_b"][pos]) if "dose_delta_b" in cache else None,
        )
        if self.absorption_enabled():
            H_born_static = self.cached_static_hamiltonian(
                basis,
                diff_idx,
                cache["unique_diffs"],
                cache["unique_inv_d2"],
                unique_ug,
                row_idx,
                absorb=False,
                dose_delta_b=float(cache["dose_delta_b"][pos]) if "dose_delta_b" in cache else None,
            )
        else:
            H_born_static = H_dyn_static
        h10_born = H_born_static[1, 0]
        dyn_vals = []
        inverse_dyn_vals = []
        born_vals = []
        for sg_basis in sg_basis_rows:
            H = self.hamiltonian_from_static(H_dyn_static, sg_basis)
            if bool(getattr(self.cfg, "paired_inverse_hand", False)):
                forward, reverse = self.propagate_direct_to_target_thickness_average_pair(H)
                dyn_vals.append(forward)
                inverse_dyn_vals.append(reverse)
            else:
                dyn_vals.append(self.propagate_direct_to_target_thickness_average(H))
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            sinc = np.sinc(delta * self.thick)
            born_t = (2.0 * math.pi) ** 2 * (abs(h10_born) ** 2) * (self.thick**2) * (sinc**2)
            born_vals.append(float(np.sum(self.thick_w * born_t)))
        dyn = float(np.sum(weights * np.asarray(dyn_vals, dtype=float)))
        born = float(np.sum(weights * np.asarray(born_vals, dtype=float)))
        support = [[int(v) for v in row] for row in basis.tolist()]
        result = {
            "dyn": dyn,
            "born": born,
            "basis_size": int(n),
            "support_hkls": support,
            "max_sg_solve_residual": float(cache["sg_residual_max"][pos]) if "sg_residual_max" in cache else 0.0,
            "center_sg_inv_A": float(cache["center_sg"][pos]),
            "omega_center_deg": float(cache["omega_center"][pos]),
            "omega_bragg_deg": float(cache["omega_bragg"][pos]),
            "omega_bragg_minus_center_deg": float(cache["omega_delta"][pos]),
            "omega_bragg_root_available": bool(
                cache.get(
                    "omega_bragg_root_available",
                    np.isfinite(cache["omega_bragg"]),
                )[pos]
            ),
            "sg_window_inv_A": float(cache["sg_window"][pos]),
            "sg_dSg_domega_inv_A_per_deg": float(cache["sg_domega"][pos]),
            "sg_dSg_dframe_inv_A": float(cache["sg_dframe"][pos]),
            "sg_window_raw_inv_A": float(cache["sg_window_raw"][pos]),
            "sg_window_half_frames": float(cache["sg_window_half_frames"][pos]),
            "sg_window_clipped": bool(cache["sg_window_clipped"][pos]),
            "sg_jitter_sigma_inv_A": float(cache["sg_jitter_sigma"][pos]),
            "sg_jitter_d_inv_A": float(cache["sg_jitter_d_inv"][pos]),
            "sg_jitter_frame_fraction": float(cache["sg_jitter_frame_fraction"][pos]),
            "sg_jitter_kernel_points": int(cache["sg_jitter_kernel_points"][pos]),
            "sg_profile_points_total": int(cache["sg_profile_points_total"][pos]),
            "sg_profile_weight_sum": float(cache["sg_profile_weight_sum"][pos]),
            "sg_profile_same_branch_continuation_used": bool(
                cache["sg_profile_same_branch_continuation_used"][pos]
            ),
            "sg_profile_same_branch_continuation_segments": int(
                cache["sg_profile_same_branch_continuation_segments"][pos]
            ),
            "dose_delta_b_A2": float(cache["dose_delta_b"][pos]),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
        }
        if inverse_dyn_vals:
            result["dyn_inverse"] = float(
                np.sum(weights * np.asarray(inverse_dyn_vals, dtype=float))
            )
        return row_idx, result

    def row_prediction_from_active_cache(self, pos: int):
        return self.row_prediction_from_cache(self._active_row_plan_cache, int(pos), self._active_unique_ug)

    def predict_rows_jax_batched(self, indices, chunksize=8):
        """Predict rows by batching Bloch eigensolvers through JAX."""

        t0_all = time.perf_counter()
        indices = np.asarray(indices, dtype=int)
        n_basis = int(getattr(self.cfg, "n_basis", 20))
        row_block_size = 256 if n_basis >= 128 else (384 if n_basis >= 80 else 512)
        if len(indices) > row_block_size:
            chunks = []
            for start in range(0, len(indices), row_block_size):
                block = indices[start : start + row_block_size]
                print(
                    f"JAX row block {start + 1}-{start + len(block)} of {len(indices)}",
                    flush=True,
                )
                chunks.append(self.predict_rows_jax_batched(block, chunksize=chunksize))
            out = self.merge_prediction_chunks(chunks)
            out["seconds"] = time.perf_counter() - t0_all
            return out
        plans = []
        jobs_by_size = {}
        backend = str(getattr(self.cfg, "compute_backend", "jax") or "jax")
        matrix_bucket = _jax_matrix_bucket_size(self.cfg, backend)
        t0_plan = time.perf_counter()
        for pos, idx in enumerate(indices):
            plan = self.row_prediction_plan(int(idx))
            plans.append(plan)
            for profile_pos, H in enumerate(plan["_dyn_matrices"]):
                n = int(H.shape[0])
                if matrix_bucket > 0:
                    n = int(math.ceil(n / matrix_bucket) * matrix_bucket)
                jobs_by_size.setdefault(n, []).append((pos, profile_pos, H))
            if (pos + 1) % 500 == 0 or pos + 1 == len(indices):
                elapsed = time.perf_counter() - t0_plan
                rate = (pos + 1) / max(elapsed, 1e-12)
                eta = (len(indices) - (pos + 1)) / max(rate, 1e-12)
                print(f"prepared {pos + 1}/{len(indices)} rows for JAX | rate={rate:.2f}/s | ETA={eta/60:.1f}m", flush=True)
        dyn_profile_vals = [np.zeros(len(p["_dyn_matrices"]), dtype=float) for p in plans]
        batch_size = int(getattr(self.cfg, "jax_batch_size", 0))
        absorption = self.absorption_enabled()
        total_matrices = 0
        total_eig_seconds = 0.0
        verbose_jax = str(os.environ.get("GROND_JAX_VERBOSE", "")).strip().lower() in {"1", "true", "yes", "on"}
        for n in sorted(jobs_by_size):
            jobs = jobs_by_size[n]
            if matrix_bucket > 0:
                H_batch = _stack_square_matrices((j[2] for j in jobs), pad_to=n)
            else:
                H_batch = np.stack([j[2] for j in jobs], axis=0)
            t0 = time.perf_counter()
            vals = _jax_propagate_batch(H_batch, self.thick, self.thick_w, backend, absorption, batch_size)
            elapsed = time.perf_counter() - t0
            total_matrices += len(jobs)
            total_eig_seconds += elapsed
            rate = len(jobs) / max(elapsed, 1e-12)
            if verbose_jax:
                print(
                    f"JAX Bloch size={n} matrices={len(jobs)} devices={len(_jax_modules(_jax_module_key(backend))[0].devices())} "
                    f"matrix_bucket={matrix_bucket} rate={rate:.2f}/s wall={elapsed:.2f}s",
                    flush=True,
                )
            for val, (row_pos, profile_pos, _H) in zip(vals, jobs, strict=True):
                dyn_profile_vals[row_pos][profile_pos] = float(val)
        if total_matrices:
            print(
                f"JAX Bloch matrices={total_matrices} groups={len(jobs_by_size)} "
                f"matrix_bucket={matrix_bucket} eig_wall={total_eig_seconds:.2f}s "
                f"matrix_rate={total_matrices / max(total_eig_seconds, 1e-12):.2f}/s",
                flush=True,
            )
        results = [self.finish_row_prediction_plan(plan, dyn_profile_vals[pos]) for pos, plan in enumerate(plans)]
        out = self.pack_prediction_results(indices, results)
        elapsed = time.perf_counter() - t0_all
        out["seconds"] = elapsed
        print(
            f"predicted {len(indices)}/{len(indices)} rows | "
            f"wall={elapsed:.2f}s | rate={len(indices) / max(elapsed, 1e-12):.2f}/s",
            flush=True,
        )
        return out

    def predict_rows_with_row_cache(self, indices, workers=1, chunksize=8):
        """Predict rows from an immutable row-geometry cache.

        This freezes row identity, support basis, Sg profiles, quadrature
        weights, and h_j-h_i lookup indices.  Current-model U_g values are still
        recomputed before prediction, so this cache is meant for LS1-style
        macrocycles where the geometry/support contract is held fixed while the
        structure changes.
        """

        indices = np.asarray(indices, dtype=int)
        t0 = time.perf_counter()
        cache = self.load_or_build_row_plan_cache(indices)
        if cache is None:
            return self.predict_rows(indices, workers=workers, chunksize=chunksize)
        if not np.array_equal(np.asarray(cache["indices"], dtype=int), indices):
            raise RuntimeError("row-plan cache index order does not match requested row order")
        unique_ug = self.current_unique_ug(cache["unique_diffs"])
        self._active_row_plan_cache = cache
        self._active_unique_ug = unique_ug

        def pool_context_names():
            import platform

            if platform.system() == "Darwin":
                return ["forkserver", "spawn"]
            # Linux fork shares the initialized port/cache state copy-on-write;
            # forkserver is a safe fallback but is much slower for high-worker
            # repeated Bloch predictions on large servers.
            return ["fork", "forkserver", "spawn"]

        results = [None] * len(indices)
        parallel_errors: list[str] = []

        def commit_attempt(attempt_results):
            ordered = _ordered_prediction_attempt(indices, attempt_results)
            for pos, r in enumerate(ordered):
                results[pos] = r

        try:
            if int(workers) > 1 and len(indices) > 1:
                ran_parallel = False
                for context_name in pool_context_names():
                    try:
                        ctx = multiprocessing.get_context(context_name)
                    except Exception as exc:
                        parallel_errors.append(f"{context_name}: {type(exc).__name__}: {str(exc)[:160]}")
                        continue
                    try:
                        with ctx.Pool(
                            processes=int(workers),
                            initializer=_init_prediction_worker,
                            initargs=(self,),
                        ) as pool:
                            attempt_results = list(
                                pool.imap_unordered(
                                    _predict_row_cache_worker,
                                    list(range(len(indices))),
                                    chunksize=max(1, int(chunksize)),
                                )
                            )
                        commit_attempt(attempt_results)
                        ran_parallel = True
                        break
                    except (LocalSgCrossingError, InfeasibleSgJitterProfileError):
                        raise
                    except Exception as exc:
                        parallel_errors.append(f"{context_name}: {type(exc).__name__}: {str(exc)[:160]}")
                        ran_parallel = False
                        continue
                if not ran_parallel:
                    if parallel_errors:
                        print(
                            "Cached row prediction fell back to serial after multiprocessing failures: "
                            + " | ".join(parallel_errors[:3]),
                            flush=True,
                        )
                    attempt_results = [
                        self.row_prediction_from_cache(cache, pos, unique_ug)
                        for pos in range(len(indices))
                    ]
                    commit_attempt(attempt_results)
            else:
                attempt_results = [
                    self.row_prediction_from_cache(cache, pos, unique_ug)
                    for pos in range(len(indices))
                ]
                commit_attempt(attempt_results)
        finally:
            self._active_row_plan_cache = None
            self._active_unique_ug = None

        out = self.pack_prediction_results(indices, results)
        elapsed = time.perf_counter() - t0
        out["seconds"] = elapsed
        print(
            f"cached predicted {len(indices)}/{len(indices)} rows | "
            f"wall={elapsed:.2f}s | rate={len(indices) / max(elapsed, 1e-12):.2f}/s",
            flush=True,
        )
        out["row_plan_cache_status"] = cache.get("status", {})
        if parallel_errors:
            out["row_plan_cache_status"] = dict(out["row_plan_cache_status"], parallel_errors=parallel_errors[:3])
        out["row_plan_cache_unique_diffs"] = int(len(cache.get("unique_diffs", [])))
        return out

    def merge_prediction_chunks(self, chunks):
        if not chunks:
            return self.pack_prediction_results(np.zeros(0, dtype=int), [])
        out = {}
        for key in chunks[0]:
            if key == "support_hkls":
                merged = []
                for chunk in chunks:
                    merged.extend(chunk[key])
                out[key] = merged
            elif key == "seconds":
                out[key] = float(sum(float(chunk.get(key, 0.0)) for chunk in chunks))
            else:
                vals = [chunk[key] for chunk in chunks]
                if isinstance(vals[0], np.ndarray):
                    out[key] = np.concatenate(vals)
                else:
                    out[key] = vals
        return out

    def pack_prediction_results(self, indices, results):
        indices = np.asarray(indices, dtype=int)
        dyn = np.zeros(len(indices), dtype=float)
        include_inverse = bool(results) and all(
            isinstance(result, dict) and "dyn_inverse" in result for result in results
        )
        dyn_inverse = np.zeros(len(indices), dtype=float) if include_inverse else None
        born = np.zeros(len(indices), dtype=float)
        basis_size = np.zeros(len(indices), dtype=int)
        basis_support_hash64 = np.zeros(len(indices), dtype=np.uint64)
        sg_resid = np.zeros(len(indices), dtype=float)
        center_sg = np.zeros(len(indices), dtype=float)
        omega_bragg = np.zeros(len(indices), dtype=float)
        omega_delta = np.zeros(len(indices), dtype=float)
        omega_bragg_root_available = np.ones(len(indices), dtype=bool)
        sg_window = np.zeros(len(indices), dtype=float)
        sg_domega = np.zeros(len(indices), dtype=float)
        sg_dframe = np.zeros(len(indices), dtype=float)
        sg_window_raw = np.zeros(len(indices), dtype=float)
        sg_window_half_frames = np.full(len(indices), np.nan, dtype=float)
        sg_window_clipped = np.zeros(len(indices), dtype=bool)
        sg_jitter_sigma = np.zeros(len(indices), dtype=float)
        sg_jitter_d_inv = np.full(len(indices), np.nan, dtype=float)
        sg_jitter_frame_fraction = np.full(len(indices), np.nan, dtype=float)
        sg_jitter_kernel_points = np.ones(len(indices), dtype=int)
        sg_profile_points_total = np.zeros(len(indices), dtype=int)
        sg_profile_weight_sum = np.zeros(len(indices), dtype=float)
        sg_profile_same_branch_continuation_used = np.zeros(len(indices), dtype=bool)
        sg_profile_same_branch_continuation_segments = np.ones(len(indices), dtype=int)
        support_hkls = [None] * len(indices)
        position = {int(idx): int(pos) for pos, idx in enumerate(indices)}

        def accept_result(idx, r):
            n = position[int(idx)]
            dyn[n] = r["dyn"]
            if dyn_inverse is not None:
                dyn_inverse[n] = r["dyn_inverse"]
            born[n] = r["born"]
            basis_size[n] = r["basis_size"]
            support_array = np.asarray(r["support_hkls"], dtype=np.int32)
            basis_support_hash64[n] = int.from_bytes(
                hashlib.sha256(support_array.tobytes(order="C")).digest()[:8],
                byteorder="big",
                signed=False,
            )
            sg_resid[n] = r["max_sg_solve_residual"]
            center_sg[n] = r["center_sg_inv_A"]
            omega_bragg[n] = r["omega_bragg_deg"]
            omega_delta[n] = r["omega_bragg_minus_center_deg"]
            omega_bragg_root_available[n] = bool(
                r.get("omega_bragg_root_available", np.isfinite(r["omega_bragg_deg"]))
            )
            sg_window[n] = r["sg_window_inv_A"]
            sg_domega[n] = r["sg_dSg_domega_inv_A_per_deg"]
            sg_dframe[n] = r["sg_dSg_dframe_inv_A"]
            sg_window_raw[n] = r["sg_window_raw_inv_A"]
            sg_window_half_frames[n] = r["sg_window_half_frames"]
            sg_window_clipped[n] = bool(r["sg_window_clipped"])
            sg_jitter_sigma[n] = r["sg_jitter_sigma_inv_A"]
            sg_jitter_d_inv[n] = r["sg_jitter_d_inv_A"]
            sg_jitter_frame_fraction[n] = r["sg_jitter_frame_fraction"]
            sg_jitter_kernel_points[n] = int(r["sg_jitter_kernel_points"])
            sg_profile_points_total[n] = int(r["sg_profile_points_total"])
            sg_profile_weight_sum[n] = float(r["sg_profile_weight_sum"])
            sg_profile_same_branch_continuation_used[n] = bool(
                r["sg_profile_same_branch_continuation_used"]
            )
            sg_profile_same_branch_continuation_segments[n] = int(
                r["sg_profile_same_branch_continuation_segments"]
            )
            support_hkls[n] = r["support_hkls"]

        for idx, r in zip(indices, results, strict=True):
            accept_result(int(idx), r)

        packed = {
            "indices": indices,
            "dyn": dyn,
            "born": born,
            "basis_size": basis_size,
            "basis_support_hash64": basis_support_hash64,
            "support_hkls": support_hkls,
            "sg_residual": sg_resid,
            "center_sg": center_sg,
            "omega_bragg": omega_bragg,
            "omega_delta": omega_delta,
            "omega_bragg_root_available": omega_bragg_root_available,
            "sg_window": sg_window,
            "sg_dSg_domega": sg_domega,
            "sg_dSg_dframe": sg_dframe,
            "sg_window_raw": sg_window_raw,
            "sg_window_half_frames": sg_window_half_frames,
            "sg_window_clipped": sg_window_clipped,
            "sg_jitter_sigma": sg_jitter_sigma,
            "sg_jitter_d_inv": sg_jitter_d_inv,
            "sg_jitter_frame_fraction": sg_jitter_frame_fraction,
            "sg_jitter_kernel_points": sg_jitter_kernel_points,
            "sg_profile_points_total": sg_profile_points_total,
            "sg_profile_weight_sum": sg_profile_weight_sum,
            "sg_profile_same_branch_continuation_used": sg_profile_same_branch_continuation_used,
            "sg_profile_same_branch_continuation_segments": sg_profile_same_branch_continuation_segments,
            "seconds": 0.0,
        }
        if dyn_inverse is not None:
            packed["dyn_inverse"] = dyn_inverse
        return packed

    def predict_rows(self, indices=None, workers=1, chunksize=8):
        if indices is None:
            indices = np.arange(len(self.data.iobs))
        indices = np.asarray(indices, dtype=int)
        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if self.row_plan_cache_mode() != "off" and backend not in ALL_JAX_BACKENDS:
            return self.predict_rows_with_row_cache(indices, workers=workers, chunksize=chunksize)
        if backend in ALL_JAX_BACKENDS and len(indices) > 1:
            if int(workers) > 1:
                print("JAX backend requested; ignoring multiprocessing workers and batching in one process.", flush=True)
            return self.predict_rows_jax_batched(indices, chunksize=chunksize)
        t0 = time.perf_counter()

        def pool_context_names():
            import platform

            if platform.system() == "Darwin":
                return ["forkserver", "spawn"]
            # Linux fork shares the initialized port/cache state copy-on-write;
            # forkserver is a safe fallback but is much slower for high-worker
            # repeated Bloch predictions on large servers.
            return ["fork", "forkserver", "spawn"]

        results = None
        if int(workers) > 1 and len(indices) > 1:
            for context_name in pool_context_names():
                try:
                    ctx = multiprocessing.get_context(context_name)
                except Exception:
                    continue
                try:
                    with ctx.Pool(
                        processes=int(workers),
                        initializer=_init_prediction_worker,
                        initargs=(self,),
                    ) as pool:
                        attempt_results = list(
                            pool.imap_unordered(
                                _predict_row_worker,
                                [int(v) for v in indices],
                                chunksize=max(1, int(chunksize)),
                            )
                        )
                    results = _ordered_prediction_attempt(indices, attempt_results)
                    break
                except (LocalSgCrossingError, InfeasibleSgJitterProfileError):
                    raise
                except Exception:
                    continue
        if results is None:
            serial_attempt = [
                (int(idx), self.row_prediction(int(idx)))
                for idx in indices
            ]
            results = _ordered_prediction_attempt(indices, serial_attempt)
        out = self.pack_prediction_results(indices, results)
        elapsed = time.perf_counter() - t0
        out["seconds"] = elapsed
        print(
            f"predicted {len(indices)}/{len(indices)} rows | "
            f"wall={elapsed:.2f}s | rate={len(indices) / max(elapsed, 1e-12):.2f}/s",
            flush=True,
        )
        return out


def effective_sigma(data: XDSData, cfg: Config):
    corr_term = np.maximum(0.0, 1.0 - data.corr / 100.0)
    peak_term = np.maximum(0.0, 1.0 - data.peak / 100.0)
    out = np.full(len(data.sigma), np.nan, dtype=float)
    valid = np.isfinite(data.sigma) & (np.asarray(data.sigma, dtype=float) > 0.0)
    out[valid] = np.sqrt(
        data.sigma[valid] ** 2
        + (cfg.variance_a * data.iobs[valid]) ** 2
        + (cfg.variance_b_corr * data.iobs[valid] * corr_term[valid]) ** 2
        + (cfg.variance_c_peak * data.iobs[valid] * peak_term[valid]) ** 2
    )
    return out


def reflection_orbit_key(hkl, symmetry_ops=None, include_friedel: bool = True):
    """Return a canonical reciprocal-space orbit key for held-out splitting."""

    h = np.asarray(hkl, dtype=float)
    rotations = [np.eye(3)]
    for operation in symmetry_ops or []:
        rotation = operation.get("rot") if isinstance(operation, dict) else operation[0]
        rotations.append(np.asarray(rotation, dtype=float))
    equivalents = []
    for rotation in rotations:
        transformed = np.linalg.inv(rotation).T @ h
        rounded = np.rint(transformed).astype(int)
        if not np.allclose(transformed, rounded, rtol=0.0, atol=1e-7):
            raise ValueError("symmetry operation did not map a Miller index to integer values")
        key = tuple(int(value) for value in rounded)
        equivalents.append(key)
        if include_friedel:
            equivalents.append(tuple(-value for value in key))
    return min(equivalents)


def split_masks(hkls, free_fraction=0.10, seed=42, symmetry_ops=None, include_friedel: bool = True):
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("free_fraction must be between 0 and 1")
    orbit_keys = [reflection_orbit_key(h, symmetry_ops, include_friedel) for h in hkls]
    unique = sorted(set(orbit_keys))
    if len(unique) < 2:
        raise ValueError("held-out splitting requires at least two unique HKLs")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    n_free = min(len(unique) - 1, max(1, int(round(len(unique) * float(free_fraction)))))
    free = set(unique[:n_free])
    masks = {"all": np.ones(len(hkls), dtype=bool), "work": np.zeros(len(hkls), dtype=bool), "free": np.zeros(len(hkls), dtype=bool)}
    for i, key in enumerate(orbit_keys):
        masks["free"][i] = key in free
    masks["work"] = ~masks["free"]
    return masks


def _canonical_json_sha256(value) -> str:
    """Hash one JSON-compatible value with a stable, whitespace-free encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symmetry_rotation_contract(symmetry_ops) -> list[list[list[float]]]:
    """Return the reciprocal-orbit-relevant part of a model symmetry contract."""

    rotations = []
    for operation in symmetry_ops or []:
        rotation = operation.get("rot") if isinstance(operation, dict) else operation[0]
        array = np.asarray(rotation, dtype=float)
        if array.shape != (3, 3) or not np.all(np.isfinite(array)):
            raise ValueError("model symmetry rotation must be one finite 3x3 matrix")
        rotations.append(
            [[round(float(value), 12) for value in row] for row in array]
        )
    return sorted(rotations, key=lambda value: json.dumps(value, separators=(",", ":")))


def _reciprocal_metric_from_cell(cell) -> np.ndarray:
    """Return G* for a six-parameter direct cell."""

    unit_cell = gemmi.UnitCell(*[float(value) for value in cell])
    basis = np.eye(3, dtype=int)
    metric = np.zeros((3, 3), dtype=float)
    for i in range(3):
        metric[i, i] = float(unit_cell.calculate_1_d2(basis[i]))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair = basis[i] + basis[j]
        cross = 0.5 * (
            float(unit_cell.calculate_1_d2(pair))
            - metric[i, i]
            - metric[j, j]
        )
        metric[i, j] = metric[j, i] = cross
    return metric


def metric_filter_signed_axis_transforms(
    hkl_path: Path,
    model_path: Path,
    transforms,
    *,
    relative_tolerance: float,
    dataset_label: str = "",
) -> tuple[list[str], dict]:
    """Retain signed-axis transforms compatible with source/model cell metrics.

    This prefilter uses only the source and model unit-cell metrics.  It never
    reads intensities, coordinate hand, atom types, expected chemistry, or a
    known configuration.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("transform metric relative tolerance must be finite and > 0")
    hkl_path = Path(hkl_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    data = XDSData(hkl_path, hkl_transform="identity")
    model = parse_model(model_path)
    source_cell = (
        data.raw_cell.a,
        data.raw_cell.b,
        data.raw_cell.c,
        data.raw_cell.alpha,
        data.raw_cell.beta,
        data.raw_cell.gamma,
    )
    model_cell = tuple(float(value) for value in model["cell"])
    source_metric = _reciprocal_metric_from_cell(source_cell)
    model_metric = _reciprocal_metric_from_cell(model_cell)
    denominator = max(float(np.linalg.norm(model_metric, ord="fro")), 1.0e-30)
    # XDS normally writes INTEGRATE.HKL in the primitive integration basis,
    # even when CORRECT subsequently constrains and reindexes the data in the
    # solved-model space group.  The raw P1 angles and lengths can therefore be
    # several percent away from the symmetry-constrained model cell.  Keep the
    # ordinary full-metric gate strict, but provide one bounded, geometry-only
    # fallback for that explicit XDS P1 -> higher-symmetry situation.  The
    # fallback still operates only on cells and signed-axis permutations; it
    # never uses intensities, atoms, chemistry, or a known hand.
    p1_fallback_full_metric_tolerance = 0.05
    p1_fallback_max_axis_length_fraction = 0.03
    source_space_group_text = str(data.header.get("SPACE_GROUP_NUMBER", "")).split()
    try:
        source_space_group_number = (
            int(float(source_space_group_text[0])) if source_space_group_text else None
        )
    except ValueError:
        source_space_group_number = None
    model_symmetry_order = len(model.get("symmetry_ops") or [])
    source_kind = str(data.header.get("GROND_SOURCE", "XDS") or "XDS").strip().upper()
    p1_fallback_applicable = bool(
        source_kind == "XDS"
        and source_space_group_number == 1
        and model_symmetry_order > 1
    )

    records = []
    retained = []
    seen = set()
    for transform in transforms:
        info = parse_hkl_transform_info(transform)
        if info.canonical in seen:
            continue
        seen.add(info.canonical)
        matrix = np.asarray(info.matrix, dtype=float)
        inverse = np.linalg.inv(matrix)
        candidate_metric = inverse.T @ source_metric @ inverse
        difference = candidate_metric - model_metric
        relative_frobenius_mismatch = float(
            np.linalg.norm(difference, ord="fro") / denominator
        )
        max_absolute_mismatch = float(np.max(np.abs(difference)))
        candidate_direct_metric = np.linalg.inv(candidate_metric)
        candidate_axis_lengths = np.sqrt(
            np.maximum(np.diag(candidate_direct_metric), 0.0)
        )
        model_axis_lengths = np.asarray(model_cell[:3], dtype=float)
        axis_length_fractional_mismatch = np.abs(
            candidate_axis_lengths - model_axis_lengths
        ) / model_axis_lengths
        max_axis_length_fractional_mismatch = float(
            np.max(axis_length_fractional_mismatch)
        )
        keep = bool(relative_frobenius_mismatch <= tolerance)
        if keep:
            retained.append(info.canonical)
        records.append(
            {
                "transform": info.canonical,
                "determinant": info.determinant,
                "status": "retained" if keep else "rejected",
                "reason": (
                    "source/model reciprocal-cell metrics agree within tolerance"
                    if keep
                    else "source/model reciprocal-cell metric mismatch exceeds tolerance"
                ),
                "relative_frobenius_mismatch": relative_frobenius_mismatch,
                "max_absolute_metric_mismatch_A^-2": max_absolute_mismatch,
                "candidate_axis_lengths_A": candidate_axis_lengths.tolist(),
                "axis_length_fractional_mismatch": (
                    axis_length_fractional_mismatch.tolist()
                ),
                "max_axis_length_fractional_mismatch": (
                    max_axis_length_fractional_mismatch
                ),
                "retention_mode": "strict_full_metric" if keep else "rejected",
            }
        )
    fallback_applied = False
    if not retained:
        fallback_records = [
            record
            for record in records
            if p1_fallback_applicable
            and float(record["relative_frobenius_mismatch"])
            <= p1_fallback_full_metric_tolerance
            and float(record["max_axis_length_fractional_mismatch"])
            <= p1_fallback_max_axis_length_fraction
        ]
        if fallback_records:
            fallback_applied = True
            fallback_transforms = {
                str(record["transform"]) for record in fallback_records
            }
            retained = [
                str(record["transform"])
                for record in records
                if str(record["transform"]) in fallback_transforms
            ]
            for record in records:
                if str(record["transform"]) in fallback_transforms:
                    record.update(
                        {
                            "status": "retained",
                            "reason": (
                                "bounded XDS P1 integration-cell distortion fallback: "
                                "source/model axis lengths and full reciprocal metrics "
                                "agree within fixed fallback limits"
                            ),
                            "retention_mode": "xds_p1_distortion_fallback",
                        }
                    )
        else:
            best = min(
                records,
                key=lambda record: float(record["relative_frobenius_mismatch"]),
            )
            dataset_context = f"dataset {dataset_label!r}: " if dataset_label else ""
            if source_kind == "DIALS":
                raise ValueError(
                    f"{dataset_context}unit-cell metric compatibility filter rejected every "
                    "transform candidate for a DIALS bridge; "
                    f"best={best['transform']} full-metric mismatch="
                    f"{float(best['relative_frobenius_mismatch']):.4f}, max axis-length "
                    f"mismatch={float(best['max_axis_length_fractional_mismatch']):.4f}; "
                    f"source cell={tuple(round(float(v), 6) for v in source_cell)}, "
                    f"model cell={tuple(round(float(v), 6) for v in model_cell)}. "
                    "dials.cosym can make experiment indices mutually consistent, but it "
                    "does not prove that an external RES/HKL pair is in that same basis. "
                    "If the model and baseline HKL were exported from this exact post-cosym "
                    "scaled.expt/scaled.refl basis with no later reindexing or inversion, use "
                    "--hkl-transform identity --trust-hkl-transform. Otherwise inspect this "
                    "named crystal as a possible cell outlier or supply the verified transform."
                )
            raise ValueError(
                f"{dataset_context}unit-cell metric compatibility filter rejected every "
                "transform candidate; "
                f"best={best['transform']} full-metric mismatch="
                f"{float(best['relative_frobenius_mismatch']):.4f}, max axis-length "
                f"mismatch={float(best['max_axis_length_fractional_mismatch']):.4f}; "
                f"source cell={tuple(round(float(v), 6) for v in source_cell)}, "
                f"model cell={tuple(round(float(v), 6) for v in model_cell)}. "
                "The automatic XDS P1 fallback requires an explicit source "
                "SPACE_GROUP_NUMBER=1, a higher-symmetry model, full-metric mismatch "
                f"<={p1_fallback_full_metric_tolerance:.3f}, and maximum axis-length "
                f"mismatch<={p1_fallback_max_axis_length_fraction:.3f}."
            )
    contract = {
        "schema_version": 1,
        "policy": "source_model_reciprocal_cell_metric_only_v1",
        "supported_transform_family": "signed_axis_permutations_only",
        "relative_frobenius_tolerance": tolerance,
        "filter_mode": (
            "xds_p1_distortion_fallback" if fallback_applied else "strict_full_metric"
        ),
        "source_space_group_number": source_space_group_number,
        "model_symmetry_order": model_symmetry_order,
        "xds_p1_distortion_fallback": {
            "applicable": p1_fallback_applicable,
            "applied": fallback_applied,
            "full_metric_relative_tolerance": p1_fallback_full_metric_tolerance,
            "max_axis_length_fraction": p1_fallback_max_axis_length_fraction,
            "reason": (
                "INTEGRATE.HKL can retain a distorted P1 integration cell while "
                "CORRECT and the solved model use a symmetry-constrained setting"
            ),
        },
        "uses_intensities": False,
        "uses_atomic_coordinates_or_hand": False,
        "uses_atom_types_or_known_chemistry": False,
        "uses_known_configuration": False,
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": sha256(hkl_path),
        "source_model": str(model_path),
        "source_model_sha256": sha256(model_path),
        "source_cell": [float(value) for value in source_cell],
        "source_cell_basis": "raw_UNIT_CELL_CONSTANTS",
        "axis_derived_source_cell": [
            data.axis_derived_source_cell.a,
            data.axis_derived_source_cell.b,
            data.axis_derived_source_cell.c,
            data.axis_derived_source_cell.alpha,
            data.axis_derived_source_cell.beta,
            data.axis_derived_source_cell.gamma,
        ],
        "axis_derived_source_cell_basis": data.axis_derived_source_cell_basis,
        "model_cell": [float(value) for value in model_cell],
        "source_reciprocal_metric": source_metric.tolist(),
        "model_reciprocal_metric": model_metric.tolist(),
        "generated_candidate_count": len(records),
        "retained_candidate_count": len(retained),
        "rejected_candidate_count": len(records) - len(retained),
        "retained_transforms": retained,
        "records": records,
    }
    contract["contract_payload_sha256"] = _canonical_json_sha256(contract)
    return retained, contract


def candidate_invariant_split_membership(
    raw_hkls,
    symmetry_ops,
    transforms,
    *,
    free_fraction: float = 0.10,
    seed: int = 42,
    include_friedel: bool = True,
):
    """Build one work/free split whose orbit groups are valid for every candidate.

    Rows are unioned whenever any candidate transform places them in the same
    model-symmetry/Friedel orbit.  Assigning whole union components to work or
    free therefore prevents a row (or one of its symmetry-equivalent
    duplicates) from changing split merely because a different indexing
    candidate is being scored.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    raw = np.asarray(raw_hkls, dtype=int)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw HKLs must be an N by 3 integer array")
    if len(raw) < 2:
        raise ValueError("candidate-invariant held-out splitting requires at least two rows")
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("free_fraction must be between 0 and 1")

    canonical_transforms = []
    matrices = []
    for transform in transforms:
        info = parse_hkl_transform_info(transform)
        if info.canonical in canonical_transforms:
            continue
        canonical_transforms.append(info.canonical)
        matrices.append(np.asarray(info.matrix, dtype=int))
    if not canonical_transforms:
        raise ValueError("candidate-invariant held-out splitting requires transforms")

    parent = np.arange(len(raw), dtype=int)
    rank = np.zeros(len(raw), dtype=np.int8)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    # One orbit registry must span *all* candidates.  A row transformed by
    # candidate A and another row transformed by candidate B represent the
    # same held-out observation whenever those transformed HKLs share a
    # model-symmetry/Friedel orbit.
    first_by_orbit = {}
    for matrix in matrices:
        transformed = (raw @ matrix.T).astype(int)
        for row_index, hkl in enumerate(transformed):
            key = reflection_orbit_key(
                hkl,
                symmetry_ops=symmetry_ops,
                include_friedel=include_friedel,
            )
            first = first_by_orbit.setdefault(key, row_index)
            union(first, row_index)

    components = {}
    for row_index in range(len(raw)):
        components.setdefault(find(row_index), []).append(row_index)
    ordered_components = sorted(components.values(), key=lambda rows: rows[0])
    if len(ordered_components) < 2:
        raise ValueError(
            "candidate-union symmetry/Friedel grouping leaves fewer than two "
            "independent reflection groups"
        )

    shuffled_group_ids = np.arange(len(ordered_components), dtype=int)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled_group_ids)
    n_free_groups = min(
        len(ordered_components) - 1,
        max(1, int(round(len(ordered_components) * float(free_fraction)))),
    )
    free_groups = set(int(value) for value in shuffled_group_ids[:n_free_groups])
    group_by_row = np.zeros(len(raw), dtype=int)
    membership = np.zeros(len(raw), dtype=np.uint8)
    for group_id, row_indices in enumerate(ordered_components):
        group_by_row[row_indices] = group_id
        if group_id in free_groups:
            membership[row_indices] = 1
    if not np.any(membership == 0) or not np.any(membership == 1):
        raise ValueError("candidate-invariant split must contain both work and free rows")
    return {
        "candidate_transforms": canonical_transforms,
        "membership": membership,
        "orbit_group_by_row": group_by_row,
        "group_count": len(ordered_components),
        "work_group_count": len(ordered_components) - len(free_groups),
        "free_group_count": len(free_groups),
    }


def build_transform_split_contract(
    hkl_path: Path,
    model_path: Path,
    transforms,
    *,
    dataset: str,
    free_fraction: float = 0.10,
    seed: int = 42,
    include_friedel: bool = True,
) -> dict:
    """Build the immutable raw-row split shared by transform and hand scouts."""

    hkl_path = Path(hkl_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    data = XDSData(hkl_path, hkl_transform="identity")
    model = parse_model(model_path)
    rotations = _symmetry_rotation_contract(model.get("symmetry_ops"))
    result = candidate_invariant_split_membership(
        data.raw_hkl,
        model.get("symmetry_ops"),
        transforms,
        free_fraction=free_fraction,
        seed=seed,
        include_friedel=include_friedel,
    )
    membership = np.asarray(result["membership"], dtype=np.uint8)
    group_by_row = np.asarray(result["orbit_group_by_row"], dtype=int)
    raw_hkl_rows = [[int(value) for value in row] for row in data.raw_hkl]
    payload = {
        "schema_version": 1,
        "kind": "grond_transform_hand_split_contract",
        "algorithm": "candidate_union_model_symmetry_friedel_orbits_v1",
        "dataset": str(dataset),
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": sha256(hkl_path),
        "source_model": str(model_path),
        "source_model_sha256": sha256(model_path),
        "source_raw_hkl_sha256": _canonical_json_sha256(raw_hkl_rows),
        "model_symmetry_rotations": rotations,
        "model_symmetry_rotations_sha256": _canonical_json_sha256(rotations),
        "candidate_transforms": list(result["candidate_transforms"]),
        "candidate_transforms_sha256": _canonical_json_sha256(
            list(result["candidate_transforms"])
        ),
        "include_friedel": bool(include_friedel),
        "free_fraction": float(free_fraction),
        "split_seed": int(seed),
        "row_count": int(len(membership)),
        "group_count": int(result["group_count"]),
        "work_group_count": int(result["work_group_count"]),
        "free_group_count": int(result["free_group_count"]),
        "work_row_count": int(np.count_nonzero(membership == 0)),
        "free_row_count": int(np.count_nonzero(membership == 1)),
        "membership_encoding": "0=work,1=free; ordered raw numeric reflection rows",
        "membership": [int(value) for value in membership],
        "membership_sha256": hashlib.sha256(membership.tobytes()).hexdigest(),
        "orbit_group_by_row": [int(value) for value in group_by_row],
        "orbit_group_by_row_sha256": _canonical_json_sha256(
            [int(value) for value in group_by_row]
        ),
    }
    payload["contract_payload_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_transform_split_contract(
    payload,
    *,
    source_hkl: Path | None = None,
    current_transform: str = "",
    model=None,
) -> dict:
    """Validate the split payload and optionally bind it to live data/model."""

    from grond.hkl_transform import parse_hkl_transform_info

    if not isinstance(payload, dict):
        raise ValueError("transform/hand split contract must contain one JSON object")
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("transform/hand split contract schema_version must be 1")
    if payload.get("kind") != "grond_transform_hand_split_contract":
        raise ValueError(
            "transform/hand split contract kind must be "
            "'grond_transform_hand_split_contract'"
        )
    if payload.get("algorithm") != "candidate_union_model_symmetry_friedel_orbits_v1":
        raise ValueError("unsupported transform/hand split-contract algorithm")
    recorded_contract_hash = str(payload.get("contract_payload_sha256") or "")
    without_hash = dict(payload)
    without_hash.pop("contract_payload_sha256", None)
    if recorded_contract_hash != _canonical_json_sha256(without_hash):
        raise ValueError("transform/hand split contract payload hash mismatch")

    membership = np.asarray(payload.get("membership"), dtype=np.int64)
    groups = np.asarray(payload.get("orbit_group_by_row"), dtype=np.int64)
    row_count = int(payload.get("row_count") or -1)
    if membership.shape != (row_count,) or groups.shape != (row_count,):
        raise ValueError("transform/hand split membership/group lengths do not match row_count")
    if not np.all(np.isin(membership, (0, 1))):
        raise ValueError("transform/hand split membership must contain only 0=work and 1=free")
    if hashlib.sha256(membership.astype(np.uint8).tobytes()).hexdigest() != str(
        payload.get("membership_sha256") or ""
    ):
        raise ValueError("transform/hand split membership hash mismatch")
    if _canonical_json_sha256([int(value) for value in groups]) != str(
        payload.get("orbit_group_by_row_sha256") or ""
    ):
        raise ValueError("transform/hand split orbit-group hash mismatch")
    for group_id in np.unique(groups):
        local = membership[groups == group_id]
        if len(np.unique(local)) != 1:
            raise ValueError(
                "one symmetry/Friedel orbit group crosses the work/free boundary"
            )
    work_rows = int(np.count_nonzero(membership == 0))
    free_rows = int(np.count_nonzero(membership == 1))
    if work_rows != int(payload.get("work_row_count") or -1) or free_rows != int(
        payload.get("free_row_count") or -1
    ):
        raise ValueError("transform/hand split recorded row counts do not match membership")
    transforms = [
        parse_hkl_transform_info(value).canonical
        for value in payload.get("candidate_transforms") or []
    ]
    if not transforms:
        raise ValueError("transform/hand split contract has no candidate transforms")
    if _canonical_json_sha256(transforms) != str(
        payload.get("candidate_transforms_sha256") or ""
    ):
        raise ValueError("transform/hand split candidate-transform hash mismatch")
    if current_transform:
        current = parse_hkl_transform_info(current_transform).canonical
        if current not in transforms:
            raise ValueError(
                f"active transform {current!r} is absent from the immutable split contract"
            )
    if source_hkl is not None:
        source_hkl = Path(source_hkl).expanduser().resolve()
        if not source_hkl.is_file() or sha256(source_hkl) != str(
            payload.get("source_hkl_sha256") or ""
        ):
            raise ValueError("transform/hand split source-HKL hash mismatch")
        data = XDSData(source_hkl, hkl_transform="identity")
        if len(data.raw_hkl) != row_count:
            raise ValueError("transform/hand split row count does not match live source HKL")
        raw_rows = [[int(value) for value in row] for row in data.raw_hkl]
        if _canonical_json_sha256(raw_rows) != str(
            payload.get("source_raw_hkl_sha256") or ""
        ):
            raise ValueError("transform/hand split ordered raw-HKL hash mismatch")
    if model is not None:
        rotations = _symmetry_rotation_contract(model.get("symmetry_ops"))
        if _canonical_json_sha256(rotations) != str(
            payload.get("model_symmetry_rotations_sha256") or ""
        ):
            raise ValueError("transform/hand split model-symmetry rotation hash mismatch")
    return {
        "status": "validated",
        "schema_version": 1,
        "algorithm": str(payload["algorithm"]),
        "dataset": str(payload.get("dataset") or ""),
        "row_count": row_count,
        "group_count": int(payload.get("group_count") or 0),
        "work_row_count": work_rows,
        "free_row_count": free_rows,
        "free_fraction": float(payload["free_fraction"]),
        "split_seed": int(payload["split_seed"]),
        "candidate_transforms": transforms,
        "candidate_transforms_sha256": str(payload["candidate_transforms_sha256"]),
        "membership_sha256": str(payload["membership_sha256"]),
        "orbit_group_by_row_sha256": str(payload["orbit_group_by_row_sha256"]),
        "source_hkl_sha256": str(payload["source_hkl_sha256"]),
        "source_raw_hkl_sha256": str(payload["source_raw_hkl_sha256"]),
        "model_symmetry_rotations_sha256": str(
            payload["model_symmetry_rotations_sha256"]
        ),
        "contract_payload_sha256": recorded_contract_hash,
        "masks": {
            "all": np.ones(row_count, dtype=bool),
            "work": membership == 0,
            "free": membership == 1,
        },
    }


def load_transform_split_contract_masks(path: Path, data, model) -> tuple[dict, dict]:
    """Load a split contract and return full-length masks plus audit metadata."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    validation = validate_transform_split_contract(
        payload,
        source_hkl=data.path,
        current_transform=data.hkl_transform_spec,
        model=model,
    )
    masks = validation.pop("masks")
    validation.update({"path": str(path), "sha256": sha256(path)})
    return masks, validation


def scale_by_groups(obs, sig, pred, groups, fit_mask):
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups, dtype=int)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(obs) == len(sig) == len(pred) == len(groups) == len(fit_mask)):
        raise ValueError("observation, sigma, prediction, group, and fit-mask lengths differ")
    scaled = np.full_like(pred, np.nan, dtype=float)
    scales = {}
    valid_measurement = np.isfinite(obs) & np.isfinite(sig) & (sig > 0.0)
    weights = np.zeros_like(sig, dtype=float)
    weights[valid_measurement] = 1.0 / (sig[valid_measurement] * sig[valid_measurement])
    global_fit = fit_mask & valid_measurement & np.isfinite(pred) & (pred > 0.0)
    if np.count_nonzero(global_fit) < 2:
        raise ValueError("at least two work rows with positive sigma and prediction are required to fit scales")
    x_global, y_global, w_global = pred[global_fit], obs[global_fit], weights[global_fit]
    global_scale = max(
        float(np.sum(w_global * x_global * y_global) / max(np.sum(w_global * x_global * x_global), 1e-30)),
        1e-30,
    )
    for g in np.unique(groups):
        fit = (groups == g) & fit_mask & valid_measurement & np.isfinite(pred) & (pred > 0)
        apply = groups == g
        if np.count_nonzero(fit) < 2:
            scales[int(g)] = global_scale
            finite_apply = apply & np.isfinite(pred)
            scaled[finite_apply] = pred[finite_apply] * global_scale
            continue
        x, y, w = pred[fit], obs[fit], weights[fit]
        scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
        scale = max(scale, 0.0)
        finite_apply = apply & np.isfinite(pred)
        scaled[finite_apply] = pred[finite_apply] * scale
        scales[int(g)] = scale
    return scaled, scales


def metrics(obs, sig, pred, mask):
    mask = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
        & np.isfinite(pred)
    )
    obs_m, sig_m, pred_m = obs[mask], sig[mask], pred[mask]
    gt = obs_m > 2.0 * sig_m
    fobs = np.sqrt(np.maximum(obs_m, 0.0))
    fpred = np.sqrt(np.maximum(pred_m, 0.0))
    eps = 1e-30
    def r1(local):
        if not np.any(local):
            return float("nan")
        return float(np.sum(np.abs(fobs[local] - fpred[local])) / max(np.sum(fobs[local]), eps))
    p = (np.maximum(obs_m, 0.0) + 2.0 * np.maximum(pred_m, 0.0)) / 3.0
    w = 1.0 / np.maximum(sig_m * sig_m + (0.05 * p) ** 2, eps)
    wr2 = float(np.sqrt(np.sum(w * (obs_m - pred_m) ** 2) / max(np.sum(w * obs_m * obs_m), eps)))
    cc = float(np.corrcoef(obs_m, pred_m)[0, 1]) if len(obs_m) > 2 and np.std(pred_m) > 0 and np.std(obs_m) > 0 else float("nan")
    return {"n": int(np.count_nonzero(mask)), "n_gt": int(np.count_nonzero(gt)), "R1_all": r1(np.ones_like(gt, dtype=bool)), "R1_gt": r1(gt), "wR2_all": wr2, "CC": cc}


def evaluate_predictions(data, cfg, pred, fit_scales_on="work", model=None):
    sig = effective_sigma(data, cfg)
    symmetry_ops = model.get("symmetry_ops") if model else None
    masks = split_masks(data.hkl, cfg.free_fraction, cfg.split_seed, symmetry_ops=symmetry_ops)
    group_indices = pred["groups"]
    fit_mask = masks[fit_scales_on]
    dyn_scaled, dyn_scales = scale_by_groups(data.iobs, sig, pred["dyn"], group_indices, fit_mask)
    born_scaled, born_scales = scale_by_groups(data.iobs, sig, pred["born"], group_indices, fit_mask)
    out = {"fit_scales_on": fit_scales_on, "splits": {}, "group_scale_count": int(len(dyn_scales))}
    for name, mask in masks.items():
        dyn_m = metrics(data.iobs, sig, dyn_scaled, mask)
        born_m = metrics(data.iobs, sig, born_scaled, mask)
        out["splits"][name] = {
            "dyn": dyn_m,
            "born": born_m,
            "delta_R1_all_dyn_minus_born": dyn_m["R1_all"] - born_m["R1_all"],
            "delta_R1_gt_dyn_minus_born": dyn_m["R1_gt"] - born_m["R1_gt"],
            "delta_wR2_all_dyn_minus_born": dyn_m["wR2_all"] - born_m["wR2_all"],
        }
    return out


def read_row_index_file(path: str | Path) -> list[int]:
    """Read zero-based row indices from text or CSV, including one-column CSV."""

    with open(path, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "," in sample:
            reader = csv.DictReader(handle)
            field = (
                "row_index"
                if reader.fieldnames and "row_index" in reader.fieldnames
                else (reader.fieldnames[0] if reader.fieldnames else None)
            )
            if field is None:
                raise ValueError(f"No columns found in {path}")
            values = [int(row[field]) for row in reader]
        else:
            values = []
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.split()[0]
                if token.lower() in {"row_index", "index"}:
                    continue
                try:
                    values.append(int(token))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid row index {token!r} at {path}:{line_number}"
                    ) from exc
    return sorted(set(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--hkl-transform", default="identity", help="Signed axis permutation mapping XDS HKLs into the model basis, e.g. 'l,-k,h'.")
    ap.add_argument("--out", default=str(ROOT / "outputs/engine_summary.json"))
    ap.add_argument("--cache-dir", default=str(runtime_cache_dir("legacy_engine")), help="Local Ug cube cache directory. Use 'none' to disable.")
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--sg-window", type=float, default=0.006)
    ap.add_argument("--sg-window-mode", choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"], default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50, help="Minimum half-width in frames for ZOBS/ZCAL adaptive Sg-window modes.")
    ap.add_argument("--sg-jitter-mode", choices=["none", "gaussian", "student_t"], default="none", help="Optional target-Sg orientation-distribution convolution.")
    ap.add_argument("--sg-jitter-sigma0", type=float, default=0.0, help="Base target-Sg jitter sigma in inverse Angstrom.")
    ap.add_argument("--sg-jitter-sigma-d-inv", type=float, default=0.0, help="Target-Sg jitter sigma slope versus 1/d.")
    ap.add_argument("--sg-jitter-sigma-frame", type=float, default=0.0, help="Target-Sg jitter sigma slope versus normalized frame number.")
    ap.add_argument("--sg-jitter-points", type=int, default=5, help="Kernel quadrature points for Sg-jitter convolution.")
    ap.add_argument("--sg-jitter-student-df", type=float, default=4.0, help="Student-t degrees of freedom for Sg-jitter mode.")
    ap.add_argument("--sg-jitter-width-factor", type=float, default=3.0, help="Truncation half-width in sigma units for Student-t jitter mode.")
    ap.add_argument("--mosaicity-mode", choices=["none", "gaussian_omega", "gaussian_cone"], default="none", help="Optional projected one-axis or incoherent two-axis Gaussian mosaic convolution.")
    ap.add_argument("--mosaicity-sigma-deg", type=float, default=0.0, help="Projected Gaussian mosaicity sigma in degrees; never inferred from the XDS reflecting range automatically.")
    ap.add_argument("--mosaicity-max-sigma-deg", type=float, default=0.5, help="Hard upper bound for the projected mosaicity sigma.")
    ap.add_argument("--mosaicity-points", type=int, default=3, help="Gauss-Hermite points for direct angular mosaicity quadrature.")
    ap.add_argument("--mosaicity-support-sigma-deg", type=float, default=0.0, help="Largest tested two-axis cone sigma. Every candidate uses the beam basis selected over this same envelope.")
    ap.add_argument("--mosaicity-support-points", type=int, default=5, help="Points per scan/cone axis used only to construct the common enclosing beam basis.")
    ap.add_argument("--mosaicity-support-width-sigma", type=float, default=4.0, help="Half-width of the common basis cone in support-sigma units.")
    ap.add_argument("--geometry-path", default="", help="Optional geometry-only Chebyshev slew/two-axis tilt JSON produced independently of intensities.")
    ap.add_argument("--profile-basis-mode", choices=["center", "common_profile"], default="center", help="Use the row-center beam basis or a fixed beam union over the complete rocking-profile support.")
    ap.add_argument("--profile-coordinate", choices=["target_sg", "direct_angle"], default="target_sg", help="Integrate through solved target-Sg points or directly in recorded scan angle.")
    ap.add_argument("--profile-angle-max-half-width-deg", type=float, default=2.0, help="Hard half-width bound for direct-angle integration.")
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument(
        "--max-support-sg",
        type=float,
        default=DEFAULT_MAX_SUPPORT_SG,
        help="Maximum absolute excitation error for beam-basis support (default 0.01 A^-1).",
    )
    ap.add_argument("--thickness", type=float, default=1240.0)
    ap.add_argument("--thickness-model", choices=["single", "uniform_gradient", "gaussian", "gaussian_gradient"], default="uniform_gradient")
    ap.add_argument("--thickness-grad-frac", type=float, default=0.25)
    ap.add_argument("--thickness-sigma-frac", type=float, default=0.0)
    ap.add_argument("--thickness-points", type=int, default=7)
    ap.add_argument("--group-width-deg", type=float, default=3.0)
    ap.add_argument("--variance-a", type=float, default=0.0)
    ap.add_argument("--variance-b-corr", type=float, default=0.0)
    ap.add_argument("--variance-c-peak", type=float, default=0.0)
    ap.add_argument("--frame-offset", type=float, default=0.0, help="Global frame offset added to ZCAL before omega replay.")
    ap.add_argument("--ug-d-min", type=float, default=0.30, help="Resolution limit for model-dependent U_g cache. Keep 0.30 for endpoint/audit runs; larger values are diagnostic speedups.")
    ap.add_argument("--absorption-mode", choices=["none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"], default="none")
    ap.add_argument("--absorption-mu", type=float, default=0.0, help="Uniform imaginary diagonal damping in inverse Angstrom.")
    ap.add_argument("--absorption-frac", type=float, default=0.0, help="Imaginary absorptive off-diagonal fraction relative to U_g.")
    ap.add_argument("--absorption-mu-g2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus |g|^2.")
    ap.add_argument("--absorption-mu-theta2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus Bragg-angle^2.")
    ap.add_argument("--absorption-tds-b", type=float, default=0.0, help="B_TDS in eta*U_g*(1-exp(-B_TDS*|g|^2/4)) for offdiag_tds mode.")
    ap.add_argument("--dose-b-mode", choices=["none", "linear_frame"], default="none")
    ap.add_argument("--dose-b-per-frame", type=float, default=0.0, help="Extra isotropic B added per XDS frame after the reference frame.")
    ap.add_argument("--dose-b-ref-frame", type=float, default=float("nan"), help="Reference ZCAL frame for dose B. Default is the minimum selected/data frame.")
    ap.add_argument(
        "--matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention. j_minus_i is the production signed-row default; i_minus_j is a diagnostic.",
    )
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true", help="Use alternate fractional coordinates as an alternate-hand control run.")
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all rows.")
    ap.add_argument("--top-isigma", type=int, default=0, help="Keep the strongest I/sigma rows after other filters.")
    ap.add_argument("--row-index-file", default="", help="Optional text/CSV file of zero-based XDS row indices to predict.")
    ap.add_argument("--save-row-table", default="", help="Optional CSV with row-wise metadata, predictions, and residuals.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--compute-backend", choices=COMPUTE_BACKENDS, default="scipy")
    ap.add_argument("--jax-batch-size", type=int, default=0, help="JAX matrix batch size. 0 chooses a conservative size from matrix dimension and device count.")
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0, help="Pad JAX matrix dimensions to this multiple. 0 keeps exact sizes unless a *_padded backend is selected, where 0 means 8.")
    ap.add_argument("--bethe-strong-count", type=int, default=48, help="Strong-beam block size for scipy_bethe. Values >= n_basis recover the dense SciPy matrix.")
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target", help="Strong/weak-beam reduction used by scipy_bethe.")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005, help="Promote weak beams whose diagonal denominator is this close to the Schur energy.")
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5, help="Small-denominator floor for scipy_bethe Schur corrections.")
    ap.add_argument(
        "--row-plan-cache",
        choices=["off", "exact", "readwrite", "rebuild"],
        default="off",
        help=(
            "Cache row geometry, basis, Sg profiles, and h_j-h_i indices. "
            "exact keys the basis to the current model; readwrite is the "
            "opt-in frozen-basis mode across model updates."
        ),
    )
    ap.add_argument("--positive-only", action="store_true", help="Ignore non-positive XDS IOBS rows for fitting/evaluation.")
    ap.add_argument("--min-peak", type=float, default=90.0, help="Default PEAK/partiality cutoff for XDS row selection. Use 0 for a true full-row diagnostic.")
    ap.add_argument("--min-corr", type=float, default=-999.0)
    ap.add_argument("--max-resolution", type=float, default=0.0, help="Minimum d spacing in Angstrom; 0 disables.")
    args = ap.parse_args()

    cfg = Config(
        n_basis=args.n_basis,
        n_points=args.n_points,
        sg_window=args.sg_window,
        sg_window_mode=args.sg_window_mode,
        sg_window_min=args.sg_window_min,
        sg_window_max=args.sg_window_max,
        sg_window_scale=args.sg_window_scale,
        sg_window_frame_floor=args.sg_window_frame_floor,
        sg_jitter_mode=args.sg_jitter_mode,
        sg_jitter_sigma0=args.sg_jitter_sigma0,
        sg_jitter_sigma_d_inv=args.sg_jitter_sigma_d_inv,
        sg_jitter_sigma_frame=args.sg_jitter_sigma_frame,
        sg_jitter_points=args.sg_jitter_points,
        sg_jitter_student_df=args.sg_jitter_student_df,
        sg_jitter_width_factor=args.sg_jitter_width_factor,
        mosaicity_mode=args.mosaicity_mode,
        mosaicity_sigma_deg=args.mosaicity_sigma_deg,
        mosaicity_max_sigma_deg=args.mosaicity_max_sigma_deg,
        mosaicity_points=args.mosaicity_points,
        mosaicity_support_sigma_deg=args.mosaicity_support_sigma_deg,
        mosaicity_support_points=args.mosaicity_support_points,
        mosaicity_support_width_sigma=args.mosaicity_support_width_sigma,
        geometry_path_file=args.geometry_path,
        profile_basis_mode=args.profile_basis_mode,
        profile_coordinate=args.profile_coordinate,
        profile_angle_max_half_width_deg=args.profile_angle_max_half_width_deg,
        profile_quadrature=args.profile_quadrature,
        profile_accumulation=args.profile_accumulation,
        topup_d_min=args.topup_d_min,
        max_support_sg=args.max_support_sg,
        thickness_A=args.thickness,
        thickness_model=args.thickness_model,
        thickness_grad_frac=args.thickness_grad_frac,
        thickness_sigma_frac=args.thickness_sigma_frac,
        thickness_points=args.thickness_points,
        group_width_deg=args.group_width_deg,
        variance_a=args.variance_a,
        variance_b_corr=args.variance_b_corr,
        variance_c_peak=args.variance_c_peak,
        frame_offset=args.frame_offset,
        ug_d_min=args.ug_d_min,
        absorption_mode=args.absorption_mode,
        absorption_mu=args.absorption_mu,
        absorption_frac=args.absorption_frac,
        absorption_mu_g2=args.absorption_mu_g2,
        absorption_mu_theta2=args.absorption_mu_theta2,
        absorption_tds_b=args.absorption_tds_b,
        dose_b_mode=args.dose_b_mode,
        dose_b_per_frame=args.dose_b_per_frame,
        dose_b_ref_frame=args.dose_b_ref_frame,
        compute_backend=args.compute_backend,
        jax_batch_size=args.jax_batch_size,
        jax_matrix_bucket_size=args.jax_matrix_bucket_size,
        bethe_strong_count=args.bethe_strong_count,
        bethe_variant=args.bethe_variant,
        bethe_promote_denom=args.bethe_promote_denom,
        bethe_denom_eps=args.bethe_denom_eps,
        matrix_convention=args.matrix_convention,
        row_plan_cache=args.row_plan_cache,
        free_fraction=args.free_fraction,
        split_seed=args.split_seed,
    )
    data_all = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)

    if args.row_index_file:
        row_indices = read_row_index_file(args.row_index_file)
        indices = np.asarray(sorted(set(row_indices)), dtype=int)
        indices = indices[(indices >= 0) & (indices < len(data_all.iobs))]
    else:
        mask = np.isfinite(data_all.sigma) & (data_all.sigma > 0.0) & np.isfinite(data_all.zcal)
        if args.positive_only:
            mask &= data_all.iobs > 0.0
        if args.min_peak > 0:
            mask &= data_all.peak >= args.min_peak
        if args.min_corr > -999:
            mask &= data_all.corr >= args.min_corr
        if args.max_resolution > 0:
            dvals = np.asarray([data_all.cell.calculate_d(tuple(int(v) for v in h)) for h in data_all.hkl])
            mask &= dvals >= args.max_resolution
        indices = np.where(mask)[0]
    if args.top_isigma and args.top_isigma > 0 and len(indices) > args.top_isigma:
        isigi = data_all.iobs[indices] / np.maximum(data_all.sigma[indices], 1e-12)
        indices = indices[np.argsort(isigi)[::-1][: int(args.top_isigma)]]
    if args.limit and args.limit > 0:
        indices = indices[: int(args.limit)]

    model = parse_model(Path(args.res))
    if args.alternate_hand:
        model = alternate_hand_model(model)
    res_hash = sha256(Path(args.res))
    cache_dir = None if str(args.cache_dir).lower() == "none" else Path(args.cache_dir)
    wall0 = time.perf_counter()
    port = XDSBlochPort(data_all, model, cfg, cache_dir=cache_dir, res_hash=res_hash)
    pred_subset = port.predict_rows(indices, workers=args.workers, chunksize=args.chunksize)
    wall_seconds = time.perf_counter() - wall0
    full_dyn = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_born = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_basis_size = np.zeros(len(data_all.iobs), dtype=int)
    full_sg_resid = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_center_sg = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_omega_bragg = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_omega_delta = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_omega_bragg_root_available = np.zeros(len(data_all.iobs), dtype=bool)
    full_sg_window = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_domega = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_dframe = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_window_raw = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_window_half_frames = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_window_clipped = np.zeros(len(data_all.iobs), dtype=bool)
    full_sg_jitter_sigma = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_jitter_d_inv = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_jitter_frame_fraction = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_jitter_kernel_points = np.zeros(len(data_all.iobs), dtype=int)
    full_sg_profile_points_total = np.zeros(len(data_all.iobs), dtype=int)
    full_sg_profile_weight_sum = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_profile_same_branch_continuation_used = np.zeros(len(data_all.iobs), dtype=bool)
    full_sg_profile_same_branch_continuation_segments = np.ones(len(data_all.iobs), dtype=int)
    full_dyn[pred_subset["indices"]] = pred_subset["dyn"]
    full_born[pred_subset["indices"]] = pred_subset["born"]
    full_basis_size[pred_subset["indices"]] = pred_subset["basis_size"]
    full_sg_resid[pred_subset["indices"]] = pred_subset["sg_residual"]
    full_center_sg[pred_subset["indices"]] = pred_subset["center_sg"]
    full_omega_bragg[pred_subset["indices"]] = pred_subset["omega_bragg"]
    full_omega_delta[pred_subset["indices"]] = pred_subset["omega_delta"]
    full_omega_bragg_root_available[pred_subset["indices"]] = pred_subset.get(
        "omega_bragg_root_available",
        np.isfinite(pred_subset["omega_bragg"]),
    )
    full_sg_window[pred_subset["indices"]] = pred_subset["sg_window"]
    full_sg_domega[pred_subset["indices"]] = pred_subset["sg_dSg_domega"]
    full_sg_dframe[pred_subset["indices"]] = pred_subset["sg_dSg_dframe"]
    full_sg_window_raw[pred_subset["indices"]] = pred_subset["sg_window_raw"]
    full_sg_window_half_frames[pred_subset["indices"]] = pred_subset["sg_window_half_frames"]
    full_sg_window_clipped[pred_subset["indices"]] = pred_subset["sg_window_clipped"]
    full_sg_jitter_sigma[pred_subset["indices"]] = pred_subset["sg_jitter_sigma"]
    full_sg_jitter_d_inv[pred_subset["indices"]] = pred_subset["sg_jitter_d_inv"]
    full_sg_jitter_frame_fraction[pred_subset["indices"]] = pred_subset["sg_jitter_frame_fraction"]
    full_sg_jitter_kernel_points[pred_subset["indices"]] = pred_subset["sg_jitter_kernel_points"]
    full_sg_profile_points_total[pred_subset["indices"]] = pred_subset["sg_profile_points_total"]
    full_sg_profile_weight_sum[pred_subset["indices"]] = pred_subset["sg_profile_weight_sum"]
    full_sg_profile_same_branch_continuation_used[pred_subset["indices"]] = pred_subset[
        "sg_profile_same_branch_continuation_used"
    ]
    full_sg_profile_same_branch_continuation_segments[pred_subset["indices"]] = pred_subset[
        "sg_profile_same_branch_continuation_segments"
    ]
    support_by_idx = {
        int(idx): basis
        for idx, basis in zip(pred_subset["indices"], pred_subset["support_hkls"], strict=True)
    }
    pred = {"dyn": full_dyn, "born": full_born, "groups": port.group_indices}
    valid = np.isfinite(full_dyn) & np.isfinite(full_born)
    eval_data = data_all
    eval_out = evaluate_predictions(eval_data, cfg, pred, fit_scales_on="work", model=model)
    row_table_path = ""
    if args.save_row_table:
        row_table_path = str(Path(args.save_row_table).resolve())
        sig_eff = effective_sigma(data_all, cfg)
        work_mask = split_masks(
            data_all.hkl,
            cfg.free_fraction,
            cfg.split_seed,
            symmetry_ops=model.get("symmetry_ops"),
        )["work"]
        dyn_scaled, dyn_scales = scale_by_groups(data_all.iobs, sig_eff, full_dyn, port.group_indices, work_mask)
        born_scaled, born_scales = scale_by_groups(data_all.iobs, sig_eff, full_born, port.group_indices, work_mask)
        dvals = np.asarray([data_all.cell.calculate_d(tuple(int(v) for v in h)) for h in data_all.hkl])
        Path(args.save_row_table).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_row_table, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "row_index",
                    "h",
                    "k",
                    "l",
                    "d_A",
                    "iobs",
                    "sigma",
                    "isigi",
                    "xcal",
                    "ycal",
                    "zcal",
                    "xobs",
                    "yobs",
                    "zobs",
                    "dx_obs_cal",
                    "dy_obs_cal",
                    "dz_obs_cal",
                    "rlp",
                    "peak",
                    "corr",
                    "maxc",
                    "psi",
                    "omega_deg",
                    "omega_bragg_deg",
                    "omega_bragg_minus_center_deg",
                    "omega_bragg_root_available",
                    "center_sg_inv_A",
                    "max_sg_solve_residual",
                    "sg_window_inv_A",
                    "sg_window_raw_inv_A",
                    "sg_dSg_domega_inv_A_per_deg",
                    "sg_dSg_dframe_inv_A",
                    "sg_window_half_frames",
                    "sg_window_clipped",
                    "sg_jitter_sigma_inv_A",
                    "sg_jitter_d_inv_A",
                    "sg_jitter_frame_fraction",
                    "sg_jitter_kernel_points",
                    "sg_profile_points_total",
                    "sg_profile_weight_sum",
                    "sg_profile_same_branch_continuation_used",
                    "sg_profile_same_branch_continuation_segments",
                    "group",
                    "basis_size",
                    "support_hkls_json",
                    "dyn_raw",
                    "born_raw",
                    "dyn_scaled",
                    "born_scaled",
                    "dyn_minus_iobs_over_sigma",
                    "born_minus_iobs_over_sigma",
                    "dyn_over_born",
                    "dyn_group_scale",
                    "born_group_scale",
                ]
            )
            for idx in indices:
                g = int(port.group_indices[idx])
                writer.writerow(
                    [
                        int(idx),
                        *[int(v) for v in data_all.hkl[idx]],
                        float(dvals[idx]),
                        float(data_all.iobs[idx]),
                        float(sig_eff[idx]),
                        float(data_all.iobs[idx] / max(sig_eff[idx], 1e-12)),
                        float(data_all.xcal[idx]),
                        float(data_all.ycal[idx]),
                        float(data_all.zcal[idx]),
                        float(data_all.xobs[idx]),
                        float(data_all.yobs[idx]),
                        float(data_all.zobs[idx]),
                        float(data_all.xobs[idx] - data_all.xcal[idx]),
                        float(data_all.yobs[idx] - data_all.ycal[idx]),
                        float(data_all.zobs[idx] - data_all.zcal[idx]),
                        float(data_all.rlp[idx]),
                        float(data_all.peak[idx]),
                        float(data_all.corr[idx]),
                        float(data_all.maxc[idx]),
                        float(data_all.psi[idx]),
                        float(port.omega_from_z(data_all.zcal[idx])),
                        float(full_omega_bragg[idx]),
                        float(full_omega_delta[idx]),
                        int(bool(full_omega_bragg_root_available[idx])),
                        float(full_center_sg[idx]),
                        float(full_sg_resid[idx]),
                        float(full_sg_window[idx]),
                        float(full_sg_window_raw[idx]),
                        float(full_sg_domega[idx]),
                        float(full_sg_dframe[idx]),
                        float(full_sg_window_half_frames[idx]),
                        bool(full_sg_window_clipped[idx]),
                        float(full_sg_jitter_sigma[idx]),
                        float(full_sg_jitter_d_inv[idx]),
                        float(full_sg_jitter_frame_fraction[idx]),
                        int(full_sg_jitter_kernel_points[idx]),
                        int(full_sg_profile_points_total[idx]),
                        float(full_sg_profile_weight_sum[idx]),
                        int(full_sg_profile_same_branch_continuation_used[idx]),
                        int(full_sg_profile_same_branch_continuation_segments[idx]),
                        g,
                        int(full_basis_size[idx]),
                        json.dumps(support_by_idx.get(int(idx), []), separators=(",", ":")),
                        float(full_dyn[idx]),
                        float(full_born[idx]),
                        float(dyn_scaled[idx]),
                        float(born_scaled[idx]),
                        float((dyn_scaled[idx] - data_all.iobs[idx]) / max(sig_eff[idx], 1e-12)) if np.isfinite(dyn_scaled[idx]) else float("nan"),
                        float((born_scaled[idx] - data_all.iobs[idx]) / max(sig_eff[idx], 1e-12)) if np.isfinite(born_scaled[idx]) else float("nan"),
                        float(full_dyn[idx] / max(full_born[idx], 1e-30)) if np.isfinite(full_dyn[idx]) and np.isfinite(full_born[idx]) else float("nan"),
                        float(dyn_scales.get(g, float("nan"))),
                        float(born_scales.get(g, float("nan"))),
                    ]
                )
    out = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "hkl_transform": data_all.hkl_transform_spec,
            "raw_xds_cell": [
                data_all.raw_cell.a,
                data_all.raw_cell.b,
                data_all.raw_cell.c,
                data_all.raw_cell.alpha,
                data_all.raw_cell.beta,
                data_all.raw_cell.gamma,
            ],
            "axis_derived_source_cell": [
                data_all.axis_derived_source_cell.a,
                data_all.axis_derived_source_cell.b,
                data_all.axis_derived_source_cell.c,
                data_all.axis_derived_source_cell.alpha,
                data_all.axis_derived_source_cell.beta,
                data_all.axis_derived_source_cell.gamma,
            ],
            "axis_derived_source_cell_basis": data_all.axis_derived_source_cell_basis,
            "simulation_cell": [
                data_all.cell.a,
                data_all.cell.b,
                data_all.cell.c,
                data_all.cell.alpha,
                data_all.cell.beta,
                data_all.cell.gamma,
            ],
            "simulation_cell_basis": data_all.working_cell_basis,
            "integration_profile_provenance": data_all.integration_profile_provenance(),
            "res": str(Path(args.res).resolve()),
            "res_sha256": res_hash,
        },
        "bloch_kernel_contract": frozen_kernel_contract(cfg),
        "config": cfg.__dict__,
        "model_hand": "alternate_fractional_xyz" if args.alternate_hand else "original",
        "filters": {
            "positive_only": bool(args.positive_only),
            "min_peak": args.min_peak,
            "min_corr": args.min_corr,
            "max_resolution": args.max_resolution,
            "limit": args.limit,
            "top_isigma": args.top_isigma,
            "row_index_file": args.row_index_file,
            "selected_rows": int(len(indices)),
            "valid_prediction_rows": int(np.count_nonzero(valid)),
        },
        "row_table": row_table_path,
        "timing": {"prediction_seconds": pred_subset["seconds"], "wall_seconds": wall_seconds},
        "diagnostics": {
            "ug_cache": getattr(port, "ug_cache_status", {"status": "disabled"}),
            "row_plan_cache": pred_subset.get("row_plan_cache_status", {"status": "off"}),
            "row_plan_cache_unique_diffs": int(pred_subset.get("row_plan_cache_unique_diffs", 0)),
            "basis_size_min": int(np.min(pred_subset["basis_size"])) if len(indices) else 0,
            "basis_size_median": float(np.median(pred_subset["basis_size"])) if len(indices) else 0.0,
            "basis_size_max": int(np.max(pred_subset["basis_size"])) if len(indices) else 0,
            "max_sg_solve_residual": float(np.max(pred_subset["sg_residual"])) if len(indices) else 0.0,
            "sg_window_mode": cfg.sg_window_mode,
            "sg_window_min_used": float(np.nanmin(pred_subset["sg_window"])) if len(indices) else float("nan"),
            "sg_window_median_used": float(np.nanmedian(pred_subset["sg_window"])) if len(indices) else float("nan"),
            "sg_window_max_used": float(np.nanmax(pred_subset["sg_window"])) if len(indices) else float("nan"),
            "sg_window_clipped_count": int(np.count_nonzero(pred_subset["sg_window_clipped"])) if len(indices) else 0,
            "sg_jitter_mode": cfg.sg_jitter_mode,
            "mosaicity_mode": cfg.mosaicity_mode,
            "mosaicity_sigma_deg": cfg.mosaicity_sigma_deg,
            "mosaicity_max_sigma_deg": cfg.mosaicity_max_sigma_deg,
            "mosaicity_points": cfg.mosaicity_points,
            "geometry_path": (
                None
                if port.geometry_path is None
                else {
                    "path": port.geometry_path["path"],
                    "sha256": port.geometry_path["sha256"],
                    "frame_center": port.geometry_path["frame_center"],
                    "frame_half_range": port.geometry_path["frame_half_range"],
                    "coefficients": {
                        name: values.tolist()
                        for name, values in port.geometry_path["coefficients"].items()
                    },
                    "tilt_axes_lab": port.geometry_path["tilt_axes_lab"].tolist(),
                    "source_hkl_sha256": port.geometry_path["source_hkl_sha256"],
                    "bounds": port.geometry_path["bounds"],
                    "fit_source": port.geometry_path.get("fit_source"),
                    "validation": port.geometry_path.get("validation"),
                }
            ),
            "profile_quadrature": cfg.profile_quadrature,
            "profile_accumulation": cfg.profile_accumulation,
            "sg_jitter_sigma_median_used": float(np.nanmedian(pred_subset["sg_jitter_sigma"])) if len(indices) else float("nan"),
            "sg_jitter_sigma_max_used": float(np.nanmax(pred_subset["sg_jitter_sigma"])) if len(indices) else float("nan"),
            "sg_profile_points_median_total": float(np.median(pred_subset["sg_profile_points_total"])) if len(indices) else 0.0,
            "sg_profile_weight_sum_median": float(np.nanmedian(pred_subset["sg_profile_weight_sum"])) if len(indices) else 0.0,
            "sg_profile_same_branch_continuation_rows": int(
                np.count_nonzero(pred_subset["sg_profile_same_branch_continuation_used"])
            )
            if len(indices)
            else 0,
            "sg_profile_same_branch_continuation_max_segments": int(
                np.max(
                    pred_subset["sg_profile_same_branch_continuation_segments"][
                        pred_subset["sg_profile_same_branch_continuation_used"]
                    ]
                )
            )
            if len(indices) and np.any(pred_subset["sg_profile_same_branch_continuation_used"])
            else 1,
            "median_abs_dSg_dframe": float(np.nanmedian(np.abs(pred_subset["sg_dSg_dframe"]))) if len(indices) else float("nan"),
            "median_dyn_over_born": float(np.nanmedian(full_dyn[valid] / np.clip(full_born[valid], 1e-30, None))) if np.any(valid) else float("nan"),
        },
        "metrics": eval_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=True))
    print(json.dumps(out["metrics"], indent=2, allow_nan=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
