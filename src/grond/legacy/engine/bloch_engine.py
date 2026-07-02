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
import time
from dataclasses import dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import gemmi
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig, eigh, expm
from scipy.optimize import minimize
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
ROW_PLAN_CACHE_SCHEMA = 1


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
    The padded backends default to multiples of 8 because 370 benchmarks showed
    this keeps compilation count low without paying too much extra eigensolver
    work from full max-size padding.
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
    for t, w in zip(np.asarray(thick, dtype=float), np.asarray(thick_w, dtype=float)):
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


def row_plan_cache_validation_errors(cache: dict, indices, expected_key: str, expected_meta: dict) -> list[str]:
    """Return reasons a row-plan cache payload cannot be trusted."""

    required = {
        "indices",
        "basis_offsets",
        "basis_flat",
        "profile_offsets",
        "profile_weights_flat",
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
    for field in ("hkl_sha256", "hkl_transform", "indices_sha256", "n_basis", "n_points", "profile_quadrature", "profile_accumulation"):
        if meta.get(field) != expected_meta.get(field):
            errors.append(f"metadata mismatch: {field}")
    if "indices" in cache and not np.array_equal(np.asarray(cache["indices"], dtype=np.int64), np.asarray(indices, dtype=np.int64)):
        errors.append("index order mismatch")
    if "basis_offsets" in cache:
        offsets = np.asarray(cache["basis_offsets"], dtype=np.int64)
        if len(offsets) != len(indices) + 1:
            errors.append("basis offset count mismatch")
        elif np.any(np.diff(offsets) <= 1):
            errors.append("basis offsets do not contain direct and target beams for every row")
    if "profile_offsets" in cache:
        offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        if len(offsets) != len(indices) + 1:
            errors.append("profile offset count mismatch")
        elif np.any(np.diff(offsets) <= 0):
            errors.append("profile offsets contain an empty row profile")
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


def parse_hkl_transform(spec: str | None) -> tuple[np.ndarray, str]:
    """Parse a signed axis permutation mapping XDS HKLs into model HKLs.

    The returned matrix T uses column-vector convention:
    h_model = T @ h_xds.  For row arrays this is hkl_model = hkl_xds @ T.T.
    """
    text = "identity" if spec is None else str(spec).strip()
    if text.lower() in {"", "identity", "none", "h,k,l", "(h,k,l)"}:
        return np.eye(3, dtype=int), "identity"
    cleaned = text.lower().replace("(", "").replace(")", "").replace(" ", "")
    parts = cleaned.split(",")
    if len(parts) != 3:
        raise ValueError(f"HKL transform must look like 'h,k,l' or 'l,-k,h', got {spec!r}")
    axes = {"h": 0, "k": 1, "l": 2}
    T = np.zeros((3, 3), dtype=int)
    used = []
    canonical = []
    for out_i, part in enumerate(parts):
        sign = 1
        token = part
        if token.startswith("+"):
            token = token[1:]
        elif token.startswith("-"):
            sign = -1
            token = token[1:]
        if token not in axes:
            raise ValueError(f"Unknown HKL transform axis {part!r} in {spec!r}")
        in_i = axes[token]
        if in_i in used:
            raise ValueError(f"HKL transform must be a permutation, got repeated axis in {spec!r}")
        used.append(in_i)
        T[out_i, in_i] = sign
        canonical.append(("-" if sign < 0 else "") + token)
    det = round(float(np.linalg.det(T)))
    if abs(det) != 1:
        raise ValueError(f"HKL transform must have determinant +/-1, got det={det} for {spec!r}")
    return T, ",".join(canonical)


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
    element = tokens[1]
    vals = [float(v) for v in tokens[2:]]
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
        text = row.get("_space_group_symop_operation_xyz") or row.get("_symmetry_equiv_pos_as_xyz")
        if text:
            records.append(gemmi_op_record(str(text).strip().strip("'\"")))
    if records:
        return unique_symmetry_records(records)
    number = block.find_value("_space_group_IT_number") or block.find_value("_symmetry_Int_Tables_number")
    if number:
        sg = gemmi.find_spacegroup_by_number(int(cif_number(number)))
        if sg is not None:
            return unique_symmetry_records(gemmi_op_record(op.triplet()) for op in sg.operations())
    return [gemmi_op_record("x,y,z")]


def shelx_occupancy(value: float) -> float:
    occ = float(value)
    if abs(occ) >= 10.0:
        occ = (-1.0 if occ < 0.0 else 1.0) * (abs(occ) - 10.0)
    return occ


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
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("REM"):
            continue
        tokens = stripped.split()
        key = tokens[0].upper()
        if key == "CELL":
            cell = tuple(float(v) for v in tokens[2:8])
        elif key == "SFAC":
            sfac.append(parse_sfac(tokens))
        elif key == "LATT" and len(tokens) > 1:
            latt = int(float(tokens[1]))
        elif key == "SYMM":
            symm_texts.append(stripped[4:].strip())
        elif key in {"END", "HKLF"}:
            if key == "HKLF":
                break
        elif key in {
            "AFIX",
            "RIGU",
            "L.S.",
            "PLAN",
            "CONF",
            "BOND",
            "LIST",
            "FMAP",
            "MORE",
            "ACTA",
            "EXTI",
            "BASF",
            "TWIN",
            "SWAT",
            "SQUEEZE",
            "OMIT",
            "WGHT",
            "FVAR",
            "UNIT",
            "ZERR",
            "TITL",
        }:
            continue
        elif len(tokens) >= 7 and tokens[1].lstrip("-").isdigit():
            sfac_index = int(tokens[1]) - 1
            vals = [float(v) for v in tokens[6:]]
            atom = {
                "name": tokens[0],
                "sfac_index": sfac_index,
                "element": sfac[sfac_index]["element"] if 0 <= sfac_index < len(sfac) else "",
                "x": float(tokens[2]),
                "y": float(tokens[3]),
                "z": float(tokens[4]),
                "occ": shelx_occupancy(float(tokens[5])),
                "raw_adp": vals,
            }
            if len(vals) >= 6:
                atom.update({"adp_type": "anis", "u11": vals[0], "u22": vals[1], "u33": vals[2], "u23": vals[3], "u13": vals[4], "u12": vals[5]})
                current_parent_ueq = (vals[0] + vals[1] + vals[2]) / 3.0
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
    return {"cell": cell, "sfac": sfac, "atoms": atoms, "model_source": "shelx_res", "ignored_exti": None, "symmetry_ops": shelx_symmetry_records(latt, symm_texts), "shelx_latt": latt}


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
    if not all(np.isfinite(cell)):
        raise ValueError(f"No usable cell constants in {path}")

    _, sfac_rows = loop_rows(block, "_atom_type_symbol")
    sfac = []
    symbol_to_index = {}
    for row in sfac_rows:
        symbol = str(row.get("_atom_type_symbol", "")).strip().strip("'\"")
        if not symbol:
            continue
        pairs = []
        for i in range(1, 5):
            a = cif_number(row.get(f"_atom_type_scat_Cromer_Mann_a{i}", 0.0), 0.0)
            b = cif_number(row.get(f"_atom_type_scat_Cromer_Mann_b{i}", 0.0), 0.0)
            pairs.append((a, b))
        c = cif_number(row.get("_atom_type_scat_Cromer_Mann_c", 0.0), 0.0)
        symbol_to_index[symbol.upper()] = len(sfac)
        sfac.append({"element": symbol, "pairs": pairs, "c": c, "raw": [v for pair in pairs for v in pair] + [c]})
    if not sfac:
        raise ValueError(f"No _atom_type scattering-factor loop found in {path}")

    _, aniso_rows = loop_rows(block, "_atom_site_aniso_label")
    aniso = {}
    for row in aniso_rows:
        label = str(row.get("_atom_site_aniso_label", "")).strip().strip("'\"")
        if not label:
            continue
        aniso[label] = {
            "adp_type": "anis",
            "u11": cif_number(row.get("_atom_site_aniso_U_11", 0.0), 0.0),
            "u22": cif_number(row.get("_atom_site_aniso_U_22", 0.0), 0.0),
            "u33": cif_number(row.get("_atom_site_aniso_U_33", 0.0), 0.0),
            "u12": cif_number(row.get("_atom_site_aniso_U_12", 0.0), 0.0),
            "u13": cif_number(row.get("_atom_site_aniso_U_13", 0.0), 0.0),
            "u23": cif_number(row.get("_atom_site_aniso_U_23", 0.0), 0.0),
        }

    _, atom_rows = loop_rows(block, "_atom_site_label")
    atoms = []
    for row in atom_rows:
        label = str(row.get("_atom_site_label", "")).strip().strip("'\"")
        element = str(row.get("_atom_site_type_symbol", "")).strip().strip("'\"")
        if not label or not element:
            continue
        sfac_index = symbol_to_index.get(element.upper())
        if sfac_index is None:
            raise ValueError(f"Atom {label} uses element {element}, but no matching _atom_type row was found in {path}")
        atom = {
            "name": label,
            "sfac_index": int(sfac_index),
            "element": element,
            "x": cif_number(row.get("_atom_site_fract_x")),
            "y": cif_number(row.get("_atom_site_fract_y")),
            "z": cif_number(row.get("_atom_site_fract_z")),
            "occ": cif_number(row.get("_atom_site_occupancy", 1.0), 1.0),
        }
        if label in aniso and str(row.get("_atom_site_adp_type", "")).strip().lower() == "uani":
            atom.update(aniso[label])
        else:
            atom["adp_type"] = "iso"
            atom["uiso"] = cif_number(row.get("_atom_site_U_iso_or_equiv", 0.05), 0.05)
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


def atom_dw(atom, hkl, cell_tuple):
    h, k, l = (float(v) for v in hkl)
    if atom.get("adp_type") == "anis":
        uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
        o1 = uc.orthogonalize(gemmi.Fractional(1, 0, 0))
        o2 = uc.orthogonalize(gemmi.Fractional(0, 1, 0))
        o3 = uc.orthogonalize(gemmi.Fractional(0, 0, 1))
        O = np.array([
            [o1.x, o2.x, o3.x],
            [o1.y, o2.y, o3.y],
            [o1.z, o2.z, o3.z]
        ])
        F = np.linalg.inv(O)
        s_orth = F.T @ np.array([h, k, l])
        u11, u22, u33 = float(atom["u11"]), float(atom["u22"]), float(atom["u33"])
        u12, u13, u23 = float(atom["u12"]), float(atom["u13"]), float(atom["u23"])
        U = np.array([
            [u11, u12, u13],
            [u12, u22, u23],
            [u13, u23, u33]
        ])
        arg = s_orth.T @ U @ s_orth
        return math.exp(-2.0 * math.pi * math.pi * arg)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, cell_tuple)
    return math.exp(-8.0 * math.pi * math.pi * float(atom.get("uiso", 0.05)) * s2)


def atom_dw_array(atom, h, k, l, s2, cell_tuple):
    if atom.get("adp_type") == "anis":
        uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
        o1 = uc.orthogonalize(gemmi.Fractional(1, 0, 0))
        o2 = uc.orthogonalize(gemmi.Fractional(0, 1, 0))
        o3 = uc.orthogonalize(gemmi.Fractional(0, 0, 1))
        O = np.array([
            [o1.x, o2.x, o3.x],
            [o1.y, o2.y, o3.y],
            [o1.z, o2.z, o3.z]
        ])
        F = np.linalg.inv(O)
        hb, kb, lb = np.broadcast_arrays(h, k, l)
        shape = hb.shape
        hkl_stack = np.stack([hb.ravel(), kb.ravel(), lb.ravel()], axis=0)
        s_orth = (F.T @ hkl_stack).reshape((3,) + shape)
        u11, u22, u33 = float(atom["u11"]), float(atom["u22"]), float(atom["u33"])
        u12, u13, u23 = float(atom["u12"]), float(atom["u13"]), float(atom["u23"])
        sx = s_orth[0]
        sy = s_orth[1]
        sz = s_orth[2]
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


def fcalc_res(model, hkl):
    h, k, l = (int(v) for v in hkl)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, model["cell"])
    total = 0.0 + 0.0j
    for atom in model["atoms"]:
        sf = sfac_value(model["sfac"][atom["sfac_index"]], s2)
        dw = atom_dw(atom, (h, k, l), model["cell"])
        xyz = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
        for rot, trans in model_symmetry_ops(model):
            pos = rot @ xyz + trans
            phase = 2.0 * math.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            total += atom["occ"] * sf * dw * complex(math.cos(phase), math.sin(phase))
    return total


def fcalc_res_grid(model, h, k, l):
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    l = np.asarray(l, dtype=float)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, model["cell"])
    total = np.zeros(h.shape, dtype=np.complex128)
    symops = model_symmetry_ops(model)
    for atom in model["atoms"]:
        sf = sfac_value_array(model["sfac"][atom["sfac_index"]], s2)
        dw = atom_dw_array(atom, h, k, l, s2, model["cell"])
        amp = float(atom["occ"]) * sf * dw
        xyz = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
        for rot, trans in symops:
            pos = rot @ xyz + trans
            phase = 2.0 * math.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            total += amp * np.exp(1j * phase)
    return total


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
    profile_quadrature: str = "uniform"
    profile_accumulation: str = "sum"
    topup_d_min: float = 0.8
    max_support_sg: float = 0.005
    thickness_A: float = 1240.0
    thickness_model: str = "uniform_gradient"
    thickness_grad_frac: float = 0.25
    thickness_sigma_frac: float = 0.0
    thickness_points: int = 7
    gamma: float = 1.3914
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
        "geometry": {
            "g(alpha)": "R(alpha) @ UB @ h",
            "Sg": "(k^2 - |s0 + g(alpha)|^2) / (2*k)",
            "profile_mode": "target_sg_to_angle",
            "profile_quadrature": None if cfg is None else str(getattr(cfg, "profile_quadrature", "uniform")),
            "profile_accumulation": None if cfg is None else str(getattr(cfg, "profile_accumulation", "sum")),
            "profile_weight_mode": "rocking-profile samples are summed by default; legacy normalized averaging is available as profile_accumulation=mean",
            "profile_width_mode": "fixed unless cfg.sg_window_mode explicitly requests an auditable XDS metadata-derived per-row width",
            "orientation_distribution": "optional target-Sg jitter convolution after the base uniform-Sg grid; disabled by default",
            "basis_sg_recompute": "all basis Sg recomputed at every solved target-Sg angle",
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
            "absorption": "optional non-Hermitian optical-potential diagnostics; default none. Modes are mean, beam_dependent, offdiag_fraction, offdiag_tds, and midgley_like.",
            "rejected_nonhermitian_modes": [
                "row-specific imaginary diagonal terms",
                "EXTI-like extinction inside the raw Bloch matrix",
                "per-row fitted absorptive form factors",
            ],
            "solver": "scipy.linalg.eigh/eig by default; optional experimental scipy expm/expm_multiply, promoted Bethe/Schur, and JAX batched paths only when explicitly requested and probed",
        },
        "basis": {
            "mode": "direct_target_topup",
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
                row[0] = int(row[0])
                row[1] = int(row[1])
                row[2] = int(row[2])
                row[4] = abs(row[4])
                rows.append(tuple(row))
        arr = np.array(rows, dtype=float)
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
        self.alf0 = arr[:, 15].astype(float)
        self.bet0 = arr[:, 16].astype(float)
        self.alf1 = arr[:, 17].astype(float)
        self.bet1 = arr[:, 18].astype(float)
        self.psi = arr[:, 19].astype(float)
        self.iseg = arr[:, 20].astype(int)

        self.wavelength = float(self.header.get("X-RAY_WAVELENGTH", self.header.get("WAVELENGTH", "0.025079")).split()[0])
        self.k = 1.0 / self.wavelength
        self.starting_frame = float(self.header["STARTING_FRAME"].split()[0])
        self.starting_angle = float(self.header["STARTING_ANGLE"].split()[0])
        self.oscillation_range = float(self.header["OSCILLATION_RANGE"].split()[0])
        self.axis = np.array([float(x) for x in self.header["ROTATION_AXIS"].split()[:3]], dtype=float)
        self.axis = self.axis / max(float(np.linalg.norm(self.axis)), 1e-30)
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
        self.s0 = self.k * beam_dir / np.linalg.norm(beam_dir)
        cell_vals = [float(x) for x in self.header["UNIT_CELL_CONSTANTS"].split()[:6]]
        self.cell = gemmi.UnitCell(*cell_vals)
        va = np.array([float(x) for x in self.header["UNIT_CELL_A-AXIS"].split()[:3]], dtype=float)
        vb = np.array([float(x) for x in self.header["UNIT_CELL_B-AXIS"].split()[:3]], dtype=float)
        vc = np.array([float(x) for x in self.header["UNIT_CELL_C-AXIS"].split()[:3]], dtype=float)
        self.real_matrix = np.stack([va, vb, vc], axis=1)
        self.A = np.linalg.inv(self.real_matrix).T
        self.A_at_scan_points = None
        scan_a_file = self.header.get("DIALS_A_AT_SCAN_POINTS_FILE", "").split()
        if scan_a_file:
            p = Path(scan_a_file[0])
            if not p.exists() and not p.is_absolute():
                p = self.path.parent / p
            mats = []
            with p.open() as f:
                for line in f:
                    vals = [float(x) for x in line.split()]
                    if len(vals) == 9:
                        mats.append(np.asarray(vals, dtype=float).reshape(3, 3))
            if mats:
                self.A_at_scan_points = np.asarray(mats, dtype=float)
        self.variance_model = tuple(float(v) for v in self.header.get("VARIANCE_MODEL", "1.0 0.0").split()[:2])
        self.apply_hkl_transform(hkl_transform)

    def apply_hkl_transform(self, spec: str | None):
        T, canonical = parse_hkl_transform(spec)
        self.hkl_transform_spec = canonical
        self.hkl_transform_matrix = T
        self.raw_hkl = self.hkl.copy()
        self.raw_A = np.asarray(self.A, dtype=float).copy()
        self.raw_real_matrix = np.asarray(self.real_matrix, dtype=float).copy()
        self.raw_cell = gemmi.UnitCell(self.cell.a, self.cell.b, self.cell.c, self.cell.alpha, self.cell.beta, self.cell.gamma)
        if canonical == "identity":
            return
        inv_t = np.linalg.inv(T).astype(float)
        self.hkl = (self.hkl @ T.T).astype(int)
        self.A = np.asarray(self.A, dtype=float) @ inv_t
        self.real_matrix = np.linalg.inv(self.A.T)
        self.cell = cell_from_real_matrix(self.real_matrix)
        if self.A_at_scan_points is not None:
            self.A_at_scan_points = np.einsum("nij,jk->nik", np.asarray(self.A_at_scan_points, dtype=float), inv_t)
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
        z = self.z_from_omega(omega)
        if z <= 0:
            return self.A_at_scan_points[0]
        if z >= len(self.A_at_scan_points) - 1:
            return self.A_at_scan_points[-1]
        lo = int(math.floor(z))
        frac = float(z) - lo
        return (1.0 - frac) * self.A_at_scan_points[lo] + frac * self.A_at_scan_points[lo + 1]

    def recorded_proxy(self):
        return self.iobs * np.maximum(self.peak, 0.0) / 100.0 / np.maximum(self.rlp, 1e-12)


class XDSBlochPort:
    def __init__(self, data: XDSData, res_model, cfg: Config, cache_dir: Path | None = None, res_hash: str | None = None):
        self.data = data
        self.model = copy.deepcopy(res_model)
        self.cfg = cfg
        self.res_hash = res_hash or "unknown"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model["cell"] = (data.cell.a, data.cell.b, data.cell.c, data.cell.alpha, data.cell.beta, data.cell.gamma)
        self.h_lim, self.k_lim, self.l_lim = data.cell.get_hkl_limits(float(self.cfg.ug_d_min))
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
            "res_sha256": self.res_hash,
            "model_content_sha256": model_content_hash(self.model),
            "cell": [round(float(v), 6) for v in cell],
            "lims": [int(self.h_lim), int(self.k_lim), int(self.l_lim)],
            "gamma": float(self.cfg.gamma),
            "ug_d_min_A": float(self.cfg.ug_d_min),
            "formula": "gamma*Fe/(pi*V); P212121 RES/SFAC parser",
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16], payload

    def load_or_build_ug_cube(self):
        if self.cache_dir is None:
            return self.build_ug_cube()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key, meta = self.ug_cache_key()
        path = self.cache_dir / f"ug_cube_{key}.npz"
        if path.exists():
            loaded = np.load(path, allow_pickle=False)
            cube = loaded["ug_cube"]
            if tuple(int(v) for v in loaded["lims"]) == (self.h_lim, self.k_lim, self.l_lim):
                self.ug_cache_status = {"status": "hit", "path": str(path), "key": key, "meta": meta}
                return cube
        cube = self.build_ug_cube()
        np.savez_compressed(path, ug_cube=cube, lims=np.asarray([self.h_lim, self.k_lim, self.l_lim], dtype=int), meta=json.dumps(meta, sort_keys=True))
        self.ug_cache_status = {"status": "built", "path": str(path), "key": key, "meta": meta}
        return cube

    def build_ug_cube(self):
        h_vals = np.arange(-self.h_lim, self.h_lim + 1, dtype=float)
        k_vals = np.arange(-self.k_lim, self.k_lim + 1, dtype=float)
        l_vals = np.arange(-self.l_lim, self.l_lim + 1, dtype=float)
        hh, kk, ll = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
        factor = float(self.cfg.gamma) / (math.pi * self.data.cell.volume)
        return factor * fcalc_res_grid(self.model, hh, kk, ll)

    def ug(self, hkl):
        h, k, l = (int(v) for v in hkl)
        i, j, m = h + self.h_lim, k + self.k_lim, l + self.l_lim
        if i < 0 or j < 0 or m < 0 or i >= self.ug_cube.shape[0] or j >= self.ug_cube.shape[1] or m >= self.ug_cube.shape[2]:
            return 0.0 + 0.0j
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
            & (i < self.ug_cube.shape[0])
            & (j < self.ug_cube.shape[1])
            & (m < self.ug_cube.shape[2])
        )
        out = np.zeros(len(flat), dtype=np.complex128)
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
        hkls, uvals = [], []
        for h in range(-h_lim, h_lim + 1):
            for k in range(-k_lim, k_lim + 1):
                for l in range(-l_lim, l_lim + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    if self.data.cell.calculate_d((h, k, l)) >= self.cfg.topup_d_min:
                        hkls.append((h, k, l))
                        uvals.append(self.ug((h, k, l)))
        return np.asarray(hkls, dtype=int), np.asarray(uvals, dtype=np.complex128)

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
        omegas = np.array([self.omega_from_z(z) for z in self.data.zcal])
        return ((omegas - np.min(omegas)) // float(self.cfg.group_width_deg)).astype(int)

    def omega_from_z(self, z):
        return self.data.omega_from_z(float(z) + float(self.cfg.frame_offset))

    def gvecs(self, hkls, omega_deg):
        A = self.data.A_for_omega(omega_deg)
        theta = np.radians(float(omega_deg))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        R = self.data.axis_eye + s * self.data.axis_cross + (1.0 - c) * self.data.axis_cross2
        return (R @ A @ np.asarray(hkls, dtype=float).T).T

    def sg_for_hkls(self, hkls, omega_deg):
        g = self.gvecs(hkls, omega_deg)
        return (self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)) / (2.0 * self.data.k)

    def solve_omega_for_sg(self, hkl, omega0, target_sg):
        omega = float(omega0)
        for _ in range(10):
            sg = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega)[0])
            resid = sg - float(target_sg)
            if abs(resid) < 1e-8:
                break
            step = 1e-4
            sp = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega + step)[0])
            sm = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega - step)[0])
            deriv = (sp - sm) / (2.0 * step)
            if abs(deriv) < 1e-12:
                break
            omega -= resid / deriv
        return omega

    def sg_derivative_per_deg(self, hkl, omega_deg):
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
        for idx in np.argsort(score)[::-1]:
            if not ok[idx]:
                continue
            h = tuple(int(v) for v in self.pool_hkl[idx])
            if h in seen:
                continue
            basis.append(h)
            seen.add(h)
            if len(basis) >= int(self.cfg.n_basis):
                break
        return np.asarray(basis, dtype=int)

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
        return H

    def hamiltonian_from_static(self, static_H, sg_basis):
        H = np.array(static_H, dtype=np.complex128, copy=True)
        diag = np.diag_indices(H.shape[0])
        H[diag] += np.asarray(sg_basis, dtype=float)
        return H

    def hamiltonian(self, basis, sg_basis, absorb: bool = True, row_idx=None):
        static_H = self.hamiltonian_static_part(basis, absorb=absorb, row_idx=row_idx)
        return self.hamiltonian_from_static(static_H, sg_basis)

    def propagate_direct_to_target(self, H, thickness):
        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend in BETHE_BACKENDS:
            H = promoted_bethe_effective_hamiltonian(H, self.cfg)
            backend = "scipy"
        if backend in {"scipy", "numpy_scipy_cpu"}:
            if self.absorption_enabled():
                vals, vecs = eig(H)
                try:
                    inv_vecs = np.linalg.inv(vecs)
                except np.linalg.LinAlgError:
                    inv_vecs = np.linalg.pinv(vecs)
                coupling = vecs[1, :] * inv_vecs[:, 0]
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
                vals, vecs = eig(H)
                try:
                    inv_vecs = np.linalg.inv(vecs)
                except np.linalg.LinAlgError:
                    inv_vecs = np.linalg.pinv(vecs)
                coupling = vecs[1, :] * inv_vecs[:, 0]
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

    def row_prediction_plan(self, row_idx):
        """Build one row's Bloch matrices and metadata before eigensolver evaluation."""

        hkl = self.data.hkl[row_idx]
        zcal = self.data.zcal[row_idx]
        omega_center = self.omega_from_z(zcal)
        center_sg = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega_center)[0])
        omega_bragg = self.solve_omega_for_sg(hkl, omega_center, 0.0)
        basis = self.select_basis(hkl, omega_bragg, row_idx=row_idx)
        if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(basis[1], np.asarray(hkl, dtype=int)):
            raise RuntimeError("Frozen kernel basis contract violated: basis must start with direct beam then exact signed target hkl.")
        sg_window_i, sg_width_meta = self.row_sg_window(row_idx, hkl, omega_bragg)
        requested, profile_weights, jitter_meta = self.row_profile_sg_points(row_idx, hkl, sg_window_i)
        H_dyn_static = self.hamiltonian_static_part(basis, absorb=True, row_idx=row_idx)
        if self.absorption_enabled():
            H_born_static = self.hamiltonian_static_part(basis, absorb=False, row_idx=row_idx)
        else:
            H_born_static = H_dyn_static
        h10_born = H_born_static[1, 0]
        dyn_matrices, born_vals, sg_resid = [], [], []
        for target_sg in requested:
            omega = self.solve_omega_for_sg(hkl, omega_bragg, target_sg)
            sg_basis = self.sg_for_hkls(basis, omega)
            sg_resid.append(float(sg_basis[1] - target_sg))
            H = self.hamiltonian_from_static(H_dyn_static, sg_basis)
            dyn_matrices.append(H)
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            sinc = np.sinc(delta * self.thick)
            born_t = (2.0 * math.pi) ** 2 * (abs(h10_born) ** 2) * (self.thick**2) * (sinc**2)
            born_vals.append(float(np.sum(self.thick_w * born_t)))
        return {
            "_dyn_matrices": dyn_matrices,
            "_profile_weights": np.asarray(profile_weights, dtype=float),
            "_born_vals": np.asarray(born_vals, dtype=float),
            "_sg_resid": np.asarray(sg_resid, dtype=float),
            "basis_size": int(len(basis)),
            "support_hkls": [[int(v) for v in row] for row in basis.tolist()],
            "center_sg_inv_A": center_sg,
            "omega_center_deg": float(omega_center),
            "omega_bragg_deg": float(omega_bragg),
            "omega_bragg_minus_center_deg": float(omega_bragg - omega_center),
            "sg_window_inv_A": float(sg_window_i),
            "dose_delta_b_A2": float(self.row_delta_b(row_idx)),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
            **sg_width_meta,
            **jitter_meta,
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
        dyn_vals = [self.propagate_direct_to_target_thickness_average(H) for H in plan["_dyn_matrices"]]
        return self.finish_row_prediction_plan(plan, dyn_vals)

    def row_plan_cache_mode(self) -> str:
        """Return the requested frozen row-plan cache mode."""

        mode = str(getattr(self.cfg, "row_plan_cache", "off") or "off").lower()
        if mode in {"0", "false", "no", "none"}:
            return "off"
        if mode in {"1", "true", "yes", "on"}:
            return "readwrite"
        if mode not in {"off", "readwrite", "rebuild"}:
            raise ValueError(f"unknown row_plan_cache={mode!r}; expected off, readwrite, or rebuild")
        return mode

    def row_plan_cache_key(self, indices):
        """Return a geometry/config cache key for frozen row plans.

        The key intentionally omits the current structural model.  The stored
        basis is selected by the model that first builds the cache, then reused
        by later macrocycles while the Fourier coefficients are recomputed from
        the current model.  The builder model hash is saved in metadata so this
        frozen-basis choice is auditable.
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
            "profile_quadrature": str(self.cfg.profile_quadrature),
            "profile_accumulation": str(self.cfg.profile_accumulation),
            "topup_d_min": float(self.cfg.topup_d_min),
            "max_support_sg": float(self.cfg.max_support_sg),
            "frame_offset": float(self.cfg.frame_offset),
            "matrix_convention": str(self.cfg.matrix_convention),
            "dose_b_mode": str(self.cfg.dose_b_mode),
            "dose_b_per_frame": float(self.cfg.dose_b_per_frame),
            "dose_b_ref_frame": float(self.dose_ref_frame),
            "cell": [float(v) for v in self.model["cell"]],
            "wavelength": float(self.data.wavelength),
        }
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
                "basis_policy": "frozen_from_cache_builder_model",
            }
        )
        unique_diff_index: dict[tuple[int, int, int], int] = {}
        unique_diffs: list[tuple[int, int, int]] = []
        basis_offsets = [0]
        basis_flat = []
        profile_offsets = [0]
        profile_weights_flat = []
        sg_offsets = [0]
        sg_basis_flat = []
        diff_offsets = [0]
        diff_idx_flat = []
        center_sg = []
        omega_center = []
        omega_bragg = []
        omega_delta = []
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
        sg_residual_max = []
        dose_delta_b = []
        convention = str(getattr(self.cfg, "matrix_convention", "j_minus_i") or "j_minus_i")
        if convention not in {"j_minus_i", "i_minus_j"}:
            raise ValueError(f"unknown matrix_convention={convention!r}; expected j_minus_i or i_minus_j")

        t0 = time.perf_counter()
        for pos, row_idx in enumerate(indices):
            row_idx = int(row_idx)
            hkl = self.data.hkl[row_idx]
            zcal = self.data.zcal[row_idx]
            omega_c = self.omega_from_z(zcal)
            c_sg = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega_c)[0])
            omega_b = self.solve_omega_for_sg(hkl, omega_c, 0.0)
            basis = self.select_basis(hkl, omega_b, row_idx=row_idx)
            if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(basis[1], np.asarray(hkl, dtype=int)):
                raise RuntimeError("Frozen row-plan basis contract violated: basis must start with direct beam then exact signed target hkl.")
            sg_window_i, sg_width_meta = self.row_sg_window(row_idx, hkl, omega_b)
            requested, profile_weights, jitter_meta = self.row_profile_sg_points(row_idx, hkl, sg_window_i)
            sg_basis_rows = []
            sg_resid = []
            for target_sg in requested:
                omega = self.solve_omega_for_sg(hkl, omega_b, target_sg)
                sg_basis = self.sg_for_hkls(basis, omega)
                sg_basis_rows.append(np.asarray(sg_basis, dtype=float))
                sg_resid.append(float(sg_basis[1] - target_sg))

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
            profile_weights_flat.extend(float(v) for v in profile_weights)
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
            sg_residual_max.append(float(np.max(np.abs(sg_resid))) if len(sg_resid) else 0.0)
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
        loaded = np.load(path, allow_pickle=False)
        cache = {k: loaded[k] for k in loaded.files if k != "meta_json"}
        cache["meta"] = json.loads(str(loaded["meta_json"]))
        return cache

    def load_or_build_row_plan_cache(self, indices):
        mode = self.row_plan_cache_mode()
        if mode == "off":
            return None
        key, meta = self.row_plan_cache_key(indices)
        path = self.row_plan_cache_path(key)
        if mode == "readwrite" and path.exists():
            cache = self.load_row_plan_cache_file(path)
            errors = row_plan_cache_validation_errors(cache, indices, key, meta)
            if not errors:
                cache["status"] = {"status": "hit", "path": str(path), "key": key}
                return cache
            print(
                "Ignoring stale or incomplete row-plan cache "
                f"{path}: {'; '.join(errors[:4])}{'; ...' if len(errors) > 4 else ''}",
                flush=True,
            )
        cache = self.build_row_plan_cache(indices)
        self.save_row_plan_cache(path, cache)
        cache["status"] = {"status": "built" if mode == "readwrite" else "rebuilt", "path": str(path), "key": key}
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
            factor = float(self.cfg.gamma) / (math.pi * self.data.cell.volume)
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
        born_vals = []
        for sg_basis in sg_basis_rows:
            H = self.hamiltonian_from_static(H_dyn_static, sg_basis)
            dyn_vals.append(self.propagate_direct_to_target_thickness_average(H))
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            sinc = np.sinc(delta * self.thick)
            born_t = (2.0 * math.pi) ** 2 * (abs(h10_born) ** 2) * (self.thick**2) * (sinc**2)
            born_vals.append(float(np.sum(self.thick_w * born_t)))
        dyn = float(np.sum(weights * np.asarray(dyn_vals, dtype=float)))
        born = float(np.sum(weights * np.asarray(born_vals, dtype=float)))
        support = [[int(v) for v in row] for row in basis.tolist()]
        return row_idx, {
            "dyn": dyn,
            "born": born,
            "basis_size": int(n),
            "support_hkls": support,
            "max_sg_solve_residual": float(cache["sg_residual_max"][pos]) if "sg_residual_max" in cache else 0.0,
            "center_sg_inv_A": float(cache["center_sg"][pos]),
            "omega_center_deg": float(cache["omega_center"][pos]),
            "omega_bragg_deg": float(cache["omega_bragg"][pos]),
            "omega_bragg_minus_center_deg": float(cache["omega_delta"][pos]),
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
            "dose_delta_b_A2": float(cache["dose_delta_b"][pos]),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
        }

    def row_prediction_from_active_cache(self, pos: int):
        return self.row_prediction_from_cache(self._active_row_plan_cache, int(pos), self._active_unique_ug)

    def predict_rows_jax_batched(self, indices, chunksize=8):
        """Predict rows by batching Bloch eigensolvers through JAX."""

        indices = np.asarray(indices, dtype=int)
        n_basis = int(getattr(self.cfg, "n_basis", 20))
        row_block_size = 256 if n_basis >= 128 else (384 if n_basis >= 80 else 512)
        if len(indices) > row_block_size:
            chunks = []
            t0_all = time.perf_counter()
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
            for val, (row_pos, profile_pos, _H) in zip(vals, jobs):
                dyn_profile_vals[row_pos][profile_pos] = float(val)
        if total_matrices:
            print(
                f"JAX Bloch matrices={total_matrices} groups={len(jobs_by_size)} "
                f"matrix_bucket={matrix_bucket} eig_wall={total_eig_seconds:.2f}s "
                f"matrix_rate={total_matrices / max(total_eig_seconds, 1e-12):.2f}/s",
                flush=True,
            )
        results = [self.finish_row_prediction_plan(plan, dyn_profile_vals[pos]) for pos, plan in enumerate(plans)]
        return self.pack_prediction_results(indices, results)

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
        position = {int(idx): int(pos) for pos, idx in enumerate(indices)}
        completed = 0
        parallel_errors: list[str] = []

        def accept(idx, r):
            nonlocal completed
            pos = position.get(int(idx))
            if pos is None:
                raise RuntimeError(f"cached prediction returned unexpected row index {idx}")
            results[pos] = r
            completed += 1
            if completed % 500 == 0 or completed == len(indices):
                elapsed = time.perf_counter() - t0
                rate = completed / max(elapsed, 1e-12)
                eta = (len(indices) - completed) / max(rate, 1e-12)
                print(f"cached predicted {completed}/{len(indices)} rows | rate={rate:.2f}/s | ETA={eta/60:.1f}m", flush=True)

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
                            for idx, r in pool.imap_unordered(
                                _predict_row_cache_worker,
                                list(range(len(indices))),
                                chunksize=max(1, int(chunksize)),
                            ):
                                accept(idx, r)
                        ran_parallel = True
                        break
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
                    for pos in range(len(indices)):
                        idx, r = self.row_prediction_from_cache(cache, pos, unique_ug)
                        accept(idx, r)
            else:
                for pos in range(len(indices)):
                    idx, r = self.row_prediction_from_cache(cache, pos, unique_ug)
                    accept(idx, r)
        finally:
            self._active_row_plan_cache = None
            self._active_unique_ug = None

        out = self.pack_prediction_results(indices, results)
        out["seconds"] = time.perf_counter() - t0
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
        born = np.zeros(len(indices), dtype=float)
        basis_size = np.zeros(len(indices), dtype=int)
        sg_resid = np.zeros(len(indices), dtype=float)
        center_sg = np.zeros(len(indices), dtype=float)
        omega_bragg = np.zeros(len(indices), dtype=float)
        omega_delta = np.zeros(len(indices), dtype=float)
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
        support_hkls = []
        support_hkls = [None] * len(indices)
        position = {int(idx): int(pos) for pos, idx in enumerate(indices)}
        completed = 0

        def accept_result(idx, r):
            nonlocal completed
            n = position[int(idx)]
            dyn[n] = r["dyn"]
            born[n] = r["born"]
            basis_size[n] = r["basis_size"]
            sg_resid[n] = r["max_sg_solve_residual"]
            center_sg[n] = r["center_sg_inv_A"]
            omega_bragg[n] = r["omega_bragg_deg"]
            omega_delta[n] = r["omega_bragg_minus_center_deg"]
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
            support_hkls[n] = r["support_hkls"]
            completed += 1
            if completed % 500 == 0 or completed == len(indices):
                elapsed = time.perf_counter() - t0
                rate = completed / max(elapsed, 1e-12)
                eta = (len(indices) - completed) / max(rate, 1e-12)
                print(f"predicted {completed}/{len(indices)} rows | rate={rate:.2f}/s | ETA={eta/60:.1f}m", flush=True)

        t0 = time.perf_counter()
        for idx, r in zip(indices, results):
            accept_result(int(idx), r)

        return {
            "indices": indices,
            "dyn": dyn,
            "born": born,
            "basis_size": basis_size,
            "support_hkls": support_hkls,
            "sg_residual": sg_resid,
            "center_sg": center_sg,
            "omega_bragg": omega_bragg,
            "omega_delta": omega_delta,
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
            "seconds": time.perf_counter() - t0,
        }

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
        dyn = np.zeros(len(indices), dtype=float)
        born = np.zeros(len(indices), dtype=float)
        basis_size = np.zeros(len(indices), dtype=int)
        sg_resid = np.zeros(len(indices), dtype=float)
        center_sg = np.zeros(len(indices), dtype=float)
        omega_bragg = np.zeros(len(indices), dtype=float)
        omega_delta = np.zeros(len(indices), dtype=float)
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
        support_hkls = []
        t0 = time.perf_counter()
        support_hkls = [None] * len(indices)
        position = {int(idx): int(pos) for pos, idx in enumerate(indices)}
        completed = 0

        def accept_result(idx, r):
            nonlocal completed
            n = position[int(idx)]
            dyn[n] = r["dyn"]
            born[n] = r["born"]
            basis_size[n] = r["basis_size"]
            sg_resid[n] = r["max_sg_solve_residual"]
            center_sg[n] = r["center_sg_inv_A"]
            omega_bragg[n] = r["omega_bragg_deg"]
            omega_delta[n] = r["omega_bragg_minus_center_deg"]
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
            support_hkls[n] = r["support_hkls"]
            completed += 1
            if completed % 500 == 0 or completed == len(indices):
                elapsed = time.perf_counter() - t0
                rate = completed / max(elapsed, 1e-12)
                eta = (len(indices) - completed) / max(rate, 1e-12)
                print(f"predicted {completed}/{len(indices)} rows | rate={rate:.2f}/s | ETA={eta/60:.1f}m", flush=True)

        def pool_context_names():
            import platform

            if platform.system() == "Darwin":
                return ["forkserver", "spawn"]
            # Linux fork shares the initialized port/cache state copy-on-write;
            # forkserver is a safe fallback but is much slower for high-worker
            # repeated Bloch predictions on large servers.
            return ["fork", "forkserver", "spawn"]

        def run_serial():
            for idx in indices:
                accept_result(int(idx), self.row_prediction(int(idx)))

        if int(workers) > 1 and len(indices) > 1:
            ran_parallel = False
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
                        for idx, r in pool.imap_unordered(
                            _predict_row_worker,
                            [int(v) for v in indices],
                            chunksize=max(1, int(chunksize)),
                        ):
                            accept_result(idx, r)
                    ran_parallel = True
                    break
                except Exception:
                    ran_parallel = False
                    continue
            if not ran_parallel:
                run_serial()
        else:
            run_serial()
        return {
            "indices": np.asarray(indices, dtype=int),
            "dyn": dyn,
            "born": born,
            "basis_size": basis_size,
            "sg_residual": sg_resid,
            "center_sg": center_sg,
            "omega_bragg": omega_bragg,
            "omega_delta": omega_delta,
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
            "support_hkls": support_hkls,
            "seconds": time.perf_counter() - t0,
        }


def effective_sigma(data: XDSData, cfg: Config):
    corr_term = np.maximum(0.0, 1.0 - data.corr / 100.0)
    peak_term = np.maximum(0.0, 1.0 - data.peak / 100.0)
    return np.sqrt(data.sigma**2 + (cfg.variance_a * data.iobs) ** 2 + (cfg.variance_b_corr * data.iobs * corr_term) ** 2 + (cfg.variance_c_peak * data.iobs * peak_term) ** 2)


def split_masks(hkls, free_fraction=0.10, seed=42):
    unique = sorted(set(tuple(int(v) for v in h) for h in hkls))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    n_free = max(1, int(round(len(unique) * float(free_fraction))))
    free = set(unique[:n_free])
    masks = {"all": np.ones(len(hkls), dtype=bool), "work": np.zeros(len(hkls), dtype=bool), "free": np.zeros(len(hkls), dtype=bool)}
    for i, h in enumerate(hkls):
        masks["free"][i] = tuple(int(v) for v in h) in free
    masks["work"] = ~masks["free"]
    return masks


def scale_by_groups(obs, sig, pred, groups, fit_mask):
    scaled = np.full_like(pred, np.nan, dtype=float)
    scales = {}
    weights = 1.0 / np.maximum(sig * sig, 1e-12)
    for g in np.unique(groups):
        fit = (groups == g) & fit_mask & np.isfinite(pred) & (pred > 0)
        apply = groups == g
        if not np.any(fit):
            scales[int(g)] = 0.0
            continue
        x, y, w = pred[fit], obs[fit], weights[fit]
        scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
        scale = max(scale, 0.0)
        finite_apply = apply & np.isfinite(pred)
        scaled[finite_apply] = pred[finite_apply] * scale
        scales[int(g)] = scale
    return scaled, scales


def metrics(obs, sig, pred, mask):
    mask = np.asarray(mask, dtype=bool) & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred)
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


def evaluate_predictions(data, cfg, pred, fit_scales_on="work"):
    sig = effective_sigma(data, cfg)
    masks = split_masks(data.hkl, cfg.free_fraction, cfg.split_seed)
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
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
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
    ap.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off", help="Freeze row geometry, basis, Sg profiles, and h_j-h_i indices for reuse across model updates.")
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
        row_indices = []
        with open(args.row_index_file, newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            if "," in sample:
                reader = csv.DictReader(f)
                field = "row_index" if reader.fieldnames and "row_index" in reader.fieldnames else (reader.fieldnames[0] if reader.fieldnames else None)
                if field is None:
                    raise ValueError(f"No columns found in {args.row_index_file}")
                for row in reader:
                    row_indices.append(int(row[field]))
            else:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    row_indices.append(int(line.split()[0]))
        indices = np.asarray(sorted(set(row_indices)), dtype=int)
        indices = indices[(indices >= 0) & (indices < len(data_all.iobs))]
    else:
        mask = np.ones(len(data_all.iobs), dtype=bool)
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
    full_dyn[pred_subset["indices"]] = pred_subset["dyn"]
    full_born[pred_subset["indices"]] = pred_subset["born"]
    full_basis_size[pred_subset["indices"]] = pred_subset["basis_size"]
    full_sg_resid[pred_subset["indices"]] = pred_subset["sg_residual"]
    full_center_sg[pred_subset["indices"]] = pred_subset["center_sg"]
    full_omega_bragg[pred_subset["indices"]] = pred_subset["omega_bragg"]
    full_omega_delta[pred_subset["indices"]] = pred_subset["omega_delta"]
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
    support_by_idx = {int(idx): basis for idx, basis in zip(pred_subset["indices"], pred_subset["support_hkls"])}
    pred = {"dyn": full_dyn, "born": full_born, "groups": port.group_indices}
    valid = np.isfinite(full_dyn) & np.isfinite(full_born)
    eval_data = data_all
    eval_out = evaluate_predictions(eval_data, cfg, pred, fit_scales_on="work")
    row_table_path = ""
    if args.save_row_table:
        row_table_path = str(Path(args.save_row_table).resolve())
        sig_eff = effective_sigma(data_all, cfg)
        dyn_scaled, dyn_scales = scale_by_groups(data_all.iobs, sig_eff, full_dyn, port.group_indices, split_masks(data_all.hkl, cfg.free_fraction, cfg.split_seed)["work"])
        born_scaled, born_scales = scale_by_groups(data_all.iobs, sig_eff, full_born, port.group_indices, split_masks(data_all.hkl, cfg.free_fraction, cfg.split_seed)["work"])
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
            "simulation_cell": [
                data_all.cell.a,
                data_all.cell.b,
                data_all.cell.c,
                data_all.cell.alpha,
                data_all.cell.beta,
                data_all.cell.gamma,
            ],
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
            "profile_quadrature": cfg.profile_quadrature,
            "profile_accumulation": cfg.profile_accumulation,
            "sg_jitter_sigma_median_used": float(np.nanmedian(pred_subset["sg_jitter_sigma"])) if len(indices) else float("nan"),
            "sg_jitter_sigma_max_used": float(np.nanmax(pred_subset["sg_jitter_sigma"])) if len(indices) else float("nan"),
            "sg_profile_points_median_total": float(np.median(pred_subset["sg_profile_points_total"])) if len(indices) else 0.0,
            "sg_profile_weight_sum_median": float(np.nanmedian(pred_subset["sg_profile_weight_sum"])) if len(indices) else 0.0,
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
