"""Optional acceleration and strong/weak-beam reduction.

Dense SciPy remains the reference. These alternatives are retained for explicit
agreement tests, not selected because a device or library happens to exist.
In our tests expm_multiply was slower than dense diagonalisation, and small
adaptive matrices gave no end-to-end Bethe speed benefit, so neither is the
default. The real-embedded Metal path uses float32 and doubles the matrix
dimension; faster device arithmetic is not proof of accurate row intensities."""

from __future__ import annotations

import math
import os

import numpy as np

from grond.bloch.config import Config
from grond.bloch.constants import JAX_PADDED_BACKENDS, JAX_REAL_EXPM_BACKENDS, JAX_REAL_EXPM_PADDED_BACKENDS

_JAX_MODULES = {}
_JAX_PROPAGATORS = {}
_JAX_REAL_EXPM_PROPAGATORS = {}


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
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except Exception as exc:
        raise NotImplementedError(
            f"compute_backend={backend!r} requested JAX, but JAX could not be imported: {type(exc).__name__}: {exc}"
        ) from exc
    _JAX_MODULES[key] = (jax, jnp)
    return _JAX_MODULES[key]


def _jax_devices(backend: str):
    """Resolve the requested platform explicitly; an earlier CPU import must not masquerade as an accelerator."""

    key = _jax_backend_base(backend)
    platforms = {"jax": None, "jax_cpu": "cpu", "jax_cuda": "cuda", "jax_mps": "METAL"}
    if key not in platforms:
        raise NotImplementedError(f"unknown JAX backend {backend!r}")
    jax, _jnp = _jax_modules(key)
    try:
        devices = tuple(jax.devices(platforms[key]))
    except Exception as error:
        raise NotImplementedError(f"compute_backend={backend!r}: requested platform is unavailable: {error}") from error
    if not devices:
        raise NotImplementedError(f"compute_backend={backend!r} found no requested devices")
    return devices


def _jax_module_key(backend: str) -> str:
    """Return the JAX module-cache key of a backend name, which is its base JAX backend."""

    return _jax_backend_base(backend)


def _jax_host_buffer_bytes() -> int:
    """Limit each retained-plan and dense-input buffer; numerical-library workspaces are additional."""

    gib = float(os.environ.get("GROND_JAX_HOST_BUFFER_GIB", "0.25"))
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError("GROND_JAX_HOST_BUFFER_GIB must be positive and finite")
    return max(1, int(gib * 2**30))


def _stack_square_matrices(matrices, pad_to: int | None = None, *, count: int | None = None) -> np.ndarray:
    """Stack square matrices with zero-beam padding; count and pad_to permit bounded generator consumption."""

    if count is None:
        matrices = list(matrices)
        if not matrices:
            return np.zeros((0, 0, 0), dtype=np.complex128)
        count = len(matrices)
        pad_to = int(pad_to or max(np.asarray(H).shape[0] for H in matrices))
    elif pad_to is None:
        raise ValueError("a streaming matrix count requires pad_to")
    n = int(pad_to)
    out = np.zeros((int(count), n, n), dtype=np.complex128)
    written = 0
    for i, H in enumerate(matrices):
        if i >= count:
            raise ValueError("more matrices than the declared count")
        H = np.asarray(H, dtype=np.complex128)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"expected square matrix, got shape {H.shape}")
        if H.shape[0] > n:
            raise ValueError(f"matrix dimension {H.shape[0]} exceeds requested pad_to={n}")
        out[i, : H.shape[0], : H.shape[1]] = H
        written += 1
    if written != count:
        raise ValueError("fewer matrices than the declared count")
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
    devices = _jax_devices(backend)
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
        inv_scale = dtype(1.0 / float(2 ** int(scale_steps)))

        def _eval(Rs):
            """Return |psi_target|^2 per matrix from the real 2n x 2n embedding of i*2*pi*H*t.

            exp is a Taylor series of degree ``order`` after dividing by 2**scale_steps, then squared scale_steps times.
            Rows 1 and n + 1 of column 0 are the real and imaginary target amplitudes for a unit direct beam.
            """

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
        # Committed input placement also selects the JIT device; avoid first creating a float64 array on Metal
        # merely because Metal happens to be the process default while this request explicitly selects CPU.
        vals = np.asarray(fn(jax.device_put(sub, devices[0])))
        if not np.all(np.isfinite(vals)):
            raise NotImplementedError("JAX real-expm returned non-finite intensities; use the SciPy reference")
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


def _elastic_matrix_dtype(matrices):
    """Return float64 only when every elastic matrix passes the CPU real-H criterion; otherwise complex128.

    Test each matrix separately: a large real matrix must not hide another matrix's meaningful imaginary part.
    As before, accelerator solves use double precision; this decision uses the precision of the supplied matrices.
    This helper is for elastic matrices only: a caller must split uniform loss or use a general complex solver.
    """

    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or matrices.shape[1] < 2:
        raise ValueError(f"expected a stack of square matrices with direct and target beams, got {matrices.shape}")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("Bloch matrices must be finite")
    if np.iscomplexobj(matrices) and len(matrices):
        real = matrices.real
        bound = 8.0 * np.finfo(real.dtype).eps * np.max(np.abs(real), axis=(-2, -1))
        if not np.all(np.max(np.abs(matrices.imag), axis=(-2, -1)) <= bound):
            return np.complex128
    return np.float64


def _jax_eigensolver_input(matrices, *, absorption: bool) -> np.ndarray:
    """Prepare a batch, retaining all absorptive entries even when their imaginary part is numerically small."""

    matrices = np.asarray(matrices)
    elastic_dtype = _elastic_matrix_dtype(matrices)  # Also checks shape and finite values.
    dtype = np.complex128 if absorption else elastic_dtype
    return np.ascontiguousarray(matrices.real if dtype == np.float64 else matrices, dtype=dtype)


def _jax_general_eig_implementation(jax, devices):
    """Select general complex cuSOLVER explicitly on GPU; never accept host LAPACK as GPU execution."""

    platforms = {device.platform for device in devices}
    if platforms == {"cpu"}:
        return None
    if platforms <= {"gpu", "cuda"}:
        implementations = getattr(jax.lax.linalg, "EigImplementation", None)
        cusolver = getattr(implementations, "CUSOLVER", None)
        if cusolver is not None:
            return cusolver
    raise NotImplementedError(
        "general absorptive GPU propagation requires JAX with explicit cuSOLVER eig support; "
        "use the SciPy reference or a supported CUDA environment"
    )


def _inverse_condition_upper_bound(vectors, inverse, *, xp=np):
    """Bound cond_2(V) using the computed inverse X, without another singular-value decomposition.

    For R=I-VX with ||R||_F<1, V^-1=X(I-R)^-1 and
    cond_2(V) <= ||V||_F ||X||_F / (1-||R||_F).
    Allow for complex dot-product roundoff before using this bound. An uncertain inverse is rejected (infinity),
    not assigned a favorable condition estimate. The same 1e10 acceptance threshold remains in the caller.
    """

    n = vectors.shape[-1]
    residual = xp.eye(n, dtype=vectors.dtype) - vectors @ inverse
    error = xp.linalg.norm(residual, ord="fro", axis=(-2, -1))
    product = (xp.linalg.norm(vectors, ord="fro", axis=(-2, -1))
               * xp.linalg.norm(inverse, ord="fro", axis=(-2, -1)))
    rounding = 8.0 * n * xp.finfo(vectors.real.dtype).eps
    gamma = rounding / (1.0 - rounding) if rounding < 1.0 else math.inf
    product_upper = product * (1.0 + gamma)
    error_upper = error + gamma * (product_upper + math.sqrt(n))
    denominator = xp.where(error_upper < 1.0, 1.0 - error_upper, 1.0)
    return xp.where(error_upper < 1.0, product_upper / denominator, xp.inf)


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
    devices = _jax_devices(backend)
    H_batch = _jax_eigensolver_input(H_batch, absorption=absorption)
    n_real = int(H_batch.shape[0])
    if n_real == 0:
        return np.zeros(0, dtype=float)
    device_limit = int(batch_size) if int(batch_size or 0) > 0 else len(devices)
    devices = devices[:min(len(devices), n_real, device_limit)]
    n_devices = len(devices)
    n = int(H_batch.shape[1])
    chunk = int(batch_size) if int(batch_size or 0) > 0 else _auto_jax_batch_size(n, n_devices)
    chunk = max(n_devices, (chunk // n_devices) * n_devices)
    chunk = min(chunk, int(math.ceil(n_real / n_devices)) * n_devices)
    per_device = max(1, chunk // n_devices)
    eig_implementation = _jax_general_eig_implementation(jax, devices) if absorption else None
    cache_key = (key, bool(absorption), H_batch.dtype.str, str(eig_implementation), n, len(thick),
                 n_devices, per_device, tuple(str(d) for d in devices))

    if cache_key not in _JAX_PROPAGATORS:
        if absorption:

            def _eval_shard(Hs, tvals, wvals):
                """Return thickness-weighted target intensities for a shard of absorbing (non-Hermitian) Hamiltonians.

                Uses eig and an explicit inverse, with t in Å:
                psi_target(t) = sum_j V[1, j] V^-1[j, 0] exp(2*pi*i*lambda_j*t).
                """

                if eig_implementation is None:
                    vals, vecs = jnp.linalg.eig(Hs)
                else:
                    vals, vecs = jax.lax.linalg.eig(
                        Hs, compute_left_eigenvectors=False, compute_right_eigenvectors=True,
                        implementation=eig_implementation,
                    )
                inv_vecs = jnp.linalg.inv(vecs)
                condition = _inverse_condition_upper_bound(vecs, inv_vecs, xp=jnp)
                reconstruction = (vecs * vals[:, None, :]) @ inv_vecs
                relative_error = jnp.linalg.norm(reconstruction - Hs, axis=(-2, -1)) / jnp.maximum(
                    jnp.linalg.norm(Hs, axis=(-2, -1)), jnp.finfo(jnp.float64).eps
                )
                stable = jnp.isfinite(condition) & (condition <= 1e10) & jnp.isfinite(relative_error) & (relative_error <= 1e-8)
                coupling = vecs[:, 1, :] * inv_vecs[:, :, 0]
                phase = jnp.exp((2j * jnp.pi) * vals[:, :, None] * tvals[None, None, :])
                amp = jnp.sum(coupling[:, :, None] * phase, axis=1)
                per_t = jnp.real(jnp.abs(amp) ** 2)
                return jnp.where(stable, jnp.sum(per_t * wvals[None, :], axis=1), jnp.nan)
        else:

            def _eval_shard(Hs, tvals, wvals):
                """Return thickness-weighted target intensities for a shard of Hermitian Hamiltonians.

                Uses eigh: psi_target(t) = sum_j V[1, j] conj(V[0, j]) exp(2*pi*i*lambda_j*t), t in Å.
                """

                vals, vecs = jnp.linalg.eigh(Hs)
                coupling = vecs[:, 1, :] * jnp.conj(vecs[:, 0, :])
                phase = jnp.exp((2j * jnp.pi) * vals[:, :, None] * tvals[None, None, :])
                amp = jnp.sum(coupling[:, :, None] * phase, axis=1)
                per_t = jnp.real(jnp.abs(amp) ** 2)
                return jnp.sum(per_t * wvals[None, :], axis=1)

        _JAX_PROPAGATORS[cache_key] = jax.pmap(_eval_shard, in_axes=(0, None, None), devices=devices)

    fn = _JAX_PROPAGATORS[cache_key]
    tvals = np.asarray(thick, dtype=np.float64)
    wvals = np.asarray(thick_w, dtype=np.float64)
    out = np.zeros(n_real, dtype=float)
    for start in range(0, n_real, chunk):
        sub = H_batch[start : start + chunk]
        pad = chunk - len(sub)
        if pad:
            sub = np.concatenate([sub, np.repeat(sub[-1:, :, :], pad, axis=0)], axis=0)
        shard = sub.reshape(n_devices, per_device, n, n)
        # pmap owns placement across its explicit devices; host arrays avoid using the process default first.
        vals = np.asarray(fn(shard, tvals, wvals)).reshape(-1)
        if not np.all(np.isfinite(vals)):
            raise NotImplementedError("JAX eigensystem is ill-conditioned or non-finite; use the SciPy reference")
        out[start : start + min(chunk, n_real - start)] = vals[: min(chunk, n_real - start)]
    return out


def safe_complex_inverse(values: np.ndarray, eps: float) -> np.ndarray:
    """Return 1/z with magnitude bounded by 1/eps and the reciprocal's complex phase.

    At exactly zero, use the positive real bound. Non-finite values are errors, not small denominators.
    """

    values = np.asarray(values, dtype=np.complex128)
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("the inverse denominator bound must be positive finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("inverse denominators must be finite")
    out = np.empty_like(values)
    mag = np.abs(values)
    ok = mag >= float(eps)
    out[ok] = 1.0 / values[ok]
    if np.any(~ok):
        small = values[~ok]
        phase = np.ones(len(small), dtype=np.complex128)
        nonzero = np.abs(small) > 0.0
        phase[nonzero] = np.conj(small[nonzero]) / np.abs(small[nonzero])
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
        raise ValueError(f"unknown bethe_variant={variant!r}; expected truncate, schur0, schur_target, or schur_mean")

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
