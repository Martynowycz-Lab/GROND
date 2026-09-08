"""Execute complete prediction attempts and pack rows in their original order.

Worker retries are transactional: partial output from a failed context is not
combined with a retry. BLAS helper threads remain bounded by the entry facade."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import time

import numpy as np

from grond.bloch.backends import (
    _jax_matrix_bucket_size,
    _jax_module_key,
    _jax_modules,
    _jax_propagate_batch,
    _stack_square_matrices,
)
from grond.bloch.constants import ALL_JAX_BACKENDS
from grond.bloch.errors import (
    InfeasibleSgJitterProfileError,
    LocalSgCrossingError,
    MissingObservedCentroidError,
    NonPassiveAbsorptionError,
)

# Another process context cannot repair missing observations or invalid physics.
_NON_RETRYABLE_ERRORS = (
    LocalSgCrossingError,
    InfeasibleSgJitterProfileError,
    NonPassiveAbsorptionError,
    MissingObservedCentroidError,
)

_WORKER_PORT = None


def _init_prediction_worker(port):
    global _WORKER_PORT
    _WORKER_PORT = port


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


class PredictionMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

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
                print(
                    f"prepared {pos + 1}/{len(indices)} rows for JAX | rate={rate:.2f}/s | ETA={eta / 60:.1f}m",
                    flush=True,
                )
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
                    except _NON_RETRYABLE_ERRORS:
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
                        self.row_prediction_from_cache(cache, pos, unique_ug) for pos in range(len(indices))
                    ]
                    commit_attempt(attempt_results)
            else:
                attempt_results = [self.row_prediction_from_cache(cache, pos, unique_ug) for pos in range(len(indices))]
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
            omega_bragg_root_available[n] = bool(r.get("omega_bragg_root_available", np.isfinite(r["omega_bragg_deg"])))
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
            sg_profile_same_branch_continuation_used[n] = bool(r["sg_profile_same_branch_continuation_used"])
            sg_profile_same_branch_continuation_segments[n] = int(r["sg_profile_same_branch_continuation_segments"])
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
                print(
                    "JAX backend requested; ignoring multiprocessing workers and batching in one process.", flush=True
                )
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
                except _NON_RETRYABLE_ERRORS:
                    raise
                except Exception:
                    continue
        if results is None:
            serial_attempt = [(int(idx), self.row_prediction(int(idx))) for idx in indices]
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
