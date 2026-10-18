"""Execute complete prediction attempts and pack rows in their original order.

Worker retries are transactional: partial output from a failed context is not
combined with a retry. BLAS helper threads remain bounded by the entry facade."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from grond.bloch.backends import (
    _auto_jax_batch_size,
    _jax_backend_uses_real_expm,
    _jax_devices,
    _jax_host_buffer_bytes,
    _jax_matrix_bucket_size,
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
from grond.bloch.numerics import sha256, validate_row_indices

# Another process context cannot repair missing observations or invalid physics.
_NON_RETRYABLE_ERRORS = (
    LocalSgCrossingError,
    InfeasibleSgJitterProfileError,
    NonPassiveAbsorptionError,
    MissingObservedCentroidError,
)

_WORKER_PORT = None
# Opt-in GPU row workers next to the CPU pool, e.g. "0,1,2"; a repeated id starts another process on that device.
# GROND_HYBRID_GPU_BATCH sets the matrices per device call (default: sized from the matrix dimension).
HYBRID_GPU_ENV = "GROND_HYBRID_GPU_DEVICES"


def hybrid_gpu_devices() -> list[str]:
    """Device ids of the opt-in GPU row workers named by GROND_HYBRID_GPU_DEVICES (empty when unset)."""

    value = os.environ.get(HYBRID_GPU_ENV, "").strip()
    return [item.strip() for item in value.split(",") if item.strip()]


def _pool_context_names():
    """Return process start methods to try in order: forkserver and spawn on macOS, fork first elsewhere."""

    import platform

    if platform.system() == "Darwin":
        return ["forkserver", "spawn"]
    # Linux fork shares the initialized port/cache state copy-on-write;
    # forkserver is a safe fallback but is much slower for high-worker
    # repeated Bloch predictions on large servers.
    return ["fork", "forkserver", "spawn"]


def _init_prediction_worker(port):
    """Pool initializer: keep the port as this worker process's prediction port."""

    global _WORKER_PORT
    _WORKER_PORT = port


def _predict_row_worker(row_index):
    """Return (row index, row_prediction) from the worker's port; RuntimeError before initialization."""

    if _WORKER_PORT is None:
        raise RuntimeError("prediction worker was not initialized")
    return int(row_index), _WORKER_PORT.row_prediction(int(row_index))


def _gpu_pool_context():
    """Start method of GPU row workers: spawn, so that JAX is first imported after the device is chosen."""

    return multiprocessing.get_context("spawn")


def _init_gpu_prediction_worker(port, device):
    """GPU pool initializer: pin this process to one CUDA device before JAX loads, then keep the port."""

    import sys

    if "jax" in sys.modules:
        raise RuntimeError("a GPU prediction worker must select its device before JAX is imported")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("GROND_JAX_HOST_BUFFER_GIB", "4")  # one full-cone row's compact buffers
    _init_prediction_worker(port)


def _predict_row_gpu_worker(row_index):
    """Return (row index, row prediction) with the row's eigensystems solved on this worker's GPU."""

    from grond.legacy.engine.spectra_stage import device_row_prediction

    if _WORKER_PORT is None:
        raise RuntimeError("prediction worker was not initialized")
    batch = int(os.environ.get("GROND_HYBRID_GPU_BATCH", "0") or 0)  # matrices per device call; 0 sizes it from N
    return int(row_index), device_row_prediction(_WORKER_PORT, int(row_index), batch_size=batch)


def _predict_row_cache_worker(row_position):
    """Return the prediction of position ``row_position`` in the worker port's active row-plan cache."""

    if _WORKER_PORT is None:
        raise RuntimeError("prediction worker was not initialized")
    return _WORKER_PORT.row_prediction_from_active_cache(int(row_position))


def _ordered_prediction_attempt(indices, attempt_results):
    """Validate one complete backend attempt before exposing any row results.

    Multiprocessing iterators can yield valid rows before a worker or context
    fails.  Keeping those rows would make a later context retry or serial
    fallback count and apply some rows twice.  This helper is the transaction
    boundary: an attempt is committed only when it returned every requested row
    exactly once.
    """

    requested = np.asarray(indices, dtype=int)
    position_by_row_index = {int(row_index): int(row_position) for row_position, row_index in enumerate(requested)}
    if len(position_by_row_index) != len(requested):
        raise ValueError("prediction row indices must be unique")
    ordered = [None] * len(requested)
    for row_index, result in attempt_results:
        row_index = int(row_index)
        row_position = position_by_row_index.get(row_index)
        if row_position is None:
            raise RuntimeError(f"prediction returned unexpected row index {row_index}")
        if ordered[row_position] is not None:
            raise RuntimeError(f"prediction returned duplicate row index {row_index}")
        ordered[row_position] = result
    missing = [int(requested[row_position]) for row_position, result in enumerate(ordered) if result is None]
    if missing:
        preview = ", ".join(str(row_index) for row_index in missing[:8])
        raise RuntimeError(
            f"prediction attempt returned {len(requested) - len(missing)}/{len(requested)} rows; "
            f"missing row indices: {preview}"
        )
    return ordered


class PredictionMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def _verify_prediction_sources(self):
        """Refuse source replacement before calculation or publication of its result."""

        verify = getattr(self.data, "verify_source_files", None)
        if verify is not None:
            verify()
        geometry = getattr(self, "geometry_path", None)
        if geometry is not None and sha256(Path(geometry["path"])) != geometry["sha256"]:
            raise ValueError("loaded geometry path changed after reading; reload the Bloch port")

    def _predict_jax_plan_block(self, plans, backend, buffer_bytes):
        """Evaluate a compact row-plan block through bounded dense matrix batches."""

        matrix_bucket = _jax_matrix_bucket_size(self.cfg, backend)
        jobs_by_size = {}
        for row_position, plan in enumerate(plans):
            matrices = plan["_dyn_matrices"]
            common_size = matrices.shape[1] if hasattr(matrices, "storage_nbytes") else None
            for profile_position in range(len(matrices)):
                size = int(common_size if common_size is not None else matrices[profile_position].shape[0])
                if matrix_bucket > 0:
                    size = int(math.ceil(size / matrix_bucket) * matrix_bucket)
                # Store locations, not dense matrices. ProfileHamiltonians materializes only the requested batch.
                jobs_by_size.setdefault(size, []).append((row_position, profile_position))
        values_by_row = [np.zeros(len(plan["_dyn_matrices"])) for plan in plans]
        devices = _jax_devices(backend)
        if _jax_backend_uses_real_expm(backend):
            devices = devices[:1]
        configured_batch = int(getattr(self.cfg, "jax_batch_size", 0))
        absorption = self.absorption_enabled()
        diagnostics = {"matrices": 0, "propagation_seconds": 0.0, "max_input_batch_bytes": 0}
        verbose = str(os.environ.get("GROND_JAX_VERBOSE", "")).lower() in {"1", "true", "yes", "on"}
        for size, jobs in sorted(jobs_by_size.items()):
            capacity = buffer_bytes // (16 * size * size)
            if capacity < 1:
                raise MemoryError(f"one {size} by {size} JAX matrix exceeds GROND_JAX_HOST_BUFFER_GIB")
            preferred = configured_batch if configured_batch > 0 else _auto_jax_batch_size(size, len(devices))
            count = max(1, min(preferred, capacity))
            if count >= len(devices):
                count = max(len(devices), (count // len(devices)) * len(devices))
            for start in range(0, len(jobs), count):
                batch_jobs = jobs[start:start + count]
                matrices = (plans[row]["_dyn_matrices"][profile] for row, profile in batch_jobs)
                batch = _stack_square_matrices(matrices, pad_to=size, count=len(batch_jobs))
                started = time.perf_counter()
                values = _jax_propagate_batch(batch, self.thick, self.thick_w, backend, absorption, count)
                seconds = time.perf_counter() - started
                diagnostics["matrices"] += len(batch_jobs)
                diagnostics["propagation_seconds"] += seconds
                diagnostics["max_input_batch_bytes"] = max(diagnostics["max_input_batch_bytes"], batch.nbytes)
                for value, (row, profile) in zip(values, batch_jobs, strict=True):
                    values_by_row[row][profile] = float(value)
                if verbose:
                    print(f"JAX Bloch size={size} matrices={len(batch_jobs)} devices={min(len(devices), len(batch_jobs))} "
                          f"host_batch_MiB={batch.nbytes / 2**20:.2f} wall={seconds:.2f}s", flush=True)
                del batch, values
        return [self.finish_row_prediction_plan(plan, values) for plan, values in zip(plans, values_by_row, strict=True)], diagnostics

    def predict_rows_jax_batched(self, indices, chunksize=8):
        """Predict every requested row with bounded plan and dense-input buffers."""

        started = time.perf_counter()
        indices = validate_row_indices(indices, row_count=len(self.data.iobs))
        self._verify_prediction_sources()
        n_basis = int(getattr(self.cfg, "n_basis", 20))
        row_block_size = 256 if n_basis >= 128 else (384 if n_basis >= 80 else 512)
        buffer_limit = _jax_host_buffer_bytes()
        backend = str(getattr(self.cfg, "compute_backend", "jax") or "jax")
        _jax_devices(backend)  # Refuse unavailable platforms before building expensive row plans.
        plans, results = [], []
        retained_bytes = 0
        diagnostics = {"host_buffer_limit_bytes": buffer_limit, "peak_plan_array_bytes": 0,
                       "max_input_batch_bytes": 0, "matrices": 0, "propagation_seconds": 0.0}

        def flush():
            nonlocal retained_bytes
            completed, block = self._predict_jax_plan_block(plans, backend, buffer_limit)
            results.extend(completed)
            diagnostics["matrices"] += block["matrices"]
            diagnostics["propagation_seconds"] += block["propagation_seconds"]
            diagnostics["max_input_batch_bytes"] = max(diagnostics["max_input_batch_bytes"], block["max_input_batch_bytes"])
            plans.clear()
            retained_bytes = 0
            print(f"predicted {len(results)}/{len(indices)} JAX rows", flush=True)

        for row_index in indices:
            plan = self.row_prediction_plan(int(row_index))
            matrices = plan["_dyn_matrices"]
            storage = (matrices.storage_nbytes if hasattr(matrices, "storage_nbytes")
                       else sum(np.asarray(matrix).nbytes for matrix in matrices))
            storage += sum(value.nbytes for key, value in plan.items()
                           if key != "_dyn_matrices" and isinstance(value, np.ndarray))
            if storage > buffer_limit:
                raise MemoryError(f"row {row_index}: profile plan needs {storage / 2**20:.1f} MiB, "
                                  "above GROND_JAX_HOST_BUFFER_GIB; increase it only with sufficient memory")
            if plans and (retained_bytes + storage > buffer_limit or len(plans) >= row_block_size):
                flush()
            plans.append(plan)
            retained_bytes += storage
            diagnostics["peak_plan_array_bytes"] = max(diagnostics["peak_plan_array_bytes"], retained_bytes)
        if plans:
            flush()
        self.jax_prediction_diagnostics = diagnostics
        self._verify_prediction_sources()
        output = self.pack_prediction_results(indices, results)
        output["seconds"] = time.perf_counter() - started
        print(f"JAX Bloch matrices={diagnostics['matrices']} plan_buffer_MiB={diagnostics['peak_plan_array_bytes'] / 2**20:.2f} "
              f"max_input_batch_MiB={diagnostics['max_input_batch_bytes'] / 2**20:.2f} "
              f"wall={output['seconds']:.2f}s", flush=True)
        return output

    def _hybrid_gpu_devices(self):
        """GPU row-worker devices, when this port's rows can use them (elastic SciPy rows, node thickness average)."""

        devices = hybrid_gpu_devices()
        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if (not devices or backend not in {"scipy", "numpy_scipy_cpu"} or self.absorption_enabled()
                or self.thickness_cf_parameters() is not None or bool(getattr(self.cfg, "paired_inverse_hand", False))):
            return []
        return devices

    def _predict_rows_hybrid(self, indices, workers, devices):
        """(row, prediction) of every row from the CPU pool and one process per listed GPU, fed from one queue.

        A GPU worker that fails for a non-physical reason (device, memory, process) is retired and its rows go back
        to the queue for the CPU workers; physical row errors and CPU failures are raised.
        """

        from collections import deque
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        cpu = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context(_pool_context_names()[0]),
                                  initializer=_init_prediction_worker, initargs=(self,))
        pools = [(cpu, _predict_row_worker, int(workers) + max(1, int(workers) // 4))]
        for device in devices:
            gpu = ProcessPoolExecutor(max_workers=1, mp_context=_gpu_pool_context(),
                                      initializer=_init_gpu_prediction_worker, initargs=(self, device))
            pools.append((gpu, _predict_row_gpu_worker, 2))
        pending = deque(int(v) for v in indices)
        running, retired, results = {}, set(), []
        counts = [0] * len(pools)

        def refill(slot):
            pool, worker, depth = pools[slot]
            while pending and slot not in retired and sum(s == slot for s, _ in running.values()) < depth:
                row = pending.popleft()
                running[pool.submit(worker, row)] = (slot, row)

        try:
            for slot in range(len(pools)):
                refill(slot)
            while running:
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    slot, row = running.pop(future)
                    try:
                        results.append(future.result())
                        counts[slot] += 1
                    except _NON_RETRYABLE_ERRORS:
                        raise
                    except Exception as error:
                        if slot == 0:
                            raise
                        if slot not in retired:
                            print(f"GPU row worker on device {devices[slot - 1]} failed ({type(error).__name__}: {error}); "
                                  "its rows go to the CPU workers", flush=True)
                        retired.add(slot)
                        pending.appendleft(row)
                    for target in sorted({slot, 0}):
                        refill(target)
        finally:
            for pool, _, _ in pools:
                pool.shutdown(wait=True, cancel_futures=True)
        print(f"hybrid prediction: {counts[0]} rows on {int(workers)} CPU workers, {sum(counts[1:])} rows on GPU devices "
              f"{','.join(devices)}" + (f" ({len(retired)} GPU workers retired)" if retired else ""), flush=True)
        return results

    def predict_rows_with_row_cache(self, indices, workers=1, chunksize=8):
        """Predict rows from an immutable row-geometry cache.

        This freezes row identity, support basis, Sg profiles, quadrature
        weights, and h_j-h_i lookup indices.  Current-model U_g values are still
        recomputed before prediction, so this cache is meant for LS1-style
        macrocycles where the geometry/support contract is held fixed while the
        structure changes.
        """

        indices = validate_row_indices(indices, row_count=len(self.data.iobs))
        self._verify_prediction_sources()
        t0 = time.perf_counter()
        cache = self.load_or_build_row_plan_cache(indices)
        if cache is None:
            return self.predict_rows(indices, workers=workers, chunksize=chunksize)
        if not np.array_equal(np.asarray(cache["indices"], dtype=int), indices):
            raise RuntimeError("row-plan cache index order does not match requested row order")
        unique_ug = self.current_unique_ug(cache["unique_diffs"])
        self._active_row_plan_cache = cache
        self._active_unique_ug = unique_ug

        results = [None] * len(indices)
        parallel_errors: list[str] = []

        def commit_attempt(attempt_results):
            """Check one complete attempt and copy its rows into ``results`` in the order of ``indices``."""

            ordered = _ordered_prediction_attempt(indices, attempt_results)
            for row_position, r in enumerate(ordered):
                results[row_position] = r

        try:
            if int(workers) > 1 and len(indices) > 1:
                ran_parallel = False
                for context_name in _pool_context_names():
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
                        self.row_prediction_from_cache(cache, row_position, unique_ug) for row_position in range(len(indices))
                    ]
                    commit_attempt(attempt_results)
            else:
                attempt_results = [self.row_prediction_from_cache(cache, row_position, unique_ug) for row_position in range(len(indices))]
                commit_attempt(attempt_results)
        finally:
            self._active_row_plan_cache = None
            self._active_unique_ug = None

        self._verify_prediction_sources()
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
        """Merge packed chunks key by key: arrays concatenated, support HKL lists joined, ``seconds`` summed."""

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
        """Pack per-row prediction dicts into one dict of arrays, in the order of ``indices``.

        Each row also gets a 64-bit hash of its support HKLs. ``dyn_inverse`` is kept only when every row has it, and
        ``seconds`` is 0.0 for the caller to fill.
        """

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
        position_by_row_index = {int(row_index): int(row_position) for row_position, row_index in enumerate(indices)}

        def accept_result(row_index, r):
            """Copy one row's result into the packed arrays at that row's position."""

            row_position = position_by_row_index[int(row_index)]
            dyn[row_position] = r["dyn"]
            if dyn_inverse is not None:
                dyn_inverse[row_position] = r["dyn_inverse"]
            born[row_position] = r["born"]
            basis_size[row_position] = r["basis_size"]
            support_array = np.asarray(r["support_hkls"], dtype=np.int32)
            basis_support_hash64[row_position] = int.from_bytes(
                hashlib.sha256(support_array.tobytes(order="C")).digest()[:8],
                byteorder="big",
                signed=False,
            )
            sg_resid[row_position] = r["max_sg_solve_residual"]
            center_sg[row_position] = r["center_sg_inv_A"]
            omega_bragg[row_position] = r["omega_bragg_deg"]
            omega_delta[row_position] = r["omega_bragg_minus_center_deg"]
            omega_bragg_root_available[row_position] = bool(r.get("omega_bragg_root_available", np.isfinite(r["omega_bragg_deg"])))
            sg_window[row_position] = r["sg_window_inv_A"]
            sg_domega[row_position] = r["sg_dSg_domega_inv_A_per_deg"]
            sg_dframe[row_position] = r["sg_dSg_dframe_inv_A"]
            sg_window_raw[row_position] = r["sg_window_raw_inv_A"]
            sg_window_half_frames[row_position] = r["sg_window_half_frames"]
            sg_window_clipped[row_position] = bool(r["sg_window_clipped"])
            sg_jitter_sigma[row_position] = r["sg_jitter_sigma_inv_A"]
            sg_jitter_d_inv[row_position] = r["sg_jitter_d_inv_A"]
            sg_jitter_frame_fraction[row_position] = r["sg_jitter_frame_fraction"]
            sg_jitter_kernel_points[row_position] = int(r["sg_jitter_kernel_points"])
            sg_profile_points_total[row_position] = int(r["sg_profile_points_total"])
            sg_profile_weight_sum[row_position] = float(r["sg_profile_weight_sum"])
            sg_profile_same_branch_continuation_used[row_position] = bool(r["sg_profile_same_branch_continuation_used"])
            sg_profile_same_branch_continuation_segments[row_position] = int(r["sg_profile_same_branch_continuation_segments"])
            support_hkls[row_position] = r["support_hkls"]

        for row_index, r in zip(indices, results, strict=True):
            accept_result(int(row_index), r)

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
        """Predict dynamical and kinematical intensities of ``indices`` (all rows by default) and return them packed.

        The backend chooses the row-plan cache or batched JAX. Otherwise rows run in a process pool (start methods
        tried in turn, then serially); an attempt counts only when it returns every row, and physical errors are
        raised rather than retried.
        """

        if indices is None:
            indices = np.arange(len(self.data.iobs))
        indices = validate_row_indices(indices, row_count=len(self.data.iobs))
        self._verify_prediction_sources()
        from grond.bloch import shared_grid

        path, reason = shared_grid.choose_path(self, indices, workers, purpose="predict")
        if shared_grid.mode() != "off":
            print(f"Bloch evaluator for {len(indices)} rows: {path} ({reason})", flush=True)
        if path == "shared_grid":
            t0 = time.perf_counter()
            results = shared_grid.predict_rows(self, indices, workers=workers)
            self._verify_prediction_sources()
            out = self.pack_prediction_results(indices, results)
            out["seconds"] = time.perf_counter() - t0
            print(f"predicted {len(indices)}/{len(indices)} rows (shared grid) | wall={out['seconds']:.2f}s | "
                  f"rate={len(indices) / max(out['seconds'], 1e-12):.2f}/s", flush=True)
            return out
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

        results = None
        devices = self._hybrid_gpu_devices() if int(workers) > 1 and len(indices) > 1 else []
        if devices:
            results = _ordered_prediction_attempt(indices, self._predict_rows_hybrid(indices, workers, devices))
        elif int(workers) > 1 and len(indices) > 1:
            for context_name in _pool_context_names():
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
            serial_attempt = [(int(row_index), self.row_prediction(int(row_index))) for row_index in indices]
            results = _ordered_prediction_attempt(indices, serial_attempt)
        self._verify_prediction_sources()
        out = self.pack_prediction_results(indices, results)
        elapsed = time.perf_counter() - t0
        out["seconds"] = elapsed
        print(
            f"predicted {len(indices)}/{len(indices)} rows | "
            f"wall={elapsed:.2f}s | rate={len(indices) / max(elapsed, 1e-12):.2f}/s",
            flush=True,
        )
        return out
