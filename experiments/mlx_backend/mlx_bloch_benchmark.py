#!/usr/bin/env python3
"""Benchmark experimental MLX propagation paths on real GROND Bloch matrices.

This is deliberately an experiment, not a production backend.  It reuses the
GROND row-planning code to build the same Hamiltonians used by the SciPy engine,
then swaps only the propagation kernel.  The goal is to answer whether Apple
Metal can accelerate either of these paths:

* MLX complex eig/eigh for the ordinary Bloch diagonalization.
* A real-embedded matrix exponential that avoids eig/eigh and uses batched
  matrix multiplication on Metal.

The benchmark writes a JSON summary with capability checks, timing, and parity
against SciPy.  Promotion should require small row-intensity error and a real
end-to-end speed win.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.linalg import eig, eigh

from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    XDSData,
    _stack_square_matrices,
    parse_model,
    promoted_bethe_effective_hamiltonian,
    sha256,
)


DEFAULT_HKL = Path(
    "/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/"
    "xds20230630_1x/no_prf20260511/INTEGRATE.HKL"
)
DEFAULT_RES = Path(
    "/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/"
    "xds20230630_1x/temp_a.res"
)


def import_mlx():
    """Import MLX lazily so the script can still report a useful error."""

    try:
        import mlx
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - depends on optional package.
        return None, None, {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return mlx, mx, {"available": True, "version": getattr(mlx, "__version__", "unknown")}


def mlx_capabilities(mx) -> dict:
    """Probe the MLX operations needed by the two experimental paths."""

    out = {
        "metal_available": None,
        "default_device": None,
        "ops": {},
    }
    if mx is None:
        return out
    try:
        out["metal_available"] = bool(mx.metal.is_available())
    except Exception as exc:
        out["metal_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["default_device"] = str(mx.default_device())
    except Exception as exc:
        out["default_device_error"] = f"{type(exc).__name__}: {exc}"

    H = np.eye(4, dtype=np.complex64)
    H[0, 1] = 0.01 + 0.02j
    H[1, 0] = 0.01 - 0.02j
    H_mx = mx.array(H)
    eye_mx = mx.array(np.eye(4, dtype=np.complex64))
    real_batch = mx.array(np.repeat(np.eye(8, dtype=np.float32)[None, :, :], 16, axis=0))

    def record(name: str, stream_name: str, func):
        try:
            t0 = time.perf_counter()
            result = func()
            if isinstance(result, tuple):
                mx.eval(*result)
            else:
                mx.eval(result)
            out["ops"][f"{name}_{stream_name}"] = {"ok": True, "seconds": time.perf_counter() - t0}
        except Exception as exc:
            out["ops"][f"{name}_{stream_name}"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            }

    for stream_name, stream in (("cpu", mx.cpu), ("gpu", mx.gpu)):
        record("eig", stream_name, lambda s=stream: mx.linalg.eig(H_mx, stream=s))
        record("eigh", stream_name, lambda s=stream: mx.linalg.eigh(H_mx, stream=s))
        record("eigvals", stream_name, lambda s=stream: mx.linalg.eigvals(H_mx, stream=s))
        record("inv", stream_name, lambda s=stream: mx.linalg.inv(H_mx, stream=s))
        record("solve", stream_name, lambda s=stream: mx.linalg.solve(H_mx, eye_mx, stream=s))
    record("batched_real_matmul", "gpu", lambda: mx.matmul(real_batch, real_batch))
    return out


def select_indices(data: XDSData, limit: int, top_isigma: bool) -> np.ndarray:
    """Choose a deterministic subset for benchmarking."""

    indices = np.arange(len(data.iobs), dtype=int)
    if top_isigma:
        score = data.iobs / np.maximum(data.sigma, 1e-12)
        indices = indices[np.argsort(score)[::-1]]
    if limit > 0:
        indices = indices[: int(limit)]
    return np.asarray(indices, dtype=int)


def build_port(args: argparse.Namespace) -> tuple[XDSData, XDSBlochPort, Config]:
    """Construct the real GROND row planner with benchmark settings."""

    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    model = parse_model(Path(args.res))
    cfg = Config(
        n_basis=int(args.n_basis),
        n_points=int(args.n_points),
        thickness_A=float(args.thickness_A),
        thickness_model=str(args.thickness_model),
        thickness_grad_frac=float(args.thickness_grad_frac),
        thickness_sigma_frac=float(args.thickness_sigma_frac),
        thickness_points=int(args.thickness_points),
        topup_d_min=float(args.topup_d_min),
        max_support_sg=float(args.max_support_sg),
        sg_window=float(args.sg_window),
        sg_window_mode=str(args.sg_window_mode),
        profile_quadrature=str(args.profile_quadrature),
        profile_accumulation="sum",
        absorption_mode=str(args.absorption_mode),
        absorption_mu=float(args.absorption_mu),
        absorption_frac=float(args.absorption_frac),
        absorption_mu_g2=float(args.absorption_mu_g2),
        absorption_mu_theta2=float(args.absorption_mu_theta2),
        absorption_tds_b=float(args.absorption_tds_b),
        ug_d_min=float(args.ug_d_min),
        compute_backend="scipy",
        bethe_strong_count=int(args.bethe_strong_count),
        bethe_variant=str(args.bethe_variant),
        bethe_promote_denom=float(args.bethe_promote_denom),
        bethe_denom_eps=float(args.bethe_denom_eps),
    )
    port = XDSBlochPort(data, model, cfg, cache_dir=Path(args.cache_dir) if args.cache_dir else None, res_hash=sha256(Path(args.res)))
    return data, port, cfg


def scipy_propagate_matrix(H: np.ndarray, thick: np.ndarray, thick_w: np.ndarray, absorption: bool) -> float:
    """Reference SciPy propagation for one Hamiltonian."""

    if absorption:
        vals, vecs = eig(H)
        try:
            inv_vecs = np.linalg.inv(vecs)
        except np.linalg.LinAlgError:
            inv_vecs = np.linalg.pinv(vecs)
        coupling = vecs[1, :] * inv_vecs[:, 0]
    else:
        vals, vecs = eigh(H)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
    phase = np.exp((2j * np.pi) * vals[:, None] * thick[None, :])
    per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
    return float(np.sum(thick_w * per_t))


def scipy_propagate_batch(H_batch: np.ndarray, thick: np.ndarray, thick_w: np.ndarray, absorption: bool) -> np.ndarray:
    """Reference SciPy propagation for a batch of Hamiltonians."""

    return np.asarray([scipy_propagate_matrix(H, thick, thick_w, absorption) for H in H_batch], dtype=float)


def real_embedding_generator_batch(H_batch: np.ndarray, thickness: float, dtype=np.float32) -> np.ndarray:
    """Return real block matrices for exp((2*pi*i*thickness)*H)."""

    H_batch = np.asarray(H_batch, dtype=np.complex128)
    n_batch, n, _ = H_batch.shape
    factor = 2.0 * math.pi * float(thickness)
    real_part = (-factor * H_batch.imag).astype(dtype, copy=False)
    imag_part = (factor * H_batch.real).astype(dtype, copy=False)
    out = np.zeros((n_batch, 2 * n, 2 * n), dtype=dtype)
    out[:, :n, :n] = real_part
    out[:, :n, n:] = -imag_part
    out[:, n:, :n] = imag_part
    out[:, n:, n:] = real_part
    return out


def scale_steps_for_batch(real_batch: np.ndarray, theta: float) -> int:
    """Choose a single scaling count for a batch."""

    norm_inf = float(np.max(np.sum(np.abs(real_batch), axis=2))) if real_batch.size else 0.0
    if not np.isfinite(norm_inf) or norm_inf <= 0.0:
        return 0
    return max(0, int(math.ceil(math.log2(norm_inf / max(float(theta), 1e-12)))))


def mlx_taylor_expm_column0(mx, real_batch: np.ndarray, *, order: int, theta: float) -> tuple[np.ndarray, dict]:
    """Approximate exp(real_batch) column 0 on the MLX default device."""

    if real_batch.ndim != 3 or real_batch.shape[1] != real_batch.shape[2]:
        raise ValueError(f"expected square real batch, got {real_batch.shape}")
    n2 = int(real_batch.shape[1])
    n = n2 // 2
    scale_steps = scale_steps_for_batch(real_batch, theta)
    inv_scale = np.float32(1.0 / float(2**scale_steps))
    t0 = time.perf_counter()
    A = mx.array(real_batch * inv_scale, dtype=mx.float32)
    eye = mx.eye(n2, dtype=mx.float32)
    term = mx.broadcast_to(eye, A.shape)
    result = term
    for k in range(1, int(order) + 1):
        term = mx.matmul(term, A) / np.float32(k)
        result = result + term
    for _ in range(scale_steps):
        result = mx.matmul(result, result)
    real_target = result[:, 1, 0]
    imag_target = result[:, n + 1, 0]
    vals = real_target * real_target + imag_target * imag_target
    mx.eval(vals)
    return np.asarray(vals, dtype=float), {
        "seconds": time.perf_counter() - t0,
        "scale_steps": int(scale_steps),
        "order": int(order),
        "theta": float(theta),
        "n_real": int(n2),
    }


def mlx_real_expm_propagate_batch(
    mx,
    H_batch: np.ndarray,
    thick: np.ndarray,
    thick_w: np.ndarray,
    *,
    order: int,
    theta: float,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    """Experimental Metal propagation with real-embedded Taylor expm."""

    if not bool(mx.metal.is_available()):
        raise RuntimeError("MLX Metal backend is not available")
    mx.set_default_device(mx.gpu)
    H_batch = np.asarray(H_batch, dtype=np.complex128)
    out = np.zeros(H_batch.shape[0], dtype=float)
    total_seconds = 0.0
    chunks = 0
    scale_steps = []
    for start in range(0, len(H_batch), int(batch_size)):
        sub = H_batch[start : start + int(batch_size)]
        chunk_out = np.zeros(len(sub), dtype=float)
        for t, w in zip(thick, thick_w):
            real_batch = real_embedding_generator_batch(sub, float(t), dtype=np.float32)
            vals, meta = mlx_taylor_expm_column0(mx, real_batch, order=order, theta=theta)
            chunk_out += float(w) * vals
            total_seconds += float(meta["seconds"])
            scale_steps.append(int(meta["scale_steps"]))
        out[start : start + len(sub)] = chunk_out
        chunks += 1
    return out, {
        "seconds": float(total_seconds),
        "chunks": int(chunks),
        "batch_size": int(batch_size),
        "order": int(order),
        "theta": float(theta),
        "scale_steps_min": int(min(scale_steps)) if scale_steps else 0,
        "scale_steps_max": int(max(scale_steps)) if scale_steps else 0,
        "scale_steps_median": float(np.median(scale_steps)) if scale_steps else 0.0,
    }


def collect_jobs(port: XDSBlochPort, indices: np.ndarray, *, bucket_size: int, bethe: bool) -> tuple[list[dict], dict[int, list[tuple[int, int, np.ndarray]]], dict]:
    """Build row plans and group Hamiltonian jobs by padded matrix size."""

    plans: list[dict] = []
    jobs_by_size: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    t0 = time.perf_counter()
    for pos, idx in enumerate(indices):
        plan = port.row_prediction_plan(int(idx))
        plans.append(plan)
        for profile_pos, H in enumerate(plan["_dyn_matrices"]):
            H_use = promoted_bethe_effective_hamiltonian(H, port.cfg) if bethe else H
            n = int(H_use.shape[0])
            if bucket_size > 0:
                n = int(math.ceil(n / int(bucket_size)) * int(bucket_size))
            jobs_by_size.setdefault(n, []).append((pos, profile_pos, H_use))
        if (pos + 1) % 100 == 0 or pos + 1 == len(indices):
            elapsed = time.perf_counter() - t0
            rate = (pos + 1) / max(elapsed, 1e-12)
            print(f"planned {pos + 1}/{len(indices)} rows | rate={rate:.2f}/s", flush=True)
    plan_seconds = time.perf_counter() - t0
    dims = {int(n): int(len(jobs)) for n, jobs in sorted(jobs_by_size.items())}
    return plans, jobs_by_size, {"plan_seconds": float(plan_seconds), "matrix_count": int(sum(dims.values())), "matrix_dimensions": dims}


def values_to_rows(plans: list[dict], values_by_key: dict[tuple[int, int], float]) -> np.ndarray:
    """Reduce profile-matrix values back to row-level dynamical values."""

    row_vals = np.zeros(len(plans), dtype=float)
    for row_pos, plan in enumerate(plans):
        weights = np.asarray(plan["_profile_weights"], dtype=float)
        vals = np.asarray([values_by_key[(row_pos, j)] for j in range(len(plan["_dyn_matrices"]))], dtype=float)
        row_vals[row_pos] = float(np.sum(weights * vals))
    return row_vals


def summarize_error(trial: np.ndarray, reference: np.ndarray) -> dict:
    """Return robust absolute and relative error summaries."""

    trial = np.asarray(trial, dtype=float)
    reference = np.asarray(reference, dtype=float)
    abs_err = np.abs(trial - reference)
    rel_err = abs_err / np.maximum(np.abs(reference), 1e-30)
    return {
        "max_abs": float(np.max(abs_err)) if abs_err.size else 0.0,
        "median_abs": float(np.median(abs_err)) if abs_err.size else 0.0,
        "p95_abs": float(np.percentile(abs_err, 95)) if abs_err.size else 0.0,
        "max_rel": float(np.max(rel_err)) if rel_err.size else 0.0,
        "median_rel": float(np.median(rel_err)) if rel_err.size else 0.0,
        "p95_rel": float(np.percentile(rel_err, 95)) if rel_err.size else 0.0,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the benchmark and return a JSON-serializable summary."""

    mlx, mx, mlx_info = import_mlx()
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "platform": platform.platform(),
        "mlx": mlx_info,
        "capabilities": mlx_capabilities(mx),
        "inputs": {
            "hkl": str(Path(args.hkl)),
            "res": str(Path(args.res)),
            "hkl_transform": args.hkl_transform,
        },
        "config": {},
        "jobs": {},
        "results": {},
        "verdict": {},
    }
    if mx is None:
        summary["verdict"] = {"promote": False, "reason": "MLX is not importable"}
        return summary

    data, port, cfg = build_port(args)
    summary["config"] = asdict(cfg)
    indices = select_indices(data, int(args.rows), bool(args.top_isigma))
    summary["row_selection"] = {
        "rows": int(len(indices)),
        "top_isigma": bool(args.top_isigma),
        "first_index": int(indices[0]) if len(indices) else None,
        "last_index": int(indices[-1]) if len(indices) else None,
        "total_available_rows": int(len(data.iobs)),
    }
    plans, jobs_by_size, job_meta = collect_jobs(port, indices, bucket_size=int(args.bucket_size), bethe=bool(args.bethe))
    summary["jobs"] = job_meta
    absorption = port.absorption_enabled()
    summary["absorption_enabled"] = bool(absorption)

    scipy_values_by_key: dict[tuple[int, int], float] = {}
    scipy_seconds = 0.0
    scipy_rows = None
    if args.reference_array:
        ref = np.load(Path(args.reference_array), allow_pickle=False)
        ref_indices = np.asarray(ref["indices"], dtype=int)
        if len(ref_indices) != len(indices) or not np.array_equal(ref_indices, indices):
            raise ValueError("reference array indices do not match this benchmark row selection")
        scipy_rows = np.asarray(ref["dyn"], dtype=float)
        summary["results"]["external_reference"] = {
            "array_file": str(Path(args.reference_array)),
            "rows": int(len(scipy_rows)),
        }
    elif not args.skip_serial_reference:
        for n, jobs in sorted(jobs_by_size.items()):
            H_batch = _stack_square_matrices((j[2] for j in jobs), pad_to=n)
            t0 = time.perf_counter()
            vals = scipy_propagate_batch(H_batch, port.thick, port.thick_w, absorption)
            elapsed = time.perf_counter() - t0
            scipy_seconds += elapsed
            for val, (row_pos, profile_pos, _H) in zip(vals, jobs):
                scipy_values_by_key[(row_pos, profile_pos)] = float(val)
            print(f"SciPy size={n} matrices={len(jobs)} wall={elapsed:.3f}s rate={len(jobs)/max(elapsed,1e-12):.2f}/s", flush=True)
        scipy_rows = values_to_rows(plans, scipy_values_by_key)
        summary["results"]["scipy_reference"] = {
            "seconds": float(scipy_seconds),
            "matrix_rate": float(job_meta["matrix_count"] / max(scipy_seconds, 1e-12)),
            "row_rate": float(len(indices) / max(scipy_seconds + job_meta["plan_seconds"], 1e-12)),
        }

    if int(args.scipy_worker_reference) > 0 and not bool(args.bethe):
        t0 = time.perf_counter()
        pred = port.predict_rows(
            indices,
            workers=int(args.scipy_worker_reference),
            chunksize=max(1, int(args.scipy_worker_chunksize)),
        )
        wall = time.perf_counter() - t0
        worker_rows = np.asarray(pred["dyn"], dtype=float)
        worker_result = {
            "workers": int(args.scipy_worker_reference),
            "chunksize": int(args.scipy_worker_chunksize),
            "seconds": float(wall),
            "pred_seconds": float(pred.get("seconds", wall)),
            "row_rate": float(len(indices) / max(wall, 1e-12)),
            "basis_size_median": float(np.median(pred["basis_size"])) if len(pred["basis_size"]) else 0.0,
            "basis_size_max": int(np.max(pred["basis_size"])) if len(pred["basis_size"]) else 0,
        }
        if scipy_rows is not None:
            worker_result["row_error_vs_reference"] = summarize_error(worker_rows, scipy_rows)
        else:
            scipy_rows = worker_rows
            worker_result["used_as_reference"] = True
        summary["results"]["scipy_production_workers"] = worker_result

    if args.try_mlx_cpu_eig:
        summary["results"]["mlx_cpu_eig"] = {
            "promote": False,
            "reason": "MLX eig/eigh are CPU-only here and MLX complex inverse is unsupported; not benchmarked as a production path.",
        }

    for order in [int(v) for v in str(args.expm_orders).split(",") if str(v).strip()]:
        trial_values_by_key: dict[tuple[int, int], float] = {}
        total_seconds = 0.0
        metas = []
        failed = None
        try:
            for n, jobs in sorted(jobs_by_size.items()):
                H_batch = _stack_square_matrices((j[2] for j in jobs), pad_to=n)
                vals, meta = mlx_real_expm_propagate_batch(
                    mx,
                    H_batch,
                    port.thick,
                    port.thick_w,
                    order=order,
                    theta=float(args.expm_theta),
                    batch_size=int(args.mlx_batch_size),
                )
                total_seconds += float(meta["seconds"])
                metas.append({"size": int(n), "matrices": int(len(jobs)), **meta})
                for val, (row_pos, profile_pos, _H) in zip(vals, jobs):
                    trial_values_by_key[(row_pos, profile_pos)] = float(val)
                print(
                    f"MLX real-expm order={order} size={n} matrices={len(jobs)} "
                    f"wall={meta['seconds']:.3f}s rate={len(jobs)/max(meta['seconds'],1e-12):.2f}/s",
                    flush=True,
                )
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
        key = f"mlx_gpu_real_expm_order_{order}"
        if failed is not None:
            summary["results"][key] = {"ok": False, "error": failed}
            continue
        trial_rows = values_to_rows(plans, trial_values_by_key)
        result = {
            "ok": True,
            "seconds": float(total_seconds),
            "matrix_rate": float(job_meta["matrix_count"] / max(total_seconds, 1e-12)),
            "row_rate_including_plan": float(len(indices) / max(total_seconds + job_meta["plan_seconds"], 1e-12)),
            "matrix_group_meta": metas,
        }
        if scipy_rows is not None:
            result["row_error_vs_reference"] = summarize_error(trial_rows, scipy_rows)
            result["row_error_vs_scipy"] = result["row_error_vs_reference"]
        else:
            result["row_error_vs_reference"] = None
        summary["results"][key] = result

    best = None
    for key, result in summary["results"].items():
        if not key.startswith("mlx_gpu_real_expm") or not result.get("ok"):
            continue
        err = result.get("row_error_vs_reference")
        if err is None:
            continue
        if err["p95_rel"] < float(args.accept_p95_rel) and err["max_rel"] < float(args.accept_max_rel):
            if best is None or result["seconds"] < summary["results"][best]["seconds"]:
                best = key
    if best is None:
        summary["verdict"] = {
            "promote": False,
            "reason": "No MLX Metal real-expm run met the SciPy parity thresholds.",
            "accept_p95_rel": float(args.accept_p95_rel),
            "accept_max_rel": float(args.accept_max_rel),
        }
    else:
        best_total = float(summary["jobs"]["plan_seconds"]) + float(summary["results"][best]["seconds"])
        speedup = None
        if "scipy_reference" in summary["results"]:
            speedup = summary["results"]["scipy_reference"]["seconds"] / max(summary["results"][best]["seconds"], 1e-12)
        worker_speedup = None
        if "scipy_production_workers" in summary["results"]:
            worker_speedup = summary["results"]["scipy_production_workers"]["seconds"] / max(best_total, 1e-12)
        serial_ok = True if speedup is None else speedup > 1.1
        worker_ok = True if worker_speedup is None else worker_speedup > 1.1
        has_timing_reference = speedup is not None or worker_speedup is not None
        summary["verdict"] = {
            "promote": bool(has_timing_reference and serial_ok and worker_ok),
            "best_mlx": best,
            "speedup_vs_scipy_propagation_only": None if speedup is None else float(speedup),
            "speedup_vs_scipy_workers_end_to_end": None if worker_speedup is None else float(worker_speedup),
            "mlx_end_to_end_seconds": float(best_total),
            "reason": (
                "Promote only if parity thresholds are met and MLX beats both single-core "
                "propagation and the production CPU-worker path when that reference is measured; "
                "external reference arrays alone prove parity, not speed promotion."
            ),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--hkl-transform", default=None)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--top-isigma", action="store_true")
    ap.add_argument("--n-basis", type=int, default=128)
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--thickness-A", type=float, default=1240.0)
    ap.add_argument("--thickness-model", default="uniform_gradient")
    ap.add_argument("--thickness-grad-frac", type=float, default=0.25)
    ap.add_argument("--thickness-sigma-frac", type=float, default=0.0)
    ap.add_argument("--thickness-points", type=int, default=7)
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--sg-window", type=float, default=0.006)
    ap.add_argument("--sg-window-mode", default="fixed")
    ap.add_argument("--profile-quadrature", default="uniform", choices=["uniform", "trapezoid", "gauss_legendre"])
    ap.add_argument("--absorption-mode", default="midgley_like")
    ap.add_argument("--absorption-mu", type=float, default=0.0)
    ap.add_argument("--absorption-frac", type=float, default=0.05)
    ap.add_argument("--absorption-mu-g2", type=float, default=0.0)
    ap.add_argument("--absorption-mu-theta2", type=float, default=0.0)
    ap.add_argument("--absorption-tds-b", type=float, default=0.0)
    ap.add_argument("--ug-d-min", type=float, default=0.30)
    ap.add_argument("--bucket-size", type=int, default=8)
    ap.add_argument("--bethe", action="store_true", help="Apply the existing SciPy Bethe/Schur reduction before propagation.")
    ap.add_argument("--bethe-strong-count", type=int, default=48)
    ap.add_argument("--bethe-variant", default="schur_target", choices=["truncate", "schur0", "schur_target", "schur_mean"])
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005)
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    ap.add_argument("--mlx-batch-size", type=int, default=8)
    ap.add_argument("--expm-orders", default="12,18,24,32")
    ap.add_argument("--expm-theta", type=float, default=0.5)
    ap.add_argument("--try-mlx-cpu-eig", action="store_true")
    ap.add_argument("--scipy-worker-reference", type=int, default=0, help="Also run production port.predict_rows with this many SciPy CPU workers.")
    ap.add_argument("--scipy-worker-chunksize", type=int, default=1)
    ap.add_argument("--skip-serial-reference", action="store_true", help="Skip the single-process SciPy propagation reference.")
    ap.add_argument("--reference-array", default="", help="NPZ from scipy_worker_scaling.py --save-arrays to use as the full-row reference.")
    ap.add_argument("--accept-p95-rel", type=float, default=1e-3)
    ap.add_argument("--accept-max-rel", type=float, default=5e-2)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out", default="experiments/mlx_backend/mlx_bloch_benchmark_summary.json")
    args = ap.parse_args()

    summary = run_benchmark(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["verdict"], indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
