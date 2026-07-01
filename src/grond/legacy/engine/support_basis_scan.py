#!/usr/bin/env python3
"""Count Sg-limited Bloch basis sizes without running matrix propagation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from native_row_regularize import TARGET_MODES, row_selection, target_arrays
from bloch_engine import Config, XDSBlochPort, XDSData, alternate_hand_model, parse_model, sha256


_WORKER_PORT = None


def _pool_context_names() -> list[str]:
    import platform

    if platform.system() == "Darwin":
        return ["forkserver", "spawn"]
    return ["forkserver", "fork", "spawn"]


def _init_worker(port):
    global _WORKER_PORT
    _WORKER_PORT = port


def _basis_size_worker(idx):
    if _WORKER_PORT is None:
        raise RuntimeError("support scan worker was not initialized")
    port = _WORKER_PORT
    row_idx = int(idx)
    hkl = port.data.hkl[row_idx]
    omega_center = port.omega_from_z(port.data.zcal[row_idx])
    omega_bragg = port.solve_omega_for_sg(hkl, omega_center, 0.0)
    basis = port.select_basis(hkl, omega_bragg, row_idx=row_idx)
    return row_idx, int(len(basis))


def scan_basis_sizes(port: XDSBlochPort, indices: np.ndarray, workers: int, chunksize: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    sizes = np.zeros(len(indices), dtype=int)
    position = {int(idx): pos for pos, idx in enumerate(indices)}

    def accept(idx, size):
        sizes[position[int(idx)]] = int(size)

    def run_serial():
        _init_worker(port)
        for idx in indices:
            row_idx, size = _basis_size_worker(int(idx))
            accept(row_idx, size)

    if int(workers) > 1 and len(indices) > 1:
        ran_parallel = False
        for context_name in _pool_context_names():
            try:
                ctx = multiprocessing.get_context(context_name)
            except Exception:
                continue
            try:
                with ctx.Pool(processes=int(workers), initializer=_init_worker, initargs=(port,)) as pool:
                    for idx, size in pool.imap_unordered(
                        _basis_size_worker,
                        [int(v) for v in indices],
                        chunksize=max(1, int(chunksize)),
                    ):
                        accept(idx, size)
                ran_parallel = True
                break
            except Exception:
                ran_parallel = False
                continue
        if not ran_parallel:
            run_serial()
    else:
        run_serial()
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-basis-limit", type=int, default=512)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--min-final", type=int, default=20)
    ap.add_argument("--max-final", type=int, default=500)
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=99.0)
    ap.add_argument("--target-mode", choices=list(TARGET_MODES), default="iobs")
    ap.add_argument("--positive-only", action="store_true", default=True)
    ap.add_argument("--include-nonpositive", dest="positive_only", action="store_false")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true")
    args = ap.parse_args()
    effective_topup_d_min = min(float(args.topup_d_min), float(args.min_d_spacing)) if args.min_d_spacing > 0 else float(args.topup_d_min)

    t0 = time.perf_counter()
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    target, _base_sig = target_arrays(data, args.target_mode)
    indices = row_selection(data, target, args.positive_only, args.min_d_spacing, args.min_peak, args.max_rows, args.geometry_percentile)
    model = parse_model(Path(args.res))
    if args.alternate_hand:
        model = alternate_hand_model(model)
    cfg = Config(
        n_basis=int(args.n_basis_limit),
        n_points=1,
        topup_d_min=effective_topup_d_min,
        max_support_sg=float(args.max_support_sg),
    )
    port = XDSBlochPort(data, model, cfg, cache_dir=Path(args.out).resolve().parent / "support_cache", res_hash=sha256(Path(args.res)))
    sizes = scan_basis_sizes(port, indices, args.workers, args.chunksize)
    max_observed = int(np.max(sizes)) if len(sizes) else 0
    uncapped_recommended = max(int(args.min_final), max_observed + int(args.margin))
    recommended = min(int(args.max_final), uncapped_recommended)
    summary = {
        "rows": int(len(indices)),
        "n_basis_limit": int(args.n_basis_limit),
        "margin": int(args.margin),
        "min_final": int(args.min_final),
        "max_final": int(args.max_final),
        "selection_rule": "recommended_n_basis = min(max_final, max(min_final, observed_rowwise_max_basis + margin))",
        "observed_max_plus_margin": int(max_observed + int(args.margin)),
        "uncapped_recommended_n_basis": int(uncapped_recommended),
        "recommended_n_basis": int(recommended),
        "cap_hit": bool(recommended < uncapped_recommended),
        "basis_size_stats": {
            "min": int(np.min(sizes)) if len(sizes) else 0,
            "median": float(np.median(sizes)) if len(sizes) else 0.0,
            "p95": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
            "p99": float(np.percentile(sizes, 99)) if len(sizes) else 0.0,
            "max": max_observed,
        },
        "truncated_by_limit": bool(max_observed >= int(args.n_basis_limit)),
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "hkl_transform": data.hkl_transform_spec,
            "topup_d_min": float(effective_topup_d_min),
            "requested_topup_d_min": float(args.topup_d_min),
            "max_support_sg": float(args.max_support_sg),
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
