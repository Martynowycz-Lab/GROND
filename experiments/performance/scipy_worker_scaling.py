#!/usr/bin/env python3
"""Measure GROND Bloch backend scaling on one fixed row workload.

The default measures the production SciPy multiprocessing path.  Optional
backend arguments allow full-row timing of the same GROND workload with the
strong/weak Bethe-Schur reduction or accelerator backends.  This script is for
benchmarking complete row prediction, not isolated eigensolver microkernels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256  # noqa: E402


def parse_workers(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(1, int(part)))
    return out or [1]


def select_indices(data: XDSData, rows: int, top_isigma: bool) -> np.ndarray:
    indices = np.arange(len(data.iobs), dtype=int)
    if top_isigma:
        score = data.iobs / np.maximum(data.sigma, 1e-12)
        indices = indices[np.argsort(score)[::-1]]
    if rows > 0:
        indices = indices[: int(rows)]
    return np.asarray(indices, dtype=int)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
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
        compute_backend=str(args.compute_backend),
        jax_batch_size=int(args.jax_batch_size),
        jax_matrix_bucket_size=int(args.jax_matrix_bucket_size),
        bethe_strong_count=int(args.bethe_strong_count),
        bethe_variant=str(args.bethe_variant),
        bethe_promote_denom=float(args.bethe_promote_denom),
        bethe_denom_eps=float(args.bethe_denom_eps),
    )


def summarize_prediction(pred: dict, seconds: float) -> dict:
    basis = np.asarray(pred.get("basis_size", []), dtype=float)
    profile_points = np.asarray(pred.get("sg_profile_points_total", []), dtype=float)
    return {
        "seconds": float(seconds),
        "pred_seconds": float(pred.get("seconds", seconds)),
        "row_rate": float(len(pred["indices"]) / max(seconds, 1e-12)),
        "rows": int(len(pred["indices"])),
        "basis_size_median": float(np.median(basis)) if basis.size else 0.0,
        "basis_size_max": int(np.max(basis)) if basis.size else 0,
        "profile_points_total": int(np.sum(profile_points)) if profile_points.size else 0,
        "dyn_sum": float(np.sum(np.asarray(pred.get("dyn", []), dtype=float))),
        "born_sum": float(np.sum(np.asarray(pred.get("born", []), dtype=float))),
    }


def run(args: argparse.Namespace) -> dict:
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    model = parse_model(Path(args.res))
    cfg = build_config(args)
    port = XDSBlochPort(
        data,
        model,
        cfg,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        res_hash=sha256(Path(args.res)),
    )
    indices = select_indices(data, int(args.rows), bool(args.top_isigma))
    out = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.executable,
        "cpu_count": os.cpu_count(),
        "inputs": {
            "hkl": str(Path(args.hkl)),
            "res": str(Path(args.res)),
            "hkl_transform": args.hkl_transform,
        },
        "config": asdict(cfg),
        "row_selection": {
            "rows": int(len(indices)),
            "top_isigma": bool(args.top_isigma),
            "total_available_rows": int(len(data.iobs)),
        },
        "runs": {},
    }
    for workers in parse_workers(args.workers):
        t0 = time.perf_counter()
        pred = port.predict_rows(indices, workers=workers, chunksize=max(1, int(args.chunksize)))
        seconds = time.perf_counter() - t0
        out["runs"][str(workers)] = summarize_prediction(pred, seconds)
        print(
            f"workers={workers} rows={len(indices)} wall={seconds:.3f}s "
            f"rate={len(indices)/max(seconds,1e-12):.2f}/s "
            f"basis_median={out['runs'][str(workers)]['basis_size_median']:.1f} "
            f"basis_max={out['runs'][str(workers)]['basis_size_max']}",
            flush=True,
        )
        if args.save_arrays:
            arr_path = Path(args.out).with_suffix(f".workers{workers}.npz")
            np.savez_compressed(
                arr_path,
                indices=np.asarray(pred["indices"], dtype=int),
                dyn=np.asarray(pred["dyn"], dtype=float),
                born=np.asarray(pred["born"], dtype=float),
                basis_size=np.asarray(pred["basis_size"], dtype=int),
                sg_profile_points_total=np.asarray(pred["sg_profile_points_total"], dtype=int),
            )
            out["runs"][str(workers)]["array_file"] = str(arr_path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--hkl-transform", default=None)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--top-isigma", action="store_true")
    ap.add_argument("--workers", default="1,10")
    ap.add_argument("--chunksize", type=int, default=2)
    ap.add_argument("--compute-backend", default="scipy")
    ap.add_argument("--jax-batch-size", type=int, default=0)
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    ap.add_argument("--bethe-strong-count", type=int, default=48)
    ap.add_argument("--bethe-variant", default="schur_target", choices=["truncate", "schur0", "schur_target", "schur_mean"])
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005)
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5)
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
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-arrays", action="store_true")
    args = ap.parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
