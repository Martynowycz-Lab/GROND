#!/usr/bin/env python3
"""Compare dense GROND Bloch scoring against adaptive Bethe beam splitting.

This is an experimental full-row scoring driver. It uses the same row selection,
Born calculation, scale fitting, and R-factor scorer as the dense GROND engine,
then swaps only the dynamical propagation step for the near-resonance-promoted
strong/weak Bethe approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import platform
import time
from pathlib import Path

import numpy as np
from bethe_split_probe import bethe_effective_hamiltonian, summarize_errors

from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    XDSData,
    evaluate_predictions,
    parse_model,
    sha256,
)

_WORKER_PORT = None
_WORKER_BETHE_SETTINGS = None


def _init_bethe_worker(port: XDSBlochPort, settings: dict) -> None:
    """Initialize one multiprocessing worker with a read-only Bloch port."""

    global _WORKER_PORT, _WORKER_BETHE_SETTINGS
    _WORKER_PORT = port
    _WORKER_BETHE_SETTINGS = dict(settings)


def _predict_bethe_worker(row_idx: int) -> tuple[int, dict, dict]:
    """Multiprocessing wrapper for one Bethe row prediction."""

    if _WORKER_PORT is None or _WORKER_BETHE_SETTINGS is None:
        raise RuntimeError("Bethe prediction worker was not initialized")
    result, meta = predict_one_bethe_row(_WORKER_PORT, int(row_idx), **_WORKER_BETHE_SETTINGS)
    return int(row_idx), result, meta


def select_indices(
    data: XDSData,
    *,
    min_peak: float,
    min_corr: float,
    max_resolution: float,
    top_isigma: int,
    limit: int,
) -> np.ndarray:
    """Return the same style of selected row indices used by the engine CLI."""

    mask = np.ones(len(data.iobs), dtype=bool)
    if min_peak > 0.0:
        mask &= data.peak >= float(min_peak)
    if min_corr > 0.0:
        mask &= data.corr >= float(min_corr)
    if max_resolution > 0.0:
        dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl])
        mask &= dvals >= float(max_resolution)
    indices = np.where(mask)[0]
    if top_isigma > 0 and len(indices) > top_isigma:
        isigi = data.iobs[indices] / np.maximum(data.sigma[indices], 1e-12)
        indices = indices[np.argsort(isigi)[::-1][: int(top_isigma)]]
    if limit > 0:
        indices = indices[: int(limit)]
    return np.asarray(indices, dtype=int)


def full_prediction_dict(data: XDSData, port: XDSBlochPort, pred_subset: dict) -> dict:
    """Expand selected-row predictions to the full row array expected by the scorer."""

    full_dyn = np.full(len(data.iobs), np.nan, dtype=float)
    full_born = np.full(len(data.iobs), np.nan, dtype=float)
    full_dyn[pred_subset["indices"]] = pred_subset["dyn"]
    full_born[pred_subset["indices"]] = pred_subset["born"]
    return {"dyn": full_dyn, "born": full_born, "groups": port.group_indices}


def predict_one_bethe_row(
    port: XDSBlochPort,
    row_idx: int,
    *,
    strong_count: int,
    variant: str,
    promote_denom: float,
    denom_eps: float,
) -> tuple[dict, dict]:
    """Predict one row and return the row result plus diagnostic metadata."""

    t_plan = time.perf_counter()
    plan = port.row_prediction_plan(int(row_idx))
    plan_seconds = time.perf_counter() - t_plan
    dyn_vals = []
    actual_strong_counts = []
    near_denominators = 0
    matrix_count = 0
    prop_seconds = 0.0
    for H in plan["_dyn_matrices"]:
        matrix_count += 1
        t_prop = time.perf_counter()
        Heff, near, actual_m = bethe_effective_hamiltonian(
            H,
            strong_count,
            variant,
            denom_eps,
            promote_denom=promote_denom,
        )
        dyn_vals.append(port.propagate_direct_to_target_thickness_average(Heff))
        prop_seconds += time.perf_counter() - t_prop
        near_denominators += int(near)
        actual_strong_counts.append(int(actual_m))
    return port.finish_row_prediction_plan(plan, dyn_vals), {
        "actual_strong_counts": actual_strong_counts,
        "near_denominators": int(near_denominators),
        "matrix_count": int(matrix_count),
        "plan_seconds": float(plan_seconds),
        "prop_seconds": float(prop_seconds),
    }


def pool_context_names() -> list[str]:
    """Return multiprocessing contexts in the same conservative order as GROND."""

    if platform.system() == "Darwin":
        return ["forkserver", "spawn"]
    return ["forkserver", "fork", "spawn"]


def predict_rows_bethe(
    port: XDSBlochPort,
    indices: np.ndarray,
    *,
    strong_count: int,
    variant: str,
    promote_denom: float,
    denom_eps: float,
    workers: int,
    chunksize: int,
) -> dict:
    """Predict rows using a promoted strong/weak Bethe reduced Hamiltonian."""

    result_by_idx = {}
    actual_strong_counts = []
    near_denominators = 0
    matrix_count = 0
    plan_seconds = 0.0
    prop_seconds = 0.0
    wall0 = time.perf_counter()
    indices = np.asarray(indices, dtype=int)
    settings = {
        "strong_count": int(strong_count),
        "variant": variant,
        "promote_denom": float(promote_denom),
        "denom_eps": float(denom_eps),
    }

    def accept(row_idx: int, result: dict, meta: dict, pos: int) -> None:
        nonlocal near_denominators, matrix_count, plan_seconds, prop_seconds
        result_by_idx[int(row_idx)] = result
        near_denominators += int(meta["near_denominators"])
        matrix_count += int(meta["matrix_count"])
        plan_seconds += float(meta["plan_seconds"])
        prop_seconds += float(meta["prop_seconds"])
        actual_strong_counts.extend(int(v) for v in meta["actual_strong_counts"])
        if (pos + 1) % 500 == 0 or pos + 1 == len(indices):
            elapsed = time.perf_counter() - wall0
            rate = (pos + 1) / max(elapsed, 1e-12)
            eta = (len(indices) - (pos + 1)) / max(rate, 1e-12)
            print(f"Bethe predicted {pos + 1}/{len(indices)} rows | rate={rate:.2f}/s | ETA={eta / 60:.1f}m", flush=True)

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
                    initializer=_init_bethe_worker,
                    initargs=(port, settings),
                ) as pool:
                    for pos, (row_idx, result, meta) in enumerate(
                        pool.imap_unordered(
                            _predict_bethe_worker,
                            [int(v) for v in indices],
                            chunksize=max(1, int(chunksize)),
                        )
                    ):
                        accept(row_idx, result, meta, pos)
                ran_parallel = True
                break
            except Exception:
                ran_parallel = False
                continue
        if not ran_parallel:
            print("Bethe multiprocessing failed; falling back to serial.", flush=True)
            for pos, row_idx in enumerate(indices):
                result, meta = predict_one_bethe_row(port, int(row_idx), **settings)
                accept(int(row_idx), result, meta, pos)
    else:
        for pos, row_idx in enumerate(indices):
            result, meta = predict_one_bethe_row(port, int(row_idx), **settings)
            accept(int(row_idx), result, meta, pos)

    results = [result_by_idx[int(idx)] for idx in indices]
    pred = port.pack_prediction_results(indices, results)
    wall = time.perf_counter() - wall0
    pred["seconds"] = wall
    pred["bethe_meta"] = {
        "strong_count": int(strong_count),
        "variant": variant,
        "promote_denom": float(promote_denom),
        "denom_eps": float(denom_eps),
        "matrix_count": int(matrix_count),
        "near_denominators": int(near_denominators),
        "workers": int(workers),
        "plan_seconds": float(plan_seconds),
        "prop_seconds": float(prop_seconds),
        "wall_seconds": float(wall),
        "actual_strong_median": float(np.median(actual_strong_counts)) if actual_strong_counts else 0.0,
        "actual_strong_p90": float(np.percentile(actual_strong_counts, 90)) if actual_strong_counts else 0.0,
        "actual_strong_max": int(np.max(actual_strong_counts)) if actual_strong_counts else 0,
    }
    return pred


def compact_stats(eval_out: dict) -> dict:
    """Keep the top-level R-factor fields that matter for quick comparisons."""

    out = {"group_scale_count": eval_out["group_scale_count"], "splits": {}}
    for split_name, split in eval_out["splits"].items():
        dyn = split["dyn"]
        out["splits"][split_name] = {
            "n": dyn["n"],
            "n_gt": dyn["n_gt"],
            "R1_all": dyn["R1_all"],
            "R1_gt": dyn["R1_gt"],
            "wR2_all": dyn["wR2_all"],
            "CC": dyn["CC"],
        }
    return out


def write_metric_csv(path: Path, dense_eval: dict, bethe_eval: dict) -> None:
    """Write a compact dense-vs-Bethe table."""

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "split", "n", "n_gt", "R1_all", "R1_gt", "wR2_all", "CC", "group_scale_count"])
        for method, eval_out in [("dense", dense_eval), ("bethe", bethe_eval)]:
            for split_name, split in eval_out["splits"].items():
                dyn = split["dyn"]
                writer.writerow(
                    [
                        method,
                        split_name,
                        dyn["n"],
                        dyn["n_gt"],
                        dyn["R1_all"],
                        dyn["R1_gt"],
                        dyn["wR2_all"],
                        dyn["CC"],
                        eval_out["group_scale_count"],
                    ]
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--n-basis", type=int, default=128)
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--sg-window", type=float, default=0.006)
    ap.add_argument("--max-support-sg", type=float, default=0.05)
    ap.add_argument("--topup-d-min", type=float, default=0.70)
    ap.add_argument("--max-resolution", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--min-corr", type=float, default=0.0)
    ap.add_argument("--top-isigma", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thickness", type=float, default=1200.0)
    ap.add_argument("--thickness-model", choices=["single", "uniform_gradient", "gaussian", "gaussian_gradient"], default="uniform_gradient")
    ap.add_argument("--thickness-grad-frac", type=float, default=0.4)
    ap.add_argument("--thickness-sigma-frac", type=float, default=0.0)
    ap.add_argument("--thickness-points", type=int, default=3)
    ap.add_argument("--group-width-deg", type=float, default=1.0)
    ap.add_argument("--dense-workers", type=int, default=1)
    ap.add_argument("--bethe-workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--bethe-strong-count", type=int, default=48)
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005)
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    model = parse_model(Path(args.res))
    cfg = Config(
        n_basis=args.n_basis,
        n_points=args.n_points,
        sg_window=args.sg_window,
        max_support_sg=args.max_support_sg,
        topup_d_min=args.topup_d_min,
        thickness_A=args.thickness,
        thickness_model=args.thickness_model,
        thickness_grad_frac=args.thickness_grad_frac,
        thickness_sigma_frac=args.thickness_sigma_frac,
        thickness_points=args.thickness_points,
        group_width_deg=args.group_width_deg,
        compute_backend="scipy",
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    setup0 = time.perf_counter()
    port = XDSBlochPort(data, model, cfg, cache_dir=cache_dir, res_hash=sha256(Path(args.res)))
    setup_seconds = time.perf_counter() - setup0
    indices = select_indices(
        data,
        min_peak=args.min_peak,
        min_corr=args.min_corr,
        max_resolution=args.max_resolution,
        top_isigma=args.top_isigma,
        limit=args.limit,
    )
    print(f"selected {len(indices)} rows", flush=True)

    dense0 = time.perf_counter()
    dense_pred_subset = port.predict_rows(indices, workers=args.dense_workers, chunksize=args.chunksize)
    dense_seconds = time.perf_counter() - dense0
    dense_pred = full_prediction_dict(data, port, dense_pred_subset)
    dense_eval = evaluate_predictions(data, cfg, dense_pred, fit_scales_on="work")

    bethe_pred_subset = predict_rows_bethe(
        port,
        indices,
        strong_count=args.bethe_strong_count,
        variant=args.bethe_variant,
        promote_denom=args.bethe_promote_denom,
        denom_eps=args.bethe_denom_eps,
        workers=args.bethe_workers,
        chunksize=args.chunksize,
    )
    bethe_pred = full_prediction_dict(data, port, bethe_pred_subset)
    bethe_eval = evaluate_predictions(data, cfg, bethe_pred, fit_scales_on="work")

    q_error = summarize_errors(
        dense_pred_subset["dyn"] / np.maximum(dense_pred_subset["born"], 1e-30),
        bethe_pred_subset["dyn"] / np.maximum(bethe_pred_subset["born"], 1e-30),
    )
    summary = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "hkl_transform": args.hkl_transform,
        },
        "config": vars(args),
        "rows": int(len(indices)),
        "setup_seconds": float(setup_seconds),
        "dense": {
            "prediction_seconds": float(dense_pred_subset.get("seconds", dense_seconds)),
            "wall_seconds": float(dense_seconds),
            "rows_per_second": float(len(indices) / max(dense_seconds, 1e-12)),
            "basis_size_median": float(np.median(dense_pred_subset["basis_size"])) if len(indices) else 0.0,
            "basis_size_max": int(np.max(dense_pred_subset["basis_size"])) if len(indices) else 0,
            "metrics": compact_stats(dense_eval),
        },
        "bethe": {
            "prediction_seconds": float(bethe_pred_subset["seconds"]),
            "rows_per_second": float(len(indices) / max(bethe_pred_subset["seconds"], 1e-12)),
            "meta": bethe_pred_subset["bethe_meta"],
            "metrics": compact_stats(bethe_eval),
        },
        "bethe_vs_dense_q_error": q_error,
        "speedup_bethe_vs_dense_wall": float(dense_seconds / max(bethe_pred_subset["seconds"], 1e-12)),
    }
    summary_path = out_dir / "bethe_full_run_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    write_metric_csv(out_dir / "bethe_full_run_compare_metrics.csv", dense_eval, bethe_eval)
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"wrote {summary_path}")
    print(f"wrote {out_dir / 'bethe_full_run_compare_metrics.csv'}")


if __name__ == "__main__":
    main()
