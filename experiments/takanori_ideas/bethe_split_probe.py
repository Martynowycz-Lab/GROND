#!/usr/bin/env python3
"""Probe strong/weak Bethe-style beam splitting against dense GROND Bloch rows.

This is an experimental parity/speed script, not a production backend.  It keeps
GROND's normal row contract, basis ordering, Sg quadrature, and dense Hamiltonian
construction, then replaces only the propagation matrix by a reduced strong-beam
effective Hamiltonian.  The acceptance target is rowwise q = Idyn/Iborn parity
against the dense calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256


def safe_inverse(values: np.ndarray, eps: float) -> np.ndarray:
    """Return a stable reciprocal without hiding near-resonant denominators."""

    values = np.asarray(values, dtype=np.complex128)
    out = np.empty_like(values)
    mag = np.abs(values)
    ok = mag >= eps
    out[ok] = 1.0 / values[ok]
    phase = np.ones(np.count_nonzero(~ok), dtype=np.complex128)
    if np.any(~ok):
        small = values[~ok]
        nonzero = np.abs(small) > 0.0
        phase[nonzero] = small[nonzero] / np.abs(small[nonzero])
        out[~ok] = phase / eps
    return out


def bethe_effective_hamiltonian(
    H: np.ndarray,
    strong_count: int,
    variant: str,
    eps: float,
    promote_denom: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """Return a reduced strong-beam Hamiltonian and near-singular denominator count."""

    H = np.asarray(H, dtype=np.complex128)
    n = int(H.shape[0])
    m = max(2, min(int(strong_count), n))
    if m >= n:
        return H.copy(), 0, n
    if promote_denom > 0.0 and variant in {"schur0", "schur_target", "schur_mean"}:
        strong_diag0 = np.diag(H[:m, :m]).astype(np.complex128)
        weak_diag0 = np.diag(H[m:, m:]).astype(np.complex128)
        if variant == "schur0":
            energy0 = 0.0 + 0.0j
        elif variant == "schur_target":
            energy0 = H[1, 1]
        else:
            energy0 = np.mean(strong_diag0)
        promote = np.abs(energy0 - weak_diag0) < float(promote_denom)
        if np.any(promote):
            weak_indices = np.arange(m, n)
            promoted = weak_indices[promote]
            remaining = weak_indices[~promote]
            perm = np.concatenate([np.arange(m), promoted, remaining])
            H = H[np.ix_(perm, perm)]
            m = int(m + len(promoted))
            if m >= n:
                return H.copy(), 0, m
    Hss = H[:m, :m].copy()
    Hsw = H[:m, m:]
    Hws = H[m:, :m]
    weak_diag = np.diag(H[m:, m:]).astype(np.complex128)
    strong_diag = np.diag(Hss).astype(np.complex128)
    near = 0

    if variant == "truncate":
        return Hss, 0, m

    if variant in {"schur0", "schur_target", "schur_mean"}:
        if variant == "schur0":
            energy = 0.0 + 0.0j
        elif variant == "schur_target":
            energy = Hss[1, 1]
        else:
            energy = np.mean(strong_diag)
        denom = energy - weak_diag
        near = int(np.count_nonzero(np.abs(denom) < eps))
        return Hss + (Hsw * safe_inverse(denom, eps)[None, :]) @ Hws, near, m

    if variant == "sym":
        correction = np.zeros_like(Hss)
        for i in range(m):
            for j in range(m):
                denom_i = strong_diag[i] - weak_diag
                denom_j = strong_diag[j] - weak_diag
                near += int(np.count_nonzero(np.abs(denom_i) < eps))
                near += int(np.count_nonzero(np.abs(denom_j) < eps))
                inv = 0.5 * (safe_inverse(denom_i, eps) + safe_inverse(denom_j, eps))
                correction[i, j] = np.sum(Hsw[i, :] * Hws[:, j] * inv)
        return Hss + correction, near, m

    raise ValueError(f"unknown Bethe variant {variant!r}")


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    aa = a[mask] - float(np.mean(a[mask]))
    bb = b[mask] - float(np.mean(b[mask]))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 0.0 else float("nan")


def summarize_errors(dense_q: np.ndarray, trial_q: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(dense_q) & np.isfinite(trial_q) & (dense_q > 0.0) & (trial_q > 0.0)
    if not np.any(mask):
        return {
            "valid_q_rows": 0,
            "median_abs_logq_error": float("nan"),
            "p90_abs_logq_error": float("nan"),
            "p99_abs_logq_error": float("nan"),
            "max_abs_logq_error": float("nan"),
            "median_frac_q_error": float("nan"),
            "p90_frac_q_error": float("nan"),
            "q_corr": float("nan"),
        }
    logerr = np.abs(np.log(np.asarray(trial_q[mask], dtype=float) / np.asarray(dense_q[mask], dtype=float)))
    frac = np.abs(np.asarray(trial_q[mask], dtype=float) / np.asarray(dense_q[mask], dtype=float) - 1.0)
    return {
        "valid_q_rows": int(np.count_nonzero(mask)),
        "median_abs_logq_error": float(np.median(logerr)),
        "p90_abs_logq_error": float(np.percentile(logerr, 90)),
        "p99_abs_logq_error": float(np.percentile(logerr, 99)),
        "max_abs_logq_error": float(np.max(logerr)),
        "median_frac_q_error": float(np.median(frac)),
        "p90_frac_q_error": float(np.percentile(frac, 90)),
        "q_corr": finite_corr(np.asarray(dense_q[mask], dtype=float), np.asarray(trial_q[mask], dtype=float)),
    }


def selected_indices(data: XDSData, min_peak: float, max_resolution: float, limit: int, top_isigma: int) -> np.ndarray:
    mask = np.ones(len(data.iobs), dtype=bool)
    if min_peak > 0:
        mask &= data.peak >= float(min_peak)
    if max_resolution > 0:
        dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl])
        mask &= dvals >= float(max_resolution)
    indices = np.where(mask)[0]
    if top_isigma > 0 and len(indices) > top_isigma:
        isigi = data.iobs[indices] / np.maximum(data.sigma[indices], 1e-12)
        indices = indices[np.argsort(isigi)[::-1][: int(top_isigma)]]
    if limit > 0:
        indices = indices[: int(limit)]
    return np.asarray(indices, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-basis", type=int, default=128)
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--max-support-sg", type=float, default=0.05)
    ap.add_argument("--topup-d-min", type=float, default=0.7)
    ap.add_argument("--max-resolution", type=float, default=0.7)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--top-isigma", type=int, default=0)
    ap.add_argument("--thickness", type=float, default=1200.0)
    ap.add_argument("--thickness-model", choices=["single", "uniform_gradient"], default="uniform_gradient")
    ap.add_argument("--thickness-grad-frac", type=float, default=0.4)
    ap.add_argument("--thickness-points", type=int, default=3)
    ap.add_argument("--strong-counts", default="4,6,8,12,16,24,32,48,64")
    ap.add_argument("--variants", default="truncate,schur0,schur_target,schur_mean,sym")
    ap.add_argument("--denom-eps", type=float, default=1e-5)
    ap.add_argument(
        "--promote-denom",
        type=float,
        default=0.0,
        help="Promote near-resonant weak beams into the strong block when |E-Hww| is below this threshold.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    model = parse_model(Path(args.res))
    cfg = Config(
        n_basis=args.n_basis,
        n_points=args.n_points,
        max_support_sg=args.max_support_sg,
        topup_d_min=args.topup_d_min,
        thickness_A=args.thickness,
        thickness_model=args.thickness_model,
        thickness_grad_frac=args.thickness_grad_frac,
        thickness_points=args.thickness_points,
        compute_backend="scipy",
    )
    port = XDSBlochPort(data, model, cfg, cache_dir=out / "cache", res_hash=sha256(Path(args.res)))
    indices = selected_indices(data, args.min_peak, args.max_resolution, args.limit, args.top_isigma)
    strong_counts = [int(v) for v in args.strong_counts.replace(";", ",").split(",") if v.strip()]
    variants = [v.strip() for v in args.variants.replace(";", ",").split(",") if v.strip()]

    dense_q = np.zeros(len(indices), dtype=float)
    dense_dyn = np.zeros(len(indices), dtype=float)
    born = np.zeros(len(indices), dtype=float)
    basis_size = np.zeros(len(indices), dtype=int)
    matrices = 0
    dense_prop_seconds = 0.0
    bethe = {
        (m, variant): {
            "dyn": np.zeros(len(indices), dtype=float),
            "prop_seconds": 0.0,
            "near_denominators": 0,
            "matrix_count": 0,
            "actual_strong_counts": [],
        }
        for m in strong_counts
        for variant in variants
    }

    wall0 = time.perf_counter()
    for row_pos, row_idx in enumerate(indices):
        plan = port.row_prediction_plan(int(row_idx))
        profile_weights = np.asarray(plan["_profile_weights"], dtype=float)
        born_vals = np.asarray(plan["_born_vals"], dtype=float)
        basis_size[row_pos] = int(plan["basis_size"])
        born[row_pos] = float(np.sum(profile_weights * born_vals))
        dense_profile = []
        trial_profile = {key: [] for key in bethe}
        for H in plan["_dyn_matrices"]:
            matrices += 1
            t0 = time.perf_counter()
            dense_profile.append(port.propagate_direct_to_target_thickness_average(H))
            dense_prop_seconds += time.perf_counter() - t0
            for key in bethe:
                strong_count, variant = key
                t1 = time.perf_counter()
                Heff, near, actual_m = bethe_effective_hamiltonian(
                    H,
                    strong_count,
                    variant,
                    args.denom_eps,
                    promote_denom=args.promote_denom,
                )
                value = port.propagate_direct_to_target_thickness_average(Heff)
                bethe[key]["prop_seconds"] += time.perf_counter() - t1
                bethe[key]["near_denominators"] += near
                bethe[key]["matrix_count"] += 1
                bethe[key]["actual_strong_counts"].append(actual_m)
                trial_profile[key].append(value)
        dense_dyn[row_pos] = float(np.sum(profile_weights * np.asarray(dense_profile, dtype=float)))
        dense_q[row_pos] = dense_dyn[row_pos] / max(born[row_pos], 1e-30)
        for key, values in trial_profile.items():
            bethe[key]["dyn"][row_pos] = float(np.sum(profile_weights * np.asarray(values, dtype=float)))
        if (row_pos + 1) % 25 == 0 or row_pos + 1 == len(indices):
            elapsed = time.perf_counter() - wall0
            rate = (row_pos + 1) / max(elapsed, 1e-12)
            print(f"processed {row_pos + 1}/{len(indices)} rows | rate={rate:.2f}/s", flush=True)

    summary_rows = []
    for key, rec in bethe.items():
        strong_count, variant = key
        trial_q = rec["dyn"] / np.maximum(born, 1e-30)
        err = summarize_errors(dense_q, trial_q)
        prop = float(rec["prop_seconds"])
        summary_rows.append(
            {
                "strong_count": int(strong_count),
                "variant": variant,
                "rows": int(len(indices)),
                "matrices": int(rec["matrix_count"]),
                "dense_prop_seconds": float(dense_prop_seconds),
                "bethe_prop_seconds": prop,
                "prop_speedup": float(dense_prop_seconds / max(prop, 1e-12)),
                "near_denominators": int(rec["near_denominators"]),
                "actual_strong_median": float(np.median(rec["actual_strong_counts"]))
                if rec["actual_strong_counts"]
                else float(strong_count),
                "actual_strong_max": int(np.max(rec["actual_strong_counts"]))
                if rec["actual_strong_counts"]
                else int(strong_count),
                "basis_size_median": float(np.median(basis_size)) if len(basis_size) else 0.0,
                "basis_size_max": int(np.max(basis_size)) if len(basis_size) else 0,
                **err,
            }
        )
    summary_rows.sort(key=lambda r: (r["p90_abs_logq_error"], -r["prop_speedup"]))

    csv_path = out / "bethe_split_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = out / "bethe_split_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "hkl": str(Path(args.hkl).resolve()),
                    "res": str(Path(args.res).resolve()),
                    "hkl_transform": args.hkl_transform,
                },
                "config": vars(args),
                "rows": int(len(indices)),
                "matrices": int(matrices),
                "wall_seconds": float(time.perf_counter() - wall0),
                "dense_prop_seconds": float(dense_prop_seconds),
                "basis_size": {
                    "median": float(np.median(basis_size)) if len(basis_size) else 0.0,
                    "max": int(np.max(basis_size)) if len(basis_size) else 0,
                },
                "best_by_p90_logq": summary_rows[:10],
                "summary_csv": str(csv_path),
            },
            indent=2,
            allow_nan=True,
        )
        + "\n"
    )
    print(json.dumps(json.loads(json_path.read_text())["best_by_p90_logq"][:5], indent=2, allow_nan=True))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
