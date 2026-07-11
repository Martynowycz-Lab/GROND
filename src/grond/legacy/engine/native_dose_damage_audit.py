#!/usr/bin/env python3
"""Audit whether residuals support a resolution-dependent dose/damage term.

This is a gate for the expensive pre-matrix `dose_b_mode=linear_frame` Bloch
kernel.  It does not change predictions.  It tests whether late-frame,
high-resolution rows are systematically lower than the current model in a way
ordinary frame scales cannot absorb.

For positive rows it fits

    log(Iobs / Icalc) = b0 + bz*z + bs*s2 + bzs*z*s2 + eps

where z is frame offset in XDS frames and s2 = 1/d^2.  If a structure-factor
dose term is needed, bzs should be negative.  In the simple kinematic limit,
B_dose_per_frame is approximately -2*bzs because intensities scale roughly as
exp(-B_dose*z*s2/2).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .bloch_engine import XDSData, sha256
    from .native_row_regularize import DEFAULT_HKL, DEFAULT_RES
else:  # pragma: no cover - direct script execution
    from bloch_engine import XDSData, sha256
    from native_row_regularize import DEFAULT_HKL, DEFAULT_RES


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows read from {path}")
    return rows


def fcol(rows: list[dict[str, str]], name: str, default: float = float("nan")) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, default)))
        except (TypeError, ValueError):
            out.append(default)
    return np.asarray(out, dtype=float)


def icol(rows: list[dict[str, str]], name: str, default: int = -1) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(int(float(r.get(name, default))))
        except (TypeError, ValueError):
            out.append(default)
    return np.asarray(out, dtype=int)


def scol(rows: list[dict[str, str]], name: str, default: str = "") -> np.ndarray:
    return np.asarray([str(r.get(name, default)) for r in rows])


def add_zcal(rows: list[dict[str, str]], hkl_path: Path) -> np.ndarray:
    data = XDSData(hkl_path)
    row_index = icol(rows, "row_index")
    z = np.full(len(rows), np.nan, dtype=float)
    good = (row_index >= 0) & (row_index < len(data.hkl))
    if np.any(good):
        z[good] = data.zcal[row_index[good]]
    return z


def weighted_lstsq(X: np.ndarray, y: np.ndarray, w: np.ndarray, mask: np.ndarray) -> dict:
    m = mask & np.isfinite(y) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(X), axis=1)
    if np.count_nonzero(m) <= X.shape[1] + 2:
        return {"ok": False, "n": int(np.count_nonzero(m))}
    sw = np.sqrt(w[m])
    Xw = X[m] * sw[:, None]
    yw = y[m] * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y[m] - X[m] @ beta
    dof = max(1, np.count_nonzero(m) - X.shape[1])
    sigma2 = float(np.sum(w[m] * resid * resid) / dof)
    xtwx = Xw.T @ Xw
    cov = sigma2 * np.linalg.pinv(xtwx)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {"ok": True, "n": int(np.count_nonzero(m)), "beta": beta, "se": se, "sigma2": sigma2, "mask": m}


def quantile_mask(values: np.ndarray, qlo: float, qhi: float) -> np.ndarray:
    finite = np.isfinite(values)
    lo = np.nanpercentile(values[finite], qlo)
    hi = np.nanpercentile(values[finite], qhi)
    return finite & (values >= lo) & (values <= hi)


def bin_summary(name: str, mask: np.ndarray, y: np.ndarray, zres: np.ndarray) -> dict:
    m = mask & np.isfinite(y)
    if np.count_nonzero(m) == 0:
        return {"bin": name, "n": 0}
    return {
        "bin": name,
        "n": int(np.count_nonzero(m)),
        "median_log_obs_calc": float(np.nanmedian(y[m])),
        "mean_log_obs_calc": float(np.nanmean(y[m])),
        "median_residual_over_sigma": float(np.nanmedian(zres[m])) if np.any(np.isfinite(zres[m])) else float("nan"),
        "median_abs_residual_over_sigma": float(np.nanmedian(np.abs(zres[m]))) if np.any(np.isfinite(zres[m])) else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--obs-column", default="target_I")
    ap.add_argument("--pred-column", default="pred_after_bias")
    ap.add_argument("--sigma-column", default="xds_sigma")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--min-i-over-sigma-for-fit", type=float, default=1.0)
    ap.add_argument("--min-positive", type=float, default=1e-6)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_path = Path(args.row_table)
    rows = read_rows(row_path)
    obs = fcol(rows, args.obs_column)
    pred = fcol(rows, args.pred_column)
    sig = np.maximum(fcol(rows, args.sigma_column), 1e-12)
    d = fcol(rows, "d_A")
    split = scol(rows, args.split_column)
    zcal = add_zcal(rows, Path(args.hkl))
    s2 = 1.0 / np.maximum(d, 1e-12) ** 2
    valid = (
        np.isfinite(obs)
        & np.isfinite(pred)
        & np.isfinite(sig)
        & np.isfinite(s2)
        & np.isfinite(zcal)
        & (obs > args.min_positive)
        & (pred > args.min_positive)
        & (obs / sig >= float(args.min_i_over_sigma_for_fit))
    )
    z0 = float(np.nanmin(zcal[valid])) if np.any(valid) else 0.0
    dz = zcal - z0
    y = np.full(len(rows), np.nan, dtype=float)
    y[valid] = np.log(obs[valid] / pred[valid])
    zres = (obs - pred) / sig
    w = np.zeros(len(rows), dtype=float)
    # Approximate log-ratio variance: var(log I) ~= (sigma/I)^2.
    w[valid] = (obs[valid] / sig[valid]) ** 2
    X = np.column_stack([np.ones(len(rows)), dz, s2, dz * s2])
    names = ["intercept", "frame", "inv_d2", "frame_x_inv_d2"]
    free = valid & (split == "free")
    work = valid & ~free
    if np.count_nonzero(free) == 0:
        idx = np.arange(len(rows))
        free = valid & ((idx % 10) == 0)
        work = valid & ~free
    all_fit = weighted_lstsq(X, y, w, valid)
    work_fit = weighted_lstsq(X, y, w, work)
    free_fit = weighted_lstsq(X, y, w, free)

    def fit_record(label: str, fit: dict) -> dict:
        if not fit.get("ok"):
            return {"split": label, "ok": False, "n": fit.get("n", 0)}
        beta = fit["beta"]
        se = fit["se"]
        rec = {"split": label, "ok": True, "n": fit["n"]}
        for i, name in enumerate(names):
            rec[f"beta_{name}"] = float(beta[i])
            rec[f"se_{name}"] = float(se[i])
            rec[f"z_{name}"] = float(beta[i] / se[i]) if se[i] > 0 else float("nan")
        rec["approx_Bdose_per_frame_A2"] = float(-2.0 * beta[3])
        rec["se_Bdose_per_frame_A2"] = float(2.0 * se[3])
        rec["total_frame_span"] = float(np.nanmax(dz[valid]) - np.nanmin(dz[valid]))
        rec["approx_total_Bdose_A2"] = rec["approx_Bdose_per_frame_A2"] * rec["total_frame_span"]
        return rec

    fits = [fit_record("all", all_fit), fit_record("work", work_fit), fit_record("free", free_fit)]
    write_csv(out / "dose_damage_fit.csv", fits)

    early = quantile_mask(dz, 0, 25)
    late = quantile_mask(dz, 75, 100)
    lowres = quantile_mask(s2, 0, 25)
    highres = quantile_mask(s2, 75, 100)
    bins = []
    for split_name, split_mask in [("all", valid), ("work", work), ("free", free)]:
        for frame_name, frame_mask in [("early", early), ("late", late)]:
            for res_name, res_mask in [("lowres", lowres), ("highres", highres)]:
                bins.append(bin_summary(f"{split_name}_{frame_name}_{res_name}", split_mask & frame_mask & res_mask, y, zres))
    write_csv(out / "dose_damage_bins.csv", bins)

    work_ok = work_fit.get("ok", False)
    free_ok = free_fit.get("ok", False)
    work_beta = float(work_fit["beta"][3]) if work_ok else float("nan")
    work_se = float(work_fit["se"][3]) if work_ok else float("nan")
    free_beta = float(free_fit["beta"][3]) if free_ok else float("nan")
    work_z = work_beta / work_se if np.isfinite(work_se) and work_se > 0 else float("nan")
    accept = bool(work_ok and work_beta < 0.0 and abs(work_z) >= 2.0 and (not free_ok or free_beta <= 0.0))

    summary = {
        "inputs": {
            "row_table": str(row_path.resolve()),
            "row_table_sha256": sha256(row_path),
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "res": str(Path(args.res).resolve()),
            "res_sha256": sha256(Path(args.res)),
        },
        "columns": {"obs": args.obs_column, "prediction": args.pred_column, "sigma": args.sigma_column},
        "fit_rows": {"valid": int(np.count_nonzero(valid)), "work": int(np.count_nonzero(work)), "free": int(np.count_nonzero(free)), "min_i_over_sigma": args.min_i_over_sigma_for_fit},
        "fits": fits,
        "accept_dose_b_grid": accept,
        "acceptance_rule": "work beta(frame*inv_d2) < 0 with |z| >= 2 and free beta not alternate sign",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    by = {r["bin"]: r for r in bins}
    report = [
        "# Resolution-Dependent Dose/Damage Audit",
        "",
        "This audit checks whether residuals show the dose slope required before spending compute on a `dose_b_mode=linear_frame` Bloch grid. The prediction is fixed.",
        "",
        "```text",
        f"row_table: {row_path}",
        f"prediction: {args.pred_column}",
        f"sigma: {args.sigma_column}",
        f"fit rows: {np.count_nonzero(valid)}",
        f"minimum I/sigma for slope fit: {args.min_i_over_sigma_for_fit:g}",
        "model: log(Iobs/Icalc) = b0 + bz*frame + bs*(1/d^2) + bzs*frame*(1/d^2)",
        "```",
        "",
        "## Interaction Fit",
        "",
        "| split | n | beta frame*1/d2 | se | z | approx Bdose/frame (A^2) | approx total Bdose (A^2) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in fits:
        if not rec.get("ok"):
            report.append(f"| {rec['split']} | {rec['n']} | NA | NA | NA | NA | NA |")
        else:
            report.append(
                f"| {rec['split']} | {rec['n']} | {rec['beta_frame_x_inv_d2']:.6g} | {rec['se_frame_x_inv_d2']:.6g} | "
                f"{rec['z_frame_x_inv_d2']:.3f} | {rec['approx_Bdose_per_frame_A2']:.6g} | {rec['approx_total_Bdose_A2']:.3f} |"
            )
    report.extend([
        "",
        "## Early/Late Shell Check",
        "",
        "| bin | n | median log(Iobs/Icalc) | median residual/sigma |",
        "|---|---:|---:|---:|",
    ])
    for key in [
        "all_early_lowres",
        "all_early_highres",
        "all_late_lowres",
        "all_late_highres",
        "free_early_lowres",
        "free_early_highres",
        "free_late_lowres",
        "free_late_highres",
    ]:
        r = by.get(key, {"n": 0})
        report.append(f"| {key} | {r.get('n', 0)} | {r.get('median_log_obs_calc', float('nan')):.6g} | {r.get('median_residual_over_sigma', float('nan')):.6g} |")
    report.extend([
        "",
        "## Verdict",
        "",
    ])
    if accept:
        report.append("The residuals support a pre-matrix dose-B grid: the work-set frame-by-resolution interaction is significantly negative and the held-out sign is not contradictory. Run `dose_b_mode=linear_frame` as an explicit Bloch recomputation grid, not as a post-hoc intensity scale.")
    else:
        report.append("The residuals do not pass the dose-slope gate. Do not promote a dose-B grid for this table; it is plausible physics but not supported by this residual pattern.")
    report.append("")
    (out / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({"out_dir": str(out), "accept_dose_b_grid": accept, "work_beta_frame_inv_d2": work_beta, "work_z": work_z, "free_beta_frame_inv_d2": free_beta}, indent=2))


if __name__ == "__main__":
    main()
