#!/usr/bin/env python3
"""Post-scale additive background-bias audit for native dynamic row tables.

The Bloch predictor is kept frozen.  This command reads a `native_dyn_calc.csv`
table, fits an additive residual model on work rows,

    Iobs = Icalc_scaled + B(features) + eps,

and reports whether held-out weak/high-resolution residual medians improve
without making strong held-out reflections worse.  It is deliberately a
post-calculation audit, not a default production correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from native_row_regularize import (
    DEFAULT_HKL,
    DEFAULT_RES,
    fit_frame_detector_scales,
    fit_group_scales,
    native_metrics,
    parse_float_grid,
    parse_wght,
    shelx_weights,
)
from bloch_engine import XDSData, sha256


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows read from {path}")
    return rows


def fcol(rows: list[dict[str, str]], name: str, default: float = float("nan")) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(name, default)))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=float)


def icol(rows: list[dict[str, str]], name: str, default: int = -1) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            vals.append(int(float(r.get(name, default))))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=int)


def scol(rows: list[dict[str, str]], name: str, default: str = "") -> np.ndarray:
    return np.asarray([str(r.get(name, default)) for r in rows])


@dataclass
class FeatureMatrix:
    X: np.ndarray
    names: list[str]
    penalties: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def add_feature(cols: list[np.ndarray], names: list[str], penalties: list[float], name: str, values, penalty: float = 1.0) -> None:
    v = np.asarray(values, dtype=float)
    cols.append(v)
    names.append(name)
    penalties.append(float(penalty))


def gaussian_rbf(values: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    width = max(float(width), 1e-6)
    return np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2)


def build_features(rows: list[dict[str, str]], hkl_path: Path, train_mask: np.ndarray, frame_basis: int, xy_basis: int) -> FeatureMatrix:
    data = XDSData(hkl_path)
    row_index = icol(rows, "row_index")
    n = len(rows)
    valid_idx = (row_index >= 0) & (row_index < len(data.hkl))

    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    z = np.full(n, np.nan)
    xobs = np.full(n, np.nan)
    yobs = np.full(n, np.nan)
    zobs = np.full(n, np.nan)
    rlp = fcol(rows, "rlp")
    peak = fcol(rows, "peak")
    corr = fcol(rows, "corr")
    d = fcol(rows, "d_A")
    if np.any(valid_idx):
        idx = row_index[valid_idx]
        x[valid_idx] = data.xcal[idx]
        y[valid_idx] = data.ycal[idx]
        z[valid_idx] = data.zcal[idx]
        xobs[valid_idx] = data.xobs[idx]
        yobs[valid_idx] = data.yobs[idx]
        zobs[valid_idx] = data.zobs[idx]
        if not np.any(np.isfinite(rlp)):
            rlp[valid_idx] = data.rlp[idx]
        if not np.any(np.isfinite(peak)):
            peak[valid_idx] = data.peak[idx]
        if not np.any(np.isfinite(corr)):
            corr[valid_idx] = data.corr[idx]

    cols: list[np.ndarray] = []
    names: list[str] = []
    penalties: list[float] = []

    add_feature(cols, names, penalties, "intercept", np.ones(n), 0.0)
    inv_d2 = 1.0 / np.maximum(d, 1e-12) ** 2
    add_feature(cols, names, penalties, "inv_d2", inv_d2, 1.0)
    add_feature(cols, names, penalties, "log_rlp", np.log(np.maximum(rlp, 1e-8)), 1.0)
    add_feature(cols, names, penalties, "peak_defect", 1.0 - np.clip(peak, 0.0, 100.0) / 100.0, 1.0)
    add_feature(cols, names, penalties, "corr_defect", 1.0 - np.clip(corr, 0.0, 100.0) / 100.0, 1.0)
    add_feature(cols, names, penalties, "zobs_minus_zcal", zobs - z, 1.0)
    add_feature(cols, names, penalties, "xyobs_minus_xycal", np.hypot(xobs - x, yobs - y), 1.0)

    x0 = np.nanmedian(x[train_mask & np.isfinite(x)])
    y0 = np.nanmedian(y[train_mask & np.isfinite(y)])
    xn = (x - x0) / 512.0
    yn = (y - y0) / 512.0
    rn = np.hypot(xn, yn)
    add_feature(cols, names, penalties, "x_norm", xn, 1.0)
    add_feature(cols, names, penalties, "y_norm", yn, 1.0)
    add_feature(cols, names, penalties, "det_radius_norm", rn, 1.0)
    add_feature(cols, names, penalties, "x2_norm", xn * xn, 2.0)
    add_feature(cols, names, penalties, "y2_norm", yn * yn, 2.0)
    add_feature(cols, names, penalties, "xy_norm", xn * yn, 2.0)

    zfinite = z[train_mask & np.isfinite(z)]
    if len(zfinite):
        zmin, zmax = float(np.nanmin(zfinite)), float(np.nanmax(zfinite))
        zspan = max(zmax - zmin, 1.0)
        zn = (z - zmin) / zspan
        add_feature(cols, names, penalties, "frame_norm", zn, 1.0)
        add_feature(cols, names, penalties, "frame_norm2", zn * zn, 2.0)
        add_feature(cols, names, penalties, "frame_norm3", zn * zn * zn, 3.0)
        if frame_basis > 0:
            centers = np.linspace(zmin, zmax, int(frame_basis))
            width = max((zmax - zmin) / max(int(frame_basis) - 1, 1), 1.0)
            rbfs = gaussian_rbf(z, centers, width)
            for j in range(rbfs.shape[1]):
                add_feature(cols, names, penalties, f"frame_rbf_{j:02d}", rbfs[:, j], 8.0)

    if xy_basis > 1 and np.any(train_mask & np.isfinite(x) & np.isfinite(y)):
        xq = np.linspace(np.nanpercentile(x[train_mask], 10), np.nanpercentile(x[train_mask], 90), int(xy_basis))
        yq = np.linspace(np.nanpercentile(y[train_mask], 10), np.nanpercentile(y[train_mask], 90), int(xy_basis))
        width = max(float(np.nanmedian(np.diff(xq))) if len(xq) > 1 else 256.0, 64.0)
        for ix, cx in enumerate(xq):
            for iy, cy in enumerate(yq):
                vals = np.exp(-0.5 * (((x - cx) / width) ** 2 + ((y - cy) / width) ** 2))
                add_feature(cols, names, penalties, f"xy_rbf_{ix:02d}_{iy:02d}", vals, 12.0)

    raw = np.vstack(cols).T
    raw[~np.isfinite(raw)] = 0.0
    means = np.zeros(raw.shape[1], dtype=float)
    stds = np.ones(raw.shape[1], dtype=float)
    keep = np.ones(raw.shape[1], dtype=bool)
    for j, name in enumerate(names):
        if name == "intercept":
            continue
        vals = raw[train_mask, j]
        means[j] = float(np.mean(vals))
        stds[j] = float(np.std(vals))
        if stds[j] < 1e-10:
            keep[j] = False
            continue
        raw[:, j] = (raw[:, j] - means[j]) / stds[j]

    keep[0] = True
    return FeatureMatrix(raw[:, keep], [n for n, k in zip(names, keep) if k], np.asarray([p for p, k in zip(penalties, keep) if k], dtype=float), means[keep], stds[keep])


def fit_weighted_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, fit_mask: np.ndarray, penalties: np.ndarray, alpha_factor: float) -> tuple[np.ndarray, float]:
    m = fit_mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(m) <= X.shape[1]:
        raise ValueError(f"not enough fit rows ({np.count_nonzero(m)}) for {X.shape[1]} features")
    sw = np.sqrt(weights[m])
    Xw = X[m] * sw[:, None]
    yw = y[m] * sw
    A = Xw.T @ Xw
    rhs = Xw.T @ yw
    diag = np.diag(A)
    penalized = penalties > 0
    base = float(np.median(diag[penalized & np.isfinite(diag) & (diag > 0)])) if np.any(penalized) else 1.0
    if not np.isfinite(base) or base <= 0:
        base = 1.0
    ridge = float(alpha_factor) * base * np.diag(penalties)
    beta = np.linalg.solve(A + ridge, rhs)
    return beta, float(alpha_factor) * base


def metric_block(obs, sig, pred, mask, a, b) -> dict:
    m = np.asarray(mask, dtype=bool) & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred)
    out = native_metrics(obs, sig, pred, m, a, b, n_params=0)
    if np.count_nonzero(m):
        res_sig = (obs[m] - pred[m]) / np.maximum(sig[m], 1e-12)
        out["median_signed_residual_over_sigma"] = float(np.nanmedian(res_sig))
        out["median_abs_residual_over_sigma"] = float(np.nanmedian(np.abs(res_sig)))
        out["median_residual"] = float(np.nanmedian(obs[m] - pred[m]))
    else:
        out["median_signed_residual_over_sigma"] = float("nan")
        out["median_abs_residual_over_sigma"] = float("nan")
        out["median_residual"] = float("nan")
    return out


def choose_masks(obs, sig, d, split: np.ndarray, holdout_name: str) -> dict[str, np.ndarray]:
    valid = np.isfinite(obs) & np.isfinite(sig) & (sig > 0) & np.isfinite(d)
    isigi = obs / np.maximum(sig, 1e-12)
    inv_d2 = 1.0 / np.maximum(d, 1e-12) ** 2
    q25 = float(np.nanpercentile(isigi[valid], 25))
    q75 = float(np.nanpercentile(isigi[valid], 75))
    r75 = float(np.nanpercentile(inv_d2[valid], 75))
    free = valid & (split == holdout_name)
    work = valid & ~free
    if np.count_nonzero(free) == 0:
        row_ids = np.arange(len(obs))
        free = valid & ((row_ids % 10) == 0)
        work = valid & ~free
    return {
        "all": valid,
        "work": work,
        "free": free,
        "free_weak": free & (isigi <= q25),
        "free_highres": free & (inv_d2 >= r75),
        "free_weak_highres": free & (isigi <= q25) & (inv_d2 >= r75),
        "free_strong": free & (isigi >= q75),
        "free_strong_highres": free & (isigi >= q75) & (inv_d2 >= r75),
    }


def selection_score(blocks: dict[str, dict], base_blocks: dict[str, dict]) -> float:
    wh = blocks["free_weak_highres"]
    if wh["n"] < 5:
        wh = blocks["free_highres"] if blocks["free_highres"]["n"] >= 5 else blocks["free_weak"]
    strong = blocks["free_strong"]
    base_strong = base_blocks["free_strong"]
    strong_worse = max(0.0, strong["median_abs_residual_over_sigma"] - base_strong["median_abs_residual_over_sigma"])
    return (
        abs(wh["median_signed_residual_over_sigma"])
        + 0.25 * wh["median_abs_residual_over_sigma"]
        + 2.0 * strong_worse
        + 0.2 * blocks["free"]["wR2_all"]
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(x: float) -> float:
    return 100.0 * float(x) if np.isfinite(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pred-column", default="dyn_scaled")
    ap.add_argument("--raw-pred-column", default="", help="Optional raw predictor to rescale before additive-bias fitting, e.g. dyn_raw.")
    ap.add_argument("--scale-mode", choices=["fixed", "group", "frame_detector"], default="fixed")
    ap.add_argument("--scale-fit-split", choices=["work", "all"], default="work")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--detector-bin-column", default="detector_bin")
    ap.add_argument("--obs-column", default="target_I")
    ap.add_argument("--sigma-column", default="sigma_eff")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--holdout-name", default="free")
    ap.add_argument("--wght-a", type=float, default=None, help="Audit WGHT a value. Defaults to the RES file WGHT value.")
    ap.add_argument("--wght-b", type=float, default=None, help="Audit WGHT b value. Defaults to the RES file WGHT value.")
    ap.add_argument("--alpha-grid", default="0,0.0001,0.001,0.01,0.1,1,10,100")
    ap.add_argument("--frame-basis", type=int, default=8)
    ap.add_argument("--xy-basis", type=int, default=4)
    ap.add_argument("--bias-clip-sigma", type=float, default=float("nan"), help="Optional absolute bias clip in sigma units. Default no clip.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_path = Path(args.row_table)
    rows = read_rows(row_path)
    obs = fcol(rows, args.obs_column)
    sig = np.maximum(fcol(rows, args.sigma_column), 1e-12)
    pred = fcol(rows, args.pred_column)
    d = fcol(rows, "d_A")
    split = scol(rows, args.split_column, "")
    masks = choose_masks(obs, sig, d, split, args.holdout_name)
    train = masks["work"] & np.isfinite(pred)
    scale_metadata = {"mode": "fixed", "source_column": args.pred_column}
    source_wght = parse_wght(Path(args.res))
    if args.wght_a is None:
        args.wght_a = source_wght[0]
    if args.wght_b is None:
        args.wght_b = source_wght[1]

    if args.raw_pred_column:
        raw = fcol(rows, args.raw_pred_column)
        groups = icol(rows, args.group_column)
        det_bins = icol(rows, args.detector_bin_column)
        scale_fit = masks["all"] if args.scale_fit_split == "all" else masks["work"]
        if args.scale_mode == "group":
            pred, group_scales = fit_group_scales(obs, sig, raw, groups, scale_fit, args.wght_a, args.wght_b)
            scale_metadata = {"mode": "group", "source_column": args.raw_pred_column, "fit_split": args.scale_fit_split, "n_group_scales": len(group_scales)}
        elif args.scale_mode == "frame_detector":
            pred, group_scales, det_scales = fit_frame_detector_scales(obs, sig, raw, groups, det_bins, scale_fit, args.wght_a, args.wght_b)
            scale_metadata = {
                "mode": "frame_detector",
                "source_column": args.raw_pred_column,
                "fit_split": args.scale_fit_split,
                "n_group_scales": len(group_scales),
                "n_detector_scales": len(det_scales),
            }
        elif args.scale_mode != "fixed":
            raise ValueError(f"unknown scale mode {args.scale_mode!r}")
        train = masks["work"] & np.isfinite(pred)
    Xinfo = build_features(rows, Path(args.hkl), train, args.frame_basis, args.xy_basis)
    residual = obs - pred
    weights = shelx_weights(obs, sig, pred, args.wght_a, args.wght_b)
    alpha_values = parse_float_grid(args.alpha_grid)

    base_blocks = {name: metric_block(obs, sig, pred, mask & np.isfinite(pred), args.wght_a, args.wght_b) for name, mask in masks.items()}
    trials = []
    best = None
    best_rows = None
    best_beta = None
    for alpha in alpha_values:
        beta, ridge_lambda = fit_weighted_ridge(Xinfo.X, residual, weights, train, Xinfo.penalties, alpha)
        bias = Xinfo.X @ beta
        if np.isfinite(args.bias_clip_sigma):
            bias = np.clip(bias, -abs(args.bias_clip_sigma) * sig, abs(args.bias_clip_sigma) * sig)
        corrected = pred + bias
        blocks = {name: metric_block(obs, sig, corrected, mask & np.isfinite(corrected), args.wght_a, args.wght_b) for name, mask in masks.items()}
        score = selection_score(blocks, base_blocks)
        row = {
            "alpha_factor": alpha,
            "ridge_lambda": ridge_lambda,
            "selection_score": score,
            "all_R1": blocks["all"]["R1_all"],
            "all_wR2": blocks["all"]["wR2_all"],
            "free_R1": blocks["free"]["R1_all"],
            "free_wR2": blocks["free"]["wR2_all"],
            "free_weak_highres_n": blocks["free_weak_highres"]["n"],
            "free_weak_highres_median_res_sig": blocks["free_weak_highres"]["median_signed_residual_over_sigma"],
            "free_weak_highres_median_abs_res_sig": blocks["free_weak_highres"]["median_abs_residual_over_sigma"],
            "free_strong_median_abs_res_sig": blocks["free_strong"]["median_abs_residual_over_sigma"],
            "bias_p05": float(np.nanpercentile(bias[np.isfinite(bias)], 5)),
            "bias_median": float(np.nanmedian(bias[np.isfinite(bias)])),
            "bias_p95": float(np.nanpercentile(bias[np.isfinite(bias)], 95)),
        }
        trials.append(row)
        if best is None or score < best["trial"]["selection_score"]:
            best = {"trial": row, "blocks": blocks}
            best_rows = corrected
            best_beta = beta

    write_csv(out / "additive_bias_trials.csv", trials)
    assert best is not None and best_rows is not None and best_beta is not None

    corrected_rows = []
    bias = best_rows - pred
    for r, p, b, c in zip(rows, pred, bias, best_rows):
        rr = dict(r)
        rr["bias_additive"] = f"{float(b):.10g}"
        rr["pred_before_bias"] = f"{float(p):.10g}"
        rr["pred_after_bias"] = f"{float(c):.10g}"
        rr["residual_after_bias"] = f"{float(float(r.get(args.obs_column, 'nan')) - c):.10g}"
        corrected_rows.append(rr)
    write_csv(out / "bias_corrected_rows.csv", corrected_rows)

    coef_rows = [
        {"feature": name, "coefficient": float(coef), "penalty": float(penalty)}
        for name, coef, penalty in zip(Xinfo.names, best_beta, Xinfo.penalties)
    ]
    write_csv(out / "bias_coefficients.csv", coef_rows)

    summary = {
        "inputs": {
            "row_table": str(row_path.resolve()),
            "row_table_sha256": sha256(row_path),
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "res": str(Path(args.res).resolve()),
            "res_sha256": sha256(Path(args.res)),
        },
        "columns": {"obs": args.obs_column, "sigma": args.sigma_column, "prediction": args.pred_column},
        "scale_model": scale_metadata,
        "weights": {"native_a": args.wght_a, "native_b": args.wght_b, "source_res_a": source_wght[0], "source_res_b": source_wght[1]},
        "features": {"n_features": len(Xinfo.names), "names": Xinfo.names, "penalties": [float(x) for x in Xinfo.penalties]},
        "best_trial": best["trial"],
        "baseline_blocks": base_blocks,
        "corrected_blocks": best["blocks"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    wh_base = base_blocks["free_weak_highres"]
    wh_corr = best["blocks"]["free_weak_highres"]
    strong_base = base_blocks["free_strong"]
    strong_corr = best["blocks"]["free_strong"]
    report = [
        "# Additive Background-Bias Audit",
        "",
        "This is a post-scale additive residual audit. The Bloch predictions were not recomputed.",
        "",
        "```text",
        f"row_table: {row_path}",
        f"prediction column: {args.raw_pred_column if args.raw_pred_column else args.pred_column}",
        f"scale mode: {scale_metadata['mode']}",
        f"native weights: a={args.wght_a:g}, b={args.wght_b:g}",
        f"features: {len(Xinfo.names)}",
        f"best alpha_factor: {best['trial']['alpha_factor']:g}",
        f"best ridge_lambda: {best['trial']['ridge_lambda']:.6g}",
        "```",
        "",
        "## Overall Metrics",
        "",
        "| model | R1 all (%) | wR2 all (%) | free R1 (%) | free wR2 (%) |",
        "|---|---:|---:|---:|---:|",
        f"| before additive bias | {pct(base_blocks['all']['R1_all']):.3f} | {pct(base_blocks['all']['wR2_all']):.3f} | {pct(base_blocks['free']['R1_all']):.3f} | {pct(base_blocks['free']['wR2_all']):.3f} |",
        f"| after additive bias | {pct(best['blocks']['all']['R1_all']):.3f} | {pct(best['blocks']['all']['wR2_all']):.3f} | {pct(best['blocks']['free']['R1_all']):.3f} | {pct(best['blocks']['free']['wR2_all']):.3f} |",
        "",
        "## Held-Out Acceptance Signals",
        "",
        "| held-out subset | n | before median residual/sigma | after median residual/sigma | before median abs residual/sigma | after median abs residual/sigma |",
        "|---|---:|---:|---:|---:|---:|",
        f"| weak high-resolution | {wh_corr['n']} | {wh_base['median_signed_residual_over_sigma']:.4f} | {wh_corr['median_signed_residual_over_sigma']:.4f} | {wh_base['median_abs_residual_over_sigma']:.4f} | {wh_corr['median_abs_residual_over_sigma']:.4f} |",
        f"| strong | {strong_corr['n']} | {strong_base['median_signed_residual_over_sigma']:.4f} | {strong_corr['median_signed_residual_over_sigma']:.4f} | {strong_base['median_abs_residual_over_sigma']:.4f} | {strong_corr['median_abs_residual_over_sigma']:.4f} |",
        "",
        "## Verdict",
        "",
    ]
    wh_improved = abs(wh_corr["median_signed_residual_over_sigma"]) < abs(wh_base["median_signed_residual_over_sigma"])
    strong_not_worse = strong_corr["median_abs_residual_over_sigma"] <= strong_base["median_abs_residual_over_sigma"] + 0.05
    if wh_improved and strong_not_worse:
        report.append("The additive model passes the narrow held-out bias test: the weak/high-resolution median residual moves toward zero and strong-reflection held-out scatter is not materially worse. This should be promoted to a larger all-row nuisance screen before it is used for endpoint export.")
    else:
        report.append("The additive model does not pass the narrow held-out bias test. Keep it as a diagnostic; do not promote it into the production correction path without revising the feature set or regularization.")
    report.extend([
        "",
        "Artifacts: `additive_bias_trials.csv`, `bias_corrected_rows.csv`, `bias_coefficients.csv`, and `summary.json`.",
        "",
    ])
    (out / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({"out_dir": str(out), "best": best["trial"], "passes_heldout_bias_gate": bool(wh_improved and strong_not_worse)}, indent=2))


if __name__ == "__main__":
    main()
