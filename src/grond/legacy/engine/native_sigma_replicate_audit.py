#!/usr/bin/env python3
"""Calibrate native row sigmas from equivalent-row scatter.

This audit keeps the Bloch predictions fixed.  It estimates row noise from
scatter inside P222-equivalent reflection groups after subtracting the frozen
dynamic prediction, fits a nonnegative variance model on training groups, and
scores held-out groups with the learned sigma model.

The fitted model has the form

    sigma_cal^2 = sigma_int^2 + sum_j c_j feature_j, c_j >= 0

where the available XDS/native row features are proxies for the requested
foreground/background/crowding terms when raw shoebox quantities are absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .bloch_engine import XDSData, sha256
    from .native_row_regularize import (
        DEFAULT_HKL,
        DEFAULT_RES,
        fit_frame_detector_scales,
        fit_group_scales,
        native_metrics,
        parse_wght,
    )
else:  # pragma: no cover - direct script execution
    from bloch_engine import XDSData, sha256
    from native_row_regularize import (
        DEFAULT_HKL,
        DEFAULT_RES,
        fit_frame_detector_scales,
        fit_group_scales,
        native_metrics,
        parse_wght,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts; raise if it has no rows."""

    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows read from {path}")
    return rows


def fcol(rows: list[dict[str, str]], name: str, default: float = float("nan")) -> np.ndarray:
    """Return one column as a float array, with default for missing or unparsable cells."""

    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(name, default)))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=float)


def icol(rows: list[dict[str, str]], name: str, default: int = -1) -> np.ndarray:
    """Return one column as an int array, with default for missing or unparsable cells."""

    vals = []
    for r in rows:
        try:
            vals.append(int(float(r.get(name, default))))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=int)


def p222_canonical(h: int, k: int, l: int) -> tuple[int, int, int]:
    """Return the smallest of the four P222 rotation mates of (h, k, l); Friedel mates stay separate."""

    # Proper orthorhombic rotations only. Friedel mates are intentionally not merged.
    mates = [(h, k, l), (h, -k, -l), (-h, k, -l), (-h, -k, l)]
    return min(mates)


def stable_group_split(key: tuple[int, int, int], holdout_fraction: float, seed: int) -> bool:
    """Return whether an equivalence group is held out, from a seeded SHA-256 hash of its key."""

    text = f"{seed}:{key[0]}:{key[1]}:{key[2]}".encode()
    val = int(hashlib.sha256(text).hexdigest()[:12], 16) / float(16**12)
    return val < float(holdout_fraction)


def add_xds_metadata(rows: list[dict[str, str]], hkl_path: Path) -> dict[str, np.ndarray]:
    """Return XDS calculated and observed centroids, RLP, PEAK and CORR per row; NaN for invalid row_index."""

    data = XDSData(hkl_path)
    row_index = icol(rows, "row_index")
    n = len(rows)
    good = (row_index >= 0) & (row_index < len(data.hkl))
    out = {name: np.full(n, np.nan, dtype=float) for name in ["xcal", "ycal", "zcal", "xobs", "yobs", "zobs", "rlp_xds", "peak_xds", "corr_xds"]}
    if np.any(good):
        idx = row_index[good]
        out["xcal"][good] = data.xcal[idx]
        out["ycal"][good] = data.ycal[idx]
        out["zcal"][good] = data.zcal[idx]
        out["xobs"][good] = data.xobs[idx]
        out["yobs"][good] = data.yobs[idx]
        out["zobs"][good] = data.zobs[idx]
        out["rlp_xds"][good] = data.rlp[idx]
        out["peak_xds"][good] = data.peak[idx]
        out["corr_xds"][good] = data.corr[idx]
    return out


def build_variance_features(rows: list[dict[str, str]], pred: np.ndarray, hkl_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return the variance-model feature matrix and feature names for each row.

    Features are a constant, |Iobs|, Iobs² and Idyn², and squared intensity-scaled proxies: CORR and PEAK
    defects, centroid offsets, detector radius, RLP ratio and 1/d².
    """

    obs = fcol(rows, "target_I")
    d = fcol(rows, "d_A")
    rlp = fcol(rows, "rlp")
    peak = fcol(rows, "peak")
    corr = fcol(rows, "corr")
    meta = add_xds_metadata(rows, hkl_path)
    if not np.any(np.isfinite(rlp)):
        rlp = meta["rlp_xds"]
    if not np.any(np.isfinite(peak)):
        peak = meta["peak_xds"]
    if not np.any(np.isfinite(corr)):
        corr = meta["corr_xds"]
    abs_i = np.abs(obs)
    pred_pos = np.maximum(pred, 0.0)
    corr_def = np.maximum(0.0, 1.0 - np.clip(corr, 0.0, 100.0) / 100.0)
    peak_def = np.maximum(0.0, 1.0 - np.clip(peak, 0.0, 100.0) / 100.0)
    xy_resid = np.hypot(meta["xobs"] - meta["xcal"], meta["yobs"] - meta["ycal"])
    z_resid = np.abs(meta["zobs"] - meta["zcal"])
    x0 = np.nanmedian(meta["xcal"][np.isfinite(meta["xcal"])])
    y0 = np.nanmedian(meta["ycal"][np.isfinite(meta["ycal"])])
    det_radius = np.hypot(meta["xcal"] - x0, meta["ycal"] - y0) / 512.0
    rlp_med = np.nanmedian(rlp[np.isfinite(rlp) & (rlp > 0)])
    rlp_ratio = rlp / max(float(rlp_med), 1e-12)
    inv_d2 = 1.0 / np.maximum(d, 1e-12) ** 2
    features = [
        ("constant", np.ones(len(rows), dtype=float)),
        ("abs_Iobs", abs_i),
        ("Iobs2", obs * obs),
        ("Idyn2", pred_pos * pred_pos),
        ("corr_defect_I2", (corr_def * abs_i) ** 2),
        ("peak_defect_I2", (peak_def * abs_i) ** 2),
        ("xy_residual_I2", (np.nan_to_num(xy_resid, nan=0.0) * abs_i) ** 2),
        ("z_residual_I2", (np.nan_to_num(z_resid, nan=0.0) * abs_i) ** 2),
        ("det_radius_I2", (np.nan_to_num(det_radius, nan=0.0) * abs_i) ** 2),
        ("rlp_ratio_I2", (np.nan_to_num(rlp_ratio, nan=1.0) * abs_i) ** 2),
        ("inv_d2_I2", (np.nan_to_num(inv_d2, nan=0.0) * abs_i) ** 2),
    ]
    X = np.vstack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for _, v in features]).T
    return X, [name for name, _ in features]


def fit_nonnegative_variance(X: np.ndarray, y: np.ndarray, mask: np.ndarray, ridge_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit nonnegative variance coefficients to scatter targets on mask; return (coefficients, scales).

    Columns are scaled by their median absolute value and optionally ridge-augmented; if NNLS fails, a
    clipped least-squares solution is used.
    """

    m = mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if np.count_nonzero(m) < 10:
        raise ValueError("not enough training rows for variance fit")
    Xfit = X[m].astype(float)
    yfit = np.maximum(y[m].astype(float), 0.0)
    scales = np.nanmedian(np.abs(Xfit), axis=0)
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    Xs = Xfit / scales[None, :]
    if ridge_alpha > 0:
        Xs = np.vstack([Xs, math.sqrt(float(ridge_alpha)) * np.eye(Xs.shape[1])])
        yfit = np.concatenate([yfit, np.zeros(Xs.shape[1], dtype=float)])
    try:
        from scipy.optimize import nnls

        beta_scaled, _ = nnls(Xs, yfit)
    except Exception:
        beta_scaled = np.linalg.lstsq(Xs, yfit, rcond=None)[0]
        beta_scaled = np.maximum(beta_scaled, 0.0)
    coeff = beta_scaled / scales
    return coeff, scales


def group_scatter_targets(obs: np.ndarray, pred: np.ndarray, hkls: np.ndarray, min_multiplicity: int, holdout_fraction: float, seed: int):
    """Return per-row replicate scatter around each P222 group mean, with train and test group masks.

    Scatter is the squared deviation of the residual from its group mean times n/(n-1). Groups with fewer
    than min_multiplicity finite rows are unused; held-out groups come from stable_group_split.
    """

    keys = [p222_canonical(int(h), int(k), int(l)) for h, k, l in hkls]
    by_key: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(keys):
        by_key.setdefault(key, []).append(i)
    scatter = np.full(len(obs), np.nan, dtype=float)
    mult = np.ones(len(obs), dtype=int)
    holdout = np.zeros(len(obs), dtype=bool)
    usable = np.zeros(len(obs), dtype=bool)
    residual = obs - pred
    usable_groups = 0
    holdout_groups = 0
    for key, idxs in by_key.items():
        idx = np.asarray(idxs, dtype=int)
        finite = idx[np.isfinite(residual[idx])]
        mult[idx] = len(finite)
        if len(finite) < int(min_multiplicity):
            continue
        vals = residual[finite]
        center = float(np.mean(vals))
        local = (vals - center) ** 2 * len(vals) / max(len(vals) - 1, 1)
        scatter[finite] = local
        usable[finite] = True
        usable_groups += 1
        if stable_group_split(key, holdout_fraction, seed):
            holdout[finite] = True
            holdout_groups += 1
    train = usable & ~holdout
    test = usable & holdout
    return {
        "keys": keys,
        "scatter": scatter,
        "multiplicity": mult,
        "usable": usable,
        "train": train,
        "test": test,
        "n_groups": len(by_key),
        "n_usable_groups": usable_groups,
        "n_holdout_groups": holdout_groups,
    }


def metric_summary(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """Return native metrics (unit weights) plus z-score summaries of (Iobs - Icalc)/sigma in mask."""

    out = native_metrics(obs, sig, pred, mask, 0.0, 0.0, n_params=0)
    m = mask & np.isfinite(obs) & np.isfinite(sig) & (sig > 0) & np.isfinite(pred)
    if np.count_nonzero(m):
        z = (obs[m] - pred[m]) / np.maximum(sig[m], 1e-12)
        out.update(
            {
                "z_median": float(np.nanmedian(z)),
                "z_mad": float(np.nanmedian(np.abs(z - np.nanmedian(z)))),
                "z_rms": float(np.sqrt(np.nanmean(z * z))),
                "z_abs_median": float(np.nanmedian(np.abs(z))),
                "z_abs_p95": float(np.nanpercentile(np.abs(z), 95)),
                "frac_abs_z_gt_2": float(np.mean(np.abs(z) > 2.0)),
                "frac_abs_z_gt_3": float(np.mean(np.abs(z) > 3.0)),
            }
        )
    else:
        out.update({"z_median": float("nan"), "z_mad": float("nan"), "z_rms": float("nan"), "z_abs_median": float("nan"), "z_abs_p95": float("nan"), "frac_abs_z_gt_2": float("nan"), "frac_abs_z_gt_3": float("nan")})
    return out


def choose_masks(obs: np.ndarray, sig: np.ndarray, d: np.ndarray, scatter_masks: dict) -> dict[str, np.ndarray]:
    """Return the all, scatter train and test masks and the test weak, strong and high-resolution subsets.

    Weak and strong are the lower and upper I/sigma quartiles; high resolution is the top 1/d² quartile.
    """

    valid = np.isfinite(obs) & np.isfinite(sig) & (sig > 0) & np.isfinite(d)
    isigi = obs / np.maximum(sig, 1e-12)
    inv_d2 = 1.0 / np.maximum(d, 1e-12) ** 2
    q25 = float(np.nanpercentile(isigi[valid], 25))
    q75 = float(np.nanpercentile(isigi[valid], 75))
    r75 = float(np.nanpercentile(inv_d2[valid], 75))
    test = scatter_masks["test"]
    train = scatter_masks["train"]
    return {
        "all": valid,
        "scatter_train": train,
        "scatter_test": test,
        "test_weak": test & (isigi <= q25),
        "test_highres": test & (inv_d2 >= r75),
        "test_weak_highres": test & (isigi <= q25) & (inv_d2 >= r75),
        "test_strong": test & (isigi >= q75),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file with the first row's keys as header; an empty list writes an empty file."""

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(x: float) -> float:
    """Return a fraction as a percentage, or NaN if it is not finite."""

    return 100.0 * float(x) if np.isfinite(x) else float("nan")


def main() -> None:
    """Fit the replicate-scatter sigma model on training groups and score it on held-out groups.

    Writes the coefficient and calibrated-row CSVs, summary.json and REPORT.md to --out-dir.
    """

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--obs-column", default="target_I")
    ap.add_argument("--base-sigma-column", default="xds_sigma")
    ap.add_argument("--pred-column", default="dyn_scaled")
    ap.add_argument("--raw-pred-column", default="", help="Optional raw predictor to rescale first, e.g. dyn_raw.")
    ap.add_argument("--scale-mode", choices=["fixed", "group", "frame_detector"], default="fixed")
    ap.add_argument("--scale-fit-split", choices=["all", "scatter_train"], default="all")
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--detector-bin-column", default="detector_bin")
    ap.add_argument("--min-multiplicity", type=int, default=2)
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=137)
    ap.add_argument("--ridge-alpha", type=float, default=1e-4)
    ap.add_argument("--min-sigma-scale", type=float, default=0.25)
    ap.add_argument("--base-scale-mode", choices=["fixed_additive", "fit_base_scale"], default="fixed_additive", help="fixed_additive keeps sigma_int^2 fixed and only adds variance. fit_base_scale learns a nonnegative coefficient on sigma_int^2 from replicate scatter.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_path = Path(args.row_table)
    rows = read_rows(row_path)
    obs = fcol(rows, args.obs_column)
    base_sig = np.maximum(fcol(rows, args.base_sigma_column), 1e-12)
    pred = fcol(rows, args.pred_column)
    hkls = np.asarray([[int(float(r["h"])), int(float(r["k"])), int(float(r["l"]))] for r in rows], dtype=int)
    d = fcol(rows, "d_A")
    scale_metadata = {"mode": "fixed", "source_column": args.pred_column}
    if args.raw_pred_column:
        raw = fcol(rows, args.raw_pred_column)
        groups = icol(rows, args.group_column)
        det_bins = icol(rows, args.detector_bin_column)
        preliminary_masks = group_scatter_targets(obs, pred=np.nan_to_num(raw, nan=0.0), hkls=hkls, min_multiplicity=args.min_multiplicity, holdout_fraction=args.holdout_fraction, seed=args.split_seed)
        fit_mask = preliminary_masks["train"] if args.scale_fit_split == "scatter_train" else np.isfinite(obs) & np.isfinite(base_sig) & np.isfinite(raw)
        if args.scale_mode == "group":
            pred, group_scales = fit_group_scales(obs, base_sig, raw, groups, fit_mask, 0.0, 0.0)
            scale_metadata = {"mode": "group", "source_column": args.raw_pred_column, "fit_split": args.scale_fit_split, "n_group_scales": len(group_scales)}
        elif args.scale_mode == "frame_detector":
            pred, group_scales, det_scales = fit_frame_detector_scales(obs, base_sig, raw, groups, det_bins, fit_mask, 0.0, 0.0)
            scale_metadata = {"mode": "frame_detector", "source_column": args.raw_pred_column, "fit_split": args.scale_fit_split, "n_group_scales": len(group_scales), "n_detector_scales": len(det_scales)}
        elif args.scale_mode != "fixed":
            raise ValueError(args.scale_mode)

    scatter_info = group_scatter_targets(obs, pred, hkls, args.min_multiplicity, args.holdout_fraction, args.split_seed)
    X, feature_names = build_variance_features(rows, pred, Path(args.hkl))
    if args.base_scale_mode == "fixed_additive":
        fit_target = np.maximum(scatter_info["scatter"] - base_sig * base_sig, 0.0)
        coeff, feature_scales = fit_nonnegative_variance(X, fit_target, scatter_info["train"], args.ridge_alpha)
        add_var = np.maximum(X @ coeff, 0.0)
        base_var_component = base_sig * base_sig
        sigma_cal_var = base_var_component + add_var
        full_feature_names = feature_names
        full_coeff = coeff
        full_scales = feature_scales
    else:
        X_full = np.column_stack([base_sig * base_sig, X])
        full_feature_names = ["sigma_int2_scale"] + feature_names
        full_coeff, full_scales = fit_nonnegative_variance(X_full, np.maximum(scatter_info["scatter"], 0.0), scatter_info["train"], args.ridge_alpha)
        sigma_cal_var = np.maximum(X_full @ full_coeff, 0.0)
        base_var_component = (base_sig * base_sig) * full_coeff[0]
        add_var = np.maximum(sigma_cal_var - base_var_component, 0.0)
    min_var = (float(args.min_sigma_scale) * base_sig) ** 2
    sigma_cal = np.sqrt(np.maximum(sigma_cal_var, min_var))

    masks = choose_masks(obs, base_sig, d, scatter_info)
    blocks = {}
    for name, mask in masks.items():
        blocks[name] = {
            "base": metric_summary(obs, base_sig, pred, mask),
            "calibrated": metric_summary(obs, sigma_cal, pred, mask),
        }
        if np.count_nonzero(mask & scatter_info["usable"]):
            ratio = scatter_info["scatter"][mask & scatter_info["usable"]] / np.maximum(sigma_cal[mask & scatter_info["usable"]] ** 2, 1e-30)
            blocks[name]["replicate_scatter_over_sigma2_median"] = float(np.nanmedian(ratio))
            blocks[name]["replicate_scatter_over_sigma2_p95"] = float(np.nanpercentile(ratio, 95))

    coef_rows = [
        {"feature": name, "coefficient": float(c), "feature_scale": float(s)}
        for name, c, s in zip(full_feature_names, full_coeff, full_scales, strict=False)
    ]
    write_csv(out / "sigma_variance_coefficients.csv", coef_rows)

    row_out = []
    keys = scatter_info["keys"]
    for i, r in enumerate(rows):
        rr = dict(r)
        rr["p222_canonical_hkl"] = f"{keys[i][0]} {keys[i][1]} {keys[i][2]}"
        rr["replicate_multiplicity"] = int(scatter_info["multiplicity"][i])
        rr["replicate_scatter_target"] = float(scatter_info["scatter"][i]) if np.isfinite(scatter_info["scatter"][i]) else ""
        rr["sigma_base"] = float(base_sig[i])
        rr["sigma_replicate_calibrated"] = float(sigma_cal[i])
        rr["sigma_additional_variance"] = float(add_var[i])
        rr["replicate_train"] = bool(scatter_info["train"][i])
        rr["replicate_test"] = bool(scatter_info["test"][i])
        rr["pred_scaled_for_sigma"] = float(pred[i])
        row_out.append(rr)
    write_csv(out / "sigma_calibrated_rows.csv", row_out)

    source_wght = parse_wght(Path(args.res))
    summary = {
        "inputs": {
            "row_table": str(row_path.resolve()),
            "row_table_sha256": sha256(row_path),
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "res": str(Path(args.res).resolve()),
            "res_sha256": sha256(Path(args.res)),
        },
        "columns": {"obs": args.obs_column, "base_sigma": args.base_sigma_column, "prediction": args.raw_pred_column if args.raw_pred_column else args.pred_column},
        "scale_model": scale_metadata,
        "equivalence": {"mode": "P222 proper rotations only; no Friedel merge", "n_groups": scatter_info["n_groups"], "n_usable_groups": scatter_info["n_usable_groups"], "n_holdout_groups": scatter_info["n_holdout_groups"], "min_multiplicity": args.min_multiplicity},
        "fit": {"ridge_alpha": args.ridge_alpha, "holdout_fraction": args.holdout_fraction, "split_seed": args.split_seed, "source_res_wght_a": source_wght[0], "source_res_wght_b": source_wght[1]},
        "base_scale_mode": args.base_scale_mode,
        "coefficients": coef_rows,
        "blocks": blocks,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    test = blocks["scatter_test"]
    weak = blocks["test_weak_highres"]
    strong = blocks["test_strong"]
    report = [
        "# Replicate-Scatter Sigma Calibration",
        "",
        "This audit calibrates native row sigmas from P222-equivalent row scatter. Friedel mates are not merged. The Bloch predictions are fixed.",
        "",
        "```text",
        f"row_table: {row_path}",
        f"prediction: {args.raw_pred_column if args.raw_pred_column else args.pred_column}",
        f"scale mode: {scale_metadata['mode']}",
        f"base sigma: {args.base_sigma_column}",
        f"base scale mode: {args.base_scale_mode}",
        f"usable equivalent groups: {scatter_info['n_usable_groups']}",
        f"held-out equivalent groups: {scatter_info['n_holdout_groups']}",
        "```",
        "",
        "## Held-Out Metrics",
        "",
        "| subset | sigma model | n | R1 all (%) | wR2 all (%) | GooF | z RMS | median abs z | abs z p95 | scatter/sigma2 median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset_name, subset in [("scatter_test", test), ("weak_highres_test", weak), ("strong_test", strong)]:
        for label in ["base", "calibrated"]:
            m = subset[label]
            report.append(
                f"| {subset_name} | {label} | {m['n']} | {pct(m['R1_all']):.3f} | {pct(m['wR2_all']):.3f} | {m['GooF']:.3f} | {m['z_rms']:.3f} | {m['z_abs_median']:.3f} | {m['z_abs_p95']:.3f} | {subset.get('replicate_scatter_over_sigma2_median', float('nan')):.3f} |"
            )
    report.extend([
        "",
        "## Largest Variance Terms",
        "",
        "| feature | coefficient |",
        "|---|---:|",
    ])
    for row in sorted(coef_rows, key=lambda r: abs(float(r["coefficient"])), reverse=True)[:12]:
        report.append(f"| {row['feature']} | {row['coefficient']:.6g} |")
    report.extend([
        "",
        "## Verdict",
        "",
    ])
    base_rms = test["base"]["z_rms"]
    cal_rms = test["calibrated"]["z_rms"]
    weak_med = weak["calibrated"]["z_abs_median"]
    weak_base_med = weak["base"]["z_abs_median"]
    if args.base_scale_mode == "fit_base_scale":
        weak_p95 = weak["calibrated"]["z_abs_p95"]
        strong_rms = strong["calibrated"]["z_rms"]
        passes = np.isfinite(cal_rms) and abs(cal_rms - 1.0) < abs(base_rms - 1.0) and weak_p95 < 2.5 and strong_rms < 1.5
    else:
        passes = np.isfinite(cal_rms) and abs(cal_rms - 1.0) < abs(base_rms - 1.0) and weak_med <= weak_base_med * 1.15
    if passes:
        report.append("The replicate-calibrated sigma model improves held-out residual normalization without creating pathological weak/high-resolution or strong-row normalized residuals. It is a plausible native uncertainty model and should be tested in the full best-case pipeline.")
    else:
        report.append("The replicate-calibrated sigma model does not clearly improve held-out residual normalization. Keep it diagnostic; do not use it to claim better uncertainties.")
    report.append("")
    (out / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({"out_dir": str(out), "heldout_base_z_rms": base_rms, "heldout_calibrated_z_rms": cal_rms, "usable_groups": scatter_info["n_usable_groups"], "holdout_groups": scatter_info["n_holdout_groups"]}, indent=2))


if __name__ == "__main__":
    main()
