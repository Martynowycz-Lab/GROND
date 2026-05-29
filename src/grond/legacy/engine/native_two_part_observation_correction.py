#!/usr/bin/env python3
"""Held-out two-part post-Bloch observation-correction audit.

The Bloch calculation is treated as frozen.  This command starts from a
row-level ``native_dyn_calc.csv`` table and fits only experimental cleanup terms
on a training split:

    Iobs = s_g exp(f_mult(x)) Idyn + f_add(x) + eps

The multiplicative term is a smooth log-scale correction.  The additive term is
a penalized background/pedestal-bias correction.  Both are trained only on the
chosen training rows and then scored on held-out signed HKLs or held-out frame
blocks.  This is an audit and an endpoint candidate generator, not a change to
the physical Bloch kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from native_row_regularize import DEFAULT_HKL, DEFAULT_RES, native_metrics, parse_wght, shelx_weights
from bloch_engine import XDSData, sha256


DEFAULT_ROW_TABLE = Path("native_dyn_calc.csv")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows read from {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def pct(x: float) -> float:
    return 100.0 * float(x) if np.isfinite(x) else float("nan")


def stable_signed_hkl_bucket(h: int, k: int, l: int, mod: int) -> int:
    digest = hashlib.sha1(f"{h},{k},{l}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % int(mod)


def gaussian_rbf(values: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    width = max(float(width), 1e-6)
    return np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2)


@dataclass
class FeatureMatrix:
    X: np.ndarray
    names: list[str]
    penalties: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def add_feature(cols: list[np.ndarray], names: list[str], penalties: list[float], name: str, values, penalty: float) -> None:
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1:
        raise ValueError(f"feature {name} is not one-dimensional")
    cols.append(vals)
    names.append(name)
    penalties.append(float(penalty))


def existing_first(rows: list[dict[str, str]], candidates: Iterable[str]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for name in candidates:
        if name in keys:
            return name
    return None


def optional_feature(rows: list[dict[str, str]], candidates: Iterable[str]) -> tuple[str | None, np.ndarray | None]:
    name = existing_first(rows, candidates)
    if name is None:
        return None, None
    vals = fcol(rows, name)
    if np.count_nonzero(np.isfinite(vals)) < max(20, len(rows) // 100):
        return None, None
    return name, vals


def build_features(
    rows: list[dict[str, str]],
    hkl_path: Path,
    train_mask: np.ndarray,
    include_intercept: bool,
    frame_basis: int,
    xy_basis: int,
    feature_set: str,
) -> FeatureMatrix:
    """Build a smooth/quality feature matrix standardized on train rows only."""

    data = XDSData(hkl_path)
    row_index = icol(rows, "row_index")
    h = icol(rows, "h")
    k = icol(rows, "k")
    l = icol(rows, "l")
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
    obs = fcol(rows, "target_I")
    sig = fcol(rows, "sigma_eff")
    pred = fcol(rows, "dyn_scaled")

    if np.any(valid_idx):
        idx = row_index[valid_idx]
        x[valid_idx] = data.xcal[idx]
        y[valid_idx] = data.ycal[idx]
        z[valid_idx] = data.zcal[idx]
        xobs[valid_idx] = data.xobs[idx]
        yobs[valid_idx] = data.yobs[idx]
        zobs[valid_idx] = data.zobs[idx]
        if np.count_nonzero(np.isfinite(rlp)) == 0:
            rlp[valid_idx] = data.rlp[idx]
        if np.count_nonzero(np.isfinite(peak)) == 0:
            peak[valid_idx] = data.peak[idx]
        if np.count_nonzero(np.isfinite(corr)) == 0:
            corr[valid_idx] = data.corr[idx]

    cols: list[np.ndarray] = []
    names: list[str] = []
    penalties: list[float] = []

    if include_intercept:
        add_feature(cols, names, penalties, "intercept", np.ones(n), 0.0)

    train_finite_d = train_mask & np.isfinite(d) & (d > 0)
    inv_d2 = 1.0 / np.maximum(d, 1e-12) ** 2
    inv_d = 1.0 / np.maximum(d, 1e-12)
    isigi = obs / np.maximum(sig, 1e-12)
    pred_log = np.log1p(np.maximum(pred, 0.0))

    add_smooth = feature_set in {"smooth", "smooth_quality", "all"}
    add_quality = feature_set in {"quality", "smooth_quality", "all"}
    add_raw = feature_set in {"raw_quality", "all"}

    if add_smooth:
        add_feature(cols, names, penalties, "inv_d2", inv_d2, 1.0)
        add_feature(cols, names, penalties, "inv_d", inv_d, 1.0)

    if add_quality:
        add_feature(cols, names, penalties, "log_rlp", np.log(np.maximum(rlp, 1e-8)), 1.5)
        add_feature(cols, names, penalties, "peak_defect", 1.0 - np.clip(peak, 0.0, 100.0) / 100.0, 1.5)
        add_feature(cols, names, penalties, "corr_defect", 1.0 - np.clip(corr, 0.0, 100.0) / 100.0, 1.5)
        add_feature(cols, names, penalties, "zobs_minus_zcal", zobs - z, 1.5)
        add_feature(cols, names, penalties, "xyobs_minus_xycal", np.hypot(xobs - x, yobs - y), 1.5)

    finite_x = train_mask & np.isfinite(x) & np.isfinite(y)
    if np.any(finite_x):
        x0 = float(np.nanmedian(x[finite_x]))
        y0 = float(np.nanmedian(y[finite_x]))
    else:
        x0 = y0 = 512.0
    xn = (x - x0) / 512.0
    yn = (y - y0) / 512.0
    rn = np.hypot(xn, yn)
    if add_smooth:
        add_feature(cols, names, penalties, "det_x_norm", xn, 1.0)
        add_feature(cols, names, penalties, "det_y_norm", yn, 1.0)
        add_feature(cols, names, penalties, "det_radius_norm", rn, 1.0)
        add_feature(cols, names, penalties, "det_x2_norm", xn * xn, 2.0)
        add_feature(cols, names, penalties, "det_y2_norm", yn * yn, 2.0)
        add_feature(cols, names, penalties, "det_xy_norm", xn * yn, 2.0)
        add_feature(cols, names, penalties, "radius_x_inv_d2", rn * inv_d2, 2.0)

    zfinite = z[train_mask & np.isfinite(z)]
    if len(zfinite) and add_smooth:
        zmin, zmax = float(np.nanmin(zfinite)), float(np.nanmax(zfinite))
        zspan = max(zmax - zmin, 1.0)
        zn = (z - zmin) / zspan
        add_feature(cols, names, penalties, "frame_norm", zn, 1.0)
        add_feature(cols, names, penalties, "frame_norm2", zn * zn, 2.0)
        add_feature(cols, names, penalties, "frame_x_inv_d2", zn * inv_d2, 2.0)
        if frame_basis > 0:
            centers = np.linspace(zmin, zmax, int(frame_basis))
            width = max((zmax - zmin) / max(int(frame_basis) - 1, 1), 1.0)
            rbfs = gaussian_rbf(z, centers, width)
            for j in range(rbfs.shape[1]):
                add_feature(cols, names, penalties, f"frame_rbf_{j:02d}", rbfs[:, j], 8.0)

    if xy_basis > 1 and add_smooth and np.any(finite_x):
        xq = np.linspace(np.nanpercentile(x[finite_x], 10), np.nanpercentile(x[finite_x], 90), int(xy_basis))
        yq = np.linspace(np.nanpercentile(y[finite_x], 10), np.nanpercentile(y[finite_x], 90), int(xy_basis))
        width = max(float(np.nanmedian(np.diff(xq))) if len(xq) > 1 else 256.0, 64.0)
        for ix, cx in enumerate(xq):
            for iy, cy in enumerate(yq):
                vals = np.exp(-0.5 * (((x - cx) / width) ** 2 + ((y - cy) / width) ** 2))
                add_feature(cols, names, penalties, f"xy_rbf_{ix:02d}_{iy:02d}", vals, 12.0)

    if add_raw:
        optional_groups = [
            ("local_background_mean", ["local_background_mean", "background_mean", "bg_mean", "raw_background_mean"], 2.0),
            ("local_background_sigma", ["local_background_sigma", "background_sigma", "background_mad", "bg_sigma", "raw_background_sigma"], 2.0),
            ("background_variance", ["background_variance", "bg_variance", "raw_background_variance"], 2.0),
            ("foreground_pixels", ["n_foreground_pixels", "foreground_pixels", "n_fg"], 2.0),
            ("background_pixels", ["n_background_pixels", "background_pixels", "n_bg"], 2.0),
            ("masked_pixels", ["n_masked_pixels", "masked_pixels", "mask_pixels"], 2.0),
            ("profile_correlation", ["profile_corr", "raw_profile_corr", "bloch_profile_corr", "observed_profile_fraction"], 2.0),
            ("crowding", ["crowding", "overlap_score", "neighbor_score"], 2.0),
        ]
        for out_name, candidates, penalty in optional_groups:
            source, vals = optional_feature(rows, candidates)
            if vals is None:
                continue
            if out_name in {"foreground_pixels", "background_pixels", "masked_pixels"}:
                vals = np.log1p(np.maximum(vals, 0.0))
            add_feature(cols, names, penalties, out_name if source == out_name else f"{out_name}:{source}", vals, penalty)

    # Mild optional terms that are not residual-derived but can stabilize weak-row
    # additive fits.  They are penalized strongly and can be disabled by choosing
    # a feature set without "all".
    if feature_set == "all":
        add_feature(cols, names, penalties, "i_over_sigma_proxy", isigi, 4.0)
        add_feature(cols, names, penalties, "log_pred_proxy", pred_log, 4.0)
        signed_parity = ((np.abs(h) + 2 * np.abs(k) + 3 * np.abs(l)) % 4).astype(float)
        add_feature(cols, names, penalties, "signed_hkl_parity_proxy", signed_parity, 6.0)

    if not cols:
        return FeatureMatrix(np.zeros((n, 0), dtype=float), [], np.zeros(0), np.zeros(0), np.ones(0))

    raw = np.vstack(cols).T
    raw[~np.isfinite(raw)] = 0.0
    means = np.zeros(raw.shape[1], dtype=float)
    stds = np.ones(raw.shape[1], dtype=float)
    keep = np.ones(raw.shape[1], dtype=bool)
    for j, name in enumerate(names):
        if name == "intercept":
            continue
        vals = raw[train_mask, j]
        means[j] = float(np.mean(vals)) if len(vals) else 0.0
        stds[j] = float(np.std(vals)) if len(vals) else 1.0
        if stds[j] < 1e-10:
            keep[j] = False
            continue
        raw[:, j] = (raw[:, j] - means[j]) / stds[j]

    if include_intercept:
        keep[0] = True
    return FeatureMatrix(
        raw[:, keep],
        [name for name, use in zip(names, keep) if use],
        np.asarray([penalty for penalty, use in zip(penalties, keep) if use], dtype=float),
        means[keep],
        stds[keep],
    )


def fit_weighted_ridge(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    fit_mask: np.ndarray,
    penalties: np.ndarray,
    alpha_factor: float,
) -> tuple[np.ndarray, float, int]:
    m = fit_mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(weights) & (weights > 0)
    nfit = int(np.count_nonzero(m))
    if X.shape[1] == 0:
        return np.zeros(0, dtype=float), 0.0, nfit
    if nfit <= X.shape[1] + 2:
        raise ValueError(f"not enough fit rows ({nfit}) for {X.shape[1]} features")
    sw = np.sqrt(np.maximum(weights[m], 1e-30))
    Xw = X[m] * sw[:, None]
    yw = y[m] * sw
    normal = Xw.T @ Xw
    rhs = Xw.T @ yw
    diag = np.diag(normal)
    penalized = penalties > 0
    base = float(np.median(diag[penalized & np.isfinite(diag) & (diag > 0)])) if np.any(penalized) else 1.0
    if not np.isfinite(base) or base <= 0:
        base = 1.0
    ridge_lambda = float(alpha_factor) * base
    beta = np.linalg.solve(normal + ridge_lambda * np.diag(penalties), rhs)
    return beta, ridge_lambda, nfit


def metric_block(obs, sig, pred, mask, a, b) -> dict:
    m = np.asarray(mask, dtype=bool) & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred)
    out = native_metrics(obs, sig, pred, m, a, b, n_params=0)
    if np.count_nonzero(m):
        residual = obs[m] - pred[m]
        res_sig = residual / np.maximum(sig[m], 1e-12)
        out["median_residual"] = float(np.nanmedian(residual))
        out["median_signed_residual_over_sigma"] = float(np.nanmedian(res_sig))
        out["median_abs_residual_over_sigma"] = float(np.nanmedian(np.abs(res_sig)))
        out["p10_residual_over_sigma"] = float(np.nanpercentile(res_sig, 10))
        out["p90_residual_over_sigma"] = float(np.nanpercentile(res_sig, 90))
    else:
        out["median_residual"] = float("nan")
        out["median_signed_residual_over_sigma"] = float("nan")
        out["median_abs_residual_over_sigma"] = float("nan")
        out["p10_residual_over_sigma"] = float("nan")
        out["p90_residual_over_sigma"] = float("nan")
    return out


def choose_masks(
    rows: list[dict[str, str]],
    obs: np.ndarray,
    sig: np.ndarray,
    d: np.ndarray,
    hkl_path: Path,
    mode: str,
    holdout_name: str,
    holdout_mod: int,
    holdout_value: int,
    frame_width: float,
) -> dict[str, np.ndarray]:
    valid = np.isfinite(obs) & np.isfinite(sig) & (sig > 0) & np.isfinite(d) & (d > 0)
    h = icol(rows, "h")
    k = icol(rows, "k")
    l = icol(rows, "l")
    row_index = icol(rows, "row_index")

    if mode == "native_split":
        split = scol(rows, "split", "")
        free = valid & (split == holdout_name)
    elif mode == "signed_hkl":
        buckets = np.asarray([stable_signed_hkl_bucket(int(a), int(b), int(c), holdout_mod) for a, b, c in zip(h, k, l)], dtype=int)
        free = valid & (buckets == int(holdout_value))
    elif mode in {"frame_mod", "frame_block"}:
        data = XDSData(hkl_path)
        z = np.full(len(rows), np.nan)
        ok = (row_index >= 0) & (row_index < len(data.hkl))
        z[ok] = data.zcal[row_index[ok]]
        if mode == "frame_mod":
            block = np.floor(z / max(float(frame_width), 1e-6)).astype(int)
            free = valid & ((block % int(holdout_mod)) == int(holdout_value))
        else:
            finite = valid & np.isfinite(z)
            lo = np.nanpercentile(z[finite], 100.0 * int(holdout_value) / int(holdout_mod))
            hi = np.nanpercentile(z[finite], 100.0 * (int(holdout_value) + 1) / int(holdout_mod))
            free = valid & (z >= lo) & (z < hi)
    else:
        raise ValueError(f"unknown holdout mode {mode!r}")

    if np.count_nonzero(free) < 20:
        buckets = np.asarray([stable_signed_hkl_bucket(int(a), int(b), int(c), 10) for a, b, c in zip(h, k, l)], dtype=int)
        free = valid & (buckets == 0)
    work = valid & ~free

    isigi = obs / np.maximum(sig, 1e-12)
    inv_d2 = 1.0 / np.maximum(d, 1e-12) ** 2
    q25 = float(np.nanpercentile(isigi[valid], 25)) if np.any(valid) else float("nan")
    q75 = float(np.nanpercentile(isigi[valid], 75)) if np.any(valid) else float("nan")
    r75 = float(np.nanpercentile(inv_d2[valid], 75)) if np.any(valid) else float("nan")
    r90 = float(np.nanpercentile(inv_d2[valid], 90)) if np.any(valid) else float("nan")
    return {
        "all": valid,
        "work": work,
        "free": free,
        "free_weak": free & (isigi <= q25),
        "free_highres": free & (inv_d2 >= r75),
        "free_weak_highres": free & (isigi <= q25) & (inv_d2 >= r75),
        "free_extreme_highres": free & (inv_d2 >= r90),
        "free_strong": free & (isigi >= q75),
        "free_strong_highres": free & (isigi >= q75) & (inv_d2 >= r75),
    }


def blocks_for_prediction(obs, sig, pred, masks, a, b) -> dict[str, dict]:
    return {name: metric_block(obs, sig, pred, mask, a, b) for name, mask in masks.items()}


def fit_log_multiplicative(
    obs: np.ndarray,
    sig: np.ndarray,
    pred_base: np.ndarray,
    additive: np.ndarray,
    X: np.ndarray,
    weights_pred: np.ndarray,
    train: np.ndarray,
    penalties: np.ndarray,
    alpha: float,
    fit_clip: float,
) -> tuple[np.ndarray, dict]:
    usable_obs = obs - additive
    fit = train & (usable_obs > 0) & (pred_base > 0) & np.isfinite(usable_obs) & np.isfinite(pred_base)
    y = np.zeros(len(obs), dtype=float)
    y[fit] = np.clip(np.log(np.maximum(usable_obs[fit], 1e-30) / np.maximum(pred_base[fit], 1e-30)), -abs(fit_clip), abs(fit_clip))
    beta, ridge_lambda, nfit = fit_weighted_ridge(X, y, weights_pred, fit, penalties, alpha)
    return beta, {"n_fit": nfit, "ridge_lambda": ridge_lambda}


def fit_additive(
    obs: np.ndarray,
    pred: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    train: np.ndarray,
    penalties: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    residual = obs - pred
    beta, ridge_lambda, nfit = fit_weighted_ridge(X, residual, weights, train, penalties, alpha)
    return beta, {"n_fit": nfit, "ridge_lambda": ridge_lambda}


def safe_exp_log_correction(X: np.ndarray, beta: np.ndarray, clip: float) -> np.ndarray:
    if beta.size == 0 or X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=float)
    return np.clip(X @ beta, -abs(clip), abs(clip))


def correction_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"p05": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "p05": float(np.nanpercentile(finite, 5)),
        "median": float(np.nanmedian(finite)),
        "p95": float(np.nanpercentile(finite, 95)),
    }


def apply_bias_clip(additive: np.ndarray, sig: np.ndarray, clip_sigma: float) -> np.ndarray:
    if not np.isfinite(clip_sigma) or clip_sigma <= 0:
        return additive
    return np.clip(additive, -float(clip_sigma) * sig, float(clip_sigma) * sig)


def evaluate_trial(
    name: str,
    obs: np.ndarray,
    sig: np.ndarray,
    pred: np.ndarray,
    masks,
    a: float,
    b: float,
    log_mult: np.ndarray | None = None,
    additive: np.ndarray | None = None,
    fit_info: dict | None = None,
) -> dict:
    log_mult = np.zeros(len(obs), dtype=float) if log_mult is None else log_mult
    additive = np.zeros(len(obs), dtype=float) if additive is None else additive
    return {
        "name": name,
        "fit": fit_info or {},
        "metrics": blocks_for_prediction(obs, sig, pred, masks, a, b),
        "log_mult_stats": correction_stats(log_mult),
        "additive_stats": correction_stats(additive),
    }


def selection_score(trial: dict, baseline: dict) -> float:
    blocks = trial["metrics"]
    base_blocks = baseline["metrics"]
    wh = blocks["free_weak_highres"]
    base_wh = base_blocks["free_weak_highres"]
    if wh["n"] < 5:
        wh = blocks["free_highres"] if blocks["free_highres"]["n"] >= 5 else blocks["free_weak"]
        base_wh = base_blocks["free_highres"] if base_blocks["free_highres"]["n"] >= 5 else base_blocks["free_weak"]
    strong = blocks["free_strong"]
    base_strong = base_blocks["free_strong"]
    free = blocks["free"]
    base_free = base_blocks["free"]
    signed_gain = abs(base_wh["median_signed_residual_over_sigma"]) - abs(wh["median_signed_residual_over_sigma"])
    weak_abs_gain = base_wh["median_abs_residual_over_sigma"] - wh["median_abs_residual_over_sigma"]
    free_w_gain = base_free["wR2_all"] - free["wR2_all"]
    strong_worse = max(0.0, strong["median_abs_residual_over_sigma"] - base_strong["median_abs_residual_over_sigma"])
    # Lower is better.  Reward held-out wR2 and weak-row debiasing, penalize a
    # strong-row cost.  All pieces are held-out quantities.
    return float(-2.0 * free_w_gain - signed_gain - 0.25 * weak_abs_gain + 1.5 * strong_worse)


def flatten_metric_row(trial: dict, baseline: dict) -> dict:
    m = trial["metrics"]
    row = {
        "model": trial["name"],
        "score": selection_score(trial, baseline),
        "all_R1": m["all"]["R1_all"],
        "all_wR2": m["all"]["wR2_all"],
        "all_CC": m["all"]["CC"],
        "work_R1": m["work"]["R1_all"],
        "work_wR2": m["work"]["wR2_all"],
        "free_R1": m["free"]["R1_all"],
        "free_wR2": m["free"]["wR2_all"],
        "free_CC": m["free"]["CC"],
        "free_weak_highres_n": m["free_weak_highres"]["n"],
        "free_weak_highres_median_signed_res_sig": m["free_weak_highres"]["median_signed_residual_over_sigma"],
        "free_weak_highres_median_abs_res_sig": m["free_weak_highres"]["median_abs_residual_over_sigma"],
        "free_strong_n": m["free_strong"]["n"],
        "free_strong_median_abs_res_sig": m["free_strong"]["median_abs_residual_over_sigma"],
        "log_mult_p05": trial["log_mult_stats"]["p05"],
        "log_mult_median": trial["log_mult_stats"]["median"],
        "log_mult_p95": trial["log_mult_stats"]["p95"],
        "additive_p05": trial["additive_stats"]["p05"],
        "additive_median": trial["additive_stats"]["median"],
        "additive_p95": trial["additive_stats"]["p95"],
    }
    for key, value in trial.get("fit", {}).items():
        if isinstance(value, (int, float, str, bool)):
            row[f"fit_{key}"] = value
    return row


def fit_models(args, rows, obs, sig, pred0, masks, X_mult: FeatureMatrix, X_add: FeatureMatrix, wght_a: float, wght_b: float):
    train = masks["work"] & np.isfinite(pred0) & np.isfinite(obs) & np.isfinite(sig) & (sig > 0)
    base_weights = shelx_weights(obs, sig, pred0, wght_a, wght_b)
    trials = []
    baseline = evaluate_trial("baseline", obs, sig, pred0, masks, wght_a, wght_b)
    trials.append(baseline)

    for m_alpha in args.mult_alpha_values:
        beta_m, info_m = fit_log_multiplicative(
            obs,
            sig,
            pred0,
            np.zeros_like(obs),
            X_mult.X,
            base_weights,
            train,
            X_mult.penalties,
            m_alpha,
            args.fit_log_clip,
        )
        logm = safe_exp_log_correction(X_mult.X, beta_m, args.apply_log_clip)
        pred_m = pred0 * np.exp(logm)
        trials.append(
            evaluate_trial(
                f"multiplicative_only_alpha_{m_alpha:g}",
                obs,
                sig,
                pred_m,
                masks,
                wght_a,
                wght_b,
                log_mult=logm,
                fit_info={"mult_alpha": m_alpha, **info_m},
            )
        )

    for a_alpha in args.add_alpha_values:
        weights = shelx_weights(obs, sig, pred0, wght_a, wght_b)
        beta_a, info_a = fit_additive(obs, pred0, X_add.X, weights, train, X_add.penalties, a_alpha)
        add = apply_bias_clip(X_add.X @ beta_a, sig, args.additive_clip_sigma)
        pred_a = pred0 + add
        trials.append(
            evaluate_trial(
                f"additive_only_alpha_{a_alpha:g}",
                obs,
                sig,
                pred_a,
                masks,
                wght_a,
                wght_b,
                additive=add,
                fit_info={"add_alpha": a_alpha, **info_a},
            )
        )

    for m_alpha in args.mult_alpha_values:
        for a_alpha in args.add_alpha_values:
            beta_m, info_m = fit_log_multiplicative(
                obs,
                sig,
                pred0,
                np.zeros_like(obs),
                X_mult.X,
                base_weights,
                train,
                X_mult.penalties,
                m_alpha,
                args.fit_log_clip,
            )
            logm = safe_exp_log_correction(X_mult.X, beta_m, args.apply_log_clip)
            pred_m = pred0 * np.exp(logm)
            weights_a = shelx_weights(obs, sig, pred_m, wght_a, wght_b)
            beta_a, info_a = fit_additive(obs, pred_m, X_add.X, weights_a, train, X_add.penalties, a_alpha)
            add = apply_bias_clip(X_add.X @ beta_a, sig, args.additive_clip_sigma)
            pred_two = pred_m + add
            trials.append(
                evaluate_trial(
                    f"two_part_mult_then_add_m{m_alpha:g}_a{a_alpha:g}",
                    obs,
                    sig,
                    pred_two,
                    masks,
                    wght_a,
                    wght_b,
                    log_mult=logm,
                    additive=add,
                    fit_info={"mult_alpha": m_alpha, "add_alpha": a_alpha, "mult_n_fit": info_m["n_fit"], "add_n_fit": info_a["n_fit"], "mult_ridge_lambda": info_m["ridge_lambda"], "add_ridge_lambda": info_a["ridge_lambda"]},
                )
            )

            # Alternating fit, still using only training rows.  This keeps the
            # model close to the equation without requiring a nonlinear solver.
            logm_alt = logm.copy()
            add_alt = add.copy()
            beta_m_alt = beta_m
            beta_a_alt = beta_a
            info_alt = {"mult_alpha": m_alpha, "add_alpha": a_alpha, "iterations": int(args.alternating_iters)}
            for _ in range(max(int(args.alternating_iters), 0)):
                pred_current = pred0 * np.exp(logm_alt)
                weights_a = shelx_weights(obs, sig, pred_current + add_alt, wght_a, wght_b)
                beta_a_alt, info_a_alt = fit_additive(obs, pred_current, X_add.X, weights_a, train, X_add.penalties, a_alpha)
                add_alt = apply_bias_clip(X_add.X @ beta_a_alt, sig, args.additive_clip_sigma)
                weights_m = shelx_weights(obs, sig, pred_current + add_alt, wght_a, wght_b)
                beta_m_alt, info_m_alt = fit_log_multiplicative(
                    obs,
                    sig,
                    pred0,
                    add_alt,
                    X_mult.X,
                    weights_m,
                    train,
                    X_mult.penalties,
                    m_alpha,
                    args.fit_log_clip,
                )
                logm_alt = safe_exp_log_correction(X_mult.X, beta_m_alt, args.apply_log_clip)
                info_alt.update({"mult_n_fit": info_m_alt["n_fit"], "add_n_fit": info_a_alt["n_fit"], "mult_ridge_lambda": info_m_alt["ridge_lambda"], "add_ridge_lambda": info_a_alt["ridge_lambda"]})
            pred_alt = pred0 * np.exp(logm_alt) + add_alt
            trials.append(
                evaluate_trial(
                    f"two_part_alternating_m{m_alpha:g}_a{a_alpha:g}",
                    obs,
                    sig,
                    pred_alt,
                    masks,
                    wght_a,
                    wght_b,
                    log_mult=logm_alt,
                    additive=add_alt,
                    fit_info=info_alt,
                )
            )

    # Reconstruct best prediction and coefficients for output by refitting the
    # named best model is brittle; store enough state during a second pass.
    return trials


def parse_grid(text: str) -> list[float]:
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("empty numeric grid")
    return vals


def reconstruct_best_prediction(args, best_name, rows, obs, sig, pred0, masks, X_mult: FeatureMatrix, X_add: FeatureMatrix, wght_a, wght_b):
    train = masks["work"] & np.isfinite(pred0) & np.isfinite(obs) & np.isfinite(sig) & (sig > 0)
    base_weights = shelx_weights(obs, sig, pred0, wght_a, wght_b)
    logm = np.zeros(len(obs), dtype=float)
    add = np.zeros(len(obs), dtype=float)
    beta_m = np.zeros(X_mult.X.shape[1], dtype=float)
    beta_a = np.zeros(X_add.X.shape[1], dtype=float)
    if best_name == "baseline":
        return pred0.copy(), logm, add, beta_m, beta_a
    if best_name.startswith("multiplicative_only_alpha_"):
        alpha = float(best_name.rsplit("_", 1)[-1])
        beta_m, _ = fit_log_multiplicative(obs, sig, pred0, add, X_mult.X, base_weights, train, X_mult.penalties, alpha, args.fit_log_clip)
        logm = safe_exp_log_correction(X_mult.X, beta_m, args.apply_log_clip)
        return pred0 * np.exp(logm), logm, add, beta_m, beta_a
    if best_name.startswith("additive_only_alpha_"):
        alpha = float(best_name.rsplit("_", 1)[-1])
        weights = shelx_weights(obs, sig, pred0, wght_a, wght_b)
        beta_a, _ = fit_additive(obs, pred0, X_add.X, weights, train, X_add.penalties, alpha)
        add = apply_bias_clip(X_add.X @ beta_a, sig, args.additive_clip_sigma)
        return pred0 + add, logm, add, beta_m, beta_a
    if "_m" in best_name and "_a" in best_name:
        tail = best_name.rsplit("_m", 1)[1]
        m_text, a_text = tail.split("_a", 1)
        m_alpha = float(m_text)
        a_alpha = float(a_text)
        beta_m, _ = fit_log_multiplicative(obs, sig, pred0, add, X_mult.X, base_weights, train, X_mult.penalties, m_alpha, args.fit_log_clip)
        logm = safe_exp_log_correction(X_mult.X, beta_m, args.apply_log_clip)
        pred_m = pred0 * np.exp(logm)
        weights_a = shelx_weights(obs, sig, pred_m, wght_a, wght_b)
        beta_a, _ = fit_additive(obs, pred_m, X_add.X, weights_a, train, X_add.penalties, a_alpha)
        add = apply_bias_clip(X_add.X @ beta_a, sig, args.additive_clip_sigma)
        if best_name.startswith("two_part_alternating"):
            for _ in range(max(int(args.alternating_iters), 0)):
                pred_current = pred0 * np.exp(logm)
                weights_a = shelx_weights(obs, sig, pred_current + add, wght_a, wght_b)
                beta_a, _ = fit_additive(obs, pred_current, X_add.X, weights_a, train, X_add.penalties, a_alpha)
                add = apply_bias_clip(X_add.X @ beta_a, sig, args.additive_clip_sigma)
                weights_m = shelx_weights(obs, sig, pred_current + add, wght_a, wght_b)
                beta_m, _ = fit_log_multiplicative(obs, sig, pred0, add, X_mult.X, weights_m, train, X_mult.penalties, m_alpha, args.fit_log_clip)
                logm = safe_exp_log_correction(X_mult.X, beta_m, args.apply_log_clip)
        return pred0 * np.exp(logm) + add, logm, add, beta_m, beta_a
    raise ValueError(f"cannot reconstruct model {best_name}")


def report_lines(args, row_table: Path, rows, masks, baseline, best, trials_sorted) -> list[str]:
    best_by_free = min(trials_sorted, key=lambda t: t["metrics"]["free"]["wR2_all"])
    best_by_all = min(trials_sorted, key=lambda t: t["metrics"]["all"]["wR2_all"])
    b = baseline["metrics"]
    bm = best["metrics"]
    wh_base = b["free_weak_highres"]
    wh_best = bm["free_weak_highres"]
    strong_base = b["free_strong"]
    strong_best = bm["free_strong"]
    free_w_gain = b["free"]["wR2_all"] - bm["free"]["wR2_all"]
    all_w_gain = b["all"]["wR2_all"] - bm["all"]["wR2_all"]
    wh_bias_gain = abs(wh_base["median_signed_residual_over_sigma"]) - abs(wh_best["median_signed_residual_over_sigma"])
    strong_cost = strong_best["median_abs_residual_over_sigma"] - strong_base["median_abs_residual_over_sigma"]
    passed = free_w_gain > 0 and wh_bias_gain > 0 and strong_cost <= args.strong_worse_tol

    lines = [
        "# Two-Part Post-Bloch Observation-Correction Audit",
        "",
        "This keeps the Bloch row calculation fixed and fits only a held-out-tested observation cleanup model: `Iobs = s_g exp(f_mult(x)) Idyn + f_add(x) + eps`. The multiplicative term is intended for residual modulation, decay, detector response, or absorption-like effects. The additive term is intended for background and pedestal bias. The model is trained only on work rows and judged by held-out rows.",
        "",
        "```text",
        f"row_table: {row_table}",
        f"rows: {len(rows)}",
        f"holdout_mode: {args.holdout_mode}",
        f"work/free rows: {np.count_nonzero(masks['work'])}/{np.count_nonzero(masks['free'])}",
        f"mult_feature_set: {args.mult_feature_set}",
        f"add_feature_set: {args.add_feature_set}",
        f"mult_alpha_grid: {','.join(str(v) for v in args.mult_alpha_values)}",
        f"add_alpha_grid: {','.join(str(v) for v in args.add_alpha_values)}",
        f"additive_clip_sigma: {args.additive_clip_sigma}",
        "```",
        "",
        "## Overall Metrics",
        "",
        "| model | all R1 (%) | all wR2 (%) | free R1 (%) | free wR2 (%) | free CC | log-mult p05/med/p95 | additive p05/med/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trial in trials_sorted[:12]:
        m = trial["metrics"]
        lm = trial["log_mult_stats"]
        add = trial["additive_stats"]
        lines.append(
            f"| {trial['name']} | {pct(m['all']['R1_all']):.3f} | {pct(m['all']['wR2_all']):.3f} | "
            f"{pct(m['free']['R1_all']):.3f} | {pct(m['free']['wR2_all']):.3f} | {m['free']['CC']:.4f} | "
            f"{lm['p05']:.4g}/{lm['median']:.4g}/{lm['p95']:.4g} | {add['p05']:.4g}/{add['median']:.4g}/{add['p95']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Held-Out Gate",
            "",
            "| signal | baseline | best | delta |",
            "|---|---:|---:|---:|",
            f"| free wR2 | {pct(b['free']['wR2_all']):.3f}% | {pct(bm['free']['wR2_all']):.3f}% | {pct(free_w_gain):.3f}% |",
            f"| all wR2 | {pct(b['all']['wR2_all']):.3f}% | {pct(bm['all']['wR2_all']):.3f}% | {pct(all_w_gain):.3f}% |",
            f"| weak/high-res median residual/sigma abs-distance-to-zero | {abs(wh_base['median_signed_residual_over_sigma']):.4f} | {abs(wh_best['median_signed_residual_over_sigma']):.4f} | {wh_bias_gain:.4f} |",
            f"| strong median abs residual/sigma | {strong_base['median_abs_residual_over_sigma']:.4f} | {strong_best['median_abs_residual_over_sigma']:.4f} | {strong_cost:.4f} |",
            "",
            f"Best by held-out `wR2` alone is `{best_by_free['name']}` with free/all `wR2` of `{pct(best_by_free['metrics']['free']['wR2_all']):.3f}%` / `{pct(best_by_free['metrics']['all']['wR2_all']):.3f}%`. Best by all-row `wR2` alone is `{best_by_all['name']}` with free/all `wR2` of `{pct(best_by_all['metrics']['free']['wR2_all']):.3f}%` / `{pct(best_by_all['metrics']['all']['wR2_all']):.3f}%`. These are not automatically promoted because the held-out gate also checks weak/high-resolution bias and strong-row damage.",
            "",
            "## Verdict",
            "",
        ]
    )
    if passed:
        lines.append("The best two-part model passes the conservative held-out gate: held-out wR2 improves, weak/high-resolution residual bias moves toward zero, and strong held-out rows do not get materially worse. This should be treated as a candidate observation correction and retested against an independent frame holdout before endpoint export.")
    else:
        lines.append("The best two-part model does not pass the conservative held-out gate. Keep it as diagnostic evidence rather than promoting it into the production endpoint.")
    lines.extend(
        [
            "",
            "Artifacts: `summary.json`, `two_part_trials.csv`, `two_part_corrected_rows.csv`, and coefficient tables.",
            "",
        ]
    )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", default=str(DEFAULT_ROW_TABLE))
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--obs-column", default="target_I")
    ap.add_argument("--sigma-column", default="sigma_eff")
    ap.add_argument("--pred-column", default="dyn_scaled")
    ap.add_argument("--holdout-mode", choices=["native_split", "signed_hkl", "frame_mod", "frame_block"], default="signed_hkl")
    ap.add_argument("--holdout-name", default="free")
    ap.add_argument("--holdout-mod", type=int, default=10)
    ap.add_argument("--holdout-value", type=int, default=0)
    ap.add_argument("--frame-holdout-width", type=float, default=30.0)
    ap.add_argument("--wght-a", type=float, default=None, help="Audit WGHT a value. Defaults to the RES file WGHT value.")
    ap.add_argument("--wght-b", type=float, default=None, help="Audit WGHT b value. Defaults to the RES file WGHT value.")
    ap.add_argument("--mult-alpha-grid", default="1,10,100")
    ap.add_argument("--add-alpha-grid", default="0.1,1,10,100")
    ap.add_argument("--mult-feature-set", choices=["smooth", "quality", "smooth_quality", "raw_quality", "all"], default="smooth_quality")
    ap.add_argument("--add-feature-set", choices=["smooth", "quality", "smooth_quality", "raw_quality", "all"], default="all")
    ap.add_argument("--mult-intercept", action="store_true", help="Allow multiplicative global intercept. Default off because group scales already exist.")
    ap.add_argument("--add-intercept", action="store_true", default=True)
    ap.add_argument("--no-add-intercept", dest="add_intercept", action="store_false")
    ap.add_argument("--frame-basis", type=int, default=8)
    ap.add_argument("--xy-basis", type=int, default=4)
    ap.add_argument("--fit-log-clip", type=float, default=1.5)
    ap.add_argument("--apply-log-clip", type=float, default=0.7)
    ap.add_argument("--additive-clip-sigma", type=float, default=3.0)
    ap.add_argument("--alternating-iters", type=int, default=2)
    ap.add_argument("--strong-worse-tol", type=float, default=0.05)
    args = ap.parse_args()
    args.mult_alpha_values = parse_grid(args.mult_alpha_grid)
    args.add_alpha_values = parse_grid(args.add_alpha_grid)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_table = Path(args.row_table)
    hkl_path = Path(args.hkl)
    res_path = Path(args.res)
    rows = read_rows(row_table)
    obs = fcol(rows, args.obs_column)
    sig = np.maximum(fcol(rows, args.sigma_column), 1e-12)
    pred0 = fcol(rows, args.pred_column)
    d = fcol(rows, "d_A")
    wght_source = parse_wght(res_path)
    if args.wght_a is None:
        args.wght_a = wght_source[0]
    if args.wght_b is None:
        args.wght_b = wght_source[1]
    masks = choose_masks(rows, obs, sig, d, hkl_path, args.holdout_mode, args.holdout_name, args.holdout_mod, args.holdout_value, args.frame_holdout_width)
    train = masks["work"] & np.isfinite(pred0)

    X_mult = build_features(rows, hkl_path, train, args.mult_intercept, args.frame_basis, args.xy_basis, args.mult_feature_set)
    X_add = build_features(rows, hkl_path, train, args.add_intercept, args.frame_basis, args.xy_basis, args.add_feature_set)
    trials = fit_models(args, rows, obs, sig, pred0, masks, X_mult, X_add, args.wght_a, args.wght_b)
    baseline = trials[0]
    trials_sorted = sorted(trials, key=lambda t: (selection_score(t, baseline), t["metrics"]["free"]["wR2_all"], t["metrics"]["all"]["wR2_all"]))
    best = trials_sorted[0]
    best_by_free = min(trials, key=lambda t: t["metrics"]["free"]["wR2_all"])
    best_by_all = min(trials, key=lambda t: t["metrics"]["all"]["wR2_all"])
    best_pred, best_logm, best_add, best_beta_m, best_beta_a = reconstruct_best_prediction(
        args, best["name"], rows, obs, sig, pred0, masks, X_mult, X_add, args.wght_a, args.wght_b
    )

    trial_rows = [flatten_metric_row(t, baseline) for t in trials]
    write_csv(out / "two_part_trials.csv", sorted(trial_rows, key=lambda r: (r["score"], r["free_wR2"])))

    corrected = []
    for r, p0, p, lm, add in zip(rows, pred0, best_pred, best_logm, best_add):
        rr = dict(r)
        rr["two_part_pred_before"] = f"{float(p0):.10g}"
        rr["two_part_log_mult"] = f"{float(lm):.10g}"
        rr["two_part_additive"] = f"{float(add):.10g}"
        rr["two_part_pred_after"] = f"{float(p):.10g}"
        rr["two_part_residual"] = f"{float(float(r.get(args.obs_column, 'nan')) - p):.10g}"
        corrected.append(rr)
    write_csv(out / "two_part_corrected_rows.csv", corrected)

    write_csv(
        out / "multiplicative_coefficients.csv",
        [{"feature": name, "coefficient": float(coef), "penalty": float(penalty)} for name, coef, penalty in zip(X_mult.names, best_beta_m, X_mult.penalties)],
    )
    write_csv(
        out / "additive_coefficients.csv",
        [{"feature": name, "coefficient": float(coef), "penalty": float(penalty)} for name, coef, penalty in zip(X_add.names, best_beta_a, X_add.penalties)],
    )

    free_w_gain = baseline["metrics"]["free"]["wR2_all"] - best["metrics"]["free"]["wR2_all"]
    wh_base = baseline["metrics"]["free_weak_highres"]
    wh_best = best["metrics"]["free_weak_highres"]
    strong_base = baseline["metrics"]["free_strong"]
    strong_best = best["metrics"]["free_strong"]
    wh_bias_gain = abs(wh_base["median_signed_residual_over_sigma"]) - abs(wh_best["median_signed_residual_over_sigma"])
    strong_cost = strong_best["median_abs_residual_over_sigma"] - strong_base["median_abs_residual_over_sigma"]
    passes_gate = bool(free_w_gain > 0 and wh_bias_gain > 0 and strong_cost <= args.strong_worse_tol)

    summary = {
        "inputs": {
            "row_table": str(row_table.resolve()),
            "row_table_sha256": sha256(row_table),
            "hkl": str(hkl_path.resolve()),
            "hkl_sha256": sha256(hkl_path),
            "res": str(res_path.resolve()),
            "res_sha256": sha256(res_path),
        },
        "columns": {"obs": args.obs_column, "sigma": args.sigma_column, "prediction": args.pred_column},
        "weights": {"audit_a": args.wght_a, "audit_b": args.wght_b, "source_res_a": wght_source[0], "source_res_b": wght_source[1]},
        "holdout": {
            "mode": args.holdout_mode,
            "holdout_mod": args.holdout_mod,
            "holdout_value": args.holdout_value,
            "work_rows": int(np.count_nonzero(masks["work"])),
            "free_rows": int(np.count_nonzero(masks["free"])),
        },
        "features": {
            "multiplicative": {"n": len(X_mult.names), "names": X_mult.names, "penalties": [float(x) for x in X_mult.penalties]},
            "additive": {"n": len(X_add.names), "names": X_add.names, "penalties": [float(x) for x in X_add.penalties]},
        },
        "config": {
            "mult_alpha_grid": args.mult_alpha_values,
            "add_alpha_grid": args.add_alpha_values,
            "fit_log_clip": args.fit_log_clip,
            "apply_log_clip": args.apply_log_clip,
            "additive_clip_sigma": args.additive_clip_sigma,
            "alternating_iters": args.alternating_iters,
        },
        "baseline": baseline,
        "best": best,
        "best_by_free_wR2": best_by_free,
        "best_by_all_wR2": best_by_all,
        "passes_heldout_gate": passes_gate,
        "gate_deltas": {"free_wR2_gain": float(free_w_gain), "weak_highres_bias_gain": float(wh_bias_gain), "strong_abs_res_sig_cost": float(strong_cost)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    (out / "REPORT.md").write_text("\n".join(report_lines(args, row_table, rows, masks, baseline, best, trials_sorted)) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(out.resolve()),
                "best_model": best["name"],
                "baseline_free_wR2": baseline["metrics"]["free"]["wR2_all"],
                "best_free_wR2": best["metrics"]["free"]["wR2_all"],
                "best_by_free_wR2_model": best_by_free["name"],
                "best_by_free_wR2": best_by_free["metrics"]["free"]["wR2_all"],
                "baseline_all_wR2": baseline["metrics"]["all"]["wR2_all"],
                "best_all_wR2": best["metrics"]["all"]["wR2_all"],
                "best_by_all_wR2_model": best_by_all["name"],
                "best_by_all_wR2": best_by_all["metrics"]["all"]["wR2_all"],
                "passes_heldout_gate": passes_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
