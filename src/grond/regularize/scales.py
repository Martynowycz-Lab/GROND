"""Native intensity scale fitting and residual metrics. A lower fitted residual is not proof of a better structure; use the existing matched controls and endpoint checks."""

from __future__ import annotations

import math

import numpy as np

from grond.legacy.engine.detector_scale_regularization import (
    solve_regularized_detector_field,
)


def shelx_weights(obs, sig, pred, a: float | np.ndarray, b: float | np.ndarray):
    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 / np.maximum(sig * sig + (a * p) ** 2 + b * p, 1e-30)


def solve_regularized_scales(diag_sums, x_y_sums, unique_groups, lambda_smooth):
    n = len(unique_groups)
    if n == 0:
        return {}
    if n == 1:
        return {int(unique_groups[0]): max(float(x_y_sums[0] / max(diag_sums[0], 1e-30)), 1e-30)}
    d = np.array(diag_sums, dtype=float)
    d[0] += float(lambda_smooth)
    d[1:-1] += 2.0 * float(lambda_smooth)
    d[-1] += float(lambda_smooth)
    c = np.full(n - 1, -float(lambda_smooth), dtype=float)
    b = np.array(x_y_sums, dtype=float)
    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    c_prime[0] = c[0] / max(d[0], 1e-30)
    d_prime[0] = b[0] / max(d[0], 1e-30)
    for i in range(1, n - 1):
        denom = d[i] - c[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / max(denom, 1e-30)
        d_prime[i] = (b[i] - c[i-1] * d_prime[i-1]) / max(denom, 1e-30)
    denom = d[n-1] - c[n-2] * c_prime[n-2]
    d_prime[n-1] = (b[n-1] - c[n-2] * d_prime[n-2]) / max(denom, 1e-30)
    x = np.zeros(n, dtype=float)
    x[-1] = d_prime[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return {int(g): max(float(val), 1e-30) for g, val in zip(unique_groups, x, strict=True)}


def _scale_fit_mask(obs, sig, pred, fit_mask):
    return (
        np.asarray(fit_mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
        & np.isfinite(pred)
        & (np.asarray(pred, dtype=float) > 0.0)
    )


def _fit_global_scale(obs, sig, pred, fit_mask, wght_a: float, wght_b: float, n_iter: int):
    fit = _scale_fit_mask(obs, sig, pred, fit_mask)
    if np.count_nonzero(fit) < 2:
        raise ValueError("at least two work rows with positive sigma and prediction are required to fit scales")
    x = np.asarray(pred[fit], dtype=float)
    y = np.asarray(obs[fit], dtype=float)
    sfit = np.asarray(sig[fit], dtype=float)
    w = 1.0 / (sfit * sfit)
    scale = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
    for _ in range(int(n_iter)):
        w = shelx_weights(y, sfit, scale * x, wght_a, wght_b)
        new_scale = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
        if abs(new_scale - scale) <= 1e-8 * max(1.0, abs(scale)):
            return new_scale
        scale = new_scale
    return scale


def fit_group_scales(obs, sig, pred, groups, fit_mask, wght_a: float, wght_b: float, n_iter: int = 6, scale_smoothness: float = 0.0):
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(obs) == len(sig) == len(pred) == len(groups) == len(fit_mask)):
        raise ValueError("observation, sigma, prediction, group, and fit-mask lengths differ")
    scaled = np.full_like(pred, np.nan, dtype=float)
    scales: dict[int, float] = {}
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        return scaled, scales
    global_scale = _fit_global_scale(obs, sig, pred, fit_mask, wght_a, wght_b, n_iter)

    if scale_smoothness > 0.0:
        diag_sums = np.zeros(len(unique_groups), dtype=float)
        x_y_sums = np.zeros(len(unique_groups), dtype=float)
        for j, g in enumerate(unique_groups):
            fit = (groups == g) & _scale_fit_mask(obs, sig, pred, fit_mask)
            if np.count_nonzero(fit) >= 2:
                x = pred[fit]
                y = obs[fit]
                sfit = sig[fit]
                w = 1.0 / np.maximum(sfit * sfit, 1e-30)
                diag_sums[j] = np.sum(w * x * x)
                x_y_sums[j] = np.sum(w * x * y)
        solved = solve_regularized_scales(diag_sums, x_y_sums, unique_groups, scale_smoothness)
        for g in unique_groups:
            value = float(solved[int(g)])
            scales[int(g)] = value if np.isfinite(value) and value > 1e-20 else global_scale

        for _ in range(int(n_iter)):
            diag_sums = np.zeros(len(unique_groups), dtype=float)
            x_y_sums = np.zeros(len(unique_groups), dtype=float)
            for j, g in enumerate(unique_groups):
                fit = (groups == g) & _scale_fit_mask(obs, sig, pred, fit_mask)
                if np.count_nonzero(fit) >= 2:
                    x = pred[fit]
                    y = obs[fit]
                    sfit = sig[fit]
                    pcalc = scales[int(g)] * x
                    w = shelx_weights(y, sfit, pcalc, wght_a, wght_b)
                    diag_sums[j] = np.sum(w * x * x)
                    x_y_sums[j] = np.sum(w * x * y)
            solved = solve_regularized_scales(diag_sums, x_y_sums, unique_groups, scale_smoothness)
            max_diff = 0.0
            for g in unique_groups:
                max_diff = max(max_diff, abs(solved[int(g)] - scales[int(g)]) / max(1.0, abs(scales[int(g)])))
            for g in unique_groups:
                value = float(solved[int(g)])
                scales[int(g)] = value if np.isfinite(value) and value > 1e-20 else global_scale
            if max_diff <= 1e-8:
                break

        for g in unique_groups:
            apply = (groups == g) & np.isfinite(pred)
            scaled[apply] = pred[apply] * scales[int(g)]
    else:
        for g in unique_groups:
            fit = (groups == g) & _scale_fit_mask(obs, sig, pred, fit_mask)
            apply = (groups == g) & np.isfinite(pred)
            if np.count_nonzero(fit) < 2:
                scales[int(g)] = global_scale
                scaled[apply] = pred[apply] * global_scale
                continue
            x = pred[fit]
            y = obs[fit]
            sfit = sig[fit]
            w = 1.0 / np.maximum(sfit * sfit, 1e-30)
            scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
            scale = scale if np.isfinite(scale) and scale > 1e-20 else global_scale
            for _ in range(int(n_iter)):
                pcalc = scale * x
                w = shelx_weights(y, sfit, pcalc, wght_a, wght_b)
                new_scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
                new_scale = new_scale if np.isfinite(new_scale) and new_scale > 1e-20 else global_scale
                if abs(new_scale - scale) <= 1e-8 * max(1.0, abs(scale)):
                    scale = new_scale
                    break
                scale = new_scale
            scales[int(g)] = scale
            scaled[apply] = pred[apply] * scale
    return scaled, scales


def fit_frame_detector_scales(
    obs,
    sig,
    pred,
    groups,
    det_bins,
    fit_mask,
    wght_a: float,
    wght_b: float,
    n_iter: int = 12,
    scale_smoothness: float = 0.0,
    detector_grid_size: int | None = None,
    detector_ridge: float = 0.05,
    detector_smoothness: float = 0.10,
    detector_min_rows: int = 20,
):
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    det_bins = np.asarray(det_bins)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(obs) == len(sig) == len(pred) == len(groups) == len(det_bins) == len(fit_mask)):
        raise ValueError("observation, sigma, prediction, scale-group, detector-bin, and fit-mask lengths differ")
    scaled = np.full_like(pred, np.nan, dtype=float)
    frame_scales: dict[int, float] = {}
    det_scales: dict[int, float] = {}
    good = _scale_fit_mask(obs, sig, pred, fit_mask)
    global_scale = _fit_global_scale(obs, sig, pred, fit_mask, wght_a, wght_b, n_iter)
    unique_groups = np.asarray(sorted(np.unique(groups[np.isfinite(groups)])), dtype=int)
    present_det = np.asarray(sorted(np.unique(det_bins[np.isfinite(det_bins)])), dtype=int)
    if len(unique_groups) == 0 or len(present_det) == 0:
        return scaled, frame_scales, det_scales
    gmap = {int(g): i for i, g in enumerate(unique_groups)}
    gi = np.asarray([gmap.get(int(g), -1) for g in groups], dtype=int)
    n_bins = int(detector_grid_size or math.ceil(math.sqrt(int(np.max(present_det)) + 1)))
    n_tiles = n_bins * n_bins
    di = np.asarray(
        [int(d) if np.isfinite(d) and 0 <= int(d) < n_tiles else -1 for d in det_bins],
        dtype=int,
    )
    frame = np.ones(len(unique_groups), dtype=float)
    det = np.ones(n_tiles, dtype=float)
    frame[:] = global_scale
    det[:] = 1.0
    if n_bins <= 1:
        raise ValueError("detector scale grid must be at least 2x2")
    for _ in range(int(n_iter)):
        base = pred * det[np.maximum(di, 0)]
        if scale_smoothness > 0.0:
            diag_sums = np.zeros(len(unique_groups), dtype=float)
            x_y_sums = np.zeros(len(unique_groups), dtype=float)
            for j, _g in enumerate(unique_groups):
                m = good & (gi == j) & np.isfinite(base) & (base > 0)
                if np.count_nonzero(m) >= 2:
                    x = base[m]
                    y = obs[m]
                    sfit = sig[m]
                    w = shelx_weights(y, sfit, x * frame[j], wght_a, wght_b)
                    diag_sums[j] = np.sum(w * x * x)
                    x_y_sums[j] = np.sum(w * x * y)
            solved_scales = solve_regularized_scales(diag_sums, x_y_sums, unique_groups, scale_smoothness)
            for j, g in enumerate(unique_groups):
                value = float(solved_scales[int(g)])
                frame[j] = value if np.isfinite(value) and value > 1e-20 else global_scale
        else:
            for j, _g in enumerate(unique_groups):
                m = good & (gi == j) & np.isfinite(base) & (base > 0)
                if np.count_nonzero(m) < 2:
                    continue
                x = base[m]
                y = obs[m]
                sfit = sig[m]
                w = shelx_weights(y, sfit, x * frame[j], wght_a, wght_b)
                frame[j] = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
        base = pred * frame[np.maximum(gi, 0)]
        diag = np.zeros(n_tiles, dtype=float)
        rhs = np.zeros(n_tiles, dtype=float)
        counts = np.zeros(n_tiles, dtype=int)
        for j in range(n_tiles):
            m = good & (di == j) & np.isfinite(base) & (base > 0)
            counts[j] = int(np.count_nonzero(m))
            if counts[j] < 2:
                continue
            x = base[m]
            y = obs[m]
            sfit = sig[m]
            w = shelx_weights(y, sfit, x * det[j], wght_a, wght_b)
            diag[j] = float(np.sum(w * x * x))
            rhs[j] = float(np.sum(w * x * y))
        det = solve_regularized_detector_field(
            diag,
            rhs,
            counts,
            n_bins=n_bins,
            ridge=float(detector_ridge),
            smoothness=float(detector_smoothness),
            min_rows=int(detector_min_rows),
        )
        median_det = float(np.median(det[np.isfinite(det) & (det > 0)]))
        if median_det > 0:
            det /= median_det
            frame *= median_det
    apply = np.isfinite(pred) & (gi >= 0) & (di >= 0)
    scaled[apply] = pred[apply] * frame[gi[apply]] * det[di[apply]]
    frame_scales = {str(int(g)): float(frame[j]) for j, g in enumerate(unique_groups)}
    det_scales = {str(j): float(det[j]) for j in range(n_tiles)}
    return scaled, frame_scales, det_scales


def native_metrics(obs, sig, pred, mask, wght_a: float, wght_b: float, n_params: int = 0):
    m = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
        & np.isfinite(pred)
    )
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "R2_all": float("nan"), "wR2_all": float("nan"), "GooF": float("nan"), "mean_weighted_abs_residual": float("nan"), "CC": float("nan")}
    gt = obs_m > 2.0 * sig_m
    fobs = np.sqrt(np.maximum(obs_m, 0.0))
    fpred = np.sqrt(np.maximum(pred_m, 0.0))
    eps = 1e-30

    def r1(local):
        if not np.any(local):
            return float("nan")
        return float(np.sum(np.abs(fobs[local] - fpred[local])) / max(np.sum(fobs[local]), eps))

    w = shelx_weights(obs_m, sig_m, pred_m, wght_a, wght_b)
    res = obs_m - pred_m
    wr2 = float(np.sqrt(np.sum(w * res * res) / max(np.sum(w * obs_m * obs_m), eps)))
    dof = max(1, len(obs_m) - int(n_params))
    goof = float(np.sqrt(np.sum(w * res * res) / dof))
    r2 = float(np.sum(np.abs(res)) / max(np.sum(np.abs(obs_m)), eps))
    cc = float(np.corrcoef(obs_m, pred_m)[0, 1]) if len(obs_m) > 2 and np.std(obs_m) > 0 and np.std(pred_m) > 0 else float("nan")
    mw_abs = float(np.mean(np.sqrt(w) * np.abs(res)))
    return {
        "n": int(len(obs_m)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "R2_all": r2,
        "wR2_all": wr2,
        "GooF": goof,
        "predictive_rms_z": float(np.sqrt(np.mean((res / sig_m) ** 2))),
        "weighted_rms_residual": float(np.sqrt(np.mean(w * res * res))),
        "parameter_count": int(n_params),
        "mean_weighted_abs_residual": mw_abs,
        "CC": cc,
        "dof": int(dof),
    }
