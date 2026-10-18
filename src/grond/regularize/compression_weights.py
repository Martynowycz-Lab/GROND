"""Frozen-prediction scale objectives and fits.

Publication-style row factors and native fitting residuals have different weights, so their objective
definitions are kept separate.
"""

from __future__ import annotations

import math

import numpy as np

from grond.engine.detector_scale_regularization import (
    infer_square_grid_size,
    solve_regularized_detector_field,
)


def safe_float(value, default=float("nan")) -> float:
    """Return value as a float, or default when it cannot be converted."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def col(rows: list[dict[str, str]], name: str, default=float("nan")) -> np.ndarray:
    """Return one column of the row table as a float array, with default for missing or non-numeric values."""

    return np.asarray([safe_float(r.get(name, default), default) for r in rows], dtype=float)


def source_sigma_array(rows: list[dict[str, str]]) -> np.ndarray:
    """Return immutable imported sigma(Iobs), with historical fallbacks."""

    for name in ("source_sigma", "xds_sigma", "sigma", "sigma_eff"):
        if name in rows[0]:
            values = col(rows, name)
            if np.any(np.isfinite(values) & (values > 0)):
                return values
    return np.full(len(rows), np.nan, dtype=float)


def jana_adjusted_observations(obs: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the weak/negative-observation convention reported by Klar et al."""

    weak = obs < 0.01 * sigma
    return np.where(weak, 0.0, obs), weak


def row_sqrt_weights(obs: np.ndarray, sigma: np.ndarray, instability_u: float) -> np.ndarray:
    """Return the row/MR least-squares weights w**2 of each intensity residual.

    1/w is the sigma of sqrt(Iobs), sigma / (2 sqrt(Iobs)), with instability_u * sqrt(Iobs) added in quadrature.
    Weak rows (Iobs < 0.01 sigma) count as Iobs = 0 and use 5 sqrt(sigma) as that sigma.
    """

    obs_adjusted, weak = jana_adjusted_observations(obs, sigma)
    sqrt_obs = np.sqrt(np.maximum(obs_adjusted, 0.0))
    sigma_sqrt = sigma / np.maximum(2.0 * sqrt_obs, 1e-30)
    sigma_sqrt[weak] = 5.0 * np.sqrt(sigma[weak])
    denom = np.sqrt(sigma_sqrt * sigma_sqrt + (float(instability_u) * sqrt_obs) ** 2)
    # The GROND row/MR wR expression squares w*(Iobs-Icalc), so the least-squares
    # normal equations use w**2 as the intensity residual weight.
    return 1.0 / np.maximum(denom * denom, 1e-30)


def shelx_weights(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, a: float, b: float) -> np.ndarray:
    """Return SHELXL WGHT weights 1 / (sigma^2 + (a P)^2 + b P), with P = (max(Iobs, 0) + 2 max(Icalc, 0)) / 3."""

    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    return 1.0 / np.maximum(sig * sig + (float(a) * p) ** 2 + float(b) * p, 1e-30)


def objective_weights(
    obs: np.ndarray,
    sig: np.ndarray,
    pred_scaled: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
) -> np.ndarray:
    """Return the scale-fit weights of one objective: sigma, native or row_mr.

    sigma is 1/sigma^2, native the SHELXL WGHT a, b weights and row_mr the row_sqrt_weights.
    """

    if objective == "sigma":
        return 1.0 / np.maximum(sig * sig, 1e-30)
    if objective == "native":
        return shelx_weights(obs, sig, pred_scaled, native_wght_a, native_wght_b)
    if objective == "row_mr":
        return row_sqrt_weights(obs, sig, instability_u)
    raise ValueError(f"unknown objective {objective!r}")


def solve_scale(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """Return the weighted least-squares scale k minimizing sum weight (y - k x)^2, clipped at 0."""

    return max(float(np.sum(weight * x * y) / max(np.sum(weight * x * x), 1e-30)), 0.0)


def fit_global_scale(
    obs: np.ndarray,
    sig: np.ndarray,
    raw: np.ndarray,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    n_iter: int,
) -> float:
    """Fit the shared fallback scale without consulting held-out rows."""

    fit = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0)
    if np.count_nonzero(fit) < 2:
        raise ValueError("at least two scale-fit rows with positive sigma and prediction are required")
    x = raw[fit]
    y = obs[fit]
    s = sig[fit]
    weight = 1.0 / np.maximum(s * s, 1e-30)
    scale = max(solve_scale(x, y, weight), 1e-30)
    for _ in range(int(n_iter)):
        pred_scaled = scale * x
        weight = objective_weights(y, s, pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
        new_scale = max(solve_scale(x, y, weight), 1e-30)
        if abs(new_scale - scale) <= 1e-9 * max(1.0, abs(scale)):
            return new_scale
        scale = new_scale
    return scale


def require_scale_coverage(
    raw: np.ndarray,
    scaled: np.ndarray,
    *topology: np.ndarray,
) -> None:
    """Fail closed if scale replay loses any valid raw prediction."""

    expected = np.isfinite(raw) & (raw > 0)
    missing = expected & (~np.isfinite(scaled) | (scaled <= 0))
    for values in topology:
        missing |= expected & ~np.isfinite(values)
    if np.any(missing):
        raise ValueError(
            "scale fit did not cover "
            f"{int(np.count_nonzero(missing))} valid raw prediction(s); refusing a partial row contract"
        )


def fit_group_scales(
    obs: np.ndarray,
    sig: np.ndarray,
    raw: np.ndarray,
    groups: np.ndarray,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    n_iter: int = 8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one scale per frame group on the fit_mask rows and apply it to every finite prediction of the group.

    Returns (scaled predictions, {group: scale}). Weights start at 1/sigma^2 and are recomputed for the objective
    for up to n_iter passes. A group with fewer than two fit rows takes the global scale.
    """

    scaled = np.full_like(raw, np.nan, dtype=float)
    scales: dict[str, float] = {}
    good0 = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0)
    global_scale = fit_global_scale(
        obs,
        sig,
        raw,
        fit_mask,
        objective,
        native_wght_a,
        native_wght_b,
        instability_u,
        n_iter,
    )
    finite_groups = np.isfinite(groups)
    for group in sorted(set(groups[finite_groups].astype(int))):
        fit = good0 & (groups == group)
        apply = np.isfinite(raw) & finite_groups & (groups == group)
        if np.count_nonzero(fit) < 2:
            scales[str(int(group))] = float(global_scale)
            scaled[apply] = raw[apply] * global_scale
            continue
        x = raw[fit]
        y = obs[fit]
        s = sig[fit]
        weight = 1.0 / np.maximum(s * s, 1e-30)
        scale = solve_scale(x, y, weight)
        for _ in range(int(n_iter)):
            pred_scaled = scale * x
            weight = objective_weights(y, s, pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            new_scale = solve_scale(x, y, weight)
            if abs(new_scale - scale) <= 1e-9 * max(1.0, abs(scale)):
                scale = new_scale
                break
            scale = new_scale
        scales[str(int(group))] = float(scale)
        scaled[apply] = raw[apply] * scale
    require_scale_coverage(raw, scaled, groups)
    return scaled, scales


def fit_group_detector_scales(
    obs: np.ndarray,
    sig: np.ndarray,
    raw: np.ndarray,
    groups: np.ndarray,
    det_bins: np.ndarray,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    n_iter: int = 16,
    detector_grid_size: int | None = None,
    detector_ridge: float = 0.05,
    detector_smoothness: float = 0.10,
    detector_min_rows: int = 20,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """Fit frame-group scales times a regularized square detector field by alternating updates on the fit_mask rows.

    Returns (scaled predictions, {group: frame scale}, {tile: detector scale}). The detector field is
    ridge-regularized toward 1 and spatially smoothed; after each pass it is divided by its median and the frame
    scales absorb that factor.
    """

    scaled = np.full_like(raw, np.nan, dtype=float)
    good = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0) & np.isfinite(groups) & np.isfinite(det_bins)
    unique_groups = np.asarray(sorted(set(groups[np.isfinite(groups)].astype(int))), dtype=int)
    present_det = np.asarray(sorted(set(det_bins[np.isfinite(det_bins)].astype(int))), dtype=int)
    if len(unique_groups) == 0 or len(present_det) == 0:
        require_scale_coverage(raw, scaled, groups, det_bins)
        return scaled, {}, {}
    gmap = {int(g): i for i, g in enumerate(unique_groups)}
    gi = np.asarray([gmap.get(int(g), -1) if np.isfinite(g) else -1 for g in groups], dtype=int)
    n_bins = infer_square_grid_size(det_bins, detector_grid_size)
    n_tiles = n_bins * n_bins
    di = np.asarray(
        [int(d) if np.isfinite(d) and 0 <= int(d) < n_tiles else -1 for d in det_bins],
        dtype=int,
    )
    global_scale = fit_global_scale(
        obs,
        sig,
        raw,
        fit_mask,
        objective,
        native_wght_a,
        native_wght_b,
        instability_u,
        n_iter,
    )
    frame = np.full(len(unique_groups), global_scale, dtype=float)
    det = np.ones(n_tiles, dtype=float)
    for _ in range(int(n_iter)):
        base = raw * det[np.maximum(di, 0)]
        for j in range(len(unique_groups)):
            m = good & (gi == j) & np.isfinite(base) & (base > 0)
            if np.count_nonzero(m) < 2:
                continue
            pred_scaled = base[m] * frame[j]
            weight = objective_weights(obs[m], sig[m], pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            frame[j] = max(solve_scale(base[m], obs[m], weight), 1e-30)
        base = raw * frame[np.maximum(gi, 0)]
        diag = np.zeros(n_tiles, dtype=float)
        rhs = np.zeros(n_tiles, dtype=float)
        counts = np.zeros(n_tiles, dtype=int)
        for j in range(n_tiles):
            m = good & (di == j) & np.isfinite(base) & (base > 0)
            counts[j] = int(np.count_nonzero(m))
            if counts[j] < 2:
                continue
            pred_scaled = base[m] * det[j]
            weight = objective_weights(obs[m], sig[m], pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            diag[j] = float(np.sum(weight * base[m] * base[m]))
            rhs[j] = float(np.sum(weight * base[m] * obs[m]))
        det = solve_regularized_detector_field(
            diag,
            rhs,
            counts,
            n_bins=n_bins,
            ridge=float(detector_ridge),
            smoothness=float(detector_smoothness),
            min_rows=int(detector_min_rows),
        )
        finite = det[np.isfinite(det) & (det > 0)]
        if len(finite):
            median_det = float(np.median(finite))
            if median_det > 0:
                det /= median_det
                frame *= median_det
    apply = np.isfinite(raw) & (gi >= 0) & (di >= 0)
    scaled[apply] = raw[apply] * frame[gi[apply]] * det[di[apply]]
    require_scale_coverage(raw, scaled, groups, det_bins)
    frame_scales = {str(int(g)): float(frame[j]) for j, g in enumerate(unique_groups)}
    det_scales = {str(j): float(det[j]) for j in range(n_tiles)}
    return scaled, frame_scales, det_scales


def row_mr_metrics(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, mask: np.ndarray, instability_u: float, observed_sigma_cut: float) -> dict:
    """Return JANA-style unmerged row statistics of pred on the masked rows: Nall, Nobs, Robs, Rall, wRall.

    R values are fractions. Robs and Rall are amplitude R factors on sqrt(I); observed rows have Iobs >
    observed_sigma_cut * sigma. Weak rows (Iobs < 0.01 sigma) count as Iobs = 0. wRall is on intensities with the
    row_sqrt_weights error model.
    """

    m = mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"Nall": 0, "Nobs": 0, "Robs": float("nan"), "Rall": float("nan"), "wRall": float("nan")}
    obs_original = obs_m.copy()
    obs_m, _weak = jana_adjusted_observations(obs_m, sig_m)
    obs_pos = np.maximum(obs_m, 0.0)
    pred_pos = np.maximum(pred_m, 0.0)
    fobs = np.sqrt(obs_pos)
    fcalc = np.sqrt(pred_pos)
    observed = obs_original > float(observed_sigma_cut) * sig_m

    def r_factor(local_mask: np.ndarray) -> float:
        """Return sum |Fobs - Fcalc| / sum Fobs over the local_mask rows, NaN when there are none."""

        if not np.any(local_mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[local_mask] - fcalc[local_mask])) / max(np.sum(fobs[local_mask]), 1e-30))

    w = np.sqrt(row_sqrt_weights(obs_original, sig_m, instability_u))
    wr = float(np.sqrt(np.sum((w * np.abs(obs_m - pred_m)) ** 2) / max(np.sum((w * obs_m) ** 2), 1e-30)))
    return {
        "Nall": int(len(obs_m)),
        "Nobs": int(np.count_nonzero(observed)),
        "Robs": r_factor(observed),
        "Rall": r_factor(np.ones_like(observed, dtype=bool)),
        "wRall": wr,
    }


def native_metrics(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, mask: np.ndarray, wght_a: float, wght_b: float, n_params: int = 0) -> dict:
    """Return SHELXL-style statistics of pred on the masked rows: R1_all, R1_gt, wR2_all, GooF and chi2.

    R values are fractions; R1 is on sqrt(I) (R1_gt: Iobs > 2 sigma), wR2 on I with SHELXL WGHT weights. dof = n -
    n_params, at least 1.
    """

    m = mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "wR2_all": float("nan"), "GooF": float("nan")}
    gt = obs_m > 2.0 * sig_m
    fobs = np.sqrt(np.maximum(obs_m, 0.0))
    fpred = np.sqrt(np.maximum(pred_m, 0.0))

    def r1(local_mask: np.ndarray) -> float:
        """Return sum |Fobs - Fcalc| / sum Fobs over the local_mask rows, NaN when there are none."""

        if not np.any(local_mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[local_mask] - fpred[local_mask])) / max(np.sum(fobs[local_mask]), 1e-30))

    w = shelx_weights(obs_m, sig_m, pred_m, wght_a, wght_b)
    res = obs_m - pred_m
    dof = max(1, len(obs_m) - int(n_params))
    chi2 = float(np.sum(w * res * res))
    return {
        "n": int(len(obs_m)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "wR2_all": float(np.sqrt(chi2 / max(np.sum(w * obs_m * obs_m), 1e-30))),
        "GooF": float(np.sqrt(chi2 / dof)),
        "chi2": chi2,
        "reduced_chi2": float(chi2 / dof),
        "dof": int(dof),
    }


def information_criteria(native: dict, n_params: int) -> dict[str, float]:
    """Return Gaussian information criteria for a candidate scale contract.

    The known-sigma form treats the supplied endpoint sigmas as calibrated.  The
    variance-fitted form is more conservative when absolute sigmas are still
    under audit and compares relative residual compression while estimating one
    global residual variance.  GROND uses these as diagnostics; final promotion
    should still require the holdout/strict residual gates to pass.
    """

    n = max(1, int(native.get("n", 0)))
    k = max(0, int(n_params))
    chi2 = max(float(native.get("chi2", float("nan"))), 1e-30)
    logn = math.log(max(float(n), 2.0))
    aic_known = chi2 + 2.0 * k
    bic_known = chi2 + logn * k
    aic_variance = float(n) * math.log(chi2 / float(n)) + 2.0 * k
    bic_variance = float(n) * math.log(chi2 / float(n)) + logn * k
    return {
        "aic_known_sigma": float(aic_known),
        "bic_known_sigma": float(bic_known),
        "aic_fitted_variance": float(aic_variance),
        "bic_fitted_variance": float(bic_variance),
    }
