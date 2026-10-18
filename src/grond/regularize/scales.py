"""SHELXL weights and native residual metrics of scaled row predictions. A lower fitted residual is not proof of a better structure; use the existing matched controls and endpoint checks."""

from __future__ import annotations

import numpy as np


def shelx_weights(obs, sig, pred, a: float | np.ndarray, b: float | np.ndarray):
    """Return SHELXL WGHT weights 1 / (sigma^2 + (a P)^2 + b P), P = (max(Iobs, 0) + 2 max(Icalc, 0)) / 3.

    a and b may be per-row arrays.
    """

    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 / np.maximum(sig * sig + (a * p) ** 2 + b * p, 1e-30)


def native_metrics(obs, sig, pred, mask, wght_a: float, wght_b: float, n_params: int = 0):
    """Return SHELXL-style statistics of pred on the masked rows: R1, R2, wR2, GooF, CC and residual summaries.

    R values are fractions; R1_all and R1_gt (obs > 2 sigma) are on sqrt(I), R2_all and wR2_all on I, with SHELXL
    WGHT weights. A zero or nonfinite normalization denominator gives an undefined
    R factor (NaN). GooF uses dof = n - n_params (at least 1).
    """

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

    def ratio(numerator, denominator):
        """Return a normalized residual only when its normalization is defined."""

        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
            return float("nan")
        return float(numerator / denominator)

    def r1(local):
        """Return sum |Fobs - Fcalc| / sum Fobs over the local rows, NaN when there are none."""

        if not np.any(local):
            return float("nan")
        return ratio(np.sum(np.abs(fobs[local] - fpred[local])), np.sum(fobs[local]))

    w = shelx_weights(obs_m, sig_m, pred_m, wght_a, wght_b)
    res = obs_m - pred_m
    wr2 = float(np.sqrt(ratio(np.sum(w * res * res), np.sum(w * obs_m * obs_m))))
    dof = max(1, len(obs_m) - int(n_params))
    goof = float(np.sqrt(np.sum(w * res * res) / dof))
    r2 = ratio(np.sum(np.abs(res)), np.sum(np.abs(obs_m)))
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
