"""Post-Bloch endpoint uncertainty calibration.

The corrected-HKL endpoint is a model-derived target: raw integration
uncertainties have passed through Bloch regularization, row scales, and endpoint
merging before SHELXL sees them.  This module applies one deliberately simple
calibration for that endpoint: fit the current model scale, measure the
root-mean-square normalized residual, and multiply every endpoint sigma by that
single number.

The calibration is not a new claim about detector counting noise.  It says that
the corrected endpoint is a transformed observation, and that the reported
weighted residuals should use a declared endpoint uncertainty scale rather than
blindly inheriting the integration sigmas.  It is not used to choose thickness,
absorption, geometry, hand, or resolution.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

try:  # Package import.
    from .bloch_engine import fcalc_res_grid
except ImportError:  # Direct script execution from this directory.
    from bloch_engine import fcalc_res_grid


HklRow = tuple[int, int, int, float, float]


def scale_row_sigmas(rows: Iterable[Sequence[float]], sigma_scale: float) -> list[HklRow]:
    """Return HKLF rows with intensities unchanged and sigmas scaled uniformly."""

    scale = float(sigma_scale)
    return [
        (int(h), int(k), int(l), float(iobs), float(sig) * scale)
        for h, k, l, iobs, sig in rows
    ]


def _deterministic_fraction(h: int, k: int, l: int, seed: int) -> float:
    """Return a reproducible pseudo-random fraction in [0, 1) for one HKL and seed (BLAKE2b hash)."""

    text = f"{int(seed)}:{int(h)}:{int(k)}:{int(l)}".encode("ascii")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _rms(values: np.ndarray) -> float:
    """Return the RMS of the finite values, or NaN when there are none."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _median_abs(values: np.ndarray) -> float:
    """Return the median absolute finite value, or NaN when there are none."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values)))


def _fit_model_intensity_scale(iobs: np.ndarray, sigma: np.ndarray, fc2: np.ndarray) -> float:
    """Fit Iobs ~= scale * Fc^2 with inverse-variance weights."""

    w = 1.0 / np.maximum(sigma * sigma, 1e-30)
    denom = float(np.sum(w * fc2 * fc2))
    if denom > 0.0 and math.isfinite(denom):
        scale = float(np.sum(w * iobs * fc2) / denom)
        if math.isfinite(scale) and scale > 0.0:
            return scale
    mask = (fc2 > 0.0) & np.isfinite(iobs) & np.isfinite(fc2)
    if np.any(mask):
        scale = float(np.nanmedian(iobs[mask] / np.maximum(fc2[mask], 1e-30)))
        if math.isfinite(scale) and scale > 0.0:
            return scale
    return 1.0


def fit_uniform_z_rms_sigma_scale(
    rows: Sequence[Sequence[float]],
    model: dict[str, Any],
    *,
    holdout_fraction: float = 0.20,
    seed: int = 20260603,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    min_rows: int = 20,
) -> dict[str, Any]:
    """Fit one endpoint sigma multiplier from current-model residual RMS(z).

    The fitted residual is ``z = (Iobs - K*Fc^2) / sigma`` where ``K`` is a
    single least-squares model scale.  A deterministic HKL hash split is used
    for the scale diagnostic when enough reflections are available; the final
    multiplier is still one global scalar applied to every endpoint sigma.
    """

    if not rows:
        return {"policy": "uniform_z_rms", "status": "skipped_no_rows"}

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return {"policy": "uniform_z_rms", "status": "skipped_bad_rows", "row_count": int(len(rows))}

    h = arr[:, 0].astype(int)
    k = arr[:, 1].astype(int)
    l = arr[:, 2].astype(int)
    iobs = arr[:, 3].astype(float)
    sigma = arr[:, 4].astype(float)
    fcalc = fcalc_res_grid(model, h, k, l)
    fc2 = np.abs(fcalc) ** 2
    valid = (
        np.isfinite(iobs)
        & np.isfinite(sigma)
        & np.isfinite(fc2)
        & (sigma > 0.0)
        & (fc2 >= 0.0)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count < int(min_rows):
        return {
            "policy": "uniform_z_rms",
            "status": "skipped_too_few_rows",
            "row_count": int(len(rows)),
            "valid_rows": valid_count,
            "min_rows": int(min_rows),
        }

    hv, kv, lv = h[valid], k[valid], l[valid]
    iobs_v = iobs[valid]
    sigma_v = sigma[valid]
    fc2_v = fc2[valid]
    fractions = np.asarray([
        _deterministic_fraction(int(hh), int(kk), int(ll), int(seed))
        for hh, kk, ll in zip(hv, kv, lv, strict=False)
    ])
    holdout_fraction = float(np.clip(holdout_fraction, 0.0, 0.8))
    test_mask = fractions < holdout_fraction
    train_mask = ~test_mask
    if int(np.count_nonzero(train_mask)) < int(min_rows):
        train_mask = np.ones(valid_count, dtype=bool)
        test_mask = np.zeros(valid_count, dtype=bool)
    elif int(np.count_nonzero(test_mask)) < max(5, int(min_rows // 4)):
        test_mask = np.zeros(valid_count, dtype=bool)

    model_scale = _fit_model_intensity_scale(iobs_v[train_mask], sigma_v[train_mask], fc2_v[train_mask])
    z = (iobs_v - model_scale * fc2_v) / np.maximum(sigma_v, 1e-30)
    z_train = z[train_mask]
    z_test = z[test_mask] if np.any(test_mask) else np.asarray([], dtype=float)
    raw_scale = _rms(z_train)
    if not math.isfinite(raw_scale) or raw_scale <= 0.0:
        return {
            "policy": "uniform_z_rms",
            "status": "skipped_bad_rms",
            "row_count": int(len(rows)),
            "valid_rows": valid_count,
            "model_intensity_scale": model_scale,
        }
    sigma_scale = float(np.clip(raw_scale, float(min_scale), float(max_scale)))
    clipped = abs(sigma_scale - raw_scale) > max(1e-12, 1e-9 * abs(raw_scale))
    out = {
        "policy": "uniform_z_rms",
        "status": "ok_clipped" if clipped else "ok",
        "row_count": int(len(rows)),
        "valid_rows": valid_count,
        "fit_rows": int(np.count_nonzero(train_mask)),
        "heldout_rows": int(np.count_nonzero(test_mask)),
        "holdout_fraction": holdout_fraction,
        "seed": int(seed),
        "model_intensity_scale": model_scale,
        "sigma_scale_raw": raw_scale,
        "sigma_scale": sigma_scale,
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "train_rms_z_before": _rms(z_train),
        "train_median_abs_z_before": _median_abs(z_train),
        "train_rms_z_after": _rms(z_train / sigma_scale),
        "train_median_abs_z_after": _median_abs(z_train / sigma_scale),
        "all_rms_z_before": _rms(z),
        "all_median_abs_z_before": _median_abs(z),
        "all_rms_z_after": _rms(z / sigma_scale),
        "all_median_abs_z_after": _median_abs(z / sigma_scale),
    }
    if z_test.size:
        out.update(
            {
                "heldout_rms_z_before": _rms(z_test),
                "heldout_median_abs_z_before": _median_abs(z_test),
                "heldout_rms_z_after": _rms(z_test / sigma_scale),
                "heldout_median_abs_z_after": _median_abs(z_test / sigma_scale),
            }
        )
    return out


# ----------------------------------------------------------------------------- per-crystal error model
# JANA-style error model for the SHELXL/merge sigmas of one crystal:
#     sigma'^2 = k^2 (sigma^2 + (u P)^2),   P = (max(Iobs, 0) + 2 K Fc^2) / 3,
# fitted on work rows against the current model (K: one least-squares model scale).  u is chosen so that
# the robust RMS of z = (Iobs - K Fc^2) / sqrt(sigma^2 + (u P)^2) is flat across intensity bins, and k so
# that it is 1 overall.  One uniform factor (uniform_z_rms) cannot do this: the extra model error of
# corrected electron intensities grows with the intensity, so a single factor large enough for the strong
# rows over-inflates the weak rows (for example SHELXL GooF 0.3 for the weak and 1.0 for the strong rows of
# one endpoint).  P is SHELXL's weighting intensity, which keeps a noisy Iobs from choosing its own weight.
ERROR_MODEL_HUBER_C = 3.0
ERROR_MODEL_U_MAX = 1.0
ERROR_MODEL_BINS = 10
ERROR_MODEL_SCALE_ROUNDS = 3


def _huber_beta(c: float) -> float:
    """E[min(Z^2, c^2)] for a standard normal Z: the consistency constant of the robust RMS."""

    tail = 1.0 - math.erf(c / math.sqrt(2.0))
    inner = math.erf(c / math.sqrt(2.0)) - 2.0 * c * math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)
    return inner + c * c * tail


def robust_rms(values: np.ndarray, c: float = ERROR_MODEL_HUBER_C) -> float:
    """Huber M-estimate of scale: s with mean(min((z/s)^2, c^2)) = E[min(Z^2, c^2)] for Z ~ N(0, 1).

    It equals the RMS for Gaussian z and is not dominated by a few gross outliers.
    """

    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    beta = _huber_beta(float(c))
    scale = 1.4826 * float(np.median(np.abs(z)))
    if not (math.isfinite(scale) and scale > 0.0):
        scale = _rms(z)
    if not (math.isfinite(scale) and scale > 0.0):
        return float("nan")
    for _ in range(200):
        ratio = np.minimum((z / scale) ** 2, float(c) * float(c))
        new = scale * math.sqrt(float(np.mean(ratio)) / beta)
        if abs(new - scale) <= 1e-12 * max(scale, 1e-300):
            scale = new
            break
        scale = new
    return float(scale)


def _intensity_bin_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Return quantile edges that split p into n_bins bins, with open outer edges."""

    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(p, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bin_index(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the bin number of each p for the given edges."""

    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)


def _bin_table(z: np.ndarray, p: np.ndarray, edges: np.ndarray) -> list[dict[str, Any]]:
    """Return the row count, P range, RMS and robust RMS of z in each intensity bin."""

    index = _bin_index(p, edges)
    table = []
    for b in range(len(edges) - 1):
        mask = index == b
        values = z[mask]
        table.append(
            {
                "bin": int(b),
                "n": int(np.count_nonzero(mask)),
                "P_min": float(np.min(p[mask])) if np.any(mask) else float("nan"),
                "P_max": float(np.max(p[mask])) if np.any(mask) else float("nan"),
                "rms_z": _rms(values),
                "robust_rms_z": robust_rms(values),
            }
        )
    return table


def _flatness(z: np.ndarray, p: np.ndarray, edges: np.ndarray, min_bin_rows: int) -> float:
    """Row-weighted mean squared log of the per-bin robust RMS relative to the overall robust RMS."""

    overall = robust_rms(z)
    index = _bin_index(p, edges)
    total, weight = 0.0, 0
    for b in range(len(edges) - 1):
        values = z[index == b]
        if values.size < int(min_bin_rows):
            continue
        s = robust_rms(values)
        if math.isfinite(s) and s > 0.0 and math.isfinite(overall) and overall > 0.0:
            total += values.size * math.log(s / overall) ** 2
            weight += values.size
    return total / weight if weight else float("inf")


def _error_model_arrays(rows: Sequence[Sequence[float]], model: dict[str, Any]):
    """Return the row array, Iobs, sigma, |Fcalc|² and the validity mask used by the error-model fit."""

    arr = np.asarray(rows, dtype=float).reshape(-1, 5) if len(rows) else np.zeros((0, 5))
    h, k, l = (arr[:, i].astype(int) for i in range(3))
    iobs = arr[:, 3].astype(float)
    sigma = arr[:, 4].astype(float)
    fc2 = np.abs(fcalc_res_grid(model, h, k, l)) ** 2 if len(arr) else np.zeros(0)
    valid = np.isfinite(iobs) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(fc2) & (fc2 >= 0.0)
    return arr, iobs, sigma, fc2, valid


def fit_error_model_sigma(
    rows: Sequence[Sequence[float]],
    model: dict[str, Any],
    *,
    free_rows: Sequence[Sequence[float]] = (),
    n_bins: int = ERROR_MODEL_BINS,
    min_rows: int = 20,
    u_max: float = ERROR_MODEL_U_MAX,
    label: str = "",
) -> tuple[list[HklRow], dict[str, Any]]:
    """Fit sigma'^2 = k^2 (sigma^2 + (u P)^2) on one crystal's work rows and return the recalibrated rows.

    ``rows`` are the crystal's corrected work rows (h, k, l, I, sigma) that go into the SHELXL HKL; they are
    the only rows the fit reads.  ``free_rows`` (never written, never fitted) are scored with the fitted
    (K, k, u) and reported as the held-out RMS(z).  Fails closed (ValueError) when too few rows are usable.
    """

    where = f" for {label}" if label else ""
    arr, iobs, sigma, fc2, valid = _error_model_arrays(rows, model)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < int(min_rows):
        raise ValueError(
            f"error-model sigma fit{where} needs at least {int(min_rows)} work rows with finite I, sigma > 0 and "
            f"|Fcalc|^2, found {n_valid}; use --dataset-sigma-policy uniform_z_rms or none for this run"
        )
    if not np.all(valid):
        raise ValueError(
            f"error-model sigma fit{where}: {int(len(valid) - n_valid)} endpoint rows have non-finite values or "
            "non-positive sigma; they cannot be written to a SHELXL HKL"
        )
    bins = max(1, min(int(n_bins), n_valid // 20))
    min_bin_rows = max(5, n_valid // (4 * bins))
    grid = np.concatenate([[0.0], np.logspace(-4.0, math.log10(float(u_max)), 41)])
    weights_sigma = sigma
    scale_history = []
    # The model scale K is refitted with the current error-model weights (SHELXL fits its OSF with its
    # weights too); three rounds settle K, u and k to well below their reported precision.
    for _round in range(ERROR_MODEL_SCALE_ROUNDS):
        scale = _fit_model_intensity_scale(iobs, weights_sigma, fc2)
        scale_history.append(float(scale))
        resid = iobs - scale * fc2
        p = (np.maximum(iobs, 0.0) + 2.0 * scale * fc2) / 3.0
        edges = _intensity_bin_edges(p, bins)

        def objective(u: float, resid=resid, p=p, edges=edges) -> float:
            """Return the bin flatness of z for one u, with this round's residuals, P and bins."""

            # This round's residuals, weighting intensities and bins (the last round's after the loop).
            z = resid / np.sqrt(sigma * sigma + (u * p) ** 2)
            return _flatness(z, p, edges, min_bin_rows)

        values = np.asarray([objective(float(u)) for u in grid], dtype=float)
        best = int(np.nanargmin(values))
        lo = float(grid[max(best - 1, 0)])
        hi = float(grid[min(best + 1, len(grid) - 1)])
        u_best, j_best = float(grid[best]), float(values[best])
        if hi > lo:
            from scipy.optimize import minimize_scalar

            refined = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
            if refined.success and math.isfinite(float(refined.fun)) and float(refined.fun) <= j_best:
                u_best, j_best = float(refined.x), float(refined.fun)
        weights_sigma = np.sqrt(sigma * sigma + (u_best * p) ** 2)
    base = np.sqrt(sigma * sigma + (u_best * p) ** 2)
    k_best = robust_rms(resid / base)
    if not (math.isfinite(k_best) and k_best > 0.0):
        raise ValueError(f"error-model sigma fit{where} produced a non-positive k ({k_best!r})")
    new_sigma = k_best * base
    new_rows = [
        (int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(s))
        for row, s in zip(arr, new_sigma, strict=True)
    ]
    z_raw = resid / sigma
    uniform = robust_rms(z_raw)
    z_after = resid / new_sigma
    meta: dict[str, Any] = {
        "policy": "error_model",
        "status": "ok",
        "definition": "sigma'^2 = k^2 (sigma^2 + (u P)^2), P = (max(Iobs,0) + 2 K Fc^2)/3; fitted on work rows",
        "fit_rows": n_valid,
        "model_intensity_scale": float(scale),
        "model_intensity_scale_rounds": scale_history,
        "k": float(k_best),
        "u": float(u_best),
        "u_bounds": [0.0, float(u_max)],
        "u_at_upper_bound": bool(u_best >= float(u_max) * (1.0 - 1e-6)),
        "flatness_objective": float(j_best),
        "flatness_objective_u0": float(objective(0.0)),
        "robust_rms_c": ERROR_MODEL_HUBER_C,
        "n_intensity_bins": int(bins),
        "work_rms_z_raw": _rms(z_raw),
        "work_robust_rms_z_raw": float(uniform),
        "work_rms_z_after": _rms(z_after),
        "work_robust_rms_z_after": robust_rms(z_after),
        "work_median_abs_z_after": _median_abs(z_after),
        "work_bins_uniform_z_rms": _bin_table(z_raw / uniform, p, edges),
        "work_bins_error_model": _bin_table(z_after, p, edges),
        "free_rows": 0,
    }
    if len(free_rows):
        _farr, f_iobs, f_sigma, f_fc2, f_valid = _error_model_arrays(free_rows, model)
        f_iobs, f_sigma, f_fc2 = f_iobs[f_valid], f_sigma[f_valid], f_fc2[f_valid]
        f_resid = f_iobs - scale * f_fc2
        f_p = (np.maximum(f_iobs, 0.0) + 2.0 * scale * f_fc2) / 3.0
        f_sigma_new = k_best * np.sqrt(f_sigma * f_sigma + (u_best * f_p) ** 2)
        f_z = f_resid / f_sigma_new
        meta.update(
            {
                "free_rows": int(f_iobs.size),
                "free_rms_z_raw": _rms(f_resid / f_sigma),
                "free_rms_z_uniform": _rms(f_resid / f_sigma / uniform),
                "free_rms_z_after": _rms(f_z),
                "free_robust_rms_z_after": robust_rms(f_z),
                "free_median_abs_z_after": _median_abs(f_z),
                "free_bins_error_model": _bin_table(f_z, f_p, edges),
                "free_role": "reported only; free rows never enter the fit or any SHELXL HKL",
            }
        )
    return new_rows, meta


def apply_error_model_sigma(
    rows: Sequence[Sequence[float]],
    model: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[HklRow], dict[str, Any]]:
    """Give other rows of one crystal the sigmas of a fitted error model; nothing is fitted.

    ``meta`` is the record of ``fit_error_model_sigma`` (K = model_intensity_scale, k, u) and ``model`` the model it
    was fitted against.  sigma'^2 = k^2 (sigma^2 + (u P)^2) with P = (max(I, 0) + 2 K Fc^2) / 3, the same arithmetic
    as the fit, so the fitted work rows come back bit for bit.  Rows without finite I, sigma > 0 and |Fcalc|^2 are
    dropped and counted; the RMS(z) of z = (I - K Fc^2) / sigma' is reported.
    """

    if str(meta.get("status", "")) != "ok" or meta.get("policy") != "error_model":
        raise ValueError(f"no fitted error model to apply (status {meta.get('status')!r})")
    arr, iobs, sigma, fc2, valid = _error_model_arrays(rows, model)
    scale, k, u = float(meta["model_intensity_scale"]), float(meta["k"]), float(meta["u"])
    iobs, sigma, fc2, kept = iobs[valid], sigma[valid], fc2[valid], arr[valid]
    p = (np.maximum(iobs, 0.0) + 2.0 * scale * fc2) / 3.0
    new_sigma = k * np.sqrt(sigma * sigma + (u * p) ** 2)
    z = (iobs - scale * fc2) / new_sigma
    new_rows = [
        (int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(s))
        for row, s in zip(kept, new_sigma, strict=True)
    ]
    return new_rows, {
        "rows": int(iobs.size),
        "rows_dropped_invalid": int(np.count_nonzero(~valid)),
        "rms_z": _rms(z),
        "robust_rms_z": robust_rms(z) if z.size else float("nan"),
        "median_abs_z": _median_abs(z),
        "error_model": {"k": k, "u": u, "model_intensity_scale": scale},
    }


def rms_z_against_model(
    rows: Sequence[Sequence[float]], model: dict[str, Any], model_intensity_scale: float
) -> dict[str, Any]:
    """RMS of z = (I - K Fc^2) / sigma of rows whose sigmas are final, with a K fitted elsewhere (nothing fitted)."""

    _arr, iobs, sigma, fc2, valid = _error_model_arrays(rows, model)
    z = (iobs[valid] - float(model_intensity_scale) * fc2[valid]) / sigma[valid]
    return {
        "rows": int(z.size),
        "rows_dropped_invalid": int(np.count_nonzero(~valid)),
        "rms_z": _rms(z),
        "robust_rms_z": robust_rms(z) if z.size else float("nan"),
        "median_abs_z": _median_abs(z),
        "model_intensity_scale": float(model_intensity_scale),
    }
