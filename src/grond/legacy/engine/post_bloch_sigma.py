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
absorption, geometry, mirror polarity, or resolution.
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
    text = f"{int(seed)}:{int(h)}:{int(k)}:{int(l)}".encode("ascii")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _median_abs(values: np.ndarray) -> float:
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
        for hh, kk, ll in zip(hv, kv, lv)
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
