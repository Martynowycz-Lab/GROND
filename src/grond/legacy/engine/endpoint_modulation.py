"""Audits for multiplicative corrected-HKL intensity corrections.

Corrected-HKL endpoints are free to choose one overall intensity scale.  A raw
percentile of the applied multiplier therefore confounds harmless global
rescaling with row-dependent modulation.  This module removes that gauge by
centering positive correction factors in log space, then reports the symmetric
fold change ``max(c/c0, c0/c)``.  The resulting statistic is invariant to any
common positive multiplier and treats an ``x``-fold increase and decrease
equally.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

MODULATION_METRIC = "scale_invariant_symmetric_fold_change_p95"
MODULATION_DEFINITION = (
    "95th percentile of max(c/c0, c0/c), where c is the positive multiplicative "
    "intensity correction actually applied to each unmerged endpoint row and c0 "
    "is its geometric median"
)


def _safe_exp(value: float) -> float:
    """Exponentiate a finite log value without raising on genuine overflow."""

    if not math.isfinite(value):
        return float("nan")
    if value > math.log(np.finfo(float).max):
        return float("inf")
    return float(math.exp(value))


def symmetric_fold_change_audit(
    corrections: Iterable[float] | np.ndarray,
    valid: Iterable[bool] | np.ndarray | None = None,
) -> dict[str, object]:
    """Return a scale-invariant audit of positive row correction factors.

    Invalid, zero, or negative factors are excluded from the numerical
    percentile but counted.  An endpoint with no positive finite factor is
    explicitly unavailable so downstream publication gates can fail closed.
    """

    values = np.asarray(corrections, dtype=float).reshape(-1)
    mask = np.isfinite(values) & (values > 0.0)
    if valid is not None:
        requested = np.asarray(valid, dtype=bool).reshape(-1)
        if requested.shape != values.shape:
            raise ValueError(
                "multiplicative-correction audit mask shape does not match corrections: "
                f"{requested.shape} != {values.shape}"
            )
        mask &= requested

    selected = values[mask]
    base: dict[str, object] = {
        "status": "ok" if selected.size else "unavailable_no_positive_finite_corrections",
        "metric": MODULATION_METRIC,
        "definition": MODULATION_DEFINITION,
        "normalization": "geometric_median",
        "rows_total": int(values.size),
        "rows_valid_positive": int(selected.size),
        "rows_excluded": int(values.size - selected.size),
    }
    if not selected.size:
        return {
            **base,
            "normalization_factor": None,
            "normalized_correction_p05": None,
            "normalized_correction_median": None,
            "normalized_correction_p95": None,
            "fold_change_median": None,
            "fold_change_p95": None,
            "fold_change_max": None,
        }

    logs = np.log(selected)
    center = float(np.median(logs))
    centered = logs - center
    abs_centered = np.abs(centered)
    normalized_quantiles = np.percentile(centered, [5.0, 50.0, 95.0])
    fold_quantiles = np.percentile(abs_centered, [50.0, 95.0, 100.0])
    return {
        **base,
        "normalization_factor": _safe_exp(center),
        "normalized_correction_p05": _safe_exp(float(normalized_quantiles[0])),
        "normalized_correction_median": _safe_exp(float(normalized_quantiles[1])),
        "normalized_correction_p95": _safe_exp(float(normalized_quantiles[2])),
        "fold_change_median": _safe_exp(float(fold_quantiles[0])),
        "fold_change_p95": _safe_exp(float(fold_quantiles[1])),
        "fold_change_max": _safe_exp(float(fold_quantiles[2])),
    }


def modulation_manifest_fields(
    corrections: Iterable[float] | np.ndarray,
    valid: Iterable[bool] | np.ndarray | None = None,
) -> dict[str, object]:
    """Return canonical top-level and detailed manifest fields for a gate."""

    audit = symmetric_fold_change_audit(corrections, valid)
    value = audit.get("fold_change_p95")
    return {
        "modulation_metric": MODULATION_METRIC,
        "modulation_p95": value,
        "mod_p95": value,
        "modulation_audit": audit,
    }
