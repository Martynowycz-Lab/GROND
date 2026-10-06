"""Keep DIALS row acceptance separate from intensity completion and sigma choices."""

from __future__ import annotations

import numpy as np


def _choose_intensity(
    arrays: dict[str, np.ndarray],
    mode: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return I, sigma, RLP-proxy, PEAK-proxy, and a plain-English mode note."""

    partiality = np.asarray(arrays["partiality"], dtype=float)
    lp = np.asarray(arrays["lp"], dtype=float)
    qe = np.asarray(arrays.get("qe", np.ones_like(lp)), dtype=float)
    if qe.shape != lp.shape:
        raise ValueError("DIALS qe column must have the same shape as lp")
    correction = np.full_like(lp, np.nan, dtype=float)
    valid_correction = np.isfinite(lp) & (lp > 0.0) & np.isfinite(qe) & (qe > 0.0)
    correction[valid_correction] = lp[valid_correction] / qe[valid_correction]
    peak = np.clip(partiality * 100.0, 0.0, 999.0)
    if mode == "sum_complete":
        factor = correction / np.maximum(partiality, eps)
        iobs = np.asarray(arrays["i_sum"], dtype=float) * factor
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_sum"], dtype=float), 0.0)) * np.abs(factor)
        note = "intensity.sum.value multiplied by (lp/qe)/partiality; PEAK=100*partiality and RLP=lp/qe"
        return iobs, sigma, correction, peak, note
    if mode == "sum_raw":
        iobs = np.asarray(arrays["i_sum"], dtype=float)
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_sum"], dtype=float), 0.0))
        note = "raw intensity.sum.value with no lp/partiality completion; PEAK=100*partiality"
        return iobs, sigma, np.ones_like(iobs), peak, note
    if mode == "scale":
        iobs = np.asarray(arrays["i_scale"], dtype=float)
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_scale"], dtype=float), 0.0))
        note = "DIALS intensity.scale.value from scaled.refl; PEAK still records 100*partiality"
        return iobs, sigma, np.ones_like(iobs), peak, note
    raise ValueError(f"unknown DIALS intensity mode {mode!r}")


def _bool_array(arrays: dict[str, np.ndarray], name: str, n: int, default: bool = False) -> np.ndarray:
    """Read a DIALS flag column; only an absent optional column may use its default."""

    if name not in arrays:
        return np.full(n, default, dtype=bool)
    arr = np.asarray(arrays[name], dtype=bool)
    if arr.shape != (n,):
        raise ValueError(f"DIALS {name} flag must have one value per reflection")
    return arr


def _dials_row_selection_keep(
    arrays: dict[str, np.ndarray],
    n: int,
    row_selection: str,
) -> np.ndarray:
    """Select summation-integrated rows, optionally restricted to accepted scaling rows.

    A ``scaled.refl`` table normally retains the original summation columns as
    well as the ``scaled`` flag.  ``row_selection='scaled'`` therefore permits
    GROND to use raw summation intensities while respecting the observations
    accepted by ``dials.scale`` (including delta-CC1/2 or other scaling
    exclusions).  This is intentionally independent of ``intensity_mode``.
    """

    selection = str(row_selection or "integrated").strip().lower()
    if selection not in {"integrated", "scaled"}:
        raise ValueError(f"unknown DIALS row selection {row_selection!r}")
    if "integrated_sum" not in arrays:
        raise ValueError("DIALS row selection requires the integrated_sum flag")
    integrated_sum = _bool_array(arrays, "integrated_sum", n, default=False)
    keep = integrated_sum.copy()
    if selection == "scaled":
        scaled = _bool_array(arrays, "scaled", n, default=False)
        if not np.any(scaled):
            raise ValueError(
                "DIALS row-selection 'scaled' was requested, but the reflection "
                "table contains no rows flagged as accepted by scaling. Use a "
                "scaled.refl table or select 'integrated'."
            )
        keep &= scaled
    return keep


def _dials_sigma_preflight_keep(
    arrays: dict[str, np.ndarray],
    iobs: np.ndarray,
    sigma: np.ndarray,
    mode: str,
    row_selection: str = "integrated",
) -> np.ndarray:
    """Rows used to calibrate DIALS sigmas from equivalent-observation scatter."""

    n = len(iobs)
    partiality = np.asarray(arrays.get("partiality", np.ones(n, dtype=float)), dtype=float)
    if len(partiality) != n:
        partiality = np.ones(n, dtype=float)
    keep = np.isfinite(iobs) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(partiality) & (partiality >= 0.90)
    effective_selection = "scaled" if str(mode).strip().lower() == "scale" else row_selection
    keep &= _dials_row_selection_keep(arrays, n, effective_selection)
    keep &= ~_bool_array(arrays, "dont_integrate", n)
    for flag in (
        "overloaded",
        "in_powder_ring",
        "background_includes_bad_pixels",
        "foreground_includes_bad_pixels",
        "includes_bad_pixels",
        "bad_shoebox",
        "failed_during_background_modelling",
        "failed_during_summation",
    ):
        keep &= ~_bool_array(arrays, flag, n)
    return keep
