"""Optional DIALS variance diagnostics; production preserves the supplied variances."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy.optimize import nnls

from grond.io.dials_selection import _dials_sigma_preflight_keep


def _detector_gain(metadata: dict[str, Any]) -> float:
    """Return the DIALS detector gain if it is available and physically usable."""

    detector = metadata.get("detector") or {}
    try:
        gain = float(detector.get("gain"))
    except Exception:
        return 1.0
    if not math.isfinite(gain) or gain <= 0:
        return 1.0
    return gain


def _finite_i_over_sigma(iobs: np.ndarray, sigma: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return finite I/sigma values for summary and bridge policy decisions."""

    mask = keep & np.isfinite(iobs) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(mask):
        return np.asarray([], dtype=float)
    return np.asarray(iobs[mask], dtype=float) / np.asarray(sigma[mask], dtype=float)


def _space_group_rotations(metadata: dict[str, Any]) -> list[np.ndarray]:
    """Return integer reciprocal-space rotation matrices from DIALS metadata."""

    try:
        import gemmi
    except Exception:
        return [np.eye(3, dtype=int)]

    name = str(metadata.get("space_group") or "").strip()
    sg = None
    if name:
        try:
            sg = gemmi.find_spacegroup_by_name(name)
        except Exception:
            sg = None
    if sg is None and name:
        match = re.search(r"No\.\s*(\d+)", name)
        if match:
            try:
                sg = gemmi.find_spacegroup_by_number(int(match.group(1)))
            except Exception:
                sg = None
    if sg is None:
        return [np.eye(3, dtype=int)]

    rotations: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for op in sg.operations():
        rot = np.rint(np.asarray(op.rot, dtype=float) / float(op.DEN)).astype(int)
        key = tuple(int(v) for v in rot.reshape(-1))
        if key not in seen:
            seen.add(key)
            rotations.append(rot)
    return rotations or [np.eye(3, dtype=int)]


def _canonical_reciprocal_hkl(hkl: np.ndarray, rotations: list[np.ndarray]) -> tuple[int, int, int]:
    """Canonical key for symmetry-equivalent observations, without adding Friedel mates."""

    row = np.asarray(hkl, dtype=int)
    keys: list[tuple[int, int, int]] = []
    for rot in rotations:
        for mat in (rot, np.linalg.inv(rot)):
            equiv = np.rint(row @ mat).astype(int)
            keys.append(tuple(int(v) for v in equiv))
    return min(set(keys))


def _fit_empirical_sigma_model(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    iobs: np.ndarray,
    sigma: np.ndarray,
    *,
    mode: str,
    row_selection: str = "integrated",
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Fit sigma_eff^2 = m^2 sigma_raw^2 + (a Iobs)^2 + c^2 from duplicate scatter."""

    hkl = np.asarray(arrays.get("hkl", np.empty((0, 3), dtype=int)), dtype=int)
    if hkl.ndim != 2 or hkl.shape[1] != 3 or len(hkl) != len(iobs):
        return None

    keep = _dials_sigma_preflight_keep(arrays, iobs, sigma, mode, row_selection=row_selection)
    if int(np.count_nonzero(keep)) < 50:
        return None

    rotations = _space_group_rotations(metadata)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for idx in np.where(keep)[0]:
        groups.setdefault(_canonical_reciprocal_hkl(hkl[idx], rotations), []).append(int(idx))

    features: list[list[float]] = []
    target: list[float] = []
    usable_groups = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        usable_groups += 1
        y = np.asarray([iobs[i] for i in indices], dtype=float)
        s = np.asarray([sigma[i] for i in indices], dtype=float)
        mean = float(np.mean(y))
        leverage = max(0.0, 1.0 - 1.0 / float(len(indices)))
        for yy, ss in zip(y, s, strict=True):
            features.append([float(ss * ss * leverage), float(yy * yy * leverage), float(leverage)])
            target.append(float((yy - mean) ** 2))

    dof = len(target) - usable_groups
    if usable_groups < 20 or dof < 30:
        return None

    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    coeff, _ = nnls(x, y)
    coeff[0] = max(float(coeff[0]), 1.0)
    coeff[1] = max(float(coeff[1]), 0.0)
    coeff[2] = max(float(coeff[2]), 0.0)

    fitted_var = coeff[0] * sigma * sigma + coeff[1] * iobs * iobs + coeff[2]
    sigma_eff = np.sqrt(np.maximum(fitted_var, sigma * sigma))

    fit_rows = np.any(x > 0, axis=1)
    equivalent_multiplier = math.sqrt(float(np.sum(x[fit_rows] @ coeff) / max(np.sum(x[fit_rows, 0]), 1e-30)))
    ratios = sigma_eff[keep] / np.maximum(sigma[keep], 1e-30)
    summary = {
        "status": "ok",
        "equation": "sigma_eff^2 = m^2*sigma_raw^2 + (a*Iobs)^2 + c^2",
        "source": "auto_empirical_replicate_variance",
        "groups": int(len(groups)),
        "usable_groups": int(usable_groups),
        "fit_rows": int(len(target)),
        "dof": int(dof),
        "space_group": str(metadata.get("space_group") or ""),
        "m": float(math.sqrt(coeff[0])),
        "a": float(math.sqrt(coeff[1])),
        "c": float(math.sqrt(coeff[2])),
        "coefficient_sigma2": float(coeff[0]),
        "coefficient_iobs2": float(coeff[1]),
        "coefficient_constant": float(coeff[2]),
        "equivalent_global_multiplier": float(equivalent_multiplier),
        "median_sigma_ratio": float(np.nanmedian(ratios)) if len(ratios) else None,
        "p90_sigma_ratio": float(np.nanpercentile(ratios, 90)) if len(ratios) else None,
    }
    return sigma_eff, summary


def _dials_sigma_multiplier(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    iobs: np.ndarray,
    sigma: np.ndarray,
    *,
    mode: str,
    row_selection: str = "integrated",
    sigma_mode: str,
    sigma_scale: float | None,
) -> tuple[float, dict[str, Any]]:
    """Choose a DIALS sigma multiplier and explain the decision.

    DIALS stores detector gain on the detector model, but rotation-data
    summation variances can still be too optimistic for electron diffraction.
    Auto mode first fits an empirical replicate-scatter variance model.  If
    there are not enough equivalent rows, it falls back to the older conservative
    detector-gain gross-overprecision gate.
    """

    requested = str(sigma_mode or "auto").strip().lower()
    if requested not in {"auto", "raw", "detector_gain"}:
        raise ValueError(f"unknown DIALS sigma mode {sigma_mode!r}")
    if sigma_scale is not None and float(sigma_scale) > 0:
        mult = float(sigma_scale)
        source = "explicit_sigma_scale"
    elif requested == "raw":
        mult = 1.0
        source = "raw"
    else:
        gain = _detector_gain(metadata)
        partiality = np.asarray(arrays["partiality"], dtype=float)
        keep = np.isfinite(partiality) & (partiality >= 0.90)
        ios_raw = _finite_i_over_sigma(iobs, sigma, keep)
        median_ios = float(np.nanmedian(ios_raw)) if len(ios_raw) else None
        p95_ios = float(np.nanpercentile(ios_raw, 95)) if len(ios_raw) else None
        if requested == "detector_gain":
            mult = math.sqrt(gain)
            source = "detector_gain"
        elif empirical := _fit_empirical_sigma_model(
            arrays, metadata, iobs, sigma, mode=mode, row_selection=row_selection
        ):
            sigma_eff, empirical_summary = empirical
            mult = float(empirical_summary["equivalent_global_multiplier"])
            source = str(empirical_summary["source"])
        elif gain > 1.01 and median_ios is not None and p95_ios is not None and median_ios > 50.0 and p95_ios > 250.0:
            mult = math.sqrt(gain)
            source = "auto_detector_gain_gross_overprecision"
        else:
            mult = 1.0
            source = "auto_raw"
    partiality = np.asarray(arrays.get("partiality", np.ones(len(iobs), dtype=float)), dtype=float)
    if len(partiality) != len(iobs):
        partiality = np.ones(len(iobs), dtype=float)
    keep = np.isfinite(partiality) & (partiality >= 0.90)
    sigma_after = sigma_eff if "sigma_eff" in locals() else sigma * mult
    ios_before = _finite_i_over_sigma(iobs, sigma, keep)
    ios_after = _finite_i_over_sigma(iobs, sigma_after, keep)
    summary = {
        "mode": requested,
        "source": source,
        "detector_gain": _detector_gain(metadata),
        "sigma_multiplier": float(mult),
        "explicit_sigma_scale": float(sigma_scale) if sigma_scale is not None and float(sigma_scale) > 0 else None,
        "i_over_sigma_before_median": float(np.nanmedian(ios_before)) if len(ios_before) else None,
        "i_over_sigma_after_median": float(np.nanmedian(ios_after)) if len(ios_after) else None,
        "i_over_sigma_before_p95": float(np.nanpercentile(ios_before, 95)) if len(ios_before) else None,
        "i_over_sigma_after_p95": float(np.nanpercentile(ios_after, 95)) if len(ios_after) else None,
        "note": (
            "DIALS sigma values are passed through unchanged; intensities are unchanged."
            if float(mult) == 1.0
            else "DIALS empirical sigma calibration is applied to bridge sigmas only; intensities are unchanged."
        ),
    }
    if "sigma_eff" in locals():
        summary["empirical_variance_model"] = empirical_summary
        summary["_sigma_effective"] = sigma_eff
    return float(mult), summary
