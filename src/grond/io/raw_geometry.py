"""Geometry-only frame-offset calibration and exact signed-row scan helpers.

This records how ZCAL is interpreted; it does not reassign pixels or observations."""

from __future__ import annotations

import math

import numpy as np

from grond.io.raw_detector import gvec_for_hkl
from grond.io.xds import XDSData


def group_id_from_z(data: XDSData, zcal: float, group_width_frames: float) -> int:
    return int(math.floor((float(zcal) - float(data.starting_frame)) / max(float(group_width_frames), 1e-12)))


def z_from_omega(data: XDSData, omega_deg: float) -> float:
    return float(data.starting_frame + (float(omega_deg) - float(data.starting_angle)) / float(data.oscillation_range))


def sg_for_hkl(data: XDSData, hkl, omega_deg: float) -> float:
    g = gvec_for_hkl(data, hkl, omega_deg)
    return float((data.k**2 - float(np.sum((data.s0 + g) ** 2))) / (2.0 * data.k))


def solve_zero_sg_angle(data: XDSData, hkl, omega0_deg: float) -> tuple[float, float]:
    omega = float(omega0_deg)
    for _ in range(12):
        sg = sg_for_hkl(data, hkl, omega)
        if abs(sg) < 1e-8:
            break
        step = 1e-4
        deriv = (sg_for_hkl(data, hkl, omega + step) - sg_for_hkl(data, hkl, omega - step)) / (2.0 * step)
        if abs(deriv) < 1e-12:
            break
        omega -= sg / deriv
    return float(omega), sg_for_hkl(data, hkl, omega)


def calibration_delta_for_idx(calibration_by_idx: dict[int, float] | None, idx: int) -> float:
    if not calibration_by_idx:
        return 0.0
    return float(calibration_by_idx.get(int(idx), 0.0))


def evaluate_frame_correction(
    data: XDSData,
    row_indices,
    deltas: np.ndarray,
    zero_sg_frames: np.ndarray,
    raw_z: np.ndarray,
):
    sg_abs = []
    det_err = []
    zero_delta = []
    for n, idx in enumerate(row_indices):
        zc = float(data.zcal[idx]) + float(deltas[n])
        omega = float(data.omega_from_z(zc))
        hkl = data.hkl[idx]
        sg_abs.append(abs(sg_for_hkl(data, hkl, omega)))
        zero_delta.append(float(zero_sg_frames[n] - zc))
        det_err.append(math.nan)
    sg_abs = np.asarray(sg_abs, dtype=float)
    zero_delta = np.asarray(zero_delta, dtype=float)
    raw_delta = raw_z - (np.asarray([data.zcal[idx] for idx in row_indices], dtype=float) + deltas)
    return {
        "median_abs_center_sg_inv_A": (float(np.nanmedian(sg_abs)) if len(sg_abs) else None),
        "p95_abs_center_sg_inv_A": (float(np.nanpercentile(sg_abs, 95)) if len(sg_abs) else None),
        "zero_sg_delta_frame_median_after": (float(np.nanmedian(zero_delta)) if len(zero_delta) else None),
        "zero_sg_delta_frame_p95_abs_after": (
            float(np.nanpercentile(np.abs(zero_delta), 95)) if len(zero_delta) else None
        ),
        "raw_z_minus_calibrated_center_median": (
            float(np.nanmedian(raw_delta)) if np.any(np.isfinite(raw_delta)) else None
        ),
        "raw_z_minus_calibrated_center_p95_abs": (
            float(np.nanpercentile(np.abs(raw_delta), 95)) if np.any(np.isfinite(raw_delta)) else None
        ),
        "frame_correction_median": float(np.nanmedian(deltas)) if len(deltas) else 0.0,
        "frame_correction_p95_abs": (float(np.nanpercentile(np.abs(deltas), 95)) if len(deltas) else 0.0),
    }


def fit_geometry_calibration(data: XDSData, row_indices, integrations, requested_model: str, threshold_sg: float):
    """Fit explicit, auditable Z-frame correction models.

    The correction is a frame offset added to XDS ZCAL before replaying omega.
    It does not move detector pixels or raw shoebox ownership; it only records
    how the row-center geometry should be interpreted by the Bloch calculator.
    """
    row_indices = np.asarray(row_indices, dtype=int)
    zcal = np.asarray([float(data.zcal[idx]) for idx in row_indices], dtype=float)
    omega0 = np.asarray([float(data.omega_from_z(z)) for z in zcal], dtype=float)
    zero_angles = []
    zero_resid = []
    for idx, om in zip(row_indices, omega0, strict=False):
        angle, resid = solve_zero_sg_angle(data, data.hkl[idx], om)
        zero_angles.append(angle)
        zero_resid.append(resid)
    zero_angles = np.asarray(zero_angles, dtype=float)
    zero_sg_frames = np.asarray([z_from_omega(data, a) for a in zero_angles], dtype=float)
    target_delta = zero_sg_frames - zcal
    raw_z = np.asarray(
        [float(integ.get("raw_z_centroid", np.nan)) for integ in integrations],
        dtype=float,
    )
    raw_delta = raw_z - zcal
    x = zcal - float(np.nanmedian(zcal)) if len(zcal) else zcal
    candidates: dict[str, np.ndarray] = {}
    candidates["none"] = np.zeros_like(target_delta)
    candidates["global_offset"] = np.full_like(
        target_delta, float(np.nanmedian(target_delta)) if len(target_delta) else 0.0
    )
    if len(target_delta) >= 3 and np.any(np.isfinite(target_delta)):
        coeff = np.polyfit(x[np.isfinite(target_delta)], target_delta[np.isfinite(target_delta)], 1)
        candidates["linear_drift"] = coeff[0] * x + coeff[1]
    else:
        coeff = np.asarray([0.0, float(np.nanmedian(target_delta)) if len(target_delta) else 0.0])
        candidates["linear_drift"] = candidates["global_offset"].copy()
    if len(target_delta) >= 12:
        order = np.argsort(zcal)
        n_bins = min(9, max(3, int(math.sqrt(len(zcal)))))
        bins = np.array_split(order, n_bins)
        knots_z = []
        knots_delta = []
        for b in bins:
            if len(b) == 0:
                continue
            knots_z.append(float(np.nanmedian(zcal[b])))
            knots_delta.append(float(np.nanmedian(target_delta[b])))
        candidates["spline"] = np.interp(zcal, np.asarray(knots_z), np.asarray(knots_delta))
    else:
        knots_z = [float(np.nanmedian(zcal))] if len(zcal) else [0.0]
        knots_delta = [float(np.nanmedian(target_delta))] if len(target_delta) else [0.0]
        candidates["spline"] = candidates["global_offset"].copy()

    summaries = {
        name: evaluate_frame_correction(data, row_indices, vals, zero_sg_frames, raw_z)
        for name, vals in candidates.items()
    }
    priority = ["none", "global_offset", "linear_drift", "spline"]
    if requested_model != "auto":
        selected = requested_model
    else:
        selected = min(
            priority,
            key=lambda name: (
                summaries[name]["median_abs_center_sg_inv_A"]
                if summaries[name]["median_abs_center_sg_inv_A"] is not None
                else float("inf")
            ),
        )
        for name in priority:
            med = summaries[name]["median_abs_center_sg_inv_A"]
            if med is not None and float(med) <= float(threshold_sg):
                selected = name
                break
    selected_deltas = candidates[selected]
    calibration_by_idx = {int(idx): float(delta) for idx, delta in zip(row_indices, selected_deltas, strict=False)}
    return {
        "enabled": requested_model != "none",
        "requested_model": requested_model,
        "selected_model": selected,
        "target": "zero_sg_frame_minus_xds_zcal",
        "threshold_median_abs_center_sg_inv_A": float(threshold_sg),
        "row_count": int(len(row_indices)),
        "raw_centroid_delta_frame_median_before": (
            float(np.nanmedian(raw_delta)) if np.any(np.isfinite(raw_delta)) else None
        ),
        "raw_centroid_delta_frame_p95_abs_before": (
            float(np.nanpercentile(np.abs(raw_delta), 95)) if np.any(np.isfinite(raw_delta)) else None
        ),
        "zero_sg_delta_frame_median_before": (float(np.nanmedian(target_delta)) if len(target_delta) else None),
        "zero_sg_delta_frame_p95_abs_before": (
            float(np.nanpercentile(np.abs(target_delta), 95)) if len(target_delta) else None
        ),
        "max_zero_sg_solve_residual_inv_A": (float(np.nanmax(np.abs(zero_resid))) if len(zero_resid) else None),
        "linear_coefficients_delta_frame_vs_centered_z": [float(v) for v in coeff],
        "spline_knots": {
            "zcal_frame": [float(v) for v in knots_z],
            "delta_frame": [float(v) for v in knots_delta],
        },
        "candidate_summaries": summaries,
        "selected_summary": summaries[selected],
        "calibration_by_idx": calibration_by_idx,
    }
