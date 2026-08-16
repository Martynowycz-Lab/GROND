#!/usr/bin/env python3
"""Fit a small geometry-only XDS scan path from reflection centroids.

The fit never reads intensities, a structure model, atom types, an expected
configuration, or GROND residuals.  Along-scan slew is inferred from
ZOBS-ZCAL.  Two bounded transverse rotations are inferred from XOBS/YCAL and
YOBS/YCAL after applying the slew correction.  The resulting JSON can be
passed to ``bloch_engine.py --geometry-path`` as a fixed geometry contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from grond.legacy.engine.bloch_engine import (
    XDSData,
    _rotation_matrix_axis_angle,
    sha256,
)


def _embedded_float(text: str, name: str, *, first: bool = False) -> float:
    if first:
        value = str(text).split()[0]
    else:
        match = re.search(rf"(?:^|\s){re.escape(name)}=\s*([-+0-9.eE]+)", str(text))
        if match is None:
            raise ValueError(f"missing XDS detector parameter {name}")
        value = match.group(1)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite XDS detector parameter {name}")
    return parsed


def detector_contract(data: XDSData) -> dict:
    header = data.header
    org = str(header.get("ORGX", ""))
    size = str(header.get("NX", ""))
    if not org or not size:
        raise ValueError("XDS centroid-path fitting requires ORGX/ORGY/distance and QX/QY")
    axes = []
    for key in ("DIRECTION_OF_DETECTOR_X-AXIS", "DIRECTION_OF_DETECTOR_Y-AXIS"):
        values = np.asarray([float(v) for v in str(header.get(key, "")).split()[:3]])
        if values.shape != (3,) or not np.all(np.isfinite(values)) or np.linalg.norm(values) <= 0.0:
            raise ValueError(f"XDS centroid-path fitting requires {key}")
        axes.append(values / np.linalg.norm(values))
    return {
        "orgx": _embedded_float(org, "ORGX", first=True),
        "orgy": _embedded_float(org, "ORGY"),
        "distance_mm": _embedded_float(org, "DETECTOR_DISTANCE"),
        "qx_mm": _embedded_float(size, "QX"),
        "qy_mm": _embedded_float(size, "QY"),
        "axes": np.asarray(axes, dtype=float),
        "beam_axis": np.asarray(data.s0, dtype=float) / float(data.k),
    }


def _chebyshev_matrix(z: np.ndarray, center: float, half_range: float, degree: int) -> np.ndarray:
    normalized = np.clip((np.asarray(z, dtype=float) - center) / half_range, -1.0, 1.0)
    return np.polynomial.chebyshev.chebvander(normalized, int(degree))


def _coefficient_bounds(max_abs: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounds whose absolute Chebyshev sum cannot exceed max_abs."""

    maximum = float(max_abs)
    if maximum <= 0.0 or not math.isfinite(maximum):
        raise ValueError("geometry-path bounds must be finite and positive")
    if degree <= 0:
        budget = np.asarray([maximum], dtype=float)
    else:
        budget = np.full(degree + 1, maximum / (2.0 * degree), dtype=float)
        budget[0] = maximum / 2.0
    return -budget, budget


def _fit_chebyshev(
    values: np.ndarray,
    design: np.ndarray,
    fit_mask: np.ndarray,
    *,
    max_abs: float,
    f_scale: float,
) -> np.ndarray:
    degree = design.shape[1] - 1
    lower, upper = _coefficient_bounds(max_abs, degree)
    result = least_squares(
        lambda coefficients: design[fit_mask] @ coefficients - values[fit_mask],
        np.zeros(degree + 1, dtype=float),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=float(f_scale),
        max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError(f"slew path fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _project_xy(data: XDSData, detector: dict, g: np.ndarray) -> np.ndarray:
    s1 = np.asarray(data.s0, dtype=float) + np.asarray(g, dtype=float)
    denominator = float(np.dot(s1, detector["beam_axis"]))
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.asarray([np.nan, np.nan], dtype=float)
    ray = s1 * (float(detector["distance_mm"]) / denominator)
    axes = detector["axes"]
    return np.asarray(
        [
            float(detector["orgx"]) + float(np.dot(ray, axes[0])) / float(detector["qx_mm"]),
            float(detector["orgy"]) + float(np.dot(ray, axes[1])) / float(detector["qy_mm"]),
        ],
        dtype=float,
    )


def _row_g(data: XDSData, row_idx: int, corrected_z: float) -> np.ndarray:
    omega = data.omega_from_z(float(corrected_z))
    a_matrix = data.A_for_omega(data.omega_from_z(float(data.zcal[row_idx])))
    theta = math.radians(float(omega))
    rotation = (
        data.axis_eye
        + math.sin(theta) * data.axis_cross
        + (1.0 - math.cos(theta)) * data.axis_cross2
    )
    return (
        data.setting_rotation
        @ rotation
        @ data.fixed_rotation
        @ a_matrix
        @ np.asarray(data.hkl[row_idx], dtype=float)
    )


def _centroid_linearization(
    data: XDSData,
    detector: dict,
    indices: np.ndarray,
    slew_frames: np.ndarray,
    *,
    derivative_step_deg: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.zeros((len(indices), 2), dtype=float)
    jacobian = np.zeros((len(indices), 2, 2), dtype=float)
    axes = detector["axes"]
    for pos, (row_idx, slew) in enumerate(zip(indices, slew_frames, strict=True)):
        g = _row_g(data, int(row_idx), float(data.zcal[row_idx]) + float(slew))
        base[pos] = _project_xy(data, detector, g)
        for axis_idx, axis in enumerate(axes):
            perturbed = _rotation_matrix_axis_angle(axis, derivative_step_deg) @ g
            jacobian[pos, :, axis_idx] = (
                _project_xy(data, detector, perturbed) - base[pos]
            ) / derivative_step_deg
    return base, jacobian


def _fit_tilt_coefficients(
    observed_xy: np.ndarray,
    base_xy: np.ndarray,
    jacobian: np.ndarray,
    design: np.ndarray,
    fit_mask: np.ndarray,
    *,
    max_abs_tilt_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    degree = design.shape[1] - 1
    count = degree + 1
    lower_one, upper_one = _coefficient_bounds(max_abs_tilt_deg, degree)
    lower = np.concatenate([lower_one, lower_one])
    upper = np.concatenate([upper_one, upper_one])

    def residual(coefficients):
        tilt_x = design @ coefficients[:count]
        tilt_y = design @ coefficients[count:]
        predicted = (
            base_xy
            + jacobian[:, :, 0] * tilt_x[:, None]
            + jacobian[:, :, 1] * tilt_y[:, None]
        )
        return (predicted[fit_mask] - observed_xy[fit_mask]).reshape(-1)

    result = least_squares(
        residual,
        np.zeros(2 * count, dtype=float),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=3000,
    )
    if not result.success:
        raise RuntimeError(f"transverse orientation path fit failed: {result.message}")
    return np.asarray(result.x[:count]), np.asarray(result.x[count:])


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values))) if values.size else float("nan")


def fit_geometry_path(
    hkl: Path,
    out: Path,
    *,
    degree: int = 3,
    min_peak: float = 90.0,
    min_corr: float = 0.0,
    max_abs_slew_frames: float = 12.0,
    max_abs_tilt_deg: float = 0.25,
    folds: int = 5,
) -> dict:
    if int(degree) < 2 or int(degree) > 5:
        raise ValueError("geometry-path degree must be 2 through 5 (3 through 6 coefficients)")
    if int(folds) < 3:
        raise ValueError("geometry-path validation requires at least three contiguous folds")
    data = XDSData(Path(hkl), hkl_transform="identity")
    detector = detector_contract(data)
    mask = (
        np.isfinite(data.zcal)
        & np.isfinite(data.zobs)
        & np.isfinite(data.xobs)
        & np.isfinite(data.yobs)
        & (data.zobs > 0.0)
        & (data.xobs > 0.0)
        & (data.yobs > 0.0)
        & (data.peak >= float(min_peak))
        & (data.corr >= float(min_corr))
    )
    indices = np.where(mask)[0]
    if len(indices) < max(100, 20 * (int(degree) + 1)):
        raise ValueError("too few valid XDS centroid rows for a smooth geometry path")
    z = np.asarray(data.zcal[indices], dtype=float)
    center = 0.5 * (float(np.min(z)) + float(np.max(z)))
    half_range = max(0.5 * (float(np.max(z)) - float(np.min(z))), 1.0)
    design = _chebyshev_matrix(z, center, half_range, int(degree))
    observed_dz = np.asarray(data.zobs[indices] - data.zcal[indices], dtype=float)
    # A constant frame datum is already an explicit GROND nuisance parameter.
    # Fit only the smooth, zero-median scan-path departure here so the two
    # effects cannot be counted twice.
    frame_datum = float(np.median(observed_dz))
    # If a reflection predicted at z_cal is observed at z_obs, the orientation
    # correction at that physical frame is z_cal-z_obs.  The sign is therefore
    # opposite to the observed centroid displacement.
    observed_slew = -(observed_dz - frame_datum)
    observed_xy = np.column_stack((data.xobs[indices], data.yobs[indices]))
    fold_index = np.minimum(
        int(folds) - 1,
        np.floor((z - np.min(z)) / max(np.ptp(z), 1e-12) * int(folds)).astype(int),
    )

    cv_rows = []
    cv_slew_pred = np.full(len(indices), np.nan, dtype=float)
    cv_xy_pred = np.full((len(indices), 2), np.nan, dtype=float)
    for fold in range(int(folds)):
        free = fold_index == fold
        work = ~free
        slew_coeff = _fit_chebyshev(
            observed_slew,
            design,
            work,
            max_abs=max_abs_slew_frames,
            f_scale=0.5,
        )
        slew_coeff[0] -= float(np.median(design[work] @ slew_coeff))
        slew = design @ slew_coeff
        base_xy, jacobian = _centroid_linearization(
            data, detector, indices, slew
        )
        tilt_x_coeff, tilt_y_coeff = _fit_tilt_coefficients(
            observed_xy,
            base_xy,
            jacobian,
            design,
            work,
            max_abs_tilt_deg=max_abs_tilt_deg,
        )
        tilt_x = design @ tilt_x_coeff
        tilt_y = design @ tilt_y_coeff
        predicted_xy = (
            base_xy
            + jacobian[:, :, 0] * tilt_x[:, None]
            + jacobian[:, :, 1] * tilt_y[:, None]
        )
        cv_slew_pred[free] = slew[free]
        cv_xy_pred[free] = predicted_xy[free]
        cv_rows.append(
            {
                "fold": int(fold),
                "work_rows": int(np.count_nonzero(work)),
                "held_out_rows": int(np.count_nonzero(free)),
                "held_out_slew_rms_frames": _rms(observed_slew[free] - slew[free]),
                "held_out_centroid_rms_px": _rms(
                    np.linalg.norm(observed_xy[free] - predicted_xy[free], axis=1)
                ),
            }
        )

    all_rows = np.ones(len(indices), dtype=bool)
    slew_coeff = _fit_chebyshev(
        observed_slew,
        design,
        all_rows,
        max_abs=max_abs_slew_frames,
        f_scale=0.5,
    )
    slew_coeff[0] -= float(np.median(design @ slew_coeff))
    slew = design @ slew_coeff
    base_xy, jacobian = _centroid_linearization(data, detector, indices, slew)
    tilt_x_coeff, tilt_y_coeff = _fit_tilt_coefficients(
        observed_xy,
        base_xy,
        jacobian,
        design,
        all_rows,
        max_abs_tilt_deg=max_abs_tilt_deg,
    )
    tilt_x = design @ tilt_x_coeff
    tilt_y = design @ tilt_y_coeff
    candidate_predicted_xy = (
        base_xy
        + jacobian[:, :, 0] * tilt_x[:, None]
        + jacobian[:, :, 1] * tilt_y[:, None]
    )
    baseline_centroid_rms = _rms(
        np.linalg.norm(
            observed_xy
            - np.column_stack((data.xcal[indices], data.ycal[indices])),
            axis=1,
        )
    )
    held_out_centroid_rms = _rms(
        np.linalg.norm(observed_xy - cv_xy_pred, axis=1)
    )
    held_out_centroid_gain_fraction = (
        (baseline_centroid_rms - held_out_centroid_rms)
        / max(baseline_centroid_rms, 1e-30)
    )
    lower_tilt, upper_tilt = _coefficient_bounds(max_abs_tilt_deg, int(degree))
    hit_bound = bool(
        np.any(np.isclose(np.abs(tilt_x_coeff), upper_tilt, rtol=0.0, atol=1e-6))
        or np.any(np.isclose(np.abs(tilt_y_coeff), upper_tilt, rtol=0.0, atol=1e-6))
    )
    tilt_promoted = bool(held_out_centroid_gain_fraction >= 0.01 and not hit_bound)
    diagnostic_tilt_x_coeff = tilt_x_coeff.copy()
    diagnostic_tilt_y_coeff = tilt_y_coeff.copy()
    if not tilt_promoted:
        tilt_x_coeff = np.zeros_like(tilt_x_coeff)
        tilt_y_coeff = np.zeros_like(tilt_y_coeff)
        tilt_x = np.zeros_like(tilt_x)
        tilt_y = np.zeros_like(tilt_y)
        predicted_xy = base_xy
    else:
        predicted_xy = candidate_predicted_xy

    contract = {
        "schema_version": 1,
        "kind": "grond_geometry_path",
        "basis": "chebyshev",
        "fit_source": (
            "XDS centroids only; no intensities, structure, hand, or GROND residuals; "
            "slew is zero-median and excludes the separately fitted global frame datum"
        ),
        "source_hkl": str(Path(hkl).expanduser().resolve()),
        "source_hkl_sha256": sha256(Path(hkl)),
        "frame_center": center,
        "frame_half_range": half_range,
        "degree": int(degree),
        "coefficients": {
            "slew_frames": slew_coeff.tolist(),
            "tilt_x_deg": tilt_x_coeff.tolist(),
            "tilt_y_deg": tilt_y_coeff.tolist(),
        },
        "diagnostic_candidate_coefficients": {
            "tilt_x_deg": diagnostic_tilt_x_coeff.tolist(),
            "tilt_y_deg": diagnostic_tilt_y_coeff.tolist(),
        },
        "tilt_axes_lab": detector["axes"].tolist(),
        "bounds": {
            "max_abs_slew_frames": float(max_abs_slew_frames),
            "max_abs_tilt_deg": float(max_abs_tilt_deg),
        },
        "integration_profile_provenance": data.integration_profile_provenance(),
        "validation": {
            "method": f"{int(folds)} contiguous frame folds",
            "rows": int(len(indices)),
            "zobs_minus_zcal_median_frame_datum": frame_datum,
            "slew_definition": "-(ZOBS-ZCAL-median(ZOBS-ZCAL)); global datum excluded",
            "baseline_zobs_minus_zcal_rms_frames": _rms(observed_dz),
            "baseline_centered_slew_rms_frames": _rms(observed_slew),
            "fitted_centered_slew_rms_frames": _rms(observed_slew - slew),
            "held_out_centered_slew_rms_frames": _rms(observed_slew - cv_slew_pred),
            "baseline_centroid_rms_px": baseline_centroid_rms,
            "slew_only_centroid_rms_px": _rms(
                np.linalg.norm(observed_xy - base_xy, axis=1)
            ),
            "candidate_slew_plus_tilt_centroid_rms_px": _rms(
                np.linalg.norm(observed_xy - candidate_predicted_xy, axis=1)
            ),
            "fitted_centroid_rms_px": _rms(
                np.linalg.norm(observed_xy - predicted_xy, axis=1)
            ),
            "held_out_centroid_rms_px": held_out_centroid_rms,
            "held_out_centroid_gain_fraction": held_out_centroid_gain_fraction,
            "transverse_path_hit_coefficient_bound": hit_bound,
            "transverse_path_promoted": tilt_promoted,
            "transverse_path_gate": (
                "requires at least 1% held-out centroid RMS improvement and no active coefficient bound"
            ),
            "slew_range_frames": [float(np.min(slew)), float(np.max(slew))],
            "tilt_x_range_deg": [float(np.min(tilt_x)), float(np.max(tilt_x))],
            "tilt_y_range_deg": [float(np.min(tilt_y)), float(np.max(tilt_y))],
            "folds": cv_rows,
        },
    }
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    csv_path = out.with_suffix(".rows.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row_index",
                "zcal",
                "zobs_minus_zcal",
                "slew_fit_frames",
                "tilt_x_fit_deg",
                "tilt_y_fit_deg",
                "xobs_minus_xcal_px",
                "yobs_minus_ycal_px",
                "xobs_minus_path_px",
                "yobs_minus_path_px",
                "contiguous_fold",
            ]
        )
        for pos, row_idx in enumerate(indices):
            writer.writerow(
                [
                    int(row_idx),
                    float(data.zcal[row_idx]),
                    float(observed_dz[pos]),
                    float(slew[pos]),
                    float(tilt_x[pos]),
                    float(tilt_y[pos]),
                    float(data.xobs[row_idx] - data.xcal[row_idx]),
                    float(data.yobs[row_idx] - data.ycal[row_idx]),
                    float(data.xobs[row_idx] - predicted_xy[pos, 0]),
                    float(data.yobs[row_idx] - predicted_xy[pos, 1]),
                    int(fold_index[pos]),
                ]
            )
    contract["row_table"] = str(csv_path)
    contract["row_table_sha256"] = sha256(csv_path)
    out.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hkl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--min-peak", type=float, default=90.0)
    parser.add_argument("--min-corr", type=float, default=0.0)
    parser.add_argument("--max-abs-slew-frames", type=float, default=12.0)
    parser.add_argument("--max-abs-tilt-deg", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = fit_geometry_path(
        Path(args.hkl),
        Path(args.out),
        degree=args.degree,
        min_peak=args.min_peak,
        min_corr=args.min_corr,
        max_abs_slew_frames=args.max_abs_slew_frames,
        max_abs_tilt_deg=args.max_abs_tilt_deg,
        folds=args.folds,
    )
    print(json.dumps(result["validation"], indent=2))
    print(f"wrote {Path(args.out).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
