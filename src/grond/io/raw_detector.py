"""XDS detector datum, reciprocal-ray replay and frame geometry records."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from grond.bloch.numerics import rotation_matrix, sha256
from grond.io.xds import XDSData
from grond.legacy.engine.dynrow_contract import frame_contract_fields


def header_vector(data: XDSData, key: str, default=None):
    text = data.header.get(key)
    if text is None:
        return default
    vals = [float(x) for x in text.split()[:3]]
    return np.asarray(vals, dtype=float)


def header_float(data: XDSData, key: str, default=None):
    text = data.header.get(key)
    if text is None:
        return default
    return float(text.split()[0])


def xds_detector_origin_and_beam_center(
    *,
    fast,
    slow,
    beam_direction,
    orgx: float,
    orgy: float,
    qx: float,
    qy: float,
    distance: float,
):
    """Convert XDS detector datum coordinates into a physical lab origin.

    XDS ``ORGX/ORGY`` locate the detector coordinate origin; they are not the
    direct-beam intersection when the incident beam is not parallel to the
    detector normal. ``DETECTOR_DISTANCE`` is measured along that normal.
    """

    # Copy because callers commonly pass XDSData.s0 and axis arrays directly;
    # in-place normalization must never mutate the forward-model geometry.
    fast = np.array(fast, dtype=float, copy=True)
    slow = np.array(slow, dtype=float, copy=True)
    beam = np.array(beam_direction, dtype=float, copy=True)
    fast /= np.linalg.norm(fast)
    slow /= np.linalg.norm(slow)
    beam /= np.linalg.norm(beam)
    normal = np.cross(fast, slow)
    normal /= np.linalg.norm(normal)
    beam_normal = float(np.dot(beam, normal))
    if abs(beam_normal) <= 1e-12:
        raise ValueError("XDS incident beam is parallel to the detector plane")
    origin = float(distance) * normal - float(orgx) * float(qx) * fast - float(orgy) * float(qy) * slow
    ray_distance = float(distance) / beam_normal
    beam_center = np.asarray(
        [
            float(orgx) + ray_distance * float(np.dot(beam, fast)) / float(qx),
            float(orgy) + ray_distance * float(np.dot(beam, slow)) / float(qy),
        ],
        dtype=float,
    )
    return origin, beam_center, normal


def header_detector_geometry(data: XDSData):
    nx = None
    ny = None
    qx = None
    qy = None
    tokens = data.header.get("NX", "").replace("=", " ").split()
    # Header value after the first equals is usually:
    # "1024  NY=  1024    QX=  0.056000  QY=  0.056000"
    for i, tok in enumerate(tokens):
        u = tok.upper()
        if i == 0:
            with contextlib.suppress(ValueError):
                nx = int(float(tok))
        elif u == "NY" and i + 1 < len(tokens):
            ny = int(float(tokens[i + 1]))
        elif u == "QX" and i + 1 < len(tokens):
            qx = float(tokens[i + 1])
        elif u == "QY" and i + 1 < len(tokens):
            qy = float(tokens[i + 1])
    org_tokens = data.header.get("ORGX", "").replace("=", " ").split()
    orgx = orgy = distance = None
    for i, tok in enumerate(org_tokens):
        u = tok.upper()
        if i == 0:
            orgx = float(tok)
        elif u == "ORGY" and i + 1 < len(org_tokens):
            orgy = float(org_tokens[i + 1])
        elif u == "DETECTOR_DISTANCE" and i + 1 < len(org_tokens):
            distance = float(org_tokens[i + 1])
    fast = header_vector(data, "DIRECTION_OF_DETECTOR_X-AXIS", np.asarray([1.0, 0.0, 0.0]))
    slow = header_vector(data, "DIRECTION_OF_DETECTOR_Y-AXIS", np.asarray([0.0, 1.0, 0.0]))
    qx = 1.0 if qx is None else qx
    qy = 1.0 if qy is None else qy
    orgx = 0.0 if orgx is None else orgx
    orgy = 0.0 if orgy is None else orgy
    distance = 0.0 if distance is None else distance
    origin, beam_center, normal = xds_detector_origin_and_beam_center(
        fast=fast,
        slow=slow,
        beam_direction=data.s0,
        orgx=orgx,
        orgy=orgy,
        qx=qx,
        qy=qy,
        distance=distance,
    )
    return {
        "detector": data.header.get("DETECTOR", "").strip(),
        "nx": nx,
        "ny": ny,
        "pixel_size_mm": [float(qx), float(qy)],
        "xds_org_px": [float(orgx), float(orgy)],
        "beam_center_px": beam_center.astype(float).tolist(),
        "detector_distance_mm": float(distance),
        "detector_origin_mm": origin.astype(float).tolist(),
        "detector_fast_axis": fast.astype(float).tolist(),
        "detector_slow_axis": slow.astype(float).tolist(),
        "detector_normal": normal.astype(float).tolist(),
        "n_segments": (
            int(float(data.header.get("NUMBER OF DETECTOR SEGMENTS", "1").split()[0]))
            if data.header.get("NUMBER OF DETECTOR SEGMENTS")
            else 1
        ),
    }


def parse_xparm(path: Path | None):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    raw_lines = [ln.split() for ln in path.read_text(errors="replace").splitlines() if ln.split()]
    lines = []
    for parts in raw_lines:
        try:
            float(parts[0])
        except ValueError:
            continue
        lines.append(parts)
    if len(lines) < 11:
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "parse_warning": "too_few_lines",
        }
    try:
        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 = lines[:11]
        fast = [float(v) for v in l9[:3]]
        slow = [float(v) for v in l10[:3]]
        incident_beam = [float(v) for v in l2[1:4]]
        orgx, orgy, distance = [float(v) for v in l8[:3]]
        qx, qy = [float(l7[3]), float(l7[4])]
        origin, beam_center, normal = xds_detector_origin_and_beam_center(
            fast=fast,
            slow=slow,
            beam_direction=incident_beam,
            orgx=orgx,
            orgy=orgy,
            qx=qx,
            qy=qy,
            distance=distance,
        )
        out = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "starting_frame": int(float(l1[0])),
            "starting_angle_deg": float(l1[1]),
            "oscillation_range_deg": float(l1[2]),
            "rotation_axis_lab": [float(v) for v in l1[3:6]],
            "wavelength_A": float(l2[0]),
            "incident_beam_vector": incident_beam,
            "space_group_number": int(float(l3[0])),
            "unit_cell": [float(v) for v in l3[1:7]],
            "real_space_axes": [
                [float(v) for v in l4[:3]],
                [float(v) for v in l5[:3]],
                [float(v) for v in l6[:3]],
            ],
            "segment_count": int(float(l7[0])),
            "nx": int(float(l7[1])),
            "ny": int(float(l7[2])),
            "pixel_size_mm": [qx, qy],
            "xds_org_px": [orgx, orgy],
            "beam_center_px": beam_center.astype(float).tolist(),
            "detector_distance_mm": distance,
            "detector_origin_mm": origin.astype(float).tolist(),
            "detector_fast_axis": fast,
            "detector_slow_axis": slow,
            "detector_normal": normal.astype(float).tolist(),
        }
        return out
    except Exception as exc:
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "parse_warning": str(exc),
        }


def frame_geometry_record(
    data: XDSData,
    image_number: int,
    group_id: int,
    source_hkl: Path,
    model_meta: dict,
    detector: dict,
    xparm_meta: dict | None,
):
    image_number = int(image_number)
    angle = float(data.omega_from_z(image_number))
    return {
        **frame_contract_fields("xds_integrate_hkl_raw_image_frame"),
        "dataset_id": "sample",
        "frame_id": image_number,
        "angle_deg": angle,
        "image_number": image_number,
        "wavelength_A": float(data.wavelength),
        "k_inv_A": float(data.k),
        "s0_lab": data.s0.astype(float).tolist(),
        "rotation_axis_lab": data.rotation_axis_lab.astype(float).tolist(),
        "UB": (data.setting_rotation @ data.fixed_rotation @ data.A_for_omega(angle)).astype(float).tolist(),
        "detector_origin_mm": detector["detector_origin_mm"],
        "detector_fast_axis": detector["detector_fast_axis"],
        "detector_slow_axis": detector["detector_slow_axis"],
        "pixel_size_mm": detector["pixel_size_mm"],
        "beam_center_px": detector["beam_center_px"],
        "detector_distance_mm": detector["detector_distance_mm"],
        "detector_nx": detector["nx"],
        "detector_ny": detector["ny"],
        "detector_name": detector["detector"],
        "detector_segment_geometry": {
            "n_segments": detector["n_segments"],
            "iseg_values_present": sorted(set(int(v) for v in data.iseg.tolist())),
        },
        "group_id": int(group_id),
        "source": "xds_geometry",
        "geometry_source": str(source_hkl.resolve()),
        "xparm": xparm_meta,
        "starting_model": model_meta,
    }


def gvec_for_hkl(data: XDSData, hkl, omega_deg: float) -> np.ndarray:
    """Map one signed HKL through the complete S R F A geometry chain."""

    return (
        data.setting_rotation
        @ rotation_matrix(data.axis, omega_deg)
        @ data.fixed_rotation
        @ data.A_for_omega(omega_deg)
        @ np.asarray(hkl, dtype=float)
    )


def replay_detector_xy(data: XDSData, hkl, omega_deg: float, detector: dict):
    fast = np.asarray(detector["detector_fast_axis"], dtype=float)
    slow = np.asarray(detector["detector_slow_axis"], dtype=float)
    origin = np.asarray(detector["detector_origin_mm"], dtype=float)
    qx, qy = [float(v) for v in detector["pixel_size_mm"]]
    g = gvec_for_hkl(data, hkl, omega_deg)
    ray = data.s0 + g
    mat = np.stack([fast, slow, -ray], axis=1)
    try:
        u_mm, v_mm, _ = np.linalg.solve(mat, -origin)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(u_mm / qx), float(v_mm / qy)


def geometry_replay_summary(
    data: XDSData,
    row_indices,
    detector: dict,
    calibration_by_idx: dict[int, float] | None = None,
):
    calibration_by_idx = calibration_by_idx or {}
    sg_abs = []
    dx = []
    dy = []
    dz_corr = []
    for idx in row_indices:
        z_center = float(data.zcal[idx]) + float(calibration_by_idx.get(int(idx), 0.0))
        omega = float(data.omega_from_z(z_center))
        hkl = data.hkl[idx]
        g = gvec_for_hkl(data, hkl, omega)
        sg = (data.k**2 - float(np.sum((data.s0 + g) ** 2))) / (2.0 * data.k)
        x_pred, y_pred = replay_detector_xy(data, hkl, omega, detector)
        sg_abs.append(abs(float(sg)))
        dx.append(float(x_pred - data.xcal[idx]))
        dy.append(float(y_pred - data.ycal[idx]))
        dz_corr.append(float(calibration_by_idx.get(int(idx), 0.0)))
    sg_abs = np.asarray(sg_abs, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dr = np.sqrt(dx * dx + dy * dy)
    return {
        "n_rows": int(len(row_indices)),
        "median_abs_sg_inv_A": float(np.nanmedian(sg_abs)) if len(sg_abs) else None,
        "p95_abs_sg_inv_A": (float(np.nanpercentile(sg_abs, 95)) if len(sg_abs) else None),
        "median_detector_error_px": float(np.nanmedian(dr)) if len(dr) else None,
        "p95_detector_error_px": float(np.nanpercentile(dr, 95)) if len(dr) else None,
        "median_dx_px": float(np.nanmedian(dx)) if len(dx) else None,
        "median_dy_px": float(np.nanmedian(dy)) if len(dy) else None,
        "median_frame_correction_applied": (float(np.nanmedian(dz_corr)) if len(dz_corr) else 0.0),
        "p95_abs_frame_correction_applied": (float(np.nanpercentile(np.abs(dz_corr), 95)) if len(dz_corr) else 0.0),
        "convention": "g=R(axis,omega)@UB@h; ray=s0+g; detector intersection in XDS header lab frame; optional calibrated ZCAL frame correction applied before omega replay",
    }
