#!/usr/bin/env python3
"""Build row-contract signed dynrows from XDS predictions and raw images.

Inputs
------
* XDS `INTEGRATE.HKL`, including the header geometry and row-level signed HKLs,
  predicted detector positions, and predicted scan positions.
* Raw CBF images.
* A starting model file, usually CIF or SHELX RES, recorded as provenance for
  downstream Bloch calculations.

Outputs
-------
* `frames.jsonl`, one line per XDS-derived observation group.
* `dynrows.jsonl`, one line per exact signed observation row.
* `observations.jsonl`, a compatibility alias with the same exact signed rows.
* optional XDS-like HKL, so the existing `bloch_engine.py` scorer can
  evaluate the same raw observation table without changing the Bloch kernel.

The raw integration is deliberately transparent.  The default target is a
circular aperture sum over a small frame range around XDS ZCAL with an annular
local background.  A second `gaussian_profile` mode uses the same pixels and
background but estimates a profile-fit intensity from a 3D Gaussian matched
profile.  A third `bloch_profile_fit` mode fits raw pixels directly to a local
background plane plus a Gaussian detector-plane profile multiplied by the
Bloch-predicted rocking profile.  None is claimed to be raw-image equivalence; these
are explicit raw-image observation contracts that can be audited row by row.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import json
import math
import re
from pathlib import Path

import gemmi
import numpy as np

if __package__:
    from .bloch_engine import (
        Config,
        XDSBlochPort,
        XDSData,
        parse_model,
        parse_res,
        rotation_matrix,
        sha256,
    )
    from .dynrow_contract import (
        DYNROW_SCHEMA_VERSION,
        FROZEN_ENGINE_CONTRACT,
        FROZEN_ENGINE_CONTRACT_VERSION,
        contract_metadata,
        frame_contract_fields,
        observation_contract_fields,
        summarize_contract_validation,
    )
else:  # pragma: no cover - direct script execution
    from bloch_engine import (
        Config,
        XDSBlochPort,
        XDSData,
        parse_model,
        parse_res,
        rotation_matrix,
        sha256,
    )
    from dynrow_contract import (
        DYNROW_SCHEMA_VERSION,
        FROZEN_ENGINE_CONTRACT,
        FROZEN_ENGINE_CONTRACT_VERSION,
        contract_metadata,
        frame_contract_fields,
        observation_contract_fields,
        summarize_contract_validation,
    )


def require_fabio():
    """Load the optional raw-image reader when this workflow is used."""
    try:
        import fabio
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Raw CBF extraction requires the optional 'fabio' dependency. "
            "Install it with: pip install 'grond[raw]'"
        ) from exc
    return fabio


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
    fast = header_vector(
        data, "DIRECTION_OF_DETECTOR_X-AXIS", np.asarray([1.0, 0.0, 0.0])
    )
    slow = header_vector(
        data, "DIRECTION_OF_DETECTOR_Y-AXIS", np.asarray([0.0, 1.0, 0.0])
    )
    beam_dir = data.s0 / np.linalg.norm(data.s0)
    qx = 1.0 if qx is None else qx
    qy = 1.0 if qy is None else qy
    orgx = 0.0 if orgx is None else orgx
    orgy = 0.0 if orgy is None else orgy
    distance = 0.0 if distance is None else distance
    origin = distance * beam_dir - orgx * qx * fast - orgy * qy * slow
    return {
        "detector": data.header.get("DETECTOR", "").strip(),
        "nx": nx,
        "ny": ny,
        "pixel_size_mm": [float(qx), float(qy)],
        "beam_center_px": [float(orgx), float(orgy)],
        "detector_distance_mm": float(distance),
        "detector_origin_mm": origin.astype(float).tolist(),
        "detector_fast_axis": fast.astype(float).tolist(),
        "detector_slow_axis": slow.astype(float).tolist(),
        "detector_normal": np.cross(fast, slow).astype(float).tolist(),
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
    raw_lines = [
        ln.split() for ln in path.read_text(errors="replace").splitlines() if ln.split()
    ]
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
        out = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "starting_frame": int(float(l1[0])),
            "starting_angle_deg": float(l1[1]),
            "oscillation_range_deg": float(l1[2]),
            "rotation_axis_lab": [float(v) for v in l1[3:6]],
            "wavelength_A": float(l2[0]),
            "incident_beam_vector": [float(v) for v in l2[1:4]],
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
            "pixel_size_mm": [float(l7[3]), float(l7[4])],
            "beam_center_px": [float(l8[0]), float(l8[1])],
            "detector_distance_mm": float(l8[2]),
            "detector_fast_axis": [float(v) for v in l9[:3]],
            "detector_slow_axis": [float(v) for v in l10[:3]],
            "detector_normal": [float(v) for v in l11[:3]],
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
        "UB": (
            data.setting_rotation
            @ data.fixed_rotation
            @ data.A_for_omega(angle)
        ).astype(float).tolist(),
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
        "p95_abs_sg_inv_A": (
            float(np.nanpercentile(sg_abs, 95)) if len(sg_abs) else None
        ),
        "median_detector_error_px": float(np.nanmedian(dr)) if len(dr) else None,
        "p95_detector_error_px": float(np.nanpercentile(dr, 95)) if len(dr) else None,
        "median_dx_px": float(np.nanmedian(dx)) if len(dx) else None,
        "median_dy_px": float(np.nanmedian(dy)) if len(dy) else None,
        "median_frame_correction_applied": (
            float(np.nanmedian(dz_corr)) if len(dz_corr) else 0.0
        ),
        "p95_abs_frame_correction_applied": (
            float(np.nanpercentile(np.abs(dz_corr), 95)) if len(dz_corr) else 0.0
        ),
        "convention": "g=R(axis,omega)@UB@h; ray=s0+g; detector intersection in XDS header lab frame; optional calibrated ZCAL frame correction applied before omega replay",
    }


def default_frame_path(cbf_dir: Path, frame: int, pattern: str):
    return cbf_dir / pattern.format(frame=int(frame))


class FrameCache:
    def __init__(self, cbf_dir: Path, pattern: str, max_items: int):
        self.cbf_dir = Path(cbf_dir)
        self.pattern = pattern
        self.max_items = int(max_items)
        self._fabio = require_fabio()
        self.cache = {}
        self.order = []

    def get(self, frame: int):
        frame = int(frame)
        if frame in self.cache:
            return self.cache[frame]
        path = default_frame_path(self.cbf_dir, frame, self.pattern)
        if not path.exists():
            return None, path
        arr = self._fabio.open(str(path)).data.astype(np.float32)
        self.cache[frame] = (arr, path)
        self.order.append(frame)
        while len(self.order) > self.max_items:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return arr, path


def robust_location_sigma(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    mad = float(1.4826 * np.median(np.abs(vals - med)))
    if not np.isfinite(mad) or mad <= 0:
        mad = float(np.std(vals)) if len(vals) > 1 else 1.0
    return med, max(mad, 1.0)


def profile_correlation(profile):
    if len(profile) < 2:
        return float("nan")
    frames = np.asarray([p["frame"] for p in profile], dtype=float)
    signals = np.asarray([p["signal"] for p in profile], dtype=float)
    if np.std(frames) <= 0 or np.std(signals) <= 0:
        return float("nan")
    return float(np.corrcoef(frames, signals)[0, 1])


def profile_model_correlation(profile):
    if len(profile) < 2:
        return float("nan")
    signals = np.asarray([p["signal"] for p in profile], dtype=float)
    model = np.asarray([p.get("model_profile_z", float("nan")) for p in profile], dtype=float)
    ok = np.isfinite(signals) & np.isfinite(model)
    if np.count_nonzero(ok) < 2 or np.std(signals[ok]) <= 0 or np.std(model[ok]) <= 0:
        return float("nan")
    return float(np.corrcoef(signals[ok], model[ok])[0, 1])


class BlochRockingProfileModel:
    """Per-row Bloch rocking-profile evaluator for raw-pixel profile fits."""

    def __init__(self, data: XDSData, model_path: Path, args):
        hkl_transform = str(getattr(args, "bloch_profile_hkl_transform", "identity") or "identity")
        profile_data = XDSData(data.path, hkl_transform=hkl_transform) if hkl_transform != "identity" else data
        model = parse_model(Path(model_path))
        cfg = Config(
            n_basis=int(args.bloch_profile_n_basis),
            n_points=1,
            sg_window=float(args.bloch_profile_sg_window),
            topup_d_min=float(args.bloch_profile_topup_d_min),
            max_support_sg=float(args.bloch_profile_max_support_sg),
            thickness_A=float(args.bloch_profile_thickness),
            thickness_model=str(args.bloch_profile_thickness_model),
            thickness_grad_frac=float(args.bloch_profile_thickness_grad_frac),
            thickness_sigma_frac=float(args.bloch_profile_thickness_sigma_frac),
            thickness_points=int(args.bloch_profile_thickness_points),
            absorption_mode=str(args.bloch_profile_absorption_mode),
            absorption_mu=float(args.bloch_profile_absorption_mu),
            absorption_frac=float(args.bloch_profile_absorption_frac),
            absorption_mu_g2=float(args.bloch_profile_absorption_mu_g2),
            absorption_mu_theta2=float(args.bloch_profile_absorption_mu_theta2),
            absorption_tds_b=float(args.bloch_profile_absorption_tds_b),
            ug_d_min=float(args.bloch_profile_ug_d_min),
            matrix_convention=str(args.bloch_profile_matrix_convention),
        )
        cfg._raw_profile_z_smooth_frames = float(args.bloch_profile_z_smooth_frames)
        cfg._raw_profile_z_gaussian_blend = float(args.bloch_profile_z_gaussian_blend)
        cfg._raw_profile_z_gaussian_sigma = float(args.bloch_profile_z_gaussian_sigma)
        cache_dir = (
            Path(args.bloch_profile_cache_dir)
            if args.bloch_profile_cache_dir
            else Path(__file__).resolve().parent / "cache"
        )
        self.port = XDSBlochPort(profile_data, model, cfg, cache_dir=cache_dir, res_hash=sha256(Path(model_path)))
        self.data = profile_data
        self._cache = {}

    def frame_profile(self, row_idx: int, frames) -> dict[int, float]:
        frames = tuple(sorted({int(f) for f in frames}))
        key = (int(row_idx), frames)
        if key in self._cache:
            return dict(self._cache[key])
        hkl = self.data.hkl[int(row_idx)]
        omega_center = self.port.omega_from_z(float(self.data.zcal[int(row_idx)]))
        omega_bragg = self.port.solve_omega_for_sg(hkl, omega_center, 0.0)
        basis = self.port.select_basis(hkl, omega_bragg, row_idx=int(row_idx))
        vals = []
        for frame in frames:
            omega = self.port.omega_from_z(float(frame))
            sg_basis = self.port.sg_for_hkls(basis, omega)
            H = self.port.hamiltonian(basis, sg_basis, absorb=True, row_idx=int(row_idx))
            intensity_t = [
                self.port.propagate_direct_to_target(H, float(t))
                for t in self.port.thick
            ]
            vals.append(float(np.sum(self.port.thick_w * np.asarray(intensity_t, dtype=float))))
        arr = np.asarray(vals, dtype=float)
        arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
        frame_arr = np.asarray(frames, dtype=float)
        smooth_sigma = float(getattr(self.port.cfg, "_raw_profile_z_smooth_frames", 0.0))
        if smooth_sigma > 0.0 and len(arr) > 1:
            dist = frame_arr[:, None] - frame_arr[None, :]
            kern = np.exp(-0.5 * (dist / max(smooth_sigma, 1e-12)) ** 2)
            arr = (kern @ arr) / np.maximum(np.sum(kern, axis=1), 1e-30)
        blend = float(getattr(self.port.cfg, "_raw_profile_z_gaussian_blend", 0.0))
        if blend > 0.0:
            z_sigma = float(getattr(self.port.cfg, "_raw_profile_z_gaussian_sigma", 0.0))
            if z_sigma <= 0.0:
                z_sigma = 1.0
            zcal = float(self.data.zcal[int(row_idx)])
            gauss = np.exp(-0.5 * ((frame_arr - zcal) / max(z_sigma, 1e-12)) ** 2)
            if float(np.max(gauss)) > 0.0:
                gauss /= float(np.max(gauss))
            arr = (1.0 - blend) * arr + blend * gauss
        if float(np.max(arr)) <= 0.0:
            arr = np.ones_like(arr, dtype=float)
        arr /= max(float(np.max(arr)), 1e-30)
        out = {int(frame): float(v) for frame, v in zip(frames, arr, strict=False)}
        self._cache[key] = dict(out)
        return out


def fit_plane_profile_amplitude(pixels, xcal, ycal, zcal, profile_z_by_frame, args, sigma_xy):
    rows = []
    y = []
    for p in pixels:
        raw = p.get("raw_value")
        if raw is None or p.get("mask_state") != "valid":
            continue
        frame = int(p["frame"])
        dx = float(p["dx_px"])
        dy = float(p["dy_px"])
        dz = float(p["dz_frame"])
        pz = float(profile_z_by_frame.get(frame, 0.0))
        pxy = math.exp(-0.5 * ((dx / max(sigma_xy, 1e-12)) ** 2 + (dy / max(sigma_xy, 1e-12)) ** 2))
        rows.append(
            [
                1.0,
                dx / max(float(args.xy_half_width), 1.0),
                dy / max(float(args.xy_half_width), 1.0),
                dz / max(float(args.frame_radius), 1.0),
                pxy * pz,
            ]
        )
        y.append(float(raw))
    if len(rows) < 8:
        raise ValueError("too few valid pixels for plane-plus-profile fit")
    X = np.asarray(rows, dtype=float)
    y = np.asarray(y, dtype=float)
    profile_sum = float(np.sum(X[:, -1]))
    if profile_sum <= 0.0:
        raise ValueError("zero Bloch/spatial profile support")
    X[:, -1] /= profile_sum

    keep = np.ones(len(y), dtype=bool)
    beta = None
    residual = None
    clip_sigma = float(getattr(args, "profile_fit_clip_sigma", 0.0))
    for _ in range(2 if clip_sigma > 0 else 1):
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        residual = y - X @ beta
        if clip_sigma <= 0:
            break
        med, sig = robust_location_sigma(residual[keep])
        new_keep = np.abs(residual - med) <= clip_sigma * max(sig, 1.0)
        if np.count_nonzero(new_keep) < max(8, X.shape[1] + 2) or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if beta is None or residual is None:
        raise RuntimeError("profile fit failed unexpectedly")
    _, resid_sig = robust_location_sigma(residual[keep])
    xtx_inv = np.linalg.pinv(X[keep].T @ X[keep])
    amp_sigma = float(max(resid_sig, 1.0) * math.sqrt(max(float(xtx_inv[-1, -1]), 0.0)))
    amp = float(beta[-1])
    if getattr(args, "profile_fit_nonnegative", False) and amp < 0.0:
        amp = 0.0
    return {
        "amplitude": amp,
        "sigma": float(max(amp_sigma, 1.0)),
        "residual_sigma": float(resid_sig),
        "background_plane": [float(v) for v in beta[:-1]],
        "n_fit_pixels": int(np.count_nonzero(keep)),
        "n_fit_pixels_rejected": int(len(y) - np.count_nonzero(keep)),
        "profile_support_sum_before_norm": profile_sum,
        "profile_weight2_sum_after_norm": float(np.sum(X[keep, -1] ** 2)),
    }


def integrate_observation(cache: FrameCache, data: XDSData, row_idx: int, args, bloch_profile_model=None):
    zcal = float(data.zcal[row_idx])
    xcal = float(data.xcal[row_idx])
    ycal = float(data.ycal[row_idx])
    center = int(round(zcal))
    xy_half = int(args.xy_half_width)
    frame_radius = int(args.frame_radius)
    fg_radius = float(args.foreground_radius)
    bg_inner = float(args.bg_inner)
    bg_outer = float(args.bg_outer)
    gain_sigma = float(args.sigma_gain)
    sigma_floor = float(args.sigma_floor)
    profile = []
    pixels = []
    missing_frames = []
    rejected_frames = []
    bg_values = []
    frame_bg_stats = {}
    fg_refs = []
    masked_count = 0
    for frame in range(center - frame_radius, center + frame_radius + 1):
        dz = float(frame) - zcal
        arr, path = cache.get(frame)
        if arr is None:
            missing_frames.append({"frame": int(frame), "path": str(path)})
            for y0 in range(int(round(ycal)) - xy_half, int(round(ycal)) + xy_half + 1):
                for x0 in range(
                    int(round(xcal)) - xy_half, int(round(xcal)) + xy_half + 1
                ):
                    dx = float(x0) - xcal
                    dy = float(y0) - ycal
                    r_xy = math.sqrt(dx * dx + dy * dy)
                    is_fg = r_xy <= fg_radius
                    is_bg = bg_inner <= r_xy <= bg_outer
                    pixels.append(
                        {
                            "frame": int(frame),
                            "x_px": int(x0),
                            "y_px": int(y0),
                            "raw_value": None,
                            "mask_state": "missing_frame",
                            "foreground": bool(is_fg),
                            "background": bool(is_bg),
                            "dx_px": dx,
                            "dy_px": dy,
                            "dz_frame": dz,
                            "r_xy_px": r_xy,
                            "r3": math.sqrt(dx * dx + dy * dy + dz * dz),
                        }
                    )
                    masked_count += 1
            continue
        frame_fg_values = []
        frame_bg_values = []
        for y0 in range(int(round(ycal)) - xy_half, int(round(ycal)) + xy_half + 1):
            for x0 in range(int(round(xcal)) - xy_half, int(round(xcal)) + xy_half + 1):
                dx = float(x0) - xcal
                dy = float(y0) - ycal
                r_xy = math.sqrt(dx * dx + dy * dy)
                is_fg = r_xy <= fg_radius
                is_bg = bg_inner <= r_xy <= bg_outer
                valid = 0 <= y0 - 1 < arr.shape[0] and 0 <= x0 - 1 < arr.shape[1]
                raw = None
                mask_state = "valid"
                if valid:
                    raw = float(arr[y0 - 1, x0 - 1])
                    if not math.isfinite(raw):
                        raw = None
                        mask_state = "nonfinite"
                        masked_count += 1
                else:
                    mask_state = "outside_detector"
                    masked_count += 1
                rec = {
                    "frame": int(frame),
                    "x_px": int(x0),
                    "y_px": int(y0),
                    "raw_value": raw,
                    "mask_state": mask_state,
                    "foreground": bool(is_fg),
                    "background": bool(is_bg),
                    "dx_px": dx,
                    "dy_px": dy,
                    "dz_frame": dz,
                    "r_xy_px": r_xy,
                    "r3": math.sqrt(dx * dx + dy * dy + dz * dz),
                }
                pixels.append(rec)
                if raw is not None and mask_state == "valid":
                    if is_bg:
                        bg_values.append(raw)
                        frame_bg_values.append(raw)
                    if is_fg:
                        frame_fg_values.append(raw)
                        fg_refs.append((raw, x0, y0, frame, dx, dy, dz, r_xy))
        bg_med_frame, bg_sig_frame = robust_location_sigma(frame_bg_values)
        frame_bg_stats[int(frame)] = (
            bg_med_frame,
            bg_sig_frame,
            int(len(frame_bg_values)),
        )
        signal_frame = (
            float(np.sum(np.asarray(frame_fg_values, dtype=float) - bg_med_frame))
            if frame_fg_values
            else 0.0
        )
        profile.append(
            {
                "frame": int(frame),
                "angle_deg": float(data.omega_from_z(frame)),
                "signal": signal_frame,
                "background_median": bg_med_frame,
                "background_sigma": bg_sig_frame,
                "n_foreground_pixels": int(len(frame_fg_values)),
                "n_background_pixels": int(len(frame_bg_values)),
            }
        )

    n_fg_possible = sum(1 for p in pixels if p["foreground"])
    n_fg_valid = sum(
        1
        for p in pixels
        if p["foreground"] and p["raw_value"] is not None and p["mask_state"] == "valid"
    )
    n_bg_valid = sum(
        1
        for p in pixels
        if p["background"] and p["raw_value"] is not None and p["mask_state"] == "valid"
    )
    if n_fg_valid == 0:
        return None, {
            "missing_frames": missing_frames,
            "rejected_frames": rejected_frames,
        }

    bg_mean, bg_sigma = robust_location_sigma(bg_values)
    fg_values = np.asarray([r[0] for r in fg_refs], dtype=float)
    frame_bg_for_fg = np.asarray(
        [frame_bg_stats.get(int(r[3]), (bg_mean, bg_sigma, 0))[0] for r in fg_refs],
        dtype=float,
    )
    frame_bg_sig_for_fg = np.asarray(
        [frame_bg_stats.get(int(r[3]), (bg_mean, bg_sigma, 0))[1] for r in fg_refs],
        dtype=float,
    )
    if args.background_model == "per_frame_median":
        bg_for_fg = frame_bg_for_fg
        bg_sigma_effective = (
            float(np.nanmedian(frame_bg_sig_for_fg))
            if len(frame_bg_sig_for_fg)
            else bg_sigma
        )
    else:
        bg_for_fg = np.full_like(fg_values, bg_mean, dtype=float)
        bg_sigma_effective = bg_sigma
    signal_terms = fg_values - bg_for_fg
    aperture_signal = float(np.sum(signal_terms))
    var_bg_mean = (bg_sigma_effective * bg_sigma_effective) / max(n_bg_valid, 1)
    aperture_sigma2 = float(
        np.sum(np.maximum(fg_values, 0.0))
        + (n_fg_valid * n_fg_valid) * var_bg_mean
        + gain_sigma * gain_sigma
        + sigma_floor * sigma_floor
    )
    aperture_sigma = float(math.sqrt(max(aperture_sigma2, 1.0)))

    dxs = np.asarray([r[4] for r in fg_refs], dtype=float)
    dys = np.asarray([r[5] for r in fg_refs], dtype=float)
    dzs = np.asarray([r[6] for r in fg_refs], dtype=float)
    sigma_xy = (
        float(args.profile_sigma_xy)
        if float(args.profile_sigma_xy) > 0
        else max(float(fg_radius) / 2.0, 0.75)
    )
    sigma_z = (
        float(args.profile_sigma_z)
        if float(args.profile_sigma_z) > 0
        else max(float(frame_radius) / 2.0, 0.75)
    )
    prof_w = np.exp(
        -0.5 * ((dxs / sigma_xy) ** 2 + (dys / sigma_xy) ** 2 + (dzs / sigma_z) ** 2)
    )
    ww = float(np.sum(prof_w * prof_w))
    sw = float(np.sum(prof_w))
    if ww > 0.0 and sw > 0.0:
        amp = float(np.sum(prof_w * signal_terms) / ww)
        profile_signal = float(amp * sw)
        residual = signal_terms - amp * prof_w
        resid_loc, resid_sig = robust_location_sigma(residual)
        # Profile-fit total-intensity variance from the fitted amplitude, plus
        # the same background-mean and floor terms used by the aperture target.
        pix_var = max(
            float(resid_sig * resid_sig),
            float(bg_sigma_effective * bg_sigma_effective),
            1.0,
        )
        profile_sigma2 = float(
            pix_var * (sw * sw) / ww
            + (sw * sw) * var_bg_mean
            + gain_sigma * gain_sigma
            + sigma_floor * sigma_floor
        )
        profile_sigma = float(math.sqrt(max(profile_sigma2, 1.0)))
        profile_amp = amp
        profile_resid_sigma = resid_sig
    else:
        profile_signal = aperture_signal
        profile_sigma = aperture_sigma
        profile_amp = float("nan")
        profile_resid_sigma = float("nan")

    bloch_profile_signal = profile_signal
    bloch_profile_sigma = profile_sigma
    bloch_profile_amp = float("nan")
    bloch_profile_resid_sigma = float("nan")
    bloch_profile_n_fit = 0
    bloch_profile_n_rejected = 0
    bloch_profile_background_plane = [float("nan")] * 4
    bloch_profile_weight2_sum = float("nan")
    bloch_profile_model_corr = float("nan")
    bloch_profile_fit_status = "not_requested"
    bloch_profile_fit_error = ""
    if args.raw_integration_mode == "bloch_profile_fit":
        bloch_profile_fit_status = "fallback"
        if bloch_profile_model is None:
            raise ValueError("bloch_profile_fit requires a BlochRockingProfileModel")
        valid_frames = sorted(
            {
                int(p["frame"])
                for p in pixels
                if p.get("raw_value") is not None and p.get("mask_state") == "valid"
            }
        )
        profile_z_by_frame = bloch_profile_model.frame_profile(int(row_idx), valid_frames)
        for frame_rec in profile:
            frame_rec["model_profile_z"] = float(profile_z_by_frame.get(int(frame_rec["frame"]), 0.0))
            frame_rec["target_sg_inv_A"] = float(
                sg_for_hkl(
                    data,
                    data.hkl[int(row_idx)],
                    data.omega_from_z(float(frame_rec["frame"])),
                )
            )
        bloch_profile_model_corr = profile_model_correlation(profile)
        try:
            fit = fit_plane_profile_amplitude(
                pixels, xcal, ycal, zcal, profile_z_by_frame, args, sigma_xy
            )
            bloch_profile_signal = float(fit["amplitude"])
            bloch_profile_sigma = float(
                math.sqrt(
                    max(float(fit["sigma"]) ** 2 + gain_sigma * gain_sigma + sigma_floor * sigma_floor, 1.0)
                )
            )
            bloch_profile_amp = float(fit["amplitude"])
            bloch_profile_resid_sigma = float(fit["residual_sigma"])
            bloch_profile_n_fit = int(fit["n_fit_pixels"])
            bloch_profile_n_rejected = int(fit["n_fit_pixels_rejected"])
            bloch_profile_background_plane = fit["background_plane"]
            bloch_profile_weight2_sum = float(fit["profile_weight2_sum_after_norm"])
            bloch_profile_fit_status = "ok"
        except Exception as exc:
            bloch_profile_signal = profile_signal
            bloch_profile_sigma = profile_sigma
            bloch_profile_fit_error = str(exc)

    if args.raw_integration_mode == "gaussian_profile":
        signal = profile_signal
        sigma = profile_sigma
    elif args.raw_integration_mode == "bloch_profile_fit":
        signal = bloch_profile_signal
        sigma = bloch_profile_sigma
    else:
        signal = aperture_signal
        sigma = aperture_sigma
    weights = np.maximum(signal_terms, 0.0)
    if float(np.sum(weights)) > 0.0:
        xs = np.asarray([r[1] for r in fg_refs], dtype=float)
        ys = np.asarray([r[2] for r in fg_refs], dtype=float)
        zs = np.asarray([r[3] for r in fg_refs], dtype=float)
        x_centroid = float(np.sum(weights * xs) / np.sum(weights))
        y_centroid = float(np.sum(weights * ys) / np.sum(weights))
        z_centroid = float(np.sum(weights * zs) / np.sum(weights))
    else:
        x_centroid = xcal
        y_centroid = ycal
        z_centroid = zcal
    centroid_resid = math.sqrt(
        (x_centroid - xcal) ** 2 + (y_centroid - ycal) ** 2 + (z_centroid - zcal) ** 2
    )
    observed_profile_fraction = float(n_fg_valid / max(n_fg_possible, 1))
    return (
        {
            "raw_integration_version": f"row_contract_v2_{args.raw_integration_mode}",
            "raw_integration_mode": args.raw_integration_mode,
            "background_model": args.background_model,
            "shoebox_half_width_px": int(xy_half),
            "shoebox_frame_radius": int(frame_radius),
            "foreground_radius_px": fg_radius,
            "background_inner_px": bg_inner,
            "background_outer_px": bg_outer,
            "profile_sigma_xy_px": sigma_xy,
            "profile_sigma_z_frames": sigma_z,
            "Iobs_raw": signal,
            "sigma_Iobs_raw": sigma,
            "I_raw_sum": signal,
            "sigma_raw_sum": sigma,
            "I_raw_aperture_sum": aperture_signal,
            "sigma_raw_aperture_sum": aperture_sigma,
            "I_raw_profile_fit": profile_signal,
            "sigma_raw_profile_fit": profile_sigma,
            "profile_fit_amplitude": profile_amp,
            "profile_fit_residual_sigma": profile_resid_sigma,
            "profile_fit_weight_sum": sw,
            "profile_fit_weight2_sum": ww,
            "I_raw_bloch_profile_fit": bloch_profile_signal,
            "sigma_raw_bloch_profile_fit": bloch_profile_sigma,
            "bloch_profile_fit_amplitude": bloch_profile_amp,
            "bloch_profile_fit_residual_sigma": bloch_profile_resid_sigma,
            "bloch_profile_fit_n_pixels": bloch_profile_n_fit,
            "bloch_profile_fit_n_rejected": bloch_profile_n_rejected,
            "bloch_profile_fit_background_plane": bloch_profile_background_plane,
            "bloch_profile_fit_weight2_sum": bloch_profile_weight2_sum,
            "bloch_profile_model_corr": bloch_profile_model_corr,
            "bloch_profile_fit_status": bloch_profile_fit_status,
            "bloch_profile_fit_error": bloch_profile_fit_error,
            "background_mean": bg_mean,
            "background_sigma": bg_sigma,
            "background_sigma_effective": bg_sigma_effective,
            "n_foreground_pixels": int(n_fg_valid),
            "n_background_pixels": int(n_bg_valid),
            "n_masked_pixels": int(masked_count),
            "observed_profile_fraction": observed_profile_fraction,
            "centroid_x": x_centroid,
            "centroid_y": y_centroid,
            "centroid_z": z_centroid,
            "centroid_residual_px": float(centroid_resid),
            "profile_corr": profile_correlation(profile),
            "overlap_flag": False,
            "raw_z_centroid": z_centroid,
            "profile": profile,
            "pixels": pixels,
            "missing_frames": missing_frames,
            "rejected_frames": rejected_frames,
        },
        None,
    )


def model_metadata(path: Path):
    meta = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "kind": path.suffix.lower().lstrip("."),
    }
    try:
        if path.suffix.lower() == ".res" or path.suffix.lower() == ".ins":
            model = parse_res(path)
            meta["cell"] = [float(v) for v in model["cell"]]
            meta["n_atoms"] = int(len(model["atoms"]))
            meta["source_parser"] = "SHELX_RES"
        else:
            doc = gemmi.cif.read_file(str(path))
            block = doc.sole_block()
            vals = []
            for tag in (
                "_cell_length_a",
                "_cell_length_b",
                "_cell_length_c",
                "_cell_angle_alpha",
                "_cell_angle_beta",
                "_cell_angle_gamma",
            ):
                val = block.find_value(tag)
                vals.append(float(str(val).split("(")[0]) if val is not None else None)
            meta["cell"] = vals
            meta["source_parser"] = "gemmi_cif"
    except Exception as exc:
        meta["parse_warning"] = str(exc)
    return meta


def group_id_from_z(data: XDSData, zcal: float, group_width_frames: float) -> int:
    return int(
        math.floor(
            (float(zcal) - float(data.starting_frame))
            / max(float(group_width_frames), 1e-12)
        )
    )


def z_from_omega(data: XDSData, omega_deg: float) -> float:
    return float(
        data.starting_frame
        + (float(omega_deg) - float(data.starting_angle))
        / float(data.oscillation_range)
    )


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
        deriv = (
            sg_for_hkl(data, hkl, omega + step) - sg_for_hkl(data, hkl, omega - step)
        ) / (2.0 * step)
        if abs(deriv) < 1e-12:
            break
        omega -= sg / deriv
    return float(omega), sg_for_hkl(data, hkl, omega)


def calibration_delta_for_idx(
    calibration_by_idx: dict[int, float] | None, idx: int
) -> float:
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
    raw_delta = raw_z - (
        np.asarray([data.zcal[idx] for idx in row_indices], dtype=float) + deltas
    )
    return {
        "median_abs_center_sg_inv_A": (
            float(np.nanmedian(sg_abs)) if len(sg_abs) else None
        ),
        "p95_abs_center_sg_inv_A": (
            float(np.nanpercentile(sg_abs, 95)) if len(sg_abs) else None
        ),
        "zero_sg_delta_frame_median_after": (
            float(np.nanmedian(zero_delta)) if len(zero_delta) else None
        ),
        "zero_sg_delta_frame_p95_abs_after": (
            float(np.nanpercentile(np.abs(zero_delta), 95)) if len(zero_delta) else None
        ),
        "raw_z_minus_calibrated_center_median": (
            float(np.nanmedian(raw_delta)) if np.any(np.isfinite(raw_delta)) else None
        ),
        "raw_z_minus_calibrated_center_p95_abs": (
            float(np.nanpercentile(np.abs(raw_delta), 95))
            if np.any(np.isfinite(raw_delta))
            else None
        ),
        "frame_correction_median": float(np.nanmedian(deltas)) if len(deltas) else 0.0,
        "frame_correction_p95_abs": (
            float(np.nanpercentile(np.abs(deltas), 95)) if len(deltas) else 0.0
        ),
    }


def fit_geometry_calibration(
    data: XDSData, row_indices, integrations, requested_model: str, threshold_sg: float
):
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
    zero_sg_frames = np.asarray(
        [z_from_omega(data, a) for a in zero_angles], dtype=float
    )
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
        coeff = np.polyfit(
            x[np.isfinite(target_delta)], target_delta[np.isfinite(target_delta)], 1
        )
        candidates["linear_drift"] = coeff[0] * x + coeff[1]
    else:
        coeff = np.asarray(
            [0.0, float(np.nanmedian(target_delta)) if len(target_delta) else 0.0]
        )
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
        candidates["spline"] = np.interp(
            zcal, np.asarray(knots_z), np.asarray(knots_delta)
        )
    else:
        knots_z = [float(np.nanmedian(zcal))] if len(zcal) else [0.0]
        knots_delta = (
            [float(np.nanmedian(target_delta))] if len(target_delta) else [0.0]
        )
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
    calibration_by_idx = {
        int(idx): float(delta) for idx, delta in zip(row_indices, selected_deltas, strict=False)
    }
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
            float(np.nanpercentile(np.abs(raw_delta), 95))
            if np.any(np.isfinite(raw_delta))
            else None
        ),
        "zero_sg_delta_frame_median_before": (
            float(np.nanmedian(target_delta)) if len(target_delta) else None
        ),
        "zero_sg_delta_frame_p95_abs_before": (
            float(np.nanpercentile(np.abs(target_delta), 95))
            if len(target_delta)
            else None
        ),
        "max_zero_sg_solve_residual_inv_A": (
            float(np.nanmax(np.abs(zero_resid))) if len(zero_resid) else None
        ),
        "linear_coefficients_delta_frame_vs_centered_z": [float(v) for v in coeff],
        "spline_knots": {
            "zcal_frame": [float(v) for v in knots_z],
            "delta_frame": [float(v) for v in knots_delta],
        },
        "candidate_summaries": summaries,
        "selected_summary": summaries[selected],
        "calibration_by_idx": calibration_by_idx,
    }


def build_observation_group_grid(
    data: XDSData,
    row_indices,
    width_deg: float,
    step_deg: float,
    calibration_by_idx: dict[int, float] | None = None,
):
    calibration_by_idx = calibration_by_idx or {}
    angles = np.asarray(
        [
            data.omega_from_z(
                data.zcal[idx] + calibration_delta_for_idx(calibration_by_idx, int(idx))
            )
            for idx in row_indices
        ],
        dtype=float,
    )
    if len(angles) == 0:
        return []
    width = float(width_deg)
    step = float(step_deg)
    half = 0.5 * width
    first_center = float(data.starting_angle) + half
    while first_center > float(np.nanmin(angles)) + half:
        first_center -= step
    while first_center + half < float(np.nanmin(angles)):
        first_center += step
    centers = []
    center = first_center
    max_angle = float(np.nanmax(angles))
    while center - half <= max_angle:
        centers.append(center)
        center += step
    return centers


def choose_observation_group(angle_deg: float, centers: list[float]) -> tuple[int, float]:
    if not centers:
        return 0, float("nan")
    diffs = [abs(float(angle_deg) - c) for c in centers]
    idx = int(np.argmin(diffs))
    return idx + 1, float(centers[idx])


def observation_group_record(
    data: XDSData,
    frame_id: int,
    center_angle: float,
    width_deg: float,
    step_deg: float,
    source_hkl: Path,
    model_meta: dict,
    detector: dict,
    xparm_meta: dict | None,
):
    half = 0.5 * float(width_deg)
    start_angle = float(center_angle) - half
    end_angle = float(center_angle) + half
    z_start = z_from_omega(data, start_angle)
    z_end = z_from_omega(data, end_angle)
    center_frame = z_from_omega(data, center_angle)
    return {
        **frame_contract_fields("xds_integrate_hkl_row_contract_observation_group"),
        "dataset_id": "sample",
        "frame_id": int(frame_id),
        "observation_group_id": int(frame_id),
        "angle_deg": float(center_angle),
        "angle_center_deg": float(center_angle),
        "angle_start_deg": start_angle,
        "angle_end_deg": end_angle,
        "angle_width_deg": float(width_deg),
        "angle_step_deg": float(step_deg),
        "precession_angle_deg": half,
        "image_number": int(round(center_frame)),
        "image_center_frame": float(center_frame),
        "source_image_range": [
            int(math.floor(min(z_start, z_end))),
            int(math.ceil(max(z_start, z_end))),
        ],
        "wavelength_A": float(data.wavelength),
        "k_inv_A": float(data.k),
        "s0_lab": data.s0.astype(float).tolist(),
        "rotation_axis_lab": data.rotation_axis_lab.astype(float).tolist(),
        "UB": (
            data.setting_rotation
            @ data.fixed_rotation
            @ data.A_for_omega(center_angle)
        ).astype(float).tolist(),
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
        "group_id": int(frame_id),
        "source": "xds_geometry_row_contract_observation_group",
        "geometry_source": str(source_hkl.resolve()),
        "xparm": xparm_meta,
        "starting_model": model_meta,
        "observation_group_scale": 1.0,
        "observation_group_scale_source": "initial_unity_prior",
    }


def write_frames(
    path: Path,
    data: XDSData,
    row_indices,
    args,
    source_hkl: Path,
    model_meta: dict,
    detector: dict,
    xparm_meta: dict | None,
    calibration_by_idx: dict[int, float] | None = None,
):
    calibration_by_idx = calibration_by_idx or {}
    centers = build_observation_group_grid(
        data,
        row_indices,
        args.observation_group_width_deg,
        args.observation_group_step_deg,
        calibration_by_idx,
    )
    frames = {
        i
        + 1: observation_group_record(
            data,
            i + 1,
            center,
            args.observation_group_width_deg,
            args.observation_group_step_deg,
            source_hkl,
            model_meta,
            detector,
            xparm_meta,
        )
        for i, center in enumerate(centers)
    }
    assignment = {}
    for idx in row_indices:
        z_calibrated = float(data.zcal[idx]) + calibration_delta_for_idx(
            calibration_by_idx, int(idx)
        )
        omega = float(data.omega_from_z(z_calibrated))
        fid, center = choose_observation_group(omega, centers)
        assignment[int(idx)] = {
            "observation_group_id": int(fid),
            "group_angle_center_deg": center,
            "zcal_calibrated_frame": z_calibrated,
            "frame_correction_applied": calibration_delta_for_idx(
                calibration_by_idx, int(idx)
            ),
        }
        if fid in frames:
            rec = frames[fid]
            rec.setdefault("supporting_observation_ids", []).append(int(idx))
            rec["n_observations"] = int(len(rec["supporting_observation_ids"]))
            rec["zcal_min"] = min(
                float(rec.get("zcal_min", data.zcal[idx])), float(data.zcal[idx])
            )
            rec["zcal_max"] = max(
                float(rec.get("zcal_max", data.zcal[idx])), float(data.zcal[idx])
            )
            rec["zcal_calibrated_min"] = min(
                float(rec.get("zcal_calibrated_min", z_calibrated)), z_calibrated
            )
            rec["zcal_calibrated_max"] = max(
                float(rec.get("zcal_calibrated_max", z_calibrated)), z_calibrated
            )
            rec["omega_min"] = min(float(rec.get("omega_min", omega)), omega)
            rec["omega_max"] = max(float(rec.get("omega_max", omega)), omega)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in sorted(frames.values(), key=lambda r: r["observation_group_id"]):
            rec = dict(rec)
            rec["real_space_axes"] = data.real_matrix.astype(float).tolist()
            rec["unit_cell"] = [
                float(data.cell.a),
                float(data.cell.b),
                float(data.cell.c),
                float(data.cell.alpha),
                float(data.cell.beta),
                float(data.cell.gamma),
            ]
            rec["image_range"] = rec["source_image_range"]
            f.write(json.dumps(rec, allow_nan=True) + "\n")
    return len(frames), assignment


def observation_record(
    data: XDSData, idx: int, integration: dict, args, observation_group_assignment: dict
):
    z = float(data.zcal[idx])
    assignment = observation_group_assignment.get(
        int(idx),
        {
            "observation_group_id": group_id_from_z(data, z, args.group_width_frames),
            "group_angle_center_deg": float(data.omega_from_z(z)),
            "zcal_calibrated_frame": z,
            "frame_correction_applied": 0.0,
        },
    )
    gid = int(assignment["observation_group_id"])
    h, k, l = (int(v) for v in data.hkl[idx])
    z_calibrated = float(assignment.get("zcal_calibrated_frame", z))
    frame_correction = float(
        assignment.get("frame_correction_applied", z_calibrated - z)
    )
    omega_xds_zcal = float(data.omega_from_z(z))
    omega_zcal = float(data.omega_from_z(z_calibrated))
    sg_at_xds_zcal = sg_for_hkl(data, (h, k, l), omega_xds_zcal)
    sg_at_zcal = sg_for_hkl(data, (h, k, l), omega_zcal)
    zero_sg_angle, zero_sg_residual = solve_zero_sg_angle(data, (h, k, l), omega_zcal)
    zero_sg_frame = z_from_omega(data, zero_sg_angle)
    peak_fraction = max(float(data.peak[idx]) / 100.0, 1e-12)
    rlp = float(data.rlp[idx])
    i_recorded = float(integration["I_raw_sum"])
    sig_recorded = float(integration["sigma_raw_sum"])
    i_complete = float(i_recorded * rlp / peak_fraction)
    sig_complete = float(sig_recorded * abs(rlp / peak_fraction))
    return {
        **observation_contract_fields("xds_integrate_hkl", h, k, l),
        "obs_id": int(idx),
        "dataset_id": "sample",
        "h": h,
        "k": k,
        "l": l,
        "signed_h": h,
        "signed_k": k,
        "signed_l": l,
        "hkl": [h, k, l],
        "signed_hkl_source": "xds_integrate_hkl",
        "frame_id": int(round(z)),
        "image_number": int(round(z)),
        "calibrated_frame_id": int(round(z_calibrated)),
        "zcal_original_frame": z,
        "zcal_calibrated_frame": z_calibrated,
        "frame_correction_applied": frame_correction,
        "observation_group_id": int(gid),
        "group_angle_center_deg": float(assignment["group_angle_center_deg"]),
        "group_angle_width_deg": float(args.observation_group_width_deg),
        "group_angle_step_deg": float(args.observation_group_step_deg),
        "group_id": int(gid),
        "target_alpha0_deg": omega_zcal,
        "target_alpha0_xds_deg": omega_xds_zcal,
        "target_omega_zcal_deg": omega_zcal,
        "target_omega_xds_zcal_deg": omega_xds_zcal,
        "target_zero_sg_angle_deg": zero_sg_angle,
        "target_zero_sg_frame": zero_sg_frame,
        "target_zero_sg_delta_deg": float(zero_sg_angle - omega_zcal),
        "target_zero_sg_delta_frame": float(zero_sg_frame - z_calibrated),
        "target_sg_at_zcal_inv_A": sg_at_zcal,
        "target_sg_at_xds_zcal_inv_A": sg_at_xds_zcal,
        "target_zero_sg_residual_inv_A": zero_sg_residual,
        "xcal_px": float(data.xcal[idx]),
        "ycal_px": float(data.ycal[idx]),
        "zcal_frame": z,
        "xobs_px": float(data.xobs[idx]),
        "yobs_px": float(data.yobs[idx]),
        "zobs_frame": float(data.zobs[idx]),
        "rlp": float(data.rlp[idx]),
        "peak_percent": float(data.peak[idx]),
        "corr": float(data.corr[idx]),
        "maxc": float(data.maxc[idx]),
        "iseg": int(data.iseg[idx]),
        "raw_box_id": f"sample_{int(idx)}",
        "status": "candidate",
        "shoebox_pixels_file": "shoebox_pixels.jsonl.gz",
        "shoebox_pixel_record_id": int(idx),
        "control_values": {
            "Iobs_xds_profile_lp_corrected": float(data.iobs[idx]),
            "sigma_xds": float(data.sigma[idx]),
            "recorded_intensity_proxy_Iobs_peak_over_rlp": float(
                data.recorded_proxy()[idx]
            ),
            "xds_complete_from_raw_recorded": i_complete,
            "sigma_complete_from_raw_recorded": sig_complete,
            "xds_completion_factor_rlp_over_peak_fraction": float(rlp / peak_fraction),
        },
        "xds_profile_diagnostics": {
            "alf0": float(data.alf0[idx]),
            "bet0": float(data.bet0[idx]),
            "alf1": float(data.alf1[idx]),
            "bet1": float(data.bet1[idx]),
            "psi": float(data.psi[idx]),
        },
        "Iobs_raw": float(integration["Iobs_raw"]),
        "sigma_Iobs_raw": float(integration["sigma_Iobs_raw"]),
        "I_raw_sum": i_recorded,
        "sigma_raw_sum": sig_recorded,
        "I_raw_aperture_sum": float(integration.get("I_raw_aperture_sum", i_recorded)),
        "sigma_raw_aperture_sum": float(
            integration.get("sigma_raw_aperture_sum", sig_recorded)
        ),
        "I_raw_profile_fit": float(integration.get("I_raw_profile_fit", i_recorded)),
        "sigma_raw_profile_fit": float(
            integration.get("sigma_raw_profile_fit", sig_recorded)
        ),
        "profile_fit_amplitude": float(
            integration.get("profile_fit_amplitude", float("nan"))
        ),
        "profile_fit_residual_sigma": float(
            integration.get("profile_fit_residual_sigma", float("nan"))
        ),
        "profile_fit_weight_sum": float(
            integration.get("profile_fit_weight_sum", float("nan"))
        ),
        "profile_fit_weight2_sum": float(
            integration.get("profile_fit_weight2_sum", float("nan"))
        ),
        "I_raw_bloch_profile_fit": float(
            integration.get("I_raw_bloch_profile_fit", integration.get("I_raw_profile_fit", i_recorded))
        ),
        "sigma_raw_bloch_profile_fit": float(
            integration.get(
                "sigma_raw_bloch_profile_fit",
                integration.get("sigma_raw_profile_fit", sig_recorded),
            )
        ),
        "bloch_profile_fit_amplitude": float(
            integration.get("bloch_profile_fit_amplitude", float("nan"))
        ),
        "bloch_profile_fit_residual_sigma": float(
            integration.get("bloch_profile_fit_residual_sigma", float("nan"))
        ),
        "bloch_profile_fit_n_pixels": int(
            integration.get("bloch_profile_fit_n_pixels", 0)
        ),
        "bloch_profile_fit_n_rejected": int(
            integration.get("bloch_profile_fit_n_rejected", 0)
        ),
        "bloch_profile_fit_weight2_sum": float(
            integration.get("bloch_profile_fit_weight2_sum", float("nan"))
        ),
        "bloch_profile_model_corr": (
            float(integration["bloch_profile_model_corr"])
            if math.isfinite(float(integration.get("bloch_profile_model_corr", float("nan"))))
            else None
        ),
        "bloch_profile_fit_status": integration.get(
            "bloch_profile_fit_status", "not_requested"
        ),
        "bloch_profile_fit_error": integration.get("bloch_profile_fit_error", ""),
        "I_recorded_raw": i_recorded,
        "sigma_recorded_raw": sig_recorded,
        "I_complete_xdslike": i_complete,
        "sigma_complete_xdslike": sig_complete,
        "completion_factor_rlp_over_peak_fraction": float(rlp / peak_fraction),
        "target_interpretations": {
            "recorded_raw": {
                "I": i_recorded,
                "sigma": sig_recorded,
                "definition": "background-subtracted raw shoebox sum, closer to XDS recorded intensity I",
            },
            "complete_xdslike": {
                "I": i_complete,
                "sigma": sig_complete,
                "definition": "I_raw_sum * RLP / (PEAK/100), matching XDS IOBS completion convention",
            },
        },
        "background_mean": float(integration["background_mean"]),
        "background_sigma": float(integration["background_sigma"]),
        "background_sigma_effective": float(
            integration.get(
                "background_sigma_effective", integration["background_sigma"]
            )
        ),
        "n_foreground_pixels": int(integration["n_foreground_pixels"]),
        "n_background_pixels": int(integration["n_background_pixels"]),
        "n_masked_pixels": int(integration["n_masked_pixels"]),
        "observed_profile_fraction": float(integration["observed_profile_fraction"]),
        "centroid_x": float(integration["centroid_x"]),
        "centroid_y": float(integration["centroid_y"]),
        "centroid_z": float(integration["centroid_z"]),
        "centroid_residual_px": float(integration["centroid_residual_px"]),
        "profile_corr": (
            float(integration["profile_corr"])
            if math.isfinite(float(integration["profile_corr"]))
            else None
        ),
        "overlap_flag": bool(integration["overlap_flag"]),
        "raw_integration_version": integration["raw_integration_version"],
        "raw_integration_mode": integration.get("raw_integration_mode", "sum"),
        "background_model": integration.get("background_model", "global_median"),
        "primary_intensity_field": "Iobs_raw",
        "primary_sigma_field": "sigma_Iobs_raw",
        "Iobs_xds": float(data.iobs[idx]),
        "sigma_xds": float(data.sigma[idx]),
        "xcal": float(data.xcal[idx]),
        "ycal": float(data.ycal[idx]),
        "zcal": z,
        "xobs": float(data.xobs[idx]),
        "yobs": float(data.yobs[idx]),
        "zobs": float(data.zobs[idx]),
        "raw_z_centroid": float(integration["raw_z_centroid"]),
        "bbox_model": {
            "type": "3d_square_box_with_circular_foreground_and_annular_background",
            "frame_radius": int(args.frame_radius),
            "xy_half_width_px": int(args.xy_half_width),
            "foreground_radius_px": float(args.foreground_radius),
            "bg_inner_px": float(args.bg_inner),
            "bg_outer_px": float(args.bg_outer),
        },
        "quality": {
            "rlp": float(data.rlp[idx]),
            "peak": float(data.peak[idx]),
            "peak_percent": float(data.peak[idx]),
            "corr": float(data.corr[idx]),
            "maxc": float(data.maxc[idx]),
            "alf0": float(data.alf0[idx]),
            "bet0": float(data.bet0[idx]),
            "alf1": float(data.alf1[idx]),
            "bet1": float(data.bet1[idx]),
            "psi": float(data.psi[idx]),
            "iseg": int(data.iseg[idx]),
            "large_geometry_residual_flag": bool(
                abs(sg_at_zcal) > float(args.max_center_sg_for_quality)
            ),
            "bad_centroid_flag": bool(
                float(integration["centroid_residual_px"])
                > float(args.max_centroid_residual_px)
            ),
            "poor_profile_corr_flag": bool(
                integration["profile_corr"] is not None
                and math.isfinite(float(integration["profile_corr"]))
                and abs(float(integration["profile_corr"]))
                < float(args.min_abs_profile_corr)
            ),
            "bad_background_flag": bool(
                int(integration["n_background_pixels"])
                < int(args.min_background_pixels)
                or float(integration["background_sigma"]) <= 0.0
            ),
            "edge_flag": bool(
                data.xcal[idx] - args.xy_half_width < 1
                or data.ycal[idx] - args.xy_half_width < 1
                or (
                    data.xcal[idx] + args.xy_half_width
                    > (args.detector_nx_for_flags or 1e12)
                )
                or (
                    data.ycal[idx] + args.xy_half_width
                    > (args.detector_ny_for_flags or 1e12)
                )
            ),
            "overlap_flag": False,
        },
        "profile": integration["profile"],
        "shoebox_support": {
            "xy_half_width_px": int(integration["shoebox_half_width_px"]),
            "frame_radius": int(integration["shoebox_frame_radius"]),
            "foreground_radius_px": float(integration["foreground_radius_px"]),
            "background_inner_px": float(integration["background_inner_px"]),
            "background_outer_px": float(integration["background_outer_px"]),
            "raw_integration_mode": integration.get("raw_integration_mode", "sum"),
            "background_model": integration.get("background_model", "global_median"),
            "profile_sigma_xy_px": float(
                integration.get("profile_sigma_xy_px", float("nan"))
            ),
            "profile_sigma_z_frames": float(
                integration.get("profile_sigma_z_frames", float("nan"))
            ),
        },
        "missing_frames": integration["missing_frames"],
        "rejected_frames": integration["rejected_frames"],
    }


def write_xds_like(
    path: Path,
    data: XDSData,
    row_indices,
    integrations,
    source_hkl: Path,
    target_mode: str,
    calibration_by_idx: dict[int, float] | None = None,
):
    calibration_by_idx = calibration_by_idx or {}
    if target_mode not in {"recorded_raw", "complete_xdslike"}:
        raise ValueError(f"unknown target_mode {target_mode}")
    header_lines = []
    for line in source_hkl.read_text(errors="replace").splitlines():
        if not line.startswith("!"):
            break
        if line.startswith("!OUTPUT_FILE="):
            header_lines.append("!OUTPUT_FILE=RAW_DYNROWS_XDSLIKE.HKL")
        elif line.startswith("!END_OF_HEADER"):
            continue
        else:
            header_lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write(
            "!RAW_DYNROWS_SOURCE= build_dynrows_xds.py raw CBF 3D shoebox integration\n"
        )
        f.write(f"!RAW_DYNROWS_TARGET_MODE= {target_mode}\n")
        f.write(
            "!RAW_DYNROWS_GEOMETRY= XDS predicted XCAL YCAL ZCAL and XOBS YOBS ZOBS are preserved; raw centroids are stored in dynrows.jsonl\n"
        )
        if target_mode == "recorded_raw":
            f.write(
                "!RAW_DYNROWS_TARGET_DEFINITION= IOBS=I_raw_sum SIGMA=sigma_raw_sum\n"
            )
        else:
            f.write(
                "!RAW_DYNROWS_TARGET_DEFINITION= IOBS=I_raw_sum*RLP/(PEAK/100) SIGMA=sigma_raw_sum*RLP/(PEAK/100)\n"
            )
        f.write("!END_OF_HEADER\n")
        for idx, integ in zip(row_indices, integrations, strict=False):
            if target_mode == "recorded_raw":
                iobs = float(integ["I_raw_sum"])
                sig = float(integ["sigma_raw_sum"])
            else:
                peak_fraction = max(float(data.peak[idx]) / 100.0, 1e-12)
                factor = float(data.rlp[idx]) / peak_fraction
                iobs = float(integ["I_raw_sum"] * factor)
                sig = float(abs(integ["sigma_raw_sum"] * factor))
            row = [
                *data.hkl[idx].tolist(),
                iobs,
                sig,
                data.xcal[idx],
                data.ycal[idx],
                data.zcal[idx]
                + calibration_delta_for_idx(calibration_by_idx, int(idx)),
                data.rlp[idx],
                data.peak[idx],
                data.corr[idx],
                data.maxc[idx],
                data.xobs[idx],
                data.yobs[idx],
                data.zobs[idx],
                data.alf0[idx],
                data.bet0[idx],
                data.alf1[idx],
                data.bet1[idx],
                data.psi[idx],
                data.iseg[idx],
            ]
            f.write(
                " ".join(
                    str(int(v)) if n in {0, 1, 2, 20} else f"{float(v):.6g}"
                    for n, v in enumerate(row)
                )
                + "\n"
            )


def write_xds_control(
    path: Path,
    data: XDSData,
    row_indices,
    source_hkl: Path,
    target_mode: str,
    calibration_by_idx: dict[int, float] | None = None,
):
    calibration_by_idx = calibration_by_idx or {}
    if target_mode not in {"xds_iobs", "xds_recorded_proxy"}:
        raise ValueError(f"unknown control target_mode {target_mode}")
    header_lines = []
    for line in source_hkl.read_text(errors="replace").splitlines():
        if not line.startswith("!"):
            break
        if line.startswith("!OUTPUT_FILE="):
            header_lines.append(f"!OUTPUT_FILE={target_mode.upper()}_CALIBRATED.HKL")
        elif line.startswith("!END_OF_HEADER"):
            continue
        else:
            header_lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write(
            "!RAW_DYNROWS_SOURCE= build_dynrows_xds.py control target with calibrated ZCAL geometry\n"
        )
        f.write(f"!RAW_DYNROWS_TARGET_MODE= {target_mode}\n")
        f.write(
            "!RAW_DYNROWS_GEOMETRY= ZCAL is corrected by geometry_calibration.json; detector coordinates and signed HKL are unchanged\n"
        )
        f.write("!END_OF_HEADER\n")
        recorded = data.recorded_proxy()
        recorded_sigma = (
            data.sigma
            * np.maximum(data.peak, 0.0)
            / 100.0
            / np.maximum(data.rlp, 1e-12)
        )
        for idx in row_indices:
            if target_mode == "xds_iobs":
                iobs = float(data.iobs[idx])
                sig = float(data.sigma[idx])
            else:
                iobs = float(recorded[idx])
                sig = float(abs(recorded_sigma[idx]))
            row = [
                *data.hkl[idx].tolist(),
                iobs,
                sig,
                data.xcal[idx],
                data.ycal[idx],
                data.zcal[idx]
                + calibration_delta_for_idx(calibration_by_idx, int(idx)),
                data.rlp[idx],
                data.peak[idx],
                data.corr[idx],
                data.maxc[idx],
                data.xobs[idx],
                data.yobs[idx],
                data.zobs[idx],
                data.alf0[idx],
                data.bet0[idx],
                data.alf1[idx],
                data.bet1[idx],
                data.psi[idx],
                data.iseg[idx],
            ]
            f.write(
                " ".join(
                    str(int(v)) if n in {0, 1, 2, 20} else f"{float(v):.6g}"
                    for n, v in enumerate(row)
                )
                + "\n"
            )


def mark_overlap_flags(records: list[dict], xy_px: float, z_frames: float):
    """Mark simple 3D predicted-position overlaps without filtering rows."""
    if not records:
        return
    xy_px = float(xy_px)
    z_frames = float(z_frames)
    z_radius = int(math.ceil(max(z_frames, 0.0)))
    buckets: dict[int, list[int]] = {}
    for i, rec in enumerate(records):
        zbin = int(round(float(rec["zcal_frame"])))
        overlaps = False
        for dz in range(-z_radius, z_radius + 1):
            for j in buckets.get(zbin + dz, []):
                other = records[j]
                if (
                    abs(float(rec["zcal_frame"]) - float(other["zcal_frame"]))
                    > z_frames
                ):
                    continue
                dx = float(rec["xcal_px"]) - float(other["xcal_px"])
                dy = float(rec["ycal_px"]) - float(other["ycal_px"])
                if math.sqrt(dx * dx + dy * dy) <= xy_px:
                    overlaps = True
                    other["overlap_flag"] = True
                    other.setdefault("quality", {})["overlap_flag"] = True
                    rec.setdefault("overlap_partners", []).append(int(other["obs_id"]))
                    other.setdefault("overlap_partners", []).append(int(rec["obs_id"]))
            if overlaps:
                rec["overlap_flag"] = True
                rec.setdefault("quality", {})["overlap_flag"] = True
        buckets.setdefault(zbin, []).append(i)


def write_quality_mask(records: list[dict], path: Path, calibration: dict, args):
    excluded_flags = [
        "large_geometry_residual_flag",
        "bad_centroid_flag",
        "poor_profile_corr_flag",
        "bad_background_flag",
        "edge_flag",
        "overlap_flag",
    ]
    accepted_local = []
    accepted_obs = []
    excluded = []
    flag_counts = {k: 0 for k in excluded_flags}
    for local_idx, rec in enumerate(records):
        q = rec.get("quality") or {}
        active = [k for k in excluded_flags if bool(q.get(k, False))]
        for k in active:
            flag_counts[k] += 1
        if active:
            excluded.append(
                {
                    "local_row_index": int(local_idx),
                    "obs_id": int(rec["obs_id"]),
                    "hkl": [int(rec["h"]), int(rec["k"]), int(rec["l"])],
                    "reasons": active,
                    "target_sg_at_zcal_inv_A": rec.get("target_sg_at_zcal_inv_A"),
                    "centroid_residual_px": rec.get("centroid_residual_px"),
                    "profile_corr": rec.get("profile_corr"),
                }
            )
        else:
            accepted_local.append(int(local_idx))
            accepted_obs.append(int(rec["obs_id"]))
    payload = {
        "name": "pre_registered_row_contract_quality_mask",
        "purpose": "Secondary two-tier benchmark only. Full all-row metrics remain the primary headline.",
        "row_count_total": int(len(records)),
        "row_count_accepted": int(len(accepted_local)),
        "row_count_excluded": int(len(excluded)),
        "accepted_local_row_indices": accepted_local,
        "accepted_obs_ids": accepted_obs,
        "excluded_flag_keys": excluded_flags,
        "flag_counts": flag_counts,
        "criteria": {
            "max_center_sg_for_quality_inv_A": float(args.max_center_sg_for_quality),
            "max_centroid_residual_px": float(args.max_centroid_residual_px),
            "min_abs_profile_corr": float(args.min_abs_profile_corr),
            "min_background_pixels": int(args.min_background_pixels),
            "edge_definition": "calculated shoebox touches detector edge",
            "overlap_definition": f"predicted centers within {float(args.overlap_xy_px)} px and {float(args.overlap_z_frames)} frames",
        },
        "geometry_calibration": {
            "selected_model": calibration.get("selected_model"),
            "selected_summary": calibration.get("selected_summary"),
            "requested_model": calibration.get("requested_model"),
        },
        "excluded_examples": excluded[:100],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrate-hkl", required=True)
    ap.add_argument(
        "--xparm",
        default="",
        help="Optional XPARM.XDS/GXPARM.XDS geometry file. Defaults to XPARM.XDS next to INTEGRATE.HKL when present.",
    )
    ap.add_argument("--cbf-dir", default="")
    ap.add_argument(
        "--images", default="", help='Public CLI image glob, e.g. "/path/to/cbf/*.cbf".'
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Starting CIF/RES/INS model. Recorded as provenance.",
    )
    ap.add_argument("--out-dir", "--out", required=True)
    ap.add_argument("--frame-pattern", default="image_{frame:05d}.cbf")
    ap.add_argument(
        "--frame-radius",
        type=int,
        default=3,
        help="Half-width in frames for the 3D shoebox.",
    )
    ap.add_argument(
        "--box-frames", type=int, default=None, help="Public alias for --frame-radius."
    )
    ap.add_argument(
        "--xy-half-width",
        type=int,
        default=5,
        help="Half-width in detector pixels for the square shoebox.",
    )
    ap.add_argument(
        "--box-pixels", type=int, default=None, help="Public alias for --xy-half-width."
    )
    ap.add_argument(
        "--foreground-radius",
        type=float,
        default=3.0,
        help="Circular foreground radius in pixels inside the shoebox.",
    )
    ap.add_argument(
        "--aperture-radius",
        type=float,
        default=None,
        help="Deprecated alias for --foreground-radius.",
    )
    ap.add_argument("--bg-inner", type=float, default=4.0)
    ap.add_argument("--bg-outer", type=float, default=6.0)
    ap.add_argument("--background-shell-pixels", type=float, default=None)
    ap.add_argument(
        "--raw-integration-mode",
        choices=["sum", "gaussian_profile", "bloch_profile_fit"],
        default="sum",
        help="Raw target written to I_raw_sum. bloch_profile_fit fits raw pixels to background plane plus Gaussian detector PSF times Bloch rocking profile.",
    )
    ap.add_argument(
        "--background-model",
        choices=["global_median", "per_frame_median"],
        default="global_median",
        help="Background subtracted from foreground pixels. per_frame_median uses the annular median from each frame.",
    )
    ap.add_argument(
        "--profile-sigma-xy",
        type=float,
        default=0.0,
        help="Gaussian profile sigma in pixels. 0 derives from foreground radius.",
    )
    ap.add_argument(
        "--profile-sigma-z",
        type=float,
        default=0.0,
        help="Gaussian profile sigma in frames. 0 derives from frame radius.",
    )
    ap.add_argument("--profile-fit-clip-sigma", type=float, default=5.0)
    ap.add_argument("--profile-fit-nonnegative", action="store_true")
    ap.add_argument("--bloch-profile-n-basis", type=int, default=20)
    ap.add_argument("--bloch-profile-sg-window", type=float, default=0.004)
    ap.add_argument("--bloch-profile-topup-d-min", type=float, default=0.8)
    ap.add_argument("--bloch-profile-max-support-sg", type=float, default=0.005)
    ap.add_argument("--bloch-profile-thickness", type=float, default=1200.0)
    ap.add_argument(
        "--bloch-profile-thickness-model",
        choices=["single", "uniform_gradient", "gaussian", "gaussian_gradient"],
        default="uniform_gradient",
    )
    ap.add_argument("--bloch-profile-thickness-grad-frac", type=float, default=1.0)
    ap.add_argument("--bloch-profile-thickness-sigma-frac", type=float, default=0.0)
    ap.add_argument("--bloch-profile-thickness-points", type=int, default=5)
    ap.add_argument(
        "--bloch-profile-absorption-mode",
        choices=["none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"],
        default="midgley_like",
    )
    ap.add_argument("--bloch-profile-absorption-mu", type=float, default=0.00005)
    ap.add_argument("--bloch-profile-absorption-frac", type=float, default=0.02)
    ap.add_argument("--bloch-profile-absorption-mu-g2", type=float, default=0.0)
    ap.add_argument("--bloch-profile-absorption-mu-theta2", type=float, default=0.0)
    ap.add_argument("--bloch-profile-absorption-tds-b", type=float, default=0.0)
    ap.add_argument("--bloch-profile-ug-d-min", type=float, default=0.30)
    ap.add_argument(
        "--bloch-profile-hkl-transform",
        default="identity",
        help="Signed axis permutation for Bloch profile fitting only, e.g. '-h,k,l'. Raw output HKLs remain the original XDS row indices.",
    )
    ap.add_argument(
        "--bloch-profile-matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention for the Bloch rocking-profile raw fit.",
    )
    ap.add_argument("--bloch-profile-z-smooth-frames", type=float, default=0.0)
    ap.add_argument("--bloch-profile-z-gaussian-blend", type=float, default=0.0)
    ap.add_argument("--bloch-profile-z-gaussian-sigma", type=float, default=0.0)
    ap.add_argument(
        "--bloch-profile-cache-dir",
        default="",
        help="Optional U_g cache for Bloch profile fitting. Empty uses the default xds_bloch cache.",
    )
    ap.add_argument("--preserve-negative", default="true", choices=["true", "false"])
    ap.add_argument(
        "--calibrate-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist an explicit ZCAL frame calibration before writing dynrow/control HKLs.",
    )
    ap.add_argument(
        "--geometry-calibration-model",
        choices=["auto", "none", "global_offset", "linear_drift", "spline"],
        default="auto",
        help="Frame-correction model. auto chooses the simplest model meeting the median |Sg| threshold.",
    )
    ap.add_argument(
        "--target-modes",
        default="all",
        help="Compatibility option. 'all' writes raw, complete, XDS IOBS, and XDS recorded-proxy controls.",
    )
    ap.add_argument("--sigma-gain", type=float, default=0.0)
    ap.add_argument("--sigma-floor", type=float, default=1.0)
    ap.add_argument("--group-width-frames", type=float, default=3.0)
    ap.add_argument(
        "--observation-group-width-deg",
        type=float,
        default=5.0,
        help="row-contract observation-group angular width.",
    )
    ap.add_argument(
        "--observation-group-step-deg",
        type=float,
        default=3.0,
        help="row-contract observation-group angular step.",
    )
    ap.add_argument(
        "--max-center-sg-for-quality",
        type=float,
        default=0.002,
        help="Quality flag threshold for |Sg| at XDS ZCAL.",
    )
    ap.add_argument("--max-centroid-residual-px", type=float, default=2.0)
    ap.add_argument("--min-abs-profile-corr", type=float, default=0.0)
    ap.add_argument("--min-background-pixels", type=int, default=10)
    ap.add_argument("--overlap-xy-px", type=float, default=6.0)
    ap.add_argument("--overlap-z-frames", type=float, default=2.0)
    ap.add_argument(
        "--positive-only",
        action="store_true",
        help="Drop non-positive XDS IOBS rows before raw integration.",
    )
    ap.add_argument(
        "--top-isigma", type=int, default=0, help="Debug filter. 0 means all rows."
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug limit after all filters. 0 means all rows.",
    )
    ap.add_argument(
        "--even-sample",
        type=int,
        default=0,
        help="Evenly sample this many candidate rows across the sorted rotation range before --limit. 0 disables.",
    )
    ap.add_argument(
        "--write-xds-like", action=argparse.BooleanOptionalAction, default=True
    )
    args = ap.parse_args()
    if args.box_frames is not None:
        args.frame_radius = int(args.box_frames)
    if args.box_pixels is not None:
        args.xy_half_width = int(args.box_pixels)
        if args.foreground_radius == 3.0:
            args.foreground_radius = max(1.0, 0.6 * float(args.box_pixels))
    if args.background_shell_pixels is not None:
        args.bg_inner = float(args.xy_half_width)
        args.bg_outer = float(args.xy_half_width) + float(args.background_shell_pixels)
    if args.preserve_negative != "true":
        raise ValueError(
            "preserve-negative=false is blocked for native dynrows; preserve weak and negative rows."
        )
    if not args.calibrate_geometry:
        args.geometry_calibration_model = "none"
    if args.images:
        matches = sorted(glob.glob(args.images))
        if not matches:
            raise FileNotFoundError(f"--images matched no files: {args.images}")
        parents = {str(Path(m).parent.resolve()) for m in matches}
        if len(parents) != 1:
            raise ValueError(
                "--images must currently match files in exactly one directory"
            )
        args.cbf_dir = next(iter(parents))
        if args.frame_pattern == "image_{frame:05d}.cbf":
            first = Path(matches[0]).name
            m = re.search(r"(\d+)(?=\.cbf$)", first, flags=re.IGNORECASE)
            if m:
                args.frame_pattern = (
                    first[: m.start(1)]
                    + "{frame:0"
                    + str(len(m.group(1)))
                    + "d}"
                    + first[m.end(1) :]
                )
    if not args.cbf_dir:
        raise ValueError("Provide either --images '/path/to/cbf/*.cbf' or --cbf-dir.")
    if args.aperture_radius is not None:
        args.foreground_radius = float(args.aperture_radius)

    integrate_hkl = Path(args.integrate_hkl)
    cbf_dir = Path(args.cbf_dir)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = XDSData(integrate_hkl)
    model_meta = model_metadata(model_path)
    xparm_path = Path(args.xparm) if args.xparm else integrate_hkl.parent / "XPARM.XDS"
    xparm_meta = parse_xparm(xparm_path)
    detector_meta = header_detector_geometry(data)
    args.detector_nx_for_flags = detector_meta.get("nx")
    args.detector_ny_for_flags = detector_meta.get("ny")

    mask = np.ones(len(data.iobs), dtype=bool)
    if args.positive_only:
        mask &= data.iobs > 0.0
    row_indices = np.where(mask)[0]
    if args.top_isigma and len(row_indices) > args.top_isigma:
        isigi = data.iobs[row_indices] / np.maximum(data.sigma[row_indices], 1e-12)
        row_indices = row_indices[np.argsort(isigi)[::-1][: int(args.top_isigma)]]
    row_indices = np.asarray(
        sorted(row_indices, key=lambda i: (float(data.zcal[i]), int(i))), dtype=int
    )
    if args.even_sample and int(args.even_sample) > 0 and len(row_indices) > int(args.even_sample):
        sample_pos = np.linspace(0, len(row_indices) - 1, int(args.even_sample))
        row_indices = row_indices[np.unique(np.rint(sample_pos).astype(int))]
    if args.limit and args.limit > 0:
        row_indices = row_indices[: int(args.limit)]

    cache = FrameCache(
        cbf_dir, args.frame_pattern, max_items=2 * int(args.frame_radius) + 12
    )
    bloch_profile_model = None
    if args.raw_integration_mode == "bloch_profile_fit":
        bloch_profile_model = BlochRockingProfileModel(data, model_path, args)

    written_indices = []
    integrations = []
    rejected = []
    for n, idx in enumerate(row_indices, start=1):
        integration, failure = integrate_observation(
            cache, data, int(idx), args, bloch_profile_model=bloch_profile_model
        )
        if integration is None:
            rejected.append(
                {
                    "row_index": int(idx),
                    "hkl": [int(v) for v in data.hkl[idx]],
                    **failure,
                }
            )
            continue
        written_indices.append(int(idx))
        integrations.append(integration)
        if n % 500 == 0:
            print(f"integrated {n}/{len(row_indices)} candidate rows", flush=True)

    # Restore exact XDS row order in observation outputs.
    order = np.argsort(np.asarray(written_indices, dtype=int))
    written_indices = [written_indices[int(i)] for i in order]
    integrations = [integrations[int(i)] for i in order]

    frames_path = out_dir / "frames.jsonl"
    dynrows_path = out_dir / "dynrows.jsonl"
    obs_path = out_dir / "observations.jsonl"
    shoebox_path = out_dir / "shoebox_pixels.jsonl.gz"
    geometry_calibration = fit_geometry_calibration(
        data,
        written_indices,
        integrations,
        args.geometry_calibration_model,
        args.max_center_sg_for_quality,
    )
    calibration_by_idx = {
        int(k): float(v)
        for k, v in geometry_calibration.get("calibration_by_idx", {}).items()
    }
    calibration_public = dict(geometry_calibration)
    calibration_public["calibration_by_idx"] = {
        str(k): v for k, v in list(calibration_by_idx.items())[:200]
    }
    calibration_public["calibration_by_idx_truncated"] = len(calibration_by_idx) > 200
    geometry_calibration_path = out_dir / "geometry_calibration.json"
    geometry_calibration_path.write_text(
        json.dumps(calibration_public, indent=2, allow_nan=True) + "\n"
    )
    n_frames, observation_group_assignment = write_frames(
        frames_path,
        data,
        written_indices,
        args,
        integrate_hkl,
        model_meta,
        detector_meta,
        xparm_meta,
        calibration_by_idx,
    )
    dynrow_records = [
        observation_record(data, idx, integration, args, observation_group_assignment)
        for idx, integration in zip(written_indices, integrations, strict=False)
    ]
    mark_overlap_flags(dynrow_records, args.overlap_xy_px, args.overlap_z_frames)
    contract_validation = summarize_contract_validation(dynrow_records)
    if not contract_validation["valid"]:
        examples = contract_validation["invalid_examples"][:5]
        raise RuntimeError(
            "dynrow contract validation failed before writing outputs: "
            + json.dumps(examples, allow_nan=True)
        )
    quality_mask_path = out_dir / "quality_mask.json"
    quality_mask = write_quality_mask(
        dynrow_records, quality_mask_path, geometry_calibration, args
    )
    with gzip.open(shoebox_path, "wt") as f:
        for idx, rec, integration in zip(written_indices, dynrow_records, integrations, strict=False):
            f.write(
                json.dumps(
                    {
                        "obs_id": int(idx),
                        "raw_box_id": rec["raw_box_id"],
                        "raw_integration_version": integration[
                            "raw_integration_version"
                        ],
                        "support": rec["shoebox_support"],
                        "pixels": integration["pixels"],
                    },
                    allow_nan=True,
                )
                + "\n"
            )
    with dynrows_path.open("w") as f:
        for rec in dynrow_records:
            f.write(json.dumps(rec, allow_nan=True) + "\n")
    with obs_path.open("w") as f:
        for rec in dynrow_records:
            f.write(json.dumps(rec, allow_nan=True) + "\n")

    xds_like_recorded_path = None
    xds_like_complete_path = None
    xds_iobs_calibrated_path = None
    xds_recorded_proxy_calibrated_path = None
    if args.write_xds_like:
        xds_like_recorded_path = out_dir / "raw_recorded_dynrows.HKL"
        xds_like_complete_path = out_dir / "raw_complete_dynrows.HKL"
        xds_iobs_calibrated_path = out_dir / "xds_iobs_calibrated_dynrows.HKL"
        xds_recorded_proxy_calibrated_path = (
            out_dir / "xds_recorded_proxy_calibrated_dynrows.HKL"
        )
        write_xds_like(
            xds_like_recorded_path,
            data,
            written_indices,
            integrations,
            integrate_hkl,
            "recorded_raw",
            calibration_by_idx,
        )
        write_xds_like(
            xds_like_complete_path,
            data,
            written_indices,
            integrations,
            integrate_hkl,
            "complete_xdslike",
            calibration_by_idx,
        )
        write_xds_control(
            xds_iobs_calibrated_path,
            data,
            written_indices,
            integrate_hkl,
            "xds_iobs",
            calibration_by_idx,
        )
        write_xds_control(
            xds_recorded_proxy_calibrated_path,
            data,
            written_indices,
            integrate_hkl,
            "xds_recorded_proxy",
            calibration_by_idx,
        )

    raw_i = np.asarray([r["Iobs_raw"] for r in integrations], dtype=float)
    sig_i = np.asarray([r["sigma_Iobs_raw"] for r in integrations], dtype=float)
    complete_i = []
    complete_sig = []
    for idx, integ in zip(written_indices, integrations, strict=False):
        peak_fraction = max(float(data.peak[idx]) / 100.0, 1e-12)
        factor = float(data.rlp[idx]) / peak_fraction
        complete_i.append(float(integ["I_raw_sum"] * factor))
        complete_sig.append(float(abs(integ["sigma_raw_sum"] * factor)))
    complete_i = np.asarray(complete_i, dtype=float)
    complete_sig = np.asarray(complete_sig, dtype=float)
    replay = geometry_replay_summary(
        data, written_indices, detector_meta, calibration_by_idx
    )
    raw_boxes_npz_path = out_dir / "raw_boxes.npz"
    np.savez_compressed(
        raw_boxes_npz_path,
        obs_id=np.asarray(written_indices, dtype=np.int64),
        hkl=data.hkl[written_indices].astype(np.int32),
        xcal=data.xcal[written_indices].astype(np.float32),
        ycal=data.ycal[written_indices].astype(np.float32),
        zcal=data.zcal[written_indices].astype(np.float32),
        zcal_calibrated=np.asarray(
            [r["zcal_calibrated_frame"] for r in dynrow_records], dtype=np.float32
        ),
        frame_correction_applied=np.asarray(
            [r["frame_correction_applied"] for r in dynrow_records], dtype=np.float32
        ),
        I_raw_sum=np.asarray([r["I_raw_sum"] for r in integrations], dtype=np.float32),
        sigma_raw_sum=np.asarray(
            [r["sigma_raw_sum"] for r in integrations], dtype=np.float32
        ),
        I_raw_aperture_sum=np.asarray(
            [r.get("I_raw_aperture_sum", r["I_raw_sum"]) for r in integrations],
            dtype=np.float32,
        ),
        sigma_raw_aperture_sum=np.asarray(
            [r.get("sigma_raw_aperture_sum", r["sigma_raw_sum"]) for r in integrations],
            dtype=np.float32,
        ),
        I_raw_profile_fit=np.asarray(
            [r.get("I_raw_profile_fit", r["I_raw_sum"]) for r in integrations],
            dtype=np.float32,
        ),
        sigma_raw_profile_fit=np.asarray(
            [r.get("sigma_raw_profile_fit", r["sigma_raw_sum"]) for r in integrations],
            dtype=np.float32,
        ),
        I_raw_bloch_profile_fit=np.asarray(
            [r.get("I_raw_bloch_profile_fit", r.get("I_raw_profile_fit", r["I_raw_sum"])) for r in integrations],
            dtype=np.float32,
        ),
        sigma_raw_bloch_profile_fit=np.asarray(
            [r.get("sigma_raw_bloch_profile_fit", r.get("sigma_raw_profile_fit", r["sigma_raw_sum"])) for r in integrations],
            dtype=np.float32,
        ),
        bloch_profile_fit_status=np.asarray(
            [str(r.get("bloch_profile_fit_status", "not_requested")) for r in integrations],
            dtype="<U16",
        ),
        I_complete_xdslike=complete_i.astype(np.float32),
        sigma_complete_xdslike=complete_sig.astype(np.float32),
        background_mean=np.asarray(
            [r["background_mean"] for r in integrations], dtype=np.float32
        ),
        background_sigma=np.asarray(
            [r["background_sigma"] for r in integrations], dtype=np.float32
        ),
        n_foreground_pixels=np.asarray(
            [r["n_foreground_pixels"] for r in integrations], dtype=np.int32
        ),
        n_background_pixels=np.asarray(
            [r["n_background_pixels"] for r in integrations], dtype=np.int32
        ),
        n_masked_pixels=np.asarray(
            [r["n_masked_pixels"] for r in integrations], dtype=np.int32
        ),
        centroid_x=np.asarray(
            [r["centroid_x"] for r in integrations], dtype=np.float32
        ),
        centroid_y=np.asarray(
            [r["centroid_y"] for r in integrations], dtype=np.float32
        ),
        centroid_z=np.asarray(
            [r["centroid_z"] for r in integrations], dtype=np.float32
        ),
        observation_group_id=np.asarray(
            [r["observation_group_id"] for r in dynrow_records], dtype=np.int32
        ),
        target_sg_at_zcal_inv_A=np.asarray(
            [r["target_sg_at_zcal_inv_A"] for r in dynrow_records], dtype=np.float32
        ),
        target_sg_at_xds_zcal_inv_A=np.asarray(
            [r["target_sg_at_xds_zcal_inv_A"] for r in dynrow_records], dtype=np.float32
        ),
        target_zero_sg_delta_frame=np.asarray(
            [r["target_zero_sg_delta_frame"] for r in dynrow_records], dtype=np.float32
        ),
    )
    quality_counts = {}
    for key in [
        "large_geometry_residual_flag",
        "bad_centroid_flag",
        "poor_profile_corr_flag",
        "bad_background_flag",
        "edge_flag",
        "overlap_flag",
    ]:
        quality_counts[key] = int(
            sum(bool(rec.get("quality", {}).get(key, False)) for rec in dynrow_records)
        )
    manifest = {
        "contract": "row-contract signed dynrows from XDS predictions and raw CBF integration",
        "dynrow_schema_version": DYNROW_SCHEMA_VERSION,
        "engine_contract_version": FROZEN_ENGINE_CONTRACT_VERSION,
        "frozen_engine_contract": FROZEN_ENGINE_CONTRACT,
        "dynrow_contract": contract_metadata("build_dynrows_xds.py"),
        "contract_validation": contract_validation,
        "inputs": {
            "integrate_hkl": str(integrate_hkl.resolve()),
            "integrate_hkl_sha256": sha256(integrate_hkl),
            "xparm": str(xparm_path.resolve()) if xparm_meta else None,
            "xparm_sha256": xparm_meta.get("sha256") if xparm_meta else None,
            "cbf_dir": str(cbf_dir.resolve()),
            "frame_pattern": args.frame_pattern,
            "starting_model": model_meta,
        },
        "xds_geometry": {
            "wavelength_A": float(data.wavelength),
            "starting_frame": float(data.starting_frame),
            "starting_angle_deg": float(data.starting_angle),
            "oscillation_range_deg": float(data.oscillation_range),
            "rotation_axis": data.axis.astype(float).tolist(),
            "s0_vector": data.s0.astype(float).tolist(),
            "A_matrix": data.A.astype(float).tolist(),
            "unit_cell": [
                float(data.cell.a),
                float(data.cell.b),
                float(data.cell.c),
                float(data.cell.alpha),
                float(data.cell.beta),
                float(data.cell.gamma),
            ],
            "detector": detector_meta,
            "xparm": xparm_meta,
        },
        "geometry_replay": replay,
        "geometry_calibration": {
            "path": str(geometry_calibration_path.resolve()),
            "enabled": bool(args.calibrate_geometry),
            "requested_model": geometry_calibration.get("requested_model"),
            "selected_model": geometry_calibration.get("selected_model"),
            "selected_summary": geometry_calibration.get("selected_summary"),
            "candidate_summaries": geometry_calibration.get("candidate_summaries"),
        },
        "integration_model": {
            "type": (
                "circular_aperture_annular_background"
                if args.raw_integration_mode == "sum"
                else (
                    "bloch_rocking_profile_plane_background"
                    if args.raw_integration_mode == "bloch_profile_fit"
                    else "gaussian_profile_annular_background"
                )
            ),
            "frame_contract": "row-contract observation groups with exact signed XDS observations and raw-box/control intensity targets",
            "shoebox_type": "3d_square_box_with_circular_foreground_and_annular_background",
            "raw_integration_mode": args.raw_integration_mode,
            "background_model": args.background_model,
            "frame_radius": int(args.frame_radius),
            "xy_half_width_px": int(args.xy_half_width),
            "foreground_radius_px": float(args.foreground_radius),
            "bg_inner_px": float(args.bg_inner),
            "bg_outer_px": float(args.bg_outer),
            "profile_sigma_xy_px": (
                float(args.profile_sigma_xy)
                if float(args.profile_sigma_xy) > 0
                else max(float(args.foreground_radius) / 2.0, 0.75)
            ),
            "profile_sigma_z_frames": (
                float(args.profile_sigma_z)
                if float(args.profile_sigma_z) > 0
                else max(float(args.frame_radius) / 2.0, 0.75)
            ),
            "bloch_profile": {
                "enabled": args.raw_integration_mode == "bloch_profile_fit",
                "equation": "raw_pixel = b0 + bx*x + by*y + bz*z + A*Pxy(x,y)*Pbloch(frame) + error",
                "hkl_transform": args.bloch_profile_hkl_transform,
                "n_basis": int(args.bloch_profile_n_basis),
                "thickness_A": float(args.bloch_profile_thickness),
                "thickness_model": args.bloch_profile_thickness_model,
                "thickness_grad_frac": float(args.bloch_profile_thickness_grad_frac),
                "thickness_points": int(args.bloch_profile_thickness_points),
                "absorption_mode": args.bloch_profile_absorption_mode,
                "absorption_mu": float(args.bloch_profile_absorption_mu),
                "absorption_frac": float(args.bloch_profile_absorption_frac),
                "absorption_mu_g2": float(args.bloch_profile_absorption_mu_g2),
                "absorption_mu_theta2": float(args.bloch_profile_absorption_mu_theta2),
                "absorption_tds_b": float(args.bloch_profile_absorption_tds_b),
                "matrix_convention": args.bloch_profile_matrix_convention,
                "z_smooth_frames": float(args.bloch_profile_z_smooth_frames),
                "z_gaussian_blend": float(args.bloch_profile_z_gaussian_blend),
                "z_gaussian_sigma_frames": float(args.bloch_profile_z_gaussian_sigma),
                "profile_fit_clip_sigma": float(args.profile_fit_clip_sigma),
                "profile_fit_nonnegative": bool(args.profile_fit_nonnegative),
            },
            "group_width_frames": float(args.group_width_frames),
            "observation_group_width_deg": float(args.observation_group_width_deg),
            "observation_group_step_deg": float(args.observation_group_step_deg),
            "quality_thresholds": {
                "max_center_sg_for_quality_inv_A": float(
                    args.max_center_sg_for_quality
                ),
                "max_centroid_residual_px": float(args.max_centroid_residual_px),
                "min_abs_profile_corr": float(args.min_abs_profile_corr),
                "min_background_pixels": int(args.min_background_pixels),
                "overlap_xy_px": float(args.overlap_xy_px),
                "overlap_z_frames": float(args.overlap_z_frames),
            },
            "sigma_gain": float(args.sigma_gain),
            "sigma_floor": float(args.sigma_floor),
            "variance_note": "sigma is a conservative raw-count/background-MAD proxy plus configurable gain/floor terms, not a calibrated detector variance model",
        },
        "row_counts": {
            "xds_rows": int(len(data.iobs)),
            "candidate_rows": int(len(row_indices)),
            "written_observations": int(len(written_indices)),
            "rejected_observations": int(len(rejected)),
            "frames": int(n_frames),
        },
        "quality_counts": quality_counts,
        "raw_intensity_summary": {
            "target_modes": [
                "recorded_raw",
                "complete_xdslike",
                "xds_iobs_calibrated",
                "xds_recorded_proxy_calibrated",
            ],
            "recorded_raw_definition": "I_recorded_raw = I_raw_sum",
            "complete_xdslike_definition": "I_complete_xdslike = I_raw_sum * RLP / (PEAK/100)",
            "median": float(np.median(raw_i)) if len(raw_i) else None,
            "p05": float(np.percentile(raw_i, 5)) if len(raw_i) else None,
            "p95": float(np.percentile(raw_i, 95)) if len(raw_i) else None,
            "positive_fraction": float(np.mean(raw_i > 0.0)) if len(raw_i) else None,
            "median_sigma": float(np.median(sig_i)) if len(sig_i) else None,
            "complete_median": (
                float(np.median(complete_i)) if len(complete_i) else None
            ),
            "complete_p05": (
                float(np.percentile(complete_i, 5)) if len(complete_i) else None
            ),
            "complete_p95": (
                float(np.percentile(complete_i, 95)) if len(complete_i) else None
            ),
            "complete_positive_fraction": (
                float(np.mean(complete_i > 0.0)) if len(complete_i) else None
            ),
            "complete_median_sigma": (
                float(np.median(complete_sig)) if len(complete_sig) else None
            ),
        },
        "quality_mask": {
            "path": str(quality_mask_path.resolve()),
            "row_count_total": quality_mask["row_count_total"],
            "row_count_accepted": quality_mask["row_count_accepted"],
            "row_count_excluded": quality_mask["row_count_excluded"],
            "flag_counts": quality_mask["flag_counts"],
            "purpose": quality_mask["purpose"],
        },
        "outputs": {
            "frames_jsonl": str(frames_path.resolve()),
            "dynrows_jsonl": str(dynrows_path.resolve()),
            "observations_jsonl": str(obs_path.resolve()),
            "geometry_calibration_json": str(geometry_calibration_path.resolve()),
            "quality_mask_json": str(quality_mask_path.resolve()),
            "shoebox_pixels_jsonl_gz": str(shoebox_path.resolve()),
            "raw_boxes_npz": str(raw_boxes_npz_path.resolve()),
            "xds_like_hkl": (
                str(xds_like_recorded_path.resolve())
                if xds_like_recorded_path
                else None
            ),
            "xds_like_recorded_hkl": (
                str(xds_like_recorded_path.resolve())
                if xds_like_recorded_path
                else None
            ),
            "xds_like_complete_hkl": (
                str(xds_like_complete_path.resolve())
                if xds_like_complete_path
                else None
            ),
            "xds_iobs_calibrated_hkl": (
                str(xds_iobs_calibrated_path.resolve())
                if xds_iobs_calibrated_path
                else None
            ),
            "xds_recorded_proxy_calibrated_hkl": (
                str(xds_recorded_proxy_calibrated_path.resolve())
                if xds_recorded_proxy_calibrated_path
                else None
            ),
        },
        "rejected_examples": rejected[:50],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    (out_dir / "integration_audit.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True) + "\n"
    )
    print(
        json.dumps(
            {
                "frames": n_frames,
                "observations": len(written_indices),
                "rejected": len(rejected),
                "manifest": str(manifest_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
