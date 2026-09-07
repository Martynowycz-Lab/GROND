"""Explicit raw-pixel integration contracts and bounded image reading.

Aperture, Gaussian profile and Bloch-profile estimators retain their individual
background and variance conventions. They are diagnostic observation builders,
not a claim of equivalence to the originating XDS or DIALS integration."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from grond.io.raw_geometry import sg_for_hkl
from grond.io.xds import XDSData


def require_fabio():
    """Load the optional raw-image reader when this workflow is used."""
    try:
        import fabio
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Raw CBF extraction requires the optional 'fabio' dependency. Install it with: pip install 'grond[raw]'"
        ) from exc
    return fabio


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
    # This is a bounded two-fit diagnostic, not clipping iterated to convergence.
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
                for x0 in range(int(round(xcal)) - xy_half, int(round(xcal)) + xy_half + 1):
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
            float(np.sum(np.asarray(frame_fg_values, dtype=float) - bg_med_frame)) if frame_fg_values else 0.0
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
    n_fg_valid = sum(1 for p in pixels if p["foreground"] and p["raw_value"] is not None and p["mask_state"] == "valid")
    n_bg_valid = sum(1 for p in pixels if p["background"] and p["raw_value"] is not None and p["mask_state"] == "valid")
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
        bg_sigma_effective = float(np.nanmedian(frame_bg_sig_for_fg)) if len(frame_bg_sig_for_fg) else bg_sigma
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
    sigma_xy = float(args.profile_sigma_xy) if float(args.profile_sigma_xy) > 0 else max(float(fg_radius) / 2.0, 0.75)
    sigma_z = float(args.profile_sigma_z) if float(args.profile_sigma_z) > 0 else max(float(frame_radius) / 2.0, 0.75)
    prof_w = np.exp(-0.5 * ((dxs / sigma_xy) ** 2 + (dys / sigma_xy) ** 2 + (dzs / sigma_z) ** 2))
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
            pix_var * (sw * sw) / ww + (sw * sw) * var_bg_mean + gain_sigma * gain_sigma + sigma_floor * sigma_floor
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
            {int(p["frame"]) for p in pixels if p.get("raw_value") is not None and p.get("mask_state") == "valid"}
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
            fit = fit_plane_profile_amplitude(pixels, xcal, ycal, zcal, profile_z_by_frame, args, sigma_xy)
            bloch_profile_signal = float(fit["amplitude"])
            bloch_profile_sigma = float(
                math.sqrt(max(float(fit["sigma"]) ** 2 + gain_sigma * gain_sigma + sigma_floor * sigma_floor, 1.0))
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
    centroid_resid = math.sqrt((x_centroid - xcal) ** 2 + (y_centroid - ycal) ** 2 + (z_centroid - zcal) ** 2)
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
