# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""Serialize raw observation/group contracts, XDS compatibility files and quality masks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.model import parse_res
from grond.bloch.numerics import sha256
from grond.io.raw_geometry import (
    calibration_delta_for_idx,
    group_id_from_z,
    sg_for_hkl,
    solve_zero_sg_angle,
    z_from_omega,
)
from grond.io.xds import XDSData
from grond.legacy.engine.dynrow_contract import frame_contract_fields, observation_contract_fields


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
            data.omega_from_z(data.zcal[idx] + calibration_delta_for_idx(calibration_by_idx, int(idx)))
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
        "UB": (data.setting_rotation @ data.fixed_rotation @ data.A_for_omega(center_angle)).astype(float).tolist(),
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
        i + 1: observation_group_record(
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
        z_calibrated = float(data.zcal[idx]) + calibration_delta_for_idx(calibration_by_idx, int(idx))
        omega = float(data.omega_from_z(z_calibrated))
        fid, center = choose_observation_group(omega, centers)
        assignment[int(idx)] = {
            "observation_group_id": int(fid),
            "group_angle_center_deg": center,
            "zcal_calibrated_frame": z_calibrated,
            "frame_correction_applied": calibration_delta_for_idx(calibration_by_idx, int(idx)),
        }
        if fid in frames:
            rec = frames[fid]
            rec.setdefault("supporting_observation_ids", []).append(int(idx))
            rec["n_observations"] = int(len(rec["supporting_observation_ids"]))
            rec["zcal_min"] = min(float(rec.get("zcal_min", data.zcal[idx])), float(data.zcal[idx]))
            rec["zcal_max"] = max(float(rec.get("zcal_max", data.zcal[idx])), float(data.zcal[idx]))
            rec["zcal_calibrated_min"] = min(float(rec.get("zcal_calibrated_min", z_calibrated)), z_calibrated)
            rec["zcal_calibrated_max"] = max(float(rec.get("zcal_calibrated_max", z_calibrated)), z_calibrated)
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


def observation_record(data: XDSData, idx: int, integration: dict, args, observation_group_assignment: dict):
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
    frame_correction = float(assignment.get("frame_correction_applied", z_calibrated - z))
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
            "recorded_intensity_proxy_Iobs_peak_over_rlp": float(data.recorded_proxy()[idx]),
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
        "sigma_raw_aperture_sum": float(integration.get("sigma_raw_aperture_sum", sig_recorded)),
        "I_raw_profile_fit": float(integration.get("I_raw_profile_fit", i_recorded)),
        "sigma_raw_profile_fit": float(integration.get("sigma_raw_profile_fit", sig_recorded)),
        "profile_fit_amplitude": float(integration.get("profile_fit_amplitude", float("nan"))),
        "profile_fit_residual_sigma": float(integration.get("profile_fit_residual_sigma", float("nan"))),
        "profile_fit_weight_sum": float(integration.get("profile_fit_weight_sum", float("nan"))),
        "profile_fit_weight2_sum": float(integration.get("profile_fit_weight2_sum", float("nan"))),
        "I_raw_bloch_profile_fit": float(
            integration.get("I_raw_bloch_profile_fit", integration.get("I_raw_profile_fit", i_recorded))
        ),
        "sigma_raw_bloch_profile_fit": float(
            integration.get(
                "sigma_raw_bloch_profile_fit",
                integration.get("sigma_raw_profile_fit", sig_recorded),
            )
        ),
        "bloch_profile_fit_amplitude": float(integration.get("bloch_profile_fit_amplitude", float("nan"))),
        "bloch_profile_fit_residual_sigma": float(integration.get("bloch_profile_fit_residual_sigma", float("nan"))),
        "bloch_profile_fit_n_pixels": int(integration.get("bloch_profile_fit_n_pixels", 0)),
        "bloch_profile_fit_n_rejected": int(integration.get("bloch_profile_fit_n_rejected", 0)),
        "bloch_profile_fit_weight2_sum": float(integration.get("bloch_profile_fit_weight2_sum", float("nan"))),
        "bloch_profile_model_corr": (
            float(integration["bloch_profile_model_corr"])
            if math.isfinite(float(integration.get("bloch_profile_model_corr", float("nan"))))
            else None
        ),
        "bloch_profile_fit_status": integration.get("bloch_profile_fit_status", "not_requested"),
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
            integration.get("background_sigma_effective", integration["background_sigma"])
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
            float(integration["profile_corr"]) if math.isfinite(float(integration["profile_corr"])) else None
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
            "large_geometry_residual_flag": bool(abs(sg_at_zcal) > float(args.max_center_sg_for_quality)),
            "bad_centroid_flag": bool(
                float(integration["centroid_residual_px"]) > float(args.max_centroid_residual_px)
            ),
            "poor_profile_corr_flag": bool(
                integration["profile_corr"] is not None
                and math.isfinite(float(integration["profile_corr"]))
                and abs(float(integration["profile_corr"])) < float(args.min_abs_profile_corr)
            ),
            "bad_background_flag": bool(
                int(integration["n_background_pixels"]) < int(args.min_background_pixels)
                or float(integration["background_sigma"]) <= 0.0
            ),
            "edge_flag": bool(
                data.xcal[idx] - args.xy_half_width < 1
                or data.ycal[idx] - args.xy_half_width < 1
                or (data.xcal[idx] + args.xy_half_width > (args.detector_nx_for_flags or 1e12))
                or (data.ycal[idx] + args.xy_half_width > (args.detector_ny_for_flags or 1e12))
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
            "profile_sigma_xy_px": float(integration.get("profile_sigma_xy_px", float("nan"))),
            "profile_sigma_z_frames": float(integration.get("profile_sigma_z_frames", float("nan"))),
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
        f.write("!RAW_DYNROWS_SOURCE= build_dynrows_xds.py raw CBF 3D shoebox integration\n")
        f.write(f"!RAW_DYNROWS_TARGET_MODE= {target_mode}\n")
        f.write(
            "!RAW_DYNROWS_GEOMETRY= XDS predicted XCAL YCAL ZCAL and XOBS YOBS ZOBS are preserved; raw centroids are stored in dynrows.jsonl\n"
        )
        if target_mode == "recorded_raw":
            f.write("!RAW_DYNROWS_TARGET_DEFINITION= IOBS=I_raw_sum SIGMA=sigma_raw_sum\n")
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
                data.zcal[idx] + calibration_delta_for_idx(calibration_by_idx, int(idx)),
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
                " ".join(str(int(v)) if n in {0, 1, 2, 20} else f"{float(v):.6g}" for n, v in enumerate(row)) + "\n"
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
        f.write("!RAW_DYNROWS_SOURCE= build_dynrows_xds.py control target with calibrated ZCAL geometry\n")
        f.write(f"!RAW_DYNROWS_TARGET_MODE= {target_mode}\n")
        f.write(
            "!RAW_DYNROWS_GEOMETRY= ZCAL is corrected by geometry_calibration.json; detector coordinates and signed HKL are unchanged\n"
        )
        f.write("!END_OF_HEADER\n")
        recorded = data.recorded_proxy()
        recorded_sigma = data.sigma * np.maximum(data.peak, 0.0) / 100.0 / np.maximum(data.rlp, 1e-12)
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
                data.zcal[idx] + calibration_delta_for_idx(calibration_by_idx, int(idx)),
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
                " ".join(str(int(v)) if n in {0, 1, 2, 20} else f"{float(v):.6g}" for n, v in enumerate(row)) + "\n"
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
                if abs(float(rec["zcal_frame"]) - float(other["zcal_frame"])) > z_frames:
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
