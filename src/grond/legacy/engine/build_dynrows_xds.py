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
import contextlib as contextlib
import glob
import gzip
import json
import math as math
import re
from pathlib import Path

import gemmi as gemmi
import numpy as np

if __package__:
    from .bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        XDSBlochPort,
        XDSData,
        parse_model,
        parse_res as parse_res,
        rotation_matrix as rotation_matrix,
        sha256,
    )
    from .dynrow_contract import (
        DYNROW_SCHEMA_VERSION,
        FROZEN_ENGINE_CONTRACT,
        FROZEN_ENGINE_CONTRACT_VERSION,
        contract_metadata,
        frame_contract_fields as frame_contract_fields,
        observation_contract_fields as observation_contract_fields,
        summarize_contract_validation,
    )
else:  # pragma: no cover - direct script execution
    from bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        XDSBlochPort,
        XDSData,
        parse_model,
        parse_res as parse_res,
        rotation_matrix as rotation_matrix,
        sha256,
    )
    from dynrow_contract import (
        DYNROW_SCHEMA_VERSION,
        FROZEN_ENGINE_CONTRACT,
        FROZEN_ENGINE_CONTRACT_VERSION,
        contract_metadata,
        frame_contract_fields as frame_contract_fields,
        observation_contract_fields as observation_contract_fields,
        summarize_contract_validation,
    )


from grond.io.raw_contracts import (
    build_observation_group_grid as build_observation_group_grid,
    choose_observation_group as choose_observation_group,
    mark_overlap_flags as mark_overlap_flags,
    model_metadata as model_metadata,
    observation_group_record as observation_group_record,
    observation_record as observation_record,
    write_frames as write_frames,
    write_quality_mask as write_quality_mask,
    write_xds_control as write_xds_control,
    write_xds_like as write_xds_like,
)
from grond.io.raw_detector import (
    frame_geometry_record as frame_geometry_record,
    geometry_replay_summary as geometry_replay_summary,
    gvec_for_hkl as gvec_for_hkl,
    header_detector_geometry as header_detector_geometry,
    header_float as header_float,
    header_vector as header_vector,
    parse_xparm as parse_xparm,
    replay_detector_xy as replay_detector_xy,
    xds_detector_origin_and_beam_center as xds_detector_origin_and_beam_center,
)
from grond.io.raw_geometry import (
    calibration_delta_for_idx as calibration_delta_for_idx,
    evaluate_frame_correction as evaluate_frame_correction,
    fit_geometry_calibration as fit_geometry_calibration,
    group_id_from_z as group_id_from_z,
    sg_for_hkl as sg_for_hkl,
    solve_zero_sg_angle as solve_zero_sg_angle,
    z_from_omega as z_from_omega,
)
from grond.io.raw_pixels import (
    FrameCache as FrameCache,
    default_frame_path as default_frame_path,
    fit_plane_profile_amplitude as fit_plane_profile_amplitude,
    integrate_observation as integrate_observation,
    profile_correlation as profile_correlation,
    profile_model_correlation as profile_model_correlation,
    require_fabio as require_fabio,
    robust_location_sigma as robust_location_sigma,
)


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
            intensity_t = [self.port.propagate_direct_to_target(H, float(t)) for t in self.port.thick]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrate-hkl", required=True)
    ap.add_argument(
        "--xparm",
        default="",
        help="Optional XPARM.XDS/GXPARM.XDS geometry file. Defaults to XPARM.XDS next to INTEGRATE.HKL when present.",
    )
    ap.add_argument("--cbf-dir", default="")
    ap.add_argument("--images", default="", help='Public CLI image glob, e.g. "/path/to/cbf/*.cbf".')
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
    ap.add_argument("--box-frames", type=int, default=None, help="Public alias for --frame-radius.")
    ap.add_argument(
        "--xy-half-width",
        type=int,
        default=5,
        help="Half-width in detector pixels for the square shoebox.",
    )
    ap.add_argument("--box-pixels", type=int, default=None, help="Public alias for --xy-half-width.")
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
    ap.add_argument(
        "--bloch-profile-max-support-sg",
        type=float,
        default=DEFAULT_MAX_SUPPORT_SG,
    )
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
    ap.add_argument("--top-isigma", type=int, default=0, help="Debug filter. 0 means all rows.")
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
    ap.add_argument("--write-xds-like", action=argparse.BooleanOptionalAction, default=True)
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
        raise ValueError("preserve-negative=false is blocked for native dynrows; preserve weak and negative rows.")
    if not args.calibrate_geometry:
        args.geometry_calibration_model = "none"
    if args.images:
        matches = sorted(glob.glob(args.images))
        if not matches:
            raise FileNotFoundError(f"--images matched no files: {args.images}")
        parents = {str(Path(m).parent.resolve()) for m in matches}
        if len(parents) != 1:
            raise ValueError("--images must currently match files in exactly one directory")
        args.cbf_dir = next(iter(parents))
        if args.frame_pattern == "image_{frame:05d}.cbf":
            first = Path(matches[0]).name
            m = re.search(r"(\d+)(?=\.cbf$)", first, flags=re.IGNORECASE)
            if m:
                args.frame_pattern = first[: m.start(1)] + "{frame:0" + str(len(m.group(1))) + "d}" + first[m.end(1) :]
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
    row_indices = np.asarray(sorted(row_indices, key=lambda i: (float(data.zcal[i]), int(i))), dtype=int)
    if args.even_sample and int(args.even_sample) > 0 and len(row_indices) > int(args.even_sample):
        sample_pos = np.linspace(0, len(row_indices) - 1, int(args.even_sample))
        row_indices = row_indices[np.unique(np.rint(sample_pos).astype(int))]
    if args.limit and args.limit > 0:
        row_indices = row_indices[: int(args.limit)]

    cache = FrameCache(cbf_dir, args.frame_pattern, max_items=2 * int(args.frame_radius) + 12)
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
    calibration_by_idx = {int(k): float(v) for k, v in geometry_calibration.get("calibration_by_idx", {}).items()}
    calibration_public = dict(geometry_calibration)
    calibration_public["calibration_by_idx"] = {str(k): v for k, v in list(calibration_by_idx.items())[:200]}
    calibration_public["calibration_by_idx_truncated"] = len(calibration_by_idx) > 200
    geometry_calibration_path = out_dir / "geometry_calibration.json"
    geometry_calibration_path.write_text(json.dumps(calibration_public, indent=2, allow_nan=True) + "\n")
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
            "dynrow contract validation failed before writing outputs: " + json.dumps(examples, allow_nan=True)
        )
    quality_mask_path = out_dir / "quality_mask.json"
    quality_mask = write_quality_mask(dynrow_records, quality_mask_path, geometry_calibration, args)
    with gzip.open(shoebox_path, "wt") as f:
        for idx, rec, integration in zip(written_indices, dynrow_records, integrations, strict=False):
            f.write(
                json.dumps(
                    {
                        "obs_id": int(idx),
                        "raw_box_id": rec["raw_box_id"],
                        "raw_integration_version": integration["raw_integration_version"],
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
        xds_recorded_proxy_calibrated_path = out_dir / "xds_recorded_proxy_calibrated_dynrows.HKL"
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
    replay = geometry_replay_summary(data, written_indices, detector_meta, calibration_by_idx)
    raw_boxes_npz_path = out_dir / "raw_boxes.npz"
    np.savez_compressed(
        raw_boxes_npz_path,
        obs_id=np.asarray(written_indices, dtype=np.int64),
        hkl=data.hkl[written_indices].astype(np.int32),
        xcal=data.xcal[written_indices].astype(np.float32),
        ycal=data.ycal[written_indices].astype(np.float32),
        zcal=data.zcal[written_indices].astype(np.float32),
        zcal_calibrated=np.asarray([r["zcal_calibrated_frame"] for r in dynrow_records], dtype=np.float32),
        frame_correction_applied=np.asarray([r["frame_correction_applied"] for r in dynrow_records], dtype=np.float32),
        I_raw_sum=np.asarray([r["I_raw_sum"] for r in integrations], dtype=np.float32),
        sigma_raw_sum=np.asarray([r["sigma_raw_sum"] for r in integrations], dtype=np.float32),
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
            [
                r.get("sigma_raw_bloch_profile_fit", r.get("sigma_raw_profile_fit", r["sigma_raw_sum"]))
                for r in integrations
            ],
            dtype=np.float32,
        ),
        bloch_profile_fit_status=np.asarray(
            [str(r.get("bloch_profile_fit_status", "not_requested")) for r in integrations],
            dtype="<U16",
        ),
        I_complete_xdslike=complete_i.astype(np.float32),
        sigma_complete_xdslike=complete_sig.astype(np.float32),
        background_mean=np.asarray([r["background_mean"] for r in integrations], dtype=np.float32),
        background_sigma=np.asarray([r["background_sigma"] for r in integrations], dtype=np.float32),
        n_foreground_pixels=np.asarray([r["n_foreground_pixels"] for r in integrations], dtype=np.int32),
        n_background_pixels=np.asarray([r["n_background_pixels"] for r in integrations], dtype=np.int32),
        n_masked_pixels=np.asarray([r["n_masked_pixels"] for r in integrations], dtype=np.int32),
        centroid_x=np.asarray([r["centroid_x"] for r in integrations], dtype=np.float32),
        centroid_y=np.asarray([r["centroid_y"] for r in integrations], dtype=np.float32),
        centroid_z=np.asarray([r["centroid_z"] for r in integrations], dtype=np.float32),
        observation_group_id=np.asarray([r["observation_group_id"] for r in dynrow_records], dtype=np.int32),
        target_sg_at_zcal_inv_A=np.asarray([r["target_sg_at_zcal_inv_A"] for r in dynrow_records], dtype=np.float32),
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
        quality_counts[key] = int(sum(bool(rec.get("quality", {}).get(key, False)) for rec in dynrow_records))
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
                "max_center_sg_for_quality_inv_A": float(args.max_center_sg_for_quality),
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
            "complete_median": (float(np.median(complete_i)) if len(complete_i) else None),
            "complete_p05": (float(np.percentile(complete_i, 5)) if len(complete_i) else None),
            "complete_p95": (float(np.percentile(complete_i, 95)) if len(complete_i) else None),
            "complete_positive_fraction": (float(np.mean(complete_i > 0.0)) if len(complete_i) else None),
            "complete_median_sigma": (float(np.median(complete_sig)) if len(complete_sig) else None),
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
            "xds_like_hkl": (str(xds_like_recorded_path.resolve()) if xds_like_recorded_path else None),
            "xds_like_recorded_hkl": (str(xds_like_recorded_path.resolve()) if xds_like_recorded_path else None),
            "xds_like_complete_hkl": (str(xds_like_complete_path.resolve()) if xds_like_complete_path else None),
            "xds_iobs_calibrated_hkl": (str(xds_iobs_calibrated_path.resolve()) if xds_iobs_calibrated_path else None),
            "xds_recorded_proxy_calibrated_hkl": (
                str(xds_recorded_proxy_calibrated_path.resolve()) if xds_recorded_proxy_calibrated_path else None
            ),
        },
        "rejected_examples": rejected[:50],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    (out_dir / "integration_audit.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
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
