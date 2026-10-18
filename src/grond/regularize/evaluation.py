"""Fixed-model row evaluation, q controls, and exact inverse-pair assembly. The physical rows and the Born reference stay matched; the resulting residuals are conditional on the supplied model."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict

import numpy as np

from grond.validation import heldout

from .contracts import Nuisance, ScaleModel
from .observations import group_indices, row_resolution_modulation, sigma_eff
from .scales import native_metrics


def pspline_trial_scales(data, target, base_sig, dyn, born, groups, valid, scale_fit, scale_model: ScaleModel, orbit_keys) -> dict:
    """P-spline frame scales fitted on the scale-fit rows only.

    As in tests/reference/reference_lib.py: scan angle omega(ZCAL), row_mr weights from the
    immutable source sigma, (n_basis, lambda) chosen by inner 3-fold orbit CV R_all
    restricted to the scale-fit rows (heldout.select_pspline_by_inner_cv), angles of
    predicted rows clamped to the fitted range. Born chooses its own setting by the same
    inner CV on the Born rows, so that Born scales are fitted independently: the
    matched Born control and the hand-invariant Born transform score must never depend
    on hand-sensitive Bloch predictions.
    """

    if getattr(data, "pets_exact_a_center", None) is not None:
        raise ValueError("P-spline frame scales need a continuous rotation-scan angle; PETS exact-row input has none.")
    if orbit_keys is None or len(orbit_keys) != len(target):
        raise ValueError("P-spline frame scales need one reflection-orbit key per row for their inner CV")
    rows = np.where(np.asarray(valid, dtype=bool))[0]
    fit = np.asarray(scale_fit, dtype=bool)[rows]
    if np.count_nonzero(fit) < 10:
        raise ValueError(f"P-spline frame scales need at least 10 scale-fit rows; found {int(np.count_nonzero(fit))}")
    ang = np.asarray([data.omega_from_z(float(z)) for z in np.asarray(data.zcal, dtype=float)[rows]], dtype=float)
    y = np.asarray(target, dtype=float)[rows]
    sig = np.asarray(base_sig, dtype=float)[rows]
    group = np.asarray(groups, dtype=int)[rows]
    keys = np.asarray(orbit_keys, dtype=object)[rows]
    grid = {"n_basis": tuple(int(n) for n in scale_model.n_basis_grid), "lambda": tuple(float(v) for v in scale_model.lambda_grid)}
    selection = heldout.select_pspline_by_inner_cv(
        y, sig, np.asarray(dyn, dtype=float)[rows], ang, group, fit, keys,
        grid=grid, clamp=bool(scale_model.clamp), objective=scale_model.objective,
    )
    if not math.isfinite(float(selection["inner_cv_r_all"])):
        raise ValueError("P-spline inner orbit CV had no usable fold; too few scale-fit rows")
    born_selection = heldout.select_pspline_by_inner_cv(
        y, sig, np.asarray(born, dtype=float)[rows], ang, group, fit, keys,
        grid=grid, clamp=bool(scale_model.clamp), objective=scale_model.objective,
    )
    if not math.isfinite(float(born_selection["inner_cv_r_all"])):
        raise ValueError("P-spline Born inner orbit CV had no usable fold; too few scale-fit rows")
    chosen = dict(selection["scale"])
    born_chosen = dict(born_selection["scale"])
    dyn_fit = heldout.pspline_fit_predict(y, sig, np.asarray(dyn, dtype=float)[rows], ang, group, fit, chosen, bool(scale_model.clamp))
    born_fit = heldout.pspline_fit_predict(y, sig, np.asarray(born, dtype=float)[rows], ang, group, fit, born_chosen, bool(scale_model.clamp))
    dyn_scaled = np.full(len(target), np.nan, dtype=float)
    born_scaled = np.full(len(target), np.nan, dtype=float)
    angle = np.full(len(target), np.nan, dtype=float)
    dyn_scaled[rows] = dyn_fit["prediction"]
    born_scaled[rows] = born_fit["prediction"]
    angle[rows] = ang
    info = {
        **scale_model.to_dict(),
        "n_basis": int(chosen["n_basis"]),
        "lambda": float(chosen["lambda"]),
        "selection": "inner_orbit_cv" if len(selection["grid"]) > 1 else "pinned",
        "inner_folds": 3,
        "inner_salt": heldout.INNER_CV_SALT,
        "inner_cv_r_all": float(selection["inner_cv_r_all"]),
        "grid": selection["grid"],
        "edf": float(dyn_fit["edf"]),
        "born_n_basis": int(born_chosen["n_basis"]),
        "born_lambda": float(born_chosen["lambda"]),
        "born_inner_cv_r_all": float(born_selection["inner_cv_r_all"]),
        "born_edf": float(born_fit["edf"]),
        "fit_rows": int(np.count_nonzero(fit)),
        "angle": "omega(ZCAL) in degrees",
        "fit_angle_range_deg": [float(np.min(ang[fit])), float(np.max(ang[fit]))],
    }
    return {"dyn_scaled": dyn_scaled, "born_scaled": born_scaled, "angle": angle, "info": info}


def evaluate_trial(data, target, base_sig, pred, indices, masks, nui: Nuisance, wght_a, wght_b, scale_fit_split: str = "work", *, scale_model: ScaleModel, orbit_keys):
    """Scale and score one nuisance trial's Bloch (dyn) and Born row predictions against the target rows.

    The q control and the row resolution factor act on dyn first. P-spline frame scales are fitted on the
    masks[scale_fit_split] rows only. Returns native_metrics of dyn and Born for every split in masks, the fitted
    scales, the parameter counts and the row arrays. When scales are fitted on work rows, the free-row metrics count
    no fitted parameter in their dof.
    """

    dyn = np.full(len(target), np.nan, dtype=float)
    dyn_inverse = np.full(len(target), np.nan, dtype=float)
    born = np.full(len(target), np.nan, dtype=float)
    basis_size = np.zeros(len(target), dtype=int)
    basis_support_hash64 = np.zeros(len(target), dtype=np.uint64)
    dyn[pred["indices"]] = pred["dyn"]
    if "dyn_inverse" in pred:
        dyn_inverse[pred["indices"]] = pred["dyn_inverse"]
    born[pred["indices"]] = pred["born"]
    dyn_unmodulated = dyn.copy()
    dyn_inverse_unmodulated = dyn_inverse.copy()
    if "basis_size" in pred:
        basis_size[pred["indices"]] = np.asarray(pred["basis_size"], dtype=int)
    if "basis_support_hash64" in pred:
        basis_support_hash64[pred["indices"]] = np.asarray(
            pred["basis_support_hash64"], dtype=np.uint64
        )
    pred_indices = np.asarray(pred["indices"], dtype=int)

    def expand_pred_array(name: str, *, dtype=float, default=np.nan):
        """Return a full-length row array holding the named prediction column at pred's rows and default elsewhere."""

        values = np.full(len(target), default, dtype=dtype)
        if name in pred:
            values[pred_indices] = np.asarray(pred[name], dtype=dtype)
        return values

    sg_arrays = {
        "center_sg": expand_pred_array("center_sg"),
        "omega_bragg": expand_pred_array("omega_bragg"),
        "omega_delta": expand_pred_array("omega_delta"),
        "omega_bragg_root_available": expand_pred_array(
            "omega_bragg_root_available", dtype=bool, default=False
        ),
        "sg_window": expand_pred_array("sg_window"),
        "sg_window_raw": expand_pred_array("sg_window_raw"),
        "sg_window_half_frames": expand_pred_array("sg_window_half_frames"),
        "sg_window_clipped": expand_pred_array("sg_window_clipped", dtype=bool, default=False),
        "sg_dSg_domega": expand_pred_array("sg_dSg_domega"),
        "sg_dSg_dframe": expand_pred_array("sg_dSg_dframe"),
        "sg_jitter_sigma": expand_pred_array("sg_jitter_sigma"),
        "sg_jitter_d_inv": expand_pred_array("sg_jitter_d_inv"),
        "sg_jitter_frame_fraction": expand_pred_array("sg_jitter_frame_fraction"),
        "sg_jitter_kernel_points": expand_pred_array("sg_jitter_kernel_points", dtype=int, default=1),
        "sg_profile_points_total": expand_pred_array("sg_profile_points_total", dtype=int, default=0),
        "sg_profile_weight_sum": expand_pred_array("sg_profile_weight_sum"),
        "sg_profile_same_branch_continuation_used": expand_pred_array(
            "sg_profile_same_branch_continuation_used",
            dtype=bool,
            default=False,
        ),
        "sg_profile_same_branch_continuation_segments": expand_pred_array(
            "sg_profile_same_branch_continuation_segments",
            dtype=int,
            default=1,
        ),
        "sg_residual": expand_pred_array("sg_residual"),
    }
    groups = group_indices(data, nui.group_width_deg, nui.frame_offset)
    # Every row is in detector tile 0: the row table keeps its detector_bin column.
    det_bins = np.zeros(len(data.hkl), dtype=int)
    sig = sigma_eff(data, target, base_sig, nui)
    use_wght_a = float(nui.native_wght_a) if np.isfinite(float(nui.native_wght_a)) else float(wght_a)
    use_wght_b = float(nui.native_wght_b) if np.isfinite(float(nui.native_wght_b)) else float(wght_b)
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn)
        & np.isfinite(born)
        & (dyn > 0.0)
        & (born > 0.0)
    )
    # The real row-wise Bloch/Born q (the retired shuffled-q controls permuted log(Idyn/Iborn) here).
    q_control_meta = {"mode": "real", "description": "real row-wise Bloch/Born q"}
    row_mod, row_mod_meta = row_resolution_modulation(data, valid, nui.row_resolution_b)
    dyn = dyn * row_mod
    dyn_inverse = dyn_inverse * row_mod
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn)
        & np.isfinite(born)
        & (dyn > 0.0)
        & (born > 0.0)
    )
    if scale_fit_split not in masks:
        raise ValueError(f"unknown scale_fit_split={scale_fit_split!r}; choices are {sorted(masks)}")
    scale_fit = masks[scale_fit_split] & valid
    spline = pspline_trial_scales(data, target, base_sig, dyn, born, groups, valid, scale_fit, scale_model, orbit_keys)
    # Integer consumers get ceil(EDF); the EDF itself is reported separately.
    n_scales = int(math.ceil(spline["info"]["edf"]))
    frame_scale_count = int(n_scales)
    n_abs = int(str(nui.absorption_mode) != "none" and (
        abs(float(nui.absorption_mu)) > 0.0
        or abs(float(nui.absorption_frac)) > 0.0
        or abs(float(nui.absorption_mu_g2)) > 0.0
        or abs(float(nui.absorption_mu_theta2)) > 0.0
    ))
    # gamma (mean, shape) has as many parameters as uniform_gradient (t0, grad): its shape takes the gradient's slot.
    n_thick_shape = int(str(nui.thickness_model) not in {"uniform_gradient", "gamma"} or abs(float(nui.thickness_sigma_frac)) > 0.0)
    n_dose = int(str(nui.dose_b_mode) != "none" and abs(float(nui.dose_b_per_frame)) > 0.0)
    n_jitter = int(str(nui.sg_jitter_mode) != "none" and (
        abs(float(nui.sg_jitter_sigma0)) > 0.0
        or abs(float(nui.sg_jitter_sigma_d_inv)) > 0.0
        or abs(float(nui.sg_jitter_sigma_frame)) > 0.0
    ))
    n_mosaicity = int(
        str(nui.mosaicity_mode) != "none"
        and abs(float(nui.mosaicity_sigma_deg)) > 0.0
    )
    n_resolution = int(abs(float(nui.row_resolution_b)) > 0.0)
    n_weight = int(np.isfinite(float(nui.native_wght_a)) or np.isfinite(float(nui.native_wght_b)))
    n_params = (
        n_scales
        + 7
        + n_abs
        + n_thick_shape
        + n_dose
        + n_jitter
        + n_mosaicity
        + n_resolution
        + n_weight
    )
    dyn_scaled = spline["dyn_scaled"]
    born_scaled = spline["born_scaled"]
    # P-spline Born scales carry their own EDF.
    born_n_params = n_params - frame_scale_count + int(math.ceil(spline["info"]["born_edf"]))
    metrics = {}
    for split, mask in masks.items():
        # Parameters fitted on work rows do not consume held-out residual dof.
        held_out = split == "free" and scale_fit_split == "work"
        split_n_params = 0 if held_out else n_params
        metrics[split] = {
            "dyn": native_metrics(target, sig, dyn_scaled, mask & valid, use_wght_a, use_wght_b, n_params=split_n_params),
            "born": native_metrics(target, sig, born_scaled, mask & valid, use_wght_a, use_wght_b, n_params=0 if held_out else born_n_params),
        }
    out = {
        "nuisance": asdict(nui),
        "native_weight": {"a": use_wght_a, "b": use_wght_b, "source_a": float(wght_a), "source_b": float(wght_b)},
        "metrics": metrics,
        "valid_rows": int(np.count_nonzero(valid)),
        "scale_count": int(n_scales),
        "frame_scale_count": int(frame_scale_count),
        # The detector-position scale field was removed; its record stays, always empty.
        "detector_scale_count": 0,
        "detector_scale_support": {
            "grid_bins_per_axis": 0,
            "nominal_parameter_count": 0,
            "adequately_populated_tiles": 0,
            "underpopulated_tiles": 0,
            "minimum_rows_per_tile": 0,
            "median_rows_per_tile": 0.0,
            "required_rows_per_tile": int(nui.detector_scale_min_rows),
            "all_tiles_adequate": True,
            "scale_min": 1.0,
            "scale_max": 1.0,
        },
        "scale_fit_split": scale_fit_split,
        "row_resolution": row_mod_meta,
        "q_control": q_control_meta,
        "basis_size_stats": {
            "min": int(np.min(basis_size[valid])) if np.count_nonzero(valid) else 0,
            "median": float(np.median(basis_size[valid])) if np.count_nonzero(valid) else 0.0,
            "p95": float(np.percentile(basis_size[valid], 95)) if np.count_nonzero(valid) else 0.0,
            "p99": float(np.percentile(basis_size[valid], 99)) if np.count_nonzero(valid) else 0.0,
            "max": int(np.max(basis_size[valid])) if np.count_nonzero(valid) else 0,
        },
        "basis_support_sha256": hashlib.sha256(
            np.asarray(basis_support_hash64[valid], dtype=">u8").tobytes(order="C")
        ).hexdigest(),
        # P-spline scales are per row (arrays dyn_row_scale), not per frame group or detector tile.
        "dyn_scales": {},
        "born_scales": {},
        "dyn_detector_scales": {},
        "born_detector_scales": {},
        "arrays": {
            "dyn_raw": dyn,
            "dyn_inverse_raw": dyn_inverse,
            "dyn_raw_unmodulated": dyn_unmodulated,
            "dyn_inverse_raw_unmodulated": dyn_inverse_unmodulated,
            "born_raw": born,
            "row_resolution_factor": row_mod,
            "dyn_scaled": dyn_scaled,
            "born_scaled": born_scaled,
            "sigma_eff": sig,
            "groups": groups,
            "detector_bins": det_bins,
            "basis_size": basis_size,
            "basis_support_hash64": basis_support_hash64,
            **sg_arrays,
            "valid": valid,
        },
    }
    out["scale_model"] = spline["info"]
    out["scale_edf"] = float(spline["info"]["edf"])
    out["arrays"]["dyn_row_scale"] = np.where(valid, dyn_scaled / np.where(dyn > 0.0, dyn, np.nan), np.nan)
    out["arrays"]["scale_angle_deg"] = spline["angle"]
    return out


def paired_inverse_prediction(best: dict) -> dict:
    """Recover unmodulated inverse predictions and their exact physical support."""

    inverse_raw = np.asarray(best["arrays"]["dyn_inverse_raw_unmodulated"], dtype=float)
    paired_indices = np.where(
        np.isfinite(inverse_raw)
        & np.isfinite(np.asarray(best["arrays"]["born_raw"], dtype=float))
    )[0]
    if len(paired_indices) == 0:
        raise RuntimeError(
            "paired inverse-hand calculation produced no finite inverse intensities"
        )
    inverse_pred = {
        "indices": paired_indices,
        "dyn": inverse_raw[paired_indices],
        "born": np.asarray(best["arrays"]["born_raw"], dtype=float)[paired_indices],
    }
    for key in (
        "basis_size",
        "basis_support_hash64",
        "center_sg",
        "omega_bragg",
        "omega_delta",
        "omega_bragg_root_available",
        "sg_window",
        "sg_window_raw",
        "sg_window_half_frames",
        "sg_window_clipped",
        "sg_dSg_domega",
        "sg_dSg_dframe",
        "sg_jitter_sigma",
        "sg_jitter_d_inv",
        "sg_jitter_frame_fraction",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "sg_profile_same_branch_continuation_used",
        "sg_profile_same_branch_continuation_segments",
        "sg_residual",
    ):
        values = best["arrays"].get(key)
        if values is not None:
            inverse_pred[key] = np.asarray(values)[paired_indices]
    return inverse_pred
