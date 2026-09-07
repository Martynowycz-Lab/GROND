"""Fixed-model row evaluation, q controls, and exact inverse-pair assembly. The physical rows and the Born reference stay matched; the resulting residuals are conditional on the supplied model."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np

from grond.legacy.engine.bloch_engine import XDSData
from grond.legacy.engine.detector_scale_regularization import detector_field_diagnostics

from .contracts import Nuisance
from .observations import detector_bin_indices, group_indices, row_resolution_modulation, sigma_eff
from .scales import fit_frame_detector_scales, fit_group_scales, native_metrics


def resolution_shell_bins(data: XDSData, valid, shell_count: int) -> np.ndarray:
    """Assign rows to approximately equal-population resolution bins."""

    bins = np.full(len(data.hkl), -1, dtype=int)
    valid = np.asarray(valid, dtype=bool)
    n_shells = max(1, int(shell_count))
    if not np.any(valid):
        return bins
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    inv_d = np.full(len(dvals), np.nan, dtype=float)
    finite = valid & np.isfinite(dvals) & (dvals > 0.0)
    inv_d[finite] = 1.0 / dvals[finite]
    values = inv_d[finite]
    if values.size == 0:
        return bins
    if values.size < n_shells:
        n_shells = max(1, values.size)
    quantiles = np.linspace(0.0, 1.0, n_shells + 1)
    edges = np.quantile(values, quantiles)
    edges = np.maximum.accumulate(edges)
    for i, val in zip(np.where(finite)[0], inv_d[finite], strict=True):
        shell = int(np.searchsorted(edges, val, side="right") - 1)
        bins[i] = min(max(shell, 0), n_shells - 1)
    return bins


def apply_q_control(data: XDSData, dyn: np.ndarray, born: np.ndarray, valid, groups: np.ndarray, q_control_args=None) -> tuple[np.ndarray, dict]:
    """Apply an explicit row-space q null control before ordinary scale fitting."""

    mode = str(getattr(q_control_args, "q_control", "real") if q_control_args is not None else "real")
    meta = {
        "mode": mode,
        "description": "real row-wise Bloch/Born q" if mode == "real" else "row-space permutation of log(Idyn/Iborn)",
    }
    if mode == "real":
        return dyn, meta
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dyn) & np.isfinite(born) & (dyn > 0.0) & (born > 0.0)
    out = np.array(dyn, dtype=float, copy=True)
    q = np.full(len(dyn), np.nan, dtype=float)
    q[valid] = dyn[valid] / np.maximum(born[valid], 1e-30)
    logq = np.full(len(dyn), np.nan, dtype=float)
    logq[valid] = np.log(np.clip(q[valid], 1e-30, np.inf))
    seed = int(getattr(q_control_args, "q_control_seed", 12345) if q_control_args is not None else 12345)
    rng = np.random.default_rng(seed)
    shell_count = int(getattr(q_control_args, "q_control_shells", 10) if q_control_args is not None else 10)
    shell_bins = resolution_shell_bins(data, valid, shell_count)
    if mode == "shuffle_global":
        keys = np.zeros(len(dyn), dtype=int)
    elif mode == "shuffle_shell":
        keys = shell_bins
    elif mode == "shuffle_shell_scan":
        scan_groups = np.asarray(groups, dtype=int)
        keys = shell_bins.astype(np.int64) * 1000003 + scan_groups.astype(np.int64)
    else:
        raise ValueError(f"unknown q_control {mode!r}")
    shuffled_logq = np.array(logq, dtype=float, copy=True)
    shuffled_rows = 0
    singleton_bins = 0
    active_bins = 0
    for key in np.unique(keys[valid]):
        idx = np.where(valid & (keys == key))[0]
        if idx.size < 2:
            singleton_bins += int(idx.size == 1)
            continue
        active_bins += 1
        shuffled_logq[idx] = rng.permutation(shuffled_logq[idx])
        shuffled_rows += int(idx.size)
    out[valid] = born[valid] * np.exp(shuffled_logq[valid])
    meta.update(
        {
            "seed": seed,
            "shell_count": shell_count,
            "valid_rows": int(np.count_nonzero(valid)),
            "shuffled_rows": int(shuffled_rows),
            "active_bins": int(active_bins),
            "singleton_bins": int(singleton_bins),
            "logq_corr_original_shuffled": float(np.corrcoef(logq[valid], shuffled_logq[valid])[0, 1]) if np.count_nonzero(valid) > 2 else float("nan"),
        }
    )
    return out, meta


def evaluate_trial(data, target, base_sig, pred, indices, masks, nui: Nuisance, wght_a, wght_b, scale_fit_split: str = "work", q_control_args=None):
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
    det_bins = detector_bin_indices(data, nui.detector_scale_bins)
    sig = sigma_eff(data, target, base_sig, nui)
    use_wght_a = float(nui.native_wght_a) if np.isfinite(float(nui.native_wght_a)) else float(wght_a)
    use_wght_b = float(nui.native_wght_b) if np.isfinite(float(nui.native_wght_b)) else float(wght_b)
    q_control_meta = {}
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn)
        & np.isfinite(born)
        & (dyn > 0.0)
        & (born > 0.0)
    )
    dyn, q_control_meta = apply_q_control(data, dyn, born, valid, groups, q_control_args)
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
    n_scales = len([g for g in np.unique(groups) if np.count_nonzero((groups == g) & scale_fit) >= 2])
    frame_scale_count = int(n_scales)
    detector_scale_count = 0
    if int(nui.detector_scale_bins) > 1:
        detector_scale_count = int(nui.detector_scale_bins) * int(nui.detector_scale_bins)
        n_scales += detector_scale_count
    n_abs = int(str(nui.absorption_mode) != "none" and (
        abs(float(nui.absorption_mu)) > 0.0
        or abs(float(nui.absorption_frac)) > 0.0
        or abs(float(nui.absorption_mu_g2)) > 0.0
        or abs(float(nui.absorption_mu_theta2)) > 0.0
    ))
    n_thick_shape = int(str(nui.thickness_model) != "uniform_gradient" or abs(float(nui.thickness_sigma_frac)) > 0.0)
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
    if int(nui.detector_scale_bins) > 1:
        detector_fit_kwargs = {
            "scale_smoothness": nui.scale_smoothness,
            "detector_grid_size": int(nui.detector_scale_bins),
            "detector_ridge": float(nui.detector_scale_ridge),
            "detector_smoothness": float(nui.detector_scale_smoothness),
            "detector_min_rows": int(nui.detector_scale_min_rows),
        }
        dyn_scaled, dyn_scales, dyn_detector_scales = fit_frame_detector_scales(
            target, sig, dyn, groups, det_bins, scale_fit, use_wght_a, use_wght_b,
            **detector_fit_kwargs,
        )
        born_scaled, born_scales, born_detector_scales = fit_frame_detector_scales(
            target, sig, born, groups, det_bins, scale_fit, use_wght_a, use_wght_b,
            **detector_fit_kwargs,
        )
    else:
        dyn_scaled, dyn_scales = fit_group_scales(target, sig, dyn, groups, scale_fit, use_wght_a, use_wght_b, scale_smoothness=nui.scale_smoothness)
        born_scaled, born_scales = fit_group_scales(target, sig, born, groups, scale_fit, use_wght_a, use_wght_b, scale_smoothness=nui.scale_smoothness)
        dyn_detector_scales = {}
        born_detector_scales = {}
    detector_counts = (
        np.bincount(
            det_bins[scale_fit].astype(int),
            minlength=int(nui.detector_scale_bins) ** 2,
        )
        if int(nui.detector_scale_bins) > 1
        else np.asarray([], dtype=int)
    )
    detector_support = detector_field_diagnostics(
        np.asarray(
            [
                float(dyn_detector_scales.get(str(i), 1.0))
                for i in range(len(detector_counts))
            ],
            dtype=float,
        ),
        detector_counts,
        n_bins=int(nui.detector_scale_bins),
        min_rows=int(nui.detector_scale_min_rows),
    )
    metrics = {}
    for split, mask in masks.items():
        # Parameters fitted on work rows do not consume held-out residual dof.
        split_n_params = 0 if split == "free" and scale_fit_split == "work" else n_params
        metrics[split] = {
            "dyn": native_metrics(target, sig, dyn_scaled, mask & valid, use_wght_a, use_wght_b, n_params=split_n_params),
            "born": native_metrics(target, sig, born_scaled, mask & valid, use_wght_a, use_wght_b, n_params=split_n_params),
        }
    return {
        "nuisance": asdict(nui),
        "native_weight": {"a": use_wght_a, "b": use_wght_b, "source_a": float(wght_a), "source_b": float(wght_b)},
        "metrics": metrics,
        "valid_rows": int(np.count_nonzero(valid)),
        "scale_count": int(n_scales),
        "frame_scale_count": int(frame_scale_count),
        "detector_scale_count": int(detector_scale_count),
        "detector_scale_support": detector_support,
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
        "dyn_scales": {str(k): float(v) for k, v in dyn_scales.items()},
        "born_scales": {str(k): float(v) for k, v in born_scales.items()},
        "dyn_detector_scales": {str(k): float(v) for k, v in dyn_detector_scales.items()},
        "born_detector_scales": {str(k): float(v) for k, v in born_detector_scales.items()},
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
