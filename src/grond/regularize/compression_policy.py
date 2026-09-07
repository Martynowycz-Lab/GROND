"""Existing compact-scale and detector-field acceptance diagnostics. A flexible detector field can absorb model error; the held-out, support, and parity checks are deliberate safeguards."""

from __future__ import annotations

import hashlib

import numpy as np

from grond.legacy.engine.detector_scale_regularization import (
    detector_field_diagnostics,
)

from .compression_topology import coarsen_detector
from .compression_weights import (
    fit_group_detector_scales,
    fit_group_scales,
    row_mr_metrics,
    safe_float,
)


def _stable_detector_split_masks(
    rows: list[dict[str, str]],
    valid: np.ndarray,
    *,
    seeds: tuple[int, ...] = (1729, 2718, 3141),
    holdout_fraction: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic row-identity splits independent of fitted residuals."""

    threshold = int(round(float(holdout_fraction) * 10_000))
    out = []
    for seed in seeds:
        free = np.zeros(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            if not bool(valid[i]):
                continue
            identity = "|".join(
                str(row.get(key, ""))
                for key in ("row_index", "h", "k", "l", "group", "detector_bin")
            )
            digest = hashlib.blake2b(
                f"{seed}|{identity}".encode(), digest_size=8
            ).digest()
            free[i] = int.from_bytes(digest, "big") % 10_000 < threshold
        work = np.asarray(valid, dtype=bool) & ~free
        free &= np.asarray(valid, dtype=bool)
        if np.count_nonzero(work) >= 2 and np.count_nonzero(free) >= 2:
            out.append((work, free))
    return out


def _field_agreement(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(left) & set(right), key=int)
    if not keys:
        return {"correlation": float("nan"), "log_rms": float("inf")}
    a = np.log(np.maximum([safe_float(left[key]) for key in keys], 1e-30))
    b = np.log(np.maximum([safe_float(right[key]) for key in keys], 1e-30))
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    if len(keys) > 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = 1.0 if rms <= 0.02 else 0.0
    return {"correlation": corr, "log_rms": rms}


def detector_scale_policy_diagnostics(
    *,
    rows: list[dict[str, str]],
    obs: np.ndarray,
    sig: np.ndarray,
    raw_dyn: np.ndarray,
    raw_born: np.ndarray,
    groups: np.ndarray,
    source_detector_bins: np.ndarray,
    detector_base: int,
    detector_target: int,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    detector_ridge: float = 0.05,
    detector_smoothness: float = 0.10,
    detector_min_rows: int = 20,
    min_holdout_improvement: float = 0.002,
    required_improved_splits: int = 2,
    max_bloch_gain_loss: float = 0.002,
    min_field_correlation: float = 0.50,
    max_field_log_rms: float = 0.10,
    independent_calibration: bool = False,
) -> dict[str, object]:
    """Decide whether a detector field earns promotion on frozen predictions.

    The no-detector fit is the reference.  Added detector parameters must help
    on several deterministic row holdouts, retain the Born-to-Bloch advantage,
    have adequate tile support, reproduce across data halves, and agree when
    inferred from the Born and Bloch calculations.  An 8x8 field remains a
    diagnostic unless independent calibration is explicitly declared.
    """

    detector_target = int(detector_target)
    if detector_target <= 1:
        return {
            "status": "reference_no_detector_field",
            "eligible": True,
            "detector_bins": 0,
            "reasons": [],
        }
    det_bins = coarsen_detector(source_detector_bins, detector_base, detector_target)
    valid = (
        np.isfinite(obs)
        & np.isfinite(sig)
        & (sig > 0)
        & np.isfinite(raw_dyn)
        & (raw_dyn > 0)
        & np.isfinite(raw_born)
        & (raw_born > 0)
        & np.isfinite(groups)
        & np.isfinite(det_bins)
    )
    fit_good = np.asarray(fit_mask, dtype=bool) & valid
    counts = np.bincount(
        det_bins[fit_good].astype(int), minlength=detector_target * detector_target
    )

    def fit(raw: np.ndarray, mask: np.ndarray, use_detector: bool):
        if not use_detector:
            scaled, frame = fit_group_scales(
                obs,
                sig,
                raw,
                groups,
                mask,
                objective,
                native_wght_a,
                native_wght_b,
                instability_u,
            )
            return scaled, frame, {}
        return fit_group_detector_scales(
            obs,
            sig,
            raw,
            groups,
            det_bins,
            mask,
            objective,
            native_wght_a,
            native_wght_b,
            instability_u,
            detector_grid_size=detector_target,
            detector_ridge=detector_ridge,
            detector_smoothness=detector_smoothness,
            detector_min_rows=detector_min_rows,
        )

    split_contracts: list[tuple[object, np.ndarray, np.ndarray]] = []
    predeclared_free = valid & ~fit_good
    if np.count_nonzero(fit_good) >= 2 and np.count_nonzero(predeclared_free) >= 2:
        split_contracts.append(("predeclared_free", fit_good, predeclared_free))
    split_contracts.extend(
        (index, work, free)
        for index, (work, free) in enumerate(
            _stable_detector_split_masks(rows, fit_good)
        )
    )
    split_rows = []
    for split_label, work, free in split_contracts:
        ref_dyn, _, _ = fit(raw_dyn, work, False)
        ref_born, _, _ = fit(raw_born, work, False)
        candidate_dyn, _, _ = fit(raw_dyn, work, True)
        candidate_born, _, _ = fit(raw_born, work, True)
        ref_dyn_metrics = row_mr_metrics(obs, sig, ref_dyn, free, instability_u, 3.0)
        ref_born_metrics = row_mr_metrics(obs, sig, ref_born, free, instability_u, 3.0)
        dyn_metrics = row_mr_metrics(obs, sig, candidate_dyn, free, instability_u, 3.0)
        born_metrics = row_mr_metrics(obs, sig, candidate_born, free, instability_u, 3.0)
        ref_dyn_rall = safe_float(ref_dyn_metrics.get("Rall"))
        ref_born_rall = safe_float(ref_born_metrics.get("Rall"))
        dyn_rall = safe_float(dyn_metrics.get("Rall"))
        born_rall = safe_float(born_metrics.get("Rall"))
        split_rows.append(
            {
                "split": split_label,
                "reference_dyn_Rall": ref_dyn_rall,
                "candidate_dyn_Rall": dyn_rall,
                "detector_improvement_Rall": ref_dyn_rall - dyn_rall,
                "reference_bloch_gain_Rall": ref_born_rall - ref_dyn_rall,
                "candidate_bloch_gain_Rall": born_rall - dyn_rall,
                "bloch_gain_change_Rall": (born_rall - dyn_rall)
                - (ref_born_rall - ref_dyn_rall),
                "reference_dyn_wRall": ref_dyn_metrics.get("wRall"),
                "candidate_dyn_wRall": dyn_metrics.get("wRall"),
                "reference_bloch_gain_wRall": safe_float(ref_born_metrics.get("wRall"))
                - safe_float(ref_dyn_metrics.get("wRall")),
                "candidate_bloch_gain_wRall": safe_float(born_metrics.get("wRall"))
                - safe_float(dyn_metrics.get("wRall")),
            }
        )

    _, _, dyn_field = fit(raw_dyn, fit_good, True)
    _, _, born_field = fit(raw_born, fit_good, True)
    shared_agreement = _field_agreement(dyn_field, born_field)
    half_splits = _stable_detector_split_masks(
        rows, fit_good, seeds=(5779,), holdout_fraction=0.50
    )
    if half_splits:
        half_a, half_b = half_splits[0]
        _, _, field_a = fit(raw_dyn, half_a, True)
        _, _, field_b = fit(raw_dyn, half_b, True)
        half_agreement = _field_agreement(field_a, field_b)
    else:
        half_agreement = {"correlation": float("nan"), "log_rms": float("inf")}

    support = detector_field_diagnostics(
        np.asarray([safe_float(dyn_field.get(str(i)), 1.0) for i in range(len(counts))]),
        counts,
        n_bins=detector_target,
        min_rows=detector_min_rows,
    )
    improved_count = sum(
        safe_float(row["detector_improvement_Rall"], -float("inf"))
        >= float(min_holdout_improvement)
        for row in split_rows
    )
    reasons: list[str] = []
    if len(split_rows) < int(required_improved_splits):
        reasons.append("insufficient_deterministic_holdout_splits")
    elif improved_count < int(required_improved_splits):
        reasons.append("detector_field_did_not_improve_multiple_holdouts")
    if any(
        safe_float(row["bloch_gain_change_Rall"], -float("inf"))
        < -float(max_bloch_gain_loss)
        for row in split_rows
    ):
        reasons.append("detector_field_reduced_heldout_born_to_bloch_advantage")
    if not bool(support["all_tiles_adequate"]):
        reasons.append("underpopulated_detector_tiles")
    for label, agreement in (
        ("data_half", half_agreement),
        ("born_bloch", shared_agreement),
    ):
        if safe_float(agreement["log_rms"], float("inf")) > float(max_field_log_rms):
            reasons.append(f"{label}_detector_field_log_rms_exceeded")
        if safe_float(agreement["correlation"], -float("inf")) < float(
            min_field_correlation
        ):
            reasons.append(f"{label}_detector_field_correlation_below_minimum")
    if detector_target >= 8 and not bool(independent_calibration):
        reasons.append("eight_by_eight_requires_independent_calibration")
    return {
        "status": "eligible" if not reasons else "diagnostic_only",
        "eligible": not reasons,
        "detector_bins": detector_target,
        "reasons": reasons,
        "holdout_splits": split_rows,
        "improved_split_count": int(improved_count),
        "required_improved_split_count": int(required_improved_splits),
        "selection_metric": "held-out JANA-style unmerged row Rall",
        "secondary_report_metric": "held-out JANA-style unmerged row wRall",
        "bloch_advantage_metric": "Born Rall - Bloch Rall",
        "support": support,
        "data_half_agreement": half_agreement,
        "born_bloch_field_agreement": shared_agreement,
        "regularization": {
            "ridge_to_unity": float(detector_ridge),
            "spatial_smoothness": float(detector_smoothness),
            "minimum_rows_per_tile": int(detector_min_rows),
        },
        "independent_calibration": bool(independent_calibration),
    }


def metric_delta(candidate: dict, baseline: dict) -> dict:
    out = {}
    for key in ("Robs", "Rall", "wRall"):
        out[key] = safe_float(candidate.get(key)) - safe_float(baseline.get(key))
    for split in ("work", "free"):
        candidate_split = candidate.get(f"row_{split}") or {}
        baseline_split = baseline.get(f"row_{split}") or {}
        for key in ("Rall", "wRall"):
            out[f"{split}_{key}"] = safe_float(candidate_split.get(key)) - safe_float(
                baseline_split.get(key)
            )
    out["native_R1_gt"] = safe_float(candidate.get("native", {}).get("R1_gt")) - safe_float(baseline.get("native", {}).get("R1_gt"))
    out["native_R1_all"] = safe_float(candidate.get("native", {}).get("R1_all")) - safe_float(baseline.get("native", {}).get("R1_all"))
    out["native_wR2_all"] = safe_float(candidate.get("native", {}).get("wR2_all")) - safe_float(baseline.get("native", {}).get("wR2_all"))
    return out


def within_tolerance(delta: dict, metrics: tuple[str, ...], tolerance: float) -> bool:
    return all(np.isfinite(safe_float(delta.get(m))) and safe_float(delta.get(m)) <= float(tolerance) for m in metrics)


def candidate_row(candidate: dict) -> dict[str, object]:
    delta = candidate["delta_vs_baseline"]
    native = candidate["native"]
    info = candidate.get("information", {})
    return {
        "name": candidate["name"],
        "frame_model": candidate.get("frame_model", ""),
        "mode": candidate["mode"],
        "objective": candidate["objective"],
        "group_stride": candidate["group_stride"],
        "group_offset": candidate.get("group_offset", 0),
        "adaptive_segments": candidate.get("adaptive_segments", ""),
        "window_deg": candidate.get("window_deg", ""),
        "detector_bins": candidate["detector_bins"],
        "scale_params": candidate["scale_params"],
        "frame_scale_params": candidate.get("group_scale_params", ""),
        "detector_scale_params": candidate.get("detector_scale_params", ""),
        "total_params": candidate["total_params"],
        "Nobs": candidate["Nobs"],
        "Nall": candidate["Nall"],
        "Robs": candidate["Robs"],
        "Rall": candidate["Rall"],
        "wRall": candidate["wRall"],
        "work_Rall": (candidate.get("row_work") or {}).get("Rall"),
        "work_wRall": (candidate.get("row_work") or {}).get("wRall"),
        "free_Rall": (candidate.get("row_free") or {}).get("Rall"),
        "free_wRall": (candidate.get("row_free") or {}).get("wRall"),
        "dRobs": delta["Robs"],
        "dRall": delta["Rall"],
        "dwRall": delta["wRall"],
        "dwork_Rall": delta.get("work_Rall"),
        "dwork_wRall": delta.get("work_wRall"),
        "dfree_Rall": delta.get("free_Rall"),
        "dfree_wRall": delta.get("free_wRall"),
        "native_R1_gt": native["R1_gt"],
        "native_R1_all": native["R1_all"],
        "native_wR2_all": native["wR2_all"],
        "native_GooF": native.get("GooF", ""),
        "native_chi2": native.get("chi2", ""),
        "aic_known_sigma": info.get("aic_known_sigma", ""),
        "bic_known_sigma": info.get("bic_known_sigma", ""),
        "aic_fitted_variance": info.get("aic_fitted_variance", ""),
        "bic_fitted_variance": info.get("bic_fitted_variance", ""),
        "native_dR1_gt": delta["native_R1_gt"],
        "native_dR1_all": delta["native_R1_all"],
        "native_dwR2_all": delta["native_wR2_all"],
        "fit_scale_log_step_rms": candidate.get("fit_scale_log_step_rms", ""),
        "baseline_within_group_log_rms": candidate.get("baseline_within_group_log_rms", ""),
        "within_strict": candidate["within_strict"],
        "within_r_only": candidate["within_r_only"],
        "detector_policy_eligible": bool(
            (candidate.get("detector_policy") or {}).get("eligible", False)
        ),
        "detector_policy_status": (candidate.get("detector_policy") or {}).get(
            "status", ""
        ),
        "detector_policy_reasons": ";".join(
            str(reason)
            for reason in (candidate.get("detector_policy") or {}).get("reasons", [])
        ),
    }
