#!/usr/bin/env python3
"""Compress final row-space scale factors after a GROND Bloch calculation.

This is a post-regularization polish.  It does not recompute Bloch waves and it
does not move the structure.  It reads a final ``native_dyn_calc.csv`` table,
keeps ``dyn_raw`` and ``born_raw`` fixed, and asks how simple the multiplicative
scale model can be while staying close to the current full-scale result.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

try:
    from .detector_scale_regularization import (
        detector_field_diagnostics,
        infer_square_grid_size,
        solve_regularized_detector_field,
    )
except ImportError:  # pragma: no cover - direct script execution
    from detector_scale_regularization import (
        detector_field_diagnostics,
        infer_square_grid_size,
        solve_regularized_detector_field,
    )


def safe_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def col(rows: list[dict[str, str]], name: str, default=float("nan")) -> np.ndarray:
    return np.asarray([safe_float(r.get(name, default), default) for r in rows], dtype=float)


def str_col(rows: list[dict[str, str]], name: str, default: str = "") -> np.ndarray:
    return np.asarray([str(r.get(name, default)) for r in rows])


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_csv_ints(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"global", "inf", "infinite"}:
            out.append(10**9)
        else:
            out.append(int(float(part)))
    return sorted(set(out), key=out.index)


def parse_csv_metrics(text: str) -> tuple[str, ...]:
    metrics = tuple(part.strip() for part in str(text).split(",") if part.strip())
    allowed = {
        "Robs",
        "Rall",
        "wRall",
        "work_Rall",
        "work_wRall",
        "free_Rall",
        "free_wRall",
        "native_R1_gt",
        "native_R1_all",
        "native_wR2_all",
    }
    bad = sorted(set(metrics) - allowed)
    if bad:
        raise ValueError(f"Unknown selection metrics: {bad}; allowed={sorted(allowed)}")
    return metrics or ("Robs", "Rall", "wRall")


def parse_frame_models(text: str) -> tuple[str, ...]:
    """Return requested frame-scale model families."""

    models = tuple(part.strip().lower() for part in str(text).split(",") if part.strip())
    allowed = {"adaptive", "fixed"}
    bad = sorted(set(models) - allowed)
    if bad:
        raise ValueError(f"Unknown frame-scale model(s): {bad}; allowed={sorted(allowed)}")
    return models or ("adaptive", "fixed")


def parse_adaptive_counts(text: str, n_groups: int) -> list[int]:
    """Return target segment counts for adaptive neighboring-frame compression."""

    n_groups = max(1, int(n_groups))
    raw = str(text or "auto").strip().lower()
    if raw in {"off", "none", "false", "0"}:
        return []
    if raw == "auto":
        fractions = (0.35, 0.45, 0.50, 0.55, 0.60, 0.67, 0.75, 0.85)
        counts = [int(round(n_groups * frac)) for frac in fractions]
        if n_groups > 120:
            counts.extend([40, 60, 80, 100, 150, 200, 250, 300, 360, 420])
    else:
        counts = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if part.endswith("%"):
                counts.append(int(round(n_groups * float(part[:-1]) / 100.0)))
            else:
                counts.append(int(float(part)))
    return sorted({max(1, min(n_groups, int(v))) for v in counts if int(v) > 0})


def parse_detector_targets(text: str, detector_base: int) -> list[int]:
    """Parse detector flat-field choices, including the production-safe exact mode."""

    detector_base = max(1, int(detector_base))
    out: list[int] = []
    for part in str(text or "exact").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"none", "off", "0"}:
            out.append(0)
        elif token == "exact":
            out.append(detector_base)
        elif token == "auto":
            out.extend([detector_base])
        else:
            out.append(int(float(token)))
    out = [v for v in out if 0 <= int(v) <= detector_base]
    if not out:
        out = [detector_base]
    return sorted(set(out), key=out.index)


def infer_group_width_deg(summary: dict) -> float:
    """Return the physical angle represented by one existing row-table group."""

    nui = (summary.get("best_native") or {}).get("nuisance") or {}
    value = safe_float(nui.get("group_width_deg"), float("nan"))
    return value if np.isfinite(value) and value > 0 else float("nan")


def build_group_strides(args: argparse.Namespace, summary: dict) -> tuple[list[int], dict[str, object]]:
    """Build group strides from explicit strides and optional physical window size."""

    strides = parse_csv_ints(args.group_strides)
    group_width_deg = infer_group_width_deg(summary)
    scan = {
        "enabled": bool(float(args.window_scan_deg) > 0),
        "requested_window_scan_deg": float(args.window_scan_deg),
        "source_group_width_deg": group_width_deg,
        "max_stride_from_window": None,
    }
    if float(args.window_scan_deg) > 0:
        if not np.isfinite(group_width_deg) or group_width_deg <= 0:
            raise ValueError("--window-scan-deg requires best_native.nuisance.group_width_deg in the source summary")
        max_stride = max(1, int(math.ceil(float(args.window_scan_deg) / group_width_deg)))
        scan["max_stride_from_window"] = max_stride
        strides.extend(range(1, max_stride + 1))

    finite = sorted({int(s) for s in strides if int(s) < 10**8})
    has_global = any(int(s) >= 10**8 for s in strides)
    out = finite + ([10**9] if has_global else [])
    return out, scan


def split_masks(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    split = str_col(rows, "split")
    masks = {"all": np.ones(len(rows), dtype=bool)}
    for name in sorted(set(split)):
        if name and name.lower() not in {"nan", "none"}:
            masks[name] = split == name
    if "work" not in masks:
        masks["work"] = masks["all"]
    if "free" not in masks:
        masks["free"] = masks["all"]
    return masks


def source_sigma_array(rows: list[dict[str, str]]) -> np.ndarray:
    """Return immutable imported sigma(Iobs), with historical fallbacks."""

    for name in ("source_sigma", "xds_sigma", "sigma", "sigma_eff"):
        if name in rows[0]:
            values = col(rows, name)
            if np.any(np.isfinite(values) & (values > 0)):
                return values
    return np.full(len(rows), np.nan, dtype=float)


def jana_adjusted_observations(obs: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the weak/negative-observation convention reported by Klar et al."""

    weak = obs < 0.01 * sigma
    return np.where(weak, 0.0, obs), weak


def row_sqrt_weights(obs: np.ndarray, sigma: np.ndarray, instability_u: float) -> np.ndarray:
    obs_adjusted, weak = jana_adjusted_observations(obs, sigma)
    sqrt_obs = np.sqrt(np.maximum(obs_adjusted, 0.0))
    sigma_sqrt = sigma / np.maximum(2.0 * sqrt_obs, 1e-30)
    sigma_sqrt[weak] = 5.0 * np.sqrt(sigma[weak])
    denom = np.sqrt(sigma_sqrt * sigma_sqrt + (float(instability_u) * sqrt_obs) ** 2)
    # The GROND row/MR wR expression squares w*(Iobs-Icalc), so the least-squares
    # normal equations use w**2 as the intensity residual weight.
    return 1.0 / np.maximum(denom * denom, 1e-30)


def shelx_weights(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, a: float, b: float) -> np.ndarray:
    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    return 1.0 / np.maximum(sig * sig + (float(a) * p) ** 2 + float(b) * p, 1e-30)


def objective_weights(
    obs: np.ndarray,
    sig: np.ndarray,
    pred_scaled: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
) -> np.ndarray:
    if objective == "sigma":
        return 1.0 / np.maximum(sig * sig, 1e-30)
    if objective == "native":
        return shelx_weights(obs, sig, pred_scaled, native_wght_a, native_wght_b)
    if objective == "row_mr":
        return row_sqrt_weights(obs, sig, instability_u)
    raise ValueError(f"unknown objective {objective!r}")


def solve_scale(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    return max(float(np.sum(weight * x * y) / max(np.sum(weight * x * x), 1e-30)), 0.0)


def fit_global_scale(
    obs: np.ndarray,
    sig: np.ndarray,
    raw: np.ndarray,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    n_iter: int,
) -> float:
    """Fit the shared fallback scale without consulting held-out rows."""

    fit = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0)
    if np.count_nonzero(fit) < 2:
        raise ValueError("at least two scale-fit rows with positive sigma and prediction are required")
    x = raw[fit]
    y = obs[fit]
    s = sig[fit]
    weight = 1.0 / np.maximum(s * s, 1e-30)
    scale = max(solve_scale(x, y, weight), 1e-30)
    for _ in range(int(n_iter)):
        pred_scaled = scale * x
        weight = objective_weights(y, s, pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
        new_scale = max(solve_scale(x, y, weight), 1e-30)
        if abs(new_scale - scale) <= 1e-9 * max(1.0, abs(scale)):
            return new_scale
        scale = new_scale
    return scale


def require_scale_coverage(
    raw: np.ndarray,
    scaled: np.ndarray,
    *topology: np.ndarray,
) -> None:
    """Fail closed if scale replay loses any valid raw prediction."""

    expected = np.isfinite(raw) & (raw > 0)
    missing = expected & (~np.isfinite(scaled) | (scaled <= 0))
    for values in topology:
        missing |= expected & ~np.isfinite(values)
    if np.any(missing):
        raise ValueError(
            "scale fit did not cover "
            f"{int(np.count_nonzero(missing))} valid raw prediction(s); refusing a partial row contract"
        )


def fit_group_scales(
    obs: np.ndarray,
    sig: np.ndarray,
    raw: np.ndarray,
    groups: np.ndarray,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    n_iter: int = 8,
) -> tuple[np.ndarray, dict[str, float]]:
    scaled = np.full_like(raw, np.nan, dtype=float)
    scales: dict[str, float] = {}
    good0 = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0)
    global_scale = fit_global_scale(
        obs,
        sig,
        raw,
        fit_mask,
        objective,
        native_wght_a,
        native_wght_b,
        instability_u,
        n_iter,
    )
    finite_groups = np.isfinite(groups)
    for group in sorted(set(groups[finite_groups].astype(int))):
        fit = good0 & (groups == group)
        apply = np.isfinite(raw) & finite_groups & (groups == group)
        if np.count_nonzero(fit) < 2:
            scales[str(int(group))] = float(global_scale)
            scaled[apply] = raw[apply] * global_scale
            continue
        x = raw[fit]
        y = obs[fit]
        s = sig[fit]
        weight = 1.0 / np.maximum(s * s, 1e-30)
        scale = solve_scale(x, y, weight)
        for _ in range(int(n_iter)):
            pred_scaled = scale * x
            weight = objective_weights(y, s, pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            new_scale = solve_scale(x, y, weight)
            if abs(new_scale - scale) <= 1e-9 * max(1.0, abs(scale)):
                scale = new_scale
                break
            scale = new_scale
        scales[str(int(group))] = float(scale)
        scaled[apply] = raw[apply] * scale
    require_scale_coverage(raw, scaled, groups)
    return scaled, scales


def fit_group_detector_scales(
    obs: np.ndarray,
    sig: np.ndarray,
    raw: np.ndarray,
    groups: np.ndarray,
    det_bins: np.ndarray,
    fit_mask: np.ndarray,
    objective: str,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
    n_iter: int = 16,
    detector_grid_size: int | None = None,
    detector_ridge: float = 0.05,
    detector_smoothness: float = 0.10,
    detector_min_rows: int = 20,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    scaled = np.full_like(raw, np.nan, dtype=float)
    good = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0) & np.isfinite(groups) & np.isfinite(det_bins)
    unique_groups = np.asarray(sorted(set(groups[np.isfinite(groups)].astype(int))), dtype=int)
    present_det = np.asarray(sorted(set(det_bins[np.isfinite(det_bins)].astype(int))), dtype=int)
    if len(unique_groups) == 0 or len(present_det) == 0:
        require_scale_coverage(raw, scaled, groups, det_bins)
        return scaled, {}, {}
    gmap = {int(g): i for i, g in enumerate(unique_groups)}
    gi = np.asarray([gmap.get(int(g), -1) if np.isfinite(g) else -1 for g in groups], dtype=int)
    n_bins = infer_square_grid_size(det_bins, detector_grid_size)
    n_tiles = n_bins * n_bins
    di = np.asarray(
        [int(d) if np.isfinite(d) and 0 <= int(d) < n_tiles else -1 for d in det_bins],
        dtype=int,
    )
    global_scale = fit_global_scale(
        obs,
        sig,
        raw,
        fit_mask,
        objective,
        native_wght_a,
        native_wght_b,
        instability_u,
        n_iter,
    )
    frame = np.full(len(unique_groups), global_scale, dtype=float)
    det = np.ones(n_tiles, dtype=float)
    for _ in range(int(n_iter)):
        base = raw * det[np.maximum(di, 0)]
        for j in range(len(unique_groups)):
            m = good & (gi == j) & np.isfinite(base) & (base > 0)
            if np.count_nonzero(m) < 2:
                continue
            pred_scaled = base[m] * frame[j]
            weight = objective_weights(obs[m], sig[m], pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            frame[j] = max(solve_scale(base[m], obs[m], weight), 1e-30)
        base = raw * frame[np.maximum(gi, 0)]
        diag = np.zeros(n_tiles, dtype=float)
        rhs = np.zeros(n_tiles, dtype=float)
        counts = np.zeros(n_tiles, dtype=int)
        for j in range(n_tiles):
            m = good & (di == j) & np.isfinite(base) & (base > 0)
            counts[j] = int(np.count_nonzero(m))
            if counts[j] < 2:
                continue
            pred_scaled = base[m] * det[j]
            weight = objective_weights(obs[m], sig[m], pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            diag[j] = float(np.sum(weight * base[m] * base[m]))
            rhs[j] = float(np.sum(weight * base[m] * obs[m]))
        det = solve_regularized_detector_field(
            diag,
            rhs,
            counts,
            n_bins=n_bins,
            ridge=float(detector_ridge),
            smoothness=float(detector_smoothness),
            min_rows=int(detector_min_rows),
        )
        finite = det[np.isfinite(det) & (det > 0)]
        if len(finite):
            median_det = float(np.median(finite))
            if median_det > 0:
                det /= median_det
                frame *= median_det
    apply = np.isfinite(raw) & (gi >= 0) & (di >= 0)
    scaled[apply] = raw[apply] * frame[gi[apply]] * det[di[apply]]
    require_scale_coverage(raw, scaled, groups, det_bins)
    frame_scales = {str(int(g)): float(frame[j]) for j, g in enumerate(unique_groups)}
    det_scales = {str(j): float(det[j]) for j in range(n_tiles)}
    return scaled, frame_scales, det_scales


def row_mr_metrics(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, mask: np.ndarray, instability_u: float, observed_sigma_cut: float) -> dict:
    m = mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"Nall": 0, "Nobs": 0, "Robs": float("nan"), "Rall": float("nan"), "wRall": float("nan")}
    obs_original = obs_m.copy()
    obs_m, _weak = jana_adjusted_observations(obs_m, sig_m)
    obs_pos = np.maximum(obs_m, 0.0)
    pred_pos = np.maximum(pred_m, 0.0)
    fobs = np.sqrt(obs_pos)
    fcalc = np.sqrt(pred_pos)
    observed = obs_original > float(observed_sigma_cut) * sig_m

    def r_factor(local_mask: np.ndarray) -> float:
        if not np.any(local_mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[local_mask] - fcalc[local_mask])) / max(np.sum(fobs[local_mask]), 1e-30))

    w = np.sqrt(row_sqrt_weights(obs_original, sig_m, instability_u))
    wr = float(np.sqrt(np.sum((w * np.abs(obs_m - pred_m)) ** 2) / max(np.sum((w * obs_m) ** 2), 1e-30)))
    return {
        "Nall": int(len(obs_m)),
        "Nobs": int(np.count_nonzero(observed)),
        "Robs": r_factor(observed),
        "Rall": r_factor(np.ones_like(observed, dtype=bool)),
        "wRall": wr,
    }


def native_metrics(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, mask: np.ndarray, wght_a: float, wght_b: float, n_params: int = 0) -> dict:
    m = mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "wR2_all": float("nan"), "GooF": float("nan")}
    gt = obs_m > 2.0 * sig_m
    fobs = np.sqrt(np.maximum(obs_m, 0.0))
    fpred = np.sqrt(np.maximum(pred_m, 0.0))

    def r1(local_mask: np.ndarray) -> float:
        if not np.any(local_mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[local_mask] - fpred[local_mask])) / max(np.sum(fobs[local_mask]), 1e-30))

    w = shelx_weights(obs_m, sig_m, pred_m, wght_a, wght_b)
    res = obs_m - pred_m
    dof = max(1, len(obs_m) - int(n_params))
    chi2 = float(np.sum(w * res * res))
    return {
        "n": int(len(obs_m)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "wR2_all": float(np.sqrt(chi2 / max(np.sum(w * obs_m * obs_m), 1e-30))),
        "GooF": float(np.sqrt(chi2 / dof)),
        "chi2": chi2,
        "reduced_chi2": float(chi2 / dof),
        "dof": int(dof),
    }


def information_criteria(native: dict, n_params: int) -> dict[str, float]:
    """Return Gaussian information criteria for a candidate scale contract.

    The known-sigma form treats the supplied endpoint sigmas as calibrated.  The
    variance-fitted form is more conservative when absolute sigmas are still
    under audit and compares relative residual compression while estimating one
    global residual variance.  GROND uses these as diagnostics; final promotion
    should still require the holdout/strict residual gates to pass.
    """

    n = max(1, int(native.get("n", 0)))
    k = max(0, int(n_params))
    chi2 = max(float(native.get("chi2", float("nan"))), 1e-30)
    logn = math.log(max(float(n), 2.0))
    aic_known = chi2 + 2.0 * k
    bic_known = chi2 + logn * k
    aic_variance = float(n) * math.log(chi2 / float(n)) + 2.0 * k
    bic_variance = float(n) * math.log(chi2 / float(n)) + logn * k
    return {
        "aic_known_sigma": float(aic_known),
        "bic_known_sigma": float(bic_known),
        "aic_fitted_variance": float(aic_variance),
        "bic_fitted_variance": float(bic_variance),
    }


def infer_detector_base(detector_bin: np.ndarray) -> int:
    finite = detector_bin[np.isfinite(detector_bin)]
    if len(finite) == 0:
        return 1
    max_bin = int(np.nanmax(finite))
    return max(1, int(math.ceil(math.sqrt(max_bin + 1))))


def coarsen_groups(groups: np.ndarray, stride: int, offset: int = 0) -> np.ndarray:
    if int(stride) >= 10**8:
        return np.zeros(len(groups), dtype=int)
    return np.floor_divide(groups.astype(int) - int(offset), max(1, int(stride))).astype(int)


def coarsen_detector(detector_bin: np.ndarray, base_bins: int, target_bins: int) -> np.ndarray:
    if int(target_bins) <= 1:
        return np.zeros(len(detector_bin), dtype=int)
    base_bins = max(1, int(base_bins))
    target_bins = max(1, int(target_bins))
    det = detector_bin.astype(int)
    x = np.mod(det, base_bins)
    y = np.floor_divide(det, base_bins)
    cx = np.minimum(np.floor_divide(x * target_bins, base_bins), target_bins - 1)
    cy = np.minimum(np.floor_divide(y * target_bins, base_bins), target_bins - 1)
    return (cy * target_bins + cx).astype(int)


def count_scale_params(groups: np.ndarray, det_bins: np.ndarray, fit_good: np.ndarray, target_detector_bins: int) -> tuple[int, int, int]:
    group_count = int(len([g for g in sorted(set(groups[fit_good].astype(int))) if np.count_nonzero((groups == g) & fit_good) >= 2]))
    det_count = int(target_detector_bins) * int(target_detector_bins) if int(target_detector_bins) > 1 else 0
    return group_count + det_count, group_count, det_count


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


def log_scale_step_rms(scales: dict[str, float]) -> float:
    """Estimate roughness of adjacent fitted frame/group scales in log space."""

    vals = []
    for _key, value in sorted(scales.items(), key=lambda kv: int(kv[0])):
        v = safe_float(value)
        if np.isfinite(v) and v > 0:
            vals.append(math.log(v))
    if len(vals) < 2:
        return float("nan")
    diff = np.diff(np.asarray(vals, dtype=float))
    return float(np.sqrt(np.mean(diff * diff)))


def baseline_scale_log_by_row(summary: dict, original_groups: np.ndarray) -> np.ndarray:
    """Map each row to the log of its original full-scale frame factor."""

    scales = ((summary.get("best_native") or {}).get("dyn_scales") or {})
    if not scales:
        return np.full(len(original_groups), float("nan"), dtype=float)
    out = np.full(len(original_groups), float("nan"), dtype=float)
    for group in sorted(set(original_groups.astype(int))):
        value = safe_float(scales.get(str(int(group))), float("nan"))
        if np.isfinite(value) and value > 0:
            out[original_groups == group] = math.log(value)
    return out


def baseline_scale_log_by_fit(
    raw: np.ndarray,
    baseline: np.ndarray,
    original_groups: np.ndarray,
    fit_good: np.ndarray,
) -> np.ndarray:
    """Estimate the original per-frame log scale directly from row predictions."""

    out = np.full(len(original_groups), float("nan"), dtype=float)
    groups = np.asarray(original_groups, dtype=int)
    for group in sorted(set(groups[fit_good].astype(int))):
        m = fit_good & (groups == group) & np.isfinite(raw) & np.isfinite(baseline) & (raw > 0) & (baseline > 0)
        if np.count_nonzero(m) < 2:
            continue
        scale = solve_scale(raw[m], baseline[m], np.ones(np.count_nonzero(m), dtype=float))
        if np.isfinite(scale) and scale > 0:
            out[groups == group] = math.log(scale)
    return out


def adaptive_segment_groups(
    original_groups: np.ndarray,
    baseline_log_scale: np.ndarray,
    fit_good: np.ndarray,
    target_segments: int,
) -> tuple[np.ndarray, int]:
    """Merge neighboring frame groups whose fitted scale curve is most similar."""

    groups = np.asarray(original_groups, dtype=int)
    unique_groups = np.asarray(sorted(set(groups[fit_good].astype(int))), dtype=int)
    if len(unique_groups) == 0:
        return np.zeros(len(groups), dtype=int), 0
    target_segments = max(1, min(int(target_segments), len(unique_groups)))
    values = np.zeros(len(unique_groups), dtype=float)
    weights = np.ones(len(unique_groups), dtype=float)
    finite_positions: list[int] = []
    for i, group in enumerate(unique_groups):
        m = fit_good & (groups == group) & np.isfinite(baseline_log_scale)
        weights[i] = max(1.0, float(np.count_nonzero(fit_good & (groups == group))))
        if np.count_nonzero(m):
            values[i] = float(np.nanmean(baseline_log_scale[m]))
            finite_positions.append(i)
        else:
            values[i] = float("nan")
    if finite_positions:
        finite = np.asarray(finite_positions, dtype=int)
        values = np.interp(np.arange(len(values)), finite, values[finite])
    else:
        values[:] = 0.0

    def make_seg(start: int, end: int) -> dict[str, float | int]:
        w = float(np.sum(weights[start : end + 1]))
        s = float(np.sum(weights[start : end + 1] * values[start : end + 1]))
        ss = float(np.sum(weights[start : end + 1] * values[start : end + 1] * values[start : end + 1]))
        return {"start": int(start), "end": int(end), "w": w, "s": s, "ss": ss}

    def sse(seg: dict[str, float | int]) -> float:
        return max(0.0, float(seg["ss"]) - float(seg["s"]) * float(seg["s"]) / max(float(seg["w"]), 1e-30))

    def merge_cost(left: dict[str, float | int], right: dict[str, float | int]) -> float:
        merged = {
            "start": int(left["start"]),
            "end": int(right["end"]),
            "w": float(left["w"]) + float(right["w"]),
            "s": float(left["s"]) + float(right["s"]),
            "ss": float(left["ss"]) + float(right["ss"]),
        }
        return sse(merged) - sse(left) - sse(right)

    segments = [make_seg(i, i) for i in range(len(unique_groups))]
    while len(segments) > target_segments:
        costs = [merge_cost(segments[i], segments[i + 1]) for i in range(len(segments) - 1)]
        j = int(np.argmin(np.asarray(costs, dtype=float)))
        merged = {
            "start": int(segments[j]["start"]),
            "end": int(segments[j + 1]["end"]),
            "w": float(segments[j]["w"]) + float(segments[j + 1]["w"]),
            "s": float(segments[j]["s"]) + float(segments[j + 1]["s"]),
            "ss": float(segments[j]["ss"]) + float(segments[j + 1]["ss"]),
        }
        segments[j : j + 2] = [merged]

    group_to_segment: dict[int, int] = {}
    for segment_id, seg in enumerate(segments):
        for group in unique_groups[int(seg["start"]) : int(seg["end"]) + 1]:
            group_to_segment[int(group)] = int(segment_id)
    mapped = np.asarray([group_to_segment.get(int(group), -1) for group in groups], dtype=int)
    missing = mapped < 0
    if np.any(missing):
        mapped[missing] = np.maximum(0, np.searchsorted(unique_groups, groups[missing], side="right") - 1)
        mapped[missing] = np.minimum(mapped[missing], len(segments) - 1)
    return mapped, len(segments)


def baseline_within_group_log_rms(
    baseline_log_scale: np.ndarray,
    candidate_groups: np.ndarray,
    fit_good: np.ndarray,
) -> float:
    """Measure how much the original fitted frame scales vary inside merged groups."""

    good = fit_good & np.isfinite(baseline_log_scale) & np.isfinite(candidate_groups)
    if np.count_nonzero(good) < 2:
        return float("nan")
    groups = candidate_groups[good].astype(int)
    logs = baseline_log_scale[good]
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    sums = np.bincount(inv, weights=logs)
    means = sums / np.maximum(counts, 1.0)
    residual = logs - means[inv]
    return float(np.sqrt(np.mean(residual * residual)))


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


def fmt(value: object, digits: int = 4) -> str:
    v = safe_float(value)
    return "n/a" if not np.isfinite(v) else f"{v:.{digits}f}"


def load_baseline_counts(summary: dict, n_rows: int) -> tuple[int, int, int]:
    best = summary.get("best_native") or {}
    scale_count = int(safe_float(best.get("scale_count"), 0.0))
    dof = safe_float((((best.get("metrics") or {}).get("all") or {}).get("dyn") or {}).get("dof"), float("nan"))
    total = int(max(0, round(n_rows - dof))) if np.isfinite(dof) else scale_count
    physical = max(0, total - scale_count)
    return scale_count, total, physical


def native_weight_from_summary(summary: dict, cli_a: float, cli_b: float) -> tuple[float, float]:
    weight = (summary.get("best_native") or {}).get("native_weight") or {}
    a = cli_a if np.isfinite(cli_a) else safe_float(weight.get("a"), 0.0)
    b = cli_b if np.isfinite(cli_b) else safe_float(weight.get("b"), 0.0)
    return float(a), float(b)


def write_report(path: Path, payload: dict) -> None:
    baseline = payload["baseline"]
    selected = payload["selected"]
    widest = payload.get("selected_widest")
    r_only = payload["selected_r_only"]
    selected_ic = payload.get("selected_information")
    lines = [
        f"# {payload['title']}",
        "",
        "This is a frozen-Bloch post-regularization scale-compression pass. The structural model and raw Bloch predictions were not changed.",
        "",
        "## Baseline",
        "",
        f"Baseline row table: `{payload['row_table']}`",
        f"Baseline scale parameters: `{baseline['scale_params']}`; total nuisance/scale parameters used for dof accounting: `{baseline['total_params']}`.",
        f"Baseline GROND row/MR dynamical metrics: Robs `{fmt(baseline['Robs'])}`, Rall `{fmt(baseline['Rall'])}`, wRall `{fmt(baseline['wRall'])}`.",
        f"Baseline native metrics: R1gt `{fmt(baseline['native']['R1_gt'])}`, R1all `{fmt(baseline['native']['R1_all'])}`, wR2all `{fmt(baseline['native']['wR2_all'])}`, GooF `{fmt(baseline['native'].get('GooF'))}`.",
        "",
        "## Selected Strict Model",
        "",
    ]
    if selected:
        lines.extend(
            [
                f"Selected model: `{selected['name']}`.",
                f"Physical scale-window width: `{fmt(selected.get('window_deg'))}` deg; group offset: `{selected.get('group_offset', 0)}`.",
                f"Scale parameters: `{selected['scale_params']}`; total parameters: `{selected['total_params']}`.",
                f"GROND row/MR metrics: Robs `{fmt(selected['Robs'])}` ({fmt(selected['delta_vs_baseline']['Robs'])}), Rall `{fmt(selected['Rall'])}` ({fmt(selected['delta_vs_baseline']['Rall'])}), wRall `{fmt(selected['wRall'])}` ({fmt(selected['delta_vs_baseline']['wRall'])}).",
                f"Native metrics: R1gt `{fmt(selected['native']['R1_gt'])}` ({fmt(selected['delta_vs_baseline']['native_R1_gt'])}), R1all `{fmt(selected['native']['R1_all'])}` ({fmt(selected['delta_vs_baseline']['native_R1_all'])}), wR2all `{fmt(selected['native']['wR2_all'])}` ({fmt(selected['delta_vs_baseline']['native_wR2_all'])}), GooF `{fmt(selected['native'].get('GooF'))}`.",
            ]
        )
    else:
        lines.append("No candidate satisfied the strict selection metrics within the requested tolerance.")
    lines.extend(["", "## Widest Strict Window", ""])
    if widest:
        lines.extend(
            [
                f"Widest strict model: `{widest['name']}`.",
                f"Physical scale-window width: `{fmt(widest.get('window_deg'))}` deg; group offset: `{widest.get('group_offset', 0)}`.",
                f"Scale parameters: `{widest['scale_params']}`; total parameters: `{widest['total_params']}`.",
                f"GROND row/MR metrics: Robs `{fmt(widest['Robs'])}` ({fmt(widest['delta_vs_baseline']['Robs'])}), Rall `{fmt(widest['Rall'])}` ({fmt(widest['delta_vs_baseline']['Rall'])}), wRall `{fmt(widest['wRall'])}` ({fmt(widest['delta_vs_baseline']['wRall'])}).",
                f"Native metrics: R1gt `{fmt(widest['native']['R1_gt'])}` ({fmt(widest['delta_vs_baseline']['native_R1_gt'])}), R1all `{fmt(widest['native']['R1_all'])}` ({fmt(widest['delta_vs_baseline']['native_R1_all'])}), wR2all `{fmt(widest['native']['wR2_all'])}` ({fmt(widest['delta_vs_baseline']['native_wR2_all'])}).",
            ]
        )
    else:
        lines.append("No candidate satisfied the strict selection metrics within the requested tolerance.")
    lines.extend(["", "## R-Only Fallback", ""])
    if r_only:
        lines.extend(
            [
                f"Selected R-only model: `{r_only['name']}`.",
                f"Physical scale-window width: `{fmt(r_only.get('window_deg'))}` deg; group offset: `{r_only.get('group_offset', 0)}`.",
                f"Scale parameters: `{r_only['scale_params']}`; total parameters: `{r_only['total_params']}`.",
                f"GROND row/MR metrics: Robs `{fmt(r_only['Robs'])}` ({fmt(r_only['delta_vs_baseline']['Robs'])}), Rall `{fmt(r_only['Rall'])}` ({fmt(r_only['delta_vs_baseline']['Rall'])}), wRall `{fmt(r_only['wRall'])}` ({fmt(r_only['delta_vs_baseline']['wRall'])}).",
            ]
        )
    else:
        lines.append("No candidate satisfied the R-only tolerance either.")
    lines.extend(["", "## Information-Criterion Selection", ""])
    if selected_ic:
        lines.extend(
            [
                "Among candidates that pass the strict residual gates, GROND also reports an information-criterion choice. This does not replace the residual gates; it answers how many scale parameters are justified after the candidate is already close enough to the full-scale result.",
                f"Selected information model: `{selected_ic['name']}`.",
                f"Scale parameters: `{selected_ic['scale_params']}`; total parameters: `{selected_ic['total_params']}`.",
                f"BIC with fitted residual variance: `{fmt(selected_ic.get('information', {}).get('bic_fitted_variance'))}`; AIC with fitted residual variance: `{fmt(selected_ic.get('information', {}).get('aic_fitted_variance'))}`.",
                f"Known-sigma BIC: `{fmt(selected_ic.get('information', {}).get('bic_known_sigma'))}`; known-sigma AIC: `{fmt(selected_ic.get('information', {}).get('aic_known_sigma'))}`.",
                f"Recommended compact scale contract for automatic output: `{selected_ic['name']}`. This is the strict-gated model with the best information score, not the lowest-parameter model overall.",
            ]
        )
    else:
        lines.append("No strict candidate was available for information-criterion selection.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"Strict tolerance metrics: `{', '.join(payload['selection_metrics'])}` with tolerance `{payload['tolerance']}`.",
            "The default automatic gate prioritizes all-row residuals and treats strong-row residuals as report-only diagnostics.",
            f"Window scan: `{payload.get('window_scan', {})}`; scan offsets: `{payload.get('scan_offsets', False)}`.",
            f"Frame-scale models tested: `{payload.get('frame_models', ())}`; adaptive segment counts: `{payload.get('adaptive_frame_counts', ())}`.",
            "Positive deltas mean the compressed model is worse than the baseline. Negative deltas mean the compressed refit improved that metric.",
            "AIC/BIC values are computed from the native row-space weighted chi-square and total nuisance/scale parameter count. Because endpoint sigmas are still audited separately, GROND uses information criteria as a parsimony diagnostic inside the strict residual gate rather than as the only promotion rule.",
            f"Candidate CSV: `{payload['candidate_csv']}`",
            f"Selected run-like directory: `{payload['selected_run_dir']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--summary", default="", help="Existing GROND run summary. Defaults to row-table parent/summary.json.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="GROND scale-compression polish")
    ap.add_argument("--raw-dyn-column", default="dyn_raw")
    ap.add_argument("--raw-born-column", default="born_raw")
    ap.add_argument("--baseline-dyn-column", default="dyn_scaled")
    ap.add_argument("--baseline-born-column", default="born_scaled")
    ap.add_argument("--fit-split", default="", help="Rows used for fitting scales. Defaults to summary scale_fit_split or all.")
    ap.add_argument("--frame-models", default="adaptive,fixed", help="Comma list of frame-scale compression families to test: adaptive,fixed.")
    ap.add_argument("--adaptive-frame-counts", default="auto", help="Adaptive neighboring-frame segment counts. Use auto, off, integers, or percentages such as 50%%.")
    ap.add_argument("--group-strides", default="1,2,4,8,16,32,64,128,global")
    ap.add_argument("--window-scan-deg", type=float, default=0.0, help="Also scan every integer group stride up to this physical window size.")
    ap.add_argument("--scan-offsets", action="store_true", help="For each non-global stride, test every possible block-boundary offset.")
    ap.add_argument("--detector-bins", default="0,2,4,8", help="Detector flat-field models. Automatic policy compares none, 2x2, 4x4, and diagnostic 8x8 fields on frozen predictions.")
    ap.add_argument("--detector-ridge", type=float, default=0.05, help="Dimensionless shrinkage of detector factors toward unity.")
    ap.add_argument("--detector-smoothness", type=float, default=0.10, help="Dimensionless nearest-neighbor detector-field smoothness penalty.")
    ap.add_argument("--detector-min-rows", type=int, default=20, help="Minimum work rows required for every independently supported detector tile.")
    ap.add_argument("--detector-min-holdout-improvement", type=float, default=0.002, help="Minimum absolute held-out wR2 improvement required on multiple deterministic splits.")
    ap.add_argument("--detector-required-improved-splits", type=int, default=2)
    ap.add_argument("--detector-max-bloch-gain-loss", type=float, default=0.002, help="Maximum loss of held-out Born-minus-Bloch wR2 advantage on any deterministic split.")
    ap.add_argument("--detector-min-field-correlation", type=float, default=0.50)
    ap.add_argument("--detector-max-field-log-rms", type=float, default=0.10)
    ap.add_argument("--detector-independent-calibration", action="store_true", help="Allow an 8x8 detector field to be promotable because an independent detector calibration supports it.")
    ap.add_argument("--scale-objective", choices=["row_mr", "native", "sigma"], default="row_mr")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument(
        "--selection-metrics",
        default="free_Rall,work_Rall",
        help=(
            "Row-factor gates used to earn scale compression. The default protects held-out and work "
            "JANA-style Rall; wRall is a secondary tie-break/report, and SHELXL/native wR2 do not select topology."
        ),
    )
    ap.add_argument("--instability-u", type=float, default=0.01)
    ap.add_argument("--observed-sigma-cut", type=float, default=3.0)
    ap.add_argument("--native-wght-a", type=float, default=float("nan"))
    ap.add_argument("--native-wght-b", type=float, default=float("nan"))
    args = ap.parse_args()

    row_table = Path(args.row_table).expanduser().resolve()
    rows = read_rows(row_table)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else row_table.parent / "summary.json"
    source_summary = read_json(summary_path)

    for required in (args.raw_dyn_column, args.raw_born_column, args.baseline_dyn_column, args.baseline_born_column):
        if required not in rows[0]:
            raise ValueError(f"Required column {required!r} is not present in {row_table}")

    obs = col(rows, "target_I")
    sig = source_sigma_array(rows)
    raw_dyn = col(rows, args.raw_dyn_column)
    raw_born = col(rows, args.raw_born_column)
    base_dyn = col(rows, args.baseline_dyn_column)
    groups0 = col(rows, "group").astype(int)
    det0 = col(rows, "detector_bin").astype(int)
    masks = split_masks(rows)
    fit_split = args.fit_split or str((source_summary.get("best_native") or {}).get("scale_fit_split") or "all")
    if fit_split not in masks:
        raise ValueError(f"Unknown fit split {fit_split!r}; available {sorted(masks)}")
    fit_mask = masks[fit_split]
    fit_good0 = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw_dyn) & (sig > 0) & (raw_dyn > 0)
    baseline_log_scale = baseline_scale_log_by_row(source_summary, groups0)
    if np.count_nonzero(np.isfinite(baseline_log_scale) & fit_good0) < 2:
        baseline_log_scale = baseline_scale_log_by_fit(raw_dyn, base_dyn, groups0, fit_good0)
    selection_metrics = parse_csv_metrics(args.selection_metrics)
    native_wght_a, native_wght_b = native_weight_from_summary(source_summary, args.native_wght_a, args.native_wght_b)
    base_scale_count, base_total_params, physical_params = load_baseline_counts(source_summary, len(rows))

    baseline_row_mr = row_mr_metrics(obs, sig, base_dyn, masks["all"], args.instability_u, args.observed_sigma_cut)
    baseline_row_work = row_mr_metrics(obs, sig, base_dyn, masks["work"], args.instability_u, args.observed_sigma_cut)
    baseline_row_free = row_mr_metrics(obs, sig, base_dyn, masks["free"], args.instability_u, args.observed_sigma_cut)
    baseline_native = native_metrics(obs, sig, base_dyn, masks["all"], native_wght_a, native_wght_b, n_params=base_total_params)
    group_width_deg = infer_group_width_deg(source_summary)
    source_detector_bins = int(
        safe_float(
            ((source_summary.get("best_native") or {}).get("nuisance") or {}).get(
                "detector_scale_bins"
            ),
            0.0,
        )
    )
    source_detector_params = (
        source_detector_bins * source_detector_bins if source_detector_bins > 1 else 0
    )
    baseline = {
        "name": "baseline_full_scale",
        "mode": "current_dyn_scaled",
        "objective": "existing",
        "group_stride": 1,
        "group_offset": 0,
        "window_deg": group_width_deg if np.isfinite(group_width_deg) else "",
        "detector_bins": "existing",
        "scale_params": base_scale_count,
        "group_scale_params": max(0, base_scale_count - source_detector_params),
        "detector_scale_params": source_detector_params,
        "total_params": base_total_params,
        **baseline_row_mr,
        "row_work": baseline_row_work,
        "row_free": baseline_row_free,
        "native": baseline_native,
        "information": information_criteria(baseline_native, base_total_params),
        "delta_vs_baseline": {
            "Robs": 0.0,
            "Rall": 0.0,
            "wRall": 0.0,
            "work_Rall": 0.0,
            "work_wRall": 0.0,
            "free_Rall": 0.0,
            "free_wRall": 0.0,
            "native_R1_gt": 0.0,
            "native_R1_all": 0.0,
            "native_wR2_all": 0.0,
        },
        "within_strict": True,
        "within_r_only": True,
        "detector_policy": {
            "status": "legacy_source_reference",
            "eligible": source_detector_bins < 8
            or bool(args.detector_independent_calibration),
            "detector_bins": source_detector_bins,
            "reasons": (
                []
                if source_detector_bins < 8 or bool(args.detector_independent_calibration)
                else ["eight_by_eight_requires_independent_calibration"]
            ),
        },
    }

    detector_base = infer_detector_base(det0)
    target_detector_bins = parse_detector_targets(args.detector_bins, detector_base)
    group_strides, window_scan = build_group_strides(args, source_summary)
    frame_models = parse_frame_models(args.frame_models)
    n_source_groups = int(len(set(groups0[fit_good0].astype(int)))) if np.count_nonzero(fit_good0) else int(len(set(groups0.astype(int))))
    adaptive_counts = parse_adaptive_counts(args.adaptive_frame_counts, n_source_groups) if "adaptive" in frame_models else []
    candidates: list[dict] = [baseline]

    def evaluate_grouping(
        *,
        name_prefix: str,
        frame_model: str,
        groups: np.ndarray,
        group_stride: int,
        group_offset: int,
        window_deg: float,
        adaptive_segments: int | str = "",
    ) -> None:
        fit_good = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw_dyn) & (sig > 0) & (raw_dyn > 0)
        for det_bins_target in target_detector_bins:
            if det_bins_target <= 1:
                dyn_scaled, frame_scales = fit_group_scales(obs, sig, raw_dyn, groups, fit_mask, args.scale_objective, native_wght_a, native_wght_b, args.instability_u)
                det_scales: dict[str, float] = {}
                mode = "frame_groups"
                det_label = 0
                det_bins = np.zeros(len(rows), dtype=int)
            else:
                det_bins = coarsen_detector(det0, detector_base, det_bins_target)
                dyn_scaled, frame_scales, det_scales = fit_group_detector_scales(
                    obs, sig, raw_dyn, groups, det_bins, fit_mask,
                    args.scale_objective, native_wght_a, native_wght_b,
                    args.instability_u, detector_grid_size=det_bins_target,
                    detector_ridge=args.detector_ridge,
                    detector_smoothness=args.detector_smoothness,
                    detector_min_rows=args.detector_min_rows,
                )
                mode = "frame_groups_plus_detector"
                det_label = det_bins_target
            scale_params, n_group_scales, n_detector_scales = count_scale_params(groups, det_bins, fit_good, det_label)
            total_params = int(scale_params + physical_params)
            metrics = row_mr_metrics(obs, sig, dyn_scaled, masks["all"], args.instability_u, args.observed_sigma_cut)
            row_work = row_mr_metrics(obs, sig, dyn_scaled, masks["work"], args.instability_u, args.observed_sigma_cut)
            row_free = row_mr_metrics(obs, sig, dyn_scaled, masks["free"], args.instability_u, args.observed_sigma_cut)
            native = native_metrics(obs, sig, dyn_scaled, masks["all"], native_wght_a, native_wght_b, n_params=total_params)
            candidate = {
                "name": f"{name_prefix}_det_{det_label}",
                "frame_model": frame_model,
                "mode": mode,
                "objective": args.scale_objective,
                "group_stride": int(group_stride),
                "group_offset": int(group_offset),
                "adaptive_segments": adaptive_segments,
                "window_deg": window_deg if np.isfinite(window_deg) else "",
                "detector_bins": int(det_label),
                "scale_params": int(scale_params),
                "group_scale_params": int(n_group_scales),
                "detector_scale_params": int(n_detector_scales),
                "total_params": int(total_params),
                "fit_scale_log_step_rms": log_scale_step_rms(frame_scales),
                "baseline_within_group_log_rms": baseline_within_group_log_rms(baseline_log_scale, groups, fit_good),
                "frame_scales": frame_scales,
                "detector_scales": det_scales,
                **metrics,
                "row_work": row_work,
                "row_free": row_free,
                "native": native,
                "information": information_criteria(native, total_params),
                "arrays": {"dyn": dyn_scaled},
            }
            candidate["detector_policy"] = detector_scale_policy_diagnostics(
                rows=rows,
                obs=obs,
                sig=sig,
                raw_dyn=raw_dyn,
                raw_born=raw_born,
                groups=groups,
                source_detector_bins=det0,
                detector_base=detector_base,
                detector_target=det_label,
                fit_mask=fit_mask,
                objective=args.scale_objective,
                native_wght_a=native_wght_a,
                native_wght_b=native_wght_b,
                instability_u=args.instability_u,
                detector_ridge=args.detector_ridge,
                detector_smoothness=args.detector_smoothness,
                detector_min_rows=args.detector_min_rows,
                min_holdout_improvement=args.detector_min_holdout_improvement,
                required_improved_splits=args.detector_required_improved_splits,
                max_bloch_gain_loss=args.detector_max_bloch_gain_loss,
                min_field_correlation=args.detector_min_field_correlation,
                max_field_log_rms=args.detector_max_field_log_rms,
                independent_calibration=args.detector_independent_calibration,
            )
            candidate["delta_vs_baseline"] = metric_delta(candidate, baseline)
            candidate["within_strict"] = within_tolerance(candidate["delta_vs_baseline"], selection_metrics, args.tolerance)
            candidate["within_r_only"] = within_tolerance(
                candidate["delta_vs_baseline"],
                ("free_Rall", "work_Rall"),
                args.tolerance,
            )
            candidates.append(candidate)

    if "adaptive" in frame_models:
        for segments in adaptive_counts:
            groups, actual_segments = adaptive_segment_groups(groups0, baseline_log_scale, fit_good0, segments)
            evaluate_grouping(
                name_prefix=f"adaptive_{actual_segments}",
                frame_model="adaptive",
                groups=groups,
                group_stride=0,
                group_offset=0,
                window_deg=float("nan"),
                adaptive_segments=int(actual_segments),
            )

    if "fixed" in frame_models:
        for stride in group_strides:
            if args.scan_offsets and int(stride) < 10**8:  # noqa: SIM108 - clearer than an inline range expression
                offsets = range(max(1, int(stride)))
            else:
                offsets = range(1)
            for offset in offsets:
                groups = coarsen_groups(groups0, stride, offset=offset)
                window_deg = float(stride) * group_width_deg if int(stride) < 10**8 and np.isfinite(group_width_deg) else float("nan")
                stride_label = stride if stride < 10**8 else "global"
                offset_label = f"_offset_{offset}" if (args.scan_offsets and int(stride) < 10**8) else ""
                evaluate_grouping(
                    name_prefix=f"stride_{stride_label}{offset_label}",
                    frame_model="fixed",
                    groups=groups,
                    group_stride=int(stride),
                    group_offset=int(offset),
                    window_deg=window_deg,
                )

    non_baseline = [c for c in candidates if c["name"] != "baseline_full_scale"]
    strict = [
        c
        for c in non_baseline
        if c["within_strict"]
        and bool((c.get("detector_policy") or {}).get("eligible", False))
    ]
    r_only = [
        c
        for c in non_baseline
        if c["within_r_only"]
        and bool((c.get("detector_policy") or {}).get("eligible", False))
    ]
    strict_added_detector = [c for c in strict if int(c.get("detector_bins", 0)) > 1]
    strict_no_detector = [c for c in strict if int(c.get("detector_bins", 0)) <= 1]
    promotion_pool = strict_no_detector or strict_added_detector
    selected = min(
        promotion_pool,
        key=lambda c: (
            c["scale_params"],
            c["total_params"],
            safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
            safe_float((c.get("row_free") or {}).get("wRall"), float("inf")),
            safe_float((c.get("row_work") or {}).get("Rall"), float("inf")),
        ),
    ) if promotion_pool else None
    selected_information = min(
        strict,
        key=lambda c: (
            safe_float(c.get("information", {}).get("bic_fitted_variance"), float("inf")),
            safe_float(c.get("information", {}).get("bic_known_sigma"), float("inf")),
            c["scale_params"],
            safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
        ),
    ) if strict else None
    selected_r_only = min(
        r_only,
        key=lambda c: (
            c["scale_params"],
            c["total_params"],
            safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
            safe_float((c.get("row_work") or {}).get("Rall"), float("inf")),
        ),
    ) if r_only else None
    selected_widest = max(strict, key=lambda c: (safe_float(c.get("window_deg"), -1.0), c["group_stride"], -c["scale_params"], -c["wRall"])) if strict else None
    selected_widest_r_only = max(r_only, key=lambda c: (safe_float(c.get("window_deg"), -1.0), c["group_stride"], -c["scale_params"], -c["wRall"])) if r_only else None
    policy_eligible = [
        c
        for c in non_baseline
        if bool((c.get("detector_policy") or {}).get("eligible", False))
    ]
    selected_for_output = selected or selected_information or selected_r_only or min(
        policy_eligible or non_baseline,
        key=lambda c: (
            max(0.0, safe_float(c["delta_vs_baseline"].get("free_Rall"), float("inf"))),
            max(0.0, safe_float(c["delta_vs_baseline"].get("work_Rall"), float("inf"))),
            c["scale_params"],
        ),
    )
    candidate_rows = [
        candidate_row(c)
        for c in sorted(
            candidates,
            key=lambda c: (
                c["scale_params"],
                safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
                safe_float((c.get("row_free") or {}).get("wRall"), float("inf")),
            ),
        )
    ]
    candidate_csv = out / "scale_compression_candidates.csv"
    write_rows(candidate_csv, candidate_rows, list(candidate_rows[0].keys()))

    def arrays_for_candidate(candidate: dict) -> dict[str, np.ndarray]:
        if candidate.get("frame_model") == "adaptive":
            candidate_groups, _ = adaptive_segment_groups(
                groups0,
                baseline_log_scale,
                fit_good0,
                int(candidate.get("adaptive_segments") or candidate.get("group_scale_params") or 1),
            )
        else:
            candidate_groups = coarsen_groups(
                groups0,
                int(candidate["group_stride"]),
                int(candidate.get("group_offset", 0)),
            )
        detector_target = int(candidate["detector_bins"])
        if detector_target <= 1:
            born, _ = fit_group_scales(
                obs, sig, raw_born, candidate_groups, fit_mask, args.scale_objective,
                native_wght_a, native_wght_b, args.instability_u,
            )
        else:
            candidate_detector = coarsen_detector(det0, detector_base, detector_target)
            born, _, _ = fit_group_detector_scales(
                obs, sig, raw_born, candidate_groups, candidate_detector, fit_mask,
                args.scale_objective, native_wght_a, native_wght_b,
                args.instability_u, detector_grid_size=detector_target,
                detector_ridge=args.detector_ridge,
                detector_smoothness=args.detector_smoothness,
                detector_min_rows=args.detector_min_rows,
            )
        return {"dyn": candidate["arrays"]["dyn"], "born": born}

    def materialize_candidate_run(run_dir: Path, candidate: dict) -> None:
        arrays = arrays_for_candidate(candidate)
        run_dir.mkdir(parents=True, exist_ok=True)
        updated_rows: list[dict[str, object]] = []
        for i, row in enumerate(rows):
            new = dict(row)
            new["dyn_full_scaled"] = row.get(args.baseline_dyn_column, "")
            new["born_full_scaled"] = row.get(args.baseline_born_column, "")
            new["dyn_scaled"] = f"{arrays['dyn'][i]:.17g}" if np.isfinite(arrays["dyn"][i]) else "nan"
            new["born_scaled"] = f"{arrays['born'][i]:.17g}" if np.isfinite(arrays["born"][i]) else "nan"
            new["scale_compression_model"] = candidate["name"]
            updated_rows.append(new)
        fieldnames = list(rows[0].keys())
        for extra in ("dyn_full_scaled", "born_full_scaled", "scale_compression_model"):
            if extra not in fieldnames:
                fieldnames.append(extra)
        write_rows(run_dir / "native_dyn_calc.csv", updated_rows, fieldnames)

        compact_summary = copy.deepcopy(source_summary)
        compact_summary["scale_compression"] = {
            "source_row_table": str(row_table),
            "source_summary": str(summary_path),
            "objective": args.scale_objective,
            "fit_split": fit_split,
            "tolerance": float(args.tolerance),
            "selection_metrics": selection_metrics,
            "frame_models": frame_models,
            "adaptive_frame_counts": adaptive_counts,
            "window_scan": window_scan,
            "scan_offsets": bool(args.scan_offsets),
            "baseline": {k: v for k, v in baseline.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
            "selected_strict": {k: v for k, v in (selected or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected else None,
            "selected_information": {k: v for k, v in (selected_information or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_information else None,
            "selected_r_only": {k: v for k, v in (selected_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_r_only else None,
            "selected_widest_strict": {k: v for k, v in (selected_widest or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest else None,
            "selected_widest_r_only": {k: v for k, v in (selected_widest_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest_r_only else None,
            "selected_output": {k: v for k, v in candidate.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
            "candidate_csv": str(candidate_csv),
            "detector_policy": candidate.get("detector_policy") or {},
        }
        compact_summary.setdefault("best_native", {})
        compact_summary["best_native"]["scale_count"] = int(candidate["scale_params"])
        compact_summary["best_native"]["frame_scale_count"] = int(
            candidate.get("group_scale_params", 0)
        )
        compact_summary["best_native"]["detector_scale_count"] = int(
            candidate.get("detector_scale_params", 0)
        )
        compact_summary["best_native"]["scale_fit_split"] = fit_split
        compact_summary["best_native"]["metrics"] = {
            split: {
                kind: native_metrics(
                    obs, sig, arrays[kind], masks[split], native_wght_a, native_wght_b,
                    n_params=int(candidate["total_params"]),
                )
                for kind in ("dyn", "born")
            }
            for split in ("all", "work", "free")
        }
        compact_summary["best_native"]["nuisance"] = {
            **((source_summary.get("best_native") or {}).get("nuisance") or {}),
            "scale_compression_model": candidate["name"],
            "scale_compression_objective": args.scale_objective,
            "scale_compression_group_stride": int(candidate["group_stride"]),
            "scale_compression_detector_bins": int(candidate["detector_bins"]),
        }
        (run_dir / "summary.json").write_text(json.dumps(compact_summary, indent=2, allow_nan=True) + "\n")

    selected_dir = out / "selected_compact_run"
    materialize_candidate_run(selected_dir, selected_for_output)
    selected_strict_dir = out / "selected_strict_run"
    if selected:
        materialize_candidate_run(selected_strict_dir, selected)
    if summary_path.exists():
        shutil.copy2(summary_path, out / "source_summary.json")

    payload = {
        "title": args.title,
        "row_table": str(row_table),
        "summary": str(summary_path),
        "candidate_csv": str(candidate_csv),
        "selected_run_dir": str(selected_dir),
        "selected_strict_run_dir": str(selected_strict_dir) if selected else "",
        "tolerance": float(args.tolerance),
        "selection_metrics": selection_metrics,
        "frame_models": frame_models,
        "adaptive_frame_counts": adaptive_counts,
        "fit_split": fit_split,
        "detector_base_bins": int(detector_base),
        "detector_policy": {
            "candidate_grids": target_detector_bins,
            "ridge_to_unity": float(args.detector_ridge),
            "spatial_smoothness": float(args.detector_smoothness),
            "minimum_rows_per_tile": int(args.detector_min_rows),
            "minimum_holdout_improvement": float(args.detector_min_holdout_improvement),
            "required_improved_splits": int(args.detector_required_improved_splits),
            "maximum_bloch_gain_loss": float(args.detector_max_bloch_gain_loss),
            "minimum_field_correlation": float(args.detector_min_field_correlation),
            "maximum_field_log_rms": float(args.detector_max_field_log_rms),
            "independent_calibration": bool(args.detector_independent_calibration),
        },
        "physical_params_carried_from_source": int(physical_params),
        "window_scan": window_scan,
        "scan_offsets": bool(args.scan_offsets),
        "baseline": {k: v for k, v in baseline.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
        "selected": {k: v for k, v in (selected or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected else None,
        "selected_information": {k: v for k, v in (selected_information or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_information else None,
        "selected_r_only": {k: v for k, v in (selected_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_r_only else None,
        "selected_widest": {k: v for k, v in (selected_widest or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest else None,
        "selected_widest_r_only": {k: v for k, v in (selected_widest_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest_r_only else None,
        "selected_output": {k: v for k, v in selected_for_output.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
        "top_candidates": [candidate_row(c) for c in sorted(candidates, key=lambda c: (c["scale_params"], c["wRall"], c["Rall"]))[:20]],
    }
    (out / "scale_compression_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    write_report(out / "REPORT.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
