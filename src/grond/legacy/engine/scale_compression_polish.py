#!/usr/bin/env python3
"""Compress final row-space scale factors after a GROND Bloch calculation.

This is a post-regularization polish.  It does not recompute Bloch waves and it
does not move the structure.  It reads a final ``native_dyn_calc.csv`` table,
keeps ``dyn_raw`` and ``born_raw`` fixed, and asks how simple the multiplicative
scale model can be while staying close to the current full-scale result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


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
    allowed = {"Robs", "Rall", "wRall", "native_R1_gt", "native_R1_all", "native_wR2_all"}
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


def row_sqrt_weights(obs: np.ndarray, sigma: np.ndarray, instability_u: float) -> np.ndarray:
    obs_pos = np.maximum(obs, 0.0)
    sqrt_obs = np.sqrt(obs_pos)
    sigma_sqrt = sigma / np.maximum(2.0 * sqrt_obs, 1e-6)
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
    for group in sorted(set(groups[good0].astype(int))):
        fit = good0 & (groups == group)
        apply = np.isfinite(raw) & (groups == group)
        if np.count_nonzero(fit) < 2:
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
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    scaled = np.full_like(raw, np.nan, dtype=float)
    good = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw) & (sig > 0) & (raw > 0) & np.isfinite(groups) & np.isfinite(det_bins)
    unique_groups = np.asarray(sorted(set(groups[good].astype(int))), dtype=int)
    unique_det = np.asarray(sorted(set(det_bins[good].astype(int))), dtype=int)
    if len(unique_groups) == 0 or len(unique_det) == 0:
        return scaled, {}, {}
    gmap = {int(g): i for i, g in enumerate(unique_groups)}
    dmap = {int(d): i for i, d in enumerate(unique_det)}
    gi = np.asarray([gmap.get(int(g), -1) if np.isfinite(g) else -1 for g in groups], dtype=int)
    di = np.asarray([dmap.get(int(d), -1) if np.isfinite(d) else -1 for d in det_bins], dtype=int)
    frame = np.ones(len(unique_groups), dtype=float)
    det = np.ones(len(unique_det), dtype=float)
    frame[:] = max(float(np.sum(raw[good] * obs[good]) / max(np.sum(raw[good] * raw[good]), 1e-30)), 1e-30)
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
        for j in range(len(unique_det)):
            m = good & (di == j) & np.isfinite(base) & (base > 0)
            if np.count_nonzero(m) < 10:
                continue
            pred_scaled = base[m] * det[j]
            weight = objective_weights(obs[m], sig[m], pred_scaled, objective, native_wght_a, native_wght_b, instability_u)
            det[j] = max(solve_scale(base[m], obs[m], weight), 1e-30)
        finite = det[np.isfinite(det) & (det > 0)]
        if len(finite):
            median_det = float(np.median(finite))
            if median_det > 0:
                det /= median_det
                frame *= median_det
    apply = np.isfinite(raw) & (gi >= 0) & (di >= 0)
    scaled[apply] = raw[apply] * frame[gi[apply]] * det[di[apply]]
    frame_scales = {str(int(g)): float(frame[j]) for j, g in enumerate(unique_groups)}
    det_scales = {str(int(d)): float(det[j]) for j, d in enumerate(unique_det)}
    return scaled, frame_scales, det_scales


def row_mr_metrics(obs: np.ndarray, sig: np.ndarray, pred: np.ndarray, mask: np.ndarray, instability_u: float, observed_sigma_cut: float) -> dict:
    m = mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"Nall": 0, "Nobs": 0, "Robs": float("nan"), "Rall": float("nan"), "wRall": float("nan")}
    obs_pos = np.maximum(obs_m, 0.0)
    pred_pos = np.maximum(pred_m, 0.0)
    fobs = np.sqrt(obs_pos)
    fcalc = np.sqrt(pred_pos)
    observed = obs_m > float(observed_sigma_cut) * sig_m

    def r_factor(local_mask: np.ndarray) -> float:
        if not np.any(local_mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[local_mask] - fcalc[local_mask])) / max(np.sum(fobs[local_mask]), 1e-30))

    w = np.sqrt(row_sqrt_weights(obs_m, sig_m, instability_u))
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
    return {
        "n": int(len(obs_m)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "wR2_all": float(np.sqrt(np.sum(w * res * res) / max(np.sum(w * obs_m * obs_m), 1e-30))),
        "GooF": float(np.sqrt(np.sum(w * res * res) / dof)),
        "dof": int(dof),
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


def log_scale_step_rms(scales: dict[str, float]) -> float:
    """Estimate roughness of adjacent fitted frame/group scales in log space."""

    vals = []
    for key, value in sorted(scales.items(), key=lambda kv: int(kv[0])):
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
    out["native_R1_gt"] = safe_float(candidate.get("native", {}).get("R1_gt")) - safe_float(baseline.get("native", {}).get("R1_gt"))
    out["native_R1_all"] = safe_float(candidate.get("native", {}).get("R1_all")) - safe_float(baseline.get("native", {}).get("R1_all"))
    out["native_wR2_all"] = safe_float(candidate.get("native", {}).get("wR2_all")) - safe_float(baseline.get("native", {}).get("wR2_all"))
    return out


def within_tolerance(delta: dict, metrics: tuple[str, ...], tolerance: float) -> bool:
    return all(np.isfinite(safe_float(delta.get(m))) and safe_float(delta.get(m)) <= float(tolerance) for m in metrics)


def candidate_row(candidate: dict) -> dict[str, object]:
    delta = candidate["delta_vs_baseline"]
    native = candidate["native"]
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
        "total_params": candidate["total_params"],
        "Nobs": candidate["Nobs"],
        "Nall": candidate["Nall"],
        "Robs": candidate["Robs"],
        "Rall": candidate["Rall"],
        "wRall": candidate["wRall"],
        "dRobs": delta["Robs"],
        "dRall": delta["Rall"],
        "dwRall": delta["wRall"],
        "native_R1_gt": native["R1_gt"],
        "native_R1_all": native["R1_all"],
        "native_wR2_all": native["wR2_all"],
        "native_dR1_gt": delta["native_R1_gt"],
        "native_dR1_all": delta["native_R1_all"],
        "native_dwR2_all": delta["native_wR2_all"],
        "fit_scale_log_step_rms": candidate.get("fit_scale_log_step_rms", ""),
        "baseline_within_group_log_rms": candidate.get("baseline_within_group_log_rms", ""),
        "within_strict": candidate["within_strict"],
        "within_r_only": candidate["within_r_only"],
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
        f"Baseline native metrics: R1gt `{fmt(baseline['native']['R1_gt'])}`, R1all `{fmt(baseline['native']['R1_all'])}`, wR2all `{fmt(baseline['native']['wR2_all'])}`.",
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
                f"Native metrics: R1gt `{fmt(selected['native']['R1_gt'])}` ({fmt(selected['delta_vs_baseline']['native_R1_gt'])}), R1all `{fmt(selected['native']['R1_all'])}` ({fmt(selected['delta_vs_baseline']['native_R1_all'])}), wR2all `{fmt(selected['native']['wR2_all'])}` ({fmt(selected['delta_vs_baseline']['native_wR2_all'])}).",
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
    ap.add_argument("--detector-bins", default="exact", help="Detector flat-field models. Use exact for the source detector-bin grid, or explicit NxN values; 0 disables detector scaling.")
    ap.add_argument("--scale-objective", choices=["row_mr", "native", "sigma"], default="native")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--selection-metrics", default="Rall,wRall,native_R1_all,native_wR2_all")
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
    sig = col(rows, "sigma_eff")
    raw_dyn = col(rows, args.raw_dyn_column)
    raw_born = col(rows, args.raw_born_column)
    base_dyn = col(rows, args.baseline_dyn_column)
    base_born = col(rows, args.baseline_born_column)
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
    baseline_native = native_metrics(obs, sig, base_dyn, masks["all"], native_wght_a, native_wght_b, n_params=base_total_params)
    group_width_deg = infer_group_width_deg(source_summary)
    baseline = {
        "name": "baseline_full_scale",
        "mode": "current_dyn_scaled",
        "objective": "existing",
        "group_stride": 1,
        "group_offset": 0,
        "window_deg": group_width_deg if np.isfinite(group_width_deg) else "",
        "detector_bins": "existing",
        "scale_params": base_scale_count,
        "total_params": base_total_params,
        **baseline_row_mr,
        "native": baseline_native,
        "delta_vs_baseline": {"Robs": 0.0, "Rall": 0.0, "wRall": 0.0, "native_R1_gt": 0.0, "native_R1_all": 0.0, "native_wR2_all": 0.0},
        "within_strict": True,
        "within_r_only": True,
    }

    detector_base = infer_detector_base(det0)
    target_detector_bins = parse_detector_targets(args.detector_bins, detector_base)
    group_strides, window_scan = build_group_strides(args, source_summary)
    frame_models = parse_frame_models(args.frame_models)
    n_source_groups = int(len(set(groups0[fit_good0].astype(int)))) if np.count_nonzero(fit_good0) else int(len(set(groups0.astype(int))))
    adaptive_counts = parse_adaptive_counts(args.adaptive_frame_counts, n_source_groups) if "adaptive" in frame_models else []
    candidates: list[dict] = [baseline]
    best_arrays: dict[str, np.ndarray] = {"dyn": base_dyn, "born": base_born}

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
                dyn_scaled, frame_scales, det_scales = fit_group_detector_scales(obs, sig, raw_dyn, groups, det_bins, fit_mask, args.scale_objective, native_wght_a, native_wght_b, args.instability_u)
                mode = "frame_groups_plus_detector"
                det_label = det_bins_target
            scale_params, n_group_scales, n_detector_scales = count_scale_params(groups, det_bins, fit_good, det_label)
            total_params = int(scale_params + physical_params)
            metrics = row_mr_metrics(obs, sig, dyn_scaled, masks["all"], args.instability_u, args.observed_sigma_cut)
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
                "native": native,
                "arrays": {"dyn": dyn_scaled},
            }
            candidate["delta_vs_baseline"] = metric_delta(candidate, baseline)
            candidate["within_strict"] = within_tolerance(candidate["delta_vs_baseline"], selection_metrics, args.tolerance)
            candidate["within_r_only"] = within_tolerance(candidate["delta_vs_baseline"], ("Robs", "Rall"), args.tolerance)
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
            if args.scan_offsets and int(stride) < 10**8:
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
    strict = [c for c in non_baseline if c["within_strict"]]
    r_only = [c for c in non_baseline if c["within_r_only"]]
    selected = min(strict, key=lambda c: (c["scale_params"], c["total_params"], c["wRall"], c["Rall"])) if strict else None
    selected_r_only = min(r_only, key=lambda c: (c["scale_params"], c["total_params"], c["Rall"], c["wRall"])) if r_only else None
    selected_widest = max(strict, key=lambda c: (safe_float(c.get("window_deg"), -1.0), c["group_stride"], -c["scale_params"], -c["wRall"])) if strict else None
    selected_widest_r_only = max(r_only, key=lambda c: (safe_float(c.get("window_deg"), -1.0), c["group_stride"], -c["scale_params"], -c["wRall"])) if r_only else None
    selected_for_output = selected or selected_r_only or min(non_baseline, key=lambda c: (max(0.0, c["delta_vs_baseline"]["Rall"]), c["scale_params"]))
    if selected_for_output.get("frame_model") == "adaptive":
        selected_groups, _ = adaptive_segment_groups(
            groups0,
            baseline_log_scale,
            fit_good0,
            int(selected_for_output.get("adaptive_segments") or selected_for_output.get("group_scale_params") or 1),
        )
    else:
        selected_groups = coarsen_groups(groups0, int(selected_for_output["group_stride"]), int(selected_for_output.get("group_offset", 0)))
    selected_det_target = int(selected_for_output["detector_bins"])
    if selected_det_target <= 1:
        born_for_output, _ = fit_group_scales(obs, sig, raw_born, selected_groups, fit_mask, args.scale_objective, native_wght_a, native_wght_b, args.instability_u)
    else:
        selected_det = coarsen_detector(det0, detector_base, selected_det_target)
        born_for_output, _, _ = fit_group_detector_scales(obs, sig, raw_born, selected_groups, selected_det, fit_mask, args.scale_objective, native_wght_a, native_wght_b, args.instability_u)
    best_arrays = {"dyn": selected_for_output["arrays"]["dyn"], "born": born_for_output}

    candidate_rows = [candidate_row(c) for c in sorted(candidates, key=lambda c: (c["scale_params"], c["wRall"], c["Rall"]))]
    candidate_csv = out / "scale_compression_candidates.csv"
    write_rows(candidate_csv, candidate_rows, list(candidate_rows[0].keys()))

    selected_dir = out / "selected_compact_run"
    selected_dir.mkdir(parents=True, exist_ok=True)
    updated_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        new = dict(row)
        new["dyn_full_scaled"] = row.get(args.baseline_dyn_column, "")
        new["born_full_scaled"] = row.get(args.baseline_born_column, "")
        new["dyn_scaled"] = f"{best_arrays['dyn'][i]:.17g}" if np.isfinite(best_arrays["dyn"][i]) else "nan"
        new["born_scaled"] = f"{best_arrays['born'][i]:.17g}" if np.isfinite(best_arrays["born"][i]) else "nan"
        new["scale_compression_model"] = selected_for_output["name"]
        updated_rows.append(new)
    fieldnames = list(rows[0].keys())
    for extra in ("dyn_full_scaled", "born_full_scaled", "scale_compression_model"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_rows(selected_dir / "native_dyn_calc.csv", updated_rows, fieldnames)

    compact_summary = dict(source_summary)
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
        "selected_r_only": {k: v for k, v in (selected_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_r_only else None,
        "selected_widest_strict": {k: v for k, v in (selected_widest or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest else None,
        "selected_widest_r_only": {k: v for k, v in (selected_widest_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest_r_only else None,
        "selected_output": {k: v for k, v in selected_for_output.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
        "candidate_csv": str(candidate_csv),
    }
    compact_summary.setdefault("best_native", {})
    compact_summary["best_native"]["scale_count"] = int(selected_for_output["scale_params"])
    compact_summary["best_native"]["scale_fit_split"] = fit_split
    compact_summary["best_native"]["metrics"] = {
        "all": {
            "dyn": selected_for_output["native"],
            "born": native_metrics(obs, sig, best_arrays["born"], masks["all"], native_wght_a, native_wght_b, n_params=int(selected_for_output["total_params"])),
        }
    }
    compact_summary["best_native"]["nuisance"] = {
        **((source_summary.get("best_native") or {}).get("nuisance") or {}),
        "scale_compression_model": selected_for_output["name"],
        "scale_compression_objective": args.scale_objective,
        "scale_compression_group_stride": int(selected_for_output["group_stride"]),
        "scale_compression_detector_bins": int(selected_for_output["detector_bins"]),
    }
    (selected_dir / "summary.json").write_text(json.dumps(compact_summary, indent=2, allow_nan=True) + "\n")
    if summary_path.exists():
        shutil.copy2(summary_path, out / "source_summary.json")

    payload = {
        "title": args.title,
        "row_table": str(row_table),
        "summary": str(summary_path),
        "candidate_csv": str(candidate_csv),
        "selected_run_dir": str(selected_dir),
        "tolerance": float(args.tolerance),
        "selection_metrics": selection_metrics,
        "frame_models": frame_models,
        "adaptive_frame_counts": adaptive_counts,
        "fit_split": fit_split,
        "detector_base_bins": int(detector_base),
        "physical_params_carried_from_source": int(physical_params),
        "window_scan": window_scan,
        "scan_offsets": bool(args.scan_offsets),
        "baseline": {k: v for k, v in baseline.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
        "selected": {k: v for k, v in (selected or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected else None,
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
