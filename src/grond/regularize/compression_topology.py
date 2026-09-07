"""Scale-group topology and adaptive segmentation on frozen predictions. Compactness is useful only while physical row coverage and the matched control remain intact."""

from __future__ import annotations

import argparse
import math

import numpy as np

from .compression_weights import safe_float, solve_scale


def str_col(rows: list[dict[str, str]], name: str, default: str = "") -> np.ndarray:
    return np.asarray([str(r.get(name, default)) for r in rows])


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
