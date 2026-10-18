"""Rank HKL transform candidates by intensity overlap with a solved HKL file, to shorten the Bloch transform scout."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from grond.engine.bloch_engine import parse_hkl_transform
from grond.engine.shelx_hkl import parse_hkl_observation_line


def read_hkl_intensity_rows(path: str | Path, *, max_rows: int = 0) -> list[tuple[int, int, int, float, float]]:
    """Read the HKL, intensity, and sigma columns from XDS or SHELX-style files."""

    rows: list[tuple[int, int, int, float, float]] = []
    hkl_path = Path(path).expanduser()
    if not hkl_path.exists():
        return rows
    with hkl_path.open(errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith(("!", "#")):
                continue
            try:
                h, k, ell, iobs, sigma = parse_hkl_observation_line(line)
                sigma = abs(sigma)
            except ValueError:
                continue
            if (h, k, ell) == (0, 0, 0):
                continue
            if not math.isfinite(iobs) or not math.isfinite(sigma):
                continue
            rows.append((h, k, ell, iobs, sigma))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def aggregate_intensity_by_hkl(rows: list[tuple[int, int, int, float, float]]) -> dict[tuple[int, int, int], float]:
    """Average repeated observations into one intensity per signed HKL."""

    buckets: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for h, k, ell, iobs, _sigma in rows:
        buckets[(h, k, ell)].append(float(iobs))
    return {hkl: float(sum(values) / len(values)) for hkl, values in buckets.items() if values}


def transform_hkl(hkl: tuple[int, int, int], transform: str) -> tuple[int, int, int]:
    """Apply a GROND signed-axis HKL transform to one index triplet."""

    matrix, _canonical = parse_hkl_transform(transform)
    return tuple((np.asarray(hkl, dtype=int) @ matrix.T).astype(int).tolist())


def score_transform_against_baseline(
    moving_rows: list[tuple[int, int, int, float, float]],
    baseline_by_hkl: dict[tuple[int, int, int], float],
    transform: str,
) -> dict:
    """Score one transform by direct intensity overlap with a solved/XSCALE HKL file."""

    pairs: list[tuple[float, float]] = []
    transformed_keys: set[tuple[int, int, int]] = set()
    for h, k, ell, iobs, _sigma in moving_rows:
        mapped = transform_hkl((h, k, ell), transform)
        transformed_keys.add(mapped)
        baseline_i = baseline_by_hkl.get(mapped)
        if baseline_i is None or iobs <= 0.0 or baseline_i <= 0.0:
            continue
        pairs.append((float(iobs), float(baseline_i)))
    unique_overlap = len(transformed_keys.intersection(baseline_by_hkl))
    if len(pairs) < 3:
        return {
            "transform": transform,
            "paired_rows": len(pairs),
            "unique_overlap": unique_overlap,
            "log_intensity_corr": float("nan"),
            "sqrt_intensity_r": float("nan"),
        }
    moving = np.asarray([p[0] for p in pairs], dtype=float)
    baseline = np.asarray([p[1] for p in pairs], dtype=float)
    log_moving = np.log1p(moving)
    log_baseline = np.log1p(baseline)
    if np.std(log_moving) > 0.0 and np.std(log_baseline) > 0.0:
        corr = float(np.corrcoef(log_moving, log_baseline)[0, 1])
    else:
        corr = float("nan")
    sqrt_moving = np.sqrt(np.maximum(moving, 0.0))
    sqrt_baseline = np.sqrt(np.maximum(baseline, 0.0))
    scale = float(np.median(sqrt_baseline / np.maximum(sqrt_moving, 1e-9)))
    denom = float(np.sum(np.abs(sqrt_baseline)))
    sqrt_r = float(np.sum(np.abs(scale * sqrt_moving - sqrt_baseline)) / denom) if denom > 0.0 else float("nan")
    return {
        "transform": transform,
        "paired_rows": len(pairs),
        "unique_overlap": unique_overlap,
        "log_intensity_corr": corr,
        "sqrt_intensity_r": sqrt_r,
    }


def rank_transforms_from_baseline_hkl(
    moving_hkl: str | Path,
    baseline_hkl: str | Path,
    transforms: list[str],
    *,
    max_rows: int = 0,
) -> dict:
    """Rank transform candidates using existing INTEGRATE/XDS and solved-HKL intensities."""

    transforms = list(dict.fromkeys(transforms))
    moving_rows = read_hkl_intensity_rows(moving_hkl, max_rows=max_rows)
    baseline_rows = read_hkl_intensity_rows(baseline_hkl)
    if not moving_rows or not baseline_rows:
        return {
            "status": "missing_input",
            "moving_hkl": str(moving_hkl),
            "baseline_hkl": str(baseline_hkl),
            "moving_rows": len(moving_rows),
            "baseline_rows": len(baseline_rows),
            "records": [],
        }
    baseline_by_hkl = aggregate_intensity_by_hkl(baseline_rows)
    records = [score_transform_against_baseline(moving_rows, baseline_by_hkl, transform) for transform in transforms]
    records = sorted(
        records,
        key=lambda row: (
            -float(row["log_intensity_corr"]) if math.isfinite(float(row["log_intensity_corr"])) else float("inf"),
            float(row["sqrt_intensity_r"]) if math.isfinite(float(row["sqrt_intensity_r"])) else float("inf"),
            -int(row["paired_rows"]),
        ),
    )
    return {
        "status": "ranked",
        "moving_hkl": str(Path(moving_hkl).expanduser()),
        "baseline_hkl": str(Path(baseline_hkl).expanduser()),
        "moving_rows": len(moving_rows),
        "baseline_rows": len(baseline_rows),
        "baseline_unique_hkl": len(baseline_by_hkl),
        "records": records,
    }


def metadata_prefilter_transforms(
    moving_hkl: str | Path,
    baseline_hkl: str | Path,
    transforms: list[str],
) -> tuple[list[str], dict]:
    """Rank the transforms by solved-HKL overlap (a diagnostic order; the caller never drops a candidate)."""

    max_candidates, min_pairs, min_correlation, min_correlation_gap = 3, 100, 0.70, 0.05
    payload = rank_transforms_from_baseline_hkl(moving_hkl, baseline_hkl, transforms, max_rows=0)
    payload.update(
        {
            "mode": "auto",
            "max_candidates": max_candidates,
            "min_pairs": min_pairs,
            "min_correlation": min_correlation,
            "min_correlation_gap": min_correlation_gap,
            "input_transforms": list(transforms),
        }
    )
    records = payload.get("records") or []
    if not records:
        payload["status"] = payload.get("status") or "missing_input"
        payload["selected_transforms"] = list(transforms)
        return list(transforms), payload
    best = records[0]
    best_corr = float(best.get("log_intensity_corr", float("nan")))
    paired = int(best.get("paired_rows", 0))
    if paired < min_pairs or not math.isfinite(best_corr) or best_corr < min_correlation:
        payload["status"] = "insufficient_confidence"
        payload["selected_transforms"] = list(transforms)
        return list(transforms), payload
    second_corr = float(records[1].get("log_intensity_corr", float("nan"))) if len(records) > 1 else float("-inf")
    corr_gap = best_corr - second_corr if math.isfinite(second_corr) else float("inf")
    payload["correlation_gap"] = corr_gap
    if corr_gap >= min_correlation_gap:
        payload["status"] = "promoted"
        payload["selected_transforms"] = [str(best["transform"])]
        return [str(best["transform"])], payload
    selected = [str(row["transform"]) for row in records[:max_candidates]]
    payload["status"] = "ambiguous_top_candidates"
    payload["selected_transforms"] = selected
    return selected, payload
