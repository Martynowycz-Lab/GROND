"""Summarize residual and correction distributions from the recorded row tables."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path

from grond.reports.values import _quick_float, _quick_number


def _quick_residual_z_summary(csv_paths: Path | list[Path]) -> dict[str, dict[str, float | int]]:
    """Summarize residual z values across one or more frozen native tables."""

    rows: dict[str, list[float]] = {"all": [], "work": [], "free": []}
    paths = [csv_paths] if isinstance(csv_paths, Path) else list(csv_paths)
    for csv_path in paths:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    z = _quick_float(row.get("dyn_residual_over_sigma"))
                    if not math.isfinite(z):
                        continue
                    rows["all"].append(z)
                    split = str(row.get("split", "")).lower()
                    if split in {"work", "free"}:
                        rows[split].append(z)
        except OSError:
            continue

    out: dict[str, dict[str, float | int]] = {}
    for split, values in rows.items():
        if not values:
            continue
        abs_values = sorted(abs(v) for v in values)
        values_sorted = sorted(values)
        n = len(values)
        out[split] = {
            "n": n,
            "mean_z": float(sum(values) / n),
            "rms_z": float(math.sqrt(sum(v * v for v in values) / n)),
            "median_z": float(values_sorted[n // 2] if n % 2 else 0.5 * (values_sorted[n // 2 - 1] + values_sorted[n // 2])),
            "median_abs_z": float(abs_values[n // 2] if n % 2 else 0.5 * (abs_values[n // 2 - 1] + abs_values[n // 2])),
        }
    return out


def _quick_quantile(sorted_values: list[float], q: float) -> float:
    """Return a linear-interpolated quantile from sorted finite values."""

    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float((1.0 - frac) * sorted_values[lo] + frac * sorted_values[hi])


def _quick_numeric_distribution(values: list[float]) -> dict[str, float | int]:
    """Return compact finite-value distribution statistics."""

    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "min": float(finite[0]),
        "p05": _quick_quantile(finite, 0.05),
        "median": _quick_quantile(finite, 0.50),
        "mean": float(sum(finite) / len(finite)),
        "p95": _quick_quantile(finite, 0.95),
        "max": float(finite[-1]),
    }


def _quick_csv_numeric_audit(csv_path: Path, columns: list[str]) -> dict[str, object]:
    """Summarize numeric row-table columns without pandas."""

    out: dict[str, object] = {
        "path": str(csv_path),
        "exists": csv_path.exists(),
        "rows": 0,
        "fieldnames": [],
        "columns": {},
        "split_counts": {},
        "group_count": 0,
        "detector_bin_count": 0,
        "sg_window_clipped_count": 0,
    }
    if not csv_path.exists():
        return out
    values: dict[str, list[float]] = {column: [] for column in columns}
    split_counts: dict[str, int] = {}
    groups: set[str] = set()
    detector_bins: set[str] = set()
    clipped = 0
    try:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            out["fieldnames"] = list(reader.fieldnames or [])
            for row in reader:
                out["rows"] = int(out["rows"]) + 1
                split = str(row.get("split") or "unknown")
                split_counts[split] = split_counts.get(split, 0) + 1
                if row.get("group") not in {None, ""}:
                    groups.add(str(row.get("group")))
                if row.get("detector_bin") not in {None, ""}:
                    detector_bins.add(str(row.get("detector_bin")))
                if str(row.get("sg_window_clipped", "")).strip() in {"1", "true", "True"}:
                    clipped += 1
                for column in columns:
                    value = _quick_float(row.get(column))
                    if math.isfinite(value):
                        values[column].append(value)
    except OSError as exc:
        out["error"] = str(exc)
        return out
    out["split_counts"] = split_counts
    out["group_count"] = len(groups)
    out["detector_bin_count"] = len(detector_bins)
    out["sg_window_clipped_count"] = clipped
    out["columns"] = {column: _quick_numeric_distribution(vals) for column, vals in values.items()}
    return out


def _quick_distribution_row(summary: Mapping[str, object], label: str, units: str = "") -> str:
    """Format one numeric-distribution row for a Markdown table."""

    if int(summary.get("n", 0) or 0) <= 0:
        return f"| {label} | {units} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {units} | {summary.get('n', 'n/a')} | "
        f"{_quick_number(summary.get('min'), 6)} | {_quick_number(summary.get('p05'), 6)} | "
        f"{_quick_number(summary.get('median'), 6)} | {_quick_number(summary.get('mean'), 6)} | "
        f"{_quick_number(summary.get('p95'), 6)} | {_quick_number(summary.get('max'), 6)} |"
    )
