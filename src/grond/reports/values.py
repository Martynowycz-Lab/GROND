"""Read recorded summary metrics and format absent values explicitly."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json


def _quick_percent(value: object) -> str:
    """Format a fractional statistic as a percent for quick run reports."""

    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _quick_fraction(value: object) -> float | None:
    """Return a finite fractional statistic or None."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quick_number(value: object, digits: int = 3) -> str:
    """Format a small numeric value without raising."""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _quick_count_pair(row: Mapping[str, object]) -> str:
    """Return Nobs/Nall for a row-MR metrics block."""

    nobs = row.get("Nobs", "n/a")
    nall = row.get("Nall", "n/a")
    return f"{nobs}/{nall}"


def _quick_final_run(result: Mapping[str, object]) -> Path | None:
    """Return the final row-level run directory when one was recorded."""

    primary = result.get("primary_run")
    if isinstance(primary, str) and primary:
        return Path(primary)
    combined = result.get("combined")
    if isinstance(combined, str) and combined:
        return Path(combined)
    return None


def _quick_load_run_summary(run_dir: Path | None) -> Mapping[str, object]:
    """Load a single-dataset or combined multi-dataset row summary."""

    if run_dir is None:
        return {}
    summary = try_read_json(run_dir / "summary.json")
    if summary:
        return summary
    combined_summary = try_read_json(run_dir / "combined_summary.json")
    return combined_summary if isinstance(combined_summary, Mapping) else {}


def _quick_combined_native_metrics(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return single-summary-shaped native metrics for a combined run."""

    native_joint = summary.get("native_joint") if isinstance(summary.get("native_joint"), Mapping) else {}
    dyn = native_joint.get("dyn") if isinstance(native_joint.get("dyn"), Mapping) else {}
    born = native_joint.get("born") if isinstance(native_joint.get("born"), Mapping) else {}
    if not dyn and not born:
        return {}
    metrics: dict[str, dict[str, object]] = {}
    for split in ("all", "work", "free"):
        metrics[split] = {}
        if isinstance(dyn.get(split), Mapping):
            metrics[split]["dyn"] = dyn.get(split)
        if isinstance(born.get(split), Mapping):
            metrics[split]["born"] = born.get(split)
    return metrics


def _quick_best_native(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return the best native block from a native row run summary."""

    value = summary.get("best_native")
    if isinstance(value, Mapping):
        return value
    metrics = _quick_combined_native_metrics(summary)
    if not metrics:
        return {}
    wght = summary.get("wght") if isinstance(summary.get("wght"), Mapping) else {}
    nuisance = {"native_weight_mode": wght.get("mode", "per_dataset_native")}
    return {
        "metrics": metrics,
        "native_weight": wght,
        "nuisance": nuisance,
    }


def _quick_best_endpoint(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return the best corrected-HKL endpoint block from a native row run summary."""

    value = summary.get("best_endpoint")
    if isinstance(value, Mapping):
        return value
    combined_endpoint = summary.get("combined_endpoint")
    if isinstance(combined_endpoint, Mapping):
        shelxl = combined_endpoint.get("shelxl")
        return {"shelxl": shelxl} if isinstance(shelxl, Mapping) else {}
    return {}


def _quick_nuisance(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return the final nuisance parameter block for quick reporting."""

    best_native = _quick_best_native(summary)
    nuisance = best_native.get("nuisance")
    if isinstance(nuisance, Mapping):
        return nuisance
    config = summary.get("config")
    return config if isinstance(config, Mapping) else {}


def _quick_dataset_nuisance_contracts(summary: Mapping[str, object]) -> list[dict[str, object]]:
    """Return exact per-dataset nuisance blocks from a combined summary."""

    runs = summary.get("runs")
    if not isinstance(runs, list):
        return []
    contracts: list[dict[str, object]] = []
    for raw in runs:
        if not isinstance(raw, Mapping):
            continue
        best = raw.get("best_native_summary")
        best = best if isinstance(best, Mapping) else {}
        nuisance = best.get("nuisance")
        if not isinstance(nuisance, Mapping) or not nuisance:
            continue
        contracts.append(
            {
                "dataset": str(raw.get("name") or f"dataset_{len(contracts) + 1}"),
                "run_dir": str(raw.get("run_dir") or ""),
                "nuisance": dict(nuisance),
                "scale_count": best.get("scale_count"),
                "native_weight": dict(best.get("native_weight") or {})
                if isinstance(best.get("native_weight"), Mapping)
                else {},
            }
        )
    return contracts


def _quick_metric(metrics: Mapping[str, object], split: str, branch: str, key: str) -> object:
    """Read one nested native metric without exposing KeyError to report writing."""

    split_block = metrics.get(split)
    if not isinstance(split_block, Mapping):
        return None
    branch_block = split_block.get(branch)
    if not isinstance(branch_block, Mapping):
        return None
    return branch_block.get(key)


def _quick_dyn_metric(summary: Mapping[str, object], split: str, key: str) -> object:
    """Read one dynamical row-space metric from a native-stage summary."""

    best_native = _quick_best_native(summary)
    metrics = best_native.get("metrics") if isinstance(best_native.get("metrics"), Mapping) else {}
    return _quick_metric(metrics, split, "dyn", key)


def _quick_born_metric(summary: Mapping[str, object], split: str, key: str) -> object:
    """Read one matched Born row-space metric from a native-stage summary."""

    best_native = _quick_best_native(summary)
    metrics = best_native.get("metrics") if isinstance(best_native.get("metrics"), Mapping) else {}
    return _quick_metric(metrics, split, "born", key)


def _quick_float(value: object, default: float = float("nan")) -> float:
    """Return a finite float or a default."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _quick_int(value: object) -> int | None:
    """Return an int when a report value is integer-like."""

    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out
