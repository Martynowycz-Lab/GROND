"""Choose usable model files and require hash-bound SHELXL feedback health records."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import read_json, try_read_json
from grond.io.config import DatasetSpec, PrepConfig


def best_nuisance(run_dir: Path) -> dict:
    """Return the best nuisance-parameter block from a stage summary."""

    summary = read_json(run_dir / "summary.json")
    return (summary.get("best_native") or {}).get("nuisance") or {}


def resolved_frame_offset_candidates(run_dir: Path) -> list[float]:
    """Return the geometry-resolved frame probes recorded by a native stage."""

    summary = read_json(run_dir / "summary.json")
    grids = summary.get("resolved_grids") if isinstance(summary.get("resolved_grids"), Mapping) else {}
    raw = grids.get("frame_offset_frames", []) if isinstance(grids, Mapping) else []
    values: list[float] = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and not any(abs(value - existing) <= 1e-12 for existing in values):
            values.append(value)
    return values


def shelx_model_is_usable(path: Path | str) -> bool:
    """Return whether a SHELX model path can be reused as a GROND structure."""

    p = Path(path)
    try:
        from grond.engine.native_row_regularize import shelx_res_health

        health = shelx_res_health(p, p.with_suffix(".lst"))
    except (ImportError, OSError, TypeError, ValueError):
        return False
    return bool(
        health.get("ok") and int(health.get("atom_count", 0) or 0) > 0 and int(health.get("sfac_count", 0) or 0) > 0
    )


def shelxl_feedback_record_safe_for_reuse(record: Mapping[str, object]) -> bool:
    """Return whether a feedback-loop record is safe as a later starting model."""

    endpoint_status = str(record.get("endpoint_shelxl_status", "")).strip().lower()
    if endpoint_status not in {"ok", "reused_validated_input_model"}:
        return False
    health = record.get("endpoint_res_health")
    if not isinstance(health, Mapping):
        return False
    if not bool(health.get("ok", False)):
        return False
    if health.get("safe_for_feedback") is False:
        return False
    try:
        atom_count = int(health.get("atom_count", 0) or 0)
        sfac_count = int(health.get("sfac_count", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    if atom_count <= 0 or sfac_count <= 0:
        return False
    try:
        npd_count = int(health.get("npd_count", 0) or 0)
    except (TypeError, ValueError):
        return False
    if npd_count > 0 or bool(health.get("has_npd")):
        return False

    output_value = str(record.get("output_model") or "")
    health_path_value = str(health.get("path") or "")
    recorded_sha = str(health.get("sha256") or "")
    if not output_value or not health_path_value or not recorded_sha:
        # Older summaries did not bind the output bytes to their health
        # record.  They are intentionally recomputed rather than reused.
        return False
    output = Path(output_value).expanduser()
    health_path = Path(health_path_value).expanduser()
    try:
        if output.resolve() != health_path.resolve():
            return False
        from grond.engine.native_row_regularize import shelx_res_health

        current = shelx_res_health(output, output.with_suffix(".lst"))
    except (ImportError, OSError, TypeError, ValueError):
        return False
    return bool(
        current.get("ok")
        and current.get("safe_for_feedback") is not False
        and int(current.get("atom_count", 0) or 0) > 0
        and int(current.get("sfac_count", 0) or 0) > 0
        and str(current.get("sha256") or "") == recorded_sha
        and int(current.get("npd_count", 0) or 0) <= 0
        and not bool(current.get("has_npd"))
    )


def best_endpoint_model(run_dir: Path) -> Path | None:
    """Return the polished endpoint RES path for a completed native stage."""

    summary = try_read_json(run_dir / "summary.json")
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    lst = shelxl.get("lst")
    if not lst:
        return None
    res = Path(lst).with_suffix(".res")
    record = {
        "output_model": str(res),
        "endpoint_shelxl_status": shelxl.get("status"),
        "endpoint_res_health": shelxl.get("res_health"),
    }
    return res if shelx_model_is_usable(res) and shelxl_feedback_record_safe_for_reuse(record) else None


def resolve_structure_update_strategy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Choose the structure-update engine for the current run.

    The SHELXL feedback loop is the default whenever a SHELXL executable is
    available.  Multi-dataset runs alternate in the same way as single-dataset
    runs: per-dataset Bloch rows are merged into one corrected endpoint HKL,
    one shared SHELXL update is run, and the updated shared model is sent back
    through every dataset.  --fast, --full-fast, --native-only or a missing
    SHELXL leave the structure unchanged (off).
    """

    from grond.cli_support.tools import shelxl_is_available

    requested = getattr(args, "structure_update", "auto")
    if requested in {"off", "shelxl_feedback"}:
        return requested
    if getattr(args, "fast", False) or getattr(args, "full_fast", False) or getattr(args, "native_only", False):
        return "off"
    if shelxl_is_available(getattr(args, "shelxl", "shelxl")):
        return "shelxl_feedback"
    return "off"


def shelxl_feedback_current_model(run_dir: Path) -> Path | None:
    """Return the latest model produced by a SHELXL feedback-loop run."""

    summary = try_read_json(run_dir / "shelxl_feedback_loop_summary.json")
    runs = summary.get("runs") or []
    current = summary.get("current_model")
    if current and shelx_model_is_usable(current):
        current_path = Path(str(current))
        if current_path == Path(str(summary.get("start_res", ""))) or any(
            rec.get("output_model")
            and Path(str(rec.get("output_model"))) == current_path
            and shelxl_feedback_record_safe_for_reuse(rec)
            for rec in runs
        ):
            return current_path
    for rec in reversed(runs):
        output = rec.get("output_model")
        if output and shelx_model_is_usable(output) and shelxl_feedback_record_safe_for_reuse(rec):
            return Path(output)
    return None


def shelxl_feedback_current_run(run_dir: Path) -> Path | None:
    """Return the run directory corresponding to the latest usable feedback model."""

    summary = try_read_json(run_dir / "shelxl_feedback_loop_summary.json")
    current = summary.get("current_model")
    for rec in reversed(summary.get("runs") or []):
        rec_run = rec.get("run_dir")
        if not rec_run:
            continue
        output = rec.get("output_model")
        if output and shelx_model_is_usable(output) and shelxl_feedback_record_safe_for_reuse(rec):
            return Path(rec_run).expanduser()
        if (
            current
            and output
            and Path(output) == Path(str(current))
            and shelx_model_is_usable(current)
            and shelxl_feedback_record_safe_for_reuse(rec)
        ):
            return Path(rec_run).expanduser()
    return None


def should_run_endpoint_feedback(run_dir: Path, args: argparse.Namespace) -> tuple[bool, dict]:
    """Decide whether a promoted endpoint factor should trigger one more cycle."""

    if getattr(args, "endpoint_feedback", "auto") == "off":
        return False, {"status": "disabled"}
    summary = try_read_json(run_dir / "summary.json")
    decision = summary.get("endpoint_factor_decision") or {}
    if getattr(args, "endpoint_feedback", "auto") == "on":
        return True, decision or {"status": "forced"}
    return bool(decision.get("promote_endpoint_resolution_scale", False)), decision


def dataset_baseline_hkl(prep: PrepConfig, ds: DatasetSpec) -> str:
    """Return the dataset-specific baseline HKL or the prep-level fallback."""

    return ds.baseline_hkl or prep.baseline_hkl
