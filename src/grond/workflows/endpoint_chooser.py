"""Choose the publication endpoint among a run's verified endpoint candidates.

Candidates come from native and combined summaries; each one's RES/HKL/LST provenance is
hash-verified. Writes publication_endpoint_choice.json with Markdown and LaTeX reports.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from grond.engine.shelx_diagnostics import parse_shelx_lst_metrics, parse_shelx_lst_pathologies
from grond.engine.shelx_hkl import audit_fixed_width_precision
from grond.forensics.resolution import classify_endpoint_candidate

# Preferred SHELXL GooF range; outside it the chooser warns but never blocks an endpoint.
GOOF_WARNING_MIN = 0.85
GOOF_WARNING_MAX = 1.15


def _float(value, fallback: float = float("nan")) -> float:
    """Convert a scalar to a finite float when possible."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _int(value, fallback: int = 0) -> int:
    """Convert a scalar to a non-negative integer when possible."""

    try:
        out = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(0, out)


def _read_json(path: Path) -> dict:
    """Read a JSON file and return an empty dict when it is absent or malformed."""

    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _percent(value: object) -> str:
    """Format a fractional residual as a percentage for reports."""

    value_f = _float(value)
    return "n/a" if not math.isfinite(value_f) else f"{100.0 * value_f:.2f}%"


def _metric_key(candidate: Mapping) -> tuple[float, float, float, str]:
    """Sort endpoint diagnostics without letting SHELXL wR2 dominate."""

    return (
        _float(candidate.get("R1_all"), float("inf")),
        _float(candidate.get("R1_gt"), float("inf")),
        _float(candidate.get("wR2_all"), float("inf")),
        str(candidate.get("id", "")),
    )


def _is_combined_candidate(candidate: Mapping) -> bool:
    """Return whether an endpoint came from a merged multi-dataset row table."""

    return str(candidate.get("source_kind", "")).startswith("combined_")


# The unmerged multi-crystal SHELXL endpoint (batch = crystal, MERG 0, BASF).  Its R1 is over unmerged rows,
# a different denominator from the merged endpoints, so among physical endpoints it ranks after them.
UNMERGED_SOURCE_KIND = "combined_unmerged_corrected_hkl"


def _publication_rank_key(candidate: Mapping) -> tuple:
    """Sort key for physical endpoints: merged endpoints before the unmerged batch endpoint, then ``_metric_key``."""

    return (str(candidate.get("source_kind", "")) == UNMERGED_SOURCE_KIND, *_metric_key(candidate))


def candidate_wght_convergence(candidate: Mapping) -> dict[str, object]:
    """Applied versus SHELXL-suggested WGHT of one endpoint (from its hash-verified LST when available)."""

    shelxl = candidate.get("shelxl") if isinstance(candidate.get("shelxl"), Mapping) else {}
    recorded = shelxl.get("wght_convergence") if isinstance(shelxl.get("wght_convergence"), Mapping) else {}
    applied = suggested = None
    lst_text = str(candidate.get("lst") or "").strip()
    if lst_text and candidate.get("artifact_provenance_ok") is True and Path(lst_text).is_file():
        from grond.refine.shelxl_diagnostics import parse_lst, parse_recommended_wght

        live = parse_lst(Path(lst_text))
        if live.get("active_wght_a") is not None:
            applied = [float(live["active_wght_a"]), float(live.get("active_wght_b", 0.0))]
        recommended = parse_recommended_wght(Path(lst_text))
        if recommended is not None:
            suggested = [float(recommended[0]), float(recommended[1])]
    if applied is None and shelxl.get("active_wght_a") is not None:
        applied = [_float(shelxl.get("active_wght_a")), _float(shelxl.get("active_wght_b"), 0.0)]
    if suggested is None and shelxl.get("recommended_wght_a") is not None:
        suggested = [_float(shelxl.get("recommended_wght_a")), _float(shelxl.get("recommended_wght_b"), 0.0)]
    from grond.refine.shelxl_iterations import WGHT_CONVERGE_TOL_A, WGHT_CONVERGE_TOL_B

    if applied is None or suggested is None:
        return {"status": "unknown", "applied": applied, "suggested": suggested, "recorded": dict(recorded)}
    agree = abs(suggested[0] - applied[0]) < WGHT_CONVERGE_TOL_A and abs(suggested[1] - applied[1]) < WGHT_CONVERGE_TOL_B
    return {
        "status": "converged" if agree else "not_converged",
        "applied": applied,
        "suggested": suggested,
        "recorded": dict(recorded),
    }


def _validated_combined_baseline(
    r1_all: object,
    source: object,
) -> tuple[float | None, str]:
    """Return a usable conventional combined baseline and its provenance."""

    value = _float(r1_all)
    source_text = str(source or "").strip()
    if not math.isfinite(value) or value < 0.0 or not source_text:
        return None, source_text
    return value, source_text


def _combined_baseline_gate(
    candidate: Mapping,
    *,
    baseline_r1_all: object = None,
    baseline_source: object = "",
    required: bool = False,
) -> dict:
    """Apply the conventional-baseline no-harm gate to one combined endpoint."""

    baseline, source = _validated_combined_baseline(baseline_r1_all, baseline_source)
    endpoint = _float(candidate.get("R1_all"))
    available = baseline is not None
    gate = {
        "status": "not_required_unavailable",
        "passed": True,
        "required": bool(required),
        "available": available,
        "baseline_R1_all": baseline,
        "baseline_source": source,
        "endpoint_R1_all": endpoint if math.isfinite(endpoint) else None,
        "delta_R1_all": None,
        "reasons": [],
        "warnings": [],
    }
    if not available:
        if required:
            warning = (
                "combined conventional-baseline comparison was requested but is unavailable; "
                "a validated finite baseline R1(F)all and non-empty source are needed for the "
                "comparison, but the corrected endpoint remains available for model handoff"
            )
            gate.update({"status": "warning_baseline_unavailable", "passed": True, "warnings": [warning]})
        return gate
    if not math.isfinite(endpoint):
        reason = "combined endpoint R1(F)all is unavailable, so it cannot pass the conventional-baseline no-harm gate"
        gate.update({"status": "failed_endpoint_unavailable", "passed": False, "reasons": [reason]})
        return gate

    delta = endpoint - baseline
    gate["delta_R1_all"] = delta
    if delta > 1e-12:
        warning = (
            f"combined endpoint R1(F)all {_percent(endpoint)} is worse than conventional baseline "
            f"{_percent(baseline)} by {100.0 * delta:.2f} percentage points "
            f"(source: {source}); warning only, keep the endpoint available for model inspection "
            "and continued refinement"
        )
        gate.update({"status": "warning_worse_than_baseline", "passed": True, "warnings": [warning]})
        return gate
    gate["status"] = "passed"
    return gate


def _candidate_id(label: str, endpoint: Mapping, index: int) -> str:
    """Build a stable human-readable endpoint candidate identifier."""

    lam = _float(endpoint.get("lambda_q"), float("nan"))
    endpoint_b = _float(endpoint.get("endpoint_b"), 0.0)
    exti = "exti" if bool(endpoint.get("keep_exti", False)) else "noexti"
    parts = [str(label)]
    if math.isfinite(lam) and abs(lam - 1.0) > 1e-9:
        parts.append(f"qexp={lam:g}")
    parts.extend([f"B={endpoint_b:g}", exti, f"candidate={index}"])
    return ":".join(parts)


def _artifact_provenance(
    name: str,
    path_records: list[tuple[str, object]],
    hash_records: list[tuple[str, object]],
) -> dict[str, object]:
    """Verify every recorded path/hash for one endpoint artifact."""

    recorded_paths = [(source, str(value).strip()) for source, value in path_records if str(value or "").strip()]
    recorded_hashes = [(source, str(value).strip().lower()) for source, value in hash_records if str(value or "").strip()]
    upper = name.upper()
    base: dict[str, object] = {
        "name": name,
        "passed": False,
        "recorded_paths": [{"source": source, "path": value} for source, value in recorded_paths],
        "recorded_hashes": [{"source": source, "sha256": value} for source, value in recorded_hashes],
        "resolved_path": "",
        "actual_sha256": "",
    }
    if not recorded_paths:
        return {**base, "status": f"missing_{name}_path", "reason": f"endpoint {upper} path is not recorded"}
    try:
        resolved = [(source, Path(value).expanduser().resolve()) for source, value in recorded_paths]
    except OSError as exc:
        return {**base, "status": f"{name}_path_error", "reason": f"could not resolve endpoint {upper} path: {exc}"}
    unique_paths = {path for _source, path in resolved}
    if len(unique_paths) != 1:
        return {
            **base,
            "status": f"conflicting_{name}_paths",
            "reason": f"recorded endpoint {upper} paths do not identify the same file",
        }
    path = next(iter(unique_paths))
    base["resolved_path"] = str(path)
    if not path.is_file():
        return {**base, "status": f"missing_{name}_file", "reason": f"endpoint {upper} file is missing: {path}"}
    if not recorded_hashes:
        return {**base, "status": f"missing_{name}_sha256", "reason": f"endpoint {upper} SHA-256 is not recorded"}
    try:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return {**base, "status": f"{name}_read_error", "reason": f"could not hash endpoint {upper} file: {exc}"}
    base["actual_sha256"] = actual_sha
    mismatches = [source for source, expected in recorded_hashes if expected != actual_sha]
    if mismatches:
        return {
            **base,
            "status": f"{name}_sha256_mismatch",
            "reason": f"endpoint {upper} SHA-256 does not match current bytes",
            "mismatched_sources": mismatches,
        }
    return {
        **base,
        "status": "ok",
        "passed": True,
        "reason": f"recorded {upper} path and SHA-256 match",
    }


def endpoint_artifact_provenance(endpoint: Mapping) -> dict[str, object]:
    """Verify exact RES/HKL/LST provenance and audit the hash-bound LST."""

    shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
    health = shelxl.get("res_health") if isinstance(shelxl.get("res_health"), Mapping) else {}
    artifacts = endpoint.get("artifacts") if isinstance(endpoint.get("artifacts"), Mapping) else {}
    manifest_path_text = str(endpoint.get("endpoint_manifest") or endpoint.get("manifest") or "").strip()
    manifest = _read_json(Path(manifest_path_text)) if manifest_path_text else {}
    manifest_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), Mapping) else {}

    path_records = {
        "res": [
            ("endpoint.res", endpoint.get("res")),
            ("endpoint.endpoint_res", endpoint.get("endpoint_res")),
            ("artifacts.res", artifacts.get("res")),
            ("shelxl.res_health.path", health.get("path")),
            ("manifest.artifacts.res", manifest_artifacts.get("res")),
        ],
        "hkl": [
            ("endpoint.hkl", endpoint.get("hkl")),
            ("endpoint.endpoint_hkl", endpoint.get("endpoint_hkl")),
            ("artifacts.hkl", artifacts.get("hkl")),
            ("manifest.artifacts.hkl", manifest_artifacts.get("hkl")),
        ],
        "lst": [
            ("endpoint.lst", endpoint.get("lst")),
            ("endpoint.endpoint_lst", endpoint.get("endpoint_lst")),
            ("artifacts.lst", artifacts.get("lst")),
            ("shelxl.lst", shelxl.get("lst")),
            ("shelxl.res_health.lst", health.get("lst")),
            ("manifest.artifacts.lst", manifest_artifacts.get("lst")),
        ],
    }
    hash_records = {
        "res": [
            ("endpoint.res_sha256", endpoint.get("res_sha256")),
            ("endpoint.endpoint_res_sha256", endpoint.get("endpoint_res_sha256")),
            ("artifacts.res_sha256", artifacts.get("res_sha256")),
            ("shelxl.res_health.sha256", health.get("sha256")),
            ("manifest.artifacts.res_sha256", manifest_artifacts.get("res_sha256")),
        ],
        "hkl": [
            ("endpoint.hkl_sha256", endpoint.get("hkl_sha256")),
            ("endpoint.endpoint_hkl_sha256", endpoint.get("endpoint_hkl_sha256")),
            ("artifacts.hkl_sha256", artifacts.get("hkl_sha256")),
            ("manifest.artifacts.hkl_sha256", manifest_artifacts.get("hkl_sha256")),
        ],
        "lst": [
            ("endpoint.lst_sha256", endpoint.get("lst_sha256")),
            ("endpoint.endpoint_lst_sha256", endpoint.get("endpoint_lst_sha256")),
            ("artifacts.lst_sha256", artifacts.get("lst_sha256")),
            ("manifest.artifacts.lst_sha256", manifest_artifacts.get("lst_sha256")),
        ],
    }
    audited = {
        name: _artifact_provenance(name, path_records[name], hash_records[name])
        for name in ("res", "hkl", "lst")
    }
    first_failure = next((audited[name] for name in ("res", "hkl", "lst") if audited[name].get("passed") is not True), None)
    passed = first_failure is None
    out: dict[str, object] = {
        "required": True,
        "passed": passed,
        "status": "ok" if passed else str(first_failure.get("status")),
        "reason": (
            "recorded RES, HKL, and LST paths and SHA-256 values match current bytes"
            if passed
            else str(first_failure.get("reason"))
        ),
        "manifest": manifest_path_text,
        "artifacts": audited,
        "canonical_artifacts": {
            name: {
                "path": str(audited[name].get("resolved_path") or ""),
                "sha256": str(audited[name].get("actual_sha256") or ""),
            }
            for name in ("res", "hkl", "lst")
        },
        # Compatibility fields retained for existing audit consumers.
        "resolved_res": str(audited["res"].get("resolved_path") or ""),
        "actual_sha256": str(audited["res"].get("actual_sha256") or ""),
    }
    if audited["lst"].get("passed") is True:
        lst_path = Path(str(audited["lst"]["resolved_path"]))
        try:
            out["lst_pathology"] = parse_shelx_lst_pathologies(lst_path)
            out["lst_metrics"] = parse_shelx_lst_metrics(lst_path)
        except OSError as exc:
            out["lst_pathology"] = {
                "lst": str(lst_path),
                "pathology_status": "read_error",
                "error": str(exc),
            }
            out["lst_metrics"] = {
                "lst": str(lst_path),
                "metric_status": "read_error",
                "error": str(exc),
            }
    else:
        out["lst_pathology"] = {
            "lst": str(audited["lst"].get("resolved_path") or ""),
            "pathology_status": "unverified",
        }
        out["lst_metrics"] = {
            "lst": str(audited["lst"].get("resolved_path") or ""),
            "metric_status": "unverified",
        }
    return out


def endpoint_lst_metric_consistency(recorded: Mapping, provenance: Mapping) -> dict[str, object]:
    """Compare stored endpoint statistics with the hash-verified LST values."""

    live = provenance.get("lst_metrics") if isinstance(provenance.get("lst_metrics"), Mapping) else {}
    base: dict[str, object] = {
        "required": True,
        "passed": False,
        "status": "unavailable",
        "recorded": {key: recorded.get(key) for key in ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")},
        "live": {key: live.get(key) for key in ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")},
        "mismatches": [],
    }
    if str(live.get("metric_status") or "") != "complete":
        return {
            **base,
            "status": "live_metrics_incomplete",
            "reason": "hash-verified endpoint LST does not contain complete final R1/wR2/GooF/N statistics",
        }
    mismatches: list[dict[str, object]] = []
    printed_decimals = live.get("printed_decimals") if isinstance(live.get("printed_decimals"), Mapping) else {}
    for key in ("R1_gt", "R1_all", "wR2_all", "GooF"):
        stored_value = _float(recorded.get(key))
        live_value = _float(live.get(key))
        decimals = _int(printed_decimals.get(key), 6)
        tolerance = 0.5 * (10.0 ** (-decimals)) + 1e-12
        if not math.isfinite(stored_value) or not math.isfinite(live_value) or abs(stored_value - live_value) > tolerance:
            mismatches.append(
                {
                    "field": key,
                    "recorded": recorded.get(key),
                    "live": live.get(key),
                    "absolute_tolerance": tolerance,
                }
            )
    for key in ("N_gt", "N_all"):
        try:
            stored_value = int(recorded.get(key))
            live_value = int(live.get(key))
        except (TypeError, ValueError, OverflowError):
            mismatches.append({"field": key, "recorded": recorded.get(key), "live": live.get(key), "exact": True})
            continue
        if stored_value != live_value:
            mismatches.append({"field": key, "recorded": stored_value, "live": live_value, "exact": True})
    if mismatches:
        return {
            **base,
            "status": "stored_live_mismatch",
            "reason": "stored endpoint statistics disagree with the hash-verified SHELXL LST",
            "mismatches": mismatches,
        }
    return {
        **base,
        "status": "ok",
        "passed": True,
        "reason": "stored endpoint statistics agree with the hash-verified SHELXL LST within print rounding",
    }


def _bool_value(value: object) -> bool | None:
    """Read a bool, 0/1 or yes/no/true/false/on/off text as a bool; None when it is none of these."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0"}:
        return False
    return None


def endpoint_promotion_field_contract(endpoint: Mapping) -> dict[str, object]:
    """Require promotion fields in the producer record; use manifest only to corroborate."""

    manifest_path_text = str(endpoint.get("endpoint_manifest") or endpoint.get("manifest") or "").strip()
    manifest = _read_json(Path(manifest_path_text)) if manifest_path_text else {}
    direct = {
        "hkl_safe_factor": endpoint.get("hkl_safe_factor"),
        "modulation_p95": endpoint.get("mod_p95", endpoint.get("modulation_p95")),
        "endpoint_b": endpoint.get("endpoint_b"),
        "keep_exti": endpoint.get("keep_exti") if "keep_exti" in endpoint else None,
    }
    manifest_values = {
        "hkl_safe_factor": manifest.get("hkl_safe_factor"),
        "modulation_p95": manifest.get("mod_p95", manifest.get("modulation_p95")),
        "endpoint_b": manifest.get("endpoint_b"),
        "keep_exti": manifest.get("keep_exti") if "keep_exti" in manifest else None,
    }
    errors: list[str] = []
    warnings: list[str] = []
    for key in ("hkl_safe_factor", "endpoint_b"):
        value = _float(direct.get(key))
        if not math.isfinite(value):
            errors.append(f"producer endpoint record is missing finite {key}")
            continue
        corroborating = manifest_values.get(key)
        if corroborating is not None:
            manifest_value = _float(corroborating)
            if not math.isfinite(manifest_value) or abs(value - manifest_value) > 1e-12:
                errors.append(f"producer endpoint record {key} disagrees with manifest")
    modulation = _float(direct.get("modulation_p95"))
    if not math.isfinite(modulation):
        warnings.append("producer endpoint record is missing finite modulation_p95")
    elif manifest_values.get("modulation_p95") is not None:
        manifest_modulation = _float(manifest_values.get("modulation_p95"))
        if not math.isfinite(manifest_modulation) or abs(modulation - manifest_modulation) > 1e-12:
            warnings.append("producer endpoint record modulation_p95 disagrees with manifest")
    direct_exti = _bool_value(direct.get("keep_exti"))
    if direct_exti is None:
        errors.append("producer endpoint record is missing boolean keep_exti")
    manifest_exti = _bool_value(manifest_values.get("keep_exti"))
    if manifest_values.get("keep_exti") is not None and (manifest_exti is None or manifest_exti != direct_exti):
        errors.append("producer endpoint record keep_exti disagrees with manifest")
    return {
        "required": True,
        "passed": not errors,
        "status": "ok" if not errors else "failed",
        "reason": "producer promotion fields are explicit and manifest-consistent" if not errors else "; ".join(errors),
        "direct": direct,
        "manifest": manifest_values,
        "manifest_path": manifest_path_text,
        "errors": errors,
        "warnings": warnings,
    }


def _normalize_endpoint(label: str, endpoint: Mapping, *, run_dir: Path | None, source_kind: str, index: int) -> dict:
    """Normalize one endpoint dictionary from a native or combined summary."""

    shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
    artifacts = endpoint.get("artifacts") if isinstance(endpoint.get("artifacts"), Mapping) else {}
    health = shelxl.get("res_health") if isinstance(shelxl.get("res_health"), Mapping) else {}
    lst_text = str(shelxl.get("lst") or artifacts.get("lst") or "")
    res_text = str(endpoint.get("res") or artifacts.get("res") or health.get("path") or "")
    if not res_text and lst_text:
        res_text = str(Path(lst_text).with_suffix(".res"))
    provenance = endpoint_artifact_provenance(endpoint)
    pathology = provenance.get("lst_pathology") if isinstance(provenance.get("lst_pathology"), Mapping) else {}
    pathology_complete = str(pathology.get("pathology_status") or "") == "complete"
    recorded_npd = shelxl.get("npd_count", endpoint.get("npd_count"))
    recorded_split = shelxl.get("split_count", endpoint.get("split_count"))
    recorded_atoms = list(shelxl.get("npd_atoms", endpoint.get("npd_atoms", [])) or [])
    recorded_has_npd = shelxl.get("has_npd", endpoint.get("has_npd"))
    recorded_has_split = shelxl.get("has_split_warnings", endpoint.get("has_split_warnings"))
    live_atoms = list(pathology.get("npd_atoms") or []) if pathology_complete else []
    npd_count = _int(recorded_npd) if recorded_npd is not None else None
    split_count = _int(recorded_split) if recorded_split is not None else None
    if pathology_complete:
        npd_count = max(npd_count or 0, _int(pathology.get("npd_count")))
        split_count = max(split_count or 0, _int(pathology.get("split_count")))
    npd_atoms = sorted({str(atom) for atom in [*recorded_atoms, *live_atoms] if str(atom).strip()})
    has_npd = (
        bool(recorded_has_npd) or bool(pathology.get("has_npd")) or bool((npd_count or 0) > 0)
        if recorded_has_npd is not None or pathology_complete or npd_count is not None
        else None
    )
    has_split = (
        bool(recorded_has_split) or bool(pathology.get("has_split_warnings")) or bool((split_count or 0) > 0)
        if recorded_has_split is not None or pathology_complete or split_count is not None
        else None
    )
    metric_consistency = endpoint_lst_metric_consistency(shelxl, provenance)
    live_metrics = provenance.get("lst_metrics") if isinstance(provenance.get("lst_metrics"), Mapping) else {}
    live_metrics_complete = str(live_metrics.get("metric_status") or "") == "complete"
    promotion_fields = endpoint_promotion_field_contract(endpoint)
    canonical = provenance.get("canonical_artifacts") if isinstance(provenance.get("canonical_artifacts"), Mapping) else {}
    canonical_res = canonical.get("res") if isinstance(canonical.get("res"), Mapping) else {}
    canonical_hkl = canonical.get("hkl") if isinstance(canonical.get("hkl"), Mapping) else {}
    canonical_lst = canonical.get("lst") if isinstance(canonical.get("lst"), Mapping) else {}
    endpoint_policy_complete = "endpoint_b" in endpoint and "keep_exti" in endpoint
    candidate = {
        "id": _candidate_id(label, endpoint, index),
        "label": label,
        "source_kind": source_kind,
        "run_dir": str(run_dir.resolve()) if run_dir else "",
        "lambda_q": _float(endpoint.get("lambda_q"), float("nan")),
        "endpoint_b": _float(endpoint.get("endpoint_b")),
        "keep_exti": bool(endpoint.get("keep_exti")) if "keep_exti" in endpoint else None,
        "endpoint_policy_required": True,
        "endpoint_policy_complete": endpoint_policy_complete,
        "hkl_safe_factor": _float(endpoint.get("hkl_safe_factor")),
        "modulation_p95": _float(endpoint.get("mod_p95", endpoint.get("modulation_p95"))),
        "modulation_metric": str(endpoint.get("modulation_metric") or ""),
        "R1_gt": _float(live_metrics.get("R1_gt") if live_metrics_complete else shelxl.get("R1_gt")),
        "R1_all": _float(live_metrics.get("R1_all") if live_metrics_complete else shelxl.get("R1_all")),
        "wR2_all": _float(live_metrics.get("wR2_all") if live_metrics_complete else shelxl.get("wR2_all")),
        "GooF": _float(live_metrics.get("GooF") if live_metrics_complete else shelxl.get("GooF")),
        "N_gt": live_metrics.get("N_gt") if live_metrics_complete else shelxl.get("N_gt"),
        "N_all": live_metrics.get("N_all") if live_metrics_complete else shelxl.get("N_all"),
        "metric_consistency_required": True,
        "metric_consistency_ok": metric_consistency.get("passed") is True,
        "metric_consistency": metric_consistency,
        "unique": shelxl.get("unique"),
        "pathology_required": True,
        "pathology_status": str(pathology.get("pathology_status") or "unverified"),
        "npd_count": npd_count,
        "split_count": split_count,
        "npd_atoms": npd_atoms,
        "has_npd": has_npd,
        "has_split_warnings": has_split,
        "hkl": str(canonical_hkl.get("path") or endpoint.get("hkl") or artifacts.get("hkl") or ""),
        "lst": str(canonical_lst.get("path") or lst_text),
        "res": str(canonical_res.get("path") or res_text),
        "res_sha256": str(canonical_res.get("sha256") or ""),
        "hkl_sha256": str(canonical_hkl.get("sha256") or ""),
        "lst_sha256": str(canonical_lst.get("sha256") or ""),
        "endpoint_manifest": str(endpoint.get("endpoint_manifest") or ""),
        "endpoint_row_table": str(endpoint.get("endpoint_row_table") or ""),
        "provenance_required": True,
        "artifact_provenance_ok": provenance.get("passed") is True,
        "artifact_provenance": provenance,
        "res_provenance_ok": provenance.get("passed") is True,
        "res_provenance": provenance,
        "promotion_fields_required": True,
        "promotion_fields_ok": promotion_fields.get("passed") is True,
        "promotion_field_contract": promotion_fields,
        "shelxl": dict(shelxl),
        "wght_mode": endpoint.get("wght_mode") or endpoint.get("endpoint_wght_mode") or shelxl.get("wght_mode"),
    }
    return candidate


def _collect_native_candidates(runs: Mapping[str, Path]) -> list[dict]:
    """Collect endpoint candidates from completed native run summaries."""

    candidates: list[dict] = []
    for label, run in runs.items():
        run_dir = Path(run)
        summary = _read_json(run_dir / "summary.json")
        endpoints = summary.get("endpoint_results_sorted") or []
        if not endpoints and summary.get("best_endpoint"):
            endpoints = [summary["best_endpoint"]]
        for index, endpoint in enumerate(endpoints, start=1):
            candidates.append(_normalize_endpoint(label, endpoint, run_dir=run_dir, source_kind="native_corrected_hkl", index=index))
    return candidates


def _collect_combined_candidate(combined_run: Path | None) -> list[dict]:
    """Collect the merged multi-dataset endpoint candidate and, when written, the unmerged batch endpoint."""

    if not combined_run:
        return []
    run_dir = Path(combined_run)
    summary = _read_json(run_dir / "combined_summary.json")
    endpoint = summary.get("combined_endpoint") or {}
    if not endpoint:
        return []
    candidates = [_normalize_endpoint("combined", endpoint, run_dir=run_dir, source_kind="combined_corrected_hkl", index=1)]
    unmerged = summary.get("combined_unmerged_endpoint") or {}
    if isinstance(unmerged, Mapping) and unmerged.get("shelxl"):
        candidate = _normalize_endpoint(
            "combined_unmerged", unmerged, run_dir=run_dir, source_kind=UNMERGED_SOURCE_KIND, index=1
        )
        candidate["basf"] = unmerged.get("basf")
        candidate["basf_sanity"] = unmerged.get("basf_sanity")
        candidate["batches"] = unmerged.get("batches")
        candidates.append(candidate)
    return candidates


def classify_publication_endpoint(
    candidate: Mapping,
    *,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    allow_npd: bool = False,
) -> dict:
    """Classify one endpoint as publishable physical output or diagnostic only."""

    safety = classify_endpoint_candidate(
        candidate,
        min_hkl_safe_factor=min_hkl_safe_factor,
        max_modulation_p95=max_modulation_p95,
    ).to_dict()
    reasons: list[str] = list(safety.get("reasons") or [])
    warnings: list[str] = list(safety.get("warnings") or [])
    hkl_path = str(candidate.get("hkl") or candidate.get("endpoint_hkl") or "").strip()
    hkl_precision = audit_fixed_width_precision(Path(hkl_path)) if hkl_path else {"status": "not_provided"}
    if hkl_precision["status"] == "invalid":
        reasons.append(f"written HKL validation failed: {hkl_precision['reason']}")
    elif hkl_precision["status"] == "ok" and hkl_precision["rounding_bound_sigma_units_max"] > 0.01:
        warnings.append(
            f"HKLF4 rounding can reach {hkl_precision['rounding_bound_sigma_units_max']:.4g} sigma(I); "
            "inspect the weakest observations"
        )
    endpoint_b = _float(candidate.get("endpoint_b"), 0.0)
    lambda_q = _float(candidate.get("lambda_q"), 1.0)
    if candidate.get("endpoint_policy_required") is True and candidate.get("endpoint_policy_complete") is not True:
        reasons.append(
            "endpoint correction policy is incomplete; the producer must explicitly record endpoint_b and keep_exti"
        )
    if abs(lambda_q - 1.0) > 1e-9:
        reasons.append("non-unit q exponent is diagnostic-only; GROND publication endpoints use the full Bloch/Born correction")
    if bool(candidate.get("keep_exti", False)):
        reasons.append("EXTI endpoint correction is comparison-only; GROND publication endpoints keep EXTI off")
    if candidate.get("endpoint_policy_complete") is True and not math.isfinite(_float(candidate.get("endpoint_b"))):
        reasons.append("endpoint-B is unavailable; the producer must record a finite endpoint_b")
    elif abs(endpoint_b) > 1e-12:
        reasons.append("nonzero endpoint-B is diagnostic-only because it reshapes corrected HKL intensities and can distort ADPs or maps")
    if candidate.get("provenance_required") is True and candidate.get("artifact_provenance_ok") is not True:
        provenance = (
            candidate.get("artifact_provenance")
            if isinstance(candidate.get("artifact_provenance"), Mapping)
            else {}
        )
        detail = str(provenance.get("reason") or provenance.get("status") or "missing endpoint artifact provenance")
        reasons.append(f"endpoint RES/HKL/LST provenance failed: {detail}")
    if candidate.get("pathology_required") is True and str(candidate.get("pathology_status") or "") != "complete":
        reasons.append(
            "endpoint LST pathology audit is missing or incomplete; publication endpoints require a hash-bound final "
            "SHELXL NPD/split summary"
        )
    if candidate.get("metric_consistency_required") is True and candidate.get("metric_consistency_ok") is not True:
        consistency = (
            candidate.get("metric_consistency")
            if isinstance(candidate.get("metric_consistency"), Mapping)
            else {}
        )
        detail = str(consistency.get("reason") or consistency.get("status") or "unverified endpoint statistics")
        reasons.append(f"endpoint stored/live SHELXL metric audit failed: {detail}")
    if candidate.get("promotion_fields_required") is True and candidate.get("promotion_fields_ok") is not True:
        contract = (
            candidate.get("promotion_field_contract")
            if isinstance(candidate.get("promotion_field_contract"), Mapping)
            else {}
        )
        detail = str(contract.get("reason") or contract.get("status") or "missing producer promotion fields")
        reasons.append(f"endpoint producer promotion-field contract failed: {detail}")
    promotion_contract = (
        candidate.get("promotion_field_contract")
        if isinstance(candidate.get("promotion_field_contract"), Mapping)
        else {}
    )
    warnings.extend(str(value) for value in promotion_contract.get("warnings", []) if str(value))
    npd_count = _int(candidate.get("npd_count", 0))
    if not allow_npd and (npd_count > 0 or bool(candidate.get("has_npd", False))):
        atoms = candidate.get("npd_atoms")
        atom_text = ""
        if isinstance(atoms, list) and atoms:
            atom_text = f" ({', '.join(str(atom) for atom in atoms[:6])}{', ...' if len(atoms) > 6 else ''})"
        reasons.append(f"endpoint has {npd_count} NPD atom(s){atom_text}; publication endpoints require NPD == 0")
    wght = candidate_wght_convergence(candidate)
    if wght["status"] == "not_converged":
        warnings.append(
            f"applied WGHT {wght['applied'][0]:.4f} {wght['applied'][1]:.4f} differs from SHELXL's suggested "
            f"{wght['suggested'][0]:.4f} {wght['suggested'][1]:.4f}; Olex2 flags this as an unconverged weighting scheme"
        )
    basf_sanity = candidate.get("basf_sanity") if isinstance(candidate.get("basf_sanity"), Mapping) else None
    if basf_sanity is not None and not basf_sanity.get("ok", False):
        warnings.append("batch scale factors (BASF) failed the sanity check: " + "; ".join(basf_sanity.get("problems") or []))
    goof = _float(candidate.get("GooF"))
    if not math.isfinite(goof):
        reasons.append("endpoint GooF is unavailable; publication endpoints must have conventional SHELXL GooF")
    elif goof < GOOF_WARNING_MIN or goof > GOOF_WARNING_MAX:
        warnings.append(
            f"endpoint GooF {goof:.3f} is outside the preferred range "
            f"[{GOOF_WARNING_MIN:.3f}, {GOOF_WARNING_MAX:.3f}]; inspect weights, sigmas, maps, and "
            "continued-refinement behavior, but keep the endpoint available for model building"
        )
    metrics_ok = math.isfinite(_float(candidate.get("wR2_all"))) and math.isfinite(_float(candidate.get("R1_all")))
    if not metrics_ok and "endpoint metrics are incomplete" not in reasons:
        reasons.append("endpoint metrics are incomplete")
    promotable = bool(metrics_ok and not reasons)
    return {
        "publication_class": "physical" if promotable else "diagnostic",
        "promotable": promotable,
        "rejection_reasons": reasons,
        "warnings": warnings,
        "safety": safety,
        "hkl_precision": hkl_precision,
        "wght_convergence": wght,
    }


def choose_publication_endpoints(
    candidates: list[dict],
    *,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    prefer_combined: bool = False,
    allow_npd: bool = False,
    combined_baseline_r1_all: float | None = None,
    combined_baseline_source: str = "",
    require_combined_baseline: bool = False,
    global_publication_blockers: list[str] | None = None,
) -> dict:
    """Choose best physical and diagnostic endpoints from normalized candidates."""

    blockers = [str(reason).strip() for reason in (global_publication_blockers or []) if str(reason).strip()]
    classified: list[dict] = []
    for candidate in candidates:
        decision = classify_publication_endpoint(
            candidate,
            min_hkl_safe_factor=min_hkl_safe_factor,
            max_modulation_p95=max_modulation_p95,
            allow_npd=allow_npd,
        )
        if blockers:
            reasons = list(decision.get("rejection_reasons") or [])
            reasons.extend(f"global publication blocker: {reason}" for reason in blockers)
            decision.update(
                {
                    "rejection_reasons": reasons,
                    "publication_class": "diagnostic",
                    "promotable": False,
                    "global_publication_blockers": blockers,
                }
            )
        if _is_combined_candidate(candidate):
            baseline_gate = _combined_baseline_gate(
                candidate,
                baseline_r1_all=combined_baseline_r1_all,
                baseline_source=combined_baseline_source,
                required=require_combined_baseline,
            )
            decision["combined_baseline_gate"] = baseline_gate
            decision["warnings"] = [
                *list(decision.get("warnings") or []),
                *(str(warning) for warning in baseline_gate.get("warnings", []) if str(warning)),
            ]
            if not baseline_gate["passed"]:
                reasons = list(decision.get("rejection_reasons") or [])
                reasons.extend(str(reason) for reason in baseline_gate.get("reasons", []))
                decision["rejection_reasons"] = reasons
                decision["publication_class"] = "diagnostic"
                decision["promotable"] = False
        classified.append({**candidate, **decision})

    physical = sorted([c for c in classified if c.get("promotable")], key=_publication_rank_key)
    combined_physical = [c for c in physical if _is_combined_candidate(c)]
    if prefer_combined:
        best_physical = combined_physical[0] if combined_physical else {}
    else:
        best_physical = physical[0] if physical else {}
    best_physical_id = best_physical.get("id")
    single_physical = [c for c in physical if not _is_combined_candidate(c)]
    best_single_physical = single_physical[0] if single_physical else {}
    rejected: list[dict] = []
    reported_candidates: list[dict] = []
    for candidate in classified:
        if candidate.get("id") == best_physical_id:
            reported_candidates.append(candidate)
            continue
        candidate = dict(candidate)
        if candidate.get("promotable"):
            if prefer_combined and not _is_combined_candidate(candidate):
                candidate["intrinsic_promotable"] = True
                candidate["promotable"] = False
                candidate["publication_class"] = "single_dataset_diagnostic"
                candidate["rejection_reasons"] = [
                    "single-dataset endpoint is physically valid for its crystal but cannot substitute for the required multi-dataset combined publication endpoint"
                ]
            else:
                candidate["publication_class"] = "physical_not_selected"
                candidate["rejection_reasons"] = ["physically promotable, but worse than the selected physical endpoint by the endpoint ranking metric"]
        rejected.append(candidate)
        reported_candidates.append(candidate)

    diagnostic = sorted([c for c in rejected if c.get("publication_class") != "physical_not_selected"], key=_metric_key)
    single_diagnostic = sorted(
        [c for c in diagnostic if not _is_combined_candidate(c)],
        key=_metric_key,
    )
    best_single_diagnostic = single_diagnostic[0] if single_diagnostic else {}
    if prefer_combined and not best_physical and best_single_diagnostic:
        best_diagnostic = best_single_diagnostic
    else:
        best_diagnostic = diagnostic[0] if diagnostic else {}
    verdict = "ok" if best_physical else ("diagnostic_only" if diagnostic else "no_endpoint_candidates")
    baseline, baseline_source = _validated_combined_baseline(
        combined_baseline_r1_all,
        combined_baseline_source,
    )
    baseline_policy = {
        "required": bool(require_combined_baseline),
        "available": baseline is not None,
        "R1_all": baseline,
        "source": baseline_source,
    }
    baseline_requirement = (
        "combined endpoints are compared with the validated conventional combined-baseline R1(F)all; "
        "a missing comparison or worse value is reported as a warning but does not withhold an otherwise usable model"
        if require_combined_baseline
        else "when supplied, the conventional combined-baseline R1(F)all is a warning-only comparison"
    )
    return {
        "status": verdict,
        "best_physical_endpoint": best_physical,
        "best_single_dataset_physical_endpoint": best_single_physical,
        "best_single_dataset_diagnostic_endpoint": best_single_diagnostic,
        "best_diagnostic_endpoint": best_diagnostic,
        "global_publication_blockers": blockers,
        "rejected_corrections": sorted(rejected, key=_metric_key),
        "candidates": sorted(reported_candidates, key=_metric_key),
        "policy": {
            "physical_endpoint_requires": [
                "finite endpoint R factors",
                "q exponent equal to one",
                "EXTI off",
                "endpoint-B equal to zero",
                "finite positive HKL export scale; written HKLF4 values are finite with positive sigmas",
                f"HKL rescaling below {min_hkl_safe_factor:g} warns but cannot by itself reject a model",
                f"95th percentile modulation is reported and warned above {max_modulation_p95:g}, but never blocks promotion",
                f"SHELXL GooF is finite; values outside {GOOF_WARNING_MIN:g}-{GOOF_WARNING_MAX:g} are warned but do not block model handoff",
                "hash-bound final LST pathology audit is complete and NPD == 0; split warnings are reported but do not block promotion",
                "stored endpoint statistics agree with hash-verified live LST statistics within SHELXL print rounding",
                "collected endpoint RES, HKL, and LST paths and SHA-256 values must match the current files",
                "safety, endpoint-B, and EXTI fields are explicit in the producer endpoint record and manifest-consistent; modulation provenance mismatches are warnings",
                "no unresolved run-level publication blocker",
                baseline_requirement,
            ],
            "ranking": (
                "multi-dataset runs require a passing combined endpoint and never substitute a single-dataset endpoint; "
                "within the eligible class merged endpoints rank before the unmerged batch endpoint (its R1 is over "
                "unmerged rows), then by endpoint R1(F)all, R1(F)gt, and wR2(F^2)all"
                if prefer_combined
                else "rank endpoint R1(F)all, R1(F)gt, and wR2(F^2)all"
            ),
            "allow_npd_override": bool(allow_npd),
            "combined_baseline": baseline_policy,
        },
    }


def _candidate_line(candidate: Mapping) -> str:
    """Render one endpoint candidate row for Markdown."""

    reasons = "; ".join(str(r) for r in candidate.get("rejection_reasons", [])[:3])
    warnings = "; ".join(str(r) for r in candidate.get("warnings", [])[:3])
    goof = _float(candidate.get("GooF"))
    goof_text = "n/a" if not math.isfinite(goof) else f"{goof:.3f}"
    pathology_status = str(candidate.get("pathology_status") or "")
    if candidate.get("pathology_required") is True and pathology_status != "complete":
        pathology = f"unverified ({pathology_status or 'missing'})"
    else:
        pathology = f"NPD {_int(candidate.get('npd_count', 0))}; split {_int(candidate.get('split_count', 0))}"
    baseline_gate = candidate.get("combined_baseline_gate")
    baseline_gate = baseline_gate if isinstance(baseline_gate, Mapping) else {}
    baseline_gate_text = str(baseline_gate.get("status") or "n/a")
    modulation = _float(candidate.get("modulation_p95"))
    modulation_text = "n/a" if not math.isfinite(modulation) else f"{modulation:.3g}x"
    return (
        f"| {candidate.get('label', '')} | {candidate.get('source_kind', '')} | "
        f"{_float(candidate.get('endpoint_b'), 0.0):.3g} | "
        f"{'on' if candidate.get('keep_exti') else 'off'} | {candidate.get('publication_class', '')} | "
        f"{_percent(candidate.get('R1_gt'))} | {_percent(candidate.get('R1_all'))} | {_percent(candidate.get('wR2_all'))} | "
        f"{goof_text} | {modulation_text} | {pathology} | "
        f"{baseline_gate_text} | {warnings} | {reasons} |"
    )


def _write_markdown(out_dir: Path, decision: Mapping) -> Path:
    """Write the human-readable endpoint chooser report."""

    best_physical = decision.get("best_physical_endpoint") or {}
    best_single_physical = decision.get("best_single_dataset_physical_endpoint") or {}
    best_single_diagnostic = decision.get("best_single_dataset_diagnostic_endpoint") or {}
    best_diagnostic = decision.get("best_diagnostic_endpoint") or {}
    policy = decision.get("policy") if isinstance(decision.get("policy"), Mapping) else {}
    baseline_policy = policy.get("combined_baseline") if isinstance(policy.get("combined_baseline"), Mapping) else {}
    global_blockers = [str(reason) for reason in (decision.get("global_publication_blockers") or [])]
    lines = [
        "# Publication Endpoint Chooser",
        "",
        "This report separates the endpoint suitable for publication from endpoints that are useful diagnostics but rely on corrections GROND does not promote by default.",
        "",
        "## Verdict",
        "",
    ]
    if best_physical:
        lines.append(
            f"Best physical endpoint: `{best_physical.get('label') or best_physical.get('id')}` with `R1(F)all={_percent(best_physical.get('R1_all'))}` and `wR2(F^2)all={_percent(best_physical.get('wR2_all'))}`."
        )
        lines.extend(["", "Selected files:", ""])
        for kind in ("res", "hkl", "lst"):
            if best_physical.get(kind):
                lines.append(f"- {kind.upper()}: `{best_physical[kind]}`")
        lines.extend(["", "Candidate IDs identify report entries, not filenames or directories.", ""])
    else:
        lines.append("No physical endpoint passed the publication gate. The run should be reported as diagnostic-only until a no-EXTI, B=0 endpoint with finite metrics is available.")
    if global_blockers:
        lines.append("Run-level publication blocker(s): " + "; ".join(global_blockers) + ".")
    if best_diagnostic:
        lines.append(
            f"Best diagnostic endpoint: `{best_diagnostic.get('id')}` with `R1(F)all={_percent(best_diagnostic.get('R1_all'))}` and `wR2(F^2)all={_percent(best_diagnostic.get('wR2_all'))}`."
        )
    else:
        lines.append("No rejected diagnostic endpoint beat or supplemented the physical candidate set.")
    if best_single_physical and best_single_physical.get("id") != best_physical.get("id"):
        lines.append(
            f"Best single-dataset physical diagnostic: `{best_single_physical.get('id')}` with `R1(F)all={_percent(best_single_physical.get('R1_all'))}` and `wR2(F^2)all={_percent(best_single_physical.get('wR2_all'))}`."
        )
    if best_single_diagnostic and best_single_diagnostic.get("id") != best_single_physical.get("id"):
        lines.append(
            f"Best single-dataset diagnostic: `{best_single_diagnostic.get('id')}` with `R1(F)all={_percent(best_single_diagnostic.get('R1_all'))}` and `wR2(F^2)all={_percent(best_single_diagnostic.get('wR2_all'))}`."
        )
    if baseline_policy:
        availability = "available" if baseline_policy.get("available") else "unavailable"
        requirement = "required" if baseline_policy.get("required") else "optional"
        lines.append(
            "Combined conventional-baseline gate: "
            f"`{requirement}; {availability}`; R1(F)all `{_percent(baseline_policy.get('R1_all'))}`; "
            f"source `{baseline_policy.get('source', '')}`."
        )
    lines.extend(
        [
            "",
            "Promotion rule: finite endpoint metrics, full q correction, EXTI off, endpoint-B equal to zero, exact current RES/HKL/LST paths and SHA-256 values, a complete hash-bound final LST pathology audit, no NPD pathology unless explicitly overridden, and safe HKL rescaling. SHELXL GooF outside the preferred range, large modulation p95, a missing conventional comparison, and an endpoint R1all worse than the conventional baseline are reported as warnings, not mathematical-validity or model-handoff gates. Multi-dataset runs never substitute a single-dataset endpoint for a failed combined physical endpoint. Split warnings are reported but do not block promotion.",
            "",
            "## Endpoint Candidates",
            "",
            "| label | source | endpoint B | EXTI | class | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | modulation p95 | pathology | combined baseline gate | warnings | main rejection reason |",
            "|---|---|---:|:---:|---|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for candidate in decision.get("candidates", []):
        lines.append(_candidate_line(candidate))
    lines.extend(["", "## Rejected Corrections", ""])
    rejected = decision.get("rejected_corrections") or []
    if rejected:
        for candidate in rejected:
            reasons = "; ".join(str(r) for r in candidate.get("rejection_reasons", [])) or "not selected"
            lines.append(f"- `{candidate.get('id')}`: {reasons}.")
    else:
        lines.append("No endpoint corrections were rejected.")
    path = out_dir / "PUBLICATION_ENDPOINT_CHOOSER.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_tex(out_dir: Path, decision: Mapping) -> Path:
    """Write the endpoint candidates as a small LaTeX table."""

    rows = []
    for candidate in decision.get("candidates", []):
        rows.append(
            " & ".join(
                [
                    str(candidate.get("label", "")).replace("_", r"\_"),
                    str(candidate.get("source_kind", "")).replace("_", r"\_"),
                    f"{_float(candidate.get('endpoint_b'), 0.0):.3g}",
                    "on" if candidate.get("keep_exti") else "off",
                    str(candidate.get("publication_class", "")).replace("_", r"\_"),
                    _percent(candidate.get("R1_all")).replace("%", r"\%"),
                    _percent(candidate.get("wR2_all")).replace("%", r"\%"),
                ]
            )
            + r" \\"
        )
    text = "\n".join(
        [
            r"\begin{tabular}{llrllll}",
            r"\hline",
            r"Label & Source & B & EXTI & Class & R1(F)all & wR2(F^2)all \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    path = out_dir / "publication_endpoint_chooser.tex"
    path.write_text(text)
    return path


def write_publication_endpoint_choice(
    runs: Mapping[str, Path],
    out_dir: Path,
    *,
    combined_run: Path | None = None,
    extra_candidates: list[dict] | None = None,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    allow_npd: bool = False,
    combined_baseline_r1_all: float | None = None,
    combined_baseline_source: str = "",
    require_combined_baseline: bool = False,
    global_publication_blockers: list[str] | None = None,
) -> dict:
    """Collect endpoint candidates, choose publication endpoints, and write reports."""

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    candidates.extend(_collect_native_candidates(runs))
    candidates.extend(_collect_combined_candidate(combined_run))
    candidates.extend(extra_candidates or [])
    decision = choose_publication_endpoints(
        candidates,
        min_hkl_safe_factor=min_hkl_safe_factor,
        max_modulation_p95=max_modulation_p95,
        prefer_combined=combined_run is not None,
        allow_npd=allow_npd,
        combined_baseline_r1_all=combined_baseline_r1_all,
        combined_baseline_source=combined_baseline_source,
        require_combined_baseline=require_combined_baseline,
        global_publication_blockers=global_publication_blockers,
    )
    json_path = out_dir / "publication_endpoint_choice.json"
    md_path = _write_markdown(out_dir, decision)
    tex_path = _write_tex(out_dir, decision)
    artifacts = {
        **decision,
        "artifacts": {
            "json": str(json_path.resolve()),
            "markdown": str(md_path.resolve()),
            "latex": str(tex_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(artifacts, indent=2, allow_nan=True) + "\n")
    return artifacts
