"""Describe candidate endpoint files and the exact model and statistics that produced them."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.reports.shelxl import _quick_shelxl_details
from grond.reports.values import _quick_fraction, _quick_int
from grond.workflows.decision import safe_float
from grond.workflows.endpoint_chooser import (
    endpoint_artifact_provenance,
    endpoint_lst_metric_consistency,
    endpoint_promotion_field_contract,
)


def _publication_model_endpoint_candidate(result: Mapping[str, object]) -> dict[str, object]:
    """Describe the adopted corrected-HKL model, including an earlier safe cycle."""

    model_text = str(result.get("model", "") or "").strip()
    if not model_text:
        return {}
    res = Path(model_text).expanduser()
    lst = res.with_suffix(".lst")
    hkl = res.with_suffix(".hkl")
    if not res.exists() or not lst.exists():
        return {}
    origin_summary_path = res.parent.parent / ("combined_summary.json" if result.get("combined") else "summary.json")
    origin_summary = try_read_json(origin_summary_path)
    origin_endpoint: Mapping[str, object] = {}
    if result.get("combined"):
        value = origin_summary.get("combined_endpoint")
        origin_endpoint = value if isinstance(value, Mapping) else {}
    else:
        raw_endpoints = origin_summary.get("endpoint_results_sorted")
        candidates = raw_endpoints if isinstance(raw_endpoints, list) else []
        best = origin_summary.get("best_endpoint")
        if isinstance(best, Mapping):
            candidates = [best, *candidates]
        for value in candidates:
            if not isinstance(value, Mapping):
                continue
            value_shelxl = value.get("shelxl") if isinstance(value.get("shelxl"), Mapping) else {}
            value_lst = str(value_shelxl.get("lst") or "").strip()
            if value_lst and Path(value_lst).expanduser().with_suffix(".res").resolve() == res.resolve():
                origin_endpoint = value
                break
    origin_shelxl = origin_endpoint.get("shelxl") if isinstance(origin_endpoint.get("shelxl"), Mapping) else {}
    artifact_provenance = endpoint_artifact_provenance(
        {**origin_endpoint, "res": str(res), "hkl": str(hkl), "lst": str(lst)}
    )
    provenance_ok = artifact_provenance.get("passed") is True
    pathology = (
        artifact_provenance.get("lst_pathology")
        if isinstance(artifact_provenance.get("lst_pathology"), Mapping)
        else {}
    )
    pathology_complete = str(pathology.get("pathology_status") or "") == "complete"
    metric_consistency = endpoint_lst_metric_consistency(origin_shelxl, artifact_provenance)
    live_metrics = (
        artifact_provenance.get("lst_metrics") if isinstance(artifact_provenance.get("lst_metrics"), Mapping) else {}
    )
    live_metrics_complete = str(live_metrics.get("metric_status") or "") == "complete"
    origin_details = dict(origin_shelxl)
    for key in (
        "active_wght_a",
        "active_wght_b",
        "recommended_wght_a",
        "recommended_wght_b",
    ):
        if origin_details.get(key) is None:
            origin_details[key] = origin_endpoint.get(key, origin_endpoint.get(f"shelxl_{key}"))
    details = _quick_shelxl_details(origin_details, lst) if provenance_ok else origin_details
    if live_metrics_complete:
        details.update({key: live_metrics[key] for key in ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")})
    promotion_fields = endpoint_promotion_field_contract(origin_endpoint)
    canonical = (
        artifact_provenance.get("canonical_artifacts")
        if isinstance(artifact_provenance.get("canonical_artifacts"), Mapping)
        else {}
    )
    canonical_res = canonical.get("res") if isinstance(canonical.get("res"), Mapping) else {}
    canonical_hkl = canonical.get("hkl") if isinstance(canonical.get("hkl"), Mapping) else {}
    canonical_lst = canonical.get("lst") if isinstance(canonical.get("lst"), Mapping) else {}
    if _quick_fraction(details.get("R1_all")) is None or _quick_fraction(details.get("wR2_all")) is None:
        return {}
    recorded_npd = _quick_int(details.get("npd_count"))
    recorded_split = _quick_int(details.get("split_count"))
    npd_count = max(recorded_npd or 0, int(pathology.get("npd_count", 0) or 0)) if pathology_complete else recorded_npd
    split_count = (
        max(recorded_split or 0, int(pathology.get("split_count", 0) or 0)) if pathology_complete else recorded_split
    )
    recorded_atoms = details.get("npd_atoms") if isinstance(details.get("npd_atoms"), (list, tuple, set)) else []
    npd_atoms = sorted(
        {
            str(atom)
            for atom in [*recorded_atoms, *((pathology.get("npd_atoms") or []) if pathology_complete else [])]
            if str(atom).strip()
        }
    )
    has_npd = (
        bool(details.get("has_npd")) or bool(pathology.get("has_npd")) or bool((npd_count or 0) > 0)
        if details.get("has_npd") is not None or pathology_complete or npd_count is not None
        else None
    )
    has_split = (
        bool(details.get("has_split_warnings"))
        or bool(pathology.get("has_split_warnings"))
        or bool((split_count or 0) > 0)
        if details.get("has_split_warnings") is not None or pathology_complete or split_count is not None
        else None
    )
    endpoint_policy_complete = "endpoint_b" in origin_endpoint and "keep_exti" in origin_endpoint
    shelxl = {
        "R1_gt": details.get("R1_gt"),
        "R1_all": details.get("R1_all"),
        "wR2_all": details.get("wR2_all"),
        "GooF": details.get("GooF"),
        "N_gt": details.get("N_gt"),
        "N_all": details.get("N_all"),
        "Rint": details.get("Rint"),
        "Rsigma": details.get("Rsigma"),
        "parameters": details.get("parameters"),
        "restraints": details.get("restraints"),
        "npd_count": npd_count,
        "split_count": split_count,
        "npd_atoms": npd_atoms,
        "has_npd": has_npd,
        "has_split_warnings": has_split,
        "pathology_status": str(pathology.get("pathology_status") or "unverified"),
        "active_wght_a": details.get("active_wght_a"),
        "active_wght_b": details.get("active_wght_b"),
        "recommended_wght_a": details.get("recommended_wght_a"),
        "recommended_wght_b": details.get("recommended_wght_b"),
        "lst": str(lst),
    }
    source_kind = "combined_corrected_hkl" if result.get("combined") else "native_corrected_hkl"
    return {
        # Keep the saved candidate ID stable; display the actual model filename.
        "id": "final_ls1_corrected_hkl",
        "label": res.name,
        "source_kind": source_kind,
        "run_dir": str(res.parent),
        "lambda_q": 1.0,
        "endpoint_b": safe_float(origin_endpoint.get("endpoint_b"), float("nan")),
        "keep_exti": bool(origin_endpoint.get("keep_exti")) if "keep_exti" in origin_endpoint else None,
        "endpoint_policy_required": True,
        "endpoint_policy_complete": endpoint_policy_complete,
        "hkl_safe_factor": (safe_float(origin_endpoint.get("hkl_safe_factor"), 0.0) if provenance_ok else 0.0),
        "modulation_p95": (
            safe_float(origin_endpoint.get("mod_p95", origin_endpoint.get("modulation_p95")), float("nan"))
            if provenance_ok
            else float("inf")
        ),
        "R1_gt": details.get("R1_gt"),
        "R1_all": details.get("R1_all"),
        "wR2_all": details.get("wR2_all"),
        "GooF": details.get("GooF"),
        "N_gt": details.get("N_gt"),
        "N_all": details.get("N_all"),
        "metric_consistency_required": True,
        "metric_consistency_ok": metric_consistency.get("passed") is True,
        "metric_consistency": metric_consistency,
        "npd_count": npd_count,
        "split_count": split_count,
        "npd_atoms": npd_atoms,
        "has_npd": has_npd,
        "has_split_warnings": has_split,
        "hkl": str(canonical_hkl.get("path") or hkl),
        "lst": str(canonical_lst.get("path") or lst),
        "res": str(canonical_res.get("path") or res),
        "res_sha256": str(canonical_res.get("sha256") or ""),
        "hkl_sha256": str(canonical_hkl.get("sha256") or ""),
        "lst_sha256": str(canonical_lst.get("sha256") or ""),
        "shelxl": shelxl,
        "endpoint_origin_summary": str(origin_summary_path),
        "endpoint_row_table": str(origin_summary_path.parent / "native_dyn_calc.csv") if not result.get("combined") else "",
        "artifact_provenance_ok": provenance_ok,
        "artifact_provenance": artifact_provenance,
        "res_provenance_ok": provenance_ok,
        "provenance_required": True,
        "res_provenance": artifact_provenance,
        "pathology_required": True,
        "pathology_status": str(pathology.get("pathology_status") or "unverified"),
        "promotion_fields_required": True,
        "promotion_fields_ok": promotion_fields.get("passed") is True,
        "promotion_field_contract": promotion_fields,
        # combined summaries record the policy as endpoint_wght_mode and the run's own as shelxl.wght_mode
        "wght_mode": origin_endpoint.get("wght_mode")
        or origin_endpoint.get("endpoint_wght_mode")
        or origin_shelxl.get("wght_mode"),
        "wght_convergence": origin_shelxl.get("wght_convergence"),
    }


def _multi_feedback_endpoint_candidates(
    result: Mapping[str, object],
    run_root: Path,
    policy: str = "all",
) -> list[dict[str, object]]:
    """Expose shared SHELXL feedback cycles to the endpoint chooser.

    ``kept`` (AUTO) offers only the cycle the loop kept (``kept_cycle``), the model that every row table, held-out
    number and map of the run describes.  Ranking the other cycles by in-sample SHELXL R1 compares refinements
    against differently corrected data and can publish an earlier, less refined model (for example cycle 2
    chosen while the loop kept cycle 5).  Under ``all`` every adopted cycle is a candidate and the ordinary
    publication gates and ranking decide.
    """

    if not result.get("combined"):
        return []
    if policy not in {"kept", "all"}:
        raise ValueError(f"unknown --publication-feedback-cycles {policy!r}")
    summary_path = run_root / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    summary = try_read_json(summary_path)
    raw_cycles = summary.get("cycles") if isinstance(summary.get("cycles"), list) else []
    kept_cycle = _quick_int(summary.get("kept_cycle"))
    candidates: list[dict[str, object]] = []
    for raw_cycle in raw_cycles:
        if not isinstance(raw_cycle, Mapping) or not bool(raw_cycle.get("adopted_for_feedback", False)):
            continue
        if policy == "kept" and _quick_int(raw_cycle.get("cycle")) != kept_cycle:
            continue
        model_text = str(raw_cycle.get("output_model") or "").strip()
        combined_text = str(raw_cycle.get("combined") or "").strip()
        if not model_text or not combined_text:
            continue
        candidate = _publication_model_endpoint_candidate(
            {
                **result,
                "model": model_text,
                "combined": combined_text,
            }
        )
        if not candidate:
            continue
        cycle_number = _quick_int(raw_cycle.get("cycle"))
        cycle_label = (
            f"shared_ls1_cycle_{cycle_number:02d}"
            if cycle_number is not None
            else str(raw_cycle.get("label") or "shared_ls1_cycle")
        )
        modulation_p95 = safe_float(raw_cycle.get("endpoint_modulation_p95"), float("nan"))
        candidate.update(
            {
                "id": cycle_label,
                "label": cycle_label,
                "source_kind": "combined_feedback_cycle",
                "feedback_cycle": cycle_number,
                "feedback_summary": str(summary_path),
            }
        )
        if math.isfinite(modulation_p95):
            candidate["modulation_p95"] = modulation_p95
            candidate["modulation_metric"] = str(raw_cycle.get("endpoint_modulation_metric") or "")
        candidates.append(candidate)
    return candidates
