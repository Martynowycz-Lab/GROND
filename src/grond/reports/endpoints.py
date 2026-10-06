"""Normalize selected endpoint records and resolve the files shown in final reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from grond.reports.shelxl import _quick_shelxl_details
from grond.reports.values import _quick_fraction, _quick_int
from grond.workflows.endpoint_chooser import (
    endpoint_artifact_provenance,
    endpoint_lst_metric_consistency,
    endpoint_promotion_field_contract,
)


def _normalize_publication_endpoint_block(endpoint: Mapping[str, object], *, selection_source: str) -> dict[str, object]:
    """Normalize endpoint-chooser and endpoint-calibration records to one shape."""

    shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
    artifact_provenance = endpoint_artifact_provenance(endpoint)
    provenance_ok = artifact_provenance.get("passed") is True
    pathology = (
        artifact_provenance.get("lst_pathology")
        if isinstance(artifact_provenance.get("lst_pathology"), Mapping)
        else {}
    )
    pathology_complete = str(pathology.get("pathology_status") or "") == "complete"
    promotion_fields = endpoint_promotion_field_contract(endpoint)
    canonical = (
        artifact_provenance.get("canonical_artifacts")
        if isinstance(artifact_provenance.get("canonical_artifacts"), Mapping)
        else {}
    )
    canonical_res = canonical.get("res") if isinstance(canonical.get("res"), Mapping) else {}
    canonical_hkl = canonical.get("hkl") if isinstance(canonical.get("hkl"), Mapping) else {}
    canonical_lst = canonical.get("lst") if isinstance(canonical.get("lst"), Mapping) else {}

    def first_text(*keys: str) -> str:
        """Return the first non-empty endpoint field among keys as text, or an empty string."""

        for key in keys:
            value = endpoint.get(key)
            if value:
                return str(value)
        return ""

    def first_number(*keys: str) -> object:
        """Return the first non-None value among keys, in the endpoint and then its SHELXL block; None if none."""

        for key in keys:
            if key in endpoint and endpoint.get(key) is not None:
                return endpoint.get(key)
            if key in shelxl and shelxl.get(key) is not None:
                return shelxl.get(key)
        return None

    out = dict(endpoint)
    out["selection_source"] = selection_source
    out["source_kind"] = str(endpoint.get("source_kind") or ("endpoint_calibration" if selection_source == "endpoint_calibration" else ""))
    out["endpoint_res"] = str(canonical_res.get("path") or first_text("endpoint_res", "res"))
    out["endpoint_hkl"] = str(canonical_hkl.get("path") or first_text("endpoint_hkl", "hkl"))
    out["endpoint_lst"] = str(
        canonical_lst.get("path") or first_text("endpoint_lst", "lst") or shelxl.get("lst", "") or ""
    )
    out["endpoint_ins"] = first_text("endpoint_ins", "ins")
    if not out["endpoint_ins"] and out["endpoint_res"]:
        out["endpoint_ins"] = str(Path(out["endpoint_res"]).with_suffix(".ins"))
    if not out["endpoint_hkl"] and out["endpoint_res"]:
        endpoint_hkl = Path(out["endpoint_res"]).with_suffix(".hkl")
        if endpoint_hkl.exists():
            out["endpoint_hkl"] = str(endpoint_hkl)
    out["endpoint_manifest"] = first_text("endpoint_manifest", "manifest")
    if not out["endpoint_manifest"] and out["endpoint_res"]:
        manifest = Path(out["endpoint_res"]).with_name("manifest.json")
        if manifest.exists():
            out["endpoint_manifest"] = str(manifest)
    out["endpoint_row_table"] = first_text("endpoint_row_table", "row_table")
    if not out["endpoint_row_table"] and str(out.get("source_kind", "")) == "observation_polish":
        run_dir = first_text("run_dir")
        if run_dir:
            out["endpoint_row_table"] = str(Path(run_dir) / "promoted_observation_rows.csv")
    out["spec"] = str(endpoint.get("spec") or endpoint.get("scale_spec") or endpoint.get("source_kind") or endpoint.get("label") or "")
    out["shelxl_R1gt"] = first_number("shelxl_R1gt", "R1_gt")
    out["shelxl_R1all"] = first_number("shelxl_R1all", "R1_all")
    out["shelxl_wR2all"] = first_number("shelxl_wR2all", "wR2_all")
    out["shelxl_GooF"] = first_number("shelxl_GooF", "GooF")
    out["shelxl_Ngt"] = first_number("shelxl_Ngt", "N_gt")
    out["shelxl_Nall"] = first_number("shelxl_Nall", "N_all")
    recorded_metrics = {
        "R1_gt": out["shelxl_R1gt"],
        "R1_all": out["shelxl_R1all"],
        "wR2_all": out["shelxl_wR2all"],
        "GooF": out["shelxl_GooF"],
        "N_gt": out["shelxl_Ngt"],
        "N_all": out["shelxl_Nall"],
    }
    metric_consistency = endpoint_lst_metric_consistency(recorded_metrics, artifact_provenance)
    live_metrics = (
        artifact_provenance.get("lst_metrics")
        if isinstance(artifact_provenance.get("lst_metrics"), Mapping)
        else {}
    )
    if str(live_metrics.get("metric_status") or "") == "complete":
        out.update(
            {
                "shelxl_R1gt": live_metrics.get("R1_gt"),
                "shelxl_R1all": live_metrics.get("R1_all"),
                "shelxl_wR2all": live_metrics.get("wR2_all"),
                "shelxl_GooF": live_metrics.get("GooF"),
                "shelxl_Ngt": live_metrics.get("N_gt"),
                "shelxl_Nall": live_metrics.get("N_all"),
            }
        )
    shelxl_details = (
        _quick_shelxl_details(shelxl, Path(out["endpoint_lst"]) if out["endpoint_lst"] else None)
        if provenance_ok
        else dict(shelxl)
    )
    fallback_pairs = {
        "shelxl_R1gt": "R1_gt",
        "shelxl_R1all": "R1_all",
        "shelxl_wR2all": "wR2_all",
        "shelxl_GooF": "GooF",
        "shelxl_Ngt": "N_gt",
        "shelxl_Nall": "N_all",
        "shelxl_Rint": "Rint",
        "shelxl_Rsigma": "Rsigma",
        "shelxl_parameters": "parameters",
        "shelxl_restraints": "restraints",
        "shelxl_npd_count": "npd_count",
        "shelxl_npd_atoms": "npd_atoms",
        "shelxl_split_count": "split_count",
        "shelxl_has_npd": "has_npd",
        "shelxl_has_split_warnings": "has_split_warnings",
        "shelxl_active_wght_a": "active_wght_a",
        "shelxl_active_wght_b": "active_wght_b",
        "shelxl_recommended_wght_a": "recommended_wght_a",
        "shelxl_recommended_wght_b": "recommended_wght_b",
    }
    pathology_count_pairs = {"shelxl_npd_count", "shelxl_split_count"}
    pathology_flag_pairs = {"shelxl_has_npd", "shelxl_has_split_warnings"}
    for out_key, detail_key in fallback_pairs.items():
        old_value = out.get(out_key)
        if detail_key not in shelxl_details:
            continue
        detail_value = shelxl_details[detail_key]
        if out_key in pathology_count_pairs:
            old_count = _quick_int(old_value)
            detail_count = _quick_int(detail_value)
            if old_count is None:
                out[out_key] = detail_value
            elif detail_count is not None:
                out[out_key] = max(old_count, detail_count)
        elif out_key == "shelxl_npd_atoms":
            old_atoms = old_value if isinstance(old_value, (list, tuple, set)) else []
            detail_atoms = detail_value if isinstance(detail_value, (list, tuple, set)) else []
            out[out_key] = sorted(
                {str(atom) for atom in [*old_atoms, *detail_atoms] if str(atom).strip()}
            )
        elif out_key in pathology_flag_pairs:
            out[out_key] = bool(old_value) or bool(detail_value)
        elif old_value is None or old_value == "":
            out[out_key] = detail_value
    out["shelxl_lst_exists"] = shelxl_details.get("lst_exists", shelxl_details.get("exists", False))
    out["provenance_required"] = True
    out["artifact_provenance_ok"] = provenance_ok
    out["artifact_provenance"] = artifact_provenance
    out["res_provenance_ok"] = provenance_ok
    out["res_provenance"] = artifact_provenance
    out["res_sha256"] = str(canonical_res.get("sha256") or "")
    out["hkl_sha256"] = str(canonical_hkl.get("sha256") or "")
    out["lst_sha256"] = str(canonical_lst.get("sha256") or "")
    out["endpoint_res_sha256"] = out["res_sha256"]
    out["endpoint_hkl_sha256"] = out["hkl_sha256"]
    out["endpoint_lst_sha256"] = out["lst_sha256"]
    out["metric_consistency_required"] = True
    out["metric_consistency_ok"] = metric_consistency.get("passed") is True
    out["metric_consistency"] = metric_consistency
    out["promotion_fields_required"] = True
    out["promotion_fields_ok"] = promotion_fields.get("passed") is True
    out["promotion_field_contract"] = promotion_fields
    out["pathology_required"] = True
    out["pathology_status"] = str(pathology.get("pathology_status") or "unverified")
    if pathology_complete:
        out["shelxl_npd_count"] = max(
            _quick_int(out.get("shelxl_npd_count")) or 0,
            int(pathology.get("npd_count", 0) or 0),
        )
        old_atoms = out.get("shelxl_npd_atoms")
        old_atoms = old_atoms if isinstance(old_atoms, (list, tuple, set)) else []
        out["shelxl_npd_atoms"] = sorted(
            {str(atom) for atom in [*old_atoms, *(pathology.get("npd_atoms") or [])] if str(atom).strip()}
        )
        out["shelxl_split_count"] = max(
            _quick_int(out.get("shelxl_split_count")) or 0,
            int(pathology.get("split_count", 0) or 0),
        )
        out["shelxl_has_npd"] = bool(out.get("shelxl_has_npd")) or bool(pathology.get("has_npd"))
        out["shelxl_has_split_warnings"] = bool(out.get("shelxl_has_split_warnings")) or bool(
            pathology.get("has_split_warnings")
        )
    else:
        # Preserve every recorded warning in diagnostic output.  An
        # unverified/truncated LST cannot prove cleanliness, but it must not
        # erase pathology already recorded by the producer either.
        out.setdefault("shelxl_npd_count", None)
        out.setdefault("shelxl_npd_atoms", [])
        out.setdefault("shelxl_split_count", None)
        out.setdefault("shelxl_has_npd", None)
        out.setdefault("shelxl_has_split_warnings", None)
    if not out.get("wght_mode") and provenance_ok:
        wght_a = _quick_fraction(out.get("shelxl_active_wght_a"))
        wght_b = _quick_fraction(out.get("shelxl_active_wght_b"))
        if wght_a == 0.0 and wght_b == 0.0:
            out["wght_mode"] = "freeze-zero"
    if not provenance_ok:
        out["provenance_error"] = (
            "publication endpoint RES/HKL/LST no longer match the exact paths and hashes selected by GROND; "
            "use GROND push-endpoint for an edited model"
        )
    elif not pathology_complete:
        out["provenance_error"] = (
            "publication endpoint LST does not contain a complete final SHELXL NPD/split summary"
        )
    elif metric_consistency.get("passed") is not True:
        out["provenance_error"] = str(metric_consistency.get("reason") or "stored/live endpoint metrics disagree")
    elif promotion_fields.get("passed") is not True:
        out["provenance_error"] = str(promotion_fields.get("reason") or "producer promotion fields are incomplete")
    return out


def _endpoint_calibration_publication_endpoint(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the endpoint-calibration publication replay, if present."""

    if result.get("combined"):
        # Historical multi summaries may contain an unsafe single-table replay.
        # Multi-dataset scale contracts are independent and can only be
        # calibrated through a future per-dataset replay-and-recombine path.
        return {}
    endpoint_calibration = result.get("endpoint_calibration") if isinstance(result.get("endpoint_calibration"), Mapping) else {}
    selection = endpoint_calibration.get("publication_selection", {}) if isinstance(endpoint_calibration, Mapping) else {}
    endpoint = selection.get("publication_endpoint") if isinstance(selection, Mapping) else {}
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}
    return _normalize_publication_endpoint_block(endpoint, selection_source="endpoint_calibration")


def _chooser_publication_endpoint(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the endpoint chooser's best physical endpoint, if present."""

    chooser = result.get("publication_endpoint_chooser") if isinstance(result.get("publication_endpoint_chooser"), Mapping) else {}
    endpoint = chooser.get("best_physical_endpoint") if isinstance(chooser, Mapping) else {}
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}
    return _normalize_publication_endpoint_block(endpoint, selection_source="endpoint_chooser")


def _chooser_diagnostic_endpoint(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the endpoint chooser's best rejected diagnostic endpoint, if present."""

    chooser = result.get("publication_endpoint_chooser") if isinstance(result.get("publication_endpoint_chooser"), Mapping) else {}
    endpoint = chooser.get("best_diagnostic_endpoint") if isinstance(chooser, Mapping) else {}
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}
    return _normalize_publication_endpoint_block(endpoint, selection_source="endpoint_chooser_diagnostic")


def _quick_endpoint_file_block(publication_endpoint: Mapping[str, object], final_model: Path | None) -> dict[str, str]:
    """Collect the endpoint paths users most often need to open."""

    final_endpoint_dir = final_model.parent if final_model else None
    final_endpoint_stem = final_model.stem if final_model else ""

    def existing(path: Path | None) -> str:
        """Return path as text when the file exists, else an empty string."""

        return str(path) if path is not None and path.exists() else ""

    return {
        "final_feedback_res": str(final_model or ""),
        "final_feedback_hkl": existing(final_endpoint_dir / (final_endpoint_stem + ".hkl") if final_endpoint_dir else None),
        "final_feedback_lst": existing(final_endpoint_dir / (final_endpoint_stem + ".lst") if final_endpoint_dir else None),
        "final_feedback_manifest": existing(final_endpoint_dir / "manifest.json" if final_endpoint_dir else None),
        "publication_res": str(publication_endpoint.get("endpoint_res", "")),
        "publication_ins": str(publication_endpoint.get("endpoint_ins", "")),
        "publication_hkl": str(publication_endpoint.get("endpoint_hkl", "")),
        "publication_lst": str(publication_endpoint.get("endpoint_lst", "")),
        "publication_manifest": str(publication_endpoint.get("endpoint_manifest", "")),
        "publication_row_table": str(publication_endpoint.get("endpoint_row_table", "")),
    }
