"""Bind hand evidence to the actual model, rows and physics configuration."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path

from grond.macromolecule import sha256_file
from grond.workflows.decision import safe_float


def _summary_model_binding(
    summary: Mapping[str, object],
    expected_model: Path,
    *,
    semantic_hand: str,
) -> dict[str, object]:
    """Bind one native summary to exact model bytes and explicit hand semantics."""

    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    recorded_path = str(inputs.get("res") or "")
    recorded_sha = str(inputs.get("res_sha256") or "")
    model_tag = str(inputs.get("model_tag") or "")
    inversion_key = "alternate_hand" if "alternate_hand" in config else "invert_hand" if "invert_hand" in config else ""
    engine_inversion = config.get(inversion_key) if inversion_key else None
    errors: list[str] = []
    expected_path = expected_model.expanduser().resolve()
    expected_sha = ""
    if not expected_path.is_file():
        errors.append(f"expected model is missing: {expected_path}")
    else:
        try:
            expected_sha = sha256_file(expected_path)
        except OSError as exc:
            errors.append(f"cannot hash expected model {expected_path}: {exc}")
    if not recorded_path:
        errors.append("summary inputs.res is missing")
    else:
        try:
            resolved_recorded = Path(recorded_path).expanduser().resolve()
        except OSError:
            resolved_recorded = Path(recorded_path).expanduser()
        if resolved_recorded != expected_path:
            errors.append(f"summary inputs.res does not match expected model: {resolved_recorded} != {expected_path}")
    if not recorded_sha:
        errors.append("summary inputs.res_sha256 is missing")
    elif expected_sha and recorded_sha != expected_sha:
        errors.append(
            f"summary inputs.res_sha256 does not match expected model bytes: {recorded_sha} != {expected_sha}"
        )
    if semantic_hand not in {"original", "alternate"}:
        errors.append(f"unknown semantic hand label {semantic_hand!r}")
    if not model_tag:
        errors.append("summary inputs.model_tag is missing")
    elif model_tag.split(";", 1)[0].strip().lower() != "original":
        errors.append(f"summary model_tag {model_tag!r} indicates an additional engine-side hand transformation")
    if not inversion_key:
        errors.append("summary config does not record alternate_hand/invert_hand")
    elif bool(engine_inversion):
        errors.append(
            f"summary config {inversion_key}={engine_inversion!r} applies an unexpected second hand inversion"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "semantic_hand": semantic_hand,
        "expected_model": str(expected_path),
        "expected_model_sha256": expected_sha,
        "summary_inputs_res": recorded_path,
        "summary_inputs_res_sha256": recorded_sha,
        "summary_model_tag": model_tag,
        "summary_inversion_key": inversion_key,
        "summary_engine_inversion": engine_inversion,
        "semantic_rule": (
            "the named model bytes already embody this semantic hand; native recomputation must use model_tag "
            "original with no additional engine-side inversion"
        ),
        "errors": errors,
    }


def _summary_hkl_binding(summary: Mapping[str, object], expected_hkl: Path) -> dict[str, object]:
    """Bind one native summary to the exact dataset observation-file bytes."""

    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    recorded_path = str(inputs.get("hkl") or "")
    recorded_sha = str(inputs.get("hkl_sha256") or "")
    expected_path = expected_hkl.expanduser().resolve()
    expected_sha = ""
    errors: list[str] = []
    if not expected_path.is_file():
        errors.append(f"expected dataset HKL is missing: {expected_path}")
    else:
        try:
            expected_sha = sha256_file(expected_path)
        except OSError as exc:
            errors.append(f"cannot hash expected dataset HKL {expected_path}: {exc}")
    if not recorded_path:
        errors.append("summary inputs.hkl is missing")
    else:
        try:
            resolved_recorded = Path(recorded_path).expanduser().resolve()
        except OSError:
            resolved_recorded = Path(recorded_path).expanduser()
        if resolved_recorded != expected_path:
            errors.append(f"summary inputs.hkl does not match DatasetSpec.hkl: {resolved_recorded} != {expected_path}")
    if not recorded_sha:
        errors.append("summary inputs.hkl_sha256 is missing")
    elif expected_sha and recorded_sha != expected_sha:
        errors.append(
            f"summary inputs.hkl_sha256 does not match DatasetSpec.hkl bytes: {recorded_sha} != {expected_sha}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "expected_hkl": str(expected_path),
        "expected_hkl_sha256": expected_sha,
        "summary_inputs_hkl": recorded_path,
        "summary_inputs_hkl_sha256": recorded_sha,
        "errors": errors,
    }


def _summary_native_row_table_binding(
    summary: Mapping[str, object],
    expected_table: Path,
) -> dict[str, object]:
    """Bind a source final row table to the producer-recorded path and SHA."""

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    recorded_path = str(artifacts.get("native_row_table") or "")
    recorded_sha = str(artifacts.get("native_row_table_sha256") or "")
    expected_path = expected_table.expanduser().resolve()
    expected_sha = ""
    errors: list[str] = []
    if not expected_path.is_file():
        errors.append(f"source final native row table is missing: {expected_path}")
    else:
        try:
            expected_sha = sha256_file(expected_path)
        except OSError as exc:
            errors.append(f"cannot hash source final native row table {expected_path}: {exc}")
    if not recorded_path:
        errors.append("summary artifacts.native_row_table is missing")
    else:
        try:
            resolved_recorded = Path(recorded_path).expanduser().resolve()
        except OSError:
            resolved_recorded = Path(recorded_path).expanduser()
        if resolved_recorded != expected_path:
            errors.append(
                f"summary native row-table path does not match source final table: {resolved_recorded} != {expected_path}"
            )
    if not recorded_sha:
        errors.append("summary artifacts.native_row_table_sha256 is missing")
    elif expected_sha and recorded_sha != expected_sha:
        errors.append(
            "summary native_row_table_sha256 does not match current source final row-table bytes: "
            f"{recorded_sha} != {expected_sha}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "expected_table": str(expected_path),
        "expected_table_sha256": expected_sha,
        "summary_native_row_table": recorded_path,
        "summary_native_row_table_sha256": recorded_sha,
        "errors": errors,
    }


def _policy_selected_table_binding(
    source_path: Path,
    selected_path: Path,
) -> dict[str, object]:
    """Verify a policy-annotated selected table preserves every source data cell."""

    policy_columns = {
        "dataset",
        "dataset_role",
        "raw_row_weight",
        "endpoint_weight",
        "sigma_eff_original",
        "policy_sigma_weight_applied",
    }

    key_fields = ("row_index", "h", "k", "l")

    def load(path: Path) -> tuple[list[str], dict[tuple[str, ...], dict[str, str]], int, list[str]]:
        errors: list[str] = []
        fields: list[str] = []
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        duplicates = 0
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                missing = [field for field in key_fields if field not in fields]
                if missing:
                    return fields, rows, duplicates, [f"{path} is missing key columns: {', '.join(missing)}"]
                for row in reader:
                    key = tuple(str(row.get(field, "")).strip() for field in key_fields)
                    if key in rows:
                        duplicates += 1
                    else:
                        rows[key] = {field: str(row.get(field, "")) for field in fields}
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
        return fields, rows, duplicates, errors

    source_fields, source_rows, source_duplicates, source_errors = load(source_path)
    selected_fields, selected_rows, selected_duplicates, selected_errors = load(selected_path)
    source_contract_fields = [field for field in source_fields if field not in policy_columns]
    selected_contract_fields = [field for field in selected_fields if field not in policy_columns]
    errors = [*source_errors, *selected_errors]
    if set(source_contract_fields) != set(selected_contract_fields):
        errors.append(
            "policy-selected table non-policy columns differ from source final table: "
            f"source_only={sorted(set(source_contract_fields) - set(selected_contract_fields))}, "
            f"selected_only={sorted(set(selected_contract_fields) - set(source_contract_fields))}"
        )
    if source_duplicates or selected_duplicates:
        errors.append(
            f"policy/source row keys are ambiguous (source={source_duplicates}, selected={selected_duplicates})"
        )
    joined = set(source_rows) & set(selected_rows)
    if len(joined) != len(source_rows) or len(joined) != len(selected_rows):
        errors.append(
            f"policy/source row join is not exact (source={len(source_rows)}, "
            f"selected={len(selected_rows)}, joined={len(joined)})"
        )
    mismatch_counts = {field: 0 for field in source_contract_fields}
    examples: list[dict[str, object]] = []
    comparable_fields = [field for field in source_contract_fields if field in selected_contract_fields]
    for key in sorted(joined):
        for field in comparable_fields:
            if source_rows[key].get(field, "") != selected_rows[key].get(field, ""):
                mismatch_counts[field] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "source": source_rows[key].get(field, ""),
                            "selected": selected_rows[key].get(field, ""),
                        }
                    )
    changed = {field: count for field, count in mismatch_counts.items() if count}
    if changed:
        errors.append(f"policy-selected table changed non-policy source cells: {changed}")
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "source_table": str(source_path.expanduser().resolve()),
        "selected_table": str(selected_path.expanduser().resolve()),
        "source_table_sha256": sha256_file(source_path) if source_path.is_file() else "",
        "selected_table_sha256": sha256_file(selected_path) if selected_path.is_file() else "",
        "policy_columns_allowed_to_differ": sorted(policy_columns),
        "non_policy_columns_compared": source_contract_fields,
        "source_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "joined_rows": len(joined),
        "duplicate_keys_source": source_duplicates,
        "duplicate_keys_selected": selected_duplicates,
        "non_policy_mismatch_counts": changed,
        "mismatch_examples": examples,
        "errors": errors,
    }


def _native_physics_config_contract(
    selected_summary: Mapping[str, object],
    opposite_summary: Mapping[str, object],
) -> dict[str, object]:
    """Verify the two hand branches used the same physics-bearing stage config."""

    selected = selected_summary.get("config") if isinstance(selected_summary.get("config"), Mapping) else {}
    opposite = opposite_summary.get("config") if isinstance(opposite_summary.get("config"), Mapping) else {}
    exact_fields = (
        "n_basis",
        "n_points",
        "thickness_points",
        "sg_window_mode",
        "sg_jitter_mode_grid",
        "profile_quadrature",
        "profile_accumulation",
        "split_seed",
        "scale_fit_split",
        "target_mode",
        "positive_only",
        "hkl_transform",
        "compute_backend",
        "bethe_strong_count",
        "bethe_variant",
        "matrix_convention",
    )
    numeric_fields = (
        "sg_window",
        "sg_window_min",
        "sg_window_max",
        "sg_window_scale",
        "sg_window_frame_floor",
        "topup_d_min",
        "max_support_sg",
        "free_fraction",
        "scale_smoothness",
        "min_d_spacing",
        "min_peak",
        "geometry_percentile",
        "bethe_promote_denom",
        "bethe_denom_eps",
    )
    errors: list[str] = []
    exact_mismatches: dict[str, dict[str, object]] = {}
    numeric_mismatches: dict[str, dict[str, object]] = {}
    for field in exact_fields:
        if field not in selected or field not in opposite:
            errors.append(f"physics config field {field} is missing from one or both hand branches")
            continue
        if selected.get(field) != opposite.get(field):
            exact_mismatches[field] = {
                "selected": selected.get(field),
                "opposite": opposite.get(field),
            }
    for field in numeric_fields:
        selected_value = safe_float(selected.get(field), float("nan"))
        opposite_value = safe_float(opposite.get(field), float("nan"))
        if not math.isfinite(selected_value) or not math.isfinite(opposite_value):
            errors.append(f"physics config field {field} is missing or nonfinite in one or both hand branches")
            continue
        if not math.isclose(selected_value, opposite_value, rel_tol=1e-12, abs_tol=1e-12):
            numeric_mismatches[field] = {
                "selected": selected_value,
                "opposite": opposite_value,
                "absolute_delta": abs(opposite_value - selected_value),
            }
    if exact_mismatches:
        errors.append(f"exact physics config differs between hand branches: {exact_mismatches}")
    if numeric_mismatches:
        errors.append(f"numeric physics config differs between hand branches: {numeric_mismatches}")
    selected_transform = (
        (selected_summary.get("inputs") or {}).get("hkl_transform")
        if isinstance(selected_summary.get("inputs"), Mapping)
        else None
    )
    opposite_transform = (
        (opposite_summary.get("inputs") or {}).get("hkl_transform")
        if isinstance(opposite_summary.get("inputs"), Mapping)
        else None
    )
    if not selected_transform or not opposite_transform:
        errors.append("summary inputs.hkl_transform is missing from one or both hand branches")
    elif str(selected_transform) != str(opposite_transform):
        errors.append(
            f"resolved HKL transform differs between hand branches: {selected_transform!r} != {opposite_transform!r}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "exact_fields_checked": list(exact_fields),
        "numeric_fields_checked": list(numeric_fields),
        "selected_resolved_hkl_transform": selected_transform,
        "opposite_resolved_hkl_transform": opposite_transform,
        "exact_mismatches": exact_mismatches,
        "numeric_mismatches": numeric_mismatches,
        "errors": errors,
    }
