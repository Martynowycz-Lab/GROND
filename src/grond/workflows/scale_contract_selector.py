from __future__ import annotations

import math
from collections.abc import Mapping


def _finite(mapping: Mapping[str, object], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(key)
    return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = _finite(mapping, key)
    integer = int(value)
    if value != integer:
        raise ValueError(key)
    return integer


def select_multi_scale_contract(
    full: Mapping[str, object],
    compact: Mapping[str, object],
    *,
    min_scale_reduction: float = 0.10,
    require_bic_improvement: bool = False,
    max_row_rall_cost: float = 0.01,
    max_endpoint_r1_cost: float = 0.005,
    max_endpoint_wr2_cost: float = 0.01,
    goof_min: float = 0.85,
    goof_max: float = 1.15,
    max_input_row_loss_fraction: float = 0.001,
    min_bloch_rall_gain: float = 0.002,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    require_endpoint_health: bool = True,
) -> dict[str, object]:
    """Choose a compact multi-dataset scale contract or fail closed to full.

    The function is deliberately pure: callers retain both endpoint artifacts,
    while this decision records whether parsimony was earned without excessive
    held-out/work JANA-style row Rall cost or loss of the Born-to-Bloch Rall
    advantage. Endpoint factors are reports; coverage and pathology remain gates.
    """

    reasons: list[str] = []
    warnings: list[str] = []
    diagnostics: dict[str, object] = {}
    try:
        full_scales = _integer(full, "scale_params")
        compact_scales = _integer(compact, "scale_params")
        full_bic = _finite(full, "aggregate_bic")
        compact_bic = _finite(compact, "aggregate_bic")
        full_input_rows = _integer(full, "input_endpoint_rows")
        compact_input_rows = _integer(compact, "input_endpoint_rows")
        full_rows = full["row_rall"]
        compact_rows = compact["row_rall"]
        full_endpoint = full["endpoint"]
        compact_endpoint = compact["endpoint"]
        compact_bloch_gain = compact["bloch_gain_rall"]
        if not all(
            isinstance(value, Mapping)
            for value in (full_rows, compact_rows, full_endpoint, compact_endpoint, compact_bloch_gain)
        ):
            raise ValueError("nested contract")
        full_row_values = {split: _finite(full_rows, split) for split in ("all", "work", "free")}
        compact_row_values = {split: _finite(compact_rows, split) for split in ("all", "work", "free")}
        full_r1_all = _finite(full_endpoint, "R1_all")
        full_r1_gt = _finite(full_endpoint, "R1_gt")
        full_wr2 = _finite(full_endpoint, "wR2_all")
        compact_r1_all = _finite(compact_endpoint, "R1_all")
        compact_r1_gt = _finite(compact_endpoint, "R1_gt")
        compact_wr2 = _finite(compact_endpoint, "wR2_all")
        compact_goof = float(compact_endpoint["GooF"])
        full_reflections = _integer(full_endpoint, "n_reflections")
        compact_reflections = _integer(compact_endpoint, "n_reflections")
        full_splits = _integer(full_endpoint, "split_count")
        compact_splits = _integer(compact_endpoint, "split_count")
        compact_npd = _integer(compact_endpoint, "npd_count")
        compact_bloch_gains = {
            split: _finite(compact_bloch_gain, split)
            for split in ("all", "work", "free")
        }
        compact_hkl_safe_factor = _finite(compact_endpoint, "hkl_safe_factor")
        compact_modulation_p95 = float(compact_endpoint.get("modulation_p95", float("nan")))
        compact_returncode = _integer(compact_endpoint, "returncode")
        compact_status = str(compact_endpoint["status"]).strip().lower()
        compact_safe_for_feedback = compact_endpoint["safe_for_feedback"] is True
        compact_artifacts_exist = compact_endpoint["artifacts_exist"] is True
        if full_scales <= 0 or compact_scales <= 0 or full_input_rows <= 0:
            raise ValueError("nonpositive contract size")
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            "selected_contract": "full",
            "accepted": False,
            "reasons": ["invalid_candidate_contract"],
            "warnings": warnings,
            "diagnostics": diagnostics,
        }

    scale_reduction = (full_scales - compact_scales) / float(full_scales)
    input_row_loss = max(0, full_input_rows - compact_input_rows) / float(full_input_rows)
    diagnostics.update(
        {
            "scale_reduction_fraction": scale_reduction,
            "aggregate_bic_delta": compact_bic - full_bic,
            "input_endpoint_row_loss_fraction": input_row_loss,
            "row_rall_cost": {
                split: compact_row_values[split] - full_row_values[split]
                for split in ("all", "work", "free")
            },
            "endpoint_cost": {
                "R1_all": compact_r1_all - full_r1_all,
                "R1_gt": compact_r1_gt - full_r1_gt,
                "wR2_all": compact_wr2 - full_wr2,
            },
            "compact_bloch_rall_gain": compact_bloch_gains,
            "compact_hkl_safe_factor": compact_hkl_safe_factor,
            "compact_modulation_p95": compact_modulation_p95,
        }
    )

    epsilon = 1e-12
    full_physics = full.get("physics_contract")
    compact_physics = compact.get("physics_contract")
    if (full_physics is not None or compact_physics is not None) and full_physics != compact_physics:
        reasons.append("physics_contract_mismatch")
    if scale_reduction + epsilon < float(min_scale_reduction):
        reasons.append("scale_reduction_below_minimum")
    if require_bic_improvement and compact_bic >= full_bic:
        warnings.append("aggregate_bic_not_improved_diagnostic_only")
    if input_row_loss > float(max_input_row_loss_fraction) + epsilon:
        reasons.append("input_endpoint_row_loss_exceeded")
    for split in ("all", "work", "free"):
        if compact_row_values[split] - full_row_values[split] > float(max_row_rall_cost) + epsilon:
            reasons.append(f"row_rall_{split}_cost_exceeded")
        if compact_bloch_gains[split] + epsilon < float(min_bloch_rall_gain):
            reasons.append(f"pooled_bloch_rall_gain_{split}_below_minimum")
    if compact_r1_all - full_r1_all > float(max_endpoint_r1_cost) + epsilon:
        warnings.append("endpoint_r1_all_cost_exceeded_diagnostic_only")
    if compact_r1_gt - full_r1_gt > float(max_endpoint_r1_cost) + epsilon:
        warnings.append("endpoint_r1_gt_cost_exceeded_diagnostic_only")
    if compact_wr2 - full_wr2 > float(max_endpoint_wr2_cost) + epsilon:
        warnings.append("endpoint_wr2_all_cost_exceeded_diagnostic_only")
    if not math.isfinite(compact_goof):
        warnings.append("endpoint_goof_unavailable")
    elif compact_goof < float(goof_min) - epsilon or compact_goof > float(goof_max) + epsilon:
        warnings.append(
            f"endpoint_goof_outside_preferred_range:{compact_goof:.6g} not in {float(goof_min):.6g}-{float(goof_max):.6g}"
        )
    if compact_npd > 0:
        reasons.append("endpoint_npd_present")
    if compact_splits > full_splits:
        reasons.append("endpoint_new_split_pathology")
    if compact_reflections != full_reflections:
        reasons.append("merged_reflection_count_mismatch")
    if require_endpoint_health:
        if compact_returncode != 0 or compact_status != "ok":
            reasons.append("endpoint_shelxl_failed")
        if not compact_safe_for_feedback:
            reasons.append("endpoint_res_unsafe_for_feedback")
        if not compact_artifacts_exist:
            reasons.append("endpoint_artifacts_missing")
    if compact_hkl_safe_factor + epsilon < float(min_hkl_safe_factor):
        reasons.append("endpoint_hkl_safety_below_minimum")
    if not math.isfinite(compact_modulation_p95):
        warnings.append("endpoint modulation p95 is unavailable")
    elif compact_modulation_p95 > float(max_modulation_p95) + epsilon:
        warnings.append(
            f"endpoint modulation p95 {compact_modulation_p95:.4g} exceeds warning level {float(max_modulation_p95):.4g}"
        )
    diagnostics["warnings"] = warnings

    accepted = not reasons
    return {
        "selected_contract": "compact" if accepted else "full",
        "accepted": accepted,
        "reasons": reasons,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
