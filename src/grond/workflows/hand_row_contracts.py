"""Require the two hand branches to use the same physical observations."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping
from pathlib import Path

from grond.workflows.decision import safe_float, score_native_summary

UNIFORM_HAND_PAIR_ABSORPTION_MODES = {"mean", "midgley_like", "midgley"}


def elastic_hand_nuisance(nuisance: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Freeze the nuisance for an exact negative-thickness hand pair.

    A uniform absorptive potential is kept (H0 + i*mu*I keeps the pair exact); any non-uniform term is refused
    instead of silently dropped.
    """

    elastic = dict(nuisance)
    absorption_keys = (
        "absorption_mode",
        "absorption_mu",
        "absorption_frac",
        "absorption_mu_g2",
        "absorption_mu_theta2",
        "absorption_tds_b",
    )
    source = {key: elastic.get(key) for key in absorption_keys}
    mode = str(elastic.get("absorption_mode") or "none").strip().lower()
    nonuniform = {
        key: safe_float(elastic.get(key, 0.0), float("nan"))
        for key in ("absorption_frac", "absorption_mu_g2", "absorption_mu_theta2", "absorption_tds_b")
    }
    mu = safe_float(elastic.get("absorption_mu", 0.0), float("nan"))
    active_nonuniform = {key: value for key, value in nonuniform.items() if value != 0.0}
    if mode == "none" or (mode in UNIFORM_HAND_PAIR_ABSORPTION_MODES and not active_nonuniform):
        if not math.isfinite(mu) or mu < 0.0:
            raise ValueError(f"hand pair absorption_mu must be finite and nonnegative, got {mu!r}")
        if mode == "none" and (mu != 0.0 or active_nonuniform):
            raise ValueError("absorption_mode none carries nonzero absorptive coefficients; refusing the hand pair")
        return elastic, source
    raise ValueError(
        "the exact hand pair keeps only uniform absorption (mean/midgley_like/midgley with zero "
        f"absorption_frac, absorption_mu_g2, absorption_mu_theta2, absorption_tds_b); this nuisance has "
        f"absorption_mode={mode!r} with {active_nonuniform or 'a non-uniform mode'}"
    )


def score_native_metrics(summary: Mapping[str, object], metrics: Mapping[str, object]) -> dict[str, object]:
    """Score alternate metrics while retaining the source summary's non-metric context."""

    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    return score_native_summary(
        {
            **dict(summary),
            "best_native": {**dict(best), "metrics": dict(metrics)},
        }
    )


def _positive_summary_int(config: Mapping[str, object], key: str) -> int | None:
    """Return one required positive integer from a native-stage config."""

    try:
        value = int(config.get(key, 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def _args_with_recorded_native_contract(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> argparse.Namespace:
    """Clone CLI arguments with physics-bearing native settings from a summary."""

    cloned = argparse.Namespace(**vars(args))
    direct_keys = (
        "sg_window",
        "sg_window_mode",
        "sg_window_min",
        "sg_window_max",
        "sg_window_scale",
        "sg_window_frame_floor",
        "sg_jitter_student_df",
        "sg_jitter_width_factor",
        # The recorded engine scalars; stages pinned from the recorded nuisance forward
        # the same cone, and the gate options are not used after the final stage.
        "mosaicity_mode",
        "mosaicity_sigma_deg",
        "mosaicity_points",
        "mosaicity_max_sigma_deg",
        "profile_quadrature",
        "profile_accumulation",
        "topup_d_min",
        "max_support_sg",
        "free_fraction",
        "split_seed",
        "split_scheme",
        "split_salt",
        "exclude_orbits_json",
        "selection_metric",
        "prediction_cache_compression",
        "row_plan_cache",
        "matrix_convention",
    )
    for key in direct_keys:
        if key in config and config.get(key) is not None:
            setattr(cloned, key, config.get(key))
    if config.get("thickness_points") is not None:
        cloned.final_thickness_points = int(config["thickness_points"])
    return cloned


def _row_contract_nullable_numeric_fields(
    config: Mapping[str, object],
    nuisance: Mapping[str, object],
) -> set[str]:
    """Return row fields that are physically not applicable for this run."""

    nullable: set[str] = set()
    sg_window_mode = str(config.get("sg_window_mode", "") or "").strip().lower()
    if sg_window_mode in {"fixed", "rlp_peak"}:
        nullable.add("sg_window_half_frames")
    sg_jitter_mode = str(nuisance.get("sg_jitter_mode", config.get("sg_jitter_mode", "")) or "").strip().lower()
    if sg_jitter_mode == "none":
        nullable.update({"sg_jitter_d_inv_A", "sg_jitter_frame_fraction"})
    return nullable


def _paired_row_split_contract(
    best_path: Path,
    other_path: Path,
    *,
    nullable_numeric_fields: set[str] | None = None,
) -> dict[str, object]:
    """Verify identical selected/opposite observation and row-selection contracts."""

    key_fields = ("row_index", "h", "k", "l")
    exact_fields = (
        "split",
        "group",
        "detector_bin",
        "basis_size",
        "omega_bragg_root_available",
        "sg_window_clipped",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
    )
    numeric_fields = (
        "target_I",
        "sigma_eff",
        "xds_sigma",
        "d_A",
        "peak",
        "corr",
        "rlp",
        "row_resolution_factor",
        "center_sg_inv_A",
        "omega_bragg_deg",
        "omega_bragg_minus_center_deg",
        "sg_window_inv_A",
        "sg_window_raw_inv_A",
        "sg_window_half_frames",
        "sg_dSg_domega_inv_A_per_deg",
        "sg_dSg_dframe_inv_A",
        "sg_jitter_sigma_inv_A",
        "sg_jitter_d_inv_A",
        "sg_jitter_frame_fraction",
        "sg_profile_weight_sum",
        "max_sg_solve_residual",
    )
    nullable_numeric_fields = set(nullable_numeric_fields or ())
    unknown_nullable_fields = nullable_numeric_fields - set(numeric_fields)
    if unknown_nullable_fields:
        raise ValueError("unknown nullable row-contract fields: " + ", ".join(sorted(unknown_nullable_fields)))
    relative_tolerance = 1e-12
    absolute_tolerance = 1e-12
    conditional_nan_fields = {
        # These are reporting-only diagnostics.  A scan-centered propagation
        # profile remains well defined when no local Sg=0 crossing exists.
        # Accept the paired NaN sentinel only when both branches explicitly
        # record that the diagnostic root is unavailable.
        "omega_bragg_deg": ("omega_bragg_root_available", "0"),
        "omega_bragg_minus_center_deg": ("omega_bragg_root_available", "0"),
    }

    def indexed(path: Path) -> tuple[dict[tuple[str, ...], dict[str, str]], int, list[str]]:
        """Read a row table keyed by ``key_fields``; return the compared fields per row, duplicate count, errors."""
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        duplicates = 0
        errors: list[str] = []
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                missing = [field for field in (*key_fields, *exact_fields, *numeric_fields) if field not in fields]
                if missing:
                    return {}, 0, [f"{path} is missing columns: {', '.join(missing)}"]
                for row in reader:
                    key = tuple(str(row.get(field, "")).strip() for field in key_fields)
                    if key in rows:
                        duplicates += 1
                    else:
                        rows[key] = {
                            field: str(row.get(field, "")).strip() for field in (*exact_fields, *numeric_fields)
                        }
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
        return rows, duplicates, errors

    best, duplicates_best, errors_best = indexed(best_path)
    other, duplicates_other, errors_other = indexed(other_path)
    joined = set(best) & set(other)
    exact_mismatch_counts = {field: 0 for field in exact_fields}
    numeric_mismatch_counts = {field: 0 for field in numeric_fields}
    nonfinite_counts = {field: 0 for field in numeric_fields}
    matched_nullable_nan_counts = {field: 0 for field in nullable_numeric_fields}
    matched_conditional_nan_counts = {field: 0 for field in conditional_nan_fields}
    mismatch_examples: list[dict[str, object]] = []

    def parsed_numeric_token(raw: str) -> tuple[str, float]:
        """Classify a table cell as ``finite``, ``nan`` (the literal NaN sentinel) or ``invalid``, with its value."""
        token = str(raw).strip()
        if token.lower() == "nan":
            return "nan", float("nan")
        try:
            value = float(token)
        except (TypeError, ValueError):
            return "invalid", float("nan")
        return ("finite", value) if math.isfinite(value) else ("invalid", value)

    for key in sorted(joined):
        for field in exact_fields:
            if best[key][field] != other[key][field]:
                exact_mismatch_counts[field] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "selected": best[key][field],
                            "opposite": other[key][field],
                        }
                    )
        for field in numeric_fields:
            selected_kind, selected_value = parsed_numeric_token(best[key][field])
            opposite_kind, opposite_value = parsed_numeric_token(other[key][field])
            if selected_kind != "finite" or opposite_kind != "finite":
                conditional = conditional_nan_fields.get(field)
                if field in nullable_numeric_fields and selected_kind == "nan" and opposite_kind == "nan":
                    matched_nullable_nan_counts[field] += 1
                    continue
                if (
                    conditional is not None
                    and selected_kind == "nan"
                    and opposite_kind == "nan"
                    and best[key][conditional[0]] == conditional[1]
                    and other[key][conditional[0]] == conditional[1]
                ):
                    matched_conditional_nan_counts[field] += 1
                    continue
                nonfinite_counts[field] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "selected": best[key][field],
                            "opposite": other[key][field],
                            "reason": (
                                "nan_not_allowed"
                                if selected_kind == "nan" and opposite_kind == "nan"
                                else "invalid_or_nonfinite_numeric_token"
                            ),
                        }
                    )
            elif not math.isclose(
                selected_value,
                opposite_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                numeric_mismatch_counts[field] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "selected": selected_value,
                            "opposite": opposite_value,
                            "absolute_delta": abs(opposite_value - selected_value),
                        }
                    )
    errors = [*errors_best, *errors_other]
    if duplicates_best or duplicates_other:
        errors.append(f"duplicate row keys are ambiguous (selected={duplicates_best}, opposite={duplicates_other})")
    if len(joined) != len(best) or len(joined) != len(other):
        errors.append(f"row-key join is not exact (selected={len(best)}, opposite={len(other)}, joined={len(joined)})")
    for field, count in exact_mismatch_counts.items():
        if count:
            errors.append(f"{count} joined rows change exact contract column {field}")
    for field, count in nonfinite_counts.items():
        if count:
            errors.append(f"{count} joined rows have nonfinite selected/opposite values for {field}")
    for field, count in numeric_mismatch_counts.items():
        if count:
            errors.append(
                f"{count} joined rows change numeric contract column {field} beyond "
                f"rtol={relative_tolerance:g}, atol={absolute_tolerance:g}"
            )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "key": list(key_fields),
        "selected_rows": len(best),
        "opposite_rows": len(other),
        "joined_rows": len(joined),
        "duplicate_keys_selected": duplicates_best,
        "duplicate_keys_opposite": duplicates_other,
        "exact_columns_checked": list(exact_fields),
        "numeric_columns_checked": list(numeric_fields),
        "nullable_numeric_columns": sorted(nullable_numeric_fields),
        "matched_nullable_nan_counts": matched_nullable_nan_counts,
        "conditional_nan_contracts": {
            field: {"exact_field": exact_field, "required_value": required_value}
            for field, (exact_field, required_value) in conditional_nan_fields.items()
        },
        "matched_conditional_nan_counts": matched_conditional_nan_counts,
        "ignored_policy_columns": ["raw_row_weight", "endpoint_weight", "sigma_eff_original"],
        "ignored_policy_reason": (
            "policy annotation columns are not consumed by absolute_structure_diagnostic; sigma_eff itself must "
            "still match exactly within tolerance"
        ),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "split_mismatch_count": exact_mismatch_counts["split"],
        "exact_mismatch_counts": exact_mismatch_counts,
        "numeric_mismatch_counts": numeric_mismatch_counts,
        "nonfinite_counts": nonfinite_counts,
        "mismatch_examples": mismatch_examples,
        "errors": errors,
    }
