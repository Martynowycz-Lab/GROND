"""Combine matched hand evidence and explain weak or conflicting diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from grond.workflows.decision import safe_float


def _adjusted_hand_z_block(stats: Mapping[str, object], split: str) -> dict[str, object]:
    """Return the declared Klar--Palatinus adjusted-Z block for one split."""

    support = stats.get(split) if isinstance(stats, Mapping) else {}
    support = support if isinstance(support, Mapping) else {}
    adjusted = support.get("klar_palatinus_adjusted_hand_z")
    return dict(adjusted) if isinstance(adjusted, Mapping) else {}


def _adjusted_hand_blocks_match(
    recorded: Mapping[str, object],
    recomputed: Mapping[str, object],
    *,
    allow_unavailable: bool = False,
) -> bool:
    """Replay finite statistics, allowing matching absent diagnostic splits only."""

    if recorded.get("status") != recomputed.get("status") or int(recorded.get("n") or 0) != int(
        recomputed.get("n") or 0
    ):
        return False
    left = safe_float(recorded.get("z"), float("nan"))
    right = safe_float(recomputed.get("z"), float("nan"))
    if math.isfinite(left) and math.isfinite(right):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14)
    return bool(
        allow_unavailable
        and math.isnan(left)
        and math.isnan(right)
        and recorded.get("status") != "ok"
        and int(recorded.get("n") or 0) == 0
    )


def _combine_adjusted_hand_z(
    diagnostics: Iterable[Mapping[str, object]],
    split: str,
) -> dict[str, object]:
    """Combine independent-dataset adjusted hand counts with a declared rule.

    Each physical dataset contributes its signed reflection-count numerator and
    adjusted variance.  The result is positive when the currently evaluated
    coordinate hand is favored and negative when its inverse is favored.
    """

    numerator = 0.0
    variance = 0.0
    rows = 0
    datasets = 0
    excluded: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        adjusted = _adjusted_hand_z_block(diagnostic, split)
        n = int(adjusted.get("n", 0) or 0)
        k_effective = safe_float(
            adjusted.get(
                "k_effective_selected_wins_plus_half_ties",
                adjusted.get("k_selected_lower_abs_intensity_residual"),
            ),
            float("nan"),
        )
        w = safe_float(adjusted.get("w"), float("nan"))
        if (
            str(adjusted.get("status") or "") != "ok"
            or n <= 0
            or not math.isfinite(k_effective)
            or not math.isfinite(w)
            or n - w <= 0.0
        ):
            excluded.append(str(index))
            continue
        numerator += 2.0 * k_effective - n
        variance += n - w
        rows += n
        datasets += 1
    z = numerator / math.sqrt(variance) if datasets and variance > 0.0 else float("nan")
    return {
        "status": "ok" if math.isfinite(z) else "unavailable",
        "z": z,
        "n": rows,
        "datasets": datasets,
        "signed_numerator": numerator,
        "adjusted_variance": variance,
        "excluded_dataset_indices": excluded,
        "positive_favors_current_coordinates": True,
        "combination_rule": "sum(2*k_effective-n)/sqrt(sum(n-w)) across physical datasets",
        "confidence_interpretation": (
            "Conditional count contrast; nominal normal scaling assumes independent observations. "
            "Shared nuisance/model fitting and repeated rows can invalidate calibrated sigma claims."
        ),
        "probability_is_calibrated": False,
    }


def hand_diagnostic_supports_other(stats: Mapping[str, object], args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Use the all-row adjusted ZKP to decide whether to invert the model.

    The held-out split is deliberately diagnostic: a small or conflicting free
    Z emits a warning but does not veto a decisive all-row statistic.
    """

    threshold = float(getattr(args, "hand_flip_zkp_all", 3.5))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("hand inversion threshold must be finite and positive")
    all_adjusted = _adjusted_hand_z_block(stats, "all")
    all_z = safe_float(all_adjusted.get("z"), float("nan"))
    reasons: list[str] = []
    if stats.get("calculation_status") == "error" or stats.get("status") == "calculation_error":
        reasons.extend(str(error) for error in stats.get("calculation_errors", []))
        if not reasons:
            reasons.append("all-row hand calculation has an explicit error")
    if str(all_adjusted.get("status") or "") != "ok" or not math.isfinite(all_z):
        reasons.append("all-row Klar-Palatinus adjusted hand Z is unavailable")
    elif all_z > -threshold:
        reasons.append(
            f"all-row Klar-Palatinus adjusted hand Z {all_z:.4g} does not favor inversion "
            f"at the {-threshold:.4g} decision boundary"
        )
    return not reasons, reasons


def hand_preference_summary(
    all_z: float,
    *,
    active_hand: str,
    threshold: float,
    calculation_valid: bool = True,
) -> dict[str, object]:
    """Report the measured preference separately from the exported representation."""

    valid = bool(calculation_valid and math.isfinite(all_z))
    opposite = "alternate" if active_hand == "original" else "original"
    preferred = (
        active_hand
        if valid and all_z > 0.0
        else opposite
        if valid and all_z < 0.0
        else "no_preference"
        if valid
        else "unavailable"
    )
    confidence = (
        "calculation_error"
        if not valid
        else "no_sensitivity"
        if all_z == 0.0
        else "strong_conditional"
        if abs(all_z) >= threshold
        else "weak_conditional"
    )
    return {
        "preferred_hand": preferred,
        "hand_confidence": confidence,
        "all_row_adjusted_z_positive_favors_active": all_z if valid else None,
        "active_hand_before_selection": active_hand,
        "primary_evidence": "all_rows",
        "free_evidence_role": "diagnostic_warning_only",
        "statistic_is_calibrated_probability": False,
    }


def _hand_evidence_warnings(
    *,
    all_z: float,
    free_z: float,
    switched: bool,
    threshold: float,
) -> list[str]:
    """Warn when final-hand support is weak, conflicting, or not held out."""

    sign = -1.0 if switched else 1.0
    oriented_all = sign * all_z if math.isfinite(all_z) else float("nan")
    oriented_free = sign * free_z if math.isfinite(free_z) else float("nan")
    warnings: list[str] = [
        "ZKP is a conditional reflection-count contrast, not calibrated sigma confidence; "
        "shared scales, repeated observations, and model selection can correlate rows."
    ]
    if not math.isfinite(oriented_all):
        warnings.append("all-row Klar-Palatinus adjusted hand Z is unavailable")
    elif oriented_all < threshold:
        warnings.append(f"final-hand all-row ZKP is {oriented_all:.3g}, below the decisive threshold {threshold:.3g}")
    if not math.isfinite(oriented_free):
        warnings.append("held-out Klar-Palatinus adjusted hand Z is unavailable")
    elif oriented_free < 0.0:
        warnings.append(f"held-out ZKP {oriented_free:.3g} conflicts with the all-row selected hand")
    elif oriented_free < threshold:
        warnings.append(f"held-out ZKP {oriented_free:.3g} supports the selected hand only weakly (< {threshold:.3g})")
    return warnings


def _pooled_hand_pair_statistics(
    pairs: list[tuple[str, Path, Path]],
) -> dict[str, dict[str, object]]:
    """Pool exact per-row hand contrasts across crystals without merging rows."""

    deltas: dict[str, list[float]] = {"all": [], "work": [], "free": []}
    for dataset, selected_path, inverse_path in pairs:
        with selected_path.open(newline="") as handle:
            selected_rows = list(csv.DictReader(handle))
        with inverse_path.open(newline="") as handle:
            inverse_rows = {
                (row.get("row_index"), row.get("h"), row.get("k"), row.get("l")): row for row in csv.DictReader(handle)
            }
        for selected in selected_rows:
            key = (
                selected.get("row_index"),
                selected.get("h"),
                selected.get("k"),
                selected.get("l"),
            )
            inverse = inverse_rows.get(key)
            if inverse is None:
                raise ValueError(f"{dataset}: inverse hand table lacks row key {key}")
            selected_residual = float(selected["dyn_residual_over_sigma"])
            inverse_residual = float(inverse["dyn_residual_over_sigma"])
            if not math.isfinite(selected_residual) or not math.isfinite(inverse_residual):
                raise ValueError(f"{dataset}: nonfinite paired residual at row key {key}")
            delta = inverse_residual * inverse_residual - selected_residual * selected_residual
            split = str(selected.get("split") or "")
            deltas["all"].append(delta)
            if split in {"work", "free"}:
                deltas[split].append(delta)

    result: dict[str, dict[str, object]] = {}
    for split, values in deltas.items():
        n_rows = len(values)
        mean = math.fsum(values) / n_rows if n_rows else float("nan")
        if n_rows > 1:
            variance = math.fsum((value - mean) ** 2 for value in values) / (n_rows - 1)
            t_style = mean / math.sqrt(variance / n_rows) if variance > 0.0 else math.copysign(float("inf"), mean)
        else:
            t_style = float("nan")
        result[split] = {
            "n": n_rows,
            "mean_delta_chi2_inverse_minus_selected": mean,
            "paired_row_t_delta_chi2": t_style,
            "positive_favors_selected": True,
        }
    return result
