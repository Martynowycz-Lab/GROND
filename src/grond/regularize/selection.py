"""Nuisance-trial ranking and existing promotion policy. These are heuristic validation rules, not an independent structural Rfree or a uniqueness proof."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .contracts import Nuisance


def metric_row(trial):
    m = trial["metrics"]
    n = trial["nuisance"]
    q_control = trial.get("q_control") or {}
    return {
        **n,
        "q_control": q_control.get("mode", "real"),
        "q_control_shuffled_rows": q_control.get("shuffled_rows", 0),
        "q_control_active_bins": q_control.get("active_bins", 0),
        "q_control_logq_corr": q_control.get("logq_corr_original_shuffled", float("nan")),
        "valid_rows": trial["valid_rows"],
        "scale_count": trial["scale_count"],
        "work_dyn_wR2": m["work"]["dyn"]["wR2_all"],
        "free_dyn_wR2": m["free"]["dyn"]["wR2_all"],
        "all_dyn_wR2": m["all"]["dyn"]["wR2_all"],
        "all_dyn_R1": m["all"]["dyn"]["R1_all"],
        "all_dyn_R1gt": m["all"]["dyn"]["R1_gt"],
        "all_born_wR2": m["all"]["born"]["wR2_all"],
        "all_born_R1": m["all"]["born"]["R1_all"],
        "all_dyn_minus_born_wR2": m["all"]["dyn"]["wR2_all"] - m["all"]["born"]["wR2_all"],
    }


def trial_metric(trial, metric_name: str) -> float:
    if metric_name == "work_born_wR2":
        return float(trial["metrics"]["work"]["born"]["wR2_all"])
    split, value = metric_name.split("_", 1)
    if split not in trial["metrics"]:
        raise ValueError(f"unknown selection split {split!r} in {metric_name!r}")
    key = "wR2_all" if value == "wR2" else "R1_all"
    return float(trial["metrics"][split]["dyn"][key])


def rank_trials(trials, metric_name: str):
    """Rank nuisance trials without using held-out rows as an implicit tie-breaker."""

    return sorted(trials, key=lambda trial: trial_metric(trial, metric_name))


_NUISANCE_CATEGORICAL_FIELDS = {
    "thickness_model",
    "sg_jitter_mode",
    "mosaicity_mode",
    "geometry_path_file",
    "absorption_mode",
    "dose_b_mode",
    "matrix_convention",
}


_NUISANCE_NULLABLE_NUMERIC_FIELDS = {
    "dose_b_ref_frame",
    "native_wght_a",
    "native_wght_b",
}


def nuisance_reference_matches(reference: dict, nuisance: dict) -> bool:
    """Return whether a trial reproduces an explicitly supplied nuisance state.

    The reference comes from the preceding automatic stage and may contain
    additional summary metadata.  Only fields in the frozen ``Nuisance``
    contract participate.  Matching NaNs are accepted for intentionally
    disabled terms such as ``dose_b_ref_frame``; one-sided NaNs and infinities
    are never treated as equal.
    """

    checked = 0
    for key in Nuisance.__dataclass_fields__:
        if key not in reference:
            continue
        checked += 1
        if key not in nuisance:
            return False
        expected = reference[key]
        actual = nuisance[key]
        if key in _NUISANCE_CATEGORICAL_FIELDS:
            if str(expected) != str(actual):
                return False
            continue
        try:
            expected_value = float(expected)
            actual_value = float(actual)
        except (TypeError, ValueError):
            return False
        if math.isnan(expected_value) and math.isnan(actual_value):
            if key in _NUISANCE_NULLABLE_NUMERIC_FIELDS:
                continue
            return False
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            return False
        if not math.isclose(expected_value, actual_value, rel_tol=1e-12, abs_tol=1e-12):
            return False
    return checked > 0


def _trial_metric_value(trial: dict, split: str, metric: str) -> float:
    try:
        return float(trial["metrics"][split]["dyn"][metric])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _trial_split_count(trial: dict, split: str) -> int:
    value = _trial_metric_value(trial, split, "n")
    return max(0, int(value)) if math.isfinite(value) else 0


def _trial_full_row_score(trial: dict) -> float:
    """Mirror the public native-summary score for an in-memory trial."""

    dyn_wr2 = _trial_metric_value(trial, "all", "wR2_all")
    free_wr2 = _trial_metric_value(trial, "free", "wR2_all")
    dyn_r1 = _trial_metric_value(trial, "all", "R1_all")
    try:
        born_r1gt = float(trial["metrics"]["all"]["born"]["R1_gt"])
        born_cc = float(trial["metrics"]["all"]["born"]["CC"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    values = (dyn_wr2, free_wr2, dyn_r1, born_r1gt, born_cc)
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return (
        dyn_wr2
        + 0.25 * free_wr2
        + 0.10 * dyn_r1
        + max(0.0, born_r1gt - 0.35)
        + max(0.0, 0.5 - born_cc)
    )


def _trial_valid_mask(trial: dict) -> np.ndarray | None:
    arrays = trial.get("arrays")
    if not isinstance(arrays, dict) or "valid" not in arrays:
        return None
    mask = np.asarray(arrays["valid"])
    if mask.ndim != 1:
        return None
    return np.asarray(mask, dtype=bool)


def _valid_mask_sha256(mask: np.ndarray | None) -> str:
    if mask is None:
        return ""
    payload = np.asarray(mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode())
    digest.update(b"\0")
    digest.update(payload.tobytes())
    return digest.hexdigest()


def _promotion_trial_record(trial: dict, selection_metric: str) -> dict:
    return {
        "nuisance": dict(trial.get("nuisance") or {}),
        "selection_metric": str(selection_metric),
        "selection_value": trial_metric(trial, selection_metric),
        "valid_rows": int(trial.get("valid_rows", 0)),
        "scale_count": int(trial.get("scale_count", 0)),
        "scale_fit_split": str(trial.get("scale_fit_split", "")),
        "metrics": trial.get("metrics") or {},
    }


def select_trial_with_promotion_gate(
    ranked_trials: list[dict],
    baseline_nuisance: dict,
    *,
    selection_metric: str,
    min_gain: float,
    max_free_wr2_loss: float,
    max_all_r1_loss: float,
) -> tuple[dict, dict]:
    """Gate one work-ranked nuisance winner against the exact starting trial.

    Ranking remains blind to held-out rows.  Only after the single work winner
    is known do free rows act as a no-harm validation gate.  Both candidates
    were already evaluated in the same grid with the same model, observations,
    split, and scale policy, so rejecting a non-generalizing winner adds no
    Bloch calculation and restores the actual baseline arrays.
    """

    if not ranked_trials:
        raise RuntimeError("held-out promotion gate received no nuisance trials")
    if str(selection_metric) != "work_wR2":
        raise RuntimeError(
            "held-out promotion gate requires work_wR2 ranking; refusing a selection that used free or all rows"
        )
    thresholds = {
        "min_gain": float(min_gain),
        "max_free_wr2_loss": float(max_free_wr2_loss),
        "max_all_r1_loss": float(max_all_r1_loss),
    }
    invalid_thresholds = [
        name for name, value in thresholds.items() if not math.isfinite(value) or value < 0.0
    ]
    if invalid_thresholds:
        raise RuntimeError(
            "held-out promotion thresholds must be finite and nonnegative: "
            + ", ".join(invalid_thresholds)
        )
    matches = [
        trial
        for trial in ranked_trials
        if nuisance_reference_matches(baseline_nuisance, trial.get("nuisance") or {})
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "held-out promotion gate could not identify exactly one starting-nuisance trial "
            f"(matches={len(matches)}); refusing to promote a work-only winner"
        )

    candidate = ranked_trials[0]
    baseline = matches[0]
    nonwork_scale_trials = [
        label
        for label, trial in (("baseline", baseline), ("candidate", candidate))
        if str(trial.get("scale_fit_split", "")) != "work"
    ]
    if nonwork_scale_trials:
        raise RuntimeError(
            "held-out promotion gate requires work-only scale fitting for "
            + ", ".join(nonwork_scale_trials)
        )
    candidate_is_baseline = candidate is baseline
    split_names = ("all", "work", "free")
    baseline_counts = {split: _trial_split_count(baseline, split) for split in split_names}
    candidate_counts = {split: _trial_split_count(candidate, split) for split in split_names}
    baseline_value = trial_metric(baseline, selection_metric)
    candidate_value = trial_metric(candidate, selection_metric)
    baseline_free = _trial_metric_value(baseline, "free", "wR2_all")
    candidate_free = _trial_metric_value(candidate, "free", "wR2_all")
    baseline_all_r1 = _trial_metric_value(baseline, "all", "R1_all")
    candidate_all_r1 = _trial_metric_value(candidate, "all", "R1_all")
    baseline_all_wr2 = _trial_metric_value(baseline, "all", "wR2_all")
    candidate_all_wr2 = _trial_metric_value(candidate, "all", "wR2_all")
    baseline_full_row_score = _trial_full_row_score(baseline)
    candidate_full_row_score = _trial_full_row_score(candidate)
    baseline_valid_mask = _trial_valid_mask(baseline)
    candidate_valid_mask = _trial_valid_mask(candidate)
    gain = baseline_value - candidate_value
    free_wr2_loss = candidate_free - baseline_free
    all_r1_loss = candidate_all_r1 - baseline_all_r1
    all_wr2_loss = candidate_all_wr2 - baseline_all_wr2
    full_row_score_gain = baseline_full_row_score - candidate_full_row_score

    reasons: list[str] = []
    for split in split_names:
        if baseline_counts[split] <= 0:
            reasons.append(f"baseline_missing_positive_{split}_rows")
        if candidate_counts[split] <= 0:
            reasons.append(f"candidate_missing_positive_{split}_rows")
        if baseline_counts[split] != candidate_counts[split]:
            reasons.append(f"{split}_row_count_mismatch")
    finite_values = (
        baseline_value,
        candidate_value,
        baseline_free,
        candidate_free,
        baseline_all_r1,
        candidate_all_r1,
        baseline_all_wr2,
        candidate_all_wr2,
        baseline_full_row_score,
        candidate_full_row_score,
    )
    valid_row_mismatch_count: int | None = None
    if baseline_valid_mask is None:
        reasons.append("baseline_missing_valid_row_mask")
    if candidate_valid_mask is None:
        reasons.append("candidate_missing_valid_row_mask")
    if baseline_valid_mask is not None and candidate_valid_mask is not None:
        if baseline_valid_mask.shape != candidate_valid_mask.shape:
            reasons.append("valid_row_mask_shape_mismatch")
        else:
            valid_row_mismatch_count = int(np.count_nonzero(baseline_valid_mask != candidate_valid_mask))
            if valid_row_mismatch_count:
                reasons.append("valid_row_identity_mismatch")
    if not all(math.isfinite(value) for value in finite_values):
        reasons.append("missing_finite_promotion_metrics")
    elif not candidate_is_baseline:
        if gain < thresholds["min_gain"]:
            reasons.append("gain_below_minimum")
        if full_row_score_gain < thresholds["min_gain"]:
            reasons.append("full_row_score_gain_below_minimum")
        if free_wr2_loss > thresholds["max_free_wr2_loss"]:
            reasons.append("free_wR2_loss")
        if all_r1_loss > thresholds["max_all_r1_loss"]:
            reasons.append("all_R1_loss")

    promoted = bool(not candidate_is_baseline and not reasons)
    selected = candidate if promoted else baseline
    status = "accepted" if promoted else ("baseline_already_best" if candidate_is_baseline else "rejected")
    decision = {
        "family": "discrete_thickness_polish",
        "status": status,
        "promoted": promoted,
        "candidate_was_baseline": candidate_is_baseline,
        "selection_metric": str(selection_metric),
        "objective_gain": gain,
        "free_wR2_loss": free_wr2_loss,
        "all_R1_loss": all_r1_loss,
        "all_wR2_loss": all_wr2_loss,
        "baseline_full_row_score": baseline_full_row_score,
        "candidate_full_row_score": candidate_full_row_score,
        "full_row_score_gain": full_row_score_gain,
        "min_promotion_gain": thresholds["min_gain"],
        "max_free_wR2_loss": thresholds["max_free_wr2_loss"],
        "max_all_R1_loss": thresholds["max_all_r1_loss"],
        "baseline_row_counts": baseline_counts,
        "candidate_row_counts": candidate_counts,
        "baseline_valid_row_mask_sha256": _valid_mask_sha256(baseline_valid_mask),
        "candidate_valid_row_mask_sha256": _valid_mask_sha256(candidate_valid_mask),
        "valid_row_mismatch_count": valid_row_mismatch_count,
        "promotion_reasons": list(dict.fromkeys(reasons)),
        "ranked_candidate": _promotion_trial_record(candidate, selection_metric),
        "baseline": _promotion_trial_record(baseline, selection_metric),
        "selected": _promotion_trial_record(selected, selection_metric),
    }
    return selected, decision
