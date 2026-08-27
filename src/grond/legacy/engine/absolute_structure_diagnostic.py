#!/usr/bin/env python3
"""Compare two native row tables with an alternate-hand diagnostic.

This is intentionally row-level. It does not pretend to be a SHELXL Flack
parameter or a merged absolute-structure statistic. It asks whether the same
signed observations have a lower scaled-residual sum for one selected native
prediction than for the alternate-hand check after each branch has had its
declared nuisance scales applied.  The default is the Bloch prediction;
``--prediction-column born_scaled`` provides the matched Born control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KEY_COLUMNS = ("row_index", "h", "k", "l")
REQUIRED_COLUMNS = KEY_COLUMNS
BASE_PREDICTION_COLUMNS = ("target_I",)
SOURCE_SIGMA_COLUMNS = ("source_sigma", "xds_sigma", "sigma_eff")
SUPPORTED_PREDICTION_COLUMNS = ("dyn_scaled", "born_scaled")
ADJUSTED_HAND_METHOD = "klar_palatinus_adjusted_reflection_count_hand_z"
ADJUSTED_HAND_METHOD_VERSION = "3"
ADJUSTED_HAND_TIE_POLICY = "strict_lower_residual_klar_2023"
KLAR_PALATINUS_REFERENCE = {
    "citation": (
        "Klar, P. B. et al. Accurate structure models and absolute configuration determination using "
        "dynamical effects in continuous-rotation 3D electron diffraction data. Nature Chemistry 15, "
        "848-855 (2023)."
    ),
    "doi": "10.1038/s41557-023-01186-1",
}


def adjusted_hand_method_metadata(prediction_column: str) -> dict[str, str]:
    """Return versioned adjusted-hand method and exact-tie semantics."""

    return {
        "method": ADJUSTED_HAND_METHOD,
        "method_version": ADJUSTED_HAND_METHOD_VERSION,
        "tie_policy": ADJUSTED_HAND_TIE_POLICY,
        "prediction_column": prediction_column,
    }


def finite_float(value: object) -> float:
    """Return a finite float or NaN."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a native row CSV with DictReader."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return list(reader), fields


def missing_columns(fields: list[str], prediction_column: str) -> list[str]:
    """Return key columns absent from a CSV header.

    Legacy dynamic residual tables may contain ``dyn_residual_over_sigma`` but
    not the intensity columns needed for the adjusted hand statistic.  Those
    remain useful for the paired residual diagnostic, so do not reject their
    row join here.  The adjusted diagnostic reports its own explicit missing
    intensity-column contract below.
    """

    del prediction_column
    present = set(fields)
    return [name for name in REQUIRED_COLUMNS if name not in present]


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    """Return the string key used to identify a signed observation row."""

    return tuple(str(row.get(name, "")).strip() for name in KEY_COLUMNS)


def indexed_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, int], dict[str, str]]:
    """Index rows by key plus duplicate occurrence to avoid many-to-many joins."""

    seen: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    out = {}
    for row in rows:
        key = row_key(row)
        occurrence = seen[key]
        seen[key] += 1
        out[(*key, occurrence)] = row
    return out


def duplicate_count(rows: list[dict[str, str]]) -> int:
    """Return the number of rows whose row key is duplicated."""

    counts = Counter(row_key(row) for row in rows)
    return int(sum(count - 1 for count in counts.values() if count > 1))


def normal_cdf(value: float) -> float:
    """Return the standard-normal cumulative distribution function."""

    return float(0.5 * math.erfc(-value / math.sqrt(2.0)))


def same_observation(left: float, right: float) -> bool:
    """Return whether two branch tables carry the same observation metadata."""

    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def adjusted_hand_missing_columns(
    best_fields: list[str],
    other_fields: list[str],
    prediction_column: str,
) -> dict[str, list[str]]:
    """Return columns missing for the adjusted hand-Z diagnostic."""

    required = (*BASE_PREDICTION_COLUMNS, prediction_column)
    missing = {
        "selected": [name for name in required if name not in set(best_fields)],
        "other": [name for name in required if name not in set(other_fields)],
    }
    if not any(name in set(best_fields) and name in set(other_fields) for name in SOURCE_SIGMA_COLUMNS):
        label = "one common source sigma column: " + "|".join(SOURCE_SIGMA_COLUMNS)
        missing["selected"].append(label)
        missing["other"].append(label)
    return missing


def adjusted_hand_sigma_column(rows: list[tuple[dict[str, str], dict[str, str]]]) -> str:
    """Select one immutable observation-sigma column shared by both hands."""

    if not rows:
        return ""
    selected, other = rows[0]
    return next(
        (
            name
            for name in SOURCE_SIGMA_COLUMNS
            if name in selected and name in other
        ),
        "",
    )


def klar_palatinus_adjusted_hand_z(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    missing: dict[str, list[str]],
    prediction_column: str,
) -> dict[str, Any]:
    """Compute the adjusted reflection-count hand Z on matched native rows.

    The selected (``best``) and other-hand tables must carry identical observed
    intensities and immutable source sigmas. Branch-specific values from the selected
    prediction column are then compared on exactly the joined signed
    observations.  Rows with invalid numbers, non-positive sigmas, or
    inconsistent observation metadata are excluded explicitly rather than
    silently borrowing values across branches.
    """

    base: dict[str, Any] = {
        **adjusted_hand_method_metadata(prediction_column),
        "n_joined_candidates": int(len(rows)),
        "required_columns": [
            *BASE_PREDICTION_COLUMNS,
            prediction_column,
            "one common source sigma: " + "|".join(SOURCE_SIGMA_COLUMNS),
        ],
    }
    if missing["selected"] or missing["other"]:
        return {
            **base,
            "status": "missing_required_columns",
            "missing_columns": missing,
            "n": 0,
        }

    sigma_column = adjusted_hand_sigma_column(rows)
    valid: list[tuple[float, float, float, float]] = []
    invalid_numeric = 0
    nonpositive_sigma = 0
    observation_mismatch = 0
    sigma_mismatch = 0
    dataset_mismatch = 0
    datasets: set[str] = set()
    for selected, other in rows:
        target_selected = finite_float(selected.get("target_I"))
        target_other = finite_float(other.get("target_I"))
        calc_selected = finite_float(selected.get(prediction_column))
        calc_other = finite_float(other.get(prediction_column))
        sigma_selected = finite_float(selected.get(sigma_column))
        sigma_other = finite_float(other.get(sigma_column))
        values = (target_selected, target_other, calc_selected, calc_other, sigma_selected, sigma_other)
        if not all(math.isfinite(value) for value in values):
            invalid_numeric += 1
            continue
        if sigma_selected <= 0.0 or sigma_other <= 0.0:
            nonpositive_sigma += 1
            continue
        if not same_observation(target_selected, target_other):
            observation_mismatch += 1
            continue
        if not same_observation(sigma_selected, sigma_other):
            sigma_mismatch += 1
            continue
        selected_dataset = str(selected.get("dataset", "")).strip()
        other_dataset = str(other.get("dataset", "")).strip()
        if selected_dataset and other_dataset and selected_dataset != other_dataset:
            dataset_mismatch += 1
            continue
        dataset = selected_dataset or other_dataset
        if dataset:
            datasets.add(dataset)
        valid.append((target_selected, calc_selected, calc_other, sigma_selected))

    exclusions = {
        "invalid_numeric": int(invalid_numeric),
        "nonpositive_sigma": int(nonpositive_sigma),
        "target_I_mismatch_between_hands": int(observation_mismatch),
        "source_sigma_mismatch_between_hands": int(sigma_mismatch),
        "dataset_label_mismatch_between_hands": int(dataset_mismatch),
    }
    if len(datasets) > 1:
        return {
            **base,
            "status": "multiple_datasets_not_pooled",
            "n": 0,
            "dataset_labels": sorted(datasets),
            "exclusions": exclusions,
            "combination_policy": (
                "No pooled multi-dataset Z is computed implicitly. Compute one adjusted Z per physical dataset, "
                "then combine only with a declared independent-dataset method."
            ),
        }

    n = len(valid)
    if n == 0:
        return {
            **base,
            "status": "no_valid_rows",
            "n": 0,
            "dataset_labels": sorted(datasets),
            "dataset_scope": "single_labeled_dataset" if datasets else "unlabeled_table",
            "exclusions": exclusions,
        }

    k = 0
    tied = 0
    w = 0.0
    for target, calc_selected, calc_other, sigma in valid:
        residual_selected = abs(target - calc_selected)
        residual_other = abs(target - calc_other)
        if residual_selected < residual_other:
            k += 1
        elif residual_selected == residual_other:
            tied += 1
        separation = abs(calc_selected - calc_other) / (2.0 * sigma)
        w += 1.0 - normal_cdf(separation)

    denominator_variance = float(n - w)
    k_effective = float(k)
    z = float((2 * k - n) / math.sqrt(denominator_variance))
    k_neutral = float(k + 0.5 * tied)
    z_neutral_ties = float((2.0 * k_neutral - n) / math.sqrt(denominator_variance))
    probability = normal_cdf(z)
    probability_neutral_ties = normal_cdf(z_neutral_ties)
    return {
        **base,
        "status": "ok",
        "n": int(n),
        "k_selected_lower_abs_intensity_residual": int(k),
        "n_tied_abs_intensity_residual": int(tied),
        "k_effective_selected_wins_plus_half_ties": k_effective,
        "k_neutral_half_tie_diagnostic": k_neutral,
        "w": float(w),
        "n_minus_w": denominator_variance,
        "z": z,
        "probability_phi_z": probability,
        "probability_one_sided_other_hand": float(normal_cdf(-z)),
        "z_neutral_half_tie_diagnostic": z_neutral_ties,
        "probability_phi_z_neutral_half_tie_diagnostic": probability_neutral_ties,
        "probability_one_sided_other_hand_neutral_half_tie_diagnostic": float(
            normal_cdf(-z_neutral_ties)
        ),
        "sigma_column": sigma_column,
        "sigma_contract": (
            "experimental source sigma(Iobs), preferring source_sigma then xds_sigma; "
            "sigma_eff is accepted only for historical row tables lacking an immutable source column"
        ),
        "dataset_labels": sorted(datasets),
        "dataset_scope": "single_labeled_dataset" if datasets else "unlabeled_table",
        "exclusions": exclusions,
    }


def joined_rows(best: list[dict[str, str]], other: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return one-to-one joined rows shared by both tables."""

    best_by_key = indexed_rows(best)
    other_by_key = indexed_rows(other)
    keys = sorted(set(best_by_key) & set(other_by_key))
    return [(best_by_key[key], other_by_key[key]) for key in keys]


def block(
    label: str,
    rows: list[tuple[dict[str, str], dict[str, str]]],
    adjusted_missing: dict[str, list[str]],
    prediction_column: str,
) -> dict[str, Any]:
    """Compute paired row-level contrast statistics."""

    adjusted = klar_palatinus_adjusted_hand_z(rows, adjusted_missing, prediction_column)
    rb: list[float] = []
    ro: list[float] = []
    for best, other in rows:
        target_best = finite_float(best.get("target_I"))
        target_other = finite_float(other.get("target_I"))
        calc_best = finite_float(best.get(prediction_column))
        calc_other = finite_float(other.get(prediction_column))
        sigma_best = finite_float(best.get("sigma_eff"))
        sigma_other = finite_float(other.get("sigma_eff"))
        if (
            math.isfinite(target_best)
            and math.isfinite(target_other)
            and math.isfinite(calc_best)
            and math.isfinite(calc_other)
            and math.isfinite(sigma_best)
            and math.isfinite(sigma_other)
            and sigma_best > 0.0
            and sigma_other > 0.0
        ):
            b = (target_best - calc_best) / sigma_best
            o = (target_other - calc_other) / sigma_other
            rb.append(b)
            ro.append(o)
        elif prediction_column == "dyn_scaled":
            # Backwards-compatible support for legacy row tables that store
            # only the already-standardized dynamical residual.  This fallback
            # is deliberately unavailable to the Born control because a
            # dynamical residual must never be relabelled as Born evidence.
            b = finite_float(best.get("dyn_residual_over_sigma"))
            o = finite_float(other.get("dyn_residual_over_sigma"))
            if math.isfinite(b) and math.isfinite(o):
                rb.append(b)
                ro.append(o)
    n = len(rb)
    if n == 0:
        return {"label": label, "n": 0, "klar_palatinus_adjusted_hand_z": adjusted}
    d = [o * o - b * b for b, o in zip(rb, ro, strict=True)]
    delta_chi2 = float(sum(d))
    mean_d = float(delta_chi2 / n)
    if n > 1:
        var = sum((x - mean_d) ** 2 for x in d) / (n - 1)
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = float("nan")
    paired_t = float(mean_d / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    return {
        "label": label,
        "n": n,
        "best_chi2": float(sum(x * x for x in rb)),
        "other_chi2": float(sum(x * x for x in ro)),
        "delta_chi2_other_minus_best": delta_chi2,
        "sqrt_positive_delta_chi2": float(math.sqrt(max(delta_chi2, 0.0))),
        "paired_row_t_delta_chi2": paired_t,
        "mean_row_delta_chi2": mean_d,
        "median_row_delta_chi2": float(
            sorted(d)[n // 2] if n % 2 else 0.5 * (sorted(d)[n // 2 - 1] + sorted(d)[n // 2])
        ),
        "best_lower_abs_residual_fraction": float(sum(abs(b) < abs(o) for b, o in zip(rb, ro, strict=True)) / n),
        "best_lower_squared_residual_fraction": float(sum(x > 0.0 for x in d) / n),
        "klar_palatinus_adjusted_hand_z": adjusted,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", required=True, help="Native row CSV for the active/best hand assignment.")
    ap.add_argument("--other", required=True, help="Native row CSV for the alternate-hand comparison.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument(
        "--prediction-column",
        choices=SUPPORTED_PREDICTION_COLUMNS,
        default="dyn_scaled",
        help="Branch prediction to compare; dyn_scaled is the Bloch default and born_scaled is the matched Born control.",
    )
    args = ap.parse_args()

    best_path = Path(args.best)
    other_path = Path(args.other)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_files = [str(path) for path in (best_path, other_path) if not path.exists()]
    if missing_files:
        out = {
            "status": "missing_input_files",
            "best_csv": str(best_path.resolve()),
            "other_csv": str(other_path.resolve()),
            "missing_files": missing_files,
            "row_join": {
                "key": list(KEY_COLUMNS),
                "n_best": 0,
                "n_other": 0,
                "n_joined": 0,
                "duplicate_keys_best": 0,
                "duplicate_keys_other": 0,
                "missing_columns": {"best": [], "other": []},
            },
            "interpretation": {
                "diagnostic": "row-level alternate-hand check was skipped because one or more native row CSV files were absent",
                "recommendation": "Treat this as an incomplete diagnostic artifact, not as evidence for either hand.",
            },
            "all": {
                "label": "all",
                "n": 0,
                "klar_palatinus_adjusted_hand_z": {
                    **adjusted_hand_method_metadata(args.prediction_column),
                    "status": "missing_input_files",
                    "n": 0,
                },
            },
            "work": {
                "label": "work",
                "n": 0,
                "klar_palatinus_adjusted_hand_z": {
                    **adjusted_hand_method_metadata(args.prediction_column),
                    "status": "missing_input_files",
                    "n": 0,
                },
            },
            "free": {
                "label": "free",
                "n": 0,
                "klar_palatinus_adjusted_hand_z": {
                    **adjusted_hand_method_metadata(args.prediction_column),
                    "status": "missing_input_files",
                    "n": 0,
                },
            },
        }
        out_path.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n")
        return

    best_rows, best_fields = read_rows(best_path)
    other_rows, other_fields = read_rows(other_path)
    missing = {
        "best": missing_columns(best_fields, args.prediction_column),
        "other": missing_columns(other_fields, args.prediction_column),
    }
    adjusted_missing = adjusted_hand_missing_columns(best_fields, other_fields, args.prediction_column)
    if missing["best"] or missing["other"]:
        joined: list[tuple[dict[str, str], dict[str, str]]] = []
        status = "missing_required_columns"
    else:
        joined = joined_rows(best_rows, other_rows)
        status = "ok" if joined else "no_common_rows"

    work = [(best, other) for best, other in joined if best.get("split") == "work"]
    free = [(best, other) for best, other in joined if best.get("split") == "free"]
    out = {
        "status": status,
        "best_csv": str(best_path.resolve()),
        "other_csv": str(other_path.resolve()),
        "row_join": {
            "key": list(KEY_COLUMNS),
            "n_best": int(len(best_rows)),
            "n_other": int(len(other_rows)),
            "n_joined": int(len(joined)),
            "duplicate_keys_best": duplicate_count(best_rows),
            "duplicate_keys_other": duplicate_count(other_rows),
            "missing_columns": missing,
            "klar_palatinus_adjusted_hand_z_missing_columns": adjusted_missing,
        },
        "interpretation": {
            "diagnostic": "row-level alternate-hand check; not a Flack parameter and not a merged absolute-structure statistic",
            "prediction_column": args.prediction_column,
            "delta_chi2": "sum((other residual/sigma)^2 - (best residual/sigma)^2); positive favors best",
            "paired_row_t": "paired t-style contrast of per-row delta_chi2 values",
            "sqrt_positive_delta_chi2": "square root of positive delta_chi2; a scale indicator, not a probability or hand-assignment score",
            "klar_palatinus_adjusted_hand_z": {
                "label": "adjusted reflection-count absolute-structure Z; positive favors the selected/best hand",
                "row_support": (
                    "exact one-to-one joined signed observations identified by row_index,h,k,l and duplicate occurrence"
                ),
                "k": (
                    "number of joined valid rows where |target_I-prediction_selected| is strictly smaller "
                    "than |target_I-prediction_other| for prediction_column"
                ),
                "k_effective": "strict k from Klar et al.; exact ties do not increment k",
                "w": "sum(1-Phi(|prediction_selected-prediction_other|/(2*source_sigma(Iobs))))",
                "z": "(2*k-N)/sqrt(N-w)",
                "probability": "Phi(z), reported as probability_phi_z; this is distinct from paired_row_t_delta_chi2",
                "ties": (
                    "the primary statistic follows Klar et al.'s strict inequality; a separate neutral-half-tie "
                    "diagnostic is retained only to expose the otherwise negligible exact-tie edge case"
                ),
                "tie_policy": ADJUSTED_HAND_TIE_POLICY,
                "sigma_contract": (
                    "target_I and positive immutable source sigma(Iobs) must match between hand tables; source_sigma "
                    "is preferred, xds_sigma is its historical alias, and sigma_eff is a legacy fallback only"
                ),
                "multi_dataset_policy": (
                    "multiple dataset labels are not silently row-pooled; compute per-dataset Z values before any explicitly declared combination"
                ),
                "reference": KLAR_PALATINUS_REFERENCE,
            },
        },
        "all": block("all", joined, adjusted_missing, args.prediction_column),
        "work": block("work", work, adjusted_missing, args.prediction_column),
        "free": block("free", free, adjusted_missing, args.prediction_column),
    }
    out_path.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
