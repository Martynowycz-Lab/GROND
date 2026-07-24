"""Reproducible row-artifact evidence for transform and hand decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from grond.legacy.engine.absolute_structure_diagnostic import (
    adjusted_hand_missing_columns,
    klar_palatinus_adjusted_hand_z,
)
from grond.legacy.engine.native_row_regularize import (
    Nuisance,
    fit_frame_detector_scales,
    fit_group_scales,
    native_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hashable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_hashable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if math.isnan(float(value)):
            label = "nan"
        elif float(value) > 0:
            label = "+inf"
        else:
            label = "-inf"
        return {"__grond_nonfinite_float__": label}
    return value


def _json_safe_contract(value: object, *, field: str = "") -> object:
    """Return a strict-JSON representation of a replay contract.

    Nullable numerical nuisance fields use ``None`` in persisted/public
    payloads.  Their canonical candidate representation remains the string
    ``"nan"`` so hashing and CSV replay do not conflate the sentinel with an
    ordinary missing key.  Non-finite values anywhere else fail closed instead
    of being disguised as missing data.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_contract(item, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_contract(item, field=field) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if field in _NUISANCE_NULLABLE_FIELDS and math.isnan(float(value)):
            return None
        raise ValueError(
            f"non-finite value is forbidden in strict-JSON contract field "
            f"{field!r}: {value!r}"
        )
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_hashable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _read_summary_and_rows(
    summary_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, str]], dict[str, str]]:
    summary_path = Path(summary_path).expanduser().resolve()
    if not summary_path.is_file():
        raise ValueError(f"native summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"native summary must contain one JSON object: {summary_path}")
    artifacts = summary.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    row_path = Path(str(artifacts.get("native_row_table") or "")).expanduser().resolve()
    if not row_path.is_file():
        raise ValueError(f"native row table is missing: {row_path}")
    row_sha256 = _sha256(row_path)
    if row_sha256 != str(artifacts.get("native_row_table_sha256") or ""):
        raise ValueError(f"native row-table hash mismatch: {row_path}")
    with row_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "row_index",
        "h",
        "k",
        "l",
        "target_I",
        "sigma_eff",
        "xds_sigma",
        "group",
        "detector_bin",
        "split",
        "dyn_raw",
        "born_raw",
        "dyn_scaled",
        "born_scaled",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(
            f"native row table {row_path} lacks required evidence columns: "
            + ", ".join(missing)
        )
    binding = {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "row_table_path": str(row_path),
        "row_table_sha256": row_sha256,
    }
    return summary, row_path, rows, binding


def _declared_nuisance_contract(config: Mapping[str, object]) -> dict[str, str]:
    mapping = {
        "thickness_A": "thickness_grid",
        "thickness_model": "thickness_model_grid",
        "thickness_grad_frac": "thickness_grad_grid",
        "thickness_sigma_frac": "thickness_sigma_frac_grid",
        "frame_offset": "frame_offset_grid",
        "group_width_deg": "group_width_grid",
        "detector_scale_bins": "detector_scale_bins",
        "sigma_a": "sigma_a_grid",
        "sigma_b_corr": "sigma_b_corr_grid",
        "sigma_c_peak": "sigma_c_peak_grid",
        "sg_jitter_mode": "sg_jitter_mode_grid",
        "sg_jitter_sigma0": "sg_jitter_sigma0_grid",
        "sg_jitter_sigma_d_inv": "sg_jitter_sigma_d_inv_grid",
        "sg_jitter_sigma_frame": "sg_jitter_sigma_frame_grid",
        "absorption_mode": "absorption_mode_grid",
        "absorption_mu": "absorption_mu_grid",
        "absorption_frac": "absorption_frac_grid",
        "absorption_mu_g2": "absorption_mu_g2_grid",
        "absorption_mu_theta2": "absorption_mu_theta2_grid",
        "absorption_tds_b": "absorption_tds_b_grid",
        "dose_b_mode": "dose_b_mode_grid",
        "dose_b_per_frame": "dose_b_per_frame_grid",
        "row_resolution_b": "row_resolution_b_grid",
    }
    return {
        output: str(config.get(source) or "").strip()
        for output, source in mapping.items()
    }


def _grid_physics_contract(config: Mapping[str, object]) -> dict[str, object]:
    """Return physics settings shared by every candidate in one grid run."""

    fields = (
        "n_basis",
        "n_points",
        "sg_window",
        "sg_window_mode",
        "sg_window_min",
        "sg_window_max",
        "sg_window_scale",
        "sg_window_frame_floor",
        "sg_jitter_points",
        "sg_jitter_student_df",
        "sg_jitter_width_factor",
        "profile_quadrature",
        "profile_accumulation",
        "min_d_spacing",
        "min_peak",
        "geometry_percentile",
        "target_mode",
        "positive_only",
        "thickness_points",
        "topup_d_min",
        "max_support_sg",
        "free_fraction",
        "split_seed",
        "scale_fit_split",
        "selection_metric",
        "compute_backend",
        "jax_batch_size",
        "jax_matrix_bucket_size",
        "bethe_strong_count",
        "bethe_variant",
        "bethe_promote_denom",
        "bethe_denom_eps",
        "matrix_convention",
        "row_plan_cache",
        "max_rows",
    )
    return {field: config.get(field) for field in fields}


def _job_contract(summary: Mapping[str, object]) -> dict[str, object]:
    inputs = summary.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    config = summary.get("config")
    config = config if isinstance(config, Mapping) else {}
    row_selection = summary.get("row_selection")
    row_selection = row_selection if isinstance(row_selection, Mapping) else {}
    split = row_selection.get("transform_hand_split_contract")
    split = split if isinstance(split, Mapping) else {}
    best = summary.get("best_native")
    best = best if isinstance(best, Mapping) else {}
    nuisance = best.get("nuisance")
    nuisance = nuisance if isinstance(nuisance, Mapping) else {}
    kernel_contract = summary.get("bloch_kernel_contract")
    kernel_contract = (
        kernel_contract if isinstance(kernel_contract, Mapping) else {}
    )

    hkl_path = Path(str(inputs.get("hkl") or "")).expanduser().resolve()
    model_path = Path(str(inputs.get("res") or "")).expanduser().resolve()
    if not hkl_path.is_file() or _sha256(hkl_path) != str(
        inputs.get("hkl_sha256") or ""
    ):
        raise ValueError("native summary source-HKL identity is missing or stale")
    if not model_path.is_file() or _sha256(model_path) != str(
        inputs.get("res_sha256") or ""
    ):
        raise ValueError("native summary source-model identity is missing or stale")
    split_path_text = str(config.get("split_contract") or "").strip()
    split_path = (
        Path(split_path_text).expanduser().resolve()
        if split_path_text
        else None
    )
    split_sha256 = ""
    if split_path is not None:
        if not split_path.is_file():
            raise ValueError("native summary split-contract path is stale")
        split_sha256 = _sha256(split_path)
        if split_sha256 != str(split.get("sha256") or ""):
            raise ValueError(
                "native summary split-contract hash does not match row selection"
            )
    declared_nuisance = _declared_nuisance_contract(config)
    grid_physics = _grid_physics_contract(config)
    return {
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": str(inputs.get("hkl_sha256") or ""),
        "source_model": str(model_path),
        "source_model_sha256": str(inputs.get("res_sha256") or ""),
        "hkl_transform": str(inputs.get("hkl_transform") or ""),
        "model_tag": str(inputs.get("model_tag") or ""),
        "alternate_hand": bool(config.get("alternate_hand")),
        "split_contract_path": str(split_path) if split_path is not None else "",
        "split_contract_sha256": split_sha256,
        "split_membership_sha256": str(
            split.get("membership_sha256") or ""
        ),
        "selected_rows": int(row_selection.get("selected_rows") or 0),
        "work_rows": int(row_selection.get("work_rows") or 0),
        "free_rows": int(row_selection.get("free_rows") or 0),
        "scale_fit_split": str(best.get("scale_fit_split") or ""),
        "selection_metric": str(summary.get("selection_metric") or ""),
        "n_basis": int(config.get("n_basis") or 0),
        "n_points": int(config.get("n_points") or 0),
        "max_rows": int(config.get("max_rows") or 0),
        "split_seed": int(config.get("split_seed") or 0),
        "free_fraction": float(config.get("free_fraction") or 0.0),
        "declared_nuisance_contract": declared_nuisance,
        "declared_nuisance_contract_sha256": _json_sha256(
            declared_nuisance
        ),
        "grid_physics_contract": grid_physics,
        "grid_physics_contract_sha256": _json_sha256(grid_physics),
        "selected_nuisance": _json_safe_contract(dict(nuisance)),
        "selected_nuisance_sha256": _json_sha256(dict(nuisance)),
        "bloch_kernel_contract": dict(kernel_contract),
        "bloch_kernel_contract_sha256": (
            _json_sha256(dict(kernel_contract)) if kernel_contract else ""
        ),
    }


_NUISANCE_CATEGORICAL_FIELDS = {
    "thickness_model",
    "sg_jitter_mode",
    "absorption_mode",
    "dose_b_mode",
    "matrix_convention",
}
_NUISANCE_INTEGER_FIELDS = {"detector_scale_bins", "sg_jitter_points"}
_NUISANCE_NULLABLE_FIELDS = {
    "dose_b_ref_frame",
    "native_wght_a",
    "native_wght_b",
}
_NUISANCE_FIELDS = tuple(Nuisance.__dataclass_fields__)


def _canonical_nuisance_value(field: str, value: object) -> object:
    if field in _NUISANCE_CATEGORICAL_FIELDS:
        text = str(value).strip()
        if not text:
            raise ValueError(f"nuisance field {field!r} is empty")
        return text
    if (
        field in _NUISANCE_NULLABLE_FIELDS
        and (
            value is None
            or not str(value).strip()
            or str(value).strip().lower() in {"nan", "none", "null"}
        )
    ):
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"nuisance field {field!r} is not numeric: {value!r}"
        ) from exc
    if math.isnan(numeric) and field in _NUISANCE_NULLABLE_FIELDS:
        return "nan"
    if not math.isfinite(numeric):
        raise ValueError(
            f"nuisance field {field!r} is not finite: {value!r}"
        )
    if field in _NUISANCE_INTEGER_FIELDS:
        integer = int(numeric)
        if float(integer) != numeric:
            raise ValueError(
                f"nuisance field {field!r} is not integral: {value!r}"
            )
        return integer
    # One deterministic text representation avoids differences caused only by
    # CSV formatting while retaining exact IEEE-754 round trips.
    return format(numeric, ".17g")


def _nuisance_contract_from_row(
    row: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], str]:
    canonical = {
        field: _canonical_nuisance_value(field, row.get(field))
        for field in _NUISANCE_FIELDS
    }
    typed: dict[str, object] = {}
    for field, value in canonical.items():
        if field in _NUISANCE_CATEGORICAL_FIELDS:
            typed[field] = value
        elif field in _NUISANCE_INTEGER_FIELDS:
            typed[field] = int(value)
        elif value == "nan" and field in _NUISANCE_NULLABLE_FIELDS:
            typed[field] = None
        else:
            typed[field] = float(value)
    return canonical, typed, _json_sha256(canonical)


def _nuisance_grid_artifact(
    summary_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    summary_path = Path(summary_path).expanduser().resolve()
    if not summary_path.is_file():
        raise ValueError(f"native summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(
            f"native summary must contain one JSON object: {summary_path}"
        )
    job = _job_contract(summary)
    trials_path = (summary_path.parent / "nuisance_trials.csv").resolve()
    if not trials_path.is_file():
        raise ValueError(f"native nuisance-trial table is missing: {trials_path}")
    with trials_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(
            set(_NUISANCE_FIELDS).union({"work_dyn_wR2"}) - fields
        )
        if missing:
            raise ValueError(
                f"native nuisance-trial table {trials_path} lacks required "
                "columns: " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"native nuisance-trial table is empty: {trials_path}")
    candidates: dict[str, dict[str, object]] = {}
    for row in rows:
        canonical, typed, candidate_sha256 = _nuisance_contract_from_row(row)
        work_dyn_wr2 = _finite_float(row.get("work_dyn_wR2"))
        if not math.isfinite(work_dyn_wr2):
            raise ValueError(
                "native nuisance-trial table contains a nonfinite work-row "
                f"dynamical wR2 for candidate {candidate_sha256}"
            )
        if candidate_sha256 in candidates:
            raise ValueError(
                "native nuisance-trial table contains a duplicate nuisance "
                f"candidate: {candidate_sha256}"
            )
        candidates[candidate_sha256] = {
            "candidate_sha256": candidate_sha256,
            "canonical_nuisance": canonical,
            "typed_nuisance": typed,
            "work_dyn_wR2": work_dyn_wr2,
        }
    artifact = {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "nuisance_trials_path": str(trials_path),
        "nuisance_trials_sha256": _sha256(trials_path),
    }
    return artifact, job, candidates


def select_branch_symmetric_nuisance(
    original_summary_path: Path,
    alternate_summary_path: Path,
) -> dict[str, object]:
    """Select one nuisance state by a hand-label-invariant paired objective.

    Both branches must have evaluated the exact same predeclared nuisance
    candidates on one immutable work/free split.  The selected common nuisance
    minimizes the arithmetic mean of the two work-row dynamical wR2 values.
    Exact objective ties are resolved by the SHA-256 of the canonical nuisance
    payload, never by branch label, free rows, known chemistry, or CSV order.
    """

    original_artifact, original_job, original_candidates = (
        _nuisance_grid_artifact(original_summary_path)
    )
    alternate_artifact, alternate_job, alternate_candidates = (
        _nuisance_grid_artifact(alternate_summary_path)
    )
    paired_job_fields = (
        "source_hkl_sha256",
        "source_model_sha256",
        "hkl_transform",
        "split_contract_sha256",
        "split_membership_sha256",
        "selected_rows",
        "work_rows",
        "free_rows",
        "scale_fit_split",
        "n_basis",
        "n_points",
        "max_rows",
        "split_seed",
        "free_fraction",
        "declared_nuisance_contract_sha256",
        "grid_physics_contract_sha256",
    )
    mismatches = {
        field: {
            "original": original_job.get(field),
            "alternate": alternate_job.get(field),
        }
        for field in paired_job_fields
        if original_job.get(field) != alternate_job.get(field)
    }
    original_role = str(original_job.get("model_tag") or "").split(";", 1)[0]
    alternate_role = str(alternate_job.get("model_tag") or "").split(";", 1)[0]
    if mismatches:
        raise ValueError(
            f"original/alternate nuisance-grid job contracts differ: {mismatches}"
        )
    if (
        original_job.get("alternate_hand") is not False
        or original_role != "original"
        or alternate_job.get("alternate_hand") is not True
        or alternate_role != "alternate_fractional_xyz"
        or original_job.get("scale_fit_split") != "work"
    ):
        raise ValueError(
            "paired nuisance selection requires original and alternate-only "
            "branches under one work-scale contract"
        )
    if set(original_candidates) != set(alternate_candidates):
        raise ValueError(
            "original/alternate hand branches did not evaluate the exact same "
            "predeclared nuisance candidate grid"
        )
    candidate_hashes = sorted(original_candidates)
    candidate_grid_sha256 = _json_sha256(candidate_hashes)
    records = []
    for candidate_sha256 in candidate_hashes:
        original = original_candidates[candidate_sha256]
        alternate = alternate_candidates[candidate_sha256]
        paired_mean = 0.5 * (
            float(original["work_dyn_wR2"])
            + float(alternate["work_dyn_wR2"])
        )
        records.append(
            {
                "candidate_sha256": candidate_sha256,
                "canonical_nuisance": original["canonical_nuisance"],
                "original_work_dyn_wR2": original["work_dyn_wR2"],
                "alternate_work_dyn_wR2": alternate["work_dyn_wR2"],
                "paired_mean_work_dyn_wR2": paired_mean,
            }
        )
    selected = min(
        records,
        key=lambda record: (
            float(record["paired_mean_work_dyn_wR2"]),
            str(record["candidate_sha256"]),
        ),
    )
    selected_candidate = original_candidates[str(selected["candidate_sha256"])]
    return {
        "schema_version": 1,
        "status": "selected",
        "selection_rule": "min_paired_mean_work_dyn_wR2_then_nuisance_sha256_v1",
        "selection_objective": "paired_mean_work_dyn_wR2",
        "hand_label_invariant": True,
        "free_rows_used_for_selection": False,
        "known_configuration_used": False,
        "candidate_count": len(records),
        "candidate_grid_sha256": candidate_grid_sha256,
        "selected_candidate_sha256": selected["candidate_sha256"],
        "selected_paired_mean_work_dyn_wR2": selected[
            "paired_mean_work_dyn_wR2"
        ],
        "selected_nuisance": selected_candidate["typed_nuisance"],
        "paired_candidate_records": records,
        "original_artifact_binding": original_artifact,
        "alternate_artifact_binding": alternate_artifact,
        "original_job_contract": original_job,
        "alternate_job_contract": alternate_job,
    }


def _row_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        "target": np.asarray(
            [_finite_float(row.get("target_I")) for row in rows], dtype=float
        ),
        "sigma_eff": np.asarray(
            [_finite_float(row.get("sigma_eff")) for row in rows], dtype=float
        ),
        "xds_sigma": np.asarray(
            [_finite_float(row.get("xds_sigma")) for row in rows], dtype=float
        ),
        "dyn_raw": np.asarray(
            [_finite_float(row.get("dyn_raw")) for row in rows], dtype=float
        ),
        "born_raw": np.asarray(
            [_finite_float(row.get("born_raw")) for row in rows], dtype=float
        ),
        "dyn_scaled": np.asarray(
            [_finite_float(row.get("dyn_scaled")) for row in rows], dtype=float
        ),
        "born_scaled": np.asarray(
            [_finite_float(row.get("born_scaled")) for row in rows], dtype=float
        ),
        "groups": np.asarray(
            [int(float(row.get("group") or 0)) for row in rows], dtype=int
        ),
        "detector_bins": np.asarray(
            [int(float(row.get("detector_bin") or 0)) for row in rows],
            dtype=int,
        ),
        "work": np.asarray(
            [str(row.get("split") or "").strip() == "work" for row in rows],
            dtype=bool,
        ),
        "free": np.asarray(
            [str(row.get("split") or "").strip() == "free" for row in rows],
            dtype=bool,
        ),
    }


def _score_from_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    all_block = metrics["all"]
    free_block = metrics["free"]
    all_block = all_block if isinstance(all_block, Mapping) else {}
    free_block = free_block if isinstance(free_block, Mapping) else {}
    dyn = all_block.get("dyn")
    born = all_block.get("born")
    free_dyn = free_block.get("dyn")
    dyn = dyn if isinstance(dyn, Mapping) else {}
    born = born if isinstance(born, Mapping) else {}
    free_dyn = free_dyn if isinstance(free_dyn, Mapping) else {}
    dyn_w = float(dyn["wR2_all"])
    free_w = float(free_dyn["wR2_all"])
    dyn_r1 = float(dyn["R1_all"])
    born_r1gt = float(born["R1_gt"])
    born_cc = float(born["CC"])
    score = (
        dyn_w
        + 0.25 * free_w
        + 0.10 * dyn_r1
        + max(0.0, born_r1gt - 0.35)
        + max(0.0, 0.5 - born_cc)
    )
    return {
        "score": score,
        "dyn_wR2_all": dyn_w,
        "free_dyn_wR2_all": free_w,
        "dyn_R1_all": dyn_r1,
        "dyn_R1_gt": float(dyn["R1_gt"]),
        "born_R1_gt": born_r1gt,
        "born_CC": born_cc,
        "dyn_CC": float(dyn["CC"]),
    }


def replay_transform_work_born(summary_path: Path) -> dict[str, object]:
    """Recompute a transform score from its hash-bound work-row artifact."""

    summary, _, rows, binding = _read_summary_and_rows(summary_path)
    arrays = _row_arrays(rows)
    best = summary.get("best_native")
    best = best if isinstance(best, Mapping) else {}
    native_weight = best.get("native_weight")
    native_weight = native_weight if isinstance(native_weight, Mapping) else {}
    weight_a = _finite_float(native_weight.get("a"))
    weight_b = _finite_float(native_weight.get("b"))
    if not math.isfinite(weight_a) or not math.isfinite(weight_b):
        raise ValueError("native summary lacks finite selected row-weight parameters")
    valid = (
        arrays["work"]
        & np.isfinite(arrays["target"])
        & np.isfinite(arrays["sigma_eff"])
        & (arrays["sigma_eff"] > 0.0)
        & np.isfinite(arrays["born_scaled"])
    )
    metric = native_metrics(
        arrays["target"],
        arrays["sigma_eff"],
        arrays["born_scaled"],
        valid,
        weight_a,
        weight_b,
    )
    score = (
        float(metric["wR2_all"])
        + 0.10 * float(metric["R1_all"])
        + max(0.0, float(metric["R1_gt"]) - 0.35)
        + max(0.0, 0.5 - float(metric["CC"]))
    )
    job_contract = _job_contract(summary)
    row_support_payload = [
        {
            "row_index": str(row.get("row_index") or "").strip(),
            "target_I": str(row.get("target_I") or "").strip(),
            "xds_sigma": str(row.get("xds_sigma") or "").strip(),
            "split": str(row.get("split") or "").strip(),
        }
        for row in rows
    ]
    job_contract["row_support_sha256"] = _json_sha256(
        row_support_payload
    )
    return {
        "schema_version": 1,
        "status": "replayed",
        "selection_target": "matched_born_work_only_hand_invariant_v1",
        "artifact_binding": binding,
        "job_contract": job_contract,
        "row_weight": {"a": weight_a, "b": weight_b},
        "work_row_count": int(np.count_nonzero(valid)),
        "work_born_metrics": metric,
        "score": score,
        "coordinate_hand_sensitive_terms_used": False,
        "free_rows_used_for_transform_selection": False,
    }


def _ordered_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = tuple(
            str(row.get(field) or "").strip()
            for field in ("row_index", "h", "k", "l")
        )
        if key in result:
            raise ValueError(f"duplicate signed-row evidence key: {key}")
        result[key] = row
    return result


def _refit_branch(
    arrays: Mapping[str, np.ndarray],
    *,
    sigma: np.ndarray,
) -> tuple[dict[str, dict[str, dict[str, float]]], np.ndarray, np.ndarray]:
    target = arrays["target"]
    groups = arrays["groups"]
    detector_bins = arrays["detector_bins"]
    valid = (
        np.isfinite(target)
        & np.isfinite(sigma)
        & (sigma > 0.0)
        & np.isfinite(arrays["dyn_raw"])
        & np.isfinite(arrays["born_raw"])
        & (arrays["dyn_raw"] > 0.0)
        & (arrays["born_raw"] > 0.0)
    )
    work = arrays["work"] & valid
    if np.count_nonzero(work) < 2:
        raise ValueError("common-sigma hand replay has fewer than two work rows")
    if len(np.unique(detector_bins[valid])) > 1:
        dyn_scaled, _, _ = fit_frame_detector_scales(
            target,
            sigma,
            arrays["dyn_raw"],
            groups,
            detector_bins,
            work,
            0.0,
            0.0,
            scale_smoothness=0.0,
        )
        born_scaled, _, _ = fit_frame_detector_scales(
            target,
            sigma,
            arrays["born_raw"],
            groups,
            detector_bins,
            work,
            0.0,
            0.0,
            scale_smoothness=0.0,
        )
    else:
        dyn_scaled, _ = fit_group_scales(
            target,
            sigma,
            arrays["dyn_raw"],
            groups,
            work,
            0.0,
            0.0,
            scale_smoothness=0.0,
        )
        born_scaled, _ = fit_group_scales(
            target,
            sigma,
            arrays["born_raw"],
            groups,
            work,
            0.0,
            0.0,
            scale_smoothness=0.0,
        )
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for label, mask in (
        ("all", np.ones(len(target), dtype=bool)),
        ("work", arrays["work"]),
        ("free", arrays["free"]),
    ):
        metrics[label] = {
            "dyn": native_metrics(
                target, sigma, dyn_scaled, mask & valid, 0.0, 0.0
            ),
            "born": native_metrics(
                target, sigma, born_scaled, mask & valid, 0.0, 0.0
            ),
        }
    return metrics, dyn_scaled, born_scaled


def build_common_sigma_hand_evidence(
    original_summary_path: Path,
    alternate_summary_path: Path,
    *,
    minimum_abs_adjusted_z: float,
) -> dict[str, object]:
    """Replay both hands with one immutable source-sigma and scale-design rule."""

    original_summary, _, original_rows, original_binding = _read_summary_and_rows(
        original_summary_path
    )
    alternate_summary, _, alternate_rows, alternate_binding = _read_summary_and_rows(
        alternate_summary_path
    )
    original_job = _job_contract(original_summary)
    alternate_job = _job_contract(alternate_summary)
    paired_job_fields = (
        "source_hkl_sha256",
        "source_model_sha256",
        "hkl_transform",
        "split_contract_sha256",
        "split_membership_sha256",
        "selected_rows",
        "work_rows",
        "free_rows",
        "scale_fit_split",
        "n_basis",
        "n_points",
        "max_rows",
        "split_seed",
        "free_fraction",
        "declared_nuisance_contract_sha256",
        "selected_nuisance_sha256",
        "bloch_kernel_contract_sha256",
    )
    job_mismatches = {
        field: {
            "original": original_job.get(field),
            "alternate": alternate_job.get(field),
        }
        for field in paired_job_fields
        if original_job.get(field) != alternate_job.get(field)
    }
    if job_mismatches:
        raise ValueError(
            f"original/alternate hand job contracts differ: {job_mismatches}"
        )
    original_model_role = str(original_job["model_tag"]).split(";", 1)[0]
    alternate_model_role = str(alternate_job["model_tag"]).split(";", 1)[0]
    if (
        original_job["alternate_hand"] is not False
        or original_model_role != "original"
        or alternate_job["alternate_hand"] is not True
        or alternate_model_role != "alternate_fractional_xyz"
        or original_job["scale_fit_split"] != "work"
    ):
        raise ValueError(
            "hand evidence does not bind original and alternate-only model "
            "branches under one work-scale contract"
        )
    original_by_key = _ordered_rows(original_rows)
    alternate_by_key = _ordered_rows(alternate_rows)
    if set(original_by_key) != set(alternate_by_key):
        raise ValueError(
            "original/alternate hand row artifacts do not contain the same signed rows"
        )
    keys = sorted(original_by_key)
    paired_original: list[dict[str, str]] = []
    paired_alternate: list[dict[str, str]] = []
    contract_fields = (
        "target_I",
        "xds_sigma",
        "group",
        "detector_bin",
        "split",
    )
    for key in keys:
        left = original_by_key[key]
        right = alternate_by_key[key]
        for field in contract_fields:
            if str(left.get(field) or "").strip() != str(
                right.get(field) or ""
            ).strip():
                raise ValueError(
                    f"hand row contract mismatch for {key}, column {field}"
                )
        paired_original.append(left)
        paired_alternate.append(right)
    original_arrays = _row_arrays(paired_original)
    alternate_arrays = _row_arrays(paired_alternate)
    common_sigma = original_arrays["xds_sigma"]
    if (
        not np.all(np.isfinite(common_sigma))
        or np.any(common_sigma <= 0.0)
        or not np.array_equal(common_sigma, alternate_arrays["xds_sigma"])
    ):
        raise ValueError(
            "source XDS sigma is not one exact positive finite vector across hands"
        )
    if not np.array_equal(original_arrays["target"], alternate_arrays["target"]):
        raise ValueError("target intensity vector differs between hand branches")
    if not np.array_equal(original_arrays["groups"], alternate_arrays["groups"]):
        raise ValueError("scale-group design differs between hand branches")
    if not np.array_equal(
        original_arrays["detector_bins"], alternate_arrays["detector_bins"]
    ):
        raise ValueError("detector-scale design differs between hand branches")
    if not np.array_equal(original_arrays["work"], alternate_arrays["work"]):
        raise ValueError("work/free membership differs between hand branches")

    original_metrics, original_dyn, _ = _refit_branch(
        original_arrays, sigma=common_sigma
    )
    alternate_metrics, alternate_dyn, _ = _refit_branch(
        alternate_arrays, sigma=common_sigma
    )
    original_score = _score_from_metrics(original_metrics)
    alternate_score = _score_from_metrics(alternate_metrics)
    delta = float(original_score["score"] - alternate_score["score"])

    original_z_rows: list[dict[str, str]] = []
    alternate_z_rows: list[dict[str, str]] = []
    for index, (left, right) in enumerate(
        zip(paired_original, paired_alternate, strict=True)
    ):
        original_z_rows.append(
            {
                **left,
                "sigma_eff": f"{common_sigma[index]:.17g}",
                "dyn_scaled": f"{original_dyn[index]:.17g}",
            }
        )
        alternate_z_rows.append(
            {
                **right,
                "sigma_eff": f"{common_sigma[index]:.17g}",
                "dyn_scaled": f"{alternate_dyn[index]:.17g}",
            }
        )
    missing = adjusted_hand_missing_columns(
        list(original_z_rows[0]) if original_z_rows else [],
        list(alternate_z_rows[0]) if alternate_z_rows else [],
        "dyn_scaled",
    )

    def adjusted(mask_label: str) -> dict[str, Any]:
        joined = [
            (left, right)
            for left, right in zip(
                original_z_rows, alternate_z_rows, strict=True
            )
            if mask_label == "all"
            or str(left.get("split") or "").strip() == mask_label
        ]
        return klar_palatinus_adjusted_hand_z(joined, missing, "dyn_scaled")

    adjusted_all = adjusted("all")
    adjusted_free = adjusted("free")
    z_original = _finite_float(adjusted_all.get("z"))
    preferred = "alternate" if delta > 0.0 else "original"
    branch_aligned_z = z_original if preferred == "original" else -z_original
    minimum_z = float(minimum_abs_adjusted_z)
    sign_agrees = bool(math.isfinite(branch_aligned_z) and branch_aligned_z > 0.0)
    threshold_passed = bool(
        sign_agrees and abs(branch_aligned_z) >= minimum_z
    )
    sigma_payload = [
        {
            "row_key": list(key),
            "sigma": f"{common_sigma[index]:.17g}",
        }
        for index, key in enumerate(keys)
    ]
    scale_design_payload = [
        {
            "row_key": list(key),
            "group": int(original_arrays["groups"][index]),
            "detector_bin": int(original_arrays["detector_bins"][index]),
            "split": (
                "work" if bool(original_arrays["work"][index]) else "free"
            ),
        }
        for index, key in enumerate(keys)
    ]
    return {
        "schema_version": 1,
        "status": "replayed",
        "comparison_rule": "matched_rows_common_source_sigma_refit_scales_v1",
        "common_sigma_rule": "source_xds_sigma_exact_matched_rows_v1",
        "common_sigma_sha256": _json_sha256(sigma_payload),
        "common_sigma_row_count": len(keys),
        "scale_design_rule": (
            "same_group_detector_split_design; independent inverse-variance "
            "work-row nuisance-scale fit per hand; WGHT a=b=0"
        ),
        "scale_design_sha256": _json_sha256(scale_design_payload),
        "original_artifact_binding": original_binding,
        "alternate_artifact_binding": alternate_binding,
        "original_job_contract": original_job,
        "alternate_job_contract": alternate_job,
        "original_summary_selection_metric": str(
            original_summary.get("selection_metric") or ""
        ),
        "alternate_summary_selection_metric": str(
            alternate_summary.get("selection_metric") or ""
        ),
        "original_score": original_score,
        "alternate_score": alternate_score,
        "score_delta_original_minus_alternate": delta,
        "adjusted_hand_z_original_favored": {
            "all": adjusted_all,
            "free": adjusted_free,
        },
        "residual_preferred_branch": preferred,
        "branch_aligned_adjusted_z": branch_aligned_z,
        "minimum_abs_adjusted_z": minimum_z,
        "branch_z_sign_agreement": sign_agrees,
        "adjusted_z_threshold_passed": threshold_passed,
        "known_configuration_used": False,
    }
