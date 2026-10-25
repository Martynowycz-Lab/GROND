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

from grond.engine.native_row_regularize import native_metrics


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hashable(value: object) -> object:
    """Return ``value`` with NaN and infinities replaced by tagged objects, so it hashes as strict JSON."""

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
    """Return the SHA-256 of ``value`` as canonical JSON (sorted keys, no spaces, tagged non-finite floats)."""

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
    """Return ``value`` as a float, or NaN when it is missing or not finite."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _read_summary_and_rows(
    summary_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, str]], dict[str, str]]:
    """Read a native summary and its hash-checked row table; return the summary, table path, rows and binding.

    The binding records both files' paths and SHA-256. Missing evidence columns raise ValueError.
    """

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
    """Return the nuisance grid options a native run was given, keyed by nuisance field."""

    mapping = {
        "thickness_A": "thickness_grid",
        "thickness_model": "thickness_model_grid",
        "thickness_grad_frac": "thickness_grad_grid",
        "frame_offset": "frame_offset_grid",
        "group_width_deg": "group_width_grid",
        # No detector scale field: every run declares 0 bins.
        "detector_scale_bins": None,
        "sigma_a": "sigma_a_grid",
        "sigma_b_corr": "sigma_b_corr_grid",
        "sigma_c_peak": "sigma_c_peak_grid",
        "sg_jitter_mode": "sg_jitter_mode_grid",
        "sg_jitter_sigma0": "sg_jitter_sigma0_grid",
        "sg_jitter_sigma_d_inv": "sg_jitter_sigma_d_inv_grid",
        "sg_jitter_sigma_frame": "sg_jitter_sigma_frame_grid",
        "absorption_mode": "absorption_mode_grid",
        "absorption_mu": "absorption_mu_grid",
        "dose_b_mode": "dose_b_mode_grid",
        "dose_b_per_frame": "dose_b_per_frame_grid",
        "row_resolution_b": "row_resolution_b_grid",
    }
    return {
        output: "0" if source is None else str(config.get(source) or "").strip()
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
        "split_scheme",
        "split_salt",
        "scale_fit_split",
        "selection_metric",
        "matrix_convention",
        "row_plan_cache",
        "max_rows",
    )
    return {field: config.get(field) for field in fields}


def _job_contract(summary: Mapping[str, object]) -> dict[str, object]:
    """Return the identity of one native run: inputs, split, row counts, declared grids and selected nuisance.

    The source HKL, model and split contract must still match their recorded SHA-256; each contract block carries its
    own canonical JSON hash.
    """

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
        "split_scheme": str(config.get("split_scheme") or "legacy"),
        "split_salt": str(config.get("split_salt") or ""),
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


_NUISANCE_NULLABLE_FIELDS = {
    "dose_b_ref_frame",
    "native_wght_a",
    "native_wght_b",
}


def _row_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Return the numeric columns, groups, detector bins and work/free masks of native row-table rows."""

    return {
        "target": np.asarray(
            [_finite_float(row.get("target_I")) for row in rows], dtype=float
        ),
        "sigma_eff": np.asarray(
            [_finite_float(row.get("sigma_eff")) for row in rows], dtype=float
        ),
        "xds_sigma": np.asarray(
            # Retain the historical array key, but source_sigma is the
            # immutable integration sigma in target_I units when a declared
            # diagnostic target factor has been applied.
            [_finite_float(row.get("source_sigma", row.get("xds_sigma"))) for row in rows], dtype=float
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
