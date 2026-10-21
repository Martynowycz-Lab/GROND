"""Validated numeric prediction-cache storage. Content, loaded geometry, and physical row identity matter; a matching filename alone is insufficient."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from grond.engine.bloch_engine import (
    Config,
    XDSData,
    sha256,
)


def external_geometry_fingerprints(data: XDSData, cfg: Config) -> dict:
    """Bind outer predictions to geometry contents before any cache hit."""

    if isinstance(data, XDSData):
        data.verify_source_files()
    result = {}
    for label, value in (
        ("geometry_path", cfg.geometry_path_file),
    ):
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"required {label} file is missing: {path}")
            result[label] = {"path": str(path), "sha256": sha256(path)}
    # A programmatic caller may retain an XDSData object while a sidecar is
    # replaced. Bind the arrays actually loaded into that object as well as
    # current file bytes, so it cannot seed a hit for a subsequently reloaded
    # object with different effective geometry.
    for name in ("A_at_scan_points",):
        value = getattr(data, name, None)
        if value is not None:
            array = np.ascontiguousarray(value, dtype=">f8")
            result[f"loaded_{name}"] = {
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    return result


def canonical_prediction_cache_payload(payload: dict) -> str:
    """Return a deterministic representation that preserves NaN sentinels.

    ``Config.dose_b_ref_frame`` uses NaN to mean that no reference frame was
    supplied. JSON round-trips that sentinel correctly for our local NPZ
    metadata, but Python dictionary equality does not because NaN is unequal
    to itself. Comparing this canonical representation makes a valid cache
    round-trip stable without weakening any content checks.
    """

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def write_prediction_cache(path: Path, save: dict, payload: dict, compression: str) -> None:
    """Write the prediction arrays and the JSON metadata to an NPZ file through a temporary file and a rename.

    compression none writes it uncompressed; any other value compresses it.
    """

    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    if compression == "none":
        np.savez(tmp, **save, meta_json=json.dumps(payload, sort_keys=True))
    else:
        np.savez_compressed(tmp, **save, meta_json=json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def cacheable_prediction_arrays(pred: dict) -> dict:
    """Return the numeric prediction arrays that are safe to store in NPZ.

    Prediction dictionaries can also contain live diagnostics such as row-cache
    status dictionaries.  NumPy stores those as object arrays, which cannot be
    reloaded with allow_pickle=False and should not be part of the portable
    prediction cache.
    """

    skip = {"support_hkls", "cache_status", "cache_path", "row_plan_cache_status"}
    save = {}
    for key, value in pred.items():
        if key in skip:
            continue
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            continue
        if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
            continue
        save[key] = arr
    return save


def read_prediction_cache(path: Path) -> dict:
    """Read a prediction cache, ignoring stale object-array diagnostics."""

    pred = {}
    skipped = []
    with np.load(path, allow_pickle=False) as loaded:
        if "meta_json" not in loaded.files:
            raise ValueError(f"prediction cache {path} has no metadata")
        pred["prediction_cache_meta"] = json.loads(str(loaded["meta_json"]))
        for key in loaded.files:
            if key == "meta_json":
                continue
            try:
                value = loaded[key]
            except ValueError as exc:
                if "Object arrays cannot be loaded" in str(exc):
                    skipped.append(key)
                    continue
                raise
            if key == "seconds" and np.ndim(value) == 0:
                pred[key] = float(value)
            elif key == "row_plan_cache_unique_diffs" and np.ndim(value) == 0:
                pred[key] = int(value)
            else:
                pred[key] = value
    if skipped:
        pred["prediction_cache_skipped_keys"] = skipped
    return pred


def prediction_cache_validation_errors(pred: dict, payload: dict, indices) -> list[str]:
    """Return reasons a prediction cache cannot satisfy the requested calculation."""

    errors = []
    meta = pred.get("prediction_cache_meta")
    if not isinstance(meta, dict) or canonical_prediction_cache_payload(meta) != canonical_prediction_cache_payload(payload):
        errors.append("metadata does not match the requested prediction contract")
    expected_indices = np.asarray(indices, dtype=np.int64)
    cached_indices = np.asarray(pred.get("indices", []), dtype=np.int64)
    if not np.array_equal(cached_indices, expected_indices):
        errors.append("cached row indices differ")
    for key in ("dyn", "born"):
        values = np.asarray(pred.get(key, []))
        if values.shape != (len(expected_indices),):
            errors.append(f"{key} has shape {values.shape}, expected {(len(expected_indices),)}")
        elif not np.all(np.isfinite(values)) or np.any(values < 0.0):
            errors.append(f"{key} contains non-finite or negative predictions")
    if "basis_size" in pred:
        basis_size = np.asarray(pred["basis_size"])
        if basis_size.shape != (len(expected_indices),) or np.any(basis_size < 2):
            errors.append("basis_size is incomplete or invalid")
    if int(payload.get("cache_schema", -1)) >= 6:
        basis_support_hash64 = np.asarray(
            pred.get("basis_support_hash64", []), dtype=np.uint64
        )
        if basis_support_hash64.shape != (len(expected_indices),):
            errors.append("basis-support hash provenance is incomplete")
        elif np.any(basis_support_hash64 == 0):
            errors.append("basis-support hash provenance is invalid")
    if int(payload.get("cache_schema", -1)) >= 5:
        continuation_used = np.asarray(
            pred.get("sg_profile_same_branch_continuation_used", [])
        )
        continuation_segments = np.asarray(
            pred.get("sg_profile_same_branch_continuation_segments", [])
        )
        if continuation_used.shape != (len(expected_indices),):
            errors.append("profile continuation usage provenance is incomplete")
        elif (
            not np.all(np.isfinite(continuation_used))
            or np.any(~np.isin(continuation_used, (False, True)))
        ):
            errors.append("profile continuation usage provenance is invalid")
        if continuation_segments.shape != (len(expected_indices),):
            errors.append("profile continuation segment provenance is incomplete")
        elif (
            not np.all(np.isfinite(continuation_segments))
            or np.any(continuation_segments < 1)
            or np.any(continuation_segments != np.floor(continuation_segments))
        ):
            errors.append("profile continuation segment provenance is invalid")
        if (
            continuation_used.shape == (len(expected_indices),)
            and continuation_segments.shape == (len(expected_indices),)
        ):
            continuation_used = continuation_used.astype(bool)
            continuation_segments = continuation_segments.astype(np.int64)
            if np.any(continuation_used & (continuation_segments < 2)):
                errors.append("continued profile rows must record at least two local segments")
            if np.any(~continuation_used & (continuation_segments != 1)):
                errors.append("direct profile rows must record exactly one local segment")
    return errors
