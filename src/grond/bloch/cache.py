"""Bounds-checked potential storage and structural validation of row-plan caches.

Cache acceptance checks the numerical observation contract; a readable NPZ is
not enough. Optional diagnostic NaNs are permitted only in their named modes."""

from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from grond.bloch.constants import (
    DEFAULT_MAX_DENSE_UG_GIB,
    DENSE_UG_BUILD_BYTES_PER_VOXEL,
    DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL,
    ROW_PLAN_CACHE_SCHEMA,
)


def dense_ug_memory_plan(h_lim: int, k_lim: int, l_lim: int) -> dict[str, int | list[int]]:
    """Estimate dense U_g grid storage and peak working memory before allocation."""

    limits = tuple(int(value) for value in (h_lim, k_lim, l_lim))
    if any(value < 0 for value in limits):
        raise ValueError(f"dense U_g limits must be nonnegative, got {limits}")
    shape = tuple(2 * value + 1 for value in limits)
    voxels = math.prod(shape)
    return {
        "shape": list(shape),
        "voxels": int(voxels),
        "cube_bytes": int(voxels * np.dtype(np.complex128).itemsize),
        "estimated_cache_load_peak_bytes": int(voxels * DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL),
        "estimated_build_peak_bytes": int(voxels * DENSE_UG_BUILD_BYTES_PER_VOXEL),
    }


def dense_ug_memory_limit_bytes() -> int:
    """Return the dense U_g working-memory ceiling, with an explicit env override."""

    raw = os.environ.get("GROND_MAX_DENSE_UG_GIB", str(DEFAULT_MAX_DENSE_UG_GIB)).strip()
    try:
        gib = float(raw)
    except ValueError as exc:
        raise ValueError("GROND_MAX_DENSE_UG_GIB must be a positive finite number") from exc
    if not math.isfinite(gib) or gib <= 0.0:
        raise ValueError("GROND_MAX_DENSE_UG_GIB must be a positive finite number")
    return int(gib * (1024**3))


def enforce_dense_ug_memory_limit(
    plan: dict[str, int | list[int]], *, operation: str
) -> dict[str, int | str | list[int]]:
    """Fail before a dense U_g allocation whose estimated peak exceeds the limit."""

    estimate_key = {
        "build": "estimated_build_peak_bytes",
        "cache load": "estimated_cache_load_peak_bytes",
    }.get(operation)
    if estimate_key is None:
        raise ValueError(f"unknown dense U_g memory operation {operation!r}")
    estimated = int(plan[estimate_key])
    limit = dense_ug_memory_limit_bytes()
    status: dict[str, int | str | list[int]] = {
        **plan,
        "operation": operation,
        "estimated_peak_bytes": estimated,
        "limit_bytes": limit,
    }
    if estimated > limit:
        shape = " x ".join(str(value) for value in plan["shape"])
        gib = 1024**3
        raise MemoryError(
            f"Dense U_g {operation} refused before allocation: grid {shape} "
            f"({int(plan['voxels']):,} voxels, {int(plan['cube_bytes']) / gib:.2f} GiB stored) "
            f"has an estimated {estimated / gib:.2f} GiB peak, above the {limit / gib:.2f} GiB "
            "safety limit. Increase ug_d_min to reduce the reciprocal grid, or set "
            "GROND_MAX_DENSE_UG_GIB to a larger value only after confirming available memory."
        )
    return status


def row_plan_cache_validation_errors(cache: dict, indices, expected_key: str, expected_meta: dict) -> list[str]:
    """Return reasons a row-plan cache payload cannot be trusted."""

    required = {
        "indices",
        "basis_offsets",
        "basis_flat",
        "profile_offsets",
        "profile_weights_flat",
        "profile_omega_flat",
        "profile_tilts_flat",
        "sg_offsets",
        "sg_basis_flat",
        "diff_offsets",
        "diff_idx_flat",
        "unique_diffs",
        "unique_inv_d2",
        "center_sg",
        "omega_center",
        "omega_bragg",
        "omega_delta",
        "sg_window",
        "sg_domega",
        "sg_dframe",
        "sg_window_raw",
        "sg_window_half_frames",
        "sg_window_clipped",
        "sg_jitter_sigma",
        "sg_jitter_d_inv",
        "sg_jitter_frame_fraction",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "sg_profile_same_branch_continuation_used",
        "sg_profile_same_branch_continuation_segments",
        "sg_residual_max",
        "dose_delta_b",
    }
    errors: list[str] = []
    missing = sorted(k for k in required if k not in cache)
    if missing:
        errors.append(f"missing arrays: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    meta = cache.get("meta")
    if not isinstance(meta, dict):
        errors.append("missing metadata")
        meta = {}
    if meta.get("key") != expected_key:
        errors.append("cache key mismatch")
    if meta.get("cache_schema") != expected_meta.get("cache_schema", ROW_PLAN_CACHE_SCHEMA):
        errors.append("cache schema mismatch")
    for field, expected_value in expected_meta.items():
        if meta.get(field) != expected_value:
            errors.append(f"metadata mismatch: {field}")
    # Reject malformed containers before indexing their endpoints or comparing
    # differently shaped row arrays. Missing fields still receive the useful
    # independent checks below when the present arrays have valid shapes.
    structural_errors = []
    matrix_arrays = {"basis_flat", "unique_diffs", "profile_tilts_flat"}
    integer_arrays = {
        "indices", "basis_offsets", "profile_offsets", "sg_offsets", "diff_offsets",
        "basis_flat", "unique_diffs", "diff_idx_flat",
    }
    variable_arrays = {
        "basis_offsets", "basis_flat", "profile_offsets", "profile_weights_flat",
        "profile_omega_flat", "profile_tilts_flat", "sg_offsets", "sg_basis_flat",
        "diff_offsets", "diff_idx_flat", "unique_diffs", "unique_inv_d2",
    }
    for name in (required | {"omega_bragg_root_available"}).intersection(cache):
        try:
            values = np.asarray(cache[name])
        except (TypeError, ValueError):
            structural_errors.append(f"{name} is not a numerical array")
            continue
        if values.dtype.kind not in {"b", "i", "u", "f"}:
            structural_errors.append(f"{name} is not a real numerical array")
            continue
        if values.ndim != (2 if name in matrix_arrays else 1):
            structural_errors.append(f"{name} has the wrong number of dimensions")
        if name.endswith("_offsets") and values.shape != (len(indices) + 1,):
            structural_errors.append(f"{name} must have one boundary per row plus the final endpoint")
        if name not in variable_arrays and values.shape != (len(indices),):
            structural_errors.append(f"{name} must have one value per row")
        if name in integer_arrays:
            # Integer storage is already integral; avoid a float copy of the
            # potentially large beam-difference lookup merely to confirm it.
            invalid = values.dtype.kind == "f" and (
                not np.all(np.isfinite(values))
                or np.any(values != np.floor(values))
                or np.any(values < -(2**63))
                or np.any(values >= 2**63)
            )
            if invalid or (values.dtype.kind == "u" and np.any(values >= 2**63)):
                structural_errors.append(f"{name} must contain finite int64-representable integers")
    if structural_errors:
        return errors + structural_errors
    if "indices" in cache and not np.array_equal(
        np.asarray(cache["indices"], dtype=np.int64), np.asarray(indices, dtype=np.int64)
    ):
        errors.append("index order mismatch")
    if "basis_offsets" in cache:
        offsets = np.asarray(cache["basis_offsets"], dtype=np.int64)
        if len(offsets) != len(indices) + 1:
            errors.append("basis offset count mismatch")
        elif offsets.ndim != 1 or offsets[0] != 0 or np.any(offsets < 0):
            errors.append("basis offsets must be a nonnegative vector starting at zero")
        elif np.any(np.diff(offsets) <= 1):
            errors.append("basis offsets do not contain direct and target beams for every row")
        elif "basis_flat" in cache and offsets[-1] != len(cache["basis_flat"]):
            errors.append("basis offsets do not span basis_flat")
    if "basis_flat" in cache:
        basis = np.asarray(cache["basis_flat"])
        if (
            basis.ndim != 2
            or basis.shape[1:] != (3,)
            or not np.all(np.isfinite(basis))
            or not np.all(basis == np.rint(basis))
        ):
            errors.append("basis_flat must be a finite integer N by 3 array")
    if "profile_offsets" in cache:
        offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        if len(offsets) != len(indices) + 1:
            errors.append("profile offset count mismatch")
        elif offsets.ndim != 1 or offsets[0] != 0 or np.any(offsets < 0):
            errors.append("profile offsets must be a nonnegative vector starting at zero")
        elif np.any(np.diff(offsets) <= 0):
            errors.append("profile offsets contain an empty row profile")
        elif "profile_weights_flat" in cache and offsets[-1] != len(cache["profile_weights_flat"]):
            errors.append("profile offsets do not span profile_weights_flat")
    if "profile_weights_flat" in cache:
        weights = np.asarray(cache["profile_weights_flat"], dtype=float)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            errors.append("profile weights contain non-finite or negative values")
        if "profile_offsets" in cache:
            offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
            if (
                len(offsets) == len(indices) + 1
                and offsets.ndim == 1
                and offsets[0] == 0
                and offsets[-1] == len(weights)
                and any(float(np.sum(weights[offsets[i] : offsets[i + 1]])) <= 0.0 for i in range(len(indices)))
            ):
                errors.append("profile weights contain an all-zero row profile")
    if "profile_omega_flat" in cache:
        values = np.asarray(cache["profile_omega_flat"], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            errors.append("profile omega values must be a finite vector")
        elif "profile_offsets" in cache:
            offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
            if offsets[-1] != len(values):
                errors.append("profile offsets do not span profile_omega_flat")
    if "profile_tilts_flat" in cache:
        values = np.asarray(cache["profile_tilts_flat"], dtype=float)
        if values.ndim != 2 or values.shape[1:] != (2,) or not np.all(np.isfinite(values)):
            errors.append("profile tilts must be a finite N by 2 array")
        elif "profile_offsets" in cache:
            offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
            if offsets[-1] != len(values):
                errors.append("profile offsets do not span profile_tilts_flat")
    if "sg_offsets" in cache:
        offsets = np.asarray(cache["sg_offsets"], dtype=np.int64)
        if (
            len(offsets) != len(indices) + 1
            or offsets.ndim != 1
            or offsets[0] != 0
            or np.any(offsets < 0)
            or np.any(np.diff(offsets) <= 0)
        ):
            errors.append("Sg offsets are incomplete")
        elif "sg_basis_flat" in cache and offsets[-1] != len(cache["sg_basis_flat"]):
            errors.append("Sg offsets do not span sg_basis_flat")
    if "sg_basis_flat" in cache:
        sg_basis = np.asarray(cache["sg_basis_flat"], dtype=float)
        if sg_basis.ndim != 1 or not np.all(np.isfinite(sg_basis)):
            errors.append("Sg basis contains non-finite values or has the wrong shape")
    if "diff_offsets" in cache:
        offsets = np.asarray(cache["diff_offsets"], dtype=np.int64)
        if (
            len(offsets) != len(indices) + 1
            or offsets.ndim != 1
            or offsets[0] != 0
            or np.any(offsets < 0)
            or np.any(np.diff(offsets) < 4)
        ):
            errors.append("difference offsets are incomplete")
        elif "diff_idx_flat" in cache and offsets[-1] != len(cache["diff_idx_flat"]):
            errors.append("difference offsets do not span diff_idx_flat")
    if "unique_diffs" in cache:
        diffs = np.asarray(cache["unique_diffs"])
        if diffs.ndim != 2 or diffs.shape[1:] != (3,) or not np.all(np.isfinite(diffs)):
            errors.append("unique_diffs must be a finite N by 3 array")
        if "unique_inv_d2" in cache:
            inv_d2 = np.asarray(cache["unique_inv_d2"], dtype=float)
            if inv_d2.shape != (len(diffs),) or not np.all(np.isfinite(inv_d2)) or np.any(inv_d2 < 0.0):
                errors.append("unique_inv_d2 is incomplete or invalid")
        if "diff_idx_flat" in cache:
            diff_indices = np.asarray(cache["diff_idx_flat"], dtype=np.int64)
            if np.any(diff_indices < 0) or np.any(diff_indices >= len(diffs)):
                errors.append("difference indices are out of range")
    if all(name in cache for name in ("basis_offsets", "profile_offsets", "sg_offsets", "diff_offsets")):
        basis_offsets = np.asarray(cache["basis_offsets"], dtype=np.int64)
        profile_offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        sg_offsets = np.asarray(cache["sg_offsets"], dtype=np.int64)
        diff_offsets = np.asarray(cache["diff_offsets"], dtype=np.int64)
        if all(
            offsets.ndim == 1 and len(offsets) == len(indices) + 1 and offsets[0] == 0 and np.all(offsets >= 0)
            for offsets in (basis_offsets, profile_offsets, sg_offsets, diff_offsets)
        ):
            basis_counts = np.diff(basis_offsets)
            profile_counts = np.diff(profile_offsets)
            if not np.array_equal(np.diff(sg_offsets), basis_counts * profile_counts):
                errors.append("Sg offsets do not contain one basis vector per profile point")
            if not np.array_equal(np.diff(diff_offsets), basis_counts * basis_counts):
                errors.append("difference offsets do not contain one index per basis-matrix entry")
    finite_row_arrays = (
        "center_sg",
        "omega_center",
        "sg_window",
        "sg_domega",
        "sg_dframe",
        "sg_window_raw",
        "sg_window_half_frames",
        "sg_window_clipped",
        "sg_jitter_sigma",
        "sg_jitter_d_inv",
        "sg_jitter_frame_fraction",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "sg_profile_same_branch_continuation_used",
        "sg_profile_same_branch_continuation_segments",
        "sg_residual_max",
        "dose_delta_b",
    )
    optional_nan_contracts = {
        # These are audit-only quantities which are undefined for the stated
        # physical contract.  The cache builder records that fact as NaN.  Do
        # not relax validation for any geometry-defining array or for a mode
        # in which the quantity is applicable.
        "sg_window_half_frames": (
            "sg_window_mode",
            {"fixed", "rlp_peak"},
        ),
        "sg_jitter_d_inv": (
            "sg_jitter_mode",
            {"none"},
        ),
        "sg_jitter_frame_fraction": (
            "sg_jitter_mode",
            {"none"},
        ),
    }
    for name in finite_row_arrays:
        if name not in cache:
            continue
        values = np.asarray(cache[name])
        if values.shape != (len(indices),):
            errors.append(f"{name} is incomplete or non-finite")
            continue
        optional_contract = optional_nan_contracts.get(name)
        allows_nan = False
        if optional_contract is not None:
            contract_field, nonapplicable_values = optional_contract
            allows_nan = str(expected_meta.get(contract_field, "")).lower() in nonapplicable_values
        numeric = np.asarray(values, dtype=float)
        valid = ~np.isinf(numeric) if allows_nan else np.isfinite(numeric)
        if not np.all(valid):
            errors.append(f"{name} is incomplete or non-finite")
    if "omega_bragg" in cache or "omega_delta" in cache:
        root_available = np.asarray(
            cache.get(
                "omega_bragg_root_available",
                np.isfinite(np.asarray(cache.get("omega_bragg", []), dtype=float)),
            ),
            dtype=bool,
        )
        if root_available.shape != (len(indices),):
            errors.append("omega_bragg_root_available is incomplete")
        for name in ("omega_bragg", "omega_delta"):
            values = np.asarray(cache.get(name, []), dtype=float)
            if values.shape != (len(indices),):
                errors.append(f"{name} is incomplete or non-finite")
                continue
            if root_available.shape == values.shape and np.any(root_available & ~np.isfinite(values)):
                errors.append(f"{name} is non-finite despite an available local root")
            if np.any(np.isinf(values)):
                errors.append(f"{name} contains an infinite diagnostic")
    if all(
        name in cache
        for name in (
            "sg_profile_same_branch_continuation_used",
            "sg_profile_same_branch_continuation_segments",
        )
    ):
        continuation_used = np.asarray(cache["sg_profile_same_branch_continuation_used"], dtype=bool)
        continuation_segments = np.asarray(cache["sg_profile_same_branch_continuation_segments"], dtype=np.int64)
        if np.any(continuation_segments < 1):
            errors.append("profile continuation segment counts must be positive")
        if np.any(continuation_used & (continuation_segments < 2)):
            errors.append("profile continuation rows must record at least two strict local segments")
        if np.any(~continuation_used & (continuation_segments != 1)):
            errors.append("direct profile rows must record exactly one local segment")
    if "profile_offsets" in cache:
        offsets = np.asarray(cache["profile_offsets"], dtype=np.int64)
        if offsets.ndim == 1 and len(offsets) == len(indices) + 1 and offsets[0] == 0 and np.all(offsets >= 0):
            profile_counts = np.diff(offsets)
            if "sg_profile_points_total" in cache and not np.array_equal(
                np.asarray(cache["sg_profile_points_total"], dtype=np.int64),
                profile_counts,
            ):
                errors.append("profile point totals do not match profile offsets")
            if "profile_weights_flat" in cache and "sg_profile_weight_sum" in cache:
                weights = np.asarray(cache["profile_weights_flat"], dtype=float)
                if offsets[-1] == len(weights):
                    sums = np.asarray(
                        [np.sum(weights[offsets[i] : offsets[i + 1]]) for i in range(len(indices))],
                        dtype=float,
                    )
                    if not np.allclose(
                        sums,
                        np.asarray(cache["sg_profile_weight_sum"], dtype=float),
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        errors.append("profile weight sums do not match cached row metadata")
    return errors


def validate_ug_cube(cube: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """Return a finite complex128 U_g cube of exactly the expected shape."""

    cube = np.asarray(cube)
    if cube.shape != expected_shape:
        raise ValueError(f"U_g cache shape {cube.shape} does not match expected {expected_shape}")
    if cube.dtype != np.dtype(np.complex128):
        raise ValueError(f"U_g cache dtype {cube.dtype} is not complex128")
    flat = cube.reshape(-1)
    for start in range(0, len(flat), 1_000_000):
        if not np.isfinite(flat[start : start + 1_000_000]).all():
            raise ValueError("U_g cache contains non-finite values")
    return cube


def load_validated_ug_cache(
    path: Path,
    *,
    expected_lims: tuple[int, int, int],
    expected_meta: dict,
) -> np.ndarray:
    """Load one U_g cache only after its schema, shape, metadata, and values agree."""

    expected_shape = tuple(2 * int(value) + 1 for value in expected_lims)
    expected_bytes = math.prod(expected_shape) * np.dtype(np.complex128).itemsize
    with zipfile.ZipFile(path) as archive:
        cube_member = archive.getinfo("ug_cube.npy")
        if not expected_bytes <= cube_member.file_size <= expected_bytes + 65536:
            raise ValueError(
                f"U_g cache member size {cube_member.file_size} does not match expected grid bytes {expected_bytes}"
            )
        for name, maximum in (("lims.npy", 65536), ("meta_json.npy", 1024 * 1024)):
            member = archive.getinfo(name)
            if member.file_size > maximum:
                raise ValueError(f"U_g cache member {name} is unexpectedly large ({member.file_size} bytes)")

    with np.load(path, allow_pickle=False) as loaded:
        required = {"ug_cube", "lims", "meta_json"}
        missing = sorted(required.difference(loaded.files))
        if missing:
            raise ValueError(f"U_g cache is missing arrays: {', '.join(missing)}")
        lims = tuple(int(value) for value in np.asarray(loaded["lims"]).reshape(-1))
        if lims != expected_lims:
            raise ValueError(f"U_g cache limits {lims} do not match expected {expected_lims}")
        meta_value = np.asarray(loaded["meta_json"])
        if meta_value.ndim != 0:
            raise ValueError("U_g cache metadata must be a scalar JSON value")
        meta_text = meta_value.item()
        if isinstance(meta_text, bytes):
            meta_text = meta_text.decode("utf-8")
        meta = json.loads(str(meta_text))
        if meta != expected_meta:
            raise ValueError("U_g cache metadata does not match the requested model/configuration")
        cube = np.asarray(loaded["ug_cube"])

    return validate_ug_cube(cube, expected_shape)


def write_ug_cache_atomic(
    path: Path,
    *,
    cube: np.ndarray,
    lims: tuple[int, int, int],
    meta: dict,
) -> None:
    """Write a U_g cache through a same-directory temporary file and atomic replace."""

    expected_shape = tuple(2 * int(value) + 1 for value in lims)
    cube = validate_ug_cube(cube, expected_shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
    try:
        np.savez_compressed(
            tmp,
            ug_cube=cube,
            lims=np.asarray(lims, dtype=np.int64),
            meta_json=json.dumps(meta, sort_keys=True),
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
