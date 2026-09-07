"""Symmetry-aware and candidate-invariant held-out membership contracts.

Indexing/hand candidates must be compared on the same physical work/free rows."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.model import parse_model
from grond.bloch.numerics import sha256
from grond.io.xds import XDSData


def reflection_orbit_key(hkl, symmetry_ops=None, include_friedel: bool = True):
    """Return a canonical reciprocal-space orbit key for held-out splitting."""

    h = np.asarray(hkl, dtype=float)
    rotations = [np.eye(3)]
    for operation in symmetry_ops or []:
        rotation = operation.get("rot") if isinstance(operation, dict) else operation[0]
        rotations.append(np.asarray(rotation, dtype=float))
    equivalents = []
    for rotation in rotations:
        transformed = np.linalg.inv(rotation).T @ h
        rounded = np.rint(transformed).astype(int)
        if not np.allclose(transformed, rounded, rtol=0.0, atol=1e-7):
            raise ValueError("symmetry operation did not map a Miller index to integer values")
        key = tuple(int(value) for value in rounded)
        equivalents.append(key)
        if include_friedel:
            equivalents.append(tuple(-value for value in key))
    return min(equivalents)


def split_masks(hkls, free_fraction=0.10, seed=42, symmetry_ops=None, include_friedel: bool = True):
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("free_fraction must be between 0 and 1")
    orbit_keys = [reflection_orbit_key(h, symmetry_ops, include_friedel) for h in hkls]
    unique = sorted(set(orbit_keys))
    if len(unique) < 2:
        raise ValueError("held-out splitting requires at least two unique HKLs")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    n_free = min(len(unique) - 1, max(1, int(round(len(unique) * float(free_fraction)))))
    free = set(unique[:n_free])
    masks = {
        "all": np.ones(len(hkls), dtype=bool),
        "work": np.zeros(len(hkls), dtype=bool),
        "free": np.zeros(len(hkls), dtype=bool),
    }
    for i, key in enumerate(orbit_keys):
        masks["free"][i] = key in free
    masks["work"] = ~masks["free"]
    return masks


def _canonical_json_sha256(value) -> str:
    """Hash one JSON-compatible value with a stable, whitespace-free encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symmetry_rotation_contract(symmetry_ops) -> list[list[list[float]]]:
    """Return the reciprocal-orbit-relevant part of a model symmetry contract."""

    rotations = []
    for operation in symmetry_ops or []:
        rotation = operation.get("rot") if isinstance(operation, dict) else operation[0]
        array = np.asarray(rotation, dtype=float)
        if array.shape != (3, 3) or not np.all(np.isfinite(array)):
            raise ValueError("model symmetry rotation must be one finite 3x3 matrix")
        rotations.append([[round(float(value), 12) for value in row] for row in array])
    return sorted(rotations, key=lambda value: json.dumps(value, separators=(",", ":")))


def _reciprocal_metric_from_cell(cell) -> np.ndarray:
    """Return G* for a six-parameter direct cell."""

    unit_cell = gemmi.UnitCell(*[float(value) for value in cell])
    basis = np.eye(3, dtype=int)
    metric = np.zeros((3, 3), dtype=float)
    for i in range(3):
        metric[i, i] = float(unit_cell.calculate_1_d2(basis[i]))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair = basis[i] + basis[j]
        cross = 0.5 * (float(unit_cell.calculate_1_d2(pair)) - metric[i, i] - metric[j, j])
        metric[i, j] = metric[j, i] = cross
    return metric


def metric_filter_signed_axis_transforms(
    hkl_path: Path,
    model_path: Path,
    transforms,
    *,
    relative_tolerance: float,
    dataset_label: str = "",
) -> tuple[list[str], dict]:
    """Retain signed-axis transforms compatible with source/model cell metrics.

    This prefilter uses only the source and model unit-cell metrics.  It never
    reads intensities, coordinate hand, atom types, expected chemistry, or a
    known configuration.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("transform metric relative tolerance must be finite and > 0")
    hkl_path = Path(hkl_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    data = XDSData(hkl_path, hkl_transform="identity")
    model = parse_model(model_path)
    source_cell = (
        data.raw_cell.a,
        data.raw_cell.b,
        data.raw_cell.c,
        data.raw_cell.alpha,
        data.raw_cell.beta,
        data.raw_cell.gamma,
    )
    model_cell = tuple(float(value) for value in model["cell"])
    source_metric = _reciprocal_metric_from_cell(source_cell)
    model_metric = _reciprocal_metric_from_cell(model_cell)
    denominator = max(float(np.linalg.norm(model_metric, ord="fro")), 1.0e-30)
    # XDS normally writes INTEGRATE.HKL in the primitive integration basis,
    # even when CORRECT subsequently constrains and reindexes the data in the
    # solved-model space group.  The raw P1 angles and lengths can therefore be
    # several percent away from the symmetry-constrained model cell.  Keep the
    # ordinary full-metric gate strict, but provide one bounded, geometry-only
    # fallback for that explicit XDS P1 -> higher-symmetry situation.  The
    # fallback still operates only on cells and signed-axis permutations; it
    # never uses intensities, atoms, chemistry, or a known hand.
    p1_fallback_full_metric_tolerance = 0.05
    p1_fallback_max_axis_length_fraction = 0.03
    source_space_group_text = str(data.header.get("SPACE_GROUP_NUMBER", "")).split()
    try:
        source_space_group_number = int(float(source_space_group_text[0])) if source_space_group_text else None
    except ValueError:
        source_space_group_number = None
    model_symmetry_order = len(model.get("symmetry_ops") or [])
    source_kind = str(data.header.get("GROND_SOURCE", "XDS") or "XDS").strip().upper()
    p1_fallback_applicable = bool(source_kind == "XDS" and source_space_group_number == 1 and model_symmetry_order > 1)

    records = []
    retained = []
    seen = set()
    for transform in transforms:
        info = parse_hkl_transform_info(transform)
        if info.canonical in seen:
            continue
        seen.add(info.canonical)
        matrix = np.asarray(info.matrix, dtype=float)
        inverse = np.linalg.inv(matrix)
        candidate_metric = inverse.T @ source_metric @ inverse
        difference = candidate_metric - model_metric
        relative_frobenius_mismatch = float(np.linalg.norm(difference, ord="fro") / denominator)
        max_absolute_mismatch = float(np.max(np.abs(difference)))
        candidate_direct_metric = np.linalg.inv(candidate_metric)
        candidate_axis_lengths = np.sqrt(np.maximum(np.diag(candidate_direct_metric), 0.0))
        model_axis_lengths = np.asarray(model_cell[:3], dtype=float)
        axis_length_fractional_mismatch = np.abs(candidate_axis_lengths - model_axis_lengths) / model_axis_lengths
        max_axis_length_fractional_mismatch = float(np.max(axis_length_fractional_mismatch))
        keep = bool(relative_frobenius_mismatch <= tolerance)
        if keep:
            retained.append(info.canonical)
        records.append(
            {
                "transform": info.canonical,
                "determinant": info.determinant,
                "status": "retained" if keep else "rejected",
                "reason": (
                    "source/model reciprocal-cell metrics agree within tolerance"
                    if keep
                    else "source/model reciprocal-cell metric mismatch exceeds tolerance"
                ),
                "relative_frobenius_mismatch": relative_frobenius_mismatch,
                "max_absolute_metric_mismatch_A^-2": max_absolute_mismatch,
                "candidate_axis_lengths_A": candidate_axis_lengths.tolist(),
                "axis_length_fractional_mismatch": (axis_length_fractional_mismatch.tolist()),
                "max_axis_length_fractional_mismatch": (max_axis_length_fractional_mismatch),
                "retention_mode": "strict_full_metric" if keep else "rejected",
            }
        )
    fallback_applied = False
    if not retained:
        fallback_records = [
            record
            for record in records
            if p1_fallback_applicable
            and float(record["relative_frobenius_mismatch"]) <= p1_fallback_full_metric_tolerance
            and float(record["max_axis_length_fractional_mismatch"]) <= p1_fallback_max_axis_length_fraction
        ]
        if fallback_records:
            fallback_applied = True
            fallback_transforms = {str(record["transform"]) for record in fallback_records}
            retained = [
                str(record["transform"]) for record in records if str(record["transform"]) in fallback_transforms
            ]
            for record in records:
                if str(record["transform"]) in fallback_transforms:
                    record.update(
                        {
                            "status": "retained",
                            "reason": (
                                "bounded XDS P1 integration-cell distortion fallback: "
                                "source/model axis lengths and full reciprocal metrics "
                                "agree within fixed fallback limits"
                            ),
                            "retention_mode": "xds_p1_distortion_fallback",
                        }
                    )
        else:
            best = min(
                records,
                key=lambda record: float(record["relative_frobenius_mismatch"]),
            )
            dataset_context = f"dataset {dataset_label!r}: " if dataset_label else ""
            if source_kind == "DIALS":
                raise ValueError(
                    f"{dataset_context}unit-cell metric compatibility filter rejected every "
                    "transform candidate for a DIALS bridge; "
                    f"best={best['transform']} full-metric mismatch="
                    f"{float(best['relative_frobenius_mismatch']):.4f}, max axis-length "
                    f"mismatch={float(best['max_axis_length_fractional_mismatch']):.4f}; "
                    f"source cell={tuple(round(float(v), 6) for v in source_cell)}, "
                    f"model cell={tuple(round(float(v), 6) for v in model_cell)}. "
                    "dials.cosym can make experiment indices mutually consistent, but it "
                    "does not prove that an external RES/HKL pair is in that same basis. "
                    "If the model and baseline HKL were exported from this exact post-cosym "
                    "scaled.expt/scaled.refl basis with no later reindexing or inversion, use "
                    "--hkl-transform identity --trust-hkl-transform. Otherwise inspect this "
                    "named crystal as a possible cell outlier or supply the verified transform."
                )
            raise ValueError(
                f"{dataset_context}unit-cell metric compatibility filter rejected every "
                "transform candidate; "
                f"best={best['transform']} full-metric mismatch="
                f"{float(best['relative_frobenius_mismatch']):.4f}, max axis-length "
                f"mismatch={float(best['max_axis_length_fractional_mismatch']):.4f}; "
                f"source cell={tuple(round(float(v), 6) for v in source_cell)}, "
                f"model cell={tuple(round(float(v), 6) for v in model_cell)}. "
                "The automatic XDS P1 fallback requires an explicit source "
                "SPACE_GROUP_NUMBER=1, a higher-symmetry model, full-metric mismatch "
                f"<={p1_fallback_full_metric_tolerance:.3f}, and maximum axis-length "
                f"mismatch<={p1_fallback_max_axis_length_fraction:.3f}."
            )
    contract = {
        "schema_version": 1,
        "policy": "source_model_reciprocal_cell_metric_only_v1",
        "supported_transform_family": "signed_axis_permutations_only",
        "relative_frobenius_tolerance": tolerance,
        "filter_mode": ("xds_p1_distortion_fallback" if fallback_applied else "strict_full_metric"),
        "source_space_group_number": source_space_group_number,
        "model_symmetry_order": model_symmetry_order,
        "xds_p1_distortion_fallback": {
            "applicable": p1_fallback_applicable,
            "applied": fallback_applied,
            "full_metric_relative_tolerance": p1_fallback_full_metric_tolerance,
            "max_axis_length_fraction": p1_fallback_max_axis_length_fraction,
            "reason": (
                "INTEGRATE.HKL can retain a distorted P1 integration cell while "
                "CORRECT and the solved model use a symmetry-constrained setting"
            ),
        },
        "uses_intensities": False,
        "uses_atomic_coordinates_or_hand": False,
        "uses_atom_types_or_known_chemistry": False,
        "uses_known_configuration": False,
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": sha256(hkl_path),
        "source_model": str(model_path),
        "source_model_sha256": sha256(model_path),
        "source_cell": [float(value) for value in source_cell],
        "source_cell_basis": "raw_UNIT_CELL_CONSTANTS",
        "axis_derived_source_cell": [
            data.axis_derived_source_cell.a,
            data.axis_derived_source_cell.b,
            data.axis_derived_source_cell.c,
            data.axis_derived_source_cell.alpha,
            data.axis_derived_source_cell.beta,
            data.axis_derived_source_cell.gamma,
        ],
        "axis_derived_source_cell_basis": data.axis_derived_source_cell_basis,
        "model_cell": [float(value) for value in model_cell],
        "source_reciprocal_metric": source_metric.tolist(),
        "model_reciprocal_metric": model_metric.tolist(),
        "generated_candidate_count": len(records),
        "retained_candidate_count": len(retained),
        "rejected_candidate_count": len(records) - len(retained),
        "retained_transforms": retained,
        "records": records,
    }
    contract["contract_payload_sha256"] = _canonical_json_sha256(contract)
    return retained, contract


def candidate_invariant_split_membership(
    raw_hkls,
    symmetry_ops,
    transforms,
    *,
    free_fraction: float = 0.10,
    seed: int = 42,
    include_friedel: bool = True,
):
    """Build one work/free split whose orbit groups are valid for every candidate.

    Rows are unioned whenever any candidate transform places them in the same
    model-symmetry/Friedel orbit.  Assigning whole union components to work or
    free therefore prevents a row (or one of its symmetry-equivalent
    duplicates) from changing split merely because a different indexing
    candidate is being scored.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    raw = np.asarray(raw_hkls, dtype=int)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw HKLs must be an N by 3 integer array")
    if len(raw) < 2:
        raise ValueError("candidate-invariant held-out splitting requires at least two rows")
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("free_fraction must be between 0 and 1")

    canonical_transforms = []
    matrices = []
    for transform in transforms:
        info = parse_hkl_transform_info(transform)
        if info.canonical in canonical_transforms:
            continue
        canonical_transforms.append(info.canonical)
        matrices.append(np.asarray(info.matrix, dtype=int))
    if not canonical_transforms:
        raise ValueError("candidate-invariant held-out splitting requires transforms")

    parent = np.arange(len(raw), dtype=int)
    rank = np.zeros(len(raw), dtype=np.int8)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    # One orbit registry must span *all* candidates.  A row transformed by
    # candidate A and another row transformed by candidate B represent the
    # same held-out observation whenever those transformed HKLs share a
    # model-symmetry/Friedel orbit.
    first_by_orbit = {}
    for matrix in matrices:
        transformed = (raw @ matrix.T).astype(int)
        for row_index, hkl in enumerate(transformed):
            key = reflection_orbit_key(
                hkl,
                symmetry_ops=symmetry_ops,
                include_friedel=include_friedel,
            )
            first = first_by_orbit.setdefault(key, row_index)
            union(first, row_index)

    components = {}
    for row_index in range(len(raw)):
        components.setdefault(find(row_index), []).append(row_index)
    ordered_components = sorted(components.values(), key=lambda rows: rows[0])
    if len(ordered_components) < 2:
        raise ValueError(
            "candidate-union symmetry/Friedel grouping leaves fewer than two independent reflection groups"
        )

    shuffled_group_ids = np.arange(len(ordered_components), dtype=int)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled_group_ids)
    n_free_groups = min(
        len(ordered_components) - 1,
        max(1, int(round(len(ordered_components) * float(free_fraction)))),
    )
    free_groups = set(int(value) for value in shuffled_group_ids[:n_free_groups])
    group_by_row = np.zeros(len(raw), dtype=int)
    membership = np.zeros(len(raw), dtype=np.uint8)
    for group_id, row_indices in enumerate(ordered_components):
        group_by_row[row_indices] = group_id
        if group_id in free_groups:
            membership[row_indices] = 1
    if not np.any(membership == 0) or not np.any(membership == 1):
        raise ValueError("candidate-invariant split must contain both work and free rows")
    return {
        "candidate_transforms": canonical_transforms,
        "membership": membership,
        "orbit_group_by_row": group_by_row,
        "group_count": len(ordered_components),
        "work_group_count": len(ordered_components) - len(free_groups),
        "free_group_count": len(free_groups),
    }


def build_transform_split_contract(
    hkl_path: Path,
    model_path: Path,
    transforms,
    *,
    dataset: str,
    free_fraction: float = 0.10,
    seed: int = 42,
    include_friedel: bool = True,
) -> dict:
    """Build the immutable raw-row split shared by transform and hand scouts."""

    hkl_path = Path(hkl_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    data = XDSData(hkl_path, hkl_transform="identity")
    model = parse_model(model_path)
    rotations = _symmetry_rotation_contract(model.get("symmetry_ops"))
    result = candidate_invariant_split_membership(
        data.raw_hkl,
        model.get("symmetry_ops"),
        transforms,
        free_fraction=free_fraction,
        seed=seed,
        include_friedel=include_friedel,
    )
    membership = np.asarray(result["membership"], dtype=np.uint8)
    group_by_row = np.asarray(result["orbit_group_by_row"], dtype=int)
    raw_hkl_rows = [[int(value) for value in row] for row in data.raw_hkl]
    payload = {
        "schema_version": 1,
        "kind": "grond_transform_hand_split_contract",
        "algorithm": "candidate_union_model_symmetry_friedel_orbits_v1",
        "dataset": str(dataset),
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": sha256(hkl_path),
        "source_model": str(model_path),
        "source_model_sha256": sha256(model_path),
        "source_raw_hkl_sha256": _canonical_json_sha256(raw_hkl_rows),
        "model_symmetry_rotations": rotations,
        "model_symmetry_rotations_sha256": _canonical_json_sha256(rotations),
        "candidate_transforms": list(result["candidate_transforms"]),
        "candidate_transforms_sha256": _canonical_json_sha256(list(result["candidate_transforms"])),
        "include_friedel": bool(include_friedel),
        "free_fraction": float(free_fraction),
        "split_seed": int(seed),
        "row_count": int(len(membership)),
        "group_count": int(result["group_count"]),
        "work_group_count": int(result["work_group_count"]),
        "free_group_count": int(result["free_group_count"]),
        "work_row_count": int(np.count_nonzero(membership == 0)),
        "free_row_count": int(np.count_nonzero(membership == 1)),
        "membership_encoding": "0=work,1=free; ordered raw numeric reflection rows",
        "membership": [int(value) for value in membership],
        "membership_sha256": hashlib.sha256(membership.tobytes()).hexdigest(),
        "orbit_group_by_row": [int(value) for value in group_by_row],
        "orbit_group_by_row_sha256": _canonical_json_sha256([int(value) for value in group_by_row]),
    }
    payload["contract_payload_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_transform_split_contract(
    payload,
    *,
    source_hkl: Path | None = None,
    current_transform: str = "",
    model=None,
) -> dict:
    """Validate the split payload and optionally bind it to live data/model."""

    from grond.hkl_transform import parse_hkl_transform_info

    if not isinstance(payload, dict):
        raise ValueError("transform/hand split contract must contain one JSON object")
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("transform/hand split contract schema_version must be 1")
    if payload.get("kind") != "grond_transform_hand_split_contract":
        raise ValueError("transform/hand split contract kind must be 'grond_transform_hand_split_contract'")
    if payload.get("algorithm") != "candidate_union_model_symmetry_friedel_orbits_v1":
        raise ValueError("unsupported transform/hand split-contract algorithm")
    recorded_contract_hash = str(payload.get("contract_payload_sha256") or "")
    without_hash = dict(payload)
    without_hash.pop("contract_payload_sha256", None)
    if recorded_contract_hash != _canonical_json_sha256(without_hash):
        raise ValueError("transform/hand split contract payload hash mismatch")

    membership = np.asarray(payload.get("membership"), dtype=np.int64)
    groups = np.asarray(payload.get("orbit_group_by_row"), dtype=np.int64)
    row_count = int(payload.get("row_count") or -1)
    if membership.shape != (row_count,) or groups.shape != (row_count,):
        raise ValueError("transform/hand split membership/group lengths do not match row_count")
    if not np.all(np.isin(membership, (0, 1))):
        raise ValueError("transform/hand split membership must contain only 0=work and 1=free")
    if hashlib.sha256(membership.astype(np.uint8).tobytes()).hexdigest() != str(payload.get("membership_sha256") or ""):
        raise ValueError("transform/hand split membership hash mismatch")
    if _canonical_json_sha256([int(value) for value in groups]) != str(payload.get("orbit_group_by_row_sha256") or ""):
        raise ValueError("transform/hand split orbit-group hash mismatch")
    for group_id in np.unique(groups):
        local = membership[groups == group_id]
        if len(np.unique(local)) != 1:
            raise ValueError("one symmetry/Friedel orbit group crosses the work/free boundary")
    work_rows = int(np.count_nonzero(membership == 0))
    free_rows = int(np.count_nonzero(membership == 1))
    if work_rows != int(payload.get("work_row_count") or -1) or free_rows != int(payload.get("free_row_count") or -1):
        raise ValueError("transform/hand split recorded row counts do not match membership")
    transforms = [parse_hkl_transform_info(value).canonical for value in payload.get("candidate_transforms") or []]
    if not transforms:
        raise ValueError("transform/hand split contract has no candidate transforms")
    if _canonical_json_sha256(transforms) != str(payload.get("candidate_transforms_sha256") or ""):
        raise ValueError("transform/hand split candidate-transform hash mismatch")
    if current_transform:
        current = parse_hkl_transform_info(current_transform).canonical
        if current not in transforms:
            raise ValueError(f"active transform {current!r} is absent from the immutable split contract")
    if source_hkl is not None:
        source_hkl = Path(source_hkl).expanduser().resolve()
        if not source_hkl.is_file() or sha256(source_hkl) != str(payload.get("source_hkl_sha256") or ""):
            raise ValueError("transform/hand split source-HKL hash mismatch")
        data = XDSData(source_hkl, hkl_transform="identity")
        if len(data.raw_hkl) != row_count:
            raise ValueError("transform/hand split row count does not match live source HKL")
        raw_rows = [[int(value) for value in row] for row in data.raw_hkl]
        if _canonical_json_sha256(raw_rows) != str(payload.get("source_raw_hkl_sha256") or ""):
            raise ValueError("transform/hand split ordered raw-HKL hash mismatch")
    if model is not None:
        rotations = _symmetry_rotation_contract(model.get("symmetry_ops"))
        if _canonical_json_sha256(rotations) != str(payload.get("model_symmetry_rotations_sha256") or ""):
            raise ValueError("transform/hand split model-symmetry rotation hash mismatch")
    return {
        "status": "validated",
        "schema_version": 1,
        "algorithm": str(payload["algorithm"]),
        "dataset": str(payload.get("dataset") or ""),
        "row_count": row_count,
        "group_count": int(payload.get("group_count") or 0),
        "work_row_count": work_rows,
        "free_row_count": free_rows,
        "free_fraction": float(payload["free_fraction"]),
        "split_seed": int(payload["split_seed"]),
        "candidate_transforms": transforms,
        "candidate_transforms_sha256": str(payload["candidate_transforms_sha256"]),
        "membership_sha256": str(payload["membership_sha256"]),
        "orbit_group_by_row_sha256": str(payload["orbit_group_by_row_sha256"]),
        "source_hkl_sha256": str(payload["source_hkl_sha256"]),
        "source_raw_hkl_sha256": str(payload["source_raw_hkl_sha256"]),
        "model_symmetry_rotations_sha256": str(payload["model_symmetry_rotations_sha256"]),
        "contract_payload_sha256": recorded_contract_hash,
        "masks": {
            "all": np.ones(row_count, dtype=bool),
            "work": membership == 0,
            "free": membership == 1,
        },
    }


def load_transform_split_contract_masks(path: Path, data, model) -> tuple[dict, dict]:
    """Load a split contract and return full-length masks plus audit metadata."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    validation = validate_transform_split_contract(
        payload,
        source_hkl=data.path,
        current_transform=data.hkl_transform_spec,
        model=model,
    )
    masks = validation.pop("masks")
    validation.update({"path": str(path), "sha256": sha256(path)})
    return masks, validation
