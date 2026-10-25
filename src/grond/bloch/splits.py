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
from grond.validation import heldout

SPLIT_SCHEMES = ("orbit-hash",)
ORBIT_HASH_SPLIT_ALGORITHM = "candidate_union_model_symmetry_friedel_orbit_hash_v1"
# Every crystal's split contract is built from one shared candidate set (the closure of all crystals' candidates).
SHARED_CANDIDATE_POLICY = "shared_relative_candidates_closed_modulo_model_laue_group_v1"
# The transform scout scouts one representative per point-group equivalence class of the retained candidates.
TRANSFORM_EQUIVALENCE_POLICY = "model_point_group_proper_rotations_v1"
# Finite subgroups of GL(3, Z) have at most 48 elements (m-3m); a larger closure is not a lattice symmetry.
MAX_FINITE_INTEGER_GROUP_ORDER = 48


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


def load_split_exclusion(path) -> dict:
    """Load a lockbox orbit exclusion list (hash verified) with its listing order."""

    path = Path(path).expanduser().resolve()
    loaded = heldout.load_orbit_exclusion(path)
    ordered = [str(key) for key in json.loads(path.read_text())["lockbox_orbits"]]
    return {**loaded, "ordered_orbits": ordered, "path": str(path)}


def require_orbit_hash_work_and_free(
    *, work_rows: int, free_rows: int, orbit_count: int, salt: str, free_fraction: float
) -> None:
    """Refuse an orbit-hash split without free (or work) rows; one message for every code path.

    The hash fold of a small dataset can be empty, and a run must not continue with no held-out rows (or no rows
    to fit).
    """

    n_folds = max(2, int(round(1.0 / float(free_fraction))))
    rule = f"fold 0 of {n_folds} sha256(salt|key) folds, salt {str(salt)!r}, free fraction {float(free_fraction):g}"
    if int(free_rows) == 0:
        raise ValueError(
            f"orbit-hash free split is empty: none of the {int(orbit_count)} non-excluded Laue orbits lies in the "
            f"free fold ({rule}); this dataset is too small for a held-out set at this fraction. Use more "
            "reflections or a larger --free-fraction"
        )
    if int(work_rows) == 0:
        # "Held out as free" rather than "in the free fold": a candidate-union component is free as a
        # whole when any of its orbit keys is in the fold.
        raise ValueError(
            f"orbit-hash split has no work rows: every one of the {int(orbit_count)} non-excluded Laue orbits is "
            f"held out as free ({rule}); use more reflections or a smaller --free-fraction"
        )


def orbit_hash_split_masks(
    hkls,
    free_fraction=0.10,
    salt=heldout.DEFAULT_FREE_SALT,
    symmetry_ops=None,
    *,
    excluded_orbits=None,
    include_friedel: bool = True,
):
    """Crystal-independent work/free split keyed by the Laue-orbit key (grond.validation.heldout.orbit_key).

    A row is free iff its model-symmetry/Friedel orbit key hashes into the
    salted free fold (``heldout.free_mask``), so one orbit is held out in every
    crystal at once.  Rows whose orbit is in ``excluded_orbits`` are neither
    work nor free; callers drop them before anything is fitted.  An empty free
    (or work) set is refused, as in the transform/hand split contract.
    """

    keys = heldout.orbit_keys(hkls, symmetry_ops, include_friedel)
    free = heldout.free_mask(list(keys), str(salt), float(free_fraction))
    excluded_set = set(excluded_orbits or ())
    excluded = np.asarray([str(key) in excluded_set for key in keys], dtype=bool)
    masks = {
        "all": np.ones(len(keys), dtype=bool),
        "work": ~free & ~excluded,
        "free": free & ~excluded,
    }
    require_orbit_hash_work_and_free(
        work_rows=int(np.count_nonzero(masks["work"])),
        free_rows=int(np.count_nonzero(masks["free"])),
        orbit_count=len({str(key) for key in keys} - excluded_set),
        salt=str(salt),
        free_fraction=float(free_fraction),
    )
    return masks, excluded


def resolve_split_masks(
    data,
    model,
    *,
    split_contract: str = "",
    free_fraction: float = 0.10,
    split_scheme: str = "orbit-hash",
    split_salt: str = heldout.DEFAULT_FREE_SALT,
    exclude_orbits_json: str = "",
):
    """Return ``(masks, excluded, contract_summary, scheme_summary)`` for one native stage.

    An immutable split contract is authoritative; the declared scheme, salt,
    fraction and exclusion list must match it exactly.  Rows in
    ``excluded`` must be dropped from the row selection by the caller.
    """

    scheme = str(split_scheme)
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; expected one of {list(SPLIT_SCHEMES)}")
    exclusion = load_split_exclusion(exclude_orbits_json) if str(exclude_orbits_json or "").strip() else None
    contract_summary: dict = {}
    if str(split_contract or "").strip():
        masks, contract_summary = load_transform_split_contract_masks(Path(split_contract), data, model)
        excluded = np.asarray(masks.pop("excluded", np.zeros(len(masks["all"]), dtype=bool)), dtype=bool)
        if not math.isclose(float(free_fraction), float(contract_summary["free_fraction"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("--free-fraction does not match the immutable transform/hand split contract")
        if str(split_salt) != str(contract_summary.get("split_salt") or ""):
            raise ValueError("--split-salt does not match the immutable transform/hand split contract")
        recorded_exclusion = str(contract_summary.get("excluded_orbits_sha256") or "")
        requested_exclusion = str(exclusion["sha256"]) if exclusion is not None else ""
        if recorded_exclusion != requested_exclusion:
            raise ValueError(
                "--exclude-orbits-json does not match the orbit exclusion recorded in the immutable "
                f"transform/hand split contract (contract={recorded_exclusion or 'none'}, "
                f"requested={requested_exclusion or 'none'})"
            )
    else:
        masks, excluded = orbit_hash_split_masks(
            data.hkl,
            free_fraction,
            split_salt,
            symmetry_ops=model.get("symmetry_ops"),
            excluded_orbits=exclusion["orbits"] if exclusion is not None else None,
        )
    scheme_summary = {
        "split_scheme": scheme,
        "free_fraction": float(free_fraction),
        "split_salt": str(split_salt),
        "orbit_key": "grond.validation.heldout.orbit_key (model Laue group + Friedel, crystal-independent)",
        "split_contract": str(split_contract or ""),
        "excluded_orbits_sha256": str(exclusion["sha256"]) if exclusion is not None else "",
        "excluded_orbits_source": str(exclusion["path"]) if exclusion is not None else "",
        "excluded_source_rows": int(np.count_nonzero(excluded)),
    }
    return masks, excluded, contract_summary, scheme_summary


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


# ----------------------------------------------------------------------------- symmetry in the HKL basis
#
# Transforms follow GROND's column convention h_model = T @ h_raw.  A direct-space model operation x -> R x + t acts
# on Miller indices as h -> inv(R).T @ h (the orbit-key convention above).  Everything below works with those
# integer HKL-basis matrices, so centred, trigonal/hexagonal and cubic settings need no special cases.

_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_NEGATIVE_IDENTITY = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))


def _int_matrix(array) -> tuple:
    """Return a matrix as a hashable tuple of int tuples."""

    return tuple(tuple(int(value) for value in row) for row in np.asarray(array))


def _matrix_product(left, right) -> tuple:
    """Return the int64 product ``left @ right`` as a hashable tuple matrix."""

    return _int_matrix(np.asarray(left, dtype=np.int64) @ np.asarray(right, dtype=np.int64))


def _matrix_determinant(matrix) -> int:
    """Return the determinant of an integer matrix rounded to an int."""

    return int(round(float(np.linalg.det(np.asarray(matrix, dtype=float)))))


def _is_signed_permutation(matrix) -> bool:
    """Return True when every row and column holds exactly one entry of +1 or -1 and zeros elsewhere."""

    array = np.abs(np.asarray(matrix, dtype=np.int64))
    return bool(np.all(array <= 1) and np.all(array.sum(axis=0) == 1) and np.all(array.sum(axis=1) == 1))


def hkl_matrix_label(matrix) -> str:
    """GROND's transform spelling for a signed permutation, else rows such as 'h-k' (e.g. hexagonal 3-folds)."""

    from grond.hkl_transform import _canonical_from_matrix

    if _is_signed_permutation(matrix):
        return _canonical_from_matrix(_int_matrix(matrix))
    rows = []
    for row in _int_matrix(matrix):
        text = ""
        for coefficient, axis in zip(row, "hkl", strict=True):
            if coefficient == 0:
                continue
            sign = "-" if coefficient < 0 else ("+" if text else "")
            magnitude = "" if abs(coefficient) == 1 else str(abs(coefficient))
            text += f"{sign}{magnitude}{axis}"
        rows.append(text or "0")
    return ",".join(rows)


def transform_priority(spec: str) -> tuple[int, int, str]:
    """Fixed, data-independent order: identity, then proper and then improper signed permutations by spelling.

    For proper transforms this is the order of ``grond.workflows.decision.ALL_PROPER_SIGNED_PERMUTATIONS``.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    info = parse_hkl_transform_info(spec)
    return (0 if info.proper else 1, 0 if info.canonical == "identity" else 1, info.canonical)


def model_hkl_rotations(symmetry_ops) -> list[tuple]:
    """The model's rotation parts as integer HKL-basis matrices inv(R).T (identity always included)."""

    out = {_IDENTITY}
    for operation in symmetry_ops or []:
        rotation = operation.get("rot") if isinstance(operation, dict) else operation[0]
        array = np.asarray(rotation, dtype=float)
        if array.shape != (3, 3) or not np.all(np.isfinite(array)):
            raise ValueError("model symmetry rotation must be one finite 3x3 matrix")
        hkl_rotation = np.linalg.inv(array).T
        rounded = np.rint(hkl_rotation)
        if not np.allclose(hkl_rotation, rounded, rtol=0.0, atol=1e-7):
            raise ValueError("model symmetry rotation is not an integer matrix in the HKL basis")
        matrix = _int_matrix(rounded.astype(np.int64))
        if abs(_matrix_determinant(matrix)) != 1:
            raise ValueError("model symmetry rotation is not unimodular in the HKL basis")
        out.add(matrix)
    return sorted(out)


def _close_integer_group(generators, *, what: str) -> set[tuple]:
    """Every product of the generators; refuses a set that does not generate a finite lattice group."""

    group = {_IDENTITY, *(_int_matrix(generator) for generator in generators)}
    frontier = list(group)
    while frontier:
        found = []
        for left in frontier:
            for right in list(group):
                for product in (_matrix_product(left, right), _matrix_product(right, left)):
                    if product in group:
                        continue
                    group.add(product)
                    found.append(product)
                    if len(group) > MAX_FINITE_INTEGER_GROUP_ORDER:
                        raise ValueError(
                            f"{what} generate more than {MAX_FINITE_INTEGER_GROUP_ORDER} integer matrices, so they are "
                            "not the symmetries of one lattice; check the model cell and the transform candidates"
                        )
        frontier = found
    return group


def point_group_proper_rotations(symmetry_ops) -> list[tuple]:
    """Proper rotations (det +1) of the model point group in the HKL basis.

    Centrosymmetric models include -1 in their operators (SHELX LATT > 0 or a full CIF list); the proper part is the
    rotation subgroup (Pc: 1 only; P21/c: 1 and 2).  Improper point-group elements (mirrors, glides, -1) are also
    symmetries of the structure and give identical dynamical intensities, but an improper transform carries GROND's
    parity bookkeeping ((model, T) == (inverted model, -T)), so they are deliberately not used: a proper and an improper
    candidate are never merged, and improper candidates are merged only with each other, by proper rotations.
    """

    group = _close_integer_group(model_hkl_rotations(symmetry_ops), what="model symmetry rotations")
    return sorted(matrix for matrix in group if _matrix_determinant(matrix) == 1)


def laue_proper_rotations(symmetry_ops) -> list[tuple]:
    """Proper part of the model Laue group (point group plus -1) in the HKL basis.

    The orbit keys are invariant under the whole Laue group; with Friedel mates merged, its proper part suffices.
    """

    group = _close_integer_group(
        [*model_hkl_rotations(symmetry_ops), _NEGATIVE_IDENTITY], what="model symmetry rotations"
    )
    return sorted(matrix for matrix in group if _matrix_determinant(matrix) == 1)


def _unique_transform_matrices(transforms) -> list[tuple[str, tuple]]:
    """Return (canonical name, integer matrix) for each distinct HKL transform, in first-seen order."""

    from grond.hkl_transform import parse_hkl_transform_info

    out: list[tuple[str, tuple]] = []
    seen: set[str] = set()
    for transform in transforms:
        info = parse_hkl_transform_info(transform)
        if info.canonical not in seen:
            seen.add(info.canonical)
            out.append((info.canonical, _int_matrix(info.matrix)))
    return out


def transform_equivalence_classes(transforms, symmetry_ops) -> dict:
    """Collapse HKL transforms that are one indexing under the model point group's proper rotations.

    T and T' are equivalent when T' = g T for a proper rotation g of the model point group (HKL basis).  The model
    indices then differ by g, the crystal potential is invariant under the space-group operation behind g, and the
    Bloch Hamiltonians of the two indexings differ only by a diagonal phase (gauge) matrix, so every dynamical
    intensity is identical.  Genuinely distinct indexings (metric or pseudo-merohedral ambiguities, and polar
    alternatives such as the 2-fold of Pc) stay in separate classes, and a class never mixes proper and improper
    transforms (see ``point_group_proper_rotations``).  Each class is represented by the identity when it is a
    member, else by the first member in ``transform_priority`` order, so the choice never depends on input order.
    """

    rotations = point_group_proper_rotations(symmetry_ops)
    rotation_set = set(rotations)
    candidates = _unique_transform_matrices(transforms)
    if not candidates:
        raise ValueError("transform canonicalisation requires at least one transform candidate")
    matrices = dict(candidates)
    classes: list[list[str]] = []
    for name, matrix in candidates:
        for members in classes:
            # T' T^-1 with T a signed permutation (T^-1 = T.T).
            if _matrix_product(matrix, np.asarray(matrices[members[0]]).T) in rotation_set:
                members.append(name)
                break
        else:
            classes.append([name])
    records = []
    for members in classes:
        representative = min(members, key=transform_priority)
        inverse = np.asarray(matrices[representative]).T
        records.append(
            {
                "representative": representative,
                "members": members,
                "rotation_from_representative": {
                    name: hkl_matrix_label(_matrix_product(matrices[name], inverse)) for name in members
                },
            }
        )
    return {
        "policy": TRANSFORM_EQUIVALENCE_POLICY,
        "rule": (
            "T ~ T' iff T' = g T with g a proper rotation of the model point group in the HKL basis (derived from "
            "the model symmetry operators); equivalent indexings give identical dynamical intensities and only one "
            "representative per class is scouted: identity when present, else the first class member in the fixed "
            "proper-then-improper signed-permutation priority"
        ),
        "point_group_proper_rotations_hkl": [hkl_matrix_label(rotation) for rotation in rotations],
        "point_group_proper_order": len(rotations),
        "input_candidates": [name for name, _matrix in candidates],
        "classes": records,
        "representatives": [record["representative"] for record in records],
        "input_candidate_count": len(candidates),
        "class_count": len(records),
        "collapsed_candidate_count": len(candidates) - len(records),
    }


def shared_candidate_group_sha256(shared_group, laue_rotations) -> str:
    """Return the canonical-JSON SHA-256 of a shared candidate group and its model Laue rotation labels."""

    return _canonical_json_sha256(
        {"shared_group": [str(value) for value in shared_group], "laue": [str(value) for value in laue_rotations]}
    )


def shared_candidate_closure(retained_by_dataset, symmetry_ops) -> dict:
    """One candidate set for every crystal's split contract, so free/excluded status depends on the orbit alone.

    Each crystal's raw indices are in that crystal's own basis, so its candidates are compared in the model setting
    as relative transforms T T0^-1 (T0 = its first retained candidate).  The union of every crystal's relative
    candidates and the model Laue rotations, closed under composition, is a finite group A.  Crystal i's split
    candidates are the signed-axis elements of A times its T0 (this contains all its retained candidates).  For any
    transform the scout later selects, the model-setting orbit keys of a row are then the Laue-orbit keys of its whole
    A-orbit, identical in every crystal, so an orbit is free (or excluded) in all crystals or in none.  When A is the
    Laue group (no pseudo-symmetric candidates) every component is one Laue orbit.  When every crystal's
    raw basis is the model's (identity retained), A is the union of the retained candidates closed modulo the Laue
    group.
    """

    laue = laue_proper_rotations(symmetry_ops)
    generators = set(laue)
    datasets: dict[str, dict] = {}
    bases: dict[str, np.ndarray] = {}
    for name, transforms in retained_by_dataset.items():
        candidates = _unique_transform_matrices(transforms)
        if not candidates:
            raise ValueError(f"dataset {name!r} has no retained transform candidates for the split contract")
        bases[str(name)] = np.asarray(candidates[0][1], dtype=np.int64)
        relative = [_matrix_product(matrix, bases[str(name)].T) for _name, matrix in candidates]
        generators.update(relative)
        datasets[str(name)] = {
            "retained_candidates": [candidate for candidate, _matrix in candidates],
            "relative_to": candidates[0][0],
            "relative_candidates": [hkl_matrix_label(matrix) for matrix in relative],
        }
    group = _close_integer_group(
        generators, what="the crystals' relative transform candidates together with the model Laue rotations"
    )
    signed = sorted(
        (matrix for matrix in group if _is_signed_permutation(matrix)),
        key=lambda matrix: transform_priority(hkl_matrix_label(matrix)),
    )
    covered = {_matrix_product(rotation, matrix) for rotation in laue for matrix in signed}
    if covered != group:
        raise ValueError(
            "the shared transform-candidate closure has Laue cosets without a signed-axis permutation, so it cannot "
            "be written as a split candidate set; supply verified transforms with --trust-hkl-transform"
        )
    coset_representatives: list[tuple] = []
    seen: set[frozenset] = set()
    for matrix in signed:
        coset = frozenset(_matrix_product(rotation, matrix) for rotation in laue)
        if coset not in seen:
            seen.add(coset)
            coset_representatives.append(matrix)
    shared = [hkl_matrix_label(matrix) for matrix in signed]
    laue_labels = [hkl_matrix_label(rotation) for rotation in laue]
    for name, entry in datasets.items():
        coset_of_crystal = {hkl_matrix_label(_matrix_product(matrix, bases[name])) for matrix in signed}
        retained = list(entry["retained_candidates"])
        missing = sorted(set(retained) - coset_of_crystal)
        if missing:  # impossible by construction; fail closed rather than write a contract that omits a candidate
            raise ValueError(f"dataset {name!r}: the shared candidate closure omits retained candidates {missing}")
        entry["split_candidates"] = retained + sorted(coset_of_crystal - set(retained), key=transform_priority)
    return {
        "schema": "grond.split_candidate_union.v1",
        "policy": SHARED_CANDIDATE_POLICY,
        "rule": (
            "every crystal's relative candidates T T0^-1 (model setting) and the model Laue rotations generate one "
            "finite group A; crystal i's split candidates are the signed-axis elements of A times its T0, so the "
            "orbit-hash status of a row is a function of its model-setting A-orbit alone, identical in every crystal"
        ),
        "laue_proper_rotations_hkl": laue_labels,
        "laue_proper_order": len(laue),
        "shared_group": shared,
        "shared_group_order": len(group),
        "laue_coset_count": len(group) // len(laue),
        "laue_coset_representatives": [hkl_matrix_label(matrix) for matrix in coset_representatives],
        "pseudo_symmetric": len(group) > len(laue),
        "shared_group_sha256": shared_candidate_group_sha256(shared, laue_labels),
        "datasets": datasets,
    }


def contract_candidate_set(union: dict, dataset: str) -> dict | None:
    """The block a shared-closure split contract records (None for per-crystal contracts, which stay unchanged)."""

    if str(union.get("policy") or "") != SHARED_CANDIDATE_POLICY:
        return None
    entry = union["datasets"][dataset]
    return {
        "policy": SHARED_CANDIDATE_POLICY,
        "shared_group": list(union["shared_group"]),
        "laue_proper_rotations_hkl": list(union["laue_proper_rotations_hkl"]),
        "shared_group_sha256": str(union["shared_group_sha256"]),
        "laue_coset_count": int(union["laue_coset_count"]),
        "dataset_retained_candidates": list(entry["retained_candidates"]),
    }


def validate_contract_candidate_set(candidate_set, candidate_transforms) -> None:
    """A shared-closure contract's candidates must be its shared group times the crystal's first retained candidate."""

    from grond.hkl_transform import parse_hkl_transform_info

    if not isinstance(candidate_set, dict) or candidate_set.get("policy") != SHARED_CANDIDATE_POLICY:
        raise ValueError("transform/hand split contract has an unsupported candidate_set policy")
    shared = [str(value) for value in candidate_set.get("shared_group") or []]
    laue = [str(value) for value in candidate_set.get("laue_proper_rotations_hkl") or []]
    if not shared or shared_candidate_group_sha256(shared, laue) != str(candidate_set.get("shared_group_sha256") or ""):
        raise ValueError("transform/hand split contract shared candidate-group hash mismatch")
    retained = [parse_hkl_transform_info(value).canonical for value in candidate_set.get("dataset_retained_candidates") or []]
    if not retained or not set(retained) <= set(candidate_transforms):
        raise ValueError("transform/hand split contract candidates do not contain the crystal's retained candidates")
    base = np.asarray(parse_hkl_transform_info(retained[0]).matrix, dtype=np.int64)
    expected = {
        hkl_matrix_label(_matrix_product(parse_hkl_transform_info(value).matrix, base)) for value in shared
    }
    if expected != set(candidate_transforms) or len(expected) != len(candidate_transforms):
        raise ValueError(
            "transform/hand split contract candidates are not the shared candidate group times the crystal's first "
            "retained candidate"
        )


def row_orbit_key_sets(hkls, transforms, symmetry_ops) -> list[frozenset[str]]:
    """Laue-orbit keys (``heldout.orbit_key``, Friedel merged) of every row under every transform.

    Transforms that differ by a model rotation give the same key, so one per such class is evaluated.  Under a
    shared candidate closure (a crystal's split candidates on its raw rows, or the shared group on model-setting rows)
    each set is the row's whole model-setting orbit class, the same in every crystal.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    invariance = _key_invariance_group(symmetry_ops, True)
    representatives: list[np.ndarray] = []
    seen: set[frozenset] = set()
    for spec in transforms:
        matrix = _int_matrix(parse_hkl_transform_info(spec).matrix)
        coset = frozenset(_matrix_product(rotation, matrix) for rotation in invariance) if invariance else None
        if coset is None or coset not in seen:
            if coset is not None:
                seen.add(coset)
            representatives.append(np.asarray(matrix, dtype=np.int64))
    rows = np.asarray(hkls, dtype=np.int64).reshape(-1, 3)
    return [
        frozenset(heldout.orbit_key(matrix @ row, symmetry_ops, True) for matrix in representatives) for row in rows
    ]


def shared_orbit_split_masks(
    hkls,
    shared_group,
    free_fraction=0.10,
    salt=heldout.DEFAULT_FREE_SALT,
    symmetry_ops=None,
    *,
    excluded_orbits=None,
):
    """Model-setting orbit-hash masks under a shared candidate closure (rows of an HKL file in the model setting).

    A row is excluded when any Laue-orbit key of its shared-group orbit is excluded, else free when any is free:
    the membership every crystal's split contract gives the same reflection.  With no pseudo-symmetric candidates
    the shared group is the proper Laue group and this is exactly ``orbit_hash_split_masks``.
    """

    if not list(shared_group):
        raise ValueError("shared orbit split requires a non-empty shared candidate group")
    rows = np.asarray(hkls, dtype=np.int64).reshape(-1, 3)
    keys_by_row = row_orbit_key_sets(rows, shared_group, symmetry_ops)
    all_keys = sorted(set().union(*keys_by_row)) if keys_by_row else []
    free_by_key = dict(zip(all_keys, heldout.free_mask(all_keys, str(salt), float(free_fraction)), strict=True))
    excluded_set = set(excluded_orbits or ())
    excluded = np.asarray([bool(keys & excluded_set) for keys in keys_by_row], dtype=bool)
    free = np.asarray([any(bool(free_by_key[key]) for key in keys) for keys in keys_by_row], dtype=bool) & ~excluded
    masks = {"all": np.ones(len(rows), dtype=bool), "work": ~free & ~excluded, "free": free}
    require_orbit_hash_work_and_free(
        work_rows=int(np.count_nonzero(masks["work"])),
        free_rows=int(np.count_nonzero(masks["free"])),
        orbit_count=len(set(all_keys) - excluded_set),
        salt=str(salt),
        free_fraction=float(free_fraction),
    )
    return masks, excluded


def _key_invariance_group(symmetry_ops, include_friedel: bool = True) -> set[tuple] | None:
    """HKL rotations (with Friedel signs) under which both orbit-key functions are invariant, or None.

    The keys take the extreme over the listed model rotations, so they are invariant under that set only when it is
    closed under composition (always true for a complete SHELX or CIF operator list).
    """

    try:
        rotations = set(model_hkl_rotations(symmetry_ops))
    except ValueError:
        return None
    if any(_matrix_product(left, right) not in rotations for left in rotations for right in rotations):
        return None
    if not include_friedel:
        return rotations
    return rotations | {_matrix_product(_NEGATIVE_IDENTITY, rotation) for rotation in rotations}


def candidate_invariant_split_membership(
    raw_hkls,
    symmetry_ops,
    transforms,
    *,
    free_fraction: float = 0.10,
    include_friedel: bool = True,
    scheme: str = "orbit-hash",
    salt: str = heldout.DEFAULT_FREE_SALT,
    excluded_orbits=None,
):
    """Build one work/free split whose orbit groups are valid for every candidate.

    Rows are unioned whenever any candidate transform places them in the same
    model-symmetry/Friedel orbit.  Assigning whole union components to work or
    free therefore prevents a row (or one of its symmetry-equivalent
    duplicates) from changing split merely because a different indexing
    candidate is being scored.

    The orbit-hash scheme is crystal-independent: a component is free iff any
    of its model-orbit keys (``heldout.orbit_key`` under any candidate) lies in
    the salted free fold, and excluded (membership 2) iff any key is in
    ``excluded_orbits``.  With Laue-equivalent candidates every component is
    one orbit and the rule reduces to ``heldout.free_mask`` of that orbit.
    """

    from grond.hkl_transform import parse_hkl_transform_info

    raw = np.asarray(raw_hkls, dtype=int)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw HKLs must be an N by 3 integer array")
    if len(raw) < 2:
        raise ValueError("candidate-invariant held-out splitting requires at least two rows")
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("free_fraction must be between 0 and 1")
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; expected one of {list(SPLIT_SCHEMES)}")

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
    # Candidates that differ by a model rotation give every row the same orbit keys, so one per such class is keyed
    # (identical components and membership, fewer key evaluations under a shared candidate closure).
    invariance = _key_invariance_group(symmetry_ops, include_friedel)
    if invariance is not None:
        keyed: list[np.ndarray] = []
        for matrix in matrices:
            if not any(_matrix_product(matrix, kept.T) in invariance for kept in keyed):
                keyed.append(matrix)
        matrices = keyed

    parent = np.arange(len(raw), dtype=int)
    rank = np.zeros(len(raw), dtype=np.int8)

    def find(index: int) -> int:
        """Return the root of a row in the union-find forest, halving the path on the way."""

        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        """Merge the sets of two rows (union by rank)."""

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
    orbit_keys_by_row: list[set[str]] = [set() for _ in range(len(raw))]
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
            orbit_keys_by_row[row_index].add(heldout.orbit_key(hkl, symmetry_ops, include_friedel))

    components = {}
    for row_index in range(len(raw)):
        components.setdefault(find(row_index), []).append(row_index)
    ordered_components = sorted(components.values(), key=lambda rows: rows[0])
    if len(ordered_components) < 2:
        raise ValueError(
            "candidate-union symmetry/Friedel grouping leaves fewer than two independent reflection groups"
        )

    group_by_row = np.zeros(len(raw), dtype=int)
    membership = np.zeros(len(raw), dtype=np.uint8)
    all_keys = sorted(set().union(*orbit_keys_by_row))
    free_by_key = dict(zip(all_keys, heldout.free_mask(all_keys, str(salt), float(free_fraction)), strict=True))
    excluded_set = set(excluded_orbits or ())
    free_groups: set[int] = set()
    excluded_groups: set[int] = set()
    multi_orbit_groups = 0
    mixed_status_groups = 0
    # Orbit keys of the components that are not excluded: a free key merged into an excluded
    # component is not held out, so it must not be counted by the empty-free-set refusal.
    kept_keys: set[str] = set()
    group_orbit_keys: list[list[str]] = []
    for group_id, row_indices in enumerate(ordered_components):
        group_by_row[row_indices] = group_id
        keys = set().union(*(orbit_keys_by_row[row] for row in row_indices))
        group_orbit_keys.append(sorted(keys))
        multi_orbit_groups += int(len(keys) > 1)
        status = {bool(free_by_key[key]) for key in keys}
        mixed_status_groups += int(len(status) > 1)
        if keys & excluded_set:
            excluded_groups.add(group_id)
            membership[row_indices] = 2
            continue
        kept_keys |= keys
        if True in status:
            free_groups.add(group_id)
            membership[row_indices] = 1
    require_orbit_hash_work_and_free(
        work_rows=int(np.count_nonzero(membership == 0)),
        free_rows=int(np.count_nonzero(membership == 1)),
        orbit_count=len(kept_keys),
        salt=str(salt),
        free_fraction=float(free_fraction),
    )
    return {
        "candidate_transforms": canonical_transforms,
        "membership": membership,
        "orbit_group_by_row": group_by_row,
        "group_count": len(ordered_components),
        "work_group_count": len(ordered_components) - len(free_groups) - len(excluded_groups),
        "free_group_count": len(free_groups),
        "excluded_group_count": len(excluded_groups),
        "multi_orbit_group_count": multi_orbit_groups,
        "mixed_status_group_count": mixed_status_groups,
        # Model-setting Laue-orbit keys of every component (group id order); reported, not in the contract.
        "group_orbit_keys": group_orbit_keys,
    }


REALISED_FREE_FRACTION_WARN_RATIO = 1.5


def orbit_hash_realised_free_fraction(
    *,
    free_rows: int,
    rows: int,
    free_components: int,
    components: int,
    requested: float,
    mixed_status_group_count: int,
) -> dict:
    """Realised free fraction of a candidate-union orbit-hash contract, with its warnings.

    A component is free when any of its orbit keys is free, so components that merge orbits of
    different hash status (other indexings allowed by the metric) raise the free fraction above the
    requested one.  Membership is not changed here; the excess is recorded and reported.
    """

    # None (not NaN) for an empty denominator: the contract hash is strict JSON.
    row_fraction = float(free_rows) / float(rows) if int(rows) > 0 else None
    component_fraction = float(free_components) / float(components) if int(components) > 0 else None
    limit = REALISED_FREE_FRACTION_WARN_RATIO * float(requested)
    warnings = []
    if int(mixed_status_group_count) > 0:
        warnings.append(
            f"{int(mixed_status_group_count)} candidate-union components merge free and work orbit keys "
            "(indexing candidates that are not Laue-equivalent); each is held out whole"
        )
    for label, value in (("rows", row_fraction), ("components", component_fraction)):
        if value is not None and value > limit:
            warnings.append(
                f"realised free fraction of {label} {value:.4f} exceeds {REALISED_FREE_FRACTION_WARN_RATIO:g} x "
                f"the requested {float(requested):g}"
            )
    return {
        "requested": float(requested),
        "rows": row_fraction,
        "components": component_fraction,
        "free_rows": int(free_rows),
        "row_count": int(rows),
        "free_components": int(free_components),
        "component_count": int(components),
        "warn_ratio": REALISED_FREE_FRACTION_WARN_RATIO,
        "warnings": warnings,
    }


def build_transform_split_contract(
    hkl_path: Path,
    model_path: Path,
    transforms,
    *,
    dataset: str,
    free_fraction: float = 0.10,
    include_friedel: bool = True,
    scheme: str = "orbit-hash",
    salt: str = heldout.DEFAULT_FREE_SALT,
    excluded_orbits=None,
    candidate_set: dict | None = None,
) -> dict:
    """Build the immutable raw-row split shared by transform and hand scouts.

    ``excluded_orbits`` is the ordered orbit listing of a lockbox file (its
    SHA-256 is recorded and re-verified).  ``candidate_set`` is the
    shared-closure block (``contract_candidate_set``) when ``transforms`` is a
    crystal's share of the shared candidate closure; it is recorded and
    re-validated, and per-crystal contracts carry none.
    """

    hkl_path = Path(hkl_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    data = XDSData(hkl_path, hkl_transform="identity")
    model = parse_model(model_path)
    rotations = _symmetry_rotation_contract(model.get("symmetry_ops"))
    excluded_listing = [str(key) for key in (excluded_orbits or ())]
    result = candidate_invariant_split_membership(
        data.raw_hkl,
        model.get("symmetry_ops"),
        transforms,
        free_fraction=free_fraction,
        include_friedel=include_friedel,
        scheme=scheme,
        salt=salt,
        excluded_orbits=set(excluded_listing),
    )
    membership = np.asarray(result["membership"], dtype=np.uint8)
    group_by_row = np.asarray(result["orbit_group_by_row"], dtype=int)
    raw_hkl_rows = [[int(value) for value in row] for row in data.raw_hkl]
    payload = {
        "schema_version": 1,
        "kind": "grond_transform_hand_split_contract",
        "algorithm": ORBIT_HASH_SPLIT_ALGORITHM,
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
        # Field of the schema-1 contract layout; the orbit-hash membership does not depend on it.
        "split_seed": 42,
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
    realised = orbit_hash_realised_free_fraction(
        free_rows=payload["free_row_count"],
        rows=payload["row_count"],
        free_components=payload["free_group_count"],
        components=payload["group_count"],
        requested=float(free_fraction),
        mixed_status_group_count=int(result["mixed_status_group_count"]),
    )
    for warning in realised["warnings"]:
        print(f"WARN split {dataset}: {warning}", flush=True)
    payload.update(
        {
            "realised_free_fraction": realised,
            "split_scheme": "orbit-hash",
            "split_salt": str(salt),
            "orbit_key": "grond.validation.heldout.orbit_key (model Laue group + Friedel, crystal-independent)",
            "free_rule": (
                "component free iff any of its model-orbit keys under any candidate is in fold 0 of "
                "round(1/free_fraction) sha256(salt|key) folds; excluded iff any key is excluded"
            ),
            "membership_encoding": "0=work,1=free,2=excluded; ordered raw numeric reflection rows",
            "excluded_row_count": int(np.count_nonzero(membership == 2)),
            "excluded_group_count": int(result["excluded_group_count"]),
            "excluded_orbits": excluded_listing,
            "excluded_orbits_sha256": (
                hashlib.sha256("\n".join(excluded_listing).encode()).hexdigest() if excluded_listing else ""
            ),
            "multi_orbit_group_count": int(result["multi_orbit_group_count"]),
            "mixed_status_group_count": int(result["mixed_status_group_count"]),
        }
    )
    if candidate_set is not None:
        validate_contract_candidate_set(candidate_set, list(result["candidate_transforms"]))
        payload["candidate_set"] = dict(candidate_set)
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
    if payload.get("algorithm") != ORBIT_HASH_SPLIT_ALGORITHM:
        raise ValueError(
            f"unsupported transform/hand split-contract algorithm {payload.get('algorithm')!r}; only the orbit-hash "
            "split is supported"
        )
    if payload.get("split_scheme") != "orbit-hash":
        raise ValueError("orbit-hash transform/hand split contract must declare split_scheme 'orbit-hash'")
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
    excluded_listing = [str(key) for key in payload.get("excluded_orbits") or []]
    excluded_sha = str(payload.get("excluded_orbits_sha256") or "")
    if (excluded_listing or excluded_sha) and hashlib.sha256(
        "\n".join(excluded_listing).encode()
    ).hexdigest() != excluded_sha:
        raise ValueError("transform/hand split excluded-orbit listing hash mismatch")
    allowed_membership = (0, 1, 2) if excluded_listing else (0, 1)
    if not np.all(np.isin(membership, allowed_membership)):
        raise ValueError(
            "transform/hand split membership must contain only 0=work and 1=free"
            + (" and 2=excluded" if excluded_listing else "")
        )
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
    excluded_rows = int(np.count_nonzero(membership == 2))
    if excluded_rows != int(payload.get("excluded_row_count", -1)):
        raise ValueError("transform/hand split recorded excluded-row count does not match membership")
    transforms = [parse_hkl_transform_info(value).canonical for value in payload.get("candidate_transforms") or []]
    if not transforms:
        raise ValueError("transform/hand split contract has no candidate transforms")
    if _canonical_json_sha256(transforms) != str(payload.get("candidate_transforms_sha256") or ""):
        raise ValueError("transform/hand split candidate-transform hash mismatch")
    candidate_set = payload.get("candidate_set")
    if candidate_set is not None:
        validate_contract_candidate_set(candidate_set, transforms)
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
        if candidate_set is not None and [
            hkl_matrix_label(rotation) for rotation in laue_proper_rotations(model.get("symmetry_ops"))
        ] != list(candidate_set.get("laue_proper_rotations_hkl") or []):
            raise ValueError("transform/hand split shared candidate set was closed with another model Laue group")
    scheme_fields = {
        "split_scheme": "orbit-hash",
        "split_salt": str(payload.get("split_salt") or ""),
        "excluded_orbits_sha256": excluded_sha,
        "excluded_row_count": excluded_rows,
        "multi_orbit_group_count": int(payload.get("multi_orbit_group_count") or 0),
        "mixed_status_group_count": int(payload.get("mixed_status_group_count") or 0),
        # Absent from contracts written before the realised fraction was recorded.
        "realised_free_fraction": payload.get("realised_free_fraction"),
    }
    masks = {
        "all": np.ones(row_count, dtype=bool),
        "work": membership == 0,
        "free": membership == 1,
        "excluded": membership == 2,
    }
    return {
        **scheme_fields,
        **({"candidate_set": dict(candidate_set)} if candidate_set is not None else {}),
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
        "masks": masks,
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
