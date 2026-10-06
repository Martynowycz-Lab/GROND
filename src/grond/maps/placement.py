"""Put a model on the reference model's copy of the structure, in the same setting.

Two refinements of one crystal (for example a never-H and a full-H model) may
describe the molecule through different but equally valid copies: another
space-group operation, a lattice translation, or an allowed origin shift of the
same setting.  Figures need the models and their maps to overlay, so this module
finds the operation that brings a moving model onto a reference model and
records it.

Rules:

* Only the moving model's own setting is searched.  Its cell and space-group
  operations must match the reference; a change of axes is refused here (the
  never-H workflow converts settings before refinement).
* Candidate operations are the space-group operations of the reference combined
  with lattice translations (chosen per atom) and the allowed origin shifts of
  the group (discrete shifts plus continuous shifts along polar directions).
* In a non-centrosymmetric group an improper operation is never applied by
  default.  Placements composed with an inversion (the hand or absolute-structure
  change) are scored and reported, never applied.  An improper operation of the
  group itself (a glide in Pna21, say) maps the crystal onto itself, but is still
  only used with an explicit opt-in.  When either fits better than every
  applicable placement the placement is refused, even if an applicable one is
  within the RMSD limit (a pseudo-symmetric molecule can be fitted that way with
  the atoms assigned differently, which would overlay opposite hands).
* The placement is refused when the RMSD of the common non-H atoms is above a
  threshold (default 0.5 A).

A space-group operation (with lattice translations) maps the crystal, and hence
its periodic map, onto itself.  An origin shift does not: a map calculated from
the moving model has to be shifted with it.  ``map_origin_shift_fractional``
records that shift; it is zero when the operation is a symmetry of the crystal.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment

from grond.bloch.model import reciprocal_lengths, shelx_u_matrix

HYDROGEN_ELEMENTS = frozenset({"H", "D"})
# Origin shifts of all space groups lie on this grid (plus continuous shifts
# along polar directions, which are fitted separately).
ORIGIN_SHIFT_GRID = (0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75)
# A continuous polar-axis shift shorter than this is refinement noise between
# two models of one crystal; it is reported but not applied.
CONTINUOUS_SHIFT_MIN_A = 0.1
BOND_TOLERANCE_A = 0.4
_IMAGES = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
_TOL = 1e-5


class PlacementRefused(ValueError):
    """The moving model does not match the reference well enough to be placed."""

    def __init__(self, message: str, report: Mapping[str, object]):
        """Keep the message and a copy of the placement report."""

        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class PlacementOptions:
    """Controls for placing a model on a reference model."""

    rmsd_max_A: float = 0.5
    allow_improper_space_group_operations: bool = False
    cell_tolerance: float = 0.02


def _operations(model: Mapping[str, object]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a model's symmetry operations as (rotation, translation mod 1); identity when it lists none."""

    records = model.get("symmetry_ops") or []
    if not records:
        return [(np.eye(3), np.zeros(3))]
    return [
        (np.asarray(op["rot"], dtype=float), np.asarray(op["trans"], dtype=float) % 1.0)
        for op in records  # type: ignore[union-attr]
    ]


def _periodic_zero(vector: np.ndarray, tol: float = _TOL) -> bool:
    """Return True when a fractional vector is a lattice vector (each component within ``tol`` of an integer)."""

    vector = np.asarray(vector, dtype=float)
    return bool(np.all(np.abs(vector - np.rint(vector)) < tol))


def _find_operation(rotation, translation, operations) -> int | None:
    """Return the index of the operation equal to (rotation, translation) modulo lattice translations, else None."""

    for index, (other_rotation, other_translation) in enumerate(operations):
        if np.allclose(rotation, other_rotation, atol=1e-7) and _periodic_zero(translation - other_translation):
            return index
    return None


def operation_triplet(rotation: np.ndarray, translation: np.ndarray) -> str:
    """Return an x,y,z triplet; non-crystallographic shifts are written as decimals."""

    from grond.maps.native_dynamic import _op_triplet

    return _op_triplet({"rot": np.asarray(rotation, dtype=float), "trans": np.asarray(translation, dtype=float)})


def spacegroup_of(model: Mapping[str, object]) -> gemmi.SpaceGroup | None:
    """Return the gemmi space group generated by a model's operations, if any."""

    ops = [gemmi.Op(operation_triplet(rotation, translation)) for rotation, translation in _operations(model)]
    try:
        return gemmi.find_spacegroup_by_ops(gemmi.GroupOps(ops))
    except (RuntimeError, ValueError):
        return None


def is_centrosymmetric(operations: Sequence[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Return True when the operations include a rotation part of -I (an inversion)."""

    return any(np.allclose(rotation, -np.eye(3), atol=1e-7) for rotation, _ in operations)


def _free_basis(operations) -> np.ndarray:
    """Directions along which the origin is not fixed by the group (polar axes)."""

    constraints = np.vstack([np.eye(3) - rotation for rotation, _ in operations])
    _, singular, vt = np.linalg.svd(constraints)
    rank = int(np.sum(singular > 1e-7))
    return vt[rank:].T  # 3 x k


def allowed_origin_shifts(operations) -> tuple[list[np.ndarray], np.ndarray]:
    """Discrete origin shifts that keep the group, and the continuous free basis.

    A shift s is allowed when every operation (R, t) moved to the new origin,
    (R, t + (I - R) s), is again an operation of the group.  Shifts along the
    free (polar) directions are continuous; they are projected out here and
    fitted during scoring.
    """

    rotations = np.asarray([rotation for rotation, _ in operations], dtype=float)
    translations = np.asarray([translation for _, translation in operations], dtype=float)
    same_rotation = [
        np.where(np.all(np.abs(rotations - rotations[index]) < 1e-7, axis=(1, 2)))[0]
        for index in range(len(operations))
    ]
    free = _free_basis(operations)
    identity = np.eye(3)
    shifts: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate in itertools.product(ORIGIN_SHIFT_GRID, repeat=3):
        shift = np.asarray(candidate, dtype=float)
        if free.size:
            shift = shift - free @ (free.T @ shift)
        shift = shift % 1.0
        key = tuple(int(v) for v in np.rint(shift * 1200).astype(int) % 1200)
        if key in seen:
            continue
        moved = translations + np.einsum("gij,j->gi", identity - rotations, shift)
        if all(
            any(_periodic_zero(moved[index] - translations[other]) for other in same_rotation[index])
            for index in range(len(operations))
        ):
            seen.add(key)
            shifts.append(shift)
    return shifts, free


def _metric(cell) -> tuple[np.ndarray, np.ndarray]:
    """Return a cell's orthogonalisation matrix O (fractional to Cartesian Å) and its metric tensor O^T O."""

    orth = np.asarray(gemmi.UnitCell(*[float(v) for v in cell[:6]]).orth.mat, dtype=float)
    return orth, orth.T @ orth


def _cell_difference(left, right) -> float:
    """Return the largest relative cell-length difference or angle difference / 90 degrees, whichever is larger."""

    a = np.asarray(left[:6], dtype=float)
    b = np.asarray(right[:6], dtype=float)
    lengths = np.max(np.abs(a[:3] - b[:3]) / b[:3])
    angles = np.max(np.abs(a[3:] - b[3:])) / 90.0
    return float(max(lengths, angles))


def _same_operations(left, right) -> bool:
    """Return True when both lists hold the same operations (same count, each found modulo lattice translations)."""

    return len(left) == len(right) and all(
        _find_operation(rotation, translation, right) is not None for rotation, translation in left
    )


def _atom_xyz(atoms) -> np.ndarray:
    """Return the atoms' fractional coordinates as an (n, 3) array."""

    return np.asarray([[float(atom["x"]), float(atom["y"]), float(atom["z"])] for atom in atoms], dtype=float).reshape(
        -1, 3
    )


def _element(atom) -> str:
    """Return an atom's element symbol in upper case."""

    return str(atom.get("element", "")).strip().upper()


def _heavy_indices(atoms) -> list[int]:
    """Return the indices of the non-hydrogen atoms."""

    return [index for index, atom in enumerate(atoms) if _element(atom) not in HYDROGEN_ELEMENTS]


def _nearest_images(delta: np.ndarray, orth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals and integer shifts of the nearest lattice images."""

    base = np.rint(delta)
    offsets = (delta - base)[:, None, :] - _IMAGES[None, :, :]
    squared = np.sum((offsets @ orth.T) ** 2, axis=-1)
    best = np.argmin(squared, axis=1)
    rows = np.arange(len(delta))
    return offsets[rows, best], base[rows] + _IMAGES[best]


def _nearest_neighbour_pairs(moving_xyz, reference_xyz, compatible, orth) -> tuple[np.ndarray, np.ndarray]:
    """Return (moving, reference) index pairs of a minimum-distance assignment of element-compatible atoms."""

    delta = reference_xyz[None, :, :] - moving_xyz[:, None, :]
    delta -= np.rint(delta)
    offsets = delta[:, :, None, :] - _IMAGES[None, None, :, :]
    squared = np.sum((offsets @ orth.T) ** 2, axis=-1).min(axis=2)
    squared = np.where(compatible, squared, 1e12)
    rows, cols = linear_sum_assignment(squared)
    keep = squared[rows, cols] < 1e11
    return rows[keep], cols[keep]


def _score_candidate(
    moving_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    pairs: tuple[np.ndarray, np.ndarray] | None,
    compatible: np.ndarray | None,
    orth: np.ndarray,
    metric: np.ndarray,
    free: np.ndarray,
) -> dict[str, object] | None:
    """Place the moving atoms with one candidate operation and score them against the reference; None without pairs.

    Continuous origin shifts along the free (polar) directions are fitted by up to eight least-squares updates. The
    result holds the RMSD and largest deviation (Å), the atom pairs, their lattice images and distances, and the fitted
    continuous shift.
    """

    transformed = moving_xyz @ rotation.T + translation
    continuous = np.zeros(free.shape[1])

    def assign(placed):
        """Return the given atom pairs, or a nearest-neighbour assignment for the placed coordinates."""

        if pairs is not None:
            return pairs
        return _nearest_neighbour_pairs(placed, reference_xyz, compatible, orth)

    if free.size:
        normal = free.T @ metric @ free
        for _ in range(8):
            placed = transformed + free @ continuous
            rows, cols = assign(placed)
            if len(rows) == 0:
                return None
            residual, _ = _nearest_images(reference_xyz[cols] - placed[rows], orth)
            update = np.linalg.solve(normal, free.T @ metric @ residual.mean(axis=0))
            continuous = continuous + update
            if float(np.sqrt(update @ normal @ update)) < 1e-7:
                break
    placed = transformed + (free @ continuous if free.size else 0.0)
    rows, cols = assign(placed)
    if len(rows) == 0:
        return None
    residual, images = _nearest_images(reference_xyz[cols] - placed[rows], orth)
    distances = np.sqrt(np.sum((residual @ orth.T) ** 2, axis=1))
    return {
        "rmsd_A": float(np.sqrt(np.mean(distances**2))),
        "max_A": float(np.max(distances)),
        "rows": rows,
        "cols": cols,
        "images": images,
        "distances": distances,
        "continuous": continuous,
    }


def _candidates(operations, centrosymmetric: bool):
    """Yield (matrix, translation, sign, origin shift) for every distinct candidate.

    sign -1 composes the operation with an inversion through the origin; in a
    non-centrosymmetric group these candidates are the hand (absolute-structure)
    change and are only scored.
    """

    shifts, _ = allowed_origin_shifts(operations)
    seen: set[tuple[int, ...]] = set()
    signs = (1.0,) if centrosymmetric else (1.0, -1.0)
    for sign in signs:
        for rotation, translation in operations:
            for shift in shifts:
                matrix = sign * rotation
                vector = (sign * translation + shift) % 1.0
                key = tuple(np.rint(matrix).astype(int).ravel()) + tuple(np.rint(vector * 1200).astype(int) % 1200)
                if key in seen:
                    continue
                seen.add(key)
                yield matrix, vector, sign, shift


def _map_shift(rotation, translation, operations) -> np.ndarray | None:
    """Shift of the map needed for this operation; None when no group rotation matches."""

    matches = [trans for rot, trans in operations if np.allclose(rot, rotation, atol=1e-7)]
    if not matches:
        return None
    shift = (translation - matches[0] + 0.5) % 1.0 - 0.5
    for other in matches[1:]:
        candidate = (translation - other + 0.5) % 1.0 - 0.5
        if np.linalg.norm(candidate) < np.linalg.norm(shift):
            shift = candidate
    shift[np.abs(shift) < 1e-9] = 0.0
    return shift


def _transform_adp(atom: dict, rotation: np.ndarray, cell) -> None:
    """Rotate an anisotropic atom's Uij in place with a placement's fractional rotation (isotropic: unchanged)."""

    if atom.get("adp_type") != "anis":
        return
    scale = np.diag(reciprocal_lengths(cell))
    inverse = np.linalg.inv(scale)
    change = inverse @ rotation @ scale
    tensor = change @ shelx_u_matrix(atom) @ change.T
    for key, (i, j) in zip(
        ("u11", "u22", "u33", "u23", "u13", "u12"),
        ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
        strict=True,
    ):
        atom[key] = float(tensor[i, j])


def _follow_parents(placed: np.ndarray, anchored: dict[int, np.ndarray], orth: np.ndarray) -> dict[int, np.ndarray]:
    """Give each unmatched atom the lattice translation of its nearest matched atom."""

    anchors = sorted(anchored)
    shifts = dict(anchored)
    if not anchors:
        return {index: np.zeros(3) for index in range(len(placed))}
    anchor_xyz = placed[anchors]
    for index in range(len(placed)):
        if index in shifts:
            continue
        delta = placed[index][None, :] - anchor_xyz
        base = np.rint(delta)
        offsets = (delta - base)[:, None, :] - _IMAGES[None, :, :]
        squared = np.sum((offsets @ orth.T) ** 2, axis=-1)
        flat = int(np.argmin(squared))
        which, image = np.unravel_index(flat, squared.shape)
        # placed[index] - (base + image) sits next to the unshifted anchor.
        image_shift = base[which] + _IMAGES[image]
        shifts[index] = anchored[anchors[which]] - image_shift
    return shifts


def _label_pairs(moving_atoms, reference_atoms, moving_heavy, reference_heavy):
    """Return (moving, reference) indices of non-H atoms with the same label (case-insensitive) and element."""

    reference_by_label = {
        str(reference_atoms[index]["name"]).upper(): index for index in reference_heavy
    }
    rows, cols = [], []
    for index in moving_heavy:
        label = str(moving_atoms[index]["name"]).upper()
        other = reference_by_label.get(label)
        if other is not None and _element(reference_atoms[other]) == _element(moving_atoms[index]):
            rows.append(index)
            cols.append(other)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _operation_record(rotation, vector, continuous_vector, operations, sign, shift) -> dict[str, object]:
    """Return the JSON record of a placement: matrix, translation, triplet, properness, origin shifts, map shift."""

    total = vector + continuous_vector
    group_index = _find_operation(rotation, total, operations)
    map_shift = _map_shift(rotation, total, operations)
    return {
        "rotation": np.rint(rotation).astype(int).tolist(),
        "translation": [float(v) for v in total],
        "triplet": operation_triplet(rotation, total),
        "determinant": int(round(float(np.linalg.det(rotation)))),
        "proper": bool(np.linalg.det(rotation) > 0),
        "space_group_operation": group_index is not None,
        "discrete_origin_shift_fractional": [float(v) for v in shift],
        "continuous_origin_shift_fractional": [float(v) for v in continuous_vector],
        "inversion_composed": bool(sign < 0),
        "map_origin_shift_fractional": None if map_shift is None else [float(v) for v in map_shift],
        "map_transform_required": bool(map_shift is not None and not _periodic_zero(map_shift)),
    }


def place_model_on_reference(
    moving: Mapping[str, object],
    reference: Mapping[str, object],
    options: PlacementOptions | None = None,
) -> tuple[dict, dict[str, object]]:
    """Return a copy of ``moving`` on the reference copy of the structure, and a report.

    Raises :class:`PlacementRefused` (with the report) when the setting differs,
    no applicable operation brings the common non-H atoms within the RMSD
    limit, or an inverted placement (a hand change) or a disallowed improper
    operation fits better than every applicable one.
    """

    options = options or PlacementOptions()
    moving_atoms = list(moving.get("atoms") or [])
    reference_atoms = list(reference.get("atoms") or [])
    reference_ops = _operations(reference)
    moving_ops = _operations(moving)
    group = spacegroup_of(reference)
    centro = is_centrosymmetric(reference_ops)
    report: dict[str, object] = {
        "rmsd_max_A": float(options.rmsd_max_A),
        "space_group": group.xhm() if group else "unknown",
        "centrosymmetric": bool(centro),
        "moving_cell": [float(v) for v in moving["cell"]],
        "reference_cell": [float(v) for v in reference["cell"]],
        "cell_max_relative_difference": _cell_difference(moving["cell"], reference["cell"]),
        "allow_improper_space_group_operations": bool(options.allow_improper_space_group_operations),
        "definition": (
            "RMSD of common non-H atoms (label and element, else nearest-neighbour assignment) after a "
            "space-group operation, per-atom lattice translations and an allowed origin shift of the reference "
            "setting; in a non-centrosymmetric group an inverted placement (hand change) is only reported, and an "
            "improper space-group operation is applied only when explicitly allowed; either one fitting better "
            "than every applicable placement refuses the placement"
        ),
    }
    if report["cell_max_relative_difference"] > options.cell_tolerance:
        report["status"] = "refused_cell"
        raise PlacementRefused(
            "moving and reference cells differ by more than "
            f"{options.cell_tolerance:.0%}; put the model in the reference setting first",
            report,
        )
    if not _same_operations(moving_ops, reference_ops):
        report["status"] = "refused_space_group"
        raise PlacementRefused("moving and reference models have different space-group operations", report)

    orth, metric = _metric(reference["cell"])
    moving_xyz = _atom_xyz(moving_atoms)
    reference_xyz = _atom_xyz(reference_atoms)
    moving_heavy = _heavy_indices(moving_atoms)
    reference_heavy = _heavy_indices(reference_atoms)
    if not moving_heavy or not reference_heavy:
        report["status"] = "refused_no_atoms"
        raise PlacementRefused("placement needs non-H atoms in both models", report)
    label_rows, label_cols = _label_pairs(moving_atoms, reference_atoms, moving_heavy, reference_heavy)
    label_minimum = max(3, int(math.ceil(0.5 * min(len(moving_heavy), len(reference_heavy)))))
    report["label_matched_atoms"] = int(len(label_rows))
    report["label_matching_minimum"] = int(label_minimum)
    compatible = np.asarray(
        [[_element(moving_atoms[i]) == _element(reference_atoms[j]) for j in reference_heavy] for i in moving_heavy],
        dtype=bool,
    )
    candidates = list(_candidates(reference_ops, centro))
    free = _free_basis(reference_ops)

    def category(rotation, sign) -> str:
        """Classify a candidate as applicable, hand_change or improper_space_group_operation."""

        if centro:
            return "applicable"
        if sign < 0:
            return "hand_change"
        if np.linalg.det(rotation) > 0 or options.allow_improper_space_group_operations:
            return "applicable"
        return "improper_space_group_operation"

    def search(method: str):
        """Return the best-scoring candidate of each category, pairing atoms by label or by nearest neighbour.

        At equal RMSD (6 decimals) smaller origin shifts win, then smaller translations.
        """

        best: dict[str, tuple] = {}
        for rotation, vector, sign, shift in candidates:
            if method == "label":
                scored = _score_candidate(
                    moving_xyz, reference_xyz, rotation, vector, pairs=(label_rows, label_cols), compatible=None,
                    orth=orth, metric=metric, free=free,
                )
            else:
                scored = _score_candidate(
                    moving_xyz[moving_heavy], reference_xyz[reference_heavy], rotation, vector, pairs=None,
                    compatible=compatible, orth=orth, metric=metric, free=free,
                )
                if scored is not None:
                    scored["rows"] = np.asarray(moving_heavy)[scored["rows"]]
                    scored["cols"] = np.asarray(reference_heavy)[scored["cols"]]
            if scored is None:
                continue
            scored.update(rotation=rotation, vector=vector, sign=sign, shift=shift)
            # At equal RMSD prefer operations that need no map transform, then smaller shifts.
            key = (round(float(scored["rmsd_A"]), 6), float(np.linalg.norm(shift)), float(np.linalg.norm(vector)))
            label = category(rotation, sign)
            if label not in best or key < best[label][0]:
                best[label] = (key, scored)
        return best

    method = "label" if len(label_rows) >= label_minimum else "nearest_neighbour"
    found = search(method)
    applicable = found.get("applicable")
    report["label_search_rmsd_A"] = float(applicable[1]["rmsd_A"]) if (method == "label" and applicable) else None
    if method == "label" and (applicable is None or applicable[1]["rmsd_A"] > options.rmsd_max_A):
        method = "nearest_neighbour"
        found = search(method)
        applicable = found.get("applicable")
    report["matching"] = method
    report["candidates_evaluated"] = len(candidates)

    def summary(entry):
        """Return the RMSD, largest deviation, matched-atom count and operation of a search result, or None."""

        if entry is None:
            return None
        scored = entry[1]
        continuous_vector = free @ scored["continuous"] if free.size else np.zeros(3)
        return {
            "rmsd_A": float(scored["rmsd_A"]),
            "max_deviation_A": float(scored["max_A"]),
            "matched_atoms": int(len(scored["rows"])),
            "operation": _operation_record(
                scored["rotation"], scored["vector"], continuous_vector, reference_ops, scored["sign"], scored["shift"]
            ),
        }

    hand = found.get("hand_change")
    improper = found.get("improper_space_group_operation")
    report["best_applicable"] = summary(applicable)
    report["best_hand_change_not_applied"] = summary(hand)
    report["best_improper_space_group_operation_not_applied"] = summary(improper)

    def better(entry) -> bool:
        """Return True when a candidate is within the RMSD limit and fits better than the best applicable one."""

        return bool(
            entry is not None
            and entry[1]["rmsd_A"] <= options.rmsd_max_A
            and (applicable is None or entry[1]["rmsd_A"] < applicable[1]["rmsd_A"] - 1e-6)
        )

    report["hand_change_indicated"] = better(hand)
    report["improper_space_group_operation_fits_better"] = better(improper)
    applicable_fits = applicable is not None and applicable[1]["rmsd_A"] <= options.rmsd_max_A
    # An inverted (or disallowed improper) placement that fits better than every
    # applicable one means the moving model is the other hand (or the mirror
    # copy).  For a pseudo-symmetric molecule a proper operation can still fit
    # within the limit by assigning the atoms differently; applying it would
    # overlay opposite hands, so the placement is refused in that case too.
    if (
        not applicable_fits
        or report["hand_change_indicated"]
        or report["improper_space_group_operation_fits_better"]
    ):
        found_text = "no candidate" if applicable is None else f"best RMSD {applicable[1]['rmsd_A']:.3f} A"
        how = "better" if applicable_fits else "only"
        if report["hand_change_indicated"]:
            report["status"] = "refused_hand_change"
            raise PlacementRefused(
                f"the model matches the reference {how} after an inversion "
                f"(RMSD {hand[1]['rmsd_A']:.3f} A; applicable operations: {found_text}): a hand or "
                "absolute-structure change, reported and not applied",
                report,
            )
        if report["improper_space_group_operation_fits_better"]:
            report["status"] = "refused_improper_operation"
            raise PlacementRefused(
                f"the model matches the reference {how} through an improper space-group operation "
                f"(RMSD {improper[1]['rmsd_A']:.3f} A; applicable operations: {found_text}), which is not "
                "applied in a non-centrosymmetric group by default; pass --allow-improper-space-group-operations "
                "to use it",
                report,
            )
        report["status"] = "refused_rmsd"
        raise PlacementRefused(
            f"no applicable operation places the model within {options.rmsd_max_A:.3g} A of the reference "
            f"({found_text})",
            report,
        )

    best = applicable
    scored = best[1]
    rotation = scored["rotation"]
    continuous_vector = free @ scored["continuous"] if free.size else np.zeros(3)
    continuous_length = float(np.linalg.norm(orth @ continuous_vector))
    report["continuous_origin_shift_A"] = continuous_length
    if 0.0 < continuous_length < CONTINUOUS_SHIFT_MIN_A:
        report["continuous_origin_shift_applied"] = False
        report["continuous_origin_shift_note"] = (
            f"polar-axis shift of {continuous_length:.4f} A is below {CONTINUOUS_SHIFT_MIN_A} A and is not applied"
        )
        continuous_vector = np.zeros(3)
    else:
        report["continuous_origin_shift_applied"] = bool(continuous_length > 0.0)
    vector = scored["vector"] + continuous_vector
    placed = moving_xyz @ rotation.T + vector
    rows, cols = scored["rows"], scored["cols"]
    residual, images = _nearest_images(reference_xyz[cols] - placed[rows], orth)
    anchored = {int(row): image for row, image in zip(rows, images, strict=True)}
    lattice = _follow_parents(placed, anchored, orth)
    aligned = copy.deepcopy(dict(moving))
    aligned["atoms"] = copy.deepcopy(moving_atoms)
    for index, atom in enumerate(aligned["atoms"]):
        position = placed[index] + lattice[index]
        atom["x"], atom["y"], atom["z"] = (float(value) for value in position)
        _transform_adp(atom, rotation, moving["cell"])
    distances = np.sqrt(np.sum((residual @ orth.T) ** 2, axis=1))
    distinct = {tuple(int(v) for v in np.rint(shift)) for shift in lattice.values()}
    report.update(
        status="aligned",
        rmsd_A=float(np.sqrt(np.mean(distances**2))),
        max_deviation_A=float(np.max(distances)),
        matched_atoms=int(len(rows)),
        operation=_operation_record(rotation, scored["vector"], continuous_vector, reference_ops, scored["sign"],
                                    scored["shift"]),
        identity=bool(
            np.allclose(rotation, np.eye(3)) and _periodic_zero(vector) and len(distinct) == 1 and distinct == {(0, 0, 0)}
        ),
        lattice_translations={
            "distinct": sorted(list(item) for item in distinct),
            "common": list(next(iter(distinct))) if len(distinct) == 1 else None,
        },
        atom_pairs=[
            {
                "moving": str(moving_atoms[int(row)]["name"]),
                "reference": str(reference_atoms[int(col)]["name"]),
                "distance_A": float(distance),
            }
            for row, col, distance in zip(rows, cols, distances, strict=True)
        ],
    )
    # A better inverted or improper fit is refused above, so an applied placement carries no such warning.
    report["warnings"] = []
    return aligned, report


def assemble_molecules(model: Mapping[str, object]) -> tuple[dict, dict[str, object]]:
    """Move atoms by lattice translations so every bonded fragment is contiguous.

    Bonds are covalent-radius contacts (sum + 0.4 A) through the nearest lattice
    image.  Only whole lattice vectors are applied, so the crystal and every map
    of it are unchanged; the first atom of each fragment keeps its position.
    """

    atoms = list(model.get("atoms") or [])
    xyz = _atom_xyz(atoms)
    orth, _ = _metric(model["cell"])
    radii = np.asarray(
        [gemmi.Element(_element(atom).capitalize() or "C").covalent_r for atom in atoms], dtype=float
    )
    count = len(atoms)
    shifts = np.zeros((count, 3))
    visited = np.zeros(count, dtype=bool)
    fragments = 0
    for start in range(count):
        if visited[start]:
            continue
        fragments += 1
        visited[start] = True
        queue = [start]
        while queue:
            current = queue.pop()
            here = xyz[current] + shifts[current]
            delta = xyz - here
            base = np.rint(delta)
            offsets = (delta - base)[:, None, :] - _IMAGES[None, :, :]
            squared = np.sum((offsets @ orth.T) ** 2, axis=-1)
            best = np.argmin(squared, axis=1)
            distance = np.sqrt(squared[np.arange(count), best])
            limit = radii + radii[current] + BOND_TOLERANCE_A
            for neighbour in np.where((~visited) & (distance < limit) & (distance > 0.1))[0]:
                shifts[neighbour] = -(base[neighbour] + _IMAGES[best[neighbour]])
                visited[neighbour] = True
                queue.append(int(neighbour))
    moved = [index for index in range(count) if np.any(np.abs(shifts[index]) > 0.5)]
    assembled = copy.deepcopy(dict(model))
    assembled["atoms"] = copy.deepcopy(atoms)
    for index in moved:
        atom = assembled["atoms"][index]
        atom["x"], atom["y"], atom["z"] = (float(v) for v in xyz[index] + shifts[index])
    return assembled, {
        "fragments": int(fragments),
        "atoms_moved_by_lattice_translation": [str(atoms[index]["name"]) for index in moved],
        "already_contiguous": not moved,
        "definition": "bonded fragments made contiguous with whole lattice translations only",
    }
