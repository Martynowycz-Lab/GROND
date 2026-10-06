"""Parse signed HKL-axis permutations and describe their handedness.

A transform such as ``l,-k,h`` reindexes integrated rows into the model setting. Its
determinant (+1 proper, -1 improper) decides whether the model's coordinates must also be
inverted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

_AXES = {"h": 0, "k": 1, "l": 2}
_AXIS_NAMES = ("h", "k", "l")


@dataclass(frozen=True)
class HklTransformInfo:
    """Exact parity metadata for one signed HKL-axis permutation.

    ``matrix`` follows GROND's column-vector convention,
    ``h_model = matrix @ h_input``.  The determinant is therefore the
    handedness of the basis change that is applied to the integrated rows.
    """

    canonical: str
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    determinant: int
    parity: str
    proper: bool
    sign_paired_transform: str
    proper_canonical_transform: str
    coordinate_inversion_required_for_proper_canonicalization: bool

    def to_dict(self) -> dict[str, object]:
        """Return the fields as a plain JSON-ready dictionary."""

        return asdict(self)


def _determinant_3x3(matrix: tuple[tuple[int, int, int], ...]) -> int:
    """Return the determinant of an integer 3x3 matrix."""

    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _canonical_from_matrix(matrix: tuple[tuple[int, int, int], ...]) -> str:
    """Spell a signed-permutation matrix as ``h,k,l``-style text, or ``identity``."""

    parts: list[str] = []
    for row in matrix:
        nonzero = [(index, value) for index, value in enumerate(row) if value]
        if len(nonzero) != 1 or nonzero[0][1] not in {-1, 1}:
            raise ValueError("HKL transform matrix must be a signed permutation")
        index, sign = nonzero[0]
        parts.append(("-" if sign < 0 else "") + _AXIS_NAMES[index])
    canonical = ",".join(parts)
    return "identity" if canonical == "h,k,l" else canonical


def _negate_matrix(
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Return the matrix with every entry sign-flipped (the sign-paired transform)."""

    return tuple(tuple(-value for value in row) for row in matrix)  # type: ignore[return-value]


def parse_hkl_transform_info(spec: str | None) -> HklTransformInfo:
    """Parse and fully characterize a signed HKL-axis permutation."""

    text = "identity" if spec is None else str(spec).strip()
    if text.lower() in {"", "identity", "none", "h,k,l", "(h,k,l)"}:
        matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    else:
        cleaned = text.lower().replace("(", "").replace(")", "").replace(" ", "")
        parts = cleaned.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"HKL transform must look like 'h,k,l' or 'l,-k,h', got {spec!r}"
            )
        rows: list[tuple[int, int, int]] = []
        used: set[int] = set()
        for part in parts:
            sign = 1
            token = part
            if token.startswith("+"):
                token = token[1:]
            elif token.startswith("-"):
                sign = -1
                token = token[1:]
            if token not in _AXES:
                raise ValueError(f"Unknown HKL transform axis {part!r} in {spec!r}")
            index = _AXES[token]
            if index in used:
                raise ValueError(
                    f"HKL transform must be a permutation, got repeated axis in {spec!r}"
                )
            used.add(index)
            row = [0, 0, 0]
            row[index] = sign
            rows.append(tuple(row))
        matrix = tuple(rows)  # type: ignore[assignment]

    determinant = _determinant_3x3(matrix)
    if determinant not in {-1, 1}:
        raise ValueError(
            f"HKL transform must have determinant +/-1, got det={determinant} for {spec!r}"
        )
    canonical = _canonical_from_matrix(matrix)
    sign_paired = _canonical_from_matrix(_negate_matrix(matrix))
    proper_canonical = canonical if determinant == 1 else sign_paired
    return HklTransformInfo(
        canonical=canonical,
        matrix=matrix,
        determinant=determinant,
        parity="proper" if determinant == 1 else "improper",
        proper=determinant == 1,
        sign_paired_transform=sign_paired,
        proper_canonical_transform=proper_canonical,
        coordinate_inversion_required_for_proper_canonicalization=determinant == -1,
    )


def _determinant_float_3x3(matrix: list[list[float]]) -> float:
    """Return the determinant of a real 3x3 matrix."""

    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def source_orientation_contract(
    hkl_path: str | Path,
    transform: str | None,
) -> dict[str, object]:
    """Audit XDS-compatible real-space cell axes before and after reindexing."""

    path = Path(hkl_path).expanduser()
    info = parse_hkl_transform_info(transform)
    base: dict[str, object] = {
        "source_hkl": str(path),
        "transform": info.canonical,
        "transform_determinant": info.determinant,
        "source_basis_determinant": None,
        "source_basis_sign": 0,
        "transformed_basis_determinant": None,
        "transformed_basis_sign": 0,
        "right_handed_source": False,
        "right_handed_transformed": False,
        "absolute_claim_geometry_ready": False,
    }
    if not path.is_file():
        return {**base, "status": "missing_hkl"}
    header: dict[str, str] = {}
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                if not line.startswith("!"):
                    break
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    header[key.strip().upper()] = value.strip()
    except OSError as exc:
        return {**base, "status": "unreadable_hkl", "error": str(exc)}
    axes: list[list[float]] = []
    for key in ("UNIT_CELL_A-AXIS", "UNIT_CELL_B-AXIS", "UNIT_CELL_C-AXIS"):
        try:
            values = [float(value) for value in header.get(key, "").split()[:3]]
        except ValueError:
            values = []
        if len(values) != 3:
            return {
                **base,
                "status": "missing_source_axes",
                "missing_or_invalid_axis": key,
            }
        axes.append(values)
    # Header records are vectors; put them into matrix columns.
    matrix = [[axes[column][row] for column in range(3)] for row in range(3)]
    determinant = _determinant_float_3x3(matrix)
    transformed = determinant * info.determinant
    tolerance = 1.0e-12
    source_sign = 1 if determinant > tolerance else (-1 if determinant < -tolerance else 0)
    transformed_sign = (
        1 if transformed > tolerance else (-1 if transformed < -tolerance else 0)
    )
    source_right = source_sign == 1
    transformed_right = transformed_sign == 1
    status = (
        "right_handed"
        if source_right and transformed_right
        else (
            "left_handed_source"
            if source_sign == -1
            else (
                "left_handed_after_transform"
                if transformed_sign == -1
                else "degenerate_source_axes"
            )
        )
    )
    return {
        **base,
        "status": status,
        "source_real_space_axis_matrix_columns": matrix,
        "source_basis_determinant": determinant,
        "source_basis_sign": source_sign,
        "transformed_basis_determinant": transformed,
        "transformed_basis_sign": transformed_sign,
        "right_handed_source": source_right,
        "right_handed_transformed": transformed_right,
        "absolute_claim_geometry_ready": source_right and transformed_right,
    }


def absolute_configuration_claim_contract(
    spec: str | None,
    *,
    externally_calibrated_parity: bool = False,
    calibration_reference: str = "",
    calibration_validation_status: str = "",
) -> dict[str, object]:
    """Describe what configuration claim is identifiable for this transform.

    An improper transform is the exact gauge partner of coordinate inversion:
    ``(model, T) == (inverted model, -T)``.  It must therefore be
    canonicalized to the proper ``-T`` representation before any named
    absolute configuration is reported.  A proper operator alone still does
    not calibrate detector/acquisition/index parity; that requires an external
    reference retained in the provenance ledger.
    """

    info = parse_hkl_transform_info(spec)
    calibration_attached = bool(externally_calibrated_parity and calibration_reference)
    calibrated = bool(
        calibration_attached
        and info.proper
        and calibration_validation_status == "validated"
    )
    if info.determinant == -1:
        status = "parity_unresolved"
        reason = (
            "selected HKL transform is improper (determinant -1); coordinate inversion and "
            "transform negation are gauge-equivalent, so named absolute configuration is forbidden "
            "until the model is inverted and the transform is canonicalized to the proper sign pair"
        )
    elif calibrated:
        status = "absolute_configuration_calibrated"
        reason = "proper transform is bound to an external acquisition/index-parity calibration"
    elif calibration_attached:
        status = "external_calibration_attached_unvalidated"
        reason = (
            "an external calibration reference is attached but has not passed the required "
            "setup/decision-hash/parity/selection-independence schema; named absolute "
            "configuration remains forbidden"
        )
    else:
        status = "conditional_inversion_preference"
        reason = (
            "selected HKL transform is proper, but detector/acquisition/index parity has not been "
            "externally calibrated; report inversion preference, not a blind named configuration"
        )
    return {
        "schema_version": 1,
        "selected_transform": info.canonical,
        "transform": info.to_dict(),
        "configuration_status": status,
        "absolute_configuration_claim_allowed": calibrated,
        "externally_calibrated_parity": calibrated,
        "calibration_validation_status": (
            "validated" if calibrated else calibration_validation_status or "not_attached"
        ),
        "calibration_reference": calibration_reference if calibrated else "",
        "reason": reason,
        "gauge_equivalence": "(model,T) == (inverted_model,-T)",
    }
