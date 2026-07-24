from __future__ import annotations

from collections.abc import Mapping
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
        return asdict(self)


def _determinant_3x3(matrix: tuple[tuple[int, int, int], ...]) -> int:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _canonical_from_matrix(matrix: tuple[tuple[int, int, int], ...]) -> str:
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


def hkl_transform_determinant(spec: str | None) -> int:
    """Return +1 for proper and -1 for improper signed permutations."""

    return parse_hkl_transform_info(spec).determinant


def canonicalize_hkl_transform(spec: str | None) -> str:
    """Return GROND's stable spelling for a signed permutation."""

    return parse_hkl_transform_info(spec).canonical


def _determinant_float_3x3(matrix: list[list[float]]) -> float:
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


def validate_external_parity_calibration(
    payload: Mapping[str, object],
    *,
    setup_id: str,
    proper_transform: str,
    transform_decision_sha256: str,
    matched_hand_decision_sha256: str,
    matched_selected_hand: str,
) -> dict[str, object]:
    """Validate the minimal post-selection external parity-calibration schema.

    This boundary intentionally contains no ranking logic.  It can only bind a
    calibrant after the transform and matched-hand decisions have been frozen
    and hashed.
    """

    required_text = (
        "setup_id",
        "calibrant_dataset_id",
        "calibrant_physical_crystal_id",
        "calibrant_acquisition_fingerprint",
        "setup_manifest_sha256",
        "fixed_transform",
        "transform_decision_sha256",
        "matched_hand_decision_sha256",
        "expected_known_configuration",
        "expected_known_branch",
        "observed_branch",
    )
    missing = [key for key in required_text if not str(payload.get(key) or "").strip()]
    if missing:
        raise ValueError(
            "external parity calibration is missing required field(s): "
            + ", ".join(missing)
        )
    if str(payload.get("kind") or "") != "grond_external_parity_calibration":
        raise ValueError(
            "external parity calibration kind must be 'grond_external_parity_calibration'"
        )
    for key in ("calibrant_acquisition_fingerprint", "setup_manifest_sha256"):
        value = str(payload.get(key) or "").strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(
                f"external parity calibration {key} must be one SHA-256 hex digest"
            )
    try:
        schema_version = int(payload.get("schema_version") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("external parity calibration schema_version must be 1") from exc
    if schema_version != 1:
        raise ValueError("external parity calibration schema_version must be 1")
    if str(payload["setup_id"]) != str(setup_id):
        raise ValueError(
            f"external parity calibration setup_id mismatch: {payload['setup_id']!r} != {setup_id!r}"
        )
    fixed_info = parse_hkl_transform_info(str(payload["fixed_transform"]))
    expected_info = parse_hkl_transform_info(proper_transform)
    try:
        fixed_determinant = int(payload.get("fixed_transform_determinant") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "external parity calibration fixed_transform_determinant must be +1"
        ) from exc
    if not fixed_info.proper or fixed_determinant != 1:
        raise ValueError(
            "external parity calibration fixed transform and recorded determinant must both be proper (+1)"
        )
    if fixed_info.canonical != expected_info.canonical:
        raise ValueError(
            "external parity calibration fixed transform does not match the canonical endpoint transform"
        )
    if str(payload["transform_decision_sha256"]) != str(transform_decision_sha256):
        raise ValueError("external parity calibration transform-decision hash mismatch")
    if str(payload["matched_hand_decision_sha256"]) != str(
        matched_hand_decision_sha256
    ):
        raise ValueError("external parity calibration matched-hand-decision hash mismatch")
    if payload.get("reserved_from_validation") is not True:
        raise ValueError(
            "external parity calibrant must declare reserved_from_validation: true"
        )
    try:
        source_determinant = float(payload.get("source_orientation_determinant"))
        transformed_determinant = float(
            payload.get("transformed_orientation_determinant")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "external parity calibration must record numeric source_orientation_determinant "
            "and transformed_orientation_determinant"
        ) from exc
    if (
        source_determinant <= 0.0
        or transformed_determinant <= 0.0
        or payload.get("source_orientation_right_handed") is not True
    ):
        raise ValueError(
            "external parity calibration source and transformed orientation must both be "
            "explicitly right-handed with positive determinants"
        )
    expected_branch = str(payload["expected_known_branch"])
    observed_branch = str(payload["observed_branch"])
    if expected_branch not in {"original", "alternate"} or observed_branch not in {
        "original",
        "alternate",
    }:
        raise ValueError(
            "expected_known_branch and observed_branch must be original or alternate"
        )
    if observed_branch != matched_selected_hand:
        raise ValueError(
            "external parity calibration observed branch does not match the frozen hand decision"
        )
    setup_parity_multiplier = 1 if observed_branch == expected_branch else -1
    sign_mapping = payload.get("observed_sign_mapping")
    if not isinstance(sign_mapping, Mapping):
        raise ValueError("external parity calibration observed_sign_mapping must be an object")
    try:
        original_sign = int(sign_mapping["original"])
        alternate_sign = int(sign_mapping["alternate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "observed_sign_mapping must contain integer original and alternate signs"
        ) from exc
    if original_sign != 1 or alternate_sign != -1:
        raise ValueError(
            "observed_sign_mapping must use the fixed coordinate convention "
            "original=+1 and alternate=-1"
        )
    independence = payload.get("selection_independence")
    if not isinstance(independence, Mapping):
        raise ValueError("external parity calibration selection_independence must be an object")
    required_independence = {
        "calibration_used_for_transform_candidate_generation": False,
        "known_configuration_used_for_transform_ranking": False,
        "calibration_attached_after_transform_and_hand_frozen": True,
    }
    for key, expected in required_independence.items():
        if independence.get(key) is not expected:
            raise ValueError(
                f"external parity calibration selection_independence.{key} must be {expected}"
            )
    return {
        "status": "validated",
        "schema_version": 1,
        "setup_id": str(payload["setup_id"]),
        "calibrant_dataset_id": str(payload["calibrant_dataset_id"]),
        "calibrant_physical_crystal_id": str(
            payload["calibrant_physical_crystal_id"]
        ),
        "calibrant_acquisition_fingerprint": str(
            payload["calibrant_acquisition_fingerprint"]
        ),
        "setup_manifest_sha256": str(payload["setup_manifest_sha256"]),
        "fixed_transform": fixed_info.canonical,
        "fixed_transform_determinant": 1,
        "expected_known_configuration": str(payload["expected_known_configuration"]),
        "expected_known_branch": expected_branch,
        "observed_branch": observed_branch,
        "setup_parity_multiplier": setup_parity_multiplier,
        "observed_sign_mapping": {
            "original": original_sign,
            "alternate": alternate_sign,
        },
        "reserved_from_validation": True,
        "source_orientation_determinant": source_determinant,
        "transformed_orientation_determinant": transformed_determinant,
        "source_orientation_right_handed": True,
        "selection_independence": dict(required_independence),
    }
