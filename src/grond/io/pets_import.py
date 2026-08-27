"""Import a native PETS2 ``cif_pets`` observation table into GROND.

This bridge deliberately consumes the ordinary PETS2 integration file: one
measured signed reflection assigned to one measured zone axis.  It refuses a
virtual-frame dynamical file when the same signed reflection has been copied
into more than one zone, because those copies are not independent
observations.

PETS writes the frame orientation as ``Rz(omega) Rx(alpha) Ry(beta) UB`` in a
coordinate system whose incident beam points along ``-z``.  GROND's existing
scan-varying-A path can represent that geometry without inventing detector
coordinates or virtual frames.  The nominal scan rotation is placed about the
beam direction (which cannot change excitation errors); the PETS frame A
matrices therefore remain the sole source of the physical orientation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from grond.hkl_transform import parse_hkl_transform_info

_AXIS_POSITION = re.compile(
    r"(?:rotation|tilt)\s+axis\s+position:\s*"
    r"([-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PetsImportOutput:
    out_dir: Path
    hkl: Path
    scan_points: Path
    summary: Path
    prep: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(text: str, *, label: str) -> float:
    value_text = str(text).strip()
    if "(" in value_text:
        value_text = value_text.split("(", 1)[0]
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"PETS field {label} is not numeric: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"PETS field {label} must be finite")
    return value


def _required_pair(block: gemmi.cif.Block, tag: str) -> str:
    value = block.find_value(tag)
    if value is None or str(value).strip() in {"", ".", "?"}:
        raise ValueError(f"PETS file is missing required field {tag}")
    return str(value)


def _rotation(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * cross + (1.0 - c) * (cross @ cross)


def _cell_matrix(parameters: np.ndarray) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = np.asarray(parameters, dtype=float)
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sg = np.sin(gamma)
    volume_factor = np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cb, c * (ca - cb * cg) / sg, c * volume_factor / sg],
        ]
    )


def _busing_levy(parameters: np.ndarray) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = np.asarray(parameters, dtype=float)
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sg = np.sin(gamma)
    volume = a * b * c * np.sqrt(
        1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    )
    return np.asarray(
        [
            [1.0 / a, 0.0, 0.0],
            [-cg / (a * sg), 1.0 / (b * sg), 0.0],
            [
                b * c / volume * (cg * (ca - cb * cg) / sg - cb * sg),
                a * c / (volume * sg) * (ca - cb * cg),
                a * b * sg / volume,
            ],
        ]
    )


def _axis_position(block: gemmi.cif.Block) -> float:
    for tag in ("_diffrn_measurement_details", "_diffrn_reflns_reduction_process"):
        text = block.find_value(tag)
        if text is None:
            continue
        match = _AXIS_POSITION.search(str(text))
        if match:
            return _number(match.group(1), label="PETS rotation axis position")
    raise ValueError(
        "PETS file does not record 'rotation axis position:' or 'tilt axis position:'; "
        "the exact rocking axis cannot be assumed"
    )


def _continuous_rotation_contract(block: gemmi.cif.Block) -> dict[str, str]:
    """Fail closed unless PETS explicitly declares continuous rotation."""

    precession = str(
        block.find_value("_diffrn_measurement_method_precession") or ""
    ).strip()
    rotation_mode = str(
        block.find_value("_diffrn_measurement_rotation_mode") or ""
    ).strip()
    details = "\n".join(
        str(block.find_value(tag) or "")
        for tag in (
            "_diffrn_measurement_details",
            "_diffrn_reflns_reduction_process",
        )
    )
    precession_off = precession.lower() in {"no", "false", "0", "off"}
    continuous = rotation_mode.lower() in {
        "rotation",
        "continuous_rotation",
        "continuous rotation",
    } or "continuous rotation" in details.lower()
    if not precession_off or not continuous:
        raise ValueError(
            "PETS import currently supports explicitly declared continuous-rotation "
            "data only; precession/conical acquisition needs a different rocking path "
            f"(method_precession={precession!r}, rotation_mode={rotation_mode!r})"
        )
    return {
        "method_precession": precession,
        "rotation_mode": rotation_mode,
        "geometry": "continuous_rotation",
    }


def _operation_key(rotation: np.ndarray, translation: np.ndarray) -> tuple[float, ...]:
    values = [*np.rint(np.asarray(rotation, dtype=float)).reshape(-1).tolist()]
    reduced_translation = np.mod(np.asarray(translation, dtype=float), 1.0)
    reduced_translation[np.isclose(reduced_translation, 1.0, atol=1e-8)] = 0.0
    values.extend(np.round(reduced_translation, 8).tolist())
    return tuple(float(value) for value in values)


def _validate_structure_setting(
    structure: Path, *, space_group_number: int, pets_cell: np.ndarray
) -> dict[str, Any]:
    """Require the supplied model to use the declared PETS space-group setting."""

    from grond.legacy.engine.bloch_engine import parse_model

    path = Path(structure).expanduser().resolve()
    model = parse_model(path)
    model_operations = {
        _operation_key(operation["rot"], operation["trans"])
        for operation in model.get("symmetry_ops", [])
    }
    requested_operations = set()
    requested_group = gemmi.SpaceGroup(int(space_group_number))
    for operation in requested_group.operations():
        matrix = np.asarray(operation.float_seitz(), dtype=float)
        requested_operations.add(_operation_key(matrix[:3, :3], matrix[:3, 3]))
    if model_operations != requested_operations:
        raise ValueError(
            "supplied structure symmetry does not match --space-group-number in the "
            "same coordinate setting; resolve the PETS/model setting before import "
            f"(model operations={len(model_operations)}, requested "
            f"{requested_group.xhm()} operations={len(requested_operations)})"
        )
    model_cell = np.asarray(model["cell"], dtype=float)
    pets_cell = np.asarray(pets_cell, dtype=float)
    length_fraction = float(
        np.max(np.abs(model_cell[:3] - pets_cell[:3]) / np.maximum(pets_cell[:3], 1e-30))
    )
    angle_difference = float(np.max(np.abs(model_cell[3:] - pets_cell[3:])))
    if length_fraction > 0.03 or angle_difference > 1.0:
        raise ValueError(
            "supplied structure cell is incompatible with PETS geometry "
            f"(max length fraction {length_fraction:.4f}, max angle difference "
            f"{angle_difference:.4f} degrees)"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "space_group_number": int(space_group_number),
        "space_group_hm": requested_group.xhm(),
        "operation_count": len(requested_operations),
        "max_cell_length_fraction": length_fraction,
        "max_cell_angle_difference_deg": angle_difference,
    }


def _loop_columns(block: gemmi.cif.Block, tags: tuple[str, ...]) -> list[list[str]]:
    columns = [list(block.find_values(tag)) for tag in tags]
    lengths = {len(values) for values in columns}
    if lengths == {0}:
        raise ValueError(f"PETS file has no loop containing {tags[0]}")
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"PETS loop columns do not have a common nonzero length: {tags}")
    return columns


def _read_pets_hkl_sidecar(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith(("#", "!")):
                continue
            try:
                row = [float(value) for value in line.split()]
            except ValueError as exc:
                raise ValueError(
                    f"PETS HKL sidecar line {line_number} contains nonnumeric values"
                ) from exc
            if len(row) != 13:
                raise ValueError(
                    f"PETS HKL sidecar line {line_number} must contain 13 columns; "
                    f"found {len(row)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"PETS HKL sidecar contains no observations: {path}")
    result = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("PETS HKL sidecar contains nonfinite values")
    if not np.all(result[:, :3] == np.rint(result[:, :3])):
        raise ValueError("PETS HKL sidecar contains noninteger Miller indices")
    if not np.all(result[:, 5] == np.rint(result[:, 5])):
        raise ValueError("PETS HKL sidecar contains noninteger frame identifiers")
    return result


def _read_static_shelx_hkl(path: Path) -> list[tuple[int, int, int, float, float]]:
    """Read the row-preserving HKLF4 file exported by PETS itself."""

    rows: list[tuple[int, int, int, float, float]] = []
    terminated = False
    with path.open(errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            body = line.rstrip("\r\n")
            if len(body) < 28:
                raise ValueError(
                    f"PETS static HKLF4 row {line_number} is shorter than 28 columns"
                )
            try:
                h, k, ell = int(body[:4]), int(body[4:8]), int(body[8:12])
                intensity, sigma = float(body[12:20]), float(body[20:28])
            except ValueError as exc:
                raise ValueError(
                    f"PETS static HKLF4 row {line_number} is not fixed-width h,k,l,I,sigma"
                ) from exc
            if (h, k, ell) == (0, 0, 0):
                terminated = True
                continue
            if terminated:
                raise ValueError(
                    f"PETS static HKLF4 contains a reflection after its terminator on row {line_number}"
                )
            if not math.isfinite(intensity) or not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    f"PETS static HKLF4 row {line_number} has invalid intensity or sigma"
                )
            rows.append((h, k, ell, intensity, sigma))
    # PETS' own SHELX export commonly ends at EOF without a 0,0,0 record.
    # SHELXL accepts that producer convention; GROND's corrected writer will
    # still materialize an explicit terminator in every generated endpoint.
    return rows


def _validate_static_endpoint_hkl(
    path: Path,
    observations: list[tuple],
) -> dict[str, Any]:
    """Prove that PETS' HKLF4 endpoint is the same ordered observation table."""

    endpoint = Path(path).expanduser().resolve()
    if not endpoint.is_file():
        raise FileNotFoundError(f"PETS static HKLF4 does not exist: {endpoint}")
    rows = _read_static_shelx_hkl(endpoint)
    if len(rows) != len(observations):
        raise ValueError(
            "PETS static HKLF4 and cif_pets observation counts differ "
            f"({len(rows)} versus {len(observations)})"
        )
    max_intensity_delta = 0.0
    max_sigma_delta = 0.0
    for index, (endpoint_row, observation) in enumerate(
        zip(rows, observations, strict=True), start=1
    ):
        source_hkl = tuple(int(value) for value in observation[:3])
        if endpoint_row[:3] != source_hkl:
            raise ValueError(
                f"PETS static HKLF4 HKL differs from cif_pets at row {index}: "
                f"{endpoint_row[:3]} versus {source_hkl}"
            )
        intensity_delta = abs(float(endpoint_row[3]) - float(observation[3]))
        sigma_delta = abs(float(endpoint_row[4]) - float(observation[4]))
        max_intensity_delta = max(max_intensity_delta, intensity_delta)
        max_sigma_delta = max(max_sigma_delta, sigma_delta)
        if intensity_delta > 0.011 or sigma_delta > 0.011:
            raise ValueError(
                f"PETS static HKLF4 observation differs from cif_pets at row {index}"
            )
    return {
        "status": "validated",
        "path": str(endpoint),
        "sha256": _sha256(endpoint),
        "rows": len(rows),
        "mapping": "ordered one-to-one cif_pets observation to PETS-exported HKLF4 row",
        "hkl_match": True,
        "intensity_sigma_match": True,
        "rounding_tolerance": 0.011,
        "max_abs_intensity_delta": max_intensity_delta,
        "max_abs_sigma_delta": max_sigma_delta,
    }


def _read_jana_reflistout(path: Path) -> dict[tuple[int, int, int, int], int]:
    categories: dict[tuple[int, int, int, int], int] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(
                    f"JANA eldyn_reflistout line {line_number} must have seven fields"
                )
            try:
                key = (
                    int(fields[0]),
                    int(fields[1]),
                    int(fields[2]),
                    int(fields[5]),
                )
                category = int(fields[6])
            except ValueError as exc:
                raise ValueError(
                    f"JANA eldyn_reflistout line {line_number} has invalid HKL/zone/category"
                ) from exc
            if key in categories:
                raise ValueError(f"JANA eldyn_reflistout repeats row key {key}")
            categories[key] = category
    return categories


def read_pets_contract(
    path: Path,
    *,
    pets_hkl: Path | None = None,
    allow_virtual_frames: bool = False,
) -> dict[str, Any]:
    """Read and strictly validate one native PETS2 observation contract."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PETS cif_pets file does not exist: {source}")
    block = gemmi.cif.read_file(str(source)).sole_block()

    cell = np.asarray(
        [
            _number(_required_pair(block, tag), label=tag)
            for tag in (
                "_cell_length_a",
                "_cell_length_b",
                "_cell_length_c",
                "_cell_angle_alpha",
                "_cell_angle_beta",
                "_cell_angle_gamma",
            )
        ],
        dtype=float,
    )
    wavelength = _number(
        _required_pair(block, "_diffrn_radiation_wavelength"),
        label="_diffrn_radiation_wavelength",
    )
    if wavelength <= 0.0:
        raise ValueError("PETS wavelength must be positive")
    acquisition_geometry = _continuous_rotation_contract(block)
    ub = np.asarray(
        [
            [
                _number(
                    _required_pair(block, f"_diffrn_orient_matrix_UB_{i}{j}"),
                    label=f"_diffrn_orient_matrix_UB_{i}{j}",
                )
                for j in range(1, 4)
            ]
            for i in range(1, 4)
        ],
        dtype=float,
    )
    if abs(float(np.linalg.det(ub))) <= 1e-12:
        raise ValueError("PETS UB matrix is singular")
    axis_position_deg = _axis_position(block)
    cell_matrix = _cell_matrix(cell)
    orientation_u = ub @ np.linalg.inv(_busing_levy(cell))
    axis_correction = _rotation((0.0, 0.0, 1.0), -axis_position_deg)

    zone_tags = (
        "_diffrn_zone_axis_id",
        "_diffrn_zone_axis_u",
        "_diffrn_zone_axis_v",
        "_diffrn_zone_axis_w",
        "_diffrn_zone_axis_precession_angle",
        "_diffrn_zone_axis_alpha",
        "_diffrn_zone_axis_beta",
        "_diffrn_zone_axis_omega",
        "_diffrn_zone_axis_scale",
    )
    zone_columns = _loop_columns(block, zone_tags)
    zone_rows = list(zip(*zone_columns, strict=True))
    zone_ids = [str(row[0]) for row in zone_rows]
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError("PETS zone-axis identifiers must be unique")

    beam_unit = np.asarray([0.0, 0.0, -1.0])
    a_matrices: list[np.ndarray] = []
    zone_errors: list[float] = []
    half_widths: list[float] = []
    zone_records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(zone_rows, start=1):
        identifier = str(row[0])
        u, v, w, half_width, alpha, beta, omega, scale = (
            _number(value, label=f"{tag}[{identifier}]")
            for tag, value in zip(zone_tags[1:], row[1:], strict=True)
        )
        if half_width <= 0.0:
            raise ValueError(f"PETS zone {identifier} has nonpositive integration half-width")
        goniometer_rotation = (
            _rotation((0.0, 0.0, 1.0), omega)
            @ _rotation((1.0, 0.0, 0.0), alpha)
            @ _rotation((0.0, 1.0, 0.0), beta)
        )
        orientation = axis_correction @ goniometer_rotation @ orientation_u
        reciprocal_rows = np.linalg.pinv(cell_matrix @ orientation.T).T
        frame_a = reciprocal_rows.T
        recorded = np.asarray([u, v, w], dtype=float)
        predicted = -(frame_a.T @ beam_unit)
        predicted /= max(float(np.max(np.abs(predicted))), 1e-30)
        error = float(np.linalg.norm(predicted - recorded))
        if error > 2.5e-3:
            raise ValueError(
                f"PETS zone {identifier} is inconsistent with Rz(omega) Rx(alpha) "
                f"Ry(beta) UB geometry (normalized zone-axis error {error:.4g})"
            )
        a_matrices.append(frame_a)
        zone_errors.append(error)
        half_widths.append(float(half_width))
        zone_records.append(
            {
                "source_id": identifier,
                "ordinal": ordinal,
                "alpha_deg": alpha,
                "beta_deg": beta,
                "omega_deg": omega,
                "integration_half_width_deg": half_width,
                "source_scale": scale,
                "zone_axis_error": error,
            }
        )

    reflection_tags = (
        "_refln_index_h",
        "_refln_index_k",
        "_refln_index_l",
        "_refln_intensity_meas",
        "_refln_intensity_sigma",
        "_refln_zone_axis_id",
    )
    reflection_columns = _loop_columns(block, reflection_tags)
    reflection_rows = list(zip(*reflection_columns, strict=True))
    sidecar: Path | None = None
    sidecar_rows: np.ndarray | None = None
    if not allow_virtual_frames:
        sidecar = (
            Path(pets_hkl).expanduser().resolve()
            if pets_hkl is not None
            else source.with_suffix(".hkl")
        )
        if not sidecar.is_file():
            raise ValueError(
                "PETS cif_pets does not contain the row's acquisition/Bragg frame. "
                f"The companion 13-column PETS .hkl file is required but missing: {sidecar}"
            )
        sidecar_rows = _read_pets_hkl_sidecar(sidecar)
        if len(sidecar_rows) != len(reflection_rows):
            raise ValueError(
                "PETS cif_pets and .hkl sidecar row counts differ; virtual-frame or "
                f"nonmatching inputs are not allowed ({len(reflection_rows)} versus "
                f"{len(sidecar_rows)})"
            )
    zone_to_ordinal = {record["source_id"]: record["ordinal"] for record in zone_records}
    observations: list[tuple[int, int, int, float, float, str, int]] = []
    occurrence_keys: set[tuple[int, int, int, str]] = set()
    hkl_zones: dict[tuple[int, int, int], set[str]] = {}
    paired_sidecar: list[np.ndarray | None] = (
        [None] * len(reflection_rows)
        if sidecar_rows is None
        else [row for row in sidecar_rows]
    )
    for index, (row, sidecar_row) in enumerate(
        zip(reflection_rows, paired_sidecar, strict=True)
    ):
        try:
            h, k, ell = (int(str(value)) for value in row[:3])
        except ValueError as exc:
            raise ValueError(f"PETS reflection row {index + 1} has noninteger HKL") from exc
        intensity = _number(row[3], label=f"intensity row {index + 1}")
        sigma = _number(row[4], label=f"sigma row {index + 1}")
        if sigma <= 0.0:
            raise ValueError(f"PETS reflection row {index + 1} has nonpositive sigma")
        cif_zone_id = str(row[5])
        if sidecar_row is None:
            frame_id = cif_zone_id
            incident = [float("nan")] * 3
            diffracted = [float("nan")] * 3
        else:
            sidecar_hkl = tuple(int(value) for value in sidecar_row[:3])
            if sidecar_hkl != (h, k, ell):
                raise ValueError(
                    f"PETS cif_pets and .hkl HKLs differ at row {index + 1}: "
                    f"{(h, k, ell)} versus {sidecar_hkl}"
                )
            if not math.isclose(
                float(sidecar_row[3]), intensity, rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ValueError(
                    f"PETS cif_pets and .hkl intensities differ at row {index + 1}"
                )
            if not math.isclose(
                float(sidecar_row[4]), sigma, rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ValueError(
                    f"PETS cif_pets and .hkl sigmas differ at row {index + 1}"
                )
            frame_id = str(int(sidecar_row[5]))
            incident = [float(value) for value in sidecar_row[7:10]]
            diffracted = [float(value) for value in sidecar_row[10:13]]
        if frame_id not in zone_to_ordinal:
            raise ValueError(
                f"PETS reflection row {index + 1} uses .hkl frame {frame_id}, which "
                "is missing from the cif_pets zone-axis table"
            )
        key = (h, k, ell, frame_id)
        if key in occurrence_keys:
            raise ValueError(f"PETS file repeats identical HKL/zone occurrence {key}")
        occurrence_keys.add(key)
        hkl_zones.setdefault((h, k, ell), set()).add(frame_id)
        observations.append(
            (
                h,
                k,
                ell,
                intensity,
                sigma,
                frame_id,
                zone_to_ordinal[frame_id],
                cif_zone_id,
                incident,
                diffracted,
            )
        )

    repeated_hkls = {hkl: zones for hkl, zones in hkl_zones.items() if len(zones) > 1}
    if repeated_hkls and not allow_virtual_frames:
        maximum = max(len(zones) for zones in repeated_hkls.values())
        raise ValueError(
            "PETS file repeats signed HKLs across zone axes; this is a virtual-frame "
            f"materialization, not a unique integrated-row contract ({len(repeated_hkls)} "
            f"repeated HKLs; maximum multiplicity {maximum}). Use the ordinary cif_pets "
            "integration file, not *_dyn.cif_pets."
        )

    orientation_steps = []
    for first, second in zip(a_matrices[:-1], a_matrices[1:], strict=True):
        relative = second @ np.linalg.inv(first)
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        orientation_steps.append(math.degrees(math.acos(cosine)))
    orientation_step = float(np.median(orientation_steps)) if orientation_steps else 1.0
    if not math.isfinite(orientation_step) or orientation_step <= 0.0:
        raise ValueError("PETS zone-axis orientation step is not finite and positive")

    real_matrix = np.linalg.inv(ub).T
    return {
        "source": source,
        "pets_hkl": sidecar,
        "cell": cell,
        "axis_position_deg": axis_position_deg,
        "acquisition_geometry": acquisition_geometry,
        "wavelength": wavelength,
        "ub": ub,
        "real_matrix": real_matrix,
        "a_matrices": np.asarray(a_matrices, dtype=float),
        "zones": zone_records,
        "observations": observations,
        "orientation_step_deg": orientation_step,
        "orientation_step_range_deg": [
            float(np.min(orientation_steps)) if orientation_steps else orientation_step,
            float(np.max(orientation_steps)) if orientation_steps else orientation_step,
        ],
        "orientation_step_source": (
            "median_successive_zone_rotation"
            if orientation_steps
            else "single_zone_placeholder_not_used_by_exact_geometry"
        ),
        "virtual_frames": bool(allow_virtual_frames),
        "repeated_signed_hkls": len(repeated_hkls),
        "maximum_virtual_multiplicity": (
            max((len(zones) for zones in repeated_hkls.values()), default=1)
        ),
        "zone_axis_error_max": float(max(zone_errors)),
        "zone_axis_error_rms": float(np.sqrt(np.mean(np.square(zone_errors)))),
    }


def import_pets_to_grond(
    cif_pets: Path,
    out_dir: Path,
    *,
    pets_hkl: Path | None = None,
    allow_virtual_frames: bool = False,
    klar_rsg: float | None = None,
    klar_dsg: float = 0.0,
    exclude_systematic_absences: bool = False,
    jana_reflistout: Path | None = None,
    dataset_name: str = "pets2",
    space_group_number: int,
    hkl_transform: str = "identity",
    structure: Path | None = None,
    baseline_hkl: Path | None = None,
    prep_out: Path | None = None,
) -> PetsImportOutput:
    """Write a PETS-owned, row-preserving GROND bridge directory."""

    if klar_rsg is not None and (
        not math.isfinite(float(klar_rsg)) or float(klar_rsg) <= 0.0
    ):
        raise ValueError("--klar-rsg must be finite and positive when supplied")
    if not math.isfinite(float(klar_dsg)):
        raise ValueError("--klar-dsg must be finite")
    contract = read_pets_contract(
        cif_pets,
        pets_hkl=pets_hkl,
        allow_virtual_frames=allow_virtual_frames,
    )
    transform = parse_hkl_transform_info(hkl_transform)
    if transform.canonical != "identity":
        raise ValueError(
            "native PETS import currently requires hkl_transform=identity; a nonidentity "
            "setting needs symmetry conjugation that is not yet implemented. Hand remains "
            "a separate negative-thickness coordinate comparison."
        )
    structure_validation = (
        _validate_structure_setting(
            Path(structure),
            space_group_number=int(space_group_number),
            pets_cell=np.asarray(contract["cell"], dtype=float),
        )
        if structure is not None
        else None
    )
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_observations = list(contract["observations"])
    static_endpoint_contract: dict[str, Any] = {
        "status": "unavailable",
        "reason": "no baseline HKLF4 supplied",
    }
    if baseline_hkl is not None:
        if bool(contract["virtual_frames"]):
            static_endpoint_contract = {
                "status": "unavailable",
                "reason": "PETS virtual-frame occurrences are not a static HKLF4 observation table",
            }
        elif klar_rsg is not None or bool(exclude_systematic_absences):
            static_endpoint_contract = {
                "status": "unavailable",
                "reason": "row filtering changed the ordinary PETS observation support",
            }
        else:
            static_endpoint_contract = _validate_static_endpoint_hkl(
                Path(baseline_hkl), all_observations
            )
    operations = gemmi.SpaceGroup(int(space_group_number)).operations()
    k_magnitude = 1.0 / float(contract["wavelength"])
    s0 = np.asarray([0.0, 0.0, -k_magnitude])
    selected_observations = []
    selection_ledger = []
    klar_kept = 0
    symmetry_kept = 0
    jana_categories = (
        _read_jana_reflistout(Path(jana_reflistout).expanduser().resolve())
        if jana_reflistout is not None
        else {}
    )
    for source_index, row in enumerate(all_observations):
        hkl = np.asarray(row[:3], dtype=int)
        ordinal = int(row[6])
        source_zone_id = str(row[5])
        try:
            jana_zone_id = int(source_zone_id)
        except ValueError:
            if jana_categories:
                raise ValueError(
                    "--jana-reflistout requires numeric PETS source zone IDs; "
                    f"found {source_zone_id!r}"
                ) from None
            jana_zone_id = ordinal
        matrix = np.asarray(contract["a_matrices"][ordinal - 1], dtype=float)
        g_vector = matrix @ hkl
        sg = abs(
            (
                k_magnitude**2
                - float(np.sum(np.square(s0 + g_vector)))
            )
            / (2.0 * k_magnitude)
        )
        semiangle = float(contract["zones"][ordinal - 1]["integration_half_width_deg"])
        sg_max = float(np.linalg.norm(g_vector[1:]) * math.radians(semiangle))
        klar_ok = True
        if klar_rsg is not None:
            klar_ok = bool(
                sg_max > 0.0
                and sg / sg_max < float(klar_rsg)
                and sg_max - sg > float(klar_dsg)
            )
        symmetry_ok = not operations.is_systematically_absent(tuple(int(v) for v in hkl))
        if klar_ok:
            klar_kept += 1
        if symmetry_ok:
            symmetry_kept += 1
        keep = klar_ok and (symmetry_ok or not exclude_systematic_absences)
        if keep:
            selected_observations.append(row)
        selection_ledger.append(
            {
                "source_row_index": source_index,
                "h": int(hkl[0]),
                "k": int(hkl[1]),
                "l": int(hkl[2]),
                "zone_ordinal": ordinal,
                "sg_center_inv_A": sg,
                "sg_max_inv_A": sg_max,
                "klar_keep": klar_ok,
                "systematic_absence": not symmetry_ok,
                "jana_category": jana_categories.get(
                    (int(hkl[0]), int(hkl[1]), int(hkl[2]), jana_zone_id)
                ),
                "selected": keep,
            }
        )
    contract["observations"] = selected_observations
    if not selected_observations:
        raise ValueError("PETS row selection retained no observations")
    selected_d_values = [
        gemmi.UnitCell(*[float(value) for value in contract["cell"]]).calculate_d(
            (int(row[0]), int(row[1]), int(row[2]))
        )
        for row in selected_observations
    ]
    selected_d_values = [
        float(value)
        for value in selected_d_values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not selected_d_values:
        raise ValueError("PETS row selection has no finite positive d spacings")
    selected_d_min = float(min(selected_d_values))
    # The engine applies an inclusive d>=cutoff test. Move by one floating-point
    # representable value toward zero so a row exactly at the PETS edge survives
    # serialization/reload without weakening the declared support.
    engine_d_min = float(np.nextafter(selected_d_min, 0.0))

    scan_path = destination / "PETS_A_AT_ZONE_AXES.txt"
    with scan_path.open("w") as stream:
        for matrix in contract["a_matrices"]:
            stream.write(" ".join(f"{float(value): .12e}" for value in matrix.reshape(-1)) + "\n")

    geometry_path = destination / "PETS_EXACT_GEOMETRY.npz"
    np.savez_compressed(
        geometry_path,
        a_center=np.asarray(contract["a_matrices"], dtype=float),
        integration_semiangle_deg=np.asarray(
            [row["integration_half_width_deg"] for row in contract["zones"]],
            dtype=float,
        ),
        source_zone_id=np.arange(1, len(contract["zones"]) + 1, dtype=np.int64),
        source_zone_id_text=np.asarray(
            [str(row["source_id"]) for row in contract["zones"]], dtype="U64"
        ),
        rotation_axis_position_deg=np.asarray(contract["axis_position_deg"], dtype=float),
    )
    selection_path = destination / "PETS_ROW_SELECTION.json"
    selection_path.write_text(json.dumps(selection_ledger, indent=2) + "\n")

    cell = contract["cell"]
    real = contract["real_matrix"]
    step = float(contract["orientation_step_deg"])
    zone_by_ordinal = {int(row["ordinal"]): row for row in contract["zones"]}
    hkl_path = destination / "INTEGRATE.HKL"
    with hkl_path.open("w") as stream:
        stream.write("!OUTPUT_FILE=INTEGRATE.HKL\n")
        stream.write("!GENERATED_BY=GROND import-pets EXPERIMENTAL\n")
        stream.write("!GROND_SOURCE=PETS2\n")
        stream.write(f"!PETS_CIF= {contract['source']}\n")
        stream.write(f"!PETS_CIF_SHA256= {_sha256(contract['source'])}\n")
        stream.write(f"!PETS_EXACT_GEOMETRY_FILE= {geometry_path.name}\n")
        if contract["pets_hkl"] is not None:
            stream.write(f"!PETS_HKL= {contract['pets_hkl']}\n")
            stream.write(f"!PETS_HKL_SHA256= {_sha256(contract['pets_hkl'])}\n")
        stream.write(
            "!PETS_OBSERVATION_CONTRACT= "
            + (
                "PETS_owned_virtual_frame_occurrences\n"
                if contract["virtual_frames"]
                else "unique_signed_hkl_one_measured_frame\n"
            )
        )
        stream.write(
            f"!PETS_VIRTUAL_FRAMES= {'TRUE' if contract['virtual_frames'] else 'FALSE'}\n"
        )
        stream.write(
            "!PETS_STATIC_HKLF4_VALIDATED= "
            f"{'TRUE' if static_endpoint_contract.get('status') == 'validated' else 'FALSE'}\n"
        )
        if static_endpoint_contract.get("status") == "validated":
            stream.write(
                f"!PETS_STATIC_HKLF4= {static_endpoint_contract['path']}\n"
            )
            stream.write(
                f"!PETS_STATIC_HKLF4_SHA256= {static_endpoint_contract['sha256']}\n"
            )
        stream.write("!PETS_INTENSITY_SOURCE= _refln_intensity_meas\n")
        stream.write("!PETS_SIGMA_SOURCE= _refln_intensity_sigma\n")
        stream.write("!PETS_PARTIALITY= unavailable; PEAK=100 is an inclusion sentinel only\n")
        stream.write(f"!X-RAY_WAVELENGTH= {contract['wavelength']:.12g}\n")
        stream.write(f"!SPACE_GROUP_NUMBER= {int(space_group_number)}\n")
        stream.write("!STARTING_FRAME= 1\n")
        stream.write("!STARTING_ANGLE= 0\n")
        stream.write(f"!OSCILLATION_RANGE= {step:.12g}\n")
        stream.write("!ROTATION_AXIS= 0 0 -1\n")
        stream.write("!GROND_ROTATION_AXIS_DATUM= 0 0 -1\n")
        stream.write("!INCIDENT_BEAM_DIRECTION= 0 0 -1\n")
        stream.write("!GROND_FIXED_ROTATION= 1 0 0 0 1 0 0 0 1\n")
        stream.write("!GROND_SETTING_ROTATION= 1 0 0 0 1 0 0 0 1\n")
        stream.write(f"!UNIT_CELL_CONSTANTS= {' '.join(f'{v:.10g}' for v in cell)}\n")
        for label, vector in zip("ABC", real.T, strict=True):
            stream.write(f"!UNIT_CELL_{label}-AXIS= {' '.join(f'{v:.12g}' for v in vector)}\n")
        stream.write(f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_path.name}\n")
        stream.write("!COLUMN_CONTRACT= H K L IOBS SIGMA XCAL YCAL ZCAL RLP PEAK CORR MAXC XOBS YOBS ZOBS ALF0 BET0 ALF1 BET1 PSI ISEG\n")
        stream.write("!END_OF_HEADER\n")
        for (
            h,
            k,
            ell,
            intensity,
            sigma,
            _frame_id,
            ordinal,
            _cif_zone_id,
            _incident,
            _diffracted,
        ) in contract["observations"]:
            half_width_deg = float(
                zone_by_ordinal[int(ordinal)]["integration_half_width_deg"]
            )
            half_width_frames = half_width_deg / step
            stream.write(
                f"{h:5d} {k:5d} {ell:5d} {intensity: .8e} {sigma: .8e} "
                f"nan nan {float(ordinal):.5f} 1.00000000e+00 100.00000 100.00 0.0 "
                f"nan nan nan {float(ordinal) - half_width_frames:.5f} "
                f"{float(ordinal) + half_width_frames:.5f} nan nan 0.00000 0\n"
            )

    zone_ledger = destination / "PETS_ZONE_AXIS_LEDGER.json"
    zone_ledger.write_text(json.dumps(contract["zones"], indent=2) + "\n")
    summary_payload = {
        "schema": "grond.pets2_import.experimental.v1",
        "status": "ok",
        "source": str(contract["source"]),
        "source_sha256": _sha256(contract["source"]),
        "pets_hkl": str(contract["pets_hkl"]) if contract["pets_hkl"] else None,
        "pets_hkl_sha256": (
            _sha256(contract["pets_hkl"]) if contract["pets_hkl"] else None
        ),
        "output_hkl": str(hkl_path),
        "scan_points": str(scan_path),
        "exact_geometry": str(geometry_path),
        "row_selection_ledger": str(selection_path),
        "zone_axis_ledger": str(zone_ledger),
        "dataset_name": dataset_name,
        "structure_validation": structure_validation,
        "space_group_number": int(space_group_number),
        "hkl_transform": transform.to_dict(),
        "rows_input": len(all_observations),
        "rows": len(contract["observations"]),
        "selected_min_d_spacing_A": selected_d_min,
        "engine_min_d_spacing_A": engine_d_min,
        "zone_axes": len(contract["zones"]),
        "unique_signed_hkls": len(
            {(row[0], row[1], row[2]) for row in contract["observations"]}
        ),
        "virtual_frames": contract["virtual_frames"],
        "repeated_signed_hkls": contract["repeated_signed_hkls"],
        "maximum_virtual_multiplicity": contract["maximum_virtual_multiplicity"],
        "row_selection": {
            "klar_rsg": klar_rsg,
            "klar_dsg": klar_dsg if klar_rsg is not None else None,
            "klar_kept": klar_kept,
            "space_group_number": int(space_group_number),
            "full_systematic_absence_filter": bool(exclude_systematic_absences),
            "symmetry_allowed": symmetry_kept,
            "selected": len(selected_observations),
            "selected_systematic_absences": int(
                sum(
                    bool(row["selected"]) and bool(row["systematic_absence"])
                    for row in selection_ledger
                )
            ),
            "jana_reflistout": (
                str(Path(jana_reflistout).expanduser().resolve())
                if jana_reflistout is not None
                else None
            ),
            "jana_category_counts": {
                str(category): int(sum(value == category for value in jana_categories.values()))
                for category in sorted(set(jana_categories.values()))
            },
            "klar_nonabsence_intersection_with_jana_category_5": int(
                sum(
                    bool(row["klar_keep"])
                    and not bool(row["systematic_absence"])
                    and row["jana_category"] == 5
                    for row in selection_ledger
                )
            ),
        },
        "wavelength_A": float(contract["wavelength"]),
        "cell": [float(value) for value in contract["cell"]],
        "orientation": {
            "formula": (
                "U=UB B(cell)^-1; O=Rz(-axis_position) Rz(omega) "
                "Rx(alpha) Ry(beta) U; A_zone=orientation_basis(cell,O)^T"
            ),
            "incident_beam": [0.0, 0.0, -1.0],
            "zone_step_deg_from_median_successive_orientation_rotation": step,
            "zone_step_range_deg": contract["orientation_step_range_deg"],
            "zone_step_source": contract["orientation_step_source"],
            "zone_axis_normalized_error_max": contract["zone_axis_error_max"],
            "zone_axis_normalized_error_rms": contract["zone_axis_error_rms"],
            "acquisition_geometry": contract["acquisition_geometry"],
        },
        "intensity_contract": {
            "intensity": "PETS _refln_intensity_meas unchanged",
            "sigma": "PETS _refln_intensity_sigma unchanged",
            "row_orientation": (
                "cif_pets virtual zone axis owned by PETS"
                if contract["virtual_frames"]
                else "companion PETS .hkl column 6 acquisition frame; cif_pets _refln_zone_axis_id retained only as provenance"
            ),
            "partiality": "not provided by cif_pets; no partiality filter applied",
            "virtual_frames": contract["virtual_frames"],
            "detector_coordinates": "not provided; detector scale fields must remain disabled",
        },
        "static_endpoint_contract": static_endpoint_contract,
        "recommended_run_contract": {
            "hkl_transform": "identity; nonidentity PETS/model setting conjugation is not yet implemented",
            "min_peak": 0.0,
            "detector_scale_bins": "0",
            "profile_geometry": "PETS_EXACT_GEOMETRY.npz direct-angle rocking path",
            "profile_accumulation": "sum",
            "beam_and_quadrature": "must be converged for the structure and PETS settings",
            "shelxl_endpoint": (
                "validated row-preserving PETS HKLF4; q correction and SHELXL feedback are available"
                if static_endpoint_contract.get("status") == "validated"
                else "not available without a row-preserving PETS static HKLF4 contract"
            ),
        },
    }
    summary_path = destination / "pets2_bridge_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    prep_path: Path | None = None
    if structure is not None:
        from .config import DatasetSpec, PrepConfig, write_prep

        structure_path = Path(structure).expanduser().resolve()
        prep_path = (
            Path(prep_out).expanduser().resolve()
            if prep_out is not None
            else destination / "grond_prep.json"
        )
        prep = PrepConfig(
            structure=str(structure_path),
            baseline_hkl=(
                str(Path(baseline_hkl).expanduser().resolve()) if baseline_hkl else ""
            ),
            notes=(
                "Experimental native PETS2 row-space import; PETS-owned correlated virtual-frame occurrences."
                if contract["virtual_frames"]
                else "Experimental native PETS2 row-space import; one measured acquisition frame per signed row."
            ),
            datasets=[
                DatasetSpec(
                    name=dataset_name,
                    hkl=str(hkl_path),
                    xds_dir=str(destination),
                    hkl_transform=transform.canonical,
                    min_d_spacing=engine_d_min,
                    min_peak=0.0,
                    target_mode="iobs",
                    include_nonpositive=True,
                    geometry_percentile=100.0,
                )
            ],
            metadata={"pets2_bridge": summary_payload},
        )
        write_prep(prep_path, prep)

    return PetsImportOutput(destination, hkl_path, scan_path, summary_path, prep_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-pets", type=Path, required=True)
    parser.add_argument("--pets-hkl", type=Path)
    parser.add_argument(
        "--allow-virtual-frames",
        action="store_true",
        help=(
            "Accept PETS-owned virtual-frame occurrences from *_dyn.cif_pets. "
            "These are correlated occurrences, not independent source observations."
        ),
    )
    parser.add_argument("--klar-rsg", type=float)
    parser.add_argument("--klar-dsg", type=float, default=0.0)
    parser.add_argument("--exclude-systematic-absences", action="store_true")
    parser.add_argument("--jana-reflistout", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="pets2")
    parser.add_argument("--space-group-number", type=int, required=True)
    parser.add_argument(
        "--hkl-transform",
        default="identity",
        help="Reserved for future setting conversion; the current importer requires identity.",
    )
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--baseline-hkl", type=Path)
    parser.add_argument("--prep-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = import_pets_to_grond(
        args.cif_pets,
        args.out_dir,
        pets_hkl=args.pets_hkl,
        allow_virtual_frames=args.allow_virtual_frames,
        klar_rsg=args.klar_rsg,
        klar_dsg=args.klar_dsg,
        exclude_systematic_absences=args.exclude_systematic_absences,
        jana_reflistout=args.jana_reflistout,
        dataset_name=args.dataset_name,
        space_group_number=args.space_group_number,
        hkl_transform=args.hkl_transform,
        structure=args.structure,
        baseline_hkl=args.baseline_hkl,
        prep_out=args.prep_out,
    )
    print(json.dumps({key: str(value) if value is not None else None for key, value in asdict(output).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
