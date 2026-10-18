"""Read PETS observations and verify their model, row and static-endpoint contracts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from grond.io.pets_geometry import (
    _axis_position,
    _busing_levy,
    _cell_matrix,
    _continuous_rotation_contract,
    _number,
    _required_pair,
    _rotation,
    _validate_cell_geometry,
    _validate_ub_metric,
)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _operation_key(rotation: np.ndarray, translation: np.ndarray) -> tuple[float, ...]:
    """Return a comparable key for a symmetry operation: rounded rotation and translation reduced to [0, 1)."""

    values = [*np.rint(np.asarray(rotation, dtype=float)).reshape(-1).tolist()]
    reduced_translation = np.mod(np.asarray(translation, dtype=float), 1.0)
    reduced_translation[np.isclose(reduced_translation, 1.0, atol=1e-8)] = 0.0
    values.extend(np.round(reduced_translation, 8).tolist())
    return tuple(float(value) for value in values)


def _validate_structure_setting(structure: Path, *, space_group_number: int, pets_cell: np.ndarray) -> dict[str, Any]:
    """Require the supplied model to use the declared PETS space-group setting."""

    from grond.engine.bloch_engine import parse_model

    path = Path(structure).expanduser().resolve()
    model = parse_model(path)
    model_operations = {
        _operation_key(operation["rot"], operation["trans"]) for operation in model.get("symmetry_ops", [])
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
    length_fraction = float(np.max(np.abs(model_cell[:3] - pets_cell[:3]) / np.maximum(pets_cell[:3], 1e-30)))
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
    """Return the values of the given CIF loop tags, one list per tag; raise ValueError unless
    all have one nonzero length.
    """

    columns = [list(block.find_values(tag)) for tag in tags]
    lengths = {len(values) for values in columns}
    if lengths == {0}:
        raise ValueError(f"PETS file has no loop containing {tags[0]}")
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"PETS loop columns do not have a common nonzero length: {tags}")
    return columns


def _read_pets_hkl_sidecar(path: Path) -> np.ndarray:
    """Read the 13-column PETS ``.hkl`` file that accompanies a ``cif_pets`` file.

    Columns 1-3 are h, k, l, 4-5 intensity and sigma, 6 the frame number, 8-10 the incident
    and 11-13 the diffracted beam directions. Miller indices and frame numbers must be
    integers.
    """

    rows: list[list[float]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith(("#", "!")):
                continue
            try:
                row = [float(value) for value in line.split()]
            except ValueError as exc:
                raise ValueError(f"PETS HKL sidecar line {line_number} contains nonnumeric values") from exc
            if len(row) != 13:
                raise ValueError(f"PETS HKL sidecar line {line_number} must contain 13 columns; found {len(row)}")
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
                raise ValueError(f"PETS static HKLF4 row {line_number} is shorter than 28 columns")
            try:
                h, k, ell = int(body[:4]), int(body[4:8]), int(body[8:12])
                intensity, sigma = float(body[12:20]), float(body[20:28])
            except ValueError as exc:
                raise ValueError(f"PETS static HKLF4 row {line_number} is not fixed-width h,k,l,I,sigma") from exc
            if (h, k, ell) == (0, 0, 0):
                terminated = True
                continue
            if terminated:
                raise ValueError(f"PETS static HKLF4 contains a reflection after its terminator on row {line_number}")
            if not math.isfinite(intensity) or not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"PETS static HKLF4 row {line_number} has invalid intensity or sigma")
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
            f"PETS static HKLF4 and cif_pets observation counts differ ({len(rows)} versus {len(observations)})"
        )
    max_intensity_delta = 0.0
    max_sigma_delta = 0.0
    for index, (endpoint_row, observation) in enumerate(zip(rows, observations, strict=True), start=1):
        source_hkl = tuple(int(value) for value in observation[:3])
        if endpoint_row[:3] != source_hkl:
            raise ValueError(
                f"PETS static HKLF4 HKL differs from cif_pets at row {index}: {endpoint_row[:3]} versus {source_hkl}"
            )
        intensity_delta = abs(float(endpoint_row[3]) - float(observation[3]))
        sigma_delta = abs(float(endpoint_row[4]) - float(observation[4]))
        max_intensity_delta = max(max_intensity_delta, intensity_delta)
        max_sigma_delta = max(max_sigma_delta, sigma_delta)
        if intensity_delta > 0.011 or sigma_delta > 0.011:
            raise ValueError(f"PETS static HKLF4 observation differs from cif_pets at row {index}")
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
    """Read a Jana2020 ``eldyn_reflistout`` file as {(h, k, l, zone): category}."""

    categories: dict[tuple[int, int, int, int], int] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(f"JANA eldyn_reflistout line {line_number} must have seven fields")
            try:
                key = (
                    int(fields[0]),
                    int(fields[1]),
                    int(fields[2]),
                    int(fields[5]),
                )
                category = int(fields[6])
            except ValueError as exc:
                raise ValueError(f"JANA eldyn_reflistout line {line_number} has invalid HKL/zone/category") from exc
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
    _validate_cell_geometry(cell)
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
    _validate_ub_metric(ub, cell)
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
    zone_records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(zone_rows, start=1):
        identifier = str(row[0])
        u, v, w, half_width, alpha, beta, omega, scale = (
            _number(value, label=f"{tag}[{identifier}]") for tag, value in zip(zone_tags[1:], row[1:], strict=True)
        )
        if half_width <= 0.0:
            raise ValueError(f"PETS zone {identifier} has nonpositive integration half-width")
        goniometer_rotation = (
            _rotation((0.0, 0.0, 1.0), omega) @ _rotation((1.0, 0.0, 0.0), alpha) @ _rotation((0.0, 1.0, 0.0), beta)
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
        sidecar = Path(pets_hkl).expanduser().resolve() if pets_hkl is not None else source.with_suffix(".hkl")
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
        [None] * len(reflection_rows) if sidecar_rows is None else [row for row in sidecar_rows]
    )
    for index, (row, sidecar_row) in enumerate(zip(reflection_rows, paired_sidecar, strict=True)):
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
                    f"PETS cif_pets and .hkl HKLs differ at row {index + 1}: {(h, k, ell)} versus {sidecar_hkl}"
                )
            if not math.isclose(float(sidecar_row[3]), intensity, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"PETS cif_pets and .hkl intensities differ at row {index + 1}")
            if not math.isclose(float(sidecar_row[4]), sigma, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"PETS cif_pets and .hkl sigmas differ at row {index + 1}")
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
        "maximum_virtual_multiplicity": (max((len(zones) for zones in repeated_hkls.values()), default=1)),
        "zone_axis_error_max": float(max(zone_errors)),
        "zone_axis_error_rms": float(np.sqrt(np.mean(np.square(zone_errors)))),
    }
