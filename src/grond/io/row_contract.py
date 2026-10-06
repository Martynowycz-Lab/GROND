"""Portable GROND row tables: a CSV of signed observations plus a JSON geometry sidecar.

Each row needs h, k, l, Iobs and sigma_Iobs, and a frame or rotation angle. The sidecar
(schema ``grond-row-v1``) declares the wavelength, beam, rotation axis, scan origin, cell
and orientation matrix. A validated table can be written as an XDS-like INTEGRATE.HKL
directory for the engine.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

GROND_ROW_SCHEMA_VERSION = "grond-row-v1"

REQUIRED_ROW_FIELDS = ("h", "k", "l", "Iobs", "sigma_Iobs")
OPTIONAL_ROW_FIELDS = (
    "row_id",
    "dataset_id",
    "frame",
    "alpha_deg",
    "x_px",
    "y_px",
    "frame_obs",
    "partiality",
    "lp",
    "profile_correlation",
    "rlp",
    "peak",
    "corr",
    "bbox_z0",
    "bbox_z1",
    "panel",
    "source",
)


@dataclass(frozen=True)
class RowContractSummary:
    """Short validation summary for a portable GROND row table."""

    schema_version: str
    rows: int
    finite_rows: int
    positive_sigma_rows: int
    has_frame: bool
    has_alpha_deg: bool
    dataset_ids: list[str]
    warnings: list[str]


def _as_float(value: Any, *, field: str, row_number: int) -> float:
    """Parse a row field as a float (blank gives NaN); raise ValueError naming the row and field otherwise."""

    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: field {field!r} is not numeric: {value!r}") from exc


def _as_int(value: Any, *, field: str, row_number: int) -> int:
    """Parse a required row field as an integer Miller index; raise ValueError naming the row and field otherwise."""

    text = str(value).strip()
    if text == "":
        raise ValueError(f"row {row_number}: required integer field {field!r} is blank")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: field {field!r} is not an integer: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"row {row_number}: field {field!r} must be an integer Miller index: {value!r}")
    return int(number)


def _normalized_row(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    """Return one row with canonical field names and typed numeric values."""

    out = dict(row)
    for field in REQUIRED_ROW_FIELDS:
        if field not in out or str(out[field]).strip() == "":
            raise ValueError(f"row {row_number}: missing required field {field!r}")
    out["h"] = _as_int(out["h"], field="h", row_number=row_number)
    out["k"] = _as_int(out["k"], field="k", row_number=row_number)
    out["l"] = _as_int(out["l"], field="l", row_number=row_number)
    out["Iobs"] = _as_float(out["Iobs"], field="Iobs", row_number=row_number)
    out["sigma_Iobs"] = _as_float(out["sigma_Iobs"], field="sigma_Iobs", row_number=row_number)
    for field in OPTIONAL_ROW_FIELDS:
        if field in out and field not in {"row_id", "dataset_id", "source"}:
            out[field] = _as_float(out[field], field=field, row_number=row_number)
    if not str(out.get("row_id", "")).strip():
        out["row_id"] = row_number - 1
    return out


def read_row_table(path: str | Path) -> list[dict[str, Any]]:
    """Read a portable GROND row table from CSV, TSV, or JSON Lines."""

    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(source.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: JSONL rows must be objects")
            rows.append(_normalized_row(payload, len(rows) + 1))
    else:
        delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError(f"{source} has no header row")
            for row in reader:
                rows.append(_normalized_row(dict(row), len(rows) + 1))
    if not rows:
        raise ValueError(f"no observation rows found in {source}")
    return rows


def load_row_metadata(path: str | Path) -> dict[str, Any]:
    """Load and lightly normalize a portable row-contract metadata sidecar."""

    metadata = json.loads(Path(path).expanduser().read_text())
    if not isinstance(metadata, dict):
        raise ValueError("row metadata must be a JSON object")
    return metadata


def _matrix_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    """Return the reciprocal orientation matrix A (1/A) from the geometry sidecar.

    Uses ``reciprocal_A_matrix`` when given, else the inverse transpose of
    ``real_space_axes_A`` (real axes as columns).
    """

    geometry = metadata.get("geometry") or {}
    if "reciprocal_A_matrix" in geometry:
        mat = np.asarray(geometry["reciprocal_A_matrix"], dtype=float)
        if mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
            raise ValueError("geometry.reciprocal_A_matrix must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(mat))) <= 1e-15:
            raise ValueError("geometry.reciprocal_A_matrix must be nonsingular")
        return mat
    if "real_space_axes_A" in geometry:
        real = np.asarray(geometry["real_space_axes_A"], dtype=float)
        if real.shape != (3, 3) or not np.all(np.isfinite(real)):
            raise ValueError("geometry.real_space_axes_A must be a finite 3x3 matrix with axes as columns")
        if abs(float(np.linalg.det(real))) <= 1e-15:
            raise ValueError("geometry.real_space_axes_A must be nonsingular")
        return np.linalg.inv(real).T
    raise ValueError("row metadata must include geometry.reciprocal_A_matrix or geometry.real_space_axes_A")


def _real_space_axes_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    """Return the real-space axes (A, as columns) from the sidecar, or derive them from A."""

    geometry = metadata.get("geometry") or {}
    if "real_space_axes_A" in geometry:
        real = np.asarray(geometry["real_space_axes_A"], dtype=float)
        if real.shape != (3, 3):
            raise ValueError("geometry.real_space_axes_A must be 3x3 with axes as columns")
        return real
    return np.linalg.inv(_matrix_from_metadata(metadata).T)


def _rotation_matrix_from_metadata(geometry: dict[str, Any], key: str) -> np.ndarray:
    """Return an optional proper rotation matrix from portable geometry metadata."""

    if key not in geometry:
        return np.eye(3, dtype=float)
    matrix = np.asarray(geometry[key], dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"geometry.{key} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"geometry.{key} must be an orthogonal rotation matrix")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError(f"geometry.{key} must be a proper rotation matrix")
    return matrix


def _required_geometry(metadata: dict[str, Any]) -> dict[str, Any]:
    """Check the sidecar's schema version, dataset ID and geometry block; return the geometry.

    The orientation matrix must agree with the unit cell, and optional scan-point matrices
    and fixed/setting rotations must be valid.
    """

    schema_version = str(metadata.get("schema_version", "")).strip()
    if schema_version != GROND_ROW_SCHEMA_VERSION:
        raise ValueError(f"row metadata schema_version must be {GROND_ROW_SCHEMA_VERSION!r}")
    if not str(metadata.get("dataset_id", "")).strip():
        raise ValueError("row metadata must include a non-empty dataset_id")

    geometry = metadata.get("geometry") or {}
    required = (
        "wavelength_A",
        "incident_beam_direction",
        "rotation_axis",
        "starting_frame",
        "starting_angle_deg",
        "oscillation_range_deg",
        "unit_cell",
    )
    missing = [field for field in required if field not in geometry]
    if missing:
        raise ValueError("row metadata is missing geometry field(s): " + ", ".join(missing))
    if len(geometry["unit_cell"]) != 6:
        raise ValueError("geometry.unit_cell must have six values: a,b,c,alpha,beta,gamma")
    wavelength = float(geometry["wavelength_A"])
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("geometry.wavelength_A must be finite and positive")
    beam = np.asarray(geometry["incident_beam_direction"], dtype=float)
    if beam.shape != (3,) or not np.all(np.isfinite(beam)) or float(np.linalg.norm(beam)) <= 0.0:
        raise ValueError("geometry.incident_beam_direction must be a finite nonzero 3-vector")
    cell = np.asarray(geometry["unit_cell"], dtype=float)
    if not np.all(np.isfinite(cell)) or np.any(cell[:3] <= 0.0) or np.any(cell[3:] <= 0.0) or np.any(cell[3:] >= 180.0):
        raise ValueError("geometry.unit_cell must contain positive lengths and angles between 0 and 180 degrees")
    alpha, beta, gamma = np.radians(cell[3:])
    expected_metric = np.asarray(
        [
            [cell[0] ** 2, cell[0] * cell[1] * np.cos(gamma), cell[0] * cell[2] * np.cos(beta)],
            [cell[0] * cell[1] * np.cos(gamma), cell[1] ** 2, cell[1] * cell[2] * np.cos(alpha)],
            [cell[0] * cell[2] * np.cos(beta), cell[1] * cell[2] * np.cos(alpha), cell[2] ** 2],
        ],
        dtype=float,
    )
    if float(np.linalg.det(expected_metric)) <= 1e-12:
        raise ValueError("geometry.unit_cell has zero or invalid volume")
    scan_values = np.asarray(
        [
            geometry["starting_frame"],
            geometry["starting_angle_deg"],
            geometry["oscillation_range_deg"],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(scan_values)):
        raise ValueError("geometry scan origin and oscillation values must be finite")
    rotation_axis = np.asarray(geometry["rotation_axis"], dtype=float)
    if (
        rotation_axis.shape != (3,)
        or not np.all(np.isfinite(rotation_axis))
        or float(np.linalg.norm(rotation_axis)) <= 0.0
    ):
        raise ValueError("geometry.rotation_axis must be a finite nonzero 3-vector")
    reciprocal = _matrix_from_metadata(metadata)
    real = np.linalg.inv(reciprocal).T
    if "real_space_axes_A" in geometry:
        declared_real = np.asarray(geometry["real_space_axes_A"], dtype=float)
        if not np.allclose(declared_real, real, rtol=1e-7, atol=1e-9):
            raise ValueError("geometry reciprocal_A_matrix and real_space_axes_A are inconsistent")
    if not np.allclose(real.T @ real, expected_metric, rtol=1e-5, atol=1e-7):
        raise ValueError("geometry reciprocal/real-space axes are inconsistent with geometry.unit_cell")
    for index, matrix in enumerate(geometry.get("A_scan_points", [])):
        scan_matrix = np.asarray(matrix, dtype=float)
        if (
            scan_matrix.shape != (3, 3)
            or not np.all(np.isfinite(scan_matrix))
            or abs(float(np.linalg.det(scan_matrix))) <= 1e-15
        ):
            raise ValueError(f"geometry.A_scan_points[{index}] must be a finite nonsingular 3x3 matrix")
    _rotation_matrix_from_metadata(geometry, "fixed_rotation_matrix")
    _rotation_matrix_from_metadata(geometry, "setting_rotation_matrix")
    return geometry


def validate_row_contract(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> RowContractSummary:
    """Validate that rows and metadata contain the minimum information GROND needs."""

    if not rows:
        raise ValueError("no observation rows supplied")
    geometry = _required_geometry(metadata)
    warnings: list[str] = []
    finite_rows = 0
    positive_sigma_rows = 0
    has_frame = False
    has_alpha_deg = False
    dataset_ids = sorted({str(row.get("dataset_id") or metadata.get("dataset_id") or "dataset") for row in rows})

    for index, row in enumerate(rows, start=1):
        row = _normalized_row(row, index)
        if all(math.isfinite(float(row[field])) for field in ("Iobs", "sigma_Iobs")):
            finite_rows += 1
        if math.isfinite(float(row["sigma_Iobs"])) and float(row["sigma_Iobs"]) > 0:
            positive_sigma_rows += 1
        row_has_frame = "frame" in row and math.isfinite(float(row["frame"]))
        row_has_alpha = "alpha_deg" in row and math.isfinite(float(row["alpha_deg"]))
        has_frame = has_frame or row_has_frame
        has_alpha_deg = has_alpha_deg or row_has_alpha
        if not row_has_frame and not row_has_alpha:
            raise ValueError(f"row {index}: provide either frame or alpha_deg")

    if positive_sigma_rows != len(rows):
        warnings.append(
            "some rows have non-positive or non-finite sigma_Iobs and will be unusable for weighted scoring"
        )
    if len(dataset_ids) > 1:
        warnings.append("portable row files should normally contain one dataset; use prep JSON for multiple datasets")
    if abs(float(geometry["oscillation_range_deg"])) < 1e-12:
        warnings.append(
            "oscillation_range_deg is near zero; frame rows have no angular progression "
            "and alpha_deg rows cannot be converted to frame coordinates"
        )

    return RowContractSummary(
        schema_version=str(metadata.get("schema_version") or GROND_ROW_SCHEMA_VERSION),
        rows=len(rows),
        finite_rows=finite_rows,
        positive_sigma_rows=positive_sigma_rows,
        has_frame=has_frame,
        has_alpha_deg=has_alpha_deg,
        dataset_ids=dataset_ids,
        warnings=warnings,
    )


def example_row_csv() -> str:
    """Return a tiny portable row table template."""

    fields = (
        "row_id",
        "dataset_id",
        "h",
        "k",
        "l",
        "Iobs",
        "sigma_Iobs",
        "frame",
        "x_px",
        "y_px",
        "partiality",
        "lp",
        "profile_correlation",
        "bbox_z0",
        "bbox_z1",
    )
    rows = [
        {
            "row_id": "0",
            "dataset_id": "crystal_a",
            "h": "1",
            "k": "0",
            "l": "2",
            "Iobs": "1284.5",
            "sigma_Iobs": "42.1",
            "frame": "18.0",
            "x_px": "1912.4",
            "y_px": "2105.7",
            "partiality": "1.00",
            "lp": "1.0",
            "profile_correlation": "91.4",
            "bbox_z0": "17.5",
            "bbox_z1": "18.5",
        },
        {
            "row_id": "1",
            "dataset_id": "crystal_a",
            "h": "0",
            "k": "1",
            "l": "3",
            "Iobs": "422.8",
            "sigma_Iobs": "27.6",
            "frame": "18.0",
            "x_px": "2063.9",
            "y_px": "1988.2",
            "partiality": "0.95",
            "lp": "1.0",
            "profile_correlation": "88.7",
            "bbox_z0": "17.4",
            "bbox_z1": "18.6",
        },
    ]
    out = StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def example_metadata() -> dict[str, Any]:
    """Return a documented metadata sidecar template for portable rows."""

    return {
        "schema_version": GROND_ROW_SCHEMA_VERSION,
        "dataset_id": "crystal_a",
        "description": "Portable GROND row-contract metadata. Replace every numeric value with your integration geometry.",
        "geometry": {
            "wavelength_A": 0.02508,
            "incident_beam_direction": [0.0, 0.0, 1.0],
            "rotation_axis": [1.0, 0.0, 0.0],
            "starting_frame": 1.0,
            "starting_angle_deg": 0.0,
            "oscillation_range_deg": 0.5,
            "unit_cell": [7.0, 8.0, 9.0, 90.0, 90.0, 90.0],
            "reciprocal_A_matrix": [
                [0.1428571429, 0.0, 0.0],
                [0.0, 0.125, 0.0],
                [0.0, 0.0, 0.1111111111],
            ],
            "fixed_rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "setting_rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "notes": "Geometry uses g(alpha) = S R(alpha) F A(alpha) h; rotation_axis is the axis of R.",
        },
        "columns": {
            "Iobs": "background-subtracted integrated intensity before GROND correction",
            "sigma_Iobs": "standard uncertainty of Iobs in the same scale",
            "frame": "calculated or centroid image/frame coordinate in the same convention as starting_frame",
            "alpha_deg": "optional direct goniometer angle; use this instead of frame when it is more natural",
            "partiality": "optional 0..1 row support value; GROND writes 100*partiality into the compatibility PEAK field",
        },
    }


def write_template(rows_path: str | Path, metadata_path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Write a portable row CSV template and matching metadata JSON template."""

    rows_out = Path(rows_path).expanduser()
    meta_out = Path(metadata_path).expanduser()
    for path in (rows_out, meta_out):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    rows_out.write_text(example_row_csv())
    meta_out.write_text(json.dumps(example_metadata(), indent=2) + "\n")
    return rows_out.resolve(), meta_out.resolve()


def _row_frame(row: dict[str, Any], geometry: dict[str, Any]) -> float:
    """Return a row's frame number, converting the goniometer angle ``alpha_deg`` (degrees) with the scan
    origin and oscillation range when no frame is given.
    """

    if "frame" in row and math.isfinite(float(row["frame"])):
        return float(row["frame"])
    omega_deg = float(row["alpha_deg"])
    osc = float(geometry["oscillation_range_deg"])
    if abs(osc) < 1e-12:
        raise ValueError("cannot convert alpha_deg to frame because oscillation_range_deg is zero")
    return float(geometry["starting_frame"]) + (omega_deg - float(geometry["starting_angle_deg"])) / osc


def write_xds_compatibility_dir(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an XDS-like compatibility directory consumed by the current engine."""

    rows = [_normalized_row(row, index) for index, row in enumerate(rows, start=1)]
    summary = validate_row_contract(rows, metadata)
    if summary.finite_rows != len(rows):
        raise ValueError(
            "all portable rows must have finite Iobs and sigma_Iobs before writing the compatibility directory"
        )
    if summary.positive_sigma_rows != len(rows):
        raise ValueError("all portable rows must have positive sigma_Iobs before writing the compatibility directory")

    geometry = _required_geometry(metadata)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    hkl_path = out / "INTEGRATE.HKL"
    if hkl_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {hkl_path}")

    real = _real_space_axes_from_metadata(metadata)
    fixed_rotation = _rotation_matrix_from_metadata(geometry, "fixed_rotation_matrix")
    setting_rotation = _rotation_matrix_from_metadata(geometry, "setting_rotation_matrix")
    rotation_axis = np.asarray(geometry["rotation_axis"], dtype=float)
    rotation_axis_lab = setting_rotation @ rotation_axis
    scan_points = metadata.get("geometry", {}).get("A_scan_points")
    scan_points_name = ""
    if scan_points:
        mats = np.asarray(scan_points, dtype=float)
        if mats.ndim != 3 or mats.shape[1:] != (3, 3):
            raise ValueError("geometry.A_scan_points must be a list of 3x3 matrices")
        scan_points_name = "GROND_A_AT_SCAN_POINTS.txt"
        with (out / scan_points_name).open("w") as handle:
            for mat in mats:
                handle.write(" ".join(f"{float(v):.12g}" for v in np.asarray(mat).reshape(-1)) + "\n")

    with hkl_path.open("w") as handle:
        handle.write("!FORMAT=GROND_PORTABLE_XDS_COMPAT\n")
        handle.write(f"!GROND_ROW_SCHEMA_VERSION= {metadata.get('schema_version', GROND_ROW_SCHEMA_VERSION)}\n")
        handle.write(f"!GROND_DATASET_ID= {metadata.get('dataset_id', 'dataset')}\n")
        handle.write(f"!X-RAY_WAVELENGTH= {float(geometry['wavelength_A']):.12g}\n")
        handle.write(f"!STARTING_FRAME= {float(geometry['starting_frame']):.12g}\n")
        handle.write(f"!STARTING_ANGLE= {float(geometry['starting_angle_deg']):.12g}\n")
        handle.write(f"!OSCILLATION_RANGE= {float(geometry['oscillation_range_deg']):.12g}\n")
        handle.write("!ROTATION_AXIS= " + " ".join(f"{float(v):.12g}" for v in rotation_axis_lab) + "\n")
        handle.write("!GROND_ROTATION_AXIS_DATUM= " + " ".join(f"{float(v):.12g}" for v in rotation_axis) + "\n")
        handle.write(
            "!GROND_FIXED_ROTATION= " + " ".join(f"{float(v):.12g}" for v in fixed_rotation.reshape(-1)) + "\n"
        )
        handle.write(
            "!GROND_SETTING_ROTATION= " + " ".join(f"{float(v):.12g}" for v in setting_rotation.reshape(-1)) + "\n"
        )
        handle.write(
            "!INCIDENT_BEAM_DIRECTION= "
            + " ".join(f"{float(v):.12g}" for v in geometry["incident_beam_direction"])
            + "\n"
        )
        handle.write("!UNIT_CELL_CONSTANTS= " + " ".join(f"{float(v):.12g}" for v in geometry["unit_cell"]) + "\n")
        handle.write("!UNIT_CELL_A-AXIS= " + " ".join(f"{float(v):.12g}" for v in real[:, 0]) + "\n")
        handle.write("!UNIT_CELL_B-AXIS= " + " ".join(f"{float(v):.12g}" for v in real[:, 1]) + "\n")
        handle.write("!UNIT_CELL_C-AXIS= " + " ".join(f"{float(v):.12g}" for v in real[:, 2]) + "\n")
        handle.write("!VARIANCE_MODEL= 1.0 0.0\n")
        if scan_points_name:
            handle.write(f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_points_name}\n")
        handle.write(
            "!H K L IOBS SIGMA XCAL YCAL ZCAL RLP PEAK CORR MAXC XOBS YOBS ZOBS ALF0 BET0 ALF1 BET1 PSI ISEG\n"
        )
        for row in rows:
            frame = _row_frame(row, geometry)
            xcal = float(row.get("x_px", 0.0))
            ycal = float(row.get("y_px", 0.0))
            zobs = float(row.get("frame_obs", frame))
            rlp = float(row.get("rlp", row.get("lp", 1.0)))
            peak = float(row.get("peak", 100.0 * float(row.get("partiality", 1.0))))
            corr = float(row.get("corr", row.get("profile_correlation", 100.0)))
            alf0 = float(row.get("bbox_z0", frame - 0.5))
            bet0 = float(row.get("bbox_z1", frame + 0.5))
            iseg = int(float(row.get("panel", 0) or 0))
            values = [
                int(row["h"]),
                int(row["k"]),
                int(row["l"]),
                float(row["Iobs"]),
                float(row["sigma_Iobs"]),
                xcal,
                ycal,
                frame,
                rlp,
                peak,
                corr,
                0.0,
                xcal,
                ycal,
                zobs,
                alf0,
                bet0,
                0.0,
                0.0,
                0.0,
                iseg,
            ]
            # Explicit separators prevent large fields from running together;
            # 17 significant digits preserve finite binary64 observations/sigmas.
            handle.write(
                " ".join(str(v) for v in values[:3]) + " "
                + " ".join(f"{float(v):.17g}" for v in values[3:20])
                + f" {values[20]}\n"
            )

    manifest = {
        "schema_version": GROND_ROW_SCHEMA_VERSION,
        "source": "portable_row_contract",
        "rows": len(rows),
        "integrate_hkl": str(hkl_path.resolve()),
        "metadata": metadata,
    }
    (out / "GROND_ROW_CONTRACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return hkl_path.resolve()
