from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
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
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: field {field!r} is not numeric: {value!r}") from exc


def _as_int(value: Any, *, field: str, row_number: int) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"row {row_number}: required integer field {field!r} is blank")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: field {field!r} is not an integer: {value!r}") from exc


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
    out["sigma_Iobs"] = abs(_as_float(out["sigma_Iobs"], field="sigma_Iobs", row_number=row_number))
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
    metadata.setdefault("schema_version", GROND_ROW_SCHEMA_VERSION)
    return metadata


def _matrix_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    geometry = metadata.get("geometry") or {}
    if "reciprocal_A_matrix" in geometry:
        mat = np.asarray(geometry["reciprocal_A_matrix"], dtype=float)
        if mat.shape != (3, 3):
            raise ValueError("geometry.reciprocal_A_matrix must be 3x3")
        return mat
    if "real_space_axes_A" in geometry:
        real = np.asarray(geometry["real_space_axes_A"], dtype=float)
        if real.shape != (3, 3):
            raise ValueError("geometry.real_space_axes_A must be 3x3 with axes as columns")
        return np.linalg.inv(real).T
    raise ValueError("row metadata must include geometry.reciprocal_A_matrix or geometry.real_space_axes_A")


def _real_space_axes_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    geometry = metadata.get("geometry") or {}
    if "real_space_axes_A" in geometry:
        real = np.asarray(geometry["real_space_axes_A"], dtype=float)
        if real.shape != (3, 3):
            raise ValueError("geometry.real_space_axes_A must be 3x3 with axes as columns")
        return real
    return np.linalg.inv(_matrix_from_metadata(metadata).T)


def _required_geometry(metadata: dict[str, Any]) -> dict[str, Any]:
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
    _matrix_from_metadata(metadata)
    return geometry


def validate_row_contract(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> RowContractSummary:
    """Validate that rows and metadata contain the minimum information GROND needs."""

    geometry = _required_geometry(metadata)
    warnings: list[str] = []
    finite_rows = 0
    positive_sigma_rows = 0
    has_frame = False
    has_alpha_deg = False
    dataset_ids = sorted({str(row.get("dataset_id") or metadata.get("dataset_id") or "dataset") for row in rows})

    for index, row in enumerate(rows, start=1):
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
        warnings.append("some rows have non-positive or non-finite sigma_Iobs and will be unusable for weighted scoring")
    if len(dataset_ids) > 1:
        warnings.append("portable row files should normally contain one dataset; use prep JSON for multiple datasets")
    if abs(float(geometry["oscillation_range_deg"])) < 1e-12:
        warnings.append("oscillation_range_deg is near zero; alpha_deg rows can still be converted but frame rows are ambiguous")

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
    out = []
    writer = csv.DictWriter(_ListWriter(out), fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return "".join(out)


class _ListWriter:
    """Tiny file-like adapter used to build CSV templates without StringIO import noise."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    def write(self, value: str) -> None:
        self.parts.append(value)


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
            "notes": "reciprocal_A_matrix maps signed h,k,l to a reciprocal vector in lab coordinates before goniometer rotation.",
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
    if "frame" in row and math.isfinite(float(row["frame"])):
        return float(row["frame"])
    alpha = float(row["alpha_deg"])
    osc = float(geometry["oscillation_range_deg"])
    if abs(osc) < 1e-12:
        raise ValueError("cannot convert alpha_deg to frame because oscillation_range_deg is zero")
    return float(geometry["starting_frame"]) + (alpha - float(geometry["starting_angle_deg"])) / osc


def write_xds_compatibility_dir(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an XDS-like compatibility directory consumed by the current engine."""

    geometry = _required_geometry(metadata)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    hkl_path = out / "INTEGRATE.HKL"
    if hkl_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {hkl_path}")

    real = _real_space_axes_from_metadata(metadata)
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
        handle.write("!ROTATION_AXIS= " + " ".join(f"{float(v):.12g}" for v in geometry["rotation_axis"]) + "\n")
        handle.write("!INCIDENT_BEAM_DIRECTION= " + " ".join(f"{float(v):.12g}" for v in geometry["incident_beam_direction"]) + "\n")
        handle.write("!UNIT_CELL_CONSTANTS= " + " ".join(f"{float(v):.12g}" for v in geometry["unit_cell"]) + "\n")
        handle.write("!UNIT_CELL_A-AXIS= " + " ".join(f"{float(v):.12g}" for v in real[:, 0]) + "\n")
        handle.write("!UNIT_CELL_B-AXIS= " + " ".join(f"{float(v):.12g}" for v in real[:, 1]) + "\n")
        handle.write("!UNIT_CELL_C-AXIS= " + " ".join(f"{float(v):.12g}" for v in real[:, 2]) + "\n")
        handle.write("!VARIANCE_MODEL= 1.0 0.0\n")
        if scan_points_name:
            handle.write(f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_points_name}\n")
        handle.write("!H K L IOBS SIGMA XCAL YCAL ZCAL RLP PEAK CORR MAXC XOBS YOBS ZOBS ALF0 BET0 ALF1 BET1 PSI ISEG\n")
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
            handle.write(
                f"{values[0]:5d}{values[1]:5d}{values[2]:5d}"
                + "".join(f"{float(v):14.5f}" for v in values[3:20])
                + f"{values[20]:5d}\n"
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

