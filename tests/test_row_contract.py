from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from grond.io.row_contract import (
    load_row_metadata,
    read_row_table,
    validate_row_contract,
    write_template,
    write_xds_compatibility_dir,
)
from grond.legacy.engine.bloch_engine import XDSBlochPort, XDSData


def test_row_contract_template_validates_and_writes_xds_compat(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")

    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    summary = validate_row_contract(rows, metadata)

    assert summary.schema_version == "grond-row-v1"
    assert summary.rows == 2
    assert summary.positive_sigma_rows == 2
    assert summary.has_frame is True

    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "bridge")
    assert hkl.name == "INTEGRATE.HKL"
    text = hkl.read_text()
    assert "!FORMAT=GROND_PORTABLE_XDS_COMPAT" in text
    assert "!GROND_ROW_SCHEMA_VERSION= grond-row-v1" in text

    data = XDSData(hkl)
    assert len(data.hkl) == 2
    assert tuple(data.hkl[0]) == (1, 0, 2)
    assert data.iobs[0] == 1284.5
    assert data.peak[1] == 95.0


def test_portable_geometry_preserves_full_setting_rotation_fixed_rotation_chain(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    fixed = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    setting = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    metadata["geometry"]["rotation_axis"] = [0.0, 1.0, 0.0]
    metadata["geometry"]["fixed_rotation_matrix"] = fixed.tolist()
    metadata["geometry"]["setting_rotation_matrix"] = setting.tolist()

    hkl_path = write_xds_compatibility_dir(rows, metadata, tmp_path / "full_geometry")
    data = XDSData(hkl_path)
    port = object.__new__(XDSBlochPort)
    port.data = data

    omega = 31.0
    theta = np.radians(omega)
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    hkl = np.array([[2, -1, 3]], dtype=int)
    expected = (setting @ rotation @ fixed @ data.A @ hkl[0]).reshape(1, 3)

    assert data.fixed_rotation == pytest.approx(fixed)
    assert data.setting_rotation == pytest.approx(setting)
    assert data.axis == pytest.approx([0.0, 1.0, 0.0])
    assert data.rotation_axis_lab == pytest.approx(setting @ np.array([0.0, 1.0, 0.0]))
    assert port.gvecs(hkl, omega) == pytest.approx(expected, abs=1e-12)


def test_portable_geometry_rejects_improper_fixed_rotation(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    metadata["geometry"]["fixed_rotation_matrix"] = np.diag([1.0, 1.0, -1.0]).tolist()

    with pytest.raises(ValueError, match="proper rotation matrix"):
        validate_row_contract(rows, metadata)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wavelength", "wavelength_A"),
        ("beam", "incident_beam_direction"),
        ("cell", "unit_cell"),
        ("singular_a", "reciprocal_A_matrix must be nonsingular"),
        (
            "inconsistent_axes",
            "reciprocal_A_matrix and real_space_axes_A are inconsistent",
        ),
    ],
)
def test_portable_geometry_rejects_unphysical_metadata(tmp_path: Path, case: str, message: str):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    geometry = metadata["geometry"]
    if case == "wavelength":
        geometry["wavelength_A"] = 0.0
    elif case == "beam":
        geometry["incident_beam_direction"] = [0.0, 0.0, 0.0]
    elif case == "cell":
        geometry["unit_cell"][3] = 0.0
    elif case == "singular_a":
        geometry["reciprocal_A_matrix"] = np.zeros((3, 3)).tolist()
    elif case == "inconsistent_axes":
        geometry["real_space_axes_A"] = np.linalg.inv(
            np.asarray(geometry["reciprocal_A_matrix"], dtype=float)
        ).T.tolist()
        geometry["real_space_axes_A"][0][0] *= 2.0

    with pytest.raises(ValueError, match=message):
        validate_row_contract(rows, metadata)


def test_portable_row_schema_is_shipped_with_source_tree():
    root = Path(__file__).resolve().parents[1]
    schema_path = root / "schemas" / "grond-row-v1.schema.json"
    package_schema_path = files("grond") / "schemas" / "grond-row-v1.schema.json"
    manifest_path = root / "MANIFEST.in"

    schema = json.loads(schema_path.read_text())
    package_schema = json.loads(package_schema_path.read_text())
    assert package_schema == schema
    assert schema["$id"].endswith("/schemas/grond-row-v1.schema.json")
    assert schema["properties"]["schema_version"]["const"] == "grond-row-v1"
    assert "recursive-include schemas *.json" in manifest_path.read_text()


def test_row_contract_accepts_alpha_degrees(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    text = rows_path.read_text().replace(",18.0,", ",,").replace("frame,", "alpha_deg,")
    rows_path.write_text(text)

    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    for row in rows:
        row["alpha_deg"] = 8.5
        row.pop("frame", None)

    summary = validate_row_contract(rows, metadata)
    assert summary.has_alpha_deg is True
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "bridge")
    data = XDSData(hkl)
    assert data.zcal[0] == 18.0


def test_row_contract_rejects_fractional_miller_indices(tmp_path: Path):
    rows_path = tmp_path / "rows.csv"
    rows_path.write_text("h,k,l,Iobs,sigma_Iobs,frame\n1.5,0,2,100.0,5.0,1.0\n")

    with pytest.raises(ValueError, match="integer Miller index"):
        read_row_table(rows_path)


def test_row_contract_preserves_invalid_sigma_for_validation(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows_path.write_text(rows_path.read_text().replace("42.1", "-42.1", 1))

    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    summary = validate_row_contract(rows, metadata)

    assert rows[0]["sigma_Iobs"] == -42.1
    assert summary.positive_sigma_rows == 1
    assert summary.warnings == [
        "some rows have non-positive or non-finite sigma_Iobs and will be unusable for weighted scoring"
    ]
    with pytest.raises(ValueError, match="positive sigma_Iobs"):
        write_xds_compatibility_dir(rows, metadata, tmp_path / "bridge")


def test_row_contract_rejects_wrong_metadata_schema_version(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = "grond-row-v0"
    metadata_path.write_text(json.dumps(metadata))

    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    with pytest.raises(ValueError, match="schema_version"):
        validate_row_contract(rows, metadata)


def test_xds_reader_rejects_empty_reflection_table(tmp_path: Path):
    path = tmp_path / "empty.HKL"
    path.write_text("!UNIT_CELL_CONSTANTS= 10 10 10 90 90 90\n")

    with pytest.raises(ValueError, match="no numeric reflection rows"):
        XDSData(path)


def test_xds_reader_preserves_invalid_sigma_sign(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "bridge")
    lines = hkl.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("!"):
            fields = line.split()
            fields[4] = "-42.1"
            lines[index] = " ".join(fields)
            break
    hkl.write_text("\n".join(lines) + "\n")

    data = XDSData(hkl)

    assert data.sigma[0] == -42.1


def test_xds_reader_rejects_fractional_miller_indices(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "fractional_bridge")
    lines = hkl.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("!"):
            fields = line.split()
            fields[0] = "1.5"
            lines[index] = " ".join(fields)
            break
    hkl.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="non-integer Miller indices"):
        XDSData(hkl)


def test_scan_varying_a_matrices_are_indexed_relative_to_starting_frame(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    metadata["geometry"]["starting_frame"] = 5.0
    first = np.diag([0.10, 0.11, 0.12])
    second = np.diag([0.20, 0.21, 0.22])
    metadata["geometry"]["A_scan_points"] = [first.tolist(), second.tolist()]
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "scan_bridge")
    data = XDSData(hkl)

    np.testing.assert_allclose(data.A_for_omega(data.starting_angle), first)
    np.testing.assert_allclose(
        data.A_for_omega(data.starting_angle + 0.5 * data.oscillation_range),
        0.5 * (first + second),
    )
    np.testing.assert_allclose(data.A_for_omega(data.starting_angle + data.oscillation_range), second)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "contains no matrices"),
        ("1 0 0 0 1 0 0 0\n", "exactly nine"),
        ("1 0 0 0 nope 0 0 0 1\n", "nonnumeric"),
        (
            "1 0 0 0 1 0 0 0 1\n1 0 0 0 1 0 0 0\n",
            "exactly nine",
        ),
    ],
)
def test_declared_scan_varying_a_sidecar_is_strict(tmp_path: Path, contents: str, message: str):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    metadata["geometry"]["A_scan_points"] = [np.eye(3).tolist()]
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "scan_bridge")
    (hkl.parent / "GROND_A_AT_SCAN_POINTS.txt").write_text(contents)

    with pytest.raises(ValueError, match=message):
        XDSData(hkl)
