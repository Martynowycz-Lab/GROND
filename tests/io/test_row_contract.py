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
from grond.legacy.engine.native_row_regularize import geometry_tail_mask, row_selection
from grond.workflows.decision import ALL_PROPER_SIGNED_PERMUTATIONS
from tests._support import REPO_ROOT


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


def test_xds_reader_normalizes_missing_observed_centroid_without_dropping_row(tmp_path: Path):
    rows_path, metadata_path = write_template(
        tmp_path / "rows.csv", tmp_path / "rows_metadata.json"
    )
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "bridge")
    lines = hkl.read_text().splitlines()
    numeric = [index for index, line in enumerate(lines) if not line.startswith("!")]
    fields = lines[numeric[0]].split()
    fields[12:15] = ["0", "0", "0"]
    lines[numeric[0]] = " ".join(fields)
    hkl.write_text("\n".join(lines) + "\n")

    data = XDSData(hkl)
    assert len(data.hkl) == 2
    assert np.isnan(data.xobs[0])
    assert np.isnan(data.yobs[0])
    assert np.isnan(data.zobs[0])

    keep, summary = geometry_tail_mask(
        data, np.ones(len(data.hkl), dtype=bool), 97.5
    )
    assert keep[0]
    assert summary["observed_geometry_rows"] == 1
    assert summary["missing_observed_geometry_rows"] == 1


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
    root = REPO_ROOT
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


def _write_independently_rounded_cell_xds(tmp_path: Path) -> Path:
    """Write XDS-like geometry whose cell and axes round independently."""

    real = np.asarray(
        [
            [-3.968, -4.940, -9.948],
            [-0.477, 7.859, -13.704],
            [3.367, -4.667, -13.384],
        ],
        dtype=float,
    )
    source_a = np.linalg.inv(real).T
    scan_path = tmp_path / "DIALS_A_AT_SCAN_POINTS.txt"
    scan_path.write_text(
        "\n".join(
            " ".join(f"{float(value):.16g}" for value in matrix.reshape(-1))
            for matrix in (source_a, source_a * 1.0001)
        )
        + "\n"
    )
    hkl = tmp_path / "INTEGRATE.HKL"
    rows = [
        (-1, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
    ]
    lines = [
        "!X-RAY_WAVELENGTH= 0.025079",
        "!STARTING_FRAME= 1.0",
        "!STARTING_ANGLE= 0.0",
        "!OSCILLATION_RANGE= 1.0",
        "!ROTATION_AXIS= 0.0 1.0 0.0",
        "!INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0",
        "!UNIT_CELL_CONSTANTS= 5.226 10.390 21.585 89.002 89.520 89.855",
        "!UNIT_CELL_A-AXIS= -3.968 -0.477 3.367",
        "!UNIT_CELL_B-AXIS= -4.940 7.859 -4.667",
        "!UNIT_CELL_C-AXIS= -9.948 -13.704 -13.384",
        f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_path.name}",
    ]
    for ordinal, (h, k, ell) in enumerate(rows, start=1):
        lines.append(
            f"{h} {k} {ell} {100.0 + ordinal} 2.0 "
            f"100.0 200.0 {float(ordinal):.1f} 0.1 95.0 100.0 "
            f"100.0 100.0 200.0 {float(ordinal):.1f} "
            "0.0 0.0 0.0 0.0 0.0 0"
        )
    hkl.write_text("\n".join(lines) + "\n")
    return hkl


def _cell_parameters(cell) -> np.ndarray:
    return np.asarray(
        [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        dtype=float,
    )


def test_working_cell_uses_axis_metric_for_identity_and_every_proper_transform(
    tmp_path: Path,
):
    hkl = _write_independently_rounded_cell_xds(tmp_path)
    identity = XDSData(hkl, hkl_transform="identity")
    raw_hkl = identity.raw_hkl.copy()
    raw_cell_parameters = np.asarray(
        [5.226, 10.390, 21.585, 89.002, 89.520, 89.855],
        dtype=float,
    )

    np.testing.assert_allclose(
        _cell_parameters(identity.raw_cell),
        raw_cell_parameters,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _cell_parameters(identity.cell),
        _cell_parameters(identity.axis_derived_source_cell),
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(
        _cell_parameters(identity.raw_cell),
        _cell_parameters(identity.axis_derived_source_cell),
        rtol=0.0,
        atol=1e-8,
    )
    assert identity.axis_derived_source_cell_basis == "UNIT_CELL_[ABC]-AXIS"
    assert identity.working_cell_basis == "axis_derived_oriented_model_basis"
    assert identity.header["HKL_TRANSFORM_TO_MODEL_BASIS"] == "identity"

    threshold_hkl = tuple(int(value) for value in raw_hkl[0])
    d_from_header = float(identity.raw_cell.calculate_d(threshold_hkl))
    d_from_axes = float(identity.cell.calculate_d(threshold_hkl))
    cutoff = 0.5 * (d_from_header + d_from_axes)
    assert d_from_header > cutoff > d_from_axes
    reference_d = np.asarray(
        [
            identity.cell.calculate_d(tuple(int(value) for value in row))
            for row in identity.hkl
        ],
        dtype=float,
    )
    reference_selection = row_selection(
        identity,
        identity.iobs,
        positive_only=True,
        min_d=cutoff,
        min_peak=90.0,
        geometry_percentile=100.0,
    )
    assert 0 not in reference_selection

    for transform in ALL_PROPER_SIGNED_PERMUTATIONS:
        transformed = XDSData(hkl, hkl_transform=transform)
        transformed_d = np.asarray(
            [
                transformed.cell.calculate_d(tuple(int(value) for value in row))
                for row in transformed.hkl
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            _cell_parameters(transformed.raw_cell),
            raw_cell_parameters,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _cell_parameters(transformed.axis_derived_source_cell),
            _cell_parameters(identity.axis_derived_source_cell),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(transformed_d, reference_d, rtol=2e-15, atol=2e-15)
        assert transformed.cell.volume == pytest.approx(
            identity.cell.volume,
            rel=2e-15,
            abs=2e-12,
        )
        np.testing.assert_allclose(
            (transformed.A @ transformed.hkl.T).T,
            (identity.A @ raw_hkl.T).T,
            rtol=2e-15,
            atol=2e-15,
        )
        for scan_index in range(len(identity.A_at_scan_points)):
            np.testing.assert_allclose(
                (transformed.A_at_scan_points[scan_index] @ transformed.hkl.T).T,
                (identity.A_at_scan_points[scan_index] @ raw_hkl.T).T,
                rtol=2e-15,
                atol=2e-15,
            )
        selected = row_selection(
            transformed,
            transformed.iobs,
            positive_only=True,
            min_d=cutoff,
            min_peak=90.0,
            geometry_percentile=100.0,
        )
        np.testing.assert_array_equal(selected, reference_selection)
        assert transformed.header["HKL_TRANSFORM_TO_MODEL_BASIS"] == (
            "identity" if transform == "identity" else transform
        )
        materialized_cell = np.asarray(
            [
                float(value)
                for value in transformed.header[
                    "UNIT_CELL_CONSTANTS_MODEL_BASIS"
                ].split()
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            materialized_cell,
            _cell_parameters(transformed.cell),
            rtol=0.0,
            atol=5e-7,
        )


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


def test_relative_scan_varying_a_sidecar_cannot_be_hijacked_by_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    expected = np.diag([0.10, 0.11, 0.12])
    metadata["geometry"]["A_scan_points"] = [expected.tolist()]
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "dataset")

    hostile_cwd = tmp_path / "hostile_cwd"
    hostile_cwd.mkdir()
    (hostile_cwd / "GROND_A_AT_SCAN_POINTS.txt").write_text(
        "0.2 0 0 0 0.2 0 0 0 0.2\n"
    )
    monkeypatch.chdir(hostile_cwd)

    data = XDSData(hkl)

    assert data.a_scan_points_path == (hkl.parent / "GROND_A_AT_SCAN_POINTS.txt").resolve()
    np.testing.assert_allclose(data.A_at_scan_points, expected[None, :, :])


def test_missing_relative_scan_varying_a_sidecar_does_not_fall_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    metadata["geometry"]["A_scan_points"] = [np.eye(3).tolist()]
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "dataset")
    (hkl.parent / "GROND_A_AT_SCAN_POINTS.txt").unlink()

    hostile_cwd = tmp_path / "hostile_cwd"
    hostile_cwd.mkdir()
    (hostile_cwd / "GROND_A_AT_SCAN_POINTS.txt").write_text(
        "0.2 0 0 0 0.2 0 0 0 0.2\n"
    )
    monkeypatch.chdir(hostile_cwd)

    with pytest.raises(FileNotFoundError) as exc_info:
        XDSData(hkl)

    assert exc_info.value.filename == str(hkl.parent / "GROND_A_AT_SCAN_POINTS.txt")


def test_absolute_scan_varying_a_sidecar_is_preserved(tmp_path: Path):
    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows = read_row_table(rows_path)
    metadata = load_row_metadata(metadata_path)
    metadata["geometry"]["A_scan_points"] = [np.eye(3).tolist()]
    hkl = write_xds_compatibility_dir(rows, metadata, tmp_path / "dataset")
    external_sidecar = tmp_path / "external" / "absolute_scan_points.txt"
    external_sidecar.parent.mkdir()
    expected = np.diag([0.20, 0.21, 0.22])
    external_sidecar.write_text(" ".join(str(value) for value in expected.reshape(-1)) + "\n")
    hkl.write_text(
        hkl.read_text().replace(
            "!DIALS_A_AT_SCAN_POINTS_FILE= GROND_A_AT_SCAN_POINTS.txt",
            f"!DIALS_A_AT_SCAN_POINTS_FILE= {external_sidecar}",
        )
    )

    data = XDSData(hkl)

    assert data.a_scan_points_path == external_sidecar.resolve()
    np.testing.assert_allclose(data.A_at_scan_points, expected[None, :, :])


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
