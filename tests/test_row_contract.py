from __future__ import annotations

from pathlib import Path

from grond.io.row_contract import (
    load_row_metadata,
    read_row_table,
    validate_row_contract,
    write_template,
    write_xds_compatibility_dir,
)
from grond.legacy.engine.bloch_engine import XDSData


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

