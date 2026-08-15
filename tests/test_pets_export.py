from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grond.io._dials_pets_helper import (
    _read_xds_profile_fitting,
    _xds_correct_acceptance,
)
from grond.io.pets_export import (
    _validate_and_repair_pets,
    export_pets,
    xds_recorded_observation,
)


def _pets_text(*, dstarmax: float = 0.0) -> str:
    return f"""data_pets
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_diffrn_radiation_wavelength 0.02508
_diffrn_orient_matrix_UB_11 0.1
_diffrn_orient_matrix_UB_12 0
_diffrn_orient_matrix_UB_13 0
_diffrn_orient_matrix_UB_21 0
_diffrn_orient_matrix_UB_22 0.09090909
_diffrn_orient_matrix_UB_23 0
_diffrn_orient_matrix_UB_31 0
_diffrn_orient_matrix_UB_32 0
_diffrn_orient_matrix_UB_33 0.08333333
_diffrn_measurement_details
;
data collection geometry: continuous rotation
dstarmax:  {dstarmax:.3f}
mosaicity:  0.0
;

loop_
_diffrn_zone_axis_id
_diffrn_zone_axis_u
_diffrn_zone_axis_v
_diffrn_zone_axis_w
_diffrn_zone_axis_precession_angle
_diffrn_zone_axis_alpha
_diffrn_zone_axis_beta
_diffrn_zone_axis_omega
_diffrn_zone_axis_scale
1 0 0 1 0 0 0 0 1

loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_intensity_meas
_refln_intensity_sigma
_refln_zone_axis_id
1 0 0 100 5 1
0 2 0 40 4 1
"""


def test_xds_recorded_observation_inverts_documented_correction() -> None:
    intensity, sigma = xds_recorded_observation(800.0, 40.0, 80.0, 2.0)
    assert intensity == pytest.approx(320.0)
    assert sigma == pytest.approx(16.0)
    assert intensity * 2.0 / 0.8 == pytest.approx(800.0)


def test_pets_validation_repairs_dstarmax(tmp_path: Path) -> None:
    path = tmp_path / "test.cif_pets"
    path.write_text(_pets_text())
    payload = _validate_and_repair_pets(path)
    assert payload["rows"] == 2
    assert payload["zone_axes"] == 1
    assert payload["dstarmax_inverse_angstrom"] == pytest.approx(2.0 / 11.0)
    assert "dstarmax:  0.181818" in path.read_text()


def test_pets_validation_rejects_unknown_zone_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.cif_pets"
    path.write_text(_pets_text().replace("0 2 0 40 4 1", "0 2 0 40 4 9"))
    with pytest.raises(ValueError, match="invalid reflection"):
        _validate_and_repair_pets(path)


def test_xds_profile_fitting_contract(tmp_path: Path) -> None:
    xds_inp = tmp_path / "XDS.INP"
    integrate = tmp_path / "INTEGRATE.HKL"
    xds_inp.write_text("! PROFILE_FITTING=TRUE is only a comment\nPROFILE_FITTING=FALSE\n")
    integrate.write_text("!PROFILE_FITTING= FALSE\n")
    assert _read_xds_profile_fitting(xds_inp, integrate) is False
    integrate.write_text("!PROFILE_FITTING= TRUE\n")
    with pytest.raises(ValueError, match="disagree"):
        _read_xds_profile_fitting(xds_inp, integrate)


def test_xds_correct_rows_bind_by_position_and_reindexed_hkl(tmp_path: Path) -> None:
    xds_ascii = tmp_path / "XDS_ASCII.HKL"
    xds_ascii.write_text(
        "!FORMAT=XDS_ASCII\n"
        " 1 0 0 100 10 12.34 23.45 34.56 1 1 1 1\n"
        " 0 1 0 80 8 15.00 25.00 35.00 1 1 1 1\n"
    )
    handle = SimpleNamespace(xyzcal=np.asarray([[12.34, 23.45, 34.56], [15.0, 25.0, 35.0]]))
    keep, row_count = _xds_correct_acceptance(
        handle,
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=int),
        xds_ascii,
    )
    assert row_count == 2
    assert keep.tolist() == [True, True]


def test_export_pets_writes_manifest_and_hashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    refl = tmp_path / "input.refl"
    expt = tmp_path / "input.expt"
    refl.write_bytes(b"reflection-table")
    expt.write_text("{}\n")
    out_dir = tmp_path / "export"

    def fake_run(command, *, text, capture_output, check, env):
        assert text and capture_output and not check
        assert env["MPLCONFIGDIR"]
        config_path = Path(command[command.index("--config") + 1])
        result_path = Path(command[command.index("--result") + 1])
        config = json.loads(config_path.read_text())
        output = Path(config["out_dir"]) / "sample.cif_pets"
        output.write_text(_pets_text())
        result_path.write_text(
            json.dumps(
                {
                    "source_kind": "DIALS",
                    "dials_version": "test",
                    "outputs": [{"path": str(output)}],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("grond.io.pets_export.subprocess.run", fake_run)
    result = export_pets(
        out_dir=out_dir,
        dataset_name="sample",
        dials_python="python",
        dials_refl=refl,
        dials_expt=expt,
    )
    manifest = json.loads(result.manifest.read_text())
    assert result.files == (out_dir / "sample.cif_pets",)
    assert manifest["status"] == "complete"
    assert manifest["outputs"][0]["rows"] == 2
    assert manifest["source_sha256"][str(refl.resolve())]
    assert result.readme.is_file()


def test_export_pets_requires_exactly_one_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one source"):
        export_pets(out_dir=tmp_path, dataset_name="sample")
