"""CLI input validation regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
from grond.cli import (
    build_parser,
    check_sfac_contract,
)


def test_adp_stabilized_model_prep_targets_npd_atoms(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "\n".join(
            [
                "TITL test",
                "CELL 0.025079 7 8 9 90 90 90",
                "ZERR 1 0 0 0 0 0 0",
                "LATT -1",
                "SFAC C O H",
                "UNIT 1 1 1",
                "FVAR 1",
                "C005 1 0.100000 0.200000 0.300000 11.00000 0.00100 0.00200 =",
                " 0.00300 0.00000 0.00000 0.00000",
                "O001 2 0.400000 0.500000 0.600000 11.00000 0.02000",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    res.with_suffix(".lst").write_text(" 0.1 0.2 0.3 C005 ** NON POSITIVE DEFINITE **\n")
    args = argparse.Namespace(start_res=str(res), model_prep="adp-stabilized", set_ls_cycles=1)

    prepared, audit = cli.prepare_push_endpoint_model(args, tmp_path / "push")
    text = prepared.read_text()
    payload = json.loads(audit.read_text())

    assert "RIGU" in text
    assert "SIMU 0.04 0.08" in text
    assert "ISOR 0.02 0.04 C005" in text
    assert any(line.startswith("C005") and "0.00600" in line and "=" not in line for line in text.splitlines())
    assert payload["adp_stabilized_npd_atoms"] == ["C005"]
    assert payload["adp_stabilized_isotropic_fallback"][0]["label"] == "C005"


@pytest.mark.parametrize(
    ("mode", "anisotropic"),
    [
        ("free-h-isotropic-no-restraints", False),
        ("free-h-anis-no-restraints", True),
    ],
)
def test_free_h_model_prep_removes_afix_and_restraints(
    tmp_path: Path, mode: str, anisotropic: bool
):
    res = tmp_path / "model.res"
    res.write_text(
        "\n".join(
            [
                "TITL test",
                "CELL 0.025079 7 8 9 90 90 90",
                "LATT -1",
                "SFAC C H",
                "UNIT 1 1",
                "DFIX 1.09 C1 H1",
                "FVAR 1",
                "C1 1 0.100000 0.200000 0.300000 11.00000 0.02000 0.03000 =",
                " 0.04000 0.00100 0.00200 0.00300",
                "AFIX 13",
                "OH1 2 0.200000 0.200000 0.300000 11.00000 -1.20000",
                "AFIX 0",
                "L.S. 10",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    args = argparse.Namespace(start_res=str(res), model_prep=mode, set_ls_cycles=1)

    prepared, audit = cli.prepare_push_endpoint_model(args, tmp_path / mode)
    text = prepared.read_text()
    payload = json.loads(audit.read_text())

    assert "AFIX" not in text
    assert "DFIX" not in text
    assert "L.S. 1" in text
    assert payload["removed_afix"] == 2
    assert payload["removed_restraints"] == ["DFIX 1.09 C1 H1"]
    assert payload["converted_hydrogens"][0]["initial_adp_type"] == (
        "anisotropic" if anisotropic else "isotropic"
    )
    h_line = next(line for line in text.splitlines() if line.startswith("OH1"))
    assert float(h_line.split()[6]) > 0
    assert h_line.rstrip().endswith("=") is anisotropic


def test_sfac_check_rejects_element_only_shelx_factors(tmp_path: Path, capsys):
    model = tmp_path / "bad.res"
    model.write_text("TITL bad\nCELL 1 5 6 7 90 90 90\nSFAC C H N O\nHKLF 4\nEND\n")

    args = build_parser().parse_args(["sfac-check", "--structure", str(model)])
    assert args.func(args) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["ok"] is False
    assert payload["sfac_rows"][0]["kind"] == "element_only"
    assert "explicit electron scattering coefficient rows" in payload["messages"][1]["message"]


def test_sfac_check_rejects_cif_without_electron_factor_contract(tmp_path: Path):
    model = tmp_path / "ordinary.cif"
    model.write_text(
        "data_test\n"
        "_cell_length_a 10\n"
        "_cell_length_b 10\n"
        "_cell_length_c 10\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
    )

    payload = check_sfac_contract(model)

    assert payload["status"] == "error"
    assert payload["ok"] is False
    assert "electron-scattering model" in payload["messages"][0]["message"]


def test_sfac_check_audits_embedded_shelx_cif(tmp_path: Path):
    model = tmp_path / "embedded.cif"
    model.write_text(
        """data_test
_shelx_res_file
;
TITL embedded
CELL 0.025079 10 10 10 90 90 90
LATT -1
SFAC C 1 0 1 0 1 0 1 0
C1 1 0.1 0.2 0.3 11 0.05
HKLF 4
END
;
"""
    )

    payload = check_sfac_contract(model)

    assert payload["ok"] is True
    assert payload["cif_model_contract"] == "embedded_shelx"
    assert payload["sfac_rows"][0]["kind"] == "numeric_coefficients"


def test_sfac_check_accepts_numeric_rows_and_warns_on_placeholder_wavelength(tmp_path: Path):
    model = tmp_path / "ok.res"
    model.write_text("TITL ok\nCELL 1 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\nHKLF 4\nEND\n")

    payload = check_sfac_contract(model)

    assert payload["status"] == "warning"
    assert payload["ok"] is True
    assert payload["sfac_rows"][0]["kind"] == "numeric_coefficients"
    assert payload["cell"]["endpoint_wavelength_A"] == pytest.approx(0.025079)


def test_sfac_check_warns_acta_and_hklf3_endpoint_conversion(tmp_path: Path):
    model = tmp_path / "hklf3.res"
    model.write_text("TITL ok\nCELL 0.02508 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\nACTA\nHKLF 3\nEND\n")

    payload = check_sfac_contract(model)

    assert payload["ok"] is True
    assert payload["status"] == "warning"
    assert payload["endpoint_contract"]["hklf"] == "3"
    assert payload["endpoint_contract"]["hklf_status"] == "source_amplitudes_endpoint_forced_hklf4"
    messages = " ".join(item["message"] for item in payload["messages"])
    assert "ACTA" in messages
    assert "HKLF 3" in messages


def test_sfac_check_rejects_unsupported_shelx_endpoint_contracts(tmp_path: Path):
    cases = [
        ("HKLF 5", "HKLF 5"),
        ("TWIN 0 1 0 1 0 0 0 0 -1\nHKLF 4", "TWIN"),
        ("BASF 0.2\nHKLF 4", "BASF"),
        ("SWAT 0.1 1.0\nHKLF 4", "SWAT"),
        ("REM PLATON SQUEEZE applied\nHKLF 4", "SQUEEZE"),
    ]
    for body, expected in cases:
        model = tmp_path / f"{expected.lower()}.res"
        model.write_text(f"TITL bad\nCELL 0.02508 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\n{body}\nEND\n")

        payload = check_sfac_contract(model)

        assert payload["ok"] is False
        assert payload["status"] == "error"
        assert any(expected in item["message"] for item in payload["messages"])


def test_row_validate_reports_bad_portable_rows_without_traceback(tmp_path: Path, capsys):
    from grond.io.row_contract import write_template

    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows_path.write_text(rows_path.read_text().replace("42.1", "-42.1", 1))

    args = build_parser().parse_args(
        [
            "row-validate",
            "--rows",
            str(rows_path),
            "--metadata",
            str(metadata_path),
            "--write-xds-dir",
            str(tmp_path / "bridge"),
        ]
    )
    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "positive sigma_Iobs" in payload["error"]
    assert not (tmp_path / "bridge" / "INTEGRATE.HKL").exists()
