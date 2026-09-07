from __future__ import annotations

from pathlib import Path

import pytest

from grond.diffbloch_endpoint import materialize_diffbloch_res
from grond.legacy.engine.bloch_engine import parse_res


def _source_res(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "TITL endpoint",
                "CELL 0.02508 5 6 7 90 90 90",
                "LATT -1",
                "SFAC C 1 0 1 0 1 0 1 0",
                "SFAC H 1 0 1 0 1 0 1 0",
                "UNIT 1 1",
                "C1    1    0.100000  0.200000  0.300000  11.00000  0.02000  0.02100 =",
                " 0.02200 0.00100 0.00200 0.00300",
                "AFIX 13",
                "H1    2    0.120000  0.220000  0.320000  11.00000 -1.20000",
                "AFIX 0",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )


def _refined_cif(path: Path, *, u11: float = 0.030, a: float = 5.0) -> None:
    path.write_text(
        f"""data_refined
_cell_length_a {a}
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_adp_type
_atom_site_U_iso_or_equiv
C1 C 0.110000 0.210000 0.310000 1 Uani 0.03

loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
C1 {u11} 0.031 0.032 0.001 0.002 0.003
"""
    )


def test_diffbloch_cif_updates_heavy_atoms_and_preserves_shelx_hydrogens(tmp_path: Path) -> None:
    source = tmp_path / "source.res"
    cif = tmp_path / "refined.cif"
    output = tmp_path / "candidate.res"
    audit = tmp_path / "handoff.json"
    _source_res(source)
    _refined_cif(cif)
    payload = materialize_diffbloch_res(source, cif, output, audit_json=audit)
    model = parse_res(output)
    by_name = {atom["name"]: atom for atom in model["atoms"]}
    assert by_name["C1"]["x"] == pytest.approx(0.11)
    assert by_name["C1"]["u11"] == pytest.approx(0.03)
    assert by_name["H1"]["x"] == pytest.approx(0.12)
    output_text = output.read_text()
    assert "AFIX 13" in output_text
    assert "H1    2    0.120000  0.220000  0.320000 11.00000 -1.20000" in output_text
    assert payload["missing_hydrogens_preserved_from_source"] == ["H1"]
    assert payload["npd_atoms"] == []
    assert audit.is_file()


def test_diffbloch_cif_reports_npd_and_rejects_cell_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.res"
    cif = tmp_path / "refined.cif"
    output = tmp_path / "candidate.res"
    _source_res(source)
    _refined_cif(cif, u11=-0.03)
    payload = materialize_diffbloch_res(source, cif, output)
    assert payload["status"] == "rejected_npd"
    assert payload["npd_atoms"] == ["C1"]
    _refined_cif(cif, a=5.2)
    with pytest.raises(ValueError, match="cell mismatch"):
        materialize_diffbloch_res(source, cif, output)
