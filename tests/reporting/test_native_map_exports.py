"""Map CIFs must parse in an independent reader and retain beam metadata."""

import gemmi
import pytest

from grond.maps.native_dynamic import _cif_token, _model_wavelength_A


@pytest.mark.parametrize(
    "value", ["two words", "apostrophe's value", "both ' and \" quotes", "loop_", "_tag", "two\nlines"]
)
def test_cif_string_tokens_roundtrip(value):
    block = gemmi.cif.read_string("data_test\n_value " + _cif_token(value) + "\n").sole_block()
    assert gemmi.cif.as_string(block.find_value("_value")) == value


@pytest.mark.parametrize("value, expected", [("0.019687", 0.019687), ("0.019687(3)", 0.019687), ("?", 0.025079)])
def test_cif_wavelength_is_preserved(value, expected, tmp_path):
    path = tmp_path / "model.cif"
    path.write_text("data_model\n_diffrn_radiation_wavelength " + value + "\n")
    assert _model_wavelength_A(path) == pytest.approx(expected)


def test_missing_cif_wavelength_uses_declared_default(tmp_path):
    path = tmp_path / "model.cif"
    path.write_text("data_model\n_cell_length_a 10\n")
    assert _model_wavelength_A(path) == pytest.approx(0.025079)


def test_exported_ueq_uses_cartesian_tensor_in_skew_cell(tmp_path):
    import numpy as np

    from grond.maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package

    cell = gemmi.UnitCell(9, 10, 11, 70, 80, 65)
    transform = np.array(cell.orth.mat) @ np.diag(cell.reciprocal().parameters[:3])
    inverse = np.linalg.inv(transform)
    tensor = inverse @ (0.02 * np.eye(3)) @ inverse.T
    terms = [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[1, 2], tensor[0, 2], tensor[0, 1]]
    model = tmp_path / "model.res"
    model.write_text(
        "TITL skew cell\nCELL 0.02 9 10 11 70 80 65\nLATT -1\n"
        "SFAC C 1 0 0 0 0 0 0 0 0\nUNIT 1\nFVAR 1\n"
        "C1 1 0.2 0.3 0.4 11 " + " ".join(str(v) for v in terms) + "\nHKLF 4\nEND\n"
    )
    rows = tmp_path / "rows.csv"
    rows.write_text("h,k,l,target_I,sigma_eff,dyn_scaled\n1,0,0,1,0.1,1\n")
    result = write_native_dynamic_map_package(
        row_table=rows, model_path=model, out_dir=tmp_path / "maps",
        options=NativeDynamicMapOptions(grid_spacing_A=1),
    )
    block = gemmi.cif.read_file(result["outputs"]["dynamic_cif"]).sole_block()
    assert float(block.find_values("_atom_site_U_iso_or_equiv")[0]) == pytest.approx(0.02)
