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
