from pathlib import Path

import pytest

from grond.legacy.engine.shelx_hkl import (
    format_fixed_width_row,
    safe_fixed_width_factor,
    write_fixed_width_hkl,
)


def test_positive_and_negative_f82_limits_are_scaled_asymmetrically():
    rows = [
        (1, 0, 0, 216159.425428, 95374.978516),
        (2, 0, 0, 1.0, 1.0),
    ]

    factor = safe_fixed_width_factor(rows)

    assert factor == pytest.approx(0.416359359864)
    assert max(value * factor for row in rows for value in row[3:]) <= 90000.0

    negative_factor = safe_fixed_width_factor([(1, 0, 0, -12000.0, 1.0)])
    assert negative_factor == pytest.approx(0.75)
    assert -12000.0 * negative_factor == pytest.approx(-9000.0)


def test_fixed_width_formatter_rejects_positive_and_negative_overflow():
    assert len(format_fixed_width_row(1, 2, 3, 99999.99, 99999.99).rstrip("\n")) == 28
    assert len(format_fixed_width_row(1, 2, 3, -9999.99, 1.0).rstrip("\n")) == 28
    with pytest.raises(ValueError, match="field overflow"):
        format_fixed_width_row(1, 2, 3, 100000.0, 1.0)
    with pytest.raises(ValueError, match="field overflow"):
        format_fixed_width_row(1, 2, 3, -10000.0, 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fixed_width_formatter_rejects_nonfinite_observations(value):
    with pytest.raises(ValueError, match="must be finite"):
        format_fixed_width_row(1, 2, 3, value, 1.0)
    with pytest.raises(ValueError, match="must be finite"):
        format_fixed_width_row(1, 2, 3, 1.0, value)


def test_writer_produces_strict_28_character_hklf4_rows(tmp_path: Path):
    path = tmp_path / "strict.hkl"
    rows = [(1, 2, 3, 90000.0, 123.45), (-1, -2, -3, -9000.0, 12.34)]

    write_fixed_width_hkl(path, rows)

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert all(len(line) == 28 for line in lines)
