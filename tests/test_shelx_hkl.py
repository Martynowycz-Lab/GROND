from pathlib import Path

import pytest

from grond.engine.native_row_regularize import enforce_hkl_row_width
from grond.engine.shelx_hkl import (
    adaptive_observation_field,
    audit_fixed_width_precision,
    format_fixed_width_row,
    safe_fixed_width_factor,
    write_fixed_width_hkl,
)


@pytest.mark.parametrize("value", [0.0, 1.234567, -0.000012345, 12345.6789, 1.234567e11, -9.876543e12, 1.2345e-12])
def test_adaptive_fields_keep_sign_and_dynamic_range_in_eight_columns(value):
    field = adaptive_observation_field(value)
    assert len(field) == 8 and "." in field
    assert float(field) == pytest.approx(value, rel=5e-3, abs=0.0)


def test_adaptive_export_keeps_small_sigmas_beside_large_ones(tmp_path):
    rows = [(1, 2, 3, 0.0001234, 0.00001), (2, 3, 4, 1234.5678, 1.234567e11)]
    path = tmp_path / "adaptive.hkl"
    write_fixed_width_hkl(path, rows, adaptive=True)
    lines = path.read_text().splitlines()
    assert all(len(line) == 28 for line in lines)
    assert float(lines[0][20:28]) == pytest.approx(0.00001)
    report = audit_fixed_width_precision(path)
    assert report["status"] == "ok"
    assert report["rounding_bound_sigma_units_max"] <= 0.00500000001


def test_both_hkl_readers_keep_touching_fixed_columns_and_trailing_batches(tmp_path):
    from grond.engine.combine_native_nuisance_runs import read_hkl
    from grond.workflows.transform_metadata import read_hkl_intensity_rows

    rows = [(1, -2, 3, 1234.567, 1.234e11), (4, 5, -6, -123.456, 0.00001234)]
    path = tmp_path / "fixed_batch.hkl"
    path.write_text("".join(format_fixed_width_row(*row, adaptive=True).rstrip("\n") + "   2\n" for row in rows)
                    + format_fixed_width_row(0, 0, 0, 0.0, 0.0))
    assert read_hkl(path) == rows
    assert read_hkl_intensity_rows(path) == rows


@pytest.mark.parametrize("line", ["1 2 3 123456789012 1.0", "1.0 2.0 3.0 123456789012 1.0",
                                  "1.0e0 2.0e0 3.0e0 123456789012 1.0"])
def test_free_format_large_values_and_float_written_indices_are_unambiguous(line):
    from grond.engine.shelx_hkl import parse_hkl_observation_line

    assert parse_hkl_observation_line(line) == (1, 2, 3, 123456789012.0, 1.0)


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


def test_common_width_guard_reapplies_after_sigma_calibration():
    merged = [(1, 0, 0, 100.0, 150000.0)]
    unmerged = [(1, 0, 0, 50.0, 120000.0)]

    safe_merged, safe_unmerged, factor = enforce_hkl_row_width(
        merged, unmerged, "shelx_fixed"
    )

    assert factor == pytest.approx(0.6)
    assert safe_merged[0][3] == pytest.approx(60.0)
    assert safe_merged[0][4] == pytest.approx(90000.0)
    assert safe_unmerged[0][3] == pytest.approx(30.0)
    assert safe_unmerged[0][4] == pytest.approx(72000.0)
