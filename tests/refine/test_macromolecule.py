from __future__ import annotations

import json
from pathlib import Path

import gemmi
import pytest

from grond.macromolecule import (
    FREE_FLAG_LABEL,
    corrected_row_values,
    export_corrected_mtz,
    make_free_flags,
    propagate_free_flags,
    read_free_flags,
    render_protein_endpoint_command,
    run_protein_endpoint,
    write_table,
)


def test_free_flags_are_unique_hkl_and_propagate_to_signed_rows(tmp_path: Path):
    sg = gemmi.find_spacegroup_by_name("P 1")
    assert sg is not None
    cell = gemmi.UnitCell(10, 11, 12, 90, 90, 90)
    rows = [
        {"h": "1", "k": "0", "l": "0", "Icorr": "100", "SIGIcorr": "10", "d_A": "10.0"},
        {"h": "-1", "k": "0", "l": "0", "Icorr": "95", "SIGIcorr": "11", "d_A": "10.0"},
        {"h": "0", "k": "1", "l": "0", "Icorr": "40", "SIGIcorr": "8", "d_A": "11.0"},
    ]
    flag_rows, summary = make_free_flags(
        rows,
        sg=sg,
        cell=cell,
        fraction=0.49,
        min_free_count=1,
        seed=7,
        shells=1,
        merge_anomalous=True,
    )
    assert summary.unique_reflections == 2
    assert summary.free_reflections == 1

    flags_path = write_table(tmp_path / "free_flags.csv", flag_rows)
    flags = read_free_flags(flags_path, sg=sg, merge_anomalous=True)
    propagated = propagate_free_flags(rows, flags, sg=sg, merge_anomalous=True)
    assert propagated[0][FREE_FLAG_LABEL] == propagated[1][FREE_FLAG_LABEL]
    assert {int(row[FREE_FLAG_LABEL]) for row in propagated} == {0, 1}


def test_min_free_count_does_not_overwhelm_tiny_unique_sets():
    sg = gemmi.find_spacegroup_by_name("P 1")
    assert sg is not None
    rows = [{"h": str(i), "k": "0", "l": "0", "Icorr": "100", "SIGIcorr": "10"} for i in range(1, 21)]
    flag_rows, summary = make_free_flags(rows, sg=sg, fraction=0.05, min_free_count=500, seed=7, shells=1)
    assert summary.unique_reflections == 20
    assert summary.free_reflections == 1
    assert sum(int(row["free_flag"]) for row in flag_rows) == 1


def test_export_corrected_rows_to_mtz_preserves_flags_and_negative_intensity(tmp_path: Path):
    sg = gemmi.find_spacegroup_by_name("P 1")
    assert sg is not None
    cell = gemmi.UnitCell(10, 11, 12, 90, 90, 90)
    rows = [
        {"h": "1", "k": "0", "l": "0", "Icorr": "100", "SIGIcorr": "10", "d_A": "10.0"},
        {"h": "-1", "k": "0", "l": "0", "Icorr": "80", "SIGIcorr": "20", "d_A": "10.0"},
        {"h": "0", "k": "1", "l": "0", "Icorr": "-5", "SIGIcorr": "3", "d_A": "11.0"},
    ]
    rows_path = write_table(tmp_path / "rows.csv", rows)
    flag_rows, _ = make_free_flags(rows, sg=sg, cell=cell, fraction=0.49, min_free_count=1, seed=3, shells=1)
    flags_path = write_table(tmp_path / "free_flags.csv", flag_rows)

    summary = export_corrected_mtz(
        row_table=rows_path,
        free_flags=flags_path,
        out_mtz=tmp_path / "corrected.mtz",
        out_dir=tmp_path / "mtz_out",
        cell=cell,
        sg=sg,
        sigma_policy="chisq_overdispersion",
    )

    assert summary.merged_reflections == 2
    assert summary.corrected_row_count == 3
    mtz = gemmi.read_mtz_file(summary.mtz)
    labels = [column.label for column in mtz.columns]
    assert "IMEAN" in labels
    assert "SIGIMEAN" in labels
    assert FREE_FLAG_LABEL in labels
    assert "NOBS" in labels
    data = mtz.array
    assert any(data[:, labels.index("IMEAN")] < 0)
    saved_summary = json.loads((tmp_path / "mtz_out" / "export_mtz_summary.json").read_text())
    assert saved_summary["merged_reflections"] == 2


@pytest.mark.parametrize(
    ("d_A", "policy", "expected_I", "expected_sigma", "expected_ramp"),
    [
        (6.0, "low_resolution_damped", 100.0, 10.0, 0.0),
        (2.0, "low_resolution_damped", 50.0, 5.0, 1.0),
        (4.0, "low_resolution_damped", pytest.approx(70.710678, rel=1e-6), pytest.approx(7.071068, rel=1e-6), 0.5),
        (2.0, "off", 100.0, 10.0, 0.0),
        (2.0, "full", 50.0, 5.0, 1.0),
    ],
)
def test_corrected_row_q_policy_and_low_resolution_ramp(d_A, policy, expected_I, expected_sigma, expected_ramp):
    row = {
        "h": "1",
        "k": "0",
        "l": "0",
        "target_I": "100",
        "sigma_eff": "10",
        "q_dyn_over_born": "2",
        "dyn_group_scale": "1",
        "d_A": str(d_A),
    }
    intensity, sigma, ramp = corrected_row_values(row, q_policy=policy, q_off_d_A=5.0, q_full_d_A=3.0)
    assert intensity == expected_I
    assert sigma == expected_sigma
    assert ramp == expected_ramp


def test_protein_endpoint_dry_run_records_command(tmp_path: Path):
    command = render_protein_endpoint_command(
        "echo refine {model} {mtz} {cycles} {free_label}",
        model="start.pdb",
        mtz="corrected.mtz",
        out_dir=tmp_path / "endpoint",
        cycles=1,
        free_label=FREE_FLAG_LABEL,
    )
    assert command == ["echo", "refine", "start.pdb", "corrected.mtz", "1", FREE_FLAG_LABEL]

    payload = run_protein_endpoint(
        command=command,
        out_dir=tmp_path / "endpoint",
        output_model="{out_dir}/refined.pdb",
        model="start.pdb",
        mtz="corrected.mtz",
        cycles=1,
        dry_run=True,
    )
    assert payload["status"] == "dry_run"
    assert (tmp_path / "endpoint" / "command.json").exists()
    assert (tmp_path / "endpoint" / "summary.json").exists()
