"""CLI hydrogen audit regression tests."""


import json

import pytest

import grond.cli as cli
from grond.cli import (
    build_parser,
)
from tests.cli._support import _write_hydrogen_audit_map, _write_hydrogen_audit_model


def test_hydrogen_omit_cli_writes_hash_bound_prospective_contract(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h.res"
    fofc = tmp_path / "fofc.map"
    out = tmp_path / "audit"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_model(never_h, include_h=False)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            f"native={fofc}",
            "--out-dir",
            str(out),
        ]
    )

    assert args.func(args) == 0
    contract = json.loads((out / "input_contract.json").read_text())
    assert contract["reference_hydrogen_count"] == 1
    assert contract["never_h_hydrogen_count"] == 0
    assert contract["model_cell_max_length_relative_mismatch"] == pytest.approx(0.0)
    assert contract["common_non_h_count"] == 3
    assert contract["common_non_h_rms_A"] == pytest.approx(0.0)
    assert contract["maps"][0]["sha256"] == cli.sha256_file(fofc)
    assert (out / "hydrogen_omit_peaks.csv").is_file()
    assert '"status": "ok"' in capsys.readouterr().out


def test_hydrogen_omit_cli_rejects_unaligned_common_nonh_origin(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h_shifted.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_model(never_h, include_h=False, coordinate_shift=0.25)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 2
    assert "not aligned" in capsys.readouterr().out


def test_hydrogen_omit_cli_rejects_h_contaminated_never_h_model(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(reference),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 2
    assert "contains hydrogen atoms" in capsys.readouterr().out


def test_hydrogen_omit_cli_rejects_reference_model_setting_mismatch(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h_wrong_cell.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_model(never_h, cell_a=12.0, include_h=False)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 2
    assert "cell is incompatible" in capsys.readouterr().out


def test_hydrogen_omit_cli_accepts_equivalent_half_cell_symmetry_syntax(tmp_path):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(
        reference,
        include_h=True,
        symm="-X-1/2,-Y,Z-1/2",
    )
    _write_hydrogen_audit_model(
        never_h,
        include_h=False,
        symm="-X+1/2,-Y,Z+1/2",
    )
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 0
