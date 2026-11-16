"""Bad user input stops with one line and exit code 1 instead of a Python traceback: a malformed --dataset
(grond prep, grond manual setup), a --prep file that is missing or is not a prep file, and a missing baseline HKL
(given with --baseline-hkl or named in the prep file)."""

from __future__ import annotations

import json

import pytest

from grond.cli_options.parser import build_parser
from grond.commands.preparation import cmd_prep, prep_from_args
from grond.io.config import PrepConfig, parse_dataset_spec, write_prep


@pytest.fixture
def crystal(tmp_path):
    xds = tmp_path / "c1"
    xds.mkdir()
    (xds / "INTEGRATE.HKL").write_text("!END_OF_HEADER\n")
    return xds


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("{xds}", "--dataset must look like name="),
        ("c1={xds},dmin=abc", "--dataset c1: dmin must be a number, got 'abc'"),
        ("c1={xds},min_peak=high", "--dataset c1: min_peak must be a number, got 'high'"),
        ("c1={xds},dmn=0.8", "unknown dataset option"),
        ("c1=", "has no path"),
        ("c1={xds},include_nonpositive=maybe", "include_nonpositive must be one of"),
    ],
)
def test_grond_prep_malformed_dataset_is_one_line(tmp_path, crystal, spec, message):
    out = tmp_path / "prep.json"
    args = build_parser().parse_args(["prep", "--structure", str(tmp_path / "model.res"),
                                      "--dataset", spec.format(xds=crystal), "--out", str(out)])
    with pytest.raises(SystemExit) as stop:
        cmd_prep(args)
    assert str(stop.value.code).startswith("grond prep: ")
    assert message in str(stop.value.code)
    assert not out.exists()


def test_grond_prep_valid_dataset_unchanged(tmp_path, crystal):
    out = tmp_path / "prep.json"
    args = build_parser().parse_args(["prep", "--structure", str(tmp_path / "model.res"),
                                      "--dataset", f"c1={crystal},dmin=0.80,min_peak=85", "--out", str(out)])
    assert cmd_prep(args) == 0
    ds = json.loads(out.read_text())["datasets"][0]
    assert (ds["min_d_spacing"], ds["min_peak"], ds["geometry_percentile"]) == (0.8, 85.0, 99.0)


def test_grond_manual_setup_malformed_dataset_is_one_line(tmp_path, crystal, monkeypatch):
    from grond.io import structure_contract
    from grond.manual.setup import manual_setup

    monkeypatch.setattr(structure_contract, "check_sfac_contract", lambda path: {"ok": True})
    model = tmp_path / "model.res"
    model.write_text("TITL x\nEND\n")
    args = build_parser().parse_args(["manual", "setup", "--out-dir", str(tmp_path / "run"), "--structure",
                                      str(model), "--dataset", f"c1={crystal},dmin=abc"])
    with pytest.raises(SystemExit, match="grond manual setup: --dataset c1: dmin must be a number"):
        manual_setup(args)


def _auto(tmp_path, *extra):
    return build_parser().parse_args(["auto", "--out-dir", str(tmp_path / "out"), *extra])


def test_auto_prep_file_missing_or_not_a_prep_file(tmp_path):
    with pytest.raises(SystemExit, match=r"grond auto: --prep \S+ does not exist"):
        prep_from_args(_auto(tmp_path, "--prep", str(tmp_path / "no_such_prep.json")))
    not_json = tmp_path / "notes.json"
    not_json.write_text("not json\n")
    with pytest.raises(SystemExit, match="is not a GROND prep file .JSONDecodeError"):
        prep_from_args(_auto(tmp_path, "--prep", str(not_json)))
    not_prep = tmp_path / "other.json"
    not_prep.write_text('{"foo": 1}\n')
    with pytest.raises(SystemExit, match="is not a GROND prep file .KeyError"):
        prep_from_args(_auto(tmp_path, "--prep", str(not_prep)))


def test_auto_missing_baseline_hkl(tmp_path, crystal):
    missing = str(tmp_path / "no_such.hkl")
    model = str(tmp_path / "model.res")
    with pytest.raises(SystemExit, match=r"grond auto: --baseline-hkl \S+ does not exist"):
        prep_from_args(_auto(tmp_path, "--hkl", str(crystal), "--structure", model, "--baseline-hkl", missing))
    prep_path = tmp_path / "prep.json"
    write_prep(prep_path, PrepConfig(structure=model, baseline_hkl=missing,
                                      datasets=[parse_dataset_spec(f"c1={crystal},dmin=0.80")]))
    with pytest.raises(SystemExit, match="the baseline HKL .* named in --prep .* does not exist"):
        prep_from_args(_auto(tmp_path, "--prep", str(prep_path)))
    with pytest.raises(SystemExit, match=r"--baseline-hkl \S+ does not exist"):
        prep_from_args(_auto(tmp_path, "--prep", str(prep_path), "--baseline-hkl", missing))
    baseline = tmp_path / "model.hkl"
    baseline.write_text("   0   0   0    0.00    0.00\n")
    prep = prep_from_args(_auto(tmp_path, "--prep", str(prep_path), "--baseline-hkl", str(baseline)))
    assert prep.baseline_hkl == str(baseline.resolve())
