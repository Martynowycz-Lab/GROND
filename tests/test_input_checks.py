"""A crystal path without reflection rows is refused when it is given (`grond prep --dataset`, `grond auto --hkl`,
`grond manual setup`), not written into a prep file that fails deep in the run."""

from __future__ import annotations

import pytest

from grond.cli_options.parser import build_parser
from grond.commands.preparation import cmd_prep, prep_from_args
from grond.io.config import parse_dataset_spec


def test_prep_refuses_a_crystal_without_reflection_rows(tmp_path):
    empty = tmp_path / "bridge_that_failed"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="has no INTEGRATE.HKL"):
        parse_dataset_spec(f"c1={empty},dmin=0.80")
    with pytest.raises(FileNotFoundError, match="no reflection file or directory"):
        parse_dataset_spec(f"c1={tmp_path / 'never_written'},dmin=0.80")
    (empty / "INTEGRATE.HKL").write_text("!END_OF_HEADER\n")
    assert parse_dataset_spec(f"c1={empty},dmin=0.80").hkl == str((empty / "INTEGRATE.HKL").resolve())


def test_grond_prep_stops_on_a_missing_crystal_directory(tmp_path):
    out = tmp_path / "prep.json"
    args = build_parser().parse_args(["prep", "--structure", str(tmp_path / "model.res"), "--dataset",
                                      f"c1={tmp_path / 'no_such_dir'},dmin=0.80", "--out", str(out)])
    with pytest.raises(SystemExit, match="grond prep: no reflection file or directory"):
        cmd_prep(args)
    assert not out.exists()


@pytest.mark.parametrize("flag", ["--hkl", "--xds-dir"])
def test_grond_auto_stops_on_a_missing_reflection_path(tmp_path, flag):
    empty = tmp_path / "empty"
    empty.mkdir()
    for path, message in ((tmp_path / "no_such_dir", "no reflection file or directory"),
                          (empty, "has no INTEGRATE.HKL")):
        args = build_parser().parse_args(["auto", flag, str(path), "--structure", str(tmp_path / "model.res"),
                                          "--out-dir", str(tmp_path / "out")])
        with pytest.raises(SystemExit, match=message):
            prep_from_args(args)
