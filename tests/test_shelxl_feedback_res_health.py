import sys
from pathlib import Path

import pytest

from grond.legacy.engine import native_row_regularize as nr
from grond.legacy.engine import shelxl_feedback_loop as fb


def valid_res_text() -> str:
    return "TITL ok\nCELL 0.02508 5.0 6.0 7.0 90 90 90\nSFAC C\nEND\n"


def test_feedback_endpoint_res_rejects_empty_direct_endpoint(tmp_path: Path):
    run = tmp_path / "cycle"
    direct = run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    direct.parent.mkdir(parents=True)
    direct.write_text("")

    with pytest.raises(FileNotFoundError, match="empty_res"):
        fb.endpoint_res(run)


def test_feedback_endpoint_res_skips_empty_and_uses_valid_fallback(tmp_path: Path):
    run = tmp_path / "cycle"
    direct = run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    fallback = run / "endpoint_lam_0p9_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    direct.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    direct.write_text("")
    fallback.write_text(valid_res_text())

    assert fb.endpoint_res(run) == fallback


def test_native_run_shelxl_marks_empty_res_invalid(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('')\n"
        "pathlib.Path(stem + '.res').write_text('')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()

    # run_shelxl calls [SHELXL, stem], so wrap the fake script with a small
    # executable shim that inserts the script path before the stem argument.
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=2)

    assert result["status"] == "invalid_res_output"
    assert result["invalid_res_reason"] == "empty_res"
    assert result["res_health"]["ok"] is False
    assert result["iterations"] == 2
    assert len(result["iteration_history"]) == 1


def test_native_run_shelxl_marks_npd_res_pathological(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('''\\n"
        "  -0.0010  -0.0010  -0.0010   O001   ** NON POSITIVE DEFINITE **\\n"
        " ** Warning:     0  atoms may be split and     1  atoms NPD **\\n"
        "''')\n"
        "pathlib.Path(stem + '.res').write_text('TITL npd\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C\\nEND\\n')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=3)

    assert result["status"] == "adp_pathology"
    assert result["invalid_res_reason"] == "adp_pathology"
    assert result["npd_count"] == 1
    assert result["npd_atoms"] == ["O001"]
    assert result["has_npd"] is True
    assert result["res_health"]["ok"] is True
    assert result["res_health"]["safe_for_feedback"] is False
    assert result["iterations"] == 3
    assert len(result["iteration_history"]) == 1


def test_native_run_shelxl_reports_split_warning_without_blocking_feedback(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('''\\n"
        " ** Warning:     2  atoms may be split and     0  atoms NPD **\\n"
        " wR2 = 0.1220, GooF = S = 1.010, Restrained GooF = 1.000 for all data\\n"
        " R1 = 0.0432 for 684 Fo > 4sig(Fo) and 0.1046 for all 1900 data\\n"
        "''')\n"
        "pathlib.Path(stem + '.res').write_text('TITL split\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C\\nEND\\n')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=3)

    assert result["status"] == "ok"
    assert result["split_count"] == 2
    assert result["npd_count"] == 0
    assert result["has_split_warnings"] is True
    assert result["res_health"]["safe_for_feedback"] is True
