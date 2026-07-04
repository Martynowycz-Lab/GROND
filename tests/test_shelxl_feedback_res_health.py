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
    assert result["res_health"]["ok"] is True
    assert result["res_health"]["safe_for_feedback"] is False
    assert result["iterations"] == 3
    assert len(result["iteration_history"]) == 1


def test_native_run_shelxl_accepts_damped_retry_after_npd(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "ins = pathlib.Path(stem + '.ins')\n"
        "text = ins.read_text() if ins.exists() else ''\n"
        "if 'DAMP' in text:\n"
        "    pathlib.Path(stem + '.lst').write_text('''\\n"
        " R1 = 0.1000 for 10 Fo > 4sig(Fo) and 0.2000 for all 20 data\\n"
        " wR2 = 0.3000, GooF = S = 1.000\\n"
        " ** Warning:     0  atoms may be split and     0  atoms NPD **\\n"
        "''')\n"
        "    pathlib.Path(stem + '.res').write_text('TITL damped\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C\\nEND\\n')\n"
        "else:\n"
        "    pathlib.Path(stem + '.lst').write_text('''\\n"
        "  -0.0010  -0.0010  -0.0010   O001   ** NON POSITIVE DEFINITE **\\n"
        " ** Warning:     0  atoms may be split and     1  atoms NPD **\\n"
        "''')\n"
        "    pathlib.Path(stem + '.res').write_text('TITL npd\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C\\nEND\\n')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()
    (work / "dyn_nuisance_corrected.ins").write_text(valid_res_text())
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=1, npd_damp_retries="0.25", npd_damp_limse=10.0)

    assert result["status"] == "accepted_damp_retry"
    assert result["damp_retry_accepted"] is True
    assert result["damp_retry_count"] == 1
    assert result["active_damp"] == pytest.approx(0.25)
    assert result["active_damp_limse"] == pytest.approx(10.0)
    assert result["res_health"]["safe_for_feedback"] is True
    assert (work / "shelxl_iter_1_damp_retry_1.log").exists()
    assert "DAMP      0.2500     10.0000" in (work / "dyn_nuisance_corrected.res").read_text()
