import hashlib
import json
from pathlib import Path

import pytest

from grond.legacy.engine import run_shelxl_baseline as baseline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    res = tmp_path / "input.res"
    hkl = tmp_path / "input.hkl"
    res.write_text(
        "TITL conventional baseline\n"
        "CELL 0.025079 5 6 7 90 90 90\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "WGHT 0.12 0.34\n"
        "EXTI 0.001\n"
        "FVAR 1\n"
        "C1 1 0.10 0.20 0.30 11.0 0.05\n"
        "L.S. 100\n"
        "HKLF 4\n"
        "END\n"
    )
    hkl.write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")
    return res, hkl


def _fake_shelxl(
    tmp_path: Path,
    *,
    npd: bool = False,
    returncode: int = 0,
    include_statistics: bool = True,
    include_pathology_footer: bool = True,
) -> Path:
    executable = tmp_path / ("fake_shelxl_npd" if npd else "fake_shelxl")
    pathology = (
        " -0.0010  -0.0010  -0.0010   C1   ** NON POSITIVE DEFINITE **\\n"
        " ** Warning:     0  atoms may be split and     1  atoms NPD **\\n"
        if npd
        else " ** Warning:     0  atoms may be split and     0  atoms NPD **\\n"
    )
    if not include_pathology_footer:
        pathology = ""
    statistics = (
        " 1 Unique reflections, of which 0 suppressed\\n"
        " wR2 = 0.1000, GooF = S = 1.000, Restrained GooF = 1.000 for all data\\n"
        " R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0800 for all 1 data\\n"
        if include_statistics
        else "SHELXL ended without final statistics\\n"
    )
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"returncode = {returncode}\n"
        "stem = sys.argv[1]\n"
        "if returncode == 0:\n"
        "    Path(stem + '.res').write_text(Path(stem + '.ins').read_text())\n"
        f"    Path(stem + '.lst').write_text({(pathology + statistics)!r})\n"
        "    Path(stem + '.fcf').write_text('fake FCF\\n')\n"
        "raise SystemExit(returncode)\n"
    )
    executable.chmod(0o755)
    return executable


def _run(res: Path, hkl: Path, out_dir: Path, executable: Path) -> tuple[int, dict]:
    returncode = baseline.main(
        [
            "--res",
            str(res),
            "--hkl",
            str(hkl),
            "--out-dir",
            str(out_dir),
            "--shelxl",
            str(executable),
        ]
    )
    return returncode, json.loads((out_dir / "summary.json").read_text())


def test_baseline_runner_honors_explicit_shelxl_and_records_provenance(tmp_path: Path) -> None:
    res, hkl = _write_inputs(tmp_path)
    executable = _fake_shelxl(tmp_path)
    out_dir = tmp_path / "baseline"

    returncode, summary = _run(res, hkl, out_dir, executable)

    assert returncode == 0
    assert summary["status"] == "ok"
    assert summary["safe_for_feedback"] is True
    assert summary["executable"]["resolved"] == str(executable.resolve())
    assert summary["inputs"]["res"]["sha256"] == _sha256(res)
    assert summary["inputs"]["hkl"]["sha256"] == _sha256(hkl)
    assert summary["outputs"]["res"]["sha256"] == _sha256(out_dir / "temp_a.res")
    assert summary["outputs"]["hkl"]["sha256"] == _sha256(out_dir / "temp_a.hkl")
    assert summary["shelxl"]["R1_all"] == pytest.approx(0.08)
    assert summary["shelxl"]["wR2_all"] == pytest.approx(0.10)
    assert summary["shelxl"]["res_health"]["safe_for_feedback"] is True
    refinement_input = (out_dir / "temp_a.ins").read_text()
    assert "EXTI 0.001" in refinement_input
    assert "WGHT 0.12 0.34" in refinement_input
    assert "L.S. 100" not in refinement_input
    assert "L.S. 1" in refinement_input or "CGLS 1" in refinement_input
    assert summary["shelxl"]["res_normalized_for_reopen"] is True


def test_baseline_runner_records_npd_as_unsafe_without_process_failure(tmp_path: Path) -> None:
    res, hkl = _write_inputs(tmp_path)
    executable = _fake_shelxl(tmp_path, npd=True)
    out_dir = tmp_path / "baseline_npd"

    returncode, summary = _run(res, hkl, out_dir, executable)

    assert returncode == 0
    assert summary["status"] == "adp_pathology"
    assert summary["safe_for_feedback"] is False
    assert summary["shelxl"]["returncode"] == 0
    assert summary["shelxl"]["npd_count"] == 1
    assert summary["shelxl"]["has_npd"] is True
    assert summary["shelxl"]["res_health"]["safe_for_feedback"] is False


@pytest.mark.parametrize(
    ("include_statistics", "include_pathology_footer", "expected_status"),
    [
        (False, False, "invalid_shelxl_statistics"),
        (True, False, "incomplete_shelxl_diagnostics"),
    ],
)
def test_baseline_runner_preserves_non_npd_shelxl_rejection_status(
    tmp_path: Path,
    include_statistics: bool,
    include_pathology_footer: bool,
    expected_status: str,
) -> None:
    res, hkl = _write_inputs(tmp_path)
    executable = _fake_shelxl(
        tmp_path,
        include_statistics=include_statistics,
        include_pathology_footer=include_pathology_footer,
    )
    out_dir = tmp_path / expected_status

    returncode, summary = _run(res, hkl, out_dir, executable)

    # The executable completed; the wrapper records an unsafe scientific
    # control in its summary while reserving nonzero exit codes for execution
    # and input failures (the same contract used for an explicit NPD).
    assert returncode == 0
    assert summary["status"] == expected_status
    assert summary["status"] == summary["shelxl"]["status"]
    assert summary["shelxl"]["has_npd"] is False
    assert summary["safe_for_feedback"] is False


def test_baseline_runner_returns_nonzero_for_shelxl_process_failure(tmp_path: Path) -> None:
    res, hkl = _write_inputs(tmp_path)
    executable = _fake_shelxl(tmp_path, returncode=7)
    out_dir = tmp_path / "baseline_failed"

    returncode, summary = _run(res, hkl, out_dir, executable)

    assert returncode != 0
    assert summary["status"] == "shelxl_failed"
    assert summary["safe_for_feedback"] is False
    assert summary["shelxl"]["returncode"] == 7


def test_baseline_runner_returns_nonzero_for_missing_input_or_tool(tmp_path: Path) -> None:
    res, hkl = _write_inputs(tmp_path)
    missing_input_out = tmp_path / "missing_input"
    returncode = baseline.main(
        [
            "--res",
            str(tmp_path / "missing.res"),
            "--hkl",
            str(hkl),
            "--out-dir",
            str(missing_input_out),
            "--shelxl",
            str(tmp_path / "missing_shelxl"),
        ]
    )
    missing_input = json.loads((missing_input_out / "summary.json").read_text())
    assert returncode != 0
    assert missing_input["status"] == "missing_input"
    assert missing_input["missing_inputs"] == ["res"]

    missing_tool_out = tmp_path / "missing_tool"
    returncode = baseline.main(
        [
            "--res",
            str(res),
            "--hkl",
            str(hkl),
            "--out-dir",
            str(missing_tool_out),
            "--shelxl",
            str(tmp_path / "missing_shelxl"),
        ]
    )
    missing_tool = json.loads((missing_tool_out / "summary.json").read_text())
    assert returncode != 0
    assert missing_tool["status"] == "missing_shelxl"
    assert missing_tool["executable"]["resolved"] == ""
