from pathlib import Path

import pytest

from grond.legacy.engine import (
    combine_native_nuisance_runs as combined,
    native_row_regularize as native,
    scale_contract_replay as replay,
)


def _write_start_files(work: Path, stem: str) -> None:
    work.mkdir()
    (work / f"{stem}.ins").write_text(
        "TITL start\n"
        "CELL 0.025079 5 6 7 90 90 90\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "WGHT 0 0\n"
        "FVAR 1\n"
        "C1 1 0.10 0.20 0.30 11.0 0.05\n"
        "CGLS 1\n"
        "HKLF 4\n"
        "END\n"
    )
    (work / f"{stem}.hkl").write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")


def _safe_then_npd_shelxl(tmp_path: Path) -> Path:
    executable = tmp_path / "safe_then_npd_shelxl"
    executable.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

stem = sys.argv[1]
counter = Path(stem + ".call_count")
iteration = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(iteration))
safe = iteration == 1
title = "SAFE_ITERATION" if safe else "REJECTED_NPD_ITERATION"
r1_all = "0.1000" if safe else "0.0900"
wr2_all = "0.1400" if safe else "0.1300"
pathology = (
    " ** Warning:     0  atoms may be split and     0  atoms NPD **\\n"
    if safe
    else " -0.0010  -0.0010  -0.0010   C6   ** NON POSITIVE DEFINITE **\\n"
         " ** Warning:     0  atoms may be split and     1  atoms NPD **\\n"
)
Path(stem + ".lst").write_text(
    pathology
    + " 1 Unique reflections, of which 0 suppressed\\n"
    + f" wR2 = {wr2_all}, GooF = S = 0.987, Restrained GooF = 0.987 for all data\\n"
    + f" R1 = 0.0500 for 1 Fo > 4sig(Fo) and {r1_all} for all 1 data\\n"
)
Path(stem + ".res").write_text(
    f"TITL {title}\\n"
    "CELL 0.025079 5 6 7 90 90 90\\n"
    "SFAC C 1 0 1 0 1 0 1 0\\n"
    "WGHT 0 0\\n"
    "FVAR 1\\n"
    "C1 1 0.10 0.20 0.30 11.0 0.05\\n"
    "CGLS 1\\n"
    "HKLF 4\\n"
    "END\\n"
)
Path(stem + ".fcf").write_text(("SAFE" if safe else "REJECTED_NPD") + " FCF\\n")
print(f"completed fake SHELXL iteration {iteration}")
"""
    )
    executable.chmod(0o755)
    return executable


def _npd_until_damp_shelxl(tmp_path: Path) -> Path:
    executable = tmp_path / "npd_until_damp_shelxl"
    executable.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import re
import sys

stem = sys.argv[1]
ins = Path(stem + ".ins").read_text()
match = re.search(r"^DAMP\\s+([0-9.eE+-]+)", ins, flags=re.MULTILINE | re.IGNORECASE)
damp = float(match.group(1)) if match else 0.0
npd = 0 if damp >= 1000.0 else 1
Path(stem + ".lst").write_text(
    f" ** Warning:     0  atoms may be split and     {npd}  atoms NPD **\\n"
    " 1 Unique reflections, of which 0 suppressed\\n"
    " wR2 = 0.1400, GooF = S = 0.987, Restrained GooF = 0.987 for all data\\n"
    " R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.1000 for all 1 data\\n"
)
Path(stem + ".res").write_text(ins)
Path(stem + ".fcf").write_text(f"DAMP {damp:g}\\n")
"""
    )
    executable.chmod(0o755)
    return executable


def _assert_safe_iteration_restored(work: Path, stem: str, result: dict) -> None:
    assert result["status"] == "ok"
    assert result["selected_iteration"] == 1
    assert result["restored_last_safe_iteration"] is True
    assert result["rejected_iteration"] == 2
    assert result["rejected_status"] == "adp_pathology"
    assert result["R1_all"] == pytest.approx(0.1000)
    assert result["wR2_all"] == pytest.approx(0.1400)
    assert result["npd_count"] == 0
    assert result["has_npd"] is False
    assert result["res_health"]["safe_for_feedback"] is True
    assert len(result["iteration_history"]) == 2

    safe_res = work / "shelxl_safe_iter_01.res"
    safe_lst = work / "shelxl_safe_iter_01.lst"
    safe_fcf = work / "shelxl_safe_iter_01.fcf"
    rejected_res = work / "shelxl_rejected_iter_02.res"
    rejected_lst = work / "shelxl_rejected_iter_02.lst"
    rejected_fcf = work / "shelxl_rejected_iter_02.fcf"
    for path in (safe_res, safe_lst, safe_fcf, rejected_res, rejected_lst, rejected_fcf):
        assert path.exists(), path

    assert "SAFE_ITERATION" in (work / f"{stem}.res").read_text()
    assert "NON POSITIVE DEFINITE" not in (work / f"{stem}.lst").read_text()
    assert (work / f"{stem}.fcf").read_text() == "SAFE FCF\n"
    assert result["res_health"]["sha256"] == native.sha256(work / f"{stem}.res")
    assert "REJECTED_NPD_ITERATION" in rejected_res.read_text()
    assert "NON POSITIVE DEFINITE" in rejected_lst.read_text()
    assert rejected_fcf.read_text() == "REJECTED_NPD FCF\n"

    rejected_health = result["rejected_res_health"]
    assert rejected_health["path"] == str(rejected_res)
    assert rejected_health["lst"] == str(rejected_lst)
    assert rejected_health["has_npd"] is True
    rejected_record = result["iteration_history"][1]
    assert rejected_record["lst"] == str(rejected_lst)
    assert rejected_record["res_health"]["path"] == str(rejected_res)
    assert rejected_record["res_health"]["lst"] == str(rejected_lst)
    assert rejected_record["rejected_snapshot_fcf"] == str(rejected_fcf)


def test_native_shelxl_restores_last_safe_iteration_after_npd(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "native"
    stem = "dyn_nuisance_corrected"
    _write_start_files(work, stem)
    executable = _safe_then_npd_shelxl(tmp_path)
    monkeypatch.setattr(native, "SHELXL", str(executable))

    result = native.run_shelxl(work, stem, iterations=3, wght_mode="freeze-zero")

    _assert_safe_iteration_restored(work, stem, result)


def test_combined_shelxl_restores_last_safe_iteration_after_npd(tmp_path: Path) -> None:
    work = tmp_path / "combined"
    stem = "multi_dyn_corrected"
    _write_start_files(work, stem)
    executable = _safe_then_npd_shelxl(tmp_path)

    result = combined.run_shelxl(work, stem, str(executable), iterations=3, wght_mode="freeze-zero")

    _assert_safe_iteration_restored(work, stem, result)


def test_scale_replay_shelxl_restores_last_safe_iteration_after_npd(tmp_path: Path) -> None:
    work = tmp_path / "replay"
    stem = "dyn_scale_replay"
    _write_start_files(work, stem)
    executable = _safe_then_npd_shelxl(tmp_path)

    result = replay.run_shelxl_replay(work, stem, str(executable), iterations=3, wght_mode="freeze-zero")

    _assert_safe_iteration_restored(work, stem, result)


@pytest.mark.parametrize("implementation", ["combined", "replay"])
def test_shared_endpoint_paths_accept_first_safe_damp_recovery(
    tmp_path: Path,
    implementation: str,
) -> None:
    work = tmp_path / implementation
    stem = "endpoint"
    _write_start_files(work, stem)
    executable = _npd_until_damp_shelxl(tmp_path)

    if implementation == "combined":
        result = combined.run_shelxl(
            work,
            stem,
            str(executable),
            iterations=1,
            wght_mode="freeze-zero",
        )
    else:
        result = replay.run_shelxl_replay(
            work,
            stem,
            str(executable),
            iterations=1,
            wght_mode="freeze-zero",
        )

    assert result["status"] == "ok"
    assert result["npd_count"] == 0
    assert result["has_npd"] is False
    assert result["npd_recovery"]["status"] == "accepted_first_safe_damped_step"
    assert result["npd_recovery"]["accepted_damp"] == pytest.approx(1000.0)
    assert [attempt["npd_count"] for attempt in result["npd_recovery"]["attempts"]] == [1, 1, 1, 0]


def test_scale_replay_reports_missing_explicit_shelxl_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist" / "shelxl"

    result = replay.run_shelxl_replay(
        tmp_path,
        "dyn_scale_replay",
        str(missing),
        iterations=1,
        wght_mode="freeze-zero",
    )

    assert result == {
        "returncode": None,
        "status": "missing_shelxl",
        "executable": str(missing),
    }


@pytest.mark.parametrize(
    "require_tool",
    [
        lambda path: combined.require_combined_endpoint_shelxl(
            str(path),
            endpoint_rows_available=True,
        ),
        lambda path: replay.require_scale_replay_shelxl(str(path), ["freeze-zero"]),
    ],
)
def test_endpoint_wrappers_fail_closed_for_missing_or_non_executable_shelxl(
    tmp_path: Path,
    require_tool,
) -> None:
    missing = tmp_path / "missing" / "shelxl"
    non_executable = tmp_path / "private-ccp4" / "shelxl"
    non_executable.parent.mkdir()
    non_executable.write_text("#!/bin/sh\nexit 0\n")

    for unavailable in (missing, non_executable):
        with pytest.raises(SystemExit, match="missing or not executable"):
            require_tool(unavailable)

    non_executable.chmod(0o755)
    require_tool(non_executable)


def test_endpoint_wrappers_preserve_intentional_no_endpoint_modes(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "shelxl"

    combined.require_combined_endpoint_shelxl(
        str(missing),
        endpoint_rows_available=False,
    )
    replay.require_scale_replay_shelxl(str(missing), [])


def test_combined_endpoint_wrapper_preserves_path_fallback(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "path-bin" / "shelxl2019"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(
        combined.shutil,
        "which",
        lambda name: str(executable) if name == "shelxl2019" else None,
    )

    resolved = combined.require_combined_endpoint_shelxl(
        "",
        endpoint_rows_available=True,
    )

    assert resolved == str(executable)


def test_last_safe_restore_does_not_report_rejected_files_that_were_not_written(tmp_path: Path) -> None:
    work = tmp_path / "missing_rejected_outputs"
    stem = "endpoint"
    _write_start_files(work, stem)
    safe_stats = {
        "iteration": 1,
        "returncode": 0,
        "status": "ok",
        "R1_all": 0.10,
        "wR2_all": 0.14,
        "N_all": 1,
    }
    # The helper snapshots canonical .res output, as the production callers do.
    (work / f"{stem}.res").write_text((work / f"{stem}.ins").read_text())
    (work / f"{stem}.lst").write_text("     0  atoms may be split and     0  atoms NPD\n")
    safe_stats["res_health"] = native.shelx_res_health(work / f"{stem}.res")
    last_safe = native.snapshot_safe_shelxl_iteration(work, stem, 1, safe_stats)
    last_safe["stats"] = dict(safe_stats)

    # Model a later SHELXL process failure after callers cleared canonical
    # outputs and before SHELXL managed to create RES/LST/FCF artifacts.
    for suffix in ("res", "lst", "fcf"):
        (work / f"{stem}.{suffix}").unlink(missing_ok=True)
    failed = {
        "iteration": 2,
        "returncode": 1,
        "status": "shelxl_failed",
        "invalid_res_reason": "SHELXL returned 1",
        "res_health": native.shelx_res_health(work / f"{stem}.res"),
    }
    history = [dict(safe_stats), failed]

    result = native.select_last_safe_shelxl_iteration(work, stem, history, last_safe)

    assert result["restored_last_safe_iteration"] is True
    assert result["rejected_snapshot_res"] == ""
    assert result["rejected_snapshot_lst"] == ""
    assert result["rejected_snapshot_fcf"] == ""
    assert result["rejected_res_health"]["path"] == ""
    assert result["rejected_res_health"]["lst"] == ""
    assert history[-1]["lst"] == ""
    assert (work / f"{stem}.res").exists()
    assert result["res_health"]["sha256"] == native.sha256(work / f"{stem}.res")
