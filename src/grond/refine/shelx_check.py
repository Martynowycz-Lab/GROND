"""Olex2/SHELXLE readiness check of one SHELXL endpoint (``grond check-shelx DIR``).

The check reads the endpoint's .res/.hkl/.lst/.fcf, then runs SHELXL once on a copy (the .res as the .ins,
exactly what Olex2 or SHELXLE do when a user opens and refines it) in a new directory, and reports:

* weighting: one active WGHT, and applied == SHELXL-suggested WGHT after the refinement;
* GooF (preferred 0.9-1.1), R1/wR2, max shift/esd;
* NPD atoms;
* BASF batch scales of an unmerged multi-crystal endpoint (one per extra batch, finite, well determined);
* file consistency: HKL fixed-width format and reflection count, the .res HklSrc name, ZERR with non-zero
  e.s.d.s for free cell parameters, stale Q-peaks in the input, LIST code of the .res and the .fcf, and the
  cell of the .fcf against the .res.

The endpoint files are never modified.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from grond.engine.shelx_diagnostics import parse_shelx_lst_pathologies
from grond.engine.shelx_hkl import audit_fixed_width_hklf4

from .shelx_input import Q_PEAK_RE, parse_basf_values, parse_shelx_wght_terms
from .shelxl_diagnostics import parse_lst, parse_recommended_wght
from .shelxl_iterations import WGHT_CONVERGE_TOL_A, WGHT_CONVERGE_TOL_B

GOOF_RANGE = (0.9, 1.1)
BASF_RANGE = (0.2, 5.0)
_SNAPSHOT_PREFIXES = ("shelxl_safe_iter_", "shelxl_attempt_", "shelxl_rejected_", "shelxl_iter_")


def find_endpoint(path: Path, stem: str = "") -> tuple[Path, str]:
    """(directory, stem) of the endpoint: a .res file, or the one .res with a matching .hkl in a directory."""

    path = Path(path).expanduser()
    if path.is_file():
        return path.parent, path.stem
    if not path.is_dir():
        raise FileNotFoundError(f"no SHELXL endpoint at {path}")
    if stem:
        if not (path / f"{stem}.res").is_file():
            raise FileNotFoundError(f"{path / (stem + '.res')} does not exist")
        return path, stem
    stems = sorted(
        res.stem
        for res in path.glob("*.res")
        if (path / f"{res.stem}.hkl").is_file() and not res.stem.startswith(_SNAPSHOT_PREFIXES)
    )
    if not stems:
        raise FileNotFoundError(f"{path} has no .res with a matching .hkl")
    if len(stems) > 1:
        raise ValueError(f"{path} has several endpoints ({', '.join(stems)}); choose one with --stem")
    return path, stems[0]


def _active_lines(text: str) -> list[str]:
    """Return the lines of a SHELX .res/.ins before its first HKLF or END record."""

    out = []
    for line in text.splitlines():
        head = line.split()[:1]
        if head and head[0].upper() in {"HKLF", "END"}:
            break
        out.append(line)
    return out


def static_res_checks(res_text: str, hkl_name: str) -> dict:
    """Checks of the .res itself: WGHT, ZERR, Q-peaks, LIST, MERG, BASF and the HKL source name."""

    active = _active_lines(res_text)
    wght = [parse_shelx_wght_terms(line) for line in active if line.split()[:1] and line.split()[0].upper() == "WGHT"]
    zerr = next((line.split() for line in active if line.split()[:1] and line.split()[0].upper() == "ZERR"), None)
    cell = next((line.split() for line in active if line.split()[:1] and line.split()[0].upper() == "CELL"), None)
    list_code = next((line.split()[1] for line in active if line.split()[:1] and line.split()[0].upper() == "LIST" and len(line.split()) > 1), "")
    merg = next((line.split()[1] for line in active if line.split()[:1] and line.split()[0].upper() == "MERG" and len(line.split()) > 1), "")
    basf = parse_basf_values(res_text)
    hkl_src = re.findall(r'<HklSrc\s+"%\./([^"]+)"', res_text)
    q_peaks = sum(1 for line in active if Q_PEAK_RE.match(line.strip()))
    zerr_esds = [float(value) for value in zerr[2:8]] if zerr and len(zerr) >= 8 else []
    return {
        "active_wght": wght,
        "single_active_wght": len(wght) == 1,
        "zerr": " ".join(zerr) if zerr else "",
        "zerr_esds": zerr_esds,
        "zerr_nonzero": bool(zerr_esds) and any(value > 0.0 for value in zerr_esds[:3]),
        "cell": [float(value) for value in cell[2:8]] if cell and len(cell) >= 8 else [],
        "list_code": list_code,
        "merg": merg,
        "basf": basf,
        "hkl_source": hkl_src[-1] if hkl_src else "",
        "hkl_source_matches": (hkl_src[-1] == hkl_name) if hkl_src else None,
        "q_peaks_in_input": q_peaks,
    }


def hkl_checks(hkl_path: Path) -> dict:
    """Fixed-width HKLF 4 audit, reflection count and the batch numbers of columns 29-32."""

    try:
        audit = audit_fixed_width_hklf4(hkl_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    batches: set[int] = set()
    for line in hkl_path.read_text(errors="replace").splitlines():
        if len(line) >= 32 and line[:12].split() != ["0", "0", "0"] and line[28:32].strip().lstrip("-").isdigit():
            batches.add(int(line[28:32]))
    return {"ok": True, "reflections": int(audit["reflection_count"]), "batches": sorted(batches)}


def fcf_checks(fcf_path: Path) -> dict:
    """Return whether a .fcf exists, its LIST code, cell and number of reflection lines."""

    if not fcf_path.is_file():
        return {"exists": False}
    text = fcf_path.read_text(errors="replace")
    code = re.findall(r"_shelx_refln_list_code\s+(\S+)", text)
    cell = [float(value) for value in re.findall(r"_cell_(?:length_[abc]|angle_(?:alpha|beta|gamma))\s+([-\d.]+)", text)]
    rows = sum(1 for line in text.splitlines() if re.match(r"^\s*-?\d+\s+-?\d+\s+-?\d+\s+[-\d.]", line))
    return {"exists": True, "list_code": code[-1] if code else "", "cell": cell, "reflections": rows}


def basf_from_lst(text: str) -> dict:
    """Return the BASF batch-scale values and e.s.d.s from a SHELXL .lst parameter table, in BASF order."""

    values: dict[int, tuple[float, float]] = {}
    for match in re.finditer(r"^\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+[-\d.]+\s+BASF\s+(\d+)", text, flags=re.M):
        values[int(match.group(3))] = (float(match.group(1)), float(match.group(2)))
    ordered = [values[key] for key in sorted(values)]
    return {"values": [value for value, _esd in ordered], "esds": [esd for _value, esd in ordered]}


def check_shelx_endpoint(
    path: Path,
    *,
    shelxl: str,
    stem: str = "",
    out_dir: Path | None = None,
    acta: bool = False,
) -> dict:
    """Run the readiness check; returns the report (``passed`` plus one entry per check)."""

    directory, stem = find_endpoint(Path(path), stem)
    res, hkl, lst, fcf = (directory / f"{stem}.{suffix}" for suffix in ("res", "hkl", "lst", "fcf"))
    exe = shelxl or shutil.which("shelxl") or ""
    if not exe or ("/" in exe and not Path(exe).exists()) or ("/" not in exe and shutil.which(exe) is None):
        raise FileNotFoundError(f"SHELXL executable not found: {exe or 'shelxl'}; pass --shelxl")
    res_text = res.read_text(errors="replace")
    static = static_res_checks(res_text, hkl.name)
    hkl_info = hkl_checks(hkl)
    fcf_before = fcf_checks(fcf)
    lst_before = parse_lst(lst) if lst.is_file() else {"status": "missing"}

    work = Path(out_dir) if out_dir else directory / f"check_shelx_{time.strftime('%Y%m%d_%H%M%S')}"
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"{work} exists and is not empty; the check never overwrites earlier results")
    work.mkdir(parents=True, exist_ok=True)
    ins_lines = res_text.splitlines()
    if acta and not any(line.split()[:1] and line.split()[0].upper() == "ACTA" for line in _active_lines(res_text)):
        at = next((i for i, line in enumerate(ins_lines) if line.strip().upper().startswith(("L.S.", "CGLS"))), 0)
        ins_lines.insert(at + 1, "ACTA")
    (work / f"{stem}.ins").write_text("\n".join(ins_lines) + "\n")
    shutil.copy2(hkl, work / f"{stem}.hkl")
    started = time.perf_counter()
    proc = subprocess.run([exe, stem], cwd=str(work), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / "shelxl_stdout.log").write_text(proc.stdout)
    run_lst = work / f"{stem}.lst"
    stats = parse_lst(run_lst) if run_lst.is_file() else {"status": "missing"}
    lst_text = run_lst.read_text(errors="replace") if run_lst.is_file() else ""
    recommended = parse_recommended_wght(run_lst) if run_lst.is_file() else None
    applied = [stats.get("active_wght_a"), stats.get("active_wght_b", 0.0)] if stats.get("active_wght_a") is not None else None
    pathology = parse_shelx_lst_pathologies(run_lst) if run_lst.is_file() else {}
    basf = basf_from_lst(lst_text)
    fcf_after = fcf_checks(work / f"{stem}.fcf")

    checks: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str, *, critical: bool = True) -> None:
        """Record one check result; a failed critical check fails the endpoint, a failed non-critical one only warns."""

        checks[name] = {"ok": bool(ok), "critical": bool(critical), "detail": detail}

    record("shelxl_ran", proc.returncode == 0 and str(stats.get("status")) == "ok", f"returncode {proc.returncode}, status {stats.get('status')}")
    record("single_active_wght", static["single_active_wght"], f"{len(static['active_wght'])} active WGHT record(s): {static['active_wght']}")
    if applied is not None and recommended is not None:
        da = abs(float(recommended[0]) - float(applied[0]))
        db = abs(float(recommended[1]) - float(applied[1]))
        record(
            "wght_converged",
            da < WGHT_CONVERGE_TOL_A and db < WGHT_CONVERGE_TOL_B,
            f"applied {applied[0]:.4f} {applied[1]:.4f}, suggested {recommended[0]:.4f} {recommended[1]:.4f} "
            f"(tolerance {WGHT_CONVERGE_TOL_A:g}, {WGHT_CONVERGE_TOL_B:g})",
        )
    else:
        record("wght_converged", False, "applied or suggested WGHT missing from the listing")
    goof = stats.get("GooF")
    record(
        "goof",
        goof is not None and GOOF_RANGE[0] <= float(goof) <= GOOF_RANGE[1],
        f"GooF {goof} (preferred {GOOF_RANGE[0]}-{GOOF_RANGE[1]})",
        critical=False,
    )
    shift = stats.get("max_shift_esd")
    record("max_shift_esd", shift is not None and abs(float(shift)) < 0.05, f"max shift/esd {shift}", critical=False)
    npd = int(pathology.get("npd_count", 0) or 0)
    record("npd", npd == 0 and str(pathology.get("pathology_status", "complete")) == "complete", f"{npd} NPD atom(s) {pathology.get('npd_atoms', [])}")
    batches = hkl_info.get("batches") or []
    if len(batches) > 1 or static["basf"]:
        problems = []
        if len(basf["values"]) != max(0, len(batches) - 1):
            problems.append(f"{len(basf['values'])} refined BASF for {len(batches)} batches")
        for index, (value, esd) in enumerate(zip(basf["values"], basf["esds"], strict=True), start=1):
            if not (math.isfinite(value) and BASF_RANGE[0] <= value <= BASF_RANGE[1]):
                problems.append(f"BASF {index} = {value:g} outside {BASF_RANGE}")
            if not (math.isfinite(esd) and esd > 0.0 and abs(value) > 3.0 * esd):
                problems.append(f"BASF {index} = {value:g} not determined (e.s.d. {esd:g})")
        record("basf", not problems, "; ".join(problems) or f"BASF {basf['values']} (e.s.d. {basf['esds']}), batches {batches}")
        record("merg_unmerged", static["merg"] == "0", f"MERG {static['merg'] or 'absent (SHELXL default 2 merges the batches)'}")
    record("hkl_format", bool(hkl_info.get("ok")), str(hkl_info.get("error") or f"{hkl_info.get('reflections')} reflections"))
    read = stats.get("reflections_read")
    record(
        "hkl_reflections_read",
        read is not None and hkl_info.get("reflections") is not None and int(read) == int(hkl_info["reflections"]),
        f"SHELXL read {read}, the HKL has {hkl_info.get('reflections')}",
    )
    record(
        "hkl_source",
        static["hkl_source_matches"] is not False,
        f"HklSrc {static['hkl_source'] or 'not recorded'} for {hkl.name}",
        critical=False,
    )
    free_zero = [
        name
        for name, value in zip(("a", "b", "c"), (static["zerr_esds"] or [0.0, 0.0, 0.0])[:3], strict=True)
        if value <= 0.0
    ]
    record(
        "zerr",
        bool(static["zerr"]) and not free_zero,
        f"{static['zerr'] or 'ZERR missing'}" + (f"; zero e.s.d. for {', '.join(free_zero)}" if free_zero else ""),
    )
    record("no_stale_q_peaks", static["q_peaks_in_input"] == 0, f"{static['q_peaks_in_input']} Q-peak(s) before HKLF in the .res", critical=False)
    record(
        "fcf_for_maps",
        bool(fcf_after.get("exists")) and str(fcf_after.get("list_code")) in {"6", "4"},
        f"LIST {static['list_code'] or 'absent'}; .fcf list code {fcf_after.get('list_code') or 'none'} "
        "(SHELXLE needs LIST 6, Olex2 reads 4 or 6)",
    )
    if fcf_before.get("exists"):
        cell_ok = (
            len(fcf_before.get("cell") or []) == 6
            and len(static["cell"]) == 6
            and all(abs(a - b) < 1e-3 for a, b in zip(fcf_before["cell"], static["cell"], strict=True))
        )
        record(
            "endpoint_fcf_consistent",
            cell_ok and str(fcf_before.get("list_code")) == str(static["list_code"] or fcf_before.get("list_code")),
            f"endpoint .fcf list code {fcf_before.get('list_code')} and cell {fcf_before.get('cell')} vs .res LIST "
            f"{static['list_code']} and cell {static['cell']}",
        )
    if acta:
        cif = work / f"{stem}.cif"
        record("acta_cif", cif.is_file(), f"{cif if cif.is_file() else 'no CIF written'}")
    passed = all(item["ok"] for item in checks.values() if item["critical"])
    report = {
        "schema": "grond.check-shelx.v1",
        "endpoint": {"directory": str(directory), "stem": stem, "res": str(res), "hkl": str(hkl)},
        "work_dir": str(work),
        "shelxl": exe,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "passed": passed,
        "checks": checks,
        "statistics": {
            key: stats.get(key)
            for key in ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all", "max_shift_esd", "active_wght_a", "active_wght_b")
        },
        "recommended_wght": list(recommended) if recommended else None,
        "basf": basf,
        "endpoint_listing_before_check": {key: lst_before.get(key) for key in ("R1_all", "wR2_all", "GooF")},
    }
    (work / "check_shelx.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    return report


def format_check_report(report: dict) -> str:
    """Format a check-shelx report as plain text: PASS/FAIL, statistics, WGHT and one line per check."""

    lines = [
        f"SHELXL endpoint check: {report['endpoint']['directory']}/{report['endpoint']['stem']}.res "
        f"({'PASS' if report['passed'] else 'FAIL'})",
    ]
    stats = report.get("statistics") or {}
    lines.append(
        f"  R1gt {stats.get('R1_gt')}  R1all {stats.get('R1_all')}  wR2 {stats.get('wR2_all')}  GooF {stats.get('GooF')}  "
        f"WGHT {stats.get('active_wght_a')} {stats.get('active_wght_b')} -> suggested {report.get('recommended_wght')}"
    )
    for name, item in report["checks"].items():
        mark = "ok  " if item["ok"] else ("FAIL" if item["critical"] else "warn")
        lines.append(f"  [{mark}] {name}: {item['detail']}")
    lines.append(f"  report: {report['work_dir']}/check_shelx.json")
    return "\n".join(lines)
