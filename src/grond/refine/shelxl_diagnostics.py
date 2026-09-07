"""SHELXL output integrity and pathology checks. An existing RES path is not evidence that refinement succeeded; inspect matching statistics, health, and provenance."""

from __future__ import annotations

import math
import re
from pathlib import Path

from grond.legacy.engine.bloch_engine import (
    parse_model,
    sha256,
)
from grond.legacy.engine.shelx_diagnostics import parse_shelx_lst_pathologies, shelx_pathology_safe_for_feedback

from .shelx_input import _iter_recommended_wght_terms, parse_shelx_wght_terms


def parse_lst(path: Path):
    out = {"lst": str(path), "status": "missing"}
    if not path.exists():
        return out
    out["status"] = "ok"
    out.update(parse_shelx_lst_pathologies(path))
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        terms = parse_shelx_wght_terms(line)
        if terms:
            out["active_wght_a"] = terms[0]
            out["active_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
            out["active_wght_terms"] = terms
            break
    m = re.findall(r"(\d+)\s+Reflections read, of which\s+(\d+)\s+rejected", text)
    if m:
        out["reflections_read"] = int(m[-1][0])
        out["reflections_rejected"] = int(m[-1][1])
    m = re.findall(r"(\d+)\s+Unique reflections, of which\s+(\d+)\s+suppressed", text)
    if m:
        out["unique"] = int(m[-1][0])
        out["suppressed"] = int(m[-1][1])
    m = re.findall(r"R\(int\) =\s*([0-9.]+)\s+R\(sigma\) =\s*([0-9.]+)", text)
    if m:
        out["Rint"] = float(m[-1][0])
        out["Rsigma"] = float(m[-1][1])
    m = re.findall(r"wR2 =\s*([0-9.]+), GooF = S =\s*([0-9.]+).* for all data", text)
    if m:
        out["wR2_all"] = float(m[-1][0])
        out["GooF"] = float(m[-1][1])
    m = re.findall(r"R1 =\s*([0-9.]+) for\s+(\d+) Fo > 4sig\(Fo\) and\s+([0-9.]+) for all\s+(\d+) data", text)
    if m:
        out["R1_gt"] = float(m[-1][0])
        out["N_gt"] = int(m[-1][1])
        out["R1_all"] = float(m[-1][2])
        out["N_all"] = int(m[-1][3])
    if "wR2_all" not in out or "R1_all" not in out:
        if "HAS WRONG FORMAT" in text:
            out["status"] = "bad_hkl_format"
        else:
            out["status"] = "no_regularization_stats"
    return out


def parse_recommended_wght(lst_path: Path) -> tuple[float, float] | None:
    """Return SHELXL's recommended WGHT values from an LST file, if present."""
    if not lst_path.exists():
        return None
    text = lst_path.read_text(errors="replace")
    matches = list(_iter_recommended_wght_terms(text))
    if not matches:
        return None
    terms = matches[-1]
    return terms[0], terms[1] if len(terms) >= 2 else 0.0


def shelx_res_health(
    res_path: Path,
    lst_path: Path | None = None,
    *,
    require_lst: bool = True,
) -> dict:
    """Return whether a SHELXL RES file is usable as the next model.

    SHELXL can fail in ways that still leave a path on disk.  In particular,
    interrupted or failed app-bundle runs on macOS have produced zero-byte
    ``.res`` files.  Those must not be fed back into the next Bloch cycle.
    """

    if not res_path.exists():
        return {"ok": False, "reason": "missing_res", "path": str(res_path)}
    try:
        size = res_path.stat().st_size
    except OSError as exc:
        return {"ok": False, "reason": "stat_failed", "path": str(res_path), "error": str(exc)}
    if size <= 0:
        return {"ok": False, "reason": "empty_res", "path": str(res_path), "size": int(size)}
    try:
        lines = res_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return {"ok": False, "reason": "read_failed", "path": str(res_path), "size": int(size), "error": str(exc)}
    has_cell = any(line.strip().upper().startswith("CELL ") for line in lines)
    if not has_cell:
        return {"ok": False, "reason": "missing_cell", "path": str(res_path), "size": int(size)}
    try:
        model = parse_model(res_path)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "invalid_structure_model",
            "path": str(res_path),
            "size": int(size),
            "error": f"{type(exc).__name__}: {exc}",
        }
    atom_count = len(model.get("atoms") or [])
    sfac_count = len(model.get("sfac") or [])
    if atom_count <= 0 or sfac_count <= 0:
        return {
            "ok": False,
            "reason": "missing_model_records",
            "path": str(res_path),
            "size": int(size),
            "atom_count": int(atom_count),
            "sfac_count": int(sfac_count),
        }
    if not require_lst and lst_path is None:
        # Structural-only validation is reserved for a caller that is
        # deliberately reusing an already selected input model without running
        # SHELXL.  Do not let an unrelated sibling LST alter that explicit
        # decision; generated endpoint models always require a conclusive LST.
        pathology = {
            "lst": "",
            "pathology_status": "not_applicable_structural_only",
            "npd_count": None,
            "split_count": None,
            "npd_atoms": [],
            "has_npd": None,
            "has_split_warnings": None,
        }
        pathology_safe = True
    else:
        resolved_lst = lst_path or res_path.with_suffix(".lst")
        pathology = parse_shelx_lst_pathologies(resolved_lst)
        pathology_safe = shelx_pathology_safe_for_feedback(pathology)
    return {
        "ok": True,
        "reason": "ok",
        "path": str(res_path),
        "size": int(size),
        "sha256": sha256(res_path),
        "atom_count": int(atom_count),
        "sfac_count": int(sfac_count),
        **pathology,
        "lst_required": bool(require_lst),
        "safe_for_feedback": pathology_safe,
    }


def shelxl_statistics_complete(stats: dict) -> bool:
    """Return whether SHELXL printed a complete, finite endpoint record."""

    try:
        r1_all = float(stats.get("R1_all"))
        wr2_all = float(stats.get("wR2_all"))
        n_all = int(stats.get("N_all", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        stats.get("returncode") == 0
        and math.isfinite(r1_all)
        and math.isfinite(wr2_all)
        and n_all > 0
    )


def classify_shelxl_iteration_rejection(stats: dict) -> tuple[str, str] | None:
    """Classify a structurally valid, zero-return-code SHELXL iteration.

    ``safe_for_feedback=False`` deliberately covers both an explicit NPD and
    missing/truncated final pathology diagnostics.  Keep those cases distinct:
    an NPD is an ADP pathology, while absent statistics or an incomplete final
    diagnostic footer are separate output-integrity failures.
    """

    health = stats.get("res_health") if isinstance(stats.get("res_health"), dict) else {}
    if bool(health.get("has_npd")) or int(health.get("npd_count", 0) or 0) > 0:
        return "adp_pathology", "adp_pathology"
    if not shelxl_statistics_complete(stats):
        parsed_status = str(stats.get("status", "missing"))
        return (
            "invalid_shelxl_statistics",
            "SHELXL output did not contain a complete finite R1/wR2/N statistics record "
            f"(parse status={parsed_status})",
        )
    if health.get("safe_for_feedback") is False:
        pathology_status = str(health.get("pathology_status", "missing"))
        return (
            "incomplete_shelxl_diagnostics",
            "SHELXL output did not contain a conclusive final NPD/split pathology summary "
            f"(pathology status={pathology_status})",
        )
    return None


def shelxl_iteration_safe(stats: dict) -> bool:
    """Return whether one completed SHELXL iteration is safe to retain."""

    health = stats.get("res_health")
    return bool(
        shelxl_statistics_complete(stats)
        and isinstance(health, dict)
        and health.get("ok") is True
        and int(health.get("atom_count", 0) or 0) > 0
        and int(health.get("sfac_count", 0) or 0) > 0
        and health.get("safe_for_feedback") is not False
    )
