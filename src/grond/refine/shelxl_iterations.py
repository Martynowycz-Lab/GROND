"""Bounded SHELXL attempts, archive records, and recovery of the last safe model. Damping limits the proposed step; it does not excuse NPD atoms or repair bad observations."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

from .shelx_input import set_active_wght_lines
from .shelxl_diagnostics import (
    classify_shelxl_iteration_rejection,
    parse_lst,
    parse_recommended_wght,
    shelx_res_health,
    shelxl_iteration_safe,
)


def snapshot_safe_shelxl_iteration(work_dir: Path, stem: str, iteration: int, stats: dict) -> dict:
    """Snapshot matching safe SHELXL artifacts before a later LS1 pass can overwrite them."""

    if not shelxl_iteration_safe(stats):
        raise ValueError("cannot snapshot an unsafe SHELXL iteration")
    source_res = work_dir / f"{stem}.res"
    source_lst = work_dir / f"{stem}.lst"
    source_fcf = work_dir / f"{stem}.fcf"
    snapshot_res = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.res"
    snapshot_lst = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.lst"
    snapshot_fcf = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.fcf"
    shutil.copy2(source_res, snapshot_res)
    if source_lst.exists():
        shutil.copy2(source_lst, snapshot_lst)
    else:
        snapshot_lst.unlink(missing_ok=True)
    if source_fcf.exists():
        shutil.copy2(source_fcf, snapshot_fcf)
    else:
        snapshot_fcf.unlink(missing_ok=True)
    return {
        "iteration": int(iteration),
        "res": str(snapshot_res),
        "lst": str(snapshot_lst) if snapshot_lst.exists() else "",
        "fcf": str(snapshot_fcf) if snapshot_fcf.exists() else "",
        "stats": dict(stats),
    }


def select_last_safe_shelxl_iteration(
    work_dir: Path,
    stem: str,
    history: list[dict],
    last_safe: dict | None,
) -> dict:
    """Select the final safe iteration, restoring it after a later rejected pass."""

    attempted = dict(history[-1]) if history else {"returncode": None, "status": "not_run"}
    attempted_safe = shelxl_iteration_safe(attempted)
    restored = bool(last_safe and not attempted_safe)
    rejected_snapshot_res = ""
    rejected_snapshot_lst = ""
    rejected_snapshot_fcf = ""
    if history and restored:
        try:
            rejected_iteration = int(attempted.get("iteration", len(history)))
        except (TypeError, ValueError):
            rejected_iteration = len(history)
        canonical_res = work_dir / f"{stem}.res"
        canonical_lst = work_dir / f"{stem}.lst"
        canonical_fcf = work_dir / f"{stem}.fcf"
        rejected_res = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.res"
        rejected_lst = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.lst"
        rejected_fcf = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.fcf"
        if canonical_res.exists():
            shutil.copy2(canonical_res, rejected_res)
            rejected_snapshot_res = str(rejected_res)
        else:
            rejected_res.unlink(missing_ok=True)
        if canonical_lst.exists():
            shutil.copy2(canonical_lst, rejected_lst)
            rejected_snapshot_lst = str(rejected_lst)
        else:
            rejected_lst.unlink(missing_ok=True)
        attempted["lst"] = rejected_snapshot_lst
        if canonical_fcf.exists():
            shutil.copy2(canonical_fcf, rejected_fcf)
            rejected_snapshot_fcf = str(rejected_fcf)
        else:
            rejected_fcf.unlink(missing_ok=True)
        attempted["rejected_snapshot_res"] = rejected_snapshot_res
        attempted["rejected_snapshot_lst"] = rejected_snapshot_lst
        attempted["rejected_snapshot_fcf"] = rejected_snapshot_fcf
        rejected_health = attempted.get("res_health")
        if isinstance(rejected_health, dict):
            rejected_health = dict(rejected_health)
            rejected_health["path"] = rejected_snapshot_res
            rejected_health["lst"] = rejected_snapshot_lst
            attempted["res_health"] = rejected_health
        history[-1].update(attempted)
    final = attempted
    if restored:
        canonical_res = work_dir / f"{stem}.res"
        canonical_lst = work_dir / f"{stem}.lst"
        canonical_fcf = work_dir / f"{stem}.fcf"
        shutil.copy2(Path(str(last_safe["res"])), canonical_res)
        snapshot_lst = str(last_safe.get("lst", ""))
        if snapshot_lst and Path(snapshot_lst).exists():
            shutil.copy2(Path(snapshot_lst), canonical_lst)
        else:
            canonical_lst.unlink(missing_ok=True)
        snapshot_fcf = str(last_safe.get("fcf", ""))
        if snapshot_fcf and Path(snapshot_fcf).exists():
            shutil.copy2(Path(snapshot_fcf), canonical_fcf)
        else:
            canonical_fcf.unlink(missing_ok=True)
        final = dict(last_safe["stats"])
        final["status"] = "ok"
        final.pop("invalid_res_reason", None)
        final["res_health"] = shelx_res_health(canonical_res, canonical_lst)

    selected_iteration = int(last_safe["iteration"]) if last_safe and (attempted_safe or restored) else None
    final["selected_iteration"] = selected_iteration
    final["restored_last_safe_iteration"] = restored
    if not attempted_safe:
        final["rejected_iteration"] = attempted.get("iteration")
        final["rejected_status"] = str(attempted.get("status", ""))
        final["rejected_invalid_res_reason"] = str(attempted.get("invalid_res_reason", ""))
        rejected_health = attempted.get("res_health")
        final["rejected_res_health"] = dict(rejected_health) if isinstance(rejected_health, dict) else {}
        final["rejected_snapshot_res"] = rejected_snapshot_res
        final["rejected_snapshot_lst"] = rejected_snapshot_lst
        final["rejected_snapshot_fcf"] = rejected_snapshot_fcf
    return final


def write_active_wght(path: Path, a: float, b: float) -> dict:
    """Force one active WGHT line in a SHELX input or output file."""

    if not path.exists():
        return {"written": False, "reason": "missing_file"}
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        lines.append(line)
        if line.strip().upper().startswith("END"):
            break
    lines = set_active_wght_lines(lines, a, b)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return {"written": True, "active_wght_a": float(a), "active_wght_b": float(b)}


def promote_recommended_wght(ins_path: Path, recommended: tuple[float, float] | None) -> dict:
    """Move SHELXL's recommended WGHT into the active pre-END input block.

    SHELXL writes the suggested next WGHT after END in the RES file.  Copying RES
    to INS verbatim leaves that recommendation ignored on the next cycle, so the
    first active WGHT continues to control the weights.  GROND promotes the
    recommendation explicitly and discards the post-END tail in the working INS.
    """
    if recommended is None or not ins_path.exists():
        return {"promoted": False}
    a, b = recommended
    lines = ins_path.read_text(errors="replace").splitlines()
    active_lines: list[str] = []
    saw_end = False
    for line in lines:
        active_lines.append(line)
        if re.match(r"^\s*END\b", line, flags=re.IGNORECASE):
            saw_end = True
            break
    if not active_lines:
        return {"promoted": False}

    del saw_end
    active_lines = set_active_wght_lines(active_lines, a, b)
    ins_path.write_text("\n".join(active_lines).rstrip() + "\n")
    return {"promoted": True, "active_wght_a": float(a), "active_wght_b": float(b)}


def _active_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    try:
        a = float(stats["active_wght_a"])
        b = float(stats.get("active_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def recommended_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    try:
        a = float(stats["recommended_wght_a"])
        b = float(stats.get("recommended_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


DEFAULT_NPD_DAMP_RETRIES = (100.0, 300.0, 1000.0, 3000.0, 10000.0)


DEFAULT_NPD_DAMP_LIMSE = 15.0


def write_active_damp(path: Path, damp: float, limse: float = DEFAULT_NPD_DAMP_LIMSE) -> dict:
    """Write one active pre-END SHELXL DAMP instruction.

    This controls only the size of the least-squares model step.  It does not
    add restraints, permit NPD atoms, or alter the corrected reflection data.
    """

    if not path.exists():
        return {"written": False, "reason": "missing_file"}
    lines = path.read_text(errors="replace").splitlines()
    active_end = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith("END")),
        len(lines),
    )
    active_damp = [
        index
        for index, line in enumerate(lines[:active_end])
        if line.strip().upper().startswith("DAMP")
    ]
    damp_line = f"DAMP {float(damp):.8g} {float(limse):.8g}"
    if active_damp:
        lines[active_damp[0]] = damp_line
        for index in reversed(active_damp[1:]):
            del lines[index]
    else:
        insert_at = next(
            (
                index
                for index, line in enumerate(lines[:active_end])
                if line.strip().upper().startswith("FVAR")
            ),
            active_end,
        )
        lines.insert(insert_at, damp_line)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "written": True,
        "active_damp": float(damp),
        "active_damp_limse": float(limse),
    }


def _shelxl_attempt_label(value: float | None) -> str:
    if value is None:
        return "undamped"
    return f"damp_{float(value):g}".replace(".", "p").replace("-", "m")


def archive_shelxl_attempt(
    work_dir: Path,
    stem: str,
    iteration: int,
    attempt: int,
    damp: float | None,
) -> dict:
    """Preserve one SHELXL attempt before a bounded recovery overwrites it."""

    label = _shelxl_attempt_label(damp)
    prefix = f"shelxl_attempt_iter_{int(iteration):02d}_{int(attempt):02d}_{label}"
    archived: dict[str, object] = {
        "attempt": int(attempt),
        "label": label,
        "damp": None if damp is None else float(damp),
    }
    for suffix in ("res", "lst", "fcf"):
        source = work_dir / f"{stem}.{suffix}"
        target = work_dir / f"{prefix}.{suffix}"
        if source.exists():
            shutil.copy2(source, target)
            archived[suffix] = str(target)
        else:
            target.unlink(missing_ok=True)
            archived[suffix] = ""
    return archived


def archived_shelxl_attempt_record(stats: dict, archived: dict) -> dict:
    """Bind an attempt's health paths to its immutable archived artifacts."""

    record = {**dict(stats), **dict(archived)}
    health = record.get("res_health")
    if isinstance(health, dict):
        health = dict(health)
        health["path"] = str(archived.get("res", ""))
        health["lst"] = str(archived.get("lst", ""))
        record["res_health"] = health
    return record


def run_one_shelxl_attempt(
    work_dir: Path,
    stem: str,
    executable: str,
    *,
    iteration: int,
    log_name: str,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    """Run and classify one SHELXL attempt after clearing stale outputs."""

    res = work_dir / f"{stem}.res"
    lst = work_dir / f"{stem}.lst"
    fcf = work_dir / f"{stem}.fcf"
    res.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    fcf.unlink(missing_ok=True)
    proc = subprocess.run(
        [executable, stem],
        cwd=str(work_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (work_dir / log_name).write_text(proc.stdout)
    stats = {"iteration": int(iteration), "returncode": proc.returncode, **parse_lst(lst)}
    recommended_wght = parse_recommended_wght(lst)
    if recommended_wght is not None:
        stats["recommended_wght_a"] = recommended_wght[0]
        stats["recommended_wght_b"] = recommended_wght[1]
    health = shelx_res_health(res, lst)
    stats["res_health"] = health
    if proc.returncode != 0:
        stats["status"] = "shelxl_failed"
        stats["invalid_res_reason"] = f"SHELXL returned {proc.returncode}"
    elif not health["ok"]:
        stats["status"] = "invalid_res_output"
        stats["invalid_res_reason"] = health["reason"]
    else:
        rejection = classify_shelxl_iteration_rejection(stats)
        if rejection is not None:
            stats["status"], stats["invalid_res_reason"] = rejection
    return proc, stats


def recover_npd_shelxl_attempt(
    work_dir: Path,
    stem: str,
    executable: str,
    *,
    iteration: int,
    base_ins_text: str,
    initial_stats: dict,
    damp_values: tuple[float, ...] = DEFAULT_NPD_DAMP_RETRIES,
    limse: float = DEFAULT_NPD_DAMP_LIMSE,
) -> tuple[subprocess.CompletedProcess[str] | None, dict]:
    """Retry one rejected NPD step from its unchanged input with bounded DAMP.

    The first zero-NPD, complete SHELXL result is accepted.  Every attempt is
    archived and all failed trials remain visible in the returned provenance.
    """

    attempts: list[dict] = []
    initial_archive = archive_shelxl_attempt(work_dir, stem, iteration, 0, None)
    attempts.append(archived_shelxl_attempt_record(initial_stats, initial_archive))
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_stats = dict(initial_stats)
    ins = work_dir / f"{stem}.ins"
    for attempt_number, damp in enumerate(damp_values, start=1):
        ins.write_text(base_ins_text)
        damp_write = write_active_damp(ins, damp, limse)
        proc, stats = run_one_shelxl_attempt(
            work_dir,
            stem,
            executable,
            iteration=iteration,
            log_name=(
                f"shelxl_iter_{int(iteration):02d}_recovery_"
                f"{int(attempt_number):02d}_{_shelxl_attempt_label(damp)}.log"
            ),
        )
        stats["recovery_attempt"] = int(attempt_number)
        stats["active_damp"] = float(damp)
        stats["active_damp_limse"] = float(limse)
        stats["damp_write"] = damp_write
        archived = archive_shelxl_attempt(
            work_dir,
            stem,
            iteration,
            attempt_number,
            damp,
        )
        attempts.append(archived_shelxl_attempt_record(stats, archived))
        last_proc = proc
        last_stats = stats
        if shelxl_iteration_safe(stats):
            stats["status"] = "ok"
            stats["npd_recovery"] = {
                "trigger": "adp_pathology",
                "status": "accepted_first_safe_damped_step",
                "accepted_damp": float(damp),
                "limse": float(limse),
                "attempts": attempts,
            }
            return proc, stats
        if str(stats.get("status", "")) != "adp_pathology":
            break
    last_stats["npd_recovery"] = {
        "trigger": "adp_pathology",
        "status": "exhausted_without_safe_step",
        "accepted_damp": None,
        "limse": float(limse),
        "attempts": attempts,
    }
    return last_proc, last_stats
