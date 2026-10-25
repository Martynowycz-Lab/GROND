"""Bounded SHELXL attempts, archive records, and recovery of the last safe model.

Damping limits the proposed step; it does not excuse NPD atoms or repair bad observations.

Two NPD recoveries exist. ``isor`` (the AUTO default) restrains only the atoms SHELXL reported non-positive
definite with ``ISOR 0.01 0.02 <atom>`` and repeats the same call from its unchanged input.  ``damp`` is the
legacy escalation ladder DAMP 100..10000.  DAMP does not move the fixed point of the lagged Bloch/SHELXL map;
it only slows the step, so carrying a rescue DAMP forward turns every later call into a damped step (a rescue
``DAMP 300 15`` can then reach the publication endpoint).  An ISOR restraint is recorded and surfaced as a
warning because it can also hide a real model error such as the wrong hand.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

from .shelx_input import set_active_wght_lines
from .shelxl_diagnostics import (
    classify_shelxl_iteration_rejection,
    npd_atoms_from_lst_text,
    parse_lst,
    parse_recommended_wght,
    parse_suggested_wght_res,
    shelx_res_health,
    shelxl_iteration_safe,
)


def snapshot_safe_shelxl_iteration(work_dir: Path, stem: str, iteration: int, stats: dict) -> dict:
    """Snapshot matching safe SHELXL artifacts before a later LS1 pass can overwrite them.

    The call's input ``stem.ins`` (after an accepted NPD recovery, the input that recovery ran) is kept as
    ``shelxl_safe_iter_NN.ins``, so a restore after a rejected later call can put it back next to its .res and .lst.
    """

    if not shelxl_iteration_safe(stats):
        raise ValueError("cannot snapshot an unsafe SHELXL iteration")
    source_res = work_dir / f"{stem}.res"
    source_lst = work_dir / f"{stem}.lst"
    source_fcf = work_dir / f"{stem}.fcf"
    source_ins = work_dir / f"{stem}.ins"
    snapshot_res = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.res"
    snapshot_lst = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.lst"
    snapshot_fcf = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.fcf"
    snapshot_ins = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.ins"
    shutil.copy2(source_res, snapshot_res)
    if source_ins.exists():
        shutil.copy2(source_ins, snapshot_ins)
    else:
        snapshot_ins.unlink(missing_ok=True)
    if source_lst.exists():
        shutil.copy2(source_lst, snapshot_lst)
    else:
        snapshot_lst.unlink(missing_ok=True)
    if source_fcf.exists():
        shutil.copy2(source_fcf, snapshot_fcf)
    else:
        snapshot_fcf.unlink(missing_ok=True)
    snapshot = {
        "iteration": int(iteration),
        "res": str(snapshot_res),
        "lst": str(snapshot_lst) if snapshot_lst.exists() else "",
        "fcf": str(snapshot_fcf) if snapshot_fcf.exists() else "",
        "ins": str(snapshot_ins) if snapshot_ins.exists() else "",
        "stats": dict(stats),
    }
    # An ACTA endpoint's CIF belongs to this call's .res; keep it with the snapshot so a restore brings it back.
    source_cif = work_dir / f"{stem}.cif"
    if source_cif.exists():
        snapshot_cif = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.cif"
        shutil.copy2(source_cif, snapshot_cif)
        snapshot["cif"] = str(snapshot_cif)
    return snapshot


def select_last_safe_shelxl_iteration(
    work_dir: Path,
    stem: str,
    history: list[dict],
    last_safe: dict | None,
) -> dict:
    """Select the final safe iteration, restoring it after a later rejected pass.

    A restore puts back the kept call's .res, .lst, .fcf, .cif and input .ins, so the files named ``stem.*`` belong
    to one call; the rejected call's files, its input included, are kept as ``shelxl_rejected_iter_NN.*``.
    """

    attempted = dict(history[-1]) if history else {"returncode": None, "status": "not_run"}
    attempted_safe = shelxl_iteration_safe(attempted)
    restored = bool(last_safe and not attempted_safe)
    rejected_snapshot_res = ""
    rejected_snapshot_lst = ""
    rejected_snapshot_fcf = ""
    rejected_snapshot_ins = ""
    rejected_snapshot_cif = ""
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
        # the rejected call's input (after NPD recovery attempts, the last one it ran)
        canonical_ins = work_dir / f"{stem}.ins"
        rejected_ins = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.ins"
        if canonical_ins.exists():
            shutil.copy2(canonical_ins, rejected_ins)
            rejected_snapshot_ins = str(rejected_ins)
        else:
            rejected_ins.unlink(missing_ok=True)
        canonical_cif = work_dir / f"{stem}.cif"
        rejected_cif = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.cif"
        if canonical_cif.exists():
            shutil.copy2(canonical_cif, rejected_cif)
            rejected_snapshot_cif = str(rejected_cif)
        else:
            rejected_cif.unlink(missing_ok=True)
        attempted["rejected_snapshot_res"] = rejected_snapshot_res
        attempted["rejected_snapshot_lst"] = rejected_snapshot_lst
        attempted["rejected_snapshot_fcf"] = rejected_snapshot_fcf
        attempted["rejected_snapshot_ins"] = rejected_snapshot_ins
        attempted["rejected_snapshot_cif"] = rejected_snapshot_cif
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
        canonical_cif = work_dir / f"{stem}.cif"
        snapshot_cif = str(last_safe.get("cif", ""))
        if snapshot_cif and Path(snapshot_cif).exists():
            shutil.copy2(Path(snapshot_cif), canonical_cif)
        elif canonical_cif.exists():
            # A CIF of the rejected call must not stand next to the restored .res.
            if rejected_snapshot_cif:
                canonical_cif.unlink()
            else:
                rejected_iteration = int(attempted.get("iteration") or len(history))
                rejected_cif = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.cif"
                shutil.move(str(canonical_cif), str(rejected_cif))
                rejected_snapshot_cif = str(rejected_cif)
        # Nor its input: the kept call's .ins goes back next to its .res and .lst.
        snapshot_ins = str(last_safe.get("ins", ""))
        restored_ins = bool(snapshot_ins and Path(snapshot_ins).exists())
        if restored_ins:
            shutil.copy2(Path(snapshot_ins), work_dir / f"{stem}.ins")
        final = dict(last_safe["stats"])
        final["status"] = "ok"
        final.pop("invalid_res_reason", None)
        final["res_health"] = shelx_res_health(canonical_res, canonical_lst)
        final["restored_last_safe_ins"] = restored_ins

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
        final["rejected_snapshot_ins"] = rejected_snapshot_ins
        final["rejected_snapshot_cif"] = rejected_snapshot_cif
    return final


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
    for line in lines:
        active_lines.append(line)
        if re.match(r"^\s*END\b", line, flags=re.IGNORECASE):
            break
    if not active_lines:
        return {"promoted": False}

    active_lines = set_active_wght_lines(active_lines, a, b)
    ins_path.write_text("\n".join(active_lines).rstrip() + "\n")
    return {"promoted": True, "active_wght_a": float(a), "active_wght_b": float(b)}


def _active_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    """Return the active WGHT (a, b) from parsed .lst statistics, or None when missing or not finite."""

    try:
        a = float(stats["active_wght_a"])
        b = float(stats.get("active_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def recommended_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    """Return SHELXL's recommended WGHT (a, b) from parsed .lst statistics, or None when missing or not finite."""

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
    """Return a file-name-safe label for a SHELXL attempt: ``undamped`` or ``damp_<value>``
    with ``.`` as ``p`` and ``-`` as ``m``.
    """

    if value is None:
        return "undamped"
    return f"damp_{float(value):g}".replace(".", "p").replace("-", "m")


def archive_shelxl_attempt(
    work_dir: Path,
    stem: str,
    iteration: int,
    attempt: int,
    damp: float | None,
    *,
    label: str | None = None,
) -> dict:
    """Preserve one SHELXL attempt before a bounded recovery overwrites it."""

    label = label or _shelxl_attempt_label(damp)
    prefix = f"shelxl_attempt_iter_{int(iteration):02d}_{int(attempt):02d}_{label}"
    archived: dict[str, object] = {
        "attempt": int(attempt),
        "label": label,
        "damp": None if damp is None else float(damp),
    }
    for suffix in ("ins", "res", "lst", "fcf", "cif"):
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
    (work_dir / f"{stem}.cif").unlink(missing_ok=True)
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
    if recommended_wght is None and res.exists():
        # SHELXL also writes its suggested next WGHT after END in the RES file.
        recommended_wght = parse_suggested_wght_res(res.read_text(errors="replace"))
        if recommended_wght is not None:
            stats["recommended_wght_source"] = "res_after_end"
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


NPD_RECOVERY_METHODS = ("isor", "damp")


ISOR_RESTRAINT_S = 0.01


ISOR_RESTRAINT_ST = 0.02
# One tighter tier for an atom that is still NPD under the standard ISOR (for example an atom that is NPD under
# ISOR 0.01 0.02 and positive definite under 0.005 0.01 at unchanged R1 and GooF).
ISOR_TIGHT_S = 0.005
ISOR_TIGHT_ST = 0.01


def _active_input_lines(text: str) -> list[str]:
    """Return the lines of a SHELX input up to and including its END line."""

    lines = []
    for line in text.splitlines():
        lines.append(line)
        if line.strip().upper().startswith("END"):
            break
    return lines


def isor_restrained_atoms(text: str) -> list[str]:
    """Atom names already named by an active (pre-END) ISOR restraint."""

    atoms: list[str] = []
    for line in _active_input_lines(text):
        parts = line.split()
        if not parts or parts[0].upper() != "ISOR":
            continue
        names = parts[1:]
        for _ in range(2):
            if names and _is_number(names[0]):
                names = names[1:]
        for name in names:
            if name.upper() not in {a.upper() for a in atoms}:
                atoms.append(name)
    return atoms


def active_damp_present(text: str) -> bool:
    """Return whether a SHELX input or RES carries an active (pre-END) DAMP instruction."""

    return any(line.split()[:1] and line.split()[0].upper() == "DAMP" for line in _active_input_lines(text))


def _is_number(token: str) -> bool:
    """Return whether a token parses as a float."""

    try:
        float(token)
    except ValueError:
        return False
    return True


def write_isor_restraints(
    path: Path,
    atoms: list[str],
    *,
    s: float = ISOR_RESTRAINT_S,
    st: float = ISOR_RESTRAINT_ST,
) -> dict:
    """Insert one ``ISOR s st atom`` line per atom before FVAR (by default ``ISOR 0.01 0.02 <atom>``)."""

    if not path.exists():
        return {"written": False, "reason": "missing_file"}
    lines = path.read_text(errors="replace").splitlines()
    active_end = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith("END")),
        len(lines),
    )
    insert_at = next(
        (index for index, line in enumerate(lines[:active_end]) if line.strip().upper().startswith("FVAR")),
        active_end,
    )
    restraint_lines = [f"ISOR {float(s):g} {float(st):g} {atom}" for atom in atoms]
    lines[insert_at:insert_at] = restraint_lines
    path.write_text("\n".join(lines).rstrip() + "\n")
    return {"written": True, "isor_atoms": list(atoms), "lines": restraint_lines}


def _attempt_npd_atoms(work_dir: Path, stem: str, stats: dict) -> list[str]:
    """Return the non-positive-definite atoms of an attempt, from the .res health check and its .lst."""

    health = stats.get("res_health") if isinstance(stats.get("res_health"), dict) else {}
    atoms = [str(atom) for atom in (health.get("npd_atoms") or [])]
    lst = work_dir / f"{stem}.lst"
    if lst.exists():
        for atom in npd_atoms_from_lst_text(lst.read_text(errors="replace")):
            if atom.upper() not in {a.upper() for a in atoms}:
                atoms.append(atom)
    return atoms


def recover_npd_with_isor(
    work_dir: Path,
    stem: str,
    executable: str,
    *,
    iteration: int,
    base_ins_text: str,
    initial_stats: dict,
    max_retries: int = 10,
) -> tuple[subprocess.CompletedProcess[str] | None, dict]:
    """Retry one NPD step from its unchanged input with ISOR on the offending atoms only.

    Every newly NPD atom gets ``ISOR 0.01 0.02 <atom>`` and the same call is repeated.  An atom that is
    still NPD under that restraint gets one tighter tier, ``ISOR 0.005 0.01``.  If SHELXL still reports NPD only for atoms at the tight tier, the recovery is
    exhausted and the step stays unsafe (the macro-cycle loop then keeps the previous clean model).  Every
    attempt is archived.
    """

    attempts: list[dict] = []
    initial_archive = archive_shelxl_attempt(work_dir, stem, iteration, 0, None)
    attempts.append(archived_shelxl_attempt_record(initial_stats, initial_archive))
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_stats = dict(initial_stats)
    restrained = {atom.upper() for atom in isor_restrained_atoms(base_ins_text)}
    added: list[str] = []
    tightened: list[str] = []
    npd_atoms = _attempt_npd_atoms(work_dir, stem, initial_stats)
    ins = work_dir / f"{stem}.ins"
    reason = ""
    for attempt_number in range(1, int(max_retries) + 1):
        new_atoms = [atom for atom in npd_atoms if atom.upper() not in restrained]
        if new_atoms:
            added.extend(new_atoms)
            restrained.update(atom.upper() for atom in new_atoms)
        else:
            # Every NPD atom is already restrained: tighten those once, then give up.
            loosely = [atom for atom in npd_atoms if atom.upper() not in {a.upper() for a in tightened}]
            if not loosely:
                reason = (
                    "NPD atoms are already restrained with the tight ISOR" if npd_atoms else "SHELXL named no NPD atom"
                )
                break
            tightened.extend(loosely)
        ins.write_text(base_ins_text)
        tight_set = {a.upper() for a in tightened}
        isor_write = write_isor_restraints(ins, [a for a in added if a.upper() not in tight_set])
        if tightened:
            isor_write = {
                "standard": isor_write,
                "tight": write_isor_restraints(ins, tightened, s=ISOR_TIGHT_S, st=ISOR_TIGHT_ST),
            }
        proc, stats = run_one_shelxl_attempt(
            work_dir,
            stem,
            executable,
            iteration=iteration,
            log_name=f"shelxl_iter_{int(iteration):02d}_recovery_{int(attempt_number):02d}_isor.log",
        )
        stats["recovery_attempt"] = int(attempt_number)
        stats["isor_atoms_added"] = list(added)
        stats["isor_atoms_tightened"] = list(tightened)
        stats["isor_write"] = isor_write
        npd_atoms = _attempt_npd_atoms(work_dir, stem, stats)
        archived = archive_shelxl_attempt(work_dir, stem, iteration, attempt_number, None, label="isor")
        attempts.append(archived_shelxl_attempt_record(stats, archived))
        last_proc = proc
        last_stats = stats
        if shelxl_iteration_safe(stats):
            stats["status"] = "ok"
            stats["npd_recovery"] = {
                "trigger": "adp_pathology",
                "method": "isor",
                "status": "accepted_isor_restrained_step",
                "isor_atoms": list(added),
                "isor_atoms_tightened": list(tightened),
                "tight_isor": [ISOR_TIGHT_S, ISOR_TIGHT_ST],
                "warning": "ISOR was added for NPD atoms; check the model (hand, disorder) before publication",
                "attempts": attempts,
            }
            return proc, stats
        if str(stats.get("status", "")) != "adp_pathology":
            reason = f"retry ended with {stats.get('status')}"
            break
    else:
        reason = f"no safe step after {int(max_retries)} ISOR retries"
    last_stats["npd_recovery"] = {
        "trigger": "adp_pathology",
        "method": "isor",
        "status": "exhausted_without_safe_step",
        "reason": reason,
        "isor_atoms": list(added),
        "isor_atoms_tightened": list(tightened),
        "attempts": attempts,
    }
    return last_proc, last_stats


def recover_npd_attempt(
    work_dir: Path,
    stem: str,
    executable: str,
    *,
    iteration: int,
    base_ins_text: str,
    initial_stats: dict,
    method: str = "damp",
) -> tuple[subprocess.CompletedProcess[str] | None, dict]:
    """Dispatch one NPD recovery: ``isor`` on the offending atoms or the legacy ``damp`` ladder."""

    if method == "isor":
        return recover_npd_with_isor(
            work_dir, stem, executable, iteration=iteration, base_ins_text=base_ins_text, initial_stats=initial_stats
        )
    if method == "damp":
        return recover_npd_shelxl_attempt(
            work_dir, stem, executable, iteration=iteration, base_ins_text=base_ins_text, initial_stats=initial_stats
        )
    raise ValueError(f"unknown NPD recovery {method!r}; use one of {', '.join(NPD_RECOVERY_METHODS)}")


# SHELXL weighting-scheme policies.  converge (every corrected-HKL endpoint) starts at WGHT 0 0 and promotes
# SHELXL's suggested WGHT on every call until the applied and suggested weights agree and the last L.S. cycle
# has converged, so the written .res carries one converged WGHT.  inherit keeps the model's own WGHT (the
# conventional baseline replay of the user's model and HKL).
WGHT_MODES = ("inherit", "converge")
WGHT_CONVERGE_TOL_A = 0.002
WGHT_CONVERGE_TOL_B = 0.02
WGHT_CONVERGE_DEFAULT_MAX_CALLS = 20


def endpoint_initial_wght(mode: str) -> tuple[float, float] | None:
    """Active WGHT of the first SHELXL call of one endpoint handoff: WGHT 0 0 under converge; inherit keeps the
    source model's WGHT (None)."""

    if mode == "converge":
        return (0.0, 0.0)
    return None


def wght_promotes(mode: str) -> bool:
    """Whether later calls of this weighting mode apply SHELXL's suggested WGHT."""

    return mode == "converge"


def wght_call_limit(mode: str, iterations: int, converge_max_calls: int) -> int:
    """SHELXL calls one endpoint may make: ``--wght-converge-max-calls`` under converge, else ``iterations``."""

    if mode == "converge":
        return max(1, int(converge_max_calls))
    return max(1, int(iterations))


def wght_agreement(
    stats: dict,
    *,
    tol_a: float = WGHT_CONVERGE_TOL_A,
    tol_b: float = WGHT_CONVERGE_TOL_B,
) -> dict:
    """Compare the WGHT one SHELXL call applied with the WGHT SHELXL suggests for the next call."""

    applied = _active_wght_from_stats(stats)
    suggested = recommended_wght_from_stats(stats)
    out: dict[str, object] = {
        "applied": list(applied) if applied else None,
        "suggested": list(suggested) if suggested else None,
        "tol_a": float(tol_a),
        "tol_b": float(tol_b),
    }
    if applied is None or suggested is None:
        out.update({"agree": False, "reason": "applied or suggested WGHT was not reported by SHELXL"})
        return out
    da = abs(float(suggested[0]) - float(applied[0]))
    db = abs(float(suggested[1]) - float(applied[1]))
    out.update({"da": da, "db": db, "agree": bool(da < float(tol_a) and db < float(tol_b))})
    return out


def endpoint_call_converged(stats: dict, mode: str, tolerance: float) -> bool:
    """Stop test of one call: shift/esd below ``tolerance``; under converge also applied == suggested WGHT."""

    if not shelxl_shift_converged(stats, tolerance):
        return False
    if mode == "converge":
        return bool(wght_agreement(stats).get("agree"))
    return True


def wght_convergence_summary(
    history: list[dict],
    final: dict,
    *,
    mode: str,
    max_calls: int,
    shift_tolerance: float,
) -> dict:
    """Per-endpoint record of the weighting scheme: each call's applied/suggested WGHT and the final state."""

    calls = []
    for index, item in enumerate(history):
        agreement = wght_agreement(item)
        calls.append(
            {
                "call": int(item.get("iteration", index + 1) or index + 1),
                "applied": agreement.get("applied"),
                "suggested": agreement.get("suggested"),
                "max_shift_esd": item.get("max_shift_esd"),
                "GooF": item.get("GooF"),
            }
        )
    agreement = wght_agreement(final)
    shift_ok = shelxl_shift_converged(final, shift_tolerance)
    safe = shelxl_iteration_safe(final)
    converged = bool(agreement.get("agree")) and shift_ok and safe
    if mode != "converge":
        status = f"not_required_{mode}"
    elif converged:
        status = "converged"
    elif not safe or final.get("restored_last_safe_iteration"):
        # A later call went unsafe (NPD after recovery) and the last safe call's unconverged model was kept.
        status = "not_converged_unsafe_call"
    elif len(history) >= int(max_calls):
        status = "not_converged_cap"
    else:
        status = "not_converged"
    return {
        "mode": str(mode),
        "status": status,
        "converged": converged,
        "weights_agree": bool(agreement.get("agree")),
        "shift_esd_converged": bool(shift_ok),
        "final_applied": agreement.get("applied"),
        "final_suggested": agreement.get("suggested"),
        "da": agreement.get("da"),
        "db": agreement.get("db"),
        "tol_a": WGHT_CONVERGE_TOL_A,
        "tol_b": WGHT_CONVERGE_TOL_B,
        "tol_shift_esd": float(shift_tolerance),
        "max_calls": int(max_calls),
        "calls_made": len(history),
        "calls": calls,
    }


def shelxl_shift_converged(stats: dict, tolerance: float) -> bool:
    """Return whether the last L.S. cycle's |max shift/esd| is below ``tolerance`` (0 disables the test)."""

    if not float(tolerance) > 0.0:
        return False
    try:
        value = float(stats.get("max_shift_esd"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value) < float(tolerance)


def shelxl_model_restraint_audit(res_path: Path) -> dict:
    """DAMP/ISOR state of a written model: what the next handoff would inherit."""

    text = res_path.read_text(errors="replace") if res_path.exists() else ""
    isor = isor_restrained_atoms(text)
    return {"damp_present": active_damp_present(text), "isor_atoms": isor, "n_isor": len(isor)}
