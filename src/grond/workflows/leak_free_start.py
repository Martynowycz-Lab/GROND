"""Leak-free start model: refine the supplied structure conventionally on work orbits only.

The supplied start model is normally the conventional refinement against every reflection, so the orbits AUTO later
holds out (split=free, and any --exclude-orbits-json orbits) have already shaped it. Every later stage would inherit
that fit, and the free-row R of the whole run would be optimistic. A dynamical refinement started from such a model
also appears to lose that advantage cycle by cycle, which is easily mistaken for the macro-cycle loop drifting.

This stage removes the held-out orbits from the supplied conventional HKLF 4 file, using exactly the orbit-hash
membership of the row engine (grond.bloch.splits.orbit_hash_split_masks; under a pseudo-symmetric shared candidate
closure, grond.bloch.splits.shared_orbit_split_masks, the membership every crystal's split contract gives the same
reflection), and refines the supplied model against the
rest to convergence: same instructions (EXTI, restraints), SHELXL-suggested weights promoted each call, stop at
|max shift/esd| < 0.01. The result replaces the start model for every later AUTO stage. Kinematic refinement is
hand-blind, so the hand of the supplied model is unchanged.

A call that leaves atoms non-positive definite gets the NPD recovery of the SHELXL loop
(grond.refine.shelxl_iterations.recover_npd_with_isor): the same call is repeated from its unchanged input with
``ISOR 0.01 0.02`` on the NPD atoms, then ``ISOR 0.005 0.01`` on an atom still NPD.  The restraints stay in the model
(SHELXL copies them into the .res), as in the loop, and are recorded with the outcome.  Only when the recovery also
fails does the stage fall back to the supplied model.  A refinement without NPD atoms never reaches the recovery, so
its files and record are unchanged.

Under --endpoint-free-validation on (AUTO default) the finished model is then scored, reporting only, on the free
reflections of the same baseline HKL (same masks) with SHELXL L.S. 0 next to its own work reflections
(grond.validation.endpoint_free): the honest conventional R_free on the orbits GROND holds out.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from grond.bloch.model import parse_model
from grond.bloch.splits import (
    SHARED_CANDIDATE_POLICY,
    load_split_exclusion,
    orbit_hash_split_masks,
    shared_orbit_split_masks,
)
from grond.io.config import PrepConfig
from grond.macromolecule import sha256_file
from grond.refine.shelx_input import set_active_wght_lines
from grond.refine.shelxl_diagnostics import (
    npd_atoms_from_lst_text,
    parse_lst,
    parse_shift_esd,
    parse_suggested_wght_res,
    shelx_res_health,
)
from grond.refine.shelxl_iterations import recover_npd_with_isor
from grond.refine.zero_cycle import active_wght_line
from grond.validation.endpoint_free import conventional_free_validation, record_endpoint_reproduction
from grond.validation.heldout import DEFAULT_FREE_SALT
from grond.workflows.conventional_baseline import _publication_baseline_hkl

SHIFT_ESD_TOL = 0.01
MAX_CALLS = 15
LS_CYCLES = 10
# Poorly determined overall parameters (EXTI above all) can creep for many calls after every atom has converged;
# they do not carry the held-out rows' information that the leak-free start exists to remove.
OVERALL_PARAMETERS = ("EXTI", "OSF", "BASF", "SWAT", "FVAR")
OVERALL_SHIFT_ESD_TOL = 1.0


def structurally_converged(stats: dict) -> bool:
    """Max shift/esd below 0.01, or every parameter but one overall scalar converged (mean below 0.01)."""

    max_shift = abs(float(stats.get("max_shift_esd", 1e9)))
    if max_shift < SHIFT_ESD_TOL:
        return True
    parameter = str(stats.get("max_shift_param") or "").split()[0].upper() if stats.get("max_shift_param") else ""
    return (
        parameter in OVERALL_PARAMETERS
        and abs(float(stats.get("mean_shift_esd", 1e9))) < SHIFT_ESD_TOL
        and max_shift < OVERALL_SHIFT_ESD_TOL
    )


def leak_free_start_mode(args: argparse.Namespace) -> str:
    """Return the ``--leak-free-start`` setting (on/off; off when absent); raise on any other value."""

    mode = str(getattr(args, "leak_free_start", "off") or "off")
    if mode not in {"on", "off"}:
        raise ValueError(f"unknown --leak-free-start {mode!r}")
    return mode


def endpoint_free_validation_mode(args: argparse.Namespace) -> str:
    """Return the ``--endpoint-free-validation`` setting (on/off; off when absent); raise on any other value."""

    mode = str(getattr(args, "endpoint_free_validation", "off") or "off")
    if mode not in {"on", "off"}:
        raise ValueError(f"unknown --endpoint-free-validation {mode!r}")
    return mode


def leak_free_split_masks(
    rows: list[tuple[tuple[int, int, int], str]],
    model: dict,
    *,
    free_fraction: float,
    salt: str,
    excluded_orbits=None,
    union: dict | None = None,
) -> tuple[dict, object, dict | None]:
    """Orbit-hash masks of the baseline HKL rows (and the candidate-set record under a shared candidate closure)."""

    union = dict(union or {})
    hkls = [hkl for hkl, _line in rows]
    if str(union.get("policy") or "") == SHARED_CANDIDATE_POLICY:
        masks, excluded = shared_orbit_split_masks(
            hkls, list(union["shared_group"]), float(free_fraction), str(salt), model.get("symmetry_ops"),
            excluded_orbits=excluded_orbits,
        )
        return masks, excluded, {
            "policy": SHARED_CANDIDATE_POLICY,
            "shared_group_sha256": str(union["shared_group_sha256"]),
            "laue_coset_count": int(union["laue_coset_count"]),
        }
    masks, excluded = orbit_hash_split_masks(
        hkls, float(free_fraction), str(salt), model.get("symmetry_ops"), excluded_orbits=excluded_orbits
    )
    return masks, excluded, None


def work_hkl_text(rows: list[tuple[tuple[int, int, int], str]], masks: dict) -> str:
    """The leak-free refinement HKL: the baseline's work lines, verbatim, and the 0 0 0 terminator."""

    work_lines = [line for (_hkl, line), keep in zip(rows, masks["work"], strict=True) if keep]
    return "\n".join(work_lines) + "\n   0   0   0    0.00    0.00\n"


def score_leak_free_start(
    out_dir: Path, rows: list[tuple[tuple[int, int, int], str]], masks: dict, *, shelxl: str, stem: str = "start"
) -> dict[str, object]:
    """Free validation of a finished leak-free start: start_final.res at L.S. 0 on the free and work reflections.

    The WGHT is the one the model was refined with in its final call (``stem``.ins, the final call's input), so the work
    statistics reproduce the refinement's last listing.  SHELXL writes that WGHT into the .res before END as well and
    its suggestion for a next call after END; both are recorded.
    """

    out_dir = Path(out_dir)
    ins = out_dir / f"{stem}.ins"
    final = out_dir / f"{stem}_final.res"
    final_text = final.read_text(errors="replace")
    applied = active_wght_line(ins.read_text(errors="replace")) if ins.is_file() else None
    suggested = parse_suggested_wght_res(final_text)
    record = conventional_free_validation(
        out_dir, rows, masks, res=final, wght_line=applied, shelxl=shelxl,
        wght_source=f"final refinement call ({ins.name})" if applied else f"{final.name}",
    )
    record["wght_applied_in_final_call"] = applied
    record["wght_in_final_res"] = active_wght_line(final_text)
    record["wght_suggested_in_final_res"] = f"WGHT {suggested[0]:g} {suggested[1]:g}" if suggested else None
    listing = parse_lst(out_dir / f"{stem}.lst")
    if isinstance(record.get("work"), dict) and listing.get("status") == "ok":
        # the refinement's last call lists the parameters before its final shift; converged, they agree to rounding
        record_endpoint_reproduction(record, listing)
    return record


def read_hklf4_rows(path: Path) -> list[tuple[tuple[int, int, int], str]]:
    """(hkl, original line) for every reflection before the 0 0 0 terminator of a fixed-format HKLF 4 file."""

    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if len(line) < 12 or not line[:12].strip():
            continue
        try:
            hkl = (int(line[0:4]), int(line[4:8]), int(line[8:12]))
        except ValueError:
            continue
        if hkl == (0, 0, 0):
            break
        rows.append((hkl, line))
    return rows


def _with_ls_cycles(text: str, cycles: int) -> str:
    """Set every active L.S./CGLS count to `cycles` (insert L.S. before FVAR when the model has none)."""

    lines, found = [], False
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0].upper() in {"L.S.", "CGLS"}:
            lines.append(f"{tokens[0]} {int(cycles)}" + (" " + " ".join(tokens[2:]) if len(tokens) > 2 else ""))
            found = True
            continue
        if not found and tokens and tokens[0].upper() == "FVAR":
            lines.append(f"L.S. {int(cycles)}")
            found = True
        lines.append(line)
    return "\n".join(lines) + "\n"


_SHELXL_STAMP = re.compile(r"^\s*created by SHELXL\S* at .*$", re.MULTILINE)


def same_model_apart_from_shelxl_stamp(first: Path, second: Path) -> bool:
    """True when two SHELXL .res files differ at most in SHELXL's 'created by SHELXL-... at <time>' header line."""

    return _SHELXL_STAMP.sub("", first.read_text(errors="replace")) == _SHELXL_STAMP.sub(
        "", second.read_text(errors="replace")
    )


def _res_before_end(text: str) -> str:
    """The instruction/model part of a SHELXL .res up to and including END, without Q-peaks."""

    head, sep, _tail = text.partition("\nEND")
    kept = [line for line in head.splitlines() if not re.match(r"^Q\d+\s", line)]
    return "\n".join(kept) + (sep + "\n" if sep else "\n")


def _restraint_lines(isor_write: object) -> list[str]:
    """The ISOR lines one recovery attempt wrote (standard tier, then tight tier)."""

    if not isinstance(isor_write, dict):
        return []
    if "standard" in isor_write or "tight" in isor_write:
        return [*_restraint_lines(isor_write.get("standard")), *_restraint_lines(isor_write.get("tight"))]
    return [str(line) for line in isor_write.get("lines") or []]


def recover_leak_free_npd(
    out_dir: Path, stem: str, shelxl: str, *, call: int, ins_text: str, logs: Path
) -> dict[str, object]:
    """The SHELXL loop's NPD recovery for one leak-free start call that left NPD atoms.

    recover_npd_with_isor repeats the call from its unchanged input (``ins_text``) with ``ISOR 0.01 0.02`` on the NPD
    atoms and ``ISOR 0.005 0.01`` on an atom that stays NPD, archiving every attempt in ``out_dir``.  It is accepted
    when SHELXL reports a safe step and names no NPD atom; ``stem``.res/.lst/.ins are then the restrained call.
    """

    out_dir = Path(out_dir)
    lst_path, res_path = out_dir / f"{stem}.lst", out_dir / f"{stem}.res"
    initial = {"iteration": int(call) + 1, "returncode": 0, **parse_lst(lst_path)}
    initial["res_health"] = shelx_res_health(res_path, lst_path)
    initial["status"] = "adp_pathology"
    proc, stats = recover_npd_with_isor(
        out_dir, stem, shelxl, iteration=int(call) + 1, base_ins_text=ins_text, initial_stats=initial
    )
    if proc is not None:
        with (Path(logs) / "leak_free_start_shelxl.log").open("a") as handle:
            handle.write(proc.stdout)
    recovery = stats.get("npd_recovery") if isinstance(stats.get("npd_recovery"), dict) else {}
    lst = lst_path.read_text(errors="replace") if lst_path.exists() else ""
    res = res_path.read_text(errors="replace") if res_path.exists() else ""
    accepted = (
        recovery.get("status") == "accepted_isor_restrained_step"
        and proc is not None and proc.returncode == 0
        and "FVAR" in res.upper() and not npd_atoms_from_lst_text(lst)
    )
    attempts = []
    for attempt in recovery.get("attempts") or []:
        health = attempt.get("res_health") if isinstance(attempt.get("res_health"), dict) else {}
        attempts.append({
            "attempt": attempt.get("attempt"),
            "status": attempt.get("status"),
            "npd_atoms": list(health.get("npd_atoms") or []),
            "isor_atoms_added": attempt.get("isor_atoms_added", []),
            "isor_atoms_tightened": attempt.get("isor_atoms_tightened", []),
            "res": attempt.get("res"),
            "lst": attempt.get("lst"),
        })
    return {
        "method": "isor",
        "status": recovery.get("status"),
        "reason": recovery.get("reason", ""),
        "accepted": bool(accepted),
        "isor_atoms": list(recovery.get("isor_atoms") or []),
        "isor_atoms_tightened": list(recovery.get("isor_atoms_tightened") or []),
        "restraints_added": _restraint_lines(stats.get("isor_write")) if accepted else [],
        "attempts": attempts,
    }


def run_leak_free_start(
    prep: PrepConfig, args: argparse.Namespace, logs: Path, *, split_plan=None
) -> dict[str, object]:
    """Refine the start model on work orbits only; returns a record with the new model path when status is ok.

    ``split_plan`` is the run's split candidate plan (``plan_split_candidates``; AUTO passes it).  Under a shared
    candidate closure the removed rows are those of the shared-group orbits the split contracts hold out, so a row
    that is free in the contracts because a pseudo-symmetric candidate merged its orbit with a free one is removed
    too.  Without a plan the membership is the Laue-orbit one (identical whenever no candidate is pseudo-symmetric).
    """

    if leak_free_start_mode(args) == "off":
        return {"status": "disabled", "leak": "start model as supplied (refined on every reflection)"}
    if str(getattr(args, "split_scheme", "legacy")) != "orbit-hash":
        return {"status": "not_applicable_legacy_split",
                "leak": "start model as supplied; the legacy per-dataset split has no crystal-independent free orbits"}
    hkl_text = _publication_baseline_hkl(prep)
    if not hkl_text:
        print("WARN leak-free start unavailable: no conventional baseline HKL was supplied; every stage starts from the "
              "supplied model, so free-row R stays optimistic (pass --baseline-hkl to enable it)", flush=True)
        return {"status": "unavailable", "reason": "no conventional baseline HKL was supplied",
                "leak": "start model as supplied (refined on every reflection)"}
    out_dir = Path(args.out_dir) / "controls" / "leak_free_start"
    record: dict[str, object] = {"out_dir": str(out_dir), "input_res": str(prep.structure), "input_hkl": hkl_text,
                                 "split_scheme": "orbit-hash", "split_salt": str(getattr(args, "split_salt", DEFAULT_FREE_SALT)),
                                 "free_fraction": float(getattr(args, "free_fraction", 0.10))}
    if bool(getattr(args, "dry_run", False)):
        return {**record, "status": "dry_run"}
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_model(Path(prep.structure))
    rows = read_hklf4_rows(Path(hkl_text))
    if not rows:
        raise RuntimeError(f"leak-free start: no reflections read from {hkl_text}")
    exclusion_path = str(getattr(args, "exclude_orbits_json", "") or "").strip()
    exclusion = load_split_exclusion(exclusion_path) if exclusion_path else None
    masks, excluded, candidate_set = leak_free_split_masks(
        rows,
        model,
        free_fraction=float(getattr(args, "free_fraction", 0.10)),
        salt=str(getattr(args, "split_salt", DEFAULT_FREE_SALT)),
        excluded_orbits=exclusion["orbits"] if exclusion else None,
        union=(split_plan or {}).get("union"),
    )
    if candidate_set is not None:
        record["split_candidate_set"] = candidate_set
    work_lines = [line for (hkl, line), keep in zip(rows, masks["work"], strict=True) if keep]
    if not work_lines:
        raise RuntimeError("leak-free start: every reflection is free or excluded")
    stem = "start"
    (out_dir / f"{stem}.hkl").write_text(work_hkl_text(rows, masks))
    ins_text = _with_ls_cycles(Path(prep.structure).read_text(errors="replace"), LS_CYCLES)
    shelxl = str(getattr(args, "shelxl", "shelxl") or "shelxl")
    calls: list[dict[str, object]] = []
    status = "not_converged"
    for call in range(MAX_CALLS):
        (out_dir / f"{stem}.ins").write_text(ins_text)
        proc = subprocess.run([shelxl, stem], cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        with (Path(logs) / "leak_free_start_shelxl.log").open("a") as handle:
            handle.write(proc.stdout)
        lst_path, res_path = out_dir / f"{stem}.lst", out_dir / f"{stem}.res"
        lst = lst_path.read_text(errors="replace") if lst_path.exists() else proc.stdout
        res = res_path.read_text(errors="replace") if res_path.exists() else ""
        stats: dict[str, object] = {"call": call, "returncode": proc.returncode, **parse_shift_esd(lst)}
        stats["npd_atoms"] = npd_atoms_from_lst_text(lst)
        calls.append(stats)
        if proc.returncode != 0 or "FVAR" not in res.upper():
            status = "shelxl_failed"
            break
        shutil.copy(res_path, out_dir / f"{stem}_call{call:02d}.res")
        if stats["npd_atoms"]:
            # the SHELXL loop's NPD recovery (ISOR on the NPD atoms, then the tight tier), from this call's input
            recovery = recover_leak_free_npd(out_dir, stem, shelxl, call=call, ins_text=ins_text, logs=logs)
            stats["npd_recovery"] = recovery
            if not recovery["accepted"]:
                # fail closed: a conventional refinement that breaks an ADP on 90% of the data is not a safe start
                status = "npd"
                break
            # the call's result is now the restrained one
            lst = lst_path.read_text(errors="replace")
            res = res_path.read_text(errors="replace")
            stats["npd_attempt"] = {key: stats.get(key) for key in ("mean_shift_esd", "max_shift_esd", "max_shift_param")}
            stats.update(parse_shift_esd(lst))
            stats["npd_atoms_after_recovery"] = npd_atoms_from_lst_text(lst)
            shutil.copy(res_path, out_dir / f"{stem}_call{call:02d}_isor.res")
        suggested = parse_suggested_wght_res(res)
        next_text = _res_before_end(res)
        if suggested is not None:
            next_text = "\n".join(set_active_wght_lines(next_text.splitlines(), *suggested)) + "\n"
        ins_text = next_text
        if structurally_converged(stats):
            stats["converged_on"] = "all_parameters" if abs(float(stats.get("max_shift_esd", 1e9))) < SHIFT_ESD_TOL else (
                f"structure; {stats.get('max_shift_param')} still moving (shift/esd {stats.get('max_shift_esd')})")
            status = "ok"
            break
    record.update({
        "n_reflections": len(rows), "n_work": len(work_lines), "n_free_removed": int(masks["free"].sum()),
        "n_excluded_removed": int(excluded.sum()), "calls": calls,
    })
    recoveries = [call_stats["npd_recovery"] for call_stats in calls if "npd_recovery" in call_stats]
    if recoveries:
        record["npd_recovery"] = {
            "method": "isor (grond.refine.shelxl_iterations.recover_npd_with_isor, as in the SHELXL loop)",
            "calls": [int(call_stats["call"]) for call_stats in calls if "npd_recovery" in call_stats],
            "accepted": all(item["accepted"] for item in recoveries),
            "restraints_added": [line for item in recoveries for line in item["restraints_added"]],
        }
    if status != "ok":
        record.update({"status": status, "leak": "start model as supplied (the work-orbit refinement did not finish cleanly)"})
        (out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"WARN leak-free start {status}; continuing from the supplied model (free-row R stays optimistic)", flush=True)
        return record
    final = out_dir / f"{stem}_final.res"
    # SHELXL writes the wall-clock time into the .res header, so a rerun (a resumed AUTO) gives the same model under
    # a new SHA-256 and every stage that reads the start model would rerun. Keep the file when only that line differs.
    if not (final.is_file() and same_model_apart_from_shelxl_stamp(final, out_dir / f"{stem}.res")):
        shutil.copy(out_dir / f"{stem}.res", final)
    record.update({"status": "ok", "res": str(final), "res_sha256": sha256_file(final),
                   "leak": "none: start model refined on work orbits only"})
    if recoveries:
        record["leak"] = ("none: start model refined on work orbits only, with ISOR on atoms that went NPD ("
                          + "; ".join(record["npd_recovery"]["restraints_added"]) + ")")
    if endpoint_free_validation_mode(args) == "on":
        try:  # reporting only: a failed validation never changes the start model or stops the run
            record["free_validation"] = score_leak_free_start(out_dir, rows, masks, shelxl=shelxl, stem=stem)
        except Exception as exc:
            record["free_validation"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    (out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def prep_with_leak_free_start(prep: PrepConfig, record: dict[str, object]) -> PrepConfig:
    """The prep with the leak-free model as its structure (unchanged unless the stage finished cleanly)."""

    if str(record.get("status")) != "ok":
        return prep
    return replace(prep, structure=str(record["res"]))
