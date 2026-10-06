#!/usr/bin/env python3
"""Combine separately fitted native nuisance runs without merging raw rows first.

This is a reporting/export bridge for multi-crystal tests.  Each input run is a
single-dataset `native_row_regularize.py` output directory.  The per-dataset
predictions and scales remain independent; this script concatenates the
resulting signed row tables and reports joint native metrics on the same
calculation definitions.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from grond.refine.shelx_input import ABSOLUTE_HKL_KAPPA, parse_basf_values, requests_acta
from grond.refine.shelxl_iterations import (
    NPD_RECOVERY_METHODS,
    WGHT_CONVERGE_DEFAULT_MAX_CALLS,
    WGHT_MODES,
    endpoint_call_converged,
    endpoint_initial_wght as endpoint_initial_wght,
    recover_npd_attempt,
    shelxl_model_restraint_audit,
    shelxl_shift_converged,
    wght_agreement,
    wght_call_limit,
    wght_convergence_summary,
    wght_promotes,
)
from grond.refine.zero_cycle import active_wght_line
from grond.regularize.arguments_combination import build_parser as build_combination_parser
from grond.regularize.multidataset import (
    apply_hkl_weight as apply_hkl_weight,
    apply_native_row_weight as apply_native_row_weight,
    apply_sigma_holdout_floor as apply_sigma_holdout_floor,
    audit_combined_endpoint_modulation as audit_combined_endpoint_modulation,
    canonical_model_friedel as canonical_model_friedel,
    canonical_model_symmetry as canonical_model_symmetry,
    endpoint_key_function as endpoint_key_function,
    endpoint_rows_and_corrections_from_native as endpoint_rows_and_corrections_from_native,
    estimate_overlap_scales as estimate_overlap_scales,
    file_provenance as file_provenance,
    filter_hkl_rows_by_resolution as filter_hkl_rows_by_resolution,
    filter_native_rows_by_resolution as filter_native_rows_by_resolution,
    hkl_resolution_stats as hkl_resolution_stats,
    merge_hkl_rows as merge_hkl_rows,
    metrics as metrics,
    native_metrics_by_split as native_metrics_by_split,
    safe_float as safe_float,
    scale_contract_provenance as scale_contract_provenance,
    scale_hkl_rows as scale_hkl_rows,
    selected_native_wght as selected_native_wght,
    split_metrics as split_metrics,
)
from grond.validation.endpoint_free import record_endpoint_reproduction, score_model_on_work_and_free

try:  # Package import.
    from .bloch_engine import parse_model, sha256
    from .native_row_regularize import (
        fail_if_shelxl_became_unavailable,
        parse_shelx_wght_terms,
        recommended_wght_from_stats,
        require_endpoint_shelxl,
        run_one_shelxl_attempt,
        select_last_safe_shelxl_iteration,
        shelx_res_health,
        shelxl_iteration_safe,
        snapshot_safe_shelxl_iteration,
        write_active_wght,
        write_endpoint_ins,
    )
    from .post_bloch_sigma import (
        apply_error_model_sigma,
        fit_error_model_sigma,
        fit_uniform_z_rms_sigma_scale,
        rms_z_against_model,
        scale_row_sigmas,
    )
    from .shelx_hkl import (
        format_fixed_width_row,
        parse_hkl_observation_line,
        safe_fixed_width_factor,
        write_fixed_width_batch_hkl,
        write_fixed_width_hkl,
    )
except ImportError:  # Direct script execution from this directory.
    from bloch_engine import parse_model, sha256
    from native_row_regularize import (
        fail_if_shelxl_became_unavailable,
        parse_shelx_wght_terms,
        recommended_wght_from_stats,
        require_endpoint_shelxl,
        run_one_shelxl_attempt,
        select_last_safe_shelxl_iteration,
        shelx_res_health,
        shelxl_iteration_safe,
        snapshot_safe_shelxl_iteration,
        write_active_wght,
        write_endpoint_ins,
    )
    from post_bloch_sigma import (
        apply_error_model_sigma,
        fit_error_model_sigma,
        fit_uniform_z_rms_sigma_scale,
        rms_z_against_model,
        scale_row_sigmas,
    )
    from shelx_hkl import (
        format_fixed_width_row,
        parse_hkl_observation_line,
        safe_fixed_width_factor,
        write_fixed_width_batch_hkl,
        write_fixed_width_hkl,
    )


def read_json(path: Path) -> dict:
    """Read a JSON file, or return an empty dict if it does not exist."""

    return json.loads(path.read_text()) if path.exists() else {}


def read_rows(run_dir: Path, name: str) -> list[dict]:
    """Read one run's native_dyn_calc.csv, labelling every row with its dataset and filling contract defaults.

    A missing target_mode comes from the run's config; role, row and endpoint weights default to primary and
    1.0; sigma_eff is reset to sigma_eff_original when that is positive.
    """

    path = run_dir / "native_dyn_calc.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    config = read_json(run_dir / "summary.json").get("config") or {}
    recorded_target_mode = str(config.get("target_mode", "iobs"))
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            # Older diagnostic rows did not carry the restoration factor.
            # Preserve their declared mode so endpoint export fails explicitly
            # instead of silently treating those values as ordinary IOBS.
            r.setdefault("target_mode", recorded_target_mode)
            r["dataset"] = name
            r.setdefault("dataset_role", "primary")
            r.setdefault("raw_row_weight", "1.0")
            r.setdefault("endpoint_weight", r.get("raw_row_weight", "1.0"))
            r.setdefault("sigma_eff_original", r.get("sigma_eff", ""))
            original_sigma = safe_float(r.get("sigma_eff_original"))
            if math.isfinite(original_sigma) and original_sigma > 0.0:
                r["sigma_eff"] = f"{original_sigma:.17g}"
            rows.append(r)
    return rows


def read_hkl(path: Path):
    """Read (h, k, l, I, sigma) rows from a free-format or fixed-width HKL file, without the 0 0 0 terminator."""

    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        h, k, ell, intensity, sig = parse_hkl_observation_line(line)
        if h == k == ell == 0:
            continue
        rows.append((h, k, ell, intensity, sig))
    return rows


def write_hkl(path: Path, rows):
    """Write rows as a validated fixed-width HKLF 4 file."""

    write_fixed_width_hkl(path, rows)


def calibrate_dataset_endpoint_rows(
    rows,
    model,
    *,
    policy: str,
    holdout_fraction: float,
    min_scale: float,
    max_scale: float,
    holdout_floor: bool,
):
    """Calibrate original endpoint sigmas before applying dataset policy weight."""

    if policy != "uniform_z_rms":
        return rows, {"policy": policy, "status": "not_requested"}
    try:
        meta = fit_uniform_z_rms_sigma_scale(
            rows,
            model,
            holdout_fraction=float(holdout_fraction),
            min_scale=float(min_scale),
            max_scale=float(max_scale),
        )
        if holdout_floor:
            meta = apply_sigma_holdout_floor(meta)
        calibrated = rows
        if str(meta.get("status", "")).startswith("ok"):
            calibrated = scale_row_sigmas(rows, float(meta["sigma_scale"]))
        return calibrated, meta
    except Exception as exc:
        return rows, {
            "policy": "uniform_z_rms",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def error_model_rows_by_dataset(
    rows_by_dataset: dict[str, list[tuple]],
    model,
    free_rows_by_dataset: dict[str, list[tuple]],
    weight_by_name: dict[str, float],
    *,
    weighted: bool,
) -> tuple[dict[str, list[tuple]], dict[str, dict]]:
    """Fit each crystal's error model on its own work rows against ``model``; ``weighted`` applies the policy weight."""

    calibrated: dict[str, list[tuple]] = {}
    metas: dict[str, dict] = {}
    for name, rows_for_dataset in rows_by_dataset.items():
        if not rows_for_dataset:
            calibrated[name] = rows_for_dataset
            metas[name] = {"policy": "error_model", "status": "skipped_no_rows"}
            continue
        rows, meta = fit_error_model_sigma(
            rows_for_dataset,
            model,
            free_rows=free_rows_by_dataset.get(name, []),
            label=f"dataset {name}",
        )
        calibrated[name] = apply_hkl_weight(rows, weight_by_name.get(name, 1.0)) if weighted else rows
        metas[name] = meta
    return calibrated, metas


def write_no_exti_ins(
    src_res: Path,
    dst_ins: Path,
    hkl_name: str,
    *,
    wght: tuple[float, float] | None = None,
    ls_cycles: int = 1,
    keep_damp: bool = False,
    osf: float | None = None,
    **native_outputs,
):
    """Write a no-EXTI endpoint input; ``native_outputs`` are the write_endpoint_ins SHELXL-native options."""

    write_endpoint_ins(
        src_res,
        dst_ins,
        hkl_name=hkl_name,
        keep_exti=False,
        wght=wght,
        ls_cycles=ls_cycles,
        keep_damp=keep_damp,
        osf=osf,
        **native_outputs,
    )


def parse_zerr_esds(text: str) -> list[float] | None:
    """Six comma-separated cell e.s.d.s from --zerr-esds, or None when empty (keep the model's ZERR)."""

    if not str(text or "").strip():
        return None
    values = [float(value) for value in str(text).split(",")]
    if len(values) != 6 or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"--zerr-esds needs six finite nonnegative values a,b,c,alpha,beta,gamma, got {text!r}")
    return values


def endpoint_native_outputs(args, *, final: bool = True) -> dict:
    """write_endpoint_ins options of the SHELXL-native outputs requested on the command line."""

    out: dict = {}
    zerr = parse_zerr_esds(getattr(args, "zerr_esds", ""))
    if zerr is not None:
        out["zerr_esds"] = zerr
    if bool(getattr(args, "strip_q_peaks", False)):
        out["strip_q_peaks"] = True
    if int(getattr(args, "endpoint_list_code", 0) or 0) > 0:
        out["list_code"] = int(args.endpoint_list_code)
    if final and bool(getattr(args, "acta", False)):
        out["acta"] = True
    return out


def refined_basf(res_path: Path, lst_path: Path) -> dict:
    """BASF values of a refined .res and their e.s.d.s from the last L.S. cycle of the listing."""

    # SHELXL writes seven values per BASF record, so a run with more than eight crystals has several records.
    values = parse_basf_values(res_path.read_text(errors="replace")) if res_path.exists() else []
    esds: dict[int, float] = {}
    if lst_path.exists():
        pattern = re.compile(r"^\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+[-\d.]+\s+BASF\s+(\d+)", flags=re.M)
        for match in pattern.finditer(lst_path.read_text(errors="replace")):
            esds[int(match.group(3))] = float(match.group(2))
    return {"values": values, "esds": [esds.get(i + 1) for i in range(len(values))]}


def basf_sanity(start: list[float], refined: dict, *, low: float = 0.2, high: float = 5.0) -> dict:
    """A refined batch scale should be finite, positive, well determined and within [low, high]."""

    values = refined.get("values") or []
    esds = refined.get("esds") or []
    problems = []
    if len(values) != len(start):
        problems.append(f"{len(values)} refined BASF values for {len(start)} extra crystals")
    for index, value in enumerate(values):
        esd = esds[index] if index < len(esds) else None
        if not (math.isfinite(value) and low <= value <= high):
            problems.append(f"BASF {index + 1} = {value:g} outside [{low:g}, {high:g}]")
        if esd is None or not math.isfinite(float(esd)) or float(esd) <= 0.0:
            problems.append(f"BASF {index + 1} has no positive e.s.d. in the listing")
        elif abs(value) < 3.0 * float(esd):
            problems.append(f"BASF {index + 1} = {value:g} is not determined (e.s.d. {float(esd):g})")
    return {"ok": not problems, "problems": problems, "bounds": [low, high]}


def parse_lst(path: Path):
    """Parse the active WGHT, wR2, GooF, R1 and unique/suppressed counts from a SHELXL LST file.

    status is "missing" without the file and "no_regularization_stats" without a wR2 line.
    """

    out = {"status": "missing", "lst": str(path)}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    out["status"] = "ok"
    for line in text.splitlines():
        terms = parse_shelx_wght_terms(line)
        if terms:
            out["active_wght_a"] = terms[0]
            out["active_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
            out["active_wght_terms"] = terms
            break
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
    m = re.findall(r"(\d+)\s+Unique reflections, of which\s+(\d+)\s+suppressed", text)
    if m:
        out["unique"] = int(m[-1][0])
        out["suppressed"] = int(m[-1][1])
    if "wR2_all" not in out:
        out["status"] = "no_regularization_stats"
    return out


def promote_recommended_wght(ins_path: Path, recommended: tuple[float, float] | None) -> dict:
    """Write the recommended WGHT into the INS file and return what was set.

    Replaces the first WGHT or inserts one before FVAR, else before END or at the end; lines after END are
    dropped.
    """

    if recommended is None or not ins_path.exists():
        return {"promoted": False}
    a, b = recommended
    active_lines = []
    saw_end = False
    for line in ins_path.read_text(errors="replace").splitlines():
        active_lines.append(line)
        if re.match(r"^\s*END\b", line, flags=re.IGNORECASE):
            saw_end = True
            break
    if not active_lines:
        return {"promoted": False}
    wght_line = f"WGHT      {a:.4f}      {b:.4f}"
    for i, line in enumerate(active_lines):
        if line.strip().upper().startswith("WGHT"):
            active_lines[i] = wght_line
            break
    else:
        insert_at = None
        for i, line in enumerate(active_lines):
            if line.strip().upper().startswith("FVAR"):
                insert_at = i
                break
        if insert_at is None:
            insert_at = max(0, len(active_lines) - (1 if saw_end else 0))
        active_lines.insert(insert_at, wght_line)
    ins_path.write_text("\n".join(active_lines).rstrip() + "\n")
    return {"promoted": True, "active_wght_a": float(a), "active_wght_b": float(b)}


def _active_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    """Return the finite active WGHT (a, b) from SHELXL statistics, or None."""

    try:
        a = float(stats["active_wght_a"])
        b = float(stats.get("active_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def run_shelxl(
    work: Path,
    stem: str,
    shelxl: str,
    iterations: int = 5,
    *,
    wght_mode: str = "freeze-zero",
    endpoint_ls_cycles: int = 1,
    convergence_shift_esd: float = 0.0,
    npd_recovery: str = "damp",
    keep_damp: bool = False,
    wght_converge_max_calls: int = WGHT_CONVERGE_DEFAULT_MAX_CALLS,
    allow_basf: bool = False,
):
    """Run up to ``iterations`` SHELXL calls of ``endpoint_ls_cycles`` L.S. cycles on one fixed HKL.

    With ``convergence_shift_esd`` > 0 the calls stop once the last L.S. cycle has |max shift/esd| below it
    (AUTO: 15 calls of L.S. 10, tolerance 0.01), so each macro-cycle hands a converged model to the next
    Bloch calculation instead of one least-squares step.  ``wght_mode="converge"`` makes up to
    ``wght_converge_max_calls`` calls from WGHT 0 0, promoting SHELXL's suggested WGHT, and stops only once the
    applied and suggested weights agree as well.  ``allow_basf`` keeps the batch-scale BASF instructions of
    the unmerged multi-crystal endpoint.
    """

    exe = shelxl or shutil.which("shelxl") or shutil.which("shelxl2019") or ""
    if not exe:
        return {"status": "missing_shelxl", "executable": ""}
    if "/" in exe and not Path(exe).exists():
        return {"status": "missing_shelxl", "executable": exe}
    history = []
    last_safe = None
    last_stdout = ""
    mode = str(wght_mode or "freeze-zero")
    n_iter = wght_call_limit(mode, iterations, wght_converge_max_calls)
    n_ls = max(1, int(endpoint_ls_cycles))
    tolerance = float(convergence_shift_esd or 0.0)
    freeze_zero = mode == "freeze-zero"
    if mode not in WGHT_MODES:
        return {"returncode": None, "status": "invalid_wght_mode", "wght_mode": mode}
    if mode == "converge" and not tolerance > 0.0:
        return {"returncode": None, "status": "invalid_converge_tolerance", "wght_mode": mode}
    if npd_recovery not in NPD_RECOVERY_METHODS:
        return {"returncode": None, "status": "invalid_npd_recovery", "npd_recovery": str(npd_recovery)}
    # An endpoint written with ACTA keeps it on every call, so the CIF is SHELXL's CIF of the kept .res.
    acta = requests_acta(work / f"{stem}.ins")
    for i in range(n_iter):
        if freeze_zero:
            write_active_wght(work / f"{stem}.ins", 0.0, 0.0)
        ins = work / f"{stem}.ins"
        base_ins_text = ins.read_text(errors="replace") if ins.exists() else ""
        proc, stats = run_one_shelxl_attempt(
            work,
            stem,
            exe,
            iteration=i + 1,
            log_name=f"shelxl_iter_{i + 1}.log",
        )
        last_stdout = proc.stdout
        if str(stats.get("status", "")) == "adp_pathology" and base_ins_text:
            recovered_proc, stats = recover_npd_attempt(
                work,
                stem,
                exe,
                iteration=i + 1,
                base_ins_text=base_ins_text,
                initial_stats=stats,
                method=npd_recovery,
            )
            if recovered_proc is not None:
                last_stdout = recovered_proc.stdout
        history.append(stats)
        if not shelxl_iteration_safe(stats):
            break
        last_safe = snapshot_safe_shelxl_iteration(work, stem, i + 1, stats)
        stats["safe_snapshot_res"] = last_safe["res"]
        stats["safe_snapshot_lst"] = last_safe["lst"]
        stats["safe_snapshot_fcf"] = last_safe["fcf"]
        stats["shift_esd_converged"] = shelxl_shift_converged(stats, tolerance)
        stats["wght_agreement"] = wght_agreement(stats)
        last_safe["stats"] = dict(stats)
        if endpoint_call_converged(stats, mode, tolerance):
            break
        res = work / f"{stem}.res"
        if i + 1 < n_iter and res.exists():
            active = (0.0, 0.0) if freeze_zero else _active_wght_from_stats(stats)
            write_endpoint_ins(
                work / f"{stem}.res",
                work / f"{stem}.ins",
                hkl_name=f"{stem}.hkl",
                keep_exti=False,
                wght=active,
                ls_cycles=n_ls,
                keep_damp=keep_damp,
                allow_basf=allow_basf,
                acta=acta,
            )
            if wght_promotes(mode):
                stats["next_wght"] = promote_recommended_wght(
                    work / f"{stem}.ins",
                    recommended_wght_from_stats(stats),
                )
            elif freeze_zero:
                stats["next_wght"] = {"promoted": False, "frozen": True, "active_wght_a": 0.0, "active_wght_b": 0.0}
            else:
                stats["next_wght"] = {"promoted": False, "inherited": True}
    (work / "shelxl_stdout.log").write_text(last_stdout)
    final = select_last_safe_shelxl_iteration(work, stem, history, last_safe)
    final.setdefault("res_health", shelx_res_health(work / f"{stem}.res", work / f"{stem}.lst"))
    if final.get("returncode") not in {None, 0}:
        final["status"] = "shelxl_failed"
        final["invalid_res_reason"] = f"SHELXL returned {final['returncode']}"
    elif final["res_health"].get("ok"):
        active = (0.0, 0.0) if freeze_zero else _active_wght_from_stats(final)
        write_endpoint_ins(
            work / f"{stem}.res",
            work / f"{stem}.res",
            hkl_name=f"{stem}.hkl",
            keep_exti=False,
            wght=active,
            ls_cycles=n_ls,
            keep_damp=keep_damp,
            allow_basf=allow_basf,
            acta=acta,
        )
        final["res_normalized_for_reopen"] = True
        final["res_health"] = shelx_res_health(work / f"{stem}.res", work / f"{stem}.lst")
        if not final["res_health"].get("ok"):
            final["status"] = "invalid_res_output"
            final["invalid_res_reason"] = final["res_health"].get("reason")
    else:
        final["status"] = "invalid_res_output"
        final["invalid_res_reason"] = final["res_health"].get("reason")
    final["iterations"] = n_iter
    final["iterations_completed"] = len(history)
    final["endpoint_ls_cycles"] = n_ls
    final["wght_mode"] = mode
    final["convergence_shift_esd"] = tolerance
    final["converged"] = bool(shelxl_iteration_safe(final) and endpoint_call_converged(final, mode, tolerance))
    final["wght_convergence"] = wght_convergence_summary(
        history, final, mode=mode, max_calls=n_iter, shift_tolerance=tolerance
    )
    final["npd_recovery_method"] = str(npd_recovery)
    final["keep_damp"] = bool(keep_damp)
    if acta:
        cif = work / f"{stem}.cif"
        final["acta_cif"] = str(cif) if cif.is_file() else ""
    final.update(shelxl_model_restraint_audit(work / f"{stem}.res"))
    final["iteration_history"] = history
    pathology = final.get("res_health") or {}
    final["npd_count"] = int(pathology.get("npd_count", 0) or 0)
    final["split_count"] = int(pathology.get("split_count", 0) or 0)
    final["npd_atoms"] = pathology.get("npd_atoms", [])
    final["has_npd"] = bool(pathology.get("has_npd", False))
    final["has_split_warnings"] = bool(pathology.get("has_split_warnings", False))
    return final


def require_combined_endpoint_shelxl(
    executable: str,
    *,
    endpoint_rows_available: bool,
) -> str:
    """Require SHELXL only when a combined endpoint will be refined."""

    resolved = str(executable or shutil.which("shelxl") or shutil.which("shelxl2019") or "")
    require_endpoint_shelxl(
        resolved,
        skip_endpoint_refinement=not endpoint_rows_available,
    )
    return resolved


def endpoint_lam_dir(run: Path, lam: str) -> Path:
    """Return a run's endpoint directory for one lambda, e.g. endpoint_lam_1p0."""

    return run / f"endpoint_lam_{lam.replace('.', 'p')}"


def endpoint_hkl_path(run: Path, lam: str) -> Path:
    """Return a run's corrected endpoint HKL for lambda, preferring the unmerged file.

    Searches the plain and the B/noexti endpoint directories; returns the default path if none exists.
    """

    lam_tag = str(lam).replace(".", "p")
    candidate_dirs = [
        endpoint_lam_dir(run, lam),
        run / f"endpoint_lam_{lam_tag}_B_0p0_noexti",
    ]
    candidate_dirs.extend(sorted(run.glob(f"endpoint_lam_{lam_tag}_B_*_noexti")))
    for ep_dir in candidate_dirs:
        for name in ("dyn_nuisance_corrected_unmerged.hkl", "dyn_nuisance_corrected.hkl"):
            path = ep_dir / name
            if path.exists():
                return path
    return endpoint_lam_dir(run, lam) / "dyn_nuisance_corrected_unmerged.hkl"


def run_unmerged_batch_endpoint(
    out: Path,
    args,
    rows_by_dataset: dict[str, list[tuple]],
    corrections_by_dataset: dict[str, np.ndarray | None],
    post_bloch_sigma: dict,
    shelxl_executable: str,
    endpoint_key_for,
) -> dict:
    """Unmerged multi-crystal SHELXL endpoint: every corrected work row, batch = crystal index, MERG 0, BASF.

    The rows are those of the merged endpoint before the inter-dataset scale and the merge: the same
    per-crystal sigma policy and dataset weight (and the combined post-Bloch factor when that is active).
    SHELXL refines one batch scale per extra crystal (BASF, relative to crystal 1 on the OSF), starting
    from the overlap estimate of the relative intensity scale, with the same weighting policy.
    """

    names = [name for name, rows in rows_by_dataset.items() if rows]
    if len(names) < 2:
        return {"status": "skipped_fewer_than_two_crystals", "datasets": names}
    stem = "multi_dyn_unmerged"
    ep_dir = out / f"combined_unmerged_endpoint_lam_{args.endpoint_lambda.replace('.', 'p')}"
    ep_dir.mkdir(exist_ok=True)
    sigma_scale = 1.0
    if str(post_bloch_sigma.get("status", "")).startswith("ok"):
        sigma_scale = float(post_bloch_sigma["sigma_scale"])
    rows: list[tuple] = []
    batches: list[int] = []
    batch_of: dict[str, int] = {}
    for batch, name in enumerate(names, start=1):
        dataset_rows = scale_row_sigmas(rows_by_dataset[name], sigma_scale) if sigma_scale != 1.0 else rows_by_dataset[name]
        rows.extend(dataset_rows)
        batches.extend([batch] * len(dataset_rows))
        batch_of[name] = batch
    hkl_safe_factor = safe_fixed_width_factor(rows)
    if hkl_safe_factor < 1.0:
        rows = [(h, k, ell, intensity * hkl_safe_factor, sig * hkl_safe_factor) for h, k, ell, intensity, sig in rows]
    overlap = estimate_overlap_scales({name: rows_by_dataset[name] for name in names}, canonicalizer=endpoint_key_for)
    scales = overlap.get("scales") or {}
    reference = safe_float(scales.get(names[0], 1.0), 1.0)
    basf_start = []
    for name in names[1:]:
        relative = reference / safe_float(scales.get(name, 1.0), 1.0)
        # I_j = BASF_j * I_1 on a common model: the overlap scales multiply I_j onto crystal 1.
        basf_start.append(float(relative) if math.isfinite(relative) and relative > 0.0 else 1.0)
    write_fixed_width_batch_hkl(ep_dir / f"{stem}.hkl", rows, batches)
    kappa_effective = ABSOLUTE_HKL_KAPPA * hkl_safe_factor
    osf = math.sqrt(kappa_effective) if args.rescale == "absolute" else None
    write_no_exti_ins(
        Path(args.res),
        ep_dir / f"{stem}.ins",
        f"{stem}.hkl",
        wght=endpoint_initial_wght(args.endpoint_wght_mode),
        ls_cycles=args.endpoint_ls_cycles,
        keep_damp=bool(args.keep_damp),
        osf=osf,
        basf=basf_start,
        merg=0,
        **endpoint_native_outputs(args),
    )
    shelxl = run_shelxl(
        ep_dir,
        stem,
        shelxl_executable,
        iterations=args.shelxl_iterations,
        wght_mode=args.endpoint_wght_mode,
        endpoint_ls_cycles=args.endpoint_ls_cycles,
        convergence_shift_esd=args.shelxl_convergence_shift_esd,
        npd_recovery=args.shelxl_npd_recovery,
        keep_damp=bool(args.keep_damp),
        wght_converge_max_calls=args.wght_converge_max_calls,
        allow_basf=True,
    )
    fail_if_shelxl_became_unavailable([{"shelxl": shelxl}], context="unmerged multi-crystal endpoint refinement")
    res, lst, hkl = ep_dir / f"{stem}.res", ep_dir / f"{stem}.lst", ep_dir / f"{stem}.hkl"
    basf = refined_basf(res, lst)
    modulation, _per_dataset = audit_combined_endpoint_modulation(
        {name: corrections_by_dataset.get(name) for name in names},
        {name: 1.0 for name in names},
        endpoint_source=args.endpoint_source,
    )
    return {
        "status": "ok" if str(shelxl.get("status", "")) == "ok" else str(shelxl.get("status", "")),
        "merge": "unmerged_batch",
        "definition": (
            "every corrected work row unmerged (MERG 0, Friedel opposites kept), HKLF 4 batch number = crystal "
            "index in columns 29-32, one refined BASF per extra crystal (scale of batch m relative to batch 1)"
        ),
        "lambda_q": args.endpoint_lambda,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "rows": len(rows),
        "batches": batch_of,
        "hkl_safe_factor": float(hkl_safe_factor),
        **modulation,
        "sigma_policy": {"dataset": args.dataset_sigma_policy, "post_bloch_sigma_scale": sigma_scale},
        "basf_start": basf_start,
        "basf_start_source": "overlap estimate of the relative intensity scale (estimate_overlap_scales)",
        "basf": basf,
        "basf_sanity": basf_sanity(basf_start, basf),
        "endpoint_wght_mode": args.endpoint_wght_mode,
        "intensity_scale": {
            "mode": args.rescale,
            "kappa_effective": kappa_effective if args.rescale == "absolute" else None,
            "osf_reset": osf,
        },
        "shelxl": shelxl,
        "artifacts": {
            "directory": str(ep_dir.resolve()),
            "res": str(res.resolve()),
            "lst": str(lst.resolve()),
            "hkl": str(hkl.resolve()),
            "res_sha256": sha256(res) if res.exists() else "",
            "lst_sha256": sha256(lst) if lst.exists() else "",
            "hkl_sha256": sha256(hkl) if hkl.exists() else "",
        },
    }


FREE_VALIDATION_DIR = "free_validation"


def free_validation_rows_by_dataset(
    free_inputs: dict[str, dict],
    args,
    model,
    *,
    dataset_sigma: dict[str, dict],
    sigma_model,
    weight_by_name: dict[str, float],
) -> tuple[dict[str, list[tuple]], dict[str, dict]]:
    """Corrected free rows of every crystal, built exactly like its work rows; nothing is fitted on them.

    Same final-model q, kappa scale and q-exclusion rule (the work rows' divisor median is the reference), the
    per-crystal sigma policy as fitted on the work rows (error model: K, k, u against ``sigma_model``; uniform_z_rms:
    the crystal's sigma factor) and the same dataset weight.  Returns the rows before the inter-dataset scale.
    """

    rows_by_dataset: dict[str, list[tuple]] = {}
    audit: dict[str, dict] = {}
    for name, item in free_inputs.items():
        correction: dict = {}
        rows, _corrections = endpoint_rows_and_corrections_from_native(
            item["rows"],
            float(args.endpoint_lambda),
            q_policy=args.endpoint_q_policy,
            rescale=args.rescale,
            model=model if args.rescale == "absolute" else None,
            audit=correction,
            reference=item["work_correction"],
        )
        record: dict = {
            "input_free_rows": len(item["rows"]),
            "corrected_rows": len(rows),
            "correction": correction,
            "sigma_policy": args.dataset_sigma_policy,
        }
        meta = dataset_sigma.get(name) or {}
        if args.dataset_sigma_policy == "error_model":
            if str(meta.get("status", "")) != "ok":
                record["reason"] = f"no fitted error model for this crystal (status {meta.get('status')!r})"
                rows = []
            elif rows:
                rows, record["scoring"] = apply_error_model_sigma(rows, sigma_model, meta)
        elif args.dataset_sigma_policy == "uniform_z_rms" and str(meta.get("status", "")).startswith("ok"):
            rows = scale_row_sigmas(rows, float(meta["sigma_scale"]))
            record["sigma_scale"] = float(meta["sigma_scale"])
            if rows and math.isfinite(safe_float(meta.get("model_intensity_scale"))):
                record["scoring"] = rms_z_against_model(rows, model, float(meta["model_intensity_scale"]))
        weight = float(weight_by_name.get(name, 1.0))
        rows = apply_hkl_weight(rows, weight)
        record["endpoint_weight_applied_to_sigma"] = weight
        record["rows"] = len(rows)
        rows_by_dataset[name] = rows
        audit[name] = record
    return rows_by_dataset, audit


def fixed_width_hkl_indices(path: Path) -> list[tuple[int, int, int]]:
    """h, k, l of every reflection of a fixed-format HKLF 4 file (columns 1-12) before the 0 0 0 terminator."""

    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if len(line) < 12 or not line[:12].strip():
            continue
        hkl = (int(line[0:4]), int(line[4:8]), int(line[8:12]))
        if hkl == (0, 0, 0):
            break
        out.append(hkl)
    return out


FREE_RMS_Z_DEFINITION = (
    "RMS of z = (I - K Fc^2) / sigma over the corrected free rows of every crystal, with that crystal's work-fitted "
    "sigma model and its K (per-crystal error model or uniform factor; before the dataset weight and the merge)"
)


def _pooled_free_rms_z(audit: dict[str, dict]) -> float | None:
    """Return the row-weighted RMS of the per-crystal free RMS(z), or None without scored rows."""

    total, count = 0.0, 0
    for record in audit.values():
        scoring = record.get("scoring") or {}
        n, rms = int(scoring.get("rows", 0) or 0), safe_float(scoring.get("rms_z"))
        if n and math.isfinite(rms):
            total += n * rms * rms
            count += n
    return math.sqrt(total / count) if count else None


def _fixed_width_rows(rows: list[tuple], factor: float) -> tuple[list[tuple], float | None]:
    """Rows times the endpoint's HKL factor; an extra factor below 1 when a row would still overflow F8.2."""

    scaled = [(h, k, ell, intensity * factor, sig * factor) for h, k, ell, intensity, sig in rows]
    extra = safe_fixed_width_factor(scaled) if scaled else 1.0
    if extra >= 1.0:
        return scaled, None
    return [(h, k, ell, intensity * extra, sig * extra) for h, k, ell, intensity, sig in scaled], extra


def _hkl_lines(rows: list[tuple]) -> list[str]:
    """Format rows as fixed-width HKLF 4 lines without newlines."""

    return [format_fixed_width_row(*row).rstrip("\n") for row in rows]


def merged_endpoint_free_validation(
    ep_dir: Path,
    *,
    free_rows_by_dataset: dict[str, list[tuple]],
    free_audit: dict[str, dict],
    interdataset_scales: dict[str, float],
    merge_sigma_policy: str,
    merge_key,
    post_bloch_sigma: dict,
    hkl_safe_factor: float,
    merged_work_rows: list[tuple],
    refined_res: Path,
    work_hkl: Path,
    endpoint_shelxl: dict,
    shelxl_executable: str,
    wght_line: str | None = None,
) -> dict:
    """free_validation of the merged combined endpoint: the free rows merged like the work rows, scored at L.S. 0.

    ``wght_line`` replaces the model's active WGHT in the scoring run (None keeps the .res's own, which a refinement
    of this engine writes as the WGHT its last call applied).
    """

    unmerged: list[tuple] = []
    for name, rows in free_rows_by_dataset.items():
        unmerged.extend(scale_hkl_rows(rows, safe_float(interdataset_scales.get(name, 1.0), float("nan"))))
    record: dict = {
        "endpoint": "merged",
        "free_rows": {"per_dataset": free_audit, "unmerged_rows": len(unmerged)},
        "free_rms_z": _pooled_free_rms_z(free_audit),
        "free_rms_z_definition": FREE_RMS_Z_DEFINITION,
    }
    if not unmerged:
        return {**record, "status": "skipped_no_free_rows"}
    merged, merge_meta = merge_hkl_rows(unmerged, merge_sigma_policy, canonicalizer=merge_key)
    if str(post_bloch_sigma.get("status", "")).startswith("ok"):
        merged = scale_row_sigmas(merged, float(post_bloch_sigma["sigma_scale"]))
        record["post_bloch_sigma_scale"] = float(post_bloch_sigma["sigma_scale"])
    work_keys = {(int(h), int(k), int(ell)) for h, k, ell, _i, _s in merged_work_rows}
    free_keys = {(int(h), int(k), int(ell)) for h, k, ell, _i, _s in merged}
    rows, extra = _fixed_width_rows(merged, float(hkl_safe_factor))
    directory = ep_dir / FREE_VALIDATION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    write_hkl(directory / "free.hkl", rows)
    record["free_rows"].update(
        {
            "merged_rows": len(merged),
            "merge_sigma": merge_meta,
            "merge_key_classes_also_in_work_hkl": len(free_keys & work_keys),
            "hkl_safe_factor": float(hkl_safe_factor),
            "extra_hkl_factor": extra,
        }
    )
    scored = score_model_on_work_and_free(
        directory, res=refined_res, work_hkl=work_hkl, free_lines=_hkl_lines(rows), shelxl=shelxl_executable,
        work_rescale=extra, wght_line=wght_line,
    )
    record_endpoint_reproduction(scored, endpoint_shelxl)
    if record["free_rows"]["merge_key_classes_also_in_work_hkl"]:
        scored["status"] = "free_reflections_overlap_work"
    artifacts = {
        "directory": str(directory.resolve()),
        "free_hkl": str((directory / "free.hkl").resolve()),
        "free_hkl_sha256": sha256(directory / "free.hkl"),
        "validation_hkl": str((directory / "validation.hkl").resolve()),
    }
    return {**record, **scored, "artifacts": artifacts}


def unmerged_endpoint_free_validation(
    unmerged_endpoint: dict,
    *,
    free_rows_by_dataset: dict[str, list[tuple]],
    free_audit: dict[str, dict],
    post_bloch_sigma: dict,
    merge_key,
    shelxl_executable: str,
) -> dict:
    """free_validation of the unmerged batch endpoint: every free row, batch = crystal, scored with BASF fixed.

    free_unmerged.hkl carries the batch numbers.  In the scoring run a free row of batch m > 1 is flagged -1 and put on
    batch 1's scale by dividing I and sigma by the refined BASF(m - 1), SHELXL's own correction of batch m's Fo^2
    (its .fcf and R1 show Fo^2 / BASF).
    """

    artifacts = unmerged_endpoint.get("artifacts") or {}
    batches = {str(name): int(batch) for name, batch in (unmerged_endpoint.get("batches") or {}).items()}
    basf = [float(value) for value in ((unmerged_endpoint.get("basf") or {}).get("values") or [])]
    sigma_scale = float(post_bloch_sigma["sigma_scale"]) if str(post_bloch_sigma.get("status", "")).startswith("ok") else 1.0
    rows: list[tuple] = []
    row_batches: list[int] = []
    for name, dataset_rows in free_rows_by_dataset.items():
        if name not in batches:
            continue
        scaled = scale_row_sigmas(dataset_rows, sigma_scale) if sigma_scale != 1.0 else dataset_rows
        rows.extend(scaled)
        row_batches.extend([batches[name]] * len(scaled))
    record: dict = {
        "endpoint": "unmerged_batch",
        "free_rows": {"per_dataset": free_audit, "unmerged_rows": len(rows), "batches": batches, "basf": basf},
        "free_rms_z": _pooled_free_rms_z(free_audit),
        "free_rms_z_definition": FREE_RMS_Z_DEFINITION,
    }
    if not rows:
        return {**record, "status": "skipped_no_free_rows"}
    if max(row_batches) - 1 > len(basf):
        return {**record, "status": "failed", "error": f"{len(basf)} refined BASF values for batches {sorted(set(row_batches))}"}
    work_hkl = Path(str(artifacts.get("hkl") or ""))
    work_keys = {merge_key(*hkl) for hkl in fixed_width_hkl_indices(work_hkl)}
    free_keys = {merge_key(int(h), int(k), int(ell)) for h, k, ell, _i, _s in rows}
    factor = float(unmerged_endpoint.get("hkl_safe_factor", 1.0))
    written = [(h, k, ell, intensity * factor, sig * factor) for h, k, ell, intensity, sig in rows]
    directory = Path(str(artifacts.get("directory"))) / FREE_VALIDATION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    on_batch1 = [
        (h, k, ell, intensity / (basf[batch - 2] if batch > 1 else 1.0), sig / (basf[batch - 2] if batch > 1 else 1.0))
        for (h, k, ell, intensity, sig), batch in zip(rows, row_batches, strict=True)
    ]
    scaled_batch1, extra = _fixed_width_rows(on_batch1, factor)
    if extra is not None:
        written = [(h, k, ell, intensity * extra, sig * extra) for h, k, ell, intensity, sig in written]
    write_fixed_width_batch_hkl(directory / "free_unmerged.hkl", written, row_batches)
    record["free_rows"].update(
        {
            "merge_key_classes_also_in_work_hkl": len(free_keys & work_keys),
            "hkl_safe_factor": factor,
            "extra_hkl_factor": extra,
            "batch_scale_in_scoring_run": "I and sigma of batch m > 1 divided by BASF(m - 1); flagged -1",
        }
    )
    scored = score_model_on_work_and_free(
        directory, res=Path(str(artifacts.get("res"))), work_hkl=work_hkl, free_lines=_hkl_lines(scaled_batch1),
        shelxl=shelxl_executable, work_rescale=extra, stem="validation_unmerged",
    )
    record_endpoint_reproduction(scored, unmerged_endpoint.get("shelxl") or {})
    if record["free_rows"]["merge_key_classes_also_in_work_hkl"]:
        scored["status"] = "free_reflections_overlap_work"
    artifacts_out = {
        "directory": str(directory.resolve()),
        "free_hkl": str((directory / "free_unmerged.hkl").resolve()),
        "free_hkl_sha256": sha256(directory / "free_unmerged.hkl"),
        "validation_hkl": str((directory / "validation_unmerged.hkl").resolve()),
    }
    return {**record, **scored, "artifacts": artifacts_out}


def free_validation_report_line(label: str, block: dict) -> str:
    """One REPORT.md line of an endpoint's held-out validation (reporting only)."""

    if not isinstance(block.get("free"), dict):
        return f"Held-out SHELXL validation ({label}, reporting only): status `{block.get('status')}` {block.get('error', '')}".rstrip() + "."

    def text(record: dict) -> str:
        """Format one SHELXL record as N, R1gt, R1all, wR2 and GooF."""

        return (
            f"N {record.get('N_all')}, R1gt {100 * safe_float(record.get('R1_gt')):.2f}%, R1all "
            f"{100 * safe_float(record.get('R1_all')):.2f}%, wR2 {100 * safe_float(record.get('wR2_all')):.2f}%, GooF "
            f"{safe_float(record.get('GooF')):.3f}"
        )

    return (
        f"Held-out SHELXL validation ({label}, L.S. 0, reporting only): status `{block.get('status')}`; free "
        f"{text(block['free'])}; work {text(block['work'])}."
    )


def _free_validation_failure(exc: Exception) -> dict:
    """Return the reporting-only record of a held-out validation that failed."""

    return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "role": "reporting only"}


SCORE_ONLY_STATISTICS = ("status", "R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all", "active_wght_a", "active_wght_b")


def score_only_inputs(scored_res: Path, rebuilt_hkl: Path) -> dict:
    """What a ``--score-res`` score needs, checked: the model's own HKL is the rebuilt one, and its applied WGHT.

    The model's HKL, listing and .res sit together (``multi_dyn_corrected.{res,hkl,lst}``).  The applied WGHT is the
    one the model's last refinement call applied: the WGHT instruction its listing echoes.  This engine writes the same
    WGHT into the kept .res before END (SHELXL's suggestion for a next call goes after END), so the .res line is used and
    must agree with the listing.  Status ``ok`` only when the rebuilt HKL has the same bytes as the model's HKL, the
    listing has complete statistics and the two WGHTs agree.
    """

    scored_res, rebuilt_hkl = Path(scored_res), Path(rebuilt_hkl)
    scored_hkl, scored_lst = scored_res.with_suffix(".hkl"), scored_res.with_suffix(".lst")
    record: dict = {
        "scored_res": str(scored_res.resolve()),
        "scored_res_sha256": sha256(scored_res),
        "scored_hkl": str(scored_hkl.resolve()),
        "scored_hkl_sha256": sha256(scored_hkl) if scored_hkl.is_file() else "",
        "scored_lst": str(scored_lst.resolve()),
        "scored_lst_sha256": sha256(scored_lst) if scored_lst.is_file() else "",
        "rebuilt_hkl": str(rebuilt_hkl.resolve()),
        "rebuilt_hkl_sha256": sha256(rebuilt_hkl),
    }
    record["work_hkl_bytes_same"] = bool(record["scored_hkl_sha256"]) and record["scored_hkl_sha256"] == record["rebuilt_hkl_sha256"]
    listing = parse_lst(scored_lst)
    record["listing_statistics"] = {key: listing.get(key) for key in SCORE_ONLY_STATISTICS}
    wght = active_wght_line(scored_res.read_text(errors="replace"))
    res_terms = parse_shelx_wght_terms(wght or "") if wght else []
    applied = listing.get("active_wght_terms") or []
    record["wght"] = wght
    record["wght_applied_in_listing"] = list(applied)
    record["wght_same"] = bool(res_terms) and bool(applied) and all(
        abs(float(x) - float(y)) <= 5e-5 for x, y in zip((list(res_terms) + [0.0])[:2], (list(applied) + [0.0])[:2], strict=True)
    )
    if not record["work_hkl_bytes_same"]:
        record["status"] = "work_hkl_not_reproduced"
    elif listing.get("status") != "ok" or listing.get("R1_all") is None or listing.get("N_all") is None:
        record["status"] = f"scored_model_listing_{listing.get('status')}"
    elif not record["wght_same"]:
        record["status"] = "applied_wght_not_determined"
    else:
        record["status"] = "ok"
    return record


def main():
    """Concatenate the per-dataset native row tables and write the combined endpoint and report.

    Reports joint native metrics, writes the combined HKL and SHELXL endpoint, combined_summary.json and
    REPORT.md to --out-dir.
    """

    started_monotonic = time.perf_counter()
    ap = build_combination_parser(
        description=__doc__,
    )
    args = ap.parse_args()

    try:
        endpoint_lambda_value = float(args.endpoint_lambda)
    except (TypeError, ValueError) as exc:
        ap.error(f"--endpoint-lambda must be finite: {exc}")
    if not math.isfinite(endpoint_lambda_value):
        ap.error("--endpoint-lambda must be finite")
    if not math.isfinite(float(args.min_d_spacing)) or float(args.min_d_spacing) < 0.0:
        ap.error("--endpoint-min-d-spacing/--min-d-spacing must be finite and nonnegative")
    # Native endpoint directories use ``str(float(lambda))``.  Canonicalizing
    # here keeps values such as ``1`` and ``1.0`` on one artifact identity.
    args.endpoint_lambda = str(endpoint_lambda_value)
    if args.dataset_sigma_policy == "error_model" and args.endpoint_include_free_rows:
        ap.error(
            "--dataset-sigma-policy error_model is fitted on work rows only and needs --no-endpoint-include-free-rows "
            "(free rows in the HKL would enter the fit); use uniform_z_rms with the legacy free-row HKL"
        )
    if args.dataset_sigma_policy == "error_model" and args.endpoint_source != "native":
        ap.error("--dataset-sigma-policy error_model needs --endpoint-source native (row splits and the current model)")
    if args.dataset_sigma_policy == "error_model" and args.post_bloch_sigma_policy != "none":
        ap.error(
            "--dataset-sigma-policy error_model already sets per-crystal sigmas; use --post-bloch-sigma-policy none "
            "(a second uniform rescale of the merged HKL would undo the relative weights)"
        )
    free_validation = str(getattr(args, "endpoint_free_validation", "off")) == "on"
    if free_validation and args.endpoint_include_free_rows:
        ap.error(
            "--endpoint-free-validation on needs --no-endpoint-include-free-rows (with the free rows in the refinement "
            "HKL nothing is held out)"
        )
    if free_validation and args.endpoint_source != "native":
        ap.error("--endpoint-free-validation on needs --endpoint-source native (free rows are corrected from the row tables)")
    if args.endpoint_wght_mode == "converge" and not float(args.shelxl_convergence_shift_esd or 0.0) > 0.0:
        ap.error("--endpoint-wght-mode converge needs --shelxl-convergence-shift-esd > 0 (AUTO 0.01)")
    try:
        parse_zerr_esds(args.zerr_esds)
    except ValueError as exc:
        ap.error(str(exc))
    score_res = str(getattr(args, "score_res", "") or "").strip()
    score_pass = int(getattr(args, "score_error_model_pass", 1) or 1)
    score_sigma_model = str(getattr(args, "score_sigma_model", "") or "").strip()
    if score_res:
        refit_passes = 1 + (max(0, int(args.error_model_refits)) if args.dataset_sigma_policy == "error_model" else 0)
        if not free_validation:
            ap.error("--score-res needs --endpoint-free-validation on (it only scores)")
        if args.unmerged_endpoint:
            ap.error("--score-res scores the merged endpoint only; drop --unmerged-endpoint")
        if not Path(score_res).is_file():
            ap.error(f"--score-res: no file {score_res}")
        if not 1 <= score_pass <= refit_passes:
            ap.error(f"--score-error-model-pass {score_pass}: this command's error model has passes 1..{refit_passes}")
        if (score_pass > 1) != bool(score_sigma_model):
            ap.error("--score-sigma-model is the fitted model of a kept pass above 1, and only of such a pass")
        if score_sigma_model and not Path(score_sigma_model).is_file():
            ap.error(f"--score-sigma-model: no file {score_sigma_model}")
    elif score_sigma_model or score_pass != 1:
        ap.error("--score-error-model-pass and --score-sigma-model need --score-res")

    structure_path = Path(args.res)
    model = parse_model(structure_path)
    symmetry_ops = model.get("symmetry_ops", [])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    dataset_summaries = []
    endpoint_rows_by_dataset: dict[str, list[tuple]] = {}
    endpoint_corrections_by_dataset: dict[str, np.ndarray | None] = {}
    endpoint_weight_by_name: dict[str, float] = {}
    dataset_sigma_by_name: dict[str, dict] = {}
    free_endpoint_rows_by_dataset: dict[str, list[tuple]] = {}
    free_validation_inputs: dict[str, dict] = {}
    per_dataset_wght: dict[str, dict[str, float]] = {}
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must be name=/path")
        name, path_text = item.split("=", 1)
        run = Path(path_text)
        row_table_path = run / "native_dyn_calc.csv"
        summary_path = run / "summary.json"
        summary = read_json(summary_path)
        source_wght_a, source_wght_b = selected_native_wght(summary)
        dataset_wght_a = float(args.wght_a) if args.wght_a is not None else source_wght_a
        dataset_wght_b = float(args.wght_b) if args.wght_b is not None else source_wght_b
        per_dataset_wght[name] = {"a": dataset_wght_a, "b": dataset_wght_b}
        unfiltered_rows = read_rows(run, name)
        rows, resolution_filter = filter_native_rows_by_resolution(
            unfiltered_rows,
            model,
            float(args.min_d_spacing),
        )
        role = str(rows[0].get("dataset_role", "primary")) if rows else "missing"
        raw_weight = safe_float(rows[0].get("raw_row_weight"), 1.0) if rows else float("nan")
        endpoint_weight = safe_float(rows[0].get("endpoint_weight"), raw_weight) if rows else float("nan")
        native_rows = []
        if role not in {"completeness_only", "exclude"} and raw_weight > 0.0:
            for row in rows:
                weighted = apply_native_row_weight(row, raw_weight)
                weighted["native_wght_a"] = f"{dataset_wght_a:.17g}"
                weighted["native_wght_b"] = f"{dataset_wght_b:.17g}"
                native_rows.append(weighted)
        endpoint_input_rows = [
            r
            for r in rows
            if str(r.get("dataset_role", "primary")) != "exclude"
            and endpoint_weight > 0.0
        ]
        free_rows_excluded = 0
        free_input_rows = [r for r in endpoint_input_rows if str(r.get("split", "work")) == "free"]
        if not args.endpoint_include_free_rows:
            # Free rows stay in the row tables and joint metrics (predicted and scored every cycle) but are
            # never refined against, so SHELXL-side and row-side R_free stay honest.
            kept_rows = [r for r in endpoint_input_rows if str(r.get("split", "work")) == "work"]
            free_rows_excluded = len(endpoint_input_rows) - len(kept_rows)
            endpoint_input_rows = kept_rows
        source_best_native = summary.get("best_native", {})
        if not isinstance(source_best_native, dict):
            source_best_native = {}
        filtered_native_metrics = native_metrics_by_split(
            native_rows,
            dataset_wght_a,
            dataset_wght_b,
        )
        filtered_best_native = dict(source_best_native)
        filtered_best_native["metrics"] = filtered_native_metrics
        filtered_best_native["metrics_contract"] = {
            "scope": "combined_shared_cell_resolution_filtered_rows",
            "cell_source": "shared_structure_res",
            "requested_min_d_spacing_A": float(args.min_d_spacing),
            "input_rows": int(resolution_filter.get("input_rows", 0)),
            "retained_resolution_rows": len(rows),
            "rows_used": len(native_rows),
            "dataset_role": role,
            "raw_row_weight": raw_weight,
            "raw_row_weight_applied_to_sigma": bool(
                native_rows and math.isfinite(raw_weight) and abs(raw_weight - 1.0) > 1e-12
            ),
            "native_wght_a": dataset_wght_a,
            "native_wght_b": dataset_wght_b,
        }
        all_rows.extend(native_rows)
        exact = {tuple(int(r[x]) for x in ("h", "k", "l")) for r in rows}
        canon = {canonical_model_friedel(*hkl, symmetry_ops) for hkl in exact}
        q = np.asarray([safe_float(r["q_dyn_over_born"]) for r in endpoint_input_rows], dtype=float)
        basis = np.asarray([safe_float(r.get("basis_size")) for r in endpoint_input_rows], dtype=float)
        basis = basis[np.isfinite(basis)]
        endpoint_weight_by_name[name] = endpoint_weight
        dataset_summaries.append(
            {
                "name": name,
                "run_dir": str(run.resolve()),
                "source_artifacts": {
                    "row_table": file_provenance(row_table_path),
                    "summary": file_provenance(summary_path),
                },
                "scale_contract": scale_contract_provenance(summary, unfiltered_rows),
                "resolution_filter": resolution_filter,
                "rows": len(rows),
                "native_rows": len(native_rows),
                "endpoint_rows": len(endpoint_input_rows),
                "endpoint_free_rows_excluded": free_rows_excluded,
                "dataset_role": role,
                "raw_row_weight": raw_weight,
                "endpoint_weight": endpoint_weight,
                "native_wght_a": dataset_wght_a,
                "native_wght_b": dataset_wght_b,
                "unique_signed_hkl": len(exact),
                "unique_symmetry_friedel": len(canon),
                "unique_p222_friedel": len(canon),
                # Keep the historical field useful to downstream readers, but
                # project its metrics onto the exact rows used by this combined
                # run.  The verbatim source block remains available separately
                # for nuisance-model provenance and before-cutoff comparisons.
                "best_native_summary": filtered_best_native,
                "resolution_filtered_native_metrics": filtered_native_metrics,
                "source_best_native_summary": source_best_native,
                "basis_size_stats": {
                    "min": int(np.min(basis)) if len(basis) else 0,
                    "median": float(np.median(basis)) if len(basis) else 0.0,
                    "p95": float(np.percentile(basis, 95)) if len(basis) else 0.0,
                    "p99": float(np.percentile(basis, 99)) if len(basis) else 0.0,
                    "max": int(np.max(basis)) if len(basis) else 0,
                },
                "q_median": float(np.nanmedian(q)) if len(q) else float("nan"),
                "q_p05": float(np.nanpercentile(q, 5)) if len(q) else float("nan"),
                "q_p95": float(np.nanpercentile(q, 95)) if len(q) else float("nan"),
            }
        )
        if args.endpoint_source == "hkl":
            source_hkl_path = endpoint_hkl_path(run, args.endpoint_lambda)
            hkl_rows = read_hkl(source_hkl_path)
            source_run_text = (summary.get("dataset_policy") or {}).get("run_dir", "")
            source_run = Path(source_run_text) if source_run_text else None
            if not hkl_rows and source_run and source_run.exists():
                source_hkl_path = endpoint_hkl_path(source_run, args.endpoint_lambda)
                hkl_rows = read_hkl(source_hkl_path)
            hkl_input_count = len(hkl_rows)
            hkl_rows = filter_hkl_rows_by_resolution(hkl_rows, model, float(args.min_d_spacing))
            dataset_summaries[-1]["source_artifacts"]["endpoint_hkl"] = file_provenance(source_hkl_path)
            dataset_summaries[-1]["endpoint_hkl_resolution_filter"] = {
                "requested_min_d_spacing_A": float(args.min_d_spacing),
                "input_rows": hkl_input_count,
                "kept_rows": len(hkl_rows),
                "retained_rows": len(hkl_rows),
                "dropped_rows": hkl_input_count - len(hkl_rows),
                **hkl_resolution_stats(hkl_rows, model),
            }
            endpoint_rows_by_dataset[name] = hkl_rows
            endpoint_corrections_by_dataset[name] = None
        else:
            correction_audit: dict = {}
            endpoint_rows, endpoint_corrections = endpoint_rows_and_corrections_from_native(
                endpoint_input_rows,
                float(args.endpoint_lambda),
                q_policy=args.endpoint_q_policy,
                rescale=args.rescale,
                model=model if args.rescale == "absolute" else None,
                audit=correction_audit,
            )
            dataset_summaries[-1]["endpoint_correction"] = correction_audit
            endpoint_rows_by_dataset[name] = endpoint_rows
            endpoint_corrections_by_dataset[name] = endpoint_corrections
            if free_validation:
                free_validation_inputs[name] = {"rows": free_input_rows, "work_correction": dict(correction_audit)}
            if args.dataset_sigma_policy == "error_model" and free_input_rows:
                # Corrected free rows (their own q-ratio median) are scored by the fitted error model and
                # reported; they are never fitted and never written to an HKL.
                free_endpoint_rows_by_dataset[name], _free_corrections = endpoint_rows_and_corrections_from_native(
                    free_input_rows,
                    float(args.endpoint_lambda),
                    q_policy=args.endpoint_q_policy,
                    rescale=args.rescale,
                    model=model if args.rescale == "absolute" else None,
                )
        dataset_sigma_by_name[name] = {"policy": args.dataset_sigma_policy, "status": "not_requested"}

    uncalibrated_rows_by_dataset = dict(endpoint_rows_by_dataset)
    policy_weight_by_name = dict(endpoint_weight_by_name)
    if args.dataset_sigma_policy == "error_model" and getattr(args, "error_model_dataset_weight", "apply") == "off":
        # The error model already weights each crystal by its own misfit; a fractional policy weight on top would
        # count that crystal's error twice.  Weight 0 (excluded) crystals never reach this point with rows.
        endpoint_weight_by_name = {name: (1.0 if weight > 0.0 else weight) for name, weight in endpoint_weight_by_name.items()}
    for item in dataset_summaries:
        applied = endpoint_weight_by_name.get(item["name"], item.get("endpoint_weight"))
        item["endpoint_weight_applied_to_shelxl_sigma"] = applied
        item["endpoint_weight_policy"] = (
            "error_model_only"
            if args.dataset_sigma_policy == "error_model" and getattr(args, "error_model_dataset_weight", "apply") == "off"
            else "policy_weight_applied"
        )
        item["endpoint_weight_from_dataset_policy"] = policy_weight_by_name.get(item["name"], item.get("endpoint_weight"))
    if args.dataset_sigma_policy == "error_model":
        endpoint_rows_by_dataset, dataset_sigma_by_name = error_model_rows_by_dataset(
            uncalibrated_rows_by_dataset, model, free_endpoint_rows_by_dataset, endpoint_weight_by_name, weighted=False
        )
    if args.dataset_sigma_policy == "uniform_z_rms":
        for name, rows_for_dataset in list(endpoint_rows_by_dataset.items()):
            calibrated, sigma_meta = calibrate_dataset_endpoint_rows(
                rows_for_dataset,
                model,
                policy=args.dataset_sigma_policy,
                holdout_fraction=args.dataset_sigma_holdout_fraction,
                min_scale=args.dataset_sigma_min_scale,
                max_scale=args.dataset_sigma_max_scale,
                holdout_floor=bool(args.dataset_sigma_holdout_floor),
            )
            endpoint_rows_by_dataset[name] = calibrated
            dataset_sigma_by_name[name] = sigma_meta

    for name, rows_for_dataset in list(endpoint_rows_by_dataset.items()):
        endpoint_rows_by_dataset[name] = apply_hkl_weight(
            rows_for_dataset,
            endpoint_weight_by_name.get(name, 1.0),
        )

    wght_a = 0.126534 if args.wght_a is None else float(args.wght_a)
    wght_b = 0.0 if args.wght_b is None else float(args.wght_b)
    endpoint_key_for = endpoint_key_function(args.endpoint_merge_key, symmetry_ops)
    if args.rescale == "absolute" and args.endpoint_source == "hkl":
        raise ValueError("--rescale absolute needs --endpoint-source native (row predictions and the current model)")
    interdataset_scale_diagnostic: dict = {}
    if args.interdataset_scale == "overlap":
        interdataset_scale = estimate_overlap_scales(endpoint_rows_by_dataset, canonicalizer=endpoint_key_for)
    else:
        interdataset_scale = {
            "iterations": 0,
            "overlap_counts": {name: 0 for name in endpoint_rows_by_dataset},
            "scales": {name: 1.0 for name in endpoint_rows_by_dataset},
        }
        if args.rescale == "absolute" and len(endpoint_rows_by_dataset) > 1:
            # Every crystal is already on kappa*|F|^2; the overlap estimate is reported, not applied (expected ~1).
            interdataset_scale_diagnostic = estimate_overlap_scales(endpoint_rows_by_dataset, canonicalizer=endpoint_key_for)
    endpoint_rows = []
    for name, rows_for_dataset in endpoint_rows_by_dataset.items():
        dataset_scale = safe_float(interdataset_scale["scales"].get(name, 1.0), float("nan"))
        endpoint_rows.extend(scale_hkl_rows(rows_for_dataset, dataset_scale))
    combined_modulation, per_dataset_modulation = audit_combined_endpoint_modulation(
        endpoint_corrections_by_dataset,
        interdataset_scale["scales"],
        endpoint_source=args.endpoint_source,
    )
    for dataset_summary in dataset_summaries:
        name = str(dataset_summary.get("name", ""))
        dataset_summary["endpoint_modulation"] = per_dataset_modulation.get(name, {})
    exact_by_dataset = []
    canon_by_dataset = []
    for ds in dataset_summaries:
        rows = [r for r in all_rows if r["dataset"] == ds["name"]]
        exact_by_dataset.append({tuple(int(r[x]) for x in ("h", "k", "l")) for r in rows})
        canon_by_dataset.append({canonical_model_friedel(*tuple(int(r[x]) for x in ("h", "k", "l")), symmetry_ops) for r in rows})
    exact_union = set().union(*exact_by_dataset) if exact_by_dataset else set()
    canon_union = set().union(*canon_by_dataset) if canon_by_dataset else set()
    exact_intersection = set.intersection(*exact_by_dataset) if len(exact_by_dataset) >= 2 else set()
    canon_intersection = set.intersection(*canon_by_dataset) if len(canon_by_dataset) >= 2 else set()

    native = {
        "dyn": split_metrics(all_rows, "dyn_scaled", wght_a, wght_b),
        "born": split_metrics(all_rows, "born_scaled", wght_a, wght_b),
    }
    merged_endpoint = []
    merge_sigma = {"policy": args.merge_sigma_policy, "status": "skipped_no_endpoint_rows"}
    shelxl = {}
    endpoint_artifacts: dict[str, object] = {}
    score_only: dict[str, object] = {}
    endpoint_shelxl = require_combined_endpoint_shelxl(
        args.shelxl,
        endpoint_rows_available=bool(endpoint_rows),
    )
    if endpoint_rows:
        merged_endpoint, merge_sigma = merge_hkl_rows(endpoint_rows, args.merge_sigma_policy, canonicalizer=endpoint_key_for)
        merge_sigma["status"] = "ok"
        post_bloch_sigma = {"policy": args.post_bloch_sigma_policy, "status": "not_requested"}
        if args.post_bloch_sigma_policy == "uniform_z_rms":
            try:
                post_bloch_sigma = fit_uniform_z_rms_sigma_scale(
                    merged_endpoint,
                    model,
                    holdout_fraction=float(args.post_bloch_sigma_holdout_fraction),
                    min_scale=float(args.post_bloch_sigma_min_scale),
                    max_scale=float(args.post_bloch_sigma_max_scale),
                )
                if str(post_bloch_sigma.get("status", "")).startswith("ok"):
                    merged_endpoint = scale_row_sigmas(merged_endpoint, float(post_bloch_sigma["sigma_scale"]))
                    endpoint_rows = scale_row_sigmas(endpoint_rows, float(post_bloch_sigma["sigma_scale"]))
            except Exception as exc:
                post_bloch_sigma = {
                    "policy": "uniform_z_rms",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        n_over_f8_2 = sum(1 for _h, _k, _l, intensity, _sig in merged_endpoint if intensity > 99999.0)
        tag = args.endpoint_lambda.replace(".", "p")
        hkl_name = "multi_dyn_corrected.hkl"
        refits = max(0, int(args.error_model_refits)) if args.dataset_sigma_policy == "error_model" else 0

        def write_pass_hkl(ep_dir: Path, merged_rows, unmerged_rows):
            """Write one merged endpoint HKL (fixed-width safe); returns (hkl factor, merged rows, unmerged rows)."""

            safe_factor = min(safe_fixed_width_factor(unmerged_rows), safe_fixed_width_factor(merged_rows))
            if safe_factor < 1.0:
                unmerged_rows = [
                    (h, k, ell, intensity * safe_factor, sig * safe_factor) for h, k, ell, intensity, sig in unmerged_rows
                ]
                merged_rows = [
                    (h, k, ell, intensity * safe_factor, sig * safe_factor) for h, k, ell, intensity, sig in merged_rows
                ]
            ep_dir.mkdir(exist_ok=True)
            write_hkl(ep_dir / hkl_name, merged_rows)
            return safe_factor, merged_rows, unmerged_rows

        def refine_pass(ep_dir: Path, merged_rows, unmerged_rows, start_model: Path):
            """Write one merged endpoint from ``start_model`` and refine it; returns (shelxl, hkl factor, rows)."""

            safe_factor, merged_rows, unmerged_rows = write_pass_hkl(ep_dir, merged_rows, unmerged_rows)
            # Absolute scale: Fo^2 = kappa_eff*|F|^2 with kappa_eff = kappa*hkl_safe_factor, so the OSF restarts at
            # sqrt(kappa_eff) for every handoff instead of chasing a per-cycle median factor.
            pass_osf = math.sqrt(ABSOLUTE_HKL_KAPPA * safe_factor) if args.rescale == "absolute" else None
            write_no_exti_ins(
                start_model,
                ep_dir / "multi_dyn_corrected.ins",
                hkl_name,
                wght=endpoint_initial_wght(args.endpoint_wght_mode),
                ls_cycles=args.endpoint_ls_cycles,
                keep_damp=bool(args.keep_damp),
                osf=pass_osf,
                **endpoint_native_outputs(args),
            )
            result = run_shelxl(
                ep_dir,
                "multi_dyn_corrected",
                endpoint_shelxl,
                iterations=args.shelxl_iterations,
                wght_mode=args.endpoint_wght_mode,
                endpoint_ls_cycles=args.endpoint_ls_cycles,
                convergence_shift_esd=args.shelxl_convergence_shift_esd,
                npd_recovery=args.shelxl_npd_recovery,
                keep_damp=bool(args.keep_damp),
                wght_converge_max_calls=args.wght_converge_max_calls,
            )
            fail_if_shelxl_became_unavailable(
                [{"shelxl": result}],
                context="combined corrected-HKL endpoint refinement",
            )
            return result, safe_factor, merged_rows, unmerged_rows

        def refit_pass_state(pass_index: int, sigma_model_path: Path) -> dict:
            """Rows of error-model pass ``pass_index`` > 1: every crystal refitted against ``sigma_model_path``."""

            rows_by_dataset, sigma_by_name = error_model_rows_by_dataset(
                uncalibrated_rows_by_dataset,
                parse_model(sigma_model_path),
                free_endpoint_rows_by_dataset,
                endpoint_weight_by_name,
                weighted=True,
            )
            pass_rows = []
            for name, rows_for_dataset in rows_by_dataset.items():
                dataset_scale = safe_float(interdataset_scale["scales"].get(name, 1.0), float("nan"))
                pass_rows.extend(scale_hkl_rows(rows_for_dataset, dataset_scale))
            pass_merged, pass_merge_sigma = merge_hkl_rows(pass_rows, args.merge_sigma_policy, canonicalizer=endpoint_key_for)
            pass_merge_sigma["status"] = "ok"
            return {
                "pass": pass_index,
                "rows_by_dataset": rows_by_dataset,
                "rows": pass_rows,
                "merged": pass_merged,
                "merge_sigma": pass_merge_sigma,
                "sigma": sigma_by_name,
                "start": sigma_model_path,
            }

        start_model_path = Path(args.res)
        error_model_passes: list[dict] = []
        if score_res:
            # Validation only: the rows of the pass the scored endpoint kept, rebuilt without refining anything (a
            # kept pass above 1 was fitted against the previous pass's refined model, which the run recorded).
            if score_pass > 1:
                kept = refit_pass_state(score_pass, Path(score_sigma_model))
                endpoint_rows_by_dataset = kept["rows_by_dataset"]
                endpoint_rows = kept["rows"]
                merged_endpoint = kept["merged"]
                merge_sigma = kept["merge_sigma"]
                dataset_sigma_by_name = kept["sigma"]
                start_model_path = kept["start"]
            ep_dir = out / f"combined_endpoint_lam_{tag}"
            hkl_safe_factor, merged_endpoint, endpoint_rows = write_pass_hkl(ep_dir, merged_endpoint, endpoint_rows)
            endpoint_res = Path(score_res)
            endpoint_lst = endpoint_res.with_suffix(".lst")
            endpoint_hkl = ep_dir / hkl_name
            score_only = score_only_inputs(endpoint_res, endpoint_hkl)
            score_only.update({"error_model_pass": score_pass, "sigma_model": str(start_model_path)})
            endpoint_artifacts = {
                "directory": str(ep_dir.resolve()),
                "res": score_only["scored_res"],
                "lst": score_only["scored_lst"],
                "hkl": str(endpoint_hkl.resolve()),
                "res_sha256": score_only["scored_res_sha256"],
                "lst_sha256": score_only["scored_lst_sha256"],
                "hkl_sha256": score_only["rebuilt_hkl_sha256"],
            }
        if refits > 0 and not score_res:
            # Error-model passes: pass 1 fits every crystal against the entering model; each refit fits against the
            # model SHELXL refined in the previous pass (work rows only; the corrected intensities are unchanged)
            # and refines again from it.  Each pass has its own directory.  A pass that ends unsafe (NPD) stops the
            # refits and the last clean pass is kept, as the macro-cycles keep their last clean model.  The kept
            # pass is then written and refined again as the canonical endpoint (the same rows and start model).
            state = {
                "pass": 1,
                "rows_by_dataset": endpoint_rows_by_dataset,
                "rows": endpoint_rows,
                "merged": merged_endpoint,
                "merge_sigma": merge_sigma,
                "sigma": dataset_sigma_by_name,
                "start": start_model_path,
            }
            first_state, kept = state, None
            for pass_index in range(1, refits + 2):
                if pass_index > 1:
                    state = refit_pass_state(pass_index, start_model_path)
                pass_dir = out / f"combined_endpoint_lam_{tag}_error_model_pass{pass_index}"
                pass_shelxl, _factor, _merged, _rows = refine_pass(pass_dir, state["merged"], state["rows"], state["start"])
                record = {
                    "pass": pass_index,
                    "directory": str(pass_dir.resolve()),
                    "sigma_model": str(state["start"]),
                    "start_model": str(state["start"]),
                    "error_model": {
                        name: {key: meta.get(key) for key in ("k", "u", "model_intensity_scale", "work_rms_z_after", "free_rms_z_after")}
                        for name, meta in state["sigma"].items()
                    },
                    "shelxl": {key: pass_shelxl.get(key) for key in ("status", "R1_gt", "R1_all", "wR2_all", "GooF", "active_wght_a", "active_wght_b", "recommended_wght_a", "recommended_wght_b", "npd_atoms")},
                    "kept": False,
                }
                error_model_passes.append(record)
                refined_res = pass_dir / "multi_dyn_corrected.res"
                if str(pass_shelxl.get("status", "")) != "ok" or not refined_res.exists():
                    record["reason"] = f"SHELXL status {pass_shelxl.get('status')!r}: refits stop, the last clean pass is kept"
                    break
                kept = state
                start_model_path = refined_res
            if kept is None:
                # Not even pass 1 was clean: the canonical endpoint repeats pass 1 and reports its status.
                kept = first_state
            error_model_passes[kept["pass"] - 1]["kept"] = True
            endpoint_rows_by_dataset = kept["rows_by_dataset"]
            endpoint_rows = kept["rows"]
            merged_endpoint = kept["merged"]
            merge_sigma = kept["merge_sigma"]
            dataset_sigma_by_name = kept["sigma"]
            start_model_path = kept["start"]
        if not score_res:
            ep_dir = out / f"combined_endpoint_lam_{tag}"
            shelxl, hkl_safe_factor, merged_endpoint, endpoint_rows = refine_pass(
                ep_dir, merged_endpoint, endpoint_rows, start_model_path
            )
            endpoint_res = ep_dir / "multi_dyn_corrected.res"
            endpoint_lst = ep_dir / "multi_dyn_corrected.lst"
            endpoint_hkl = ep_dir / hkl_name
            endpoint_artifacts = {
                "directory": str(ep_dir.resolve()),
                "res": str(endpoint_res.resolve()),
                "lst": str(endpoint_lst.resolve()),
                "hkl": str(endpoint_hkl.resolve()),
                "res_sha256": sha256(endpoint_res) if endpoint_res.exists() else "",
                "lst_sha256": sha256(endpoint_lst) if endpoint_lst.exists() else "",
                "hkl_sha256": sha256(endpoint_hkl) if endpoint_hkl.exists() else "",
            }

    free_validation_rows: dict[str, list[tuple]] = {}
    free_validation_audit: dict[str, dict] = {}
    merged_free_validation: dict = {}
    if free_validation and endpoint_rows:
        try:
            # The error model of the kept pass was fitted against its start model (the entering model for pass 1).
            free_validation_rows, free_validation_audit = free_validation_rows_by_dataset(
                free_validation_inputs,
                args,
                model,
                dataset_sigma=dataset_sigma_by_name,
                sigma_model=parse_model(start_model_path) if args.dataset_sigma_policy == "error_model" else model,
                weight_by_name=endpoint_weight_by_name,
            )
            if score_res and score_only.get("status") != "ok":
                # the scored model's own HKL is not the one rebuilt here (or its applied WGHT is unknown): not scored
                merged_free_validation = {"status": str(score_only.get("status"))}
            elif not score_res and (str(shelxl.get("status", "")) != "ok" or not endpoint_res.exists()):
                merged_free_validation = {"status": f"skipped_endpoint_status_{shelxl.get('status')}"}
            else:
                merged_free_validation = merged_endpoint_free_validation(
                    ep_dir,
                    free_rows_by_dataset=free_validation_rows,
                    free_audit=free_validation_audit,
                    interdataset_scales=interdataset_scale["scales"],
                    merge_sigma_policy=args.merge_sigma_policy,
                    merge_key=endpoint_key_for,
                    post_bloch_sigma=post_bloch_sigma,
                    hkl_safe_factor=hkl_safe_factor,
                    merged_work_rows=merged_endpoint,
                    refined_res=endpoint_res,
                    work_hkl=endpoint_hkl,
                    endpoint_shelxl=score_only["listing_statistics"] if score_res else shelxl,
                    shelxl_executable=endpoint_shelxl,
                    wght_line=score_only["wght"] if score_res else None,
                )
            merged_free_validation["sigma_model"] = str(start_model_path)
        except Exception as exc:  # reporting only: a failed validation never fails the endpoint
            merged_free_validation = _free_validation_failure(exc)

    unmerged_endpoint: dict = {"status": "not_requested"}
    if args.unmerged_endpoint and endpoint_rows:
        unmerged_endpoint = run_unmerged_batch_endpoint(
            out,
            args,
            endpoint_rows_by_dataset,
            endpoint_corrections_by_dataset,
            post_bloch_sigma,
            endpoint_shelxl,
            endpoint_key_for,
        )
        if free_validation:
            try:
                if str(unmerged_endpoint.get("status", "")) != "ok":
                    unmerged_endpoint["free_validation"] = {
                        "status": f"skipped_endpoint_status_{unmerged_endpoint.get('status')}"
                    }
                else:
                    unmerged_endpoint["free_validation"] = unmerged_endpoint_free_validation(
                        unmerged_endpoint,
                        free_rows_by_dataset=free_validation_rows,
                        free_audit=free_validation_audit,
                        post_bloch_sigma=post_bloch_sigma,
                        merge_key=endpoint_key_for,
                        shelxl_executable=endpoint_shelxl,
                    )
            except Exception as exc:  # reporting only
                unmerged_endpoint["free_validation"] = _free_validation_failure(exc)

    endpoint_resolution = hkl_resolution_stats(merged_endpoint, model)
    resolution_filters = [dataset["resolution_filter"] for dataset in dataset_summaries]
    retained_native_minima = [
        safe_float(item.get("actual_retained_min_d_spacing_A"))
        for item in resolution_filters
        if math.isfinite(safe_float(item.get("actual_retained_min_d_spacing_A")))
    ]
    source_scale_counts = [
        dataset.get("scale_contract", {}).get("parameter_count")
        for dataset in dataset_summaries
    ]
    all_scale_counts_known = all(isinstance(value, int) for value in source_scale_counts)
    resolution_contract = {
        "requested_min_d_spacing_A": float(args.min_d_spacing),
        "cell_source": "shared_structure_res",
        "cell_A_deg": [float(value) for value in model["cell"]],
        "input_native_rows": int(sum(int(item.get("input_rows", 0)) for item in resolution_filters)),
        "retained_native_rows": int(sum(int(item.get("kept_rows", 0)) for item in resolution_filters)),
        "dropped_native_rows": int(sum(int(item.get("dropped_rows", 0)) for item in resolution_filters)),
        "actual_retained_native_min_d_spacing_A": min(retained_native_minima) if retained_native_minima else float("nan"),
        "actual_merged_endpoint_min_d_spacing_A": endpoint_resolution["actual_min_d_spacing_A"],
        "merged_endpoint_rows": endpoint_resolution["rows"],
    }
    compute_wall_seconds = float(time.perf_counter() - started_monotonic)

    result = {
        "provenance": {
            "structure": file_provenance(structure_path),
            "endpoint_source": args.endpoint_source,
            "resolution_contract": resolution_contract,
            "scale_contract": {
                "per_dataset": {
                    dataset["name"]: dataset["scale_contract"]
                    for dataset in dataset_summaries
                },
                "total_parameter_count": int(sum(source_scale_counts)) if all_scale_counts_known else None,
                "all_parameter_counts_known": all_scale_counts_known,
            },
        },
        "timing": {
            "wall_seconds": compute_wall_seconds,
            "compute_wall_seconds": compute_wall_seconds,
        },
        "runs": dataset_summaries,
        "wght": {
            "a": wght_a,
            "b": wght_b,
            "mode": "explicit_override" if args.wght_a is not None or args.wght_b is not None else "per_dataset_native",
            "per_dataset": per_dataset_wght,
        },
        "native_joint": native,
        "coverage": {
            "rows_total": len(all_rows),
            "endpoint_rows_total": len(endpoint_rows),
            "unique_signed_hkl_union": len(exact_union),
            "unique_signed_hkl_intersection": len(exact_intersection),
            "unique_p222_friedel_union": len(canon_union),
            "unique_p222_friedel_intersection": len(canon_intersection),
        },
        "interdataset_scale": {"mode": args.interdataset_scale, **interdataset_scale},
        "combined_endpoint": {
            "lambda_q": args.endpoint_lambda,
            "min_d_spacing_A": float(args.min_d_spacing),
            "requested_min_d_spacing_A": float(args.min_d_spacing),
            "actual_min_d_spacing_A": endpoint_resolution["actual_min_d_spacing_A"],
            "resolution_contract": resolution_contract,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "merge_key": args.endpoint_merge_key,
            "overlap_scale_key": args.endpoint_merge_key if args.interdataset_scale == "overlap" else "none",
            "input_unmerged_rows": len(endpoint_rows),
            "merged_endpoint_rows": len(merged_endpoint),
            "merged_exact_hkl_rows": len(merged_endpoint),
            "hkl_safe_factor": hkl_safe_factor if endpoint_rows else 1.0,
            **combined_modulation,
            "per_dataset_modulation": per_dataset_modulation,
            "dataset_sigma": dataset_sigma_by_name,
            "error_model_passes": error_model_passes if endpoint_rows else [],
            "error_model_refits": int(args.error_model_refits) if args.dataset_sigma_policy == "error_model" else 0,
            "merge_sigma": merge_sigma,
            "post_bloch_sigma": post_bloch_sigma if endpoint_rows else {"policy": args.post_bloch_sigma_policy, "status": "skipped_no_endpoint_rows"},
            "shelxl": shelxl,
            "artifacts": endpoint_artifacts,
            "endpoint_wght_mode": args.endpoint_wght_mode,
            "endpoint_initial_wght": list(endpoint_initial_wght(args.endpoint_wght_mode) or []),
            "endpoint_ls_cycles": max(1, int(args.endpoint_ls_cycles)),
            "shelxl_convergence_shift_esd": float(args.shelxl_convergence_shift_esd),
            "shelxl_npd_recovery": args.shelxl_npd_recovery,
            "keep_damp": bool(args.keep_damp),
            "free_rows": {
                "policy": "include" if args.endpoint_include_free_rows else "exclude",
                # read by reports/heldout.py (leak flag for the SHELXL HKL)
                "free_rows_in_hkl": bool(args.endpoint_include_free_rows),
                "hkl_split": "all" if args.endpoint_include_free_rows else "work",
                "n_excluded": int(sum(int(ds.get("endpoint_free_rows_excluded", 0)) for ds in dataset_summaries)),
            },
            "q_policy": args.endpoint_q_policy,
            "n_excluded_q_rows": int(
                sum(int((ds.get("endpoint_correction") or {}).get("n_excluded_q_rows", 0)) for ds in dataset_summaries)
            ),
            "intensity_scale": {
                "mode": args.rescale,
                "kappa": ABSOLUTE_HKL_KAPPA if args.rescale == "absolute" else None,
                "kappa_effective": (ABSOLUTE_HKL_KAPPA * hkl_safe_factor) if (endpoint_rows and args.rescale == "absolute") else None,
                "osf_reset": math.sqrt(ABSOLUTE_HKL_KAPPA * hkl_safe_factor) if (endpoint_rows and args.rescale == "absolute") else None,
                "n_rows_over_f8_2_at_kappa": int(n_over_f8_2) if endpoint_rows else 0,
            },
            "interdataset_scale_diagnostic": interdataset_scale_diagnostic,
        },
        "combined_unmerged_endpoint": unmerged_endpoint,
    }
    if free_validation:
        # only under --endpoint-free-validation on, so the legacy summary keeps its exact shape
        result["combined_endpoint"]["free_validation"] = merged_free_validation or {"status": "skipped_no_endpoint_rows"}
    if score_res:
        # validation only: nothing was refined; "shelxl" stays empty and "artifacts" names the scored model
        result["combined_endpoint"]["mode"] = "score_only"
        result["combined_endpoint"]["score_only"] = score_only or {"status": "skipped_no_endpoint_rows"}
    (out / "combined_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    lines = [
        "# Multi-Crystal Native Nuisance Combination",
        "",
        "The per-dataset nuisance fits are kept separate. This report concatenates signed native rows after each dataset has its own scales/thickness/cell/geometry treatment.",
        f"An explicit reporting/endpoint resolution cutoff of `d >= {float(args.min_d_spacing):g} A` was applied to every supplied row table." if float(args.min_d_spacing) > 0.0 else "No additional reporting/endpoint resolution cutoff was applied to the supplied row tables.",
        f"The discrete highest-resolution retained native row is `d={safe_float(resolution_contract.get('actual_retained_native_min_d_spacing_A')):.6f} A`; the merged endpoint reaches `d={safe_float(resolution_contract.get('actual_merged_endpoint_min_d_spacing_A')):.6f} A`. Membership was calculated from the shared structure RES cell, not per-dataset stored d values.",
        "Per-dataset native metrics below were recalculated from those retained rows with the combined run's dataset-policy and native WGHT contract; they are not copied from the unfiltered source-run summary.",
        f"Combined-endpoint compute wall time: `{compute_wall_seconds:.3f} s`.",
        "",
        "| dataset | role | source rows retained/input | scale model | scale params | raw/endpoint weight | rows native/endpoint | unique signed | unique model/Friedel | basis median/p95/max | dyn R1 all | dyn wR2 all | born R1 all | born wR2 all | q p05/median/p95 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in dataset_summaries:
        bn = ds["best_native_summary"]
        m = bn.get("metrics", {})
        dyn = ((m.get("all") or {}).get("dyn") or {})
        born = ((m.get("all") or {}).get("born") or {})
        bstats = ds.get("basis_size_stats", {})
        resolution_filter = ds.get("resolution_filter") or {}
        scale_contract = ds.get("scale_contract") or {}
        lines.append(
            f"| {ds['name']} | {ds['dataset_role']} | {resolution_filter.get('kept_rows', 0)}/{resolution_filter.get('input_rows', 0)} | "
            f"`{scale_contract.get('model', 'unknown')}` | {scale_contract.get('parameter_count', 'n/a')} | "
            f"{safe_float(ds.get('raw_row_weight')):.3f}/{safe_float(ds.get('endpoint_weight')):.3f}"
            + (
                f" (SHELXL {safe_float(ds.get('endpoint_weight_applied_to_shelxl_sigma')):.3f})"
                if ds.get("endpoint_weight_policy") == "error_model_only"
                else ""
            )
            + " | "
            f"{ds['native_rows']}/{ds['endpoint_rows']} | {ds['unique_signed_hkl']} | {ds['unique_symmetry_friedel']} | "
            f"{safe_float(bstats.get('median')):.0f}/{safe_float(bstats.get('p95')):.0f}/{safe_float(bstats.get('max')):.0f} | "
            f"{100*safe_float(dyn.get('R1_all')):.2f} | {100*safe_float(dyn.get('wR2_all')):.2f} | "
            f"{100*safe_float(born.get('R1_all')):.2f} | {100*safe_float(born.get('wR2_all')):.2f} | "
            f"{ds['q_p05']:.3f}/{ds['q_median']:.3f}/{ds['q_p95']:.3f} |"
        )
    nd = native["dyn"]["all"]
    nb = native["born"]["all"]
    lines.extend(
        [
            "",
            "## Joint Native Metrics",
            "",
            f"Rows in raw-row joint metrics: `{len(all_rows)}`. Endpoint/completeness rows: `{len(endpoint_rows)}`. Unique signed HKL union over raw-row metric rows: `{len(exact_union)}`. Unique model-symmetry/Friedel union: `{len(canon_union)}`.",
            "",
            f"Bloch joint: `R1_I_all={100*nd['R1_all']:.2f}%`, `R1_I_gt={100*nd['R1_gt']:.2f}%`, `wR2_I_all={100*nd['wR2_all']:.2f}%`, `CC={nd['CC']:.4f}`.",
            f"Born joint: `R1_I_all={100*nb['R1_all']:.2f}%`, `R1_I_gt={100*nb['R1_gt']:.2f}%`, `wR2_I_all={100*nb['wR2_all']:.2f}%`, `CC={nb['CC']:.4f}`.",
            "",
            "## Combined Corrected-HKL Endpoint",
            "",
        ]
    )
    if score_res and score_only:
        lines.append(
            f"Validation only (`--score-res`): nothing was refined. The merged endpoint HKL was rebuilt with this "
            f"command's recipe (error-model pass `{score_only.get('error_model_pass')}`, sigma model "
            f"`{score_only.get('sigma_model')}`) and the refined model `{score_only.get('scored_res')}` was scored at "
            f"L.S. 0 with WGHT `{score_only.get('wght')}`. Rebuilt HKL identical to the model's own HKL: "
            f"`{score_only.get('work_hkl_bytes_same')}`; status `{score_only.get('status')}`."
        )
        block = result["combined_endpoint"].get("free_validation")
        if isinstance(block, dict):
            lines.append(free_validation_report_line("merged", block))
    elif shelxl:
        scale_bits = ", ".join(f"{name}={safe_float(scale):.3g}" for name, scale in sorted(interdataset_scale.get("scales", {}).items()))
        lines.append(f"Inter-dataset endpoint scaling: `{args.interdataset_scale}`. Scale factors: {scale_bits or 'n/a'}.")
        if args.dataset_sigma_policy == "uniform_z_rms":
            dataset_bits = ", ".join(
                f"{name}: {safe_float(meta.get('sigma_scale')):.3f} ({meta.get('status')})"
                for name, meta in sorted(dataset_sigma_by_name.items())
            )
            lines.append(f"Per-dataset endpoint sigma calibration: {dataset_bits or 'n/a'}.")
        lines.append(
            f"Merge sigma policy `{args.merge_sigma_policy}`; inflated rows `{merge_sigma.get('inflated_rows', 0)}`/"
            f"`{merge_sigma.get('merged_rows', 0)}`, median/p95/max inflation "
            f"`{safe_float(merge_sigma.get('inflation_median')):.3f}`/"
            f"`{safe_float(merge_sigma.get('inflation_p95')):.3f}`/"
            f"`{safe_float(merge_sigma.get('inflation_max')):.3f}`."
        )
        lines.append(
            f"Overlap scaling and endpoint merging use `{args.endpoint_merge_key}`. "
            "The merged HKLF4 endpoint is written after this GROND-side inverse-variance merge, "
            "so the shared SHELXL LS step sees one corrected observation per selected merge key."
        )
        lines.append(
            f"Corrected-intensity scale `{args.rescale}`; q policy `{args.endpoint_q_policy}` "
            f"(`{result['combined_endpoint']['n_excluded_q_rows']}` rows excluded); free rows "
            f"`{result['combined_endpoint']['free_rows']['policy']}` (`{result['combined_endpoint']['free_rows']['n_excluded']}` excluded); "
            f"NPD recovery `{args.shelxl_npd_recovery}`; SHELXL calls stop at |max shift/esd| < "
            f"`{float(args.shelxl_convergence_shift_esd):g}` (converged `{shelxl.get('converged')}`)."
        )
        lines.append(
            f"Shared endpoint SHELXL policy: `{args.endpoint_wght_mode}` weights and `L.S./CGLS {max(1, int(args.endpoint_ls_cycles))}`. "
            f"Active WGHT reported by SHELXL: `{safe_float(shelxl.get('active_wght_a')):.4f}` "
            f"`{safe_float(shelxl.get('active_wght_b')):.4f}`."
        )
        lines.append("")
        lines.append(
            f"Lambda `{args.endpoint_lambda}` merged `{len(endpoint_rows)}` unmerged endpoint rows to `{len(merged_endpoint)}` corrected endpoint rows. SHELXL status `{shelxl.get('status')}`; `R1gt={100*safe_float(shelxl.get('R1_gt')):.2f}%`, `R1all={100*safe_float(shelxl.get('R1_all')):.2f}%`, `wR2all={100*safe_float(shelxl.get('wR2_all')):.2f}%`, `GooF={safe_float(shelxl.get('GooF')):.3f}`."
        )
        wght_record = shelxl.get("wght_convergence") or {}
        if wght_record:
            lines.append(
                f"SHELXL weights `{wght_record.get('status')}` after `{wght_record.get('calls_made')}` calls: applied "
                f"`{wght_record.get('final_applied')}`, suggested `{wght_record.get('final_suggested')}`."
            )
        if args.dataset_sigma_policy == "error_model":
            for name, meta in sorted(dataset_sigma_by_name.items()):
                lines.append(
                    f"Error model `{name}`: k `{safe_float(meta.get('k')):.3f}`, u `{safe_float(meta.get('u')):.4f}`, "
                    f"work RMS(z) raw/after `{safe_float(meta.get('work_rms_z_raw')):.2f}`/"
                    f"`{safe_float(meta.get('work_rms_z_after')):.3f}`, free RMS(z) after "
                    f"`{safe_float(meta.get('free_rms_z_after')):.3f}` over `{meta.get('free_rows', 0)}` free rows."
                )
        if unmerged_endpoint.get("shelxl"):
            unmerged_shelxl = unmerged_endpoint["shelxl"]
            lines.append(
                f"Unmerged batch endpoint (MERG 0, batches {unmerged_endpoint.get('batches')}): status "
                f"`{unmerged_shelxl.get('status')}`; `R1gt={100*safe_float(unmerged_shelxl.get('R1_gt')):.2f}%`, "
                f"`R1all={100*safe_float(unmerged_shelxl.get('R1_all')):.2f}%`, "
                f"`wR2all={100*safe_float(unmerged_shelxl.get('wR2_all')):.2f}%`, "
                f"`GooF={safe_float(unmerged_shelxl.get('GooF')):.3f}`; BASF start "
                f"`{unmerged_endpoint.get('basf_start')}` refined `{(unmerged_endpoint.get('basf') or {}).get('values')}`."
            )
        for label, block in (
            ("merged", result["combined_endpoint"].get("free_validation")),
            ("unmerged", unmerged_endpoint.get("free_validation")),
        ):
            if isinstance(block, dict):
                lines.append(free_validation_report_line(label, block))
        sig_meta = result["combined_endpoint"].get("post_bloch_sigma") or {}
        if sig_meta.get("policy") == "uniform_z_rms":
            lines.append(
                f"Post-Bloch combined endpoint sigma calibration status `{sig_meta.get('status')}`; "
                f"sigma scale `{safe_float(sig_meta.get('sigma_scale')):.3f}`, "
                f"held-out RMS(z) after `{safe_float(sig_meta.get('heldout_rms_z_after', sig_meta.get('all_rms_z_after'))):.3f}`."
            )
    else:
        lines.append("No endpoint HKL rows were found for the requested lambda.")
    lines.extend(
        [
            "",
            "## Source Provenance",
            "",
            f"Shared structure RES: `{result['provenance']['structure']['path']}`; SHA-256 `{result['provenance']['structure']['sha256']}`.",
            "",
            "| dataset | row table SHA-256 | summary SHA-256 | endpoint-source HKL SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for dataset in dataset_summaries:
        sources = dataset.get("source_artifacts") or {}
        lines.append(
            f"| {dataset['name']} | `{(sources.get('row_table') or {}).get('sha256', '')}` | "
            f"`{(sources.get('summary') or {}).get('sha256', '')}` | "
            f"`{(sources.get('endpoint_hkl') or {}).get('sha256', '')}` |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out.resolve()), "report": str((out / "REPORT.md").resolve()), "native_wR2_all": native["dyn"]["all"]["wR2_all"], "endpoint": shelxl}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
