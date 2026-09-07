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

from grond.regularize.arguments_combination import build_parser as build_combination_parser
from grond.regularize.multidataset import (
    apply_hkl_weight as apply_hkl_weight,
    apply_native_row_weight as apply_native_row_weight,
    apply_sigma_holdout_floor as apply_sigma_holdout_floor,
    audit_combined_endpoint_modulation as audit_combined_endpoint_modulation,
    canonical_model_friedel as canonical_model_friedel,
    canonical_model_symmetry as canonical_model_symmetry,
    canonical_p222_friedel as canonical_p222_friedel,
    d_spacing_from_model as d_spacing_from_model,
    endpoint_key_function as endpoint_key_function,
    endpoint_rows_and_corrections_from_native as endpoint_rows_and_corrections_from_native,
    endpoint_rows_from_native as endpoint_rows_from_native,
    estimate_overlap_scales as estimate_overlap_scales,
    file_provenance as file_provenance,
    filter_hkl_rows_by_resolution as filter_hkl_rows_by_resolution,
    filter_native_rows_by_resolution as filter_native_rows_by_resolution,
    hkl_resolution_stats as hkl_resolution_stats,
    merge_hkl_rows as merge_hkl_rows,
    metrics as metrics,
    native_metrics_by_split as native_metrics_by_split,
    reciprocal_equivalent_keys as reciprocal_equivalent_keys,
    safe_float as safe_float,
    scale_contract_provenance as scale_contract_provenance,
    scale_hkl_rows as scale_hkl_rows,
    selected_native_wght as selected_native_wght,
    shelx_weights as shelx_weights,
    split_metrics as split_metrics,
)

try:  # Package import.
    from .bloch_engine import parse_model, sha256
    from .endpoint_modulation import modulation_manifest_fields as modulation_manifest_fields
    from .native_row_regularize import (
        fail_if_shelxl_became_unavailable,
        parse_shelx_wght_terms,
        recommended_wght_from_stats,
        recover_npd_shelxl_attempt,
        require_endpoint_shelxl,
        run_one_shelxl_attempt,
        select_last_safe_shelxl_iteration,
        shelx_res_health,
        shelxl_iteration_safe,
        snapshot_safe_shelxl_iteration,
        write_active_wght,
        write_endpoint_ins,
    )
    from .post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
    from .shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl
except ImportError:  # Direct script execution from this directory.
    from bloch_engine import parse_model, sha256
    from native_row_regularize import (
        fail_if_shelxl_became_unavailable,
        parse_shelx_wght_terms,
        recommended_wght_from_stats,
        recover_npd_shelxl_attempt,
        require_endpoint_shelxl,
        run_one_shelxl_attempt,
        select_last_safe_shelxl_iteration,
        shelx_res_health,
        shelxl_iteration_safe,
        snapshot_safe_shelxl_iteration,
        write_active_wght,
        write_endpoint_ins,
    )
    from post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
    from shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def read_rows(run_dir: Path, name: str) -> list[dict]:
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
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            h, k, ell = int(parts[0]), int(parts[1]), int(parts[2])
            intensity = safe_float(parts[3])
            sig = safe_float(parts[4])
        else:
            h, k, ell = int(line[:4]), int(line[4:8]), int(line[8:12])
            intensity = safe_float(line[12:20])
            sig = safe_float(line[20:28])
        if h == k == ell == 0:
            continue
        rows.append((h, k, ell, intensity, sig))
    return rows


def write_hkl(path: Path, rows):
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


def write_no_exti_ins(
    src_res: Path,
    dst_ins: Path,
    hkl_name: str,
    *,
    wght: tuple[float, float] | None = None,
    ls_cycles: int = 1,
):
    write_endpoint_ins(src_res, dst_ins, hkl_name=hkl_name, keep_exti=False, wght=wght, ls_cycles=ls_cycles)


def parse_lst(path: Path):
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


def parse_recommended_wght(lst_path: Path) -> tuple[float, float] | None:
    if not lst_path.exists():
        return None
    text = lst_path.read_text(errors="replace")
    matches = []
    for line in text.splitlines():
        upper = line.upper()
        if "RECOMMENDED WEIGHTING SCHEME" not in upper or "WGHT" not in upper:
            continue
        terms = parse_shelx_wght_terms(line[upper.find("WGHT") :])
        if terms:
            matches.append(terms)
    if not matches:
        return None
    terms = matches[-1]
    return terms[0], terms[1] if len(terms) >= 2 else 0.0


def promote_recommended_wght(ins_path: Path, recommended: tuple[float, float] | None) -> dict:
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
):
    exe = shelxl or shutil.which("shelxl") or shutil.which("shelxl2019") or ""
    if not exe:
        return {"status": "missing_shelxl", "executable": ""}
    if "/" in exe and not Path(exe).exists():
        return {"status": "missing_shelxl", "executable": exe}
    history = []
    last_safe = None
    last_stdout = ""
    n_iter = max(1, int(iterations))
    n_ls = max(1, int(endpoint_ls_cycles))
    mode = str(wght_mode or "freeze-zero")
    freeze_zero = mode == "freeze-zero"
    if mode not in {"inherit", "freeze-zero", "promote"}:
        return {"returncode": None, "status": "invalid_wght_mode", "wght_mode": mode}
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
            recovered_proc, stats = recover_npd_shelxl_attempt(
                work,
                stem,
                exe,
                iteration=i + 1,
                base_ins_text=base_ins_text,
                initial_stats=stats,
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
        last_safe["stats"] = dict(stats)
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
            )
            if mode == "promote":
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
    return run / f"endpoint_lam_{lam.replace('.', 'p')}"


def endpoint_hkl_path(run: Path, lam: str) -> Path:
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


def main():
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
            endpoint_rows, endpoint_corrections = endpoint_rows_and_corrections_from_native(
                endpoint_input_rows,
                float(args.endpoint_lambda),
            )
            endpoint_rows_by_dataset[name] = endpoint_rows
            endpoint_corrections_by_dataset[name] = endpoint_corrections
        dataset_sigma_by_name[name] = {"policy": args.dataset_sigma_policy, "status": "not_requested"}

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
    if args.interdataset_scale == "overlap":
        interdataset_scale = estimate_overlap_scales(endpoint_rows_by_dataset, canonicalizer=endpoint_key_for)
    else:
        interdataset_scale = {
            "iterations": 0,
            "overlap_counts": {name: 0 for name in endpoint_rows_by_dataset},
            "scales": {name: 1.0 for name in endpoint_rows_by_dataset},
        }
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
        hkl_safe_factor = min(safe_fixed_width_factor(endpoint_rows), safe_fixed_width_factor(merged_endpoint))
        if hkl_safe_factor < 1.0:
            endpoint_rows = [
                (h, k, ell, intensity * hkl_safe_factor, sig * hkl_safe_factor)
                for h, k, ell, intensity, sig in endpoint_rows
            ]
            merged_endpoint = [
                (h, k, ell, intensity * hkl_safe_factor, sig * hkl_safe_factor)
                for h, k, ell, intensity, sig in merged_endpoint
            ]
        ep_dir = out / f"combined_endpoint_lam_{args.endpoint_lambda.replace('.', 'p')}"
        ep_dir.mkdir(exist_ok=True)
        hkl_name = "multi_dyn_corrected.hkl"
        write_hkl(ep_dir / hkl_name, merged_endpoint)
        initial_wght = (0.0, 0.0) if args.endpoint_wght_mode == "freeze-zero" else None
        write_no_exti_ins(
            Path(args.res),
            ep_dir / "multi_dyn_corrected.ins",
            hkl_name,
            wght=initial_wght,
            ls_cycles=args.endpoint_ls_cycles,
        )
        shelxl = run_shelxl(
            ep_dir,
            "multi_dyn_corrected",
            endpoint_shelxl,
            iterations=args.shelxl_iterations,
            wght_mode=args.endpoint_wght_mode,
            endpoint_ls_cycles=args.endpoint_ls_cycles,
        )
        fail_if_shelxl_became_unavailable(
            [{"shelxl": shelxl}],
            context="combined corrected-HKL endpoint refinement",
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
            "merge_sigma": merge_sigma,
            "post_bloch_sigma": post_bloch_sigma if endpoint_rows else {"policy": args.post_bloch_sigma_policy, "status": "skipped_no_endpoint_rows"},
            "shelxl": shelxl,
            "artifacts": endpoint_artifacts,
            "endpoint_wght_mode": args.endpoint_wght_mode,
            "endpoint_ls_cycles": max(1, int(args.endpoint_ls_cycles)),
        },
    }
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
            f"{safe_float(ds.get('raw_row_weight')):.3f}/{safe_float(ds.get('endpoint_weight')):.3f} | "
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
    if shelxl:
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
            f"Shared endpoint SHELXL policy: `{args.endpoint_wght_mode}` weights and `L.S./CGLS {max(1, int(args.endpoint_ls_cycles))}`. "
            f"Active WGHT reported by SHELXL: `{safe_float(shelxl.get('active_wght_a')):.4f}` "
            f"`{safe_float(shelxl.get('active_wght_b')):.4f}`."
        )
        lines.append("")
        lines.append(
            f"Lambda `{args.endpoint_lambda}` merged `{len(endpoint_rows)}` unmerged endpoint rows to `{len(merged_endpoint)}` corrected endpoint rows. SHELXL status `{shelxl.get('status')}`; `R1gt={100*safe_float(shelxl.get('R1_gt')):.2f}%`, `R1all={100*safe_float(shelxl.get('R1_all')):.2f}%`, `wR2all={100*safe_float(shelxl.get('wR2_all')):.2f}%`, `GooF={safe_float(shelxl.get('GooF')):.3f}`."
        )
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
