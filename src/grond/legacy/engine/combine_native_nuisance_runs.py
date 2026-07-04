#!/usr/bin/env python3
"""Combine separately fitted native nuisance runs without merging raw rows first.

This is a reporting/export bridge for multi-crystal tests.  Each input run is a
single-dataset `native_row_regularize.py` output directory.  The per-dataset
predictions and scales remain independent; this script concatenates the
resulting signed row tables and reports joint native metrics on the same
calculation definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

try:  # Package import.
    from .bloch_engine import parse_model
    from .native_row_regularize import parse_damp_retry_grid, parse_shelx_wght_terms, shelx_res_health, write_active_damp, write_endpoint_ins
    from .post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
except ImportError:  # Direct script execution from this directory.
    from bloch_engine import parse_model
    from native_row_regularize import parse_damp_retry_grid, parse_shelx_wght_terms, shelx_res_health, write_active_damp, write_endpoint_ins
    from post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def safe_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_rows(run_dir: Path, name: str) -> list[dict]:
    path = run_dir / "native_dyn_calc.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["dataset"] = name
            r.setdefault("dataset_role", "primary")
            r.setdefault("raw_row_weight", "1.0")
            r.setdefault("endpoint_weight", r.get("raw_row_weight", "1.0"))
            rows.append(r)
    return rows


def canonical_p222_friedel(h, k, l):
    variants = [
        (h, k, l),
        (-h, -k, l),
        (h, -k, -l),
        (-h, k, -l),
    ]
    variants += [(-a, -b, -c) for a, b, c in variants]
    return min(variants)


def reciprocal_equivalent_keys(
    h: int,
    k: int,
    l: int,
    symmetry_ops: list[dict],
    *,
    merge_friedel: bool = True,
) -> list[tuple[int, int, int]]:
    """Return reciprocal equivalents from the model symmetry operations."""

    hkl = np.asarray([int(h), int(k), int(l)], dtype=float)
    keys: list[tuple[int, int, int]] = []
    for op in symmetry_ops or [{"rot": np.eye(3).tolist()}]:
        rot = np.asarray(op.get("rot", np.eye(3)), dtype=float)
        for mat in (rot, np.linalg.inv(rot)):
            equiv = np.rint(hkl @ mat).astype(int)
            key = tuple(int(v) for v in equiv)
            keys.append(key)
            if merge_friedel:
                keys.append(tuple(-v for v in key))
    return sorted(set(keys))


def canonical_model_friedel(h: int, k: int, l: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    return min(reciprocal_equivalent_keys(h, k, l, symmetry_ops, merge_friedel=True))


def canonical_model_symmetry(h: int, k: int, l: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    return min(reciprocal_equivalent_keys(h, k, l, symmetry_ops, merge_friedel=False))


def endpoint_key_function(mode: str, symmetry_ops: list[dict]):
    """Return the HKL key used for endpoint scaling or merging."""

    if mode == "exact_signed_hkl":
        return lambda h, k, l: (int(h), int(k), int(l))
    if mode == "model_symmetry":
        return lambda h, k, l: canonical_model_symmetry(int(h), int(k), int(l), symmetry_ops)
    if mode == "model_symmetry_friedel":
        return lambda h, k, l: canonical_model_friedel(int(h), int(k), int(l), symmetry_ops)
    raise ValueError(f"unknown endpoint merge key {mode!r}")


def shelx_weights(obs, sig, pred, a: float, b: float):
    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    return 1.0 / np.maximum(sig * sig + (float(a) * p) ** 2 + float(b) * p, 1e-30)


def metrics(rows: list[dict], pred_col: str, wght_a: float, wght_b: float):
    obs = np.asarray([safe_float(r["target_I"]) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r["sigma_eff"]) for r in rows], dtype=float)
    pred = np.asarray([safe_float(r[pred_col]) for r in rows], dtype=float)
    m = np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs = obs[m]
    sig = sig[m]
    pred = pred[m]
    if len(obs) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "wR2_all": float("nan"), "CC": float("nan"), "mw_abs": float("nan")}
    gt = obs > 2.0 * sig
    fobs = np.sqrt(np.maximum(obs, 0.0))
    fcalc = np.sqrt(np.maximum(pred, 0.0))

    def r1(mask):
        if not np.any(mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[mask] - fcalc[mask])) / max(np.sum(fobs[mask]), 1e-30))

    w = shelx_weights(obs, sig, pred, wght_a, wght_b)
    res = obs - pred
    wr2 = float(np.sqrt(np.sum(w * res * res) / max(np.sum(w * obs * obs), 1e-30)))
    cc = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 2 and np.std(obs) > 0 and np.std(pred) > 0 else float("nan")
    return {
        "n": int(len(obs)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "wR2_all": wr2,
        "CC": cc,
        "mw_abs": float(np.mean(np.sqrt(w) * np.abs(res))),
    }


def split_metrics(rows: list[dict], pred_col: str, wght_a: float, wght_b: float):
    out = {}
    for split in ("all", "work", "free"):
        use = rows if split == "all" else [r for r in rows if r.get("split") == split]
        out[split] = metrics(use, pred_col, wght_a, wght_b)
    return out


def read_hkl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            I = safe_float(parts[3])
            sig = safe_float(parts[4])
        else:
            h, k, l = int(line[:4]), int(line[4:8]), int(line[8:12])
            I = safe_float(line[12:20])
            sig = safe_float(line[20:28])
        if h == k == l == 0:
            continue
        rows.append((h, k, l, I, sig))
    return rows


def write_hkl(path: Path, rows):
    max_abs = max((max(abs(float(I)), abs(float(sig))) for _, _, _, I, sig in rows), default=0.0)
    if max_abs >= 9999.995:
        raise ValueError(
            f"{path} would overflow SHELXL HKLF4 8.2 intensity fields; "
            "normalize endpoint HKLs before combination"
        )
    with path.open("w") as f:
        for h, k, l, I, sig in rows:
            f.write(f"{int(h):4d}{int(k):4d}{int(l):4d}{float(I):8.2f}{float(sig):8.2f}\n")
        f.write(f"{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}\n")


def safe_fixed_width_factor(rows) -> float:
    """Return a common multiplier that keeps HKLF4 8.2 fields in range."""

    max_abs = max((max(abs(float(I)), abs(float(sig))) for _, _, _, I, sig in rows), default=0.0)
    if max_abs <= 0.0 or max_abs < 9999.995:
        return 1.0
    return 9000.0 / max_abs


def endpoint_rows_from_native(rows: list[dict], lambda_q: float):
    hkl = [(int(r["h"]), int(r["k"]), int(r["l"])) for r in rows]
    target = np.asarray([safe_float(r["target_I"]) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r["sigma_eff"]) for r in rows], dtype=float)
    dyn_raw = np.asarray([safe_float(r["dyn_raw"]) for r in rows], dtype=float)
    dyn_scaled = np.asarray([safe_float(r["dyn_scaled"]) for r in rows], dtype=float)
    q = np.asarray([safe_float(r["q_dyn_over_born"]) for r in rows], dtype=float)
    valid = np.isfinite(target) & np.isfinite(sig) & np.isfinite(dyn_raw) & np.isfinite(dyn_scaled) & np.isfinite(q) & (q > 0.0) & (dyn_raw > 0.0)
    q_clip = np.clip(q, 0.05, 20.0)
    scales = np.maximum(dyn_scaled / np.maximum(dyn_raw, 1e-30), 1e-30)
    denom = np.maximum(scales * (q_clip ** float(lambda_q)), 1e-30)
    icorr = target / denom
    sigcorr = sig / denom
    src = float(np.nanmedian(target[valid])) if np.any(valid) else 1.0
    dst = float(np.nanmedian(icorr[valid])) if np.any(valid) else 1.0
    factor = src / max(dst, 1e-30)
    icorr *= factor
    sigcorr *= factor
    return [(*hkl[i], float(icorr[i]), float(sigcorr[i])) for i in np.where(valid)[0]]


def apply_hkl_weight(rows, weight: float):
    if weight <= 0.0 or abs(weight - 1.0) < 1e-12:
        return rows
    factor = 1.0 / math.sqrt(max(weight, 1e-12))
    return [(h, k, l, I, sig * factor) for h, k, l, I, sig in rows]


def scale_hkl_rows(rows, scale: float):
    scale = float(scale)
    return [(h, k, l, I * scale, sig * abs(scale)) for h, k, l, I, sig in rows]


def apply_sigma_holdout_floor(meta: dict) -> dict:
    """Avoid making a dataset tighter than its own held-out residual scatter."""

    out = dict(meta)
    if not str(out.get("status", "")).startswith("ok"):
        return out
    current = safe_float(out.get("sigma_scale"))
    holdout_before = safe_float(out.get("heldout_rms_z_before"))
    if not (math.isfinite(current) and current > 0.0 and math.isfinite(holdout_before) and holdout_before > 0.0):
        return out
    new_scale = max(current, holdout_before)
    if new_scale <= current + max(1e-12, 1e-9 * abs(current)):
        out["holdout_floor_applied"] = False
        return out
    out["sigma_scale_pre_holdout_floor"] = current
    out["sigma_scale"] = float(new_scale)
    out["holdout_floor_applied"] = True
    for key in ("train", "all", "heldout"):
        before_key = f"{key}_rms_z_before"
        after_key = f"{key}_rms_z_after"
        before = safe_float(out.get(before_key))
        if math.isfinite(before):
            out[after_key] = float(before / new_scale)
        med_before_key = f"{key}_median_abs_z_before"
        med_after_key = f"{key}_median_abs_z_after"
        med_before = safe_float(out.get(med_before_key))
        if math.isfinite(med_before):
            out[med_after_key] = float(med_before / new_scale)
    return out


def estimate_overlap_scales(rows_by_dataset: dict[str, list[tuple]], iterations: int = 6, canonicalizer=None) -> dict:
    """Estimate per-dataset endpoint scales from overlapping corrected HKLs."""

    key_for = canonicalizer or canonical_p222_friedel
    scales = {name: 1.0 for name in rows_by_dataset}
    overlap_counts = {name: 0 for name in rows_by_dataset}
    if len(rows_by_dataset) <= 1:
        return {"iterations": 0, "overlap_counts": overlap_counts, "scales": scales}
    for _ in range(max(1, int(iterations))):
        bins: dict[tuple[int, int, int], dict[str, float]] = {}
        for name, rows in rows_by_dataset.items():
            scale = scales.get(name, 1.0)
            for h, k, l, intensity, sigma in rows:
                value = float(intensity) * scale
                sigma_s = max(abs(float(sigma) * scale), 1e-6)
                if not (math.isfinite(value) and value > 0.0 and math.isfinite(sigma_s)):
                    continue
                key = key_for(int(h), int(k), int(l))
                weight = 1.0 / (sigma_s * sigma_s)
                bucket = bins.setdefault(key, {"w": 0.0, "wI": 0.0, "n": 0})
                bucket["w"] += weight
                bucket["wI"] += weight * value
                bucket["n"] += 1
        reference = {key: bucket["wI"] / max(bucket["w"], 1e-30) for key, bucket in bins.items() if bucket["n"] >= 2 and bucket["w"] > 0.0}
        updated = {}
        for name, rows in rows_by_dataset.items():
            ratios = []
            scale = scales.get(name, 1.0)
            for h, k, l, intensity, _sigma in rows:
                ref = reference.get(key_for(int(h), int(k), int(l)))
                value = float(intensity) * scale
                if ref is not None and ref > 0.0 and value > 0.0 and math.isfinite(value):
                    ratios.append(ref / value)
            overlap_counts[name] = len(ratios)
            updated[name] = scale * (float(np.median(ratios)) if ratios else 1.0)
        anchor = float(np.median(list(updated.values()))) if updated else 1.0
        scales = {name: value / max(anchor, 1e-30) for name, value in updated.items()}
    return {
        "iterations": int(iterations),
        "overlap_counts": {name: int(count) for name, count in overlap_counts.items()},
        "scales": {name: float(value) for name, value in scales.items()},
    }


def merge_hkl_rows(rows, sigma_policy: str = "formal", canonicalizer=None):
    key_for = canonicalizer or (lambda h, k, l: (int(h), int(k), int(l)))
    bins: dict[tuple[int, int, int], dict[str, float]] = {}
    for h, k, l, I, sig in rows:
        key = key_for(int(h), int(k), int(l))
        sig = max(abs(float(sig)), 1e-6)
        w = 1.0 / (sig * sig)
        b = bins.setdefault(key, {"w": 0.0, "wI": 0.0, "n": 0, "values": [], "weights": []})
        b["w"] += w
        b["wI"] += w * float(I)
        b["n"] += 1
        b["values"].append(float(I))
        b["weights"].append(float(w))
    out = []
    inflations = []
    for key, b in bins.items():
        mean = b["wI"] / max(b["w"], 1e-30)
        sigma = (1.0 / max(b["w"], 1e-30)) ** 0.5
        inflation = 1.0
        if sigma_policy == "chisq_overdispersion" and b["n"] > 1:
            values = np.asarray(b["values"], dtype=float)
            weights = np.asarray(b["weights"], dtype=float)
            chi2 = float(np.sum(weights * (values - mean) ** 2) / max(int(b["n"]) - 1, 1))
            if math.isfinite(chi2) and chi2 > 1.0:
                inflation = math.sqrt(chi2)
                sigma *= inflation
        inflations.append(inflation)
        out.append((*key, mean, sigma))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    inflations_arr = np.asarray(inflations, dtype=float) if inflations else np.asarray([], dtype=float)
    meta = {
        "policy": sigma_policy,
        "merged_rows": int(len(out)),
        "rows_with_multiple_observations": int(sum(1 for b in bins.values() if b["n"] > 1)),
        "inflated_rows": int(np.count_nonzero(inflations_arr > 1.0)) if inflations_arr.size else 0,
        "inflation_median": float(np.median(inflations_arr)) if inflations_arr.size else float("nan"),
        "inflation_p95": float(np.percentile(inflations_arr, 95)) if inflations_arr.size else float("nan"),
        "inflation_max": float(np.max(inflations_arr)) if inflations_arr.size else float("nan"),
    }
    return out, meta


def write_no_exti_ins(src_res: Path, dst_ins: Path, hkl_name: str):
    write_endpoint_ins(src_res, dst_ins, hkl_name=hkl_name, keep_exti=False)


def parse_lst(path: Path):
    out = {"status": "missing", "lst": str(path)}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    out["status"] = "ok"
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


def run_shelxl(work: Path, stem: str, shelxl: str, iterations: int = 5, npd_damp_retries: str = "0.5,0.25,0.1", npd_damp_limse: float = 15.0):
    exe = shelxl or shutil.which("shelxl") or shutil.which("shelxl2019") or ""
    if not exe:
        return {"status": "missing_shelxl", "executable": ""}
    if "/" in exe and not Path(exe).exists():
        return {"status": "missing_shelxl", "executable": exe}
    history = []
    last_stdout = ""
    n_iter = max(1, int(iterations))
    damp_retries = parse_damp_retry_grid(npd_damp_retries, npd_damp_limse)
    active_damp: tuple[float, float] | None = None

    def run_once(iteration: int, *, retry: int | None = None, damp: tuple[float, float] | None = None):
        label = f"shelxl_iter_{iteration}" if retry is None else f"shelxl_iter_{iteration}_damp_retry_{retry}"
        proc = subprocess.run([exe, stem], cwd=str(work), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (work / f"{label}.log").write_text(proc.stdout)
        lst = work / f"{stem}.lst"
        stats = {"iteration": iteration, "returncode": proc.returncode, **parse_lst(lst)}
        if retry is not None:
            stats["retry"] = retry
            stats["retry_reason"] = "adp_pathology"
        if damp is not None:
            stats["active_damp"] = float(damp[0])
            stats["active_damp_limse"] = float(damp[1])
        recommended = parse_recommended_wght(lst)
        if recommended is not None:
            stats["recommended_wght_a"] = recommended[0]
            stats["recommended_wght_b"] = recommended[1]
        health = shelx_res_health(work / f"{stem}.res", lst)
        stats["res_health"] = health
        if not health["ok"]:
            stats["status"] = "invalid_res_output"
            stats["invalid_res_reason"] = health["reason"]
        elif health.get("safe_for_feedback") is False:
            stats["status"] = "adp_pathology"
            stats["invalid_res_reason"] = "adp_pathology"
        return proc, stats, recommended

    for i in range(n_iter):
        ins = work / f"{stem}.ins"
        base_ins_text = ins.read_text(errors="replace") if ins.exists() else ""
        if active_damp is not None:
            write_active_damp(ins, active_damp[0], active_damp[1])
        proc, stats, recommended = run_once(i + 1, damp=active_damp)
        last_stdout = proc.stdout
        history.append(stats)
        health = stats["res_health"]
        if not health["ok"]:
            break
        if health.get("safe_for_feedback") is False:
            accepted_retry = False
            for retry_index, retry_damp in enumerate(damp_retries, start=1):
                if base_ins_text:
                    ins.write_text(base_ins_text)
                write_active_damp(ins, retry_damp[0], retry_damp[1])
                retry_proc, retry_stats, retry_recommended = run_once(i + 1, retry=retry_index, damp=retry_damp)
                last_stdout = retry_proc.stdout
                history.append(retry_stats)
                retry_health = retry_stats["res_health"]
                if retry_health["ok"] and retry_health.get("safe_for_feedback") is not False:
                    stats = retry_stats
                    recommended = retry_recommended
                    active_damp = retry_damp
                    accepted_retry = True
                    retry_stats["status"] = "accepted_damp_retry"
                    break
            if not accepted_retry:
                break
        if i + 1 < n_iter and (work / f"{stem}.res").exists():
            write_endpoint_ins(work / f"{stem}.res", ins, hkl_name=f"{stem}.hkl", keep_exti=False, damp=active_damp)
            stats["next_wght"] = promote_recommended_wght(ins, recommended)
    (work / "shelxl_stdout.log").write_text(last_stdout)
    final = dict(history[-1]) if history else {"status": "not_run"}
    final.setdefault("res_health", shelx_res_health(work / f"{stem}.res", work / f"{stem}.lst"))
    if final["res_health"].get("ok"):
        write_endpoint_ins(work / f"{stem}.res", work / f"{stem}.res", hkl_name=f"{stem}.hkl", keep_exti=False, damp=active_damp)
        final["res_normalized_for_reopen"] = True
    else:
        final["status"] = "invalid_res_output"
        final["invalid_res_reason"] = final["res_health"].get("reason")
    final["iterations"] = n_iter
    final["iteration_history"] = history
    final["npd_damp_retries"] = [{"damp": float(d), "limse": float(l)} for d, l in damp_retries]
    final["damp_retry_count"] = sum(1 for item in history if "retry" in item)
    final["damp_retry_accepted"] = any(item.get("status") == "accepted_damp_retry" for item in history)
    if active_damp is not None:
        final["active_damp"] = float(active_damp[0])
        final["active_damp_limse"] = float(active_damp[1])
    pathology = final.get("res_health") or {}
    final["npd_count"] = int(pathology.get("npd_count", 0) or 0)
    final["split_count"] = int(pathology.get("split_count", 0) or 0)
    final["npd_atoms"] = pathology.get("npd_atoms", [])
    final["has_npd"] = bool(pathology.get("has_npd", False))
    final["has_split_warnings"] = bool(pathology.get("has_split_warnings", False))
    return final


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="name=/path/to/native_nuisance_run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--wght-a", type=float, default=None)
    ap.add_argument("--wght-b", type=float, default=None)
    ap.add_argument(
        "--endpoint-lambda",
        default="1.0",
        help="Diagnostic endpoint q exponent. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument(
        "--endpoint-source",
        choices=("native", "hkl"),
        default="native",
        help="Build corrected endpoint rows from native_dyn_calc.csv, or read normalized endpoint HKL rows from each run.",
    )
    ap.add_argument(
        "--interdataset-scale",
        choices=("overlap", "none"),
        default="overlap",
        help="Fit one scalar per dataset from overlapping corrected reflections before merging the multi-dataset endpoint.",
    )
    ap.add_argument(
        "--endpoint-merge-key",
        choices=("model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"),
        default="model_symmetry",
        help=(
            "Reflection key used before the shared endpoint LS step. model_symmetry merges corrected observations related "
            "by the model space-group symmetry while preserving Friedel opposites in acentric groups; model_symmetry_friedel "
            "also merges Friedel mates; exact_signed_hkl is a diagnostic legacy mode."
        ),
    )
    ap.add_argument(
        "--dataset-sigma-policy",
        choices=("none", "uniform_z_rms"),
        default="none",
        help=(
            "Endpoint uncertainty policy applied independently to each dataset before interdataset scaling and merging. "
            "This keeps a weak crystal from setting the sigma scale for stronger crystals."
        ),
    )
    ap.add_argument("--dataset-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--dataset-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--dataset-sigma-max-scale", type=float, default=4.0)
    ap.add_argument(
        "--dataset-sigma-holdout-floor",
        action="store_true",
        help="For per-dataset uniform_z_rms, do not deflate sigmas below that dataset's held-out RMS(z).",
    )
    ap.add_argument(
        "--merge-sigma-policy",
        choices=("formal", "chisq_overdispersion"),
        default="formal",
        help=(
            "How to assign sigmas after merging corrected endpoint rows. formal uses propagated inverse-variance errors; "
            "chisq_overdispersion inflates a merged reflection when its contributors disagree more than their sigmas predict."
        ),
    )
    ap.add_argument("--shelxl", default="")
    ap.add_argument("--shelxl-iterations", type=int, default=5)
    ap.add_argument("--npd-damp-retries", default="0.5,0.25,0.1", help="DAMP values retried when the combined SHELXL endpoint has NPD or split pathology. Use off to disable.")
    ap.add_argument("--npd-damp-limse", type=float, default=15.0, help="Default SHELXL DAMP LIMSE for NPD/split retry values.")
    ap.add_argument(
        "--post-bloch-sigma-policy",
        choices=("none", "uniform_z_rms"),
        default="none",
        help=(
            "Endpoint uncertainty policy applied after multi-dataset merging and before SHELXL. "
            "uniform_z_rms fits one residual RMS(z) scalar from the supplied model and scales endpoint sigmas only."
        ),
    )
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    args = ap.parse_args()

    model = parse_model(Path(args.res))
    symmetry_ops = model.get("symmetry_ops", [])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    dataset_summaries = []
    endpoint_rows_by_dataset: dict[str, list[tuple]] = {}
    dataset_sigma_by_name: dict[str, dict] = {}
    wght_a = args.wght_a
    wght_b = args.wght_b
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must be name=/path")
        name, path_text = item.split("=", 1)
        run = Path(path_text)
        summary = read_json(run / "summary.json")
        if wght_a is None:
            wght_a = safe_float((summary.get("shelx_weight") or {}).get("a"), 0.126534)
        if wght_b is None:
            wght_b = safe_float((summary.get("shelx_weight") or {}).get("b"), 0.0)
        rows = read_rows(run, name)
        native_rows = [r for r in rows if str(r.get("dataset_role", "primary")) not in {"completeness_only", "exclude"}]
        endpoint_input_rows = [r for r in rows if str(r.get("dataset_role", "primary")) != "exclude" and safe_float(r.get("endpoint_weight"), 1.0) > 0.0]
        all_rows.extend(native_rows)
        exact = {tuple(int(r[x]) for x in ("h", "k", "l")) for r in rows}
        canon = {canonical_model_friedel(*hkl, symmetry_ops) for hkl in exact}
        q = np.asarray([safe_float(r["q_dyn_over_born"]) for r in endpoint_input_rows], dtype=float)
        basis = np.asarray([safe_float(r.get("basis_size")) for r in endpoint_input_rows], dtype=float)
        basis = basis[np.isfinite(basis)]
        role = str(rows[0].get("dataset_role", "primary")) if rows else "missing"
        raw_weight = safe_float(rows[0].get("raw_row_weight"), 1.0) if rows else float("nan")
        endpoint_weight = safe_float(rows[0].get("endpoint_weight"), raw_weight) if rows else float("nan")
        dataset_summaries.append(
            {
                "name": name,
                "run_dir": str(run.resolve()),
                "rows": len(rows),
                "native_rows": len(native_rows),
                "endpoint_rows": len(endpoint_input_rows),
                "dataset_role": role,
                "raw_row_weight": raw_weight,
                "endpoint_weight": endpoint_weight,
                "unique_signed_hkl": len(exact),
                "unique_symmetry_friedel": len(canon),
                "unique_p222_friedel": len(canon),
                "best_native_summary": summary.get("best_native", {}),
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
            hkl_rows = read_hkl(endpoint_hkl_path(run, args.endpoint_lambda))
            source_run_text = (summary.get("dataset_policy") or {}).get("run_dir", "")
            source_run = Path(source_run_text) if source_run_text else None
            if not hkl_rows and source_run and source_run.exists():
                hkl_rows = read_hkl(endpoint_hkl_path(source_run, args.endpoint_lambda))
            endpoint_rows_by_dataset[name] = apply_hkl_weight(hkl_rows, endpoint_weight)
        else:
            endpoint_rows_by_dataset[name] = endpoint_rows_from_native(endpoint_input_rows, float(args.endpoint_lambda))
        dataset_sigma_by_name[name] = {"policy": args.dataset_sigma_policy, "status": "not_requested"}

    if args.dataset_sigma_policy == "uniform_z_rms":
        for name, rows_for_dataset in list(endpoint_rows_by_dataset.items()):
            try:
                sigma_meta = fit_uniform_z_rms_sigma_scale(
                    rows_for_dataset,
                    model,
                    holdout_fraction=float(args.dataset_sigma_holdout_fraction),
                    min_scale=float(args.dataset_sigma_min_scale),
                    max_scale=float(args.dataset_sigma_max_scale),
                )
                if args.dataset_sigma_holdout_floor:
                    sigma_meta = apply_sigma_holdout_floor(sigma_meta)
                if str(sigma_meta.get("status", "")).startswith("ok"):
                    endpoint_rows_by_dataset[name] = scale_row_sigmas(rows_for_dataset, float(sigma_meta["sigma_scale"]))
                dataset_sigma_by_name[name] = sigma_meta
            except Exception as exc:
                dataset_sigma_by_name[name] = {
                    "policy": "uniform_z_rms",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    wght_a = 0.126534 if wght_a is None else float(wght_a)
    wght_b = 0.0 if wght_b is None else float(wght_b)
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
        endpoint_rows.extend(scale_hkl_rows(rows_for_dataset, interdataset_scale["scales"].get(name, 1.0)))
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
            endpoint_rows = [(h, k, l, I * hkl_safe_factor, sig * hkl_safe_factor) for h, k, l, I, sig in endpoint_rows]
            merged_endpoint = [(h, k, l, I * hkl_safe_factor, sig * hkl_safe_factor) for h, k, l, I, sig in merged_endpoint]
        ep_dir = out / f"combined_endpoint_lam_{args.endpoint_lambda.replace('.', 'p')}"
        ep_dir.mkdir(exist_ok=True)
        hkl_name = "multi_dyn_corrected.hkl"
        write_hkl(ep_dir / hkl_name, merged_endpoint)
        write_no_exti_ins(Path(args.res), ep_dir / "multi_dyn_corrected.ins", hkl_name)
        shelxl = run_shelxl(
            ep_dir,
            "multi_dyn_corrected",
            args.shelxl,
            iterations=args.shelxl_iterations,
            npd_damp_retries=args.npd_damp_retries,
            npd_damp_limse=args.npd_damp_limse,
        )

    result = {
        "runs": dataset_summaries,
        "wght": {"a": wght_a, "b": wght_b},
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
            "merge_key": args.endpoint_merge_key,
            "overlap_scale_key": args.endpoint_merge_key if args.interdataset_scale == "overlap" else "none",
            "input_unmerged_rows": len(endpoint_rows),
            "merged_endpoint_rows": len(merged_endpoint),
            "merged_exact_hkl_rows": len(merged_endpoint),
            "hkl_safe_factor": hkl_safe_factor if endpoint_rows else 1.0,
            "dataset_sigma": dataset_sigma_by_name,
            "merge_sigma": merge_sigma,
            "post_bloch_sigma": post_bloch_sigma if endpoint_rows else {"policy": args.post_bloch_sigma_policy, "status": "skipped_no_endpoint_rows"},
            "shelxl": shelxl,
        },
    }
    (out / "combined_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    lines = [
        "# Multi-Crystal Native Nuisance Combination",
        "",
        "The per-dataset nuisance fits are kept separate. This report concatenates signed native rows after each dataset has its own scales/thickness/cell/geometry treatment.",
        "",
        "| dataset | role | raw/endpoint weight | rows native/endpoint | unique signed | unique model/Friedel | basis median/p95/max | dyn R1 all | dyn wR2 all | born R1 all | born wR2 all | q p05/median/p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in dataset_summaries:
        bn = ds["best_native_summary"]
        m = bn.get("metrics", {})
        dyn = ((m.get("all") or {}).get("dyn") or {})
        born = ((m.get("all") or {}).get("born") or {})
        bstats = ds.get("basis_size_stats", {})
        lines.append(
            f"| {ds['name']} | {ds['dataset_role']} | {safe_float(ds.get('raw_row_weight')):.3f}/{safe_float(ds.get('endpoint_weight')):.3f} | "
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
        lines.append("")
        lines.append(
            f"Lambda `{args.endpoint_lambda}` merged `{len(endpoint_rows)}` unmerged endpoint rows to `{len(merged_endpoint)}` corrected endpoint rows. SHELXL status `{shelxl.get('status')}`; `R1gt={100*safe_float(shelxl.get('R1_gt')):.2f}%`, `R1all={100*safe_float(shelxl.get('R1_all')):.2f}%`, `wR2all={100*safe_float(shelxl.get('wR2_all')):.2f}%`, `GooF={safe_float(shelxl.get('GooF')):.3f}`."
        )
        if int(safe_float(shelxl.get("damp_retry_count"), 0) or 0) > 0 or shelxl.get("active_damp") is not None:
            lines.append(
                f"SHELXL DAMP retry: tried `{int(safe_float(shelxl.get('damp_retry_count'), 0) or 0)}` retry step(s); "
                f"accepted `{bool(shelxl.get('damp_retry_accepted'))}`; active DAMP `{safe_float(shelxl.get('active_damp')):.3f}` "
                f"LIMSE `{safe_float(shelxl.get('active_damp_limse')):.3f}`."
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
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out.resolve()), "report": str((out / "REPORT.md").resolve()), "native_wR2_all": native["dyn"]["all"]["wR2_all"], "endpoint": shelxl}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
