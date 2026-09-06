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
import time
from pathlib import Path

import numpy as np

try:  # Package import.
    from .bloch_engine import parse_model, sha256
    from .endpoint_modulation import modulation_manifest_fields
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
    from endpoint_modulation import modulation_manifest_fields
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


def file_provenance(path: Path) -> dict[str, object]:
    """Return a stable source-artifact identity for a combined endpoint."""

    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": int(resolved.stat().st_size) if exists else None,
        "sha256": sha256(resolved) if exists else "",
    }


def scale_contract_provenance(summary: dict, rows: list[dict]) -> dict[str, object]:
    """Describe the scale table actually materialized in one source run."""

    best_native = summary.get("best_native") or {}
    nuisance = best_native.get("nuisance") or {}
    row_model_fields = ("scale_replay_model", "scale_compression_model")
    model_names: list[str] = []
    source = ""
    for field in row_model_fields:
        values = sorted(
            {
                str(row.get(field, "")).strip()
                for row in rows
                if str(row.get(field, "")).strip()
            }
        )
        if values:
            model_names = values
            source = f"row_table.{field}"
            break

    if not model_names:
        replay_candidate = (summary.get("scale_contract_replay") or {}).get("candidate") or {}
        replay_name = str(replay_candidate.get("name", "")).strip()
        compression_candidate = (summary.get("scale_compression") or {}).get("selected_output") or {}
        compression_name = str(compression_candidate.get("name", "")).strip()
        candidates = (
            (replay_name, "summary.scale_contract_replay.candidate.name"),
            (str(nuisance.get("scale_replay_model", "")).strip(), "summary.best_native.nuisance.scale_replay_model"),
            (compression_name, "summary.scale_compression.selected_output.name"),
            (str(nuisance.get("scale_compression_model", "")).strip(), "summary.best_native.nuisance.scale_compression_model"),
        )
        for candidate, candidate_source in candidates:
            if candidate:
                model_names = [candidate]
                source = candidate_source
                break

    if not model_names:
        group_width = safe_float(nuisance.get("group_width_deg"))
        detector_bins = safe_float(nuisance.get("detector_scale_bins"))
        if math.isfinite(group_width) or math.isfinite(detector_bins):
            width_text = f"{group_width:g}" if math.isfinite(group_width) else "unknown"
            detector_text = str(int(detector_bins)) if math.isfinite(detector_bins) else "unknown"
            model_names = [f"group_width_{width_text}_deg_det_{detector_text}"]
            source = "summary.best_native.nuisance"

    count_value = safe_float(best_native.get("scale_count"))
    count = int(count_value) if math.isfinite(count_value) and count_value >= 0.0 else None
    return {
        "model": model_names[0] if len(model_names) == 1 else ("mixed" if model_names else "unknown"),
        "models": model_names,
        "model_source": source or "unavailable",
        "parameter_count": count,
        "fit_split": str(best_native.get("scale_fit_split", "")) or "unknown",
    }


def d_spacing_from_model(model: dict, h: int, k: int, ell: int) -> float:
    """Return d(hkl) from the parsed model cell without changing HKL signs."""

    try:
        a, b, c, alpha_deg, beta_deg, gamma_deg = [float(value) for value in model["cell"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("parsed model does not contain a usable six-value unit cell") from exc
    alpha, beta, gamma = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    direct_metric = np.asarray(
        [
            [a * a, a * b * math.cos(gamma), a * c * math.cos(beta)],
            [a * b * math.cos(gamma), b * b, b * c * math.cos(alpha)],
            [a * c * math.cos(beta), b * c * math.cos(alpha), c * c],
        ],
        dtype=float,
    )
    try:
        reciprocal_metric = np.linalg.inv(direct_metric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("parsed model unit cell is singular") from exc
    hkl = np.asarray([int(h), int(k), int(ell)], dtype=float)
    inv_d2 = float(hkl @ reciprocal_metric @ hkl)
    if not math.isfinite(inv_d2) or inv_d2 <= 0.0:
        raise ValueError(f"cannot calculate a finite d spacing for HKL {(h, k, ell)}")
    return 1.0 / math.sqrt(inv_d2)


def filter_native_rows_by_resolution(rows: list[dict], model: dict, min_d_spacing: float) -> tuple[list[dict], dict]:
    """Apply a shared-model-cell endpoint/reporting d cutoff to native rows.

    Native tables may carry ``d_A`` calculated from each experiment's fitted
    cell.  A merged paper endpoint has one structure cell, so its resolution
    boundary must instead be calculated from signed HKL and that shared cell.
    """

    cutoff = float(min_d_spacing)
    kept: list[dict] = []
    input_d_spacings: list[float] = []
    retained_d_spacings: list[float] = []
    stored_membership_disagreements = 0
    for row in rows:
        try:
            h, k, ell = (int(float(row[key])) for key in ("h", "k", "l"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resolution filtering needs integer h,k,l in every native row") from exc
        d_value = d_spacing_from_model(model, h, k, ell)
        input_d_spacings.append(d_value)
        stored_d = safe_float(row.get("d_A"))
        if cutoff > 0.0 and math.isfinite(stored_d) and stored_d > 0.0 and ((stored_d + 1e-12 >= cutoff) != (d_value + 1e-12 >= cutoff)):
            stored_membership_disagreements += 1
        if cutoff <= 0.0 or d_value + 1e-12 >= cutoff:
            kept.append(row)
            retained_d_spacings.append(d_value)
    return kept, {
        "requested_min_d_spacing_A": cutoff,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "retained_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "retained_fraction": float(len(kept) / len(rows)) if rows else float("nan"),
        "actual_input_min_d_spacing_A": min(input_d_spacings) if input_d_spacings else float("nan"),
        "actual_retained_min_d_spacing_A": min(retained_d_spacings) if retained_d_spacings else float("nan"),
        "actual_retained_max_d_spacing_A": max(retained_d_spacings) if retained_d_spacings else float("nan"),
        "d_spacings_calculated_from_model": len(rows),
        "stored_d_A_membership_disagreements": stored_membership_disagreements,
        "cell_A_deg": [float(value) for value in model["cell"]],
    }


def filter_hkl_rows_by_resolution(rows: list[tuple], model: dict, min_d_spacing: float) -> list[tuple]:
    """Filter a corrected-HKL source with the same crystallographic cutoff."""

    cutoff = float(min_d_spacing)
    if cutoff <= 0.0:
        return rows
    return [
        row
        for row in rows
        if d_spacing_from_model(model, int(row[0]), int(row[1]), int(row[2])) + 1e-12 >= cutoff
    ]


def hkl_resolution_stats(rows: list[tuple], model: dict) -> dict[str, object]:
    """Report the discrete resolution range represented by HKL rows."""

    spacings = [d_spacing_from_model(model, int(row[0]), int(row[1]), int(row[2])) for row in rows]
    return {
        "rows": len(rows),
        "actual_min_d_spacing_A": min(spacings) if spacings else float("nan"),
        "actual_max_d_spacing_A": max(spacings) if spacings else float("nan"),
        "cell_A_deg": [float(value) for value in model["cell"]],
    }


def selected_native_wght(summary: dict) -> tuple[float, float]:
    """Read the native row weights written by native_row_regularize."""

    block = summary.get("shelx_weight") or {}
    a = safe_float(
        block.get("selected_native_a", block.get("source_a", block.get("a"))),
        0.126534,
    )
    b = safe_float(
        block.get("selected_native_b", block.get("source_b", block.get("b"))),
        0.0,
    )
    return a, b


def apply_native_row_weight(row: dict, weight: float) -> dict:
    """Apply one dataset-policy raw-row weight to an unmodified sigma."""

    out = dict(row)
    sigma = safe_float(out.get("sigma_eff"))
    if math.isfinite(sigma) and sigma > 0.0 and weight > 0.0 and abs(weight - 1.0) > 1e-12:
        out["sigma_eff"] = f"{sigma / math.sqrt(weight):.17g}"
    return out


def canonical_p222_friedel(h, k, ell):
    variants = [
        (h, k, ell),
        (-h, -k, ell),
        (h, -k, -ell),
        (-h, k, -ell),
    ]
    variants += [(-a, -b, -c) for a, b, c in variants]
    return min(variants)


def reciprocal_equivalent_keys(
    h: int,
    k: int,
    ell: int,
    symmetry_ops: list[dict],
    *,
    merge_friedel: bool = True,
) -> list[tuple[int, int, int]]:
    """Return reciprocal equivalents from the model symmetry operations."""

    hkl = np.asarray([int(h), int(k), int(ell)], dtype=float)
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


def canonical_model_friedel(h: int, k: int, ell: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    return min(reciprocal_equivalent_keys(h, k, ell, symmetry_ops, merge_friedel=True))


def canonical_model_symmetry(h: int, k: int, ell: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    return min(reciprocal_equivalent_keys(h, k, ell, symmetry_ops, merge_friedel=False))


def endpoint_key_function(mode: str, symmetry_ops: list[dict]):
    """Return the HKL key used for endpoint scaling or merging."""

    if mode == "exact_signed_hkl":
        return lambda h, k, ell: (int(h), int(k), int(ell))
    if mode == "model_symmetry":
        return lambda h, k, ell: canonical_model_symmetry(int(h), int(k), int(ell), symmetry_ops)
    if mode == "model_symmetry_friedel":
        return lambda h, k, ell: canonical_model_friedel(int(h), int(k), int(ell), symmetry_ops)
    raise ValueError(f"unknown endpoint merge key {mode!r}")


def shelx_weights(obs, sig, pred, a, b):
    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    return 1.0 / np.maximum(
        sig * sig + (np.asarray(a, dtype=float) * p) ** 2 + np.asarray(b, dtype=float) * p,
        1e-30,
    )


def metrics(rows: list[dict], pred_col: str, wght_a: float, wght_b: float):
    obs = np.asarray([safe_float(r["target_I"]) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r["sigma_eff"]) for r in rows], dtype=float)
    pred = np.asarray([safe_float(r[pred_col]) for r in rows], dtype=float)
    a = np.asarray([safe_float(r.get("native_wght_a"), wght_a) for r in rows], dtype=float)
    b = np.asarray([safe_float(r.get("native_wght_b"), wght_b) for r in rows], dtype=float)
    m = np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs = obs[m]
    sig = sig[m]
    pred = pred[m]
    a = a[m]
    b = b[m]
    if len(obs) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "wR2_all": float("nan"), "CC": float("nan"), "mw_abs": float("nan")}
    gt = obs > 2.0 * sig
    fobs = np.sqrt(np.maximum(obs, 0.0))
    fcalc = np.sqrt(np.maximum(pred, 0.0))

    def r1(mask):
        if not np.any(mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[mask] - fcalc[mask])) / max(np.sum(fobs[mask]), 1e-30))

    w = shelx_weights(obs, sig, pred, a, b)
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


def native_metrics_by_split(rows: list[dict], wght_a: float, wght_b: float) -> dict[str, dict[str, dict]]:
    """Return native metrics in the single-run summary's split-first shape."""

    dyn = split_metrics(rows, "dyn_scaled", wght_a, wght_b)
    born = split_metrics(rows, "born_scaled", wght_a, wght_b)
    return {
        split: {"dyn": dyn[split], "born": born[split]}
        for split in ("all", "work", "free")
    }


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


def endpoint_rows_and_corrections_from_native(rows: list[dict], lambda_q: float):
    """Return corrected endpoint rows and their applied intensity multipliers."""

    hkl = [(int(r["h"]), int(r["k"]), int(r["l"])) for r in rows]
    target = np.asarray([safe_float(r["target_I"]) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r["sigma_eff"]) for r in rows], dtype=float)
    restore = []
    for row in rows:
        mode = str(row.get("target_mode", "iobs"))
        value = row.get("source_to_target_factor")
        if value is None and mode != "iobs":
            raise ValueError(f"cannot restore {mode} endpoint units: row lacks source_to_target_factor")
        factor = 1.0 if value is None else safe_float(value)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"invalid source_to_target_factor for endpoint export: {value!r}")
        restore.append(factor)
    restore = np.asarray(restore, dtype=float)
    target = target / restore
    sig = sig / restore
    dyn_raw = np.asarray([safe_float(r["dyn_raw"]) for r in rows], dtype=float)
    dyn_scaled = np.asarray([safe_float(r["dyn_scaled"]) for r in rows], dtype=float)
    q = np.asarray([safe_float(r["q_dyn_over_born"]) for r in rows], dtype=float)
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn_raw)
        & (dyn_raw > 0.0)
        & np.isfinite(dyn_scaled)
        & (dyn_scaled > 0.0)
        & np.isfinite(q)
        & (q > 0.0)
    )
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
    selected = np.where(valid)[0]
    endpoint_rows = [(*hkl[i], float(icorr[i]), float(sigcorr[i])) for i in selected]
    corrections = np.asarray(float(factor) / denom[selected], dtype=float)
    return endpoint_rows, corrections


def endpoint_rows_from_native(rows: list[dict], lambda_q: float):
    """Compatibility wrapper returning only corrected endpoint rows."""

    endpoint_rows, _corrections = endpoint_rows_and_corrections_from_native(rows, lambda_q)
    return endpoint_rows


def apply_hkl_weight(rows, weight: float):
    if weight <= 0.0:
        return []
    if abs(weight - 1.0) < 1e-12:
        return rows
    factor = 1.0 / math.sqrt(max(weight, 1e-12))
    return [(h, k, ell, intensity, sig * factor) for h, k, ell, intensity, sig in rows]


def scale_hkl_rows(rows, scale: float):
    scale = float(scale)
    return [(h, k, ell, intensity * scale, sig * abs(scale)) for h, k, ell, intensity, sig in rows]


def audit_combined_endpoint_modulation(
    corrections_by_dataset: dict[str, np.ndarray | None],
    interdataset_scales: dict[str, float],
    *,
    endpoint_source: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Audit row corrections after applying relative interdataset scales."""

    combined_corrections: list[float] = []
    per_dataset: dict[str, dict[str, object]] = {}
    for name, source_corrections in corrections_by_dataset.items():
        dataset_scale = safe_float(interdataset_scales.get(name, 1.0), float("nan"))
        if source_corrections is None:
            fields = modulation_manifest_fields([])
            fields["modulation_audit"] = {
                **dict(fields["modulation_audit"]),
                "status": "unavailable_endpoint_source_hkl",
                "reason": "row-level source intensity multipliers are not recoverable from a prewritten HKL",
            }
        else:
            adjusted = np.asarray(source_corrections, dtype=float) * dataset_scale
            fields = modulation_manifest_fields(adjusted)
            combined_corrections.extend(float(value) for value in adjusted)
        per_dataset[name] = {**fields, "interdataset_scale": dataset_scale}

    combined = modulation_manifest_fields(combined_corrections)
    if endpoint_source == "hkl":
        combined["modulation_audit"] = {
            **dict(combined["modulation_audit"]),
            "status": "unavailable_endpoint_source_hkl",
            "reason": "combined modulation requires row-level native correction factors including interdataset scales",
        }
    return combined, per_dataset


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


def estimate_overlap_scales(rows_by_dataset: dict[str, list[tuple]], iterations: int = 6, canonicalizer=None) -> dict:
    """Estimate dataset scales from HKLs observed by distinct datasets.

    Replicates are first merged within each dataset.  Every scale update uses a
    leave-one-dataset-out reference, so a dataset can neither manufacture its
    own overlap nor pull its reference toward itself.
    """

    key_for = canonicalizer or canonical_p222_friedel
    names = list(rows_by_dataset)
    scales = {name: 1.0 for name in names}
    overlap_counts = {name: 0 for name in names}
    if len(names) <= 1:
        return {"iterations": 0, "overlap_counts": overlap_counts, "scales": scales}

    merged: dict[str, dict[tuple[int, int, int], tuple[float, float]]] = {}
    contributors: dict[tuple[int, int, int], set[str]] = {}
    for name, rows in rows_by_dataset.items():
        bins: dict[tuple[int, int, int], dict[str, float]] = {}
        for h, k, ell, intensity, sigma in rows:
            value = float(intensity)
            sigma_value = abs(float(sigma))
            if not (math.isfinite(value) and value > 0.0 and math.isfinite(sigma_value) and sigma_value > 0.0):
                continue
            key = key_for(int(h), int(k), int(ell))
            weight = 1.0 / (sigma_value * sigma_value)
            bucket = bins.setdefault(key, {"w": 0.0, "wI": 0.0})
            bucket["w"] += weight
            bucket["wI"] += weight * value
        merged[name] = {
            key: (bucket["wI"] / bucket["w"], math.sqrt(1.0 / bucket["w"]))
            for key, bucket in bins.items()
            if bucket["w"] > 0.0
        }
        for key in merged[name]:
            contributors.setdefault(key, set()).add(name)

    shared_keys = {key for key, datasets in contributors.items() if len(datasets) >= 2}
    adjacency = {name: set() for name in names}
    for key in shared_keys:
        datasets = contributors[key]
        for name in datasets:
            adjacency[name].update(datasets - {name})
    components: list[set[str]] = []
    unseen = set(names)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            new = adjacency[current] & unseen
            unseen -= new
            component |= new
            frontier.extend(new)
        components.append(component)

    overlap_counts = {
        name: int(sum(key in shared_keys for key in merged[name]))
        for name in names
    }
    if not shared_keys:
        return {"iterations": 0, "overlap_counts": overlap_counts, "scales": scales}

    n_iterations = max(1, int(iterations))
    for component in components:
        component_names = sorted(component)
        if len(component_names) <= 1 or not any(adjacency[name] for name in component_names):
            continue
        local_index = {name: i for i, name in enumerate(component_names)}
        equations = []
        rhs = []
        base_weights = []
        for key in shared_keys:
            present = sorted(contributors[key] & component)
            for i, left in enumerate(present):
                left_i, left_sigma = merged[left][key]
                for right in present[i + 1 :]:
                    right_i, right_sigma = merged[right][key]
                    row = np.zeros(len(component_names), dtype=float)
                    row[local_index[left]] = 1.0
                    row[local_index[right]] = -1.0
                    equations.append(row)
                    rhs.append(math.log(right_i) - math.log(left_i))
                    relative_variance = (left_sigma / left_i) ** 2 + (right_sigma / right_i) ** 2
                    base_weights.append(1.0 / max(relative_variance, 1e-12))
        if not equations:
            continue
        design = np.asarray(equations, dtype=float)
        target = np.asarray(rhs, dtype=float)
        weights = np.asarray(base_weights, dtype=float)
        robust = np.ones_like(weights)
        solution = np.zeros(len(component_names), dtype=float)
        for _ in range(n_iterations):
            sqrt_w = np.sqrt(weights * robust)
            weighted_design = design * sqrt_w[:, None]
            weighted_target = target * sqrt_w
            gauge_weight = math.sqrt(float(np.median(weights)) * max(1, len(component_names)))
            weighted_design = np.vstack([weighted_design, np.ones(len(component_names)) * gauge_weight])
            weighted_target = np.append(weighted_target, 0.0)
            solution = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
            residual = design @ solution - target
            mad = float(np.median(np.abs(residual - np.median(residual))))
            if mad <= 1e-12:
                break
            cutoff = 1.345 * 1.4826 * mad
            robust = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-30))
        solution -= float(np.mean(solution))
        for name, value in zip(component_names, solution, strict=False):
            scales[name] = float(math.exp(value))
    return {
        "iterations": n_iterations,
        "overlap_counts": {name: int(count) for name, count in overlap_counts.items()},
        "scales": {name: float(value) for name, value in scales.items()},
    }


def merge_hkl_rows(rows, sigma_policy: str = "formal", canonicalizer=None):
    key_for = canonicalizer or (lambda h, k, ell: (int(h), int(k), int(ell)))
    bins: dict[tuple[int, int, int], dict[str, float]] = {}
    for h, k, ell, intensity, sig in rows:
        key = key_for(int(h), int(k), int(ell))
        sig = max(abs(float(sig)), 1e-6)
        w = 1.0 / (sig * sig)
        b = bins.setdefault(key, {"w": 0.0, "wI": 0.0, "n": 0, "values": [], "weights": []})
        b["w"] += w
        b["wI"] += w * float(intensity)
        b["n"] += 1
        b["values"].append(float(intensity))
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
    ap.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Internal L.S./CGLS cycle count written to the combined corrected-HKL endpoint. Production keeps this at 1.",
    )
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=("freeze-zero", "inherit", "promote"),
        default="freeze-zero",
        help=(
            "SHELXL weight policy for the shared combined endpoint. freeze-zero is the production default after "
            "GROND sigma calibration; inherit preserves the source active WGHT; promote uses SHELXL recommendations on later passes."
        ),
    )
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
    ap.add_argument(
        "--min-d-spacing",
        "--endpoint-min-d-spacing",
        dest="min_d_spacing",
        type=float,
        default=0.0,
        help=(
            "Optional reporting/endpoint cutoff applied to existing native row tables before joint metrics, "
            "sigma calibration, merging, and SHELXL. Zero preserves every row in the supplied runs."
        ),
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
