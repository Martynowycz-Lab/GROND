#!/usr/bin/env python3
"""Replay frozen GROND scale contracts through corrected-HKL endpoints.

This diagnostic starts from an existing ``native_dyn_calc.csv`` row table, fits
one or more lower-dimensional scale contracts without recomputing Bloch waves,
writes corrected HKL files, and runs SHELXL.  It is intended to answer whether a
compact scale model remains endpoint-safe, not to select thickness, geometry, or
the Bloch basis.
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

try:
    from .native_row_regularize import (
        exact_hkl_merge,
        parse_lst,
        parse_model,
        parse_recommended_wght,
        promote_recommended_wght,
        safe_fixed_width_factor,
        write_endpoint_ins,
        write_hkl,
    )
    from .post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
    from .scale_compression_polish import (
        adaptive_segment_groups,
        baseline_scale_log_by_fit,
        baseline_scale_log_by_row,
        candidate_row,
        coarsen_detector,
        coarsen_groups,
        col,
        count_scale_params,
        fit_group_detector_scales,
        fit_group_scales,
        infer_detector_base,
        infer_group_width_deg,
        information_criteria,
        load_baseline_counts,
        log_scale_step_rms,
        native_metrics,
        native_weight_from_summary,
        read_json,
        read_rows,
        row_mr_metrics,
        safe_float,
        split_masks,
        write_rows,
    )
except ImportError:  # pragma: no cover - direct script execution
    from native_row_regularize import (
        exact_hkl_merge,
        parse_lst,
        parse_model,
        parse_recommended_wght,
        promote_recommended_wght,
        safe_fixed_width_factor,
        write_endpoint_ins,
        write_hkl,
    )
    from post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
    from scale_compression_polish import (
        adaptive_segment_groups,
        baseline_scale_log_by_fit,
        baseline_scale_log_by_row,
        candidate_row,
        coarsen_detector,
        coarsen_groups,
        col,
        count_scale_params,
        fit_group_detector_scales,
        fit_group_scales,
        infer_detector_base,
        infer_group_width_deg,
        information_criteria,
        load_baseline_counts,
        log_scale_step_rms,
        native_metrics,
        native_weight_from_summary,
        read_json,
        read_rows,
        row_mr_metrics,
        safe_float,
        split_masks,
        write_rows,
    )


def parse_specs(text: str) -> list[str]:
    specs = [part.strip() for part in str(text).split(",") if part.strip()]
    return specs or ["baseline"]


def parse_spec_token(spec: str, group_width_deg: float) -> dict[str, object]:
    token = str(spec).strip()
    lower = token.lower()
    if lower == "baseline":
        return {"name": "baseline", "kind": "baseline", "detector_bins": "existing"}
    parts = lower.split(":")
    if parts[0] == "adaptive":
        if len(parts) < 2:
            raise ValueError(f"adaptive spec needs a segment count: {spec!r}")
        det = 8
        for part in parts[2:]:
            if part.startswith("det"):
                det = int(float(part[3:]))
        return {"name": f"adaptive_{int(float(parts[1]))}_det_{det}", "kind": "adaptive", "segments": int(float(parts[1])), "detector_bins": det}
    if parts[0] == "window":
        if len(parts) < 2:
            raise ValueError(f"window spec needs degrees: {spec!r}")
        deg = float(parts[1])
        if not np.isfinite(group_width_deg) or group_width_deg <= 0:
            raise ValueError("window specs require source group_width_deg in the summary")
        stride = max(1, int(round(deg / group_width_deg)))
        det = 8
        for part in parts[2:]:
            if part.startswith("det"):
                det = int(float(part[3:]))
        return {"name": f"window_{deg:g}deg_det_{det}", "kind": "fixed", "stride": stride, "window_deg": deg, "detector_bins": det}
    if parts[0] == "stride":
        if len(parts) < 2:
            raise ValueError(f"stride spec needs a stride: {spec!r}")
        stride = int(float(parts[1]))
        det = 8
        for part in parts[2:]:
            if part.startswith("det"):
                det = int(float(part[3:]))
        return {"name": f"stride_{stride}_det_{det}", "kind": "fixed", "stride": stride, "window_deg": stride * group_width_deg if np.isfinite(group_width_deg) else "", "detector_bins": det}
    raise ValueError(f"Unknown scale spec {spec!r}; use baseline, adaptive:N:det8, window:DEG:det8, or stride:N:det8")


def write_active_wght(ins_path: Path, a: float, b: float) -> None:
    lines = ins_path.read_text(errors="replace").splitlines()
    wght_line = f"WGHT      {float(a):.4f}      {float(b):.4f}"
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("WGHT"):
            lines[i] = wght_line
            break
    else:
        insert_at = 0
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("FVAR"):
                insert_at = i
                break
        lines.insert(insert_at, wght_line)
    ins_path.write_text("\n".join(lines).rstrip() + "\n")


def active_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    try:
        a = float(stats["active_wght_a"])
        b = float(stats.get("active_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def run_shelxl_replay(work_dir: Path, stem: str, executable: str, iterations: int, wght_mode: str) -> dict:
    history = []
    n_iter = max(1, int(iterations))
    exe = executable
    if "/" not in exe and shutil.which(exe) is None:
        return {"returncode": None, "status": "missing_shelxl", "executable": exe}
    freeze = wght_mode.startswith("freeze-")
    zero = wght_mode.endswith("zero")
    if zero:
        write_active_wght(work_dir / f"{stem}.ins", 0.0, 0.0)
    for i in range(n_iter):
        if freeze and zero:
            write_active_wght(work_dir / f"{stem}.ins", 0.0, 0.0)
        proc = subprocess.run([exe, stem], cwd=str(work_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (work_dir / f"shelxl_iter_{i + 1}.log").write_text(proc.stdout)
        lst = work_dir / f"{stem}.lst"
        stats = {"iteration": i + 1, "returncode": proc.returncode, **parse_lst(lst)}
        recommended_wght = parse_recommended_wght(lst)
        if recommended_wght is not None:
            stats["recommended_wght_a"] = recommended_wght[0]
            stats["recommended_wght_b"] = recommended_wght[1]
        history.append(stats)
        res = work_dir / f"{stem}.res"
        ins = work_dir / f"{stem}.ins"
        if i + 1 < n_iter and res.exists():
            write_endpoint_ins(res, ins, hkl_name=f"{stem}.hkl", keep_exti=False)
            if freeze:
                if zero:
                    write_active_wght(ins, 0.0, 0.0)
                    stats["next_wght"] = {"promoted": False, "frozen": True, "active_wght_a": 0.0, "active_wght_b": 0.0}
                else:
                    stats["next_wght"] = {"promoted": False, "frozen": True}
            else:
                stats["next_wght"] = promote_recommended_wght(ins, recommended_wght)
    if history:
        (work_dir / "shelxl_stdout.log").write_text((work_dir / f"shelxl_iter_{n_iter}.log").read_text(errors="replace"))
    final = dict(history[-1]) if history else {"returncode": None, "status": "not_run"}
    res = work_dir / f"{stem}.res"
    if final.get("status") in {None, "ok"} or res.exists():
        active = (0.0, 0.0) if zero else active_wght_from_stats(final)
        if res.exists():
            write_endpoint_ins(res, res, hkl_name=f"{stem}.hkl", keep_exti=False, wght=active)
            final["res_normalized_for_reopen"] = True
    final["iterations"] = n_iter
    final["iteration_history"] = history
    final["wght_mode"] = wght_mode
    return final


def endpoint_from_row_table(
    *,
    run_dir: Path,
    rows: list[dict[str, str]],
    res_path: Path,
    out_dir: Path,
    lambda_q: float,
    rescale: str,
    hkl_format: str,
    post_bloch_sigma_policy: str,
    post_bloch_sigma_holdout_fraction: float,
    post_bloch_sigma_min_scale: float,
    post_bloch_sigma_max_scale: float,
    shelxl: str,
    shelxl_iterations: int,
    wght_mode: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = col(rows, "h").astype(int)
    k = col(rows, "k").astype(int)
    l = col(rows, "l").astype(int)
    target = col(rows, "target_I")
    sig = col(rows, "sigma_eff")
    dyn_scaled = col(rows, "dyn_scaled")
    born_raw = col(rows, "born_raw")
    valid = np.isfinite(target) & np.isfinite(sig) & np.isfinite(dyn_scaled) & np.isfinite(born_raw) & (sig > 0) & (dyn_scaled > 0) & (born_raw > 0)
    correction = np.power(np.maximum(dyn_scaled / np.maximum(born_raw, 1e-30), 1e-30), float(lambda_q))
    Icorr = target / correction
    sigcorr = sig / correction
    if rescale == "median-iobs":
        factor = float(np.nanmedian(target[valid])) / max(float(np.nanmedian(Icorr[valid])), 1e-30)
        Icorr *= factor
        sigcorr *= factor
    else:
        factor = 1.0
    hkls = np.column_stack([h, k, l])
    rows_unmerged = [(h[i], k[i], l[i], Icorr[i], sigcorr[i]) for i in np.where(valid)[0]]
    rows_merged = exact_hkl_merge(hkls[valid], Icorr[valid], sigcorr[valid])
    hkl_safe_factor = 1.0
    if hkl_format == "shelx_fixed":
        hkl_safe_factor = min(safe_fixed_width_factor(rows_merged), safe_fixed_width_factor(rows_unmerged))
        if hkl_safe_factor < 1.0:
            rows_merged = [(hh, kk, ll, ii * hkl_safe_factor, ss * hkl_safe_factor) for hh, kk, ll, ii, ss in rows_merged]
            rows_unmerged = [(hh, kk, ll, ii * hkl_safe_factor, ss * hkl_safe_factor) for hh, kk, ll, ii, ss in rows_unmerged]
    post_bloch_sigma = {"policy": post_bloch_sigma_policy, "status": "not_requested"}
    if post_bloch_sigma_policy == "uniform_z_rms":
        try:
            model = parse_model(res_path)
            post_bloch_sigma = fit_uniform_z_rms_sigma_scale(
                rows_merged,
                model,
                holdout_fraction=float(post_bloch_sigma_holdout_fraction),
                min_scale=float(post_bloch_sigma_min_scale),
                max_scale=float(post_bloch_sigma_max_scale),
            )
            if str(post_bloch_sigma.get("status", "")).startswith("ok"):
                scale = float(post_bloch_sigma["sigma_scale"])
                rows_merged = scale_row_sigmas(rows_merged, scale)
                rows_unmerged = scale_row_sigmas(rows_unmerged, scale)
        except Exception as exc:
            post_bloch_sigma = {"policy": "uniform_z_rms", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_hkl(out_dir / "dyn_scale_replay.hkl", rows_merged, hkl_format)
    write_hkl(out_dir / "dyn_scale_replay_unmerged.hkl", rows_unmerged, hkl_format)
    write_endpoint_ins(res_path, out_dir / "dyn_scale_replay.ins", hkl_name="dyn_scale_replay.hkl", keep_exti=False)
    shelxl_stats = run_shelxl_replay(out_dir, "dyn_scale_replay", shelxl, shelxl_iterations, wght_mode)
    manifest = {
        "source_run_dir": str(run_dir),
        "row_table": str(run_dir / "native_dyn_calc.csv"),
        "res": str(res_path),
        "lambda_q": float(lambda_q),
        "rescale": rescale,
        "rescale_factor": factor,
        "hkl_safe_factor": hkl_safe_factor,
        "rows_unmerged": len(rows_unmerged),
        "rows_merged_exact_hkl": len(rows_merged),
        "post_bloch_sigma": post_bloch_sigma,
        "shelxl": shelxl_stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    return manifest


def build_replay_rows(
    *,
    rows: list[dict[str, str]],
    summary: dict,
    spec: dict[str, object],
    fit_split: str,
    detector_base: int,
    native_wght_a: float,
    native_wght_b: float,
    instability_u: float,
) -> tuple[list[dict[str, object]], dict]:
    obs = col(rows, "target_I")
    sig = col(rows, "sigma_eff")
    raw_dyn = col(rows, "dyn_raw")
    raw_born = col(rows, "born_raw")
    base_dyn = col(rows, "dyn_scaled")
    base_born = col(rows, "born_scaled")
    groups0 = col(rows, "group").astype(int)
    det0 = col(rows, "detector_bin").astype(int)
    masks = split_masks(rows)
    fit_mask = masks.get(fit_split, masks["all"])
    fit_good = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw_dyn) & (sig > 0) & (raw_dyn > 0)
    base_scale_count, base_total_params, physical_params = load_baseline_counts(summary, len(rows))
    if spec["kind"] == "baseline":
        dyn_scaled = base_dyn
        born_scaled = base_born
        scale_params = base_scale_count
        group_scale_params = base_scale_count
        detector_scale_params = 0
        groups = groups0
        det_bins = det0
        frame_scales = {}
    else:
        baseline_log_scale = baseline_scale_log_by_row(summary, groups0)
        if np.count_nonzero(np.isfinite(baseline_log_scale) & fit_good) < 2:
            baseline_log_scale = baseline_scale_log_by_fit(raw_dyn, base_dyn, groups0, fit_good)
        if spec["kind"] == "adaptive":
            groups, actual = adaptive_segment_groups(groups0, baseline_log_scale, fit_good, int(spec["segments"]))
            spec["actual_segments"] = actual
        else:
            groups = coarsen_groups(groups0, int(spec["stride"]), 0)
        det_target = int(spec.get("detector_bins", detector_base))
        if det_target <= 1:
            dyn_scaled, frame_scales = fit_group_scales(obs, sig, raw_dyn, groups, fit_mask, "native", native_wght_a, native_wght_b, instability_u)
            born_scaled, _ = fit_group_scales(obs, sig, raw_born, groups, fit_mask, "native", native_wght_a, native_wght_b, instability_u)
            det_bins = np.zeros(len(rows), dtype=int)
            detector_scales = {}
        else:
            det_bins = coarsen_detector(det0, detector_base, det_target)
            dyn_scaled, frame_scales, detector_scales = fit_group_detector_scales(obs, sig, raw_dyn, groups, det_bins, fit_mask, "native", native_wght_a, native_wght_b, instability_u)
            born_scaled, _, _ = fit_group_detector_scales(obs, sig, raw_born, groups, det_bins, fit_mask, "native", native_wght_a, native_wght_b, instability_u)
        scale_params, group_scale_params, detector_scale_params = count_scale_params(groups, det_bins, fit_good, int(spec.get("detector_bins", detector_base)))
    total_params = int(scale_params + physical_params)
    row_metrics = row_mr_metrics(obs, sig, dyn_scaled, masks["all"], instability_u, 3.0)
    row_metrics_work = row_mr_metrics(obs, sig, dyn_scaled, masks["work"], instability_u, 3.0)
    row_metrics_free = row_mr_metrics(obs, sig, dyn_scaled, masks["free"], instability_u, 3.0)
    native = native_metrics(obs, sig, dyn_scaled, masks["all"], native_wght_a, native_wght_b, n_params=total_params)
    native_work = native_metrics(
        obs,
        sig,
        dyn_scaled,
        masks["work"],
        native_wght_a,
        native_wght_b,
        n_params=total_params,
    )
    native_free = native_metrics(
        obs,
        sig,
        dyn_scaled,
        masks["free"],
        native_wght_a,
        native_wght_b,
        n_params=total_params,
    )
    candidate = {
        "name": spec["name"],
        "frame_model": str(spec["kind"]),
        "mode": "replay",
        "objective": "native",
        "group_stride": int(spec.get("stride", 0) or 0),
        "group_offset": 0,
        "adaptive_segments": spec.get("actual_segments", spec.get("segments", "")),
        "window_deg": spec.get("window_deg", ""),
        "detector_bins": int(spec.get("detector_bins", detector_base)) if spec["kind"] != "baseline" else "existing",
        "scale_params": int(scale_params),
        "group_scale_params": int(group_scale_params),
        "detector_scale_params": int(detector_scale_params),
        "total_params": int(total_params),
        "fit_scale_log_step_rms": log_scale_step_rms(frame_scales),
        **row_metrics,
        "row_work": row_metrics_work,
        "row_free": row_metrics_free,
        "native": native,
        "native_work": native_work,
        "native_free": native_free,
        "information": information_criteria(native, total_params),
    }
    updated = []
    fields = list(rows[0].keys())
    for extra in ("dyn_replay_source_scaled", "born_replay_source_scaled", "scale_replay_model"):
        if extra not in fields:
            fields.append(extra)
    for i, row in enumerate(rows):
        new = dict(row)
        new["dyn_replay_source_scaled"] = row.get("dyn_scaled", "")
        new["born_replay_source_scaled"] = row.get("born_scaled", "")
        new["dyn_scaled"] = f"{dyn_scaled[i]:.17g}" if np.isfinite(dyn_scaled[i]) else "nan"
        new["born_scaled"] = f"{born_scaled[i]:.17g}" if np.isfinite(born_scaled[i]) else "nan"
        new["scale_replay_model"] = str(spec["name"])
        updated.append(new)
    return updated, {"candidate": candidate, "fieldnames": fields}


def _finite_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _percent_or_blank(value) -> str:
    val = _finite_float(value)
    if not math.isfinite(val):
        return ""
    return f"{100.0 * val:.2f}%"


def choose_publication_endpoint(
    rows_out: list[dict[str, object]],
    *,
    goof_min: float,
    goof_max: float,
    max_r1_cost: float,
    free_wr2_tolerance: float,
    prefer_wght_mode: str,
    selection_policy: str = "bic",
    information_criterion: str = "bic_fitted_variance",
) -> dict[str, object]:
    """Choose a publication endpoint from replayed scale contracts.

    The low-R and held-out row branches are kept as diagnostics.  By default,
    the publication branch must pass only the hard endpoint sanity gates
    (preferred weight mode and conventional GooF range), then GROND selects the
    minimum-information-criterion scale contract.  The older "smallest scale
    count after R/free gates" rule remains available as ``gated-smallest`` for
    reproducibility of older audits.
    """

    finite = [
        row for row in rows_out
        if math.isfinite(_finite_float(row.get("shelxl_R1all")))
        and math.isfinite(_finite_float(row.get("shelxl_GooF")))
    ]
    if not finite:
        return {
            "status": "no_finite_endpoints",
            "publication_endpoint": None,
            "low_r_diagnostic": None,
            "rejections": [],
        }

    low_r = min(finite, key=lambda r: (_finite_float(r.get("shelxl_R1all"), float("inf")), int(r.get("scale_params", 10**9))))
    best_free = min(
        finite,
        key=lambda r: (
            _finite_float(r.get("native_free_wR2_all"), float("inf")),
            _finite_float(r.get("native_wR2_all"), float("inf")),
            int(r.get("scale_params", 10**9)),
        ),
    )
    low_r_r1 = _finite_float(low_r.get("shelxl_R1all"), float("inf"))
    best_free_wr2 = _finite_float(best_free.get("native_free_wR2_all"), float("inf"))
    prefer = str(prefer_wght_mode)
    policy = str(selection_policy or "bic").strip().lower()
    ic_name = str(information_criterion or "bic_fitted_variance").strip()
    if policy not in {"bic", "information", "gated-smallest", "legacy"}:
        policy = "bic"
    if policy == "information":
        policy = "bic"
    if policy == "legacy":
        policy = "gated-smallest"
    passing: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for row in finite:
        reasons = []
        goof = _finite_float(row.get("shelxl_GooF"), float("inf"))
        r1 = _finite_float(row.get("shelxl_R1all"), float("inf"))
        free_wr2 = _finite_float(row.get("native_free_wR2_all"), float("inf"))
        ic_value = _finite_float(row.get(ic_name), float("inf"))
        if str(row.get("wght_mode")) != prefer:
            reasons.append(f"not preferred weight mode {prefer!r}")
        if goof < float(goof_min) or goof > float(goof_max):
            reasons.append(f"GooF {goof:.3g} outside {goof_min:.3g}-{goof_max:.3g}")
        if policy == "bic" and not math.isfinite(ic_value):
            reasons.append(f"missing finite information criterion {ic_name!r}")
        if policy == "gated-smallest":
            if r1 > low_r_r1 + float(max_r1_cost):
                reasons.append(f"R1all cost {100.0 * (r1 - low_r_r1):.2f}% exceeds {100.0 * max_r1_cost:.2f}%")
            if free_wr2 > best_free_wr2 + float(free_wr2_tolerance):
                reasons.append(
                    f"held-out row wR2 cost {100.0 * (free_wr2 - best_free_wr2):.2f}% exceeds "
                    f"{100.0 * free_wr2_tolerance:.2f}%"
                )
        if reasons:
            rejections.append(
                {
                    "spec": row.get("spec"),
                    "wght_mode": row.get("wght_mode"),
                    "reasons": reasons,
                    "shelxl_R1all": row.get("shelxl_R1all"),
                    "shelxl_GooF": row.get("shelxl_GooF"),
                    "native_free_wR2_all": row.get("native_free_wR2_all"),
                    "scale_params": row.get("scale_params"),
                    information_criterion: row.get(information_criterion),
                    "endpoint_dir": row.get("endpoint_dir"),
                }
            )
        else:
            passing.append(row)

    if passing:
        if policy == "bic":
            selected = min(
                passing,
                key=lambda r: (
                    _finite_float(r.get(ic_name), float("inf")),
                    int(r.get("scale_params", 10**9)),
                    _finite_float(r.get("native_free_wR2_all"), float("inf")),
                    _finite_float(r.get("shelxl_R1all"), float("inf")),
                ),
            )
        else:
            selected = min(
                passing,
                key=lambda r: (
                    int(r.get("scale_params", 10**9)),
                    _finite_float(r.get("native_free_wR2_all"), float("inf")),
                    _finite_float(r.get("shelxl_R1all"), float("inf")),
                ),
            )
        status = "passed"
    else:
        selected = min(
            [r for r in finite if str(r.get("wght_mode")) == prefer] or finite,
            key=lambda r: (
                max(float(goof_min) - _finite_float(r.get("shelxl_GooF"), 0.0), 0.0)
                + max(_finite_float(r.get("shelxl_GooF"), float("inf")) - float(goof_max), 0.0),
                _finite_float(r.get(ic_name), float("inf")) if policy == "bic" else 0.0,
                max(_finite_float(r.get("shelxl_R1all"), float("inf")) - low_r_r1 - float(max_r1_cost), 0.0),
                int(r.get("scale_params", 10**9)),
            ),
        )
        status = "no_candidate_passed_all_gates"

    best_ic = None
    if policy == "bic":
        finite_ic = [_finite_float(r.get(ic_name), float("inf")) for r in passing]
        finite_ic = [v for v in finite_ic if math.isfinite(v)]
        best_ic = min(finite_ic) if finite_ic else None
    selected_ic = _finite_float(selected.get(ic_name), float("nan")) if selected else float("nan")
    selected_delta_ic = selected_ic - float(best_ic) if best_ic is not None and math.isfinite(selected_ic) else float("nan")
    r1_cost = _finite_float(selected.get("shelxl_R1all"), float("nan")) - low_r_r1 if selected else float("nan")
    free_wr2_cost = _finite_float(selected.get("native_free_wR2_all"), float("nan")) - best_free_wr2 if selected else float("nan")
    return {
        "status": status,
        "publication_endpoint": selected,
        "low_r_diagnostic": low_r,
        "best_heldout_row": best_free,
        "selection_policy": policy,
        "information_criterion": ic_name,
        "best_information_criterion": best_ic,
        "selected_information_criterion": selected_ic,
        "selected_delta_information_criterion": selected_delta_ic,
        "selected_r1_cost_vs_low_r": r1_cost,
        "selected_free_wr2_cost_vs_best_free": free_wr2_cost,
        "gates": {
            "goof_min": float(goof_min),
            "goof_max": float(goof_max),
            "max_r1_cost": float(max_r1_cost),
            "free_wr2_tolerance": float(free_wr2_tolerance),
            "preferred_wght_mode": prefer,
            "selection_policy": policy,
            "information_criterion": ic_name,
        },
        "passing_count": len(passing),
        "candidate_count": len(finite),
        "rejections": rejections,
    }


def write_publication_report(path: Path, selection: dict[str, object], rows_out: list[dict[str, object]], elapsed_s: float) -> None:
    selected = selection.get("publication_endpoint") or {}
    low_r = selection.get("low_r_diagnostic") or {}
    best_free = selection.get("best_heldout_row") or {}
    lines = [
        "# GROND Endpoint Calibration Report",
        "",
        "This report separates the low-R diagnostic corrected-HKL export from the BIC-selected publication export. "
        "The publication branch uses fixed row corrections, a declared scale contract, endpoint sigma calibration, "
        "and a fixed `WGHT 0 0` branch unless requested otherwise. The default selector applies hard endpoint sanity gates, "
        "then chooses the scale contract with the lowest information criterion.",
        "",
        f"Runtime: `{elapsed_s:.2f}` seconds.",
        f"Selection status: `{selection.get('status')}`.",
        f"Selection policy: `{selection.get('selection_policy')}`; "
        f"information criterion: `{selection.get('information_criterion')}`.",
        "",
        "## Selected Publication Endpoint",
        "",
    ]
    if selected:
        lines += [
            f"Spec: `{selected.get('spec')}`; weight mode: `{selected.get('wght_mode')}`.",
            f"Scale parameters: `{selected.get('scale_params')}`; total row-model parameters: `{selected.get('total_params')}`.",
            f"Selection IC: `{_finite_float(selected.get(str(selection.get('information_criterion') or 'bic_fitted_variance'))):.3f}`; "
            f"delta from best passing IC `{_finite_float(selection.get('selected_delta_information_criterion')):.3f}`.",
            f"SHELXL: R1gt `{_percent_or_blank(selected.get('shelxl_R1gt'))}`, "
            f"R1all `{_percent_or_blank(selected.get('shelxl_R1all'))}`, "
            f"wR2all `{_percent_or_blank(selected.get('shelxl_wR2all'))}`, "
            f"GooF `{_finite_float(selected.get('shelxl_GooF')):.3f}`.",
            f"Row-space all: R1all `{_percent_or_blank(selected.get('native_R1_all'))}`, "
            f"wR2all `{_percent_or_blank(selected.get('native_wR2_all'))}`, "
            f"S_GROND `{_finite_float(selected.get('native_GooF')):.3f}`.",
            f"Row-space held-out: R1all `{_percent_or_blank(selected.get('native_free_R1_all'))}`, "
            f"wR2all `{_percent_or_blank(selected.get('native_free_wR2_all'))}`, "
            f"S_GROND_free `{_finite_float(selected.get('native_free_GooF')):.3f}`.",
            f"Endpoint sigma scale: `{_finite_float(selected.get('sigma_scale')):.4g}`; "
            f"held-out RMS(z) after calibration `{_finite_float(selected.get('heldout_rms_z_after')):.3f}`.",
            f"Files: endpoint directory `{selected.get('endpoint_dir')}`; "
            f"RES `{selected.get('endpoint_res')}`; HKL `{selected.get('endpoint_hkl')}`; "
            f"LST `{selected.get('endpoint_lst')}`.",
            "",
        ]
    else:
        lines.append("No finite publication endpoint was available.")
        lines.append("")

    if low_r:
        r_cost = _finite_float(selected.get("shelxl_R1all")) - _finite_float(low_r.get("shelxl_R1all")) if selected else float("nan")
        lines += [
            "## Low-R Diagnostic Endpoint",
            "",
            f"Spec: `{low_r.get('spec')}`; weight mode: `{low_r.get('wght_mode')}`.",
            f"SHELXL: R1all `{_percent_or_blank(low_r.get('shelxl_R1all'))}`, "
            f"wR2all `{_percent_or_blank(low_r.get('shelxl_wR2all'))}`, "
            f"GooF `{_finite_float(low_r.get('shelxl_GooF')):.3f}`.",
            f"Publication R1all cost versus this low-R branch: `{100.0 * r_cost:.2f}%`.",
            f"Files: endpoint directory `{low_r.get('endpoint_dir')}`; "
            f"RES `{low_r.get('endpoint_res')}`; HKL `{low_r.get('endpoint_hkl')}`; "
            f"LST `{low_r.get('endpoint_lst')}`.",
            "",
        ]
    if best_free:
        lines += [
            "## Best Held-Out Row Endpoint",
            "",
            f"Spec: `{best_free.get('spec')}`; weight mode: `{best_free.get('wght_mode')}`; "
            f"held-out row wR2all `{_percent_or_blank(best_free.get('native_free_wR2_all'))}`.",
            "",
        ]

    lines += [
        "## Candidate Table",
        "",
        "| spec | WGHT mode | scale params | selection IC | R1all | wR2all | GooF | row free wR2all | sigma scale | endpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ic_name = str(selection.get("information_criterion") or "bic_fitted_variance")
    for row in sorted(
        rows_out,
        key=lambda r: (
            str(r.get("wght_mode")) != str((selection.get("gates") or {}).get("preferred_wght_mode", "freeze-zero")),
            _finite_float(r.get(ic_name), float("inf")),
            int(r.get("scale_params", 10**9)),
        ),
    ):
        lines.append(
            f"| `{row.get('spec')}` | `{row.get('wght_mode')}` | {row.get('scale_params')} | "
            f"{_finite_float(row.get(ic_name)):.3f} | "
            f"{_percent_or_blank(row.get('shelxl_R1all'))} | {_percent_or_blank(row.get('shelxl_wR2all'))} | "
            f"{_finite_float(row.get('shelxl_GooF')):.3f} | {_percent_or_blank(row.get('native_free_wR2_all'))} | "
            f"{_finite_float(row.get('sigma_scale')):.4g} | `{row.get('endpoint_dir')}` |"
        )

    if selection.get("rejections"):
        lines += ["", "## Gate Rejections", ""]
        for rejection in selection["rejections"][:30]:
            reasons = "; ".join(str(r) for r in rejection.get("reasons", []))
            lines.append(f"- `{rejection.get('spec')}` / `{rejection.get('wght_mode')}`: {reasons}")

    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--res", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--specs", default="baseline,adaptive:293:det8,adaptive:200:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8")
    ap.add_argument("--fit-split", default="")
    ap.add_argument(
        "--lambda-q",
        type=float,
        default=1.0,
        help="Diagnostic endpoint q exponent. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument("--shelxl", default="shelxl")
    ap.add_argument("--shelxl-iterations", type=int, default=5)
    ap.add_argument("--wght-modes", default="promote,freeze-zero")
    ap.add_argument("--publication-goof-min", type=float, default=0.85)
    ap.add_argument("--publication-goof-max", type=float, default=1.15)
    ap.add_argument("--publication-max-r1-cost", type=float, default=0.010)
    ap.add_argument("--publication-free-wr2-tolerance", type=float, default=0.040)
    ap.add_argument("--publication-prefer-wght-mode", default="freeze-zero")
    ap.add_argument(
        "--publication-selection-policy",
        choices=["bic", "gated-smallest"],
        default="bic",
        help="Endpoint scale-count selector. 'bic' applies hard sanity gates then chooses the minimum information criterion; 'gated-smallest' reproduces the older smallest-passing-scale rule.",
    )
    ap.add_argument(
        "--publication-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
        help="Information criterion used when --publication-selection-policy=bic.",
    )
    ap.add_argument("--instability-u", type=float, default=0.01)
    ap.add_argument("--native-wght-a", type=float, default=float("nan"))
    ap.add_argument("--native-wght-b", type=float, default=float("nan"))
    args = ap.parse_args()
    t0 = __import__("time").perf_counter()

    row_table = Path(args.row_table).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else row_table.parent / "summary.json"
    summary = read_json(summary_path)
    res_path = Path(args.res).expanduser().resolve() if args.res else Path(
        ((summary.get("inputs") or {}).get("res") or (summary.get("config") or {}).get("res") or "")
    ).expanduser().resolve()
    if not res_path.exists():
        raise SystemExit(f"Could not resolve source RES for endpoint replay: {res_path}")
    rows = read_rows(row_table)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    group_width_deg = infer_group_width_deg(summary)
    specs = [parse_spec_token(spec, group_width_deg) for spec in parse_specs(args.specs)]
    masks = split_masks(rows)
    fit_split = args.fit_split or str((summary.get("best_native") or {}).get("scale_fit_split") or "all")
    if fit_split not in masks:
        raise SystemExit(f"Unknown fit split {fit_split!r}; available {sorted(masks)}")
    detector_base = infer_detector_base(col(rows, "detector_bin").astype(int))
    native_wght_a, native_wght_b = native_weight_from_summary(summary, args.native_wght_a, args.native_wght_b)
    wght_modes = [m.strip() for m in str(args.wght_modes).split(",") if m.strip()]
    rows_out = []
    for spec in specs:
        spec_dir = out / re.sub(r"[^A-Za-z0-9_.-]+", "_", str(spec["name"]))
        replay_rows, meta = build_replay_rows(
            rows=rows,
            summary=summary,
            spec=spec,
            fit_split=fit_split,
            detector_base=detector_base,
            native_wght_a=native_wght_a,
            native_wght_b=native_wght_b,
            instability_u=float(args.instability_u),
        )
        spec_dir.mkdir(parents=True, exist_ok=True)
        write_rows(spec_dir / "native_dyn_calc.csv", replay_rows, meta["fieldnames"])
        spec_summary = {
            "source_row_table": str(row_table),
            "source_summary": str(summary_path),
            "source_res": str(res_path),
            "fit_split": fit_split,
            "spec": spec,
            **meta["candidate"],
        }
        (spec_dir / "summary.json").write_text(json.dumps(spec_summary, indent=2, allow_nan=True) + "\n")
        for wght_mode in wght_modes:
            endpoint_dir = spec_dir / f"endpoint_{wght_mode}"
            manifest = endpoint_from_row_table(
                run_dir=spec_dir,
                rows=replay_rows,
                res_path=res_path,
                out_dir=endpoint_dir,
                lambda_q=args.lambda_q,
                rescale=args.rescale,
                hkl_format=args.hkl_float_format,
                post_bloch_sigma_policy=args.post_bloch_sigma_policy,
                post_bloch_sigma_holdout_fraction=args.post_bloch_sigma_holdout_fraction,
                post_bloch_sigma_min_scale=args.post_bloch_sigma_min_scale,
                post_bloch_sigma_max_scale=args.post_bloch_sigma_max_scale,
                shelxl=args.shelxl,
                shelxl_iterations=args.shelxl_iterations,
                wght_mode=wght_mode,
            )
            sh = manifest.get("shelxl") or {}
            sig = manifest.get("post_bloch_sigma") or {}
            rows_out.append({
                "spec": spec["name"],
                "wght_mode": wght_mode,
                "scale_params": meta["candidate"]["scale_params"],
                "total_params": meta["candidate"]["total_params"],
                "row_Robs": meta["candidate"]["Robs"],
                "row_Rall": meta["candidate"]["Rall"],
                "row_wRall": meta["candidate"]["wRall"],
                "native_R1_gt": meta["candidate"]["native"]["R1_gt"],
                "native_R1_all": meta["candidate"]["native"]["R1_all"],
                "native_wR2_all": meta["candidate"]["native"]["wR2_all"],
                "native_GooF": meta["candidate"]["native"]["GooF"],
                "native_chi2": meta["candidate"]["native"]["chi2"],
                "native_work_R1_gt": meta["candidate"]["native_work"]["R1_gt"],
                "native_work_R1_all": meta["candidate"]["native_work"]["R1_all"],
                "native_work_wR2_all": meta["candidate"]["native_work"]["wR2_all"],
                "native_work_GooF": meta["candidate"]["native_work"]["GooF"],
                "native_free_R1_gt": meta["candidate"]["native_free"]["R1_gt"],
                "native_free_R1_all": meta["candidate"]["native_free"]["R1_all"],
                "native_free_wR2_all": meta["candidate"]["native_free"]["wR2_all"],
                "native_free_GooF": meta["candidate"]["native_free"]["GooF"],
                "aic_known_sigma": meta["candidate"]["information"]["aic_known_sigma"],
                "bic_known_sigma": meta["candidate"]["information"]["bic_known_sigma"],
                "aic_fitted_variance": meta["candidate"]["information"]["aic_fitted_variance"],
                "bic_fitted_variance": meta["candidate"]["information"]["bic_fitted_variance"],
                "shelxl_R1gt": sh.get("R1_gt", ""),
                "shelxl_Ngt": sh.get("N_gt", ""),
                "shelxl_R1all": sh.get("R1_all", ""),
                "shelxl_Nall": sh.get("N_all", ""),
                "shelxl_wR2all": sh.get("wR2_all", ""),
                "shelxl_GooF": sh.get("GooF", ""),
                "shelxl_Rint": sh.get("Rint", ""),
                "shelxl_Rsigma": sh.get("Rsigma", ""),
                "sigma_scale": sig.get("sigma_scale", ""),
                "heldout_rms_z_after": sig.get("heldout_rms_z_after", ""),
                "endpoint_dir": str(endpoint_dir),
                "endpoint_res": str(endpoint_dir / "dyn_scale_replay.res"),
                "endpoint_ins": str(endpoint_dir / "dyn_scale_replay.ins"),
                "endpoint_hkl": str(endpoint_dir / "dyn_scale_replay.hkl"),
                "endpoint_lst": str(endpoint_dir / "dyn_scale_replay.lst"),
                "endpoint_manifest": str(endpoint_dir / "manifest.json"),
                "endpoint_row_table": str(spec_dir / "native_dyn_calc.csv"),
            })
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    write_rows(out / "scale_contract_endpoint_replay.csv", rows_out, fieldnames)
    elapsed = float(__import__("time").perf_counter() - t0)
    selection = choose_publication_endpoint(
        rows_out,
        goof_min=float(args.publication_goof_min),
        goof_max=float(args.publication_goof_max),
        max_r1_cost=float(args.publication_max_r1_cost),
        free_wr2_tolerance=float(args.publication_free_wr2_tolerance),
        prefer_wght_mode=str(args.publication_prefer_wght_mode),
        selection_policy=str(args.publication_selection_policy),
        information_criterion=str(args.publication_information_criterion),
    )
    summary_out = {
        "rows": rows_out,
        "publication_selection": selection,
        "wall_seconds": elapsed,
    }
    (out / "summary.json").write_text(json.dumps(summary_out, indent=2, allow_nan=True) + "\n")
    write_publication_report(out / "REPORT.md", selection, rows_out, elapsed)
    print(json.dumps(summary_out, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
