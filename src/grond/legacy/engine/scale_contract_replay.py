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
            shutil.copy2(res, ins)
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
    native = native_metrics(obs, sig, dyn_scaled, masks["all"], native_wght_a, native_wght_b, n_params=total_params)
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
        "native": native,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--res", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--specs", default="baseline,adaptive:293:det8,adaptive:200:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8")
    ap.add_argument("--fit-split", default="")
    ap.add_argument("--lambda-q", type=float, default=1.0)
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument("--shelxl", default="shelxl")
    ap.add_argument("--shelxl-iterations", type=int, default=5)
    ap.add_argument("--wght-modes", default="promote,freeze-zero")
    ap.add_argument("--instability-u", type=float, default=0.01)
    ap.add_argument("--native-wght-a", type=float, default=float("nan"))
    ap.add_argument("--native-wght-b", type=float, default=float("nan"))
    args = ap.parse_args()

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
            })
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    write_rows(out / "scale_contract_endpoint_replay.csv", rows_out, fieldnames)
    (out / "summary.json").write_text(json.dumps({"rows": rows_out}, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"rows": rows_out}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
