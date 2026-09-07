#!/usr/bin/env python3
"""Promote held-out post-Bloch observation corrections into endpoint HKLs.

This is deliberately downstream of the Bloch calculation.  It consumes one or
more held-out-passing ``native_two_part_observation_correction.py`` outputs,
combines their smooth additive/log-multiplicative corrections, then writes
dynamic-corrected HKLs and SHELXL/map statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

if __package__:
    from .endpoint_modulation import modulation_manifest_fields
    from .native_row_regularize import (
        DEFAULT_HKL,
        DEFAULT_RES,
        SHELXL as DEFAULT_SHELXL,
        configure_shelxl_executable,
        enforce_hkl_row_width,
        exact_hkl_merge,
        fail_if_shelxl_became_unavailable,
        native_metrics,
        parse_wght,
        require_endpoint_shelxl,
        run_maps,
        run_shelxl,
        sha256,
        write_hkl,
        write_no_exti_ins,
    )
else:  # pragma: no cover - direct script execution
    from endpoint_modulation import modulation_manifest_fields
    from native_row_regularize import (
        DEFAULT_HKL,
        DEFAULT_RES,
        SHELXL as DEFAULT_SHELXL,
        configure_shelxl_executable,
        enforce_hkl_row_width,
        exact_hkl_merge,
        fail_if_shelxl_became_unavailable,
        native_metrics,
        parse_wght,
        require_endpoint_shelxl,
        run_maps,
        run_shelxl,
        sha256,
        write_hkl,
        write_no_exti_ins,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows read from {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fcol(rows: list[dict[str, str]], name: str, default: float = float("nan")) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(name, default)))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=float)


def icol(rows: list[dict[str, str]], name: str, default: int = -999999) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            vals.append(int(float(r.get(name, default))))
        except (TypeError, ValueError):
            vals.append(default)
    return np.asarray(vals, dtype=int)


def scol(rows: list[dict[str, str]], name: str, default: str = "") -> np.ndarray:
    return np.asarray([str(r.get(name, default)) for r in rows])


def parse_grid(text: str) -> list[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("empty lambda grid")
    return vals


def pct(v: float) -> float:
    return 100.0 * float(v) if np.isfinite(v) else float("nan")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def row_key_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return icol(rows, "row_index"), icol(rows, "h"), icol(rows, "k"), icol(rows, "l")


def assert_same_rows(base: list[dict[str, str]], other: list[dict[str, str]], label: str) -> None:
    if len(base) != len(other):
        raise ValueError(f"{label}: row count mismatch {len(other)} != {len(base)}")
    b = row_key_arrays(base)
    o = row_key_arrays(other)
    for name, lhs, rhs in zip(("row_index", "h", "k", "l"), b, o, strict=False):
        if not np.array_equal(lhs, rhs):
            raise ValueError(f"{label}: {name} values do not match row table")


def load_candidate(path: Path, require_pass: bool) -> dict:
    cdir = path if path.is_dir() else path.parent
    summary = load_json(cdir / "summary.json")
    if require_pass and not bool(summary.get("passes_heldout_gate", False)):
        raise ValueError(f"{cdir} did not pass held-out gate")
    rows = read_rows(cdir / "two_part_corrected_rows.csv")
    return {
        "dir": cdir,
        "summary": summary,
        "rows": rows,
        "additive": fcol(rows, "two_part_additive", 0.0),
        "log_mult": fcol(rows, "two_part_log_mult", 0.0),
        "pred_after": fcol(rows, "two_part_pred_after", float("nan")),
        "model": ((summary.get("best_model") or {}).get("name") or ""),
        "all_wR2": float(((summary.get("best_model") or {}).get("metrics") or {}).get("all", {}).get("wR2_all", float("inf"))),
        "free_wR2": float(((summary.get("best_model") or {}).get("metrics") or {}).get("free", {}).get("wR2_all", float("inf"))),
    }


def choose_correction(candidates: list[dict], combine_mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if not candidates:
        raise ValueError("no correction candidates supplied")
    if combine_mode == "best_all":
        chosen = min(candidates, key=lambda c: c["all_wR2"])
        return chosen["additive"], chosen["log_mult"], {"mode": combine_mode, "chosen": str(chosen["dir"])}
    if combine_mode == "best_free":
        chosen = min(candidates, key=lambda c: c["free_wR2"])
        return chosen["additive"], chosen["log_mult"], {"mode": combine_mode, "chosen": str(chosen["dir"])}
    if combine_mode != "ensemble":
        raise ValueError(f"unknown combine mode {combine_mode!r}")
    add_stack = np.vstack([c["additive"] for c in candidates])
    log_stack = np.vstack([c["log_mult"] for c in candidates])
    return (
        np.nanmean(add_stack, axis=0),
        np.nanmean(log_stack, axis=0),
        {
            "mode": combine_mode,
            "candidate_count": len(candidates),
            "candidate_dirs": [str(c["dir"]) for c in candidates],
            "additive_pair_rms": float(np.sqrt(np.nanmean((add_stack[0] - add_stack[-1]) ** 2))) if len(candidates) >= 2 else 0.0,
            "log_mult_pair_rms": float(np.sqrt(np.nanmean((log_stack[0] - log_stack[-1]) ** 2))) if len(candidates) >= 2 else 0.0,
        },
    )


def metric_blocks(obs, sig, predictions: dict[str, np.ndarray], split: np.ndarray, wght_a: float, wght_b: float) -> dict:
    valid = np.isfinite(obs) & np.isfinite(sig) & (sig > 0)
    masks = {
        "all": valid,
        "work": valid & (split != "free"),
        "free": valid & (split == "free"),
    }
    out = {}
    for pred_name, pred in predictions.items():
        out[pred_name] = {name: native_metrics(obs, sig, pred, mask, wght_a, wght_b) for name, mask in masks.items()}
    return out


def correction_quality(obs, sig, base_pred, polished_pred, additive, log_mult, split, wght_a, wght_b) -> dict:
    metrics = metric_blocks(
        obs,
        sig,
        {"baseline_dyn": base_pred, "polished_observation": polished_pred},
        split,
        wght_a,
        wght_b,
    )
    base = metrics["baseline_dyn"]["all"]
    polished = metrics["polished_observation"]["all"]
    free_base = metrics["baseline_dyn"]["free"]
    free_polished = metrics["polished_observation"]["free"]
    return {
        "metrics": metrics,
        "accepted_raw": bool(
            polished["wR2_all"] < base["wR2_all"]
            and polished["R1_all"] <= base["R1_all"] + 0.002
            and (not np.isfinite(free_base["wR2_all"]) or free_polished["wR2_all"] <= free_base["wR2_all"] + 0.002)
        ),
        "all_wR2_gain": float(base["wR2_all"] - polished["wR2_all"]),
        "all_R1_gain": float(base["R1_all"] - polished["R1_all"]),
        "free_wR2_gain": float(free_base["wR2_all"] - free_polished["wR2_all"]) if np.isfinite(free_base["wR2_all"]) else float("nan"),
        "additive_stats": {
            "p05": float(np.nanpercentile(additive, 5)),
            "median": float(np.nanmedian(additive)),
            "p95": float(np.nanpercentile(additive, 95)),
        },
        "log_mult_stats": {
            "p05": float(np.nanpercentile(log_mult, 5)),
            "median": float(np.nanmedian(log_mult)),
            "p95": float(np.nanpercentile(log_mult, 95)),
        },
    }


def endpoint_arrays(rows: list[dict[str, str]], additive, log_mult, lambda_q: float):
    h = icol(rows, "h")
    k = icol(rows, "k")
    l = icol(rows, "l")
    obs = fcol(rows, "target_I")
    sig = fcol(rows, "sigma_eff")
    dyn_raw = fcol(rows, "dyn_raw")
    born_raw = fcol(rows, "born_raw")
    dyn_scaled = fcol(rows, "dyn_scaled")
    q = np.clip(dyn_raw / np.maximum(born_raw, 1e-30), 0.05, 20.0)
    scale = dyn_scaled / np.maximum(dyn_raw, 1e-30)
    clean_obs = obs - additive
    denom = np.maximum(np.exp(log_mult) * scale * (q ** float(lambda_q)), 1e-30)
    i_corr = clean_obs / denom
    sig_corr = sig / denom
    valid = np.isfinite(i_corr) & np.isfinite(sig_corr) & np.isfinite(q) & (sig_corr > 0) & (q > 0)
    return h, k, l, i_corr, sig_corr, valid, q, clean_obs, denom


def write_endpoint(
    out_dir: Path,
    rows: list[dict[str, str]],
    additive,
    log_mult,
    res_path: Path,
    lambda_q: float,
    hkl_format: str,
    rescale: str,
    make_maps: bool,
    shelxl_iterations: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, k, l, i_corr, sig_corr, valid, q, clean_obs, denom = endpoint_arrays(rows, additive, log_mult, lambda_q)
    factor = 1.0
    if rescale == "median-iobs":
        src = float(np.nanmedian(fcol(rows, "target_I")[valid]))
        dst = float(np.nanmedian(i_corr[valid]))
        factor = src / max(abs(dst), 1e-30)
        i_corr *= factor
        sig_corr *= abs(factor)
    modulation_audit = modulation_manifest_fields(float(factor) / np.maximum(denom, 1e-30), valid)
    hkls = np.vstack([h, k, l]).T
    rows_unmerged = [(int(h[i]), int(k[i]), int(l[i]), float(i_corr[i]), float(sig_corr[i])) for i in np.where(valid)[0]]
    rows_merged = exact_hkl_merge(hkls[valid], i_corr[valid], sig_corr[valid])
    rows_merged, rows_unmerged, hkl_safe_factor = enforce_hkl_row_width(
        rows_merged, rows_unmerged, hkl_format
    )
    write_hkl(out_dir / "dyn_observation_polished_unmerged.hkl", rows_unmerged, hkl_format)
    write_hkl(out_dir / "dyn_observation_polished.hkl", rows_merged, hkl_format)
    write_no_exti_ins(res_path, out_dir / "dyn_observation_polished.ins", hkl_name="dyn_observation_polished.hkl")
    shelxl = run_shelxl(out_dir, "dyn_observation_polished", iterations=shelxl_iterations)
    maps = run_maps(out_dir, "dyn_observation_polished") if make_maps else {}
    endpoint_res = out_dir / "dyn_observation_polished.res"
    endpoint_hkl = out_dir / "dyn_observation_polished.hkl"
    endpoint_lst = out_dir / "dyn_observation_polished.lst"
    manifest = {
        "lambda_q": float(lambda_q),
        "endpoint_b": 0.0,
        "keep_exti": False,
        "q_min": 0.05,
        "q_max": 20.0,
        "rescale": rescale,
        "rescale_factor": float(factor),
        "hkl_safe_factor": hkl_safe_factor,
        **modulation_audit,
        "rows_unmerged": len(rows_unmerged),
        "rows_merged_exact_hkl": len(rows_merged),
        "clean_obs_p05": float(np.nanpercentile(clean_obs[valid], 5)),
        "clean_obs_median": float(np.nanmedian(clean_obs[valid])),
        "clean_obs_p95": float(np.nanpercentile(clean_obs[valid], 95)),
        "denom_p05": float(np.nanpercentile(denom[valid], 5)),
        "denom_median": float(np.nanmedian(denom[valid])),
        "denom_p95": float(np.nanpercentile(denom[valid], 95)),
        "q_median": float(np.nanmedian(q[valid])),
        "q_p05": float(np.nanpercentile(q[valid], 5)),
        "q_p95": float(np.nanpercentile(q[valid], 95)),
        "shelxl": shelxl,
        "maps": maps,
        "hkl": str((out_dir / "dyn_observation_polished.hkl").resolve()),
        "hkl_unmerged": str((out_dir / "dyn_observation_polished_unmerged.hkl").resolve()),
        "artifacts": {
            "res": str(endpoint_res.resolve()),
            "res_sha256": sha256(endpoint_res) if endpoint_res.exists() else "",
            "hkl": str(endpoint_hkl.resolve()),
            "hkl_sha256": sha256(endpoint_hkl) if endpoint_hkl.exists() else "",
            "lst": str(endpoint_lst.resolve()),
            "lst_sha256": sha256(endpoint_lst) if endpoint_lst.exists() else "",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    return manifest


def copy_promoted_endpoint(best: dict, out_dir: Path) -> dict:
    src_dir = Path(best["endpoint_dir"])
    promoted = out_dir / "promoted_endpoint"
    if promoted.exists():
        shutil.rmtree(promoted)
    shutil.copytree(src_dir, promoted)
    copied = dict(best)
    copied["endpoint_dir"] = str(promoted.resolve())
    copied["hkl"] = str((promoted / "dyn_observation_polished.hkl").resolve())
    copied["hkl_unmerged"] = str((promoted / "dyn_observation_polished_unmerged.hkl").resolve())
    return copied


def report_lines(summary: dict) -> list[str]:
    raw = summary["raw_row_quality"]
    base = raw["metrics"]["baseline_dyn"]["all"]
    pol = raw["metrics"]["polished_observation"]["all"]
    free_base = raw["metrics"]["baseline_dyn"]["free"]
    free_pol = raw["metrics"]["polished_observation"]["free"]
    best = summary.get("best_endpoint") or {}
    control = summary.get("best_control_endpoint") or {}
    sh = best.get("shelxl") or {}
    control_sh = control.get("shelxl") or {}
    lines = [
        "# Observation-Polished Endpoint",
        "",
        f"Promotion status: `{summary['accepted']}`. Combine mode: `{summary['correction']['mode']}`.",
        "",
        "## Raw Row-Space Metrics",
        "",
        "| split | model | n | n_gt | R1all % | R1strong % | R2all % | wR2all % | CC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("all", "work", "free"):
        for model_name, block in raw["metrics"].items():
            m = block[split_name]
            lines.append(
                f"| {split_name} | {model_name} | {m['n']} | {m['n_gt']} | {pct(m['R1_all']):.3f} | "
                f"{pct(m['R1_gt']):.3f} | {pct(m['R2_all']):.3f} | {pct(m['wR2_all']):.3f} | {m['CC']:.5f} |"
            )
    lines.extend(
        [
            "",
            f"All-row gains: `R1all {pct(base['R1_all']):.3f}% -> {pct(pol['R1_all']):.3f}%`, "
            f"`wR2all {pct(base['wR2_all']):.3f}% -> {pct(pol['wR2_all']):.3f}%`. "
            f"Free-row `wR2all {pct(free_base['wR2_all']):.3f}% -> {pct(free_pol['wR2_all']):.3f}%`.",
            "",
            "Control endpoint is the same row table and lambda grid with no additive/log-multiplicative observation polish.",
            "",
            "## SHELXL-Style Endpoint Metrics",
            "",
            "| family | lambda_q | accepted | R1gt % | R1all % | wR2all % | GooF | Rint | Rsigma | Ngt/Nall | unique |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ep in summary["control_endpoint_results_sorted"]:
        e = ep["shelxl"]
        lines.append(
            f"| control | {ep['lambda_q']:.3f} | false | {pct(e.get('R1_gt', float('nan'))):.3f} | "
            f"{pct(e.get('R1_all', float('nan'))):.3f} | {pct(e.get('wR2_all', float('nan'))):.3f} | "
            f"{e.get('GooF', float('nan')):.3f} | {e.get('Rint', float('nan')):.4f} | {e.get('Rsigma', float('nan')):.4f} | "
            f"{e.get('N_gt', '')}/{e.get('N_all', '')} | {e.get('unique', '')} |"
        )
    for ep in summary["endpoint_results_sorted"]:
        e = ep["shelxl"]
        lines.append(
            f"| polished | {ep['lambda_q']:.3f} | {ep.get('accepted_endpoint', False)} | {pct(e.get('R1_gt', float('nan'))):.3f} | "
            f"{pct(e.get('R1_all', float('nan'))):.3f} | {pct(e.get('wR2_all', float('nan'))):.3f} | "
            f"{e.get('GooF', float('nan')):.3f} | {e.get('Rint', float('nan')):.4f} | {e.get('Rsigma', float('nan')):.4f} | "
            f"{e.get('N_gt', '')}/{e.get('N_all', '')} | {e.get('unique', '')} |"
        )
    if control_sh:
        lines.extend(
            [
                "",
                f"Best control endpoint: `lambda_q={control['lambda_q']:.3f}`, "
                f"`R1all={pct(control_sh.get('R1_all', float('nan'))):.3f}%`, "
                f"`wR2all={pct(control_sh.get('wR2_all', float('nan'))):.3f}%`.",
            ]
        )
    if sh:
        lines.extend(
            [
                "",
                f"Best endpoint: `lambda_q={best['lambda_q']:.3f}`, `R1gt={pct(sh.get('R1_gt', float('nan'))):.3f}%`, "
                f"`R1all={pct(sh.get('R1_all', float('nan'))):.3f}%`, `wR2all={pct(sh.get('wR2_all', float('nan'))):.3f}%`.",
            ]
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- `{summary['summary_json']}`",
            f"- `{summary.get('promoted_endpoint', {}).get('endpoint_dir', '')}`",
        ]
    )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL), help="Recorded for provenance; row-table HKLs are used for endpoint writing.")
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidate-dir", action="append", required=True)
    ap.add_argument("--combine-mode", choices=["ensemble", "best_all", "best_free"], default="ensemble")
    ap.add_argument(
        "--lambda-grid",
        default="1.0",
        help="Diagnostic q-exponent grid. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    ap.add_argument("--make-maps", action="store_true")
    ap.add_argument("--shelxl", default=DEFAULT_SHELXL)
    ap.add_argument("--shelxl-iterations", type=int, default=5)
    ap.add_argument("--require-passed-candidates", action="store_true", default=True)
    ap.add_argument("--no-require-passed-candidates", dest="require_passed_candidates", action="store_false")
    ap.add_argument("--min-raw-wR2-gain", type=float, default=0.002)
    ap.add_argument("--endpoint-wR2-tolerance", type=float, default=0.0005)
    args = ap.parse_args()
    configure_shelxl_executable(args.shelxl)
    require_endpoint_shelxl(args.shelxl, skip_endpoint_refinement=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_table = Path(args.row_table)
    rows = read_rows(row_table)
    candidates = [load_candidate(Path(p), args.require_passed_candidates) for p in args.candidate_dir]
    for c in candidates:
        assert_same_rows(rows, c["rows"], str(c["dir"]))
    additive, log_mult, correction = choose_correction(candidates, args.combine_mode)

    obs = fcol(rows, "target_I")
    sig = np.maximum(fcol(rows, "sigma_eff"), 1e-12)
    dyn_scaled = fcol(rows, "dyn_scaled")
    born_scaled = fcol(rows, "born_scaled")
    split = scol(rows, "split", "work")
    wght_a, wght_b = parse_wght(Path(args.res))
    polished_pred = dyn_scaled * np.exp(log_mult) + additive
    raw_quality = correction_quality(obs, sig, dyn_scaled, polished_pred, additive, log_mult, split, wght_a, wght_b)
    raw_quality["metrics"]["born_scaled"] = metric_blocks(obs, sig, {"born_scaled": born_scaled}, split, wght_a, wght_b)["born_scaled"]

    endpoint_results = []
    control_results = []
    zeros = np.zeros(len(rows), dtype=float)
    for lam in parse_grid(args.lambda_grid):
        control_dir = out_dir / f"control_endpoint_lam_{str(lam).replace('.', 'p')}"
        control = write_endpoint(control_dir, rows, zeros, zeros, Path(args.res), lam, args.hkl_float_format, args.rescale, False, args.shelxl_iterations)
        control["endpoint_dir"] = str(control_dir.resolve())
        control_results.append(control)
        ep_dir = out_dir / f"endpoint_lam_{str(lam).replace('.', 'p')}"
        manifest = write_endpoint(ep_dir, rows, additive, log_mult, Path(args.res), lam, args.hkl_float_format, args.rescale, args.make_maps, args.shelxl_iterations)
        manifest["endpoint_dir"] = str(ep_dir.resolve())
        endpoint_results.append(manifest)
    fail_if_shelxl_became_unavailable(
        [*control_results, *endpoint_results],
        context="observation-polish endpoint refinement",
    )
    control_results = sorted(control_results, key=lambda r: (r["shelxl"].get("wR2_all", 999), r["shelxl"].get("R1_all", 999)))
    endpoint_results = sorted(endpoint_results, key=lambda r: (r["shelxl"].get("wR2_all", 999), r["shelxl"].get("R1_all", 999)))
    best_control = control_results[0] if control_results else {}
    best_endpoint = endpoint_results[0] if endpoint_results else {}

    raw_gain_ok = raw_quality["all_wR2_gain"] >= args.min_raw_wR2_gain
    endpoint_ok = bool(best_endpoint and best_endpoint.get("shelxl", {}).get("wR2_all", 999) < 999)
    control_sh = best_control.get("shelxl", {}) if best_control else {}
    polished_sh = best_endpoint.get("shelxl", {}) if best_endpoint else {}
    endpoint_not_worse = bool(
        endpoint_ok
        and control_sh
        and polished_sh.get("wR2_all", 999) <= control_sh.get("wR2_all", -999) + float(args.endpoint_wR2_tolerance)
        and polished_sh.get("R1_all", 999) <= control_sh.get("R1_all", -999) + 0.002
    )
    for ep in endpoint_results:
        ep["accepted_endpoint"] = bool(endpoint_not_worse and ep is best_endpoint and raw_gain_ok)
    accepted = bool(raw_quality["accepted_raw"] and raw_gain_ok and endpoint_not_worse)
    promoted = copy_promoted_endpoint(best_endpoint, out_dir) if accepted and best_endpoint else {}

    summary = {
        "inputs": {
            "row_table": str(row_table.resolve()),
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "candidate_dirs": [str(Path(p).resolve()) for p in args.candidate_dir],
        },
        "correction": correction,
        "candidate_summaries": [
            {
                "dir": str(c["dir"]),
                "model": c["model"],
                "passes_heldout_gate": bool(c["summary"].get("passes_heldout_gate", False)),
                "all_wR2": c["all_wR2"],
                "free_wR2": c["free_wR2"],
            }
            for c in candidates
        ],
        "raw_row_quality": raw_quality,
        "control_endpoint_results_sorted": control_results,
        "best_control_endpoint": best_control,
        "endpoint_gate": {
            "raw_gain_ok": raw_gain_ok,
            "endpoint_not_worse": endpoint_not_worse,
            "endpoint_wR2_tolerance": float(args.endpoint_wR2_tolerance),
            "control_wR2_all": control_sh.get("wR2_all"),
            "polished_wR2_all": polished_sh.get("wR2_all"),
            "control_R1_all": control_sh.get("R1_all"),
            "polished_R1_all": polished_sh.get("R1_all"),
        },
        "endpoint_results_sorted": endpoint_results,
        "best_endpoint": best_endpoint,
        "accepted": accepted,
        "promoted_endpoint": promoted,
        "summary_json": str((out_dir / "summary.json").resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    (out_dir / "REPORT.md").write_text("\n".join(report_lines(summary)) + "\n")

    corrected_rows = []
    for r, add, lm, pp in zip(rows, additive, log_mult, polished_pred, strict=False):
        rr = dict(r)
        rr["promoted_additive"] = f"{float(add):.10g}"
        rr["promoted_log_mult"] = f"{float(lm):.10g}"
        rr["promoted_pred_after"] = f"{float(pp):.10g}"
        corrected_rows.append(rr)
    write_csv(out_dir / "promoted_observation_rows.csv", corrected_rows)
    print(json.dumps({"out_dir": str(out_dir.resolve()), "accepted": accepted, "best_endpoint": best_endpoint, "report": str((out_dir / "REPORT.md").resolve())}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
