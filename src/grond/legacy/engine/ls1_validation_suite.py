#!/usr/bin/env python3
"""Validate the GROND Bloch/LS1 fixed-point loop against native controls.

This script is intentionally a workflow wrapper around the already-audited row
regularizer, SHELXL feedback loop, and native finite-difference updater.  It
does not invent a new objective.  It runs the compact checks that distinguish a
useful lagged dynamical surrogate from a cosmetic corrected-HKL loop:

* updated-q LS1 macrocycles,
* frozen-q fixed corrected-HKL control,
* q=1 Born corrected-HKL control,
* optional alternate-hand Bloch control, and
* optional native finite-difference polish with the same accepted Bloch physics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import shelxl_feedback_loop as fb
else:  # pragma: no cover - direct script execution
    import shelxl_feedback_loop as fb


def shell_join(cmd: list[str]) -> str:
    """Return a copy-pasteable command line."""

    return " ".join(shlex.quote(str(x)) for x in cmd)


def read_summary(path: Path) -> dict:
    """Read a GROND JSON summary if it exists."""

    if not path.exists():
        return {}
    return json.loads(path.read_text())


def as_feedback_args(args: argparse.Namespace, *, lambda_grid: str, shelxl_iterations: int) -> argparse.Namespace:
    """Build the Namespace expected by shelxl_feedback_loop helpers."""

    return SimpleNamespace(
        python=args.python,
        hkl=args.hkl,
        baseline_hkl=args.baseline_hkl,
        hkl_transform=args.hkl_transform,
        min_d_spacing=args.min_d_spacing,
        min_peak=args.min_peak,
        geometry_percentile=args.geometry_percentile,
        lambda_grid=lambda_grid,
        shelxl=str(args.shelxl),
        shelxl_iterations=int(shelxl_iterations),
        endpoint_ls_cycles=int(args.endpoint_ls_cycles),
        endpoint_wght_mode=str(args.endpoint_wght_mode),
        post_bloch_sigma_policy=args.post_bloch_sigma_policy,
        post_bloch_sigma_holdout_fraction=args.post_bloch_sigma_holdout_fraction,
        post_bloch_sigma_min_scale=args.post_bloch_sigma_min_scale,
        post_bloch_sigma_max_scale=args.post_bloch_sigma_max_scale,
        rescale=args.rescale,
        cache_compression=args.cache_compression,
        compute_backend=args.compute_backend,
        row_plan_cache=args.row_plan_cache,
        jax_batch_size=args.jax_batch_size,
        jax_matrix_bucket_size=args.jax_matrix_bucket_size,
        bethe_strong_count=args.bethe_strong_count,
        bethe_variant=args.bethe_variant,
        bethe_promote_denom=args.bethe_promote_denom,
        bethe_denom_eps=args.bethe_denom_eps,
        topup_d_min=args.topup_d_min,
        max_rows=args.max_rows,
        workers=args.workers,
        chunksize=args.chunksize,
    )


def run_one_native_endpoint(
    args: argparse.Namespace,
    source_summary: dict,
    model: Path,
    run_dir: Path,
    *,
    label: str,
    lambda_grid: str,
    shelxl_iterations: int,
    alternate_hand: bool = False,
    q_control: str = "real",
) -> dict:
    """Run one fixed-q or q=1 native-row endpoint control."""

    ns = as_feedback_args(args, lambda_grid=lambda_grid, shelxl_iterations=shelxl_iterations)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = fb.native_command(ns, source_summary, model, run_dir, n_basis=args.n_basis, n_points=args.n_points)
    if alternate_hand:
        cmd.append("--alternate-hand")
    if q_control != "real":
        cmd.extend(["--q-control", q_control, "--q-control-seed", str(int(args.q_control_seed)), "--q-control-shells", str(int(args.q_control_shells))])
    (run_dir / "COMMAND.sh").write_text(shell_join(cmd) + "\n")
    status = fb.run_command(cmd, run_dir / "run.log", os.environ.copy())
    summary = read_summary(run_dir / "summary.json")
    return {
        "label": label,
        "kind": "native_row_regularize_endpoint",
        "run_dir": str(run_dir.resolve()),
        "input_model": str(model.resolve()),
        "lambda_grid": lambda_grid,
        "shelxl_iterations": int(shelxl_iterations),
        "alternate_hand": bool(alternate_hand),
        "q_control": q_control,
        "command": cmd,
        **status,
        "metrics": fb.selected_metrics(summary) if summary else {},
    }


def run_updated_q_loop(args: argparse.Namespace, source_summary: dict, out: Path) -> dict:
    """Run updated-q LS1 macrocycles and choose the earliest near-best cycle."""

    ns = as_feedback_args(args, lambda_grid="1.0", shelxl_iterations=args.shelxl_iterations)
    current_model = Path(args.start_res).expanduser().resolve()
    runs = []
    t0 = time.perf_counter()
    for cycle in range(1, int(args.cycles) + 1):
        rec = fb.run_cycle(
            ns,
            source_summary,
            current_model,
            out / f"cycle_{cycle:02d}",
            n_basis=args.n_basis,
            n_points=args.n_points,
            label=f"cycle_{cycle:02d}",
            env=os.environ.copy(),
            make_maps=False,
        )
        runs.append(rec)
        if rec.get("returncode") != 0 or not rec.get("output_model"):
            break
        current_model = Path(rec["output_model"]).resolve()

    # native_row_regularize reports row-level native metrics for the model used
    # to compute q.  In an updated-q loop, cycle N+1 therefore rechecks the
    # endpoint model produced by cycle N.  Keep both views visible and select
    # cycles by the endpoint-aligned native metric when it is available.
    for idx, rec in enumerate(runs):
        rec["input_native_metrics"] = dict(rec.get("metrics") or {})
        if idx + 1 < len(runs):
            rec["endpoint_native_metrics"] = dict(runs[idx + 1].get("metrics") or {})
            rec["endpoint_native_metrics_source"] = runs[idx + 1].get("label", "")
        else:
            rec["endpoint_native_metrics"] = {}
            rec["endpoint_native_metrics_source"] = ""

    finite = []
    for rec in runs:
        endpoint_metrics = rec.get("endpoint_native_metrics") or {}
        metric = endpoint_metrics.get("native_dyn_wR2_all")
        if metric is None and len(runs) == 1:
            metric = rec.get("metrics", {}).get("native_dyn_wR2_all")
        with contextlib.suppress(TypeError, ValueError):
            finite.append((float(metric), rec))
    best_metric = min((x[0] for x in finite), default=float("nan"))
    tolerance = float(args.best_cycle_tolerance)
    selected = None
    for metric, rec in finite:
        if metric <= best_metric + tolerance:
            selected = rec
            break
    if selected is None and runs:
        selected = runs[-1]
    result = {
        "label": "updated_q_ls1",
        "kind": "updated_q_feedback_loop",
        "cycles_requested": int(args.cycles),
        "best_cycle_tolerance": tolerance,
        "selected_cycle": selected.get("label") if selected else "",
        "selected_model": selected.get("output_model") if selected else "",
        "wall_seconds": time.perf_counter() - t0,
        "runs": runs,
    }
    (out / "updated_q_loop_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def native_polish_command(args: argparse.Namespace, source_summary: dict, model: Path, run_dir: Path) -> list[str]:
    """Build a native finite-difference polish command with matched Bloch physics."""

    nui = fb.nuisance(source_summary)
    inputs = source_summary.get("inputs") or {}
    cfg = source_summary.get("config") or {}
    hkl = args.hkl or inputs.get("hkl") or cfg.get("hkl")
    baseline_hkl = args.baseline_hkl or inputs.get("baseline_hkl") or cfg.get("baseline_hkl") or ""
    hkl_transform = args.hkl_transform or inputs.get("hkl_transform") or cfg.get("hkl_transform") or "identity"
    if not hkl:
        raise SystemExit("No HKL was supplied and source summary does not record inputs.hkl/config.hkl")
    script = Path(__file__).with_name("native_structure_update.py")
    cmd = [
        args.python,
        str(script),
        "--hkl",
        str(hkl),
        "--res",
        str(model),
        f"--hkl-transform={hkl_transform}",
        "--out-dir",
        str(run_dir),
        "--n-basis",
        str(int(args.native_polish_n_basis or args.n_basis)),
        "--n-points",
        str(int(args.native_polish_n_points or args.n_points)),
        "--sg-window",
        fb.grid_value(fb.summary_value(source_summary, "sg_window", 0.004), "0.004"),
        "--sg-window-mode",
        str(fb.summary_value(source_summary, "sg_window_mode", "fixed")),
        "--min-d-spacing",
        fb.grid_value(args.min_d_spacing if args.min_d_spacing > 0 else fb.summary_value(source_summary, "min_d_spacing", 0.7), "0.7"),
        "--min-peak",
        fb.grid_value(args.min_peak if args.min_peak >= 0 else fb.summary_value(source_summary, "min_peak", 0.0), "0.0"),
        "--geometry-percentile",
        fb.grid_value(args.geometry_percentile if args.geometry_percentile > 0 else fb.summary_value(source_summary, "geometry_percentile", 100.0), "100.0"),
        "--target-mode",
        str(fb.summary_value(source_summary, "target_mode", "iobs")),
        "--objective-rows",
        str(int(args.native_polish_rows)),
        "--objective-group-width-deg",
        fb.grid_value(nui.get("group_width_deg"), "1.0"),
        "--block-schedule",
        str(args.native_polish_blocks),
        "--atom-selection",
        str(args.native_polish_atom_selection),
        "--max-nfev",
        str(int(args.native_polish_max_nfev)),
        "--compute-backend",
        str(args.compute_backend),
        "--jax-batch-size",
        str(int(args.jax_batch_size)),
        "--jax-matrix-bucket-size",
        str(int(args.jax_matrix_bucket_size)),
        "--bethe-strong-count",
        str(int(args.bethe_strong_count)),
        "--bethe-variant",
        str(args.bethe_variant),
        "--bethe-promote-denom",
        str(float(args.bethe_promote_denom)),
        "--bethe-denom-eps",
        str(float(args.bethe_denom_eps)),
        "--profile-quadrature",
        str(fb.summary_value(source_summary, "profile_quadrature", "uniform")),
        "--profile-accumulation",
        str(fb.summary_value(source_summary, "profile_accumulation", "sum")),
        "--thickness",
        fb.grid_value(nui.get("thickness_A"), "1200"),
        "--thickness-grad-frac",
        fb.grid_value(nui.get("thickness_grad_frac"), "0.4"),
        "--thickness-points",
        str(int(float(fb.summary_value(source_summary, "thickness_points", 5)))),
        "--topup-d-min",
        fb.grid_value(args.topup_d_min if args.topup_d_min > 0 else fb.summary_value(source_summary, "topup_d_min", 0.8), "0.8"),
        "--group-width-deg",
        fb.grid_value(nui.get("group_width_deg"), "1.0"),
        "--frame-offset",
        fb.grid_value(nui.get("frame_offset"), "0.0"),
        "--row-resolution-b",
        fb.grid_value(nui.get("row_resolution_b"), "0.0"),
        "--detector-scale-bins",
        str(int(float(nui.get("detector_scale_bins", 0)))),
        "--sg-jitter-mode",
        str(nui.get("sg_jitter_mode", "none")),
        "--sg-jitter-sigma0",
        fb.grid_value(nui.get("sg_jitter_sigma0"), "0.0"),
        "--sg-jitter-sigma-d-inv",
        fb.grid_value(nui.get("sg_jitter_sigma_d_inv"), "0.0"),
        "--sg-jitter-sigma-frame",
        fb.grid_value(nui.get("sg_jitter_sigma_frame"), "0.0"),
        "--sg-jitter-points",
        str(int(float(nui.get("sg_jitter_points", 1)))),
        "--sg-jitter-student-df",
        fb.grid_value(nui.get("sg_jitter_student_df", 4.0), "4.0"),
        "--sg-jitter-width-factor",
        fb.grid_value(nui.get("sg_jitter_width_factor", 3.0), "3.0"),
        "--absorption-mode",
        str(nui.get("absorption_mode", "none")),
        "--absorption-mu",
        fb.grid_value(nui.get("absorption_mu"), "0.0"),
        "--absorption-frac",
        fb.grid_value(nui.get("absorption_frac"), "0.0"),
        "--absorption-mu-g2",
        fb.grid_value(nui.get("absorption_mu_g2"), "0.0"),
        "--absorption-mu-theta2",
        fb.grid_value(nui.get("absorption_mu_theta2"), "0.0"),
        "--absorption-tds-b",
        fb.grid_value(nui.get("absorption_tds_b"), "0.0"),
        "--sigma-a",
        fb.grid_value(nui.get("sigma_a"), "0.0"),
        "--sigma-b-corr",
        fb.grid_value(nui.get("sigma_b_corr"), "0.0"),
        "--sigma-c-peak",
        fb.grid_value(nui.get("sigma_c_peak"), "0.0"),
        "--lambda-grid",
        "1.0",
        "--workers",
        str(int(args.workers)),
        "--chunksize",
        str(int(args.chunksize)),
        "--early-stop-window",
        str(int(args.native_polish_early_stop_window)),
        "--early-stop-min-evals",
        str(int(args.native_polish_early_stop_min_evals)),
        "--early-stop-min-improvement",
        str(float(args.native_polish_early_stop_min_improvement)),
        "--scale-fit-split",
        "all",
        "--acceptance-metric",
        "all_wR2",
        "--free-wR2-tolerance",
        str(float(args.native_polish_free_wR2_tolerance)),
    ]
    if baseline_hkl:
        cmd.extend(["--baseline-hkl", str(baseline_hkl)])
    return cmd


def run_native_polish(args: argparse.Namespace, source_summary: dict, model: str, out: Path) -> dict:
    """Run the optional matched-physics native finite-difference polish."""

    run_dir = out / "native_polish"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = native_polish_command(args, source_summary, Path(model).expanduser().resolve(), run_dir)
    (run_dir / "COMMAND.sh").write_text(shell_join(cmd) + "\n")
    status = fb.run_command(cmd, run_dir / "run.log", os.environ.copy())
    summary = read_summary(run_dir / "summary.json")
    return {
        "label": "native_polish",
        "kind": "native_structure_update",
        "run_dir": str(run_dir.resolve()),
        "input_model": model,
        "command": cmd,
        **status,
        "accepted": (summary.get("acceptance") or {}).get("accepted") if summary else None,
        "acceptance": summary.get("acceptance") if summary else {},
        "best_endpoint": summary.get("best_endpoint") if summary else {},
    }


def report_markdown(result: dict) -> str:
    """Render a human-readable validation report."""

    def pct(value) -> str:
        try:
            return f"{100.0 * float(value):.2f}"
        except (TypeError, ValueError):
            return "nan"

    def num(value) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "nan"

    lines = [
        "# GROND LS1 Validation Suite",
        "",
        "This run checks whether the Bloch/LS1 loop behaves like a useful lagged dynamical surrogate rather than a cosmetic corrected-HKL loop.",
        "",
        "## Updated-q LS1",
        "",
    ]
    updated = result.get("updated_q_ls1") or {}
    lines.append(f"Selected cycle: `{updated.get('selected_cycle', '')}` using earliest-cycle native wR2 tolerance `{updated.get('best_cycle_tolerance', '')}`.")
    lines.append("")
    lines.append("For updated-q cycles, native row metrics are aligned to the LS1 endpoint by using the next cycle's q recomputation where available. The final cycle has no post-step native recheck unless another cycle is run.")
    lines.append("")
    lines.extend(["| cycle | native check | row R(I)all % | row wR(I)all/work/free % | Born row wR(I)all % | endpoint R1(F)all % | endpoint wR2(F^2)all % | GooF |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    runs = updated.get("runs", [])
    for idx, rec in enumerate(runs):
        inferred_endpoint = runs[idx + 1].get("metrics") if idx + 1 < len(runs) else {}
        native_m = rec.get("endpoint_native_metrics") or inferred_endpoint or rec.get("metrics") or {}
        endpoint_m = rec.get("metrics") or {}
        source = rec.get("endpoint_native_metrics_source") or (runs[idx + 1].get("label") if inferred_endpoint else "input")
        lines.append(
            f"| {rec.get('label')} | {source} | {pct(native_m.get('native_dyn_R1_all'))} | "
            f"{pct(native_m.get('native_dyn_wR2_all'))}/{pct(native_m.get('native_dyn_work_wR2_all'))}/{pct(native_m.get('native_dyn_free_wR2_all'))} | "
            f"{pct(native_m.get('native_born_wR2_all'))} | "
            f"{pct(endpoint_m.get('endpoint_R1_all'))} | "
            f"{pct(endpoint_m.get('endpoint_wR2_all'))} | "
            f"{num(endpoint_m.get('endpoint_GooF'))} |"
        )
    lines.extend(["", "## Controls", "", "| control | status | row wR(I)all/work/free % | endpoint wR2(F^2)all % | q rows | logq corr | notes |", "|---|---:|---:|---:|---:|---:|---|"])
    for ctrl in result.get("controls", []):
        m = ctrl.get("metrics") or {}
        status = ctrl.get("returncode", "")
        notes = ctrl.get("note", "")
        lines.append(
            f"| {ctrl.get('label')} | {status} | "
            f"{pct(m.get('native_dyn_wR2_all'))}/{pct(m.get('native_dyn_work_wR2_all'))}/{pct(m.get('native_dyn_free_wR2_all'))} | "
            f"{pct(m.get('endpoint_wR2_all'))} | "
            f"{m.get('q_control_shuffled_rows', '')} | {num(m.get('q_control_logq_corr'))} | {notes} |"
        )
    if result.get("native_polish"):
        pol = result["native_polish"]
        lines.extend([
            "",
            "## Native Polish",
            "",
            f"Native polish return code: `{pol.get('returncode')}`. Accepted: `{pol.get('accepted')}`.",
            f"Directory: `{pol.get('run_dir')}`.",
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Updated-q should beat q=1/Born and frozen-q controls on row-level intensity metrics. If frozen-q is essentially identical to updated-q, the fixed-point update is not doing much for this case. If native polish improves row wR(I)all by more than about 0.5-1.0 percentage point or causes chemically meaningful shifts, LS1 should be treated as a preconditioner rather than the endpoint.",
        "",
        "The shuffled-q controls permute row-space log(Idyn/Iborn) before normal scale fitting and endpoint export. Global shuffle destroys all row identity; shell shuffle preserves the resolution distribution; shell+scan shuffle preserves both resolution and local scan/block distribution while still breaking row identity.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-summary", required=True, help="Existing native_row_regularize summary whose nuisance values should be pinned.")
    ap.add_argument("--start-res", required=True, help="Initial structure for updated-q cycle 1.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--hkl", default="")
    ap.add_argument("--baseline-hkl", default="")
    ap.add_argument("--hkl-transform", default="")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--n-basis", type=int, default=30)
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--shelxl-iterations", type=int, default=1)
    ap.add_argument("--endpoint-ls-cycles", type=int, default=1)
    ap.add_argument("--endpoint-wght-mode", choices=["freeze-zero", "inherit", "promote"], default="freeze-zero")
    ap.add_argument("--shelxl", default=os.environ.get("SHELXL", "shelxl"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy_bethe")
    ap.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off")
    ap.add_argument("--jax-batch-size", type=int, default=0)
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    ap.add_argument("--bethe-strong-count", type=int, default=48)
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005)
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    ap.add_argument("--min-d-spacing", type=float, default=0.0)
    ap.add_argument("--topup-d-min", type=float, default=0.0)
    ap.add_argument("--min-peak", type=float, default=-1.0)
    ap.add_argument("--geometry-percentile", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="none")
    ap.add_argument("--best-cycle-tolerance", type=float, default=0.001, help="Earliest cycle within this row wR(I)all absolute fraction of the best cycle is selected.")
    ap.add_argument("--q-control-seed", type=int, default=12345)
    ap.add_argument("--q-control-shells", type=int, default=10)
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--wrong-hand-control", dest="wrong_hand_control", action=argparse.BooleanOptionalAction, default=True, help="Run the alternate-hand null control.")
    ap.add_argument("--native-polish", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--native-polish-rows", type=int, default=200)
    ap.add_argument("--native-polish-n-basis", type=int, default=0)
    ap.add_argument("--native-polish-n-points", type=int, default=0)
    ap.add_argument("--native-polish-blocks", default="adp_ueq,xyz")
    ap.add_argument("--native-polish-atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--native-polish-max-nfev", type=int, default=12)
    ap.add_argument("--native-polish-early-stop-window", type=int, default=32)
    ap.add_argument("--native-polish-early-stop-min-evals", type=int, default=0)
    ap.add_argument("--native-polish-early-stop-min-improvement", type=float, default=1e-4)
    ap.add_argument("--native-polish-free-wR2-tolerance", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    source_summary = read_summary(Path(args.source_summary).expanduser())
    if not source_summary:
        raise SystemExit(f"could not read source summary: {args.source_summary}")
    env = os.environ.copy()
    env["SHELXL"] = str(args.shelxl)
    os.environ["SHELXL"] = str(args.shelxl)

    t0 = time.perf_counter()
    result = {
        "source_summary": str(Path(args.source_summary).expanduser().resolve()),
        "start_res": str(Path(args.start_res).expanduser().resolve()),
        "out_dir": str(out.resolve()),
        "config": vars(args),
        "controls": [],
    }
    updated = run_updated_q_loop(args, source_summary, out / "updated_q_ls1")
    result["updated_q_ls1"] = updated
    selected_model = updated.get("selected_model") or str(Path(args.start_res).expanduser().resolve())

    if not args.skip_controls:
        result["controls"].append(
            run_one_native_endpoint(
                args,
                source_summary,
                Path(args.start_res).expanduser().resolve(),
                out / "controls" / "frozen_q_fixed_hkl",
                label="frozen_q_fixed_hkl",
                lambda_grid="1.0",
                shelxl_iterations=max(1, int(args.cycles)),
            )
        )
        result["controls"].append(
            run_one_native_endpoint(
                args,
                source_summary,
                Path(args.start_res).expanduser().resolve(),
                out / "controls" / "born_lambda0_q1",
                label="born_lambda0_q1",
                lambda_grid="0.0",
                shelxl_iterations=max(1, int(args.cycles)),
            )
        )
        if args.wrong_hand_control:
            result["controls"].append(
                run_one_native_endpoint(
                    args,
                    source_summary,
                    Path(args.start_res).expanduser().resolve(),
                    out / "controls" / "alternate_hand_q",
                    label="alternate_hand_q",
                    lambda_grid="1.0",
                    shelxl_iterations=1,
                    alternate_hand=True,
                )
            )
        for mode, label in [
            ("shuffle_global", "shuffled_q_global"),
            ("shuffle_shell", "shuffled_q_shell"),
            ("shuffle_shell_scan", "shuffled_q_shell_scan"),
        ]:
            result["controls"].append(
                run_one_native_endpoint(
                    args,
                    source_summary,
                    Path(args.start_res).expanduser().resolve(),
                    out / "controls" / label,
                    label=label,
                    lambda_grid="1.0",
                    shelxl_iterations=1,
                    q_control=mode,
                )
            )

    if args.native_polish and selected_model:
        result["native_polish"] = run_native_polish(args, source_summary, selected_model, out)

    result["wall_seconds"] = time.perf_counter() - t0
    result["summary_path"] = str((out / "ls1_validation_summary.json").resolve())
    (out / "ls1_validation_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    (out / "LS1_VALIDATION_REPORT.md").write_text(report_markdown(result))
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
