#!/usr/bin/env python3
"""Iterate fixed-nuisance Bloch corrections with SHELXL structure update.

This diagnostic asks whether expensive native finite-difference dynamical
structure gradients can be replaced by a cheaper loop:

1. calculate fixed-structure Bloch row predictions,
2. export a merged corrected HKLF4 endpoint,
3. run no-EXTI SHELXL model update with the model's existing restraints/ADPs,
4. feed the updated ``.res`` into the next Bloch calculation.

The nuisance parameter values are read from an existing GROND
``native_row_regularize.py`` summary and pinned during the loop.  This makes
the benchmark a test of the structure-update hook, not another grid search.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    """Read a JSON file, allowing GROND summaries that contain NaN tokens."""

    return json.loads(path.read_text())


def finite(value) -> bool:
    """Return whether a value is a finite float."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def grid_value(value, default: str = "0.0") -> str:
    """Format a pinned scalar as a one-value GROND grid string."""

    if value is None:
        return default
    if isinstance(value, str):
        return value
    if finite(value):
        return f"{float(value):.12g}"
    return default


def summary_value(summary: dict, key: str, default=None):
    """Return a config value first, then a best-nuisance value."""

    cfg = summary.get("config") or {}
    nui = ((summary.get("best_native") or {}).get("nuisance") or {})
    if key in cfg:
        return cfg[key]
    return nui.get(key, default)


def nuisance(summary: dict) -> dict:
    """Return the selected nuisance dictionary from a native summary."""

    out = dict(((summary.get("best_native") or {}).get("nuisance") or {}))
    if not out:
        raise SystemExit("source summary does not contain best_native.nuisance")
    return out


def endpoint_res(run_dir: Path) -> Path:
    """Return the no-EXTI B=0 lambda=1 endpoint RES file from one cycle."""

    direct = run_dir / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    if direct.exists():
        return direct
    matches = sorted(run_dir.glob("endpoint_lam_*_B_0p0_noexti/dyn_nuisance_corrected.res"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no no-EXTI B=0 endpoint RES found under {run_dir}")


def selected_metrics(summary: dict) -> dict:
    """Extract compact native and endpoint metrics for one cycle."""

    best = summary.get("best_native") or {}
    all_dyn = (((best.get("metrics") or {}).get("all") or {}).get("dyn") or {})
    all_born = (((best.get("metrics") or {}).get("all") or {}).get("born") or {})
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    return {
        "native_dyn_R1_all": all_dyn.get("R1_all"),
        "native_dyn_R1_gt": all_dyn.get("R1_gt"),
        "native_dyn_wR2_all": all_dyn.get("wR2_all"),
        "native_dyn_CC": all_dyn.get("CC"),
        "native_born_R1_all": all_born.get("R1_all"),
        "native_born_wR2_all": all_born.get("wR2_all"),
        "endpoint_R1_gt": shelxl.get("R1_gt"),
        "endpoint_R1_all": shelxl.get("R1_all"),
        "endpoint_wR2_all": shelxl.get("wR2_all"),
        "endpoint_GooF": shelxl.get("GooF"),
        "endpoint_N_gt": shelxl.get("N_gt"),
        "endpoint_N_all": shelxl.get("N_all"),
        "endpoint_status": shelxl.get("status"),
    }


def append_arg(cmd: list[str], name: str, value) -> None:
    """Append a command-line argument pair."""

    cmd.extend([name, str(value)])


def native_command(
    args: argparse.Namespace,
    source_summary: dict,
    model: Path,
    run_dir: Path,
    *,
    n_basis: int,
    n_points: int,
    make_maps: bool = False,
) -> list[str]:
    """Build one fixed-nuisance native Bloch/SHELXL endpoint command."""

    nui = nuisance(source_summary)
    inputs = source_summary.get("inputs") or {}
    cfg = source_summary.get("config") or {}
    hkl = args.hkl or inputs.get("hkl") or cfg.get("hkl")
    baseline_hkl = args.baseline_hkl or inputs.get("baseline_hkl") or cfg.get("baseline_hkl") or ""
    hkl_transform = args.hkl_transform or inputs.get("hkl_transform") or cfg.get("hkl_transform") or "identity"
    if not hkl:
        raise SystemExit("No HKL was supplied and source summary does not record inputs.hkl/config.hkl")

    script = Path(__file__).with_name("native_row_regularize.py")
    cmd = [
        args.python,
        str(script),
        "--hkl",
        str(hkl),
        "--res",
        str(model),
        "--hkl-transform",
        str(hkl_transform),
        "--out-dir",
        str(run_dir),
        "--n-basis",
        str(int(n_basis)),
        "--n-points",
        str(int(n_points)),
        "--sg-window",
        grid_value(summary_value(source_summary, "sg_window", 0.004), "0.004"),
        "--sg-window-mode",
        str(summary_value(source_summary, "sg_window_mode", "fixed")),
        "--min-d-spacing",
        grid_value(args.min_d_spacing if args.min_d_spacing > 0 else summary_value(source_summary, "min_d_spacing", 0.7), "0.7"),
        "--min-peak",
        grid_value(args.min_peak if args.min_peak >= 0 else summary_value(source_summary, "min_peak", 0.0), "0.0"),
        "--geometry-percentile",
        grid_value(args.geometry_percentile if args.geometry_percentile > 0 else summary_value(source_summary, "geometry_percentile", 100.0), "100.0"),
        "--target-mode",
        str(summary_value(source_summary, "target_mode", "iobs")),
        "--thickness-grid",
        grid_value(nui.get("thickness_A")),
        "--thickness-model-grid",
        str(nui.get("thickness_model", "single")),
        "--thickness-grad-grid",
        grid_value(nui.get("thickness_grad_frac")),
        "--thickness-sigma-frac-grid",
        grid_value(nui.get("thickness_sigma_frac")),
        "--frame-offset-grid",
        grid_value(nui.get("frame_offset")),
        "--group-width-grid",
        grid_value(nui.get("group_width_deg")),
        "--detector-scale-bins",
        str(int(float(nui.get("detector_scale_bins", 0)))),
        "--sigma-a-grid",
        grid_value(nui.get("sigma_a")),
        "--sigma-b-corr-grid",
        grid_value(nui.get("sigma_b_corr")),
        "--sigma-c-peak-grid",
        grid_value(nui.get("sigma_c_peak")),
        "--sg-jitter-mode-grid",
        str(nui.get("sg_jitter_mode", "none")),
        "--sg-jitter-sigma0-grid",
        grid_value(nui.get("sg_jitter_sigma0")),
        "--sg-jitter-sigma-d-inv-grid",
        grid_value(nui.get("sg_jitter_sigma_d_inv")),
        "--sg-jitter-sigma-frame-grid",
        grid_value(nui.get("sg_jitter_sigma_frame")),
        "--sg-jitter-points",
        str(int(float(nui.get("sg_jitter_points", 5)))),
        "--sg-jitter-student-df",
        grid_value(nui.get("sg_jitter_student_df", 4.0), "4.0"),
        "--sg-jitter-width-factor",
        grid_value(nui.get("sg_jitter_width_factor", 3.0), "3.0"),
        "--profile-quadrature",
        str(summary_value(source_summary, "profile_quadrature", "uniform")),
        "--profile-accumulation",
        str(summary_value(source_summary, "profile_accumulation", "sum")),
        "--absorption-mode-grid",
        str(nui.get("absorption_mode", "none")),
        "--absorption-mu-grid",
        grid_value(nui.get("absorption_mu")),
        "--absorption-frac-grid",
        grid_value(nui.get("absorption_frac")),
        "--absorption-mu-g2-grid",
        grid_value(nui.get("absorption_mu_g2")),
        "--absorption-mu-theta2-grid",
        grid_value(nui.get("absorption_mu_theta2")),
        "--absorption-tds-b-grid",
        grid_value(nui.get("absorption_tds_b")),
        "--dose-b-mode-grid",
        str(nui.get("dose_b_mode", "none")),
        "--dose-b-per-frame-grid",
        grid_value(nui.get("dose_b_per_frame")),
        "--row-resolution-b-grid",
        grid_value(nui.get("row_resolution_b")),
        "--wght-a-grid",
        grid_value(nui.get("native_wght_a"), ""),
        "--wght-b-grid",
        grid_value(nui.get("native_wght_b"), ""),
        "--thickness-points",
        str(int(float(summary_value(source_summary, "thickness_points", 5)))),
        "--topup-d-min",
        grid_value(args.topup_d_min if args.topup_d_min > 0 else summary_value(source_summary, "topup_d_min", 0.8), "0.8"),
        "--max-support-sg",
        grid_value(summary_value(source_summary, "max_support_sg", 0.005), "0.005"),
        "--free-fraction",
        grid_value(summary_value(source_summary, "free_fraction", 0.10), "0.10"),
        "--split-seed",
        str(int(float(summary_value(source_summary, "split_seed", 42)))),
        "--scale-fit-split",
        str(summary_value(source_summary, "scale_fit_split", "all")),
        "--selection-metric",
        str(summary_value(source_summary, "selection_metric", "all_wR2")),
        "--lambda-grid",
        str(args.lambda_grid),
        "--scale-smoothness",
        grid_value(nui.get("scale_smoothness", 0.0)),
        "--endpoint-b-grid",
        "0.0",
        "--endpoint-exti-grid",
        "off",
        "--shelxl-iterations",
        str(int(args.shelxl_iterations)),
        "--rescale",
        str(args.rescale),
        "--hkl-float-format",
        "shelx_fixed",
        "--workers",
        str(int(args.workers)),
        "--chunksize",
        str(int(args.chunksize)),
        "--prediction-cache-dir",
        str(run_dir.parent / "cache" / run_dir.name),
        "--cache-compression",
        str(args.cache_compression),
        "--compute-backend",
        str(args.compute_backend),
        "--jax-batch-size",
        str(int(args.jax_batch_size)),
        "--jax-matrix-bucket-size",
        str(int(args.jax_matrix_bucket_size)),
        "--matrix-convention",
        str(nui.get("matrix_convention", summary_value(source_summary, "matrix_convention", "j_minus_i"))),
        "--positive-only",
    ]
    if baseline_hkl:
        cmd.extend(["--baseline-hkl", str(baseline_hkl)])
    if args.max_rows > 0:
        cmd.extend(["--max-rows", str(int(args.max_rows))])
    if make_maps:
        cmd.append("--make-maps")
    return cmd


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> dict:
    """Run a command, tee stdout to a log, and return timing/status."""

    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout)
    return {"returncode": proc.returncode, "elapsed_seconds": elapsed, "log": str(log_path)}


def run_cycle(
    args: argparse.Namespace,
    source_summary: dict,
    model: Path,
    run_dir: Path,
    *,
    n_basis: int,
    n_points: int,
    label: str,
    env: dict[str, str],
    make_maps: bool = False,
) -> dict:
    """Run one Bloch/SHELXL feedback cycle."""

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = native_command(args, source_summary, model, run_dir, n_basis=n_basis, n_points=n_points, make_maps=make_maps)
    (run_dir / "COMMAND.sh").write_text(" ".join(shlex.quote(str(x)) for x in cmd) + "\n")
    status = run_command(cmd, run_dir / "run.log", env)
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    updated_res = endpoint_res(run_dir) if status["returncode"] == 0 else Path("")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "input_model": str(model),
        "output_model": str(updated_res) if updated_res else "",
        "n_basis": int(n_basis),
        "n_points": int(n_points),
        "command": cmd,
        **status,
        "metrics": selected_metrics(summary) if summary else {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-summary", required=True, help="Existing native_row_regularize summary whose selected nuisance values should be pinned.")
    ap.add_argument("--start-res", required=True, help="Initial structure for cycle 1.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--hkl", default="")
    ap.add_argument("--baseline-hkl", default="")
    ap.add_argument("--hkl-transform", default="")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--final-n-basis", type=int, default=0, help="Optional high-N final polish basis. 0 disables the final polish.")
    ap.add_argument("--final-n-points", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--shelxl", default=os.environ.get("SHELXL", "shelxl"))
    ap.add_argument("--shelxl-iterations", type=int, default=3)
    ap.add_argument("--lambda-grid", default="1.0")
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="none")
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded"], default="scipy")
    ap.add_argument("--jax-batch-size", type=int, default=0)
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    ap.add_argument("--min-d-spacing", type=float, default=0.0)
    ap.add_argument("--topup-d-min", type=float, default=0.0)
    ap.add_argument("--min-peak", type=float, default=-1.0)
    ap.add_argument("--geometry-percentile", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--make-maps-on-final", action="store_true", help="Generate maps only for the last requested cycle or final polish.")
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    source_summary_path = Path(args.source_summary).expanduser()
    source_summary = read_json(source_summary_path)
    current_model = Path(args.start_res).expanduser().resolve()
    env = os.environ.copy()
    env["SHELXL"] = str(args.shelxl)

    result = {
        "source_summary": str(source_summary_path.resolve()),
        "start_res": str(current_model),
        "shelxl": str(args.shelxl),
        "cycles_requested": int(args.cycles),
        "fast_n_basis": int(args.n_basis),
        "fast_n_points": int(args.n_points),
        "final_n_basis": int(args.final_n_basis),
        "final_n_points": int(args.final_n_points),
        "runs": [],
    }
    t0 = time.perf_counter()
    for i in range(1, int(args.cycles) + 1):
        maps_this_cycle = bool(args.make_maps_on_final and int(args.final_n_basis) <= 0 and i == int(args.cycles))
        rec = run_cycle(
            args,
            source_summary,
            current_model,
            out / f"cycle_{i:02d}_N{int(args.n_basis)}Q{int(args.n_points)}",
            n_basis=args.n_basis,
            n_points=args.n_points,
            label=f"cycle_{i:02d}",
            env=env,
            make_maps=maps_this_cycle,
        )
        result["runs"].append(rec)
        if rec["returncode"] != 0:
            break
        current_model = Path(rec["output_model"]).resolve()
        result["current_model"] = str(current_model)

    if (not result["runs"] or result["runs"][-1].get("returncode") == 0) and int(args.final_n_basis) > 0:
        rec = run_cycle(
            args,
            source_summary,
            current_model,
            out / f"final_N{int(args.final_n_basis)}Q{int(args.final_n_points)}",
            n_basis=args.final_n_basis,
            n_points=args.final_n_points,
            label="final_polish",
            env=env,
            make_maps=bool(args.make_maps_on_final),
        )
        result["runs"].append(rec)
        if rec["returncode"] == 0:
            result["current_model"] = rec["output_model"]

    result["wall_seconds"] = time.perf_counter() - t0
    result["summary_path"] = str((out / "shelxl_feedback_loop_summary.json").resolve())
    (out / "shelxl_feedback_loop_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
