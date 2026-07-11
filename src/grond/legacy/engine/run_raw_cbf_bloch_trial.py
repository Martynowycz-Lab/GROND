#!/usr/bin/env python3
"""Brute-force raw-CBF Bloch trial around XDS spot predictions.

This is an observation-model driver, not a new Bloch kernel.  It uses XDS
`INTEGRATE.HKL` as the signed-row/predicted-spot contract, integrates local raw
CBF shoeboxes with several auditable aperture/background/profile settings, then runs
the existing frozen Bloch nuisance search on the raw-recorded and XDS-complete
targets.  The intent is a small Beanland/Cleverley-style graft: keep the close
structure and XDS geometry, but replace scalar XDS IOBS with explicitly
background-subtracted raw profile observations before dynamical correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "engine"


@dataclass(frozen=True)
class RawGrid:
    name: str
    frame_radius: int
    xy_half_width: int
    foreground_radius: float
    bg_inner: float
    bg_outer: float


def parse_raw_grid(text: str) -> list[RawGrid]:
    """Parse `name:frame,xy,fg,bgin,bgout;...`."""
    out: list[RawGrid] = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, vals = chunk.split(":", 1)
        else:
            name, vals = f"grid{len(out) + 1}", chunk
        parts = [p.strip() for p in vals.replace("/", ",").split(",") if p.strip()]
        if len(parts) != 5:
            raise ValueError(
                f"bad raw grid item {chunk!r}; expected name:frame,xy,fg,bgin,bgout"
            )
        frame, xy, fg, bgin, bgout = parts
        out.append(
            RawGrid(
                name.strip(), int(frame), int(xy), float(fg), float(bgin), float(bgout)
            )
        )
    if not out:
        raise ValueError("raw grid is empty")
    return out


def parse_float_grid(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x.strip()]


def parse_text_grid(text: str) -> list[str]:
    out = [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]
    if not out:
        raise ValueError("empty text grid")
    return out


def run_command(cmd: list[str], log_path: Path, cwd: Path = ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True
        )
        f.write(
            f"\n[returncode] {proc.returncode}\n[seconds] {time.perf_counter() - t0:.3f}\n"
        )
    return int(proc.returncode)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def summarize_native(report_dir: Path) -> dict:
    summary = load_json(report_dir / "summary.json")
    native = load_json(report_dir / "best_native_fit.json")
    out: dict = {
        "summary_json": (
            str((report_dir / "summary.json").resolve())
            if (report_dir / "summary.json").exists()
            else ""
        ),
        "report_md": (
            str((report_dir / "REPORT.md").resolve())
            if (report_dir / "REPORT.md").exists()
            else ""
        ),
    }
    best = summary.get("best_endpoint") or {}
    if best:
        sh = best.get("shelxl") or {}
        out.update(
            {
                "endpoint_R1gt": sh.get("R1_gt"),
                "endpoint_R1all": sh.get("R1_all"),
                "endpoint_wR2all": sh.get("wR2_all"),
                "endpoint_GooF": sh.get("GooF"),
                "endpoint_Nall": sh.get("N_all"),
                "endpoint_unique": sh.get("unique"),
                "endpoint_q_median": best.get("q_median"),
                "endpoint_q_p05": best.get("q_p05"),
                "endpoint_q_p95": best.get("q_p95"),
            }
        )
    metrics = native.get("metrics") or native.get("best", {}).get("metrics") or {}
    splits = metrics.get("splits") or native.get("splits") or metrics
    all_split = splits.get("all") or {}
    free_split = splits.get("free") or {}
    dyn_all = all_split.get("dyn") or {}
    born_all = all_split.get("born") or {}
    dyn_free = free_split.get("dyn") or {}
    born_free = free_split.get("born") or {}
    out.update(
        {
            "native_R1_dyn": dyn_all.get("R1_all"),
            "native_R1_born": born_all.get("R1_all"),
            "native_wR2_dyn": dyn_all.get("wR2_all"),
            "native_wR2_born": born_all.get("wR2_all"),
            "native_CC_dyn": dyn_all.get("CC"),
            "native_free_wR2_dyn": dyn_free.get("wR2_all"),
            "native_free_wR2_born": born_free.get("wR2_all"),
            "native_rows": dyn_all.get("n"),
        }
    )
    params = (
        native.get("params")
        or native.get("nuisance")
        or native.get("best", {}).get("params")
        or {}
    )
    out.update({f"param_{k}": v for k, v in params.items()})
    return out


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt_pct(x) -> str:
    if x is None or x == "":
        return "n/a"
    try:
        return f"{100.0 * float(x):.2f}"
    except Exception:
        return "n/a"


def write_report(path: Path, rows: list[dict], config: dict) -> None:
    lines = [
        "# Raw-CBF Bloch Trial",
        "",
        "This is a brute-force observation-model audit around the frozen sample Bloch kernel. XDS supplies exact signed rows and predicted spot positions; raw CBF frames supply local background-subtracted intensities; the existing native nuisance fitter supplies group scales, thickness, gradient, and corrected-HKL endpoint statistics.",
        "",
        "## Inputs",
        "",
        f"`INTEGRATE.HKL`: `{config['integrate_hkl']}`",
        "",
        f"`model`: `{config['model']}`",
        "",
        f"`CBF`: `{config['images'] or config['cbf_dir']}`",
        "",
        "## Results",
        "",
        "| raw grid | matrix | target | rows | native R1 dyn | native wR2 dyn | native wR2 Born | free wR2 dyn | endpoint R1all | endpoint wR2all | q p05/med/p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        q = "n/a"
        if r.get("endpoint_q_median") is not None:
            q = f"{float(r['endpoint_q_p05']):.3g}/{float(r['endpoint_q_median']):.3g}/{float(r['endpoint_q_p95']):.3g}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("raw_grid", "")),
                    str(r.get("matrix_convention", "")),
                    str(r.get("target", "")),
                    str(r.get("native_rows", "")),
                    fmt_pct(r.get("native_R1_dyn")),
                    fmt_pct(r.get("native_wR2_dyn")),
                    fmt_pct(r.get("native_wR2_born")),
                    fmt_pct(r.get("native_free_wR2_dyn")),
                    fmt_pct(r.get("endpoint_R1all")),
                    fmt_pct(r.get("endpoint_wR2all")),
                    q,
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A useful raw-CBF target must beat its matched Born model on all/work/free rows and should approach the current XDS IOBS native endpoint. If the raw targets lose badly to XDS IOBS, the issue is the raw observation model or background/profile ownership, not the Bloch Hamiltonian.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrate-hkl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline-hkl", default="")
    ap.add_argument(
        "--hkl-transform",
        default="identity",
        help="Signed axis permutation mapping raw XDS HKLs into the model basis, e.g. '-h,k,l'.",
    )
    ap.add_argument("--cbf-dir", default="")
    ap.add_argument("--images", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--raw-grid", default="small:2,4,2.5,4,6;medium:3,5,3,5,8;wide:4,6,3.5,6,10"
    )
    ap.add_argument("--raw-integration-modes", default="sum,gaussian_profile")
    ap.add_argument("--background-models", default="global_median")
    ap.add_argument(
        "--profile-sigma-xy-grid",
        default="0",
        help="Comma grid for gaussian_profile. 0 derives from foreground radius.",
    )
    ap.add_argument(
        "--profile-sigma-z-grid",
        default="0",
        help="Comma grid for gaussian_profile. 0 derives from frame radius.",
    )
    ap.add_argument("--bloch-profile-n-basis", type=int, default=20)
    ap.add_argument("--bloch-profile-thickness", type=float, default=1200.0)
    ap.add_argument("--bloch-profile-thickness-grad-frac", type=float, default=1.0)
    ap.add_argument("--bloch-profile-absorption-mode", default="midgley_like")
    ap.add_argument("--bloch-profile-absorption-mu", type=float, default=0.00005)
    ap.add_argument("--bloch-profile-absorption-frac", type=float, default=0.02)
    ap.add_argument("--bloch-profile-absorption-mu-g2", type=float, default=0.0)
    ap.add_argument("--bloch-profile-absorption-mu-theta2", type=float, default=0.0)
    ap.add_argument("--bloch-profile-absorption-tds-b", type=float, default=0.0)
    ap.add_argument("--bloch-profile-z-smooth-frames", type=float, default=0.0)
    ap.add_argument(
        "--matrix-conventions",
        default="j_minus_i",
        help="Comma grid of Bloch off-diagonal conventions: j_minus_i and/or i_minus_j.",
    )
    ap.add_argument("--bloch-profile-z-gaussian-blend", type=float, default=0.0)
    ap.add_argument("--bloch-profile-z-gaussian-sigma", type=float, default=0.0)
    ap.add_argument("--targets", default="raw_recorded,raw_complete")
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--thickness-grid", default="800,1200,1800,2400")
    ap.add_argument("--thickness-grad-grid", default="0.0,0.4,0.8")
    ap.add_argument("--frame-offset-grid", default="0.0")
    ap.add_argument("--group-width-grid", default="1.5,3.0")
    ap.add_argument("--detector-scale-bins", type=int, default=0)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument(
        "--lambda-grid",
        default="1.0",
        help="Diagnostic q-exponent grid. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--prediction-cache-dir", default="")
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="none")
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy")
    ap.add_argument("--jax-batch-size", type=int, default=0)
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    ap.add_argument("--bethe-strong-count", type=int, default=48)
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005)
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    ap.add_argument(
        "--include-nonpositive",
        action="store_true",
        help="Keep zero/negative raw target intensities in the native fit instead of dropping them.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug row limit for raw integration. 0 means all rows.",
    )
    ap.add_argument(
        "--top-isigma",
        type=int,
        default=0,
        help="Debug row filter for raw integration. 0 means all rows.",
    )
    ap.add_argument(
        "--even-sample",
        type=int,
        default=0,
        help="Evenly sample this many raw-integration candidates across the rotation range before --limit. 0 disables.",
    )
    ap.add_argument(
        "--calibrate-geometry", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--make-maps", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    logs = out / "logs"
    raw_root = out / "raw_dynpacks"
    fit_root = out / "native_fits"
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    grids = parse_raw_grid(args.raw_grid)
    integration_modes = [
        m.strip()
        for m in args.raw_integration_modes.replace(",", " ").split()
        if m.strip()
    ]
    background_models = [
        m.strip() for m in args.background_models.replace(",", " ").split() if m.strip()
    ]
    profile_sigma_xy_grid = parse_float_grid(args.profile_sigma_xy_grid)
    profile_sigma_z_grid = parse_float_grid(args.profile_sigma_z_grid)
    matrix_conventions = parse_text_grid(args.matrix_conventions)
    for convention in matrix_conventions:
        if convention not in {"j_minus_i", "i_minus_j"}:
            raise ValueError("matrix conventions must be j_minus_i and/or i_minus_j")
    targets = [t.strip() for t in args.targets.replace(",", " ").split() if t.strip()]
    target_to_hkl = {
        "raw_recorded": "raw_recorded_dynrows.HKL",
        "raw_complete": "raw_complete_dynrows.HKL",
        "xds_iobs_calibrated": "xds_iobs_calibrated_dynrows.HKL",
        "xds_recorded_proxy_calibrated": "xds_recorded_proxy_calibrated_dynrows.HKL",
    }
    rows: list[dict] = []
    python = sys.executable

    for grid in grids:
        for raw_mode in integration_modes:
            if raw_mode not in {"sum", "gaussian_profile", "bloch_profile_fit"}:
                raise ValueError(
                    "raw integration modes must be sum, gaussian_profile, or bloch_profile_fit"
                )
            for background_model in background_models:
                if background_model not in {"global_median", "per_frame_median"}:
                    raise ValueError(
                        "background models must be global_median or per_frame_median"
                    )
                xy_values = (
                    profile_sigma_xy_grid if raw_mode in {"gaussian_profile", "bloch_profile_fit"} else [0.0]
                )
                z_values = (
                    profile_sigma_z_grid if raw_mode == "gaussian_profile" else [0.0]
                )
                for profile_sigma_xy in xy_values:
                    for profile_sigma_z in z_values:
                        for matrix_convention in matrix_conventions:
                            suffix = raw_mode
                            if background_model != "global_median":
                                suffix += f"_{background_model}"
                            if len(matrix_conventions) > 1 or matrix_convention != "j_minus_i":
                                suffix += f"_{matrix_convention}"
                            if raw_mode == "gaussian_profile" and (
                                float(profile_sigma_xy) > 0 or float(profile_sigma_z) > 0
                            ):
                                suffix += f"_sx{str(profile_sigma_xy).replace('.', 'p')}_sz{str(profile_sigma_z).replace('.', 'p')}"
                            grid_run_name = f"{grid.name}_{suffix}"
                            dynpack = raw_root / grid_run_name
                            build_cmd = [
                                python,
                                str(PORT / "build_dynrows_xds.py"),
                                "--integrate-hkl",
                                args.integrate_hkl,
                                "--model",
                                args.model,
                                "--out-dir",
                                str(dynpack),
                                "--frame-radius",
                                str(grid.frame_radius),
                                "--xy-half-width",
                                str(grid.xy_half_width),
                                "--foreground-radius",
                                str(grid.foreground_radius),
                                "--bg-inner",
                                str(grid.bg_inner),
                                "--bg-outer",
                                str(grid.bg_outer),
                                "--raw-integration-mode",
                                raw_mode,
                                "--background-model",
                                background_model,
                                "--profile-sigma-xy",
                                str(profile_sigma_xy),
                                "--profile-sigma-z",
                                str(profile_sigma_z),
                                f"--bloch-profile-hkl-transform={args.hkl_transform}",
                                "--bloch-profile-matrix-convention",
                                matrix_convention,
                                "--bloch-profile-n-basis",
                                str(args.bloch_profile_n_basis),
                                "--bloch-profile-thickness",
                                str(args.bloch_profile_thickness),
                                "--bloch-profile-thickness-grad-frac",
                                str(args.bloch_profile_thickness_grad_frac),
                                "--bloch-profile-absorption-mode",
                                args.bloch_profile_absorption_mode,
                                "--bloch-profile-absorption-mu",
                                str(args.bloch_profile_absorption_mu),
                                "--bloch-profile-absorption-frac",
                                str(args.bloch_profile_absorption_frac),
                                "--bloch-profile-absorption-mu-g2",
                                str(args.bloch_profile_absorption_mu_g2),
                                "--bloch-profile-absorption-mu-theta2",
                                str(args.bloch_profile_absorption_mu_theta2),
                                "--bloch-profile-absorption-tds-b",
                                str(args.bloch_profile_absorption_tds_b),
                                "--bloch-profile-z-smooth-frames",
                                str(args.bloch_profile_z_smooth_frames),
                                "--bloch-profile-z-gaussian-blend",
                                str(args.bloch_profile_z_gaussian_blend),
                                "--bloch-profile-z-gaussian-sigma",
                                str(args.bloch_profile_z_gaussian_sigma),
                                "--preserve-negative",
                                "true",
                                "--write-xds-like",
                            ]
                            if args.images:
                                build_cmd += ["--images", args.images]
                            else:
                                build_cmd += ["--cbf-dir", args.cbf_dir]
                            if args.calibrate_geometry:
                                build_cmd += ["--calibrate-geometry"]
                            else:
                                build_cmd += ["--no-calibrate-geometry"]
                            if args.limit:
                                build_cmd += ["--limit", str(args.limit)]
                            if args.top_isigma:
                                build_cmd += ["--top-isigma", str(args.top_isigma)]
                            if args.even_sample:
                                build_cmd += ["--even-sample", str(args.even_sample)]
                            if args.dry_run:
                                print("DRY", " ".join(build_cmd))
                                rc = 0
                            else:
                                rc = run_command(
                                    build_cmd, logs / f"build_{grid_run_name}.log"
                                )
                            if rc != 0:
                                rows.append(
                                    {
                                        "raw_grid": grid_run_name,
                                        "matrix_convention": matrix_convention,
                                        "stage": "build",
                                        "returncode": rc,
                                    }
                                )
                                continue
                            manifest = load_json(dynpack / "manifest.json")

                            for target in targets:
                                if target not in target_to_hkl:
                                    raise ValueError(
                                        f"unknown target {target}; choose from {sorted(target_to_hkl)}"
                                    )
                                hkl = dynpack / target_to_hkl[target]
                                fit_dir = fit_root / f"{grid_run_name}_{target}"
                                cache_dir = (
                                    Path(args.prediction_cache_dir).expanduser()
                                    if args.prediction_cache_dir
                                    else out / "cache" / "native_predictions"
                                )
                                fit_cmd = [
                                    python,
                                    str(PORT / "native_row_regularize.py"),
                                    "--hkl",
                                    str(hkl),
                                    "--res",
                                    args.model,
                                    f"--hkl-transform={args.hkl_transform}",
                                    "--matrix-convention",
                                    matrix_convention,
                                    "--out-dir",
                                    str(fit_dir),
                                    "--n-basis",
                                    str(args.n_basis),
                                    "--n-points",
                                    str(args.n_points),
                                    "--sg-window",
                                    str(args.sg_window),
                                    "--sg-window-mode",
                                    "fixed",
                                    "--min-d-spacing",
                                    str(args.min_d_spacing),
                                    "--min-peak",
                                    str(args.min_peak),
                                    "--target-mode",
                                    "iobs",
                                    "--thickness-grid",
                                    args.thickness_grid,
                                    "--thickness-grad-grid",
                                    args.thickness_grad_grid,
                                    f"--frame-offset-grid={args.frame_offset_grid}",
                                    "--group-width-grid",
                                    args.group_width_grid,
                                    "--detector-scale-bins",
                                    str(args.detector_scale_bins),
                                    "--sigma-a-grid",
                                    "0.0",
                                    "--sigma-b-corr-grid",
                                    "0.0",
                                    "--sigma-c-peak-grid",
                                    "0.0",
                                    "--thickness-points",
                                    str(args.thickness_points),
                                    "--free-fraction",
                                    str(args.free_fraction),
                                    "--split-seed",
                                    str(args.split_seed),
                                    "--scale-fit-split",
                                    "work",
                                    "--selection-metric",
                                    "free_wR2",
                                    "--lambda-grid",
                                    args.lambda_grid,
                                    "--rescale",
                                    "median-iobs",
                                    "--hkl-float-format",
                                    "shelx_fixed",
                                    "--workers",
                                    str(args.workers),
                                    "--chunksize",
                                    str(args.chunksize),
                                    "--prediction-cache-dir",
                                    str(cache_dir),
                                    "--cache-compression",
                                    args.cache_compression,
                                    "--compute-backend",
                                    args.compute_backend,
                                    "--jax-batch-size",
                                    str(args.jax_batch_size),
                                    "--jax-matrix-bucket-size",
                                    str(args.jax_matrix_bucket_size),
                                    "--bethe-strong-count",
                                    str(args.bethe_strong_count),
                                    "--bethe-variant",
                                    args.bethe_variant,
                                    "--bethe-promote-denom",
                                    str(args.bethe_promote_denom),
                                    "--bethe-denom-eps",
                                    str(args.bethe_denom_eps),
                                ]
                                fit_cmd.append("--include-nonpositive" if args.include_nonpositive else "--positive-only")
                                if args.baseline_hkl:
                                    fit_cmd += ["--baseline-hkl", args.baseline_hkl]
                                if args.make_maps:
                                    fit_cmd += ["--make-maps"]
                                if args.dry_run:
                                    print("DRY", " ".join(fit_cmd))
                                    rc = 0
                                else:
                                    rc = run_command(
                                        fit_cmd, logs / f"fit_{grid_run_name}_{target}.log"
                                    )
                                row = {
                                    "raw_grid": grid_run_name,
                                    "raw_base_grid": grid.name,
                                    "raw_integration_mode": raw_mode,
                                    "background_model": background_model,
                                    "matrix_convention": matrix_convention,
                                    "hkl_transform": args.hkl_transform,
                                    "profile_sigma_xy": profile_sigma_xy,
                                    "profile_sigma_z": profile_sigma_z,
                                    "target": target,
                                    "build_returncode": 0,
                                    "fit_returncode": rc,
                                    "dynpack": str(dynpack.resolve()),
                                    "fit_dir": str(fit_dir.resolve()),
                                    "raw_written_rows": (
                                        manifest.get("row_counts") or {}
                                    ).get("written_observations"),
                                    "raw_positive_fraction": (
                                        manifest.get("raw_intensity_summary") or {}
                                    ).get("positive_fraction"),
                                    "raw_median": (
                                        manifest.get("raw_intensity_summary") or {}
                                    ).get("median"),
                                    "raw_quality_accepted": (
                                        manifest.get("quality_mask") or {}
                                    ).get("row_count_accepted"),
                                }
                                if rc == 0:
                                    row.update(summarize_native(fit_dir))
                                rows.append(row)

    write_summary_csv(out / "raw_cbf_bloch_summary.csv", rows)
    write_report(out / "REPORT.md", rows, config)
    (out / "summary.json").write_text(
        json.dumps({"config": config, "results": rows}, indent=2, allow_nan=True) + "\n"
    )
    print(
        json.dumps(
            {
                "out_dir": str(out),
                "report": str((out / "REPORT.md").resolve()),
                "results": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
