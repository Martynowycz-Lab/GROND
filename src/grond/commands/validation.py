"""Compile installed code and launch fixed-point refinement validation."""

from __future__ import annotations

import argparse
import compileall
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from grond.cli_support.tools import choose_python, external_tool_statuses
from grond.paths import LEGACY_ROOT, PACKAGE_ROOT, engine_script


def cmd_validate(args: argparse.Namespace) -> int:
    """CLI entrypoint that compile-checks the vendored Python engine."""

    python = choose_python(args.python)
    scripts = [
        engine_script("bloch_engine.py"),
        engine_script("native_row_regularize.py"),
        engine_script("continuous_thickness_polish.py"),
        engine_script("row_mr_report.py"),
        engine_script("resolution_limit_audit.py"),
        engine_script("residual_compass.py"),
        engine_script("premerge_rfactor_audit.py"),
        engine_script("native_two_part_observation_correction.py"),
        engine_script("native_observation_polish_endpoint.py"),
        engine_script("native_additive_bias_audit.py"),
        engine_script("native_dose_damage_audit.py"),
        engine_script("native_sigma_replicate_audit.py"),
        engine_script("run_raw_cbf_bloch_trial.py"),
        engine_script("build_dynrows_xds.py"),
        engine_script("dynrow_contract.py"),
        engine_script("support_basis_scan.py"),
        engine_script("ls1_validation_suite.py"),
        engine_script("native_structure_update.py"),
        engine_script("native_multi_structure_update.py"),
        engine_script("combine_native_nuisance_runs.py"),
        engine_script("run_shelxl_baseline.py"),
    ]
    cmd = [python, "-m", "py_compile", *map(str, scripts)]
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="grond-validate-pycache-") as pycache_prefix:
        env["PYTHONPYCACHEPREFIX"] = pycache_prefix
        subprocess.run(cmd, check=True, env=env)
    print(
        json.dumps(
            {
                "python": python,
                "compiled": len(scripts),
                "tool_config": getattr(args, "tool_config_path", ""),
                "external_tools": external_tool_statuses(args),
            },
            indent=2,
        )
    )
    return 0


def cmd_validate_ls1(args: argparse.Namespace) -> int:
    """Run the LS1 fixed-point validation suite for a pinned nuisance contract."""

    python = choose_python(args.python)
    cmd = [
        python,
        str(engine_script("ls1_validation_suite.py")),
        "--source-summary",
        args.source_summary,
        "--start-res",
        args.start_res,
        "--out-dir",
        args.out_dir,
        "--python",
        python,
        "--cycles",
        str(args.cycles),
        "--n-basis",
        str(args.n_basis),
        "--n-points",
        str(args.n_points),
        "--shelxl-iterations",
        str(args.shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.endpoint_ls_cycles),
        "--endpoint-wght-mode",
        args.endpoint_wght_mode,
        "--shelxl",
        args.shelxl,
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        args.row_plan_cache,
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--min-d-spacing",
        f"{args.min_d_spacing:g}",
        "--topup-d-min",
        f"{args.topup_d_min:g}",
        "--min-peak",
        f"{args.min_peak:g}",
        "--geometry-percentile",
        f"{args.geometry_percentile:g}",
        "--max-rows",
        str(args.max_rows),
        "--post-bloch-sigma-policy",
        args.post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--rescale",
        args.rescale,
        "--cache-compression",
        args.prediction_cache_compression,
        "--best-cycle-tolerance",
        f"{args.best_cycle_tolerance:g}",
        "--q-control-seed",
        str(args.q_control_seed),
        "--q-control-shells",
        str(args.q_control_shells),
        "--native-polish-rows",
        str(args.native_polish_rows),
        "--native-polish-n-basis",
        str(args.native_polish_n_basis),
        "--native-polish-n-points",
        str(args.native_polish_n_points),
        "--native-polish-blocks",
        args.native_polish_blocks,
        "--native-polish-atom-selection",
        args.native_polish_atom_selection,
        "--native-polish-max-nfev",
        str(args.native_polish_max_nfev),
        "--native-polish-early-stop-window",
        str(args.native_polish_early_stop_window),
        "--native-polish-early-stop-min-evals",
        str(args.native_polish_early_stop_min_evals),
        "--native-polish-early-stop-min-improvement",
        f"{args.native_polish_early_stop_min_improvement:g}",
        "--native-polish-free-wR2-tolerance",
        f"{args.native_polish_free_wR2_tolerance:g}",
    ]
    if args.hkl:
        cmd += ["--hkl", args.hkl]
    if args.baseline_hkl:
        cmd += ["--baseline-hkl", args.baseline_hkl]
    if args.hkl_transform:
        cmd += [f"--hkl-transform={args.hkl_transform}"]
    if args.skip_controls:
        cmd.append("--skip-controls")
    if not args.wrong_hand_control:
        cmd.append("--no-wrong-hand-control")
    if args.native_polish:
        cmd.append("--native-polish")
    if args.dry_run:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        plan = {
            "command": cmd,
            "tool_config": getattr(args, "tool_config_path", ""),
            "external_tools": external_tool_statuses(args),
        }
        (Path(args.out_dir) / "validate_ls1_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0
    subprocess.run(cmd, check=True)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """CLI entrypoint that compile-checks package and vendored Python modules."""

    root = PACKAGE_ROOT.parent
    python = choose_python(args.python)
    cmd = [python, "-m", "py_compile", *map(str, sorted(LEGACY_ROOT.rglob("*.py")))]
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="grond-compile-pycache-") as pycache_prefix:
        old_prefix = sys.pycache_prefix
        sys.pycache_prefix = pycache_prefix
        try:
            ok = compileall.compile_dir(str(root), quiet=1, force=bool(args.force), workers=1)
        finally:
            sys.pycache_prefix = old_prefix
        if not ok:
            return 1
        env["PYTHONPYCACHEPREFIX"] = pycache_prefix
        subprocess.run(cmd, check=True, env=env)
    print(
        json.dumps(
            {
                "package_root": str(root),
                "engine_root": str(LEGACY_ROOT),
                "python": python,
                "bytecode_cache": "temporary",
                "status": "compile_check_passed",
                "tool_config": getattr(args, "tool_config_path", ""),
                "external_tools": external_tool_statuses(args),
            },
            indent=2,
        )
    )
    return 0
