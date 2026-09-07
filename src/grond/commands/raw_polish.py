"""Launch the optional raw-frame forward-polish experiment and record its command."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grond.cli_support.execution import configure_progress, run_logged
from grond.cli_support.tools import choose_python, external_tool_statuses
from grond.cli_support.values import FULL_Q_LAMBDA
from grond.io.config import load_prep, resolve_hkl
from grond.io.structure_contract import infer_cbf_dir_from_hkl, is_shelx_model, prepare_structure_for_legacy
from grond.paths import engine_script
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.nuisance_policy import first


def cmd_raw_polish(args: argparse.Namespace) -> int:
    """CLI entrypoint for the optional raw-CBF forward-polish experiment."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    configure_progress(logs, resume=bool(args.resume), total=1)

    if args.prep:
        prep = load_prep(Path(args.prep))
        if args.dataset_name:
            matches = [ds for ds in prep.datasets if ds.name == args.dataset_name]
            if not matches:
                raise SystemExit(f"Dataset {args.dataset_name!r} not found in {args.prep}")
            ds = matches[0]
        else:
            ds = prep.datasets[0]
        structure = args.structure or prep.structure
        hkl = ds.hkl
        hkl_transform = args.hkl_transform or ds.hkl_transform or prep.default_hkl_transform or "identity"
        baseline_hkl = args.baseline_hkl or ds.baseline_hkl or prep.baseline_hkl
    else:
        hkl, _xds_dir = resolve_hkl(args.xds_dir or args.hkl)
        if not args.structure:
            raise SystemExit("Provide --structure when not using --prep.")
        structure = args.structure
        hkl_transform = args.hkl_transform or "identity"
        baseline_hkl = args.baseline_hkl

    structure, structure_prep = prepare_structure_for_legacy(structure, out_dir)
    if (
        getattr(args, "make_maps", False)
        and is_shelx_model(structure)
        and not getattr(args, "allow_missing_tools", False)
        and not getattr(args, "dry_run", False)
    ):
        statuses = external_tool_statuses(args)
        if not statuses.get("shelx2map", {}).get("available"):
            raise SystemExit(
                "Missing required external tools:\n- shelx2map is required for raw-polish map generation; set tools.shelx2map in a GROND config, set SHELX2MAP, or add --no-maps.\n"
                "Create a template with `GROND init-config --out grond.config.json`, then rerun with `--config grond.config.json` or set GROND_CONFIG."
            )
    cbf_dir = args.cbf_dir or ("" if args.images else infer_cbf_dir_from_hkl(hkl))
    if not args.images and not cbf_dir:
        raise SystemExit("Provide --cbf-dir/--images; no sibling CBF directory could be inferred.")

    n_basis = 20 if args.fast else args.n_basis
    n_points = 5 if args.fast else args.n_points
    absorption_mode = "none" if args.fast else args.bloch_profile_absorption_mode
    absorption_mu = 0.0 if args.fast else args.bloch_profile_absorption_mu
    absorption_frac = 0.0 if args.fast else args.bloch_profile_absorption_frac
    matrix_conventions = args.matrix_conventions or ("j_minus_i,i_minus_j" if not args.fast else "j_minus_i")
    cmd = [
        python,
        str(engine_script("run_raw_cbf_bloch_trial.py")),
        "--integrate-hkl",
        hkl,
        "--model",
        structure,
        f"--hkl-transform={hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--raw-grid",
        args.raw_grid,
        "--raw-integration-modes",
        args.raw_integration_modes,
        "--background-models",
        args.background_models,
        "--profile-sigma-xy-grid",
        args.profile_sigma_xy_grid,
        "--profile-sigma-z-grid",
        args.profile_sigma_z_grid,
        "--matrix-conventions",
        matrix_conventions,
        "--bloch-profile-n-basis",
        str(args.bloch_profile_n_basis if not args.fast else 20),
        "--bloch-profile-thickness",
        str(args.bloch_profile_thickness),
        "--bloch-profile-thickness-grad-frac",
        str(args.bloch_profile_thickness_grad_frac),
        "--bloch-profile-absorption-mode",
        absorption_mode,
        "--bloch-profile-absorption-mu",
        str(absorption_mu),
        "--bloch-profile-absorption-frac",
        str(absorption_frac),
        "--bloch-profile-z-smooth-frames",
        str(args.bloch_profile_z_smooth_frames),
        "--bloch-profile-z-gaussian-blend",
        str(args.bloch_profile_z_gaussian_blend),
        "--bloch-profile-z-gaussian-sigma",
        str(args.bloch_profile_z_gaussian_sigma),
        "--targets",
        args.targets,
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(n_points),
        "--sg-window",
        str(args.sg_window),
        "--min-d-spacing",
        str(args.min_d_spacing),
        "--min-peak",
        str(args.min_peak),
        "--thickness-grid",
        args.thickness_grid,
        "--thickness-grad-grid",
        args.thickness_grad_grid,
        f"--frame-offset-grid={args.frame_offset_grid}",
        "--group-width-grid",
        args.group_width_grid,
        "--detector-scale-bins",
        str(0 if args.fast else args.detector_scale_bins),
        "--thickness-points",
        str(3 if args.fast else args.thickness_points),
        "--lambda-grid",
        first(args.lambda_grid, FULL_Q_LAMBDA) if args.fast else args.lambda_grid,
        "--free-fraction",
        str(args.free_fraction),
        "--split-seed",
        str(args.split_seed),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--prediction-cache-dir",
        args.prediction_cache_dir or str(out_dir / "cache" / "native_predictions"),
        "--cache-compression",
        args.prediction_cache_compression,
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
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
    ]
    if args.images:
        cmd += ["--images", args.images]
    else:
        cmd += ["--cbf-dir", cbf_dir]
    if baseline_hkl:
        cmd += ["--baseline-hkl", baseline_hkl]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.top_isigma:
        cmd += ["--top-isigma", str(args.top_isigma)]
    if args.even_sample:
        cmd += ["--even-sample", str(args.even_sample)]
    if args.include_nonpositive:
        cmd.append("--include-nonpositive")
    cmd.append("--calibrate-geometry" if args.calibrate_geometry else "--no-calibrate-geometry")
    if args.make_maps:
        cmd.append("--make-maps")
    config = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": python,
        "tool_config": {
            "path": getattr(args, "tool_config_path", ""),
            "configured_tools": getattr(args, "configured_tools", {}),
            "external_tools": external_tool_statuses(args),
            "allow_missing_tools": bool(getattr(args, "allow_missing_tools", False)),
        },
        "hkl": hkl,
        "hkl_transform": hkl_transform,
        "structure": structure,
        "structure_preparation": structure_prep,
        "baseline_hkl": baseline_hkl,
        "cbf_dir": cbf_dir,
        "images": args.images,
        "fast": bool(args.fast),
        "workers": args.workers,
        "chunksize": args.chunksize,
    }
    (out_dir / "raw_polish_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_logged(cmd, logs, "raw_polish", dry_run=args.dry_run)
    print(json.dumps({"out_dir": str(out_dir), "report": str((out_dir / "REPORT.md").resolve())}, indent=2))
    return 0
