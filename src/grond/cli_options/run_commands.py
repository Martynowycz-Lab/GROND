"""Build the public command grammar in its established display order."""

from __future__ import annotations

import argparse


def register_runs(sub: argparse._SubParsersAction) -> None:
    """Register runs commands."""

    from grond.cli_options.common import add_common_run_args
    from grond.cli_support.tools import add_tool_config_args
    from grond.cli_support.values import FULL_Q_LAMBDA
    from grond.commands.raw_polish import cmd_raw_polish
    from grond.commands.run import cmd_auto, cmd_run

    p_run = sub.add_parser("run", help="Run fixed-structure Bloch, dynamic regularization, or full auto mode.")
    add_common_run_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_auto = sub.add_parser("auto", help="Simple one-command full automatic workflow.")
    add_common_run_args(p_auto, expose_lambda_controls=False)
    p_auto.set_defaults(func=cmd_auto)

    p_raw = sub.add_parser(
        "raw-polish", help="Experimental raw-CBF Beanland/Cleverley-style forward observation polish."
    )
    p_raw.add_argument("--prep", default="")
    p_raw.add_argument("--dataset-name", default="")
    p_raw.add_argument("--hkl", default="")
    p_raw.add_argument("--xds-dir", default="")
    p_raw.add_argument("--structure", default="")
    p_raw.add_argument("--baseline-hkl", default="")
    p_raw.add_argument("--hkl-transform", default="")
    p_raw.add_argument("--cbf-dir", default="")
    p_raw.add_argument("--images", default="")
    p_raw.add_argument("--out-dir", required=True)
    p_raw.add_argument("--fast", action="store_true")
    p_raw.add_argument("--python", default="")
    add_tool_config_args(p_raw)
    p_raw.add_argument("--dry-run", action="store_true")
    p_raw.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p_raw.add_argument("--workers", type=int, default=0)
    p_raw.add_argument("--max-core-fraction", type=float, default=0.85)
    p_raw.add_argument("--chunksize", type=int, default=0)
    p_raw.add_argument(
        "--compute-backend",
        choices=[
            "scipy",
            "scipy_bethe",
            "scipy_expm",
            "scipy_expm_multiply",
            "jax",
            "jax_auto",
            "jax_cpu",
            "jax_mps",
            "jax_cuda",
            "jax_padded",
            "jax_auto_padded",
            "jax_cpu_padded",
            "jax_mps_padded",
            "jax_cuda_padded",
            "jax_cpu_real_expm",
            "jax_mps_real_expm",
            "jax_mps_real_expm_padded",
        ],
        default="scipy",
    )
    p_raw.add_argument(
        "--row-plan-cache",
        choices=["off", "exact", "readwrite", "rebuild"],
        default="off",
        help="Row-geometry cache. exact is keyed to the current structural model.",
    )
    p_raw.add_argument("--jax-batch-size", type=int, default=0)
    p_raw.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    p_raw.add_argument("--bethe-strong-count", type=int, default=48)
    p_raw.add_argument(
        "--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target"
    )
    p_raw.add_argument("--bethe-promote-denom", type=float, default=0.005)
    p_raw.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    p_raw.add_argument("--prediction-cache-dir", default="")
    p_raw.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_raw.add_argument("--raw-grid", default="small:2,4,2.5,4,6")
    p_raw.add_argument("--raw-integration-modes", default="sum,bloch_profile_fit")
    p_raw.add_argument("--background-models", default="global_median")
    p_raw.add_argument("--profile-sigma-xy-grid", default="0")
    p_raw.add_argument("--profile-sigma-z-grid", default="0")
    p_raw.add_argument("--matrix-conventions", default="j_minus_i,i_minus_j")
    p_raw.add_argument("--targets", default="raw_complete,xds_iobs_calibrated")
    p_raw.add_argument("--bloch-profile-n-basis", type=int, default=20)
    p_raw.add_argument("--bloch-profile-thickness", type=float, default=1200.0)
    p_raw.add_argument("--bloch-profile-thickness-grad-frac", type=float, default=1.0)
    p_raw.add_argument("--bloch-profile-absorption-mode", default="midgley_like")
    p_raw.add_argument("--bloch-profile-absorption-mu", type=float, default=0.00005)
    p_raw.add_argument("--bloch-profile-absorption-frac", type=float, default=0.02)
    p_raw.add_argument("--bloch-profile-z-smooth-frames", type=float, default=0.0)
    p_raw.add_argument("--bloch-profile-z-gaussian-blend", type=float, default=0.0)
    p_raw.add_argument("--bloch-profile-z-gaussian-sigma", type=float, default=0.0)
    p_raw.add_argument("--n-basis", type=int, default=20)
    p_raw.add_argument("--n-points", type=int, default=10)
    p_raw.add_argument("--sg-window", type=float, default=0.004)
    p_raw.add_argument("--min-d-spacing", type=float, default=0.70)
    p_raw.add_argument("--min-peak", type=float, default=90.0)
    p_raw.add_argument("--thickness-grid", default="1000,1200,1400")
    p_raw.add_argument("--thickness-grad-grid", default="0.4,1.0")
    p_raw.add_argument(
        "--frame-offset-grid",
        default="auto",
        help=(
            "Profile-center offsets in XDS frames, or 'auto'. Auto reconciles the global XDS header "
            "orientation with row ZCAL positions before work-row nuisance selection."
        ),
    )
    p_raw.add_argument("--group-width-grid", default="auto")
    p_raw.add_argument("--detector-scale-bins", type=int, default=0)
    p_raw.add_argument("--thickness-points", type=int, default=5)
    p_raw.add_argument(
        "--lambda-grid",
        default=FULL_Q_LAMBDA,
        help="Diagnostic q-exponent grid. Production auto uses the full Bloch/Born correction with lambda_q=1.0.",
    )
    p_raw.add_argument("--free-fraction", type=float, default=0.10)
    p_raw.add_argument("--split-seed", type=int, default=42)
    p_raw.add_argument("--limit", type=int, default=0)
    p_raw.add_argument("--top-isigma", type=int, default=0)
    p_raw.add_argument("--even-sample", type=int, default=0)
    p_raw.add_argument("--include-nonpositive", action="store_true")
    p_raw.add_argument("--calibrate-geometry", action=argparse.BooleanOptionalAction, default=True)
    p_raw.add_argument("--make-maps", dest="make_maps", action="store_true", default=True)
    p_raw.add_argument("--no-maps", dest="make_maps", action="store_false")
    p_raw.set_defaults(func=cmd_raw_polish)
