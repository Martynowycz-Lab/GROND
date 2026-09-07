"""Build the public command grammar in its established display order."""

from __future__ import annotations

import argparse

from grond.runtime import auto_worker_count


def register_ls1(sub: argparse._SubParsersAction) -> None:
    """Register ls1 commands."""

    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.commands.validation import cmd_validate_ls1

    p_validate_ls1 = sub.add_parser(
        "validate-ls1", help="Run LS1 fixed-point controls and optional matched native finite-difference polish."
    )
    p_validate_ls1.add_argument(
        "--source-summary",
        required=True,
        help="Existing native_row_regularize summary whose selected nuisance values should be pinned.",
    )
    p_validate_ls1.add_argument("--start-res", required=True, help="Initial RES model for updated-q cycle 1.")
    p_validate_ls1.add_argument("--out-dir", required=True)
    p_validate_ls1.add_argument("--python", default="")
    add_tool_config_args(p_validate_ls1)
    p_validate_ls1.add_argument("--dry-run", action="store_true")
    p_validate_ls1.add_argument("--hkl", default="", help="Override HKL path if the source summary does not record it.")
    p_validate_ls1.add_argument(
        "--baseline-hkl", default="", help="Optional conventional HKL for baseline/reference endpoint comparisons."
    )
    p_validate_ls1.add_argument(
        "--hkl-transform", default="", help="Override signed HKL transform. Defaults to the source summary."
    )
    p_validate_ls1.add_argument("--cycles", type=int, default=5, help="Updated-q LS1 macrocycles to run.")
    p_validate_ls1.add_argument(
        "--n-basis",
        type=int,
        default=30,
        help="Bloch basis for validation cycles. Use the final adaptive basis for publication adjudication.",
    )
    p_validate_ls1.add_argument("--n-points", type=int, default=20)
    p_validate_ls1.add_argument("--shelxl", default=default_shelxl_executable())
    p_validate_ls1.add_argument(
        "--shelxl-iterations", type=int, default=1, help="SHELXL iterations per updated-q macrocycle."
    )
    p_validate_ls1.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Diagnostic internal L.S./CGLS cycle count while each Bloch q field is frozen. Production keeps this at 1.",
    )
    p_validate_ls1.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote"],
        default="freeze-zero",
        help="SHELXL endpoint weighting policy used by the validation cycles.",
    )
    p_validate_ls1.add_argument("--workers", type=int, default=auto_worker_count(0, 0.85))
    p_validate_ls1.add_argument("--chunksize", type=int, default=0)
    p_validate_ls1.add_argument(
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
        default="scipy_bethe",
    )
    p_validate_ls1.add_argument(
        "--row-plan-cache",
        choices=["off", "exact", "readwrite", "rebuild"],
        default="off",
        help="Row-geometry cache. exact is keyed to the current structural model.",
    )
    p_validate_ls1.add_argument("--jax-batch-size", type=int, default=0)
    p_validate_ls1.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    p_validate_ls1.add_argument("--bethe-strong-count", type=int, default=48)
    p_validate_ls1.add_argument(
        "--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target"
    )
    p_validate_ls1.add_argument("--bethe-promote-denom", type=float, default=0.005)
    p_validate_ls1.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    p_validate_ls1.add_argument("--min-d-spacing", type=float, default=0.0)
    p_validate_ls1.add_argument("--topup-d-min", type=float, default=0.0)
    p_validate_ls1.add_argument("--min-peak", type=float, default=-1.0)
    p_validate_ls1.add_argument("--geometry-percentile", type=float, default=0.0)
    p_validate_ls1.add_argument("--max-rows", type=int, default=0)
    p_validate_ls1.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    p_validate_ls1.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_validate_ls1.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    p_validate_ls1.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_validate_ls1.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_validate_ls1.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_validate_ls1.add_argument(
        "--best-cycle-tolerance",
        type=float,
        default=0.001,
        help="Pick the earliest cycle within this row wR(I)all fraction of the best cycle.",
    )
    p_validate_ls1.add_argument("--q-control-seed", type=int, default=12345)
    p_validate_ls1.add_argument("--q-control-shells", type=int, default=10)
    p_validate_ls1.add_argument("--skip-controls", action="store_true")
    p_validate_ls1.add_argument(
        "--wrong-hand-control",
        dest="wrong_hand_control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the alternate-hand null control.",
    )
    p_validate_ls1.add_argument("--native-polish", action=argparse.BooleanOptionalAction, default=False)
    p_validate_ls1.add_argument("--native-polish-rows", type=int, default=200)
    p_validate_ls1.add_argument("--native-polish-n-basis", type=int, default=0)
    p_validate_ls1.add_argument("--native-polish-n-points", type=int, default=0)
    p_validate_ls1.add_argument("--native-polish-blocks", default="adp_ueq,xyz")
    p_validate_ls1.add_argument("--native-polish-atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    p_validate_ls1.add_argument("--native-polish-max-nfev", type=int, default=12)
    p_validate_ls1.add_argument("--native-polish-early-stop-window", type=int, default=32)
    p_validate_ls1.add_argument("--native-polish-early-stop-min-evals", type=int, default=0)
    p_validate_ls1.add_argument("--native-polish-early-stop-min-improvement", type=float, default=1e-4)
    p_validate_ls1.add_argument("--native-polish-free-wR2-tolerance", type=float, default=0.02)
    p_validate_ls1.set_defaults(func=cmd_validate_ls1)


def register_runtime(sub: argparse._SubParsersAction) -> None:
    """Register runtime commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.doctor import cmd_doctor
    from grond.commands.validation import cmd_compile, cmd_validate

    p_val = sub.add_parser("validate", help="Compile the vendored engine scripts with the selected Python.")
    p_val.add_argument("--python", default="")
    add_tool_config_args(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_doc = sub.add_parser(
        "doctor", help="Report Python packages, accelerator visibility, and the active compute core."
    )
    p_doc.add_argument("--python", default="")
    add_tool_config_args(p_doc)
    p_doc.add_argument(
        "--probe-bloch", action="store_true", help="Also test complex eig/eigh support needed by the Bloch wave kernel."
    )
    p_doc.add_argument(
        "--workflow",
        choices=["core", "small-molecule", "native-maps", "endpoint-maps"],
        default="core",
        help="Readiness scope: core/native-maps need no external program; small-molecule adds SHELXL; endpoint-maps also adds shelx2map.",
    )
    p_doc.add_argument(
        "--all-tools",
        action="store_true",
        help="Also report optional endpoint tools and experimental accelerator backends.",
    )
    p_doc.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when required packages, installed features, or requested Bloch operations are not ready.",
    )
    p_doc.set_defaults(func=cmd_doctor)

    p_compile = sub.add_parser(
        "compile", help="Compile-check GROND and vendored engine modules without writing bytecode into the source tree."
    )
    p_compile.add_argument("--python", default="")
    add_tool_config_args(p_compile)
    p_compile.add_argument("--force", action="store_true")
    p_compile.set_defaults(func=cmd_compile)
