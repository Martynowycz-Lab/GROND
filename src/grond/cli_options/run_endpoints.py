"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

from grond.cli_support.tools import default_shelxl_executable
from grond.cli_support.values import FULL_Q_LAMBDA


def add_endpoints(ap: argparse.ArgumentParser, *, expose_lambda_controls: bool) -> None:
    """Add endpoints options in the established order."""

    if expose_lambda_controls:
        ap.add_argument(
            "--lambda-grid",
            default=FULL_Q_LAMBDA,
            help="Diagnostic q-exponent grid for lower-level runs. Production auto uses the full Bloch/Born correction and does not tune this value.",
        )
    else:
        ap.set_defaults(lambda_grid=FULL_Q_LAMBDA)
    ap.add_argument(
        "--scale-smoothness",
        type=float,
        default=0.0,
        help="Experimental L2 frame-scale smoothness coefficient. Default 0 keeps per-group scales independent; GROND's default parameter reduction is the post-hoc scale-polish gate.",
    )
    ap.add_argument(
        "--endpoint-b-grid",
        default="0.0",
        help="Corrected-HKL endpoint resolution-scale diagnostic grid. Nonzero values are opt-in only; GROND keeps B=0 as the final promotable endpoint whenever it is present.",
    )
    ap.add_argument(
        "--endpoint-exti-grid",
        default="off",
        help="Endpoint EXTI diagnostic grid. Final promoted outputs use off; explicit 'on' runs are comparison-only.",
    )
    ap.add_argument(
        "--shelxl-iterations",
        type=int,
        default=5,
        help="SHELXL executable calls per corrected-HKL endpoint in lower-level full mode. GROND normalizes each endpoint input to one internal LS/CGLS cycle unless --endpoint-ls-cycles is changed for diagnostics.",
    )
    ap.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Diagnostic internal L.S./CGLS cycle count for generated corrected-HKL endpoint files. Production keeps this at 1.",
    )
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote"],
        default="freeze-zero",
        help="SHELXL weight policy for corrected-HKL endpoint files. freeze-zero is the production default after GROND sigma calibration; inherit keeps the source-model WGHT; promote uses SHELXL recommendations on later external calls.",
    )
    ap.add_argument(
        "--shelxl",
        default=default_shelxl_executable(),
        help="SHELXL executable used for corrected-HKL endpoints and SHELXL feedback loops.",
    )
    ap.add_argument(
        "--shelxl-feedback-cycles",
        type=int,
        default=5,
        help="Number of high-N/Q Bloch -> one-SHELXL-LS feedback cycles used by the automatic SHELXL structure-update strategy.",
    )
    ap.add_argument(
        "--shelxl-feedback-shelxl-iterations",
        type=int,
        default=1,
        help="SHELXL passes per feedback cycle. Default 1 keeps each feedback update to one least-squares model update.",
    )
    ap.add_argument(
        "--shelxl-feedback-endpoint-ls-cycles",
        type=int,
        default=1,
        help="Diagnostic internal L.S./CGLS cycle count for each SHELXL feedback macrocycle. Production keeps this at 1.",
    )
    ap.add_argument(
        "--dataset-sigma-policy",
        choices=["auto", "none", "uniform_z_rms"],
        default="auto",
        help="Per-dataset endpoint sigma policy for multi-dataset combination. Auto calibrates each crystal independently before merging.",
    )
    ap.add_argument(
        "--dataset-sigma-holdout-fraction",
        type=float,
        default=0.20,
        help="Deterministic HKL holdout fraction reported for per-dataset endpoint sigma calibration.",
    )
    ap.add_argument(
        "--dataset-sigma-min-scale",
        type=float,
        default=0.25,
        help="Lower guardrail for per-dataset endpoint sigma scaling.",
    )
    ap.add_argument(
        "--dataset-sigma-max-scale",
        type=float,
        default=4.0,
        help="Upper guardrail for per-dataset endpoint sigma scaling.",
    )
    ap.add_argument(
        "--dataset-sigma-holdout-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For multi-dataset auto, prevent a dataset from being assigned tighter sigmas than its held-out normalized residuals support.",
    )
    ap.add_argument(
        "--merge-sigma-policy",
        choices=["auto", "formal", "chisq_overdispersion"],
        default="auto",
        help="Sigma policy used when corrected rows are merged. Auto inflates merged sigmas when independent datasets disagree beyond formal errors.",
    )
    ap.add_argument(
        "--post-bloch-sigma-policy",
        choices=["auto", "none", "uniform_z_rms"],
        default="auto",
        help="Endpoint uncertainty policy. Auto uses uniform_z_rms only for final corrected endpoints and SHELXL feedback cycles, leaving early physics-selection endpoints uncalibrated.",
    )
    ap.add_argument(
        "--post-bloch-sigma-holdout-fraction",
        type=float,
        default=0.20,
        help="Deterministic HKL holdout fraction reported for uniform_z_rms endpoint sigma calibration.",
    )
    ap.add_argument(
        "--post-bloch-sigma-min-scale",
        type=float,
        default=0.25,
        help="Lower guardrail for uniform_z_rms endpoint sigma scaling.",
    )
    ap.add_argument(
        "--post-bloch-sigma-max-scale",
        type=float,
        default=4.0,
        help="Upper guardrail for uniform_z_rms endpoint sigma scaling.",
    )
    ap.add_argument(
        "--endpoint-feedback",
        choices=["auto", "on", "off"],
        default="auto",
        help="Optionally feed the polished B=0 no-EXTI endpoint model into one more Bloch endpoint cycle. Full auto uses auto; --fast skips it.",
    )
    ap.add_argument(
        "--publication-endpoint-chooser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the publication endpoint chooser that separates physical and diagnostic corrected-HKL endpoints.",
    )
    ap.add_argument(
        "--publication-endpoint-min-hkl-safe-factor",
        type=float,
        default=0.25,
        help="Minimum fixed-width HKL safety factor for a publication endpoint.",
    )
    ap.add_argument(
        "--publication-endpoint-max-modulation-p95",
        type=float,
        default=20.0,
        help="Warning level for 95th-percentile endpoint intensity modulation; never a promotion or validity gate.",
    )
    ap.add_argument(
        "--publication-allow-npd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicit diagnostic override allowing endpoints with NPD pathology to pass publication selection gates. Default requires NPD == 0. The general endpoint chooser reports split warnings without withholding a usable model; strict compact detector/scale promotion can still require zero splits.",
    )
    ap.add_argument(
        "--endpoint-calibration",
        choices=["auto", "on", "off"],
        default="auto",
        help="Replay the final frozen row table through corrected-HKL endpoint calibration after auto completes.",
    )
    ap.add_argument(
        "--endpoint-calibration-specs",
        default="baseline,adaptive:293:det0,adaptive:293:det2,adaptive:293:det4,adaptive:293:det8,adaptive:200:det0,adaptive:200:det2,adaptive:200:det4,adaptive:200:det8,adaptive:120:det0,adaptive:120:det2,adaptive:120:det4,adaptive:120:det8,adaptive:80:det0,adaptive:80:det2,adaptive:80:det4,adaptive:80:det8,window:1.0:det0,window:1.0:det4,window:1.0:det8,window:3.0:det0,window:3.0:det2,window:3.0:det4,window:3.0:det8,window:5.0:det0,window:5.0:det4,window:5.0:det8",
    )
    if expose_lambda_controls:
        ap.add_argument(
            "--endpoint-calibration-lambda-q",
            type=float,
            default=1.0,
            help="Diagnostic endpoint q exponent for replay tests. Production auto pins this to 1.0.",
        )
    else:
        ap.set_defaults(endpoint_calibration_lambda_q=1.0)
    ap.add_argument(
        "--endpoint-calibration-wght-modes",
        default="freeze-zero,promote",
        help="Endpoint-calibration weight branches. The default tests fixed WGHT 0 0 before the compatibility branch.",
    )
    ap.add_argument("--endpoint-calibration-prefer-wght-mode", default="freeze-zero")
    ap.add_argument(
        "--endpoint-calibration-goof-min",
        type=float,
        default=0.85,
        help="Lower preferred GooF warning threshold; never a model-handoff gate.",
    )
    ap.add_argument(
        "--endpoint-calibration-goof-max",
        type=float,
        default=1.15,
        help="Upper preferred GooF warning threshold; never a model-handoff gate.",
    )
    ap.add_argument("--endpoint-calibration-max-r1-cost", type=float, default=0.010)
    ap.add_argument("--endpoint-calibration-free-wr2-tolerance", type=float, default=0.040)
    ap.add_argument("--endpoint-calibration-free-rall-tolerance", type=float, default=0.010)
    ap.add_argument("--endpoint-calibration-work-rall-tolerance", type=float, default=0.010)
    ap.add_argument("--endpoint-calibration-free-wrall-tolerance", type=float, default=0.050)
    ap.add_argument(
        "--endpoint-calibration-selection-policy",
        choices=["bic", "gated-smallest"],
        default="gated-smallest",
        help="Final scale-contract selector. Scale topology is chosen from held-out/work JANA-style row factors; endpoint factors are reporting only.",
    )
    ap.add_argument(
        "--endpoint-calibration-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
        help="Information criterion used by endpoint calibration when the BIC policy is active.",
    )
    ap.add_argument("--endpoint-calibration-sigma-min-scale", type=float, default=0.05)
    ap.add_argument("--endpoint-calibration-sigma-max-scale", type=float, default=4.0)
    ap.add_argument(
        "--endpoint-calibration-shelxl-iterations",
        type=int,
        default=3,
        help="External one-cycle SHELXL calls for the final fixed-weight endpoint replay. This does not change LS1 feedback; it stabilizes the openable presentation endpoint.",
    )
    ap.add_argument(
        "--instability-u",
        type=float,
        default=0.01,
        help="JANA/PETS-style row-weight instability factor u used by wR/MwR reporting and row-scale diagnostics.",
    )
