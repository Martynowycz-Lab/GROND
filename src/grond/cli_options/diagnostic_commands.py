"""Register the post-run diagnostics: forensics, scale-polish, scale-replay, calibrate-endpoint, target-ablation."""

from __future__ import annotations

import argparse
import os


def register_diagnostics(sub: argparse._SubParsersAction) -> None:
    """Register diagnostics commands."""

    from grond.cli_options.forensics import add_forensic_args
    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.cli_support.values import XDS_TARGET_MODES
    from grond.commands.forensics import cmd_calibrate_endpoint, cmd_forensics
    from grond.commands.scale import cmd_scale_polish, cmd_scale_replay
    from grond.commands.target_ablation import cmd_target_ablation

    p_forensics = sub.add_parser("forensics", help="Run post-Bloch row-space forensic audits on native_dyn_calc.csv.")
    p_forensics.add_argument("--row-table", required=True)
    p_forensics.add_argument("--hkl", required=True)
    p_forensics.add_argument("--res", required=True)
    p_forensics.add_argument("--out-dir", required=True)
    p_forensics.add_argument("--python", default="")
    add_tool_config_args(p_forensics)
    p_forensics.add_argument("--dry-run", action="store_true")
    p_forensics.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_forensic_args(p_forensics)
    p_forensics.set_defaults(func=cmd_forensics)

    p_scale = sub.add_parser(
        "scale-polish", help="Compress frozen-Bloch row-space scale factors after a final GROND run."
    )
    p_scale.add_argument("--row-table", required=True)
    p_scale.add_argument("--summary", default="")
    p_scale.add_argument("--out-dir", required=True)
    p_scale.add_argument("--title", default="GROND scale-compression polish")
    p_scale.add_argument("--python", default="")
    add_tool_config_args(p_scale)
    p_scale.add_argument("--dry-run", action="store_true")
    p_scale.add_argument("--fit-split", default="")
    p_scale.add_argument("--frame-models", default="adaptive,fixed")
    p_scale.add_argument("--adaptive-frame-counts", default="auto")
    p_scale.add_argument("--group-strides", default="1,2,4,8,16,32,64,128,global")
    p_scale.add_argument(
        "--window-scan-deg",
        type=float,
        default=0.0,
        help="Also scan every integer group stride up to this physical window size.",
    )
    p_scale.add_argument(
        "--scan-offsets", action="store_true", help="Test all block-boundary offsets for each finite group stride."
    )
    p_scale.add_argument("--detector-bins", default="exact")
    p_scale.add_argument("--scale-objective", choices=["row_mr", "native", "sigma"], default="row_mr")
    p_scale.add_argument("--tolerance", type=float, default=0.01)
    p_scale.add_argument("--selection-metrics", default="free_Rall,work_Rall")
    p_scale.add_argument("--native-wght-a", type=float, default=None)
    p_scale.add_argument("--native-wght-b", type=float, default=None)
    p_scale.set_defaults(func=cmd_scale_polish)

    p_scale_replay = sub.add_parser(
        "scale-replay", help="Replay compact scale contracts through corrected-HKL refinement endpoints."
    )
    p_scale_replay.add_argument("--row-table", required=True)
    p_scale_replay.add_argument("--summary", default="")
    p_scale_replay.add_argument(
        "--res", default="", help="Source RES file. Defaults to the RES path recorded in the row summary."
    )
    p_scale_replay.add_argument("--out-dir", required=True)
    p_scale_replay.add_argument("--python", default="")
    add_tool_config_args(p_scale_replay)
    p_scale_replay.add_argument("--dry-run", action="store_true")
    p_scale_replay.add_argument(
        "--specs",
        default="baseline,adaptive:293:det8,adaptive:200:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8",
        help="Comma list of scale contracts: baseline, adaptive:N:det8, window:DEG:det8, or stride:N:det8.",
    )
    p_scale_replay.add_argument("--fit-split", default="all")
    p_scale_replay.add_argument(
        "--lambda-q",
        type=float,
        default=1.0,
        help="Diagnostic endpoint q exponent. Production auto pins this to 1.0.",
    )
    p_scale_replay.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_scale_replay.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    p_scale_replay.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    p_scale_replay.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_scale_replay.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    p_scale_replay.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_scale_replay.add_argument("--shelxl", default="shelxl")
    p_scale_replay.add_argument("--shelxl-iterations", type=int, default=5)
    p_scale_replay.add_argument(
        "--wght-modes",
        default="promote,freeze-zero",
        help="Comma list of endpoint weight policies; freeze-zero keeps active WGHT 0 0.",
    )
    p_scale_replay.add_argument("--publication-goof-min", type=float, default=0.85)
    p_scale_replay.add_argument("--publication-goof-max", type=float, default=1.15)
    p_scale_replay.add_argument("--publication-max-r1-cost", type=float, default=0.010)
    p_scale_replay.add_argument("--publication-free-wr2-tolerance", type=float, default=0.040)
    p_scale_replay.add_argument("--publication-free-rall-tolerance", type=float, default=0.010)
    p_scale_replay.add_argument("--publication-work-rall-tolerance", type=float, default=0.010)
    p_scale_replay.add_argument("--publication-free-wrall-tolerance", type=float, default=0.050)
    p_scale_replay.add_argument(
        "--publication-max-modulation-p95",
        type=float,
        default=20.0,
        help="Warning level only; modulation p95 never blocks endpoint selection.",
    )
    p_scale_replay.add_argument("--publication-prefer-wght-mode", default="freeze-zero")
    p_scale_replay.add_argument(
        "--publication-allow-npd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicit diagnostic override allowing NPD pathology in the selected compact-scale endpoint. Default requires NPD == 0; this strict scale-contract promotion also requires zero split warnings, even though the general endpoint chooser reports splits without withholding model handoff.",
    )
    p_scale_replay.add_argument(
        "--publication-selection-policy",
        choices=["bic", "gated-smallest"],
        default="gated-smallest",
        help="Scale topology selector. Held-out/work JANA-style row factors are authoritative; endpoint factors are reporting only.",
    )
    p_scale_replay.add_argument(
        "--publication-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
    )
    p_scale_replay.add_argument("--instability-u", type=float, default=0.01)
    p_scale_replay.add_argument("--native-wght-a", type=float, default=None)
    p_scale_replay.add_argument("--native-wght-b", type=float, default=None)
    p_scale_replay.add_argument("--detector-ridge", type=float, default=0.05)
    p_scale_replay.add_argument("--detector-smoothness", type=float, default=0.10)
    p_scale_replay.add_argument("--detector-min-rows", type=int, default=20)
    p_scale_replay.add_argument("--detector-min-holdout-improvement", type=float, default=0.002)
    p_scale_replay.add_argument("--detector-required-improved-splits", type=int, default=2)
    p_scale_replay.add_argument("--detector-max-bloch-gain-loss", type=float, default=0.002)
    p_scale_replay.add_argument("--detector-min-field-correlation", type=float, default=0.50)
    p_scale_replay.add_argument("--detector-max-field-log-rms", type=float, default=0.10)
    p_scale_replay.add_argument("--detector-independent-calibration", action="store_true")
    p_scale_replay.set_defaults(func=cmd_scale_replay)

    p_cal_endpoint = sub.add_parser(
        "calibrate-endpoint", help="Choose a corrected-HKL publication endpoint from a frozen GROND row table."
    )
    p_cal_endpoint.add_argument("--row-table", required=True)
    p_cal_endpoint.add_argument("--summary", default="")
    p_cal_endpoint.add_argument(
        "--res", default="", help="Source RES file. Defaults to the RES path recorded in the row summary."
    )
    p_cal_endpoint.add_argument("--out-dir", required=True)
    p_cal_endpoint.add_argument("--python", default="")
    add_tool_config_args(p_cal_endpoint)
    p_cal_endpoint.add_argument("--dry-run", action="store_true")
    p_cal_endpoint.add_argument(
        "--specs",
        default="baseline,adaptive:293:det8,adaptive:200:det8,adaptive:120:det8,adaptive:80:det8,window:0.5:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8,window:1.0:det4,window:3.0:det4,window:5.0:det4,window:3.0:det1",
        help="Comma list of scale contracts: baseline, adaptive:N:det8, window:DEG:det8, or stride:N:det8.",
    )
    p_cal_endpoint.add_argument("--fit-split", default="all")
    p_cal_endpoint.add_argument(
        "--lambda-q",
        type=float,
        default=1.0,
        help="Diagnostic endpoint q exponent. Production auto pins this to 1.0.",
    )
    p_cal_endpoint.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_cal_endpoint.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    p_cal_endpoint.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    p_cal_endpoint.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_cal_endpoint.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.05)
    p_cal_endpoint.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_cal_endpoint.add_argument("--shelxl", default="shelxl")
    p_cal_endpoint.add_argument("--shelxl-iterations", type=int, default=3)
    p_cal_endpoint.add_argument(
        "--wght-modes",
        default="freeze-zero,promote",
        help="Comma list of endpoint weight policies; freeze-zero keeps active WGHT 0 0.",
    )
    p_cal_endpoint.add_argument(
        "--publication-goof-min",
        type=float,
        default=0.85,
        help="Lower preferred GooF warning threshold; never a model-handoff gate.",
    )
    p_cal_endpoint.add_argument(
        "--publication-goof-max",
        type=float,
        default=1.15,
        help="Upper preferred GooF warning threshold; never a model-handoff gate.",
    )
    p_cal_endpoint.add_argument("--publication-max-r1-cost", type=float, default=0.010)
    p_cal_endpoint.add_argument("--publication-free-wr2-tolerance", type=float, default=0.040)
    p_cal_endpoint.add_argument("--publication-free-rall-tolerance", type=float, default=0.010)
    p_cal_endpoint.add_argument("--publication-work-rall-tolerance", type=float, default=0.010)
    p_cal_endpoint.add_argument("--publication-free-wrall-tolerance", type=float, default=0.050)
    p_cal_endpoint.add_argument(
        "--publication-max-modulation-p95",
        type=float,
        default=20.0,
        help="Warning level only; modulation p95 never blocks endpoint selection.",
    )
    p_cal_endpoint.add_argument("--publication-prefer-wght-mode", default="freeze-zero")
    p_cal_endpoint.add_argument(
        "--publication-allow-npd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicit diagnostic override allowing NPD pathology in the selected calibrated endpoint. Default requires NPD == 0; strict compact scale/calibration promotion can require zero split warnings, while the general endpoint chooser reports splits without withholding model handoff.",
    )
    p_cal_endpoint.add_argument(
        "--publication-selection-policy",
        choices=["bic", "gated-smallest"],
        default="gated-smallest",
        help="Scale topology selector. Held-out/work JANA-style row factors are authoritative; endpoint factors are reporting only.",
    )
    p_cal_endpoint.add_argument(
        "--publication-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
    )
    p_cal_endpoint.add_argument("--instability-u", type=float, default=0.01)
    p_cal_endpoint.add_argument("--native-wght-a", type=float, default=None)
    p_cal_endpoint.add_argument("--native-wght-b", type=float, default=None)
    p_cal_endpoint.add_argument("--detector-ridge", type=float, default=0.05)
    p_cal_endpoint.add_argument("--detector-smoothness", type=float, default=0.10)
    p_cal_endpoint.add_argument("--detector-min-rows", type=int, default=20)
    p_cal_endpoint.add_argument("--detector-min-holdout-improvement", type=float, default=0.002)
    p_cal_endpoint.add_argument("--detector-required-improved-splits", type=int, default=2)
    p_cal_endpoint.add_argument("--detector-max-bloch-gain-loss", type=float, default=0.002)
    p_cal_endpoint.add_argument("--detector-min-field-correlation", type=float, default=0.50)
    p_cal_endpoint.add_argument("--detector-max-field-log-rms", type=float, default=0.10)
    p_cal_endpoint.add_argument("--detector-independent-calibration", action="store_true")
    p_cal_endpoint.set_defaults(func=cmd_calibrate_endpoint)

    if os.environ.get("GROND_ENABLE_INTERNAL_CLI", "").strip() == "1":
        p_target_ablation = sub.add_parser(
            "target-ablation",
            help="Internal diagnostic: run XDS IOBS/RLP/PEAK target-transform ablations.",
        )
        p_target_ablation.add_argument("--xds-dir", default="", help="XDS directory containing INTEGRATE.HKL.")
        p_target_ablation.add_argument(
            "--hkl", default="", help="Explicit XDS INTEGRATE.HKL path. Use this instead of --xds-dir when needed."
        )
        p_target_ablation.add_argument("--structure", required=True)
        p_target_ablation.add_argument("--baseline-hkl", required=True)
        p_target_ablation.add_argument("--out-dir", required=True)
        p_target_ablation.add_argument(
            "--variants", default=",".join(XDS_TARGET_MODES), help="Comma list of XDS target modes, or all."
        )
        p_target_ablation.add_argument("--python", default="")
        add_tool_config_args(p_target_ablation)
        p_target_ablation.add_argument("--shelxl", default=default_shelxl_executable())
        p_target_ablation.add_argument("--shelxl-feedback-cycles", type=int, default=5)
        p_target_ablation.add_argument("--shelxl-feedback-shelxl-iterations", type=int, default=1)
        p_target_ablation.add_argument("--endpoint-ls-cycles", type=int, default=1)
        p_target_ablation.add_argument("--hkl-transform", default="identity")
        p_target_ablation.add_argument("--trust-hkl-transform", action="store_true")
        p_target_ablation.add_argument("--min-d-spacing", type=float, default=0.70)
        p_target_ablation.add_argument("--min-peak", type=float, default=90.0)
        p_target_ablation.add_argument("--geometry-percentile", type=float, default=99.0)
        p_target_ablation.add_argument("--workers", type=int, default=0)
        p_target_ablation.add_argument("--max-core-fraction", type=float, default=0.85)
        p_target_ablation.add_argument("--chunksize", type=int, default=0)
        p_target_ablation.add_argument("--compute-backend", default="scipy")
        p_target_ablation.add_argument(
            "--row-plan-cache",
            choices=["auto", "off", "exact", "readwrite", "rebuild"],
            default="auto",
        )
        p_target_ablation.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
        p_target_ablation.add_argument("--hand-control", action=argparse.BooleanOptionalAction, default=True)
        p_target_ablation.add_argument(
            "--make-maps",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Ablation skips maps by default to keep the four branches fast.",
        )
        p_target_ablation.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
        p_target_ablation.add_argument("--dry-run", action="store_true")
        p_target_ablation.add_argument("--fail-fast", action="store_true")
        p_target_ablation.add_argument(
            "--extra-auto-args",
            default="",
            help="Advanced: quoted argument string appended to every inner auto command.",
        )
        p_target_ablation.set_defaults(func=cmd_target_ablation)
