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
        help="SHELXL executable calls per corrected-HKL endpoint (per-dataset endpoints and the final combined endpoint); calls stop early once |max shift/esd| < --shelxl-convergence-shift-esd. Per-dataset endpoints use --endpoint-ls-cycles internal L.S./CGLS cycles per call, combined endpoints --shelxl-feedback-endpoint-ls-cycles.",
    )
    ap.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Diagnostic internal L.S./CGLS cycle count for generated corrected-HKL endpoint files. Production keeps this at 1.",
    )
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote", "promote-gold", "converge"],
        default="converge",
        help="SHELXL weight policy for corrected-HKL endpoint files. converge (default) starts each handoff at WGHT 0 0 and promotes SHELXL's suggested WGHT until the applied and suggested WGHT agree (|da| < 0.002, |db| < 0.02) and max shift/esd < --shelxl-convergence-shift-esd, at most --wght-converge-max-calls calls, so the written .res carries the converged WGHT; promote-gold starts at WGHT 0.1 0 and promotes on later calls until shift/esd converges; promote keeps the source-model WGHT on the first call and then promotes; freeze-zero is the legacy WGHT 0 0 after GROND sigma calibration; inherit keeps the source-model WGHT.",
    )
    ap.add_argument(
        "--wght-converge-max-calls",
        type=int,
        default=20,
        help="Maximum SHELXL calls of one endpoint handoff under the converge weighting mode.",
    )
    ap.add_argument(
        "--shelxl-feedback-wght-mode",
        choices=["endpoint", "freeze-zero", "inherit", "promote", "promote-gold", "converge"],
        default="endpoint",
        help="SHELXL weight policy of the structure-feedback macro-cycles. endpoint (default) uses --endpoint-wght-mode; the other values are the --endpoint-wght-mode policies (promote-gold is cheaper than converge). The rule used is recorded in the loop summary.",
    )
    ap.add_argument(
        "--endpoint-intensity-scale",
        choices=["absolute", "median-iobs"],
        default="absolute",
        help="Corrected-intensity scale of every dynamical handoff HKL. absolute (default) writes kappa*|F|^2 * Iobs/Icalc with kappa=10 and resets the SHELXL OSF to sqrt(kappa) at every handoff; median-iobs is the legacy per-cycle median rescale.",
    )
    ap.add_argument(
        "--endpoint-q-policy",
        choices=["exclude", "clip"],
        default="exclude",
        help="Rows whose correction ratio relative to the dataset median lies outside (0.05, 20). exclude (default) drops and counts them; clip is the legacy q clip into [0.05, 20].",
    )
    ap.add_argument(
        "--endpoint-include-free-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write split=free rows into the feedback-loop and publication HKLs. Default off: free rows are predicted and scored every cycle but never refined against, so R_free is honest. --endpoint-include-free-rows is the legacy setting.",
    )
    ap.add_argument(
        "--shelxl-convergence-shift-esd",
        type=float,
        default=0.01,
        help="Stop the SHELXL calls of one handoff once the last L.S. cycle has |max shift/esd| below this value. 0 disables the test (legacy: a fixed number of calls).",
    )
    ap.add_argument(
        "--shelxl-npd-recovery",
        choices=["isor", "damp"],
        default="isor",
        help="Recovery when an ADP goes non-positive definite: isor (default) restrains only the NPD atoms with ISOR 0.01 0.02 and repeats the call; damp is the legacy DAMP 100..10000 ladder.",
    )
    ap.add_argument(
        "--shelxl-carry-damp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy: carry an active DAMP instruction into later SHELXL calls, macro-cycles and the publication endpoint. Default strips DAMP at every handoff.",
    )
    ap.add_argument(
        "--shelxl",
        default=default_shelxl_executable(),
        help="SHELXL executable (default: from --config or PATH).",
    )
    ap.add_argument(
        "--shelxl-feedback-cycles",
        type=int,
        default=5,
        help="Most rounds of Bloch calculation and SHELXL refinement (default %(default)s).",
    )
    ap.add_argument(
        "--shelxl-feedback-shelxl-iterations",
        type=int,
        default=15,
        help="Maximum SHELXL calls per feedback macro-cycle; calls stop once |max shift/esd| < --shelxl-convergence-shift-esd. 1 is the legacy one-step update.",
    )
    ap.add_argument(
        "--shelxl-feedback-endpoint-ls-cycles",
        type=int,
        default=10,
        help="Internal L.S./CGLS cycles per SHELXL call in the feedback macro-cycles and the shared combined endpoints. 1 runs a single cycle per call.",
    )
    ap.add_argument(
        "--shelxl-feedback-xyz-tol",
        type=float,
        default=0.002,
        help="Stop the macro-cycles once the max non-H coordinate shift between successive models is below this value in A. 0 disables the test (legacy: always run --shelxl-feedback-cycles).",
    )
    ap.add_argument(
        "--shelxl-feedback-keep",
        choices=["best-free", "last"],
        default="last",
        help=(
            "Macro-cycle model to keep: last (default) keeps the last adopted cycle, so free rows never choose the "
            "model; best-free keeps the output with the lowest free-row R_obs, ties to the later cycle, and is flagged "
            "as a free-row selection in the report."
        ),
    )
    ap.add_argument(
        "--dataset-sigma-policy",
        choices=["auto", "none", "uniform_z_rms", "error_model"],
        default="error_model",
        help=(
            "Per-crystal SHELXL/merge sigma policy. error_model (default) fits sigma'^2 = k^2 (sigma^2 + (u P)^2) per "
            "crystal on work rows against the current model, so merged crystals get correct relative inverse-variance "
            "weights, and reports the free-row RMS(z); it applies to the combined endpoints and to every final and "
            "feedback per-crystal endpoint, and the combined post-Bloch rescale resolves to none. auto applies "
            "uniform_z_rms per crystal before merging and on the combined HKL."
        ),
    )
    ap.add_argument(
        "--error-model-refits",
        type=int,
        default=1,
        help="error_model only: after SHELXL has refined an endpoint, refit each crystal's error model against the refined model (work rows only, intensities unchanged) and refine again, this many times. 0 fits against the entering model only.",
    )
    ap.add_argument(
        "--error-model-dataset-weight",
        choices=["off", "apply"],
        default="off",
        help=(
            "error_model only: off (default) lets each crystal's fitted error model set its SHELXL weight alone, so a "
            "dataset-policy downweight (for example 0.25) does not inflate that crystal's sigmas a second time; "
            "crystals the policy excludes stay out. apply also scales the crystal's SHELXL sigmas by the policy weight "
            "(sigma / sqrt(w)). The row-table (JANA-style) statistics are unaffected."
        ),
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
        choices=["auto", "none", "uniform_z_rms", "error_model"],
        default="auto",
        help="Endpoint uncertainty policy. Auto leaves early physics-selection endpoints uncalibrated; for final corrected endpoints and SHELXL feedback cycles it uses the per-crystal error model under --dataset-sigma-policy error_model (none on the merged combined HKL, which already carries per-crystal sigmas) and uniform_z_rms otherwise.",
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
        "--unmerged-endpoint",
        choices=["on", "off"],
        default="on",
        help="Multi-crystal runs: next to the merged combined endpoint, write and refine an unmerged SHELXL endpoint (every corrected work row, batch = crystal index, MERG 0, one refined BASF per extra crystal). off is the legacy setting.",
    )
    ap.add_argument(
        "--endpoint-free-validation",
        choices=["on", "off"],
        default="on",
        help="Reporting only: score every final SHELXL model on the held-out (free) orbits with SHELXL L.S. 0 (no parameter moves) next to its own work reflections: the GROND merged and unmerged combined endpoints (free rows corrected exactly like the work rows) and the conventional leak-free start (free reflections of the baseline HKL). R1gt, R1all, wR2 and GooF on free and on work go to HELDOUT_VALIDATION and Table 1; nothing is chosen on them. off is the legacy setting (the combine command is then unchanged).",
    )
    ap.add_argument(
        "--cell-esd-policy",
        choices=["auto", "source"],
        default="auto",
        help="Cell e.s.d.s (ZERR) of the SHELXL endpoints. auto (default): --cell-esd if given, else the sample standard deviation of the crystals' cells (two or more crystals), else the crystal's XDS CORRECT.LP e.s.d.s or DIALS scan-varying cell spread; symmetry-fixed parameters get 0. Without a measured source the starting model's ZERR e.s.d.s are used (its ZERR is kept unchanged when they are zero) and a warning is printed. source writes the model's ZERR unchanged.",
    )
    ap.add_argument(
        "--cell-esd",
        default="",
        help="Explicit cell e.s.d.s a,b,c,alpha,beta,gamma (A, deg) in the model setting for ZERR.",
    )
    ap.add_argument(
        "--shelx-ins-cleanup",
        choices=["on", "off"],
        default="on",
        help="Olex2/SHELXLE readiness of generated SHELXL inputs: drop stale difference-map peaks (Q1, Q2, ...) and write LIST 6 so every endpoint has an .fcf with phases for maps. off is the legacy setting (the model's peaks and LIST kept).",
    )
    ap.add_argument(
        "--shelx-acta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add ACTA to the final endpoints so SHELXL writes a CIF next to the .res (off by default; grond check-shelx --acta does the same for any endpoint).",
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
        "--publication-feedback-cycles",
        choices=["kept", "all"],
        default="kept",
        help=(
            "SHELXL feedback cycles offered to the publication chooser: kept (default) offers only the cycle the loop "
            "kept, the model the row tables, held-out numbers and maps describe; all ranks every adopted cycle by "
            "in-sample R1, which compares refinements against differently corrected data."
        ),
    )
    ap.add_argument(
        "--publication-endpoint-min-hkl-safe-factor",
        type=float,
        default=0.25,
        help="Warn below this fixed-width HKL rescaling factor; written precision, not intensity units, determines validity.",
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
        help="Explicit diagnostic override allowing endpoints with NPD pathology to pass publication selection gates. Default requires NPD == 0. The endpoint chooser reports split warnings without withholding a usable model.",
    )
    ap.add_argument(
        "--instability-u",
        type=float,
        default=0.01,
        help="JANA/PETS-style row-weight instability factor u used by wR/MwR reporting and row-scale diagnostics.",
    )

