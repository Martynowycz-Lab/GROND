"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

from grond.validation.heldout import DEFAULT_FREE_SALT


def add_decision_controls(ap: argparse.ArgumentParser) -> None:
    """Add decision controls options in the established order."""

    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument(
        "--scout-max-rows", type=int, default=1000, help="Representative row limit for automatic transform/hand scouts."
    )
    ap.add_argument("--scout-n-basis", type=int, default=8, help="Basis size for cheap decision scouts.")
    ap.add_argument("--scout-n-points", type=int, default=3, help="Quadrature points for cheap decision scouts.")
    ap.add_argument(
        "--transform-rerank-top",
        type=int,
        default=3,
        help="Number of coarse transform-scout winners to rerun at better N/Q; 0 disables the second stage.",
    )
    ap.add_argument(
        "--transform-rerank-n-basis", type=int, default=24, help="Basis size for second-stage transform reranking."
    )
    ap.add_argument(
        "--transform-rerank-n-points",
        type=int,
        default=10,
        help="Quadrature points for second-stage transform reranking.",
    )
    ap.add_argument(
        "--transform-rerank-max-rows",
        type=int,
        default=0,
        help="Row limit for second-stage transform reranking; 0 reuses the preflight scout row budget.",
    )
    ap.add_argument(
        "--transform-candidates",
        default="auto",
        help=(
            "Semicolon-separated signed-axis transforms, or auto/common/off. "
            "Calibration-grade auto/common enumerates all 24 determinant-positive "
            "signed-axis permutations before the cell-metric filter."
        ),
    )
    ap.add_argument(
        "--transform-metric-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum relative Frobenius mismatch between source and model "
            "reciprocal-cell metrics before a signed-axis transform is rejected. "
            "This filter uses cells only, never intensities, coordinates, chemistry, or hand."
        ),
    )
    ap.add_argument(
        "--transform-parity-policy",
        choices=["proper-only", "all-diagnostic"],
        default="proper-only",
        help=(
            "Use only determinant +1 transforms for automatic selection (default). "
            "all-diagnostic also scouts mixed-parity transforms for diagnostics, but cannot by itself "
            "support a named absolute-configuration claim."
        ),
    )
    ap.add_argument(
        "--transform-canonicalization",
        choices=["point-group", "off"],
        default="point-group",
        help=(
            "point-group (default): metric-compatible transforms that differ by a proper rotation of the model "
            "point group are one indexing (identical dynamical intensities); scout one representative per class "
            "(identity when present) and record the classes. off: scout every retained candidate."
        ),
    )
    ap.add_argument(
        "--transform-scout-jobs",
        type=int,
        default=0,
        help="Concurrent transform-scout subprocesses across candidates and datasets; 0 derives a core-safe value from --workers and --max-core-fraction.",
    )
    ap.add_argument(
        "--transform-scout-workers-per-job",
        type=int,
        default=1,
        help="Worker processes inside each transform-scout subprocess. Keep low when many scout jobs run concurrently.",
    )
    ap.add_argument(
        "--transform-scout-window",
        choices=["fixed-0.004", "inherit"],
        default="fixed-0.004",
        help=(
            "Rocking window of the HKL transform scouts. fixed-0.004 (default) ranks transforms with the fixed "
            "0.004 1/A window they were validated with; inherit uses --sg-window and the resolved "
            "--sg-window-mode."
        ),
    )
    ap.add_argument(
        "--transform-metadata-prefilter",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Use solved/XSCALE baseline-HKL overlap as a diagnostic ordering aid. "
            "Default auto never eliminates candidates; explicit on may eliminate candidates "
            "but is not eligible for parity calibration."
        ),
    )
    ap.add_argument(
        "--transform-metadata-max-candidates",
        type=int,
        default=3,
        help="Number of metadata-ranked candidates retained only by the explicit diagnostic --transform-metadata-prefilter on mode. Calibration-grade auto does not eliminate metric-compatible candidates.",
    )
    ap.add_argument(
        "--transform-metadata-min-pairs",
        type=int,
        default=100,
        help="Minimum paired positive-intensity rows required before trusting metadata transform ranking.",
    )
    ap.add_argument(
        "--transform-metadata-min-correlation",
        type=float,
        default=0.70,
        help="Minimum log-intensity correlation required before metadata can prefilter transform scouts.",
    )
    ap.add_argument(
        "--transform-metadata-min-correlation-gap",
        type=float,
        default=0.05,
        help="Correlation gap used only by the explicit diagnostic metadata-elimination mode; calibration-grade auto still Bloch-scouts every metric-compatible candidate.",
    )
    ap.add_argument(
        "--transform-metadata-max-rows",
        type=int,
        default=0,
        help="Maximum INTEGRATE/XDS rows used by metadata transform ranking; 0 uses all rows.",
    )
    ap.add_argument(
        "--trust-hkl-transform",
        action="store_true",
        help="Trust the supplied --hkl-transform or prep-file transform and skip Bloch transform scouting while keeping the rest of auto-decision enabled.",
    )
    ap.add_argument(
        "--hand-accept-delta",
        dest="hand_accept_delta",
        metavar="DELTA",
        type=float,
        default=0.015,
        help="Adaptive score margin required before switching to the alternate hand.",
    )
    ap.add_argument(
        "--hand-max-wR2-all",
        type=float,
        default=0.45,
        help="Maximum alternate-hand all-row wR2 allowed before automatic hand switching. Poor row contracts still report the hand diagnostic but keep the input hand.",
    )
    ap.add_argument(
        "--hand-max-free-wR2-all",
        type=float,
        default=0.50,
        help="Maximum alternate-hand free-row wR2 allowed before automatic hand switching.",
    )
    ap.add_argument(
        "--hand-min-free-wR2-gain",
        type=float,
        default=0.0,
        help="Minimum free-row wR2 improvement required before automatic scout-stage hand switching.",
    )
    ap.add_argument(
        "--hand-min-adjusted-z",
        type=float,
        default=2.0,
        help=(
            "Minimum absolute common-source-sigma Klar-Palatinus adjusted hand Z "
            "required before a matched hand decision may train or consume an "
            "external setup-parity calibration. Weaker decisions remain "
            "residual-preference diagnostics."
        ),
    )
    ap.add_argument(
        "--hand-flip-zkp-all",
        type=float,
        default=3.5,
        help=(
            "Legacy --hand-decision-evidence all-row-zkp only: invert the acentric model once when the "
            "all-row Klar-Palatinus adjusted Z favors the inverse by at least this magnitude (default 3.5). "
            "The ZKP is always reported as a diagnostic."
        ),
    )
    ap.add_argument(
        "--hand-decision-evidence",
        choices=["heldout-orbit-sign", "all-row-zkp"],
        default="heldout-orbit-sign",
        help=(
            "Evidence that may switch the hand. heldout-orbit-sign (default): out-of-fold work rows (3 salted "
            "orbit folds, scales refitted on training orbits) plus free rows, pooled over crystals; switch iff "
            "the orbit sign z favours the inverse by --hand-flip-orbit-z AND the paired orbit-bootstrap "
            "delta R_obs 95%% CI excludes zero in favour of the inverse. all-row-zkp: legacy all-row ZKP rule."
        ),
    )
    ap.add_argument(
        "--hand-flip-orbit-z",
        type=float,
        default=3.0,
        help="Minimum held-out orbit sign z in favour of the inverse before switching hands (default 3.0).",
    )
    ap.add_argument(
        "--hand-decision-stage",
        choices=["early", "final"],
        default="early",
        help=(
            "early (default): decide the hand right after the screen on the hand-blind start model, so thickness, "
            "gates and the SHELXL loop refine the chosen hand; the final elastic pair is then a confirmatory "
            "check with the same rule. final: legacy, decide only after all refinement."
        ),
    )
    ap.add_argument(
        "--hand-pair-scale-fit-split",
        choices=["work", "inherit"],
        default="work",
        help=(
            "Rows used for both hands' scale fits in the elastic hand pair: work (default; free orbits held out) "
            "or inherit (legacy: the source run's split, 'all' under --production-scales)."
        ),
    )
    ap.add_argument(
        "--hand-pair-absorption",
        choices=["uniform", "strip"],
        default="uniform",
        help=(
            "uniform (default): keep a uniform absorptive potential in the exact hand pair (H0 + i*mu*I) and "
            "refuse non-uniform terms; strip: legacy, zero every absorption term for the hand decision."
        ),
    )
    ap.add_argument(
        "--hand-early-escalation",
        choices=["final-fidelity", "off"],
        default="final-fidelity",
        help=(
            "final-fidelity (default): when the early rule is not met at screen fidelity and the evidence is not "
            "clearly against switching (the mirrored rule does not hold), recompute the early exact pair at the "
            "final N/Q and final thickness points with the screen nuisance, before any refinement, and decide on "
            "that. off: legacy, the early decision uses the screen-fidelity pair only."
        ),
    )
    ap.add_argument(
        "--hand-evidence-datasets",
        choices=["not-excluded", "policy-active"],
        default="not-excluded",
        help=(
            "Crystals pooled in the final (confirmatory) hand check of a multi-crystal run. not-excluded "
            "(default): every crystal except one the final dataset policy excluded for a recorded reason; "
            "downweighted and completeness-only crystals stay in the evidence. policy-active: legacy, primary "
            "and downweighted crystals only."
        ),
    )
    ap.add_argument(
        "--validation-console",
        choices=["on", "off"],
        default="on",
        help=(
            "on (default): print the validation numbers at every step (split, held-out R after the screen, hand "
            "blocks, nuisance decisions, thickness, SHELXL macro-cycles), read back from the recorded JSON "
            "artifacts. off: legacy console."
        ),
    )
    ap.add_argument(
        "--report-layout",
        choices=["heldout-first", "legacy"],
        default="heldout-first",
        help=(
            "heldout-first (default): FINAL_SUMMARY.md, QUICK_STATS.md and the terminal finish summary open with "
            "the held-out validation and show the held-out hand evidence, with the all-row ZKP as a labelled "
            "diagnostic. legacy: the summaries open with the all-row statistics, and the held-out block follows them."
        ),
    )
    ap.add_argument(
        "--free-set",
        choices=["off", "on"],
        default="off",
        help=(
            "off (default): after validation, refine the chosen model once more against all reflections. on: keep the held-out reflections out of the final model (all runs in the paper)."
        ),
    )
    ap.add_argument(
        "--split-scheme",
        choices=["orbit-hash", "legacy"],
        default="orbit-hash",
        help=(
            "Free-set definition. orbit-hash (default): crystal-independent salted hash of the model Laue-orbit "
            "key (grond.validation.heldout), so one orbit is free in every crystal. legacy: per-dataset shuffle "
            "with --split-seed."
        ),
    )
    ap.add_argument(
        "--split-salt",
        default=DEFAULT_FREE_SALT,
        help=f"Salt of the orbit-hash free set (default {DEFAULT_FREE_SALT}).",
    )
    ap.add_argument(
        "--metric-incompatible-datasets",
        choices=["exclude", "fail"],
        default="exclude",
        help=(
            "A crystal of a multi-crystal run whose unit cell is incompatible with the model under every transform "
            "candidate (cell-metric filter). exclude (default): drop it before any scout, with the reason recorded in "
            "decisions/metric_excluded_datasets.json; the run fails only if no crystal is left. fail: stop the run."
        ),
    )
    ap.add_argument(
        "--split-candidate-set",
        choices=["shared-closure", "per-crystal"],
        default="shared-closure",
        help=(
            "Transform candidates behind every crystal's split contract. shared-closure (default): one set for all "
            "crystals, the union of their retained candidates closed under composition modulo the model Laue "
            "group, so free and excluded status depends on the orbit alone (pseudo-symmetric cells and one-sided "
            "XDS P1 fallbacks proceed). per-crystal: each crystal's own retained candidates."
        ),
    )
    ap.add_argument(
        "--exclude-orbits-json",
        default="",
        help=(
            "Lockbox file (GROND splits make): an optional set of orbits held back from every stage (orbit-hash "
            "only). Rows in those orbits are dropped from every fit, selection and report; the list SHA-256 is "
            "verified and recorded in the split contract."
        ),
    )
    ap.add_argument(
        "--leak-free-start",
        choices=["on", "off"],
        default="on",
        help=(
            "on (default): before any stage, refine the supplied model conventionally against the baseline HKL with "
            "the free (and excluded) orbits removed, and start every stage from that model, so free-row R is not "
            "inflated by a start refined on the free rows. off: legacy, start from the supplied model."
        ),
    )
    ap.add_argument("--hand-min-z-all", type=float, default=2.0, help=argparse.SUPPRESS)
    ap.add_argument("--hand-min-z-free", type=float, default=0.5, help=argparse.SUPPRESS)
    ap.add_argument("--hand-min-free-rows", type=int, default=10, help=argparse.SUPPRESS)
    ap.add_argument("--stop-on-bad-preflight", action="store_true", default=False)
    ap.add_argument("--stop-on-bad-fit", action="store_true", default=False)
    ap.add_argument(
        "--stop-if-wR2-above",
        type=float,
        default=0.75,
        help="Bad-data stop threshold for native all-row dynamic wR2 when --stop-on-bad-fit is enabled.",
    )
    ap.add_argument(
        "--min-bloch-wR2-gain",
        type=float,
        default=-0.01,
        help="Minimum wR2 gain over Born before structure update is considered reliable.",
    )
    ap.add_argument(
        "--feature-accept-delta",
        type=float,
        default=0.002,
        help="Minimum adaptive-score improvement before keeping a staged row-resolution/absorption/jitter/error/dose feature family.",
    )
    ap.add_argument(
        "--nuisance-score-rows",
        choices=["work", "legacy"],
        default="work",
        help=(
            "Rows behind the in-sample nuisance scores (feature gates, Sg-window scout, discrete and continuous "
            "thickness promotion, multi-dataset policy). work (default): work-row wR2 + 0.10 work-row R1, work-row "
            "R1 vetoes and no free-row vetoes, so free rows stay held out. legacy: all-row wR2 + 0.25 free wR2 + "
            "0.10 all-row R1 with the all-row R1 and free-wR2 no-harm vetoes."
        ),
    )
