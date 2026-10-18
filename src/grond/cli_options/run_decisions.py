"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

from grond.validation.heldout import DEFAULT_FREE_SALT


def add_decision_controls(ap: argparse.ArgumentParser) -> None:
    """Add decision controls options in the established order."""

    ap.add_argument("--max-rows", type=int, default=0)
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
        "--trust-hkl-transform",
        action="store_true",
        help="Trust the supplied --hkl-transform or prep-file transform and skip Bloch transform scouting while keeping the rest of auto-decision enabled.",
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
