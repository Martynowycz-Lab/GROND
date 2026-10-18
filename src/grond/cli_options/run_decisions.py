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
            "Threshold of the all-row Klar-Palatinus adjusted Z reported with the hand evidence (default 3.5); "
            "the held-out orbit-sign rule decides the hand and the ZKP is a diagnostic."
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
