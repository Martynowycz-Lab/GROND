"""Build the public command grammar in its established display order."""

from __future__ import annotations

import argparse


def register_parity(sub: argparse._SubParsersAction) -> None:
    """Register parity commands."""

    from grond.commands.parity import cmd_apply_parity_calibration, cmd_canonicalize_hand, cmd_create_parity_calibration

    p_canonicalize = sub.add_parser(
        "canonicalize-hand",
        help=(
            "Canonicalize an endpoint model/improper HKL transform pair to an explicitly "
            "inverted RES and the proper sign-paired transform."
        ),
    )
    p_canonicalize.add_argument("--input-res", required=True)
    p_canonicalize.add_argument("--input-hkl", required=True)
    p_canonicalize.add_argument("--hkl-transform", required=True)
    p_canonicalize.add_argument("--output-res", required=True)
    p_canonicalize.add_argument("--output-hkl", required=True)
    p_canonicalize.add_argument("--manifest", default="")
    p_canonicalize.add_argument(
        "--allow-proper-noop",
        action="store_true",
        help="Allow an audited copy when the supplied transform is already proper.",
    )
    p_canonicalize.add_argument(
        "--matched-hand-decision",
        default="",
        help="Frozen matched-hand decision JSON to hash and bind after transform selection.",
    )
    p_canonicalize.add_argument(
        "--transform-decision",
        default="",
        help="Optional frozen pre-canonicalization transform-decision JSON to hash in the audit manifest.",
    )
    p_canonicalize.add_argument("--overwrite", action="store_true")
    p_canonicalize.set_defaults(func=cmd_canonicalize_hand)

    p_create_calibration = sub.add_parser(
        "create-parity-calibration",
        help=(
            "Legacy-only: create a setup-parity manifest from a reserved known calibrant using the older "
            "branch-symmetric/common-sigma hand-decision artifact. Current final-elastic auto artifacts are not accepted."
        ),
        description=(
            "Create a setup-parity manifest from one reserved known calibrant. This replay currently accepts only "
            "the older hash-bound branch-symmetric/common-sigma hand-decision format; it does not accept the current "
            "final negative-thickness auto artifact."
        ),
    )
    p_create_calibration.add_argument("--setup-id", required=True)
    p_create_calibration.add_argument("--calibrant-dataset-id", required=True)
    p_create_calibration.add_argument(
        "--calibrant-physical-crystal-id",
        required=True,
        help="Stable specimen/crystal identity; this crystal is excluded from validation.",
    )
    p_create_calibration.add_argument(
        "--calibrant-acquisition-fingerprint",
        required=True,
        help="SHA-256 of the calibrant raw-acquisition/image manifest.",
    )
    p_create_calibration.add_argument(
        "--setup-manifest-sha256",
        required=True,
        help="SHA-256 of the acquisition/import setup manifest shared by calibrant and validation.",
    )
    p_create_calibration.add_argument("--transform-decision", required=True)
    p_create_calibration.add_argument("--matched-hand-decision", required=True)
    p_create_calibration.add_argument("--expected-known-configuration", required=True)
    p_create_calibration.add_argument(
        "--expected-known-branch",
        choices=["original", "alternate"],
        required=True,
    )
    p_create_calibration.add_argument(
        "--reserved-from-validation",
        action="store_true",
        help="Required acknowledgement that this calibrant is excluded from validation.",
    )
    p_create_calibration.add_argument("--out", required=True)
    p_create_calibration.add_argument("--overwrite", action="store_true")
    p_create_calibration.set_defaults(func=cmd_create_parity_calibration)

    p_apply_calibration = sub.add_parser(
        "apply-parity-calibration",
        help=(
            "Legacy-only: apply a reserved setup-parity calibration to a different proper-locked validation "
            "decision in the older branch-symmetric/common-sigma format."
        ),
        description=(
            "Apply a legacy setup-parity manifest prospectively to a different validation crystal. Both the "
            "calibrant and validation hand decisions must use the older branch-symmetric/common-sigma format; "
            "current final-elastic auto artifacts are not yet supported."
        ),
    )
    p_apply_calibration.add_argument("--calibration", required=True)
    p_apply_calibration.add_argument("--setup-id", required=True)
    p_apply_calibration.add_argument("--validation-dataset-id", required=True)
    p_apply_calibration.add_argument(
        "--validation-physical-crystal-id",
        required=True,
        help="Stable specimen/crystal identity; must differ from the reserved calibrant.",
    )
    p_apply_calibration.add_argument(
        "--validation-acquisition-fingerprint",
        required=True,
        help="SHA-256 of the validation raw-acquisition/image manifest.",
    )
    p_apply_calibration.add_argument(
        "--setup-manifest-sha256",
        required=True,
        help="SHA-256 of the same setup manifest used by the reserved calibrant.",
    )
    p_apply_calibration.add_argument("--transform-decision", required=True)
    p_apply_calibration.add_argument("--matched-hand-decision", required=True)
    p_apply_calibration.add_argument("--out", required=True)
    p_apply_calibration.add_argument("--overwrite", action="store_true")
    p_apply_calibration.set_defaults(func=cmd_apply_parity_calibration)
