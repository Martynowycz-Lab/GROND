"""Record the discrete thickness promotion under one row contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json


def discrete_thickness_promotion_record(run: Path, *, dry_run: bool) -> dict:
    """Return the engine's auditable post-ranking thickness decision."""

    run = Path(run)
    artifact = run / "discrete_thickness_promotion.json"
    if dry_run:
        return {
            "family": "discrete_thickness_polish",
            "status": "planned",
            "run": str(run),
            "artifact": str(artifact),
        }
    summary_path = run / "summary.json"
    summary = try_read_json(summary_path)
    decision = summary.get("promotion_decision")
    if not isinstance(decision, Mapping) or str(decision.get("status", "")) == "not_requested":
        raise RuntimeError(
            f"discrete thickness stage did not record the required held-out promotion decision: {summary_path}"
        )
    return {
        **dict(decision),
        "run": str(run),
        "artifact": str(artifact),
    }


def discrete_thickness_baseline_nuisance(args: argparse.Namespace, nuisance: Mapping[str, object] | None) -> dict:
    """Require a real starting nuisance, with a planning-only dry-run seed."""

    if nuisance:
        return dict(nuisance)
    if bool(getattr(args, "dry_run", False)):
        return {
            "thickness_A": 1200.0,
            "thickness_grad_frac": 0.0,
            "frame_offset": 0.0,
        }
    raise RuntimeError(
        "discrete thickness polishing has no starting nuisance from the preceding stage; "
        "refusing to run an unguarded work-only search"
    )
