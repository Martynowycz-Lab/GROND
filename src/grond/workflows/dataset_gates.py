"""Apply dataset inclusion and weighting decisions to the runs used for joint refinement."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from grond.workflows.dataset_policy import build_dataset_policy, policy_by_name, write_policy_artifacts
from grond.workflows.decision import nuisance_score_rows, safe_float


def dataset_policy_kwargs(args: argparse.Namespace) -> dict:
    """Collect CLI thresholds for multi-dataset inclusion policy decisions.

    Under --nuisance-score-rows work the quality metrics read work rows only (free rows stay held out);
    legacy reads all rows.
    """

    return {
        "selection_rows": "work" if nuisance_score_rows(args) == "work" else "all",
        "min_rows": args.dataset_policy_min_rows,
        "min_weight": args.dataset_policy_min_weight,
        "primary_min_quality": args.dataset_policy_primary_min_quality,
        "min_raw_quality": args.dataset_policy_min_raw_quality,
        "min_cc": args.dataset_policy_min_cc,
        "max_wr2": args.dataset_policy_max_wr2,
        "max_r1": args.dataset_policy_max_r1,
        "unique_gain_fraction": args.dataset_policy_unique_gain_fraction,
        "wr2_scale": args.dataset_policy_wr2_scale,
        "r1_scale": args.dataset_policy_r1_scale,
        "cc_scale": args.dataset_policy_cc_scale,
    }


def all_primary_policy(runs: dict[str, Path], *, status: str) -> dict:
    """Return a no-op full-weight policy for disabled or dry-run contexts."""

    return {
        "status": status,
        "definition": "multi-dataset inclusion/downweight policy",
        "datasets": [
            {
                "name": name,
                "run_dir": str(Path(run).resolve()),
                "action": "primary",
                "raw_row_weight": 1.0,
                "endpoint_weight": 1.0,
                "include_in_row_regularization": True,
                "include_in_endpoint": True,
                "reasons": ["policy not evaluated; dataset kept at full weight"],
            }
            for name, run in runs.items()
        ],
    }


def run_dataset_policy_gate(runs: dict[str, Path], out_dir: Path, args: argparse.Namespace, *, label: str) -> dict:
    """Decide which datasets should be full, downweighted, endpoint-only, or excluded."""

    mode = getattr(args, "dataset_policy", "auto")
    if mode == "off" or len(runs) < 2:
        return all_primary_policy(runs, status="disabled" if mode == "off" else "single_dataset")
    if args.dry_run:
        return all_primary_policy(runs, status="dry_run")
    missing = [
        str(Path(run) / "native_dyn_calc.csv")
        for run in runs.values()
        if not (Path(run) / "native_dyn_calc.csv").exists()
    ]
    if missing:
        return {"status": "missing_row_table", "missing": missing}
    policy = build_dataset_policy(runs, **dataset_policy_kwargs(args))
    policy["label"] = label
    policy["artifacts"] = write_policy_artifacts(policy, out_dir)
    return policy


def require_materialized_dataset_policy(policy: Mapping[str, object], *, context: str, dry_run: bool) -> None:
    """Stop a multi workflow instead of degrading after a missing row table."""

    if dry_run or str(policy.get("status") or "") != "missing_row_table":
        return
    missing = policy.get("missing") if isinstance(policy.get("missing"), list) else []
    raise RuntimeError(
        f"multi-dataset {context} policy is missing a required materialized native row table; "
        "refusing to silently fall back to equal dataset weights: " + ", ".join(str(path) for path in missing)
    )


def apply_dataset_policy_to_nuisance(nuisance: dict[str, dict], policy: dict) -> dict[str, dict]:
    """Merge dataset policy weights into the datasets CSV nuisance rows."""

    decisions = policy_by_name(policy)
    out: dict[str, dict] = {}
    for name, nui in nuisance.items():
        row = dict(nui or {})
        decision = decisions.get(name, {})
        action = str(decision.get("action", "primary"))
        raw_weight = safe_float(decision.get("raw_row_weight"), 1.0)
        endpoint_weight = safe_float(decision.get("endpoint_weight"), raw_weight)
        include_raw = action in {"primary", "downweight"}
        row["include"] = include_raw
        row["dataset_role"] = action
        row["dataset_weight"] = raw_weight if include_raw else 0.0
        row["raw_row_weight"] = row["dataset_weight"]
        row["endpoint_weight"] = endpoint_weight
        out[name] = row
    return out


def policy_active_runs(runs: dict[str, Path], policy: dict) -> dict[str, Path]:
    """Return runs that should contribute to raw-row metrics/regularization."""

    decisions = policy_by_name(policy)
    active: dict[str, Path] = {}
    for name, run in runs.items():
        action = str(decisions.get(name, {}).get("action", "primary"))
        if action in {"primary", "downweight"}:
            active[name] = run
    return active


def require_policy_active_runs(runs: Mapping[str, Path], *, context: str) -> None:
    """Stop when an evaluated policy leaves no raw-row anchor dataset."""

    if runs:
        return
    raise RuntimeError(
        f"multi-dataset {context} policy selected no primary or downweighted dataset; "
        "refusing to resurrect excluded/completeness-only crystals or publish an unanchored endpoint"
    )
