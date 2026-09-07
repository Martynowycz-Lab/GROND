"""Materialize a chosen row-scale contract without changing its observations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.paths import engine_script


def materialize_scale_contract_cmd(
    python: str,
    raw_run_dir: Path,
    out_dir: Path,
    spec: str,
) -> list[str]:
    """Build the endpoint-free compact scale replay for a frozen native row table."""

    return [
        python,
        str(engine_script("scale_contract_replay.py")),
        "--row-table",
        str(raw_run_dir / "native_dyn_calc.csv"),
        "--summary",
        str(raw_run_dir / "summary.json"),
        "--out-dir",
        str(out_dir),
        "--specs",
        str(spec),
        "--skip-endpoint-refinement",
    ]


def materialize_dataset_scale_contract(
    python: str,
    raw_run_dir: Path,
    args: argparse.Namespace,
    dataset_name: str,
    logs: Path,
    *,
    label: str,
) -> Path:
    """Return raw rows or their explicitly requested compact-scale replay."""

    specifications = getattr(args, "dataset_scale_specs", {})
    specifications = specifications if isinstance(specifications, Mapping) else {}
    spec = str(specifications.get(dataset_name, "")).strip()
    if not spec:
        return raw_run_dir
    out_dir = raw_run_dir / "scale_contract_replay"
    run_logged(
        materialize_scale_contract_cmd(python, raw_run_dir, out_dir, spec),
        logs,
        f"{dataset_name}_{label}_scale_contract",
        dry_run=False,
    )
    payload = try_read_json(out_dir / "summary.json")
    materialized = payload.get("materialized_runs") if isinstance(payload, Mapping) else {}
    selected = ""
    if isinstance(materialized, Mapping) and len(materialized) == 1:
        selected = str(next(iter(materialized.values()))).strip()
    run_dir = Path(selected) if selected else None
    if run_dir is None or not (run_dir / "native_dyn_calc.csv").is_file() or not (run_dir / "summary.json").is_file():
        raise RuntimeError(
            f"{dataset_name} scale contract {spec!r} did not materialize a complete row-table run under {out_dir}"
        )
    return run_dir
