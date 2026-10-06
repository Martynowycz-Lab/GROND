"""Record enough model, row and scale information to continue a completed run."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.workflows.model_selection import best_nuisance


def _checkpoint_path(raw: object, *, base: Path | None = None) -> str:
    """Return an absolute path string for a checkpoint field."""

    if raw in (None, ""):
        return ""
    path = Path(str(raw)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return str(path.resolve())


def _summary_section(summary: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return ``summary[key]`` when it is a mapping, else an empty dict."""

    value = summary.get(key)
    return value if isinstance(value, Mapping) else {}


def _checkpoint_dataset_contract(
    ds: DatasetSpec, summary_path: Path | None
) -> tuple[dict[str, object], dict[str, object]]:
    """Return checkpoint dataset fields, preferring the final row-contract summary."""

    row: dict[str, object] = asdict(ds)
    final_config: dict[str, object] = {}
    if summary_path is not None and summary_path.exists():
        summary = try_read_json(summary_path)
        config = _summary_section(summary, "config")
        inputs = _summary_section(summary, "inputs")
        hkl_transform = str(config.get("hkl_transform") or inputs.get("hkl_transform") or "").strip()
        if hkl_transform:
            row["hkl_transform"] = hkl_transform
        for key in (
            "hkl_transform",
            "n_basis",
            "n_points",
            "thickness_points",
            "max_support_sg",
            "sg_window",
            "sg_window_mode",
            "sg_window_min",
            "sg_window_max",
            "sg_window_scale",
            "sg_window_frame_floor",
            "profile_quadrature",
            "profile_accumulation",
            "topup_d_min",
            "compute_backend",
            "row_plan_cache",
            "chunksize",
            "post_bloch_sigma_policy",
            "min_d_spacing",
            "min_peak",
            "geometry_percentile",
        ):
            if key in config and config[key] not in (None, ""):
                final_config[key] = config[key]
    return row, final_config


def write_continue_checkpoint(prep: PrepConfig, result: Mapping[str, object], args: argparse.Namespace) -> dict:
    """Write the small JSON file used to continue a finished endpoint."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    checkpoint_path = out_dir / "grond_continue_checkpoint.json"
    dataset_count = len(prep.datasets)
    dataset_name = prep.datasets[0].name if prep.datasets else "dataset"
    runs = result.get("runs") if isinstance(result.get("runs"), Mapping) else {}
    source_run = _checkpoint_path(result.get("primary_run"))
    source_summary = _checkpoint_path(Path(source_run) / "summary.json") if source_run else ""
    recommended_start_res = _checkpoint_path(result.get("model"))
    baseline_hkl = _checkpoint_path(prep.baseline_hkl) if prep.baseline_hkl else ""
    hkl = _checkpoint_path(prep.datasets[0].hkl) if prep.datasets else ""
    hkl_transform = prep.datasets[0].hkl_transform if prep.datasets else ""
    source_summary_exists = bool(source_summary and Path(source_summary).exists())
    start_model_exists = bool(recommended_start_res and Path(recommended_start_res).exists())
    datasets: list[dict[str, object]] = []
    for ds in prep.datasets:
        run_text = str(runs.get(ds.name, "")) if runs else (source_run if dataset_count == 1 else "")
        run_path = Path(run_text).expanduser().resolve() if run_text else None
        summary_path = run_path / "summary.json" if run_path is not None else None
        dataset_contract, final_config = _checkpoint_dataset_contract(ds, summary_path)
        datasets.append(
            {
                **dataset_contract,
                "source_run": str(run_path) if run_path is not None else "",
                "source_summary": str(summary_path) if summary_path is not None else "",
                "nuisance": best_nuisance(run_path) if run_path is not None else {},
                "final_config": final_config,
            }
        )
    dataset_summaries_exist = (
        all(bool(row.get("source_summary") and Path(str(row["source_summary"])).exists()) for row in datasets)
        if datasets
        else False
    )
    ready = bool(
        start_model_exists
        and not getattr(args, "dry_run", False)
        and ((dataset_count == 1 and source_summary_exists) or (dataset_count > 1 and dataset_summaries_exist))
    )
    checkpoint = {
        "schema_version": 1,
        "kind": "grond_continue_checkpoint",
        "status": "ready" if ready else "not_ready",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(out_dir),
        "dataset_count": dataset_count,
        "dataset": dataset_name,
        "source_run": source_run,
        "source_summary": source_summary,
        "recommended_start_res": recommended_start_res,
        "hkl": hkl,
        "baseline_hkl": baseline_hkl,
        "hkl_transform": hkl_transform,
        "datasets": datasets,
        "structure_dataset_policy": ((result.get("dataset_policy") or {}).get("structure_update") or {}),
        "final_dataset_policy": ((result.get("dataset_policy") or {}).get("final") or {}),
        "scale_contract_selection": result.get("scale_contract_selection") or {},
        "multi_endpoint_merge_key": getattr(args, "multi_endpoint_merge_key", "model_symmetry"),
        "recommended_command": [
            "GROND",
            "push-endpoint",
            "--checkpoint",
            str(checkpoint_path),
            "--start-res",
            "edited_model.res",
            "--out-dir",
            str(out_dir / "endpoint_push_from_edited_model"),
            "--cycles",
            "3",
        ],
        "notes": [
            "Use this for deliberate continuation from a changed model while reusing the pinned GROND row/nuisance contract.",
            "For multi-dataset checkpoints, GROND recomputes each dataset from the edited shared model, merges corrected endpoint observations, then runs one shared SHELXL LS step per macrocycle.",
            "Checkpoint dataset source runs retain the full physical native calculations; any selected compact scale contract is endpoint provenance only. Multi-dataset push refits the pinned full grouping and does not automatically rerun compact scale selection, so a changed-model publication endpoint needs a separate post-push compact-scale selection.",
            "For an interrupted GROND auto run, rerun the original command with the same --out-dir instead.",
        ],
    }
    if dataset_count == 1 and not source_summary:
        checkpoint["status_reason"] = "no final source summary was recorded"
    elif dataset_count > 1 and not datasets:
        checkpoint["status_reason"] = "no datasets were recorded"
    elif dataset_count > 1 and not dataset_summaries_exist:
        checkpoint["status_reason"] = "one or more recorded dataset final summaries do not exist yet"
    elif not recommended_start_res:
        checkpoint["status_reason"] = "no recommended start model was recorded"
    elif dataset_count == 1 and not source_summary_exists:
        checkpoint["status_reason"] = "recorded final source summary does not exist yet"
    elif not start_model_exists:
        checkpoint["status_reason"] = "recorded start model does not exist yet"
    elif getattr(args, "dry_run", False):
        checkpoint["status_reason"] = "dry-run checkpoints are plans, not usable continuation checkpoints"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
    return {"path": str(checkpoint_path), **checkpoint}
