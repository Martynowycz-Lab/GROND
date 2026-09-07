"""Feed a shared model through each crystal and recompute rows after the final model update."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.io.config import PrepConfig
from grond.macromolecule import sha256_file
from grond.workflows.dataset_gates import policy_active_runs, require_policy_active_runs
from grond.workflows.dataset_policy import write_policy_weighted_runs
from grond.workflows.decision import safe_float, write_json
from grond.workflows.model_selection import (
    dataset_baseline_hkl,
    shelx_model_is_usable,
    shelxl_feedback_record_safe_for_reuse,
)
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import resolve_final_n_basis
from grond.workflows.report_commands import combine_cmd
from grond.workflows.scale_materialization import materialize_dataset_scale_contract


def multi_combined_endpoint_model(combined_dir: Path) -> Path | None:
    """Return the exact summary-selected shared SHELXL model.

    Endpoint directories are durable resume artifacts.  Selecting by glob can
    therefore pair current health/metrics with an older lambda's RES file.
    Require the current summary's lambda, artifact path, health path, and hash
    to identify the same bytes before allowing feedback reuse.
    """

    summary = try_read_json(combined_dir / "combined_summary.json")
    endpoint = summary.get("combined_endpoint")
    endpoint = endpoint if isinstance(endpoint, Mapping) else {}
    try:
        lam = float(endpoint["lambda_q"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(lam):
        return None
    lam_tag = str(lam).replace(".", "p")
    expected = combined_dir / f"combined_endpoint_lam_{lam_tag}" / "multi_dyn_corrected.res"
    artifacts = endpoint.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    recorded_res = str(artifacts.get("res") or "")
    if not recorded_res:
        return None
    candidate = Path(recorded_res).expanduser()
    try:
        candidate.resolve().relative_to(combined_dir.resolve())
    except (OSError, ValueError):
        return None
    if candidate.resolve() != expected.resolve():
        return None

    shelxl = endpoint.get("shelxl")
    shelxl = shelxl if isinstance(shelxl, Mapping) else {}
    if not shelx_model_is_usable(candidate):
        return None
    health = shelxl.get("res_health")
    health = health if isinstance(health, Mapping) else {}
    health_path = str(health.get("path") or "")
    health_sha = str(health.get("sha256") or "")
    artifact_sha = str(artifacts.get("res_sha256") or "")
    try:
        health_path_matches = bool(health_path) and Path(health_path).resolve() == candidate.resolve()
        current_sha = sha256_file(candidate)
    except OSError:
        return None
    return (
        candidate
        if health_path_matches and artifact_sha and health_sha and artifact_sha == health_sha == current_sha
        else None
    )


def multi_combined_endpoint_shelxl(combined_dir: Path) -> dict[str, object]:
    """Return the recorded SHELXL result for a combined multi-dataset endpoint."""

    summary = try_read_json(combined_dir / "combined_summary.json")
    endpoint = summary.get("combined_endpoint")
    endpoint = endpoint if isinstance(endpoint, Mapping) else {}
    shelxl = endpoint.get("shelxl")
    return dict(shelxl) if isinstance(shelxl, Mapping) else {}


def run_multi_final_row_recompute(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    model: str,
    nuisance: dict[str, dict],
    structure_dataset_policy: dict,
) -> dict[str, object]:
    """Recompute final-model rows without taking an additional LS1 step.

    A feedback cycle's native rows describe its input model, while the shared
    SHELXL endpoint is its output model.  Publication row statistics and
    dynamical maps must never pair those two different structural states. This
    final fixed-nuisance pass produces exactly the rows of the accepted output
    model and deliberately writes no new corrected-HKL endpoint.
    """

    final_root = root / "runs" / "shared_shelxl_feedback" / "final_row_recompute"
    raw_runs: dict[str, Path] = {}
    scale_contract_runs: dict[str, Path] = {}
    for ds in prep.datasets:
        run_dir = final_root / ds.name
        n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, model, "final_row_recompute")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                model,
                dataset_baseline_hkl(prep, ds),
                run_dir,
                args,
                n_basis=n_basis,
                n_points=args.n_points,
                stage="final_row_recompute",
                pinned=nuisance[ds.name],
                make_maps=False,
                endpoint=False,
                strict_pinned_contract=True,
            ),
            logs,
            f"{ds.name}_final_row_recompute",
            dry_run=False,
        )
        raw_runs[ds.name] = run_dir
        scale_contract_runs[ds.name] = materialize_dataset_scale_contract(
            python,
            run_dir,
            args,
            ds.name,
            logs,
            label="final_row_recompute",
        )
    if structure_dataset_policy.get("status") in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
        active_runs = scale_contract_runs
        weighted_runs = scale_contract_runs
    else:
        weighted_runs = write_policy_weighted_runs(
            scale_contract_runs,
            structure_dataset_policy,
            final_root / "weighted_runs",
            include_endpoint_only=False,
        )
        active_runs = policy_active_runs(weighted_runs, structure_dataset_policy)
        require_policy_active_runs(active_runs, context="final-model row recompute")
    return {
        "status": "ok",
        "model": str(model),
        "definition": "fixed-nuisance Bloch rows for the accepted shared SHELXL output model; no additional LS1 step",
        "raw_runs": {name: str(path) for name, path in raw_runs.items()},
        "runs": {name: str(path) for name, path in scale_contract_runs.items()},
        "weighted_runs": {name: str(path) for name, path in weighted_runs.items()},
        "active_runs": {name: str(path) for name, path in active_runs.items()},
    }


def run_multi_shelxl_feedback_loop(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    start_model: str,
    nuisance: dict[str, dict],
    structure_dataset_policy: dict,
) -> tuple[str, dict]:
    """Run shared-model Bloch -> merged HKL -> SHELXL LS1 macrocycles."""

    feedback_root = root / "runs" / "shared_shelxl_feedback"
    cycles_requested = max(0, int(getattr(args, "shelxl_feedback_cycles", 0) or 0))
    recompute_final_rows = bool(getattr(args, "recompute_final_rows", False))
    row_contracts = getattr(args, "dataset_row_contracts", {})
    row_contracts = row_contracts if isinstance(row_contracts, Mapping) else {}
    row_contract_provenance = {
        name: {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": sha256_file(path) if Path(path).expanduser().is_file() else "",
        }
        for name, path in row_contracts.items()
    }
    scale_specs = getattr(args, "dataset_scale_specs", {})
    scale_specs = scale_specs if isinstance(scale_specs, Mapping) else {}
    summary: dict = {
        "status": "not_run" if cycles_requested <= 0 else "running",
        "strategy": "shelxl_feedback",
        "start_model": start_model,
        "current_model": start_model,
        "cycles_requested": cycles_requested,
        "shelxl_iterations_per_cycle": int(getattr(args, "shelxl_feedback_shelxl_iterations", 1) or 1),
        "dataset_row_contracts": row_contract_provenance,
        "dataset_scale_specs": dict(scale_specs),
        "cycles": [],
    }
    if cycles_requested <= 0:
        summary["status"] = "disabled_zero_cycles"
        if recompute_final_rows and not args.dry_run:
            feedback_root.mkdir(parents=True, exist_ok=True)
            try:
                summary["final_row_recompute"] = run_multi_final_row_recompute(
                    prep,
                    args,
                    python,
                    logs,
                    root,
                    model=start_model,
                    nuisance=nuisance,
                    structure_dataset_policy=structure_dataset_policy,
                )
            except Exception as exc:
                summary["status"] = "failed_final_row_recompute"
                summary["final_row_recompute"] = {
                    "status": "failed",
                    "model": str(start_model),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
                raise
            summary["status"] = "ok_recomputed_zero_cycles"
        write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
        return start_model, summary
    if args.dry_run:
        summary["status"] = "dry_run"
        write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
        return start_model, summary

    current_model = start_model
    feedback_root.mkdir(parents=True, exist_ok=True)
    for cycle in range(1, cycles_requested + 1):
        cycle_label = f"multi_ls1_cycle_{cycle:02d}"
        cycle_dir = feedback_root / f"cycle_{cycle:02d}"
        raw_cycle_runs: dict[str, Path] = {}
        cycle_runs: dict[str, Path] = {}
        nuisance_revalidations: list[dict[str, object]] = []
        for ds in prep.datasets:
            run_dir = cycle_dir / ds.name
            final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, current_model, cycle_label)
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    current_model,
                    dataset_baseline_hkl(prep, ds),
                    run_dir,
                    args,
                    n_basis=final_n_basis,
                    n_points=args.n_points,
                    stage=cycle_label,
                    pinned=nuisance[ds.name],
                    make_maps=False,
                    endpoint=False,
                    strict_pinned_contract=True,
                ),
                logs,
                f"{ds.name}_{cycle_label}",
                dry_run=False,
            )
            raw_cycle_runs[ds.name] = run_dir
            cycle_runs[ds.name] = materialize_dataset_scale_contract(
                python,
                run_dir,
                args,
                ds.name,
                logs,
                label=cycle_label,
            )
            selected_nuisance = (try_read_json(run_dir / "summary.json").get("best_native") or {}).get("nuisance") or {}
            requested_mode = str((nuisance.get(ds.name) or {}).get("sg_jitter_mode", "none"))
            selected_mode = str(selected_nuisance.get("sg_jitter_mode", requested_mode))
            if selected_mode != requested_mode:
                nuisance_revalidations.append(
                    {
                        "dataset": ds.name,
                        "field": "sg_jitter_mode",
                        "checkpoint_value": requested_mode,
                        "selected_value": selected_mode,
                        "reason": "pinned Sg-jitter candidate was revalidated alongside the explicit no-jitter fallback",
                        "run": str(run_dir),
                    }
                )

        if structure_dataset_policy.get("status") in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
            cycle_input_runs = cycle_runs
        else:
            weighted_runs = write_policy_weighted_runs(
                cycle_runs,
                structure_dataset_policy,
                cycle_dir / "weighted_runs",
                include_endpoint_only=False,
            )
            cycle_input_runs = policy_active_runs(weighted_runs, structure_dataset_policy)
            require_policy_active_runs(cycle_input_runs, context=f"structure-feedback cycle {cycle}")

        combined_dir = cycle_dir / "combined"
        run_logged(
            combine_cmd(
                python,
                cycle_input_runs,
                current_model,
                combined_dir,
                args,
                shelxl_iterations=int(getattr(args, "shelxl_feedback_shelxl_iterations", 1) or 1),
            ),
            logs,
            f"{cycle_label}_combine",
            dry_run=False,
        )
        output_model = multi_combined_endpoint_model(combined_dir)
        endpoint_shelxl = multi_combined_endpoint_shelxl(combined_dir)
        combined_summary = try_read_json(combined_dir / "combined_summary.json")
        combined_endpoint = (
            combined_summary.get("combined_endpoint")
            if isinstance(combined_summary.get("combined_endpoint"), Mapping)
            else {}
        )
        endpoint_modulation_p95 = safe_float(
            combined_endpoint.get("mod_p95", combined_endpoint.get("modulation_p95")),
            float("nan"),
        )
        max_modulation_p95 = float(getattr(args, "publication_endpoint_max_modulation_p95", 20.0))
        modulation_warning = (
            "missing finite modulation p95"
            if not math.isfinite(endpoint_modulation_p95)
            else f"modulation p95 {endpoint_modulation_p95:.4g} exceeds warning level {max_modulation_p95:.4g}"
            if endpoint_modulation_p95 > max_modulation_p95
            else ""
        )
        endpoint_health = endpoint_shelxl.get("res_health")
        endpoint_health = dict(endpoint_health) if isinstance(endpoint_health, Mapping) else {}
        record = {
            "cycle": cycle,
            "label": cycle_label,
            "input_model": current_model,
            "output_model": str(output_model) if output_model else "",
            "endpoint_shelxl_status": str(endpoint_shelxl.get("status", "")),
            "endpoint_res_health": endpoint_health,
            "endpoint_modulation_p95": endpoint_modulation_p95,
            "endpoint_modulation_metric": str(combined_endpoint.get("modulation_metric") or ""),
            "endpoint_modulation_gate": {
                "passed": True,
                "blocking": False,
                "maximum_p95": max_modulation_p95,
                "status": "warning" if modulation_warning else "ok",
                "reason": modulation_warning or "within warning level",
            },
            "combined": str(combined_dir),
            "raw_runs": {name: str(path) for name, path in raw_cycle_runs.items()},
            "runs": {name: str(path) for name, path in cycle_runs.items()},
            "combined_input_runs": {name: str(path) for name, path in cycle_input_runs.items()},
            "nuisance_revalidations": nuisance_revalidations,
        }
        summary["cycles"].append(record)
        if output_model is None:
            summary["status"] = "failed_no_usable_endpoint_model"
            write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
            raise RuntimeError(f"{cycle_label} did not produce a usable shared SHELXL .res model in {combined_dir}")
        if not shelxl_feedback_record_safe_for_reuse(record):
            record["adopted_for_feedback"] = False
            endpoint_status = str(record.get("endpoint_shelxl_status", "")).strip().lower()
            try:
                endpoint_npd_count = int(endpoint_health.get("npd_count", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                endpoint_npd_count = 0
            try:
                endpoint_split_count = int(endpoint_health.get("split_count", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                endpoint_split_count = 0
            has_adp_pathology = bool(
                endpoint_status == "adp_pathology" or endpoint_health.get("has_npd") is True or endpoint_npd_count > 0
            )
            if has_adp_pathology:
                summary["status"] = "stopped_adp_pathology"
                summary["stopped_reason"] = (
                    "Shared SHELXL feedback cycle produced ADP pathology; keeping the previous clean model. "
                    f"NPD atoms={endpoint_npd_count}, split warnings={endpoint_split_count}."
                )
                summary["pathological_model"] = str(output_model)
            else:
                reason = str(
                    endpoint_shelxl.get("invalid_res_reason") or endpoint_health.get("reason") or "unsafe endpoint"
                )
                summary["status"] = "stopped_unsafe_endpoint"
                summary["stopped_reason"] = (
                    "Shared SHELXL feedback cycle produced an unsafe endpoint; keeping the previous clean model. "
                    f"SHELXL status={endpoint_status or 'unknown'}, reason={reason}."
                )
                summary["unsafe_model"] = str(output_model)
            write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
            break
        record["adopted_for_feedback"] = True
        current_model = str(output_model)
        summary["current_model"] = current_model
        summary["status"] = "ok"
        write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)

    if summary.get("status") == "ok":
        try:
            summary["final_row_recompute"] = run_multi_final_row_recompute(
                prep,
                args,
                python,
                logs,
                root,
                model=current_model,
                nuisance=nuisance,
                structure_dataset_policy=structure_dataset_policy,
            )
        except Exception as exc:
            summary["status"] = "failed_final_row_recompute"
            summary["final_row_recompute"] = {
                "status": "failed",
                "model": str(current_model),
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
            raise
        write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)

    return current_model, summary
