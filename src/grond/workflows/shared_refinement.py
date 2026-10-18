"""Feed a shared model through each crystal and recompute rows after the final model update."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.hashing import sha256_file
from grond.io.config import PrepConfig
from grond.refine.feedback_trajectory import (
    COMPLETED_LOOP_STATUSES,
    KEEP_POLICIES,
    completed_loop_status,
    row_r_by_split,
    select_kept_cycle,
    trajectory_entry,
)
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.dataset_gates import policy_active_runs, require_policy_active_runs
from grond.workflows.dataset_policy import write_policy_weighted_runs
from grond.workflows.decision import safe_float, write_json
from grond.workflows.model_selection import (
    dataset_baseline_hkl,
    shelx_model_is_usable,
    shelxl_feedback_record_safe_for_reuse,
)
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nested_nuisance import MultiFeedbackThickness, feedback_thickness_update_enabled
from grond.workflows.nuisance_policy import resolve_final_n_basis, resolved_feedback_wght_mode
from grond.workflows.report_commands import combine_cmd


def pinned_physics_drift(pinned: Mapping[str, object], selected: Mapping[str, object]) -> list[str]:
    """Cone and dose terms the feedback cycle must carry unchanged from the pinned nuisance (fail closed)."""

    drift = []
    for key in ("mosaicity_mode", "dose_b_mode"):
        if key in pinned and str(selected.get(key, "none")) != str(pinned.get(key)):
            drift.append(f"{key} {pinned.get(key)!r} -> {selected.get(key)!r}")
    for key in ("mosaicity_sigma_deg", "dose_b_per_frame"):
        if key in pinned:
            try:
                same = abs(float(selected.get(key, 0.0)) - float(pinned.get(key) or 0.0)) <= 1e-12 * max(1.0, abs(float(pinned.get(key) or 0.0)))
            except (TypeError, ValueError):
                same = False
            if not same:
                drift.append(f"{key} {pinned.get(key)!r} -> {selected.get(key)!r}")
    return drift

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
    for ds in prep.datasets:
        run_dir = final_root / ds.name
        per_crystal = crystal_arguments(args, ds.name)
        n_basis = resolve_final_n_basis(prep, ds, per_crystal, python, logs, root, model, "final_row_recompute")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                model,
                dataset_baseline_hkl(prep, ds),
                run_dir,
                per_crystal,
                n_basis=n_basis,
                n_points=per_crystal.n_points,
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
    if structure_dataset_policy.get("status") in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
        active_runs = raw_runs
        weighted_runs = raw_runs
    else:
        weighted_runs = write_policy_weighted_runs(
            raw_runs,
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
        "runs": {name: str(path) for name, path in raw_runs.items()},
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
    """Run shared-model Bloch -> merged HKL -> SHELXL macro-cycles as a monitored fixed-point iteration.

    Each macro-cycle is the lagged map p_{k+1} = T(p_k): Bloch rows of the entering model p_k give
    kinematically equivalent intensities with the dynamical ratio frozen at p_k, and SHELXL solves the
    kinematic least squares against them.  Its stationary point is not the dynamical least-squares
    solution (the gradient term through d ln q/dp is dropped), so the loop can contract towards a biased,
    even unphysical point.  The legacy single-step settings make that worse (one L.S. step per cycle, DAMP
    carried forward, free rows and all-row per-frame scales inside the refinement, q clipping, an arbitrary
    per-cycle intensity scale): such a loop can drift to an NPD ADP while the entering-model row R rises.

    The loop therefore records a per-cycle ``trajectory`` (held-out and work row R of the entering model,
    converged SHELXL statistics, max coordinate shift, NPD/ISOR/DAMP state, excluded q rows), stops when
    the max non-H shift between successive models is below ``--shelxl-feedback-xyz-tol`` (stop_reason
    ``converged``) or at ``--shelxl-feedback-cycles`` (``cap``; status ``ok`` under the legacy rule of
    tolerance 0 and keep last, otherwise the stop reason), still stops on ADP pathology, and keeps the
    last adopted output model (``--shelxl-feedback-keep last``, the default) or, with ``best-free``, the
    output with the lowest free-row R_obs, reported as ``kept_cycle``.  The output of cycle k is scored by the rows of cycle k+1,
    and the last output by the final fixed-nuisance row recompute, all with the same N/Q and nuisance.
    See grond.refine.feedback_trajectory for the derivation.

    With the feedback thickness update (grond.workflows.nested_nuisance) every crystal's thickness distribution
    and lambda are re-selected on the spectra of each new structure (the nested variant fixed) before the cycle
    that corrects with it, and once for the final model; the trajectory records each cycle's thickness and
    ``thickness_update.final`` the thickness of the kept model, which the later stages pin.
    """

    feedback_root = root / "runs" / "shared_shelxl_feedback"
    cycles_requested = max(0, int(getattr(args, "shelxl_feedback_cycles", 0) or 0))
    xyz_tol = float(getattr(args, "shelxl_feedback_xyz_tol", 0.0) or 0.0)
    keep_policy = str(getattr(args, "shelxl_feedback_keep", "last") or "last")
    if keep_policy not in KEEP_POLICIES:
        raise ValueError(f"unknown --shelxl-feedback-keep {keep_policy!r}; use one of {', '.join(KEEP_POLICIES)}")
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
    summary: dict = {
        "status": "not_run" if cycles_requested <= 0 else "running",
        "strategy": "shelxl_feedback",
        "start_model": start_model,
        "current_model": start_model,
        "cycles_requested": cycles_requested,
        "shelxl_iterations_per_cycle": int(getattr(args, "shelxl_feedback_shelxl_iterations", 1) or 1),
        "dataset_row_contracts": row_contract_provenance,
        "crystal_parameters": getattr(args, "crystal_parameters", {}),
        "handoff": {
            "shelxl_convergence_shift_esd": float(getattr(args, "shelxl_convergence_shift_esd", 0.0) or 0.0),
            "endpoint_ls_cycles": int(getattr(args, "shelxl_feedback_endpoint_ls_cycles", 1) or 1),
            "shelxl_npd_recovery": str(getattr(args, "shelxl_npd_recovery", "damp")),
            "carry_damp": bool(getattr(args, "shelxl_carry_damp", False)),
            "endpoint_include_free_rows": bool(getattr(args, "endpoint_include_free_rows", True)),
            "endpoint_q_policy": str(getattr(args, "endpoint_q_policy", "clip")),
            "endpoint_intensity_scale": str(getattr(args, "endpoint_intensity_scale", "median-iobs")),
            "endpoint_wght_mode": str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
            # The weighting rule the macro-cycles actually used (--shelxl-feedback-wght-mode).
            "feedback_wght_mode": resolved_feedback_wght_mode(args),
            "dataset_sigma_policy": str(getattr(args, "dataset_sigma_policy", "auto")),
        },
        "xyz_tol_A": xyz_tol,
        "keep_policy": keep_policy,
        "kept_cycle": 0,
        "cycles": [],
        "trajectory": [],
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
    thickness = (
        MultiFeedbackThickness(prep, root, logs, python, int(getattr(args, "workers", 1) or 1))
        if feedback_thickness_update_enabled(args)
        else None
    )
    # Pinned nuisance of the structure being corrected; with the thickness update each crystal's thickness follows
    # the structure (nuisance holds the nested pick for the start model).
    cycle_nuisance = {name: dict(value) for name, value in nuisance.items()}
    previous_runs: dict[str, Path] = {}

    def thickness_for(model: str, label: str) -> dict[str, dict]:
        """Pin each crystal's thickness selected for ``model`` into the cycle nuisance; return the records."""
        records = {}
        for ds in prep.datasets:
            cycle_nuisance[ds.name], records[ds.name] = thickness.pinned(
                ds.name, nuisance[ds.name], model, previous_runs.get(ds.name), label
            )
        return records

    def record_thickness_update() -> None:
        """Store the thickness-update trackers and the current model's picks in the summary (when enabled)."""
        if thickness is not None:
            summary["thickness_update"] = {"enabled": True, "crystals": thickness.summary(), "final": thickness.final(current_model)}
    # Adopted macro-cycle outputs in order; "rows" is the row-R record that scores each output model.
    adopted: list[dict] = []
    stopped = False
    stop_status = "cap"
    for cycle in range(1, cycles_requested + 1):
        cycle_label = f"multi_ls1_cycle_{cycle:02d}"
        cycle_dir = feedback_root / f"cycle_{cycle:02d}"
        raw_cycle_runs: dict[str, Path] = {}
        cycle_runs: dict[str, Path] = {}
        nuisance_revalidations: list[dict[str, object]] = []
        cycle_thickness = None
        if thickness is not None:
            try:
                cycle_thickness = thickness_for(current_model, f"cycle_{cycle:02d}")
            except Exception as exc:
                summary["status"] = "failed_thickness_update"
                summary["stopped_reason"] = f"thickness re-selection before cycle {cycle}: {type(exc).__name__}: {exc}"
                record_thickness_update()
                write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
                raise
        for ds in prep.datasets:
            run_dir = cycle_dir / ds.name
            per_crystal = crystal_arguments(args, ds.name)
            final_n_basis = resolve_final_n_basis(prep, ds, per_crystal, python, logs, root, current_model, cycle_label)
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    current_model,
                    dataset_baseline_hkl(prep, ds),
                    run_dir,
                    per_crystal,
                    n_basis=final_n_basis,
                    n_points=per_crystal.n_points,
                    stage=cycle_label,
                    pinned=cycle_nuisance[ds.name],
                    make_maps=False,
                    endpoint=False,
                    strict_pinned_contract=True,
                ),
                logs,
                f"{ds.name}_{cycle_label}",
                dry_run=False,
            )
            raw_cycle_runs[ds.name] = run_dir
            previous_runs[ds.name] = run_dir
            cycle_runs[ds.name] = run_dir
            selected_nuisance = (try_read_json(run_dir / "summary.json").get("best_native") or {}).get("nuisance") or {}
            if not args.dry_run:
                drifted = pinned_physics_drift(cycle_nuisance.get(ds.name) or {}, selected_nuisance)
                if thickness is not None:
                    drifted += thickness.drift(ds.name, current_model, selected_nuisance)
                if drifted:
                    raise RuntimeError(
                        f"{ds.name} feedback cycle {cycle_label} did not reproduce the pinned physics: " + "; ".join(drifted)
                    )
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
            weighted_runs = cycle_runs
        else:
            weighted_runs = write_policy_weighted_runs(
                cycle_runs,
                structure_dataset_policy,
                cycle_dir / "weighted_runs",
                include_endpoint_only=False,
            )
            cycle_input_runs = policy_active_runs(weighted_runs, structure_dataset_policy)
            require_policy_active_runs(cycle_input_runs, context=f"structure-feedback cycle {cycle}")
        # Rows of the model entering this cycle: they score the previous cycle's output model.
        entering_rows = row_r_by_split(cycle_input_runs.values())
        if adopted:
            adopted[-1]["rows"] = {**entering_rows, "source": f"{cycle_label} entering-model rows"}

        combined_dir = cycle_dir / "combined"
        run_logged(
            combine_cmd(
                python,
                cycle_input_runs,
                current_model,
                combined_dir,
                args,
                shelxl_iterations=int(getattr(args, "shelxl_feedback_shelxl_iterations", 1) or 1),
                feedback_cycle=True,
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
            "weighted_runs": {name: str(path) for name, path in weighted_runs.items()},
            "combined_input_runs": {name: str(path) for name, path in cycle_input_runs.items()},
            "nuisance_revalidations": nuisance_revalidations,
        }
        trajectory = trajectory_entry(
            cycle,
            entering_rows,
            endpoint_shelxl,
            input_model=current_model,
            output_model=str(output_model) if output_model else "",
            n_excluded_q_rows=combined_endpoint.get("n_excluded_q_rows"),
        )
        if cycle_thickness is not None:
            record["thickness"] = cycle_thickness
            trajectory["thickness"] = cycle_thickness
        record["max_xyz_shift_A"] = trajectory["max_xyz_shift_A"]
        summary["cycles"].append(record)
        summary["trajectory"].append(trajectory)
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
            trajectory["adopted_for_feedback"] = False
            write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
            stopped = True
            break
        record["adopted_for_feedback"] = True
        trajectory["adopted_for_feedback"] = True
        current_model = str(output_model)
        adopted.append({"cycle": cycle, "model": current_model, "rows": None})
        summary["current_model"] = current_model
        summary["status"] = "running"
        write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
        shift = trajectory["max_xyz_shift_A"]
        if xyz_tol > 0.0 and shift is not None and shift < xyz_tol:
            stop_status = "converged"
            summary["convergence_reason"] = f"max non-H coordinate shift {shift:.5f} A < {xyz_tol:g} A"
            break
    if not stopped:
        summary["stop_reason"] = stop_status
        summary["status"] = completed_loop_status(stop_status, xyz_tol=xyz_tol, keep_policy=keep_policy)

    if summary.get("status") in COMPLETED_LOOP_STATUSES:
        if thickness is not None:
            try:
                thickness_for(current_model, "final")
            except Exception as exc:
                summary["status"] = "failed_thickness_update"
                summary["stopped_reason"] = f"thickness re-selection for the final model: {type(exc).__name__}: {exc}"
                record_thickness_update()
                write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
                raise
        try:
            summary["final_row_recompute"] = run_multi_final_row_recompute(
                prep,
                args,
                python,
                logs,
                root,
                model=current_model,
                nuisance=cycle_nuisance,
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
        if adopted:
            adopted[-1]["rows"] = {
                **row_r_by_split(Path(path) for path in summary["final_row_recompute"]["active_runs"].values()),
                "source": "final_row_recompute",
            }

    try:
        current_model = _keep_macro_cycle_model(summary, adopted, keep_policy, current_model)
    except RuntimeError as exc:
        # Refuse to guess, but leave the trajectory and the reason on disk first.
        summary["status"] = "failed_keep_best" if summary.get("status") in COMPLETED_LOOP_STATUSES else summary["status"]
        summary["kept_reason"] = str(exc)
        record_thickness_update()
        write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
        raise
    # The kept model's thickness: selected before the cycle it entered, or for the final model after the loop.
    record_thickness_update()
    write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
    return current_model, summary


def _keep_macro_cycle_model(summary: dict, adopted: list[dict], keep_policy: str, current_model: str) -> str:
    """Record ``kept_cycle`` and bind the final rows to the kept model; returns the kept model.

    A kept model other than the last adopted one reuses the entering-model rows of the next cycle, which
    are the same fixed-nuisance final-N/Q rows of exactly that model, so no extra Bloch pass is needed.
    """

    summary["adopted_cycle_scores"] = [
        {"cycle": item["cycle"], "free_r_obs": (item.get("rows") or {}).get("free_r_obs"),
         "work_r_obs": (item.get("rows") or {}).get("work_r_obs"), "source": (item.get("rows") or {}).get("source")}
        for item in adopted
    ]
    kept_cycle, reason = select_kept_cycle(
        [(item["cycle"], (item.get("rows") or {}).get("free_r_obs")) for item in adopted], keep_policy
    )
    summary["kept_cycle"] = kept_cycle
    summary["kept_reason"] = reason
    if kept_cycle <= 0 or keep_policy == "last":
        return current_model
    kept = next(item for item in adopted if item["cycle"] == kept_cycle)
    if kept is adopted[-1] and summary.get("status") in COMPLETED_LOOP_STATUSES:
        return current_model
    next_record = next(record for record in summary["cycles"] if int(record.get("cycle", 0)) == kept_cycle + 1)
    if "final_row_recompute" in summary:
        summary["last_model_row_recompute"] = summary.pop("final_row_recompute")
    summary["final_row_recompute"] = {
        "status": "ok",
        "model": kept["model"],
        "definition": (
            f"fixed-nuisance Bloch rows of the kept macro-cycle {kept_cycle} output model, computed as the "
            f"entering-model rows of cycle {kept_cycle + 1}; no additional LS step"
        ),
        "source_cycle": kept_cycle + 1,
        "raw_runs": dict(next_record.get("raw_runs") or {}),
        "runs": dict(next_record.get("runs") or {}),
        "weighted_runs": dict(next_record.get("weighted_runs") or next_record.get("combined_input_runs") or {}),
        "active_runs": dict(next_record.get("combined_input_runs") or {}),
    }
    summary["current_model"] = kept["model"]
    return kept["model"]
