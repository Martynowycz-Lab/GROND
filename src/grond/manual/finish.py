"""Finish manual refinement with hash-bound endpoint statistics and optional maps."""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping
from pathlib import Path

from grond.engine.shelx_diagnostics import parse_shelx_lst_metrics, parse_shelx_lst_pathologies
from grond.io.config import load_prep
from grond.manual.execution import _command_elapsed
from grond.manual.reporting import (
    _manual_markdown_summary,
    _manual_row_statistics,
    _manual_scale_summary,
    _percent,
    _selected_hand_summary,
)
from grond.manual.state import (
    _load_workflow,
    _print_handoff,
    _read_json,
    _record_stage,
    _require_stage,
    _sha256,
    _stage,
    _write_json,
)
from grond.maps.model_box import DEFAULT_MODEL_BOX_MARGIN_A, native_package_entries, write_model_box_maps
from grond.maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package


def manual_finish(args) -> int:
    """Run the ``finish`` stage: endpoint and row statistics, optional native maps, and the final summaries.

    The starting conventional endpoint counts as comparable only when its model and HKL match. Writes
    manual_final_summary.json and MANUAL_FINAL_SUMMARY.md, copied to the workflow root as MANUAL_TOP_LEVEL_SUMMARY.*.
    """

    workflow_path, workflow = _load_workflow(args)
    feedback = _require_stage(workflow, "feedback")
    root = Path(workflow["root"])
    out = root / "06_finish"
    out.mkdir(parents=True, exist_ok=True)
    model = Path(str(feedback["outputs"]["selected_model"])).resolve()
    row_runs = {name: Path(path) for name, path in feedback["outputs"]["selected_row_runs"].items()}
    single_dataset_endpoint = len(row_runs) == 1
    maps: dict[str, object]
    started = time.perf_counter()
    if args.skip or args.map_mode == "off":
        maps = {"status": "skipped"}
    else:
        maps = write_native_dynamic_map_package(
            row_table=[run / "native_dyn_calc.csv" for run in row_runs.values()],
            row_table_labels=list(row_runs),
            model_path=model,
            out_dir=out / "native_dynamic_maps",
            options=NativeDynamicMapOptions(
                grid_spacing_A=float(args.native_map_spacing),
                d_min_A=None,
                min_i_over_sigma=0.0,
                max_grid_points=8_000_000,
            ),
        )
        if maps.get("status") == "ok" and getattr(args, "model_box_maps", True):
            try:
                maps["model_box"] = write_model_box_maps(
                    native_package_entries(maps),
                    model,
                    margin_A=float(getattr(args, "model_box_margin", DEFAULT_MODEL_BOX_MARGIN_A)),
                    write_res=False,
                )
            except Exception as exc:  # figure boxes never block the finish summary
                maps["model_box"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - started
    endpoint_res = model
    endpoint_hkl = endpoint_res.with_suffix(".hkl")
    endpoint_lst = endpoint_res.with_suffix(".lst")
    metrics = parse_shelx_lst_metrics(endpoint_lst) if endpoint_lst.is_file() else {}
    pathology = parse_shelx_lst_pathologies(endpoint_lst) if endpoint_lst.is_file() else {}
    source_model_text = str(workflow.get("source_model") or "").strip()
    source_model = Path(source_model_text) if source_model_text else None
    source_lst = source_model.with_suffix(".lst") if source_model is not None else None
    source_hkl = source_model.with_suffix(".hkl") if source_model is not None else None
    setup_stage = _stage(workflow, "setup")
    setup_outputs = setup_stage.get("outputs") if isinstance(setup_stage.get("outputs"), Mapping) else {}
    baseline_hkl_text = str(setup_outputs.get("baseline_hkl") or "").strip()
    baseline_hkl = Path(baseline_hkl_text) if baseline_hkl_text else None
    baseline_summary_text = str(workflow.get("conventional_baseline") or "").strip()
    baseline_summary_path = Path(baseline_summary_text) if baseline_summary_text else None
    fresh_baseline = (
        _read_json(baseline_summary_path)
        if baseline_summary_path is not None and baseline_summary_path.is_file()
        else {}
    )
    fresh_inputs = fresh_baseline.get("inputs") if isinstance(fresh_baseline.get("inputs"), Mapping) else {}
    fresh_shelxl = fresh_baseline.get("shelxl") if isinstance(fresh_baseline.get("shelxl"), Mapping) else {}
    fresh_res_input = fresh_inputs.get("res") if isinstance(fresh_inputs.get("res"), Mapping) else {}
    fresh_hkl_input = fresh_inputs.get("hkl") if isinstance(fresh_inputs.get("hkl"), Mapping) else {}
    fresh_baseline_current = bool(
        source_model is not None
        and source_model.is_file()
        and baseline_hkl is not None
        and baseline_hkl.is_file()
        and str(fresh_res_input.get("sha256") or "") == _sha256(source_model)
        and str(fresh_hkl_input.get("sha256") or "") == _sha256(baseline_hkl)
        and str(fresh_shelxl.get("status") or "") == "ok"
    )
    sibling_baseline_current = bool(
        source_lst is not None
        and source_lst.is_file()
        and source_hkl is not None
        and source_hkl.is_file()
        and baseline_hkl is not None
        and baseline_hkl.is_file()
        and _sha256(source_hkl) == _sha256(baseline_hkl)
    )
    starting_endpoint_comparable = fresh_baseline_current or sibling_baseline_current
    if fresh_baseline_current:
        source_lst = Path(str(fresh_shelxl.get("lst") or ""))
        starting_metrics = dict(fresh_shelxl)
        starting_pathology = {
            key: fresh_shelxl.get(key)
            for key in (
                "pathology_status",
                "npd_count",
                "split_count",
                "npd_atoms",
                "has_npd",
                "has_split_warnings",
            )
        }
        starting_reason = "fresh one-cycle SHELXL control on the supplied source model and baseline HKL"
    elif sibling_baseline_current:
        starting_metrics = parse_shelx_lst_metrics(source_lst)
        starting_pathology = parse_shelx_lst_pathologies(source_lst)
        starting_reason = "source-model LST and supplied baseline HKL are hash-matched"
    else:
        starting_metrics = {}
        starting_pathology = {}
        starting_reason = (
            "starting SHELXL statistics omitted because no hash-bound source-model/baseline control is available"
        )
    prep_path = Path(str(workflow.get("prep") or ""))
    transforms: dict[str, str] = {}
    if prep_path.is_file():
        prep = load_prep(prep_path)
        transforms = {dataset.name: dataset.hkl_transform for dataset in prep.datasets}
    dataset_payload: dict[str, object] = {}
    for name, run in row_runs.items():
        summary = _read_json(run / "summary.json")
        best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
        dataset_payload[name] = {
            "row_table": str(run / "native_dyn_calc.csv"),
            "summary": str(run / "summary.json"),
            "metrics": best.get("metrics", {}),
            "nuisance": best.get("nuisance", {}),
            "hkl_transform": transforms.get(name, ""),
        }
    try:
        combined_row_stats, per_dataset_row_stats = _manual_row_statistics(row_runs, model)
        row_statistics_status = "complete"
        row_statistics_reason = ""
    except (OSError, ValueError, KeyError) as exc:
        combined_row_stats = {}
        per_dataset_row_stats = {}
        row_statistics_status = "unavailable"
        row_statistics_reason = str(exc)
    jana_json = out / "JANA_ROW_STATISTICS.json"
    jana_per_dataset_json = out / "JANA_ROW_STATISTICS_PER_DATASET.json"
    _write_json(jana_json, combined_row_stats)
    _write_json(jana_per_dataset_json, per_dataset_row_stats)
    hand_summary = _selected_hand_summary(workflow)
    scale_summary = _manual_scale_summary(workflow, row_runs)
    feedback_outputs = feedback.get("outputs") if isinstance(feedback.get("outputs"), Mapping) else {}
    decision_path = Path(
        str(feedback_outputs.get("decision") or root / "04_feedback" / "multi_endpoint_push_decision.json")
    )
    decision = _read_json(decision_path) if decision_path.is_file() else {}
    selected_cycle = int(feedback_outputs.get("selected_cycle", 0) or 0)
    feedback_summary_path = Path(str(feedback_outputs.get("feedback_summary") or ""))
    feedback_summary = _read_json(feedback_summary_path) if feedback_summary_path.is_file() else {}
    single_run_metrics = {
        int(str(run.get("label") or "0").split("_")[-1]): run.get("metrics", {})
        for run in feedback_summary.get("runs", [])
        if str(run.get("label") or "").startswith("cycle_")
    }
    feedback_cycles = []
    for row in decision.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        cycle = int(row.get("cycle", 0) or 0)
        run_metrics = single_run_metrics.get(cycle, {})
        feedback_cycles.append(
            {
                **dict(row),
                "dyn_R1_all": row.get("dyn_R1_all", run_metrics.get("native_dyn_R1_all")),
                "dyn_wR2_all": row.get(
                    "dyn_wR2_all",
                    run_metrics.get("native_dyn_wR2_all", row.get("native_wR2_all")),
                ),
                "npd_count": row.get("npd_count", len(row.get("npd_atoms") or [])),
                "selected": cycle == selected_cycle,
            }
        )
    selected_decision_raw = decision.get("selected_cycle") or decision.get("selected_safe_cycle")
    selected_decision = selected_decision_raw if isinstance(selected_decision_raw, Mapping) else {}
    stage_timings = {}
    for stage_name, stage in workflow.get("stages", {}).items():
        command_record = stage.get("command") if isinstance(stage.get("command"), Mapping) else {}
        stage_timings[stage_name] = _command_elapsed(command_record)
    map_outputs = maps.get("outputs") if isinstance(maps.get("outputs"), Mapping) else {}
    payload = {
        "schema_version": 1,
        "kind": "grond_manual_final_summary",
        "workflow": str(workflow_path),
        "model": str(model),
        "selected_cycle": selected_cycle,
        "selection_reason": decision.get("selection_reason", ""),
        "feedback_architecture": {
            "per_dataset_step": "Bloch/Born row recomputation and pinned nuisance-scale fit; no separate coordinate refinement",
            "joint_step": (
                "not applicable for one crystal"
                if single_dataset_endpoint
                else "model-symmetry merge with cross-dataset overlap scaling and inverse-variance weights"
            ),
            "coordinate_step": (
                "one single-crystal SHELXL LS1 update per macrocycle"
                if single_dataset_endpoint
                else "one shared SHELXL LS1 update per macrocycle"
            ),
            "summary": (
                "Each macrocycle recomputes the crystal rows/scales and executes one SHELXL LS1 coordinate update."
                if single_dataset_endpoint
                else "Each macrocycle recomputes rows separately, performs one inverse-variance/model-symmetry merge, and executes one shared SHELXL LS1 coordinate update."
            ),
            "rows_recomputed_after_selection": True,
        },
        "starting_endpoint": {
            "model": str(source_model) if source_model is not None else "",
            "lst": str(source_lst) if source_lst is not None and source_lst.is_file() else "",
            "baseline_hkl": str(baseline_hkl) if baseline_hkl is not None else "",
            "comparable_to_selected_support": starting_endpoint_comparable,
            "reason": (starting_reason),
            **starting_metrics,
            **starting_pathology,
        },
        "endpoint": {
            "res": str(endpoint_res),
            "hkl": str(endpoint_hkl) if endpoint_hkl.exists() else "",
            "lst": str(endpoint_lst) if endpoint_lst.exists() else "",
            **metrics,
            **pathology,
            "parameters": selected_decision.get("endpoint_parameters", selected_decision.get("parameters")),
            "restraints": selected_decision.get("endpoint_restraints", selected_decision.get("restraints")),
        },
        "datasets": dataset_payload,
        "hand": hand_summary,
        "row_statistics": {
            "definition": "JANA/PETS-style GROND calculation; distinct from SHELXL endpoint statistics",
            "status": row_statistics_status,
            "reason": row_statistics_reason,
            "instability_u": 0.01,
            "combined": combined_row_stats,
            "per_dataset": per_dataset_row_stats,
            "json": str(jana_json),
            "per_dataset_json": str(jana_per_dataset_json),
        },
        "scales": scale_summary,
        "feedback_cycles": feedback_cycles,
        "timings_seconds": stage_timings,
        "map_manifest": str(map_outputs.get("manifest", "")),
        "maps": maps,
        "elapsed_seconds": elapsed,
    }
    summary_json = out / "manual_final_summary.json"
    summary_md = out / "MANUAL_FINAL_SUMMARY.md"
    _write_json(summary_json, payload)
    summary_md.write_text(_manual_markdown_summary(payload))
    root_summary_json = root / "MANUAL_TOP_LEVEL_SUMMARY.json"
    root_summary_md = root / "MANUAL_TOP_LEVEL_SUMMARY.md"
    shutil.copy2(summary_json, root_summary_json)
    shutil.copy2(summary_md, root_summary_md)
    applied_scales = scale_summary.get("applied") if isinstance(scale_summary.get("applied"), Mapping) else {}
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="finish",
        output_dir=out,
        status="complete",
        outputs={
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "top_level_summary_json": str(root_summary_json),
            "top_level_summary_markdown": str(root_summary_md),
            "model": str(model),
            "endpoint_hkl": str(endpoint_hkl) if endpoint_hkl.exists() else "",
            "map_status": maps.get("status", ""),
            "map_manifest": str(map_outputs.get("manifest", "")),
            "headline": {
                "endpoint": (
                    f"R1gt {_percent(metrics.get('R1_gt'))}; R1all {_percent(metrics.get('R1_all'))}; "
                    f"wR2all {_percent(metrics.get('wR2_all'))}; GooF {metrics.get('GooF', 'n/a')}"
                ),
                "hand": (
                    f"pre-feedback {hand_summary.get('choice', 'n/a')} selected; "
                    f"recommendation {hand_summary.get('recommendation', 'n/a')}"
                ),
                "scales": (
                    f"{applied_scales.get('frame_parameters', 'n/a')} frame + "
                    f"{applied_scales.get('detector_parameters', 'n/a')} detector; full pinned contract applied"
                ),
                "summary": str(root_summary_md),
            },
        },
        command={"elapsed_seconds": elapsed, "map_mode": args.map_mode},
        next_command="",
    )
    _print_handoff(handoff)
    return 0
