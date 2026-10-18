"""Feed updated models through pinned dynamical calculations and audit scale choices."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from grond.io.config import load_prep
from grond.manual.execution import _child_base, _run_stage_command, _tool_args, _verbose_child_output
from grond.manual.state import (
    _load_workflow,
    _next_command,
    _print_handoff,
    _read_json,
    _record_stage,
    _require_stage,
    _sha256,
    _stage,
    _summary_nuisance,
    _write_json,
)


def _manual_checkpoint(workflow: Mapping[str, object], output: Path) -> dict[str, object]:
    """Write a continue checkpoint for ``grond refine`` from the thickness-stage summaries; return it.

    Each crystal keeps its selected summary's nuisance and prediction settings; the start model is the workflow's chosen
    model.
    """

    prep = load_prep(Path(str(workflow["prep"])))
    thickness = _stage(workflow, "thickness")
    selected = dict(thickness["outputs"]["selected_summaries"])
    datasets = []
    for dataset in prep.datasets:
        summary_path = Path(selected[dataset.name]).resolve()
        summary, nuisance, config = _summary_nuisance(summary_path)
        final_config = {
            key: config[key]
            for key in (
                "hkl_transform",
                "n_basis",
                "n_points",
                "topup_d_min",
                "row_plan_cache",
                "chunksize",
                "post_bloch_sigma_policy",
                "min_d_spacing",
                "min_peak",
                "geometry_percentile",
            )
            if key in config and config[key] not in (None, "")
        }
        datasets.append(
            {
                **asdict(dataset),
                "source_run": str(summary_path.parent),
                "source_summary": str(summary_path),
                "nuisance": nuisance,
                "final_config": final_config,
                "source_summary_sha256": _sha256(summary_path),
                "row_table": str(summary_path.parent / "native_dyn_calc.csv"),
            }
        )
    model = Path(str(workflow.get("chosen_model") or prep.structure)).resolve()
    single_dataset = datasets[0] if len(datasets) == 1 else None
    checkpoint = {
        "schema_version": 1,
        "kind": "grond_continue_checkpoint",
        "status": "ready",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(output.parent),
        "dataset_count": len(datasets),
        "dataset": datasets[0]["name"],
        "source_run": str(single_dataset["source_run"]) if single_dataset else "",
        "source_summary": str(single_dataset["source_summary"]) if single_dataset else "",
        "recommended_start_res": str(model),
        "hkl": datasets[0]["hkl"],
        "baseline_hkl": prep.baseline_hkl,
        "hkl_transform": datasets[0]["hkl_transform"],
        "datasets": datasets,
        "structure_dataset_policy": {},
        "final_dataset_policy": {},
        "multi_endpoint_merge_key": "model_symmetry",
        "manual_workflow": str(workflow.get("root", "")),
    }
    _write_json(output, checkpoint)
    return checkpoint


def manual_feedback(args) -> int:
    """Run the ``feedback`` stage: SHELXL feedback macro-cycles from the chosen model, then its final rows.

    The selected cycle must be clean (no NPD atoms). Its rows are recomputed with ``--cycles 0`` so later stages read
    rows of exactly that model.
    """

    workflow_path, workflow = _load_workflow(args)
    hand = _require_stage(workflow, "hand")
    if hand.get("status") == "needs_choice" or not workflow.get("chosen_model"):
        raise SystemExit(
            "Hand stage needs an explicit choice. Run the next command printed by `grond manual hand`, "
            "or use `grond manual hand --workflow ... --choice skip`."
        )
    root = Path(workflow["root"])
    out = root / "04_feedback"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "input_checkpoint.json"
    checkpoint = _manual_checkpoint(workflow, checkpoint_path)
    single_dataset = int(checkpoint.get("dataset_count", 0) or 0) == 1
    ignore_source_baseline = single_dataset and not str(checkpoint.get("baseline_hkl") or "").strip()
    # Several crystals: each crystal's error model (--dataset-sigma-policy error_model, the AUTO default) sets the
    # merged HKL's sigmas and the combine refuses a second uniform rescale of it, as in AUTO's shared loop.
    post_bloch_sigma_policy = "uniform_z_rms" if single_dataset else "none"
    command_line = [
        *_child_base(args),
        "refine",
        "--checkpoint",
        str(checkpoint_path),
        "--start-res",
        str(workflow["chosen_model"]),
        "--out-dir",
        str(out),
        "--cycles",
        str(args.cycles),
        "--n-points",
        str(args.n_points),
        "--final-n-points",
        str(args.n_points),
        "--workers",
        str(args.workers),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        "1",
        "--endpoint-ls-cycles",
        "1",
        "--post-bloch-sigma-policy",
        post_bloch_sigma_policy,
        "--lambda-grid",
        "1.0",
        "--row-plan-cache",
        "off",
        *_tool_args(args),
    ]
    if int(args.n_basis) > 0:
        command_line += ["--n-basis", str(args.n_basis)]
    if ignore_source_baseline:
        command_line.append("--ignore-source-baseline-hkl")
    command = _run_stage_command(command_line, verbose=_verbose_child_output(args))
    if single_dataset:
        push_summary_path = out / "shelxl_feedback_loop_summary.json"
        decision_path = out / "endpoint_push_decision.json"
        feedback_summary_path = push_summary_path
        report_path = out / "ENDPOINT_PUSH_REPORT.md"
        push_summary = _read_json(push_summary_path)
        decision = _read_json(decision_path)
        selected = decision.get("selected_safe_cycle")
    else:
        push_summary_path = out / "push_endpoint_summary.json"
        decision_path = out / "multi_endpoint_push_decision.json"
        feedback_summary_path = out / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
        report_path = out / "MULTI_ENDPOINT_PUSH_REPORT.md"
        push_summary = _read_json(push_summary_path)
        decision = _read_json(decision_path)
        selected = decision.get("selected_cycle")
    if not isinstance(selected, Mapping) or not selected.get("model"):
        raise RuntimeError(
            "Feedback produced no clean selected model; diagnostic best-native/NPD cycles are not eligible for "
            f"manual handoff. Inspect {report_path}."
        )
    selected_model = str(Path(str(selected["model"])).resolve())
    selected_npd_count = int(selected.get("npd_count", len(selected.get("npd_atoms") or [])) or 0)
    selected_health = (
        selected.get("endpoint_res_health") if isinstance(selected.get("endpoint_res_health"), Mapping) else {}
    )
    if selected_npd_count > 0 or selected.get("has_npd") is True or selected_health.get("safe_for_feedback") is False:
        raise RuntimeError(
            f"Feedback selected only an unsafe diagnostic model (NPD={selected_npd_count}); "
            f"manual handoff requires a clean cycle. Inspect {report_path}."
        )
    feedback_summary = _read_json(feedback_summary_path)
    cycle_number = int(selected.get("cycle", 0) or 0)
    if single_dataset:
        cycle_record = next(
            (
                row
                for row in feedback_summary.get("runs", [])
                if str(row.get("label") or "") == f"cycle_{cycle_number:02d}"
            ),
            {},
        )
    else:
        cycle_record = next(
            (row for row in feedback_summary.get("cycles", []) if int(row.get("cycle", 0) or 0) == cycle_number),
            {},
        )
    selected_rows_out = out / "selected_rows"
    recompute_command = [
        *_child_base(args),
        "refine",
        "--checkpoint",
        str(checkpoint_path),
        "--start-res",
        selected_model,
        "--out-dir",
        str(selected_rows_out),
        "--cycles",
        "0",
        "--recompute-final-rows",
        "--n-points",
        str(args.n_points),
        "--final-n-points",
        str(args.n_points),
        "--workers",
        str(args.workers),
        "--shelxl",
        args.shelxl,
        "--row-plan-cache",
        "off",
        *_tool_args(args),
    ]
    if int(args.n_basis) > 0:
        recompute_command += ["--n-basis", str(args.n_basis)]
    if ignore_source_baseline:
        recompute_command.append("--ignore-source-baseline-hkl")
    recompute = _run_stage_command(recompute_command, verbose=_verbose_child_output(args))
    selected_feedback_path = (
        selected_rows_out / "shelxl_feedback_loop_summary.json"
        if single_dataset
        else selected_rows_out / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    )
    selected_feedback = _read_json(selected_feedback_path)
    if single_dataset:
        final_run_text = str(selected_feedback.get("final_row_stats_run") or "")
        final_run = Path(final_run_text)
        if not final_run.is_dir() or not (final_run / "native_dyn_calc.csv").is_file():
            raise RuntimeError("Selected-model single-dataset row recomputation did not complete")
        raw_runs = {str(checkpoint.get("dataset") or "dataset"): str(final_run.resolve())}
    else:
        final_rows = selected_feedback.get("final_row_recompute")
        if not isinstance(final_rows, Mapping) or final_rows.get("status") != "ok":
            raise RuntimeError("Selected-model row recomputation did not complete")
        raw_runs = dict(final_rows.get("raw_runs") or {})
    workflow["chosen_model"] = selected_model
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="feedback",
        output_dir=out,
        status="ready",
        outputs={
            "checkpoint": str(checkpoint_path),
            "selected_model": selected_model,
            "selected_cycle": cycle_number,
            "selected_endpoint_combined_dir": str(cycle_record.get("combined") or Path(selected_model).parent),
            "selected_row_runs": raw_runs,
            "feedback_summary": str(feedback_summary_path),
            "selected_row_recompute_summary": str(selected_feedback_path),
            "report": str(report_path),
            "push_summary": str(push_summary_path),
            "feedback_status": push_summary.get("status") or ("ready" if selected_model else ""),
            "decision": str(decision_path),
            "headline": {
                "selected_cycle": cycle_number,
                "endpoint_R1all": f"{100.0 * float(selected.get('endpoint_R1_all', float('nan'))):.2f}%",
                "endpoint_wR2all": f"{100.0 * float(selected.get('endpoint_wR2_all', float('nan'))):.2f}%",
                "endpoint_GooF": selected.get("endpoint_GooF", "n/a"),
                "NPD/split": f"{selected.get('npd_count', len(selected.get('npd_atoms') or []))}/{selected.get('split_count', 'n/a')}",
                "coordinate_updates": (
                    "one single-crystal SHELXL LS1 per macrocycle"
                    if single_dataset
                    else "one joint inverse-variance SHELXL LS1 per macrocycle"
                ),
            },
        },
        command={"feedback": command, "selected_row_recompute": recompute},
        next_command=_next_command("scales", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0


def manual_scales(args) -> int:
    """Run the ``scales`` stage: record that the full pinned frame scales stay applied."""

    workflow_path, workflow = _load_workflow(args)
    _require_stage(workflow, "feedback")
    root = Path(workflow["root"])
    out = root / "05_scales"
    out.mkdir(parents=True, exist_ok=True)
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="scales",
        output_dir=out,
        status="skipped",
        outputs={
            "datasets": {},
            "application": "full pinned scales remain active",
            "headline": {"application": "full pinned frame scales retained"},
        },
        next_command=_next_command("finish", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0
