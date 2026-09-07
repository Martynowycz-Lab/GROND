"""Summarize endpoint-push cycles and the selected usable model."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import read_json, try_read_json
from grond.reports.shelxl import _quick_shelxl_details


def write_multi_push_endpoint_report(out_dir: Path, *, best_cycle_tolerance: float) -> Path | None:
    """Write a compact safety report for a multi-dataset endpoint-push run."""

    summary_path = out_dir / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    compact_scale_specs = summary.get("dataset_scale_specs")
    compact_scale_specs = compact_scale_specs if isinstance(compact_scale_specs, Mapping) else {}
    compact_scale_replayed = bool(compact_scale_specs)
    scale_contract_note = (
        "Each cycle replays the declared compact per-dataset scale topology after fresh Bloch prediction and before "
        "the shared endpoint. The topology is frozen; this continuation does not select a new scale model."
        if compact_scale_replayed
        else "These cycles refit the pinned full scale grouping. Compact scale reselection is not performed inside "
        "`push-endpoint`; run it separately on the selected safe cycle before calling it the most compact publication endpoint."
    )
    rows = []
    for record in summary.get("cycles", []):
        combined = Path(str(record.get("combined") or ""))
        combined_summary = try_read_json(combined / "combined_summary.json")
        native = combined_summary.get("native_joint") or {}
        endpoint = combined_summary.get("combined_endpoint") or {}
        shelxl = endpoint.get("shelxl") or {}
        model_text = str(record.get("output_model") or "")
        model = None if model_text in {"", "."} else Path(model_text)
        lst = model.with_suffix(".lst") if model is not None else None
        shelxl_details = _quick_shelxl_details(shelxl, lst)
        post_sigma = endpoint.get("post_bloch_sigma") or {}
        dyn_all = (native.get("dyn") or {}).get("all") or {}
        born_all = (native.get("born") or {}).get("all") or {}
        rows.append(
            {
                "cycle": int(record.get("cycle") or len(rows) + 1),
                "model": str(record.get("output_model") or ""),
                "combined": str(combined) if str(combined) != "." else "",
                "merge_key": endpoint.get("merge_key", ""),
                "unmerged_rows": endpoint.get("input_unmerged_rows"),
                "merged_rows": endpoint.get("merged_endpoint_rows", endpoint.get("merged_exact_hkl_rows")),
                "dyn_R1_all": dyn_all.get("R1_all"),
                "dyn_wR2_all": dyn_all.get("wR2_all"),
                "born_R1_all": born_all.get("R1_all"),
                "born_wR2_all": born_all.get("wR2_all"),
                "endpoint_R1_all": shelxl.get("R1_all"),
                "endpoint_wR2_all": shelxl.get("wR2_all"),
                "endpoint_GooF": shelxl.get("GooF"),
                "endpoint_parameters": shelxl_details.get("parameters"),
                "endpoint_restraints": shelxl_details.get("restraints"),
                "npd_count": int(shelxl_details.get("npd_count", 0) or 0),
                "npd_atoms": shelxl_details.get("npd_atoms", []),
                "split_count": int(shelxl_details.get("split_count", 0) or 0),
                "sigma_scale": post_sigma.get("sigma_scale"),
            }
        )
    finite = [r for r in rows if r["dyn_wR2_all"] is not None and math.isfinite(float(r["dyn_wR2_all"]))]
    best = min(finite, key=lambda r: float(r["dyn_wR2_all"])) if finite else None
    safe = [r for r in finite if int(r.get("npd_count", 0) or 0) == 0]
    if safe:
        best_safe = min(safe, key=lambda r: float(r["dyn_wR2_all"]))
        threshold = float(best_safe["dyn_wR2_all"]) + best_cycle_tolerance
        within = [r for r in safe if float(r["dyn_wR2_all"]) <= threshold]
        selected = min(within or safe, key=lambda r: r["cycle"])
        selection_status = "promoted_safe_no_npd"
        selection_reason = "selected earliest no-NPD cycle within the native row-intensity tolerance; split warnings are reported only"
    elif finite:
        best_pathology = min(int(r.get("npd_count", 0) or 0) for r in finite)
        low_pathology = [
            r
            for r in finite
            if int(r.get("npd_count", 0) or 0) == best_pathology
        ]
        best_low_pathology = min(low_pathology, key=lambda r: float(r["dyn_wR2_all"]))
        threshold = float(best_low_pathology["dyn_wR2_all"]) + best_cycle_tolerance
        within = [r for r in low_pathology if float(r["dyn_wR2_all"]) <= threshold]
        selected = min(within or low_pathology, key=lambda r: r["cycle"])
        selection_status = "diagnostic_no_safe_cycle"
        selection_reason = "no no-NPD cycle was available; selected the lowest-NPD diagnostic cycle instead of chasing row wR(I)"
    else:
        selected = None
        selection_status = "no_finite_row_metrics"
        selection_reason = "no finite row-level dynamical metrics were found"
    decision = {
        "summary": str(summary_path),
        "best_cycle_tolerance": best_cycle_tolerance,
        "compact_scale_reselection_performed": False,
        "compact_scale_contract_replayed": compact_scale_replayed,
        "dataset_scale_specs": dict(compact_scale_specs),
        "scale_contract_note": scale_contract_note,
        "best_native_cycle": best,
        "selected_cycle": selected,
        "selected_status": selection_status,
        "selection_reason": selection_reason,
        "rows": rows,
    }
    decision_path = out_dir / "multi_endpoint_push_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    report_path = out_dir / "MULTI_ENDPOINT_PUSH_REPORT.md"
    with report_path.open("w") as handle:
        handle.write("# GROND Multi-Dataset Endpoint Push Report\n\n")
        handle.write(
            "This report summarizes a deliberate continuation from a finished multi-dataset GROND checkpoint. "
            "Each cycle recomputes Bloch rows for every dataset from the same edited model, merges corrected endpoint observations, "
            "runs one shared SHELXL LS step, and sends the updated shared model back to all datasets.\n\n"
        )
        handle.write(scale_contract_note + "\n\n")
        handle.write("| cycle | merge key | unmerged rows | merged rows | row dyn R(I)all | row dyn wR(I)all | row Born R(I)all | row Born wR(I)all | endpoint R1(F)all | endpoint wR2(F^2)all | GooF | NPD | split | sigma scale | model |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            npd_atoms = row.get("npd_atoms")
            npd_text = str(row.get("npd_count", ""))
            if isinstance(npd_atoms, list) and npd_atoms:
                npd_text += f" ({', '.join(str(atom) for atom in npd_atoms[:5])}{', ...' if len(npd_atoms) > 5 else ''})"
            handle.write(
                f"| {row['cycle']} | {row['merge_key'] or '-'} | {row['unmerged_rows'] or ''} | {row['merged_rows'] or ''} | "
                f"{100.0 * float(row['dyn_R1_all'] or math.nan):.2f}% | {100.0 * float(row['dyn_wR2_all'] or math.nan):.2f}% | "
                f"{100.0 * float(row['born_R1_all'] or math.nan):.2f}% | {100.0 * float(row['born_wR2_all'] or math.nan):.2f}% | "
                f"{100.0 * float(row['endpoint_R1_all'] or math.nan):.2f}% | {100.0 * float(row['endpoint_wR2_all'] or math.nan):.2f}% | "
                f"{float(row['endpoint_GooF'] or math.nan):.3f} | {npd_text} | {row.get('split_count', '')} | "
                f"{float(row['sigma_scale'] or math.nan):.3f} | `{row['model']}` |\n"
            )
        handle.write("\n")
        if selected:
            handle.write(f"Selected cycle: {selected['cycle']} at `{selected['model']}`.\n\n")
            handle.write(f"Selection status: `{selection_status}`. {selection_reason}.\n")
        else:
            handle.write(f"Selected cycle: none. {selection_reason}.\n")
        if best:
            handle.write(f"\nLowest native row wR(I)all cycle: {best['cycle']} at `{best['model']}`.\n")
    return report_path


def write_push_endpoint_report(out_dir: Path, *, best_cycle_tolerance: float) -> Path | None:
    """Write a compact safety report for a completed endpoint-push run."""

    summary_path = out_dir / "shelxl_feedback_loop_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = []
    for run in summary.get("runs", []):
        metrics = run.get("metrics") or {}
        model_text = str(run.get("output_model") or "")
        model = None if model_text in {"", "."} else Path(model_text)
        lst = model.with_suffix(".lst") if model is not None else None
        text = lst.read_text(errors="ignore") if lst is not None and lst.exists() else ""
        split_match = re.search(r"(\d+)\s+atoms may be split", text)
        param_match = re.search(r"Total number of l\.s\. parameters =\s+(\d+)", text)
        restraint_match = re.search(r"Restrained GooF =\s+[0-9.]+ for\s+(\d+) restraints", text)
        recorded_health = run.get("endpoint_res_health")
        recorded_health = recorded_health if isinstance(recorded_health, Mapping) else {}
        endpoint_health = _quick_shelxl_details(recorded_health, lst)
        npd_atoms = endpoint_health.get("npd_atoms")
        npd_atoms = list(npd_atoms) if isinstance(npd_atoms, (list, tuple, set)) else []
        npd_count = int(endpoint_health.get("npd_count", len(npd_atoms)) or 0)
        has_npd = bool(endpoint_health.get("has_npd", False) or npd_count > 0)
        split_count = int(endpoint_health.get("split_count", int(split_match.group(1)) if split_match else 0) or 0)
        safe_for_feedback = bool(
            endpoint_health.get("lst_safe_for_feedback") is True
            and endpoint_health.get("safe_for_feedback") is not False
            and not has_npd
        )
        rows.append(
            {
                "cycle": len(rows) + 1,
                "model": str(model) if model is not None else "",
                "status": endpoint_health.get("reason") or metrics.get("endpoint_status") or "ok",
                "elapsed_seconds": run.get("elapsed_seconds"),
                "parameters": int(param_match.group(1)) if param_match else None,
                "restraints": int(restraint_match.group(1)) if restraint_match else None,
                "npd_atoms": npd_atoms,
                "npd_count": npd_count,
                "has_npd": has_npd,
                "safe_for_feedback": safe_for_feedback,
                "endpoint_res_health": endpoint_health,
                "split_count": split_count,
                "native_wR2_all": metrics.get("native_dyn_wR2_all"),
                "native_free_wR2_all": metrics.get("native_dyn_free_wR2_all"),
                "endpoint_R1_all": metrics.get("endpoint_R1_all"),
                "endpoint_wR2_all": metrics.get("endpoint_wR2_all"),
                "endpoint_GooF": metrics.get("endpoint_GooF"),
                "sigma_scale": metrics.get("post_bloch_sigma_scale"),
            }
        )
    finite_rows = [r for r in rows if r["native_wR2_all"] is not None and math.isfinite(float(r["native_wR2_all"]))]
    best_native = min(finite_rows, key=lambda r: float(r["native_wR2_all"])) if finite_rows else None
    threshold = float(best_native["native_wR2_all"]) + best_cycle_tolerance if best_native else math.inf
    safe_rows = [
        r
        for r in finite_rows
        if r.get("safe_for_feedback") is True and int(r.get("npd_count", 0) or 0) == 0
    ]
    within = [r for r in safe_rows if float(r["native_wR2_all"]) <= threshold]
    selected = min(within or safe_rows, key=lambda r: (r["cycle"], float(r["native_wR2_all"]))) if safe_rows else None
    decision = {
        "summary": str(summary_path),
        "best_cycle_tolerance": best_cycle_tolerance,
        "best_native_cycle": best_native,
        "selected_safe_cycle": selected,
        "rows": rows,
    }
    decision_path = out_dir / "endpoint_push_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    report_path = out_dir / "ENDPOINT_PUSH_REPORT.md"
    with report_path.open("w") as handle:
        handle.write("# GROND Endpoint Push Report\n\n")
        handle.write("This report summarizes a continuation run from an existing pinned GROND row/nuisance summary. The selected model is the earliest no-NPD cycle within the requested native-row tolerance when possible. Cycles with NPD atoms are kept as diagnostics, not promoted; split warnings are reported but do not block selection.\n\n")
        handle.write("| cycle | status | seconds | params | restraints | NPD atoms | split warnings | row wR(I)all | row free wR(I)all | endpoint R1(F)all | endpoint wR2(F^2)all | GooF | sigma scale |\n")
        handle.write("|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            npd = ",".join(row["npd_atoms"]) if row["npd_atoms"] else "-"
            handle.write(
                f"| {row['cycle']} | {row['status']} | {float(row['elapsed_seconds'] or 0):.1f} | {row['parameters']} | {row['restraints']} | "
                f"{npd} | {row['split_count']} | {float(row['native_wR2_all'] or math.nan):.4f} | "
                f"{float(row['native_free_wR2_all'] or math.nan):.4f} | {float(row['endpoint_R1_all'] or math.nan):.4f} | "
                f"{float(row['endpoint_wR2_all'] or math.nan):.4f} | {float(row['endpoint_GooF'] or math.nan):.3f} | "
                f"{float(row['sigma_scale'] or math.nan):.3f} |\n"
            )
        handle.write("\n")
        if selected:
            handle.write(f"Selected safe cycle: {selected['cycle']} at `{selected['model']}`.\n")
        else:
            handle.write("Selected safe cycle: none. All completed cycles had pathology warnings or missing metrics.\n")
        if best_native:
            handle.write(f"\nLowest row wR(I)all cycle: {best_native['cycle']} at `{best_native['model']}`.\n")
    return report_path
