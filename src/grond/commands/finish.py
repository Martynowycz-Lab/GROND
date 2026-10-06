"""Finish an existing result with selected endpoint artifacts and requested maps."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.io.config import load_prep
from grond.reports.progress import _quick_progress_finish_info
from grond.reports.quick_stats import write_quick_stats
from grond.reports.terminal import _format_terminal_finish_summary
from grond.workflows.decision import safe_float
from grond.workflows.endpoint_selection import (
    _multi_endpoint_calibration_not_applicable,
    run_publication_endpoint_chooser,
)
from grond.workflows.finish_maps import _finish_run_dir, _publication_endpoint_from_result, run_finish_maps


def cmd_finish(args: argparse.Namespace) -> int:
    """Regenerate the compact user-facing package for a completed GROND run."""

    finish_started_monotonic = time.perf_counter()
    run_dir = _finish_run_dir(args)
    result_path = run_dir / "grond_summary.json"
    prep_path = run_dir / "run_prep.json"
    result = try_read_json(result_path)
    if not result:
        raise SystemExit(f"Could not read completed GROND run summary: {result_path}")
    if not prep_path.exists():
        raise SystemExit(f"Could not read completed GROND prep file: {prep_path}")
    prep = load_prep(prep_path)
    if result.get("combined"):
        result["endpoint_calibration"] = {
            **_multi_endpoint_calibration_not_applicable(result),
            "refreshed_by_finish": True,
        }
        run_config = try_read_json(run_dir / "run_config.json")
        chooser_config = run_config.get("publication_endpoint_chooser")
        chooser_config = chooser_config if isinstance(chooser_config, Mapping) else {}
        endpoint_config = run_config.get("endpoint_calibration")
        endpoint_config = endpoint_config if isinstance(endpoint_config, Mapping) else {}
        chooser_args = argparse.Namespace(
            out_dir=str(run_dir),
            dry_run=False,
            publication_endpoint_chooser=True,
            publication_endpoint_min_hkl_safe_factor=safe_float(chooser_config.get("min_hkl_safe_factor"), 0.25),
            publication_endpoint_max_modulation_p95=safe_float(chooser_config.get("max_modulation_p95"), 20.0),
            endpoint_calibration_goof_min=safe_float(endpoint_config.get("goof_min"), 0.85),
            endpoint_calibration_goof_max=safe_float(endpoint_config.get("goof_max"), 1.15),
            publication_allow_npd=bool(chooser_config.get("allow_npd", False)),
            publication_feedback_cycles=str(
                getattr(args, "publication_feedback_cycles", None) or chooser_config.get("feedback_cycles") or "kept"
            ),
        )
        result["publication_endpoint_chooser"] = run_publication_endpoint_chooser(prep, result, chooser_args)
    quick = write_quick_stats(prep, result, argparse.Namespace(out_dir=str(run_dir)))
    result["quick_stats"] = quick
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    maps = run_finish_maps(run_dir, result, args)
    endpoint = _publication_endpoint_from_result(result)
    endpoint_res_text = str(endpoint.get("endpoint_res", "") or "").strip()
    endpoint_res = Path(endpoint_res_text).expanduser() if endpoint_res_text else None
    cif = endpoint_res.with_suffix(".cif") if endpoint_res is not None else None
    native_cif = ""
    native = maps.get("native_dynamic") if isinstance(maps, Mapping) else {}
    if isinstance(native, Mapping):
        outputs = native.get("outputs") if isinstance(native.get("outputs"), Mapping) else {}
        native_cif = str(outputs.get("dynamic_cif", "") or "")
    finish_status = "ok" if maps.get("status") in {"ok", "disabled"} else maps.get("status", "warning")
    manifest = {
        "status": finish_status,
        "warnings": list(maps.get("warnings") or []),
        "run_dir": str(run_dir),
        "summary": str(result_path),
        "prep": str(prep_path),
        "final_summary": quick.get("final_summary", ""),
        "quick_stats": quick.get("path", ""),
        "maps": maps,
        "cif": {
            "status": "native_dynamic_written"
            if native_cif
            else ("endpoint_exists" if cif is not None and cif.exists() else "endpoint_missing"),
            "native_dynamic_cif": native_cif,
            "endpoint_cif": str(cif) if cif is not None else "",
            "note": "The native GROND CIF records the row-level dynamical coefficient export. Endpoint CIF is reported only if an external endpoint already wrote one.",
        },
    }
    manifest_path = run_dir / "finish_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    console = quick.get("console") if isinstance(quick.get("console"), Mapping) else {}
    finish_elapsed_seconds = max(0.0, time.perf_counter() - finish_started_monotonic)
    payload = {
        **console,
        "status": finish_status,
        "warnings": [*list(console.get("warnings") or []), *manifest["warnings"]],
        "finished": _quick_progress_finish_info(run_dir),
        "timing_context": {
            "kind": "finish_replay",
            "finish_invocation_elapsed_seconds": finish_elapsed_seconds,
        },
        "finish_manifest": str(manifest_path),
        "maps": maps,
        "cif": manifest["cif"],
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, allow_nan=True))
    else:
        print(_format_terminal_finish_summary(payload), flush=True)
        for warning in manifest["warnings"]:
            print(f"Map warning: {warning}", flush=True)
    return 0
