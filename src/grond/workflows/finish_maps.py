"""Bind final maps to the selected model and the actual rows used for that model."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.tools import external_tool_statuses
from grond.cli_support.values import try_read_json
from grond.legacy.engine.shelx_diagnostics import parse_shelx_lst_metrics
from grond.macromolecule import sha256_file
from grond.maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package
from grond.reports.endpoints import _chooser_publication_endpoint, _endpoint_calibration_publication_endpoint
from grond.reports.values import _quick_final_run, _quick_load_run_summary
from grond.workflows.conventional_baseline import (
    _endpoint_baseline_no_harm_gate,
    _validated_fresh_conventional_baseline,
)
from grond.workflows.endpoint_selection import _multi_hand_publication_blockers


def _finish_run_dir(args: argparse.Namespace) -> Path:
    """Resolve the GROND run directory for `finish`."""

    if getattr(args, "checkpoint", ""):
        checkpoint = try_read_json(Path(args.checkpoint).expanduser().resolve())
        run_dir = checkpoint.get("run_dir")
        if not run_dir:
            raise SystemExit(f"Checkpoint does not record run_dir: {args.checkpoint}")
        return Path(str(run_dir)).expanduser().resolve()
    return Path(getattr(args, "run_dir", "") or ".").expanduser().resolve()


def _publication_endpoint_from_result(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the selected publication endpoint block from a run result."""

    def no_harm_checked(endpoint: Mapping[str, object]) -> Mapping[str, object]:
        if not endpoint:
            return {}
        if result.get("combined") and _multi_hand_publication_blockers(result):
            return {}
        if endpoint.get("artifact_provenance_ok") is not True:
            return {}
        if str(endpoint.get("pathology_status") or "") != "complete":
            return {}
        if endpoint.get("metric_consistency_ok") is not True:
            return {}
        if endpoint.get("promotion_fields_ok") is not True:
            return {}
        source_kind = str(endpoint.get("source_kind") or "")
        if result.get("combined") and "endpoint_calibration" in source_kind:
            return {}
        final_summary = _quick_load_run_summary(_quick_final_run(result))
        combined_baseline = _validated_fresh_conventional_baseline(result) if result.get("combined") else {}
        gate = _endpoint_baseline_no_harm_gate(
            final_summary,
            endpoint,
            baseline_override=combined_baseline,
            require_baseline=bool(result.get("combined")),
        )
        if not bool(gate.get("passed", False)):
            return {}
        checked = dict(endpoint)
        checked["baseline_no_harm_gate"] = gate
        return checked

    calibration_endpoint = _endpoint_calibration_publication_endpoint(result)
    if calibration_endpoint:
        checked_calibration = no_harm_checked(calibration_endpoint)
        if checked_calibration:
            return checked_calibration
    chooser_endpoint = _chooser_publication_endpoint(result)
    if chooser_endpoint:
        return no_harm_checked(chooser_endpoint)
    chooser = (
        result.get("publication_endpoint_chooser")
        if isinstance(result.get("publication_endpoint_chooser"), Mapping)
        else {}
    )
    chooser_status = str(chooser.get("status", "") if isinstance(chooser, Mapping) else "")
    if chooser and chooser_status not in {"", "disabled", "dry_run", "no_endpoint_runs"}:
        return {}
    return {}


def _shelx2map_executable(args: argparse.Namespace) -> str:
    """Return the configured shelx2map executable, if any."""

    status = external_tool_statuses(args).get("shelx2map", {})
    resolved = str(status.get("resolved", "") or "")
    if resolved:
        return resolved
    env = os.environ.get("SHELX2MAP", "")
    if env:
        return env
    return shutil.which("shelx2map") or "shelx2map"


def _shelx_fcf_list_code(path: Path) -> int | None:
    """Return the SHELXL reflection-list code recorded in an FCF file."""

    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    match = re.search(r"_(?:shelx_)?refln_list_code\s+([0-9]+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _write_finish_list6_ins(source_res: Path, destination: Path) -> None:
    """Write an isolated zero-cycle LIST 6 replay without changing the model."""

    lines: list[str] = []
    saw_cycles = False
    saw_list = False
    for original in source_res.read_text(errors="replace").splitlines():
        stripped = original.strip()
        upper = stripped.upper()
        if upper.startswith("ACTA"):
            continue
        cycle_match = re.match(r"^(\s*)(L\.S\.|LS|CGLS)(?=\s|$)", original, flags=re.IGNORECASE)
        if cycle_match:
            indent, command = cycle_match.groups()
            original = f"{indent}{command} 0"
            saw_cycles = True
        if upper.startswith("LIST"):
            original = "LIST 6"
            saw_list = True
        lines.append(original)
        if upper.startswith("END"):
            break
    insert_at = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith(("HKLF", "END"))),
        len(lines),
    )
    additions: list[str] = []
    if not saw_cycles:
        additions.append("L.S. 0")
    if not saw_list:
        additions.append("LIST 6")
    lines[insert_at:insert_at] = additions
    destination.write_text("\n".join(lines) + "\n")


def _finish_list6_fcf(
    endpoint: Mapping[str, object],
    endpoint_res: Path,
    source_fcf: Path,
    args: argparse.Namespace,
) -> tuple[Path | None, dict[str, object]]:
    """Return a LIST 6 FCF, creating a zero-cycle SHELXL replay if required."""

    source_list_code = _shelx_fcf_list_code(source_fcf)
    if source_list_code == 6:
        return source_fcf, {
            "status": "existing_list6",
            "source_fcf": str(source_fcf.resolve()),
            "source_list_code": source_list_code,
        }

    endpoint_hkl_text = str(endpoint.get("endpoint_hkl", "") or "").strip()
    endpoint_hkl = Path(endpoint_hkl_text).expanduser() if endpoint_hkl_text else endpoint_res.with_suffix(".hkl")
    if not endpoint_hkl.is_file():
        return None, {
            "status": "missing_endpoint_hkl",
            "endpoint_hkl": str(endpoint_hkl),
            "source_fcf": str(source_fcf),
            "source_list_code": source_list_code,
        }

    shelxl_status = external_tool_statuses(args).get("shelxl", {})
    shelxl = str(shelxl_status.get("resolved", "") or "")
    if not shelxl:
        return None, {
            "status": "missing_shelxl_for_list6_replay",
            "source_fcf": str(source_fcf),
            "source_list_code": source_list_code,
            "note": "shelx2map needs a LIST 6 FCF; configure SHELXL so GROND can make a zero-cycle replay.",
        }

    replay_dir = endpoint_res.resolve().parent / "grond_finish_list6_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    stem = endpoint_res.stem
    replay_ins = replay_dir / f"{stem}.ins"
    replay_hkl = replay_dir / f"{stem}.hkl"
    replay_fcf = replay_dir / f"{stem}.fcf"
    replay_lst = replay_dir / f"{stem}.lst"
    _write_finish_list6_ins(endpoint_res, replay_ins)
    shutil.copy2(endpoint_hkl, replay_hkl)
    for stale in (replay_fcf, replay_lst, replay_dir / f"{stem}.res"):
        stale.unlink(missing_ok=True)
    proc = subprocess.run(
        [shelxl, stem],
        cwd=str(replay_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path = replay_dir / "grond_finish_list6_shelxl.log"
    log_path.write_text(proc.stdout)
    replay_list_code = _shelx_fcf_list_code(replay_fcf)
    source_lst = endpoint_res.with_suffix(".lst")
    source_metrics = parse_shelx_lst_metrics(source_lst)
    replay_metrics = parse_shelx_lst_metrics(replay_lst)
    metric_keys = ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")
    comparable = source_metrics.get("metric_status") == "complete" and replay_metrics.get("metric_status") == "complete"
    metrics_unchanged = bool(
        comparable and all(source_metrics.get(key) == replay_metrics.get(key) for key in metric_keys)
    )
    replay = {
        "status": "ok" if proc.returncode == 0 and replay_fcf.is_file() and replay_list_code == 6 else "failed",
        "source_fcf": str(source_fcf.resolve()),
        "source_list_code": source_list_code,
        "endpoint_hkl": str(endpoint_hkl.resolve()),
        "replay_dir": str(replay_dir),
        "replay_ins": str(replay_ins),
        "replay_fcf": str(replay_fcf),
        "replay_list_code": replay_list_code,
        "shelxl": shelxl,
        "log": str(log_path),
        "returncode": proc.returncode,
        "source_metrics": source_metrics,
        "replay_metrics": replay_metrics,
        "metrics_compared": comparable,
        "metrics_unchanged": metrics_unchanged if comparable else None,
        "definition": "zero-cycle SHELXL LIST 6 replay of the selected endpoint; coordinates, ADPs, weights, and corrected HKL are unchanged",
    }
    if replay["status"] != "ok":
        return None, replay
    if comparable and not metrics_unchanged:
        replay["status"] = "metric_mismatch"
        return None, replay
    return replay_fcf, replay


def _shelx2map_compatible_fcf(source: Path) -> tuple[Path, dict[str, object]]:
    """Keep long descriptive titles out of shelx2map's fixed-width header reader.

    The legacy reader loops forever after a header line exceeds its buffer.
    Shorten only the title in a separate copy. Reflection data, cell and
    symmetry must stay exactly as written by SHELXL.
    """

    with source.open(encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    changed = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "_refln_phase_calc":
            break
        header_bytes = len(line.rstrip("\r\n").encode("utf-8"))
        if header_bytes <= 79:
            continue
        if stripped.split(maxsplit=1)[:1] == ["_shelx_title"]:
            lines[index] = "_shelx_title GROND_endpoint_map\n"
            changed.append(index + 1)
        else:
            raise ValueError(
                f"shelx2map cannot safely read FCF header line {index + 1} "
                f"({header_bytes} UTF-8 bytes); source preserved: {source}"
            )
    record = {"status": "unchanged", "source_fcf": str(source.resolve()), "source_sha256": sha256_file(source)}
    if not changed:
        return source, record
    directory = source.parent / "grond_finish_map_input"
    directory.mkdir(exist_ok=True)
    compatible = directory / "map_input.fcf"
    with compatible.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(lines))
    record.update(status="shortened_descriptive_title", title_lines=changed,
                  map_input_fcf=str(compatible.resolve()), map_input_sha256=sha256_file(compatible))
    return compatible, record


def run_finish_endpoint_maps(result: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Generate corrected-HKL endpoint maps from the selected publication endpoint."""

    endpoint = _publication_endpoint_from_result(result)
    endpoint_res_text = str(endpoint.get("endpoint_res", "") or "").strip()
    if not endpoint_res_text:
        return {"status": "missing_endpoint_res", "endpoint_res": ""}
    endpoint_res = Path(endpoint_res_text).expanduser()
    if not endpoint_res.exists():
        return {"status": "missing_endpoint_res", "endpoint_res": str(endpoint_res)}
    work_dir = endpoint_res.resolve().parent
    stem = endpoint_res.stem
    source_fcf = work_dir / f"{stem}.fcf"
    if not source_fcf.exists():
        return {
            "status": "missing_fcf",
            "fcf": str(source_fcf),
            "note": "Run the endpoint branch with SHELXL available so an FCF file is written.",
        }
    fcf, list6_replay = _finish_list6_fcf(endpoint, endpoint_res, source_fcf, args)
    if fcf is None:
        return {"status": "list6_replay_failed", "endpoint_res": str(endpoint_res), "list6_replay": list6_replay}
    exe = _shelx2map_executable(args)
    if ("/" in exe and not Path(exe).exists()) or ("/" not in exe and shutil.which(exe) is None):
        return {"status": "missing_shelx2map", "executable": exe}
    try:
        fcf, fcf_compatibility = _shelx2map_compatible_fcf(fcf)
    except ValueError as exc:
        return {"status": "fcf_header_incompatible", "error": str(exc), "fcf": str(fcf.resolve()),
                "endpoint_res": str(endpoint_res.resolve()), "list6_replay": list6_replay}
    map_work_dir = fcf.resolve().parent
    map_stem = fcf.stem
    maps: dict[str, dict[str, object]] = {}
    map_specs = [
        ("fofc", "0", "1"),
        ("fo", "1", "1"),
        ("2fofc", "2", "1"),
        ("fofc_unweighted_diagnostic", "0", "0"),
    ]
    for label, mode, map_weight in map_specs:
        out_map = work_dir / f"grond_corrected_{label}.map"
        log_path = work_dir / f"grond_finish_shelx2map_{label}.log"
        out_map.unlink(missing_ok=True)
        try:
            proc = subprocess.run(
                [exe, "-w", map_weight, "-m", mode, "-o", str(out_map), map_stem],
                cwd=str(map_work_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has killed and reaped the converter. Its partial
            # output is not a map that a later caller should mistake for usable.
            out_map.unlink(missing_ok=True)
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            log_path.write_text(output + "\nshelx2map exceeded its 120-second limit; endpoint artifacts are preserved.\n")
            maps[label] = {"status": "timed_out", "map": str(out_map), "requested_map": str(out_map),
                           "log": str(log_path), "returncode": None, "exists": out_map.exists(),
                           "timeout_seconds": 120, "shelx2map_weight": float(map_weight),
                           "display_role": "diagnostic_unweighted" if map_weight == "0" else "primary_weighted"}
            continue
        log_path.write_text(proc.stdout)
        actual = out_map
        for line in proc.stdout.splitlines():
            match = re.search(r'Map data written to "([^"]+)"', line)
            if match:
                reported = Path(match.group(1))
                actual = reported if reported.is_absolute() else map_work_dir / reported
        maps[label] = {
            "map": str(actual),
            "requested_map": str(out_map),
            "log": str(log_path),
            "returncode": proc.returncode,
            "exists": actual.exists(),
            "shelx2map_weight": float(map_weight),
            "display_role": "diagnostic_unweighted" if map_weight == "0" else "primary_weighted",
        }
    maps_ok = all(item.get("returncode") == 0 and item.get("exists") for item in maps.values())
    return {
        "status": "ok" if maps_ok else "failed",
        "definition": "GROND-corrected endpoint maps from the selected corrected-HKL publication endpoint; primary maps use explicit shelx2map -w 1 model-reliability weighting, and the raw unweighted Fo-Fc map is retained as a labeled diagnostic. These are not native dynamical residual-potential maps.",
        "endpoint_res": str(endpoint_res.resolve()),
        "fcf": str(fcf.resolve()),
        "fcf_compatibility": fcf_compatibility,
        "list6_replay": list6_replay,
        "maps": maps,
    }


def _finish_native_row_tables(result: Mapping[str, object]) -> list[tuple[str, Path]]:
    """Return labeled row tables for native dynamical maps."""

    if result.get("combined"):
        runs_block = result.get("publication_runs") or result.get("combined_input_runs") or result.get("runs") or {}
        if isinstance(runs_block, Mapping):
            tables = [
                (str(name), Path(str(run_dir)).expanduser() / "native_dyn_calc.csv")
                for name, run_dir in sorted(runs_block.items(), key=lambda item: str(item[0]))
                if str(run_dir)
            ]
            if tables:
                return tables

    endpoint = _publication_endpoint_from_result(result)
    source_kind = str(endpoint.get("source_kind", "") or "")
    endpoint_row_table = str(endpoint.get("endpoint_row_table", "") or "").strip()
    endpoint_run_dir = str(endpoint.get("run_dir", "") or "").strip()
    endpoint_label = str(endpoint.get("label", "") or "publication_endpoint").strip()

    if source_kind in {"native_corrected_hkl", "observation_polish"}:
        if endpoint_row_table:
            return [(endpoint_label, Path(endpoint_row_table).expanduser())]
        if endpoint_run_dir:
            return [(endpoint_label, Path(endpoint_run_dir).expanduser() / "native_dyn_calc.csv")]

    if source_kind == "combined_corrected_hkl":
        runs_block = result.get("combined_input_runs") or result.get("publication_runs") or result.get("runs") or {}
        if isinstance(runs_block, Mapping):
            tables = [
                (str(name), Path(str(run_dir)).expanduser() / "native_dyn_calc.csv")
                for name, run_dir in sorted(runs_block.items(), key=lambda item: str(item[0]))
                if str(run_dir)
            ]
            if tables:
                return tables

    if endpoint_row_table:
        return [(endpoint_label, Path(endpoint_row_table).expanduser())]
    final_run = _quick_final_run(result)
    if final_run is not None:
        label = "combined" if result.get("combined") else "dataset"
        return [(label, final_run / "native_dyn_calc.csv")]
    return []


def _finish_native_model_contract(
    result: Mapping[str, object],
    args: argparse.Namespace,
    row_tables: list[tuple[str, Path]],
) -> tuple[Path | None, dict[str, object]]:
    """Return the exact model that generated the frozen native row tables.

    A corrected-HKL SHELXL endpoint may have coordinates or ADPs that differ
    from the model used for the row-level Bloch predictions.  Mixing that RES
    with the older ``dyn_scaled`` values would make native map coefficients
    internally inconsistent.  Prefer the hash-bound ``inputs.res`` contract
    recorded beside every row table and require all contributing crystals to
    agree.  Historical summaries without this provenance fall back explicitly
    to ``result.model``; they never silently borrow the publication endpoint.
    """

    override = str(getattr(args, "model", "") or "").strip()
    if override:
        model = Path(override).expanduser()
        return model, {
            "status": "explicit_override",
            "model": str(model),
            "definition": "user-supplied native-map model override",
            "row_models": [],
        }

    records: list[dict[str, object]] = []
    missing_contracts: list[str] = []
    for label, row_table in row_tables:
        summary_path = row_table.parent / "summary.json"
        summary = try_read_json(summary_path)
        inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
        model_text = str(inputs.get("res") or "").strip()
        if not model_text:
            missing_contracts.append(f"{label} ({summary_path})")
            continue
        model = Path(model_text).expanduser()
        expected_hash = str(inputs.get("res_sha256") or "").strip()
        if not model.is_file():
            raise RuntimeError(f"native-map row model for {label} is missing: {model}; recorded by {summary_path}")
        actual_hash = sha256_file(model)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                f"native-map row model hash changed for {label}: expected {expected_hash}, "
                f"found {actual_hash} at {model}"
            )
        records.append(
            {
                "dataset": label,
                "row_table": str(row_table),
                "summary": str(summary_path),
                "model": str(model.resolve()),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
        )

    if records and missing_contracts:
        raise RuntimeError(
            "native-map model provenance is present for only some contributing row tables; "
            "refusing a mixed verified/unverified map contract: missing " + ", ".join(missing_contracts)
        )
    if records:
        identities = {
            str(record.get("expected_sha256") or record.get("actual_sha256") or record.get("model"))
            for record in records
        }
        if len(identities) != 1:
            detail = "; ".join(f"{record['dataset']}={record['model']}@{record['actual_sha256']}" for record in records)
            raise RuntimeError(
                "native-map row tables were generated from different structural models; "
                "a single coherent dynamical map cannot be written: " + detail
            )
        model = Path(str(records[0]["model"]))
        return model, {
            "status": "hash_bound_row_model",
            "model": str(model),
            "model_sha256": records[0]["actual_sha256"],
            "definition": "exact shared structural model used to generate every frozen native row table",
            "row_models": records,
        }

    model_text = str(result.get("model", "") or "").strip()
    model = Path(model_text).expanduser() if model_text else None
    return model, {
        "status": "unverified_legacy_result_model" if model_text else "missing_model",
        "model": str(model) if model is not None else "",
        "definition": "legacy fallback because row summaries did not record inputs.res provenance",
        "row_models": [],
    }


def run_finish_native_dynamic_maps(
    run_dir: Path, result: Mapping[str, object], args: argparse.Namespace
) -> dict[str, object]:
    """Generate native row-level dynamical Fourier maps and coefficient files."""

    row_tables = _finish_native_row_tables(result)
    if not row_tables:
        return {"status": "missing_row_table", "row_table": ""}
    model, model_contract = _finish_native_model_contract(result, args, row_tables)
    if model is None:
        return {"status": "missing_model", "model": ""}
    d_min = getattr(args, "native_map_d_min", 0.0)
    options = NativeDynamicMapOptions(
        grid_spacing_A=float(getattr(args, "native_map_spacing", 0.25)),
        d_min_A=float(d_min) if float(d_min) > 0.0 else None,
        min_i_over_sigma=float(getattr(args, "native_map_min_i_over_sigma", 0.0)),
        max_grid_points=int(getattr(args, "native_map_max_grid_points", 8_000_000)),
    )
    package = write_native_dynamic_map_package(
        row_table=[path for _label, path in row_tables],
        row_table_labels=[label for label, _path in row_tables],
        model_path=model,
        out_dir=run_dir / "native_dynamic_maps",
        options=options,
    )
    package["row_model_contract"] = model_contract
    return package


def run_finish_maps(run_dir: Path, result: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Run requested map families and retain concrete incomplete-output causes."""

    if not getattr(args, "make_maps", True):
        return {"status": "disabled"}
    mode = str(getattr(args, "map_mode", "native") or "native").lower()
    if mode == "off":
        return {"status": "disabled"}
    if mode not in {"native", "shelx2map", "both"}:
        return {"mode": mode, "status": "warning", "warnings": [f"unknown requested map mode: {mode}"]}
    maps: dict[str, object] = {"mode": mode}
    requested = []
    if mode in {"native", "both"}:
        requested.append(("native_dynamic", lambda: run_finish_native_dynamic_maps(run_dir, result, args)))
    if mode in {"shelx2map", "both"}:
        requested.append(("shelx2map", lambda: run_finish_endpoint_maps(result, args)))
    warnings: list[str] = []
    succeeded = 0
    for label, generate in requested:
        try:
            record = generate()
        except Exception as exc:
            record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        maps[label] = record
        if record.get("status") == "ok":
            succeeded += 1
            continue
        details = [str(record.get("status") or "missing map result status")]
        for key in ("error", "note", "endpoint_res", "fcf", "row_table", "model", "executable"):
            if record.get(key):
                details.append(f"{key}: {record[key]}")
        replay = record.get("list6_replay")
        if isinstance(replay, Mapping):
            for key in ("status", "error", "note", "log", "endpoint_hkl"):
                if replay.get(key):
                    details.append(f"LIST 6 {key}: {replay[key]}")
        outputs = record.get("maps")
        if isinstance(outputs, Mapping):
            for name, item in outputs.items():
                if isinstance(item, Mapping) and (item.get("returncode") != 0 or not item.get("exists")):
                    details.append(
                        f"{name}: returncode={item.get('returncode')}, exists={item.get('exists')}, log={item.get('log', '')}"
                    )
        warnings.append(f"{label} maps: " + "; ".join(details))
    maps["status"] = "ok" if succeeded == len(requested) else "partial" if succeeded else "warning"
    maps["warnings"] = warnings
    return maps
