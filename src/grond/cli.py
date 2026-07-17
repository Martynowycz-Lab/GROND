from __future__ import annotations

import argparse
import compileall
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .accelerators import probe_accelerators, probe_bloch_ops
from .io.config import (
    DatasetSpec,
    PrepConfig,
    load_prep,
    parse_dataset_spec,
    resolve_hkl,
    write_datasets_csv,
    write_prep,
)
from .io.tool_config import (
    configured_tools,
    find_tool_config,
    load_tool_config,
    resolve_executable,
    tool_env_updates,
    tool_statuses,
    write_config_template,
)
from .legacy.engine.shelx_diagnostics import parse_shelx_lst_pathologies
from .macromolecule import (
    FREE_FLAG_LABEL,
    export_corrected_mtz,
    infer_cell_and_spacegroup,
    make_free_flags,
    propagate_free_flags,
    read_free_flags,
    render_protein_endpoint_command,
    run_protein_endpoint,
    sha256_file,
)
from .macromolecule import (
    read_table as read_macromolecule_table,
)
from .macromolecule import (
    write_table as write_macromolecule_table,
)
from .maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package
from .paths import LEGACY_ROOT, engine_script
from .progress import ProgressState, format_seconds
from .refine.endpoints import configured_endpoint_blocks, endpoint_report
from .runtime import apply_thread_limits, auto_chunksize, auto_worker_count
from .workflows.dataset_policy import (
    build_dataset_policy,
    policy_by_name,
    write_policy_artifacts,
    write_policy_weighted_runs,
)
from .workflows.decision import (
    apply_preflight,
    expand_transform_candidates,
    preflight_dataset,
    safe_float,
    safe_label,
    score_native_summary,
    should_stop_for_bad_data,
    thickness_polish_nuisance,
    write_json,
)
from .workflows.endpoint_chooser import (
    endpoint_artifact_provenance,
    endpoint_lst_metric_consistency,
    endpoint_promotion_field_contract,
    write_publication_endpoint_choice,
)
from .workflows.scale_contract_selector import select_multi_scale_contract
from .workflows.transform_metadata import metadata_prefilter_transforms

_PROGRESS: ProgressState | None = None
_RUN_LOG_LOCK = threading.Lock()
_TRANSFORM_SCOUT_SEMAPHORE: threading.BoundedSemaphore | None = None
_TRANSFORM_SCOUT_SEMAPHORE_LIMIT = 0
FULL_Q_LAMBDA = "1.0"
DEFAULT_ELECTRON_WAVELENGTH_A = 0.025079
XDS_TARGET_MODES = ("iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy")
XDS_TARGET_MODE_LABELS = {
    "iobs": "IOBS",
    "iobs_over_rlp": "IOBS/RLP",
    "iobs_times_peak": "IOBS*PEAK/100",
    "recorded_proxy": "IOBS*PEAK/(100*RLP)",
}


def configure_progress(
    log_dir: Path,
    *,
    resume: bool,
    total: int,
    invocation_started_monotonic: float | None = None,
    invocation_started_at: float | None = None,
    begin_invocation: bool = False,
) -> None:
    """Attach CLI progress and resume tracking to the current run."""

    global _PROGRESS
    _PROGRESS = ProgressState(
        root=log_dir.parent,
        resume=resume,
        total=total,
        started_at=time.time() if invocation_started_at is None else invocation_started_at,
        invocation_started_monotonic=(
            time.perf_counter() if invocation_started_monotonic is None else invocation_started_monotonic
        ),
    )
    if begin_invocation:
        _PROGRESS.begin_invocation()


def _stage_output_dir(cmd: list[str]) -> Path | None:
    """Find the output directory in a engine command, if it has one."""

    for flag in ("--out-dir", "--out"):
        if flag in cmd:
            try:
                return Path(cmd[cmd.index(flag) + 1])
            except (IndexError, ValueError):
                return None
    return None


def _percent(value: object) -> str:
    """Format a fractional metric as a percentage without raising."""

    try:
        return f"{100.0 * float(value):.3f}%"
    except (TypeError, ValueError):
        return "n/a"


def _print_stage_metrics(cmd: list[str]) -> None:
    """Print the newest raw-row and SHELXL-style statistics for a stage."""

    out_dir = _stage_output_dir(cmd)
    if not out_dir:
        return
    summary = try_read_json(out_dir / "summary.json")
    if not summary:
        return
    best_native = summary.get("best_native") or {}
    metrics = best_native.get("metrics") or {}
    dyn_all = ((metrics.get("all") or {}).get("dyn") or {})
    if dyn_all:
        nuisance = best_native.get("nuisance") or {}
        print(
            "  row I:",
            f"R(I)all={_percent(dyn_all.get('R1_all'))}",
            f"R(I)strong={_percent(dyn_all.get('R1_gt'))}",
            f"wR(I)all={_percent(dyn_all.get('wR2_all'))}",
            f"CC={safe_float(dyn_all.get('CC'), float('nan')):.5f}",
            f"rowB={safe_float(nuisance.get('row_resolution_b'), 0.0):.3g}",
            flush=True,
        )
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    if shelxl:
        lam = safe_float(endpoint.get("lambda_q"), 1.0)
        endpoint_bits = [
            f"B={safe_float(endpoint.get('endpoint_b'), 0.0):.3g}",
            f"EXTI={'on' if endpoint.get('keep_exti') else 'off'}",
        ]
        if abs(lam - 1.0) > 1e-9:
            endpoint_bits.insert(0, f"qexp={lam:.3g}")
        print(
            "  SHELXL endpoint:",
            *endpoint_bits,
            f"R1(F)gt={_percent(shelxl.get('R1_gt'))}",
            f"R1(F)all={_percent(shelxl.get('R1_all'))}",
            f"wR2(F^2)all={_percent(shelxl.get('wR2_all'))}",
            f"Rint={_percent(shelxl.get('Rint'))}",
            f"N={shelxl.get('N_all', shelxl.get('unique', 'n/a'))}",
            flush=True,
        )


def _estimated_forensic_steps(args: argparse.Namespace) -> int:
    """Estimate post-final audit commands for progress reporting."""

    if not getattr(args, "forensic_gate", True):
        return 0
    depth = "quick" if getattr(args, "fast", False) and getattr(args, "forensic_depth", "standard") == "standard" else getattr(args, "forensic_depth", "standard")
    if depth == "quick":
        return 2
    if depth == "full":
        return 9
    return 6


def _estimated_resolution_steps(args: argparse.Namespace) -> int:
    """Estimate resolution-promotion commands for progress reporting."""

    return 1 if getattr(args, "resolution_limit_gate", True) else 0


def parse_numeric_grid(text: str) -> list[float]:
    """Parse a comma/space separated numeric grid while preserving order."""

    values: list[float] = []
    seen: set[float] = set()
    for token in str(text or "").replace(",", " ").replace(";", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def parse_int_grid(text: str) -> list[int]:
    """Parse a comma/space separated integer grid while preserving order."""

    values: list[int] = []
    seen: set[int] = set()
    for token in str(text or "").replace(",", " ").replace(";", " ").split():
        try:
            value = int(float(token))
        except ValueError:
            continue
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def parse_name_grid(text: str) -> list[str]:
    """Parse a comma/space separated label grid while preserving order."""

    values: list[str] = []
    seen: set[str] = set()
    for token in str(text or "").replace(",", " ").replace(";", " ").split():
        value = token.strip().lower()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def format_numeric_grid(values: list[float]) -> str:
    """Render a float list as a GROND engine grid string."""

    return ",".join(f"{float(v):.8g}" for v in values)


def centered_grid(center: float, *, window: float, step: float, lower: float, upper: float) -> list[float]:
    """Return a bounded local grid that always contains the center value."""

    if not math.isfinite(center):
        return []
    step = abs(float(step))
    window = abs(float(window))
    if step <= 0.0:
        step = max(window, 1.0)
    lo = max(float(lower), float(center) - window)
    hi = min(float(upper), float(center) + window)
    values = [float(center)]
    n_steps = int(math.ceil(max(0.0, hi - lo) / step))
    for idx in range(n_steps + 1):
        value = lo + idx * step
        if value <= hi + 1e-9:
            values.append(value)
    values = sorted({round(float(v), 8) for v in values if lower - 1e-9 <= v <= upper + 1e-9})
    return values


def _estimated_sg_window_scout_steps(args: argparse.Namespace) -> int:
    """Estimate late-stage rocking-window scout commands."""

    mode = getattr(args, "sg_window_scout", "auto")
    if mode == "off" or getattr(args, "fast", False):
        return 0
    return len(parse_numeric_grid(getattr(args, "sg_window_scout_grid", "")))


def estimate_run_steps(prep: PrepConfig, args: argparse.Namespace) -> int:
    """Estimate how many external stages this run will launch."""

    total = 1 if _publication_baseline_hkl(prep) else 0
    for ds in prep.datasets:
        if getattr(args, "auto_decide", True) and not getattr(args, "trust_hkl_transform", False):
            transform_candidates = expand_transform_candidates(ds.hkl_transform, args.transform_candidates)
            total += len(transform_candidates)
            rerank_top = max(0, int(getattr(args, "transform_rerank_top", 0)))
            if len(transform_candidates) > 1 and rerank_top > 0:
                total += min(rerank_top, len(transform_candidates))
        total += 1
        if getattr(args, "auto_decide", True) and getattr(args, "hand_control", True):
            total += 2
        if not getattr(args, "fast", False):
            total += 1
            if continuous_thickness_enabled(args):
                # The optimizer works on its bounded diagnostic row subset.  A
                # second stage materializes the selected nuisance on the full
                # native row contract before any downstream gate consumes it.
                total += 2
            total += sum(1 for family in ("row_resolution", "absorption", "sg_jitter", "sigma_error", "dose") if feature_is_enabled(args, family))
            if getattr(args, "mode", "auto") in {"regularize", "auto"}:
                total += 1
            total += 1
        total += 1
        if len(prep.datasets) == 1 and getattr(args, "hand_control", True):
            total += 2
        if getattr(args, "h_omit", "off") in {"on", "both", "only"}:
            total += 2
        if getattr(args, "scale_polish", "auto") != "off":
            total += 1
        total += _estimated_resolution_steps(args)
        total += _estimated_sg_window_scout_steps(args)
        total += _estimated_forensic_steps(args)
    if len(prep.datasets) > 1:
        if getattr(args, "hand_control", True):
            # One shared opposite-model materialization plus one native
            # recomputation and paired-row diagnostic per publication crystal.
            total += 1 + 2 * len(prep.datasets)
        total += 2
        total += _estimated_resolution_steps(args)
    return max(total, 1)


def has_runtime_deps(python: str) -> bool:
    """Return true when a Python executable can import GROND runtime deps."""

    code = "import gemmi, numpy, scipy"
    try:
        return subprocess.run([python, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def choose_python(explicit: str = "") -> str:
    """Choose the Python interpreter used for vendored worker scripts."""

    candidates = [explicit] if explicit else []
    candidates += [sys.executable, shutil.which("python3") or "python3", shutil.which("python") or "python"]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if has_runtime_deps(candidate):
            return candidate
    return explicit or sys.executable


def run_logged(cmd: list[str], log_dir: Path, label: str, *, dry_run: bool = False, cwd: Path = LEGACY_ROOT) -> None:
    """Run one engine command with logs, progress, resume state, and timings."""

    log_dir.mkdir(parents=True, exist_ok=True)
    env = apply_thread_limits(os.environ.copy())
    progress = _PROGRESS
    rec = {
        "label": label,
        "cwd": str(cwd),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": cmd,
    }
    with _RUN_LOG_LOCK:
        if progress and not dry_run:
            if progress.should_skip(label, cmd):
                print(progress.line_prefix(label, skipped=True), flush=True)
                return
            previous = progress.completed_record(label)
            if previous.get("returncode") == 0 and previous.get("command") != cmd:
                print(f"RERUN {label}: previous completed command differs from this command.", flush=True)
        with (log_dir / "commands.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if dry_run:
            print("DRY", " ".join(cmd), flush=True)
            return
        if progress:
            print(progress.line_prefix(label), flush=True)
            progress.mark_started(label, cmd)
        else:
            print("START", label, flush=True)
    t0 = time.perf_counter()
    log_file = log_dir / f"{label}.log"
    try:
        with log_file.open("w") as out:
            out.write("$ " + " ".join(cmd) + "\n\n")
            out.flush()
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=out, stderr=subprocess.STDOUT)
            out.write(f"\n[returncode] {proc.returncode}\n[seconds] {time.perf_counter() - t0:.3f}\n")
    except KeyboardInterrupt:
        with _RUN_LOG_LOCK:
            if progress:
                progress.mark_interrupted(label)
        print(f"\nInterrupted during {label}. Re-run the same command to resume completed stages.", flush=True)
        raise
    done = {
        "label": label,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.perf_counter() - t0,
        "returncode": proc.returncode,
    }
    with _RUN_LOG_LOCK:
        with (log_dir / "commands.jsonl").open("a") as f:
            f.write(json.dumps(done) + "\n")
        if progress:
            progress.mark_completed(label, cmd, done["elapsed_seconds"], proc.returncode, log_file)
        print("DONE", label, "returncode", proc.returncode, "elapsed", format_seconds(done["elapsed_seconds"]), flush=True)
    _print_stage_metrics(cmd)
    if proc.returncode != 0:
        try:
            tail_lines = log_file.read_text(errors="replace").splitlines()[-12:]
        except OSError:
            tail_lines = []
        failure = {
            "status": "failed",
            "label": label,
            "returncode": int(proc.returncode),
            "log": str(log_file.resolve()),
            "cwd": str(Path(cwd).resolve()),
            "command": cmd,
            "log_tail": tail_lines,
            "resume_hint": "Fix the reported cause, then rerun the same GROND command; completed fingerprint-matching stages will resume.",
        }
        failure_text = json.dumps(failure, indent=2) + "\n"
        for failure_path in (
            log_dir / f"{safe_label(label)}.failure.json",
            log_dir / "failure.json",
        ):
            try:
                failure_path.write_text(failure_text)
            except OSError as exc:
                print(
                    f"WARNING: could not write failure artifact {failure_path}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        print(f"FAILED {label} (return code {proc.returncode}). Log: {log_file.resolve()}", flush=True)
        if tail_lines:
            print("Last log lines:", flush=True)
            for line in tail_lines:
                print(f"  {line}", flush=True)
        print(failure["resume_hint"], flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def read_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict when it does not exist."""

    return json.loads(path.read_text()) if path.exists() else {}


def try_read_json(path: Path) -> dict:
    """Read JSON without letting a missing or malformed file stop reporting."""

    try:
        return read_json(path)
    except Exception:
        return {}


def transform_scout_worker_count(args: argparse.Namespace) -> int:
    """Worker processes assigned inside each transform-scout subprocess."""

    return max(1, int(getattr(args, "transform_scout_workers_per_job", 1) or 1))


def transform_scout_job_count(args: argparse.Namespace, task_count: int | None = None) -> int:
    """Maximum number of transform-scout subprocesses GROND should launch at once."""

    requested = int(getattr(args, "transform_scout_jobs", 0) or 0)
    if requested > 0:
        jobs = requested
    else:
        budget = auto_worker_count(int(getattr(args, "workers", 0) or 0), float(getattr(args, "max_core_fraction", 0.85)))
        jobs = max(1, budget // transform_scout_worker_count(args))
    if task_count is not None:
        jobs = min(max(1, int(task_count)), jobs)
    return max(1, int(jobs))


def transform_scout_semaphore(args: argparse.Namespace) -> threading.BoundedSemaphore:
    """Return the global semaphore that bounds concurrent transform-scout subprocesses."""

    global _TRANSFORM_SCOUT_SEMAPHORE, _TRANSFORM_SCOUT_SEMAPHORE_LIMIT
    limit = transform_scout_job_count(args)
    with _RUN_LOG_LOCK:
        if _TRANSFORM_SCOUT_SEMAPHORE is None or limit != _TRANSFORM_SCOUT_SEMAPHORE_LIMIT:
            _TRANSFORM_SCOUT_SEMAPHORE = threading.BoundedSemaphore(limit)
            _TRANSFORM_SCOUT_SEMAPHORE_LIMIT = limit
        return _TRANSFORM_SCOUT_SEMAPHORE


def best_nuisance(run_dir: Path) -> dict:
    """Return the best nuisance-parameter block from a stage summary."""

    summary = read_json(run_dir / "summary.json")
    return ((summary.get("best_native") or {}).get("nuisance") or {})


def resolved_frame_offset_candidates(run_dir: Path) -> list[float]:
    """Return the geometry-resolved frame probes recorded by a native stage."""

    summary = read_json(run_dir / "summary.json")
    grids = summary.get("resolved_grids") if isinstance(summary.get("resolved_grids"), Mapping) else {}
    raw = grids.get("frame_offset_frames", []) if isinstance(grids, Mapping) else []
    values: list[float] = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and not any(abs(value - existing) <= 1e-12 for existing in values):
            values.append(value)
    return values


def shelx_model_is_usable(path: Path | str) -> bool:
    """Return whether a SHELX model path can be reused as a GROND structure."""

    p = Path(path)
    try:
        from .legacy.engine.native_row_regularize import shelx_res_health

        health = shelx_res_health(p, p.with_suffix(".lst"))
    except (ImportError, OSError, TypeError, ValueError):
        return False
    return bool(
        health.get("ok")
        and int(health.get("atom_count", 0) or 0) > 0
        and int(health.get("sfac_count", 0) or 0) > 0
    )


def shelxl_feedback_record_safe_for_reuse(record: Mapping[str, object]) -> bool:
    """Return whether a feedback-loop record is safe as a later starting model."""

    endpoint_status = str(record.get("endpoint_shelxl_status", "")).strip().lower()
    if endpoint_status not in {"ok", "reused_validated_input_model"}:
        return False
    health = record.get("endpoint_res_health")
    if not isinstance(health, Mapping):
        return False
    if not bool(health.get("ok", False)):
        return False
    if health.get("safe_for_feedback") is False:
        return False
    try:
        atom_count = int(health.get("atom_count", 0) or 0)
        sfac_count = int(health.get("sfac_count", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    if atom_count <= 0 or sfac_count <= 0:
        return False
    try:
        npd_count = int(health.get("npd_count", 0) or 0)
    except (TypeError, ValueError):
        return False
    if npd_count > 0 or bool(health.get("has_npd")):
        return False

    output_value = str(record.get("output_model") or "")
    health_path_value = str(health.get("path") or "")
    recorded_sha = str(health.get("sha256") or "")
    if not output_value or not health_path_value or not recorded_sha:
        # Older summaries did not bind the output bytes to their health
        # record.  They are intentionally recomputed rather than reused.
        return False
    output = Path(output_value).expanduser()
    health_path = Path(health_path_value).expanduser()
    try:
        if output.resolve() != health_path.resolve():
            return False
        from .legacy.engine.native_row_regularize import shelx_res_health

        current = shelx_res_health(output, output.with_suffix(".lst"))
    except (ImportError, OSError, TypeError, ValueError):
        return False
    return bool(
        current.get("ok")
        and current.get("safe_for_feedback") is not False
        and int(current.get("atom_count", 0) or 0) > 0
        and int(current.get("sfac_count", 0) or 0) > 0
        and str(current.get("sha256") or "") == recorded_sha
        and int(current.get("npd_count", 0) or 0) <= 0
        and not bool(current.get("has_npd"))
    )


def best_endpoint_model(run_dir: Path) -> Path | None:
    """Return the polished endpoint RES path for a completed native stage."""

    summary = try_read_json(run_dir / "summary.json")
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    lst = shelxl.get("lst")
    if not lst:
        return None
    res = Path(lst).with_suffix(".res")
    record = {
        "output_model": str(res),
        "endpoint_shelxl_status": shelxl.get("status"),
        "endpoint_res_health": shelxl.get("res_health"),
    }
    return res if shelx_model_is_usable(res) and shelxl_feedback_record_safe_for_reuse(record) else None


def shelxl_is_available(executable: str) -> bool:
    """Return true when a SHELXL executable can be launched by worker scripts."""

    exe = str(executable or "shelxl")
    if "/" in exe:
        return Path(exe).exists()
    return shutil.which(exe) is not None


def default_shelxl_executable() -> str:
    """Return the default SHELXL executable name or explicit user override."""

    env_value = os.environ.get("SHELXL")
    if env_value:
        return env_value
    return "shelxl"


def add_tool_config_args(ap: argparse.ArgumentParser) -> None:
    """Attach GROND external-tool config options to a subcommand."""

    ap.add_argument(
        "--config",
        default="",
        help="JSON/YAML file that points GROND to external tools such as SHELXL and shelx2map. Defaults to GROND_CONFIG or grond.config.json in this tree.",
    )
    ap.add_argument(
        "--allow-missing-tools",
        action="store_true",
        help="Record missing optional external tools instead of stopping early. Use this only for native-only diagnostics or deliberate incomplete runs.",
    )


def apply_tool_config(args: argparse.Namespace) -> None:
    """Load a GROND tool config and propagate executable paths into args/env."""

    explicit = getattr(args, "config", "")
    path = find_tool_config(explicit)
    config = load_tool_config(path) if path else {}
    tools = configured_tools(config)
    for env_name, value in tool_env_updates(config).items():
        os.environ[env_name] = value
    if hasattr(args, "python") and not getattr(args, "python", "") and tools.get("python"):
        args.python = tools["python"]
    if hasattr(args, "shelxl") and tools.get("shelxl") and getattr(args, "shelxl", "shelxl") in {"", "shelxl"}:
        args.shelxl = tools["shelxl"]
    if hasattr(args, "shelxl"):
        requested_shelxl = str(getattr(args, "shelxl", "") or "").strip()
        args.shelxl_requested = requested_shelxl
        status = resolve_executable(
            "shelxl",
            configured=requested_shelxl,
            aliases=("shelxl", "shelxl2019"),
        )
        if not status.get("available") and requested_shelxl == "shelxl":
            status = resolve_executable(
                "shelxl",
                aliases=("shelxl", "shelxl2019"),
            )
        if status.get("available") and status.get("resolved"):
            # Workers execute from the vendored engine directory, not from the
            # user's launch directory.  Freeze the preflight resolution now so
            # relative paths and PATH aliases keep the exact same meaning.
            args.shelxl = str(status["resolved"])
    args.tool_config_path = str(path) if path else ""
    args.tool_config = config
    args.configured_tools = tools


def external_tool_statuses(args: argparse.Namespace) -> dict[str, dict]:
    """Return the active external-tool status block for run configs and doctor."""

    statuses = tool_statuses(getattr(args, "tool_config", {}) or {})
    explicit_shelxl = str(getattr(args, "shelxl", "") or "").strip()
    if explicit_shelxl:
        statuses["shelxl"] = resolve_executable(
            "shelxl",
            configured=explicit_shelxl,
            aliases=("shelxl", "shelxl2019"),
        )
        statuses["shelxl"]["source"] = "argument:--shelxl"
        statuses["shelxl"]["requested"] = str(
            getattr(args, "shelxl_requested", explicit_shelxl) or explicit_shelxl
        )
    return statuses


def required_external_tool_errors(args: argparse.Namespace, *, structure: str = "", multi_dataset: bool = False) -> list[str]:
    """Return early-stop messages for tools required by the selected workflow."""

    if getattr(args, "dry_run", False) or getattr(args, "allow_missing_tools", False):
        return []
    statuses = external_tool_statuses(args)
    errors: list[str] = []
    structure_path = structure or getattr(args, "structure", "")
    shelx_like = bool(structure_path and is_shelx_model(structure_path))
    native_only = bool(getattr(args, "native_only", False))
    mode = getattr(args, "mode", "auto")
    endpoint_expected = shelx_like and not native_only and mode in {"bloch", "regularize", "auto"}
    if endpoint_expected and not statuses.get("shelxl", {}).get("available"):
        errors.append("SHELXL is required for corrected-HKL endpoint polishing/statistics; set tools.shelxl in a GROND config, set SHELXL, or use --native-only.")
    if endpoint_expected and getattr(args, "make_maps", False) and not statuses.get("shelx2map", {}).get("available"):
        errors.append("shelx2map is required for automatic map generation; set tools.shelx2map in a GROND config, set SHELX2MAP, or add --no-maps.")
    if (
        getattr(args, "structure_update", "auto") == "shelxl_feedback"
        and not statuses.get("shelxl", {}).get("available")
    ):
        errors.append("The requested --structure-update shelxl_feedback needs SHELXL; set tools.shelxl or choose native_gradient/off.")
    return errors


def raise_for_missing_external_tools(args: argparse.Namespace, *, structure: str = "", multi_dataset: bool = False) -> None:
    """Stop a run with a concise, actionable missing-tool message."""

    errors = required_external_tool_errors(args, structure=structure, multi_dataset=multi_dataset)
    if not errors:
        return
    hint = (
        "Create a template with `GROND init-config --out grond.config.json`, "
        "then rerun with `--config grond.config.json` or set GROND_CONFIG."
    )
    raise SystemExit("Missing required external tools:\n- " + "\n- ".join(errors) + "\n" + hint)


def resolve_structure_update_strategy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Choose the structure-update engine for the current run.

    The SHELXL feedback loop is the preferred default whenever a
    SHELXL executable is available because it reaches the same high-N/Q
    row-space endpoint as finite-difference dynamical gradients at a small
    fraction of the wall time.  Multi-dataset runs use the same lagged LS1
    idea as single-dataset runs: per-dataset Bloch rows are merged into one
    corrected endpoint HKL, one shared SHELXL update is run, and the updated
    shared model is sent back through every dataset.
    """

    requested = getattr(args, "structure_update", "auto")
    if requested in {"off", "native_gradient", "shelxl_feedback"}:
        return requested
    if getattr(args, "fast", False) or getattr(args, "full_fast", False) or getattr(args, "native_only", False):
        return "native_gradient"
    if shelxl_is_available(getattr(args, "shelxl", "shelxl")):
        return "shelxl_feedback"
    return "native_gradient"


def shelxl_feedback_current_model(run_dir: Path) -> Path | None:
    """Return the latest model produced by a SHELXL feedback-loop run."""

    summary = try_read_json(run_dir / "shelxl_feedback_loop_summary.json")
    runs = summary.get("runs") or []
    current = summary.get("current_model")
    if current and shelx_model_is_usable(current):
        current_path = Path(str(current))
        if current_path == Path(str(summary.get("start_res", ""))) or any(
            rec.get("output_model") and Path(str(rec.get("output_model"))) == current_path and shelxl_feedback_record_safe_for_reuse(rec)
            for rec in runs
        ):
            return current_path
    for rec in reversed(runs):
        output = rec.get("output_model")
        if output and shelx_model_is_usable(output) and shelxl_feedback_record_safe_for_reuse(rec):
            return Path(output)
    return None


def shelxl_feedback_current_run(run_dir: Path) -> Path | None:
    """Return the run directory corresponding to the latest usable feedback model."""

    summary = try_read_json(run_dir / "shelxl_feedback_loop_summary.json")
    current = summary.get("current_model")
    for rec in reversed(summary.get("runs") or []):
        rec_run = rec.get("run_dir")
        if not rec_run:
            continue
        output = rec.get("output_model")
        if output and shelx_model_is_usable(output) and shelxl_feedback_record_safe_for_reuse(rec):
            return Path(rec_run).expanduser()
        if current and output and Path(output) == Path(str(current)) and shelx_model_is_usable(current) and shelxl_feedback_record_safe_for_reuse(rec):
            return Path(rec_run).expanduser()
    return None


def resolved_post_bloch_sigma_policy(args: argparse.Namespace, *, final_endpoint: bool = False, feedback_loop: bool = False) -> str:
    """Resolve the endpoint sigma calibration policy for a workflow stage."""

    requested = getattr(args, "post_bloch_sigma_policy", "auto")
    if requested == "auto":
        return "uniform_z_rms" if (final_endpoint or feedback_loop) else "none"
    return str(requested)


def resolved_dataset_sigma_policy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Resolve per-dataset endpoint sigma calibration for combination stages."""

    requested = getattr(args, "dataset_sigma_policy", "auto")
    if requested == "auto":
        return "uniform_z_rms" if multi_dataset else "none"
    return str(requested)


def resolved_merge_sigma_policy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Resolve the sigma model used when corrected endpoint rows are merged."""

    requested = getattr(args, "merge_sigma_policy", "auto")
    if requested == "auto":
        return "chisq_overdispersion" if multi_dataset else "formal"
    return str(requested)


def should_run_endpoint_feedback(run_dir: Path, args: argparse.Namespace) -> tuple[bool, dict]:
    """Decide whether a promoted endpoint factor should trigger one more cycle."""

    if getattr(args, "endpoint_feedback", "auto") == "off":
        return False, {"status": "disabled"}
    summary = try_read_json(run_dir / "summary.json")
    decision = summary.get("endpoint_factor_decision") or {}
    if getattr(args, "endpoint_feedback", "auto") == "on":
        return True, decision or {"status": "forced"}
    return bool(decision.get("promote_endpoint_resolution_scale", False)), decision


def forensic_command_plan(
    python: str,
    row_table: Path,
    hkl: str,
    res: str,
    out_dir: Path,
    *,
    depth: str,
    bins: int,
    top_rows: int,
    top_groups: int,
) -> list[tuple[str, list[str]]]:
    """Build the row-space forensic command list for a completed row table."""

    common = ["--row-table", str(row_table), "--hkl", hkl, "--res", res]
    commands: list[tuple[str, list[str]]] = [
        (
            "residual_compass",
            [
                python,
                str(engine_script("residual_compass.py")),
                *common,
                "--out-dir",
                str(out_dir / "residual_compass"),
                "--bins",
                str(bins),
                "--top-rows",
                str(top_rows),
            ],
        ),
        (
            "premerge_rfactor",
            [
                python,
                str(engine_script("premerge_rfactor_audit.py")),
                *common,
                "--out-dir",
                str(out_dir / "premerge_rfactor"),
                "--bins",
                str(bins),
                "--top-groups",
                str(top_groups),
            ],
        ),
    ]
    if depth in {"standard", "full"}:
        commands.extend(
            [
                (
                    "dose_damage",
                    [
                        python,
                        str(engine_script("native_dose_damage_audit.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / "dose_damage"),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                    ],
                ),
                (
                    "two_part_signed",
                    [
                        python,
                        str(engine_script("native_two_part_observation_correction.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / "two_part_signed"),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                        "--holdout-mode",
                        "signed_hkl",
                        "--holdout-mod",
                        "10",
                        "--holdout-value",
                        "0",
                        "--frame-basis",
                        "6",
                        "--xy-basis",
                        "4",
                        "--alternating-iters",
                        "2",
                    ],
                ),
                (
                    "two_part_frame_mod",
                    [
                        python,
                        str(engine_script("native_two_part_observation_correction.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / "two_part_frame_mod"),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                        "--holdout-mode",
                        "frame_mod",
                        "--holdout-mod",
                        "5",
                        "--holdout-value",
                        "2",
                        "--frame-holdout-width",
                        "30",
                        "--frame-basis",
                        "6",
                        "--xy-basis",
                        "4",
                        "--alternating-iters",
                        "2",
                    ],
                ),
            ]
        )
    if depth == "full":
        for label, value in (("two_part_frame_block_early", "0"), ("two_part_frame_block_late", "9")):
            commands.append(
                (
                    label,
                    [
                        python,
                        str(engine_script("native_two_part_observation_correction.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / label),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                        "--holdout-mode",
                        "frame_block",
                        "--holdout-mod",
                        "10",
                        "--holdout-value",
                        value,
                        "--frame-basis",
                        "6",
                        "--xy-basis",
                        "4",
                        "--alternating-iters",
                        "2",
                    ],
                )
            )
        commands.append(
            (
                "sigma_replicate",
                [
                    python,
                    str(engine_script("native_sigma_replicate_audit.py")),
                    *common,
                    "--out-dir",
                    str(out_dir / "sigma_replicate"),
                    "--pred-column",
                    "dyn_scaled",
                    "--base-sigma-column",
                    "xds_sigma",
                ],
            )
        )
    return commands


def summarize_forensic_gate(out_dir: Path, row_table: Path, hkl: str, res: str, depth: str) -> dict:
    """Summarize forensic outputs and decide what they recommend."""

    residual = try_read_json(out_dir / "residual_compass" / "summary.json")
    premerge = try_read_json(out_dir / "premerge_rfactor" / "summary.json")
    dose = try_read_json(out_dir / "dose_damage" / "summary.json")
    signed = try_read_json(out_dir / "two_part_signed" / "summary.json")
    frame_mod = try_read_json(out_dir / "two_part_frame_mod" / "summary.json")
    early = try_read_json(out_dir / "two_part_frame_block_early" / "summary.json")
    late = try_read_json(out_dir / "two_part_frame_block_late" / "summary.json")
    observation_polish = try_read_json(out_dir / "observation_polish" / "summary.json")
    recommendations: list[str] = []
    dyn_all = (((residual.get("model_summary") or {}).get("dyn") or {}).get("all") or {})
    dyn_strong_gap = safe_float(dyn_all.get("R1_all"), 0.0) - safe_float(dyn_all.get("R1_gt"), 0.0)
    if dyn_strong_gap > 0.04:
        recommendations.append("R1all is much worse than R1strong; branch to weak-row, resolution, and observation-correction diagnostics before adding beams")
    scale_pressure = ((((residual.get("raw_scale_summary") or {}).get("dyn") or {}).get("scale_pressure")) or {})
    frame_corr = safe_float((scale_pressure.get("log_scale_correlations") or {}).get("frame_zcal"), 0.0)
    if abs(frame_corr) > 0.4:
        recommendations.append("row scale is strongly frame-correlated; promote explicit frame nuisance/background/fluence checks")
    pmode = premerge.get("primary_group_mode")
    psummary = ((premerge.get("group_modes") or {}).get(pmode or "") or {})
    if psummary:
        scalar = psummary.get("group_floor") or {}
        dyn = psummary.get("dyn") or {}
        if safe_float(scalar.get("R2_all"), 0.0) > safe_float(dyn.get("R2_all"), 0.0) + 0.02:
            recommendations.append("row-specific Bloch predictions beat scalar equivalent-row collapse; keep pre-merge row modeling")
    for item in premerge.get("recommendations") or []:
        if "equivalent-row modulation is anti-correlated" in item and item not in recommendations:
            recommendations.append(item)
    if dose:
        accepted = bool(dose.get("accept_dose_b_grid", False))
        if not accepted:
            recommendations.append("dose-B gate did not pass; keep dose terms off unless a later residual table shows a significant slope")
    signed_pass = bool(signed.get("passes_heldout_gate", False))
    frame_pass = bool(frame_mod.get("passes_heldout_gate", False))
    if signed_pass and frame_pass:
        recommendations.append("smooth additive observation correction passed signed-HKL and interleaved-frame gates; keep as candidate final polish")
    elif signed or frame_mod:
        recommendations.append("observation correction did not pass both main holdouts; keep diagnostic-only")
    if early and late and (not bool(early.get("passes_heldout_gate", False)) or not bool(late.get("passes_heldout_gate", False))):
        recommendations.append("edge-frame holdout is not uniformly stable; require stronger regularization before default endpoint promotion")
    if observation_polish:
        if bool(observation_polish.get("accepted", False)):
            endpoint = observation_polish.get("best_endpoint") or {}
            sh = endpoint.get("shelxl") or {}
            recommendations.append(
                "observation polish passed raw-row and SHELXL endpoint gates; promote polished corrected-HKL endpoint"
            )
            if sh:
                lam = safe_float(endpoint.get("lambda_q"), 1.0)
                q_note = f" qexp={lam:.3g}," if abs(lam - 1.0) > 1e-9 else ""
                recommendations.append(
                    f"best polished endpoint:{q_note} R1all={100.0 * safe_float(sh.get('R1_all'), float('nan')):.2f}%, "
                    f"wR2all={100.0 * safe_float(sh.get('wR2_all'), float('nan')):.2f}%"
                )
        else:
            recommendations.append("observation polish was tested but not promoted because the raw-row or SHELXL endpoint gate did not pass")
    summary = {
        "row_table": str(row_table.resolve()),
        "hkl": hkl,
        "res": res,
        "depth": depth,
        "dyn_all": dyn_all,
        "premerge_primary_mode": pmode,
        "two_part_signed_pass": signed_pass if signed else None,
        "two_part_frame_mod_pass": frame_pass if frame_mod else None,
        "dose_gate_pass": bool(dose.get("accept_dose_b_grid", False)) if dose else None,
        "observation_polish_accepted": bool(observation_polish.get("accepted", False)) if observation_polish else None,
        "best_observation_polish_endpoint": observation_polish.get("best_endpoint", {}) if observation_polish else {},
        "recommendations": recommendations,
        "artifacts": {
            "residual_compass": str((out_dir / "residual_compass" / "REPORT.md").resolve()),
            "premerge_rfactor": str((out_dir / "premerge_rfactor" / "REPORT.md").resolve()),
            "dose_damage": str((out_dir / "dose_damage" / "REPORT.md").resolve()) if dose else "",
            "two_part_signed": str((out_dir / "two_part_signed" / "REPORT.md").resolve()) if signed else "",
            "two_part_frame_mod": str((out_dir / "two_part_frame_mod" / "REPORT.md").resolve()) if frame_mod else "",
            "observation_polish": str((out_dir / "observation_polish" / "REPORT.md").resolve()) if observation_polish else "",
        },
    }
    (out_dir / "forensic_gate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    return summary


def should_run_observation_polish(summary: dict, args: argparse.Namespace, depth: str) -> bool:
    """Return true when the raw-row correction passed enough gates to test."""

    mode = getattr(args, "observation_polish", "auto")
    if mode == "off":
        return False
    if depth == "quick":
        return False
    signed_pass = bool(summary.get("two_part_signed_pass", False))
    frame_pass = bool(summary.get("two_part_frame_mod_pass", False))
    if mode == "on":
        return signed_pass and frame_pass
    return signed_pass and frame_pass


def observation_polish_cmd(python: str, row_table: Path, hkl: str, res: str, out_dir: Path, args: argparse.Namespace) -> list[str]:
    """Build the command that turns a passing observation correction into HKLs."""

    lambda_grid = getattr(args, "observation_polish_lambda_grid", "") or getattr(args, "lambda_grid", FULL_Q_LAMBDA)
    cmd = [
        python,
        str(engine_script("native_observation_polish_endpoint.py")),
        "--row-table",
        str(row_table),
        "--hkl",
        hkl,
        "--res",
        res,
        "--out-dir",
        str(out_dir / "observation_polish"),
        "--candidate-dir",
        str(out_dir / "two_part_signed"),
        "--candidate-dir",
        str(out_dir / "two_part_frame_mod"),
        "--combine-mode",
        getattr(args, "observation_polish_combine", "ensemble"),
        "--lambda-grid",
        lambda_grid,
        "--rescale",
        getattr(args, "rescale", "median-iobs"),
        "--hkl-float-format",
        getattr(args, "hkl_float_format", "shelx_fixed"),
        "--min-raw-wR2-gain",
        str(getattr(args, "observation_polish_min_raw_wR2_gain", 0.002)),
        "--endpoint-wR2-tolerance",
        str(getattr(args, "observation_polish_endpoint_wR2_tolerance", 0.0005)),
        "--shelxl",
        str(getattr(args, "shelxl", "") or default_shelxl_executable()),
        "--shelxl-iterations",
        str(getattr(args, "shelxl_iterations", 5)),
    ]
    if getattr(args, "forensic_maps", False) or getattr(args, "make_maps", False):
        cmd.append("--make-maps")
    return cmd


def run_forensic_gate(
    python: str,
    row_table: Path,
    hkl: str,
    res: str,
    out_dir: Path,
    logs: Path,
    label_prefix: str,
    args: argparse.Namespace,
) -> dict:
    """Run row-space forensics and optional observation-polish promotion."""

    if not getattr(args, "forensic_gate", True):
        return {}
    row_table = Path(row_table)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not row_table.exists():
        return {"status": "missing_row_table", "row_table": str(row_table)}
    depth = "quick" if getattr(args, "fast", False) and args.forensic_depth == "standard" else args.forensic_depth
    commands = forensic_command_plan(
        python,
        row_table,
        hkl,
        res,
        out_dir,
        depth=depth,
        bins=args.forensic_bins,
        top_rows=args.forensic_top_rows,
        top_groups=args.forensic_top_groups,
    )
    for name, cmd in commands:
        run_logged(cmd, logs, f"{label_prefix}_forensics_{name}", dry_run=args.dry_run)
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "commands": len(commands)}
    summary = summarize_forensic_gate(out_dir, row_table, hkl, res, depth)
    if should_run_observation_polish(summary, args, depth):
        cmd = observation_polish_cmd(python, row_table, hkl, res, out_dir, args)
        try:
            run_logged(cmd, logs, f"{label_prefix}_forensics_observation_polish", dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            if getattr(args, "observation_polish", "auto") == "on":
                raise
            summary["observation_polish_error"] = {"returncode": exc.returncode, "command": exc.cmd}
            (out_dir / "forensic_gate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
            return summary
        summary = summarize_forensic_gate(out_dir, row_table, hkl, res, depth)
    return summary


def dataset_baseline_hkl(prep: PrepConfig, ds: DatasetSpec) -> str:
    """Return the dataset-specific baseline HKL or the prep-level fallback."""

    return ds.baseline_hkl or prep.baseline_hkl


def _cif_number(text: str) -> int | None:
    """Parse a CIF integer field that may carry uncertainty notation."""

    cleaned = str(text or "").strip().strip("'\"")
    if not cleaned or cleaned in {".", "?"}:
        return None
    cleaned = cleaned.split("(", 1)[0]
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _is_number(text: str) -> bool:
    """Return true when text can be parsed as a floating-point number."""

    try:
        float(str(text).replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def _shelx_cell_wavelength_status(value: object) -> dict[str, object]:
    """Classify a SHELX CELL wavelength for GROND's electron endpoint contract."""

    try:
        wavelength = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return {
            "status": "warning",
            "wavelength_A": None,
            "endpoint_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
            "message": "CELL wavelength is not numeric; endpoint INS files will use the GROND electron default.",
        }
    if not math.isfinite(wavelength) or wavelength <= 0.0 or abs(wavelength - 1.0) < 1e-6 or wavelength >= 0.1:
        return {
            "status": "warning",
            "wavelength_A": wavelength,
            "endpoint_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
            "message": "CELL wavelength is a placeholder or non-electron value; endpoint INS files will use the GROND electron default.",
        }
    return {
        "status": "ok",
        "wavelength_A": wavelength,
        "endpoint_wavelength_A": wavelength,
        "message": "CELL wavelength is in the expected electron range.",
    }


def check_sfac_contract(structure: str | Path) -> dict[str, object]:
    """Check whether a model declares explicit numeric SFAC rows for GROND.

    SHELXL accepts element-only lines such as ``SFAC C H N O`` by using its
    built-in scattering table.  GROND's Bloch calculator needs explicit
    electron-scattering coefficient rows, so fail this case early and clearly.
    """

    path = Path(structure).expanduser()
    report: dict[str, object] = {
        "schema_version": 1,
        "structure": str(path),
        "status": "ok",
        "ok": True,
        "default_electron_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
        "sfac_rows": [],
        "messages": [],
    }
    messages: list[dict[str, str]] = []
    sfac_rows: list[dict[str, object]] = []
    endpoint_contract: dict[str, object] = {
        "hklf": None,
        "hklf_status": "missing",
        "stripped_endpoint_commands": [],
        "unsupported_features": [],
    }
    suffix = path.suffix.lower()
    if suffix == ".cif":
        try:
            embedded_shelx = _extract_cif_shelx_block(path)
        except Exception as exc:
            report.update(
                status="error",
                ok=False,
                messages=[
                    {
                        "severity": "error",
                        "message": f"could not read CIF model: {exc}",
                    }
                ],
            )
            return report
        if embedded_shelx:
            with tempfile.TemporaryDirectory(prefix="grond_sfac_check_") as tmp:
                embedded_path = Path(tmp) / "embedded.res"
                embedded_path.write_text(embedded_shelx)
                embedded_report = check_sfac_contract(embedded_path)
            embedded_report["structure"] = str(path)
            embedded_report["cif_model_contract"] = "embedded_shelx"
            embedded_report.setdefault("messages", []).insert(
                0,
                {
                    "severity": "info",
                    "message": "Checked the complete SHELX instruction block embedded in the CIF.",
                },
            )
            return embedded_report
        try:
            from .legacy.engine.bloch_engine import parse_cif

            model = parse_cif(path)
        except (OSError, RuntimeError, ValueError) as exc:
            report.update(
                status="error",
                ok=False,
                messages=[
                    {
                        "severity": "error",
                        "message": f"CIF is not a usable electron-scattering model: {exc}",
                    }
                ],
            )
            return report
        report.update(
            status="warning",
            ok=True,
            cif_model_contract="explicit_electron_coefficients",
            sfac_rows=[
                {
                    "element": str(row["element"]),
                    "kind": "numeric_coefficients",
                    "coefficient_count": len(row.get("raw", [])),
                }
                for row in model["sfac"]
            ],
            cell={
                "status": "warning",
                "wavelength_A": None,
                "endpoint_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
                "message": "CIF has no SHELX CELL wavelength; generated endpoint files use the GROND electron default.",
            },
            messages=[
                {
                    "severity": "warning",
                    "message": "The CIF has explicit electron coefficients and is usable by the lower-level Bloch parser, but GROND run needs a complete embedded SHELX instruction block for model-update and endpoint files. Provide that block or use a RES/INS model.",
                }
            ],
        )
        return report
    if suffix not in {".res", ".ins"}:
        report.update(
            status="warning",
            ok=True,
            messages=[{"severity": "warning", "message": "SFAC check only audits SHELX .res/.ins models."}],
        )
        return report
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return {
            **report,
            "status": "error",
            "ok": False,
            "messages": [{"severity": "error", "message": f"could not read model: {exc}"}],
        }

    cell_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("!", "#")):
            continue
        parts = line.split()
        head = parts[0].upper()
        if head == "ACTA":
            endpoint_contract["stripped_endpoint_commands"].append({"line": raw, "command": "ACTA"})
            messages.append(
                {
                    "severity": "warning",
                    "message": "ACTA is accepted in the input model but stripped from GROND endpoint files because it can interfere with zero-cycle/check runs and is not needed for corrected-HKL refinement.",
                }
            )
            continue
        if head == "HKLF":
            hklf = parts[1] if len(parts) > 1 else ""
            endpoint_contract["hklf"] = hklf
            if hklf == "4":
                endpoint_contract["hklf_status"] = "ok"
            elif hklf == "3":
                endpoint_contract["hklf_status"] = "source_amplitudes_endpoint_forced_hklf4"
                messages.append(
                    {
                        "severity": "warning",
                        "message": "Input model uses HKLF 3 amplitudes. GROND can read the model, but every corrected endpoint is written as HKLF 4 intensities.",
                    }
                )
            else:
                endpoint_contract["hklf_status"] = "unsupported"
                endpoint_contract["unsupported_features"].append({"line": raw, "feature": f"HKLF {hklf or '?'}"})
                messages.append(
                    {
                        "severity": "error",
                        "message": f"HKLF {hklf or '?'} is not supported. GROND currently writes corrected intensity endpoints as HKLF 4; HKLF 5/twin endpoints are a current limitation.",
                    }
                )
            continue
        if head in {"BASF", "TWIN", "SWAT"}:
            endpoint_contract["unsupported_features"].append({"line": raw, "feature": head})
            messages.append(
                {
                    "severity": "error",
                    "message": f"{head} is not supported by GROND corrected-HKLF4 endpoints yet. Twin/BASF/SWAT refinement is a current limitation.",
                }
            )
            continue
        if head == "SQUEEZE" or "SQUEEZE" in line.upper():
            endpoint_contract["unsupported_features"].append({"line": raw, "feature": "SQUEEZE"})
            messages.append(
                {
                    "severity": "error",
                    "message": "SQUEEZE/solvent-mask corrected SHELX models are not supported by GROND corrected-HKLF4 endpoints yet.",
                }
            )
            continue
        if head == "CELL":
            cell_seen = True
            if len(parts) >= 2:
                status = _shelx_cell_wavelength_status(parts[1])
                report["cell"] = status
                if status["status"] != "ok":
                    messages.append({"severity": "warning", "message": str(status["message"])})
            else:
                messages.append({"severity": "error", "message": "CELL line is missing a wavelength."})
            continue
        if head != "SFAC":
            if head == "END":
                break
            continue
        if len(parts) < 2:
            sfac_rows.append({"line": raw, "status": "error", "reason": "empty_sfac"})
            messages.append({"severity": "error", "message": "SFAC line is empty."})
            continue
        if len(parts) >= 3 and _is_number(parts[2]):
            element = parts[1].capitalize()
            coeff_count = len(parts) - 2
            row_status = "ok" if coeff_count >= 8 else "error"
            row = {
                "line": raw,
                "status": row_status,
                "kind": "numeric_coefficients",
                "element": element,
                "coefficient_count": coeff_count,
            }
            sfac_rows.append(row)
            if coeff_count < 8:
                messages.append(
                    {
                        "severity": "error",
                        "message": f"SFAC {element} has {coeff_count} numeric coefficients; GROND needs explicit electron rows with at least 8 numbers.",
                    }
                )
            continue
        elements = [token.capitalize() for token in parts[1:]]
        sfac_rows.append({"line": raw, "status": "error", "kind": "element_only", "elements": elements})
        messages.append(
            {
                "severity": "error",
                "message": "Element-only SHELX SFAC lines use built-in SHELXL factors. Replace them with explicit electron scattering coefficient rows before running GROND.",
            }
        )
    if not cell_seen:
        messages.append({"severity": "error", "message": "No CELL line was found."})
    if not sfac_rows:
        messages.append({"severity": "error", "message": "No SFAC rows were found."})
    report["sfac_rows"] = sfac_rows
    report["endpoint_contract"] = endpoint_contract
    report["messages"] = messages
    has_error = any(item.get("severity") == "error" for item in messages) or any(row.get("status") == "error" for row in sfac_rows)
    has_warning = any(item.get("severity") == "warning" for item in messages)
    report["ok"] = not has_error
    report["status"] = "error" if has_error else ("warning" if has_warning else "ok")
    return report


def estimate_model_nonh_atoms(structure: str) -> int | None:
    """Estimate the number of non-hydrogen atoms in a CIF or SHELX model.

    This intentionally avoids deep structure interpretation. The estimate is
    used only for CLI scheduling, so a conservative count is enough to choose a
    fast structure-update budget.
    """

    path = Path(structure).expanduser()
    suffix = path.suffix.lower()
    try:
        if suffix == ".cif":
            import gemmi

            block = gemmi.cif.read_file(str(path)).sole_block()
            type_col = block.find_loop("_atom_site_type_symbol")
            label_col = block.find_loop("_atom_site_label")
            if type_col:
                return sum(1 for item in type_col if not str(item).strip().upper().startswith("H"))
            if label_col:
                return sum(1 for item in label_col if not str(item).strip().upper().startswith("H"))
            return None

        if suffix not in {".res", ".ins"}:
            return None
        sfac: list[str] = []
        count = 0
        controls = {
            "TITL",
            "CELL",
            "ZERR",
            "LATT",
            "SYMM",
            "SFAC",
            "UNIT",
            "FVAR",
            "WGHT",
            "EXTI",
            "MERG",
            "FMAP",
            "PLAN",
            "L.S.",
            "CGLS",
            "ACTA",
            "BOND",
            "CONF",
            "HTAB",
            "LIST",
            "TEMP",
            "SIZE",
            "OMIT",
            "SHEL",
            "BASF",
            "TWIN",
            "SWAT",
            "HKLF",
            "END",
            "REM",
            "AFIX",
            "PART",
            "RESI",
            "SAME",
            "SIMU",
            "RIGU",
            "DELU",
            "ISOR",
            "EADP",
            "EXYZ",
            "DFIX",
            "DANG",
            "SADI",
            "FLAT",
            "CHIV",
            "ANIS",
        }
        for raw in path.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith(("!", "#")):
                continue
            parts = line.split()
            head = parts[0].upper()
            if head == "SFAC":
                sfac = [p.capitalize() for p in parts[1:]]
                continue
            if head in controls or len(parts) < 5:
                continue
            if not parts[1].lstrip("+-").isdigit():
                continue
            if not (_is_number(parts[2]) and _is_number(parts[3]) and _is_number(parts[4])):
                continue
            sfac_index = int(parts[1])
            element = sfac[sfac_index - 1] if 0 < sfac_index <= len(sfac) else parts[0].lstrip("0123456789").capitalize()
            if not element.upper().startswith("H") and not parts[0].upper().startswith("Q"):
                count += 1
        return count or None
    except Exception:
        return None


def absolute_structure_policy(structure: str) -> dict:
    """Decide whether an alternate hand check makes sense for a model."""

    path = Path(structure)
    suffix = path.suffix.lower()
    policy = {
        "structure": str(path),
        "run_hand_control": True,
        "run_absolute_structure_control": True,
        "reason": "alternate hand check may be informative for acentric models",
    }
    try:
        if suffix in {".res", ".ins"}:
            shelx_lines = path.read_text(errors="replace").splitlines()
            chiv_lines = [
                line.strip()
                for line in shelx_lines
                if line.strip().split() and line.strip().split()[0].upper() == "CHIV"
            ]
            if chiv_lines:
                policy.update(
                    run_hand_control=False,
                    run_absolute_structure_control=False,
                    reason=(
                        "SHELX model contains signed CHIV restraints; automatic hand inversion is disabled "
                        "because GROND does not yet transform CHIV targets"
                    ),
                    unsupported_hand_records=chiv_lines,
                )
                return policy
            for line in shelx_lines:
                parts = line.strip().split()
                if parts and parts[0].upper() == "LATT" and len(parts) > 1:
                    latt = int(float(parts[1]))
                    policy["shelx_latt"] = latt
                    if latt > 0:
                        policy.update(
                            run_hand_control=False,
                            run_absolute_structure_control=False,
                            reason="SHELX LATT is centrosymmetric; alternate hand check is not meaningful",
                        )
                    else:
                        policy.update(
                            run_hand_control=True,
                            run_absolute_structure_control=True,
                            reason="SHELX LATT is acentric; alternate hand check is applicable",
                        )
                    return policy
            policy["reason"] = "no SHELX LATT record found; keeping alternate hand control enabled"
            return policy
        if suffix == ".cif":
            import gemmi

            block = gemmi.cif.read_file(str(path)).sole_block()
            number = _cif_number(block.find_value("_space_group_IT_number") or block.find_value("_symmetry_Int_Tables_number") or "")
            if number is not None:
                sg = gemmi.find_spacegroup_by_number(number)
                if sg is not None:
                    policy["space_group_number"] = number
                    policy["space_group_hm"] = str(sg.hm)
                    if sg.is_centrosymmetric():
                        policy.update(
                            run_hand_control=False,
                            run_absolute_structure_control=False,
                            reason="CIF space group is centrosymmetric; alternate hand check is not meaningful",
                        )
                    else:
                        policy.update(
                            run_hand_control=True,
                            run_absolute_structure_control=True,
                            reason="CIF space group is acentric; alternate hand check is applicable",
                        )
                    return policy
            for tag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
                col = block.find_loop(tag)
                if not col:
                    continue
                loop = col.get_loop()
                tags = [str(t) for t in loop.tags]
                tag_i = tags.index(tag)
                for i in range(loop.length()):
                    op = gemmi.Op(str(loop[i, tag_i]).strip().strip("'\""))
                    den = int(op.DEN)
                    inv = [[-den, 0, 0], [0, -den, 0], [0, 0, -den]]
                    if [list(row) for row in op.rot] == inv:
                        policy.update(
                            run_hand_control=False,
                            run_absolute_structure_control=False,
                            reason="CIF symmetry operators include inversion; alternate hand check is not meaningful",
                        )
                        return policy
            return policy
    except Exception as exc:
        policy.update(reason=f"could not determine hand-control symmetry policy ({exc}); keeping alternate hand control enabled")
        return policy
    return policy


def _clean_embedded_shelx(text: str) -> str:
    """Normalize a SHELX instruction block extracted from a CIF text field."""

    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and lines[0].strip() == ";":
        lines.pop(0)
    while lines and lines[-1].strip() == ";":
        lines.pop()
    if not any(line.strip().upper() == "END" for line in lines):
        lines.append("END")
    return "\n".join(lines) + "\n"


def _extract_cif_shelx_block(path: Path) -> str:
    """Extract an embedded SHELX .res/.ins block from an CIF."""

    import gemmi

    block = gemmi.cif.read_file(str(path)).sole_block()
    iucr_shelx_tag = "_iucr_" + "ref" + "ine_instructions_details"
    for tag in (iucr_shelx_tag, "_shelx_res_file"):
        value = block.find_value(tag)
        if value and "CELL" in str(value).upper() and "SFAC" in str(value).upper():
            return _clean_embedded_shelx(str(value))
    return ""


def prepare_structure_for_legacy(structure: str, out_dir: Path) -> tuple[str, dict]:
    """Return a SHELX-compatible structure path for engine regularization scripts.

    The Bloch calculator can read CIF coordinates directly, but map generation
    and native structure update currently preserve and rewrite SHELX records.
    Some CIF exports carry exact SHELX instructions in a text block;
    extracting that block is safer than rebuilding AFIX/WGHT/SFAC records.
    """
    src = Path(structure).expanduser().resolve()
    if src.suffix.lower() != ".cif":
        return str(src), {"source": str(src), "prepared": False}
    prepared_dir = out_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    dst = prepared_dir / f"{src.stem}.res"
    shelx = _extract_cif_shelx_block(src)
    if not shelx:
        raise RuntimeError(
            f"{src} is a CIF, but no embedded SHELX model-update block was found. "
            "Provide a .res/.ins model or a CIF that embeds SHELX instruction text."
        )
    dst.write_text(shelx)
    return str(dst), {"source": str(src), "prepared": True, "prepared_structure": str(dst), "method": "embedded_shelx_block"}


def resolve_final_n_basis(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    structure: str,
    stage: str,
    *,
    alternate_hand: bool = False,
) -> int:
    """Choose the final Bloch basis size from a cheap support scan."""

    if args.fast:
        return 20
    if int(args.n_basis) > 0:
        return int(args.n_basis)
    if getattr(args, "full_fast", False):
        return int(args.full_fast_n_basis)
    if args.dry_run:
        return int(args.final_basis_fallback)
    scan = root / "decisions" / ds.name / f"{safe_label(stage)}_support_basis.json"
    cmd = [
        python,
        str(engine_script("support_basis_scan.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        f"--hkl-transform={ds.hkl_transform}",
        "--out",
        str(scan),
        "--n-basis-limit",
        str(args.final_basis_scan_limit),
        "--margin",
        str(args.final_basis_margin),
        "--min-final",
        str(args.final_basis_min),
        "--max-final",
        str(args.final_basis_max),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
    ]
    cmd.append("--include-nonpositive" if ds.include_nonpositive else "--positive-only")
    if args.max_rows:
        cmd += ["--max-rows", str(args.max_rows)]
    if alternate_hand:
        cmd.append("--alternate-hand")
    run_logged(cmd, logs, f"{ds.name}_{safe_label(stage)}_support_basis", dry_run=args.dry_run)
    payload = read_json(scan)
    n_basis = int(payload.get("recommended_n_basis") or args.final_basis_fallback)
    return max(int(args.final_basis_min), min(int(args.final_basis_max), n_basis))


def first(value: str, fallback: str) -> str:
    """Return the first comma-separated grid value or a fallback."""

    text = str(value or "").strip()
    return text.split(",")[0] if text else fallback


def grid_from_nuisance(nui: dict, key: str, fallback: str) -> str:
    """Return a single-value grid string from a pinned nuisance parameter."""

    value = nui.get(key, fallback)
    try:
        return f"{float(value):.15g}"
    except (TypeError, ValueError):
        return str(value or fallback)


def grid_has_variation(text: str) -> bool:
    """Return true when a grid string contains more than one unique value."""

    parts = [p for p in str(text or "").replace(",", " ").split() if p]
    return len(dict.fromkeys(parts)) > 1


def compact_base_nuisance(args: argparse.Namespace) -> dict:
    """Build the starting nuisance grid for fast or full automatic runs."""

    if args.fast:
        return {
            "thickness_A": "1000,1200,1400",
            "thickness_model": "uniform_gradient",
            "thickness_grad_frac": "0.0,0.4",
            "frame_offset": "0.0",
            "group_width_deg": "frame,2frame",
            "detector_scale_bins": 0,
            "sigma_a": "0.0",
            "sigma_b_corr": "0.0",
            "sigma_c_peak": "0.0",
            "sg_jitter_mode": "none",
            "absorption_mode": "none",
            "dose_b_mode": "none",
            "row_resolution_b": "0.0",
        }
    return {
        "thickness_A": args.thickness_grid,
        "thickness_model": "uniform_gradient",
        "thickness_grad_frac": args.thickness_grad_grid,
        "thickness_sigma_frac": "0.0",
        "frame_offset": args.frame_offset_grid,
        "group_width_deg": args.group_width_grid,
        "detector_scale_bins": args.detector_scale_bins,
        "sigma_a": "0.0",
        "sigma_b_corr": "0.0",
        "sigma_c_peak": "0.0",
        "sg_jitter_mode": "none",
        "sg_jitter_sigma0": "0.0",
        "sg_jitter_sigma_d_inv": "0.0",
        "sg_jitter_sigma_frame": "0.0",
        "absorption_mode": "none",
        "absorption_mu": "0.0",
        "absorption_frac": "0.0",
        "absorption_mu_g2": "0.0",
        "absorption_mu_theta2": "0.0",
        "absorption_tds_b": "0.0",
        "dose_b_mode": "none",
        "dose_b_per_frame": "0.0",
        "row_resolution_b": "0.0",
    }


def apply_speed_profile(args: argparse.Namespace, structure: str | None = None) -> dict[str, object]:
    """Apply named run profiles after parsing and return a provenance record."""

    if getattr(args, "fast", False) and getattr(args, "full_fast", False):
        raise SystemExit("Choose either --fast or --full-fast. --fast is triage-only; --full-fast runs the automatic gates with cheap N/Q.")
    if not getattr(args, "full_fast", False):
        return {"profile": "standard"}

    changes: dict[str, object] = {"profile": "full_fast", "reason": "automatic gates with cheap Bloch settings and minimal endpoint polish"}

    def assign(name: str, value: object) -> None:
        old = getattr(args, name)
        setattr(args, name, value)
        if old != value:
            changes[name] = {"old": old, "new": value}

    n_basis = int(args.full_fast_n_basis)
    n_points = int(args.full_fast_n_points)
    assign("screen_n_basis", n_basis)
    assign("screen_n_points", n_points)
    assign("transform_rerank_n_basis", min(int(args.transform_rerank_n_basis), n_basis))
    assign("transform_rerank_n_points", min(int(args.transform_rerank_n_points), n_points))
    assign("hand_n_basis", min(int(args.hand_n_basis), n_basis))
    assign("hand_n_points", min(int(args.hand_n_points), n_points))
    assign("n_points", n_points)
    assign("structure_n_basis", min(int(args.structure_n_basis), n_basis))
    assign("structure_n_points", min(int(args.structure_n_points), max(3, n_points)))
    atom_count = estimate_model_nonh_atoms(structure or getattr(args, "structure", "") or "")
    auto_rows = 250
    auto_nfev = 2
    if atom_count is not None and atom_count > 80:
        auto_rows = 120
        auto_nfev = 1
    elif atom_count is not None and atom_count > 45:
        auto_rows = 160
        auto_nfev = 1
    row_limit = int(args.full_fast_structure_rows) if int(args.full_fast_structure_rows) > 0 else auto_rows
    eval_limit = int(args.full_fast_structure_max_nfev) if int(args.full_fast_structure_max_nfev) > 0 else auto_nfev
    changes["structure_budget"] = {
        "non_h_atoms": atom_count,
        "row_limit": row_limit,
        "max_eval_limit": eval_limit,
        "source": "explicit" if int(args.full_fast_structure_rows) > 0 or int(args.full_fast_structure_max_nfev) > 0 else "atom_count_auto",
    }
    assign("structure_objective_rows", min(int(args.structure_objective_rows), row_limit))
    assign("structure_max_nfev", min(int(args.structure_max_nfev), eval_limit))
    assign("shelxl_iterations", min(int(args.shelxl_iterations), 1))
    if args.endpoint_feedback == "auto":
        assign("endpoint_feedback", "off")
    if args.observation_polish == "auto":
        assign("observation_polish", "off")
    if args.forensic_depth == "standard":
        assign("forensic_depth", "quick")
    return changes


def nuisance_with_updates(nui: dict, **updates) -> dict:
    """Return a nuisance dictionary with only non-empty updates applied."""

    out = dict(nui)
    for key, value in updates.items():
        if value is not None and str(value).strip() != "":
            out[key] = value
    return out


def feature_grid(args: argparse.Namespace, family: str, nui: dict) -> dict:
    """Expand one optional feature family into a candidate nuisance grid."""

    if family == "row_resolution":
        return nuisance_with_updates(nui, row_resolution_b=args.row_resolution_b_grid)
    if family == "absorption":
        return nuisance_with_updates(
            nui,
            absorption_mode=args.absorption_mode_grid,
            absorption_mu=args.absorption_mu_grid,
            absorption_frac=args.absorption_frac_grid,
            absorption_mu_g2=args.absorption_mu_g2_grid,
            absorption_mu_theta2=args.absorption_mu_theta2_grid,
            absorption_tds_b=args.absorption_tds_b_grid,
        )
    if family == "sg_jitter":
        return nuisance_with_updates(
            nui,
            sg_jitter_mode=args.sg_jitter_mode_grid,
            sg_jitter_sigma0=args.sg_jitter_sigma0_grid,
            sg_jitter_sigma_d_inv=args.sg_jitter_sigma_d_inv_grid,
            sg_jitter_sigma_frame=args.sg_jitter_sigma_frame_grid,
            sg_jitter_points=args.sg_jitter_points,
        )
    if family == "sigma_error":
        return nuisance_with_updates(
            nui,
            sigma_a=args.sigma_a_grid,
            sigma_b_corr=args.sigma_b_corr_grid,
            sigma_c_peak=args.sigma_c_peak_grid,
        )
    if family == "dose":
        return nuisance_with_updates(
            nui,
            dose_b_mode=args.dose_b_mode_grid,
            dose_b_per_frame=args.dose_b_per_frame_grid,
        )
    raise ValueError(f"unknown feature family {family!r}")


def feature_is_enabled(args: argparse.Namespace, family: str) -> bool:
    """Return true when an optional feature family has a real search grid."""

    if args.fast:
        return False
    if family == "row_resolution":
        return grid_has_variation(args.row_resolution_b_grid)
    if family == "absorption":
        return (
            grid_has_variation(args.absorption_mode_grid)
            or grid_has_variation(args.absorption_mu_grid)
            or grid_has_variation(args.absorption_frac_grid)
            or grid_has_variation(args.absorption_mu_g2_grid)
            or grid_has_variation(args.absorption_mu_theta2_grid)
            or grid_has_variation(args.absorption_tds_b_grid)
        )
    if family == "sg_jitter":
        return (
            grid_has_variation(args.sg_jitter_mode_grid)
            or grid_has_variation(args.sg_jitter_sigma0_grid)
            or grid_has_variation(args.sg_jitter_sigma_d_inv_grid)
            or grid_has_variation(args.sg_jitter_sigma_frame_grid)
        )
    if family == "sigma_error":
        return (
            grid_has_variation(args.sigma_a_grid)
            or grid_has_variation(args.sigma_b_corr_grid)
            or grid_has_variation(args.sigma_c_peak_grid)
        )
    if family == "dose":
        return grid_has_variation(args.dose_b_mode_grid) or grid_has_variation(args.dose_b_per_frame_grid)
    return False


def is_shelx_model(path: str) -> bool:
    """Return true when a model path is a SHELX .res or .ins file."""

    return Path(path).suffix.lower() in {".res", ".ins"}


def prediction_cache_dir(args: argparse.Namespace, ds: DatasetSpec, stage: str) -> Path:
    """Choose the prediction-cache directory for a dataset and stage."""

    root = Path(args.prediction_cache_dir).expanduser() if args.prediction_cache_dir else Path(args.out_dir) / "cache"
    if args.prediction_cache_scope == "stage":
        return root / "stage_predictions" / stage / ds.name
    return root / "native_predictions" / ds.name


def effective_topup_d_min(args: argparse.Namespace, ds: DatasetSpec | None = None, min_d_spacing: float | None = None) -> float:
    """Return the Bloch support depth, never shallower than selected data."""

    requested = float(getattr(args, "topup_d_min", 0.8))
    data_limit = float(min_d_spacing if min_d_spacing is not None else (ds.min_d_spacing if ds is not None else requested))
    if data_limit > 0.0:
        return min(requested, data_limit)
    return requested


def resolved_sg_window_mode(args: argparse.Namespace, ds: DatasetSpec) -> str:
    """Choose the rocking-window source for this dataset.

    DIALS rows generated by GROND carry the reflection-table bbox z-range in
    ALF0/BET0.  The bbox is kept for diagnostics, but auto mode keeps the
    reference fixed Sg window because the DIALS integration support is usually
    broader than the physical Bloch quadrature width.
    """

    requested = str(getattr(args, "sg_window_mode", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested
    return "fixed"


def infer_cbf_dir_from_hkl(hkl: str) -> str:
    """Guess the raw CBF directory near an HKL file for optional raw polish."""

    path = Path(hkl).expanduser().resolve()
    for parent in [path.parent, *path.parents]:
        candidate = parent / "cbf"
        if candidate.is_dir() and any(candidate.glob("*.cbf")):
            return str(candidate)
        sibling = parent.parent / "cbf" if parent.parent != parent else parent / "cbf"
        if sibling.is_dir() and any(sibling.glob("*.cbf")):
            return str(sibling)
    return ""


def make_noh_model(python: str, structure: str, out_dir: Path, logs: Path, dry_run: bool) -> Path:
    """Create or reuse a hydrogen-stripped model variant."""

    src = Path(structure)
    dst = out_dir / "input" / f"{src.stem}_noH{src.suffix or '.res'}"
    if dst.exists():
        return dst
    cmd = [python, str(engine_script("make_res_variants.py")), "--input", str(src), "--output", str(dst), "--remove-h"]
    run_logged(cmd, logs, "make_noH_model", dry_run=dry_run)
    return dst


def make_alternate_hand_model(python: str, structure: str, out_dir: Path, logs: Path, dry_run: bool) -> Path:
    """Create or content-validate the alternate-hand model variant."""

    src = Path(structure).expanduser().resolve()
    dst = out_dir / "input" / f"{src.stem}_alternate_hand{src.suffix or '.res'}"
    manifest_path = dst.with_suffix(dst.suffix + ".source.json")
    expected_manifest = {
        "schema_version": 1,
        "variant": "alternate_hand",
        "materialization_contract": "invert_atoms_and_shelx_symmetry_translations_v1",
        "source_path": str(src),
        "source_sha256": sha256_file(src),
    }
    recorded = try_read_json(manifest_path)
    output_matches = bool(
        dst.is_file()
        and recorded.get("output_sha256")
        and recorded.get("output_sha256") == sha256_file(dst)
    )
    if output_matches and all(recorded.get(key) == value for key, value in expected_manifest.items()):
        return dst
    if not dry_run:
        dst.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    cmd = [python, str(engine_script("make_res_variants.py")), "--input", str(src), "--output", str(dst), "--alternate-hand"]
    run_logged(cmd, logs, "make_alternate_hand_model", dry_run=dry_run)
    if not dry_run:
        if not dst.is_file() or dst.stat().st_size <= 0:
            raise RuntimeError(f"alternate-hand model materialization did not produce {dst}")
        write_json(manifest_path, {**expected_manifest, "output_sha256": sha256_file(dst)})
    return dst


def native_stage_cmd(
    python: str,
    ds: DatasetSpec,
    structure: str,
    baseline_hkl: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    n_basis: int,
    n_points: int,
    stage: str,
    pinned: dict | None = None,
    alternate_hand: bool = False,
    make_maps: bool = True,
    endpoint: bool = True,
    max_rows_override: int | None = None,
    workers_override: int | None = None,
    sg_window_override: float | None = None,
    sg_window_mode_override: str | None = None,
    scale_fit_split_override: str | None = None,
    selection_metric_override: str | None = None,
    strict_pinned_contract: bool = False,
    promotion_baseline: Mapping[str, object] | None = None,
) -> list[str]:
    """Build one native nuisance-regularization command for the vendored engine."""

    pinned = pinned or {}
    fast = bool(args.fast)
    endpoint_ok = bool(endpoint and is_shelx_model(structure) and not args.native_only)
    lambda_grid = FULL_Q_LAMBDA if endpoint_ok else ""
    endpoint_b_grid = args.endpoint_b_grid if endpoint_ok and not fast else "0.0"
    endpoint_exti_grid = args.endpoint_exti_grid if endpoint_ok and not fast else "off"
    shelxl_iterations = args.shelxl_iterations if endpoint_ok and not fast else 1
    finalish_stage = stage in {"final", "noH", "hand", "adaptive_final"} or stage.endswith("_final")
    post_bloch_sigma_policy = resolved_post_bloch_sigma_policy(args, final_endpoint=bool(endpoint_ok and finalish_stage and not fast))
    thickness_grid = grid_from_nuisance(pinned, "thickness_A", "1200") if pinned else ("1000,1200,1400" if fast else args.thickness_grid)
    grad_grid = grid_from_nuisance(pinned, "thickness_grad_frac", "0.4") if pinned else ("0.0,0.4" if fast else args.thickness_grad_grid)
    frame_grid = grid_from_nuisance(pinned, "frame_offset", "0.0") if pinned else ("0.0" if fast else args.frame_offset_grid)
    group_grid = grid_from_nuisance(pinned, "group_width_deg", "1.0") if pinned else ("frame,2frame" if fast else args.group_width_grid)
    sigma_a = grid_from_nuisance(pinned, "sigma_a", "0.0") if pinned else ("0.0" if fast else args.sigma_a_grid)
    sigma_b = grid_from_nuisance(pinned, "sigma_b_corr", "0.0") if pinned else "0.0"
    sigma_c = grid_from_nuisance(pinned, "sigma_c_peak", "0.0") if pinned else ("0.0" if fast else args.sigma_c_peak_grid)
    row_resolution_b = grid_from_nuisance(pinned, "row_resolution_b", "0.0") if pinned else ("0.0" if fast else args.row_resolution_b_grid)
    detector_scale_bins = grid_from_nuisance(pinned, "detector_scale_bins", "0") if pinned else ("0" if fast else args.detector_scale_bins)
    scale_fit_split = scale_fit_split_override or ("all" if args.production_scales else "work")
    if scale_fit_split not in {"work", "all"}:
        raise ValueError(f"invalid scale-fit split override {scale_fit_split!r}")
    pinned_jitter_mode = grid_from_nuisance(pinned, "sg_jitter_mode", "none") if pinned else ""
    jitter_mode_grid = pinned_jitter_mode if pinned else ("none" if fast else args.sg_jitter_mode_grid)
    if (
        pinned
        and str(getattr(args, "pinned_sg_jitter_fallback", "none")) == "none"
        and pinned_jitter_mode.strip().lower() != "none"
    ):
        # A previously selected convolution can become geometrically
        # infeasible after a resolution/model change. Keep the strict local
        # root bound and score the ordinary no-jitter physics as an explicit
        # fallback candidate instead of widening, truncating, or crashing.
        jitter_mode_grid = f"{pinned_jitter_mode},none"
    cmd = [
        python,
        str(engine_script("native_row_regularize.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        "--baseline-hkl",
        baseline_hkl or str(out_dir / "NO_BASELINE_HKL_SUPPLIED.hkl"),
        f"--hkl-transform={ds.hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(n_points),
        "--sg-window",
        f"{float(args.sg_window if sg_window_override is None else sg_window_override):g}",
        "--sg-window-mode",
        sg_window_mode_override or resolved_sg_window_mode(args, ds),
        "--sg-window-min",
        f"{args.sg_window_min:g}",
        "--sg-window-max",
        f"{args.sg_window_max:g}",
        "--sg-window-scale",
        f"{args.sg_window_scale:g}",
        "--sg-window-frame-floor",
        f"{args.sg_window_frame_floor:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--thickness-grid",
        thickness_grid,
        "--thickness-model-grid",
        grid_from_nuisance(pinned, "thickness_model", "uniform_gradient") if pinned else args.thickness_model_grid,
        "--thickness-grad-grid",
        grad_grid,
        "--thickness-sigma-frac-grid",
        grid_from_nuisance(pinned, "thickness_sigma_frac", "0.0") if pinned else "0.0",
        f"--frame-offset-grid={frame_grid}",
        "--group-width-grid",
        group_grid,
        "--detector-scale-bins",
        detector_scale_bins,
        "--sigma-a-grid",
        sigma_a,
        "--sigma-b-corr-grid",
        sigma_b,
        "--sigma-c-peak-grid",
        sigma_c,
        "--sg-jitter-mode-grid",
        jitter_mode_grid,
        "--sg-jitter-sigma0-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma0", "0.0") if pinned else ("0.0" if fast else args.sg_jitter_sigma0_grid),
        "--sg-jitter-sigma-d-inv-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma_d_inv", "0.0") if pinned else ("0.0" if fast else args.sg_jitter_sigma_d_inv_grid),
        "--sg-jitter-sigma-frame-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma_frame", "0.0") if pinned else "0.0",
        "--sg-jitter-points",
        str(int(float(pinned.get("sg_jitter_points", args.sg_jitter_points if not fast else 1)))),
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--absorption-mode-grid",
        grid_from_nuisance(pinned, "absorption_mode", "none") if pinned else ("none" if fast else args.absorption_mode_grid),
        "--absorption-mu-grid",
        grid_from_nuisance(pinned, "absorption_mu", "0.0") if pinned else ("0.0" if fast else args.absorption_mu_grid),
        "--absorption-frac-grid",
        grid_from_nuisance(pinned, "absorption_frac", "0.0") if pinned else ("0.0" if fast else args.absorption_frac_grid),
        "--absorption-mu-g2-grid",
        grid_from_nuisance(pinned, "absorption_mu_g2", "0.0") if pinned else ("0.0" if fast else args.absorption_mu_g2_grid),
        "--absorption-mu-theta2-grid",
        grid_from_nuisance(pinned, "absorption_mu_theta2", "0.0") if pinned else "0.0",
        "--absorption-tds-b-grid",
        grid_from_nuisance(pinned, "absorption_tds_b", "0.0") if pinned else "0.0",
        "--dose-b-mode-grid",
        grid_from_nuisance(pinned, "dose_b_mode", "none") if pinned else ("none" if fast else args.dose_b_mode_grid),
        "--dose-b-per-frame-grid",
        grid_from_nuisance(pinned, "dose_b_per_frame", "0.0") if pinned else ("0.0" if fast else args.dose_b_per_frame_grid),
        "--row-resolution-b-grid",
        row_resolution_b,
        "--thickness-points",
        str(args.final_thickness_points if (pinned and finalish_stage) else (3 if fast else args.thickness_points)),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--free-fraction",
        f"{args.free_fraction:g}",
        "--split-seed",
        str(args.split_seed),
        "--scale-fit-split",
        scale_fit_split,
        "--selection-metric",
        selection_metric_override or args.selection_metric,
        "--lambda-grid",
        lambda_grid,
        "--scale-smoothness",
        str(args.scale_smoothness),
        "--endpoint-b-grid",
        endpoint_b_grid,
        "--endpoint-exti-grid",
        endpoint_exti_grid,
        "--shelxl",
        str(getattr(args, "shelxl", "") or default_shelxl_executable()),
        "--shelxl-iterations",
        str(shelxl_iterations),
        "--endpoint-wght-mode",
        str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
        "--post-bloch-sigma-policy",
        post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--workers",
        str(args.workers if workers_override is None else max(1, int(workers_override))),
        "--chunksize",
        str(args.chunksize),
        "--prediction-cache-dir",
        str(prediction_cache_dir(args, ds, stage)),
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        getattr(args, "row_plan_cache", "off"),
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--matrix-convention",
        str(pinned.get("matrix_convention", args.matrix_convention)) if strict_pinned_contract else args.matrix_convention,
        "--hkl-float-format",
        "shelx_fixed",
    ]
    if strict_pinned_contract:
        cmd.extend(
            [
                "--sg-jitter-student-df",
                grid_from_nuisance(pinned, "sg_jitter_student_df", "4.0"),
                "--sg-jitter-width-factor",
                grid_from_nuisance(pinned, "sg_jitter_width_factor", "3.0"),
                "--dose-b-ref-frame",
                grid_from_nuisance(pinned, "dose_b_ref_frame", "nan"),
                "--wght-a-grid",
                grid_from_nuisance(pinned, "native_wght_a", "") if "native_wght_a" in pinned else "",
                "--wght-b-grid",
                grid_from_nuisance(pinned, "native_wght_b", "") if "native_wght_b" in pinned else "",
            ]
        )
    if promotion_baseline:
        cmd.extend(
            [
                "--promotion-baseline-json",
                json.dumps(dict(promotion_baseline), sort_keys=True, allow_nan=True),
                "--promotion-min-gain",
                str(float(args.continuous_thickness_min_gain)),
                "--promotion-max-free-wr2-loss",
                str(float(args.continuous_thickness_max_free_wr2_loss)),
                "--promotion-max-all-r1-loss",
                str(float(args.continuous_thickness_max_all_r1_loss)),
            ]
        )
    if ds.include_nonpositive:
        cmd.append("--include-nonpositive")
    else:
        cmd.append("--positive-only")
    max_rows = args.max_rows if max_rows_override is None else max_rows_override
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    if alternate_hand:
        cmd.append("--alternate-hand")
    if make_maps and endpoint_ok:
        cmd.append("--make-maps")
    if not endpoint_ok:
        cmd.append("--skip-endpoint-refinement")
    return cmd


def continuous_thickness_cmd(
    python: str,
    ds: DatasetSpec,
    structure: str,
    out_dir: Path,
    args: argparse.Namespace,
    pinned: dict,
    *,
    n_basis: int,
    n_points: int,
    stage: str,
) -> list[str]:
    """Build the phase-only continuous thickness-polish command."""

    thickness = grid_from_nuisance(pinned, "thickness_A", "1200")
    grad = grid_from_nuisance(pinned, "thickness_grad_frac", "0.4")
    frame_offset = grid_from_nuisance(pinned, "frame_offset", "0.0")
    group_width = grid_from_nuisance(pinned, "group_width_deg", "1.0")
    detector_bins = grid_from_nuisance(pinned, "detector_scale_bins", "0")
    sigma_a = grid_from_nuisance(pinned, "sigma_a", "0.0")
    sigma_b = grid_from_nuisance(pinned, "sigma_b_corr", "0.0")
    sigma_c = grid_from_nuisance(pinned, "sigma_c_peak", "0.0")
    row_resolution_b = grid_from_nuisance(pinned, "row_resolution_b", "0.0")
    cmd = [
        python,
        str(engine_script("continuous_thickness_polish.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        f"--hkl-transform={ds.hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(n_points),
        "--sg-window",
        f"{args.sg_window:g}",
        "--sg-window-mode",
        resolved_sg_window_mode(args, ds),
        "--sg-window-min",
        f"{args.sg_window_min:g}",
        "--sg-window-max",
        f"{args.sg_window_max:g}",
        "--sg-window-scale",
        f"{args.sg_window_scale:g}",
        "--sg-window-frame-floor",
        f"{args.sg_window_frame_floor:g}",
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--start-thickness",
        thickness,
        "--start-grad",
        grad,
        "--thickness-points",
        str(args.thickness_points),
        "--frame-offset",
        frame_offset,
        "--group-width-deg",
        group_width,
        "--detector-scale-bins",
        detector_bins,
        "--sigma-a",
        sigma_a,
        "--sigma-b-corr",
        sigma_b,
        "--sigma-c-peak",
        sigma_c,
        "--row-resolution-b",
        row_resolution_b,
        "--scale-smoothness",
        str(args.scale_smoothness),
        "--matrix-convention",
        args.matrix_convention,
        "--free-fraction",
        f"{args.free_fraction:g}",
        "--split-seed",
        str(args.split_seed),
        "--scale-fit-split",
        "all" if args.production_scales else "work",
        "--selection-metric",
        args.continuous_thickness_metric,
        "--strong-rows",
        str(args.continuous_thickness_strong_rows),
        "--bracket-frac",
        str(args.continuous_thickness_bracket_frac),
        "--grid-thickness-step",
        str(args.continuous_thickness_grid_step),
        "--grid-grad-step",
        str(args.continuous_thickness_grad_step),
        "--powell-maxiter",
        str(args.continuous_thickness_powell_maxiter),
        "--min-promotion-gain",
        str(args.continuous_thickness_min_gain),
        "--max-free-wr2-loss",
        str(args.continuous_thickness_max_free_wr2_loss),
        "--max-all-r1-loss",
        str(args.continuous_thickness_max_all_r1_loss),
    ]
    if ds.include_nonpositive:
        cmd.append("--include-nonpositive")
    max_rows = int(getattr(args, "continuous_thickness_max_rows", 0) or 0)
    if max_rows > 0:
        cmd += ["--max-rows", str(max_rows)]
    return cmd


def continuous_thickness_enabled(args: argparse.Namespace) -> bool:
    """Return true when automatic continuous thickness polish should run."""

    mode = str(getattr(args, "continuous_thickness_polish", "auto") or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return not bool(getattr(args, "fast", False))


def continuous_thickness_seed_nuisance(args: argparse.Namespace, *candidates: Mapping[str, object] | None) -> dict:
    """Return the pinned nuisance state for continuous thickness polish.

    Real runs use the best JSON summary from the preceding screen/regularize stage.
    Dry-runs do not have those summaries yet, so this supplies a conservative
    single-point seed only to make the generated execution plan complete.
    """

    for candidate in candidates:
        if candidate:
            return dict(candidate)
    if not bool(getattr(args, "dry_run", False)):
        return {}
    return {
        "thickness_A": first(getattr(args, "thickness_grid", "1200"), "1200"),
        "thickness_grad_frac": first(getattr(args, "thickness_grad_grid", "0.4"), "0.4"),
        "frame_offset": first(getattr(args, "frame_offset_grid", "0.0"), "0.0"),
        "group_width_deg": "1.0",
        "detector_scale_bins": "0",
        "sigma_a": "0.0",
        "sigma_b_corr": "0.0",
        "sigma_c_peak": "0.0",
        "row_resolution_b": "0.0",
    }


def numeric_nuisance_contract_matches(expected_value: object, actual_value: object) -> bool:
    """Compare one numeric nuisance value without mishandling N/A NaNs.

    Controlled engine summaries use NaN for numeric fields that do not apply,
    such as ``dose_b_ref_frame`` when dose correction is disabled.  Matching
    NaN sentinels describe the same contract; one-sided NaNs, infinities, and
    non-numeric values do not.  Finite values retain the existing relative
    replay tolerance.
    """

    try:
        expected = float(expected_value)
        actual = float(actual_value)
    except (TypeError, ValueError):
        return False
    if math.isnan(expected) and math.isnan(actual):
        return True
    if not math.isfinite(expected) or not math.isfinite(actual):
        return False
    tolerance = 1e-8 * max(1.0, abs(expected))
    return abs(actual - expected) <= tolerance


def discrete_thickness_promotion_record(run: Path, *, dry_run: bool) -> dict:
    """Return the engine's auditable post-ranking thickness decision."""

    run = Path(run)
    artifact = run / "discrete_thickness_promotion.json"
    if dry_run:
        return {
            "family": "discrete_thickness_polish",
            "status": "planned",
            "run": str(run),
            "artifact": str(artifact),
        }
    summary_path = run / "summary.json"
    summary = try_read_json(summary_path)
    decision = summary.get("promotion_decision")
    if not isinstance(decision, Mapping) or str(decision.get("status", "")) == "not_requested":
        raise RuntimeError(
            "discrete thickness stage did not record the required held-out promotion decision: "
            f"{summary_path}"
        )
    return {
        **dict(decision),
        "run": str(run),
        "artifact": str(artifact),
    }


def discrete_thickness_baseline_nuisance(args: argparse.Namespace, nuisance: Mapping[str, object] | None) -> dict:
    """Require a real starting nuisance, with a planning-only dry-run seed."""

    if nuisance:
        return dict(nuisance)
    if bool(getattr(args, "dry_run", False)):
        return {
            "thickness_A": 1200.0,
            "thickness_grad_frac": 0.0,
            "frame_offset": 0.0,
        }
    raise RuntimeError(
        "discrete thickness polishing has no starting nuisance from the preceding stage; "
        "refusing to run an unguarded work-only search"
    )


def materialize_continuous_nuisance_run(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    *,
    active_model: str,
    baseline_run: Path,
    continuous_run: Path,
    selected_nuisance: Mapping[str, object],
    n_basis: int,
    n_points: int,
    stage: str,
) -> tuple[dict, Path, dict]:
    """Materialize a continuous-thickness choice on the full row contract.

    ``continuous_thickness_polish.py`` intentionally optimizes a bounded row
    subset and writes a summary rather than ``native_dyn_calc.csv``.  Feature
    gates and multi-dataset policy decisions require a complete, comparable
    row table.  Reusing the optimizer directory as if it were a native run
    silently disables those gates, so the selected nuisance is replayed once
    through the ordinary native stage before it can flow downstream.
    """

    nuisance = dict(selected_nuisance or {})
    if not nuisance:
        raise RuntimeError(
            f"continuous thickness stage for {ds.name} did not record a selected nuisance: "
            f"{continuous_run / 'summary.json'}"
        )
    source_row_table = continuous_run / "native_dyn_calc.csv"
    materialized = continuous_run if source_row_table.exists() else continuous_run / "materialized_full_rows"
    if not source_row_table.exists():
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                materialized,
                args,
                n_basis=int(n_basis),
                n_points=int(n_points),
                stage=stage,
                pinned=nuisance,
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_{stage}",
            dry_run=args.dry_run,
        )
    row_table = materialized / "native_dyn_calc.csv"
    if not args.dry_run and not row_table.exists():
        raise RuntimeError(
            "continuous-thickness materialization did not produce the required full row table: "
            f"{row_table}"
        )
    materialized_nuisance = best_nuisance(materialized) if not args.dry_run else nuisance
    if not args.dry_run and not materialized_nuisance:
        raise RuntimeError(
            "continuous-thickness materialization did not record a reusable nuisance block: "
            f"{materialized / 'summary.json'}"
        )
    numeric_contract_keys = (
        "thickness_A",
        "thickness_grad_frac",
        "thickness_sigma_frac",
        "frame_offset",
        "group_width_deg",
        "detector_scale_bins",
        "sigma_a",
        "sigma_b_corr",
        "sigma_c_peak",
        "row_resolution_b",
        "scale_smoothness",
        "sg_jitter_sigma0",
        "sg_jitter_sigma_d_inv",
        "sg_jitter_sigma_frame",
        "sg_jitter_points",
        "sg_jitter_student_df",
        "sg_jitter_width_factor",
        "absorption_mu",
        "absorption_frac",
        "absorption_mu_g2",
        "absorption_mu_theta2",
        "absorption_tds_b",
        "dose_b_per_frame",
        "dose_b_ref_frame",
    )
    categorical_contract_keys = (
        "thickness_model",
        "sg_jitter_mode",
        "absorption_mode",
        "dose_b_mode",
        "matrix_convention",
    )
    mismatches: list[str] = []
    if not args.dry_run:
        for key in numeric_contract_keys:
            if key not in nuisance:
                continue
            if key not in materialized_nuisance or not numeric_nuisance_contract_matches(
                nuisance.get(key), materialized_nuisance.get(key)
            ):
                mismatches.append(f"{key}: selected={nuisance.get(key)!r}, materialized={materialized_nuisance.get(key)!r}")
        for key in categorical_contract_keys:
            if key in nuisance and str(materialized_nuisance.get(key)) != str(nuisance.get(key)):
                mismatches.append(f"{key}: selected={nuisance.get(key)!r}, materialized={materialized_nuisance.get(key)!r}")
        if mismatches:
            raise RuntimeError(
                "continuous-thickness full-row materialization changed the selected nuisance contract: "
                + "; ".join(mismatches)
            )
    baseline_summary_path = Path(baseline_run) / "summary.json"
    materialized_summary_path = materialized / "summary.json"
    if not args.dry_run and not baseline_summary_path.exists():
        raise RuntimeError(
            "continuous-thickness full-row promotion has no comparable baseline summary: "
            f"{baseline_summary_path}"
        )
    baseline_summary = try_read_json(baseline_summary_path) if not args.dry_run else {}
    candidate_summary = try_read_json(materialized_summary_path) if not args.dry_run else {}
    baseline_score = score_native_summary(baseline_summary) if not args.dry_run else {}
    candidate_score = score_native_summary(candidate_summary) if not args.dry_run else {}
    promotion_reasons: list[str] = []
    score_gain = float("nan")
    free_wR2_loss = float("nan")
    all_R1_loss = float("nan")
    baseline_free_rows = 0
    candidate_free_rows = 0
    if not args.dry_run:
        def positive_free_rows(summary: Mapping[str, object]) -> int:
            best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
            metrics = best.get("metrics") if isinstance(best.get("metrics"), Mapping) else {}
            free = metrics.get("free") if isinstance(metrics.get("free"), Mapping) else {}
            free_dyn = free.get("dyn") if isinstance(free.get("dyn"), Mapping) else {}
            count = safe_float(free_dyn.get("n"), float("nan"))
            if not math.isfinite(count):
                selection = summary.get("row_selection") if isinstance(summary.get("row_selection"), Mapping) else {}
                count = safe_float(selection.get("free_rows"), 0.0)
            return max(0, int(count)) if math.isfinite(count) else 0

        baseline_free_rows = positive_free_rows(baseline_summary)
        candidate_free_rows = positive_free_rows(candidate_summary)
        if baseline_free_rows <= 0:
            promotion_reasons.append("baseline_missing_positive_free_rows")
        if candidate_free_rows <= 0:
            promotion_reasons.append("candidate_missing_positive_free_rows")
        baseline_total = safe_float(baseline_score.get("score"), float("nan"))
        candidate_total = safe_float(candidate_score.get("score"), float("nan"))
        baseline_free = safe_float(baseline_score.get("free_dyn_wR2_all"), float("nan"))
        candidate_free = safe_float(candidate_score.get("free_dyn_wR2_all"), float("nan"))
        baseline_r1 = safe_float(baseline_score.get("dyn_R1_all"), float("nan"))
        candidate_r1 = safe_float(candidate_score.get("dyn_R1_all"), float("nan"))
        if not all(
            math.isfinite(value)
            for value in (baseline_total, candidate_total, baseline_free, candidate_free, baseline_r1, candidate_r1)
        ):
            promotion_reasons.append("missing_finite_full_row_metrics")
        else:
            score_gain = baseline_total - candidate_total
            free_wR2_loss = candidate_free - baseline_free
            all_R1_loss = candidate_r1 - baseline_r1
            if score_gain < float(args.continuous_thickness_min_gain):
                promotion_reasons.append("full_row_gain_below_minimum")
            if free_wR2_loss > float(args.continuous_thickness_max_free_wr2_loss):
                promotion_reasons.append("full_row_free_wR2_loss")
            if all_R1_loss > float(args.continuous_thickness_max_all_r1_loss):
                promotion_reasons.append("full_row_all_R1_loss")
    accepted = bool(args.dry_run or not promotion_reasons)
    selected_run = materialized if accepted else Path(baseline_run)
    selected_nuisance = (
        dict(materialized_nuisance or nuisance)
        if accepted
        else best_nuisance(Path(baseline_run))
    )
    if not args.dry_run and not selected_nuisance:
        raise RuntimeError(
            "continuous-thickness full-row rejection could not restore the preceding nuisance contract: "
            f"{baseline_summary_path}"
        )
    return selected_nuisance, selected_run, {
        "family": "continuous_thickness_materialization",
        "status": "dry_run" if args.dry_run else ("accepted_full_row" if accepted else "rejected_full_row"),
        "source": str(continuous_run),
        "run": str(materialized),
        "row_table": str(row_table),
        "baseline_run": str(baseline_run),
        "selected_run": str(selected_run),
        "full_row_contract": True,
        "nuisance_contract_verified": not mismatches,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "full_row_score_gain": score_gain,
        "full_row_free_wR2_loss": free_wR2_loss,
        "full_row_all_R1_loss": all_R1_loss,
        "baseline_free_rows": baseline_free_rows,
        "candidate_free_rows": candidate_free_rows,
        "min_promotion_gain": float(args.continuous_thickness_min_gain),
        "max_free_wR2_loss": float(args.continuous_thickness_max_free_wr2_loss),
        "max_all_R1_loss": float(args.continuous_thickness_max_all_r1_loss),
        "promotion_reasons": promotion_reasons,
    }


def structure_cmd(python: str, ds: DatasetSpec, structure: str, baseline_hkl: str, out_dir: Path, args: argparse.Namespace, nui: dict) -> list[str]:
    """Build the optional restrained structure-update command."""

    return [
        python,
        str(engine_script("native_structure_update.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        "--baseline-hkl",
        baseline_hkl or str(out_dir / "NO_BASELINE_HKL_SUPPLIED.hkl"),
        f"--hkl-transform={ds.hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--n-basis",
        str(args.structure_n_basis),
        "--n-points",
        str(args.structure_n_points),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--sg-window",
        f"{args.sg_window:g}",
        "--sg-window-mode",
        resolved_sg_window_mode(args, ds),
        "--sg-window-min",
        f"{args.sg_window_min:g}",
        "--sg-window-max",
        f"{args.sg_window_max:g}",
        "--sg-window-scale",
        f"{args.sg_window_scale:g}",
        "--sg-window-frame-floor",
        f"{args.sg_window_frame_floor:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--objective-rows",
        str(args.structure_objective_rows),
        "--objective-bins",
        "10",
        "--objective-group-width-deg",
        grid_from_nuisance(nui, "group_width_deg", "1.0"),
        "--block-schedule",
        args.structure_blocks,
        "--atom-selection",
        args.structure_atom_selection,
        "--max-nfev",
        str(args.structure_max_nfev),
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        getattr(args, "row_plan_cache", "off"),
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--early-stop-window",
        str(args.structure_early_stop_window),
        "--early-stop-min-evals",
        str(args.structure_early_stop_min_evals),
        "--early-stop-min-improvement",
        f"{args.structure_early_stop_min_improvement:g}",
        "--early-stop-metric",
        args.structure_early_stop_metric,
        "--thickness",
        grid_from_nuisance(nui, "thickness_A", "1200"),
        "--thickness-grad-frac",
        grid_from_nuisance(nui, "thickness_grad_frac", "0.4"),
        "--frame-offset",
        grid_from_nuisance(nui, "frame_offset", "0.0"),
        "--row-resolution-b",
        grid_from_nuisance(nui, "row_resolution_b", "0.0"),
        "--sg-jitter-mode",
        str(nui.get("sg_jitter_mode", "none")),
        "--sg-jitter-sigma0",
        grid_from_nuisance(nui, "sg_jitter_sigma0", "0.0"),
        "--sg-jitter-sigma-d-inv",
        grid_from_nuisance(nui, "sg_jitter_sigma_d_inv", "0.0"),
        "--sg-jitter-sigma-frame",
        grid_from_nuisance(nui, "sg_jitter_sigma_frame", "0.0"),
        "--sg-jitter-points",
        str(int(float(nui.get("sg_jitter_points", 1)))),
        "--sg-jitter-student-df",
        grid_from_nuisance(nui, "sg_jitter_student_df", "4.0"),
        "--sg-jitter-width-factor",
        grid_from_nuisance(nui, "sg_jitter_width_factor", "3.0"),
        "--absorption-mode",
        str(nui.get("absorption_mode", "none")),
        "--absorption-mu",
        grid_from_nuisance(nui, "absorption_mu", "0.0"),
        "--absorption-frac",
        grid_from_nuisance(nui, "absorption_frac", "0.0"),
        "--absorption-mu-g2",
        grid_from_nuisance(nui, "absorption_mu_g2", "0.0"),
        "--absorption-mu-theta2",
        grid_from_nuisance(nui, "absorption_mu_theta2", "0.0"),
        "--absorption-tds-b",
        grid_from_nuisance(nui, "absorption_tds_b", "0.0"),
        "--group-width-deg",
        grid_from_nuisance(nui, "group_width_deg", "1.0"),
        "--detector-scale-bins",
        str(int(float(nui.get("detector_scale_bins", 0)))),
        "--sigma-a",
        grid_from_nuisance(nui, "sigma_a", "0.0"),
        "--sigma-b-corr",
        grid_from_nuisance(nui, "sigma_b_corr", "0.0"),
        "--sigma-c-peak",
        grid_from_nuisance(nui, "sigma_c_peak", "0.0"),
        "--lambda-grid",
        FULL_Q_LAMBDA,
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--acceptance-metric",
        "all_wR2",
        "--free-wR2-tolerance",
        "0.02",
    ]


def shelxl_feedback_cmd(
    python: str,
    ds: DatasetSpec,
    start_model: str,
    source_summary: Path,
    baseline_hkl: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    n_basis: int,
) -> list[str]:
    """Build the high-N/Q SHELXL feedback-loop command."""

    cmd = [
        python,
        str(engine_script("shelxl_feedback_loop.py")),
        "--source-summary",
        str(source_summary),
        "--start-res",
        start_model,
        "--out-dir",
        str(out_dir),
        "--python",
        python,
        "--hkl",
        ds.hkl,
        "--baseline-hkl",
        baseline_hkl or "",
        f"--hkl-transform={ds.hkl_transform}",
        "--cycles",
        str(args.shelxl_feedback_cycles),
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(args.n_points),
        "--final-n-basis",
        "0",
        "--final-n-points",
        str(args.n_points),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(args.shelxl_feedback_shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.shelxl_feedback_endpoint_ls_cycles),
        "--endpoint-wght-mode",
        str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
        "--post-bloch-sigma-policy",
        resolved_post_bloch_sigma_policy(args, feedback_loop=True),
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--lambda-grid",
        FULL_Q_LAMBDA,
        "--rescale",
        "median-iobs",
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        getattr(args, "row_plan_cache", "off"),
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
    ]
    if args.max_rows:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.make_maps:
        cmd.append("--make-maps-on-final")
    return cmd


def combine_cmd(
    python: str,
    runs: dict[str, Path],
    structure: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    shelxl_iterations: int | None = None,
) -> list[str]:
    """Build the multi-dataset corrected-HKL combination command."""

    iterations = int(shelxl_iterations if shelxl_iterations is not None else getattr(args, "shelxl_iterations", 5))
    endpoint_ls_cycles = int(
        getattr(args, "shelxl_feedback_endpoint_ls_cycles", getattr(args, "endpoint_ls_cycles", 1)) or 1
    )
    cmd = [
        python,
        str(engine_script("combine_native_nuisance_runs.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        structure,
        "--endpoint-lambda",
        FULL_Q_LAMBDA,
        "--endpoint-source",
        "native",
        "--interdataset-scale",
        "overlap",
        "--endpoint-merge-key",
        getattr(args, "multi_endpoint_merge_key", "model_symmetry"),
        "--dataset-sigma-policy",
        resolved_dataset_sigma_policy(args, multi_dataset=True),
        "--dataset-sigma-holdout-fraction",
        f"{args.dataset_sigma_holdout_fraction:g}",
        "--dataset-sigma-min-scale",
        f"{args.dataset_sigma_min_scale:g}",
        "--dataset-sigma-max-scale",
        f"{args.dataset_sigma_max_scale:g}",
        "--merge-sigma-policy",
        resolved_merge_sigma_policy(args, multi_dataset=True),
        "--post-bloch-sigma-policy",
        resolved_post_bloch_sigma_policy(args, final_endpoint=True),
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--shelxl",
        str(getattr(args, "shelxl", "") or ""),
        "--shelxl-iterations",
        str(iterations),
        "--endpoint-ls-cycles",
        str(endpoint_ls_cycles),
        "--endpoint-wght-mode",
        str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
    ]
    endpoint_min_d_spacing = float(
        getattr(args, "endpoint_min_d_spacing", 0.0)
        or getattr(args, "min_d_spacing", 0.0)
        or 0.0
    )
    if endpoint_min_d_spacing > 0.0:
        cmd += ["--min-d-spacing", f"{endpoint_min_d_spacing:g}"]
    if args.dataset_sigma_holdout_floor:
        cmd.append("--dataset-sigma-holdout-floor")
    for name, run in runs.items():
        cmd += ["--run", f"{name}={run}"]
    return cmd


def publication_report_cmd(
    python: str,
    prep: PrepConfig,
    result: dict,
    out_dir: Path,
    title: str,
    *,
    instability_u: float = 0.01,
) -> list[str]:
    """Build the GROND row/MR publication-table command for final GROND rows."""

    cmd = [
        python,
        str(engine_script("row_mr_report.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        str(result.get("model") or prep.structure),
        "--title",
        title,
        "--instability-u",
        f"{float(instability_u):g}",
    ]
    if result.get("combined"):
        for name, run in (result.get("publication_runs") or result.get("runs") or {}).items():
            cmd += ["--run", f"{name}={run}"]
        for name, run in (result.get("publication_full_runs") or {}).items():
            cmd += ["--full-run", f"{name}={run}"]
        scale_contract = result.get("scale_contract_selection")
        if isinstance(scale_contract, Mapping):
            selected_contract = str(scale_contract.get("selected_contract", "")).strip()
            if selected_contract:
                cmd += ["--selected-scale-contract", selected_contract]
        combined_summary = try_read_json(Path(str(result["combined"])) / "combined_summary.json")
        combined_endpoint = combined_summary.get("combined_endpoint")
        combined_endpoint = combined_endpoint if isinstance(combined_endpoint, Mapping) else {}
        requested_dmin = safe_float(
            combined_endpoint.get(
                "requested_min_d_spacing_A",
                combined_endpoint.get("min_d_spacing_A"),
            ),
            0.0,
        )
        if requested_dmin > 0.0:
            cmd += ["--requested-min-d-spacing", f"{requested_dmin:g}"]
    elif result.get("primary_run"):
        name = prep.datasets[0].name if prep.datasets else "dataset"
        cmd += ["--run", f"{name}={result['primary_run']}"]
    cmd += scale_polish_report_args(prep, result)
    cmd += resolution_limit_report_args(result)
    return cmd


def scale_polish_report_args(prep: PrepConfig, result: dict) -> list[str]:
    """Return final-table arguments for compact scale-manifold summaries."""

    scale_polish = result.get("scale_polish")
    if not isinstance(scale_polish, Mapping):
        return []

    args: list[str] = []
    if result.get("combined"):
        scale_contract = result.get("scale_contract_selection")
        if isinstance(scale_contract, Mapping):
            selected_contract = str(scale_contract.get("selected_contract", "full"))
            if selected_contract != "full":
                projected = scale_contract.get("publication_scale_polish_summaries")
                if not isinstance(projected, Mapping) or not projected:
                    return []
                for name, summary_path in projected.items():
                    if str(summary_path).strip():
                        args += ["--scale-polish", f"{name}={summary_path}"]
                return args
        publication_names = set(
            (result.get("publication_full_runs") or result.get("publication_runs") or {}).keys()
        )
        for name, payload in scale_polish.items():
            if publication_names and name not in publication_names:
                continue
            if not isinstance(payload, Mapping):
                continue
            out_dir = payload.get("out_dir")
            if out_dir:
                args += ["--scale-polish", f"{name}={Path(str(out_dir)) / 'scale_compression_summary.json'}"]
        return args

    out_dir = scale_polish.get("out_dir")
    if out_dir:
        name = prep.datasets[0].name if prep.datasets else "dataset"
        args += ["--scale-polish", f"{name}={Path(str(out_dir)) / 'scale_compression_summary.json'}"]
    return args


def resolution_limit_report_args(result: dict) -> list[str]:
    """Return final-table arguments for promoted resolution-limit summaries."""

    args: list[str] = []
    merged = result.get("merged_resolution_limit")
    if isinstance(merged, Mapping):
        summary = ((merged.get("artifacts") or {}).get("summary") or "")
        if summary:
            args += ["--resolution-limit", f"combined={summary}"]

    resolution_limits = result.get("resolution_limits")
    if isinstance(resolution_limits, Mapping):
        for name, payload in resolution_limits.items():
            if not isinstance(payload, Mapping):
                continue
            summary = ((payload.get("artifacts") or {}).get("summary") or "")
            if summary:
                args += ["--resolution-limit", f"{name}={summary}"]
    return args


def resolution_limit_cmd(
    python: str,
    runs: dict[str, Path],
    res: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    title: str,
    scope: str,
) -> list[str]:
    """Build the held-out resolution-limit promotion command."""

    cmd = [
        python,
        str(engine_script("resolution_limit_audit.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        res,
        "--title",
        title,
        "--scope",
        scope,
        "--shells",
        str(args.resolution_shells),
        "--strict-split-cc",
        str(args.resolution_strict_split_cc),
        "--strict-mean-i-over-sigma",
        str(args.resolution_strict_mean_i_over_sigma),
        "--exploratory-split-cc",
        str(args.resolution_exploratory_split_cc),
        "--exploratory-mean-i-over-sigma",
        str(args.resolution_exploratory_mean_i_over_sigma),
        "--min-common-pairs",
        str(args.resolution_min_common_pairs),
        "--min-shell-rows",
        str(args.resolution_min_shell_rows),
        "--min-strong-fraction",
        str(args.resolution_min_strong_fraction),
        "--max-shell-mwrall",
        str(args.resolution_max_shell_mwrall),
        "--max-free-work-mwr-gap",
        str(args.resolution_max_free_work_mwr_gap),
        "--min-free-rows",
        str(args.resolution_min_free_rows),
        "--max-cumulative-mwr-penalty",
        str(args.resolution_max_cumulative_mwr_penalty),
    ]
    for name, run in runs.items():
        cmd += ["--run", f"{name}={run}"]
    return cmd


def run_resolution_limit_gate(
    python: str,
    runs: dict[str, Path],
    res: str,
    out_dir: Path,
    logs: Path,
    label: str,
    args: argparse.Namespace,
    *,
    title: str,
    scope: str,
) -> dict:
    """Run held-out resolution-limit promotion on final row tables."""

    if not getattr(args, "resolution_limit_gate", True):
        return {"status": "disabled"}
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "runs": {k: str(v) for k, v in runs.items()}}
    missing = [str(Path(run) / "native_dyn_calc.csv") for run in runs.values() if not (Path(run) / "native_dyn_calc.csv").exists()]
    if missing:
        return {"status": "missing_row_table", "missing": missing}
    cmd = resolution_limit_cmd(python, runs, res, out_dir, args, title=title, scope=scope)
    run_logged(cmd, logs, f"{label}_resolution_limit", dry_run=False)
    summary = try_read_json(out_dir / "summary.json")
    if summary:
        return {"status": "ok", **summary}
    return {
        "status": "missing_artifact_summary",
        "summary": str((out_dir / "summary.json").resolve()),
        "report": str((out_dir / "REPORT.md").resolve()),
        "shells_csv": str((out_dir / "resolution_shells.csv").resolve()),
    }


def dataset_policy_kwargs(args: argparse.Namespace) -> dict:
    """Collect CLI thresholds for multi-dataset inclusion policy decisions."""

    return {
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
    missing = [str(Path(run) / "native_dyn_calc.csv") for run in runs.values() if not (Path(run) / "native_dyn_calc.csv").exists()]
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
        "refusing to silently fall back to equal dataset weights: "
        + ", ".join(str(path) for path in missing)
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


_SCALE_CONTRACT_MUTABLE_COLUMNS = {
    "dyn_scaled",
    "born_scaled",
    "dyn_full_scaled",
    "born_full_scaled",
    "dyn_replay_source_scaled",
    "born_replay_source_scaled",
    "scale_compression_model",
    "scale_replay_model",
}


def validate_scale_contract_run_map(
    full_runs: Mapping[str, Path],
    candidate_runs: Mapping[str, Path],
    *,
    min_bloch_wr2_gain: float = 0.0,
    max_dataset_row_wr2_cost: float = 0.015,
    max_preexisting_bloch_gain_loss: float = 0.01,
) -> dict:
    """Prove that compact tables changed only fitted scale predictions."""

    if set(full_runs) != set(candidate_runs):
        return {"ok": False, "reason": "dataset_set_mismatch"}
    datasets: dict[str, dict[str, object]] = {}
    for name in sorted(full_runs):
        full_path = Path(full_runs[name]) / "native_dyn_calc.csv"
        candidate_path = Path(candidate_runs[name]) / "native_dyn_calc.csv"
        if not full_path.exists() or not candidate_path.exists():
            return {"ok": False, "reason": "missing_row_table", "dataset": name}
        with full_path.open(newline="") as handle:
            full_reader = csv.DictReader(handle)
            full_columns = list(full_reader.fieldnames or [])
            full_rows = list(full_reader)
        with candidate_path.open(newline="") as handle:
            candidate_reader = csv.DictReader(handle)
            candidate_columns = list(candidate_reader.fieldnames or [])
            candidate_rows = list(candidate_reader)
        unknown_columns = sorted(set(candidate_columns) - set(full_columns) - _SCALE_CONTRACT_MUTABLE_COLUMNS)
        if unknown_columns:
            return {
                "ok": False,
                "reason": "unknown_candidate_columns",
                "dataset": name,
                "columns": unknown_columns,
            }
        if len(full_rows) != len(candidate_rows):
            return {
                "ok": False,
                "reason": "row_count_mismatch",
                "dataset": name,
                "full_rows": len(full_rows),
                "candidate_rows": len(candidate_rows),
            }
        immutable = [column for column in full_columns if column not in _SCALE_CONTRACT_MUTABLE_COLUMNS]
        for index, (full_row, candidate_row) in enumerate(zip(full_rows, candidate_rows, strict=True)):
            missing = [column for column in immutable if column not in candidate_row]
            changed = [column for column in immutable if candidate_row.get(column) != full_row.get(column)]
            if missing or changed:
                return {
                    "ok": False,
                    "reason": "immutable_row_contract_mismatch",
                    "dataset": name,
                    "row_index": index,
                    "missing_columns": missing,
                    "changed_columns": changed[:20],
                }
        full_summary = try_read_json(Path(full_runs[name]) / "summary.json")
        summary = try_read_json(Path(candidate_runs[name]) / "summary.json")
        best_native = summary.get("best_native") or {}
        metrics = best_native.get("metrics") or {}
        full_metrics = ((full_summary.get("best_native") or {}).get("metrics") or {})
        if any(not isinstance(metrics.get(split), Mapping) or not {"dyn", "born"}.issubset(metrics[split]) for split in ("all", "work", "free")):
            return {"ok": False, "reason": "incomplete_split_metrics", "dataset": name}
        if any(not isinstance(full_metrics.get(split), Mapping) or not {"dyn", "born"}.issubset(full_metrics[split]) for split in ("all", "work", "free")):
            return {"ok": False, "reason": "incomplete_full_split_metrics", "dataset": name}
        bloch_gains: dict[str, float] = {}
        full_bloch_gains: dict[str, float] = {}
        row_wr2_costs: dict[str, float] = {}
        for split in ("all", "work", "free"):
            dyn_wr2 = safe_float((metrics[split].get("dyn") or {}).get("wR2_all"), float("nan"))
            born_wr2 = safe_float((metrics[split].get("born") or {}).get("wR2_all"), float("nan"))
            full_dyn_wr2 = safe_float((full_metrics[split].get("dyn") or {}).get("wR2_all"), float("nan"))
            full_born_wr2 = safe_float((full_metrics[split].get("born") or {}).get("wR2_all"), float("nan"))
            gain = born_wr2 - dyn_wr2
            full_gain = full_born_wr2 - full_dyn_wr2
            row_cost = dyn_wr2 - full_dyn_wr2
            if not all(math.isfinite(value) for value in (gain, full_gain, row_cost)):
                return {"ok": False, "reason": "nonfinite_split_metrics", "dataset": name, "split": split}
            if row_cost > float(max_dataset_row_wr2_cost):
                return {
                    "ok": False,
                    "reason": "dataset_row_wr2_cost_exceeded",
                    "dataset": name,
                    "split": split,
                    "cost": row_cost,
                    "maximum": float(max_dataset_row_wr2_cost),
                }
            if full_gain >= float(min_bloch_wr2_gain) and gain < float(min_bloch_wr2_gain):
                return {
                    "ok": False,
                    "reason": "bloch_wr2_gain_below_minimum",
                    "dataset": name,
                    "split": split,
                    "full_gain": full_gain,
                    "gain": gain,
                    "minimum": float(min_bloch_wr2_gain),
                }
            if (
                full_gain < float(min_bloch_wr2_gain)
                and gain < full_gain - float(max_preexisting_bloch_gain_loss)
            ):
                return {
                    "ok": False,
                    "reason": "preexisting_bloch_gain_loss_exceeded",
                    "dataset": name,
                    "split": split,
                    "full_gain": full_gain,
                    "gain": gain,
                    "loss": full_gain - gain,
                    "maximum": float(max_preexisting_bloch_gain_loss),
                }
            bloch_gains[split] = gain
            full_bloch_gains[split] = full_gain
            row_wr2_costs[split] = row_cost
        try:
            scale_count = int(best_native["scale_count"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"ok": False, "reason": "missing_materialized_scale_count", "dataset": name}
        if scale_count <= 0:
            return {"ok": False, "reason": "invalid_materialized_scale_count", "dataset": name}
        datasets[name] = {
            "rows": len(full_rows),
            "immutable_columns": immutable,
            "scale_count": scale_count,
            "bloch_wr2_gain": bloch_gains,
            "full_bloch_wr2_gain": full_bloch_gains,
            "row_wr2_cost": row_wr2_costs,
        }
    return {"ok": True, "datasets": datasets}


def scale_contract_totals(scale_polish: Mapping[str, Mapping], selection_key: str) -> tuple[int, float] | None:
    """Sum materialized scale count and fitted-variance BIC across datasets."""

    scales = 0
    bic = 0.0
    for payload in scale_polish.values():
        selection = payload.get(selection_key)
        if not isinstance(selection, Mapping):
            return None
        scale_value = safe_float(selection.get("scale_params"), float("nan"))
        bic_value = safe_float((selection.get("information") or {}).get("bic_fitted_variance"), float("nan"))
        if not math.isfinite(scale_value) or not math.isfinite(bic_value):
            return None
        scales += int(round(scale_value))
        bic += bic_value
    return scales, bic


def combined_scale_contract_candidate(label: str, combined_dir: Path, *, scale_params: int, aggregate_bic: float) -> dict:
    """Convert one combined replay summary into the pure selector contract."""

    summary = try_read_json(combined_dir / "combined_summary.json")
    native = summary.get("native_joint") or {}
    endpoint = summary.get("combined_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    endpoint_artifacts = endpoint.get("artifacts") or {}
    endpoint_health = shelxl.get("res_health") or {}
    dataset_sigma = endpoint.get("dataset_sigma") or {}
    physics_contract = {
        "lambda_q": endpoint.get("lambda_q"),
        "min_d_spacing_A": endpoint.get("min_d_spacing_A", 0.0),
        "endpoint_b": endpoint.get("endpoint_b"),
        "keep_exti": endpoint.get("keep_exti"),
        "merge_key": endpoint.get("merge_key"),
        "interdataset_scale": (summary.get("interdataset_scale") or {}).get("mode"),
        "dataset_sigma_policy": sorted((name, (meta or {}).get("policy")) for name, meta in dataset_sigma.items()),
        "merge_sigma_policy": (endpoint.get("merge_sigma") or {}).get("policy"),
        "post_bloch_sigma_policy": (endpoint.get("post_bloch_sigma") or {}).get("policy"),
        "endpoint_wght_mode": endpoint.get("endpoint_wght_mode"),
        "endpoint_ls_cycles": endpoint.get("endpoint_ls_cycles"),
        "datasets": sorted(
            (
                row.get("name"),
                row.get("dataset_role"),
                row.get("raw_row_weight"),
                row.get("endpoint_weight"),
            )
            for row in summary.get("runs", [])
        ),
    }
    return {
        "label": label,
        "directory": str(combined_dir.resolve()),
        "scale_params": int(scale_params),
        "aggregate_bic": float(aggregate_bic),
        "input_endpoint_rows": endpoint.get("input_unmerged_rows"),
        "row_wr2": {
            split: ((native.get("dyn") or {}).get(split) or {}).get("wR2_all")
            for split in ("all", "work", "free")
        },
        "bloch_gain_wr2": {
            split: safe_float(((native.get("born") or {}).get(split) or {}).get("wR2_all"))
            - safe_float(((native.get("dyn") or {}).get(split) or {}).get("wR2_all"))
            for split in ("all", "work", "free")
        },
        "endpoint": {
            "R1_all": shelxl.get("R1_all"),
            "R1_gt": shelxl.get("R1_gt"),
            "wR2_all": shelxl.get("wR2_all"),
            "GooF": shelxl.get("GooF"),
            "n_reflections": endpoint.get("merged_endpoint_rows", endpoint.get("merged_exact_hkl_rows")),
            "npd_count": shelxl.get("npd_count", 0),
            "split_count": shelxl.get("split_count", 0),
            "returncode": shelxl.get("returncode"),
            "status": shelxl.get("status"),
            "safe_for_feedback": endpoint_health.get("safe_for_feedback") is True,
            "hkl_safe_factor": endpoint.get("hkl_safe_factor"),
            "modulation_p95": endpoint.get("mod_p95", endpoint.get("modulation_p95")),
            "artifacts_exist": bool(endpoint_artifacts)
            and all(
                str(endpoint_artifacts.get(key, "")).strip()
                and Path(str(endpoint_artifacts[key])).is_file()
                for key in ("res", "lst", "hkl")
            ),
            "artifacts": dict(endpoint_artifacts),
        },
        "physics_contract": physics_contract,
    }


def write_multi_scale_contract_report(out_dir: Path, decision: Mapping[str, object]) -> None:
    """Write a durable human-readable compact-scale promotion report."""

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "decision.json", dict(decision))
    lines = [
        "# GROND Multi-Dataset Scale-Contract Selection",
        "",
        "Full and compact candidates reuse identical frozen Bloch predictions. Compact promotion is fail-closed and requires matched physics, improved aggregate fitted-variance BIC, bounded all/work/free row costs, bounded endpoint costs, unchanged merged coverage, publication-like GooF, and no new ADP pathology.",
        "",
        f"Status: `{decision.get('status', 'unknown')}`. Reason: `{decision.get('reason', 'completed candidate gate')}`. Wall time: `{safe_float(decision.get('wall_seconds')):.3f} s`.",
        "",
        f"Selected contract: `{decision.get('selected_contract', 'full')}`.",
        "",
        "| contract | scales | aggregate BIC | row wR2 all/work/free | endpoint R1all/R1gt/wR2 | GooF | NPD/split | accepted | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    evaluations = decision.get("evaluations") if isinstance(decision.get("evaluations"), list) else []
    candidates = decision.get("candidates") if isinstance(decision.get("candidates"), Mapping) else {}
    accepted_by_label = {str(row.get("label")): row for row in evaluations if isinstance(row, Mapping)}
    for label, candidate in candidates.items():
        if not isinstance(candidate, Mapping):
            continue
        row = accepted_by_label.get(str(label), {})
        wr = candidate.get("row_wr2") or {}
        ep = candidate.get("endpoint") or {}
        lines.append(
            f"| `{label}` | {candidate.get('scale_params', '')} | {safe_float(candidate.get('aggregate_bic')):.2f} | "
            f"{safe_float(wr.get('all')):.4f}/{safe_float(wr.get('work')):.4f}/{safe_float(wr.get('free')):.4f} | "
            f"{safe_float(ep.get('R1_all')):.4f}/{safe_float(ep.get('R1_gt')):.4f}/{safe_float(ep.get('wR2_all')):.4f} | "
            f"{safe_float(ep.get('GooF')):.3f} | {ep.get('npd_count', '')}/{ep.get('split_count', '')} | "
            f"{row.get('accepted', label == 'full')} | {', '.join(str(reason) for reason in row.get('reasons', []))} |"
        )
    thresholds = decision.get("thresholds") if isinstance(decision.get("thresholds"), Mapping) else {}
    if thresholds:
        lines.extend(["", "## Promotion thresholds", ""])
        for key, value in thresholds.items():
            lines.append(f"{key}: `{value}`.")
    preflight = decision.get("preflight") if isinstance(decision.get("preflight"), Mapping) else {}
    if preflight:
        lines.extend(["", "## Candidate preflight", ""])
        for label, payload in preflight.items():
            if isinstance(payload, Mapping):
                lines.append(
                    f"`{label}`: ok=`{payload.get('ok', False)}`; reason=`{payload.get('reason', 'passed')}`."
                )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def _multi_scale_candidate_runs(
    scale_polish: Mapping[str, Mapping],
    *,
    run_key: str,
) -> tuple[dict[str, Path], str]:
    """Return an all-or-none map of strict materialized compact runs."""

    runs: dict[str, Path] = {}
    for name, payload in scale_polish.items():
        if str(payload.get("status", "")) != "accepted":
            return {}, f"{name}:scale_polish_not_strictly_accepted"
        raw_path = str(payload.get(run_key, "")).strip()
        if not raw_path:
            return {}, f"{name}:missing_{run_key}"
        run_dir = Path(raw_path)
        if not (run_dir / "native_dyn_calc.csv").is_file() or not (run_dir / "summary.json").is_file():
            return {}, f"{name}:incomplete_{run_key}"
        runs[str(name)] = run_dir
    return runs, ""


def _policy_weight_scale_contract_runs(
    runs: Mapping[str, Path],
    policy: Mapping[str, object],
    out_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Apply the already-decided final dataset policy to one scale contract."""

    status = str(policy.get("status", ""))
    if status in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
        copied = {str(name): Path(path) for name, path in runs.items()}
        return copied, copied
    weighted = write_policy_weighted_runs(
        {str(name): Path(path) for name, path in runs.items()},
        dict(policy),
        out_dir,
        include_endpoint_only=True,
    )
    if not weighted:
        raise RuntimeError("multi-dataset scale contract has no endpoint-contributing dataset after final policy")
    publication = policy_active_runs(weighted, dict(policy))
    require_policy_active_runs(publication, context="compact-scale publication")
    return weighted, publication


def write_selected_scale_polish_summaries(
    scale_polish: Mapping[str, Mapping],
    *,
    selected_contract: str,
    selected_publication_runs: Mapping[str, Path],
    out_dir: Path,
) -> dict[str, str]:
    """Project per-dataset scale summaries onto the globally selected rows."""

    if selected_contract not in {"compact_bic", "compact_min_strict"}:
        return {}
    selection_key = "selected_output" if selected_contract == "compact_bic" else "selected"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, selected_run in selected_publication_runs.items():
        payload = scale_polish.get(name)
        if not isinstance(payload, Mapping):
            continue
        source_path = Path(str(payload.get("out_dir", ""))) / "scale_compression_summary.json"
        source = try_read_json(source_path)
        selected = payload.get(selection_key)
        if not source or not isinstance(selected, Mapping):
            continue
        projected = dict(source)
        projected["selected_output"] = dict(selected)
        projected["selected_run_dir"] = str(Path(selected_run).resolve())
        projected["publication_scale_contract"] = {
            "selected_contract": selected_contract,
            "row_source": str(Path(selected_run).resolve()),
            "source_summary": str(source_path.resolve()),
        }
        path = out_dir / f"{safe_label(name)}.json"
        write_json(path, projected)
        written[str(name)] = str(path.resolve())
    return written


def run_multi_scale_contract_selection(
    *,
    python: str,
    full_runs: Mapping[str, Path],
    full_input_runs: Mapping[str, Path],
    full_publication_runs: Mapping[str, Path],
    scale_polish: Mapping[str, Mapping],
    final_dataset_policy: Mapping[str, object],
    final_model: str,
    root: Path,
    logs: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, object]]:
    """Replay and select the smallest earned multi-dataset scale contract.

    The expensive Bloch predictions remain frozen. Every candidate is passed
    through the same dataset policy, sigma calibration, overlap scaling,
    symmetry merge, and fixed-weight SHELXL endpoint as the full-scale control.
    Any incomplete provenance or failed gate leaves the full contract selected.
    """

    mode = str(getattr(args, "multi_scale_contract_selection", "auto"))
    selection_dir = root / "scale_contract_selection"
    started = time.perf_counter()
    thresholds = {
        "minimum_scale_reduction": float(getattr(args, "multi_scale_min_reduction", 0.10)),
        "require_bic_improvement": bool(getattr(args, "multi_scale_require_bic_improvement", True)),
        "maximum_row_wr2_cost": float(getattr(args, "multi_scale_max_row_wr2_cost", 0.01)),
        "maximum_endpoint_r1_cost": float(getattr(args, "multi_scale_max_endpoint_r1_cost", 0.005)),
        "maximum_endpoint_wr2_cost": float(getattr(args, "multi_scale_max_endpoint_wr2_cost", 0.01)),
        "goof_min": float(getattr(args, "multi_scale_goof_min", 0.85)),
        "goof_max": float(getattr(args, "multi_scale_goof_max", 1.15)),
        "maximum_input_row_loss_fraction": float(
            getattr(args, "multi_scale_max_input_row_loss_fraction", 0.001)
        ),
        "minimum_bloch_wr2_gain": float(getattr(args, "multi_scale_min_bloch_wr2_gain", 0.002)),
        "maximum_dataset_row_wr2_cost": float(
            getattr(args, "multi_scale_max_dataset_row_wr2_cost", 0.015)
        ),
        "maximum_preexisting_bloch_gain_loss": float(
            getattr(args, "multi_scale_max_preexisting_bloch_gain_loss", 0.01)
        ),
        "minimum_hkl_safe_factor": float(
            getattr(args, "publication_endpoint_min_hkl_safe_factor", 0.25)
        ),
        "maximum_modulation_p95": float(
            getattr(args, "publication_endpoint_max_modulation_p95", 20.0)
        ),
    }
    decision: dict[str, object] = {
        "status": "disabled" if mode == "off" else "not_applicable",
        "selected_contract": "full",
        "accepted": False,
        "candidates": {},
        "evaluations": [],
        "thresholds": thresholds,
    }
    full_input = {str(name): Path(path) for name, path in full_input_runs.items()}
    full_publication = {str(name): Path(path) for name, path in full_publication_runs.items()}
    decision["full_input_runs"] = {name: str(path) for name, path in full_input.items()}
    decision["full_publication_runs"] = {
        name: str(path) for name, path in full_publication.items()
    }

    def finish(
        selected_input: dict[str, Path],
        selected_publication: dict[str, Path],
    ) -> tuple[dict[str, Path], dict[str, Path], dict[str, object]]:
        decision["wall_seconds"] = time.perf_counter() - started
        decision["artifact_dir"] = str(selection_dir.resolve())
        write_multi_scale_contract_report(selection_dir, decision)
        return selected_input, selected_publication, decision

    if mode == "off" or args.dry_run or len(full_runs) < 2:
        decision["reason"] = "disabled_by_option" if mode == "off" else "dry_run_or_single_dataset"
        return finish(full_input, full_publication)

    endpoint_names = set(full_input)
    if not endpoint_names or not endpoint_names.issubset(full_runs) or not endpoint_names.issubset(scale_polish):
        decision.update(status="failed_closed", reason="endpoint_contributor_contract_incomplete")
        return finish(full_input, full_publication)
    selection_full_runs = {name: Path(full_runs[name]) for name in sorted(endpoint_names)}
    selection_scale_polish = {name: scale_polish[name] for name in sorted(endpoint_names)}
    full_totals = scale_contract_totals(selection_scale_polish, "baseline")
    if full_totals is None:
        decision.update(status="failed_closed", reason="incomplete_full_scale_information_contract")
        return finish(full_input, full_publication)

    specifications = (
        ("compact_bic", "selected_compact_run", "selected_output"),
        ("compact_min_strict", "selected_strict_run", "selected"),
    )
    preflight: dict[str, dict[str, object]] = {}
    min_bloch_gain = float(thresholds["minimum_bloch_wr2_gain"])
    for label, run_key, selection_key in specifications:
        try:
            candidate_runs, reason = _multi_scale_candidate_runs(
                selection_scale_polish,
                run_key=run_key,
            )
            if reason:
                preflight[label] = {"ok": False, "reason": reason}
                continue
            validation = validate_scale_contract_run_map(
                selection_full_runs,
                candidate_runs,
                min_bloch_wr2_gain=min_bloch_gain,
                max_dataset_row_wr2_cost=float(thresholds["maximum_dataset_row_wr2_cost"]),
                max_preexisting_bloch_gain_loss=float(
                    thresholds["maximum_preexisting_bloch_gain_loss"]
                ),
            )
            totals = scale_contract_totals(selection_scale_polish, selection_key)
            if not validation.get("ok"):
                preflight[label] = {
                    "ok": False,
                    "reason": "row_contract_validation_failed",
                    "validation": validation,
                }
                continue
            if totals is None:
                preflight[label] = {
                    "ok": False,
                    "reason": "incomplete_scale_information_contract",
                }
                continue
            materialized_total = sum(
                int((row or {}).get("scale_count", 0))
                for row in (validation.get("datasets") or {}).values()
            )
            if materialized_total != int(totals[0]):
                preflight[label] = {
                    "ok": False,
                    "reason": "materialized_scale_count_mismatch",
                    "reported": int(totals[0]),
                    "materialized": materialized_total,
                }
                continue
            preflight[label] = {
                "ok": True,
                "runs": {name: str(path) for name, path in candidate_runs.items()},
                "scale_params": int(totals[0]),
                "aggregate_bic": float(totals[1]),
                "validation": validation,
            }
        except Exception as exc:
            preflight[label] = {
                "ok": False,
                "reason": "candidate_preflight_failed",
                "failure_stage": "preflight",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    viable = {label: payload for label, payload in preflight.items() if payload.get("ok")}
    if not viable:
        decision.update(status="failed_closed", reason="no_complete_strict_compact_contract", preflight=preflight)
        return finish(full_input, full_publication)

    controls_root = root / "runs" / "combined_scale_contracts"
    full_combined = controls_root / "full"
    run_logged(
        combine_cmd(python, full_input, final_model, full_combined, args),
        logs,
        "combine_scale_contract_full",
        dry_run=False,
    )
    full_candidate = combined_scale_contract_candidate(
        "full",
        full_combined,
        scale_params=int(full_totals[0]),
        aggregate_bic=float(full_totals[1]),
    )
    candidates: dict[str, dict[str, object]] = {"full": full_candidate}
    evaluations: list[dict[str, object]] = []
    selected_payloads: dict[str, tuple[dict[str, Path], dict[str, Path]]] = {}

    def reject_candidate(label: str, *, reason: str, stage: str, exc: Exception) -> None:
        failure = {
            "failure_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        candidates[label] = {
            "label": label,
            "status": "failed",
            "failure": failure,
        }
        evaluations.append(
            {
                "label": label,
                "selected_contract": "full",
                "accepted": False,
                "reasons": [reason],
                "diagnostics": failure,
                "validation": viable[label]["validation"],
            }
        )

    for label, payload in viable.items():
        try:
            candidate_input, candidate_publication = _policy_weight_scale_contract_runs(
                payload["runs"],
                final_dataset_policy,
                root / "dataset_policy" / "final" / f"weighted_runs_{label}",
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_policy_weighting_failed",
                stage="policy_weighting",
                exc=exc,
            )
            continue
        combined_dir = controls_root / label
        try:
            run_logged(
                combine_cmd(python, candidate_input, final_model, combined_dir, args),
                logs,
                f"combine_scale_contract_{label}",
                dry_run=False,
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_replay_failed",
                stage="combined_replay",
                exc=exc,
            )
            continue
        try:
            candidate = combined_scale_contract_candidate(
                label,
                combined_dir,
                scale_params=int(payload["scale_params"]),
                aggregate_bic=float(payload["aggregate_bic"]),
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_summary_parsing_failed",
                stage="summary_parsing",
                exc=exc,
            )
            continue
        candidates[label] = candidate
        try:
            evaluation = select_multi_scale_contract(
                full_candidate,
                candidate,
                min_scale_reduction=float(getattr(args, "multi_scale_min_reduction", 0.10)),
                require_bic_improvement=bool(
                    getattr(args, "multi_scale_require_bic_improvement", True)
                ),
                max_row_wr2_cost=float(getattr(args, "multi_scale_max_row_wr2_cost", 0.01)),
                max_endpoint_r1_cost=float(
                    getattr(args, "multi_scale_max_endpoint_r1_cost", 0.005)
                ),
                max_endpoint_wr2_cost=float(
                    getattr(args, "multi_scale_max_endpoint_wr2_cost", 0.01)
                ),
                goof_min=float(getattr(args, "multi_scale_goof_min", 0.85)),
                goof_max=float(getattr(args, "multi_scale_goof_max", 1.15)),
                max_input_row_loss_fraction=float(
                    getattr(args, "multi_scale_max_input_row_loss_fraction", 0.001)
                ),
                min_bloch_wr2_gain=min_bloch_gain,
                min_hkl_safe_factor=float(
                    getattr(args, "publication_endpoint_min_hkl_safe_factor", 0.25)
                ),
                max_modulation_p95=float(
                    getattr(args, "publication_endpoint_max_modulation_p95", 20.0)
                ),
                require_endpoint_health=True,
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_evaluation_failed",
                stage="selection_gate",
                exc=exc,
            )
            continue
        evaluation = {**evaluation, "label": label, "validation": payload["validation"]}
        evaluations.append(evaluation)
        selected_payloads[label] = (candidate_input, candidate_publication)

    accepted_labels = [
        str(row["label"])
        for row in evaluations
        if bool(row.get("accepted")) and str(row.get("label")) in selected_payloads
    ]
    if accepted_labels:
        selected_label = min(
            accepted_labels,
            key=lambda label: (
                int(candidates[label]["scale_params"]),
                safe_float((candidates[label].get("endpoint") or {}).get("wR2_all"), float("inf")),
            ),
        )
        selected_input, selected_publication = selected_payloads[selected_label]
        accepted = True
        status = "selected_compact"
    else:
        selected_label = "full"
        selected_input, selected_publication = full_input, full_publication
        accepted = False
        status = "kept_full"
    publication_scale_polish_summaries = write_selected_scale_polish_summaries(
        selection_scale_polish,
        selected_contract=selected_label,
        selected_publication_runs=selected_publication,
        out_dir=selection_dir / "publication_scale_polish",
    )
    decision.update(
        status=status,
        selected_contract=selected_label,
        accepted=accepted,
        candidates=candidates,
        evaluations=evaluations,
        preflight=preflight,
        selection_policy="smallest accepted scale count, then endpoint wR2all",
        selected_input_runs={name: str(path) for name, path in selected_input.items()},
        selected_publication_runs={name: str(path) for name, path in selected_publication.items()},
        publication_scale_polish_summaries=publication_scale_polish_summaries,
    )
    return finish(selected_input, selected_publication)


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
        if health_path_matches
        and artifact_sha
        and health_sha
        and artifact_sha == health_sha == current_sha
        else None
    )


def multi_combined_endpoint_shelxl(combined_dir: Path) -> dict[str, object]:
    """Return the recorded SHELXL result for a combined multi-dataset endpoint."""

    summary = try_read_json(combined_dir / "combined_summary.json")
    endpoint = summary.get("combined_endpoint")
    endpoint = endpoint if isinstance(endpoint, Mapping) else {}
    shelxl = endpoint.get("shelxl")
    return dict(shelxl) if isinstance(shelxl, Mapping) else {}


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
    summary: dict = {
        "status": "not_run" if cycles_requested <= 0 else "running",
        "strategy": "shelxl_feedback",
        "start_model": start_model,
        "current_model": start_model,
        "cycles_requested": cycles_requested,
        "shelxl_iterations_per_cycle": int(getattr(args, "shelxl_feedback_shelxl_iterations", 1) or 1),
        "cycles": [],
    }
    if cycles_requested <= 0:
        summary["status"] = "disabled_zero_cycles"
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
                ),
                logs,
                f"{ds.name}_{cycle_label}",
                dry_run=False,
            )
            cycle_runs[ds.name] = run_dir
            selected_nuisance = ((try_read_json(run_dir / "summary.json").get("best_native") or {}).get("nuisance") or {})
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
        modulation_gate_passed = bool(
            math.isfinite(endpoint_modulation_p95)
            and endpoint_modulation_p95 <= max_modulation_p95
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
                "passed": modulation_gate_passed,
                "maximum_p95": max_modulation_p95,
                "reason": (
                    "passed"
                    if modulation_gate_passed
                    else "missing finite modulation p95"
                    if not math.isfinite(endpoint_modulation_p95)
                    else f"modulation p95 {endpoint_modulation_p95:.4g} exceeds {max_modulation_p95:.4g}"
                ),
            },
            "combined": str(combined_dir),
            "runs": {name: str(path) for name, path in cycle_runs.items()},
            "combined_input_runs": {name: str(path) for name, path in cycle_input_runs.items()},
            "nuisance_revalidations": nuisance_revalidations,
        }
        summary["cycles"].append(record)
        if output_model is None:
            summary["status"] = "failed_no_usable_endpoint_model"
            write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
            raise RuntimeError(f"{cycle_label} did not produce a usable shared SHELXL .res model in {combined_dir}")
        if not modulation_gate_passed:
            record["adopted_for_feedback"] = False
            summary["status"] = "stopped_endpoint_modulation_gate"
            summary["stopped_reason"] = (
                "Shared SHELXL feedback cycle lacks a bounded, measured multiplicative intensity-correction "
                f"contract; keeping the previous clean model. {record['endpoint_modulation_gate']['reason']}."
            )
            summary["unsafe_model"] = str(output_model)
            write_json(feedback_root / "multi_shelxl_feedback_loop_summary.json", summary)
            break
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
                endpoint_status == "adp_pathology"
                or endpoint_health.get("has_npd") is True
                or endpoint_npd_count > 0
            )
            if has_adp_pathology:
                summary["status"] = "stopped_adp_pathology"
                summary["stopped_reason"] = (
                    "Shared SHELXL feedback cycle produced ADP pathology; keeping the previous clean model. "
                    f"NPD atoms={endpoint_npd_count}, split warnings={endpoint_split_count}."
                )
                summary["pathological_model"] = str(output_model)
            else:
                reason = str(endpoint_shelxl.get("invalid_res_reason") or endpoint_health.get("reason") or "unsafe endpoint")
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

    return current_model, summary


def run_publication_report(python: str, prep: PrepConfig, result: dict, args: argparse.Namespace, logs: Path) -> dict:
    """Write top-level MD/TeX/PDF publication tables for a completed run."""

    if args.dry_run:
        return {"status": "dry_run"}
    if not (result.get("primary_run") or result.get("runs")):
        return {"status": "no_final_row_tables"}
    out_dir = Path(args.out_dir) / "tables"
    cmd = publication_report_cmd(
        python,
        prep,
        result,
        out_dir,
        title=f"GROND regularization summary: {', '.join(ds.name for ds in prep.datasets)}",
        instability_u=float(getattr(args, "instability_u", 0.01)),
    )
    run_logged(cmd, logs, "publication_row_mr_tables", dry_run=False)
    artifacts_path = out_dir / "publication_artifacts.json"
    artifacts = try_read_json(artifacts_path)
    if artifacts:
        if result.get("resolution_limits"):
            artifacts["resolution_limits"] = result.get("resolution_limits")
        if result.get("merged_resolution_limit"):
            artifacts["merged_resolution_limit"] = result.get("merged_resolution_limit")
        if result.get("dataset_policy"):
            artifacts["dataset_policy"] = result.get("dataset_policy")
        if result.get("publication_runs"):
            artifacts["publication_runs"] = result.get("publication_runs")
        if result.get("publication_full_runs"):
            artifacts["publication_full_runs"] = result.get("publication_full_runs")
        if result.get("scale_contract_selection"):
            artifacts["scale_contract_selection"] = result.get("scale_contract_selection")
        with _RUN_LOG_LOCK:
            write_json(artifacts_path, artifacts)
            if _PROGRESS is not None and not _PROGRESS.refresh_output_fingerprints("publication_row_mr_tables"):
                raise RuntimeError(
                    "publication report was written, but its resumable output fingerprint could not be refreshed"
                )
        return {"status": "ok", **artifacts}
    return {
        "status": "missing_artifact_summary",
        "markdown": str((out_dir / "publication_table.md").resolve()),
        "latex": str((out_dir / "publication_table.tex").resolve()),
        "summary": str((out_dir / "row_mr_metrics.json").resolve()),
    }


def _multi_hand_publication_blockers(result: Mapping[str, object]) -> list[str]:
    """Return fail-closed publication blockers from the final shared-hand audit."""

    if not result.get("combined"):
        return []
    raw = result.get("post_feedback_multi_hand_diagnostic")
    diagnostic = raw if isinstance(raw, Mapping) else {}
    if not diagnostic:
        path_text = str(result.get("post_feedback_multi_hand_diagnostic_path") or "").strip()
        diagnostic = try_read_json(Path(path_text)) if path_text else {}
    status = str(diagnostic.get("status") or "missing")
    consistency = diagnostic.get("consistency_gate") if isinstance(diagnostic.get("consistency_gate"), Mapping) else {}
    explicitly_not_applicable = status in {"not_applicable", "skipped_not_applicable"} and str(
        consistency.get("status") or ""
    ) == "not_applicable"
    if status == "consistent" or explicitly_not_applicable:
        return []
    warnings = [str(value) for value in (consistency.get("warnings") or diagnostic.get("warnings") or []) if str(value)]
    detail = "; ".join(warnings[:4]) or "the diagnostic is missing, incomplete, disabled, or materially favors the opposite hand"
    return [
        f"shared post-feedback hand diagnostic status {status!r} is not publication-consistent: {detail}; "
        "absolute structure remains unresolved and GROND does not auto-switch the hand"
    ]


def run_publication_endpoint_chooser(prep: PrepConfig, result: dict, args: argparse.Namespace) -> dict:
    """Write the publication-vs-diagnostic endpoint chooser report."""

    if not getattr(args, "publication_endpoint_chooser", True):
        return {"status": "disabled"}
    out_dir = Path(getattr(args, "out_dir", ".")) / "endpoint_chooser"
    if getattr(args, "dry_run", False):
        return {"status": "dry_run", "out_dir": str(out_dir)}
    if result.get("combined"):
        runs = {name: Path(run) for name, run in (result.get("publication_runs") or result.get("runs") or {}).items()}
        combined_run = Path(result["combined"])
        forensic_summaries = result.get("forensics") if isinstance(result.get("forensics"), Mapping) else {}
    elif result.get("primary_run"):
        label = prep.datasets[0].name if prep.datasets else "dataset"
        runs = {label: Path(result["primary_run"])}
        combined_run = None
        forensic = result.get("forensics")
        forensic_summaries = {label: forensic} if isinstance(forensic, Mapping) else {}
    else:
        return {"status": "no_endpoint_runs"}
    if not runs and not combined_run:
        return {"status": "no_endpoint_runs"}
    extra_candidates = []
    calibration_candidate = _endpoint_calibration_chooser_candidate(result)
    if calibration_candidate:
        extra_candidates.append(calibration_candidate)
    model_candidate = _publication_model_endpoint_candidate(result)
    if model_candidate:
        extra_candidates.append(model_candidate)
    combined_baseline = _validated_fresh_conventional_baseline(result) if result.get("combined") else {}
    global_publication_blockers = _multi_hand_publication_blockers(result)
    return write_publication_endpoint_choice(
        runs,
        out_dir,
        combined_run=combined_run,
        forensic_summaries=forensic_summaries,
        extra_candidates=extra_candidates,
        min_hkl_safe_factor=getattr(args, "publication_endpoint_min_hkl_safe_factor", 0.25),
        max_modulation_p95=getattr(args, "publication_endpoint_max_modulation_p95", 20.0),
        publication_goof_min=getattr(args, "endpoint_calibration_goof_min", 0.85),
        publication_goof_max=getattr(args, "endpoint_calibration_goof_max", 1.15),
        allow_npd=bool(getattr(args, "publication_allow_npd", False)),
        combined_baseline_r1_all=(
            safe_float(combined_baseline.get("R1_all"), float("nan"))
            if combined_baseline
            else None
        ),
        combined_baseline_source=str(combined_baseline.get("baseline_source") or ""),
        require_combined_baseline=bool(result.get("combined")),
        global_publication_blockers=global_publication_blockers,
    )


def _endpoint_calibration_chooser_candidate(result: Mapping[str, object]) -> dict[str, object]:
    """Expose the calibrated publication replay to the final endpoint chooser."""

    calibration = result.get("endpoint_calibration")
    if not isinstance(calibration, Mapping):
        return {}
    selection = calibration.get("publication_selection")
    if not isinstance(selection, Mapping):
        return {}
    if str(selection.get("status", "")) != "passed":
        return {}
    endpoint = selection.get("publication_endpoint")
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}

    lst = str(endpoint.get("endpoint_lst", "") or "")
    manifest_path = str(endpoint.get("endpoint_manifest", "") or "")
    manifest = try_read_json(Path(manifest_path)) if manifest_path else {}
    manifest_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), Mapping) else {}
    hkl_safe_factor = endpoint.get("hkl_safe_factor")
    try:
        hkl_safe_factor = float(hkl_safe_factor)
    except (TypeError, ValueError):
        hkl_safe_factor = 0.0
    if not math.isfinite(hkl_safe_factor):
        hkl_safe_factor = 0.0
    modulation_p95 = safe_float(
        endpoint.get("mod_p95", endpoint.get("modulation_p95")),
        float("nan"),
    )
    endpoint_b_recorded = endpoint.get("endpoint_b")
    keep_exti_recorded = endpoint.get("keep_exti")
    endpoint_policy_complete = endpoint_b_recorded is not None and keep_exti_recorded is not None
    artifact_provenance = endpoint_artifact_provenance(
        {
            **endpoint,
            "res": str(endpoint.get("endpoint_res", "") or ""),
            "hkl": str(endpoint.get("endpoint_hkl", "") or ""),
            "lst": lst,
            "endpoint_manifest": manifest_path,
            "artifacts": manifest_artifacts,
        }
    )
    pathology = (
        artifact_provenance.get("lst_pathology")
        if isinstance(artifact_provenance.get("lst_pathology"), Mapping)
        else {}
    )
    pathology_complete = str(pathology.get("pathology_status") or "") == "complete"
    recorded_npd = _quick_int(endpoint.get("shelxl_npd_count"))
    recorded_split = _quick_int(endpoint.get("shelxl_split_count"))
    npd_count = max(recorded_npd or 0, int(pathology.get("npd_count", 0) or 0)) if pathology_complete else recorded_npd
    split_count = (
        max(recorded_split or 0, int(pathology.get("split_count", 0) or 0))
        if pathology_complete
        else recorded_split
    )
    raw_recorded_atoms = endpoint.get("shelxl_npd_atoms", [])
    if isinstance(raw_recorded_atoms, str):
        recorded_atoms = [atom.strip() for atom in raw_recorded_atoms.split(",") if atom.strip()]
    elif isinstance(raw_recorded_atoms, (list, tuple, set)):
        recorded_atoms = [str(atom) for atom in raw_recorded_atoms]
    else:
        recorded_atoms = []
    npd_atoms = sorted(
        {
            str(atom)
            for atom in [*recorded_atoms, *((pathology.get("npd_atoms") or []) if pathology_complete else [])]
            if str(atom).strip()
        }
    )
    recorded_has_npd = endpoint.get("shelxl_has_npd")
    recorded_has_split = endpoint.get("shelxl_has_split_warnings")
    has_npd = (
        bool(recorded_has_npd) or bool(pathology.get("has_npd")) or bool((npd_count or 0) > 0)
        if recorded_has_npd is not None or pathology_complete or npd_count is not None
        else None
    )
    has_split = (
        bool(recorded_has_split) or bool(pathology.get("has_split_warnings")) or bool((split_count or 0) > 0)
        if recorded_has_split is not None or pathology_complete or split_count is not None
        else None
    )
    shelxl = {
        "R1_gt": endpoint.get("shelxl_R1gt"),
        "R1_all": endpoint.get("shelxl_R1all"),
        "wR2_all": endpoint.get("shelxl_wR2all"),
        "GooF": endpoint.get("shelxl_GooF"),
        "N_gt": endpoint.get("shelxl_Ngt"),
        "N_all": endpoint.get("shelxl_Nall"),
        "Rint": endpoint.get("shelxl_Rint"),
        "Rsigma": endpoint.get("shelxl_Rsigma"),
        "npd_count": npd_count,
        "split_count": split_count,
        "npd_atoms": npd_atoms,
        "has_npd": has_npd,
        "has_split_warnings": has_split,
        "lst": lst,
    }
    metric_consistency = endpoint_lst_metric_consistency(shelxl, artifact_provenance)
    live_metrics = (
        artifact_provenance.get("lst_metrics")
        if isinstance(artifact_provenance.get("lst_metrics"), Mapping)
        else {}
    )
    live_metrics_complete = str(live_metrics.get("metric_status") or "") == "complete"
    promotion_fields = endpoint_promotion_field_contract(endpoint)
    canonical = (
        artifact_provenance.get("canonical_artifacts")
        if isinstance(artifact_provenance.get("canonical_artifacts"), Mapping)
        else {}
    )
    canonical_res = canonical.get("res") if isinstance(canonical.get("res"), Mapping) else {}
    canonical_hkl = canonical.get("hkl") if isinstance(canonical.get("hkl"), Mapping) else {}
    canonical_lst = canonical.get("lst") if isinstance(canonical.get("lst"), Mapping) else {}
    spec = str(endpoint.get("spec", "calibrated"))
    combined = str(calibration.get("label", "")) == "combined" or bool(result.get("combined"))
    return {
        "id": f"endpoint_calibration:{spec}",
        "label": f"endpoint_calibration:{spec}",
        "source_kind": "combined_endpoint_calibration" if combined else "endpoint_calibration",
        "run_dir": str(endpoint.get("endpoint_dir", calibration.get("out_dir", "")) or ""),
        "lambda_q": calibration.get("lambda_q", 1.0),
        "endpoint_b": safe_float(endpoint_b_recorded, float("nan")),
        "keep_exti": bool(keep_exti_recorded) if keep_exti_recorded is not None else None,
        "endpoint_policy_required": True,
        "endpoint_policy_complete": endpoint_policy_complete,
        "hkl_safe_factor": hkl_safe_factor,
        "modulation_p95": modulation_p95,
        "modulation_metric": str(endpoint.get("modulation_metric") or manifest.get("modulation_metric") or ""),
        "R1_gt": live_metrics.get("R1_gt") if live_metrics_complete else shelxl["R1_gt"],
        "R1_all": live_metrics.get("R1_all") if live_metrics_complete else shelxl["R1_all"],
        "wR2_all": live_metrics.get("wR2_all") if live_metrics_complete else shelxl["wR2_all"],
        "GooF": live_metrics.get("GooF") if live_metrics_complete else shelxl["GooF"],
        "N_gt": live_metrics.get("N_gt") if live_metrics_complete else shelxl["N_gt"],
        "N_all": live_metrics.get("N_all") if live_metrics_complete else shelxl["N_all"],
        "metric_consistency_required": True,
        "metric_consistency_ok": metric_consistency.get("passed") is True,
        "metric_consistency": metric_consistency,
        "npd_count": npd_count,
        "split_count": shelxl["split_count"],
        "npd_atoms": npd_atoms,
        "has_npd": shelxl["has_npd"],
        "has_split_warnings": shelxl["has_split_warnings"],
        "hkl": str(canonical_hkl.get("path") or endpoint.get("endpoint_hkl", "") or ""),
        "lst": str(canonical_lst.get("path") or lst),
        "res": str(canonical_res.get("path") or endpoint.get("endpoint_res", "") or ""),
        "res_sha256": str(canonical_res.get("sha256") or ""),
        "hkl_sha256": str(canonical_hkl.get("sha256") or ""),
        "lst_sha256": str(canonical_lst.get("sha256") or ""),
        "provenance_required": True,
        "artifact_provenance_ok": artifact_provenance.get("passed") is True,
        "artifact_provenance": artifact_provenance,
        "res_provenance_ok": artifact_provenance.get("passed") is True,
        "res_provenance": artifact_provenance,
        "pathology_required": True,
        "pathology_status": str(pathology.get("pathology_status") or "unverified"),
        "promotion_fields_required": True,
        "promotion_fields_ok": promotion_fields.get("passed") is True,
        "promotion_field_contract": promotion_fields,
        "shelxl": shelxl,
        "scale_spec": spec,
        "scale_params": endpoint.get("scale_params"),
        "wght_mode": endpoint.get("wght_mode"),
        "endpoint_manifest": endpoint.get("endpoint_manifest", ""),
        "endpoint_row_table": endpoint.get("endpoint_row_table", ""),
    }


def _publication_model_endpoint_candidate(result: Mapping[str, object]) -> dict[str, object]:
    """Return the last LS1 corrected-HKL endpoint as a chooser candidate."""

    model_text = str(result.get("model", "") or "").strip()
    if not model_text:
        return {}
    res = Path(model_text).expanduser()
    lst = res.with_suffix(".lst")
    hkl = res.with_suffix(".hkl")
    if not res.exists() or not lst.exists():
        return {}
    origin_summary_path = res.parent.parent / (
        "combined_summary.json" if result.get("combined") else "summary.json"
    )
    origin_summary = try_read_json(origin_summary_path)
    origin_endpoint: Mapping[str, object] = {}
    if result.get("combined"):
        value = origin_summary.get("combined_endpoint")
        origin_endpoint = value if isinstance(value, Mapping) else {}
    else:
        raw_endpoints = origin_summary.get("endpoint_results_sorted")
        candidates = raw_endpoints if isinstance(raw_endpoints, list) else []
        best = origin_summary.get("best_endpoint")
        if isinstance(best, Mapping):
            candidates = [best, *candidates]
        for value in candidates:
            if not isinstance(value, Mapping):
                continue
            value_shelxl = value.get("shelxl") if isinstance(value.get("shelxl"), Mapping) else {}
            value_lst = str(value_shelxl.get("lst") or "").strip()
            if value_lst and Path(value_lst).expanduser().with_suffix(".res").resolve() == res.resolve():
                origin_endpoint = value
                break
    origin_shelxl = origin_endpoint.get("shelxl") if isinstance(origin_endpoint.get("shelxl"), Mapping) else {}
    artifact_provenance = endpoint_artifact_provenance(
        {**origin_endpoint, "res": str(res), "hkl": str(hkl), "lst": str(lst)}
    )
    provenance_ok = artifact_provenance.get("passed") is True
    pathology = (
        artifact_provenance.get("lst_pathology")
        if isinstance(artifact_provenance.get("lst_pathology"), Mapping)
        else {}
    )
    pathology_complete = str(pathology.get("pathology_status") or "") == "complete"
    metric_consistency = endpoint_lst_metric_consistency(origin_shelxl, artifact_provenance)
    live_metrics = (
        artifact_provenance.get("lst_metrics")
        if isinstance(artifact_provenance.get("lst_metrics"), Mapping)
        else {}
    )
    live_metrics_complete = str(live_metrics.get("metric_status") or "") == "complete"
    origin_details = dict(origin_shelxl)
    for key in (
        "active_wght_a",
        "active_wght_b",
        "recommended_wght_a",
        "recommended_wght_b",
    ):
        if origin_details.get(key) is None:
            origin_details[key] = origin_endpoint.get(key, origin_endpoint.get(f"shelxl_{key}"))
    details = _quick_shelxl_details(origin_details, lst) if provenance_ok else origin_details
    if live_metrics_complete:
        details.update({key: live_metrics[key] for key in ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")})
    promotion_fields = endpoint_promotion_field_contract(origin_endpoint)
    canonical = (
        artifact_provenance.get("canonical_artifacts")
        if isinstance(artifact_provenance.get("canonical_artifacts"), Mapping)
        else {}
    )
    canonical_res = canonical.get("res") if isinstance(canonical.get("res"), Mapping) else {}
    canonical_hkl = canonical.get("hkl") if isinstance(canonical.get("hkl"), Mapping) else {}
    canonical_lst = canonical.get("lst") if isinstance(canonical.get("lst"), Mapping) else {}
    if _quick_fraction(details.get("R1_all")) is None or _quick_fraction(details.get("wR2_all")) is None:
        return {}
    recorded_npd = _quick_int(details.get("npd_count"))
    recorded_split = _quick_int(details.get("split_count"))
    npd_count = max(recorded_npd or 0, int(pathology.get("npd_count", 0) or 0)) if pathology_complete else recorded_npd
    split_count = (
        max(recorded_split or 0, int(pathology.get("split_count", 0) or 0))
        if pathology_complete
        else recorded_split
    )
    recorded_atoms = details.get("npd_atoms") if isinstance(details.get("npd_atoms"), (list, tuple, set)) else []
    npd_atoms = sorted(
        {
            str(atom)
            for atom in [*recorded_atoms, *((pathology.get("npd_atoms") or []) if pathology_complete else [])]
            if str(atom).strip()
        }
    )
    has_npd = (
        bool(details.get("has_npd")) or bool(pathology.get("has_npd")) or bool((npd_count or 0) > 0)
        if details.get("has_npd") is not None or pathology_complete or npd_count is not None
        else None
    )
    has_split = (
        bool(details.get("has_split_warnings"))
        or bool(pathology.get("has_split_warnings"))
        or bool((split_count or 0) > 0)
        if details.get("has_split_warnings") is not None or pathology_complete or split_count is not None
        else None
    )
    endpoint_policy_complete = "endpoint_b" in origin_endpoint and "keep_exti" in origin_endpoint
    shelxl = {
        "R1_gt": details.get("R1_gt"),
        "R1_all": details.get("R1_all"),
        "wR2_all": details.get("wR2_all"),
        "GooF": details.get("GooF"),
        "N_gt": details.get("N_gt"),
        "N_all": details.get("N_all"),
        "Rint": details.get("Rint"),
        "Rsigma": details.get("Rsigma"),
        "parameters": details.get("parameters"),
        "restraints": details.get("restraints"),
        "npd_count": npd_count,
        "split_count": split_count,
        "npd_atoms": npd_atoms,
        "has_npd": has_npd,
        "has_split_warnings": has_split,
        "pathology_status": str(pathology.get("pathology_status") or "unverified"),
        "active_wght_a": details.get("active_wght_a"),
        "active_wght_b": details.get("active_wght_b"),
        "recommended_wght_a": details.get("recommended_wght_a"),
        "recommended_wght_b": details.get("recommended_wght_b"),
        "lst": str(lst),
    }
    source_kind = "combined_corrected_hkl" if result.get("combined") else "native_corrected_hkl"
    return {
        "id": "final_ls1_corrected_hkl",
        "label": "final_ls1_corrected_hkl",
        "source_kind": source_kind,
        "run_dir": str(res.parent),
        "lambda_q": 1.0,
        "endpoint_b": safe_float(origin_endpoint.get("endpoint_b"), float("nan")),
        "keep_exti": bool(origin_endpoint.get("keep_exti")) if "keep_exti" in origin_endpoint else None,
        "endpoint_policy_required": True,
        "endpoint_policy_complete": endpoint_policy_complete,
        "hkl_safe_factor": (
            safe_float(origin_endpoint.get("hkl_safe_factor"), 0.0)
            if provenance_ok
            else 0.0
        ),
        "modulation_p95": (
            safe_float(origin_endpoint.get("mod_p95", origin_endpoint.get("modulation_p95")), float("nan"))
            if provenance_ok
            else float("inf")
        ),
        "R1_gt": details.get("R1_gt"),
        "R1_all": details.get("R1_all"),
        "wR2_all": details.get("wR2_all"),
        "GooF": details.get("GooF"),
        "N_gt": details.get("N_gt"),
        "N_all": details.get("N_all"),
        "metric_consistency_required": True,
        "metric_consistency_ok": metric_consistency.get("passed") is True,
        "metric_consistency": metric_consistency,
        "npd_count": npd_count,
        "split_count": split_count,
        "npd_atoms": npd_atoms,
        "has_npd": has_npd,
        "has_split_warnings": has_split,
        "hkl": str(canonical_hkl.get("path") or hkl),
        "lst": str(canonical_lst.get("path") or lst),
        "res": str(canonical_res.get("path") or res),
        "res_sha256": str(canonical_res.get("sha256") or ""),
        "hkl_sha256": str(canonical_hkl.get("sha256") or ""),
        "lst_sha256": str(canonical_lst.get("sha256") or ""),
        "shelxl": shelxl,
        "endpoint_origin_summary": str(origin_summary_path),
        "artifact_provenance_ok": provenance_ok,
        "artifact_provenance": artifact_provenance,
        "res_provenance_ok": provenance_ok,
        "provenance_required": True,
        "res_provenance": artifact_provenance,
        "pathology_required": True,
        "pathology_status": str(pathology.get("pathology_status") or "unverified"),
        "promotion_fields_required": True,
        "promotion_fields_ok": promotion_fields.get("passed") is True,
        "promotion_field_contract": promotion_fields,
        "wght_mode": origin_endpoint.get("wght_mode"),
    }


def _endpoint_calibration_lambda(summary: Mapping[str, object], args: argparse.Namespace) -> float:
    """Return the diagnostic q exponent to replay for a frozen row table."""

    del summary
    value = safe_float(getattr(args, "endpoint_calibration_lambda_q", 1.0), 1.0)
    return value if value > 0.0 else 1.0


def _publication_baseline_hkl(prep: PrepConfig) -> str:
    """Return the conventional HKL that is comparable to the publication endpoint."""

    if str(prep.baseline_hkl or "").strip():
        return str(prep.baseline_hkl)
    if len(prep.datasets) == 1 and str(prep.datasets[0].baseline_hkl or "").strip():
        return str(prep.datasets[0].baseline_hkl)
    return ""


def run_fresh_conventional_baseline(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
) -> dict[str, object]:
    """Replay the supplied conventional model/HKL once with exact provenance."""

    hkl_text = _publication_baseline_hkl(prep)
    if not hkl_text:
        return {
            "status": "unavailable",
            "reason": "no global conventional baseline HKL was supplied",
        }
    out_dir = Path(args.out_dir) / "controls" / "fresh_conventional_baseline"
    if args.dry_run:
        return {
            "status": "dry_run",
            "res": str(prep.structure),
            "hkl": hkl_text,
            "out_dir": str(out_dir),
        }
    run_logged(
        [
            python,
            str(engine_script("run_shelxl_baseline.py")),
            "--res",
            str(prep.structure),
            "--hkl",
            hkl_text,
            "--out-dir",
            str(out_dir),
            "--shelxl",
            str(args.shelxl),
        ],
        logs,
        "fresh_conventional_baseline",
        dry_run=False,
    )
    summary = try_read_json(out_dir / "summary.json")
    if not summary:
        raise RuntimeError(f"fresh conventional baseline did not write {out_dir / 'summary.json'}")
    return {**summary, "out_dir": str(out_dir), "summary": str(out_dir / "summary.json")}


def _provenance_file_is_current(record: Mapping[str, object] | None) -> bool:
    """Return true when a provenance record still matches the file on disk."""

    if not isinstance(record, Mapping):
        return False
    path_text = str(record.get("path") or "").strip()
    expected = str(record.get("sha256") or "").strip()
    if not path_text or not expected:
        return False
    path = Path(path_text).expanduser()
    try:
        return bool(path.is_file() and sha256_file(path) == expected)
    except OSError:
        return False


def _validated_fresh_conventional_baseline(result: Mapping[str, object]) -> dict[str, object]:
    """Return a healthy, hash-bound supplied-input conventional baseline."""

    baseline = result.get("conventional_baseline")
    if not isinstance(baseline, Mapping) or str(baseline.get("status") or "") != "ok":
        return {}
    if baseline.get("safe_for_feedback") is not True:
        return {}
    inputs = baseline.get("inputs") if isinstance(baseline.get("inputs"), Mapping) else {}
    outputs = baseline.get("outputs") if isinstance(baseline.get("outputs"), Mapping) else {}
    required_records = [
        inputs.get("res"),
        inputs.get("hkl"),
        outputs.get("ins"),
        outputs.get("hkl"),
        outputs.get("res"),
        outputs.get("lst"),
    ]
    if not all(_provenance_file_is_current(record) for record in required_records):
        return {}
    shelxl = baseline.get("shelxl") if isinstance(baseline.get("shelxl"), Mapping) else {}
    health = shelxl.get("res_health") if isinstance(shelxl.get("res_health"), Mapping) else {}
    r1_all = safe_float(shelxl.get("R1_all"), float("nan"))
    npd_count = int(safe_float(shelxl.get("npd_count", health.get("npd_count", 0)), 0.0))
    split_count = int(safe_float(shelxl.get("split_count", health.get("split_count", 0)), 0.0))
    if not (
        math.isfinite(r1_all)
        and safe_float(shelxl.get("returncode"), 1.0) == 0.0
        and npd_count == 0
        and split_count == 0
        and health.get("safe_for_feedback") is not False
    ):
        return {}
    out = dict(shelxl)
    out.update(
        {
            "R1_all": r1_all,
            "npd_count": npd_count,
            "split_count": split_count,
            "baseline_source": "fresh_supplied_input_conventional_control",
            "res": str((outputs.get("res") or {}).get("path") or ""),
            "hkl": str((outputs.get("hkl") or {}).get("path") or ""),
            "lst": str((outputs.get("lst") or {}).get("path") or ""),
            "source_res": str((inputs.get("res") or {}).get("path") or ""),
            "source_hkl": str((inputs.get("hkl") or {}).get("path") or ""),
            "summary": str(baseline.get("summary") or ""),
        }
    )
    return out


def _fresh_baseline_shelxl(summary: Mapping[str, object]) -> dict[str, object]:
    """Return a healthy conventional baseline rerun from a native summary."""

    baseline = summary.get("baseline_exti") if isinstance(summary, Mapping) else None
    if not isinstance(baseline, Mapping) or not baseline:
        return {}
    health = baseline.get("res_health")
    health = health if isinstance(health, Mapping) else {}
    r1_all = safe_float(baseline.get("R1_all"), float("nan"))
    status = str(baseline.get("status") or "")
    npd_count = int(safe_float(baseline.get("npd_count", health.get("npd_count", 0)), 0.0))
    split_count = int(safe_float(baseline.get("split_count", health.get("split_count", 0)), 0.0))
    safe_for_feedback = health.get("safe_for_feedback")
    healthy = bool(
        math.isfinite(r1_all)
        and status in {"", "ok"}
        and npd_count == 0
        and split_count == 0
        and safe_for_feedback is not False
    )
    if not healthy:
        return {}
    out = dict(baseline)
    out["R1_all"] = r1_all
    out["npd_count"] = npd_count
    out["split_count"] = split_count
    out["baseline_source"] = "final_run_summary.baseline_exti"
    if health.get("path"):
        out["res"] = str(health["path"])
    return out


def _endpoint_baseline_no_harm_gate(
    summary: Mapping[str, object],
    endpoint: Mapping[str, object],
    *,
    r1_tolerance: float = 0.0,
    baseline_override: Mapping[str, object] | None = None,
    require_baseline: bool = False,
) -> dict[str, object]:
    """Reject a corrected endpoint whose R1all is worse than a fresh baseline."""

    baseline = dict(baseline_override or {}) or _fresh_baseline_shelxl(summary)
    if not baseline:
        return {
            "status": "failed_missing_baseline" if require_baseline else "not_applicable",
            "passed": not require_baseline,
            "reason": "fresh healthy conventional baseline is unavailable",
        }
    baseline_r1 = safe_float(baseline.get("R1_all"), float("nan"))
    endpoint_r1 = safe_float(
        endpoint.get("shelxl_R1all", endpoint.get("R1_all")),
        float("nan"),
    )
    tolerance = max(0.0, float(r1_tolerance))
    passed = bool(math.isfinite(endpoint_r1) and endpoint_r1 <= baseline_r1 + tolerance)
    delta = endpoint_r1 - baseline_r1 if math.isfinite(endpoint_r1) else float("nan")
    reasons = []
    if not math.isfinite(endpoint_r1):
        reasons.append("corrected endpoint R1all is unavailable")
    elif not passed:
        reasons.append(
            f"corrected endpoint R1all {endpoint_r1:.6g} is worse than fresh conventional "
            f"baseline {baseline_r1:.6g} by {delta:.6g}"
        )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "baseline_source": baseline.get("baseline_source"),
        "baseline_R1all": baseline_r1,
        "endpoint_R1all": endpoint_r1,
        "delta_R1all": delta,
        "r1_tolerance": tolerance,
        "baseline_res": baseline.get("res", ""),
        "baseline_hkl": baseline.get("hkl", ""),
        "baseline_lst": baseline.get("lst", ""),
        "reasons": reasons,
    }


def _recorded_native_row_tables(
    result: Mapping[str, object],
    *,
    include_endpoint_only: bool = False,
) -> dict[str, str]:
    """Return the native row tables that define the raw-row result.

    ``publication_runs`` excludes completeness-only crystals from raw-row
    metrics while retaining primary and downweighted datasets.  The broader
    ``combined_input_runs`` contract is useful only when endpoint-completeness
    provenance is explicitly requested.
    """

    if result.get("combined"):
        runs = (
            result.get("combined_input_runs")
            if include_endpoint_only
            else result.get("publication_runs")
        ) or result.get("combined_input_runs") or result.get("runs") or {}
        if isinstance(runs, Mapping):
            return {
                str(name): str(Path(str(run_dir)).expanduser() / "native_dyn_calc.csv")
                for name, run_dir in sorted(runs.items(), key=lambda item: str(item[0]))
                if str(run_dir)
            }
        return {}
    primary = str(result.get("primary_run") or "").strip()
    return {"dataset": str(Path(primary).expanduser() / "native_dyn_calc.csv")} if primary else {}


def _multi_endpoint_calibration_not_applicable(result: Mapping[str, object]) -> dict[str, object]:
    """Describe why single-table scale replay is unsafe for a multi run."""

    combined = Path(str(result.get("combined") or "")).expanduser()
    row_tables = _recorded_native_row_tables(result, include_endpoint_only=True)
    return {
        "status": "not_applicable_multi_dataset",
        "reason": (
            "Multi-dataset scale contracts remain independent by crystal. The single-table endpoint-calibration "
            "replay would incorrectly share frame groups and detector scales, and would bypass the validated "
            "per-dataset sigma, policy-weight, overlap-scale, symmetry-merge, and overdispersion sequence. "
            "The combined corrected-HKL endpoint and publication chooser remain authoritative."
        ),
        "combined_summary": str(combined / "combined_summary.json") if str(combined) not in {"", "."} else "",
        "row_tables": row_tables,
        "row_tables_exist": {name: Path(path).exists() for name, path in row_tables.items()},
        "future_contract": "replay each scale candidate independently per dataset, then evaluate through the normal combined endpoint path",
    }


def run_endpoint_calibration(prep: PrepConfig, result: dict, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Replay the final frozen row table through the endpoint-calibration chooser."""

    mode = getattr(args, "endpoint_calibration", "auto")
    if mode == "off":
        return {"status": "disabled"}
    if mode == "auto" and getattr(args, "fast", False):
        return {"status": "disabled_fast_mode"}
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(Path(args.out_dir) / "endpoint_calibration")}
    if result.get("combined"):
        if mode == "on":
            raise RuntimeError(
                "--endpoint-calibration on is not supported for multi-dataset runs: scale contracts must be "
                "replayed independently per dataset and then passed through the normal combined endpoint. "
                "Use --endpoint-calibration auto (safe explicit skip) or calibrate each dataset separately."
            )
        return _multi_endpoint_calibration_not_applicable(result)
    label = "combined"
    run_dir: Path | None = None
    if result.get("combined"):
        run_dir = Path(str(result["combined"]))
    elif result.get("primary_run"):
        label = prep.datasets[0].name if prep.datasets else "dataset"
        run_dir = Path(str(result["primary_run"]))
    if run_dir is None:
        return {"status": "no_frozen_row_table"}
    row_table = run_dir / "native_dyn_calc.csv"
    summary_path = run_dir / "summary.json"
    if not row_table.exists() or not summary_path.exists():
        return {
            "status": "missing_frozen_row_table",
            "row_table": str(row_table),
            "summary": str(summary_path),
        }
    summary = try_read_json(summary_path)
    lambda_q = _endpoint_calibration_lambda(summary, args)
    out_dir = Path(args.out_dir) / "endpoint_calibration" / safe_label(label)
    cmd = [
        python,
        str(engine_script("scale_contract_replay.py")),
        "--row-table",
        str(row_table),
        "--summary",
        str(summary_path),
        "--out-dir",
        str(out_dir),
        "--specs",
        str(getattr(args, "endpoint_calibration_specs", "")),
        "--lambda-q",
        f"{lambda_q:g}",
        "--post-bloch-sigma-policy",
        resolved_post_bloch_sigma_policy(args, final_endpoint=True),
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{getattr(args, 'endpoint_calibration_sigma_min_scale', 0.05):g}",
        "--post-bloch-sigma-max-scale",
        f"{getattr(args, 'endpoint_calibration_sigma_max_scale', 4.0):g}",
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(getattr(args, "endpoint_calibration_shelxl_iterations", 1)),
        "--wght-modes",
        str(getattr(args, "endpoint_calibration_wght_modes", "freeze-zero,promote")),
        "--publication-goof-min",
        f"{getattr(args, 'endpoint_calibration_goof_min', 0.85):g}",
        "--publication-goof-max",
        f"{getattr(args, 'endpoint_calibration_goof_max', 1.15):g}",
        "--publication-max-r1-cost",
        f"{getattr(args, 'endpoint_calibration_max_r1_cost', 0.010):g}",
        "--publication-free-wr2-tolerance",
        f"{getattr(args, 'endpoint_calibration_free_wr2_tolerance', 0.040):g}",
        "--publication-max-modulation-p95",
        f"{getattr(args, 'publication_endpoint_max_modulation_p95', 20.0):g}",
        "--publication-prefer-wght-mode",
        str(getattr(args, "endpoint_calibration_prefer_wght_mode", "freeze-zero")),
        "--publication-selection-policy",
        str(getattr(args, "endpoint_calibration_selection_policy", "bic")),
        "--publication-information-criterion",
        str(getattr(args, "endpoint_calibration_information_criterion", "bic_fitted_variance")),
        "--instability-u",
        f"{getattr(args, 'instability_u', 0.01):g}",
    ]
    if bool(getattr(args, "publication_allow_npd", False)):
        cmd.append("--publication-allow-npd")
    try:
        run_logged(cmd, logs, f"{safe_label(label)}_endpoint_calibration", dry_run=False)
    except subprocess.CalledProcessError as exc:
        if mode == "on":
            raise
        return {"status": "failed", "returncode": exc.returncode, "command": exc.cmd, "out_dir": str(out_dir)}
    replay_summary = try_read_json(out_dir / "summary.json")
    selection = replay_summary.get("publication_selection", {})
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    endpoint = selection.get("publication_endpoint")
    endpoint = endpoint if isinstance(endpoint, Mapping) else {}
    if endpoint:
        no_harm_gate = _endpoint_baseline_no_harm_gate(summary, endpoint)
        selection["baseline_no_harm_gate"] = no_harm_gate
        if bool(no_harm_gate.get("passed", False)):
            checked_endpoint = dict(endpoint)
            checked_endpoint["baseline_no_harm_gate"] = no_harm_gate
            selection["publication_endpoint"] = checked_endpoint
        else:
            selection["status"] = "failed_baseline_no_harm_gate"
            selection["baseline_rejected_endpoint"] = dict(endpoint)
            selection.pop("publication_endpoint", None)
    return {
        "status": "ok",
        "label": label,
        "run_dir": str(run_dir),
        "lambda_q": lambda_q,
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "endpoint_csv": str((out_dir / "scale_contract_endpoint_replay.csv").resolve()),
        "publication_selection": selection,
    }


def run_scale_window_polish(python: str, run_dir: Path, out_dir: Path, logs: Path, label: str, args: argparse.Namespace) -> dict:
    """Run the post-hoc scale-window compression gate for a finished row table."""

    mode = getattr(args, "scale_polish", "auto")
    if mode == "off":
        return {"status": "disabled"}
    row_table = run_dir / "native_dyn_calc.csv"
    summary = run_dir / "summary.json"
    if args.dry_run:
        return {"status": "dry_run", "row_table": str(row_table), "out_dir": str(out_dir)}
    if not row_table.exists():
        return {"status": "missing_row_table", "row_table": str(row_table)}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(engine_script("scale_compression_polish.py")),
        "--row-table",
        str(row_table),
        "--summary",
        str(summary),
        "--out-dir",
        str(out_dir),
        "--title",
        f"GROND scale-window polish: {label}",
        "--group-strides",
        "1",
        "--window-scan-deg",
        str(getattr(args, "scale_polish_window_deg", 5.0)),
        "--detector-bins",
        getattr(args, "scale_polish_detector_bins", "exact"),
        "--scale-objective",
        getattr(args, "scale_polish_objective", "native"),
        "--tolerance",
        str(getattr(args, "scale_polish_tolerance", 0.01)),
        "--selection-metrics",
        getattr(args, "scale_polish_selection_metrics", "Rall,wRall,native_R1_all,native_wR2_all"),
        "--frame-models",
        getattr(args, "scale_polish_frame_models", "adaptive,fixed"),
        "--adaptive-frame-counts",
        getattr(args, "scale_polish_adaptive_counts", "auto"),
    ]
    if getattr(args, "scale_polish_scan_offsets", False):
        cmd.append("--scan-offsets")
    try:
        run_logged(cmd, logs, f"{safe_label(label)}_scale_window_polish", dry_run=False)
    except subprocess.CalledProcessError as exc:
        if mode == "on":
            raise
        return {"status": "failed", "returncode": exc.returncode, "command": exc.cmd, "out_dir": str(out_dir)}
    summary_payload = try_read_json(out_dir / "scale_compression_summary.json")
    selected = summary_payload.get("selected")
    selected_output = summary_payload.get("selected_output")
    widest = summary_payload.get("selected_widest")
    return {
        "status": "accepted" if selected else "diagnostic_only",
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "candidate_csv": str((out_dir / "scale_compression_candidates.csv").resolve()),
        "selected_compact_run": str((out_dir / "selected_compact_run").resolve()) if (out_dir / "selected_compact_run").exists() else "",
        "selected_strict_run": str((out_dir / "selected_strict_run").resolve()) if (out_dir / "selected_strict_run").exists() else "",
        "selected": selected or {},
        "selected_output": selected_output or {},
        "baseline": summary_payload.get("baseline") or {},
        "selected_information": summary_payload.get("selected_information") or {},
        "selected_widest": widest or {},
    }


def native_dyn_metrics(summary: Mapping) -> dict:
    """Return all/work/free dynamical metrics from a native-stage summary."""

    best = summary.get("best_native") if isinstance(summary, Mapping) else {}
    metrics = best.get("metrics") if isinstance(best, Mapping) else {}
    out = {}
    for split in ("all", "work", "free"):
        out[split] = ((metrics.get(split) or {}).get("dyn") or {}) if isinstance(metrics, Mapping) else {}
    return out


def sg_window_score(record: Mapping) -> tuple[float, float, float]:
    """Rank SG-window candidates by held-out behavior before work/all residuals."""

    free = record.get("free") or {}
    all_rows = record.get("all") or {}
    return (
        safe_float(free.get("wR2_all"), float("inf")),
        safe_float(all_rows.get("wR2_all"), float("inf")),
        safe_float(all_rows.get("R1_all"), float("inf")),
    )


def run_sg_window_scout(
    python: str,
    prep: PrepConfig,
    ds: DatasetSpec,
    structure: str,
    run_dir: Path,
    out_dir: Path,
    logs: Path,
    label: str,
    args: argparse.Namespace,
) -> dict:
    """Run a late, held-out scout over simple fixed Sg rocking-window widths."""

    mode = getattr(args, "sg_window_scout", "auto")
    if mode == "off" or getattr(args, "fast", False):
        return {"status": "disabled"}
    widths = parse_numeric_grid(getattr(args, "sg_window_scout_grid", ""))
    if not widths:
        return {"status": "empty_grid"}
    pinned = best_nuisance(run_dir)
    if not pinned:
        return {"status": "missing_nuisance", "run_dir": str(run_dir)}
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "widths": widths}

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_width = float(getattr(args, "sg_window", 0.004))
    candidates: list[dict] = []
    for width in widths:
        tag = str(width).replace("-", "m").replace(".", "p")
        trial_dir = out_dir / f"fixed_sg_{tag}"
        cmd = native_stage_cmd(
            python,
            ds,
            structure,
            dataset_baseline_hkl(prep, ds),
            trial_dir,
            args,
            n_basis=int(args.sg_window_scout_n_basis),
            n_points=int(args.sg_window_scout_n_points),
            stage=f"sg_window_scout_{tag}",
            pinned=pinned,
            make_maps=False,
            endpoint=False,
            max_rows_override=int(args.sg_window_scout_max_rows) or None,
            sg_window_override=width,
            sg_window_mode_override="fixed",
        )
        try:
            run_logged(cmd, logs, f"{safe_label(label)}_sg_window_scout_{tag}", dry_run=False)
            summary = read_json(trial_dir / "summary.json")
            metrics = native_dyn_metrics(summary)
            candidates.append(
                {
                    "status": "ok",
                    "sg_window": float(width),
                    "run_dir": str(trial_dir),
                    "n_basis": int(args.sg_window_scout_n_basis),
                    "n_points": int(args.sg_window_scout_n_points),
                    "all": metrics["all"],
                    "work": metrics["work"],
                    "free": metrics["free"],
                    "score": score_native_summary(summary),
                }
            )
        except subprocess.CalledProcessError as exc:
            if mode == "on":
                raise
            candidates.append(
                {
                    "status": "failed",
                    "sg_window": float(width),
                    "run_dir": str(trial_dir),
                    "returncode": exc.returncode,
                    "command": exc.cmd,
                }
            )

    ok = [row for row in candidates if row.get("status") == "ok"]
    current = min(ok, key=lambda r: abs(float(r["sg_window"]) - baseline_width), default=None)
    best = min(ok, key=sg_window_score, default=None)
    accepted = False
    reason = "no_successful_candidates"
    if current and best:
        best_all = best.get("all") or {}
        cur_all = current.get("all") or {}
        best_free = best.get("free") or {}
        cur_free = current.get("free") or {}
        all_gain = safe_float(cur_all.get("wR2_all"), float("inf")) - safe_float(best_all.get("wR2_all"), float("inf"))
        free_penalty = safe_float(best_free.get("wR2_all"), float("inf")) - safe_float(cur_free.get("wR2_all"), float("inf"))
        same_width = abs(float(best["sg_window"]) - float(current["sg_window"])) <= 1e-12
        accepted = bool(
            same_width
            or (
                all_gain >= float(args.sg_window_scout_min_wR2_gain)
                and free_penalty <= float(args.sg_window_scout_free_wR2_tolerance)
            )
        )
        reason = (
            "baseline_width_is_best"
            if same_width
            else (
                "held_out_gate_passed"
                if accepted
                else "held_out_gate_rejected_lower_work_residual_or_free_penalty"
            )
        )
    summary = {
        "status": "accepted" if accepted else "diagnostic_only",
        "reason": reason,
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "baseline_sg_window": baseline_width,
        "current": current or {},
        "selected": best or {},
        "candidates": candidates,
        "policy": {
            "min_wR2_gain": float(args.sg_window_scout_min_wR2_gain),
            "free_wR2_tolerance": float(args.sg_window_scout_free_wR2_tolerance),
            "note": "This scout only promotes a global fixed rocking-window width. Per-row window choice from Iobs is deliberately not implemented.",
        },
    }
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# Sg Rocking-Window Scout",
        "",
        "This is a late diagnostic over a few fixed top-hat excitation-error widths. It uses the final pinned nuisance parameters and held-out rows; it does not choose a rocking width separately for each reflection.",
        "",
        "| Sg window | status | all R1 % | all wR2 % | free R1 % | free wR2 % | run |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        all_rows = row.get("all") or {}
        free = row.get("free") or {}
        lines.append(
            f"| {safe_float(row.get('sg_window'), float('nan')):.5f} | {row.get('status', '')} | "
            f"{100*safe_float(all_rows.get('R1_all'), float('nan')):.2f} | "
            f"{100*safe_float(all_rows.get('wR2_all'), float('nan')):.2f} | "
            f"{100*safe_float(free.get('R1_all'), float('nan')):.2f} | "
            f"{100*safe_float(free.get('wR2_all'), float('nan')):.2f} | {row.get('run_dir', '')} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{summary['status']}`. Reason: `{reason}`.",
            "Per-reflection post-hoc rocking-window selection is not part of GROND because it would fit the observed intensity unless the window is supplied by independent raw-image profile metadata.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(out_dir / "summary.json", summary)
    return summary


def dataset_with_transform(ds: DatasetSpec, transform: str) -> DatasetSpec:
    """Return a dataset copy using a different HKL transform."""

    return DatasetSpec(
        name=ds.name,
        hkl=ds.hkl,
        xds_dir=ds.xds_dir,
        baseline_hkl=ds.baseline_hkl,
        hkl_transform=transform,
        min_d_spacing=ds.min_d_spacing,
        min_peak=ds.min_peak,
        target_mode=ds.target_mode,
        include_nonpositive=ds.include_nonpositive,
        geometry_percentile=ds.geometry_percentile,
    )


def transform_scout_records(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    scout_root: Path,
    transforms: list[str],
    *,
    scout_stage: str,
    n_basis: int,
    n_points: int,
    max_rows: int,
) -> list[dict]:
    """Run one transform-scout stage and return scored decision records."""

    def run_one(rank: int, transform: str) -> dict:
        cand = dataset_with_transform(ds, transform)
        out_dir = scout_root / safe_label(transform)
        base_record = {
            "transform": transform,
            "out_dir": str(out_dir),
            "scout_stage": scout_stage,
            "n_basis": int(n_basis),
            "n_points": int(n_points),
            "max_rows": int(max_rows),
            "rank_in": rank,
        }
        try:
            cmd = native_stage_cmd(
                python,
                cand,
                prep.structure,
                dataset_baseline_hkl(prep, cand),
                out_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage=f"scout_transform_{scout_stage}_{safe_label(transform)}",
                pinned=compact_base_nuisance(args),
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                workers_override=transform_scout_worker_count(args),
            )
            if args.dry_run:
                run_logged(cmd, logs, f"{ds.name}_scout_transform_{scout_stage}_{safe_label(transform)}", dry_run=True)
            else:
                semaphore = transform_scout_semaphore(args)
                with semaphore:
                    run_logged(cmd, logs, f"{ds.name}_scout_transform_{scout_stage}_{safe_label(transform)}", dry_run=False)
            if args.dry_run:
                return {**base_record, "status": "dry_run", "score": float("inf")}
            summary = read_json(out_dir / "summary.json")
            score = score_native_summary(summary)
            return {**base_record, "status": "ok", **score}
        except Exception as exc:
            return {**base_record, "status": "failed", "error": str(exc), "score": float("inf")}

    if len(transforms) <= 1:
        return [run_one(rank, transform) for rank, transform in enumerate(transforms, start=1)]
    records: list[dict] = []
    max_workers = transform_scout_job_count(args, len(transforms))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"grond-{ds.name}-{scout_stage}-transform") as pool:
        futures = [pool.submit(run_one, rank, transform) for rank, transform in enumerate(transforms, start=1)]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda row: int(row.get("rank_in", 0)))


def ranked_transform_records(records: list[dict], *, dry_run: bool) -> list[dict]:
    """Return transform scout records in promotion order."""

    if dry_run:
        return [r for r in records if r.get("status") == "dry_run"]
    ok = [r for r in records if r.get("status") == "ok" and float(r.get("score", float("inf"))) < float("inf")]
    return sorted(ok, key=lambda r: float(r["score"]))


def choose_transform_by_scout(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    scout_max_rows: int | None = None,
) -> tuple[DatasetSpec, dict]:
    """Choose the HKL alignment using cheap all-candidate scouts, then rerank top hits."""

    candidates = expand_transform_candidates(ds.hkl_transform, args.transform_candidates)
    if getattr(args, "trust_hkl_transform", False):
        selected_transform = ds.hkl_transform or "identity"
        selected = {
            "transform": selected_transform,
            "status": "trusted_input",
            "score": None,
            "reason": "user supplied --trust-hkl-transform; transform scouting was skipped",
        }
        payload = {
            "records": [],
            "coarse_records": [],
            "rerank_records": [],
            "coarse_selected": selected,
            "selected": selected,
            "rerank_top": 0,
            "metadata_transform_decision": {},
            "scout_parallelism": {
                "max_concurrent_scout_jobs": 0,
                "workers_per_scout_job": 0,
                "coarse_candidate_count": 0,
                "coarse_parallel_jobs": 0,
                "reranked_candidate_count": 0,
                "reranked_parallel_jobs": 0,
            },
            "rule": "trusted user-supplied HKL transform; no Bloch transform scout was run",
        }
        write_json(root / "decisions" / ds.name / "transform_decision.json", payload)
        return dataset_with_transform(ds, selected_transform), payload

    metadata_transform_decision: dict = {}
    baseline_hkl = dataset_baseline_hkl(prep, ds)
    if getattr(args, "transform_metadata_prefilter", "auto") != "off" and baseline_hkl:
        candidates, metadata_transform_decision = metadata_prefilter_transforms(
            ds.hkl,
            baseline_hkl,
            candidates,
            mode=getattr(args, "transform_metadata_prefilter", "auto"),
            max_candidates=int(getattr(args, "transform_metadata_max_candidates", 3)),
            min_pairs=int(getattr(args, "transform_metadata_min_pairs", 100)),
            min_correlation=float(getattr(args, "transform_metadata_min_correlation", 0.70)),
            min_correlation_gap=float(getattr(args, "transform_metadata_min_correlation_gap", 0.05)),
            max_rows=int(getattr(args, "transform_metadata_max_rows", 0)),
        )
    scout_root = root / "decisions" / ds.name / "transform_scout"
    coarse_max_rows = args.scout_max_rows if scout_max_rows is None else scout_max_rows
    coarse_records = transform_scout_records(
        prep,
        ds,
        args,
        python,
        logs,
        scout_root,
        candidates,
        scout_stage="coarse",
        n_basis=args.scout_n_basis,
        n_points=args.scout_n_points,
        max_rows=coarse_max_rows,
    )
    ranked_coarse = ranked_transform_records(coarse_records, dry_run=args.dry_run)
    if not ranked_coarse and not args.dry_run:
        write_json(root / "decisions" / ds.name / "transform_decision.json", {"records": coarse_records, "coarse_records": coarse_records, "selected": None})
        raise RuntimeError(f"all transform scouts failed for dataset {ds.name}; see {scout_root}")

    coarse_selected = ranked_coarse[0] if ranked_coarse else {"transform": ds.hkl_transform}
    rerank_top = max(0, int(getattr(args, "transform_rerank_top", 0)))
    rerank_records: list[dict] = []
    selected = coarse_selected
    if len(candidates) > 1 and rerank_top > 0 and ranked_coarse:
        top_records = ranked_coarse[: min(rerank_top, len(ranked_coarse))]
        top_transforms = [str(r["transform"]) for r in top_records]
        rerank_max_rows = int(getattr(args, "transform_rerank_max_rows", 0) or 0)
        if rerank_max_rows <= 0:
            rerank_max_rows = int(coarse_max_rows)
        rerank_records = transform_scout_records(
            prep,
            ds,
            args,
            python,
            logs,
            scout_root / "updated",
            top_transforms,
            scout_stage="updated",
            n_basis=int(args.transform_rerank_n_basis),
            n_points=int(args.transform_rerank_n_points),
            max_rows=rerank_max_rows,
        )
        ranked_rerank = ranked_transform_records(rerank_records, dry_run=args.dry_run)
        if ranked_rerank:
            selected = ranked_rerank[0]

    records = [*coarse_records, *rerank_records]
    payload = {
        "records": records,
        "coarse_records": coarse_records,
        "rerank_records": rerank_records,
        "coarse_selected": coarse_selected,
        "selected": selected,
        "rerank_top": rerank_top,
        "metadata_transform_decision": metadata_transform_decision,
        "scout_parallelism": {
            "max_concurrent_scout_jobs": transform_scout_job_count(args),
            "workers_per_scout_job": transform_scout_worker_count(args),
            "coarse_candidate_count": len(candidates),
            "coarse_parallel_jobs": transform_scout_job_count(args, len(candidates)),
            "reranked_candidate_count": len(rerank_records),
            "reranked_parallel_jobs": transform_scout_job_count(args, len(rerank_records)) if rerank_records else 0,
        },
        "rule": "metadata HKL-overlap prefilter when confident, then coarse transform scan followed by higher N/Q rerank of the top candidates",
    }
    write_json(root / "decisions" / ds.name / "transform_decision.json", payload)
    return dataset_with_transform(ds, selected["transform"]), payload


def raw_free_dyn_wr2(summary: Mapping[str, object]) -> float:
    """Return the recorded free-row dynamical wR2 without score fallbacks."""

    best = summary.get("best_native")
    best = best if isinstance(best, Mapping) else {}
    metrics = best.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    free = metrics.get("free")
    free = free if isinstance(free, Mapping) else {}
    dyn = free.get("dyn")
    dyn = dyn if isinstance(dyn, Mapping) else {}
    return safe_float(dyn.get("wR2_all"), float("nan"))


def choose_hand_by_scout(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    active_run: Path,
    *,
    scout_max_rows: int | None = None,
) -> tuple[str, dict, Path]:
    """Compare matched original- and alternate-hand scouts.

    ``active_run`` supplies the production nuisance seed only.  It is not used
    as the original-hand score because its row count can differ from the
    bounded hand scout.  Both evidence runs below use the same model, rows,
    N/Q, nuisance grid, transform, and split seed; only ``--alternate-hand``
    differs.
    """

    production_active_summary = read_json(active_run / "summary.json")
    production_active_score = score_native_summary(production_active_summary)
    nui = best_nuisance(active_run)
    hand_root = root / "decisions" / ds.name
    original_dir = hand_root / "hand_scout_original"
    alternate_dir = hand_root / "hand_scout_alt"
    n_basis = 20 if args.fast else int(args.hand_n_basis)
    n_points = 5 if args.fast else int(args.hand_n_points)
    max_rows = int(args.scout_max_rows if scout_max_rows is None else scout_max_rows)
    comparison_contract: dict[str, object] = {
        "dataset": ds.name,
        "structure": prep.structure,
        "hkl_transform": ds.hkl_transform,
        "n_basis": n_basis,
        "n_points": n_points,
        "max_rows": max_rows,
        "split_seed": int(args.split_seed),
        "scale_fit_split": "work",
        "production_scale_preference_ignored_for_holdout": bool(args.production_scales),
        "endpoint": False,
        "pinned_nuisance": nui,
        "only_physical_difference": "alternate-hand inversion",
    }
    original_score: dict = {}
    alternate_score: dict = {}
    original_summary: dict = {}
    alternate_summary: dict = {}
    try:
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                original_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage="scout_hand_original",
                pinned=nui,
                alternate_hand=False,
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                scale_fit_split_override="work",
            ),
            logs,
            f"{ds.name}_scout_hand_original",
            dry_run=args.dry_run,
        )
        original_summary = read_json(original_dir / "summary.json")
        original_score = score_native_summary(original_summary)
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                alternate_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage="scout_hand_alternate",
                pinned=nui,
                alternate_hand=True,
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                scale_fit_split_override="work",
            ),
            logs,
            f"{ds.name}_scout_hand_alt",
            dry_run=args.dry_run,
        )
        alternate_summary = read_json(alternate_dir / "summary.json")
        alternate_score = score_native_summary(alternate_summary)
    except Exception as exc:
        payload = {
            "selected": "original",
            "selected_hand": "original",
            "scout_selected_before_quality_gate": "original",
            "quality_gate_status": "scout_failed",
            "quality_gate_reasons": [f"matched hand scout failed: {exc}"],
            "active_score": original_score,
            "matched_original_score": original_score,
            "alternate_score": alternate_score,
            "production_active_score": production_active_score,
            "score_delta_original_minus_alternate": float("-inf"),
            "score_delta_original_minus_alternate_hand": float("-inf"),
            "accept_delta": float(args.hand_accept_delta),
            "production_active_run": str(active_run),
            "original_run": str(original_dir),
            "alternate_run": str(alternate_dir),
            "comparison_contract": comparison_contract,
            "warning": f"matched hand scout failed: {exc}",
        }
        write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", payload)
        return "original", payload, alternate_dir

    original_rows = original_summary.get("row_selection")
    original_rows = dict(original_rows) if isinstance(original_rows, Mapping) else {}
    alternate_rows = alternate_summary.get("row_selection")
    alternate_rows = dict(alternate_rows) if isinstance(alternate_rows, Mapping) else {}
    row_contract_keys = ("selected_rows", "work_rows", "free_rows")
    original_row_counts = _validated_row_selection_counts(original_rows)
    alternate_row_counts = _validated_row_selection_counts(alternate_rows)
    row_selection_match = bool(
        original_row_counts is not None
        and alternate_row_counts is not None
        and original_row_counts == alternate_row_counts
    )
    original_free_rows = original_row_counts[2] if original_row_counts is not None else 0
    alternate_free_rows = alternate_row_counts[2] if alternate_row_counts is not None else 0
    positive_matched_free_rows = bool(
        original_free_rows > 0
        and alternate_free_rows > 0
        and original_free_rows == alternate_free_rows
    )
    original_raw_free_wr2 = raw_free_dyn_wr2(original_summary)
    alternate_raw_free_wr2 = raw_free_dyn_wr2(alternate_summary)
    finite_raw_free_metrics = math.isfinite(original_raw_free_wr2) and math.isfinite(alternate_raw_free_wr2)
    comparison_contract.update(
        {
            "original_row_selection": {key: original_rows.get(key) for key in row_contract_keys},
            "alternate_row_selection": {key: alternate_rows.get(key) for key in row_contract_keys},
            "original_row_selection_valid": original_row_counts is not None,
            "alternate_row_selection_valid": alternate_row_counts is not None,
            "row_selection_match": row_selection_match,
            "positive_matched_free_rows": positive_matched_free_rows,
            "original_raw_free_dyn_wR2_all": original_raw_free_wr2 if math.isfinite(original_raw_free_wr2) else None,
            "alternate_raw_free_dyn_wR2_all": alternate_raw_free_wr2 if math.isfinite(alternate_raw_free_wr2) else None,
            "finite_raw_free_metrics": finite_raw_free_metrics,
        }
    )
    delta = original_score["score"] - alternate_score["score"]
    scout_selected = "alternate" if delta > float(args.hand_accept_delta) else "original"
    selected = scout_selected
    gate_reasons: list[str] = []
    if not row_selection_match:
        gate_reasons.append("matched hand scouts did not report identical selected/work/free row counts")
    if not positive_matched_free_rows:
        gate_reasons.append("matched hand scouts did not report the same positive number of held-out free rows")
    if not finite_raw_free_metrics:
        gate_reasons.append("matched hand scouts did not report finite raw free-row dynamical wR2 metrics")
    if scout_selected == "alternate":
        max_all = float(getattr(args, "hand_max_wR2_all", 0.45))
        max_free = float(getattr(args, "hand_max_free_wR2_all", 0.50))
        min_free_gain = float(getattr(args, "hand_min_free_wR2_gain", 0.0))
        alt_all = safe_float(alternate_score.get("dyn_wR2_all"))
        alt_free = alternate_raw_free_wr2
        active_free = original_raw_free_wr2
        if not math.isfinite(alt_all) or alt_all > max_all:
            gate_reasons.append(f"alternate all-row wR2 {alt_all:.4g} exceeds hand gate {max_all:.4g}")
        if not math.isfinite(alt_free) or alt_free > max_free:
            gate_reasons.append(f"alternate free-row wR2 {alt_free:.4g} exceeds hand gate {max_free:.4g}")
        if math.isfinite(active_free) and math.isfinite(alt_free):
            free_gain = active_free - alt_free
            if free_gain < min_free_gain:
                gate_reasons.append(f"free-row wR2 gain {free_gain:.4g} is below hand gate {min_free_gain:.4g}")
        else:
            gate_reasons.append("free-row wR2 values were not finite")
        if gate_reasons:
            selected = "original"
    payload = {
        "selected": selected,
        "selected_hand": selected,
        "scout_selected_before_quality_gate": scout_selected,
        "quality_gate_status": "rejected_switch" if gate_reasons else "passed",
        "quality_gate_reasons": gate_reasons,
        "active_score": original_score,
        "matched_original_score": original_score,
        "alternate_score": alternate_score,
        "production_active_score": production_active_score,
        "score_delta_original_minus_alternate": delta,
        "score_delta_original_minus_alternate_hand": delta,
        "accept_delta": float(args.hand_accept_delta),
        "hand_max_wR2_all": float(getattr(args, "hand_max_wR2_all", 0.45)),
        "hand_max_free_wR2_all": float(getattr(args, "hand_max_free_wR2_all", 0.50)),
        "hand_min_free_wR2_gain": float(getattr(args, "hand_min_free_wR2_gain", 0.0)),
        "production_active_run": str(active_run),
        "original_run": str(original_dir),
        "alternate_run": str(alternate_dir),
        "comparison_contract": comparison_contract,
    }
    write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", payload)
    return selected, payload, alternate_dir


def aggregate_shared_hand_decisions(
    decisions: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Choose one hand for a multi-dataset model from per-dataset scouts.

    Hand is a property of the shared structure, not of an individual dataset.
    A switch is therefore allowed only when every dataset independently passes
    the existing hand-quality gates and exceeds the score margin.  This
    deliberately treats failed, weak, and mixed evidence as support for keeping
    the input hand.
    """

    accept_delta = float(args.hand_accept_delta)
    evidence: list[dict[str, object]] = []
    blocking_reasons: list[str] = []
    finite_deltas: list[float] = []

    def finite_metric(score: Mapping[str, object], key: str) -> float | None:
        value = safe_float(score.get(key), float("nan"))
        return value if math.isfinite(value) else None

    for name in sorted(decisions):
        decision = decisions[name]
        dataset_selected = str(decision.get("selected_hand") or decision.get("selected") or "original")
        gate_status = str(decision.get("quality_gate_status") or "not_passed")
        delta = safe_float(decision.get("score_delta_original_minus_alternate"), float("nan"))
        warning = str(decision.get("warning") or "")
        if math.isfinite(delta):
            finite_deltas.append(delta)

        reasons: list[str] = []
        if warning:
            reasons.append(warning)
        if not math.isfinite(delta):
            reasons.append("hand-score difference was not finite")
        elif delta <= accept_delta:
            reasons.append(f"score improvement {delta:.6g} did not exceed {accept_delta:.6g}")
        if gate_status != "passed":
            reasons.append(f"per-dataset quality gate status was {gate_status}")
        if dataset_selected != "alternate":
            reasons.append(f"per-dataset scout retained {dataset_selected}")
        supports_alternate = not reasons
        if not supports_alternate:
            blocking_reasons.append(f"{name}: {'; '.join(reasons)}")

        active_score = decision.get("active_score")
        active_score = active_score if isinstance(active_score, Mapping) else {}
        alternate_score = decision.get("alternate_score")
        alternate_score = alternate_score if isinstance(alternate_score, Mapping) else {}
        evidence.append(
            {
                "dataset": name,
                "dataset_scout_selected_hand": dataset_selected,
                "quality_gate_status": gate_status,
                "score_delta_original_minus_alternate": delta if math.isfinite(delta) else None,
                "active_dyn_wR2_all": finite_metric(active_score, "dyn_wR2_all"),
                "active_free_dyn_wR2_all": finite_metric(active_score, "free_dyn_wR2_all"),
                "alternate_dyn_wR2_all": finite_metric(alternate_score, "dyn_wR2_all"),
                "alternate_free_dyn_wR2_all": finite_metric(alternate_score, "free_dyn_wR2_all"),
                "supports_shared_alternate": supports_alternate,
                "blocking_reasons": reasons,
            }
        )

    aggregate_delta = sum(finite_deltas) / len(finite_deltas) if finite_deltas else float("nan")
    all_support_alternate = bool(evidence) and not blocking_reasons and len(finite_deltas) == len(evidence)
    if all_support_alternate and aggregate_delta <= accept_delta:
        blocking_reasons.append(
            f"mean score improvement {aggregate_delta:.6g} did not exceed shared threshold {accept_delta:.6g}"
        )
        all_support_alternate = False
    selected = "alternate" if all_support_alternate else "original"
    return {
        "status": "selected_alternate" if selected == "alternate" else "retained_original",
        "selected": selected,
        "selected_hand": selected,
        "accept_delta": accept_delta,
        "dataset_count": len(evidence),
        "alternate_support_count": sum(bool(row["supports_shared_alternate"]) for row in evidence),
        "aggregate_score_delta_original_minus_alternate": aggregate_delta if math.isfinite(aggregate_delta) else None,
        "quality_gate_status": "passed" if selected == "alternate" else "retained_original",
        "quality_gate_reasons": blocking_reasons,
        "dataset_evidence": evidence,
        "rule": (
            "switch the one shared model only when every dataset has finite alternate-hand evidence, "
            "passes the per-dataset all/free-row quality gates, and exceeds the score margin; "
            "failed, weak, or mixed evidence retains the input hand"
        ),
    }


def choose_shared_hand_by_scout(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    active_runs: Mapping[str, Path],
    scout_rows_by_name: Mapping[str, int],
) -> tuple[str, dict[str, object], dict[str, Path]]:
    """Run per-dataset hand scouts and persist one shared model decision."""

    shared_path = root / "decisions" / "shared_absolute_structure_decision.json"
    policy = absolute_structure_policy(prep.structure)
    skipped_status = ""
    skipped_reason = ""
    if not getattr(args, "auto_decide", True):
        skipped_status = "skipped_auto_decide_disabled"
        skipped_reason = "automatic decisions were disabled"
    elif not getattr(args, "hand_control", True):
        skipped_status = "skipped_hand_control_disabled"
        skipped_reason = "alternate-hand control was disabled"
    elif not policy.get("run_hand_control", True):
        skipped_status = "skipped_not_applicable"
        skipped_reason = str(policy.get("reason") or "alternate-hand control is not applicable")
    elif getattr(args, "dry_run", False):
        skipped_status = "skipped_dry_run"
        skipped_reason = "dry-run mode does not execute hand scouts"

    if skipped_status:
        shared: dict[str, object] = {
            "status": skipped_status,
            "selected": "original",
            "selected_hand": "original",
            "quality_gate_status": "not_run",
            "quality_gate_reasons": [skipped_reason],
            "dataset_count": len(prep.datasets),
            "dataset_evidence": [],
            "symmetry_policy": policy,
            "selected_model": prep.structure,
            "rule": "retain the input hand when the shared multi-dataset hand gate is not run",
        }
        write_json(shared_path, shared)
        for ds in prep.datasets:
            dataset_payload = {
                **policy,
                "status": skipped_status,
                "selected": "original",
                "selected_hand": "original",
                "dataset_scout_selected_hand": "not_run",
                "shared_selected_hand": "original",
                "shared_decision": str(shared_path),
                "reason": skipped_reason,
                "selection_scope": "shared_multi_dataset_model",
            }
            write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", dataset_payload)
        return "original", shared, {}

    per_dataset: dict[str, dict[str, object]] = {}
    alternate_runs: dict[str, Path] = {}
    for ds in prep.datasets:
        _selected, decision, alternate_run = choose_hand_by_scout(
            prep,
            ds,
            args,
            python,
            logs,
            root,
            active_runs[ds.name],
            scout_max_rows=scout_rows_by_name.get(ds.name, int(getattr(args, "scout_max_rows", 1000))),
        )
        per_dataset[ds.name] = {
            **decision,
            "active_run": str(active_runs[ds.name]),
            "scout_max_rows": int(scout_rows_by_name.get(ds.name, int(getattr(args, "scout_max_rows", 1000)))),
        }
        alternate_runs[ds.name] = alternate_run

    shared = aggregate_shared_hand_decisions(per_dataset, args)
    shared["symmetry_policy"] = policy
    shared["per_dataset_decisions"] = {
        ds.name: str(root / "decisions" / ds.name / "absolute_structure_decision.json") for ds in prep.datasets
    }
    shared["active_runs"] = {ds.name: str(active_runs[ds.name]) for ds in prep.datasets}
    shared["alternate_runs"] = {ds.name: str(alternate_runs[ds.name]) for ds in prep.datasets}
    shared["scout_max_rows_by_dataset"] = {
        ds.name: int(scout_rows_by_name.get(ds.name, int(getattr(args, "scout_max_rows", 1000))))
        for ds in prep.datasets
    }
    if shared.get("selected_hand") == "alternate" and not is_shelx_model(prep.structure):
        shared["status"] = "retained_original_unmaterializable_model"
        shared["selected"] = "original"
        shared["selected_hand"] = "original"
        shared["quality_gate_status"] = "retained_original"
        reasons = list(shared.get("quality_gate_reasons") or [])
        reasons.append("the alternate shared model can currently be materialized only from a SHELX .res/.ins input")
        shared["quality_gate_reasons"] = reasons
    shared["selected_model"] = prep.structure
    write_json(shared_path, shared)

    shared_selected = str(shared["selected_hand"])
    for ds in prep.datasets:
        decision = dict(per_dataset[ds.name])
        dataset_selected = str(decision.get("selected_hand") or decision.get("selected") or "original")
        decision.update(
            {
                "selected": shared_selected,
                "selected_hand": shared_selected,
                "dataset_scout_selected_hand": dataset_selected,
                "shared_selected_hand": shared_selected,
                "shared_quality_gate_status": shared.get("quality_gate_status"),
                "shared_quality_gate_reasons": shared.get("quality_gate_reasons", []),
                "shared_decision": str(shared_path),
                "selection_scope": "shared_multi_dataset_model",
            }
        )
        write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", decision)
    return shared_selected, shared, alternate_runs


def hand_diagnostic_supports_other(stats: Mapping[str, object], args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Return whether row-level paired t-style diagnostics support ``other``.

    ``absolute_structure_diagnostic.py`` defines positive paired-row t statistics as
    support for the current best/final row table.  A post-feedback promotion
    switches to the opposite-hand row table, so the required sign is negative.
    This statistic is not a calibrated normal Z score.
    """

    min_all = float(getattr(args, "hand_min_z_all", 2.0))
    min_free = float(getattr(args, "hand_min_z_free", 0.5))
    min_free_rows = max(0, int(getattr(args, "hand_min_free_rows", 10) or 0))
    reasons: list[str] = []
    all_block = stats.get("all") if isinstance(stats, Mapping) else {}
    free_block = stats.get("free") if isinstance(stats, Mapping) else {}
    all_t = safe_float((all_block or {}).get("paired_row_t_delta_chi2") if isinstance(all_block, Mapping) else float("nan"))
    free_t = safe_float((free_block or {}).get("paired_row_t_delta_chi2") if isinstance(free_block, Mapping) else float("nan"))
    try:
        free_n = int((free_block or {}).get("n", 0) if isinstance(free_block, Mapping) else 0)
    except (TypeError, ValueError, OverflowError):
        free_n = 0
    if not math.isfinite(all_t) or all_t > -min_all:
        reasons.append(
            f"all-row paired t-style statistic {all_t:.4g} does not support switching to other "
            f"at threshold {min_all:.4g}"
        )
    if min_free > 0.0 and (not math.isfinite(free_t) or free_t > -min_free):
        reasons.append(
            f"free-row paired t-style statistic {free_t:.4g} does not support switching to other "
            f"at threshold {min_free:.4g}"
        )
    if free_n < min_free_rows:
        reasons.append(
            f"free-row paired t-style statistic has n={free_n}, below the minimum {min_free_rows} "
            "required for material opposite-hand support"
        )
    return not reasons, reasons


def _positive_summary_int(config: Mapping[str, object], key: str) -> int | None:
    """Return one required positive integer from a native-stage config."""

    try:
        value = int(config.get(key, 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def _args_with_recorded_native_contract(
    args: argparse.Namespace,
    config: Mapping[str, object],
) -> argparse.Namespace:
    """Clone CLI arguments with physics-bearing native settings from a summary."""

    cloned = argparse.Namespace(**vars(args))
    direct_keys = (
        "sg_window_min",
        "sg_window_max",
        "sg_window_scale",
        "sg_window_frame_floor",
        "profile_quadrature",
        "profile_accumulation",
        "topup_d_min",
        "max_support_sg",
        "free_fraction",
        "split_seed",
        "selection_metric",
        "scale_smoothness",
        "prediction_cache_compression",
        "compute_backend",
        "row_plan_cache",
        "jax_batch_size",
        "jax_matrix_bucket_size",
        "bethe_strong_count",
        "bethe_variant",
        "bethe_promote_denom",
        "bethe_denom_eps",
        "matrix_convention",
    )
    for key in direct_keys:
        if key in config and config.get(key) is not None:
            setattr(cloned, key, config.get(key))
    if config.get("thickness_points") is not None:
        cloned.final_thickness_points = int(config["thickness_points"])
    return cloned


def _native_split_dyn_wr2(summary: Mapping[str, object], split: str) -> float:
    """Return one split's native dynamical wR2 without applying a fallback."""

    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    metrics = best.get("metrics") if isinstance(best.get("metrics"), Mapping) else {}
    split_metrics = metrics.get(split) if isinstance(metrics.get(split), Mapping) else {}
    dyn = split_metrics.get("dyn") if isinstance(split_metrics.get("dyn"), Mapping) else {}
    return safe_float(dyn.get("wR2_all"), float("nan"))


def _validated_row_selection_counts(rows: Mapping[str, object]) -> tuple[int, int, int] | None:
    """Return an exact selected/work/free count contract, or ``None``.

    Hand comparisons must not treat two absent or equally malformed values as
    matching evidence. These mappings come from JSON summaries, so requiring
    actual non-boolean integers is both unambiguous and intentionally strict.
    """

    keys = ("selected_rows", "work_rows", "free_rows")
    values: list[int] = []
    for key in keys:
        if key not in rows:
            return None
        value = rows.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        values.append(value)
    selected_rows, work_rows, free_rows = values
    if selected_rows != work_rows + free_rows:
        return None
    return selected_rows, work_rows, free_rows


def _pinned_nuisance_mismatches(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> list[str]:
    """Return material differences in a recomputed one-point nuisance contract."""

    numeric_keys = (
        "thickness_A",
        "thickness_grad_frac",
        "thickness_sigma_frac",
        "frame_offset",
        "group_width_deg",
        "detector_scale_bins",
        "sigma_a",
        "sigma_b_corr",
        "sigma_c_peak",
        "row_resolution_b",
        "scale_smoothness",
        "sg_jitter_sigma0",
        "sg_jitter_sigma_d_inv",
        "sg_jitter_sigma_frame",
        "sg_jitter_points",
        "sg_jitter_student_df",
        "sg_jitter_width_factor",
        "absorption_mu",
        "absorption_frac",
        "absorption_mu_g2",
        "absorption_mu_theta2",
        "absorption_tds_b",
        "dose_b_per_frame",
        "dose_b_ref_frame",
        "native_wght_a",
        "native_wght_b",
    )
    categorical_keys = (
        "thickness_model",
        "sg_jitter_mode",
        "absorption_mode",
        "dose_b_mode",
        "matrix_convention",
    )
    mismatches: list[str] = []
    for key in numeric_keys:
        if key not in expected:
            continue
        if key not in actual or not numeric_nuisance_contract_matches(expected.get(key), actual.get(key)):
            mismatches.append(f"{key}: selected={expected.get(key)!r}, opposite={actual.get(key)!r}")
    for key in categorical_keys:
        if key in expected and str(actual.get(key)) != str(expected.get(key)):
            mismatches.append(f"{key}: selected={expected.get(key)!r}, opposite={actual.get(key)!r}")
    return mismatches


def _summary_model_binding(
    summary: Mapping[str, object],
    expected_model: Path,
    *,
    semantic_hand: str,
) -> dict[str, object]:
    """Bind one native summary to exact model bytes and explicit hand semantics."""

    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    recorded_path = str(inputs.get("res") or "")
    recorded_sha = str(inputs.get("res_sha256") or "")
    model_tag = str(inputs.get("model_tag") or "")
    inversion_key = "alternate_hand" if "alternate_hand" in config else "invert_hand" if "invert_hand" in config else ""
    engine_inversion = config.get(inversion_key) if inversion_key else None
    errors: list[str] = []
    expected_path = expected_model.expanduser().resolve()
    expected_sha = ""
    if not expected_path.is_file():
        errors.append(f"expected model is missing: {expected_path}")
    else:
        try:
            expected_sha = sha256_file(expected_path)
        except OSError as exc:
            errors.append(f"cannot hash expected model {expected_path}: {exc}")
    if not recorded_path:
        errors.append("summary inputs.res is missing")
    else:
        try:
            resolved_recorded = Path(recorded_path).expanduser().resolve()
        except OSError:
            resolved_recorded = Path(recorded_path).expanduser()
        if resolved_recorded != expected_path:
            errors.append(
                f"summary inputs.res does not match expected model: {resolved_recorded} != {expected_path}"
            )
    if not recorded_sha:
        errors.append("summary inputs.res_sha256 is missing")
    elif expected_sha and recorded_sha != expected_sha:
        errors.append(
            f"summary inputs.res_sha256 does not match expected model bytes: {recorded_sha} != {expected_sha}"
        )
    if semantic_hand not in {"original", "alternate"}:
        errors.append(f"unknown semantic hand label {semantic_hand!r}")
    if not model_tag:
        errors.append("summary inputs.model_tag is missing")
    elif model_tag.split(";", 1)[0].strip().lower() != "original":
        errors.append(
            f"summary model_tag {model_tag!r} indicates an additional engine-side hand transformation"
        )
    if not inversion_key:
        errors.append("summary config does not record alternate_hand/invert_hand")
    elif bool(engine_inversion):
        errors.append(
            f"summary config {inversion_key}={engine_inversion!r} applies an unexpected second hand inversion"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "semantic_hand": semantic_hand,
        "expected_model": str(expected_path),
        "expected_model_sha256": expected_sha,
        "summary_inputs_res": recorded_path,
        "summary_inputs_res_sha256": recorded_sha,
        "summary_model_tag": model_tag,
        "summary_inversion_key": inversion_key,
        "summary_engine_inversion": engine_inversion,
        "semantic_rule": (
            "the named model bytes already embody this semantic hand; native recomputation must use model_tag "
            "original with no additional engine-side inversion"
        ),
        "errors": errors,
    }


def _summary_hkl_binding(summary: Mapping[str, object], expected_hkl: Path) -> dict[str, object]:
    """Bind one native summary to the exact dataset observation-file bytes."""

    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    recorded_path = str(inputs.get("hkl") or "")
    recorded_sha = str(inputs.get("hkl_sha256") or "")
    expected_path = expected_hkl.expanduser().resolve()
    expected_sha = ""
    errors: list[str] = []
    if not expected_path.is_file():
        errors.append(f"expected dataset HKL is missing: {expected_path}")
    else:
        try:
            expected_sha = sha256_file(expected_path)
        except OSError as exc:
            errors.append(f"cannot hash expected dataset HKL {expected_path}: {exc}")
    if not recorded_path:
        errors.append("summary inputs.hkl is missing")
    else:
        try:
            resolved_recorded = Path(recorded_path).expanduser().resolve()
        except OSError:
            resolved_recorded = Path(recorded_path).expanduser()
        if resolved_recorded != expected_path:
            errors.append(
                f"summary inputs.hkl does not match DatasetSpec.hkl: {resolved_recorded} != {expected_path}"
            )
    if not recorded_sha:
        errors.append("summary inputs.hkl_sha256 is missing")
    elif expected_sha and recorded_sha != expected_sha:
        errors.append(
            f"summary inputs.hkl_sha256 does not match DatasetSpec.hkl bytes: {recorded_sha} != {expected_sha}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "expected_hkl": str(expected_path),
        "expected_hkl_sha256": expected_sha,
        "summary_inputs_hkl": recorded_path,
        "summary_inputs_hkl_sha256": recorded_sha,
        "errors": errors,
    }


def _summary_native_row_table_binding(
    summary: Mapping[str, object],
    expected_table: Path,
) -> dict[str, object]:
    """Bind a source final row table to the producer-recorded path and SHA."""

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    recorded_path = str(artifacts.get("native_row_table") or "")
    recorded_sha = str(artifacts.get("native_row_table_sha256") or "")
    expected_path = expected_table.expanduser().resolve()
    expected_sha = ""
    errors: list[str] = []
    if not expected_path.is_file():
        errors.append(f"source final native row table is missing: {expected_path}")
    else:
        try:
            expected_sha = sha256_file(expected_path)
        except OSError as exc:
            errors.append(f"cannot hash source final native row table {expected_path}: {exc}")
    if not recorded_path:
        errors.append("summary artifacts.native_row_table is missing")
    else:
        try:
            resolved_recorded = Path(recorded_path).expanduser().resolve()
        except OSError:
            resolved_recorded = Path(recorded_path).expanduser()
        if resolved_recorded != expected_path:
            errors.append(
                f"summary native row-table path does not match source final table: {resolved_recorded} != {expected_path}"
            )
    if not recorded_sha:
        errors.append("summary artifacts.native_row_table_sha256 is missing")
    elif expected_sha and recorded_sha != expected_sha:
        errors.append(
            "summary native_row_table_sha256 does not match current source final row-table bytes: "
            f"{recorded_sha} != {expected_sha}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "expected_table": str(expected_path),
        "expected_table_sha256": expected_sha,
        "summary_native_row_table": recorded_path,
        "summary_native_row_table_sha256": recorded_sha,
        "errors": errors,
    }


def _policy_selected_table_binding(
    source_path: Path,
    selected_path: Path,
) -> dict[str, object]:
    """Verify a policy-annotated selected table preserves every source data cell."""

    policy_columns = {
        "dataset",
        "dataset_role",
        "raw_row_weight",
        "endpoint_weight",
        "sigma_eff_original",
        "policy_sigma_weight_applied",
    }

    key_fields = ("row_index", "h", "k", "l")

    def load(path: Path) -> tuple[list[str], dict[tuple[str, ...], dict[str, str]], int, list[str]]:
        errors: list[str] = []
        fields: list[str] = []
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        duplicates = 0
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                missing = [field for field in key_fields if field not in fields]
                if missing:
                    return fields, rows, duplicates, [f"{path} is missing key columns: {', '.join(missing)}"]
                for row in reader:
                    key = tuple(str(row.get(field, "")).strip() for field in key_fields)
                    if key in rows:
                        duplicates += 1
                    else:
                        rows[key] = {field: str(row.get(field, "")) for field in fields}
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
        return fields, rows, duplicates, errors

    source_fields, source_rows, source_duplicates, source_errors = load(source_path)
    selected_fields, selected_rows, selected_duplicates, selected_errors = load(selected_path)
    source_contract_fields = [field for field in source_fields if field not in policy_columns]
    selected_contract_fields = [field for field in selected_fields if field not in policy_columns]
    errors = [*source_errors, *selected_errors]
    if set(source_contract_fields) != set(selected_contract_fields):
        errors.append(
            "policy-selected table non-policy columns differ from source final table: "
            f"source_only={sorted(set(source_contract_fields) - set(selected_contract_fields))}, "
            f"selected_only={sorted(set(selected_contract_fields) - set(source_contract_fields))}"
        )
    if source_duplicates or selected_duplicates:
        errors.append(
            f"policy/source row keys are ambiguous (source={source_duplicates}, selected={selected_duplicates})"
        )
    joined = set(source_rows) & set(selected_rows)
    if len(joined) != len(source_rows) or len(joined) != len(selected_rows):
        errors.append(
            f"policy/source row join is not exact (source={len(source_rows)}, "
            f"selected={len(selected_rows)}, joined={len(joined)})"
        )
    mismatch_counts = {field: 0 for field in source_contract_fields}
    examples: list[dict[str, object]] = []
    comparable_fields = [field for field in source_contract_fields if field in selected_contract_fields]
    for key in sorted(joined):
        for field in comparable_fields:
            if source_rows[key].get(field, "") != selected_rows[key].get(field, ""):
                mismatch_counts[field] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "source": source_rows[key].get(field, ""),
                            "selected": selected_rows[key].get(field, ""),
                        }
                    )
    changed = {field: count for field, count in mismatch_counts.items() if count}
    if changed:
        errors.append(f"policy-selected table changed non-policy source cells: {changed}")
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "source_table": str(source_path.expanduser().resolve()),
        "selected_table": str(selected_path.expanduser().resolve()),
        "source_table_sha256": sha256_file(source_path) if source_path.is_file() else "",
        "selected_table_sha256": sha256_file(selected_path) if selected_path.is_file() else "",
        "policy_columns_allowed_to_differ": sorted(policy_columns),
        "non_policy_columns_compared": source_contract_fields,
        "source_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "joined_rows": len(joined),
        "duplicate_keys_source": source_duplicates,
        "duplicate_keys_selected": selected_duplicates,
        "non_policy_mismatch_counts": changed,
        "mismatch_examples": examples,
        "errors": errors,
    }


def _native_physics_config_contract(
    selected_summary: Mapping[str, object],
    opposite_summary: Mapping[str, object],
) -> dict[str, object]:
    """Verify the two hand branches used the same physics-bearing stage config."""

    selected = selected_summary.get("config") if isinstance(selected_summary.get("config"), Mapping) else {}
    opposite = opposite_summary.get("config") if isinstance(opposite_summary.get("config"), Mapping) else {}
    exact_fields = (
        "n_basis",
        "n_points",
        "thickness_points",
        "sg_window_mode",
        "sg_jitter_mode_grid",
        "profile_quadrature",
        "profile_accumulation",
        "split_seed",
        "scale_fit_split",
        "target_mode",
        "positive_only",
        "hkl_transform",
        "compute_backend",
        "bethe_strong_count",
        "bethe_variant",
        "matrix_convention",
    )
    numeric_fields = (
        "sg_window",
        "sg_window_min",
        "sg_window_max",
        "sg_window_scale",
        "sg_window_frame_floor",
        "topup_d_min",
        "max_support_sg",
        "free_fraction",
        "scale_smoothness",
        "min_d_spacing",
        "min_peak",
        "geometry_percentile",
        "bethe_promote_denom",
        "bethe_denom_eps",
    )
    errors: list[str] = []
    exact_mismatches: dict[str, dict[str, object]] = {}
    numeric_mismatches: dict[str, dict[str, object]] = {}
    for field in exact_fields:
        if field not in selected or field not in opposite:
            errors.append(f"physics config field {field} is missing from one or both hand branches")
            continue
        if selected.get(field) != opposite.get(field):
            exact_mismatches[field] = {
                "selected": selected.get(field),
                "opposite": opposite.get(field),
            }
    for field in numeric_fields:
        selected_value = safe_float(selected.get(field), float("nan"))
        opposite_value = safe_float(opposite.get(field), float("nan"))
        if not math.isfinite(selected_value) or not math.isfinite(opposite_value):
            errors.append(f"physics config field {field} is missing or nonfinite in one or both hand branches")
            continue
        if not math.isclose(selected_value, opposite_value, rel_tol=1e-12, abs_tol=1e-12):
            numeric_mismatches[field] = {
                "selected": selected_value,
                "opposite": opposite_value,
                "absolute_delta": abs(opposite_value - selected_value),
            }
    if exact_mismatches:
        errors.append(f"exact physics config differs between hand branches: {exact_mismatches}")
    if numeric_mismatches:
        errors.append(f"numeric physics config differs between hand branches: {numeric_mismatches}")
    selected_transform = (
        (selected_summary.get("inputs") or {}).get("hkl_transform")
        if isinstance(selected_summary.get("inputs"), Mapping)
        else None
    )
    opposite_transform = (
        (opposite_summary.get("inputs") or {}).get("hkl_transform")
        if isinstance(opposite_summary.get("inputs"), Mapping)
        else None
    )
    if not selected_transform or not opposite_transform:
        errors.append("summary inputs.hkl_transform is missing from one or both hand branches")
    elif str(selected_transform) != str(opposite_transform):
        errors.append(
            f"resolved HKL transform differs between hand branches: {selected_transform!r} != {opposite_transform!r}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "exact_fields_checked": list(exact_fields),
        "numeric_fields_checked": list(numeric_fields),
        "selected_resolved_hkl_transform": selected_transform,
        "opposite_resolved_hkl_transform": opposite_transform,
        "exact_mismatches": exact_mismatches,
        "numeric_mismatches": numeric_mismatches,
        "errors": errors,
    }


def _row_contract_nullable_numeric_fields(
    config: Mapping[str, object],
    nuisance: Mapping[str, object],
) -> set[str]:
    """Return row fields that are physically not applicable for this run."""

    nullable: set[str] = set()
    sg_window_mode = str(config.get("sg_window_mode", "") or "").strip().lower()
    if sg_window_mode in {"fixed", "rlp_peak"}:
        nullable.add("sg_window_half_frames")
    sg_jitter_mode = str(
        nuisance.get("sg_jitter_mode", config.get("sg_jitter_mode", "")) or ""
    ).strip().lower()
    if sg_jitter_mode == "none":
        nullable.update({"sg_jitter_d_inv_A", "sg_jitter_frame_fraction"})
    return nullable


def _paired_row_split_contract(
    best_path: Path,
    other_path: Path,
    *,
    nullable_numeric_fields: set[str] | None = None,
) -> dict[str, object]:
    """Verify identical selected/opposite observation and row-selection contracts."""

    key_fields = ("row_index", "h", "k", "l")
    exact_fields = (
        "split",
        "group",
        "detector_bin",
        "basis_size",
        "sg_window_clipped",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
    )
    numeric_fields = (
        "target_I",
        "sigma_eff",
        "xds_sigma",
        "d_A",
        "peak",
        "corr",
        "rlp",
        "row_resolution_factor",
        "center_sg_inv_A",
        "omega_bragg_deg",
        "omega_bragg_minus_center_deg",
        "sg_window_inv_A",
        "sg_window_raw_inv_A",
        "sg_window_half_frames",
        "sg_dSg_domega_inv_A_per_deg",
        "sg_dSg_dframe_inv_A",
        "sg_jitter_sigma_inv_A",
        "sg_jitter_d_inv_A",
        "sg_jitter_frame_fraction",
        "sg_profile_weight_sum",
        "max_sg_solve_residual",
    )
    nullable_numeric_fields = set(nullable_numeric_fields or ())
    unknown_nullable_fields = nullable_numeric_fields - set(numeric_fields)
    if unknown_nullable_fields:
        raise ValueError(
            "unknown nullable row-contract fields: " + ", ".join(sorted(unknown_nullable_fields))
        )
    relative_tolerance = 1e-12
    absolute_tolerance = 1e-12

    def indexed(path: Path) -> tuple[dict[tuple[str, ...], dict[str, str]], int, list[str]]:
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        duplicates = 0
        errors: list[str] = []
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                missing = [
                    field
                    for field in (*key_fields, *exact_fields, *numeric_fields)
                    if field not in fields
                ]
                if missing:
                    return {}, 0, [f"{path} is missing columns: {', '.join(missing)}"]
                for row in reader:
                    key = tuple(str(row.get(field, "")).strip() for field in key_fields)
                    if key in rows:
                        duplicates += 1
                    else:
                        rows[key] = {field: str(row.get(field, "")).strip() for field in (*exact_fields, *numeric_fields)}
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
        return rows, duplicates, errors

    best, duplicates_best, errors_best = indexed(best_path)
    other, duplicates_other, errors_other = indexed(other_path)
    joined = set(best) & set(other)
    exact_mismatch_counts = {field: 0 for field in exact_fields}
    numeric_mismatch_counts = {field: 0 for field in numeric_fields}
    nonfinite_counts = {field: 0 for field in numeric_fields}
    matched_nullable_nan_counts = {field: 0 for field in nullable_numeric_fields}
    mismatch_examples: list[dict[str, object]] = []

    def parsed_numeric_token(raw: str) -> tuple[str, float]:
        token = str(raw).strip()
        if token.lower() == "nan":
            return "nan", float("nan")
        try:
            value = float(token)
        except (TypeError, ValueError):
            return "invalid", float("nan")
        return ("finite", value) if math.isfinite(value) else ("invalid", value)

    for key in sorted(joined):
        for field in exact_fields:
            if best[key][field] != other[key][field]:
                exact_mismatch_counts[field] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "selected": best[key][field],
                            "opposite": other[key][field],
                        }
                    )
        for field in numeric_fields:
            selected_kind, selected_value = parsed_numeric_token(best[key][field])
            opposite_kind, opposite_value = parsed_numeric_token(other[key][field])
            if selected_kind != "finite" or opposite_kind != "finite":
                if (
                    field in nullable_numeric_fields
                    and selected_kind == "nan"
                    and opposite_kind == "nan"
                ):
                    matched_nullable_nan_counts[field] += 1
                    continue
                nonfinite_counts[field] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "selected": best[key][field],
                            "opposite": other[key][field],
                            "reason": (
                                "nan_not_allowed"
                                if selected_kind == "nan" and opposite_kind == "nan"
                                else "invalid_or_nonfinite_numeric_token"
                            ),
                        }
                    )
            elif not math.isclose(
                selected_value,
                opposite_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                numeric_mismatch_counts[field] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "key": list(key),
                            "column": field,
                            "selected": selected_value,
                            "opposite": opposite_value,
                            "absolute_delta": abs(opposite_value - selected_value),
                        }
                    )
    errors = [*errors_best, *errors_other]
    if duplicates_best or duplicates_other:
        errors.append(
            f"duplicate row keys are ambiguous (selected={duplicates_best}, opposite={duplicates_other})"
        )
    if len(joined) != len(best) or len(joined) != len(other):
        errors.append(
            f"row-key join is not exact (selected={len(best)}, opposite={len(other)}, joined={len(joined)})"
        )
    for field, count in exact_mismatch_counts.items():
        if count:
            errors.append(f"{count} joined rows change exact contract column {field}")
    for field, count in nonfinite_counts.items():
        if count:
            errors.append(f"{count} joined rows have nonfinite selected/opposite values for {field}")
    for field, count in numeric_mismatch_counts.items():
        if count:
            errors.append(
                f"{count} joined rows change numeric contract column {field} beyond "
                f"rtol={relative_tolerance:g}, atol={absolute_tolerance:g}"
            )
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "key": list(key_fields),
        "selected_rows": len(best),
        "opposite_rows": len(other),
        "joined_rows": len(joined),
        "duplicate_keys_selected": duplicates_best,
        "duplicate_keys_opposite": duplicates_other,
        "exact_columns_checked": list(exact_fields),
        "numeric_columns_checked": list(numeric_fields),
        "nullable_numeric_columns": sorted(nullable_numeric_fields),
        "matched_nullable_nan_counts": matched_nullable_nan_counts,
        "ignored_policy_columns": ["raw_row_weight", "endpoint_weight", "sigma_eff_original"],
        "ignored_policy_reason": (
            "policy annotation columns are not consumed by absolute_structure_diagnostic; sigma_eff itself must "
            "still match exactly within tolerance"
        ),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "split_mismatch_count": exact_mismatch_counts["split"],
        "exact_mismatch_counts": exact_mismatch_counts,
        "numeric_mismatch_counts": numeric_mismatch_counts,
        "nonfinite_counts": nonfinite_counts,
        "mismatch_examples": mismatch_examples,
        "errors": errors,
    }


def run_multi_post_feedback_hand_diagnostic(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    final_model: str,
    selected_hand: str,
    final_runs: Mapping[str, Path],
    publication_runs: Mapping[str, Path],
    shared_hand_decision: Mapping[str, object],
) -> dict[str, object]:
    """Audit the frozen shared hand independently on each publication crystal.

    This is deliberately diagnostic-only.  It inverts the final shared model,
    recomputes every publication-eligible crystal with that crystal's exact
    recorded final nuisance and N/Q contract, and compares matched row tables.
    The initial shared hand gate remains authoritative; this function can only
    pass the consistency audit or emit a fail-closed warning.
    """

    artifact = root / "decisions" / "shared_post_feedback_hand_diagnostic.json"
    opposite_hand = "alternate" if str(selected_hand) == "original" else "original"
    eligible = [ds.name for ds in prep.datasets if ds.name in publication_runs]
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "not_run",
        "selection_scope": "shared_multi_dataset_model",
        "selected_hand_authoritative": str(selected_hand),
        "opposite_hand_diagnostic": opposite_hand,
        "selected_model": str(final_model),
        "shared_hand_decision": str(root / "decisions" / "shared_absolute_structure_decision.json"),
        "artifact": str(artifact),
        "publication_eligible_datasets": eligible,
        "selected_row_table_policy": (
            "use each policy-active publication_runs/native_dyn_calc.csv table exactly; policy annotation columns are "
            "retained, while the paired residual diagnostic uses dyn_residual_over_sigma and ignores raw_row_weight"
        ),
        "datasets": {},
        "merged_t_style_statistic": None,
        "merged_t_style_policy": (
            "not computed: crystals retain independent row, sigma, scale, and free-set contracts"
        ),
        "merged_z": None,
        "merged_z_policy": (
            "compatibility alias only; the paired Student-t-style contrast is not a calibrated normal Z score, "
            "and no pooled statistic is computed"
        ),
        "decision_rule": (
            "diagnostic only; retain the initial shared hand after final refinement and warn if a "
            "per-dataset diagnostic is incomplete or materially favors the opposite hand"
        ),
        "hand_switched": False,
    }
    symmetry_policy = (
        shared_hand_decision.get("symmetry_policy")
        if isinstance(shared_hand_decision.get("symmetry_policy"), Mapping)
        else {}
    )
    if not bool(getattr(args, "hand_control", True)):
        payload.update(
            {
                "status": "skipped_disabled",
                "consistency_gate": {
                    "status": "not_applicable",
                    "passed": None,
                    "warnings": ["absolute-structure hand control was disabled"],
                },
            }
        )
        write_json(artifact, payload)
        return payload
    if symmetry_policy and not bool(symmetry_policy.get("run_hand_control", True)):
        payload.update(
            {
                "status": "skipped_not_applicable",
                "consistency_gate": {
                    "status": "not_applicable",
                    "passed": None,
                    "warnings": [str(symmetry_policy.get("reason") or "hand control is not applicable")],
                },
            }
        )
        write_json(artifact, payload)
        return payload
    if bool(getattr(args, "dry_run", False)):
        payload.update(
            {
                "status": "skipped_dry_run",
                "consistency_gate": {
                    "status": "not_run",
                    "passed": None,
                    "warnings": ["dry-run mode does not materialize final row diagnostics"],
                },
            }
        )
        write_json(artifact, payload)
        return payload

    datasets_by_name = {ds.name: ds for ds in prep.datasets}
    records: dict[str, dict[str, object]] = {}
    runnable: dict[str, dict[str, object]] = {}
    for name in eligible:
        ds = datasets_by_name[name]
        selected_run = Path(publication_runs[name])
        source_run = Path(final_runs.get(name, selected_run))
        selected_table = selected_run / "native_dyn_calc.csv"
        source_table = source_run / "native_dyn_calc.csv"
        source_summary_path = source_run / "summary.json"
        summary = try_read_json(source_summary_path)
        config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
        nuisance = (
            (summary.get("best_native") or {}).get("nuisance")
            if isinstance(summary.get("best_native"), Mapping)
            else {}
        )
        nuisance = dict(nuisance) if isinstance(nuisance, Mapping) else {}
        n_basis = _positive_summary_int(config, "n_basis")
        n_points = _positive_summary_int(config, "n_points")
        scale_fit_split = str(config.get("scale_fit_split") or "")
        selected_model_binding = _summary_model_binding(
            summary,
            Path(final_model),
            semantic_hand=str(selected_hand),
        )
        selected_hkl_binding = _summary_hkl_binding(summary, Path(ds.hkl))
        source_row_table_binding = _summary_native_row_table_binding(summary, source_table)
        policy_selected_table_binding = _policy_selected_table_binding(source_table, selected_table)
        missing: list[str] = []
        if not selected_table.is_file():
            missing.append(f"selected row table is missing: {selected_table}")
        if not source_summary_path.is_file():
            missing.append(f"source summary is missing: {source_summary_path}")
        if n_basis is None:
            missing.append("final summary does not record a positive n_basis")
        if n_points is None:
            missing.append("final summary does not record a positive n_points")
        if not nuisance:
            missing.append("final summary does not record the selected nuisance contract")
        if scale_fit_split not in {"work", "all"}:
            missing.append("final summary does not record a valid scale_fit_split")
        missing.extend(
            f"selected-model provenance: {error}"
            for error in selected_model_binding.get("errors", [])
        )
        missing.extend(
            f"selected-HKL provenance: {error}"
            for error in selected_hkl_binding.get("errors", [])
        )
        missing.extend(
            f"source-row provenance: {error}"
            for error in source_row_table_binding.get("errors", [])
        )
        missing.extend(
            f"policy-selected row table: {error}"
            for error in policy_selected_table_binding.get("errors", [])
        )
        record: dict[str, object] = {
            "dataset": name,
            "status": "warning_incomplete" if missing else "ready",
            "selected_hand": str(selected_hand),
            "opposite_hand": opposite_hand,
            "selected_run": str(selected_run),
            "selected_source_run": str(source_run),
            "selected_source_row_table": str(source_table),
            "selected_row_table": str(selected_table),
            "selected_summary": str(source_summary_path),
            "comparison_contract": {
                "n_basis": n_basis,
                "n_points": n_points,
                "nuisance": nuisance,
                "scale_fit_split": scale_fit_split or None,
                "sg_window": config.get("sg_window"),
                "sg_window_mode": config.get("sg_window_mode"),
                "max_rows": config.get("max_rows"),
                "same_final_shared_model": bool(selected_model_binding.get("passed", False)),
                "selected_model_binding": selected_model_binding,
                "selected_hkl_binding": selected_hkl_binding,
                "source_row_table_binding": source_row_table_binding,
                "policy_selected_table_binding": policy_selected_table_binding,
                "endpoint_refinement": False,
                "maps": False,
            },
            "selected_row_table_choice": (
                "publication_runs policy-active frozen row table; it may carry policy annotations, but the paired "
                "diagnostic consumes dyn_residual_over_sigma and does not merge or apply raw_row_weight"
            ),
            "warnings": missing,
        }
        records[name] = record
        if not missing:
            runnable[name] = {
                "dataset": ds,
                "summary": summary,
                "config": config,
                "nuisance": nuisance,
                "n_basis": n_basis,
                "n_points": n_points,
                "scale_fit_split": scale_fit_split,
                "selected_table": selected_table,
            }

    if not eligible:
        payload.update(
            {
                "status": "warning_incomplete",
                "datasets": records,
                "consistency_gate": {
                    "status": "warning",
                    "passed": False,
                    "warnings": ["no publication-eligible crystal was available for a final hand diagnostic"],
                },
            }
        )
        write_json(artifact, payload)
        return payload

    opposite_model: Path | None = None
    materialization_error = ""
    if runnable:
        if not is_shelx_model(final_model):
            materialization_error = "the final shared model is not a materializable SHELX model"
        else:
            try:
                opposite_model = make_alternate_hand_model(python, final_model, root, logs, False)
            except Exception as exc:
                materialization_error = f"opposite-hand model materialization failed: {type(exc).__name__}: {exc}"
    payload["opposite_model"] = str(opposite_model or "")

    for name, context in runnable.items():
        record = records[name]
        if opposite_model is None:
            record["status"] = "warning_incomplete"
            record["warnings"] = [materialization_error or "opposite-hand model is unavailable"]
            continue
        ds = context["dataset"]
        config = context["config"]
        opposite_run = root / "runs" / str(name) / "multi_post_feedback_opposite_hand"
        diagnostic_path = root / "decisions" / "post_feedback_multi_hand" / f"{safe_label(str(name))}.json"
        record.update(
            {
                "opposite_model": str(opposite_model),
                "opposite_run": str(opposite_run),
                "opposite_row_table": str(opposite_run / "native_dyn_calc.csv"),
                "diagnostic": str(diagnostic_path),
            }
        )
        try:
            sg_window = safe_float(config.get("sg_window"), float("nan"))
            sg_window_override = sg_window if math.isfinite(sg_window) else None
            diagnostic_args = _args_with_recorded_native_contract(args, config)
            max_rows = config.get("max_rows")
            try:
                max_rows_override = int(max_rows) if max_rows not in {None, ""} else None
            except (TypeError, ValueError, OverflowError):
                max_rows_override = None
            if max_rows_override is not None and max_rows_override <= 0:
                max_rows_override = None
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    str(opposite_model),
                    dataset_baseline_hkl(prep, ds),
                    opposite_run,
                    diagnostic_args,
                    n_basis=int(context["n_basis"]),
                    n_points=int(context["n_points"]),
                    stage="multi_post_feedback_opposite_hand_final",
                    pinned=dict(context["nuisance"]),
                    make_maps=False,
                    endpoint=False,
                    max_rows_override=max_rows_override,
                    sg_window_override=sg_window_override,
                    sg_window_mode_override=str(config.get("sg_window_mode") or "") or None,
                    scale_fit_split_override=str(context["scale_fit_split"]),
                    strict_pinned_contract=True,
                ),
                logs,
                f"{name}_multi_post_feedback_opposite_hand",
                dry_run=False,
            )
            run_logged(
                [
                    python,
                    str(engine_script("absolute_structure_diagnostic.py")),
                    "--best",
                    str(context["selected_table"]),
                    "--other",
                    str(opposite_run / "native_dyn_calc.csv"),
                    "--out",
                    str(diagnostic_path),
                ],
                logs,
                f"{name}_multi_post_feedback_absolute_structure_diagnostic",
                dry_run=False,
            )
            stats = try_read_json(diagnostic_path)
            opposite_summary = try_read_json(opposite_run / "summary.json")
            row_pair_contract = _paired_row_split_contract(
                Path(context["selected_table"]),
                opposite_run / "native_dyn_calc.csv",
                nullable_numeric_fields=_row_contract_nullable_numeric_fields(
                    context["config"],
                    context["nuisance"],
                ),
            )
            row_join = stats.get("row_join") if isinstance(stats.get("row_join"), Mapping) else {}
            row_counts = {
                "selected": row_join.get("n_best", 0),
                "opposite": row_join.get("n_other", 0),
                "joined": row_join.get("n_joined", 0),
            }
            paired_t_style: dict[str, object] = {}
            paired_z_compatibility: dict[str, object] = {}
            score_margins: dict[str, object] = {}
            incomplete_reasons: list[str] = [
                f"row-pair contract: {error}"
                for error in row_pair_contract.get("errors", [])
            ]
            for split in ("all", "work", "free"):
                block = stats.get(split) if isinstance(stats.get(split), Mapping) else {}
                n_rows = int(block.get("n", 0) or 0)
                paired = safe_float(block.get("paired_row_t_delta_chi2"), float("nan"))
                chi2_margin = safe_float(block.get("delta_chi2_other_minus_best"), float("nan"))
                selected_wr2 = _native_split_dyn_wr2(context["summary"], split)
                opposite_wr2 = _native_split_dyn_wr2(opposite_summary, split)
                paired_t_style[split] = {
                    "t": paired,
                    "n": n_rows,
                    "df": max(0, n_rows - 1),
                }
                paired_z_compatibility[split] = paired
                row_counts[split] = n_rows
                score_margins[split] = {
                    "delta_chi2_opposite_minus_selected": chi2_margin,
                    "dyn_wR2_opposite_minus_selected": (
                        opposite_wr2 - selected_wr2
                        if math.isfinite(selected_wr2) and math.isfinite(opposite_wr2)
                        else float("nan")
                    ),
                    "positive_favors_selected_hand": True,
                }
                if n_rows <= 1 or not math.isfinite(paired) or not math.isfinite(chi2_margin):
                    incomplete_reasons.append(f"{split} split lacks a finite paired-row diagnostic")
            if str(stats.get("status")) != "ok":
                incomplete_reasons.append(f"diagnostic status is {stats.get('status', 'missing')}")
            n_best = int(row_join.get("n_best", 0) or 0)
            n_other = int(row_join.get("n_other", 0) or 0)
            n_joined = int(row_join.get("n_joined", 0) or 0)
            if n_joined <= 0 or n_joined != n_best or n_joined != n_other:
                incomplete_reasons.append(
                    f"row join is not exact (selected={n_best}, opposite={n_other}, joined={n_joined})"
                )
            diagnostic_duplicates_best = int(row_join.get("duplicate_keys_best", 0) or 0)
            diagnostic_duplicates_other = int(row_join.get("duplicate_keys_other", 0) or 0)
            if diagnostic_duplicates_best or diagnostic_duplicates_other:
                incomplete_reasons.append(
                    "diagnostic row keys are ambiguous "
                    f"(selected duplicates={diagnostic_duplicates_best}, "
                    f"opposite duplicates={diagnostic_duplicates_other})"
                )
            if (
                int(row_pair_contract.get("joined_rows", 0) or 0) != n_joined
                or int(row_pair_contract.get("selected_rows", 0) or 0) != n_best
                or int(row_pair_contract.get("opposite_rows", 0) or 0) != n_other
            ):
                incomplete_reasons.append(
                    "independent row-pair audit disagrees with the diagnostic row-join counts"
                )
            selected_score = score_native_summary(dict(context["summary"]))
            opposite_score = score_native_summary(opposite_summary)
            score_delta = safe_float(selected_score.get("score"), float("nan")) - safe_float(
                opposite_score.get("score"), float("nan")
            )
            opposite_gate_ok, opposite_gate_reasons = hand_diagnostic_supports_other(stats, args)
            materially_opposite = bool(
                not incomplete_reasons
                and math.isfinite(score_delta)
                and score_delta > float(getattr(args, "hand_accept_delta", 0.0))
                and opposite_gate_ok
            )
            opposite_config = (
                opposite_summary.get("config")
                if isinstance(opposite_summary.get("config"), Mapping)
                else {}
            )
            opposite_best = (
                opposite_summary.get("best_native")
                if isinstance(opposite_summary.get("best_native"), Mapping)
                else {}
            )
            opposite_nuisance = (
                opposite_best.get("nuisance")
                if isinstance(opposite_best.get("nuisance"), Mapping)
                else {}
            )
            nuisance_mismatches = _pinned_nuisance_mismatches(
                context["nuisance"],
                opposite_nuisance,
            )
            opposite_model_binding = _summary_model_binding(
                opposite_summary,
                opposite_model,
                semantic_hand=opposite_hand,
            )
            opposite_hkl_binding = _summary_hkl_binding(
                opposite_summary,
                Path(ds.hkl),
            )
            opposite_row_table_binding = _summary_native_row_table_binding(
                opposite_summary,
                opposite_run / "native_dyn_calc.csv",
            )
            physics_config_contract = _native_physics_config_contract(
                context["summary"],
                opposite_summary,
            )
            opposite_n_basis = _positive_summary_int(opposite_config, "n_basis")
            opposite_n_points = _positive_summary_int(opposite_config, "n_points")
            if opposite_n_basis != int(context["n_basis"]) or opposite_n_points != int(context["n_points"]):
                incomplete_reasons.append(
                    "opposite-hand recomputation did not preserve the selected N/Q contract"
                )
                materially_opposite = False
            if str(opposite_config.get("scale_fit_split") or "") != str(context["scale_fit_split"]):
                incomplete_reasons.append(
                    "opposite-hand recomputation did not preserve the selected scale-fit split"
                )
                materially_opposite = False
            if nuisance_mismatches:
                incomplete_reasons.extend(
                    f"pinned nuisance mismatch: {mismatch}" for mismatch in nuisance_mismatches
                )
                materially_opposite = False
            if not bool(opposite_model_binding.get("passed", False)):
                incomplete_reasons.extend(
                    f"opposite-model provenance: {error}"
                    for error in opposite_model_binding.get("errors", [])
                )
                materially_opposite = False
            if not bool(opposite_hkl_binding.get("passed", False)):
                incomplete_reasons.extend(
                    f"opposite-HKL provenance: {error}"
                    for error in opposite_hkl_binding.get("errors", [])
                )
                materially_opposite = False
            if not bool(opposite_row_table_binding.get("passed", False)):
                incomplete_reasons.extend(
                    f"opposite-row provenance: {error}"
                    for error in opposite_row_table_binding.get("errors", [])
                )
                materially_opposite = False
            if not bool(physics_config_contract.get("passed", False)):
                incomplete_reasons.extend(
                    f"physics config: {error}"
                    for error in physics_config_contract.get("errors", [])
                )
                materially_opposite = False
            comparison_contract = record.get("comparison_contract")
            if isinstance(comparison_contract, dict):
                selected_binding = comparison_contract.get("selected_model_binding")
                selected_binding_passed = bool(
                    isinstance(selected_binding, Mapping) and selected_binding.get("passed", False)
                )
                comparison_contract.update(
                    {
                        "opposite_n_basis": opposite_n_basis,
                        "opposite_n_points": opposite_n_points,
                        "opposite_scale_fit_split": opposite_config.get("scale_fit_split"),
                        "opposite_selected_nuisance": dict(opposite_nuisance),
                        "pinned_nuisance_match": not nuisance_mismatches,
                        "pinned_nuisance_mismatches": nuisance_mismatches,
                        "opposite_model_binding": opposite_model_binding,
                        "opposite_hkl_binding": opposite_hkl_binding,
                        "opposite_row_table_binding": opposite_row_table_binding,
                        "physics_config_contract": physics_config_contract,
                        "model_hand_semantics_aligned": bool(
                            selected_binding_passed and opposite_model_binding.get("passed", False)
                        ),
                    }
                )
            record.update(
                {
                    "status": (
                        "warning_incomplete"
                        if incomplete_reasons
                        else "warning_material_opposite_support"
                        if materially_opposite
                        else "consistent_no_material_opposite_support"
                    ),
                    "row_counts": row_counts,
                    "row_pair_contract": row_pair_contract,
                    "paired_t_style_positive_favors_selected": paired_t_style,
                    "paired_t_style_z_positive_favors_selected": paired_z_compatibility,
                    "paired_t_style_z_compatibility_note": (
                        "legacy field name only; values are paired Student-t-style statistics with df=n-1, "
                        "not calibrated normal Z scores"
                    ),
                    "score_margins_positive_favors_selected": score_margins,
                    "selected_score": selected_score,
                    "opposite_score": opposite_score,
                    "score_delta_selected_minus_opposite_positive_favors_opposite": score_delta,
                    "composite_score_margin_opposite_minus_selected_positive_favors_selected": -score_delta,
                    "materially_favors_opposite": materially_opposite,
                    "opposite_support_gate": {
                        "passed": opposite_gate_ok,
                        "reasons": opposite_gate_reasons,
                        "minimum_z_all": float(getattr(args, "hand_min_z_all", 2.0)),
                        "minimum_z_free": float(getattr(args, "hand_min_z_free", 0.5)),
                        "minimum_free_rows": int(getattr(args, "hand_min_free_rows", 10)),
                        "minimum_score_delta": float(getattr(args, "hand_accept_delta", 0.0)),
                    },
                    "warnings": incomplete_reasons,
                }
            )
        except Exception as exc:
            record["status"] = "warning_incomplete"
            record["warnings"] = [f"diagnostic execution failed: {type(exc).__name__}: {exc}"]

    warnings: list[str] = []
    for name in eligible:
        record = records[name]
        if str(record.get("status")) == "warning_incomplete":
            warnings.append(f"{name}: final opposite-hand diagnostic is incomplete")
        elif bool(record.get("materially_favors_opposite", False)):
            warnings.append(f"{name}: final row diagnostic materially favors the opposite hand")
    if warnings:
        status = (
            "warning_incomplete_and_opposite_support"
            if any("incomplete" in warning for warning in warnings)
            and any("opposite hand" in warning for warning in warnings)
            else "warning_incomplete"
            if any("incomplete" in warning for warning in warnings)
            else "warning_opposite_hand_support"
        )
    else:
        status = "consistent"
    payload.update(
        {
            "status": status,
            "datasets": records,
            "consistency_gate": {
                "status": "passed" if not warnings else "warning",
                "passed": not warnings,
                "warnings": warnings,
            },
            "selected_hand_after_diagnostic": str(selected_hand),
        }
    )
    write_json(artifact, payload)
    return payload


def run_feature_gate(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    family: str,
    active_model: str,
    active_run: Path,
    active_nui: dict,
) -> tuple[dict, Path, dict]:
    """Test one optional physics/nuisance family and promote it only if useful."""

    if not feature_is_enabled(args, family):
        return active_nui, active_run, {"family": family, "status": "skipped", "reason": "no varied grid values"}
    baseline_score = score_native_summary(read_json(active_run / "summary.json"))
    gate_dir = root / "runs" / ds.name / f"adaptive_{family}_gate"
    try:
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                gate_dir,
                args,
                n_basis=args.screen_n_basis,
                n_points=args.screen_n_points,
                stage=f"adaptive_{family}_gate",
                pinned=feature_grid(args, family, active_nui),
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_adaptive_{family}_gate",
            dry_run=args.dry_run,
        )
        gate_score = score_native_summary(read_json(gate_dir / "summary.json"))
    except Exception as exc:
        return active_nui, active_run, {"family": family, "status": "failed", "error": str(exc), "run": str(gate_dir)}
    improvement = safe_float(baseline_score.get("score")) - safe_float(gate_score.get("score"))
    accepted = improvement > float(args.feature_accept_delta)
    record = {
        "family": family,
        "status": "accepted" if accepted else "rejected",
        "run": str(gate_dir),
        "baseline_score": baseline_score,
        "gate_score": gate_score,
        "score_improvement": improvement,
        "accept_delta": float(args.feature_accept_delta),
    }
    if accepted:
        return best_nuisance(gate_dir) or active_nui, gate_dir, record
    return active_nui, active_run, record


def run_single_adaptive(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run the full automatic staged path for one dataset."""

    root = Path(args.out_dir)
    ds0 = prep.datasets[0]
    preflight = preflight_dataset(
        ds0,
        requested_min_peak=ds0.min_peak,
        requested_geometry_percentile=ds0.geometry_percentile,
        requested_scout_rows=args.scout_max_rows,
    )
    write_json(root / "decisions" / ds0.name / "preflight.json", preflight)
    if preflight.status == "fatal":
        raise RuntimeError(f"fatal preflight failure for {ds0.name}: {preflight.warnings}")
    ds = apply_preflight(ds0, preflight)
    scout_rows = preflight.scout_max_rows if preflight.scout_max_rows > 0 else args.scout_max_rows
    if preflight.status == "bad" and args.stop_on_bad_preflight:
        return {"status": "stopped_bad_preflight", "preflight": str(root / "decisions" / ds0.name / "preflight.json")}

    ds, transform_decision = choose_transform_by_scout(prep, ds, args, python, logs, root, scout_max_rows=scout_rows)
    screen = root / "runs" / ds.name / "adaptive_screen"
    run_logged(
        native_stage_cmd(
            python,
            ds,
            prep.structure,
            dataset_baseline_hkl(prep, ds),
            screen,
            args,
            n_basis=20 if args.fast else args.screen_n_basis,
            n_points=5 if args.fast else args.screen_n_points,
            stage="adaptive_screen",
            pinned=compact_base_nuisance(args),
            make_maps=False,
            endpoint=False,
        ),
        logs,
        f"{ds.name}_adaptive_screen",
        dry_run=args.dry_run,
    )
    screen_summary = read_json(screen / "summary.json")
    screen_frame_offsets = resolved_frame_offset_candidates(screen)
    stop, reasons = should_stop_for_bad_data(screen_summary, max_wR2=args.stop_if_wR2_above, min_improvement=args.min_bloch_wR2_gain)
    if stop and args.stop_on_bad_fit:
        write_json(root / "decisions" / ds.name / "stop_decision.json", {"stop": True, "reasons": reasons, "screen": str(screen)})
        return {"status": "stopped_bad_fit", "primary_run": str(screen), "reasons": reasons}

    active_model = prep.structure
    active_nui = best_nuisance(screen)
    absolute_structure_decision = absolute_structure_policy(prep.structure)
    active_hand = "original"
    if args.hand_control and absolute_structure_decision.get("run_hand_control", True) and not args.dry_run:
        selected_hand, absolute_structure_decision, alternate_probe = choose_hand_by_scout(prep, ds, args, python, logs, root, screen, scout_max_rows=scout_rows)
        if selected_hand == "alternate" and is_shelx_model(prep.structure):
            active_model = str(make_alternate_hand_model(python, prep.structure, root, logs, args.dry_run))
            active_nui = best_nuisance(alternate_probe) or active_nui
            active_hand = "alternate"

    regularized_nui = active_nui
    thickness_run = screen
    feature_records: list[dict] = []
    if not args.fast:
        thickness_run = root / "runs" / ds.name / "adaptive_thickness_polish"
        thickness_baseline = discrete_thickness_baseline_nuisance(args, active_nui)
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                thickness_run,
                args,
                n_basis=args.screen_n_basis,
                n_points=args.screen_n_points,
                stage="adaptive_thickness_polish",
                pinned=thickness_polish_nuisance(thickness_baseline, screen_frame_offsets),
                promotion_baseline=thickness_baseline,
                scale_fit_split_override="work",
                selection_metric_override="work_wR2",
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_adaptive_thickness_polish",
            dry_run=args.dry_run,
        )
        regularized_nui = best_nuisance(thickness_run) or active_nui
        feature_source = thickness_run
        feature_records.append(discrete_thickness_promotion_record(thickness_run, dry_run=bool(args.dry_run)))
        continuous_seed = continuous_thickness_seed_nuisance(args, regularized_nui, active_nui)
        if continuous_thickness_enabled(args) and continuous_seed:
            continuous_baseline_run = feature_source
            continuous_run = root / "runs" / ds.name / "adaptive_thickness_continuous"
            run_logged(
                continuous_thickness_cmd(
                    python,
                    ds,
                    active_model,
                    continuous_run,
                    args,
                    continuous_seed,
                    n_basis=args.screen_n_basis,
                    n_points=args.screen_n_points,
                    stage="adaptive_thickness_continuous",
                ),
                logs,
                f"{ds.name}_adaptive_thickness_continuous",
                dry_run=args.dry_run,
            )
            regularized_nui = best_nuisance(continuous_run) or regularized_nui or continuous_seed
            regularized_nui, feature_source, materialization_record = materialize_continuous_nuisance_run(
                prep,
                ds,
                args,
                python,
                logs,
                active_model=active_model,
                baseline_run=continuous_baseline_run,
                continuous_run=continuous_run,
                selected_nuisance=regularized_nui,
                n_basis=args.screen_n_basis,
                n_points=args.screen_n_points,
                stage="adaptive_thickness_continuous_materialized",
            )
            feature_records.append(materialization_record)
        for family in ["row_resolution", "absorption", "sg_jitter", "sigma_error", "dose"]:
            regularized_nui, feature_source, record = run_feature_gate(
                prep,
                ds,
                args,
                python,
                logs,
                root,
                family=family,
                active_model=active_model,
                active_run=feature_source,
                active_nui=regularized_nui,
            )
            feature_records.append(record)
        thickness_run = feature_source

    final_model = active_model
    final_n_basis = 20 if args.fast else int(args.final_basis_fallback)
    structure_strategy = resolve_structure_update_strategy(args, multi_dataset=False)
    structure_update_run = None
    feedback_final_run: Path | None = None
    if args.mode in {"regularize", "auto"} and not args.fast:
        if structure_strategy == "shelxl_feedback":
            final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, active_model, "adaptive_shelxl_feedback")
            struct_dir = root / "runs" / ds.name / "adaptive_shelxl_feedback"
            source_summary = thickness_run / "summary.json"
            run_logged(
                shelxl_feedback_cmd(
                    python,
                    ds,
                    active_model,
                    source_summary,
                    dataset_baseline_hkl(prep, ds),
                    struct_dir,
                    args,
                    n_basis=final_n_basis,
                ),
                logs,
                f"{ds.name}_adaptive_shelxl_feedback",
                dry_run=args.dry_run,
            )
            structure_update_run = struct_dir
            if args.dry_run:
                feedback_final_run = struct_dir / f"cycle_{int(args.shelxl_feedback_cycles):02d}_N{int(final_n_basis)}Q{int(args.n_points)}"
            else:
                feedback_summary = try_read_json(struct_dir / "shelxl_feedback_loop_summary.json")
                feedback_final_run = shelxl_feedback_current_run(struct_dir)
                if feedback_final_run is None:
                    runs = feedback_summary.get("runs") or []
                    failed = runs[-1] if runs else {}
                    raise RuntimeError(
                        "SHELXL feedback did not produce a usable RES model. "
                        f"Last cycle status: returncode={failed.get('returncode', 'n/a')}; "
                        f"endpoint={failed.get('output_model', 'n/a')}; "
                        f"health={failed.get('endpoint_res_health', {})}. "
                        f"See {struct_dir / 'shelxl_feedback_loop_summary.json'} and the cycle run.log."
                    )
                feedback_model = shelxl_feedback_current_model(struct_dir)
                if feedback_model is not None:
                    final_model = str(feedback_model)
        elif structure_strategy == "native_gradient":
            struct_dir = root / "runs" / ds.name / "adaptive_structure_update"
            run_logged(structure_cmd(python, ds, active_model, dataset_baseline_hkl(prep, ds), struct_dir, args, regularized_nui), logs, f"{ds.name}_adaptive_structure_update", dry_run=args.dry_run)
            structure_update_run = struct_dir
            updated = struct_dir / "bloch_updated_model.res"
            if updated.exists():
                final_model = str(updated)

    if feedback_final_run is not None:
        final = feedback_final_run
    else:
        final = root / "runs" / ds.name / ("adaptive_fast_final" if args.fast else "adaptive_final")
        final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, final_model, "adaptive_final")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                final,
                args,
                n_basis=final_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="adaptive_final",
                pinned=regularized_nui,
                make_maps=args.make_maps,
                endpoint=True,
            ),
            logs,
            f"{ds.name}_adaptive_final",
            dry_run=args.dry_run,
        )
    endpoint_feedback_decision: dict = {}
    endpoint_feedback_run = None
    if args.mode in {"regularize", "auto"} and not args.fast and not args.dry_run and feedback_final_run is None:
        run_feedback, endpoint_feedback_decision = should_run_endpoint_feedback(final, args)
        endpoint_model = best_endpoint_model(final)
        if run_feedback and endpoint_model is not None:
            endpoint_feedback_run = root / "runs" / ds.name / "adaptive_endpoint_feedback"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    str(endpoint_model),
                    dataset_baseline_hkl(prep, ds),
                    endpoint_feedback_run,
                    args,
                    n_basis=final_n_basis,
                    n_points=args.n_points,
                    stage="adaptive_endpoint_feedback",
                    pinned=best_nuisance(final) or regularized_nui,
                    make_maps=args.make_maps,
                    endpoint=True,
                ),
                logs,
                f"{ds.name}_adaptive_endpoint_feedback",
                dry_run=args.dry_run,
            )
            final = endpoint_feedback_run
            final_model = str(best_endpoint_model(endpoint_feedback_run) or endpoint_model)
        elif run_feedback:
            endpoint_feedback_decision = {**endpoint_feedback_decision, "status": "promoted_but_no_endpoint_model_found"}
    absolute_structure_diagnostic = ""
    post_feedback_hand_promotion: dict = {}
    if args.hand_control and absolute_structure_decision.get("run_hand_control", True) and not args.dry_run:
        other = root / "runs" / ds.name / "adaptive_other_hand_check"
        # Hand inversion is involutive.  Probe the opposite of the *refined
        # current model* in both directions so the paired diagnostic is not
        # confounded by comparing refined coordinates with the input model.
        other_model = final_model
        alternate_other = True
        run_logged(
            native_stage_cmd(
                python,
                ds,
                other_model,
                dataset_baseline_hkl(prep, ds),
                other,
                args,
                n_basis=20 if args.fast else final_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="adaptive_other_hand",
                pinned=best_nuisance(final) or regularized_nui,
                alternate_hand=alternate_other,
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_adaptive_other_hand_check",
            dry_run=args.dry_run,
        )
        stats_path = root / "runs" / ds.name / "adaptive_absolute_structure_diagnostic.json"
        run_logged(
            [
                python,
                str(engine_script("absolute_structure_diagnostic.py")),
                "--best",
                str(final / "native_dyn_calc.csv"),
                "--other",
                str(other / "native_dyn_calc.csv"),
                "--out",
                str(stats_path),
            ],
            logs,
            f"{ds.name}_adaptive_absolute_structure_diagnostic",
            dry_run=args.dry_run,
        )
        absolute_structure_diagnostic = str(stats_path)
        final_summary = read_json(final / "summary.json")
        other_summary = read_json(other / "summary.json")
        final_score = score_native_summary(final_summary)
        other_score = score_native_summary(other_summary)
        post_feedback_delta = final_score.get("score", float("inf")) - other_score.get("score", float("inf"))
        post_feedback_hand_promotion = {
            "status": "not_promoted",
            "active_hand": active_hand,
            "opposite_hand": "alternate" if active_hand == "original" else "original",
            "final_score": final_score,
            "opposite_score": other_score,
            "score_delta_final_minus_opposite": post_feedback_delta,
            "accept_delta": float(args.hand_accept_delta),
            "diagnostic": str(stats_path),
            "opposite_run": str(other),
        }
        diagnostic_gate_ok, diagnostic_gate_reasons = hand_diagnostic_supports_other(
            try_read_json(stats_path) or {},
            args,
        )
        post_feedback_hand_promotion["diagnostic_quality_gate_status"] = "passed" if diagnostic_gate_ok else "rejected_switch"
        post_feedback_hand_promotion["diagnostic_quality_gate_reasons"] = diagnostic_gate_reasons
        if post_feedback_delta > float(args.hand_accept_delta) and diagnostic_gate_ok and is_shelx_model(final_model):
            promoted_model = make_alternate_hand_model(python, final_model, root, logs, args.dry_run)
            promoted = root / "runs" / ds.name / "adaptive_promoted_hand_final"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    str(promoted_model),
                    dataset_baseline_hkl(prep, ds),
                    promoted,
                    args,
                    n_basis=20 if args.fast else final_n_basis,
                    n_points=5 if args.fast else args.n_points,
                    stage="adaptive_promoted_hand_final",
                    pinned=best_nuisance(final) or regularized_nui,
                    alternate_hand=False,
                    make_maps=args.make_maps,
                    endpoint=True,
                ),
                logs,
                f"{ds.name}_adaptive_promoted_hand_final",
                dry_run=args.dry_run,
            )
            promoted_summary = read_json(promoted / "summary.json")
            promoted_score = score_native_summary(promoted_summary)
            final_score_after = score_native_summary(read_json(final / "summary.json"))
            promoted_delta = final_score_after.get("score", float("inf")) - promoted_score.get("score", float("inf"))
            post_feedback_hand_promotion.update(
                {
                    "status": "promoted" if promoted_delta > float(args.hand_accept_delta) else "tested_not_promoted",
                    "promoted_run": str(promoted),
                    "promoted_model_input": str(promoted_model),
                    "promoted_score": promoted_score,
                    "score_delta_final_minus_promoted": promoted_delta,
                }
            )
            if promoted_delta > float(args.hand_accept_delta):
                final = promoted
                final_model = str(best_endpoint_model(promoted) or promoted_model)
                active_hand = post_feedback_hand_promotion["opposite_hand"]
                absolute_structure_decision = {
                    **absolute_structure_decision,
                    "post_feedback_selected": active_hand,
                    "post_feedback_selected_hand": active_hand,
                    "post_feedback_promotion": post_feedback_hand_promotion,
                }
        elif post_feedback_delta > float(args.hand_accept_delta) and not diagnostic_gate_ok:
            post_feedback_hand_promotion["status"] = "rejected_by_diagnostic_quality_gate"

    no_h_run = None
    if args.h_omit in {"on", "both", "only"} and is_shelx_model(final_model):
        no_h = make_noh_model(python, final_model, root, logs, args.dry_run)
        no_h_run = root / "runs" / ds.name / "adaptive_final_noH"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                str(no_h),
                dataset_baseline_hkl(prep, ds),
                no_h_run,
                args,
                n_basis=final_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="adaptive_noH",
                pinned=best_nuisance(final) or regularized_nui,
                make_maps=args.make_maps,
                endpoint=True,
            ),
            logs,
            f"{ds.name}_adaptive_final_noH",
            dry_run=args.dry_run,
        )

    scale_polish = run_scale_window_polish(
        python,
        final,
        root / "scale_polish" / ds.name,
        logs,
        f"{ds.name}_adaptive_final",
        args,
    )
    resolution_limit = run_resolution_limit_gate(
        python,
        {ds.name: final},
        final_model,
        root / "resolution_limits" / ds.name,
        logs,
        ds.name,
        args,
        title=f"GROND resolution-limit promotion: {ds.name}",
        scope="dataset",
    )
    forensics = run_forensic_gate(
        python,
        final / "native_dyn_calc.csv",
        ds.hkl,
        final_model,
        root / "forensics" / ds.name,
        logs,
        ds.name,
        args,
    )
    sg_window_scout = run_sg_window_scout(
        python,
        prep,
        ds,
        final_model,
        final,
        root / "sg_window_scout" / ds.name,
        logs,
        f"{ds.name}_adaptive_final",
        args,
    )
    decision_summary = {
        "status": "ok",
        "dataset": ds.name,
        "selected_transform": ds.hkl_transform,
        "selected_hand": active_hand,
        "structure_update_strategy": structure_strategy,
        "structure_update_run": str(structure_update_run) if structure_update_run else "",
        "transform_decision": transform_decision,
        "absolute_structure_decision": absolute_structure_decision,
        "feature_decisions": feature_records,
        "endpoint_feedback_decision": endpoint_feedback_decision,
        "endpoint_feedback_run": str(endpoint_feedback_run) if endpoint_feedback_run else "",
        "post_feedback_hand_promotion": post_feedback_hand_promotion,
        "scale_polish": scale_polish,
        "resolution_limit": resolution_limit,
        "sg_window_scout": sg_window_scout,
        "screen_run": str(screen),
        "thickness_run": str(thickness_run),
        "final_run": str(final),
        "final_model": final_model,
        "forensics": forensics,
        "preflight": str(root / "decisions" / ds0.name / "preflight.json"),
    }
    write_json(root / "decisions" / ds.name / "decision_tree.json", decision_summary)
    return {
        "primary_run": str(final),
        "no_h_run": str(no_h_run) if no_h_run else "",
        "model": final_model,
        "absolute_structure_diagnostic": absolute_structure_diagnostic,
        "post_feedback_hand_promotion": post_feedback_hand_promotion,
        "forensics": forensics,
        "scale_polish": scale_polish,
        "resolution_limits": {ds.name: resolution_limit},
        "sg_window_scout": {ds.name: sg_window_scout},
        "decision_tree": str(root / "decisions" / ds.name / "decision_tree.json"),
    }


def run_single(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run either the automatic or fixed single-dataset workflow."""

    if args.auto_decide:
        return run_single_adaptive(prep, args, python, logs)
    ds = prep.datasets[0]
    root = Path(args.out_dir)
    make_maps = args.make_maps
    n_basis = 20 if args.fast else args.screen_n_basis
    n_points = 5 if args.fast else args.screen_n_points
    screen = root / "runs" / ds.name / ("fast_N20_Q5" if args.fast else "screen")
    run_logged(
        native_stage_cmd(
            python,
            ds,
            prep.structure,
            dataset_baseline_hkl(prep, ds),
            screen,
            args,
            n_basis=n_basis,
            n_points=n_points,
            stage="screen",
            make_maps=make_maps and args.mode == "bloch",
        ),
        logs,
        f"{ds.name}_bloch",
        dry_run=args.dry_run,
    )
    nui = best_nuisance(screen)
    if args.mode in {"regularize", "auto"} and not args.fast and not args.dry_run and not nui:
        raise RuntimeError(f"No best nuisance result found in {screen}; refusing to launch an unpinned final grid.")
    final_model = prep.structure
    if args.mode in {"regularize", "auto"} and not args.fast:
        struct_dir = root / "runs" / ds.name / "structure_update"
        run_logged(structure_cmd(python, ds, prep.structure, dataset_baseline_hkl(prep, ds), struct_dir, args, nui), logs, f"{ds.name}_structure_update", dry_run=args.dry_run)
        updated = struct_dir / "bloch_updated_model.res"
        if updated.exists():
            final_model = str(updated)
    final = screen
    absolute_structure_diagnostic = ""
    if args.mode in {"regularize", "auto"} and not args.fast:
        final = root / "runs" / ds.name / "final"
        final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, final_model, "final")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                final,
                args,
                n_basis=final_n_basis,
                n_points=args.n_points,
                stage="final",
                pinned=nui,
                make_maps=make_maps,
            ),
            logs,
            f"{ds.name}_final",
            dry_run=args.dry_run,
        )
    hand_policy = absolute_structure_policy(final_model)
    if args.hand_control and hand_policy.get("run_hand_control", True) and not args.dry_run:
        hand_run = root / "runs" / ds.name / "hand_alt"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                hand_run,
                args,
                n_basis=20 if args.fast else args.hand_n_basis,
                n_points=5 if args.fast else args.hand_n_points,
                stage="hand",
                pinned=best_nuisance(final) or nui,
                alternate_hand=True,
                make_maps=False,
            ),
            logs,
            f"{ds.name}_hand_alt",
            dry_run=args.dry_run,
        )
        stats_path = root / "runs" / ds.name / "absolute_structure_diagnostic.json"
        run_logged(
            [
                python,
                str(engine_script("absolute_structure_diagnostic.py")),
                "--best",
                str(final / "native_dyn_calc.csv"),
                "--other",
                str(hand_run / "native_dyn_calc.csv"),
                "--out",
                str(stats_path),
            ],
            logs,
            f"{ds.name}_absolute_structure_diagnostic",
            dry_run=args.dry_run,
        )
        absolute_structure_diagnostic = str(stats_path)
    no_h_run = None
    if args.h_omit in {"on", "both", "only"} and is_shelx_model(final_model):
        no_h = make_noh_model(python, final_model, root, logs, args.dry_run)
        no_h_run = root / "runs" / ds.name / "final_noH"
        no_h_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, str(no_h), "final_noH")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                str(no_h),
                dataset_baseline_hkl(prep, ds),
                no_h_run,
                args,
                n_basis=20 if args.fast else no_h_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="noH",
                pinned=best_nuisance(final) or nui,
                make_maps=make_maps,
            ),
            logs,
            f"{ds.name}_final_noH",
            dry_run=args.dry_run,
        )
    scale_polish = run_scale_window_polish(
        python,
        final,
        root / "scale_polish" / ds.name,
        logs,
        f"{ds.name}_final",
        args,
    )
    resolution_limit = run_resolution_limit_gate(
        python,
        {ds.name: final},
        final_model,
        root / "resolution_limits" / ds.name,
        logs,
        ds.name,
        args,
        title=f"GROND resolution-limit promotion: {ds.name}",
        scope="dataset",
    )
    forensics = run_forensic_gate(
        python,
        final / "native_dyn_calc.csv",
        ds.hkl,
        final_model,
        root / "forensics" / ds.name,
        logs,
        ds.name,
        args,
    )
    sg_window_scout = run_sg_window_scout(
        python,
        prep,
        ds,
        final_model,
        final,
        root / "sg_window_scout" / ds.name,
        logs,
        f"{ds.name}_final",
        args,
    )
    return {
        "primary_run": str(final),
        "no_h_run": str(no_h_run) if no_h_run else "",
        "model": final_model,
        "absolute_structure_diagnostic": absolute_structure_diagnostic,
        "forensics": forensics,
        "scale_polish": scale_polish,
        "resolution_limits": {ds.name: resolution_limit},
        "sg_window_scout": {ds.name: sg_window_scout},
    }


def run_multi(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run one structure against multiple datasets and combine endpoints."""

    root = Path(args.out_dir)
    scout_rows_by_name = {ds.name: int(getattr(args, "scout_max_rows", 1000)) for ds in prep.datasets}
    if args.auto_decide:
        def decide_dataset(ds0: DatasetSpec) -> tuple[DatasetSpec, int]:
            preflight = preflight_dataset(
                ds0,
                requested_min_peak=ds0.min_peak,
                requested_geometry_percentile=ds0.geometry_percentile,
                requested_scout_rows=args.scout_max_rows,
            )
            write_json(root / "decisions" / ds0.name / "preflight.json", preflight)
            if preflight.status == "fatal":
                raise RuntimeError(f"fatal preflight failure for {ds0.name}: {preflight.warnings}")
            ds = apply_preflight(ds0, preflight)
            scout_rows = preflight.scout_max_rows if preflight.scout_max_rows > 0 else args.scout_max_rows
            ds, _transform_decision = choose_transform_by_scout(prep, ds, args, python, logs, root, scout_max_rows=scout_rows)
            return ds, int(scout_rows)

        if len(prep.datasets) > 1 and not args.dry_run:
            decided_by_name: dict[str, DatasetSpec] = {}
            decided_scout_rows: dict[str, int] = {}
            max_workers = min(len(prep.datasets), transform_scout_job_count(args))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="grond-dataset-transform") as pool:
                futures = {pool.submit(decide_dataset, ds0): ds0.name for ds0 in prep.datasets}
                for future in as_completed(futures):
                    decided, scout_rows = future.result()
                    decided_by_name[decided.name] = decided
                    decided_scout_rows[decided.name] = scout_rows
            decided = [decided_by_name[ds0.name] for ds0 in prep.datasets]
            scout_rows_by_name = {ds0.name: decided_scout_rows[ds0.name] for ds0 in prep.datasets}
        else:
            decided_rows = [decide_dataset(ds0) for ds0 in prep.datasets]
            decided = [item[0] for item in decided_rows]
            scout_rows_by_name = {item[0].name: item[1] for item in decided_rows}
        prep = PrepConfig(
            structure=prep.structure,
            datasets=decided,
            baseline_hkl=prep.baseline_hkl,
            notes=prep.notes,
            default_hkl_transform=prep.default_hkl_transform,
            geometry_percentile=prep.geometry_percentile,
            metadata=prep.metadata,
        )
    screen_runs: dict[str, Path] = {}
    initial_runs: dict[str, Path] = {}
    nuisance: dict[str, dict] = {}
    for ds in prep.datasets:
        run_dir = root / "runs" / ds.name / ("fast_N20_Q5" if args.fast else "screen")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                run_dir,
                args,
                n_basis=20 if args.fast else args.screen_n_basis,
                n_points=5 if args.fast else args.screen_n_points,
                stage="screen",
                pinned=compact_base_nuisance(args),
                make_maps=False,
            ),
            logs,
            f"{ds.name}_screen",
            dry_run=args.dry_run,
        )
        screen_runs[ds.name] = run_dir
        initial_runs[ds.name] = run_dir
        nuisance[ds.name] = best_nuisance(run_dir)

    selected_hand, shared_hand_decision, alternate_hand_runs = choose_shared_hand_by_scout(
        prep,
        args,
        python,
        logs,
        root,
        screen_runs,
        scout_rows_by_name,
    )
    shared_model = prep.structure
    if selected_hand == "alternate":
        shared_model = str(make_alternate_hand_model(python, prep.structure, root, logs, args.dry_run))
        for ds in prep.datasets:
            nuisance[ds.name] = best_nuisance(alternate_hand_runs[ds.name]) or nuisance[ds.name]
    shared_hand_decision["selected_model"] = shared_model
    write_json(root / "decisions" / "shared_absolute_structure_decision.json", shared_hand_decision)
    for ds in prep.datasets:
        decision_path = root / "decisions" / ds.name / "absolute_structure_decision.json"
        decision = try_read_json(decision_path)
        decision["selected_model"] = shared_model
        write_json(decision_path, decision)

    if args.fast and selected_hand == "alternate":
        for ds in prep.datasets:
            selected_run = root / "runs" / ds.name / "fast_N20_Q5_selected_hand"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    shared_model,
                    dataset_baseline_hkl(prep, ds),
                    selected_run,
                    args,
                    n_basis=20,
                    n_points=5,
                    stage="multi_fast_selected_hand",
                    pinned=nuisance[ds.name],
                    make_maps=False,
                ),
                logs,
                f"{ds.name}_multi_fast_selected_hand",
                dry_run=args.dry_run,
            )
            initial_runs[ds.name] = selected_run
            nuisance[ds.name] = best_nuisance(selected_run) or nuisance[ds.name]

    for ds in prep.datasets:
        active_run = initial_runs[ds.name]
        active_nui = nuisance[ds.name]
        screen_frame_offsets = resolved_frame_offset_candidates(screen_runs[ds.name])
        if not args.fast:
            thick = root / "runs" / ds.name / "multi_thickness_polish"
            thickness_baseline = discrete_thickness_baseline_nuisance(args, active_nui)
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    shared_model,
                    dataset_baseline_hkl(prep, ds),
                    thick,
                    args,
                    n_basis=args.screen_n_basis,
                    n_points=args.screen_n_points,
                    stage="multi_thickness_polish",
                    pinned=thickness_polish_nuisance(thickness_baseline, screen_frame_offsets),
                    promotion_baseline=thickness_baseline,
                    scale_fit_split_override="work",
                    selection_metric_override="work_wR2",
                    make_maps=False,
                    endpoint=False,
                ),
                logs,
                f"{ds.name}_multi_thickness_polish",
                dry_run=args.dry_run,
            )
            active_run = thick
            active_nui = best_nuisance(thick) or active_nui
            feature_records: list[dict] = [
                discrete_thickness_promotion_record(thick, dry_run=bool(args.dry_run))
            ]
            continuous_seed = continuous_thickness_seed_nuisance(args, active_nui)
            if continuous_thickness_enabled(args) and continuous_seed:
                continuous_baseline_run = active_run
                continuous = root / "runs" / ds.name / "multi_thickness_continuous"
                run_logged(
                    continuous_thickness_cmd(
                        python,
                        ds,
                        shared_model,
                        continuous,
                        args,
                        continuous_seed,
                        n_basis=args.screen_n_basis,
                        n_points=args.screen_n_points,
                        stage="multi_thickness_continuous",
                    ),
                    logs,
                    f"{ds.name}_multi_thickness_continuous",
                    dry_run=args.dry_run,
                )
                active_nui = best_nuisance(continuous) or active_nui or continuous_seed
                active_nui, active_run, materialization_record = materialize_continuous_nuisance_run(
                    prep,
                    ds,
                    args,
                    python,
                    logs,
                    active_model=shared_model,
                    baseline_run=continuous_baseline_run,
                    continuous_run=continuous,
                    selected_nuisance=active_nui,
                    n_basis=args.screen_n_basis,
                    n_points=args.screen_n_points,
                    stage="multi_thickness_continuous_materialized",
                )
            else:
                materialization_record = None
            if materialization_record is not None:
                feature_records.append(materialization_record)
            for family in ["row_resolution", "absorption", "sg_jitter", "sigma_error", "dose"]:
                active_nui, active_run, record = run_feature_gate(
                    prep,
                    ds,
                    args,
                    python,
                    logs,
                    root,
                    family=family,
                    active_model=shared_model,
                    active_run=active_run,
                    active_nui=active_nui,
                )
                feature_records.append(record)
            write_json(root / "decisions" / ds.name / "feature_decisions.json", feature_records)
        initial_runs[ds.name] = active_run
        nuisance[ds.name] = active_nui

    structure_dataset_policy = run_dataset_policy_gate(
        initial_runs,
        root / "dataset_policy" / "structure_update",
        args,
        label="structure_update",
    )
    require_materialized_dataset_policy(
        structure_dataset_policy,
        context="structure-update",
        dry_run=bool(args.dry_run),
    )
    datasets_csv = root / "prep_datasets_with_nuisance.csv"
    write_datasets_csv(datasets_csv, prep, apply_dataset_policy_to_nuisance(nuisance, structure_dataset_policy))
    final_model = shared_model
    structure_strategy = resolve_structure_update_strategy(args, multi_dataset=True)
    structure_feedback: dict = {"status": "not_run", "strategy": structure_strategy}
    if args.mode in {"regularize", "auto"} and not args.fast and structure_strategy != "off":
        if structure_strategy == "shelxl_feedback":
            final_model, structure_feedback = run_multi_shelxl_feedback_loop(
                prep,
                args,
                python,
                logs,
                root,
                start_model=shared_model,
                nuisance=nuisance,
                structure_dataset_policy=structure_dataset_policy,
            )
        elif structure_strategy == "native_gradient":
            shared = root / "runs" / "shared_structure_update"
            cmd = [
                python,
                str(engine_script("native_multi_structure_update.py")),
                "--datasets-csv",
                str(datasets_csv),
                "--res",
                shared_model,
                "--out-dir",
                str(shared),
                "--n-basis",
                str(args.structure_n_basis),
                "--n-points",
                str(args.structure_n_points),
                "--topup-d-min",
                f"{effective_topup_d_min(args, min_d_spacing=min(ds.min_d_spacing for ds in prep.datasets)):g}",
                "--min-d-spacing",
                "0.70",
                "--min-peak",
                "90",
                "--geometry-percentile",
                f"{prep.geometry_percentile:g}",
                "--block-schedule",
                args.structure_blocks,
                "--atom-selection",
                args.structure_atom_selection,
                "--max-nfev",
                str(args.structure_max_nfev),
                "--compute-backend",
                args.compute_backend,
                "--jax-batch-size",
                str(args.jax_batch_size),
                "--jax-matrix-bucket-size",
                str(args.jax_matrix_bucket_size),
                "--bethe-strong-count",
                str(args.bethe_strong_count),
                "--bethe-variant",
                args.bethe_variant,
                "--bethe-promote-denom",
                f"{args.bethe_promote_denom:g}",
                "--bethe-denom-eps",
                f"{args.bethe_denom_eps:g}",
                "--profile-quadrature",
                args.profile_quadrature,
                "--profile-accumulation",
                args.profile_accumulation,
                "--early-stop-window",
                str(args.structure_early_stop_window),
                "--early-stop-min-evals",
                str(args.structure_early_stop_min_evals),
                "--early-stop-min-improvement",
                f"{args.structure_early_stop_min_improvement:g}",
                "--early-stop-metric",
                args.structure_early_stop_metric,
                "--restraints",
                args.restraints,
                "--scale-fit-split",
                "work",
                "--workers",
                str(args.workers),
                "--chunksize",
                str(args.chunksize),
            ]
            run_logged(cmd, logs, "shared_structure_update", dry_run=args.dry_run)
            accepted = shared / "accepted_model.res"
            structure_feedback = {"status": "ok" if accepted.exists() else "no_accepted_model", "strategy": structure_strategy, "run": str(shared)}
            if accepted.exists():
                final_model = str(accepted)
        else:
            structure_feedback = {"status": "unsupported_strategy", "strategy": structure_strategy}
    final_runs: dict[str, Path] = {}
    forensic_runs: dict[str, dict] = {}
    scale_polish_runs: dict[str, dict] = {}
    resolution_limits: dict[str, dict] = {}
    sg_window_scout_runs: dict[str, dict] = {}
    for ds in prep.datasets:
        run_dir = initial_runs[ds.name] if args.fast else root / "runs" / ds.name / "final"
        if not args.fast:
            final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, final_model, "final")
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    final_model,
                    dataset_baseline_hkl(prep, ds),
                    run_dir,
                    args,
                    n_basis=final_n_basis,
                    n_points=args.n_points,
                    stage="final",
                    pinned=nuisance[ds.name],
                    make_maps=args.make_maps,
                ),
                logs,
                f"{ds.name}_final",
                dry_run=args.dry_run,
            )
        final_runs[ds.name] = run_dir
        scale_polish_runs[ds.name] = run_scale_window_polish(
            python,
            run_dir,
            root / "scale_polish" / ds.name,
            logs,
            f"{ds.name}_final",
            args,
        )
        resolution_limits[ds.name] = run_resolution_limit_gate(
            python,
            {ds.name: run_dir},
            final_model,
            root / "resolution_limits" / ds.name,
            logs,
            ds.name,
            args,
            title=f"GROND resolution-limit promotion: {ds.name}",
            scope="dataset",
        )
        forensic_runs[ds.name] = run_forensic_gate(
            python,
            run_dir / "native_dyn_calc.csv",
            ds.hkl,
            final_model,
            root / "forensics" / ds.name,
            logs,
            ds.name,
            args,
        )
    final_dataset_policy = run_dataset_policy_gate(
        final_runs,
        root / "dataset_policy" / "final",
        args,
        label="final",
    )
    require_materialized_dataset_policy(
        final_dataset_policy,
        context="final",
        dry_run=bool(args.dry_run),
    )
    if args.dry_run or final_dataset_policy.get("status") in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
        full_input_runs = final_runs
        full_publication_runs = final_runs
    else:
        weighted_runs = write_policy_weighted_runs(
            final_runs,
            final_dataset_policy,
            root / "dataset_policy" / "final" / "weighted_runs",
            include_endpoint_only=True,
        )
        if not weighted_runs:
            raise RuntimeError(
                "multi-dataset final policy selected no endpoint-contributing dataset; "
                "refusing to fall back to all final runs"
            )
        full_input_runs = weighted_runs
        full_publication_runs = policy_active_runs(weighted_runs, final_dataset_policy)
        require_policy_active_runs(full_publication_runs, context="final publication")
    post_feedback_multi_hand_diagnostic = run_multi_post_feedback_hand_diagnostic(
        prep,
        args,
        python,
        logs,
        root,
        final_model=final_model,
        selected_hand=selected_hand,
        final_runs=final_runs,
        publication_runs=full_publication_runs,
        shared_hand_decision=shared_hand_decision,
    )
    combined_input_runs, publication_runs, scale_contract_selection = run_multi_scale_contract_selection(
        python=python,
        full_runs=final_runs,
        full_input_runs=full_input_runs,
        full_publication_runs=full_publication_runs,
        scale_polish=scale_polish_runs,
        final_dataset_policy=final_dataset_policy,
        final_model=final_model,
        root=root,
        logs=logs,
        args=args,
    )
    combined = root / "runs" / "combined"
    run_logged(combine_cmd(python, combined_input_runs, final_model, combined, args), logs, "combine_final", dry_run=args.dry_run)
    merged_resolution_limit = run_resolution_limit_gate(
        python,
        combined_input_runs,
        final_model,
        root / "resolution_limits" / "combined",
        logs,
        "combined",
        args,
        title=f"GROND merged resolution-limit promotion: {', '.join(ds.name for ds in prep.datasets)}",
        scope="merged",
    )
    for ds in prep.datasets:
        sg_window_scout_runs[ds.name] = run_sg_window_scout(
            python,
            prep,
            ds,
            final_model,
            final_runs[ds.name],
            root / "sg_window_scout" / ds.name,
            logs,
            f"{ds.name}_final",
            args,
        )
    return {
        "combined": str(combined),
        "model": final_model,
        "selected_hand": selected_hand,
        "shared_absolute_structure_decision": str(
            root / "decisions" / "shared_absolute_structure_decision.json"
        ),
        "runs": {k: str(v) for k, v in final_runs.items()},
        "combined_input_runs": {k: str(v) for k, v in combined_input_runs.items()},
        "publication_runs": {k: str(v) for k, v in publication_runs.items()},
        "publication_full_runs": {k: str(v) for k, v in full_publication_runs.items()},
        "post_feedback_multi_hand_diagnostic": post_feedback_multi_hand_diagnostic,
        "structure_update_strategy": structure_strategy,
        "structure_feedback": structure_feedback,
        "forensics": forensic_runs,
        "scale_polish": scale_polish_runs,
        "scale_contract_selection": scale_contract_selection,
        "resolution_limits": resolution_limits,
        "merged_resolution_limit": merged_resolution_limit,
        "sg_window_scout": sg_window_scout_runs,
        "dataset_policy": {
            "structure_update": structure_dataset_policy,
            "final": final_dataset_policy,
        },
    }


def prep_from_args(args: argparse.Namespace) -> PrepConfig:
    """Create a preparation record from CLI flags or an existing prep file."""

    if args.prep:
        prep = load_prep(Path(args.prep))
    else:
        prep_metadata = {}
        if args.xds_dir:
            hkl, xds_dir = resolve_hkl(args.xds_dir)
        elif args.hkl:
            hkl, xds_dir = resolve_hkl(args.hkl)
        elif getattr(args, "dials_refl", ""):
            if not getattr(args, "dials_expt", ""):
                raise SystemExit("Provide --dials-expt with --dials-refl.")
            from .io.dials_bridge import convert_dials_to_grond

            bridge_dir = Path(args.out_dir).expanduser().resolve() / "prepared" / f"dials_{safe_label(args.dataset_name)}"
            bridge = convert_dials_to_grond(
                args.dials_refl,
                args.dials_expt,
                bridge_dir,
                intensity_mode=args.dials_intensity_mode,
                sigma_mode=args.dials_sigma_mode,
                sigma_scale=args.dials_sigma_scale,
                dials_python=args.dials_python,
                min_partiality_write=args.dials_min_partiality_write,
                keep_dont_integrate=bool(args.dials_keep_dont_integrate) and not bool(args.dials_drop_dont_integrate),
                drop_bad_flags=not bool(args.dials_keep_bad_flags),
                dataset_name=args.dataset_name,
            )
            hkl, xds_dir = str(bridge.hkl), str(bridge.out_dir)
            prep_metadata["dials_bridge_summary"] = str(bridge.summary)
        else:
            raise SystemExit("Provide --prep, --hkl/--xds-dir, or --dials-refl/--dials-expt.")
        if not args.structure:
            raise SystemExit("Provide --structure when not using --prep.")
        prep = PrepConfig(
            structure=str(Path(args.structure).expanduser().resolve()),
            baseline_hkl=str(Path(args.baseline_hkl).expanduser().resolve()) if args.baseline_hkl else "",
            default_hkl_transform=args.hkl_transform,
            geometry_percentile=args.geometry_percentile,
            datasets=[
                DatasetSpec(
                    name=args.dataset_name,
                    hkl=hkl,
                    xds_dir=xds_dir,
                    hkl_transform=args.hkl_transform,
                    min_d_spacing=args.min_d_spacing,
                    min_peak=args.min_peak,
                    target_mode=args.target_mode,
                    include_nonpositive=args.include_nonpositive,
                    geometry_percentile=args.geometry_percentile,
                )
            ],
            metadata=prep_metadata,
        )
    for ds in prep.datasets:
        if args.geometry_percentile is not None:
            ds.geometry_percentile = float(args.geometry_percentile)
    return prep


def cmd_prep(args: argparse.Namespace) -> int:
    """CLI entrypoint for writing a reusable prep JSON file."""

    datasets = [
        parse_dataset_spec(item, default_hkl_transform=args.hkl_transform, geometry_percentile=args.geometry_percentile)
        for item in args.dataset
    ]
    prep = PrepConfig(
        structure=str(Path(args.structure).expanduser().resolve()),
        baseline_hkl=str(Path(args.baseline_hkl).expanduser().resolve()) if args.baseline_hkl else "",
        notes=args.notes,
        default_hkl_transform=args.hkl_transform,
        geometry_percentile=args.geometry_percentile,
        datasets=datasets,
    )
    write_prep(Path(args.out), prep)
    write_datasets_csv(Path(args.out).with_suffix(".datasets.csv"), prep)
    print(json.dumps({"prep": str(Path(args.out).resolve()), "datasets": len(datasets)}, indent=2))
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    """CLI entrypoint that writes an external-tool config template."""

    out = write_config_template(args.out, fmt=args.format, overwrite=bool(args.overwrite))
    print(json.dumps({"config": str(out), "format": args.format}, indent=2))
    return 0


def cmd_row_template(args: argparse.Namespace) -> int:
    """CLI entrypoint that writes portable row-contract template files."""

    from .io.row_contract import write_template

    rows_path, metadata_path = write_template(args.rows_out, args.metadata_out, overwrite=bool(args.overwrite))
    print(json.dumps({"rows": str(rows_path), "metadata": str(metadata_path)}, indent=2))
    return 0


def cmd_row_validate(args: argparse.Namespace) -> int:
    """CLI entrypoint that validates portable row-contract files."""

    from dataclasses import asdict

    from .io.row_contract import (
        load_row_metadata,
        read_row_table,
        validate_row_contract,
        write_xds_compatibility_dir,
    )

    try:
        rows = read_row_table(args.rows)
        metadata = load_row_metadata(args.metadata)
        summary = validate_row_contract(rows, metadata)
        payload: dict[str, object] = {"status": "ok", "summary": asdict(summary)}
        if args.write_xds_dir:
            hkl_path = write_xds_compatibility_dir(rows, metadata, args.write_xds_dir, overwrite=bool(args.overwrite))
            payload["xds_compatibility"] = {"integrate_hkl": str(hkl_path)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def cmd_sfac_check(args: argparse.Namespace) -> int:
    """CLI entrypoint that checks model SFAC and endpoint wavelength readiness."""

    payload = check_sfac_contract(args.structure)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        payload["out"] = str(out.resolve())
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 2


def cmd_make_free_flags(args: argparse.Namespace) -> int:
    """Assign deterministic macromolecular FreeR flags at unique-HKL level."""

    try:
        cell, sg = infer_cell_and_spacegroup(cell_text=args.cell, space_group_text=args.space_group, model=args.model or None)
        rows = read_macromolecule_table(args.rows)
        flag_rows, summary = make_free_flags(
            rows,
            sg=sg,
            cell=cell,
            fraction=float(args.fraction),
            min_free_count=int(args.min_free_count),
            seed=int(args.seed),
            shells=int(args.shells),
            merge_anomalous=bool(args.merge_anomalous),
            free_value=int(args.free_value),
            work_value=int(args.work_value),
        )
        out = write_macromolecule_table(args.out, flag_rows)
        summary_path = Path(args.summary).expanduser() if args.summary else out.with_suffix(".summary.json")
        payload: dict[str, object] = {
            "status": "ok",
            "flags": str(out),
            "summary": asdict(summary),
        }
        if args.write_rows:
            flags = read_free_flags(out, sg=sg, merge_anomalous=bool(args.merge_anomalous))
            row_out = write_macromolecule_table(
                args.write_rows,
                propagate_free_flags(
                    rows,
                    flags,
                    sg=sg,
                    merge_anomalous=bool(args.merge_anomalous),
                    flag_column=FREE_FLAG_LABEL,
                ),
            )
            payload["rows_with_flags"] = str(row_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2) + "\n")
        payload["summary_file"] = str(summary_path.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def cmd_export_mtz(args: argparse.Namespace) -> int:
    """Export corrected GROND rows to a macromolecular MTZ with fixed FreeR flags."""

    try:
        cell, sg = infer_cell_and_spacegroup(cell_text=args.cell, space_group_text=args.space_group, model=args.model or None)
        summary = export_corrected_mtz(
            row_table=args.row_table,
            free_flags=args.free_flags,
            out_mtz=args.out_mtz,
            out_dir=args.out_dir or None,
            cell=cell,
            sg=sg,
            dataset=args.dataset,
            merge_anomalous=bool(args.merge_anomalous),
            sigma_policy=args.sigma_policy,
            q_policy=args.q_policy,
            q_off_d_A=float(args.q_off_d_A),
            q_full_d_A=float(args.q_full_d_A),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"status": "ok", "summary": asdict(summary)}, indent=2))
    return 0


def _format_protein_output_model_template(
    template: str,
    *,
    model: str | Path,
    mtz: str | Path,
    out_dir: str | Path,
    cycles: int,
    free_label: str,
) -> str:
    values = {
        "model": str(model),
        "mtz": str(mtz),
        "reflections": str(mtz),
        "out_dir": str(out_dir),
        "cycles": int(cycles),
        "free_label": str(free_label),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"protein output_model template uses unknown placeholder {exc.args[0]!r}") from exc


def cmd_protein_ls1(args: argparse.Namespace) -> int:
    """Run one configured macromolecular endpoint macrocycle against a corrected MTZ."""

    endpoint = str(args.endpoint).strip().lower().replace("-", "_")
    blocks = configured_endpoint_blocks(getattr(args, "tool_config", {}) or {})
    block = blocks.get(endpoint, {})
    command_template = (args.endpoint_command or str(block.get("command", ""))).strip()
    output_template = (args.output_model or str(block.get("output_model", ""))).strip()
    if not command_template or not output_template:
        payload = {
            "status": "error",
            "error": (
                "protein-ls1 needs --command and --output-model, or matching endpoints."
                f"{endpoint}.command/output_model entries in grond.config.json"
            ),
        }
        print(json.dumps(payload, indent=2))
        return 2
    try:
        out_dir = Path(args.out_dir).expanduser()
        output_model = _format_protein_output_model_template(
            output_template,
            model=args.model,
            mtz=args.mtz,
            out_dir=out_dir,
            cycles=int(args.cycles),
            free_label=args.free_label,
        )
        command = render_protein_endpoint_command(
            command_template,
            model=args.model,
            mtz=args.mtz,
            out_dir=out_dir,
            cycles=int(args.cycles),
            free_label=args.free_label,
        )
        payload = run_protein_endpoint(
            command=command,
            out_dir=out_dir,
            output_model=output_model,
            model=args.model,
            mtz=args.mtz,
            cycles=int(args.cycles),
            dry_run=bool(args.dry_run),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def cmd_endpoints(args: argparse.Namespace) -> int:
    """CLI entrypoint that reports configured model-update endpoint adapters."""

    payload = endpoint_report(getattr(args, "tool_config", {}) or {})
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for row in payload["endpoints"]:
        marker = "available" if row.get("available") else "missing"
        print(f"{row['name']}: {row['status']} ({marker})")
        print(f"  {row['description']}")
        resolved = ((row.get("executable") or {}).get("resolved") or "").strip()
        if resolved:
            print(f"  executable: {resolved}")
        if row.get("configured_command_template"):
            print("  command: configured from grond.config.json")
        if row.get("notes"):
            print(f"  {row['notes']}")
    return 0


def cmd_import_dials(args: argparse.Namespace) -> int:
    """CLI entrypoint for converting DIALS `.refl`/`.expt` files into GROND rows."""

    from .io.dials_bridge import convert_dials_to_grond

    bridge = convert_dials_to_grond(
        args.refl,
        args.expt,
        args.out_dir,
        intensity_mode=args.intensity_mode,
        sigma_mode=args.sigma_mode,
        sigma_scale=args.sigma_scale,
        dials_python=args.dials_python,
        min_partiality_write=args.min_partiality_write,
        keep_dont_integrate=bool(args.keep_dont_integrate) and not bool(args.drop_dont_integrate),
        drop_bad_flags=not bool(args.keep_bad_flags),
        structure=args.structure or None,
        baseline_hkl=args.baseline_hkl or None,
        dataset_name=args.dataset_name,
        prep_out=args.prep_out or None,
    )
    payload = {
        "out_dir": str(bridge.out_dir),
        "hkl": str(bridge.hkl),
        "scan_points": str(bridge.scan_points),
        "summary": str(bridge.summary),
    }
    if bridge.prep:
        payload["prep"] = str(bridge.prep)
    print(json.dumps(payload, indent=2))
    return 0


def _finalize_quick_stats(prep: PrepConfig, result: Mapping[str, object], args: argparse.Namespace) -> dict:
    """Write real-run statistics, or describe an intentionally unmaterialized dry run."""

    if bool(getattr(args, "dry_run", False)):
        return {
            "status": "dry_run",
            "out_dir": str(Path(args.out_dir).expanduser().resolve()),
            "note": (
                "Command planning completed without executing scientific stages. "
                "No endpoint or row-level statistics were materialized."
            ),
        }
    return write_quick_stats(prep, result, args)


def _write_run_completion_artifacts(
    prep: PrepConfig,
    result: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Write reports and the run summary before completion is persisted."""

    result.pop("finished", None)
    result.pop("finish_preview", None)
    quick = _finalize_quick_stats(prep, result, args)
    result["quick_stats"] = quick
    (Path(args.out_dir) / "grond_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return quick


def cmd_run(args: argparse.Namespace) -> int:
    """CLI entrypoint for the configurable `GROND run` workflow."""

    invocation_started_monotonic = time.perf_counter()
    invocation_started_at = time.time()
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logs = Path(args.out_dir) / "logs"
    python = choose_python(args.python)
    prep = prep_from_args(args)
    if len(prep.datasets) > 1 and str(getattr(args, "endpoint_calibration", "auto")) == "on":
        raise SystemExit(
            "--endpoint-calibration on cannot be forced for a multi-dataset run. "
            "Each crystal has an independent scale and sigma contract; a single concatenated replay would mix them. "
            "Leave the option at auto for the safe combined-endpoint workflow."
        )
    profile = apply_speed_profile(args, prep.structure)
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    original_structure = prep.structure
    prep.structure, structure_prep = prepare_structure_for_legacy(prep.structure, Path(args.out_dir))
    sfac_check = check_sfac_contract(prep.structure)
    sfac_check_path = Path(args.out_dir) / "sfac_check.json"
    sfac_check_path.write_text(json.dumps(sfac_check, indent=2) + "\n")
    if not sfac_check.get("ok", False):
        error_text = "; ".join(str(item.get("message", "")) for item in sfac_check.get("messages", []) if item.get("severity") == "error")
        raise SystemExit(
            "GROND model SFAC check failed. "
            + (error_text or "See sfac_check.json for details.")
            + f" Report: {sfac_check_path}"
        )
    raise_for_missing_external_tools(args, structure=prep.structure, multi_dataset=len(prep.datasets) > 1)
    configure_progress(
        logs,
        resume=bool(args.resume),
        total=estimate_run_steps(prep, args),
        invocation_started_monotonic=invocation_started_monotonic,
        invocation_started_at=invocation_started_at,
        begin_invocation=True,
    )
    prep.metadata = dict(prep.metadata or {})
    if structure_prep.get("prepared"):
        prep.metadata["original_structure"] = original_structure
        prep.metadata["structure_preparation"] = structure_prep
    write_prep(Path(args.out_dir) / "run_prep.json", prep)
    tools = external_tool_statuses(args)
    config = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": python,
        "tool_config": {
            "path": getattr(args, "tool_config_path", ""),
            "configured_tools": getattr(args, "configured_tools", {}),
            "external_tools": tools,
            "allow_missing_tools": bool(getattr(args, "allow_missing_tools", False)),
        },
        "accelerators": probe_accelerators(python),
        "mode": args.mode,
        "fast": args.fast,
        "full_fast": args.full_fast,
        "speed_profile": profile,
        "datasets": [ds.__dict__ for ds in prep.datasets],
        "structure": prep.structure,
        "original_structure": original_structure,
        "structure_preparation": structure_prep,
        "structure_sfac_check": sfac_check,
        "baseline_hkl": prep.baseline_hkl,
        "workers": args.workers,
        "chunksize": args.chunksize,
        "max_core_fraction": args.max_core_fraction,
        "n_basis": args.n_basis,
        "final_basis_scan_limit": args.final_basis_scan_limit,
        "final_basis_margin": args.final_basis_margin,
        "final_basis_min": args.final_basis_min,
        "final_basis_max": args.final_basis_max,
        "final_basis_fallback": args.final_basis_fallback,
        "prediction_cache_scope": args.prediction_cache_scope,
        "prediction_cache_compression": args.prediction_cache_compression,
        "compute_backend": args.compute_backend,
        "jax_batch_size": args.jax_batch_size,
        "jax_matrix_bucket_size": args.jax_matrix_bucket_size,
        "matrix_convention": args.matrix_convention,
        "structure_update": {
            "requested": args.structure_update,
            "resolved": resolve_structure_update_strategy(args, multi_dataset=len(prep.datasets) > 1),
            "shelxl": args.shelxl,
            "shelxl_feedback_cycles": args.shelxl_feedback_cycles,
            "shelxl_feedback_shelxl_iterations": args.shelxl_feedback_shelxl_iterations,
        },
        "post_bloch_sigma": {
            "policy": args.post_bloch_sigma_policy,
            "resolved_final_endpoint": resolved_post_bloch_sigma_policy(args, final_endpoint=True),
            "resolved_feedback_loop": resolved_post_bloch_sigma_policy(args, feedback_loop=True),
            "holdout_fraction": args.post_bloch_sigma_holdout_fraction,
            "min_scale": args.post_bloch_sigma_min_scale,
            "max_scale": args.post_bloch_sigma_max_scale,
        },
        "multi_dataset_sigma": {
            "dataset_policy": args.dataset_sigma_policy,
            "resolved_dataset_policy": resolved_dataset_sigma_policy(args, multi_dataset=len(prep.datasets) > 1),
            "dataset_holdout_fraction": args.dataset_sigma_holdout_fraction,
            "dataset_min_scale": args.dataset_sigma_min_scale,
            "dataset_max_scale": args.dataset_sigma_max_scale,
            "dataset_holdout_floor": args.dataset_sigma_holdout_floor,
            "merge_policy": args.merge_sigma_policy,
            "resolved_merge_policy": resolved_merge_sigma_policy(args, multi_dataset=len(prep.datasets) > 1),
        },
        "scale_polish": {
            "mode": args.scale_polish,
            "frame_models": args.scale_polish_frame_models,
            "adaptive_counts": args.scale_polish_adaptive_counts,
            "window_deg": args.scale_polish_window_deg,
            "detector_bins": args.scale_polish_detector_bins,
            "objective": args.scale_polish_objective,
            "tolerance": args.scale_polish_tolerance,
            "selection_metrics": args.scale_polish_selection_metrics,
            "scan_offsets": args.scale_polish_scan_offsets,
        },
        "multi_scale_contract_selection": {
            "mode": args.multi_scale_contract_selection,
            "minimum_scale_reduction": args.multi_scale_min_reduction,
            "require_bic_improvement": args.multi_scale_require_bic_improvement,
            "maximum_row_wr2_cost": args.multi_scale_max_row_wr2_cost,
            "maximum_endpoint_r1_cost": args.multi_scale_max_endpoint_r1_cost,
            "maximum_endpoint_wr2_cost": args.multi_scale_max_endpoint_wr2_cost,
            "goof_range": [args.multi_scale_goof_min, args.multi_scale_goof_max],
            "maximum_input_row_loss_fraction": args.multi_scale_max_input_row_loss_fraction,
            "minimum_bloch_wr2_gain": args.multi_scale_min_bloch_wr2_gain,
            "maximum_dataset_row_wr2_cost": args.multi_scale_max_dataset_row_wr2_cost,
            "maximum_preexisting_bloch_gain_loss": args.multi_scale_max_preexisting_bloch_gain_loss,
        },
        "resolution_limit": {
            "enabled": args.resolution_limit_gate,
            "shells": args.resolution_shells,
            "strict_split_cc": args.resolution_strict_split_cc,
            "strict_mean_i_over_sigma": args.resolution_strict_mean_i_over_sigma,
            "exploratory_split_cc": args.resolution_exploratory_split_cc,
            "exploratory_mean_i_over_sigma": args.resolution_exploratory_mean_i_over_sigma,
            "min_common_pairs": args.resolution_min_common_pairs,
            "min_shell_rows": args.resolution_min_shell_rows,
            "min_strong_fraction": args.resolution_min_strong_fraction,
            "max_shell_mwrall": args.resolution_max_shell_mwrall,
            "max_free_work_mwr_gap": args.resolution_max_free_work_mwr_gap,
            "min_free_rows": args.resolution_min_free_rows,
            "max_cumulative_mwr_penalty": args.resolution_max_cumulative_mwr_penalty,
        },
        "sg_window_scout": {
            "mode": args.sg_window_scout,
            "grid": args.sg_window_scout_grid,
            "n_basis": args.sg_window_scout_n_basis,
            "n_points": args.sg_window_scout_n_points,
            "max_rows": args.sg_window_scout_max_rows,
            "min_wR2_gain": args.sg_window_scout_min_wR2_gain,
            "free_wR2_tolerance": args.sg_window_scout_free_wR2_tolerance,
        },
        "dataset_policy": {
            "mode": args.dataset_policy,
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
        },
        "publication_endpoint_chooser": {
            "enabled": args.publication_endpoint_chooser,
            "min_hkl_safe_factor": args.publication_endpoint_min_hkl_safe_factor,
            "max_modulation_p95": args.publication_endpoint_max_modulation_p95,
            "allow_npd": args.publication_allow_npd,
        },
        "endpoint_calibration": {
            "mode": args.endpoint_calibration,
            "specs": args.endpoint_calibration_specs,
            "lambda_q": args.endpoint_calibration_lambda_q,
            "wght_modes": args.endpoint_calibration_wght_modes,
            "preferred_wght_mode": args.endpoint_calibration_prefer_wght_mode,
            "selection_policy": args.endpoint_calibration_selection_policy,
            "information_criterion": args.endpoint_calibration_information_criterion,
            "goof_min": args.endpoint_calibration_goof_min,
            "goof_max": args.endpoint_calibration_goof_max,
            "max_r1_cost": args.endpoint_calibration_max_r1_cost,
            "free_wr2_tolerance": args.endpoint_calibration_free_wr2_tolerance,
            "sigma_min_scale": args.endpoint_calibration_sigma_min_scale,
            "sigma_max_scale": args.endpoint_calibration_sigma_max_scale,
            "allow_npd": args.publication_allow_npd,
        },
    }
    (Path(args.out_dir) / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    conventional_baseline = run_fresh_conventional_baseline(prep, args, python, logs)
    result = run_single(prep, args, python, logs) if len(prep.datasets) == 1 else run_multi(prep, args, python, logs)
    result["conventional_baseline"] = conventional_baseline
    result["endpoint_calibration"] = run_endpoint_calibration(prep, result, args, python, logs)
    result["publication_endpoint_chooser"] = run_publication_endpoint_chooser(prep, result, args)
    result["publication_tables"] = run_publication_report(python, prep, result, args, logs)
    result["continue_checkpoint"] = write_continue_checkpoint(prep, result, args)
    if bool(getattr(args, "dry_run", False)):
        result["status"] = "dry_run"
    quick = _write_run_completion_artifacts(prep, result, args)
    finished = (
        _PROGRESS.mark_finished(str(result.get("status", "ok")))
        if _PROGRESS
        else _quick_progress_finish_info(Path(args.out_dir))
    )
    console = quick.get("console") if isinstance(quick.get("console"), Mapping) else {}
    payload = (
        {"status": result.get("status", "ok"), "finished": finished, **console}
        if console
        else {"status": result.get("status", "ok"), "finished": finished, "out_dir": str(Path(args.out_dir).resolve()), "quick_stats": quick}
    )
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, allow_nan=True))
    else:
        print(_format_terminal_finish_summary(payload), flush=True)
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Run the ordinary one-command GROND auto workflow."""

    args.mode = "auto"
    return cmd_run(args)


def _checkpoint_path(raw: object, *, base: Path | None = None) -> str:
    """Return an absolute path string for a checkpoint field."""

    if raw in (None, ""):
        return ""
    path = Path(str(raw)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return str(path.resolve())


def _summary_section(summary: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = summary.get(key)
    return value if isinstance(value, Mapping) else {}


def _checkpoint_dataset_contract(ds: DatasetSpec, summary_path: Path | None) -> tuple[dict[str, object], dict[str, object]]:
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
    dataset_summaries_exist = all(
        bool(row.get("source_summary") and Path(str(row["source_summary"])).exists()) for row in datasets
    ) if datasets else False
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


def _quick_percent(value: object) -> str:
    """Format a fractional statistic as a percent for quick run reports."""

    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _quick_fraction(value: object) -> float | None:
    """Return a finite fractional statistic or None."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quick_number(value: object, digits: int = 3) -> str:
    """Format a small numeric value without raising."""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _quick_count_pair(row: Mapping[str, object]) -> str:
    """Return Nobs/Nall for a row-MR metrics block."""

    nobs = row.get("Nobs", "n/a")
    nall = row.get("Nall", "n/a")
    return f"{nobs}/{nall}"


def _quick_progress_source_roots(out_dir: Path) -> list[tuple[str, int]]:
    """Return GROND source roots recorded in completed stage commands."""

    progress = try_read_json(out_dir / "grond_progress.json")
    completed = progress.get("completed", {}) if isinstance(progress.get("completed"), Mapping) else {}
    counts: dict[str, int] = {}
    marker = "/src/grond/legacy/engine/"
    for record in completed.values():
        if not isinstance(record, Mapping):
            continue
        command = record.get("command", [])
        if not isinstance(command, list):
            continue
        text = " ".join(str(part) for part in command)
        if marker not in text:
            continue
        root = text.split(marker, 1)[0].split()[-1]
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _quick_progress_timings(out_dir: Path) -> list[dict[str, object]]:
    """Return completed stage timings from the resumable progress state."""

    progress = try_read_json(out_dir / "grond_progress.json")
    completed = progress.get("completed", {}) if isinstance(progress.get("completed"), Mapping) else {}
    if not completed:
        return []
    event_order: list[str] = []
    events = progress.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping) or event.get("event") not in {"completed", "failed"}:
                continue
            label = str(event.get("label", ""))
            if label and label not in event_order:
                event_order.append(label)
    labels = [label for label in event_order if label in completed]
    labels.extend(label for label in completed if label not in labels)
    rows: list[dict[str, object]] = []
    for label in labels:
        record = completed.get(label)
        if not isinstance(record, Mapping):
            continue
        try:
            elapsed = float(record.get("elapsed_seconds", 0.0))
        except (TypeError, ValueError):
            elapsed = 0.0
        rows.append(
            {
                "stage": label,
                "returncode": record.get("returncode", "n/a"),
                "elapsed_seconds": elapsed,
                "elapsed": format_seconds(elapsed),
                "log": record.get("log", ""),
            }
        )
    return rows


def _quick_progress_finish_info(out_dir: Path) -> dict[str, object]:
    """Return a compact completion/timing block from progress state."""

    progress = try_read_json(out_dir / "grond_progress.json")
    finished = progress.get("finished") if isinstance(progress.get("finished"), Mapping) else {}
    completed = progress.get("completed", {}) if isinstance(progress.get("completed"), Mapping) else {}
    completed_ok = len([v for v in completed.values() if isinstance(v, Mapping) and v.get("returncode") == 0])
    failed = len([v for v in completed.values() if isinstance(v, Mapping) and v.get("returncode") not in {0, None}])
    total_elapsed = sum(
        _quick_float(v.get("elapsed_seconds"), 0.0)
        for v in completed.values()
        if isinstance(v, Mapping) and v.get("returncode") == 0
    )
    out = dict(finished)
    out.update(
        {
            "completed_stages": completed_ok,
            "failed_stages": failed,
            "cumulative_stage_elapsed_seconds": total_elapsed,
            "stage_elapsed_seconds": total_elapsed,
            "cumulative_stage_elapsed": format_seconds(total_elapsed),
            "stage_elapsed": format_seconds(total_elapsed),
        }
    )
    if finished:
        if out.get("completed_this_invocation") is not None:
            try:
                completed_here = max(0, min(completed_ok, int(out["completed_this_invocation"])))
            except (TypeError, ValueError):
                completed_here = 0
            out["completed_this_invocation"] = completed_here
            out["reused_completed_stages"] = max(0, completed_ok - completed_here)
        if out.get("invocation_elapsed_seconds") is not None:
            out["invocation_elapsed"] = format_seconds(out.get("invocation_elapsed_seconds"))
    return out


def _quick_final_run(result: Mapping[str, object]) -> Path | None:
    """Return the final row-level run directory when one was recorded."""

    primary = result.get("primary_run")
    if isinstance(primary, str) and primary:
        return Path(primary)
    combined = result.get("combined")
    if isinstance(combined, str) and combined:
        return Path(combined)
    return None


def _quick_load_run_summary(run_dir: Path | None) -> Mapping[str, object]:
    """Load a single-dataset or combined multi-dataset row summary."""

    if run_dir is None:
        return {}
    summary = try_read_json(run_dir / "summary.json")
    if summary:
        return summary
    combined_summary = try_read_json(run_dir / "combined_summary.json")
    return combined_summary if isinstance(combined_summary, Mapping) else {}


def _quick_combined_native_metrics(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return single-summary-shaped native metrics for a combined run."""

    native_joint = summary.get("native_joint") if isinstance(summary.get("native_joint"), Mapping) else {}
    dyn = native_joint.get("dyn") if isinstance(native_joint.get("dyn"), Mapping) else {}
    born = native_joint.get("born") if isinstance(native_joint.get("born"), Mapping) else {}
    if not dyn and not born:
        return {}
    metrics: dict[str, dict[str, object]] = {}
    for split in ("all", "work", "free"):
        metrics[split] = {}
        if isinstance(dyn.get(split), Mapping):
            metrics[split]["dyn"] = dyn.get(split)
        if isinstance(born.get(split), Mapping):
            metrics[split]["born"] = born.get(split)
    return metrics


def _quick_best_native(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return the best native block from a native row run summary."""

    value = summary.get("best_native")
    if isinstance(value, Mapping):
        return value
    metrics = _quick_combined_native_metrics(summary)
    if not metrics:
        return {}
    wght = summary.get("wght") if isinstance(summary.get("wght"), Mapping) else {}
    nuisance = {"native_weight_mode": wght.get("mode", "per_dataset_native")}
    return {
        "metrics": metrics,
        "native_weight": wght,
        "nuisance": nuisance,
    }


def _quick_best_endpoint(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return the best corrected-HKL endpoint block from a native row run summary."""

    value = summary.get("best_endpoint")
    if isinstance(value, Mapping):
        return value
    combined_endpoint = summary.get("combined_endpoint")
    if isinstance(combined_endpoint, Mapping):
        shelxl = combined_endpoint.get("shelxl")
        return {"shelxl": shelxl} if isinstance(shelxl, Mapping) else {}
    return {}


def _quick_nuisance(summary: Mapping[str, object]) -> Mapping[str, object]:
    """Return the final nuisance parameter block for quick reporting."""

    best_native = _quick_best_native(summary)
    nuisance = best_native.get("nuisance")
    if isinstance(nuisance, Mapping):
        return nuisance
    config = summary.get("config")
    return config if isinstance(config, Mapping) else {}


def _quick_dataset_nuisance_contracts(summary: Mapping[str, object]) -> list[dict[str, object]]:
    """Return exact per-dataset nuisance blocks from a combined summary."""

    runs = summary.get("runs")
    if not isinstance(runs, list):
        return []
    contracts: list[dict[str, object]] = []
    for raw in runs:
        if not isinstance(raw, Mapping):
            continue
        best = raw.get("best_native_summary")
        best = best if isinstance(best, Mapping) else {}
        nuisance = best.get("nuisance")
        if not isinstance(nuisance, Mapping) or not nuisance:
            continue
        contracts.append(
            {
                "dataset": str(raw.get("name") or f"dataset_{len(contracts) + 1}"),
                "run_dir": str(raw.get("run_dir") or ""),
                "nuisance": dict(nuisance),
                "scale_count": best.get("scale_count"),
                "native_weight": dict(best.get("native_weight") or {})
                if isinstance(best.get("native_weight"), Mapping)
                else {},
            }
        )
    return contracts


def _quick_metric(metrics: Mapping[str, object], split: str, branch: str, key: str) -> object:
    """Read one nested native metric without exposing KeyError to report writing."""

    split_block = metrics.get(split)
    if not isinstance(split_block, Mapping):
        return None
    branch_block = split_block.get(branch)
    if not isinstance(branch_block, Mapping):
        return None
    return branch_block.get(key)


def _quick_dyn_metric(summary: Mapping[str, object], split: str, key: str) -> object:
    """Read one dynamical row-space metric from a native-stage summary."""

    best_native = _quick_best_native(summary)
    metrics = best_native.get("metrics") if isinstance(best_native.get("metrics"), Mapping) else {}
    return _quick_metric(metrics, split, "dyn", key)


def _quick_born_metric(summary: Mapping[str, object], split: str, key: str) -> object:
    """Read one matched Born row-space metric from a native-stage summary."""

    best_native = _quick_best_native(summary)
    metrics = best_native.get("metrics") if isinstance(best_native.get("metrics"), Mapping) else {}
    return _quick_metric(metrics, split, "born", key)


def _quick_float(value: object, default: float = float("nan")) -> float:
    """Return a finite float or a default."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _quick_int(value: object) -> int | None:
    """Return an int when a report value is integer-like."""

    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


_QUICK_SHELX_FLOAT_TOKEN_RE = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][-+]?\d+)?")


def _quick_parse_wght_terms(line: str) -> list[float]:
    """Parse numeric terms from one SHELX WGHT line without crossing lines."""

    parts = line.strip().split()
    if not parts or parts[0].upper() != "WGHT":
        return []
    terms: list[float] = []
    for token in parts[1:]:
        if not _QUICK_SHELX_FLOAT_TOKEN_RE.fullmatch(token):
            break
        terms.append(float(token))
    return terms


def _quick_iter_recommended_wght_terms(text: str):
    for line in text.splitlines():
        upper = line.upper()
        if "RECOMMENDED WEIGHTING SCHEME" not in upper or "WGHT" not in upper:
            continue
        terms = _quick_parse_wght_terms(line[upper.find("WGHT") :])
        if terms:
            yield terms


def _quick_lst_diagnostics(lst_path: Path | None) -> dict[str, object]:
    """Parse the SHELXL LST fields users need for endpoint sanity checks."""

    if lst_path is None:
        return {"lst": "", "exists": False, "pathology_status": "missing", "safe_for_feedback": False}
    out: dict[str, object] = {
        "lst": str(lst_path),
        "exists": lst_path.exists(),
        "pathology_status": "missing",
        "safe_for_feedback": False,
    }
    if not lst_path.exists():
        return out
    try:
        text = lst_path.read_text(errors="replace")
    except OSError as exc:
        out.update({"exists": False, "error": str(exc)})
        return out

    matches = re.findall(r"wR2 =\s*([0-9.]+), GooF = S =\s*([0-9.]+).* for all data", text)
    if matches:
        wr2, goof = matches[-1]
        out["wR2_all"] = float(wr2)
        out["GooF"] = float(goof)
    matches = re.findall(r"\+\s+wR2:\s*([0-9.]+),\s*GooF:\s*([0-9.]+)", text)
    if matches:
        wr2, goof = matches[-1]
        out["wR2_all"] = float(wr2)
        out["GooF"] = float(goof)
    matches = re.findall(r"R1 =\s*([0-9.]+) for\s+(\d+) Fo > 4sig\(Fo\) and\s+([0-9.]+) for all\s+(\d+) data", text)
    if matches:
        r1_gt, n_gt, r1_all, n_all = matches[-1]
        out["R1_gt"] = float(r1_gt)
        out["N_gt"] = int(n_gt)
        out["R1_all"] = float(r1_all)
        out["N_all"] = int(n_all)
    matches_gt = re.findall(r"\+\s+R1:\s*([0-9.]+) for\s+(\d+) reflections", text)
    matches_all = re.findall(r"\+\s+R1 \(all\):\s*([0-9.]+) for\s+(\d+) reflections", text)
    if matches_gt and matches_all:
        r1_gt, n_gt = matches_gt[-1]
        r1_all, n_all = matches_all[-1]
        out["R1_gt"] = float(r1_gt)
        out["N_gt"] = int(n_gt)
        out["R1_all"] = float(r1_all)
        out["N_all"] = int(n_all)
    matches = re.findall(r"R\(int\) =\s*([0-9.]+)\s+R\(sigma\) =\s*([0-9.]+)", text)
    if matches:
        rint, rsigma = matches[-1]
        out["Rint"] = float(rint)
        out["Rsigma"] = float(rsigma)
    recommended_wght_terms = list(_quick_iter_recommended_wght_terms(text))
    if recommended_wght_terms:
        terms = recommended_wght_terms[-1]
        out["recommended_wght_a"] = terms[0]
        out["recommended_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
        out["recommended_wght_terms"] = terms
    for line in text.splitlines():
        terms = _quick_parse_wght_terms(line)
        if terms:
            out["active_wght_a"] = terms[0]
            out["active_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
            out["active_wght_terms"] = terms
            break
    matches = re.findall(r"Total number of l\.s\. parameters =\s+(\d+)", text)
    if matches:
        out["parameters"] = int(matches[-1])
    matches = re.findall(r"Restrained GooF =\s+[0-9.]+ for\s+(\d+) restraints", text)
    if matches:
        out["restraints"] = int(matches[-1])
    matches = re.findall(r"\+\++\s*(\d+)\s+Constraints\s+\|\s+(\d+)\s+Restraints\s+\|\s+(\d+)\s+Parameters", text)
    if matches:
        _constraints, restraints, parameters = matches[-1]
        out["parameters"] = int(parameters)
        out["restraints"] = int(restraints)
    npd_atoms = sorted(set(re.findall(r"\s([A-Z][A-Z0-9]{1,7})\s+\*\* NON POSITIVE DEFINITE \*\*", text)))
    npd_matches = re.findall(r"(\d+)\s+atoms NPD", text)
    if npd_matches or npd_atoms:
        out["npd_atoms"] = npd_atoms
        out["npd_count"] = max(int(npd_matches[-1]) if npd_matches else 0, len(npd_atoms))
        out["has_npd"] = int(out["npd_count"]) > 0
    split_matches = re.findall(r"(\d+)\s+atoms may be split", text)
    if split_matches:
        out["split_count"] = int(split_matches[-1])
        out["has_split_warnings"] = int(out["split_count"]) > 0
    pathology = parse_shelx_lst_pathologies(lst_path)
    out["pathology_status"] = pathology.get("pathology_status", "missing")
    out["safe_for_feedback"] = bool(
        pathology.get("pathology_status") == "complete" and not pathology.get("has_npd", False)
    )
    for key in ("npd_count", "split_count"):
        parsed_count = _quick_int(pathology.get(key))
        current_count = _quick_int(out.get(key))
        if parsed_count is not None:
            out[key] = parsed_count if current_count is None else max(parsed_count, current_count)
    parsed_atoms = pathology.get("npd_atoms") if isinstance(pathology.get("npd_atoms"), list) else []
    current_atoms = out.get("npd_atoms") if isinstance(out.get("npd_atoms"), list) else []
    if parsed_atoms or current_atoms:
        out["npd_atoms"] = sorted({str(atom) for atom in [*parsed_atoms, *current_atoms]})
    out["has_npd"] = bool(out.get("has_npd", False) or pathology.get("has_npd", False))
    out["has_split_warnings"] = bool(
        out.get("has_split_warnings", False) or pathology.get("has_split_warnings", False)
    )
    return out


def _quick_shelxl_details(source: Mapping[str, object] | None = None, lst_path: Path | None = None) -> dict[str, object]:
    """Merge recorded SHELXL stats with LST-derived health diagnostics."""

    out: dict[str, object] = dict(source or {})
    if lst_path is None:
        text = str(out.get("lst") or out.get("endpoint_lst") or "")
        lst_path = Path(text) if text else None
    diagnostics = _quick_lst_diagnostics(lst_path)
    out.setdefault("lst", diagnostics.get("lst", ""))
    out["lst_exists"] = bool(diagnostics.get("exists"))
    out["pathology_status"] = diagnostics.get("pathology_status", "missing")
    out["lst_safe_for_feedback"] = bool(diagnostics.get("safe_for_feedback", False))
    pathology_count_keys = {"npd_count", "split_count"}
    pathology_flag_keys = {"has_npd", "has_split_warnings"}
    for key in (
        "R1_gt",
        "R1_all",
        "wR2_all",
        "GooF",
        "N_gt",
        "N_all",
        "Rint",
        "Rsigma",
        "parameters",
        "restraints",
        "npd_count",
        "npd_atoms",
        "split_count",
        "has_npd",
        "has_split_warnings",
        "active_wght_a",
        "active_wght_b",
        "recommended_wght_a",
        "recommended_wght_b",
    ):
        old_value = out.get(key)
        if key not in diagnostics:
            continue
        if key in pathology_count_keys:
            old_count = _quick_int(old_value)
            new_count = _quick_int(diagnostics[key])
            if old_count is None:
                out[key] = diagnostics[key]
            elif new_count is not None:
                # A later, un-hash-bound LST may add a warning, but it must
                # never erase pathology recorded by the endpoint producer.
                out[key] = max(old_count, new_count)
        elif key == "npd_atoms":
            old_atoms = old_value if isinstance(old_value, (list, tuple, set)) else []
            new_atoms = diagnostics[key] if isinstance(diagnostics[key], (list, tuple, set)) else []
            out[key] = sorted({str(atom) for atom in [*old_atoms, *new_atoms] if str(atom).strip()})
        elif key in pathology_flag_keys:
            out[key] = bool(old_value) or bool(diagnostics[key])
        elif old_value is None or old_value == "":
            # Recorded summary metrics are the coherent endpoint result.  An
            # arbitrary file that merely occupies the LST path is only a
            # fallback source; without an expected hash it cannot replace
            # already recorded R values, weights, or parameter counts.
            out[key] = diagnostics[key]
    return out


def _quick_selected_feedback_shelxl(
    result: Mapping[str, object],
    final_model: Path | None,
    fallback: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load one internally coherent adopted-feedback SHELXL endpoint.

    Multi-dataset auto can finish with a later diagnostic combined endpoint
    after the shared feedback loop has already retained an earlier clean
    model.  Metrics from that later endpoint must never be merged with the LST
    or pathology status of the retained model.
    """

    if final_model is None:
        return _quick_shelxl_details(fallback or {}, None)
    try:
        model = final_model.expanduser().resolve()
    except OSError:
        model = final_model.expanduser()
    lst = model.with_suffix(".lst")
    source: Mapping[str, object] = {}
    feedback = result.get("structure_feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    cycles = feedback.get("cycles") if isinstance(feedback.get("cycles"), list) else []
    matched_feedback_record = False
    for raw in cycles:
        if not isinstance(raw, Mapping) or raw.get("adopted_for_feedback") is not True:
            continue
        output_text = str(raw.get("output_model") or "").strip()
        if not output_text:
            continue
        try:
            output = Path(output_text).expanduser().resolve()
        except OSError:
            output = Path(output_text).expanduser()
        if output != model:
            continue
        matched_feedback_record = True
        combined_text = str(raw.get("combined") or "").strip()
        combined_summary = try_read_json(Path(combined_text) / "combined_summary.json") if combined_text else {}
        endpoint = combined_summary.get("combined_endpoint") if isinstance(combined_summary, Mapping) else {}
        endpoint = endpoint if isinstance(endpoint, Mapping) else {}
        shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
        artifacts = endpoint.get("artifacts") if isinstance(endpoint.get("artifacts"), Mapping) else {}
        health = shelxl.get("res_health") if isinstance(shelxl.get("res_health"), Mapping) else {}
        artifact_path = str(artifacts.get("res") or "").strip()
        health_path = str(health.get("path") or "").strip()
        artifact_sha = str(artifacts.get("res_sha256") or "").strip()
        health_sha = str(health.get("sha256") or "").strip()
        try:
            paths_match = bool(artifact_path and health_path) and Path(artifact_path).resolve() == model and Path(health_path).resolve() == model
            current_sha = sha256_file(model)
        except OSError:
            paths_match = False
            current_sha = ""
        if paths_match and artifact_sha and health_sha and artifact_sha == health_sha == current_sha:
            source = shelxl
        break
    if source:
        return _quick_shelxl_details(source, lst)
    if lst.exists():
        # The exact LST is still safer than combining statistics recorded for a
        # different endpoint with this model's pathology record.
        return _quick_shelxl_details({}, lst)
    if matched_feedback_record:
        return _quick_shelxl_details({}, None)
    return _quick_shelxl_details(fallback or {}, None)


def _quick_baseline_shelxl(
    prep: PrepConfig,
    final_summary: Mapping[str, object] | None = None,
    result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prefer the hash-bound supplied-input replay over historical fallbacks."""

    structure = Path(prep.structure).expanduser() if prep.structure else None
    hkl = ""
    if prep.datasets:
        try:
            hkl = dataset_baseline_hkl(prep, prep.datasets[0])
        except Exception:
            hkl = prep.baseline_hkl
    elif prep.baseline_hkl:
        hkl = prep.baseline_hkl
    supplied_fresh = _validated_fresh_conventional_baseline(result or {})
    if supplied_fresh:
        lst_text = str(supplied_fresh.get("lst") or "")
        details = _quick_shelxl_details(supplied_fresh, Path(lst_text) if lst_text else None)
        details["res"] = str(supplied_fresh.get("res") or structure or "")
        details["hkl"] = str(supplied_fresh.get("hkl") or hkl or "")
        details["source_res"] = str(supplied_fresh.get("source_res") or structure or "")
        details["source_hkl"] = str(supplied_fresh.get("source_hkl") or hkl or "")
        details["baseline_source"] = "fresh_supplied_input_conventional_control"
        details["fresh_rerun"] = True
        return details
    fresh = _fresh_baseline_shelxl(final_summary or {})
    if fresh:
        lst_text = str(fresh.get("lst") or "")
        details = _quick_shelxl_details(fresh, Path(lst_text) if lst_text else None)
        details["res"] = str(fresh.get("res") or structure or "")
        details["hkl"] = str(hkl or fresh.get("hkl") or "")
        details["baseline_source"] = "final_run_summary.baseline_exti"
        details["fresh_rerun"] = True
        return details

    lst = structure.with_suffix(".lst") if structure is not None else None
    if lst is not None and not lst.exists():
        log = structure.with_suffix(".log") if structure is not None else None
        if log is not None and log.exists():
            lst = log
    details = _quick_shelxl_details({}, lst)
    details["res"] = str(structure or "")
    details["hkl"] = str(hkl or "")
    details["baseline_source"] = "source_sibling_lst_or_log"
    details["fresh_rerun"] = False
    return details


def _quick_shelxl_issue_text(details: Mapping[str, object]) -> str:
    """Format NPD/split problems in one compact cell."""

    pathology_status = str(details.get("pathology_status") or "unverified")
    npd = _quick_int(details.get("npd_count"))
    split = _quick_int(details.get("split_count"))
    atoms = details.get("npd_atoms")
    atom_text = ""
    if isinstance(atoms, list) and atoms:
        shown = ", ".join(str(v) for v in atoms[:5])
        atom_text = f" ({shown}{', ...' if len(atoms) > 5 else ''})"
    return (
        f"LST pathology `{pathology_status}`; "
        f"NPD `{npd if npd is not None else 'n/a'}`{atom_text}; "
        f"split `{split if split is not None else 'n/a'}`"
    )


def _quick_shelxl_weight_text(details: Mapping[str, object]) -> str:
    """Format active and recommended SHELXL weights."""

    active_a = details.get("active_wght_a")
    active_b = details.get("active_wght_b")
    rec_a = details.get("recommended_wght_a")
    rec_b = details.get("recommended_wght_b")
    active = f"{_quick_number(active_a, 4)} {_quick_number(active_b, 4)}" if active_a is not None else "n/a"
    recommended = f"{_quick_number(rec_a, 4)} {_quick_number(rec_b, 4)}" if rec_a is not None else "n/a"
    return f"active `{active}`; recommended `{recommended}`"


def _quick_endpoint_scale_text(endpoint: Mapping[str, object]) -> str:
    """Return a readable endpoint scale/source line without blank parameter text."""

    spec = str(endpoint.get("scale_spec") or endpoint.get("spec") or endpoint.get("source_kind") or "n/a")
    params = endpoint.get("scale_params")
    if params in {None, "", "n/a"}:
        return f"{spec}; source {endpoint.get('source', endpoint.get('selection_source', 'n/a'))}/{endpoint.get('source_kind', 'n/a')}"
    return f"{spec} ({params} parameters)"


def _quick_goof_warning(goof_value: object) -> str:
    """Return a short endpoint-calibration warning for non-boring GooF values."""

    goof = _quick_float(goof_value)
    if not math.isfinite(goof):
        return ""
    if 0.85 <= goof <= 1.15:
        return ""
    return "endpoint GooF is outside 0.85-1.15; treat this endpoint as diagnostic unless the row-level and weighting audits justify it"


def _quick_residual_z_summary(csv_paths: Path | list[Path]) -> dict[str, dict[str, float | int]]:
    """Summarize residual z values across one or more frozen native tables."""

    rows: dict[str, list[float]] = {"all": [], "work": [], "free": []}
    paths = [csv_paths] if isinstance(csv_paths, Path) else list(csv_paths)
    for csv_path in paths:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    z = _quick_float(row.get("dyn_residual_over_sigma"))
                    if not math.isfinite(z):
                        continue
                    rows["all"].append(z)
                    split = str(row.get("split", "")).lower()
                    if split in {"work", "free"}:
                        rows[split].append(z)
        except OSError:
            continue

    out: dict[str, dict[str, float | int]] = {}
    for split, values in rows.items():
        if not values:
            continue
        abs_values = sorted(abs(v) for v in values)
        values_sorted = sorted(values)
        n = len(values)
        out[split] = {
            "n": n,
            "mean_z": float(sum(values) / n),
            "rms_z": float(math.sqrt(sum(v * v for v in values) / n)),
            "median_z": float(values_sorted[n // 2] if n % 2 else 0.5 * (values_sorted[n // 2 - 1] + values_sorted[n // 2])),
            "median_abs_z": float(abs_values[n // 2] if n % 2 else 0.5 * (abs_values[n // 2 - 1] + abs_values[n // 2])),
        }
    return out


def _quick_quantile(sorted_values: list[float], q: float) -> float:
    """Return a linear-interpolated quantile from sorted finite values."""

    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float((1.0 - frac) * sorted_values[lo] + frac * sorted_values[hi])


def _quick_numeric_distribution(values: list[float]) -> dict[str, float | int]:
    """Return compact finite-value distribution statistics."""

    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "min": float(finite[0]),
        "p05": _quick_quantile(finite, 0.05),
        "median": _quick_quantile(finite, 0.50),
        "mean": float(sum(finite) / len(finite)),
        "p95": _quick_quantile(finite, 0.95),
        "max": float(finite[-1]),
    }


def _quick_csv_numeric_audit(csv_path: Path, columns: list[str]) -> dict[str, object]:
    """Summarize numeric row-table columns without pandas."""

    out: dict[str, object] = {
        "path": str(csv_path),
        "exists": csv_path.exists(),
        "rows": 0,
        "fieldnames": [],
        "columns": {},
        "split_counts": {},
        "group_count": 0,
        "detector_bin_count": 0,
        "sg_window_clipped_count": 0,
    }
    if not csv_path.exists():
        return out
    values: dict[str, list[float]] = {column: [] for column in columns}
    split_counts: dict[str, int] = {}
    groups: set[str] = set()
    detector_bins: set[str] = set()
    clipped = 0
    try:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            out["fieldnames"] = list(reader.fieldnames or [])
            for row in reader:
                out["rows"] = int(out["rows"]) + 1
                split = str(row.get("split") or "unknown")
                split_counts[split] = split_counts.get(split, 0) + 1
                if row.get("group") not in {None, ""}:
                    groups.add(str(row.get("group")))
                if row.get("detector_bin") not in {None, ""}:
                    detector_bins.add(str(row.get("detector_bin")))
                if str(row.get("sg_window_clipped", "")).strip() in {"1", "true", "True"}:
                    clipped += 1
                for column in columns:
                    value = _quick_float(row.get(column))
                    if math.isfinite(value):
                        values[column].append(value)
    except OSError as exc:
        out["error"] = str(exc)
        return out
    out["split_counts"] = split_counts
    out["group_count"] = len(groups)
    out["detector_bin_count"] = len(detector_bins)
    out["sg_window_clipped_count"] = clipped
    out["columns"] = {column: _quick_numeric_distribution(vals) for column, vals in values.items()}
    return out


def _quick_distribution_row(summary: Mapping[str, object], label: str, units: str = "") -> str:
    """Format one numeric-distribution row for a Markdown table."""

    if int(summary.get("n", 0) or 0) <= 0:
        return f"| {label} | {units} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {units} | {summary.get('n', 'n/a')} | "
        f"{_quick_number(summary.get('min'), 6)} | {_quick_number(summary.get('p05'), 6)} | "
        f"{_quick_number(summary.get('median'), 6)} | {_quick_number(summary.get('mean'), 6)} | "
        f"{_quick_number(summary.get('p95'), 6)} | {_quick_number(summary.get('max'), 6)} |"
    )


def _write_bloch_diagnostics_report(
    out_dir: Path,
    final_run: Path | None,
    final_summary: Mapping[str, object],
    row_metrics: Mapping[str, object],
    file_block: Mapping[str, str],
    *,
    row_table_override: Path | None = None,
    report_name: str = "BLOCH_DIAGNOSTICS.md",
) -> Path:
    """Write a human-readable Bloch-matrix/row-contract audit report."""

    report = out_dir / report_name
    row_table = (
        row_table_override
        if row_table_override is not None
        else final_run / "native_dyn_calc.csv"
        if final_run
        else Path(str(file_block.get("publication_row_table") or ""))
    )
    config = final_summary.get("config") if isinstance(final_summary.get("config"), Mapping) else {}
    best_native = _quick_best_native(final_summary)
    nuisance = _quick_nuisance(final_summary)
    native_weight = best_native.get("native_weight") if isinstance(best_native.get("native_weight"), Mapping) else {}
    basis_stats = best_native.get("basis_size_stats") if isinstance(best_native.get("basis_size_stats"), Mapping) else {}
    row_selection = final_summary.get("row_selection") if isinstance(final_summary.get("row_selection"), Mapping) else {}
    filters = final_summary.get("filters") if isinstance(final_summary.get("filters"), Mapping) else {}
    diagnostics = final_summary.get("diagnostics") if isinstance(final_summary.get("diagnostics"), Mapping) else {}
    kernel = final_summary.get("bloch_kernel_contract") if isinstance(final_summary.get("bloch_kernel_contract"), Mapping) else {}
    metadata = row_metrics.get("metadata", {}) if isinstance(row_metrics.get("metadata"), Mapping) else {}
    columns = [
        "d_A",
        "basis_size",
        "q_dyn_over_born",
        "center_sg_inv_A",
        "max_sg_solve_residual",
        "sg_window_inv_A",
        "sg_window_raw_inv_A",
        "sg_window_half_frames",
        "sg_dSg_domega_inv_A_per_deg",
        "sg_dSg_dframe_inv_A",
        "sg_jitter_sigma_inv_A",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "omega_bragg_minus_center_deg",
        "row_resolution_factor",
        "dyn_residual_over_sigma",
        "born_residual_over_sigma",
        "sigma_eff",
        "peak",
        "corr",
        "rlp",
    ]
    audit = _quick_csv_numeric_audit(row_table, columns)
    column_stats = audit.get("columns") if isinstance(audit.get("columns"), Mapping) else {}
    basis_summary = column_stats.get("basis_size") if isinstance(column_stats.get("basis_size"), Mapping) else {}
    basis_median = _quick_float(basis_summary.get("median", basis_stats.get("median")))
    basis_p95 = _quick_float(basis_summary.get("p95", basis_stats.get("p95")))
    basis_max = _quick_float(basis_summary.get("max", basis_stats.get("max")))
    bethe_strong = _quick_float(config.get("bethe_strong_count"), 0.0)
    if not math.isfinite(bethe_strong) or bethe_strong <= 0.0:
        bethe_strong = _quick_float(kernel.get("bethe_strong_count"), 0.0)

    def strong_weak_text(basis_value: float) -> str:
        if not math.isfinite(basis_value) or basis_value <= 0 or not math.isfinite(bethe_strong) or bethe_strong <= 0:
            return "n/a"
        strong = min(int(round(bethe_strong)), int(round(basis_value)))
        weak = max(0, int(round(basis_value)) - strong)
        return f"{strong} strong / {weak} weak"

    row_plan = diagnostics.get("row_plan_cache") if isinstance(diagnostics.get("row_plan_cache"), Mapping) else {}
    ug_cache = diagnostics.get("ug_cache") if isinstance(diagnostics.get("ug_cache"), Mapping) else {}
    geometry = row_selection.get("geometry_filter") if isinstance(row_selection.get("geometry_filter"), Mapping) else {}
    row_count = audit.get("rows") or row_selection.get("selected_rows") or filters.get("selected_rows") or metadata.get("row_count", "n/a")
    lines = [
        "# GROND Bloch Diagnostics",
        "",
        "This report audits the forward Bloch calculation and row contract used by the final GROND row table. It is meant for later method review, not for endpoint promotion by itself.",
        "",
        "## Inputs",
        "",
        "| item | value |",
        "|---|---|",
        f"| final row table | `{row_table}` |",
        f"| row table exists | `{audit.get('exists')}` |",
        f"| rows in audit table | `{row_count}` |",
        f"| split counts | `{audit.get('split_counts', {})}` |",
        f"| scale groups / detector bins | `{audit.get('group_count', 'n/a')}` / `{audit.get('detector_bin_count', 'n/a')}` |",
        f"| final model | `{file_block.get('final_feedback_res', '')}` |",
        "",
        "## Bloch Kernel Contract",
        "",
        "| item | value |",
        "|---|---|",
        f"| backend | `{config.get('compute_backend', kernel.get('compute_backend', 'n/a'))}` |",
        f"| N / Q | `{config.get('n_basis', metadata.get('n_basis', 'n/a'))}` / `{config.get('n_points', metadata.get('n_points', 'n/a'))}` |",
        f"| matrix convention | `{config.get('matrix_convention', nuisance.get('matrix_convention', 'n/a'))}` |",
        f"| profile quadrature / accumulation | `{config.get('profile_quadrature', 'n/a')}` / `{config.get('profile_accumulation', 'n/a')}` |",
        f"| max support Sg | `{_quick_number(config.get('max_support_sg'), 6)}` 1/A |",
        f"| top-up dmin | `{_quick_number(config.get('topup_d_min'), 3)}` A |",
        f"| row-plan cache | `{row_plan.get('status', row_plan or config.get('row_plan_cache', 'n/a'))}`; unique diffs `{diagnostics.get('row_plan_cache_unique_diffs', 0)}` |",
        f"| U_g cache | `{ug_cache.get('status', ug_cache or 'n/a')}` |",
        "",
        "## Bethe Approximation",
        "",
        "| item | value |",
        "|---|---|",
        f"| backend requested | `{config.get('compute_backend', 'n/a')}` |",
        f"| strong-beam request | `{config.get('bethe_strong_count', 'n/a')}` |",
        f"| variant / promote denom / eps | `{config.get('bethe_variant', 'n/a')}` / `{config.get('bethe_promote_denom', 'n/a')}` / `{config.get('bethe_denom_eps', 'n/a')}` |",
        f"| median basis strong/weak | `{strong_weak_text(basis_median)}` |",
        f"| p95 basis strong/weak | `{strong_weak_text(basis_p95)}` |",
        f"| max basis strong/weak | `{strong_weak_text(basis_max)}` |",
        "",
        "If the backend is not `scipy_bethe`, the strong/weak counts are only the configured Bethe diagnostic capacity and no Schur reduction was used for the final prediction.",
        "",
        "## Sg, Rocking, And Quadrature",
        "",
        "| quantity | units | n | min | p05 | median | mean | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _quick_distribution_row(column_stats.get("center_sg_inv_A", {}), "center excitation error Sg", "1/A"),
        _quick_distribution_row(column_stats.get("max_sg_solve_residual", {}), "max target-Sg solve residual", "1/A"),
        _quick_distribution_row(column_stats.get("sg_window_inv_A", {}), "accepted Sg window", "1/A"),
        _quick_distribution_row(column_stats.get("sg_window_raw_inv_A", {}), "raw Sg window before clipping", "1/A"),
        _quick_distribution_row(column_stats.get("sg_window_half_frames", {}), "Sg window half frames", "frames"),
        _quick_distribution_row(column_stats.get("sg_dSg_domega_inv_A_per_deg", {}), "dSg/domega", "1/A/deg"),
        _quick_distribution_row(column_stats.get("sg_dSg_dframe_inv_A", {}), "dSg/dframe", "1/A/frame"),
        _quick_distribution_row(column_stats.get("sg_jitter_sigma_inv_A", {}), "Sg jitter sigma", "1/A"),
        _quick_distribution_row(column_stats.get("sg_profile_points_total", {}), "profile points after jitter", "points"),
        _quick_distribution_row(column_stats.get("sg_profile_weight_sum", {}), "profile weight sum", "weight"),
        _quick_distribution_row(column_stats.get("omega_bragg_minus_center_deg", {}), "omega Bragg minus row center", "deg"),
        f"| Sg-window clipped rows | count | {audit.get('sg_window_clipped_count', 0)} | n/a | n/a | n/a | n/a | n/a | n/a |",
        "",
        "## Matrix Size And Dynamical Strength",
        "",
        "| quantity | units | n | min | p05 | median | mean | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _quick_distribution_row(column_stats.get("basis_size", {}), "basis size", "beams"),
        _quick_distribution_row(column_stats.get("q_dyn_over_born", {}), "q = Idyn/Iborn", "ratio"),
        _quick_distribution_row(column_stats.get("row_resolution_factor", {}), "row resolution factor", "ratio"),
        _quick_distribution_row(column_stats.get("dyn_residual_over_sigma", {}), "dynamical residual/sigma", "sigma"),
        _quick_distribution_row(column_stats.get("born_residual_over_sigma", {}), "Born residual/sigma", "sigma"),
        "",
        "## Row Contract And Geometry Proxies",
        "",
        "| quantity | units | n | min | p05 | median | mean | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _quick_distribution_row(column_stats.get("d_A", {}), "d spacing", "A"),
        _quick_distribution_row(column_stats.get("sigma_eff", {}), "effective sigma", "I units"),
        _quick_distribution_row(column_stats.get("peak", {}), "PEAK", "XDS percent"),
        _quick_distribution_row(column_stats.get("corr", {}), "CORR", "XDS percent"),
        _quick_distribution_row(column_stats.get("rlp", {}), "RLP", "XDS units"),
        "",
        "## Geometry And Nuisance Settings",
        "",
        "| item | value |",
        "|---|---|",
        f"| row selection min d / min peak | `{config.get('min_d_spacing', 'n/a')}` A / `{config.get('min_peak', 'n/a')}` |",
        f"| geometry percentile gate | `{config.get('geometry_percentile', 'n/a')}` |",
        f"| geometry filter summary | `{geometry}` |",
        f"| thickness model | `{nuisance.get('thickness_model', 'n/a')}`; center `{_quick_number(nuisance.get('thickness_A', nuisance.get('thickness_grid')), 3)}` A; gradient `{_quick_number(nuisance.get('thickness_grad_frac', nuisance.get('thickness_grad_grid')), 6)}` |",
        f"| Sg jitter | `{nuisance.get('sg_jitter_mode', 'n/a')}`; sigma0 `{_quick_number(nuisance.get('sg_jitter_sigma0'), 6)}`; sigma d^-1 `{_quick_number(nuisance.get('sg_jitter_sigma_d_inv'), 6)}`; sigma frame `{_quick_number(nuisance.get('sg_jitter_sigma_frame'), 6)}`; points `{nuisance.get('sg_jitter_points', 'n/a')}` |",
        f"| absorption | `{nuisance.get('absorption_mode', 'n/a')}`; mu `{_quick_number(nuisance.get('absorption_mu'), 6)}`; frac `{_quick_number(nuisance.get('absorption_frac'), 6)}` |",
        f"| scale model | group width `{_quick_number(nuisance.get('group_width_deg'), 6)}` deg; detector bins `{nuisance.get('detector_scale_bins', 'n/a')}`; scale count `{best_native.get('scale_count', 'n/a')}` |",
        f"| native row weights | a `{_quick_number(native_weight.get('a'), 6)}`; b `{_quick_number(native_weight.get('b'), 6)}` |",
        "",
        "## Availability Notes",
        "",
        "Rows from older GROND versions may not contain the Sg/quadrature columns. In that case this report still records the high-level config and row residual distributions, but per-row excitation-error distributions require rerunning with this version or using a row-plan cache that records Sg profiles.",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def _write_bloch_diagnostics_reports(
    out_dir: Path,
    row_tables: list[tuple[str, Path]],
    final_run: Path | None,
    final_summary: Mapping[str, object],
    row_metrics: Mapping[str, object],
    file_block: Mapping[str, str],
) -> Path:
    """Write one exact Bloch audit per dataset and a multi-run index."""

    if len(row_tables) <= 1:
        override = row_tables[0][1] if row_tables else None
        summary = try_read_json(override.parent / "summary.json") if override is not None else final_summary
        return _write_bloch_diagnostics_report(
            out_dir,
            final_run,
            summary if isinstance(summary, Mapping) and summary else final_summary,
            row_metrics,
            file_block,
            row_table_override=override,
        )

    entries: list[tuple[str, Path, int]] = []
    for label, row_table in row_tables:
        summary = try_read_json(row_table.parent / "summary.json")
        report_name = f"BLOCH_DIAGNOSTICS_{safe_label(label)}.md"
        report = _write_bloch_diagnostics_report(
            out_dir,
            row_table.parent,
            summary if isinstance(summary, Mapping) and summary else {},
            row_metrics,
            file_block,
            row_table_override=row_table,
            report_name=report_name,
        )
        audit = _quick_csv_numeric_audit(row_table, [])
        entries.append((label, report, int(audit.get("rows", 0) or 0)))
    index = out_dir / "BLOCH_DIAGNOSTICS.md"
    lines = [
        "# GROND Multi-Dataset Bloch Diagnostics",
        "",
        "Each crystal retains its own frozen row table, nuisance model, scale field, and Bloch audit. No synthetic combined row table is implied by this index.",
        "",
        "| dataset | frozen rows | diagnostics | row table |",
        "|---|---:|---|---|",
    ]
    table_by_label = {label: path for label, path in row_tables}
    for label, report, rows in entries:
        lines.append(f"| {label} | {rows} | `{report}` | `{table_by_label[label]}` |")
    index.write_text("\n".join(lines) + "\n")
    return index


def _quick_hand_block_t_style(block: Mapping[str, object], *, final_is_other: bool) -> dict[str, object]:
    """Orient a paired row-level t-style contrast so positive favors the final hand."""

    sign = -1.0 if final_is_other else 1.0
    n = _quick_int(block.get("n"))
    df = n - 1 if n is not None and n >= 1 else None
    raw_delta = _quick_float(block.get("delta_chi2_other_minus_best"))
    raw_t = _quick_float(block.get("paired_row_t_delta_chi2"))
    oriented_delta = sign * raw_delta if math.isfinite(raw_delta) else float("nan")
    oriented_t = sign * raw_t if math.isfinite(raw_t) else float("nan")
    return {
        "n": n,
        "df": df,
        "delta_chi2_rejected_minus_final": oriented_delta,
        "paired_t_style_positive_favors_final": oriented_t,
        "paired_t_style_definition": (
            "paired Student-t-style contrast of per-row delta_chi2; positive favors the final selected hand; "
            "not a calibrated normal Z score"
        ),
        "sqrt_positive_delta_chi2": math.sqrt(max(oriented_delta, 0.0)) if math.isfinite(oriented_delta) else float("nan"),
    }


def _quick_hand_t_style_text(block: Mapping[str, object]) -> str:
    """Format one paired hand contrast with its sample count and degrees of freedom."""

    return (
        f"t {_quick_number(block.get('paired_t_style_positive_favors_final'), 3)} "
        f"(n {block.get('n', 'n/a')}, df {block.get('df', 'n/a')})"
    )


def _quick_final_hand_summary(decision_tree: Mapping[str, object], diagnostic: Mapping[str, object]) -> dict[str, object]:
    """Summarize initial and final hand decisions from the decision-tree artifacts."""

    initial = decision_tree.get("absolute_structure_decision")
    initial_decision = initial if isinstance(initial, Mapping) else {}
    promotion = decision_tree.get("post_feedback_hand_promotion")
    promotion_decision = promotion if isinstance(promotion, Mapping) else {}
    final_hand = str(decision_tree.get("selected_hand") or initial_decision.get("selected_hand") or initial_decision.get("selected") or "n/a")
    initial_hand = str(initial_decision.get("selected_hand") or initial_decision.get("selected") or "n/a")
    final_is_other = final_hand == str(promotion_decision.get("opposite_hand"))
    blocks: dict[str, dict[str, object]] = {}
    for split in ("all", "work", "free"):
        block = diagnostic.get(split)
        if isinstance(block, Mapping):
            blocks[split] = _quick_hand_block_t_style(block, final_is_other=final_is_other)
    return {
        "initial_hand": initial_hand,
        "final_hand": final_hand,
        "promotion_status": promotion_decision.get("status", "n/a") if promotion_decision else "n/a",
        "initial_score_delta_original_minus_alternate": initial_decision.get("score_delta_original_minus_alternate"),
        "post_feedback_score_delta_final_minus_opposite": promotion_decision.get("score_delta_final_minus_opposite"),
        "diagnostic_blocks": blocks,
    }


def _quick_multi_hand_diagnostic(result: Mapping[str, object]) -> dict[str, object]:
    """Load the shared post-feedback multi-crystal hand audit."""

    value = result.get("post_feedback_multi_hand_diagnostic")
    if isinstance(value, Mapping):
        return dict(value)
    if value:
        return try_read_json(Path(str(value)))
    path = result.get("post_feedback_multi_hand_diagnostic_path")
    return try_read_json(Path(str(path))) if path else {}


def _quick_multi_hand_console_rows(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return compact per-dataset hand evidence for console/report consumers."""

    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), Mapping) else {}
    rows: dict[str, dict[str, object]] = {}
    for name, raw in sorted(datasets.items(), key=lambda item: str(item[0])):
        record = raw if isinstance(raw, Mapping) else {}
        paired = (
            record.get("paired_t_style_positive_favors_selected")
            if isinstance(record.get("paired_t_style_positive_favors_selected"), Mapping)
            else {}
        )
        paired_compatibility = (
            record.get("paired_t_style_z_positive_favors_selected")
            if isinstance(record.get("paired_t_style_z_positive_favors_selected"), Mapping)
            else {}
        )
        counts = record.get("row_counts") if isinstance(record.get("row_counts"), Mapping) else {}
        margins = (
            record.get("score_margins_positive_favors_selected")
            if isinstance(record.get("score_margins_positive_favors_selected"), Mapping)
            else {}
        )
        rows[str(name)] = {
            "status": record.get("status", "missing"),
            "rows_all": counts.get("all", 0),
            "rows_work": counts.get("work", 0),
            "rows_free": counts.get("free", 0),
            "t_all": (paired.get("all") or {}).get("t") if isinstance(paired.get("all"), Mapping) else paired_compatibility.get("all"),
            "t_work": (paired.get("work") or {}).get("t") if isinstance(paired.get("work"), Mapping) else paired_compatibility.get("work"),
            "t_free": (paired.get("free") or {}).get("t") if isinstance(paired.get("free"), Mapping) else paired_compatibility.get("free"),
            "df_all": (paired.get("all") or {}).get("df") if isinstance(paired.get("all"), Mapping) else None,
            "df_work": (paired.get("work") or {}).get("df") if isinstance(paired.get("work"), Mapping) else None,
            "df_free": (paired.get("free") or {}).get("df") if isinstance(paired.get("free"), Mapping) else None,
            "chi2_margin_all": (
                (margins.get("all") or {}).get("delta_chi2_opposite_minus_selected")
                if isinstance(margins.get("all"), Mapping)
                else None
            ),
            "chi2_margin_work": (
                (margins.get("work") or {}).get("delta_chi2_opposite_minus_selected")
                if isinstance(margins.get("work"), Mapping)
                else None
            ),
            "chi2_margin_free": (
                (margins.get("free") or {}).get("delta_chi2_opposite_minus_selected")
                if isinstance(margins.get("free"), Mapping)
                else None
            ),
            "composite_margin": record.get(
                "composite_score_margin_opposite_minus_selected_positive_favors_selected"
            ),
            "selected_row_table": record.get("selected_row_table", ""),
            "opposite_row_table": record.get("opposite_row_table", ""),
            "diagnostic": record.get("diagnostic", ""),
        }
    return rows


def _quick_multi_hand_evidence_lines(
    payload: Mapping[str, object],
    *,
    compact: bool = False,
) -> list[str]:
    """Render per-crystal paired t-style hand evidence without pooling crystals."""

    rows = _quick_multi_hand_console_rows(payload)
    lines = [
        "## Per-Dataset Final Hand Evidence",
        "",
        (
            "The final shared hand remains the initial shared-gate decision. Each row below compares one crystal's "
            "frozen final row table with an opposite-hand recomputation at the identical pinned nuisance and N/Q. "
            "Positive paired t-style statistics and score margins favor the selected shared hand. Each t-style "
            "statistic is reported with n and df=n-1; it is not a calibrated normal Z score. No pooled statistic is "
            "computed across independent crystal, sigma, scale, and free-set contracts."
        ),
        "",
    ]
    if not rows:
        lines.extend(
            [
                f"Consistency gate: `{payload.get('status', 'warning_missing_diagnostic')}`. No per-dataset "
                "diagnostic rows were recorded; treat final hand evidence as incomplete.",
                "",
            ]
        )
        return lines
    if compact:
        lines.extend(
            [
                f"Consistency gate: `{payload.get('status', 'missing')}`.",
                "",
                "| dataset | status | rows all/work/free | paired t-style all/work/free | df all/work/free |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for name, row in rows.items():
            lines.append(
                f"| {name} | {row.get('status', 'missing')} | "
                f"{row.get('rows_all', 0)}/{row.get('rows_work', 0)}/{row.get('rows_free', 0)} | "
                f"{_quick_number(row.get('t_all'), 3)}/{_quick_number(row.get('t_work'), 3)}/"
                f"{_quick_number(row.get('t_free'), 3)} | "
                f"{row.get('df_all', 'n/a')}/{row.get('df_work', 'n/a')}/{row.get('df_free', 'n/a')} |"
            )
    else:
        lines.extend(
            [
                f"Consistency gate: `{payload.get('status', 'missing')}`; artifact `{payload.get('artifact', '')}`.",
                "",
                "| dataset | status | rows all/work/free | paired t-style all/work/free | df all/work/free | chi2 margin all/work/free | composite score margin | selected/opposite row tables |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name, row in rows.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(row.get("status", "missing")),
                        f"{row.get('rows_all', 0)}/{row.get('rows_work', 0)}/{row.get('rows_free', 0)}",
                        f"{_quick_number(row.get('t_all'), 3)}/{_quick_number(row.get('t_work'), 3)}/{_quick_number(row.get('t_free'), 3)}",
                        f"{row.get('df_all', 'n/a')}/{row.get('df_work', 'n/a')}/{row.get('df_free', 'n/a')}",
                        f"{_quick_number(row.get('chi2_margin_all'), 3)}/{_quick_number(row.get('chi2_margin_work'), 3)}/{_quick_number(row.get('chi2_margin_free'), 3)}",
                        _quick_number(row.get("composite_margin"), 4),
                        f"`{row.get('selected_row_table', '')}` / `{row.get('opposite_row_table', '')}`",
                    ]
                )
                + " |"
            )
    lines.append("")
    return lines


def _normalize_publication_endpoint_block(endpoint: Mapping[str, object], *, selection_source: str) -> dict[str, object]:
    """Normalize endpoint-chooser and endpoint-calibration records to one shape."""

    shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
    artifact_provenance = endpoint_artifact_provenance(endpoint)
    provenance_ok = artifact_provenance.get("passed") is True
    pathology = (
        artifact_provenance.get("lst_pathology")
        if isinstance(artifact_provenance.get("lst_pathology"), Mapping)
        else {}
    )
    pathology_complete = str(pathology.get("pathology_status") or "") == "complete"
    promotion_fields = endpoint_promotion_field_contract(endpoint)
    canonical = (
        artifact_provenance.get("canonical_artifacts")
        if isinstance(artifact_provenance.get("canonical_artifacts"), Mapping)
        else {}
    )
    canonical_res = canonical.get("res") if isinstance(canonical.get("res"), Mapping) else {}
    canonical_hkl = canonical.get("hkl") if isinstance(canonical.get("hkl"), Mapping) else {}
    canonical_lst = canonical.get("lst") if isinstance(canonical.get("lst"), Mapping) else {}

    def first_text(*keys: str) -> str:
        for key in keys:
            value = endpoint.get(key)
            if value:
                return str(value)
        return ""

    def first_number(*keys: str) -> object:
        for key in keys:
            if key in endpoint and endpoint.get(key) is not None:
                return endpoint.get(key)
            if key in shelxl and shelxl.get(key) is not None:
                return shelxl.get(key)
        return None

    out = dict(endpoint)
    out["selection_source"] = selection_source
    out["source_kind"] = str(endpoint.get("source_kind") or ("endpoint_calibration" if selection_source == "endpoint_calibration" else ""))
    out["endpoint_res"] = str(canonical_res.get("path") or first_text("endpoint_res", "res"))
    out["endpoint_hkl"] = str(canonical_hkl.get("path") or first_text("endpoint_hkl", "hkl"))
    out["endpoint_lst"] = str(
        canonical_lst.get("path") or first_text("endpoint_lst", "lst") or shelxl.get("lst", "") or ""
    )
    out["endpoint_ins"] = first_text("endpoint_ins", "ins")
    if not out["endpoint_ins"] and out["endpoint_res"]:
        out["endpoint_ins"] = str(Path(out["endpoint_res"]).with_suffix(".ins"))
    if not out["endpoint_hkl"] and out["endpoint_res"]:
        endpoint_hkl = Path(out["endpoint_res"]).with_suffix(".hkl")
        if endpoint_hkl.exists():
            out["endpoint_hkl"] = str(endpoint_hkl)
    out["endpoint_manifest"] = first_text("endpoint_manifest", "manifest")
    if not out["endpoint_manifest"] and out["endpoint_res"]:
        manifest = Path(out["endpoint_res"]).with_name("manifest.json")
        if manifest.exists():
            out["endpoint_manifest"] = str(manifest)
    out["endpoint_row_table"] = first_text("endpoint_row_table", "row_table")
    if not out["endpoint_row_table"] and str(out.get("source_kind", "")) == "observation_polish":
        run_dir = first_text("run_dir")
        if run_dir:
            out["endpoint_row_table"] = str(Path(run_dir) / "promoted_observation_rows.csv")
    out["spec"] = str(endpoint.get("spec") or endpoint.get("scale_spec") or endpoint.get("source_kind") or endpoint.get("label") or "")
    out["shelxl_R1gt"] = first_number("shelxl_R1gt", "R1_gt")
    out["shelxl_R1all"] = first_number("shelxl_R1all", "R1_all")
    out["shelxl_wR2all"] = first_number("shelxl_wR2all", "wR2_all")
    out["shelxl_GooF"] = first_number("shelxl_GooF", "GooF")
    out["shelxl_Ngt"] = first_number("shelxl_Ngt", "N_gt")
    out["shelxl_Nall"] = first_number("shelxl_Nall", "N_all")
    recorded_metrics = {
        "R1_gt": out["shelxl_R1gt"],
        "R1_all": out["shelxl_R1all"],
        "wR2_all": out["shelxl_wR2all"],
        "GooF": out["shelxl_GooF"],
        "N_gt": out["shelxl_Ngt"],
        "N_all": out["shelxl_Nall"],
    }
    metric_consistency = endpoint_lst_metric_consistency(recorded_metrics, artifact_provenance)
    live_metrics = (
        artifact_provenance.get("lst_metrics")
        if isinstance(artifact_provenance.get("lst_metrics"), Mapping)
        else {}
    )
    if str(live_metrics.get("metric_status") or "") == "complete":
        out.update(
            {
                "shelxl_R1gt": live_metrics.get("R1_gt"),
                "shelxl_R1all": live_metrics.get("R1_all"),
                "shelxl_wR2all": live_metrics.get("wR2_all"),
                "shelxl_GooF": live_metrics.get("GooF"),
                "shelxl_Ngt": live_metrics.get("N_gt"),
                "shelxl_Nall": live_metrics.get("N_all"),
            }
        )
    shelxl_details = (
        _quick_shelxl_details(shelxl, Path(out["endpoint_lst"]) if out["endpoint_lst"] else None)
        if provenance_ok
        else dict(shelxl)
    )
    fallback_pairs = {
        "shelxl_R1gt": "R1_gt",
        "shelxl_R1all": "R1_all",
        "shelxl_wR2all": "wR2_all",
        "shelxl_GooF": "GooF",
        "shelxl_Ngt": "N_gt",
        "shelxl_Nall": "N_all",
        "shelxl_Rint": "Rint",
        "shelxl_Rsigma": "Rsigma",
        "shelxl_parameters": "parameters",
        "shelxl_restraints": "restraints",
        "shelxl_npd_count": "npd_count",
        "shelxl_npd_atoms": "npd_atoms",
        "shelxl_split_count": "split_count",
        "shelxl_has_npd": "has_npd",
        "shelxl_has_split_warnings": "has_split_warnings",
        "shelxl_active_wght_a": "active_wght_a",
        "shelxl_active_wght_b": "active_wght_b",
        "shelxl_recommended_wght_a": "recommended_wght_a",
        "shelxl_recommended_wght_b": "recommended_wght_b",
    }
    pathology_count_pairs = {"shelxl_npd_count", "shelxl_split_count"}
    pathology_flag_pairs = {"shelxl_has_npd", "shelxl_has_split_warnings"}
    for out_key, detail_key in fallback_pairs.items():
        old_value = out.get(out_key)
        if detail_key not in shelxl_details:
            continue
        detail_value = shelxl_details[detail_key]
        if out_key in pathology_count_pairs:
            old_count = _quick_int(old_value)
            detail_count = _quick_int(detail_value)
            if old_count is None:
                out[out_key] = detail_value
            elif detail_count is not None:
                out[out_key] = max(old_count, detail_count)
        elif out_key == "shelxl_npd_atoms":
            old_atoms = old_value if isinstance(old_value, (list, tuple, set)) else []
            detail_atoms = detail_value if isinstance(detail_value, (list, tuple, set)) else []
            out[out_key] = sorted(
                {str(atom) for atom in [*old_atoms, *detail_atoms] if str(atom).strip()}
            )
        elif out_key in pathology_flag_pairs:
            out[out_key] = bool(old_value) or bool(detail_value)
        elif old_value is None or old_value == "":
            out[out_key] = detail_value
    out["shelxl_lst_exists"] = shelxl_details.get("lst_exists", shelxl_details.get("exists", False))
    out["provenance_required"] = True
    out["artifact_provenance_ok"] = provenance_ok
    out["artifact_provenance"] = artifact_provenance
    out["res_provenance_ok"] = provenance_ok
    out["res_provenance"] = artifact_provenance
    out["res_sha256"] = str(canonical_res.get("sha256") or "")
    out["hkl_sha256"] = str(canonical_hkl.get("sha256") or "")
    out["lst_sha256"] = str(canonical_lst.get("sha256") or "")
    out["endpoint_res_sha256"] = out["res_sha256"]
    out["endpoint_hkl_sha256"] = out["hkl_sha256"]
    out["endpoint_lst_sha256"] = out["lst_sha256"]
    out["metric_consistency_required"] = True
    out["metric_consistency_ok"] = metric_consistency.get("passed") is True
    out["metric_consistency"] = metric_consistency
    out["promotion_fields_required"] = True
    out["promotion_fields_ok"] = promotion_fields.get("passed") is True
    out["promotion_field_contract"] = promotion_fields
    out["pathology_required"] = True
    out["pathology_status"] = str(pathology.get("pathology_status") or "unverified")
    if pathology_complete:
        out["shelxl_npd_count"] = max(
            _quick_int(out.get("shelxl_npd_count")) or 0,
            int(pathology.get("npd_count", 0) or 0),
        )
        old_atoms = out.get("shelxl_npd_atoms")
        old_atoms = old_atoms if isinstance(old_atoms, (list, tuple, set)) else []
        out["shelxl_npd_atoms"] = sorted(
            {str(atom) for atom in [*old_atoms, *(pathology.get("npd_atoms") or [])] if str(atom).strip()}
        )
        out["shelxl_split_count"] = max(
            _quick_int(out.get("shelxl_split_count")) or 0,
            int(pathology.get("split_count", 0) or 0),
        )
        out["shelxl_has_npd"] = bool(out.get("shelxl_has_npd")) or bool(pathology.get("has_npd"))
        out["shelxl_has_split_warnings"] = bool(out.get("shelxl_has_split_warnings")) or bool(
            pathology.get("has_split_warnings")
        )
    else:
        # Preserve every recorded warning in diagnostic output.  An
        # unverified/truncated LST cannot prove cleanliness, but it must not
        # erase pathology already recorded by the producer either.
        out.setdefault("shelxl_npd_count", None)
        out.setdefault("shelxl_npd_atoms", [])
        out.setdefault("shelxl_split_count", None)
        out.setdefault("shelxl_has_npd", None)
        out.setdefault("shelxl_has_split_warnings", None)
    if not out.get("wght_mode") and provenance_ok:
        wght_a = _quick_fraction(out.get("shelxl_active_wght_a"))
        wght_b = _quick_fraction(out.get("shelxl_active_wght_b"))
        if wght_a == 0.0 and wght_b == 0.0:
            out["wght_mode"] = "freeze-zero"
    if not provenance_ok:
        out["provenance_error"] = (
            "publication endpoint RES/HKL/LST no longer match the exact paths and hashes selected by GROND; "
            "use GROND push-endpoint for an edited model"
        )
    elif not pathology_complete:
        out["provenance_error"] = (
            "publication endpoint LST does not contain a complete final SHELXL NPD/split summary"
        )
    elif metric_consistency.get("passed") is not True:
        out["provenance_error"] = str(metric_consistency.get("reason") or "stored/live endpoint metrics disagree")
    elif promotion_fields.get("passed") is not True:
        out["provenance_error"] = str(promotion_fields.get("reason") or "producer promotion fields are incomplete")
    return out


def _endpoint_calibration_publication_endpoint(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the endpoint-calibration publication replay, if present."""

    if result.get("combined"):
        # Historical multi summaries may contain an unsafe single-table replay.
        # Multi-dataset scale contracts are independent and can only be
        # calibrated through a future per-dataset replay-and-recombine path.
        return {}
    endpoint_calibration = result.get("endpoint_calibration") if isinstance(result.get("endpoint_calibration"), Mapping) else {}
    selection = endpoint_calibration.get("publication_selection", {}) if isinstance(endpoint_calibration, Mapping) else {}
    endpoint = selection.get("publication_endpoint") if isinstance(selection, Mapping) else {}
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}
    return _normalize_publication_endpoint_block(endpoint, selection_source="endpoint_calibration")


def _chooser_publication_endpoint(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the endpoint chooser's best physical endpoint, if present."""

    chooser = result.get("publication_endpoint_chooser") if isinstance(result.get("publication_endpoint_chooser"), Mapping) else {}
    endpoint = chooser.get("best_physical_endpoint") if isinstance(chooser, Mapping) else {}
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}
    return _normalize_publication_endpoint_block(endpoint, selection_source="endpoint_chooser")


def _chooser_diagnostic_endpoint(result: Mapping[str, object]) -> Mapping[str, object]:
    """Return the endpoint chooser's best rejected diagnostic endpoint, if present."""

    chooser = result.get("publication_endpoint_chooser") if isinstance(result.get("publication_endpoint_chooser"), Mapping) else {}
    endpoint = chooser.get("best_diagnostic_endpoint") if isinstance(chooser, Mapping) else {}
    if not isinstance(endpoint, Mapping) or not endpoint:
        return {}
    return _normalize_publication_endpoint_block(endpoint, selection_source="endpoint_chooser_diagnostic")


def _quick_endpoint_file_block(publication_endpoint: Mapping[str, object], final_model: Path | None) -> dict[str, str]:
    """Collect the endpoint paths users most often need to open."""

    final_endpoint_dir = final_model.parent if final_model else None
    final_endpoint_stem = final_model.stem if final_model else ""

    def existing(path: Path | None) -> str:
        return str(path) if path is not None and path.exists() else ""

    return {
        "final_feedback_res": str(final_model or ""),
        "final_feedback_hkl": existing(final_endpoint_dir / (final_endpoint_stem + ".hkl") if final_endpoint_dir else None),
        "final_feedback_lst": existing(final_endpoint_dir / (final_endpoint_stem + ".lst") if final_endpoint_dir else None),
        "final_feedback_manifest": existing(final_endpoint_dir / "manifest.json" if final_endpoint_dir else None),
        "publication_res": str(publication_endpoint.get("endpoint_res", "")),
        "publication_ins": str(publication_endpoint.get("endpoint_ins", "")),
        "publication_hkl": str(publication_endpoint.get("endpoint_hkl", "")),
        "publication_lst": str(publication_endpoint.get("endpoint_lst", "")),
        "publication_manifest": str(publication_endpoint.get("endpoint_manifest", "")),
        "publication_row_table": str(publication_endpoint.get("endpoint_row_table", "")),
    }


def _quick_transform_summary(transform_decision: Mapping[str, object]) -> dict[str, object]:
    """Return compact transform details for the quick summary."""

    selected = transform_decision.get("selected")
    selected_block = selected if isinstance(selected, Mapping) else {}
    metadata = transform_decision.get("metadata_transform_decision")
    metadata_block = metadata if isinstance(metadata, Mapping) else {}
    return {
        "selected_transform": selected_block.get("transform", "n/a"),
        "selected_status": selected_block.get("status", "n/a"),
        "metadata_status": metadata_block.get("status", "n/a"),
        "metadata_log_intensity_corr": (
            metadata_block.get("records", [{}])[0].get("log_intensity_corr")
            if isinstance(metadata_block.get("records"), list) and metadata_block.get("records")
            else None
        ),
        "metadata_correlation_gap": metadata_block.get("correlation_gap"),
        "metadata_paired_rows": (
            metadata_block.get("records", [{}])[0].get("paired_rows")
            if isinstance(metadata_block.get("records"), list) and metadata_block.get("records")
            else None
        ),
        "metadata_unique_overlap": (
            metadata_block.get("records", [{}])[0].get("unique_overlap")
            if isinstance(metadata_block.get("records"), list) and metadata_block.get("records")
            else None
        ),
    }


def _quick_flat_hand_t_style_text(
    t_value: object,
    n_value: object,
    df_value: object,
) -> str:
    """Format a flat console-payload paired t-style record."""

    n = _quick_int(n_value)
    df = _quick_int(df_value)
    return f"t {_quick_number(t_value, 3)} (n {n if n is not None else 'n/a'}, df {df if df is not None else 'n/a'})"


def _quick_delta_percent(before: object, after: object) -> str:
    """Format an absolute percent-point change for fractional metrics."""

    b = _quick_fraction(before)
    a = _quick_fraction(after)
    if b is None or a is None:
        return "n/a"
    sign = "+" if (a - b) >= 0 else ""
    return f"{sign}{100.0 * (a - b):.2f} pp"


def _quick_source_shelx_controls(path: Path | None) -> dict[str, object]:
    """Read the source RES/INS control records that affect baseline interpretation."""

    out: dict[str, object] = {"hklf": "", "exti": "", "wght": "", "acta": False}
    if path is None or not path.exists():
        return out
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        head = line.split()[0].upper()
        if head == "END":
            break
        if head == "HKLF":
            out["hklf"] = " ".join(line.split()[:2])
        elif head == "EXTI":
            out["exti"] = " ".join(line.split()[:2])
        elif head == "WGHT":
            terms = _quick_parse_wght_terms(line)
            out["wght"] = "WGHT " + " ".join(_quick_number(value, 6) for value in terms) if terms else line
        elif head == "ACTA":
            out["acta"] = True
    return out


def _quick_source_shelx_control_text(controls: Mapping[str, object]) -> str:
    """Format source SHELX controls in one compact report cell."""

    hklf = str(controls.get("hklf") or "HKLF n/a")
    exti = str(controls.get("exti") or "EXTI none")
    wght = str(controls.get("wght") or "WGHT n/a")
    acta = "ACTA present" if controls.get("acta") else "ACTA absent"
    return f"{hklf}; {exti}; {wght}; {acta}"


def _quick_row_weight_definition(
    nuisance: Mapping[str, object],
    dataset_contracts: list[Mapping[str, object]] | None = None,
) -> str:
    """Return the exact row intensity weighting convention used by the final native report."""

    definition = (
        "Raw signed-row intensity `wR(I)all` is `sqrt(sum_r w_r (Iobs_r - Icalc_r)^2 / sum_r w_r Iobs_r^2)` "
        "over signed observation rows. For dynamical rows `Icalc_r = Idyn_r`; for matched Born rows "
        "`Icalc_r = Iborn_r` under the same row contract, thickness, geometry, scales, and sigma model. "
        "`w_r = 1 / [sigma_eff_r^2 + (a P_r)^2 + b P_r]`, with "
        "`P_r = (max(Iobs_r, 0) + 2 max(Icalc_r, 0)) / 3`. "
    )
    if dataset_contracts:
        weight_text = []
        for contract in dataset_contracts:
            block = contract.get("native_weight") if isinstance(contract.get("native_weight"), Mapping) else {}
            nui = contract.get("nuisance") if isinstance(contract.get("nuisance"), Mapping) else {}
            a = block.get("a", nui.get("native_wght_a"))
            b = block.get("b", nui.get("native_wght_b"))
            weight_text.append(
                f"{contract.get('dataset', 'dataset')} a={_quick_number(a, 6)}, b={_quick_number(b, 6)}"
            )
        definition += (
            "This multi-dataset result uses independent per-crystal native weights (`per_dataset_native`): "
            + "; ".join(weight_text)
            + ". The combined-summary fallback a/b pair is not an active joint weighting equation. "
        )
    else:
        a = nuisance.get("native_wght_a", nuisance.get("wght_a_grid"))
        b = nuisance.get("native_wght_b", nuisance.get("wght_b_grid"))
        definition += f"This run reports final native row weights `a={_quick_number(a, 6)}` and `b={_quick_number(b, 6)}`. "
    return (
        definition
        + "This is the signed-row intensity statistic to compare with native dynamical-refinement row "
        "targets; it is not SHELXL's merged-reflection `wR2(F^2)all`."
    )


def _quick_complete_jana_metrics(metrics: object) -> bool:
    """Return whether one row block contains the complete core JANA family."""

    if not isinstance(metrics, Mapping):
        return False
    nobs = _quick_int(metrics.get("Nobs"))
    nall = _quick_int(metrics.get("Nall"))
    mnobs = _quick_int(metrics.get("MNobs"))
    mnall = _quick_int(metrics.get("MNall"))
    if None in {nobs, nall, mnobs, mnall}:
        return False
    if nobs < 0 or nall <= 0 or nobs > nall or mnobs < 0 or mnall <= 0 or mnobs > mnall:
        return False
    if any(_quick_fraction(metrics.get(key)) is None for key in ("Rall", "wRall", "MRall", "MwRall")):
        return False
    if nobs > 0 and _quick_fraction(metrics.get("Robs")) is None:
        return False
    return not (mnobs > 0 and _quick_fraction(metrics.get("MRobs")) is None)


def _quick_final_jana_row_summary(row_metrics: Mapping[str, object]) -> dict[str, object]:
    """Return the final JANA/PETS-style row block, with legacy fallbacks."""

    direct = row_metrics.get("row_mr")
    direct_dynamic = direct.get("dynamic") if isinstance(direct, Mapping) else None
    if _quick_complete_jana_metrics(direct_dynamic):
        out = dict(direct)
        out["source"] = "row_mr"
        return out

    contracts = row_metrics.get("row_resolution_scale_contracts")
    candidates = contracts.get("contracts") if isinstance(contracts, Mapping) else None
    if not isinstance(candidates, list):
        legacy = row_metrics.get("resolution_scale_endpoints")
        candidates = legacy.get("endpoints") if isinstance(legacy, Mapping) else None
    if not isinstance(candidates, list):
        return {}
    preferred = (
        "suggested_resolution_compact_scales",
        "requested_resolution_compact_scales",
        "requested_resolution_full_scales",
    )
    by_label = {str(row.get("label")): row for row in candidates if isinstance(row, Mapping)}
    for label in preferred:
        candidate = by_label.get(label)
        summary = candidate.get("row_mr") if isinstance(candidate, Mapping) else None
        summary_dynamic = summary.get("dynamic") if isinstance(summary, Mapping) else None
        if _quick_complete_jana_metrics(summary_dynamic):
            out = dict(summary)
            out["source"] = f"row_resolution_scale_contracts:{label}"
            return out
    return {}


def _table1_jana_record(
    label: str,
    role: str,
    metrics: Mapping[str, object] | None,
    source: str,
) -> dict[str, object]:
    """Normalize one JANA/PETS-style row-statistics record for Table 1."""

    available = _quick_complete_jana_metrics(metrics)
    row = metrics if isinstance(metrics, Mapping) else {}
    ninput = _quick_int(row.get("Ninput")) if available else None
    nexcluded = _quick_int(row.get("Nexcluded")) if available else None
    unavailable_reason = source
    if isinstance(metrics, Mapping) and not available:
        unavailable_reason = f"not available: incomplete core JANA/PETS factor family in {source}"
    return {
        "model": label,
        "role": role,
        "status": "available" if available else "not_available",
        "status_reason": "" if available else unavailable_reason,
        "Ninput": ninput,
        "Nexcluded": nexcluded,
        "Nmatched": _quick_int(row.get("Nmatched")) if available else None,
        "Nobs": _quick_int(row.get("Nobs")) if available else None,
        "Nall": _quick_int(row.get("Nall")) if available else None,
        "Robs": _quick_fraction(row.get("Robs")) if available else None,
        "Rall": _quick_fraction(row.get("Rall")) if available else None,
        "wRobs": _quick_fraction(row.get("wRobs")) if available else None,
        "wRall": _quick_fraction(row.get("wRall")) if available else None,
        "Nmerge_input": _quick_int(row.get("Nmerge_input")) if available else None,
        "Nmerge_used": _quick_int(row.get("Nmerge_used")) if available else None,
        "Nmerge_excluded": _quick_int(row.get("Nmerge_excluded")) if available else None,
        "MNobs": _quick_int(row.get("MNobs")) if available else None,
        "MNall": _quick_int(row.get("MNall")) if available else None,
        "MRobs": _quick_fraction(row.get("MRobs")) if available else None,
        "MRall": _quick_fraction(row.get("MRall")) if available else None,
        "MwRobs": _quick_fraction(row.get("MwRobs")) if available else None,
        "MwRall": _quick_fraction(row.get("MwRall")) if available else None,
        "source": source if available else "",
    }


def _table1_shelxl_record(
    label: str,
    role: str,
    details: Mapping[str, object],
    controls: Mapping[str, object],
    *,
    res: str,
    hkl: str,
    lst: str,
) -> dict[str, object]:
    """Normalize one SHELXL endpoint record for Table 1."""

    available = _quick_fraction(details.get("R1_all")) is not None
    return {
        "model": label,
        "role": role,
        "status": "available" if available else "not_available",
        "status_reason": "" if available else "no authoritative SHELXL listing/statistics file was found",
        "Ngt": _quick_int(details.get("N_gt")) if available else None,
        "Nall": _quick_int(details.get("N_all")) if available else None,
        "R1gt": _quick_fraction(details.get("R1_gt")) if available else None,
        "R1all": _quick_fraction(details.get("R1_all")) if available else None,
        "wR2all": _quick_fraction(details.get("wR2_all")) if available else None,
        "GooF": _quick_fraction(details.get("GooF")) if available else None,
        "parameters": _quick_int(details.get("parameters")) if available else None,
        "restraints": _quick_int(details.get("restraints")) if available else None,
        "WGHT": str(controls.get("wght") or "WGHT n/a") if available else None,
        "EXTI": str(controls.get("exti") or "EXTI none") if available else None,
        "NPD": _quick_int(details.get("npd_count")) if available else None,
        "split": _quick_int(details.get("split_count")) if available else None,
        "res": res,
        "hkl": hkl,
        "lst": lst,
    }


def _table1_pair(row: Mapping[str, object], first: str, second: str) -> str:
    """Format a count pair from a normalized Table 1 row."""

    a = row.get(first)
    b = row.get(second)
    return f"{a}/{b}" if a is not None and b is not None else "n/a"


def _write_final_statistics_table1(
    out_dir: Path,
    *,
    row_metrics: Mapping[str, object],
    row_metrics_path: Path,
    input_shelxl: Mapping[str, object],
    publication_shelxl: Mapping[str, object],
    publication_endpoint: Mapping[str, object],
    file_block: Mapping[str, object],
    input_controls: Mapping[str, object],
    final_row_table: str,
) -> dict[str, object]:
    """Write the canonical final statistics Table 1 without mixing metric families."""

    jana = _quick_final_jana_row_summary(row_metrics)
    dynamic = jana.get("dynamic") if isinstance(jana.get("dynamic"), Mapping) else None
    born = jana.get("born") if isinstance(jana.get("born"), Mapping) else None
    if born is None and isinstance(jana.get("kinematic"), Mapping):
        born = jana.get("kinematic")
    jana_source = str(jana.get("source") or row_metrics_path)
    jana_definition = jana.get("definition") if isinstance(jana.get("definition"), Mapping) else {}
    jana_instability_u = _quick_fraction(jana_definition.get("instability_u"))
    jana_instability_u_text = (
        f"{jana_instability_u:g}" if jana_instability_u is not None else "n/a"
    )

    row_records = [
        _table1_jana_record(
            "Conventional kinematic refinement",
            "merged conventional refinement",
            None,
            "not available: no audited matched unmerged prediction/scale contract; use the SHELXL endpoint panel",
        ),
        _table1_jana_record(
            "GROND Born (matched kinematical row control)",
            "single-scattering control at the final coordinates under the same signed-row contract and scales",
            born,
            jana_source,
        ),
        _table1_jana_record(
            "GROND Bloch dynamical",
            "Bloch multiple-scattering prediction at the final coordinates",
            dynamic,
            jana_source,
        ),
    ]

    publication_res_text = str(file_block.get("publication_res") or "")
    publication_res = Path(publication_res_text) if publication_res_text else None
    publication_controls = _quick_source_shelx_controls(publication_res)
    conventional_endpoint = _table1_shelxl_record(
        "Conventional kinematic reference refinement",
        "starting merged HKLF4 refinement",
        input_shelxl,
        input_controls,
        res=str(input_shelxl.get("res") or ""),
        hkl=str(input_shelxl.get("hkl") or ""),
        lst=str(input_shelxl.get("lst") or ""),
    )
    born_endpoint = _table1_shelxl_record(
        "GROND Born corrected-HKL endpoint",
        "single-scattering endpoint control",
        {},
        {},
        res="",
        hkl="",
        lst="",
    )
    born_endpoint["status_reason"] = (
        "not generated: no actual Born-corrected SHELXL endpoint refinement exists for this run"
    )
    dynamic_endpoint = _table1_shelxl_record(
        "GROND dynamical corrected-HKL publication endpoint",
        "final merged corrected-HKL SHELXL endpoint",
        publication_shelxl,
        publication_controls,
        res=publication_res_text,
        hkl=str(file_block.get("publication_hkl") or ""),
        lst=str(file_block.get("publication_lst") or ""),
    )
    endpoint_records = [conventional_endpoint, born_endpoint, dynamic_endpoint]

    payload = {
        "schema": "grond.table1_final_statistics.v1",
        "title": "Table 1. Final model and refinement statistics",
        "status": (
            "available"
            if any(row["status"] == "available" for row in row_records + endpoint_records)
            else "not_available"
        ),
        "reporting_contract": {
            "endpoint": "SHELXL style: R1(F)gt, R1(F)all, wR2(F^2)all, GooF, and merged endpoint counts",
            "row": "JANA/PETS style: Robs, Rall, wRall on unmerged rows and MRobs, MRall, MwRall after post-refinement symmetry merging; wRobs and MwRobs are optional extensions",
            "separation_rule": "SHELXL endpoint and JANA/PETS row factors have different targets and denominators and are never placed in a shared generic R column.",
            "born_rule": "Born is a matched kinematical row control, not a separately refined conventional SHELXL endpoint.",
            "row_sigma_rule": "JANA/PETS-style weights use sigma_eff from the frozen row table. A separate raw_row_weight is not multiplied into the reported R/MR factors unless its effect is already encoded in sigma_eff.",
            "row_instability_u": jana_instability_u,
            "matched_support_rule": "Born and Bloch statistics must use one identical finite observation support; Ninput/Nmatched/Nexcluded and merge-HKL exclusions are reported explicitly when known.",
            "missing_rule": "Unavailable statistics remain explicit null/n/a values and are never inferred from a different target class.",
            "numeric_units": "fractions in JSON and CSV; percentages in Markdown",
        },
        "sources": {
            "jana_row_metrics": str(row_metrics_path),
            "final_row_table": final_row_table,
            "publication_endpoint_selection": str(
                publication_endpoint.get("selection_source")
                or publication_endpoint.get("source_kind")
                or ""
            ),
        },
        "shelxl_endpoints": endpoint_records,
        "jana_pets_rows": row_records,
    }

    md_path = out_dir / "TABLE1_FINAL_STATISTICS.md"
    json_path = out_dir / "TABLE1_FINAL_STATISTICS.json"
    endpoint_csv = out_dir / "TABLE1_SHELXL_ENDPOINTS.csv"
    row_csv = out_dir / "TABLE1_JANA_ROWS.csv"
    alias_md = out_dir / "TABLE_1.md"
    alias_json = out_dir / "TABLE_1.json"

    lines = [
        "# Table 1. Final Model And Refinement Statistics",
        "",
        "This is the canonical GROND statistics table. Panel A contains only SHELXL endpoint statistics. Panel B contains only JANA/PETS-style signed-row and post-refinement merged-row factors computed by GROND. The two panels have different targets and denominators and must not be collapsed into generic `R` columns.",
        "",
        "## A. SHELXL Endpoint Statistics",
        "",
        "| model | status | Ngt/Nall | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | parameters/restraints | WGHT | EXTI | NPD/split |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in endpoint_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["status"]).replace("_", " "),
                    _table1_pair(row, "Ngt", "Nall"),
                    _quick_percent(row.get("R1gt")),
                    _quick_percent(row.get("R1all")),
                    _quick_percent(row.get("wR2all")),
                    _quick_number(row.get("GooF"), 3),
                    _table1_pair(row, "parameters", "restraints"),
                    str(row.get("WGHT") or "n/a"),
                    str(row.get("EXTI") or "n/a"),
                    _table1_pair(row, "NPD", "split"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "SHELXL definitions: `R1(F)gt` is the conventional amplitude R factor for SHELXL's observed/greater-than subset; `R1(F)all` uses all endpoint reflections; `wR2(F^2)all` and `GooF` use the endpoint sigma and active `WGHT` contract. `Ngt` is not the JANA/PETS `Nobs` count.",
            "",
            "## B. JANA/PETS-Style Row And Post-Refinement Merged Statistics",
            "",
            "| model | status | Ninput/excluded | Nmatched | Nobs/Nall | Nmerge used/excluded | MNobs/MNall | Robs | Rall | wRobs | wRall | MRobs | MRall | MwRobs | MwRall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in row_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["status"]).replace("_", " "),
                    _table1_pair(row, "Ninput", "Nexcluded"),
                    str(row.get("Nmatched")) if row.get("Nmatched") is not None else "n/a",
                    _table1_pair(row, "Nobs", "Nall"),
                    _table1_pair(row, "Nmerge_used", "Nmerge_excluded"),
                    _table1_pair(row, "MNobs", "MNall"),
                    _quick_percent(row.get("Robs")),
                    _quick_percent(row.get("Rall")),
                    _quick_percent(row.get("wRobs")),
                    _quick_percent(row.get("wRall")),
                    _quick_percent(row.get("MRobs")),
                    _quick_percent(row.get("MRall")),
                    _quick_percent(row.get("MwRobs")),
                    _quick_percent(row.get("MwRall")),
                ]
            )
            + " |"
        )
    unavailable = [row for row in row_records + endpoint_records if row.get("status") != "available"]
    lines.extend(
        [
            "",
            f"Row definitions: `Robs`/`Rall` are R1-form amplitude residuals formed from `sqrt(max(I,0))`; observed subsets satisfy `Iobs > 3 sigma(Iobs)`. `wRall` uses the JANA/PETS weighted intensity-residual form with instability factor `u={jana_instability_u_text}`. Born and Bloch factors use the same finite support. `Nmerge excluded` counts supported rows with missing, non-finite, or non-integral HKL indices. `MR`/`MwR` repeat the calculations after arithmetic merging of valid model-symmetry-related observed and calculated rows with propagated merged sigma. Optional `wRobs`/`MwRobs` apply the weighted form to the observed subsets. Weights use `sigma_eff` from the frozen row table; a separate `raw_row_weight` is not multiplied into the reported factors. Values are computed by GROND and are not parsed from JANA or PETS.",
            "",
            "The Born row is the matched single-scattering control under the final GROND row contract. It is not a third SHELXL endpoint.",
        ]
    )
    if unavailable:
        lines.extend(["", "## Explicitly Unavailable Statistics", ""])
        for row in unavailable:
            lines.append(f"- **{row['model']}:** {row.get('status_reason') or 'not available'}.")
    lines.extend(
        [
            "",
            "## Authoritative Sources",
            "",
            f"- JANA/PETS-style row metrics: `{row_metrics_path}`",
            f"- Final signed row table(s): `{final_row_table}`",
            f"- Conventional SHELXL listing/statistics: `{input_shelxl.get('lst', '')}`",
            f"- Publication SHELXL listing: `{file_block.get('publication_lst', '')}`",
            "",
        ]
    )

    json_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    md_text = "\n".join(lines)
    json_path.write_text(json_text)
    md_path.write_text(md_text)
    alias_json.write_text(json_text)
    alias_md.write_text(md_text)

    for path, records in ((endpoint_csv, endpoint_records), (row_csv, row_records)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {
        "status": payload["status"],
        "markdown": str(md_path),
        "json": str(json_path),
        "endpoint_csv": str(endpoint_csv),
        "row_csv": str(row_csv),
        "short_markdown": str(alias_md),
        "short_json": str(alias_json),
        "payload": payload,
    }


def _quick_thickness_text(
    nuisance: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    dataset_contracts: list[Mapping[str, object]] | None = None,
) -> str:
    """Describe effective thickness as a mean plus its distribution contract."""

    per_dataset: list[tuple[str, float, float, str]] = []
    for contract in dataset_contracts or []:
        block = contract.get("nuisance") if isinstance(contract.get("nuisance"), Mapping) else {}
        thickness_a = safe_float(block.get("thickness_A"), float("nan"))
        grad = safe_float(block.get("thickness_grad_frac"), float("nan"))
        if not math.isfinite(thickness_a) or thickness_a <= 0.0:
            continue
        per_dataset.append(
            (
                str(contract.get("dataset") or f"dataset_{len(per_dataset) + 1}"),
                thickness_a,
                grad,
                str(block.get("thickness_model") or "single"),
            )
        )
    if per_dataset:
        centers = [row[1] / 10.0 for row in per_dataset]
        details = []
        for name, thickness_a, grad, model in per_dataset:
            grad_text = f", g={grad:.3f}" if math.isfinite(grad) else ""
            details.append(f"{name} {thickness_a / 10.0:.2f} nm ({model}{grad_text})")
        return (
            "per-dataset centers "
            + "; ".join(details)
            + f"; center range {min(centers):.2f}-{max(centers):.2f} nm"
        )

    meta = metadata if isinstance(metadata, Mapping) else {}
    thickness_a = safe_float(
        nuisance.get("thickness_A", meta.get("thickness_A_mean")),
        float("nan"),
    )
    if not math.isfinite(thickness_a):
        return "n/a"
    thickness_nm = thickness_a / 10.0
    model = str(nuisance.get("thickness_model") or meta.get("thickness_model") or "single")
    if model == "uniform_gradient":
        grad = max(
            0.0,
            safe_float(
                nuisance.get("thickness_grad_frac", meta.get("thickness_grad_frac")),
                0.0,
            ),
        )
        low_nm = max(0.0, thickness_a * (1.0 - grad)) / 10.0
        high_nm = max(0.0, thickness_a * (1.0 + grad)) / 10.0
        return (
            f"{thickness_nm:.2f} nm mean ({thickness_a:.1f} A); "
            f"uniform-gradient support {low_nm:.2f}-{high_nm:.2f} nm"
        )
    if model in {"gaussian", "gaussian_gradient"}:
        sigma_frac = abs(
            safe_float(
                nuisance.get("thickness_sigma_frac", meta.get("thickness_sigma_frac")),
                0.0,
            )
        )
        return (
            f"{thickness_nm:.2f} nm mean ({thickness_a:.1f} A); "
            f"{model} sigma {sigma_frac * thickness_nm:.2f} nm"
        )
    return f"{thickness_nm:.2f} nm ({thickness_a:.1f} A); {model}"


def _quick_dataset_nuisance_table_lines(contracts: list[Mapping[str, object]]) -> list[str]:
    """Render exact per-crystal nuisance contracts for a combined run."""

    if not contracts:
        return []
    lines = [
        "### Per-Dataset Nuisance Contracts",
        "",
        "The crystals retain independent thickness, scale, detector, absorption, and rocking contracts. Their arithmetic mean is not a physical thickness distribution.",
        "",
        "| dataset | thickness center/support | gradient | frame offset | group width / detector bins | absorption | Sg jitter | row-resolution B | scales | native WGHT a/b |",
        "|---|---|---:|---:|---|---|---|---:|---:|---|",
    ]
    for contract in contracts:
        nuisance = contract.get("nuisance") if isinstance(contract.get("nuisance"), Mapping) else {}
        thickness_a = safe_float(nuisance.get("thickness_A"), float("nan"))
        grad = safe_float(nuisance.get("thickness_grad_frac"), float("nan"))
        model = str(nuisance.get("thickness_model") or "single")
        if math.isfinite(thickness_a) and model == "uniform_gradient" and math.isfinite(grad):
            low = max(0.0, thickness_a * (1.0 - grad)) / 10.0
            high = max(0.0, thickness_a * (1.0 + grad)) / 10.0
            thickness_text = f"{thickness_a / 10.0:.2f} nm; support {low:.2f}-{high:.2f} nm"
        elif math.isfinite(thickness_a):
            thickness_text = f"{thickness_a / 10.0:.2f} nm; {model}"
        else:
            thickness_text = "n/a"
        native_weight = contract.get("native_weight") if isinstance(contract.get("native_weight"), Mapping) else {}
        a = nuisance.get("native_wght_a", native_weight.get("a"))
        b = nuisance.get("native_wght_b", native_weight.get("b"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(contract.get("dataset") or "n/a"),
                    thickness_text,
                    _quick_number(grad, 3),
                    _quick_number(nuisance.get("frame_offset"), 3),
                    f"{_quick_number(nuisance.get('group_width_deg'), 3)} deg / {nuisance.get('detector_scale_bins', 'n/a')}",
                    f"{nuisance.get('absorption_mode', 'n/a')}; frac {_quick_number(nuisance.get('absorption_frac'), 4)}",
                    str(nuisance.get("sg_jitter_mode", "n/a")),
                    _quick_number(nuisance.get("row_resolution_b"), 4),
                    str(contract.get("scale_count", "n/a")),
                    f"{_quick_number(a, 6)} / {_quick_number(b, 6)}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _format_terminal_finish_summary(payload: Mapping[str, object]) -> str:
    """Return the short human-facing completion report printed by auto/run/finish."""

    status = str(payload.get("status", "ok"))
    out_dir = str(payload.get("out_dir", ""))
    if status == "dry_run":
        return "\n".join(
            [
                "",
                "GROND dry run completed successfully.",
                f"Output directory: {out_dir}",
                "Commands were planned, but no scientific stages ran and no endpoint statistics were generated.",
            ]
        )
    final_hand = str(payload.get("final_hand", "n/a"))
    transform = str(payload.get("selected_transform", "n/a"))
    t_all = payload.get("final_hand_t_style_all", payload.get("final_hand_z_all"))
    t_free = payload.get("final_hand_t_style_free", payload.get("final_hand_z_free"))
    t_all_n = payload.get("final_hand_t_style_all_n")
    t_free_n = payload.get("final_hand_t_style_free_n")
    t_all_df = payload.get("final_hand_t_style_all_df")
    t_free_df = payload.get("final_hand_t_style_free_df")
    per_dataset_hand = (
        payload.get("final_hand_per_dataset")
        if isinstance(payload.get("final_hand_per_dataset"), Mapping)
        else {}
    )
    input_model = payload.get("input_model") if isinstance(payload.get("input_model"), Mapping) else {}
    input_shelxl = payload.get("input_shelxl") if isinstance(payload.get("input_shelxl"), Mapping) else {}
    final_row = payload.get("final_row_model") if isinstance(payload.get("final_row_model"), Mapping) else {}
    jana = (
        payload.get("jana_pets_row_statistics")
        if isinstance(payload.get("jana_pets_row_statistics"), Mapping)
        else {}
    )
    jana_dynamic = jana.get("dynamic") if isinstance(jana.get("dynamic"), Mapping) else {}
    jana_born = jana.get("born") if isinstance(jana.get("born"), Mapping) else {}
    if not _quick_complete_jana_metrics(jana_dynamic):
        jana_dynamic = {}
    if not _quick_complete_jana_metrics(jana_born):
        jana_born = {}
    feedback = payload.get("final_feedback_endpoint") if isinstance(payload.get("final_feedback_endpoint"), Mapping) else {}
    endpoint = payload.get("publication_endpoint") if isinstance(payload.get("publication_endpoint"), Mapping) else {}
    diagnostic = payload.get("diagnostic_endpoint") if isinstance(payload.get("diagnostic_endpoint"), Mapping) else {}
    endpoint_available = _quick_fraction(endpoint.get("R1all")) is not None
    endpoint_warning = _quick_goof_warning(endpoint.get("GooF")) if endpoint_available else ""
    files = payload.get("files") if isinstance(payload.get("files"), Mapping) else {}
    finished = payload.get("finished") if isinstance(payload.get("finished"), Mapping) else {}
    timing_context = payload.get("timing_context")
    timing_context = timing_context if isinstance(timing_context, Mapping) else {}
    timing_kind = str(timing_context.get("kind", "workflow_invocation"))
    finish_elapsed = (
        format_seconds(timing_context.get("finish_invocation_elapsed_seconds"))
        if timing_context.get("finish_invocation_elapsed_seconds") is not None
        else ""
    )
    invocation_elapsed = (
        str(finished.get("invocation_elapsed") or format_seconds(finished.get("invocation_elapsed_seconds")))
        if finished.get("invocation_elapsed_seconds") is not None
        else ""
    )
    cumulative_elapsed = str(
        finished.get("cumulative_stage_elapsed")
        or finished.get("stage_elapsed")
        or format_seconds(
            finished.get(
                "cumulative_stage_elapsed_seconds",
                finished.get("stage_elapsed_seconds", 0.0),
            )
        )
    )
    completed = finished.get("completed_stages", "n/a")
    estimated = finished.get("estimated_stages")
    stage_note = f"{completed} stages"
    if estimated not in {None, "", "n/a"}:
        try:
            if int(estimated) != int(completed):
                stage_note += f" completed; initial estimate was {estimated}"
            else:
                stage_note += " completed"
        except (TypeError, ValueError):
            stage_note += " completed"
    else:
        stage_note += " completed"
    if finished.get("failed_stages", 0):
        stage_note += f"; {finished.get('failed_stages')} failed stage records in progress file"
    if finished.get("completed_this_invocation") is not None:
        invocation_label = "recorded workflow invocation" if timing_kind == "finish_replay" else "this invocation"
        stage_note += f"; {finished.get('completed_this_invocation')} ran in {invocation_label}"
        if finished.get("reused_completed_stages", 0):
            prior_label = "prior workflow invocations" if timing_kind == "finish_replay" else "prior invocations"
            stage_note += f"; {finished.get('reused_completed_stages')} reused from {prior_label}"

    if timing_kind == "finish_replay":
        if status == "ok":
            headline = (
                f"GROND finish completed successfully in {finish_elapsed} wall time."
                if finish_elapsed
                else "GROND finish completed successfully."
            )
        else:
            headline = (
                f"GROND finish completed with status {status} after {finish_elapsed} wall time."
                if finish_elapsed
                else f"GROND finish completed with status {status}."
            )
    elif status == "ok":
        headline = (
            f"GROND finished successfully in {invocation_elapsed} wall time for this invocation."
            if invocation_elapsed
            else "GROND finished successfully."
        )
    else:
        headline = (
            f"GROND finished with status {status} after {invocation_elapsed} wall time for this invocation."
            if invocation_elapsed
            else f"GROND finished with status {status}."
        )
    if timing_kind == "finish_replay":
        recorded_wall = (
            f"recorded workflow wall time {invocation_elapsed}"
            if invocation_elapsed
            else "recorded workflow wall time unavailable"
        )
        timing_line = (
            f"Timing: {recorded_wall}; cumulative completed-stage runtime {cumulative_elapsed}. "
            "The cumulative value sums saved stage runtimes; concurrent stages overlap and resumed "
            "stages may be from earlier invocations, so it is not workflow wall time."
        )
    else:
        timing_line = (
            f"Timing: cumulative completed-stage runtime {cumulative_elapsed}. "
            "This sums saved stage runtimes; concurrent stages overlap and resumed stages may be from "
            "earlier invocations, so it is not workflow wall time."
        )
    publication_is_direct_ls1 = bool(files.get("publication_res") and files.get("publication_res") == files.get("final_feedback_res"))
    endpoint_lines = ["SHELXL endpoint" if publication_is_direct_ls1 else "SHELXL endpoints"]
    if not endpoint_available:
        endpoint_lines.extend(
            [
                "  No corrected endpoint passed the fresh conventional-baseline no-harm gate.",
                f"  Retain conventional baseline: R1(F)all {_quick_percent(input_shelxl.get('R1all'))}; RES {files.get('baseline_res', '')}; HKL {files.get('baseline_hkl', '')}",
            ]
        )
    elif publication_is_direct_ls1:
        endpoint_lines.append(
            f"  Publication endpoint (direct LS1 corrected-HKL): R1(F)gt {_quick_percent(endpoint.get('R1gt'))}, R1(F)all {_quick_percent(endpoint.get('R1all'))}, wR2(F^2)all {_quick_percent(endpoint.get('wR2all'))}, GooF {_quick_number(endpoint.get('GooF'), 3)}; {_quick_shelxl_issue_text(endpoint)}"
        )
    else:
        endpoint_lines.extend(
            [
                f"  Direct LS1 corrected-HKL candidate: R1(F)gt {_quick_percent(feedback.get('R1gt'))}, R1(F)all {_quick_percent(feedback.get('R1all'))}, wR2(F^2)all {_quick_percent(feedback.get('wR2all'))}, GooF {_quick_number(feedback.get('GooF'), 3)}; {_quick_shelxl_issue_text(feedback)}",
                f"  Publication endpoint: R1(F)gt {_quick_percent(endpoint.get('R1gt'))}, R1(F)all {_quick_percent(endpoint.get('R1all'))}, wR2(F^2)all {_quick_percent(endpoint.get('wR2all'))}, GooF {_quick_number(endpoint.get('GooF'), 3)}; {_quick_shelxl_issue_text(endpoint)}",
            ]
        )
    if diagnostic and diagnostic.get("R1all") is not None:
        reasons = diagnostic.get("rejection_reasons")
        first_reason = str(reasons[0]) if isinstance(reasons, list) and reasons else "diagnostic-only"
        endpoint_lines.append(
            f"  Diagnostic endpoint not promoted: {diagnostic.get('source_kind', 'n/a')} R1(F)all {_quick_percent(diagnostic.get('R1all'))}, wR2(F^2)all {_quick_percent(diagnostic.get('wR2all'))}, GooF {_quick_number(diagnostic.get('GooF'), 3)}; reason: {first_reason}"
        )
    if endpoint_available:
        endpoint_lines.append(
            f"  Endpoint source/scales: {_quick_endpoint_scale_text(endpoint)}, WGHT {endpoint.get('wght_mode', 'n/a')}"
        )
        if endpoint.get("selection_policy") or endpoint.get("information_criterion"):
            endpoint_lines.append(
                f"  Scale selector: {endpoint.get('selection_policy', 'n/a')} / {endpoint.get('information_criterion', 'n/a')} = {_quick_number(endpoint.get('information_value'), 3)}"
            )
    if endpoint_warning:
        endpoint_lines.append(f"  Warning: {endpoint_warning}")
    hand_scope = str(payload.get("final_hand_t_style_scope") or payload.get("final_hand_z_scope") or "")
    if hand_scope == "per_dataset":
        evidence = "; ".join(
            f"{name}=t {_quick_number((row or {}).get('t_all'), 3)}/{_quick_number((row or {}).get('t_free'), 3)} "
            f"(df {(row or {}).get('df_all', 'n/a')}/{(row or {}).get('df_free', 'n/a')})"
            for name, row in per_dataset_hand.items()
            if isinstance(row, Mapping)
        )
        hand_line = (
            f"  Hand: {final_hand} (per-dataset paired t-style all/free: "
            f"{evidence or 'diagnostic missing'}; no pooled statistic; "
            f"consistency {payload.get('final_hand_consistency_status') or 'missing'})"
        )
    else:
        hand_line = (
            f"  Hand: {final_hand} (paired t-style all/free = "
            f"{_quick_flat_hand_t_style_text(t_all, t_all_n, t_all_df)} / "
            f"{_quick_flat_hand_t_style_text(t_free, t_free_n, t_free_df)}; "
            "positive favors the selected hand; not a calibrated normal Z score)"
        )
    if jana_dynamic or jana_born:
        row_stat_lines = [
            "Final JANA/PETS-style row statistics (computed by GROND)",
            (
                "  Matched Born: "
                f"Nobs/Nall {_quick_count_pair(jana_born)}, "
                f"Robs {_quick_percent(jana_born.get('Robs'))}, Rall {_quick_percent(jana_born.get('Rall'))}, "
                f"wRall {_quick_percent(jana_born.get('wRall'))}; "
                f"MRobs {_quick_percent(jana_born.get('MRobs'))}, MRall {_quick_percent(jana_born.get('MRall'))}, "
                f"MwRall {_quick_percent(jana_born.get('MwRall'))}"
            ),
            (
                "  Bloch dynamical: "
                f"Nobs/Nall {_quick_count_pair(jana_dynamic)}, "
                f"Robs {_quick_percent(jana_dynamic.get('Robs'))}, Rall {_quick_percent(jana_dynamic.get('Rall'))}, "
                f"wRall {_quick_percent(jana_dynamic.get('wRall'))}; "
                f"MRobs {_quick_percent(jana_dynamic.get('MRobs'))}, MRall {_quick_percent(jana_dynamic.get('MRall'))}, "
                f"MwRall {_quick_percent(jana_dynamic.get('MwRall'))}"
            ),
        ]
    else:
        row_stat_lines = [
            "Legacy native-objective row diagnostics (not JANA/PETS publication factors)",
            f"  Input dynamical rows: R(I)all {_quick_percent(input_model.get('row_R1all'))}, wR(I)all {_quick_percent(input_model.get('row_wR2all'))}",
            f"  Input matched Born rows: R(I)all {_quick_percent(input_model.get('born_R1all'))}, wR(I)all {_quick_percent(input_model.get('born_wR2all'))}",
            f"  Final dynamical rows: R(I)all {_quick_percent(final_row.get('row_R1all'))}, wR(I)all {_quick_percent(final_row.get('row_wR2all'))}, free wR(I)all {_quick_percent(final_row.get('row_free_wR2all'))}",
            f"  Final matched Born rows: R(I)all {_quick_percent(final_row.get('born_R1all'))}, wR(I)all {_quick_percent(final_row.get('born_wR2all'))}, free wR(I)all {_quick_percent(final_row.get('born_free_wR2all'))}",
            f"  Change: R(I)all {_quick_delta_percent(input_model.get('row_R1all'), final_row.get('row_R1all'))}, wR(I)all {_quick_delta_percent(input_model.get('row_wR2all'), final_row.get('row_wR2all'))}",
        ]
    lines = [
        "",
        headline,
        f"Output directory: {out_dir}",
        f"Progress: {stage_note}. Optional gates can make the final stage count smaller than the starting ETA denominator.",
        timing_line,
        "",
        "Decision summary",
        hand_line,
        f"  HKL transform: {transform}",
        "",
        "Starting SHELXL reference",
        f"  R1(F)gt {_quick_percent(input_shelxl.get('R1gt'))}, R1(F)all {_quick_percent(input_shelxl.get('R1all'))}, wR2(F^2)all {_quick_percent(input_shelxl.get('wR2all'))}, GooF {_quick_number(input_shelxl.get('GooF'), 3)}",
        f"  Parameters/restraints {input_shelxl.get('parameters', 'n/a')} / {input_shelxl.get('restraints', 'n/a')}; NPD {input_shelxl.get('npd_count', 'n/a')}; split {input_shelxl.get('split_count', 'n/a')}",
        "",
        *row_stat_lines,
        "",
        *endpoint_lines,
        "",
        "Open these first",
        f"  Canonical Table 1: {payload.get('table1', '')}",
        f"  Table 1 JSON: {payload.get('table1_json', '')}",
        f"  Final summary: {payload.get('final_summary', '')}",
        f"  Quick stats: {payload.get('quick_stats', '')}",
        f"  Full JSON: {payload.get('full_json', '')}",
        f"  Publication model: {files.get('publication_res', '')}",
        f"  Publication HKL: {files.get('publication_hkl', '')}",
        f"  Row table(s): {final_row.get('row_table', '')}",
    ]
    return "\n".join(lines)


def write_quick_stats(prep: PrepConfig, result: Mapping[str, object], args: argparse.Namespace) -> dict:
    """Write a short human-first statistics summary at the run root."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    quick_path = out_dir / "QUICK_STATS.md"
    final_summary_path = out_dir / "FINAL_SUMMARY.md"
    row_metrics_path = out_dir / "tables" / "row_mr_metrics.json"
    row_metrics = try_read_json(row_metrics_path)
    endpoint_calibration = result.get("endpoint_calibration") if isinstance(result.get("endpoint_calibration"), Mapping) else {}
    endpoint_selection = endpoint_calibration.get("publication_selection", {}) if isinstance(endpoint_calibration, Mapping) else {}
    publication_endpoint = _publication_endpoint_from_result(result)
    diagnostic_endpoint = _chooser_diagnostic_endpoint(result)
    metadata = row_metrics.get("metadata", {}) if isinstance(row_metrics.get("metadata"), Mapping) else {}
    source_roots = _quick_progress_source_roots(out_dir)
    timing_rows = _quick_progress_timings(out_dir)
    finish_info = _quick_progress_finish_info(out_dir)
    sfac_check = try_read_json(out_dir / "sfac_check.json")
    final_run = _quick_final_run(result)
    final_model = Path(str(result.get("model"))) if result.get("model") else None
    final_summary = _quick_load_run_summary(final_run)
    dataset_nuisance_contracts = _quick_dataset_nuisance_contracts(final_summary)
    recorded_row_tables = _recorded_native_row_tables(result)
    missing_row_tables = {
        name: path
        for name, path in recorded_row_tables.items()
        if not Path(path).expanduser().is_file()
    }
    if missing_row_tables:
        missing_text = ", ".join(f"{name}={path}" for name, path in missing_row_tables.items())
        raise RuntimeError(
            "cannot write a coherent final report because a recorded native row table is missing; "
            "refusing to mix saved joint metrics with partial diagnostics: "
            + missing_text
        )
    frozen_row_tables = [
        (name, Path(path).expanduser())
        for name, path in recorded_row_tables.items()
    ]
    if not publication_endpoint:
        rejected_endpoint = _chooser_publication_endpoint(result) or _endpoint_calibration_publication_endpoint(result)
        rejected_gate = _endpoint_baseline_no_harm_gate(
            final_summary,
            rejected_endpoint,
            baseline_override=_validated_fresh_conventional_baseline(result) if result.get("combined") else None,
            require_baseline=bool(result.get("combined")),
        )
        if rejected_endpoint and str(rejected_gate.get("status")) == "failed":
            diagnostic_endpoint = dict(rejected_endpoint)
            diagnostic_endpoint["baseline_no_harm_gate"] = rejected_gate
            diagnostic_endpoint["rejection_reasons"] = list(rejected_gate.get("reasons") or [])
    best_native = _quick_best_native(final_summary)
    best_endpoint = _quick_best_endpoint(final_summary)
    native_metrics = best_native.get("metrics") if isinstance(best_native.get("metrics"), Mapping) else {}
    nuisance = _quick_nuisance(final_summary)
    decision_tree_path = Path(str(result.get("decision_tree", ""))) if result.get("decision_tree") else None
    if result.get("combined"):
        shared_path_text = str(result.get("shared_absolute_structure_decision") or "").strip()
        shared_decision = try_read_json(Path(shared_path_text)) if shared_path_text else {}
        decision_tree = {
            "selected_hand": str(result.get("selected_hand") or shared_decision.get("selected_hand") or "n/a"),
            "absolute_structure_decision": shared_decision,
        }
    elif decision_tree_path is not None:
        decision_tree = try_read_json(decision_tree_path)
    elif prep.datasets:
        decision_tree = try_read_json(out_dir / "decisions" / prep.datasets[0].name / "decision_tree.json")
    else:
        decision_tree = {}
    transform_decision = decision_tree.get("transform_decision") if isinstance(decision_tree.get("transform_decision"), Mapping) else {}
    if not transform_decision and prep.datasets:
        transform_decision = try_read_json(out_dir / "decisions" / prep.datasets[0].name / "transform_decision.json")
    transform_summary = _quick_transform_summary(transform_decision)
    per_dataset_transforms: dict[str, dict[str, object]] = {}
    if result.get("combined"):
        for ds in prep.datasets:
            transform_payload = try_read_json(out_dir / "decisions" / ds.name / "transform_decision.json")
            per_dataset_transforms[ds.name] = _quick_transform_summary(transform_payload)
        selected_transform_text = "; ".join(
            f"{name}={row.get('selected_transform', 'n/a')}"
            for name, row in per_dataset_transforms.items()
        )
        unique_transforms = {
            str(row.get("selected_transform"))
            for row in per_dataset_transforms.values()
            if str(row.get("selected_transform") or "n/a") != "n/a"
        }
        transform_summary = {
            **transform_summary,
            "selected_transform": next(iter(unique_transforms)) if len(unique_transforms) == 1 else selected_transform_text,
            "per_dataset": per_dataset_transforms,
            "per_dataset_text": selected_transform_text,
        }
    screen_run = Path(str(decision_tree.get("screen_run", ""))) if isinstance(decision_tree.get("screen_run"), str) else None
    screen_summary = try_read_json(screen_run / "summary.json") if screen_run else {}
    promotion = decision_tree.get("post_feedback_hand_promotion")
    promotion_decision = promotion if isinstance(promotion, Mapping) else {}
    diagnostic_path_text = (
        promotion_decision.get("diagnostic")
        or result.get("absolute_structure_diagnostic")
        or decision_tree.get("absolute_structure_diagnostic")
        or ""
    )
    hand_diagnostic = try_read_json(Path(str(diagnostic_path_text))) if diagnostic_path_text else {}
    hand_summary = _quick_final_hand_summary(decision_tree, hand_diagnostic)
    multi_hand_diagnostic = _quick_multi_hand_diagnostic(result) if result.get("combined") else {}
    if result.get("combined") and not multi_hand_diagnostic:
        multi_hand_diagnostic = {
            "status": "warning_missing_diagnostic",
            "artifact": str(out_dir / "decisions" / "shared_post_feedback_hand_diagnostic.json"),
            "datasets": {},
            "consistency_gate": {
                "status": "warning",
                "passed": False,
                "warnings": ["the shared post-feedback multi-dataset hand diagnostic is missing"],
            },
        }
    multi_hand_rows = _quick_multi_hand_console_rows(multi_hand_diagnostic)
    if multi_hand_diagnostic:
        hand_summary["promotion_status"] = multi_hand_diagnostic.get("status", "missing")
    z_summary = _quick_residual_z_summary([path for _name, path in frozen_row_tables])
    free_z = z_summary.get("free", {})
    all_z = z_summary.get("all", {})
    hand_blocks = hand_summary.get("diagnostic_blocks") if isinstance(hand_summary.get("diagnostic_blocks"), Mapping) else {}
    hand_all = hand_blocks.get("all", {}) if isinstance(hand_blocks.get("all"), Mapping) else {}
    hand_work = hand_blocks.get("work", {}) if isinstance(hand_blocks.get("work"), Mapping) else {}
    hand_free = hand_blocks.get("free", {}) if isinstance(hand_blocks.get("free"), Mapping) else {}
    file_block = _quick_endpoint_file_block(publication_endpoint, final_model)
    bloch_diagnostics_path = _write_bloch_diagnostics_reports(
        out_dir,
        frozen_row_tables,
        final_run,
        final_summary,
        row_metrics,
        file_block,
    )
    file_block["bloch_diagnostics"] = str(bloch_diagnostics_path)
    input_shelxl = _quick_baseline_shelxl(prep, final_summary, result)
    file_block["baseline_res"] = str(input_shelxl.get("res") or "")
    file_block["baseline_hkl"] = str(input_shelxl.get("hkl") or "")
    file_block["baseline_lst"] = str(input_shelxl.get("lst") or "")
    source_structure = Path(prep.structure).expanduser() if prep.structure else None
    source_controls = _quick_source_shelx_controls(source_structure)
    input_res_text = str(input_shelxl.get("res") or "").strip()
    input_res = Path(input_res_text).expanduser() if input_res_text else None
    input_controls = (
        _quick_source_shelx_controls(input_res)
        if input_res is not None and input_res.is_file()
        else source_controls
    )
    best_endpoint_shelxl = best_endpoint.get("shelxl") if isinstance(best_endpoint.get("shelxl"), Mapping) else {}
    final_feedback_shelxl = _quick_selected_feedback_shelxl(result, final_model, best_endpoint_shelxl)
    publication_shelxl = _quick_shelxl_details(
        {
            "R1_gt": publication_endpoint.get("shelxl_R1gt"),
            "R1_all": publication_endpoint.get("shelxl_R1all"),
            "wR2_all": publication_endpoint.get("shelxl_wR2all"),
            "GooF": publication_endpoint.get("shelxl_GooF"),
            "N_gt": publication_endpoint.get("shelxl_Ngt"),
            "N_all": publication_endpoint.get("shelxl_Nall"),
            "parameters": publication_endpoint.get("shelxl_parameters"),
            "restraints": publication_endpoint.get("shelxl_restraints"),
            "npd_count": publication_endpoint.get("shelxl_npd_count"),
            "npd_atoms": publication_endpoint.get("shelxl_npd_atoms"),
            "split_count": publication_endpoint.get("shelxl_split_count"),
            "has_npd": publication_endpoint.get("shelxl_has_npd"),
            "has_split_warnings": publication_endpoint.get("shelxl_has_split_warnings"),
            "pathology_status": publication_endpoint.get("pathology_status"),
            "active_wght_a": publication_endpoint.get("shelxl_active_wght_a"),
            "active_wght_b": publication_endpoint.get("shelxl_active_wght_b"),
            "recommended_wght_a": publication_endpoint.get("shelxl_recommended_wght_a"),
            "recommended_wght_b": publication_endpoint.get("shelxl_recommended_wght_b"),
            "lst": publication_endpoint.get("endpoint_lst"),
        },
        Path(file_block["publication_lst"]) if file_block.get("publication_lst") else None,
    )
    diagnostic_shelxl = _quick_shelxl_details(
        {
            "R1_gt": diagnostic_endpoint.get("shelxl_R1gt"),
            "R1_all": diagnostic_endpoint.get("shelxl_R1all"),
            "wR2_all": diagnostic_endpoint.get("shelxl_wR2all"),
            "GooF": diagnostic_endpoint.get("shelxl_GooF"),
            "parameters": diagnostic_endpoint.get("shelxl_parameters"),
            "restraints": diagnostic_endpoint.get("shelxl_restraints"),
            "npd_count": diagnostic_endpoint.get("shelxl_npd_count"),
            "split_count": diagnostic_endpoint.get("shelxl_split_count"),
            "lst": diagnostic_endpoint.get("endpoint_lst"),
        },
        Path(str(diagnostic_endpoint.get("endpoint_lst"))) if diagnostic_endpoint.get("endpoint_lst") else None,
    )
    endpoints = []
    resolution_block = row_metrics.get("row_resolution_scale_contracts")
    if not isinstance(resolution_block, Mapping) or not isinstance(
        resolution_block.get("contracts"), list
    ):
        resolution_block = row_metrics.get("resolution_scale_endpoints", {})
    if isinstance(resolution_block, Mapping):
        raw_endpoints = resolution_block.get("contracts", resolution_block.get("endpoints", []))
        endpoints = raw_endpoints if isinstance(raw_endpoints, list) else []
    row_count_text = str(metadata.get("row_count", "n/a"))
    source_control_text = _quick_source_shelx_control_text(input_controls)
    endpoint_status_text = (
        "corrected endpoint accepted"
        if publication_endpoint
        else "no corrected endpoint passed the fresh conventional-baseline no-harm gate"
    )
    frozen_row_file_lines = [
        f"- Frozen Bloch row table ({name}): `{path}`"
        for name, path in frozen_row_tables
    ] or ["- Frozen Bloch row tables: `none found`"]
    frozen_row_console_text = "; ".join(f"{name}={path}" for name, path in frozen_row_tables)
    final_row_summary_path = (
        final_run / ("combined_summary.json" if result.get("combined") else "summary.json")
        if final_run
        else None
    )
    final_row_table_sources = "; ".join(f"{name}={path}" for name, path in frozen_row_tables)
    table1 = _write_final_statistics_table1(
        out_dir,
        row_metrics=row_metrics,
        row_metrics_path=row_metrics_path,
        input_shelxl=input_shelxl,
        publication_shelxl=publication_shelxl,
        publication_endpoint=publication_endpoint,
        file_block=file_block,
        input_controls=input_controls,
        final_row_table=frozen_row_console_text,
    )
    jana_summary = _quick_final_jana_row_summary(row_metrics)
    jana_dynamic = jana_summary.get("dynamic") if isinstance(jana_summary.get("dynamic"), Mapping) else {}
    jana_born = jana_summary.get("born") if isinstance(jana_summary.get("born"), Mapping) else {}
    if not jana_born and isinstance(jana_summary.get("kinematic"), Mapping):
        jana_born = jana_summary.get("kinematic", {})
    if not _quick_complete_jana_metrics(jana_dynamic):
        jana_dynamic = {}
    if not _quick_complete_jana_metrics(jana_born):
        jana_born = {}
    file_block["table1_markdown"] = str(table1["markdown"])
    file_block["table1_json"] = str(table1["json"])
    jana_md_path = row_metrics_path.parent / "JANA_ROW_STATISTICS.md"
    jana_json_path = row_metrics_path.parent / "JANA_ROW_STATISTICS.json"
    per_dataset_jana_md_path = row_metrics_path.parent / "JANA_ROW_STATISTICS_PER_DATASET.md"
    per_dataset_jana_json_path = row_metrics_path.parent / "JANA_ROW_STATISTICS_PER_DATASET.json"
    file_block["jana_row_statistics"] = str(
        jana_md_path if jana_md_path.is_file() else table1["markdown"]
    )
    file_block["jana_row_statistics_json"] = str(
        jana_json_path if jana_json_path.is_file() else row_metrics_path
    )
    file_block["jana_row_statistics_per_dataset"] = (
        str(per_dataset_jana_md_path) if per_dataset_jana_md_path.is_file() else "not available for this historical run"
    )
    file_block["jana_row_statistics_per_dataset_json"] = (
        str(per_dataset_jana_json_path)
        if per_dataset_jana_json_path.is_file()
        else "not available for this historical run"
    )
    if result.get("combined"):
        hand_t_report_line = (
            "| final hand paired t-style, all/work/free | "
            "`per-dataset evidence with n and df below; no pooled statistic is computed` |"
        )
        hand_delta_report_line = (
            "| final hand sqrt(delta chi2), all/work/free | "
            "`per-dataset score margins below; no merged statistic is computed` |"
        )
    else:
        hand_t_report_line = (
            f"| final hand paired t-style, all/work/free | `{_quick_hand_t_style_text(hand_all)}` / "
            f"`{_quick_hand_t_style_text(hand_work)}` / "
            f"`{_quick_hand_t_style_text(hand_free)}`; positive favors final selected hand; "
            "not a calibrated normal Z score |"
        )
        hand_delta_report_line = (
            f"| final hand sqrt(delta chi2), all/work/free | `{_quick_number(hand_all.get('sqrt_positive_delta_chi2'), 3)}` / "
            f"`{_quick_number(hand_work.get('sqrt_positive_delta_chi2'), 3)}` / "
            f"`{_quick_number(hand_free.get('sqrt_positive_delta_chi2'), 3)}`; oriented as rejected-minus-final |"
        )

    lines = [
        "# GROND Quick Stats",
        "",
        "This is the short run summary. Use the detailed reports for diagnostics and audit trails.",
        "",
        "## Canonical Final Statistics",
        "",
        f"- Table 1, human-readable: `{table1['markdown']}`",
        f"- Table 1, machine-readable: `{table1['json']}`",
        f"- JANA/PETS-style row report: `{file_block['jana_row_statistics']}`",
        f"- Per-dataset JANA/PETS-style row report: `{file_block['jana_row_statistics_per_dataset']}`",
        "- Table 1 panel A contains only SHELXL endpoint statistics. Panel B contains only GROND-computed JANA/PETS-style row and post-refinement merged-row factors. Missing comparator classes remain explicit `n/a`; values are never substituted across panels.",
        "",
        "## Run",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Datasets: `{', '.join(ds.name for ds in prep.datasets) if prep.datasets else 'n/a'}`",
        f"- Final model: `{result.get('model', '')}`",
        f"- Publication endpoint model: `{publication_endpoint.get('endpoint_res', '')}`",
        f"- Publication corrected HKL: `{publication_endpoint.get('endpoint_hkl', '')}`",
        f"- Publication endpoint source: `{publication_endpoint.get('selection_source', 'n/a')}` / `{publication_endpoint.get('source_kind', 'n/a')}`",
        f"- Endpoint decision: `{endpoint_status_text}`",
        f"- Rows in final row table: `{metadata.get('row_count', 'n/a')}`",
        f"- Final N/Q: `{metadata.get('n_basis', 'n/a')}/{metadata.get('n_points', 'n/a')}`",
        f"- Effective thickness: `{_quick_thickness_text(nuisance, metadata, dataset_nuisance_contracts)}`",
        f"- Sg window: `{metadata.get('sg_window', 'n/a')}`; absorption: `{metadata.get('absorption_mode', 'n/a')}`; Sg jitter: `{metadata.get('sg_jitter_mode', 'n/a')}`",
        "",
    ]
    if len(source_roots) > 1:
        lines.extend(
            [
                "## Resume Warning",
                "",
                "This output folder contains completed stages from more than one GROND source tree. Treat it as a mixed-code resume, not a clean benchmark. Start a new output directory for final numbers.",
                "",
                "| source root | completed stages |",
                "|---|---:|",
            ]
        )
        for root, count in source_roots:
            lines.append(f"| `{root}` | {count} |")
        lines.append("")
    elif source_roots:
        root, count = source_roots[0]
        lines.extend(["## Source", "", f"- Completed stages were produced from `{root}` ({count} stages).", ""])

    if timing_rows:
        total_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in timing_rows if row.get("returncode") == 0)
        timing_summary_lines = []
        if finish_info.get("invocation_elapsed_seconds") is not None:
            timing_summary_lines.append(
                f"- Recorded workflow wall time: `{format_seconds(finish_info.get('invocation_elapsed_seconds'))}`"
            )
        timing_summary_lines.extend(
            [
                f"- Cumulative completed-stage runtime: `{format_seconds(total_elapsed)}`",
                "- The cumulative value sums saved stage runtimes. Concurrent stages overlap, and resumed stages may come from earlier invocations, so this is not workflow wall time.",
            ]
        )
        lines.extend(
            [
                "## Stage Timings",
                "",
                *timing_summary_lines,
                "",
                "| stage | returncode | elapsed | log |",
                "|---|---:|---:|---|",
            ]
        )
        for row in timing_rows:
            log = str(row.get("log", ""))
            lines.append(f"| `{row.get('stage', 'n/a')}` | {row.get('returncode', 'n/a')} | `{row.get('elapsed', 'n/a')}` | `{log}` |")
        lines.append("")

    lines.extend(
        [
            "## Starting SHELXL Reference",
            "",
            "GROND first reports the isolated one-cycle replay of the supplied starting RES/HKL, with inherited EXTI/WGHT and hash-bound inputs and outputs. Historical runs fall back to a healthy final-stage `baseline_exti` control or, last, the supplied model's sibling `.lst`/`.log`; those fallbacks are labeled in the table and do not satisfy the required multi-dataset publication no-harm gate.",
            "",
            "| RES | HKL | stats file found | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | parameters | restraints | pathology | WGHT |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            "| "
            + " | ".join(
                [
                    f"`{input_shelxl.get('res', '')}`",
                    f"`{input_shelxl.get('hkl', '')}`",
                    "`yes`" if input_shelxl.get("lst_exists") else "`no`",
                    _quick_percent(input_shelxl.get("R1_gt")),
                    _quick_percent(input_shelxl.get("R1_all")),
                    _quick_percent(input_shelxl.get("wR2_all")),
                    _quick_number(input_shelxl.get("GooF"), 3),
                    str(input_shelxl.get("parameters", "n/a")),
                    str(input_shelxl.get("restraints", "n/a")),
                    _quick_shelxl_issue_text(input_shelxl),
                    _quick_shelxl_weight_text(input_shelxl),
                ]
            )
            + " |",
            "",
        ]
    )

    lines.extend(
        [
            "## Input Contract, Hand, And Paired T-Style Diagnostics",
            "",
            "XDS, DIALS, and other integrators can produce different signed-row contracts from the same frames. Row filtering, partiality/Lorentz handling, scan-varying geometry, sigmas, and scale conventions can move nuisance fits such as thickness, frame offset, absorption, and scale counts. GROND therefore treats the selected HKL transform, final absolute-structure hand, and raw row-level Bloch/Born statistics as the most portable cross-pipeline checks; nuisance values are fit-contract parameters, not direct physical measurements unless the row contracts agree.",
            "",
            "| item | value |",
            "|---|---|",
            f"| SFAC / endpoint wavelength check | status `{sfac_check.get('status', 'n/a') if isinstance(sfac_check, Mapping) else 'n/a'}`; endpoint wavelength `{_quick_number((sfac_check.get('cell') or {}).get('endpoint_wavelength_A') if isinstance(sfac_check.get('cell'), Mapping) else None, 6)}` A |",
            f"| selected HKL transform | `{transform_summary.get('selected_transform', 'n/a')}` |",
            *(
                [f"| per-dataset HKL transforms | `{transform_summary.get('per_dataset_text', '')}` |"]
                if transform_summary.get("per_dataset_text")
                else []
            ),
            f"| transform scout status | `{transform_summary.get('selected_status', 'n/a')}` |",
            f"| metadata transform check | `{transform_summary.get('metadata_status', 'n/a')}`; log-I corr `{_quick_number(transform_summary.get('metadata_log_intensity_corr'), 4)}`; gap `{_quick_number(transform_summary.get('metadata_correlation_gap'), 4)}`; pairs `{transform_summary.get('metadata_paired_rows', 'n/a')}`; unique overlap `{transform_summary.get('metadata_unique_overlap', 'n/a')}` |",
            f"| initial hand scout | `{hand_summary.get('initial_hand', 'n/a')}`; delta original-minus-alternate `{_quick_number(hand_summary.get('initial_score_delta_original_minus_alternate'), 4)}` |",
            f"| final hand | `{hand_summary.get('final_hand', 'n/a')}`; post-feedback status `{hand_summary.get('promotion_status', 'n/a')}` |",
            hand_t_report_line,
            hand_delta_report_line,
            f"| free-row I residual Z | n `{free_z.get('n', 'n/a')}`; RMSZ `{_quick_number(free_z.get('rms_z'), 3)}`; meanZ `{_quick_number(free_z.get('mean_z'), 3)}`; median absZ `{_quick_number(free_z.get('median_abs_z'), 3)}` |",
            f"| all-row I residual Z | n `{all_z.get('n', 'n/a')}`; RMSZ `{_quick_number(all_z.get('rms_z'), 3)}`; meanZ `{_quick_number(all_z.get('mean_z'), 3)}`; median absZ `{_quick_number(all_z.get('median_abs_z'), 3)}` |",
            "",
            *_quick_multi_hand_evidence_lines(multi_hand_diagnostic),
            "## Input Model Row Statistics",
            "",
            "These are the first adaptive-screen row-space statistics for the supplied start model after GROND has selected the row contract used by the run. They are the practical input-model comparator, not final endpoint statistics.",
            "",
            "| model state | dyn R(I)all | dyn wR(I)all | dyn free wR(I)all | matched Born R(I)all | matched Born wR(I)all |",
            "|---|---:|---:|---:|---:|---:|",
            f"| input adaptive screen | `{_quick_percent(_quick_dyn_metric(screen_summary, 'all', 'R1_all'))}` | `{_quick_percent(_quick_dyn_metric(screen_summary, 'all', 'wR2_all'))}` | `{_quick_percent(_quick_dyn_metric(screen_summary, 'free', 'wR2_all'))}` | `{_quick_percent(_quick_born_metric(screen_summary, 'all', 'R1_all'))}` | `{_quick_percent(_quick_born_metric(screen_summary, 'all', 'wR2_all'))}` |",
            "",
            "## Final JANA/PETS-Style Row Statistics",
            "",
            "These are the publication-facing row factors computed from the final frozen signed-row tables. `Robs`/`Rall` and `wRobs`/`wRall` are unmerged-row statistics; `MRobs`/`MRall` and `MwRobs`/`MwRall` are recomputed after arithmetic symmetry merging. The observed subset is `Iobs > 3 sigma(Iobs)`. These values are not SHELXL endpoint R factors.",
            "",
            "| model | Ninput/excluded | Nobs/Nall | Nmerge used/excluded | MNobs/MNall | Robs | Rall | wRobs | wRall | MRobs | MRall | MwRobs | MwRall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| matched Born | {_table1_pair(jana_born, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_born)} | {_table1_pair(jana_born, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_born, 'MNobs', 'MNall')} | {_quick_percent(jana_born.get('Robs'))} | {_quick_percent(jana_born.get('Rall'))} | {_quick_percent(jana_born.get('wRobs'))} | {_quick_percent(jana_born.get('wRall'))} | {_quick_percent(jana_born.get('MRobs'))} | {_quick_percent(jana_born.get('MRall'))} | {_quick_percent(jana_born.get('MwRobs'))} | {_quick_percent(jana_born.get('MwRall'))} |",
            f"| Bloch dynamical | {_table1_pair(jana_dynamic, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_dynamic)} | {_table1_pair(jana_dynamic, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_dynamic, 'MNobs', 'MNall')} | {_quick_percent(jana_dynamic.get('Robs'))} | {_quick_percent(jana_dynamic.get('Rall'))} | {_quick_percent(jana_dynamic.get('wRobs'))} | {_quick_percent(jana_dynamic.get('wRall'))} | {_quick_percent(jana_dynamic.get('MRobs'))} | {_quick_percent(jana_dynamic.get('MRall'))} | {_quick_percent(jana_dynamic.get('MwRobs'))} | {_quick_percent(jana_dynamic.get('MwRall'))} |",
            "",
            "## Legacy Native-Objective Audit Matrix (Not Table 1)",
            "",
            "This retained engineering matrix contains GROND's historical native-objective `R1_all`/`wR2_all` diagnostics alongside endpoint diagnostics. Those internal row-objective names are not JANA/PETS publication factors and this matrix is not the manuscript table. Use `TABLE1_FINAL_STATISTICS.md`: its SHELXL and JANA/PETS panels have authoritative labels, targets, and denominators.",
            "",
            "| object | target class | Rgt | Rall | weighted R | free/held-out weighted R | GooF | N/reflections | parameters/scales | source and notes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
            "| "
            + " | ".join(
                [
                    "input conventional SHELXL",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(input_shelxl.get("R1_gt")),
                    _quick_percent(input_shelxl.get("R1_all")),
                    _quick_percent(input_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(input_shelxl.get("GooF"), 3),
                    str(input_shelxl.get("N_all", "n/a")),
                    f"{input_shelxl.get('parameters', 'n/a')} params; {input_shelxl.get('restraints', 'n/a')} restraints",
                    f"{source_control_text}; {_quick_shelxl_issue_text(input_shelxl)}; `{input_shelxl.get('lst', '')}`",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "input matched Born rows",
                    "raw signed rows (I), matched Born",
                    "n/a",
                    _quick_percent(_quick_born_metric(screen_summary, "all", "R1_all")),
                    _quick_percent(_quick_born_metric(screen_summary, "all", "wR2_all")),
                    _quick_percent(_quick_born_metric(screen_summary, "free", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "screen nuisance/scales",
                    "`adaptive_screen summary.json`; same signed rows as input dynamical comparator",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "input dynamical rows",
                    "raw signed rows (I), Bloch dynamical",
                    "n/a",
                    _quick_percent(_quick_dyn_metric(screen_summary, "all", "R1_all")),
                    _quick_percent(_quick_dyn_metric(screen_summary, "all", "wR2_all")),
                    _quick_percent(_quick_dyn_metric(screen_summary, "free", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "screen nuisance/scales",
                    "`adaptive_screen summary.json`; supplied model under first accepted GROND row contract",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final matched Born rows",
                    "raw signed rows (I), matched Born",
                    "n/a",
                    _quick_percent(_quick_metric(native_metrics, "all", "born", "R1_all")),
                    _quick_percent(_quick_metric(native_metrics, "all", "born", "wR2_all")),
                    _quick_percent(_quick_metric(native_metrics, "free", "born", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "final nuisance/scales",
                    f"`{final_row_summary_path or ''}`; Born comparator at final model",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final dynamical rows",
                    "raw signed rows (I), Bloch dynamical",
                    "n/a",
                    _quick_percent(_quick_metric(native_metrics, "all", "dyn", "R1_all")),
                    _quick_percent(_quick_metric(native_metrics, "all", "dyn", "wR2_all")),
                    _quick_percent(_quick_metric(native_metrics, "free", "dyn", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "final nuisance/scales",
                    f"`{final_row_table_sources}`; final raw-row statistic",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "direct LS1 corrected-HKL candidate",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(final_feedback_shelxl.get("R1_gt")),
                    _quick_percent(final_feedback_shelxl.get("R1_all")),
                    _quick_percent(final_feedback_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(final_feedback_shelxl.get("GooF"), 3),
                    str(final_feedback_shelxl.get("N_all", "n/a")),
                    f"{final_feedback_shelxl.get('parameters', 'n/a')} params; {final_feedback_shelxl.get('restraints', 'n/a')} restraints",
                    f"{_quick_shelxl_issue_text(final_feedback_shelxl)}; `{file_block['final_feedback_lst']}`",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "best diagnostic endpoint, not promoted",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(diagnostic_shelxl.get("R1_gt")),
                    _quick_percent(diagnostic_shelxl.get("R1_all")),
                    _quick_percent(diagnostic_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(diagnostic_shelxl.get("GooF"), 3),
                    str(diagnostic_shelxl.get("N_all", "n/a")),
                    f"{diagnostic_shelxl.get('parameters', 'n/a')} params; {diagnostic_shelxl.get('restraints', 'n/a')} restraints",
                    f"{_quick_shelxl_issue_text(diagnostic_shelxl)}; `{diagnostic_endpoint.get('endpoint_lst', '')}`",
                ]
            )
            + " |" if diagnostic_endpoint else "",
            "| "
            + " | ".join(
                [
                    "publication corrected-HKL endpoint",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(publication_shelxl.get("R1_gt")),
                    _quick_percent(publication_shelxl.get("R1_all")),
                    _quick_percent(publication_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(publication_shelxl.get("GooF"), 3),
                    str(publication_shelxl.get("N_all", "n/a")),
                    f"{publication_endpoint.get('spec') or publication_endpoint.get('source_kind') or 'n/a'}; {publication_endpoint.get('scale_params', 'n/a')} scales; {publication_shelxl.get('parameters', 'n/a')} params",
                    f"{_quick_shelxl_issue_text(publication_shelxl)}; `{file_block['publication_lst']}`",
                ]
            )
            + " |",
            "",
            "## Legacy Native-Objective Delta Matrix (Not Table 1)",
            "",
            "Negative values are improvements for R factors. Deltas are only meaningful within the same target class.",
            "",
            "| comparison | target class | dRall | d weighted R | d free/held-out weighted R | interpretation |",
            "|---|---|---:|---:|---:|---|",
            "| "
            + " | ".join(
                [
                    "input dynamical rows minus input matched Born rows",
                    "raw signed rows",
                    _quick_delta_percent(
                        _quick_born_metric(screen_summary, "all", "R1_all"),
                        _quick_dyn_metric(screen_summary, "all", "R1_all"),
                    ),
                    _quick_delta_percent(
                        _quick_born_metric(screen_summary, "all", "wR2_all"),
                        _quick_dyn_metric(screen_summary, "all", "wR2_all"),
                    ),
                    _quick_delta_percent(
                        _quick_born_metric(screen_summary, "free", "wR2_all"),
                        _quick_dyn_metric(screen_summary, "free", "wR2_all"),
                    ),
                    "starting-model dynamical correction benefit over the matched Born control",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final dynamical rows minus final matched Born rows",
                    "raw signed rows",
                    _quick_delta_percent(
                        _quick_metric(native_metrics, "all", "born", "R1_all"),
                        _quick_metric(native_metrics, "all", "dyn", "R1_all"),
                    ),
                    _quick_delta_percent(
                        _quick_metric(native_metrics, "all", "born", "wR2_all"),
                        _quick_metric(native_metrics, "all", "dyn", "wR2_all"),
                    ),
                    _quick_delta_percent(
                        _quick_metric(native_metrics, "free", "born", "wR2_all"),
                        _quick_metric(native_metrics, "free", "dyn", "wR2_all"),
                    ),
                    "final model dynamical correction benefit over the final Born control",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final dynamical rows minus input dynamical rows",
                    "raw signed rows",
                    _quick_delta_percent(
                        _quick_dyn_metric(screen_summary, "all", "R1_all"),
                        _quick_metric(native_metrics, "all", "dyn", "R1_all"),
                    ),
                    _quick_delta_percent(
                        _quick_dyn_metric(screen_summary, "all", "wR2_all"),
                        _quick_metric(native_metrics, "all", "dyn", "wR2_all"),
                    ),
                    _quick_delta_percent(
                        _quick_dyn_metric(screen_summary, "free", "wR2_all"),
                        _quick_metric(native_metrics, "free", "dyn", "wR2_all"),
                    ),
                    "net row-level movement from the supplied model to the final model",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "publication endpoint minus input conventional SHELXL",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_delta_percent(input_shelxl.get("R1_all"), publication_shelxl.get("R1_all")),
                    _quick_delta_percent(input_shelxl.get("wR2_all"), publication_shelxl.get("wR2_all")),
                    "n/a",
                    "headline conventional endpoint change; compare only with matching reflection coverage caveats",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "direct LS1 endpoint minus input conventional SHELXL",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_delta_percent(input_shelxl.get("R1_all"), final_feedback_shelxl.get("R1_all")),
                    _quick_delta_percent(input_shelxl.get("wR2_all"), final_feedback_shelxl.get("wR2_all")),
                    "n/a",
                    "unreplayed corrected-HKL endpoint change before publication scale selection",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "publication endpoint minus direct LS1 endpoint",
                    "merged SHELXL endpoint",
                    _quick_delta_percent(final_feedback_shelxl.get("R1_all"), publication_shelxl.get("R1_all")),
                    _quick_delta_percent(final_feedback_shelxl.get("wR2_all"), publication_shelxl.get("wR2_all")),
                    "n/a",
                    "cost or gain of the publication endpoint replay relative to the direct LS1 candidate",
                ]
            )
            + " |",
            "",
            "## Row Intensity R-Factor Definitions",
            "",
            _quick_row_weight_definition(nuisance, dataset_nuisance_contracts),
            "",
            "## Final Nuisance Contract",
            "",
            "| field | value |",
            "|---|---|",
            *(
                [
                    f"| thickness | `{_quick_thickness_text(nuisance, metadata, dataset_nuisance_contracts)}` |",
                    "| native row-weight mode | `per_dataset_native`; no active joint a/b pair; exact per-dataset values follow |",
                ]
                if dataset_nuisance_contracts
                else [
                    f"| thickness | `{_quick_thickness_text(nuisance, metadata)}` |",
                    f"| frame and rocking | frame offset `{_quick_number(nuisance.get('frame_offset', nuisance.get('frame_offset_grid')), 3)}`; Sg window `{metadata.get('sg_window', 'n/a')}`; matrix `{nuisance.get('matrix_convention', 'n/a')}` |",
                    f"| absorption and jitter | absorption `{nuisance.get('absorption_mode', nuisance.get('absorption_mode_grid', 'n/a'))}`; frac `{_quick_number(nuisance.get('absorption_frac', nuisance.get('absorption_frac_grid')), 4)}`; jitter `{nuisance.get('sg_jitter_mode', nuisance.get('sg_jitter_mode_grid', 'n/a'))}` |",
                    f"| native row weights | a `{_quick_number(nuisance.get('native_wght_a', nuisance.get('wght_a_grid')), 6)}`; b `{_quick_number(nuisance.get('native_wght_b', nuisance.get('wght_b_grid')), 6)}` |",
                ]
            ),
            f"| row dyn R(I)all / wR(I)all / free wR(I)all | `{_quick_percent(_quick_metric(native_metrics, 'all', 'dyn', 'R1_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'all', 'dyn', 'wR2_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'free', 'dyn', 'wR2_all'))}` |",
            f"| row Born R(I)all / wR(I)all / free wR(I)all | `{_quick_percent(_quick_metric(native_metrics, 'all', 'born', 'R1_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'all', 'born', 'wR2_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'free', 'born', 'wR2_all'))}` |",
            f"| q p05 / median / p95 | `{_quick_number(best_endpoint.get('q_p05'), 3)}` / `{_quick_number(best_endpoint.get('q_median'), 3)}` / `{_quick_number(best_endpoint.get('q_p95'), 3)}` |",
            "",
            *_quick_dataset_nuisance_table_lines(dataset_nuisance_contracts),
        ]
    )

    lines.extend(
        [
            "## JANA/PETS-Style Row And Post-Refinement Merged Statistics By Resolution/Scale Contract",
            "",
            "These GROND-computed factors follow the JANA/PETS row-statistics family and remain separate from endpoint refinement. `Nobs/Nall` is the accepted observed-row coverage at that resolution gate; `Rall` and `wRall` use unmerged signed rows, while `MRall` and `MwRall` are recomputed after post-refinement symmetry merging. They are not SHELXL least-squares statistics.",
            "",
            "| endpoint | dmin A | scales | Nobs/Nall | dyn Rall | dyn wRall | dyn MRall | dyn MwRall | kin Rall | kin wRall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if endpoints:
        for endpoint in endpoints:
            if not isinstance(endpoint, Mapping):
                continue
            row_mr = endpoint.get("row_mr", {}) if isinstance(endpoint.get("row_mr"), Mapping) else {}
            dyn = row_mr.get("dynamic", {}) if isinstance(row_mr.get("dynamic"), Mapping) else {}
            kin = row_mr.get("born", {}) if isinstance(row_mr.get("born"), Mapping) else {}
            if not kin and isinstance(row_mr.get("kinematic"), Mapping):
                kin = row_mr.get("kinematic", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(endpoint.get("label", "n/a")),
                        _quick_number(endpoint.get("dmin_A"), 3),
                        str(endpoint.get("scale_params", "n/a")),
                        _quick_count_pair(dyn),
                        _quick_percent(dyn.get("Rall")),
                        _quick_percent(dyn.get("wRall")),
                        _quick_percent(dyn.get("MRall")),
                        _quick_percent(dyn.get("MwRall")),
                        _quick_percent(kin.get("Rall")),
                        _quick_percent(kin.get("wRall")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    selection_policy = str(endpoint_selection.get("selection_policy", "n/a")) if isinstance(endpoint_selection, Mapping) else "n/a"
    information_criterion = str(endpoint_selection.get("information_criterion", "n/a")) if isinstance(endpoint_selection, Mapping) else "n/a"
    lines.extend(
        [
            "",
            "## Publication Endpoint",
            "",
            "This is the endpoint GROND currently recommends opening for the final structure. The endpoint-calibration table below remains the conservative scale-replay audit.",
            "",
            "| source | kind/spec | WGHT | scales | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | params/restraints | pathology |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
            "| "
            + " | ".join(
                [
                    str(publication_endpoint.get("selection_source", "n/a")),
                    str(publication_endpoint.get("spec") or publication_endpoint.get("source_kind") or "n/a"),
                    str(publication_endpoint.get("wght_mode", "n/a")),
                    str(publication_endpoint.get("scale_params", "n/a")),
                    _quick_percent(publication_shelxl.get("R1_gt")),
                    _quick_percent(publication_shelxl.get("R1_all")),
                    _quick_percent(publication_shelxl.get("wR2_all")),
                    _quick_number(publication_shelxl.get("GooF"), 3),
                    f"{publication_shelxl.get('parameters', 'n/a')} / {publication_shelxl.get('restraints', 'n/a')}",
                    _quick_shelxl_issue_text(publication_shelxl),
                ]
            )
            + " |",
            "",
            (
                "**Endpoint warning:** "
                + _quick_goof_warning(publication_shelxl.get("GooF"))
                + "."
                if _quick_goof_warning(publication_shelxl.get("GooF"))
                else ""
            ),
            "## Endpoint Calibration",
            "",
            f"Status: `{endpoint_calibration.get('status', 'n/a') if isinstance(endpoint_calibration, Mapping) else 'n/a'}`. {endpoint_calibration.get('reason', '') if isinstance(endpoint_calibration, Mapping) else ''}",
            "",
            f"Selection policy: `{selection_policy}`; information criterion: `{information_criterion}`.",
            "",
            "| branch | spec | WGHT | scales | R1(F)all | wR2(F^2)all | GooF | row-free wR(I)all | sigma scale |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    branch_rows = [
        ("selected_scale_replay", endpoint_selection.get("publication_endpoint") if isinstance(endpoint_selection, Mapping) else None),
        ("low_R_diagnostic", endpoint_selection.get("low_r_diagnostic") if isinstance(endpoint_selection, Mapping) else None),
        ("best_heldout_row", endpoint_selection.get("best_heldout_row") if isinstance(endpoint_selection, Mapping) else None),
    ]
    wrote_branch = False
    for label, row in branch_rows:
        if not isinstance(row, Mapping):
            continue
        wrote_branch = True
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row.get("spec", "n/a")),
                    str(row.get("wght_mode", "n/a")),
                    str(row.get("scale_params", "n/a")),
                    _quick_percent(row.get("shelxl_R1all")),
                    _quick_percent(row.get("shelxl_wR2all")),
                    _quick_number(row.get("shelxl_GooF"), 3),
                    _quick_percent(row.get("native_free_wR2_all")),
                    _quick_number(row.get("sigma_scale"), 3),
                ]
            )
            + " |"
        )
    if not wrote_branch:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Files To Open Next",
            "",
            f"- Canonical final Table 1: `{table1['markdown']}`",
            f"- Machine-readable final Table 1: `{table1['json']}`",
            f"- Combined JANA/PETS-style row factors: `{file_block['jana_row_statistics']}`",
            f"- Per-dataset JANA/PETS-style row factors: `{file_block['jana_row_statistics_per_dataset']}`",
            f"- Fresh conventional baseline RES to retain when no corrected endpoint passes: `{file_block['baseline_res']}`",
            f"- Fresh conventional baseline HKL: `{file_block['baseline_hkl']}`",
            f"- Fresh conventional baseline SHELXL LST: `{file_block['baseline_lst']}`",
            f"- Final LS1 feedback endpoint RES: `{file_block['final_feedback_res']}`",
            f"- Final LS1 feedback corrected HKL: `{file_block['final_feedback_hkl']}`",
            f"- Final LS1 feedback SHELXL LST: `{file_block['final_feedback_lst']}`",
            f"- Final LS1 feedback manifest: `{file_block['final_feedback_manifest']}`",
            f"- Publication endpoint RES to open in SHELXLE/Olex2: `{file_block['publication_res']}`",
            f"- Publication endpoint INS used for the reported run: `{file_block['publication_ins']}`",
            f"- Publication corrected HKL used by that endpoint: `{file_block['publication_hkl']}`",
            f"- Publication SHELXL LST with the reported R/GooF values: `{file_block['publication_lst']}`",
            f"- Publication endpoint manifest: `{file_block['publication_manifest']}`",
            *frozen_row_file_lines,
            f"- Endpoint replay row table after compact scale selection: `{file_block['publication_row_table']}`",
            f"- Bloch matrix and row-contract diagnostics: `{bloch_diagnostics_path}`",
            f"- Detailed row/MR table: `{row_metrics_path}`",
            f"- Endpoint calibration report: `{endpoint_calibration.get('report', '') if isinstance(endpoint_calibration, Mapping) else ''}`",
            f"- Publication endpoint chooser: `{out_dir / 'endpoint_chooser' / 'PUBLICATION_ENDPOINT_CHOOSER.md'}`",
            "",
        ]
    )
    quick_path.write_text("\n".join(lines))

    selected_row = publication_endpoint if publication_endpoint else {}
    final_row_r1 = _quick_dyn_metric(final_summary, "all", "R1_all")
    final_row_wr2 = _quick_dyn_metric(final_summary, "all", "wR2_all")
    final_free_wr2 = _quick_dyn_metric(final_summary, "free", "wR2_all")
    input_row_r1 = _quick_dyn_metric(screen_summary, "all", "R1_all")
    input_row_wr2 = _quick_dyn_metric(screen_summary, "all", "wR2_all")
    console = {
        "out_dir": str(out_dir),
        "final_summary": str(final_summary_path),
        "quick_stats": str(quick_path),
        "full_json": str(out_dir / "grond_summary.json"),
        "final_hand": hand_summary.get("final_hand", "n/a"),
        "final_hand_t_style_all": (
            None
            if result.get("combined")
            else _quick_fraction(hand_all.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_t_style_all_n": None if result.get("combined") else _quick_int(hand_all.get("n")),
        "final_hand_t_style_all_df": None if result.get("combined") else _quick_int(hand_all.get("df")),
        "final_hand_t_style_free": (
            None
            if result.get("combined")
            else _quick_fraction(hand_free.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_t_style_free_n": None if result.get("combined") else _quick_int(hand_free.get("n")),
        "final_hand_t_style_free_df": None if result.get("combined") else _quick_int(hand_free.get("df")),
        "final_hand_t_style_scope": "per_dataset" if result.get("combined") else "single_dataset",
        "final_hand_z_all": (
            None
            if result.get("combined")
            else _quick_fraction(hand_all.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_z_free": (
            None
            if result.get("combined")
            else _quick_fraction(hand_free.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_z_scope": "per_dataset" if result.get("combined") else "single_dataset",
        "final_hand_z_compatibility_note": (
            "legacy field names only; values are paired Student-t-style contrasts, not calibrated normal Z scores"
        ),
        "final_hand_per_dataset": multi_hand_rows,
        "final_hand_consistency_status": multi_hand_diagnostic.get("status", "") if multi_hand_diagnostic else "",
        "selected_transform": transform_summary.get("selected_transform", "n/a"),
        "input_shelxl": {
            "R1gt": _quick_fraction(input_shelxl.get("R1_gt")),
            "R1all": _quick_fraction(input_shelxl.get("R1_all")),
            "wR2all": _quick_fraction(input_shelxl.get("wR2_all")),
            "GooF": _quick_fraction(input_shelxl.get("GooF")),
            "parameters": input_shelxl.get("parameters"),
            "restraints": input_shelxl.get("restraints"),
            "npd_count": input_shelxl.get("npd_count"),
            "split_count": input_shelxl.get("split_count"),
            "lst": input_shelxl.get("lst", ""),
        },
        "input_model": {
            "row_R1all": _quick_fraction(input_row_r1),
            "row_wR2all": _quick_fraction(input_row_wr2),
            "born_R1all": _quick_fraction(_quick_born_metric(screen_summary, "all", "R1_all")),
            "born_wR2all": _quick_fraction(_quick_born_metric(screen_summary, "all", "wR2_all")),
        },
        "final_row_model": {
            "row_R1all": _quick_fraction(final_row_r1),
            "row_wR2all": _quick_fraction(final_row_wr2),
            "row_free_wR2all": _quick_fraction(final_free_wr2),
            "born_R1all": _quick_fraction(_quick_metric(native_metrics, "all", "born", "R1_all")),
            "born_wR2all": _quick_fraction(_quick_metric(native_metrics, "all", "born", "wR2_all")),
            "born_free_wR2all": _quick_fraction(_quick_metric(native_metrics, "free", "born", "wR2_all")),
            "row_table": frozen_row_console_text,
            "row_tables": {name: str(path) for name, path in frozen_row_tables},
        },
        "final_feedback_endpoint": {
            "R1gt": _quick_fraction(final_feedback_shelxl.get("R1_gt")),
            "R1all": _quick_fraction(final_feedback_shelxl.get("R1_all")),
            "wR2all": _quick_fraction(final_feedback_shelxl.get("wR2_all")),
            "GooF": _quick_fraction(final_feedback_shelxl.get("GooF")),
            "parameters": final_feedback_shelxl.get("parameters"),
            "restraints": final_feedback_shelxl.get("restraints"),
            "pathology_status": final_feedback_shelxl.get("pathology_status", "unverified"),
            "npd_count": final_feedback_shelxl.get("npd_count"),
            "npd_atoms": final_feedback_shelxl.get("npd_atoms", []),
            "split_count": final_feedback_shelxl.get("split_count"),
            "has_npd": final_feedback_shelxl.get("has_npd"),
            "has_split_warnings": final_feedback_shelxl.get("has_split_warnings"),
            "lst": final_feedback_shelxl.get("lst", file_block["final_feedback_lst"]),
        },
        "publication_endpoint": {
            "R1gt": _quick_fraction(selected_row.get("shelxl_R1gt")),
            "R1all": _quick_fraction(selected_row.get("shelxl_R1all")),
            "wR2all": _quick_fraction(selected_row.get("shelxl_wR2all")),
            "GooF": _quick_fraction(selected_row.get("shelxl_GooF")),
            "scale_spec": selected_row.get("spec", ""),
            "scale_params": selected_row.get("scale_params", ""),
            "wght_mode": selected_row.get("wght_mode") or "n/a",
            "parameters": publication_shelxl.get("parameters"),
            "restraints": publication_shelxl.get("restraints"),
            "pathology_status": selected_row.get("pathology_status")
            or publication_shelxl.get("pathology_status", "unverified"),
            "npd_count": publication_shelxl.get("npd_count"),
            "npd_atoms": publication_shelxl.get("npd_atoms", []),
            "split_count": publication_shelxl.get("split_count"),
            "has_npd": publication_shelxl.get("has_npd"),
            "has_split_warnings": publication_shelxl.get("has_split_warnings"),
            "active_wght_a": publication_shelxl.get("active_wght_a"),
            "active_wght_b": publication_shelxl.get("active_wght_b"),
            "source": selected_row.get("selection_source", ""),
            "source_kind": selected_row.get("source_kind", ""),
            "selection_policy": endpoint_selection.get("selection_policy", "") if isinstance(endpoint_selection, Mapping) else "",
            "information_criterion": endpoint_selection.get("information_criterion", "") if isinstance(endpoint_selection, Mapping) else "",
            "information_value": endpoint_selection.get("selected_information_criterion", "") if isinstance(endpoint_selection, Mapping) else "",
        },
        "diagnostic_endpoint": {
            "R1gt": _quick_fraction(diagnostic_endpoint.get("shelxl_R1gt")),
            "R1all": _quick_fraction(diagnostic_endpoint.get("shelxl_R1all")),
            "wR2all": _quick_fraction(diagnostic_endpoint.get("shelxl_wR2all")),
            "GooF": _quick_fraction(diagnostic_endpoint.get("shelxl_GooF")),
            "source": diagnostic_endpoint.get("selection_source", ""),
            "source_kind": diagnostic_endpoint.get("source_kind", ""),
            "id": diagnostic_endpoint.get("id", ""),
            "rejection_reasons": diagnostic_endpoint.get("rejection_reasons", []),
        },
        "jana_pets_row_statistics": {
            "source": jana_summary.get("source", str(row_metrics_path)),
            "born": dict(jana_born),
            "dynamic": dict(jana_dynamic),
        },
        "table1": str(table1["markdown"]),
        "table1_json": str(table1["json"]),
        "files": file_block,
    }
    policy = str(console["publication_endpoint"].get("selection_policy") or "").strip()
    criterion = str(console["publication_endpoint"].get("information_criterion") or "").strip()
    criterion_value = console["publication_endpoint"].get("information_value")
    if not selected_row:
        endpoint_note = (
            "No endpoint chooser-approved physical endpoint was selected. Endpoint-calibration scale replays are diagnostic only for this run."
        )
    elif policy or criterion:
        endpoint_note = (
            f"Endpoint calibration used `{policy or 'n/a'}` with `{criterion or 'n/a'}` = "
            f"`{_quick_number(criterion_value, 3)}`. If the endpoint chooser selected an accepted observation-polish endpoint, "
            "that endpoint is reported above and the calibration replay remains an audit branch."
        )
    else:
        endpoint_note = (
            "The endpoint chooser selected the publication endpoint. Endpoint-calibration replay details, when present, "
            "are listed in `QUICK_STATS.md` as an audit branch."
        )
    endpoint_warning = _quick_goof_warning(publication_shelxl.get("GooF"))
    input_endpoint_available = _quick_fraction(input_shelxl.get("R1_all")) is not None
    publication_endpoint_available = _quick_fraction(publication_shelxl.get("R1_all")) is not None
    jana_born_available = _quick_complete_jana_metrics(jana_born)
    jana_dynamic_available = _quick_complete_jana_metrics(jana_dynamic)

    if result.get("combined"):
        final_hand_t_line = (
            f"| final hand paired t-style, all/free | `per-dataset only with n and df; no pooled statistic`; consistency "
            f"`{multi_hand_diagnostic.get('status', 'missing')}` |"
        )
    else:
        final_hand_t_line = (
            f"| final hand paired t-style, all/free | "
            f"`{_quick_flat_hand_t_style_text(console['final_hand_t_style_all'], console['final_hand_t_style_all_n'], console['final_hand_t_style_all_df'])}` / "
            f"`{_quick_flat_hand_t_style_text(console['final_hand_t_style_free'], console['final_hand_t_style_free_n'], console['final_hand_t_style_free_df'])}`; "
            "positive favors the selected hand; not a calibrated normal Z score |"
        )

    final_lines = [
        "# GROND Final Summary",
        "",
        "This is the compact user-facing summary. `QUICK_STATS.md` and the JSON reports keep the full audit trail.",
        "",
        "## Decision",
        "",
        "| item | value |",
        "|---|---|",
        f"| final hand | `{console['final_hand']}` |",
        final_hand_t_line,
        f"| selected HKL transform | `{console['selected_transform']}` |",
        f"| final N/Q | `{metadata.get('n_basis', 'n/a')}/{metadata.get('n_points', 'n/a')}` |",
        f"| effective thickness model | `{_quick_thickness_text(nuisance, metadata, dataset_nuisance_contracts)}`; Sg window `{metadata.get('sg_window', 'n/a')}` |",
        f"| publication endpoint source | `{console['publication_endpoint'].get('source') or 'none'}` / `{console['publication_endpoint'].get('source_kind') or 'no_physical_endpoint'}` |",
        "",
        *_quick_multi_hand_evidence_lines(multi_hand_diagnostic, compact=True),
        "## Canonical Statistics",
        "",
        f"The authoritative human-readable separation is in `{table1['markdown']}` and the machine-readable form is `{table1['json']}`. SHELXL endpoint factors and JANA/PETS-style row factors are never combined into generic R columns.",
        "",
        "### A. SHELXL Endpoint Statistics",
        "",
        "| model | status | Ngt/Nall | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | parameters/restraints | NPD/split |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| conventional kinematic reference | {'available' if input_endpoint_available else 'not available'} | {_table1_pair({'Ngt': input_shelxl.get('N_gt'), 'Nall': input_shelxl.get('N_all')}, 'Ngt', 'Nall')} | {_quick_percent(input_shelxl.get('R1_gt'))} | {_quick_percent(input_shelxl.get('R1_all'))} | {_quick_percent(input_shelxl.get('wR2_all'))} | {_quick_number(input_shelxl.get('GooF'), 3)} | {_table1_pair({'parameters': input_shelxl.get('parameters'), 'restraints': input_shelxl.get('restraints')}, 'parameters', 'restraints')} | {_table1_pair({'NPD': input_shelxl.get('npd_count'), 'split': input_shelxl.get('split_count')}, 'NPD', 'split')} |",
        "| GROND Born corrected-HKL endpoint | not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a |",
        f"| GROND Bloch dynamical publication endpoint | {'available' if publication_endpoint_available else 'not available'} | {_table1_pair({'Ngt': publication_shelxl.get('N_gt'), 'Nall': publication_shelxl.get('N_all')}, 'Ngt', 'Nall')} | {_quick_percent(publication_shelxl.get('R1_gt'))} | {_quick_percent(publication_shelxl.get('R1_all'))} | {_quick_percent(publication_shelxl.get('wR2_all'))} | {_quick_number(publication_shelxl.get('GooF'), 3)} | {_table1_pair({'parameters': publication_shelxl.get('parameters'), 'restraints': publication_shelxl.get('restraints')}, 'parameters', 'restraints')} | {_table1_pair({'NPD': publication_shelxl.get('npd_count'), 'split': publication_shelxl.get('split_count')}, 'NPD', 'split')} |",
        "",
        "### B. JANA/PETS-Style Row And Post-Refinement Merged Statistics",
        "",
        "| model | status | Ninput/excluded | Nobs/Nall | Nmerge used/excluded | MNobs/MNall | Robs | Rall | wRall | MRobs | MRall | MwRall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| conventional kinematic refinement | not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |",
        f"| GROND matched Born | {'available' if jana_born_available else 'not available'} | {_table1_pair(jana_born, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_born)} | {_table1_pair(jana_born, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_born, 'MNobs', 'MNall')} | {_quick_percent(jana_born.get('Robs'))} | {_quick_percent(jana_born.get('Rall'))} | {_quick_percent(jana_born.get('wRall'))} | {_quick_percent(jana_born.get('MRobs'))} | {_quick_percent(jana_born.get('MRall'))} | {_quick_percent(jana_born.get('MwRall'))} |",
        f"| GROND Bloch dynamical | {'available' if jana_dynamic_available else 'not available'} | {_table1_pair(jana_dynamic, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_dynamic)} | {_table1_pair(jana_dynamic, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_dynamic, 'MNobs', 'MNall')} | {_quick_percent(jana_dynamic.get('Robs'))} | {_quick_percent(jana_dynamic.get('Rall'))} | {_quick_percent(jana_dynamic.get('wRall'))} | {_quick_percent(jana_dynamic.get('MRobs'))} | {_quick_percent(jana_dynamic.get('MRall'))} | {_quick_percent(jana_dynamic.get('MwRall'))} |",
        "",
        "## Within-Family Deltas",
        "",
        "| comparison | dRall | d weighted R | dMRall | dMwRall |",
        "|---|---:|---:|---:|---:|",
        f"| Bloch minus matched Born, JANA/PETS-style rows | {_quick_delta_percent(jana_born.get('Rall'), jana_dynamic.get('Rall'))} | {_quick_delta_percent(jana_born.get('wRall'), jana_dynamic.get('wRall'))} | {_quick_delta_percent(jana_born.get('MRall'), jana_dynamic.get('MRall'))} | {_quick_delta_percent(jana_born.get('MwRall'), jana_dynamic.get('MwRall'))} |",
        f"| GROND publication minus conventional reference, SHELXL endpoints | {_quick_delta_percent(input_shelxl.get('R1_all'), publication_shelxl.get('R1_all'))} | {_quick_delta_percent(input_shelxl.get('wR2_all'), publication_shelxl.get('wR2_all'))} | n/a | n/a |",
        "",
        f"**Endpoint warning:** {endpoint_warning}." if endpoint_warning else "",
        "",
        endpoint_note,
        "",
        "## Files",
        "",
        f"- Canonical final Table 1: `{table1['markdown']}`",
        f"- Machine-readable final Table 1: `{table1['json']}`",
        f"- Combined JANA/PETS-style row report: `{file_block['jana_row_statistics']}`",
        f"- Per-dataset JANA/PETS-style row report: `{file_block['jana_row_statistics_per_dataset']}`",
        f"- Open this publication endpoint model: `{file_block['publication_res']}`",
        f"- Use this corrected HKL with that model: `{file_block['publication_hkl']}`",
        f"- The endpoint statistics above come from: `{file_block['publication_lst']}`",
        *frozen_row_file_lines,
        f"- Bloch matrix and row-contract diagnostics: `{bloch_diagnostics_path}`",
        f"- Full quick report: `{quick_path}`",
        f"- Full machine-readable run summary: `{out_dir / 'grond_summary.json'}`",
        "",
        "## Continue",
        "",
        f"- Checkpoint: `{out_dir / 'grond_continue_checkpoint.json'}`",
        f"- To push from an edited model: `GROND push-endpoint --checkpoint {out_dir / 'grond_continue_checkpoint.json'} --start-res edited.res --out-dir {out_dir / 'endpoint_push_edited'}`",
        f"- To regenerate this package: `GROND finish --run-dir {out_dir}`",
        "",
        "## Notes",
        "",
        "Native maps made by `GROND finish` use the frozen per-dataset GROND row table or tables, the final model phase, and dynamically inferred electron amplitudes. Use `--map-mode shelx2map` only for a fast corrected-HKL endpoint-map check.",
    ]
    final_summary_path.write_text("\n".join(final_lines) + "\n")
    return {"status": "ok", "path": str(quick_path), "final_summary": str(final_summary_path), "console": console}


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

    chooser_endpoint = _chooser_publication_endpoint(result)
    if chooser_endpoint:
        goof = chooser_endpoint.get("shelxl_GooF", chooser_endpoint.get("GooF"))
        if _quick_goof_warning(goof):
            return no_harm_checked(_endpoint_calibration_publication_endpoint(result))
        return no_harm_checked(chooser_endpoint)
    chooser = result.get("publication_endpoint_chooser") if isinstance(result.get("publication_endpoint_chooser"), Mapping) else {}
    chooser_status = str(chooser.get("status", "") if isinstance(chooser, Mapping) else "")
    if chooser and chooser_status not in {"", "disabled", "dry_run", "no_endpoint_runs"}:
        return {}
    return no_harm_checked(_endpoint_calibration_publication_endpoint(result))


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
    fcf = work_dir / f"{stem}.fcf"
    if not fcf.exists():
        return {"status": "missing_fcf", "fcf": str(fcf), "note": "Run the endpoint branch with SHELXL available so an FCF file is written."}
    exe = _shelx2map_executable(args)
    if ("/" in exe and not Path(exe).exists()) or ("/" not in exe and shutil.which(exe) is None):
        return {"status": "missing_shelx2map", "executable": exe}
    maps: dict[str, dict[str, object]] = {}
    for label, mode in [("fofc", "0"), ("fo", "1"), ("2fofc", "2")]:
        out_map = work_dir / f"grond_corrected_{label}.map"
        proc = subprocess.run(
            [exe, "-m", mode, "-o", out_map.name, stem],
            cwd=str(work_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path = work_dir / f"grond_finish_shelx2map_{label}.log"
        log_path.write_text(proc.stdout)
        actual = out_map
        for line in proc.stdout.splitlines():
            match = re.search(r'Map data written to "([^"]+)"', line)
            if match:
                actual = work_dir / match.group(1)
        maps[label] = {
            "map": str(actual),
            "requested_map": str(out_map),
            "log": str(log_path),
            "returncode": proc.returncode,
            "exists": actual.exists(),
        }
    return {
        "status": "ok",
        "definition": "GROND-corrected endpoint maps from the selected corrected-HKL publication endpoint; not native dynamical residual-potential maps.",
        "endpoint_res": str(endpoint_res.resolve()),
        "fcf": str(fcf.resolve()),
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
            raise RuntimeError(
                f"native-map row model for {label} is missing: {model}; recorded by {summary_path}"
            )
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
            "refusing a mixed verified/unverified map contract: missing "
            + ", ".join(missing_contracts)
        )
    if records:
        identities = {
            str(record.get("expected_sha256") or record.get("actual_sha256") or record.get("model"))
            for record in records
        }
        if len(identities) != 1:
            detail = "; ".join(
                f"{record['dataset']}={record['model']}@{record['actual_sha256']}" for record in records
            )
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


def run_finish_native_dynamic_maps(run_dir: Path, result: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
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
    """Run the requested finish-map mode."""

    if not getattr(args, "make_maps", True):
        return {"status": "disabled"}
    mode = str(getattr(args, "map_mode", "native") or "native").lower()
    if mode == "off":
        return {"status": "disabled"}
    maps: dict[str, object] = {"mode": mode}
    if mode in {"native", "both"}:
        maps["native_dynamic"] = run_finish_native_dynamic_maps(run_dir, result, args)
    if mode in {"shelx2map", "both"}:
        maps["shelx2map"] = run_finish_endpoint_maps(result, args)
    if mode not in {"native", "shelx2map", "both", "off"}:
        maps["status"] = "unknown_mode"
    else:
        maps["status"] = "ok"
    return maps


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
    manifest = {
        "status": "ok",
        "run_dir": str(run_dir),
        "summary": str(result_path),
        "prep": str(prep_path),
        "final_summary": quick.get("final_summary", ""),
        "quick_stats": quick.get("path", ""),
        "maps": maps,
        "cif": {
            "status": "native_dynamic_written" if native_cif else ("endpoint_exists" if cif is not None and cif.exists() else "endpoint_missing"),
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
        "status": "ok",
        **console,
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
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """CLI entrypoint that compile-checks the vendored Python engine."""

    python = choose_python(args.python)
    scripts = [
        engine_script("bloch_engine.py"),
        engine_script("native_row_regularize.py"),
        engine_script("continuous_thickness_polish.py"),
        engine_script("row_mr_report.py"),
        engine_script("resolution_limit_audit.py"),
        engine_script("residual_compass.py"),
        engine_script("premerge_rfactor_audit.py"),
        engine_script("native_two_part_observation_correction.py"),
        engine_script("native_observation_polish_endpoint.py"),
        engine_script("native_additive_bias_audit.py"),
        engine_script("native_dose_damage_audit.py"),
        engine_script("native_sigma_replicate_audit.py"),
        engine_script("run_raw_cbf_bloch_trial.py"),
        engine_script("build_dynrows_xds.py"),
        engine_script("dynrow_contract.py"),
        engine_script("support_basis_scan.py"),
        engine_script("ls1_validation_suite.py"),
        engine_script("native_structure_update.py"),
        engine_script("native_multi_structure_update.py"),
        engine_script("combine_native_nuisance_runs.py"),
        engine_script("run_shelxl_baseline.py"),
    ]
    cmd = [python, "-m", "py_compile", *map(str, scripts)]
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="grond-validate-pycache-") as pycache_prefix:
        env["PYTHONPYCACHEPREFIX"] = pycache_prefix
        subprocess.run(cmd, check=True, env=env)
    print(
        json.dumps(
            {
                "python": python,
                "compiled": len(scripts),
                "tool_config": getattr(args, "tool_config_path", ""),
                "external_tools": external_tool_statuses(args),
            },
            indent=2,
        )
    )
    return 0


def cmd_validate_ls1(args: argparse.Namespace) -> int:
    """Run the LS1 fixed-point validation suite for a pinned nuisance contract."""

    python = choose_python(args.python)
    cmd = [
        python,
        str(engine_script("ls1_validation_suite.py")),
        "--source-summary",
        args.source_summary,
        "--start-res",
        args.start_res,
        "--out-dir",
        args.out_dir,
        "--python",
        python,
        "--cycles",
        str(args.cycles),
        "--n-basis",
        str(args.n_basis),
        "--n-points",
        str(args.n_points),
        "--shelxl-iterations",
        str(args.shelxl_iterations),
        "--shelxl",
        args.shelxl,
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        args.row_plan_cache,
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--min-d-spacing",
        f"{args.min_d_spacing:g}",
        "--topup-d-min",
        f"{args.topup_d_min:g}",
        "--min-peak",
        f"{args.min_peak:g}",
        "--geometry-percentile",
        f"{args.geometry_percentile:g}",
        "--max-rows",
        str(args.max_rows),
        "--post-bloch-sigma-policy",
        args.post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--rescale",
        args.rescale,
        "--cache-compression",
        args.prediction_cache_compression,
        "--best-cycle-tolerance",
        f"{args.best_cycle_tolerance:g}",
        "--q-control-seed",
        str(args.q_control_seed),
        "--q-control-shells",
        str(args.q_control_shells),
        "--native-polish-rows",
        str(args.native_polish_rows),
        "--native-polish-n-basis",
        str(args.native_polish_n_basis),
        "--native-polish-n-points",
        str(args.native_polish_n_points),
        "--native-polish-blocks",
        args.native_polish_blocks,
        "--native-polish-atom-selection",
        args.native_polish_atom_selection,
        "--native-polish-max-nfev",
        str(args.native_polish_max_nfev),
        "--native-polish-early-stop-window",
        str(args.native_polish_early_stop_window),
        "--native-polish-early-stop-min-evals",
        str(args.native_polish_early_stop_min_evals),
        "--native-polish-early-stop-min-improvement",
        f"{args.native_polish_early_stop_min_improvement:g}",
        "--native-polish-free-wR2-tolerance",
        f"{args.native_polish_free_wR2_tolerance:g}",
    ]
    if args.hkl:
        cmd += ["--hkl", args.hkl]
    if args.baseline_hkl:
        cmd += ["--baseline-hkl", args.baseline_hkl]
    if args.hkl_transform:
        cmd += [f"--hkl-transform={args.hkl_transform}"]
    if args.skip_controls:
        cmd.append("--skip-controls")
    if not args.wrong_hand_control:
        cmd.append("--no-wrong-hand-control")
    if args.native_polish:
        cmd.append("--native-polish")
    if args.dry_run:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        plan = {"command": cmd, "tool_config": getattr(args, "tool_config_path", ""), "external_tools": external_tool_statuses(args)}
        (Path(args.out_dir) / "validate_ls1_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0
    subprocess.run(cmd, check=True)
    return 0


SHELX_RESTRAINT_COMMANDS = {
    "RIGU",
    "DFIX",
    "DANG",
    "SADI",
    "FLAT",
    "CHIV",
    "SAME",
    "DELU",
    "SIMU",
    "ISOR",
    "EADP",
    "EXYZ",
}


def is_shelx_atom_line(line: str) -> bool:
    """Return true for ordinary SHELX atom records."""

    toks = line.split()
    if len(toks) < 7:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", toks[0]):
        return False
    try:
        int(toks[1])
        [float(x) for x in toks[2:7]]
    except ValueError:
        return False
    return True


def shelx_label_element(label: str) -> str:
    """Return the leading element-like character from a SHELX atom label."""

    match = re.match(r"^[A-Za-z]+", label)
    return match.group(0).upper()[0] if match else ""


def prepare_push_endpoint_model(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path | None]:
    """Prepare the start model for an endpoint-push run."""

    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.start_res).expanduser().resolve()
    if args.model_prep == "as-is" and int(args.set_ls_cycles) <= 0:
        return source, None

    lines = source.read_text().splitlines()
    active: list[str] = []
    for line in lines:
        active.append(line)
        if line.strip().upper().startswith("END"):
            break

    cell = [1.0, 1.0, 1.0]
    for line in active:
        toks = line.split()
        if toks and toks[0].upper() == "CELL" and len(toks) >= 8:
            cell = [float(x) for x in toks[2:5]]
            break

    heavy: list[dict[str, object]] = []
    i = 0
    while i < len(active):
        line = active[i]
        if is_shelx_atom_line(line):
            toks = line.split()
            label = toks[0]
            if shelx_label_element(label) != "H" and line.rstrip().endswith("=") and i + 1 < len(active):
                cont = active[i + 1].split()
                try:
                    ueq = (float(toks[6]) + float(toks[7]) + float(cont[0])) / 3.0
                    heavy.append(
                        {
                            "label": label,
                            "xyz": tuple(float(x) % 1.0 for x in toks[2:5]),
                            "ueq": max(0.006, min(0.08, ueq)),
                        }
                    )
                except (ValueError, IndexError):
                    pass
        i += 1

    def distance2(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
        dx = ((p[0] - q[0] + 0.5) % 1.0) - 0.5
        dy = ((p[1] - q[1] + 0.5) % 1.0) - 0.5
        dz = ((p[2] - q[2] + 0.5) % 1.0) - 0.5
        return (dx * cell[0]) ** 2 + (dy * cell[1]) ** 2 + (dz * cell[2]) ** 2

    def nearest_heavy_u(xyz: tuple[float, float, float]) -> tuple[float, str | None]:
        if not heavy:
            return 0.03, None
        rec = min(heavy, key=lambda item: distance2(xyz, item["xyz"]))  # type: ignore[arg-type]
        return float(rec["ueq"]), str(rec["label"])

    remove_restraints = args.model_prep in {"no-restraints", "free-h-anis-no-restraints"}
    free_h_anis = args.model_prep == "free-h-anis-no-restraints"
    adp_stabilized = args.model_prep == "adp-stabilized"
    npd_diagnostics = _quick_lst_diagnostics(source.with_suffix(".lst"))
    if not npd_diagnostics.get("exists"):
        npd_diagnostics = _quick_lst_diagnostics(source.with_suffix(".log"))
    npd_targets = {str(atom) for atom in (npd_diagnostics.get("npd_atoms") or [])}
    existing_restraints = {line.split()[0].upper() for line in active if line.split() and line.split()[0].upper() in SHELX_RESTRAINT_COMMANDS}
    new_lines: list[str] = []
    removed_restraints: list[str] = []
    removed_afix = 0
    converted_h: list[dict[str, object]] = []
    adp_isotropic: list[dict[str, object]] = []
    adp_restraints_added: list[str] = []
    inserted_rem = False
    inserted_adp_stabilizers = False

    def add_adp_stabilizers() -> None:
        nonlocal inserted_adp_stabilizers
        if inserted_adp_stabilizers or not adp_stabilized:
            return
        new_lines.append("REM GROND endpoint-push ADP-stabilized replay branch")
        new_lines.append("REM GROND endpoint-push keeps/adds ADP restraints and converts listed NPD atoms to isotropic fallback")
        if npd_targets:
            new_lines.append("REM GROND endpoint-push NPD atoms: " + " ".join(sorted(npd_targets)))
        if "RIGU" not in existing_restraints:
            new_lines.append("RIGU")
            adp_restraints_added.append("RIGU")
        if "SIMU" not in existing_restraints:
            new_lines.append("SIMU 0.04 0.08")
            adp_restraints_added.append("SIMU 0.04 0.08")
        if npd_targets and "ISOR" not in existing_restraints:
            labels = sorted(npd_targets)
            for offset in range(0, len(labels), 8):
                command = "ISOR 0.02 0.04 " + " ".join(labels[offset : offset + 8])
                new_lines.append(command)
                adp_restraints_added.append(command)
        inserted_adp_stabilizers = True

    def isotropic_atom_line(toks: list[str], u_value: float) -> str:
        label = toks[0]
        sfac = int(toks[1])
        x, y, z = [float(v) for v in toks[2:5]]
        occ = float(toks[5])
        u = max(0.006, min(0.08, abs(float(u_value))))
        return f"{label:<4} {sfac:>2d} {x:11.6f} {y:11.6f} {z:11.6f} {occ:11.5f} {u:10.5f}"

    i = 0
    while i < len(active):
        line = active[i]
        toks = line.split()
        key = toks[0].upper() if toks else ""

        if not inserted_rem and key == "LATT":
            new_lines.append(f"REM GROND endpoint-push prepared model: {args.model_prep}")
            if int(args.set_ls_cycles) > 0:
                new_lines.append(f"REM GROND endpoint-push set L.S. {int(args.set_ls_cycles)}")
            inserted_rem = True

        if adp_stabilized and not inserted_adp_stabilizers and (is_shelx_atom_line(line) or key == "HKLF" or key == "END"):
            add_adp_stabilizers()
        if remove_restraints and key in SHELX_RESTRAINT_COMMANDS:
            removed_restraints.append(line)
            i += 1
            continue
        if free_h_anis and key == "AFIX":
            removed_afix += 1
            i += 1
            continue
        if key in {"L.S.", "LS", "CGLS"} and int(args.set_ls_cycles) > 0:
            new_lines.append(f"{toks[0]} {int(args.set_ls_cycles)}")
            i += 1
            continue
        if key == "END":
            if adp_stabilized and not inserted_adp_stabilizers:
                add_adp_stabilizers()
            if free_h_anis:
                new_lines.append(f"REM GROND endpoint-push free anisotropic H count: {len(converted_h)}")
            new_lines.append("END")
            i += 1
            continue
        if adp_stabilized and is_shelx_atom_line(line) and toks[0] in npd_targets:
            label = toks[0]
            old_u = float(toks[6]) if len(toks) > 6 else 0.02
            ueq = old_u
            consumed = 1
            if line.rstrip().endswith("=") and i + 1 < len(active):
                cont = active[i + 1].split()
                try:
                    ueq = (float(toks[6]) + float(toks[7]) + float(cont[0])) / 3.0
                    consumed = 2
                except (ValueError, IndexError):
                    ueq = old_u
            new_lines.append(isotropic_atom_line(toks, ueq))
            adp_isotropic.append({"label": label, "old_u": old_u, "fallback_u": max(0.006, min(0.08, abs(float(ueq))))})
            i += consumed
            continue
        if free_h_anis and is_shelx_atom_line(line) and shelx_label_element(toks[0]) == "H":
            label = toks[0]
            sfac = int(toks[1])
            x, y, z = [float(v) for v in toks[2:5]]
            occ = float(toks[5])
            old_u = float(toks[6]) if len(toks) > 6 else -1.2
            parent_ueq, parent = nearest_heavy_u((x % 1.0, y % 1.0, z % 1.0))
            multiplier = abs(old_u) if old_u < 0 else 1.2
            if multiplier < 0.5 or multiplier > 2.0:
                multiplier = 1.2
            u = max(0.010, min(0.080, parent_ueq * multiplier))
            new_lines.append(f"{label:<4} {sfac:>2d} {x:11.6f} {y:11.6f} {z:11.6f} {occ:11.5f} {u:10.5f} {u:10.5f} =")
            new_lines.append(f" {u:10.5f} {0.0:10.5f} {0.0:10.5f} {0.0:10.5f}")
            converted_h.append({"label": label, "parent": parent, "parent_ueq": parent_ueq, "init_u_diag": u, "old_u": old_u})
            i += 1
            continue

        new_lines.append(line)
        i += 1

    prepared = out_dir / "start_prepared.res"
    audit = out_dir / "start_prepared_audit.json"
    prepared.write_text("\n".join(new_lines) + "\n")
    audit.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(prepared),
                "model_prep": args.model_prep,
                "set_ls_cycles": int(args.set_ls_cycles),
                "discarded_post_end_lines": max(0, len(lines) - len(active)),
                "removed_restraints": removed_restraints,
                "removed_afix": removed_afix,
                "converted_hydrogens": converted_h,
                "adp_stabilized_npd_atoms": sorted(npd_targets),
                "adp_stabilized_restraints_added": adp_restraints_added,
                "adp_stabilized_isotropic_fallback": adp_isotropic,
            },
            indent=2,
        )
        + "\n"
    )
    return prepared, audit


def apply_push_checkpoint(args: argparse.Namespace) -> dict:
    """Fill push-endpoint inputs from a continue checkpoint when requested."""

    checkpoint_path = str(getattr(args, "checkpoint", "") or "")
    if not checkpoint_path:
        if not getattr(args, "source_summary", ""):
            raise SystemExit("Provide --source-summary, or use --checkpoint from a finished GROND auto run.")
        if not getattr(args, "start_res", ""):
            raise SystemExit("Provide --start-res, or use --checkpoint with a recorded start model.")
        return {}

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Continue checkpoint not found: {path}")
    checkpoint = read_json(path)
    if checkpoint.get("kind") != "grond_continue_checkpoint":
        raise SystemExit(f"{path} is not a GROND continue checkpoint.")
    if checkpoint.get("status") != "ready":
        reason = checkpoint.get("status_reason") or "checkpoint is not marked ready"
        raise SystemExit(f"{path} cannot be used for endpoint continuation: {reason}")

    base = path.parent
    dataset_count = int(checkpoint.get("dataset_count") or len(checkpoint.get("datasets") or []) or 1)
    if not getattr(args, "source_summary", ""):
        args.source_summary = _checkpoint_path(checkpoint.get("source_summary"), base=base)
    if not getattr(args, "start_res", ""):
        args.start_res = _checkpoint_path(checkpoint.get("recommended_start_res"), base=base)
    if dataset_count == 1 and not getattr(args, "hkl", "") and checkpoint.get("hkl"):
        args.hkl = _checkpoint_path(checkpoint.get("hkl"), base=base)
    if dataset_count == 1 and not getattr(args, "baseline_hkl", "") and checkpoint.get("baseline_hkl"):
        args.baseline_hkl = _checkpoint_path(checkpoint.get("baseline_hkl"), base=base)
    if dataset_count == 1 and not getattr(args, "hkl_transform", "") and checkpoint.get("hkl_transform"):
        args.hkl_transform = str(checkpoint.get("hkl_transform"))
    if dataset_count > 1 and not getattr(args, "multi_endpoint_merge_key", "") and checkpoint.get("multi_endpoint_merge_key"):
        args.multi_endpoint_merge_key = str(checkpoint.get("multi_endpoint_merge_key"))

    if dataset_count == 1 and not getattr(args, "source_summary", ""):
        raise SystemExit(f"{path} does not record source_summary; provide --source-summary.")
    if not getattr(args, "start_res", ""):
        raise SystemExit(f"{path} does not record recommended_start_res; provide --start-res.")
    return {
        "checkpoint": str(path),
        "checkpoint_status": checkpoint.get("status"),
        "dataset_count": dataset_count,
        "checkpoint_data": checkpoint,
    }


def hydrate_push_args_for_multi(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing multi-dataset continuation attributes from normal run defaults."""

    defaults = build_parser().parse_args(["run", "--out-dir", ".", "--hkl", "INTEGRATE.HKL", "--structure", "model.res"])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.fast = False
    args.native_only = False
    args.dry_run = bool(getattr(args, "dry_run", False))
    args.shelxl_feedback_cycles = int(getattr(args, "cycles", 3) or 0)
    args.shelxl_feedback_shelxl_iterations = int(getattr(args, "shelxl_iterations", 1) or 1)
    args.shelxl_feedback_endpoint_ls_cycles = int(getattr(args, "endpoint_ls_cycles", 1) or 1)
    args.shelxl_iterations = int(getattr(args, "shelxl_iterations", 1) or 1)
    args.n_points = int(getattr(args, "n_points", 20) or 20)
    args.n_basis = int(getattr(args, "n_basis", 108) or 108)
    args.final_n_basis = int(getattr(args, "final_n_basis", 0) or 0)
    args.final_n_points = int(getattr(args, "final_n_points", args.n_points) or args.n_points)
    args.prediction_cache_compression = getattr(args, "prediction_cache_compression", "none")
    args.make_maps = bool(getattr(args, "make_maps_on_final", False))
    args.multi_endpoint_merge_key = getattr(args, "multi_endpoint_merge_key", "model_symmetry") or "model_symmetry"
    return args


def prep_from_multi_checkpoint(
    checkpoint: Mapping[str, object],
    start_model: Path | str,
    *,
    min_d_spacing: float = 0.0,
) -> tuple[PrepConfig, dict[str, dict]]:
    """Build a multi-dataset PrepConfig and nuisance map from a continue checkpoint."""

    rows = checkpoint.get("datasets")
    if not isinstance(rows, list) or len(rows) < 2:
        raise SystemExit("Multi-dataset endpoint continuation needs at least two datasets in the checkpoint.")
    datasets: list[DatasetSpec] = []
    nuisance: dict[str, dict] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise SystemExit("Malformed dataset entry in continue checkpoint.")
        values = {field: raw[field] for field in DatasetSpec.__dataclass_fields__ if field in raw and raw[field] is not None}
        if float(min_d_spacing) > 0.0:
            values["min_d_spacing"] = float(min_d_spacing)
        ds = DatasetSpec(**values)
        datasets.append(ds)
        nuisance[ds.name] = dict(raw.get("nuisance") or {})
        if not nuisance[ds.name]:
            summary = _checkpoint_path(raw.get("source_summary"))
            if summary:
                nuisance[ds.name] = ((try_read_json(Path(summary)).get("best_native") or {}).get("nuisance") or {})
        if not nuisance[ds.name]:
            raise SystemExit(f"Checkpoint does not record a pinned nuisance block for dataset {ds.name!r}.")
    prep = PrepConfig(
        structure=str(Path(start_model).expanduser().resolve()),
        datasets=datasets,
        baseline_hkl=str(checkpoint.get("baseline_hkl") or ""),
        default_hkl_transform=str(checkpoint.get("hkl_transform") or "identity"),
        geometry_percentile=float(checkpoint.get("geometry_percentile") or 99.0),
        metadata={"continued_from": checkpoint.get("run_dir", "")},
    )
    return prep, nuisance


def write_multi_push_endpoint_report(out_dir: Path, *, best_cycle_tolerance: float) -> Path | None:
    """Write a compact safety report for a multi-dataset endpoint-push run."""

    summary_path = out_dir / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
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
        "scale_contract_note": (
            "Endpoint push refits the pinned full scale grouping. Run compact-scale selection separately "
            "on the selected safe cycle before calling it the most compact publication endpoint."
        ),
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
        handle.write(
            "These cycles refit the pinned full scale grouping. Compact scale reselection is not performed inside "
            "`push-endpoint`; run it separately on the selected safe cycle before promoting a smallest-scale publication endpoint.\n\n"
        )
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


def cmd_push_endpoint_multi(
    args: argparse.Namespace,
    checkpoint_info: Mapping[str, object],
    start_model: Path,
    prep_audit: Path | None,
    python: str,
    out_dir: Path,
) -> int:
    """Continue a multi-dataset endpoint from a finished GROND checkpoint."""

    checkpoint = checkpoint_info.get("checkpoint_data")
    if not isinstance(checkpoint, Mapping):
        raise SystemExit("Internal error: multi-dataset continuation missing checkpoint data.")
    args = hydrate_push_args_for_multi(args)
    prep, nuisance = prep_from_multi_checkpoint(
        checkpoint,
        start_model,
        min_d_spacing=float(getattr(args, "min_d_spacing", 0.0) or 0.0),
    )
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    write_prep(out_dir / "push_prep.json", prep)
    write_datasets_csv(out_dir / "push_datasets.csv", prep, nuisance)
    policy = checkpoint.get("structure_dataset_policy")
    if not isinstance(policy, Mapping) or not policy:
        policy = all_primary_policy({ds.name: Path(str(ds.hkl)).parent for ds in prep.datasets}, status="checkpoint_no_policy")
    if args.dry_run:
        plan = {
            "mode": "multi_dataset",
            "checkpoint": checkpoint_info.get("checkpoint", ""),
            "prepared_start_res": str(start_model),
            "prepared_audit": str(prep_audit) if prep_audit else "",
            "cycles": int(args.shelxl_feedback_cycles),
            "datasets": [asdict(ds) for ds in prep.datasets],
            "nuisance_datasets": sorted(nuisance),
            "multi_endpoint_merge_key": args.multi_endpoint_merge_key,
            "shared_loop": "per-dataset Bloch rows -> GROND-side corrected-HKL merge -> one shared SHELXL LS step -> repeat",
            "compact_scale_reselection_performed": False,
            "tool_config": getattr(args, "tool_config_path", ""),
            "external_tools": external_tool_statuses(args),
        }
        (out_dir / "push_endpoint_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0

    final_model, feedback = run_multi_shelxl_feedback_loop(
        prep,
        args,
        python,
        logs,
        out_dir,
        start_model=str(start_model),
        nuisance=nuisance,
        structure_dataset_policy=dict(policy),
    )
    result = {
        "status": "ok",
        "mode": "multi_dataset",
        "checkpoint": checkpoint_info.get("checkpoint", ""),
        "start_model": str(start_model),
        "model": final_model,
        "structure_feedback": feedback,
        "compact_scale_reselection_performed": False,
        "scale_contract_note": (
            "The push refit the pinned full grouping. Run compact-scale selection separately on the selected safe cycle."
        ),
    }
    (out_dir / "push_endpoint_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    report = write_multi_push_endpoint_report(out_dir, best_cycle_tolerance=float(args.best_cycle_tolerance))
    if report:
        print(f"Multi-dataset endpoint push report: {report}")
    return 0


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
        npd_atoms = re.findall(r"\s([A-Z][A-Z0-9]{3})\s+\*\* NON POSITIVE DEFINITE \*\*", text)
        split_match = re.search(r"(\d+)\s+atoms may be split", text)
        split_count = int(split_match.group(1)) if split_match else 0
        param_match = re.search(r"Total number of l\.s\. parameters =\s+(\d+)", text)
        restraint_match = re.search(r"Restrained GooF =\s+[0-9.]+ for\s+(\d+) restraints", text)
        endpoint_health = run.get("endpoint_res_health") or {}
        rows.append(
            {
                "cycle": len(rows) + 1,
                "model": str(model) if model is not None else "",
                "status": endpoint_health.get("reason") or metrics.get("endpoint_status") or "ok",
                "elapsed_seconds": run.get("elapsed_seconds"),
                "parameters": int(param_match.group(1)) if param_match else None,
                "restraints": int(restraint_match.group(1)) if restraint_match else None,
                "npd_atoms": npd_atoms,
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
    safe_rows = [r for r in finite_rows if not r["npd_atoms"]]
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


def cmd_push_endpoint(args: argparse.Namespace) -> int:
    """Continue an existing GROND endpoint with pinned nuisance settings."""

    python = choose_python(args.python)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = apply_push_checkpoint(args)
    start_model, prep_audit = prepare_push_endpoint_model(args, out_dir)
    if int(checkpoint_info.get("dataset_count") or 1) > 1:
        return cmd_push_endpoint_multi(args, checkpoint_info, start_model, prep_audit, python, out_dir)
    cmd = [
        python,
        str(engine_script("shelxl_feedback_loop.py")),
        "--source-summary",
        args.source_summary,
        "--start-res",
        str(start_model),
        "--out-dir",
        str(out_dir),
        "--python",
        python,
        "--cycles",
        str(args.cycles),
        "--n-basis",
        str(args.n_basis),
        "--n-points",
        str(args.n_points),
        "--final-n-basis",
        str(args.final_n_basis),
        "--final-n-points",
        str(args.final_n_points),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(args.shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.endpoint_ls_cycles),
        "--post-bloch-sigma-policy",
        args.post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--lambda-grid",
        args.lambda_grid,
        "--rescale",
        args.rescale,
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        args.row_plan_cache,
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--min-d-spacing",
        f"{args.min_d_spacing:g}",
        "--topup-d-min",
        f"{args.topup_d_min:g}",
        "--min-peak",
        f"{args.min_peak:g}",
        "--geometry-percentile",
        f"{args.geometry_percentile:g}",
        "--max-rows",
        str(args.max_rows),
    ]
    if args.hkl:
        cmd += ["--hkl", args.hkl]
    if args.baseline_hkl:
        cmd += ["--baseline-hkl", args.baseline_hkl]
    if args.hkl_transform:
        cmd += [f"--hkl-transform={args.hkl_transform}"]
    if args.make_maps_on_final:
        cmd.append("--make-maps-on-final")
    if args.dry_run:
        plan = {
            "command": cmd,
            "prepared_start_res": str(start_model),
            "prepared_audit": str(prep_audit) if prep_audit else "",
            **checkpoint_info,
            "tool_config": getattr(args, "tool_config_path", ""),
            "external_tools": external_tool_statuses(args),
        }
        (out_dir / "push_endpoint_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0
    subprocess.run(cmd, check=True)
    report = write_push_endpoint_report(out_dir, best_cycle_tolerance=float(args.best_cycle_tolerance))
    if report:
        print(f"Endpoint push report: {report}")
    return 0


def _doctor_readiness_failures(payload: Mapping[str, object], *, require_bloch: bool) -> list[str]:
    """Return actionable failures for the portable production runtime."""

    failures: list[str] = []
    if payload.get("probe_error"):
        failures.append(f"dependency probe failed: {payload['probe_error']}")
    elif payload.get("returncode") not in {None, 0}:
        failures.append(f"dependency probe exited with return code {payload['returncode']}")

    packages = payload.get("packages")
    if isinstance(packages, Mapping):
        for name in ("numpy", "scipy", "gemmi"):
            status = packages.get(name)
            if not isinstance(status, Mapping) or not status.get("available"):
                detail = status.get("error") if isinstance(status, Mapping) else "status missing"
                failures.append(f"required package {name} is unavailable: {detail}")
    elif not payload.get("probe_error"):
        failures.append("dependency probe did not report required package status")

    grond = payload.get("grond")
    if not isinstance(grond, Mapping) or grond.get("install_status") != "current_required_features_present":
        status = grond.get("install_status") if isinstance(grond, Mapping) else "missing"
        failures.append(f"GROND install is stale or incomplete: {status}")

    if require_bloch:
        bloch_ops = payload.get("bloch_ops")
        if not isinstance(bloch_ops, Mapping):
            failures.append("Bloch operation probe did not return a status block")
        elif bloch_ops.get("probe_error"):
            failures.append(f"Bloch operation probe failed: {bloch_ops['probe_error']}")
        elif bloch_ops.get("returncode") not in {None, 0}:
            failures.append(f"Bloch operation probe exited with return code {bloch_ops['returncode']}")
        else:
            scipy_cpu = bloch_ops.get("scipy_cpu")
            if not isinstance(scipy_cpu, Mapping):
                failures.append("Bloch operation probe did not report SciPy CPU status")
            elif scipy_cpu.get("import_error"):
                failures.append(f"SciPy Bloch kernel import failed: {scipy_cpu['import_error']}")
            else:
                for operation in ("eig", "eigh"):
                    status = scipy_cpu.get(operation)
                    if not isinstance(status, Mapping) or not status.get("ok"):
                        detail = status.get("error") if isinstance(status, Mapping) else "status missing"
                        failures.append(f"SciPy {operation} Bloch operation failed: {detail}")
    return failures


def cmd_doctor(args: argparse.Namespace) -> int:
    """CLI entrypoint that reports dependency and Bloch-kernel readiness."""

    # An explicitly requested or configured interpreter is itself the subject
    # of this diagnostic. Do not silently substitute another healthy Python.
    python = str(getattr(args, "python", "") or "").strip() or choose_python("")
    payload = probe_accelerators(python)
    package_init = Path(__file__).resolve().parent / "__init__.py"
    absolute_structure_script = engine_script("absolute_structure_diagnostic.py")
    feedback_script = engine_script("shelxl_feedback_loop.py")
    native_regularize_script = engine_script("native_row_regularize.py")
    try:
        absolute_structure_text = absolute_structure_script.read_text(errors="replace")
    except OSError:
        absolute_structure_text = ""
    try:
        feedback_text = feedback_script.read_text(errors="replace")
    except OSError:
        feedback_text = ""
    try:
        native_regularize_text = native_regularize_script.read_text(errors="replace")
    except OSError:
        native_regularize_text = ""
    try:
        cli_text = Path(__file__).read_text(errors="replace")
    except OSError:
        cli_text = ""
    required_features = {
        "shelxl_feedback_res_guard": (
            "def shelx_res_health" in native_regularize_text
            and "from .native_row_regularize import shelx_res_health" in feedback_text
            and "summary-selected endpoint model hash" in feedback_text
        ),
        "usable_feedback_model_selection": "shelxl_feedback_current_model" in cli_text and "shelx_model_is_usable" in cli_text,
        "final_feedback_row_recompute": "final_row_recompute" in feedback_text and "--skip-endpoint-refinement" in feedback_text,
        "corrected_endpoint_hklf4_contract": "HKLF 4" in native_regularize_text and "_unsupported_endpoint_reason" in native_regularize_text,
        "absolute_structure_no_pandas": "import pandas" not in absolute_structure_text and "pd.read_csv" not in absolute_structure_text,
    }
    payload["grond"] = {
        "version": __version__,
        "package": str(package_init.resolve()),
        "cli": str(Path(__file__).resolve()),
        "engine_root": str(LEGACY_ROOT.resolve()),
        "absolute_structure_diagnostic": str(absolute_structure_script.resolve()),
        "absolute_structure_uses_pandas": "import pandas" in absolute_structure_text or "pd.read_csv" in absolute_structure_text,
        "required_features": required_features,
        "install_status": "current_required_features_present" if all(required_features.values()) else "stale_or_incomplete_install",
    }
    payload["tool_config"] = {
        "path": getattr(args, "tool_config_path", ""),
        "configured_tools": getattr(args, "configured_tools", {}),
        "external_tools": external_tool_statuses(args),
    }
    if args.probe_bloch:
        payload["bloch_ops"] = probe_bloch_ops(python)
    failures = _doctor_readiness_failures(payload, require_bloch=bool(args.probe_bloch))
    payload["ready"] = not failures
    payload["failures"] = failures
    print(json.dumps(payload, indent=2))
    return 1 if bool(getattr(args, "strict", False)) and failures else 0


def cmd_compile(args: argparse.Namespace) -> int:
    """CLI entrypoint that compile-checks package and vendored Python modules."""

    root = Path(__file__).resolve().parents[1]
    python = choose_python(args.python)
    cmd = [python, "-m", "py_compile", *map(str, sorted(LEGACY_ROOT.rglob("*.py")))]
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="grond-compile-pycache-") as pycache_prefix:
        old_prefix = sys.pycache_prefix
        sys.pycache_prefix = pycache_prefix
        try:
            ok = compileall.compile_dir(str(root), quiet=1, force=bool(args.force), workers=1)
        finally:
            sys.pycache_prefix = old_prefix
        if not ok:
            return 1
        env["PYTHONPYCACHEPREFIX"] = pycache_prefix
        subprocess.run(cmd, check=True, env=env)
    print(
        json.dumps(
            {
                "package_root": str(root),
                "engine_root": str(LEGACY_ROOT),
                "python": python,
                "bytecode_cache": "temporary",
                "status": "compile_check_passed",
                "tool_config": getattr(args, "tool_config_path", ""),
                "external_tools": external_tool_statuses(args),
            },
            indent=2,
        )
    )
    return 0


def cmd_raw_polish(args: argparse.Namespace) -> int:
    """CLI entrypoint for the optional raw-CBF forward-polish experiment."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    configure_progress(logs, resume=bool(args.resume), total=1)

    if args.prep:
        prep = load_prep(Path(args.prep))
        if args.dataset_name:
            matches = [ds for ds in prep.datasets if ds.name == args.dataset_name]
            if not matches:
                raise SystemExit(f"Dataset {args.dataset_name!r} not found in {args.prep}")
            ds = matches[0]
        else:
            ds = prep.datasets[0]
        structure = args.structure or prep.structure
        hkl = ds.hkl
        hkl_transform = args.hkl_transform or ds.hkl_transform or prep.default_hkl_transform or "identity"
        baseline_hkl = args.baseline_hkl or ds.baseline_hkl or prep.baseline_hkl
    else:
        hkl, _xds_dir = resolve_hkl(args.xds_dir or args.hkl)
        if not args.structure:
            raise SystemExit("Provide --structure when not using --prep.")
        structure = args.structure
        hkl_transform = args.hkl_transform or "identity"
        baseline_hkl = args.baseline_hkl

    structure, structure_prep = prepare_structure_for_legacy(structure, out_dir)
    if getattr(args, "make_maps", False) and is_shelx_model(structure) and not getattr(args, "allow_missing_tools", False) and not getattr(args, "dry_run", False):
        statuses = external_tool_statuses(args)
        if not statuses.get("shelx2map", {}).get("available"):
            raise SystemExit(
                "Missing required external tools:\n- shelx2map is required for raw-polish map generation; set tools.shelx2map in a GROND config, set SHELX2MAP, or add --no-maps.\n"
                "Create a template with `GROND init-config --out grond.config.json`, then rerun with `--config grond.config.json` or set GROND_CONFIG."
            )
    cbf_dir = args.cbf_dir or ("" if args.images else infer_cbf_dir_from_hkl(hkl))
    if not args.images and not cbf_dir:
        raise SystemExit("Provide --cbf-dir/--images; no sibling CBF directory could be inferred.")

    n_basis = 20 if args.fast else args.n_basis
    n_points = 5 if args.fast else args.n_points
    absorption_mode = "none" if args.fast else args.bloch_profile_absorption_mode
    absorption_mu = 0.0 if args.fast else args.bloch_profile_absorption_mu
    absorption_frac = 0.0 if args.fast else args.bloch_profile_absorption_frac
    matrix_conventions = args.matrix_conventions or ("j_minus_i,i_minus_j" if not args.fast else "j_minus_i")
    cmd = [
        python,
        str(engine_script("run_raw_cbf_bloch_trial.py")),
        "--integrate-hkl",
        hkl,
        "--model",
        structure,
        f"--hkl-transform={hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--raw-grid",
        args.raw_grid,
        "--raw-integration-modes",
        args.raw_integration_modes,
        "--background-models",
        args.background_models,
        "--profile-sigma-xy-grid",
        args.profile_sigma_xy_grid,
        "--profile-sigma-z-grid",
        args.profile_sigma_z_grid,
        "--matrix-conventions",
        matrix_conventions,
        "--bloch-profile-n-basis",
        str(args.bloch_profile_n_basis if not args.fast else 20),
        "--bloch-profile-thickness",
        str(args.bloch_profile_thickness),
        "--bloch-profile-thickness-grad-frac",
        str(args.bloch_profile_thickness_grad_frac),
        "--bloch-profile-absorption-mode",
        absorption_mode,
        "--bloch-profile-absorption-mu",
        str(absorption_mu),
        "--bloch-profile-absorption-frac",
        str(absorption_frac),
        "--bloch-profile-z-smooth-frames",
        str(args.bloch_profile_z_smooth_frames),
        "--bloch-profile-z-gaussian-blend",
        str(args.bloch_profile_z_gaussian_blend),
        "--bloch-profile-z-gaussian-sigma",
        str(args.bloch_profile_z_gaussian_sigma),
        "--targets",
        args.targets,
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(n_points),
        "--sg-window",
        str(args.sg_window),
        "--min-d-spacing",
        str(args.min_d_spacing),
        "--min-peak",
        str(args.min_peak),
        "--thickness-grid",
        args.thickness_grid,
        "--thickness-grad-grid",
        args.thickness_grad_grid,
        f"--frame-offset-grid={args.frame_offset_grid}",
        "--group-width-grid",
        args.group_width_grid,
        "--detector-scale-bins",
        str(0 if args.fast else args.detector_scale_bins),
        "--thickness-points",
        str(3 if args.fast else args.thickness_points),
        "--lambda-grid",
        first(args.lambda_grid, FULL_Q_LAMBDA) if args.fast else args.lambda_grid,
        "--free-fraction",
        str(args.free_fraction),
        "--split-seed",
        str(args.split_seed),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--prediction-cache-dir",
        args.prediction_cache_dir or str(out_dir / "cache" / "native_predictions"),
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
    ]
    if args.images:
        cmd += ["--images", args.images]
    else:
        cmd += ["--cbf-dir", cbf_dir]
    if baseline_hkl:
        cmd += ["--baseline-hkl", baseline_hkl]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.top_isigma:
        cmd += ["--top-isigma", str(args.top_isigma)]
    if args.even_sample:
        cmd += ["--even-sample", str(args.even_sample)]
    if args.include_nonpositive:
        cmd.append("--include-nonpositive")
    cmd.append("--calibrate-geometry" if args.calibrate_geometry else "--no-calibrate-geometry")
    if args.make_maps:
        cmd.append("--make-maps")
    config = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": python,
        "tool_config": {
            "path": getattr(args, "tool_config_path", ""),
            "configured_tools": getattr(args, "configured_tools", {}),
            "external_tools": external_tool_statuses(args),
            "allow_missing_tools": bool(getattr(args, "allow_missing_tools", False)),
        },
        "hkl": hkl,
        "hkl_transform": hkl_transform,
        "structure": structure,
        "structure_preparation": structure_prep,
        "baseline_hkl": baseline_hkl,
        "cbf_dir": cbf_dir,
        "images": args.images,
        "fast": bool(args.fast),
        "workers": args.workers,
        "chunksize": args.chunksize,
    }
    (out_dir / "raw_polish_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_logged(cmd, logs, "raw_polish", dry_run=args.dry_run)
    print(json.dumps({"out_dir": str(out_dir), "report": str((out_dir / "REPORT.md").resolve())}, indent=2))
    return 0


def add_forensic_args(ap: argparse.ArgumentParser, *, expose_lambda_controls: bool = True) -> None:
    """Attach row-space forensic and observation-polish options to a parser."""

    ap.add_argument("--forensic-gate", dest="forensic_gate", action="store_true", default=True, help="Run post-final row-space forensic audits.")
    ap.add_argument("--no-forensic-gate", dest="forensic_gate", action="store_false")
    ap.add_argument("--forensic-depth", choices=["quick", "standard", "full"], default="standard", help="quick runs residual/pre-merge audits; standard adds dose and two main observation-correction holdouts; full adds edge-frame and sigma audits.")
    ap.add_argument("--forensic-bins", type=int, default=10)
    ap.add_argument("--forensic-top-rows", type=int, default=200)
    ap.add_argument("--forensic-top-groups", type=int, default=200)
    ap.add_argument("--observation-polish", choices=["auto", "on", "off"], default="auto", help="Promote held-out observation corrections into corrected-HKL endpoints when both main gates pass.")
    ap.add_argument("--observation-polish-combine", choices=["ensemble", "best_all", "best_free"], default="ensemble")
    if expose_lambda_controls:
        ap.add_argument(
            "--observation-polish-lambda-grid",
            default="",
            help="Diagnostic q-exponent grid for promoted observation-polish endpoints. Empty uses the full Bloch/Born correction.",
        )
    else:
        ap.set_defaults(observation_polish_lambda_grid="")
    ap.add_argument("--observation-polish-min-raw-wR2-gain", type=float, default=0.002)
    ap.add_argument("--observation-polish-endpoint-wR2-tolerance", type=float, default=0.0005)
    ap.add_argument("--forensic-maps", action="store_true", help="Write maps for promoted observation-polish endpoints during standalone forensics.")


def cmd_forensics(args: argparse.Namespace) -> int:
    """CLI entrypoint for standalone row-space forensic audits."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    configure_progress(logs, resume=bool(args.resume), total=_estimated_forensic_steps(args))
    args.fast = False
    args.forensic_gate = True
    summary = run_forensic_gate(
        python,
        Path(args.row_table).expanduser().resolve(),
        str(Path(args.hkl).expanduser().resolve()),
        str(Path(args.res).expanduser().resolve()),
        out_dir,
        logs,
        "standalone",
        args,
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2, allow_nan=True))
    return 0


def cmd_scale_polish(args: argparse.Namespace) -> int:
    """CLI entrypoint for frozen-Bloch scale-compression polish."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    cmd = [
        python,
        str(engine_script("scale_compression_polish.py")),
        "--row-table",
        str(Path(args.row_table).expanduser().resolve()),
        "--out-dir",
        str(out_dir),
        "--title",
        args.title,
        "--group-strides",
        args.group_strides,
        "--window-scan-deg",
        str(args.window_scan_deg),
        "--detector-bins",
        args.detector_bins,
        "--scale-objective",
        args.scale_objective,
        "--tolerance",
        str(args.tolerance),
        "--selection-metrics",
        args.selection_metrics,
        "--frame-models",
        args.frame_models,
        "--adaptive-frame-counts",
        args.adaptive_frame_counts,
    ]
    if args.summary:
        cmd += ["--summary", str(Path(args.summary).expanduser().resolve())]
    if args.fit_split:
        cmd += ["--fit-split", args.fit_split]
    if args.scan_offsets:
        cmd.append("--scan-offsets")
    if args.native_wght_a is not None:
        cmd += ["--native-wght-a", str(args.native_wght_a)]
    if args.native_wght_b is not None:
        cmd += ["--native-wght-b", str(args.native_wght_b)]
    run_logged(cmd, logs, "scale_compression_polish", dry_run=args.dry_run)
    summary = try_read_json(out_dir / "scale_compression_summary.json")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "report": str((out_dir / "REPORT.md").resolve()),
                "candidate_csv": str((out_dir / "scale_compression_candidates.csv").resolve()),
                "baseline": summary.get("baseline", {}),
                "selected": summary.get("selected", {}),
                "selected_widest": summary.get("selected_widest", {}),
                "selected_r_only": summary.get("selected_r_only", {}),
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0


def cmd_scale_replay(args: argparse.Namespace) -> int:
    """CLI entrypoint for replaying compact scale contracts through endpoints."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    cmd = [
        python,
        str(engine_script("scale_contract_replay.py")),
        "--row-table",
        str(Path(args.row_table).expanduser().resolve()),
        "--out-dir",
        str(out_dir),
        "--specs",
        args.specs,
        "--fit-split",
        args.fit_split,
        "--lambda-q",
        str(args.lambda_q),
        "--rescale",
        args.rescale,
        "--hkl-float-format",
        args.hkl_float_format,
        "--post-bloch-sigma-policy",
        args.post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        str(args.post_bloch_sigma_holdout_fraction),
        "--post-bloch-sigma-min-scale",
        str(args.post_bloch_sigma_min_scale),
        "--post-bloch-sigma-max-scale",
        str(args.post_bloch_sigma_max_scale),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(args.shelxl_iterations),
        "--wght-modes",
        args.wght_modes,
        "--instability-u",
        str(getattr(args, "instability_u", 0.01)),
        "--publication-goof-min",
        str(args.publication_goof_min),
        "--publication-goof-max",
        str(args.publication_goof_max),
        "--publication-max-r1-cost",
        str(args.publication_max_r1_cost),
        "--publication-free-wr2-tolerance",
        str(args.publication_free_wr2_tolerance),
        "--publication-max-modulation-p95",
        str(args.publication_max_modulation_p95),
        "--publication-prefer-wght-mode",
        args.publication_prefer_wght_mode,
        "--publication-selection-policy",
        args.publication_selection_policy,
        "--publication-information-criterion",
        args.publication_information_criterion,
    ]
    if bool(getattr(args, "publication_allow_npd", False)):
        cmd.append("--publication-allow-npd")
    if args.summary:
        cmd += ["--summary", str(Path(args.summary).expanduser().resolve())]
    if args.res:
        cmd += ["--res", str(Path(args.res).expanduser().resolve())]
    if args.native_wght_a is not None:
        cmd += ["--native-wght-a", str(args.native_wght_a)]
    if args.native_wght_b is not None:
        cmd += ["--native-wght-b", str(args.native_wght_b)]
    run_logged(cmd, logs, "scale_contract_replay", dry_run=args.dry_run)
    summary = try_read_json(out_dir / "summary.json")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "report": str((out_dir / "REPORT.md").resolve()),
                "endpoint_csv": str((out_dir / "scale_contract_endpoint_replay.csv").resolve()),
                "publication_selection": summary.get("publication_selection", {}),
                "rows": summary.get("rows", []),
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0


def cmd_calibrate_endpoint(args: argparse.Namespace) -> int:
    """Production-facing alias for endpoint calibration."""

    return cmd_scale_replay(args)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries, returning an empty list when absent."""

    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float_from_mapping(mapping: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    """Read a finite-ish float value from a nested report mapping."""

    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def _thickness_summary_row(
    *,
    dataset: str,
    scale_mode: str,
    feature_mode: str,
    n_basis: int,
    n_points: int,
    run_dir: Path,
    thickness_grid: str,
    gradient_grid: str,
) -> dict[str, object]:
    """Extract the compact promoted-thickness row from one scout run."""

    summary_path = run_dir / "summary.json"
    summary = try_read_json(summary_path)
    best_native = summary.get("best_native") or {}
    nuisance = best_native.get("nuisance") or {}
    metrics = best_native.get("metrics") or {}
    dyn_all = ((metrics.get("all") or {}).get("dyn") or {})
    dyn_work = ((metrics.get("work") or {}).get("dyn") or {})
    dyn_free = ((metrics.get("free") or {}).get("dyn") or {})
    born_all = ((metrics.get("all") or {}).get("born") or {})
    row_selection = summary.get("row_selection") or {}
    thickness_A = _float_from_mapping(nuisance, "thickness_A")
    return {
        "dataset": dataset,
        "scale_mode": scale_mode,
        "feature_mode": feature_mode,
        "n_basis": int(n_basis),
        "n_points": int(n_points),
        "status": "ok" if summary else "missing_summary",
        "thickness_A": thickness_A,
        "thickness_nm": thickness_A / 10.0 if math.isfinite(thickness_A) else float("nan"),
        "thickness_model": nuisance.get("thickness_model", ""),
        "thickness_grad_frac": _float_from_mapping(nuisance, "thickness_grad_frac"),
        "detector_scale_bins": nuisance.get("detector_scale_bins", ""),
        "group_width_deg": nuisance.get("group_width_deg", ""),
        "sg_jitter_mode": nuisance.get("sg_jitter_mode", ""),
        "absorption_mode": nuisance.get("absorption_mode", ""),
        "scale_count": best_native.get("scale_count", ""),
        "valid_rows": best_native.get("valid_rows", ""),
        "selected_rows": row_selection.get("selected_rows", ""),
        "work_rows": row_selection.get("work_rows", ""),
        "free_rows": row_selection.get("free_rows", ""),
        "all_dyn_R1": dyn_all.get("R1_all", ""),
        "all_dyn_R1gt": dyn_all.get("R1_gt", ""),
        "all_dyn_wR2": dyn_all.get("wR2_all", ""),
        "all_dyn_CC": dyn_all.get("CC", ""),
        "work_dyn_wR2": dyn_work.get("wR2_all", ""),
        "free_dyn_wR2": dyn_free.get("wR2_all", ""),
        "all_born_R1": born_all.get("R1_all", ""),
        "all_born_wR2": born_all.get("wR2_all", ""),
        "wall_seconds": summary.get("wall_seconds", ""),
        "thickness_grid_A": thickness_grid,
        "gradient_grid": gradient_grid,
        "run_dir": str(run_dir.resolve()),
        "summary_json": str(summary_path.resolve()),
        "row_table": str((run_dir / "native_dyn_calc.csv").resolve()),
        "nuisance_trials_csv": str((run_dir / "nuisance_trials.csv").resolve()),
    }


def _row_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Stable key for comparing two selected row correction fields."""

    return (
        str(row.get("row_index", "")),
        str(row.get("h", "")),
        str(row.get("k", "")),
        str(row.get("l", "")),
    )


def _logq_by_row(path: Path) -> dict[tuple[str, str, str, str], float]:
    """Load log(q) values from a GROND native row table."""

    out: dict[tuple[str, str, str, str], float] = {}
    for row in _read_csv_dicts(path):
        try:
            q = float(row.get("q_dyn_over_born", "nan"))
        except ValueError:
            continue
        if math.isfinite(q) and q > 0.0:
            out[_row_key(row)] = math.log(q)
    return out


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute a small dependency-free Pearson correlation."""

    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy, strict=False)) / (sx * sy)


def _p95(values: list[float]) -> float:
    """Return an empirical 95th percentile for a non-empty list."""

    if not values:
        return float("nan")
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(math.ceil(0.95 * len(vals))) - 1))
    return vals[idx]


def _q_field_stability(reference_csv: Path, candidate_csv: Path) -> dict[str, object]:
    """Compare two selected q fields by shared signed row identity."""

    ref = _logq_by_row(reference_csv)
    cand = _logq_by_row(candidate_csv)
    keys = sorted(set(ref) & set(cand))
    ref_vals = [ref[k] for k in keys]
    cand_vals = [cand[k] for k in keys]
    diffs = [abs(a - b) for a, b in zip(ref_vals, cand_vals, strict=False)]
    return {
        "reference_row_table": str(reference_csv.resolve()),
        "candidate_row_table": str(candidate_csv.resolve()),
        "shared_rows": len(keys),
        "logq_corr": _pearson(ref_vals, cand_vals),
        "median_abs_dlogq": sorted(diffs)[len(diffs) // 2] if diffs else float("nan"),
        "p95_abs_dlogq": _p95(diffs),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write heterogeneous dictionaries as a rectangular CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_thickness_scout_report(out_dir: Path, summary: dict[str, object]) -> None:
    """Write the human-readable thickness-scout report."""

    rows = list(summary.get("runs", []))
    q_rows = list(summary.get("q_field_stability", []))
    promoted = summary.get("promoted_reference") or {}
    warnings = list(summary.get("warnings", []))

    def pct(value: object) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        return f"{100.0 * v:.2f}%"

    def num(value: object, digits: int = 3) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    lines = [
        "# GROND Thickness Scout",
        "",
        "GROND thickness is an effective dynamical thickness selected for a declared signed-row contract, scale family, support size, and nuisance-feature state. It is not a direct microscope thickness unless an independent thickness measurement or a cross-contract stability test supports that interpretation.",
        "",
        "This report is diagnostic. `GROND auto` should select thickness under the simpler early row contract, pin it through the LS1 loop, and treat final-feature thickness movement as an identifiability warning rather than an automatic promotion.",
        "",
    ]
    if promoted:
        lines.extend([
            "## Promoted Reference",
            "",
            f"The promoted reference for q-field comparison is `{promoted.get('dataset', '')}` / `{promoted.get('scale_mode', '')}` / `{promoted.get('feature_mode', '')}` at N/Q `{promoted.get('n_basis', '')}/{promoted.get('n_points', '')}` with thickness `{num(promoted.get('thickness_nm'))}` nm and gradient `{num(promoted.get('thickness_grad_frac'))}`.",
            "",
        ])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend([
        "## Selected Runs",
        "",
        "| dataset | scale | features | N/Q | thickness nm | grad | all R1 | all wR2 | free wR2 | q corr | p95 | run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    q_by_run = {row.get("candidate_row_table"): row for row in q_rows}
    for row in rows:
        q = q_by_run.get(str(row.get("row_table", "")), {})
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('scale_mode', '')} | {row.get('feature_mode', '')} | "
            f"{row.get('n_basis', '')}/{row.get('n_points', '')} | {num(row.get('thickness_nm'))} | "
            f"{num(row.get('thickness_grad_frac'))} | {pct(row.get('all_dyn_R1'))} | {pct(row.get('all_dyn_wR2'))} | "
            f"{pct(row.get('free_dyn_wR2'))} | {num(q.get('logq_corr'))} | {num(q.get('p95_abs_dlogq'))} | "
            f"`{Path(str(row.get('run_dir', ''))).name}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "If detector-scale removal or final-feature activation moves the selected thickness while the row statistics remain similar, the thickness is not uniquely identified by the scalar row contract. In that case the stable quantity to trust is the row-wise Bloch/Born correction field and the final row and endpoint validation, not the absolute thickness number.",
        "",
        "The combined trial table and machine-readable summary are written next to this report.",
        "",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def _thickness_scout_variant_args(args: argparse.Namespace, *, scale_mode: str, feature_mode: str, thickness_grid_A: str, gradient_grid: str) -> argparse.Namespace:
    """Return a copy of args with one scout scale/feature contract pinned."""

    variant = argparse.Namespace(**vars(args))
    variant.thickness_grid = thickness_grid_A
    variant.thickness_grad_grid = gradient_grid
    variant.thickness_model_grid = "uniform_gradient"
    variant.thickness_points = int(getattr(args, "thickness_scout_thickness_points", args.thickness_points))
    variant.lambda_grid = FULL_Q_LAMBDA
    variant.endpoint_b_grid = "0.0"
    variant.endpoint_exti_grid = "off"
    variant.shelxl_iterations = 1
    variant.native_only = True
    variant.make_maps = False
    variant.max_rows = int(getattr(args, "thickness_scout_max_rows", 0) or 0)
    variant.sigma_a_grid = "0.0"
    variant.sigma_b_corr_grid = "0.0"
    variant.sigma_c_peak_grid = "0.0"
    variant.row_resolution_b_grid = "0.0"
    variant.dose_b_mode_grid = "none"
    variant.dose_b_per_frame_grid = "0.0"
    if scale_mode == "production":
        variant.detector_scale_bins = str(args.detector_scale_bins)
    elif scale_mode in {"no_detector", "nodetector", "no-detector"}:
        variant.detector_scale_bins = "0"
    else:
        raise SystemExit(f"Unknown thickness-scout scale mode {scale_mode!r}. Use production or no-detector.")
    if feature_mode == "bare":
        variant.sg_jitter_mode_grid = "none"
        variant.sg_jitter_sigma0_grid = "0.0"
        variant.sg_jitter_sigma_d_inv_grid = "0.0"
        variant.sg_jitter_sigma_frame_grid = "0.0"
        variant.sg_jitter_points = 1
        variant.absorption_mode_grid = "none"
        variant.absorption_mu_grid = "0.0"
        variant.absorption_frac_grid = "0.0"
        variant.absorption_mu_g2_grid = "0.0"
        variant.absorption_mu_theta2_grid = "0.0"
        variant.absorption_tds_b_grid = "0.0"
    elif feature_mode in {"final", "final-diagnostic", "final_diagnostic"}:
        variant.sg_jitter_mode_grid = args.thickness_scout_final_sg_jitter_mode
        variant.sg_jitter_sigma0_grid = str(args.thickness_scout_final_sg_jitter_sigma0)
        variant.sg_jitter_sigma_d_inv_grid = str(args.thickness_scout_final_sg_jitter_sigma_d_inv)
        variant.sg_jitter_sigma_frame_grid = str(args.thickness_scout_final_sg_jitter_sigma_frame)
        variant.sg_jitter_points = int(args.thickness_scout_final_sg_jitter_points)
        variant.absorption_mode_grid = args.thickness_scout_final_absorption_mode
        variant.absorption_mu_grid = str(args.thickness_scout_final_absorption_mu)
        variant.absorption_frac_grid = str(args.thickness_scout_final_absorption_frac)
        variant.absorption_mu_g2_grid = str(args.thickness_scout_final_absorption_mu_g2)
        variant.absorption_mu_theta2_grid = str(args.thickness_scout_final_absorption_mu_theta2)
        variant.absorption_tds_b_grid = str(args.thickness_scout_final_absorption_tds_b)
    else:
        raise SystemExit(f"Unknown thickness-scout feature mode {feature_mode!r}. Use bare or final-diagnostic.")
    return variant


def _target_ablation_variants(text: str) -> list[str]:
    """Parse target-ablation variant names with a stable default order."""

    raw = parse_name_grid(text)
    if not raw or raw == ["all"]:
        return list(XDS_TARGET_MODES)
    aliases = {
        "iobs-rlp": "iobs_over_rlp",
        "iobs/rlp": "iobs_over_rlp",
        "iobs_peak": "iobs_times_peak",
        "iobs-peak": "iobs_times_peak",
        "peak": "iobs_times_peak",
        "proxy": "recorded_proxy",
        "recorded": "recorded_proxy",
        "recorded-proxy": "recorded_proxy",
    }
    variants: list[str] = []
    for token in raw:
        value = aliases.get(token, token)
        if value not in XDS_TARGET_MODES:
            allowed = ", ".join(XDS_TARGET_MODES)
            raise SystemExit(f"Unknown XDS target-ablation variant {token!r}. Use one of: {allowed}.")
        if value not in variants:
            variants.append(value)
    return variants


def _target_ablation_auto_command(args: argparse.Namespace, variant: str, out_dir: Path) -> list[str]:
    """Build one fixed-target GROND auto command for the ablation wrapper."""

    cmd = [
        sys.executable,
        "-m",
        "grond.cli",
        "auto",
        "--out-dir",
        str(out_dir),
        "--structure",
        str(args.structure),
        "--baseline-hkl",
        str(args.baseline_hkl),
        "--target-mode",
        variant,
        "--shelxl-feedback-cycles",
        str(args.shelxl_feedback_cycles),
        "--shelxl-feedback-shelxl-iterations",
        str(args.shelxl_feedback_shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.endpoint_ls_cycles),
        "--shelxl",
        str(args.shelxl),
        "--min-d-spacing",
        f"{float(args.min_d_spacing):g}",
        "--min-peak",
        f"{float(args.min_peak):g}",
        "--geometry-percentile",
        f"{float(args.geometry_percentile):g}",
        "--hkl-transform",
        str(args.hkl_transform),
        "--workers",
        str(int(args.workers)),
        "--max-core-fraction",
        f"{float(args.max_core_fraction):g}",
        "--chunksize",
        str(int(args.chunksize)),
        "--compute-backend",
        str(args.compute_backend),
        "--row-plan-cache",
        str(args.row_plan_cache),
        "--prediction-cache-compression",
        str(args.prediction_cache_compression),
    ]
    if args.xds_dir:
        cmd.extend(["--xds-dir", str(args.xds_dir)])
    if args.hkl:
        cmd.extend(["--hkl", str(args.hkl)])
    if args.python:
        cmd.extend(["--python", str(args.python)])
    if getattr(args, "config", ""):
        cmd.extend(["--config", str(args.config)])
    if getattr(args, "allow_missing_tools", False):
        cmd.append("--allow-missing-tools")
    if args.trust_hkl_transform:
        cmd.append("--trust-hkl-transform")
    cmd.append("--hand-control" if args.hand_control else "--no-hand-control")
    cmd.append("--resume" if args.resume else "--no-resume")
    cmd.append("--make-maps" if args.make_maps else "--no-maps")
    for token in shlex.split(str(getattr(args, "extra_auto_args", "") or "")):
        cmd.append(token)
    return cmd


def _run_target_ablation_variant(cmd: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, float]:
    """Run one auto variant while teeing terminal output into a stable log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        if dry_run:
            print("DRY", " ".join(shlex.quote(part) for part in cmd), flush=True)
            handle.write("[dry_run] true\n[returncode] 0\n[seconds] 0.000\n")
            return 0, 0.0
        env = apply_thread_limits(os.environ.copy())
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        returncode = proc.wait()
        elapsed = time.perf_counter() - start
        handle.write(f"\n[returncode] {returncode}\n[seconds] {elapsed:.3f}\n")
    return int(returncode), elapsed


def _target_ablation_summary_row(variant: str, run_dir: Path, log_path: Path, returncode: int, elapsed_seconds: float) -> dict[str, object]:
    """Collect the compact fields from one completed target-ablation run."""

    result = try_read_json(run_dir / "grond_summary.json")
    quick = result.get("quick_stats") if isinstance(result.get("quick_stats"), Mapping) else {}
    console = quick.get("console") if isinstance(quick.get("console"), Mapping) else {}
    input_model = console.get("input_model") if isinstance(console.get("input_model"), Mapping) else {}
    final_row = console.get("final_row_model") if isinstance(console.get("final_row_model"), Mapping) else {}
    feedback = console.get("final_feedback_endpoint") if isinstance(console.get("final_feedback_endpoint"), Mapping) else {}
    endpoint = console.get("publication_endpoint") if isinstance(console.get("publication_endpoint"), Mapping) else {}
    input_shelxl = console.get("input_shelxl") if isinstance(console.get("input_shelxl"), Mapping) else {}
    files = console.get("files") if isinstance(console.get("files"), Mapping) else {}
    final_summary = Path(str(console.get("final_summary") or (run_dir / "FINAL_SUMMARY.md")))
    quick_stats = Path(str(console.get("quick_stats") or (run_dir / "QUICK_STATS.md")))
    status = "ok" if returncode == 0 and bool(result) else ("failed" if returncode != 0 else "missing_summary")
    return {
        "target_mode": variant,
        "target_formula": XDS_TARGET_MODE_LABELS.get(variant, variant),
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_seconds(elapsed_seconds),
        "hand": console.get("final_hand", ""),
        "hand_t_style_all": console.get("final_hand_t_style_all", console.get("final_hand_z_all")),
        "hand_t_style_all_n": console.get("final_hand_t_style_all_n"),
        "hand_t_style_all_df": console.get("final_hand_t_style_all_df"),
        "hand_t_style_free": console.get("final_hand_t_style_free", console.get("final_hand_z_free")),
        "hand_t_style_free_n": console.get("final_hand_t_style_free_n"),
        "hand_t_style_free_df": console.get("final_hand_t_style_free_df"),
        "hand_z_all": console.get("final_hand_z_all"),
        "hand_z_free": console.get("final_hand_z_free"),
        "hand_z_compatibility_note": (
            "legacy field names only; values are paired Student-t-style contrasts, not calibrated normal Z scores"
        ),
        "hkl_transform": console.get("selected_transform", ""),
        "input_shelxl_R1gt": input_shelxl.get("R1gt"),
        "input_shelxl_R1all": input_shelxl.get("R1all"),
        "input_shelxl_wR2all": input_shelxl.get("wR2all"),
        "input_shelxl_GooF": input_shelxl.get("GooF"),
        "input_dyn_R1all": input_model.get("row_R1all"),
        "input_dyn_wR2all": input_model.get("row_wR2all"),
        "input_born_R1all": input_model.get("born_R1all"),
        "input_born_wR2all": input_model.get("born_wR2all"),
        "final_dyn_R1all": final_row.get("row_R1all"),
        "final_dyn_wR2all": final_row.get("row_wR2all"),
        "final_dyn_free_wR2all": final_row.get("row_free_wR2all"),
        "final_born_R1all": final_row.get("born_R1all"),
        "final_born_wR2all": final_row.get("born_wR2all"),
        "final_born_free_wR2all": final_row.get("born_free_wR2all"),
        "direct_ls1_R1gt": feedback.get("R1gt"),
        "direct_ls1_R1all": feedback.get("R1all"),
        "direct_ls1_wR2all": feedback.get("wR2all"),
        "direct_ls1_GooF": feedback.get("GooF"),
        "publication_R1gt": endpoint.get("R1gt"),
        "publication_R1all": endpoint.get("R1all"),
        "publication_wR2all": endpoint.get("wR2all"),
        "publication_GooF": endpoint.get("GooF"),
        "publication_source": endpoint.get("source"),
        "publication_kind": endpoint.get("source_kind"),
        "publication_scale_spec": endpoint.get("scale_spec"),
        "publication_scale_params": endpoint.get("scale_params"),
        "publication_wght_mode": endpoint.get("wght_mode"),
        "publication_parameters": endpoint.get("parameters"),
        "publication_npd_count": endpoint.get("npd_count"),
        "publication_split_count": endpoint.get("split_count"),
        "run_dir": str(run_dir.resolve()),
        "terminal_log": str(log_path.resolve()),
        "final_summary": str(final_summary.resolve()),
        "quick_stats": str(quick_stats.resolve()),
        "full_json": str((run_dir / "grond_summary.json").resolve()),
        "row_table": str(final_row.get("row_table", "")),
        "publication_res": str(files.get("publication_res", "")),
        "publication_hkl": str(files.get("publication_hkl", "")),
        "publication_lst": str(files.get("publication_lst", "")),
    }


def _write_target_ablation_report(out_dir: Path, rows: list[dict[str, object]], commands: list[dict[str, object]]) -> None:
    """Write the human-readable XDS target-transform ablation report."""

    lines = [
        "# GROND XDS Target-Transform Ablation",
        "",
        "This diagnostic runs the same GROND auto path while changing only the reduced XDS row target passed into the Bloch-to-Born correction. It is meant to test Lorentz/PEAK conventions in the current scalar XDS row contract; it is not a raw-shoebox likelihood.",
        "",
        "## Results",
        "",
        "| target | formula | status | elapsed | hand paired t-style all/free | final dyn R1/wR2 | final Born R1/wR2 | publication R1/wR2/GooF | run |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('target_mode', '')}`",
                    str(row.get("target_formula", "")),
                    str(row.get("status", "")),
                    str(row.get("elapsed", "")),
                    f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_all'), row.get('hand_t_style_all_n'), row.get('hand_t_style_all_df'))} / "
                    f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_free'), row.get('hand_t_style_free_n'), row.get('hand_t_style_free_df'))}",
                    f"{_quick_percent(row.get('final_dyn_R1all'))} / {_quick_percent(row.get('final_dyn_wR2all'))}",
                    f"{_quick_percent(row.get('final_born_R1all'))} / {_quick_percent(row.get('final_born_wR2all'))}",
                    f"{_quick_percent(row.get('publication_R1all'))} / {_quick_percent(row.get('publication_wR2all'))} / {_quick_number(row.get('publication_GooF'), 3)}",
                    f"`{Path(str(row.get('run_dir', ''))).name}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Endpoint Paths",
            "",
            "| target | publication RES | publication HKL | row table | terminal log |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('target_mode', '')}`",
                    f"`{row.get('publication_res', '')}`",
                    f"`{row.get('publication_hkl', '')}`",
                    f"`{row.get('row_table', '')}`",
                    f"`{row.get('terminal_log', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Commands", ""])
    for command in commands:
        lines.append(f"### {command.get('target_mode', '')}")
        lines.append("")
        lines.append("```bash")
        lines.append(str(command.get("command_text", "")))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Reading This",
            "",
            "The production interpretation is conservative. A target transform is interesting only if it improves row-level dynamical statistics, keeps or improves the paired t-style hand contrast, and does not win solely by producing a prettier corrected-HKL endpoint. This contrast is reported with n and df=n-1 and is not a calibrated normal Z score. If a PEAK/RLP variant lowers endpoint R but worsens row-space Born/Bloch behavior or hand separation, keep it as a diagnostic and do not promote it.",
            "",
        ]
    )
    (out_dir / "TARGET_ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n")


def cmd_target_ablation(args: argparse.Namespace) -> int:
    """Run a fixed XDS target-transform ablation through ordinary auto."""

    if not args.xds_dir and not args.hkl:
        raise SystemExit("target-ablation requires --xds-dir or --hkl for the XDS row table.")
    if not args.structure:
        raise SystemExit("target-ablation requires --structure.")
    if not args.baseline_hkl:
        raise SystemExit("target-ablation requires --baseline-hkl.")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = _target_ablation_variants(args.variants)
    rows: list[dict[str, object]] = []
    commands: list[dict[str, object]] = []
    started = time.perf_counter()
    for variant in variants:
        variant_dir = out_dir / variant
        log_path = out_dir / "logs" / f"{variant}.terminal.log"
        cmd = _target_ablation_auto_command(args, variant, variant_dir)
        command_text = " ".join(shlex.quote(part) for part in cmd)
        commands.append({"target_mode": variant, "out_dir": str(variant_dir.resolve()), "terminal_log": str(log_path.resolve()), "command": cmd, "command_text": command_text})
        print(f"\n=== XDS target ablation: {variant} ({XDS_TARGET_MODE_LABELS.get(variant, variant)}) ===", flush=True)
        returncode, elapsed = _run_target_ablation_variant(cmd, log_path, dry_run=bool(args.dry_run))
        row = _target_ablation_summary_row(variant, variant_dir, log_path, returncode, elapsed)
        rows.append(row)
        if returncode != 0 and args.fail_fast:
            break
    total_elapsed = time.perf_counter() - started
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": "target-ablation",
        "interpretation": "Reduced XDS row-target ablation. This does not invert the data back to raw detector pixels.",
        "inputs": {
            "xds_dir": str(args.xds_dir),
            "hkl": str(args.hkl),
            "structure": str(args.structure),
            "baseline_hkl": str(args.baseline_hkl),
            "variants": variants,
            "shelxl_feedback_cycles": int(args.shelxl_feedback_cycles),
            "hand_control": bool(args.hand_control),
        },
        "elapsed_seconds": total_elapsed,
        "elapsed": format_seconds(total_elapsed),
        "runs": rows,
        "commands": commands,
    }
    _write_csv(out_dir / "target_ablation_summary.csv", rows)
    (out_dir / "target_ablation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_target_ablation_report(out_dir, rows, commands)
    print("")
    print(f"GROND target ablation finished in {format_seconds(total_elapsed)}.")
    print(f"Report: {out_dir / 'TARGET_ABLATION_SUMMARY.md'}")
    print("")
    print("| target | status | hand paired t-style all/free | final dyn R1/wR2 | publication R1/wR2/GooF |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.get('target_mode', '')} | {row.get('status', '')} | "
            f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_all'), row.get('hand_t_style_all_n'), row.get('hand_t_style_all_df'))} / "
            f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_free'), row.get('hand_t_style_free_n'), row.get('hand_t_style_free_df'))} | "
            f"{_quick_percent(row.get('final_dyn_R1all'))} / {_quick_percent(row.get('final_dyn_wR2all'))} | "
            f"{_quick_percent(row.get('publication_R1all'))} / {_quick_percent(row.get('publication_wR2all'))} / {_quick_number(row.get('publication_GooF'), 3)} |"
        )
    return 1 if any(int(row.get("returncode", 0) or 0) != 0 for row in rows) else 0


def cmd_thickness_scout(args: argparse.Namespace) -> int:
    """Run a diagnostic thickness contract-sensitivity scout."""

    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    prep = prep_from_args(args)
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    original_structure = prep.structure
    prep.structure, structure_prep = prepare_structure_for_legacy(prep.structure, out_dir)
    prep.metadata = dict(prep.metadata or {})
    if structure_prep.get("prepared"):
        prep.metadata["original_structure"] = original_structure
        prep.metadata["structure_preparation"] = structure_prep
    write_prep(out_dir / "thickness_scout_prep.json", prep)

    basis_grid = parse_int_grid(args.n_basis_grid)
    if not basis_grid:
        raise SystemExit("Provide at least one integer in --n-basis-grid.")
    n_points = int(args.n_points)
    thickness_nm_grid = parse_numeric_grid(args.thickness_grid_nm)
    if not thickness_nm_grid:
        raise SystemExit("Provide at least one value in --thickness-grid-nm.")
    gradient_grid = parse_numeric_grid(args.gradient_grid)
    if not gradient_grid:
        raise SystemExit("Provide at least one value in --gradient-grid.")
    scale_modes = parse_name_grid(args.scale_modes)
    feature_modes = parse_name_grid(args.feature_modes)
    total = len(prep.datasets) * len(scale_modes) * len(feature_modes) * len(basis_grid)
    configure_progress(logs, resume=bool(args.resume), total=total)

    rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    full_grid_A = [10.0 * v for v in thickness_nm_grid]
    promoted_reference: dict[str, object] = {}

    for ds in prep.datasets:
        for scale_mode in scale_modes:
            for feature_mode in feature_modes:
                center_A = float("nan")
                center_grad = float("nan")
                for index, n_basis in enumerate(basis_grid):
                    if index == 0 or not args.local_confirmation:
                        thick_values_A = full_grid_A
                        grad_values = gradient_grid
                    else:
                        thick_values_A = centered_grid(
                            center_A,
                            window=10.0 * float(args.local_window_nm),
                            step=10.0 * float(args.local_step_nm),
                            lower=10.0 * float(args.thickness_min_nm),
                            upper=10.0 * float(args.thickness_max_nm),
                        ) or full_grid_A
                        grad_values = centered_grid(
                            center_grad,
                            window=float(args.local_gradient_window),
                            step=float(args.local_gradient_step),
                            lower=float(args.gradient_min),
                            upper=float(args.gradient_max),
                        ) or gradient_grid
                    thickness_grid_A = format_numeric_grid(thick_values_A)
                    gradient_grid_text = format_numeric_grid(grad_values)
                    variant = _thickness_scout_variant_args(
                        args,
                        scale_mode=scale_mode,
                        feature_mode=feature_mode,
                        thickness_grid_A=thickness_grid_A,
                        gradient_grid=gradient_grid_text,
                    )
                    label = f"thickness_{safe_label(ds.name)}_{safe_label(scale_mode)}_{safe_label(feature_mode)}_N{int(n_basis)}Q{n_points}"
                    run_dir = out_dir / "runs" / ds.name / f"{safe_label(scale_mode)}_{safe_label(feature_mode)}_N{int(n_basis)}Q{n_points}"
                    cmd = native_stage_cmd(
                        python,
                        ds,
                        prep.structure,
                        dataset_baseline_hkl(prep, ds),
                        run_dir,
                        variant,
                        n_basis=int(n_basis),
                        n_points=n_points,
                        stage=label,
                        make_maps=False,
                        endpoint=False,
                    )
                    run_logged(cmd, logs, label, dry_run=bool(args.dry_run))
                    row = _thickness_summary_row(
                        dataset=ds.name,
                        scale_mode=scale_mode,
                        feature_mode=feature_mode,
                        n_basis=int(n_basis),
                        n_points=n_points,
                        run_dir=run_dir,
                        thickness_grid=thickness_grid_A,
                        gradient_grid=gradient_grid_text,
                    )
                    rows.append(row)
                    if not promoted_reference and row.get("status") == "ok":
                        promoted_reference = row
                    if math.isfinite(_float_from_mapping(row, "thickness_A")):
                        center_A = _float_from_mapping(row, "thickness_A")
                    if math.isfinite(_float_from_mapping(row, "thickness_grad_frac")):
                        center_grad = _float_from_mapping(row, "thickness_grad_frac")
                    for trial in _read_csv_dicts(run_dir / "nuisance_trials.csv"):
                        trial_rows.append(
                            {
                                "dataset": ds.name,
                                "scale_mode": scale_mode,
                                "feature_mode": feature_mode,
                                "n_basis": int(n_basis),
                                "n_points": n_points,
                                "run_dir": str(run_dir.resolve()),
                                **trial,
                            }
                        )

    q_rows: list[dict[str, object]] = []
    ref_csv = Path(str(promoted_reference.get("row_table", ""))) if promoted_reference else Path()
    if promoted_reference and ref_csv.exists():
        for row in rows:
            cand_csv = Path(str(row.get("row_table", "")))
            if cand_csv.exists():
                q_rows.append({**{k: row.get(k, "") for k in ("dataset", "scale_mode", "feature_mode", "n_basis", "n_points")}, **_q_field_stability(ref_csv, cand_csv)})
    if rows:
        reference_thickness = _float_from_mapping(promoted_reference, "thickness_nm")
        for row in rows:
            thickness_nm = _float_from_mapping(row, "thickness_nm")
            if math.isfinite(reference_thickness) and math.isfinite(thickness_nm):
                delta_nm = abs(thickness_nm - reference_thickness)
                if delta_nm >= float(args.warning_shift_nm):
                    warnings.append(
                        f"{row.get('dataset')} {row.get('scale_mode')}/{row.get('feature_mode')} N{row.get('n_basis')} selected {thickness_nm:.1f} nm, {delta_nm:.1f} nm away from the promoted reference. Treat this as row-contract sensitivity, not a new physical thickness."
                    )

    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": "thickness-scout",
        "inputs": {
            "structure": prep.structure,
            "original_structure": original_structure,
            "datasets": [asdict(ds) for ds in prep.datasets],
            "n_basis_grid": basis_grid,
            "n_points": n_points,
            "thickness_grid_nm": thickness_nm_grid,
            "gradient_grid": gradient_grid,
            "scale_modes": scale_modes,
            "feature_modes": feature_modes,
        },
        "policy": {
            "interpretation": "GROND thickness is row-contract effective thickness, not a direct microscope thickness unless independently calibrated.",
            "auto_recommendation": "Use the early bare production-scale thickness for auto, pin it through LS1, and treat final-feature movement as a warning.",
            "warning_shift_nm": float(args.warning_shift_nm),
        },
        "promoted_reference": promoted_reference,
        "runs": rows,
        "q_field_stability": q_rows,
        "warnings": warnings,
    }
    _write_csv(out_dir / "thickness_scout_table.csv", rows)
    _write_csv(out_dir / "thickness_scout_trials.csv", trial_rows)
    _write_csv(out_dir / "q_field_stability.csv", q_rows)
    (out_dir / "thickness_scout_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_thickness_scout_report(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "report": str((out_dir / "REPORT.md").resolve()), "warnings": warnings}, indent=2, allow_nan=True))
    return 0


def add_common_run_args(ap: argparse.ArgumentParser, *, expose_lambda_controls: bool = True) -> None:
    """Attach the shared run/auto options to an argparse subcommand."""

    ap.add_argument("--prep", default="")
    ap.add_argument("--hkl", default="")
    ap.add_argument("--xds-dir", default="")
    ap.add_argument("--dials-refl", default="", help="DIALS integrated.refl/scaled.refl file. Auto-converted into a GROND row directory under OUT/prepared.")
    ap.add_argument("--dials-expt", default="", help="DIALS integrated.expt/scaled.expt file matching --dials-refl.")
    ap.add_argument("--dials-python", default="", help="Python executable that can import dials and dxtbx. Leave blank when the active Python already can.")
    ap.add_argument("--dials-intensity-mode", choices=["sum_complete", "sum_raw", "scale"], default="sum_complete", help="DIALS intensity mapping for the GROND row contract.")
    ap.add_argument("--dials-sigma-mode", choices=["auto", "raw", "detector_gain"], default="raw", help="DIALS sigma bridge policy. raw keeps the detector-model variance; auto empirical calibration and detector_gain scaling are explicit diagnostics.")
    ap.add_argument("--dials-sigma-scale", type=float, default=None, help="Explicit DIALS sigma multiplier. Overrides --dials-sigma-mode when positive.")
    ap.add_argument("--dials-min-partiality-write", type=float, default=1e-6, help="Minimum DIALS partiality written into the converted row file. GROND's --min-peak still controls production row selection.")
    ap.add_argument("--dials-drop-dont-integrate", action="store_true", help="Deprecated alias for the default production behavior. Rows flagged dont_integrate are dropped unless --dials-keep-dont-integrate is used.")
    ap.add_argument("--dials-keep-dont-integrate", action="store_true", help="Diagnostic only: keep DIALS rows flagged dont_integrate.")
    ap.add_argument("--dials-keep-bad-flags", action="store_true", help="Diagnostic only: keep DIALS rows with severe bad-pixel, overloaded, powder-ring, failed-background, or failed-summation flags.")
    ap.add_argument("--structure", default="")
    ap.add_argument("--baseline-hkl", default="")
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["bloch", "regularize", "auto"], default="auto")
    ap.add_argument("--fast", action="store_true", help="Use N=20/Q=5, no absorption/non-Hermitian terms, and no structure moves.")
    ap.add_argument("--full-fast", action="store_true", help="Run the automatic gates and lightweight structure update with cheap N/Q, but skip expensive final polish.")
    ap.add_argument("--full-fast-n-basis", type=int, default=24, help="Pinned basis size used by --full-fast final and screening stages. Use 10 for an even cheaper stress test.")
    ap.add_argument("--full-fast-n-points", type=int, default=10, help="Quadrature points used by --full-fast.")
    ap.add_argument("--full-fast-structure-rows", type=int, default=0, help="Maximum objective rows for lightweight --full-fast structure update; 0 chooses an atom-count-aware budget.")
    ap.add_argument("--full-fast-structure-max-nfev", type=int, default=0, help="Maximum function evaluations per lightweight --full-fast structure block; 0 chooses an atom-count-aware budget.")
    ap.add_argument("--auto-decide", dest="auto_decide", action="store_true", default=True, help="Run preflight, transform/hand scouts, and adaptive thickness decisions before final work.")
    ap.add_argument("--no-auto-decide", dest="auto_decide", action="store_false")
    ap.add_argument("--native-only", action="store_true", help="Do not write corrected-HKL endpoints.")
    ap.add_argument("--python", default="")
    add_tool_config_args(ap)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="Print the compact machine-readable payload instead of the human final summary.")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reuse completed stages recorded in OUT/grond_progress.json.")
    ap.add_argument("--workers", type=int, default=0, help="Worker processes. Default 0 uses --max-core-fraction of detected CPU cores.")
    ap.add_argument("--max-core-fraction", type=float, default=0.85, help="Fraction of detected CPU cores to use when --workers=0.")
    ap.add_argument("--chunksize", type=int, default=0, help="Multiprocessing chunksize. Default 0 chooses a small value for load balancing.")
    ap.add_argument("--prediction-cache-dir", default="", help="Optional root for prediction caches. Default is OUT/cache.")
    ap.add_argument("--prediction-cache-scope", choices=["shared", "stage"], default="shared", help="Shared reuses safe hash-keyed predictions across adaptive stages; stage isolates every run.")
    ap.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none", help="Use none for fastest local SSD/NVMe caches; compressed saves disk on slow or limited storage.")
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy", help="Bloch prediction backend. SciPy eigensolver is the production reference; scipy_bethe uses an opt-in promoted Bethe/Schur strong-beam reduction; SciPy expm/expm_multiply, real-embedded JAX expm, and JAX eigensolver backends are experimental agreement/speed probes. *_padded JAX backends pad matrix dimensions into small buckets to reduce compilation churn.")
    ap.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off", help="Experimental frozen row-geometry cache. Off is default until full workflow benchmarks show a clear win.")
    ap.add_argument("--jax-batch-size", type=int, default=0, help="JAX matrix batch size for accelerator backends. 0 chooses a conservative size from matrix dimension and device count.")
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0, help="Pad JAX matrix dimensions to this multiple. 0 keeps exact sizes unless a *_padded backend is selected, where 0 means 8.")
    ap.add_argument("--bethe-strong-count", type=int, default=48, help="Strong-beam block size for scipy_bethe. Values >= final N recover the dense SciPy matrix.")
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target", help="Strong/weak-beam reduction used by scipy_bethe.")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005, help="Promote weak beams whose diagonal denominator is this close to the Schur energy.")
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5, help="Small-denominator floor for scipy_bethe Schur corrections.")
    ap.add_argument("--matrix-convention", choices=["j_minus_i", "i_minus_j"], default="j_minus_i", help="Off-diagonal U_g indexing convention. Keep j_minus_i for production; i_minus_j is diagnostic.")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--scout-max-rows", type=int, default=1000, help="Representative row limit for automatic transform/hand scouts.")
    ap.add_argument("--scout-n-basis", type=int, default=8, help="Basis size for cheap decision scouts.")
    ap.add_argument("--scout-n-points", type=int, default=3, help="Quadrature points for cheap decision scouts.")
    ap.add_argument("--transform-rerank-top", type=int, default=3, help="Number of coarse transform-scout winners to rerun at better N/Q; 0 disables the second stage.")
    ap.add_argument("--transform-rerank-n-basis", type=int, default=24, help="Basis size for second-stage transform reranking.")
    ap.add_argument("--transform-rerank-n-points", type=int, default=10, help="Quadrature points for second-stage transform reranking.")
    ap.add_argument("--transform-rerank-max-rows", type=int, default=0, help="Row limit for second-stage transform reranking; 0 reuses the preflight scout row budget.")
    ap.add_argument("--transform-candidates", default="auto", help="Semicolon-separated transforms, or auto/common/off. Auto includes identity and common XDS-to-model permutations.")
    ap.add_argument("--transform-scout-jobs", type=int, default=0, help="Concurrent transform-scout subprocesses across candidates and datasets; 0 derives a core-safe value from --workers and --max-core-fraction.")
    ap.add_argument("--transform-scout-workers-per-job", type=int, default=1, help="Worker processes inside each transform-scout subprocess. Keep low when many scout jobs run concurrently.")
    ap.add_argument("--transform-metadata-prefilter", choices=["auto", "on", "off"], default="auto", help="Use solved/XSCALE baseline-HKL intensity overlap to shrink transform scouts before Bloch calculations.")
    ap.add_argument("--transform-metadata-max-candidates", type=int, default=3, help="Number of ranked metadata candidates kept when the intensity-overlap signal is plausible but ambiguous.")
    ap.add_argument("--transform-metadata-min-pairs", type=int, default=100, help="Minimum paired positive-intensity rows required before trusting metadata transform ranking.")
    ap.add_argument("--transform-metadata-min-correlation", type=float, default=0.70, help="Minimum log-intensity correlation required before metadata can prefilter transform scouts.")
    ap.add_argument("--transform-metadata-min-correlation-gap", type=float, default=0.05, help="Correlation gap required for auto mode to promote a single transform without Bloch-transform scouting.")
    ap.add_argument("--transform-metadata-max-rows", type=int, default=0, help="Maximum INTEGRATE/XDS rows used by metadata transform ranking; 0 uses all rows.")
    ap.add_argument("--trust-hkl-transform", action="store_true", help="Trust the supplied --hkl-transform or prep-file transform and skip Bloch transform scouting while keeping the rest of auto-decision enabled.")
    ap.add_argument("--hand-accept-delta", dest="hand_accept_delta", metavar="DELTA", type=float, default=0.015, help="Adaptive score margin required before switching to the alternate hand.")
    ap.add_argument("--hand-max-wR2-all", type=float, default=0.45, help="Maximum alternate-hand all-row wR2 allowed before automatic hand switching. Poor row contracts still report the hand diagnostic but keep the input hand.")
    ap.add_argument("--hand-max-free-wR2-all", type=float, default=0.50, help="Maximum alternate-hand free-row wR2 allowed before automatic hand switching.")
    ap.add_argument("--hand-min-free-wR2-gain", type=float, default=0.0, help="Minimum free-row wR2 improvement required before automatic scout-stage hand switching.")
    ap.add_argument("--hand-min-z-all", type=float, default=2.0, help="Legacy option name for the minimum paired all-row t-style hand statistic required before post-feedback hand promotion; this is not a calibrated normal Z score.")
    ap.add_argument("--hand-min-z-free", type=float, default=0.5, help="Legacy option name for the minimum paired free-row t-style hand statistic required before post-feedback hand promotion. Set to 0 to disable the free-row sign gate; the minimum free-row count still applies.")
    ap.add_argument("--hand-min-free-rows", type=int, default=10, help="Minimum paired held-out rows required before a post-feedback diagnostic may materially favor the opposite hand.")
    ap.add_argument("--stop-on-bad-preflight", action="store_true", default=False)
    ap.add_argument("--stop-on-bad-fit", action="store_true", default=False)
    ap.add_argument("--stop-if-wR2-above", type=float, default=0.75, help="Bad-data stop threshold for native all-row dynamic wR2 when --stop-on-bad-fit is enabled.")
    ap.add_argument("--min-bloch-wR2-gain", type=float, default=-0.01, help="Minimum wR2 gain over Born before structure update is considered reliable.")
    ap.add_argument("--feature-accept-delta", type=float, default=0.002, help="Minimum adaptive-score improvement before keeping a staged row-resolution/absorption/jitter/error/dose feature family.")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=99.0)
    ap.add_argument("--target-mode", choices=list(XDS_TARGET_MODES), default="iobs", help=argparse.SUPPRESS)
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--make-maps", dest="make_maps", action="store_true", default=True)
    ap.add_argument("--no-maps", dest="make_maps", action="store_false")
    ap.add_argument("--h-omit", choices=["off", "on", "only", "both"], default="off")
    ap.add_argument("--hand-control", dest="hand_control", action="store_true", default=True, help="Run the alternate-hand diagnostic when model symmetry makes that check meaningful.")
    ap.add_argument("--no-hand-control", dest="hand_control", action="store_false", help="Skip the alternate-hand diagnostic.")
    ap.add_argument("--hand-n-basis", dest="hand_n_basis", metavar="N", type=int, default=24, help="Basis size for default hand-diagnostic runs; keep cheap unless the call is ambiguous.")
    ap.add_argument("--hand-n-points", dest="hand_n_points", metavar="Q", type=int, default=10, help="Quadrature points for default hand-diagnostic runs.")
    ap.add_argument(
        "--production-scales",
        action="store_true",
        default=False,
        help="Fit nuisance scales on all rows for a final diagnostic; validation and automatic selection use work-only scales by default.",
    )
    ap.add_argument(
        "--validation-scales",
        dest="production_scales",
        action="store_false",
        help="Fit nuisance scales on work rows only (the default).",
    )
    ap.add_argument("--screen-n-basis", type=int, default=24, help="Basis size for initial nuisance/geometry screening.")
    ap.add_argument("--screen-n-points", type=int, default=10, help="Quadrature points for initial nuisance/geometry screening.")
    ap.add_argument("--n-basis", type=int, default=0, help="Basis size for pinned final polish runs. Default 0 scans Sg support and uses max support plus margin.")
    ap.add_argument("--n-points", type=int, default=20, help="Quadrature points for pinned final polish runs; keep <=20 for production.")
    ap.add_argument("--final-basis-scan-limit", type=int, default=500, help="Upper limit used by the cheap support-only scan when --n-basis=0.")
    ap.add_argument("--final-basis-margin", type=int, default=10, help="Extra beams above the observed maximum support for final polish.")
    ap.add_argument("--final-basis-min", type=int, default=20)
    ap.add_argument("--final-basis-max", type=int, default=500)
    ap.add_argument("--final-basis-fallback", type=int, default=128, help="Dry-run fallback when --n-basis=0.")
    ap.add_argument("--structure-n-basis", type=int, default=24)
    ap.add_argument("--structure-n-points", type=int, default=6)
    ap.add_argument("--structure-objective-rows", type=int, default=400)
    ap.add_argument("--structure-max-nfev", type=int, default=5)
    ap.add_argument("--structure-early-stop-window", type=int, default=64, help="Stop a structure-update block after this many residual calls without a meaningful data-RMS gain; 0 disables.")
    ap.add_argument("--structure-early-stop-min-evals", type=int, default=0, help="Minimum residual calls before a plateau stop can trigger; 0 uses the engine's parameter-count-aware floor.")
    ap.add_argument("--structure-early-stop-min-improvement", type=float, default=2e-5, help="Minimum RMS improvement counted as progress for plateau stopping.")
    ap.add_argument("--structure-early-stop-metric", choices=["total_rms", "data_rms"], default="data_rms", help="Trace column used by the structure-update plateau stop.")
    ap.add_argument("--structure-update", choices=["auto", "native_gradient", "shelxl_feedback", "off"], default="auto", help="Structure-update engine. Auto uses the high-N/Q SHELXL feedback loop for full runs when SHELXL is available; native_gradient is the slower finite-difference diagnostic path.")
    ap.add_argument("--structure-blocks", default="adp_ueq")
    ap.add_argument("--structure-atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--restraints", choices=["none", "start_geometry", "cctbx_rigu"], default="cctbx_rigu")
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", choices=["auto", "fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"], default="auto", help="Rocking-width source. Auto keeps the fixed Sg width; DIALS bbox z-ranges are available as an explicit diagnostic.")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument("--sg-window-scout", choices=["auto", "on", "off"], default="off", help="Late held-out scout over simple fixed Sg rocking-window widths after final scale and resolution gates. Off is the production default; use auto/on when auditing rocking-width sensitivity.")
    ap.add_argument("--sg-window-scout-grid", default="0.003,0.004,0.006,0.008", help="Comma-separated fixed Sg widths tested by the late rocking-window scout.")
    ap.add_argument("--sg-window-scout-n-basis", type=int, default=24, help="Basis size for the late Sg-window scout.")
    ap.add_argument("--sg-window-scout-n-points", type=int, default=10, help="Quadrature points for the late Sg-window scout.")
    ap.add_argument("--sg-window-scout-max-rows", type=int, default=0, help="Optional row cap for the late Sg-window scout; 0 uses the final row set.")
    ap.add_argument("--sg-window-scout-min-wR2-gain", type=float, default=0.002, help="Minimum all-row wR2 gain required before a non-baseline Sg width can be promoted.")
    ap.add_argument("--sg-window-scout-free-wR2-tolerance", type=float, default=0.003, help="Maximum held-out wR2 penalty allowed when promoting a non-baseline Sg width.")
    ap.add_argument("--topup-d-min", type=float, default=0.8, help="Nominal Bloch support top-up dmin. Auto never uses a shallower support limit than the selected data dmin.")
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule used inside Bloch row predictions.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures; mean is the legacy normalized diagnostic.")
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--selection-metric",
        choices=["work_wR2", "free_wR2", "all_wR2", "all_R1"],
        default="work_wR2",
        help="Metric used to choose nuisance trials. Work-row wR2 keeps free rows held out; all/free choices are diagnostics.",
    )
    if expose_lambda_controls:
        ap.add_argument(
            "--lambda-grid",
            default=FULL_Q_LAMBDA,
            help="Diagnostic q-exponent grid for lower-level runs. Production auto uses the full Bloch/Born correction and does not tune this value.",
        )
    else:
        ap.set_defaults(lambda_grid=FULL_Q_LAMBDA)
    ap.add_argument(
        "--scale-smoothness",
        type=float,
        default=0.0,
        help="Experimental L2 frame-scale smoothness coefficient. Default 0 keeps per-group scales independent; GROND's default parameter reduction is the post-hoc scale-polish gate.",
    )
    ap.add_argument("--endpoint-b-grid", default="0.0", help="Corrected-HKL endpoint resolution-scale diagnostic grid. Nonzero values are opt-in only; GROND keeps B=0 as the final promotable endpoint whenever it is present.")
    ap.add_argument("--endpoint-exti-grid", default="off", help="Endpoint EXTI diagnostic grid. Final promoted outputs use off; explicit 'on' runs are comparison-only.")
    ap.add_argument("--shelxl-iterations", type=int, default=5, help="SHELXL executable calls per corrected-HKL endpoint in lower-level full mode. GROND normalizes each endpoint input to one internal LS/CGLS cycle unless --endpoint-ls-cycles is changed for diagnostics.")
    ap.add_argument("--endpoint-ls-cycles", type=int, default=1, help="Diagnostic internal L.S./CGLS cycle count for generated corrected-HKL endpoint files. Production keeps this at 1.")
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote"],
        default="freeze-zero",
        help="SHELXL weight policy for corrected-HKL endpoint files. freeze-zero is the production default after GROND sigma calibration; inherit keeps the source-model WGHT; promote uses SHELXL recommendations on later external calls.",
    )
    ap.add_argument("--shelxl", default=default_shelxl_executable(), help="SHELXL executable used for corrected-HKL endpoints and SHELXL feedback loops.")
    ap.add_argument("--shelxl-feedback-cycles", type=int, default=5, help="Number of high-N/Q Bloch -> one-SHELXL-LS feedback cycles used by the automatic SHELXL structure-update strategy.")
    ap.add_argument("--shelxl-feedback-shelxl-iterations", type=int, default=1, help="SHELXL passes per feedback cycle. Default 1 keeps each feedback update to one least-squares model update.")
    ap.add_argument("--shelxl-feedback-endpoint-ls-cycles", type=int, default=1, help="Diagnostic internal L.S./CGLS cycle count for each SHELXL feedback macrocycle. Production keeps this at 1.")
    ap.add_argument("--dataset-sigma-policy", choices=["auto", "none", "uniform_z_rms"], default="auto", help="Per-dataset endpoint sigma policy for multi-dataset combination. Auto calibrates each crystal independently before merging.")
    ap.add_argument("--dataset-sigma-holdout-fraction", type=float, default=0.20, help="Deterministic HKL holdout fraction reported for per-dataset endpoint sigma calibration.")
    ap.add_argument("--dataset-sigma-min-scale", type=float, default=0.25, help="Lower guardrail for per-dataset endpoint sigma scaling.")
    ap.add_argument("--dataset-sigma-max-scale", type=float, default=4.0, help="Upper guardrail for per-dataset endpoint sigma scaling.")
    ap.add_argument("--dataset-sigma-holdout-floor", action=argparse.BooleanOptionalAction, default=True, help="For multi-dataset auto, prevent a dataset from being assigned tighter sigmas than its held-out normalized residuals support.")
    ap.add_argument("--merge-sigma-policy", choices=["auto", "formal", "chisq_overdispersion"], default="auto", help="Sigma policy used when corrected rows are merged. Auto inflates merged sigmas when independent datasets disagree beyond formal errors.")
    ap.add_argument("--post-bloch-sigma-policy", choices=["auto", "none", "uniform_z_rms"], default="auto", help="Endpoint uncertainty policy. Auto uses uniform_z_rms only for final corrected endpoints and SHELXL feedback cycles, leaving early physics-selection endpoints uncalibrated.")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20, help="Deterministic HKL holdout fraction reported for uniform_z_rms endpoint sigma calibration.")
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25, help="Lower guardrail for uniform_z_rms endpoint sigma scaling.")
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0, help="Upper guardrail for uniform_z_rms endpoint sigma scaling.")
    ap.add_argument("--endpoint-feedback", choices=["auto", "on", "off"], default="auto", help="Optionally feed the polished B=0 no-EXTI endpoint model into one more Bloch endpoint cycle. Full auto uses auto; --fast skips it.")
    ap.add_argument("--publication-endpoint-chooser", action=argparse.BooleanOptionalAction, default=True, help="Write the publication endpoint chooser that separates physical and diagnostic corrected-HKL endpoints.")
    ap.add_argument("--publication-endpoint-min-hkl-safe-factor", type=float, default=0.25, help="Minimum fixed-width HKL safety factor for a publication endpoint.")
    ap.add_argument("--publication-endpoint-max-modulation-p95", type=float, default=20.0, help="Maximum 95th-percentile endpoint intensity modulation for a publication endpoint.")
    ap.add_argument("--publication-allow-npd", action=argparse.BooleanOptionalAction, default=False, help="Explicit diagnostic override allowing endpoints with NPD pathology to pass publication selection gates. Default requires NPD == 0; split warnings are reported but do not block selection.")
    ap.add_argument("--endpoint-calibration", choices=["auto", "on", "off"], default="auto", help="Replay the final frozen row table through corrected-HKL endpoint calibration after auto completes.")
    ap.add_argument("--endpoint-calibration-specs", default="baseline,adaptive:293:det8,adaptive:200:det8,adaptive:120:det8,adaptive:80:det8,window:0.5:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8,window:1.0:det4,window:3.0:det4,window:5.0:det4,window:3.0:det1")
    if expose_lambda_controls:
        ap.add_argument(
            "--endpoint-calibration-lambda-q",
            type=float,
            default=1.0,
            help="Diagnostic endpoint q exponent for replay tests. Production auto pins this to 1.0.",
        )
    else:
        ap.set_defaults(endpoint_calibration_lambda_q=1.0)
    ap.add_argument("--endpoint-calibration-wght-modes", default="freeze-zero,promote", help="Endpoint-calibration weight branches. The default tests fixed WGHT 0 0 before the compatibility branch.")
    ap.add_argument("--endpoint-calibration-prefer-wght-mode", default="freeze-zero")
    ap.add_argument("--endpoint-calibration-goof-min", type=float, default=0.85)
    ap.add_argument("--endpoint-calibration-goof-max", type=float, default=1.15)
    ap.add_argument("--endpoint-calibration-max-r1-cost", type=float, default=0.010)
    ap.add_argument("--endpoint-calibration-free-wr2-tolerance", type=float, default=0.040)
    ap.add_argument(
        "--endpoint-calibration-selection-policy",
        choices=["bic", "gated-smallest"],
        default="bic",
        help="Final scale-contract selector. Default applies endpoint sanity gates, then chooses the minimum BIC fitted-variance branch.",
    )
    ap.add_argument(
        "--endpoint-calibration-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
        help="Information criterion used by endpoint calibration when the BIC policy is active.",
    )
    ap.add_argument("--endpoint-calibration-sigma-min-scale", type=float, default=0.05)
    ap.add_argument("--endpoint-calibration-sigma-max-scale", type=float, default=4.0)
    ap.add_argument("--endpoint-calibration-shelxl-iterations", type=int, default=3, help="External one-cycle SHELXL calls for the final fixed-weight endpoint replay. This does not change LS1 feedback; it stabilizes the openable presentation endpoint.")
    ap.add_argument(
        "--instability-u",
        type=float,
        default=0.01,
        help="JANA/PETS-style row-weight instability factor u used by wR/MwR reporting and row-scale diagnostics.",
    )
    ap.add_argument("--scale-polish", choices=["auto", "on", "off"], default="auto", help="Run a post-hoc scale-window compression gate after the final Bloch row table.")
    ap.add_argument("--scale-polish-window-deg", type=float, default=5.0, help="Maximum physical frame-window width tested by the scale-polish gate.")
    ap.add_argument("--scale-polish-frame-models", default="adaptive,fixed", help="Frame-scale compression models tested by the scale-polish gate.")
    ap.add_argument("--scale-polish-adaptive-counts", default="auto", help="Adaptive neighboring-frame segment counts for scale polish; percentages such as 50%% are accepted.")
    ap.add_argument("--scale-polish-detector-bins", default="exact", help="Detector flat-field grids tested by the scale-polish gate. Default exact preserves the fitted detector grid while compressing frame scales.")
    ap.add_argument("--scale-polish-objective", choices=["row_mr", "native", "sigma"], default="native")
    ap.add_argument("--scale-polish-tolerance", type=float, default=0.01)
    ap.add_argument("--scale-polish-selection-metrics", default="Rall,wRall,native_R1_all,native_wR2_all")
    ap.add_argument("--scale-polish-scan-offsets", action=argparse.BooleanOptionalAction, default=False, help="Test all block-boundary offsets in the scale-polish gate.")
    ap.add_argument(
        "--multi-scale-contract-selection",
        choices=["auto", "off"],
        default="auto",
        help=(
            "For multi-dataset runs, replay strict compact per-crystal scales through the complete combined "
            "endpoint and promote the smallest contract that passes physics, held-out, coverage, and SHELXL gates."
        ),
    )
    ap.add_argument("--multi-scale-min-reduction", type=float, default=0.10, help="Minimum total scale-parameter reduction required for compact promotion.")
    ap.add_argument("--multi-scale-require-bic-improvement", action=argparse.BooleanOptionalAction, default=True, help="Require aggregate fitted-variance BIC to improve before compact promotion.")
    ap.add_argument("--multi-scale-max-row-wr2-cost", type=float, default=0.01, help="Maximum absolute compact wR2 cost on each of all, work, and free rows.")
    ap.add_argument("--multi-scale-max-endpoint-r1-cost", type=float, default=0.005, help="Maximum absolute compact R1all and R1gt endpoint cost.")
    ap.add_argument("--multi-scale-max-endpoint-wr2-cost", type=float, default=0.01, help="Maximum absolute compact endpoint wR2all cost.")
    ap.add_argument("--multi-scale-goof-min", type=float, default=0.85)
    ap.add_argument("--multi-scale-goof-max", type=float, default=1.15)
    ap.add_argument("--multi-scale-max-input-row-loss-fraction", type=float, default=0.001, help="Maximum corrected-row loss allowed when materializing compact scales.")
    ap.add_argument("--multi-scale-min-bloch-wr2-gain", type=float, default=0.002, help="Minimum pooled Bloch wR2 gain and minimum retained per-dataset gain where the full-scale control already passes.")
    ap.add_argument("--multi-scale-max-dataset-row-wr2-cost", type=float, default=0.015, help="Maximum compact-versus-full wR2 cost on any individual dataset/split.")
    ap.add_argument("--multi-scale-max-preexisting-bloch-gain-loss", type=float, default=0.01, help="Maximum additional Born-minus-Bloch loss allowed where the full-scale control already lacks a split-level Bloch win.")
    ap.add_argument(
        "--endpoint-min-d-spacing",
        type=float,
        default=0.0,
        help=(
            "Optional additional d-spacing cutoff for final combined row statistics and corrected-HKL endpoints. "
            "Zero keeps the row-selection cutoff; use an explicit value for an apples-to-apples reporting endpoint."
        ),
    )
    ap.add_argument("--resolution-limit-gate", dest="resolution_limit_gate", action="store_true", default=True, help="Promote per-dataset and merged resolution limits from held-out shell stability and GROND row/MR residuals.")
    ap.add_argument("--no-resolution-limit-gate", dest="resolution_limit_gate", action="store_false")
    ap.add_argument("--resolution-shells", type=int, default=10)
    ap.add_argument("--resolution-strict-split-cc", type=float, default=0.30)
    ap.add_argument("--resolution-strict-mean-i-over-sigma", type=float, default=0.50)
    ap.add_argument("--resolution-exploratory-split-cc", type=float, default=0.20)
    ap.add_argument("--resolution-exploratory-mean-i-over-sigma", type=float, default=0.25)
    ap.add_argument("--resolution-min-common-pairs", type=int, default=8)
    ap.add_argument("--resolution-min-shell-rows", type=int, default=20)
    ap.add_argument("--resolution-min-strong-fraction", type=float, default=0.005)
    ap.add_argument("--resolution-max-shell-mwrall", type=float, default=0.60)
    ap.add_argument("--resolution-max-free-work-mwr-gap", type=float, default=0.12)
    ap.add_argument("--resolution-min-free-rows", type=int, default=5)
    ap.add_argument("--resolution-max-cumulative-mwr-penalty", type=float, default=0.03)
    ap.add_argument("--dataset-policy", choices=["auto", "off"], default="auto", help="For multi-dataset jobs, automatically keep good crystals at full weight, downweight marginal crystals, and keep weak nonredundant crystals endpoint-only.")
    ap.add_argument("--dataset-policy-min-rows", type=int, default=100)
    ap.add_argument("--dataset-policy-min-weight", type=float, default=0.25)
    ap.add_argument("--dataset-policy-primary-min-quality", type=float, default=0.75)
    ap.add_argument("--dataset-policy-min-raw-quality", type=float, default=0.20)
    ap.add_argument("--dataset-policy-min-cc", type=float, default=0.70)
    ap.add_argument("--dataset-policy-max-wr2", type=float, default=0.75)
    ap.add_argument("--dataset-policy-max-r1", type=float, default=0.60)
    ap.add_argument("--dataset-policy-unique-gain-fraction", type=float, default=0.01)
    ap.add_argument("--dataset-policy-wr2-scale", type=float, default=0.08)
    ap.add_argument("--dataset-policy-r1-scale", type=float, default=0.08)
    ap.add_argument("--dataset-policy-cc-scale", type=float, default=0.04)
    ap.add_argument(
        "--multi-endpoint-merge-key",
        choices=["model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"],
        default="model_symmetry",
        help=(
            "Multi-dataset corrected-HKL merge key before each shared SHELXL LS step. "
            "model_symmetry merges symmetry-equivalent corrected observations while preserving Friedel opposites in acentric groups."
        ),
    )
    ap.add_argument("--row-resolution-b-grid", default="0.0,14,18", help="Raw-row dynamical resolution-envelope grid. Positive values are the dynamic-gradient counterpart of negative endpoint-B diagnostics.")
    ap.add_argument("--thickness-grid", default="1000,1200,1400")
    ap.add_argument("--thickness-model-grid", default="uniform_gradient")
    ap.add_argument("--thickness-grad-grid", default="0.4,1.0")
    ap.add_argument("--continuous-thickness-polish", choices=["auto", "on", "off"], default="auto", help="After the coarse thickness grid, polish continuous thickness/gradient by reusing cached Bloch eigensystems. Auto is on for GROND auto except --fast.")
    ap.add_argument(
        "--continuous-thickness-metric",
        choices=["work_wR2", "all_wR2", "free_wR2", "all_R1", "all_R1gt"],
        default="work_wR2",
        help="Objective used by continuous thickness polish. Work-row wR2 keeps free rows held out.",
    )
    ap.add_argument("--continuous-thickness-strong-rows", type=int, default=0, help="Rows used by continuous thickness polish; 0 uses all selected rows. Strong-row-only scouts are diagnostic and are not the default.")
    ap.add_argument("--continuous-thickness-max-rows", type=int, default=0, help="Hard row limit for continuous thickness polish; 0 uses all rows selected by the data filters.")
    ap.add_argument("--continuous-thickness-bracket-frac", type=float, default=0.35, help="Local fractional thickness bracket around the coarse-grid winner.")
    ap.add_argument("--continuous-thickness-grid-step", type=float, default=100.0, help="Guard-grid thickness step in Angstrom.")
    ap.add_argument("--continuous-thickness-grad-step", type=float, default=0.2, help="Guard-grid thickness-gradient step.")
    ap.add_argument("--continuous-thickness-powell-maxiter", type=int, default=18, help="Maximum Powell iterations for the final phase-only thickness/gradient polish.")
    ap.add_argument("--continuous-thickness-min-gain", type=float, default=5e-5, help="Minimum objective gain required to promote a non-grid continuous thickness.")
    ap.add_argument("--continuous-thickness-max-free-wr2-loss", type=float, default=0.002, help="Maximum tolerated held-out/free wR2 loss for promoted continuous thickness.")
    ap.add_argument("--continuous-thickness-max-all-r1-loss", type=float, default=0.001, help="Maximum tolerated all-row R1 loss for promoted continuous thickness.")
    ap.add_argument(
        "--frame-offset-grid",
        default="auto",
        help=(
            "Profile-center offsets in XDS frames, or 'auto'. Auto reconciles the global XDS header "
            "orientation with row ZCAL positions before work-row nuisance selection."
        ),
    )
    ap.add_argument(
        "--group-width-grid",
        default="auto",
        help="Scale-group width grid. Numeric values are degrees; frame, 2frame, 3frames, and auto resolve from XDS OSCILLATION_RANGE. Auto never goes below one physical frame; explicit numeric sub-frame values are diagnostic opt-ins.",
    )
    ap.add_argument("--detector-scale-bins", default="8", help="Comma grid of NxN detector-position scale fields tested during full auto. Fast mode uses 0; use 8,12 for an explicit high-parameter detector diagnostic.")
    ap.add_argument("--sigma-a-grid", default="0.0")
    ap.add_argument("--sigma-b-corr-grid", default="0.0")
    ap.add_argument("--sigma-c-peak-grid", default="0.0,0.15")
    ap.add_argument("--sg-jitter-mode-grid", default="none,gaussian")
    ap.add_argument("--sg-jitter-sigma0-grid", default="0.0,0.0015")
    ap.add_argument("--sg-jitter-sigma-d-inv-grid", default="0.0")
    ap.add_argument("--sg-jitter-sigma-frame-grid", default="0.0")
    ap.add_argument("--sg-jitter-points", type=int, default=3)
    ap.add_argument(
        "--pinned-sg-jitter-fallback",
        choices=["none", "off"],
        default="none",
        help=(
            "When a pinned nonzero Sg-jitter convolution is locally infeasible, also score an explicit no-jitter "
            "candidate. This preserves the local-root bound and records any selected nuisance change."
        ),
    )
    ap.add_argument("--absorption-mode-grid", default="none,midgley_like")
    ap.add_argument("--absorption-mu-grid", default="0.0,0.00005")
    ap.add_argument("--absorption-frac-grid", default="0.0,0.02")
    ap.add_argument("--absorption-mu-g2-grid", default="0.0")
    ap.add_argument("--absorption-mu-theta2-grid", default="0.0")
    ap.add_argument("--absorption-tds-b-grid", default="0.0")
    ap.add_argument("--dose-b-mode-grid", default="none")
    ap.add_argument("--dose-b-per-frame-grid", default="0.0")
    ap.add_argument("--thickness-points", type=int, default=5, help="Thickness quadrature points for screen/regularization stages.")
    ap.add_argument("--final-thickness-points", type=int, default=9, help="Thickness quadrature points for pinned final-basis polish runs.")
    add_forensic_args(ap, expose_lambda_controls=expose_lambda_controls)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level GROND argument parser."""

    parser = argparse.ArgumentParser(
        prog="GROND",
        description="GROND: Global Regularization Of Non-kinematical Diffraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prep", help="Write a JSON prep file for one structure and one or more XDS datasets.")
    p_prep.add_argument("--structure", required=True)
    p_prep.add_argument("--baseline-hkl", default="")
    p_prep.add_argument("--dataset", action="append", required=True)
    p_prep.add_argument("--hkl-transform", default="identity")
    p_prep.add_argument("--geometry-percentile", type=float, default=99.0)
    p_prep.add_argument("--notes", default="")
    p_prep.add_argument("--out", required=True)
    add_tool_config_args(p_prep)
    p_prep.set_defaults(func=cmd_prep)

    p_cfg = sub.add_parser("init-config", help="Write a GROND JSON/YAML external-tool config template.")
    p_cfg.add_argument("--out", default="grond.config.json")
    p_cfg.add_argument("--format", choices=["json", "yaml"], default="json")
    p_cfg.add_argument("--overwrite", action="store_true")
    p_cfg.set_defaults(func=cmd_init_config)

    p_row_template = sub.add_parser("row-template", help="Write portable GROND row-contract CSV and metadata templates.")
    p_row_template.add_argument("--rows-out", default="grond_rows.csv")
    p_row_template.add_argument("--metadata-out", default="grond_rows_metadata.json")
    p_row_template.add_argument("--overwrite", action="store_true")
    p_row_template.set_defaults(func=cmd_row_template)

    p_row_validate = sub.add_parser("row-validate", help="Validate a portable GROND row table and optional geometry sidecar.")
    p_row_validate.add_argument("--rows", required=True, help="Portable row CSV, TSV, JSONL, or NDJSON file.")
    p_row_validate.add_argument("--metadata", required=True, help="Portable row metadata JSON file.")
    p_row_validate.add_argument("--write-xds-dir", default="", help="Optional output directory for an XDS-like compatibility INTEGRATE.HKL.")
    p_row_validate.add_argument("--overwrite", action="store_true")
    p_row_validate.set_defaults(func=cmd_row_validate)

    p_sfac = sub.add_parser("sfac-check", help="Check a RES/INS model for explicit electron SFAC rows and endpoint wavelength readiness.")
    p_sfac.add_argument("--structure", required=True, help="SHELX .res/.ins model to check.")
    p_sfac.add_argument("--out", default="", help="Optional JSON report path.")
    p_sfac.set_defaults(func=cmd_sfac_check)

    p_free = sub.add_parser("make-free-flags", help="Assign fixed macromolecular FreeR flags at unique-HKL level.")
    p_free.add_argument("--rows", required=True, help="Signed row table used to define the unique-HKL population.")
    p_free.add_argument("--out", required=True, help="Output CSV table with h,k,l,free_flag.")
    p_free.add_argument("--model", default="", help="Optional PDB/mmCIF/RES/CIF model used to read cell and space group.")
    p_free.add_argument("--cell", default="", help="Six cell constants: a b c alpha beta gamma.")
    p_free.add_argument("--space-group", default="", help="Space group symbol, for example 'P 21 21 21'.")
    p_free.add_argument("--fraction", type=float, default=0.05, help="Target FreeR fraction on unique HKLs.")
    p_free.add_argument("--min-free-count", type=int, default=500, help="Minimum FreeR reflections when enough unique reflections exist.")
    p_free.add_argument("--seed", type=int, default=20260622)
    p_free.add_argument("--shells", type=int, default=10, help="Balance FreeR flags approximately across resolution shells when d-spacings are known.")
    p_free.add_argument("--merge-anomalous", action=argparse.BooleanOptionalAction, default=True, help="Assign FreeR flags after Friedel merging. Use --no-merge-anomalous to keep anomalous mates separate.")
    p_free.add_argument("--free-value", type=int, default=1)
    p_free.add_argument("--work-value", type=int, default=0)
    p_free.add_argument("--write-rows", default="", help="Optional signed-row CSV output with FreeR_flag propagated to every row.")
    p_free.add_argument("--summary", default="", help="Optional JSON summary path. Defaults to OUT.summary.json.")
    p_free.set_defaults(func=cmd_make_free_flags)

    p_mtz = sub.add_parser("export-mtz", help="Merge corrected GROND rows into a macromolecular MTZ with fixed FreeR flags.")
    p_mtz.add_argument("--row-table", required=True, help="Corrected or GROND-native row CSV/TSV/JSONL table.")
    p_mtz.add_argument("--free-flags", required=True, help="Unique-HKL FreeR flag table from make-free-flags or an equivalent wrapper.")
    p_mtz.add_argument("--out-mtz", required=True)
    p_mtz.add_argument("--out-dir", default="", help="Directory for merged_corrected_rows.csv and export_mtz_summary.json.")
    p_mtz.add_argument("--model", default="", help="Optional model used to read cell and space group.")
    p_mtz.add_argument("--cell", default="", help="Six cell constants: a b c alpha beta gamma.")
    p_mtz.add_argument("--space-group", default="", help="Space group symbol, for example 'P 21 21 21'.")
    p_mtz.add_argument("--dataset", default="GROND")
    p_mtz.add_argument("--sigma-policy", choices=["formal", "chisq_overdispersion"], default="formal")
    p_mtz.add_argument("--q-policy", choices=["full", "low_resolution_damped", "off"], default="full")
    p_mtz.add_argument("--q-off-d-A", type=float, default=5.0, help="Low-resolution q damping is off at and below this resolution when using --q-policy low_resolution_damped.")
    p_mtz.add_argument("--q-full-d-A", type=float, default=3.0, help="Low-resolution q damping reaches full correction at and above this resolution.")
    p_mtz.add_argument("--merge-anomalous", action=argparse.BooleanOptionalAction, default=True)
    p_mtz.set_defaults(func=cmd_export_mtz)

    p_protein = sub.add_parser("protein-ls1", help="Run one configured macromolecular endpoint macrocycle against a corrected MTZ.")
    add_tool_config_args(p_protein)
    p_protein.add_argument("--model", required=True, help="Input PDB/mmCIF model for the endpoint macrocycle.")
    p_protein.add_argument("--mtz", required=True, help="Corrected MTZ from GROND export-mtz.")
    p_protein.add_argument("--out-dir", required=True)
    p_protein.add_argument("--endpoint", choices=["custom", "phenix_refine", "refmac5", "servalcat"], default="custom")
    p_protein.add_argument("--command", dest="endpoint_command", default="", help="Endpoint command template. Placeholders: {model}, {mtz}, {reflections}, {out_dir}, {cycles}, {free_label}.")
    p_protein.add_argument("--output-model", default="", help="Output model path or template, relative to --out-dir unless absolute.")
    p_protein.add_argument("--cycles", type=int, default=1)
    p_protein.add_argument("--free-label", default=FREE_FLAG_LABEL)
    p_protein.add_argument("--dry-run", action="store_true")
    p_protein.set_defaults(func=cmd_protein_ls1)

    p_endpoints = sub.add_parser("endpoints", help="List GROND model-update endpoint adapters and configured tool availability.")
    add_tool_config_args(p_endpoints)
    p_endpoints.add_argument("--json", action="store_true")
    p_endpoints.set_defaults(func=cmd_endpoints)

    p_dials = sub.add_parser("import-dials", help="Convert DIALS integrated.refl/integrated.expt into a GROND row directory.")
    p_dials.add_argument("--refl", required=True, help="DIALS integrated.refl or scaled.refl file.")
    p_dials.add_argument("--expt", required=True, help="DIALS integrated.expt or scaled.expt file.")
    p_dials.add_argument("--out-dir", required=True)
    p_dials.add_argument("--dials-python", default="", help="Python executable that can import dials and dxtbx.")
    p_dials.add_argument("--intensity-mode", choices=["sum_complete", "sum_raw", "scale"], default="sum_complete")
    p_dials.add_argument("--sigma-mode", choices=["auto", "raw", "detector_gain"], default="raw")
    p_dials.add_argument("--sigma-scale", type=float, default=None)
    p_dials.add_argument("--min-partiality-write", type=float, default=1e-6)
    p_dials.add_argument("--drop-dont-integrate", action="store_true", help="Deprecated alias for the default production behavior. Rows flagged dont_integrate are dropped unless --keep-dont-integrate is used.")
    p_dials.add_argument("--keep-dont-integrate", action="store_true", help="Diagnostic only: keep DIALS rows flagged dont_integrate.")
    p_dials.add_argument("--keep-bad-flags", action="store_true", help="Diagnostic only: keep DIALS rows with severe bad-pixel, overloaded, powder-ring, failed-background, or failed-summation flags.")
    p_dials.add_argument("--structure", default="", help="Optional RES/CIF path used to also write a prep JSON.")
    p_dials.add_argument("--baseline-hkl", default="", help="Optional solved-model HKL path stored in the prep JSON.")
    p_dials.add_argument("--dataset-name", default="dials")
    p_dials.add_argument("--prep-out", default="", help="Optional prep JSON output path. Defaults to OUT/grond_prep.json when --structure is supplied.")
    p_dials.set_defaults(func=cmd_import_dials)

    p_run = sub.add_parser("run", help="Run fixed-structure Bloch, dynamic regularization, or full auto mode.")
    add_common_run_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_auto = sub.add_parser("auto", help="Simple one-command full automatic workflow.")
    add_common_run_args(p_auto, expose_lambda_controls=False)
    p_auto.set_defaults(func=cmd_auto)

    p_raw = sub.add_parser("raw-polish", help="Experimental raw-CBF Beanland/Cleverley-style forward observation polish.")
    p_raw.add_argument("--prep", default="")
    p_raw.add_argument("--dataset-name", default="")
    p_raw.add_argument("--hkl", default="")
    p_raw.add_argument("--xds-dir", default="")
    p_raw.add_argument("--structure", default="")
    p_raw.add_argument("--baseline-hkl", default="")
    p_raw.add_argument("--hkl-transform", default="")
    p_raw.add_argument("--cbf-dir", default="")
    p_raw.add_argument("--images", default="")
    p_raw.add_argument("--out-dir", required=True)
    p_raw.add_argument("--fast", action="store_true")
    p_raw.add_argument("--python", default="")
    add_tool_config_args(p_raw)
    p_raw.add_argument("--dry-run", action="store_true")
    p_raw.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p_raw.add_argument("--workers", type=int, default=0)
    p_raw.add_argument("--max-core-fraction", type=float, default=0.85)
    p_raw.add_argument("--chunksize", type=int, default=0)
    p_raw.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy")
    p_raw.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off", help="Experimental frozen row-geometry cache. Off is default until full workflow benchmarks show a clear win.")
    p_raw.add_argument("--jax-batch-size", type=int, default=0)
    p_raw.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    p_raw.add_argument("--bethe-strong-count", type=int, default=48)
    p_raw.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    p_raw.add_argument("--bethe-promote-denom", type=float, default=0.005)
    p_raw.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    p_raw.add_argument("--prediction-cache-dir", default="")
    p_raw.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_raw.add_argument("--raw-grid", default="small:2,4,2.5,4,6")
    p_raw.add_argument("--raw-integration-modes", default="sum,bloch_profile_fit")
    p_raw.add_argument("--background-models", default="global_median")
    p_raw.add_argument("--profile-sigma-xy-grid", default="0")
    p_raw.add_argument("--profile-sigma-z-grid", default="0")
    p_raw.add_argument("--matrix-conventions", default="j_minus_i,i_minus_j")
    p_raw.add_argument("--targets", default="raw_complete,xds_iobs_calibrated")
    p_raw.add_argument("--bloch-profile-n-basis", type=int, default=20)
    p_raw.add_argument("--bloch-profile-thickness", type=float, default=1200.0)
    p_raw.add_argument("--bloch-profile-thickness-grad-frac", type=float, default=1.0)
    p_raw.add_argument("--bloch-profile-absorption-mode", default="midgley_like")
    p_raw.add_argument("--bloch-profile-absorption-mu", type=float, default=0.00005)
    p_raw.add_argument("--bloch-profile-absorption-frac", type=float, default=0.02)
    p_raw.add_argument("--bloch-profile-z-smooth-frames", type=float, default=0.0)
    p_raw.add_argument("--bloch-profile-z-gaussian-blend", type=float, default=0.0)
    p_raw.add_argument("--bloch-profile-z-gaussian-sigma", type=float, default=0.0)
    p_raw.add_argument("--n-basis", type=int, default=20)
    p_raw.add_argument("--n-points", type=int, default=10)
    p_raw.add_argument("--sg-window", type=float, default=0.004)
    p_raw.add_argument("--min-d-spacing", type=float, default=0.70)
    p_raw.add_argument("--min-peak", type=float, default=90.0)
    p_raw.add_argument("--thickness-grid", default="1000,1200,1400")
    p_raw.add_argument("--thickness-grad-grid", default="0.4,1.0")
    p_raw.add_argument(
        "--frame-offset-grid",
        default="auto",
        help=(
            "Profile-center offsets in XDS frames, or 'auto'. Auto reconciles the global XDS header "
            "orientation with row ZCAL positions before work-row nuisance selection."
        ),
    )
    p_raw.add_argument("--group-width-grid", default="auto")
    p_raw.add_argument("--detector-scale-bins", type=int, default=8)
    p_raw.add_argument("--thickness-points", type=int, default=5)
    p_raw.add_argument(
        "--lambda-grid",
        default=FULL_Q_LAMBDA,
        help="Diagnostic q-exponent grid. Production auto uses the full Bloch/Born correction with lambda_q=1.0.",
    )
    p_raw.add_argument("--free-fraction", type=float, default=0.10)
    p_raw.add_argument("--split-seed", type=int, default=42)
    p_raw.add_argument("--limit", type=int, default=0)
    p_raw.add_argument("--top-isigma", type=int, default=0)
    p_raw.add_argument("--even-sample", type=int, default=0)
    p_raw.add_argument("--include-nonpositive", action="store_true")
    p_raw.add_argument("--calibrate-geometry", action=argparse.BooleanOptionalAction, default=True)
    p_raw.add_argument("--make-maps", dest="make_maps", action="store_true", default=True)
    p_raw.add_argument("--no-maps", dest="make_maps", action="store_false")
    p_raw.set_defaults(func=cmd_raw_polish)

    p_forensics = sub.add_parser("forensics", help="Run post-Bloch row-space forensic audits on native_dyn_calc.csv.")
    p_forensics.add_argument("--row-table", required=True)
    p_forensics.add_argument("--hkl", required=True)
    p_forensics.add_argument("--res", required=True)
    p_forensics.add_argument("--out-dir", required=True)
    p_forensics.add_argument("--python", default="")
    add_tool_config_args(p_forensics)
    p_forensics.add_argument("--dry-run", action="store_true")
    p_forensics.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_forensic_args(p_forensics)
    p_forensics.set_defaults(func=cmd_forensics)

    p_scale = sub.add_parser("scale-polish", help="Compress frozen-Bloch row-space scale factors after a final GROND run.")
    p_scale.add_argument("--row-table", required=True)
    p_scale.add_argument("--summary", default="")
    p_scale.add_argument("--out-dir", required=True)
    p_scale.add_argument("--title", default="GROND scale-compression polish")
    p_scale.add_argument("--python", default="")
    add_tool_config_args(p_scale)
    p_scale.add_argument("--dry-run", action="store_true")
    p_scale.add_argument("--fit-split", default="")
    p_scale.add_argument("--frame-models", default="adaptive,fixed")
    p_scale.add_argument("--adaptive-frame-counts", default="auto")
    p_scale.add_argument("--group-strides", default="1,2,4,8,16,32,64,128,global")
    p_scale.add_argument("--window-scan-deg", type=float, default=0.0, help="Also scan every integer group stride up to this physical window size.")
    p_scale.add_argument("--scan-offsets", action="store_true", help="Test all block-boundary offsets for each finite group stride.")
    p_scale.add_argument("--detector-bins", default="exact")
    p_scale.add_argument("--scale-objective", choices=["row_mr", "native", "sigma"], default="native")
    p_scale.add_argument("--tolerance", type=float, default=0.01)
    p_scale.add_argument("--selection-metrics", default="Rall,wRall,native_R1_all,native_wR2_all")
    p_scale.add_argument("--native-wght-a", type=float, default=None)
    p_scale.add_argument("--native-wght-b", type=float, default=None)
    p_scale.set_defaults(func=cmd_scale_polish)

    p_scale_replay = sub.add_parser("scale-replay", help="Replay compact scale contracts through corrected-HKL refinement endpoints.")
    p_scale_replay.add_argument("--row-table", required=True)
    p_scale_replay.add_argument("--summary", default="")
    p_scale_replay.add_argument("--res", default="", help="Source RES file. Defaults to the RES path recorded in the row summary.")
    p_scale_replay.add_argument("--out-dir", required=True)
    p_scale_replay.add_argument("--python", default="")
    add_tool_config_args(p_scale_replay)
    p_scale_replay.add_argument("--dry-run", action="store_true")
    p_scale_replay.add_argument("--specs", default="baseline,adaptive:293:det8,adaptive:200:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8", help="Comma list of scale contracts: baseline, adaptive:N:det8, window:DEG:det8, or stride:N:det8.")
    p_scale_replay.add_argument("--fit-split", default="all")
    p_scale_replay.add_argument(
        "--lambda-q",
        type=float,
        default=1.0,
        help="Diagnostic endpoint q exponent. Production auto pins this to 1.0.",
    )
    p_scale_replay.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_scale_replay.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    p_scale_replay.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    p_scale_replay.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_scale_replay.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    p_scale_replay.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_scale_replay.add_argument("--shelxl", default="shelxl")
    p_scale_replay.add_argument("--shelxl-iterations", type=int, default=5)
    p_scale_replay.add_argument("--wght-modes", default="promote,freeze-zero", help="Comma list of endpoint weight policies; freeze-zero keeps active WGHT 0 0.")
    p_scale_replay.add_argument("--publication-goof-min", type=float, default=0.85)
    p_scale_replay.add_argument("--publication-goof-max", type=float, default=1.15)
    p_scale_replay.add_argument("--publication-max-r1-cost", type=float, default=0.010)
    p_scale_replay.add_argument("--publication-free-wr2-tolerance", type=float, default=0.040)
    p_scale_replay.add_argument("--publication-max-modulation-p95", type=float, default=20.0)
    p_scale_replay.add_argument("--publication-prefer-wght-mode", default="freeze-zero")
    p_scale_replay.add_argument("--publication-allow-npd", action=argparse.BooleanOptionalAction, default=False, help="Explicit diagnostic override allowing NPD pathology in the selected endpoint. Default requires NPD == 0; split warnings are reported but do not block selection.")
    p_scale_replay.add_argument(
        "--publication-selection-policy",
        choices=["bic", "gated-smallest"],
        default="bic",
        help="Endpoint scale-contract selector. Default applies endpoint sanity gates, then chooses the minimum BIC fitted-variance branch.",
    )
    p_scale_replay.add_argument(
        "--publication-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
    )
    p_scale_replay.add_argument("--instability-u", type=float, default=0.01)
    p_scale_replay.add_argument("--native-wght-a", type=float, default=None)
    p_scale_replay.add_argument("--native-wght-b", type=float, default=None)
    p_scale_replay.set_defaults(func=cmd_scale_replay)

    p_cal_endpoint = sub.add_parser("calibrate-endpoint", help="Choose a corrected-HKL publication endpoint from a frozen GROND row table.")
    p_cal_endpoint.add_argument("--row-table", required=True)
    p_cal_endpoint.add_argument("--summary", default="")
    p_cal_endpoint.add_argument("--res", default="", help="Source RES file. Defaults to the RES path recorded in the row summary.")
    p_cal_endpoint.add_argument("--out-dir", required=True)
    p_cal_endpoint.add_argument("--python", default="")
    add_tool_config_args(p_cal_endpoint)
    p_cal_endpoint.add_argument("--dry-run", action="store_true")
    p_cal_endpoint.add_argument(
        "--specs",
        default="baseline,adaptive:293:det8,adaptive:200:det8,adaptive:120:det8,adaptive:80:det8,window:0.5:det8,window:1.0:det8,window:1.5:det8,window:3.0:det8,window:5.0:det8,window:1.0:det4,window:3.0:det4,window:5.0:det4,window:3.0:det1",
        help="Comma list of scale contracts: baseline, adaptive:N:det8, window:DEG:det8, or stride:N:det8.",
    )
    p_cal_endpoint.add_argument("--fit-split", default="all")
    p_cal_endpoint.add_argument(
        "--lambda-q",
        type=float,
        default=1.0,
        help="Diagnostic endpoint q exponent. Production auto pins this to 1.0.",
    )
    p_cal_endpoint.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_cal_endpoint.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    p_cal_endpoint.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    p_cal_endpoint.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_cal_endpoint.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.05)
    p_cal_endpoint.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_cal_endpoint.add_argument("--shelxl", default="shelxl")
    p_cal_endpoint.add_argument("--shelxl-iterations", type=int, default=3)
    p_cal_endpoint.add_argument("--wght-modes", default="freeze-zero,promote", help="Comma list of endpoint weight policies; freeze-zero keeps active WGHT 0 0.")
    p_cal_endpoint.add_argument("--publication-goof-min", type=float, default=0.85)
    p_cal_endpoint.add_argument("--publication-goof-max", type=float, default=1.15)
    p_cal_endpoint.add_argument("--publication-max-r1-cost", type=float, default=0.010)
    p_cal_endpoint.add_argument("--publication-free-wr2-tolerance", type=float, default=0.040)
    p_cal_endpoint.add_argument("--publication-max-modulation-p95", type=float, default=20.0)
    p_cal_endpoint.add_argument("--publication-prefer-wght-mode", default="freeze-zero")
    p_cal_endpoint.add_argument("--publication-allow-npd", action=argparse.BooleanOptionalAction, default=False, help="Explicit diagnostic override allowing NPD pathology in the selected endpoint. Default requires NPD == 0; split warnings are reported but do not block selection.")
    p_cal_endpoint.add_argument(
        "--publication-selection-policy",
        choices=["bic", "gated-smallest"],
        default="bic",
        help="Endpoint scale-contract selector. Default applies endpoint sanity gates, then chooses the minimum BIC fitted-variance branch.",
    )
    p_cal_endpoint.add_argument(
        "--publication-information-criterion",
        choices=["bic_fitted_variance", "aic_fitted_variance", "bic_known_sigma", "aic_known_sigma"],
        default="bic_fitted_variance",
    )
    p_cal_endpoint.add_argument("--instability-u", type=float, default=0.01)
    p_cal_endpoint.add_argument("--native-wght-a", type=float, default=None)
    p_cal_endpoint.add_argument("--native-wght-b", type=float, default=None)
    p_cal_endpoint.set_defaults(func=cmd_calibrate_endpoint)

    if os.environ.get("GROND_ENABLE_INTERNAL_CLI", "").strip() == "1":
        p_target_ablation = sub.add_parser(
            "target-ablation",
            help="Internal diagnostic: run XDS IOBS/RLP/PEAK target-transform ablations.",
        )
        p_target_ablation.add_argument("--xds-dir", default="", help="XDS directory containing INTEGRATE.HKL.")
        p_target_ablation.add_argument("--hkl", default="", help="Explicit XDS INTEGRATE.HKL path. Use this instead of --xds-dir when needed.")
        p_target_ablation.add_argument("--structure", required=True)
        p_target_ablation.add_argument("--baseline-hkl", required=True)
        p_target_ablation.add_argument("--out-dir", required=True)
        p_target_ablation.add_argument("--variants", default=",".join(XDS_TARGET_MODES), help="Comma list of XDS target modes, or all.")
        p_target_ablation.add_argument("--python", default="")
        add_tool_config_args(p_target_ablation)
        p_target_ablation.add_argument("--shelxl", default=default_shelxl_executable())
        p_target_ablation.add_argument("--shelxl-feedback-cycles", type=int, default=5)
        p_target_ablation.add_argument("--shelxl-feedback-shelxl-iterations", type=int, default=1)
        p_target_ablation.add_argument("--endpoint-ls-cycles", type=int, default=1)
        p_target_ablation.add_argument("--hkl-transform", default="identity")
        p_target_ablation.add_argument("--trust-hkl-transform", action="store_true")
        p_target_ablation.add_argument("--min-d-spacing", type=float, default=0.70)
        p_target_ablation.add_argument("--min-peak", type=float, default=90.0)
        p_target_ablation.add_argument("--geometry-percentile", type=float, default=99.0)
        p_target_ablation.add_argument("--workers", type=int, default=0)
        p_target_ablation.add_argument("--max-core-fraction", type=float, default=0.85)
        p_target_ablation.add_argument("--chunksize", type=int, default=0)
        p_target_ablation.add_argument("--compute-backend", default="scipy")
        p_target_ablation.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off")
        p_target_ablation.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
        p_target_ablation.add_argument("--hand-control", action=argparse.BooleanOptionalAction, default=True)
        p_target_ablation.add_argument("--make-maps", action=argparse.BooleanOptionalAction, default=False, help="Ablation skips maps by default to keep the four branches fast.")
        p_target_ablation.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
        p_target_ablation.add_argument("--dry-run", action="store_true")
        p_target_ablation.add_argument("--fail-fast", action="store_true")
        p_target_ablation.add_argument("--extra-auto-args", default="", help="Advanced: quoted argument string appended to every inner auto command.")
        p_target_ablation.set_defaults(func=cmd_target_ablation)

    p_thick = sub.add_parser("thickness-scout", help="Audit effective thickness sensitivity to row contract, scale model, and accepted nuisance terms.")
    add_common_run_args(p_thick, expose_lambda_controls=False)
    p_thick.set_defaults(mode="regularize", n_points=10)
    p_thick.add_argument("--n-basis-grid", default="24,48", help="Comma grid of basis sizes. The first value uses the broad grid; later values use a local confirmation grid by default.")
    p_thick.add_argument("--thickness-grid-nm", default="50,75,110,150,180,220,260,300", help="Broad central-thickness grid in nanometers.")
    p_thick.add_argument("--gradient-grid", default="0.0,0.4,0.8,1.0", help="Physical uniform-gradient fraction grid (0 through 1).")
    p_thick.add_argument("--scale-modes", default="production,no-detector", help="Comma grid: production and/or no-detector.")
    p_thick.add_argument("--feature-modes", default="bare,final-diagnostic", help="Comma grid: bare and/or final-diagnostic.")
    p_thick.add_argument("--local-confirmation", action=argparse.BooleanOptionalAction, default=True, help="Use local thickness/gradient grids for basis values after the first broad scout.")
    p_thick.add_argument("--local-window-nm", type=float, default=30.0)
    p_thick.add_argument("--local-step-nm", type=float, default=10.0)
    p_thick.add_argument("--local-gradient-window", type=float, default=0.2)
    p_thick.add_argument("--local-gradient-step", type=float, default=0.2)
    p_thick.add_argument("--thickness-min-nm", type=float, default=10.0)
    p_thick.add_argument("--thickness-max-nm", type=float, default=500.0)
    p_thick.add_argument("--gradient-min", type=float, default=0.0)
    p_thick.add_argument("--gradient-max", type=float, default=1.0)
    p_thick.add_argument("--warning-shift-nm", type=float, default=75.0, help="Warn when a selected thickness is this far from the promoted reference.")
    p_thick.add_argument("--thickness-scout-max-rows", type=int, default=0, help="Optional row cap for scout runs. 0 uses the selected production row set.")
    p_thick.add_argument("--thickness-scout-thickness-points", type=int, default=5, help="Thickness quadrature points inside each candidate thickness model.")
    p_thick.add_argument("--thickness-scout-final-sg-jitter-mode", default="gaussian")
    p_thick.add_argument("--thickness-scout-final-sg-jitter-sigma0", type=float, default=0.0015)
    p_thick.add_argument("--thickness-scout-final-sg-jitter-sigma-d-inv", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-sg-jitter-sigma-frame", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-sg-jitter-points", type=int, default=3)
    p_thick.add_argument("--thickness-scout-final-absorption-mode", default="midgley_like")
    p_thick.add_argument("--thickness-scout-final-absorption-mu", type=float, default=0.00005)
    p_thick.add_argument("--thickness-scout-final-absorption-frac", type=float, default=0.02)
    p_thick.add_argument("--thickness-scout-final-absorption-mu-g2", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-absorption-mu-theta2", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-absorption-tds-b", type=float, default=0.0)
    p_thick.set_defaults(func=cmd_thickness_scout)

    p_validate_ls1 = sub.add_parser("validate-ls1", help="Run LS1 fixed-point controls and optional matched native finite-difference polish.")
    p_validate_ls1.add_argument("--source-summary", required=True, help="Existing native_row_regularize summary whose selected nuisance values should be pinned.")
    p_validate_ls1.add_argument("--start-res", required=True, help="Initial RES model for updated-q cycle 1.")
    p_validate_ls1.add_argument("--out-dir", required=True)
    p_validate_ls1.add_argument("--python", default="")
    add_tool_config_args(p_validate_ls1)
    p_validate_ls1.add_argument("--dry-run", action="store_true")
    p_validate_ls1.add_argument("--hkl", default="", help="Override HKL path if the source summary does not record it.")
    p_validate_ls1.add_argument("--baseline-hkl", default="", help="Optional conventional HKL for baseline/reference endpoint comparisons.")
    p_validate_ls1.add_argument("--hkl-transform", default="", help="Override signed HKL transform. Defaults to the source summary.")
    p_validate_ls1.add_argument("--cycles", type=int, default=5, help="Updated-q LS1 macrocycles to run.")
    p_validate_ls1.add_argument("--n-basis", type=int, default=30, help="Bloch basis for validation cycles. Use the final adaptive basis for publication adjudication.")
    p_validate_ls1.add_argument("--n-points", type=int, default=20)
    p_validate_ls1.add_argument("--shelxl", default=default_shelxl_executable())
    p_validate_ls1.add_argument("--shelxl-iterations", type=int, default=1, help="SHELXL iterations per updated-q macrocycle.")
    p_validate_ls1.add_argument("--workers", type=int, default=auto_worker_count(0, 0.85))
    p_validate_ls1.add_argument("--chunksize", type=int, default=0)
    p_validate_ls1.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy_bethe")
    p_validate_ls1.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off", help="Experimental frozen row-geometry cache. Off is default until full workflow benchmarks show a clear win.")
    p_validate_ls1.add_argument("--jax-batch-size", type=int, default=0)
    p_validate_ls1.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    p_validate_ls1.add_argument("--bethe-strong-count", type=int, default=48)
    p_validate_ls1.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    p_validate_ls1.add_argument("--bethe-promote-denom", type=float, default=0.005)
    p_validate_ls1.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    p_validate_ls1.add_argument("--min-d-spacing", type=float, default=0.0)
    p_validate_ls1.add_argument("--topup-d-min", type=float, default=0.0)
    p_validate_ls1.add_argument("--min-peak", type=float, default=-1.0)
    p_validate_ls1.add_argument("--geometry-percentile", type=float, default=0.0)
    p_validate_ls1.add_argument("--max-rows", type=int, default=0)
    p_validate_ls1.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms")
    p_validate_ls1.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_validate_ls1.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    p_validate_ls1.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_validate_ls1.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_validate_ls1.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_validate_ls1.add_argument("--best-cycle-tolerance", type=float, default=0.001, help="Pick the earliest cycle within this row wR(I)all fraction of the best cycle.")
    p_validate_ls1.add_argument("--q-control-seed", type=int, default=12345)
    p_validate_ls1.add_argument("--q-control-shells", type=int, default=10)
    p_validate_ls1.add_argument("--skip-controls", action="store_true")
    p_validate_ls1.add_argument("--wrong-hand-control", dest="wrong_hand_control", action=argparse.BooleanOptionalAction, default=True, help="Run the alternate-hand null control.")
    p_validate_ls1.add_argument("--native-polish", action=argparse.BooleanOptionalAction, default=False)
    p_validate_ls1.add_argument("--native-polish-rows", type=int, default=200)
    p_validate_ls1.add_argument("--native-polish-n-basis", type=int, default=0)
    p_validate_ls1.add_argument("--native-polish-n-points", type=int, default=0)
    p_validate_ls1.add_argument("--native-polish-blocks", default="adp_ueq,xyz")
    p_validate_ls1.add_argument("--native-polish-atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    p_validate_ls1.add_argument("--native-polish-max-nfev", type=int, default=12)
    p_validate_ls1.add_argument("--native-polish-early-stop-window", type=int, default=32)
    p_validate_ls1.add_argument("--native-polish-early-stop-min-evals", type=int, default=0)
    p_validate_ls1.add_argument("--native-polish-early-stop-min-improvement", type=float, default=1e-4)
    p_validate_ls1.add_argument("--native-polish-free-wR2-tolerance", type=float, default=0.02)
    p_validate_ls1.set_defaults(func=cmd_validate_ls1)

    p_push = sub.add_parser("push-endpoint", help="Continue a pinned GROND endpoint from an existing summary and start model.")
    p_push.add_argument("--checkpoint", default="", help="grond_continue_checkpoint.json from a completed auto run.")
    p_push.add_argument("--source-summary", default="", help="Existing native_row_regularize summary whose selected nuisance values should be pinned.")
    p_push.add_argument("--start-res", default="", help="Starting RES model to continue from. Overrides the checkpoint model when provided.")
    p_push.add_argument("--out-dir", required=True)
    p_push.add_argument("--python", default="")
    add_tool_config_args(p_push)
    p_push.add_argument("--dry-run", action="store_true")
    p_push.add_argument("--hkl", default="", help="Override HKL path if the source summary does not record it.")
    p_push.add_argument("--baseline-hkl", default="", help="Optional conventional HKL for baseline/reference endpoint comparisons.")
    p_push.add_argument("--hkl-transform", default="", help="Override signed HKL transform. Defaults to the source summary.")
    p_push.add_argument(
        "--multi-endpoint-merge-key",
        choices=["model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"],
        default="",
        help="For multi-dataset checkpoints, corrected-HKL merge key before each shared SHELXL LS step. Defaults to the checkpoint value or model_symmetry.",
    )
    p_push.add_argument("--cycles", type=int, default=3, help="Additional Bloch/SHELXL macrocycles.")
    p_push.add_argument("--n-basis", type=int, default=108)
    p_push.add_argument("--n-points", type=int, default=20)
    p_push.add_argument("--final-n-basis", type=int, default=0)
    p_push.add_argument("--final-n-points", type=int, default=20)
    p_push.add_argument("--shelxl", default=default_shelxl_executable())
    p_push.add_argument("--shelxl-iterations", type=int, default=1, help="SHELXL executable calls per macrocycle.")
    p_push.add_argument("--endpoint-ls-cycles", type=int, default=1, help="Diagnostic internal L.S./CGLS cycle count for each generated endpoint. Production keeps this at 1.")
    p_push.add_argument("--workers", type=int, default=auto_worker_count(0, 0.85))
    p_push.add_argument("--chunksize", type=int, default=4)
    p_push.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy")
    p_push.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off")
    p_push.add_argument(
        "--pinned-sg-jitter-fallback",
        choices=["none", "off"],
        default="none",
        help="Revalidate an infeasible pinned Sg-jitter model against an explicit no-jitter candidate.",
    )
    p_push.add_argument("--jax-batch-size", type=int, default=0)
    p_push.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    p_push.add_argument("--bethe-strong-count", type=int, default=48)
    p_push.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    p_push.add_argument("--bethe-promote-denom", type=float, default=0.005)
    p_push.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    p_push.add_argument("--min-d-spacing", type=float, default=0.0)
    p_push.add_argument(
        "--endpoint-min-d-spacing",
        type=float,
        default=0.0,
        help="Optional cutoff applied again to each combined corrected-HKL endpoint; zero follows --min-d-spacing.",
    )
    p_push.add_argument("--topup-d-min", type=float, default=0.7)
    p_push.add_argument("--min-peak", type=float, default=-1.0)
    p_push.add_argument("--geometry-percentile", type=float, default=0.0)
    p_push.add_argument("--max-rows", type=int, default=0)
    p_push.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="none")
    p_push.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_push.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    p_push.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_push.add_argument(
        "--lambda-grid",
        default=FULL_Q_LAMBDA,
        help="Diagnostic q-exponent grid. Production auto uses the full Bloch/Born correction with lambda_q=1.0.",
    )
    p_push.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    p_push.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_push.add_argument("--make-maps-on-final", action="store_true")
    p_push.add_argument("--best-cycle-tolerance", type=float, default=0.001, help="Safety report tolerance above the best row wR(I)all.")
    p_push.add_argument(
        "--model-prep",
        choices=["as-is", "no-restraints", "free-h-anis-no-restraints", "adp-stabilized"],
        default="as-is",
        help="Optional start-model preparation before the push. adp-stabilized adds RIGU/SIMU/targeted ISOR and converts known NPD atoms to isotropic fallback; free-H is a stress-test branch.",
    )
    p_push.add_argument("--set-ls-cycles", type=int, default=1, help="Rewrite L.S. to this value in the prepared start model. Use 0 to preserve it.")
    p_push.set_defaults(func=cmd_push_endpoint)

    p_finish = sub.add_parser("finish", help="Regenerate final summaries, native dynamical maps, and the GROND dynamic CIF from a completed run.")
    p_finish.add_argument("--run-dir", default=".", help="Completed GROND auto/run output directory. Defaults to the current directory.")
    p_finish.add_argument("--checkpoint", default="", help="Optional grond_continue_checkpoint.json; its run_dir is used.")
    p_finish.add_argument("--model", default="", help="Optional model override for native dynamical map phases. Defaults to the selected publication endpoint RES.")
    add_tool_config_args(p_finish)
    p_finish.add_argument(
        "--make-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate finish maps. Native row-level dynamical maps are the default; use --no-make-maps to skip.",
    )
    p_finish.add_argument(
        "--map-mode",
        choices=["native", "shelx2map", "both", "off"],
        default="native",
        help="native writes row-level dynamical maps; shelx2map writes quick corrected-HKL endpoint maps; both writes both.",
    )
    p_finish.add_argument("--native-map-spacing", type=float, default=0.25, help="Approximate CCP4 grid spacing in Angstroms for native dynamical maps.")
    p_finish.add_argument("--native-map-d-min", type=float, default=0.0, help="Optional low-pass d_min cutoff for native dynamical maps. Zero uses all row-table coefficients.")
    p_finish.add_argument("--native-map-min-i-over-sigma", type=float, default=0.0, help="Minimum row I/sigma used for native dynamical map amplitudes.")
    p_finish.add_argument("--native-map-max-grid-points", type=int, default=8_000_000, help="Safety cap on native dynamical map grid points.")
    p_finish.add_argument("--json", action="store_true", help="Print the compact JSON payload used by automation.")
    p_finish.set_defaults(func=cmd_finish)

    p_publish = sub.add_parser("publish", help="Alias for finish: regenerate final summaries, native dynamical maps, and the GROND dynamic CIF.")
    p_publish.add_argument("--run-dir", default=".", help="Completed GROND auto/run output directory. Defaults to the current directory.")
    p_publish.add_argument("--checkpoint", default="", help="Optional grond_continue_checkpoint.json; its run_dir is used.")
    p_publish.add_argument("--model", default="", help="Optional model override for native dynamical map phases. Defaults to the selected publication endpoint RES.")
    add_tool_config_args(p_publish)
    p_publish.add_argument("--make-maps", action=argparse.BooleanOptionalAction, default=True)
    p_publish.add_argument("--map-mode", choices=["native", "shelx2map", "both", "off"], default="native")
    p_publish.add_argument("--native-map-spacing", type=float, default=0.25)
    p_publish.add_argument("--native-map-d-min", type=float, default=0.0)
    p_publish.add_argument("--native-map-min-i-over-sigma", type=float, default=0.0)
    p_publish.add_argument("--native-map-max-grid-points", type=int, default=8_000_000)
    p_publish.add_argument("--json", action="store_true")
    p_publish.set_defaults(func=cmd_finish)

    p_val = sub.add_parser("validate", help="Compile the vendored engine scripts with the selected Python.")
    p_val.add_argument("--python", default="")
    add_tool_config_args(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_doc = sub.add_parser("doctor", help="Report Python packages, accelerator visibility, and the active compute core.")
    p_doc.add_argument("--python", default="")
    add_tool_config_args(p_doc)
    p_doc.add_argument("--probe-bloch", action="store_true", help="Also test complex eig/eigh support needed by the Bloch wave kernel.")
    p_doc.add_argument("--strict", action="store_true", help="Exit nonzero when required packages, installed features, or requested Bloch operations are not ready.")
    p_doc.set_defaults(func=cmd_doctor)

    p_compile = sub.add_parser("compile", help="Compile-check GROND and vendored engine modules without writing bytecode into the source tree.")
    p_compile.add_argument("--python", default="")
    add_tool_config_args(p_compile)
    p_compile.add_argument("--force", action="store_true")
    p_compile.set_defaults(func=cmd_compile)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Package entry point used by both `grond` and `GROND` scripts."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "command", "") != "init-config":
            apply_tool_config(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nGROND was interrupted. Re-run the same command with the same --out-dir to resume completed stages.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
