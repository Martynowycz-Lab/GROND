"""Run child stages with shared progress, logging and scout concurrency limits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from grond.cli_support.values import parse_numeric_grid, try_read_json
from grond.io.config import PrepConfig
from grond.paths import PACKAGE_ROOT
from grond.progress import ProgressState, format_seconds
from grond.runtime import apply_thread_limits, auto_worker_count
from grond.workflows.conventional_baseline import _publication_baseline_hkl
from grond.workflows.decision import expand_transform_candidates, safe_float, safe_label
from grond.workflows.nuisance_policy import FEATURE_GATE_FAMILIES, feature_is_enabled

_PROGRESS: ProgressState | None = None


_RUN_LOG_LOCK = threading.Lock()


_TRANSFORM_SCOUT_SEMAPHORE: threading.BoundedSemaphore | None = None


_TRANSFORM_SCOUT_SEMAPHORE_LIMIT = 0


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
    dyn_all = (metrics.get("all") or {}).get("dyn") or {}
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


def _estimated_resolution_steps(args: argparse.Namespace) -> int:
    """Estimate resolution-promotion commands for progress reporting."""

    return 1


def _estimated_sg_window_scout_steps(args: argparse.Namespace) -> int:
    """Estimate late-stage rocking-window scout commands."""

    from grond.workflows.sg_window_selection import scout_widths

    mode = getattr(args, "sg_window_scout", "auto")
    if mode == "off" or getattr(args, "fast", False):
        return 0
    grid = parse_numeric_grid(getattr(args, "sg_window_scout_grid", ""))
    return len(scout_widths(grid, float(getattr(args, "sg_window", 0.004))))


def estimate_run_steps(prep: PrepConfig, args: argparse.Namespace) -> int:
    """Estimate how many external stages this run will launch."""

    total = 1 if _publication_baseline_hkl(prep) else 0
    for ds in prep.datasets:
        if not getattr(args, "trust_hkl_transform", False):
            transform_candidates = expand_transform_candidates(
                ds.hkl_transform,
                parity_policy=getattr(args, "transform_parity_policy", "proper-only"),
            )
            total += len(transform_candidates)
            if len(transform_candidates) > 1:
                total += min(TRANSFORM_RERANK_TOP, len(transform_candidates))
        total += 1
        if getattr(args, "hand_control", True):
            # Two label-symmetric nuisance-grid evaluations followed by two
            # singleton evidence replays at the selected common nuisance.
            total += 4
            if str(getattr(args, "hand_decision_stage", "final")) == "early":
                # Early held-out hand pair and its paired-row diagnostic after the screen.
                total += 2
        if not getattr(args, "fast", False):
            total += 1
            total += sum(
                1
                for family in FEATURE_GATE_FAMILIES
                if feature_is_enabled(args, family)
            )
            if getattr(args, "mode", "auto") in {"regularize", "auto"}:
                total += 1
            total += 1
        total += 1
        if len(prep.datasets) == 1 and getattr(args, "hand_control", True):
            total += 2
        if getattr(args, "h_omit", "off") in {"on", "both", "only"}:
            total += 2
        total += _estimated_resolution_steps(args)
        total += _estimated_sg_window_scout_steps(args)
    if len(prep.datasets) > 1:
        if getattr(args, "hand_control", True):
            # One shared opposite-model materialization plus one native
            # recomputation and paired-row diagnostic per publication crystal.
            total += 1 + 2 * len(prep.datasets)
        total += 2
        total += _estimated_resolution_steps(args)
    return max(total, 1)


def run_logged(cmd: list[str], log_dir: Path, label: str, *, dry_run: bool = False, cwd: Path = PACKAGE_ROOT) -> None:
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
        print(
            "DONE", label, "returncode", proc.returncode, "elapsed", format_seconds(done["elapsed_seconds"]), flush=True
        )
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


# Second-stage transform scouts: the best TRANSFORM_RERANK_TOP coarse candidates are rerun at higher N/Q.
TRANSFORM_RERANK_TOP = 3
# One worker process inside each transform-scout subprocess; the scouts run as parallel subprocesses.
TRANSFORM_SCOUT_WORKERS_PER_JOB = 1


def transform_scout_job_count(args: argparse.Namespace, task_count: int | None = None) -> int:
    """Maximum number of transform-scout subprocesses GROND should launch at once (the core budget of --workers)."""

    budget = auto_worker_count(int(getattr(args, "workers", 0) or 0), float(getattr(args, "max_core_fraction", 0.85)))
    jobs = max(1, budget // TRANSFORM_SCOUT_WORKERS_PER_JOB)
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
