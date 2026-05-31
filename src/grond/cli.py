from __future__ import annotations

import argparse
import compileall
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    tool_env_updates,
    tool_statuses,
    write_config_template,
)
from .paths import LEGACY_ROOT, engine_script
from .progress import ProgressState, format_seconds
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
from .workflows.endpoint_chooser import write_publication_endpoint_choice
from .workflows.transform_metadata import metadata_prefilter_transforms

_PROGRESS: ProgressState | None = None
_RUN_LOG_LOCK = threading.Lock()
_TRANSFORM_SCOUT_SEMAPHORE: threading.BoundedSemaphore | None = None
_TRANSFORM_SCOUT_SEMAPHORE_LIMIT = 0


def configure_progress(log_dir: Path, *, resume: bool, total: int) -> None:
    """Attach CLI progress and resume tracking to the current run."""

    global _PROGRESS
    _PROGRESS = ProgressState(root=log_dir.parent, resume=resume, total=total)


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
            "  raw rows:",
            f"R1all={_percent(dyn_all.get('R1_all'))}",
            f"R1strong={_percent(dyn_all.get('R1_gt'))}",
            f"wR2all={_percent(dyn_all.get('wR2_all'))}",
            f"CC={safe_float(dyn_all.get('CC'), float('nan')):.5f}",
            f"rowB={safe_float(nuisance.get('row_resolution_b'), 0.0):.3g}",
            flush=True,
        )
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    if shelxl:
        endpoint_bits = [
            f"lambda={safe_float(endpoint.get('lambda_q'), float('nan')):.3g}",
            f"B={safe_float(endpoint.get('endpoint_b'), 0.0):.3g}",
            f"EXTI={'on' if endpoint.get('keep_exti') else 'off'}",
        ]
        print(
            "  SHELXL endpoint:",
            *endpoint_bits,
            f"R1gt={_percent(shelxl.get('R1_gt'))}",
            f"R1all={_percent(shelxl.get('R1_all'))}",
            f"wR2all={_percent(shelxl.get('wR2_all'))}",
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


def estimate_run_steps(prep: PrepConfig, args: argparse.Namespace) -> int:
    """Estimate how many external stages this run will launch."""

    total = 0
    for ds in prep.datasets:
        if getattr(args, "auto_decide", True):
            transform_candidates = expand_transform_candidates(ds.hkl_transform, args.transform_candidates)
            total += len(transform_candidates)
            rerank_top = max(0, int(getattr(args, "transform_rerank_top", 0)))
            if len(transform_candidates) > 1 and rerank_top > 0:
                total += min(rerank_top, len(transform_candidates))
        total += 1
        if getattr(args, "coordinate_polarity_control", True):
            total += 1
        if not getattr(args, "fast", False):
            total += 1
            total += sum(1 for family in ("row_resolution", "absorption", "sg_jitter", "sigma_error", "dose") if feature_is_enabled(args, family))
            if getattr(args, "mode", "auto") in {"regularize", "auto"}:
                total += 1
            total += 1
        total += 1
        if getattr(args, "coordinate_polarity_control", True):
            total += 2
        if getattr(args, "h_omit", "off") in {"on", "both", "only"}:
            total += 2
        if getattr(args, "scale_polish", "auto") != "off":
            total += 1
        total += _estimated_resolution_steps(args)
        total += _estimated_forensic_steps(args)
    if len(prep.datasets) > 1:
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
        if progress and not dry_run and progress.should_skip(label):
            print(progress.line_prefix(label, skipped=True), flush=True)
            return
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


def best_endpoint_model(run_dir: Path) -> Path | None:
    """Return the polished endpoint RES path for a completed native stage."""

    summary = try_read_json(run_dir / "summary.json")
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    lst = shelxl.get("lst")
    if not lst:
        return None
    res = Path(lst).with_suffix(".res")
    return res if res.exists() else None


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
    args.tool_config_path = str(path) if path else ""
    args.tool_config = config
    args.configured_tools = tools


def external_tool_statuses(args: argparse.Namespace) -> dict[str, dict]:
    """Return the active external-tool status block for run configs and doctor."""

    return tool_statuses(getattr(args, "tool_config", {}) or {})


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
        and not multi_dataset
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

    The SHELXL feedback loop is the preferred single-dataset default whenever a
    SHELXL executable is available because it reaches the same high-N/Q
    row-space endpoint as finite-difference dynamical gradients at a small
    fraction of the wall time. Multi-dataset shared-structure update still
    uses the native gradient engine until there is an equivalent combined
    feedback loop.
    """

    requested = getattr(args, "structure_update", "auto")
    if requested in {"off", "native_gradient", "shelxl_feedback"}:
        return requested
    if multi_dataset or getattr(args, "fast", False) or getattr(args, "full_fast", False) or getattr(args, "native_only", False):
        return "native_gradient"
    if shelxl_is_available(getattr(args, "shelxl", "shelxl")):
        return "shelxl_feedback"
    return "native_gradient"


def shelxl_feedback_current_model(run_dir: Path) -> Path | None:
    """Return the latest model produced by a SHELXL feedback-loop run."""

    summary = try_read_json(run_dir / "shelxl_feedback_loop_summary.json")
    current = summary.get("current_model")
    if current and Path(current).exists():
        return Path(current)
    for rec in reversed(summary.get("runs") or []):
        output = rec.get("output_model")
        if output and Path(output).exists():
            return Path(output)
    return None


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
                recommendations.append(
                    f"best polished endpoint lambda={safe_float(endpoint.get('lambda_q'), 0.0):.3g}, "
                    f"R1all={100.0 * safe_float(sh.get('R1_all'), float('nan')):.2f}%, "
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

    lambda_grid = getattr(args, "observation_polish_lambda_grid", "") or getattr(args, "lambda_grid", "0.90,0.95,1.00")
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


def coordinate_polarity_policy(structure: str) -> dict:
    """Decide whether an alternate coordinate-polarity check makes sense for a model."""

    path = Path(structure)
    suffix = path.suffix.lower()
    policy = {
        "structure": str(path),
        "run_coordinate_polarity_control": True,
        "reason": "alternate coordinate-polarity check may be informative for acentric models",
    }
    try:
        if suffix in {".res", ".ins"}:
            for line in path.read_text(errors="replace").splitlines():
                parts = line.strip().split()
                if parts and parts[0].upper() == "LATT" and len(parts) > 1:
                    latt = int(float(parts[1]))
                    policy["shelx_latt"] = latt
                    if latt > 0:
                        policy.update(run_coordinate_polarity_control=False, reason="SHELX LATT is centrosymmetric; alternate coordinate-polarity check is not meaningful")
                    else:
                        policy.update(run_coordinate_polarity_control=True, reason="SHELX LATT is acentric; alternate coordinate-polarity check is applicable")
                    return policy
            policy["reason"] = "no SHELX LATT record found; keeping polarity control enabled"
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
                        policy.update(run_coordinate_polarity_control=False, reason="CIF space group is centrosymmetric; alternate coordinate-polarity check is not meaningful")
                    else:
                        policy.update(run_coordinate_polarity_control=True, reason="CIF space group is acentric; alternate coordinate-polarity check is applicable")
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
                        policy.update(run_coordinate_polarity_control=False, reason="CIF symmetry operators include inversion; alternate coordinate-polarity check is not meaningful")
                        return policy
            return policy
    except Exception as exc:
        policy.update(reason=f"could not determine polarity-control symmetry policy ({exc}); keeping polarity control enabled")
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
    coordinate_polarity_alt: bool = False,
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
    if coordinate_polarity_alt:
        cmd.append("--coordinate-polarity-alt")
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
        return f"{float(value):.10g}"
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
    assign("polarity_n_basis", min(int(args.polarity_n_basis), n_basis))
    assign("polarity_n_points", min(int(args.polarity_n_points), n_points))
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


def make_coordinate_polarity_model(python: str, structure: str, out_dir: Path, logs: Path, dry_run: bool) -> Path:
    """Create or reuse the alternate coordinate-polarity model variant."""

    src = Path(structure)
    dst = out_dir / "input" / f"{src.stem}_coordinate_polarity_alt{src.suffix or '.res'}"
    if dst.exists():
        return dst
    cmd = [python, str(engine_script("make_res_variants.py")), "--input", str(src), "--output", str(dst), "--coordinate-polarity-alt"]
    run_logged(cmd, logs, "make_coordinate_polarity_model", dry_run=dry_run)
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
    coordinate_polarity_alt: bool = False,
    make_maps: bool = True,
    endpoint: bool = True,
    max_rows_override: int | None = None,
    workers_override: int | None = None,
) -> list[str]:
    """Build one native nuisance-regularization command for the vendored engine."""

    pinned = pinned or {}
    fast = bool(args.fast)
    endpoint_ok = bool(endpoint and is_shelx_model(structure) and not args.native_only)
    lambda_grid = args.lambda_grid if endpoint_ok else ""
    if fast and endpoint_ok:
        lambda_grid = first(args.lambda_grid, "0.95")
    endpoint_b_grid = args.endpoint_b_grid if endpoint_ok and not fast else "0.0"
    endpoint_exti_grid = args.endpoint_exti_grid if endpoint_ok and not fast else "off"
    shelxl_iterations = args.shelxl_iterations if endpoint_ok and not fast else 1
    finalish_stage = stage in {"final", "noH", "polarity", "adaptive_final"} or stage.endswith("_final")
    thickness_grid = grid_from_nuisance(pinned, "thickness_A", "1200") if pinned else ("1000,1200,1400" if fast else args.thickness_grid)
    grad_grid = grid_from_nuisance(pinned, "thickness_grad_frac", "0.4") if pinned else ("0.0,0.4" if fast else args.thickness_grad_grid)
    frame_grid = grid_from_nuisance(pinned, "frame_offset", "0.0") if pinned else ("0.0" if fast else args.frame_offset_grid)
    group_grid = grid_from_nuisance(pinned, "group_width_deg", "1.0") if pinned else ("frame,2frame" if fast else args.group_width_grid)
    sigma_a = grid_from_nuisance(pinned, "sigma_a", "0.0") if pinned else ("0.0" if fast else args.sigma_a_grid)
    sigma_b = grid_from_nuisance(pinned, "sigma_b_corr", "0.0") if pinned else "0.0"
    sigma_c = grid_from_nuisance(pinned, "sigma_c_peak", "0.0") if pinned else ("0.0" if fast else args.sigma_c_peak_grid)
    row_resolution_b = grid_from_nuisance(pinned, "row_resolution_b", "0.0") if pinned else ("0.0" if fast else args.row_resolution_b_grid)
    detector_scale_bins = grid_from_nuisance(pinned, "detector_scale_bins", "0") if pinned else ("0" if fast else args.detector_scale_bins)
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
        f"{args.sg_window:g}",
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
        grid_from_nuisance(pinned, "sg_jitter_mode", "none") if pinned else ("none" if fast else args.sg_jitter_mode_grid),
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
        "all" if args.production_scales else "work",
        "--selection-metric",
        args.selection_metric,
        "--lambda-grid",
        lambda_grid,
        "--scale-smoothness",
        str(args.scale_smoothness),
        "--endpoint-b-grid",
        endpoint_b_grid,
        "--endpoint-exti-grid",
        endpoint_exti_grid,
        "--shelxl-iterations",
        str(shelxl_iterations),
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
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--matrix-convention",
        args.matrix_convention,
        "--hkl-float-format",
        "shelx_fixed",
    ]
    if ds.include_nonpositive:
        cmd.append("--include-nonpositive")
    else:
        cmd.append("--positive-only")
    max_rows = args.max_rows if max_rows_override is None else max_rows_override
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    if coordinate_polarity_alt:
        cmd.append("--coordinate-polarity-alt")
    if make_maps and endpoint_ok:
        cmd.append("--make-maps")
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
        "--jax-batch-size",
        str(args.jax_batch_size),
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
        first(args.lambda_grid, "0.95"),
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
        "--hkl-transform",
        ds.hkl_transform,
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
        "--lambda-grid",
        first(args.lambda_grid, "1.0"),
        "--rescale",
        "median-iobs",
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--jax-batch-size",
        str(args.jax_batch_size),
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


def combine_cmd(python: str, runs: dict[str, Path], structure: str, out_dir: Path, args: argparse.Namespace) -> list[str]:
    """Build the multi-dataset corrected-HKL combination command."""

    cmd = [
        python,
        str(engine_script("combine_native_nuisance_runs.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        structure,
        "--endpoint-lambda",
        first(args.lambda_grid, "0.95"),
        "--endpoint-source",
        "native",
        "--interdataset-scale",
        "overlap",
    ]
    for name, run in runs.items():
        cmd += ["--run", f"{name}={run}"]
    return cmd


def publication_report_cmd(python: str, prep: PrepConfig, result: dict, out_dir: Path, title: str) -> list[str]:
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
    ]
    if result.get("combined"):
        for name, run in (result.get("publication_runs") or result.get("runs") or {}).items():
            cmd += ["--run", f"{name}={run}"]
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
        for name, payload in scale_polish.items():
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
    return active or runs


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
        write_json(artifacts_path, artifacts)
        return {"status": "ok", **artifacts}
    return {
        "status": "missing_artifact_summary",
        "markdown": str((out_dir / "publication_table.md").resolve()),
        "latex": str((out_dir / "publication_table.tex").resolve()),
        "summary": str((out_dir / "row_mr_metrics.json").resolve()),
    }


def run_publication_endpoint_chooser(prep: PrepConfig, result: dict, args: argparse.Namespace) -> dict:
    """Write the publication-vs-diagnostic endpoint chooser report."""

    if not getattr(args, "publication_endpoint_chooser", True):
        return {"status": "disabled"}
    out_dir = Path(args.out_dir) / "endpoint_chooser"
    if args.dry_run:
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
    return write_publication_endpoint_choice(
        runs,
        out_dir,
        combined_run=combined_run,
        forensic_summaries=forensic_summaries,
        min_hkl_safe_factor=args.publication_endpoint_min_hkl_safe_factor,
        max_modulation_p95=args.publication_endpoint_max_modulation_p95,
    )


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
    widest = summary_payload.get("selected_widest")
    return {
        "status": "accepted" if selected else "diagnostic_only",
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "candidate_csv": str((out_dir / "scale_compression_candidates.csv").resolve()),
        "selected_compact_run": str((out_dir / "selected_compact_run").resolve()) if (out_dir / "selected_compact_run").exists() else "",
        "selected": selected or {},
        "selected_widest": widest or {},
    }


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


def choose_polarity_by_scout(
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
    """Compare the active model against an alternate coordinate-polarity scout."""

    active_summary = read_json(active_run / "summary.json")
    active_score = score_native_summary(active_summary)
    nui = best_nuisance(active_run)
    polarity_dir = root / "decisions" / ds.name / "polarity_scout_alt"
    try:
        run_logged(
            native_stage_cmd(
                python,
            ds,
            prep.structure,
            dataset_baseline_hkl(prep, ds),
            polarity_dir,
                args,
                n_basis=args.scout_n_basis,
                n_points=args.scout_n_points,
                stage="scout_polarity",
                pinned=nui,
                coordinate_polarity_alt=True,
                make_maps=False,
                endpoint=False,
                max_rows_override=args.scout_max_rows if scout_max_rows is None else scout_max_rows,
            ),
            logs,
            f"{ds.name}_scout_polarity_alt",
            dry_run=args.dry_run,
        )
        alternate_summary = read_json(polarity_dir / "summary.json")
        alternate_score = score_native_summary(alternate_summary)
    except Exception as exc:
        payload = {
            "selected": "original",
            "active_score": active_score,
            "alternate_score": {},
            "score_delta_original_minus_alternate": float("-inf"),
            "accept_delta": float(args.polarity_accept_delta),
            "alternate_run": str(polarity_dir),
            "warning": f"alternate coordinate-polarity scout failed: {exc}",
        }
        write_json(root / "decisions" / ds.name / "coordinate_polarity_decision.json", payload)
        return "original", payload, polarity_dir
    delta = active_score["score"] - alternate_score["score"]
    selected = "alternate" if delta > float(args.polarity_accept_delta) else "original"
    payload = {
        "selected": selected,
        "active_score": active_score,
        "alternate_score": alternate_score,
        "score_delta_original_minus_alternate": delta,
        "accept_delta": float(args.polarity_accept_delta),
        "alternate_run": str(polarity_dir),
    }
    write_json(root / "decisions" / ds.name / "coordinate_polarity_decision.json", payload)
    return selected, payload, polarity_dir


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
    stop, reasons = should_stop_for_bad_data(screen_summary, max_wR2=args.stop_if_wR2_above, min_improvement=args.min_bloch_wR2_gain)
    if stop and args.stop_on_bad_fit:
        write_json(root / "decisions" / ds.name / "stop_decision.json", {"stop": True, "reasons": reasons, "screen": str(screen)})
        return {"status": "stopped_bad_fit", "primary_run": str(screen), "reasons": reasons}

    active_model = prep.structure
    active_nui = best_nuisance(screen)
    coordinate_polarity_decision = coordinate_polarity_policy(prep.structure)
    active_polarity = "original"
    if args.coordinate_polarity_control and coordinate_polarity_decision.get("run_coordinate_polarity_control", True) and not args.dry_run:
        selected_polarity, coordinate_polarity_decision, alternate_probe = choose_polarity_by_scout(prep, ds, args, python, logs, root, screen, scout_max_rows=scout_rows)
        if selected_polarity == "alternate" and is_shelx_model(prep.structure):
            active_model = str(make_coordinate_polarity_model(python, prep.structure, root, logs, args.dry_run))
            active_nui = best_nuisance(alternate_probe) or active_nui
            active_polarity = "alternate"

    regularized_nui = active_nui
    thickness_run = screen
    feature_records: list[dict] = []
    if not args.fast:
        thickness_run = root / "runs" / ds.name / "adaptive_thickness_polish"
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
                pinned=thickness_polish_nuisance(active_nui),
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_adaptive_thickness_polish",
            dry_run=args.dry_run,
        )
        regularized_nui = best_nuisance(thickness_run) or active_nui
        feature_source = thickness_run
        continuous_seed = continuous_thickness_seed_nuisance(args, regularized_nui, active_nui)
        if continuous_thickness_enabled(args) and continuous_seed:
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
            feature_source = continuous_run
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
                runs = feedback_summary.get("runs") or []
                if runs:
                    last_run = runs[-1].get("run_dir")
                    if last_run:
                        feedback_final_run = Path(last_run)
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
    coordinate_polarity_diagnostic = ""
    if args.coordinate_polarity_control and coordinate_polarity_decision.get("run_coordinate_polarity_control", True) and not args.dry_run:
        other = root / "runs" / ds.name / "adaptive_other_polarity_check"
        other_model = prep.structure if active_polarity == "alternate" else final_model
        alternate_other = active_polarity != "alternate"
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
                stage="adaptive_other_polarity",
                pinned=best_nuisance(final) or regularized_nui,
                coordinate_polarity_alt=alternate_other,
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_adaptive_other_polarity_check",
            dry_run=args.dry_run,
        )
        stats_path = root / "runs" / ds.name / "adaptive_coordinate_polarity_diagnostic.json"
        run_logged(
            [
                python,
                str(engine_script("coordinate_polarity_diagnostic.py")),
                "--best",
                str(final / "native_dyn_calc.csv"),
                "--other",
                str(other / "native_dyn_calc.csv"),
                "--out",
                str(stats_path),
            ],
            logs,
            f"{ds.name}_adaptive_coordinate_polarity_diagnostic",
            dry_run=args.dry_run,
        )
        coordinate_polarity_diagnostic = str(stats_path)

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
    decision_summary = {
        "status": "ok",
        "dataset": ds.name,
        "selected_transform": ds.hkl_transform,
        "selected_polarity": active_polarity,
        "structure_update_strategy": structure_strategy,
        "structure_update_run": str(structure_update_run) if structure_update_run else "",
        "transform_decision": transform_decision,
        "coordinate_polarity_decision": coordinate_polarity_decision,
        "feature_decisions": feature_records,
        "endpoint_feedback_decision": endpoint_feedback_decision,
        "endpoint_feedback_run": str(endpoint_feedback_run) if endpoint_feedback_run else "",
        "scale_polish": scale_polish,
        "resolution_limit": resolution_limit,
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
        "coordinate_polarity_diagnostic": coordinate_polarity_diagnostic,
        "forensics": forensics,
        "scale_polish": scale_polish,
        "resolution_limits": {ds.name: resolution_limit},
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
    coordinate_polarity_diagnostic = ""
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
    polarity_policy = coordinate_polarity_policy(final_model)
    if args.coordinate_polarity_control and polarity_policy.get("run_coordinate_polarity_control", True) and not args.dry_run:
        polarity = root / "runs" / ds.name / "polarity_alt"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                polarity,
                args,
                n_basis=20 if args.fast else args.polarity_n_basis,
                n_points=5 if args.fast else args.polarity_n_points,
                stage="polarity",
                pinned=best_nuisance(final) or nui,
                coordinate_polarity_alt=True,
                make_maps=False,
            ),
            logs,
            f"{ds.name}_polarity_alt",
            dry_run=args.dry_run,
        )
        stats_path = root / "runs" / ds.name / "coordinate_polarity_diagnostic.json"
        run_logged(
            [
                python,
                str(engine_script("coordinate_polarity_diagnostic.py")),
                "--best",
                str(final / "native_dyn_calc.csv"),
                "--other",
                str(polarity / "native_dyn_calc.csv"),
                "--out",
                str(stats_path),
            ],
            logs,
            f"{ds.name}_coordinate_polarity_diagnostic",
            dry_run=args.dry_run,
        )
        coordinate_polarity_diagnostic = str(stats_path)
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
    return {
        "primary_run": str(final),
        "no_h_run": str(no_h_run) if no_h_run else "",
        "model": final_model,
        "coordinate_polarity_diagnostic": coordinate_polarity_diagnostic,
        "forensics": forensics,
        "scale_polish": scale_polish,
        "resolution_limits": {ds.name: resolution_limit},
    }


def run_multi(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run one structure against multiple datasets and combine endpoints."""

    root = Path(args.out_dir)
    if args.auto_decide:
        def decide_dataset(ds0: DatasetSpec) -> DatasetSpec:
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
            return ds

        if len(prep.datasets) > 1 and not args.dry_run:
            decided_by_name: dict[str, DatasetSpec] = {}
            max_workers = min(len(prep.datasets), transform_scout_job_count(args))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="grond-dataset-transform") as pool:
                futures = {pool.submit(decide_dataset, ds0): ds0.name for ds0 in prep.datasets}
                for future in as_completed(futures):
                    decided = future.result()
                    decided_by_name[decided.name] = decided
            decided = [decided_by_name[ds0.name] for ds0 in prep.datasets]
        else:
            decided = [decide_dataset(ds0) for ds0 in prep.datasets]
        prep = PrepConfig(
            structure=prep.structure,
            datasets=decided,
            baseline_hkl=prep.baseline_hkl,
            notes=prep.notes,
            default_hkl_transform=prep.default_hkl_transform,
            geometry_percentile=prep.geometry_percentile,
            metadata=prep.metadata,
        )
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
        active_run = run_dir
        active_nui = best_nuisance(run_dir)
        if not args.fast:
            thick = root / "runs" / ds.name / "multi_thickness_polish"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    prep.structure,
                    dataset_baseline_hkl(prep, ds),
                    thick,
                    args,
                    n_basis=args.screen_n_basis,
                    n_points=args.screen_n_points,
                    stage="multi_thickness_polish",
                    pinned=thickness_polish_nuisance(active_nui),
                    make_maps=False,
                    endpoint=False,
                ),
                logs,
                f"{ds.name}_multi_thickness_polish",
                dry_run=args.dry_run,
            )
            active_run = thick
            active_nui = best_nuisance(thick) or active_nui
            continuous_seed = continuous_thickness_seed_nuisance(args, active_nui)
            if continuous_thickness_enabled(args) and continuous_seed:
                continuous = root / "runs" / ds.name / "multi_thickness_continuous"
                run_logged(
                    continuous_thickness_cmd(
                        python,
                        ds,
                        prep.structure,
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
                active_run = continuous
                active_nui = best_nuisance(continuous) or active_nui or continuous_seed
            feature_records: list[dict] = []
            for family in ["row_resolution", "absorption", "sg_jitter", "sigma_error", "dose"]:
                active_nui, active_run, record = run_feature_gate(
                    prep,
                    ds,
                    args,
                    python,
                    logs,
                    root,
                    family=family,
                    active_model=prep.structure,
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
    datasets_csv = root / "prep_datasets_with_nuisance.csv"
    write_datasets_csv(datasets_csv, prep, apply_dataset_policy_to_nuisance(nuisance, structure_dataset_policy))
    final_model = prep.structure
    if args.mode in {"regularize", "auto"} and not args.fast:
        shared = root / "runs" / "shared_structure_update"
        cmd = [
            python,
            str(engine_script("native_multi_structure_update.py")),
            "--datasets-csv",
            str(datasets_csv),
            "--res",
            prep.structure,
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
            "--workers",
            str(args.workers),
            "--chunksize",
            str(args.chunksize),
        ]
        run_logged(cmd, logs, "shared_structure_update", dry_run=args.dry_run)
        accepted = shared / "accepted_model.res"
        if accepted.exists():
            final_model = str(accepted)
    final_runs: dict[str, Path] = {}
    forensic_runs: dict[str, dict] = {}
    scale_polish_runs: dict[str, dict] = {}
    resolution_limits: dict[str, dict] = {}
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
    if args.dry_run or final_dataset_policy.get("status") in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
        combined_input_runs = final_runs
        publication_runs = final_runs
    else:
        weighted_runs = write_policy_weighted_runs(
            final_runs,
            final_dataset_policy,
            root / "dataset_policy" / "final" / "weighted_runs",
            include_endpoint_only=True,
        )
        combined_input_runs = weighted_runs or final_runs
        publication_runs = policy_active_runs(weighted_runs or final_runs, final_dataset_policy)
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
    return {
        "combined": str(combined),
        "model": final_model,
        "runs": {k: str(v) for k, v in final_runs.items()},
        "combined_input_runs": {k: str(v) for k, v in combined_input_runs.items()},
        "publication_runs": {k: str(v) for k, v in publication_runs.items()},
        "forensics": forensic_runs,
        "scale_polish": scale_polish_runs,
        "resolution_limits": resolution_limits,
        "merged_resolution_limit": merged_resolution_limit,
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


def cmd_run(args: argparse.Namespace) -> int:
    """CLI entrypoint for the configurable `GROND run` workflow."""

    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logs = Path(args.out_dir) / "logs"
    python = choose_python(args.python)
    prep = prep_from_args(args)
    profile = apply_speed_profile(args, prep.structure)
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    original_structure = prep.structure
    prep.structure, structure_prep = prepare_structure_for_legacy(prep.structure, Path(args.out_dir))
    raise_for_missing_external_tools(args, structure=prep.structure, multi_dataset=len(prep.datasets) > 1)
    configure_progress(logs, resume=bool(args.resume), total=estimate_run_steps(prep, args))
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
        "matrix_convention": args.matrix_convention,
        "structure_update": {
            "requested": args.structure_update,
            "resolved": resolve_structure_update_strategy(args, multi_dataset=len(prep.datasets) > 1),
            "shelxl": args.shelxl,
            "shelxl_feedback_cycles": args.shelxl_feedback_cycles,
            "shelxl_feedback_shelxl_iterations": args.shelxl_feedback_shelxl_iterations,
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
        },
    }
    (Path(args.out_dir) / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = run_single(prep, args, python, logs) if len(prep.datasets) == 1 else run_multi(prep, args, python, logs)
    result["publication_endpoint_chooser"] = run_publication_endpoint_chooser(prep, result, args)
    result["publication_tables"] = run_publication_report(python, prep, result, args, logs)
    (Path(args.out_dir) / "grond_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_dir": str(Path(args.out_dir).resolve()), **result}, indent=2))
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Run the ordinary one-command GROND auto workflow."""

    args.mode = "auto"
    return cmd_run(args)


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
        engine_script("native_structure_update.py"),
        engine_script("native_multi_structure_update.py"),
        engine_script("combine_native_nuisance_runs.py"),
    ]
    cmd = [python, "-m", "py_compile", *map(str, scripts)]
    subprocess.run(cmd, check=True)
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


def cmd_doctor(args: argparse.Namespace) -> int:
    """CLI entrypoint that reports dependency and Bloch-kernel readiness."""

    python = choose_python(args.python)
    payload = probe_accelerators(python)
    payload["tool_config"] = {
        "path": getattr(args, "tool_config_path", ""),
        "configured_tools": getattr(args, "configured_tools", {}),
        "external_tools": external_tool_statuses(args),
    }
    if args.probe_bloch:
        payload["bloch_ops"] = probe_bloch_ops(python)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """CLI entrypoint that precompiles package and vendored Python bytecode."""

    root = Path(__file__).resolve().parents[1]
    ok = compileall.compile_dir(str(root), quiet=1, force=bool(args.force), workers=1)
    if not ok:
        return 1
    python = choose_python(args.python)
    cmd = [python, "-m", "py_compile", *map(str, sorted(LEGACY_ROOT.rglob("*.py")))]
    subprocess.run(cmd, check=True)
    print(
        json.dumps(
            {
                "package_root": str(root),
                "engine_root": str(LEGACY_ROOT),
                "python": python,
                "status": "compiled",
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
        first(args.lambda_grid, "0.95") if args.fast else args.lambda_grid,
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


def add_forensic_args(ap: argparse.ArgumentParser) -> None:
    """Attach row-space forensic and observation-polish options to a parser."""

    ap.add_argument("--forensic-gate", dest="forensic_gate", action="store_true", default=True, help="Run post-final row-space forensic audits.")
    ap.add_argument("--no-forensic-gate", dest="forensic_gate", action="store_false")
    ap.add_argument("--forensic-depth", choices=["quick", "standard", "full"], default="standard", help="quick runs residual/pre-merge audits; standard adds dose and two main observation-correction holdouts; full adds edge-frame and sigma audits.")
    ap.add_argument("--forensic-bins", type=int, default=10)
    ap.add_argument("--forensic-top-rows", type=int, default=200)
    ap.add_argument("--forensic-top-groups", type=int, default=200)
    ap.add_argument("--observation-polish", choices=["auto", "on", "off"], default="auto", help="Promote held-out observation corrections into corrected-HKL endpoints when both main gates pass.")
    ap.add_argument("--observation-polish-combine", choices=["ensemble", "best_all", "best_free"], default="ensemble")
    ap.add_argument("--observation-polish-lambda-grid", default="", help="Lambda grid for promoted observation-polish endpoints. Empty reuses --lambda-grid when available.")
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


def add_common_run_args(ap: argparse.ArgumentParser) -> None:
    """Attach the shared run/auto options to an argparse subcommand."""

    ap.add_argument("--prep", default="")
    ap.add_argument("--hkl", default="")
    ap.add_argument("--xds-dir", default="")
    ap.add_argument("--dials-refl", default="", help="DIALS integrated.refl/scaled.refl file. Auto-converted into a GROND row directory under OUT/prepared.")
    ap.add_argument("--dials-expt", default="", help="DIALS integrated.expt/scaled.expt file matching --dials-refl.")
    ap.add_argument("--dials-python", default="", help="Python executable that can import dials and dxtbx. Leave blank when the active Python already can.")
    ap.add_argument("--dials-intensity-mode", choices=["sum_complete", "sum_raw", "scale"], default="sum_complete", help="DIALS intensity mapping for the GROND row contract.")
    ap.add_argument("--dials-sigma-mode", choices=["auto", "raw", "detector_gain"], default="auto", help="DIALS sigma bridge policy. auto keeps raw DIALS sigmas unless rows look grossly overconfident; detector_gain forces sqrt(gain) scaling.")
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
    ap.add_argument("--full-fast-n-basis", type=int, default=20, help="Pinned basis size used by --full-fast final and screening stages. Use 10 for an even cheaper stress test.")
    ap.add_argument("--full-fast-n-points", type=int, default=10, help="Quadrature points used by --full-fast.")
    ap.add_argument("--full-fast-structure-rows", type=int, default=0, help="Maximum objective rows for lightweight --full-fast structure update; 0 chooses an atom-count-aware budget.")
    ap.add_argument("--full-fast-structure-max-nfev", type=int, default=0, help="Maximum function evaluations per lightweight --full-fast structure block; 0 chooses an atom-count-aware budget.")
    ap.add_argument("--auto-decide", dest="auto_decide", action="store_true", default=True, help="Run preflight, transform/polarity scouts, and adaptive thickness decisions before final work.")
    ap.add_argument("--no-auto-decide", dest="auto_decide", action="store_false")
    ap.add_argument("--native-only", action="store_true", help="Do not write corrected-HKL endpoints.")
    ap.add_argument("--python", default="")
    add_tool_config_args(ap)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reuse completed stages recorded in OUT/grond_progress.json.")
    ap.add_argument("--workers", type=int, default=0, help="Worker processes. Default 0 uses --max-core-fraction of detected CPU cores.")
    ap.add_argument("--max-core-fraction", type=float, default=0.85, help="Fraction of detected CPU cores to use when --workers=0.")
    ap.add_argument("--chunksize", type=int, default=0, help="Multiprocessing chunksize. Default 0 chooses a small value for load balancing.")
    ap.add_argument("--prediction-cache-dir", default="", help="Optional root for prediction caches. Default is OUT/cache.")
    ap.add_argument("--prediction-cache-scope", choices=["shared", "stage"], default="shared", help="Shared reuses safe hash-keyed predictions across adaptive stages; stage isolates every run.")
    ap.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none", help="Use none for fastest local SSD/NVMe caches; compressed saves disk on slow or limited storage.")
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda"], default="scipy", help="Bloch prediction backend. SciPy eigensolver is the production reference; scipy_bethe uses an opt-in promoted Bethe/Schur strong-beam reduction; SciPy expm/expm_multiply and JAX backends are experimental parity/speed probes.")
    ap.add_argument("--jax-batch-size", type=int, default=0, help="JAX matrix batch size for accelerator backends. 0 chooses a conservative size from matrix dimension and device count.")
    ap.add_argument("--matrix-convention", choices=["j_minus_i", "i_minus_j"], default="j_minus_i", help="Off-diagonal U_g indexing convention. Keep j_minus_i for production; i_minus_j is diagnostic.")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--scout-max-rows", type=int, default=1000, help="Representative row limit for automatic transform/polarity scouts.")
    ap.add_argument("--scout-n-basis", type=int, default=8, help="Basis size for cheap decision scouts.")
    ap.add_argument("--scout-n-points", type=int, default=3, help="Quadrature points for cheap decision scouts.")
    ap.add_argument("--transform-rerank-top", type=int, default=3, help="Number of coarse transform-scout winners to rerun at better N/Q; 0 disables the second stage.")
    ap.add_argument("--transform-rerank-n-basis", type=int, default=20, help="Basis size for second-stage transform reranking.")
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
    ap.add_argument("--polarity-accept-delta", type=float, default=0.015, help="Adaptive score margin required before switching to the alternate coordinate-polarity.")
    ap.add_argument("--stop-on-bad-preflight", action="store_true", default=False)
    ap.add_argument("--stop-on-bad-fit", action="store_true", default=False)
    ap.add_argument("--stop-if-wR2-above", type=float, default=0.75, help="Bad-data stop threshold for native all-row dynamic wR2 when --stop-on-bad-fit is enabled.")
    ap.add_argument("--min-bloch-wR2-gain", type=float, default=-0.01, help="Minimum wR2 gain over Born before structure update is considered reliable.")
    ap.add_argument("--feature-accept-delta", type=float, default=0.002, help="Minimum adaptive-score improvement before keeping a staged row-resolution/absorption/jitter/error/dose feature family.")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=99.0)
    ap.add_argument("--target-mode", choices=["iobs", "recorded_proxy"], default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--make-maps", dest="make_maps", action="store_true", default=True)
    ap.add_argument("--no-maps", dest="make_maps", action="store_false")
    ap.add_argument("--h-omit", choices=["off", "on", "only", "both"], default="off")
    ap.add_argument("--polarity-control", dest="coordinate_polarity_control", action="store_true", default=True, help="Run the alternate coordinate-polarity diagnostic when model symmetry makes that check meaningful.")
    ap.add_argument("--no-polarity-control", dest="coordinate_polarity_control", action="store_false")
    ap.add_argument("--polarity-n-basis", type=int, default=20, help="Basis size for default polarity-diagnostic runs; keep cheap unless the call is ambiguous.")
    ap.add_argument("--polarity-n-points", type=int, default=10, help="Quadrature points for default polarity-diagnostic runs.")
    ap.add_argument("--production-scales", action="store_true", default=True)
    ap.add_argument("--validation-scales", dest="production_scales", action="store_false")
    ap.add_argument("--screen-n-basis", type=int, default=20, help="Basis size for initial nuisance/geometry screening.")
    ap.add_argument("--screen-n-points", type=int, default=10, help="Quadrature points for initial nuisance/geometry screening.")
    ap.add_argument("--n-basis", type=int, default=0, help="Basis size for pinned final polish runs. Default 0 scans Sg support and uses max support plus margin.")
    ap.add_argument("--n-points", type=int, default=20, help="Quadrature points for pinned final polish runs; keep <=20 for production.")
    ap.add_argument("--final-basis-scan-limit", type=int, default=500, help="Upper limit used by the cheap support-only scan when --n-basis=0.")
    ap.add_argument("--final-basis-margin", type=int, default=10, help="Extra beams above the observed maximum support for final polish.")
    ap.add_argument("--final-basis-min", type=int, default=20)
    ap.add_argument("--final-basis-max", type=int, default=500)
    ap.add_argument("--final-basis-fallback", type=int, default=128, help="Dry-run fallback when --n-basis=0.")
    ap.add_argument("--structure-n-basis", type=int, default=20)
    ap.add_argument("--structure-n-points", type=int, default=6)
    ap.add_argument("--structure-objective-rows", type=int, default=400)
    ap.add_argument("--structure-max-nfev", type=int, default=5)
    ap.add_argument("--structure-early-stop-window", type=int, default=64, help="Stop a structure-update block after this many residual calls without a meaningful data-RMS gain; 0 disables.")
    ap.add_argument("--structure-early-stop-min-evals", type=int, default=0, help="Minimum residual calls before a plateau stop can trigger; 0 uses the engine's parameter-count-aware floor.")
    ap.add_argument("--structure-early-stop-min-improvement", type=float, default=2e-5, help="Minimum RMS improvement counted as progress for plateau stopping.")
    ap.add_argument("--structure-early-stop-metric", choices=["total_rms", "data_rms"], default="data_rms", help="Trace column used by the structure-update plateau stop.")
    ap.add_argument("--structure-update", choices=["auto", "native_gradient", "shelxl_feedback", "off"], default="auto", help="Structure-update engine. Auto uses the high-N/Q SHELXL feedback loop for single-dataset full runs when SHELXL is available; otherwise it uses native finite-difference gradients.")
    ap.add_argument("--structure-blocks", default="adp_ueq")
    ap.add_argument("--structure-atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--restraints", choices=["none", "start_geometry", "cctbx_rigu"], default="cctbx_rigu")
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--topup-d-min", type=float, default=0.8, help="Nominal Bloch support top-up dmin. Auto never uses a shallower support limit than the selected data dmin.")
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule used inside Bloch row predictions.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures; mean is the legacy normalized diagnostic.")
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--selection-metric", choices=["work_wR2", "free_wR2", "all_wR2", "all_R1"], default="all_wR2")
    ap.add_argument("--lambda-grid", default="1.0")
    ap.add_argument(
        "--scale-smoothness",
        type=float,
        default=0.0,
        help="Experimental L2 frame-scale smoothness coefficient. Default 0 keeps per-group scales independent; GROND's default parameter reduction is the post-hoc scale-polish gate.",
    )
    ap.add_argument("--endpoint-b-grid", default="0.0", help="Corrected-HKL endpoint resolution-scale diagnostic grid. Nonzero values are opt-in only; GROND keeps B=0 as the final promotable endpoint whenever it is present.")
    ap.add_argument("--endpoint-exti-grid", default="off", help="Endpoint EXTI diagnostic grid. Final promoted outputs use off; explicit 'on' runs are comparison-only.")
    ap.add_argument("--shelxl-iterations", type=int, default=5, help="SHELXL polish passes per corrected-HKL endpoint in full mode. Each pass promotes SHELXL's recommended WGHT into the active input before the next cycle.")
    ap.add_argument("--shelxl", default=default_shelxl_executable(), help="SHELXL executable used for corrected-HKL endpoints and SHELXL feedback loops.")
    ap.add_argument("--shelxl-feedback-cycles", type=int, default=5, help="Number of high-N/Q Bloch -> one-SHELXL-LS feedback cycles used by the automatic SHELXL structure-update strategy.")
    ap.add_argument("--shelxl-feedback-shelxl-iterations", type=int, default=1, help="SHELXL passes per feedback cycle. Default 1 keeps each feedback update to one least-squares model update.")
    ap.add_argument("--endpoint-feedback", choices=["auto", "on", "off"], default="auto", help="Optionally feed the polished B=0 no-EXTI endpoint model into one more Bloch endpoint cycle. Full auto uses auto; --fast skips it.")
    ap.add_argument("--publication-endpoint-chooser", action=argparse.BooleanOptionalAction, default=True, help="Write the publication endpoint chooser that separates physical and diagnostic corrected-HKL endpoints.")
    ap.add_argument("--publication-endpoint-min-hkl-safe-factor", type=float, default=0.25, help="Minimum fixed-width HKL safety factor for a publication endpoint.")
    ap.add_argument("--publication-endpoint-max-modulation-p95", type=float, default=20.0, help="Maximum 95th-percentile endpoint intensity modulation for a publication endpoint.")
    ap.add_argument("--scale-polish", choices=["auto", "on", "off"], default="auto", help="Run a post-hoc scale-window compression gate after the final Bloch row table.")
    ap.add_argument("--scale-polish-window-deg", type=float, default=5.0, help="Maximum physical frame-window width tested by the scale-polish gate.")
    ap.add_argument("--scale-polish-frame-models", default="adaptive,fixed", help="Frame-scale compression models tested by the scale-polish gate.")
    ap.add_argument("--scale-polish-adaptive-counts", default="auto", help="Adaptive neighboring-frame segment counts for scale polish; percentages such as 50%% are accepted.")
    ap.add_argument("--scale-polish-detector-bins", default="exact", help="Detector flat-field grids tested by the scale-polish gate. Default exact preserves the fitted detector grid while compressing frame scales.")
    ap.add_argument("--scale-polish-objective", choices=["row_mr", "native", "sigma"], default="native")
    ap.add_argument("--scale-polish-tolerance", type=float, default=0.01)
    ap.add_argument("--scale-polish-selection-metrics", default="Rall,wRall,native_R1_all,native_wR2_all")
    ap.add_argument("--scale-polish-scan-offsets", action=argparse.BooleanOptionalAction, default=False, help="Test all block-boundary offsets in the scale-polish gate.")
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
    ap.add_argument("--row-resolution-b-grid", default="0.0,14,18", help="Raw-row dynamical resolution-envelope grid. Positive values are the dynamic-gradient counterpart of negative endpoint-B diagnostics.")
    ap.add_argument("--thickness-grid", default="1000,1200,1400")
    ap.add_argument("--thickness-model-grid", default="uniform_gradient")
    ap.add_argument("--thickness-grad-grid", default="0.4,1.0")
    ap.add_argument("--continuous-thickness-polish", choices=["auto", "on", "off"], default="auto", help="After the coarse thickness grid, polish continuous thickness/gradient by reusing cached Bloch eigensystems. Auto is on for GROND auto except --fast.")
    ap.add_argument("--continuous-thickness-metric", choices=["all_wR2", "free_wR2", "all_R1", "all_R1gt"], default="all_wR2", help="Objective used by the continuous thickness polish.")
    ap.add_argument("--continuous-thickness-strong-rows", type=int, default=0, help="Rows used by continuous thickness polish; 0 uses all selected rows. Strong-row-only scouts are diagnostic and are not the default.")
    ap.add_argument("--continuous-thickness-max-rows", type=int, default=0, help="Hard row limit for continuous thickness polish; 0 uses all rows selected by the data filters.")
    ap.add_argument("--continuous-thickness-bracket-frac", type=float, default=0.35, help="Local fractional thickness bracket around the coarse-grid winner.")
    ap.add_argument("--continuous-thickness-grid-step", type=float, default=100.0, help="Guard-grid thickness step in Angstrom.")
    ap.add_argument("--continuous-thickness-grad-step", type=float, default=0.2, help="Guard-grid thickness-gradient step.")
    ap.add_argument("--continuous-thickness-powell-maxiter", type=int, default=18, help="Maximum Powell iterations for the final phase-only thickness/gradient polish.")
    ap.add_argument("--continuous-thickness-min-gain", type=float, default=5e-5, help="Minimum objective gain required to promote a non-grid continuous thickness.")
    ap.add_argument("--continuous-thickness-max-free-wr2-loss", type=float, default=0.002, help="Maximum tolerated held-out/free wR2 loss for promoted continuous thickness.")
    ap.add_argument("--continuous-thickness-max-all-r1-loss", type=float, default=0.001, help="Maximum tolerated all-row R1 loss for promoted continuous thickness.")
    ap.add_argument("--frame-offset-grid", default="0.0,0.2")
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
    add_forensic_args(ap)


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

    p_dials = sub.add_parser("import-dials", help="Convert DIALS integrated.refl/integrated.expt into a GROND row directory.")
    p_dials.add_argument("--refl", required=True, help="DIALS integrated.refl or scaled.refl file.")
    p_dials.add_argument("--expt", required=True, help="DIALS integrated.expt or scaled.expt file.")
    p_dials.add_argument("--out-dir", required=True)
    p_dials.add_argument("--dials-python", default="", help="Python executable that can import dials and dxtbx.")
    p_dials.add_argument("--intensity-mode", choices=["sum_complete", "sum_raw", "scale"], default="sum_complete")
    p_dials.add_argument("--sigma-mode", choices=["auto", "raw", "detector_gain"], default="auto")
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
    add_common_run_args(p_auto)
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
    p_raw.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda"], default="scipy")
    p_raw.add_argument("--jax-batch-size", type=int, default=0)
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
    p_raw.add_argument("--frame-offset-grid", default="0.0,0.2")
    p_raw.add_argument("--group-width-grid", default="auto")
    p_raw.add_argument("--detector-scale-bins", type=int, default=8)
    p_raw.add_argument("--thickness-points", type=int, default=5)
    p_raw.add_argument("--lambda-grid", default="1.0")
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

    p_val = sub.add_parser("validate", help="Compile the vendored engine scripts with the selected Python.")
    p_val.add_argument("--python", default="")
    add_tool_config_args(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_doc = sub.add_parser("doctor", help="Report Python packages, accelerator visibility, and the active compute core.")
    p_doc.add_argument("--python", default="")
    add_tool_config_args(p_doc)
    p_doc.add_argument("--probe-bloch", action="store_true", help="Also test complex eig/eigh support needed by the Bloch wave kernel.")
    p_doc.set_defaults(func=cmd_doctor)

    p_compile = sub.add_parser("compile", help="Precompile GROND and vendored engine modules to bytecode.")
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
