"""Resolve Python and external tools without hiding missing dependencies."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from grond.io.structure_contract import is_shelx_model
from grond.io.tool_config import (
    configured_tools,
    find_tool_config,
    load_tool_config,
    resolve_executable,
    tool_env_updates,
    tool_statuses,
)


def has_runtime_deps(python: str) -> bool:
    """Return true when a Python executable can import GROND runtime deps."""

    code = "import gemmi, numpy, scipy"
    try:
        return (
            subprocess.run([python, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        )
    except OSError:
        return False


def choose_python(explicit: str = "") -> str:
    """Choose the Python interpreter used for the engine worker scripts."""

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
        help="Config file with the SHELXL and shelx2map paths (default: $GROND_CONFIG or grond.config.json).",
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
            # Workers execute from the engine directory, not from the
            # user's launch directory.  Freeze the preflight resolution now so
            # relative paths and PATH aliases keep the exact same meaning.
            args.shelxl = str(status["resolved"])
    args.tool_config_path = str(path) if path else ""
    args.tool_config = config
    args.configured_tools = tools


def external_tool_statuses(args: argparse.Namespace) -> dict[str, dict]:
    """Return the active external-tool status block for run configs and doctor."""

    statuses = tool_statuses(getattr(args, "tool_config", {}) or {})
    requested_python = str(getattr(args, "python", "") or "").strip()
    active_python = requested_python or sys.executable
    statuses["python"] = resolve_executable("python", configured=active_python)
    statuses["python"]["source"] = "argument/config" if requested_python else "current_interpreter"
    explicit_shelxl = str(getattr(args, "shelxl", "") or "").strip()
    if explicit_shelxl:
        statuses["shelxl"] = resolve_executable(
            "shelxl",
            configured=explicit_shelxl,
            aliases=("shelxl", "shelxl2019"),
        )
        statuses["shelxl"]["source"] = "argument:--shelxl"
        statuses["shelxl"]["requested"] = str(getattr(args, "shelxl_requested", explicit_shelxl) or explicit_shelxl)
    return statuses


def required_external_tool_errors(
    args: argparse.Namespace, *, structure: str = "", multi_dataset: bool = False
) -> list[str]:
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
        errors.append(
            "SHELXL is required for corrected-HKL endpoint polishing/statistics; set tools.shelxl in a GROND config, set SHELXL, or use --native-only."
        )
    if endpoint_expected and getattr(args, "make_maps", False) and not statuses.get("shelx2map", {}).get("available"):
        errors.append(
            "shelx2map is required for automatic map generation; set tools.shelx2map in a GROND config, set SHELX2MAP, or add --no-maps."
        )
    if getattr(args, "structure_update", "auto") == "shelxl_feedback" and not statuses.get("shelxl", {}).get(
        "available"
    ):
        errors.append(
            "The requested --structure-update shelxl_feedback needs SHELXL; set tools.shelxl or choose off."
        )
    return errors


def raise_for_missing_external_tools(
    args: argparse.Namespace, *, structure: str = "", multi_dataset: bool = False
) -> None:
    """Stop a run with a concise, actionable missing-tool message."""

    errors = required_external_tool_errors(args, structure=structure, multi_dataset=multi_dataset)
    if not errors:
        return
    hint = (
        "Create a template with `GROND init-config --out grond.config.json`, "
        "then rerun with `--config grond.config.json` or set GROND_CONFIG."
    )
    raise SystemExit("Missing required external tools:\n- " + "\n- ".join(errors) + "\n" + hint)
