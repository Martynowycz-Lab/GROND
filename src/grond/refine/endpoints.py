from __future__ import annotations

import os
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from grond.io.tool_config import configured_tools, resolve_executable, tool_statuses


@dataclass(frozen=True)
class EndpointSpec:
    """One model-update backend that can sit after a GROND corrected-HKL export."""

    name: str
    status: str
    executable_key: str
    reflection_inputs: tuple[str, ...]
    model_inputs: tuple[str, ...]
    model_outputs: tuple[str, ...]
    supports_single_cycle: bool
    default_cycles: int
    description: str
    notes: str


ENDPOINTS: dict[str, EndpointSpec] = {
    "shelxl": EndpointSpec(
        name="shelxl",
        status="implemented_default",
        executable_key="shelxl",
        reflection_inputs=("HKLF4 .hkl",),
        model_inputs=("SHELX .ins/.res",),
        model_outputs=("SHELX .res", "CIF when requested by SHELXL instructions"),
        supports_single_cycle=True,
        default_cycles=1,
        description="Production corrected-HKL least-squares endpoint used by GROND auto when a SHELX model is supplied.",
        notes="This is the tested endpoint for the full automatic feedback loop.",
    ),
    "olex2_app": EndpointSpec(
        name="olex2_app",
        status="supported_external_app",
        executable_key="olex2_app_dir",
        reflection_inputs=("HKLF4 .hkl",),
        model_inputs=("SHELX .ins/.res",),
        model_outputs=("SHELX .res", "CIF/FCF when requested by Olex2 instructions"),
        supports_single_cycle=True,
        default_cycles=1,
        description="Optional Olex2.app console endpoint for users with a local Olex2 installation.",
        notes="Configure tools.olex2_app_dir or GROND_OLEX2_APP_DIR. GROND uses the app console directory, not a bundled binary.",
    ),
    "bloch_only": EndpointSpec(
        name="bloch_only",
        status="implemented_no_refiner",
        executable_key="python",
        reflection_inputs=("signed row table", "XDS or DIALS row-preserving integration"),
        model_inputs=("RES/CIF",),
        model_outputs=("row predictions", "nuisance decisions", "corrected-HKL diagnostics", "polarity diagnostics"),
        supports_single_cycle=False,
        default_cycles=0,
        description="Run the Bloch simulation, nuisance gates, thickness search, residual reports, and polarity diagnostics without updating the model through an external refiner.",
        notes="This is the fallback when no model-update endpoint is configured. It is still useful for R-factor improvement diagnostics and polarity decisions.",
    ),
    "native_gradient": EndpointSpec(
        name="native_gradient",
        status="implemented_diagnostic",
        executable_key="python",
        reflection_inputs=("signed row table",),
        model_inputs=("RES/CIF",),
        model_outputs=("RES-like updated model", "diagnostic traces"),
        supports_single_cycle=False,
        default_cycles=1,
        description="Slow finite-difference row-space update used as a diagnostic reference for small models.",
        notes="Useful for controls, not the default production endpoint because it is much slower than corrected-HKL feedback.",
    ),
    "custom": EndpointSpec(
        name="custom",
        status="adapter_template",
        executable_key="custom_refiner",
        reflection_inputs=("user-declared corrected reflection file",),
        model_inputs=("user-declared model file",),
        model_outputs=("user-declared refined model file",),
        supports_single_cycle=True,
        default_cycles=1,
        description="Command-template adapter for labs that want to drive their own one-cycle model-update tool.",
        notes="Configure endpoints.custom.command and endpoints.custom.output_model in grond.config.json.",
    ),
}


def configured_endpoint_blocks(config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return endpoint adapter blocks from the GROND config file."""

    config = config or {}
    endpoints = config.get("endpoints") or {}
    if not isinstance(endpoints, dict):
        return {}
    return {str(key).strip().lower().replace("-", "_"): dict(value) for key, value in endpoints.items() if isinstance(value, dict)}


def _normalize_olex2_app_dir(path_text: str) -> Path:
    """Return the Olex2 console directory from either the app root or MacOS dir."""

    path = Path(os.path.expandvars(os.path.expanduser(path_text))).resolve()
    if path.name.lower().endswith(".app"):
        return path / "Contents" / "MacOS"
    if path.name == "Contents" and (path.parent.name.lower().endswith(".app")):
        return path / "MacOS"
    return path


def _candidate_olex2_dirs(config: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return configured, environment, endpoint-block, and common Olex2 app locations."""

    tools = configured_tools(config)
    blocks = configured_endpoint_blocks(config)
    candidates: list[tuple[str, str]] = []
    for key, source in (
        (tools.get("olex2_app_dir", ""), "config:tools.olex2_app_dir"),
        (str(blocks.get("olex2_app", {}).get("app_dir", "")).strip(), "config:endpoints.olex2_app.app_dir"),
        (os.environ.get("GROND_OLEX2_APP_DIR", ""), "env:GROND_OLEX2_APP_DIR"),
    ):
        if key:
            candidates.append((key, source))
    candidates.extend(
        [
            ("/Applications/olex2.app/Contents/MacOS", "default"),
            ("/Applications/Olex2.app/Contents/MacOS", "default"),
        ]
    )
    return candidates


def resolve_olex2_app(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve an installed Olex2.app console path and runtime library setup.

    The macOS app ships its console executable and private libraries under
    `Contents/MacOS`.  Calling the executable directly is the most reproducible
    way to use it from GROND because GUI launch services and shell startup files
    often hide the runtime paths.
    """

    config = config or {}
    tools = configured_tools(config)
    console_status = resolve_executable(
        "olex2_console",
        configured=tools.get("olex2_console", ""),
        envvar="GROND_OLEX2_CONSOLE",
        aliases=(),
    )
    if console_status.get("available"):
        console = Path(str(console_status["resolved"]))
        app_dir = console.parent
        lib_dirs = [app_dir / "lib", app_dir / "cctbx" / "cctbx_build" / "lib"]
        return {
            "available": True,
            "source": console_status.get("source", ""),
            "app_dir": str(app_dir),
            "console": str(console),
            "dyld_library_path": os.pathsep.join(str(p) for p in lib_dirs if p.exists()),
            "method": "Gauss-Newton",
            "checked": [str(console)],
        }

    checked: list[str] = []
    for candidate, source in _candidate_olex2_dirs(config):
        app_dir = _normalize_olex2_app_dir(candidate)
        console = app_dir / "olex2c_exe"
        checked.append(str(app_dir))
        if console.exists():
            lib_dirs = [app_dir / "lib", app_dir / "cctbx" / "cctbx_build" / "lib"]
            return {
                "available": True,
                "source": source,
                "app_dir": str(app_dir),
                "console": str(console.resolve()),
                "dyld_library_path": os.pathsep.join(str(p.resolve()) for p in lib_dirs if p.exists()),
                "method": "Gauss-Newton",
                "checked": checked,
            }
    return {
        "available": False,
        "source": "",
        "app_dir": "",
        "console": "",
        "dyld_library_path": "",
        "method": "Gauss-Newton",
        "checked": checked,
        "error": "No Olex2 app console was found. Set tools.olex2_app_dir to /Applications/olex2.app/Contents/MacOS or set GROND_OLEX2_APP_DIR.",
    }


def olex2_command_script(*, model_name: str, cycles: int = 1, method: str = "Gauss-Newton") -> str:
    """Return an Olex2 console script for a one-cycle endpoint update."""

    safe_cycles = max(1, int(cycles))
    return "\n".join(
        [
            f"reap {model_name}",
            "spy.SetParam('snum.refinement.program','olex2.refine')",
            f"spy.SetParam('snum.refinement.method',{method!r})",
            f"spy.SetParam('snum.refinement.max_cycles',{safe_cycles!r})",
            "refine",
            "",
        ]
    )


def endpoint_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return endpoint registry entries plus executable availability."""

    statuses = tool_statuses(config)
    blocks = configured_endpoint_blocks(config)
    olex2 = resolve_olex2_app(config)
    rows = []
    for key, spec in ENDPOINTS.items():
        if key == "olex2_app":
            executable_status = {
                "name": "olex2_app",
                "available": bool(olex2.get("available")),
                "resolved": olex2.get("console", ""),
                "source": olex2.get("source", ""),
                "checked": olex2.get("checked", []),
                "error": olex2.get("error", ""),
            }
            block = blocks.get(key, {})
            command_template = ""
            output_model = str(block.get("output_model", "")).strip()
            extra = {"olex2": olex2}
        else:
            executable_status = statuses.get(spec.executable_key, {})
            block = blocks.get(key, {})
            command_template = str(block.get("command", "")).strip()
            output_model = str(block.get("output_model", "")).strip()
            extra = {}
        rows.append(
            {
                **asdict(spec),
                "available": bool(executable_status.get("available")) or bool(command_template) or key == "bloch_only",
                "executable": executable_status,
                "configured_command_template": bool(command_template),
                "configured_output_model": output_model,
                **extra,
            }
        )
    return {
        "endpoints": rows,
        "configured_endpoint_blocks": sorted(blocks.keys()),
    }


def render_custom_command(
    template: str,
    *,
    model: str | Path,
    reflections: str | Path,
    out_dir: str | Path,
    cycles: int = 1,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    """Render a configured custom endpoint command into an argv list.

    The template can use `{model}`, `{reflections}`, `{out_dir}`, `{cycles}`,
    and any key supplied in `extra`.  The result is split with shell-like
    quoting so a config file can contain either a simple executable or a command
    with flags.  GROND does not assume what the external tool is; it only checks
    that the command can be rendered and that the declared output model exists
    when a workflow chooses to run that adapter.
    """

    values: dict[str, Any] = {
        "model": str(model),
        "reflections": str(reflections),
        "out_dir": str(out_dir),
        "cycles": int(cycles),
    }
    values.update(extra or {})
    allowed = set(values)
    requested = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    unknown = requested - allowed
    if unknown:
        raise ValueError("unknown custom endpoint template field(s): " + ", ".join(sorted(unknown)))
    rendered = template.format(**values)
    return shlex.split(rendered)
