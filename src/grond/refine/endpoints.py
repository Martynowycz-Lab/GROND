from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from grond.io.tool_config import tool_statuses


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


def endpoint_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return endpoint registry entries plus executable availability."""

    statuses = tool_statuses(config)
    blocks = configured_endpoint_blocks(config)
    rows = []
    for key, spec in ENDPOINTS.items():
        executable_status = statuses.get(spec.executable_key, {})
        block = blocks.get(key, {})
        command_template = str(block.get("command", "")).strip()
        output_model = str(block.get("output_model", "")).strip()
        rows.append(
            {
                **asdict(spec),
                "available": bool(executable_status.get("available")) or bool(command_template) or key == "bloch_only",
                "executable": executable_status,
                "configured_command_template": bool(command_template),
                "configured_output_model": output_model,
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
