"""Registry of the model-update endpoints (SHELXL and others) that can refine against GROND's corrected HKL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        model_outputs=("row predictions", "nuisance decisions", "corrected-HKL diagnostics", "hand diagnostics"),
        supports_single_cycle=False,
        default_cycles=0,
        description="Run the Bloch simulation, nuisance gates, thickness search, residual reports, and hand diagnostics without updating the model through an external refiner.",
        notes="This is the fallback when no model-update endpoint is configured. It is still useful for R-factor improvement diagnostics and absolute-structure decisions.",
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
    "phenix_refine": EndpointSpec(
        name="phenix_refine",
        status="macromolecular_adapter",
        executable_key="phenix_refine",
        reflection_inputs=("MTZ with IMEAN/SIGIMEAN/FreeR_flag",),
        model_inputs=("PDB/mmCIF",),
        model_outputs=("PDB/mmCIF", "maps/logs from the configured command"),
        supports_single_cycle=True,
        default_cycles=1,
        description="Configurable one-macrocycle macromolecular endpoint adapter for corrected-MTZ feedback.",
        notes="Use with fixed FreeR flags and an explicit command template; this is not wired into small-molecule GROND auto.",
    ),
    "refmac5": EndpointSpec(
        name="refmac5",
        status="macromolecular_adapter",
        executable_key="refmac5",
        reflection_inputs=("MTZ with IMEAN/SIGIMEAN/FreeR_flag",),
        model_inputs=("PDB/mmCIF plus any required dictionaries/restraints",),
        model_outputs=("PDB/mmCIF", "maps/logs from the configured command"),
        supports_single_cycle=True,
        default_cycles=1,
        description="Configurable one-macrocycle REFMAC endpoint adapter for corrected-MTZ feedback.",
        notes="Use a wrapper command when dictionaries, restraints, or electron-scattering settings need setup.",
    ),
    "servalcat": EndpointSpec(
        name="servalcat",
        status="macromolecular_adapter",
        executable_key="servalcat",
        reflection_inputs=("MTZ with IMEAN/SIGIMEAN/FreeR_flag",),
        model_inputs=("PDB/mmCIF plus any required dictionaries/restraints",),
        model_outputs=("PDB/mmCIF", "maps/logs from the configured command"),
        supports_single_cycle=True,
        default_cycles=1,
        description="Configurable one-macrocycle Servalcat endpoint adapter for corrected-MTZ feedback.",
        notes="Use a wrapper command to pin the intended scattering, solvent, restraint, and FreeR behavior.",
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
