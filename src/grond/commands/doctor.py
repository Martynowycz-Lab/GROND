"""Diagnose installed features, numerical kernels and required external tools."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

from grond import __version__
from grond.accelerators import probe_accelerators, probe_bloch_ops
from grond.cli_support.tools import choose_python, external_tool_statuses
from grond.paths import LEGACY_ROOT, PACKAGE_ROOT, engine_script


def _doctor_readiness_failures(
    payload: Mapping[str, object],
    *,
    require_bloch: bool,
    workflow: str = "core",
) -> list[str]:
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

    required_external = {
        "core": (),
        "native-maps": (),
        "small-molecule": ("shelxl",),
        "endpoint-maps": ("shelxl", "shelx2map"),
    }.get(workflow, ())
    tool_config = payload.get("tool_config")
    external_tools = tool_config.get("external_tools") if isinstance(tool_config, Mapping) else None
    if required_external and not isinstance(external_tools, Mapping):
        failures.append(f"{workflow} workflow tool status is missing")
    elif isinstance(external_tools, Mapping):
        for name in required_external:
            status = external_tools.get(name)
            if not isinstance(status, Mapping) or not status.get("available"):
                detail = status.get("error") if isinstance(status, Mapping) else "status missing"
                failures.append(f"{workflow} workflow requires {name}: {detail}")

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


def _source_text(relative_path: str) -> str:
    """Read a feature's actual owner without importing optional numerical code."""

    try:
        return (PACKAGE_ROOT / relative_path).read_text(errors="replace")
    except OSError:
        return ""


def _installed_features() -> dict[str, bool]:
    """Check implementation files; compatibility reexports are not evidence.

    These are stale-install checks, not a substitute for the regression suite.
    Keep each check beside its actual source owner when moving code again.
    """

    health = _source_text("refine/shelxl_diagnostics.py")
    feedback = _source_text("legacy/engine/shelxl_feedback_loop.py")
    feedback_contract = _source_text("refine/feedback_contract.py")
    model_selection = _source_text("workflows/model_selection.py")
    shelx_input = _source_text("refine/shelx_input.py")
    absolute_structure = _source_text("legacy/engine/absolute_structure_diagnostic.py")
    transforms = _source_text("hkl_transform.py")
    transform_selection = _source_text("workflows/transform_selection.py")
    return {
        "shelxl_feedback_res_guard": (
            "def shelx_res_health" in health
            and "native_row_regularize import" in feedback
            and "shelx_res_health" in feedback
            and "summary-selected endpoint model hash" in feedback_contract
        ),
        "usable_feedback_model_selection": (
            "def shelxl_feedback_current_model" in model_selection and "def shelx_model_is_usable" in model_selection
        ),
        "final_feedback_row_recompute": (
            "final_row_recompute" in feedback and "--skip-endpoint-refinement" in feedback
        ),
        "corrected_endpoint_hklf4_contract": (
            "HKLF 4" in shelx_input and "def _unsupported_endpoint_reason" in shelx_input
        ),
        "absolute_structure_no_pandas": (
            bool(absolute_structure)
            and "import pandas" not in absolute_structure
            and "pd.read_csv" not in absolute_structure
        ),
        "proper_hand_parity_contract": (
            "def parse_hkl_transform_info" in transforms
            and "matched_born_work_only_hand_invariant_v1" in transform_selection
            and "metadata_candidate_elimination_used" in transform_selection
        ),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    """CLI entrypoint that reports dependency and Bloch-kernel readiness."""

    # An explicitly requested or configured interpreter is itself the subject
    # of this diagnostic. Do not silently substitute another healthy Python.
    python = str(getattr(args, "python", "") or "").strip() or choose_python("")
    payload = probe_accelerators(python)
    if not bool(getattr(args, "all_tools", False)):
        payload.pop("optional_packages", None)
        engine = payload.get("engine")
        if isinstance(engine, dict):
            engine.pop("optional_backends", None)
            engine["optional_backends_note"] = "Use --all-tools to report experimental JAX backends."
    package_init = PACKAGE_ROOT / "__init__.py"
    absolute_structure_script = engine_script("absolute_structure_diagnostic.py")
    absolute_structure_text = _source_text("legacy/engine/absolute_structure_diagnostic.py")
    required_features = _installed_features()
    payload["grond"] = {
        "version": __version__,
        "package": str(package_init.resolve()),
        "cli": str((PACKAGE_ROOT / "cli.py").resolve()),
        "engine_root": str(LEGACY_ROOT.resolve()),
        "absolute_structure_diagnostic": str(absolute_structure_script.resolve()),
        "absolute_structure_uses_pandas": "import pandas" in absolute_structure_text
        or "pd.read_csv" in absolute_structure_text,
        "required_features": required_features,
        "install_status": "current_required_features_present"
        if all(required_features.values())
        else "stale_or_incomplete_install",
    }
    all_external_tools = external_tool_statuses(args)
    reported_tool_names = (
        tuple(all_external_tools)
        if bool(getattr(args, "all_tools", False))
        else (
            "shelxl",
            "shelx2map",
        )
    )
    payload["requirements"] = {
        "core_python": ["numpy", "scipy", "gemmi"],
        "default_small_molecule_endpoint": ["shelxl"],
        "native_dynamical_maps": [],
        "optional_corrected_hkl_endpoint_maps": ["shelx2map"],
        "input_note": (
            "XDS, XDSCONV, SHELXT, DIALS, and Olex2 are not general GROND runtime requirements. "
            "DIALS Python is needed only while importing .refl/.expt files; existing XDS or GROND bridge directories are read directly."
        ),
    }
    payload["tool_config"] = {
        "path": getattr(args, "tool_config_path", ""),
        "configured_tools": getattr(args, "configured_tools", {}),
        "external_tools": {
            name: all_external_tools[name] for name in reported_tool_names if name in all_external_tools
        },
        "showing": "all implemented endpoint adapters"
        if bool(getattr(args, "all_tools", False))
        else "small-molecule tools",
    }
    payload["ready_scope"] = str(getattr(args, "workflow", "core") or "core")
    if args.probe_bloch:
        bloch_ops = probe_bloch_ops(python)
        if not bool(getattr(args, "all_tools", False)):
            bloch_ops.pop("jax", None)
        payload["bloch_ops"] = bloch_ops
    failures = _doctor_readiness_failures(
        payload,
        require_bloch=bool(args.probe_bloch),
        workflow=payload["ready_scope"],
    )
    payload["ready"] = not failures
    payload["failures"] = failures
    print(json.dumps(payload, indent=2))
    return 1 if bool(getattr(args, "strict", False)) and failures else 0
