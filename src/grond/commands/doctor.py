"""Check that GROND can run: the Python packages, the Bloch-wave kernel and the external programs."""

from __future__ import annotations

import argparse
import json

from grond import __version__
from grond.accelerators import probe_accelerators, probe_bloch_ops
from grond.cli_support.tools import external_tool_statuses

TOOLS = (
    ("shelxl", "needed by auto, refine and manual"),
    ("shelx2map", "optional: SHELXL Fo-Fc maps from finish"),
)


def doctor_report(args: argparse.Namespace) -> dict:
    """Probe this Python and the external programs; `ready` is False when GROND cannot calculate."""

    env = probe_accelerators()
    bloch = probe_bloch_ops()
    tools = external_tool_statuses(args)
    packages = env.get("packages") if isinstance(env.get("packages"), dict) else {}
    kernel = bloch.get("scipy_cpu") if isinstance(bloch.get("scipy_cpu"), dict) else {}
    failures = [f"Python package {name} is missing: {(packages.get(name) or {}).get('error', 'not found')}"
                for name in ("numpy", "scipy", "gemmi") if not (packages.get(name) or {}).get("available")]
    failures += [f"SciPy {op} failed in the Bloch-wave kernel: {(kernel.get(op) or {}).get('error', 'no result')}"
                 for op in ("eig", "eigh") if not (kernel.get(op) or {}).get("ok")]
    failures += [probe["probe_error"] for probe in (env, bloch) if probe.get("probe_error")]
    return {"grond": __version__, "python": env.get("python"), "python_version": env.get("python_version"),
            "platform": env.get("platform"), "packages": packages, "bloch_kernel": kernel,
            "tools": {name: tools.get(name, {}) for name, _ in TOOLS}, "failures": failures, "ready": not failures}


def cmd_doctor(args: argparse.Namespace) -> int:
    """Print the report; exit 1 when GROND cannot calculate on this Python."""

    report = doctor_report(args)
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    rows = [("GROND", report["grond"]), ("Python", f"{report['python_version']} ({report['python']})")]
    for name, status in report["packages"].items():
        rows.append((name, status.get("version", "") if status.get("available") else "MISSING"))
    kernel_ok = all((report["bloch_kernel"].get(op) or {}).get("ok") for op in ("eig", "eigh"))
    rows.append(("Bloch kernel", "ok" if kernel_ok else "FAILED"))
    for name, role in TOOLS:
        status = report["tools"][name]
        rows.append((name, status.get("resolved") if status.get("available") else f"not found ({role})"))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    for failure in report["failures"]:
        print(f"ERROR: {failure}")
    print("Ready." if report["ready"] else "Not ready.")
    return 0 if report["ready"] else 1
