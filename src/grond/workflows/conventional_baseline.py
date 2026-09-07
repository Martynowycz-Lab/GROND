"""Create and validate the conventional control used to judge endpoint changes."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.io.config import PrepConfig
from grond.macromolecule import sha256_file
from grond.paths import engine_script
from grond.workflows.decision import safe_float


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

    from grond.cli_support.execution import run_logged

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
    """Compare endpoint R1all with baseline without withholding a usable model."""

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
    finite_endpoint = math.isfinite(endpoint_r1)
    no_harm = bool(finite_endpoint and endpoint_r1 <= baseline_r1 + tolerance)
    delta = endpoint_r1 - baseline_r1 if math.isfinite(endpoint_r1) else float("nan")
    reasons = []
    warnings = []
    if not math.isfinite(endpoint_r1):
        reasons.append("corrected endpoint R1all is unavailable")
    elif not no_harm:
        warnings.append(
            f"corrected endpoint R1all {endpoint_r1:.6g} is worse than fresh conventional "
            f"baseline {baseline_r1:.6g} by {delta:.6g}; warning only, keep the endpoint available "
            "for map inspection and continued model refinement"
        )
    return {
        "status": "passed" if no_harm else ("warning_worse_than_baseline" if finite_endpoint else "failed"),
        "passed": finite_endpoint,
        "no_harm": no_harm,
        "warnings": warnings,
        "reasons": reasons,
        "baseline_source": baseline.get("baseline_source"),
        "baseline_R1all": baseline_r1,
        "endpoint_R1all": endpoint_r1,
        "delta_R1all": delta,
        "r1_tolerance": tolerance,
        "baseline_res": baseline.get("res", ""),
        "baseline_hkl": baseline.get("hkl", ""),
        "baseline_lst": baseline.get("lst", ""),
    }
