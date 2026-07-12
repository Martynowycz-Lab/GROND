from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from grond.forensics.resolution import classify_endpoint_candidate


def _float(value, fallback: float = float("nan")) -> float:
    """Convert a scalar to a finite float when possible."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _int(value, fallback: int = 0) -> int:
    """Convert a scalar to a non-negative integer when possible."""

    try:
        out = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(0, out)


def _read_json(path: Path) -> dict:
    """Read a JSON file and return an empty dict when it is absent or malformed."""

    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _percent(value: object) -> str:
    """Format a fractional residual as a percentage for reports."""

    value_f = _float(value)
    return "n/a" if not math.isfinite(value_f) else f"{100.0 * value_f:.2f}%"


def _metric_key(candidate: Mapping) -> tuple[float, float, float, str]:
    """Sort endpoint candidates by weighted all-row endpoint quality."""

    return (
        _float(candidate.get("wR2_all"), float("inf")),
        _float(candidate.get("R1_all"), float("inf")),
        _float(candidate.get("R1_gt"), float("inf")),
        str(candidate.get("id", "")),
    )


def _is_combined_candidate(candidate: Mapping) -> bool:
    """Return whether an endpoint came from a merged multi-dataset row table."""

    return str(candidate.get("source_kind", "")).startswith("combined_")


def _candidate_id(label: str, endpoint: Mapping, index: int) -> str:
    """Build a stable human-readable endpoint candidate identifier."""

    lam = _float(endpoint.get("lambda_q"), float("nan"))
    endpoint_b = _float(endpoint.get("endpoint_b"), 0.0)
    exti = "exti" if bool(endpoint.get("keep_exti", False)) else "noexti"
    parts = [str(label)]
    if math.isfinite(lam) and abs(lam - 1.0) > 1e-9:
        parts.append(f"qexp={lam:g}")
    parts.extend([f"B={endpoint_b:g}", exti, f"candidate={index}"])
    return ":".join(parts)


def _normalize_endpoint(label: str, endpoint: Mapping, *, run_dir: Path | None, source_kind: str, index: int, accepted_polish: bool | None = None) -> dict:
    """Normalize one endpoint dictionary from a native, combined, or polish summary."""

    shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
    candidate = {
        "id": _candidate_id(label, endpoint, index),
        "label": label,
        "source_kind": source_kind,
        "run_dir": str(run_dir.resolve()) if run_dir else "",
        "lambda_q": _float(endpoint.get("lambda_q"), float("nan")),
        "endpoint_b": _float(endpoint.get("endpoint_b"), 0.0),
        "keep_exti": bool(endpoint.get("keep_exti", False)),
        "hkl_safe_factor": _float(endpoint.get("hkl_safe_factor"), 1.0),
        "modulation_p95": _float(endpoint.get("mod_p95", endpoint.get("modulation_p95")), 1.0),
        "R1_gt": _float(shelxl.get("R1_gt")),
        "R1_all": _float(shelxl.get("R1_all")),
        "wR2_all": _float(shelxl.get("wR2_all")),
        "GooF": _float(shelxl.get("GooF")),
        "N_gt": shelxl.get("N_gt"),
        "N_all": shelxl.get("N_all"),
        "unique": shelxl.get("unique"),
        "npd_count": _int(shelxl.get("npd_count", endpoint.get("npd_count", 0))),
        "split_count": _int(shelxl.get("split_count", endpoint.get("split_count", 0))),
        "npd_atoms": list(shelxl.get("npd_atoms", endpoint.get("npd_atoms", [])) or []),
        "has_npd": bool(shelxl.get("has_npd", endpoint.get("has_npd", False))),
        "has_split_warnings": bool(shelxl.get("has_split_warnings", endpoint.get("has_split_warnings", False))),
        "hkl": endpoint.get("hkl", ""),
        "lst": shelxl.get("lst", ""),
        "res": str(Path(shelxl["lst"]).with_suffix(".res")) if shelxl.get("lst") else "",
        "shelxl": dict(shelxl),
    }
    if accepted_polish is not None:
        candidate["observation_polish_accepted"] = bool(accepted_polish)
    return candidate


def _collect_native_candidates(runs: Mapping[str, Path]) -> list[dict]:
    """Collect endpoint candidates from completed native run summaries."""

    candidates: list[dict] = []
    for label, run in runs.items():
        run_dir = Path(run)
        summary = _read_json(run_dir / "summary.json")
        endpoints = summary.get("endpoint_results_sorted") or []
        if not endpoints and summary.get("best_endpoint"):
            endpoints = [summary["best_endpoint"]]
        for index, endpoint in enumerate(endpoints, start=1):
            candidates.append(_normalize_endpoint(label, endpoint, run_dir=run_dir, source_kind="native_corrected_hkl", index=index))
    return candidates


def _collect_combined_candidate(combined_run: Path | None) -> list[dict]:
    """Collect the merged multi-dataset endpoint candidate, when present."""

    if not combined_run:
        return []
    run_dir = Path(combined_run)
    summary = _read_json(run_dir / "combined_summary.json")
    endpoint = summary.get("combined_endpoint") or {}
    if not endpoint:
        return []
    endpoint = {**endpoint, "endpoint_b": 0.0, "keep_exti": False, "hkl_safe_factor": 1.0, "mod_p95": 1.0}
    return [_normalize_endpoint("combined", endpoint, run_dir=run_dir, source_kind="combined_corrected_hkl", index=1)]


def _observation_polish_summary_path(forensic_summary: Mapping) -> Path | None:
    """Resolve an observation-polish summary path from a forensic gate summary."""

    artifacts = forensic_summary.get("artifacts") if isinstance(forensic_summary.get("artifacts"), Mapping) else {}
    report = artifacts.get("observation_polish")
    if report:
        return Path(report).with_name("summary.json")
    return None


def _collect_observation_polish_candidates(forensic_summaries: Mapping[str, Mapping] | None) -> list[dict]:
    """Collect optional final-polish endpoint candidates from forensic outputs."""

    candidates: list[dict] = []
    for label, forensic_summary in (forensic_summaries or {}).items():
        if not isinstance(forensic_summary, Mapping):
            continue
        summary_path = _observation_polish_summary_path(forensic_summary)
        if not summary_path:
            continue
        summary = _read_json(summary_path)
        endpoints = summary.get("endpoint_results_sorted") or []
        accepted = bool(summary.get("accepted", False))
        for index, endpoint in enumerate(endpoints, start=1):
            endpoint_accepted = bool(accepted and endpoint.get("accepted_endpoint", False))
            candidates.append(
                _normalize_endpoint(
                    f"{label}:observation_polish",
                    endpoint,
                    run_dir=summary_path.parent,
                    source_kind="observation_polish",
                    index=index,
                    accepted_polish=endpoint_accepted,
                )
            )
    return candidates


def classify_publication_endpoint(
    candidate: Mapping,
    *,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    publication_goof_min: float = 0.85,
    publication_goof_max: float = 1.15,
    allow_npd: bool = False,
) -> dict:
    """Classify one endpoint as publishable physical output or diagnostic only."""

    safety = classify_endpoint_candidate(
        candidate,
        min_hkl_safe_factor=min_hkl_safe_factor,
        max_modulation_p95=max_modulation_p95,
    ).to_dict()
    reasons: list[str] = list(safety.get("reasons") or [])
    endpoint_b = _float(candidate.get("endpoint_b"), 0.0)
    lambda_q = _float(candidate.get("lambda_q"), 1.0)
    source_kind = str(candidate.get("source_kind", ""))
    if abs(lambda_q - 1.0) > 1e-9:
        reasons.append("non-unit q exponent is diagnostic-only; GROND publication endpoints use the full Bloch/Born correction")
    if bool(candidate.get("keep_exti", False)):
        reasons.append("EXTI endpoint correction is comparison-only; GROND publication endpoints keep EXTI off")
    if abs(endpoint_b) > 1e-12:
        reasons.append("nonzero endpoint-B is diagnostic-only because it reshapes corrected HKL intensities and can distort ADPs or maps")
    if source_kind == "observation_polish" and not bool(candidate.get("observation_polish_accepted", False)):
        reasons.append("observation-polish correction did not pass the raw-row and endpoint holdout gates")
    npd_count = _int(candidate.get("npd_count", 0))
    if not allow_npd and (npd_count > 0 or bool(candidate.get("has_npd", False))):
        atoms = candidate.get("npd_atoms")
        atom_text = ""
        if isinstance(atoms, list) and atoms:
            atom_text = f" ({', '.join(str(atom) for atom in atoms[:6])}{', ...' if len(atoms) > 6 else ''})"
        reasons.append(f"endpoint has {npd_count} NPD atom(s){atom_text}; publication endpoints require NPD == 0")
    goof = _float(candidate.get("GooF"))
    if not math.isfinite(goof):
        reasons.append("endpoint GooF is unavailable; publication endpoints must have conventional SHELXL GooF")
    elif goof < publication_goof_min or goof > publication_goof_max:
        reasons.append(
            f"endpoint GooF {goof:.3f} is outside the publication gate [{publication_goof_min:.3f}, {publication_goof_max:.3f}]"
        )
    metrics_ok = math.isfinite(_float(candidate.get("wR2_all"))) and math.isfinite(_float(candidate.get("R1_all")))
    if not metrics_ok and "endpoint metrics are incomplete" not in reasons:
        reasons.append("endpoint metrics are incomplete")
    promotable = bool(metrics_ok and not reasons)
    return {
        "publication_class": "physical" if promotable else "diagnostic",
        "promotable": promotable,
        "rejection_reasons": reasons,
        "safety": safety,
    }


def choose_publication_endpoints(
    candidates: list[dict],
    *,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    publication_goof_min: float = 0.85,
    publication_goof_max: float = 1.15,
    prefer_combined: bool = False,
    allow_npd: bool = False,
) -> dict:
    """Choose best physical and diagnostic endpoints from normalized candidates."""

    classified: list[dict] = []
    for candidate in candidates:
        decision = classify_publication_endpoint(
            candidate,
            min_hkl_safe_factor=min_hkl_safe_factor,
            max_modulation_p95=max_modulation_p95,
            publication_goof_min=publication_goof_min,
            publication_goof_max=publication_goof_max,
            allow_npd=allow_npd,
        )
        classified.append({**candidate, **decision})

    physical = sorted([c for c in classified if c.get("promotable")], key=_metric_key)
    combined_physical = [c for c in physical if _is_combined_candidate(c)]
    best_physical = (combined_physical[0] if combined_physical and prefer_combined else (physical[0] if physical else {}))
    best_physical_id = best_physical.get("id")
    single_physical = [c for c in physical if not _is_combined_candidate(c)]
    best_single_physical = single_physical[0] if single_physical else {}
    rejected: list[dict] = []
    for candidate in classified:
        if candidate.get("id") == best_physical_id:
            continue
        if candidate.get("promotable"):
            candidate = dict(candidate)
            candidate["publication_class"] = "physical_not_selected"
            if prefer_combined and not _is_combined_candidate(candidate) and best_physical:
                candidate["rejection_reasons"] = ["single-dataset endpoint is physically promotable but is reported as a diagnostic because the multi-dataset endpoint is the publication endpoint"]
            else:
                candidate["rejection_reasons"] = ["physically promotable, but worse than the selected physical endpoint by the endpoint ranking metric"]
        rejected.append(candidate)

    diagnostic = sorted([c for c in rejected if c.get("publication_class") != "physical_not_selected"], key=_metric_key)
    best_diagnostic = diagnostic[0] if diagnostic else {}
    verdict = "ok" if best_physical else ("diagnostic_only" if diagnostic else "no_endpoint_candidates")
    return {
        "status": verdict,
        "best_physical_endpoint": best_physical,
        "best_single_dataset_physical_endpoint": best_single_physical,
        "best_diagnostic_endpoint": best_diagnostic,
        "rejected_corrections": sorted(rejected, key=_metric_key),
        "candidates": sorted(classified, key=_metric_key),
        "policy": {
            "physical_endpoint_requires": [
                "finite endpoint R factors",
                "q exponent equal to one",
                "EXTI off",
                "endpoint-B equal to zero",
                f"HKL safety factor >= {min_hkl_safe_factor:g}",
                f"95th percentile modulation <= {max_modulation_p95:g}",
                f"SHELXL GooF between {publication_goof_min:g} and {publication_goof_max:g}",
                "NPD == 0; split warnings are reported but do not block promotion",
                "observation-polish endpoints must pass raw-row and endpoint holdout gates",
            ],
            "ranking": "multi-dataset combined endpoint first when present, then endpoint wR2(F^2)all, R1(F)all, and R1(F)gt",
            "allow_npd_override": bool(allow_npd),
        },
    }


def _candidate_line(candidate: Mapping) -> str:
    """Render one endpoint candidate row for Markdown."""

    reasons = "; ".join(str(r) for r in candidate.get("rejection_reasons", [])[:3])
    goof = _float(candidate.get("GooF"))
    goof_text = "n/a" if not math.isfinite(goof) else f"{goof:.3f}"
    pathology = f"NPD {_int(candidate.get('npd_count', 0))}; split {_int(candidate.get('split_count', 0))}"
    return (
        f"| {candidate.get('label', '')} | {candidate.get('source_kind', '')} | "
        f"{_float(candidate.get('endpoint_b'), 0.0):.3g} | "
        f"{'on' if candidate.get('keep_exti') else 'off'} | {candidate.get('publication_class', '')} | "
        f"{_percent(candidate.get('R1_gt'))} | {_percent(candidate.get('R1_all'))} | {_percent(candidate.get('wR2_all'))} | "
        f"{goof_text} | {pathology} | "
        f"{reasons} |"
    )


def _write_markdown(out_dir: Path, decision: Mapping) -> Path:
    """Write the human-readable endpoint chooser report."""

    best_physical = decision.get("best_physical_endpoint") or {}
    best_single_physical = decision.get("best_single_dataset_physical_endpoint") or {}
    best_diagnostic = decision.get("best_diagnostic_endpoint") or {}
    lines = [
        "# Publication Endpoint Chooser",
        "",
        "This report separates the endpoint suitable for publication from endpoints that are useful diagnostics but rely on corrections GROND does not promote by default.",
        "",
        "## Verdict",
        "",
    ]
    if best_physical:
        lines.append(
            f"Best physical endpoint: `{best_physical.get('id')}` with `R1(F)all={_percent(best_physical.get('R1_all'))}` and `wR2(F^2)all={_percent(best_physical.get('wR2_all'))}`."
        )
    else:
        lines.append("No physical endpoint passed the publication gate. The run should be reported as diagnostic-only until a no-EXTI, B=0 endpoint with finite metrics is available.")
    if best_diagnostic:
        lines.append(
            f"Best diagnostic endpoint: `{best_diagnostic.get('id')}` with `R1(F)all={_percent(best_diagnostic.get('R1_all'))}` and `wR2(F^2)all={_percent(best_diagnostic.get('wR2_all'))}`."
        )
    else:
        lines.append("No rejected diagnostic endpoint beat or supplemented the physical candidate set.")
    if best_single_physical and best_single_physical.get("id") != best_physical.get("id"):
        lines.append(
            f"Best single-dataset physical diagnostic: `{best_single_physical.get('id')}` with `R1(F)all={_percent(best_single_physical.get('R1_all'))}` and `wR2(F^2)all={_percent(best_single_physical.get('wR2_all'))}`."
        )
    lines.extend(
        [
            "",
            "Promotion rule: finite endpoint metrics, conventional SHELXL GooF inside the configured gate, full q correction, EXTI off, endpoint-B equal to zero, no NPD pathology unless explicitly overridden, safe HKL rescaling, bounded intensity modulation, and a passed observation-polish gate when a polish correction is involved. Split warnings are reported but do not block promotion.",
            "",
            "## Endpoint Candidates",
            "",
            "| label | source | endpoint B | EXTI | class | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | pathology | main rejection reason |",
            "|---|---|---:|:---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for candidate in decision.get("candidates", []):
        lines.append(_candidate_line(candidate))
    lines.extend(["", "## Rejected Corrections", ""])
    rejected = decision.get("rejected_corrections") or []
    if rejected:
        for candidate in rejected:
            reasons = "; ".join(str(r) for r in candidate.get("rejection_reasons", [])) or "not selected"
            lines.append(f"- `{candidate.get('id')}`: {reasons}.")
    else:
        lines.append("No endpoint corrections were rejected.")
    path = out_dir / "PUBLICATION_ENDPOINT_CHOOSER.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_tex(out_dir: Path, decision: Mapping) -> Path:
    """Write a small LaTeX table for supplemental reports."""

    rows = []
    for candidate in decision.get("candidates", []):
        rows.append(
            " & ".join(
                [
                    str(candidate.get("label", "")).replace("_", r"\_"),
                    str(candidate.get("source_kind", "")).replace("_", r"\_"),
                    f"{_float(candidate.get('endpoint_b'), 0.0):.3g}",
                    "on" if candidate.get("keep_exti") else "off",
                    str(candidate.get("publication_class", "")).replace("_", r"\_"),
                    _percent(candidate.get("R1_all")).replace("%", r"\%"),
                    _percent(candidate.get("wR2_all")).replace("%", r"\%"),
                ]
            )
            + r" \\"
        )
    text = "\n".join(
        [
            r"\begin{tabular}{llrllll}",
            r"\hline",
            r"Label & Source & B & EXTI & Class & R1(F)all & wR2(F^2)all \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    path = out_dir / "publication_endpoint_chooser.tex"
    path.write_text(text)
    return path


def write_publication_endpoint_choice(
    runs: Mapping[str, Path],
    out_dir: Path,
    *,
    combined_run: Path | None = None,
    forensic_summaries: Mapping[str, Mapping] | None = None,
    extra_candidates: list[dict] | None = None,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    publication_goof_min: float = 0.85,
    publication_goof_max: float = 1.15,
    allow_npd: bool = False,
) -> dict:
    """Collect endpoint candidates, choose publication endpoints, and write reports."""

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    candidates.extend(_collect_native_candidates(runs))
    candidates.extend(_collect_combined_candidate(combined_run))
    candidates.extend(_collect_observation_polish_candidates(forensic_summaries))
    candidates.extend(extra_candidates or [])
    decision = choose_publication_endpoints(
        candidates,
        min_hkl_safe_factor=min_hkl_safe_factor,
        max_modulation_p95=max_modulation_p95,
        publication_goof_min=publication_goof_min,
        publication_goof_max=publication_goof_max,
        prefer_combined=combined_run is not None,
        allow_npd=allow_npd,
    )
    json_path = out_dir / "publication_endpoint_choice.json"
    md_path = _write_markdown(out_dir, decision)
    tex_path = _write_tex(out_dir, decision)
    artifacts = {
        **decision,
        "artifacts": {
            "json": str(json_path.resolve()),
            "markdown": str(md_path.resolve()),
            "latex": str(tex_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(artifacts, indent=2, allow_nan=True) + "\n")
    return artifacts
