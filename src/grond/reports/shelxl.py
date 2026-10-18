"""Read conventional endpoint statistics, model pathologies and active weighting terms."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.engine.shelx_diagnostics import parse_shelx_lst_pathologies
from grond.hashing import sha256_file
from grond.io.config import PrepConfig
from grond.reports.values import _quick_float, _quick_int, _quick_number
from grond.workflows.conventional_baseline import _fresh_baseline_shelxl, _validated_fresh_conventional_baseline
from grond.workflows.model_selection import dataset_baseline_hkl

_QUICK_SHELX_FLOAT_TOKEN_RE = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][-+]?\d+)?")


def _quick_parse_wght_terms(line: str) -> list[float]:
    """Parse numeric terms from one SHELX WGHT line without crossing lines."""

    parts = line.strip().split()
    if not parts or parts[0].upper() != "WGHT":
        return []
    terms: list[float] = []
    for token in parts[1:]:
        if not _QUICK_SHELX_FLOAT_TOKEN_RE.fullmatch(token):
            break
        terms.append(float(token))
    return terms


def _quick_iter_recommended_wght_terms(text: str):
    """Yield the WGHT terms of every recommended weighting scheme line of a SHELXL listing, in order."""

    for line in text.splitlines():
        upper = line.upper()
        if "RECOMMENDED WEIGHTING SCHEME" not in upper or "WGHT" not in upper:
            continue
        terms = _quick_parse_wght_terms(line[upper.find("WGHT") :])
        if terms:
            yield terms


def _quick_lst_diagnostics(lst_path: Path | None) -> dict[str, object]:
    """Parse the SHELXL LST fields users need for endpoint sanity checks."""

    if lst_path is None:
        return {"lst": "", "exists": False, "pathology_status": "missing", "safe_for_feedback": False}
    out: dict[str, object] = {
        "lst": str(lst_path),
        "exists": lst_path.exists(),
        "pathology_status": "missing",
        "safe_for_feedback": False,
    }
    if not lst_path.exists():
        return out
    try:
        text = lst_path.read_text(errors="replace")
    except OSError as exc:
        out.update({"exists": False, "error": str(exc)})
        return out

    matches = re.findall(r"wR2 =\s*([0-9.]+), GooF = S =\s*([0-9.]+).* for all data", text)
    if matches:
        wr2, goof = matches[-1]
        out["wR2_all"] = float(wr2)
        out["GooF"] = float(goof)
    matches = re.findall(r"\+\s+wR2:\s*([0-9.]+),\s*GooF:\s*([0-9.]+)", text)
    if matches:
        wr2, goof = matches[-1]
        out["wR2_all"] = float(wr2)
        out["GooF"] = float(goof)
    matches = re.findall(r"R1 =\s*([0-9.]+) for\s+(\d+) Fo > 4sig\(Fo\) and\s+([0-9.]+) for all\s+(\d+) data", text)
    if matches:
        r1_gt, n_gt, r1_all, n_all = matches[-1]
        out["R1_gt"] = float(r1_gt)
        out["N_gt"] = int(n_gt)
        out["R1_all"] = float(r1_all)
        out["N_all"] = int(n_all)
    matches_gt = re.findall(r"\+\s+R1:\s*([0-9.]+) for\s+(\d+) reflections", text)
    matches_all = re.findall(r"\+\s+R1 \(all\):\s*([0-9.]+) for\s+(\d+) reflections", text)
    if matches_gt and matches_all:
        r1_gt, n_gt = matches_gt[-1]
        r1_all, n_all = matches_all[-1]
        out["R1_gt"] = float(r1_gt)
        out["N_gt"] = int(n_gt)
        out["R1_all"] = float(r1_all)
        out["N_all"] = int(n_all)
    matches = re.findall(r"R\(int\) =\s*([0-9.]+)\s+R\(sigma\) =\s*([0-9.]+)", text)
    if matches:
        rint, rsigma = matches[-1]
        out["Rint"] = float(rint)
        out["Rsigma"] = float(rsigma)
    recommended_wght_terms = list(_quick_iter_recommended_wght_terms(text))
    if recommended_wght_terms:
        terms = recommended_wght_terms[-1]
        out["recommended_wght_a"] = terms[0]
        out["recommended_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
        out["recommended_wght_terms"] = terms
    for line in text.splitlines():
        terms = _quick_parse_wght_terms(line)
        if terms:
            out["active_wght_a"] = terms[0]
            out["active_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
            out["active_wght_terms"] = terms
            break
    matches = re.findall(r"Total number of l\.s\. parameters =\s+(\d+)", text)
    if matches:
        out["parameters"] = int(matches[-1])
    matches = re.findall(r"Restrained GooF =\s+[0-9.]+ for\s+(\d+) restraints", text)
    if matches:
        out["restraints"] = int(matches[-1])
    matches = re.findall(r"\+\++\s*(\d+)\s+Constraints\s+\|\s+(\d+)\s+Restraints\s+\|\s+(\d+)\s+Parameters", text)
    if matches:
        _constraints, restraints, parameters = matches[-1]
        out["parameters"] = int(parameters)
        out["restraints"] = int(restraints)
    npd_atoms = sorted(set(re.findall(r"\s([A-Z][A-Z0-9]{1,7})\s+\*\* NON POSITIVE DEFINITE \*\*", text)))
    npd_matches = re.findall(r"(\d+)\s+atoms NPD", text)
    if npd_matches or npd_atoms:
        out["npd_atoms"] = npd_atoms
        out["npd_count"] = max(int(npd_matches[-1]) if npd_matches else 0, len(npd_atoms))
        out["has_npd"] = int(out["npd_count"]) > 0
    split_matches = re.findall(r"(\d+)\s+atoms may be split", text)
    if split_matches:
        out["split_count"] = int(split_matches[-1])
        out["has_split_warnings"] = int(out["split_count"]) > 0
    pathology = parse_shelx_lst_pathologies(lst_path)
    out["pathology_status"] = pathology.get("pathology_status", "missing")
    out["safe_for_feedback"] = bool(
        pathology.get("pathology_status") == "complete" and not pathology.get("has_npd", False)
    )
    for key in ("npd_count", "split_count"):
        parsed_count = _quick_int(pathology.get(key))
        current_count = _quick_int(out.get(key))
        if parsed_count is not None:
            out[key] = parsed_count if current_count is None else max(parsed_count, current_count)
    parsed_atoms = pathology.get("npd_atoms") if isinstance(pathology.get("npd_atoms"), list) else []
    current_atoms = out.get("npd_atoms") if isinstance(out.get("npd_atoms"), list) else []
    if parsed_atoms or current_atoms:
        out["npd_atoms"] = sorted({str(atom) for atom in [*parsed_atoms, *current_atoms]})
    out["has_npd"] = bool(out.get("has_npd", False) or pathology.get("has_npd", False))
    out["has_split_warnings"] = bool(
        out.get("has_split_warnings", False) or pathology.get("has_split_warnings", False)
    )
    return out


def _quick_shelxl_details(source: Mapping[str, object] | None = None, lst_path: Path | None = None) -> dict[str, object]:
    """Merge recorded SHELXL stats with LST-derived health diagnostics."""

    out: dict[str, object] = dict(source or {})
    if lst_path is None:
        text = str(out.get("lst") or out.get("endpoint_lst") or "")
        lst_path = Path(text) if text else None
    diagnostics = _quick_lst_diagnostics(lst_path)
    out.setdefault("lst", diagnostics.get("lst", ""))
    out["lst_exists"] = bool(diagnostics.get("exists"))
    out["pathology_status"] = diagnostics.get("pathology_status", "missing")
    out["lst_safe_for_feedback"] = bool(diagnostics.get("safe_for_feedback", False))
    pathology_count_keys = {"npd_count", "split_count"}
    pathology_flag_keys = {"has_npd", "has_split_warnings"}
    for key in (
        "R1_gt",
        "R1_all",
        "wR2_all",
        "GooF",
        "N_gt",
        "N_all",
        "Rint",
        "Rsigma",
        "parameters",
        "restraints",
        "npd_count",
        "npd_atoms",
        "split_count",
        "has_npd",
        "has_split_warnings",
        "active_wght_a",
        "active_wght_b",
        "recommended_wght_a",
        "recommended_wght_b",
    ):
        old_value = out.get(key)
        if key not in diagnostics:
            continue
        if key in pathology_count_keys:
            old_count = _quick_int(old_value)
            new_count = _quick_int(diagnostics[key])
            if old_count is None:
                out[key] = diagnostics[key]
            elif new_count is not None:
                # A later, un-hash-bound LST may add a warning, but it must
                # never erase pathology recorded by the endpoint producer.
                out[key] = max(old_count, new_count)
        elif key == "npd_atoms":
            old_atoms = old_value if isinstance(old_value, (list, tuple, set)) else []
            new_atoms = diagnostics[key] if isinstance(diagnostics[key], (list, tuple, set)) else []
            out[key] = sorted({str(atom) for atom in [*old_atoms, *new_atoms] if str(atom).strip()})
        elif key in pathology_flag_keys:
            out[key] = bool(old_value) or bool(diagnostics[key])
        elif old_value is None or old_value == "":
            # Recorded summary metrics are the coherent endpoint result.  An
            # arbitrary file that merely occupies the LST path is only a
            # fallback source; without an expected hash it cannot replace
            # already recorded R values, weights, or parameter counts.
            out[key] = diagnostics[key]
    return out


def _quick_selected_feedback_shelxl(
    result: Mapping[str, object],
    final_model: Path | None,
    fallback: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load one internally coherent adopted-feedback SHELXL endpoint.

    Multi-dataset auto can finish with a later diagnostic combined endpoint
    after the shared feedback loop has already retained an earlier clean
    model.  Metrics from that later endpoint must never be merged with the LST
    or pathology status of the retained model.
    """

    if final_model is None:
        return _quick_shelxl_details(fallback or {}, None)
    try:
        model = final_model.expanduser().resolve()
    except OSError:
        model = final_model.expanduser()
    lst = model.with_suffix(".lst")
    source: Mapping[str, object] = {}
    feedback = result.get("structure_feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    cycles = feedback.get("cycles") if isinstance(feedback.get("cycles"), list) else []
    matched_feedback_record = False
    for raw in cycles:
        if not isinstance(raw, Mapping) or raw.get("adopted_for_feedback") is not True:
            continue
        output_text = str(raw.get("output_model") or "").strip()
        if not output_text:
            continue
        try:
            output = Path(output_text).expanduser().resolve()
        except OSError:
            output = Path(output_text).expanduser()
        if output != model:
            continue
        matched_feedback_record = True
        combined_text = str(raw.get("combined") or "").strip()
        combined_summary = try_read_json(Path(combined_text) / "combined_summary.json") if combined_text else {}
        endpoint = combined_summary.get("combined_endpoint") if isinstance(combined_summary, Mapping) else {}
        endpoint = endpoint if isinstance(endpoint, Mapping) else {}
        shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), Mapping) else {}
        artifacts = endpoint.get("artifacts") if isinstance(endpoint.get("artifacts"), Mapping) else {}
        health = shelxl.get("res_health") if isinstance(shelxl.get("res_health"), Mapping) else {}
        artifact_path = str(artifacts.get("res") or "").strip()
        health_path = str(health.get("path") or "").strip()
        artifact_sha = str(artifacts.get("res_sha256") or "").strip()
        health_sha = str(health.get("sha256") or "").strip()
        try:
            paths_match = bool(artifact_path and health_path) and Path(artifact_path).resolve() == model and Path(health_path).resolve() == model
            current_sha = sha256_file(model)
        except OSError:
            paths_match = False
            current_sha = ""
        if paths_match and artifact_sha and health_sha and artifact_sha == health_sha == current_sha:
            source = shelxl
        break
    if source:
        return _quick_shelxl_details(source, lst)
    if lst.exists():
        # The exact LST is still safer than combining statistics recorded for a
        # different endpoint with this model's pathology record.
        return _quick_shelxl_details({}, lst)
    if matched_feedback_record:
        return _quick_shelxl_details({}, None)
    return _quick_shelxl_details(fallback or {}, None)


def _quick_baseline_shelxl(
    prep: PrepConfig,
    final_summary: Mapping[str, object] | None = None,
    result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prefer the hash-bound supplied-input replay over historical fallbacks."""

    structure = Path(prep.structure).expanduser() if prep.structure else None
    hkl = ""
    if prep.datasets:
        try:
            hkl = dataset_baseline_hkl(prep, prep.datasets[0])
        except Exception:
            hkl = prep.baseline_hkl
    elif prep.baseline_hkl:
        hkl = prep.baseline_hkl
    supplied_fresh = _validated_fresh_conventional_baseline(result or {})
    if supplied_fresh:
        lst_text = str(supplied_fresh.get("lst") or "")
        details = _quick_shelxl_details(supplied_fresh, Path(lst_text) if lst_text else None)
        details["res"] = str(supplied_fresh.get("res") or structure or "")
        details["hkl"] = str(supplied_fresh.get("hkl") or hkl or "")
        details["source_res"] = str(supplied_fresh.get("source_res") or structure or "")
        details["source_hkl"] = str(supplied_fresh.get("source_hkl") or hkl or "")
        details["baseline_source"] = "fresh_supplied_input_conventional_control"
        details["fresh_rerun"] = True
        return details
    fresh = _fresh_baseline_shelxl(final_summary or {})
    if fresh:
        lst_text = str(fresh.get("lst") or "")
        details = _quick_shelxl_details(fresh, Path(lst_text) if lst_text else None)
        details["res"] = str(fresh.get("res") or structure or "")
        details["hkl"] = str(hkl or fresh.get("hkl") or "")
        details["baseline_source"] = "final_run_summary.baseline_exti"
        details["fresh_rerun"] = True
        return details

    lst = structure.with_suffix(".lst") if structure is not None else None
    if lst is not None and not lst.exists():
        log = structure.with_suffix(".log") if structure is not None else None
        if log is not None and log.exists():
            lst = log
    details = _quick_shelxl_details({}, lst)
    details["res"] = str(structure or "")
    details["hkl"] = str(hkl or "")
    details["baseline_source"] = "source_sibling_lst_or_log"
    details["fresh_rerun"] = False
    return details


def _quick_shelxl_issue_text(details: Mapping[str, object]) -> str:
    """Format NPD/split problems in one compact cell."""

    pathology_status = str(details.get("pathology_status") or "unverified")
    npd = _quick_int(details.get("npd_count"))
    split = _quick_int(details.get("split_count"))
    atoms = details.get("npd_atoms")
    atom_text = ""
    if isinstance(atoms, list) and atoms:
        shown = ", ".join(str(v) for v in atoms[:5])
        atom_text = f" ({shown}{', ...' if len(atoms) > 5 else ''})"
    return (
        f"LST pathology `{pathology_status}`; "
        f"NPD `{npd if npd is not None else 'n/a'}`{atom_text}; "
        f"split `{split if split is not None else 'n/a'}`"
    )


def _quick_shelxl_weight_text(details: Mapping[str, object]) -> str:
    """Format active and recommended SHELXL weights."""

    active_a = details.get("active_wght_a")
    active_b = details.get("active_wght_b")
    rec_a = details.get("recommended_wght_a")
    rec_b = details.get("recommended_wght_b")
    active = f"{_quick_number(active_a, 4)} {_quick_number(active_b, 4)}" if active_a is not None else "n/a"
    recommended = f"{_quick_number(rec_a, 4)} {_quick_number(rec_b, 4)}" if rec_a is not None else "n/a"
    return f"active `{active}`; recommended `{recommended}`"


def _quick_endpoint_scale_text(endpoint: Mapping[str, object]) -> str:
    """Return a readable endpoint scale/source line without blank parameter text."""

    spec = str(endpoint.get("scale_spec") or endpoint.get("spec") or endpoint.get("source_kind") or "n/a")
    params = endpoint.get("scale_params")
    if params in {None, "", "n/a"}:
        return f"{spec}; source {endpoint.get('source', endpoint.get('selection_source', 'n/a'))}/{endpoint.get('source_kind', 'n/a')}"
    return f"{spec} ({params} parameters)"


def _quick_goof_warning(goof_value: object) -> str:
    """Return a short endpoint warning for GooF values outside 0.85-1.15."""

    goof = _quick_float(goof_value)
    if not math.isfinite(goof):
        return ""
    if 0.85 <= goof <= 1.15:
        return ""
    return (
        "endpoint GooF is outside the preferred 0.85-1.15 range; inspect weights, sigmas, maps, "
        "and continued refinement, but the endpoint remains available for model building"
    )
