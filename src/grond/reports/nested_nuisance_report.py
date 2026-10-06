"""Nested nuisance pick and per-macro-cycle thickness for HELDOUT_VALIDATION, QUICK_STATS and FINAL_SUMMARY.

The records come from ``decisions/<ds>/nested_nuisance.json`` (grond.workflows.nested_nuisance) and from the
``thickness`` entries of the SHELXL feedback trajectory plus ``thickness_update.final`` of the loop summary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json

PARAM_ORDER = ("mean", "shape", "t0", "grad")


def _params_text(record: Mapping[str, object]) -> str:
    """Format the thickness parameters present (mean, shape, t0, grad) as 'name value, ...', n/a when none is."""

    parts = []
    for name in PARAM_ORDER:
        if name in record and record[name] is not None:
            parts.append(f"{name} {float(record[name]):g}")
    return ", ".join(parts) or "n/a"


def _num(value: object, digits: int = 2) -> str:
    """Format a number with fixed digits, n/a when it is not a finite number."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def nested_pick_block(out_dir: Path) -> dict[str, dict] | None:
    """Per crystal: the recorded nested pick, its evaluator, CV R, edges, grid, rows and cost."""

    crystals: dict[str, dict] = {}
    for path in sorted((Path(out_dir) / "decisions").glob("*/nested_nuisance.json")):
        contract = try_read_json(path)
        if not isinstance(contract, Mapping) or not contract:
            continue
        result = contract.get("result") if isinstance(contract.get("result"), Mapping) else {}
        pick = result.get("pick") if isinstance(result.get("pick"), Mapping) else {}
        crystals[path.parent.name] = {
            "status": contract.get("status"),
            "contract": str(path),
            "variant": pick.get("variant"),
            "variant_params": dict(pick.get("variant_params") or {}),
            "thickness": dict(result.get("thickness") or {}),
            "edges": list(result.get("edges") or []),
            "at_physical_bound": list(result.get("at_physical_bound") or []),
            "grid_size": (result.get("grid") or {}).get("size"),
            "fidelity": result.get("fidelity"),
            "rows": result.get("rows"),
            "best_per_family": result.get("best_per_family"),
            "cost_seconds": (result.get("cost") or {}).get("total_seconds"),
            "free_rows_used": result.get("free_rows_used"),
        }
    return crystals or None


def feedback_thickness_block(trajectory: Mapping[str, object] | None, crystal_names: list[str]) -> dict | None:
    """Thickness of every macro-cycle (the structure entering it) and of the kept final model."""

    if not trajectory:
        return None
    rows = []
    for entry in trajectory.get("trajectory") or []:
        thickness = entry.get("thickness") if isinstance(entry, Mapping) else None
        if isinstance(thickness, Mapping):
            for name, record in _per_crystal(thickness, crystal_names).items():
                rows.append({"cycle": entry.get("cycle"), "crystal": name, **dict(record)})
    summary = try_read_json(Path(str(trajectory.get("source")))) if trajectory.get("source") else {}
    update = summary.get("thickness_update") if isinstance(summary, Mapping) else None
    final = update.get("final") if isinstance(update, Mapping) else None
    if isinstance(final, Mapping) and final:
        if isinstance(final.get("record"), Mapping):  # single dataset: {"record", "fields"}
            finals = _per_crystal(final["record"], crystal_names)
        else:  # multi-crystal: {crystal: {"record", "fields"}}
            finals = {str(k): v["record"] for k, v in final.items()
                      if isinstance(v, Mapping) and isinstance(v.get("record"), Mapping)}
        rows += [{"cycle": "final", "crystal": name, **dict(record)} for name, record in finals.items()]
    if not rows:
        return None
    return {"source": trajectory.get("source"), "rows": rows}


def _per_crystal(payload: Mapping[str, object], names: list[str]) -> dict[str, Mapping]:
    """{crystal: record} from a single-dataset record (it has a family) or a per-crystal mapping."""

    if "family" in payload:
        return {names[0] if len(names) == 1 else "crystal": payload}
    return {str(name): record for name, record in payload.items() if isinstance(record, Mapping)}


def nested_nuisance_report(out_dir: Path, trajectory: Mapping[str, object] | None) -> dict | None:
    """Return the run's nested nuisance block (picks per crystal, thickness per macro-cycle), or None."""

    picks = nested_pick_block(out_dir)
    if not picks:
        return None
    return {"crystals": picks, "feedback_thickness": feedback_thickness_block(trajectory, sorted(picks))}


def nested_nuisance_lines(block: Mapping[str, object] | None) -> list[str]:
    """Markdown of the nested pick per crystal and the thickness per macro-cycle."""

    if not block:
        return []
    lines = [
        "### Nested nuisance selection",
        "",
        "Cone, dose, thickness distribution and P-spline lambda chosen jointly per crystal by inner 3-fold orbit CV "
        "R_all on work rows (free and excluded rows are removed before the spectra).",
        "",
        "| crystal | variant | cone sigma (deg) | dose B (A^2) | family | parameters (A) | lambda | evaluator | "
        "inner CV R_all | edges | grid | work rows | N/Q | s |",
        "|---|---|---:|---:|---|---|---:|---|---:|---|---:|---:|---|---:|",
    ]
    for name, c in (block.get("crystals") or {}).items():
        t = c.get("thickness") or {}
        vp = c.get("variant_params") or {}
        dose = vp.get("dose_b_total_A2", vp.get("dose_b_per_frame"))
        fid = c.get("fidelity") or {}
        lines.append(
            f"| {name} | {c.get('variant')} | {_num(vp.get('mosaicity_sigma_deg'), 2)} | {_num(dose, 2)} | "
            f"{t.get('family', 'n/a')} | {_params_text(t)} | {_num(t.get('scale_lambda'), 3)} | {t.get('evaluator', 'n/a')} | "
            f"{_num(t.get('inner_r_all'), 3)} | {', '.join(c.get('edges') or []) or 'none'} | {c.get('grid_size', 'n/a')} | "
            f"{(c.get('rows') or {}).get('work_rows_used', 'n/a')} | {fid.get('n_basis', 'n/a')}/{fid.get('n_points', 'n/a')} | "
            f"{_num(c.get('cost_seconds'), 0)} |"
        )
    lines.append("")
    families = []
    for name, c in (block.get("crystals") or {}).items():
        for fam, best in (c.get("best_per_family") or {}).items():
            families.append(f"{name} {fam}: {_params_text(best.get('thickness') or {})}, CV R_all {_num(best.get('cv_r_all'), 3)} "
                            f"({best.get('variant')})")
    if families:
        lines += ["Best per family: " + "; ".join(families) + ".", ""]
    feedback = block.get("feedback_thickness")
    if feedback and feedback.get("rows"):
        lines += [
            "### Thickness per macro-cycle",
            "",
            "Thickness distribution and lambda of the structure each SHELXL macro-cycle corrected with (re-selected on "
            "its spectra, physics variant fixed); `final` is the kept model the final, hand-pair and publication "
            "stages use.",
            "",
            "| cycle | crystal | family | parameters (A) | lambda | evaluator | inner CV R_all | edges | source |",
            "|---|---|---|---|---:|---|---:|---|---|",
        ]
        for row in feedback["rows"]:
            lines.append(
                f"| {row.get('cycle')} | {row.get('crystal')} | {row.get('family', 'n/a')} | {_params_text(row)} | "
                f"{_num(row.get('scale_lambda'), 3)} | {row.get('evaluator', 'n/a')} | {_num(row.get('inner_r_all'), 3)} | "
                f"{', '.join(row.get('edge') or []) or 'none'} | {row.get('source', 'n/a')} |"
            )
        lines.append("")
    return lines
