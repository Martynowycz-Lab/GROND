"""The "SHELXL endpoint on held-out orbits" block of HELDOUT_VALIDATION, Table 1 Panel A and the terminal summary.

Rows come from the free_validation records written by the combine engine (combined_endpoint.free_validation and
combined_unmerged_endpoint.free_validation in combined_summary.json) and by the leak-free start
(controls/leak_free_start/summary.json); grond validate-endpoint writes the same records for a finished run into a new
directory.  Reporting only: nothing is chosen on these numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.validation.endpoint_free import DEFINITION, ROLE

STATISTICS = ("N_all", "N_gt", "R1_gt", "R1_all", "wR2_all", "GooF")
ENDPOINT_LABELS = {
    "grond_single_crystal": "GROND single-crystal endpoint",
    "grond_merged": "GROND merged endpoint",
    "grond_unmerged": "GROND unmerged endpoint",
    "conventional": "Conventional (leak-free start)",
}


def _stats(block: object) -> dict[str, object]:
    """Return the endpoint statistics (N_all, N_gt, R1_gt, R1_all, wR2_all, GooF) of one block; absent ones are None."""

    block = block if isinstance(block, Mapping) else {}
    return {key: block.get(key) for key in STATISTICS}


def _listing_path(record: Mapping[str, object]) -> str | None:
    """Return the SHELXL .lst path next to the zero-cycle HKL, or None when no HKL was recorded."""

    hkl = (record.get("zero_cycle") or {}).get("hkl") if isinstance(record.get("zero_cycle"), Mapping) else None
    return str(Path(str(hkl)).with_suffix(".lst")) if hkl else None


def endpoint_free_row(key: str, record: object, *, source: str = "") -> dict[str, object]:
    """One normalized row (fractions, GooF unitless) of a free_validation record."""

    record = record if isinstance(record, Mapping) else {}
    reproduction = record.get("work_reproduces_endpoint")
    row: dict[str, object] = {
        "endpoint": key,
        "label": ENDPOINT_LABELS.get(key, key),
        "status": str(record.get("status") or "not_recorded"),
        "free": _stats(record.get("free")),
        "work": _stats(record.get("work")),
        "gap_work_minus_free": dict(record.get("gap_work_minus_free") or {}),
        "model": record.get("model"),
        "wght": record.get("wght"),
        "free_hkl": (record.get("artifacts") or {}).get("free_hkl") if isinstance(record.get("artifacts"), Mapping)
        else (str(Path(str(record["directory"])) / "free.hkl") if record.get("directory") else None),
        "listing": _listing_path(record),
        "work_reproduces_endpoint": reproduction.get("all_same") if isinstance(reproduction, Mapping) else None,
        "model_unchanged_by_zero_cycle": record.get("model_unchanged_by_zero_cycle"),
        "free_rms_z": record.get("free_rms_z"),
        "source": source,
    }
    # statistics that reproduce the endpoint only within the rule's allowance (the GooF's last printed digit)
    within_allowance = [
        f"{name} {value.get('recorded')} / {value.get('zero_cycle')}"
        for name, value in (reproduction.items() if isinstance(reproduction, Mapping) else [])
        if isinstance(value, Mapping) and value.get("same") and int(value.get("difference_units_of_last_digit") or 0) > 0
    ]
    if within_allowance:
        row["work_reproduction_within_allowance"] = within_allowance
    if record.get("error"):
        row["error"] = record.get("error")
    return row


def endpoint_free_rows(
    combined_summary: Mapping[str, object] | None,
    leak_free: Mapping[str, object] | None,
    *,
    combined_source: str = "",
    leak_free_source: str = "",
) -> list[dict[str, object]]:
    """GROND merged, GROND unmerged and conventional rows that have a free_validation record."""

    rows = []
    summary = combined_summary if isinstance(combined_summary, Mapping) else {}
    merged = (summary.get("combined_endpoint") or {}).get("free_validation") if isinstance(summary.get("combined_endpoint"), Mapping) else None
    if isinstance(merged, Mapping):
        rows.append(endpoint_free_row("grond_merged", merged, source=combined_source))
    unmerged_endpoint = summary.get("combined_unmerged_endpoint")
    unmerged = unmerged_endpoint.get("free_validation") if isinstance(unmerged_endpoint, Mapping) else None
    if isinstance(unmerged, Mapping):
        rows.append(endpoint_free_row("grond_unmerged", unmerged, source=combined_source))
    conventional = leak_free.get("free_validation") if isinstance(leak_free, Mapping) else None
    if isinstance(conventional, Mapping):
        rows.append(endpoint_free_row("conventional", conventional, source=leak_free_source))
    return rows


def endpoint_free_block_from_rows(rows: list[dict[str, object]], *, reason: str = "") -> dict[str, object]:
    """Wrap endpoint rows in the report block, available when any row has status ok, with the role and definition."""

    available = [row for row in rows if row.get("status") == "ok"]
    return {
        "status": "available" if available else "not_available",
        "reason": "" if available else (reason or "no endpoint free validation was recorded (--endpoint-free-validation off, "
                                        "or a run that predates it: grond validate-endpoint computes it afterwards)"),
        "role": ROLE,
        "definition": DEFINITION,
        "rows": rows,
    }


def endpoint_free_block(out_dir: Path, result: Mapping[str, object]) -> dict[str, object]:
    """The block of a finished run: its combined summary and its leak-free start record."""

    combined = str(result.get("combined") or "").strip()
    combined_path = Path(combined) / "combined_summary.json" if combined else None
    leak_free = result.get("leak_free_start") if isinstance(result.get("leak_free_start"), Mapping) else None
    leak_free_path = Path(out_dir) / "controls" / "leak_free_start" / "summary.json"
    if not isinstance(leak_free, Mapping) or "free_validation" not in leak_free:
        leak_free = try_read_json(leak_free_path) or leak_free
    rows = endpoint_free_rows(
        try_read_json(combined_path) if combined_path else None,
        leak_free,
        combined_source=str(combined_path or ""),
        leak_free_source=str(leak_free_path),
    )
    reason = "" if combined else "single-crystal runs have no combined endpoint; the conventional row is shown when recorded"
    return endpoint_free_block_from_rows(rows, reason=reason)


def _percent(value: object) -> str:
    """Format a fraction as a percent with two decimals, n/a when it is not a finite number."""

    try:
        number = 100.0 * float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}" if math.isfinite(number) else "n/a"


def _signed_percent(value: object) -> str:
    """Format a fraction as a signed percent with two decimals, n/a when it is not a finite number."""

    try:
        number = 100.0 * float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:+.2f}" if math.isfinite(number) else "n/a"


def _goof(value: object, signed: bool = False) -> str:
    """Format a GooF with three decimals (signed when asked), n/a when it is not a finite number."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def endpoint_free_markdown_lines(block: Mapping[str, object] | None) -> list[str]:
    """Markdown of the "SHELXL endpoint on held-out orbits" section."""

    if not isinstance(block, Mapping):
        return []
    lines = ["### SHELXL endpoint on held-out orbits", ""]
    rows = [row for row in block.get("rows") or [] if isinstance(row, Mapping)]
    if not rows:
        return lines + [f"Not available: {block.get('reason') or 'not recorded'}.", ""]
    lines += [
        "Every final SHELXL model scored with SHELXL L.S. 0 (no parameter moves) on the held-out (free) orbits and on its "
        "own work reflections, in one run (free reflections flagged -1, SHELXL's free-R flag). **Reporting only: nothing "
        "is chosen on these numbers**; the free rows never entered scales, nuisance selection, the dataset policy, the "
        "leak-free start or any refinement HKL. R in percent; R1gt: Fo > 4sig(Fo); GooF on work: SHELXL's S (N - P), on "
        "free: sqrt(sum w D^2 / N) with the final weights at the model's scale. Gap = work minus free.",
        "",
        "| endpoint | status | N free / work | R1gt free / work | R1all free / work | wR2 free / work | GooF free / work | "
        "gap R1gt / R1all / wR2 (pp) / GooF |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        free, work, gap = row.get("free") or {}, row.get("work") or {}, row.get("gap_work_minus_free") or {}
        lines.append(
            f"| {row.get('label')} | {row.get('status')} | {free.get('N_all', 'n/a')} / {work.get('N_all', 'n/a')} | "
            f"{_percent(free.get('R1_gt'))} / {_percent(work.get('R1_gt'))} | "
            f"{_percent(free.get('R1_all'))} / {_percent(work.get('R1_all'))} | "
            f"{_percent(free.get('wR2_all'))} / {_percent(work.get('wR2_all'))} | "
            f"{_goof(free.get('GooF'))} / {_goof(work.get('GooF'))} | "
            f"{_signed_percent(gap.get('R1_gt'))} / {_signed_percent(gap.get('R1_all'))} / "
            f"{_signed_percent(gap.get('wR2_all'))} / {_goof(gap.get('GooF'), signed=True)} |"
        )
    notes = []
    for row in rows:
        bits = []
        if row.get("work_reproduces_endpoint") is not None:
            bits.append(f"work score reproduces the endpoint's own statistics: {'yes' if row['work_reproduces_endpoint'] else 'no'}")
        if row.get("work_reproduction_within_allowance"):
            bits.append(
                "within one unit of the last printed digit (endpoint / L.S. 0): "
                + ", ".join(str(item) for item in row["work_reproduction_within_allowance"])
            )
        if row.get("model_unchanged_by_zero_cycle") is not None:
            bits.append(f"model unchanged by L.S. 0: {'yes' if row['model_unchanged_by_zero_cycle'] else 'no'}")
        rms = row.get("free_rms_z")
        if rms is not None:
            bits.append(f"free-row RMS(z) with the work-fitted sigma model {_goof(rms)}")
        if row.get("error"):
            bits.append(f"error: {row['error']}")
        if bits:
            notes.append(f"- {row.get('label')}: " + "; ".join(bits) + ".")
    return lines + ["", *notes, ""]


def endpoint_free_terminal_line(block: Mapping[str, object] | None) -> str:
    """One terminal line: held-out R1gt / wR2 / GooF of every validated endpoint (free vs work)."""

    if not isinstance(block, Mapping):
        return ""
    rows = [row for row in block.get("rows") or [] if isinstance(row, Mapping)]
    if not rows:
        return f"  SHELXL endpoint on held-out orbits: not available ({block.get('reason') or 'not recorded'})"
    short = {"grond_merged": "GROND merged", "grond_unmerged": "GROND unmerged", "conventional": "conventional"}
    parts = []
    for row in rows:
        free, work = row.get("free") or {}, row.get("work") or {}
        if row.get("status") != "ok":
            parts.append(f"{short.get(str(row.get('endpoint')), row.get('label'))} {row.get('status')}")
            continue
        parts.append(
            f"{short.get(str(row.get('endpoint')), row.get('label'))} R1gt {_percent(free.get('R1_gt'))}/"
            f"{_percent(work.get('R1_gt'))} %, wR2 {_percent(free.get('wR2_all'))}/{_percent(work.get('wR2_all'))} %, "
            f"GooF {_goof(free.get('GooF'))}/{_goof(work.get('GooF'))}"
        )
    return "  SHELXL endpoint on held-out orbits (free/work, L.S. 0, reporting only): " + "; ".join(parts)


TABLE1_MODELS = {
    "grond_merged": "GROND dynamical merged endpoint: held-out orbits",
    "grond_unmerged": "GROND dynamical unmerged endpoint: held-out orbits",
    "conventional": "Conventional leak-free start (work orbits only): held-out orbits",
}


def table1_free_records(block: Mapping[str, object] | None) -> dict[str, list[dict[str, object]]]:
    """Panel A records (the Table 1 SHELXL record shape) of the held-out scores, keyed by endpoint.

    The conventional entry also carries the same zero-cycle model on its own work orbits, the like-for-like
    counterpart the Panel A conventional reference (refined on every reflection) cannot be.
    """

    out: dict[str, list[dict[str, object]]] = {}
    if not isinstance(block, Mapping):
        return out
    for row in block.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("endpoint"))
        records = []
        sets = [("work", "Conventional leak-free start (work orbits only): work orbits")] if key == "conventional" else []
        sets.append(("free", TABLE1_MODELS.get(key, key)))
        for part, label in sets:
            stats = row.get(part) or {}
            available = row.get("status") == "ok" and stats.get("R1_all") is not None
            records.append(
                {
                    "model": label,
                    "role": "SHELXL L.S. 0 of the final model (no parameter moves), reporting only"
                    + ("; GooF with N (nothing fitted to these reflections)" if part == "free" else "; GooF: SHELXL S (N - P)"),
                    "status": "available" if available else "not_available",
                    "status_reason": "" if available else f"endpoint free validation status {row.get('status')!r}",
                    "Ngt": stats.get("N_gt") if available else None,
                    "Nall": stats.get("N_all") if available else None,
                    "R1gt": stats.get("R1_gt") if available else None,
                    "R1all": stats.get("R1_all") if available else None,
                    "wR2all": stats.get("wR2_all") if available else None,
                    "GooF": stats.get("GooF") if available else None,
                    "parameters": None,
                    "restraints": None,
                    "WGHT": str(row.get("wght") or "") or None,
                    "EXTI": None,
                    "NPD": None,
                    "split": None,
                    "BASF": None,
                    "res": str(row.get("model") or ""),
                    "hkl": str(row.get("free_hkl") or "") if part == "free" else "",
                    "lst": str(row.get("listing") or ""),
                }
            )
        out[key] = records
    return out
