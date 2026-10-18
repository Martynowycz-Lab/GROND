"""Write final row and endpoint statistics with their definitions and file provenance."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from grond.reports.endpoint_free_report import table1_free_records
from grond.reports.heldout import table1_panel_c
from grond.reports.shelxl import _quick_shelxl_details
from grond.reports.source_controls import _quick_source_shelx_controls
from grond.reports.values import _quick_fraction, _quick_int, _quick_number, _quick_percent


def _quick_complete_jana_metrics(metrics: object) -> bool:
    """Return whether one row block contains the complete core JANA family."""

    if not isinstance(metrics, Mapping):
        return False
    nobs = _quick_int(metrics.get("Nobs"))
    nall = _quick_int(metrics.get("Nall"))
    mnobs = _quick_int(metrics.get("MNobs"))
    mnall = _quick_int(metrics.get("MNall"))
    if None in {nobs, nall, mnobs, mnall}:
        return False
    if nobs < 0 or nall <= 0 or nobs > nall or mnobs < 0 or mnall <= 0 or mnobs > mnall:
        return False
    if any(_quick_fraction(metrics.get(key)) is None for key in ("Rall", "wRall", "MRall", "MwRall")):
        return False
    if nobs > 0 and _quick_fraction(metrics.get("Robs")) is None:
        return False
    return not (mnobs > 0 and _quick_fraction(metrics.get("MRobs")) is None)


def _quick_final_jana_row_summary(row_metrics: Mapping[str, object]) -> dict[str, object]:
    """Return the final JANA/PETS-style row block, with legacy fallbacks."""

    direct = row_metrics.get("row_mr")
    direct_dynamic = direct.get("dynamic") if isinstance(direct, Mapping) else None
    if _quick_complete_jana_metrics(direct_dynamic):
        out = dict(direct)
        out["source"] = "row_mr"
        return out

    contracts = row_metrics.get("row_resolution_scale_contracts")
    candidates = contracts.get("contracts") if isinstance(contracts, Mapping) else None
    if not isinstance(candidates, list):
        legacy = row_metrics.get("resolution_scale_endpoints")
        candidates = legacy.get("endpoints") if isinstance(legacy, Mapping) else None
    if not isinstance(candidates, list):
        return {}
    preferred = (
        "suggested_resolution_full_scales",
        "requested_resolution_full_scales",
    )
    by_label = {str(row.get("label")): row for row in candidates if isinstance(row, Mapping)}
    for label in preferred:
        candidate = by_label.get(label)
        summary = candidate.get("row_mr") if isinstance(candidate, Mapping) else None
        summary_dynamic = summary.get("dynamic") if isinstance(summary, Mapping) else None
        if _quick_complete_jana_metrics(summary_dynamic):
            out = dict(summary)
            out["source"] = f"row_resolution_scale_contracts:{label}"
            return out
    return {}


def _table1_jana_record(
    label: str,
    role: str,
    metrics: Mapping[str, object] | None,
    source: str,
) -> dict[str, object]:
    """Normalize one JANA/PETS-style row-statistics record for Table 1."""

    available = _quick_complete_jana_metrics(metrics)
    row = metrics if isinstance(metrics, Mapping) else {}
    ninput = _quick_int(row.get("Ninput")) if available else None
    nexcluded = _quick_int(row.get("Nexcluded")) if available else None
    unavailable_reason = source
    if isinstance(metrics, Mapping) and not available:
        unavailable_reason = f"not available: incomplete core JANA/PETS factor family in {source}"
    return {
        "model": label,
        "role": role,
        "status": "available" if available else "not_available",
        "status_reason": "" if available else unavailable_reason,
        "Ninput": ninput,
        "Nexcluded": nexcluded,
        "Nmatched": _quick_int(row.get("Nmatched")) if available else None,
        "Nobs": _quick_int(row.get("Nobs")) if available else None,
        "Nall": _quick_int(row.get("Nall")) if available else None,
        "Robs": _quick_fraction(row.get("Robs")) if available else None,
        "Rall": _quick_fraction(row.get("Rall")) if available else None,
        "wRobs": _quick_fraction(row.get("wRobs")) if available else None,
        "wRall": _quick_fraction(row.get("wRall")) if available else None,
        "Nmerge_input": _quick_int(row.get("Nmerge_input")) if available else None,
        "Nmerge_used": _quick_int(row.get("Nmerge_used")) if available else None,
        "Nmerge_excluded": _quick_int(row.get("Nmerge_excluded")) if available else None,
        "MNobs": _quick_int(row.get("MNobs")) if available else None,
        "MNall": _quick_int(row.get("MNall")) if available else None,
        "MRobs": _quick_fraction(row.get("MRobs")) if available else None,
        "MRall": _quick_fraction(row.get("MRall")) if available else None,
        "MwRobs": _quick_fraction(row.get("MwRobs")) if available else None,
        "MwRall": _quick_fraction(row.get("MwRall")) if available else None,
        "source": source if available else "",
    }


def _table1_shelxl_record(
    label: str,
    role: str,
    details: Mapping[str, object],
    controls: Mapping[str, object],
    *,
    res: str,
    hkl: str,
    lst: str,
) -> dict[str, object]:
    """Normalize one SHELXL endpoint record for Table 1."""

    available = _quick_fraction(details.get("R1_all")) is not None
    return {
        "model": label,
        "role": role,
        "status": "available" if available else "not_available",
        "status_reason": "" if available else "no authoritative SHELXL listing/statistics file was found",
        "Ngt": _quick_int(details.get("N_gt")) if available else None,
        "Nall": _quick_int(details.get("N_all")) if available else None,
        "R1gt": _quick_fraction(details.get("R1_gt")) if available else None,
        "R1all": _quick_fraction(details.get("R1_all")) if available else None,
        "wR2all": _quick_fraction(details.get("wR2_all")) if available else None,
        "GooF": _quick_fraction(details.get("GooF")) if available else None,
        "parameters": _quick_int(details.get("parameters")) if available else None,
        "restraints": _quick_int(details.get("restraints")) if available else None,
        "WGHT": str(controls.get("wght") or "WGHT n/a") if available else None,
        "EXTI": str(controls.get("exti") or "EXTI none") if available else None,
        "NPD": _quick_int(details.get("npd_count")) if available else None,
        "split": _quick_int(details.get("split_count")) if available else None,
        "BASF": None,
        "res": res,
        "hkl": hkl,
        "lst": lst,
    }


def _table1_all_data_record(all_data: Mapping[str, object] | None) -> dict[str, object] | None:
    """Panel A record of the final all-data refinement (``--free-set off``), or None when the free set stayed out."""

    if not isinstance(all_data, Mapping) or str(all_data.get("status")) in {"None", "skipped_free_set_on", "dry_run"}:
        return None
    artifacts = all_data.get("artifacts") if isinstance(all_data.get("artifacts"), Mapping) else {}
    lst = str(artifacts.get("lst") or "")
    details = _quick_shelxl_details({**(all_data.get("shelxl") or {}), **(all_data.get("statistics") or {})},
                                    Path(lst) if lst else None)
    res = str(artifacts.get("res") or "")
    record = _table1_shelxl_record(
        "GROND final all-data refinement (work + free reflections)",
        "the published endpoint refined against its work and free reflections (--free-set off); the result",
        details,
        _quick_source_shelx_controls(Path(res)) if res else {},
        res=res,
        hkl=str(artifacts.get("hkl") or ""),
        lst=lst,
    )
    if all_data.get("status") != "ok":
        record.update(status="not_available", status_reason=f"{all_data.get('status')}: {all_data.get('reason') or ''}")
    return record


def _table1_unmerged_record(unmerged: Mapping[str, object] | None) -> dict[str, object] | None:
    """Panel A record of the unmerged multi-crystal SHELXL endpoint, or None when the run has none."""

    if not isinstance(unmerged, Mapping) or not isinstance(unmerged.get("shelxl"), Mapping):
        return None
    from grond.reports.shelxl import _quick_shelxl_details

    artifacts = unmerged.get("artifacts") if isinstance(unmerged.get("artifacts"), Mapping) else {}
    lst = str(artifacts.get("lst") or "")
    details = _quick_shelxl_details(dict(unmerged["shelxl"]), Path(lst) if lst else None)
    res_text = str(artifacts.get("res") or "")
    controls = _quick_source_shelx_controls(Path(res_text)) if res_text else {}
    record = _table1_shelxl_record(
        "GROND dynamical unmerged multi-crystal endpoint",
        "every corrected work row unmerged, batch = crystal, MERG 0, refined BASF",
        details,
        controls,
        res=res_text,
        hkl=str(artifacts.get("hkl") or ""),
        lst=lst,
    )
    basf = unmerged.get("basf") if isinstance(unmerged.get("basf"), Mapping) else {}
    values = list(basf.get("values") or [])
    esds = list(basf.get("esds") or [])
    if values:
        record["BASF"] = ", ".join(
            f"{value:.4f} ({esds[i]:.4f})" if i < len(esds) and esds[i] is not None else f"{value:.4f}"
            for i, value in enumerate(values)
        )
    if record["status"] != "available":
        record["status_reason"] = f"unmerged endpoint status {unmerged.get('status')!r}: no SHELXL statistics"
    return record


HELD_OUT_ROWS_DEFINITION = (
    "Held-out rows (`held-out orbits`, reporting only; nothing is chosen on them): the final model of the endpoint above scored with SHELXL L.S. 0 (no parameter moves; final WGHT, BASF and EXTI) on the free orbits, which never entered any refinement HKL. GROND free rows are corrected exactly like the work rows. R1 is SHELXL's R1(Free); wR2 and GooF are on the model's scale with the final weights, GooF with N (no parameter was fitted to these reflections). The conventional leak-free start (the supplied model refined on the work orbits of the baseline HKL) is shown on its own work orbits and on the free orbits: the conventional reference above was refined on every reflection, the free orbits included, so it has no held-out counterpart. Details: HELDOUT_VALIDATION.md, section SHELXL endpoint on held-out orbits."
)


def _table1_pair(row: Mapping[str, object], first: str, second: str) -> str:
    """Format a count pair from a normalized Table 1 row."""

    a = row.get(first)
    b = row.get(second)
    return f"{a}/{b}" if a is not None and b is not None else "n/a"


def _table1_panel_c_lines(panel: Mapping[str, object]) -> list[str]:
    """Markdown for Panel C: held-out free-row R factors, in-sample, gap and leak flags."""

    lines = ["", "## C. Held-Out Validation (Free Rows)", ""]
    if panel.get("status") != "available":
        return lines + [f"Not available: {panel.get('reason') or 'no held-out rows'}."]

    def ci(value, bounds):
        """Format a fraction and its 95% CI as percent 'value [lo, hi]', or the value alone without a CI."""

        if not bounds or len(bounds) != 2:
            return _quick_percent(value)
        return f"{_quick_percent(value)} [{_quick_percent(bounds[0])}, {_quick_percent(bounds[1])}]"

    def flag(value):
        """Format a flag as yes or no, n/a for None."""

        return "n/a" if value is None else ("yes" if value else "no")

    leak_columns = (
        "free_rows_in_scale_fit",
        "free_rows_in_nuisance_selection",
        "free_rows_in_dataset_policy",
        "free_rows_in_shelxl_hkl",
        "free_rows_in_macro_cycle_selection",
        "free_rows_in_hand_decision",
    )
    lines += [
        "| crystal | Nfree/Nobs/orbits | Rall held-out [95% CI] | Robs held-out [95% CI] | Rall/Robs in-sample | gap Rall/Robs | "
        "free rows in scale fit / nuisance selection / dataset policy / SHELXL HKL / macro-cycle selection / hand decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in panel.get("rows") or []:
        lines.append(
            f"| {row['crystal']} | {row.get('n_free')}/{row.get('n_free_obs')}/{row.get('n_orbits')} | "
            f"{ci(row.get('heldout_Rall'), row.get('heldout_Rall_ci95'))} | {ci(row.get('heldout_Robs'), row.get('heldout_Robs_ci95'))} | "
            f"{_quick_percent(row.get('insample_Rall'))}/{_quick_percent(row.get('insample_Robs'))} | "
            f"{_quick_percent(row.get('gap_Rall'))}/{_quick_percent(row.get('gap_Robs'))} | "
            + " / ".join(flag(row.get(column)) for column in leak_columns)
            + " |"
        )
    definitions = panel.get("definitions") or {}
    lines += [
        "",
        f"Panel C definitions: held-out rows are the `split=free` rows of the final row tables, predicted with the run's own scales; "
        f"Rall/Robs are the Panel B amplitude forms (Robs: Iobs > 3 sigma); 95% CIs: {definitions.get('ci')}; orbits: "
        f"{definitions.get('orbit_keys')}. In-sample is the same model on work rows; gap = held-out minus in-sample. "
        f"A `yes` leak flag means the free rows were used there, so the held-out value is optimistic. SHELXL HKL evidence: "
        f"{panel.get('shelxl_hkl_evidence')}. Macro-cycle selection evidence: {panel.get('macro_cycle_selection_evidence')}. "
        f"Hand-decision evidence: {panel.get('hand_decision_evidence')}. Dataset-policy evidence: "
        f"{panel.get('dataset_policy_evidence')}.",
    ]
    return lines


def _write_final_statistics_table1(
    out_dir: Path,
    *,
    row_metrics: Mapping[str, object],
    row_metrics_path: Path,
    input_shelxl: Mapping[str, object],
    publication_shelxl: Mapping[str, object],
    publication_endpoint: Mapping[str, object],
    file_block: Mapping[str, object],
    input_controls: Mapping[str, object],
    final_row_table: str,
    heldout: Mapping[str, object] | None = None,
    unmerged_endpoint: Mapping[str, object] | None = None,
    all_data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write the canonical final statistics Table 1 without mixing metric families.

    ``unmerged_endpoint`` (the combined summary's ``combined_unmerged_endpoint``) adds the unmerged
    multi-crystal SHELXL endpoint (batch = crystal, MERG 0, refined BASF) to Panel A.
    """

    jana = _quick_final_jana_row_summary(row_metrics)
    dynamic = jana.get("dynamic") if isinstance(jana.get("dynamic"), Mapping) else None
    born = jana.get("born") if isinstance(jana.get("born"), Mapping) else None
    if born is None and isinstance(jana.get("kinematic"), Mapping):
        born = jana.get("kinematic")
    jana_source = str(jana.get("source") or row_metrics_path)
    jana_definition = jana.get("definition") if isinstance(jana.get("definition"), Mapping) else {}
    jana_instability_u = _quick_fraction(jana_definition.get("instability_u"))
    jana_instability_u_text = (
        f"{jana_instability_u:g}" if jana_instability_u is not None else "n/a"
    )

    row_records = [
        _table1_jana_record(
            "Conventional kinematic refinement",
            "merged conventional refinement",
            None,
            "not available: no audited matched unmerged prediction/scale contract; use the SHELXL endpoint panel",
        ),
        _table1_jana_record(
            "GROND Born (matched kinematical row control)",
            "single-scattering control at the final coordinates under the same signed-row contract and scales",
            born,
            jana_source,
        ),
        _table1_jana_record(
            "GROND Bloch dynamical",
            "Bloch multiple-scattering prediction at the final coordinates",
            dynamic,
            jana_source,
        ),
    ]

    publication_res_text = str(file_block.get("publication_res") or "")
    publication_res = Path(publication_res_text) if publication_res_text else None
    publication_controls = _quick_source_shelx_controls(publication_res)
    conventional_endpoint = _table1_shelxl_record(
        "Conventional kinematic reference refinement",
        "starting merged HKLF4 refinement",
        input_shelxl,
        input_controls,
        res=str(input_shelxl.get("res") or ""),
        hkl=str(input_shelxl.get("hkl") or ""),
        lst=str(input_shelxl.get("lst") or ""),
    )
    born_endpoint = _table1_shelxl_record(
        "GROND Born corrected-HKL endpoint",
        "single-scattering endpoint control",
        {},
        {},
        res="",
        hkl="",
        lst="",
    )
    born_endpoint["status_reason"] = (
        "not generated: no actual Born-corrected SHELXL endpoint refinement exists for this run"
    )
    dynamic_endpoint = _table1_shelxl_record(
        "GROND dynamical corrected-HKL model-handoff endpoint",
        "final merged corrected-HKL SHELXL endpoint",
        publication_shelxl,
        publication_controls,
        res=publication_res_text,
        hkl=str(file_block.get("publication_hkl") or ""),
        lst=str(file_block.get("publication_lst") or ""),
    )
    # Held-out scores (SHELXL L.S. 0 on the free orbits, reporting only) sit next to the endpoint they score.
    endpoint_free = heldout.get("shelxl_endpoint_free") if isinstance(heldout, Mapping) else None
    free_records = table1_free_records(endpoint_free)
    endpoint_records = [conventional_endpoint, *free_records.get("conventional", []), born_endpoint, dynamic_endpoint]
    endpoint_records += free_records.get("grond_merged", [])
    unmerged_record = _table1_unmerged_record(unmerged_endpoint)
    if unmerged_record is not None:
        endpoint_records.append(unmerged_record)
    endpoint_records += free_records.get("grond_unmerged", [])
    all_data_record = _table1_all_data_record(all_data)
    if all_data_record is not None:
        endpoint_records.append(all_data_record)

    payload = {
        "schema": "grond.table1_final_statistics.v1",
        "title": "Table 1. Final model and refinement statistics",
        "status": (
            "available"
            if any(row["status"] == "available" for row in row_records + endpoint_records)
            else "not_available"
        ),
        "reporting_contract": {
            "endpoint": "SHELXL style: R1(F)gt, R1(F)all, wR2(F^2)all, GooF, and merged endpoint counts",
            "row": "JANA/PETS style: Robs, Rall, wRall on unmerged rows and MRobs, MRall, MwRall after post-refinement symmetry merging; wRobs and MwRobs are optional extensions; wR and MwR are the amplitude form (Jana2020 wR(all) for refinement on F), MR and MwR after Klar et al. (2023)",
            "separation_rule": "SHELXL endpoint and JANA/PETS row factors have different targets and denominators and are never placed in a shared generic R column.",
            "born_rule": "Born is a matched kinematical row control, not a separately refined conventional SHELXL endpoint.",
            "row_sigma_rule": "JANA/PETS-style weights use immutable source_sigma in target_I units, with historical xds_sigma/sigma aliases and sigma_eff only as a legacy fallback when no source sigma is available. Fitted sigma_eff and raw_row_weight are not otherwise folded into the reported R/MR factors.",
            "row_instability_u": jana_instability_u,
            "matched_support_rule": "Born and Bloch statistics must use one identical finite observation support; Ninput/Nmatched/Nexcluded and merge-HKL exclusions are reported explicitly when known.",
            "missing_rule": "Unavailable statistics remain explicit null/n/a values and are never inferred from a different target class.",
            "numeric_units": "fractions in JSON and CSV; percentages in Markdown",
        },
        "sources": {
            "jana_row_metrics": str(row_metrics_path),
            "final_row_table": final_row_table,
            "publication_endpoint_selection": str(
                publication_endpoint.get("selection_source")
                or publication_endpoint.get("source_kind")
                or ""
            ),
        },
        "shelxl_endpoints": endpoint_records,
        "jana_pets_rows": row_records,
    }
    panel_c = table1_panel_c(heldout)
    if panel_c is not None:
        payload["heldout_validation"] = panel_c
    if isinstance(endpoint_free, Mapping):
        payload["shelxl_endpoint_free"] = endpoint_free

    md_path = out_dir / "TABLE1_FINAL_STATISTICS.md"
    json_path = out_dir / "TABLE1_FINAL_STATISTICS.json"
    endpoint_csv = out_dir / "TABLE1_SHELXL_ENDPOINTS.csv"
    row_csv = out_dir / "TABLE1_JANA_ROWS.csv"
    alias_md = out_dir / "TABLE_1.md"
    alias_json = out_dir / "TABLE_1.json"

    lines = [
        "# Table 1. Final Model And Refinement Statistics",
        "",
        "This is the canonical GROND statistics table. Panel A contains only SHELXL endpoint statistics. Panel B contains only JANA/PETS-style signed-row and post-refinement merged-row factors computed by GROND. The two panels have different targets and denominators and must not be collapsed into generic `R` columns.",
        "",
        "## A. SHELXL Endpoint Statistics",
        "",
        "| model | status | Ngt/Nall | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | parameters/restraints | WGHT | EXTI | NPD/split | BASF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in endpoint_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["status"]).replace("_", " "),
                    _table1_pair(row, "Ngt", "Nall"),
                    _quick_percent(row.get("R1gt")),
                    _quick_percent(row.get("R1all")),
                    _quick_percent(row.get("wR2all")),
                    _quick_number(row.get("GooF"), 3),
                    _table1_pair(row, "parameters", "restraints"),
                    str(row.get("WGHT") or "n/a"),
                    str(row.get("EXTI") or "n/a"),
                    _table1_pair(row, "NPD", "split"),
                    str(row.get("BASF") or "n/a"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "SHELXL definitions: `R1(F)gt` is the conventional amplitude R factor for SHELXL's observed/greater-than subset; `R1(F)all` uses all endpoint reflections; `wR2(F^2)all` and `GooF` use the endpoint sigma and active `WGHT` contract. `Ngt` is not the JANA/PETS `Nobs` count. The unmerged multi-crystal endpoint, when present, refines every corrected work row unmerged (MERG 0, batch = crystal) with one BASF batch scale per extra crystal (value (e.s.d.)); its counts and R factors are over unmerged rows.",
            "",
            "## B. JANA/PETS-Style Row And Post-Refinement Merged Statistics",
            "",
            "| model | status | Ninput/excluded | Nmatched | Nobs/Nall | Nmerge used/excluded | MNobs/MNall | Robs | Rall | wRobs | wRall | MRobs | MRall | MwRobs | MwRall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in row_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["status"]).replace("_", " "),
                    _table1_pair(row, "Ninput", "Nexcluded"),
                    str(row.get("Nmatched")) if row.get("Nmatched") is not None else "n/a",
                    _table1_pair(row, "Nobs", "Nall"),
                    _table1_pair(row, "Nmerge_used", "Nmerge_excluded"),
                    _table1_pair(row, "MNobs", "MNall"),
                    _quick_percent(row.get("Robs")),
                    _quick_percent(row.get("Rall")),
                    _quick_percent(row.get("wRobs")),
                    _quick_percent(row.get("wRall")),
                    _quick_percent(row.get("MRobs")),
                    _quick_percent(row.get("MRall")),
                    _quick_percent(row.get("MwRobs")),
                    _quick_percent(row.get("MwRall")),
                ]
            )
            + " |"
        )
    if free_records:
        at = lines.index("## B. JANA/PETS-Style Row And Post-Refinement Merged Statistics")
        lines[at:at] = [
            HELD_OUT_ROWS_DEFINITION,
            "",
        ]
    unavailable = [row for row in row_records + endpoint_records if row.get("status") != "available"]
    lines.extend(
        [
            "",
            f"Row definitions: `Robs`/`Rall` are R1-form amplitude residuals formed from `sqrt(max(I,0))`; observed subsets satisfy `Iobs > 3 sigma(Iobs)`. `wRall` is the amplitude form of Jana2020's `wR(all)` for refinement on F, `sqrt(sum w (Fo - Fc)^2 / sum w Fo^2)` with `w = 1/[sigma(Fo)^2 + (u Fo)^2]`, `sigma(Fo) = sigma(Iobs)/(2 Fo)` and instability factor `u={jana_instability_u_text}`; the intensity-residual form of earlier versions is kept in the JSON output as `wRall_I`. Born and Bloch factors use the same finite support. `Nmerge excluded` counts supported rows with missing, non-finite, or non-integral HKL indices. `MR`/`MwR` repeat the calculations after arithmetic merging of valid model-symmetry-related observed and calculated rows with propagated merged sigma (Klar et al. 2023). Optional `wRobs`/`MwRobs` apply the weighted form to the observed subsets. Weights use immutable `source_sigma` in target intensity units, with historical `xds_sigma`/`sigma` aliases and `sigma_eff` only as a legacy fallback when no source sigma is available. Fitted `sigma_eff` and `raw_row_weight` are not otherwise folded into the reported factors. Values are computed by GROND and are not parsed from JANA or PETS.",
            "",
            "The Born row is the matched single-scattering control under the final GROND row contract. It is not a third SHELXL endpoint.",
        ]
    )
    if panel_c is not None:
        lines.extend(_table1_panel_c_lines(panel_c))
    if all_data_record is not None and all_data_record["status"] == "available":
        lines.extend(["", "The run used `--free-set off`: its result is the final all-data refinement (last row of Panel A), "
                      "the published endpoint refined against its work and free reflections. Panels B and C describe the "
                      "endpoint before that step, when the free reflections were still held out."])
    if unavailable:
        lines.extend(["", "## Explicitly Unavailable Statistics", ""])
        for row in unavailable:
            lines.append(f"- **{row['model']}:** {row.get('status_reason') or 'not available'}.")
    lines.extend(
        [
            "",
            "## Authoritative Sources",
            "",
            f"- JANA/PETS-style row metrics: `{row_metrics_path}`",
            f"- Final signed row table(s): `{final_row_table}`",
            f"- Conventional SHELXL listing/statistics: `{input_shelxl.get('lst', '')}`",
            f"- Publication SHELXL listing: `{file_block.get('publication_lst', '')}`",
            "",
        ]
    )

    json_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    md_text = "\n".join(lines)
    json_path.write_text(json_text)
    md_path.write_text(md_text)
    alias_json.write_text(json_text)
    alias_md.write_text(md_text)

    for path, records in ((endpoint_csv, endpoint_records), (row_csv, row_records)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {
        "status": payload["status"],
        "markdown": str(md_path),
        "json": str(json_path),
        "endpoint_csv": str(endpoint_csv),
        "row_csv": str(row_csv),
        "short_markdown": str(alias_md),
        "short_json": str(alias_json),
        "payload": payload,
    }
