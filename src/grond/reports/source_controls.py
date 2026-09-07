"""Explain input transforms, SHELX controls and the weights used in reported comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from grond.reports.shelxl import _quick_parse_wght_terms
from grond.reports.values import _quick_fraction, _quick_int, _quick_number


def _quick_transform_summary(transform_decision: Mapping[str, object]) -> dict[str, object]:
    """Return compact transform details for the quick summary."""

    selected = transform_decision.get("selected")
    selected_block = selected if isinstance(selected, Mapping) else {}
    metadata = transform_decision.get("metadata_transform_decision")
    metadata_block = metadata if isinstance(metadata, Mapping) else {}
    claim = transform_decision.get("selected_transform_claim_contract")
    claim_block = claim if isinstance(claim, Mapping) else {}
    transform_meta = claim_block.get("transform")
    transform_meta = transform_meta if isinstance(transform_meta, Mapping) else {}
    orientation = claim_block.get("source_orientation_contract")
    orientation_block = orientation if isinstance(orientation, Mapping) else {}
    return {
        "selected_transform": selected_block.get("transform", "n/a"),
        "selected_status": selected_block.get("status", "n/a"),
        "transform_determinant": selected_block.get(
            "transform_determinant", transform_meta.get("determinant")
        ),
        "transform_parity": selected_block.get(
            "transform_parity", transform_meta.get("parity", "n/a")
        ),
        "transform_parity_policy": transform_decision.get("transform_parity_policy", "n/a"),
        "configuration_status": claim_block.get("configuration_status", "parity_unresolved"),
        "absolute_configuration_claim_allowed": bool(
            claim_block.get("absolute_configuration_claim_allowed", False)
        ),
        "configuration_status_reason": claim_block.get(
            "reason", "transform parity contract is missing"
        ),
        "proper_canonical_transform": transform_meta.get(
            "proper_canonical_transform", "n/a"
        ),
        "source_orientation_status": orientation_block.get("status", "missing"),
        "source_basis_determinant": orientation_block.get("source_basis_determinant"),
        "transformed_basis_determinant": orientation_block.get(
            "transformed_basis_determinant"
        ),
        "right_handed_source": bool(orientation_block.get("right_handed_source", False)),
        "right_handed_transformed": bool(
            orientation_block.get("right_handed_transformed", False)
        ),
        "metadata_status": metadata_block.get("status", "n/a"),
        "metadata_log_intensity_corr": (
            metadata_block.get("records", [{}])[0].get("log_intensity_corr")
            if isinstance(metadata_block.get("records"), list) and metadata_block.get("records")
            else None
        ),
        "metadata_correlation_gap": metadata_block.get("correlation_gap"),
        "metadata_paired_rows": (
            metadata_block.get("records", [{}])[0].get("paired_rows")
            if isinstance(metadata_block.get("records"), list) and metadata_block.get("records")
            else None
        ),
        "metadata_unique_overlap": (
            metadata_block.get("records", [{}])[0].get("unique_overlap")
            if isinstance(metadata_block.get("records"), list) and metadata_block.get("records")
            else None
        ),
    }


def _quick_flat_hand_t_style_text(
    t_value: object,
    n_value: object,
    df_value: object,
) -> str:
    """Format a flat console-payload paired t-style record."""

    n = _quick_int(n_value)
    df = _quick_int(df_value)
    return f"t {_quick_number(t_value, 3)} (n {n if n is not None else 'n/a'}, df {df if df is not None else 'n/a'})"


def _quick_delta_percent(before: object, after: object) -> str:
    """Format an absolute percent-point change for fractional metrics."""

    b = _quick_fraction(before)
    a = _quick_fraction(after)
    if b is None or a is None:
        return "n/a"
    sign = "+" if (a - b) >= 0 else ""
    return f"{sign}{100.0 * (a - b):.2f} pp"


def _quick_source_shelx_controls(path: Path | None) -> dict[str, object]:
    """Read the source RES/INS control records that affect baseline interpretation."""

    out: dict[str, object] = {"hklf": "", "exti": "", "wght": "", "acta": False}
    if path is None or not path.exists():
        return out

    def read_controls(candidate: Path) -> dict[str, object]:
        found: dict[str, object] = {"hklf": "", "exti": "", "wght": "", "acta": False}
        try:
            text = candidate.read_text(errors="replace")
        except OSError:
            return found
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            head = line.split()[0].upper()
            if head == "END":
                break
            if head == "HKLF":
                found["hklf"] = " ".join(line.split()[:2])
            elif head == "EXTI" or line[:4].upper() == "EXTI":
                # SHELXL uses a fixed-width four-character instruction field.
                # Large values may therefore be emitted without a separating
                # blank (for example ``EXTI33651.195312``), while remaining an
                # active EXTI instruction to SHELXL.
                value = line[4:].strip().split()
                found["exti"] = "EXTI" + (f" {value[0]}" if value else "")
            elif head == "WGHT":
                terms = _quick_parse_wght_terms(line)
                found["wght"] = "WGHT " + " ".join(_quick_number(value, 6) for value in terms) if terms else line
            elif head == "ACTA":
                found["acta"] = True
        return found

    out.update(read_controls(path))
    # SHELXL can refine a very large fixed-width EXTI value from the INS yet
    # omit that instruction from the output RES.  Preserve the output RES
    # controls generally, but recover the active EXTI declaration from the
    # exact sibling INS when the RES no longer contains it.
    sibling_ins = path.with_suffix(".ins")
    if not out["exti"] and path.suffix.lower() == ".res" and sibling_ins.is_file():
        out["exti"] = read_controls(sibling_ins)["exti"]
    return out


def _quick_source_shelx_control_text(controls: Mapping[str, object]) -> str:
    """Format source SHELX controls in one compact report cell."""

    hklf = str(controls.get("hklf") or "HKLF n/a")
    exti = str(controls.get("exti") or "EXTI none")
    wght = str(controls.get("wght") or "WGHT n/a")
    acta = "ACTA present" if controls.get("acta") else "ACTA absent"
    return f"{hklf}; {exti}; {wght}; {acta}"


def _quick_row_weight_definition(
    nuisance: Mapping[str, object],
    dataset_contracts: list[Mapping[str, object]] | None = None,
) -> str:
    """Return the exact row intensity weighting convention used by the final native report."""

    definition = (
        "Raw signed-row intensity `wR(I)all` is `sqrt(sum_r w_r (Iobs_r - Icalc_r)^2 / sum_r w_r Iobs_r^2)` "
        "over signed observation rows. For dynamical rows `Icalc_r = Idyn_r`; for matched Born rows "
        "`Icalc_r = Iborn_r` under the same row contract, thickness, geometry, scales, and sigma model. "
        "`w_r = 1 / [sigma_eff_r^2 + (a P_r)^2 + b P_r]`, with "
        "`P_r = (max(Iobs_r, 0) + 2 max(Icalc_r, 0)) / 3`. "
    )
    if dataset_contracts:
        weight_text = []
        for contract in dataset_contracts:
            block = contract.get("native_weight") if isinstance(contract.get("native_weight"), Mapping) else {}
            nui = contract.get("nuisance") if isinstance(contract.get("nuisance"), Mapping) else {}
            a = block.get("a", nui.get("native_wght_a"))
            b = block.get("b", nui.get("native_wght_b"))
            weight_text.append(
                f"{contract.get('dataset', 'dataset')} a={_quick_number(a, 6)}, b={_quick_number(b, 6)}"
            )
        definition += (
            "This multi-dataset result uses independent per-crystal native weights (`per_dataset_native`): "
            + "; ".join(weight_text)
            + ". The combined-summary fallback a/b pair is not an active joint weighting equation. "
        )
    else:
        a = nuisance.get("native_wght_a", nuisance.get("wght_a_grid"))
        b = nuisance.get("native_wght_b", nuisance.get("wght_b_grid"))
        definition += f"This run reports final native row weights `a={_quick_number(a, 6)}` and `b={_quick_number(b, 6)}`. "
    return (
        definition
        + "This is the signed-row intensity statistic to compare with native dynamical-refinement row "
        "targets; it is not SHELXL's merged-reflection `wR2(F^2)all`."
    )
