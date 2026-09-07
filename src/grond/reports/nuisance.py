"""Report thickness distributions and each dataset's pinned nuisance parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping

from grond.reports.values import _quick_number
from grond.workflows.decision import safe_float


def _quick_thickness_text(
    nuisance: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    dataset_contracts: list[Mapping[str, object]] | None = None,
) -> str:
    """Describe effective thickness as a mean plus its distribution contract."""

    per_dataset: list[tuple[str, float, float, str]] = []
    for contract in dataset_contracts or []:
        block = contract.get("nuisance") if isinstance(contract.get("nuisance"), Mapping) else {}
        thickness_a = safe_float(block.get("thickness_A"), float("nan"))
        grad = safe_float(block.get("thickness_grad_frac"), float("nan"))
        if not math.isfinite(thickness_a) or thickness_a <= 0.0:
            continue
        per_dataset.append(
            (
                str(contract.get("dataset") or f"dataset_{len(per_dataset) + 1}"),
                thickness_a,
                grad,
                str(block.get("thickness_model") or "single"),
            )
        )
    if per_dataset:
        centers = [row[1] / 10.0 for row in per_dataset]
        details = []
        for name, thickness_a, grad, model in per_dataset:
            grad_text = f", g={grad:.3f}" if math.isfinite(grad) else ""
            details.append(f"{name} {thickness_a / 10.0:.2f} nm ({model}{grad_text})")
        return (
            "per-dataset centers "
            + "; ".join(details)
            + f"; center range {min(centers):.2f}-{max(centers):.2f} nm"
        )

    meta = metadata if isinstance(metadata, Mapping) else {}
    thickness_a = safe_float(
        nuisance.get("thickness_A", meta.get("thickness_A_mean")),
        float("nan"),
    )
    if not math.isfinite(thickness_a):
        return "n/a"
    thickness_nm = thickness_a / 10.0
    model = str(nuisance.get("thickness_model") or meta.get("thickness_model") or "single")
    if model == "uniform_gradient":
        grad = max(
            0.0,
            safe_float(
                nuisance.get("thickness_grad_frac", meta.get("thickness_grad_frac")),
                0.0,
            ),
        )
        low_nm = max(0.0, thickness_a * (1.0 - grad)) / 10.0
        high_nm = max(0.0, thickness_a * (1.0 + grad)) / 10.0
        return (
            f"{thickness_nm:.2f} nm mean ({thickness_a:.1f} A); "
            f"uniform-gradient support {low_nm:.2f}-{high_nm:.2f} nm"
        )
    if model in {"gaussian", "gaussian_gradient"}:
        sigma_frac = abs(
            safe_float(
                nuisance.get("thickness_sigma_frac", meta.get("thickness_sigma_frac")),
                0.0,
            )
        )
        return (
            f"{thickness_nm:.2f} nm mean ({thickness_a:.1f} A); "
            f"{model} sigma {sigma_frac * thickness_nm:.2f} nm"
        )
    return f"{thickness_nm:.2f} nm ({thickness_a:.1f} A); {model}"


def _quick_dataset_nuisance_table_lines(contracts: list[Mapping[str, object]]) -> list[str]:
    """Render exact per-crystal nuisance contracts for a combined run."""

    if not contracts:
        return []
    lines = [
        "### Per-Dataset Nuisance Contracts",
        "",
        "The crystals retain independent thickness, scale, detector, absorption, and rocking contracts. Their arithmetic mean is not a physical thickness distribution.",
        "",
        "| dataset | thickness center/support | gradient | frame offset | group width / detector bins | absorption | Sg jitter | row-resolution B | scales | native WGHT a/b |",
        "|---|---|---:|---:|---|---|---|---:|---:|---|",
    ]
    for contract in contracts:
        nuisance = contract.get("nuisance") if isinstance(contract.get("nuisance"), Mapping) else {}
        thickness_a = safe_float(nuisance.get("thickness_A"), float("nan"))
        grad = safe_float(nuisance.get("thickness_grad_frac"), float("nan"))
        model = str(nuisance.get("thickness_model") or "single")
        if math.isfinite(thickness_a) and model == "uniform_gradient" and math.isfinite(grad):
            low = max(0.0, thickness_a * (1.0 - grad)) / 10.0
            high = max(0.0, thickness_a * (1.0 + grad)) / 10.0
            thickness_text = f"{thickness_a / 10.0:.2f} nm; support {low:.2f}-{high:.2f} nm"
        elif math.isfinite(thickness_a):
            thickness_text = f"{thickness_a / 10.0:.2f} nm; {model}"
        else:
            thickness_text = "n/a"
        native_weight = contract.get("native_weight") if isinstance(contract.get("native_weight"), Mapping) else {}
        a = nuisance.get("native_wght_a", native_weight.get("a"))
        b = nuisance.get("native_wght_b", native_weight.get("b"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(contract.get("dataset") or "n/a"),
                    thickness_text,
                    _quick_number(grad, 3),
                    _quick_number(nuisance.get("frame_offset"), 3),
                    f"{_quick_number(nuisance.get('group_width_deg'), 3)} deg / {nuisance.get('detector_scale_bins', 'n/a')}",
                    f"{nuisance.get('absorption_mode', 'n/a')}; frac {_quick_number(nuisance.get('absorption_frac'), 4)}",
                    str(nuisance.get("sg_jitter_mode", "n/a")),
                    _quick_number(nuisance.get("row_resolution_b"), 4),
                    str(contract.get("scale_count", "n/a")),
                    f"{_quick_number(a, 6)} / {_quick_number(b, 6)}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines
