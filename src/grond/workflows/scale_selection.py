"""Record the multi-dataset scale-contract decision: the full per-crystal scale contract is kept."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

from grond.workflows.decision import safe_float, write_json
from grond.workflows.scale_model_policy import resolved_multi_scale_contract_selection


def write_multi_scale_contract_report(out_dir: Path, decision: Mapping[str, object]) -> None:
    """Write a durable human-readable compact-scale promotion report."""

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "decision.json", dict(decision))
    lines = [
        "# GROND Multi-Dataset Scale-Contract Selection",
        "",
        "Full and compact candidates reuse identical frozen Bloch predictions. Compact scale topology is selected from bounded all/work/free row Rall costs and preserved Born-to-Bloch Rall gain. SHELXL endpoint factors and BIC are diagnostics only; coverage, NPD/split pathology, and artifact health remain safety gates.",
        "",
        f"Status: `{decision.get('status', 'unknown')}`. Reason: `{decision.get('reason', 'completed candidate gate')}`. Wall time: `{safe_float(decision.get('wall_seconds')):.3f} s`.",
        "",
        f"Selected contract: `{decision.get('selected_contract', 'full')}`.",
        "",
        "| contract | scales | aggregate BIC report | row Rall all/work/free | endpoint R1all/R1gt/wR2 report | GooF | NPD/split | accepted | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    evaluations = decision.get("evaluations") if isinstance(decision.get("evaluations"), list) else []
    candidates = decision.get("candidates") if isinstance(decision.get("candidates"), Mapping) else {}
    accepted_by_label = {str(row.get("label")): row for row in evaluations if isinstance(row, Mapping)}
    for label, candidate in candidates.items():
        if not isinstance(candidate, Mapping):
            continue
        row = accepted_by_label.get(str(label), {})
        wr = candidate.get("row_rall") or {}
        ep = candidate.get("endpoint") or {}
        lines.append(
            f"| `{label}` | {candidate.get('scale_params', '')} | {safe_float(candidate.get('aggregate_bic')):.2f} | "
            f"{safe_float(wr.get('all')):.4f}/{safe_float(wr.get('work')):.4f}/{safe_float(wr.get('free')):.4f} | "
            f"{safe_float(ep.get('R1_all')):.4f}/{safe_float(ep.get('R1_gt')):.4f}/{safe_float(ep.get('wR2_all')):.4f} | "
            f"{safe_float(ep.get('GooF')):.3f} | {ep.get('npd_count', '')}/{ep.get('split_count', '')} | "
            f"{row.get('accepted', label == 'full')} | {', '.join(str(reason) for reason in row.get('reasons', []))} |"
        )
    thresholds = decision.get("thresholds") if isinstance(decision.get("thresholds"), Mapping) else {}
    if thresholds:
        lines.extend(["", "## Promotion thresholds", ""])
        for key, value in thresholds.items():
            lines.append(f"{key}: `{value}`.")
    preflight = decision.get("preflight") if isinstance(decision.get("preflight"), Mapping) else {}
    if preflight:
        lines.extend(["", "## Candidate preflight", ""])
        for label, payload in preflight.items():
            if isinstance(payload, Mapping):
                lines.append(f"`{label}`: ok=`{payload.get('ok', False)}`; reason=`{payload.get('reason', 'passed')}`.")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_multi_scale_contract_selection(
    *,
    full_runs: Mapping[str, Path],
    full_input_runs: Mapping[str, Path],
    full_publication_runs: Mapping[str, Path],
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, object]]:
    """Write the scale-contract decision and return the full-contract input and publication runs."""

    mode, off_reason = resolved_multi_scale_contract_selection(args)
    selection_dir = root / "scale_contract_selection"
    started = time.perf_counter()
    thresholds = {
        "minimum_scale_reduction": float(getattr(args, "multi_scale_min_reduction", 0.10)),
        "require_bic_improvement": bool(getattr(args, "multi_scale_require_bic_improvement", True)),
        "maximum_row_rall_cost": float(getattr(args, "multi_scale_max_row_rall_cost", 0.01)),
        "maximum_endpoint_r1_cost": float(getattr(args, "multi_scale_max_endpoint_r1_cost", 0.005)),
        "maximum_endpoint_wr2_cost": float(getattr(args, "multi_scale_max_endpoint_wr2_cost", 0.01)),
        "goof_min": float(getattr(args, "multi_scale_goof_min", 0.85)),
        "goof_max": float(getattr(args, "multi_scale_goof_max", 1.15)),
        "maximum_input_row_loss_fraction": float(getattr(args, "multi_scale_max_input_row_loss_fraction", 0.001)),
        "minimum_bloch_rall_gain": float(getattr(args, "multi_scale_min_bloch_rall_gain", 0.002)),
        "maximum_dataset_row_rall_cost": float(getattr(args, "multi_scale_max_dataset_row_rall_cost", 0.015)),
        "maximum_preexisting_bloch_gain_loss": float(
            getattr(args, "multi_scale_max_preexisting_bloch_gain_loss", 0.01)
        ),
        "minimum_hkl_safe_factor": float(getattr(args, "publication_endpoint_min_hkl_safe_factor", 0.25)),
        "maximum_modulation_p95": float(getattr(args, "publication_endpoint_max_modulation_p95", 20.0)),
    }
    decision: dict[str, object] = {
        "status": "disabled" if mode == "off" else "not_applicable",
        "selected_contract": "full",
        "accepted": False,
        "candidates": {},
        "evaluations": [],
        "thresholds": thresholds,
    }
    full_input = {str(name): Path(path) for name, path in full_input_runs.items()}
    full_publication = {str(name): Path(path) for name, path in full_publication_runs.items()}
    decision["full_input_runs"] = {name: str(path) for name, path in full_input.items()}
    decision["full_publication_runs"] = {name: str(path) for name, path in full_publication.items()}

    def finish(
        selected_input: dict[str, Path],
        selected_publication: dict[str, Path],
    ) -> tuple[dict[str, Path], dict[str, Path], dict[str, object]]:
        """Record the wall time and artifact directory, write the report, and return the selected runs and decision."""
        decision["wall_seconds"] = time.perf_counter() - started
        decision["artifact_dir"] = str(selection_dir.resolve())
        write_multi_scale_contract_report(selection_dir, decision)
        return selected_input, selected_publication, decision

    if mode == "off" or args.dry_run or len(full_runs) < 2:
        decision["reason"] = (off_reason or "disabled_by_option") if mode == "off" else "dry_run_or_single_dataset"
        return finish(full_input, full_publication)

    # Compact contracts were replays of scale-window polish runs, which GROND no longer writes: the full contract stays.
    decision["reason"] = "no_compact_scale_runs"
    return finish(full_input, full_publication)
