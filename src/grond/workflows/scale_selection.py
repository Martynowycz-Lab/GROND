"""Compare multi-dataset scale candidates and verify that their physical rows still match."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.workflows.dataset_gates import policy_active_runs, require_policy_active_runs
from grond.workflows.dataset_policy import write_policy_weighted_runs
from grond.workflows.decision import safe_float, safe_label, write_json
from grond.workflows.report_commands import combine_cmd
from grond.workflows.scale_contract_selector import select_multi_scale_contract
from grond.workflows.scale_model_policy import resolved_multi_scale_contract_selection

_SCALE_CONTRACT_MUTABLE_COLUMNS = {
    "dyn_scaled",
    "born_scaled",
    "dyn_full_scaled",
    "born_full_scaled",
    "dyn_replay_source_scaled",
    "born_replay_source_scaled",
    "scale_compression_model",
    "scale_replay_model",
    "scale_replay_group",
    "scale_replay_detector_bin",
}


def validate_scale_contract_run_map(
    full_runs: Mapping[str, Path],
    candidate_runs: Mapping[str, Path],
    *,
    min_bloch_rall_gain: float = 0.0,
    max_dataset_row_rall_cost: float = 0.015,
    max_preexisting_bloch_gain_loss: float = 0.01,
) -> dict:
    """Prove that compact tables changed only fitted scale predictions."""

    if set(full_runs) != set(candidate_runs):
        return {"ok": False, "reason": "dataset_set_mismatch"}
    datasets: dict[str, dict[str, object]] = {}
    for name in sorted(full_runs):
        full_path = Path(full_runs[name]) / "native_dyn_calc.csv"
        candidate_path = Path(candidate_runs[name]) / "native_dyn_calc.csv"
        if not full_path.exists() or not candidate_path.exists():
            return {"ok": False, "reason": "missing_row_table", "dataset": name}
        with full_path.open(newline="") as handle:
            full_reader = csv.DictReader(handle)
            full_columns = list(full_reader.fieldnames or [])
            full_rows = list(full_reader)
        with candidate_path.open(newline="") as handle:
            candidate_reader = csv.DictReader(handle)
            candidate_columns = list(candidate_reader.fieldnames or [])
            candidate_rows = list(candidate_reader)
        unknown_columns = sorted(set(candidate_columns) - set(full_columns) - _SCALE_CONTRACT_MUTABLE_COLUMNS)
        if unknown_columns:
            return {
                "ok": False,
                "reason": "unknown_candidate_columns",
                "dataset": name,
                "columns": unknown_columns,
            }
        if len(full_rows) != len(candidate_rows):
            return {
                "ok": False,
                "reason": "row_count_mismatch",
                "dataset": name,
                "full_rows": len(full_rows),
                "candidate_rows": len(candidate_rows),
            }
        immutable = [column for column in full_columns if column not in _SCALE_CONTRACT_MUTABLE_COLUMNS]
        for index, (full_row, candidate_row) in enumerate(zip(full_rows, candidate_rows, strict=True)):
            missing = [column for column in immutable if column not in candidate_row]
            changed = [column for column in immutable if candidate_row.get(column) != full_row.get(column)]
            if missing or changed:
                return {
                    "ok": False,
                    "reason": "immutable_row_contract_mismatch",
                    "dataset": name,
                    "row_index": index,
                    "missing_columns": missing,
                    "changed_columns": changed[:20],
                }
        full_summary = try_read_json(Path(full_runs[name]) / "summary.json")
        summary = try_read_json(Path(candidate_runs[name]) / "summary.json")
        best_native = summary.get("best_native") or {}
        metrics = best_native.get("metrics") or {}
        full_metrics = (full_summary.get("best_native") or {}).get("metrics") or {}
        if any(
            not isinstance(metrics.get(split), Mapping) or not {"dyn", "born"}.issubset(metrics[split])
            for split in ("all", "work", "free")
        ):
            return {"ok": False, "reason": "incomplete_split_metrics", "dataset": name}
        if any(
            not isinstance(full_metrics.get(split), Mapping) or not {"dyn", "born"}.issubset(full_metrics[split])
            for split in ("all", "work", "free")
        ):
            return {"ok": False, "reason": "incomplete_full_split_metrics", "dataset": name}
        bloch_gains: dict[str, float] = {}
        full_bloch_gains: dict[str, float] = {}
        row_rall_costs: dict[str, float] = {}
        for split in ("all", "work", "free"):
            dyn_rall = safe_float((metrics[split].get("dyn") or {}).get("R1_all"), float("nan"))
            born_rall = safe_float((metrics[split].get("born") or {}).get("R1_all"), float("nan"))
            full_dyn_rall = safe_float((full_metrics[split].get("dyn") or {}).get("R1_all"), float("nan"))
            full_born_rall = safe_float((full_metrics[split].get("born") or {}).get("R1_all"), float("nan"))
            gain = born_rall - dyn_rall
            full_gain = full_born_rall - full_dyn_rall
            row_cost = dyn_rall - full_dyn_rall
            if not all(math.isfinite(value) for value in (gain, full_gain, row_cost)):
                return {"ok": False, "reason": "nonfinite_split_metrics", "dataset": name, "split": split}
            if row_cost > float(max_dataset_row_rall_cost):
                return {
                    "ok": False,
                    "reason": "dataset_row_rall_cost_exceeded",
                    "dataset": name,
                    "split": split,
                    "cost": row_cost,
                    "maximum": float(max_dataset_row_rall_cost),
                }
            if full_gain >= float(min_bloch_rall_gain) and gain < float(min_bloch_rall_gain):
                return {
                    "ok": False,
                    "reason": "bloch_rall_gain_below_minimum",
                    "dataset": name,
                    "split": split,
                    "full_gain": full_gain,
                    "gain": gain,
                    "minimum": float(min_bloch_rall_gain),
                }
            if full_gain < float(min_bloch_rall_gain) and gain < full_gain - float(max_preexisting_bloch_gain_loss):
                return {
                    "ok": False,
                    "reason": "preexisting_bloch_gain_loss_exceeded",
                    "dataset": name,
                    "split": split,
                    "full_gain": full_gain,
                    "gain": gain,
                    "loss": full_gain - gain,
                    "maximum": float(max_preexisting_bloch_gain_loss),
                }
            bloch_gains[split] = gain
            full_bloch_gains[split] = full_gain
            row_rall_costs[split] = row_cost
        try:
            scale_count = int(best_native["scale_count"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"ok": False, "reason": "missing_materialized_scale_count", "dataset": name}
        if scale_count <= 0:
            return {"ok": False, "reason": "invalid_materialized_scale_count", "dataset": name}
        datasets[name] = {
            "rows": len(full_rows),
            "immutable_columns": immutable,
            "scale_count": scale_count,
            "bloch_rall_gain": bloch_gains,
            "full_bloch_rall_gain": full_bloch_gains,
            "row_rall_cost": row_rall_costs,
        }
    return {"ok": True, "datasets": datasets}


def scale_contract_totals(scale_polish: Mapping[str, Mapping], selection_key: str) -> tuple[int, float] | None:
    """Sum materialized scale count and fitted-variance BIC across datasets."""

    scales = 0
    bic = 0.0
    for payload in scale_polish.values():
        selection = payload.get(selection_key)
        if not isinstance(selection, Mapping):
            return None
        scale_value = safe_float(selection.get("scale_params"), float("nan"))
        bic_value = safe_float((selection.get("information") or {}).get("bic_fitted_variance"), float("nan"))
        if not math.isfinite(scale_value) or not math.isfinite(bic_value):
            return None
        scales += int(round(scale_value))
        bic += bic_value
    return scales, bic


def combined_scale_contract_candidate(
    label: str, combined_dir: Path, *, scale_params: int, aggregate_bic: float
) -> dict:
    """Convert one combined replay summary into the pure selector contract."""

    summary = try_read_json(combined_dir / "combined_summary.json")
    native = summary.get("native_joint") or {}
    endpoint = summary.get("combined_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    endpoint_artifacts = endpoint.get("artifacts") or {}
    endpoint_health = shelxl.get("res_health") or {}
    dataset_sigma = endpoint.get("dataset_sigma") or {}
    physics_contract = {
        "lambda_q": endpoint.get("lambda_q"),
        "min_d_spacing_A": endpoint.get("min_d_spacing_A", 0.0),
        "endpoint_b": endpoint.get("endpoint_b"),
        "keep_exti": endpoint.get("keep_exti"),
        "merge_key": endpoint.get("merge_key"),
        "interdataset_scale": (summary.get("interdataset_scale") or {}).get("mode"),
        "dataset_sigma_policy": sorted((name, (meta or {}).get("policy")) for name, meta in dataset_sigma.items()),
        "merge_sigma_policy": (endpoint.get("merge_sigma") or {}).get("policy"),
        "post_bloch_sigma_policy": (endpoint.get("post_bloch_sigma") or {}).get("policy"),
        "endpoint_wght_mode": endpoint.get("endpoint_wght_mode"),
        "endpoint_ls_cycles": endpoint.get("endpoint_ls_cycles"),
        "datasets": sorted(
            (
                row.get("name"),
                row.get("dataset_role"),
                row.get("raw_row_weight"),
                row.get("endpoint_weight"),
            )
            for row in summary.get("runs", [])
        ),
    }
    return {
        "label": label,
        "directory": str(combined_dir.resolve()),
        "scale_params": int(scale_params),
        "aggregate_bic": float(aggregate_bic),
        "input_endpoint_rows": endpoint.get("input_unmerged_rows"),
        "row_rall": {
            split: ((native.get("dyn") or {}).get(split) or {}).get("R1_all") for split in ("all", "work", "free")
        },
        "bloch_gain_rall": {
            split: safe_float(((native.get("born") or {}).get(split) or {}).get("R1_all"))
            - safe_float(((native.get("dyn") or {}).get(split) or {}).get("R1_all"))
            for split in ("all", "work", "free")
        },
        "endpoint": {
            "R1_all": shelxl.get("R1_all"),
            "R1_gt": shelxl.get("R1_gt"),
            "wR2_all": shelxl.get("wR2_all"),
            "GooF": shelxl.get("GooF"),
            "n_reflections": endpoint.get("merged_endpoint_rows", endpoint.get("merged_exact_hkl_rows")),
            "npd_count": shelxl.get("npd_count", 0),
            "split_count": shelxl.get("split_count", 0),
            "returncode": shelxl.get("returncode"),
            "status": shelxl.get("status"),
            "safe_for_feedback": endpoint_health.get("safe_for_feedback") is True,
            "hkl_safe_factor": endpoint.get("hkl_safe_factor"),
            "modulation_p95": endpoint.get("mod_p95", endpoint.get("modulation_p95")),
            "artifacts_exist": bool(endpoint_artifacts)
            and all(
                str(endpoint_artifacts.get(key, "")).strip() and Path(str(endpoint_artifacts[key])).is_file()
                for key in ("res", "lst", "hkl")
            ),
            "artifacts": dict(endpoint_artifacts),
        },
        "physics_contract": physics_contract,
    }


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


def _multi_scale_candidate_runs(
    scale_polish: Mapping[str, Mapping],
    *,
    run_key: str,
) -> tuple[dict[str, Path], str]:
    """Return an all-or-none map of strict materialized compact runs."""

    runs: dict[str, Path] = {}
    for name, payload in scale_polish.items():
        if str(payload.get("status", "")) != "accepted":
            return {}, f"{name}:scale_polish_not_strictly_accepted"
        raw_path = str(payload.get(run_key, "")).strip()
        if not raw_path:
            return {}, f"{name}:missing_{run_key}"
        run_dir = Path(raw_path)
        if not (run_dir / "native_dyn_calc.csv").is_file() or not (run_dir / "summary.json").is_file():
            return {}, f"{name}:incomplete_{run_key}"
        runs[str(name)] = run_dir
    return runs, ""


def _policy_weight_scale_contract_runs(
    runs: Mapping[str, Path],
    policy: Mapping[str, object],
    out_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Apply the already-decided final dataset policy to one scale contract."""

    status = str(policy.get("status", ""))
    if status in {"disabled", "single_dataset", "dry_run", "missing_row_table"}:
        copied = {str(name): Path(path) for name, path in runs.items()}
        return copied, copied
    weighted = write_policy_weighted_runs(
        {str(name): Path(path) for name, path in runs.items()},
        dict(policy),
        out_dir,
        include_endpoint_only=True,
    )
    if not weighted:
        raise RuntimeError("multi-dataset scale contract has no endpoint-contributing dataset after final policy")
    publication = policy_active_runs(weighted, dict(policy))
    require_policy_active_runs(publication, context="compact-scale publication")
    return weighted, publication


def write_selected_scale_polish_summaries(
    scale_polish: Mapping[str, Mapping],
    *,
    selected_contract: str,
    selected_publication_runs: Mapping[str, Path],
    out_dir: Path,
) -> dict[str, str]:
    """Project per-dataset scale summaries onto the globally selected rows."""

    if selected_contract not in {"compact_bic", "compact_min_strict"}:
        return {}
    selection_key = "selected_output" if selected_contract == "compact_bic" else "selected"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, selected_run in selected_publication_runs.items():
        payload = scale_polish.get(name)
        if not isinstance(payload, Mapping):
            continue
        source_path = Path(str(payload.get("out_dir", ""))) / "scale_compression_summary.json"
        source = try_read_json(source_path)
        selected = payload.get(selection_key)
        if not source or not isinstance(selected, Mapping):
            continue
        projected = dict(source)
        projected["selected_output"] = dict(selected)
        projected["selected_run_dir"] = str(Path(selected_run).resolve())
        projected["publication_scale_contract"] = {
            "selected_contract": selected_contract,
            "row_source": str(Path(selected_run).resolve()),
            "source_summary": str(source_path.resolve()),
        }
        path = out_dir / f"{safe_label(name)}.json"
        write_json(path, projected)
        written[str(name)] = str(path.resolve())
    return written


def run_multi_scale_contract_selection(
    *,
    python: str,
    full_runs: Mapping[str, Path],
    full_input_runs: Mapping[str, Path],
    full_publication_runs: Mapping[str, Path],
    scale_polish: Mapping[str, Mapping],
    final_dataset_policy: Mapping[str, object],
    final_model: str,
    root: Path,
    logs: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, object]]:
    """Replay and select the smallest earned multi-dataset scale contract.

    The expensive Bloch predictions remain frozen. Every candidate is passed
    through the same dataset policy, sigma calibration, overlap scaling,
    symmetry merge, and fixed-weight SHELXL endpoint as the full-scale control.
    Any incomplete provenance or failed gate leaves the full contract selected.
    """

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

    endpoint_names = set(full_input)
    if not endpoint_names or not endpoint_names.issubset(full_runs) or not endpoint_names.issubset(scale_polish):
        decision.update(status="failed_closed", reason="endpoint_contributor_contract_incomplete")
        return finish(full_input, full_publication)
    selection_full_runs = {name: Path(full_runs[name]) for name in sorted(endpoint_names)}
    selection_scale_polish = {name: scale_polish[name] for name in sorted(endpoint_names)}
    full_totals = scale_contract_totals(selection_scale_polish, "baseline")
    if full_totals is None:
        decision.update(status="failed_closed", reason="incomplete_full_scale_information_contract")
        return finish(full_input, full_publication)

    specifications = (
        ("compact_bic", "selected_compact_run", "selected_output"),
        ("compact_min_strict", "selected_strict_run", "selected"),
    )
    preflight: dict[str, dict[str, object]] = {}
    min_bloch_gain = float(thresholds["minimum_bloch_rall_gain"])
    for label, run_key, selection_key in specifications:
        try:
            candidate_runs, reason = _multi_scale_candidate_runs(
                selection_scale_polish,
                run_key=run_key,
            )
            if reason:
                preflight[label] = {"ok": False, "reason": reason}
                continue
            validation = validate_scale_contract_run_map(
                selection_full_runs,
                candidate_runs,
                min_bloch_rall_gain=min_bloch_gain,
                max_dataset_row_rall_cost=float(thresholds["maximum_dataset_row_rall_cost"]),
                max_preexisting_bloch_gain_loss=float(thresholds["maximum_preexisting_bloch_gain_loss"]),
            )
            totals = scale_contract_totals(selection_scale_polish, selection_key)
            if not validation.get("ok"):
                preflight[label] = {
                    "ok": False,
                    "reason": "row_contract_validation_failed",
                    "validation": validation,
                }
                continue
            if totals is None:
                preflight[label] = {
                    "ok": False,
                    "reason": "incomplete_scale_information_contract",
                }
                continue
            materialized_total = sum(
                int((row or {}).get("scale_count", 0)) for row in (validation.get("datasets") or {}).values()
            )
            if materialized_total != int(totals[0]):
                preflight[label] = {
                    "ok": False,
                    "reason": "materialized_scale_count_mismatch",
                    "reported": int(totals[0]),
                    "materialized": materialized_total,
                }
                continue
            preflight[label] = {
                "ok": True,
                "runs": {name: str(path) for name, path in candidate_runs.items()},
                "scale_params": int(totals[0]),
                "aggregate_bic": float(totals[1]),
                "validation": validation,
            }
        except Exception as exc:
            preflight[label] = {
                "ok": False,
                "reason": "candidate_preflight_failed",
                "failure_stage": "preflight",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    viable = {label: payload for label, payload in preflight.items() if payload.get("ok")}
    if not viable:
        decision.update(status="failed_closed", reason="no_complete_strict_compact_contract", preflight=preflight)
        return finish(full_input, full_publication)

    controls_root = root / "runs" / "combined_scale_contracts"
    full_combined = controls_root / "full"
    run_logged(
        combine_cmd(python, full_input, final_model, full_combined, args, final_outputs=False),
        logs,
        "combine_scale_contract_full",
        dry_run=False,
    )
    full_candidate = combined_scale_contract_candidate(
        "full",
        full_combined,
        scale_params=int(full_totals[0]),
        aggregate_bic=float(full_totals[1]),
    )
    candidates: dict[str, dict[str, object]] = {"full": full_candidate}
    evaluations: list[dict[str, object]] = []
    selected_payloads: dict[str, tuple[dict[str, Path], dict[str, Path]]] = {}

    def reject_candidate(label: str, *, reason: str, stage: str, exc: Exception) -> None:
        """Record a failed compact candidate and its evaluation, which keeps the full contract."""
        failure = {
            "failure_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        candidates[label] = {
            "label": label,
            "status": "failed",
            "failure": failure,
        }
        evaluations.append(
            {
                "label": label,
                "selected_contract": "full",
                "accepted": False,
                "reasons": [reason],
                "diagnostics": failure,
                "validation": viable[label]["validation"],
            }
        )

    for label, payload in viable.items():
        try:
            candidate_input, candidate_publication = _policy_weight_scale_contract_runs(
                payload["runs"],
                final_dataset_policy,
                root / "dataset_policy" / "final" / f"weighted_runs_{label}",
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_policy_weighting_failed",
                stage="policy_weighting",
                exc=exc,
            )
            continue
        combined_dir = controls_root / label
        try:
            run_logged(
                combine_cmd(python, candidate_input, final_model, combined_dir, args, final_outputs=False),
                logs,
                f"combine_scale_contract_{label}",
                dry_run=False,
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_replay_failed",
                stage="combined_replay",
                exc=exc,
            )
            continue
        try:
            candidate = combined_scale_contract_candidate(
                label,
                combined_dir,
                scale_params=int(payload["scale_params"]),
                aggregate_bic=float(payload["aggregate_bic"]),
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_summary_parsing_failed",
                stage="summary_parsing",
                exc=exc,
            )
            continue
        candidates[label] = candidate
        try:
            evaluation = select_multi_scale_contract(
                full_candidate,
                candidate,
                min_scale_reduction=float(getattr(args, "multi_scale_min_reduction", 0.10)),
                require_bic_improvement=bool(getattr(args, "multi_scale_require_bic_improvement", True)),
                max_row_rall_cost=float(getattr(args, "multi_scale_max_row_rall_cost", 0.01)),
                max_endpoint_r1_cost=float(getattr(args, "multi_scale_max_endpoint_r1_cost", 0.005)),
                max_endpoint_wr2_cost=float(getattr(args, "multi_scale_max_endpoint_wr2_cost", 0.01)),
                goof_min=float(getattr(args, "multi_scale_goof_min", 0.85)),
                goof_max=float(getattr(args, "multi_scale_goof_max", 1.15)),
                max_input_row_loss_fraction=float(getattr(args, "multi_scale_max_input_row_loss_fraction", 0.001)),
                min_bloch_rall_gain=min_bloch_gain,
                min_hkl_safe_factor=float(getattr(args, "publication_endpoint_min_hkl_safe_factor", 0.25)),
                max_modulation_p95=float(getattr(args, "publication_endpoint_max_modulation_p95", 20.0)),
                require_endpoint_health=True,
            )
        except Exception as exc:
            reject_candidate(
                label,
                reason="candidate_evaluation_failed",
                stage="selection_gate",
                exc=exc,
            )
            continue
        evaluation = {**evaluation, "label": label, "validation": payload["validation"]}
        evaluations.append(evaluation)
        selected_payloads[label] = (candidate_input, candidate_publication)

    accepted_labels = [
        str(row["label"])
        for row in evaluations
        if bool(row.get("accepted")) and str(row.get("label")) in selected_payloads
    ]
    if accepted_labels:
        selected_label = min(
            accepted_labels,
            key=lambda label: (
                int(candidates[label]["scale_params"]),
                safe_float((candidates[label].get("row_rall") or {}).get("free"), float("inf")),
                safe_float((candidates[label].get("row_rall") or {}).get("work"), float("inf")),
            ),
        )
        selected_input, selected_publication = selected_payloads[selected_label]
        accepted = True
        status = "selected_compact"
    else:
        selected_label = "full"
        selected_input, selected_publication = full_input, full_publication
        accepted = False
        status = "kept_full"
    publication_scale_polish_summaries = write_selected_scale_polish_summaries(
        selection_scale_polish,
        selected_contract=selected_label,
        selected_publication_runs=selected_publication,
        out_dir=selection_dir / "publication_scale_polish",
    )
    decision.update(
        status=status,
        selected_contract=selected_label,
        accepted=accepted,
        candidates=candidates,
        evaluations=evaluations,
        preflight=preflight,
        selection_policy="smallest accepted scale count, then held-out/work row Rall",
        selected_input_runs={name: str(path) for name, path in selected_input.items()},
        selected_publication_runs={name: str(path) for name, path in selected_publication.items()},
        publication_scale_polish_summaries=publication_scale_polish_summaries,
    )
    return finish(selected_input, selected_publication)
