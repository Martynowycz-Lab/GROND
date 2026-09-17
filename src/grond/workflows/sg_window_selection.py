"""Scout excitation-error integration windows using matched work and free row metrics."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import parse_numeric_grid, read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.workflows.decision import safe_float, safe_label, score_native_summary, write_json
from grond.workflows.model_selection import best_nuisance, dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd


def native_dyn_metrics(summary: Mapping) -> dict:
    """Return all/work/free dynamical metrics from a native-stage summary."""

    best = summary.get("best_native") if isinstance(summary, Mapping) else {}
    metrics = best.get("metrics") if isinstance(best, Mapping) else {}
    out = {}
    for split in ("all", "work", "free"):
        out[split] = ((metrics.get(split) or {}).get("dyn") or {}) if isinstance(metrics, Mapping) else {}
    return out


def sg_window_score(record: Mapping) -> tuple[float, float, float]:
    """Rank SG-window candidates by held-out behavior before work/all residuals."""

    free = record.get("free") or {}
    all_rows = record.get("all") or {}
    return (
        safe_float(free.get("wR2_all"), float("inf")),
        safe_float(all_rows.get("wR2_all"), float("inf")),
        safe_float(all_rows.get("R1_all"), float("inf")),
    )


def run_sg_window_scout(
    python: str,
    prep: PrepConfig,
    ds: DatasetSpec,
    structure: str,
    run_dir: Path,
    out_dir: Path,
    logs: Path,
    label: str,
    args: argparse.Namespace,
) -> dict:
    """Run a late, held-out scout over simple fixed Sg rocking-window widths."""

    from grond.workflows.crystal_parameters import crystal_arguments

    args = crystal_arguments(args, ds.name)
    mode = getattr(args, "sg_window_scout", "auto")
    if mode == "off" or getattr(args, "fast", False):
        return {"status": "disabled"}
    widths = parse_numeric_grid(getattr(args, "sg_window_scout_grid", ""))
    if not widths:
        return {"status": "empty_grid"}
    pinned = best_nuisance(run_dir)
    if not pinned:
        return {"status": "missing_nuisance", "run_dir": str(run_dir)}
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "widths": widths}

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_width = float(getattr(args, "sg_window", 0.004))
    candidates: list[dict] = []
    for width in widths:
        tag = str(width).replace("-", "m").replace(".", "p")
        trial_dir = out_dir / f"fixed_sg_{tag}"
        cmd = native_stage_cmd(
            python,
            ds,
            structure,
            dataset_baseline_hkl(prep, ds),
            trial_dir,
            args,
            n_basis=int(args.sg_window_scout_n_basis),
            n_points=int(args.sg_window_scout_n_points),
            stage=f"sg_window_scout_{tag}",
            pinned=pinned,
            make_maps=False,
            endpoint=False,
            max_rows_override=int(args.sg_window_scout_max_rows) or None,
            sg_window_override=width,
            sg_window_mode_override="fixed",
        )
        try:
            run_logged(cmd, logs, f"{safe_label(label)}_sg_window_scout_{tag}", dry_run=False)
            summary = read_json(trial_dir / "summary.json")
            metrics = native_dyn_metrics(summary)
            candidates.append(
                {
                    "status": "ok",
                    "sg_window": float(width),
                    "run_dir": str(trial_dir),
                    "n_basis": int(args.sg_window_scout_n_basis),
                    "n_points": int(args.sg_window_scout_n_points),
                    "all": metrics["all"],
                    "work": metrics["work"],
                    "free": metrics["free"],
                    "score": score_native_summary(summary),
                }
            )
        except subprocess.CalledProcessError as exc:
            if mode == "on":
                raise
            candidates.append(
                {
                    "status": "failed",
                    "sg_window": float(width),
                    "run_dir": str(trial_dir),
                    "returncode": exc.returncode,
                    "command": exc.cmd,
                }
            )

    ok = [row for row in candidates if row.get("status") == "ok"]
    current = min(ok, key=lambda r: abs(float(r["sg_window"]) - baseline_width), default=None)
    best = min(ok, key=sg_window_score, default=None)
    accepted = False
    reason = "no_successful_candidates"
    if current and best:
        best_all = best.get("all") or {}
        cur_all = current.get("all") or {}
        best_free = best.get("free") or {}
        cur_free = current.get("free") or {}
        all_gain = safe_float(cur_all.get("wR2_all"), float("inf")) - safe_float(best_all.get("wR2_all"), float("inf"))
        free_penalty = safe_float(best_free.get("wR2_all"), float("inf")) - safe_float(
            cur_free.get("wR2_all"), float("inf")
        )
        same_width = abs(float(best["sg_window"]) - float(current["sg_window"])) <= 1e-12
        accepted = bool(
            same_width
            or (
                all_gain >= float(args.sg_window_scout_min_wR2_gain)
                and free_penalty <= float(args.sg_window_scout_free_wR2_tolerance)
            )
        )
        reason = (
            "baseline_width_is_best"
            if same_width
            else ("held_out_gate_passed" if accepted else "held_out_gate_rejected_lower_work_residual_or_free_penalty")
        )
    summary = {
        "status": "accepted" if accepted else "diagnostic_only",
        "reason": reason,
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "baseline_sg_window": baseline_width,
        "current": current or {},
        "selected": best or {},
        "candidates": candidates,
        "policy": {
            "min_wR2_gain": float(args.sg_window_scout_min_wR2_gain),
            "free_wR2_tolerance": float(args.sg_window_scout_free_wR2_tolerance),
            "note": "This scout only promotes a global fixed rocking-window width. Per-row window choice from Iobs is deliberately not implemented.",
        },
    }
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# Sg Rocking-Window Scout",
        "",
        "This is a late diagnostic over a few fixed top-hat excitation-error widths. It uses the final pinned nuisance parameters and held-out rows; it does not choose a rocking width separately for each reflection.",
        "",
        "| Sg window | status | all R1 % | all wR2 % | free R1 % | free wR2 % | run |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        all_rows = row.get("all") or {}
        free = row.get("free") or {}
        lines.append(
            f"| {safe_float(row.get('sg_window'), float('nan')):.5f} | {row.get('status', '')} | "
            f"{100 * safe_float(all_rows.get('R1_all'), float('nan')):.2f} | "
            f"{100 * safe_float(all_rows.get('wR2_all'), float('nan')):.2f} | "
            f"{100 * safe_float(free.get('R1_all'), float('nan')):.2f} | "
            f"{100 * safe_float(free.get('wR2_all'), float('nan')):.2f} | {row.get('run_dir', '')} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{summary['status']}`. Reason: `{reason}`.",
            "Per-reflection post-hoc rocking-window selection is not part of GROND because it would fit the observed intensity unless the window is supplied by independent raw-image profile metadata.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(out_dir / "summary.json", summary)
    return summary
