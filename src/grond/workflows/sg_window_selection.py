"""Scout excitation-error integration windows using matched work and free row metrics."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import parse_numeric_grid, read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.workflows.decision import nuisance_score_rows, safe_float, safe_label, score_native_summary, write_json
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


def sg_window_score(record: Mapping, rows: str = "legacy") -> tuple[float, float, float]:
    """Rank SG-window candidates.

    rows="legacy" ranks by held-out (free) wR2 before all-row residuals.
    rows="work" ranks by work-row wR2 then work-row R1, so the free rows stay held out.
    """

    if rows == "work":
        work = record.get("work") or {}
        return (
            safe_float(work.get("wR2_all"), float("inf")),
            safe_float(work.get("R1_all"), float("inf")),
            0.0,
        )
    free = record.get("free") or {}
    all_rows = record.get("all") or {}
    return (
        safe_float(free.get("wR2_all"), float("inf")),
        safe_float(all_rows.get("wR2_all"), float("inf")),
        safe_float(all_rows.get("R1_all"), float("inf")),
    )


def scout_widths(grid: list[float], baseline: float) -> list[float]:
    """The scout's fixed widths: the grid plus the run's own width when the grid lacks it.

    A baseline outside the grid would otherwise be compared as its nearest grid width, and that width could be
    promoted as "baseline_width_is_best" (the AUTO default 0.010 lies outside the default 0.003-0.008 grid).
    """

    widths = [float(w) for w in grid]
    if widths and not any(abs(w - float(baseline)) <= 1e-12 for w in widths):
        widths = sorted([*widths, float(baseline)])
    return widths


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
    widths = scout_widths(parse_numeric_grid(getattr(args, "sg_window_scout_grid", "")), float(getattr(args, "sg_window", 0.004)))
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
    # Scale-fit split recorded by each successful scout stage (--production-scales fits scales on all rows).
    scale_fit_splits: list[str] = []
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
            scale_fit_splits.append(str((summary.get("config") or {}).get("scale_fit_split") or ""))
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
                    "score": score_native_summary(summary, nuisance_score_rows(args)),
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

    score_rows = nuisance_score_rows(args)
    ok = [row for row in candidates if row.get("status") == "ok"]
    current = min(ok, key=lambda r: abs(float(r["sg_window"]) - baseline_width), default=None)
    best = min(ok, key=lambda r: sg_window_score(r, score_rows), default=None)
    accepted = False
    reason = "no_successful_candidates"
    if current and best:
        same_width = abs(float(best["sg_window"]) - float(current["sg_window"])) <= 1e-12
        if score_rows == "work":
            # Rank and gate on work rows only: work-wR2 gain, no free-row penalty.
            best_work = best.get("work") or {}
            cur_work = current.get("work") or {}
            work_gain = safe_float(cur_work.get("wR2_all"), float("inf")) - safe_float(
                best_work.get("wR2_all"), float("inf")
            )
            accepted = bool(same_width or work_gain >= float(args.sg_window_scout_min_wR2_gain))
            reason = (
                "baseline_width_is_best"
                if same_width
                else ("work_row_gate_passed" if accepted else "work_row_gate_rejected_work_wR2_gain_below_minimum")
            )
        else:
            best_all = best.get("all") or {}
            cur_all = current.get("all") or {}
            best_free = best.get("free") or {}
            cur_free = current.get("free") or {}
            all_gain = safe_float(cur_all.get("wR2_all"), float("inf")) - safe_float(best_all.get("wR2_all"), float("inf"))
            free_penalty = safe_float(best_free.get("wR2_all"), float("inf")) - safe_float(
                cur_free.get("wR2_all"), float("inf")
            )
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
    summary["policy"].update(
        {
            "selection_rows": "work",
            # Work-row metrics still carry the free rows when the scales were fitted on all rows.
            "scale_fit_split": sorted(set(scale_fit_splits)),
            "free_rows_in_selection": any(split != "work" for split in scale_fit_splits),
            "rule": "rank by work-row wR2 then work-row R1; promote on a work-row wR2 gain of at least "
            "min_wR2_gain; free-row metrics are reported only (free_wR2_tolerance unused)",
        }
        if score_rows == "work"
        else {
            "selection_rows": "all",
            "free_rows_in_selection": True,
            "rule": "rank by free-row wR2 then all-row wR2 and R1; promote on an all-row wR2 gain of at least "
            "min_wR2_gain with a free-row wR2 penalty of at most free_wR2_tolerance",
        }
    )
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# Sg Rocking-Window Scout",
        "",
        (
            "This is a late diagnostic over a few fixed top-hat excitation-error widths. It uses the final pinned "
            "nuisance parameters and ranks and gates the widths on work rows only (free rows are reported, not "
            "used); it does not choose a rocking width separately for each reflection."
            if score_rows == "work"
            else "This is a late diagnostic over a few fixed top-hat excitation-error widths. It uses the final pinned nuisance parameters and held-out rows; it does not choose a rocking width separately for each reflection."
        ),
        "",
    ]
    # Under work the ranked (work-row) block is shown first; the legacy table is unchanged.
    blocks = ("work", "all", "free") if score_rows == "work" else ("all", "free")
    lines += [
        "| Sg window | status | " + " | ".join(f"{split} R1 % | {split} wR2 %" for split in blocks) + " | run |",
        "|---:|---|" + "---:|---:|" * len(blocks) + "---|",
    ]
    for row in candidates:
        cells = []
        for split in blocks:
            block = row.get(split) or {}
            cells.append(
                f"{100 * safe_float(block.get('R1_all'), float('nan')):.2f} | "
                f"{100 * safe_float(block.get('wR2_all'), float('nan')):.2f}"
            )
        lines.append(
            f"| {safe_float(row.get('sg_window'), float('nan')):.5f} | {row.get('status', '')} | "
            + " | ".join(cells)
            + f" | {row.get('run_dir', '')} |"
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
