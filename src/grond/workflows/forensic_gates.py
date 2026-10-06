"""Run optional residual, correction and resolution audits and retain their decisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.tools import default_shelxl_executable
from grond.cli_support.values import FULL_Q_LAMBDA, try_read_json
from grond.paths import engine_script
from grond.workflows.decision import safe_float
from grond.workflows.endpoint_commands import legacy_handoff_conflicts
from grond.workflows.report_commands import resolution_limit_cmd


def forensic_command_plan(
    python: str,
    row_table: Path,
    hkl: str,
    res: str,
    out_dir: Path,
    *,
    depth: str,
    bins: int,
    top_rows: int,
    top_groups: int,
) -> list[tuple[str, list[str]]]:
    """Build the row-space forensic command list for a completed row table."""

    common = ["--row-table", str(row_table), "--hkl", hkl, "--res", res]
    commands: list[tuple[str, list[str]]] = [
        (
            "residual_compass",
            [
                python,
                str(engine_script("residual_compass.py")),
                *common,
                "--out-dir",
                str(out_dir / "residual_compass"),
                "--bins",
                str(bins),
                "--top-rows",
                str(top_rows),
            ],
        ),
        (
            "premerge_rfactor",
            [
                python,
                str(engine_script("premerge_rfactor_audit.py")),
                *common,
                "--out-dir",
                str(out_dir / "premerge_rfactor"),
                "--bins",
                str(bins),
                "--top-groups",
                str(top_groups),
            ],
        ),
    ]
    if depth in {"standard", "full"}:
        commands.extend(
            [
                (
                    "dose_damage",
                    [
                        python,
                        str(engine_script("native_dose_damage_audit.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / "dose_damage"),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                    ],
                ),
                (
                    "two_part_signed",
                    [
                        python,
                        str(engine_script("native_two_part_observation_correction.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / "two_part_signed"),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                        "--holdout-mode",
                        "signed_hkl",
                        "--holdout-mod",
                        "10",
                        "--holdout-value",
                        "0",
                        "--frame-basis",
                        "6",
                        "--xy-basis",
                        "4",
                        "--alternating-iters",
                        "2",
                    ],
                ),
                (
                    "two_part_frame_mod",
                    [
                        python,
                        str(engine_script("native_two_part_observation_correction.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / "two_part_frame_mod"),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                        "--holdout-mode",
                        "frame_mod",
                        "--holdout-mod",
                        "5",
                        "--holdout-value",
                        "2",
                        "--frame-holdout-width",
                        "30",
                        "--frame-basis",
                        "6",
                        "--xy-basis",
                        "4",
                        "--alternating-iters",
                        "2",
                    ],
                ),
            ]
        )
    if depth == "full":
        for label, value in (("two_part_frame_block_early", "0"), ("two_part_frame_block_late", "9")):
            commands.append(
                (
                    label,
                    [
                        python,
                        str(engine_script("native_two_part_observation_correction.py")),
                        *common,
                        "--out-dir",
                        str(out_dir / label),
                        "--pred-column",
                        "dyn_scaled",
                        "--sigma-column",
                        "sigma_eff",
                        "--holdout-mode",
                        "frame_block",
                        "--holdout-mod",
                        "10",
                        "--holdout-value",
                        value,
                        "--frame-basis",
                        "6",
                        "--xy-basis",
                        "4",
                        "--alternating-iters",
                        "2",
                    ],
                )
            )
        commands.append(
            (
                "sigma_replicate",
                [
                    python,
                    str(engine_script("native_sigma_replicate_audit.py")),
                    *common,
                    "--out-dir",
                    str(out_dir / "sigma_replicate"),
                    "--pred-column",
                    "dyn_scaled",
                    "--base-sigma-column",
                    "xds_sigma",
                ],
            )
        )
    return commands


def summarize_forensic_gate(out_dir: Path, row_table: Path, hkl: str, res: str, depth: str) -> dict:
    """Summarize forensic outputs and decide what they recommend."""

    residual = try_read_json(out_dir / "residual_compass" / "summary.json")
    premerge = try_read_json(out_dir / "premerge_rfactor" / "summary.json")
    dose = try_read_json(out_dir / "dose_damage" / "summary.json")
    signed = try_read_json(out_dir / "two_part_signed" / "summary.json")
    frame_mod = try_read_json(out_dir / "two_part_frame_mod" / "summary.json")
    early = try_read_json(out_dir / "two_part_frame_block_early" / "summary.json")
    late = try_read_json(out_dir / "two_part_frame_block_late" / "summary.json")
    observation_polish = try_read_json(out_dir / "observation_polish" / "summary.json")
    recommendations: list[str] = []
    dyn_all = ((residual.get("model_summary") or {}).get("dyn") or {}).get("all") or {}
    dyn_strong_gap = safe_float(dyn_all.get("R1_all"), 0.0) - safe_float(dyn_all.get("R1_gt"), 0.0)
    if dyn_strong_gap > 0.04:
        recommendations.append(
            "R1all is much worse than R1strong; branch to weak-row, resolution, and observation-correction diagnostics before adding beams"
        )
    scale_pressure = (((residual.get("raw_scale_summary") or {}).get("dyn") or {}).get("scale_pressure")) or {}
    frame_corr = safe_float((scale_pressure.get("log_scale_correlations") or {}).get("frame_zcal"), 0.0)
    if abs(frame_corr) > 0.4:
        recommendations.append(
            "row scale is strongly frame-correlated; promote explicit frame nuisance/background/fluence checks"
        )
    pmode = premerge.get("primary_group_mode")
    psummary = (premerge.get("group_modes") or {}).get(pmode or "") or {}
    if psummary:
        scalar = psummary.get("group_floor") or {}
        dyn = psummary.get("dyn") or {}
        if safe_float(scalar.get("R2_all"), 0.0) > safe_float(dyn.get("R2_all"), 0.0) + 0.02:
            recommendations.append(
                "row-specific Bloch predictions beat scalar equivalent-row collapse; keep pre-merge row modeling"
            )
    for item in premerge.get("recommendations") or []:
        if "equivalent-row modulation is anti-correlated" in item and item not in recommendations:
            recommendations.append(item)
    if dose:
        accepted = bool(dose.get("accept_dose_b_grid", False))
        if not accepted:
            recommendations.append(
                "dose-B gate did not pass; keep dose terms off unless a later residual table shows a significant slope"
            )
    signed_pass = bool(signed.get("passes_heldout_gate", False))
    frame_pass = bool(frame_mod.get("passes_heldout_gate", False))
    if signed_pass and frame_pass:
        recommendations.append(
            "smooth additive observation correction passed signed-HKL and interleaved-frame gates; keep as candidate final polish"
        )
    elif signed or frame_mod:
        recommendations.append("observation correction did not pass both main holdouts; keep diagnostic-only")
    if (
        early
        and late
        and (not bool(early.get("passes_heldout_gate", False)) or not bool(late.get("passes_heldout_gate", False)))
    ):
        recommendations.append(
            "edge-frame holdout is not uniformly stable; require stronger regularization before default endpoint promotion"
        )
    if observation_polish:
        if bool(observation_polish.get("accepted", False)):
            endpoint = observation_polish.get("best_endpoint") or {}
            sh = endpoint.get("shelxl") or {}
            recommendations.append(
                "observation polish passed raw-row and SHELXL endpoint gates; promote polished corrected-HKL endpoint"
            )
            if sh:
                lam = safe_float(endpoint.get("lambda_q"), 1.0)
                q_note = f" qexp={lam:.3g}," if abs(lam - 1.0) > 1e-9 else ""
                recommendations.append(
                    f"best polished endpoint:{q_note} R1all={100.0 * safe_float(sh.get('R1_all'), float('nan')):.2f}%, "
                    f"wR2all={100.0 * safe_float(sh.get('wR2_all'), float('nan')):.2f}%"
                )
        else:
            recommendations.append(
                "observation polish was tested but not promoted because the raw-row or SHELXL endpoint gate did not pass"
            )
    summary = {
        "row_table": str(row_table.resolve()),
        "hkl": hkl,
        "res": res,
        "depth": depth,
        "dyn_all": dyn_all,
        "premerge_primary_mode": pmode,
        "two_part_signed_pass": signed_pass if signed else None,
        "two_part_frame_mod_pass": frame_pass if frame_mod else None,
        "dose_gate_pass": bool(dose.get("accept_dose_b_grid", False)) if dose else None,
        "observation_polish_accepted": bool(observation_polish.get("accepted", False)) if observation_polish else None,
        "best_observation_polish_endpoint": observation_polish.get("best_endpoint", {}) if observation_polish else {},
        "recommendations": recommendations,
        "artifacts": {
            "residual_compass": str((out_dir / "residual_compass" / "REPORT.md").resolve()),
            "premerge_rfactor": str((out_dir / "premerge_rfactor" / "REPORT.md").resolve()),
            "dose_damage": str((out_dir / "dose_damage" / "REPORT.md").resolve()) if dose else "",
            "two_part_signed": str((out_dir / "two_part_signed" / "REPORT.md").resolve()) if signed else "",
            "two_part_frame_mod": str((out_dir / "two_part_frame_mod" / "REPORT.md").resolve()) if frame_mod else "",
            "observation_polish": str((out_dir / "observation_polish" / "REPORT.md").resolve())
            if observation_polish
            else "",
        },
    }
    (out_dir / "forensic_gate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    return summary


def should_run_observation_polish(summary: dict, args: argparse.Namespace, depth: str) -> bool:
    """Return true when the raw-row correction passed enough gates to test."""

    mode = getattr(args, "observation_polish", "auto")
    if mode == "off":
        return False
    if depth == "quick":
        return False
    signed_pass = bool(summary.get("two_part_signed_pass", False))
    frame_pass = bool(summary.get("two_part_frame_mod_pass", False))
    if mode == "on":
        return signed_pass and frame_pass
    return signed_pass and frame_pass


def observation_polish_cmd(
    python: str, row_table: Path, hkl: str, res: str, out_dir: Path, args: argparse.Namespace
) -> list[str]:
    """Build the command that turns a passing observation correction into HKLs."""

    lambda_grid = getattr(args, "observation_polish_lambda_grid", "") or getattr(args, "lambda_grid", FULL_Q_LAMBDA)
    cmd = [
        python,
        str(engine_script("native_observation_polish_endpoint.py")),
        "--row-table",
        str(row_table),
        "--hkl",
        hkl,
        "--res",
        res,
        "--out-dir",
        str(out_dir / "observation_polish"),
        "--candidate-dir",
        str(out_dir / "two_part_signed"),
        "--candidate-dir",
        str(out_dir / "two_part_frame_mod"),
        "--combine-mode",
        getattr(args, "observation_polish_combine", "ensemble"),
        "--lambda-grid",
        lambda_grid,
        "--rescale",
        getattr(args, "rescale", "median-iobs"),
        "--hkl-float-format",
        getattr(args, "hkl_float_format", "shelx_fixed"),
        "--min-raw-wR2-gain",
        str(getattr(args, "observation_polish_min_raw_wR2_gain", 0.002)),
        "--endpoint-wR2-tolerance",
        str(getattr(args, "observation_polish_endpoint_wR2_tolerance", 0.0005)),
        "--shelxl",
        str(getattr(args, "shelxl", "") or default_shelxl_executable()),
        "--shelxl-iterations",
        str(getattr(args, "shelxl_iterations", 5)),
    ]
    if bool(getattr(args, "shelxl_carry_damp", False)):
        cmd.append("--keep-damp")
    if getattr(args, "forensic_maps", False) or getattr(args, "make_maps", False):
        cmd.append("--make-maps")
    return cmd


def run_forensic_gate(
    python: str,
    row_table: Path,
    hkl: str,
    res: str,
    out_dir: Path,
    logs: Path,
    label_prefix: str,
    args: argparse.Namespace,
) -> dict:
    """Run row-space forensics and optional observation-polish promotion."""

    if not getattr(args, "forensic_gate", True):
        return {}
    row_table = Path(row_table)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not row_table.exists():
        return {"status": "missing_row_table", "row_table": str(row_table)}
    depth = "quick" if getattr(args, "fast", False) and args.forensic_depth == "standard" else args.forensic_depth
    commands = forensic_command_plan(
        python,
        row_table,
        hkl,
        res,
        out_dir,
        depth=depth,
        bins=args.forensic_bins,
        top_rows=args.forensic_top_rows,
        top_groups=args.forensic_top_groups,
    )
    for name, cmd in commands:
        run_logged(cmd, logs, f"{label_prefix}_forensics_{name}", dry_run=args.dry_run)
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "commands": len(commands)}
    summary = summarize_forensic_gate(out_dir, row_table, hkl, res, depth)
    if should_run_observation_polish(summary, args, depth):
        conflicts = legacy_handoff_conflicts(args)
        if conflicts:
            # The polish endpoint is a publication candidate written with the legacy handoff contract.
            reason = "observation-polish endpoint uses the legacy SHELXL handoff and cannot honour: " + "; ".join(conflicts)
            if getattr(args, "observation_polish", "auto") == "on":
                raise RuntimeError(f"--observation-polish on: {reason}. Pass the legacy handoff flags or use auto.")
            summary["observation_polish_skipped"] = {"status": "not_applicable_handoff_contract", "reason": reason}
            (out_dir / "forensic_gate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
            return summary
        cmd = observation_polish_cmd(python, row_table, hkl, res, out_dir, args)
        try:
            run_logged(cmd, logs, f"{label_prefix}_forensics_observation_polish", dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            if getattr(args, "observation_polish", "auto") == "on":
                raise
            summary["observation_polish_error"] = {"returncode": exc.returncode, "command": exc.cmd}
            (out_dir / "forensic_gate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
            return summary
        summary = summarize_forensic_gate(out_dir, row_table, hkl, res, depth)
    return summary


def run_resolution_limit_gate(
    python: str,
    runs: dict[str, Path],
    res: str,
    out_dir: Path,
    logs: Path,
    label: str,
    args: argparse.Namespace,
    *,
    title: str,
    scope: str,
) -> dict:
    """Run held-out resolution-limit promotion on final row tables."""

    if not getattr(args, "resolution_limit_gate", True):
        return {"status": "disabled"}
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "runs": {k: str(v) for k, v in runs.items()}}
    missing = [
        str(Path(run) / "native_dyn_calc.csv")
        for run in runs.values()
        if not (Path(run) / "native_dyn_calc.csv").exists()
    ]
    if missing:
        return {"status": "missing_row_table", "missing": missing}
    cmd = resolution_limit_cmd(python, runs, res, out_dir, args, title=title, scope=scope)
    run_logged(cmd, logs, f"{label}_resolution_limit", dry_run=False)
    summary = try_read_json(out_dir / "summary.json")
    if summary:
        return {"status": "ok", **summary}
    return {
        "status": "missing_artifact_summary",
        "summary": str((out_dir / "summary.json").resolve()),
        "report": str((out_dir / "REPORT.md").resolve()),
        "shells_csv": str((out_dir / "resolution_shells.csv").resolve()),
    }
