"""Launch scale compression or replay a recorded scale contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.tools import choose_python
from grond.cli_support.values import try_read_json
from grond.paths import engine_script


def cmd_scale_polish(args: argparse.Namespace) -> int:
    """CLI entrypoint for frozen-Bloch scale-compression polish."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    cmd = [
        python,
        str(engine_script("scale_compression_polish.py")),
        "--row-table",
        str(Path(args.row_table).expanduser().resolve()),
        "--out-dir",
        str(out_dir),
        "--title",
        args.title,
        "--group-strides",
        args.group_strides,
        "--window-scan-deg",
        str(args.window_scan_deg),
        "--detector-bins",
        args.detector_bins,
        "--scale-objective",
        args.scale_objective,
        "--tolerance",
        str(args.tolerance),
        "--selection-metrics",
        args.selection_metrics,
        "--frame-models",
        args.frame_models,
        "--adaptive-frame-counts",
        args.adaptive_frame_counts,
    ]
    if args.summary:
        cmd += ["--summary", str(Path(args.summary).expanduser().resolve())]
    if args.fit_split:
        cmd += ["--fit-split", args.fit_split]
    if args.scan_offsets:
        cmd.append("--scan-offsets")
    if args.native_wght_a is not None:
        cmd += ["--native-wght-a", str(args.native_wght_a)]
    if args.native_wght_b is not None:
        cmd += ["--native-wght-b", str(args.native_wght_b)]
    run_logged(cmd, logs, "scale_compression_polish", dry_run=args.dry_run)
    summary = try_read_json(out_dir / "scale_compression_summary.json")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "report": str((out_dir / "REPORT.md").resolve()),
                "candidate_csv": str((out_dir / "scale_compression_candidates.csv").resolve()),
                "baseline": summary.get("baseline", {}),
                "selected": summary.get("selected", {}),
                "selected_widest": summary.get("selected_widest", {}),
                "selected_r_only": summary.get("selected_r_only", {}),
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0


def cmd_scale_replay(args: argparse.Namespace) -> int:
    """CLI entrypoint for replaying compact scale contracts through endpoints."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    cmd = [
        python,
        str(engine_script("scale_contract_replay.py")),
        "--row-table",
        str(Path(args.row_table).expanduser().resolve()),
        "--out-dir",
        str(out_dir),
        "--specs",
        args.specs,
        "--fit-split",
        args.fit_split,
        "--lambda-q",
        str(args.lambda_q),
        "--rescale",
        args.rescale,
        "--hkl-float-format",
        args.hkl_float_format,
        "--post-bloch-sigma-policy",
        args.post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        str(args.post_bloch_sigma_holdout_fraction),
        "--post-bloch-sigma-min-scale",
        str(args.post_bloch_sigma_min_scale),
        "--post-bloch-sigma-max-scale",
        str(args.post_bloch_sigma_max_scale),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(args.shelxl_iterations),
        "--wght-modes",
        args.wght_modes,
        "--instability-u",
        str(getattr(args, "instability_u", 0.01)),
        "--publication-goof-min",
        str(args.publication_goof_min),
        "--publication-goof-max",
        str(args.publication_goof_max),
        "--publication-max-r1-cost",
        str(args.publication_max_r1_cost),
        "--publication-free-wr2-tolerance",
        str(args.publication_free_wr2_tolerance),
        "--publication-free-rall-tolerance",
        str(args.publication_free_rall_tolerance),
        "--publication-work-rall-tolerance",
        str(args.publication_work_rall_tolerance),
        "--publication-free-wrall-tolerance",
        str(args.publication_free_wrall_tolerance),
        "--publication-max-modulation-p95",
        str(args.publication_max_modulation_p95),
        "--publication-prefer-wght-mode",
        args.publication_prefer_wght_mode,
        "--publication-selection-policy",
        args.publication_selection_policy,
        "--publication-information-criterion",
        args.publication_information_criterion,
        "--detector-ridge",
        str(args.detector_ridge),
        "--detector-smoothness",
        str(args.detector_smoothness),
        "--detector-min-rows",
        str(args.detector_min_rows),
        "--detector-min-holdout-improvement",
        str(args.detector_min_holdout_improvement),
        "--detector-required-improved-splits",
        str(args.detector_required_improved_splits),
        "--detector-max-bloch-gain-loss",
        str(args.detector_max_bloch_gain_loss),
        "--detector-min-field-correlation",
        str(args.detector_min_field_correlation),
        "--detector-max-field-log-rms",
        str(args.detector_max_field_log_rms),
    ]
    if bool(getattr(args, "publication_allow_npd", False)):
        cmd.append("--publication-allow-npd")
    if bool(getattr(args, "detector_independent_calibration", False)):
        cmd.append("--detector-independent-calibration")
    if args.summary:
        cmd += ["--summary", str(Path(args.summary).expanduser().resolve())]
    if args.res:
        cmd += ["--res", str(Path(args.res).expanduser().resolve())]
    if args.native_wght_a is not None:
        cmd += ["--native-wght-a", str(args.native_wght_a)]
    if args.native_wght_b is not None:
        cmd += ["--native-wght-b", str(args.native_wght_b)]
    run_logged(cmd, logs, "scale_contract_replay", dry_run=args.dry_run)
    summary = try_read_json(out_dir / "summary.json")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "report": str((out_dir / "REPORT.md").resolve()),
                "endpoint_csv": str((out_dir / "scale_contract_endpoint_replay.csv").resolve()),
                "publication_selection": summary.get("publication_selection", {}),
                "rows": summary.get("rows", []),
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0
