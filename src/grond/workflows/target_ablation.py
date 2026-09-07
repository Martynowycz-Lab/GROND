"""Run the internal XDS target-intensity ablation and retain each diagnostic result."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import XDS_TARGET_MODE_LABELS, XDS_TARGET_MODES, parse_name_grid, try_read_json
from grond.progress import format_seconds
from grond.reports.source_controls import _quick_flat_hand_t_style_text
from grond.reports.values import _quick_number, _quick_percent
from grond.runtime import apply_thread_limits


def _target_ablation_variants(text: str) -> list[str]:
    """Parse target-ablation variant names with a stable default order."""

    raw = parse_name_grid(text)
    if not raw or raw == ["all"]:
        return list(XDS_TARGET_MODES)
    aliases = {
        "iobs-rlp": "iobs_over_rlp",
        "iobs/rlp": "iobs_over_rlp",
        "iobs_peak": "iobs_times_peak",
        "iobs-peak": "iobs_times_peak",
        "peak": "iobs_times_peak",
        "proxy": "recorded_proxy",
        "recorded": "recorded_proxy",
        "recorded-proxy": "recorded_proxy",
    }
    variants: list[str] = []
    for token in raw:
        value = aliases.get(token, token)
        if value not in XDS_TARGET_MODES:
            allowed = ", ".join(XDS_TARGET_MODES)
            raise SystemExit(f"Unknown XDS target-ablation variant {token!r}. Use one of: {allowed}.")
        if value not in variants:
            variants.append(value)
    return variants


def _target_ablation_auto_command(args: argparse.Namespace, variant: str, out_dir: Path) -> list[str]:
    """Build one fixed-target GROND auto command for the ablation wrapper."""

    cmd = [
        sys.executable,
        "-m",
        "grond.cli",
        "auto",
        "--out-dir",
        str(out_dir),
        "--structure",
        str(args.structure),
        "--baseline-hkl",
        str(args.baseline_hkl),
        "--target-mode",
        variant,
        "--shelxl-feedback-cycles",
        str(args.shelxl_feedback_cycles),
        "--shelxl-feedback-shelxl-iterations",
        str(args.shelxl_feedback_shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.endpoint_ls_cycles),
        "--shelxl",
        str(args.shelxl),
        "--min-d-spacing",
        f"{float(args.min_d_spacing):g}",
        "--min-peak",
        f"{float(args.min_peak):g}",
        "--geometry-percentile",
        f"{float(args.geometry_percentile):g}",
        f"--hkl-transform={args.hkl_transform}",
        "--workers",
        str(int(args.workers)),
        "--max-core-fraction",
        f"{float(args.max_core_fraction):g}",
        "--chunksize",
        str(int(args.chunksize)),
        "--compute-backend",
        str(args.compute_backend),
        "--row-plan-cache",
        str(args.row_plan_cache),
        "--prediction-cache-compression",
        str(args.prediction_cache_compression),
    ]
    if args.xds_dir:
        cmd.extend(["--xds-dir", str(args.xds_dir)])
    if args.hkl:
        cmd.extend(["--hkl", str(args.hkl)])
    if args.python:
        cmd.extend(["--python", str(args.python)])
    if getattr(args, "config", ""):
        cmd.extend(["--config", str(args.config)])
    if getattr(args, "allow_missing_tools", False):
        cmd.append("--allow-missing-tools")
    if args.trust_hkl_transform:
        cmd.append("--trust-hkl-transform")
    cmd.append("--hand-control" if args.hand_control else "--no-hand-control")
    cmd.append("--resume" if args.resume else "--no-resume")
    cmd.append("--make-maps" if args.make_maps else "--no-maps")
    for token in shlex.split(str(getattr(args, "extra_auto_args", "") or "")):
        cmd.append(token)
    return cmd


def _run_target_ablation_variant(cmd: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, float]:
    """Run one auto variant while teeing terminal output into a stable log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        if dry_run:
            print("DRY", " ".join(shlex.quote(part) for part in cmd), flush=True)
            handle.write("[dry_run] true\n[returncode] 0\n[seconds] 0.000\n")
            return 0, 0.0
        env = apply_thread_limits(os.environ.copy())
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        returncode = proc.wait()
        elapsed = time.perf_counter() - start
        handle.write(f"\n[returncode] {returncode}\n[seconds] {elapsed:.3f}\n")
    return int(returncode), elapsed


def _target_ablation_summary_row(
    variant: str, run_dir: Path, log_path: Path, returncode: int, elapsed_seconds: float
) -> dict[str, object]:
    """Collect the compact fields from one completed target-ablation run."""

    result = try_read_json(run_dir / "grond_summary.json")
    quick = result.get("quick_stats") if isinstance(result.get("quick_stats"), Mapping) else {}
    console = quick.get("console") if isinstance(quick.get("console"), Mapping) else {}
    input_model = console.get("input_model") if isinstance(console.get("input_model"), Mapping) else {}
    final_row = console.get("final_row_model") if isinstance(console.get("final_row_model"), Mapping) else {}
    feedback = (
        console.get("final_feedback_endpoint") if isinstance(console.get("final_feedback_endpoint"), Mapping) else {}
    )
    endpoint = console.get("publication_endpoint") if isinstance(console.get("publication_endpoint"), Mapping) else {}
    input_shelxl = console.get("input_shelxl") if isinstance(console.get("input_shelxl"), Mapping) else {}
    files = console.get("files") if isinstance(console.get("files"), Mapping) else {}
    final_summary = Path(str(console.get("final_summary") or (run_dir / "FINAL_SUMMARY.md")))
    quick_stats = Path(str(console.get("quick_stats") or (run_dir / "QUICK_STATS.md")))
    status = "ok" if returncode == 0 and bool(result) else ("failed" if returncode != 0 else "missing_summary")
    return {
        "target_mode": variant,
        "target_formula": XDS_TARGET_MODE_LABELS.get(variant, variant),
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_seconds(elapsed_seconds),
        "hand": console.get("final_hand", ""),
        "hand_t_style_all": console.get("final_hand_t_style_all", console.get("final_hand_z_all")),
        "hand_t_style_all_n": console.get("final_hand_t_style_all_n"),
        "hand_t_style_all_df": console.get("final_hand_t_style_all_df"),
        "hand_zkp_all": console.get("final_hand_zkp_all"),
        "hand_zkp_free": console.get("final_hand_zkp_free"),
        "hand_warnings": console.get("final_hand_warnings", []),
        "hand_t_style_free": console.get("final_hand_t_style_free", console.get("final_hand_z_free")),
        "hand_t_style_free_n": console.get("final_hand_t_style_free_n"),
        "hand_t_style_free_df": console.get("final_hand_t_style_free_df"),
        "hand_z_all": console.get("final_hand_z_all"),
        "hand_z_free": console.get("final_hand_z_free"),
        "hand_z_compatibility_note": (
            "legacy field names only; values are paired Student-t-style contrasts, not calibrated normal Z scores"
        ),
        "hkl_transform": console.get("selected_transform", ""),
        "input_shelxl_R1gt": input_shelxl.get("R1gt"),
        "input_shelxl_R1all": input_shelxl.get("R1all"),
        "input_shelxl_wR2all": input_shelxl.get("wR2all"),
        "input_shelxl_GooF": input_shelxl.get("GooF"),
        "input_dyn_R1all": input_model.get("row_R1all"),
        "input_dyn_wR2all": input_model.get("row_wR2all"),
        "input_born_R1all": input_model.get("born_R1all"),
        "input_born_wR2all": input_model.get("born_wR2all"),
        "final_dyn_R1all": final_row.get("row_R1all"),
        "final_dyn_wR2all": final_row.get("row_wR2all"),
        "final_dyn_free_wR2all": final_row.get("row_free_wR2all"),
        "final_born_R1all": final_row.get("born_R1all"),
        "final_born_wR2all": final_row.get("born_wR2all"),
        "final_born_free_wR2all": final_row.get("born_free_wR2all"),
        "direct_ls1_R1gt": feedback.get("R1gt"),
        "direct_ls1_R1all": feedback.get("R1all"),
        "direct_ls1_wR2all": feedback.get("wR2all"),
        "direct_ls1_GooF": feedback.get("GooF"),
        "publication_R1gt": endpoint.get("R1gt"),
        "publication_R1all": endpoint.get("R1all"),
        "publication_wR2all": endpoint.get("wR2all"),
        "publication_GooF": endpoint.get("GooF"),
        "publication_source": endpoint.get("source"),
        "publication_kind": endpoint.get("source_kind"),
        "publication_scale_spec": endpoint.get("scale_spec"),
        "publication_scale_params": endpoint.get("scale_params"),
        "publication_wght_mode": endpoint.get("wght_mode"),
        "publication_parameters": endpoint.get("parameters"),
        "publication_npd_count": endpoint.get("npd_count"),
        "publication_split_count": endpoint.get("split_count"),
        "run_dir": str(run_dir.resolve()),
        "terminal_log": str(log_path.resolve()),
        "final_summary": str(final_summary.resolve()),
        "quick_stats": str(quick_stats.resolve()),
        "full_json": str((run_dir / "grond_summary.json").resolve()),
        "row_table": str(final_row.get("row_table", "")),
        "publication_res": str(files.get("publication_res", "")),
        "publication_hkl": str(files.get("publication_hkl", "")),
        "publication_lst": str(files.get("publication_lst", "")),
    }


def _write_target_ablation_report(
    out_dir: Path, rows: list[dict[str, object]], commands: list[dict[str, object]]
) -> None:
    """Write the human-readable XDS target-transform ablation report."""

    lines = [
        "# GROND XDS Target-Transform Ablation",
        "",
        "This diagnostic runs the same GROND auto path while changing only the reduced XDS row target passed into the Bloch-to-Born correction. It is meant to test Lorentz/PEAK conventions in the current scalar XDS row contract; it is not a raw-shoebox likelihood.",
        "",
        "## Results",
        "",
        "| target | formula | status | elapsed | hand paired t-style all/free | final dyn R1/wR2 | final Born R1/wR2 | publication R1/wR2/GooF | run |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('target_mode', '')}`",
                    str(row.get("target_formula", "")),
                    str(row.get("status", "")),
                    str(row.get("elapsed", "")),
                    f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_all'), row.get('hand_t_style_all_n'), row.get('hand_t_style_all_df'))} / "
                    f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_free'), row.get('hand_t_style_free_n'), row.get('hand_t_style_free_df'))}",
                    f"{_quick_percent(row.get('final_dyn_R1all'))} / {_quick_percent(row.get('final_dyn_wR2all'))}",
                    f"{_quick_percent(row.get('final_born_R1all'))} / {_quick_percent(row.get('final_born_wR2all'))}",
                    f"{_quick_percent(row.get('publication_R1all'))} / {_quick_percent(row.get('publication_wR2all'))} / {_quick_number(row.get('publication_GooF'), 3)}",
                    f"`{Path(str(row.get('run_dir', ''))).name}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Endpoint Paths",
            "",
            "| target | publication RES | publication HKL | row table | terminal log |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('target_mode', '')}`",
                    f"`{row.get('publication_res', '')}`",
                    f"`{row.get('publication_hkl', '')}`",
                    f"`{row.get('row_table', '')}`",
                    f"`{row.get('terminal_log', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Commands", ""])
    for command in commands:
        lines.append(f"### {command.get('target_mode', '')}")
        lines.append("")
        lines.append("```bash")
        lines.append(str(command.get("command_text", "")))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Reading This",
            "",
            "The production interpretation is conservative. A target transform is interesting only if it improves row-level dynamical statistics, keeps or improves the paired t-style hand contrast, and does not win solely by producing a prettier corrected-HKL endpoint. This contrast is reported with n and df=n-1 and is not a calibrated normal Z score. If a PEAK/RLP variant lowers endpoint R but worsens row-space Born/Bloch behavior or hand separation, keep it as a diagnostic and do not promote it.",
            "",
        ]
    )
    (out_dir / "TARGET_ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n")
