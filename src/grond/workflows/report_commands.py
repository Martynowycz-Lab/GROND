"""Build merge, endpoint-report and resolution-audit commands."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import FULL_Q_LAMBDA, try_read_json
from grond.io.config import PrepConfig
from grond.paths import engine_script
from grond.workflows.decision import safe_float
from grond.workflows.endpoint_commands import endpoint_handoff_args, endpoint_intensity_rescale
from grond.workflows.nuisance_policy import (
    resolved_dataset_sigma_policy,
    resolved_feedback_wght_mode,
    resolved_merge_sigma_policy,
    resolved_post_bloch_sigma_policy,
)


def combine_cmd(
    python: str,
    runs: dict[str, Path],
    structure: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    shelxl_iterations: int | None = None,
    feedback_cycle: bool = False,
    final_outputs: bool | None = None,
    free_validation: bool = False,
) -> list[str]:
    """Build the multi-dataset corrected-HKL combination command.

    ``feedback_cycle`` marks a structure-feedback macro-cycle: it uses the macro-cycle weighting rule
    (--shelxl-feedback-wght-mode).  ``final_outputs`` (default: not a feedback cycle) adds the unmerged
    endpoint and ACTA, which belong to the final combined endpoint only.  ``free_validation`` (combine_final only)
    forwards --endpoint-free-validation on when the AUTO option is on and the free rows stay out of the HKL; the
    option is never emitted otherwise, so every other command is unchanged.
    """

    iterations = int(shelxl_iterations if shelxl_iterations is not None else getattr(args, "shelxl_iterations", 5))
    endpoint_ls_cycles = int(
        getattr(args, "shelxl_feedback_endpoint_ls_cycles", getattr(args, "endpoint_ls_cycles", 1)) or 1
    )
    absolute_scale = endpoint_intensity_rescale(args) == "absolute"
    cmd = [
        python,
        str(engine_script("combine_native_nuisance_runs.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        structure,
        "--endpoint-lambda",
        FULL_Q_LAMBDA,
        "--endpoint-source",
        "native",
        "--interdataset-scale",
        # On the absolute kappa*|F|^2 scale every crystal is already on one scale; the combine then reports
        # the overlap estimate as a diagnostic instead of applying it.
        "none" if absolute_scale else "overlap",
        "--endpoint-merge-key",
        getattr(args, "multi_endpoint_merge_key", "model_symmetry"),
        "--dataset-sigma-policy",
        resolved_dataset_sigma_policy(args, multi_dataset=True),
        "--dataset-sigma-holdout-fraction",
        f"{args.dataset_sigma_holdout_fraction:g}",
        "--dataset-sigma-min-scale",
        f"{args.dataset_sigma_min_scale:g}",
        "--dataset-sigma-max-scale",
        f"{args.dataset_sigma_max_scale:g}",
        "--merge-sigma-policy",
        resolved_merge_sigma_policy(args, multi_dataset=True),
        "--post-bloch-sigma-policy",
        resolved_post_bloch_sigma_policy(args, final_endpoint=True, combined=True),
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--shelxl",
        str(getattr(args, "shelxl", "") or ""),
        "--shelxl-iterations",
        str(iterations),
        "--endpoint-ls-cycles",
        str(endpoint_ls_cycles),
        "--endpoint-wght-mode",
        resolved_feedback_wght_mode(args) if feedback_cycle else str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
        *endpoint_handoff_args(args),
    ]
    if (not feedback_cycle) if final_outputs is None else final_outputs:
        if str(getattr(args, "unmerged_endpoint", "off") or "off") == "on":
            cmd.append("--unmerged-endpoint")
        if bool(getattr(args, "shelx_acta", False)):
            cmd.append("--acta")
    endpoint_min_d_spacing = float(
        getattr(args, "endpoint_min_d_spacing", 0.0) or getattr(args, "min_d_spacing", 0.0) or 0.0
    )
    if endpoint_min_d_spacing > 0.0:
        cmd += ["--min-d-spacing", f"{endpoint_min_d_spacing:g}"]
    if args.dataset_sigma_holdout_floor:
        cmd.append("--dataset-sigma-holdout-floor")
    if free_validation and endpoint_free_validation_active(args):
        cmd += ["--endpoint-free-validation", "on"]
    for name, run in runs.items():
        cmd += ["--run", f"{name}={run}"]
    return cmd


def endpoint_free_validation_active(args: argparse.Namespace) -> bool:
    """AUTO --endpoint-free-validation on with the free rows kept out of every refinement HKL (legacy when absent)."""

    return str(getattr(args, "endpoint_free_validation", "off") or "off") == "on" and not bool(
        getattr(args, "endpoint_include_free_rows", True)
    )


def publication_report_cmd(
    python: str,
    prep: PrepConfig,
    result: dict,
    out_dir: Path,
    title: str,
    *,
    instability_u: float = 0.01,
) -> list[str]:
    """Build the GROND row/MR publication-table command for final GROND rows."""

    cmd = [
        python,
        str(engine_script("row_mr_report.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        str(result.get("model") or prep.structure),
        "--title",
        title,
        "--instability-u",
        f"{float(instability_u):g}",
    ]
    if result.get("combined"):
        for name, run in (result.get("publication_runs") or result.get("runs") or {}).items():
            cmd += ["--run", f"{name}={run}"]
        for name, run in (result.get("publication_full_runs") or {}).items():
            cmd += ["--full-run", f"{name}={run}"]
        scale_contract = result.get("scale_contract_selection")
        if isinstance(scale_contract, Mapping):
            selected_contract = str(scale_contract.get("selected_contract", "")).strip()
            if selected_contract:
                cmd += ["--selected-scale-contract", selected_contract]
        combined_summary = try_read_json(Path(str(result["combined"])) / "combined_summary.json")
        combined_endpoint = combined_summary.get("combined_endpoint")
        combined_endpoint = combined_endpoint if isinstance(combined_endpoint, Mapping) else {}
        requested_dmin = safe_float(
            combined_endpoint.get(
                "requested_min_d_spacing_A",
                combined_endpoint.get("min_d_spacing_A"),
            ),
            0.0,
        )
        if requested_dmin > 0.0:
            cmd += ["--requested-min-d-spacing", f"{requested_dmin:g}"]
    elif result.get("primary_run"):
        name = prep.datasets[0].name if prep.datasets else "dataset"
        cmd += ["--run", f"{name}={result['primary_run']}"]
    cmd += resolution_limit_report_args(result)
    return cmd


def resolution_limit_report_args(result: dict) -> list[str]:
    """Return final-table arguments for promoted resolution-limit summaries."""

    args: list[str] = []
    merged = result.get("merged_resolution_limit")
    if isinstance(merged, Mapping):
        summary = (merged.get("artifacts") or {}).get("summary") or ""
        if summary:
            args += ["--resolution-limit", f"combined={summary}"]

    resolution_limits = result.get("resolution_limits")
    if isinstance(resolution_limits, Mapping):
        for name, payload in resolution_limits.items():
            if not isinstance(payload, Mapping):
                continue
            summary = (payload.get("artifacts") or {}).get("summary") or ""
            if summary:
                args += ["--resolution-limit", f"{name}={summary}"]
    return args


def resolution_limit_cmd(
    python: str,
    runs: dict[str, Path],
    res: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    title: str,
    scope: str,
) -> list[str]:
    """Build the held-out resolution-limit promotion command."""

    cmd = [
        python,
        str(engine_script("resolution_limit_audit.py")),
        "--out-dir",
        str(out_dir),
        "--res",
        res,
        "--title",
        title,
        "--scope",
        scope,
        "--shells",
        str(args.resolution_shells),
        "--strict-split-cc",
        str(args.resolution_strict_split_cc),
        "--strict-mean-i-over-sigma",
        str(args.resolution_strict_mean_i_over_sigma),
        "--exploratory-split-cc",
        str(args.resolution_exploratory_split_cc),
        "--exploratory-mean-i-over-sigma",
        str(args.resolution_exploratory_mean_i_over_sigma),
        "--min-common-pairs",
        str(args.resolution_min_common_pairs),
        "--min-shell-rows",
        str(args.resolution_min_shell_rows),
        "--min-strong-fraction",
        str(args.resolution_min_strong_fraction),
        "--max-shell-mwrall",
        str(args.resolution_max_shell_mwrall),
        "--max-free-work-mwr-gap",
        str(args.resolution_max_free_work_mwr_gap),
        "--min-free-rows",
        str(args.resolution_min_free_rows),
        "--max-cumulative-mwr-penalty",
        str(args.resolution_max_cumulative_mwr_penalty),
    ]
    for name, run in runs.items():
        cmd += ["--run", f"{name}={run}"]
    return cmd
