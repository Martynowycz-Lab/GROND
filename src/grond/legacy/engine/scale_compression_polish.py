#!/usr/bin/env python3
"""Compress final row-space scale factors after a GROND Bloch calculation.

This is a post-regularization polish.  It does not recompute Bloch waves and it
does not move the structure.  It reads a final ``native_dyn_calc.csv`` table,
keeps ``dyn_raw`` and ``born_raw`` fixed, and asks how simple the multiplicative
scale model can be while staying close to the current full-scale result.
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
from pathlib import Path

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from grond.regularize.arguments_compression import build_parser as build_compression_parser
from grond.regularize.compression_policy import (
    candidate_row as candidate_row,
    detector_scale_policy_diagnostics as detector_scale_policy_diagnostics,
    metric_delta as metric_delta,
    within_tolerance as within_tolerance,
)
from grond.regularize.compression_topology import (
    adaptive_segment_groups as adaptive_segment_groups,
    baseline_scale_log_by_fit as baseline_scale_log_by_fit,
    baseline_scale_log_by_row as baseline_scale_log_by_row,
    baseline_within_group_log_rms as baseline_within_group_log_rms,
    build_group_strides as build_group_strides,
    coarsen_detector as coarsen_detector,
    coarsen_groups as coarsen_groups,
    count_scale_params as count_scale_params,
    infer_detector_base as infer_detector_base,
    infer_group_width_deg as infer_group_width_deg,
    log_scale_step_rms as log_scale_step_rms,
    split_masks as split_masks,
)
from grond.regularize.compression_weights import (
    col as col,
    fit_group_detector_scales as fit_group_detector_scales,
    fit_group_scales as fit_group_scales,
    information_criteria as information_criteria,
    native_metrics as native_metrics,
    row_mr_metrics as row_mr_metrics,
    safe_float as safe_float,
    source_sigma_array as source_sigma_array,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts; raise if it has no rows."""

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to a CSV with the given header, ignoring extra keys; create the parent folder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_candidate_rows(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    *,
    candidate_name: str,
    baseline_dyn_column: str,
    baseline_born_column: str,
) -> tuple[list[dict[str, object]], list[str]]:
    """Return a replayable compact row table with its exact nuisance topology."""

    expected = len(rows)
    for name in ("dyn", "born", "groups", "detector_bins"):
        values = np.asarray(arrays[name])
        if len(values) != expected:
            raise ValueError(
                f"candidate array {name!r} has {len(values)} rows; expected {expected}"
            )
    updated_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        new = dict(row)
        new["dyn_full_scaled"] = row.get(baseline_dyn_column, "")
        new["born_full_scaled"] = row.get(baseline_born_column, "")
        new["dyn_scaled"] = (
            f"{arrays['dyn'][i]:.17g}" if np.isfinite(arrays["dyn"][i]) else "nan"
        )
        new["born_scaled"] = (
            f"{arrays['born'][i]:.17g}" if np.isfinite(arrays["born"][i]) else "nan"
        )
        new["scale_replay_group"] = int(arrays["groups"][i])
        new["scale_replay_detector_bin"] = int(arrays["detector_bins"][i])
        new["scale_compression_model"] = candidate_name
        updated_rows.append(new)
    fieldnames = list(rows[0].keys())
    for extra in (
        "dyn_full_scaled",
        "born_full_scaled",
        "scale_replay_group",
        "scale_replay_detector_bin",
        "scale_compression_model",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)
    return updated_rows, fieldnames


def read_json(path: Path) -> dict:
    """Read a JSON file, or return an empty dict if it does not exist."""

    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_csv_metrics(text: str) -> tuple[str, ...]:
    """Parse the comma-separated selection metrics, defaulting to Robs, Rall and wRall; raise on unknown names."""

    metrics = tuple(part.strip() for part in str(text).split(",") if part.strip())
    allowed = {
        "Robs",
        "Rall",
        "wRall",
        "work_Rall",
        "work_wRall",
        "free_Rall",
        "free_wRall",
        "native_R1_gt",
        "native_R1_all",
        "native_wR2_all",
    }
    bad = sorted(set(metrics) - allowed)
    if bad:
        raise ValueError(f"Unknown selection metrics: {bad}; allowed={sorted(allowed)}")
    return metrics or ("Robs", "Rall", "wRall")


def parse_frame_models(text: str) -> tuple[str, ...]:
    """Return requested frame-scale model families."""

    models = tuple(part.strip().lower() for part in str(text).split(",") if part.strip())
    allowed = {"adaptive", "fixed"}
    bad = sorted(set(models) - allowed)
    if bad:
        raise ValueError(f"Unknown frame-scale model(s): {bad}; allowed={sorted(allowed)}")
    return models or ("adaptive", "fixed")


def parse_adaptive_counts(text: str, n_groups: int) -> list[int]:
    """Return target segment counts for adaptive neighboring-frame compression."""

    n_groups = max(1, int(n_groups))
    raw = str(text or "auto").strip().lower()
    if raw in {"off", "none", "false", "0"}:
        return []
    if raw == "auto":
        fractions = (0.35, 0.45, 0.50, 0.55, 0.60, 0.67, 0.75, 0.85)
        counts = [int(round(n_groups * frac)) for frac in fractions]
        if n_groups > 120:
            counts.extend([40, 60, 80, 100, 150, 200, 250, 300, 360, 420])
    else:
        counts = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if part.endswith("%"):
                counts.append(int(round(n_groups * float(part[:-1]) / 100.0)))
            else:
                counts.append(int(float(part)))
    return sorted({max(1, min(n_groups, int(v))) for v in counts if int(v) > 0})


def parse_detector_targets(text: str, detector_base: int) -> list[int]:
    """Parse detector flat-field choices, including the production-safe exact mode."""

    detector_base = max(1, int(detector_base))
    out: list[int] = []
    for part in str(text or "exact").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"none", "off", "0"}:
            out.append(0)
        elif token == "exact":
            out.append(detector_base)
        elif token == "auto":
            out.extend([detector_base])
        else:
            out.append(int(float(token)))
    out = [v for v in out if 0 <= int(v) <= detector_base]
    if not out:
        out = [detector_base]
    return sorted(set(out), key=out.index)


def fmt(value: object, digits: int = 4) -> str:
    """Format a number with the given decimals, or 'n/a' if it is not finite."""

    v = safe_float(value)
    return "n/a" if not np.isfinite(v) else f"{v:.{digits}f}"


def load_baseline_counts(summary: dict, n_rows: int) -> tuple[int, int, int]:
    """Return the baseline (scale, total, physical) parameter counts from the native summary.

    The total is the row count minus the recorded degrees of freedom when present, else the scale count.
    """

    best = summary.get("best_native") or {}
    scale_count = int(safe_float(best.get("scale_count"), 0.0))
    dof = safe_float((((best.get("metrics") or {}).get("all") or {}).get("dyn") or {}).get("dof"), float("nan"))
    total = int(max(0, round(n_rows - dof))) if np.isfinite(dof) else scale_count
    physical = max(0, total - scale_count)
    return scale_count, total, physical


def native_weight_from_summary(summary: dict, cli_a: float, cli_b: float) -> tuple[float, float]:
    """Return the native WGHT (a, b): finite command-line values, else the summary's, else 0."""

    weight = (summary.get("best_native") or {}).get("native_weight") or {}
    a = cli_a if np.isfinite(cli_a) else safe_float(weight.get("a"), 0.0)
    b = cli_b if np.isfinite(cli_b) else safe_float(weight.get("b"), 0.0)
    return float(a), float(b)


def write_report(path: Path, payload: dict) -> None:
    """Write the Markdown report: baseline, strict, widest-strict, R-only and information-criterion selections."""

    baseline = payload["baseline"]
    selected = payload["selected"]
    widest = payload.get("selected_widest")
    r_only = payload["selected_r_only"]
    selected_ic = payload.get("selected_information")
    lines = [
        f"# {payload['title']}",
        "",
        "This is a frozen-Bloch post-regularization scale-compression pass. The structural model and raw Bloch predictions were not changed.",
        "",
        "## Baseline",
        "",
        f"Baseline row table: `{payload['row_table']}`",
        f"Baseline scale parameters: `{baseline['scale_params']}`; total nuisance/scale parameters used for dof accounting: `{baseline['total_params']}`.",
        f"Baseline GROND row/MR dynamical metrics: Robs `{fmt(baseline['Robs'])}`, Rall `{fmt(baseline['Rall'])}`, wRall `{fmt(baseline['wRall'])}`.",
        f"Baseline native metrics: R1gt `{fmt(baseline['native']['R1_gt'])}`, R1all `{fmt(baseline['native']['R1_all'])}`, wR2all `{fmt(baseline['native']['wR2_all'])}`, GooF `{fmt(baseline['native'].get('GooF'))}`.",
        "",
        "## Selected Strict Model",
        "",
    ]
    if selected:
        lines.extend(
            [
                f"Selected model: `{selected['name']}`.",
                f"Physical scale-window width: `{fmt(selected.get('window_deg'))}` deg; group offset: `{selected.get('group_offset', 0)}`.",
                f"Scale parameters: `{selected['scale_params']}`; total parameters: `{selected['total_params']}`.",
                f"GROND row/MR metrics: Robs `{fmt(selected['Robs'])}` ({fmt(selected['delta_vs_baseline']['Robs'])}), Rall `{fmt(selected['Rall'])}` ({fmt(selected['delta_vs_baseline']['Rall'])}), wRall `{fmt(selected['wRall'])}` ({fmt(selected['delta_vs_baseline']['wRall'])}).",
                f"Native metrics: R1gt `{fmt(selected['native']['R1_gt'])}` ({fmt(selected['delta_vs_baseline']['native_R1_gt'])}), R1all `{fmt(selected['native']['R1_all'])}` ({fmt(selected['delta_vs_baseline']['native_R1_all'])}), wR2all `{fmt(selected['native']['wR2_all'])}` ({fmt(selected['delta_vs_baseline']['native_wR2_all'])}), GooF `{fmt(selected['native'].get('GooF'))}`.",
            ]
        )
    else:
        lines.append("No candidate satisfied the strict selection metrics within the requested tolerance.")
    lines.extend(["", "## Widest Strict Window", ""])
    if widest:
        lines.extend(
            [
                f"Widest strict model: `{widest['name']}`.",
                f"Physical scale-window width: `{fmt(widest.get('window_deg'))}` deg; group offset: `{widest.get('group_offset', 0)}`.",
                f"Scale parameters: `{widest['scale_params']}`; total parameters: `{widest['total_params']}`.",
                f"GROND row/MR metrics: Robs `{fmt(widest['Robs'])}` ({fmt(widest['delta_vs_baseline']['Robs'])}), Rall `{fmt(widest['Rall'])}` ({fmt(widest['delta_vs_baseline']['Rall'])}), wRall `{fmt(widest['wRall'])}` ({fmt(widest['delta_vs_baseline']['wRall'])}).",
                f"Native metrics: R1gt `{fmt(widest['native']['R1_gt'])}` ({fmt(widest['delta_vs_baseline']['native_R1_gt'])}), R1all `{fmt(widest['native']['R1_all'])}` ({fmt(widest['delta_vs_baseline']['native_R1_all'])}), wR2all `{fmt(widest['native']['wR2_all'])}` ({fmt(widest['delta_vs_baseline']['native_wR2_all'])}).",
            ]
        )
    else:
        lines.append("No candidate satisfied the strict selection metrics within the requested tolerance.")
    lines.extend(["", "## R-Only Fallback", ""])
    if r_only:
        lines.extend(
            [
                f"Selected R-only model: `{r_only['name']}`.",
                f"Physical scale-window width: `{fmt(r_only.get('window_deg'))}` deg; group offset: `{r_only.get('group_offset', 0)}`.",
                f"Scale parameters: `{r_only['scale_params']}`; total parameters: `{r_only['total_params']}`.",
                f"GROND row/MR metrics: Robs `{fmt(r_only['Robs'])}` ({fmt(r_only['delta_vs_baseline']['Robs'])}), Rall `{fmt(r_only['Rall'])}` ({fmt(r_only['delta_vs_baseline']['Rall'])}), wRall `{fmt(r_only['wRall'])}` ({fmt(r_only['delta_vs_baseline']['wRall'])}).",
            ]
        )
    else:
        lines.append("No candidate satisfied the R-only tolerance either.")
    lines.extend(["", "## Information-Criterion Selection", ""])
    if selected_ic:
        lines.extend(
            [
                "Among candidates that pass the strict residual gates, GROND also reports an information-criterion choice. This does not replace the residual gates; it answers how many scale parameters are justified after the candidate is already close enough to the full-scale result.",
                f"Selected information model: `{selected_ic['name']}`.",
                f"Scale parameters: `{selected_ic['scale_params']}`; total parameters: `{selected_ic['total_params']}`.",
                f"BIC with fitted residual variance: `{fmt(selected_ic.get('information', {}).get('bic_fitted_variance'))}`; AIC with fitted residual variance: `{fmt(selected_ic.get('information', {}).get('aic_fitted_variance'))}`.",
                f"Known-sigma BIC: `{fmt(selected_ic.get('information', {}).get('bic_known_sigma'))}`; known-sigma AIC: `{fmt(selected_ic.get('information', {}).get('aic_known_sigma'))}`.",
                f"Recommended compact scale contract for automatic output: `{selected_ic['name']}`. This is the strict-gated model with the best information score, not the lowest-parameter model overall.",
            ]
        )
    else:
        lines.append("No strict candidate was available for information-criterion selection.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"Strict tolerance metrics: `{', '.join(payload['selection_metrics'])}` with tolerance `{payload['tolerance']}`.",
            "The default automatic gate prioritizes all-row residuals and treats strong-row residuals as report-only diagnostics.",
            f"Window scan: `{payload.get('window_scan', {})}`; scan offsets: `{payload.get('scan_offsets', False)}`.",
            f"Frame-scale models tested: `{payload.get('frame_models', ())}`; adaptive segment counts: `{payload.get('adaptive_frame_counts', ())}`.",
            "Positive deltas mean the compressed model is worse than the baseline. Negative deltas mean the compressed refit improved that metric.",
            "AIC/BIC values are computed from the native row-space weighted chi-square and total nuisance/scale parameter count. Because endpoint sigmas are still audited separately, GROND uses information criteria as a parsimony diagnostic inside the strict residual gate rather than as the only promotion rule.",
            f"Candidate CSV: `{payload['candidate_csv']}`",
            f"Selected run-like directory: `{payload['selected_run_dir']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    """Refit simpler frame and detector scale models on the frozen Bloch rows and pick compact ones.

    Writes the candidate CSV, the selected compact and strict runs, scale_compression_summary.json and
    REPORT.md to --out-dir.
    """

    ap = build_compression_parser(
        description=__doc__,
    )
    args = ap.parse_args()

    row_table = Path(args.row_table).expanduser().resolve()
    rows = read_rows(row_table)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else row_table.parent / "summary.json"
    source_summary = read_json(summary_path)

    for required in (args.raw_dyn_column, args.raw_born_column, args.baseline_dyn_column, args.baseline_born_column):
        if required not in rows[0]:
            raise ValueError(f"Required column {required!r} is not present in {row_table}")

    obs = col(rows, "target_I")
    sig = source_sigma_array(rows)
    raw_dyn = col(rows, args.raw_dyn_column)
    raw_born = col(rows, args.raw_born_column)
    base_dyn = col(rows, args.baseline_dyn_column)
    groups0 = col(rows, "group").astype(int)
    det0 = col(rows, "detector_bin").astype(int)
    masks = split_masks(rows)
    fit_split = args.fit_split or str((source_summary.get("best_native") or {}).get("scale_fit_split") or "all")
    if fit_split not in masks:
        raise ValueError(f"Unknown fit split {fit_split!r}; available {sorted(masks)}")
    fit_mask = masks[fit_split]
    fit_good0 = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw_dyn) & (sig > 0) & (raw_dyn > 0)
    baseline_log_scale = baseline_scale_log_by_row(source_summary, groups0)
    if np.count_nonzero(np.isfinite(baseline_log_scale) & fit_good0) < 2:
        baseline_log_scale = baseline_scale_log_by_fit(raw_dyn, base_dyn, groups0, fit_good0)
    selection_metrics = parse_csv_metrics(args.selection_metrics)
    native_wght_a, native_wght_b = native_weight_from_summary(source_summary, args.native_wght_a, args.native_wght_b)
    base_scale_count, base_total_params, physical_params = load_baseline_counts(source_summary, len(rows))

    baseline_row_mr = row_mr_metrics(obs, sig, base_dyn, masks["all"], args.instability_u, args.observed_sigma_cut)
    baseline_row_work = row_mr_metrics(obs, sig, base_dyn, masks["work"], args.instability_u, args.observed_sigma_cut)
    baseline_row_free = row_mr_metrics(obs, sig, base_dyn, masks["free"], args.instability_u, args.observed_sigma_cut)
    baseline_native = native_metrics(obs, sig, base_dyn, masks["all"], native_wght_a, native_wght_b, n_params=base_total_params)
    group_width_deg = infer_group_width_deg(source_summary)
    source_detector_bins = int(
        safe_float(
            ((source_summary.get("best_native") or {}).get("nuisance") or {}).get(
                "detector_scale_bins"
            ),
            0.0,
        )
    )
    source_detector_params = (
        source_detector_bins * source_detector_bins if source_detector_bins > 1 else 0
    )
    baseline = {
        "name": "baseline_full_scale",
        "mode": "current_dyn_scaled",
        "objective": "existing",
        "group_stride": 1,
        "group_offset": 0,
        "window_deg": group_width_deg if np.isfinite(group_width_deg) else "",
        "detector_bins": "existing",
        "scale_params": base_scale_count,
        "group_scale_params": max(0, base_scale_count - source_detector_params),
        "detector_scale_params": source_detector_params,
        "total_params": base_total_params,
        **baseline_row_mr,
        "row_work": baseline_row_work,
        "row_free": baseline_row_free,
        "native": baseline_native,
        "information": information_criteria(baseline_native, base_total_params),
        "delta_vs_baseline": {
            "Robs": 0.0,
            "Rall": 0.0,
            "wRall": 0.0,
            "work_Rall": 0.0,
            "work_wRall": 0.0,
            "free_Rall": 0.0,
            "free_wRall": 0.0,
            "native_R1_gt": 0.0,
            "native_R1_all": 0.0,
            "native_wR2_all": 0.0,
        },
        "within_strict": True,
        "within_r_only": True,
        "detector_policy": {
            "status": "legacy_source_reference",
            "eligible": source_detector_bins < 8
            or bool(args.detector_independent_calibration),
            "detector_bins": source_detector_bins,
            "reasons": (
                []
                if source_detector_bins < 8 or bool(args.detector_independent_calibration)
                else ["eight_by_eight_requires_independent_calibration"]
            ),
        },
    }

    detector_base = infer_detector_base(det0)
    target_detector_bins = parse_detector_targets(args.detector_bins, detector_base)
    group_strides, window_scan = build_group_strides(args, source_summary)
    frame_models = parse_frame_models(args.frame_models)
    n_source_groups = int(len(set(groups0[fit_good0].astype(int)))) if np.count_nonzero(fit_good0) else int(len(set(groups0.astype(int))))
    adaptive_counts = parse_adaptive_counts(args.adaptive_frame_counts, n_source_groups) if "adaptive" in frame_models else []
    candidates: list[dict] = [baseline]

    def evaluate_grouping(
        *,
        name_prefix: str,
        frame_model: str,
        groups: np.ndarray,
        group_stride: int,
        group_offset: int,
        window_deg: float,
        adaptive_segments: int | str = "",
    ) -> None:
        """Fit scales for one frame grouping at each target detector grid and append the scored candidates.

        Each candidate carries row/MR and native metrics, information criteria, the detector policy, its
        deltas against the baseline and the strict and R-only tolerance flags.
        """

        fit_good = fit_mask & np.isfinite(obs) & np.isfinite(sig) & np.isfinite(raw_dyn) & (sig > 0) & (raw_dyn > 0)
        for det_bins_target in target_detector_bins:
            if det_bins_target <= 1:
                dyn_scaled, frame_scales = fit_group_scales(obs, sig, raw_dyn, groups, fit_mask, args.scale_objective, native_wght_a, native_wght_b, args.instability_u)
                det_scales: dict[str, float] = {}
                mode = "frame_groups"
                det_label = 0
                det_bins = np.zeros(len(rows), dtype=int)
            else:
                det_bins = coarsen_detector(det0, detector_base, det_bins_target)
                dyn_scaled, frame_scales, det_scales = fit_group_detector_scales(
                    obs, sig, raw_dyn, groups, det_bins, fit_mask,
                    args.scale_objective, native_wght_a, native_wght_b,
                    args.instability_u, detector_grid_size=det_bins_target,
                    detector_ridge=args.detector_ridge,
                    detector_smoothness=args.detector_smoothness,
                    detector_min_rows=args.detector_min_rows,
                )
                mode = "frame_groups_plus_detector"
                det_label = det_bins_target
            scale_params, n_group_scales, n_detector_scales = count_scale_params(groups, det_bins, fit_good, det_label)
            total_params = int(scale_params + physical_params)
            metrics = row_mr_metrics(obs, sig, dyn_scaled, masks["all"], args.instability_u, args.observed_sigma_cut)
            row_work = row_mr_metrics(obs, sig, dyn_scaled, masks["work"], args.instability_u, args.observed_sigma_cut)
            row_free = row_mr_metrics(obs, sig, dyn_scaled, masks["free"], args.instability_u, args.observed_sigma_cut)
            native = native_metrics(obs, sig, dyn_scaled, masks["all"], native_wght_a, native_wght_b, n_params=total_params)
            candidate = {
                "name": f"{name_prefix}_det_{det_label}",
                "frame_model": frame_model,
                "mode": mode,
                "objective": args.scale_objective,
                "group_stride": int(group_stride),
                "group_offset": int(group_offset),
                "adaptive_segments": adaptive_segments,
                "window_deg": window_deg if np.isfinite(window_deg) else "",
                "detector_bins": int(det_label),
                "scale_params": int(scale_params),
                "group_scale_params": int(n_group_scales),
                "detector_scale_params": int(n_detector_scales),
                "total_params": int(total_params),
                "fit_scale_log_step_rms": log_scale_step_rms(frame_scales),
                "baseline_within_group_log_rms": baseline_within_group_log_rms(baseline_log_scale, groups, fit_good),
                "frame_scales": frame_scales,
                "detector_scales": det_scales,
                **metrics,
                "row_work": row_work,
                "row_free": row_free,
                "native": native,
                "information": information_criteria(native, total_params),
                "arrays": {"dyn": dyn_scaled},
            }
            candidate["detector_policy"] = detector_scale_policy_diagnostics(
                rows=rows,
                obs=obs,
                sig=sig,
                raw_dyn=raw_dyn,
                raw_born=raw_born,
                groups=groups,
                source_detector_bins=det0,
                detector_base=detector_base,
                detector_target=det_label,
                fit_mask=fit_mask,
                objective=args.scale_objective,
                native_wght_a=native_wght_a,
                native_wght_b=native_wght_b,
                instability_u=args.instability_u,
                detector_ridge=args.detector_ridge,
                detector_smoothness=args.detector_smoothness,
                detector_min_rows=args.detector_min_rows,
                min_holdout_improvement=args.detector_min_holdout_improvement,
                required_improved_splits=args.detector_required_improved_splits,
                max_bloch_gain_loss=args.detector_max_bloch_gain_loss,
                min_field_correlation=args.detector_min_field_correlation,
                max_field_log_rms=args.detector_max_field_log_rms,
                independent_calibration=args.detector_independent_calibration,
            )
            candidate["delta_vs_baseline"] = metric_delta(candidate, baseline)
            candidate["within_strict"] = within_tolerance(candidate["delta_vs_baseline"], selection_metrics, args.tolerance)
            candidate["within_r_only"] = within_tolerance(
                candidate["delta_vs_baseline"],
                ("free_Rall", "work_Rall"),
                args.tolerance,
            )
            candidates.append(candidate)

    if "adaptive" in frame_models:
        for segments in adaptive_counts:
            groups, actual_segments = adaptive_segment_groups(groups0, baseline_log_scale, fit_good0, segments)
            evaluate_grouping(
                name_prefix=f"adaptive_{actual_segments}",
                frame_model="adaptive",
                groups=groups,
                group_stride=0,
                group_offset=0,
                window_deg=float("nan"),
                adaptive_segments=int(actual_segments),
            )

    if "fixed" in frame_models:
        for stride in group_strides:
            if args.scan_offsets and int(stride) < 10**8:  # noqa: SIM108 - clearer than an inline range expression
                offsets = range(max(1, int(stride)))
            else:
                offsets = range(1)
            for offset in offsets:
                groups = coarsen_groups(groups0, stride, offset=offset)
                window_deg = float(stride) * group_width_deg if int(stride) < 10**8 and np.isfinite(group_width_deg) else float("nan")
                stride_label = stride if stride < 10**8 else "global"
                offset_label = f"_offset_{offset}" if (args.scan_offsets and int(stride) < 10**8) else ""
                evaluate_grouping(
                    name_prefix=f"stride_{stride_label}{offset_label}",
                    frame_model="fixed",
                    groups=groups,
                    group_stride=int(stride),
                    group_offset=int(offset),
                    window_deg=window_deg,
                )

    non_baseline = [c for c in candidates if c["name"] != "baseline_full_scale"]
    strict = [
        c
        for c in non_baseline
        if c["within_strict"]
        and bool((c.get("detector_policy") or {}).get("eligible", False))
    ]
    r_only = [
        c
        for c in non_baseline
        if c["within_r_only"]
        and bool((c.get("detector_policy") or {}).get("eligible", False))
    ]
    strict_added_detector = [c for c in strict if int(c.get("detector_bins", 0)) > 1]
    strict_no_detector = [c for c in strict if int(c.get("detector_bins", 0)) <= 1]
    promotion_pool = strict_no_detector or strict_added_detector
    selected = min(
        promotion_pool,
        key=lambda c: (
            c["scale_params"],
            c["total_params"],
            safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
            safe_float((c.get("row_free") or {}).get("wRall"), float("inf")),
            safe_float((c.get("row_work") or {}).get("Rall"), float("inf")),
        ),
    ) if promotion_pool else None
    selected_information = min(
        strict,
        key=lambda c: (
            safe_float(c.get("information", {}).get("bic_fitted_variance"), float("inf")),
            safe_float(c.get("information", {}).get("bic_known_sigma"), float("inf")),
            c["scale_params"],
            safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
        ),
    ) if strict else None
    selected_r_only = min(
        r_only,
        key=lambda c: (
            c["scale_params"],
            c["total_params"],
            safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
            safe_float((c.get("row_work") or {}).get("Rall"), float("inf")),
        ),
    ) if r_only else None
    selected_widest = max(strict, key=lambda c: (safe_float(c.get("window_deg"), -1.0), c["group_stride"], -c["scale_params"], -c["wRall"])) if strict else None
    selected_widest_r_only = max(r_only, key=lambda c: (safe_float(c.get("window_deg"), -1.0), c["group_stride"], -c["scale_params"], -c["wRall"])) if r_only else None
    policy_eligible = [
        c
        for c in non_baseline
        if bool((c.get("detector_policy") or {}).get("eligible", False))
    ]
    selected_for_output = selected or selected_information or selected_r_only or min(
        policy_eligible or non_baseline,
        key=lambda c: (
            max(0.0, safe_float(c["delta_vs_baseline"].get("free_Rall"), float("inf"))),
            max(0.0, safe_float(c["delta_vs_baseline"].get("work_Rall"), float("inf"))),
            c["scale_params"],
        ),
    )
    candidate_rows = [
        candidate_row(c)
        for c in sorted(
            candidates,
            key=lambda c: (
                c["scale_params"],
                safe_float((c.get("row_free") or {}).get("Rall"), float("inf")),
                safe_float((c.get("row_free") or {}).get("wRall"), float("inf")),
            ),
        )
    ]
    candidate_csv = out / "scale_compression_candidates.csv"
    write_rows(candidate_csv, candidate_rows, list(candidate_rows[0].keys()))

    def arrays_for_candidate(candidate: dict) -> dict[str, np.ndarray]:
        """Rebuild one candidate's groups and detector bins, refit its Born scales and return the arrays."""

        if candidate.get("frame_model") == "adaptive":
            candidate_groups, _ = adaptive_segment_groups(
                groups0,
                baseline_log_scale,
                fit_good0,
                int(candidate.get("adaptive_segments") or candidate.get("group_scale_params") or 1),
            )
        else:
            candidate_groups = coarsen_groups(
                groups0,
                int(candidate["group_stride"]),
                int(candidate.get("group_offset", 0)),
            )
        detector_target = int(candidate["detector_bins"])
        if detector_target <= 1:
            candidate_detector = np.zeros(len(rows), dtype=int)
            born, _ = fit_group_scales(
                obs, sig, raw_born, candidate_groups, fit_mask, args.scale_objective,
                native_wght_a, native_wght_b, args.instability_u,
            )
        else:
            candidate_detector = coarsen_detector(det0, detector_base, detector_target)
            born, _, _ = fit_group_detector_scales(
                obs, sig, raw_born, candidate_groups, candidate_detector, fit_mask,
                args.scale_objective, native_wght_a, native_wght_b,
                args.instability_u, detector_grid_size=detector_target,
                detector_ridge=args.detector_ridge,
                detector_smoothness=args.detector_smoothness,
                detector_min_rows=args.detector_min_rows,
            )
        return {
            "dyn": candidate["arrays"]["dyn"],
            "born": born,
            "groups": np.asarray(candidate_groups, dtype=int),
            "detector_bins": np.asarray(candidate_detector, dtype=int),
        }

    def materialize_candidate_run(run_dir: Path, candidate: dict) -> None:
        """Write one candidate as a native run: its native_dyn_calc.csv and a summary.json with the new scales."""

        arrays = arrays_for_candidate(candidate)
        run_dir.mkdir(parents=True, exist_ok=True)
        updated_rows, fieldnames = materialize_candidate_rows(
            rows,
            arrays,
            candidate_name=candidate["name"],
            baseline_dyn_column=args.baseline_dyn_column,
            baseline_born_column=args.baseline_born_column,
        )
        write_rows(run_dir / "native_dyn_calc.csv", updated_rows, fieldnames)

        compact_summary = copy.deepcopy(source_summary)
        compact_summary["scale_compression"] = {
            "source_row_table": str(row_table),
            "source_summary": str(summary_path),
            "objective": args.scale_objective,
            "fit_split": fit_split,
            "tolerance": float(args.tolerance),
            "selection_metrics": selection_metrics,
            "frame_models": frame_models,
            "adaptive_frame_counts": adaptive_counts,
            "window_scan": window_scan,
            "scan_offsets": bool(args.scan_offsets),
            "baseline": {k: v for k, v in baseline.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
            "selected_strict": {k: v for k, v in (selected or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected else None,
            "selected_information": {k: v for k, v in (selected_information or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_information else None,
            "selected_r_only": {k: v for k, v in (selected_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_r_only else None,
            "selected_widest_strict": {k: v for k, v in (selected_widest or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest else None,
            "selected_widest_r_only": {k: v for k, v in (selected_widest_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest_r_only else None,
            "selected_output": {k: v for k, v in candidate.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
            "candidate_csv": str(candidate_csv),
            "detector_policy": candidate.get("detector_policy") or {},
        }
        compact_summary.setdefault("best_native", {})
        compact_summary["best_native"]["scale_count"] = int(candidate["scale_params"])
        compact_summary["best_native"]["frame_scale_count"] = int(
            candidate.get("group_scale_params", 0)
        )
        compact_summary["best_native"]["detector_scale_count"] = int(
            candidate.get("detector_scale_params", 0)
        )
        compact_summary["best_native"]["scale_fit_split"] = fit_split
        compact_summary["best_native"]["metrics"] = {
            split: {
                kind: native_metrics(
                    obs, sig, arrays[kind], masks[split], native_wght_a, native_wght_b,
                    n_params=int(candidate["total_params"]),
                )
                for kind in ("dyn", "born")
            }
            for split in ("all", "work", "free")
        }
        compact_summary["best_native"]["nuisance"] = {
            **((source_summary.get("best_native") or {}).get("nuisance") or {}),
            "scale_compression_model": candidate["name"],
            "scale_compression_objective": args.scale_objective,
            "scale_compression_group_stride": int(candidate["group_stride"]),
            "scale_compression_detector_bins": int(candidate["detector_bins"]),
        }
        (run_dir / "summary.json").write_text(json.dumps(compact_summary, indent=2, allow_nan=True) + "\n")

    selected_dir = out / "selected_compact_run"
    materialize_candidate_run(selected_dir, selected_for_output)
    selected_strict_dir = out / "selected_strict_run"
    if selected:
        materialize_candidate_run(selected_strict_dir, selected)
    if summary_path.exists():
        shutil.copy2(summary_path, out / "source_summary.json")

    payload = {
        "title": args.title,
        "row_table": str(row_table),
        "summary": str(summary_path),
        "candidate_csv": str(candidate_csv),
        "selected_run_dir": str(selected_dir),
        "selected_strict_run_dir": str(selected_strict_dir) if selected else "",
        "tolerance": float(args.tolerance),
        "selection_metrics": selection_metrics,
        "frame_models": frame_models,
        "adaptive_frame_counts": adaptive_counts,
        "fit_split": fit_split,
        "detector_base_bins": int(detector_base),
        "detector_policy": {
            "candidate_grids": target_detector_bins,
            "ridge_to_unity": float(args.detector_ridge),
            "spatial_smoothness": float(args.detector_smoothness),
            "minimum_rows_per_tile": int(args.detector_min_rows),
            "minimum_holdout_improvement": float(args.detector_min_holdout_improvement),
            "required_improved_splits": int(args.detector_required_improved_splits),
            "maximum_bloch_gain_loss": float(args.detector_max_bloch_gain_loss),
            "minimum_field_correlation": float(args.detector_min_field_correlation),
            "maximum_field_log_rms": float(args.detector_max_field_log_rms),
            "independent_calibration": bool(args.detector_independent_calibration),
        },
        "physical_params_carried_from_source": int(physical_params),
        "window_scan": window_scan,
        "scan_offsets": bool(args.scan_offsets),
        "baseline": {k: v for k, v in baseline.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
        "selected": {k: v for k, v in (selected or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected else None,
        "selected_information": {k: v for k, v in (selected_information or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_information else None,
        "selected_r_only": {k: v for k, v in (selected_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_r_only else None,
        "selected_widest": {k: v for k, v in (selected_widest or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest else None,
        "selected_widest_r_only": {k: v for k, v in (selected_widest_r_only or {}).items() if k not in {"arrays", "frame_scales", "detector_scales"}} if selected_widest_r_only else None,
        "selected_output": {k: v for k, v in selected_for_output.items() if k not in {"arrays", "frame_scales", "detector_scales"}},
        "top_candidates": [candidate_row(c) for c in sorted(candidates, key=lambda c: (c["scale_params"], c["wRall"], c["Rall"]))[:20]],
    }
    (out / "scale_compression_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    write_report(out / "REPORT.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
