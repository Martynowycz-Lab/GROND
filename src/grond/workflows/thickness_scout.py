"""Summarize thickness candidates and compare their row-wise dynamical correction fields."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import FULL_Q_LAMBDA, try_read_json


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries, returning an empty list when absent."""

    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float_from_mapping(mapping: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    """Read a finite-ish float value from a nested report mapping."""

    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def _thickness_summary_row(
    *,
    dataset: str,
    feature_mode: str,
    n_basis: int,
    n_points: int,
    run_dir: Path,
    thickness_grid: str,
    gradient_grid: str,
) -> dict[str, object]:
    """Extract the compact promoted-thickness row from one scout run."""

    summary_path = run_dir / "summary.json"
    summary = try_read_json(summary_path)
    best_native = summary.get("best_native") or {}
    nuisance = best_native.get("nuisance") or {}
    metrics = best_native.get("metrics") or {}
    dyn_all = (metrics.get("all") or {}).get("dyn") or {}
    dyn_work = (metrics.get("work") or {}).get("dyn") or {}
    dyn_free = (metrics.get("free") or {}).get("dyn") or {}
    born_all = (metrics.get("all") or {}).get("born") or {}
    row_selection = summary.get("row_selection") or {}
    thickness_A = _float_from_mapping(nuisance, "thickness_A")
    return {
        "dataset": dataset,
        "feature_mode": feature_mode,
        "n_basis": int(n_basis),
        "n_points": int(n_points),
        "status": "ok" if summary else "missing_summary",
        "thickness_A": thickness_A,
        "thickness_nm": thickness_A / 10.0 if math.isfinite(thickness_A) else float("nan"),
        "thickness_model": nuisance.get("thickness_model", ""),
        "thickness_grad_frac": _float_from_mapping(nuisance, "thickness_grad_frac"),
        "detector_scale_bins": nuisance.get("detector_scale_bins", ""),
        "group_width_deg": nuisance.get("group_width_deg", ""),
        "sg_jitter_mode": nuisance.get("sg_jitter_mode", ""),
        "absorption_mode": nuisance.get("absorption_mode", ""),
        "scale_count": best_native.get("scale_count", ""),
        "valid_rows": best_native.get("valid_rows", ""),
        "selected_rows": row_selection.get("selected_rows", ""),
        "work_rows": row_selection.get("work_rows", ""),
        "free_rows": row_selection.get("free_rows", ""),
        "all_dyn_R1": dyn_all.get("R1_all", ""),
        "all_dyn_R1gt": dyn_all.get("R1_gt", ""),
        "all_dyn_wR2": dyn_all.get("wR2_all", ""),
        "all_dyn_CC": dyn_all.get("CC", ""),
        "work_dyn_wR2": dyn_work.get("wR2_all", ""),
        "free_dyn_wR2": dyn_free.get("wR2_all", ""),
        "all_born_R1": born_all.get("R1_all", ""),
        "all_born_wR2": born_all.get("wR2_all", ""),
        "wall_seconds": summary.get("wall_seconds", ""),
        "thickness_grid_A": thickness_grid,
        "gradient_grid": gradient_grid,
        "run_dir": str(run_dir.resolve()),
        "summary_json": str(summary_path.resolve()),
        "row_table": str((run_dir / "native_dyn_calc.csv").resolve()),
        "nuisance_trials_csv": str((run_dir / "nuisance_trials.csv").resolve()),
    }


def _row_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Stable key for comparing two selected row correction fields."""

    return (
        str(row.get("row_index", "")),
        str(row.get("h", "")),
        str(row.get("k", "")),
        str(row.get("l", "")),
    )


def _logq_by_row(path: Path) -> dict[tuple[str, str, str, str], float]:
    """Load log(q) values from a GROND native row table."""

    out: dict[tuple[str, str, str, str], float] = {}
    for row in _read_csv_dicts(path):
        try:
            q = float(row.get("q_dyn_over_born", "nan"))
        except ValueError:
            continue
        if math.isfinite(q) and q > 0.0:
            out[_row_key(row)] = math.log(q)
    return out


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute a small dependency-free Pearson correlation."""

    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy, strict=False)) / (sx * sy)


def _p95(values: list[float]) -> float:
    """Return an empirical 95th percentile for a non-empty list."""

    if not values:
        return float("nan")
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(math.ceil(0.95 * len(vals))) - 1))
    return vals[idx]


def _q_field_stability(reference_csv: Path, candidate_csv: Path) -> dict[str, object]:
    """Compare two selected q fields by shared signed row identity."""

    ref = _logq_by_row(reference_csv)
    cand = _logq_by_row(candidate_csv)
    keys = sorted(set(ref) & set(cand))
    ref_vals = [ref[k] for k in keys]
    cand_vals = [cand[k] for k in keys]
    diffs = [abs(a - b) for a, b in zip(ref_vals, cand_vals, strict=False)]
    return {
        "reference_row_table": str(reference_csv.resolve()),
        "candidate_row_table": str(candidate_csv.resolve()),
        "shared_rows": len(keys),
        "logq_corr": _pearson(ref_vals, cand_vals),
        "median_abs_dlogq": sorted(diffs)[len(diffs) // 2] if diffs else float("nan"),
        "p95_abs_dlogq": _p95(diffs),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write heterogeneous dictionaries as a rectangular CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_thickness_scout_report(out_dir: Path, summary: dict[str, object]) -> None:
    """Write the human-readable thickness-scout report."""

    rows = list(summary.get("runs", []))
    q_rows = list(summary.get("q_field_stability", []))
    promoted = summary.get("promoted_reference") or {}
    warnings = list(summary.get("warnings", []))

    def pct(value: object) -> str:
        """Format a fraction as a percentage with two decimals; ``n/a`` when not finite."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        return f"{100.0 * v:.2f}%"

    def num(value: object, digits: int = 3) -> str:
        """Format a number with ``digits`` decimals; ``n/a`` when not finite."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    lines = [
        "# GROND Thickness Scout",
        "",
        "GROND thickness is an effective dynamical thickness selected for a declared signed-row contract, scale family, support size, and nuisance-feature state. It is not a direct microscope thickness unless an independent thickness measurement or a cross-contract stability test supports that interpretation.",
        "",
        "This report is diagnostic. `GROND auto` should select thickness under the simpler early row contract, pin it through the LS1 loop, and treat final-feature thickness movement as an identifiability warning rather than an automatic promotion.",
        "",
    ]
    if promoted:
        lines.extend(
            [
                "## Promoted Reference",
                "",
                f"The promoted reference for q-field comparison is `{promoted.get('dataset', '')}` / `{promoted.get('feature_mode', '')}` at N/Q `{promoted.get('n_basis', '')}/{promoted.get('n_points', '')}` with thickness `{num(promoted.get('thickness_nm'))}` nm and gradient `{num(promoted.get('thickness_grad_frac'))}`.",
                "",
            ]
        )
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(
        [
            "## Selected Runs",
            "",
            "| dataset | features | N/Q | thickness nm | grad | all R1 | all wR2 | free wR2 | q corr | p95 | run |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    q_by_run = {row.get("candidate_row_table"): row for row in q_rows}
    for row in rows:
        q = q_by_run.get(str(row.get("row_table", "")), {})
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('feature_mode', '')} | "
            f"{row.get('n_basis', '')}/{row.get('n_points', '')} | {num(row.get('thickness_nm'))} | "
            f"{num(row.get('thickness_grad_frac'))} | {pct(row.get('all_dyn_R1'))} | {pct(row.get('all_dyn_wR2'))} | "
            f"{pct(row.get('free_dyn_wR2'))} | {num(q.get('logq_corr'))} | {num(q.get('p95_abs_dlogq'))} | "
            f"`{Path(str(row.get('run_dir', ''))).name}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If final-feature activation moves the selected thickness while the row statistics remain similar, the thickness is not uniquely identified by the scalar row contract. In that case the stable quantity to trust is the row-wise Bloch/Born correction field and the final row and endpoint validation, not the absolute thickness number.",
            "",
            "The combined trial table and machine-readable summary are written next to this report.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def _thickness_scout_variant_args(
    args: argparse.Namespace, *, feature_mode: str, thickness_grid_A: str, gradient_grid: str
) -> argparse.Namespace:
    """Return a copy of args with one scout feature contract pinned."""

    variant = argparse.Namespace(**vars(args))
    variant.thickness_grid = thickness_grid_A
    variant.thickness_grad_grid = gradient_grid
    variant.thickness_model_grid = "uniform_gradient"
    variant.thickness_points = int(getattr(args, "thickness_scout_thickness_points", args.thickness_points))
    variant.lambda_grid = FULL_Q_LAMBDA
    variant.endpoint_b_grid = "0.0"
    variant.endpoint_exti_grid = "off"
    variant.shelxl_iterations = 1
    variant.native_only = True
    variant.make_maps = False
    variant.max_rows = int(getattr(args, "thickness_scout_max_rows", 0) or 0)
    variant.sigma_a_grid = "0.0"
    variant.sigma_b_corr_grid = "0.0"
    variant.sigma_c_peak_grid = "0.0"
    variant.row_resolution_b_grid = "0.0"
    variant.dose_b_mode_grid = "none"
    variant.dose_b_per_frame_grid = "0.0"
    if feature_mode == "bare":
        variant.sg_jitter_mode_grid = "none"
        variant.sg_jitter_sigma0_grid = "0.0"
        variant.sg_jitter_sigma_d_inv_grid = "0.0"
        variant.sg_jitter_sigma_frame_grid = "0.0"
        variant.sg_jitter_points = 1
        variant.absorption_mode_grid = "none"
        variant.absorption_mu_grid = "0.0"
        variant.absorption_frac_grid = "0.0"
        variant.absorption_mu_g2_grid = "0.0"
        variant.absorption_mu_theta2_grid = "0.0"
        variant.absorption_tds_b_grid = "0.0"
    elif feature_mode in {"final", "final-diagnostic", "final_diagnostic"}:
        variant.sg_jitter_mode_grid = args.thickness_scout_final_sg_jitter_mode
        variant.sg_jitter_sigma0_grid = str(args.thickness_scout_final_sg_jitter_sigma0)
        variant.sg_jitter_sigma_d_inv_grid = str(args.thickness_scout_final_sg_jitter_sigma_d_inv)
        variant.sg_jitter_sigma_frame_grid = str(args.thickness_scout_final_sg_jitter_sigma_frame)
        variant.sg_jitter_points = int(args.thickness_scout_final_sg_jitter_points)
        variant.absorption_mode_grid = args.thickness_scout_final_absorption_mode
        variant.absorption_mu_grid = str(args.thickness_scout_final_absorption_mu)
        variant.absorption_frac_grid = str(args.thickness_scout_final_absorption_frac)
        variant.absorption_mu_g2_grid = str(args.thickness_scout_final_absorption_mu_g2)
        variant.absorption_mu_theta2_grid = str(args.thickness_scout_final_absorption_mu_theta2)
        variant.absorption_tds_b_grid = str(args.thickness_scout_final_absorption_tds_b)
    else:
        raise SystemExit(f"Unknown thickness-scout feature mode {feature_mode!r}. Use bare or final-diagnostic.")
    return variant
