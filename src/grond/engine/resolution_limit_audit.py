#!/usr/bin/env python3
"""Promote resolution limits from final GROND row tables.

The audit works on row-preserving ``native_dyn_calc.csv`` files. It builds
resolution shells, calculates GROND row/MR unmerged and merged residual factors
for each shell and cumulative cutoff, checks held-out shell stability, and
writes a promoted strict cutoff plus a looser exploratory cutoff.

The weighted-residual gates (shell ``MwRall``, cumulative penalty, work/free gap)
read the intensity-residual form ``MwRall_I``, against which the thresholds were
set; Jana2020's amplitude form ``MwRall`` is reported alongside and does not
enter any decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from ..forensics.resolution import promote_resolution_limit
    from .row_mr_report import (
        canonical_hkl,
        compute_model_metrics,
        compute_row_mr_summary,
        parse_model,
        read_csv_rows,
        safe_float,
    )
else:  # pragma: no cover - direct script execution
    import sys

    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from row_mr_report import (
        canonical_hkl,
        compute_model_metrics,
        compute_row_mr_summary,
        parse_model,
        read_csv_rows,
        safe_float,
    )

    from grond.forensics.resolution import promote_resolution_limit


def parse_named_path(text: str) -> tuple[str, Path]:
    """Parse name=/path or infer a stable name from the path."""

    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(text).expanduser()
    return path.parent.name or path.stem, path


def read_input_rows(run_items: list[str]) -> list[dict]:
    """Read the row tables (native_dyn_calc.csv) of the run directories."""

    rows: list[dict] = []
    for item in run_items:
        name, run_dir = parse_named_path(item)
        rows.extend(read_csv_rows((run_dir / "native_dyn_calc.csv").resolve(), name))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    """Format a numeric value for Markdown."""

    v = safe_float(value)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def pct(value: object, digits: int = 2) -> str:
    """Format a fractional metric as a percentage."""

    v = safe_float(value)
    if not math.isfinite(v):
        return "n/a"
    return f"{100.0 * v:.{digits}f}%"


def row_d(row: dict) -> float:
    """Return a safe positive d-spacing for sorting."""

    d = safe_float(row.get("d_A"))
    return d if math.isfinite(d) and d > 0 else float("nan")


def shell_groups(rows: list[dict], n_shells: int) -> list[list[dict]]:
    """Split rows into low-to-high resolution shells by equal row count."""

    valid = [r for r in rows if math.isfinite(row_d(r))]
    valid.sort(key=row_d, reverse=True)
    if not valid:
        return []
    return [[valid[i] for i in idx] for idx in np.array_split(np.arange(len(valid)), max(1, int(n_shells))) if len(idx)]


def stable_split_id(row: dict) -> int:
    """Deterministically split rows without depending on Python hash randomization."""

    text = "|".join(str(row.get(k, "")) for k in ("dataset", "h", "k", "l", "row_index"))
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 2_147_483_647
    return value & 1


def merged_obs_by_split(rows: list[dict], model: dict) -> tuple[int, float]:
    """Return common merged split groups and their observation CC."""

    buckets: dict[tuple[int, int, int], dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        try:
            key = canonical_hkl(int(float(row["h"])), int(float(row["k"])), int(float(row["l"])), model.get("symmetry_ops", []))
        except Exception:
            continue
        obs = safe_float(row.get("target_I"))
        sig = safe_float(row.get("sigma_eff"))
        if not (math.isfinite(obs) and math.isfinite(sig) and sig > 0):
            continue
        buckets[key][stable_split_id(row)].append(obs)

    left: list[float] = []
    right: list[float] = []
    for split_rows in buckets.values():
        if split_rows[0] and split_rows[1]:
            left.append(float(np.mean(split_rows[0])))
            right.append(float(np.mean(split_rows[1])))
    if len(left) < 3:
        return len(left), float("nan")
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return len(left), float("nan")
    return len(left), float(np.corrcoef(a, b)[0, 1])


def metrics_or_empty(rows: list[dict], model: dict, calc_column: str) -> dict:
    """Calculate GROND row/MR metrics without failing on an empty row list."""

    if not rows:
        return {
            "Nall": 0,
            "Nobs": 0,
            "Rall": float("nan"),
            "wRall": float("nan"),
            "wRall_I": float("nan"),
            "MRall": float("nan"),
            "MwRall": float("nan"),
            "MwRall_I": float("nan"),
        }
    return compute_model_metrics(rows, model, calc_column)


def matched_metrics_or_empty(rows: list[dict], model: dict) -> tuple[dict, dict, dict]:
    """Return dynamic/Born metrics on one common finite support plus its audit counts."""

    if not rows:
        empty = {
            "Ninput": 0,
            "Nmatched": 0,
            "Nexcluded": 0,
            "Nall": 0,
            "Nobs": 0,
            "Rall": float("nan"),
            "wRall": float("nan"),
            "wRall_I": float("nan"),
            "MRall": float("nan"),
            "MwRall": float("nan"),
            "MwRall_I": float("nan"),
            "Nmerge_input": 0,
            "Nmerge_used": 0,
            "Nmerge_excluded": 0,
        }
        return dict(empty), dict(empty), {"Ninput": 0, "Nmatched": 0, "Nexcluded": 0, "status": "empty_matched_support"}
    summary = compute_row_mr_summary(rows, model)
    return summary["dynamic"], summary["born"], summary["support"]


def shell_record(shell_no: int, rows: list[dict], cumulative_rows: list[dict], model: dict, *, min_common_pairs: int) -> dict:
    """Calculate one shell record including cumulative GROND row/MR metrics."""

    dvals = np.asarray([row_d(r) for r in rows], dtype=float)
    obs = np.asarray([safe_float(r.get("target_I")) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r.get("sigma_eff")) for r in rows], dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sig) & (sig > 0)
    isigi = obs[valid] / np.maximum(sig[valid], 1e-12) if np.any(valid) else np.asarray([], dtype=float)
    strong = (obs > 3.0 * sig) & valid
    split_common, split_cc = merged_obs_by_split(rows, model)
    dyn, born, support = matched_metrics_or_empty(rows, model)
    cum_dyn, cum_born, cumulative_support = matched_metrics_or_empty(cumulative_rows, model)
    work = [r for r in rows if str(r.get("split", "")).lower() == "work"]
    free = [r for r in rows if str(r.get("split", "")).lower() == "free"]
    work_dyn = metrics_or_empty(work, model, "dyn_scaled")
    free_dyn = metrics_or_empty(free, model, "dyn_scaled")

    def mwr_gap(key: str) -> float:
        """Return |free - work| of the merged weighted factor ``key``, NaN unless both sets have rows."""

        value = abs(safe_float(free_dyn.get(key)) - safe_float(work_dyn.get(key)))
        if not (math.isfinite(value) and work_dyn.get("Nall", 0) and free_dyn.get("Nall", 0)):
            return float("nan")
        return value

    # The gates read the intensity-residual form (*_I), against which the thresholds were set; the amplitude
    # form (Jana2020 wR(all) for refinement on F) is reported alongside.
    gap = mwr_gap("MwRall_I")
    gap_amplitude = mwr_gap("MwRall")

    stability_mode = "split_cc" if split_common >= int(min_common_pairs) else "work_free_MwRall_I"
    return {
        "shell": int(shell_no),
        "d_low": float(np.nanmin(dvals)) if len(dvals) else float("nan"),
        "d_high": float(np.nanmax(dvals)) if len(dvals) else float("nan"),
        "n_rows": int(np.count_nonzero(valid)),
        "n_strong": int(np.count_nonzero(strong)),
        "strong_fraction": float(np.count_nonzero(strong) / max(np.count_nonzero(valid), 1)),
        "mean_I_over_sigma": float(np.nanmean(isigi)) if len(isigi) else float("nan"),
        "median_I_over_sigma": float(np.nanmedian(isigi)) if len(isigi) else float("nan"),
        "split_common": int(split_common),
        "split_cc": float(split_cc),
        "stability_mode": stability_mode,
        "matched_Ninput": int(support.get("Ninput", 0)),
        "matched_Nall": int(support.get("Nmatched", 0)),
        "matched_Nexcluded": int(support.get("Nexcluded", 0)),
        "dynamic_Nmerge_excluded": int(dyn.get("Nmerge_excluded", 0)),
        "kinematic_Nmerge_excluded": int(born.get("Nmerge_excluded", 0)),
        "dynamic_Rall": dyn.get("Rall", float("nan")),
        "dynamic_wRall": dyn.get("wRall", float("nan")),
        "dynamic_wRall_I": dyn.get("wRall_I", float("nan")),
        "dynamic_MRall": dyn.get("MRall", float("nan")),
        "dynamic_MwRall": dyn.get("MwRall", float("nan")),
        "dynamic_MwRall_I": dyn.get("MwRall_I", float("nan")),
        "kinematic_Rall": born.get("Rall", float("nan")),
        "kinematic_wRall": born.get("wRall", float("nan")),
        "kinematic_wRall_I": born.get("wRall_I", float("nan")),
        "kinematic_MRall": born.get("MRall", float("nan")),
        "kinematic_MwRall": born.get("MwRall", float("nan")),
        "kinematic_MwRall_I": born.get("MwRall_I", float("nan")),
        "cumulative_rows": int(cum_dyn.get("Nall", 0)),
        "cumulative_matched_Ninput": int(cumulative_support.get("Ninput", 0)),
        "cumulative_matched_Nexcluded": int(cumulative_support.get("Nexcluded", 0)),
        "cumulative_dynamic_Rall": cum_dyn.get("Rall", float("nan")),
        "cumulative_dynamic_wRall": cum_dyn.get("wRall", float("nan")),
        "cumulative_dynamic_wRall_I": cum_dyn.get("wRall_I", float("nan")),
        "cumulative_dynamic_MRall": cum_dyn.get("MRall", float("nan")),
        "cumulative_dynamic_MwRall": cum_dyn.get("MwRall", float("nan")),
        "cumulative_dynamic_MwRall_I": cum_dyn.get("MwRall_I", float("nan")),
        "cumulative_kinematic_MwRall": cum_born.get("MwRall", float("nan")),
        "cumulative_kinematic_MwRall_I": cum_born.get("MwRall_I", float("nan")),
        "work_Nall": int(work_dyn.get("Nall", 0)),
        "free_Nall": int(free_dyn.get("Nall", 0)),
        "work_dynamic_MwRall": work_dyn.get("MwRall", float("nan")),
        "work_dynamic_MwRall_I": work_dyn.get("MwRall_I", float("nan")),
        "free_dynamic_MwRall": free_dyn.get("MwRall", float("nan")),
        "free_dynamic_MwRall_I": free_dyn.get("MwRall_I", float("nan")),
        "free_work_MwRall_gap": gap_amplitude,
        "free_work_MwRall_I_gap": gap,
    }


def shell_records(rows: list[dict], model: dict, *, n_shells: int, min_common_pairs: int) -> list[dict]:
    """Build all low-to-high shell records."""

    records: list[dict] = []
    cumulative: list[dict] = []
    for i, shell in enumerate(shell_groups(rows, n_shells), start=1):
        cumulative.extend(shell)
        records.append(shell_record(i, shell, list(cumulative), model, min_common_pairs=min_common_pairs))
    return records


def add_decision_flags(shells: list[dict], promoted_last: int | None, exploratory_last: int | None) -> list[dict]:
    """Add report flags to shell records."""

    out: list[dict] = []
    for row in shells:
        r = dict(row)
        shell_no = int(r.get("shell", 0))
        r["inside_promoted_limit"] = bool(promoted_last is not None and shell_no <= promoted_last)
        r["inside_exploratory_limit"] = bool(exploratory_last is not None and shell_no <= exploratory_last)
        out.append(r)
    return out


def write_report(path: Path, title: str, payload: dict) -> None:
    """Write a concise Markdown report for the decision."""

    decision = payload["decision"]
    shells = payload["shells"]
    lines = [
        f"# {title}",
        "",
        "This audit promotes a resolution limit from final GROND row tables. The strict limit requires usable signal, held-out shell stability, and acceptable GROND row/MR merged weighted residuals. The weighted-residual gates read the intensity-residual form `MwRall_I`, against which the thresholds were set; `MwRall` is Jana2020's amplitude form (wR(all) for refinement on F) and is reported only. The exploratory limit is looser and is meant for diagnostic maps or sensitivity checks.",
        "",
        "## Decision",
        "",
        f"Promoted dmin: `{fmt(decision.get('promoted_min_d_spacing'))} A`.",
        f"Exploratory dmin: `{fmt(decision.get('exploratory_min_d_spacing'))} A`.",
        f"First rejection: `{decision.get('first_rejection') or 'none'}`.",
        "",
        "## Shells",
        "",
        "Born and Bloch shell factors use one common finite row support. `matched/excluded` reports supported rows versus supplied rows removed before both calculations; `merge excluded` reports matched rows whose HKL indices cannot enter symmetry merging.",
        "",
        "| shell | d range A | matched/excluded | merge excluded | I/sigma mean | split CC | dyn MRall | dyn MwRall | dyn MwRall_I (gate) | cumulative MwRall_I (gate) | kept |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in shells:
        kept = "promoted" if row.get("inside_promoted_limit") else ("exploratory" if row.get("inside_exploratory_limit") else "rejected")
        lines.append(
            f"| {row['shell']} | {fmt(row['d_high'])}-{fmt(row['d_low'])} | "
            f"{row['matched_Nall']}/{row['matched_Nexcluded']} | {row['dynamic_Nmerge_excluded']} | "
            f"{fmt(row['mean_I_over_sigma'])} | {fmt(row['split_cc'])} | {pct(row['dynamic_MRall'])} | "
            f"{pct(row['dynamic_MwRall'])} | {pct(row['dynamic_MwRall_I'])} | {pct(row['cumulative_dynamic_MwRall_I'])} | {kept} |"
        )
    lines.extend(
        [
            "",
            "Artifacts: `resolution_shells.csv`, `summary.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


# Shells and promotion thresholds of the held-out resolution-limit gate.
RESOLUTION_SHELLS = 10
THRESHOLDS = {
    "strict_split_cc": 0.30,
    "strict_mean_i_over_sigma": 0.50,
    "exploratory_split_cc": 0.20,
    "exploratory_mean_i_over_sigma": 0.25,
    "min_common_pairs": 8,
    "min_shell_rows": 20,
    "min_strong_fraction": 0.005,
    "max_shell_mwrall": 0.60,
    "max_free_work_mwr_gap": 0.12,
    "min_free_rows": 5,
    "max_cumulative_mwr_penalty": 0.03,
}


def main() -> None:
    """Build resolution shells from the row tables and promote strict and exploratory cutoffs.

    Writes resolution_shells.csv, summary.json and REPORT.md to --out-dir.
    """

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", default=[], help="Dataset run as name=/path/to/native_nuisance_run.")
    ap.add_argument("--res", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="GROND resolution-limit promotion")
    ap.add_argument("--scope", default="dataset", choices=["dataset", "merged"])
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model = parse_model(Path(args.res).expanduser())
    rows = read_input_rows(args.run)
    if not rows:
        raise SystemExit("No row tables were supplied.")

    shells = shell_records(rows, model, n_shells=RESOLUTION_SHELLS, min_common_pairs=THRESHOLDS["min_common_pairs"])
    decision = promote_resolution_limit(
        shells,
        **{("min_rows" if key == "min_shell_rows" else key): value for key, value in THRESHOLDS.items()},
    )
    decision_dict = decision.to_dict()
    shells = add_decision_flags(shells, decision.promoted_last_shell, decision.exploratory_last_shell)
    overall_dynamic, overall_kinematic, overall_support = matched_metrics_or_empty(rows, model)
    payload = {
        "status": "ok",
        "title": args.title,
        "scope": args.scope,
        "model": str(Path(args.res).expanduser().resolve()),
        "row_count": len(rows),
        "datasets": sorted(set(str(r.get("dataset", "")) for r in rows if r.get("dataset", ""))),
        "decision": decision_dict,
        "overall": {
            "support": overall_support,
            "dynamic": overall_dynamic,
            "kinematic": overall_kinematic,
        },
        "shells": shells,
        "thresholds": dict(THRESHOLDS),
        "artifacts": {
            "summary": str((out / "summary.json").resolve()),
            "report": str((out / "REPORT.md").resolve()),
            "shells_csv": str((out / "resolution_shells.csv").resolve()),
        },
    }
    write_csv(out / "resolution_shells.csv", shells)
    (out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    write_report(out / "REPORT.md", args.title, payload)
    print(json.dumps({"out_dir": str(out.resolve()), "decision": decision_dict, "report": str((out / "REPORT.md").resolve())}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
