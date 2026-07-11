from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path


def safe_float(value: object, default: float = float("nan")) -> float:
    """Parse a finite float without letting bad metadata stop policy decisions."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a floating-point value into a closed interval."""

    return max(float(lo), min(float(hi), float(value)))


def read_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dictionary when unavailable."""

    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def read_row_table(path: Path, dataset: str) -> list[dict]:
    """Read a native row table and attach the dataset name."""

    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    for row in rows:
        row.setdefault("dataset", dataset)
    return rows


def write_row_table(path: Path, rows: list[dict]) -> None:
    """Write a row table while preserving the first-row field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    extras = sorted({key for row in rows for key in row if key not in fields})
    fields.extend(extras)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hkl_key(row: Mapping) -> tuple[int, int, int] | None:
    """Return a signed HKL key for coverage accounting."""

    try:
        return (int(float(row["h"])), int(float(row["k"])), int(float(row["l"])))
    except Exception:
        return None


def simple_metrics(rows: list[dict]) -> dict:
    """Calculate simple native row metrics when a run summary is incomplete."""

    obs: list[float] = []
    sig: list[float] = []
    calc: list[float] = []
    for row in rows:
        iobs = safe_float(row.get("target_I"))
        isig = safe_float(row.get("sigma_eff"))
        ipred = safe_float(row.get("dyn_scaled"))
        if math.isfinite(iobs) and math.isfinite(isig) and math.isfinite(ipred) and isig > 0:
            obs.append(iobs)
            sig.append(isig)
            calc.append(ipred)
    if not obs:
        return {"R1_all": float("nan"), "wR2_all": float("nan"), "CC": float("nan"), "n": 0}
    fobs = [math.sqrt(max(v, 0.0)) for v in obs]
    fcalc = [math.sqrt(max(v, 0.0)) for v in calc]
    r1 = sum(abs(a - b) for a, b in zip(fobs, fcalc, strict=True)) / max(sum(fobs), 1e-30)
    weights = [1.0 / max(s * s, 1e-30) for s in sig]
    wr2 = math.sqrt(
        sum(w * (o - c) ** 2 for o, c, w in zip(obs, calc, weights, strict=True))
        / max(sum(w * o * o for o, w in zip(obs, weights, strict=True)), 1e-30)
    )
    mean_obs = sum(obs) / len(obs)
    mean_calc = sum(calc) / len(calc)
    var_obs = sum((v - mean_obs) ** 2 for v in obs)
    var_calc = sum((v - mean_calc) ** 2 for v in calc)
    cc = float("nan")
    if len(obs) > 2 and var_obs > 0 and var_calc > 0:
        cc = sum((o - mean_obs) * (c - mean_calc) for o, c in zip(obs, calc, strict=True)) / math.sqrt(var_obs * var_calc)
    return {"R1_all": r1, "wR2_all": wr2, "CC": cc, "n": len(obs)}


def summary_metrics(summary: Mapping, rows: list[dict]) -> dict:
    """Extract the best available all-row dynamic metrics for one dataset."""

    dyn = (((summary.get("best_native") or {}).get("metrics") or {}).get("all") or {}).get("dyn") or {}
    fallback = simple_metrics(rows)
    return {
        "R1_all": safe_float(dyn.get("R1_all"), fallback["R1_all"]),
        "wR2_all": safe_float(dyn.get("wR2_all"), fallback["wR2_all"]),
        "CC": safe_float(dyn.get("CC"), fallback["CC"]),
        "n": int(safe_float(dyn.get("n"), fallback["n"])),
    }


def collect_dataset_stats(runs: Mapping[str, Path]) -> list[dict]:
    """Collect quality and coverage statistics from final or scout run directories."""

    stats: list[dict] = []
    for name, run_dir in runs.items():
        run = Path(run_dir)
        rows = read_row_table(run / "native_dyn_calc.csv", name)
        summary = read_json(run / "summary.json")
        metrics = summary_metrics(summary, rows)
        keys = {key for row in rows if (key := hkl_key(row)) is not None}
        dvals = [safe_float(row.get("d_A")) for row in rows]
        finite_d = [d for d in dvals if math.isfinite(d) and d > 0]
        stats.append(
            {
                "name": name,
                "run_dir": str(run.resolve()),
                "rows": len(rows),
                "unique_hkl": len(keys),
                "hkl_keys": keys,
                "dmin_A": min(finite_d) if finite_d else float("nan"),
                **metrics,
            }
        )
    return stats


def quality_weight(
    stat: Mapping,
    *,
    best_wr2: float,
    best_r1: float,
    best_cc: float,
    wr2_scale: float,
    r1_scale: float,
    cc_scale: float,
    min_weight: float,
) -> float:
    """Map quality penalties into a smooth dataset weight."""

    wr2 = safe_float(stat.get("wR2_all"))
    r1 = safe_float(stat.get("R1_all"))
    cc = safe_float(stat.get("CC"))
    if not (math.isfinite(wr2) and math.isfinite(r1)):
        return 0.0
    wr2_penalty = max(0.0, wr2 - best_wr2)
    r1_penalty = max(0.0, r1 - best_r1)
    cc_penalty = max(0.0, best_cc - cc) if math.isfinite(cc) and math.isfinite(best_cc) else 0.0
    weight = math.exp(-wr2_penalty / max(wr2_scale, 1e-9))
    weight *= math.exp(-r1_penalty / max(r1_scale, 1e-9))
    weight *= math.exp(-cc_penalty / max(cc_scale, 1e-9))
    return clamp(weight, min_weight, 1.0)


def decide_dataset_policy(
    stats: list[dict],
    *,
    min_rows: int = 100,
    min_weight: float = 0.25,
    primary_min_quality: float = 0.75,
    min_raw_quality: float = 0.20,
    min_cc: float = 0.70,
    max_wr2: float = 0.75,
    max_r1: float = 0.60,
    unique_gain_fraction: float = 0.01,
    wr2_scale: float = 0.08,
    r1_scale: float = 0.08,
    cc_scale: float = 0.04,
) -> dict:
    """Choose primary, downweighted, completeness-only, or excluded roles.

    The policy is deliberately conservative. Good crystals get full weight.
    Marginal crystals get a smooth sigma-inflation weight. Very weak crystals
    are kept only for endpoint/completeness when they add otherwise missing
    reflections; otherwise they are excluded from the combined raw-row target.
    """

    finite_wr2 = [safe_float(s.get("wR2_all")) for s in stats if math.isfinite(safe_float(s.get("wR2_all")))]
    finite_r1 = [safe_float(s.get("R1_all")) for s in stats if math.isfinite(safe_float(s.get("R1_all")))]
    finite_cc = [safe_float(s.get("CC")) for s in stats if math.isfinite(safe_float(s.get("CC")))]
    best_wr2 = min(finite_wr2) if finite_wr2 else float("inf")
    best_r1 = min(finite_r1) if finite_r1 else float("inf")
    best_cc = max(finite_cc) if finite_cc else float("nan")

    rows: list[dict] = []
    for stat in stats:
        wr2 = safe_float(stat.get("wR2_all"))
        r1 = safe_float(stat.get("R1_all"))
        cc = safe_float(stat.get("CC"))
        q = quality_weight(
            stat,
            best_wr2=best_wr2,
            best_r1=best_r1,
            best_cc=best_cc,
            wr2_scale=wr2_scale,
            r1_scale=r1_scale,
            cc_scale=cc_scale,
            min_weight=min_weight,
        )
        bad_size = int(stat.get("rows", 0)) < int(min_rows)
        bad_quality = (
            not math.isfinite(wr2)
            or not math.isfinite(r1)
            or wr2 > max_wr2
            or r1 > max_r1
            or (math.isfinite(cc) and cc < min_cc)
        )
        primary_candidate = (not bad_size) and (not bad_quality) and q >= primary_min_quality
        rows.append({**{k: v for k, v in stat.items() if k != "hkl_keys"}, "quality": q, "bad_size": bad_size, "bad_quality": bad_quality, "primary_candidate": primary_candidate})

    if rows and not any(r["primary_candidate"] for r in rows):
        best = min(
            rows,
            key=lambda r: (
                0 if int(r.get("rows", 0)) >= int(min_rows) else 1,
                safe_float(r.get("wR2_all"), float("inf")),
                safe_float(r.get("R1_all"), float("inf")),
                -safe_float(r.get("CC"), -1.0),
            ),
        )
        best["primary_candidate"] = True
        best["forced_primary"] = True

    primary_names = {r["name"] for r in rows if r["primary_candidate"]}
    primary_union: set[tuple[int, int, int]] = set()
    key_by_name = {s["name"]: set(s.get("hkl_keys", set())) for s in stats}
    for name in primary_names:
        primary_union |= key_by_name.get(name, set())
    global_union: set[tuple[int, int, int]] = set()
    for keys in key_by_name.values():
        global_union |= keys

    decisions = []
    for row in rows:
        name = row["name"]
        keys = key_by_name.get(name, set())
        unique_gain = keys - primary_union
        unique_frac_dataset = len(unique_gain) / max(len(keys), 1)
        unique_frac_global = len(unique_gain) / max(len(global_union), 1)
        row["unique_gain_hkl"] = len(unique_gain)
        row["unique_gain_fraction_dataset"] = unique_frac_dataset
        row["unique_gain_fraction_global"] = unique_frac_global
        if row["primary_candidate"]:
            action = "primary"
            raw_weight = 1.0
            endpoint_weight = 1.0
            reasons = ["full-weight dataset: quality is within the primary gate"]
        elif row["bad_size"]:
            action = "exclude"
            raw_weight = 0.0
            endpoint_weight = 0.0
            reasons = [f"excluded: row count {int(row.get('rows', 0))} < {int(min_rows)}"]
        elif row["bad_quality"] and unique_frac_dataset >= unique_gain_fraction:
            action = "completeness_only"
            raw_weight = 0.0
            endpoint_weight = max(min_weight, min(0.50, row["quality"]))
            reasons = ["kept for endpoint completeness only: weak quality but nonredundant reflections"]
        elif row["bad_quality"]:
            action = "exclude"
            raw_weight = 0.0
            endpoint_weight = 0.0
            reasons = ["excluded: weak quality and little unique coverage"]
        elif row["quality"] >= min_raw_quality:
            action = "downweight"
            raw_weight = row["quality"]
            endpoint_weight = row["quality"]
            reasons = ["downweighted: useful but worse than the primary crystal set"]
        elif unique_frac_dataset >= unique_gain_fraction:
            action = "completeness_only"
            raw_weight = 0.0
            endpoint_weight = max(min_weight, min(0.50, row["quality"]))
            reasons = ["kept for endpoint completeness only: low raw-row weight but useful unique coverage"]
        else:
            action = "exclude"
            raw_weight = 0.0
            endpoint_weight = 0.0
            reasons = ["excluded: low quality and little unique coverage"]
        decisions.append(
            {
                **row,
                "action": action,
                "raw_row_weight": float(raw_weight),
                "endpoint_weight": float(endpoint_weight),
                "include_in_row_regularization": action in {"primary", "downweight"},
                "include_in_endpoint": action in {"primary", "downweight", "completeness_only"},
                "reasons": reasons,
            }
        )

    return {
        "status": "ok",
        "definition": "multi-dataset inclusion/downweight policy",
        "thresholds": {
            "min_rows": int(min_rows),
            "min_weight": float(min_weight),
            "primary_min_quality": float(primary_min_quality),
            "min_raw_quality": float(min_raw_quality),
            "min_cc": float(min_cc),
            "max_wr2": float(max_wr2),
            "max_r1": float(max_r1),
            "unique_gain_fraction": float(unique_gain_fraction),
            "wr2_scale": float(wr2_scale),
            "r1_scale": float(r1_scale),
            "cc_scale": float(cc_scale),
        },
        "reference": {"best_wR2_all": best_wr2, "best_R1_all": best_r1, "best_CC": best_cc},
        "coverage": {
            "global_unique_hkl": len(global_union),
            "primary_unique_hkl": len(primary_union),
        },
        "datasets": decisions,
    }


def build_dataset_policy(runs: Mapping[str, Path], **kwargs) -> dict:
    """Collect run statistics and return a policy payload."""

    return decide_dataset_policy(collect_dataset_stats(runs), **kwargs)


def write_policy_artifacts(policy: Mapping, out_dir: Path) -> dict:
    """Write JSON, CSV, and Markdown policy reports."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_inclusion_policy.json"
    csv_path = out_dir / "dataset_inclusion_policy.csv"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(policy, indent=2, allow_nan=True) + "\n")
    rows = list(policy.get("datasets", []))
    if rows:
        fields = [
            "name",
            "action",
            "raw_row_weight",
            "endpoint_weight",
            "rows",
            "unique_hkl",
            "unique_gain_hkl",
            "R1_all",
            "wR2_all",
            "CC",
            "quality",
            "dmin_A",
            "reasons",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("; ".join(row[k]) if k == "reasons" else row.get(k, "")) for k in fields})
    else:
        csv_path.write_text("")
    lines = [
        "# GROND Multi-Dataset Inclusion Policy",
        "",
        "This gate decides which crystals should carry raw-row regularization weights and which should only add endpoint completeness. Downweighting is implemented as sigma inflation, so marginal datasets contribute less to weighted residuals without being silently deleted.",
        "",
        "| dataset | action | raw weight | endpoint weight | rows | unique HKL | unique gain | row R(I)all | row wR(I)all | CC | reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reason = "; ".join(row.get("reasons", []))
        lines.append(
            f"| {row.get('name')} | {row.get('action')} | {safe_float(row.get('raw_row_weight')):.3f} | "
            f"{safe_float(row.get('endpoint_weight')):.3f} | {int(row.get('rows', 0))} | {int(row.get('unique_hkl', 0))} | "
            f"{int(row.get('unique_gain_hkl', 0))} | {100*safe_float(row.get('R1_all')):.2f}% | "
            f"{100*safe_float(row.get('wR2_all')):.2f}% | {safe_float(row.get('CC')):.4f} | {reason} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve()), "report": str(md_path.resolve())}


def policy_by_name(policy: Mapping) -> dict[str, dict]:
    """Return policy dataset rows keyed by name."""

    return {str(row.get("name")): dict(row) for row in policy.get("datasets", [])}


def write_policy_weighted_runs(
    runs: Mapping[str, Path],
    policy: Mapping,
    out_dir: Path,
    *,
    include_endpoint_only: bool = True,
) -> dict[str, Path]:
    """Write run copies annotated with policy weights for downstream application."""

    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = policy_by_name(policy)
    output: dict[str, Path] = {}
    for name, run_dir in runs.items():
        decision = decisions.get(name, {"action": "primary", "raw_row_weight": 1.0, "endpoint_weight": 1.0})
        action = str(decision.get("action", "primary"))
        if action == "exclude":
            continue
        if action == "completeness_only" and not include_endpoint_only:
            continue
        src = Path(run_dir)
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        rows = read_row_table(src / "native_dyn_calc.csv", name)
        weighted_rows = []
        for row in rows:
            new = dict(row)
            new["dataset"] = name
            new["dataset_role"] = action
            new["raw_row_weight"] = f"{safe_float(decision.get('raw_row_weight'), 0.0):.17g}"
            new["endpoint_weight"] = f"{safe_float(decision.get('endpoint_weight'), 0.0):.17g}"
            new["sigma_eff_original"] = row.get("sigma_eff", "")
            new["policy_sigma_weight_applied"] = "false"
            weighted_rows.append(new)
        write_row_table(dst / "native_dyn_calc.csv", weighted_rows)
        summary = read_json(src / "summary.json")
        summary["dataset_policy"] = {k: v for k, v in decision.items() if k != "hkl_keys"}
        (dst / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
        output[name] = dst
        for extra in ("manifest.json", "best_native_fit.json"):
            if (src / extra).exists():
                shutil.copy2(src / extra, dst / extra)
    return output
