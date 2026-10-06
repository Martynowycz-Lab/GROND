"""Validation lines AUTO prints at each step, read back from the JSON artifacts that step recorded.

With ``--validation-console on`` (default) AUTO prints, as each step finishes: the split summary (work,
free and excluded rows and orbits per crystal), the held-out (free-row) R_obs/R_all with 95% CIs after the
screen, the early and final hand blocks (pooled and per-crystal orbit sign z, dR_obs with its CI, the
escalation), every nuisance decision with its CV R and paired CI or its work-row score, the thickness, and
each SHELXL macro-cycle's free/work R, shift and NPD/ISOR state.  The final held-out table, gaps, leak
flags and phase timings are part of the terminal finish summary (grond.reports.terminal).

Every number is read from a recorded artifact; nothing is recomputed for the console.  The two artifacts this
module writes itself, ``decisions/split_summary.json`` and ``decisions/heldout_stage_<label>.json``, are
written first and then printed from the file.  ``--validation-console off`` (legacy) prints none of these
lines and writes neither file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from grond.cli_support.values import try_read_json
from grond.reports.heldout import _gap, _json_default, _shell_dmin, _split_block, read_row_table
from grond.validation import heldout

VALIDATION_CONSOLE = ("on", "off")
SPLIT_SUMMARY_SCHEMA = "grond.split_summary.v1"
STAGE_HELDOUT_SCHEMA = "grond.stage_heldout.v1"
PREFIX = "VALIDATION"


def console_enabled(args: argparse.Namespace) -> bool:
    """Print validation lines? Namespaces without the option keep the legacy (silent) console."""

    value = str(getattr(args, "validation_console", "off") or "off")
    if value not in VALIDATION_CONSOLE:
        raise ValueError(f"unknown --validation-console {value!r}; expected one of {list(VALIDATION_CONSOLE)}")
    return value == "on" and not bool(getattr(args, "dry_run", False))


def _emit(lines: Iterable[str]) -> None:
    """Print each line, flushed so the console keeps pace with the run."""

    for line in lines:
        print(line, flush=True)


def _guarded(step: str, func, *args, **kwargs):
    """Run one console report; a failure is printed as a warning and never changes the run (it decides nothing)."""

    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"WARN validation console ({step}): {type(exc).__name__}: {exc}", flush=True)
        return None


def _write(path: Path, payload: Mapping[str, object]) -> None:
    """Write payload as indented JSON, creating the parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _float(value: object) -> float:
    """Return value as a float, NaN when it cannot be converted."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _num(value: object, digits: int = 2) -> str:
    """Format a number with fixed digits, n/a when it is not a finite number."""

    out = _float(value)
    return f"{out:.{digits}f}" if math.isfinite(out) else "n/a"


def _signed(value: object, digits: int = 2) -> str:
    """Format a number with a sign and fixed digits, n/a when it is not a finite number."""

    out = _float(value)
    return f"{out:+.{digits}f}" if math.isfinite(out) else "n/a"


def _ci(ci: object, *, signed: bool = False, digits: int = 2) -> str:
    """Format a two-value confidence interval as [lo, hi], [n/a] when it is not a pair."""

    values = list(ci) if isinstance(ci, (list, tuple)) else []
    if len(values) != 2:
        return "[n/a]"
    fmt = _signed if signed else _num
    return f"[{fmt(values[0], digits)}, {fmt(values[1], digits)}]"


def _percent_of_fraction(value: object, digits: int = 2) -> str:
    """Format a fraction as a percent number (without the % sign), n/a when it is not a finite number."""

    out = _float(value)
    return f"{100.0 * out:.{digits}f}" if math.isfinite(out) else "n/a"


# ----------------------------------------------------------------------------- split summary


def _contract_counts(ds) -> dict[str, object] | None:
    """Raw-row membership counts of the dataset's recorded split contract (None when it has none)."""

    path = str(getattr(ds, "split_contract", "") or "").strip()
    if not path:
        return None
    payload = try_read_json(Path(path).expanduser())
    if not payload:
        raise ValueError(f"split contract {path} of {ds.name} is missing or unreadable")
    realised = payload.get("realised_free_fraction") if isinstance(payload.get("realised_free_fraction"), Mapping) else {}
    return {
        "path": path,
        "algorithm": payload.get("algorithm"),
        "rows": {
            "work": int(payload.get("work_row_count") or 0),
            "free": int(payload.get("free_row_count") or 0),
            "excluded": int(payload.get("excluded_row_count") or 0),
        },
        "orbit_groups": {
            "work": int(payload.get("work_group_count") or 0),
            "free": int(payload.get("free_group_count") or 0),
            "excluded": int(payload.get("excluded_group_count") or 0),
        },
        "realised_free_fraction_rows": realised.get("rows"),
    }


def split_summary(prep, args: argparse.Namespace, runs: Mapping[str, Path], root: Path) -> dict[str, object]:
    """Work/free/excluded rows and orbits per crystal and pooled; written to decisions/split_summary.json.

    Row-table counts are the rows the screen actually fitted (work) and held out (free), keyed by the model
    Laue orbit with Friedel pairs (grond.validation.heldout.orbit_key); excluded (lockbox or candidate-union)
    rows never reach a row table and are counted from the dataset's split contract.
    """

    from grond.bloch.model import parse_model

    ops = parse_model(Path(prep.structure)).get("symmetry_ops") or []
    exclusion_path = str(getattr(args, "exclude_orbits_json", "") or "").strip()
    exclusion = heldout.load_orbit_exclusion(exclusion_path) if exclusion_path else None
    record: dict[str, object] = {
        "schema": SPLIT_SUMMARY_SCHEMA,
        "split_scheme": str(getattr(args, "split_scheme", "legacy") or "legacy"),
        "split_salt": str(getattr(args, "split_salt", "") or ""),
        "split_seed": getattr(args, "split_seed", None),
        "free_fraction": float(getattr(args, "free_fraction", 0.10)),
        "excluded_orbits_json": exclusion_path,
        "excluded_orbits_sha256": exclusion["sha256"] if exclusion else "",
        "excluded_orbit_count": len(exclusion["orbits"]) if exclusion else 0,
        "orbit_key": "grond.validation.heldout.orbit_key (model Laue group + Friedel)",
        "crystals": {},
    }
    pooled_orbits: dict[str, set[str]] = {"work": set(), "free": set()}
    pooled_rows = {"work": 0, "free": 0}
    for ds in prep.datasets:
        table = Path(runs[ds.name]) / "native_dyn_calc.csv"
        rows = read_row_table(table)
        keys = heldout.orbit_keys(rows["hkl"], ops, include_friedel=True).astype(str)
        split = np.asarray(rows["split"], dtype=object)
        orbits = {name: set(keys[split == name]) for name in ("work", "free")}
        counts = {name: int(np.count_nonzero(split == name)) for name in ("work", "free")}
        for name in ("work", "free"):
            pooled_orbits[name] |= orbits[name]
            pooled_rows[name] += counts[name]
        record["crystals"][ds.name] = {
            "row_table": str(table),
            "rows": {**counts, "other": int(len(split) - sum(counts.values()))},
            "orbits": {name: len(values) for name, values in orbits.items()},
            "excluded_orbit_rows_in_table": (
                int(np.count_nonzero(np.isin(keys, sorted(exclusion["orbits"])))) if exclusion else 0
            ),
            "contract": _contract_counts(ds),
        }
    record["pooled"] = {
        "rows": dict(pooled_rows),
        "orbits": {name: len(values) for name, values in pooled_orbits.items()},
        "orbits_free_and_work": len(pooled_orbits["work"] & pooled_orbits["free"]),
    }
    _write(Path(root) / "decisions" / "split_summary.json", record)
    return record


def split_summary_lines(record: Mapping[str, object]) -> list[str]:
    """Return the console lines of a split-summary record: the split rule, then rows and orbits by split.

    Work and free rows and orbits are given per crystal and pooled.
    """

    scheme = str(record.get("split_scheme"))
    rule = (
        f"salt {record.get('split_salt')}" if scheme == "orbit-hash" else f"seed {record.get('split_seed')}"
    )
    exclusion = (
        f"excluded orbits {record.get('excluded_orbit_count')} (sha256 {str(record.get('excluded_orbits_sha256'))[:12]})"
        if record.get("excluded_orbits_json")
        else "no orbit exclusion list"
    )
    lines = [
        f"{PREFIX} split: {scheme}, {rule}, requested free fraction {_num(record.get('free_fraction'), 2)}; {exclusion}"
    ]
    for name, crystal in (record.get("crystals") or {}).items():
        rows, orbits = crystal.get("rows") or {}, crystal.get("orbits") or {}
        contract = crystal.get("contract")
        raw = (
            f"; split contract rows work {contract['rows']['work']} / free {contract['rows']['free']} / excluded "
            f"{contract['rows']['excluded']} (orbit groups {contract['orbit_groups']['work']} / "
            f"{contract['orbit_groups']['free']} / {contract['orbit_groups']['excluded']}), realised free fraction "
            f"{_num(contract.get('realised_free_fraction_rows'), 3)}"
            if isinstance(contract, Mapping)
            else "; no split contract recorded (engine split)"
        )
        lines.append(
            f"  {name}: fitted rows work {rows.get('work')} / free {rows.get('free')}, orbits work "
            f"{orbits.get('work')} / free {orbits.get('free')}{raw}"
        )
    pooled = record.get("pooled") or {}
    rows, orbits = pooled.get("rows") or {}, pooled.get("orbits") or {}
    lines.append(
        f"  pooled: rows work {rows.get('work')} / free {rows.get('free')}, orbits work {orbits.get('work')} / free "
        f"{orbits.get('free')}; orbits both free and work across crystals {pooled.get('orbits_free_and_work')}"
    )
    return lines


def report_split(prep, args: argparse.Namespace, runs: Mapping[str, Path], root: Path) -> dict[str, object] | None:
    """Write and print the split summary of the screen row tables (console on only)."""

    if not console_enabled(args):
        return None

    def report() -> dict[str, object]:
        """Write the split summary, read it back from the file and print it; returns the record."""

        split_summary(prep, args, runs, root)
        record = try_read_json(Path(root) / "decisions" / "split_summary.json")
        _emit(split_summary_lines(record))
        return record

    return _guarded("split summary", report)


# ----------------------------------------------------------------------------- stage held-out block


def stage_heldout(
    runs: Mapping[str, Path],
    *,
    model: str,
    exclusion_path: str = "",
    label: str,
) -> dict[str, object]:
    """Held-out (free-row) R_all/R_obs with single-arm orbit-bootstrap CIs for one stage's row tables.

    The same statistics as HELDOUT_VALIDATION (grond.reports.heldout), on the stage's own ``dyn_scaled``.
    """

    from grond.bloch.model import parse_model

    ops = parse_model(Path(model)).get("symmetry_ops") or []
    exclusion = heldout.load_orbit_exclusion(exclusion_path) if exclusion_path else None
    tables, keys, fits = {}, {}, {}
    for name, run in runs.items():
        table = read_row_table(Path(run) / "native_dyn_calc.csv")
        orbit = heldout.orbit_keys(table["hkl"], ops, include_friedel=True)
        if exclusion:
            keep = np.asarray([key not in exclusion["orbits"] for key in orbit], dtype=bool)
            table = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in table.items()}
            orbit = orbit[keep]
        summary = try_read_json(Path(run) / "summary.json")
        best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
        config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
        tables[name], keys[name] = table, orbit
        fits[name] = str(best.get("scale_fit_split") or config.get("scale_fit_split") or "unknown")
    names = list(tables)
    shell_dmin = _shell_dmin(np.concatenate([tables[n]["d"] for n in names]), exclusion)
    crystals = {}
    for name in names:
        t = tables[name]
        shell = heldout.shell_index(t["d"], shell_dmin)
        held = _split_block(t["y"], t["sig"], t["pred"], keys[name], shell, t["split"] == "free")
        insample = heldout.r_metrics(t["y"], t["sig"], t["pred"], t["split"] == "work")
        crystals[name] = {
            "run": str(runs[name]),
            "scale_fit_split": fits[name],
            "heldout": held,
            "in_sample": insample,
            "gap": _gap(held, insample),
        }
    cat = lambda key: np.concatenate([tables[n][key] for n in names])  # noqa: E731
    y, sig, pred, split, d = cat("y"), cat("sig"), cat("pred"), cat("split"), cat("d")
    shell = heldout.shell_index(d, shell_dmin)
    held = _split_block(y, sig, pred, np.concatenate([keys[n] for n in names]), shell, split == "free")
    insample = heldout.r_metrics(y, sig, pred, split == "work")
    return {
        "schema": STAGE_HELDOUT_SCHEMA,
        "stage": label,
        "model": str(model),
        "definitions": {
            "heldout_rows": "split == free in the stage row tables",
            "prediction": "the stage's dyn_scaled",
            "r_factor": "amplitude R1 in percent; R_obs uses Io > 3 source_sigma",
            "ci": f"single-arm merged-orbit bootstrap, {heldout.N_BOOT} shell-stratified resamples, "
            f"seed {heldout.BOOTSTRAP_SEED}",
            "shells_d_min": shell_dmin,
            "gap": "held-out minus in-sample (work rows, same model)",
        },
        "scale_fit_split": fits,
        "free_rows_were_fitted": any(value != "work" for value in fits.values()),
        "crystals": crystals,
        "pooled": {"heldout": held, "in_sample": insample, "gap": _gap(held, insample)},
    }


def _heldout_row(name: str, block: Mapping[str, object]) -> str:
    """Format the held-out and in-sample R_obs and R_all (percent, 95% CIs) and gaps (pp) of one block."""

    held, ins, gap = block.get("heldout") or {}, block.get("in_sample") or {}, block.get("gap") or {}
    return (
        f"  {name}: held-out R_obs {_num(held.get('r_obs'))} {_ci(held.get('r_obs_ci95'))} %, R_all "
        f"{_num(held.get('r_all'))} {_ci(held.get('r_all_ci95'))} % on {held.get('n_all', 'n/a')} free rows / "
        f"{held.get('n_orbits', 'n/a')} orbits; in-sample R_obs {_num(ins.get('r_obs'))} / R_all "
        f"{_num(ins.get('r_all'))} %; gap {_signed(gap.get('r_obs'))} / {_signed(gap.get('r_all'))} pp"
    )


def stage_heldout_lines(record: Mapping[str, object]) -> list[str]:
    """Return the console lines of a stage held-out record; warns when scales were not fitted on work rows only."""

    fits = record.get("scale_fit_split") or {}
    fit_text = (
        "scales fitted on work rows"
        if not record.get("free_rows_were_fitted")
        else "WARNING scales fitted on " + ", ".join(f"{k}={v}" for k, v in fits.items()) + " rows: not held out"
    )
    lines = [f"{PREFIX} held-out after {record.get('stage')} (free rows; {fit_text}; 95% orbit-bootstrap CIs):"]
    lines += [_heldout_row(name, block) for name, block in (record.get("crystals") or {}).items()]
    if len(record.get("crystals") or {}) > 1:
        lines.append(_heldout_row("pooled", record.get("pooled") or {}))
    return lines


def report_stage_heldout(
    args: argparse.Namespace,
    runs: Mapping[str, Path],
    root: Path,
    *,
    label: str,
    model: str,
) -> dict[str, object] | None:
    """Write decisions/heldout_stage_<label>.json from the stage row tables and print it (console on only)."""

    if not console_enabled(args):
        return None
    path = Path(root) / "decisions" / f"heldout_stage_{label}.json"

    def report() -> dict[str, object]:
        """Write the stage held-out record, read it back from the file and print it; returns the record."""

        exclusion = str(getattr(args, "exclude_orbits_json", "") or "")
        _write(path, stage_heldout(runs, model=model, exclusion_path=exclusion, label=label))
        record = try_read_json(path)
        _emit(stage_heldout_lines(record))
        return record

    return _guarded(f"held-out after {label}", report)


# ----------------------------------------------------------------------------- hand decision


def _hand_block_text(block: Mapping[str, object]) -> str:
    """Format one hand evidence block: orbit sign z with orbit counts, then dR_obs and dR_all with 95% CIs.

    Deltas are inverse minus selected hand, in pp.
    """

    dro = block.get("delta_r_obs") if isinstance(block.get("delta_r_obs"), Mapping) else {}
    dra = block.get("delta_r_all") if isinstance(block.get("delta_r_all"), Mapping) else {}
    return (
        f"orbit sign z {_signed(block.get('orbit_sign_z'))} (inverse better in {block.get('orbits_inverse_better', 'n/a')}, "
        f"selected in {block.get('orbits_model_better', 'n/a')} of {block.get('n_orbits', 'n/a')} orbits), "
        f"dR_obs(inverse - selected) {_signed(dro.get('delta'))} {_ci(dro.get('ci95'), signed=True)} pp, "
        f"dR_all {_signed(dra.get('delta'))} {_ci(dra.get('ci95'), signed=True)} pp"
    )


def _fidelity_text(fidelity: Mapping[str, object] | None) -> str:
    """Format a Bloch fidelity record per crystal as 'name N/Q n_basis/n_points', n/a when absent."""

    if not isinstance(fidelity, Mapping) or not fidelity:
        return "n/a"
    return ", ".join(
        f"{name} N/Q {(values or {}).get('n_basis')}/{(values or {}).get('n_points')}"
        for name, values in fidelity.items()
    )


def hand_decision_lines(doc: Mapping[str, object]) -> list[str]:
    """Return the console lines of one held-out hand decision record.

    The lines give the rule, the action, the pooled and per-crystal evidence, unmet switch rules, errors, the
    escalation, the diagnostic legacy ZKP and the early/final agreement.
    """

    stage = str(doc.get("stage") or "?")
    status = str(doc.get("status") or "")
    if status.startswith("skipped"):
        return [f"{PREFIX} hand {stage}: {status} ({doc.get('reason', '')})"]
    rule = doc.get("rule") if isinstance(doc.get("rule"), Mapping) else {}
    authority = (
        f"held-out orbit sign rule, pooled z >= {_num(rule.get('orbit_sign_z_min'), 1)} and dR_obs CI below 0"
        if doc.get("decision_authority") == "heldout_orbit_sign"
        else "legacy all-row ZKP rule; held-out block diagnostic only"
    )
    action = (
        f"switched {doc.get('selected_hand_before')} -> {doc.get('selected_hand')}"
        if doc.get("switched")
        else f"kept {doc.get('selected_hand')}"
    )
    lines = [f"{PREFIX} hand {stage} ({authority}): {action}; evidence {status}"]
    pooled = doc.get("pooled") if isinstance(doc.get("pooled"), Mapping) else {}
    if pooled:
        lines.append(f"  pooled over {len(doc.get('per_crystal') or {})} crystal(s): {_hand_block_text(pooled)}")
    scope = doc.get("evidence_dataset_scope") if isinstance(doc.get("evidence_dataset_scope"), Mapping) else {}
    if scope.get("excluded") or scope.get("demoted_included"):
        # The final check's crystal set: who the dataset policy excluded (with its reason) and who stayed in.
        excluded = "; ".join(
            f"{name} ({'; '.join(str(reason) for reason in reasons)})" for name, reasons in (scope.get("excluded") or {}).items()
        )
        demoted = ", ".join(f"{name} ({role})" for name, role in (scope.get("demoted_included") or {}).items())
        lines.append(
            f"  evidence crystals ({scope.get('scope')}): {', '.join(str(name) for name in scope.get('included') or [])}"
            + (f"; demoted by the dataset policy but included: {demoted}" if demoted else "")
            + (f"; excluded by the dataset policy: {excluded}" if excluded else "")
        )
    agreement = doc.get("per_crystal_agreement") if isinstance(doc.get("per_crystal_agreement"), Mapping) else {}
    for name, block in (doc.get("per_crystal") or {}).items():
        if not isinstance(block, Mapping):
            continue
        row = (agreement.get("crystals") or {}).get(name) or {}
        note = (
            f"; favours {row.get('preference')} ({row.get('strength')})"
            + ("; DISAGREES with the pooled decision (warned, no veto)" if row.get("disagrees_with_pooled") else "")
            if row
            else ""
        )
        lines.append(f"  {name}: {_hand_block_text(block)}{note}")
    for reason in doc.get("reasons") or []:
        lines.append(f"  switch rule not met: {reason}")
    for error in doc.get("errors") or []:
        lines.append(f"  ERROR: {error}")
    escalation = doc.get("escalation") if isinstance(doc.get("escalation"), Mapping) else {}
    if escalation:
        state = str(escalation.get("status"))
        if state == "escalated":
            screen = escalation.get("screen_evidence") or {}
            lines.append(
                f"  escalation: rule not met at screen fidelity ({_fidelity_text(escalation.get('screen_fidelity'))}; "
                f"pooled {_hand_block_text(screen.get('pooled') or {})}); decided at final fidelity "
                f"({_fidelity_text(escalation.get('final_fidelity'))})"
            )
        else:
            lines.append(f"  escalation: {state} ({escalation.get('reason', '')})")
    zkp = doc.get("legacy_all_row_zkp") if isinstance(doc.get("legacy_all_row_zkp"), Mapping) else {}
    if zkp and doc.get("decision_authority") == "heldout_orbit_sign":
        lines.append(
            f"  diagnostic only: legacy all-row ZKP {_signed(zkp.get('z_positive_favors_selected'))} "
            "(positive favours the selected hand; not the decision rule)"
        )
    final = doc.get("early_final_agreement") if isinstance(doc.get("early_final_agreement"), Mapping) else {}
    if final:
        text = f"  early/final agreement: {final.get('status')}"
        if final.get("status") == "conflict_flipped_back":
            text = f"  WARNING early/final agreement: conflict_flipped_back: {final.get('message', '')}"
        lines.append(text)
    return lines


def report_hand_decision(path: Path, args: argparse.Namespace) -> dict[str, object] | None:
    """Print decisions/hand_decision_heldout_<stage>.json (console on only)."""

    if not console_enabled(args):
        return None
    doc = try_read_json(Path(path))
    if not doc:
        _emit([f"{PREFIX} hand: no decision record at {path}"])
        return None
    _guarded("hand decision", lambda: _emit(hand_decision_lines(doc)))
    return doc


# ----------------------------------------------------------------------------- nuisance decisions


def _nuisance_values(nuisance: Mapping[str, object] | None, keys: Iterable[str]) -> str:
    """Format the named nuisance fields that are present as 'key=value, ...', n/a when none is."""

    nuisance = nuisance if isinstance(nuisance, Mapping) else {}
    return ", ".join(f"{key}={nuisance.get(key)}" for key in keys if key in nuisance) or "n/a"


THICKNESS_KEYS = ("thickness_model", "thickness_A", "thickness_grad_frac", "thickness_sigma_frac", "frame_offset")
CV_FAMILY_KEYS = {"mosaicity": ("mosaicity_mode", "mosaicity_sigma_deg"), "dose": ("dose_b_mode", "dose_b_per_frame")}


def _score(value: object) -> str:
    """Format a score, or a mapping's score entry, with five decimals."""

    if isinstance(value, Mapping):
        value = value.get("score")
    return _num(value, 5)


def feature_decision_line(record: Mapping[str, object]) -> str:
    """Format one nuisance-feature decision record as a console line.

    The format depends on the family: discrete or continuous thickness polish, an orbit-CV family with its
    candidates and paired test, or a score-gated family.
    """

    family = str(record.get("family") or "unknown")
    status = str(record.get("status") or "")
    if status in {"skipped", "planned", "dry_run"}:
        return f"  {family}: {status}" + (f" ({record.get('reason')})" if record.get("reason") else "")
    if status == "failed":
        return f"  {family}: FAILED ({record.get('error', '')})"
    if family == "discrete_thickness_polish":
        selected = (record.get("selected") or {}).get("nuisance") if isinstance(record.get("selected"), Mapping) else {}
        return (
            f"  thickness (discrete polish): {status}; kept {_nuisance_values(selected, THICKNESS_KEYS)}; "
            f"{record.get('selection_rows', 'n/a')}-row score baseline {_score(record.get('baseline_full_row_score'))} "
            f"-> candidate {_score(record.get('candidate_full_row_score'))} "
            f"(free rows in selection {record.get('free_rows_in_selection')})"
        )
    if family == "continuous_thickness_materialization":
        return (
            f"  thickness (continuous polish): {status}; {record.get('selection_rows', 'n/a')}-row score baseline "
            f"{_score(record.get('baseline_score'))} -> candidate {_score(record.get('candidate_score'))} "
            f"(gain {_signed(record.get('full_row_score_gain'), 5)}; free rows in selection "
            f"{record.get('free_rows_in_selection')})"
        )
    if isinstance(record.get("candidates"), list):
        keys = CV_FAMILY_KEYS.get(family, ())
        scheme = record.get("heldout_scheme") if isinstance(record.get("heldout_scheme"), Mapping) else {}
        candidates = "; ".join(
            f"{_nuisance_values(item, keys)}: CV R_obs {_num(item.get('cv_r_obs'))} / R_all {_num(item.get('cv_r_all'))} %"
            for item in record["candidates"]
            if isinstance(item, Mapping)
        )
        paired = record.get("paired_orbit_bootstrap_selected_minus_active")
        paired_text = (
            f"paired selected - active dR_obs {_signed(paired['r_obs'].get('delta'))} "
            f"{_ci(paired['r_obs'].get('ci95'), signed=True)} pp, dR_all {_signed(paired['r_all'].get('delta'))} "
            f"{_ci(paired['r_all'].get('ci95'), signed=True)} pp"
            if isinstance(paired, Mapping) and isinstance(paired.get("r_obs"), Mapping)
            else "active physics has the lowest CV R_all (no paired test)"
        )
        reasons = ", ".join(str(reason) for reason in record.get("rejection_reasons") or [])
        return (
            f"  {family} (orbit CV on {scheme.get('n_rows', 'n/a')} work rows / {scheme.get('n_orbits', 'n/a')} orbits, "
            f"{scheme.get('n_folds', 'n/a')} folds): {candidates}; {paired_text} -> {status}"
            + (f" ({reasons})" if reasons else "")
        )
    return (
        f"  {family}: {status}; score baseline {_score(record.get('baseline_score'))} -> gate "
        f"{_score(record.get('gate_score'))} (improvement {_signed(record.get('score_improvement'), 5)}, accept above "
        f"{_num(record.get('accept_delta'), 5)}; free rows in selection {record.get('free_rows_in_selection')})"
    )


def feature_decision_lines(dataset: str, records: Iterable[Mapping[str, object]], nuisance: Mapping | None) -> list[str]:
    """Return the console lines of one crystal's nuisance decisions, ending with its active thickness."""

    lines = [f"{PREFIX} nuisance decisions {dataset}:"]
    lines += [feature_decision_line(record) for record in records if isinstance(record, Mapping)]
    if isinstance(nuisance, Mapping) and nuisance:
        lines.append(f"  active thickness: {_nuisance_values(nuisance, THICKNESS_KEYS)}")
    return lines


def report_feature_decisions(dataset: str, path: Path, args: argparse.Namespace, *, nuisance: Mapping | None = None) -> None:
    """Print one crystal's recorded thickness and feature-gate decisions (console on only)."""

    if not console_enabled(args):
        return
    records = try_read_json(Path(path))
    if not isinstance(records, list):
        _emit([f"{PREFIX} nuisance decisions {dataset}: no decision record at {path}"])
        return
    _guarded(f"nuisance decisions {dataset}", lambda: _emit(feature_decision_lines(dataset, records, nuisance)))


# ----------------------------------------------------------------------------- SHELXL macro-cycles


def macro_cycle_lines(summary: Mapping[str, object], source: Path) -> list[str]:
    """Return the console lines of a SHELXL feedback-loop summary: its status, then one line per macro-cycle.

    Free and work R values are recorded in percent; SHELXL R1gt and wR2 are recorded as fractions and printed as
    percent.
    """

    lines = [
        f"{PREFIX} SHELXL macro-cycles: status {summary.get('status')}, stop reason {summary.get('stop_reason') or 'n/a'}, "
        f"kept cycle {summary.get('kept_cycle')} (keep policy {summary.get('keep_policy', 'n/a')}); {source}"
    ]
    if summary.get("stopped_reason"):
        lines.append(f"  stopped: {summary.get('stopped_reason')}")
    rows = summary.get("trajectory") if isinstance(summary.get("trajectory"), list) else []
    if not rows:
        return lines + ["  no per-cycle trajectory recorded"]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"  cycle {row.get('cycle')}: entering model free R_obs {_num(row.get('free_r_obs'))} / R_all "
            f"{_num(row.get('free_r_all'))} %, work R_obs {_num(row.get('work_r_obs'))} / R_all "
            f"{_num(row.get('work_r_all'))} %; SHELXL R1gt {_percent_of_fraction(row.get('r1_gt'))} %, wR2 "
            f"{_percent_of_fraction(row.get('wr2'))} %, GooF {_num(row.get('goof'), 3)}; max xyz shift "
            f"{_num(row.get('max_xyz_shift_A'), 4)} A, max shift/esd {_num(row.get('max_shift_esd'), 4)}; "
            f"NPD {row.get('n_npd')}, ISOR {row.get('n_isor')}, DAMP {'yes' if row.get('damp_present') else 'no'}, "
            f"excluded q rows {row.get('n_excluded_q_rows')}; adopted {row.get('adopted_for_feedback')}"
        )
    return lines


def report_macro_cycles(path: Path, args: argparse.Namespace) -> None:
    """Print the recorded SHELXL feedback trajectory (console on only)."""

    if not console_enabled(args):
        return
    summary = try_read_json(Path(path))
    if not summary:
        _emit([f"{PREFIX} SHELXL macro-cycles: no loop summary at {path}"])
        return
    _guarded("SHELXL macro-cycles", lambda: _emit(macro_cycle_lines(summary, Path(path))))
