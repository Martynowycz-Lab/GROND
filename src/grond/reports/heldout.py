"""Held-out validation block for FINAL_SUMMARY, QUICK_STATS and Table 1 Panel C.

Held-out rows are the ``split == free`` rows of the final per-crystal native row tables. Their
predictions are the run's own ``dyn_scaled`` values: under the AUTO default the frame scales were
fitted on work rows only, so free rows are predicted by the scale model rather than fitted by it.
R values are amplitude R1 in percent (R_all on every row, R_obs on Io > 3 sigma(Io), sigma = the
immutable source_sigma) with single-arm merged-orbit bootstrap 95% CIs
(grond.validation.heldout.paired_orbit_bootstrap: 2000 shell-stratified resamples, seed 20260925).
The in-sample line is the same model on work rows and the overfitting gap is held-out minus
in-sample. Leak flags state every place where free rows were used anyway (scale fit, nuisance
selection, dataset policy, SHELXL HKL, hand decision, macro-cycle model selection, start model), and
the parameter ledger never adds scale EDF to integer parameter counts.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from grond.cli_support.values import try_read_json
from grond.reports.endpoint_free_report import (
    endpoint_free_block,
    endpoint_free_markdown_lines,
    endpoint_free_terminal_line,
)
from grond.reports.nested_nuisance_report import nested_nuisance_lines, nested_nuisance_report
from grond.reports.progress import _quick_progress_phase_timings
from grond.validation import heldout
from grond.workflows.run_presets import preset_markdown_lines

SCHEMA = "grond.heldout_validation.v1"
N_SHELLS = 5
# Record keys by which an endpoint states whether its SHELXL HKL excluded the free rows.
HKL_SPLIT_KEYS = ("hkl_split", "shelxl_hkl_split", "endpoint_hkl_split")
HKL_FREE_FLAG_KEYS = ("free_rows_in_hkl", "free_rows_in_shelxl_hkl", "endpoint_include_free_rows")
NUISANCE_FIELDS = (
    "thickness_model", "thickness_A", "thickness_grad_frac", "thickness_sigma_frac", "frame_offset",
    "absorption_mode", "absorption_mu", "absorption_frac", "dose_b_mode", "dose_b_per_frame",
    "sg_jitter_mode", "mosaicity_mode", "mosaicity_sigma_deg", "row_resolution_b",
    "sigma_a", "sigma_b_corr", "sigma_c_peak",
)
# Named only for a gamma thickness distribution (the gradient slot is 0 there; legacy records are unchanged).
GAMMA_NUISANCE_FIELDS = ("thickness_gamma_shape", "thickness_evaluator", "thickness_min_A", "thickness_max_A")
TRAJECTORY_COLUMNS = (
    "cycle", "free_r_obs", "free_r_all", "work_r_obs", "work_r_all", "r1_gt", "r1_all", "wr2", "goof",
    "max_shift_esd", "mean_shift_esd", "max_xyz_shift_A", "n_npd", "n_isor", "n_excluded_q_rows", "damp_present",
)


# ----------------------------------------------------------------------------- inputs

def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    """Return one column of the row table as floats, NaN where missing or not numeric."""

    out = np.full(len(rows), np.nan, dtype=float)
    for i, row in enumerate(rows):
        with contextlib.suppress(TypeError, ValueError):
            out[i] = float(row.get(name, "nan"))
    return out


def read_row_table(path: Path) -> dict[str, object]:
    """Columns the held-out statistics need from one native_dyn_calc.csv."""

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = set(rows[0]) if rows else set()
    sigma_column = next((name for name in ("source_sigma", "xds_sigma", "sigma_eff") if name in fields), "")
    if not sigma_column:
        raise ValueError(f"{path} has no sigma column (source_sigma, xds_sigma or sigma_eff)")
    return {
        "hkl": np.asarray([[int(float(r["h"])), int(float(r["k"])), int(float(r["l"]))] for r in rows], dtype=int).reshape(-1, 3),
        "d": _column(rows, "d_A"),
        "y": _column(rows, "target_I"),
        "sig": _column(rows, sigma_column),
        "pred": _column(rows, "dyn_scaled"),
        "split": np.asarray([str(r.get("split", "")).strip().lower() for r in rows], dtype=object),
        "sigma_column": sigma_column,
        "scale_replay_model": sorted({str(r.get("scale_replay_model") or "") for r in rows} - {""}),
    }


def _model_symmetry(result: Mapping[str, object]) -> tuple[list, str]:
    """Return the final model's symmetry operations and the orbit-key rule used.

    Falls back to Friedel pairs only when no model is recorded or it cannot be read.
    """

    model = str(result.get("model") or "").strip()
    if not model:
        return [], "Friedel pairs only (no final model recorded)"
    try:
        from grond.bloch.model import parse_model

        ops = parse_model(Path(model).expanduser()).get("symmetry_ops") or []
    except Exception as exc:  # the report states the weaker key definition explicitly
        return [], f"Friedel pairs only (final model symmetry unreadable: {type(exc).__name__}: {exc})"
    return list(ops), "model Laue group with Friedel pairs"


def _find_values(payload: object, keys: tuple[str, ...]) -> list[object]:
    """Return every value stored under one of keys anywhere in a nested JSON payload, depth first."""

    found: list[object] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in keys:
                found.append(value)
            found.extend(_find_values(value, keys))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_find_values(value, keys))
    return found


def _excluded_orbits(out_dir: Path, summaries: list[Mapping[str, object]]) -> dict[str, object] | None:
    """Load the orbit exclusion list named in run_config.json or a run summary, or None when none is named."""

    paths = []
    for payload in [try_read_json(out_dir / "run_config.json"), *summaries]:
        paths.extend(str(v) for v in _find_values(payload, ("exclude_orbits_json",)) if str(v or "").strip())
    if not paths:
        return None
    return heldout.load_orbit_exclusion(Path(paths[0]).expanduser())



def start_model_leak(result: Mapping[str, object]) -> dict[str, object]:
    """Was the model every stage started from refined on the free rows? (grond.workflows.leak_free_start)"""

    record = result.get("leak_free_start") if isinstance(result.get("leak_free_start"), Mapping) else {}
    status = str(record.get("status") or "not_recorded")
    if status == "ok":
        return {"start_model": False, "start_model_evidence": f"refined on {record.get('n_work')} work reflections; "
                f"{record.get('n_free_removed')} free and {record.get('n_excluded_removed')} excluded removed"}
    return {"start_model": True, "start_model_evidence": f"leak-free start {status}: {record.get('leak', 'supplied model used')}"}

def shelxl_hkl_free_rows(out_dir: Path, result: Mapping[str, object]) -> tuple[bool, str]:
    """Return (free rows were in the SHELXL HKL, evidence) from the first summary that records it.

    The combined, primary and shared feedback summaries are read in that order. Without such a record the answer is
    True: the corrected-HKL endpoints merge every valid row.
    """

    sources = []
    combined = str(result.get("combined") or "").strip()
    if combined:
        sources.append(Path(combined) / "combined_summary.json")
    primary = str(result.get("primary_run") or "").strip()
    if primary:
        sources.append(Path(primary) / "summary.json")
    sources.append(out_dir / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json")
    for source in sources:
        payload = try_read_json(source)
        for value in _find_values(payload, HKL_FREE_FLAG_KEYS):
            if isinstance(value, bool):
                return value, f"{source}: free rows in SHELXL HKL = {value}"
        for value in _find_values(payload, HKL_SPLIT_KEYS):
            text = str(value).strip().lower()
            if text in {"work", "all"}:
                return text == "all", f"{source}: SHELXL HKL split = {text}"
    return True, "no record that free rows were excluded; the corrected-HKL endpoints merge every valid row"


def _feature_decision_records(out_dir: Path, result: Mapping[str, object], name: str, single: bool) -> list | None:
    """Return one crystal's recorded nuisance-feature decisions, or None when none were found.

    Single-dataset runs fall back to the AUTO result and its decision tree.
    """

    records = try_read_json(out_dir / "decisions" / name / "feature_decisions.json")
    if isinstance(records, list):
        return records
    if not single:
        return None
    # Single-dataset runs keep their gate records in the decision tree named by the AUTO result.
    tree_path = str(result.get("decision_tree") or "").strip()
    tree = try_read_json(Path(tree_path).expanduser()) if tree_path else {}
    for candidate in (result.get("feature_decisions"), tree.get("feature_decisions") if isinstance(tree, Mapping) else None):
        if isinstance(candidate, list):
            return candidate
    return None


def _free_veto_off(value: object) -> bool:
    """A free-wR2 no-harm veto is off when it is absent or +inf (NaN or a finite limit is a veto)."""

    if value is None:
        return True
    try:
        limit = float(value)
    except (TypeError, ValueError):
        return False
    return math.isinf(limit) and limit > 0.0


def _work_scored_without_free_veto(record: Mapping[str, object]) -> bool:
    """Work-row score, free veto off and work-fitted scales: the decision read no free row."""

    rows = [record.get("selection_rows"), record.get("score_rows")]
    for key in ("baseline_score", "gate_score", "candidate_score"):
        if isinstance(record.get(key), Mapping):
            rows.append(record[key].get("score_rows"))
    rows = [str(value).strip().lower() for value in rows if value not in (None, "")]
    if not rows or any(value != "work" for value in rows):
        return False
    if record.get("free_veto") is True or not _free_veto_off(record.get("max_free_wR2_loss")):
        return False
    split = record.get("scale_fit_split")
    splits = list(split.values()) if isinstance(split, Mapping) else [split]
    return bool(splits) and all(str(value).strip().lower() == "work" for value in splits)


def record_used_free_rows(record: Mapping[str, object]) -> bool:
    """Did one evaluated nuisance decision read free-row metrics?

    An explicit ``free_rows_in_selection`` wins. Otherwise a record is work-only when its held-out
    scheme states ``free_rows_used`` False (orbit-CV gates on work rows) or when it scored work rows
    with the free-wR2 veto off and work-fitted scales. Anything else counts as using free rows.
    """

    explicit = record.get("free_rows_in_selection")
    if isinstance(explicit, bool):
        return explicit
    scheme = record.get("heldout_scheme")
    if isinstance(scheme, Mapping) and scheme.get("free_rows_used") is False:
        return False
    return not _work_scored_without_free_veto(record)


def nuisance_selection_records(out_dir: Path, result: Mapping[str, object], names: list[str]) -> dict[str, list[str] | None]:
    """Per crystal: evaluated nuisance decisions whose score or veto read free-row metrics (None: no record found)."""

    out: dict[str, list[str] | None] = {}
    for name in names:
        records = _feature_decision_records(out_dir, result, name, single=len(names) == 1)
        if records is None:
            out[name] = None
            continue
        used = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            status = str(record.get("status", "")).lower()
            if status in {"", "skipped", "planned", "not_requested", "disabled", "dry_run"}:
                continue
            if not record_used_free_rows(record):
                continue
            used.append(f"{record.get('family', 'unknown')} ({status})")
        out[name] = used
    return out


def _dataset_policies(out_dir: Path, result: Mapping[str, object]) -> list[tuple[str, Mapping[str, object]]]:
    """Return the (label, policy) pairs of the multi-dataset inclusion policy.

    They come from the AUTO result, else from dataset_policy/*/dataset_inclusion_policy.json.
    """

    block = result.get("dataset_policy")
    if isinstance(block, Mapping):
        found = [(str(label), policy) for label, policy in block.items() if isinstance(policy, Mapping)]
        if found:
            return found
    found = []
    for path in sorted((Path(out_dir) / "dataset_policy").glob("*/dataset_inclusion_policy.json")):
        payload = try_read_json(path)
        if isinstance(payload, Mapping) and payload:
            found.append((path.parent.name, payload))
    return found


def dataset_policy_leak(out_dir: Path, result: Mapping[str, object], n_crystals: int) -> tuple[bool | None, str]:
    """Did the multi-dataset inclusion/weighting policy read free-row metrics? (grond.workflows.dataset_policy)"""

    policies = _dataset_policies(out_dir, result)
    if not policies:
        if n_crystals == 1:
            return False, "single dataset: no multi-dataset policy"
        return None, "no dataset-policy record found"
    used, work_only, not_evaluated = [], [], []
    for label, policy in policies:
        status = str(policy.get("status") or "")
        if status != "ok":
            not_evaluated.append(f"{label} ({status or 'unrecorded'})")
            continue
        explicit = policy.get("free_rows_in_selection")
        rows = str(policy.get("selection_rows") or "all").strip().lower()
        uses = explicit if isinstance(explicit, bool) else rows != "work"
        (used if uses else work_only).append(f"{label} ({rows} rows)")
    if used:
        return True, "policy quality metrics read free rows: " + ", ".join(used)
    if work_only:
        return False, "policy quality metrics on work rows only: " + ", ".join(work_only)
    return False, "policy not evaluated: " + ", ".join(not_evaluated)


def macro_cycle_selection_leak(trajectory: Mapping[str, object] | None) -> tuple[bool | None, str]:
    """Was the kept SHELXL macro-cycle model chosen on free-row R? (grond.refine.feedback_trajectory.select_kept_cycle)"""

    if not trajectory:
        return None, "no SHELXL feedback loop summary found"
    policy = str(trajectory.get("keep_policy") or "").strip()
    scores = trajectory.get("adopted_cycle_scores") if isinstance(trajectory.get("adopted_cycle_scores"), list) else []
    scored = []
    for item in scores:
        try:
            value = float((item or {}).get("free_r_obs"))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(value):
            scored.append(item)
    kept, reason = trajectory.get("kept_cycle"), trajectory.get("kept_reason")
    if policy == "best-free" and len(scored) > 1:
        return True, (f"keep policy best-free kept cycle {kept} of {len(scored)} adopted cycles "
                      f"compared on free-row R_obs: {reason}")
    if policy == "best-free":
        return False, f"keep policy best-free with {len(scored)} free-scored adopted cycle(s), no comparison: {reason}"
    if not policy:
        return False, f"no keep policy recorded (last adopted cycle kept); kept cycle {kept}"
    return False, f"keep policy {policy}: {reason or 'last adopted cycle kept'}"


def hand_block(out_dir: Path) -> dict[str, object] | None:
    """Return the early and final held-out hand decision records, their selected hands and whether they agree.

    None when neither was recorded. An escalated early decision also shows its superseded screen-fidelity record.
    """

    stages = {}
    for stage in ("early", "final"):
        payload = try_read_json(out_dir / "decisions" / f"hand_decision_heldout_{stage}.json")
        if payload:
            stages[stage] = payload
    if not stages:
        return None
    early_escalation = (stages.get("early") or {}).get("escalation")
    if isinstance(early_escalation, Mapping) and early_escalation.get("status") == "escalated":
        # The superseded screen-fidelity evidence of an escalated early decision, shown beside it.
        screen = try_read_json(out_dir / "decisions" / "hand_decision_heldout_early_screen_fidelity.json")
        if screen:
            stages = {"early_screen_fidelity": {**screen, "selected_hand": "n/a", "switched": "n/a"}, **stages}
    selected = {stage: str(doc.get("selected_hand", "")) for stage, doc in stages.items()}
    agreement = selected["early"] == selected["final"] if {"early", "final"} <= set(selected) else None
    return {"stages": stages, "selected": selected, "early_final_agree": agreement}


def hand_summary_sentences(hand: Mapping[str, object] | None) -> list[str]:
    """Plain sentences for the top of the reports: early decision, escalation, final check, flip-back, disagreements.

    Every number comes from decisions/hand_decision_heldout_<stage>.json.
    """

    if not hand:
        return []
    stages = hand.get("stages") or {}
    out: list[str] = []

    def block_text(doc: Mapping[str, object]) -> str:
        """Format the pooled orbit sign z and dR_obs(inverse - selected) with its 95% CI (pp)."""

        pooled = doc.get("pooled") if isinstance(doc.get("pooled"), Mapping) else {}
        dro = pooled.get("delta_r_obs") if isinstance(pooled.get("delta_r_obs"), Mapping) else {}
        ci = dro.get("ci95") if isinstance(dro.get("ci95"), list) else [None, None]
        return (
            f"pooled orbit sign z {_signed_num(pooled.get('orbit_sign_z'))}, dR_obs(inverse - selected) "
            f"{_signed_num(dro.get('delta'))} [{_signed_num(ci[0])}, {_signed_num(ci[1])}] pp"
        )

    def evidence_text(doc: Mapping[str, object]) -> str:
        """Format a stage's pooled evidence, plus its status and errors when the evidence is not complete."""

        status = doc.get("status")
        errors = "; ".join(str(error) for error in doc.get("errors") or [])
        parts = [block_text(doc)] if isinstance(doc.get("pooled"), Mapping) and doc.get("pooled") else []
        if status not in (None, "complete"):
            parts.append(f"evidence {status}" + (f": {errors}" if errors else ""))
        return "; ".join(parts) or "no pooled evidence recorded"

    early, final = stages.get("early"), stages.get("final")
    for label, doc in (("Early", early), ("Final", final)):
        if isinstance(doc, Mapping) and str(doc.get("status", "")).startswith("skipped"):
            # A skipped stage evaluated no rule (centrosymmetric model, hand control off); say so plainly.
            out.append(
                f"{label} hand check not evaluated: {doc.get('status')} ({doc.get('reason') or 'no reason recorded'}); "
                f"exported representation {doc.get('selected_hand', 'n/a')}."
            )
    if isinstance(early, Mapping) and not str(early.get("status", "")).startswith("skipped"):
        before = f"{early.get('selected_hand_before')} -> " if early.get("selected_hand_before") else "to "
        action = (
            f"switched {before}{early.get('selected_hand')}"
            if early.get("switched") is True
            else f"kept {early.get('selected_hand')}"
        )
        text = f"Early hand decision ({early.get('evidence')}, before any refinement): {action}; {evidence_text(early)}"
        escalation = early.get("escalation") if isinstance(early.get("escalation"), Mapping) else {}
        if escalation.get("status") == "escalated":
            screen = escalation.get("screen_evidence") if isinstance(escalation.get("screen_evidence"), Mapping) else {}
            text += (
                f"; decided at final fidelity because the screen-fidelity pair was inconclusive "
                f"({block_text(screen)})"
            )
        elif escalation:
            text += f"; escalation {escalation.get('status')}"
        out.append(text + ".")
    if isinstance(final, Mapping) and not str(final.get("status", "")).startswith("skipped"):
        agreement = final.get("early_final_agreement") if isinstance(final.get("early_final_agreement"), Mapping) else {}
        state = agreement.get("status") or ("switched" if final.get("switched") is True else "kept")
        text = f"Final hand check ({final.get('evidence')}): {state}; selected {final.get('selected_hand')}; "
        out.append(text + evidence_text(final) + ".")
        if agreement.get("status") == "conflict_flipped_back":
            out.append(
                "WARNING: the final held-out check reversed the early hand decision under the same pooled rule: "
                f"{agreement.get('message', '')}."
            )
    for stage in ("early", "final"):
        doc = stages.get(stage)
        per = (doc or {}).get("per_crystal_agreement") if isinstance(doc, Mapping) else None
        for message in (per or {}).get("warnings") or []:
            out.append(f"Per-crystal disagreement at the {stage} check (warned; the pooled decision stands): {message}.")
    return out


def trajectory_block(out_dir: Path, result: Mapping[str, object]) -> dict[str, object] | None:
    """Return the run's SHELXL feedback trajectory from the first loop summary that has per-cycle rows.

    Falls back to the first summary found, with an empty trajectory and a note; None when no summary exists.
    """

    candidates = [out_dir / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"]
    structure_run = str(result.get("structure_update_run") or "").strip()
    if structure_run:
        candidates.append(Path(structure_run) / "shelxl_feedback_loop_summary.json")
    candidates.extend(sorted((out_dir / "runs").glob("*/adaptive_shelxl_feedback/shelxl_feedback_loop_summary.json")))
    fallback = None
    for path in candidates:
        payload = try_read_json(path)
        if not payload:
            continue
        keep = {key: payload.get(key) for key in ("keep_policy", "kept_reason", "adopted_cycle_scores")}
        if isinstance(payload.get("trajectory"), list):
            return {"source": str(path), "status": payload.get("status"), "stop_reason": _loop_stop_reason(payload),
                    "kept_cycle": payload.get("kept_cycle"), **keep,
                    "trajectory": [dict(row) for row in payload["trajectory"] if isinstance(row, Mapping)]}
        fallback = fallback or {"source": str(path), "status": payload.get("status"),
                                "stop_reason": _loop_stop_reason(payload), "kept_cycle": payload.get("kept_cycle"),
                                **keep, "trajectory": [], "note": "loop summary has no per-cycle trajectory"}
    return fallback


def _loop_stop_reason(payload: Mapping[str, object]) -> str | None:
    """How the macro-cycle loop ended: its stop_reason when recorded, else a cap/converged status.

    Loop summaries record stop_reason separately (status is ``ok`` under the legacy stop rule); summaries
    written before that carried cap or converged in status itself.
    """

    reason = payload.get("stop_reason")
    if reason:
        return str(reason)
    status = str(payload.get("status") or "")
    return status if status in {"cap", "converged"} else None


def _scale_contract(summary: Mapping[str, object], replay: list[str] | None = None) -> dict[str, object]:
    """Return one run summary's frame-scale contract: kind, scale count, EDF, fit split and compact replays."""

    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    scale_model = best.get("scale_model") if isinstance(best.get("scale_model"), Mapping) else {}
    return {
        "kind": str(scale_model.get("kind") or "per-frame"),
        "scale_count": best.get("scale_count"),
        "scale_edf": best.get("scale_edf"),
        "scale_fit_split": best.get("scale_fit_split"),
        "compact_replay": list(replay or []),
    }


def scale_contract_consistency(out_dir: Path, tables: Mapping[str, Mapping[str, object]],
                               summaries: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Did the structure-feedback cycles and the published row tables use the same frame-scale contract?"""

    loop = try_read_json(out_dir / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json")
    adopted = [c for c in loop.get("cycles", []) if isinstance(c, Mapping) and c.get("adopted_for_feedback")] if loop else []
    runs = (adopted[-1].get("raw_runs") or adopted[-1].get("runs") or {}) if adopted else {}
    out: dict[str, object] = {}
    for name in tables:
        publication = _scale_contract(summaries.get(name) or {}, tables[name].get("scale_replay_model"))
        feedback = _scale_contract(try_read_json(Path(str(runs[name])) / "summary.json")) if name in runs else None
        same = None
        if feedback is not None:
            same = (
                feedback["kind"] == publication["kind"]
                and feedback["scale_fit_split"] == publication["scale_fit_split"]
                and not publication["compact_replay"]
                and (publication["kind"] == "pspline" or feedback["scale_count"] == publication["scale_count"])
            )
        out[name] = {"feedback_last_adopted_cycle": feedback, "publication": publication, "same_scale_contract": same}
    return out


# ----------------------------------------------------------------------------- statistics

def _shell_dmin(d: np.ndarray, exclusion: Mapping[str, object] | None) -> list[float]:
    """Return the resolution shell lower bounds d_min (A).

    These are the exclusion list's shells when it has them, else N_SHELLS equal-count shells of d.
    """

    if exclusion and exclusion.get("shell_dmin"):
        return [float(v) for v in exclusion["shell_dmin"]]
    d = d[np.isfinite(d)]
    qs = np.quantile(d, np.linspace(1.0, 0.0, N_SHELLS + 1)[1:-1])
    return [float(q) for q in qs] + [float(np.min(d))]


def _split_block(y, sig, pred, keys, shell, mask) -> dict[str, object]:
    """Return R_all and R_obs (percent) of the masked rows with single-arm orbit-bootstrap 95% CIs and the orbit count.

    When the bootstrap fails, ci_status records why and the CIs are left out.
    """

    metrics = heldout.r_metrics(y, sig, pred, mask)
    block: dict[str, object] = {**metrics}
    try:
        boot = heldout.paired_orbit_bootstrap(y, sig, pred, None, keys, shell, mask)
    except ValueError as exc:
        block["ci_status"] = str(exc)
        return block
    block["r_all_ci95"] = boot["r_all"]["ci95"]
    block["r_obs_ci95"] = boot["r_obs"]["ci95"]
    block["n_orbits"] = boot["n_orbits"]
    unavailable = {name: {key: boot[name][key] for key in ("ci_status", "n_undefined_resamples")}
                   for name in ("r_all", "r_obs") if "ci_status" in boot[name]}
    if unavailable:
        block["ci_status"] = "; ".join(f"{name}: {detail['ci_status']}" for name, detail in unavailable.items())
        block["bootstrap_diagnostics"] = unavailable
    return block


def _gap(held: Mapping[str, object], insample: Mapping[str, object]) -> dict[str, float]:
    """Return held-out minus in-sample R_all and R_obs (pp), NaN when either value is missing."""

    out = {}
    for name in ("r_all", "r_obs"):
        a, b = float(held.get(name, math.nan)), float(insample.get(name, math.nan))
        out[name] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
    return out


def ledger_entry(summary: Mapping[str, object], rows: Mapping[str, object]) -> dict[str, object]:
    """Return one crystal's parameter-ledger entry.

    It holds the row counts by split, the frame-scale count or P-spline EDF, the physics nuisance parameter count
    (native parameter_count minus scale count) and the named nuisance values.
    """

    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    scale_model = best.get("scale_model") if isinstance(best.get("scale_model"), Mapping) else {}
    kind = str(scale_model.get("kind") or "per-frame")
    scale_count = best.get("scale_count")
    metrics = ((best.get("metrics") or {}).get("all") or {}).get("dyn") or {}
    total = metrics.get("parameter_count")
    physics = int(total) - int(scale_count) if total is not None and scale_count is not None else None
    nuisance = best.get("nuisance") if isinstance(best.get("nuisance"), Mapping) else {}
    split = np.asarray(rows["split"])
    if kind == "pspline":
        scale = {"kind": "pspline_edf", "value": float(best.get("scale_edf", scale_model.get("edf", math.nan))),
                 "n_basis": scale_model.get("n_basis"), "lambda": scale_model.get("lambda")}
    else:
        scale = {"kind": "per_frame_count", "value": None if scale_count is None else int(scale_count),
                 "frame": best.get("frame_scale_count"), "detector": best.get("detector_scale_count")}
    return {
        "rows": int(len(split)),
        "work_rows": int(np.count_nonzero(split == "work")),
        "free_rows": int(np.count_nonzero(split == "free")),
        "scale": scale,
        "physics_nuisance_parameters": physics,
        "physics_nuisance_rule": "engine accounting: native parameter_count minus scale count (7 base terms plus active optional terms)",
        "named_nuisance": {field: nuisance.get(field) for field in NUISANCE_FIELDS
                           + (GAMMA_NUISANCE_FIELDS if nuisance.get("thickness_model") == "gamma" else ())
                           if field in nuisance},
    }


def combined_ledger(crystals: Mapping[str, Mapping[str, object]], shelxl: Mapping[str, object] | None) -> dict[str, object]:
    """Combine the per-crystal ledgers with the SHELXL parameters and restraints.

    Scale EDF and scale counts are summed separately and never added to the integer parameter count.
    """

    edf = [c["scale"]["value"] for c in crystals.values() if c["scale"]["kind"] == "pspline_edf"]
    counts = [c["scale"]["value"] for c in crystals.values() if c["scale"]["kind"] == "per_frame_count"]
    physics = [c["physics_nuisance_parameters"] for c in crystals.values()]
    structure = None if not shelxl else shelxl.get("parameters")
    restraints = None if not shelxl else shelxl.get("restraints")
    physics_total = None if any(v is None for v in physics) else int(sum(physics))
    integer_total = None
    if structure is not None and physics_total is not None:
        integer_total = int(structure) + physics_total
    return {
        "rows": int(sum(c["rows"] for c in crystals.values())),
        "shelxl_parameters": structure,
        "shelxl_restraints": restraints,
        "physics_nuisance_parameters": physics_total,
        "integer_parameters_shelxl_plus_physics": integer_total,
        "scale_edf_total": float(sum(edf)) if edf else None,
        "scale_count_total": int(sum(v for v in counts if v is not None)) if counts else None,
        "rule": "scale EDF and scale counts are reported separately and never added to integer parameter counts",
    }


def build_heldout_report(
    out_dir: Path,
    result: Mapping[str, object],
    row_tables: list[tuple[str, Path]],
    *,
    publication_shelxl: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compute the held-out block from the final row tables and run artifacts under out_dir."""

    out_dir = Path(out_dir)
    report: dict[str, object] = {"schema": SCHEMA, "units": "percent (R) and pp (deltas and gaps)"}
    report["phase_timings"] = _quick_progress_phase_timings(out_dir)
    report["hand"] = hand_block(out_dir)
    report["shelxl_trajectory"] = trajectory_block(out_dir, result)
    report["nested_nuisance"] = nested_nuisance_report(out_dir, report["shelxl_trajectory"])
    # SHELXL endpoint on held-out orbits (reporting only): zero-cycle scores of the final models on the free orbits
    try:
        report["shelxl_endpoint_free"] = endpoint_free_block(out_dir, result)
    except Exception as exc:  # reporting only: never costs the held-out report
        report["shelxl_endpoint_free"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "rows": []}
    if not row_tables:
        report.update(status="not_available", reason="no final native row tables were recorded")
        return report
    ops, key_rule = _model_symmetry(result)
    tables, summaries = {}, {}
    for name, path in row_tables:
        tables[name] = read_row_table(Path(path))
        summaries[name] = try_read_json(Path(path).parent / "summary.json")
    exclusion = _excluded_orbits(out_dir, list(summaries.values()))
    if exclusion and key_rule.startswith("Friedel pairs only"):
        # Friedel-only keys would not match Laue-orbit exclusion keys and could report excluded orbits.
        raise ValueError(f"orbit exclusion {exclusion['path']} needs the final model Laue group; {key_rule}")
    keys = {name: heldout.orbit_keys(t["hkl"], ops, include_friedel=True) for name, t in tables.items()}
    dropped = {}
    if exclusion:
        for name in tables:
            keep = np.asarray([k not in exclusion["orbits"] for k in keys[name]], dtype=bool)
            dropped[name] = int(np.count_nonzero(~keep))
            tables[name] = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in tables[name].items()}
            keys[name] = keys[name][keep]
    shell_dmin = _shell_dmin(np.concatenate([t["d"] for t in tables.values()]), exclusion)
    hkl_leak, hkl_evidence = shelxl_hkl_free_rows(out_dir, result)
    selection = nuisance_selection_records(out_dir, result, list(tables))
    policy_leak, policy_evidence = dataset_policy_leak(out_dir, result, len(tables))
    hand_leak, hand_evidence = _hand_used_free_rows(report["hand"])
    macro_leak, macro_evidence = macro_cycle_selection_leak(report["shelxl_trajectory"])
    crystals = {}
    ledger = {}
    for name, t in tables.items():
        shell = heldout.shell_index(t["d"], shell_dmin)
        free = t["split"] == "free"
        work = t["split"] == "work"
        held = _split_block(t["y"], t["sig"], t["pred"], keys[name], shell, free)
        insample = heldout.r_metrics(t["y"], t["sig"], t["pred"], work)
        best = summaries[name].get("best_native") if isinstance(summaries[name].get("best_native"), Mapping) else {}
        config = summaries[name].get("config") if isinstance(summaries[name].get("config"), Mapping) else {}
        fit_split = str(best.get("scale_fit_split") or config.get("scale_fit_split") or "unknown")
        scale_model = best.get("scale_model") if isinstance(best.get("scale_model"), Mapping) else {}
        crystals[name] = {
            "row_table": str(dict(row_tables)[name]),
            "sigma_column": t["sigma_column"],
            "heldout": held,
            "in_sample": insample,
            "gap": _gap(held, insample),
            "scale_model": str(scale_model.get("kind") or config.get("scale_model") or "per-frame"),
            "scale_fit_split": fit_split,
            "leaks": {
                "scale_fit": fit_split != "work",
                "nuisance_selection": None if selection.get(name) is None else bool(selection[name]),
                "nuisance_selection_stages": selection.get(name) or [],
                "dataset_policy": policy_leak,
                "shelxl_hkl": hkl_leak,
                "hand_decision": hand_leak,
                "macro_cycle_selection": macro_leak,
            },
            "excluded_orbit_rows": dropped.get(name, 0),
        }
        ledger[name] = ledger_entry(summaries[name], t)
    names = list(tables)
    cat = lambda key: np.concatenate([tables[n][key] for n in names])  # noqa: E731
    y, sig, pred, split, d = cat("y"), cat("sig"), cat("pred"), cat("split"), cat("d")
    pooled_keys = np.concatenate([keys[n] for n in names])
    shell = heldout.shell_index(d, shell_dmin)
    held = _split_block(y, sig, pred, pooled_keys, shell, split == "free")
    insample = heldout.r_metrics(y, sig, pred, split == "work")
    report.update(
        {
            "status": "available",
            "definitions": {
                "heldout_rows": "split == free in the final native row tables",
                "prediction": "run dyn_scaled (scales as fitted by the run)",
                "r_factor": "amplitude R1 on sqrt(max(I, 0)); R_obs uses Io > 3 source_sigma",
                "ci": f"single-arm merged-orbit bootstrap, {heldout.N_BOOT} shell-stratified resamples, seed {heldout.BOOTSTRAP_SEED}",
                "orbit_keys": key_rule,
                "shells_d_min": shell_dmin,
                "gap": "held-out minus in-sample (work rows, same model)",
            },
            "free_split": _free_split_record(summaries),
            "excluded_orbits": None if not exclusion else {"path": exclusion["path"], "sha256": exclusion["sha256"],
                                                           "rows_dropped": dropped},
            "crystals": crystals,
            "pooled": {"heldout": held, "in_sample": insample, "gap": _gap(held, insample)},
            "leaks": {
                "scale_fit": any(c["leaks"]["scale_fit"] for c in crystals.values()),
                "nuisance_selection": _pooled_flag([c["leaks"]["nuisance_selection"] for c in crystals.values()]),
                "dataset_policy": policy_leak,
                "dataset_policy_evidence": policy_evidence,
                "shelxl_hkl": hkl_leak,
                "shelxl_hkl_evidence": hkl_evidence,
                "hand_decision": hand_leak,
                "hand_decision_evidence": hand_evidence,
                "macro_cycle_selection": macro_leak,
                "macro_cycle_selection_evidence": macro_evidence,
                **start_model_leak(result),
            },
            "parameter_ledger": {"crystals": ledger, "combined": combined_ledger(ledger, publication_shelxl)},
            "scale_contract_consistency": scale_contract_consistency(out_dir, tables, summaries),
        }
    )
    return report


def _free_split_record(summaries: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Return each crystal's free-split settings (scheme, salt, free fraction, seed) from its run config."""

    record = {}
    for name, summary in summaries.items():
        config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
        record[name] = {
            "split_scheme": config.get("split_scheme", "legacy"),
            "split_salt": config.get("split_salt"),
            "free_fraction": config.get("free_fraction"),
            "split_seed": config.get("split_seed"),
        }
    return record


def _pooled_flag(flags: list[bool | None]) -> bool | None:
    """Any yes is yes; otherwise an unknown crystal leaves the pooled flag unknown."""

    if any(flag is True for flag in flags):
        return True
    return None if any(flag is None for flag in flags) else False


def _hand_stage_used_free_rows(doc: Mapping[str, object]) -> bool:
    """Return whether one hand-decision stage read free rows: free_rows_used when recorded, else not skipped."""

    explicit = doc.get("free_rows_used")
    if isinstance(explicit, bool):
        return explicit
    # A skipped stage evaluated no hand pair. Otherwise heldout-orbit-sign reads the free rows by design and the
    # legacy all-row-zkp rule reads every row, free rows included (final_hand_selection zkp_switch on the all-row
    # adjusted Z); an unknown rule is assumed to read them.
    return not str(doc.get("status", "")).lower().startswith("skipped")


def _hand_used_free_rows(hand: Mapping[str, object] | None) -> tuple[bool | None, str]:
    """Return (any hand stage read free rows, evidence), or (None, reason) when no hand record exists."""

    if not hand:
        return None, "no hand-decision record found"
    stages = hand["stages"]
    flag = any(_hand_stage_used_free_rows(doc) for doc in stages.values())
    evidence = "; ".join(
        f"{stage}: {doc.get('evidence', 'unrecorded rule')}"
        + (f" ({doc.get('status')})" if str(doc.get("status", "")).lower().startswith("skipped") else "")
        for stage, doc in stages.items()
    )
    return flag, evidence + " (decisions/hand_decision_heldout_*.json)"


# ----------------------------------------------------------------------------- rendering

def _num(value: object, digits: int = 2) -> str:
    """Format a number with fixed digits, n/a when it is not a finite number."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{out:.{digits}f}" if math.isfinite(out) else "n/a"


def _signed_num(value: object, digits: int = 2) -> str:
    """Format a number with a sign and fixed digits, n/a when it is not a finite number."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{out:+.{digits}f}" if math.isfinite(out) else "n/a"


def _ci(block: Mapping[str, object], name: str) -> str:
    """Format block[name] with its 95% CI as 'value [lo, hi]', or the value alone when no CI was recorded."""

    ci = block.get(f"{name}_ci95")
    value = _num(block.get(name))
    if isinstance(ci, list) and len(ci) == 2:
        return f"{value} [{_num(ci[0])}, {_num(ci[1])}]"
    return value


def _yes(value: object) -> str:
    """Format a flag as yes or no, n/a for None."""

    return "n/a" if value is None else ("yes" if value else "no")


def heldout_markdown_lines(report: Mapping[str, object] | None) -> list[str]:
    """Markdown for the "Held-out validation" section (QUICK_STATS and FINAL_SUMMARY)."""

    if not report:
        return []
    lines = ["## Held-out validation", "", *preset_markdown_lines(report.get("preset"))]
    if report.get("status") != "available":
        lines += [f"Not available: {report.get('reason') or report.get('error') or report.get('status')}.", ""]
    else:
        definitions = report.get("definitions") or {}
        lines += [
            f"Held-out rows are the `split=free` rows of the final row tables; predictions are the run's own scaled "
            f"Bloch rows. R is amplitude R1 in percent (R_obs: Io > 3 sigma); 95% CIs: {definitions.get('ci')}; "
            f"orbits: {definitions.get('orbit_keys')}. Gap = held-out minus in-sample (work rows, same model).",
            "",
            "| crystal | free rows/orbits | held-out R_all [95% CI] | held-out R_obs [95% CI] | in-sample R_all / R_obs | gap R_all / R_obs (pp) | scale model / fit rows |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        rows = [(name, c) for name, c in (report.get("crystals") or {}).items()]
        rows.append(("pooled", {**(report.get("pooled") or {}), "scale_model": "", "scale_fit_split": ""}))
        for name, c in rows:
            held, ins, gap = c.get("heldout") or {}, c.get("in_sample") or {}, c.get("gap") or {}
            scale = f"{c.get('scale_model')} / {c.get('scale_fit_split')}" if c.get("scale_model") else ""
            lines.append(
                f"| {name} | {held.get('n_all', 'n/a')}/{held.get('n_orbits', 'n/a')} | {_ci(held, 'r_all')} | "
                f"{_ci(held, 'r_obs')} | {_num(ins.get('r_all'))} / {_num(ins.get('r_obs'))} | "
                f"{_num(gap.get('r_all'))} / {_num(gap.get('r_obs'))} | {scale} |"
            )
        leaks = report.get("leaks") or {}
        stages = sorted({s for c in (report.get("crystals") or {}).values() for s in c["leaks"]["nuisance_selection_stages"]})
        fit_splits = ", ".join(f"{n}={c['scale_fit_split']}" for n, c in (report.get("crystals") or {}).items())
        lines += [
            "",
            "Leak flags (were free rows used anyway?):",
            "",
            "| where | free rows used | evidence |",
            "|---|---|---|",
            f"| frame-scale fit | {_yes(leaks.get('scale_fit'))} | scale-fit split: {fit_splits} |",
            f"| nuisance selection | {_yes(leaks.get('nuisance_selection'))} | "
            f"{', '.join(stages) or ('no nuisance decision record found' if leaks.get('nuisance_selection') is None else 'no gate read free rows')} |",
            f"| dataset policy | {_yes(leaks.get('dataset_policy'))} | {leaks.get('dataset_policy_evidence', 'no dataset-policy record')} |",
            f"| SHELXL HKL (structure refinement) | {_yes(leaks.get('shelxl_hkl'))} | {leaks.get('shelxl_hkl_evidence')} |",
            f"| macro-cycle model selection | {_yes(leaks.get('macro_cycle_selection'))} | "
            f"{leaks.get('macro_cycle_selection_evidence', 'no SHELXL feedback loop summary')} |",
            f"| hand decision | {_yes(leaks.get('hand_decision'))} | {leaks.get('hand_decision_evidence', 'no hand-decision record')} |",
            f"| start model | {_yes(leaks.get('start_model'))} | {leaks.get('start_model_evidence', 'no leak-free start record')} |",
            "",
        ]
        if report.get("excluded_orbits"):
            ex = report["excluded_orbits"]
            lines += [f"Excluded orbits (`{ex['path']}`, sha256 `{ex['sha256'][:12]}`) were dropped: {ex['rows_dropped']}.", ""]
    lines += endpoint_free_markdown_lines(report.get("shelxl_endpoint_free"))
    lines += _consistency_lines(report.get("scale_contract_consistency"))
    lines += _hand_lines(report.get("hand"), summary=report.get("report_layout") == "heldout-first")
    lines += nested_nuisance_lines(report.get("nested_nuisance"))
    lines += _trajectory_lines(report.get("shelxl_trajectory"))
    lines += _ledger_lines(report.get("parameter_ledger"))
    lines += _phase_lines(report.get("phase_timings"))
    return lines


def _consistency_lines(block: Mapping[str, object] | None) -> list[str]:
    """Return the markdown table comparing the feedback and publication frame-scale contracts.

    Empty when no feedback cycle was recorded.
    """

    if not block or not any(entry.get("feedback_last_adopted_cycle") for entry in block.values()):
        return []

    def text(contract):
        """Format one scale contract as 'kind size on split rows', with its compact replay models when any."""

        if not contract:
            return "n/a"
        size = f"EDF {_num(contract.get('scale_edf'))}" if contract["kind"] == "pspline" else f"{contract.get('scale_count')} scales"
        replay = f", compact replay {', '.join(contract['compact_replay'])}" if contract.get("compact_replay") else ""
        return f"{contract['kind']} {size} on {contract.get('scale_fit_split')} rows{replay}"

    lines = ["### Frame-scale contract: structure feedback versus publication", "",
             "| crystal | last adopted feedback cycle | publication row table | same contract |", "|---|---|---|---|"]
    for name, entry in block.items():
        lines.append(f"| {name} | {text(entry.get('feedback_last_adopted_cycle'))} | {text(entry.get('publication'))} | "
                     f"{_yes(entry.get('same_scale_contract'))} |")
    return lines + [""]


def _hand_lines(hand: Mapping[str, object] | None, *, summary: bool = False) -> list[str]:
    """Return the markdown hand section: optional plain summary, then pooled and per-crystal evidence per stage."""

    if not hand:
        return []
    sentences = hand_summary_sentences(hand) if summary else []
    lines = [
        "### Hand decision on held-out rows",
        "",
        *[f"- {'**' + text + '**' if text.startswith('WARNING') else text}" for text in sentences],
        *([""] if sentences else []),
        "| stage | block | evidence | selected | switched | orbit sign z | orbits model/inverse better | dR_obs inverse-selected [95% CI] | dR_all [95% CI] |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for stage, doc in hand["stages"].items():
        blocks = [("pooled", doc.get("pooled") or {})] + list((doc.get("per_crystal") or {}).items())
        for label, block in blocks:
            if not isinstance(block, Mapping):
                continue
            dro, dra = block.get("delta_r_obs") or {}, block.get("delta_r_all") or {}
            lines.append(
                f"| {stage} | {label} | {doc.get('evidence', 'n/a')} | {doc.get('selected_hand', 'n/a')} | "
                f"{doc.get('switched', 'n/a')} | {_num(block.get('orbit_sign_z'))} | "
                f"{block.get('orbits_model_better', 'n/a')}/{block.get('orbits_inverse_better', 'n/a')} | "
                f"{_num(dro.get('delta'))} [{_num((dro.get('ci95') or [None, None])[0])}, {_num((dro.get('ci95') or [None, None])[1])}] | "
                f"{_num(dra.get('delta'))} [{_num((dra.get('ci95') or [None, None])[0])}, {_num((dra.get('ci95') or [None, None])[1])}] |"
            )
    agree = hand.get("early_final_agree")
    lines += ["", f"Early and final hand decisions agree: {_yes(agree)}.", ""]
    return lines


def _trajectory_lines(block: Mapping[str, object] | None) -> list[str]:
    """Return the markdown SHELXL feedback trajectory section, one table row per macro-cycle."""

    if not block:
        return []
    lines = [
        "### SHELXL feedback trajectory",
        "",
        f"Status `{block.get('status')}`, stop reason `{block.get('stop_reason')}`, kept cycle `{block.get('kept_cycle')}`"
        + (f" (keep policy `{block.get('keep_policy')}`: {block.get('kept_reason')})" if block.get("keep_policy") else "")
        + f"; values as recorded in `{block.get('source')}`.",
        "",
    ]
    rows = block.get("trajectory") or []
    if not rows:
        return lines + [f"{block.get('note', 'no per-cycle trajectory recorded')}.", ""]
    lines += ["| " + " | ".join(TRAJECTORY_COLUMNS) + " |", "|" + "---:|" * len(TRAJECTORY_COLUMNS)]
    for row in rows:
        cells = []
        for column in TRAJECTORY_COLUMNS:
            value = row.get(column)
            cells.append(str(value) if isinstance(value, (bool, int, str)) or value is None else _num(value, 4))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def _ledger_lines(ledger: Mapping[str, object] | None) -> list[str]:
    """Return the markdown parameter-ledger table, per crystal and combined."""

    if not ledger:
        return []
    lines = [
        "### Parameter ledger",
        "",
        "| crystal | rows (work/free) | frame scales | physics nuisance parameters | nuisance values |",
        "|---|---:|---:|---:|---|",
    ]
    for name, entry in (ledger.get("crystals") or {}).items():
        scale = entry["scale"]
        scale_text = (f"EDF {_num(scale['value'])} (n_basis {scale.get('n_basis')}, lambda {scale.get('lambda')})"
                      if scale["kind"] == "pspline_edf" else f"{scale['value']} free scales")
        values = ", ".join(f"{k}={v}" for k, v in entry["named_nuisance"].items())
        lines.append(
            f"| {name} | {entry['rows']} ({entry['work_rows']}/{entry['free_rows']}) | {scale_text} | "
            f"{entry['physics_nuisance_parameters']} | {values} |"
        )
    combined = ledger.get("combined") or {}
    scale_parts = []
    if combined.get("scale_edf_total") is not None:
        scale_parts.append(f"EDF {_num(combined['scale_edf_total'])}")
    if combined.get("scale_count_total") is not None:
        scale_parts.append(f"{combined['scale_count_total']} free scales")
    lines += [
        f"| combined | {combined.get('rows')} | {' + '.join(scale_parts) or 'n/a'} | {combined.get('physics_nuisance_parameters')} | "
        f"SHELXL parameters/restraints {combined.get('shelxl_parameters')}/{combined.get('shelxl_restraints')}; "
        f"integer total (SHELXL + physics) {combined.get('integer_parameters_shelxl_plus_physics')} |",
        "",
        "Scale EDF and scale counts are reported separately and never added to integer parameter counts.",
        "",
    ]
    return lines


def _phase_lines(timings: Mapping[str, object] | None) -> list[str]:
    """Return the markdown phase wall-time table, empty when no timings are available."""

    if not timings or timings.get("status") != "available":
        return []
    lines = ["### Phase wall time", "", "| phase | wall | wall s | summed stage s | stages |", "|---|---:|---:|---:|---:|"]
    for row in timings.get("phases") or []:
        lines.append(f"| {row['phase']} | {row['wall']} | {_num(row['wall_seconds'], 0)} | {_num(row['stage_seconds'], 0)} | {row['stages']} |")
    lines += [
        f"| outside stages | | {_num(timings.get('outside_stage_wall_seconds'), 0)} | | |",
        f"| total | | {_num(timings.get('total_wall_seconds'), 0)} | {_num(timings.get('cumulative_stage_seconds'), 0)} | |",
        "",
        "Wall seconds are the union of stage intervals from event timestamps (1 s resolution); overlapping stages count once.",
        "",
    ]
    return lines


def heldout_report_for_run(
    out_dir: Path,
    result: Mapping[str, object],
    row_tables: list[tuple[str, Path]],
    *,
    publication_shelxl: Mapping[str, object] | None = None,
    report_layout: str = "legacy",
) -> dict[str, object]:
    """Build and write the block; a failure is recorded in the report instead of ending the run's reporting.

    ``report_layout`` heldout-first adds the plain-language hand summary (hand_summary_sentences) to the hand
    section; legacy leaves the hand section without it.
    """

    try:
        report = build_heldout_report(out_dir, result, row_tables, publication_shelxl=publication_shelxl)
    except Exception as exc:
        report = {"schema": SCHEMA, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    report["report_layout"] = str(report_layout)
    # The named preset (--preset) the run's settings came from; standard for runs that predate the option.
    preset = try_read_json(Path(out_dir) / "run_config.json").get("preset")
    report["preset"] = preset if isinstance(preset, Mapping) else {"name": "standard"}
    report["artifacts"] = write_heldout_report(out_dir, report)
    return report


def write_heldout_report(out_dir: Path, report: Mapping[str, object]) -> dict[str, str]:
    """Write HELDOUT_VALIDATION.json/.md at the run root."""

    json_path = Path(out_dir) / "HELDOUT_VALIDATION.json"
    md_path = Path(out_dir) / "HELDOUT_VALIDATION.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    md_path.write_text("\n".join(["# GROND Held-Out Validation", "", *heldout_markdown_lines(report)]) + "\n")
    return {"json": str(json_path), "markdown": str(md_path)}


def _json_default(value):
    """Convert NumPy scalars and arrays, and sets, for json.dumps; any other type raises TypeError."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def table1_panel_c(report: Mapping[str, object] | None) -> dict[str, object] | None:
    """Table 1 Panel C records in fractions (the Table 1 numeric contract)."""

    if not report or report.get("status") != "available":
        return None if not report else {"status": "not_available", "reason": report.get("reason") or report.get("error")}

    def fraction(value):
        """Return a percent value as a fraction, None when it is not a finite number."""

        try:
            out = float(value) / 100.0
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    def record(label, block, leaks):
        """Return one Panel C row (fractions) of a crystal or the pooled block, with its leak flags."""

        held, ins, gap = block.get("heldout") or {}, block.get("in_sample") or {}, block.get("gap") or {}
        return {
            "crystal": label,
            "n_free": held.get("n_all"),
            "n_free_obs": held.get("n_obs"),
            "n_orbits": held.get("n_orbits"),
            "heldout_Rall": fraction(held.get("r_all")),
            "heldout_Rall_ci95": [fraction(v) for v in held.get("r_all_ci95") or []] or None,
            "heldout_Robs": fraction(held.get("r_obs")),
            "heldout_Robs_ci95": [fraction(v) for v in held.get("r_obs_ci95") or []] or None,
            "insample_Rall": fraction(ins.get("r_all")),
            "insample_Robs": fraction(ins.get("r_obs")),
            "gap_Rall": fraction(gap.get("r_all")),
            "gap_Robs": fraction(gap.get("r_obs")),
            "free_rows_in_scale_fit": leaks.get("scale_fit"),
            "free_rows_in_nuisance_selection": leaks.get("nuisance_selection"),
            "free_rows_in_dataset_policy": leaks.get("dataset_policy"),
            "free_rows_in_shelxl_hkl": leaks.get("shelxl_hkl"),
            "free_rows_in_macro_cycle_selection": leaks.get("macro_cycle_selection"),
            "free_rows_in_hand_decision": leaks.get("hand_decision"),
        }

    rows = [record(name, c, c.get("leaks") or {}) for name, c in (report.get("crystals") or {}).items()]
    rows.append(record("pooled", report.get("pooled") or {}, report.get("leaks") or {}))
    leaks = report.get("leaks") or {}
    return {"status": "available", "rows": rows, "definitions": report.get("definitions"),
            "shelxl_hkl_evidence": leaks.get("shelxl_hkl_evidence"),
            "macro_cycle_selection_evidence": leaks.get("macro_cycle_selection_evidence"),
            "hand_decision_evidence": leaks.get("hand_decision_evidence"),
            "dataset_policy_evidence": leaks.get("dataset_policy_evidence")}


LEAK_FLAG_LABELS = (
    ("scale_fit", "frame-scale fit"),
    ("nuisance_selection", "nuisance selection"),
    ("dataset_policy", "dataset policy"),
    ("shelxl_hkl", "SHELXL HKL"),
    ("macro_cycle_selection", "macro-cycle selection"),
    ("hand_decision", "hand decision"),
    ("start_model", "start model"),
)


def terminal_block(report: Mapping[str, object] | None) -> dict[str, object]:
    """The subset of HELDOUT_VALIDATION the terminal finish summary prints, copied from the report (no recomputation)."""

    if not report:
        return {"status": "not_recorded"}
    out: dict[str, object] = {
        "status": report.get("status"),
        "reason": report.get("reason") or report.get("error"),
        "artifact": (report.get("artifacts") or {}).get("markdown") if isinstance(report.get("artifacts"), Mapping) else None,
        "hand_summary": hand_summary_sentences(report.get("hand")),
    }
    preset = report.get("preset") if isinstance(report.get("preset"), Mapping) else {}
    if preset.get("name") not in (None, "standard"):
        out["preset"] = {"name": preset.get("name"), "equivalent_flags": list(preset.get("equivalent_flags") or [])}
    if report.get("status") == "available":
        rows = []
        crystals = report.get("crystals") or {}
        for name, block in [*crystals.items(), ("pooled", report.get("pooled") or {})]:
            held, ins, gap = block.get("heldout") or {}, block.get("in_sample") or {}, block.get("gap") or {}
            rows.append(
                {
                    "crystal": name,
                    "free_rows": held.get("n_all"),
                    "free_orbits": held.get("n_orbits"),
                    "r_obs": held.get("r_obs"),
                    "r_obs_ci95": held.get("r_obs_ci95"),
                    "r_all": held.get("r_all"),
                    "r_all_ci95": held.get("r_all_ci95"),
                    "in_sample_r_obs": ins.get("r_obs"),
                    "in_sample_r_all": ins.get("r_all"),
                    "gap_r_obs": gap.get("r_obs"),
                    "gap_r_all": gap.get("r_all"),
                }
            )
        out["rows"] = rows
        leaks = report.get("leaks") or {}
        out["leaks"] = [{"label": label, "value": leaks.get(key)} for key, label in LEAK_FLAG_LABELS]
    endpoint_free = report.get("shelxl_endpoint_free")
    if isinstance(endpoint_free, Mapping):
        out["shelxl_endpoint_free_line"] = endpoint_free_terminal_line(endpoint_free)
    trajectory = report.get("shelxl_trajectory") if isinstance(report.get("shelxl_trajectory"), Mapping) else None
    if trajectory:
        rows = [row for row in trajectory.get("trajectory") or [] if isinstance(row, Mapping)]
        out["shelxl_loop"] = {
            "status": trajectory.get("status"),
            "stop_reason": trajectory.get("stop_reason"),
            "kept_cycle": trajectory.get("kept_cycle"),
            "cycles": len(rows),
            "last": {key: rows[-1].get(key) for key in TRAJECTORY_COLUMNS} if rows else {},
        }
    timings = report.get("phase_timings") if isinstance(report.get("phase_timings"), Mapping) else {}
    if timings.get("status") == "available":
        out["phase_timings"] = {
            "phases": [
                {"phase": row.get("phase"), "wall_seconds": row.get("wall_seconds")}
                for row in timings.get("phases") or []
                if isinstance(row, Mapping)
            ],
            "total_wall_seconds": timings.get("total_wall_seconds"),
        }
    return out
