"""Score-only replay of a published single-crystal corrected-HKL endpoint (``grond validate-endpoint --endpoint published``).

A single-crystal AUTO run publishes a per-crystal endpoint of the native engine (source_kind ``native_corrected_hkl``):
``<cycle>/endpoint_lam_<l>_B_<b>_<exti>/dyn_nuisance_corrected.{res,hkl,lst}``, which export_corrected_endpoint
(engine/native_row_regularize.py) wrote from the rows of the same command's row table ``<cycle>/native_dyn_calc.csv``.
The command's record is ``<cycle>/summary.json`` (``config``: its arguments; ``inputs``: the entering model and its SHA-256;
``artifacts``: the row table and its SHA-256) and the endpoint's ``manifest.json`` (lambda, q policy, the error-model passes
and the one kept).  The replay reads those records and predicts and refines nothing, like the combine engine's
``--score-res``:

1. work rows: export_corrected_endpoint's recipe on the recorded rows (its valid-row rule, the divisors of
   endpoint_correction_denominators with |Fcalc|^2 of the entering model, the exact-HKL merge, the fixed-width HKL
   factor, the error-model pass the endpoint kept, refitted on the same rows against the model that pass was fitted to).
   The rebuilt HKL must equal the published HKL byte for byte, otherwise nothing is scored;
2. free rows: the rows the table marks free, corrected by the same recipe with the work rows' divisor median as the
   q-exclusion reference (the multi-crystal free validation's rule), given the kept error model's sigmas
   (apply_error_model_sigma: nothing is fitted on them), merged by exact HKL like the work rows and put on the work
   HKL's scale; no free row may share a symmetry class (model symmetry and Friedel) with a work row;
3. the published .res is scored with SHELXL L.S. 0 on the published HKL plus the free rows flagged -1, with the WGHT its
   last refinement applied (grond.validation.endpoint_free.score_model_on_work_and_free).

Supported: the AUTO contract (``--rescale absolute``, free rows kept out of the HKL, q policy exclude, endpoint B 0,
error-model or no post-Bloch sigma policy).  The legacy median-IOBS rescale needs the per-row fitted scales, which the
row table does not record; it, the legacy q clip of older runs and the uniform_z_rms policy are not replayed: those give a
``not_supported_*`` status.  Reporting only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from grond.validation.endpoint_free import record_endpoint_reproduction, score_model_on_work_and_free

ENDPOINT_STEM = "dyn_nuisance_corrected"
FREE_VALIDATION_DIR = "free_validation"
SUMMARY_NAME = "score_only_summary.json"
SUPPORTED_SIGMA_POLICIES = ("error_model", "none")
_FLOAT_COLUMNS = ("target_I", "source_to_target_factor", "sigma_eff", "dyn_raw", "born_raw", "dyn_scaled")


class ReplayRefused(Exception):
    """The published endpoint cannot be replayed; ``status`` names why."""

    def __init__(self, status: str, reason: str):
        """Keep the status (the record's ``status``) and the reason (the exception message and ``reason``)."""

        super().__init__(reason)
        self.status = status
        self.reason = reason


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    """Return a JSON object file, or refuse the replay when it is missing or not an object."""

    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayRefused("published_endpoint_record_incomplete", f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayRefused("published_endpoint_record_incomplete", f"{path} is not a JSON object")
    return value


def read_row_table(path: Path) -> dict[str, np.ndarray]:
    """h, k, l, the free mask and the columns export_corrected_endpoint reads, in the table's (row_index) order."""

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ReplayRefused("published_endpoint_record_incomplete", f"{path} has no rows")
    missing = [name for name in ("h", "k", "l", "split", *_FLOAT_COLUMNS) if name not in rows[0]]
    if missing:
        raise ReplayRefused("published_endpoint_record_incomplete", f"{path} lacks the columns {', '.join(missing)}")
    table = {name: np.asarray([float(row[name]) for row in rows], dtype=float) for name in _FLOAT_COLUMNS}
    table["hkl"] = np.asarray([(int(row["h"]), int(row["k"]), int(row["l"])) for row in rows], dtype=int).reshape(-1, 3)
    table["free"] = np.asarray([str(row["split"]) == "free" for row in rows], dtype=bool)
    return table


def replay_contract(config: Mapping[str, object], manifest: Mapping[str, object]) -> tuple[str, str] | None:
    """None when the replay covers this endpoint's recipe, else (status, reason)."""

    if str(config.get("rescale")) != "absolute":
        return ("not_supported_legacy_rescale",
                f"--rescale {config.get('rescale')}: the median-IOBS rescale divides by the per-row fitted scales, which "
                "the row table does not record")
    if bool(config.get("endpoint_include_free_rows")):
        return ("not_supported_free_rows_in_hkl", "the endpoint HKL was written with the free rows: nothing is held out")
    q_policy = str(manifest.get("q_policy") or config.get("endpoint_q_policy"))
    if q_policy != "exclude":
        return ("not_supported_legacy_q_policy", f"--endpoint-q-policy {q_policy}: only the exclude policy is replayed")
    try:
        endpoint_b = float(manifest["endpoint_b"])
    except (KeyError, TypeError, ValueError, OverflowError):
        endpoint_b = float("nan")
    if not np.isfinite(endpoint_b) or endpoint_b != 0.0:
        return ("not_supported_endpoint_b", f"endpoint B {manifest.get('endpoint_b')}: only a finite recorded B of 0 is replayed")
    if str(config.get("hkl_float_format") or "shelx_fixed") != "shelx_fixed":
        return ("not_supported_hkl_format", f"--hkl-float-format {config.get('hkl_float_format')}: the scoring run needs "
                "the fixed-width HKLF 4 layout (free-R flag in columns 29-32)")
    policy = str(config.get("post_bloch_sigma_policy") or "none")
    if policy not in SUPPORTED_SIGMA_POLICIES:
        return ("not_supported_sigma_policy", f"--post-bloch-sigma-policy {policy} is not replayed")
    return None


def kept_error_model_pass(manifest: Mapping[str, object], entering_model: Path) -> tuple[int, Path]:
    """The error-model pass the endpoint kept and the model its error model was fitted against.

    export_corrected_endpoint fits pass 1 against the entering model and pass p > 1 against the model SHELXL refined in
    pass p - 1; each pass records that model as ``sigma_model`` and exactly one pass is ``kept``.  Without passes (no
    refit) the fit is pass 1's.
    """

    record = manifest.get("post_bloch_sigma") if isinstance(manifest.get("post_bloch_sigma"), Mapping) else {}
    passes = [item for item in record.get("passes") or [] if isinstance(item, Mapping)]
    if not passes:
        return 1, Path(entering_model)
    kept = [item for item in passes if item.get("kept") is True]
    if len(kept) != 1:
        raise ReplayRefused("published_endpoint_record_incomplete", f"the manifest records {len(kept)} kept error-model passes")
    sigma_model = Path(str(kept[0].get("sigma_model") or ""))
    if int(kept[0]["pass"]) == 1 and sigma_model.resolve() != Path(entering_model).resolve():
        raise ReplayRefused("published_endpoint_record_incomplete",
                            f"pass 1's sigma model {sigma_model} is not the command's entering model {entering_model}")
    if not sigma_model.is_file():
        raise ReplayRefused("published_endpoint_record_incomplete", f"the kept pass's sigma model {sigma_model} is missing")
    return int(kept[0]["pass"]), sigma_model


def rebuild_native_endpoint(
    table: Mapping[str, np.ndarray],
    *,
    entering_model: Path,
    sigma_model: Path,
    lambda_q: float,
    q_policy: str,
    hkl_format: str,
    sigma_policy: str,
) -> dict[str, object]:
    """Work HKL rows and corrected free rows of one crystal, by export_corrected_endpoint's recipe (see the module doc).

    Returns the merged work rows as written (``work_merged``), the free rows on the same scale before the merge
    (``free_rows``), the HKL factors and the audits.
    """

    from grond.bloch.model import fcalc_res_grid, parse_model
    from grond.engine.native_row_regularize import _merge_rows, exact_hkl_merge
    from grond.engine.post_bloch_sigma import apply_error_model_sigma, fit_error_model_sigma
    from grond.refine.shelx_input import enforce_hkl_row_width
    from grond.regularize.multidataset import endpoint_correction_denominators

    hkl = table["hkl"]
    factor = table["source_to_target_factor"]
    # endpoint_observation_arrays: the fit target back in IOBS units (rows without a positive factor are dropped)
    restorable = np.isfinite(factor) & (factor > 0.0)
    safe_factor = np.where(restorable, factor, 1.0)
    target = np.where(restorable, table["target_I"] / safe_factor, np.nan)
    sigma = np.where(restorable, table["sigma_eff"] / safe_factor, np.nan)
    dyn_raw, dyn_scaled = table["dyn_raw"], table["dyn_scaled"]
    q = dyn_raw / np.maximum(table["born_raw"], 1e-30)
    # the engine's valid-row rule; its fitted-scale condition (finite, > 0) holds wherever dyn_scaled > 0 and q > 0
    candidates = (
        np.isfinite(target) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(dyn_scaled) & (dyn_scaled > 0.0)
        & np.isfinite(q) & (q > 0.0)
    )
    free_candidates = candidates & table["free"]
    work = candidates & ~table["free"]
    fc2 = np.abs(fcalc_res_grid(parse_model(Path(entering_model)), hkl[:, 0], hkl[:, 1], hkl[:, 2])) ** 2
    denominators, work, work_audit = endpoint_correction_denominators(
        target, dyn_raw, dyn_scaled, q, float(lambda_q), work, q_policy=q_policy, rescale="absolute", fc2=fc2,
        restore=factor,
    )
    intensity, intensity_sigma = target / denominators, sigma / denominators
    unmerged = [(hkl[i, 0], hkl[i, 1], hkl[i, 2], intensity[i], intensity_sigma[i]) for i in np.where(work)[0]]
    merged = exact_hkl_merge(hkl[work], intensity[work], intensity_sigma[work])
    merged, unmerged, base_factor = enforce_hkl_row_width(merged, unmerged, hkl_format)
    free_denominators, free_kept, free_audit = endpoint_correction_denominators(
        target, dyn_raw, dyn_scaled, q, float(lambda_q), free_candidates, q_policy=q_policy, rescale="absolute",
        fc2=fc2, restore=factor, reference_median=float(work_audit["divisor_median"]),
    )
    free_scale = float(base_factor) / np.maximum(free_denominators, 1e-30)
    free_rows = [(hkl[i, 0], hkl[i, 1], hkl[i, 2], target[i] * free_scale[i], sigma[i] * free_scale[i])
                 for i in np.where(free_kept)[0]]
    record: dict[str, object] = {
        "work_correction": work_audit,
        "free_correction": free_audit,
        "work_rows_unmerged": len(unmerged),
        "base_hkl_safe_factor": float(base_factor),
        "free_candidate_rows": int(np.count_nonzero(free_candidates)),
    }
    if sigma_policy == "error_model":
        model = parse_model(Path(sigma_model))
        calibrated, error_model = fit_error_model_sigma(list(unmerged), model)
        reapplied, _scoring = apply_error_model_sigma(list(unmerged), model, error_model)
        free_rows, free_scoring = apply_error_model_sigma(free_rows, model, error_model)
        record.update({
            "error_model": {key: error_model.get(key) for key in ("k", "u", "model_intensity_scale", "fit_rows",
                                                                  "work_rms_z_after")},
            # the arithmetic apply_error_model_sigma gives the free rows returns the fitted work rows bit for bit
            "error_model_reapplied_to_work_rows_bit_for_bit": reapplied == calibrated,
            "free_error_model_scoring": free_scoring,
        })
        unmerged, merged = calibrated, _merge_rows(calibrated)
    merged, unmerged, post_factor = enforce_hkl_row_width(merged, unmerged, hkl_format)
    record.update({"work_merged": merged, "free_rows": free_rows, "post_sigma_hkl_safe_factor": float(post_factor),
                   "hkl_safe_factor": float(base_factor) * float(post_factor)})
    return record


def replay_native_endpoint(published: Mapping[str, object], out_dir: Path, *, shelxl: str) -> dict[str, object]:
    """Rebuild and score a published single-crystal endpoint in ``out_dir`` (the run is only read).

    ``shelxl`` is the SHELXL executable of the score (empty: the one the endpoint command recorded).

    Returns the endpoint record in the combine's score-only shape: ``mode``, ``score_only`` (the scored files, the
    rebuilt HKL and its SHA-256, the listing statistics and the applied WGHT), ``free_validation`` (the L.S. 0 score)
    and ``replay`` (the recorded command, the kept error-model pass and the checks of its inputs).  A refused replay
    raises ReplayRefused.
    """

    from grond.bloch.model import parse_model
    from grond.engine.combine_native_nuisance_runs import _fixed_width_rows, _hkl_lines, score_only_inputs
    from grond.engine.native_row_regularize import _merge_rows
    from grond.refine.shelx_input import write_hkl
    from grond.regularize.multidataset import canonical_model_friedel

    res = Path(str(published["res"]))
    endpoint_dir, cycle_dir = res.parent, res.parent.parent
    origin = Path(str(published.get("endpoint_origin_summary") or cycle_dir / "summary.json"))
    if origin.resolve() != (cycle_dir / "summary.json").resolve():
        raise ReplayRefused("published_endpoint_record_incomplete", f"the endpoint's command record {origin} is not {cycle_dir / 'summary.json'}")
    summary = _read_json(origin)
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    manifest = _read_json(endpoint_dir / "manifest.json")
    recorded = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), Mapping) else {}
    if recorded.get("res_sha256") != published.get("res_sha256") or recorded.get("hkl_sha256") != published.get("hkl_sha256"):
        raise ReplayRefused("published_endpoint_record_incomplete", f"{endpoint_dir / 'manifest.json'} does not record the published endpoint")
    if not config or not str(config.get("res") or ""):
        raise ReplayRefused("published_endpoint_record_incomplete", f"{origin} records no command configuration")
    refused = replay_contract(config, manifest)
    if refused:
        raise ReplayRefused(*refused)
    entering_model = Path(str(config["res"]))
    table_path = Path(str(artifacts.get("native_row_table") or cycle_dir / "native_dyn_calc.csv"))
    inputs_check = {
        "entering_model": {"path": str(entering_model), "recorded_sha256": inputs.get("res_sha256"),
                           "sha256": _sha256(entering_model) if entering_model.is_file() else None},
        "row_table": {"path": str(table_path), "recorded_sha256": artifacts.get("native_row_table_sha256"),
                      "sha256": _sha256(table_path) if table_path.is_file() else None},
    }
    for item in inputs_check.values():
        item["same"] = bool(item["recorded_sha256"]) and item["recorded_sha256"] == item["sha256"]
    if published.get("endpoint_row_table") and Path(str(published["endpoint_row_table"])).resolve() != table_path.resolve():
        inputs_check["row_table"]["same"] = False
    if not all(item["same"] for item in inputs_check.values()):
        raise ReplayRefused("published_endpoint_inputs_changed",
                            "the entering model or row table is missing or its SHA-256 differs from the command's record: "
                            + json.dumps(inputs_check))
    sigma_policy = str(config.get("post_bloch_sigma_policy") or "none")
    kept_pass, sigma_model = kept_error_model_pass(manifest, entering_model) if sigma_policy == "error_model" else (1, entering_model)
    rebuilt = rebuild_native_endpoint(
        read_row_table(table_path), entering_model=entering_model, sigma_model=sigma_model,
        lambda_q=float(manifest.get("lambda_q", 1.0)), q_policy=str(manifest.get("q_policy") or config.get("endpoint_q_policy")),
        hkl_format="shelx_fixed", sigma_policy=sigma_policy,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rebuilt_hkl = out_dir / f"{ENDPOINT_STEM}.hkl"
    write_hkl(rebuilt_hkl, rebuilt["work_merged"], "shelx_fixed")
    score_only = score_only_inputs(res, rebuilt_hkl)
    score_only.update({"error_model_pass": kept_pass, "sigma_model": str(sigma_model)})
    recorded_sigma = manifest.get("post_bloch_sigma") if isinstance(manifest.get("post_bloch_sigma"), Mapping) else {}
    replay = {
        "engine": "native_row_regularize.py export_corrected_endpoint (recipe replayed from the recorded row table)",
        "command_record": str(origin),
        "command": str(cycle_dir / "COMMAND.sh"),
        "config": dict(config),
        "inputs": inputs_check,
        "kept_error_model_pass": kept_pass,
        "sigma_model": str(sigma_model),
        "hkl_safe_factor": {"recorded": manifest.get("hkl_safe_factor"), "rebuilt": rebuilt["hkl_safe_factor"]},
        "error_model": {"recorded": {key: recorded_sigma.get(key) for key in ("k", "u", "model_intensity_scale")},
                        "rebuilt": rebuilt.get("error_model")},
        **{key: rebuilt[key] for key in rebuilt if key not in {"work_merged", "free_rows", "error_model", "hkl_safe_factor"}},
    }
    endpoint: dict[str, object] = {"mode": "score_only", "score_only": score_only, "replay": replay}
    if score_only.get("status") != "ok":
        endpoint["free_validation"] = {"status": str(score_only.get("status"))}
        return endpoint
    free_merged = _merge_rows(rebuilt["free_rows"])
    symmetry_ops = parse_model(res).get("symmetry_ops", [])
    work_keys = {canonical_model_friedel(int(h), int(k), int(ell), symmetry_ops) for h, k, ell, _i, _s in rebuilt["work_merged"]}
    free_keys = {canonical_model_friedel(int(h), int(k), int(ell), symmetry_ops) for h, k, ell, _i, _s in free_merged}
    rows, extra = _fixed_width_rows(free_merged, float(rebuilt["post_sigma_hkl_safe_factor"]))
    directory = out_dir / FREE_VALIDATION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    write_hkl(directory / "free.hkl", rows, "shelx_fixed")
    free_record: dict[str, object] = {
        "endpoint": "single_crystal",
        "free_rows": {
            "candidate_rows": rebuilt["free_candidate_rows"],
            "corrected_rows": len(rebuilt["free_rows"]),
            "correction": rebuilt["free_correction"],
            "error_model_scoring": rebuilt.get("free_error_model_scoring"),
            "merged_rows": len(free_merged),
            "symmetry_classes_also_in_work_hkl": len(free_keys & work_keys),
            "hkl_safe_factor": rebuilt["hkl_safe_factor"],
            "extra_hkl_factor": extra,
        },
        "free_rms_z": (rebuilt.get("free_error_model_scoring") or {}).get("rms_z"),
        "sigma_model": str(sigma_model),
    }
    if not rows:
        endpoint["free_validation"] = {**free_record, "status": "skipped_no_free_rows"}
        return endpoint
    scored = score_model_on_work_and_free(
        directory, res=res, work_hkl=rebuilt_hkl, free_lines=_hkl_lines(rows),
        shelxl=str(shelxl or config.get("shelxl") or "shelxl"), wght_line=score_only["wght"],
        work_rescale=extra,
    )
    record_endpoint_reproduction(scored, score_only["listing_statistics"])
    if free_record["free_rows"]["symmetry_classes_also_in_work_hkl"]:
        scored["status"] = "free_reflections_overlap_work"
    scored["artifacts"] = {
        "directory": str(directory.resolve()),
        "free_hkl": str((directory / "free.hkl").resolve()),
        "free_hkl_sha256": _sha256(directory / "free.hkl"),
        "validation_hkl": str((directory / "validation.hkl").resolve()),
    }
    endpoint["free_validation"] = {**free_record, **scored}
    return endpoint


def write_summary(out_dir: Path, endpoint: Mapping[str, object]) -> Path:
    """Write the replay's endpoint record to ``out_dir/score_only_summary.json`` and return the path."""

    path = Path(out_dir) / SUMMARY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"endpoint": endpoint}, indent=2, allow_nan=True, default=str) + "\n")
    return path
