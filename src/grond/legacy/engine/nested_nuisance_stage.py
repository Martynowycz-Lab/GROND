#!/usr/bin/env python3
"""Nested nuisance stage: Bloch spectra of every physics variant and the joint inner-CV selection on work rows.

One request (schema ``grond.nested-nuisance-request.v1``, written by grond.workflows.nested_nuisance) names the
reflection file, the model, the anchor trial rows of a native run (row identity, targets, sigmas, scan angles,
orbit keys and the engine split), the spectral Config of the anchor candidate at the nested fidelity, the physics
variants (mosaic cone x dose B), the thickness/scale grids and the support. The stage

1. keeps only work rows with a valid anchor prediction: free rows (engine split and orbit-hash free set) and
   excluded orbits are dropped before any spectrum is computed, so no free row enters a spectrum, fit or score;
2. computes the spectra of each variant (grond.legacy.engine.spectra_stage, cached on disk) and loads them one
   variant at a time; with ``"response_tables": true`` it keeps only each variant's thickness table
   (grond.legacy.engine.response_stage) and scores and evaluates every candidate on it;
3. picks (variant, thickness family and parameters, P-spline n_basis/lambda) by inner 3-fold orbit CV R_all
   (grond.regularize.nested_selection.select_nested, model hand);
4. writes the pick, the nuisance fields to pin downstream, the full CV R_all grid (variant x thickness x scale),
   the best candidate per variant and per family, the grid-edge and physical-bound flags, the evaluator and the
   cost of every step.

Mode ``reselect`` is the per-macro-cycle thickness update: one fixed variant (the physics of the entering model's
feedback cycle), the thickness and scale grids re-searched on the spectra of the new structure.

  python -m grond.legacy.engine.nested_nuisance_stage --request request.json --out result.json [--workers 8]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from grond.bloch.config import Config  # noqa: E402
from grond.regularize.nested_selection import (  # noqa: E402
    FAMILY_PARAMS,
    SelectionGrids,
    ThicknessSupport,
    Variant,
    engine_config_fields,
    select_nested,
)
from grond.regularize.trial_rows import load_trial_rows_npz  # noqa: E402
from grond.validation.heldout import INNER_CV_SALT, free_mask, load_orbit_exclusion  # noqa: E402

REQUEST_SCHEMA = "grond.nested-nuisance-request.v1"
RESULT_SCHEMA = "grond.nested-nuisance-result.v1"
MODES = ("select", "reselect")
MIN_SELECTION_ROWS = 50
# Nuisance fields a nested pick pins: the thickness distribution with its evaluator and support, and the
# physics of its variant (mosaic cone, dose B). Uniform absorption is none (mu = 0): it is degenerate with the
# thickness distribution.
THICKNESS_NUISANCE_FIELDS = (
    "thickness_model",
    "thickness_A",
    "thickness_grad_frac",
    "thickness_sigma_frac",
    "thickness_gamma_shape",
    "thickness_evaluator",
    "thickness_min_A",
    "thickness_max_A",
    "thickness_table_dt_A",
)
VARIANT_NUISANCE_FIELDS = ("mosaicity_mode", "mosaicity_sigma_deg", "mosaicity_points", "mosaicity_support_sigma_deg",
                           "mosaicity_support_points", "mosaicity_support_width_sigma", "dose_b_mode", "dose_b_per_frame")
ABSORPTION_NUISANCE_FIELDS = (
    "absorption_mode",
    "absorption_mu",
    "absorption_frac",
    "absorption_mu_g2",
    "absorption_mu_theta2",
    "absorption_tds_b",
)
# Config fields a spectrum never depends on or that only steer caches / the paired hand; the base spectral
# Config of a request resets them.
BASE_CONFIG_RESETS = {"row_plan_cache": "off", "paired_inverse_hand": False}


# ----------------------------------------------------------------------------- inputs


def file_sha256(path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path) -> dict:
    """Read a JSON file."""

    return json.loads(Path(path).read_text())


def write_json_atomic(path, payload: dict) -> None:
    """Write payload as indented JSON through a temporary file and an atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=1, allow_nan=True, default=_json_default) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _json_default(value):
    """Convert NumPy scalars and arrays, sets and tuples for JSON; raise TypeError for anything else."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def validate_request(request: dict) -> None:
    """Raise unless the request has the expected schema, a known mode, the required keys and variants.

    A reselect request must name exactly one variant.
    """

    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"nested nuisance request schema {request.get('schema')!r}; expected {REQUEST_SCHEMA}")
    if request.get("mode") not in MODES:
        raise ValueError(f"nested nuisance request mode {request.get('mode')!r}; expected one of {MODES}")
    missing = [key for key in ("hkl", "res", "rows_npz", "base_config", "variants", "grids", "support") if key not in request]
    if missing:
        raise ValueError(f"nested nuisance request lacks {missing}")
    variants = request["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("nested nuisance request needs at least one physics variant")
    if request["mode"] == "reselect" and len(variants) != 1:
        raise ValueError("a thickness re-selection keeps the physics variant fixed: exactly one variant")
    if request.get("response_tables") and (request.get("evaluator", "auto") != "table"
                                           or request.get("score_with", "table") != "table"):
        raise ValueError("a response-table request needs evaluator 'table' and score_with 'table' (no spectra are kept)")


def selection_rows(request: dict) -> dict:
    """Work rows of the anchor trial rows with a valid anchor prediction, and their selection arrays.

    Free rows (the engine split, plus the orbit-hash free set recomputed from the request) and excluded orbits are
    removed here, before any spectrum is computed: the selection never reads their targets.
    """

    rows = load_trial_rows_npz(Path(request["rows_npz"]))
    trial = int(request.get("rows_trial_index", 0))
    index = np.asarray(rows["row_index"], dtype=int)
    keys = np.asarray(rows["orbit_key"]).astype(str)
    valid = np.asarray(rows["valid"], dtype=bool)
    valid = valid[trial] if valid.ndim == 2 else valid
    held_out = np.asarray(rows["free"], dtype=bool) | ~np.asarray(rows["work"], dtype=bool)
    extra = request.get("extra_free") or {}
    if extra:
        held_out |= free_mask(list(keys), str(extra["salt"]), float(extra["fraction"]))
    excluded = np.zeros(len(keys), dtype=bool)
    exclusion = str(request.get("exclude_orbits_json") or "")
    if exclusion:
        excluded = np.isin(keys, sorted(load_orbit_exclusion(exclusion)["orbits"]))
    selected = ~held_out & ~excluded & valid
    target = np.asarray(rows["target"], dtype=float)
    sigma = np.asarray(rows["sigma"], dtype=float)
    angle = np.asarray(rows["angle_deg"], dtype=float)
    finite = np.isfinite(target) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(angle)
    selected &= finite
    if int(selected.sum()) < MIN_SELECTION_ROWS:
        raise RuntimeError(f"nested nuisance selection has only {int(selected.sum())} usable work rows")
    return {
        "row_index": index[selected],
        "y": target[selected],
        "sig": sigma[selected],
        "ang": angle[selected],
        "keys": keys[selected].astype(object),
        "summary": {
            "anchor_rows": int(len(index)),
            "work_rows_used": int(selected.sum()),
            "free_rows_skipped": int(np.count_nonzero(held_out & ~excluded)),
            "excluded_rows_skipped": int(np.count_nonzero(excluded)),
            "invalid_anchor_rows_skipped": int(np.count_nonzero(~held_out & ~excluded & ~valid)),
            "nonfinite_rows_skipped": int(np.count_nonzero(~held_out & ~excluded & valid & ~finite)),
            "orbits": int(len(np.unique(keys[selected]))),
            "free_rows_used": False,
        },
    }


def grids_from_request(payload: dict) -> SelectionGrids:
    """Build the SelectionGrids from a request's grids block."""

    return SelectionGrids(
        families=tuple(payload["families"]),
        gamma_mean=tuple(float(v) for v in payload["gamma_mean"]),
        gamma_shape=tuple(float(v) for v in payload["gamma_shape"]),
        wedge_t0=tuple(float(v) for v in payload["wedge_t0"]),
        wedge_grad=tuple(float(v) for v in payload["wedge_grad"]),
        mu=tuple(float(v) for v in payload.get("mu", [0.0])),
        n_basis=tuple(int(v) for v in payload["n_basis"]),
        lam=tuple(float(v) for v in payload["lambda"]),
    )


def variant_config(request: dict, variant: dict) -> Config:
    """Return one variant's spectral Config: the base config with cache resets and the variant's overrides."""

    fields = {**request["base_config"], **BASE_CONFIG_RESETS, **(variant.get("config") or {})}
    return Config(**fields)


# ----------------------------------------------------------------------------- pick -> nuisance


def pinned_thickness_fields(pick: dict) -> dict:
    """Nuisance fields of a pick's thickness distribution, evaluator and support (native_stage_cmd pins them)."""

    fields = engine_config_fields(pick)
    gamma = fields["thickness_model"] == "gamma"
    return {
        "thickness_model": fields["thickness_model"],
        "thickness_A": float(fields["thickness_A"]),
        "thickness_grad_frac": 0.0 if gamma else float(fields["thickness_grad_frac"]),
        "thickness_sigma_frac": 0.0,
        "thickness_gamma_shape": float(fields["thickness_gamma_shape"]) if gamma else 1.0,
        "thickness_evaluator": str(fields["thickness_evaluator"]),
        "thickness_min_A": float(fields["thickness_min_A"]),
        "thickness_max_A": float(fields["thickness_max_A"]),
        "thickness_table_dt_A": float(fields["thickness_table_dt_A"]),
    }


def pinned_nuisance_fields(pick: dict) -> dict:
    """Every nuisance field a nested pick pins: thickness, the variant's cone and dose, and no absorption."""

    fields = engine_config_fields(pick)
    variant = pick.get("variant_config") or {}
    out = pinned_thickness_fields(pick)
    for key in VARIANT_NUISANCE_FIELDS:
        if key in variant:
            out[key] = variant[key]
    for key in ABSORPTION_NUISANCE_FIELDS:
        out[key] = fields[key]
    return out


def thickness_summary(pick: dict) -> dict:
    """Compact record of a pick's thickness and scale (for trajectories and reports)."""

    thickness = dict(pick["thickness"])
    family = str(thickness.get("family"))
    return {
        "family": family,
        **{name: float(thickness[name]) for name in FAMILY_PARAMS["gamma" if family == "gamma" else "wedge"]},
        "scale_n_basis": int(pick["scale"]["n_basis"]),
        "scale_lambda": float(pick["scale"]["lambda"]),
        "evaluator": str(pick.get("evaluator")),
        "mass_outside_support": float(pick.get("mass_outside_support", float("nan"))),
        "inner_r_all": float(pick.get("inner_r_all", float("nan"))),
        "variant": str(pick.get("variant")),
        "edge": list(pick.get("edge") or []),
    }


# ----------------------------------------------------------------------------- selection


def _cache_loader(path: str, identity: dict):
    """Loader of one variant's cached spectra; the stored identity must still equal the computed one."""

    from grond.legacy.engine.spectra_stage import _read_cache

    def load() -> dict:
        """Read this variant's cached spectra, checking the stored identity."""

        return _read_cache(Path(path), identity)

    return load


def _table_loader(path: str, identity: dict):
    """Loader of one variant's cached response table; the stored identity must still equal the computed one."""

    from grond.legacy.engine.response_stage import load_response_table

    def load() -> dict:
        """Read this variant's response table, checking the stored identity."""

        return load_response_table(Path(path), identity)

    return load


def run_selection(request: dict, *, workers: int | None = None, scratch_dir=None) -> dict:
    """Run one request and return its result record (see the module docstring).

    Spectra go to ``request["spectra_cache_dir"]`` when it is set (kept for reuse); otherwise to a temporary
    directory under ``scratch_dir`` that is removed when the selection is done.
    """

    validate_request(request)
    cache_dir = str(request.get("spectra_cache_dir") or "")
    if cache_dir:
        return _run_selection(request, workers=workers, cache_dir=cache_dir)
    with tempfile.TemporaryDirectory(prefix="nested_spectra_", dir=scratch_dir) as tmp:
        return _run_selection(request, workers=workers, cache_dir=tmp)


def _run_selection(request: dict, *, workers: int | None, cache_dir: str) -> dict:
    """Compute every variant's spectra (or response tables) into cache_dir, run the nested selection, return the record."""

    from grond.legacy.engine.response_stage import compute_response_table
    from grond.legacy.engine.spectra_stage import compute_spectra

    t_start = time.perf_counter()
    workers = max(1, int(workers if workers is not None else request.get("workers", 1)))
    rows = selection_rows(request)
    grids = grids_from_request(request["grids"])
    support = ThicknessSupport(**request["support"])
    tables = bool(request.get("response_tables", False))
    transform = str(request.get("hkl_transform") or "identity")
    variants: list[Variant] = []
    spectra_cost = []
    untilted: dict[str, str] = {}  # dose fields -> spectra file of the first nonzero cone width at that dose
    for item in request["variants"]:
        cfg = variant_config(request, item)
        t0 = time.perf_counter()
        if tables:
            # Tables keep no spectra, so the untilted block of the cone widths is solved again for each width.
            table, info = compute_response_table(request["hkl"], request["res"], cfg, rows["row_index"], support,
                                                 hands=(1,), hkl_transform=transform, workers=workers,
                                                 cache_dir=cache_dir)
            del table
            loaders = {"spectra": None, "table": _table_loader(info["cache"], info["identity"])}
        else:
            dose = json.dumps({k: v for k, v in (item.get("config") or {}).items() if not k.startswith("mosaicity")},
                              sort_keys=True)
            # Every nonzero cone width solves the same untilted block at a given dose; later widths reuse it exactly.
            tilted = float(cfg.mosaicity_sigma_deg) > 0.0
            reuse = [untilted[dose]] if tilted and dose in untilted else []
            spec, info = compute_spectra(request["hkl"], request["res"], cfg, rows["row_index"], hkl_transform=transform,
                                         workers=workers, cache_dir=cache_dir, reuse=reuse)
            if tilted and info["cache"]:
                untilted.setdefault(dose, info["cache"])
            del spec  # one variant's spectra in memory at a time; the loader reads the cache file again
            loaders = {"spectra": _cache_loader(info["cache"], info["identity"]), "table": None}
        spectra_cost.append({
            "label": str(item["label"]), "seconds": round(time.perf_counter() - t0, 3), "cache_status": info["cache_status"],
            "cache_key": info["key"], "profile_points_total": info["profile_points_total"],
            "reused_points": info.get("reused_points", 0), "kind": "response_table" if tables else "spectra",
            "basis_median": info["basis_median"], "basis_max": info["basis_max"], "mu_max": info["mu_max"],
        })
        print(f"{'response table' if tables else 'spectra'} {item['label']}: {info['n_rows']} work rows, "
              f"{info['profile_points_total']} profile points ({spectra_cost[-1]['reused_points']} reused), basis max "
              f"{info['basis_max']}, {spectra_cost[-1]['seconds']:.1f} s ({info['cache_status']})", flush=True)
        variants.append(Variant(str(item["label"]), loaders["spectra"], params=dict(item.get("params") or {}),
                                config=dict(item.get("config") or {}), table=loaders["table"]))
    t_sel = time.perf_counter()
    selection = select_nested(
        variants, rows["y"], rows["sig"], rows["ang"], rows["keys"], np.ones(len(rows["y"]), dtype=bool),
        grids=grids, support=support, evaluator=str(request.get("evaluator", "auto")), hands=(1,), workers=workers,
        clamp=bool(request.get("clamp", True)), objective=str(request.get("objective", "row_mr")),
        salt=str(request.get("salt", INNER_CV_SALT)), score_with=str(request.get("score_with", "table")),
    )[1]
    selection_seconds = time.perf_counter() - t_sel
    pick = selection.pick
    thickness = grids.thickness_candidates()
    scales = grids.scale_candidates(str(request.get("objective", "row_mr")))
    n_v, n_q, n_s = len(variants), len(thickness) * len(grids.mu), len(scales)
    cube = np.asarray(selection.scores, dtype=float).reshape(n_v, n_q, n_s)
    thickness_axis = [{"family": "gamma" if fam == "gamma" else "wedge", **prm, "mu": float(mu)}
                      for fam, prm in thickness for mu in grids.mu]
    per_variant = []
    for vi, v in enumerate(variants):
        flat = cube[vi].reshape(-1)
        qi, si = divmod(int(np.nanargmin(flat)), n_s)
        per_variant.append({"variant": v.label, "params": v.params, "cv_r_all": float(flat[qi * n_s + si]),
                            "thickness": thickness_axis[qi], "scale": scales[si]})
    per_family = {}
    for fam in sorted({t["family"] for t in thickness_axis}):
        cols = [qi for qi, t in enumerate(thickness_axis) if t["family"] == fam]
        sub = cube[:, cols, :]
        vi, qj, si = np.unravel_index(int(np.nanargmin(sub)), sub.shape)
        per_family[fam] = {"variant": variants[vi].label, "cv_r_all": float(sub[vi, qj, si]),
                           "thickness": thickness_axis[cols[qj]], "scale": scales[si]}
    total = time.perf_counter() - t_start
    return {
        "schema": RESULT_SCHEMA,
        "mode": request["mode"],
        "dataset": request.get("dataset", ""),
        "hkl": str(request["hkl"]),
        "res": str(request["res"]),
        "res_sha256": file_sha256(request["res"]),
        "selection_rows": "work rows only (free and excluded rows removed before the spectra)",
        "free_rows_used": False,
        "rows": rows["summary"],
        "fidelity": {"n_basis": int(request["base_config"].get("n_basis", 0)),
                     "n_points": int(request["base_config"].get("n_points", 0))},
        "pick": pick,
        "thickness": thickness_summary(pick),
        "pinned_nuisance": pinned_nuisance_fields(pick),
        "evaluator": pick.get("evaluator"),
        "evaluator_requested": pick.get("evaluator_requested"),
        "edges": list(pick.get("edge") or []),
        "at_physical_bound": list(pick.get("at_physical_bound") or []),
        "grid": {
            "variants": [{"label": v.label, "params": v.params, "config": v.config} for v in variants],
            "thickness": thickness_axis,
            "scale": scales,
            "grids": grids.to_dict(),
            "support": support.to_dict(),
            "size": int(cube.size),
        },
        "cv_r_all": cube.tolist(),
        "cv_r_all_axes": ["variant", "thickness", "scale"],
        "best_per_variant": per_variant,
        "best_per_family": per_family,
        "cost": {"spectra": spectra_cost, "spectra_seconds": round(sum(c["seconds"] for c in spectra_cost), 3),
                 "selection_seconds": round(selection_seconds, 3), "total_seconds": round(total, 3), "workers": workers},
    }


# ----------------------------------------------------------------------------- per-macro-cycle update


def spectral_base_config(run_dir, *, n_basis: int, n_points: int) -> dict:
    """The exact Config of a native run's selected candidate (its prediction-cache record) at another fidelity.

    Every physics field (window, cone, dose, row resolution, jitter, support) is the one the run used; only the
    basis cap and the profile point count are replaced. A run without a readable record fails closed.
    """

    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"native run {run_dir} has no summary.json to take its spectral Config from")
    summary = read_json(summary_path)
    cache = ((summary.get("best_native") or {}).get("prediction_cache") or {})
    path = str(cache.get("path") or "")
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"native run {run_dir} records no readable prediction cache ({path or 'none'})")
    with np.load(path, allow_pickle=False) as z:
        if "meta_json" not in z.files:
            raise ValueError(f"prediction cache {path} has no metadata")
        meta = json.loads(str(z["meta_json"]))
    cfg = dict(meta.get("cfg") or {})
    if not cfg:
        raise ValueError(f"prediction cache {path} records no Config")
    cfg.update(n_basis=int(n_basis), n_points=int(n_points), **BASE_CONFIG_RESETS)
    Config(**cfg)  # validate
    return cfg


def reselect_request(contract: dict, *, res, base_config: dict, label: str) -> dict:
    """A thickness re-selection request: the contract's rows, grids and support, one fixed variant, a new model.

    The variant keeps the cone and dose of ``base_config`` (the physics of the cycle that pinned the entering
    model's rows); it must equal the nested pick's variant, otherwise the update refuses to run.
    """

    request = copy.deepcopy(contract["request"])
    pick = contract["result"]["pick"]
    variant_cfg = dict(pick.get("variant_config") or {})
    for key, value in variant_cfg.items():
        base_value = base_config.get(key)
        same = (str(base_value) == str(value)) if isinstance(value, str) else (
            base_value is not None and math.isclose(float(base_value), float(value), rel_tol=1e-9, abs_tol=1e-15))
        if not same:
            raise ValueError(
                f"thickness re-selection keeps the nested variant fixed, but the cycle physics has {key}={base_value!r} "
                f"and the nested pick {key}={value!r}"
            )
    request.update(mode="reselect", res=str(res), base_config=dict(base_config), label=str(label),
                   variants=[{"label": str(pick["variant"]), "params": dict(pick.get("variant_params") or {}),
                              "config": variant_cfg}])
    return request


def apply_thickness(nuisance: dict, fields: dict) -> dict:
    """A copy of a nuisance record with a pick's thickness fields."""

    out = dict(nuisance)
    out.update({key: fields[key] for key in THICKNESS_NUISANCE_FIELDS if key in fields})
    return out


def thickness_drift(fields: dict, nuisance: dict) -> list[str]:
    """Thickness fields a native run did not reproduce ("key pinned -> recorded"); [] when it used ``fields``."""

    drift = []
    for key in THICKNESS_NUISANCE_FIELDS:
        if key not in fields:
            continue
        want, got = fields[key], nuisance.get(key)
        if isinstance(want, str):
            same = str(got) == want
        else:
            try:
                same = math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-12)
            except (TypeError, ValueError):
                same = False
        if not same:
            drift.append(f"{key} {want!r} -> {got!r}")
    return drift


class FeedbackThickness:
    """Thickness per structure for one crystal's macro-cycles (keyed by the SHA-256 of the model file).

    The nested pick holds for the model it was selected on. Any other model gets a re-selection on its own
    spectra (``update``), with the variant fixed and the physics of the most recent native run of the loop.
    """

    def __init__(self, contract_path, out_dir, run_stage):
        """Load a selected nested contract and record its pick for the model it was chosen on.

        Raises unless the contract status is "selected". run_stage(request, result, label) runs a re-selection.
        """

        self.contract_path = Path(contract_path)
        self.contract = read_json(self.contract_path)
        if self.contract.get("status") != "selected":
            raise ValueError(f"nested nuisance contract {contract_path} has status {self.contract.get('status')!r}")
        self.out_dir = Path(out_dir)
        self.run_stage = run_stage
        result = self.contract["result"]
        self.fidelity = dict(result["fidelity"])
        self.by_model: dict[str, dict] = {
            result["res_sha256"]: {"fields": pinned_thickness_fields(result["pick"]), "record": {
                **thickness_summary(result["pick"]), "source": "nested_nuisance_stage", "model": result["res"],
                "result": str(self.contract_path)}}
        }
        self.updates: list[dict] = []

    def known(self, model) -> bool:
        """Return whether a thickness is recorded for this model file."""

        return file_sha256(model) in self.by_model

    def fields(self, model) -> dict:
        """Return the pinned thickness fields recorded for this model file."""

        return self.by_model[file_sha256(model)]["fields"]

    def record(self, model) -> dict:
        """Return a copy of the thickness record for this model file."""

        return dict(self.by_model[file_sha256(model)]["record"])

    def update(self, model, base_run_dir, label: str) -> dict:
        """Re-select the thickness for ``model`` unless it is known; returns its record."""

        key = file_sha256(model)
        if key in self.by_model:
            return self.record(model)
        base = spectral_base_config(base_run_dir, n_basis=self.fidelity["n_basis"], n_points=self.fidelity["n_points"])
        request = reselect_request(self.contract, res=model, base_config=base, label=label)
        stage_dir = self.out_dir / f"thickness_update_{label}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        request_path = stage_dir / "request.json"
        result_path = stage_dir / "result.json"
        write_json_atomic(request_path, request)
        self.run_stage(request_path, result_path, label)
        result = read_json(result_path)
        if result.get("res_sha256") != key:
            raise RuntimeError(f"thickness re-selection {result_path} was computed for another model")
        record = {**thickness_summary(result["pick"]), "source": f"reselected_{label}", "model": str(model),
                  "result": str(result_path), "physics_from_run": str(base_run_dir), "cost": result.get("cost")}
        self.by_model[key] = {"fields": pinned_thickness_fields(result["pick"]), "record": record}
        self.updates.append(record)
        return dict(record)


# ----------------------------------------------------------------------------- CLI


def main(argv=None) -> None:
    """Run one nested nuisance request, write the result JSON to --out and print the pick."""

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", required=True, help="Request JSON (schema grond.nested-nuisance-request.v1)")
    ap.add_argument("--out", required=True, help="Result JSON")
    ap.add_argument("--workers", type=int, default=0, help="Worker processes (0: the request's value)")
    args = ap.parse_args(argv)
    request = read_json(args.request)
    result = run_selection(request, workers=args.workers or None, scratch_dir=Path(args.out).resolve().parent)
    write_json_atomic(args.out, result)
    t = result["thickness"]
    params = ", ".join(f"{k}={t[k]:g}" for k in ("mean", "shape", "t0", "grad") if k in t)
    print(
        f"nested nuisance ({result['mode']}): variant {t['variant']}, {t['family']} {params}, lambda {t['scale_lambda']:g}, "
        f"evaluator {t['evaluator']}, inner CV R_all {t['inner_r_all']:.4f}; {result['rows']['work_rows_used']} work rows, "
        f"grid {result['grid']['size']}, edges {result['edges'] or 'none'}; {result['cost']['total_seconds']:.1f} s -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
